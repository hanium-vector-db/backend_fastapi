"""
LLM Tool Calling Wrapper
LLM의 응답을 파싱하고 필요한 tool을 실행하는 wrapper
"""

import logging
import json
import re
from typing import Dict, Any, List, Optional
from tools.tool_definitions import TOOL_DEFINITIONS, get_tool_by_name
from tools.tool_executor import ToolExecutor

logger = logging.getLogger(__name__)


class ToolCallingWrapper:
    """LLM의 Tool Calling을 지원하는 Wrapper"""

    def __init__(self, llm_handler, tool_executor: ToolExecutor):
        self.llm = llm_handler
        self.tool_executor = tool_executor
        self.max_iterations = 5  # 무한 루프 방지

    def _create_system_prompt(self) -> str:
        """시스템 프롬프트 생성 - LLM에게 tool 사용법을 알려줌"""
        from datetime import datetime
        import locale

        # 현재 날짜 및 시간 정보
        now = datetime.now()
        current_date = now.strftime('%Y년 %m월 %d일')
        current_time = now.strftime('%H시 %M분')
        weekdays = ['월요일', '화요일', '수요일', '목요일', '금요일', '토요일', '일요일']
        current_weekday = weekdays[now.weekday()]

        tools_description = "\n\n".join([
            f"**{tool['name']}**: {tool['description']}\n"
            f"Parameters: {json.dumps(tool['parameters'], ensure_ascii=False, indent=2)}"
            for tool in TOOL_DEFINITIONS
        ])

        return f"""You are a Korean language AI assistant. You MUST respond ONLY in Korean (한국어).

**CRITICAL LANGUAGE REQUIREMENT:**
- You MUST write ALL responses in Korean language (한국어)
- DO NOT use Chinese (中文), English, or any other language
- Every single word in your response must be in Korean (한국어)
- This is an absolute requirement - NO EXCEPTIONS
- 모든 답변은 반드시 한국어로만 작성하세요
- 중국어나 영어를 절대 사용하지 마세요

**현재 날짜 및 시간 정보:**
- 오늘 날짜: {current_date} ({current_weekday})
- 현재 시각: {current_time}

당신은 한국 사용자를 돕는 한국어 AI 어시스턴트입니다.

다음과 같은 도구들을 사용할 수 있습니다:

{tools_description}

**도구 사용 규칙:**
1. 사용자의 질문에 답변하기 위해 필요한 정보가 있다면, 적절한 도구를 사용하세요.
2. 도구를 호출하려면 다음 형식을 사용하세요:
   TOOL_CALL: {{"name": "도구이름", "parameters": {{...}}}}

3. 도구 호출 응답을 받은 후, 그 정보를 바탕으로 사용자에게 자연스러운 답변을 제공하세요.
4. 여러 도구를 순차적으로 호출할 수 있습니다.
5. 도구 호출이 필요 없으면 바로 답변하세요.

**관련 페이지 안내 규칙:**
답변 후 관련된 페이지가 있다면 반드시 다음 형식으로 페이지 링크를 포함하세요:
[PAGE:/page-path:설명 텍스트]

사용 가능한 페이지 링크:
- [PAGE:/health:건강 상태 자세히 보기] - 건강 관련 질문일 때
- [PAGE:/news:뉴스 더보기] - 뉴스 관련 질문일 때
- [PAGE:/finance:금융 정보 보기] - 금융/재정 관련 질문일 때
- [PAGE:/diet-plan:식단 계획 보기] - 식단 관련 질문일 때
- [PAGE:/calendar:일정 확인하기] - 일정 관련 질문일 때
- [PAGE:/price:가격 추적 보기] - 가격 관련 질문일 때
- [PAGE:/weather-detail:날씨 정보 자세히 보기] - 날씨 관련 질문일 때

**예시:**
사용자: "오늘 뉴스 알려줘"
어시스턴트: TOOL_CALL: {{"name": "get_news", "parameters": {{"limit": 5}}}}
→ 최종 답변: "오늘의 주요 뉴스입니다: [뉴스 목록] [PAGE:/news:이곳에서 더 많은 뉴스를 확인할 수 있어요]"

사용자: "내 건강 상태 어때?"
어시스턴트: TOOL_CALL: {{"name": "get_health_status", "parameters": {{}}}}
→ 최종 답변: "현재 건강 상태는 양호합니다. [상세 정보] [PAGE:/health:자세한 건강 리포트는 이곳을 확인하세요]"

사용자: "고혈압 추가해줘"
어시스턴트: TOOL_CALL: {{"name": "add_disease", "parameters": {{"name": "고혈압", "status": "active"}}}}
→ 최종 답변: "고혈압이 추가되었습니다. [PAGE:/health:건강 정보에서 관리할 수 있어요]"
"""

    def _parse_tool_call(self, response: str) -> Optional[Dict[str, Any]]:
        """LLM 응답에서 tool call 파싱"""
        try:
            # TOOL_CALL: {...} 형식 찾기 (중첩된 JSON 지원)
            pattern = r'TOOL_CALL:\s*(\{.+\})'
            match = re.search(pattern, response, re.DOTALL)

            if match:
                tool_call_str = match.group(1)
                # 중첩된 JSON을 올바르게 추출하기 위해 brace counting
                brace_count = 0
                end_index = 0
                for i, char in enumerate(tool_call_str):
                    if char == '{':
                        brace_count += 1
                    elif char == '}':
                        brace_count -= 1
                        if brace_count == 0:
                            end_index = i + 1
                            break

                if end_index > 0:
                    tool_call_str = tool_call_str[:end_index]

                tool_call = json.loads(tool_call_str)
                logger.info(f"Tool call 파싱 성공: {tool_call}")
                return tool_call
            return None
        except json.JSONDecodeError as e:
            logger.error(f"Tool call JSON 파싱 오류: {e}")
            return None
        except Exception as e:
            logger.error(f"Tool call 파싱 오류: {e}")
            return None

    async def generate_with_tools(self, user_message: str, user_id: str, max_length: int = 512) -> str:
        """Tool calling을 지원하는 생성"""
        conversation_history = []
        system_prompt = self._create_system_prompt()

        # 초기 프롬프트
        full_prompt = f"{system_prompt}\n\n사용자: {user_message}\n어시스턴트: "
        iteration = 0

        while iteration < self.max_iterations:
            iteration += 1
            logger.info(f"Tool calling iteration {iteration}/{self.max_iterations}")

            # LLM 호출
            llm_response = self.llm.generate(full_prompt, max_length=max_length, stream=False)
            logger.info(f"LLM 응답: {llm_response[:200]}...")

            # Tool call 파싱
            tool_call = self._parse_tool_call(llm_response)

            if tool_call:
                # Tool 실행
                tool_name = tool_call.get("name")
                tool_params = tool_call.get("parameters", {})

                logger.info(f"도구 실행 중: {tool_name} with params {tool_params}")
                tool_result = await self.tool_executor.execute_tool(tool_name, tool_params, user_id)

                logger.info(f"도구 실행 결과: {tool_result}")

                # Tool 결과를 대화에 추가
                tool_result_text = json.dumps(tool_result, ensure_ascii=False)
                full_prompt += f"{llm_response}\n\nTOOL_RESULT: {tool_result_text}\n\n**IMPORTANT: You MUST respond in Korean (한국어) ONLY. 반드시 한국어로만 답변하세요.**\n사용자에게 위 정보를 바탕으로 한국어로 자연스럽게 답변하세요.\n어시스턴트: "

                # 다음 iteration에서 최종 응답 생성
                continue
            else:
                # Tool call이 없으면 최종 응답
                # TOOL_CALL: 부분 제거
                final_response = re.sub(r'TOOL_CALL:.*', '', llm_response, flags=re.DOTALL).strip()
                return final_response

        # 최대 반복 횟수 초과
        logger.warning("최대 tool calling iteration 초과")
        return "죄송합니다. 요청을 처리하는 데 시간이 너무 오래 걸렸습니다."

    async def generate_with_tools_stream(self, user_message: str, user_id: str, max_length: int = 512):
        """Tool calling을 지원하는 스트리밍 생성 (간소화 버전)"""
        # 스트리밍에서는 tool calling을 먼저 처리하고, 최종 응답만 스트리밍
        system_prompt = self._create_system_prompt()
        full_prompt = f"{system_prompt}\n\n사용자: {user_message}\n어시스턴트: "

        # 첫 번째 응답으로 tool call 확인
        initial_response = self.llm.generate(full_prompt, max_length=max_length, stream=False)
        tool_call = self._parse_tool_call(initial_response)

        if tool_call:
            # Tool 실행
            tool_name = tool_call.get("name")
            tool_params = tool_call.get("parameters", {})
            tool_result = await self.tool_executor.execute_tool(tool_name, tool_params, user_id)

            # Tool 결과를 포함한 최종 프롬프트
            tool_result_text = json.dumps(tool_result, ensure_ascii=False)
            final_prompt = f"{full_prompt}{initial_response}\n\nTOOL_RESULT: {tool_result_text}\n\n사용자에게 위 정보를 바탕으로 자연스럽게 답변하세요.\n어시스턴트: "

            # 최종 응답 스트리밍
            for chunk in self.llm.generate(final_prompt, max_length=max_length, stream=True):
                yield chunk
        else:
            # Tool call이 없으면 바로 스트리밍
            for chunk in self.llm.generate(full_prompt, max_length=max_length, stream=True):
                yield chunk
