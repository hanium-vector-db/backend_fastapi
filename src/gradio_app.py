import gradio as gr
import requests
import json
import logging
import time
import re

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI 서버 주소
API_URL = "http://127.0.0.1:8001/api/v1"

def process_streaming_response(response):
    """
    스트리밍 응답을 처리하고 실시간 업데이트를 제공합니다.
    """
    content_buffer = ""
    status_message = ""
    final_result = None
    
    try:
        for line in response.iter_lines(decode_unicode=True):
            if line and line.startswith("data: "):
                try:
                    data_str = line[6:]  # "data: " 부분 제거
                    data = json.loads(data_str)
                    
                    if data.get('status') == 'streaming' and 'chunk' in data:
                        # 실시간 텍스트 청크
                        content_buffer += data['chunk']
                        yield content_buffer, status_message, None
                    
                    elif data.get('status') in ['starting', 'searching', 'processing', 'generating', 'categories', 'category_analyzing', 'overall_analyzing']:
                        # 진행 상태 메시지
                        status_message = data.get('message', '')
                        yield content_buffer, status_message, None
                    
                    elif data.get('status') == 'category_completed':
                        # 카테고리 완료 상태
                        category = data.get('category', '')
                        summary = data.get('summary', '')
                        status_message = f"✓ {category} 분석 완료"
                        yield content_buffer, status_message, None
                    
                    elif data.get('status') == 'completed':
                        # 최종 완료 상태
                        final_result = data
                        if 'summary' in data:
                            content_buffer = data['summary']
                        elif 'overall_trend' in data:
                            content_buffer = data['overall_trend']
                        status_message = "✓ 완료"
                        yield content_buffer, status_message, final_result
                        return
                    
                    elif data.get('status') == 'error':
                        # 에러 상태
                        error_msg = data.get('message', '알 수 없는 오류가 발생했습니다.')
                        yield f"❌ 오류: {error_msg}", "오류 발생", None
                        return
                        
                except json.JSONDecodeError:
                    continue
                    
    except Exception as e:
        yield f"❌ 스트리밍 처리 오류: {str(e)}", "오류", None

def handle_api_error(response):
    """
    API 응답 에러를 처리하고 사용자에게 보여줄 메시지를 반환합니다.
    성공 시 None을, 실패 시 에러 메시지 문자열을 반환합니다.
    """
    if response.status_code >= 400:
        try:
            detail = response.json().get("detail", "알 수 없는 서버 오류")
            return f"**오류가 발생했습니다 (HTTP {response.status_code})**\n\n**오류 내용:**\n{detail}\n\n**서버 응답 원문:**\n```\n{response.text}\n```"
        except json.JSONDecodeError:
            return f"**서버에서 심각한 오류가 발생했습니다 (HTTP {response.status_code})**\n\n**서버 응답 원문:**\n```\n{response.text}\n```"
    return None

def make_api_call(endpoint, payload, method="post"):
    """중복되는 API 호출 로직을 처리하는 헬퍼 함수"""
    try:
        if method.lower() == "post":
            response = requests.post(f"{API_URL}/{endpoint}", json=payload, timeout=300) # 타임아웃 증가
        else:
            response = requests.get(f"{API_URL}/{endpoint}", timeout=30)
            
        error_message = handle_api_error(response)
        if error_message:
            return {"error": error_message}
        return response.json()
    except requests.exceptions.RequestException as e:
        return {"error": f"백엔드 서버 연결 실패: {e}"}

def generate_text(prompt, model_key, streaming_mode):
    """'/generate' 엔드포인트를 호출합니다."""
    if not prompt:
        return "오류: 프롬프트를 입력해주세요.", {"error": "Prompt is empty"}
    
    payload = {"prompt": prompt, "stream": streaming_mode}
    if model_key and model_key != "기본 모델":
        payload["model_key"] = model_key

    if streaming_mode:
        # 스트리밍 모드
        try:
            response = requests.post(
                f"{API_URL}/generate", 
                json=payload, 
                stream=True,
                headers={'Accept': 'text/event-stream'},
                timeout=300
            )
            
            if response.status_code != 200:
                return f"오류: HTTP {response.status_code}", {"error": "Stream failed"}
            
            full_text = ""
            token_count = 0
            
            for line in response.iter_lines(decode_unicode=True):
                if line and line.startswith('data: '):
                    try:
                        data_str = line[6:]  # 'data: ' 제거
                        data = json.loads(data_str)
                        if 'error' in data:
                            return data['error'], {"error": data['error']}
                        if 'content' in data and data['content']:
                            full_text += data['content']
                            token_count += 1
                        if data.get('done', False):
                            break
                    except json.JSONDecodeError:
                        continue
            
            return full_text, {"model_info": {"streaming": True, "complete": True, "tokens": token_count}}
            
        except Exception as e:
            return f"스트리밍 오류: {str(e)}", {"error": str(e)}
    else:
        # 일반 모드
        result = make_api_call("generate", payload)
        
        if "error" in result:
            return result["error"], result
        else:
            return result.get("response", "응답 없음"), result.get("model_info", {})

def stream_generate_text_generator(payload):
    """스트리밍 텍스트 생성 (generator)"""
    try:
        response = requests.post(
            f"{API_URL}/generate", 
            json=payload, 
            stream=True,
            headers={'Accept': 'text/event-stream'},
            timeout=300
        )
        
        if response.status_code != 200:
            yield f"오류: HTTP {response.status_code}", {"error": "Stream failed"}
            return
        
        full_text = ""
        
        for line in response.iter_lines(decode_unicode=True):
            if line and line.startswith('data: '):
                try:
                    data_str = line[6:]  # 'data: ' 제거
                    data = json.loads(data_str)
                    if 'error' in data:
                        yield data['error'], {"error": data['error']}
                        return
                    if 'content' in data and data['content']:
                        full_text += data['content']
                        # 실시간으로 누적된 텍스트 출력
                        yield full_text, {"model_info": {"streaming": True, "tokens_so_far": len(full_text.split())}}
                    if data.get('done', False):
                        # 최종 완성된 텍스트
                        yield full_text, {"model_info": {"streaming": True, "complete": True}}
                        return
                except json.JSONDecodeError:
                    continue
        
    except Exception as e:
        yield f"스트리밍 오류: {str(e)}", {"error": str(e)}

def stream_generate_text(payload):
    """기존 호환성을 위한 래퍼 함수"""
    full_text = ""
    model_info = {}
    
    for text, info in stream_generate_text_generator(payload):
        full_text = text
        model_info = info
    
    return full_text, model_info

def stream_generate_text_with_progress(payload):
    """진행상황을 표시하면서 스트리밍 텍스트 생성"""
    import time
    try:
        response = requests.post(
            f"{API_URL}/generate", 
            json=payload, 
            stream=True,
            headers={'Accept': 'text/event-stream'},
            timeout=300
        )
        
        if response.status_code != 200:
            return f"오류: HTTP {response.status_code}", {"error": "Stream failed"}
        
        full_text = ""
        last_update_time = time.time()
        
        for line in response.iter_lines(decode_unicode=True):
            if line and line.startswith('data: '):
                try:
                    data_str = line[6:]  # 'data: ' 제거
                    data = json.loads(data_str)
                    if 'error' in data:
                        return data['error'], {"error": data['error']}
                    if 'content' in data and data['content']:
                        full_text += data['content']
                        # 짧은 지연을 통해 실시간 효과 시뮬레이션
                        current_time = time.time()
                        if current_time - last_update_time > 0.05:  # 50ms마다 업데이트
                            time.sleep(0.01)
                            last_update_time = current_time
                    if data.get('done', False):
                        break
                except json.JSONDecodeError:
                    continue
        
        return full_text, {"model_info": {"streaming": True, "complete": True, "length": len(full_text)}}
        
    except Exception as e:
        return f"스트리밍 오류: {str(e)}", {"error": str(e)}

def chat_with_bot(message, history, model_key):
    """'/chat' 엔드포인트를 호출합니다."""
    payload = {"message": message}
    if model_key and model_key != "기본 모델":
        payload["model_key"] = model_key

    result = make_api_call("chat", payload)
    if "error" in result:
        return result["error"]
        
    return result.get("response", "응답 없음")

def rag_query(question, model_key):
    """'/rag' 엔드포인트를 호출합니다."""
    if not question:
        return "오류: 질문을 입력해주세요.", "", {"error": "Question is empty"}

    payload = {"question": question}
    if model_key and model_key != "기본 모델":
        payload["model_key"] = model_key

    result = make_api_call("rag", payload)
    if "error" in result:
        return result["error"], "", result

    docs = result.get("relevant_documents", [])
    doc_str = ""
    if docs:
        for doc in docs:
            doc_str += f"### 📄 [{doc.get('title', '출처 없음')}]({doc.get('source', '#')})\n> {doc.get('content', '')}\n\n---\n"
    else:
        doc_str = "관련 문서를 찾지 못했습니다."
        
    return result.get("response", "응답 없음"), doc_str, result.get("model_info", {})

def update_rag_news(query, max_results):
    """'/rag/update-news' 엔드포인트를 호출하여 DB를 최신 뉴스로 업데이트합니다."""
    if not query:
        return "오류: 뉴스 검색어를 입력해주세요."
    
    gr.Info(f"'{query}' 관련 최신 뉴스 {max_results}개를 검색하여 DB 업데이트를 시작합니다. 잠시만 기다려주세요...")
    
    payload = {"query": query, "max_results": int(max_results)}
    result = make_api_call("rag/update-news", payload)
    
    if "error" in result:
        gr.Error("업데이트 실패!")
        return result["error"]
    
    gr.Info("업데이트 성공!")
    return result.get("message", "알 수 없는 응답")

# === 뉴스 기능 함수들 ===
def get_latest_news(categories, max_results, time_range):
    """최신 뉴스 조회"""
    try:
        # 카테고리 문자열 처리
        if categories and categories.strip():
            category_param = ",".join([cat.strip() for cat in categories.split(",")])
        else:
            category_param = ""
        
        params = {
            "max_results": max_results,
            "time_range": time_range
        }
        if category_param:
            params["categories"] = category_param
        
        response = requests.get(f"{API_URL}/news/latest", params=params, timeout=30)
        
        error_message = handle_api_error(response)
        if error_message:
            return error_message, "오류 발생"
        
        result = response.json()
        news_list = result.get("news", [])
        
        if not news_list:
            return "검색 결과가 없습니다.", "결과 없음"
        
        # 뉴스 목록 포맷팅
        formatted_news = f"**📰 총 {len(news_list)}개의 최신 뉴스**\n\n"
        for i, news in enumerate(news_list, 1):
            formatted_news += f"### {i}. {news.get('title', '제목 없음')}\n"
            formatted_news += f"**카테고리:** {news.get('category', 'N/A')} | "
            formatted_news += f"**점수:** {news.get('score', 0):.2f}\n"
            formatted_news += f"**URL:** [{news.get('url', '#')[:50]}...]({news.get('url', '#')})\n"
            content = news.get('content', '')
            if content:
                formatted_news += f"**내용 미리보기:** {content[:200]}...\n"
            formatted_news += f"**발행일:** {news.get('published_date', 'N/A')}\n\n---\n\n"
        
        summary = f"카테고리: {result.get('categories', [])} | 시간범위: {result.get('time_range', 'N/A')}"
        return formatted_news, summary
        
    except Exception as e:
        error_msg = f"최신 뉴스 조회 중 오류 발생: {str(e)}"
        return error_msg, "오류"

def search_news(query, category, max_results, time_range):
    """뉴스 검색"""
    if not query.strip():
        return "검색어를 입력해주세요.", "오류"
    
    try:
        params = {
            "query": query,
            "max_results": max_results,
            "time_range": time_range
        }
        if category and category != "전체":
            params["category"] = category
        
        response = requests.get(f"{API_URL}/news/search", params=params, timeout=30)
        
        error_message = handle_api_error(response)
        if error_message:
            return error_message, "오류 발생"
        
        result = response.json()
        news_list = result.get("news", [])
        
        if not news_list:
            return f"'{query}' 검색 결과가 없습니다.", "결과 없음"
        
        # 뉴스 검색 결과 포맷팅
        formatted_news = f"**🔍 '{query}' 검색 결과 ({len(news_list)}개)**\n\n"
        for i, news in enumerate(news_list, 1):
            formatted_news += f"### {i}. {news.get('title', '제목 없음')}\n"
            formatted_news += f"**카테고리:** {news.get('category', 'N/A')} | "
            formatted_news += f"**점수:** {news.get('score', 0):.2f}\n"
            formatted_news += f"**URL:** [{news.get('url', '#')[:50]}...]({news.get('url', '#')})\n"
            content = news.get('content', '')
            if content:
                formatted_news += f"**내용 미리보기:** {content[:200]}...\n"
            formatted_news += "---\n\n"
        
        summary = f"검색어: {query} | 카테고리: {category or '전체'} | 시간범위: {time_range}"
        return formatted_news, summary
        
    except Exception as e:
        error_msg = f"뉴스 검색 중 오류 발생: {str(e)}"
        return error_msg, "오류"

def summarize_news(query, summary_type, max_results, model_key):
    """AI 뉴스 요약 (스트리밍)"""
    if not query.strip():
        yield "요약할 뉴스 주제를 입력해주세요.", "오류", {}
        return
    
    try:
        payload = {
            "query": query,
            "summary_type": summary_type,
            "max_results": max_results
        }
        if model_key and model_key != "기본 모델":
            payload["model_key"] = model_key
        
        response = requests.post(f"{API_URL}/news/summary", json=payload, timeout=300, stream=True)
        
        if response.status_code >= 400:
            error_message = f"API 오류 ({response.status_code}): {response.text}"
            yield error_message, "오류 발생", {}
            return
        
        # 스트리밍 응답 처리
        final_result = None
        for content, status, result in process_streaming_response(response):
            if result:  # 최종 결과
                final_result = result
                articles = result.get("articles", [])
                
                # 참고 기사 목록 생성
                articles_info = f"**📊 분석된 기사 ({len(articles)}개):**\n\n"
                for i, article in enumerate(articles[:5], 1):  # 상위 5개만 표시
                    articles_info += f"{i}. **{article.get('title', '제목 없음')}**\n"
                    if article.get('url'):
                        articles_info += f"   🔗 [기사 링크]({article['url']})\n"
                    articles_info += "\n"
                
                model_info = {
                    "query": result.get("query"),
                    "summary_type": result.get("summary_type"),
                    "total_articles": result.get("total_articles"),
                    "model_info": result.get("model_info", {})
                }
                
                yield content, articles_info, model_info
                return
            else:
                # 중간 진행 상태
                yield content, f"🔄 {status}", {}
        
        # 만약 결과가 없다면
        if not final_result:
            yield "요약 생성에 실패했습니다.", "오류", {}
        
    except Exception as e:
        error_msg = f"뉴스 요약 중 오류 발생: {str(e)}"
        yield error_msg, "오류", {}

def analyze_news_trends(categories, max_results, time_range, model_key):
    """뉴스 트렌드 분석 (스트리밍)"""
    try:
        payload = {
            "max_results": max_results,
            "time_range": time_range
        }
        
        # 카테고리 처리
        if categories and categories.strip():
            category_list = [cat.strip() for cat in categories.split(",")]
            payload["categories"] = category_list
        
        if model_key and model_key != "기본 모델":
            payload["model_key"] = model_key
        
        response = requests.post(f"{API_URL}/news/analysis", json=payload, timeout=300, stream=True)
        
        if response.status_code >= 400:
            error_message = f"API 오류 ({response.status_code}): {response.text}"
            yield error_message, "오류 발생", {}
            return
        
        # 스트리밍 응답 처리
        final_result = None
        category_info_buffer = ""
        
        for content, status, result in process_streaming_response(response):
            if result:  # 최종 결과
                final_result = result
                trend_analysis = result.get("overall_trend", "트렌드 분석 실패")
                category_trends = result.get("category_trends", {})
                
                # 카테고리별 트렌드 포맷팅
                category_info = "**📊 카테고리별 트렌드:**\n\n"
                for category, trend in category_trends.items():
                    category_map = {
                        "politics": "정치", "economy": "경제", 
                        "technology": "기술", "society": "사회"
                    }
                    category_name = category_map.get(category, category)
                    category_info += f"**{category_name}:** {trend}\n\n"
                
                analysis_info = {
                    "total_articles": result.get("total_articles_analyzed"),
                    "categories": result.get("categories"),
                    "time_range": result.get("time_range"),
                    "model_info": result.get("model_info", {})
                }
                
                yield trend_analysis, category_info, analysis_info
                return
            else:
                # 중간 진행 상태
                yield content, f"🔄 {status}", {}
        
        # 만약 결과가 없다면
        if not final_result:
            yield "트렌드 분석에 실패했습니다.", "오류", {}
        
    except Exception as e:
        error_msg = f"트렌드 분석 중 오류 발생: {str(e)}"
        yield error_msg, "오류", {}

def get_news_categories():
    """뉴스 카테고리 조회"""
    try:
        response = requests.get(f"{API_URL}/news/categories", timeout=10)
        
        if response.status_code == 200:
            result = response.json()
            categories = result.get("categories", {})
            return list(categories.keys())
        else:
            # 기본 카테고리 반환
            return ["politics", "economy", "technology", "sports", "health", "culture", "society", "international"]
    except:
        return ["politics", "economy", "technology", "sports", "health", "culture", "society", "international"]

def update_model_list():
    """UI가 로드될 때 서버에서 모델 목록을 동적으로 가져옵니다."""
    logger.info("UI: 모델 목록을 서버에서 가져오는 중...")
    
    # 정적 모델 목록 (서버와 동기화)
    default_models = ["qwen2.5-7b", "llama3.1-8b", "gemma-3-4b"]
    choices = ["기본 모델"] + default_models
    
    try:
        response = requests.get(f"{API_URL}/models", timeout=5)
        if response.status_code == 200:
            server_models = list(response.json().get("supported_models", {}).keys())
            if server_models:
                choices = ["기본 모델"] + server_models
                logger.info(f"UI: 서버에서 모델 목록 가져오기 성공: {server_models}")
            else:
                logger.warning("UI: 서버에서 빈 모델 목록을 받았습니다. 기본 목록 사용")
        else:
            logger.warning(f"UI: 서버 응답 실패 (HTTP {response.status_code}). 기본 목록 사용")
    except requests.exceptions.RequestException as e:
        logger.warning(f"UI: 서버 연결 실패, 기본 모델 목록 사용: {e}")
    
    logger.info(f"UI: 최종 모델 선택지: {choices}")
    return (
        gr.Dropdown(choices=choices, value="기본 모델"),
        gr.Dropdown(choices=choices, value="기본 모델"), 
        gr.Dropdown(choices=choices, value="기본 모델"),
        gr.Dropdown(choices=choices, value="기본 모델"),  # 뉴스 요약용
        gr.Dropdown(choices=choices, value="기본 모델")   # 뉴스 분석용
    )

# --- Gradio UI 구성 ---
with gr.Blocks(theme=gr.themes.Soft(), title="LLM 서버 UI") as gradio_ui:
    gr.Markdown("# 🤖 LLM FastAPI 서버 UI")
    gr.Markdown("Gradio 인터페이스를 통해 서버의 LLM 기능을 쉽게 사용해보세요.")
    
    with gr.Tabs():
        # 1. 텍스트 생성 탭
        with gr.TabItem("📝 텍스트 생성"):
            gr.Markdown("### 💡 실시간 스트리밍을 원하시면 [전용 스트리밍 페이지](/stream)를 이용해주세요!")
            with gr.Row():
                with gr.Column(scale=2):
                    gen_prompt = gr.Textbox(lines=5, label="프롬프트", placeholder="인공지능의 미래에 대해 짧은 글을 써줘.")
                    gen_model_select = gr.Dropdown(
                        label="모델 선택", 
                        choices=["기본 모델", "qwen2.5-7b", "llama3.1-8b", "gemma-3-4b"],
                        value="기본 모델"
                    )
                    gen_streaming = gr.Checkbox(label="스트리밍 모드 (완성 후 일괄 표시)", value=False)
                    gen_button = gr.Button("생성하기", variant="primary")
                with gr.Column(scale=3):
                    gen_output = gr.Textbox(lines=10, label="생성된 텍스트", interactive=False)
                    gen_model_info = gr.JSON(label="사용된 모델 정보 / 오류 상세")

        # 2. 채팅 탭
        with gr.TabItem("💬 채팅"):
            chat_model_select = gr.Dropdown(
                label="채팅 모델 선택", 
                choices=["기본 모델", "qwen2.5-7b", "llama3.1-8b", "gemma-3-4b"],
                value="기본 모델"
            )
            gr.ChatInterface(
                fn=chat_with_bot,
                additional_inputs=[chat_model_select],
                chatbot=gr.Chatbot(height=400, label="채팅창", type="messages"),
                textbox=gr.Textbox(placeholder="메시지를 입력하세요...", label="입력"),
                submit_btn="보내기",
            )

        # 3. RAG 질의응답 탭
        with gr.TabItem("📚 RAG 질의응답"):
            with gr.Accordion("최신 뉴스로 DB 업데이트", open=False):
                with gr.Row():
                    news_query = gr.Textbox(label="뉴스 검색어", placeholder="예: 삼성전자 AI")
                    news_max_results = gr.Slider(minimum=1, maximum=20, value=5, step=1, label="최대 검색 기사 수")
                update_button = gr.Button("DB 업데이트 실행", variant="primary")
                update_status = gr.Textbox(label="업데이트 결과", interactive=False)
            
            gr.Markdown("---")
            
            with gr.Row():
                with gr.Column(scale=2):
                    rag_question = gr.Textbox(lines=2, label="질문", placeholder="삼성전자의 최신 AI 기술에 대해 알려줘.")
                    rag_model_select = gr.Dropdown(
                        label="모델 선택", 
                        choices=["기본 모델", "qwen2.5-7b", "llama3.1-8b", "gemma-3-4b"],
                        value="기본 모델"
                    )
                    rag_button = gr.Button("질문하기", variant="primary")
                with gr.Column(scale=3):
                    rag_answer = gr.Textbox(lines=5, label="답변", interactive=False)
                    rag_model_info_output = gr.JSON(label="사용된 모델 정보 / 오류 상세")
            rag_docs = gr.Markdown(label="참고 문서")

        # 4. 뉴스 기능 탭 (NEW!)
        with gr.TabItem("📰 뉴스 기능 (NEW!)"):
            gr.Markdown("### 🆕 Tavily 기반 실시간 뉴스 검색 및 AI 요약 기능")
            gr.Markdown("**주의**: 뉴스 기능을 사용하려면 `.env` 파일에 `TAVILY_API_KEY`를 설정해주세요.")
            
            with gr.Tabs():
                # 4-1. 최신 뉴스 조회
                with gr.TabItem("🔥 최신 뉴스"):
                    with gr.Row():
                        with gr.Column(scale=1):
                            latest_categories = gr.Textbox(
                                label="카테고리 (쉼표로 구분)", 
                                placeholder="technology,economy,politics",
                                value="technology,economy"
                            )
                            latest_max_results = gr.Slider(
                                minimum=1, maximum=20, value=8, step=1, 
                                label="최대 뉴스 수"
                            )
                            latest_time_range = gr.Radio(
                                choices=["d", "w", "m"], value="d",
                                label="시간 범위 (d=1일, w=1주, m=1달)"
                            )
                            latest_button = gr.Button("최신 뉴스 조회", variant="primary")
                        
                        with gr.Column(scale=2):
                            latest_output = gr.Markdown(label="최신 뉴스 목록")
                            latest_summary = gr.Textbox(label="조회 정보", interactive=False)

                # 4-2. 뉴스 검색
                with gr.TabItem("🔍 뉴스 검색"):
                    with gr.Row():
                        with gr.Column(scale=1):
                            search_query = gr.Textbox(
                                label="검색어", 
                                placeholder="ChatGPT, 삼성전자, 인공지능"
                            )
                            search_category = gr.Dropdown(
                                choices=["전체", "politics", "economy", "technology", "sports", "health", "culture", "society", "international"],
                                value="전체",
                                label="카테고리"
                            )
                            search_max_results = gr.Slider(
                                minimum=1, maximum=15, value=5, step=1,
                                label="최대 검색 결과"
                            )
                            search_time_range = gr.Radio(
                                choices=["d", "w", "m"], value="d",
                                label="시간 범위"
                            )
                            search_button = gr.Button("뉴스 검색", variant="primary")
                        
                        with gr.Column(scale=2):
                            search_output = gr.Markdown(label="검색 결과")
                            search_summary = gr.Textbox(label="검색 정보", interactive=False)

                # 4-3. AI 뉴스 요약
                with gr.TabItem("🤖 AI 뉴스 요약"):
                    with gr.Row():
                        with gr.Column(scale=1):
                            summary_query = gr.Textbox(
                                label="요약할 뉴스 주제", 
                                placeholder="ChatGPT, 부동산 정책, 전기차"
                            )
                            summary_type = gr.Radio(
                                choices=["brief", "comprehensive", "analysis"],
                                value="comprehensive",
                                label="요약 타입 (간단/포괄적/심층분석)"
                            )
                            summary_max_results = gr.Slider(
                                minimum=1, maximum=10, value=5, step=1,
                                label="분석할 뉴스 수"
                            )
                            summary_model = gr.Dropdown(
                                label="요약 모델", 
                                choices=["기본 모델", "qwen2.5-7b", "llama3.1-8b", "gemma-3-4b"],
                                value="기본 모델"
                            )
                            summary_button = gr.Button("AI 뉴스 요약", variant="primary")
                        
                        with gr.Column(scale=2):
                            summary_output = gr.Markdown(label="AI 뉴스 요약")
                            summary_articles = gr.Markdown(label="참고 기사")
                            summary_info = gr.JSON(label="요약 정보")

                # 4-4. 뉴스 트렌드 분석
                with gr.TabItem("📊 트렌드 분석"):
                    with gr.Row():
                        with gr.Column(scale=1):
                            trend_categories = gr.Textbox(
                                label="분석할 카테고리 (쉼표로 구분)", 
                                placeholder="politics,economy,technology,society",
                                value="politics,economy,technology"
                            )
                            trend_max_results = gr.Slider(
                                minimum=10, maximum=30, value=20, step=5,
                                label="분석할 총 뉴스 수"
                            )
                            trend_time_range = gr.Radio(
                                choices=["d", "w"], value="d",
                                label="분석 기간"
                            )
                            trend_model = gr.Dropdown(
                                label="분석 모델", 
                                choices=["기본 모델", "qwen2.5-7b", "llama3.1-8b", "gemma-3-4b"],
                                value="기본 모델"
                            )
                            trend_button = gr.Button("트렌드 분석", variant="primary")
                        
                        with gr.Column(scale=2):
                            trend_output = gr.Markdown(label="전체 트렌드 분석")
                            trend_categories_output = gr.Markdown(label="카테고리별 트렌드")
                            trend_info = gr.JSON(label="분석 정보")

    # --- 이벤트 핸들러 ---
    gen_button.click(fn=generate_text, inputs=[gen_prompt, gen_model_select, gen_streaming], outputs=[gen_output, gen_model_info])
    rag_button.click(fn=rag_query, inputs=[rag_question, rag_model_select], outputs=[rag_answer, rag_docs, rag_model_info_output])
    update_button.click(fn=update_rag_news, inputs=[news_query, news_max_results], outputs=update_status)
    
    # 뉴스 기능 이벤트 핸들러들
    latest_button.click(
        fn=get_latest_news, 
        inputs=[latest_categories, latest_max_results, latest_time_range], 
        outputs=[latest_output, latest_summary]
    )
    search_button.click(
        fn=search_news, 
        inputs=[search_query, search_category, search_max_results, search_time_range], 
        outputs=[search_output, search_summary]
    )
    summary_button.click(
        fn=summarize_news, 
        inputs=[summary_query, summary_type, summary_max_results, summary_model], 
        outputs=[summary_output, summary_articles, summary_info]
    )
    trend_button.click(
        fn=analyze_news_trends, 
        inputs=[trend_categories, trend_max_results, trend_time_range, trend_model], 
        outputs=[trend_output, trend_categories_output, trend_info]
    )

if __name__ == "__main__":
    gradio_ui.launch()
