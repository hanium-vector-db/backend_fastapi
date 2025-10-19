"""
DB와 LLM을 연동하는 서비스
사용자 데이터를 조회하고 LLM이 이해할 수 있는 형태로 제공
"""

import aiomysql
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime, date
import json

logger = logging.getLogger(__name__)


class DBLLMService:
    """DB 데이터를 LLM이 활용할 수 있도록 제공하는 서비스"""

    def __init__(self, db_config: Dict[str, Any]):
        self.db_config = db_config
        self.pool = None

    async def initialize(self):
        """데이터베이스 연결 풀 초기화"""
        try:
            self.pool = await aiomysql.create_pool(
                host=self.db_config['host'],
                port=self.db_config['port'],
                user=self.db_config['user'],
                password=self.db_config['password'],
                db=self.db_config['database'],
                charset=self.db_config['charset'],
                autocommit=True,
                minsize=1,
                maxsize=10
            )
            logger.info("DB 연결 풀 초기화 완료")
        except Exception as e:
            logger.error(f"DB 연결 풀 초기화 실패: {e}")
            raise

    async def close(self):
        """연결 풀 종료"""
        if self.pool:
            self.pool.close()
            await self.pool.wait_closed()

    async def get_user_context(self, user_id: str) -> str:
        """사용자의 모든 관련 정보를 LLM이 이해할 수 있는 텍스트로 변환"""
        try:
            context_parts = []

            # 사용자 기본 정보
            user_info = await self._get_user_info(user_id)
            if user_info:
                context_parts.append(f"사용자 정보: 이름은 {user_info['name']}이고, 닉네임은 {user_info['nickname']}입니다.")

            # 건강 정보
            health_context = await self._get_health_context(user_id)
            if health_context:
                context_parts.append(health_context)

            # 재정 정보
            finance_context = await self._get_finance_context(user_id)
            if finance_context:
                context_parts.append(finance_context)

            # 일정 정보
            schedule_context = await self._get_schedule_context(user_id)
            if schedule_context:
                context_parts.append(schedule_context)

            # 뉴스 관심사
            news_context = await self._get_news_interests(user_id)
            if news_context:
                context_parts.append(news_context)

            # 최근 일일 리포트
            report_context = await self._get_latest_report(user_id)
            if report_context:
                context_parts.append(report_context)

            return "\n\n".join(context_parts)

        except Exception as e:
            logger.error(f"사용자 컨텍스트 생성 실패: {e}")
            return ""

    async def _get_user_info(self, user_id: str) -> Optional[Dict]:
        """사용자 기본 정보 조회"""
        async with self.pool.acquire() as conn:
            async with conn.cursor(aiomysql.DictCursor) as cursor:
                await cursor.execute(
                    "SELECT userid, name, nickname, language FROM users WHERE userid = %s",
                    (user_id,)
                )
                return await cursor.fetchone()

    async def _get_health_context(self, user_id: str) -> str:
        """건강 관련 컨텍스트 생성"""
        parts = []

        async with self.pool.acquire() as conn:
            async with conn.cursor(aiomysql.DictCursor) as cursor:
                # 질병 정보
                await cursor.execute(
                    "SELECT name, diagnosed_date, status FROM disease WHERE user_id = %s",
                    (user_id,)
                )
                diseases = await cursor.fetchall()
                if diseases:
                    disease_list = [f"{d['name']} ({d['status']})" for d in diseases]
                    parts.append(f"현재 진단받은 질병: {', '.join(disease_list)}")

                # 복용 중인 약
                await cursor.execute(
                    "SELECT name, dosage, intake_time FROM medication WHERE user_id = %s",
                    (user_id,)
                )
                medications = await cursor.fetchall()
                if medications:
                    med_list = [f"{m['name']} {m['dosage']} (복용시간: {m['intake_time']})" for m in medications]
                    parts.append(f"복용 중인 약물: {', '.join(med_list)}")

                # 영양 목표
                await cursor.execute(
                    "SELECT goal_type, target_value, current_value, unit FROM nutrition_goals WHERE user_id = %s",
                    (user_id,)
                )
                goals = await cursor.fetchall()
                if goals:
                    goal_list = [f"{g['goal_type']} (목표: {g['target_value']}{g['unit']}, 현재: {g['current_value']}{g['unit']})" for g in goals]
                    parts.append(f"영양 목표: {', '.join(goal_list)}")

        return "건강 정보:\n" + "\n".join(parts) if parts else ""

    async def _get_finance_context(self, user_id: str) -> str:
        """재정 관련 컨텍스트 생성"""
        parts = []

        async with self.pool.acquire() as conn:
            async with conn.cursor(aiomysql.DictCursor) as cursor:
                # 재정 항목
                await cursor.execute(
                    """SELECT category, content, created_at
                       FROM finance_items
                       WHERE user_id = (SELECT id FROM users WHERE userid = %s)
                       ORDER BY created_at DESC LIMIT 10""",
                    (user_id,)
                )
                items = await cursor.fetchall()
                if items:
                    item_list = [f"{i['category']}: {i['content']}" for i in items]
                    parts.append(f"재정 항목: {', '.join(item_list)}")

                # 가격 추적
                await cursor.execute(
                    "SELECT item_name, current_price, target_price FROM price_items WHERE user_id = %s",
                    (user_id,)
                )
                prices = await cursor.fetchall()
                if prices:
                    price_list = [f"{p['item_name']} (현재: {p['current_price']:,}원, 목표: {p['target_price']:,}원)" for p in prices]
                    parts.append(f"가격 추적 항목: {', '.join(price_list)}")

        return "재정 정보:\n" + "\n".join(parts) if parts else ""

    async def _get_schedule_context(self, user_id: str) -> str:
        """일정 관련 컨텍스트 생성"""
        parts = []

        async with self.pool.acquire() as conn:
            async with conn.cursor(aiomysql.DictCursor) as cursor:
                # 다가오는 일정
                await cursor.execute(
                    """SELECT event_type, title, event_date, event_time, is_completed
                       FROM calendar_events
                       WHERE user_id = %s AND event_date >= CURDATE()
                       ORDER BY event_date, event_time LIMIT 5""",
                    (user_id,)
                )
                events = await cursor.fetchall()
                if events:
                    event_list = []
                    for e in events:
                        time_str = f" {e['event_time']}" if e['event_time'] else ""
                        status = "완료" if e['is_completed'] else "예정"
                        event_list.append(f"{e['event_date']}{time_str}: {e['title']} ({status})")
                    parts.append(f"다가오는 일정:\n" + "\n".join(event_list))

                # 미읽은 알림
                await cursor.execute(
                    """SELECT notification_type, title, message, priority
                       FROM notifications
                       WHERE user_id = %s AND is_read = FALSE
                       ORDER BY created_at DESC LIMIT 5""",
                    (user_id,)
                )
                notifications = await cursor.fetchall()
                if notifications:
                    notif_list = [f"[{n['priority']}] {n['title']}: {n['message']}" for n in notifications]
                    parts.append(f"미읽은 알림:\n" + "\n".join(notif_list))

        return "일정 및 알림:\n" + "\n".join(parts) if parts else ""

    async def _get_news_interests(self, user_id: str) -> str:
        """뉴스 관심 키워드"""
        async with self.pool.acquire() as conn:
            async with conn.cursor(aiomysql.DictCursor) as cursor:
                await cursor.execute(
                    "SELECT keyword FROM news_keywords WHERE user_id = %s AND is_excluded = FALSE",
                    (user_id,)
                )
                keywords = await cursor.fetchall()
                if keywords:
                    keyword_list = [k['keyword'] for k in keywords]
                    return f"뉴스 관심 키워드: {', '.join(keyword_list)}"
        return ""

    async def _get_latest_report(self, user_id: str) -> str:
        """최근 일일 리포트"""
        async with self.pool.acquire() as conn:
            async with conn.cursor(aiomysql.DictCursor) as cursor:
                await cursor.execute(
                    """SELECT report_date, health_score, ai_insights
                       FROM daily_reports
                       WHERE user_id = %s
                       ORDER BY report_date DESC LIMIT 1""",
                    (user_id,)
                )
                report = await cursor.fetchone()
                if report:
                    return f"최근 일일 리포트 ({report['report_date']}):\n건강 점수: {report['health_score']}/100\nAI 인사이트: {report['ai_insights']}"
        return ""

    async def query_database(self, user_id: str, query_type: str, **kwargs) -> List[Dict]:
        """특정 유형의 데이터 조회"""
        async with self.pool.acquire() as conn:
            async with conn.cursor(aiomysql.DictCursor) as cursor:
                if query_type == "medications":
                    await cursor.execute(
                        "SELECT * FROM medication WHERE user_id = %s",
                        (user_id,)
                    )
                elif query_type == "events":
                    await cursor.execute(
                        "SELECT * FROM calendar_events WHERE user_id = %s AND event_date >= CURDATE() ORDER BY event_date",
                        (user_id,)
                    )
                elif query_type == "notifications":
                    await cursor.execute(
                        "SELECT * FROM notifications WHERE user_id = %s AND is_read = FALSE ORDER BY created_at DESC",
                        (user_id,)
                    )
                elif query_type == "diet":
                    date_str = kwargs.get('date', datetime.now().strftime('%Y-%m-%d'))
                    await cursor.execute(
                        "SELECT * FROM diet_plans WHERE user_id = %s AND meal_date = %s ORDER BY meal_type",
                        (user_id, date_str)
                    )
                else:
                    return []

                results = await cursor.fetchall()
                # datetime/date 객체를 문자열로 변환
                for row in results:
                    for key, value in row.items():
                        if isinstance(value, (datetime, date)):
                            row[key] = value.isoformat()
                return results
