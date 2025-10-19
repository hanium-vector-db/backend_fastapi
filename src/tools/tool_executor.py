"""
Tool 실행 함수들
"""

import logging
import asyncio
from typing import Dict, Any
from datetime import datetime, timedelta
import aiomysql
import httpx

logger = logging.getLogger(__name__)


class ToolExecutor:
    """Tool 실행 클래스"""

    def __init__(self, db_config: Dict[str, Any], grocery_rag_service=None):
        self.db_config = db_config
        self.db_pool = None
        self.grocery_rag_service = grocery_rag_service

    async def initialize(self):
        """데이터베이스 연결 풀 초기화"""
        try:
            self.db_pool = await aiomysql.create_pool(
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
            logger.info("ToolExecutor DB 연결 풀 초기화 완료")
        except Exception as e:
            logger.error(f"ToolExecutor DB 연결 풀 초기화 실패: {e}")
            raise

    async def close(self):
        """연결 풀 종료"""
        if self.db_pool:
            self.db_pool.close()
            await self.db_pool.wait_closed()

    async def execute_tool(self, tool_name: str, parameters: Dict[str, Any], user_id: str) -> Dict[str, Any]:
        """도구 실행"""
        try:
            if tool_name == "get_news":
                return await self._get_news(parameters)
            elif tool_name == "get_weather":
                return await self._get_weather(parameters)
            elif tool_name == "get_health_status":
                return await self._get_health_status(user_id)
            elif tool_name == "get_diseases":
                return await self._get_diseases(user_id)
            elif tool_name == "get_medications":
                return await self._get_medications(user_id)
            elif tool_name == "add_disease":
                return await self._add_disease(user_id, parameters)
            elif tool_name == "add_medication":
                return await self._add_medication(user_id, parameters)
            elif tool_name == "get_finance_updates":
                return await self._get_finance_updates(parameters)
            elif tool_name == "get_calendar_events":
                return await self._get_calendar_events(user_id, parameters)
            elif tool_name == "add_calendar_event":
                return await self._add_calendar_event(user_id, parameters)
            elif tool_name == "delete_calendar_event":
                return await self._delete_calendar_event(user_id, parameters)
            elif tool_name == "get_diet_plan":
                return await self._get_diet_plan(user_id, parameters)
            elif tool_name == "get_notifications":
                return await self._get_notifications(user_id, parameters)
            elif tool_name == "get_grocery_prices":
                return await self._get_grocery_prices(parameters)
            else:
                return {"error": f"알 수 없는 도구: {tool_name}"}
        except Exception as e:
            logger.error(f"도구 실행 오류 ({tool_name}): {e}")
            return {"error": str(e)}

    async def _get_news(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """뉴스 조회"""
        try:
            from services.news_service_rss import NewsServiceRSS

            keyword = parameters.get("keyword")
            limit = parameters.get("limit", 5)

            news_service = NewsServiceRSS()
            articles = await news_service.get_news_articles(keyword=keyword, limit=limit)

            # Convert NewsArticle objects to dictionaries for JSON serialization
            articles_dict = [
                {
                    "title": article.title,
                    "url": article.url,
                    "source": article.source,
                    "published_at": article.published_at if isinstance(article.published_at, str)
                                    else (article.published_at.isoformat() if article.published_at else None)
                }
                for article in articles
            ]

            return {
                "success": True,
                "data": articles_dict,
                "message": f"{len(articles)}개의 뉴스 기사를 찾았습니다."
            }
        except Exception as e:
            logger.error(f"뉴스 조회 오류: {e}")
            return {"success": False, "error": str(e)}

    async def _get_weather(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """날씨 조회 (더미 데이터)"""
        location = parameters.get("location", "서울")

        # 실제로는 날씨 API를 호출해야 하지만, 여기서는 더미 데이터 반환
        return {
            "success": True,
            "data": {
                "location": location,
                "temperature": "22°C",
                "description": "맑음",
                "humidity": "60%",
                "wind_speed": "3m/s"
            },
            "message": f"{location}의 현재 날씨 정보입니다."
        }

    async def _get_health_status(self, user_id: str) -> Dict[str, Any]:
        """건강 상태 조회"""
        try:
            async with self.db_pool.acquire() as conn:
                async with conn.cursor(aiomysql.DictCursor) as cursor:
                    # 기저질환 조회
                    await cursor.execute(
                        "SELECT name, status FROM disease WHERE user_id = %s",
                        (user_id,)
                    )
                    diseases = await cursor.fetchall()

                    # 복용약 조회
                    await cursor.execute(
                        "SELECT name, dosage, intake_time FROM medication WHERE user_id = %s",
                        (user_id,)
                    )
                    medications = await cursor.fetchall()

                    # timedelta를 문자열로 변환
                    for med in medications:
                        if med.get('intake_time'):
                            td = med['intake_time']
                            hours, remainder = divmod(td.seconds, 3600)
                            minutes, _ = divmod(remainder, 60)
                            med['intake_time'] = f"{hours:02d}:{minutes:02d}"

                    # 최근 건강 리포트
                    await cursor.execute(
                        "SELECT health_score, ai_insights FROM daily_reports WHERE user_id = %s ORDER BY report_date DESC LIMIT 1",
                        (user_id,)
                    )
                    report = await cursor.fetchone()

                    return {
                        "success": True,
                        "data": {
                            "diseases": diseases,
                            "medications": medications,
                            "health_score": report['health_score'] if report else None,
                            "ai_insights": report['ai_insights'] if report else None
                        },
                        "message": "건강 상태를 조회했습니다."
                    }
        except Exception as e:
            logger.error(f"건강 상태 조회 오류: {e}")
            return {"success": False, "error": str(e)}

    async def _get_diseases(self, user_id: str) -> Dict[str, Any]:
        """기저질환 목록 조회"""
        try:
            async with self.db_pool.acquire() as conn:
                async with conn.cursor(aiomysql.DictCursor) as cursor:
                    await cursor.execute(
                        "SELECT disease_id, name, diagnosed_date, status FROM disease WHERE user_id = %s",
                        (user_id,)
                    )
                    diseases = await cursor.fetchall()

                    # 날짜 변환
                    for disease in diseases:
                        if disease.get('diagnosed_date'):
                            disease['diagnosed_date'] = disease['diagnosed_date'].isoformat()

                    return {
                        "success": True,
                        "data": diseases,
                        "message": f"{len(diseases)}개의 기저질환을 찾았습니다."
                    }
        except Exception as e:
            logger.error(f"기저질환 조회 오류: {e}")
            return {"success": False, "error": str(e)}

    async def _get_medications(self, user_id: str) -> Dict[str, Any]:
        """복용약 목록 조회"""
        try:
            async with self.db_pool.acquire() as conn:
                async with conn.cursor(aiomysql.DictCursor) as cursor:
                    await cursor.execute(
                        "SELECT medication_id, name, dosage, intake_time, alarm_enabled FROM medication WHERE user_id = %s",
                        (user_id,)
                    )
                    medications = await cursor.fetchall()

                    # timedelta를 문자열로 변환
                    for med in medications:
                        if med.get('intake_time'):
                            td = med['intake_time']
                            hours, remainder = divmod(td.seconds, 3600)
                            minutes, _ = divmod(remainder, 60)
                            med['intake_time'] = f"{hours:02d}:{minutes:02d}"
                        if med.get('alarm_enabled') is not None:
                            med['alarm_enabled'] = bool(med['alarm_enabled'])

                    return {
                        "success": True,
                        "data": medications,
                        "message": f"{len(medications)}개의 복용약을 찾았습니다."
                    }
        except Exception as e:
            logger.error(f"복용약 조회 오류: {e}")
            return {"success": False, "error": str(e)}

    async def _add_disease(self, user_id: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """기저질환 추가"""
        try:
            name = parameters.get("name")
            diagnosed_date = parameters.get("diagnosed_date")
            status = parameters.get("status", "active")

            async with self.db_pool.acquire() as conn:
                async with conn.cursor(aiomysql.DictCursor) as cursor:
                    await cursor.execute(
                        "INSERT INTO disease (user_id, name, diagnosed_date, status) VALUES (%s, %s, %s, %s)",
                        (user_id, name, diagnosed_date, status)
                    )
                    disease_id = cursor.lastrowid

                    return {
                        "success": True,
                        "data": {"disease_id": disease_id, "name": name},
                        "message": f"기저질환 '{name}'을(를) 추가했습니다."
                    }
        except Exception as e:
            logger.error(f"기저질환 추가 오류: {e}")
            return {"success": False, "error": str(e)}

    async def _add_medication(self, user_id: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """복용약 추가"""
        try:
            name = parameters.get("name")
            dosage = parameters.get("dosage")
            intake_time = parameters.get("intake_time")

            async with self.db_pool.acquire() as conn:
                async with conn.cursor(aiomysql.DictCursor) as cursor:
                    await cursor.execute(
                        "INSERT INTO medication (user_id, name, dosage, intake_time, alarm_enabled) VALUES (%s, %s, %s, %s, %s)",
                        (user_id, name, dosage, intake_time, True)
                    )
                    medication_id = cursor.lastrowid

                    return {
                        "success": True,
                        "data": {"medication_id": medication_id, "name": name},
                        "message": f"복용약 '{name}'을(를) 추가했습니다."
                    }
        except Exception as e:
            logger.error(f"복용약 추가 오류: {e}")
            return {"success": False, "error": str(e)}

    async def _get_finance_updates(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """금융 정보 조회 (Yahoo Finance API 사용)"""
        try:
            symbol = parameters.get("symbol", "^KS11")  # 기본값: 코스피 (^KS11)

            # Yahoo Finance API 호출
            url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}?interval=1d&range=1d"

            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(url)
                response.raise_for_status()
                data = response.json()

            if not data or 'chart' not in data or 'result' not in data['chart'] or not data['chart']['result']:
                return {
                    "success": False,
                    "error": "코스피 데이터를 가져올 수 없습니다."
                }

            result = data['chart']['result'][0]
            meta = result['meta']

            # 현재가 및 변동 정보 추출
            current_price = meta.get('regularMarketPrice', 0)
            previous_close = meta.get('chartPreviousClose', 0)
            change = current_price - previous_close
            change_percent = (change / previous_close * 100) if previous_close > 0 else 0

            return {
                "success": True,
                "data": {
                    "symbol": meta.get('symbol', symbol),
                    "current_price": round(current_price, 2),
                    "previous_close": round(previous_close, 2),
                    "change": round(change, 2),
                    "change_percent": round(change_percent, 2),
                    "day_high": meta.get('regularMarketDayHigh', 0),
                    "day_low": meta.get('regularMarketDayLow', 0),
                    "currency": meta.get('currency', 'KRW')
                },
                "message": f"코스피 현재가: {current_price:,.2f} ({'+' if change >= 0 else ''}{change:.2f}, {'+' if change_percent >= 0 else ''}{change_percent:.2f}%)"
            }
        except httpx.HTTPError as e:
            logger.error(f"Yahoo Finance API 호출 오류: {e}")
            return {
                "success": False,
                "error": f"금융 데이터 조회 중 오류가 발생했습니다: {str(e)}"
            }
        except Exception as e:
            logger.error(f"금융 정보 조회 오류: {e}")
            return {
                "success": False,
                "error": str(e)
            }

    async def _get_calendar_events(self, user_id: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """일정 조회 (DB 사용)"""
        try:
            date_filter = parameters.get("date", datetime.now().strftime('%Y-%m-%d'))
            limit = parameters.get("limit", 10)

            async with self.db_pool.acquire() as conn:
                async with conn.cursor(aiomysql.DictCursor) as cursor:
                    await cursor.execute(
                        """SELECT id as event_id, event_type, title, description, location,
                                  event_date, event_time, is_completed
                           FROM calendar_events
                           WHERE user_id = %s AND event_date >= %s
                           ORDER BY event_date, event_time
                           LIMIT %s""",
                        (user_id, date_filter, limit)
                    )
                    events = await cursor.fetchall()

                    # 날짜 및 시간 변환
                    for event in events:
                        if event.get('event_date'):
                            event['event_date'] = event['event_date'].isoformat()
                        if event.get('event_time'):
                            td = event['event_time']
                            hours, remainder = divmod(td.seconds, 3600)
                            minutes, _ = divmod(remainder, 60)
                            event['event_time'] = f"{hours:02d}:{minutes:02d}"
                        if event.get('is_completed') is not None:
                            event['is_completed'] = bool(event['is_completed'])

                    logger.info(f"일정 조회: user_id={user_id}, count={len(events)}")

                    return {
                        "success": True,
                        "data": events,
                        "message": f"{len(events)}개의 일정을 찾았습니다."
                    }
        except Exception as e:
            logger.error(f"일정 조회 오류: {e}")
            return {"success": False, "error": str(e)}

    async def _add_calendar_event(self, user_id: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """일정 추가"""
        try:
            title = parameters.get("title")
            event_date = parameters.get("event_date")
            event_time = parameters.get("event_time")
            event_type = parameters.get("event_type", "약속")
            location = parameters.get("location", "")
            description = parameters.get("description", "")

            if not title or not event_date or not event_time:
                return {
                    "success": False,
                    "error": "제목, 날짜, 시간은 필수 항목입니다."
                }

            async with self.db_pool.acquire() as conn:
                async with conn.cursor(aiomysql.DictCursor) as cursor:
                    await cursor.execute(
                        """INSERT INTO calendar_events
                           (user_id, event_type, title, description, location, event_date, event_time, is_completed, reminder_enabled)
                           VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)""",
                        (user_id, event_type, title, description, location, event_date, event_time, False, True)
                    )
                    event_id = cursor.lastrowid

                    logger.info(f"일정 추가: user_id={user_id}, event_id={event_id}, title={title}")

                    return {
                        "success": True,
                        "data": {
                            "event_id": event_id,
                            "title": title,
                            "event_date": event_date,
                            "event_time": event_time
                        },
                        "message": f"일정 '{title}'이(가) {event_date} {event_time}에 추가되었습니다."
                    }
        except Exception as e:
            logger.error(f"일정 추가 오류: {e}")
            return {"success": False, "error": str(e)}

    async def _delete_calendar_event(self, user_id: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """일정 삭제"""
        try:
            event_id = parameters.get("event_id")

            if not event_id:
                return {
                    "success": False,
                    "error": "삭제할 일정 ID가 필요합니다."
                }

            async with self.db_pool.acquire() as conn:
                async with conn.cursor(aiomysql.DictCursor) as cursor:
                    # 일정이 존재하고 사용자 소유인지 확인
                    await cursor.execute(
                        "SELECT title FROM calendar_events WHERE id = %s AND user_id = %s",
                        (event_id, user_id)
                    )
                    event = await cursor.fetchone()

                    if not event:
                        return {
                            "success": False,
                            "error": "일정을 찾을 수 없거나 삭제 권한이 없습니다."
                        }

                    # 일정 삭제
                    await cursor.execute(
                        "DELETE FROM calendar_events WHERE id = %s AND user_id = %s",
                        (event_id, user_id)
                    )

                    logger.info(f"일정 삭제: user_id={user_id}, event_id={event_id}, title={event['title']}")

                    return {
                        "success": True,
                        "data": {"event_id": event_id},
                        "message": f"일정 '{event['title']}'이(가) 삭제되었습니다."
                    }
        except Exception as e:
            logger.error(f"일정 삭제 오류: {e}")
            return {"success": False, "error": str(e)}

    async def _get_diet_plan(self, user_id: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """건강 상태 기반 식단 추천"""
        try:
            # 사용자의 건강 정보 먼저 조회
            diseases_result = await self._get_diseases(user_id)
            medications_result = await self._get_medications(user_id)

            diseases = diseases_result.get("data", []) if diseases_result.get("success") else []
            medications = medications_result.get("data", []) if medications_result.get("success") else []

            disease_names = [d.get('name', '') for d in diseases] if diseases else []
            medication_names = [m.get('name', '') for m in medications] if medications else []

            # 건강 상태 정보 반환 (LLM이 이를 기반으로 추천)
            health_context = {
                "diseases": disease_names,
                "medications": medication_names,
                "has_health_info": len(disease_names) > 0 or len(medication_names) > 0
            }

            return {
                "success": True,
                "data": health_context,
                "message": f"건강 정보를 확인했습니다. 기저질환 {len(disease_names)}개, 복용약 {len(medication_names)}개가 등록되어 있습니다."
            }
        except Exception as e:
            logger.error(f"식단 조회 오류: {e}")
            # 에러가 발생해도 기본 정보를 반환
            return {
                "success": True,
                "data": {"diseases": [], "medications": [], "has_health_info": False},
                "message": "건강 정보를 확인하지 못했습니다. 일반적인 건강식을 추천합니다."
            }

    async def _get_notifications(self, user_id: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """알림 조회"""
        try:
            limit = parameters.get("limit", 5)

            async with self.db_pool.acquire() as conn:
                async with conn.cursor(aiomysql.DictCursor) as cursor:
                    await cursor.execute(
                        """SELECT notification_id, notification_type, title, message, priority, created_at
                           FROM notifications
                           WHERE user_id = %s AND is_read = FALSE
                           ORDER BY created_at DESC LIMIT %s""",
                        (user_id, limit)
                    )
                    notifications = await cursor.fetchall()

                    # 날짜 변환
                    for notif in notifications:
                        if notif.get('created_at'):
                            notif['created_at'] = notif['created_at'].isoformat()

                    return {
                        "success": True,
                        "data": notifications,
                        "message": f"{len(notifications)}개의 미읽은 알림을 찾았습니다."
                    }
        except Exception as e:
            logger.error(f"알림 조회 오류: {e}")
            return {"success": False, "error": str(e)}

    async def _get_grocery_prices(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """식료품 가격 조회 (RAG 사용)"""
        try:
            if not self.grocery_rag_service:
                return {
                    "success": False,
                    "error": "식료품 가격 서비스가 초기화되지 않았습니다."
                }

            product = parameters.get("product", "")
            if not product:
                return {
                    "success": False,
                    "error": "상품명을 입력해주세요."
                }

            # RAG를 사용하여 식료품 가격 정보 조회
            result = self.grocery_rag_service.get_grocery_recommendations(product)

            return result
        except Exception as e:
            logger.error(f"식료품 가격 조회 오류: {e}")
            return {"success": False, "error": str(e)}
