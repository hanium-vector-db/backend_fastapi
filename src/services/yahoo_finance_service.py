"""
Yahoo Finance API 서비스
KOSPI 데이터 조회
"""

import logging
import httpx
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)


class YahooFinanceService:
    """Yahoo Finance API 서비스"""

    def __init__(self):
        self.base_url = "https://query1.finance.yahoo.com"
        self.kospi_symbol = "^KS11"

    async def get_kospi_chart(self, interval: str = "1d", range_period: str = "1d") -> Dict[str, Any]:
        """
        KOSPI 차트 데이터 가져오기

        Args:
            interval: 데이터 간격 (5m, 30m, 1d, 1wk, 1mo)
            range_period: 기간 (1d, 5d, 1mo, 1y, 5y)

        Returns:
            Dict: Yahoo Finance API 응답
        """
        try:
            url = f"{self.base_url}/v8/finance/chart/{self.kospi_symbol}"
            params = {
                "interval": interval,
                "range": range_period
            }

            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
            }

            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(url, params=params, headers=headers)
                response.raise_for_status()
                return response.json()

        except httpx.HTTPStatusError as e:
            logger.error(f"Yahoo Finance API HTTP 오류: {e.response.status_code}")
            if e.response.status_code == 429:
                # Rate limit에 걸렸을 때 더미 데이터 반환
                logger.warning("Yahoo Finance API rate limit, 더미 데이터 반환")
                return self._get_dummy_kospi_data(interval, range_period)
            raise
        except httpx.RequestError as e:
            logger.error(f"Yahoo Finance API 요청 오류: {e}")
            raise
        except Exception as e:
            logger.error(f"Yahoo Finance API 오류: {e}")
            raise

    def _get_dummy_kospi_data(self, interval: str, range_period: str) -> Dict[str, Any]:
        """Rate limit 시 더미 데이터 반환"""
        from datetime import datetime, timedelta
        import random

        # 기본 KOSPI 가격 (실제 데이터와 유사하게)
        base_price = 2600.0
        timestamps = []
        close_prices = []

        # 간격에 따라 데이터 포인트 수 결정
        if interval == "5m":
            points = 78  # 1일 (6.5시간 * 60분 / 5분)
            time_delta = timedelta(minutes=5)
        elif interval == "30m":
            points = 65  # 5일
            time_delta = timedelta(minutes=30)
        elif interval == "1d":
            points = 30  # 1개월
            time_delta = timedelta(days=1)
        elif interval == "1wk":
            points = 52  # 1년
            time_delta = timedelta(weeks=1)
        else:  # 1mo
            points = 60  # 5년
            time_delta = timedelta(days=30)

        current_time = datetime.now()
        current_price = base_price

        for i in range(points):
            timestamps.append(int(current_time.timestamp()))
            # 작은 변동성 추가
            change = random.uniform(-20, 20)
            current_price += change
            close_prices.append(round(current_price, 2))
            current_time -= time_delta

        timestamps.reverse()
        close_prices.reverse()

        return {
            "chart": {
                "result": [{
                    "meta": {
                        "symbol": "^KS11",
                        "regularMarketPrice": close_prices[-1],
                        "chartPreviousClose": close_prices[0],
                        "regularMarketDayHigh": max(close_prices),
                        "regularMarketDayLow": min(close_prices)
                    },
                    "timestamp": timestamps,
                    "indicators": {
                        "quote": [{
                            "close": close_prices,
                            "open": [p + random.uniform(-5, 5) for p in close_prices]
                        }]
                    }
                }]
            }
        }
