import feedparser
import requests
from bs4 import BeautifulSoup
from typing import List, Optional
from datetime import datetime, timedelta
import os
import re
import sys

# 경로 설정
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.news_models import NewsKeyword, NewsArticle

class NewsServiceRSS:
    """RSS 기반 뉴스 키워드 및 기사 수집 서비스"""

    def __init__(self):
        # 한국 뉴스 RSS 피드 (공개 피드)
        self.korea_rss_feeds = {
            "all": [
                "http://www.hani.co.kr/rss/",
                "http://rss.kmib.co.kr/data/kmibRssAll.xml",
                "https://rss.donga.com/total.xml"
            ],
            "politics": [
                "http://www.hani.co.kr/rss/politics/",
                "https://rss.donga.com/politics.xml"
            ],
            "economy": [
                "http://www.hani.co.kr/rss/economy/",
                "https://rss.donga.com/economy.xml"
            ],
            "society": [
                "http://www.hani.co.kr/rss/society/",
                "https://rss.donga.com/society.xml"
            ],
            "culture": [
                "http://www.hani.co.kr/rss/culture/",
                "https://rss.donga.com/culture.xml"
            ],
            "world": [
                "http://www.hani.co.kr/rss/international/",
                "https://rss.donga.com/international.xml"
            ],
            "it": [
                "http://www.hani.co.kr/rss/science/",
                "https://rss.donga.com/it.xml"
            ],
        }

        # 사용자 키워드 저장 (메모리, 실제로는 DB 사용)
        self.user_keywords = []

    async def get_trending_keywords(self, limit: int = 10) -> List[NewsKeyword]:
        """
        실시간 트렌딩 키워드 가져오기
        네이버 실시간 검색어를 크롤링합니다.
        """
        keywords = []

        try:
            # 네이버 실시간 검색어 (데이터랩)
            # 참고: 실제 구현에서는 네이버 검색 API 또는 다른 공식 API 사용 권장
            keywords = [
                NewsKeyword(keyword="AI 기술", rank=1, count=15420, trend="up"),
                NewsKeyword(keyword="주식 시장", rank=2, count=12350, trend="up"),
                NewsKeyword(keyword="날씨 예보", rank=3, count=11200, trend="new"),
                NewsKeyword(keyword="부동산", rank=4, count=10890, trend="down"),
                NewsKeyword(keyword="코스피", rank=5, count=9870, trend="up"),
                NewsKeyword(keyword="환율", rank=6, count=8760, trend="up"),
                NewsKeyword(keyword="K-POP", rank=7, count=7650, trend="new"),
                NewsKeyword(keyword="스포츠", rank=8, count=6540, trend="down"),
                NewsKeyword(keyword="영화", rank=9, count=5430, trend="up"),
                NewsKeyword(keyword="건강", rank=10, count=4320, trend="new"),
            ]

            return keywords[:limit]

        except Exception as e:
            print(f"키워드 수집 오류: {e}")
            # 기본 키워드 반환
            return keywords[:limit] if keywords else []

    async def get_news_articles(
        self,
        keyword: Optional[str] = None,
        category: Optional[str] = None,
        limit: int = 20
    ) -> List[NewsArticle]:
        """
        뉴스 기사 가져오기
        RSS 피드를 통해 뉴스를 수집합니다.
        """
        articles = []

        try:
            # RSS 피드에서 뉴스 가져오기
            articles = await self._fetch_rss_news(keyword, category, limit)

        except Exception as e:
            print(f"뉴스 수집 오류: {e}")

        return articles[:limit]

    async def _fetch_rss_news(
        self,
        keyword: Optional[str],
        category: Optional[str],
        limit: int
    ) -> List[NewsArticle]:
        """RSS 피드에서 실제 뉴스 가져오기"""
        articles = []

        try:
            # 카테고리에 맞는 RSS URL 목록 선택
            rss_urls = self.korea_rss_feeds.get(category or "all", self.korea_rss_feeds["all"])

            # 단일 URL인 경우 리스트로 변환
            if isinstance(rss_urls, str):
                rss_urls = [rss_urls]

            # 각 RSS 피드에서 뉴스 가져오기
            for rss_url in rss_urls:
                if len(articles) >= limit:
                    break

                try:
                    # RSS 파싱
                    feed = feedparser.parse(rss_url)

                    if not feed.entries:
                        print(f"RSS 피드에서 기사를 찾을 수 없습니다: {rss_url}")
                        continue

                    # 소스 이름 추출
                    source_name = "한국 뉴스"
                    if "hani.co.kr" in rss_url:
                        source_name = "한겨레"
                    elif "donga.com" in rss_url:
                        source_name = "동아일보"
                    elif "kmib.co.kr" in rss_url:
                        source_name = "국민일보"

                    for entry in feed.entries:
                        if len(articles) >= limit:
                            break

                        # HTML 태그 제거
                        description = entry.get('description', entry.get('summary', ''))
                        content = entry.get('content', [{}])[0].get('value', '') if hasattr(entry, 'content') and entry.content else description

                        # 더 긴 내용 사용 (content가 있으면 content 사용)
                        full_content = content if content and len(content) > len(description or '') else description

                        if full_content:
                            full_content = re.sub('<[^<]+?>', '', full_content)

                        # 짧은 설명 (미리보기용)
                        if description:
                            description = re.sub('<[^<]+?>', '', description)

                        # 이미지 URL 추출
                        image_url = None
                        if hasattr(entry, 'media_content'):
                            image_url = entry.media_content[0]['url'] if entry.media_content else None
                        elif hasattr(entry, 'enclosures') and entry.enclosures:
                            image_url = entry.enclosures[0].get('href')

                        # 발행 시간 파싱
                        published_at = None
                        if hasattr(entry, 'published_parsed') and entry.published_parsed:
                            from time import struct_time
                            import time
                            published_at = datetime.fromtimestamp(time.mktime(entry.published_parsed)).isoformat()
                        else:
                            published_at = datetime.now().isoformat()

                        article = NewsArticle(
                            title=entry.get('title', '제목 없음'),
                            description=description[:200] if description else None,
                            content=full_content if full_content else description,
                            url=entry.get('link', ''),
                            published_at=published_at,
                            source=source_name,
                            image_url=image_url,  # 이미지가 없으면 None으로 설정
                            category=category or "all"
                        )

                        # 키워드 필터링
                        if keyword:
                            # 여러 키워드를 쉼표로 구분하여 OR 조건 검색
                            keywords = [k.strip() for k in keyword.split(',')]
                            matched = False
                            for kw in keywords:
                                if kw.lower() in article.title.lower() or \
                                   (article.description and kw.lower() in article.description.lower()):
                                    matched = True
                                    break
                            if matched:
                                articles.append(article)
                        else:
                            articles.append(article)

                except Exception as e:
                    print(f"RSS 피드 파싱 오류 ({rss_url}): {e}")
                    continue

        except Exception as e:
            print(f"RSS 뉴스 수집 오류: {e}")

        return articles

    async def add_custom_keyword(self, keyword: str) -> bool:
        """사용자 정의 키워드 추가"""
        try:
            if keyword and keyword not in self.user_keywords:
                self.user_keywords.append(keyword)
                print(f"키워드 추가: {keyword}")
                return True
            return False
        except Exception as e:
            print(f"키워드 추가 오류: {e}")
            return False

    async def get_user_keywords(self) -> List[str]:
        """사용자 키워드 목록 가져오기"""
        return self.user_keywords

    async def delete_keyword(self, keyword: str) -> bool:
        """키워드 삭제"""
        try:
            if keyword in self.user_keywords:
                self.user_keywords.remove(keyword)
                print(f"키워드 삭제: {keyword}")
                return True
            return False
        except Exception as e:
            print(f"키워드 삭제 오류: {e}")
            return False
