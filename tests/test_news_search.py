#!/usr/bin/env python3
import sys
import os
sys.path.append('src')

from utils.helpers import search_news, create_documents_from_news
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_news_search():
    """개선된 뉴스 검색 함수를 테스트합니다"""

    print("=== 개선된 뉴스 검색 테스트 ===")

    # 테스트 쿼리들
    test_queries = [
        "인공지능",
        "경제 동향",
        "기술 혁신"
    ]

    for query in test_queries:
        print(f"\n--- '{query}' 검색 결과 ---")

        try:
            # 뉴스 검색
            news_results = search_news(query, max_results=3)

            if not news_results:
                print("검색 결과가 없습니다.")
                continue

            print(f"검색된 기사 수: {len(news_results)}")

            for i, news in enumerate(news_results):
                print(f"\n기사 {i+1}:")
                print(f"제목: {news.get('title', 'N/A')}")
                print(f"URL: {news.get('url', 'N/A')}")
                print(f"카테고리: {news.get('category', 'N/A')}")
                print(f"점수: {news.get('score', 'N/A')}")
                print(f"내용 (앞 100자): {news.get('content', '')[:100]}...")
                print(f"검색어: {news.get('search_query', 'N/A')}")

            # 문서 생성 테스트
            documents = create_documents_from_news(news_results)
            print(f"\n생성된 문서 수: {len(documents)}")

            if documents:
                print("첫 번째 문서 샘플:")
                doc = documents[0]
                print(f"메타데이터: {doc.metadata}")
                print(f"내용 길이: {len(doc.page_content)} 문자")
                print(f"내용 (앞 150자): {doc.page_content[:150]}...")

        except Exception as e:
            print(f"'{query}' 검색 중 오류: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    test_news_search()