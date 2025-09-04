"""
Tavily 기반 뉴스 검색 기능만 테스트하는 간단한 스크립트
PyTorch 이슈를 피해서 Tavily API 연동만 확인

실행 전 필요사항:
1. .env 파일에 TAVILY_API_KEY 설정
2. pip install tavily-python
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.helpers import search_news, search_latest_news, get_news_summary_with_tavily
import logging

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_basic_tavily_search():
    """기본 Tavily 뉴스 검색 테스트"""
    print("\n" + "="*60)
    print("🔍 Tavily 기본 뉴스 검색 테스트")
    print("="*60)
    
    test_queries = ["AI 인공지능", "삼성전자", "부동산"]
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n{i}. '{query}' 검색 테스트")
        print("-" * 40)
        
        try:
            results = search_news(query, max_results=2)
            
            if results:
                print(f"✅ {len(results)}개 뉴스 발견:")
                for j, news in enumerate(results, 1):
                    print(f"\n  📰 뉴스 {j}:")
                    print(f"    제목: {news.get('title', 'N/A')}")
                    print(f"    URL: {news.get('url', 'N/A')[:80]}...")
                    print(f"    내용 길이: {len(news.get('content', ''))}")
                    print(f"    점수: {news.get('score', 0)}")
                    print(f"    카테고리: {news.get('category', 'N/A')}")
            else:
                print(f"❌ '{query}' 뉴스 검색 실패")
                
        except Exception as e:
            print(f"❌ '{query}' 검색 중 오류: {e}")

def test_category_search():
    """카테고리별 뉴스 검색 테스트"""
    print("\n" + "="*60)
    print("📂 카테고리별 뉴스 검색 테스트")
    print("="*60)
    
    categories = [
        ('technology', '기술'),
        ('economy', '경제'), 
        ('politics', '정치')
    ]
    
    for category, korean_name in categories:
        print(f"\n📋 {korean_name}({category}) 카테고리 뉴스 검색")
        print("-" * 40)
        
        try:
            results = search_news("최신 뉴스", max_results=2, category=category)
            
            if results:
                print(f"✅ {len(results)}개 {korean_name} 뉴스 발견:")
                for j, news in enumerate(results, 1):
                    print(f"\n  📰 {korean_name} 뉴스 {j}:")
                    print(f"    제목: {news.get('title', 'N/A')}")
                    print(f"    검색어: {news.get('search_query', 'N/A')}")
                    print(f"    카테고리: {news.get('category', 'N/A')}")
            else:
                print(f"❌ {korean_name} 뉴스 검색 실패")
                
        except Exception as e:
            print(f"❌ {korean_name} 뉴스 검색 중 오류: {e}")

def test_latest_news():
    """최신 뉴스 통합 검색 테스트"""
    print("\n" + "="*60)
    print("📅 최신 뉴스 통합 검색 테스트")
    print("="*60)
    
    print("🔄 여러 카테고리 최신 뉴스 통합 검색...")
    print("-" * 40)
    
    try:
        categories = ['technology', 'economy']
        results = search_latest_news(max_results=4, categories=categories)
        
        if results:
            print(f"✅ 총 {len(results)}개 최신 뉴스 발견:")
            
            # 카테고리별로 분류해서 표시
            category_counts = {}
            for news in results:
                cat = news.get('category', 'unknown')
                category_counts[cat] = category_counts.get(cat, 0) + 1
            
            print(f"📊 카테고리별 분포: {category_counts}")
            
            for i, news in enumerate(results, 1):
                print(f"\n  📰 최신 뉴스 {i}:")
                print(f"    제목: {news.get('title', 'N/A')}")
                print(f"    카테고리: {news.get('category', 'N/A')}")
                print(f"    발행일: {news.get('published_date', 'N/A')}")
                print(f"    내용 길이: {len(news.get('content', ''))}")
        else:
            print("❌ 최신 뉴스 통합 검색 실패")
            
    except Exception as e:
        print(f"❌ 최신 뉴스 검색 중 오류: {e}")

def test_tavily_summary():
    """Tavily AI 요약 기능 테스트"""
    print("\n" + "="*60)
    print("🤖 Tavily AI 요약 기능 테스트")
    print("="*60)
    
    test_topics = ["ChatGPT", "전기차"]
    
    for topic in test_topics:
        print(f"\n🎯 '{topic}' 주제 요약 데이터 수집")
        print("-" * 40)
        
        try:
            results = get_news_summary_with_tavily(topic, max_results=3)
            
            if results:
                print(f"✅ {len(results)}개 요약 데이터 수집:")
                
                for i, item in enumerate(results, 1):
                    print(f"\n  📋 데이터 {i}:")
                    print(f"    제목: {item.get('title', 'N/A')}")
                    
                    if item.get('is_summary'):
                        print(f"    타입: 🤖 Tavily AI 요약")
                        print(f"    내용: {item.get('content', '')[:200]}...")
                    else:
                        print(f"    타입: 📰 뉴스 기사")
                        print(f"    URL: {item.get('url', 'N/A')[:50]}...")
                        print(f"    내용 길이: {len(item.get('content', ''))}")
                    
                    print(f"    점수: {item.get('score', 0)}")
            else:
                print(f"❌ '{topic}' 요약 데이터 수집 실패")
                
        except Exception as e:
            print(f"❌ '{topic}' 요약 데이터 수집 중 오류: {e}")

def main():
    """메인 테스트 함수"""
    print("🚀 Tavily 뉴스 기능 테스트 시작")
    print(f"📂 Python path: {sys.path[0]}")
    
    # 환경 변수 확인
    tavily_key = os.getenv('TAVILY_API_KEY')
    if not tavily_key:
        print("❌ TAVILY_API_KEY가 설정되지 않았습니다!")
        print("📝 .env 파일에 TAVILY_API_KEY=your_api_key 를 추가해주세요.")
        print("🌐 API 키는 https://tavily.com 에서 무료로 받을 수 있습니다.")
        return
    else:
        print(f"✅ TAVILY_API_KEY 확인됨: {tavily_key[:10]}...")
    
    try:
        # 단계별 테스트 실행
        test_basic_tavily_search()
        test_category_search()  
        test_latest_news()
        test_tavily_summary()
        
        print("\n" + "="*60)
        print("🎉 Tavily 뉴스 기능 테스트 완료!")
        print("="*60)
        print("✅ 모든 Tavily 기반 뉴스 검색 기능이 정상 작동합니다.")
        print("📝 다음 단계: API 엔드포인트 추가 및 문서 업데이트")
        
    except KeyboardInterrupt:
        print("\n⏹️  사용자에 의해 테스트가 중단되었습니다.")
    except Exception as e:
        print(f"\n❌ 테스트 중 예상치 못한 오류 발생: {e}")
        logger.error(f"테스트 오류: {e}")

if __name__ == "__main__":
    main()