"""
새로운 뉴스 기능 테스트 스크립트
- Tavily 기반 뉴스 검색
- RAG 서비스의 뉴스 요약 기능
- 트렌드 분석 기능

실행 전 필요사항:
1. .env 파일에 TAVILY_API_KEY 설정
2. pip install tavily-python
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.helpers import search_news, search_latest_news, get_news_summary_with_tavily
from src.services.rag_service import RAGService
from src.models.llm_handler import LLMHandler
from src.models.embedding_handler import EmbeddingHandler
import logging

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_tavily_news_search():
    """Tavily 뉴스 검색 테스트"""
    print("\n" + "="*50)
    print("🔍 Tavily 뉴스 검색 테스트")
    print("="*50)
    
    # 기본 뉴스 검색 테스트
    print("\n1. 기본 뉴스 검색 테스트 ('AI' 키워드)")
    results = search_news("AI", max_results=3)
    
    if results:
        for i, news in enumerate(results, 1):
            print(f"\n📰 뉴스 {i}:")
            print(f"제목: {news.get('title', 'N/A')}")
            print(f"URL: {news.get('url', 'N/A')}")
            print(f"내용 미리보기: {news.get('content', '')[:200]}...")
            print(f"점수: {news.get('score', 0)}")
            print(f"카테고리: {news.get('category', 'N/A')}")
    else:
        print("❌ 뉴스 검색 실패")

def test_category_news_search():
    """카테고리별 뉴스 검색 테스트"""
    print("\n" + "="*50)
    print("📂 카테고리별 뉴스 검색 테스트")
    print("="*50)
    
    categories = ['technology', 'economy', 'politics']
    
    for category in categories:
        print(f"\n2. {category} 카테고리 뉴스 검색")
        results = search_news("최신", max_results=2, category=category)
        
        if results:
            for i, news in enumerate(results, 1):
                print(f"\n📰 {category} 뉴스 {i}:")
                print(f"제목: {news.get('title', 'N/A')}")
                print(f"카테고리: {news.get('category', 'N/A')}")
                print(f"검색어: {news.get('search_query', 'N/A')}")
        else:
            print(f"❌ {category} 카테고리 뉴스 검색 실패")

def test_latest_news():
    """최신 뉴스 종합 검색 테스트"""
    print("\n" + "="*50)
    print("📅 최신 뉴스 종합 검색 테스트")
    print("="*50)
    
    print("\n3. 여러 카테고리 최신 뉴스 검색")
    results = search_latest_news(max_results=6, categories=['technology', 'economy'])
    
    if results:
        print(f"\n총 {len(results)}개 뉴스 발견:")
        for i, news in enumerate(results, 1):
            print(f"\n📰 최신 뉴스 {i}:")
            print(f"제목: {news.get('title', 'N/A')}")
            print(f"카테고리: {news.get('category', 'N/A')}")
            print(f"발행일: {news.get('published_date', 'N/A')}")
    else:
        print("❌ 최신 뉴스 검색 실패")

def test_tavily_summary():
    """Tavily 요약 기능 테스트"""
    print("\n" + "="*50)
    print("📝 Tavily 요약 기능 테스트")
    print("="*50)
    
    print("\n4. Tavily로 뉴스 요약 데이터 수집")
    results = get_news_summary_with_tavily("인공지능", max_results=3)
    
    if results:
        for i, item in enumerate(results, 1):
            print(f"\n📋 요약 데이터 {i}:")
            print(f"제목: {item.get('title', 'N/A')}")
            if item.get('is_summary'):
                print("📊 타입: Tavily AI 요약")
            else:
                print("📰 타입: 뉴스 기사")
            print(f"내용: {item.get('content', '')[:300]}...")
    else:
        print("❌ Tavily 요약 데이터 수집 실패")

def test_rag_news_features():
    """RAG 서비스의 뉴스 기능 테스트 (PyTorch 이슈로 인해 스킵)"""
    print("\n" + "="*50)
    print("🤖 RAG 서비스 뉴스 기능 테스트")
    print("="*50)
    
    print("⚠️  현재 PyTorch 버전 이슈로 인해 RAG 서비스 테스트를 스킵합니다.")
    print("📝 PyTorch 2.6+ 또는 safetensors 기반 모델이 필요합니다.")
    print("🔧 나중에 torch 업그레이드 후 다시 테스트해주세요.")
    
    return
    
    # 원래 테스트 코드는 주석 처리
    """
    try:
        print("\n5. RAG 서비스 초기화 중...")
        
        # LLM 및 임베딩 핸들러 초기화
        llm_handler = LLMHandler(model_key="qwen2.5-7b")  # 기본 모델 사용
        embedding_handler = EmbeddingHandler()
        
        # RAG 서비스 초기화
        rag_service = RAGService(llm_handler, embedding_handler)
        
        print("✅ RAG 서비스 초기화 완료")
        
        # 최신 뉴스 조회 테스트
        print("\n6. RAG 서비스로 최신 뉴스 조회")
        latest_news = rag_service.get_latest_news(categories=['technology'], max_results=3)
        
        if latest_news:
            print(f"📰 {len(latest_news)}개 최신 뉴스 조회 성공:")
            for i, news in enumerate(latest_news, 1):
                print(f"\n뉴스 {i}: {news.get('title', 'N/A')}")
                print(f"카테고리: {news.get('category', 'N/A')}")
        else:
            print("❌ 최신 뉴스 조회 실패")
        
        # 뉴스 요약 테스트 (간단한 버전만 테스트)
        print("\n7. RAG 서비스로 뉴스 요약 생성")
        summary_result = rag_service.summarize_news("ChatGPT", max_results=2, summary_type="brief")
        
        if summary_result and summary_result.get('summary'):
            print("📝 뉴스 요약 성공:")
            print(f"요약 내용: {summary_result['summary']}")
            print(f"분석된 기사 수: {summary_result.get('total_articles', 0)}")
        else:
            print("❌ 뉴스 요약 실패")
            if summary_result:
                print(f"오류: {summary_result.get('summary', 'Unknown error')}")
        
    except Exception as e:
        print(f"❌ RAG 서비스 테스트 중 오류: {e}")
        logger.error(f"RAG 서비스 테스트 오류: {e}")
    """

def main():
    """메인 테스트 함수"""
    print("🚀 뉴스 기능 통합 테스트 시작")
    print(f"Python path: {sys.path[0]}")
    
    # 환경 변수 확인
    tavily_key = os.getenv('TAVILY_API_KEY')
    if not tavily_key:
        print("❌ TAVILY_API_KEY가 설정되지 않았습니다!")
        print("📝 .env 파일에 TAVILY_API_KEY=your_api_key 를 추가해주세요.")
        return
    else:
        print(f"✅ TAVILY_API_KEY 확인됨: {tavily_key[:10]}...")
    
    try:
        # 1. Tavily 뉴스 검색 테스트들
        test_tavily_news_search()
        test_category_news_search()
        test_latest_news()
        test_tavily_summary()
        
        # 2. RAG 서비스 뉴스 기능 테스트 (더 무거운 작업이므로 마지막에)
        print("\n" + "="*50)
        print("⚠️  RAG 서비스 테스트는 모델 로딩으로 시간이 오래 걸립니다.")
        user_input = input("RAG 서비스 테스트를 진행하시겠습니까? (y/N): ").lower()
        
        if user_input == 'y':
            test_rag_news_features()
        else:
            print("RAG 서비스 테스트를 건너뜁니다.")
        
        print("\n" + "="*50)
        print("🎉 테스트 완료!")
        print("="*50)
        
    except KeyboardInterrupt:
        print("\n⏹️  사용자에 의해 테스트가 중단되었습니다.")
    except Exception as e:
        print(f"\n❌ 테스트 중 예상치 못한 오류 발생: {e}")
        logger.error(f"테스트 오류: {e}")

if __name__ == "__main__":
    main()