def format_docs(docs):
    return "\n---\n".join('Title: ' + doc.metadata.get('title', 'No Title') + '\n' + doc.page_content for doc in docs)

def validate_input(data):
    if not isinstance(data, dict):
        raise ValueError("Input must be a dictionary.")
    return True

def extract_query_params(query):
    return {key: value for key, value in query.items() if value is not None}

# --- 뉴스 검색 기능 (Tavily 사용) ---
import requests
from tavily import TavilyClient
from langchain.docstore.document import Document
import logging
import os
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

# Tavily API 키 확인
TAVILY_API_KEY = os.getenv('TAVILY_API_KEY')
if not TAVILY_API_KEY:
    logger.warning("TAVILY_API_KEY가 설정되지 않았습니다. .env 파일에 추가해주세요.")
    tavily_client = None
else:
    tavily_client = TavilyClient(api_key=TAVILY_API_KEY)

def search_news(query: str, max_results: int = 5, category: str = None, time_range: str = 'd'):
    """
    Tavily를 사용하여 최신 뉴스를 검색하고 구조화된 결과를 반환합니다.
    
    Args:
        query: 검색 키워드
        max_results: 최대 결과 수
        category: 뉴스 카테고리 (politics, economy, technology, sports, health 등)
        time_range: 시간 범위 ('d'=1일, 'w'=1주, 'm'=1달) - 현재는 Tavily에서 지원하지 않음
    """
    if not tavily_client:
        logger.error("Tavily API 키가 설정되지 않았습니다.")
        return []
    
    # 카테고리 키워드 매핑
    category_keywords = {
        'politics': '정치',
        'economy': '경제',
        'technology': '기술 IT',
        'sports': '스포츠',
        'health': '건강 의료',
        'culture': '문화 예술',
        'society': '사회',
        'international': '국제 해외'
    }
    
    # 카테고리가 지정된 경우 검색어에 추가
    search_query = query
    if category and category in category_keywords:
        search_query = f"{query} {category_keywords[category]}"
    
    logger.info(f"Tavily로 '{search_query}' 뉴스 검색 중...")
    
    try:
        # Tavily로 뉴스 검색
        response = tavily_client.search(
            query=search_query,
            search_depth="advanced",
            max_results=max_results,
            include_domains=["news.google.com", "naver.com", "daum.net", "yna.co.kr", "chosun.com"],
            topic="news"
        )
        
        results = []
        for item in response.get('results', []):
            result = {
                'title': item.get('title', ''),
                'url': item.get('url', ''),
                'content': item.get('content', ''),  # Tavily가 이미 추출한 콘텐츠
                'score': item.get('score', 0),
                'published_date': item.get('published_date', ''),
                'category': category or 'general',
                'search_query': search_query
            }
            results.append(result)
            
        logger.info(f"Tavily로 {len(results)}개의 뉴스 기사를 찾았습니다.")
        return results
        
    except Exception as e:
        logger.error(f"Tavily 뉴스 검색 중 오류 발생: {e}")
        return []

def search_latest_news(max_results: int = 10, categories: list = None, time_range: str = 'd'):
    """
    여러 카테고리의 최신 뉴스를 한번에 검색합니다.
    
    Args:
        max_results: 카테고리별 최대 결과 수
        categories: 검색할 카테고리 리스트
        time_range: 시간 범위
    """
    if categories is None:
        categories = ['politics', 'economy', 'technology', 'society']
    
    all_news = []
    for category in categories:
        news = search_news("최신 뉴스", max_results=max_results//len(categories), 
                          category=category, time_range=time_range)
        all_news.extend(news)
    
    # 날짜순으로 정렬 (최신순)
    try:
        all_news.sort(key=lambda x: x.get('date', ''), reverse=True)
    except:
        pass  # 날짜 정렬 실패 시 무시
    
    return all_news[:max_results]

def extract_content_with_tavily(url: str):
    """
    Tavily를 사용하여 특정 URL의 콘텐츠를 추출합니다.
    """
    if not tavily_client:
        logger.error("Tavily API 키가 설정되지 않았습니다.")
        return None
        
    logger.info(f"Tavily로 '{url}' 콘텐츠 추출 중...")
    
    try:
        # Tavily의 extract 기능 사용
        response = tavily_client.extract(urls=[url])
        
        if response and 'results' in response and len(response['results']) > 0:
            content = response['results'][0].get('content', '')
            return content
        else:
            logger.warning(f"Tavily로 '{url}'에서 콘텐츠를 추출할 수 없습니다.")
            return None
            
    except Exception as e:
        logger.error(f"Tavily 콘텐츠 추출 중 오류 발생: {e}")
        return None

def scrape_web_page(url: str):
    """
    웹 페이지 콘텐츠를 추출합니다. 먼저 Tavily를 시도하고 실패하면 기본 스크래핑을 사용합니다.
    """
    # 먼저 Tavily로 시도
    content = extract_content_with_tavily(url)
    if content:
        return content
    
    # Tavily 실패 시 기본 스크래핑 (백업)
    logger.info(f"기본 스크래핑으로 '{url}' 페이지 처리 중...")
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        # BeautifulSoup을 사용하여 본문 텍스트 추출 시도
        try:
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # 일반적인 기사 본문 태그들을 우선적으로 검색
            main_content = soup.find('article') or soup.find('main') or soup.find(id='content') or soup.body
            
            if main_content:
                # 불필요한 태그(스크립트, 스타일, 광고 등) 제거
                for tag in main_content(['script', 'style', 'nav', 'footer', 'aside', '.adsbygoogle']):
                    tag.decompose()
                
                text = ' '.join(p.get_text(strip=True) for p in main_content.find_all('p'))
                if len(text) > 100: # 최소한의 텍스트 길이가 확보될 경우
                    return text
        except ImportError:
            logger.warning("BeautifulSoup4가 설치되지 않았습니다.")
            
        return response.text[:5000]  # 텍스트 길이 제한

    except requests.exceptions.RequestException as e:
        logger.error(f"'{url}' 페이지 스크래핑 중 오류 발생: {e}")
        return None

def create_documents_from_news(news_results: list):
    """
    뉴스 검색 결과를 LangChain Document 객체 리스트로 변환합니다.
    Tavily 결과는 이미 콘텐츠가 추출되어 있으므로 직접 사용하고,
    필요시 추가 콘텐츠를 스크래핑합니다.
    """
    docs = []
    for news in news_results:
        # Tavily 결과에 이미 콘텐츠가 있는지 확인
        content = news.get('content', '')
        
        # 콘텐츠가 없거나 너무 짧으면 스크래핑 시도
        if not content or len(content) < 200:
            scraped_content = scrape_web_page(news['url'])
            if scraped_content:
                content = scraped_content
        
        if content and len(content) > 50:  # 최소한의 콘텐츠가 있을 때만 문서 생성
            doc = Document(
                page_content=content,
                metadata={
                    'title': news.get('title', ''),
                    'source': news.get('url', ''),
                    'date': news.get('published_date', news.get('date', '')),
                    'category': news.get('category', 'general'),
                    'score': news.get('score', 0)
                }
            )
            docs.append(doc)
    
    logger.info(f"성공적으로 {len(docs)}개의 문서를 생성했습니다.")
    return docs

def get_news_summary_with_tavily(query: str, max_results: int = 5):
    """
    Tavily를 사용하여 뉴스를 검색하고 바로 요약을 위한 구조화된 데이터를 반환합니다.
    """
    if not tavily_client:
        logger.error("Tavily API 키가 설정되지 않았습니다.")
        return []
    
    logger.info(f"Tavily로 뉴스 요약용 데이터 수집 중: '{query}'")
    
    try:
        # Tavily의 search 및 answer 기능 활용
        response = tavily_client.search(
            query=f"{query} 최신 뉴스",
            search_depth="advanced", 
            max_results=max_results,
            topic="news",
            include_answer=True,
            include_raw_content=False
        )
        
        results = []
        for item in response.get('results', []):
            result = {
                'title': item.get('title', ''),
                'url': item.get('url', ''),
                'content': item.get('content', ''),
                'score': item.get('score', 0),
                'published_date': item.get('published_date', '')
            }
            results.append(result)
        
        # Tavily의 AI 답변도 포함
        answer = response.get('answer', '')
        if answer:
            results.insert(0, {
                'title': f"{query} 관련 뉴스 요약",
                'url': '',
                'content': answer,
                'score': 1.0,
                'published_date': '',
                'is_summary': True
            })
        
        return results
        
    except Exception as e:
        logger.error(f"Tavily 뉴스 요약 데이터 수집 중 오류: {e}")
        return []
