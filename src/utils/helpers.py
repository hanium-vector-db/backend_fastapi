def format_docs(docs):
    return "\n---\n".join('Title: ' + doc.metadata.get('title', 'No Title') + '\n' + doc.page_content for doc in docs)

def validate_input(data):
    if not isinstance(data, dict):
        raise ValueError("Input must be a dictionary.")
    return True

def extract_query_params(query):
    return {key: value for key, value in query.items() if value is not None}

# --- 뉴스 검색 및 스크래핑 기능 추가 ---
import requests
from duckduckgo_search import DDGS
from langchain.docstore.document import Document
import logging

logger = logging.getLogger(__name__)

def search_news(query: str, max_results: int = 5):
    """
    DuckDuckGo를 사용하여 최신 뉴스를 검색하고 URL 목록을 반환합니다.
    """
    logger.info(f"'{query}'에 대한 뉴스 검색 중...")
    try:
        with DDGS() as ddgs:
            results = list(ddgs.news(
                keywords=query,
                region='wt-wt',
                safesearch='off',
                timelimit='d',  # 최근 1일 뉴스
                max_results=max_results
            ))
        logger.info(f"{len(results)}개의 뉴스 기사를 찾았습니다.")
        return results
    except Exception as e:
        logger.error(f"뉴스 검색 중 오류 발생: {e}")
        return []

def scrape_web_page(url: str):
    """
    주어진 URL의 웹 페이지 내용을 스크래핑하여 텍스트를 반환합니다.
    """
    logger.info(f"'{url}' 페이지 스크래핑 중...")
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
            logger.warning("BeautifulSoup4가 설치되지 않았습니다. 기본적인 텍스트 추출을 시도합니다.")
            # 간단한 텍스트 추출 로직 (정확도 낮음)
            return response.text

        # BeautifulSoup으로 의미있는 텍스트를 찾지 못한 경우 대비
        return response.text

    except requests.exceptions.RequestException as e:
        logger.error(f"'{url}' 페이지 스크래핑 중 오류 발생: {e}")
        return None

def create_documents_from_news(news_results: list):
    """
    뉴스 검색 결과를 LangChain Document 객체 리스트로 변환합니다.
    """
    docs = []
    for news in news_results:
        content = scrape_web_page(news['url'])
        if content:
            doc = Document(
                page_content=content,
                metadata={'title': news['title'], 'source': news['url'], 'date': news['date']}
            )
            docs.append(doc)
    logger.info(f"성공적으로 {len(docs)}개의 문서를 생성했습니다.")
    return docs
