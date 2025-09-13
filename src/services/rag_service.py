from langchain.schema.runnable import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
import logging
import os
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.helpers import search_news, create_documents_from_news, search_latest_news, get_news_summary_with_tavily
from utils.config_loader import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- 설정에서 읽어온 상수들 ---
DB_PERSIST_DIRECTORY = config.vector_db_path

class RAGService:
    def __init__(self, llm_handler, embedding_handler):
        self.llm_handler = llm_handler
        self.embedding_handler = embedding_handler
        self.db = None
        self.retriever = None
        self.rag_chain = None
        self._initialize_rag()

    def _initialize_rag(self):
        try:
            logger.info("Initializing RAG service...")
            
            # 데이터베이스 디렉토리가 없으면 생성
            if not os.path.exists(DB_PERSIST_DIRECTORY):
                os.makedirs(DB_PERSIST_DIRECTORY)
                logger.info(f"Created vector database directory: {DB_PERSIST_DIRECTORY}")

            # 기존 DB를 로드하거나 새로 생성
            self.db = Chroma(
                persist_directory=DB_PERSIST_DIRECTORY,
                embedding_function=self.embedding_handler.embeddings,
                collection_metadata={'hnsw:space': 'l2'}
            )
            
            logger.info(f"Vector database loaded/initialized from: {DB_PERSIST_DIRECTORY}")
            logger.info(f"Current document count: {self.db._collection.count()}")

            # Create retriever
            self.retriever = self.db.as_retriever(search_kwargs={'k': 3})

            # Setup RAG chain
            self._setup_rag_chain()
            
            logger.info("RAG service initialized successfully!")
            
        except Exception as e:
            logger.error(f"Error initializing RAG service: {e}")
            raise

    def add_documents_from_web(self, query: str, max_results: int = 5):
        """
        웹에서 뉴스를 검색하고 해당 내용을 Vector DB에 추가합니다.
        """
        try:
            logger.info(f"Starting to add documents from web for query: '{query}'")
            # 1. 뉴스 검색
            news_results = search_news(query, max_results)
            if not news_results:
                return 0, "No news articles found."

            # 2. 문서 생성 (스크래핑)
            documents = create_documents_from_news(news_results)
            if not documents:
                return 0, "Failed to create documents from news articles."

            # 3. 텍스트 분할 (설정에서 읽어옴)
            external_web_config = config.external_web_rag_config
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=external_web_config['chunk_size'], 
                chunk_overlap=external_web_config['chunk_overlap']
            )
            chunks = text_splitter.split_documents(documents)
            logger.info(f"Created {len(chunks)} chunks from {len(documents)} documents.")

            # 4. DB에 추가 (persist 에러 방지)
            if chunks:
                try:
                    self.db.add_documents(chunks)
                    logger.info(f"Successfully added {len(chunks)} new chunks to the vector database.")
                    return len(chunks), f"Successfully added {len(chunks)} new chunks to the database."
                except Exception as persist_error:
                    logger.warning(f"Persist error (ignoring): {persist_error}")
                    # persist 실패해도 메모리에는 추가되었으므로 성공으로 처리
                    return len(chunks), f"Successfully added {len(chunks)} new chunks to the database (in-memory)."
            else:
                return 0, "No processable content found in the articles."
        except Exception as e:
            logger.error(f"Error adding documents from web: {e}")
            return 0, f"An error occurred: {str(e)}"

    def _setup_rag_chain(self):
        try:
            # RAG prompt
            rag_prompt = ChatPromptTemplate.from_messages([
                ('system', '다음 Context를 사용하여 Question에 답변해주세요. 만약 Context에 정보가 없다면, 아는대로 답변해주세요. 항상 한국어로 답변해주세요.'),
                ('user', 'Context: {context}\n---\nQuestion: {question}')
            ])

            # Create the RAG chain
            self.rag_chain = (
                {"context": self.retriever | self.format_docs, "question": RunnablePassthrough()}
                | rag_prompt
                | self.llm_handler.chat_model
                | StrOutputParser()
            )
            
        except Exception as e:
            logger.error(f"Error setting up RAG chain: {e}")
            raise

    def format_docs(self, docs):
        return "\n---\n".join(f"출처: {doc.metadata.get('source', 'Unknown')}\n제목: {doc.metadata.get('title', 'Unknown')}\n내용: {doc.page_content}" for doc in docs)

    def generate_response(self, query: str) -> str:
        try:
            if self.rag_chain is None:
                return "RAG service not initialized"
            
            response = self.rag_chain.invoke(query)
            
            # Handle different response types
            if hasattr(response, 'content'):
                return response.content
            elif isinstance(response, str):
                return response
            else:
                return str(response)
            
        except Exception as e:
            logger.error(f"Error generating RAG response: {e}")
            return f"Error: {str(e)}"

    def get_relevant_documents(self, query: str, k: int = 3):
        try:
            if self.retriever is None:
                return []
            
            docs = self.retriever.invoke(query)
            return [{"title": doc.metadata.get('title', 'Unknown'), "content": doc.page_content, "source": doc.metadata.get('source', 'Unknown')}
 for doc in docs]
            
        except Exception as e:
            logger.error(f"Error retrieving documents: {e}")
            return []

    # === 새로운 뉴스 관련 기능들 ===
    
    def get_latest_news(self, categories: list = None, max_results: int = 10, time_range: str = 'd'):
        """
        최신 뉴스를 조회합니다 (데이터베이스에 저장하지 않고 조회만)
        """
        try:
            logger.info(f"최신 뉴스 조회 중... 카테고리: {categories}, 결과수: {max_results}")
            news_results = search_latest_news(max_results=max_results, categories=categories, time_range=time_range)
            
            formatted_news = []
            for news in news_results:
                formatted_news.append({
                    'title': news.get('title', ''),
                    'url': news.get('url', ''),
                    'content': news.get('content', ''),
                    'category': news.get('category', 'general'),
                    'published_date': news.get('published_date', ''),
                    'score': news.get('score', 0)
                })
            
            return formatted_news
            
        except Exception as e:
            logger.error(f"최신 뉴스 조회 오류: {e}")
            return []
    
    def summarize_news(self, query: str, max_results: int = 5, summary_type: str = "comprehensive"):
        """
        특정 주제의 뉴스를 검색하고 LLM을 이용해 요약합니다
        
        Args:
            query: 검색할 주제
            max_results: 분석할 뉴스 개수
            summary_type: 요약 타입 ("brief", "comprehensive", "analysis")
        """
        try:
            logger.info(f"'{query}' 주제 뉴스 요약 생성 중...")
            
            # Tavily로 뉴스 검색 및 기본 요약 획득
            news_data = get_news_summary_with_tavily(query, max_results)
            
            if not news_data:
                return {
                    "summary": f"'{query}' 관련 최신 뉴스를 찾을 수 없습니다.",
                    "articles": [],
                    "keywords": [],
                    "sentiment": "neutral"
                }
            
            # 요약 타입별 프롬프트 선택
            summary_prompts = {
                "brief": self._get_brief_summary_prompt(),
                "comprehensive": self._get_comprehensive_summary_prompt(),
                "analysis": self._get_analysis_summary_prompt()
            }
            
            prompt_template = summary_prompts.get(summary_type, summary_prompts["comprehensive"])
            
            # 뉴스 데이터 준비
            articles_text = "\n\n".join([
                f"제목: {article.get('title', '')}\n내용: {article.get('content', '')[:1000]}"
                for article in news_data[:max_results]
                if not article.get('is_summary', False)  # Tavily의 자동 요약 제외
            ])
            
            # LLM으로 요약 생성
            full_prompt = prompt_template.format(query=query, articles=articles_text)
            summary_response = self.llm_handler.chat_model.invoke(full_prompt)
            
            return {
                "summary": summary_response.content if hasattr(summary_response, 'content') else str(summary_response),
                "articles": news_data[:max_results],
                "query": query,
                "summary_type": summary_type,
                "total_articles": len(news_data)
            }
            
        except Exception as e:
            logger.error(f"뉴스 요약 생성 오류: {e}")
            return {
                "summary": f"뉴스 요약 생성 중 오류가 발생했습니다: {str(e)}",
                "articles": [],
                "keywords": [],
                "sentiment": "neutral"
            }
    
    def analyze_news_trends(self, categories: list = None, max_results: int = 20, time_range: str = 'd'):
        """
        여러 카테고리의 뉴스를 분석하여 트렌드를 파악합니다
        """
        try:
            logger.info(f"뉴스 트렌드 분석 시작... 카테고리: {categories}")
            
            if categories is None:
                categories = ['politics', 'economy', 'technology', 'society']
            
            # 카테고리별 뉴스 수집
            all_news = []
            category_summaries = {}
            
            for category in categories:
                category_news = search_news(
                    "최신 뉴스", 
                    max_results=max_results//len(categories), 
                    category=category,
                    time_range=time_range
                )
                
                if category_news:
                    all_news.extend(category_news)
                    
                    # 카테고리별 간단 요약
                    category_text = "\n".join([
                        f"• {news.get('title', '')}: {news.get('content', '')[:200]}"
                        for news in category_news[:3]
                    ])
                    
                    category_prompt = f"다음 {category} 카테고리 뉴스들의 주요 트렌드를 한 문장으로 요약해주세요:\n{category_text}"
                    category_summary = self.llm_handler.chat_model.invoke(category_prompt)
                    category_summaries[category] = category_summary.content if hasattr(category_summary, 'content') else str(category_summary)
            
            # 전체 트렌드 분석
            trend_analysis_prompt = self._get_trend_analysis_prompt()
            all_titles = [news.get('title', '') for news in all_news]
            titles_text = "\n".join([f"• {title}" for title in all_titles[:30]])
            
            full_trend_prompt = trend_analysis_prompt.format(
                titles=titles_text,
                category_summaries="\n".join([f"{cat}: {summary}" for cat, summary in category_summaries.items()])
            )
            
            trend_response = self.llm_handler.chat_model.invoke(full_trend_prompt)
            
            return {
                "overall_trend": trend_response.content if hasattr(trend_response, 'content') else str(trend_response),
                "category_trends": category_summaries,
                "total_articles_analyzed": len(all_news),
                "categories": categories,
                "time_range": time_range
            }
            
        except Exception as e:
            logger.error(f"뉴스 트렌드 분석 오류: {e}")
            return {
                "overall_trend": f"트렌드 분석 중 오류가 발생했습니다: {str(e)}",
                "category_trends": {},
                "total_articles_analyzed": 0
            }

    # === 프롬프트 템플릿들 ===
    
    def _get_brief_summary_prompt(self):
        """간단 요약용 프롬프트"""
        return """다음 뉴스 기사들을 바탕으로 '{query}' 주제에 대한 간단한 요약을 작성해주세요.

뉴스 기사들:
{articles}

요구사항:
1. 핵심 내용을 2-3문장으로 간단히 요약
2. 가장 중요한 포인트만 포함
3. 명확하고 이해하기 쉽게 작성
4. 한국어로 작성

간단 요약:"""

    def _get_comprehensive_summary_prompt(self):
        """포괄적 요약용 프롬프트"""
        return """다음 뉴스 기사들을 바탕으로 '{query}' 주제에 대한 포괄적인 요약을 작성해주세요.

뉴스 기사들:
{articles}

다음 형식으로 작성해주세요:

## 📰 주요 내용 요약
(핵심 내용을 3-4문장으로 요약)

## 🔍 세부 분석
• 주요 이슈: 
• 관련 인물/기관:
• 영향/결과:

## 🏷️ 키워드
(관련 키워드 3-5개를 쉼표로 구분)

## 📊 종합 평가
(전반적인 상황 평가와 향후 전망 1-2문장)

모든 내용을 한국어로 작성해주세요."""

    def _get_analysis_summary_prompt(self):
        """분석 중심 요약용 프롬프트"""
        return """다음 뉴스 기사들을 바탕으로 '{query}' 주제에 대한 심층 분석을 작성해주세요.

뉴스 기사들:
{articles}

다음 형식으로 분석해주세요:

## 🎯 핵심 이슈 분석
(가장 중요한 이슈와 그 배경)

## 📈 현황 및 트렌드
• 현재 상황:
• 변화 추이:
• 주목할 점:

## ⚡ 주요 동향
• 긍정적 요소:
• 우려사항:
• 예상 시나리오:

## 🌟 시사점 및 전망
(이 뉴스가 갖는 의미와 향후 예상되는 발전 방향)

## 🏷️ 핵심 키워드
(분석에 중요한 키워드 5-7개)

전문적이고 객관적인 시각으로 한국어로 작성해주세요."""

    def _get_trend_analysis_prompt(self):
        """트렌드 분석용 프롬프트"""
        return """다음 뉴스 제목들과 카테고리별 요약을 바탕으로 현재 뉴스 트렌드를 분석해주세요.

뉴스 제목들:
{titles}

카테고리별 요약:
{category_summaries}

다음 형식으로 트렌드를 분석해주세요:

## 🔥 오늘의 주요 트렌드
(가장 주목받는 이슈 2-3개)

## 📊 분야별 동향
• 정치: (정치 관련 주요 이슈)
• 경제: (경제 관련 주요 이슈)  
• 사회: (사회 관련 주요 이슈)
• 기술: (기술 관련 주요 이슈)

## 🎭 여론 및 관심도
(국민들이 가장 관심 갖는 이슈들과 여론의 방향)

## 🔮 주목할 포인트
(앞으로 계속 주목해야 할 이슈들)

객관적이고 균형잡힌 시각으로 한국어로 작성해주세요."""
