from langchain.schema.runnable import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
import logging
import os
from utils.helpers import search_news, create_documents_from_news

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- 상수 정의 ---
DB_PERSIST_DIRECTORY = "./data/vector_db"

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

            # 3. 텍스트 분할
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=80)
            chunks = text_splitter.split_documents(documents)
            logger.info(f"Created {len(chunks)} chunks from {len(documents)} documents.")

            # 4. DB에 추가
            if chunks:
                self.db.add_documents(chunks)
                self.db.persist() # 변경사항 저장
                logger.info(f"Successfully added {len(chunks)} new chunks to the vector database.")
                return len(chunks), f"Successfully added {len(chunks)} new chunks to the database."
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
            return response
            
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
