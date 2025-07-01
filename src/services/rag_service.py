from langchain.schema.runnable import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import WikipediaLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
import uuid
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
            
            # Load documents from Wikipedia
            examples = ['챗GPT', '인공지능', '트랜스포머_(기계_학습)', '딥러닝', '머신러닝']
            docs = []
            for query in examples:
                try:
                    loader = WikipediaLoader(query=query, lang='ko', load_max_docs=1, doc_content_chars_max=1000)
                    docs += loader.load()
                except Exception as e:
                    logger.warning(f"Failed to load Wikipedia document for {query}: {e}")

            if not docs:
                logger.error("No documents were loaded!")
                return

            logger.info(f"Loaded {len(docs)} documents")

            # Split documents into chunks
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=80)
            chunks = text_splitter.split_documents(docs)
            logger.info(f"Created {len(chunks)} chunks")

            # Create vector database
            random_dir = f"./RAG_db_{str(uuid.uuid4())[:8]}"
            logger.info(f"Creating vector database in: {random_dir}")

            self.db = Chroma.from_documents(
                documents=chunks,
                embedding=self.embedding_handler.embeddings,
                persist_directory=random_dir,
                collection_metadata={'hnsw:space': 'l2'}
            )

            # Create retriever
            self.retriever = self.db.as_retriever(search_kwargs={'k': 3})

            # Setup RAG chain
            self._setup_rag_chain()
            
            logger.info("RAG service initialized successfully!")
            
        except Exception as e:
            logger.error(f"Error initializing RAG service: {e}")
            raise

    def _setup_rag_chain(self):
        try:
            # RAG prompt
            rag_prompt = ChatPromptTemplate.from_messages([
                ('system', '다음 Context를 사용하여 Question에 답변해주세요. 한국어로 답변해주세요.'),
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
        return "\n---\n".join('주제: ' + doc.metadata.get('title', 'Unknown') + '\n' + doc.page_content for doc in docs)

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
            return [{"title": doc.metadata.get('title', 'Unknown'), "content": doc.page_content} for doc in docs]
            
        except Exception as e:
            logger.error(f"Error retrieving documents: {e}")
            return []