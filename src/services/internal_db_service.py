import os
import sys
import asyncio
import logging
from typing import List, Dict, Any, Optional
import pandas as pd
from dataclasses import dataclass

# LangChain imports
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document

# Database imports
from sqlalchemy.ext.asyncio import create_async_engine, AsyncEngine
from sqlalchemy import text
import sqlite3

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 설정 로더 임포트
from utils.config_loader import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass 
class InternalDBConfig:
    # 설정 파일에서 값들을 읽어옴
    def __init__(self):
        internal_db_config = config.internal_db_rag_config
        self.chunk_size: int = internal_db_config['chunk_size']
        self.chunk_overlap: int = internal_db_config['chunk_overlap']
        self.top_k: int = internal_db_config['top_k']
        self.score_margin: float = internal_db_config['score_margin']
        self.max_context_chars: int = internal_db_config['max_context_chars']
        self.max_new_tokens: int = internal_db_config['max_new_tokens']
        self.temperature: float = internal_db_config['temperature']

class InternalDBService:
    def __init__(self, llm_handler, embedding_handler):
        self.llm_handler = llm_handler
        self.embedding_handler = embedding_handler
        self.config = InternalDBConfig()
        self.faiss_root = config.faiss_db_path
        self.faiss_cache = {}  # 메모리 캐시
        
        # FAISS 디렉토리 생성
        os.makedirs(self.faiss_root, exist_ok=True)
        
        # 데이터베이스 설정
        self.db_host = os.getenv("DB_HOST", "127.0.0.1")
        self.db_port = int(os.getenv("DB_PORT", "53301"))
        self.db_user = os.getenv("DB_USER", "manager")
        self.db_pass = os.getenv("DB_PASS", "SqlDba-1")
        self.db_name = os.getenv("DB_NAME", "sql_db")
        
        self.async_db_url = f"mysql+aiomysql://{self.db_user}:{self.db_pass}@{self.db_host}:{self.db_port}/{self.db_name}?charset=utf8mb4"
        
        logger.info(f"InternalDBService 초기화 완료 - FAISS Root: {self.faiss_root}")

    async def get_db_tables(self) -> List[str]:
        """데이터베이스 테이블 목록을 조회합니다"""
        try:
            engine = create_async_engine(self.async_db_url)
            async with engine.begin() as conn:
                result = await conn.execute(text("SHOW TABLES"))
                tables = [row[0] for row in result.fetchall()]
            await engine.dispose()
            return tables
        except Exception as e:
            logger.warning(f"MariaDB 연결 실패, 시뮬레이션 모드 사용: {e}")
            return ["knowledge"]  # 시뮬레이션용 기본 테이블

    async def ingest_table(self, table_name: str, save_name: str, simulate: bool = False, 
                          id_col: str = None, title_col: str = None, text_cols: List[str] = None) -> Dict[str, Any]:
        """데이터베이스 테이블을 벡터화하여 FAISS 인덱스를 생성합니다"""
        try:
            logger.info(f"테이블 '{table_name}' 인제스트 시작 (simulate: {simulate})")
            
            if simulate:
                # SQLite 시뮬레이션 모드
                df = self._create_simulation_data()
                schema = {
                    "id_col": "id",
                    "title_col": "term", 
                    "text_cols": ["description", "role", "details"],
                    "all_cols": ["id", "term", "description", "role", "details", "updated_at"]
                }
            else:
                # 실제 MariaDB 연결
                df, schema = await self._load_table_data(table_name, id_col, title_col, text_cols)
            
            if df.empty:
                raise ValueError(f"테이블 '{table_name}'에서 데이터를 찾을 수 없습니다")
            
            # 문서 생성
            documents = self._create_documents_from_dataframe(df, schema)
            
            # 텍스트 분할
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.config.chunk_size,
                chunk_overlap=self.config.chunk_overlap
            )
            chunks = text_splitter.split_documents(documents)
            
            # FAISS 인덱스 생성 및 저장
            save_path = os.path.join(self.faiss_root, save_name)
            vectorstore = FAISS.from_documents(chunks, self.embedding_handler.embeddings)
            vectorstore.save_local(save_path)
            
            # 메모리 캐시에 저장
            self.faiss_cache[save_name] = vectorstore
            
            logger.info(f"FAISS 인덱스 저장 완료: {save_path}")
            
            return {
                "save_dir": save_path,
                "rows": len(df),
                "chunks": len(chunks),
                "schema": schema
            }
            
        except Exception as e:
            logger.error(f"테이블 인제스트 오류: {e}")
            raise

    async def query(self, save_name: str, question: str, top_k: int = 5, margin: float = 0.12) -> Dict[str, Any]:
        """FAISS 인덱스를 사용하여 질의응답을 수행합니다"""
        try:
            logger.info(f"질의 시작: '{question}' (save_name: {save_name})")
            
            # FAISS 인덱스 로드
            vectorstore = await self._load_faiss_index(save_name)
            
            # 유사도 검색
            docs_with_scores = vectorstore.similarity_search_with_score(question, k=top_k)
            
            # 마진 필터링 (1등 점수 대비 margin 이내의 결과만 유지)
            if docs_with_scores:
                best_score = docs_with_scores[0][1]
                filtered_docs = [
                    (doc, score) for doc, score in docs_with_scores 
                    if score <= best_score + margin
                ]
            else:
                filtered_docs = []
            
            if not filtered_docs:
                return {
                    "answer": "관련 정보를 찾을 수 없습니다. 다른 키워드로 시도해보세요.",
                    "sources": []
                }
            
            # 컨텍스트 구성
            context_parts = []
            sources = []
            
            for i, (doc, score) in enumerate(filtered_docs):
                marker = f"S{i+1}"
                context_parts.append(f"《{marker}》 {doc.page_content}")
                
                sources.append({
                    "marker": marker,
                    "id": str(doc.metadata.get("id", "Unknown")),
                    "title": str(doc.metadata.get("title", "Unknown")),
                    "content": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content,
                    "score": float(score)
                })
            
            context = "\n".join(context_parts[:self.config.max_context_chars])
            
            # LLM 답변 생성
            prompt_template = """다음 컨텍스트를 참고하여 질문에 답변해주세요. 답변은 한국어로 최소 2문장 이상 작성하세요.

컨텍스트:
{context}

질문: {question}

답변:"""
            
            full_prompt = prompt_template.format(context=context, question=question)
            
            # LLM 호출 (동기 방식을 비동기로 래핑)
            answer = await asyncio.to_thread(
                self.llm_handler.generate, 
                full_prompt, 
                max_length=self.config.max_new_tokens,
                stream=False
            )
            
            logger.info(f"질의 완료: {len(filtered_docs)}개 문서 참조")
            
            return {
                "answer": answer,
                "sources": sources
            }
            
        except Exception as e:
            logger.error(f"질의 오류: {e}")
            raise

    async def get_status(self) -> Dict[str, Any]:
        """FAISS 인덱스 상태를 확인합니다"""
        try:
            # 디스크에 저장된 인덱스 목록
            faiss_indices = []
            if os.path.exists(self.faiss_root):
                for item in os.listdir(self.faiss_root):
                    index_path = os.path.join(self.faiss_root, item)
                    if os.path.isdir(index_path) and os.path.exists(os.path.join(index_path, "index.faiss")):
                        faiss_indices.append(item)
            
            # 메모리에 로딩된 인덱스 목록
            cache_keys = list(self.faiss_cache.keys())
            
            return {
                "faiss_indices": faiss_indices,
                "cache_keys": cache_keys
            }
            
        except Exception as e:
            logger.error(f"상태 조회 오류: {e}")
            raise

    async def _load_table_data(self, table_name: str, id_col: str = None, 
                             title_col: str = None, text_cols: List[str] = None) -> tuple:
        """MariaDB에서 테이블 데이터를 로드합니다"""
        try:
            engine = create_async_engine(self.async_db_url)
            
            async with engine.begin() as conn:
                # 테이블 스키마 조회
                columns_result = await conn.execute(text(f"DESCRIBE {table_name}"))
                columns = [row[0] for row in columns_result.fetchall()]
                
                # 데이터 조회
                data_result = await conn.execute(text(f"SELECT * FROM {table_name}"))
                rows = data_result.fetchall()
                
            await engine.dispose()
            
            # DataFrame 생성
            df = pd.DataFrame(rows, columns=columns)
            
            # 스키마 추론
            schema = self._infer_schema(columns, id_col, title_col, text_cols)
            
            return df, schema
            
        except Exception as e:
            logger.error(f"MariaDB 테이블 로드 오류: {e}")
            raise

    def _create_simulation_data(self) -> pd.DataFrame:
        """시뮬레이션용 데이터를 생성합니다"""
        data = [
            {
                "id": 1,
                "term": "어텐션 메커니즘",
                "description": "어텐션은 입력의 중요한 부분에 가중치를 부여해 정보를 통합하는 기법이다. 시퀀스 처리에서 문맥 의존성을 강화한다.",
                "role": "입력 토큰 간 상호연관성을 계산하며 정보 흐름을 개선한다.",
                "details": "Transformer의 핵심 구성요소로 번역·요약 등에서 성능을 끌어올린다.",
                "updated_at": "2024-01-01 00:00:00"
            },
            {
                "id": 2,
                "term": "Self-Attention",
                "description": "Self-Attention은 동일 시퀀스 내 토큰들이 서로를 참조하여 가중합을 구한다. RNN의 순차 의존성을 줄여 병렬화를 가능케 한다.",
                "role": "장기 의존성 문제를 완화하고 각 토큰의 전역 문맥 파악을 돕는다.",
                "details": "멀티헤드로 다양한 표현 공간에서 주의를 분산해 학습을 안정화한다.",
                "updated_at": "2024-01-01 00:00:00"
            },
            {
                "id": 3,
                "term": "FAISS",
                "description": "FAISS는 대규모 벡터에 대한 빠른 유사도 검색을 제공한다. 근사 최근접 탐색을 지원한다.",
                "role": "대규모 임베딩 인덱싱과 빠른 검색을 제공한다.",
                "details": "Facebook AI Research에서 개발되었고 CPU/GPU 백엔드를 제공한다.",
                "updated_at": "2024-01-01 00:00:00"
            }
        ]
        
        return pd.DataFrame(data)

    def _infer_schema(self, columns: List[str], id_col: str = None, 
                     title_col: str = None, text_cols: List[str] = None) -> Dict[str, Any]:
        """테이블 스키마를 추론합니다"""
        schema = {
            "id_col": id_col or "id",
            "title_col": title_col or "term",
            "text_cols": text_cols or ["description", "role", "details"],
            "all_cols": columns
        }
        
        # 컬럼 존재 여부 확인 및 조정
        if schema["id_col"] not in columns:
            schema["id_col"] = columns[0] if columns else "id"
            
        if schema["title_col"] not in columns:
            # 제목에 해당할 만한 컬럼 찾기
            title_candidates = [col for col in columns if any(keyword in col.lower() for keyword in ["title", "name", "term"])]
            schema["title_col"] = title_candidates[0] if title_candidates else columns[0]
            
        # 텍스트 컬럼 확인
        valid_text_cols = [col for col in schema["text_cols"] if col in columns]
        if not valid_text_cols:
            # 텍스트성 컬럼 찾기
            text_candidates = [col for col in columns if col not in [schema["id_col"], schema["title_col"]]]
            schema["text_cols"] = text_candidates[:3] if text_candidates else ["description"]
        else:
            schema["text_cols"] = valid_text_cols
            
        return schema

    def _create_documents_from_dataframe(self, df: pd.DataFrame, schema: Dict[str, Any]) -> List[Document]:
        """DataFrame에서 LangChain Document 객체를 생성합니다"""
        documents = []
        
        for _, row in df.iterrows():
            # 텍스트 내용 구성
            text_parts = []
            title = str(row.get(schema["title_col"], "Unknown"))
            
            text_parts.append(f"제목: {title}")
            
            for col in schema["text_cols"]:
                if col in row and pd.notna(row[col]):
                    content = str(row[col]).strip()
                    if content:
                        text_parts.append(f"{col}: {content}")
            
            full_text = "\n".join(text_parts)
            
            # 메타데이터 구성
            metadata = {
                "id": row.get(schema["id_col"], "Unknown"),
                "title": title,
                "source": "internal_db"
            }
            
            documents.append(Document(page_content=full_text, metadata=metadata))
        
        return documents

    async def _load_faiss_index(self, save_name: str):
        """FAISS 인덱스를 로드합니다 (캐시 우선, 없으면 디스크에서 로드)"""
        # 메모리 캐시 확인
        if save_name in self.faiss_cache:
            logger.info(f"FAISS 인덱스 메모리 캐시에서 로드: {save_name}")
            return self.faiss_cache[save_name]
        
        # 디스크에서 로드
        save_path = os.path.join(self.faiss_root, save_name)
        if os.path.exists(save_path) and os.path.exists(os.path.join(save_path, "index.faiss")):
            logger.info(f"FAISS 인덱스 디스크에서 로드: {save_path}")
            vectorstore = FAISS.load_local(
                save_path, 
                self.embedding_handler.embeddings,
                allow_dangerous_deserialization=True
            )
            # 캐시에 저장
            self.faiss_cache[save_name] = vectorstore
            return vectorstore
        else:
            raise FileNotFoundError(f"FAISS 인덱스를 찾을 수 없습니다: {save_path}")