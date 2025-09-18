import os
import sys
import asyncio
import logging
import time
import math
import re
import unicodedata
from typing import List, Dict, Any, Optional, Tuple
import pandas as pd
from dataclasses import dataclass
from collections import defaultdict
from functools import lru_cache

# LangChain imports
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document

# Database imports
from sqlalchemy.ext.asyncio import create_async_engine, AsyncEngine
from sqlalchemy import text, inspect as sa_inspect
from sqlalchemy.engine import Result
import sqlite3

# Transformers for model handling
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 설정 로더 임포트
from utils.config_loader import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
perf_log = logging.getLogger("perf.rag")

@dataclass
class EnhancedInternalDBConfig:
    """향상된 내부 DB RAG 설정"""
    def __init__(self):
        internal_db_config = config.internal_db_rag_config
        self.chunk_size: int = internal_db_config['chunk_size']
        self.chunk_overlap: int = internal_db_config['chunk_overlap']
        self.top_k: int = internal_db_config['top_k']
        self.score_margin: float = internal_db_config['score_margin']
        self.max_context_chars: int = internal_db_config['max_context_chars']
        self.max_new_tokens: int = internal_db_config['max_new_tokens']
        self.temperature: float = internal_db_config['temperature']
        # 고급 검색 설정 추가
        self.fetch_multiplier: int = 4    # 1차 검색 배수
        self.per_title_cap: int = 3       # 같은 제목에서 최대 청크 수
        self.sim_floor: float = 0.35      # 최소 유사도
        self.min_new_tokens: int = 48     # 최소 생성 토큰
        self.top_p: float = 0.9
        self.top_k_sampling: int = 50
        self.repetition_penalty: float = 1.0

class EnhancedInternalDBService:
    """향상된 내부 DB RAG 서비스 (내부-dbms 참조 코드 기반)"""

    def __init__(self, llm_handler, embedding_handler):
        self.llm_handler = llm_handler
        self.embedding_handler = embedding_handler
        self.config = EnhancedInternalDBConfig()
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

        # 연결 상태 확인
        self._connection_available = self._check_db_connection()

        logger.info(f"EnhancedInternalDBService 초기화 완료")
        logger.info(f"  - FAISS Root: {self.faiss_root}")
        logger.info(f"  - DB 연결: {'✅ 사용 가능' if self._connection_available else '❌ 시뮬레이션 모드'}")

    def _check_db_connection(self) -> bool:
        """데이터베이스 연결 가능 여부 확인 (실제 MariaDB 연결 테스트)"""
        try:
            import pymysql
            connection = pymysql.connect(
                host=self.db_host,
                port=self.db_port,
                user=self.db_user,
                password=self.db_pass,
                database=self.db_name,
                charset='utf8mb4',
                connect_timeout=3,
                read_timeout=3
            )
            # 실제 쿼리 테스트
            with connection.cursor() as cursor:
                cursor.execute("SELECT 1")
                cursor.fetchone()
            connection.close()
            logger.info("✅ MariaDB 연결 성공")
            return True
        except Exception as e:
            logger.warning(f"⚠️ MariaDB 연결 실패: {e} - 시뮬레이션 모드로 전환")
            return False

    async def get_db_tables(self, simulate: bool = None) -> Dict[str, Any]:
        """데이터베이스 테이블 목록을 조회합니다 (자동 fallback 지원)"""
        try:
            # simulate 파라미터가 None이면 연결 상태에 따라 자동 결정
            if simulate is None:
                simulate = not self._connection_available

            if simulate or not self._connection_available:
                return {
                    "ok": True,
                    "simulate": True,
                    "backend": "sqlite+aiosqlite (simulation)",
                    "tables": ["knowledge", "products", "users", "orders"],  # 시뮬레이션 테이블 추가
                    "views": [],
                    "message": "MariaDB 연결 불가로 시뮬레이션 모드 사용"
                }

            # 실제 MariaDB 연결 시도
            engine = create_async_engine(self.async_db_url)
            async with engine.connect() as conn:
                def _list_all(sync_conn):
                    insp = sa_inspect(sync_conn)
                    tables = insp.get_table_names()
                    views = getattr(insp, "get_view_names", lambda: [])()
                    return tables, views

                tables, views = await conn.run_sync(_list_all)

            await engine.dispose()

            return {
                "ok": True,
                "simulate": False,
                "backend": "mysql+aiomysql",
                "tables": tables,
                "views": views,
                "message": "MariaDB 연결 성공"
            }

        except Exception as e:
            logger.warning(f"MariaDB 연결 실패, 시뮬레이션 모드로 자동 전환: {e}")
            return {
                "ok": True,
                "simulate": True,
                "backend": "sqlite+aiosqlite (auto-fallback)",
                "tables": ["knowledge", "products", "users", "orders"],
                "views": [],
                "error": str(e),
                "message": "자동으로 시뮬레이션 모드로 전환됨"
            }

    async def ingest_table(self, table_name: str, save_name: str, simulate: bool = None,
                          id_col: str = None, title_col: str = None, text_cols: List[str] = None) -> Dict[str, Any]:
        """데이터베이스 테이블을 벡터화하여 FAISS 인덱스를 생성합니다 (향상된 버전)"""
        try:
            # simulate 파라미터가 None이면 연결 상태에 따라 자동 결정
            if simulate is None:
                simulate = not self._connection_available

            logger.info(f"향상된 테이블 '{table_name}' 인제스트 시작")
            logger.info(f"  - 시뮬레이션 모드: {simulate}")
            logger.info(f"  - DB 연결 상태: {'사용 가능' if self._connection_available else '불가능'}")

            if simulate or not self._connection_available:
                # SQLite 시뮬레이션 모드
                df = await asyncio.to_thread(self._create_enhanced_simulation_data)
                _id, _title, _texts, all_cols = self._infer_enhanced_schema(df.columns, id_col, title_col, text_cols)
            else:
                # 실제 MariaDB 연결
                engine = await asyncio.to_thread(self._make_async_engine, simulate)
                await self._ensure_connection(engine, simulate)
                _id, _title, _texts, all_cols = await self._infer_schema(engine, table_name, id_col, title_col, text_cols)
                df = await self._fetch_table(engine, table_name)
                await engine.dispose()

            if df.empty:
                raise ValueError(f"테이블 '{table_name}'에서 데이터를 찾을 수 없습니다")

            # 향상된 문서 생성 (확장된 텍스트 포함)
            documents = await asyncio.to_thread(
                self._to_enhanced_documents, df, _id, _title, _texts
            )

            # 텍스트 분할
            chunks = await asyncio.to_thread(self._split_documents, documents)

            # FAISS 인덱스 생성 및 저장
            save_path = os.path.join(self.faiss_root, save_name)
            vectorstore = await asyncio.to_thread(
                self._build_and_save_faiss, chunks, save_path
            )

            # 메모리 캐시에 저장
            self.faiss_cache[save_name] = vectorstore

            logger.info(f"향상된 FAISS 인덱스 저장 완료: {save_path}")

            return {
                "ok": True,
                "save_dir": save_path,
                "rows": len(df),
                "chunks": len(chunks),
                "schema": {
                    "id_col": _id,
                    "title_col": _title,
                    "text_cols": _texts,
                    "all_cols": all_cols
                }
            }

        except Exception as e:
            logger.error(f"향상된 테이블 인제스트 오류: {e}")
            raise

    async def query(self, save_name: str, question: str, top_k: int = 5, margin: float = 0.12) -> Dict[str, Any]:
        """FAISS 인덱스를 사용하여 고급 질의응답을 수행합니다"""
        try:
            T0 = time.perf_counter()
            logger.info(f"향상된 질의 시작: '{question}' (save_name: {save_name})")

            # FAISS 인덱스 로드
            vectorstore = await self._load_faiss_index(save_name)

            # 고급 검색 수행 (비동기 래핑)
            result = await asyncio.to_thread(
                self._answer_question_sync, vectorstore, question, top_k, margin
            )

            T1 = time.perf_counter()
            logger.info(f"향상된 질의 완료: 총 소요시간 {T1-T0:.3f}초")

            return result

        except Exception as e:
            logger.error(f"향상된 질의 오류: {e}")
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

    async def view_table_data(self, table_name: str, simulate: bool = None, limit: int = 100) -> Dict[str, Any]:
        """테이블 데이터를 조회합니다 (UI 표시용)"""
        try:
            # simulate 파라미터가 None이면 연결 상태에 따라 자동 결정
            if simulate is None:
                simulate = not self._connection_available

            logger.info(f"테이블 '{table_name}' 데이터 조회 시작")
            logger.info(f"  - 시뮬레이션 모드: {simulate}")
            logger.info(f"  - 최대 행 수: {limit}")

            if simulate or not self._connection_available:
                # 시뮬레이션 데이터 반환
                df = await asyncio.to_thread(self._create_enhanced_simulation_data)

                # 요청된 테이블명과 일치하지 않으면 기본 데이터 반환
                if table_name not in ["knowledge", "products"]:
                    return {
                        "ok": True,
                        "simulate": True,
                        "table_name": table_name,
                        "message": f"시뮬레이션 모드에서는 '{table_name}' 테이블이 없습니다. 기본 knowledge 데이터를 표시합니다.",
                        "total_rows": len(df),
                        "displayed_rows": min(len(df), limit),
                        "columns": df.columns.tolist(),
                        "data": df.head(limit).to_dict('records')
                    }

                # 제품 데이터 시뮬레이션
                if table_name == "products":
                    products_data = pd.DataFrame([
                        {
                            "id": 1,
                            "name": "QA 시스템 Pro",
                            "category": "AI Software",
                            "description": "RAG 기반 질의응답 시스템으로 대규모 문서에서 정확한 답변을 제공합니다.",
                            "price": 299.99,
                            "features": "자동 인덱싱, 실시간 검색, 다국어 지원, API 제공",
                            "created_at": "2024-01-01 00:00:00"
                        },
                        {
                            "id": 2,
                            "name": "벡터 검색 엔진",
                            "category": "Database",
                            "description": "고성능 벡터 유사도 검색을 지원하는 전문 데이터베이스입니다.",
                            "price": 499.99,
                            "features": "FAISS 통합, 분산 처리, REST API, 실시간 업데이트",
                            "created_at": "2024-01-01 00:00:00"
                        },
                        {
                            "id": 3,
                            "name": "문서 임베딩 도구",
                            "category": "AI Tools",
                            "description": "다양한 형식의 문서를 고품질 벡터로 변환하는 도구입니다.",
                            "price": 199.99,
                            "features": "다중 형식 지원, 배치 처리, 클라우드 연동, 자동 청킹",
                            "created_at": "2024-01-01 00:00:00"
                        }
                    ])
                    df = products_data

                return {
                    "ok": True,
                    "simulate": True,
                    "table_name": table_name,
                    "message": f"시뮬레이션 모드에서 '{table_name}' 테이블 데이터를 표시합니다.",
                    "total_rows": len(df),
                    "displayed_rows": min(len(df), limit),
                    "columns": df.columns.tolist(),
                    "data": df.head(limit).to_dict('records')
                }

            else:
                # 실제 MariaDB에서 데이터 조회
                engine = await asyncio.to_thread(self._make_async_engine, False)
                await self._ensure_connection(engine, False)
                df = await self._fetch_table(engine, table_name, limit=limit)
                await engine.dispose()

                return {
                    "ok": True,
                    "simulate": False,
                    "table_name": table_name,
                    "message": f"MariaDB에서 '{table_name}' 테이블 데이터를 조회했습니다.",
                    "total_rows": len(df),
                    "displayed_rows": len(df),
                    "columns": df.columns.tolist(),
                    "data": df.to_dict('records')
                }

        except Exception as e:
            logger.error(f"테이블 데이터 조회 오류: {e}")
            # 오류 발생 시 시뮬레이션 데이터로 fallback
            try:
                df = await asyncio.to_thread(self._create_enhanced_simulation_data)
                return {
                    "ok": True,
                    "simulate": True,
                    "table_name": table_name,
                    "message": f"오류로 인해 시뮬레이션 데이터를 표시합니다. 오류: {str(e)}",
                    "total_rows": len(df),
                    "displayed_rows": min(len(df), limit),
                    "columns": df.columns.tolist(),
                    "data": df.head(limit).to_dict('records'),
                    "error": str(e)
                }
            except Exception as fallback_error:
                raise Exception(f"테이블 조회 및 fallback 모두 실패: {e}, {fallback_error}")

    # === 내부 구현 메서드들 ===

    def _make_async_engine(self, simulate: bool = False):
        """비동기 엔진 생성"""
        if not simulate:
            dsn = f"mysql+aiomysql://{self.db_user}:{self.db_pass}@{self.db_host}:{self.db_port}/{self.db_name}"
        else:
            dsn = "sqlite+aiosqlite:///:memory:"
        return create_async_engine(dsn, pool_pre_ping=True, future=True)

    async def _ensure_connection(self, engine, simulate):
        """연결 확인 및 시뮬레이션 데이터 부트스트랩"""
        async with engine.begin() as conn:
            if simulate:
                await self._bootstrap_simulation(conn)
            else:
                await conn.execute(text("SELECT 1"))
        return engine

    async def _bootstrap_simulation(self, conn):
        """SQLite 메모리DB에 테스트 테이블/데이터 주입"""
        await conn.execute(text("""
            CREATE TABLE knowledge (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                term TEXT NOT NULL,
                description TEXT NOT NULL,
                role TEXT,
                details TEXT,
                updated_at TEXT NOT NULL
            );
        """))
        rows = [
            ("어텐션 메커니즘",
             "어텐션은 입력의 중요한 부분에 가중치를 부여해 정보를 통합하는 기법이다. 시퀀스 처리에서 문맥 의존성을 강화한다.",
             "입력 토큰 간 상호연관성을 계산하며 정보 흐름을 개선한다.",
             "Transformer의 핵심 구성요소로 번역·요약 등에서 성능을 끌어올린다.",
             "2024-01-01 00:00:00"),
            ("Self-Attention",
             "Self-Attention은 동일 시퀀스 내 토큰들이 서로를 참조하여 가중합을 구한다. RNN의 순차 의존성을 줄여 병렬화를 가능케 한다.",
             "장기 의존성 문제를 완화하고 각 토큰의 전역 문맥 파악을 돕는다.",
             "멀티헤드로 다양한 표현 공간에서 주의를 분산해 학습을 안정화한다.",
             "2024-01-01 00:00:00"),
            ("FAISS",
             "FAISS는 대규모 벡터에 대한 빠른 유사도 검색을 제공한다. 근사 최근접 탐색을 지원한다.",
             "대규모 임베딩 인덱싱과 빠른 검색을 제공한다.",
             "Facebook AI Research에서 개발되었고 CPU/GPU 백엔드를 제공한다.",
             "2024-01-01 00:00:00"),
        ]
        for t, d, r, det, up in rows:
            await conn.execute(
                text("INSERT INTO knowledge(term, description, role, details, updated_at) VALUES (:t,:d,:r,:det,:u)"),
                {"t": t, "d": d, "r": r, "det": det, "u": up}
            )
        await conn.commit()

    async def _infer_schema(self, engine, table: str, id_col=None, title_col=None, text_cols=None):
        """스키마 추론"""
        cols_info = await self._async_get_columns(engine, table)
        cols = [c["name"] for c in cols_info]
        return self._infer_enhanced_schema(cols, id_col, title_col, text_cols)

    async def _async_get_columns(self, engine, table: str):
        """비동기 컬럼 조회"""
        async with engine.connect() as conn:
            def _inspect(sync_conn):
                insp = sa_inspect(sync_conn)
                return insp.get_columns(table)
            return await conn.run_sync(_inspect)

    def _infer_enhanced_schema(self, columns, id_col=None, title_col=None, text_cols=None):
        """향상된 스키마 추론"""
        TITLE_CANDIDATES = {"title","name","term","keyword","subject","heading"}
        TEXT_CANDIDATES = {"body","content","description","details","text","summary","note","notes","paragraph","article"}
        ID_CANDIDATES = {"id","pk","gid","uid"}

        def pick(cands):
            for c in columns:
                if c.lower() in cands:
                    return c
            return None

        _id = id_col or pick(ID_CANDIDATES) or (columns[0] if columns else None)
        _title = title_col or pick(TITLE_CANDIDATES)
        _texts = text_cols or [c for c in columns if c.lower() in TEXT_CANDIDATES]

        if not _texts:
            _texts = [c for c in columns if c != _title]

        return _id, _title, _texts, columns

    async def _fetch_table(self, engine, table: str, columns=None, limit=None):
        """안전한 테이블 조회"""
        def _qt(name: str) -> str:
            return f"`{str(name).replace('`', '``')}`"

        async with engine.connect() as conn:
            def _inspect(sync_conn):
                insp = sa_inspect(sync_conn)
                return {c["name"] for c in insp.get_columns(table)}
            available_cols = await conn.run_sync(_inspect)

        use_cols = sorted(list(available_cols))
        cols_sql = ", ".join(_qt(c) for c in use_cols)
        sql = f"SELECT {cols_sql} FROM {_qt(table)}"

        if limit is not None:
            try:
                n = int(limit)
                if n > 0:
                    sql += f" LIMIT {n}"
            except Exception:
                pass

        async with engine.connect() as conn:
            result = await conn.execute(text(sql))
            rows = [dict(r._mapping) for r in result.fetchall()]

        # 문자열 정규화
        for row in rows:
            for k, v in list(row.items()):
                if isinstance(v, bytes):
                    row[k] = v.decode("utf-8", "ignore")
                elif v is None:
                    row[k] = ""

        return pd.DataFrame(rows)

    def _create_enhanced_simulation_data(self) -> pd.DataFrame:
        """향상된 시뮬레이션 데이터"""
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

    def _to_enhanced_documents(self, df, id_col, title_col, text_cols):
        """향상된 문서 생성 (OO 질의 확장 포함)"""
        documents = []

        for _, row in df.iterrows():
            title_val = (row[title_col] if title_col and title_col in row else None) or ""
            base_texts = []
            for c in text_cols:
                v = row.get(c, None)
                if isinstance(v, str):
                    base_texts.append(v)
            base = " ".join([t.strip() for t in base_texts if t])
            OO = str(title_val).strip() or (base.split()[0] if base else "항목")

            # 확장된 텍스트 생성 (OO 질의 형태)
            two_sentences = self._ensure_two_sentences(base if base else f"{OO}에 대한 설명이 데이터베이스에 포함되어 있다.")
            expanded = (
                f"[정의] {OO}는 무엇인가? {two_sentences}\n"
                f"[역할] {OO}의 역할은 무엇인가? {self._ensure_two_sentences(base)}\n"
                f"[설명] {OO}를 설명하라: {self._ensure_two_sentences(base)}\n"
                f"[키워드] {OO}, 정의, 역할, 설명, 개요, 특징, 장점, 한계"
            )

            page_content = "passage: " + expanded + "\n\n" + base
            metadata = {"OO": OO}

            if id_col and id_col in row:
                metadata["id"] = int(row[id_col]) if pd.notna(row[id_col]) else None
            if title_col and title_col in row:
                metadata["title"] = str(row[title_col])
            for c in text_cols:
                v = row.get(c, None)
                if isinstance(v, str):
                    metadata[c] = v[:3000]

            documents.append(Document(page_content=page_content, metadata=metadata))

        return documents

    def _ensure_two_sentences(self, text: str) -> str:
        """2문장 보장"""
        SENT_SEP = re.compile(r"(?<=[.!?。])\s+")
        parts = [p.strip() for p in SENT_SEP.split(text) if p.strip()]
        if len(parts) >= 2:
            return " ".join(parts[:2])
        if parts:
            return parts[0] + " 추가적인 설명은 본문에 포함되어 있다."
        return "이 항목은 데이터베이스에 기술되어 있으며, 세부 내용은 본문을 참조한다."

    def _split_documents(self, docs):
        """문서 분할"""
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
            separators=["\n\n","\n","。",". ",".","? ","?","! ","!"," "]
        )
        chunks = splitter.split_documents(docs)
        logger.info(f"청크 수: {len(chunks)}")
        return chunks

    def _build_and_save_faiss(self, chunks, save_dir):
        """FAISS 구축 및 저장"""
        os.makedirs(save_dir, exist_ok=True)
        vs = FAISS.from_documents(chunks, self.embedding_handler.embeddings)
        vs.save_local(save_dir)
        logger.info(f"FAISS 저장: {save_dir}")
        return vs

    async def _load_faiss_index(self, save_name: str):
        """FAISS 인덱스 로드"""
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

    # === 고급 검색 및 처리 함수들 ===

    def _answer_question_sync(self, vs: FAISS, question: str, top_k: int, margin: float) -> Dict[str, Any]:
        """동기 방식으로 질문에 답변 (고급 검색 알고리즘 적용)"""
        T0 = time.perf_counter()

        # A) 고급 검색 수행
        tA0 = time.perf_counter()
        fetch_k = max(top_k * self.config.fetch_multiplier, top_k + 5)
        kept = self._similarity_search_with_margin(vs, question, fetch_k, margin)
        tA1 = time.perf_counter()

        if not kept:
            perf_log.warning(f"검색 결과 없음 - 질의: {question}")
            return {"answer": "지식베이스에서 답을 찾지 못했습니다.", "sources": []}

        # B) 앵커 기반 필터링
        strong, weak = self._anchors_from_query(question)
        if strong:
            kept = [(d, s) for (d, s) in kept if self._hit_strong_anchors(d, strong)]
            if not kept:
                perf_log.warning(f"강 앵커 미스 - 질의: {question}")
                return {"answer": "지식베이스에서 답을 찾지 못했습니다.", "sources": []}

        # C) 라운드로빈 다변화
        kept_final = self._diversify_results(kept, top_k)

        # D) 컨텍스트 생성 및 LLM 호출
        tB0 = time.perf_counter()
        context = self._build_marked_context(kept_final)
        topic = self._extract_topic(kept_final)
        prompt = self._build_rag_prompt(question, context, topic, strong)
        tB1 = time.perf_counter()

        tC0 = time.perf_counter()
        answer = self._qwen_answer_sync(prompt)
        if answer and topic:
            answer = self._ensure_topic_prefix(answer, topic)
        tC1 = time.perf_counter()

        # E) 소스 정보 구성
        sources = [{
            "marker": f"S{i+1}",
            "id": (doc.metadata or {}).get("id", "Unknown"),
            "title": (doc.metadata or {}).get("title", "Unknown"),
            "OO": (doc.metadata or {}).get("OO", topic),
            "score": float(score),
        } for i, (doc, score) in enumerate(kept_final)]

        T1 = time.perf_counter()
        perf_log.warning(
            f"RAG 타이밍: 검색={tA1-tA0:.3f}s 조립={tB1-tB0:.3f}s LLM={tC1-tC0:.3f}s 총={T1-T0:.3f}s "
            f"(최종={len(kept_final)}개)"
        )

        return {"answer": answer, "sources": sources}

    def _similarity_search_with_margin(self, vs: FAISS, query: str, k: int, margin: float) -> List[Tuple]:
        """마진 기반 유사도 검색"""
        q = f"query: {query.strip()}"  # bge-m3 포맷
        pairs = vs.similarity_search_with_score(q, k=k)
        if not pairs:
            return []

        # FAISS metric 확인
        metric_type = getattr(getattr(vs, "index", None), "metric_type", None)
        try:
            import faiss
            is_ip = (metric_type == faiss.METRIC_INNER_PRODUCT)
        except Exception:
            is_ip = False

        # 점수 정규화
        if is_ip:
            pairs.sort(key=lambda x: float(x[1]), reverse=True)
            raw = [float(s) for _, s in pairs]
            hi = max(1.0, max(raw))
            lo = min(-1.0, min(raw))
            sims = [(r - lo) / (hi - lo + 1e-9) for r in raw]
        else:
            pairs.sort(key=lambda x: float(x[1]))
            dists = [float(d) for _, d in pairs]
            sims = [1.0 / (1.0 + d) for d in dists]

        docs = [d for d, _ in pairs]
        best = sims[0]
        cut = max(best - margin, best * (1.0 - margin))

        kept = [(doc, s) for doc, s in zip(docs, sims) if s >= cut and s >= self.config.sim_floor]

        perf_log.debug(
            f"검색 메트릭={('IP' if is_ip else 'L2')} 최고={best:.3f} 컷오프={cut:.3f} "
            f"점수들={[round(x,3) for x in sims[:min(5, len(sims))]]} 유지={len(kept)}개"
        )

        return kept[:k]

    def _anchors_from_query(self, q: str) -> Tuple[set, set]:
        """질의에서 강/약 앵커 추출"""
        weak_ko = {"무엇", "무엇인가", "뭐야", "뭔가", "정의", "설명", "설명하라", "설명해", "역할", "개요", "특징", "장점", "한계", "의미", "소개", "예시", "예", "비교"}
        weak_en = {"what", "define", "definition", "explain", "role", "overview", "feature", "pros", "cons"}

        # 단어 추출 (정규화된 형태)
        hangul_or_word = re.compile(r"[가-힣A-Za-z0-9][가-힣A-Za-z0-9\-_/]+")
        words = set()
        for w in hangul_or_word.findall(q.lower()):
            if len(w) >= 2:
                words.add(unicodedata.normalize("NFKC", w).strip().lower())

        strong = words - weak_ko - weak_en
        weak = words & (weak_ko | weak_en)

        # 도메인 특화 앵커 추가
        n = unicodedata.normalize("NFKC", q).lower()
        if re.search(r"\brag\b", n):
            strong |= {"rag", "retrieval augmented generation", "리트리벌 증강 생성", "리트리벌"}
        if "faiss" in n:
            strong.add("faiss")
        if "self-attention" in n or "self attention" in n:
            strong |= {"self-attention", "self attention", "셀프어텐션", "셀프 어텐션", "자기주의", "자기-주의"}
        if "attention" in n:
            strong |= {"attention", "어텐션", "주의"}

        return strong, weak

    def _hit_strong_anchors(self, doc, strong_anchors: set) -> bool:
        """문서가 강 앵커를 포함하는지 확인"""
        if not strong_anchors:
            return False
        text = unicodedata.normalize("NFKC", (doc.page_content or "").replace("passage:", " ")).lower()
        title = unicodedata.normalize("NFKC", (doc.metadata or {}).get("title", "")).lower()
        return any(anchor in text or anchor in title for anchor in strong_anchors)

    def _diversify_results(self, kept: List[Tuple], top_k: int) -> List[Tuple]:
        """같은 제목/OO 그룹에서 라운드로빈으로 다변화"""
        groups = defaultdict(list)
        for d, s in kept:
            meta = d.metadata or {}
            key = unicodedata.normalize("NFKC", (meta.get("title") or meta.get("OO") or "unknown")).lower()
            groups[key].append((d, s))

        # 각 그룹을 점수순 정렬하고 캡 적용
        for k in list(groups.keys()):
            groups[k].sort(key=lambda x: float(x[1]), reverse=True)
            groups[k] = groups[k][:self.config.per_title_cap]

        # 라운드로빈 선택
        result = []
        buckets = list(groups.values())
        i = 0
        while len(result) < top_k and buckets:
            progressed = False
            for bucket in list(buckets):
                if i < len(bucket):
                    result.append(bucket[i])
                    progressed = True
                    if len(result) >= top_k:
                        break
            if not progressed:
                break
            i += 1

        return result

    def _build_marked_context(self, docs_with_scores: List[Tuple]) -> str:
        """마커 포함 컨텍스트 구성"""
        buf = ["<CONTEXT>"]
        total_chars = 0
        for i, (doc, score) in enumerate(docs_with_scores, start=1):
            meta, text = (doc.metadata or {}), (doc.page_content or "")
            remain = max(self.config.max_context_chars - total_chars, 0)
            if remain <= 0:
                break
            snippet = text[:min(900, remain)] + ("…" if len(text) > 900 else "")
            total_chars += len(snippet)
            buf.append(f"《S{i}》 [id={meta.get('id','')}] [title={meta.get('title','')}] [OO={meta.get('OO','')}] score={score:.4f}\n{snippet}\n")
        buf.append("</CONTEXT>")
        return "\n".join(buf)

    def _extract_topic(self, docs_with_scores: List[Tuple]) -> str:
        """주제 용어 추출"""
        if not docs_with_scores:
            return ""
        top_meta = (docs_with_scores[0][0].metadata or {})
        return top_meta.get("title", top_meta.get("OO", ""))

    def _build_rag_prompt(self, question: str, context: str, topic: str, strong_anchors: set) -> str:
        """RAG용 프롬프트 구성"""
        strong_txt = ", ".join(sorted(strong_anchors)) if strong_anchors else ""

        return (
            "역할: 내부 지식베이스 기반 RAG 어시스턴트.\n"
            "규칙:\n"
            "1) 반드시 한국어로만 답한다.\n"
            "2) 아래 CONTEXT 안의 정보만 사용한다. CONTEXT 밖 지식/추측/상식은 금지.\n"
            "3) 충분한 정보가 없으면 정확히 다음 문장으로만 답한다: '지식베이스에서 답을 찾지 못했습니다.'\n"
            "4) 인용한 문장 뒤에는 [S1], [S2] 형태의 마커를 붙인다.\n"
            f"주제 용어: {topic}\n"
            f"핵심어(강 앵커): {strong_txt}\n\n"
            f"{context}\n\n"
            f"사용자 질문: {question}\n"
        )

    def _qwen_answer_sync(self, prompt: str) -> str:
        """Qwen 모델로 답변 생성 (동기 방식)"""
        try:
            t0 = time.perf_counter()
            # LLM 핸들러 사용
            answer = self.llm_handler.generate(
                prompt,
                max_length=self.config.max_new_tokens,
                stream=False
            )
            t1 = time.perf_counter()
            perf_log.warning(f"Qwen 생성 시간: {t1-t0:.3f}s")
            return self._clean_korean_output(answer)
        except Exception as e:
            logger.error(f"LLM 생성 오류: {e}")
            return "지식베이스에서 답을 찾지 못했습니다."

    def _ensure_topic_prefix(self, answer: str, topic: str) -> str:
        """답변 시작에 주제 보장"""
        if not topic:
            return answer
        topic_ko = self._replace_en_to_ko(topic)
        s = answer.strip()
        if s.startswith(topic_ko) or s.lower().startswith(topic.lower()):
            return s
        # 조사만 남고 주제가 사라진 케이스 교정
        s = re.sub(r"^[,\s\-–—]*[은는]\b", "", s).lstrip()
        postposition = self._pick_eun_neun(topic_ko)
        return f"{topic_ko}{postposition} {s}"

    def _pick_eun_neun(self, word: str) -> str:
        """은/는 조사 결정"""
        if not word:
            return "는"
        last = word[-1]
        code = ord(last)
        if 0xAC00 <= code <= 0xD7A3:
            jong = (code - 0xAC00) % 28
            return "은" if jong else "는"
        return "는"

    def _replace_en_to_ko(self, text: str) -> str:
        """영어를 한국어로 치환"""
        patterns = [
            (r"\bself[\-\s]?attention\b", "셀프-어텐션"),
            (r"\battention\b", "어텐션"),
            (r"\bsequence(s)?\b", "시퀀스"),
            (r"\btoken(s)?\b", "토큰"),
            (r"\bmodel(s)?\b", "모델"),
        ]
        s = text
        for pat, ko in patterns:
            s = re.sub(pat, ko, s, flags=re.I)
        return s

    def _clean_korean_output(self, text: str) -> str:
        """한국어 출력 품질 개선"""
        if not text:
            return text
        # 유니코드 정규화
        s = unicodedata.normalize("NFC", text)
        # 따옴표 제거
        if (s.startswith((""","\"","'","「","『")) and s.endswith((""","\"","'","」","』"))
            and len(s) > 2):
            s = s[1:-1].strip()
        # 공백 정리
        s = re.sub(r"[ \t]+", " ", s)
        s = re.sub(r"\s+([,.;:!?%])", r"\1", s)
        s = re.sub(r"\(\s+", "(", s)
        s = re.sub(r"\s+\)", ")", s)
        s = re.sub(r"\[\s+", "[", s)
        s = re.sub(r"\s+\]", "]", s)
        # 중복 구두점 제거
        s = re.sub(r"([,.;:!?])\s*\1+", r"\1", s)
        return s.strip()