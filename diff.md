# RAG 시스템 코드 비교: 외부 웹 vs 내부 DBMS 기반

## 파일 비교 분석

### external-web_closed_llm.py vs internal-dbms_vdb_open_llm-final.py

## 주요 구조적 차이점

### 1. **데이터 소스**
- **external-web**: 웹 검색(네이버 뉴스 API) 기반 외부 정보 수집
- **internal-dbms**: MariaDB/SQLite 내부 데이터베이스 기반

### 2. **LLM 모델**
- **external-web**: OpenAI GPT-4o (Closed Source LLM)
- **internal-dbms**: Qwen 1.5-0.5B-Chat (Open Weight LLM)

### 3. **임베딩 모델**
- **external-web**: OpenAI Embeddings
- **internal-dbms**: BAAI/bge-m3 (HuggingFace)

### 4. **주요 기능 차이**

#### External Web 기반 시스템
```python
# 웹 검색 → 벡터화
@app.post("/upload-topic")
async def upload_topic(topic: str)

# RAG 질의 응답
@app.post("/rag-query")
async def rag_query(rag_request: RAGRequest)
```

#### Internal DBMS 기반 시스템
```python
# DB 테이블 → 벡터화
@app.post("/ingest")
async def ingest_table_api(req: IngestRequest)

# 질의 응답
@app.post("/query")
async def query_api(req: QueryRequest)

# DB 테이블 목록 조회
@app.get("/db-tables")
async def list_tables()
```

### 5. **시스템 아키텍처 업그레이드**

#### A. 비동기 처리 개선
- **External**: 기본적인 FastAPI 비동기 처리
- **Internal**: 고급 lifespan 관리, asyncio.to_thread 오프로딩

#### B. 성능 최적화
```python
# Internal에만 있는 성능 개선 사항:
- LRU 캐시를 통한 모델 로딩 최적화
- CPU 친화적 설정 (OMP_NUM_THREADS, MKL_NUM_THREADS)
- 벡터스토어 메모리 캐싱 (VECTORSTORE_CACHE)
- 성능 계측 및 헤더 노출
```

#### C. 검색 알고리즘 고도화
```python
# Internal의 고급 검색 기능:
- 앵커 기반 필터링 (strong_anchors, weak_anchors)
- 마진 기반 유사도 컷오프
- 라운드로빈 다변화
- 동적 스코어링
```

#### D. 언어 처리 개선
```python
# Internal의 한국어 최적화:
- CJK(한자/가나) 토큰 금지
- 영한 용어 자동 치환
- 한국어 출력 품질 후처리
- 문장 수 보장 (최소 2문장)
```

### 6. **API 설계 진화**

#### Request/Response 모델 비교
```python
# External (단순)
class RAGRequest(BaseModel):
    prompt: str
    source: str
    top_k: int

# Internal (고도화)
class QueryRequest(BaseModel):
    save_name: str
    question: str
    top_k: int
    margin: float  # 추가된 파라미터
```

### 7. **인프라 및 운영**

#### External Web 시스템
- OpenAI API 의존성
- 네이버 검색 API 연동
- 외부 서비스 장애 영향

#### Internal DBMS 시스템
- 완전한 온프레미스 운영
- MariaDB 컨테이너화
- 시뮬레이션 모드 지원
- 헬스체크 및 상태 모니터링

### 8. **주요 업그레이드 포인트**

1. **자립성**: 외부 API 의존 → 완전 내재화
2. **성능**: 단순 검색 → 고급 검색 알고리즘
3. **언어**: 기본 처리 → 한국어 특화 최적화
4. **운영**: 기본 API → 엔터프라이즈급 모니터링
5. **확장성**: 고정 구조 → 동적 스키마 추론

## 결론

`internal-dbms_vdb_open_llm-final.py`는 단순한 웹 기반 RAG에서 엔터프라이즈급 내부 시스템으로의 완전한 전환을 보여주는 업그레이드입니다. 특히 성능, 언어 처리, 운영 안정성 측면에서 대폭 개선되었습니다.