# 🤖 LLM FastAPI 서버

Retrieval-Augmented Generation (RAG) 기능과 **실시간 스트리밍**을 갖춘 대형 언어 모델(LLM) 배포를 위한 프로덕션 준비 완료 FastAPI 서버입니다. 3개의 엄선된 고성능 모델을 지원합니다.

## ✨ 주요 기능

- **🔥 실시간 스트리밍**: Server-Sent Events를 통한 토큰별 실시간 텍스트 생성
- **3개 고성능 LLM 모델**: 4bit 양자화를 통한 메모리 최적화
- **🌐 이중 UI 지원**: Gradio UI + 전용 스트리밍 웹 페이지
- **RAG (검색 증강 생성)**: 지능적인 문서 검색 및 컨텍스트 인식 응답
- **🆕 Tavily 뉴스 기능**: 실시간 뉴스 검색, AI 요약, 트렌드 분석
- **임베딩 생성**: BGE-M3 모델을 사용한 텍스트 임베딩 생성
- **채팅 인터페이스**: 대화 컨텍스트를 유지하는 대화형 채팅 기능
- **RESTful API**: 자동 OpenAPI 문서화가 포함된 잘 문서화된 API 엔드포인트
- **Docker 지원**: 쉬운 확장을 위한 컨테이너화된 배포

## 🏗️ 프로젝트 구조

```
llm-fastapi-server/
├── src/
│   ├── main.py                    # FastAPI 애플리케이션 진입점 (스트리밍 지원)
│   ├── gradio_app.py              # Gradio UI 인터페이스
│   ├── api/
│   │   └── routes.py              # API 엔드포인트 정의 (스트리밍 API 포함)
│   ├── models/
│   │   ├── llm_handler.py         # LLM 모델 관리 (스트리밍 기능)
│   │   └── embedding_handler.py   # 임베딩 모델 관리
│   ├── services/
│   │   └── rag_service.py         # RAG 기능
│   ├── core/
│   │   ├── config.py              # 설정 관리
│   │   └── logger.py              # 로깅 설정
│   └── utils/
│       └── helpers.py             # 유틸리티 함수
├── static/
│   └── streaming.html             # 전용 실시간 스트리밍 페이지
├── data/
│   └── vector_db/                 # 벡터 데이터베이스 저장소
├── test_qwen.py                           # 단독 모델 테스트
├── test_api.py                            # API 기능 테스트
├── test_streaming.py                      # 스트리밍 기능 테스트
├── requirements.txt                       # Python 의존성
├── environment_python311_llm_server.yml   # Conda 환경 파일 (Python 3.11)
├── Dockerfile                             # Docker 설정
├── docker-compose.yml                     # Docker Compose 설정
├── API.md                                 # API 명세서
└── README.md                              # 프로젝트 문서
```

## 🔧 설치

### 사전 요구사항

- Python 3.11+ (권장: 3.11.x)
- Anaconda 또는 Miniconda (환경 복원용)
- CUDA 호환 GPU (권장: 8GB+ VRAM)
- Git

### 설정

1. **저장소 복제**
   ```bash
   git clone https://github.com/hanium-vector-db/AWS_LOCAL_LLM.git
   cd AWS_LOCAL_LLM
   ```

2. **가상 환경 생성**
   ```bash
   python -m venv venv
   source venv/bin/activate  # Windows: venv\Scripts\activate
   ```

3. **환경 설정**
   
   **방법 1: Conda 환경 복원 (권장)**
   ```bash
   # 제공된 환경 파일로 완전한 환경 복원
   conda env create -f environment_python311_llm_server.yml
   conda activate llm_server
   ```
   
   **방법 2: 수동 설치**
   ```bash
   # 의존성 설치
   pip install -r requirements.txt
   ```

4. **Hugging Face 토큰 설정**
   ```bash
   # 환경 변수로 Hugging Face 토큰 설정
   export HUGGINGFACE_TOKEN="your_token_here"
   ```

## 🚀 사용법

### 서버 실행

#### 방법 1: Python 직접 실행 (권장)
```bash
# 아나콘다 환경 활성화 (권장)
conda activate llm_server

# 서버 시작
python src/main.py
```

#### 방법 2: uvicorn 사용
```bash
uvicorn src.main:app --host 0.0.0.0 --port 8001 --reload
```

#### 방법 3: Docker 사용
```bash
# Docker Compose로 빌드 및 실행
docker-compose up -d
```

### 📱 웹 인터페이스 접근

서버가 시작되면 다음 웹 인터페이스에 접근할 수 있습니다:

- **🔥 실시간 스트리밍 페이지**: `http://localhost:8001/stream` (추천!)
- **Gradio UI**: `http://localhost:8001/ui`
- **API 문서 (Swagger)**: `http://localhost:8001/docs`
- **ReDoc**: `http://localhost:8001/redoc`

### 🎯 스트리밍 기능 사용법

#### 🔥 실시간 스트리밍 페이지 (추천)
1. 브라우저에서 `http://localhost:8001/stream` 접속
2. 프롬프트 입력
3. 모델 선택 (선택사항)
4. "생성하기" 버튼 클릭
5. 실시간으로 토큰별 텍스트 생성 확인!

#### 📋 API 스트리밍
```bash
# 스트리밍 모드로 텍스트 생성
curl -X POST "http://localhost:8001/api/v1/generate" \
     -H "Content-Type: application/json" \
     -H "Accept: text/event-stream" \
     -d '{"prompt": "Python의 장점에 대해 설명해주세요", "stream": true}'
```

#### 사용 가능한 엔드포인트:

| 카테고리 | 엔드포인트 | 메소드 | 설명 | 스트리밍 지원 |
|----------|-----------|--------|------|-------------|
| **UI 인터페이스** | `/stream` | GET | 실시간 스트리밍 전용 페이지 | ✅ |
| | `/ui` | GET | Gradio 통합 인터페이스 | ⚡ |
| **기본 기능** | `/` | GET | 환영 메시지 및 엔드포인트 개요 | - |
| | `/api/v1/health` | GET | 상태 확인 및 서비스 상태 | - |
| | `/api/v1/generate` | POST | LLM을 사용한 텍스트 생성 | ✅ |
| | `/api/v1/chat` | POST | LLM과 채팅 | ✅ |
| | `/api/v1/embed` | POST | 텍스트 임베딩 생성 | - |
| | `/api/v1/rag` | POST | RAG 기반 질의응답 | - |
| | `/api/v1/rag/update-news` | POST | 웹 뉴스로 RAG DB 업데이트 | - |
| **🆕 뉴스 기능** | `/api/v1/news/latest` | GET | 최신 뉴스 조회 | - |
| | `/api/v1/news/search` | GET | 뉴스 검색 | - |
| | `/api/v1/news/summary` | POST | AI 뉴스 요약 | - |
| | `/api/v1/news/analysis` | POST | 뉴스 트렌드 분석 | - |
| | `/api/v1/news/categories` | GET | 뉴스 카테고리 조회 | - |
| **모델 관리** | `/api/v1/models` | GET | 지원되는 3개 모델 목록 조회 | - |
| | `/api/v1/models/categories` | GET | 모델 카테고리 정보 | - |
| | `/api/v1/models/category/{category}` | GET | 특정 카테고리의 모델들 | - |
| | `/api/v1/models/recommend` | POST | 시스템 사양 맞춤 모델 추천 | - |
| | `/api/v1/models/compare` | POST | 모델 성능 비교 | - |
| | `/api/v1/models/search` | GET | 모델 검색 및 필터링 | - |
| | `/api/v1/models/stats` | GET | 모델 통계 정보 | - |
| | `/api/v1/models/switch` | POST | 현재 사용 중인 모델 전환 | - |
| | `/api/v1/models/info/{model_key}` | GET | 특정 모델 상세 정보 | - |
| **시스템 정보** | `/api/v1/system/gpu` | GET | GPU 메모리 및 사용량 정보 | - |

## 🤖 지원 모델 (3개)

### **qwen2.5-7b** (기본)
- **모델**: Qwen/Qwen2.5-7B-Instruct
- **설명**: 고성능 범용 모델
- **요구사항**: 16GB RAM, 8GB GPU
- **특징**: 한국어, 일반 텍스트, 코딩 지원

### **llama3.1-8b**
- **모델**: meta-llama/Meta-Llama-3-8B-Instruct
- **설명**: Meta의 고성능 모델
- **요구사항**: 16GB RAM, 8GB GPU
- **특징**: 추론, 코딩, 일반 텍스트에 강함

### **gemma-3-4b**
- **모델**: google/gemma-2-9b-it
- **설명**: Google의 효율적인 중형 모델
- **요구사항**: 18GB RAM, 10GB GPU
- **특징**: 다국어 지원, 일반 텍스트 생성

## 📝 API 사용 예시

### 1. 🔥 실시간 스트리밍 텍스트 생성
```bash
curl -X POST "http://localhost:8001/api/v1/generate" \
     -H "Content-Type: application/json" \
     -H "Accept: text/event-stream" \
     -d '{"prompt": "Python의 주요 특징을 설명해주세요.", "stream": true, "max_length": 300}'
```

### 2. 일반 텍스트 생성
```bash
curl -X POST "http://localhost:8001/api/v1/generate" \
     -H "Content-Type: application/json" \
     -d '{"prompt": "인공지능의 미래에 대해 짧은 글을 써줘.", "stream": false}'
```

### 3. 특정 모델로 스트리밍 채팅
```bash
curl -X POST "http://localhost:8001/api/v1/chat" \
     -H "Content-Type: application/json" \
     -H "Accept: text/event-stream" \
     -d '{"message": "안녕하세요!", "model_key": "llama3.1-8b", "stream": true}'
```

### 4. RAG 질의응답
```bash
curl -X POST "http://localhost:8001/api/v1/rag" \
     -H "Content-Type: application/json" \
     -d '{"question": "AI 기술의 최신 동향은?"}'
```

### 5. 모델 전환
```bash
curl -X POST "http://localhost:8001/api/v1/models/switch" \
     -H "Content-Type: application/json" \
     -d '{"model_key": "gemma-3-4b"}'
```

### 6. 시스템 상태 및 GPU 정보 확인
```bash
# 서버 상태 확인
curl -X GET "http://localhost:8001/api/v1/health"

# GPU 정보 확인
curl -X GET "http://localhost:8001/api/v1/system/gpu"

# 모델 검색
curl -X GET "http://localhost:8001/api/v1/models/search?keyword=korean"

# 모델 통계
curl -X GET "http://localhost:8001/api/v1/models/stats"
```

### 🆕 7. 뉴스 기능 사용 예시
```bash
# 최신 뉴스 조회
curl -X GET "http://localhost:8001/api/v1/news/latest?categories=technology,economy&max_results=5"

# 뉴스 검색
curl -X GET "http://localhost:8001/api/v1/news/search?query=ChatGPT&category=technology"

# AI 뉴스 요약
curl -X POST "http://localhost:8001/api/v1/news/summary" \
     -H "Content-Type: application/json" \
     -d '{"query": "인공지능 ChatGPT", "summary_type": "comprehensive", "max_results": 5}'

# 뉴스 트렌드 분석
curl -X POST "http://localhost:8001/api/v1/news/analysis" \
     -H "Content-Type: application/json" \
     -d '{"categories": ["politics", "economy", "technology"], "max_results": 15}'

# 지원 카테고리 조회
curl -X GET "http://localhost:8001/api/v1/news/categories"
```

## ⚙️ 설정

### 환경 변수

| 변수 | 설명 | 기본값 |
|------|------|--------|
| `HUGGINGFACE_TOKEN` | Hugging Face API 토큰 | 필수 |
| `TAVILY_API_KEY` | Tavily 뉴스 검색 API 토큰 | 선택 (뉴스 기능용) |
| `MODEL_ID` | 기본 LLM 모델 식별자 | `qwen2.5-7b` |
| `EMBEDDING_MODEL` | 임베딩 모델 이름 | `BAAI/bge-m3` |

### 모델 저장 위치

모든 모델은 `C:\huggingface_models\`에 자동으로 다운로드 및 캐시됩니다.

## 🧪 테스트

프로젝트에는 다양한 테스트 스크립트가 포함되어 있습니다:

### 단독 모델 테스트
```bash
python test_qwen.py
```

### API 기능 테스트
```bash
python test_api.py
```

### 스트리밍 기능 테스트
```bash
python test_streaming.py
```

## 🐳 Docker 배포

### 이미지 빌드
```bash
docker build -t llm-fastapi-server .
```

### 컨테이너 실행
```bash
docker run -p 8001:8001 \
  -e HUGGINGFACE_TOKEN="your_token" \
  -v $(pwd)/data:/app/data \
  llm-fastapi-server
```

### 프로덕션 배포
```bash
docker-compose up -d
```

## 🤝 기여하기

1. 저장소 포크
2. 기능 브랜치 생성 (`git checkout -b feature/amazing-feature`)
3. 변경사항 커밋 (`git commit -m 'Add amazing feature'`)
4. 브랜치에 푸시 (`git push origin feature/amazing-feature`)
5. Pull Request 열기

## 📝 라이선스

이 프로젝트는 MIT 라이선스 하에 라이선스가 부여됩니다.

## 🙏 감사의 말

- [Hugging Face](https://huggingface.co/) - 우수한 모델 호스팅
- [FastAPI](https://fastapi.tiangolo.com/) - 놀라운 웹 프레임워크
- [LangChain](https://langchain.com/) - RAG 기능 제공
- [Gradio](https://gradio.app/) - 직관적인 UI 프레임워크

## 📞 지원

질문 및 지원:
- 이 저장소에서 이슈 생성
- **📋 [API 명세서](./API.md)** - 완전한 API 문서
- 서버 실행 시 [Swagger UI](http://localhost:8001/docs) 확인
- 실시간 스트리밍: [http://localhost:8001/stream](http://localhost:8001/stream) ⚡
- Gradio UI: [http://localhost:8001/ui](http://localhost:8001/ui)

## 🎉 최신 업데이트 (v2.2)

### ✨ 새로운 기능 (v2.2)
- **🆕 Tavily 뉴스 통합**: Tavily API 기반 실시간 뉴스 검색 시스템
- **🤖 AI 뉴스 요약**: LLM을 이용한 지능형 뉴스 요약 (3가지 타입)
  - 간단 요약 (`brief`): 2-3문장 핵심 요약
  - 포괄적 요약 (`comprehensive`): 구조화된 상세 분석
  - 심층 분석 (`analysis`): 전문적 시각의 트렌드 분석
- **📊 뉴스 트렌드 분석**: 여러 카테고리 뉴스의 종합적 트렌드 파악
- **🗂️ 카테고리별 뉴스 검색**: 8개 카테고리 지원 (정치, 경제, 기술, 스포츠 등)
- **⏰ 시간대별 뉴스 조회**: 최근 1일/1주/1달 뉴스 필터링
- **🔍 스마트 뉴스 검색**: 키워드 기반 정확한 뉴스 검색
- **📋 5개 새로운 API 엔드포인트**: 완전한 뉴스 기능 API 세트

### 🔧 기존 기능 (v2.1)
- **🔥 실시간 스트리밍**: Server-Sent Events 기반 토큰별 실시간 텍스트 생성
- **🌐 전용 스트리밍 UI**: JavaScript 기반 현대적 웹 인터페이스
- **⚡ 성능 최적화**: Attention mask 최적화 및 메모리 효율성 개선
- **🧪 통합 테스트**: 포괄적인 테스트 스위트 포함
- **📱 반응형 디자인**: 모바일/데스크톱 친화적 사용자 경험
- **📊 터미널 로깅**: 생성된 답변을 터미널에서 실시간 확인
- **🔄 자동 리로드**: 코드 변경시 서버 자동 재시작 (`--reload` 옵션)
- **📋 완전한 API 문서**: 상세한 API 명세서 (API.md) 제공

### 🛠️ 기술적 개선
- `TextIteratorStreamer`를 이용한 실시간 스트리밍 구현
- FastAPI `StreamingResponse` 지원
- JavaScript `fetch` API를 이용한 클라이언트 구현
- 개선된 오류 처리 및 로깅
- Llama 모델 토크나이저 padding 토큰 최적화
- 스트리밍 터미널 출력 간소화

---

**참고**: 이 서버는 교육 및 개발 목적으로 설계되었습니다. 프로덕션 배포의 경우 적절한 보안 조치, 인증 및 확장 구성을 보장하세요.