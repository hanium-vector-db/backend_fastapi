# LLM FastAPI 서버

Retrieval-Augmented Generation (RAG) 기능과 실시간 스트리밍을 갖춘 대형 언어 모델(LLM) 배포를 위한 프로덕션 준비 완료 FastAPI 서버입니다. 3개의 고성능 모델과 다양한 AI 기능을 제공합니다.

## 🚀 주요 기능

### 🤖 LLM 모델
- **3개 고성능 LLM 모델**: 4bit 양자화를 통한 메모리 최적화
- **실시간 스트리밍**: Server-Sent Events를 통한 토큰별 실시간 텍스트 생성
- **채팅 인터페이스**: 대화 컨텍스트를 유지하는 대화형 채팅 기능

### 🌐 Auto RAG 시스템 (NEW!)
- **자동 웹 검색 RAG**: 질문만 하면 자동으로 관련 뉴스 검색 → 벡터 DB화 → 답변 생성
- **실시간 진행 상황**: 시각적 진행률 바와 단계별 상태 표시
- **3가지 RAG 모드**:
  - 🚀 **Auto RAG**: 완전 자동화 (추천)
  - 📤 **주제 업로드**: 수동 주제 설정
  - 🗄️ **Internal-DB RAG**: 내부 데이터베이스 기반

### 📰 뉴스 기능
- **Tavily 뉴스 검색**: 실시간 뉴스 검색, AI 요약, 트렌드 분석
- **AI 뉴스 요약**: 다양한 요약 타입 지원 (간단/포괄적/심층분석)
- **뉴스 트렌드 분석**: 카테고리별 트렌드 분석

### 🎨 사용자 인터페이스
- **이중 UI 지원**: Gradio UI + 전용 스트리밍 웹 페이지
- **RESTful API**: 자동 OpenAPI 문서화가 포함된 잘 문서화된 API 엔드포인트
- **Docker 지원**: 쉬운 확장을 위한 컨테이너화된 배포

## 실행 환경 요구사항

### 하드웨어 요구사항
- **CPU**: Intel/AMD 64-bit 프로세서 (8코어 이상 권장)
- **RAM**: 최소 16GB (32GB 이상 권장)
- **GPU**: NVIDIA GPU with CUDA support
  - 최소 8GB VRAM (RTX 3070, RTX 4060 Ti 이상)
  - 권장 12GB VRAM (RTX 3080, RTX 4070 Ti 이상)
- **저장공간**: 50GB 이상 (모델 파일 포함)

### 소프트웨어 요구사항
- **운영체제**: Linux (Ubuntu 20.04+ 권장)
- **Python**: 3.11.x (필수)
- **CUDA**: 12.1 이상

## 📂 프로젝트 구조

```
AWS_LOCAL_LLM/
├── src/                           # 📦 메인 소스 코드
│   ├── main.py                    # 🚀 FastAPI 서버 진입점 (Auto RAG 포함)
│   ├── gradio_app.py              # 🎨 Gradio UI (진행률 표시 포함)
│   ├── api/
│   │   └── routes.py              # 🔌 API 엔드포인트 (스트리밍 Auto RAG)
│   ├── models/
│   │   ├── llm_handler.py         # 🤖 LLM 모델 관리
│   │   └── embedding_handler.py   # 🔤 임베딩 모델 관리
│   ├── services/
│   │   ├── rag_service.py         # 🌐 RAG 서비스 (Auto RAG 로직)
│   │   ├── internal_db_service.py # 🗄️ Internal-DB RAG
│   │   └── retrieval_service.py   # 🔍 문서 검색 서비스
│   ├── core/
│   │   ├── config.py              # ⚙️ 설정 관리
│   │   └── logger.py              # 📝 로깅 설정
│   └── utils/
│       ├── config_loader.py       # 📋 설정 로더
│       └── helpers.py             # 🔧 유틸리티 함수
├── tests/                         # 🧪 테스트 파일들
│   ├── test_auto_rag.py           # Auto RAG 테스트
│   ├── test_external_web_rag.py   # External-Web RAG 테스트
│   ├── test_news_search.py        # 뉴스 검색 테스트
│   └── debug_vector_db.py         # 벡터 DB 디버깅
├── config/                        # ⚙️ 설정 파일들
├── data/                          # 💾 데이터 저장소
├── static/                        # 🎨 정적 파일들
└── debug_py/                      # 🔍 디버깅 도구들

## 🚀 빠른 시작

### 1. 환경 설정
```bash
# 프로젝트 클론
git clone <repository-url>
cd AWS_LOCAL_LLM

# Python 가상환경 생성 및 활성화
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 또는
venv\Scripts\activate     # Windows

# 의존성 설치
pip install -r requirements.txt
```

### 2. 환경 변수 설정
```bash
# .env 파일 생성
cp .env.example .env

# 필요한 API 키 설정
# - TAVILY_API_KEY: 뉴스 검색용
# - HF_TOKEN: Hugging Face 토큰 (선택)
```

### 3. 서버 실행
```bash
# FastAPI 서버 시작
python src/main.py
```

### 4. 접속
- **메인 페이지**: http://localhost:8000
- **Gradio UI**: http://localhost:8000/ui
- **API 문서**: http://localhost:8000/docs
- **Auto RAG**: http://localhost:8000/ui → External-Web RAG → Auto RAG

## ✨ Auto RAG 사용법

### 🚀 자동 RAG (추천)
1. Gradio UI 접속: http://localhost:8000/ui
2. **External-Web RAG** 탭 클릭
3. **🚀 Auto RAG (추천!)** 탭 선택
4. 질문 입력 후 **🚀 자동 RAG 실행** 클릭
5. 실시간 진행 상황을 확인하며 답변 대기

**예시 질문:**
- "삼성전자 AI 반도체 최신 동향은?"
- "인공지능 투자 현황은 어떻습니까?"
- "ChatGPT 관련 최신 소식을 알려주세요"

### 📊 진행 상황 표시
- **🚀 시작** (5%): RAG 처리 시작
- **🔍 뉴스 검색** (20%): 웹에서 관련 뉴스 검색
- **📚 벡터 DB 저장** (50%): 검색된 뉴스를 벡터 DB에 저장
- **🤖 답변 생성** (70%): AI가 종합적인 답변 생성
- **📝 마무리** (90%): 관련 문서 정보 정리
- **✅ 완료** (100%): 최종 답변 및 참고 문서 표시

## 🔧 설치


### 설정

1. **저장소 복제**
   ```bash
   git clone https://github.com/hanium-vector-db/AWS_LOCAL_LLM.git
   cd AWS_LOCAL_LLM
   ```

2. **가상 환경 생성**
   ```bash
   python -m venv venv
   source venv/bin/activate 
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

## API 엔드포인트

### 기본 엔드포인트

| 메소드 | 엔드포인트 | 설명 | 스트리밍 지원 |
|--------|-----------|------|-------------|
| GET | `/` | 환영 메시지 및 서비스 개요 | - |
| GET | `/stream` | 실시간 스트리밍 웹 페이지 | ✅ |
| GET | `/ui` | Gradio 통합 인터페이스 | - |
| GET | `/api/v1/health` | 서버 상태 및 모델 정보 확인 | - |

### 텍스트 생성

#### POST `/api/v1/generate`
LLM을 사용한 텍스트 생성 (스트리밍 지원)

**요청 본문:**
```json
{
    "prompt": "Python의 장점에 대해 설명해주세요",
    "max_length": 512,
    "model_key": "qwen2.5-7b",
    "stream": true
}
```

**응답 (스트리밍):**
```
data: {"content": "Python은", "done": false}
data: {"content": " 간결하고", "done": false}
data: {"content": "", "done": true}
```

**응답 (일반):**
```json
{
    "response": "Python은 간결하고 읽기 쉬운 문법을 가진 프로그래밍 언어입니다...",
    "prompt": "Python의 장점에 대해 설명해주세요",
    "model_info": {
        "model_key": "qwen2.5-7b",
        "model_id": "Qwen/Qwen2.5-7B-Instruct",
        "description": "Qwen 2.5 7B - 고성능 범용 모델"
    }
}
```

#### POST `/api/v1/chat`
대화형 채팅 (스트리밍 지원)

**요청 본문:**
```json
{
    "message": "안녕하세요!",
    "model_key": "llama3.1-8b",
    "stream": true
}
```

### 임베딩 생성

#### POST `/api/v1/embed`
텍스트 임베딩 벡터 생성

**요청 본문:**
```json
{
    "text": "임베딩으로 변환할 텍스트"
}
```

**응답:**
```json
{
    "embedding": [0.1, -0.2, 0.3, ...],
    "dimension": 1024,
    "model_info": {
        "model_name": "BAAI/bge-m3",
        "embedding_dimension": 1024
    }
}
```

### RAG (검색 증강 생성)

#### POST `/api/v1/rag`
문서 기반 질의응답

**요청 본문:**
```json
{
    "question": "AI 기술의 최신 동향은?",
    "model_key": "qwen2.5-7b"
}
```

#### POST `/api/v1/rag/update-news`
최신 뉴스로 RAG 데이터베이스 업데이트

### 뉴스 기능

#### GET `/api/v1/news/latest`
최신 뉴스 조회

**쿼리 매개변수:**
- `categories`: 카테고리 (쉼표로 구분) - `technology,economy,politics`
- `max_results`: 최대 결과 수 (1-20)
- `time_range`: 시간 범위 (`d`, `w`, `m`)

#### GET `/api/v1/news/search`
키워드로 뉴스 검색

#### POST `/api/v1/news/summary`
AI 뉴스 요약 (스트리밍 지원)

#### POST `/api/v1/news/analysis`
뉴스 트렌드 분석 (스트리밍 지원)

#### GET `/api/v1/models`
지원하는 모델 목록 조회

#### POST `/api/v1/models/switch`
현재 사용 중인 모델 전환

#### POST `/api/v1/models/recommend`
시스템 사양 맞춤 모델 추천

### 시스템 정보

#### GET `/api/v1/system/gpu`
GPU 메모리 및 사용량 정보

**응답:**
```json
{
    "gpu_available": true,
    "gpu_count": 1,
    "gpu_memory": {
        "total": 12288,
        "used": 8192,
        "free": 4096
    },
    "gpu_utilization": 65.5,
    "cuda_version": "12.1"
}
```

## 설정

### 환경 변수

| 변수명 | 설명 | 기본값 | 필수 여부 |
|--------|------|--------|----------|
| `HUGGINGFACE_TOKEN` | Hugging Face API 토큰 | - | 필수 |
| `TAVILY_API_KEY` | Tavily 뉴스 검색 API 키 | - | 선택 (뉴스 기능용) |
| `MODEL_ID` | 기본 LLM 모델 식별자 | `qwen2.5-7b` | 선택 |
| `EMBEDDING_MODEL` | 임베딩 모델 이름 | `BAAI/bge-m3` | 선택 |
| `CUDA_VISIBLE_DEVICES` | 사용할 GPU 디바이스 | `0` | 선택 |

### 모델 저장 위치

모든 Hugging Face 모델은 `~/.huggingface_models/`에 자동으로 다운로드 및 캐시됩니다.

### GPU 설정

시스템은 단일 GPU (CUDA:0) 사용으로 최적화되어 있습니다:
- `CUDA_VISIBLE_DEVICES=0` 환경변수로 GPU 0번만 사용
- 모든 모델이 `cuda:0` 디바이스에 로드됨
- 병렬 처리 대신 메모리 효율적인 단일 GPU 처리

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

## API 명세서

전체 API 명세서는 다음 방법으로 확인할 수 있습니다:

- **API 명세 문서**: [API.md](./API.md) 파일 참조

## 테스트

### 기본 기능 테스트
```bash
# 모델 단독 테스트
python debug_py/test_qwen.py

# API 기능 테스트
python debug_py/test_api.py

# 스트리밍 기능 테스트
python debug_py/test_streaming.py

# 뉴스 기능 테스트
python debug_py/test_news_features.py
```

### 토큰 설정 확인
```bash
# Hugging Face 토큰 확인
python debug_py/setup_hf_token.py

# Llama 모델 접근 권한 확인
python debug_py/check_llama_access.py
```

## 문제 해결

### 일반적인 문제

**CUDA Out of Memory 오류**
- GPU 메모리 확인: `nvidia-smi`
- 더 작은 모델 사용 또는 max_length 값 감소

**Hugging Face 토큰 오류**
- 토큰 재설정: `export HUGGINGFACE_TOKEN="your_new_token"`
- 토큰 권한 확인: https://huggingface.co/settings/tokens

**모델 다운로드 실패**
- 네트워크 연결 확인: `curl -I https://huggingface.co`
- 캐시 디렉토리 권한 확인: `ls -la ~/.huggingface_models/`

**포트 충돌**
- 포트 사용 확인: `netstat -tlnp | grep :8001`
- 다른 포트 사용: `uvicorn src.main:app --port 8002`

## Docker 배포

### 기본 배포
```bash
# 이미지 빌드
docker build -t llm-fastapi-server .

# 컨테이너 실행
docker run -d \
  --name llm-server \
  --gpus all \
  -p 8001:8001 \
  -e HUGGINGFACE_TOKEN="your_token" \
  -e TAVILY_API_KEY="your_api_key" \
  -v ~/.huggingface_models:/root/.huggingface_models \
  llm-fastapi-server
```

### Docker Compose 배포
```bash
# 환경 변수 설정
cp .env.example .env
# .env 파일 편집

# 서비스 시작
docker-compose up -d
```

## 기여하기

1. 저장소 포크
2. 기능 브랜치 생성 (`git checkout -b feature/amazing-feature`)
3. 변경사항 커밋 (`git commit -m 'Add amazing feature'`)
4. 브랜치에 푸시 (`git push origin feature/amazing-feature`)
5. Pull Request 생성

## 라이선스

이 프로젝트는 MIT 라이선스 하에 배포됩니다. 자세한 내용은 [LICENSE](LICENSE) 파일을 참조하세요.

## 지원

### 문서 및 도움말
- **API 명세서**: [API.md](./API.md)
- **대화형 API 문서**: http://localhost:8001/docs
- **실시간 스트리밍 테스트**: http://localhost:8001/stream
- **Gradio UI**: http://localhost:8001/ui

### 이슈 및 버그 리포트
GitHub Issues: https://github.com/hanium-vector-db/AWS_LOCAL_LLM/issues