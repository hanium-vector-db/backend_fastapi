# LLM FastAPI 서버

Retrieval-Augmented Generation (RAG) 기능을 갖춘 대형 언어 모델(LLM) 배포를 위한 프로덕션 준비 완료 FastAPI 서버입니다. 이 프로젝트는 문서 검색, 임베딩, 채팅 기능과 같은 고급 기능으로 사용자 정의 LLM을 제공하는 포괄적인 솔루션을 제공합니다.

## 🚀 주요 기능

- **사용자 정의 LLM 배포**: 4bit 양자화를 통한 Hugging Face 모델 지원
- **RAG (검색 증강 생성)**: 지능적인 문서 검색 및 컨텍스트 인식 응답
- **임베딩 생성**: BGE-M3 모델을 사용한 텍스트 임베딩 생성
- **채팅 인터페이스**: 대화 컨텍스트를 유지하는 대화형 채팅 기능
- **RESTful API**: 자동 OpenAPI 문서화가 포함된 잘 문서화된 API 엔드포인트
- **Docker 지원**: 쉬운 확장을 위한 컨테이너화된 배포
- **프로덕션 준비**: 포괄적인 로깅, 오류 처리 및 상태 확인

## 🏗️ 프로젝트 구조

```
llm-fastapi-server/
├── src/
│   ├── main.py                    # FastAPI 애플리케이션 진입점
│   ├── api/
│   │   ├── __init__.py
│   │   └── routes.py              # API 엔드포인트 정의
│   ├── models/
│   │   ├── __init__.py
│   │   ├── llm_handler.py         # LLM 모델 관리
│   │   └── embedding_handler.py   # 임베딩 모델 관리
│   ├── services/
│   │   ├── __init__.py
│   │   └── rag_service.py         # RAG 기능
│   ├── core/
│   │   ├── __init__.py
│   │   ├── config.py              # 설정 관리
│   │   └── logger.py              # 로깅 설정
│   └── utils/
│       ├── __init__.py
│       └── helpers.py             # 유틸리티 함수
├── notebooks/
│   ├── llm_setup.ipynb            # 모델 설정 및 테스트
│   └── rag_development.ipynb      # RAG 개발 노트북
├── data/
│   └── vector_db/                 # 벡터 데이터베이스 저장소
├── requirements.txt               # Python 의존성
├── Dockerfile                     # Docker 설정
├── docker-compose.yml             # Docker Compose 설정
└── README.md                      # 프로젝트 문서
```

## 🔧 설치

### 사전 요구사항

- Python 3.11+
- CUDA 호환 GPU (대형 모델에 권장)
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

3. **의존성 설치**
   ```bash
   pip install -r requirements.txt
   ```

4. **Hugging Face 토큰 설정**
   ```bash
   # 환경 변수로 Hugging Face 토큰 설정
   export HUGGINGFACE_TOKEN="your_token_here"
   ```

## 🚀 사용법

### 서버 실행

#### 방법 1: Python 직접 실행
```bash
cd src
python main.py
```

#### 방법 2: uvicorn 사용
```bash
uvicorn src.main:app --host 0.0.0.0 --port 8000 --reload
```

#### 방법 3: Docker 사용
```bash
# Docker Compose로 빌드 및 실행
docker-compose up -d
```

### API 엔드포인트

서버가 실행되면 대화형 API 문서에 접근할 수 있습니다:

- **Swagger UI**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`

#### 사용 가능한 엔드포인트:

| 카테고리 | 엔드포인트 | 메소드 | 설명 |
|----------|-----------|--------|------|
| **기본 기능** | `/` | GET | 환영 메시지 및 엔드포인트 개요 |
| | `/api/v1/health` | GET | 상태 확인 및 서비스 상태 |
| | `/api/v1/generate` | POST | LLM을 사용한 텍스트 생성 |
| | `/api/v1/chat` | POST | LLM과 채팅 |
| | `/api/v1/embed` | POST | 텍스트 임베딩 생성 |
| | `/api/v1/rag` | POST | RAG 기반 질의응답 |
| **모델 관리** | `/api/v1/models` | GET | 지원되는 모든 모델 목록 조회 |
| | `/api/v1/models/categories` | GET | 모델 카테고리 및 분류 정보 |
| | `/api/v1/models/category/{category}` | GET | 특정 카테고리의 모델들 조회 |
| | `/api/v1/models/recommend` | POST | 시스템 사양 맞춤 모델 추천 |
| | `/api/v1/models/compare` | POST | 선택된 모델들의 성능 비교 |
| | `/api/v1/models/search` | GET | 모델 검색 및 필터링 |
| | `/api/v1/models/stats` | GET | 전체 모델 통계 정보 |
| | `/api/v1/models/switch` | POST | 현재 사용 중인 모델 전환 |
| | `/api/v1/models/info/{model_key}` | GET | 특정 모델의 상세 정보 |
| **시스템 정보** | `/api/v1/system/gpu` | GET | GPU 메모리 및 사용량 정보 |
| | `/api/v1/models/{model_key}` | GET | 특정 모델 정보 조회 |
| | `/api/v1/models/switch` | POST | 사용 중인 모델 전환 |
| **시스템** | `/api/v1/system/gpu` | GET | GPU 상태 및 메모리 정보 |

### 📝 API 사용법: LLM 모델로부터 응답 받기

서버가 실행되면, `curl`과 같은 도구를 사용하여 아래 예시처럼 간단하게 API를 호출하고 LLM의 응답을 받을 수 있습니다.

#### 1. 간단한 텍스트 생성 (`/generate`)

단순한 질문이나 지시사항을 보내고, 모델이 생성한 텍스트를 직접 받아봅니다.

```bash
curl -X POST "http://localhost:8000/api/v1/generate" \
     -H "Content-Type: application/json" \
     -d '{"prompt": "인공지능의 미래에 대해 짧은 글을 써줘."}'
```

#### 2. 대화형 채팅 (`/chat`)

대화 형식으로 메시지를 보내고, 문맥을 이해하는 듯한 답변을 받습니다.

```bash
curl -X POST "http://localhost:8000/api/v1/chat" \
     -H "Content-Type: application/json" \
     -d '{"message": "안녕? 너는 누구야?"}'
```

#### 3. RAG (검색 증강 생성) 질의응답 (`/rag`)

서버에 내장된 문서(예: 위키피디아)를 검색하여, 그 정보를 바탕으로 더 정확하고 상세한 답변을 생성합니다.

```bash
curl -X POST "http://localhost:8000/api/v1/rag" \
     -H "Content-Type: application/json" \
     -d '{"question": "트랜스포머 모델의 주요 특징은 무엇이야?"}'
```

---

#### ✨ 팁: 특정 모델 지정하여 사용하기

요청 시 `model_key`를 함께 보내면, 원하는 특정 모델을 지정하여 응답을 받을 수 있습니다. (사용 가능한 모델 목록은 `/api/v1/models` 엔드포인트로 확인)

예를 들어, 한국어에 특화된 `solar-10.7b` 모델을 사용하고 싶다면:

```bash
curl -X POST "http://localhost:8000/api/v1/generate" \
     -H "Content-Type: application/json" \
     -d '{ 
          "prompt": "한국의 수도는 어디야?",
          "model_key": "solar-10.7b"
     }'
```

---


### ⚙️ 기타 주요 API 예시

#### 모델 목록 조회
```bash
curl -X GET "http://localhost:8000/api/v1/models"
```

#### 모델 전환
```bash
curl -X POST "http://localhost:8000/api/v1/models/switch" \
     -H "Content-Type: application/json" \
     -d '{"model_key": "llama3.1-8b"}'
```

#### GPU 시스템 정보 조회
```bash
curl -X GET "http://localhost:8000/api/v1/system/gpu"
```

#### 시스템 사양에 맞는 모델 추천
```bash
curl -X POST "http://localhost:8000/api/v1/models/recommend" \
     -H "Content-Type: application/json" \
     -d '{"ram_gb": 16, "gpu_gb": 8, "use_case": "korean"}'
```

## 🔧 설정

### 환경 변수

| 변수 | 설명 | 기본값 |
|------|------|--------|
| `HUGGINGFACE_TOKEN` | Hugging Face API 토큰 | 필수 |
| `MODEL_ID` | LLM 모델 식별자 | `Qwen/Qwen2.5-1.5B-Instruct` |
| `EMBEDDING_MODEL` | 임베딩 모델 이름 | `BAAI/bge-m3` |
| `MAX_TOKENS` | 최대 생성 토큰 수 | `512` |
| `TEMPERATURE` | 생성 온도 | `0.7` |

### 모델 설정

서버는 **40개 이상**의 다양한 Hugging Face 모델을 지원합니다:

#### 🔥 카테고리별 모델 분류



**⚖️ 중형 모델 (7B-13B)**  
- `qwen2.5-7b`: Qwen 2.5 7B - 고성능 (16GB RAM, 8GB GPU)
- `llama3.1-8b`: Meta Llama 3.1 8B - 최신 (16GB RAM, 8GB GPU)
- `llama3-8b`: Meta Llama 3 8B - 안정 버전 (16GB RAM, 8GB GPU)
- `mistral-7b`: Mistral 7B - 고효율 (16GB RAM, 8GB GPU)
- `mistral-nemo`: Mistral Nemo 12B - 최신 (24GB RAM, 12GB GPU)
- `gemma2-9b`: Google Gemma 2 9B (18GB RAM, 10GB GPU)
- `yi-9b`: 01.AI Yi 1.5 9B - 다국어 (18GB RAM, 10GB GPU)
- `vicuna-7b`: LMSYS Vicuna 7B - 대화형 (16GB RAM, 8GB GPU)

**🇰🇷 한국어 특화 모델**
- `solar-10.7b`: Upstage SOLAR 10.7B - 한국어 특화 (20GB RAM, 12GB GPU)
- `kullm-polyglot-12.8b`: KULLM Polyglot 12.8B - 한국어 (24GB RAM, 14GB GPU)
- `ko-alpaca`: Beomi KoAlpaca 5.8B - 한국어 (12GB RAM, 6GB GPU)
- `eeve-korean-10.8b`: Yanolja EEVE 10.8B - 한국어 특화 (20GB RAM, 12GB GPU)





**🚀 대형 모델 (14B+)**
- `qwen2.5-14b`: Qwen 2.5 14B - 고성능 (28GB RAM, 16GB GPU)
- `qwen2.5-32b`: Qwen 2.5 32B - 대형 (64GB RAM, 32GB GPU)  
- `qwen2.5-72b`: Qwen 2.5 72B - 최고 성능 (144GB RAM, 72GB GPU)
- `llama3.1-70b`: Meta Llama 3.1 70B - 최고 성능 (140GB RAM, 70GB GPU)
- `mixtral-8x7b`: Mistral Mixtral 8x7B MoE (90GB RAM, 45GB GPU)

**🌐 다국어 특화 모델**
- `aya-23-8b`: Cohere Aya 23 8B - 다국어 (16GB RAM, 8GB GPU)
- `bloom-7b`: BigScience BLOOM 7B - 다국어 (16GB RAM, 8GB GPU)

#### 📋 모델 선택 가이드

**용도별 추천 모델:**
- **일반 대화/텍스트 생성**: `qwen2.5-1.5b`, `llama3.2-3b`, `qwen2.5-7b`
- **한국어 특화 작업**: `solar-10.7b`, `kullm-polyglot-12.8b`, `eeve-korean-10.8b`
- **다국어 지원**: `aya-23-8b`, `bloom-7b`, `yi-9b`
- **최고 성능 (리소스 충분)**: `qwen2.5-72b`, `llama3.1-70b`, `mixtral-8x7b`

**시스템 사양별 추천:**
- **4GB RAM, 2GB GPU**: `qwen2.5-0.5b`, `qwen2.5-1.5b`, `gemma2-2b`
- **8GB RAM, 4GB GPU**: `llama3.2-3b`, `phi3-mini`, `phi3.5-mini`
- **16GB RAM, 8GB GPU**: `qwen2.5-7b`, `llama3.1-8b`, `mistral-7b`, `codellama-7b`
- **32GB+ RAM, 16GB+ GPU**: `qwen2.5-14b`, `solar-10.7b`, `codellama-13b`
- **solar-10.7b**: `upstage/SOLAR-10.7B-Instruct-v1.0` - Upstage의 한국어 특화
- **kullm-polyglot-12.8b**: `nlpai-lab/kullm-polyglot-12.8b-v2` - 한국어 전용



#### 🚀 대형 모델 (30B+) - 최고 성능
- **qwen2.5-32b**: `Qwen/Qwen2.5-32B-Instruct` - 대형 고성능
- **llama3.1-70b**: `meta-llama/Meta-Llama-3.1-70B-Instruct` - 최고 성능 (멀티 GPU 필요)

## 🧪 개발

### 테스트 실행
```bash
# 개발 노트북 실행
jupyter lab notebooks/
```

### 새 모델 추가

1. 새 모델 설정으로 `llm_handler.py` 업데이트
2. 새 의존성이 필요한 경우 `requirements.txt` 수정
3. 제공된 노트북으로 테스트

## 📦 Docker 배포

### 이미지 빌드
```bash
docker build -t llm-fastapi-server .
```

### 컨테이너 실행
```bash
docker run -p 8000:8000 \
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

이 프로젝트는 MIT 라이선스 하에 라이선스가 부여됩니다 - 자세한 내용은 [LICENSE](LICENSE) 파일을 참조하세요.

## 🙏 감사의 말

- 우수한 모델 호스팅을 제공하는 [Hugging Face](https://huggingface.co/)
- 놀라운 웹 프레임워크를 제공하는 [FastAPI](https://fastapi.tiangolo.com/)
- RAG 기능을 제공하는 [LangChain](https://langchain.com/)
- 임베딩 모델을 제공하는 [Sentence Transformers](https://www.sbert.net/)

## 📞 지원

질문 및 지원:

- 이 저장소에서 이슈 생성
- 서버 실행 시 [문서](http://localhost:8000/docs) 확인
- `notebooks/` 디렉토리의 예제 노트북 검토

---

**참고**: 이 서버는 교육 및 개발 목적으로 설계되었습니다. 프로덕션 배포의 경우 적절한 보안 조치, 인증 및 확장 구성을 보장하세요.