# LLM FastAPI 서버

Retrieval-Augmented Generation (RAG) 기능을 갖춘 대형 언어 모델(LLM) 배포를 위한 프로덕션 준비 완료 FastAPI 서버입니다. 3개의 엄선된 고성능 모델을 지원합니다.

## 🚀 주요 기능

- **3개 고성능 LLM 모델**: 4bit 양자화를 통한 메모리 최적화
- **RAG (검색 증강 생성)**: 지능적인 문서 검색 및 컨텍스트 인식 응답
- **임베딩 생성**: BGE-M3 모델을 사용한 텍스트 임베딩 생성
- **채팅 인터페이스**: 대화 컨텍스트를 유지하는 대화형 채팅 기능
- **RESTful API**: 자동 OpenAPI 문서화가 포함된 잘 문서화된 API 엔드포인트
- **Gradio UI**: 직관적인 웹 인터페이스
- **Docker 지원**: 쉬운 확장을 위한 컨테이너화된 배포

## 🏗️ 프로젝트 구조

```
llm-fastapi-server/
├── src/
│   ├── main.py                    # FastAPI 애플리케이션 진입점
│   ├── api/
│   │   └── routes.py              # API 엔드포인트 정의
│   ├── models/
│   │   ├── llm_handler.py         # LLM 모델 관리
│   │   └── embedding_handler.py   # 임베딩 모델 관리
│   ├── services/
│   │   └── rag_service.py         # RAG 기능
│   ├── core/
│   │   ├── config.py              # 설정 관리
│   │   └── logger.py              # 로깅 설정
│   └── utils/
│       └── helpers.py             # 유틸리티 함수
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

- **Gradio UI**: `http://localhost:8000/ui`
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
| | `/api/v1/models/switch` | POST | 현재 사용 중인 모델 전환 |
| | `/api/v1/models/recommend` | POST | 시스템 사양 맞춤 모델 추천 |
| **시스템 정보** | `/api/v1/system/gpu` | GET | GPU 메모리 및 사용량 정보 |

## 🤖 지원 모델 (3개)

### **qwen2.5-7b** (기본)
- **모델**: Qwen/Qwen2.5-7B-Instruct
- **설명**: 고성능 범용 모델
- **요구사항**: 16GB RAM, 8GB GPU
- **특징**: 한국어, 일반 텍스트, 코딩 지원

### **llama3.1-8b**
- **모델**: meta-llama/Meta-Llama-3.1-8B-Instruct
- **설명**: Meta의 최신 고성능 모델
- **요구사항**: 16GB RAM, 8GB GPU
- **특징**: 추론, 코딩, 일반 텍스트에 강함

### **gemma-3-4b**
- **모델**: google/gemma-2-9b-it
- **설명**: Google의 효율적인 중형 모델
- **요구사항**: 18GB RAM, 10GB GPU
- **특징**: 다국어 지원, 일반 텍스트 생성

## 📝 API 사용 예시

### 1. 간단한 텍스트 생성
```bash
curl -X POST "http://localhost:8000/api/v1/generate" \
     -H "Content-Type: application/json" \
     -d '{"prompt": "인공지능의 미래에 대해 짧은 글을 써줘."}'
```

### 2. 특정 모델로 채팅
```bash
curl -X POST "http://localhost:8000/api/v1/chat" \
     -H "Content-Type: application/json" \
     -d '{"message": "안녕하세요!", "model_key": "llama3.1-8b"}'
```

### 3. RAG 질의응답
```bash
curl -X POST "http://localhost:8000/api/v1/rag" \
     -H "Content-Type: application/json" \
     -d '{"question": "AI 기술의 최신 동향은?"}'
```

### 4. 모델 전환
```bash
curl -X POST "http://localhost:8000/api/v1/models/switch" \
     -H "Content-Type: application/json" \
     -d '{"model_key": "gemma-3-4b"}'
```

## ⚙️ 설정

### 환경 변수

| 변수 | 설명 | 기본값 |
|------|------|--------|
| `HUGGINGFACE_TOKEN` | Hugging Face API 토큰 | 필수 |
| `MODEL_ID` | 기본 LLM 모델 식별자 | `qwen2.5-7b` |
| `EMBEDDING_MODEL` | 임베딩 모델 이름 | `BAAI/bge-m3` |

### 모델 저장 위치

모든 모델은 `C:\huggingface_models\`에 자동으로 다운로드 및 캐시됩니다.

## 🐳 Docker 배포

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

이 프로젝트는 MIT 라이선스 하에 라이선스가 부여됩니다.

## 🙏 감사의 말

- [Hugging Face](https://huggingface.co/) - 우수한 모델 호스팅
- [FastAPI](https://fastapi.tiangolo.com/) - 놀라운 웹 프레임워크
- [LangChain](https://langchain.com/) - RAG 기능 제공
- [Gradio](https://gradio.app/) - 직관적인 UI 프레임워크

## 📞 지원

질문 및 지원:
- 이 저장소에서 이슈 생성
- 서버 실행 시 [문서](http://localhost:8000/docs) 확인
- Gradio UI: [http://localhost:8000/ui](http://localhost:8000/ui)

---

**참고**: 이 서버는 교육 및 개발 목적으로 설계되었습니다. 프로덕션 배포의 경우 적절한 보안 조치, 인증 및 확장 구성을 보장하세요.