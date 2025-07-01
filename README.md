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

| 엔드포인트 | 메소드 | 설명 |
|-----------|--------|------|
| `/` | GET | 환영 메시지 및 엔드포인트 개요 |
| `/api/v1/health` | GET | 상태 확인 및 서비스 상태 |
| `/api/v1/generate` | POST | LLM을 사용한 텍스트 생성 |
| `/api/v1/chat` | POST | LLM과 채팅 |
| `/api/v1/embed` | POST | 텍스트 임베딩 생성 |
| `/api/v1/rag` | POST | RAG 기반 질의응답 |

### API 호출 예시

#### 텍스트 생성
```bash
curl -X POST "http://localhost:8000/api/v1/generate" \
     -H "Content-Type: application/json" \
     -d '{"prompt": "인공지능에 대해 설명해주세요", "max_length": 512}'
```

#### 채팅
```bash
curl -X POST "http://localhost:8000/api/v1/chat" \
     -H "Content-Type: application/json" \
     -d '{"message": "머신러닝이 무엇인가요?"}'
```

#### RAG 질의
```bash
curl -X POST "http://localhost:8000/api/v1/rag" \
     -H "Content-Type: application/json" \
     -d '{"question": "트랜스포머 모델이 무엇인가요?"}'
```

#### 임베딩 생성
```bash
curl -X POST "http://localhost:8000/api/v1/embed" \
     -H "Content-Type: application/json" \
     -d '{"text": "임베딩을 위한 샘플 텍스트입니다"}'
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

서버는 다양한 Hugging Face 모델을 지원합니다:

- **소형 모델 (1B-3B)**: `Qwen/Qwen2.5-1.5B-Instruct`, `meta-llama/Llama-3.2-1B-Instruct`
- **중형 모델 (7B-13B)**: `meta-llama/Llama-2-7b-chat-hf`, `mistralai/Mistral-7B-Instruct-v0.1`
- **대형 모델 (30B+)**: 멀티 GPU 설정 필요

## 📊 성능

### 시스템 요구사항

| 모델 크기 | RAM | GPU 메모리 | 권장 GPU |
|-----------|-----|-----------|----------|
| 1B-3B | 8GB | 4GB | GTX 1660, RTX 3060 |
| 7B-13B | 16GB | 8GB | RTX 3080, RTX 4070 |
| 30B+ | 32GB | 24GB+ | RTX 4090, A100 |

### 최적화 기능

- **4bit 양자화**: 메모리 사용량 ~75% 감소
- **지연 로딩**: 처음 접근 시에만 모델 로드
- **효율적인 캐싱**: 벡터 데이터베이스 지속성
- **비동기 처리**: 블로킹되지 않는 API 응답

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
│   │   └── retrieval_service.py # Handles document retrieval
│   ├── core                   # Core application components
│   │   ├── __init__.py
│   │   ├── config.py          # Configuration settings
│   │   └── logger.py          # Logging setup
│   └── utils                  # Utility functions
│       ├── __init__.py
│       └── helpers.py         # Helper functions
├── notebooks                  # Jupyter notebooks for setup and development
│   ├── llm_setup.ipynb
│   └── rag_development.ipynb
├── data                       # Directory for vector database files
│   └── vector_db
├── requirements.txt           # Project dependencies
├── Dockerfile                 # Docker image instructions
├── docker-compose.yml         # Docker application configuration
└── README.md                  # Project documentation
```

## Setup Instructions

1. **Clone the Repository**
   ```
   git clone <repository-url>
   cd llm-fastapi-server
   ```

2. **Install Dependencies**
   Use pip to install the required packages:
   ```
   pip install -r requirements.txt
   ```

3. **Run the FastAPI Server**
   You can start the FastAPI server using:
   ```
   uvicorn src.main:app --reload
   ```

4. **Access the API**
   Once the server is running, you can access the API at `http://127.0.0.1:8000`. The interactive API documentation can be found at `http://127.0.0.1:8000/docs`.

## Usage Examples

- **Generate a Response**
  Send a POST request to the `/generate` endpoint with your query to receive a response from the LLM.

- **Retrieve Documents**
  Use the `/retrieve` endpoint to fetch relevant documents from the vector database.

## Contributing

Contributions are welcome! Please submit a pull request or open an issue for any enhancements or bug fixes.

## License

This project is licensed under the MIT License. See the LICENSE file for more details.