# LLM FastAPI Server - API 명세서

## 📋 개요

이 문서는 LLM FastAPI Server의 모든 API 엔드포인트와 사용법을 상세히 설명합니다.

**베이스 URL:** `http://localhost:8001`

---

## 🔥 주요 기능

- **40개 이상의 다양한 로컬 언어 모델 지원**
- **실시간 스트리밍 텍스트 생성**
- **한국어, 코딩, 수학 특화 모델**
- **RAG (검색 증강 생성) 기능**
- **GPU 메모리 최적화**
- **실시간 모델 전환**

---

## 🎯 기본 엔드포인트

### 1. 서버 정보 조회
```http
GET /
```

**응답 예시:**
```json
{
  "message": "🚀 LLM FastAPI 서버에 오신 것을 환영합니다!",
  "version": "1.0.0",
  "description": "40개 이상의 다양한 로컬 언어 모델을 지원하는 고성능 AI 서버",
  "features": [
    "다양한 크기의 LLM 모델 지원 (0.5B-72B)",
    "한국어, 코딩, 수학 특화 모델",
    "RAG (검색 증강 생성) 기능",
    "실시간 모델 전환",
    "GPU 메모리 최적화"
  ],
  "endpoints": { ... },
  "supported_model_categories": [
    "ultra-light (0.5B)",
    "light (1-3B)", 
    "medium (7-13B)",
    "large (14B+)",
    "korean (한국어 특화)",
    "code (코딩 특화)",
    "math (수학/과학 특화)",
    "multilingual (다국어 지원)"
  ]
}
```

### 2. 서버 상태 확인
```http
GET /api/v1/health
```

**응답 예시:**
```json
{
  "status": "healthy",
  "timestamp": "2024-01-01T00:00:00Z",
  "model_loaded": true,
  "current_model": "qwen2.5-7b"
}
```

---

## 🤖 텍스트 생성 API

### 1. 기본 텍스트 생성
```http
POST /api/v1/generate
```

**요청 본문:**
```json
{
  "prompt": "Python에 대해 설명해주세요",
  "max_length": 512,
  "temperature": 0.7,
  "top_p": 0.9,
  "stream": false
}
```

**매개변수:**
- `prompt` (string, 필수): 생성할 텍스트의 프롬프트
- `max_length` (integer, 기본값: 512): 생성할 최대 토큰 수
- `temperature` (float, 기본값: 0.7): 생성의 창의성 조절 (0.0~2.0)
- `top_p` (float, 기본값: 0.9): 토큰 선택의 다양성 조절 (0.0~1.0)
- `stream` (boolean, 기본값: false): 스트리밍 모드 활성화

**일반 응답:**
```json
{
  "response": "Python은 1991년 귀도 반 로섬이 개발한 고수준 프로그래밍 언어입니다...",
  "model": "qwen2.5-7b",
  "tokens_generated": 245
}
```

**스트리밍 응답 (stream: true):**
```
data: {"content": "Python은", "done": false}

data: {"content": " 1991년", "done": false}

data: {"content": " 귀도", "done": false}

...

data: {"content": "", "done": true}
```

### 2. 채팅 생성
```http
POST /api/v1/chat
```

**요청 본문:**
```json
{
  "message": "안녕하세요! 오늘 날씨가 어떤가요?",
  "stream": false
}
```

**응답:**
```json
{
  "response": "안녕하세요! 저는 AI 어시스턴트라서 실시간 날씨 정보에 접근할 수 없습니다...",
  "model": "qwen2.5-7b"
}
```

---

## 🔗 임베딩 API

### 임베딩 생성
```http
POST /api/v1/embed
```

**요청 본문:**
```json
{
  "text": "임베딩할 텍스트입니다"
}
```

**응답:**
```json
{
  "embedding": [0.1234, -0.5678, 0.9012, ...],
  "dimension": 1024,
  "model": "BAAI/bge-m3"
}
```

---

## 📚 RAG (검색 증강 생성) API

### RAG 질의
```http
POST /api/v1/rag
```

**요청 본문:**
```json
{
  "query": "Python의 장점은 무엇인가요?",
  "k": 3,
  "stream": false
}
```

**매개변수:**
- `query` (string, 필수): 검색할 질의
- `k` (integer, 기본값: 3): 검색할 문서 개수
- `stream` (boolean, 기본값: false): 스트리밍 모드

**응답:**
```json
{
  "answer": "Python의 주요 장점들은 다음과 같습니다...",
  "sources": [
    {
      "content": "Python은 가독성이 뛰어난 언어입니다...",
      "score": 0.95
    }
  ],
  "model": "qwen2.5-7b"
}
```

---

## 🎛️ 모델 관리 API

### 1. 지원 모델 목록 조회
```http
GET /api/v1/models
```

**응답:**
```json
{
  "models": {
    "qwen2.5-7b": {
      "model_id": "Qwen/Qwen2.5-7B-Instruct",
      "description": "Qwen 2.5 7B - 고성능 범용 모델",
      "category": "medium",
      "ram_requirement": "16GB",
      "gpu_requirement": "8GB",
      "performance_score": 85,
      "use_cases": ["general", "korean", "coding"]
    },
    "llama3.1-8b": {
      "model_id": "meta-llama/Meta-Llama-3-8B-Instruct",
      "description": "Meta Llama 3 8B - 고성능 모델",
      "category": "medium",
      "ram_requirement": "16GB",
      "gpu_requirement": "8GB",
      "performance_score": 88,
      "use_cases": ["general", "coding", "reasoning"]
    },
    "gemma-3-4b": {
      "model_id": "google/gemma-2-9b-it",
      "description": "Google Gemma 2 9B - 효율적인 중형 모델",
      "category": "medium",
      "ram_requirement": "18GB",
      "gpu_requirement": "10GB",
      "performance_score": 82,
      "use_cases": ["general", "multilingual"]
    }
  }
}
```

### 2. 모델 카테고리 조회
```http
GET /api/v1/models/categories
```

**응답:**
```json
{
  "categories": ["medium"]
}
```

### 3. 카테고리별 모델 조회
```http
GET /api/v1/models/category/{category}
```

**예시:** `/api/v1/models/category/medium`

### 4. 모델 추천
```http
POST /api/v1/models/recommend
```

**요청 본문:**
```json
{
  "ram_gb": 16,
  "gpu_gb": 8,
  "use_case": "korean"
}
```

**응답:**
```json
{
  "recommendations": [
    {
      "model_key": "qwen2.5-7b",
      "model_info": { ... },
      "recommendation_score": 85,
      "reasons": [
        "RAM 요구사항 충족 (16GB)",
        "GPU 메모리 요구사항 충족 (8GB)",
        "'korean' 용도에 적합"
      ]
    }
  ]
}
```

### 5. 모델 성능 비교
```http
POST /api/v1/models/compare
```

**요청 본문:**
```json
{
  "model_keys": ["qwen2.5-7b", "llama3.1-8b"]
}
```

### 6. 모델 검색
```http
GET /api/v1/models/search?q=korean
```

### 7. 모델 통계
```http
GET /api/v1/models/stats
```

### 8. 모델 전환
```http
POST /api/v1/models/switch
```

**요청 본문:**
```json
{
  "model_key": "llama3.1-8b"
}
```

### 9. 특정 모델 정보
```http
GET /api/v1/models/info/{model_key}
```

---

## 💻 시스템 정보 API

### GPU 정보 조회
```http
GET /api/v1/system/gpu
```

**응답:**
```json
{
  "gpu_available": true,
  "gpu_count": 1,
  "gpu_memory": "8GB",
  "cuda_version": "11.8"
}
```

---

## 🎨 UI 인터페이스

### 1. Gradio UI
```
GET /ui
```
- 대화형 웹 인터페이스
- 스트리밍 모드 지원
- 실시간 텍스트 생성

### 2. 커스텀 스트리밍 UI
```
GET /stream
```
- 실시간 스트리밍 전용 인터페이스
- Server-Sent Events 기반
- 빠른 응답 속도

---

## 📖 문서

### 1. API 문서 (Swagger UI)
```
GET /docs
```

### 2. API 문서 (ReDoc)
```
GET /redoc
```

---

## 🚨 오류 코드

| 상태 코드 | 설명 |
|---------|-----|
| 200 | 성공 |
| 400 | 잘못된 요청 |
| 404 | 리소스를 찾을 수 없음 |
| 500 | 서버 내부 오류 |

**오류 응답 예시:**
```json
{
  "error": "모델을 찾을 수 없습니다",
  "detail": "지원되지 않는 모델: unknown-model"
}
```

---

## 📝 사용 예시

### Python으로 API 호출
```python
import requests
import json

# 기본 텍스트 생성
response = requests.post("http://localhost:8001/api/v1/generate", 
    json={
        "prompt": "Python의 장점을 설명해주세요",
        "max_length": 256,
        "stream": False
    }
)
print(response.json()["response"])

# 스트리밍 생성
response = requests.post("http://localhost:8001/api/v1/generate",
    json={
        "prompt": "AI에 대해 설명해주세요",
        "stream": True
    },
    stream=True
)

for line in response.iter_lines():
    if line:
        data = json.loads(line.decode().replace('data: ', ''))
        if not data.get('done'):
            print(data['content'], end='', flush=True)
```

### JavaScript으로 스트리밍 호출
```javascript
const response = await fetch('/api/v1/generate', {
    method: 'POST',
    headers: {
        'Content-Type': 'application/json',
    },
    body: JSON.stringify({
        prompt: 'JavaScript에 대해 설명해주세요',
        stream: true
    })
});

const reader = response.body.getReader();
const decoder = new TextDecoder();

while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    
    const chunk = decoder.decode(value);
    const lines = chunk.split('\n');
    
    for (const line of lines) {
        if (line.startsWith('data: ')) {
            const data = JSON.parse(line.slice(6));
            if (!data.done) {
                console.log(data.content);
            }
        }
    }
}
```

---

## 🔧 설정

### 환경 변수
- `HUGGINGFACE_TOKEN`: Hugging Face 토큰 (필수)
- `SERVER_HOST`: 서버 호스트 (기본값: 0.0.0.0)
- `SERVER_PORT`: 서버 포트 (기본값: 8001)

### 모델 캐시
- 기본 캐시 디렉토리: `C:\huggingface_models`
- 첫 실행시 모델 다운로드로 시간 소요

---

## 📞 지원

버그 리포트나 기능 요청은 GitHub Issues를 통해 제출해주세요.
  
**최종 업데이트:** 2025-08-25