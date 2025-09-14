# LLM FastAPI Server - API 명세서

## 📋 개요

이 문서는 LLM FastAPI Server의 모든 API 엔드포인트와 사용법을 상세히 설명합니다.


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
  "description": "3개의 고성능 로컬 언어 모델을 지원하는 AI 서버",
  "features": [
    "3개 고성능 LLM 모델 지원 (Qwen 2.5, Llama 3.1, Gemma 2)",
    "한국어, 코딩, 다국어 지원",
    "RAG (검색 증강 생성) 기능",
    "실시간 모델 전환",
    "GPU 메모리 최적화"
  ],
  "endpoints": { ... },
  "supported_model_categories": [
    "medium (7-9B) - 현재 지원되는 모든 모델"
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

### 1. RAG 질의
```http
POST /api/v1/rag
```

**요청 본문:**
```json
{
  "question": "Python의 장점은 무엇인가요?",
  "model_key": "qwen2.5-7b"
}
```

**매개변수:**
- `question` (string, 필수): 검색할 질의
- `model_key` (string, 선택): 사용할 모델 키

### 2. RAG 데이터베이스 업데이트
```http
POST /api/v1/rag/update-news
```

**요청 본문:**
```json
{
  "query": "Python 최신 뉴스",
  "max_results": 5
}
```

**매개변수:**
- `query` (string, 필수): 검색할 쿼리
- `max_results` (integer, 기본값: 5): 최대 결과 개수

**RAG 질의 응답:**
```json
{
  "response": "Python의 주요 장점들은 다음과 같습니다...",
  "question": "Python의 장점은 무엇인가요?",
  "relevant_documents": [
    {
      "content": "Python은 가독성이 뛰어난 언어입니다...",
      "score": 0.95
    }
  ],
  "model_info": {
    "model_key": "qwen2.5-7b",
    "model_id": "Qwen/Qwen2.5-7B-Instruct",
    "loaded": true
  }
}
```

**RAG 업데이트 응답:**
```json
{
  "message": "5개의 문서가 성공적으로 추가되었습니다.",
  "added_chunks": 12
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
  },
  "total_models": 3
}
```

### 2. 모델 카테고리 조회
```http
GET /api/v1/models/categories
```

**응답:**
```json
{
  "categories": ["medium"],
  "models_by_category": {
    "medium": ["qwen2.5-7b", "llama3.1-8b", "gemma-3-4b"]
  },
  "category_descriptions": {
    "medium": "7-20GB RAM, 성능과 효율의 균형"
  }
}
```

### 3. 카테고리별 모델 조회
```http
GET /api/v1/models/category/{category}
```

**예시:** `/api/v1/models/category/medium`

**응답:**
```json
{
  "category": "medium",
  "models": {
    "qwen2.5-7b": {
      "model_id": "Qwen/Qwen2.5-7B-Instruct",
      "description": "Qwen 2.5 7B - 고성능 범용 모델"
    },
    "llama3.1-8b": {
      "model_id": "meta-llama/Meta-Llama-3-8B-Instruct", 
      "description": "Meta Llama 3 8B - 고성능 모델"
    },
    "gemma-3-4b": {
      "model_id": "google/gemma-2-9b-it",
      "description": "Google Gemma 2 9B - 효율적인 중형 모델"
    }
  },
  "count": 3
}
```

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

## 📰 뉴스 API (NEW!)

### 1. 최신 뉴스 조회
```http
GET /api/v1/news/latest
```

**쿼리 파라미터:**
- `categories` (string, 선택): 쉼표로 구분된 카테고리 (예: "technology,economy")
- `max_results` (integer, 기본값: 10): 최대 결과 수
- `time_range` (string, 기본값: "d"): 시간 범위 (d=1일, w=1주, m=1달)

**예시:**
```
GET /api/v1/news/latest?categories=technology,economy&max_results=5
```

**응답:**
```json
{
  "news": [
    {
      "title": "AI 기술 최신 동향",
      "url": "https://example.com/ai-news",
      "content": "AI 기술이 급속도로 발전하고 있습니다...",
      "category": "technology",
      "published_date": "2025-01-20T10:00:00Z",
      "score": 0.95
    }
  ],
  "total_count": 5,
  "categories": ["technology", "economy"],
  "time_range": "d",
  "status": "success"
}
```

### 2. 뉴스 검색
```http
GET /api/v1/news/search
```

**쿼리 파라미터:**
- `query` (string, 필수): 검색 키워드
- `max_results` (integer, 기본값: 5): 최대 결과 수
- `category` (string, 선택): 검색할 카테고리
- `time_range` (string, 기본값: "d"): 시간 범위

**예시:**
```
GET /api/v1/news/search?query=ChatGPT&category=technology&max_results=3
```

### 3. AI 뉴스 요약
```http
POST /api/v1/news/summary
```

**요청 본문:**
```json
{
  "query": "인공지능 ChatGPT",
  "max_results": 5,
  "summary_type": "comprehensive",
  "model_key": "qwen2.5-7b"
}
```

**매개변수:**
- `query` (string, 필수): 요약할 뉴스 주제
- `max_results` (integer, 기본값: 5): 분석할 뉴스 개수
- `summary_type` (string, 기본값: "comprehensive"): 요약 타입
  - `"brief"`: 간단 요약 (2-3문장)
  - `"comprehensive"`: 포괄적 요약 (구조화된 상세 요약)
  - `"analysis"`: 심층 분석 (전문적 분석)
- `model_key` (string, 선택): 사용할 LLM 모델

**응답:**
```json
{
  "summary": "## 📰 주요 내용 요약\nChatGPT 관련 최신 뉴스를 분석한 결과...\n\n## 🔍 세부 분석\n• 주요 이슈: AI 기술 발전\n• 관련 인물/기관: OpenAI, Microsoft\n• 영향/결과: 업계 변화 가속화",
  "articles": [
    {
      "title": "ChatGPT 최신 업데이트",
      "content": "ChatGPT가 새로운 기능을 추가했습니다...",
      "url": "https://example.com/chatgpt-update"
    }
  ],
  "query": "인공지능 ChatGPT",
  "summary_type": "comprehensive",
  "total_articles": 5,
  "model_info": {
    "model_key": "qwen2.5-7b",
    "model_id": "Qwen/Qwen2.5-7B-Instruct"
  },
  "status": "success"
}
```

### 4. 뉴스 트렌드 분석
```http
POST /api/v1/news/analysis
```

**요청 본문:**
```json
{
  "categories": ["politics", "economy", "technology"],
  "max_results": 20,
  "time_range": "d",
  "model_key": "qwen2.5-7b"
}
```

**응답:**
```json
{
  "overall_trend": "## 🔥 오늘의 주요 트렌드\n1. AI 기술 발전 가속화\n2. 경제 회복 신호\n\n## 📊 분야별 동향\n• 정치: 정책 변화 논의\n• 경제: 시장 회복세\n• 기술: AI 혁신 지속",
  "category_trends": {
    "politics": "정치권에서 AI 규제 논의가 활발해지고 있습니다.",
    "economy": "기술주 중심으로 시장이 회복세를 보이고 있습니다.",
    "technology": "AI 기술 발전이 각종 산업에 미치는 영향이 커지고 있습니다."
  },
  "total_articles_analyzed": 18,
  "categories": ["politics", "economy", "technology"],
  "time_range": "d",
  "status": "success"
}
```

### 5. 뉴스 카테고리 조회
```http
GET /api/v1/news/categories
```

**응답:**
```json
{
  "categories": {
    "politics": "정치",
    "economy": "경제",
    "technology": "기술/IT",
    "sports": "스포츠",
    "health": "건강/의료",
    "culture": "문화/예술",
    "society": "사회",
    "international": "국제/해외"
  },
  "supported_time_ranges": {
    "d": "1일",
    "w": "1주",
    "m": "1달"
  },
  "supported_summary_types": {
    "brief": "간단 요약",
    "comprehensive": "포괄적 요약", 
    "analysis": "심층 분석"
  },
  "status": "success"
}
```

---

## 🌐 External-Web RAG API

### 1. 자동 RAG (추천)
```http
POST /api/v1/external-web/auto-rag
```

**설명:** 질의에 대해 자동으로 웹 검색하고 벡터 DB화 한 후 RAG 응답을 생성합니다. 가장 편리한 방법입니다.

**요청 본문:**
```json
{
  "query": "삼성전자 AI 반도체 최신 동향",
  "max_results": 15,
  "model_key": "qwen2.5-7b"
}
```

**매개변수:**
- `query` (string, 필수): 사용자의 질의
- `max_results` (integer, 기본값: 10): 검색할 최대 뉴스 수 (5-25)
- `model_key` (string, 선택): 사용할 모델

**응답 (스트리밍):**
```
data: {"status": "starting", "message": "삼성전자 AI 반도체 최신 동향 관련 자동 RAG 처리를 시작합니다...", "progress": 5}

data: {"status": "searching", "message": "웹에서 관련 뉴스를 검색하는 중...", "progress": 20}

data: {"status": "vectorizing", "message": "12개의 뉴스 기사를 벡터 DB에 저장 완료", "progress": 50}

data: {"status": "generating", "message": "AI가 종합적인 답변을 생성하는 중...", "progress": 70}

data: {"status": "finalizing", "message": "관련 문서 정보를 정리하는 중...", "progress": 90}

data: {"status": "completed", "response": "삼성전자의 AI 반도체 최신 동향을 분석한 결과...", "added_chunks": 12, "relevant_documents": [...], "progress": 100}
```

### 2. 주제 업로드
```http
POST /api/v1/external-web/upload-topic
```

**설명:** 특정 주제에 대한 웹 정보를 미리 수집하여 벡터 DB에 저장합니다.

**요청 본문:**
```json
{
  "topic": "인공지능 ChatGPT",
  "max_results": 20
}
```

### 3. RAG 질의응답
```http
POST /api/v1/external-web/rag-query
```

**설명:** 이미 업로드된 주제에 대해 RAG 기반 질의응답을 수행합니다.

**요청 본문:**
```json
{
  "prompt": "ChatGPT의 최신 기능은 무엇인가요?",
  "top_k": 5,
  "model_key": "qwen2.5-7b"
}
```

---

## 🗄️ Internal-DB RAG API

### 1. 테이블 목록 조회
```http
GET /api/v1/internal-db/tables
```

### 2. 테이블 인제스트
```http
POST /api/v1/internal-db/ingest
```

**요청 본문:**
```json
{
  "table": "knowledge",
  "save_name": "knowledge",
  "simulate": true,
  "id_col": "id",
  "title_col": "term",
  "text_cols": ["description", "role"]
}
```

### 3. DB RAG 질의응답
```http
POST /api/v1/internal-db/query
```

**요청 본문:**
```json
{
  "save_name": "knowledge",
  "question": "Self-Attention은 무엇인가?",
  "top_k": 5,
  "margin": 0.12
}
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
  "current_device": 0,
  "gpu_info": [
    {
      "device_id": 0,
      "name": "NVIDIA GeForce RTX 3080",
      "total_memory_gb": 8.0,
      "allocated_memory_gb": 2.5,
      "cached_memory_gb": 3.2,
      "free_memory_gb": 4.8,
      "compute_capability": "8.6"
    }
  ]
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

### 🆕 새로운 뉴스 기능 사용 예시

#### Python으로 뉴스 요약 API 호출
```python
import requests

# AI 뉴스 요약 요청
response = requests.post("http://localhost:8001/api/v1/news/summary", 
    json={
        "query": "ChatGPT 인공지능",
        "max_results": 5,
        "summary_type": "comprehensive",
        "model_key": "qwen2.5-7b"
    }
)

result = response.json()
print("📰 뉴스 요약:")
print(result["summary"])
print(f"\n📊 분석 기사 수: {result['total_articles']}")
```

#### 최신 뉴스 조회
```python
# 기술/경제 카테고리 최신 뉴스 조회
response = requests.get("http://localhost:8001/api/v1/news/latest", 
    params={
        "categories": "technology,economy",
        "max_results": 8,
        "time_range": "d"
    }
)

news_data = response.json()
print(f"📰 총 {news_data['total_count']}개 최신 뉴스:")
for news in news_data["news"]:
    print(f"• {news['title']} ({news['category']})")
```

#### 뉴스 트렌드 분석
```python
# 오늘의 뉴스 트렌드 분석
response = requests.post("http://localhost:8001/api/v1/news/analysis",
    json={
        "categories": ["politics", "economy", "technology"],
        "max_results": 15,
        "time_range": "d"
    }
)

analysis = response.json()
print("🔥 오늘의 뉴스 트렌드:")
print(analysis["overall_trend"])
```

### Python으로 기존 API 호출
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