import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.config_loader import config

# GPU 설정: 설정 파일에서 읽어옴
os.environ["CUDA_VISIBLE_DEVICES"] = config.get_backend_config('gpu', 'visible_devices')

import gradio as gr
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from contextlib import asynccontextmanager
from api.routes import router
from gradio_app import gradio_ui  # Gradio UI 임포트
import logging

# (추가) 가격 예측 라우터 import
from api.routers import price_forecast

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Starting LLM FastAPI Server...")
    logger.info("Gradio UI is available at http://<your-ip>:8001/ui")
    logger.info("Server is ready!")
    yield
    # Shutdown
    logger.info("Shutting down LLM FastAPI Server...")


# Configure logging from config
logging_config = config.get_backend_config('logging')
logging.basicConfig(
    level=getattr(logging, logging_config['level']),
    format=logging_config['format']
)
logger = logging.getLogger(__name__)

# FastAPI app with config values
server_config = config.get_backend_config('server')
ui_server_config = config.ui_server_config

app = FastAPI(
    title=ui_server_config['title'],
    description=ui_server_config['description'],
    version=ui_server_config['version'],
    lifespan=lifespan
)

# Add CORS middleware from config
security_config = config.get_backend_config('security')
app.add_middleware(
    CORSMiddleware,
    allow_origins=security_config['cors']['allow_origins'],
    allow_credentials=security_config['cors']['allow_credentials'],
    allow_methods=security_config['cors']['allow_methods'],
    allow_headers=security_config['cors']['allow_headers'],
)

app.include_router(router, prefix="/api/v1")
# ✅ (추가) 가격 예측 모듈 라우터 등록
# - /api/v1/price-forecast 엔드포인트에서 예측 결과 제공
app.include_router(price_forecast.router, prefix="/api/v1/price-forecast", tags=["Price Forecast"]

@app.get("/")
def read_root():
    return {
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
        "endpoints": {
            "기본 기능": {
                "generate": "/api/v1/generate",
                "chat": "/api/v1/chat", 
                "embed": "/api/v1/embed",
                "rag": "/api/v1/rag",
                "health": "/api/v1/health"
            },
            "UI 인터페이스": {
                "gradio_ui": "/ui",
                "streaming_ui": "/stream",
                "voice_chat": "/voice",
                "streaming_voice": "/streaming-voice"
            },
            "모델 관리": {
                "models": "/api/v1/models",
                "categories": "/api/v1/models/categories",
                "category_models": "/api/v1/models/category/{category}",
                "recommend": "/api/v1/models/recommend",
                "compare": "/api/v1/models/compare",
                "search": "/api/v1/models/search",
                "stats": "/api/v1/models/stats",
                "switch": "/api/v1/models/switch",
                "model_info": "/api/v1/models/info/{model_key}"
            },
            "시스템 정보": {
                "gpu": "/api/v1/system/gpu"
            },
            "뉴스 기능 (Tavily)": {
                "latest_news": "/api/v1/news/latest",
                "search_news": "/api/v1/news/search",
                "news_summary": "/api/v1/news/summary",
                "news_analysis": "/api/v1/news/analysis",
                "news_categories": "/api/v1/news/categories"
            },
            "뉴스 기능 (RSS)": {
                "trending_keywords": "/api/v1/news-rss/keywords",
                "news_articles": "/api/v1/news-rss/articles",
                "rss_categories": "/api/v1/news-rss/categories",
                "add_custom_keyword": "/api/v1/news-rss/keywords/custom",
                "user_keywords": "/api/v1/news-rss/keywords/user",
                "delete_keyword": "/api/v1/news-rss/keywords/custom"
            },
            "재정 관리": {
                "list_items": "/api/v1/finance/items",
                "create_item": "/api/v1/finance/items",
                "get_item": "/api/v1/finance/items/{item_id}",
                "update_item": "/api/v1/finance/items/{item_id}",
                "delete_item": "/api/v1/finance/items/{item_id}"
            },
            "External-Web RAG": {
                "auto_rag": "/api/v1/external-web/auto-rag",
                "upload_topic": "/api/v1/external-web/upload-topic",
                "rag_query": "/api/v1/external-web/rag-query"
            },
            "Internal-DB RAG": {
                "db_tables": "/api/v1/internal-db/tables",
                "ingest_table": "/api/v1/internal-db/ingest",
                "db_query": "/api/v1/internal-db/query",
                "db_status": "/api/v1/internal-db/status"
            },
            "음성 기능 (NEW!)": {
                "text_to_speech": "/api/v1/speech/text-to-speech",
                "speech_to_text": "/api/v1/speech/speech-to-text",
                "voice_chat": "/api/v1/speech/voice-chat",
                "full_voice_chat": "/api/v1/speech/full-voice-chat",
                "speech_languages": "/api/v1/speech/languages",
                "speech_status": "/api/v1/speech/status"
            },
            "실시간 스트리밍 TTS (NEW!)": {
                "streaming_generate_with_voice": "/api/v1/speech/streaming-generate-with-voice",
                "sentences_to_speech": "/api/v1/speech/sentences-to-speech",
                "text_to_sentences_and_speech": "/api/v1/speech/text-to-sentences-and-speech",
                "streaming_tts_status": "/api/v1/speech/streaming-tts/status"
            },
            "문서": {
                "docs": "/docs",
                "redoc": "/redoc"
            }
        },
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

# 정적 파일 서빙
static_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "static")
if os.path.exists(static_path):
    app.mount("/static", StaticFiles(directory=static_path), name="static")

# 스트리밍 페이지 라우트
@app.get("/stream")
async def streaming_page():
    streaming_file = os.path.join(static_path, "streaming.html")
    if os.path.exists(streaming_file):
        return FileResponse(streaming_file)
    else:
        return {"error": "Streaming page not found"}

# 음성 채팅 페이지 라우트
@app.get("/voice")
async def voice_chat_page():
    voice_file = os.path.join(static_path, "voice_chat.html")
    if os.path.exists(voice_file):
        return FileResponse(voice_file)
    else:
        return {"error": "Voice chat page not found"}

# 실시간 스트리밍 음성 페이지 라우트
@app.get("/streaming-voice")
async def streaming_voice_page():
    streaming_voice_file = os.path.join(static_path, "streaming_voice.html")
    if os.path.exists(streaming_voice_file):
        return FileResponse(streaming_voice_file)
    else:
        return {"error": "Streaming voice page not found"}

# Gradio UI를 FastAPI 앱에 마운트
app = gr.mount_gradio_app(app, gradio_ui, path="/ui")



if __name__ == "__main__":
    import uvicorn
    # Use config values for server
    uvicorn.run(
        "main:app",
        host=server_config['host'],
        port=server_config['port'],
        reload=True  # 개발 시 자동 리로드 활성화

    )
