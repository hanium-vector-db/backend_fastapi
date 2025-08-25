import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import gradio as gr
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from contextlib import asynccontextmanager
from api.routes import router
from gradio_app import gradio_ui  # Gradio UI 임포트
import logging

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Starting LLM FastAPI Server...")
    logger.info("Gradio UI is available at http://<your-ip>:8001/ui")
    logger.info("Server is ready!")
    yield
    # Shutdown
    logger.info("Shutting down LLM FastAPI Server...")


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="LLM FastAPI Server",
    description="FastAPI server for LLM, embedding, and RAG functionality",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api/v1")

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
                "streaming_ui": "/stream"
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

# Gradio UI를 FastAPI 앱에 마운트
app = gr.mount_gradio_app(app, gradio_ui, path="/ui")



if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8001, reload=True)