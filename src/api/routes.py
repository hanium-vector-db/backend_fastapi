import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.config import settings
from fastapi import APIRouter, HTTPException, Query, Header
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Optional
from models.llm_handler import LLMHandler
from models.embedding_handler import EmbeddingHandler
from services.rag_service import RAGService
from services.enhanced_internal_db_service import EnhancedInternalDBService
from services.speech_service import SpeechService
from services.streaming_tts_service import StreamingTTSService, SentenceBasedTTSService
from services.news_service_rss import NewsServiceRSS
from services.finance_service import FinanceService
from services.db_llm_service import DBLLMService
from services.grocery_rag_service import GroceryRAGService
from services.yahoo_finance_service import YahooFinanceService
from models.news_models import NewsKeyword, NewsArticle, NewsResponse
from models.finance import FinanceItem
from utils.config_loader import config
from tools.tool_executor import ToolExecutor
from tools.tool_calling_wrapper import ToolCallingWrapper
import logging
import torch
import json
import asyncio
import aiomysql
from fastapi import UploadFile, File
from datetime import datetime, timedelta
import uuid
import jwt

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter()

# Request models
class GenerateRequest(BaseModel):
    prompt: str
    max_length: int = 512
    model_key: str = None  # 사용할 모델 키
    stream: bool = False  # 스트리밍 여부

class EmbeddingRequest(BaseModel):
    text: str

class RAGRequest(BaseModel):
    question: str
    model_key: str = None  # 사용할 모델 키

class RAGUpdateRequest(BaseModel):
    query: str
    max_results: int = 5


class ChatRequest(BaseModel):
    message: str
    model_key: str = None  # 사용할 모델 키
    stream: bool = False  # 스트리밍 여부

class ChatWithToolsRequest(BaseModel):
    message: str
    user_id: Optional[str] = None  # 사용자 ID (JWT에서 자동 추출, 직접 전달 시 무시됨)
    model_key: str = None  # 사용할 모델 키
    stream: bool = False  # 스트리밍 여부

class ModelSwitchRequest(BaseModel):
    model_key: str  # 전환할 모델 키

# Request models for new endpoints
class ModelRecommendationRequest(BaseModel):
    ram_gb: int = None
    gpu_gb: int = None
    use_case: str = None  # korean, coding, math, multilingual

class PerformanceComparisonRequest(BaseModel):
    model_keys: list[str] = None

# 새로운 뉴스 관련 요청 모델들
class NewsSummaryRequest(BaseModel):
    query: str
    max_results: int = 5
    summary_type: str = "comprehensive"  # brief, comprehensive, analysis
    model_key: str = None

class NewsAnalysisRequest(BaseModel):
    categories: list[str] = None
    max_results: int = 20
    time_range: str = 'd'
    model_key: str = None

class LatestNewsRequest(BaseModel):
    categories: list[str] = None
    max_results: int = 10
    time_range: str = 'd'

# External-Web RAG 요청 모델들
class ExternalWebUploadRequest(BaseModel):
    topic: str
    max_results: int = 20

class ExternalWebQueryRequest(BaseModel):
    prompt: str
    top_k: int = 5
    model_key: str = None

class ExternalWebAutoRAGRequest(BaseModel):
    query: str
    max_results: int = 10
    model_key: str = None

# Internal-DBMS RAG 요청 모델들
class InternalDBIngestRequest(BaseModel):
    table: str = "knowledge"
    save_name: str = "knowledge"
    simulate: bool = False
    id_col: str = None
    title_col: str = None
    text_cols: list[str] = None

class InternalDBQueryRequest(BaseModel):
    save_name: str = "knowledge"
    question: str
    top_k: int = 5
    margin: float = 0.12

# 음성 관련 요청 모델들
class TextToSpeechRequest(BaseModel):
    text: str
    language: str = "ko"
    slow: bool = False

class SpeechToTextRequest(BaseModel):
    prefer_whisper: bool = True

class VoiceChatRequest(BaseModel):
    text: str = None  # 텍스트 입력 (음성 대신)
    user_id: str = "default_user"  # 사용자 ID (tool calling용)
    model_key: str = None
    voice_language: str = "ko"
    voice_slow: bool = False

# 실시간 스트리밍 TTS 요청 모델들
class StreamingTTSRequest(BaseModel):
    prompt: str
    model_key: str = None
    voice_language: str = "ko"
    voice_slow: bool = False
    read_partial: bool = True  # 부분 문장도 읽을지 여부

class SentencesTTSRequest(BaseModel):
    sentences: List[str]
    language: str = "ko"
    slow: bool = False

# DB LLM 연동 요청 모델
class ChatWithContextRequest(BaseModel):
    message: str
    user_id: str  # 사용자 ID
    model_key: str = None
    stream: bool = False

# Health 관련 요청 모델
class DiseaseRequest(BaseModel):
    name: str
    diagnosedDate: str = None
    status: str = "Active"

class MedicationRequest(BaseModel):
    name: str
    dosage: str = None
    intakeTime: str = None
    alarmEnabled: bool = True

# Initialize handlers (lazy loading)
llm_handler = None
embedding_handler = None
rag_service = None
internal_db_service = None
speech_service = None
streaming_tts_service = None
sentence_tts_service = None
news_service_rss = None
finance_service = None
yahoo_finance_service = None
db_llm_service = None
tool_executor = None
grocery_rag_service = None

def get_llm_handler(model_key: str = None):
    global llm_handler
    
    # 모델 키가 지정되었고 현재 핸들러와 다른 경우 새로 생성
    if model_key and (llm_handler is None or getattr(llm_handler, 'model_key', None) != model_key):
        logger.info(f"모델 전환 중: {model_key}")
        llm_handler = LLMHandler(model_key=model_key)
    elif llm_handler is None:
        logger.info("기본 LLM 핸들러 초기화 중...")
        llm_handler = LLMHandler()
    
    return llm_handler

def get_embedding_handler():
    global embedding_handler
    if embedding_handler is None:
        logger.info("Initializing embedding handler...")
        embedding_handler = EmbeddingHandler()
    return embedding_handler

def get_rag_service(model_key: str = None):
    global rag_service
    if rag_service is None or model_key:
        logger.info("RAG 서비스 초기화 중...")
        llm = get_llm_handler(model_key)
        emb = get_embedding_handler()
        rag_service = RAGService(llm, emb)
    return rag_service

def get_internal_db_service():
    global internal_db_service
    if internal_db_service is None:
        logger.info("Enhanced Internal DB 서비스 초기화 중...")
        llm = get_llm_handler()
        emb = get_embedding_handler()
        internal_db_service = EnhancedInternalDBService(llm, emb)
    return internal_db_service

def get_speech_service():
    global speech_service
    if speech_service is None:
        logger.info("음성 서비스 초기화 중...")
        speech_service = SpeechService()
    return speech_service

def get_streaming_tts_service():
    global streaming_tts_service
    if streaming_tts_service is None:
        logger.info("스트리밍 TTS 서비스 초기화 중...")
        speech_svc = get_speech_service()
        streaming_tts_service = StreamingTTSService(speech_svc)
    return streaming_tts_service

def get_sentence_tts_service():
    global sentence_tts_service
    if sentence_tts_service is None:
        logger.info("문장 기반 TTS 서비스 초기화 중...")
        speech_svc = get_speech_service()
        sentence_tts_service = SentenceBasedTTSService(speech_svc)
    return sentence_tts_service

def get_news_service_rss():
    global news_service_rss
    if news_service_rss is None:
        logger.info("RSS 뉴스 서비스 초기화 중...")
        news_service_rss = NewsServiceRSS()
    return news_service_rss

def get_finance_service():
    global finance_service
    if finance_service is None:
        logger.info("재정 관리 서비스 초기화 중...")
        finance_service = FinanceService()
    return finance_service

def get_yahoo_finance_service():
    global yahoo_finance_service
    if yahoo_finance_service is None:
        logger.info("Yahoo Finance 서비스 초기화 중...")
        yahoo_finance_service = YahooFinanceService()
    return yahoo_finance_service

async def get_db_llm_service():
    global db_llm_service
    if db_llm_service is None:
        logger.info("DB LLM 서비스 초기화 중...")
        db_config = config.mariadb_config
        db_llm_service = DBLLMService(db_config)
        await db_llm_service.initialize()
    return db_llm_service

def get_grocery_rag_service():
    """Grocery RAG 서비스 초기화"""
    global grocery_rag_service
    if grocery_rag_service is None:
        logger.info("Grocery RAG 서비스 초기화 중...")
        rag_svc = get_rag_service()
        grocery_rag_service = GroceryRAGService(rag_svc)
        # 식료품 데이터를 RAG에 로드
        grocery_rag_service.initialize_grocery_rag()
        logger.info("Grocery RAG 데이터 로드 완료")
    return grocery_rag_service

async def get_tool_executor():
    """Tool Executor 초기화"""
    global tool_executor
    if tool_executor is None:
        logger.info("Tool Executor 초기화 중...")
        db_config = config.mariadb_config
        # Grocery RAG 서비스 초기화
        grocery_rag_svc = get_grocery_rag_service()
        tool_executor = ToolExecutor(db_config, grocery_rag_service=grocery_rag_svc)
        await tool_executor.initialize()
    return tool_executor

@router.post("/generate")
async def generate_response(request: GenerateRequest):
    try:
        handler = get_llm_handler(request.model_key)
        
        if request.stream:
            # 스트리밍 응답 (SSE 형식)
            async def generate_stream():
                import asyncio
                for chunk in handler.generate(request.prompt, request.max_length, stream=True):
                    if chunk:
                        # SSE 형식으로 데이터 포맷팅
                        data = json.dumps({"content": chunk, "done": False}, ensure_ascii=False)
                        yield f"data: {data}\n\n"
                        await asyncio.sleep(0)  # 즉시 yield하도록 함
                # 완료 신호
                final_data = json.dumps({"content": "", "done": True})
                yield f"data: {final_data}\n\n"
            
            return StreamingResponse(
                generate_stream(), 
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no"  # nginx 버퍼링 방지
                }
            )
        else:
            # 일반 응답
            response = handler.generate(request.prompt, request.max_length, stream=False)
            return {
                "response": response, 
                "prompt": request.prompt,
                "model_info": {
                    "model_key": handler.model_key,
                    "model_id": handler.SUPPORTED_MODELS[handler.model_key]["model_id"],
                    "description": handler.SUPPORTED_MODELS[handler.model_key]["description"],
                    "category": handler.SUPPORTED_MODELS[handler.model_key]["category"],
                    "loaded": handler.model is not None
                }
            }
    except Exception as e:
        logger.error(f"생성 엔드포인트 오류: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/chat")
async def chat_response(request: ChatRequest):
    try:
        handler = get_llm_handler(request.model_key)
        
        if request.stream:
            # 스트리밍 응답 (SSE 형식)
            async def chat_stream():
                for chunk in handler.chat_generate(request.message, stream=True):
                    if chunk:
                        # SSE 형식으로 데이터 포맷팅
                        data = json.dumps({"content": chunk, "done": False})
                        yield f"data: {data}\n\n"
                # 완료 신호
                final_data = json.dumps({"content": "", "done": True})
                yield f"data: {final_data}\n\n"
            
            return StreamingResponse(
                chat_stream(), 
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive"
                }
            )
        else:
            # 일반 응답
            response = handler.chat_generate(request.message, stream=False)
            return {
                "response": response, 
                "message": request.message,
                "model_info": {
                    "model_key": handler.model_key,
                    "model_id": handler.SUPPORTED_MODELS[handler.model_key]["model_id"],
                    "description": handler.SUPPORTED_MODELS[handler.model_key]["description"],
                    "category": handler.SUPPORTED_MODELS[handler.model_key]["category"],
                    "loaded": handler.model is not None
                }
            }
    except Exception as e:
        logger.error(f"채팅 엔드포인트 오류: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/chat-with-tools")
async def chat_with_tools(request: ChatWithToolsRequest, authorization: str = Header(None)):
    """툴콜링을 지원하는 채팅 엔드포인트 (스트리밍 지원)"""
    try:
        # JWT 토큰에서 user_id 추출 (필수)
        user_id = None

        if authorization and authorization.startswith("Bearer "):
            token = authorization.replace("Bearer ", "")
            try:
                payload = jwt.decode(token, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])
                # user_id가 없으면 userid를 fallback으로 사용 (하위 호환성)
                user_id = payload.get("user_id") or payload.get("userid")

                if not user_id:
                    raise HTTPException(status_code=401, detail="유효하지 않은 토큰입니다")

                logger.info(f"JWT 인증 성공, user_id: {user_id}")
            except jwt.ExpiredSignatureError:
                logger.warning("JWT 토큰이 만료되었습니다")
                raise HTTPException(status_code=401, detail="토큰이 만료되었습니다. 다시 로그인해주세요.")
            except jwt.InvalidTokenError as e:
                logger.warning(f"JWT 토큰이 유효하지 않습니다: {e}")
                raise HTTPException(status_code=401, detail="유효하지 않은 토큰입니다")
        else:
            logger.error("JWT 토큰이 없습니다")
            raise HTTPException(status_code=401, detail="인증이 필요합니다. 로그인해주세요.")

        # LLM 핸들러 가져오기
        handler = get_llm_handler(request.model_key)

        # Tool Executor 가져오기
        executor = await get_tool_executor()

        # ToolCallingWrapper로 래핑
        wrapper = ToolCallingWrapper(handler, executor)

        if request.stream:
            # 스트리밍 응답 (SSE 형식)
            async def chat_stream():
                response_text = await wrapper.generate_with_tools(request.message, user_id)
                # 전체 응답을 청크로 나눠서 스트리밍
                chunk_size = 10  # 한 번에 보낼 글자 수
                for i in range(0, len(response_text), chunk_size):
                    chunk = response_text[i:i + chunk_size]
                    data = json.dumps({"content": chunk, "done": False}, ensure_ascii=False)
                    yield f"data: {data}\n\n"
                    await asyncio.sleep(0.05)  # 약간의 지연으로 스트리밍 효과

                # 완료 신호
                final_data = json.dumps({"content": "", "done": True})
                yield f"data: {final_data}\n\n"

            return StreamingResponse(
                chat_stream(),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive"
                }
            )
        else:
            # 일반 응답
            response_text = await wrapper.generate_with_tools(request.message, user_id)
            return {
                "response": response_text,
                "message": request.message,
                "user_id": user_id,
                "tools_used": True
            }
    except Exception as e:
        logger.error(f"툴콜링 채팅 엔드포인트 오류: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/embed")
async def create_embedding(request: EmbeddingRequest):
    try:
        handler = get_embedding_handler()
        embedding = handler.create_embedding(request.text)
        return {"embedding": embedding, "text": request.text, "dimension": len(embedding)}
    except Exception as e:
        logger.error(f"Error in embed endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/rag")
async def rag_response(request: RAGRequest):
    try:
        service = get_rag_service(request.model_key)
        response = service.generate_response(request.question)
        relevant_docs = service.get_relevant_documents(request.question)
        return {
            "response": response, 
            "question": request.question,
            "relevant_documents": relevant_docs,
            "model_info": {
                "model_key": service.llm_handler.model_key,
                "model_id": service.llm_handler.SUPPORTED_MODELS[service.llm_handler.model_key]["model_id"],
                "description": service.llm_handler.SUPPORTED_MODELS[service.llm_handler.model_key]["description"],
                "category": service.llm_handler.SUPPORTED_MODELS[service.llm_handler.model_key]["category"],
                "loaded": service.llm_handler.model is not None
            }
        }
    except Exception as e:
        logger.error(f"RAG 엔드포인트 오류: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/rag/update-news")
async def update_rag_with_news(request: RAGUpdateRequest):
    """
    웹에서 최신 뉴스를 검색하여 RAG 데이터베이스를 업데이트합니다.
    """
    try:
        service = get_rag_service() # 기본 RAG 서비스 인스턴스 가져오기
        added_chunks, message = service.add_documents_from_web(request.query, request.max_results)
        
        if added_chunks > 0:
            return {"message": message, "added_chunks": added_chunks}
        else:
            # 문서 추가에 실패했거나 찾지 못한 경우
            raise HTTPException(status_code=404, detail=message)
            
    except HTTPException as e:
        raise e # HTTP 예외는 그대로 전달
    except Exception as e:
        logger.error(f"뉴스 업데이트 엔드포인트 오류: {e}")
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")


@router.get("/health")
async def health_check():
    current_model_info = None
    if llm_handler:
        current_model_info = {
            "model_key": llm_handler.model_key,
            "model_id": llm_handler.SUPPORTED_MODELS[llm_handler.model_key]["model_id"],
            "description": llm_handler.SUPPORTED_MODELS[llm_handler.model_key]["description"],
            "category": llm_handler.SUPPORTED_MODELS[llm_handler.model_key]["category"],
            "loaded": llm_handler.model is not None
        }
    
    return {
        "status": "healthy",
        "llm_loaded": llm_handler is not None,
        "embedding_loaded": embedding_handler is not None,
        "rag_loaded": rag_service is not None,
        "current_model": current_model_info,
        "gpu_available": torch.cuda.is_available(),
        "gpu_memory": torch.cuda.get_device_properties(0).total_memory / 1024**3 if torch.cuda.is_available() else None
    }

@router.get("/models")
async def list_models():
    """지원되는 모델 목록 반환"""
    # config.py에서 정의된 모델 목록을 사용
    model_keys = settings.available_models
    
    # Gradio가 기대하는 형식으로 변환
    # 각 모델에 대한 상세 정보가 필요하다면 추가 로직이 필요하지만,
    # 현재는 키 목록만 반환하도록 수정
    models_dict = {key: {"model_id": key} for key in model_keys}
    
    return {
        "supported_models": models_dict,
        "total_models": len(model_keys)
    }

@router.get("/models/categories")
async def get_model_categories():
    """모델 카테고리 목록 반환"""
    try:
        categories = LLMHandler.get_model_categories()
        models_by_category = LLMHandler.get_models_by_category()
        
        return {
            "categories": categories,
            "models_by_category": models_by_category,
            "category_descriptions": {
                "ultra-light": "0.5GB 미만의 초경량 모델들",
                "light": "1-6GB RAM, 빠른 응답과 효율성 중심",
                "medium": "7-20GB RAM, 성능과 효율의 균형",
                "large": "30GB+ RAM, 최고 성능의 대형 모델",
                "korean": "한국어에 특화된 모델들",
                "code": "프로그래밍과 코딩에 특화된 모델들",
                "math": "수학과 과학 계산에 특화된 모델들",
                "multilingual": "다국어 지원에 강한 모델들"
            }
        }
    except Exception as e:
        logger.error(f"카테고리 조회 오류: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/models/category/{category}")
async def get_models_by_category(category: str):
    """특정 카테고리의 모델들 반환"""
    try:
        models = LLMHandler.get_models_by_category(category)
        if not models:
            raise HTTPException(status_code=404, detail=f"카테고리 '{category}'를 찾을 수 없습니다")
            
        return {
            "category": category,
            "models": models,
            "count": len(models)
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"카테고리별 모델 조회 오류: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/models/recommend")
async def recommend_models(request: ModelRecommendationRequest):
    """시스템 사양과 용도에 맞는 모델 추천"""
    try:
        recommendations = LLMHandler.recommend_model(
            ram_gb=request.ram_gb,
            gpu_gb=request.gpu_gb,
            use_case=request.use_case
        )
        
        return {
            "recommendations": recommendations,
            "criteria": {
                "ram_gb": request.ram_gb,
                "gpu_gb": request.gpu_gb,
                "use_case": request.use_case
            },
            "total_recommendations": len(recommendations)
        }
    except Exception as e:
        logger.error(f"모델 추천 오류: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/models/compare")
async def compare_models(request: PerformanceComparisonRequest):
    """모델들의 성능 비교"""
    try:
        comparison = LLMHandler.get_performance_comparison(request.model_keys)
        
        return {
            "comparison": comparison,
            "total_models": len(comparison),
            "sorted_by": "performance_score"
        }
    except Exception as e:
        logger.error(f"모델 비교 오류: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/models/search")
async def search_models(
    category: str = None,
    min_ram: int = None,
    max_ram: int = None,
    min_gpu: int = None,
    max_gpu: int = None,
    keyword: str = None
):
    """모델 검색 및 필터링"""
    try:
        all_models = LLMHandler.get_supported_models()
        filtered_models = {}
        
        for key, model in all_models.items():
            # 카테고리 필터
            if category and model.get("category") != category:
                continue
                
            # RAM 요구사항 필터
            model_ram = int(model["ram_requirement"].replace("GB", ""))
            if min_ram and model_ram < min_ram:
                continue
            if max_ram and model_ram > max_ram:
                continue
                
            # GPU 요구사항 필터
            model_gpu = int(model["gpu_requirement"].replace("GB", ""))
            if min_gpu and model_gpu < min_gpu:
                continue
            if max_gpu and model_gpu > max_gpu:
                continue
                
            # 키워드 검색
            if keyword:
                search_text = f"{key} {model['description']} {model['model_id']}".lower()
                if keyword.lower() not in search_text:
                    continue
                    
            filtered_models[key] = model
        
        return {
            "models": filtered_models,
            "total_found": len(filtered_models),
            "filters": {
                "category": category,
                "min_ram": min_ram,
                "max_ram": max_ram,
                "min_gpu": min_gpu,
                "max_gpu": max_gpu,
                "keyword": keyword
            }
        }
    except Exception as e:
        logger.error(f"모델 검색 오류: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/models/stats")
async def get_model_statistics():
    """모델 통계 정보"""
    try:
        all_models = LLMHandler.get_supported_models()
        
        # 카테고리별 통계
        category_stats = {}
        ram_stats = {"min": float('inf'), "max": 0, "avg": 0}
        gpu_stats = {"min": float('inf'), "max": 0, "avg": 0}
        
        total_ram = 0
        total_gpu = 0
        
        for model in all_models.values():
            # 카테고리 통계
            category = model.get("category", "other")
            category_stats[category] = category_stats.get(category, 0) + 1
            
            # RAM/GPU 통계
            ram_req = int(model["ram_requirement"].replace("GB", ""))
            gpu_req = int(model["gpu_requirement"].replace("GB", ""))
            
            ram_stats["min"] = min(ram_stats["min"], ram_req)
            ram_stats["max"] = max(ram_stats["max"], ram_req)
            total_ram += ram_req
            
            gpu_stats["min"] = min(gpu_stats["min"], gpu_req)
            gpu_stats["max"] = max(gpu_stats["max"], gpu_req)
            total_gpu += gpu_req
        
        model_count = len(all_models)
        ram_stats["avg"] = round(total_ram / model_count, 2)
        gpu_stats["avg"] = round(total_gpu / model_count, 2)
        
        return {
            "total_models": model_count,
            "category_distribution": category_stats,
            "ram_requirements": ram_stats,
            "gpu_requirements": gpu_stats,
            "model_size_distribution": {
                "small (≤3B)": len([k for k in all_models.keys() if any(size in k for size in ["0.5b", "1b", "1.5b", "2b", "3b"])]),
                "medium (4-20B)": len([k for k in all_models.keys() if any(size in k for size in ["7b", "8b", "9b", "10b", "12b", "13b", "14b"])]),
                "large (>20B)": len([k for k in all_models.keys() if any(size in k for size in ["32b", "33b", "70b", "72b"])])
            }
        }
    except Exception as e:
        logger.error(f"모델 통계 조회 오류: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/models/info/{model_key}")
async def get_model_info_endpoint(model_key: str):
    """특정 모델 정보 반환"""
    model_info = LLMHandler.get_model_info(model_key)
    if not model_info:
        raise HTTPException(status_code=404, detail=f"모델 '{model_key}'를 찾을 수 없습니다")
    
    return {
        "model_key": model_key,
        "model_info": model_info
    }

@router.post("/models/switch")
async def switch_model(request: ModelSwitchRequest):
    """모델 전환"""
    global llm_handler, rag_service
    
    # 모델 키 유효성 검사
    if request.model_key not in LLMHandler.get_supported_models():
        raise HTTPException(status_code=400, detail=f"지원되지 않는 모델: {request.model_key}")
    
    try:
        # 기존 핸들러 정리
        old_model = None
        if llm_handler:
            old_model = {
                "model_key": llm_handler.model_key,
                "model_id": llm_handler.SUPPORTED_MODELS[llm_handler.model_key]["model_id"],
                "description": llm_handler.SUPPORTED_MODELS[llm_handler.model_key]["description"],
                "category": llm_handler.SUPPORTED_MODELS[llm_handler.model_key]["category"],
                "loaded": llm_handler.model is not None
            }
        
        # 새 모델로 전환
        logger.info(f"모델 전환 중: {request.model_key}")
        llm_handler = LLMHandler(model_key=request.model_key)
        rag_service = None  # RAG 서비스 재초기화 필요
        
        new_model = {
            "model_key": llm_handler.model_key,
            "model_id": llm_handler.SUPPORTED_MODELS[llm_handler.model_key]["model_id"],
            "description": llm_handler.SUPPORTED_MODELS[llm_handler.model_key]["description"],
            "category": llm_handler.SUPPORTED_MODELS[llm_handler.model_key]["category"],
            "loaded": llm_handler.model is not None
        }
        
        return {
            "message": "모델 전환 완료",
            "old_model": old_model,
            "new_model": new_model
        }
        
    except Exception as e:
        logger.error(f"모델 전환 오류: {e}")
        raise HTTPException(status_code=500, detail=f"모델 전환 실패: {str(e)}")

@router.get("/system/gpu")
async def get_gpu_info():
    """GPU 시스템 정보 반환"""
    if not torch.cuda.is_available():
        return {"gpu_available": False, "message": "CUDA를 사용할 수 없습니다"}
    
    gpu_info = []
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        memory_allocated = torch.cuda.memory_allocated(i) / 1024**3
        memory_cached = torch.cuda.memory_reserved(i) / 1024**3
        memory_total = props.total_memory / 1024**3
        
        gpu_info.append({
            "device_id": i,
            "name": props.name,
            "total_memory_gb": round(memory_total, 2),
            "allocated_memory_gb": round(memory_allocated, 2),
            "cached_memory_gb": round(memory_cached, 2),
            "free_memory_gb": round(memory_total - memory_cached, 2),
            "compute_capability": f"{props.major}.{props.minor}"
        })
    
    return {
        "gpu_available": True,
        "gpu_count": torch.cuda.device_count(),
        "current_device": torch.cuda.current_device(),
        "gpu_info": gpu_info
    }

# === 새로운 뉴스 관련 API 엔드포인트들 ===

@router.get("/news/latest")
async def get_latest_news(
    categories: str = None,  # 쉼표로 구분된 카테고리 문자열
    max_results: int = 10,
    time_range: str = 'd'
):
    """최신 뉴스를 조회합니다 (데이터베이스 저장 없이)"""
    try:
        from utils.helpers import search_latest_news
        
        # 카테고리 문자열을 리스트로 변환
        category_list = None
        if categories:
            category_list = [cat.strip() for cat in categories.split(',')]
        
        logger.info(f"최신 뉴스 조회 - 카테고리: {category_list}, 결과수: {max_results}")
        
        news_results = search_latest_news(
            max_results=max_results,
            categories=category_list,
            time_range=time_range
        )
        
        return {
            "news": news_results,
            "total_count": len(news_results),
            "categories": category_list or ["politics", "economy", "technology", "society"],
            "time_range": time_range,
            "status": "success"
        }
        
    except Exception as e:
        logger.error(f"최신 뉴스 조회 오류: {e}")
        raise HTTPException(status_code=500, detail=f"최신 뉴스 조회 실패: {str(e)}")

@router.post("/news/summary")
async def summarize_news(request: NewsSummaryRequest):
    """특정 주제의 뉴스를 검색하고 LLM으로 요약합니다 (스트리밍)"""
    
    async def generate_summary_stream():
        try:
            # 시작 신호
            yield f"data: {json.dumps({'status': 'starting', 'message': '뉴스 요약을 시작합니다...'}, ensure_ascii=False)}\n\n"
            
            # RAG 서비스 대신 직접 LLM 핸들러 사용
            llm = get_llm_handler(request.model_key)
            
            logger.info(f"뉴스 요약 요청 - 주제: {request.query}, 타입: {request.summary_type}")
            logger.debug(f"LLM 핸들러 로드 완료: {llm.model_key}")
            
            yield f"data: {json.dumps({'status': 'searching', 'message': 'Tavily로 뉴스를 검색하는 중...'}, ensure_ascii=False)}\n\n"
            
            # 직접 뉴스 검색 및 요약
            from utils.helpers import get_news_summary_with_tavily
            
            logger.debug("Tavily로 뉴스 검색 시작...")
            news_data = get_news_summary_with_tavily(request.query, request.max_results)
            logger.debug(f"뉴스 검색 완료. 수집된 기사 수: {len(news_data) if news_data else 0}")
            
            if not news_data:
                no_news_message = f"'{request.query}' 관련 최신 뉴스를 찾을 수 없습니다."
                final_data = {
                    'status': 'completed', 
                    'summary': no_news_message, 
                    'articles': [], 
                    'source_articles': [], 
                    'query': request.query, 
                    'summary_type': request.summary_type, 
                    'total_articles': 0
                }
                yield f"data: {json.dumps(final_data, ensure_ascii=False)}\n\n"
                return
            
            news_count = len(news_data)
            yield f"data: {json.dumps({'status': 'processing', 'message': f'{news_count}개의 뉴스 기사를 분석하는 중...'}, ensure_ascii=False)}\n\n"
            
            # 요약 타입별 프롬프트 선택
            summary_prompts = {
                "brief": get_brief_summary_prompt(),
                "comprehensive": get_comprehensive_summary_prompt(),
                "analysis": get_analysis_summary_prompt()
            }
            
            prompt_template = summary_prompts.get(request.summary_type, summary_prompts["comprehensive"])
            
            # 뉴스 데이터 준비
            articles_text = "\n\n".join([
                f"제목: {article.get('title', '')}\n출처: {article.get('url', 'Unknown')}\n내용: {article.get('content', '')[:1000]}"
                for article in news_data[:request.max_results]
                if not article.get('is_summary', False)  # Tavily의 자동 요약 제외
            ])
            
            # 참고한 뉴스 기사 정보 준비 (응답용)
            source_articles = []
            for article in news_data[:request.max_results]:
                if not article.get('is_summary', False):
                    source_articles.append({
                        'title': article.get('title', ''),
                        'url': article.get('url', ''),
                        'published_date': article.get('published_date', ''),
                        'score': article.get('score', 0)
                    })
            
            yield f"data: {json.dumps({'status': 'generating', 'message': 'LLM이 뉴스 요약을 생성하는 중...'}, ensure_ascii=False)}\n\n"
            
            # LLM으로 요약 생성
            logger.debug(f"프롬프트 템플릿 준비 완료. 기사 텍스트 길이: {len(articles_text)} 문자")
            full_prompt = prompt_template.format(query=request.query, articles=articles_text)
            logger.debug(f"전체 프롬프트 길이: {len(full_prompt)} 문자")
            
            logger.info("LLM으로 뉴스 요약 생성 중...")
            
            # 스트리밍 모드로 생성
            summary_stream = llm.generate(full_prompt, max_length=1024, stream=True)
            
            summary_parts = []
            for chunk in summary_stream:
                if chunk:
                    summary_parts.append(chunk)
                    yield f"data: {json.dumps({'status': 'streaming', 'chunk': chunk}, ensure_ascii=False)}\n\n"
                    await asyncio.sleep(0.01)  # 약간의 지연으로 스트리밍 효과
            
            summary = ''.join(summary_parts)
            logger.debug("LLM 요약 생성 완료")
            
            # 요약에 출처 정보 추가
            summary_with_sources = summary + "\n\n" + "📰 **참고 기사:**\n" + "\n".join([
                f"• [{article['title']}]({article['url']})" + (f" ({article['published_date']})" if article['published_date'] else "")
                for article in source_articles[:5]  # 최대 5개 출처만 표시
            ])
            
            # 완료 신호
            final_result = {
                "status": "completed",
                "summary": summary_with_sources,
                "articles": news_data[:request.max_results],
                "source_articles": source_articles,
                "query": request.query,
                "summary_type": request.summary_type,
                "total_articles": len(news_data),
                "model_info": {
                    "model_key": llm.model_key,
                    "model_id": llm.SUPPORTED_MODELS[llm.model_key]["model_id"]
                }
            }
            
            yield f"data: {json.dumps(final_result, ensure_ascii=False)}\n\n"
            
        except Exception as e:
            logger.error(f"뉴스 요약 오류: {e}")
            error_msg = f'뉴스 요약 실패: {str(e)}'
            yield f"data: {json.dumps({'status': 'error', 'message': error_msg}, ensure_ascii=False)}\n\n"
    
    return StreamingResponse(generate_summary_stream(), media_type="text/plain; charset=utf-8")

def get_brief_summary_prompt():
    """간단 요약용 프롬프트"""
    return """다음 뉴스 기사들을 바탕으로 '{query}' 주제에 대한 간단한 요약을 작성해주세요.

뉴스 기사들:
{articles}

요구사항:
1. 핵심 내용을 2-3문장으로 간단히 요약
2. 가장 중요한 포인트만 포함
3. 명확하고 이해하기 쉽게 작성
4. 한국어로 작성
5. 출처 정보는 별도로 제공되므로 요약 본문에는 포함하지 마세요

간단 요약:"""

def get_comprehensive_summary_prompt():
    """포괄적 요약용 프롬프트"""
    return """다음 뉴스 기사들을 바탕으로 '{query}' 주제에 대한 포괄적인 요약을 작성해주세요.

뉴스 기사들:
{articles}

다음 형식으로 작성해주세요:

## 📰 주요 내용 요약
(핵심 내용을 3-4문장으로 요약)

## 🔍 세부 분석
• 주요 이슈: 
• 관련 인물/기관:
• 영향/결과:

## 🏷️ 키워드
(관련 키워드 3-5개를 쉼표로 구분)

## 📊 종합 평가
(전반적인 상황 평가와 향후 전망 1-2문장)

모든 내용을 한국어로 작성해주세요. 출처 정보는 별도로 제공되므로 요약 본문에는 포함하지 마세요."""

def get_analysis_summary_prompt():
    """분석 중심 요약용 프롬프트"""
    return """다음 뉴스 기사들을 바탕으로 '{query}' 주제에 대한 심층 분석을 작성해주세요.

뉴스 기사들:
{articles}

다음 형식으로 분석해주세요:

## 🎯 핵심 이슈 분석
(가장 중요한 이슈와 그 배경)

## 📈 현황 및 트렌드
• 현재 상황:
• 변화 추이:
• 주목할 점:

## ⚡ 주요 동향
• 긍정적 요소:
• 우려사항:
• 예상 시나리오:

## 🌟 시사점 및 전망
(이 뉴스가 갖는 의미와 향후 예상되는 발전 방향)

## 🏷️ 핵심 키워드
(분석에 중요한 키워드 5-7개)

전문적이고 객관적인 시각으로 한국어로 작성해주세요. 출처 정보는 별도로 제공되므로 분석 본문에는 포함하지 마세요."""

@router.post("/news/analysis")
async def analyze_news_trends(request: NewsAnalysisRequest):
    """여러 카테고리의 뉴스를 분석하여 트렌드를 파악합니다 (스트리밍)"""
    
    async def generate_analysis_stream():
        try:
            from utils.helpers import search_news
            
            # 시작 신호
            yield f"data: {json.dumps({'status': 'starting', 'message': '뉴스 트렌드 분석을 시작합니다...'}, ensure_ascii=False)}\n\n"
            
            # LLM 핸들러 가져오기
            handler = get_llm_handler(request.model_key)
            
            logger.info(f"뉴스 트렌드 분석 요청 - 카테고리: {request.categories}")
            logger.debug(f"LLM 핸들러 로드 완료: {handler.model_key}")
            
            # 기본 카테고리 설정
            categories = request.categories if request.categories else ['politics', 'economy', 'technology', 'society']
            logger.debug(f"분석할 카테고리: {categories}")
            
            categories_text = ', '.join(categories)
            yield f"data: {json.dumps({'status': 'categories', 'message': f'분석할 카테고리: {categories_text}'}, ensure_ascii=False)}\n\n"
            
            # 카테고리별 뉴스 수집
            all_news = []
            category_summaries = {}
            
            for i, category in enumerate(categories):
                search_message = f'{category} 카테고리 뉴스 검색 중... ({i+1}/{len(categories)})'
                yield f"data: {json.dumps({'status': 'searching', 'message': search_message}, ensure_ascii=False)}\n\n"
                
                logger.debug(f"'{category}' 카테고리 뉴스 검색 시작...")
                category_news = search_news(
                    "최신 뉴스", 
                    max_results=request.max_results//len(categories), 
                    category=category,
                    time_range=request.time_range
                )
                logger.debug(f"'{category}' 카테고리 뉴스 {len(category_news) if category_news else 0}개 수집 완료")
                
                if category_news:
                    all_news.extend(category_news)
                    
                    category_count = len(category_news)
                    analyzing_message = f'{category} 카테고리 ({category_count}개 기사) 분석 중...'
                    yield f"data: {json.dumps({'status': 'category_analyzing', 'message': analyzing_message}, ensure_ascii=False)}\n\n"
                    
                    # 카테고리별 간단 요약
                    category_text = "\n".join([
                        f"• {news.get('title', '')}: {news.get('content', '')[:200]}"
                        for news in category_news[:3]
                    ])
                    
                    category_prompt = f"다음 {category} 카테고리 뉴스들의 주요 트렌드를 한 문장으로 요약해주세요:\n{category_text}"
                    category_summary = handler.generate(category_prompt, max_length=256)
                    category_summaries[category] = category_summary
                    
                    yield f"data: {json.dumps({'status': 'category_completed', 'category': category, 'summary': category_summary}, ensure_ascii=False)}\n\n"
            
            total_news = len(all_news)
            overall_message = f'총 {total_news}개 기사로 전체 트렌드 분석 중...'
            yield f"data: {json.dumps({'status': 'overall_analyzing', 'message': overall_message}, ensure_ascii=False)}\n\n"
            
            # 전체 트렌드 분석 프롬프트
            trend_analysis_prompt = """다음 뉴스 제목들과 카테고리별 요약을 바탕으로 현재 뉴스 트렌드를 분석해주세요.

뉴스 제목들:
{titles}

카테고리별 요약:
{category_summaries}

다음 형식으로 트렌드를 분석해주세요:

## 🔥 오늘의 주요 트렌드
(가장 주목받는 이슈 2-3개)

## 📊 분야별 동향
• 정치: (정치 관련 주요 이슈)
• 경제: (경제 관련 주요 이슈)  
• 사회: (사회 관련 주요 이슈)
• 기술: (기술 관련 주요 이슈)

## 🎭 여론 및 관심도
(국민들이 가장 관심 갖는 이슈들과 여론의 방향)

## 🔮 주목할 포인트
(앞으로 계속 주목해야 할 이슈들)

객관적이고 균형잡힌 시각으로 한국어로 작성해주세요. 출처 정보는 별도로 제공되므로 분석 본문에는 포함하지 마세요."""
            
            all_titles = [news.get('title', '') for news in all_news]
            titles_text = "\n".join([f"• {title}" for title in all_titles[:30]])
            
            full_trend_prompt = trend_analysis_prompt.format(
                titles=titles_text,
                category_summaries="\n".join([f"{cat}: {summary}" for cat, summary in category_summaries.items()])
            )
            
            logger.info("전체 트렌드 분석을 위한 LLM 생성 중...")
            
            # 스트리밍 모드로 분석 생성
            trend_stream = handler.generate(full_trend_prompt, max_length=1024, stream=True)
            
            analysis_parts = []
            for chunk in trend_stream:
                if chunk:
                    analysis_parts.append(chunk)
                    yield f"data: {json.dumps({'status': 'streaming', 'chunk': chunk}, ensure_ascii=False)}\n\n"
                    await asyncio.sleep(0.01)
            
            trend_response = ''.join(analysis_parts)
            logger.debug("전체 트렌드 분석 완료")
            
            # 분석에 사용된 주요 뉴스 출처 정보 준비
            top_articles = sorted(all_news, key=lambda x: x.get('score', 0), reverse=True)[:10]
            source_info = "\n\n📰 **분석 기반 주요 뉴스:**\n" + "\n".join([
                f"• [{article.get('title', 'Unknown')}]({article.get('url', '#')})" + 
                (f" ({article.get('published_date', '')})" if article.get('published_date') else "")
                for article in top_articles[:5]
            ])
            
            trend_response_with_sources = trend_response + source_info
            
            # 완료 신호
            final_result = {
                "status": "completed",
                "overall_trend": trend_response_with_sources,
                "category_trends": category_summaries,
                "total_articles_analyzed": len(all_news),
                "categories": categories,
                "time_range": request.time_range,
                "analyzed_articles": [
                    {
                        'title': article.get('title', ''),
                        'url': article.get('url', ''),
                        'category': article.get('category', 'general'),
                        'published_date': article.get('published_date', ''),
                        'score': article.get('score', 0)
                    } for article in top_articles[:10]
                ],
                "model_info": {
                    "model_key": handler.model_key,
                    "model_id": handler.SUPPORTED_MODELS[handler.model_key]["model_id"]
                }
            }
            
            yield f"data: {json.dumps(final_result, ensure_ascii=False)}\n\n"
            
        except Exception as e:
            logger.error(f"뉴스 트렌드 분석 오류: {e}")
            analysis_error_msg = f'뉴스 트렌드 분석 실패: {str(e)}'
            yield f"data: {json.dumps({'status': 'error', 'message': analysis_error_msg}, ensure_ascii=False)}\n\n"
    
    return StreamingResponse(generate_analysis_stream(), media_type="text/plain; charset=utf-8")

@router.get("/news/search")
async def search_news_endpoint(
    query: str,
    max_results: int = 5,
    category: str = None,
    time_range: str = 'd'
):
    """특정 키워드로 뉴스를 검색합니다"""
    try:
        from utils.helpers import search_news
        
        logger.info(f"뉴스 검색 요청 - 쿼리: {query}, 카테고리: {category}")
        
        news_results = search_news(
            query=query,
            max_results=max_results,
            category=category,
            time_range=time_range
        )
        
        return {
            "news": news_results,
            "total_count": len(news_results),
            "query": query,
            "category": category,
            "time_range": time_range,
            "status": "success"
        }
        
    except Exception as e:
        logger.error(f"뉴스 검색 오류: {e}")
        raise HTTPException(status_code=500, detail=f"뉴스 검색 실패: {str(e)}")

@router.get("/news/categories")
async def get_news_categories():
    """지원되는 뉴스 카테고리 목록을 반환합니다"""
    categories = {
        "politics": "정치",
        "economy": "경제", 
        "technology": "기술/IT",
        "sports": "스포츠",
        "health": "건강/의료",
        "culture": "문화/예술",
        "society": "사회",
        "international": "국제/해외"
    }
    
    return {
        "categories": categories,
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

# === External-Web RAG API 엔드포인트들 ===

@router.post("/external-web/upload-topic")
async def external_web_upload_topic(request: ExternalWebUploadRequest):
    """외부 웹 검색을 통해 특정 주제의 정보를 벡터 DB에 저장합니다"""
    try:
        service = get_rag_service()
        added_chunks, message = service.add_documents_from_web(request.topic, request.max_results)
        
        if added_chunks > 0:
            return {
                "success": True,
                "message": message,
                "topic": request.topic,
                "added_chunks": added_chunks,
                "max_results": request.max_results
            }
        else:
            raise HTTPException(status_code=404, detail=message)
            
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"External-Web 업로드 오류: {e}")
        raise HTTPException(status_code=500, detail=f"External-Web 업로드 실패: {str(e)}")

@router.post("/external-web/auto-rag")
async def external_web_auto_rag(request: ExternalWebAutoRAGRequest):
    """질의에 대해 자동으로 웹 검색하고 벡터 DB화 한 후 RAG 응답을 생성합니다 (스트리밍)"""

    async def generate_auto_rag_stream():
        try:
            service = get_rag_service(request.model_key)
            logger.info(f"자동 External-Web RAG 요청: '{request.query}'")

            # 시작 신호
            start_msg = f"'{request.query}' 관련 자동 RAG 처리를 시작합니다..."
            yield f"data: {json.dumps({'status': 'starting', 'message': start_msg}, ensure_ascii=False)}\n\n"

            # 1단계: 웹 검색 시작
            yield f"data: {json.dumps({'status': 'searching', 'message': '웹에서 관련 뉴스를 검색하는 중...', 'progress': 20}, ensure_ascii=False)}\n\n"

            # 웹 검색 및 벡터 DB 추가
            added_chunks, upload_message = service.add_documents_from_web(request.query, request.max_results)

            if added_chunks == 0:
                no_results_msg = f"'{request.query}'에 대한 최신 뉴스를 찾을 수 없어 답변을 생성할 수 없습니다."
                yield f"data: {json.dumps({'status': 'no_results', 'message': no_results_msg}, ensure_ascii=False)}\n\n"
                return

            # 2단계: 벡터 DB 처리 완료
            vectorize_msg = f"{added_chunks}개의 뉴스 기사를 벡터 DB에 저장 완료"
            yield f"data: {json.dumps({'status': 'vectorizing', 'message': vectorize_msg, 'progress': 50}, ensure_ascii=False)}\n\n"

            # 3단계: RAG 응답 생성 시작
            yield f"data: {json.dumps({'status': 'generating', 'message': 'AI가 종합적인 답변을 생성하는 중...', 'progress': 70}, ensure_ascii=False)}\n\n"

            # RAG 응답 생성
            response = service.generate_response(request.query)

            # 4단계: 관련 문서 정보 가져오기
            yield f"data: {json.dumps({'status': 'finalizing', 'message': '관련 문서 정보를 정리하는 중...', 'progress': 90}, ensure_ascii=False)}\n\n"

            relevant_docs = service.get_relevant_documents(request.query, k=8)

            # 완료 신호
            final_result = {
                "status": "completed",
                "response": response,
                "query": request.query,
                "added_chunks": added_chunks,
                "relevant_documents": relevant_docs,
                "search_query": request.query,
                "upload_message": upload_message,
                "source": "external-web-auto",
                "progress": 100,
                "model_info": {
                    "model_key": service.llm_handler.model_key,
                    "model_id": service.llm_handler.SUPPORTED_MODELS[service.llm_handler.model_key]["model_id"],
                    "description": service.llm_handler.SUPPORTED_MODELS[service.llm_handler.model_key]["description"],
                    "category": service.llm_handler.SUPPORTED_MODELS[service.llm_handler.model_key]["category"],
                    "loaded": service.llm_handler.model is not None
                }
            }

            yield f"data: {json.dumps(final_result, ensure_ascii=False)}\n\n"

        except Exception as e:
            logger.error(f"자동 External-Web RAG 오류: {e}")
            error_msg = f'자동 RAG 처리 실패: {str(e)}'
            yield f"data: {json.dumps({'status': 'error', 'message': error_msg}, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        generate_auto_rag_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )

@router.post("/external-web/rag-query")
async def external_web_rag_query(request: ExternalWebQueryRequest):
    """외부 웹 정보를 기반으로 한 RAG 질의응답"""
    try:
        service = get_rag_service(request.model_key)

        # 벡터 DB에서 관련 문서 검색
        relevant_docs = service.get_relevant_documents(request.prompt, request.top_k)

        if not relevant_docs:
            return {
                "response": "외부 웹 검색 기반 지식베이스에서 충분한 결과를 찾지 못했습니다. 먼저 관련 주제를 업로드해주세요.",
                "prompt": request.prompt,
                "relevant_documents": [],
                "source": "external-web"
            }

        # RAG 응답 생성
        response = service.generate_response(request.prompt)

        return {
            "response": response,
            "prompt": request.prompt,
            "relevant_documents": relevant_docs[:request.top_k],
            "source": "external-web",
            "model_info": {
                "model_key": service.llm_handler.model_key,
                "model_id": service.llm_handler.SUPPORTED_MODELS[service.llm_handler.model_key]["model_id"],
                "description": service.llm_handler.SUPPORTED_MODELS[service.llm_handler.model_key]["description"],
                "category": service.llm_handler.SUPPORTED_MODELS[service.llm_handler.model_key]["category"],
                "loaded": service.llm_handler.model is not None
            }
        }

    except Exception as e:
        logger.error(f"External-Web RAG 질의 오류: {e}")
        raise HTTPException(status_code=500, detail=f"External-Web RAG 질의 실패: {str(e)}")

# === Internal-DBMS RAG API 엔드포인트들 ===

@router.get("/internal-db/tables")
async def get_internal_db_tables(simulate: bool = None):
    """내부 데이터베이스의 테이블 목록을 조회합니다 (자동 fallback 지원)"""
    try:
        service = get_internal_db_service()
        result = await service.get_db_tables(simulate=simulate)

        return result

    except Exception as e:
        logger.error(f"DB 테이블 조회 오류: {e}")
        raise HTTPException(status_code=500, detail=f"DB 테이블 조회 실패: {str(e)}")

@router.post("/internal-db/ingest")
async def internal_db_ingest(request: InternalDBIngestRequest):
    """내부 데이터베이스 테이블을 벡터화하여 FAISS 인덱스를 생성합니다"""
    try:
        service = get_internal_db_service()
        result = await service.ingest_table(
            table_name=request.table,
            save_name=request.save_name,
            simulate=request.simulate,
            id_col=request.id_col,
            title_col=request.title_col,
            text_cols=request.text_cols
        )
        
        return {
            "ok": True,
            "save_dir": result["save_dir"],
            "rows": result["rows"],
            "chunks": result["chunks"],
            "schema": result["schema"],
            "table": request.table,
            "simulate": request.simulate
        }
        
    except Exception as e:
        logger.error(f"Internal DB 인제스트 오류: {e}")
        raise HTTPException(status_code=500, detail=f"Internal DB 인제스트 실패: {str(e)}")

@router.post("/internal-db/query")
async def internal_db_query(request: InternalDBQueryRequest):
    """내부 데이터베이스 기반 RAG 질의응답"""
    try:
        service = get_internal_db_service()
        result = await service.query(
            save_name=request.save_name,
            question=request.question,
            top_k=request.top_k,
            margin=request.margin
        )
        
        return {
            "answer": result["answer"],
            "sources": result["sources"],
            "question": request.question,
            "save_name": request.save_name,
            "top_k": request.top_k,
            "margin": request.margin
        }
        
    except Exception as e:
        logger.error(f"Internal DB 질의 오류: {e}")
        raise HTTPException(status_code=500, detail=f"Internal DB 질의 실패: {str(e)}")

@router.get("/internal-db/status")
async def get_internal_db_status():
    """내부 DB FAISS 인덱스 상태를 확인합니다"""
    try:
        service = get_internal_db_service()
        status = await service.get_status()
        
        return {
            "faiss_indices": status["faiss_indices"],
            "cache_keys": status["cache_keys"],
            "status": "success"
        }
        
    except Exception as e:
        logger.error(f"Internal DB 상태 조회 오류: {e}")
        raise HTTPException(status_code=500, detail=f"Internal DB 상태 조회 실패: {str(e)}")

@router.get("/internal-db/view-table/{table_name}")
async def internal_db_view_table(table_name: str, simulate: bool = None, limit: int = 100):
    """Internal DB 테이블 내용을 조회합니다"""
    try:
        service = get_internal_db_service()

        # 테이블 데이터 조회 (새로운 메서드 호출)
        result = await service.view_table_data(table_name=table_name, simulate=simulate, limit=limit)

        return result

    except Exception as e:
        logger.error(f"Internal DB 테이블 조회 오류: {e}")
        raise HTTPException(status_code=500, detail=f"테이블 '{table_name}' 조회 실패: {str(e)}")

# === 음성 처리 API 엔드포인트들 ===

@router.post("/speech/text-to-speech")
async def text_to_speech(request: TextToSpeechRequest):
    """텍스트를 음성으로 변환합니다"""
    try:
        service = get_speech_service()
        result = service.text_to_speech(
            text=request.text,
            language=request.language,
            slow=request.slow
        )

        if result["success"]:
            from fastapi.responses import FileResponse
            return FileResponse(
                result["audio_file"],
                media_type="audio/mpeg",
                filename="speech.mp3",
                headers={
                    "Content-Disposition": "attachment; filename=speech.mp3",
                    "X-Duration-Estimate": str(result.get("duration_estimate", 0))
                }
            )
        else:
            raise HTTPException(status_code=400, detail=result["error"])

    except Exception as e:
        logger.error(f"TTS 오류: {e}")
        raise HTTPException(status_code=500, detail=f"음성 합성 실패: {str(e)}")

@router.post("/speech/speech-to-text")
async def speech_to_text(
    audio_file: UploadFile = File(...),
    prefer_whisper: bool = True
):
    """음성 파일을 텍스트로 변환합니다"""
    try:
        service = get_speech_service()

        # 업로드된 파일을 임시 파일로 저장
        import tempfile
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        content = await audio_file.read()
        temp_file.write(content)
        temp_file.close()

        # 음성 인식 수행
        result = service.speech_to_text(temp_file.name, prefer_whisper)

        # 임시 파일 정리
        service.cleanup_temp_file(temp_file.name)

        if result["success"]:
            return {
                "text": result["text"],
                "language": result.get("language", "unknown"),
                "confidence": result.get("confidence", 0.0),
                "method": result.get("method", "unknown"),
                "success": True
            }
        else:
            return {
                "text": "",
                "error": result["error"],
                "method": result.get("method", "unknown"),
                "success": False
            }

    except Exception as e:
        logger.error(f"STT 오류: {e}")
        raise HTTPException(status_code=500, detail=f"음성 인식 실패: {str(e)}")

@router.post("/speech/voice-chat")
async def voice_chat(request: VoiceChatRequest):
    """음성 채팅: 텍스트 입력 → LLM 응답 → 음성 출력"""
    try:
        if not request.text:
            raise HTTPException(status_code=400, detail="입력 텍스트가 필요합니다")

        # 1. LLM으로 응답 생성 (Tool Calling 지원)
        llm = get_llm_handler(request.model_key)
        executor = await get_tool_executor()
        wrapper = ToolCallingWrapper(llm, executor)
        response_text = await wrapper.generate_with_tools(request.text, request.user_id)

        # 2. 응답을 음성으로 변환
        speech_service = get_speech_service()
        tts_result = speech_service.text_to_speech(
            text=response_text,
            language=request.voice_language,
            slow=request.voice_slow
        )

        if tts_result["success"]:
            from fastapi.responses import FileResponse
            from urllib.parse import quote
            # URL encode the Korean text for HTTP header
            encoded_response = quote(response_text)
            return FileResponse(
                tts_result["audio_file"],
                media_type="audio/mpeg",
                filename="chat_response.mp3",
                headers={
                    "Content-Disposition": "attachment; filename=chat_response.mp3",
                    "X-Response-Text": encoded_response,
                    "X-Duration-Estimate": str(tts_result.get("duration_estimate", 0))
                }
            )
        else:
            # TTS 실패 시 텍스트만 반환
            return {
                "response_text": response_text,
                "audio_available": False,
                "error": tts_result["error"]
            }

    except Exception as e:
        logger.error(f"음성 채팅 오류: {e}")
        raise HTTPException(status_code=500, detail=f"음성 채팅 실패: {str(e)}")

@router.post("/speech/full-voice-chat")
async def full_voice_chat(
    audio_file: UploadFile = File(...),
    user_id: str = "default_user",
    model_key: str = None,
    voice_language: str = "ko",
    voice_slow: bool = False,
    prefer_whisper: bool = True
):
    """완전 음성 채팅: 음성 입력 → 텍스트 → LLM 응답 → 음성 출력"""
    try:
        speech_service = get_speech_service()

        # 1. 음성을 텍스트로 변환
        import tempfile
        temp_audio_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        content = await audio_file.read()
        temp_audio_file.write(content)
        temp_audio_file.close()

        stt_result = speech_service.speech_to_text(temp_audio_file.name, prefer_whisper)
        speech_service.cleanup_temp_file(temp_audio_file.name)

        if not stt_result["success"]:
            return {
                "error": f"음성 인식 실패: {stt_result['error']}",
                "stage": "speech_to_text"
            }

        user_text = stt_result["text"]

        # 2. LLM으로 응답 생성 (Tool Calling 지원)
        llm = get_llm_handler(model_key)
        executor = await get_tool_executor()
        wrapper = ToolCallingWrapper(llm, executor)
        response_text = await wrapper.generate_with_tools(user_text, user_id)

        # 3. 응답을 음성으로 변환
        tts_result = speech_service.text_to_speech(
            text=response_text,
            language=voice_language,
            slow=voice_slow
        )

        if tts_result["success"]:
            from fastapi.responses import FileResponse
            from urllib.parse import quote
            # URL encode Korean text for HTTP headers
            encoded_user_text = quote(user_text)
            encoded_response = quote(response_text)
            return FileResponse(
                tts_result["audio_file"],
                media_type="audio/mpeg",
                filename="full_chat_response.mp3",
                headers={
                    "Content-Disposition": "attachment; filename=full_chat_response.mp3",
                    "X-User-Text": encoded_user_text,
                    "X-Response-Text": encoded_response,
                    "X-STT-Method": stt_result.get("method", "unknown"),
                    "X-STT-Confidence": str(stt_result.get("confidence", 0.0)),
                    "X-Duration-Estimate": str(tts_result.get("duration_estimate", 0))
                }
            )
        else:
            # TTS 실패 시 텍스트 정보만 반환
            return {
                "user_text": user_text,
                "response_text": response_text,
                "stt_method": stt_result.get("method"),
                "stt_confidence": stt_result.get("confidence"),
                "audio_available": False,
                "tts_error": tts_result["error"]
            }

    except Exception as e:
        logger.error(f"완전 음성 채팅 오류: {e}")
        raise HTTPException(status_code=500, detail=f"완전 음성 채팅 실패: {str(e)}")

@router.get("/speech/languages")
async def get_supported_languages():
    """지원되는 언어 목록을 반환합니다"""
    try:
        service = get_speech_service()
        languages = service.get_supported_languages()

        return {
            "supported_languages": languages,
            "default_language": "ko",
            "total_languages": len(languages)
        }

    except Exception as e:
        logger.error(f"지원 언어 조회 오류: {e}")
        raise HTTPException(status_code=500, detail=f"지원 언어 조회 실패: {str(e)}")

@router.get("/speech/status")
async def get_speech_service_status():
    """음성 서비스 상태를 확인합니다"""
    try:
        service = get_speech_service()

        return {
            "whisper_available": service.whisper_model is not None,
            "google_stt_available": True,  # SpeechRecognition은 항상 사용 가능
            "gtts_available": True,  # gTTS는 항상 사용 가능
            "microphone_available": service.microphone is not None,
            "supported_languages": len(service.get_supported_languages()),
            "status": "ready"
        }

    except Exception as e:
        logger.error(f"음성 서비스 상태 조회 오류: {e}")
        raise HTTPException(status_code=500, detail=f"음성 서비스 상태 조회 실패: {str(e)}")

# === 실시간 스트리밍 TTS API 엔드포인트들 ===

@router.post("/speech/streaming-generate-with-voice")
async def streaming_generate_with_voice(request: StreamingTTSRequest):
    """텍스트를 스트리밍 생성하면서 실시간으로 음성 읽기"""
    try:
        llm = get_llm_handler(request.model_key)
        streaming_tts = get_streaming_tts_service()

        async def generate_stream():
            # 버퍼 초기화
            streaming_tts.reset_buffer()

            try:
                for chunk in streaming_tts.generate_with_realtime_speech(
                    llm_handler=llm,
                    prompt=request.prompt,
                    model_key=request.model_key,
                    voice_language=request.voice_language,
                    voice_slow=request.voice_slow,
                    read_partial=request.read_partial
                ):
                    yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"

            except Exception as e:
                logger.error(f"스트리밍 TTS 생성 오류: {e}")
                error_chunk = {
                    "type": "error",
                    "error": str(e)
                }
                yield f"data: {json.dumps(error_chunk, ensure_ascii=False)}\n\n"

        return StreamingResponse(
            generate_stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no"
            }
        )

    except Exception as e:
        logger.error(f"스트리밍 TTS 초기화 오류: {e}")
        raise HTTPException(status_code=500, detail=f"스트리밍 TTS 실패: {str(e)}")

@router.post("/speech/sentences-to-speech")
async def sentences_to_speech(request: SentencesTTSRequest):
    """여러 문장을 한번에 음성으로 변환"""
    try:
        sentence_tts = get_sentence_tts_service()

        results = await sentence_tts.convert_sentences_to_speech(
            sentences=request.sentences,
            language=request.language,
            slow=request.slow
        )

        # 성공한 음성 파일들의 URL 생성
        successful_results = []
        failed_results = []

        for result in results:
            if result["success"]:
                successful_results.append({
                    "sentence_index": result["sentence_index"],
                    "sentence_text": result["sentence_text"],
                    "audio_file": result["audio_file"],
                    "success": True
                })
            else:
                failed_results.append({
                    "sentence_index": result["sentence_index"],
                    "sentence_text": result["sentence_text"],
                    "error": result["error"],
                    "success": False
                })

        return {
            "total_sentences": len(request.sentences),
            "successful_conversions": len(successful_results),
            "failed_conversions": len(failed_results),
            "results": successful_results,
            "errors": failed_results,
            "language": request.language,
            "slow": request.slow
        }

    except Exception as e:
        logger.error(f"문장 TTS 변환 오류: {e}")
        raise HTTPException(status_code=500, detail=f"문장 TTS 변환 실패: {str(e)}")

@router.post("/speech/text-to-sentences-and-speech")
async def text_to_sentences_and_speech(request: TextToSpeechRequest):
    """텍스트를 문장으로 분할하고 각각 음성으로 변환"""
    try:
        sentence_tts = get_sentence_tts_service()

        # 텍스트를 문장으로 분할
        sentences = sentence_tts.split_text_into_sentences(request.text)

        # 각 문장을 음성으로 변환
        results = await sentence_tts.convert_sentences_to_speech(
            sentences=sentences,
            language=request.language,
            slow=request.slow
        )

        # 결과 정리
        audio_files = []
        sentence_info = []

        for result in results:
            sentence_info.append({
                "index": result["sentence_index"],
                "text": result["sentence_text"],
                "success": result["success"]
            })

            if result["success"]:
                audio_files.append(result["audio_file"])

        return {
            "original_text": request.text,
            "total_sentences": len(sentences),
            "successful_conversions": len(audio_files),
            "sentences": sentence_info,
            "audio_files": audio_files,
            "language": request.language
        }

    except Exception as e:
        logger.error(f"텍스트 분할 TTS 오류: {e}")
        raise HTTPException(status_code=500, detail=f"텍스트 분할 TTS 실패: {str(e)}")

@router.get("/speech/streaming-tts/status")
async def get_streaming_tts_status():
    """스트리밍 TTS 서비스 상태 확인"""
    try:
        # 서비스들 초기화 확인
        speech_service = get_speech_service()
        streaming_tts = get_streaming_tts_service()
        sentence_tts = get_sentence_tts_service()

        return {
            "streaming_tts_available": streaming_tts is not None,
            "sentence_tts_available": sentence_tts is not None,
            "speech_service_available": speech_service is not None,
            "whisper_available": speech_service.whisper_model is not None,
            "gtts_available": True,
            "supported_features": [
                "실시간 스트리밍 TTS",
                "문장 단위 TTS",
                "텍스트 분할 TTS",
                "부분 문장 읽기"
            ],
            "supported_languages": speech_service.get_supported_languages(),
            "status": "ready"
        }

    except Exception as e:
        logger.error(f"스트리밍 TTS 상태 조회 오류: {e}")
        raise HTTPException(status_code=500, detail=f"스트리밍 TTS 상태 조회 실패: {str(e)}")

# === RSS 뉴스 API 엔드포인트들 ===

@router.get("/news-rss/keywords", response_model=List[NewsKeyword])
async def get_trending_keywords_rss(
    limit: int = Query(default=10, ge=1, le=50, description="반환할 키워드 개수")
):
    """실시간 트렌딩 뉴스 키워드를 가져옵니다 (RSS)"""
    try:
        service = get_news_service_rss()
        keywords = await service.get_trending_keywords(limit=limit)
        return keywords
    except Exception as e:
        logger.error(f"RSS 키워드 조회 실패: {e}")
        raise HTTPException(status_code=500, detail=f"키워드 조회 실패: {str(e)}")

@router.get("/news-rss/articles", response_model=NewsResponse)
async def get_news_articles_rss(
    keyword: Optional[str] = Query(None, description="검색할 키워드"),
    category: Optional[str] = Query(None, description="뉴스 카테고리"),
    limit: int = Query(default=20, ge=1, le=100, description="반환할 기사 개수")
):
    """뉴스 기사를 검색하거나 최신 기사를 가져옵니다 (RSS)"""
    try:
        service = get_news_service_rss()
        articles = await service.get_news_articles(
            keyword=keyword,
            category=category,
            limit=limit
        )

        return NewsResponse(
            total=len(articles),
            articles=articles,
            keyword=keyword,
            category=category
        )
    except Exception as e:
        logger.error(f"RSS 뉴스 조회 실패: {e}")
        raise HTTPException(status_code=500, detail=f"뉴스 조회 실패: {str(e)}")

@router.get("/news-rss/categories")
async def get_news_categories_rss():
    """사용 가능한 뉴스 카테고리 목록을 반환합니다 (RSS)"""
    return {
        "categories": [
            {"id": "all", "name": "전체"},
            {"id": "politics", "name": "정치"},
            {"id": "economy", "name": "경제"},
            {"id": "society", "name": "사회"},
            {"id": "culture", "name": "문화"},
            {"id": "world", "name": "세계"},
            {"id": "it", "name": "IT/과학"}
        ]
    }

@router.post("/news-rss/keywords/custom")
async def add_custom_keyword_rss(keyword: str):
    """사용자 정의 키워드를 추가합니다 (RSS)"""
    try:
        service = get_news_service_rss()
        result = await service.add_custom_keyword(keyword)
        if result:
            return {"success": True, "keyword": keyword, "message": "키워드가 추가되었습니다."}
        else:
            return {"success": False, "message": "이미 존재하는 키워드이거나 추가할 수 없습니다."}
    except Exception as e:
        logger.error(f"RSS 키워드 추가 실패: {e}")
        raise HTTPException(status_code=500, detail=f"키워드 추가 실패: {str(e)}")

@router.get("/news-rss/keywords/user")
async def get_user_keywords_rss():
    """사용자가 등록한 키워드 목록을 반환합니다 (RSS)"""
    try:
        service = get_news_service_rss()
        keywords = await service.get_user_keywords()
        return {"keywords": keywords}
    except Exception as e:
        logger.error(f"RSS 키워드 조회 실패: {e}")
        raise HTTPException(status_code=500, detail=f"키워드 조회 실패: {str(e)}")

@router.delete("/news-rss/keywords/custom")
async def delete_keyword_rss(keyword: str):
    """사용자 키워드를 삭제합니다 (RSS)"""
    try:
        service = get_news_service_rss()
        result = await service.delete_keyword(keyword)
        if result:
            return {"success": True, "message": "키워드가 삭제되었습니다."}
        else:
            return {"success": False, "message": "키워드를 찾을 수 없습니다."}
    except Exception as e:
        logger.error(f"RSS 키워드 삭제 실패: {e}")
        raise HTTPException(status_code=500, detail=f"키워드 삭제 실패: {str(e)}")

# === 재정 관리 API 엔드포인트들 ===

@router.get("/finance/items", response_model=List[FinanceItem])
async def list_finance_items(category: Optional[str] = None):
    """재정 항목 목록 조회"""
    try:
        service = get_finance_service()
        items = await service.list_items(category=category)
        return items
    except Exception as e:
        logger.error(f"재정 항목 조회 실패: {e}")
        raise HTTPException(status_code=500, detail=f"재정 항목 조회 실패: {str(e)}")

@router.post("/finance/items", response_model=FinanceItem)
async def create_finance_item(item: FinanceItem):
    """재정 항목 추가"""
    try:
        service = get_finance_service()
        new_item = await service.create_item(item)
        return new_item
    except Exception as e:
        logger.error(f"재정 항목 추가 실패: {e}")
        raise HTTPException(status_code=500, detail=f"재정 항목 추가 실패: {str(e)}")

@router.get("/finance/items/{item_id}", response_model=FinanceItem)
async def get_finance_item(item_id: int):
    """특정 재정 항목 조회"""
    try:
        service = get_finance_service()
        item = await service.get_item(item_id)
        if item is None:
            raise HTTPException(status_code=404, detail="Item not found")
        return item
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"재정 항목 조회 실패: {e}")
        raise HTTPException(status_code=500, detail=f"재정 항목 조회 실패: {str(e)}")

@router.put("/finance/items/{item_id}", response_model=FinanceItem)
async def update_finance_item(item_id: int, item: FinanceItem):
    """재정 항목 수정"""
    try:
        service = get_finance_service()
        updated_item = await service.update_item(item_id, item)
        if updated_item is None:
            raise HTTPException(status_code=404, detail="Item not found")
        return updated_item
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"재정 항목 수정 실패: {e}")
        raise HTTPException(status_code=500, detail=f"재정 항목 수정 실패: {str(e)}")

@router.delete("/finance/items/{item_id}")
async def delete_finance_item(item_id: int):
    """재정 항목 삭제"""
    try:
        service = get_finance_service()
        result = await service.delete_item(item_id)
        if not result:
            raise HTTPException(status_code=404, detail="Item not found")
        return {"success": True, "message": "Item deleted"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"재정 항목 삭제 실패: {e}")
        raise HTTPException(status_code=500, detail=f"재정 항목 삭제 실패: {str(e)}")

# === Yahoo Finance (KOSPI) API 엔드포인트들 ===

@router.get("/yahoo/v8/finance/chart/{symbol}")
async def get_yahoo_finance_chart(
    symbol: str,
    interval: str = Query("1d", description="데이터 간격 (5m, 30m, 1d, 1wk, 1mo)"),
    range: str = Query("1d", description="기간 (1d, 5d, 1mo, 1y, 5y)")
):
    """Yahoo Finance 차트 데이터 조회 (KOSPI 등)"""
    try:
        service = get_yahoo_finance_service()
        data = await service.get_kospi_chart(interval=interval, range_period=range)
        return data
    except Exception as e:
        logger.error(f"Yahoo Finance 데이터 조회 실패: {e}")
        raise HTTPException(status_code=500, detail=f"Yahoo Finance 데이터 조회 실패: {str(e)}")

# === DB LLM 연동 API 엔드포인트들 ===

@router.post("/chat-with-context")
async def chat_with_db_context(request: ChatWithContextRequest):
    """사용자의 DB 데이터를 컨텍스트로 포함한 채팅"""
    try:
        # DB에서 사용자 컨텍스트 가져오기
        db_service = await get_db_llm_service()
        user_context = await db_service.get_user_context(request.user_id)

        # LLM 핸들러 가져오기
        handler = get_llm_handler(request.model_key)

        # 컨텍스트와 함께 프롬프트 구성
        enhanced_prompt = f"""다음은 사용자의 정보입니다:

{user_context}

사용자의 질문: {request.message}

위 정보를 참고하여 사용자의 질문에 답변해주세요."""

        if request.stream:
            # 스트리밍 응답
            async def chat_stream():
                for chunk in handler.generate(enhanced_prompt, max_length=512, stream=True):
                    if chunk:
                        data = json.dumps({"content": chunk, "done": False}, ensure_ascii=False)
                        yield f"data: {data}\n\n"
                yield f"data: {json.dumps({'content': '', 'done': True})}\n\n"

            return StreamingResponse(
                chat_stream(),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "Connection": "keep-alive"}
            )
        else:
            # 일반 응답
            response = handler.generate(enhanced_prompt, max_length=512, stream=False)
            return {
                "response": response,
                "message": request.message,
                "user_id": request.user_id,
                "context_included": True,
                "model_info": {
                    "model_key": handler.model_key,
                    "model_id": handler.SUPPORTED_MODELS[handler.model_key]["model_id"],
                    "description": handler.SUPPORTED_MODELS[handler.model_key]["description"],
                    "category": handler.SUPPORTED_MODELS[handler.model_key]["category"],
                    "loaded": handler.model is not None
                }
            }

    except Exception as e:
        logger.error(f"컨텍스트 채팅 오류: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/user/{user_id}/context")
async def get_user_context(user_id: str):
    """사용자의 DB 컨텍스트 조회"""
    try:
        db_service = await get_db_llm_service()
        context = await db_service.get_user_context(user_id)

        return {
            "user_id": user_id,
            "context": context,
            "success": True
        }
    except Exception as e:
        logger.error(f"사용자 컨텍스트 조회 오류: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/user/{user_id}/data/{data_type}")
async def get_user_data(user_id: str, data_type: str, date: str = None):
    """특정 유형의 사용자 데이터 조회"""
    try:
        db_service = await get_db_llm_service()

        kwargs = {}
        if date:
            kwargs['date'] = date

        data = await db_service.query_database(user_id, data_type, **kwargs)

        return {
            "user_id": user_id,
            "data_type": data_type,
            "data": data,
            "count": len(data),
            "success": True
        }
    except Exception as e:
        logger.error(f"사용자 데이터 조회 오류: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/diet/recommendations")
async def get_diet_recommendations(authorization: str = Header(None)):
    """사용자의 건강 정보를 기반으로 LLM이 생성한 개인화된 식단 추천"""
    try:
        from services.auth_service import get_current_user_id, set_request_context
        set_request_context(authorization)
        user_id = get_current_user_id()

        # DB 서비스 가져오기
        db_service = await get_db_llm_service()

        # 사용자의 건강 데이터 조회
        async with db_service.pool.acquire() as conn:
            async with conn.cursor(aiomysql.DictCursor) as cursor:
                # 기저질환 조회
                await cursor.execute(
                    "SELECT name, status FROM disease WHERE user_id = %s AND status = 'active'",
                    (user_id,)
                )
                diseases = await cursor.fetchall()

                # 복용약 조회
                await cursor.execute(
                    "SELECT name, dosage FROM medication WHERE user_id = %s",
                    (user_id,)
                )
                medications = await cursor.fetchall()

        # 건강 정보를 텍스트로 구성
        health_info = []
        if diseases:
            disease_names = [d['name'] for d in diseases]
            health_info.append(f"기저질환: {', '.join(disease_names)}")
        if medications:
            med_info = [f"{m['name']}({m['dosage']})" if m.get('dosage') else m['name'] for m in medications]
            health_info.append(f"복용약: {', '.join(med_info)}")

        health_context = " | ".join(health_info) if health_info else "특별한 건강 제약 없음"

        # LLM으로 식단 추천 요청
        handler = get_llm_handler("bllossom")  # 한국어 특화 모델 사용

        prompt = f"""사용자의 건강 정보: {health_context}

위 건강 정보를 고려하여 오늘의 개인화된 식단을 추천해주세요.

다음 JSON 형식으로만 응답하세요 (다른 설명 없이 JSON만):
[
  {{"meal": "아침", "menu": "구체적인 메뉴", "calories": 숫자}},
  {{"meal": "점심", "menu": "구체적인 메뉴", "calories": 숫자}},
  {{"meal": "저녁", "menu": "구체적인 메뉴", "calories": 숫자}}
]

건강 상태에 맞는 영양가 있고 균형 잡힌 식단을 추천해주세요."""

        # LLM 응답 생성
        llm_response = handler.generate(prompt, max_length=512, stream=False)
        logger.info(f"Diet LLM 응답: {llm_response}")

        # JSON 파싱
        import re
        # JSON 배열 추출 (```json 태그나 다른 텍스트 제거)
        json_match = re.search(r'\[\s*\{.*?\}\s*\]', llm_response, re.DOTALL)
        if json_match:
            json_str = json_match.group(0)
            diet_recommendations = json.loads(json_str)
        else:
            # JSON 파싱 실패 시 기본 식단 반환
            logger.warning("LLM 응답에서 JSON 추출 실패, 기본 식단 반환")
            diet_recommendations = [
                {"meal": "아침", "menu": "오트밀 + 바나나", "calories": 320},
                {"meal": "점심", "menu": "현미밥 + 연어구이", "calories": 580},
                {"meal": "저녁", "menu": "샐러드 + 닭가슴살", "calories": 450}
            ]

        return {
            "success": True,
            "data": diet_recommendations,
            "health_context": health_context
        }

    except Exception as e:
        logger.error(f"식단 추천 생성 오류: {e}")
        # 오류 발생 시 기본 식단 반환
        return {
            "success": False,
            "data": [
                {"meal": "아침", "menu": "오트밀 + 바나나", "calories": 320},
                {"meal": "점심", "menu": "현미밥 + 연어구이", "calories": 580},
                {"meal": "저녁", "menu": "샐러드 + 닭가슴살", "calories": 450}
            ],
            "health_context": "정보 없음",
            "error": str(e)
        }

# === Health 관리 API 엔드포인트들 ===

# Disease 관련 API
@router.get("/health/diseases")
async def list_diseases(authorization: str = Header(None)):
    """모든 기저질환 조회"""
    try:
        from services.auth_service import get_current_user_id, set_request_context
        set_request_context(authorization)
        user_id = get_current_user_id()

        db_service = await get_db_llm_service()
        async with db_service.pool.acquire() as conn:
            async with conn.cursor(aiomysql.DictCursor) as cursor:
                await cursor.execute(
                    "SELECT disease_id as diseaseId, name, diagnosed_date as diagnosedDate, status FROM disease WHERE user_id = %s ORDER BY disease_id DESC",
                    (user_id,)
                )
                diseases = await cursor.fetchall()

                # Convert datetime to string
                for disease in diseases:
                    if disease.get('diagnosedDate'):
                        disease['diagnosedDate'] = disease['diagnosedDate'].isoformat()

                return diseases
    except Exception as e:
        logger.error(f"기저질환 조회 오류: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/health/diseases")
async def create_disease(disease: DiseaseRequest, authorization: str = Header(None)):
    """새로운 기저질환 추가"""
    try:
        from services.auth_service import get_current_user_id, set_request_context
        set_request_context(authorization)
        user_id = get_current_user_id()

        db_service = await get_db_llm_service()
        async with db_service.pool.acquire() as conn:
            async with conn.cursor(aiomysql.DictCursor) as cursor:
                await cursor.execute(
                    "INSERT INTO disease (user_id, name, diagnosed_date, status) VALUES (%s, %s, %s, %s)",
                    (user_id, disease.name, disease.diagnosedDate, disease.status)
                )
                disease_id = cursor.lastrowid

                return {
                    "diseaseId": disease_id,
                    "name": disease.name,
                    "diagnosedDate": disease.diagnosedDate,
                    "status": disease.status
                }
    except Exception as e:
        logger.error(f"기저질환 추가 오류: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/health/diseases/{disease_id}")
async def get_disease(disease_id: int):
    """특정 기저질환 조회"""
    try:
        from services.auth_service import get_current_user_id
        user_id = get_current_user_id()

        db_service = await get_db_llm_service()
        async with db_service.pool.acquire() as conn:
            async with conn.cursor(aiomysql.DictCursor) as cursor:
                await cursor.execute(
                    "SELECT disease_id as diseaseId, name, diagnosed_date as diagnosedDate, status FROM disease WHERE disease_id = %s AND user_id = %s",
                    (disease_id, user_id)
                )
                disease = await cursor.fetchone()

                if not disease:
                    raise HTTPException(status_code=404, detail="질환을 찾을 수 없습니다")

                if disease.get('diagnosedDate'):
                    disease['diagnosedDate'] = disease['diagnosedDate'].isoformat()

                return disease
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"기저질환 조회 오류: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.put("/health/diseases/{disease_id}")
async def update_disease(disease_id: int, disease: DiseaseRequest):
    """기저질환 수정"""
    try:
        from services.auth_service import get_current_user_id
        user_id = get_current_user_id()

        db_service = await get_db_llm_service()
        async with db_service.pool.acquire() as conn:
            async with conn.cursor(aiomysql.DictCursor) as cursor:
                await cursor.execute(
                    "UPDATE disease SET name = %s, diagnosed_date = %s, status = %s WHERE disease_id = %s AND user_id = %s",
                    (disease.name, disease.diagnosedDate, disease.status, disease_id, user_id)
                )

                if cursor.rowcount == 0:
                    raise HTTPException(status_code=404, detail="질환을 찾을 수 없습니다")

                return {
                    "diseaseId": disease_id,
                    "name": disease.name,
                    "diagnosedDate": disease.diagnosedDate,
                    "status": disease.status
                }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"기저질환 수정 오류: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/health/diseases/{disease_id}")
async def delete_disease(disease_id: int):
    """기저질환 삭제"""
    try:
        from services.auth_service import get_current_user_id
        user_id = get_current_user_id()

        db_service = await get_db_llm_service()
        async with db_service.pool.acquire() as conn:
            async with conn.cursor(aiomysql.DictCursor) as cursor:
                await cursor.execute(
                    "DELETE FROM disease WHERE disease_id = %s AND user_id = %s",
                    (disease_id, user_id)
                )

                if cursor.rowcount == 0:
                    raise HTTPException(status_code=404, detail="질환을 찾을 수 없습니다")

                return {"success": True, "message": "질환이 삭제되었습니다"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"기저질환 삭제 오류: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Medication 관련 API
@router.get("/health/medications")
async def list_medications(authorization: str = Header(None)):
    """모든 복약 알람 조회"""
    try:
        from services.auth_service import get_current_user_id, set_request_context
        set_request_context(authorization)
        user_id = get_current_user_id()

        db_service = await get_db_llm_service()
        async with db_service.pool.acquire() as conn:
            async with conn.cursor(aiomysql.DictCursor) as cursor:
                await cursor.execute(
                    "SELECT medication_id as medicationId, name, dosage, intake_time as intakeTime, alarm_enabled as alarmEnabled FROM medication WHERE user_id = %s ORDER BY medication_id DESC",
                    (user_id,)
                )
                medications = await cursor.fetchall()

                # Convert time and bit to proper format
                for med in medications:
                    if med.get('intakeTime') is not None:
                        # Convert timedelta to string
                        total_seconds = int(med['intakeTime'].total_seconds())
                        hours = total_seconds // 3600
                        minutes = (total_seconds % 3600) // 60
                        med['intakeTime'] = f"{hours:02d}:{minutes:02d}"
                    if med.get('alarmEnabled') is not None:
                        med['alarmEnabled'] = bool(med['alarmEnabled'])

                return medications
    except Exception as e:
        logger.error(f"복약 알람 조회 오류: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/health/medications")
async def create_medication(medication: MedicationRequest, authorization: str = Header(None)):
    """새로운 복약 알람 추가"""
    try:
        from services.auth_service import get_current_user_id, set_request_context
        set_request_context(authorization)
        user_id = get_current_user_id()

        db_service = await get_db_llm_service()
        async with db_service.pool.acquire() as conn:
            async with conn.cursor(aiomysql.DictCursor) as cursor:
                await cursor.execute(
                    "INSERT INTO medication (user_id, name, dosage, intake_time, alarm_enabled) VALUES (%s, %s, %s, %s, %s)",
                    (user_id, medication.name, medication.dosage, medication.intakeTime, medication.alarmEnabled)
                )
                medication_id = cursor.lastrowid

                return {
                    "medicationId": medication_id,
                    "name": medication.name,
                    "dosage": medication.dosage,
                    "intakeTime": medication.intakeTime,
                    "alarmEnabled": medication.alarmEnabled
                }
    except Exception as e:
        logger.error(f"복약 알람 추가 오류: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/health/medications/{medication_id}")
async def get_medication(medication_id: int):
    """특정 복약 알람 조회"""
    try:
        from services.auth_service import get_current_user_id
        user_id = get_current_user_id()

        db_service = await get_db_llm_service()
        async with db_service.pool.acquire() as conn:
            async with conn.cursor(aiomysql.DictCursor) as cursor:
                await cursor.execute(
                    "SELECT medication_id as medicationId, name, dosage, intake_time as intakeTime, alarm_enabled as alarmEnabled FROM medication WHERE medication_id = %s AND user_id = %s",
                    (medication_id, user_id)
                )
                medication = await cursor.fetchone()

                if not medication:
                    raise HTTPException(status_code=404, detail="복약 알람을 찾을 수 없습니다")

                if medication.get('intakeTime') is not None:
                    total_seconds = int(medication['intakeTime'].total_seconds())
                    hours = total_seconds // 3600
                    minutes = (total_seconds % 3600) // 60
                    medication['intakeTime'] = f"{hours:02d}:{minutes:02d}"
                if medication.get('alarmEnabled') is not None:
                    medication['alarmEnabled'] = bool(medication['alarmEnabled'])

                return medication
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"복약 알람 조회 오류: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.put("/health/medications/{medication_id}")
async def update_medication(medication_id: int, medication: MedicationRequest):
    """복약 알람 수정"""
    try:
        from services.auth_service import get_current_user_id
        user_id = get_current_user_id()

        db_service = await get_db_llm_service()
        async with db_service.pool.acquire() as conn:
            async with conn.cursor(aiomysql.DictCursor) as cursor:
                await cursor.execute(
                    "UPDATE medication SET name = %s, dosage = %s, intake_time = %s, alarm_enabled = %s WHERE medication_id = %s AND user_id = %s",
                    (medication.name, medication.dosage, medication.intakeTime, medication.alarmEnabled, medication_id, user_id)
                )

                if cursor.rowcount == 0:
                    raise HTTPException(status_code=404, detail="복약 알람을 찾을 수 없습니다")

                return {
                    "medicationId": medication_id,
                    "name": medication.name,
                    "dosage": medication.dosage,
                    "intakeTime": medication.intakeTime,
                    "alarmEnabled": medication.alarmEnabled
                }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"복약 알람 수정 오류: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/health/medications/{medication_id}")
async def delete_medication(medication_id: int):
    """복약 알람 삭제"""
    try:
        from services.auth_service import get_current_user_id
        user_id = get_current_user_id()

        db_service = await get_db_llm_service()
        async with db_service.pool.acquire() as conn:
            async with conn.cursor(aiomysql.DictCursor) as cursor:
                await cursor.execute(
                    "DELETE FROM medication WHERE medication_id = %s AND user_id = %s",
                    (medication_id, user_id)
                )

                if cursor.rowcount == 0:
                    raise HTTPException(status_code=404, detail="복약 알람을 찾을 수 없습니다")

                return {"success": True, "message": "복약 알람이 삭제되었습니다"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"복약 알람 삭제 오류: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/grocery/deals")
async def get_grocery_deals():
    """식료품 가격 정보 조회"""
    try:
        from pathlib import Path

        # grocery_deals.json 파일 경로
        grocery_data_path = Path(__file__).parent.parent.parent / "data" / "grocery_deals.json"

        if not grocery_data_path.exists():
            raise HTTPException(status_code=404, detail="식료품 데이터를 찾을 수 없습니다")

        with open(grocery_data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        return data
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"식료품 데이터 조회 오류: {e}")
        raise HTTPException(status_code=500, detail=str(e))
# ===========================
# 회원가입 관련 엔드포인트
# ===========================

from typing import Optional

# JWT 설정
JWT_SECRET_KEY = "your-secret-key-change-this-in-production"  # 실제 환경에서는 환경변수로 관리
JWT_ALGORITHM = "HS256"
JWT_EXPIRATION_HOURS = 24

# Pydantic 모델
class StartRegisterRequest(BaseModel):
    language: str

class StartRegisterResponse(BaseModel):
    registrationId: str

class ConsentsRequest(BaseModel):
    registrationId: str
    consentHealth: bool
    consentFinance: bool
    consentSocial: bool
    consentClp: bool

class ProfileRequest(BaseModel):
    registrationId: str
    name: str
    nickname: str

class CredentialsRequest(BaseModel):
    registrationId: str
    userid: str
    passwordHash: str
    issueToken: bool = True

class TokenResponse(BaseModel):
    token: Optional[str]

class LoginRequest(BaseModel):
    userid: str
    passwordHash: str

# Database connection pool for registration
registration_db_pool = None

async def get_registration_db_pool():
    """회원가입용 DB 연결 풀 가져오기"""
    global registration_db_pool
    if registration_db_pool is None:
        db_config = config.mariadb_config
        registration_db_pool = await aiomysql.create_pool(
            host=db_config['host'],
            port=db_config['port'],
            user=db_config['user'],
            password=db_config['password'],
            db='euphoria',  # euphoria 데이터베이스 사용
            charset='utf8mb4',
            autocommit=True,
            minsize=1,
            maxsize=10
        )
    return registration_db_pool

@router.post("/register/start", response_model=StartRegisterResponse)
async def start_register(request: StartRegisterRequest):
    """회원가입 시작 - registration_id 발급"""
    try:
        registration_id = str(uuid.uuid4())
        expires_at = datetime.now() + timedelta(hours=1)  # 1시간 후 만료
        
        pool = await get_registration_db_pool()
        async with pool.acquire() as conn:
            async with conn.cursor() as cursor:
                await cursor.execute(
                    """INSERT INTO temp_registrations 
                       (registration_id, language, created_at, expires_at) 
                       VALUES (%s, %s, %s, %s)""",
                    (registration_id, request.language, datetime.now(), expires_at)
                )
        
        logger.info(f"회원가입 시작: {registration_id}")
        return StartRegisterResponse(registrationId=registration_id)
    except Exception as e:
        logger.error(f"회원가입 시작 오류: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/register/consents")
async def save_consents(request: ConsentsRequest):
    """동의 사항 저장"""
    try:
        pool = await get_registration_db_pool()
        async with pool.acquire() as conn:
            async with conn.cursor() as cursor:
                # registration_id 확인
                await cursor.execute(
                    "SELECT registration_id FROM temp_registrations WHERE registration_id = %s",
                    (request.registrationId,)
                )
                result = await cursor.fetchone()
                
                if not result:
                    raise HTTPException(status_code=404, detail="등록 정보를 찾을 수 없습니다")
                
                # 동의 사항 업데이트
                await cursor.execute(
                    """UPDATE temp_registrations 
                       SET consent_health = %s, consent_finance = %s, 
                           consent_social = %s, consent_clp = %s
                       WHERE registration_id = %s""",
                    (request.consentHealth, request.consentFinance, 
                     request.consentSocial, request.consentClp, request.registrationId)
                )
        
        logger.info(f"동의 사항 저장: {request.registrationId}")
        return {"success": True}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"동의 사항 저장 오류: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/register/profile")
async def save_profile(request: ProfileRequest):
    """프로필 정보 저장"""
    try:
        pool = await get_registration_db_pool()
        async with pool.acquire() as conn:
            async with conn.cursor() as cursor:
                # registration_id 확인
                await cursor.execute(
                    "SELECT registration_id FROM temp_registrations WHERE registration_id = %s",
                    (request.registrationId,)
                )
                result = await cursor.fetchone()
                
                if not result:
                    raise HTTPException(status_code=404, detail="등록 정보를 찾을 수 없습니다")
                
                # 프로필 정보 업데이트
                await cursor.execute(
                    """UPDATE temp_registrations 
                       SET name = %s, nickname = %s
                       WHERE registration_id = %s""",
                    (request.name, request.nickname, request.registrationId)
                )
        
        logger.info(f"프로필 저장: {request.registrationId}")
        return {"success": True}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"프로필 저장 오류: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/register/credentials", response_model=TokenResponse)
async def finalize_registration(request: CredentialsRequest):
    """회원가입 완료 - 사용자 생성 및 토큰 발급"""
    try:
        logger.info(f"회원가입 완료 요청: registrationId={request.registrationId}, userid={request.userid}")
        pool = await get_registration_db_pool()
        async with pool.acquire() as conn:
            async with conn.cursor(aiomysql.DictCursor) as cursor:
                # 임시 등록 정보 가져오기
                await cursor.execute(
                    "SELECT * FROM temp_registrations WHERE registration_id = %s",
                    (request.registrationId,)
                )
                temp_reg = await cursor.fetchone()

                if not temp_reg:
                    logger.error(f"등록 정보를 찾을 수 없음: {request.registrationId}")
                    raise HTTPException(status_code=404, detail="등록 정보를 찾을 수 없습니다")

                logger.info(f"임시 등록 정보 확인 완료: {request.registrationId}")

                # userid 중복 확인
                await cursor.execute(
                    "SELECT userid FROM users WHERE userid = %s",
                    (request.userid,)
                )
                existing_user = await cursor.fetchone()

                if existing_user:
                    logger.warning(f"중복 아이디: {request.userid}")
                    raise HTTPException(status_code=400, detail="이미 사용 중인 아이디입니다")
                
                # 사용자 생성
                await cursor.execute(
                    """INSERT INTO users 
                       (userid, name, nickname, password_hash, language, 
                        consent_health, consent_finance, consent_social, consent_clp,
                        created_at, updated_at)
                       VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)""",
                    (request.userid, temp_reg['name'], temp_reg['nickname'], request.passwordHash,
                     temp_reg['language'], temp_reg['consent_health'], temp_reg['consent_finance'],
                     temp_reg['consent_social'], temp_reg['consent_clp'], 
                     datetime.now(), datetime.now())
                )
                
                # 임시 등록 정보 삭제
                await cursor.execute(
                    "DELETE FROM temp_registrations WHERE registration_id = %s",
                    (request.registrationId,)
                )
        
        # JWT 토큰 발급
        token = None
        if request.issueToken:
            payload = {
                "userid": request.userid,
                "user_id": request.userid,  # user_id도 추가 (일관성을 위해)
                "exp": datetime.utcnow() + timedelta(hours=JWT_EXPIRATION_HOURS)
            }
            token = jwt.encode(payload, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)

        logger.info(f"회원가입 완료: {request.userid}, JWT에 user_id 포함됨")
        return TokenResponse(token=token)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"회원가입 완료 오류: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/login", response_model=TokenResponse)
async def login(request: LoginRequest):
    """로그인"""
    try:
        pool = await get_registration_db_pool()
        async with pool.acquire() as conn:
            async with conn.cursor(aiomysql.DictCursor) as cursor:
                # 사용자 확인
                await cursor.execute(
                    "SELECT userid, password_hash FROM users WHERE userid = %s",
                    (request.userid,)
                )
                user = await cursor.fetchone()
                
                if not user:
                    raise HTTPException(status_code=401, detail="아이디 또는 비밀번호가 올바르지 않습니다")
                
                # 비밀번호 확인
                if user['password_hash'] != request.passwordHash:
                    raise HTTPException(status_code=401, detail="아이디 또는 비밀번호가 올바르지 않습니다")
        
        # JWT 토큰 발급
        payload = {
            "userid": request.userid,
            "user_id": request.userid,  # user_id도 추가 (일관성을 위해)
            "exp": datetime.utcnow() + timedelta(hours=JWT_EXPIRATION_HOURS)
        }
        token = jwt.encode(payload, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)

        logger.info(f"로그인 성공: {request.userid}, JWT에 user_id 포함됨")
        return TokenResponse(token=token)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"로그인 오류: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/logout")
async def logout():
    """로그아웃 (클라이언트에서 토큰 삭제)"""
    return {"success": True, "message": "로그아웃되었습니다"}


# ==================== Calendar API ====================

@router.get("/calendar/events")
async def get_calendar_events(authorization: str = Header(None)):
    """일정 목록 조회 (DB 사용)"""
    from services.auth_service import get_current_user_id, set_request_context
    set_request_context(authorization)
    user_id = get_current_user_id()

    try:
        db_service = await get_db_llm_service()
        async with db_service.pool.acquire() as conn:
            async with conn.cursor(aiomysql.DictCursor) as cursor:
                await cursor.execute(
                    """SELECT id, event_type, title, description, location,
                              event_date, event_time, is_completed, reminder_enabled
                       FROM calendar_events
                       WHERE user_id = %s
                       ORDER BY event_date, event_time""",
                    (user_id,)
                )
                events = await cursor.fetchall()

                # 날짜/시간 변환
                for event in events:
                    if event.get('event_date'):
                        event['event_date'] = event['event_date'].isoformat()
                    if event.get('event_time'):
                        td = event['event_time']
                        hours, remainder = divmod(td.seconds, 3600)
                        minutes, _ = divmod(remainder, 60)
                        event['event_time'] = f"{hours:02d}:{minutes:02d}"
                    event['event_id'] = event['id']

                logger.info(f"일정 조회: user_id={user_id}, count={len(events)}")
                return events
    except Exception as e:
        logger.error(f"일정 조회 오류: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/calendar/events")
async def create_calendar_event(event: dict, authorization: str = Header(None)):
    """일정 추가 (DB 사용)"""
    from services.auth_service import get_current_user_id, set_request_context
    set_request_context(authorization)
    user_id = get_current_user_id()

    try:
        db_service = await get_db_llm_service()
        async with db_service.pool.acquire() as conn:
            async with conn.cursor(aiomysql.DictCursor) as cursor:
                # 일정 추가
                await cursor.execute(
                    """INSERT INTO calendar_events
                       (user_id, event_type, title, description, location, event_date, event_time, is_completed)
                       VALUES (%s, %s, %s, %s, %s, %s, %s, %s)""",
                    (
                        user_id,
                        event.get("event_type", "일정"),
                        event.get("summary") or event.get("title", "제목 없음"),
                        event.get("description", ""),
                        event.get("location", ""),
                        event.get("startDateTime", "").split("T")[0] if event.get("startDateTime") else event.get("event_date", ""),
                        event.get("startDateTime", "").split("T")[1][:8] if event.get("startDateTime") and "T" in event.get("startDateTime", "") else event.get("time", "09:00:00"),
                        False
                    )
                )
                await conn.commit()

                new_id = cursor.lastrowid

                # 추가된 일정 조회
                await cursor.execute(
                    """SELECT id, event_type, title, description, location,
                              event_date, event_time, is_completed
                       FROM calendar_events WHERE id = %s""",
                    (new_id,)
                )
                new_event = await cursor.fetchone()

                # 시간 변환
                if new_event.get('event_date'):
                    new_event['event_date'] = new_event['event_date'].isoformat()
                if new_event.get('event_time'):
                    td = new_event['event_time']
                    hours, remainder = divmod(td.seconds, 3600)
                    minutes, _ = divmod(remainder, 60)
                    new_event['event_time'] = f"{hours:02d}:{minutes:02d}"
                new_event['event_id'] = new_event['id']

                logger.info(f"일정 추가: user_id={user_id}, event_id={new_id}")
                return new_event
    except Exception as e:
        logger.error(f"일정 추가 오류: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/calendar/events/{event_id}")
async def delete_calendar_event(event_id: int, authorization: str = Header(None)):
    """일정 삭제 (DB 사용)"""
    from services.auth_service import get_current_user_id, set_request_context
    set_request_context(authorization)
    user_id = get_current_user_id()

    try:
        db_service = await get_db_llm_service()
        async with db_service.pool.acquire() as conn:
            async with conn.cursor() as cursor:
                await cursor.execute(
                    "DELETE FROM calendar_events WHERE id = %s AND user_id = %s",
                    (event_id, user_id)
                )
                await conn.commit()

                if cursor.rowcount == 0:
                    raise HTTPException(status_code=404, detail="일정을 찾을 수 없습니다")

                logger.info(f"일정 삭제: user_id={user_id}, event_id={event_id}")
                return {"success": True, "message": "일정이 삭제되었습니다"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"일정 삭제 오류: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/speech/text-to-speech")
async def text_to_speech(request: dict):
    """
    텍스트를 음성으로 변환 (TTS)
    """
    try:
        from services.tts_service import TTSService
        from fastapi.responses import StreamingResponse
        
        text = request.get("text", "")
        language = request.get("language", "ko")
        slow = request.get("slow", False)
        
        if not text:
            raise HTTPException(status_code=400, detail="텍스트가 필요합니다")
        
        tts_service = TTSService()
        audio_buffer = tts_service.text_to_speech(text, language, slow)
        
        logger.info(f"TTS 요청: {text[:50]}...")
        
        return StreamingResponse(
            audio_buffer,
            media_type="audio/mpeg",
            headers={
                "Content-Disposition": "attachment; filename=speech.mp3"
            }
        )
    except Exception as e:
        logger.error(f"TTS 처리 오류: {e}")
        raise HTTPException(status_code=500, detail=str(e))
