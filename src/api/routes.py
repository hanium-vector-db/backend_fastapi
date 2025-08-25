import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.config import settings
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from models.llm_handler import LLMHandler
from models.embedding_handler import EmbeddingHandler
from services.rag_service import RAGService
import logging
import torch

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

class ModelSwitchRequest(BaseModel):
    model_key: str  # 전환할 모델 키

# Request models for new endpoints
class ModelRecommendationRequest(BaseModel):
    ram_gb: int = None
    gpu_gb: int = None
    use_case: str = None  # korean, coding, math, multilingual

class PerformanceComparisonRequest(BaseModel):
    model_keys: list[str] = None

# Initialize handlers (lazy loading)
llm_handler = None
embedding_handler = None
rag_service = None

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

@router.post("/generate")
async def generate_response(request: GenerateRequest):
    try:
        handler = get_llm_handler(request.model_key)
        
        if request.stream:
            # 스트리밍 응답
            def generate_stream():
                for chunk in handler.generate(request.prompt, request.max_length, stream=True):
                    yield chunk
            
            return StreamingResponse(
                generate_stream(), 
                media_type="text/plain",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "Content-Type": "text/event-stream"
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
            # 스트리밍 응답
            def chat_stream():
                for chunk in handler.chat_generate(request.message, stream=True):
                    yield chunk
            
            return StreamingResponse(
                chat_stream(), 
                media_type="text/plain",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "Content-Type": "text/event-stream"
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