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
from services.internal_db_service import InternalDBService
import logging
import torch
import json
import asyncio

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

# Initialize handlers (lazy loading)
llm_handler = None
embedding_handler = None
rag_service = None
internal_db_service = None

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
        logger.info("Internal DB 서비스 초기화 중...")
        llm = get_llm_handler()
        emb = get_embedding_handler()
        internal_db_service = InternalDBService(llm, emb)
    return internal_db_service

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
async def get_internal_db_tables():
    """내부 데이터베이스의 테이블 목록을 조회합니다"""
    try:
        service = get_internal_db_service()
        tables = await service.get_db_tables()
        
        return {
            "tables": tables,
            "total_count": len(tables),
            "status": "success"
        }
        
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