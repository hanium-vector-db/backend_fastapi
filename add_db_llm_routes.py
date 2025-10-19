# routes.py에 추가할 코드

# 1. import 섹션에 추가
from services.db_llm_service import DBLLMService
from utils.config_loader import config

# 2. 전역 변수 섹션에 추가
db_llm_service = None

# 3. 초기화 함수 추가
async def get_db_llm_service():
    global db_llm_service
    if db_llm_service is None:
        logger.info("DB LLM 서비스 초기화 중...")
        db_config = config.mariadb_config
        db_llm_service = DBLLMService(db_config)
        await db_llm_service.initialize()
    return db_llm_service

# 4. 새로운 엔드포인트들 추가

class ChatWithContextRequest(BaseModel):
    message: str
    user_id: str  # 사용자 ID
    model_key: str = None
    stream: bool = False

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
                "model_info": handler.get_model_info()
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
