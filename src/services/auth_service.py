"""
인증 관련 서비스
"""
from fastapi import Request, Header, HTTPException
from typing import Optional
import jwt
import logging

logger = logging.getLogger(__name__)

# JWT 설정 (routes.py와 동일하게 유지)
JWT_SECRET_KEY = "your-secret-key-change-this-in-production"
JWT_ALGORITHM = "HS256"

# 전역 변수로 현재 요청 컨텍스트 저장
_current_request_context: Optional[dict] = None

def set_request_context(authorization: Optional[str]):
    """요청 컨텍스트 설정 (미들웨어에서 호출)"""
    global _current_request_context
    _current_request_context = {"authorization": authorization}

def get_current_user_id() -> str:
    """현재 로그인한 사용자 ID를 반환합니다.

    JWT 토큰에서 user_id를 추출합니다. 토큰이 없거나 유효하지 않으면 HTTPException을 발생시킵니다.
    """
    global _current_request_context

    # 요청 컨텍스트에서 Authorization 헤더 가져오기
    if _current_request_context and _current_request_context.get("authorization"):
        authorization = _current_request_context["authorization"]

        if authorization and authorization.startswith("Bearer "):
            token = authorization.replace("Bearer ", "")
            try:
                payload = jwt.decode(token, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])
                # user_id가 없으면 userid를 fallback으로 사용 (하위 호환성)
                user_id = payload.get("user_id") or payload.get("userid")

                if not user_id:
                    logger.error("❌ JWT 토큰에 user_id 또는 userid가 없습니다")
                    raise HTTPException(status_code=401, detail="유효하지 않은 토큰입니다")

                logger.info(f"✅ get_current_user_id: JWT에서 추출한 user_id = {user_id}")
                return user_id
            except jwt.ExpiredSignatureError:
                logger.warning("⚠️ get_current_user_id: JWT 토큰이 만료되었습니다")
                raise HTTPException(status_code=401, detail="토큰이 만료되었습니다. 다시 로그인해주세요.")
            except jwt.InvalidTokenError as e:
                logger.warning(f"⚠️ get_current_user_id: JWT 토큰이 유효하지 않습니다: {e}")
                raise HTTPException(status_code=401, detail="유효하지 않은 토큰입니다")

    # JWT 토큰이 없는 경우
    logger.error("❌ get_current_user_id: JWT 토큰이 없습니다")
    raise HTTPException(status_code=401, detail="인증이 필요합니다. 로그인해주세요.")
