from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
import logging

from schemas.leinuo_llm import LeinuoLLMRequest
from services.leinuo import leinuo_llm_service as llm_service

logger = logging.getLogger(__name__)

router = APIRouter()

@router.post("/chat_endpoints_V12_25", summary="雷诺LLM智慧卡对话")
async def chat_endpoint(client_request: LeinuoLLMRequest, request: Request):
    """
    雷诺LLM智慧卡对话接口
    支持意图识别、会话管理、智慧卡占卜等功能
    """
    try:
        stream_generator = await llm_service.process_leinuo_llm_chat(client_request, request)
        return StreamingResponse(stream_generator, media_type="text/plain; charset=utf-8")
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"API Error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
