from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
import logging

from schemas.ziwei import ZiweiLLMRequest
from services.ziwei import ziwei_llm_service

logger = logging.getLogger(__name__)

router = APIRouter()

@router.post("/chat_yingshis_V12_25", summary="紫微LLM对话")
async def chat_yingshis(client_request: ZiweiLLMRequest, request: Request):
    """紫微LLM对话接口"""
    try:
        stream_generator = await ziwei_llm_service.process_ziwei_llm_chat(client_request, request)
        return stream_generator
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"API Error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
