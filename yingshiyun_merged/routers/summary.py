from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
import logging

from schemas.summary import SummaryRequest
from services.summary import summary_service

logger = logging.getLogger(__name__)

router = APIRouter()

@router.post("/unified_chat_V12_25", summary="统一对话服务")
async def unified_chat(client_request: SummaryRequest, request: Request):
    """统一对话服务接口"""
    try:
        stream_generator = await summary_service.process_summary(client_request, request)
        return stream_generator
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"API Error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
