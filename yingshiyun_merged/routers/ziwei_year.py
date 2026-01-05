from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
import logging

from schemas.ziwei import ZiweiYearRequest
from services.ziwei import ziwei_year_service

logger = logging.getLogger(__name__)

router = APIRouter()

@router.post("/chat_year_V11_11", summary="紫微年运对话")
async def chat_year(client_request: ZiweiYearRequest, request: Request):
    """紫微年运对话接口"""
    try:
        stream_generator = await ziwei_year_service.process_ziwei_year(client_request, request)
        return stream_generator
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"API Error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
