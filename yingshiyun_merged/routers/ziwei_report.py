from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
import logging

from schemas.ziwei import ZiweiReportRequest
from services.ziwei import ziwei_report_service

logger = logging.getLogger(__name__)

router = APIRouter()

@router.post("/lastYearFortuneReview", summary="紫微去年运势回顾")
async def last_year_fortune_review(client_request: ZiweiReportRequest, request: Request):
    """紫微去年运势回顾接口"""
    try:
        stream_generator = await ziwei_report_service.process_ziwei_report(client_request, request)
        return stream_generator
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"API Error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
