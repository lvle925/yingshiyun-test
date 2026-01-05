from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse
import logging

from schemas.year_score import YearScoreRequest
from services.year_score import year_score_service

logger = logging.getLogger(__name__)

router = APIRouter()

@router.post("/yearFortuneScore", summary="年运势评分")
async def year_fortune_score(client_request: YearScoreRequest, request: Request):
    """年运势评分接口"""
    try:
        result = await year_score_service.process_year_score(client_request, request)
        return result
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"API Error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
