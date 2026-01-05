from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
import logging

from schemas.qimen import QimenDayRequest, QimenCalendarRequest, QimenAttributesRequest
from services.qimen import api_main_day, api_main_calendar, api_main_attributes

logger = logging.getLogger(__name__)

router = APIRouter()

@router.post("/chooseGoodDay", summary="奇门择日接口")
async def choose_good_day(request: QimenDayRequest):
    """奇门择日接口"""
    try:
        return await api_main_day.process_choose_good_day(request)
    except Exception as e:
        logger.error(f"API Error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/qiMenTimeCalendar", summary="奇门时辰日历")
async def qimen_time_calendar(request: QimenCalendarRequest):
    """奇门时辰日历接口"""
    try:
        return await api_main_calendar.process_qimen_calendar(request)
    except Exception as e:
        logger.error(f"API Error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/qiMenAuspiciousInfo", summary="奇门吉凶信息")
async def qimen_auspicious_info(request: QimenAttributesRequest):
    """奇门吉凶信息接口"""
    try:
        return await api_main_attributes.process_auspicious_info(request)
    except Exception as e:
        logger.error(f"API Error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
