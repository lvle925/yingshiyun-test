from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
import hmac
import hashlib
import logging

from schemas.qimen import QimenChatRequest
from services.qimen import llm_service
from config import APP_SECRETS

logger = logging.getLogger(__name__)
router = APIRouter()

def verify_signature(request: QimenChatRequest):
    app_secret = APP_SECRETS.get(request.appid)
    if not app_secret: raise HTTPException(401, "Invalid AppID")
    
    params = request.dict(exclude={'sign', 'card_number_pool', 'hl_ymd', 'skip_intent_check'})
    sorted_params = dict(sorted({k: str(v) for k, v in params.items()}.items()))
    string_to_sign = "".join(f"{k}{v}" for k, v in sorted_params.items())
    
    calc_sign = hmac.new(app_secret.encode(), string_to_sign.encode(), hashlib.sha256).hexdigest()
    if calc_sign != request.sign: raise HTTPException(403, "Invalid Signature")

@router.post("/chat_endpoints_V12_25", summary="奇门LLM对话")
async def chat_endpoints_v12_25(request: QimenChatRequest):
    verify_signature(request)
    try:
        return StreamingResponse(
            llm_service.process_qimen_chat(request), 
            media_type="text/plain; charset=utf-8"
        )
    except Exception as e:
        logger.error(f"API Error: {e}", exc_info=True)
        raise HTTPException(500, str(e))