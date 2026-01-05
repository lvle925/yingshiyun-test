from fastapi import APIRouter, HTTPException, Depends
from fastapi.responses import StreamingResponse
import hmac
import hashlib
import logging

from schemas.leinuo import DivinationRequest
from services.leinuo import day_service
from config import APP_SECRETS

logger = logging.getLogger(__name__)

router = APIRouter()

# 签名验证逻辑（从原服务中提取）
def verify_signature(request: DivinationRequest):
    app_secret = APP_SECRETS.get(request.appid)
    if not app_secret:
        raise HTTPException(status_code=401, detail="未授权: 无效的 AppID。")

    # 构建待签名参数
    params = request.dict(exclude={'sign', 'card_number_pool', 'hl_ymd', 'score_level'})
    # 注意：原代码逻辑中 score_level 也是 exclude 的，这里保持一致
    
    sorted_params = dict(sorted({k: str(v) for k, v in params.items()}.items()))
    string_to_sign = "".join(f"{k}{v}" for k, v in sorted_params.items())
    
    secret_bytes = app_secret.encode('utf-8')
    string_to_sign_bytes = string_to_sign.encode('utf-8')
    calculated_sign = hmac.new(secret_bytes, string_to_sign_bytes, hashlib.sha256).hexdigest()

    if calculated_sign != request.sign:
        raise HTTPException(status_code=403, detail="禁止访问: 签名验证失败。")
    return True

@router.post("/chat_daily_leipai", summary="雷诺曼每日运势")
async def chat_daily_leipai(request: DivinationRequest):
    """
    雷诺曼每日运势接口
    """
    # 1. 验证签名
    verify_signature(request)
    
    # 2. 调用服务层逻辑
    try:
        stream_generator = day_service.process_leinuo_divination(request)
        return StreamingResponse(stream_generator, media_type="text/plain; charset=utf-8")
    except Exception as e:
        logger.error(f"API Error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))