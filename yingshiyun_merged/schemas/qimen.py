from pydantic import BaseModel, Field, field_validator
from typing import Optional, List
import ast

class QimenChatRequest(BaseModel):
    appid: str = Field(..., description="应用ID")
    prompt: str = Field(..., description="用户的问题")
    format: str = Field("json", description="响应格式")
    ftime: int = Field(..., description="时间戳")
    sign: str = Field(..., description="签名")
    session_id: Optional[str] = Field(None, description="会话ID")
    hl_ymd: Optional[str] = Field(None, description="可选的日期参数")
    skip_intent_check: int = Field(0, description="是否跳过意图识别，0=不跳过，1=跳过")
    card_number_pool: Optional[List[int]] = Field(None, description="卡牌池")

    @field_validator('card_number_pool', mode='before')
    @classmethod
    def parse_and_validate_card_number_pool(cls, v):
        if v is None: return None
        if isinstance(v, str):
            try:
                v = ast.literal_eval(v)
            except: raise ValueError("Invalid list string")
        if not isinstance(v, list): raise ValueError("Must be list")
        if len(v) < 3: raise ValueError("At least 3 numbers")
        return v

    model_config = {
        "json_schema_extra": {
            "example": {
                "appid": "yingshi_appid",
                "prompt": "今天运势如何？",
                "format": "json",
                "ftime": 1700000000,
                "sign": "signature_string",
                "skip_intent_check": 0
            }
        }
    }

class QimenChatResponse(BaseModel):
    status: str = "ok"