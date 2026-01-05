from pydantic import BaseModel, Field, field_validator
from typing import Optional, List
import ast

class DivinationRequest(BaseModel):
    appid: str = Field(..., description="应用ID")
    prompt: str = Field(..., description="用户的问题")
    format: str = Field("json", description="响应格式")
    ftime: int = Field(..., description="时间戳")
    sign: str = Field(..., description="签名")
    # 设置默认值为 None，这样 Swagger 也就知道它是可选的
    session_id: Optional[str] = Field(None, description="会话ID")
    hl_ymd: Optional[str] = Field(None, description="日期参数")
    card_number_pool: Optional[List[int]] = Field(None, description="卡牌池")
    score_level: str = Field("0", description="评分等级")

    @field_validator('card_number_pool', mode='before')
    @classmethod
    def parse_and_validate_card_number_pool(cls, v):
        if v is None: return None
        if isinstance(v, str):
            try:
                v = ast.literal_eval(v)
            except: raise ValueError("Invalid list string")
        if not isinstance(v, list): raise ValueError("Must be list")
        return v

    @field_validator('score_level', mode='before')
    @classmethod
    def validate_score_level(cls, v):
        return str(v) if str(v) in ["0", "1", "2"] else "0"

    # --- 关键修改：添加配置，让Swagger默认显示简洁版 ---
    model_config = {
        "json_schema_extra": {
            "example": {
                "appid": "yingshi_appid",
                "prompt": "今天我的事业运怎么样？",
                "format": "json",
                "ftime": 1700000000,
                "sign": "你的签名字符串",
                "score_level": "0"
            }
        }
    }

class DivinationResponse(BaseModel):
    status: str = "ok"