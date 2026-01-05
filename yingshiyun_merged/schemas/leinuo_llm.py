from pydantic import BaseModel, Field, field_validator
from typing import Optional, List
import ast

class LeinuoLLMRequest(BaseModel):
    """雷诺LLM服务请求模型"""
    appid: str = Field(..., description="应用ID")
    prompt: str = Field(..., description="用户的问题，将用于生成LLM提示词")
    format: str = Field("json", description="响应格式，默认为json")
    ftime: int = Field(..., description="时间戳 (整数)，用于签名验证")
    sign: str = Field(..., description="请求签名，用于验证请求完整性")
    session_id: Optional[str] = Field(None, description="会话ID")
    hl_ymd: Optional[str] = Field(None, description="可选的日期参数")
    skip_intent_check: int = Field(0, description="是否跳过意图识别，0=不跳过（默认），1=跳过直接进行智慧卡占卜")
    card_number_pool: Optional[List[int]] = Field(
        None,
        description="可选的卡牌编号列表，将从这个列表中随机抽取3个数字作为卡牌编号。如果未提供、列表无效，或提供了无法解析的字符串，则从所有可用卡牌中抽取。"
    )

    @field_validator('card_number_pool', mode='before')
    @classmethod
    def parse_and_validate_card_number_pool(cls, v):
        if v is None:
            return None
        if isinstance(v, str):
            try:
                v = ast.literal_eval(v)
            except (ValueError, SyntaxError) as e:
                raise ValueError("card_number_pool: 输入字符串不是有效的列表字面量。")
        if not isinstance(v, list):
            raise ValueError('card_number_pool 必须是列表或表示列表的有效字符串。')
        if len(v) < 3:
            raise ValueError('card_number_pool 必须包含至少 3 个数字')
        if not all(isinstance(i, int) for i in v):
            raise ValueError('card_number_pool 必须只包含整数')
        return v

    model_config = {
        "json_schema_extra": {
            "example": {
                "appid": "yingshi_appid",
                "prompt": "我今天的感情运势如何？",
                "format": "json",
                "ftime": 1700000000,
                "sign": "你的签名字符串",
                "skip_intent_check": 0
            }
        }
    }
