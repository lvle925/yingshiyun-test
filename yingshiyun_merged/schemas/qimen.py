from pydantic import BaseModel, Field, field_validator
from typing import Optional, List
import ast

class QimenLLMRequest(BaseModel):
    """奇门LLM服务请求模型"""
    appid: str = Field(..., description="应用ID")
    prompt: str = Field(..., description="用户的问题")
    format: str = Field("json", description="响应格式")
    ftime: int = Field(..., description="时间戳")
    sign: str = Field(..., description="签名")
    session_id: Optional[str] = Field(None, description="会话ID")
    hl_ymd: Optional[str] = Field(None, description="日期参数")
    skip_intent_check: int = Field(0, description="是否跳过意图识别")

    model_config = {
        "json_schema_extra": {
            "example": {
                "appid": "yingshi_appid",
                "prompt": "我今天适合做什么？",
                "format": "json",
                "ftime": 1700000000,
                "sign": "你的签名字符串"
            }
        }
    }

class QimenDayRequest(BaseModel):
    """奇门择日请求模型"""
    gender: str = Field(..., description="性别")
    birthday: str = Field(..., description="生日")
    tag: str = Field(..., description="事项标签")
    startDate: str = Field(..., description="开始日期")
    endDate: str = Field(..., description="结束日期")

class QimenCalendarRequest(BaseModel):
    """奇门时辰日历请求模型"""
    gender: str = Field(..., description="性别")
    birthday: str = Field(..., description="生日")
    tag: str = Field(..., description="事项标签")
    startDate: str = Field(..., description="开始日期")
    endDate: str = Field(..., description="结束日期")

class QimenAttributesRequest(BaseModel):
    """奇门属性请求模型"""
    pass
