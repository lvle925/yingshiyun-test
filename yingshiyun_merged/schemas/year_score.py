from pydantic import BaseModel, Field
from typing import Optional

class YearScoreRequest(BaseModel):
    """年运势评分请求模型"""
    appid: str = Field(..., description="应用ID")
    prompt: str = Field(..., description="用户问题")
    format: str = Field("json", description="响应格式")
    ftime: int = Field(..., description="时间戳")
    sign: str = Field(..., description="签名")
