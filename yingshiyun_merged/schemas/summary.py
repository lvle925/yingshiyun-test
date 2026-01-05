from pydantic import BaseModel, Field
from typing import Optional

class SummaryRequest(BaseModel):
    """总结服务请求模型"""
    appid: str = Field(..., description="应用ID")
    prompt: str = Field(..., description="用户问题")
    format: str = Field("json", description="响应格式")
    ftime: int = Field(..., description="时间戳")
    sign: str = Field(..., description="签名")
    session_id: Optional[str] = Field(None, description="会话ID")
