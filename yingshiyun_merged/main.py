"""
萤石云主应用入口
整合所有9个项目的API服务
"""
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import logging
from config import SERVICE_HOST, SERVICE_PORT, LOG_LEVEL

# 配置日志
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 创建FastAPI应用
app = FastAPI(
    title="萤石云 - 统一API服务",
    description="整合雷诺、奇门、紫微等9个项目的统一服务",
    version="1.0.0"
)

# 添加CORS中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==================== 导入路由 ====================
# 注意: 这些导入需要在实际创建路由文件后取消注释

from routers import leinuo_day, leinuo_llm
from routers import qimen_day, qimen_llm
from routers import ziwei_llm, ziwei_report, ziwei_year
from routers import year_score, summary

# ==================== 注册路由 ====================
# 雷诺相关路由
app.include_router(leinuo_day.router, tags=["雷诺每日运势"])
app.include_router(leinuo_llm.router, tags=["LLM雷诺"])

# 奇门相关路由
app.include_router(qimen_day.router, tags=["奇门择日"])
app.include_router(qimen_llm.router, tags=["LLM奇门"])

# 紫微相关路由
app.include_router(ziwei_llm.router, tags=["LLM紫微"])
app.include_router(ziwei_report.router, tags=["紫微报告"])
app.include_router(ziwei_year.router, tags=["紫微年度报告"])

# 其他路由
app.include_router(year_score.router, tags=["年运势评分"])
app.include_router(summary.router, tags=["总结服务"])

# ==================== 健康检查端点 ====================
@app.get("/health")
async def health_check():
    """健康检查端点"""
    return {
        "status": "healthy",
        "service": "yingshiyun-unified",
        "version": "1.0.0"
    }

@app.get("/")
async def root():
    """根路径"""
    return {
        "message": "萤石云统一API服务",
        "docs": "/docs",
        "health": "/health"
    }

# ==================== 启动事件 ====================
@app.on_event("startup")
async def startup_event():
    """应用启动时执行"""
    logger.info("萤石云统一服务启动中...")
    # 这里可以添加数据库连接池初始化、缓存初始化等
    logger.info("服务启动完成")

@app.on_event("shutdown")
async def shutdown_event():
    """应用关闭时执行"""
    logger.info("萤石云统一服务关闭中...")
    # 这里可以添加资源清理逻辑
    logger.info("服务已关闭")

# ==================== 主函数 ====================
if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=SERVICE_HOST,
        port=SERVICE_PORT,
        reload=True,
        log_level=LOG_LEVEL.lower()
    )
