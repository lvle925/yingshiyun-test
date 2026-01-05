import logging
import aiohttp
from typing import Optional

try:
    from config import VLLM_REQUEST_TIMEOUT_SECONDS
except ImportError:
    VLLM_REQUEST_TIMEOUT_SECONDS = 60.0

logger = logging.getLogger(__name__)

# 全局变量
async_aiohttp_client: Optional[aiohttp.ClientSession] = None

# ---------------------------------------------------------
# 1. 初始化函数 (Main.py 调用)
# ---------------------------------------------------------
async def init_aiohttp_client():
    """在应用启动时调用，创建并配置全局共享的 aiohttp 客户端。"""
    global async_aiohttp_client
    if async_aiohttp_client is None or async_aiohttp_client.closed:
        logger.info("正在创建新的共享 aiohttp.ClientSession 实例...")
        connector = aiohttp.TCPConnector(
            limit=1000,
            limit_per_host=1000,
            enable_cleanup_closed=True,
            force_close=False,
            keepalive_timeout=120
        )
        async_aiohttp_client = aiohttp.ClientSession(
            connector=connector,
            timeout=aiohttp.ClientTimeout(
                total=float(VLLM_REQUEST_TIMEOUT_SECONDS),
                connect=50,
                sock_read=None
            )
        )
        logger.info("共享的 aiohttp.ClientSession 初始化完成。")
    else:
        logger.warning("共享的 aiohttp.ClientSession 已存在且未关闭，跳过重复初始化。")

# 为了兼容可能的旧命名调用，做一个别名
initialize_shared_client = init_aiohttp_client


# ---------------------------------------------------------
# 2. 获取函数 (Service层 调用 - 修复报错的关键)
# ---------------------------------------------------------
async def get_aiohttp_client() -> aiohttp.ClientSession:
    """获取全局客户端实例，如果未初始化则自动初始化"""
    global async_aiohttp_client
    if async_aiohttp_client is None or async_aiohttp_client.closed:
        logger.warning("检测到 HTTP 客户端未初始化或已关闭，正在进行延迟初始化...")
        await init_aiohttp_client()
    return async_aiohttp_client


# ---------------------------------------------------------
# 3. 关闭函数 (Main.py 调用)
# ---------------------------------------------------------
async def close_aiohttp_client():
    """在应用关闭时调用，清理共享客户端。"""
    global async_aiohttp_client
    if async_aiohttp_client and not async_aiohttp_client.closed:
        await async_aiohttp_client.close()
        logger.info("共享的 aiohttp.ClientSession 已关闭。")
    async_aiohttp_client = None

# 别名兼容
close_shared_client = close_aiohttp_client