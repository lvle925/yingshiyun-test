import os
from dotenv import load_dotenv

# 1. 加载 .env 文件
load_dotenv()


# =========================================================
# 服务基础配置 (新增部分，解决报错)
# =========================================================
# 默认监听 0.0.0.0 允许外部访问，默认端口 8000，默认日志级别 INFO
SERVICE_HOST = os.getenv("SERVICE_HOST", "0.0.0.0")
SERVICE_PORT = int(os.getenv("SERVICE_PORT", 8000))
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

# =========================================================
# LLM 基础配置
# =========================================================
VLLM_API_BASE_URL = os.getenv("VLLM_API_BASE_URL")
VLLM_MODEL_NAME = os.getenv("VLLM_MODEL_NAME")

# 【关键修复】这里补上了 API_KEY，解决 ImportError
API_KEY = os.getenv("API_KEY")

VLLM_REQUEST_TIMEOUT_SECONDS = float(os.getenv("VLLM_REQUEST_TIMEOUT_SECONDS", 60.0))

# --- VLLM并发控制 ---
VLLM_CONCURRENT_LIMIT = int(os.getenv('VLLM_CONCURRENT_LIMIT', 10))
VLLM_SLOT_WAIT_TIMEOUT_SECONDS = int(os.getenv('VLLM_SLOT_WAIT_TIMEOUT_SECONDS', 120))
VLLM_MAX_RETRIES = 5
VLLM_RETRY_DELAY_SECONDS = 5.0

# =========================================================
# 数据库配置
# =========================================================
DB_CONFIG = {
    'user': os.getenv('DB_USER', "root"),
    'password': os.getenv('DB_PASSWORD', "bAm5b&mp"),
    'host': os.getenv('DB_HOST', "192.168.1.106"),
    'port': int(os.getenv('DB_PORT', 3306)),
    'db': os.getenv('DB_NAME', "yingshi"),
    'autocommit': True
}

# =========================================================
# Redis 与 会话管理
# =========================================================
REDIS_URL = os.getenv("REDIS_URL")
# 默认保留6轮对话
MAX_HISTORY_MESSAGES = int(os.getenv("MAX_HISTORY_MESSAGES", "6"))
# 会话过期时间 24小时
SESSION_TTL = int(os.getenv("SESSION_TTL", "86400"))

# =========================================================
# 资源文件路径
# =========================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CARDS_CSV_PATH = os.getenv('CARDS_CSV_PATH', os.path.join(BASE_DIR, "assets", "雷牌信息集合.csv"))
MEANINGS_CSV_PATH = os.getenv('MEANINGS_CSV_PATH', os.path.join(BASE_DIR, "assets", "卡牌vs牌号.csv"))

# =========================================================
# 签名密钥
# =========================================================
APP_SECRETS = {
    "yingshi_appid": os.getenv("APP_SECRET_yingshi_appid"),
    "test_app": os.getenv("APP_SECRET_test_app"),
    "zhongzhoullm": "zhongzhoullm",
}

# =========================================================
# 日志检查
# =========================================================
import logging
logger = logging.getLogger(__name__)

# 检查关键配置
if not REDIS_URL:
    logger.warning("⚠️ 警告: REDIS_URL 未配置，会话记忆功能可能无法使用！")
if not API_KEY:
    logger.warning("⚠️ 警告: API_KEY 未配置，部分LLM服务可能无法使用！")