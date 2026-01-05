from prometheus_client import Counter, Gauge, Histogram, REGISTRY, GC_COLLECTOR, PLATFORM_COLLECTOR, PROCESS_COLLECTOR
import logging

logger = logging.getLogger(__name__)

# ---------------------------------------------------------
# 1. 基础配置：注销默认的 Python 进程指标（可选，根据需求保留或移除）
# ---------------------------------------------------------
def unregister_default_collectors():
    """注销默认的 Prometheus 收集器，避免指标过多。"""
    try:
        REGISTRY.unregister(GC_COLLECTOR)
        REGISTRY.unregister(PLATFORM_COLLECTOR)
        REGISTRY.unregister(PROCESS_COLLECTOR)
    except Exception as e:
        logger.warning(f"Failed to unregister default collectors: {e}")

# ---------------------------------------------------------
# 2. 指标统一定义
# ---------------------------------------------------------
# 使用全局字典或直接定义，确保在模块加载时只初始化一次
# 注意：prometheus_client 在同一个进程中如果多次定义同名指标会抛出 ValueError

# API 请求计数
REQUESTS_RECEIVED = Counter(
    "api_requests_received_total",
    "Total number of requests received at the API endpoints."
)

# VLLM 请求尝试计数
VLLM_REQUESTS_SENT_ATTEMPTS = Counter(
    "vllm_requests_sent_attempts_total",
    "Total attempts to send requests to VLLM after acquiring the semaphore."
)

# VLLM 成功响应计数
VLLM_RESPONSES_SUCCESS = Counter(
    "vllm_responses_success_total",
    "Total number of successful and complete responses from VLLM."
)

# VLLM 失败响应计数
VLLM_RESPONSES_FAILED = Counter(
    "vllm_responses_failed_total",
    "Total number of failed/errored responses from VLLM.",
    ["reason"]
)

# ---------------------------------------------------------
# 3. 导出所有指标
# ---------------------------------------------------------
__all__ = [
    "REQUESTS_RECEIVED",
    "VLLM_REQUESTS_SENT_ATTEMPTS",
    "VLLM_RESPONSES_SUCCESS",
    "VLLM_RESPONSES_FAILED",
    "unregister_default_collectors"
]
