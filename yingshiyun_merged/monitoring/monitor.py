# monitoring.py

from prometheus_client import Counter, REGISTRY, GC_COLLECTOR, PLATFORM_COLLECTOR, PROCESS_COLLECTOR
from dataclasses import dataclass, field

# 注销默认的 Python 进程指标
REGISTRY.unregister(GC_COLLECTOR)
REGISTRY.unregister(PLATFORM_COLLECTOR)
REGISTRY.unregister(PROCESS_COLLECTOR)

# --- 自定义应用指标 ---
REQUESTS_RECEIVED = Counter(
    "api_requests_received_total",
    "Total number of requests received at the /chat endpoint."
)

VLLM_REQUESTS_SENT_ATTEMPTS = Counter(
    "vllm_requests_sent_attempts_total",
    "Total attempts to send requests to VLLM after acquiring the semaphore."
)

VLLM_RESPONSES_SUCCESS = Counter(
    "vllm_responses_success_total",
    "Total number of successful and complete responses from VLLM."
)

VLLM_RESPONSES_FAILED = Counter(
    "vllm_responses_failed_total",
    "Total number of failed/errored responses from VLLM.",
    ["reason"]
)


def generate_request_id() -> str:
    """生成 8 位 request_id，用于链路追踪。"""
    return uuid.uuid4().hex[:8]

def log_step(
    step_name: str,
    request_id: Optional[str] = None,
    extra_data: Optional[Dict[str, Any]] = None,
    status: str = "成功",
) -> None:
    """在流程收尾或特殊节点手动写一次监控日志。"""
    rid = request_id or generate_request_id()
    monitor_logger.info(
        "req=%s | step=%s | status=%s | extra=%s",
        rid,
        step_name,
        status,
        _serialize_extra(extra_data),
    )


@dataclass
class StepMonitor:
    """用于包裹关键步骤的上下文管理器。"""

    step_name: str
    request_id: Optional[str] = None
    extra_data: Dict[str, Any] = field(default_factory=dict)
    status: str = field(default="成功", init=False)
    _start: float = field(default=0.0, init=False)

    def __enter__(self) -> "StepMonitor":
        self._start = time.perf_counter()
        rid = self.request_id or generate_request_id()
        self.request_id = rid
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        duration_ms = round((time.perf_counter() - self._start) * 1000, 2)
        if exc_type:
            self.status = "失败"
            level = monitor_logger.error
            extra = {**self.extra_data, "error": str(exc_val), "duration_ms": duration_ms}
        else:
            level = monitor_logger.info
            extra = {**self.extra_data, "duration_ms": duration_ms}

        level(
            "req=%s | step=%s | status=%s | extra=%s",
            self.request_id,
            self.step_name,
            self.status,
            _serialize_extra(extra),
        )

    def update_extra(self, **kwargs: Any) -> None:
        """动态更新附加信息。"""
        self.extra_data.update(kwargs)

    def update_status(self, status: str) -> None:
        """修改最终状态文案。"""
        self.status = status