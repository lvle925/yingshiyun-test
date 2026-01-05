from prometheus_client import Counter, Gauge, Histogram
# 所有指标在这里统一定义
REQUESTS_RECEIVED = Counter(
    "api_requests_received_total",
    "Total number of requests received at the /chat_year endpoint."
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