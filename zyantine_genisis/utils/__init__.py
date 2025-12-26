"""
工具模块
"""

from .logger import (
    SystemLogger,
    StructuredLogger,
    PerformanceLogger,
    get_logger,
    get_structured_logger,
    get_performance_logger
)

from .error_handler import (
    ErrorHandler,
    GracefulShutdown,
    handle_error,
    retry_on_error,
    safe_execute,
    register_shutdown_handler
)

from .validators import (
    DataValidator,
    InputSanitizer,
    SchemaValidator,
    validate_email,
    sanitize_text,
    validate_schema
)

from .helpers import (
    TextHelper,
    FileHelper,
    TimeHelper,
    SecurityHelper,
    PerformanceHelper,
    AsyncHelper,
    truncate_text,
    ensure_directory,
    get_timestamp,
    generate_token,
    time_function
)

from .metrics import (
    MetricsCollector,
    SystemMetrics,
    APIMetrics,
    get_collector,
    increment_counter,
    start_timer,
    get_system_metrics,
    get_api_metrics
)

__all__ = [
    # 日志
    "SystemLogger",
    "StructuredLogger",
    "PerformanceLogger",
    "get_logger",
    "get_structured_logger",
    "get_performance_logger",

    # 错误处理
    "ErrorHandler",
    "GracefulShutdown",
    "handle_error",
    "retry_on_error",
    "safe_execute",
    "register_shutdown_handler",

    # 验证器
    "DataValidator",
    "InputSanitizer",
    "SchemaValidator",
    "validate_email",
    "sanitize_text",
    "validate_schema",

    # 辅助函数
    "TextHelper",
    "FileHelper",
    "TimeHelper",
    "SecurityHelper",
    "PerformanceHelper",
    "AsyncHelper",
    "truncate_text",
    "ensure_directory",
    "get_timestamp",
    "generate_token",
    "time_function",

    # 指标收集
    "MetricsCollector",
    "SystemMetrics",
    "APIMetrics",
    "get_collector",
    "increment_counter",
    "start_timer",
    "get_system_metrics",
    "get_api_metrics"
]