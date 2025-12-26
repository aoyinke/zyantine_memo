"""
错误处理器模块 - 统一的错误处理机制
"""
import traceback
import sys
from typing import Optional, Callable, Any, Dict
from functools import wraps
from datetime import datetime

from .logger import get_logger, StructuredLogger


class ErrorHandler:
    """错误处理器"""

    def __init__(self, logger_name: str = "error_handler"):
        self.logger = StructuredLogger(logger_name)
        self.error_stats = {
            "total_errors": 0,
            "by_type": {},
            "by_function": {}
        }

    def handle_error(self, error: Exception,
                     context: Optional[Dict] = None,
                     raise_again: bool = False,
                     custom_message: Optional[str] = None):
        """
        处理错误

        Args:
            error: 异常对象
            context: 上下文信息
            raise_again: 是否重新抛出异常
            custom_message: 自定义错误消息
        """
        # 收集错误信息
        error_info = {
            "error_type": type(error).__name__,
            "error_message": str(error),
            "traceback": traceback.format_exc(),
            "timestamp": datetime.now().isoformat()
        }

        if context:
            error_info["context"] = context

        # 更新统计
        self.error_stats["total_errors"] += 1

        error_type = error_info["error_type"]
        self.error_stats["by_type"][error_type] = self.error_stats["by_type"].get(error_type, 0) + 1

        # 记录错误
        self.logger.error(
            custom_message or f"发生错误: {error_type}",
            **error_info
        )

        # 如果需要重新抛出
        if raise_again:
            raise error

    def retry_on_error(self, max_retries: int = 3,
                       delay: float = 1.0,
                       backoff: float = 2.0,
                       exceptions: tuple = (Exception,)):
        """
        装饰器：在发生错误时重试

        Args:
            max_retries: 最大重试次数
            delay: 初始延迟（秒）
            backoff: 延迟倍数
            exceptions: 捕获的异常类型
        """

        def decorator(func: Callable) -> Callable:
            @wraps(func)
            def wrapper(*args, **kwargs):
                last_exception = None
                current_delay = delay

                for attempt in range(max_retries + 1):
                    try:
                        if attempt > 0:
                            self.logger.info(f"重试 {func.__name__}, 第{attempt}次尝试")
                        return func(*args, **kwargs)

                    except exceptions as e:
                        last_exception = e

                        if attempt < max_retries:
                            self.handle_error(
                                e,
                                context={
                                    "function": func.__name__,
                                    "attempt": attempt + 1,
                                    "max_retries": max_retries,
                                    "next_delay": current_delay
                                },
                                custom_message=f"{func.__name__} 失败，准备重试"
                            )

                            # 等待
                            import time
                            time.sleep(current_delay)
                            current_delay *= backoff
                        else:
                            self.handle_error(
                                e,
                                context={
                                    "function": func.__name__,
                                    "attempt": attempt + 1,
                                    "max_retries": max_retries,
                                    "status": "max_retries_exceeded"
                                },
                                custom_message=f"{func.__name__} 达到最大重试次数",
                                raise_again=True
                            )

                # 如果所有重试都失败
                raise last_exception

            return wrapper

        return decorator

    def safe_execute(self, func: Callable,
                     default_return: Any = None,
                     log_error: bool = True,
                     context: Optional[Dict] = None,
                     *args, **kwargs) -> Any:
        """
        安全执行函数

        Args:
            func: 要执行的函数
            default_return: 出错时的默认返回值
            log_error: 是否记录错误
            context: 上下文信息
            *args, **kwargs: 函数参数

        Returns:
            函数执行结果或默认返回值
        """
        try:
            return func(*args, **kwargs)
        except Exception as e:
            if log_error:
                self.handle_error(
                    e,
                    context={
                        "function": func.__name__,
                        "context": context or {}
                    },
                    custom_message=f"安全执行失败: {func.__name__}"
                )
            return default_return

    def get_error_statistics(self) -> Dict[str, Any]:
        """获取错误统计信息"""
        return {
            "total_errors": self.error_stats["total_errors"],
            "by_type": self.error_stats["by_type"],
            "by_function": self.error_stats["by_function"],
            "error_rate": self._calculate_error_rate()
        }

    def _calculate_error_rate(self) -> float:
        """计算错误率（简化版本）"""
        # 这里可以根据实际业务逻辑调整
        total_calls = 1000  # 假设值，实际应用中应该从其他地方获取
        if total_calls > 0:
            return (self.error_stats["total_errors"] / total_calls) * 100
        return 0.0

    def clear_statistics(self):
        """清除统计信息"""
        self.error_stats = {
            "total_errors": 0,
            "by_type": {},
            "by_function": {}
        }


class GracefulShutdown:
    """优雅关闭处理器"""

    def __init__(self):
        self.shutdown_handlers = []
        self.is_shutting_down = False

    def register_shutdown_handler(self, handler: Callable,
                                  name: Optional[str] = None):
        """
        注册关闭处理器

        Args:
            handler: 处理函数
            name: 处理器名称
        """
        handler_info = {
            "handler": handler,
            "name": name or handler.__name__
        }
        self.shutdown_handlers.append(handler_info)

    def shutdown(self, reason: str = "正常关闭"):
        """执行关闭过程"""
        if self.is_shutting_down:
            return

        self.is_shutting_down = True
        logger = get_logger("shutdown")

        logger.info(f"开始优雅关闭: {reason}")

        # 逆序执行关闭处理器（后进先出）
        for handler_info in reversed(self.shutdown_handlers):
            try:
                handler_name = handler_info["name"]
                logger.info(f"执行关闭处理器: {handler_name}")
                handler_info["handler"]()
                logger.info(f"关闭处理器完成: {handler_name}")
            except Exception as e:
                logger.error(f"关闭处理器执行失败 {handler_name}: {str(e)}")

        logger.info("优雅关闭完成")

    def setup_signal_handlers(self):
        """设置信号处理器"""
        import signal

        def signal_handler(signum, frame):
            signal_name = signal.Signals(signum).name
            self.shutdown(f"收到信号: {signal_name}")
            sys.exit(0)

        # 注册常见信号
        signals = [signal.SIGINT, signal.SIGTERM]
        if hasattr(signal, 'SIGHUP'):
            signals.append(signal.SIGHUP)

        for sig in signals:
            signal.signal(sig, signal_handler)


# 全局错误处理器实例
global_error_handler = ErrorHandler()
global_shutdown_handler = GracefulShutdown()


# 快捷函数
def handle_error(error: Exception, **kwargs):
    """处理错误（快捷函数）"""
    return global_error_handler.handle_error(error, **kwargs)


def retry_on_error(max_retries: int = 3, **kwargs):
    """重试装饰器（快捷函数）"""
    return global_error_handler.retry_on_error(max_retries, **kwargs)


def safe_execute(func: Callable, **kwargs):
    """安全执行函数（快捷函数）"""
    return global_error_handler.safe_execute(func, **kwargs)


def register_shutdown_handler(handler: Callable, **kwargs):
    """注册关闭处理器（快捷函数）"""
    return global_shutdown_handler.register_shutdown_handler(handler, **kwargs)