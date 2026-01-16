"""
日志工具模块 - 提供统一的日志记录功能
"""
import logging
import logging.handlers
import os
import sys
import json
import threading
from datetime import datetime
from typing import Optional, Dict, Any, ContextManager, Union
from pathlib import Path
from contextvars import ContextVar

# 上下文变量，用于存储请求ID和其他上下文信息
request_id_var: ContextVar[Optional[str]] = ContextVar('request_id', default=None)
correlation_id_var: ContextVar[Optional[str]] = ContextVar('correlation_id', default=None)
context_data_var: ContextVar[Dict[str, Any]] = ContextVar('context_data', default={})


class SystemLogger:
    """系统日志记录器"""

    _instance = None
    _initialized = False
    _lock = threading.Lock()

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(SystemLogger, cls).__new__(cls)
            return cls._instance

    def __init__(self):
        if not self._initialized:
            with self._lock:
                if not self._initialized:
                    self.loggers: Dict[str, logging.Logger] = {}
                    self.default_log_level = logging.INFO
                    self.log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
                    self.structured_log_format = '%(message)s'
                    self.date_format = '%Y-%m-%d %H:%M:%S'
                    self.log_dir = "logs"
                    self._initialized = True

                    # 创建日志目录
                    Path(self.log_dir).mkdir(exist_ok=True)

    def get_logger(self, name: str = "zyantine",
                   level: Optional[int] = None,
                   log_to_file: bool = True,
                   max_bytes: int = 10 * 1024 * 1024,  # 10MB
                   backup_count: int = 5,
                   structured: bool = False,
                   json_indent: Optional[int] = None) -> logging.Logger:
        """
        获取或创建日志记录器

        Args:
            name: 日志器名称
            level: 日志级别
            log_to_file: 是否记录到文件
            max_bytes: 单个日志文件最大大小
            backup_count: 备份文件数量
            structured: 是否使用结构化日志格式
            json_indent: JSON缩进，None表示紧凑格式

        Returns:
            logging.Logger实例
        """
        if name in self.loggers:
            return self.loggers[name]

        # 创建日志记录器
        logger = logging.getLogger(name)

        # 设置日志级别
        if level is None:
            level = self.default_log_level
        logger.setLevel(level)

        # 清除现有处理器
        logger.handlers.clear()

        # 创建控制台处理器
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        
        if structured:
            console_formatter = logging.Formatter(self.structured_log_format)
        else:
            console_formatter = logging.Formatter(self.log_format, datefmt=self.date_format)
        
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)

        # 如果需要记录到文件
        if log_to_file:
            # 创建文件处理器
            log_file = Path(self.log_dir) / f"{name}_{datetime.now().strftime('%Y%m%d')}.log"
            file_handler = logging.handlers.RotatingFileHandler(
                log_file,
                maxBytes=max_bytes,
                backupCount=backup_count,
                encoding='utf-8'
            )
            file_handler.setLevel(level)
            
            if structured:
                file_formatter = logging.Formatter(self.structured_log_format)
            else:
                file_formatter = logging.Formatter(self.log_format, datefmt=self.date_format)
            
            file_handler.setFormatter(file_formatter)
            logger.addHandler(file_handler)

        # 防止日志传递给父记录器
        logger.propagate = False

        self.loggers[name] = logger
        return logger

    def set_default_level(self, level: int):
        """设置默认日志级别"""
        self.default_log_level = level
        for logger in self.loggers.values():
            logger.setLevel(level)
            for handler in logger.handlers:
                handler.setLevel(level)

    def get_all_loggers(self) -> Dict[str, logging.Logger]:
        """获取所有日志记录器"""
        return self.loggers.copy()

    def close_all(self):
        """关闭所有日志处理器"""
        for logger in self.loggers.values():
            for handler in logger.handlers:
                handler.close()
        self.loggers.clear()


class StructuredLogger:
    """结构化日志记录器"""

    def __init__(self, logger_name: str = "structured", structured: bool = True):
        self.logger = SystemLogger().get_logger(logger_name, structured=structured)
        self.context = {}
        self.service_name = os.getenv("SERVICE_NAME", "zyantine")
        self.environment = os.getenv("ENVIRONMENT", "development")

    def add_context(self, **kwargs):
        """添加上下文信息"""
        self.context.update(kwargs)

    def clear_context(self):
        """清除上下文信息"""
        self.context.clear()

    def set_service_name(self, service_name: str):
        """设置服务名称"""
        self.service_name = service_name

    def set_environment(self, environment: str):
        """设置环境"""
        self.environment = environment

    def log(self, level: str, message: str, **kwargs):
        """记录结构化日志"""
        # 获取上下文变量
        request_id = request_id_var.get()
        correlation_id = correlation_id_var.get()
        context_data = context_data_var.get()
        
        # 构建基础日志数据
        log_data = {
            "timestamp": datetime.now().isoformat(),
            "level": level.upper(),
            "message": message,
            "service": self.service_name,
            "environment": self.environment,
            "thread_id": threading.get_ident(),
            "process_id": os.getpid(),
            **self.context,
            **context_data,
            **kwargs
        }
        
        # 添加请求ID和关联ID（如果存在）
        if request_id:
            log_data["request_id"] = request_id
        if correlation_id:
            log_data["correlation_id"] = correlation_id

        log_message = json.dumps(log_data, ensure_ascii=False, indent=2 if kwargs.get("pretty", False) else None)

        if level.lower() == "debug":
            self.logger.debug(log_message)
        elif level.lower() == "info":
            self.logger.info(log_message)
        elif level.lower() == "warning":
            self.logger.warning(log_message)
        elif level.lower() == "error":
            self.logger.error(log_message)
        elif level.lower() == "critical":
            self.logger.critical(log_message)
    
    def bind(self, **kwargs) -> 'StructuredLogger':
        """绑定上下文，返回新的日志记录器实例"""
        new_logger = StructuredLogger()
        new_logger.context = self.context.copy()
        new_logger.context.update(kwargs)
        new_logger.service_name = self.service_name
        new_logger.environment = self.environment
        return new_logger

    def debug(self, message: str, **kwargs):
        """记录调试信息"""
        self.log("debug", message, **kwargs)

    def info(self, message: str, **kwargs):
        """记录信息"""
        self.log("info", message, **kwargs)

    def warning(self, message: str, **kwargs):
        """记录警告"""
        self.log("warning", message, **kwargs)

    def error(self, message: str, **kwargs):
        """记录错误"""
        self.log("error", message, **kwargs)

    def critical(self, message: str, **kwargs):
        """记录严重错误"""
        self.log("critical", message, **kwargs)


class PerformanceLogger:
    """性能日志记录器"""

    def __init__(self, logger_name: str = "performance"):
        self.logger = SystemLogger().get_logger(logger_name, structured=True)
        self.metrics = {}
        self.start_times = {}
        self.structured_logger = StructuredLogger(logger_name)
        self.enable_stats = True  # 是否启用性能统计
        self.stats_interval = 60  # 统计输出间隔（秒）
        self.last_stats_time = datetime.now().timestamp()

    def start_timer(self, operation: str):
        """开始计时"""
        self.start_times[operation] = datetime.now()

    def stop_timer(self, operation: str) -> float:
        """停止计时并返回耗时（秒）"""
        if operation not in self.start_times:
            return 0.0

        elapsed = (datetime.now() - self.start_times[operation]).total_seconds()

        # 记录性能指标
        self.log_performance(operation, elapsed)

        # 更新指标统计
        if operation not in self.metrics:
            self.metrics[operation] = {
                "count": 0,
                "total_time": 0.0,
                "min_time": float('inf'),
                "max_time": 0.0
            }

        stats = self.metrics[operation]
        stats["count"] += 1
        stats["total_time"] += elapsed
        stats["min_time"] = min(stats["min_time"], elapsed)
        stats["max_time"] = max(stats["max_time"], elapsed)

        del self.start_times[operation]
        return elapsed

    def log_performance(self, operation: str, elapsed: float, **kwargs):
        """记录性能日志"""
        # 计算各种时间单位
        elapsed_ms = elapsed * 1000
        elapsed_us = elapsed * 1_000_000
        
        # 构建性能日志数据
        performance_data = {
            "operation": operation,
            "elapsed_seconds": round(elapsed, 6),
            "elapsed_ms": round(elapsed_ms, 3),
            "elapsed_us": round(elapsed_us, 1),
            "category": kwargs.get("category", "general"),
            "count": kwargs.get("count", 1),
            "throughput": round(kwargs.get("count", 1) / elapsed, 2) if elapsed > 0 else 0,
            **kwargs
        }
        
        # 使用结构化日志记录
        self.structured_logger.info(
            f"Performance metric for {operation}",
            **performance_data
        )
        
        # 检查是否需要输出统计信息
        current_time = datetime.now().timestamp()
        if self.enable_stats and current_time - self.last_stats_time >= self.stats_interval:
            self.log_stats()
            self.last_stats_time = current_time
    
    def log_stats(self):
        """记录性能统计信息"""
        metrics = self.get_metrics()
        if metrics:
            self.structured_logger.info(
                "Performance statistics",
                metrics=metrics,
                interval=self.stats_interval
            )

    def get_metrics(self) -> Dict[str, Dict[str, Any]]:
        """获取性能指标统计"""
        result = {}
        for operation, stats in self.metrics.items():
            if stats["count"] > 0:
                result[operation] = {
                    "count": stats["count"],
                    "total_time": round(stats["total_time"], 3),
                    "avg_time": round(stats["total_time"] / stats["count"], 3),
                    "min_time": round(stats["min_time"], 3),
                    "max_time": round(stats["max_time"], 3)
                }
        return result


# 快捷函数
def get_logger(name: str = "zyantine") -> logging.Logger:
    """获取日志记录器"""
    return SystemLogger().get_logger(name)


def get_structured_logger(name: str = "structured") -> StructuredLogger:
    """获取结构化日志记录器"""
    return StructuredLogger(name)


def get_performance_logger(name: str = "performance") -> PerformanceLogger:
    """获取性能日志记录器"""
    return PerformanceLogger(name)


# 上下文管理工具
def set_request_id(request_id: str):
    """设置请求ID"""
    request_id_var.set(request_id)


def get_request_id() -> Optional[str]:
    """获取请求ID"""
    return request_id_var.get()


def set_correlation_id(correlation_id: str):
    """设置关联ID"""
    correlation_id_var.set(correlation_id)


def get_correlation_id() -> Optional[str]:
    """获取关联ID"""
    return correlation_id_var.get()


def set_context_data(**kwargs):
    """设置上下文数据"""
    current_data = context_data_var.get().copy()
    current_data.update(kwargs)
    context_data_var.set(current_data)


def get_context_data() -> Dict[str, Any]:
    """获取上下文数据"""
    return context_data_var.get()


def clear_context_data():
    """清除上下文数据"""
    context_data_var.set({})


class LogContext(ContextManager):
    """日志上下文管理器"""
    
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.original_data = {}
    
    def __enter__(self):
        """进入上下文"""
        # 保存原始上下文
        self.original_data = context_data_var.get().copy()
        
        # 更新上下文
        current_data = self.original_data.copy()
        current_data.update(self.kwargs)
        context_data_var.set(current_data)
        
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """退出上下文"""
        # 恢复原始上下文
        context_data_var.set(self.original_data)


class RequestContext(ContextManager):
    """请求上下文管理器"""
    
    def __init__(self, request_id: Optional[str] = None, correlation_id: Optional[str] = None, **kwargs):
        self.request_id = request_id or f"req_{datetime.now().timestamp()}_{os.getpid()}"
        self.correlation_id = correlation_id or self.request_id
        self.kwargs = kwargs
        self.original_request_id = None
        self.original_correlation_id = None
        self.original_data = {}
    
    def __enter__(self):
        """进入上下文"""
        # 保存原始上下文
        self.original_request_id = request_id_var.get()
        self.original_correlation_id = correlation_id_var.get()
        self.original_data = context_data_var.get().copy()
        
        # 设置新上下文
        request_id_var.set(self.request_id)
        correlation_id_var.set(self.correlation_id)
        
        current_data = self.original_data.copy()
        current_data.update(self.kwargs)
        context_data_var.set(current_data)
        
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """退出上下文"""
        # 恢复原始上下文
        request_id_var.set(self.original_request_id)
        correlation_id_var.set(self.original_correlation_id)
        context_data_var.set(self.original_data)


# 便捷装饰器
def with_log_context(**context_kwargs):
    """带日志上下文的装饰器"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            with LogContext(**context_kwargs):
                return func(*args, **kwargs)
        return wrapper
    return decorator


def with_request_context(request_id_param: Optional[str] = None, **context_kwargs):
    """带请求上下文的装饰器"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            # 尝试从参数中获取request_id
            request_id = None
            if request_id_param and request_id_param in kwargs:
                request_id = kwargs[request_id_param]
            
            with RequestContext(request_id=request_id, **context_kwargs):
                return func(*args, **kwargs)
        return wrapper
    return decorator