"""
日志工具模块 - 提供统一的日志记录功能
"""
import logging
import logging.handlers
import os
import sys
import json
from datetime import datetime
from typing import Optional, Dict, Any
from pathlib import Path


class SystemLogger:
    """系统日志记录器"""

    _instance = None
    _initialized = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(SystemLogger, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        if not self._initialized:
            self.loggers: Dict[str, logging.Logger] = {}
            self.default_log_level = logging.INFO
            self.log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            self.date_format = '%Y-%m-%d %H:%M:%S'
            self.log_dir = "logs"
            self._initialized = True

            # 创建日志目录
            Path(self.log_dir).mkdir(exist_ok=True)

    def get_logger(self, name: str = "zyantine",
                   level: Optional[int] = None,
                   log_to_file: bool = True,
                   max_bytes: int = 10 * 1024 * 1024,  # 10MB
                   backup_count: int = 5) -> logging.Logger:
        """
        获取或创建日志记录器

        Args:
            name: 日志器名称
            level: 日志级别
            log_to_file: 是否记录到文件
            max_bytes: 单个日志文件最大大小
            backup_count: 备份文件数量

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

    def __init__(self, logger_name: str = "structured"):
        self.logger = SystemLogger().get_logger(logger_name)
        self.context = {}

    def add_context(self, **kwargs):
        """添加上下文信息"""
        self.context.update(kwargs)

    def clear_context(self):
        """清除上下文信息"""
        self.context.clear()

    def log(self, level: str, message: str, **kwargs):
        """记录结构化日志"""
        log_data = {
            "timestamp": datetime.now().isoformat(),
            "level": level.upper(),
            "message": message,
            **self.context,
            **kwargs
        }

        log_message = json.dumps(log_data, ensure_ascii=False)

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
        self.logger = SystemLogger().get_logger(logger_name)
        self.metrics = {}
        self.start_times = {}

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
        log_data = {
            "timestamp": datetime.now().isoformat(),
            "operation": operation,
            "elapsed_seconds": round(elapsed, 3),
            "elapsed_ms": round(elapsed * 1000, 1),
            **kwargs
        }

        self.logger.info(f"PERFORMANCE: {json.dumps(log_data, ensure_ascii=False)}")

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