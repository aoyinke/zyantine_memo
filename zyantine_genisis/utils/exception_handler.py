from typing import Optional, Any, Dict
import traceback
import logging

from memory.memory_exceptions import MemoryException, handle_memory_error

# 错误级别
error_levels = {
    "critical": 50,
    "error": 40,
    "warning": 30,
    "info": 20,
    "debug": 10
}

# 错误分类
error_categories = {
    "memory": "记忆系统错误",
    "api": "API调用错误",
    "config": "配置错误",
    "processing": "处理错误",
    "validation": "验证错误",
    "system": "系统错误"
}


class ZyantineException(Exception):
    """自衍体系统基础异常类"""
    
    def __init__(self, 
                 message: str, 
                 error_code: str = "ZYAN-000",
                 category: str = "system",
                 level: str = "error",
                 details: Optional[Dict[str, Any]] = None):
        self.message: str = message
        self.error_code: str = error_code
        self.category: str = category
        self.level: str = level
        self.details: Optional[Dict[str, Any]] = details
        self.traceback: str = traceback.format_exc()
        super().__init__(self.message)
    
    def __str__(self) -> str:
        return f"[{self.error_code}] {self.message} (类别: {self.category}, 级别: {self.level})"


class APIException(ZyantineException):
    """API调用异常"""
    
    def __init__(self, 
                 message: str, 
                 error_code: str = "API-000",
                 level: str = "error",
                 details: Optional[Dict[str, Any]] = None):
        super().__init__(message, error_code, "api", level, details)


class ConfigException(ZyantineException):
    """配置异常"""
    
    def __init__(self, 
                 message: str, 
                 error_code: str = "CFG-000",
                 level: str = "error",
                 details: Optional[Dict[str, Any]] = None):
        super().__init__(message, error_code, "config", level, details)


class ProcessingException(ZyantineException):
    """处理异常"""
    
    def __init__(self, 
                 message: str, 
                 error_code: str = "PRC-000",
                 level: str = "error",
                 details: Optional[Dict[str, Any]] = None):
        super().__init__(message, error_code, "processing", level, details)


class ValidationException(ZyantineException):
    """验证异常"""
    
    def __init__(self, 
                 message: str, 
                 error_code: str = "VAL-000",
                 level: str = "warning",
                 details: Optional[Dict[str, Any]] = None):
        super().__init__(message, error_code, "validation", level, details)


class ExceptionHandler:
    """统一异常处理类"""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger("exception_handler")
    
    def handle_exception(self, 
                         exception: Exception, 
                         context: Optional[str] = None,
                         default_error_code: str = "ZYAN-000") -> ZyantineException:
        """处理异常，转换为标准化异常"""
        
        # 处理记忆系统异常
        if isinstance(exception, MemoryException):
            return self._convert_memory_exception(exception)
        
        # 处理已经是ZyantineException的异常
        if isinstance(exception, ZyantineException):
            self._log_exception(exception, context)
            return exception
        
        # 处理通用异常
        return self._handle_generic_exception(exception, context, default_error_code)
    
    def _convert_memory_exception(self, exception: MemoryException) -> ZyantineException:
        """将记忆系统异常转换为ZyantineException"""
        error_code = "MEM-000"
        details = exception.details or {}
        
        if hasattr(exception, "memory_id") and exception.memory_id:
            details["memory_id"] = exception.memory_id
        
        converted_exception = ZyantineException(
            str(exception),
            error_code=error_code,
            category="memory",
            level="error",
            details=details
        )
        
        self._log_exception(converted_exception)
        return converted_exception
    
    def _handle_generic_exception(self, 
                                  exception: Exception, 
                                  context: Optional[str] = None,
                                  default_error_code: str = "ZYAN-000") -> ZyantineException:
        """处理通用异常"""
        error_message = str(exception)
        if context:
            error_message = f"{context}: {error_message}"
        
        # 根据异常类型分类
        category = "system"
        level = "error"
        
        exception_type = type(exception).__name__
        
        if "API" in exception_type or "api" in exception_type.lower():
            category = "api"
            error_code = "API-000"
        elif "Config" in exception_type or "config" in exception_type.lower():
            category = "config"
            error_code = "CFG-000"
        elif "Validation" in exception_type or "validate" in exception_type.lower():
            category = "validation"
            level = "warning"
            error_code = "VAL-000"
        elif "Process" in exception_type or "process" in exception_type.lower():
            category = "processing"
            error_code = "PRC-000"
        else:
            error_code = default_error_code
        
        # 创建标准化异常
        standardized_exception = ZyantineException(
            error_message,
            error_code=error_code,
            category=category,
            level=level,
            details={"original_exception": exception_type}
        )
        
        self._log_exception(standardized_exception, context)
        return standardized_exception
    
    def _log_exception(self, exception: ZyantineException, context: Optional[str] = None):
        """记录异常"""
        log_message = str(exception)
        if context:
            log_message = f"{context}: {log_message}"
        
        # 根据异常级别记录
        level = error_levels.get(exception.level, logging.ERROR)
        
        # 记录详细信息
        extra = {
            "error_code": exception.error_code,
            "category": exception.category,
            "level": exception.level,
            "details": exception.details
        }
        
        self.logger.log(level, log_message, extra=extra)
        
        # 如果有traceback，记录debug信息
        if hasattr(exception, "traceback") and exception.traceback:
            self.logger.debug(f"Traceback: {exception.traceback}")
    
    def handle_api_exception(self, exception: Exception, endpoint: str) -> APIException:
        """专门处理API异常"""
        context = f"API调用失败 ({endpoint})"
        return self.handle_exception(exception, context, "API-000")
    
    def handle_config_exception(self, exception: Exception, config_key: str) -> ConfigException:
        """专门处理配置异常"""
        context = f"配置错误 ({config_key})"
        return self.handle_exception(exception, context, "CFG-000")
    
    def handle_processing_exception(self, exception: Exception, process_name: str) -> ProcessingException:
        """专门处理处理异常"""
        context = f"处理失败 ({process_name})"
        return self.handle_exception(exception, context, "PRC-000")


# 创建全局异常处理器实例
global_exception_handler = ExceptionHandler()


def handle_error(exception: Exception, 
                 context: Optional[str] = None,
                 raise_again: bool = False) -> ZyantineException:
    """全局异常处理函数"""
    handled_exception = global_exception_handler.handle_exception(exception, context)
    
    if raise_again:
        raise handled_exception
    
    return handled_exception


def handle_api_error(exception: Exception, 
                     endpoint: str, 
                     raise_again: bool = False) -> APIException:
    """全局API异常处理函数"""
    handled_exception = global_exception_handler.handle_api_exception(exception, endpoint)
    
    if raise_again:
        raise handled_exception
    
    return handled_exception


def handle_config_error(exception: Exception, 
                        config_key: str, 
                        raise_again: bool = False) -> ConfigException:
    """全局配置异常处理函数"""
    handled_exception = global_exception_handler.handle_config_exception(exception, config_key)
    
    if raise_again:
        raise handled_exception
    
    return handled_exception


def handle_processing_error(exception: Exception, 
                             process_name: str, 
                             raise_again: bool = False) -> ProcessingException:
    """全局处理异常处理函数"""
    handled_exception = global_exception_handler.handle_processing_exception(exception, process_name)
    
    if raise_again:
        raise handled_exception
    
    return handled_exception


class GracefulShutdown:
    """优雅关闭处理器"""
    
    def __init__(self):
        self.shutdown_signaled = False
        self.shutdown_handlers = []
    
    def setup_signal_handlers(self):
        """设置信号处理器"""
        import signal
        
        def signal_handler(signum, frame):
            self.shutdown()
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
    
    def register_shutdown_handler(self, handler, name="unknown"):
        """注册关闭处理器"""
        self.shutdown_handlers.append((name, handler))
    
    def shutdown(self, reason="收到关闭信号"):
        """执行优雅关闭"""
        if self.shutdown_signaled:
            return
        
        self.shutdown_signaled = True
        
        logger = logging.getLogger("graceful_shutdown")
        logger.info(f"开始优雅关闭: {reason}")
        
        # 执行所有注册的关闭处理器
        for name, handler in self.shutdown_handlers:
            try:
                logger.info(f"执行关闭处理器: {name}")
                handler()
            except Exception as e:
                logger.error(f"执行关闭处理器 {name} 时出错: {e}")
        
        logger.info("优雅关闭完成")


def register_shutdown_handler(handler, name="unknown"):
    """注册关闭处理器"""
    shutdown_handler = GracefulShutdown()
    shutdown_handler.register_shutdown_handler(handler, name)
    return shutdown_handler


class EnhancedExceptionHandler(ExceptionHandler):
    """增强的异常处理器，集成错误处理统计和重试功能"""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        super().__init__(logger)
        self.error_stats = {
            "total_errors": 0,
            "by_type": {},
            "by_function": {}
        }
    
    def handle_exception(self, 
                         exception: Exception, 
                         context: Optional[str] = None,
                         default_error_code: str = "ZYAN-000",
                         function_name: Optional[str] = None) -> ZyantineException:
        """处理异常，添加统计功能"""
        # 调用父类方法处理异常
        handled_exception = super().handle_exception(exception, context, default_error_code)
        
        # 更新错误统计
        self._update_error_stats(exception, function_name)
        
        return handled_exception
    
    def _update_error_stats(self, exception: Exception, function_name: Optional[str] = None):
        """更新错误统计信息"""
        self.error_stats["total_errors"] += 1
        
        error_type = type(exception).__name__
        self.error_stats["by_type"][error_type] = self.error_stats["by_type"].get(error_type, 0) + 1
        
        if function_name:
            self.error_stats["by_function"][function_name] = self.error_stats["by_function"].get(function_name, 0) + 1
    
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
        
        def decorator(func):
            import functools
            @functools.wraps(func)
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
                            self.handle_exception(
                                e,
                                context=f"{func.__name__} 失败，准备重试",
                                function_name=func.__name__
                            )
                            
                            # 等待
                            import time
                            time.sleep(current_delay)
                            current_delay *= backoff
                        else:
                            self.handle_exception(
                                e,
                                context=f"{func.__name__} 达到最大重试次数",
                                function_name=func.__name__
                            )
                            raise
                
                # 如果所有重试都失败
                raise last_exception
            
            return wrapper
        
        return decorator
    
    def safe_execute(self, func,
                     default_return=None,
                     log_error: bool = True,
                     context: Optional[str] = None,
                     *args, **kwargs):
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
                self.handle_exception(
                    e,
                    context=context,
                    function_name=getattr(func, "__name__", "unknown")
                )
            return default_return
    
    def get_error_statistics(self):
        """获取错误统计信息"""
        return {
            "total_errors": self.error_stats["total_errors"],
            "by_type": self.error_stats["by_type"],
            "by_function": self.error_stats["by_function"]
        }
    
    def clear_error_statistics(self):
        """清除错误统计信息"""
        self.error_stats = {
            "total_errors": 0,
            "by_type": {},
            "by_function": {}
        }


# 创建增强的全局异常处理器实例
enhanced_exception_handler = EnhancedExceptionHandler()


# 增强的全局函数
def enhanced_handle_error(exception: Exception, 
                         context: Optional[str] = None,
                         raise_again: bool = False,
                         function_name: Optional[str] = None) -> ZyantineException:
    """增强的全局异常处理函数"""
    handled_exception = enhanced_exception_handler.handle_exception(
        exception, 
        context,
        function_name=function_name
    )
    
    if raise_again:
        raise handled_exception
    
    return handled_exception


def retry_on_error(max_retries: int = 3, **kwargs):
    """重试装饰器（快捷函数）"""
    return enhanced_exception_handler.retry_on_error(max_retries, **kwargs)


def safe_execute(func, **kwargs):
    """安全执行函数（快捷函数）"""
    return enhanced_exception_handler.safe_execute(func, **kwargs)


def get_error_statistics():
    """获取错误统计信息（快捷函数）"""
    return enhanced_exception_handler.get_error_statistics()


def clear_error_statistics():
    """清除错误统计信息（快捷函数）"""
    return enhanced_exception_handler.clear_error_statistics()
