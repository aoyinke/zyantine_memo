from typing import Optional, Any


class MemoryException(Exception):
    """记忆系统基础异常类"""
    
    def __init__(self, message: str, memory_id: Optional[str] = None, details: Optional[dict] = None):
        self.message: str = message
        self.memory_id: Optional[str] = memory_id
        self.details: Optional[dict] = details
        super().__init__(self.message)
    
    def __str__(self) -> str:
        if self.memory_id:
            return f"{self.message} (记忆ID: {self.memory_id})"
        return self.message


class MemoryStorageError(MemoryException):
    """记忆存储错误"""
    pass


class MemoryRetrievalError(MemoryException):
    """记忆检索错误"""
    pass


class MemorySearchError(MemoryException):
    """记忆搜索错误"""
    pass


class MemoryValidationError(MemoryException):
    """记忆验证错误"""
    pass


class MemoryNotFoundError(MemoryException):
    """记忆未找到错误"""
    pass


class MemoryCacheError(MemoryException):
    """记忆缓存错误"""
    pass


class MemoryIndexError(MemoryException):
    """记忆索引错误"""
    pass


class MemoryTTLExpiredError(MemoryException):
    """记忆TTL过期错误"""
    pass


class MemorySizeExceededError(MemoryException):
    """记忆大小超限错误"""
    
    def __init__(self, message: str, memory_id: Optional[str] = None, 
                 size: Optional[int] = None, max_size: Optional[int] = None):
        self.size: Optional[int] = size
        self.max_size: Optional[int] = max_size
        details: Optional[dict] = {"size": size, "max_size": max_size} if size and max_size else None
        super().__init__(message, memory_id, details)


class MemoryMetadataError(MemoryException):
    """记忆元数据错误"""
    pass


class MemoryTypeError(MemoryException):
    """记忆类型错误"""
    pass


class MemoryPriorityError(MemoryException):
    """记忆优先级错误"""
    pass


class MemorySerializationError(MemoryException):
    """记忆序列化错误"""
    pass


class MemoryDeserializationError(MemoryException):
    """记忆反序列化错误"""
    pass


class MemoryDuplicateError(MemoryException):
    """记忆重复错误"""
    pass


class MemoryCleanupError(MemoryException):
    """记忆清理错误"""
    pass


class MemoryStatsError(MemoryException):
    """记忆统计错误"""
    pass


class MemoryConfigError(MemoryException):
    """记忆配置错误"""
    pass


class MemoryConnectionError(MemoryException):
    """记忆连接错误"""
    pass


class MemoryTimeoutError(MemoryException):
    """记忆超时错误"""
    pass


class MemoryConcurrentAccessError(MemoryException):
    """记忆并发访问错误"""
    pass


def handle_memory_error(error: Exception, context: Optional[str] = None) -> MemoryException:
    """将通用异常转换为记忆系统异常"""
    error_message: str = str(error)
    
    if context:
        error_message = f"{context}: {error_message}"
    
    if isinstance(error, MemoryException):
        return error
    
    if "not found" in error_message.lower():
        return MemoryNotFoundError(error_message)
    
    if "validation" in error_message.lower():
        return MemoryValidationError(error_message)
    
    if "cache" in error_message.lower():
        return MemoryCacheError(error_message)
    
    if "index" in error_message.lower():
        return MemoryIndexError(error_message)
    
    if "timeout" in error_message.lower():
        return MemoryTimeoutError(error_message)
    
    if "connection" in error_message.lower():
        return MemoryConnectionError(error_message)
    
    if "size" in error_message.lower() or "exceed" in error_message.lower():
        return MemorySizeExceededError(error_message)
    
    if "duplicate" in error_message.lower():
        return MemoryDuplicateError(error_message)
    
    return MemoryException(error_message)
