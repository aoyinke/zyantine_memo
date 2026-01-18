# ============ 记忆系统模块 ============
"""
Zyantine 记忆系统

提供完整的记忆管理功能，包括：
- 短期记忆和长期记忆管理
- 分层存储（热/温/冷数据）
- 多维度索引（主题、标签、时间、类型等）
- 缓存和性能优化
- 记忆检索和搜索

使用示例：
    from memory import MemoryManager, MemoryType, MemoryPriority
    
    # 创建记忆管理器
    manager = MemoryManager()
    
    # 添加记忆
    memory_id = manager.add_memory(
        content="用户喜欢编程",
        memory_type=MemoryType.USER_PROFILE,
        tags=["偏好", "技术"]
    )
    
    # 搜索记忆
    results = manager.search_memories("编程相关")
"""
from typing import TYPE_CHECKING

# 数据模型（始终可用）
from .models import (
    # 枚举类型
    MemoryType,
    MemoryPriority,
    MemoryLifecycleStage,
    StorageTier,
    DeduplicationStrategy,
    MemoryRetrievalStrategy,
    MemorySecurityLevel,
    MemoryPrivacyPolicy,
    # 数据类
    MemoryRecord,
    ShortTermMemory,
    PerformanceMetric,
    SearchResult,
    MemoryStats,
    # 配置类
    StorageTierConfig,
    MemorySystemConfig,
)

# 异常类（始终可用）
from .memory_exceptions import (
    MemoryException,
    MemoryStorageError,
    MemoryRetrievalError,
    MemorySearchError,
    MemoryValidationError,
    MemoryNotFoundError,
    MemoryCacheError,
    MemoryIndexError,
    MemoryTTLExpiredError,
    MemorySizeExceededError,
    MemoryMetadataError,
    MemoryTypeError,
    MemoryPriorityError,
    MemorySerializationError,
    MemoryDeserializationError,
    MemoryDuplicateError,
    MemoryCleanupError,
    MemoryStatsError,
    MemoryConfigError,
    MemoryConnectionError,
    MemoryTimeoutError,
    MemoryConcurrentAccessError,
    handle_memory_error,
)

# 存储组件（始终可用）
from .storage import (
    BaseStorage,
    StorageInterface,
    TieredStorage,
    MemoryCache,
)

# 索引组件（始终可用）
from .indexing import (
    MemoryIndex,
    ContextWindow,
)

# 工具函数（始终可用）
from .memory_utils import (
    CacheManager,
    calculate_hash,
    generate_memory_id,
    get_current_timestamp,
    calculate_age_hours,
    is_time_expired,
    to_json_str,
    StatisticsManager,
    safe_get_config,
)

# 记忆评估器（始终可用）
from .memory_evaluator import (
    MemoryEvaluator,
    MemoryEvaluationResult,
    MemoryValueDimension,
)

# 核心组件使用延迟导入（依赖外部库 mem0）
# 这样可以在不安装 mem0 的情况下使用基础功能
def _lazy_import_core():
    """延迟导入核心组件"""
    global ZyantineMemorySystem, MemoryManager
    from .memory_store import ZyantineMemorySystem
    from .memory_manager import MemoryManager
    return ZyantineMemorySystem, MemoryManager


# 类型检查时的导入
if TYPE_CHECKING:
    from .memory_store import ZyantineMemorySystem
    from .memory_manager import MemoryManager


def __getattr__(name):
    """
    延迟导入支持
    
    当访问 ZyantineMemorySystem, MemoryManager 时才实际导入
    """
    if name in ('ZyantineMemorySystem', 'MemoryManager'):
        _lazy_import_core()
        return globals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# 公共接口
__all__ = [
    # 主要类（延迟导入）
    'MemoryManager',
    'ZyantineMemorySystem',
    
    # 枚举类型
    'MemoryType',
    'MemoryPriority',
    'MemoryLifecycleStage',
    'StorageTier',
    'DeduplicationStrategy',
    'MemoryRetrievalStrategy',
    'MemorySecurityLevel',
    'MemoryPrivacyPolicy',
    
    # 数据类
    'MemoryRecord',
    'ShortTermMemory',
    'PerformanceMetric',
    'SearchResult',
    'MemoryStats',
    'StorageTierConfig',
    'MemorySystemConfig',
    
    # 存储组件
    'BaseStorage',
    'StorageInterface',
    'TieredStorage',
    'MemoryCache',
    
    # 索引组件
    'MemoryIndex',
    'ContextWindow',
    
    # 异常类
    'MemoryException',
    'MemoryStorageError',
    'MemoryRetrievalError',
    'MemorySearchError',
    'MemoryValidationError',
    'MemoryNotFoundError',
    'MemoryCacheError',
    'MemoryIndexError',
    'MemoryTTLExpiredError',
    'MemorySizeExceededError',
    'MemoryMetadataError',
    'MemoryTypeError',
    'MemoryPriorityError',
    'MemoryCleanupError',
    'MemoryConfigError',
    'handle_memory_error',
    
    # 工具函数
    'CacheManager',
    'calculate_hash',
    'generate_memory_id',
    'get_current_timestamp',
    'StatisticsManager',
    
    # 记忆评估器
    'MemoryEvaluator',
    'MemoryEvaluationResult',
    'MemoryValueDimension',
]

# 版本信息
__version__ = '2.0.0'
__author__ = 'Zyantine Team'
