# ============ 记忆系统数据模型 ============
"""
统一的数据模型定义，包含所有记忆相关的数据类和枚举类型。
将分散在 memory_manager.py 和 memory_store.py 中的模型统一到此文件。
"""
from dataclasses import dataclass, asdict, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Any, Optional, Union


# ============ 枚举类型定义 ============

class MemoryType(Enum):
    """记忆类型枚举"""
    CONVERSATION = "conversation"
    EXPERIENCE = "experience"
    USER_PROFILE = "user_profile"
    SYSTEM_EVENT = "system_event"
    KNOWLEDGE = "knowledge"
    EMOTION = "emotion"
    STRATEGY = "strategy"
    TEMPORAL = "temporal"


class MemoryPriority(Enum):
    """记忆优先级"""
    CRITICAL = "critical"  # 关键记忆，永不删除
    HIGH = "high"          # 高频访问，重要记忆
    MEDIUM = "medium"      # 常规记忆
    LOW = "low"            # 低频访问，可压缩
    ARCHIVE = "archive"    # 归档记忆


class MemoryLifecycleStage(Enum):
    """记忆生命周期阶段"""
    ACTIVE = "active"       # 活跃期：新创建，频繁访问
    MATURE = "mature"       # 成熟期：稳定，定期访问
    STABLE = "stable"       # 稳定期：访问频率降低
    DECLINING = "declining" # 衰退期：访问频率很低
    ARCHIVED = "archived"   # 归档期：极少访问，已归档
    EXPIRED = "expired"     # 过期期：等待清理


class StorageTier(Enum):
    """存储层级"""
    HOT = "hot"    # 热数据：频繁访问，内存存储
    WARM = "warm"  # 温数据：偶尔访问，压缩存储
    COLD = "cold"  # 冷数据：很少访问，归档存储


class DeduplicationStrategy(Enum):
    """去重策略"""
    SKIP = "skip"                    # 跳过重复的记忆
    OVERWRITE = "overwrite"          # 覆盖现有的记忆
    MERGE = "merge"                  # 合并记忆内容
    UPDATE_METADATA = "update_metadata"  # 仅更新元数据


class MemoryRetrievalStrategy(Enum):
    """记忆检索策略"""
    SEMANTIC = "semantic"    # 语义检索
    KEYWORD = "keyword"      # 关键词检索
    HYBRID = "hybrid"        # 混合检索
    RECENT = "recent"        # 最近优先
    PRIORITY = "priority"    # 优先级优先


class MemorySecurityLevel(Enum):
    """记忆安全级别"""
    PUBLIC = "public"           # 公开
    INTERNAL = "internal"       # 内部
    CONFIDENTIAL = "confidential"  # 机密
    RESTRICTED = "restricted"   # 受限


class MemoryPrivacyPolicy(Enum):
    """记忆隐私策略"""
    ALLOW_ALL = "allow_all"     # 允许所有访问
    ALLOW_OWNER = "allow_owner" # 仅允许所有者访问
    ALLOW_NONE = "allow_none"   # 禁止所有访问


# ============ 数据类定义 ============

@dataclass
class MemoryRecord:
    """
    记忆记录数据类
    
    统一的记忆记录结构，用于存储和检索记忆。
    """
    memory_id: str
    content: Union[str, List[Dict]]
    memory_type: MemoryType
    metadata: Dict[str, Any]
    tags: List[str]
    priority: MemoryPriority
    created_at: datetime
    last_accessed: Optional[datetime] = None
    access_count: int = 0
    emotional_intensity: float = 0.5
    strategic_score: float = 0.0
    relevance_score: float = 0.0
    size_bytes: int = 0
    version: int = 1
    updated_at: Optional[datetime] = None
    lifecycle_stage: MemoryLifecycleStage = MemoryLifecycleStage.ACTIVE

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        data = asdict(self)
        data['memory_type'] = self.memory_type.value
        data['priority'] = self.priority.value
        data['lifecycle_stage'] = self.lifecycle_stage.value
        data['created_at'] = self.created_at.isoformat()
        if self.last_accessed:
            data['last_accessed'] = self.last_accessed.isoformat()
        if self.updated_at:
            data['updated_at'] = self.updated_at.isoformat()
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MemoryRecord":
        """从字典创建实例"""
        return cls(
            memory_id=data['memory_id'],
            content=data['content'],
            memory_type=MemoryType(data['memory_type']) if isinstance(data['memory_type'], str) else data['memory_type'],
            metadata=data.get('metadata', {}),
            tags=data.get('tags', []),
            priority=MemoryPriority(data['priority']) if isinstance(data['priority'], str) else data['priority'],
            created_at=datetime.fromisoformat(data['created_at']) if isinstance(data['created_at'], str) else data['created_at'],
            last_accessed=datetime.fromisoformat(data['last_accessed']) if data.get('last_accessed') and isinstance(data['last_accessed'], str) else data.get('last_accessed'),
            access_count=data.get('access_count', 0),
            emotional_intensity=data.get('emotional_intensity', 0.5),
            strategic_score=data.get('strategic_score', 0.0),
            relevance_score=data.get('relevance_score', 0.0),
            size_bytes=data.get('size_bytes', 0),
            version=data.get('version', 1),
            updated_at=datetime.fromisoformat(data['updated_at']) if data.get('updated_at') and isinstance(data['updated_at'], str) else data.get('updated_at'),
            lifecycle_stage=MemoryLifecycleStage(data.get('lifecycle_stage', 'active')) if isinstance(data.get('lifecycle_stage'), str) else data.get('lifecycle_stage', MemoryLifecycleStage.ACTIVE)
        )

    @property
    def age_hours(self) -> float:
        """获取记忆年龄（小时）"""
        return (datetime.now() - self.created_at).total_seconds() / 3600

    @property
    def access_recency(self) -> float:
        """获取访问新近度分数"""
        if not self.last_accessed:
            return 0.0
        hours_since_access = (datetime.now() - self.last_accessed).total_seconds() / 3600
        return max(0.0, 1.0 - (hours_since_access / 168))  # 一周衰减

    def update_access(self) -> None:
        """更新访问信息"""
        self.last_accessed = datetime.now()
        self.access_count += 1


@dataclass
class ShortTermMemory:
    """
    短期记忆数据模型
    
    用于存储临时的对话记忆，有生存时间限制。
    """
    memory_id: str
    content: str
    conversation_id: str
    metadata: Dict[str, Any]
    created_at: datetime = field(default_factory=datetime.now)
    expires_at: float = 0.0  # Unix timestamp
    ttl: int = 3600  # 默认1小时

    def __post_init__(self):
        """初始化后处理"""
        if self.expires_at == 0.0:
            self.expires_at = self.created_at.timestamp() + self.ttl

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "memory_id": self.memory_id,
            "content": self.content,
            "conversation_id": self.conversation_id,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat(),
            "expires_at": self.expires_at,
            "ttl": self.ttl
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ShortTermMemory":
        """从字典创建对象"""
        instance = cls(
            memory_id=data["memory_id"],
            content=data["content"],
            conversation_id=data["conversation_id"],
            metadata=data.get("metadata", {}),
            ttl=data.get("ttl", 3600)
        )
        instance.created_at = datetime.fromisoformat(data["created_at"]) if isinstance(data.get("created_at"), str) else data.get("created_at", datetime.now())
        instance.expires_at = data.get("expires_at", instance.created_at.timestamp() + instance.ttl)
        return instance

    def is_expired(self) -> bool:
        """检查是否过期"""
        return datetime.now().timestamp() > self.expires_at

    def extend_ttl(self, additional_seconds: int) -> None:
        """延长生存时间"""
        self.expires_at += additional_seconds


@dataclass
class PerformanceMetric:
    """性能指标数据类"""
    operation: str
    duration_ms: float
    timestamp: datetime
    success: bool
    memory_id: Optional[str] = None
    memory_type: Optional[str] = None
    cache_hit: bool = False
    error_message: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "operation": self.operation,
            "duration_ms": self.duration_ms,
            "timestamp": self.timestamp.isoformat(),
            "success": self.success,
            "memory_id": self.memory_id,
            "memory_type": self.memory_type,
            "cache_hit": self.cache_hit,
            "error_message": self.error_message
        }


@dataclass
class SearchResult:
    """搜索结果数据类"""
    memory_id: str
    content: str
    metadata: Dict[str, Any]
    similarity_score: float
    final_score: float
    memory_type: str
    tags: List[str] = field(default_factory=list)
    emotional_intensity: float = 0.5
    strategic_value: Dict[str, Any] = field(default_factory=dict)
    linked_tool: Optional[str] = None
    created_at: Optional[str] = None
    access_count: int = 0
    importance_score: float = 5.0
    context_relevance: float = 0.0
    topic_relevance: float = 0.0
    time_relevance: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return asdict(self)


@dataclass
class MemoryStats:
    """记忆统计信息"""
    total_memories: int = 0
    by_type: Dict[str, int] = field(default_factory=dict)
    by_priority: Dict[str, int] = field(default_factory=dict)
    tags_distribution: Dict[str, int] = field(default_factory=dict)
    topics_distribution: Dict[str, int] = field(default_factory=dict)
    access_counts: Dict[str, int] = field(default_factory=dict)
    average_emotional_intensity: float = 0.0
    average_strategic_score: float = 0.0
    cache_hit_rate: float = 0.0
    average_response_time_ms: float = 0.0
    last_cleanup_time: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        data = asdict(self)
        if self.last_cleanup_time:
            data['last_cleanup_time'] = self.last_cleanup_time.isoformat()
        return data


# ============ 配置数据类 ============

@dataclass
class StorageTierConfig:
    """存储层级配置"""
    max_size: int
    min_access_count: int
    max_age_hours: int


@dataclass
class MemorySystemConfig:
    """记忆系统配置"""
    # 缓存配置
    cache_size: int = 1000
    cache_ttl_hours: int = 24
    
    # 短期记忆配置
    short_term_ttl: int = 3600
    short_term_max_size: int = 1000
    
    # 分层存储配置
    hot_tier: StorageTierConfig = field(default_factory=lambda: StorageTierConfig(500, 10, 24))
    warm_tier: StorageTierConfig = field(default_factory=lambda: StorageTierConfig(2000, 3, 168))
    cold_tier: StorageTierConfig = field(default_factory=lambda: StorageTierConfig(10000, 0, 720))
    
    # 去重配置
    similarity_threshold: float = 0.85
    enable_deduplication: bool = True
    
    # 上下文窗口配置
    max_context_window_size: int = 20
    min_context_window_size: int = 5
    context_window_dynamic_weight: float = 0.8
    
    # 语义记忆地图配置
    max_semantic_map_size: int = 5000
    semantic_map_cleanup_threshold: float = 0.3


# ============ 类型别名 ============

MemoryContent = Union[str, List[Dict[str, Any]]]
MemoryMetadata = Dict[str, Any]
