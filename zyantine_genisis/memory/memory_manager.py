"""
记忆管理器 - 管理所有记忆相关操作（优化版）
"""
from typing import Dict, List, Any, Optional, Union, Tuple
from datetime import datetime, timedelta
import json
import hashlib
import time
import threading
from dataclasses import dataclass, asdict
from enum import Enum
import asyncio
from collections import defaultdict
import pickle

from .memory_store import ZyantineMemorySystem
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
    MemoryCleanupError,
    MemoryStatsError,
    MemoryConfigError,
    handle_memory_error
)
from config.config_manager import ConfigManager, MemoryConfig


class MemoryType(Enum):
    """记忆类型枚举"""
    CONVERSATION = "conversation"
    EXPERIENCE = "experience"
    USER_PROFILE = "user_profile"
    SYSTEM_EVENT = "system_event"
    KNOWLEDGE = "knowledge"
    EMOTION = "emotion"
    STRATEGY = "strategy"
    TEMPORAL = "temporal"  # 新增：时间相关记忆


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


class MemoryPerformanceMonitor:
    """记忆系统性能监控器"""

    def __init__(self, max_metrics: int = 10000):
        self.max_metrics = max_metrics
        self.metrics: List[PerformanceMetric] = []
        self.operation_stats: Dict[str, Dict[str, Any]] = defaultdict(lambda: {
            "count": 0,
            "success_count": 0,
            "failure_count": 0,
            "total_duration_ms": 0.0,
            "min_duration_ms": float('inf'),
            "max_duration_ms": 0.0,
            "cache_hits": 0,
            "cache_misses": 0
        })
        
        # 存储统计
        self.storage_stats: Dict[str, Any] = {
            "total_items": 0,
            "hot_items": 0,
            "warm_items": 0,
            "cold_items": 0,
            "total_size_bytes": 0,
            "compression_ratio": 1.0,
            "history": []
        }
        
        # 生命周期统计
        self.lifecycle_stats: Dict[str, int] = defaultdict(int)
        
        # 实时监控
        self.monitoring_active = False
        self.monitoring_interval = 60  # 秒
        self.monitoring_thread = None
        
        self.lock = threading.RLock()

    def record_operation(self,
                        operation: str,
                        duration_ms: float,
                        success: bool,
                        memory_id: Optional[str] = None,
                        memory_type: Optional[str] = None,
                        cache_hit: bool = False,
                        error_message: Optional[str] = None):
        """记录操作性能"""
        metric = PerformanceMetric(
            operation=operation,
            duration_ms=duration_ms,
            timestamp=datetime.now(),
            success=success,
            memory_id=memory_id,
            memory_type=memory_type,
            cache_hit=cache_hit,
            error_message=error_message
        )

        with self.lock:
            self.metrics.append(metric)
            if len(self.metrics) > self.max_metrics:
                self.metrics.pop(0)

            stats = self.operation_stats[operation]
            stats["count"] += 1
            stats["total_duration_ms"] += duration_ms
            stats["min_duration_ms"] = min(stats["min_duration_ms"], duration_ms)
            stats["max_duration_ms"] = max(stats["max_duration_ms"], duration_ms)

            if success:
                stats["success_count"] += 1
            else:
                stats["failure_count"] += 1

            if cache_hit:
                stats["cache_hits"] += 1
            else:
                stats["cache_misses"] += 1

    def get_operation_stats(self, operation: str) -> Optional[Dict[str, Any]]:
        """获取特定操作的统计信息"""
        with self.lock:
            if operation not in self.operation_stats:
                return None

            stats = self.operation_stats[operation].copy()
            if stats["count"] > 0:
                stats["avg_duration_ms"] = stats["total_duration_ms"] / stats["count"]
                stats["success_rate"] = stats["success_count"] / stats["count"]
                stats["failure_rate"] = stats["failure_count"] / stats["count"]
                stats["cache_hit_rate"] = stats["cache_hits"] / stats["count"]
            else:
                stats["avg_duration_ms"] = 0.0
                stats["success_rate"] = 0.0
                stats["failure_rate"] = 0.0
                stats["cache_hit_rate"] = 0.0

            return stats

    def get_all_stats(self) -> Dict[str, Any]:
        """获取所有操作的统计信息"""
        with self.lock:
            all_stats = {}
            for operation, stats in self.operation_stats.items():
                all_stats[operation] = self.get_operation_stats(operation)
            return all_stats

    def get_recent_metrics(self, operation: Optional[str] = None, limit: int = 100) -> List[PerformanceMetric]:
        """获取最近的性能指标"""
        with self.lock:
            if operation:
                filtered = [m for m in self.metrics if m.operation == operation]
                return filtered[-limit:]
            return self.metrics[-limit:]

    def get_performance_summary(self) -> Dict[str, Any]:
        """获取性能摘要"""
        with self.lock:
            total_operations = sum(stats["count"] for stats in self.operation_stats.values())
            total_success = sum(stats["success_count"] for stats in self.operation_stats.values())
            total_failures = sum(stats["failure_count"] for stats in self.operation_stats.values())
            total_cache_hits = sum(stats["cache_hits"] for stats in self.operation_stats.values())
            total_cache_misses = sum(stats["cache_misses"] for stats in self.operation_stats.values())

            return {
                "total_operations": total_operations,
                "total_success": total_success,
                "total_failures": total_failures,
                "overall_success_rate": total_success / total_operations if total_operations > 0 else 0.0,
                "overall_cache_hit_rate": total_cache_hits / (total_cache_hits + total_cache_misses) if (total_cache_hits + total_cache_misses) > 0 else 0.0,
                "operation_count": len(self.operation_stats),
                "metrics_stored": len(self.metrics),
                "operations": list(self.operation_stats.keys())
            }

    def clear_metrics(self):
        """清除所有性能指标"""
        with self.lock:
            self.metrics.clear()
            self.operation_stats.clear()

    def export_metrics(self, filepath: str):
        """导出性能指标到文件"""
        with self.lock:
            metrics_data = []
            for metric in self.metrics:
                metric_dict = asdict(metric)
                metric_dict['timestamp'] = metric.timestamp.isoformat()
                metrics_data.append(metric_dict)
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump({
                    "metrics": metrics_data,
                    "stats": self.get_all_stats(),
                    "summary": self.get_performance_summary(),
                    "storage_stats": self.storage_stats,
                    "lifecycle_stats": dict(self.lifecycle_stats),
                    "exported_at": datetime.now().isoformat()
                }, f, indent=2, ensure_ascii=False)

    def update_storage_stats(self, storage_stats: Dict[str, Any]):
        """更新存储统计信息"""
        with self.lock:
            self.storage_stats.update(storage_stats)
            
            # 记录历史
            history_entry = {
                "timestamp": datetime.now().isoformat(),
                "stats": storage_stats.copy()
            }
            self.storage_stats["history"].append(history_entry)
            
            # 限制历史记录数量
            if len(self.storage_stats["history"]) > 1000:
                self.storage_stats["history"] = self.storage_stats["history"][-1000:]

    def update_lifecycle_stats(self, lifecycle_stage: str):
        """更新生命周期统计"""
        with self.lock:
            self.lifecycle_stats[lifecycle_stage] += 1

    def get_storage_stats(self) -> Dict[str, Any]:
        """获取存储统计信息"""
        with self.lock:
            return self.storage_stats.copy()

    def get_lifecycle_stats(self) -> Dict[str, int]:
        """获取生命周期统计信息"""
        with self.lock:
            return dict(self.lifecycle_stats)

    def start_monitoring(self, interval: int = 60, callback=None):
        """启动实时监控"""
        with self.lock:
            if self.monitoring_active:
                return
            
            self.monitoring_active = True
            self.monitoring_interval = interval
            
            def monitoring_loop():
                while self.monitoring_active:
                    time.sleep(self.monitoring_interval)
                    if callback:
                        callback()
            
            self.monitoring_thread = threading.Thread(target=monitoring_loop, daemon=True)
            self.monitoring_thread.start()

    def stop_monitoring(self):
        """停止实时监控"""
        with self.lock:
            self.monitoring_active = False
            if self.monitoring_thread:
                self.monitoring_thread.join(timeout=5)

    def get_comprehensive_report(self) -> Dict[str, Any]:
        """获取综合性能报告"""
        with self.lock:
            return {
                "performance": {
                    "summary": self.get_performance_summary(),
                    "operations": self.get_all_stats()
                },
                "storage": self.get_storage_stats(),
                "lifecycle": self.get_lifecycle_stats(),
                "recent_metrics": [
                    {
                        "operation": m.operation,
                        "duration_ms": m.duration_ms,
                        "success": m.success,
                        "cache_hit": m.cache_hit,
                        "timestamp": m.timestamp.isoformat()
                    }
                    for m in self.get_recent_metrics(limit=50)
                ]
            }


class MemoryPriority(Enum):
    """记忆优先级"""
    CRITICAL = "critical"  # 关键记忆，永不删除
    HIGH = "high"  # 高频访问，重要记忆
    MEDIUM = "medium"  # 常规记忆
    LOW = "low"  # 低频访问，可压缩
    ARCHIVE = "archive"  # 归档记忆


class MemoryLifecycleStage(Enum):
    """记忆生命周期阶段"""
    ACTIVE = "active"  # 活跃期：新创建，频繁访问
    MATURE = "mature"  # 成熟期：稳定，定期访问
    STABLE = "stable"  # 稳定期：访问频率降低
    DECLINING = "declining"  # 衰退期：访问频率很低
    ARCHIVED = "archived"  # 归档期：极少访问，已归档
    EXPIRED = "expired"  # 过期期：等待清理


class DeduplicationStrategy(Enum):
    """去重策略"""
    SKIP = "skip"  # 跳过重复的记忆
    OVERWRITE = "overwrite"  # 覆盖现有的记忆
    MERGE = "merge"  # 合并记忆内容
    UPDATE_METADATA = "update_metadata"  # 仅更新元数据


@dataclass
class MemoryRecord:
    """记忆记录数据类（优化版）"""
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
    relevance_score: float = 0.0  # 新增：相关度分数
    size_bytes: int = 0  # 新增：内存大小
    version: int = 1  # 新增：版本号
    updated_at: Optional[datetime] = None  # 新增：更新时间
    lifecycle_stage: MemoryLifecycleStage = MemoryLifecycleStage.ACTIVE  # 新增：生命周期阶段

    def to_dict(self) -> Dict:
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


class MemoryCache:
    """记忆缓存管理类（带版本控制和失效机制）"""

    def __init__(self, max_size: int = 1000, ttl_hours: int = 24, storage_optimizer=None):
        self.max_size = max_size
        self.ttl_seconds = ttl_hours * 3600
        self.cache: Dict[str, MemoryRecord] = {}
        self.access_times: Dict[str, float] = {}
        self.versions: Dict[str, int] = {}
        self.storage_optimizer = storage_optimizer
        self.lock = threading.RLock()

    def get(self, memory_id: str) -> Optional[MemoryRecord]:
        """获取缓存项"""
        with self.lock:
            if memory_id in self.cache:
                record = self.cache[memory_id]
                # 检查是否过期
                if time.time() - self.access_times[memory_id] > self.ttl_seconds:
                    self._invalidate(memory_id)
                    return None

                # 更新访问时间
                record.last_accessed = datetime.now()
                record.access_count += 1
                self.access_times[memory_id] = time.time()
                
                # 通知存储优化器更新访问统计
                if self.storage_optimizer:
                    self.storage_optimizer._update_access_stats(memory_id)
                
                return record
        return None

    def set(self, memory_id: str, record: MemoryRecord, version: int = 1):
        """设置缓存项（带版本控制）"""
        with self.lock:
            # 如果缓存已满，淘汰最少使用的
            if len(self.cache) >= self.max_size and memory_id not in self.cache:
                self._evict_least_used()

            # 检查版本，如果版本更新则更新缓存
            current_version = self.versions.get(memory_id, 0)
            if version > current_version:
                self.cache[memory_id] = record
                self.access_times[memory_id] = time.time()
                self.versions[memory_id] = version
            elif memory_id not in self.cache:
                # 新缓存项
                self.cache[memory_id] = record
                self.access_times[memory_id] = time.time()
                self.versions[memory_id] = version

    def delete(self, memory_id: str):
        """删除缓存项"""
        with self.lock:
            self._invalidate(memory_id)

    def clear(self):
        """清空缓存"""
        with self.lock:
            self.cache.clear()
            self.access_times.clear()
            self.versions.clear()

    def invalidate_by_prefix(self, prefix: str):
        """按前缀失效缓存项"""
        with self.lock:
            keys_to_delete = [key for key in self.cache.keys() if key.startswith(prefix)]
            for key in keys_to_delete:
                self._invalidate(key)
            return len(keys_to_delete)

    def invalidate_by_tag(self, tag: str):
        """按标签失效缓存项"""
        with self.lock:
            keys_to_delete = []
            for memory_id, record in self.cache.items():
                if tag in record.tags:
                    keys_to_delete.append(memory_id)
            for key in keys_to_delete:
                self._invalidate(key)
            return len(keys_to_delete)

    def invalidate_by_memory_type(self, memory_type: MemoryType):
        """按记忆类型失效缓存项"""
        with self.lock:
            keys_to_delete = []
            for memory_id, record in self.cache.items():
                if record.memory_type == memory_type:
                    keys_to_delete.append(memory_id)
            for key in keys_to_delete:
                self._invalidate(key)
            return len(keys_to_delete)

    def get_version(self, memory_id: str) -> int:
        """获取缓存项版本号"""
        return self.versions.get(memory_id, 0)

    def _invalidate(self, memory_id: str):
        """失效单个缓存项"""
        if memory_id in self.cache:
            del self.cache[memory_id]
            del self.access_times[memory_id]
            if memory_id in self.versions:
                del self.versions[memory_id]

    def _evict_least_used(self):
        """淘汰最少使用的缓存项"""
        if not self.cache:
            return

        # 找到访问时间最久的项目
        oldest_id = min(self.access_times.items(), key=lambda x: x[1])[0]
        self._invalidate(oldest_id)

    def get_all(self) -> Dict[str, MemoryRecord]:
        """获取所有缓存项"""
        with self.lock:
            return self.cache.copy()

    def get_stats(self) -> Dict[str, Any]:
        """获取缓存统计信息"""
        with self.lock:
            return {
                "total_items": len(self.cache),
                "max_size": self.max_size,
                "utilization": len(self.cache) / self.max_size if self.max_size > 0 else 0,
                "ttl_seconds": self.ttl_seconds,
                "total_versions": len(self.versions)
            }


class StorageTier(Enum):
    """存储层级"""
    HOT = "hot"  # 热数据：频繁访问，内存存储
    WARM = "warm"  # 温数据：偶尔访问，压缩存储
    COLD = "cold"  # 冷数据：很少访问，归档存储


class MemoryStorageOptimizer:
    """记忆存储优化器 - 分层存储管理"""

    def __init__(self,
                 hot_size: int = 500,
                 warm_size: int = 2000,
                 cold_size: int = 10000):
        """
        初始化存储优化器
        
        Args:
            hot_size: 热数据层最大容量
            warm_size: 温数据层最大容量
            cold_size: 冷数据层最大容量
        """
        self.hot_size = hot_size
        self.warm_size = warm_size
        self.cold_size = cold_size
        
        # 分层存储
        self.hot_storage: Dict[str, MemoryRecord] = {}
        self.warm_storage: Dict[str, bytes] = {}  # 压缩存储
        self.cold_storage: Dict[str, bytes] = {}  # 归档存储
        
        # 索引
        self.tier_index: Dict[str, StorageTier] = {}
        self.access_frequency: Dict[str, int] = {}
        self.last_access_time: Dict[str, float] = {}
        
        self.lock = threading.RLock()

    def get(self, memory_id: str) -> Optional[MemoryRecord]:
        """
        获取记忆记录
        
        Args:
            memory_id: 记忆ID
            
        Returns:
            记忆记录，如果不存在则返回None
        """
        with self.lock:
            # 检查热数据层
            if memory_id in self.hot_storage:
                record = self.hot_storage[memory_id]
                self._update_access_stats(memory_id)
                return record
            
            # 检查温数据层
            elif memory_id in self.warm_storage:
                record = self._decompress_record(self.warm_storage[memory_id])
                self._update_access_stats(memory_id)
                # 提升到热数据层
                self._promote_to_hot(memory_id, record)
                return record
            
            # 检查冷数据层
            elif memory_id in self.cold_storage:
                record = self._decompress_record(self.cold_storage[memory_id])
                self._update_access_stats(memory_id)
                # 提升到温数据层
                self._promote_to_warm(memory_id, record)
                return record
            
            return None

    def set(self, memory_id: str, record: MemoryRecord):
        """
        存储记忆记录
        
        Args:
            memory_id: 记忆ID
            record: 记忆记录
        """
        with self.lock:
            # 更新访问统计
            self._update_access_stats(memory_id)
            
            # 根据访问频率决定存储层级
            tier = self._determine_storage_tier(memory_id)
            
            if tier == StorageTier.HOT:
                self._store_in_hot(memory_id, record)
            elif tier == StorageTier.WARM:
                self._store_in_warm(memory_id, record)
            else:
                self._store_in_cold(memory_id, record)

    def delete(self, memory_id: str):
        """
        删除记忆记录
        
        Args:
            memory_id: 记忆ID
        """
        with self.lock:
            if memory_id in self.hot_storage:
                del self.hot_storage[memory_id]
            if memory_id in self.warm_storage:
                del self.warm_storage[memory_id]
            if memory_id in self.cold_storage:
                del self.cold_storage[memory_id]
            if memory_id in self.tier_index:
                del self.tier_index[memory_id]
            if memory_id in self.access_frequency:
                del self.access_frequency[memory_id]
            if memory_id in self.last_access_time:
                del self.last_access_time[memory_id]

    def get_all(self) -> Dict[str, MemoryRecord]:
        """
        获取所有记录
        
        Returns:
            所有记录的字典
        """
        with self.lock:
            result = {}
            result.update(self.hot_storage)
            
            for memory_id, compressed_data in self.warm_storage.items():
                record = self._decompress_record(compressed_data)
                result[memory_id] = record
            
            for memory_id, compressed_data in self.cold_storage.items():
                record = self._decompress_record(compressed_data)
                result[memory_id] = record
            
            return result

    def get_stats(self) -> Dict[str, Any]:
        """
        获取存储统计信息
        
        Returns:
            统计信息字典
        """
        with self.lock:
            hot_size = sum(len(str(record.content)) for record in self.hot_storage.values())
            warm_size = sum(len(data) for data in self.warm_storage.values())
            cold_size = sum(len(data) for data in self.cold_storage.values())
            
            return {
                "total_items": len(self.tier_index),
                "hot_items": len(self.hot_storage),
                "warm_items": len(self.warm_storage),
                "cold_items": len(self.cold_storage),
                "hot_size_bytes": hot_size,
                "warm_size_bytes": warm_size,
                "cold_size_bytes": cold_size,
                "total_size_bytes": hot_size + warm_size + cold_size,
                "compression_ratio": warm_size / hot_size if hot_size > 0 else 1.0,
                "tier_distribution": {
                    "hot": len(self.hot_storage),
                    "warm": len(self.warm_storage),
                    "cold": len(self.cold_storage)
                }
            }

    def optimize_storage(self):
        """优化存储布局，自动迁移数据"""
        with self.lock:
            # 将热数据层中不常用的数据迁移到温数据层
            self._demote_hot_to_warm()
            
            # 将温数据层中不常用的数据迁移到冷数据层
            self._demote_warm_to_cold()
            
            # 将温数据层中常用的数据提升到热数据层
            self._promote_warm_to_hot()
            
            # 将冷数据层中常用的数据提升到温数据层
            self._promote_cold_to_warm()

    def _determine_storage_tier(self, memory_id: str) -> StorageTier:
        """
        确定存储层级
        
        Args:
            memory_id: 记忆ID
            
        Returns:
            存储层级
        """
        access_freq = self.access_frequency.get(memory_id, 0)
        
        if access_freq >= 10:
            return StorageTier.HOT
        elif access_freq >= 3:
            return StorageTier.WARM
        else:
            return StorageTier.COLD

    def _store_in_hot(self, memory_id: str, record: MemoryRecord):
        """存储到热数据层"""
        # 如果热数据层已满，淘汰最少使用的
        if len(self.hot_storage) >= self.hot_size:
            self._evict_least_used_hot()
        
        self.hot_storage[memory_id] = record
        self.tier_index[memory_id] = StorageTier.HOT

    def _store_in_warm(self, memory_id: str, record: MemoryRecord):
        """存储到温数据层"""
        # 如果温数据层已满，淘汰最少使用的
        if len(self.warm_storage) >= self.warm_size:
            self._evict_least_used_warm()
        
        compressed_data = self._compress_record(record)
        self.warm_storage[memory_id] = compressed_data
        self.tier_index[memory_id] = StorageTier.WARM

    def _store_in_cold(self, memory_id: str, record: MemoryRecord):
        """存储到冷数据层"""
        # 如果冷数据层已满，淘汰最少使用的
        if len(self.cold_storage) >= self.cold_size:
            self._evict_least_used_cold()
        
        compressed_data = self._compress_record(record)
        self.cold_storage[memory_id] = compressed_data
        self.tier_index[memory_id] = StorageTier.COLD

    def _promote_to_hot(self, memory_id: str, record: MemoryRecord):
        """提升到热数据层"""
        if memory_id in self.warm_storage:
            del self.warm_storage[memory_id]
        elif memory_id in self.cold_storage:
            del self.cold_storage[memory_id]
        
        self._store_in_hot(memory_id, record)

    def _promote_to_warm(self, memory_id: str, record: MemoryRecord):
        """提升到温数据层"""
        if memory_id in self.cold_storage:
            del self.cold_storage[memory_id]
        
        self._store_in_warm(memory_id, record)

    def _demote_hot_to_warm(self):
        """将热数据层中不常用的数据迁移到温数据层"""
        if len(self.hot_storage) <= self.hot_size * 0.8:
            return
        
        # 找到访问频率最低的记录
        sorted_items = sorted(
            self.access_frequency.items(),
            key=lambda x: x[1]
        )
        
        for memory_id, freq in sorted_items:
            if memory_id in self.hot_storage and freq < 10:
                record = self.hot_storage[memory_id]
                del self.hot_storage[memory_id]
                self._store_in_warm(memory_id, record)
                
                if len(self.hot_storage) <= self.hot_size * 0.8:
                    break

    def _demote_warm_to_cold(self):
        """将温数据层中不常用的数据迁移到冷数据层"""
        if len(self.warm_storage) <= self.warm_size * 0.8:
            return
        
        # 找到访问频率最低的记录
        sorted_items = sorted(
            self.access_frequency.items(),
            key=lambda x: x[1]
        )
        
        for memory_id, freq in sorted_items:
            if memory_id in self.warm_storage and freq < 3:
                compressed_data = self.warm_storage[memory_id]
                del self.warm_storage[memory_id]
                self.cold_storage[memory_id] = compressed_data
                self.tier_index[memory_id] = StorageTier.COLD
                
                if len(self.warm_storage) <= self.warm_size * 0.8:
                    break

    def _promote_warm_to_hot(self):
        """将温数据层中常用的数据提升到热数据层"""
        if len(self.hot_storage) >= self.hot_size * 0.9:
            return
        
        # 找到访问频率最高的记录
        sorted_items = sorted(
            self.access_frequency.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        for memory_id, freq in sorted_items:
            if memory_id in self.warm_storage and freq >= 10:
                record = self._decompress_record(self.warm_storage[memory_id])
                del self.warm_storage[memory_id]
                self._store_in_hot(memory_id, record)
                
                if len(self.hot_storage) >= self.hot_size * 0.9:
                    break

    def _promote_cold_to_warm(self):
        """将冷数据层中常用的数据提升到温数据层"""
        if len(self.warm_storage) >= self.warm_size * 0.9:
            return
        
        # 找到访问频率最高的记录
        sorted_items = sorted(
            self.access_frequency.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        for memory_id, freq in sorted_items:
            if memory_id in self.cold_storage and freq >= 3:
                compressed_data = self.cold_storage[memory_id]
                record = self._decompress_record(compressed_data)
                del self.cold_storage[memory_id]
                self._store_in_warm(memory_id, record)
                
                if len(self.warm_storage) >= self.warm_size * 0.9:
                    break

    def _evict_least_used_hot(self):
        """淘汰热数据层中最少使用的记录"""
        if not self.hot_storage:
            return
        
        sorted_items = sorted(
            self.last_access_time.items(),
            key=lambda x: x[1]
        )
        
        for memory_id, _ in sorted_items:
            if memory_id in self.hot_storage:
                del self.hot_storage[memory_id]
                self.tier_index[memory_id] = StorageTier.WARM
                break

    def _evict_least_used_warm(self):
        """淘汰温数据层中最少使用的记录"""
        if not self.warm_storage:
            return
        
        sorted_items = sorted(
            self.last_access_time.items(),
            key=lambda x: x[1]
        )
        
        for memory_id, _ in sorted_items:
            if memory_id in self.warm_storage:
                del self.warm_storage[memory_id]
                self.tier_index[memory_id] = StorageTier.COLD
                break

    def _evict_least_used_cold(self):
        """淘汰冷数据层中最少使用的记录"""
        if not self.cold_storage:
            return
        
        sorted_items = sorted(
            self.last_access_time.items(),
            key=lambda x: x[1]
        )
        
        for memory_id, _ in sorted_items:
            if memory_id in self.cold_storage:
                del self.cold_storage[memory_id]
                del self.tier_index[memory_id]
                break

    def _update_access_stats(self, memory_id: str):
        """更新访问统计"""
        self.access_frequency[memory_id] = self.access_frequency.get(memory_id, 0) + 1
        self.last_access_time[memory_id] = time.time()

    def _compress_record(self, record: MemoryRecord) -> bytes:
        """压缩记录"""
        data = pickle.dumps(record.to_dict())
        return data

    def _decompress_record(self, compressed_data: bytes) -> MemoryRecord:
        """解压缩记录"""
        data = pickle.loads(compressed_data)
        return MemoryRecord(
            memory_id=data['memory_id'],
            content=data['content'],
            memory_type=MemoryType(data['memory_type']),
            metadata=data['metadata'],
            tags=data['tags'],
            priority=MemoryPriority(data['priority']),
            created_at=datetime.fromisoformat(data['created_at']),
            last_accessed=datetime.fromisoformat(data['last_accessed']) if data.get('last_accessed') else None,
            access_count=data.get('access_count', 0),
            emotional_intensity=data.get('emotional_intensity', 0.5),
            strategic_score=data.get('strategic_score', 0.0),
            relevance_score=data.get('relevance_score', 0.0),
            size_bytes=data.get('size_bytes', 0),
            version=data.get('version', 1),
            updated_at=datetime.fromisoformat(data['updated_at']) if data.get('updated_at') else None,
            lifecycle_stage=MemoryLifecycleStage(data.get('lifecycle_stage', 'active'))
        )


class MemoryDeduplicator:
    """记忆去重管理器"""

    def __init__(self,
                 similarity_threshold: float = 0.85,
                 enable_deduplication: bool = True,
                 default_strategy: DeduplicationStrategy = DeduplicationStrategy.SKIP):
        """
        初始化去重管理器
        
        Args:
            similarity_threshold: 相似度阈值，超过此值视为重复
            enable_deduplication: 是否启用去重
            default_strategy: 默认去重策略
        """
        self.similarity_threshold = similarity_threshold
        self.enable_deduplication = enable_deduplication
        self.default_strategy = default_strategy
        self.content_hashes: Dict[str, str] = {}  # content_hash -> memory_id
        self.lock = threading.RLock()

    def _compute_content_hash(self, content: Union[str, List[Dict[str, Any]]]) -> str:
        """
        计算内容哈希值
        
        Args:
            content: 记忆内容
            
        Returns:
            内容的SHA256哈希值
        """
        if isinstance(content, str):
            content_str = content.strip().lower()
        else:
            content_str = json.dumps(content, ensure_ascii=False, sort_keys=True)
        
        return hashlib.sha256(content_str.encode('utf-8')).hexdigest()

    def _compute_similarity(self, content1: Union[str, List[Dict[str, Any]]],
                           content2: Union[str, List[Dict[str, Any]]]) -> float:
        """
        计算两个内容的相似度
        
        Args:
            content1: 第一个内容
            content2: 第二个内容
            
        Returns:
            相似度分数 (0.0 - 1.0)
        """
        str1 = content1 if isinstance(content1, str) else json.dumps(content1, ensure_ascii=False)
        str2 = content2 if isinstance(content2, str) else json.dumps(content2, ensure_ascii=False)
        
        str1 = str1.strip().lower()
        str2 = str2.strip().lower()
        
        if str1 == str2:
            return 1.0
        
        # 使用简单的编辑距离算法计算相似度
        if len(str1) == 0 or len(str2) == 0:
            return 0.0
        
        # 使用更高效的算法：Jaccard相似度
        words1 = set(str1.split())
        words2 = set(str2.split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = len(words1 & words2)
        union = len(words1 | words2)
        
        return intersection / union if union > 0 else 0.0

    def check_duplicate(self,
                       content: Union[str, List[Dict[str, Any]]],
                       memory_type: MemoryType,
                       existing_memories: List[MemoryRecord]) -> Optional[MemoryRecord]:
        """
        检查是否存在重复记忆
        
        Args:
            content: 要检查的内容
            memory_type: 记忆类型
            existing_memories: 现有的记忆列表
            
        Returns:
            如果找到重复记忆则返回该记忆，否则返回None
        """
        if not self.enable_deduplication:
            return None
        
        content_hash = self._compute_content_hash(content)
        
        with self.lock:
            # 首先检查完全相同的哈希
            if content_hash in self.content_hashes:
                duplicate_id = self.content_hashes[content_hash]
                for memory in existing_memories:
                    if memory.memory_id == duplicate_id and memory.memory_type == memory_type:
                        return memory
            
            # 然后检查相似度
            for memory in existing_memories:
                if memory.memory_type == memory_type:
                    similarity = self._compute_similarity(content, memory.content)
                    if similarity >= self.similarity_threshold:
                        return memory
        
        return None

    def handle_duplicate(self,
                        duplicate_memory: MemoryRecord,
                        new_content: Union[str, List[Dict[str, Any]]],
                        new_metadata: Dict[str, Any],
                        new_tags: List[str],
                        strategy: Optional[DeduplicationStrategy] = None) -> Tuple[bool, Optional[str]]:
        """
        处理重复记忆
        
        Args:
            duplicate_memory: 重复的记忆
            new_content: 新内容
            new_metadata: 新元数据
            new_tags: 新标签
            strategy: 去重策略，如果为None则使用默认策略
            
        Returns:
            (是否跳过添加, 记忆ID)
        """
        strategy = strategy or self.default_strategy
        
        if strategy == DeduplicationStrategy.SKIP:
            return True, duplicate_memory.memory_id
        
        elif strategy == DeduplicationStrategy.OVERWRITE:
            return False, duplicate_memory.memory_id
        
        elif strategy == DeduplicationStrategy.MERGE:
            return False, duplicate_memory.memory_id
        
        elif strategy == DeduplicationStrategy.UPDATE_METADATA:
            return False, duplicate_memory.memory_id
        
        return True, None

    def register_memory(self, memory_id: str, content: Union[str, List[Dict[str, Any]]]):
        """
        注册新记忆到去重系统
        
        Args:
            memory_id: 记忆ID
            content: 记忆内容
        """
        content_hash = self._compute_content_hash(content)
        
        with self.lock:
            self.content_hashes[content_hash] = memory_id

    def unregister_memory(self, memory_id: str, content: Union[str, List[Dict[str, Any]]]):
        """
        从去重系统中注销记忆
        
        Args:
            memory_id: 记忆ID
            content: 记忆内容
        """
        content_hash = self._compute_content_hash(content)
        
        with self.lock:
            if content_hash in self.content_hashes and self.content_hashes[content_hash] == memory_id:
                del self.content_hashes[content_hash]

    def get_stats(self) -> Dict[str, Any]:
        """获取去重统计信息"""
        with self.lock:
            return {
                "enabled": self.enable_deduplication,
                "similarity_threshold": self.similarity_threshold,
                "default_strategy": self.default_strategy.value,
                "registered_hashes": len(self.content_hashes)
            }


class MemoryRetrievalStrategy(Enum):
    """记忆检索策略枚举"""
    SEMANTIC = "semantic"
    HYBRID = "hybrid"
    TAG_BASED = "tag_based"
    RECENT = "recent"
    PRIORITY = "priority"
    ADAPTIVE = "adaptive"


class MemoryRetrievalOptimizer:
    """记忆检索优化器"""
    
    def __init__(self, default_strategy: MemoryRetrievalStrategy = MemoryRetrievalStrategy.HYBRID):
        self.default_strategy = default_strategy
        self.query_history: List[str] = []
        self.max_history = 100
        self.recent_queries: Dict[str, int] = defaultdict(int)
        
    def optimize_query(self, query: str, memory_type: Optional[MemoryType] = None,
                      tags: Optional[List[str]] = None, priority: Optional[MemoryPriority] = None,
                      limit: int = 5) -> Tuple[MemoryRetrievalStrategy, Dict[str, Any]]:
        """
        优化查询策略
        
        Args:
            query: 查询字符串
            memory_type: 记忆类型
            tags: 标签列表
            priority: 优先级
            limit: 返回数量限制
            
        Returns:
            (检索策略, 优化参数)
        """
        strategy = self.default_strategy
        params = {
            "query": query,
            "memory_type": memory_type,
            "tags": tags,
            "priority": priority,
            "limit": limit,
            "similarity_threshold": 0.7,
            "use_cache": True,
            "rerank": True
        }
        
        # 处理空查询
        if not query or query.strip() == "":
            if tags and len(tags) > 0:
                strategy = MemoryRetrievalStrategy.TAG_BASED
                params["use_cache"] = True
                params["rerank"] = False
            elif priority:
                strategy = MemoryRetrievalStrategy.PRIORITY
                params["use_cache"] = True
                params["rerank"] = False
            else:
                strategy = MemoryRetrievalStrategy.RECENT
                params["use_cache"] = True
                params["rerank"] = False
        # 根据查询特征选择策略
        elif tags and len(tags) > 0:
            strategy = MemoryRetrievalStrategy.TAG_BASED
            params["similarity_threshold"] = 0.5
        elif priority:
            strategy = MemoryRetrievalStrategy.PRIORITY
            params["limit"] = min(limit * 2, 20)
        elif query and len(query.split()) <= 2:
            strategy = MemoryRetrievalStrategy.RECENT
            params["rerank"] = False
        elif query and len(query.split()) > 5:
            strategy = MemoryRetrievalStrategy.SEMANTIC
            params["similarity_threshold"] = 0.8
        else:
            strategy = MemoryRetrievalStrategy.HYBRID
        
        # 记录查询历史
        self._record_query(query, tags)
        
        return strategy, params
    
    def _record_query(self, query: str, tags: Optional[List[str]] = None):
        """记录查询历史"""
        query_key = f"{query}_{','.join(tags) if tags else ''}"
        self.recent_queries[query_key] += 1
        
        self.query_history.append(query_key)
        if len(self.query_history) > self.max_history:
            self.query_history.pop(0)
    
    def get_query_stats(self) -> Dict[str, Any]:
        """获取查询统计信息"""
        return {
            "total_queries": len(self.query_history),
            "unique_queries": len(self.recent_queries),
            "most_common_queries": sorted(
                self.recent_queries.items(),
                key=lambda x: x[1],
                reverse=True
            )[:5]
        }


class MemoryPriorityManager:
    """记忆优先级管理器"""

    def __init__(self):
        self.priority_weights = {
            MemoryPriority.CRITICAL: 1.0,
            MemoryPriority.HIGH: 0.8,
            MemoryPriority.MEDIUM: 0.5,
            MemoryPriority.LOW: 0.3,
            MemoryPriority.ARCHIVE: 0.1
        }
        self.priority_thresholds = {
            "auto_promote": {
                MemoryPriority.LOW: {"access_count": 10, "hours": 24},
                MemoryPriority.MEDIUM: {"access_count": 20, "hours": 48},
                MemoryPriority.HIGH: {"access_count": 50, "hours": 72}
            },
            "auto_demote": {
                MemoryPriority.HIGH: {"no_access_hours": 168, "age_hours": 720},
                MemoryPriority.MEDIUM: {"no_access_hours": 336, "age_hours": 1440},
                MemoryPriority.LOW: {"no_access_hours": 720, "age_hours": 2880}
            }
        }
        self.lock = threading.RLock()

    def calculate_priority_score(self, record: MemoryRecord) -> float:
        """
        计算记忆的优先级分数
        
        Args:
            record: 记忆记录
            
        Returns:
            优先级分数 (0.0 - 1.0)
        """
        base_score = self.priority_weights[record.priority]
        
        # 访问频率加成
        access_bonus = min(record.access_count / 100.0, 0.2)
        
        # 访问新近度加成
        recency_bonus = record.access_recency * 0.1
        
        # 情感强度加成
        emotional_bonus = record.emotional_intensity * 0.1
        
        # 战略分数加成
        strategic_bonus = record.strategic_score * 0.1
        
        # 年龄衰减
        age_penalty = min(record.age_hours / 8760.0, 0.3)
        
        total_score = base_score + access_bonus + recency_bonus + emotional_bonus + strategic_bonus - age_penalty
        
        return max(0.0, min(1.0, total_score))

    def should_promote(self, record: MemoryRecord) -> bool:
        """
        判断是否应该提升记忆优先级
        
        Args:
            record: 记忆记录
            
        Returns:
            是否应该提升
        """
        if record.priority == MemoryPriority.CRITICAL or record.priority == MemoryPriority.ARCHIVE:
            return False
        
        thresholds = self.priority_thresholds["auto_promote"].get(record.priority)
        if not thresholds:
            return False
        
        # 检查访问次数
        if record.access_count >= thresholds["access_count"]:
            return True
        
        # 检查时间和访问频率
        if record.age_hours >= thresholds["hours"] and record.access_count >= thresholds["access_count"] / 2:
            return True
        
        return False

    def should_demote(self, record: MemoryRecord) -> bool:
        """
        判断是否应该降低记忆优先级
        
        Args:
            record: 记忆记录
            
        Returns:
            是否应该降低
        """
        if record.priority == MemoryPriority.CRITICAL or record.priority == MemoryPriority.LOW:
            return False
        
        thresholds = self.priority_thresholds["auto_demote"].get(record.priority)
        if not thresholds:
            return False
        
        # 检查未访问时间
        if record.last_accessed:
            no_access_hours = (datetime.now() - record.last_accessed).total_seconds() / 3600
            if no_access_hours >= thresholds["no_access_hours"]:
                return True
        
        # 检查年龄和访问频率
        if record.age_hours >= thresholds["age_hours"] and record.access_count < 5:
            return True
        
        return False

    def get_promoted_priority(self, record: MemoryRecord) -> MemoryPriority:
        """
        获取提升后的优先级
        
        Args:
            record: 记忆记录
            
        Returns:
            提升后的优先级
        """
        if record.priority == MemoryPriority.LOW:
            return MemoryPriority.MEDIUM
        elif record.priority == MemoryPriority.MEDIUM:
            return MemoryPriority.HIGH
        elif record.priority == MemoryPriority.HIGH:
            return MemoryPriority.CRITICAL
        return record.priority

    def get_demoted_priority(self, record: MemoryRecord) -> MemoryPriority:
        """
        获取降低后的优先级
        
        Args:
            record: 记忆记录
            
        Returns:
            降低后的优先级
        """
        if record.priority == MemoryPriority.HIGH:
            return MemoryPriority.MEDIUM
        elif record.priority == MemoryPriority.MEDIUM:
            return MemoryPriority.LOW
        elif record.priority == MemoryPriority.LOW:
            return MemoryPriority.ARCHIVE
        return record.priority

    def adjust_priority(self, record: MemoryRecord) -> Optional[MemoryPriority]:
        """
        自动调整记忆优先级
        
        Args:
            record: 记忆记录
            
        Returns:
            调整后的优先级，如果不需要调整则返回None
        """
        if self.should_promote(record):
            return self.get_promoted_priority(record)
        elif self.should_demote(record):
            return self.get_demoted_priority(record)
        return None

    def get_priority_stats(self, records: List[MemoryRecord]) -> Dict[str, Any]:
        """
        获取优先级统计信息
        
        Args:
            records: 记忆记录列表
            
        Returns:
            统计信息字典
        """
        stats = {
            "total_count": len(records),
            "by_priority": defaultdict(int),
            "average_scores": defaultdict(list),
            "promote_candidates": 0,
            "demote_candidates": 0
        }
        
        for record in records:
            stats["by_priority"][record.priority.value] += 1
            score = self.calculate_priority_score(record)
            stats["average_scores"][record.priority.value].append(score)
            
            if self.should_promote(record):
                stats["promote_candidates"] += 1
            if self.should_demote(record):
                stats["demote_candidates"] += 1
        
        # 计算平均分数
        for priority, scores in stats["average_scores"].items():
            if scores:
                stats["average_scores"][priority] = sum(scores) / len(scores)
        
        return dict(stats)

    def sort_by_priority(self, records: List[MemoryRecord], 
                        include_score: bool = False) -> List[Union[MemoryRecord, Tuple[MemoryRecord, float]]]:
        """
        按优先级分数排序记忆
        
        Args:
            records: 记忆记录列表
            include_score: 是否包含优先级分数
            
        Returns:
            排序后的记录列表
        """
        scored_records = [(record, self.calculate_priority_score(record)) for record in records]
        scored_records.sort(key=lambda x: x[1], reverse=True)
        
        if include_score:
            return scored_records
        return [record for record, _ in scored_records]

    def get_top_priority_memories(self, records: List[MemoryRecord], 
                                  limit: int = 10,
                                  min_score: float = 0.5) -> List[MemoryRecord]:
        """
        获取高优先级记忆
        
        Args:
            records: 记忆记录列表
            limit: 返回数量限制
            min_score: 最小优先级分数
            
        Returns:
            高优先级记忆列表
        """
        sorted_records = self.sort_by_priority(records)
        top_memories = []
        
        for record in sorted_records:
            score = self.calculate_priority_score(record)
            if score >= min_score:
                top_memories.append(record)
                if len(top_memories) >= limit:
                    break
        
        return top_memories

    def update_priority_thresholds(self, promote: Optional[Dict[Union[str, MemoryPriority], Dict[str, int]]] = None,
                                   demote: Optional[Dict[Union[str, MemoryPriority], Dict[str, int]]] = None):
        """
        更新优先级阈值配置
        
        Args:
            promote: 提升阈值配置
            demote: 降低阈值配置
        """
        with self.lock:
            if promote:
                for priority, thresholds in promote.items():
                    # 转换字符串为枚举
                    if isinstance(priority, str):
                        priority_enum = MemoryPriority(priority)
                    else:
                        priority_enum = priority
                    
                    if priority_enum in self.priority_thresholds["auto_promote"]:
                        self.priority_thresholds["auto_promote"][priority_enum].update(thresholds)
            
            if demote:
                for priority, thresholds in demote.items():
                    # 转换字符串为枚举
                    if isinstance(priority, str):
                        priority_enum = MemoryPriority(priority)
                    else:
                        priority_enum = priority
                    
                    if priority_enum in self.priority_thresholds["auto_demote"]:
                        self.priority_thresholds["auto_demote"][priority_enum].update(thresholds)

    def get_priority_thresholds(self) -> Dict[str, Any]:
        """
        获取当前优先级阈值配置
        
        Returns:
            阈值配置字典
        """
        return {
            "auto_promote": dict(self.priority_thresholds["auto_promote"]),
            "auto_demote": dict(self.priority_thresholds["auto_demote"])
        }


class MemoryCompressionManager:
    """记忆压缩和归档管理器"""

    def __init__(self):
        self.compression_thresholds = {
            "size_bytes": 10000,  # 超过10KB的内容需要压缩
            "age_hours": 720,  # 超过30天的记忆可以压缩
            "access_count": 5,  # 访问次数少于5次的记忆可以压缩
            "priority": [MemoryPriority.LOW, MemoryPriority.ARCHIVE]  # 低优先级和归档优先级的记忆
        }
        self.archive_thresholds = {
            "age_hours": 2160,  # 超过90天的记忆可以归档
            "priority": [MemoryPriority.ARCHIVE],  # 归档优先级的记忆
            "no_access_hours": 1440  # 超过60天未访问的记忆可以归档
        }
        self.compression_stats = {
            "total_compressed": 0,
            "total_archived": 0,
            "total_size_saved": 0,
            "compression_ratio": 0.0
        }
        self.lock = threading.RLock()

    def should_compress(self, record: MemoryRecord) -> bool:
        """
        判断是否应该压缩记忆
        
        Args:
            record: 记忆记录
            
        Returns:
            是否应该压缩
        """
        # 检查优先级
        if record.priority not in self.compression_thresholds["priority"]:
            return False
        
        # 检查内容大小
        if record.size_bytes > self.compression_thresholds["size_bytes"]:
            return True
        
        # 检查年龄和访问次数
        if record.age_hours >= self.compression_thresholds["age_hours"] and record.access_count < self.compression_thresholds["access_count"]:
            return True
        
        return False

    def should_archive(self, record: MemoryRecord) -> bool:
        """
        判断是否应该归档记忆
        
        Args:
            record: 记忆记录
            
        Returns:
            是否应该归档
        """
        # 检查优先级
        if record.priority in self.archive_thresholds["priority"]:
            return True
        
        # 检查年龄
        if record.age_hours >= self.archive_thresholds["age_hours"]:
            return True
        
        # 检查未访问时间
        if record.last_accessed:
            no_access_hours = (datetime.now() - record.last_accessed).total_seconds() / 3600
            if no_access_hours >= self.archive_thresholds["no_access_hours"]:
                return True
        
        return False

    def compress_content(self, content: Union[str, List[Dict[str, Any]]]) -> Tuple[Union[str, bytes], int, int]:
        """
        压缩记忆内容
        
        Args:
            content: 记忆内容
            
        Returns:
            (压缩后的内容, 原始大小, 压缩后大小)
        """
        import zlib
        import json
        
        # 转换为字符串
        if isinstance(content, list):
            content_str = json.dumps(content, ensure_ascii=False)
        else:
            content_str = content
        
        original_size = len(content_str.encode('utf-8'))
        
        # 如果内容较小，不压缩
        if original_size < 1024:
            return content, original_size, original_size
        
        # 使用zlib压缩
        compressed_bytes = zlib.compress(content_str.encode('utf-8'), level=6)
        compressed_size = len(compressed_bytes)
        
        # 如果压缩后反而变大，返回原始内容
        if compressed_size >= original_size:
            return content, original_size, original_size
        
        # 返回压缩后的字节数据
        return compressed_bytes, original_size, compressed_size

    def decompress_content(self, compressed_content: Union[str, bytes]) -> Union[str, List[Dict[str, Any]]]:
        """
        解压缩记忆内容
        
        Args:
            compressed_content: 压缩后的内容
            
        Returns:
            解压后的内容
        """
        import zlib
        import json
        
        # 如果是字符串，直接返回
        if isinstance(compressed_content, str):
            # 尝试解析为JSON
            try:
                return json.loads(compressed_content)
            except json.JSONDecodeError:
                return compressed_content
        
        # 如果是字节数据，解压
        if isinstance(compressed_content, bytes):
            decompressed_str = zlib.decompress(compressed_content).decode('utf-8')
            try:
                return json.loads(decompressed_str)
            except json.JSONDecodeError:
                return decompressed_str
        
        return compressed_content

    def compress_memory(self, record: MemoryRecord) -> MemoryRecord:
        """
        压缩记忆记录
        
        Args:
            record: 记忆记录
            
        Returns:
            压缩后的记录
        """
        original_size = record.size_bytes
        
        # 压缩内容
        compressed_content, original_size, compressed_size = self.compress_content(record.content)
        
        # 更新记录
        if compressed_content != record.content:
            record.content = compressed_content
            record.size_bytes = compressed_size
            record.metadata["compressed"] = True
            record.metadata["original_size"] = original_size
            record.updated_at = datetime.now()
            
            # 更新统计
            with self.lock:
                self.compression_stats["total_compressed"] += 1
                self.compression_stats["total_size_saved"] += (original_size - compressed_size)
                total_compressed_size = self.compression_stats["total_compressed"] * compressed_size
                total_original_size = self.compression_stats["total_compressed"] * original_size
                self.compression_stats["compression_ratio"] = (1 - total_compressed_size / total_original_size) if total_original_size > 0 else 0.0
        
        return record

    def archive_memory(self, record: MemoryRecord) -> MemoryRecord:
        """
        归档记忆记录
        
        Args:
            record: 记忆记录
            
        Returns:
            归档后的记录
        """
        # 先压缩
        if not record.metadata.get("compressed", False):
            record = self.compress_memory(record)
        
        # 设置归档优先级
        record.priority = MemoryPriority.ARCHIVE
        record.metadata["archived"] = True
        record.metadata["archived_at"] = datetime.now().isoformat()
        record.updated_at = datetime.now()
        
        # 更新统计
        with self.lock:
            self.compression_stats["total_archived"] += 1
        
        return record

    def get_compression_stats(self) -> Dict[str, Any]:
        """
        获取压缩统计信息
        
        Returns:
            统计信息字典
        """
        with self.lock:
            return self.compression_stats.copy()

    def update_compression_thresholds(self, **kwargs):
        """
        更新压缩阈值配置
        
        Args:
            kwargs: 阈值配置参数
        """
        with self.lock:
            for key, value in kwargs.items():
                if key in self.compression_thresholds:
                    self.compression_thresholds[key] = value

    def update_archive_thresholds(self, **kwargs):
        """
        更新归档阈值配置
        
        Args:
            kwargs: 阈值配置参数
        """
        with self.lock:
            for key, value in kwargs.items():
                if key in self.archive_thresholds:
                    self.archive_thresholds[key] = value

    def get_thresholds(self) -> Dict[str, Any]:
        """
        获取当前阈值配置
        
        Returns:
            阈值配置字典
        """
        return {
            "compression": dict(self.compression_thresholds),
            "archive": dict(self.archive_thresholds)
        }


class MemoryLifecycleManager:
    """记忆生命周期管理器"""

    def __init__(self):
        self.lifecycle_thresholds = {
            "active": {
                "max_age_hours": 168,  # 7天
                "min_access_count": 5
            },
            "mature": {
                "max_age_hours": 720,  # 30天
                "min_access_count": 10
            },
            "stable": {
                "max_age_hours": 2160,  # 90天
                "min_access_count": 20
            },
            "declining": {
                "max_age_hours": 4320,  # 180天
                "min_access_count": 30
            },
            "archived": {
                "max_age_hours": 8760,  # 365天
                "min_access_count": 50
            }
        }
        self.lock = threading.RLock()

    def determine_lifecycle_stage(self, record: MemoryRecord) -> MemoryLifecycleStage:
        """
        确定记忆的生命周期阶段
        
        Args:
            record: 记忆记录
            
        Returns:
            生命周期阶段
        """
        age_hours = record.age_hours
        access_count = record.access_count
        
        # 检查是否已归档
        if record.metadata.get("archived", False):
            if age_hours > self.lifecycle_thresholds["archived"]["max_age_hours"]:
                return MemoryLifecycleStage.EXPIRED
            return MemoryLifecycleStage.ARCHIVED
        
        # 根据年龄和访问次数确定阶段
        if age_hours <= self.lifecycle_thresholds["active"]["max_age_hours"]:
            if access_count >= self.lifecycle_thresholds["active"]["min_access_count"]:
                return MemoryLifecycleStage.ACTIVE
            return MemoryLifecycleStage.DECLINING
        
        elif age_hours <= self.lifecycle_thresholds["mature"]["max_age_hours"]:
            if access_count >= self.lifecycle_thresholds["mature"]["min_access_count"]:
                return MemoryLifecycleStage.MATURE
            return MemoryLifecycleStage.DECLINING
        
        elif age_hours <= self.lifecycle_thresholds["stable"]["max_age_hours"]:
            if access_count >= self.lifecycle_thresholds["stable"]["min_access_count"]:
                return MemoryLifecycleStage.STABLE
            return MemoryLifecycleStage.DECLINING
        
        elif age_hours <= self.lifecycle_thresholds["declining"]["max_age_hours"]:
            if access_count >= self.lifecycle_thresholds["declining"]["min_access_count"]:
                return MemoryLifecycleStage.STABLE
            return MemoryLifecycleStage.DECLINING
        
        else:
            return MemoryLifecycleStage.DECLINING

    def update_lifecycle_stage(self, record: MemoryRecord) -> Optional[MemoryLifecycleStage]:
        """
        更新记忆的生命周期阶段
        
        Args:
            record: 记录记录
            
        Returns:
            新的生命周期阶段，如果没有变化则返回None
        """
        new_stage = self.determine_lifecycle_stage(record)
        
        if new_stage != record.lifecycle_stage:
            old_stage = record.lifecycle_stage
            record.lifecycle_stage = new_stage
            record.updated_at = datetime.now()
            return new_stage
        
        return None

    def should_promote(self, record: MemoryRecord) -> bool:
        """
        检查记忆是否应该提升到下一阶段
        
        Args:
            record: 记录记录
            
        Returns:
            是否应该提升
        """
        current_stage = record.lifecycle_stage
        
        if current_stage == MemoryLifecycleStage.ACTIVE:
            return False  # 已经是最高阶段
        
        if current_stage == MemoryLifecycleStage.DECLINING:
            # 检查是否应该提升到活跃期
            if record.age_hours <= self.lifecycle_thresholds["active"]["max_age_hours"]:
                return record.access_count >= self.lifecycle_thresholds["active"]["min_access_count"]
            return False
        
        if current_stage == MemoryLifecycleStage.MATURE:
            # 检查是否应该提升到活跃期
            return record.access_count >= self.lifecycle_thresholds["active"]["min_access_count"] * 2
        
        if current_stage == MemoryLifecycleStage.STABLE:
            # 检查是否应该提升到成熟期
            return record.access_count >= self.lifecycle_thresholds["mature"]["min_access_count"] * 2
        
        return False

    def should_demote(self, record: MemoryRecord) -> bool:
        """
        检查记忆是否应该降级到下一阶段
        
        Args:
            record: 记录记录
            
        Returns:
            是否应该降级
        """
        current_stage = record.lifecycle_stage
        
        if current_stage == MemoryLifecycleStage.EXPIRED:
            return False  # 已经是最低阶段
        
        if current_stage == MemoryLifecycleStage.ARCHIVED:
            # 检查是否应该降级到过期期
            return record.age_hours > self.lifecycle_thresholds["archived"]["max_age_hours"]
        
        if current_stage == MemoryLifecycleStage.DECLINING:
            # 检查是否应该归档
            return record.age_hours > self.lifecycle_thresholds["declining"]["max_age_hours"]
        
        if current_stage == MemoryLifecycleStage.STABLE:
            # 检查是否应该降级到衰退期
            return record.age_hours > self.lifecycle_thresholds["stable"]["max_age_hours"]
        
        if current_stage == MemoryLifecycleStage.MATURE:
            # 检查是否应该降级到稳定期
            return record.age_hours > self.lifecycle_thresholds["mature"]["max_age_hours"]
        
        if current_stage == MemoryLifecycleStage.ACTIVE:
            # 检查是否应该降级到成熟期
            return record.age_hours > self.lifecycle_thresholds["active"]["max_age_hours"]
        
        return False

    def get_lifecycle_stats(self, records: Dict[str, MemoryRecord]) -> Dict[str, Any]:
        """
        获取生命周期统计信息
        
        Args:
            records: 所有记录
            
        Returns:
            统计信息字典
        """
        stage_counts = defaultdict(int)
        stage_sizes = defaultdict(int)
        
        for record in records.values():
            stage = record.lifecycle_stage
            stage_counts[stage.value] += 1
            stage_sizes[stage.value] += record.size_bytes
        
        total_count = sum(stage_counts.values())
        total_size = sum(stage_sizes.values())
        
        return {
            "stage_counts": dict(stage_counts),
            "stage_sizes": dict(stage_sizes),
            "total_count": total_count,
            "total_size": total_size,
            "stage_percentages": {
                stage: (count / total_count * 100) if total_count > 0 else 0.0
                for stage, count in stage_counts.items()
            }
        }

    def update_lifecycle_thresholds(self, stage: str, **kwargs):
        """
        更新生命周期阈值配置
        
        Args:
            stage: 生命周期阶段
            kwargs: 阈值配置参数
        """
        with self.lock:
            if stage in self.lifecycle_thresholds:
                for key, value in kwargs.items():
                    if key in self.lifecycle_thresholds[stage]:
                        self.lifecycle_thresholds[stage][key] = value

    def get_lifecycle_thresholds(self) -> Dict[str, Any]:
        """
        获取当前生命周期阈值配置
        
        Returns:
            阈值配置字典
        """
        return dict(self.lifecycle_thresholds)

    def get_thresholds(self) -> Dict[str, Any]:
        """
        获取当前生命周期阈值配置（别名方法）
        
        Returns:
            阈值配置字典
        """
        return self.get_lifecycle_thresholds()

    def set_thresholds(self, thresholds: Dict[str, Any]) -> bool:
        """
        设置生命周期阈值配置
        
        Args:
            thresholds: 阈值配置字典
            
        Returns:
            是否设置成功
        """
        with self.lock:
            try:
                for stage, config in thresholds.items():
                    if stage in self.lifecycle_thresholds:
                        for key, value in config.items():
                            if key in self.lifecycle_thresholds[stage]:
                                self.lifecycle_thresholds[stage][key] = value
                return True
            except Exception:
                return False


class MemoryManager:
    """集中管理所有记忆操作（优化版）"""

    def __init__(self, config: Optional[MemoryConfig] = None) -> None:
        self.config: MemoryConfig = config or ConfigManager().get().memory
        self.session_id: str = ConfigManager().get().session_id
        self.user_id: str = ConfigManager().get().user_id

        # 初始化记忆系统
        self.memory_system: ZyantineMemorySystem = self._initialize_memory_system()

        # 存储优化器（在缓存之前初始化）
        self.storage_optimizer: MemoryStorageOptimizer = MemoryStorageOptimizer(
            hot_size=500,
            warm_size=2000,
            cold_size=10000
        )

        # 缓存系统
        self.cache: MemoryCache = MemoryCache(
            max_size=self.config.cache_size if hasattr(self.config, 'cache_size') else 1000,
            ttl_hours=self.config.cache_ttl_hours if hasattr(self.config, 'cache_ttl_hours') else 24,
            storage_optimizer=self.storage_optimizer
        )

        # 语义索引
        self.semantic_index: Dict[str, List[str]] = defaultdict(list)
        self.reverse_index: Dict[str, List[str]] = defaultdict(list)  # memory_id -> tags

        # 时间索引
        self.temporal_index: Dict[str, List[str]] = defaultdict(list)  # date_str -> memory_ids

        # 统计信息
        self.stats: Dict[str, Any] = {
            "total_memories": 0,
            "memory_by_type": defaultdict(int),
            "memory_by_priority": defaultdict(int),
            "average_emotional_intensity": 0.0,
            "average_strategic_score": 0.0,
            "cache_hit_rate": 0.0,
            "average_response_time_ms": 0.0,
            "last_cleanup_time": None
        }

        # 性能统计
        self._access_count: int = 0
        self._cache_hits: int = 0
        self._total_response_time: float = 0.0

        # 性能监控器
        self.performance_monitor: MemoryPerformanceMonitor = MemoryPerformanceMonitor(max_metrics=10000)

        # 去重管理器
        self.deduplicator: MemoryDeduplicator = MemoryDeduplicator(
            similarity_threshold=0.85,
            enable_deduplication=True,
            default_strategy=DeduplicationStrategy.SKIP
        )

        # 检索优化器
        self.retrieval_optimizer: MemoryRetrievalOptimizer = MemoryRetrievalOptimizer(
            default_strategy=MemoryRetrievalStrategy.HYBRID
        )

        # 优先级管理器
        self.priority_manager: MemoryPriorityManager = MemoryPriorityManager()

        # 压缩管理器
        self.compression_manager: MemoryCompressionManager = MemoryCompressionManager()

        # 生命周期管理器
        self.lifecycle_manager: MemoryLifecycleManager = MemoryLifecycleManager()

        # 安全管理器
        self.security_manager: MemorySecurityManager = MemorySecurityManager(
            enable_encryption=True,
            enable_access_control=True,
            enable_audit_log=True,
            default_security_level=MemorySecurityLevel.INTERNAL,
            default_privacy_policy=MemoryPrivacyPolicy.ALLOW_OWNER
        )

        # 锁
        self._stats_lock: threading.Lock = threading.Lock()
        self._index_lock: threading.RLock = threading.RLock()

        # 后台清理线程
        self._cleanup_thread: threading.Thread = threading.Thread(target=self._background_cleanup, daemon=True)
        self._cleanup_thread.start()

        print(f"[记忆管理器] 初始化完成，会话: {self.session_id}")

    def _initialize_memory_system(self) -> ZyantineMemorySystem:
        """初始化记忆系统"""
        config = ConfigManager().get()

        return ZyantineMemorySystem(
            base_url=config.api.base_url,
            api_key=config.api.api_key,
            user_id=self.user_id,
            session_id=self.session_id
        )

    def add_memory(self,
                   content: Union[str, List[Dict[str, Any]]],
                   memory_type: Union[str, MemoryType],
                   tags: Optional[List[str]] = None,
                   emotional_intensity: float = 0.5,
                   strategic_value: Optional[Dict[str, Any]] = None,
                   metadata: Optional[Dict[str, Any]] = None,
                   priority: Union[str, MemoryPriority] = MemoryPriority.MEDIUM,
                   ttl_hours: Optional[int] = None) -> str:
        """添加记忆（带TTL支持）"""
        start_time: float = time.time()
        memory_id: Optional[str] = None
        success: bool = False
        error_message: Optional[str] = None

        try:
            # 处理枚举类型
            if isinstance(memory_type, str):
                try:
                    memory_type_enum: MemoryType = MemoryType(memory_type)
                except ValueError as e:
                    error_message = f"无效的记忆类型: {memory_type}"
                    raise MemoryTypeError(error_message) from e
            else:
                memory_type_enum = memory_type

            if isinstance(priority, str):
                try:
                    priority_enum: MemoryPriority = MemoryPriority(priority)
                except ValueError:
                    priority_enum = MemoryPriority.MEDIUM
            else:
                priority_enum = priority

            # 准备元数据
            full_metadata: Dict[str, Any] = metadata or {}
            full_metadata.update({
                "priority": priority_enum.value,
                "added_by": "memory_manager",
                "estimated_size": len(str(content))
            })

            # 添加TTL信息
            if ttl_hours:
                expiry_time: datetime = datetime.now() + timedelta(hours=ttl_hours)
                full_metadata["expires_at"] = expiry_time.isoformat()

            # 敏感数据检测和脱敏
            content_str = str(content) if not isinstance(content, str) else content
            masked_content, detected_keywords = self.security_manager.scan_and_mask_sensitive_data(
                memory_id="pending",
                content=content_str
            )
            
            # 如果检测到敏感词，更新内容为脱敏后的内容
            if detected_keywords:
                if isinstance(content, str):
                    content = masked_content
                else:
                    full_metadata["original_content"] = content
                    full_metadata["sensitive_keywords"] = detected_keywords
                    full_metadata["content_masked"] = True

            # 检查重复记忆 - 优先使用缓存中的记忆
            # 从缓存中获取当前会话的记忆
            cached_memories = [
                record for record in self.cache.cache.values()
                if record.memory_type == memory_type_enum
            ]
            
            # 如果缓存中的记忆不足，再从搜索API获取
            if len(cached_memories) < 10:
                search_memories = self.get_all_memories(memory_type=memory_type_enum)
                # 合并缓存和搜索结果，去重
                all_memories = cached_memories.copy()
                existing_ids = {m.memory_id for m in cached_memories}
                for memory in search_memories:
                    if memory.memory_id not in existing_ids:
                        all_memories.append(memory)
                        existing_ids.add(memory.memory_id)
            else:
                all_memories = cached_memories
            
            duplicate_memory = self.deduplicator.check_duplicate(
                content=content,
                memory_type=memory_type_enum,
                existing_memories=all_memories
            )

            if duplicate_memory:
                skip_add, existing_id = self.deduplicator.handle_duplicate(
                    duplicate_memory=duplicate_memory,
                    new_content=content,
                    new_metadata=full_metadata,
                    new_tags=tags or [],
                    strategy=DeduplicationStrategy.SKIP
                )

                if skip_add:
                    print(f"[记忆管理器] 检测到重复记忆，跳过添加: {existing_id}")
                    response_time: float = (time.time() - start_time) * 1000
                    self._record_response_time(response_time)
                    success = True
                    return existing_id

            # 添加到记忆系统
            memory_id: str = self.memory_system.add_memory(
                content=content,
                memory_type=memory_type_enum.value,
                tags=tags or [],
                emotional_intensity=emotional_intensity,
                strategic_value=strategic_value or {},
                metadata=full_metadata
            )

            # 创建记忆记录
            memory_record: MemoryRecord = MemoryRecord(
                memory_id=memory_id,
                content=content,
                memory_type=memory_type_enum,
                metadata=full_metadata,
                tags=tags or [],
                priority=priority_enum,
                created_at=datetime.now(),
                emotional_intensity=emotional_intensity,
                strategic_score=strategic_value.get("score", 0.0) if strategic_value else 0.0,
                size_bytes=len(pickle.dumps(content))
            )

            # 更新缓存和索引
            self._update_cache(memory_record)
            self._update_indexes(memory_record)
            self._update_stats(memory_record)

            # 注册到去重系统
            self.deduplicator.register_memory(memory_id, content)

            # 设置安全级别（根据检测到的敏感词）
            if detected_keywords:
                self.security_manager.set_memory_security_level(
                    memory_id,
                    MemorySecurityLevel.CONFIDENTIAL
                )
            else:
                self.security_manager.set_memory_security_level(
                    memory_id,
                    MemorySecurityLevel.INTERNAL
                )

            # 如果添加的是对话记忆，清除对话历史缓存以确保获取最新数据
            if memory_type_enum == MemoryType.CONVERSATION:
                self._clear_conversation_history_cache()

            response_time: float = (time.time() - start_time) * 1000
            self._record_response_time(response_time)

            success = True
            return memory_id

        except MemoryException:
            raise
        except ValueError as e:
            error_message = str(e)
            raise MemoryTypeError(f"记忆类型或优先级参数错误: {str(e)}") from e
        except Exception as e:
            error_message = str(e)
            raise MemoryStorageError(
                f"添加记忆失败: {str(e)}",
                details={"content_length": len(str(content)), "memory_type": memory_type}
            ) from e
        finally:
            duration_ms = (time.time() - start_time) * 1000
            self.performance_monitor.record_operation(
                operation="add_memory",
                duration_ms=duration_ms,
                success=success,
                memory_id=memory_id,
                memory_type=str(memory_type) if isinstance(memory_type, str) else memory_type.value if isinstance(memory_type, MemoryType) else None,
                cache_hit=False,
                error_message=error_message
            )

    def _update_cache(self, memory_record: MemoryRecord):
        """更新缓存（带版本控制）"""
        # 获取当前版本号，如果记忆已存在则版本+1，否则为1
        current_version = self.cache.get_version(memory_record.memory_id)
        new_version = current_version + 1 if current_version > 0 else 1
        
        # 更新记忆记录的版本号
        memory_record.version = new_version
        memory_record.updated_at = datetime.now()
        
        # 更新缓存
        self.cache.set(memory_record.memory_id, memory_record, version=new_version)
        
        # 添加到存储优化器
        self.storage_optimizer.set(memory_record.memory_id, memory_record)

    def _update_indexes(self, memory_record: MemoryRecord):
        """更新所有索引"""
        with self._index_lock:
            # 语义索引（标签）
            for tag in memory_record.tags:
                if memory_record.memory_id not in self.semantic_index[tag]:
                    self.semantic_index[tag].append(memory_record.memory_id)

            # 反向索引
            self.reverse_index[memory_record.memory_id] = memory_record.tags.copy()

            # 时间索引（按天）
            date_str = memory_record.created_at.strftime("%Y-%m-%d")
            if memory_record.memory_id not in self.temporal_index[date_str]:
                self.temporal_index[date_str].append(memory_record.memory_id)

    def _update_stats(self, memory_record: MemoryRecord):
        """更新统计信息"""
        with self._stats_lock:
            self.stats["total_memories"] += 1

            # 按类型统计
            mem_type = memory_record.memory_type.value
            self.stats["memory_by_type"][mem_type] += 1

            # 按优先级统计
            priority = memory_record.priority.value
            self.stats["memory_by_priority"][priority] += 1

            # 更新平均值（使用增量计算，避免精度问题）
            total_memories = self.stats["total_memories"]
            current_avg_ei = self.stats["average_emotional_intensity"]
            current_avg_ss = self.stats["average_strategic_score"]

            self.stats["average_emotional_intensity"] = \
                current_avg_ei + (memory_record.emotional_intensity - current_avg_ei) / total_memories

            self.stats["average_strategic_score"] = \
                current_avg_ss + (memory_record.strategic_score - current_avg_ss) / total_memories

    def search_memories(self,
                        query: str,
                        memory_type: Optional[Union[str, MemoryType]] = None,
                        tags: Optional[List[str]] = None,
                        priority: Optional[Union[str, MemoryPriority]] = None,
                        limit: int = 5,
                        use_cache: bool = True,
                        similarity_threshold: float = 0.7) -> List[MemoryRecord]:
        """搜索记忆（优化版 - 使用检索优化器）"""
        start_time: float = time.time()
        success: bool = False
        error_message: Optional[str] = None
        cache_hit: bool = False

        try:
            # 使用检索优化器优化查询
            strategy, optimized_params = self.retrieval_optimizer.optimize_query(
                query=query,
                memory_type=memory_type if isinstance(memory_type, MemoryType) else MemoryType(memory_type) if memory_type else None,
                tags=tags,
                priority=priority if isinstance(priority, MemoryPriority) else MemoryPriority(priority) if priority else None,
                limit=limit
            )

            # 使用优化后的参数
            similarity_threshold = optimized_params.get("similarity_threshold", similarity_threshold)
            use_cache = optimized_params.get("use_cache", use_cache)
            rerank = optimized_params.get("rerank", True)
            search_limit = optimized_params.get("limit", limit * 2)

            # 如果查询为空，直接使用缓存检索
            if not query or query.strip() == "":
                records: List[MemoryRecord] = []
                
                # 从缓存中获取所有记忆
                all_memories = list(self.cache.cache.values())
                
                # 过滤记忆
                for record in all_memories:
                    # 类型过滤
                    if memory_type and record.memory_type != memory_type:
                        continue
                    
                    # 优先级过滤
                    if priority and record.priority != priority:
                        continue
                    
                    # 标签过滤
                    if tags and not any(tag in record.tags for tag in tags):
                        continue
                    
                    records.append(record)
                
                # 根据策略排序
                if strategy == MemoryRetrievalStrategy.PRIORITY:
                    records.sort(key=lambda x: (x.priority.value, x.created_at), reverse=True)
                elif strategy == MemoryRetrievalStrategy.RECENT:
                    records.sort(key=lambda x: x.created_at, reverse=True)
                else:
                    records.sort(key=lambda x: x.created_at, reverse=True)
                
                self._cache_hits += 1
                cache_hit = True
                success = True
                return records[:limit]

            # 首先尝试从缓存中查找
            if use_cache and tags:
                cached_results: List[MemoryRecord] = self._search_by_tags(tags, memory_type, priority, limit)
                if cached_results:
                    self._cache_hits += 1
                    cache_hit = True
                    success = True
                    return cached_results

            # 使用记忆系统搜索
            search_results: List[Dict[str, Any]] = self.memory_system.search_memories(
                query=query,
                memory_type=memory_type.value if isinstance(memory_type, MemoryType) else memory_type,
                tags=tags,
                limit=search_limit,
                rerank=rerank,
                similarity_threshold=similarity_threshold
            )

            # 转换为MemoryRecord并过滤
            records: List[MemoryRecord] = []
            for result in search_results:
                record: Optional[MemoryRecord] = self._create_or_update_record_from_search(result)
                if not record:
                    continue

                # 优先级过滤
                if priority and record.priority != priority:
                    continue

                # 类型过滤
                if memory_type and record.memory_type != memory_type:
                    continue

                records.append(record)

            # 根据策略排序
            if strategy == MemoryRetrievalStrategy.PRIORITY:
                records.sort(key=lambda x: (x.priority.value, x.created_at), reverse=True)
            elif strategy == MemoryRetrievalStrategy.RECENT:
                records.sort(key=lambda x: x.created_at, reverse=True)
            else:
                records.sort(key=lambda x: x.relevance_score, reverse=True)

            # 解压压缩的记忆
            for record in records:
                if record.metadata.get("compressed", False):
                    decompressed_content = self.compression_manager.decompress_content(record.content)
                    record.content = decompressed_content

            response_time: float = (time.time() - start_time) * 1000
            self._record_response_time(response_time)
            self._access_count += 1

            success = True
            return records[:limit]

        except MemoryException:
            raise
        except Exception as e:
            error_message = str(e)
            raise MemorySearchError(
                f"搜索记忆失败: {str(e)}",
                details={"query": query, "memory_type": memory_type, "tags": tags, "limit": limit}
            ) from e
        finally:
            duration_ms = (time.time() - start_time) * 1000
            self.performance_monitor.record_operation(
                operation="search_memories",
                duration_ms=duration_ms,
                success=success,
                memory_id=None,
                memory_type=str(memory_type) if isinstance(memory_type, str) else memory_type.value if isinstance(memory_type, MemoryType) else None,
                cache_hit=cache_hit,
                error_message=error_message
            )

    def _search_by_tags(self,
                        tags: List[str],
                        memory_type: Optional[Union[str, MemoryType]],
                        priority: Optional[Union[str, MemoryPriority]],
                        limit: int) -> List[MemoryRecord]:
        """通过标签搜索记忆（优化版）"""
        with self._index_lock:
            memory_ids: set = set()

            # 计算每个记忆的标签匹配度
            memory_scores: Dict[str, int] = defaultdict(int)
            for tag in tags:
                if tag in self.semantic_index:
                    for mem_id in self.semantic_index[tag]:
                        memory_scores[mem_id] += 1

            # 获取记录并过滤
            records: List[MemoryRecord] = []
            for mem_id, score in sorted(memory_scores.items(), key=lambda x: x[1], reverse=True):
                record: Optional[MemoryRecord] = self._get_memory_record(mem_id)
                if not record:
                    continue

                # 类型过滤
                if memory_type and record.memory_type != memory_type:
                    continue

                # 优先级过滤
                if priority and record.priority != priority:
                    continue

                record.relevance_score = score / len(tags)
                records.append(record)

                if len(records) >= limit * 2:
                    break

            # 按综合分数排序
            records.sort(key=lambda x: (
                x.relevance_score,
                x.strategic_score,
                x.access_recency
            ), reverse=True)

            return records[:limit]

    def _get_memory_record(self, memory_id: str) -> Optional[MemoryRecord]:
        """获取记忆记录，优先从缓存中获取"""
        # 先从缓存获取
        record: Optional[MemoryRecord] = self.cache.get(memory_id)
        if record:
            return record

        # 从记忆系统获取
        try:
            memory_data: Optional[Dict[str, Any]] = self.memory_system.get_memory(memory_id, include_full_data=False)
            if memory_data:
                record = self._create_record_from_memory_data(memory_data)
                if record:
                    self.cache.set(memory_id, record)
                return record
        except Exception as e:
            print(f"[记忆管理器] 获取记忆失败 {memory_id}: {e}")

        return None

    def _create_or_update_record_from_search(self, result: Dict[str, Any]) -> Optional[MemoryRecord]:
        """从搜索结果创建或更新记忆记录"""
        try:
            metadata: Dict[str, Any] = result.get("metadata", {})
            memory_id: str = metadata.get("memory_id") or \
                        hashlib.md5(json.dumps(result, sort_keys=True).encode()).hexdigest()[:16]

            # 检查缓存
            existing_record: Optional[MemoryRecord] = self.cache.get(memory_id)
            if existing_record:
                # 更新访问信息
                existing_record.last_accessed = datetime.now()
                existing_record.access_count += 1
                existing_record.relevance_score = result.get("similarity_score", 0.0)
                return existing_record

            # 创建新记录
            return self._create_record_from_memory_data(result)

        except Exception as e:
            print(f"[记忆管理器] 创建记忆记录失败: {e}")
            return None

    def _create_record_from_memory_data(self, memory_data: Dict[str, Any]) -> Optional[MemoryRecord]:
        """从记忆数据创建记忆记录"""
        try:
            metadata: Dict[str, Any] = memory_data.get("metadata", {})

            # 获取或生成记忆ID
            memory_id: str = metadata.get("memory_id") or \
                        hashlib.md5(json.dumps(memory_data, sort_keys=True).encode()).hexdigest()[:16]

            # 确定记忆类型
            memory_type_str: str = metadata.get("memory_type", "conversation")
            try:
                memory_type: MemoryType = MemoryType(memory_type_str)
            except ValueError:
                memory_type = MemoryType.CONVERSATION

            # 确定优先级
            priority_str: str = metadata.get("priority", "medium")
            try:
                priority: MemoryPriority = MemoryPriority(priority_str)
            except ValueError:
                priority = MemoryPriority.MEDIUM

            # 检查是否过期
            if "expires_at" in metadata:
                expires_at: datetime = datetime.fromisoformat(metadata["expires_at"])
                if datetime.now() > expires_at:
                    print(f"[记忆管理器] 记忆已过期: {memory_id}")
                    return None

            # 创建记录
            record: MemoryRecord = MemoryRecord(
                memory_id=memory_id,
                content=memory_data.get("content", ""),
                memory_type=memory_type,
                metadata=metadata,
                tags=metadata.get("tags", []),
                priority=priority,
                created_at=datetime.fromisoformat(
                    metadata.get("created_at", datetime.now().isoformat())
                ),
                last_accessed=datetime.now(),
                access_count=1,
                emotional_intensity=metadata.get("emotional_intensity", 0.5),
                strategic_score=metadata.get("strategic_score", 0.0),
                relevance_score=memory_data.get("similarity_score", 0.0),
                size_bytes=len(str(memory_data.get("content", "")).encode('utf-8'))
            )

            return record

        except Exception as e:
            print(f"[记忆管理器] 从数据创建记录失败: {e}")
            return None

    def find_resonant_memory(self, context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """寻找共鸣记忆"""
        start_time: float = time.time()

        try:
            result: Optional[Dict[str, Any]] = self.memory_system.find_resonant_memory(context)

            response_time: float = (time.time() - start_time) * 1000
            self._record_response_time(response_time)

            return result
        except Exception as e:
            print(f"[记忆管理器] 寻找共鸣记忆失败: {e}")
            return None

    def get_all_memories(self, memory_type: Optional[MemoryType] = None) -> List[MemoryRecord]:
        """
        获取所有记忆记录
        
        Args:
            memory_type: 可选的记忆类型过滤器
            
        Returns:
            记忆记录列表
        """
        try:
            # 首先从缓存中获取记忆
            cached_memories = [
                record for record in self.cache.cache.values()
                if memory_type is None or record.memory_type == memory_type
            ]
            
            # 如果缓存中有足够的记忆，直接返回
            if cached_memories:
                return cached_memories
            
            # 否则尝试从搜索API获取
            # 使用空字符串查询以获取所有记忆，降低相似度阈值以确保获取所有结果
            query = ""
            memory_type_str = memory_type.value if memory_type else None
            
            search_results = self.memory_system.search_memories(
                query=query,
                memory_type=memory_type_str,
                limit=1000,
                similarity_threshold=0.0,
                rerank=False
            )
            
            # 转换为MemoryRecord并过滤当前会话
            records = []
            for result in search_results:
                # 过滤当前会话的记忆
                metadata = result.get("metadata", {})
                if metadata.get("session_id") != self.session_id:
                    continue
                
                record = self._create_record_from_memory_data(result)
                if record:
                    records.append(record)
            
            return records
            
        except Exception as e:
            print(f"[记忆管理器] 获取所有记忆失败: {e}")
            # 如果搜索失败，返回缓存中的记忆
            return [
                record for record in self.cache.cache.values()
                if memory_type is None or record.memory_type == memory_type
            ]

    def get_conversation_history(self, limit: int = 100,
                                 start_date: Optional[datetime] = None,
                                 end_date: Optional[datetime] = None) -> List[Dict[str, Any]]:
        """获取对话历史（支持时间范围）"""
        # 首先尝试从缓存中获取
        cache_key = f"conversation_history_{limit}_{start_date}_{end_date}"
        cached = self.cache.get(cache_key)
        if cached and isinstance(cached.content, list):
            return cached.content[:limit]

        # 从记忆系统加载（使用按时间顺序检索）
        conversations = self.memory_system.get_recent_conversations(
            session_id=self.session_id,
            limit=limit * 2  # 获取更多用于过滤
        )

        # 过滤和时间范围筛选
        history = []
        for conv in conversations:
            try:
                conv_time = datetime.fromisoformat(
                    conv.get("metadata", {}).get("created_at", datetime.now().isoformat())
                )

                # 时间范围过滤
                if start_date and conv_time < start_date:
                    continue
                if end_date and conv_time > end_date:
                    continue

                history.append({
                    "timestamp": conv_time.isoformat(),
                    "user_input": self._extract_user_input(conv.get("content", "")),
                    "system_response": self._extract_system_response(conv.get("content", "")),
                    "context": {},
                    "vector_state": {}
                })

                if len(history) >= limit:
                    break

            except Exception as e:
                print(f"[记忆管理器] 解析对话失败: {e}")

        # 缓存结果
        cache_record = MemoryRecord(
            memory_id=cache_key,
            content=history,
            memory_type=MemoryType.CONVERSATION,
            metadata={"cached": True},
            tags=["cached", "conversation_history"],
            priority=MemoryPriority.LOW,
            created_at=datetime.now()
        )
        self.cache.set(cache_key, cache_record)

        return history

    def _clear_conversation_history_cache(self):
        """清除所有对话历史缓存（使用新的缓存失效机制）"""
        try:
            # 使用新的invalidate_by_prefix方法
            cleared_count = self.cache.invalidate_by_prefix("conversation_history_")
            print(f"[记忆管理器] 已清除 {cleared_count} 个对话历史缓存")

        except Exception as e:
            print(f"[记忆管理器] 清除对话历史缓存失败: {e}")

    def invalidate_cache_by_tag(self, tag: str) -> int:
        """按标签失效缓存"""
        try:
            cleared_count = self.cache.invalidate_by_tag(tag)
            print(f"[记忆管理器] 已清除标签 '{tag}' 的 {cleared_count} 个缓存项")
            return cleared_count
        except Exception as e:
            print(f"[记忆管理器] 按标签失效缓存失败: {e}")
            return 0

    def invalidate_cache_by_memory_type(self, memory_type: Union[str, MemoryType]) -> int:
        """按记忆类型失效缓存"""
        try:
            if isinstance(memory_type, str):
                memory_type_enum = MemoryType(memory_type)
            else:
                memory_type_enum = memory_type

            cleared_count = self.cache.invalidate_by_memory_type(memory_type_enum)
            print(f"[记忆管理器] 已清除类型 '{memory_type_enum.value}' 的 {cleared_count} 个缓存项")
            return cleared_count
        except Exception as e:
            print(f"[记忆管理器] 按记忆类型失效缓存失败: {e}")
            return 0

    def get_cache_stats(self) -> Dict[str, Any]:
        """获取缓存统计信息"""
        return self.cache.get_stats()

    def get_memory_version(self, memory_id: str) -> int:
        """获取记忆的当前版本号"""
        return self.cache.get_version(memory_id)

    def record_interaction(self, interaction_data: Dict) -> bool:
        """记录交互（优化版）"""
        try:
            # 提取关键信息
            user_input = interaction_data.get("user_input", "")
            system_response = interaction_data.get("system_response", "")

            if not user_input or not system_response:
                print("[记忆管理器] 交互数据不完整")
                return False

            # 计算情感强度
            emotional_intensity = self._calculate_emotional_intensity(user_input)

            # 构建记忆内容
            memory_content = [
                {"role": "user", "content": user_input},
                {"role": "assistant", "content": system_response}
            ]

            # 提取关键词作为标签
            tags = ["对话", "交互"]
            keywords = self._extract_keywords(user_input)
            tags.extend(keywords[:3])  # 最多添加3个关键词

            # 添加到记忆系统
            memory_id = self.add_memory(
                content=memory_content,
                memory_type=MemoryType.CONVERSATION,
                tags=tags,
                emotional_intensity=emotional_intensity,
                metadata={
                    "interaction_id": interaction_data.get("interaction_id"),
                    "context_keys": list(interaction_data.get("context", {}).keys()),
                    "retrieved_memories_count": interaction_data.get("retrieved_memories_count", 0),
                    "has_resonant_memory": interaction_data.get("resonant_memory", False),
                    "has_cognitive_result": interaction_data.get("cognitive_result", False),
                    "has_growth_result": bool(interaction_data.get("growth_result"))
                },
                priority=MemoryPriority.MEDIUM,
                ttl_hours=720  # 30天TTL
            )

            print(f"[记忆管理器] 交互记录成功，ID: {memory_id}")

            # 清除对话历史缓存，确保下次获取时包含最新对话
            self._clear_conversation_history_cache()

            return True

        except Exception as e:
            print(f"[记忆管理器] 记录交互失败: {e}")
            return False

    def _calculate_emotional_intensity(self, text: str) -> float:
        """计算情感强度"""
        # 简单的情感词检测
        positive_words = ['高兴', '快乐', '喜欢', '爱', '好', '开心', '幸福']
        negative_words = ['难过', '伤心', '生气', '愤怒', '讨厌', '恨', '不好']

        text_lower = text.lower()
        pos_count = sum(1 for word in positive_words if word in text_lower)
        neg_count = sum(1 for word in negative_words if word in text_lower)

        if pos_count > neg_count:
            return 0.7 + (pos_count * 0.1)  # 0.7-1.0
        elif neg_count > pos_count:
            return 0.3 - (neg_count * 0.1)  # 0.0-0.3
        else:
            return 0.5

    def _extract_keywords(self, text: str, max_keywords: int = 5) -> List[str]:
        """提取关键词"""
        # 简单的关键词提取（可替换为更复杂的方法）
        words = text.split()
        if len(words) <= 3:
            return words

        # 去除停用词
        stop_words = {'的', '了', '在', '是', '我', '你', '他', '她', '它'}
        keywords = [word for word in words if word not in stop_words and len(word) > 1]

        return keywords[:max_keywords]

    def cleanup_memory(self,
                       max_memories: int = 10000,
                       keep_high_priority: bool = True,
                       aggressive: bool = False):
        """清理记忆（优化版）"""
        print(f"[记忆管理器] 开始清理记忆，当前总数: {self.stats['total_memories']}")

        try:
            # 获取所有记忆
            all_memories = []
            for tag_memories in self.semantic_index.values():
                all_memories.extend(tag_memories)

            all_memories = list(set(all_memories))

            if len(all_memories) <= max_memories:
                print("[记忆管理器] 记忆数量在限制内，无需清理")
                return

            # 为每个记忆计算保留分数
            memory_scores = {}
            for mem_id in all_memories:
                record = self._get_memory_record(mem_id)
                if not record:
                    continue

                # 计算保留分数
                score = self._calculate_retention_score(record)
                memory_scores[mem_id] = score

            # 按分数排序
            sorted_memories = sorted(memory_scores.items(), key=lambda x: x[1])

            # 确定需要清理的数量
            to_remove_count = len(all_memories) - max_memories
            to_remove = [mem_id for mem_id, _ in sorted_memories[:to_remove_count]]

            # 清理
            for mem_id in to_remove:
                self._remove_memory(mem_id)

            # 清理缓存
            self.cache.clear()

            # 重建索引
            self._rebuild_indexes()

            # 更新统计
            self.stats["total_memories"] = max_memories
            self.stats["last_cleanup_time"] = datetime.now().isoformat()

            print(f"[记忆管理器] 清理完成，移除了 {len(to_remove)} 条记忆")

        except Exception as e:
            print(f"[记忆管理器] 清理记忆失败: {e}")

    def _calculate_retention_score(self, record: MemoryRecord) -> float:
        """计算保留分数（分数越低越容易被清理）"""
        score = 0.0

        # 优先级权重
        priority_weights = {
            MemoryPriority.CRITICAL: 1000,
            MemoryPriority.HIGH: 100,
            MemoryPriority.MEDIUM: 10,
            MemoryPriority.LOW: 1,
            MemoryPriority.ARCHIVE: 0.5
        }
        score += priority_weights.get(record.priority, 1)

        # 访问频率
        score += record.access_count * 2

        # 新近度
        score += record.access_recency * 50

        # 战略价值
        score += record.strategic_score * 100

        # 情感强度
        score += record.emotional_intensity * 20

        # 大小惩罚（越大越容易被清理）
        score -= record.size_bytes / 1024  # 每KB减1分

        # 年龄惩罚（越老越容易被清理）
        age_days = record.age_hours / 24
        if age_days > 30:  # 超过30天开始有惩罚
            score -= (age_days - 30) * 0.1

        return max(0.0, score)

    def _remove_memory(self, memory_id: str):
        """移除记忆"""
        # 从索引中移除
        with self._index_lock:
            if memory_id in self.reverse_index:
                for tag in self.reverse_index[memory_id]:
                    if memory_id in self.semantic_index[tag]:
                        self.semantic_index[tag].remove(memory_id)
                del self.reverse_index[memory_id]

            # 从时间索引中移除
            for date_str, mem_ids in self.temporal_index.items():
                if memory_id in mem_ids:
                    mem_ids.remove(memory_id)
                    if not mem_ids:
                        del self.temporal_index[date_str]

        # 从缓存中移除
        self.cache.delete(memory_id)

    def _rebuild_indexes(self):
        """重建索引"""
        print("[记忆管理器] 重建索引...")

        with self._index_lock:
            self.semantic_index.clear()
            self.reverse_index.clear()
            self.temporal_index.clear()

            # 重新索引所有缓存的记忆
            for memory_id, record in self.cache.cache.items():
                self._update_indexes(record)

    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息（优化版）"""
        with self._stats_lock:
            # 更新缓存命中率
            if self._access_count > 0:
                self.stats["cache_hit_rate"] = self._cache_hits / self._access_count

            # 更新平均响应时间
            if self._access_count > 0:
                self.stats["average_response_time_ms"] = self._total_response_time / self._access_count

            # 获取记忆系统统计
            system_stats = self.memory_system.get_statistics()

            # 合并统计信息
            stats = {
                **self.stats,
                "memory_system_stats": system_stats,
                "cache_size": len(self.cache.cache),
                "index_sizes": {
                    "semantic": len(self.semantic_index),
                    "temporal": len(self.temporal_index)
                },
                "session_id": self.session_id,
                "user_id": self.user_id,
                "current_time": datetime.now().isoformat()
            }

            return stats

    def _record_response_time(self, response_time_ms: float):
        """记录响应时间"""
        with self._stats_lock:
            self._total_response_time += response_time_ms

    def export_memories(self,
                        file_path: str,
                        format_type: str = "json",
                        include_cache: bool = False,
                        compress: bool = True) -> bool:
        """导出记忆（支持压缩）"""
        try:
            # 收集所有记忆
            all_memories = []

            # 从记忆系统导出
            system_success = self.memory_system.export_memories(file_path, format_type)
            if not system_success:
                print("[记忆管理器] 记忆系统导出失败")
                return False

            # 如果需要包含缓存
            if include_cache:
                cache_file = file_path.replace(".", "_cache.")
                cache_data = {
                    "cache": {mid: record.to_dict() for mid, record in self.cache.cache.items()},
                    "indexes": {
                        "semantic": dict(self.semantic_index),
                        "temporal": dict(self.temporal_index)
                    },
                    "stats": self.stats
                }

                with open(cache_file, 'w', encoding='utf-8') as f:
                    json.dump(cache_data, f, ensure_ascii=False, indent=2)

            print(f"[记忆管理器] 记忆导出成功: {file_path}")
            return True

        except Exception as e:
            print(f"[记忆管理器] 导出记忆失败: {e}")
            return False

    def test_connection(self) -> Tuple[bool, str]:
        """测试连接（返回详细结果）"""
        try:
            success = self.memory_system.test_connection()
            if success:
                return True, "记忆系统连接正常"
            else:
                return False, "记忆系统连接失败"
        except Exception as e:
            return False, f"连接测试异常: {e}"

    def _background_cleanup(self):
        """后台清理线程"""
        import time

        while True:
            try:
                # 每6小时运行一次
                time.sleep(6 * 3600)

                # 清理过期缓存
                self.cache.clear()

                # 自动清理记忆
                if self.stats["total_memories"] > 5000:
                    self.cleanup_memory(max_memories=5000, aggressive=False)

                # 更新存储统计
                storage_stats = self.storage_optimizer.get_stats()
                self.performance_monitor.update_storage_stats(storage_stats)

            except Exception as e:
                print(f"[记忆管理器] 后台清理失败: {e}")

    @staticmethod
    def _extract_user_input(content: Any) -> str:
        """提取用户输入"""
        if isinstance(content, str):
            if "user:" in content.lower():
                parts = content.split("user:", 1)
                if len(parts) > 1:
                    return parts[1].split("\n", 1)[0].strip()
            return content
        elif isinstance(content, list):
            for item in content:
                if isinstance(item, dict) and item.get("role") == "user":
                    return item.get("content", "")
        return ""

    @staticmethod
    def _extract_system_response(content: Any) -> str:
        """提取系统响应"""
        if isinstance(content, str):
            if "assistant:" in content.lower():
                parts = content.split("assistant:", 1)
                if len(parts) > 1:
                    return parts[1].strip()
        elif isinstance(content, list):
            for item in content:
                if isinstance(item, dict) and item.get("role") == "assistant":
                    return item.get("content", "")
        return ""

    def get_performance_stats(self, operation: Optional[str] = None) -> Dict[str, Any]:
        """获取性能统计信息"""
        if operation:
            return self.performance_monitor.get_operation_stats(operation) or {}
        return self.performance_monitor.get_all_stats()

    def get_performance_summary(self) -> Dict[str, Any]:
        """获取性能摘要"""
        return self.performance_monitor.get_performance_summary()

    def get_recent_performance_metrics(self, operation: Optional[str] = None, limit: int = 100) -> List[Dict[str, Any]]:
        """获取最近的性能指标"""
        metrics = self.performance_monitor.get_recent_metrics(operation, limit)
        return [asdict(metric) for metric in metrics]

    def export_performance_metrics(self, filepath: str):
        """导出性能指标到文件"""
        self.performance_monitor.export_metrics(filepath)
        print(f"[记忆管理器] 性能指标已导出到: {filepath}")

    def clear_performance_metrics(self):
        """清除所有性能指标"""
        self.performance_monitor.clear_metrics()
        print("[记忆管理器] 性能指标已清除")

    def get_detailed_statistics(self) -> Dict[str, Any]:
        """获取详细统计信息（包含性能监控和去重统计）"""
        stats = self.get_statistics()
        stats["performance"] = {
            "summary": self.get_performance_summary(),
            "operations": self.get_performance_stats()
        }
        stats["deduplication"] = self.deduplicator.get_stats()
        stats["storage"] = self.performance_monitor.get_storage_stats()
        stats["lifecycle"] = self.performance_monitor.get_lifecycle_stats()
        return stats

    def set_deduplication_strategy(self, strategy: DeduplicationStrategy):
        """设置去重策略"""
        self.deduplicator.default_strategy = strategy
        print(f"[记忆管理器] 去重策略已更新为: {strategy.value}")

    def enable_deduplication(self, enabled: bool = True):
        """启用或禁用去重功能"""
        self.deduplicator.enable_deduplication = enabled
        status = "启用" if enabled else "禁用"
        print(f"[记忆管理器] 去重功能已{status}")

    def set_similarity_threshold(self, threshold: float):
        """设置相似度阈值"""
        if 0.0 <= threshold <= 1.0:
            self.deduplicator.similarity_threshold = threshold
            print(f"[记忆管理器] 相似度阈值已设置为: {threshold}")
        else:
            print(f"[记忆管理器] 错误：相似度阈值必须在0.0到1.0之间")

    def clear_deduplication_cache(self):
        """清除去重缓存"""
        with self.deduplicator.lock:
            self.deduplicator.content_hashes.clear()
        print(f"[记忆管理器] 去重缓存已清除")

    def adjust_memory_priority(self, memory_id: str) -> Optional[MemoryPriority]:
        """
        调整记忆优先级
        
        Args:
            memory_id: 记忆ID
            
        Returns:
            调整后的优先级，如果不需要调整则返回None
        """
        record = self.cache.get(memory_id)
        if not record:
            return None
        
        new_priority = self.priority_manager.adjust_priority(record)
        if new_priority:
            old_priority = record.priority
            record.priority = new_priority
            record.updated_at = datetime.now()
            
            # 更新缓存
            self.cache.set(memory_id, record, record.version)
            
            print(f"[记忆管理器] 记忆优先级已调整: {memory_id} {old_priority.value} -> {new_priority.value}")
            
            # 更新统计
            with self._stats_lock:
                self.stats["memory_by_priority"][old_priority.value] -= 1
                self.stats["memory_by_priority"][new_priority.value] += 1
        
        return new_priority

    def batch_adjust_priorities(self, memory_type: Optional[MemoryType] = None) -> Dict[str, int]:
        """
        批量调整记忆优先级
        
        Args:
            memory_type: 可选的记忆类型过滤器
            
        Returns:
            调整统计 {"promoted": 提升数量, "demoted": 降低数量}
        """
        records = list(self.cache.cache.values())
        
        if memory_type:
            records = [r for r in records if r.memory_type == memory_type]
        
        stats = {"promoted": 0, "demoted": 0}
        
        for record in records:
            new_priority = self.priority_manager.adjust_priority(record)
            if new_priority:
                old_priority = record.priority
                record.priority = new_priority
                record.updated_at = datetime.now()
                
                # 更新缓存
                self.cache.set(record.memory_id, record, record.version)
                
                if new_priority.value in ["high", "critical"]:
                    stats["promoted"] += 1
                elif new_priority.value in ["low", "archive"]:
                    stats["demoted"] += 1
                
                # 更新统计
                with self._stats_lock:
                    self.stats["memory_by_priority"][old_priority.value] -= 1
                    self.stats["memory_by_priority"][new_priority.value] += 1
        
        print(f"[记忆管理器] 批量优先级调整完成: 提升 {stats['promoted']} 条, 降低 {stats['demoted']} 条")
        return stats

    def get_priority_stats(self) -> Dict[str, Any]:
        """
        获取优先级统计信息
        
        Returns:
            统计信息字典
        """
        records = list(self.cache.cache.values())
        return self.priority_manager.get_priority_stats(records)

    def get_top_priority_memories(self, limit: int = 10, 
                                  min_score: float = 0.5) -> List[MemoryRecord]:
        """
        获取高优先级记忆
        
        Args:
            limit: 返回数量限制
            min_score: 最小优先级分数
            
        Returns:
            高优先级记忆列表
        """
        records = list(self.cache.cache.values())
        return self.priority_manager.get_top_priority_memories(records, limit, min_score)

    def sort_memories_by_priority(self, include_score: bool = False) -> List[Union[MemoryRecord, Tuple[MemoryRecord, float]]]:
        """
        按优先级分数排序所有记忆
        
        Args:
            include_score: 是否包含优先级分数
            
        Returns:
            排序后的记忆列表
        """
        records = list(self.cache.cache.values())
        return self.priority_manager.sort_by_priority(records, include_score)

    def update_priority_thresholds(self, 
                                   promote: Optional[Dict[Union[str, MemoryPriority], Dict[str, int]]] = None,
                                   demote: Optional[Dict[Union[str, MemoryPriority], Dict[str, int]]] = None):
        """
        更新优先级阈值配置
        
        Args:
            promote: 提升阈值配置
            demote: 降低阈值配置
        """
        self.priority_manager.update_priority_thresholds(promote, demote)
        print(f"[记忆管理器] 优先级阈值配置已更新")

    def get_priority_thresholds(self) -> Dict[str, Any]:
        """
        获取当前优先级阈值配置
        
        Returns:
            阈值配置字典
        """
        return self.priority_manager.get_priority_thresholds()

    def calculate_priority_score(self, memory_id: str) -> Optional[float]:
        """
        计算记忆的优先级分数
        
        Args:
            memory_id: 记忆ID
            
        Returns:
            优先级分数，如果记忆不存在则返回None
        """
        record = self.cache.get(memory_id)
        if not record:
            return None
        return self.priority_manager.calculate_priority_score(record)

    def compress_memory(self, memory_id: str) -> Optional[MemoryRecord]:
        """
        压缩记忆
        
        Args:
            memory_id: 记忆ID
            
        Returns:
            压缩后的记录，如果不需要压缩则返回None
        """
        record = self.cache.get(memory_id)
        if not record:
            return None
        
        # 检查是否应该压缩
        if not self.compression_manager.should_compress(record):
            return None
        
        # 压缩记忆
        compressed_record = self.compression_manager.compress_memory(record)
        
        # 更新缓存
        self.cache.set(memory_id, compressed_record, compressed_record.version)
        
        # 更新统计
        with self._stats_lock:
            self.stats["total_memories"] = self.cache.get_stats()["total_items"]
        
        print(f"[记忆管理器] 记忆已压缩: {memory_id}, 原始大小: {compressed_record.metadata.get('original_size', 0)} bytes, 压缩后: {compressed_record.size_bytes} bytes")
        
        return compressed_record

    def archive_memory(self, memory_id: str) -> Optional[MemoryRecord]:
        """
        归档记忆
        
        Args:
            memory_id: 记忆ID
            
        Returns:
            归档后的记录，如果不需要归档则返回None
        """
        record = self.cache.get(memory_id)
        if not record:
            return None
        
        # 检查是否应该归档
        if not self.compression_manager.should_archive(record):
            return None
        
        # 归档记忆
        archived_record = self.compression_manager.archive_memory(record)
        
        # 更新缓存
        self.cache.set(memory_id, archived_record, archived_record.version)
        
        # 更新统计
        with self._stats_lock:
            self.stats["memory_by_priority"][record.priority.value] -= 1
            self.stats["memory_by_priority"][archived_record.priority.value] += 1
            self.stats["total_memories"] = self.cache.get_stats()["total_items"]
        
        print(f"[记忆管理器] 记忆已归档: {memory_id}")
        
        return archived_record

    def decompress_memory(self, memory_id: str) -> Optional[MemoryRecord]:
        """
        解压缩记忆
        
        Args:
            memory_id: 记忆ID
            
        Returns:
            解压后的记录，如果不需要解压则返回None
        """
        record = self.cache.get(memory_id)
        if not record:
            return None
        
        # 检查是否已压缩
        if not record.metadata.get("compressed", False):
            return None
        
        # 解压内容
        decompressed_content = self.compression_manager.decompress_content(record.content)
        
        # 更新记录
        record.content = decompressed_content
        record.metadata["compressed"] = False
        original_size = record.metadata.pop("original_size", None)
        if original_size:
            record.size_bytes = original_size
        record.updated_at = datetime.now()
        
        # 更新缓存
        self.cache.set(memory_id, record, record.version)
        
        print(f"[记忆管理器] 记忆已解压: {memory_id}")
        
        return record

    def batch_compress_memories(self, memory_ids: Optional[List[str]] = None) -> Dict[str, int]:
        """
        批量压缩记忆
        
        Args:
            memory_ids: 记忆ID列表，如果为None则压缩所有符合条件的记忆
            
        Returns:
            压缩结果统计
        """
        if memory_ids is None:
            # 获取所有记忆ID
            all_memories = self.cache.get_all()
            memory_ids = list(all_memories.keys())
        
        compressed_count = 0
        skipped_count = 0
        
        for memory_id in memory_ids:
            result = self.compress_memory(memory_id)
            if result:
                compressed_count += 1
            else:
                skipped_count += 1
        
        print(f"[记忆管理器] 批量压缩完成: 压缩 {compressed_count} 条, 跳过 {skipped_count} 条")
        
        return {
            "compressed": compressed_count,
            "skipped": skipped_count,
            "total": len(memory_ids)
        }

    def batch_archive_memories(self, memory_ids: Optional[List[str]] = None) -> Dict[str, int]:
        """
        批量归档记忆
        
        Args:
            memory_ids: 记忆ID列表，如果为None则归档所有符合条件的记忆
            
        Returns:
            归档结果统计
        """
        if memory_ids is None:
            # 获取所有记忆ID
            all_memories = self.cache.get_all()
            memory_ids = list(all_memories.keys())
        
        archived_count = 0
        skipped_count = 0
        
        for memory_id in memory_ids:
            result = self.archive_memory(memory_id)
            if result:
                archived_count += 1
            else:
                skipped_count += 1
        
        print(f"[记忆管理器] 批量归档完成: 归档 {archived_count} 条, 跳过 {skipped_count} 条")
        
        return {
            "archived": archived_count,
            "skipped": skipped_count,
            "total": len(memory_ids)
        }

    def get_compression_stats(self) -> Dict[str, Any]:
        """
        获取压缩统计信息
        
        Returns:
            统计信息字典
        """
        stats = self.compression_manager.get_compression_stats()
        
        # 添加缓存中的压缩记忆统计
        all_memories = self.cache.get_all()
        compressed_in_cache = sum(1 for record in all_memories.values() if record.metadata.get("compressed", False))
        archived_in_cache = sum(1 for record in all_memories.values() if record.metadata.get("archived", False))
        
        stats["compressed_in_cache"] = compressed_in_cache
        stats["archived_in_cache"] = archived_in_cache
        
        return stats

    def update_compression_thresholds(self, **kwargs):
        """
        更新压缩阈值配置
        
        Args:
            kwargs: 阈值配置参数
        """
        self.compression_manager.update_compression_thresholds(**kwargs)
        print(f"[记忆管理器] 压缩阈值配置已更新")

    def update_archive_thresholds(self, **kwargs):
        """
        更新归档阈值配置
        
        Args:
            kwargs: 阈值配置参数
        """
        self.compression_manager.update_archive_thresholds(**kwargs)
        print(f"[记忆管理器] 归档阈值配置已更新")

    def get_compression_thresholds(self) -> Dict[str, Any]:
        """
        获取当前压缩和归档阈值配置
        
        Returns:
            阈值配置字典
        """
        return self.compression_manager.get_thresholds()

    def auto_compress_and_archive(self) -> Dict[str, int]:
        """
        自动压缩和归档符合条件的记忆
        
        Returns:
            操作结果统计
        """
        # 获取所有记忆ID
        all_memories = self.cache.get_all()
        memory_ids = list(all_memories.keys())
        
        compressed_count = 0
        archived_count = 0
        
        for memory_id in memory_ids:
            record = all_memories[memory_id]
            
            # 检查是否应该归档
            if self.compression_manager.should_archive(record):
                result = self.archive_memory(memory_id)
                if result:
                    archived_count += 1
            # 检查是否应该压缩
            elif self.compression_manager.should_compress(record):
                result = self.compress_memory(memory_id)
                if result:
                    compressed_count += 1
        
        print(f"[记忆管理器] 自动压缩和归档完成: 压缩 {compressed_count} 条, 归档 {archived_count} 条")
        
        return {
            "compressed": compressed_count,
            "archived": archived_count
        }

    def update_lifecycle_stage(self, memory_id: str) -> Optional[MemoryRecord]:
        """
        更新记忆的生命周期阶段
        
        Args:
            memory_id: 记忆ID
            
        Returns:
            更新后的记录，如果没有变化则返回None
        """
        record = self.cache.get(memory_id)
        if not record:
            return None
        
        old_stage = record.lifecycle_stage
        new_stage = self.lifecycle_manager.update_lifecycle_stage(record)
        if new_stage:
            # 更新缓存
            self.cache.set(memory_id, record, record.version)
            print(f"[记忆管理器] 记忆生命周期阶段已更新: {memory_id} -> {new_stage.value}")
            
            # 更新生命周期统计
            self.performance_monitor.update_lifecycle_stats(new_stage.value)
            
            return record
        
        return None

    def batch_update_lifecycle_stages(self, memory_ids: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        批量更新记忆的生命周期阶段
        
        Args:
            memory_ids: 记忆ID列表，如果为None则更新所有记忆
            
        Returns:
            更新结果统计
        """
        if memory_ids is None:
            # 获取所有记忆ID
            all_memories = self.cache.get_all()
            memory_ids = list(all_memories.keys())
        
        updated_count = 0
        unchanged_count = 0
        
        for memory_id in memory_ids:
            new_stage = self.update_lifecycle_stage(memory_id)
            if new_stage:
                updated_count += 1
            else:
                unchanged_count += 1
        
        # 计算阶段分布
        all_memories = self.cache.get_all()
        stage_distribution = defaultdict(int)
        for record in all_memories.values():
            stage_distribution[record.lifecycle_stage.value] += 1
        
        print(f"[记忆管理器] 批量更新生命周期阶段完成: 更新 {updated_count} 条, 未变化 {unchanged_count} 条")
        
        return {
            "updated": updated_count,
            "unchanged": unchanged_count,
            "total": len(memory_ids),
            "stage_distribution": dict(stage_distribution)
        }

    def get_lifecycle_stats(self) -> Dict[str, Any]:
        """
        获取生命周期统计信息
        
        Returns:
            统计信息字典
        """
        all_memories = self.cache.get_all()
        stats = self.lifecycle_manager.get_lifecycle_stats(all_memories)
        
        # 添加total键以保持兼容性
        stats["total"] = stats.get("total_count", 0)
        
        # 计算平均访问次数
        if all_memories:
            total_access_count = sum(record.access_count for record in all_memories.values())
            stats["avg_access_count"] = total_access_count / len(all_memories)
        else:
            stats["avg_access_count"] = 0.0
        
        # 添加阶段分布
        stats["stage_distribution"] = stats.get("stage_counts", {})
        
        return stats

    def get_lifecycle_thresholds(self) -> Dict[str, Any]:
        """
        获取当前生命周期阈值配置
        
        Returns:
            阈值配置字典
        """
        return self.lifecycle_manager.get_lifecycle_thresholds()

    def update_lifecycle_thresholds(self, stage: str, **kwargs):
        """
        更新生命周期阈值配置
        
        Args:
            stage: 生命周期阶段
            kwargs: 阈值配置参数
        """
        self.lifecycle_manager.update_lifecycle_thresholds(stage, **kwargs)
        print(f"[记忆管理器] 生命周期阈值配置已更新: {stage}")

    def promote_memory(self, memory_id: str) -> Optional[MemoryRecord]:
        """
        提升记忆到下一生命周期阶段
        
        Args:
            memory_id: 记忆ID
            
        Returns:
            提升后的记录，如果失败则返回None
        """
        record = self.cache.get(memory_id)
        if not record:
            return None
        
        if self.lifecycle_manager.should_promote(record):
            # 提升到下一阶段
            current_stage = record.lifecycle_stage
            
            if current_stage == MemoryLifecycleStage.DECLINING:
                new_stage = MemoryLifecycleStage.ACTIVE
            elif current_stage == MemoryLifecycleStage.MATURE:
                new_stage = MemoryLifecycleStage.ACTIVE
            elif current_stage == MemoryLifecycleStage.STABLE:
                new_stage = MemoryLifecycleStage.MATURE
            else:
                return None
            
            old_stage = record.lifecycle_stage
            record.lifecycle_stage = new_stage
            record.updated_at = datetime.now()
            
            # 更新缓存
            self.cache.set(memory_id, record, record.version)
            
            print(f"[记忆管理器] 记忆生命周期阶段已提升: {memory_id} {old_stage.value} -> {new_stage.value}")
            
            return record
        
        return None

    def demote_memory(self, memory_id: str) -> Optional[MemoryRecord]:
        """
        降级记忆到下一生命周期阶段
        
        Args:
            memory_id: 记忆ID
            
        Returns:
            降级后的记录，如果失败则返回None
        """
        record = self.cache.get(memory_id)
        if not record:
            return None
        
        if self.lifecycle_manager.should_demote(record):
            # 降级到下一阶段
            current_stage = record.lifecycle_stage
            
            if current_stage == MemoryLifecycleStage.ACTIVE:
                new_stage = MemoryLifecycleStage.MATURE
            elif current_stage == MemoryLifecycleStage.MATURE:
                new_stage = MemoryLifecycleStage.STABLE
            elif current_stage == MemoryLifecycleStage.STABLE:
                new_stage = MemoryLifecycleStage.DECLINING
            elif current_stage == MemoryLifecycleStage.DECLINING:
                new_stage = MemoryLifecycleStage.ARCHIVED
            elif current_stage == MemoryLifecycleStage.ARCHIVED:
                new_stage = MemoryLifecycleStage.EXPIRED
            else:
                return None
            
            old_stage = record.lifecycle_stage
            record.lifecycle_stage = new_stage
            record.updated_at = datetime.now()
            
            # 更新缓存
            self.cache.set(memory_id, record, record.version)
            
            print(f"[记忆管理器] 记忆生命周期阶段已降级: {memory_id} {old_stage.value} -> {new_stage.value}")
            
            return record
        
        return None

    def auto_manage_lifecycle(self) -> Dict[str, int]:
        """
        自动管理记忆生命周期
        
        Returns:
            操作结果统计
        """
        # 获取所有记忆ID
        all_memories = self.cache.get_all()
        memory_ids = list(all_memories.keys())
        
        promoted_count = 0
        demoted_count = 0
        
        for memory_id in memory_ids:
            record = all_memories[memory_id]
            
            # 检查是否应该提升
            if self.lifecycle_manager.should_promote(record):
                if self.promote_memory(memory_id):
                    promoted_count += 1
            # 检查是否应该降级
            elif self.lifecycle_manager.should_demote(record):
                if self.demote_memory(memory_id):
                    demoted_count += 1
        
        print(f"[记忆管理器] 自动生命周期管理完成: 提升 {promoted_count} 条, 降级 {demoted_count} 条")
        
        return {
            "promoted": promoted_count,
            "demoted": demoted_count
        }

    def get_memories_by_lifecycle_stage(self, stage: MemoryLifecycleStage) -> List[MemoryRecord]:
        """
        获取指定生命周期阶段的记忆
        
        Args:
            stage: 生命周期阶段
            
        Returns:
            记忆记录列表
        """
        all_memories = self.cache.get_all()
        return [record for record in all_memories.values() if record.lifecycle_stage == stage]

    def delete_memory(self, memory_id: str) -> bool:
        """
        删除指定的记忆
        
        Args:
            memory_id: 记忆ID
            
        Returns:
            是否删除成功
        """
        try:
            with self._index_lock:
                # 从缓存中删除
                self.cache.delete(memory_id)
                
                # 从索引中删除
                if memory_id in self.reverse_index:
                    tags = self.reverse_index[memory_id]
                    for tag in tags:
                        if tag in self.semantic_index:
                            self.semantic_index[tag].remove(memory_id)
                    del self.reverse_index[memory_id]
                
                # 更新统计
                with self._stats_lock:
                    self.stats["total_memories"] -= 1
                
                print(f"[记忆管理器] 记忆已删除: {memory_id}")
                return True
        except Exception as e:
            print(f"[记忆管理器] 删除记忆失败: {memory_id}, 错误: {e}")
            return False

    def cleanup_expired_memories(self) -> int:
        """
        清理过期的记忆
        
        Returns:
            清理的记忆数量
        """
        expired_memories = self.get_memories_by_lifecycle_stage(MemoryLifecycleStage.EXPIRED)
        
        deleted_count = 0
        for record in expired_memories:
            try:
                self.delete_memory(record.memory_id)
                deleted_count += 1
            except Exception as e:
                print(f"[记忆管理器] 清理过期记忆失败: {record.memory_id}, 错误: {e}")
        
        return deleted_count


class MemorySecurityLevel(Enum):
    """记忆安全级别枚举"""
    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    SECRET = "secret"
    TOP_SECRET = "top_secret"


class MemoryPrivacyPolicy(Enum):
    """记忆隐私策略枚举"""
    ALLOW_ALL = "allow_all"
    ALLOW_AUTHENTICATED = "allow_authenticated"
    ALLOW_OWNER = "allow_owner"
    DENY_ALL = "deny_all"


class MemorySecurityManager:
    """记忆安全管理器 - 处理记忆的安全性和隐私保护"""
    
    def __init__(self,
                 enable_encryption: bool = True,
                 encryption_key: Optional[str] = None,
                 enable_access_control: bool = True,
                 enable_audit_log: bool = True,
                 default_security_level: MemorySecurityLevel = MemorySecurityLevel.INTERNAL,
                 default_privacy_policy: MemoryPrivacyPolicy = MemoryPrivacyPolicy.ALLOW_OWNER):
        """
        初始化安全管理器
        
        Args:
            enable_encryption: 是否启用加密
            encryption_key: 加密密钥（如果为None则自动生成）
            enable_access_control: 是否启用访问控制
            enable_audit_log: 是否启用审计日志
            default_security_level: 默认安全级别
            default_privacy_policy: 默认隐私策略
        """
        self.enable_encryption = enable_encryption
        self.enable_access_control = enable_access_control
        self.enable_audit_log = enable_audit_log
        self.default_security_level = default_security_level
        self.default_privacy_policy = default_privacy_policy
        
        # 加密密钥
        self.encryption_key = encryption_key or self._generate_encryption_key()
        
        # 访问控制列表 (ACL)
        self.acl: Dict[str, Dict[str, Any]] = {}  # memory_id -> {user_id: permission_level}
        
        # 记忆安全级别映射
        self.memory_security_levels: Dict[str, MemorySecurityLevel] = {}  # memory_id -> security_level
        
        # 记忆隐私策略映射
        self.memory_privacy_policies: Dict[str, MemoryPrivacyPolicy] = {}  # memory_id -> privacy_policy
        
        # 审计日志
        self.audit_log: List[Dict[str, Any]] = []
        self.max_audit_log_size = 10000
        
        # 敏感词列表
        self.sensitive_keywords: List[str] = [
            "密码", "password", "secret", "密钥", "token",
            "信用卡", "credit_card", "身份证", "id_card",
            "银行账号", "bank_account", "手机号", "phone_number"
        ]
        
        # 数据脱敏规则
        self.masking_rules: Dict[str, str] = {
            "phone": r"\d{3}\d{4}\d{4}",
            "email": r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}",
            "id_card": r"\d{17}[\dXx]",
            "credit_card": r"\d{16,19}"
        }
        
        self.lock = threading.RLock()
    
    def _generate_encryption_key(self) -> str:
        """
        生成加密密钥
        
        Returns:
            加密密钥
        """
        import secrets
        return secrets.token_hex(32)
    
    def _encrypt_content(self, content: str) -> str:
        """
        加密内容
        
        Args:
            content: 要加密的内容
            
        Returns:
            加密后的内容
        """
        if not self.enable_encryption:
            return content
        
        try:
            import base64
            from cryptography.fernet import Fernet
            
            # 使用密钥生成Fernet对象
            key = base64.urlsafe_b64encode(self.encryption_key.encode()[:32])
            cipher = Fernet(key)
            
            # 加密内容
            encrypted = cipher.encrypt(content.encode('utf-8'))
            return base64.b64encode(encrypted).decode('utf-8')
        except ImportError:
            # 如果没有cryptography库，使用简单的XOR加密
            key_bytes = self.encryption_key.encode()
            content_bytes = content.encode('utf-8')
            encrypted = bytes([b ^ key_bytes[i % len(key_bytes)] for i, b in enumerate(content_bytes)])
            return base64.b64encode(encrypted).decode('utf-8')
        except Exception as e:
            print(f"[安全管理器] 加密失败: {e}")
            return content
    
    def _decrypt_content(self, encrypted_content: str) -> str:
        """
        解密内容
        
        Args:
            encrypted_content: 加密的内容
            
        Returns:
            解密后的内容
        """
        if not self.enable_encryption:
            return encrypted_content
        
        try:
            import base64
            from cryptography.fernet import Fernet
            
            # 使用密钥生成Fernet对象
            key = base64.urlsafe_b64encode(self.encryption_key.encode()[:32])
            cipher = Fernet(key)
            
            # 解密内容
            encrypted = base64.b64decode(encrypted_content.encode('utf-8'))
            decrypted = cipher.decrypt(encrypted)
            return decrypted.decode('utf-8')
        except ImportError:
            # 如果没有cryptography库，使用简单的XOR解密
            key_bytes = self.encryption_key.encode()
            encrypted = base64.b64decode(encrypted_content.encode('utf-8'))
            decrypted = bytes([b ^ key_bytes[i % len(key_bytes)] for i, b in enumerate(encrypted)])
            return decrypted.decode('utf-8')
        except Exception as e:
            print(f"[安全管理器] 解密失败: {e}")
            return encrypted_content
    
    def _log_audit(self, action: str, memory_id: str, user_id: str, details: Optional[Dict[str, Any]] = None):
        """
        记录审计日志
        
        Args:
            action: 操作类型
            memory_id: 记忆ID
            user_id: 用户ID
            details: 详细信息
        """
        if not self.enable_audit_log:
            return
        
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "action": action,
            "memory_id": memory_id,
            "user_id": user_id,
            "details": details or {}
        }
        
        with self.lock:
            self.audit_log.append(log_entry)
            
            # 限制日志大小
            if len(self.audit_log) > self.max_audit_log_size:
                self.audit_log = self.audit_log[-self.max_audit_log_size:]
    
    def _detect_sensitive_content(self, content: str) -> List[str]:
        """
        检测敏感内容
        
        Args:
            content: 要检测的内容
            
        Returns:
            检测到的敏感词列表
        """
        detected = []
        content_lower = content.lower()
        
        for keyword in self.sensitive_keywords:
            if keyword.lower() in content_lower:
                detected.append(keyword)
        
        return detected
    
    def _mask_sensitive_data(self, content: str) -> str:
        """
        脱敏敏感数据
        
        Args:
            content: 要脱敏的内容
            
        Returns:
            脱敏后的内容
        """
        import re
        
        masked_content = content
        
        # 脱敏手机号
        masked_content = re.sub(r'(\d{3})\d{4}(\d{4})', r'\1****\2', masked_content)
        
        # 脱敏邮箱
        masked_content = re.sub(r'([a-zA-Z0-9._%+-]{2})[a-zA-Z0-9._%+-]+@([a-zA-Z0-9.-]+\.[a-zA-Z]{2,})',
                               r'\1***@\2', masked_content)
        
        # 脱敏身份证
        masked_content = re.sub(r'(\d{6})\d{8}(\d{4}[\dXx])', r'\1********\2', masked_content)
        
        # 脱敏银行卡
        masked_content = re.sub(r'(\d{4})\d{8,12}(\d{4})', r'\1********\2', masked_content)
        
        return masked_content
    
    def set_memory_security_level(self, memory_id: str, security_level: MemorySecurityLevel) -> bool:
        """
        设置记忆的安全级别
        
        Args:
            memory_id: 记忆ID
            security_level: 安全级别
            
        Returns:
            是否设置成功
        """
        with self.lock:
            self.memory_security_levels[memory_id] = security_level
            self._log_audit("set_security_level", memory_id, "system", {"security_level": security_level.value})
            return True
    
    def set_memory_privacy_policy(self, memory_id: str, privacy_policy: MemoryPrivacyPolicy) -> bool:
        """
        设置记忆的隐私策略
        
        Args:
            memory_id: 记忆ID
            privacy_policy: 隐私策略
            
        Returns:
            是否设置成功
        """
        with self.lock:
            self.memory_privacy_policies[memory_id] = privacy_policy
            self._log_audit("set_privacy_policy", memory_id, "system", {"privacy_policy": privacy_policy.value})
            return True
    
    def grant_access(self, memory_id: str, user_id: str, permission_level: str = "read") -> bool:
        """
        授予用户对记忆的访问权限
        
        Args:
            memory_id: 记忆ID
            user_id: 用户ID
            permission_level: 权限级别 (read, write, admin)
            
        Returns:
            是否授权成功
        """
        with self.lock:
            if memory_id not in self.acl:
                self.acl[memory_id] = {}
            
            self.acl[memory_id][user_id] = permission_level
            self._log_audit("grant_access", memory_id, user_id, {"permission_level": permission_level})
            return True
    
    def revoke_access(self, memory_id: str, user_id: str) -> bool:
        """
        撤销用户对记忆的访问权限
        
        Args:
            memory_id: 记忆ID
            user_id: 用户ID
            
        Returns:
            是否撤销成功
        """
        with self.lock:
            if memory_id in self.acl and user_id in self.acl[memory_id]:
                del self.acl[memory_id][user_id]
                self._log_audit("revoke_access", memory_id, user_id, {})
                return True
            return False
    
    def check_access(self, memory_id: str, user_id: str, required_permission: str = "read") -> bool:
        """
        检查用户是否有访问记忆的权限
        
        Args:
            memory_id: 记忆ID
            user_id: 用户ID
            required_permission: 需要的权限级别
            
        Returns:
            是否有访问权限
        """
        if not self.enable_access_control:
            return True
        
        with self.lock:
            # 检查隐私策略
            privacy_policy = self.memory_privacy_policies.get(memory_id, self.default_privacy_policy)
            
            if privacy_policy == MemoryPrivacyPolicy.ALLOW_ALL:
                return True
            elif privacy_policy == MemoryPrivacyPolicy.DENY_ALL:
                return False
            elif privacy_policy == MemoryPrivacyPolicy.ALLOW_AUTHENTICATED:
                # 假设user_id不为空表示已认证
                return bool(user_id)
            elif privacy_policy == MemoryPrivacyPolicy.ALLOW_OWNER:
                # 检查是否是所有者
                if memory_id in self.acl and user_id in self.acl[memory_id]:
                    user_permission = self.acl[memory_id][user_id]
                    permission_hierarchy = {"read": 1, "write": 2, "admin": 3}
                    return permission_hierarchy.get(user_permission, 0) >= permission_hierarchy.get(required_permission, 1)
                return False
        
        return False
    
    def encrypt_memory(self, memory_id: str, content: str) -> str:
        """
        加密记忆内容
        
        Args:
            memory_id: 记忆ID
            content: 要加密的内容
            
        Returns:
            加密后的内容
        """
        encrypted = self._encrypt_content(content)
        self._log_audit("encrypt_memory", memory_id, "system", {"success": True})
        return encrypted
    
    def decrypt_memory(self, memory_id: str, encrypted_content: str, user_id: str) -> str:
        """
        解密记忆内容
        
        Args:
            memory_id: 记忆ID
            encrypted_content: 加密的内容
            user_id: 用户ID
            
        Returns:
            解密后的内容
        """
        if not self.check_access(memory_id, user_id, "read"):
            self._log_audit("decrypt_memory", memory_id, user_id, {"success": False, "reason": "access_denied"})
            raise PermissionError(f"用户 {user_id} 没有访问记忆 {memory_id} 的权限")
        
        decrypted = self._decrypt_content(encrypted_content)
        self._log_audit("decrypt_memory", memory_id, user_id, {"success": True})
        return decrypted
    
    def scan_and_mask_sensitive_data(self, memory_id: str, content: str) -> Tuple[str, List[str]]:
        """
        扫描并脱敏敏感数据
        
        Args:
            memory_id: 记忆ID
            content: 要扫描的内容
            
        Returns:
            (脱敏后的内容, 检测到的敏感词列表)
        """
        sensitive_keywords = self._detect_sensitive_content(content)
        masked_content = self._mask_sensitive_data(content)
        
        if sensitive_keywords:
            self._log_audit("sensitive_data_detected", memory_id, "system", {
                "sensitive_keywords": sensitive_keywords,
                "masked": True
            })
        
        return masked_content, sensitive_keywords
    
    def get_security_level(self, memory_id: str) -> MemorySecurityLevel:
        """
        获取记忆的安全级别
        
        Args:
            memory_id: 记忆ID
            
        Returns:
            安全级别
        """
        return self.memory_security_levels.get(memory_id, self.default_security_level)
    
    def get_privacy_policy(self, memory_id: str) -> MemoryPrivacyPolicy:
        """
        获取记忆的隐私策略
        
        Args:
            memory_id: 记忆ID
            
        Returns:
            隐私策略
        """
        return self.memory_privacy_policies.get(memory_id, self.default_privacy_policy)
    
    def get_audit_log(self, memory_id: Optional[str] = None, limit: int = 100) -> List[Dict[str, Any]]:
        """
        获取审计日志
        
        Args:
            memory_id: 记忆ID（可选，如果指定则只返回该记忆的日志）
            limit: 返回的日志数量限制
            
        Returns:
            审计日志列表
        """
        with self.lock:
            if memory_id:
                logs = [log for log in self.audit_log if log["memory_id"] == memory_id]
            else:
                logs = self.audit_log
            
            return logs[-limit:]
    
    def clear_audit_log(self, memory_id: Optional[str] = None) -> bool:
        """
        清除审计日志
        
        Args:
            memory_id: 记忆ID（可选，如果指定则只清除该记忆的日志）
            
        Returns:
            是否清除成功
        """
        with self.lock:
            if memory_id:
                self.audit_log = [log for log in self.audit_log if log["memory_id"] != memory_id]
            else:
                self.audit_log = []
            return True
    
    def get_security_stats(self) -> Dict[str, Any]:
        """
        获取安全统计信息
        
        Returns:
            安全统计信息
        """
        with self.lock:
            security_level_counts = defaultdict(int)
            for level in self.memory_security_levels.values():
                security_level_counts[level.value] += 1
            
            privacy_policy_counts = defaultdict(int)
            for policy in self.memory_privacy_policies.values():
                privacy_policy_counts[policy.value] += 1
            
            return {
                "total_memories_with_security": len(self.memory_security_levels),
                "total_memories_with_privacy": len(self.memory_privacy_policies),
                "security_level_distribution": dict(security_level_counts),
                "privacy_policy_distribution": dict(privacy_policy_counts),
                "total_acl_entries": sum(len(acl) for acl in self.acl.values()),
                "audit_log_size": len(self.audit_log),
                "encryption_enabled": self.enable_encryption,
                "access_control_enabled": self.enable_access_control,
                "audit_log_enabled": self.enable_audit_log
            }