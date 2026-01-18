# ============ 分层存储实现 ============
"""
实现分层存储系统，包括热/温/冷数据分层和缓存管理。
整合原 memory_manager.py 中的 MemoryStorageOptimizer 和 MemoryCache。
"""
import pickle
import threading
import time
from datetime import datetime
from typing import Dict, Any, Optional, List

from .base import BaseStorage, StorageInterface
from ..models import (
    MemoryRecord, 
    StorageTier, 
    MemoryType, 
    MemoryPriority,
    MemoryLifecycleStage,
    StorageTierConfig
)


class MemoryCache:
    """
    记忆缓存管理类
    
    提供带版本控制和TTL的缓存功能。
    """

    def __init__(self, 
                 max_size: int = 1000, 
                 ttl_seconds: int = 86400,  # 24小时
                 storage_optimizer: Optional['TieredStorage'] = None):
        """
        初始化缓存
        
        Args:
            max_size: 最大缓存容量
            ttl_seconds: 缓存过期时间（秒）
            storage_optimizer: 可选的存储优化器引用
        """
        self._cache: Dict[str, Dict[str, Any]] = {}  # {key: {"value": ..., "timestamp": ..., "version": ...}}
        self._max_size = max_size
        self._ttl_seconds = ttl_seconds
        self._storage_optimizer = storage_optimizer
        self._versions: Dict[str, int] = {}
        self._lock = threading.RLock()
        
        # 统计信息
        self._hits = 0
        self._misses = 0

    def get(self, memory_id: str) -> Optional[MemoryRecord]:
        """
        获取缓存项
        
        Args:
            memory_id: 记忆ID
            
        Returns:
            缓存的记忆记录，如果不存在或过期则返回 None
        """
        with self._lock:
            if memory_id not in self._cache:
                self._misses += 1
                return None
            
            cache_entry = self._cache[memory_id]
            
            # 检查是否过期
            if time.time() - cache_entry["timestamp"] > self._ttl_seconds:
                del self._cache[memory_id]
                self._versions.pop(memory_id, None)
                self._misses += 1
                return None
            
            self._hits += 1
            record = cache_entry["value"]
            
            # 更新访问信息
            if isinstance(record, MemoryRecord):
                record.update_access()
                cache_entry["timestamp"] = time.time()
                
                # 通知存储优化器
                if self._storage_optimizer:
                    self._storage_optimizer.update_access_stats(memory_id)
            
            return record

    def set(self, memory_id: str, record: MemoryRecord, version: int = 1) -> None:
        """
        设置缓存项（带版本控制）
        
        Args:
            memory_id: 记忆ID
            record: 记忆记录
            version: 版本号
        """
        with self._lock:
            # 检查版本
            current_version = self._versions.get(memory_id, 0)
            if version <= current_version and memory_id in self._cache:
                return  # 版本过旧，不更新
            
            # 如果缓存已满，淘汰最旧的项
            if len(self._cache) >= self._max_size and memory_id not in self._cache:
                self._evict_oldest()
            
            self._cache[memory_id] = {
                "value": record,
                "timestamp": time.time(),
                "version": version
            }
            self._versions[memory_id] = version

    def delete(self, memory_id: str) -> bool:
        """删除缓存项"""
        with self._lock:
            if memory_id in self._cache:
                del self._cache[memory_id]
                self._versions.pop(memory_id, None)
                return True
            return False

    def clear(self) -> None:
        """清空缓存"""
        with self._lock:
            self._cache.clear()
            self._versions.clear()

    def get_all(self) -> Dict[str, MemoryRecord]:
        """获取所有有效的缓存项"""
        with self._lock:
            self._cleanup_expired()
            return {
                memory_id: entry["value"]
                for memory_id, entry in self._cache.items()
                if isinstance(entry["value"], MemoryRecord)
            }

    def get_version(self, memory_id: str) -> int:
        """获取记忆的版本号"""
        with self._lock:
            return self._versions.get(memory_id, 0)

    def get_stats(self) -> Dict[str, Any]:
        """获取缓存统计信息"""
        with self._lock:
            total_requests = self._hits + self._misses
            return {
                "size": len(self._cache),
                "max_size": self._max_size,
                "utilization": len(self._cache) / self._max_size if self._max_size > 0 else 0,
                "ttl_seconds": self._ttl_seconds,
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": self._hits / total_requests if total_requests > 0 else 0,
                "total_versions": len(self._versions)
            }

    def invalidate_by_prefix(self, prefix: str) -> int:
        """根据前缀无效化缓存"""
        with self._lock:
            invalidated = 0
            keys_to_delete = [k for k in self._cache.keys() if k.startswith(prefix)]
            for key in keys_to_delete:
                del self._cache[key]
                self._versions.pop(key, None)
                invalidated += 1
            return invalidated

    def invalidate_by_tag(self, tag: str) -> int:
        """根据标签无效化缓存"""
        with self._lock:
            invalidated = 0
            keys_to_delete = []
            for memory_id, entry in self._cache.items():
                record = entry["value"]
                if isinstance(record, MemoryRecord) and tag in record.tags:
                    keys_to_delete.append(memory_id)
            
            for key in keys_to_delete:
                del self._cache[key]
                self._versions.pop(key, None)
                invalidated += 1
            return invalidated

    def invalidate_by_memory_type(self, memory_type: MemoryType) -> int:
        """根据记忆类型无效化缓存"""
        with self._lock:
            invalidated = 0
            keys_to_delete = []
            for memory_id, entry in self._cache.items():
                record = entry["value"]
                if isinstance(record, MemoryRecord) and record.memory_type == memory_type:
                    keys_to_delete.append(memory_id)
            
            for key in keys_to_delete:
                del self._cache[key]
                self._versions.pop(key, None)
                invalidated += 1
            return invalidated

    def _cleanup_expired(self) -> None:
        """清理过期的缓存项"""
        current_time = time.time()
        expired_keys = [
            k for k, v in self._cache.items()
            if current_time - v["timestamp"] > self._ttl_seconds
        ]
        for key in expired_keys:
            del self._cache[key]
            self._versions.pop(key, None)

    def _evict_oldest(self) -> None:
        """淘汰最旧的缓存项"""
        if not self._cache:
            return
        oldest_key = min(self._cache.keys(), key=lambda k: self._cache[k]["timestamp"])
        del self._cache[oldest_key]
        self._versions.pop(oldest_key, None)


class TieredStorage:
    """
    分层存储管理器
    
    实现热/温/冷三层数据存储，自动根据访问频率迁移数据。
    整合原 MemoryStorageOptimizer 的功能。
    """

    def __init__(self,
                 hot_config: Optional[StorageTierConfig] = None,
                 warm_config: Optional[StorageTierConfig] = None,
                 cold_config: Optional[StorageTierConfig] = None):
        """
        初始化分层存储
        
        Args:
            hot_config: 热数据层配置
            warm_config: 温数据层配置
            cold_config: 冷数据层配置
        """
        # 默认配置
        self._hot_config = hot_config or StorageTierConfig(500, 10, 24)
        self._warm_config = warm_config or StorageTierConfig(2000, 3, 168)
        self._cold_config = cold_config or StorageTierConfig(10000, 0, 720)
        
        # 分层存储
        self._hot_storage: Dict[str, MemoryRecord] = {}
        self._warm_storage: Dict[str, bytes] = {}  # 压缩存储
        self._cold_storage: Dict[str, bytes] = {}  # 归档存储
        
        # 索引和统计
        self._tier_index: Dict[str, StorageTier] = {}
        self._access_frequency: Dict[str, int] = {}
        self._last_access_time: Dict[str, float] = {}
        
        self._lock = threading.RLock()

    def get(self, memory_id: str) -> Optional[MemoryRecord]:
        """
        获取记忆记录
        
        自动从适当的存储层获取，并根据访问频率提升存储层级。
        """
        with self._lock:
            # 检查热数据层
            if memory_id in self._hot_storage:
                record = self._hot_storage[memory_id]
                self.update_access_stats(memory_id)
                return record
            
            # 检查温数据层
            if memory_id in self._warm_storage:
                record = self._decompress_record(self._warm_storage[memory_id])
                self.update_access_stats(memory_id)
                # 提升到热数据层
                self._promote_to_hot(memory_id, record)
                return record
            
            # 检查冷数据层
            if memory_id in self._cold_storage:
                record = self._decompress_record(self._cold_storage[memory_id])
                self.update_access_stats(memory_id)
                # 提升到温数据层
                self._promote_to_warm(memory_id, record)
                return record
            
            return None

    def set(self, memory_id: str, record: MemoryRecord) -> None:
        """
        存储记忆记录
        
        根据访问频率自动选择存储层级。
        """
        with self._lock:
            self.update_access_stats(memory_id)
            tier = self._determine_storage_tier(memory_id)
            
            # 先从其他层删除
            self._remove_from_all_tiers(memory_id)
            
            if tier == StorageTier.HOT:
                self._store_in_hot(memory_id, record)
            elif tier == StorageTier.WARM:
                self._store_in_warm(memory_id, record)
            else:
                self._store_in_cold(memory_id, record)

    def delete(self, memory_id: str) -> bool:
        """删除记忆记录"""
        with self._lock:
            deleted = False
            if memory_id in self._hot_storage:
                del self._hot_storage[memory_id]
                deleted = True
            if memory_id in self._warm_storage:
                del self._warm_storage[memory_id]
                deleted = True
            if memory_id in self._cold_storage:
                del self._cold_storage[memory_id]
                deleted = True
            
            self._tier_index.pop(memory_id, None)
            self._access_frequency.pop(memory_id, None)
            self._last_access_time.pop(memory_id, None)
            
            return deleted

    def get_all(self) -> Dict[str, MemoryRecord]:
        """获取所有记录"""
        with self._lock:
            result = {}
            result.update(self._hot_storage)
            
            for memory_id, compressed_data in self._warm_storage.items():
                result[memory_id] = self._decompress_record(compressed_data)
            
            for memory_id, compressed_data in self._cold_storage.items():
                result[memory_id] = self._decompress_record(compressed_data)
            
            return result

    def get_stats(self) -> Dict[str, Any]:
        """获取存储统计信息"""
        with self._lock:
            hot_size = sum(len(str(record.content)) for record in self._hot_storage.values())
            warm_size = sum(len(data) for data in self._warm_storage.values())
            cold_size = sum(len(data) for data in self._cold_storage.values())
            
            return {
                "total_items": len(self._tier_index),
                "hot": {
                    "items": len(self._hot_storage),
                    "max_size": self._hot_config.max_size,
                    "size_bytes": hot_size
                },
                "warm": {
                    "items": len(self._warm_storage),
                    "max_size": self._warm_config.max_size,
                    "size_bytes": warm_size
                },
                "cold": {
                    "items": len(self._cold_storage),
                    "max_size": self._cold_config.max_size,
                    "size_bytes": cold_size
                },
                "total_size_bytes": hot_size + warm_size + cold_size,
                "compression_ratio": warm_size / hot_size if hot_size > 0 else 1.0
            }

    def get_tier(self, memory_id: str) -> Optional[StorageTier]:
        """获取记忆的存储层级"""
        return self._tier_index.get(memory_id)

    def optimize(self) -> Dict[str, int]:
        """
        优化存储布局
        
        自动迁移数据到合适的存储层。
        
        Returns:
            迁移统计信息
        """
        with self._lock:
            stats = {
                "hot_to_warm": 0,
                "warm_to_cold": 0,
                "warm_to_hot": 0,
                "cold_to_warm": 0
            }
            
            # 热 -> 温
            stats["hot_to_warm"] = self._demote_hot_to_warm()
            
            # 温 -> 冷
            stats["warm_to_cold"] = self._demote_warm_to_cold()
            
            # 温 -> 热
            stats["warm_to_hot"] = self._promote_warm_to_hot()
            
            # 冷 -> 温
            stats["cold_to_warm"] = self._promote_cold_to_warm()
            
            return stats

    def update_access_stats(self, memory_id: str) -> None:
        """更新访问统计"""
        self._access_frequency[memory_id] = self._access_frequency.get(memory_id, 0) + 1
        self._last_access_time[memory_id] = time.time()

    # ============ 私有方法 ============

    def _determine_storage_tier(self, memory_id: str) -> StorageTier:
        """确定存储层级"""
        access_freq = self._access_frequency.get(memory_id, 0)
        
        if access_freq >= self._hot_config.min_access_count:
            return StorageTier.HOT
        elif access_freq >= self._warm_config.min_access_count:
            return StorageTier.WARM
        else:
            return StorageTier.COLD

    def _remove_from_all_tiers(self, memory_id: str) -> None:
        """从所有存储层移除"""
        self._hot_storage.pop(memory_id, None)
        self._warm_storage.pop(memory_id, None)
        self._cold_storage.pop(memory_id, None)

    def _store_in_hot(self, memory_id: str, record: MemoryRecord) -> None:
        """存储到热数据层"""
        if len(self._hot_storage) >= self._hot_config.max_size:
            self._evict_from_hot()
        
        self._hot_storage[memory_id] = record
        self._tier_index[memory_id] = StorageTier.HOT

    def _store_in_warm(self, memory_id: str, record: MemoryRecord) -> None:
        """存储到温数据层"""
        if len(self._warm_storage) >= self._warm_config.max_size:
            self._evict_from_warm()
        
        self._warm_storage[memory_id] = self._compress_record(record)
        self._tier_index[memory_id] = StorageTier.WARM

    def _store_in_cold(self, memory_id: str, record: MemoryRecord) -> None:
        """存储到冷数据层"""
        if len(self._cold_storage) >= self._cold_config.max_size:
            self._evict_from_cold()
        
        self._cold_storage[memory_id] = self._compress_record(record)
        self._tier_index[memory_id] = StorageTier.COLD

    def _promote_to_hot(self, memory_id: str, record: MemoryRecord) -> None:
        """提升到热数据层"""
        self._warm_storage.pop(memory_id, None)
        self._cold_storage.pop(memory_id, None)
        self._store_in_hot(memory_id, record)

    def _promote_to_warm(self, memory_id: str, record: MemoryRecord) -> None:
        """提升到温数据层"""
        self._cold_storage.pop(memory_id, None)
        self._store_in_warm(memory_id, record)

    def _demote_hot_to_warm(self) -> int:
        """将热数据层中不常用的数据迁移到温数据层"""
        if len(self._hot_storage) <= self._hot_config.max_size * 0.8:
            return 0
        
        demoted = 0
        sorted_items = sorted(
            [(k, self._access_frequency.get(k, 0)) for k in self._hot_storage.keys()],
            key=lambda x: x[1]
        )
        
        for memory_id, freq in sorted_items:
            if freq < self._hot_config.min_access_count:
                record = self._hot_storage.pop(memory_id)
                self._store_in_warm(memory_id, record)
                demoted += 1
                
                if len(self._hot_storage) <= self._hot_config.max_size * 0.8:
                    break
        
        return demoted

    def _demote_warm_to_cold(self) -> int:
        """将温数据层中不常用的数据迁移到冷数据层"""
        if len(self._warm_storage) <= self._warm_config.max_size * 0.8:
            return 0
        
        demoted = 0
        sorted_items = sorted(
            [(k, self._access_frequency.get(k, 0)) for k in self._warm_storage.keys()],
            key=lambda x: x[1]
        )
        
        for memory_id, freq in sorted_items:
            if freq < self._warm_config.min_access_count:
                compressed_data = self._warm_storage.pop(memory_id)
                self._cold_storage[memory_id] = compressed_data
                self._tier_index[memory_id] = StorageTier.COLD
                demoted += 1
                
                if len(self._warm_storage) <= self._warm_config.max_size * 0.8:
                    break
        
        return demoted

    def _promote_warm_to_hot(self) -> int:
        """将温数据层中常用的数据提升到热数据层"""
        if len(self._hot_storage) >= self._hot_config.max_size * 0.9:
            return 0
        
        promoted = 0
        sorted_items = sorted(
            [(k, self._access_frequency.get(k, 0)) for k in self._warm_storage.keys()],
            key=lambda x: x[1],
            reverse=True
        )
        
        for memory_id, freq in sorted_items:
            if freq >= self._hot_config.min_access_count:
                record = self._decompress_record(self._warm_storage.pop(memory_id))
                self._store_in_hot(memory_id, record)
                promoted += 1
                
                if len(self._hot_storage) >= self._hot_config.max_size * 0.9:
                    break
        
        return promoted

    def _promote_cold_to_warm(self) -> int:
        """将冷数据层中常用的数据提升到温数据层"""
        if len(self._warm_storage) >= self._warm_config.max_size * 0.9:
            return 0
        
        promoted = 0
        sorted_items = sorted(
            [(k, self._access_frequency.get(k, 0)) for k in self._cold_storage.keys()],
            key=lambda x: x[1],
            reverse=True
        )
        
        for memory_id, freq in sorted_items:
            if freq >= self._warm_config.min_access_count:
                compressed_data = self._cold_storage.pop(memory_id)
                self._warm_storage[memory_id] = compressed_data
                self._tier_index[memory_id] = StorageTier.WARM
                promoted += 1
                
                if len(self._warm_storage) >= self._warm_config.max_size * 0.9:
                    break
        
        return promoted

    def _evict_from_hot(self) -> None:
        """从热数据层淘汰"""
        if not self._hot_storage:
            return
        
        # LRU策略
        oldest_key = min(
            self._hot_storage.keys(),
            key=lambda k: self._last_access_time.get(k, 0)
        )
        record = self._hot_storage.pop(oldest_key)
        self._store_in_warm(oldest_key, record)

    def _evict_from_warm(self) -> None:
        """从温数据层淘汰"""
        if not self._warm_storage:
            return
        
        oldest_key = min(
            self._warm_storage.keys(),
            key=lambda k: self._last_access_time.get(k, 0)
        )
        compressed_data = self._warm_storage.pop(oldest_key)
        self._cold_storage[oldest_key] = compressed_data
        self._tier_index[oldest_key] = StorageTier.COLD

    def _evict_from_cold(self) -> None:
        """从冷数据层淘汰（永久删除）"""
        if not self._cold_storage:
            return
        
        oldest_key = min(
            self._cold_storage.keys(),
            key=lambda k: self._last_access_time.get(k, 0)
        )
        del self._cold_storage[oldest_key]
        self._tier_index.pop(oldest_key, None)

    def _compress_record(self, record: MemoryRecord) -> bytes:
        """压缩记录"""
        return pickle.dumps(record.to_dict())

    def _decompress_record(self, compressed_data: bytes) -> MemoryRecord:
        """解压缩记录"""
        data = pickle.loads(compressed_data)
        return MemoryRecord.from_dict(data)
