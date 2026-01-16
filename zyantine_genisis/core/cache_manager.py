"""
缓存管理器 - 提供响应缓存、记忆缓存和上下文缓存功能
"""
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime, timedelta
import hashlib
import json
import time
from dataclasses import dataclass, field

from utils.logger import SystemLogger


@dataclass
class CacheItem:
    """缓存项"""
    key: str
    value: Any
    timestamp: float = field(default_factory=time.time)
    expiration: Optional[float] = None
    tags: List[str] = field(default_factory=list)


class CacheManager:
    """缓存管理器"""

    def __init__(self, 
                 max_size: int = 1000, 
                 default_ttl: int = 3600, 
                 enable_response_cache: bool = True, 
                 enable_memory_cache: bool = True, 
                 enable_context_cache: bool = True):
        """
        初始化缓存管理器

        Args:
            max_size: 最大缓存项数量
            default_ttl: 默认过期时间（秒）
            enable_response_cache: 是否启用响应缓存
            enable_memory_cache: 是否启用记忆缓存
            enable_context_cache: 是否启用上下文缓存
        """
        self.max_size = max_size
        self.default_ttl = default_ttl
        self.enable_response_cache = enable_response_cache
        self.enable_memory_cache = enable_memory_cache
        self.enable_context_cache = enable_context_cache
        
        # 缓存存储
        self.caches = {
            "response": {},  # 响应缓存
            "memory": {},    # 记忆缓存
            "context": {}    # 上下文缓存
        }
        
        # 缓存统计
        self.stats = {
            "hits": 0,
            "misses": 0,
            "evictions": 0,
            "sets": 0
        }
        
        self.logger = SystemLogger().get_logger("CacheManager")
        self.logger.info(f"缓存管理器初始化完成，最大缓存项: {max_size}, 默认过期时间: {default_ttl}秒")

    def _generate_key(self, *args, **kwargs) -> str:
        """生成缓存键"""
        data = json.dumps({"args": args, "kwargs": kwargs}, sort_keys=True, ensure_ascii=False)
        return hashlib.md5(data.encode()).hexdigest()

    def _is_expired(self, item: CacheItem) -> bool:
        """检查缓存项是否过期"""
        if item.expiration is None:
            return False
        return time.time() > item.expiration

    def _clean_expired(self, cache_name: str):
        """清理过期缓存"""
        cache = self.caches[cache_name]
        expired_keys = []
        
        for key, item in cache.items():
            if self._is_expired(item):
                expired_keys.append(key)
        
        for key in expired_keys:
            del cache[key]
            self.stats["evictions"] += 1
        
        if expired_keys:
            self.logger.debug(f"清理{cache_name}缓存过期项: {len(expired_keys)}")

    def _ensure_capacity(self, cache_name: str):
        """确保缓存容量"""
        cache = self.caches[cache_name]
        
        while len(cache) >= self.max_size:
            # 移除最旧的缓存项
            oldest_key = min(cache.keys(), key=lambda k: cache[k].timestamp)
            del cache[oldest_key]
            self.stats["evictions"] += 1

    def get(self, cache_name: str, key: str) -> Optional[Any]:
        """获取缓存项"""
        if not self._is_cache_enabled(cache_name):
            return None
        
        cache = self.caches[cache_name]
        
        # 清理过期缓存
        self._clean_expired(cache_name)
        
        item = cache.get(key)
        if item:
            if not self._is_expired(item):
                self.stats["hits"] += 1
                self.logger.debug(f"缓存命中: {cache_name}:{key}")
                return item.value
            else:
                # 移除过期项
                del cache[key]
                self.stats["evictions"] += 1
        
        self.stats["misses"] += 1
        self.logger.debug(f"缓存未命中: {cache_name}:{key}")
        return None

    def set(self, cache_name: str, key: str, value: Any, ttl: Optional[int] = None, tags: List[str] = None):
        """设置缓存项"""
        if not self._is_cache_enabled(cache_name):
            return
        
        cache = self.caches[cache_name]
        
        # 确保容量
        self._ensure_capacity(cache_name)
        
        # 计算过期时间
        expiration = time.time() + (ttl or self.default_ttl) if ttl is not None else None
        
        # 创建缓存项
        item = CacheItem(
            key=key,
            value=value,
            expiration=expiration,
            tags=tags or []
        )
        
        cache[key] = item
        self.stats["sets"] += 1
        self.logger.debug(f"缓存设置: {cache_name}:{key}, 过期时间: {expiration}")

    def delete(self, cache_name: str, key: str):
        """删除缓存项"""
        if not self._is_cache_enabled(cache_name):
            return
        
        cache = self.caches[cache_name]
        if key in cache:
            del cache[key]
            self.logger.debug(f"缓存删除: {cache_name}:{key}")

    def clear(self, cache_name: Optional[str] = None):
        """清空缓存"""
        if cache_name:
            if cache_name in self.caches:
                self.caches[cache_name].clear()
                self.logger.info(f"清空缓存: {cache_name}")
        else:
            # 清空所有缓存
            for name in self.caches:
                self.caches[name].clear()
            self.logger.info("清空所有缓存")

    def get_response(self, user_input: str, context_hash: str) -> Optional[str]:
        """获取响应缓存"""
        key = self._generate_key(user_input, context_hash)
        return self.get("response", key)

    def set_response(self, user_input: str, context_hash: str, response: str, ttl: int = 3600):
        """设置响应缓存"""
        key = self._generate_key(user_input, context_hash)
        self.set("response", key, response, ttl, tags=["response"])

    def get_memory(self, query: str) -> Optional[List[Any]]:
        """获取记忆缓存"""
        key = self._generate_key(query)
        return self.get("memory", key)

    def set_memory(self, query: str, memories: List[Any], ttl: int = 1800):
        """设置记忆缓存"""
        key = self._generate_key(query)
        self.set("memory", key, memories, ttl, tags=["memory"])

    def get_context(self, conversation_id: str) -> Optional[Dict[str, Any]]:
        """获取上下文缓存"""
        return self.get("context", conversation_id)

    def set_context(self, conversation_id: str, context: Dict[str, Any], ttl: int = 7200):
        """设置上下文缓存"""
        self.set("context", conversation_id, context, ttl, tags=["context"])

    def _is_cache_enabled(self, cache_name: str) -> bool:
        """检查缓存是否启用"""
        if cache_name == "response":
            return self.enable_response_cache
        elif cache_name == "memory":
            return self.enable_memory_cache
        elif cache_name == "context":
            return self.enable_context_cache
        return True

    def get_stats(self) -> Dict[str, int]:
        """获取缓存统计信息"""
        # 添加缓存大小信息
        stats = self.stats.copy()
        stats["sizes"] = {}
        
        for name, cache in self.caches.items():
            stats["sizes"][name] = len(cache)
        
        return stats

    def clear_stats(self):
        """清空统计信息"""
        self.stats = {
            "hits": 0,
            "misses": 0,
            "evictions": 0,
            "sets": 0
        }
        self.logger.info("清空缓存统计信息")

    def shutdown(self):
        """关闭缓存管理器"""
        self.clear()
        self.logger.info("缓存管理器已关闭")
