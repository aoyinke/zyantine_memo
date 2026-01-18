# ============ 存储抽象基类 ============
"""
定义存储层的抽象接口，所有存储实现都应该遵循这些接口。
"""
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Generic, TypeVar

T = TypeVar('T')


class StorageInterface(ABC, Generic[T]):
    """
    存储接口抽象基类
    
    定义所有存储实现必须提供的基本操作。
    """
    
    @abstractmethod
    def get(self, key: str) -> Optional[T]:
        """
        获取存储项
        
        Args:
            key: 存储键
            
        Returns:
            存储的值，如果不存在则返回 None
        """
        pass
    
    @abstractmethod
    def set(self, key: str, value: T) -> None:
        """
        设置存储项
        
        Args:
            key: 存储键
            value: 要存储的值
        """
        pass
    
    @abstractmethod
    def delete(self, key: str) -> bool:
        """
        删除存储项
        
        Args:
            key: 存储键
            
        Returns:
            是否成功删除
        """
        pass
    
    @abstractmethod
    def exists(self, key: str) -> bool:
        """
        检查存储项是否存在
        
        Args:
            key: 存储键
            
        Returns:
            是否存在
        """
        pass
    
    @abstractmethod
    def clear(self) -> None:
        """清空所有存储项"""
        pass
    
    @abstractmethod
    def get_all(self) -> Dict[str, T]:
        """
        获取所有存储项
        
        Returns:
            所有存储项的字典
        """
        pass
    
    @abstractmethod
    def size(self) -> int:
        """
        获取存储项数量
        
        Returns:
            存储项数量
        """
        pass
    
    @abstractmethod
    def get_stats(self) -> Dict[str, Any]:
        """
        获取存储统计信息
        
        Returns:
            统计信息字典
        """
        pass


class BaseStorage(StorageInterface[T]):
    """
    基础存储实现
    
    提供简单的内存存储实现，可作为其他存储的基类。
    """
    
    def __init__(self, max_size: int = 10000):
        """
        初始化存储
        
        Args:
            max_size: 最大存储容量
        """
        self._storage: Dict[str, T] = {}
        self._max_size = max_size
        self._access_count: Dict[str, int] = {}
        self._total_gets = 0
        self._total_sets = 0
        self._total_deletes = 0
    
    def get(self, key: str) -> Optional[T]:
        """获取存储项"""
        self._total_gets += 1
        if key in self._storage:
            self._access_count[key] = self._access_count.get(key, 0) + 1
            return self._storage[key]
        return None
    
    def set(self, key: str, value: T) -> None:
        """设置存储项"""
        self._total_sets += 1
        
        # 如果存储已满且是新键，需要淘汰
        if len(self._storage) >= self._max_size and key not in self._storage:
            self._evict_one()
        
        self._storage[key] = value
        self._access_count[key] = self._access_count.get(key, 0)
    
    def delete(self, key: str) -> bool:
        """删除存储项"""
        self._total_deletes += 1
        if key in self._storage:
            del self._storage[key]
            self._access_count.pop(key, None)
            return True
        return False
    
    def exists(self, key: str) -> bool:
        """检查存储项是否存在"""
        return key in self._storage
    
    def clear(self) -> None:
        """清空所有存储项"""
        self._storage.clear()
        self._access_count.clear()
    
    def get_all(self) -> Dict[str, T]:
        """获取所有存储项"""
        return self._storage.copy()
    
    def size(self) -> int:
        """获取存储项数量"""
        return len(self._storage)
    
    def get_stats(self) -> Dict[str, Any]:
        """获取存储统计信息"""
        return {
            "size": len(self._storage),
            "max_size": self._max_size,
            "utilization": len(self._storage) / self._max_size if self._max_size > 0 else 0,
            "total_gets": self._total_gets,
            "total_sets": self._total_sets,
            "total_deletes": self._total_deletes,
        }
    
    def _evict_one(self) -> None:
        """
        淘汰一个存储项（LFU策略）
        
        可以被子类覆盖以实现不同的淘汰策略。
        """
        if not self._storage:
            return
        
        # 找到访问次数最少的项
        min_key = min(self._access_count.keys(), key=lambda k: self._access_count.get(k, 0))
        self.delete(min_key)
    
    def get_keys(self) -> List[str]:
        """获取所有键"""
        return list(self._storage.keys())
    
    def get_access_count(self, key: str) -> int:
        """获取指定键的访问次数"""
        return self._access_count.get(key, 0)


class PersistentStorageInterface(StorageInterface[T]):
    """
    持久化存储接口
    
    扩展基础存储接口，添加持久化相关的方法。
    """
    
    @abstractmethod
    def save(self) -> bool:
        """
        保存到持久化存储
        
        Returns:
            是否成功保存
        """
        pass
    
    @abstractmethod
    def load(self) -> bool:
        """
        从持久化存储加载
        
        Returns:
            是否成功加载
        """
        pass
    
    @abstractmethod
    def get_storage_path(self) -> str:
        """
        获取存储路径
        
        Returns:
            存储路径
        """
        pass
