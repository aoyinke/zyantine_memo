# ============ 记忆系统通用工具 ============
import json
import hashlib
import time
from datetime import datetime
from typing import Dict, Any, Optional, List,Union
# ============ 缓存管理工具 ============
# 使用统一的 core.cache_manager，但保持向后兼容的接口
# 使用延迟导入以避免循环依赖


class CacheManager:
    """
    通用缓存管理类（适配器模式，使用 core.cache_manager）
    保持向后兼容的接口
    """
    
    def __init__(self, 
                 max_size: int = 1000, 
                 cache_expiry: int = 3600, 
                 name: str = "default_cache"):
        """
        初始化缓存管理器
        
        Args:
            max_size: 最大缓存大小
            cache_expiry: 缓存过期时间（秒）
            name: 缓存名称
        """
        self.max_size = max_size
        self.cache_expiry = cache_expiry
        self.name = name
        self.last_cleanup = time.time()
        
        # 延迟导入以避免循环依赖（使用局部变量名避免 UnboundLocalError）
        try:
            from core.cache_manager import CacheManager as _CoreCacheManager
        except (ImportError, ModuleNotFoundError) as e:
            # 如果导入失败，抛出更详细的错误信息
            raise ImportError(
                f"无法导入 core.cache_manager: {e}. "
                "请确保已安装所有依赖包。"
            ) from e
        
        # 使用统一的缓存管理器，使用 "memory" 作为缓存类型
        self._core_cache = _CoreCacheManager(
            max_size=max_size,
            default_ttl=cache_expiry,
            enable_memory_cache=True,
            enable_response_cache=False,
            enable_context_cache=False
        )
    
    def _get_cache_key(self, key: str) -> str:
        """生成缓存键"""
        return hashlib.md5(key.encode()).hexdigest()
    
    def get(self, key: str) -> Optional[Any]:
        """
        获取缓存值
        
        Args:
            key: 缓存键
            
        Returns:
            缓存值，如果不存在或过期则返回None
        """
        cache_key = self._get_cache_key(key)
        # 使用统一的缓存管理器，使用 "memory" 作为缓存类型
        return self._core_cache.get("memory", cache_key)
    
    def set(self, key: str, value: Any):
        """
        设置缓存值
        
        Args:
            key: 缓存键
            value: 缓存值
        """
        cache_key = self._get_cache_key(key)
        # 使用统一的缓存管理器，使用 "memory" 作为缓存类型
        self._core_cache.set("memory", cache_key, value, ttl=self.cache_expiry)
    
    def clear(self):
        """清空缓存"""
        self._core_cache.clear("memory")
        self.last_cleanup = time.time()
    
    def _cleanup_expired_cache(self):
        """
        清理过期缓存
        
        这个方法是为了保持向后兼容而添加的。
        委托给底层的 core.cache_manager 处理。
        """
        # 调用底层缓存管理器的清理方法
        # core.cache_manager 的 _clean_expired 方法会在 get/set 时自动调用
        # 但为了保持接口兼容，我们显式调用一次
        try:
            # 通过调用 get 方法触发自动清理（因为 get 方法内部会调用 _clean_expired）
            # 或者直接访问底层缓存来触发清理
            if hasattr(self._core_cache, '_clean_expired'):
                self._core_cache._clean_expired("memory")
        except Exception:
            # 如果清理失败，不影响主流程
            pass
        finally:
            # 更新清理时间戳
            self.last_cleanup = time.time()
    
    @property
    def cache(self) -> Dict[str, Any]:
        """
        获取缓存字典（为了向后兼容）
        
        注意：由于使用适配器模式，这里返回一个包装的字典视图
        以保持与旧代码的兼容性
        """
        # 返回底层缓存的内存缓存字典
        # core.cache_manager 使用 caches["memory"] 存储缓存
        if hasattr(self._core_cache, 'caches') and "memory" in self._core_cache.caches:
            # 将 CacheItem 对象转换为旧格式的字典
            cache_dict = {}
            for key, item in self._core_cache.caches["memory"].items():
                # 转换为旧格式：{"value": ..., "timestamp": ...}
                cache_dict[key] = {
                    "value": item.value,
                    "timestamp": item.timestamp
                }
            return cache_dict
        return {}
    
    def get_stats(self) -> Dict[str, Any]:
        """
        获取缓存统计信息
        
        Returns:
            缓存统计信息
        """
        core_stats = self._core_cache.get_stats()
        return {
            "name": self.name,
            "current_size": core_stats.get("sizes", {}).get("memory", 0),
            "max_size": self.max_size,
            "cache_expiry": self.cache_expiry,
            "last_cleanup": self.last_cleanup,
            "hits": core_stats.get("hits", 0),
            "misses": core_stats.get("misses", 0)
        }
    
    def get_all(self) -> Dict[str, Any]:
        """
        获取所有缓存项（注意：由于使用哈希键，无法完全恢复原始键）
        
        Returns:
            所有缓存项的字典
        """
        # 注意：core.cache_manager 没有 get_all 方法，这里返回空字典
        # 如果需要获取所有项，需要使用 core.cache_manager 的底层实现
        return {}

# ============ 哈希计算工具 ============
def calculate_hash(content: Union[str, List[Dict], Dict]) -> str:
    """计算内容的哈希值
    
    Args:
        content: 要计算哈希的内容
        
    Returns:
        哈希值
    """
    if isinstance(content, str):
        content_str = content.strip().lower()
    else:
        content_str = json.dumps(content, ensure_ascii=False, sort_keys=True)
    
    return hashlib.sha256(content_str.encode('utf-8')).hexdigest()

def generate_memory_id(content: str, memory_type: str) -> str:
    """生成记忆ID
    
    Args:
        content: 记忆内容
        memory_type: 记忆类型
        
    Returns:
        记忆ID
    """
    timestamp: str = datetime.now().strftime("%Y%m%d%H%M%S")
    content_hash: str = hashlib.md5(content.encode()).hexdigest()[:8]
    return f"{memory_type}_{timestamp}_{content_hash}"

# ============ 时间处理工具 ============
def get_current_timestamp() -> str:
    """获取当前时间戳，格式为ISO 8601
    
    Returns:
        当前时间戳
    """
    return datetime.now().isoformat()

def get_timestamp_from_iso(iso_str: str) -> float:
    """将ISO 8601格式的时间字符串转换为时间戳
    
    Args:
        iso_str: ISO 8601格式的时间字符串
        
    Returns:
        时间戳
    """
    return datetime.fromisoformat(iso_str).timestamp()

def calculate_time_diff(iso_str: str) -> float:
    """计算给定时间字符串与当前时间的差值（秒）
    
    Args:
        iso_str: ISO 8601格式的时间字符串
        
    Returns:
        时间差值（秒）
    """
    timestamp = get_timestamp_from_iso(iso_str)
    return time.time() - timestamp

def is_time_expired(iso_str: str, expiry_seconds: int) -> bool:
    """检查给定时间字符串是否已过期
    
    Args:
        iso_str: ISO 8601格式的时间字符串
        expiry_seconds: 过期时间（秒）
        
    Returns:
        是否已过期
    """
    return calculate_time_diff(iso_str) > expiry_seconds

def calculate_age_hours(iso_str: str) -> float:
    """计算给定时间字符串到现在的小时数
    
    Args:
        iso_str: ISO 8601格式的时间字符串
        
    Returns:
        小时数
    """
    return calculate_time_diff(iso_str) / 3600

# ============ JSON转换工具 ============
def to_json_str(obj: Any, ensure_ascii: bool = False) -> str:
    """将对象转换为JSON字符串
    
    Args:
        obj: 要转换的对象
        ensure_ascii: 是否确保ASCII编码
        
    Returns:
        JSON字符串
    """
    return json.dumps(obj, ensure_ascii=ensure_ascii)

def from_json_str(json_str: str) -> Any:
    """将JSON字符串转换为对象
    
    Args:
        json_str: JSON字符串
        
    Returns:
        转换后的对象
    """
    return json.loads(json_str)

def get_content_size(content: Any) -> int:
    """获取内容的大小（字节）
    
    Args:
        content: 内容
        
    Returns:
        字节大小
    """
    if isinstance(content, (str, bytes)):
        return len(content)
    return len(to_json_str(content).encode('utf-8'))

# ============ 配置管理工具 ============
def safe_get_config(config: Any, 
                   path: str, 
                   default: Any = None, 
                   separator: str = ".") -> Any:
    """
    安全地从配置对象中获取值，支持点分隔的路径
    
    Args:
        config: 配置对象
        path: 点分隔的路径，如"memory.memo0_config"
        default: 默认值
        separator: 路径分隔符
        
    Returns:
        配置值或默认值
    """
    keys = path.split(separator)
    current = config
    for key in keys:
        if isinstance(current, dict):
            current = current.get(key, None)
        elif hasattr(current, key):
            current = getattr(current, key, None)
        else:
            return default
        
        if current is None:
            return default
    
    return current

# ============ 字符串处理工具 ============
def truncate_string(content: str, max_length: int = 200, suffix: str = "...") -> str:
    """
    截断字符串
    
    Args:
        content: 要截断的字符串
        max_length: 最大长度
        suffix: 后缀
        
    Returns:
        截断后的字符串
    """
    if len(content) <= max_length:
        return content
    return content[:max_length - len(suffix)] + suffix

def extract_keywords(content: str, max_keywords: int = 5) -> List[str]:
    """
    提取关键词
    
    Args:
        content: 内容
        max_keywords: 最大关键词数量
        
    Returns:
        关键词列表
    """
    if not content:
        return []
    
    # 简单的关键词提取，实际项目中可以使用更复杂的算法
    words = content.split()
    # 去重并限制数量
    unique_words = list(dict.fromkeys(words))
    return unique_words[:max_keywords]

# ============ 统计工具 ============
class StatisticsManager:
    """统计管理器"""
    
    def __init__(self, name: str = "default_stats"):
        """
        初始化统计管理器
        
        Args:
            name: 统计名称
        """
        self.name = name
        self.stats: Dict[str, Any] = {
            "total_count": 0,
            "by_type": {},
            "tags_distribution": {},
            "topics_distribution": {},
            "access_counts": {},
            "start_time": get_current_timestamp()
        }
    
    def increment(self, key: str, value: int = 1):
        """
        增加统计计数
        
        Args:
            key: 统计键
            value: 增加的值
        """
        if key in self.stats:
            if isinstance(self.stats[key], int):
                self.stats[key] += value
        else:
            self.stats[key] = value
    
    def update_by_type(self, memory_type: str):
        """
        更新按类型统计
        
        Args:
            memory_type: 记忆类型
        """
        self.stats["by_type"][memory_type] = self.stats["by_type"].get(memory_type, 0) + 1
    
    def update_tags(self, tags: List[str]):
        """
        更新标签分布统计
        
        Args:
            tags: 标签列表
        """
        for tag in tags:
            self.stats["tags_distribution"][tag] = self.stats["tags_distribution"].get(tag, 0) + 1
    
    def update_topics(self, topics: List[str]):
        """
        更新主题分布统计
        
        Args:
            topics: 主题列表
        """
        for topic in topics:
            self.stats["topics_distribution"][topic] = self.stats["topics_distribution"].get(topic, 0) + 1
    
    def update_access_count(self, memory_id: str):
        """
        更新访问计数
        
        Args:
            memory_id: 记忆ID
        """
        self.stats["access_counts"][memory_id] = self.stats["access_counts"].get(memory_id, 0) + 1
    
    def get_stats(self) -> Dict[str, Any]:
        """
        获取统计信息
        
        Returns:
            统计信息
        """
        return self.stats.copy()
    
    def reset(self):
        """
        重置统计信息
        """
        self.stats = {
            "total_count": 0,
            "by_type": {},
            "tags_distribution": {},
            "topics_distribution": {},
            "access_counts": {},
            "start_time": get_current_timestamp()
        }
