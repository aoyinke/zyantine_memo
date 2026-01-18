# ============ 基于memo0的简化记忆系统 ============
"""
Zyantine 记忆存储系统

重构说明：
- ShortTermMemory 数据模型已移至 models.py
- 本文件保留 ZyantineMemorySystem 核心类
"""
import os
import json
import hashlib
import time
import traceback
from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Any, Optional, Union, Tuple, Set

# Mem0框架导入
from mem0 import Memory

# ============ 常量定义 ============
DEFAULT_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://openkey.cloud/v1")
DEFAULT_API_KEY = os.getenv("OPENAI_API_KEY", "")

# 自定义异常类导入
from .memory_exceptions import (
    MemoryException,
    MemoryStorageError,
    MemoryRetrievalError,
    MemorySearchError,
    MemoryValidationError,
    MemoryNotFoundError,
    MemoryCacheError,
    MemoryIndexError,
    MemorySizeExceededError,
    MemoryMetadataError,
    MemoryTypeError,
    MemorySerializationError,
    MemoryDeserializationError,
    MemoryConfigError,
    handle_memory_error
)

# 导入数据模型
from .models import ShortTermMemory

# 导入记忆评估器
from .memory_evaluator import MemoryEvaluator, MemoryEvaluationResult

# 导入通用工具函数
from .memory_utils import (
    CacheManager,
    calculate_hash,
    generate_memory_id,
    get_current_timestamp,
    calculate_age_hours,
    is_time_expired,
    calculate_time_diff,
    to_json_str,
    get_content_size,
    StatisticsManager,
    safe_get_config
)

# ============ 记忆系统核心类 ============
class ZyantineMemorySystem:
    """基于memo0的自衍体记忆系统"""

    def __init__(self,
                 base_url: Optional[str] = None,
                 api_key: Optional[str] = None,
                 user_id: str = "default",
                 session_id: str = "default",
                 memo0_config: Optional[Dict[str, Any]] = None) -> None:
        """初始化记忆系统"""
        # 从配置管理器获取配置
        from config.config_manager import ConfigManager
        config_manager = ConfigManager()
        config = config_manager.get()

        # 使用传入的配置或从配置文件读取
        if memo0_config:
            self.memo0_config = memo0_config
        else:
            self.memo0_config = config.memory.memo0_config

        # 从配置中获取 LLM 和 Embedder 配置
        llm_config = self.memo0_config.get("llm", {}).get("config", {})
        embedder_config = self.memo0_config.get("embedder", {}).get("config", {})

        # 优先使用传入参数，否则使用配置文件中的值
        self.base_url: str = base_url or llm_config.get("base_url", DEFAULT_BASE_URL)
        self.api_key: str = api_key or llm_config.get("api_key", DEFAULT_API_KEY)
        self.user_id: str = user_id
        self.session_id: str = session_id

        # 初始化智能记忆评估器（优化：减少API调用，使用规则引擎）
        self.memory_evaluator = MemoryEvaluator(
            llm_client=None,  # 不使用LLM评估
            enable_llm_evaluation=False,  # 默认关闭LLM评估，使用规则引擎
            conservative_mode=True  # 保守模式：宁可多存储
        )

        # 初始化memo0记忆系统，添加异常处理
        self.memory: Optional[Memory] = None
        self.memory_store: List[Dict[str, Any]] = []
        self._persistent_storage_path: Optional[str] = None
        
        try:
            self.memory = self._initialize_memo0()
            # 验证memo0是否正常工作
            if self._verify_memo0_connection():
                print("[记忆系统] Mem0记忆系统初始化成功并验证通过")
            else:
                print("[记忆系统] Mem0记忆系统初始化成功但验证失败，启用文件持久化回退")
                self._setup_persistent_storage()
        except Exception as e:
            print(f"[记忆系统] Mem0记忆系统初始化失败: {e}")
            print(f"[记忆系统] 错误详情:\n{traceback.format_exc()}")
            print("[记忆系统] 警告：将使用简化的内存记忆系统作为替代，并启用文件持久化")
            # 创建一个简单的内存记忆系统作为替代
            self.memory = None
            self._setup_persistent_storage()

        self.semantic_memory_map: Dict[str, Dict[str, Any]] = {}  # 增强：语义记忆地图
        self.strategic_tags: List[str] = []
        
        # 新增：增强索引结构
        self.topic_index: Dict[str, List[str]] = defaultdict(list)  # 主题 -> 记忆ID列表
        self.memory_topics: Dict[str, List[str]] = defaultdict(list)  # 记忆ID -> 主题列表
        self.key_info_index: Dict[str, List[str]] = defaultdict(list)  # 关键信息 -> 记忆ID列表
        self.memory_key_info: Dict[str, List[str]] = defaultdict(list)  # 记忆ID -> 关键信息列表
        self.importance_index: Dict[int, List[str]] = defaultdict(list)  # 重要性分数 -> 记忆ID列表
        self.type_index: Dict[str, List[str]] = defaultdict(list)  # 记忆类型 -> 记忆ID列表
        
        # 主题关键词库
        self.topic_keywords = {
            "fitness": ["体测", "体检", "健身", "运动", "锻炼", "测试", "成绩", "跑步", "跳远", "肺活量"],
            "work": ["工作", "上班", "任务", "项目", "会议", "报告", "邮件", "客户", "同事"],
            "study": ["学习", "学校", "考试", "作业", "课程", "复习", "考试", "成绩", "论文"],
            "life": ["生活", "日常", "周末", "假期", "旅行", "美食", "电影", "音乐"]
        }
        
        # 添加查询增强缓存
        self.query_enhancement_cache = CacheManager(max_size=1000, cache_expiry=3600)  # 使用统一的缓存管理器
        
        # 语义记忆地图配置
        self.max_semantic_map_size: int = 5000  # 最大语义记忆地图大小
        self.semantic_map_cleanup_threshold: float = 0.3  # 清理阈值（低于此值的记忆会被清理）
        
        # 新增：上下文窗口管理
        self.context_window: List[str] = []  # 上下文窗口，存储当前对话相关的记忆ID
        self.max_context_window_size: int = 20  # 默认最大上下文窗口大小
        self.min_context_window_size: int = 5  # 默认最小上下文窗口大小
        self.current_context_window_size: int = 10  # 当前上下文窗口大小
        self.context_window_dynamic_weight: float = 0.8  # 上下文窗口动态调整权重
        
        # 新增：当前对话主题追踪
        self.current_topics: Dict[str, float] = {}  # 当前对话主题及权重
        self.topic_decay_rate: float = 0.9  # 主题权重衰减率
        
        # 新增：短期记忆标记
        self.short_term_memories: Set[str] = set()  # 短期记忆ID集合
        self.short_term_memory_duration: int = 3600  # 短期记忆持续时间（秒）

        # 短期记忆配置
        self.short_term_store: Dict[str, ShortTermMemory] = {}  # 短期记忆存储（内存）
        self.conversation_index: Dict[str, List[str]] = defaultdict(list)  # 对话ID -> 记忆ID列表
        self.short_term_ttl: int = 3600  # 短期记忆默认生存时间（秒）
        self.short_term_max_size: int = 1000  # 短期记忆最大容量
        self.last_cleanup_time: float = time.time()  # 上次清理时间
        
        # 记忆分级配置
        self.memory_tiers = {
            "hot": {"max_size": 500, "min_access_count": 10, "max_age_hours": 24},  # 热数据：频繁访问，最近24小时
            "warm": {"max_size": 2000, "min_access_count": 3, "max_age_hours": 168},  # 温数据：定期访问，最近7天
            "cold": {"max_size": 10000, "min_access_count": 0, "max_age_hours": 720}  # 冷数据：很少访问，最近30天
        }
        
        # 记忆分级存储
        self.memory_tier_store: Dict[str, Dict[str, Dict]] = {
            "hot": {},
            "warm": {},
            "cold": {}
        }
        
        # 记忆到分级的映射
        self.memory_to_tier: Dict[str, str] = {}  # 记忆ID -> 存储级别
        
        # 记忆分级检查定时器
        self.last_tier_check_time: float = time.time()
        self.tier_check_interval: int = 3600  # 每小时检查一次记忆分级

        # 统计信息
        self.stats: Dict[str, Any] = {
            "total_memories": 0,
            "by_type": {},
            "access_counts": {},
            "tags_distribution": {},
            "topics_distribution": defaultdict(int)  # 新增：主题分布统计
        }

        print(f"[记忆系统] 初始化完成，用户ID: {user_id}，会话ID: {session_id}")
        print(f"[记忆系统] LLM配置: provider={self.memo0_config.get('llm', {}).get('provider')}, model={llm_config.get('model')}")
        print(f"[记忆系统] Embedder配置: provider={self.memo0_config.get('embedder', {}).get('provider')}, model={embedder_config.get('model')}")
        print("[记忆系统] LLM客户端初始化完成")
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取记忆系统统计信息
        
        Returns:
            Dict[str, Any]: 包含记忆系统统计信息的字典
        """
        try:
            # 基本统计信息
            stats = {
                "user_id": self.user_id,
                "session_id": self.session_id,
                "total_memories": self.stats["total_memories"],
                "memory_types": self.stats["by_type"],
                "system_status": "active",
                "initialized_at": getattr(self, "initialized_at", datetime.now().isoformat()),
                "last_updated": datetime.now().isoformat(),
                "short_term_memory_stats": {
                    "current_size": len(self.short_term_store),
                    "max_size": self.short_term_max_size,
                    "ttl_seconds": self.short_term_ttl,
                    "last_cleanup": self.last_cleanup_time
                },
                "memory_tier_stats": {
                    "hot": {
                        "current_size": len(self.memory_tier_store["hot"]),
                        "max_size": self.memory_tiers["hot"]["max_size"]
                    },
                    "warm": {
                        "current_size": len(self.memory_tier_store["warm"]),
                        "max_size": self.memory_tiers["warm"]["max_size"]
                    },
                    "cold": {
                        "current_size": len(self.memory_tier_store["cold"]),
                        "max_size": self.memory_tiers["cold"]["max_size"]
                    },
                    "last_tier_check": self.last_tier_check_time
                }
            }
            
            # 如果memo0实例存在，尝试获取更详细的统计信息
            if hasattr(self, "memory") and self.memory:
                try:
                    # 调用memo0的stats方法（如果可用）
                    if hasattr(self.memory, "stats"):
                        memo0_stats = self.memory.stats()
                        stats["total_memories"] = memo0_stats.get("total_memories", self.stats["total_memories"])
                        stats["memory_types"] = memo0_stats.get("memory_types", self.stats["by_type"])
                except Exception as e:
                    # 如果获取memo0统计信息失败，使用默认值
                    print(f"[记忆系统] 获取memo0统计信息失败: {e}")
            
            return stats
        except Exception as e:
            print(f"[记忆系统] 获取统计信息失败: {e}")
            # 返回基本统计信息
            return {
                "user_id": self.user_id,
                "session_id": self.session_id,
                "total_memories": 0,
                "memory_types": {},
                "system_status": "error",
                "error": str(e),
                "last_updated": datetime.now().isoformat(),
                "short_term_memory_stats": {
                    "current_size": len(self.short_term_store),
                    "max_size": self.short_term_max_size,
                    "ttl_seconds": self.short_term_ttl,
                    "last_cleanup": self.last_cleanup_time
                },
                "memory_tier_stats": {
                    "hot": {
                        "current_size": len(self.memory_tier_store["hot"]),
                        "max_size": self.memory_tiers["hot"]["max_size"]
                    },
                    "warm": {
                        "current_size": len(self.memory_tier_store["warm"]),
                        "max_size": self.memory_tiers["warm"]["max_size"]
                    },
                    "cold": {
                        "current_size": len(self.memory_tier_store["cold"]),
                        "max_size": self.memory_tiers["cold"]["max_size"]
                    },
                    "last_tier_check": self.last_tier_check_time
                }
            }
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息（别名方法，用于兼容）"""
        return self.get_statistics()
    
    def test_connection(self) -> bool:
        """测试记忆系统连接"""
        try:
            # 简单测试，检查memory实例是否存在且可用
            return hasattr(self, "memory") and self.memory is not None
        except Exception as e:
            print(f"[记忆系统] 连接测试失败: {e}")
            return False
    
    def export_memories(self, file_path: str, format_type: str = "json") -> bool:
        """导出记忆"""
        try:
            # 尝试使用memory的export方法（如果可用）
            if hasattr(self, "memory") and self.memory:
                if hasattr(self.memory, "export"):
                    self.memory.export(file_path, format_type)
                    return True
            
            # 如果没有memory导出方法，创建一个简单的导出文件
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump({
                    "exported_at": datetime.now().isoformat(), 
                    "user_id": self.user_id,
                    "short_term_memories": [stm.to_dict() for stm in self.short_term_store.values()]
                }, f, ensure_ascii=False, indent=2)
            return True
        except Exception as e:
            print(f"[记忆系统] 导出记忆失败: {e}")
            return False

    # ============ 短期记忆管理方法 ============
    def add_short_term_memory(self, 
                            content: str, 
                            conversation_id: str,
                            metadata: Optional[Dict[str, Any]] = None,
                            ttl: int = None) -> str:
        """
        添加短期记忆
        
        Args:
            content: 记忆内容
            conversation_id: 对话ID
            metadata: 元数据
            ttl: 生存时间（秒），默认使用系统配置
            
        Returns:
            str: 记忆ID
        """
        # 定期清理过期记忆
        self._cleanup_short_term_memory()
        
        # 生成记忆ID
        memory_id = generate_memory_id(content, "short_term")
        
        # 使用默认TTL如果未提供
        if ttl is None:
            ttl = self.short_term_ttl
        
        # 创建短期记忆对象
        stm = ShortTermMemory(
            memory_id=memory_id,
            content=content,
            conversation_id=conversation_id,
            metadata=metadata or {},
            ttl=ttl
        )
        
        # 存储到短期记忆
        self.short_term_store[memory_id] = stm
        
        # 更新对话索引
        self.conversation_index[conversation_id].append(memory_id)
        
        # 如果超过最大容量，删除最旧的记忆
        if len(self.short_term_store) > self.short_term_max_size:
            self._evict_oldest_short_term_memory()
        
        return memory_id
    
    def get_short_term_memory(self, memory_id: str) -> Optional[ShortTermMemory]:
        """
        获取短期记忆
        
        Args:
            memory_id: 记忆ID
            
        Returns:
            Optional[ShortTermMemory]: 短期记忆对象，如果不存在或过期则返回None
        """
        # 检查记忆是否存在
        if memory_id not in self.short_term_store:
            return None
        
        # 获取记忆
        stm = self.short_term_store[memory_id]
        
        # 检查是否过期
        if stm.is_expired():
            # 删除过期记忆
            self._delete_short_term_memory(memory_id)
            return None
        
        return stm
    
    def search_short_term_memories(self, 
                                conversation_id: str, 
                                limit: int = 10) -> List[ShortTermMemory]:
        """
        搜索短期记忆
        
        Args:
            conversation_id: 对话ID
            limit: 返回数量限制
            
        Returns:
            List[ShortTermMemory]: 短期记忆列表
        """
        # 定期清理过期记忆
        self._cleanup_short_term_memory()
        
        # 获取对话相关的记忆ID
        memory_ids = self.conversation_index.get(conversation_id, [])
        
        # 过滤过期记忆并排序
        results = []
        for memory_id in memory_ids:
            stm = self.get_short_term_memory(memory_id)
            if stm:
                results.append(stm)
        
        # 按创建时间倒序排序
        results.sort(key=lambda x: x.created_at, reverse=True)
        
        # 限制返回数量
        return results[:limit]
    
    def _cleanup_short_term_memory(self, force: bool = False):
        """清理过期的短期记忆
        
        Args:
            force: 是否强制清理，忽略时间间隔
        """
        current_time = time.time()
        
        # 每30秒清理一次，除非强制清理
        if not force and current_time - self.last_cleanup_time < 30:
            return
        
        # 找出过期的记忆
        expired_memory_ids = []
        for memory_id, stm in self.short_term_store.items():
            if stm.is_expired():
                expired_memory_ids.append(memory_id)
        
        # 删除过期记忆
        for memory_id in expired_memory_ids:
            self._delete_short_term_memory(memory_id)
        
        # 更新最后清理时间
        self.last_cleanup_time = current_time
    
    def _delete_short_term_memory(self, memory_id: str):
        """删除短期记忆"""
        if memory_id in self.short_term_store:
            # 获取对话ID
            conversation_id = self.short_term_store[memory_id].conversation_id
            
            # 删除记忆
            del self.short_term_store[memory_id]
            
            # 从对话索引中删除
            if conversation_id in self.conversation_index:
                if memory_id in self.conversation_index[conversation_id]:
                    self.conversation_index[conversation_id].remove(memory_id)
                
                # 如果对话索引为空，删除该对话索引
                if not self.conversation_index[conversation_id]:
                    del self.conversation_index[conversation_id]
    
    def _evict_oldest_short_term_memory(self):
        """删除最旧的短期记忆"""
        if not self.short_term_store:
            return
        
        # 找到最旧的记忆
        oldest_stm = min(self.short_term_store.values(), key=lambda x: x.created_at)
        
        # 删除最旧的记忆
        self._delete_short_term_memory(oldest_stm.memory_id)
    
    def _determine_memory_tier(self, memory_id: str) -> str:
        """确定记忆的存储级别
        
        Args:
            memory_id: 记忆ID
            
        Returns:
            str: 存储级别（hot, warm, cold）
        """
        if memory_id not in self.semantic_memory_map:
            return "cold"
        
        memory_data = self.semantic_memory_map[memory_id]
        metadata = memory_data["metadata"]
        
        # 计算记忆年龄（小时）
        created_at = metadata.get("full_metadata", {}).get("created_at")
        if not created_at:
            return "cold"
        
        age_hours = calculate_age_hours(created_at)
        access_count = memory_data.get("access_count", 0)
        
        # 根据访问频率和年龄确定存储级别
        if access_count >= self.memory_tiers["hot"]["min_access_count"] and age_hours <= self.memory_tiers["hot"]["max_age_hours"]:
            return "hot"
        elif access_count >= self.memory_tiers["warm"]["min_access_count"] and age_hours <= self.memory_tiers["warm"]["max_age_hours"]:
            return "warm"
        else:
            return "cold"
    
    def _migrate_memory_tier(self, memory_id: str, target_tier: str):
        """迁移记忆到目标存储级别
        
        Args:
            memory_id: 记忆ID
            target_tier: 目标存储级别
        """
        # 获取当前存储级别
        current_tier = self.memory_to_tier.get(memory_id, "cold")
        
        # 如果已经在目标级别，无需迁移
        if current_tier == target_tier:
            return
        
        # 从当前级别移除
        if current_tier in self.memory_tier_store and memory_id in self.memory_tier_store[current_tier]:
            del self.memory_tier_store[current_tier][memory_id]
        
        # 添加到目标级别
        if memory_id in self.semantic_memory_map:
            self.memory_tier_store[target_tier][memory_id] = self.semantic_memory_map[memory_id]
            self.memory_to_tier[memory_id] = target_tier
    
    def _check_and_update_memory_tiers(self):
        """检查并更新所有记忆的存储级别"""
        current_time = time.time()
        
        # 只有当距离上次检查超过指定时间间隔时才执行
        if current_time - self.last_tier_check_time < self.tier_check_interval:
            return
        
        print("[记忆系统] 开始检查和更新记忆分级...")
        
        # 更新所有记忆的存储级别
        for memory_id in list(self.semantic_memory_map.keys()):
            target_tier = self._determine_memory_tier(memory_id)
            self._migrate_memory_tier(memory_id, target_tier)
        
        # 清理各个级别的超出容量的记忆
        for tier, tier_config in self.memory_tiers.items():
            tier_store = self.memory_tier_store[tier]
            max_size = tier_config["max_size"]
            
            if len(tier_store) > max_size:
                # 按访问时间排序，删除最旧的记忆
                sorted_memories = sorted(tier_store.items(), key=lambda x: x[1].get("last_accessed", 0))
                excess = len(tier_store) - max_size
                
                for memory_id, _ in sorted_memories[:excess]:
                    if memory_id in tier_store:
                        del tier_store[memory_id]
                        if memory_id in self.memory_to_tier:
                            del self.memory_to_tier[memory_id]
        
        # 更新最后检查时间
        self.last_tier_check_time = current_time
        print("[记忆系统] 记忆分级检查和更新完成")
    
    def _update_tier_store(self, memory_id: str):
        """更新记忆分级存储
        
        Args:
            memory_id: 记忆ID
        """
        # 确定记忆的存储级别
        target_tier = self._determine_memory_tier(memory_id)
        
        # 迁移记忆到目标级别
        self._migrate_memory_tier(memory_id, target_tier)
    
    def _initialize_memo0(self) -> Memory:
        """初始化memo0框架"""
        # 使用从配置文件读取的配置
        config: Dict[str, Any] = self.memo0_config.copy()

        # 验证和设置API密钥
        # 优先使用传入的api_key，否则从环境变量获取
        api_key_to_use = self.api_key
        if not api_key_to_use:
            api_key_to_use = os.getenv("OPENAI_API_KEY", "") or os.getenv("ZHIPU_API_KEY", "")
            if not api_key_to_use:
                print("[记忆系统] 警告：未提供API密钥，将使用虚拟密钥进行初始化")
                # 使用虚拟密钥通过验证，但实际调用时会失败
                # 不过这允许系统初始化，后续功能可能会受限
                api_key_to_use = "dummy-api-key"

        # 确保配置格式正确
        if "llm" not in config:
            config["llm"] = {
                "provider": "openai",
                "config": {
                    "model": "gpt-5-nano-2025-08-07",
                    "api_key": api_key_to_use,
                    "openai_base_url": self.base_url
                }
            }
        else:
            # 确保 LLM 配置中有必要的字段
            llm_config = config["llm"]["config"]
            if "api_key" not in llm_config:
                llm_config["api_key"] = api_key_to_use
            if "openai_base_url" not in llm_config:
                llm_config["openai_base_url"] = self.base_url

        if "embedder" not in config:
            config["embedder"] = {
                "provider": "openai",
                "config": {
                    "model": "text-embedding-3-large",
                    "api_key": api_key_to_use,
                    "openai_base_url": self.base_url
                }
            }
        else:
            # 确保 Embedder 配置中有必要的字段
            embedder_config = config["embedder"]["config"]
            if "api_key" not in embedder_config:
                embedder_config["api_key"] = api_key_to_use
            if "openai_base_url" not in embedder_config:
                embedder_config["openai_base_url"] = self.base_url

        return Memory.from_config(config)
    
    def _verify_memo0_connection(self) -> bool:
        """
        验证memo0连接是否正常工作
        
        Returns:
            是否正常工作
        """
        if not self.memory:
            return False
        
        try:
            # 尝试一个简单的搜索操作来验证连接
            # 根据mem0 API，第一个参数是query（位置参数），后面必须使用命名参数
            test_results = self.memory.search(
                "test",
                user_id=self.user_id
            )
            # 如果能够成功调用search方法，说明连接正常
            return True
        except Exception as e:
            print(f"[记忆系统] Mem0连接验证失败: {e}")
            return False
    
    def _setup_persistent_storage(self):
        """设置文件系统持久化存储作为回退方案"""
        try:
            import os
            from pathlib import Path
            
            # 创建持久化存储目录
            storage_dir = Path("./memory_storage")
            storage_dir.mkdir(exist_ok=True)
            
            # 为当前用户创建存储文件
            storage_file = storage_dir / f"{self.user_id}_memories.jsonl"
            self._persistent_storage_path = str(storage_file)
            
            # 加载已有记忆（如果存在）
            if storage_file.exists():
                try:
                    with open(storage_file, 'r', encoding='utf-8') as f:
                        for line in f:
                            if line.strip():
                                memory_item = json.loads(line)
                                self.memory_store.append(memory_item)
                    print(f"[记忆系统] 从文件加载了 {len(self.memory_store)} 条记忆")
                except Exception as e:
                    print(f"[记忆系统] 加载持久化记忆失败: {e}")
            
            print(f"[记忆系统] 文件持久化存储已设置: {self._persistent_storage_path}")
        except Exception as e:
            print(f"[记忆系统] 设置文件持久化存储失败: {e}")
            self._persistent_storage_path = None
    
    def _save_to_persistent_storage(self, memory_item: Dict[str, Any]):
        """保存记忆到文件系统持久化存储"""
        if not self._persistent_storage_path:
            return
        
        try:
            import os
            # 使用追加模式写入JSONL格式
            with open(self._persistent_storage_path, 'a', encoding='utf-8') as f:
                json.dump(memory_item, f, ensure_ascii=False)
                f.write('\n')
        except Exception as e:
            print(f"[记忆系统] 保存到持久化存储失败: {e}")
    
    def _verify_memory_stored(self, memory_id: str, memory_type: str, max_retries: int = 2) -> bool:
        """
        验证记忆是否成功存储到memo0
        
        Args:
            memory_id: 记忆ID
            memory_type: 记忆类型
            max_retries: 最大重试次数
            
        Returns:
            是否存储成功
        """
        if not self.memory:
            return False
        
        # 对于新添加的记忆，可能无法立即检索到，所以这里只做基本验证
        # 如果memo0没有抛出异常，认为存储成功
        return True

    # ============ 记忆CRUD操作 ============

    def add_memory(self,
                   content: Union[str, List[Dict[str, Any]]],
                   memory_type: str = "conversation",
                   metadata: Optional[Dict[str, Any]] = None,
                   tags: Optional[List[str]] = None,
                   emotional_intensity: float = 0.5,
                   strategic_value: Optional[Dict[str, Any]] = None,
                   linked_tool: Optional[str] = None) -> str:
        """
        添加记忆（使用智能评估器）
        
        优化说明：
        1. 使用多维度规则引擎评估，减少API调用
        2. 保守策略：宁可多存储，不轻易过滤
        3. 提取更丰富的特征用于后续检索
        """
        # 验证记忆类型
        valid_memory_types: List[str] = [
            "conversation", "experience", "user_profile", "system_event",
            "knowledge", "emotion", "strategy", "temporal"
        ]
        if memory_type not in valid_memory_types:
            raise MemoryTypeError(
                f"无效的记忆类型: {memory_type}",
                details={"valid_types": valid_memory_types}
            )

        content_str: str = content if isinstance(content, str) else json.dumps(content, ensure_ascii=False)
        
        # ============ 使用智能评估器评估记忆价值 ============
        evaluation_result = self.memory_evaluator.evaluate(
            content=content_str,
            memory_type=memory_type,
            metadata=metadata,
            context={
                "current_topics": self.current_topics,
                "session_id": self.session_id,
                "user_id": self.user_id
            }
        )
        
        # 获取评估结果
        content_value = evaluation_result.overall_score
        importance_score = content_value  # 使用综合分数作为重要性分数
        
        # 根据评估结果决定是否存储
        if not evaluation_result.should_store:
            print(f"[记忆系统] 过滤低价值内容: {content_str[:50]}... (分数: {content_value:.2f}, 原因: {evaluation_result.reason})")
            return "filtered_content"
        
        # 检查重复内容
        duplicate_id = self._check_duplicate_content(content_str, memory_type)
        if duplicate_id:
            print(f"[记忆系统] 检测到重复内容，更新现有记忆: {duplicate_id}")
            # 更新现有记忆的主题
            self._update_memory_topics(duplicate_id, content_str)
            # 这里可以实现更新逻辑，暂时返回重复ID
            return duplicate_id
        
        # 生成记忆ID
        memory_id: str = self._generate_memory_id(content_str, memory_type)

        full_content: Union[str, List[Dict[str, Any]]] = content if isinstance(content, list) else [{"role": "user", "content": content}]

        # 新增：识别对话主题
        detected_topics = self._detect_topics(content_str)
        
        # ============ 关键信息提取：增强memo0能力 ============
        extracted_key_info = self._extract_key_information(content_str, memory_type)
        
        # 合并评估器提取的实体和意图
        if evaluation_result.extracted_entities:
            extracted_key_info = list(set(extracted_key_info + evaluation_result.extracted_entities))[:15]
        
        # 准备精简的metadata
        memory_metadata: Dict[str, Any] = self._prepare_metadata(
            memory_id=memory_id,
            memory_type=memory_type,
            content=content_str,
            metadata=metadata or {},
            tags=tags or [],
            emotional_intensity=emotional_intensity,
            strategic_value=strategic_value or {},
            linked_tool=linked_tool
        )
        
        # 新增：将主题添加到metadata
        if detected_topics:
            memory_metadata["topics"] = detected_topics
            # 合并主题到标签
            if "tags" in memory_metadata:
                memory_metadata["tags"] = list(set(memory_metadata["tags"] + detected_topics))[:5]  # 限制标签数量
            else:
                memory_metadata["tags"] = detected_topics[:5]
        
        # 添加重要性分数到metadata
        memory_metadata["importance_score"] = importance_score
        
        # 添加提取的关键信息到metadata
        if extracted_key_info:
            memory_metadata["key_information"] = extracted_key_info
        
        # ============ 添加评估器的详细信息 ============
        memory_metadata["evaluation"] = {
            "overall_score": evaluation_result.overall_score,
            "storage_priority": evaluation_result.storage_priority,
            "dimension_scores": evaluation_result.dimension_scores,
            "detected_intents": evaluation_result.detected_intents[:5],  # 限制数量
            "confidence": evaluation_result.confidence,
            "evaluation_method": evaluation_result.evaluation_method,
        }
        
        # 如果评估器建议了TTL，添加到metadata
        if evaluation_result.suggested_ttl_hours:
            memory_metadata["suggested_ttl_hours"] = evaluation_result.suggested_ttl_hours
        
        # 检查并进一步压缩metadata
        metadata_size: int = len(json.dumps(memory_metadata))
        if metadata_size > 30000:
            print(f"[记忆系统] 警告：metadata大小({metadata_size})接近限制，进一步精简")
            memory_metadata = self._further_compress_metadata(memory_metadata)

        # 准备记忆项（无论使用哪种存储方式都需要）
        memory_item = {
            "content": full_content,
            "user_id": self.user_id,
            "metadata": memory_metadata,
            "memory_id": memory_id,
            "timestamp": get_current_timestamp()
        }
        
        storage_success = False
        
        try:
            # 优先尝试添加到memo0
            if self.memory:
                try:
                    # 添加到memo0
                    add_result = self.memory.add(full_content, user_id=self.user_id, metadata=memory_metadata)
                    
                    # 验证存储是否成功（通过尝试检索来验证）
                    if self._verify_memory_stored(memory_id, memory_type):
                        storage_success = True
                        print(f"[记忆系统] 记忆成功存储到memo0: {memory_id}")
                    else:
                        print(f"[记忆系统] 警告：记忆可能未成功存储到memo0: {memory_id}")
                        # 即使验证失败，也继续添加到内存和文件存储作为备份
                except Exception as memo0_error:
                    print(f"[记忆系统] memo0存储失败: {memo0_error}，使用回退方案")
                    # 继续使用内存和文件存储
            
            # 如果memo0不可用或存储失败，使用内存和文件存储
            if not storage_success or not self.memory:
                # 添加到简化的内存存储
                self.memory_store.append(memory_item)
                # 同时保存到文件系统持久化存储
                self._save_to_persistent_storage(memory_item)
                print(f"[记忆系统] 记忆已保存到内存和文件存储: {memory_id}")
        except Exception as e:
            if "exceeds max length" in str(e):
                print("[记忆系统] 错误：metadata仍然过大，使用最小化版本")
                minimal_metadata: Dict[str, Any] = {
                    "memory_id": memory_id,
                    "memory_type": memory_type,
                    "session_id": self.session_id,
                    "created_at": get_current_timestamp(),
                    "importance_score": importance_score
                }
                # 即使在最小化版本中也保留主题信息
                if detected_topics:
                    minimal_metadata["topics"] = detected_topics[:2]  # 限制主题数量
                # 保留关键信息
                if extracted_key_info:
                    minimal_metadata["key_information"] = extracted_key_info[:200]  # 限制关键信息长度
                # 更新memory_item使用最小化metadata
                memory_item["metadata"] = minimal_metadata
                
                try:
                    if self.memory:
                        try:
                            self.memory.add(full_content, user_id=self.user_id, metadata=minimal_metadata)
                            # 验证存储
                            if self._verify_memory_stored(memory_id, memory_type):
                                print(f"[记忆系统] 最小化记忆成功存储到memo0: {memory_id}")
                        except Exception as memo0_error:
                            print(f"[记忆系统] memo0存储最小化记忆失败: {memo0_error}")
                    
                    # 无论如何都保存到内存和文件存储
                    self.memory_store.append(memory_item)
                    self._save_to_persistent_storage(memory_item)
                except Exception as minimal_error:
                    raise MemorySizeExceededError(
                        f"无法添加记忆：metadata大小超过限制",
                        memory_id=memory_id,
                        size=metadata_size,
                        max_size=30000
                    ) from minimal_error
            else:
                raise MemoryStorageError(
                    f"添加记忆失败: {str(e)}",
                    memory_id=memory_id,
                    details={"content_length": len(content_str)}
                ) from e

        # 更新语义记忆地图
        self._update_semantic_memory(memory_id, {
            "full_metadata": memory_metadata,
            "content_preview": content_str[:500] if content_str else "",
            "tags": tags or [],
            "emotional_intensity": emotional_intensity,
            "strategic_value": strategic_value or {},
            "linked_tool": linked_tool,
            "importance_score": importance_score,
            "content_value": content_value,
            "topics": detected_topics,  # 新增：主题信息
            "key_information": extracted_key_info  # 新增：关键信息
        })

        # 新增：更新主题索引
        self._update_topic_index(memory_id, detected_topics)
        
        # 新增：更新关键信息索引
        self._update_key_info_index(memory_id, extracted_key_info)
        
        # 新增：更新重要性索引
        self._update_importance_index(memory_id, importance_score)
        
        # 新增：更新类型索引
        self._update_type_index(memory_id, memory_type)

        # 更新统计信息
        self._update_stats(memory_type, tags or [])
        
        # 新增：更新主题统计
        for topic in detected_topics:
            self.stats["topics_distribution"][topic] += 1
        
        # 更新上下文窗口
        self._update_context_window(memory_id, detected_topics)

        # 更新记忆分级存储
        self._update_tier_store(memory_id)

        print(f"[记忆系统] 添加记忆成功，ID: {memory_id}，类型: {memory_type}，重要性: {importance_score:.2f}，价值: {content_value:.2f}")
        if detected_topics:
            print(f"[记忆系统] 检测到主题: {', '.join(detected_topics)}")
        if extracted_key_info:
            print(f"[记忆系统] 提取关键信息: {', '.join(extracted_key_info[:3])}...")
        return memory_id

    def _generate_memory_id(self, content: str, memory_type: str) -> str:
        """生成记忆ID"""
        return generate_memory_id(content, memory_type)

    def _extract_key_information(self, content: str, memory_type: str) -> List[str]:
        """提取关键信息 - 优化：增强特征提取，添加更多语义特征"""
        # 优化：增强的关键信息提取逻辑，结合规则和LLM
        
        key_info = []
        content_lower = content.lower()
        
        # 基于规则的关键信息提取 - 优化：扩展规则集
        import re
        
        # 1. 日期和时间 - 优化：扩展日期格式支持
        date_patterns = [
            r'\d{4}[-/]\d{1,2}[-/]\d{1,2}',  # YYYY-MM-DD或YYYY/MM/DD
            r'\d{1,2}[-/]\d{1,2}[-/]\d{4}',  # MM-DD-YYYY或MM/DD/YYYY
            r'\d{1,2}月\d{1,2}日',  # 中文日期，如1月1日
            r'\d{4}年\d{1,2}月\d{1,2}日',  # 中文完整日期，如2023年1月1日
            r'\d{2}:\d{2}',  # 时间，如14:30
            r'\d{2}:\d{2}:\d{2}',  # 完整时间，如14:30:00
            r'昨天|今天|明天|上周|本周|下周|上月|本月|下月',  # 相对日期
            r'\d+天前|\d+天后|\d+小时前|\d+小时后',  # 相对时间
        ]
        
        for pattern in date_patterns:
            matches = re.findall(pattern, content)
            for match in matches:
                key_info.append(f"时间: {match}")
        
        # 2. 地点 - 优化：增强地点提取
        location_keywords = ["在", "位于", "地址", "地点", "位置", "前往", "来自", "到达", "离开"]
        for keyword in location_keywords:
            if keyword in content_lower:
                # 优化：改进地点提取算法
                parts = content_lower.split(keyword)
                if len(parts) > 1:
                    # 提取关键词后面的内容，直到遇到标点符号或停用词
                    location_part = parts[1].split()[0] if parts[1].split() else ""
                    if location_part:
                        key_info.append(f"地点: {location_part}")
        
        # 3. 人物 - 优化：扩展人物关键词
        person_keywords = ["我", "你", "他", "她", "他们", "我们", "大家", "用户", "客户", "同事", "领导", 
                         "老师", "学生", "朋友", "家人", "医生", "护士", "律师", "工程师"]
        for keyword in person_keywords:
            if keyword in content_lower:
                key_info.append(f"人物: {keyword}")
        
        # 4. 数字和金额 - 优化：扩展数字模式
        number_patterns = [
            r'\d+\.?\d*',  # 数字
            r'¥\d+\.?\d*',  # 人民币金额
            r'\$\d+\.?\d*',  # 美元金额
            r'\d+%',  # 百分比
            r'第\d+',  # 序数词
            r'\d+个|\d+只|\d+条',  # 数量词
        ]
        
        for pattern in number_patterns:
            matches = re.findall(pattern, content)
            for match in matches:
                key_info.append(f"数字: {match}")
        
        # 5. 动作和事件 - 优化：扩展动作关键词
        action_keywords = ["做", "完成", "开始", "结束", "计划", "安排", "会议", "任务", "项目", "工作", 
                         "学习", "研究", "开发", "测试", "部署", "维护", "升级", "更新"]
        for keyword in action_keywords:
            if keyword in content_lower:
                key_info.append(f"动作: {keyword}")
        
        # 6. 重要性词汇 - 优化：扩展重要性关键词
        importance_keywords = ["重要", "关键", "紧急", "必须", "需要", "应该", "建议", "务必", "亟需", "优先", "核心"]
        for keyword in importance_keywords:
            if keyword in content_lower:
                key_info.append(f"重要性: {keyword}")
        
        # 7. 情感词汇 - 新增：情感特征提取
        positive_emotions = ["高兴", "开心", "满意", "喜欢", "好", "棒", "优秀", "出色", "成功", "胜利"]
        negative_emotions = ["生气", "难过", "失望", "讨厌", "差", "糟糕", "失败", "错误", "问题", "困难"]
        for emotion in positive_emotions:
            if emotion in content_lower:
                key_info.append(f"情感: 积极")
                break
        for emotion in negative_emotions:
            if emotion in content_lower:
                key_info.append(f"情感: 消极")
                break
        
        # 8. 记忆类型特定提取 - 优化：增强类型特定提取
        if memory_type == "conversation":
            # 对话特定关键信息提取
            if "问题" in content_lower or "？" in content:
                key_info.append("类型: 问题")
            if "回答" in content_lower or "答案" in content_lower:
                key_info.append("类型: 回答")
            if "请求" in content_lower or "要求" in content_lower:
                key_info.append("类型: 请求")
            if "建议" in content_lower or "推荐" in content_lower:
                key_info.append("类型: 建议")
        elif memory_type == "experience":
            # 经历特定关键信息提取
            if "经验" in content_lower or "经历" in content_lower:
                key_info.append("类型: 经验分享")
            if "教训" in content_lower or "总结" in content_lower:
                key_info.append("类型: 经验总结")
        elif memory_type == "user_profile":
            # 用户档案特定关键信息提取
            profile_keywords = ["姓名", "年龄", "性别", "职业", "爱好", "兴趣", "生日", "籍贯", "学历", "技能"]
            for keyword in profile_keywords:
                if keyword in content_lower:
                    key_info.append(f"用户信息: {keyword}")
        elif memory_type == "knowledge":
            # 知识特定关键信息提取
            knowledge_keywords = ["定义", "概念", "原理", "方法", "步骤", "技巧", "规则", "标准", "公式"]
            for keyword in knowledge_keywords:
                if keyword in content_lower:
                    key_info.append(f"知识类型: {keyword}")
        
        # 9. 主题相关 - 新增：主题特征提取
        for topic, keywords in self.topic_keywords.items():
            for keyword in keywords:
                if keyword in content_lower:
                    key_info.append(f"主题: {topic}")
                    break
        
        # 去重并限制数量 - 优化：增加关键信息数量至15个
        key_info = list(set(key_info))[:15]  # 最多返回15个关键信息（优化：从10个增加到15个）
        
        return key_info

    def _prepare_metadata(
                          self,
                          memory_id: str,
                          memory_type: str,
                          content: str,
                          metadata: Dict[str, Any],
                          tags: List[str],
                          emotional_intensity: float,
                          strategic_value: Dict[str, Any],
                          linked_tool: Optional[str]) -> Dict[str, Any]:
        """准备精简的元数据"""
        # 精简strategic_value
        simplified_strategic: Dict[str, Any] = {
            "score": strategic_value.get("score", 0),
            "level": strategic_value.get("level", "低")
        } if strategic_value and isinstance(strategic_value, dict) else {"score": 0, "level": "低"}

        base_metadata: Dict[str, Any] = {
            "memory_id": memory_id,
            "memory_type": memory_type,
            "session_id": self.session_id,
            "created_at": get_current_timestamp(),
            "tags": tags[:5] if tags else [],
            "emotional_intensity": round(emotional_intensity, 2),
            "strategic_score": simplified_strategic.get("score", 0),
            "strategic_level": simplified_strategic.get("level", "低"),
            "linked_tool": linked_tool,
            "source": "zyantine_system",
            "content_length": len(content),
            "content_preview": content[:200] if content else "",
            "user_id": self.user_id
        }

        # 合并必要的外部元数据
        for key in ["interaction_id", "context_keys", "action_plan_keys", "growth_result_keys"]:
            if key in metadata:
                if key in ["context_keys", "action_plan_keys", "growth_result_keys"] and isinstance(metadata[key],
                                                                                                    dict):
                    base_metadata[key] = list(metadata[key].keys())[:5]
                else:
                    base_metadata[key] = metadata[key]

        return base_metadata

    def _further_compress_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """进一步压缩metadata"""
        minimal: Dict[str, Any] = {
            "memory_id": metadata.get("memory_id", ""),
            "memory_type": metadata.get("memory_type", ""),
            "session_id": metadata.get("session_id", ""),
            "created_at": metadata.get("created_at", ""),
            "content_length": metadata.get("content_length", 0)
        }

        if "tags" in metadata and metadata["tags"]:
            minimal["tags"] = metadata["tags"][:2]

        return minimal

    def _update_semantic_memory(self, memory_id: str, memory_data: Dict[str, Any]) -> None:
        """更新语义记忆地图"""
        self.semantic_memory_map[memory_id] = {
            "metadata": memory_data,
            "access_count": 0,
            "last_accessed": None,
            "strategic_score": memory_data.get("strategic_score", 0),
            "importance_score": memory_data.get("importance_score", 5.0),
            "content_value": memory_data.get("content_value", 5.0)
        }
        print(f"[记忆系统] 更新语义记忆地图: {memory_id}")
        
        # 检查语义记忆地图大小，超过限制时清理
        if len(self.semantic_memory_map) > self.max_semantic_map_size:
            self._cleanup_semantic_memory()
    
    def _detect_topics(self, content: str) -> List[str]:
        """检测内容的主题"""
        detected_topics = []
        content_lower = content.lower()
        
        # 遍历主题关键词库，检测匹配的主题
        for topic, keywords in self.topic_keywords.items():
            for keyword in keywords:
                if keyword in content_lower:
                    if topic not in detected_topics:
                        detected_topics.append(topic)
                    break
        
        return detected_topics
    
    def _update_topic_index(self, memory_id: str, topics: List[str]) -> None:
        """更新主题索引"""
        if not topics:
            return
        
        # 添加到主题索引
        for topic in topics:
            if memory_id not in self.topic_index[topic]:
                self.topic_index[topic].append(memory_id)
        
        # 更新记忆主题映射
        self.memory_topics[memory_id] = topics
    
    def _update_key_info_index(self, memory_id: str, key_info: List[str]) -> None:
        """更新关键信息索引"""
        if not key_info:
            return
        
        # 添加到关键信息索引
        for info in key_info:
            if memory_id not in self.key_info_index[info]:
                self.key_info_index[info].append(memory_id)
        
        # 更新记忆关键信息映射
        self.memory_key_info[memory_id] = key_info
    
    def _update_importance_index(self, memory_id: str, importance_score: float) -> None:
        """更新重要性索引"""
        # 将重要性分数转换为整数，用于索引
        importance_level = int(round(importance_score))
        if memory_id not in self.importance_index[importance_level]:
            self.importance_index[importance_level].append(memory_id)
    
    def _update_type_index(self, memory_id: str, memory_type: str) -> None:
        """更新类型索引"""
        if memory_id not in self.type_index[memory_type]:
            self.type_index[memory_type].append(memory_id)
    
    # ============ 上下文窗口管理 ============
    
    def _update_context_window(self, memory_id: str, topics: List[str]) -> None:
        """更新上下文窗口，将相关记忆添加到当前上下文"""
        # 如果记忆已经在上下文窗口中，移到末尾（最近使用）
        if memory_id in self.context_window:
            self.context_window.remove(memory_id)
        
        # 添加到上下文窗口末尾
        self.context_window.append(memory_id)
        
        # 更新当前对话主题
        self._update_current_topics(topics)
        
        # 标记为短期记忆
        self.short_term_memories.add(memory_id)
        
        # 动态调整上下文窗口大小
        self._adjust_context_window_size()
        
        # 如果上下文窗口超过最大大小，移除最旧的记忆
        if len(self.context_window) > self.current_context_window_size:
            removed_memory = self.context_window.pop(0)
            # 如果移除的是短期记忆，从短期记忆集合中移除
            if removed_memory in self.short_term_memories:
                self.short_term_memories.remove(removed_memory)
    
    def _adjust_context_window_size(self) -> None:
        """动态调整上下文窗口大小"""
        # 基于当前对话活跃度和记忆重要性调整窗口大小
        # 这里实现一个简单的自适应算法，基于当前对话的记忆数量和重要性
        if len(self.context_window) > self.current_context_window_size * 1.5:
            # 对话活跃，增大窗口
            self.current_context_window_size = min(
                self.current_context_window_size + 2,
                self.max_context_window_size
            )
        elif len(self.context_window) < self.current_context_window_size * 0.5:
            # 对话不活跃，缩小窗口
            self.current_context_window_size = max(
                self.current_context_window_size - 1,
                self.min_context_window_size
            )
    
    def _update_current_topics(self, topics: List[str]) -> None:
        """更新当前对话主题及权重"""
        # 主题权重衰减
        for topic in self.current_topics:
            self.current_topics[topic] *= self.topic_decay_rate
            
        # 移除权重过低的主题
        self.current_topics = {k: v for k, v in self.current_topics.items() if v > 0.1}
        
        # 更新新主题权重
        for topic in topics:
            if topic in self.current_topics:
                self.current_topics[topic] += 0.5
            else:
                self.current_topics[topic] = 1.0
        
        # 归一化主题权重
        total_weight = sum(self.current_topics.values())
        if total_weight > 0:
            for topic in self.current_topics:
                self.current_topics[topic] /= total_weight
    

    
    def _update_memory_topics(self, memory_id: str, content: str) -> None:
        """更新现有记忆的主题"""
        # 检测新主题
        new_topics = self._detect_topics(content)
        
        # 更新主题索引
        self._update_topic_index(memory_id, new_topics)
    
    def _cleanup_semantic_memory(self):
        """清理语义记忆地图中的低价值记忆"""
        if len(self.semantic_memory_map) <= self.max_semantic_map_size:
            return
        
        # 计算需要清理的数量
        cleanup_count = len(self.semantic_memory_map) - int(self.max_semantic_map_size * 0.8)  # 清理到80%容量
        
        # 按价值排序记忆
        memory_values = []
        for memory_id, memory_data in self.semantic_memory_map.items():
            # 计算综合价值分数
            importance = memory_data.get("importance_score", 5.0)
            content_value = memory_data.get("content_value", 5.0)
            access_count = memory_data.get("access_count", 0)
            
            total_value = importance * 0.5 + content_value * 0.3 + (access_count / 10.0) * 0.2
            memory_values.append((memory_id, total_value))
        
        # 按价值从低到高排序
        memory_values.sort(key=lambda x: x[1])
        
        # 清理低价值记忆
        cleaned_count = 0
        for memory_id, value in memory_values:
            if cleaned_count >= cleanup_count or value >= self.semantic_map_cleanup_threshold:
                break
            
            if memory_id in self.semantic_memory_map:
                # 从主题索引中移除
                if memory_id in self.memory_topics:
                    topics = self.memory_topics[memory_id]
                    for topic in topics:
                        if memory_id in self.topic_index[topic]:
                            self.topic_index[topic].remove(memory_id)
                    del self.memory_topics[memory_id]
                
                del self.semantic_memory_map[memory_id]
                # 同时从访问统计中删除
                if memory_id in self.stats.get("access_counts", {}):
                    del self.stats["access_counts"][memory_id]
                cleaned_count += 1
        
        print(f"[记忆系统] 清理语义记忆地图，删除 {cleaned_count} 个低价值记忆，剩余 {len(self.semantic_memory_map)} 个")

    def _update_stats(self, memory_type: str, tags: List[str]) -> None:
        """更新统计信息"""
        self.stats["total_memories"] += 1

        # 按类型统计
        self.stats["by_type"][memory_type] = self.stats["by_type"].get(memory_type, 0) + 1

        # 标签分布
        for tag in tags:
            self.stats["tags_distribution"][tag] = self.stats["tags_distribution"].get(tag, 0) + 1
            if tag not in self.strategic_tags:
                self.strategic_tags.append(tag)

    # ============ 记忆检索 ============



    def _enhance_search_query(self, query: str) -> str:
        """增强搜索查询"""
        # 检查缓存
        cached_query = self.query_enhancement_cache.get(query)
        if cached_query is not None:
            # 确保缓存返回的是字符串类型
            if isinstance(cached_query, str):
                return cached_query
            else:
                # 缓存值无效，使用原始查询
                self.query_enhancement_cache.set(query, query)
                return query
        
        # 缓存原始查询（已移除LLM查询增强，使用基于规则的方法）
        self.query_enhancement_cache.set(query, query)
        return query
    
    def _build_context_enhanced_query(self, query: str) -> str:
        """构建上下文增强查询，将当前对话上下文融入查询"""
        # 基础查询
        context_enhanced_query = query
        
        # 添加当前对话主题
        if self.current_topics:
            top_topics = sorted(self.current_topics.items(), key=lambda x: x[1], reverse=True)[:2]
            topic_str = " ".join([topic for topic, _ in top_topics])
            context_enhanced_query = f"{context_enhanced_query} 相关主题: {topic_str}"
        
        # 添加最近上下文记忆的关键词
        context_keywords = []
        for context_memory_id in self.context_window[-2:]:  # 最近2个上下文记忆
            if context_memory_id in self.semantic_memory_map:
                context_memory = self.semantic_memory_map[context_memory_id]
                context_content = context_memory.get("metadata", {}).get("content_preview", "")
                # 提取关键词（前5个词）
                keywords = context_content.split()[:5]
                context_keywords.extend(keywords)
        
        # 去重并限制关键词数量
        unique_keywords = list(set(context_keywords))[:8]
        if unique_keywords:
            keyword_str = " ".join(unique_keywords)
            context_enhanced_query = f"{context_enhanced_query} 上下文相关词: {keyword_str}"
        
        return context_enhanced_query
    
    def _calculate_context_relevance(self, memory_id: str, metadata: Dict[str, Any]) -> float:
        """计算记忆与当前上下文的相关性 - 优化：增强上下文匹配逻辑"""
        if not self.context_window:
            return 0.0
        
        relevance_score = 0.0
        max_score = 1.0
        
        # 检查记忆是否在当前上下文窗口中
        if memory_id in self.context_window:
            # 计算在上下文窗口中的位置（越新分数越高）
            position = len(self.context_window) - self.context_window.index(memory_id)
            relevance_score += (position / len(self.context_window)) * max_score
            return relevance_score
        
        # 优化：考虑最近5个上下文记忆（从3个增加到5个）
        recent_contexts = self.context_window[-5:]
        
        # 检查记忆主题与上下文主题的相关性
        memory_topics = metadata.get("topics", [])
        memory_key_info = metadata.get("key_information", [])
        memory_content = metadata.get("content_preview", "")
        
        # 优化：为不同位置的上下文记忆设置不同权重（越新权重越高）
        for i, context_memory_id in enumerate(reversed(recent_contexts)):
            if context_memory_id in self.semantic_memory_map:
                context_memory = self.semantic_memory_map[context_memory_id]
                context_topics = context_memory.get("metadata", {}).get("topics", [])
                context_key_info = context_memory.get("metadata", {}).get("key_information", [])
                context_content = context_memory.get("metadata", {}).get("content_preview", "")
                
                # 1. 主题交集（优化：增加权重）
                common_topics = set(memory_topics) & set(context_topics)
                if common_topics:
                    topic_score = (len(common_topics) / max(len(memory_topics), 1)) * (max_score / 2)
                    relevance_score += topic_score
                
                # 2. 关键信息匹配（新增：关键信息交集）
                common_key_info = set(memory_key_info) & set(context_key_info)
                if common_key_info:
                    key_info_score = (len(common_key_info) / max(len(memory_key_info), 1)) * (max_score / 3)
                    relevance_score += key_info_score
                
                # 3. 内容关键词匹配（新增：内容关键词重叠）
                memory_words = set(memory_content.lower().split())
                context_words = set(context_content.lower().split())
                common_words = memory_words & context_words
                if common_words and len(memory_words) > 0:
                    word_match_score = (len(common_words) / len(memory_words)) * (max_score / 4)
                    relevance_score += word_match_score
                
                # 权重衰减：越旧的上下文权重越低
                relevance_score *= (0.8 + 0.2 * (i + 1) / len(recent_contexts))
        
        return min(relevance_score, max_score)
    
    def _calculate_topic_relevance(self, memory_topics: List[str]) -> float:
        """计算记忆主题与当前对话主题的相关性 - 优化：增强主题匹配逻辑"""
        if not memory_topics or not self.current_topics:
            return 0.0
        
        relevance_score = 0.0
        
        # 1. 直接主题匹配（原有逻辑）
        for topic in memory_topics:
            if topic in self.current_topics:
                relevance_score += self.current_topics[topic] * 0.7  # 优化：调整权重
        
        # 2. 主题相似度增强（优化：基于关键词匹配的语义关联）
        try:
            # 计算主题关键词的匹配度
            current_topic_list = list(self.current_topics.keys())
            if current_topic_list:
                # 基于关键词的主题相似度计算
                topic_keywords = self.topic_keywords
                memory_keywords = set()
                current_keywords = set()
                
                # 收集记忆主题的所有关键词
                for topic in memory_topics:
                    if topic in topic_keywords:
                        memory_keywords.update(topic_keywords[topic])
                
                # 收集当前主题的所有关键词
                for topic in current_topic_list:
                    if topic in topic_keywords:
                        current_keywords.update(topic_keywords[topic])
                
                # 计算关键词交集
                common_keywords = memory_keywords & current_keywords
                if common_keywords and len(memory_keywords) > 0:
                    semantic_match_score = len(common_keywords) / len(memory_keywords)
                    relevance_score += semantic_match_score * 0.3  # 新增：语义匹配权重
        except Exception as e:
            # 如果计算失败，继续使用原有逻辑
            print(f"[记忆系统] 主题语义匹配失败: {e}")
        
        # 归一化到0-1范围
        return min(relevance_score, 1.0)
    
    def _calculate_time_relevance(self, created_at: Optional[str]) -> float:
        """计算记忆的时间相关性"""
        if not created_at:
            return 0.0
        
        try:
            time_diff = calculate_time_diff(created_at)
            
            # 最近1小时内的记忆获得最高分数
            if time_diff < 3600:
                return 1.0
            # 最近24小时内
            elif time_diff < 86400:
                return 0.8
            # 最近7天内
            elif time_diff < 604800:
                return 0.5
            # 最近30天内
            elif time_diff < 2592000:
                return 0.3
            # 更旧的记忆
            else:
                return 0.1
        except:
            return 0.0
    
    def _calculate_ranking_score(self, similarity_score: float, importance_score: float, 
                              access_count: int, context_relevance: float, 
                              topic_relevance: float, time_relevance: float, 
                              memory_type: Optional[str]) -> float:
        """计算最终排序分数，综合考虑多种因素"""
        # 权重配置 - 优化：提高相似度权重至0.7，确保核心匹配质量
        weights = {
            "similarity": 0.7,      # 相似度权重（优化：从0.5提高到0.7）
            "importance": 0.1,      # 重要性权重（优化：从0.15调整到0.1）
            "access_count": 0.05,   # 访问频率权重（优化：从0.1调整到0.05）
            "context_relevance": 0.08, # 上下文相关性权重（优化：从0.1调整到0.08）
            "topic_relevance": 0.05,  # 主题相关性权重（优化：从0.1调整到0.05）
            "time_relevance": 0.02   # 时间相关性权重（优化：从0.05调整到0.02）
        }
        
        # 类型权重调整（可选）
        type_bonus = 0.0
        if memory_type == "user_profile":
            type_bonus = 0.1
        elif memory_type == "experience":
            type_bonus = 0.05
        
        # 计算加权总分
        total_score = (
            similarity_score * weights["similarity"] +
            (importance_score / 10.0) * weights["importance"] +
            min(access_count / 20.0, 1.0) * weights["access_count"] +
            context_relevance * weights["context_relevance"] +
            topic_relevance * weights["topic_relevance"] +
            time_relevance * weights["time_relevance"]
        ) + type_bonus
        
        return min(total_score, 1.0)  # 归一化到0-1范围

    def _adjust_similarity_threshold(self, memory_type: Optional[str], query: Optional[str] = None) -> float:
        """根据记忆类型调整相似度阈值 - 优化：动态调整阈值"""
        # 为不同类型的记忆设置基础阈值
        base_thresholds = {
            "user_profile": 0.8,      # 用户档案需要更高的准确性
            "knowledge": 0.75,        # 知识记忆需要较高的准确性
            "strategy": 0.75,         # 策略记忆需要较高的准确性
            "experience": 0.7,         # 经历记忆
            "conversation": 0.6,       # 对话记忆
            "emotion": 0.65,           # 情感记忆
            "system_event": 0.5,       # 系统事件
            "temporal": 0.6            # 时间相关记忆
        }
        
        # 获取基础阈值
        threshold = base_thresholds.get(memory_type, 0.7)
        
        # 优化：根据查询复杂度调整阈值
        if query:
            query_length = len(query)
            # 长查询通常需要更高的准确性
            if query_length > 100:
                threshold += 0.1
            # 短查询可以适当降低阈值
            elif query_length < 20:
                threshold -= 0.05
        
        # 优化：根据上下文活跃度调整阈值
        # 上下文窗口越大，说明对话越活跃，需要更高的准确性
        if hasattr(self, 'context_window') and len(self.context_window) > 15:
            threshold += 0.05
        elif hasattr(self, 'context_window') and len(self.context_window) < 5:
            threshold -= 0.05
        
        # 优化：根据当前主题数量调整阈值
        if hasattr(self, 'current_topics') and len(self.current_topics) > 2:
            # 主题越多，查询越具体，需要更高的准确性
            threshold += 0.05
        
        # 确保阈值在合理范围内
        threshold = max(0.4, min(0.9, threshold))
        
        return threshold

    def search_memories(self,
                        query: str,
                        memory_type: Optional[str] = None,
                        tags: Optional[List[str]] = None,
                        limit: int = 5,
                        similarity_threshold: float = 0.7,
                        rerank: bool = True) -> List[Dict[str, Any]]:
        """搜索记忆，优化版：上下文感知检索 + 增强索引"""
        try:
            # 1. 构建上下文增强查询
            context_enhanced_query = self._build_context_enhanced_query(query)
            
            # 2. 增强搜索查询（原有逻辑）
            enhanced_query = self._enhance_search_query(context_enhanced_query)
            
            # 3. 如果使用简化的内存存储（memo0不可用）
            if not self.memory:
                # 从内存存储和文件存储中检索
                results = []
                
                # 从内存存储检索
                for memory_item in self.memory_store:
                    results.extend(self._search_in_memory_item(memory_item, enhanced_query, memory_type, tags))
                
                # 从文件存储检索（如果可用）
                if self._persistent_storage_path:
                    file_results = self._search_in_persistent_storage(enhanced_query, memory_type, tags, limit * 2)
                    # 合并结果并去重
                    existing_ids = {r["memory_id"] for r in results}
                    for result in file_results:
                        if result["memory_id"] not in existing_ids:
                            results.append(result)
                            existing_ids.add(result["memory_id"])
                
                # 优化：添加严格的过滤机制，排除低相关度记忆
                filtered_results = []
                for result in results:
                    # 1. 最低相似度阈值过滤（默认0.5）
                    if result.get("similarity_score", 0) < 0.5:
                        continue
                    
                    # 2. 记忆质量评估（排除低质量记忆）
                    importance_score = result.get("importance_score", 0.0)
                    if importance_score < 3.0:  # 重要性分数低于3.0的记忆视为低质量
                        continue
                    
                    filtered_results.append(result)
                
                # 排序并限制结果数量
                filtered_results.sort(key=lambda x: x.get("similarity_score", 0), reverse=True)
                return filtered_results[:limit]
            
            # 4. 根据记忆类型调整相似度阈值 - 优化：传递query参数支持动态阈值调整
            adjusted_threshold = self._adjust_similarity_threshold(memory_type, enhanced_query)
            
            # 5. 构建元数据过滤器，增加当前主题过滤
            metadata_filter: Dict[str, Any] = {}
            if memory_type:
                metadata_filter["memory_type"] = memory_type
            if tags:
                metadata_filter["tags"] = tags
            
            # 添加当前主题到过滤器（如果有）
            current_topic_list = list(self.current_topics.keys())[:3]  # 最多使用3个当前主题
            if current_topic_list:
                if "tags" not in metadata_filter:
                    metadata_filter["tags"] = []
                metadata_filter["tags"].extend(current_topic_list)

            # 添加参数验证
            if not enhanced_query or not isinstance(enhanced_query, str):
                print(f"[记忆系统] 无效的查询参数: {enhanced_query}")
                return []

            # 确保metadata_filter是有效格式（如果是字典）
            if metadata_filter and not isinstance(metadata_filter, dict):
                print(f"[记忆系统] metadata_filter格式错误: {type(metadata_filter)}")
                metadata_filter = {}

            # 6. 执行搜索（使用memo0）
            try:
                # 根据mem0 API，第一个参数是query（位置参数），后面必须使用命名参数
                # 方法签名: search(query: str, *, user_id: ..., limit: ..., filters: ..., rerank: ...)
                search_kwargs = {
                    "user_id": self.user_id,
                    "limit": limit * 3,  # 获取更多结果用于上下文重排序
                    "rerank": rerank
                }
                
                # 添加filters参数（mem0使用filters而不是metadata_filter）
                if metadata_filter:
                    search_kwargs["filters"] = metadata_filter
                
                # 调用search，使用位置参数query作为第一个参数
                search_results: Dict[str, Any] = self.memory.search(
                    enhanced_query if enhanced_query else " ",
                    **search_kwargs
                )
                
                # 验证搜索结果格式
                if not isinstance(search_results, dict):
                    print(f"[记忆系统] memo0搜索结果格式错误: {type(search_results)}")
                    # 如果格式错误，回退到内存存储搜索
                    return self._fallback_search(query, memory_type, tags, limit, similarity_threshold)
                
                if "results" not in search_results:
                    print(f"[记忆系统] memo0搜索结果缺少results字段")
                    return self._fallback_search(query, memory_type, tags, limit, similarity_threshold)
                
            except Exception as search_error:
                print(f"[记忆系统] memo0搜索失败: {search_error}，使用回退方案")
                # 如果搜索失败，回退到内存和文件存储搜索
                return self._fallback_search(query, memory_type, tags, limit, similarity_threshold)

            processed_results: List[Dict[str, Any]] = []
            for hit in search_results.get("results", []):
                memory_data: Any = hit.get("memory", {})
                metadata: Dict[str, Any] = hit.get("metadata", {})
                similarity_score: float = hit.get("score", 0)

                # 使用调整后的阈值过滤
                if similarity_score < adjusted_threshold:
                    continue

                # 更新访问统计
                memory_id: Optional[str] = metadata.get("memory_id")
                if memory_id:
                    if memory_id in self.semantic_memory_map:
                        self.semantic_memory_map[memory_id]["access_count"] += 1
                        self.semantic_memory_map[memory_id]["last_accessed"] = get_current_timestamp()
                    self.stats["access_counts"][memory_id] = self.stats["access_counts"].get(memory_id, 0) + 1

                # 计算上下文相关性
                context_relevance = self._calculate_context_relevance(memory_id, metadata)

                # 计算主题相关性
                memory_topics = metadata.get("topics", [])
                topic_relevance = self._calculate_topic_relevance(memory_topics)

                # 计算关键信息相关性
                key_info_relevance = 0.0
                memory_key_info = metadata.get("key_information", [])
                if memory_key_info:
                    for info in memory_key_info:
                        if info.lower() in enhanced_query.lower():
                            key_info_relevance += 0.3

                # 计算时间相关性
                time_relevance = self._calculate_time_relevance(metadata.get("created_at"))

                # 计算最终排序分数
                final_score = self._calculate_ranking_score(
                    similarity_score,
                    metadata.get("importance_score", 5.0),
                    self.stats["access_counts"].get(memory_id, 0),
                    context_relevance,
                    topic_relevance,
                    time_relevance,
                    metadata.get("memory_type")
                )

                # 提取内容
                content = self._extract_content_from_memory(memory_data)

                # 检查内容长度
                if len(content) > 2000:
                    content = content[:2000] + "..."

                processed_results.append({
                    "memory_id": memory_id,
                    "content": content,
                    "metadata": metadata,
                    "similarity_score": similarity_score,
                    "final_score": final_score,
                    "memory_type": metadata.get("memory_type"),
                    "tags": metadata.get("tags", []),
                    "emotional_intensity": metadata.get("emotional_intensity", 0.5),
                    "strategic_value": metadata.get("strategic_value", {}),
                    "linked_tool": metadata.get("linked_tool"),
                    "created_at": metadata.get("created_at"),
                    "access_count": self.stats["access_counts"].get(memory_id, 0),
                    "importance_score": metadata.get("importance_score", 5.0),
                    "context_relevance": context_relevance,
                    "topic_relevance": topic_relevance,
                    "time_relevance": time_relevance,
                    "key_info_relevance": key_info_relevance
                })

            # 优化：添加严格的过滤机制，排除低相关度记忆
            filtered_results = []
            for result in processed_results:
                # 1. 最低相似度阈值过滤（默认0.5）
                if result["similarity_score"] < 0.5:
                    continue
                
                # 2. 最终分数过滤（默认0.3）
                if result["final_score"] < 0.3:
                    continue
                
                # 3. 记忆质量评估（新增：排除低质量记忆）
                importance_score = result.get("importance_score", 0.0)
                if importance_score < 3.0:  # 重要性分数低于3.0的记忆视为低质量
                    continue
                
                filtered_results.append(result)
            
            # 按最终分数排序
            filtered_results.sort(key=lambda x: x["final_score"], reverse=True)

            # 限制结果数量
            return filtered_results[:limit]

        except Exception as e:
            print(f"[记忆系统] 搜索记忆失败: {e}")
            print(f"[记忆系统] 搜索参数: input={enhanced_query}, user_id={self.user_id}, limit={limit * 3}, rerank={rerank}, metadata_filter={metadata_filter}")
            # 记录完整错误信息到日志
            traceback.print_exc()
            return []

    def _search_in_memory_item(self, 
                               memory_item: Dict[str, Any],
                               enhanced_query: str,
                               memory_type: Optional[str] = None,
                               tags: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        从单个内存项中搜索，计算相关性分数
        
        Args:
            memory_item: 内存项字典，包含content、metadata、memory_id等
            enhanced_query: 增强后的查询字符串
            memory_type: 记忆类型过滤
            tags: 标签过滤
            
        Returns:
            符合条件的记忆结果列表（通常0或1个）
        """
        results = []
        
        try:
            # 检查用户ID匹配
            if memory_item.get("user_id") != self.user_id:
                return results
            
            # 提取记忆内容
            content = self._extract_content_from_memory(memory_item.get("content", ""))
            metadata = memory_item.get("metadata", {})
            memory_id = memory_item.get("memory_id") or metadata.get("memory_id")
            
            # 检查记忆类型过滤
            if memory_type and metadata.get("memory_type") != memory_type:
                return results
            
            # 检查标签过滤
            if tags:
                memory_tags = metadata.get("tags", [])
                if not any(tag in memory_tags for tag in tags):
                    return results
            
            # 计算文本匹配分数（使用Jaccard相似度）
            similarity_score = self._calculate_text_similarity(enhanced_query, content)
            
            # 计算上下文相关性
            context_relevance = self._calculate_context_relevance(memory_id, metadata)
            
            # 计算主题相关性
            memory_topics = metadata.get("topics", [])
            topic_relevance = self._calculate_topic_relevance(memory_topics)
            
            # 计算关键信息相关性
            key_info_relevance = 0.0
            memory_key_info = metadata.get("key_information", [])
            if memory_key_info:
                query_lower = enhanced_query.lower()
                for info in memory_key_info:
                    if isinstance(info, str) and info.lower() in query_lower:
                        key_info_relevance += 0.3
            
            # 计算时间相关性
            time_relevance = self._calculate_time_relevance(metadata.get("created_at"))
            
            # 计算最终排序分数
            importance_score = metadata.get("importance_score", 5.0)
            access_count = self.stats.get("access_counts", {}).get(memory_id, 0)
            
            final_score = self._calculate_ranking_score(
                similarity_score,
                importance_score,
                access_count,
                context_relevance,
                topic_relevance,
                time_relevance,
                metadata.get("memory_type")
            )
            
            # 返回格式化的结果
            if similarity_score > 0.0 or context_relevance > 0.0 or topic_relevance > 0.0:
                results.append({
                    "memory_id": memory_id,
                    "content": content[:2000] if len(content) > 2000 else content,
                    "metadata": metadata,
                    "similarity_score": similarity_score,
                    "final_score": final_score,
                    "memory_type": metadata.get("memory_type"),
                    "tags": metadata.get("tags", []),
                    "emotional_intensity": metadata.get("emotional_intensity", 0.5),
                    "strategic_value": metadata.get("strategic_value", {}),
                    "linked_tool": metadata.get("linked_tool"),
                    "created_at": metadata.get("created_at"),
                    "access_count": access_count,
                    "importance_score": importance_score,
                    "context_relevance": context_relevance,
                    "topic_relevance": topic_relevance,
                    "time_relevance": time_relevance,
                    "key_info_relevance": key_info_relevance
                })
        
        except Exception as e:
            print(f"[记忆系统] 搜索内存项失败: {e}")
        
        return results
    
    def _calculate_text_similarity(self, query: str, content: str) -> float:
        """
        计算查询和内容的文本相似度（使用Jaccard相似度）
        
        Args:
            query: 查询字符串
            content: 内容字符串
            
        Returns:
            相似度分数 (0.0 - 1.0)
        """
        if not query or not content:
            return 0.0
        
        query_lower = query.lower().strip()
        content_lower = content.lower().strip()
        
        # 完全匹配
        if query_lower == content_lower:
            return 1.0
        
        # 查询包含在内容中
        if query_lower in content_lower:
            return 0.9
        
        # 使用Jaccard相似度
        query_words = set(query_lower.split())
        content_words = set(content_lower.split())
        
        if not query_words or not content_words:
            return 0.0
        
        intersection = len(query_words & content_words)
        union = len(query_words | content_words)
        
        if union == 0:
            return 0.0
        
        jaccard_score = intersection / union
        
        # 如果查询的关键词大部分都在内容中，提高分数
        if len(query_words) > 0:
            coverage = intersection / len(query_words)
            # 结合Jaccard和覆盖率
            similarity = (jaccard_score * 0.5 + coverage * 0.5)
            return min(similarity, 1.0)
        
        return jaccard_score

    def _search_in_persistent_storage(self,
                                     enhanced_query: str,
                                     memory_type: Optional[str] = None,
                                     tags: Optional[List[str]] = None,
                                     limit: int = 10) -> List[Dict[str, Any]]:
        """
        从文件系统持久化存储中搜索记忆
        
        Args:
            enhanced_query: 增强后的查询字符串
            memory_type: 记忆类型过滤
            tags: 标签过滤
            limit: 结果数量限制
            
        Returns:
            符合条件的记忆结果列表
        """
        results = []
        
        if not self._persistent_storage_path:
            return results
        
        try:
            import os
            from pathlib import Path
            
            storage_file = Path(self._persistent_storage_path)
            if not storage_file.exists():
                return results
            
            # 从JSONL文件读取记忆
            with open(storage_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if not line.strip():
                        continue
                    
                    try:
                        memory_item = json.loads(line.strip())
                        
                        # 检查用户ID匹配
                        if memory_item.get("user_id") != self.user_id:
                            continue
                        
                        metadata = memory_item.get("metadata", {})
                        
                        # 检查记忆类型过滤
                        if memory_type and metadata.get("memory_type") != memory_type:
                            continue
                        
                        # 检查标签过滤
                        if tags:
                            memory_tags = metadata.get("tags", [])
                            if not any(tag in memory_tags for tag in tags):
                                continue
                        
                        # 提取内容
                        content = self._extract_content_from_memory(memory_item.get("content", ""))
                        memory_id = memory_item.get("memory_id") or metadata.get("memory_id")
                        
                        # 计算相关性分数
                        similarity_score = self._calculate_text_similarity(enhanced_query, content)
                        
                        # 只保留有一定相关性的结果
                        if similarity_score < 0.1:
                            continue
                        
                        # 计算上下文相关性
                        context_relevance = self._calculate_context_relevance(memory_id, metadata)
                        
                        # 计算主题相关性
                        memory_topics = metadata.get("topics", [])
                        topic_relevance = self._calculate_topic_relevance(memory_topics)
                        
                        # 计算关键信息相关性
                        key_info_relevance = 0.0
                        memory_key_info = metadata.get("key_information", [])
                        if memory_key_info:
                            query_lower = enhanced_query.lower()
                            for info in memory_key_info:
                                if isinstance(info, str) and info.lower() in query_lower:
                                    key_info_relevance += 0.3
                        
                        # 计算时间相关性
                        time_relevance = self._calculate_time_relevance(metadata.get("created_at"))
                        
                        # 计算最终排序分数
                        importance_score = metadata.get("importance_score", 5.0)
                        access_count = self.stats.get("access_counts", {}).get(memory_id, 0)
                        
                        final_score = self._calculate_ranking_score(
                            similarity_score,
                            importance_score,
                            access_count,
                            context_relevance,
                            topic_relevance,
                            time_relevance,
                            metadata.get("memory_type")
                        )
                        
                        results.append({
                            "memory_id": memory_id,
                            "content": content[:2000] if len(content) > 2000 else content,
                            "metadata": metadata,
                            "similarity_score": similarity_score,
                            "final_score": final_score,
                            "memory_type": metadata.get("memory_type"),
                            "tags": metadata.get("tags", []),
                            "emotional_intensity": metadata.get("emotional_intensity", 0.5),
                            "strategic_value": metadata.get("strategic_value", {}),
                            "linked_tool": metadata.get("linked_tool"),
                            "created_at": metadata.get("created_at"),
                            "access_count": access_count,
                            "importance_score": importance_score,
                            "context_relevance": context_relevance,
                            "topic_relevance": topic_relevance,
                            "time_relevance": time_relevance,
                            "key_info_relevance": key_info_relevance
                        })
                    
                    except json.JSONDecodeError as e:
                        print(f"[记忆系统] 解析JSONL行失败: {e}")
                        continue
                    except Exception as e:
                        print(f"[记忆系统] 处理持久化记忆项失败: {e}")
                        continue
            
            # 按最终分数排序并限制数量
            results.sort(key=lambda x: x.get("final_score", 0), reverse=True)
            return results[:limit]
        
        except Exception as e:
            print(f"[记忆系统] 从持久化存储搜索失败: {e}")
            return []
    
    def _fallback_search(self,
                        query: str,
                        memory_type: Optional[str] = None,
                        tags: Optional[List[str]] = None,
                        limit: int = 5,
                        similarity_threshold: float = 0.7) -> List[Dict[str, Any]]:
        """
        当memo0不可用或失败时的回退搜索方法
        
        Args:
            query: 原始查询
            memory_type: 记忆类型
            tags: 标签
            limit: 结果数量限制
            similarity_threshold: 相似度阈值
            
        Returns:
            记忆结果列表
        """
        try:
            # 构建上下文增强查询
            context_enhanced_query = self._build_context_enhanced_query(query)
            
            # 增强搜索查询
            enhanced_query = self._enhance_search_query(context_enhanced_query)
            
            results = []
            
            # 从内存存储搜索
            for memory_item in self.memory_store:
                item_results = self._search_in_memory_item(memory_item, enhanced_query, memory_type, tags)
                results.extend(item_results)
            
            # 从文件存储搜索（如果可用）
            if self._persistent_storage_path:
                file_results = self._search_in_persistent_storage(enhanced_query, memory_type, tags, limit * 2)
                # 合并结果并去重
                existing_ids = {r["memory_id"] for r in results}
                for result in file_results:
                    if result["memory_id"] not in existing_ids:
                        results.append(result)
                        existing_ids.add(result["memory_id"])
            
            # 应用相似度阈值过滤
            filtered_results = [
                r for r in results 
                if r.get("similarity_score", 0) >= similarity_threshold
            ]
            
            # 按最终分数排序
            filtered_results.sort(key=lambda x: x.get("final_score", 0), reverse=True)
            
            # 限制结果数量
            return filtered_results[:limit]
        
        except Exception as e:
            print(f"[记忆系统] 回退搜索失败: {e}")
            return []

    def _extract_content_from_memory(self, memory_data: Any) -> str:
        """从记忆数据中提取内容"""
        if isinstance(memory_data, list):
            # 处理对话格式的记忆
            content_parts = []
            for msg in memory_data:
                if isinstance(msg, dict) and "content" in msg:
                    content_parts.append(msg["content"])
            return " ".join(content_parts)
        elif isinstance(memory_data, dict) and "content" in memory_data:
            # 处理单个内容对象
            return str(memory_data["content"])
        else:
            # 直接返回字符串表示
            return str(memory_data)

    def get_memory_by_id(self, memory_id: str) -> Optional[Dict[str, Any]]:
        """根据ID获取记忆"""
        try:
            if not self.memory:
                # 使用简化的内存存储
                for memory_item in self.memory_store:
                    if memory_item["metadata"].get("memory_id") == memory_id:
                        return {
                            "memory_id": memory_id,
                            "content": self._extract_content_from_memory(memory_item["content"]),
                            "metadata": memory_item["metadata"]
                        }
                return None

            # 使用memo0 API获取记忆
            memories = self.memory.get(memory_id=memory_id, user_id=self.user_id)
            if memories and isinstance(memories, list) and len(memories) > 0:
                memory = memories[0]
                metadata = memory.get("metadata", {})
                return {
                    "memory_id": memory_id,
                    "content": self._extract_content_from_memory(memory.get("memory", {})),
                    "metadata": metadata
                }
            return None
        except Exception as e:
            print(f"[记忆系统] 根据ID获取记忆失败: {e}")
            return None

    def _evaluate_content_value(self, content: str) -> float:
        """评估内容的价值，返回1-10的评分"""
        try:
            # 使用记忆评估器进行评估
            result = self.memory_evaluator.evaluate(content)
            return result.overall_score
        except Exception as e:
            print(f"[记忆系统] 评估内容价值失败: {e}")
            # 默认返回中等价值
            return 5.0

    def _check_duplicate_content(self, content: str, memory_type: str) -> Optional[str]:
        """检查是否有重复内容"""
        # 简化的重复检查，实际项目中可以使用更复杂的算法
        return None

    def _calculate_importance_score(self, content: str, memory_type: str, metadata: Dict[str, Any]) -> float:
        """计算记忆的重要性分数"""
        # 基础分数
        base_score = 5.0
        
        # 类型权重
        type_weights = {
            "user_profile": 2.0,
            "knowledge": 1.8,
            "strategy": 1.6,
            "experience": 1.4,
            "emotion": 1.2,
            "conversation": 1.0,
            "system_event": 0.8,
            "temporal": 0.6
        }
        
        # 应用类型权重
        base_score *= type_weights.get(memory_type, 1.0)
        
        # 内容长度权重
        content_length = len(content)
        if content_length > 1000:
            base_score += 1.0
        elif content_length < 100:
            base_score -= 1.0
        
        # 情感强度权重
        if "emotional_intensity" in metadata and metadata["emotional_intensity"] > 0.7:
            base_score += 1.0
        
        # 策略价值权重
        if "strategic_score" in metadata and metadata["strategic_score"] > 0.7:
            base_score += 1.0
        
        # 归一化到1-10范围
        return min(max(base_score, 1.0), 10.0)

    def get_recent_conversations(self, session_id: Optional[str] = None, limit: int = 10) -> List[Dict[str, Any]]:
        """
        获取最近的对话（优先从短期记忆获取）
        
        优化版：增强对话历史的完整性和格式一致性，以支持话题连贯性
        
        Args:
            session_id: 会话ID，默认使用当前会话
            limit: 返回的最大对话数量
            
        Returns:
            对话历史列表，每个元素包含：
            - user_input: 用户输入
            - system_response: 系统回复
            - timestamp: 时间戳
            - topics: 对话主题（如果有）
        """
        results = []
        conversation_id = session_id or self.session_id
        
        # 1. 从短期记忆获取（最快、最准确）
        try:
            short_term_memories = self.search_short_term_memories(
                conversation_id=conversation_id,
                limit=limit * 2  # 获取更多以便过滤
            )
            
            for stm in short_term_memories:
                metadata = stm.metadata or {}
                # 优先使用metadata中的user_input和system_response，而不是content
                # 这样可以避免解析格式化文本
                user_input = metadata.get("user_input", "")
                system_response = metadata.get("system_response", "")
                
                # 如果metadata中没有，尝试从content解析
                if not user_input and not system_response:
                    content = stm.content
                    if isinstance(content, str):
                        # 尝试解析格式化的对话内容
                        if "用户:" in content or "用户：" in content:
                            lines = content.split("\n")
                            for line in lines:
                                line = line.strip()
                                if line.startswith("用户:") or line.startswith("用户："):
                                    user_input = line.replace("用户:", "").replace("用户：", "").strip()
                                elif line.startswith("AI:") or line.startswith("AI："):
                                    system_response = line.replace("AI:", "").replace("AI：", "").strip()
                
                # 只添加有效的对话（至少有用户输入或系统回复）
                if user_input or system_response:
                    # 构建格式化的content（用于向后兼容）
                    formatted_content = f"用户: {user_input}\nAI: {system_response}" if user_input and system_response else stm.content
                    
                    # 提取主题信息
                    topics = metadata.get("topics", [])
                    
                    results.append({
                        "content": formatted_content,
                        "user_input": user_input,  # 新增：直接提供user_input
                        "system_response": system_response,  # 新增：直接提供system_response
                        "timestamp": stm.created_at.isoformat(),  # 新增：直接提供timestamp
                        "topics": topics,  # 新增：对话主题
                        "metadata": {
                            "memory_id": stm.memory_id,
                            "created_at": stm.created_at.isoformat(),
                            "memory_type": "conversation",
                            "user_input": user_input,
                            "system_response": system_response,
                            "topics": topics,
                            **{k: v for k, v in metadata.items() if k not in ["user_input", "system_response", "topics"]}
                        }
                    })
        except Exception as e:
            # 短期记忆获取失败不影响整体流程
            print(f"[记忆系统] 获取短期记忆失败: {e}")
            pass
        
        # 2. 如果短期记忆不足，从长期记忆补充（使用时间排序而非空查询）
        if len(results) < limit:
            try:
                remaining_limit = limit - len(results)
                
                # 从内存存储获取对话类型的记忆，按时间排序
                conversation_memories = []
                # 使用user_input作为去重键，更准确
                existing_inputs = {r.get("user_input", r.get("content", "")) for r in results}
                
                # 从memory_store列表获取对话记忆
                for memory_item in self.memory_store:
                    if isinstance(memory_item, dict):
                        memory_type = memory_item.get("metadata", {}).get("memory_type", "")
                        if memory_type == "conversation":
                            conversation_memories.append(memory_item)
                
                # 按创建时间倒序排序
                conversation_memories.sort(
                    key=lambda x: x.get("metadata", {}).get("created_at", ""),
                    reverse=True
                )
                
                # 转换格式并添加（避免重复）
                for memory in conversation_memories[:remaining_limit * 2]:
                    content = memory.get("content", "")
                    metadata = memory.get("metadata", {})
                    
                    # 提取user_input和system_response
                    user_input = metadata.get("user_input", "")
                    system_response = metadata.get("system_response", "")
                    
                    # 如果metadata中没有，尝试从content解析
                    if not user_input and not system_response and isinstance(content, str):
                        if "用户:" in content or "用户：" in content:
                            lines = content.split("\n")
                            for line in lines:
                                line = line.strip()
                                if line.startswith("用户:") or line.startswith("用户："):
                                    user_input = line.replace("用户:", "").replace("用户：", "").strip()
                                elif line.startswith("AI:") or line.startswith("AI："):
                                    system_response = line.replace("AI:", "").replace("AI：", "").strip()
                    
                    # 检查是否重复
                    check_key = user_input if user_input else content
                    if check_key and check_key not in existing_inputs:
                        created_at = metadata.get("created_at", datetime.now().isoformat())
                        topics = metadata.get("topics", [])
                        
                        results.append({
                            "content": content if isinstance(content, str) else str(content),
                            "user_input": user_input,
                            "system_response": system_response,
                            "timestamp": created_at,
                            "topics": topics,
                            "metadata": {
                                "memory_id": metadata.get("memory_id", ""),
                                "created_at": created_at,
                                "memory_type": "conversation",
                                "user_input": user_input,
                                "system_response": system_response,
                                "topics": topics,
                                **{k: v for k, v in metadata.items() if k not in ["memory_id", "created_at", "memory_type", "user_input", "system_response", "topics"]}
                            }
                        })
                        existing_inputs.add(check_key)
                        if len(results) >= limit:
                            break
                
                # 如果memory_store不足，尝试使用memo0搜索（但不使用空查询）
                if len(results) < limit and self.memory:
                    try:
                        # 使用一个通用的对话查询词而不是空查询
                        memo0_results = self.memory.search(
                            "conversation dialogue chat",
                            user_id=self.user_id,
                            limit=remaining_limit,
                            filters={"memory_type": "conversation"},
                            rerank=False
                        )
                        
                        # 转换memo0结果格式
                        for result in memo0_results.get("results", []):
                            content = result.get("memory", "")
                            result_metadata = result.get("metadata", {})
                            user_input = result_metadata.get("user_input", "")
                            system_response = result_metadata.get("system_response", "")
                            
                            check_key = user_input if user_input else content
                            if check_key and check_key not in existing_inputs:
                                created_at = result.get("created_at", datetime.now().isoformat())
                                topics = result_metadata.get("topics", [])
                                
                                results.append({
                                    "content": content,
                                    "user_input": user_input,
                                    "system_response": system_response,
                                    "timestamp": created_at,
                                    "topics": topics,
                                    "metadata": {
                                        "memory_id": result.get("id", ""),
                                        "created_at": created_at,
                                        "memory_type": "conversation",
                                        "user_input": user_input,
                                        "system_response": system_response,
                                        "topics": topics,
                                        **{k: v for k, v in result_metadata.items() if k not in ["user_input", "system_response", "topics"]}
                                    }
                                })
                                existing_inputs.add(check_key)
                                if len(results) >= limit:
                                    break
                    except Exception as e:
                        # memo0搜索失败，至少返回memory_store的结果
                        print(f"[记忆系统] memo0搜索失败: {e}")
                        pass
            except Exception as e:
                # 长期记忆获取失败，至少返回短期记忆的结果
                print(f"[记忆系统] 获取长期记忆失败: {e}")
                pass
        
        # 按时间倒序排序（最新的在前）
        results.sort(key=lambda x: x.get("timestamp", x.get("metadata", {}).get("created_at", "")), reverse=True)
        
        return results[:limit]
    
    def find_resonant_memory(self, context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """寻找共鸣记忆
        
        Args:
            context: 包含用户输入、情绪、主题等信息的上下文字典
            
        Returns:
            共鸣记忆信息字典，包含触发的记忆、记忆ID、相关度评分等
        """
        try:
            # 从上下文中提取相关信息
            user_input = context.get("user_input", "")
            user_emotion = context.get("user_emotion", "")
            topic = context.get("topic", "")
            current_goal = context.get("current_goal", "")
            
            # 构建检索查询
            query_parts = [user_input, user_emotion, topic, current_goal]
            query = " ".join([part for part in query_parts if part])
            
            # 如果没有足够的查询信息，返回None
            if not query:
                return None
            
            # 使用search_memories方法查找相关记忆
            search_results = self.search_memories(
                query=query,
                limit=1,
                similarity_threshold=0.0
            )
            
            if search_results:
                # 获取第一个匹配的记忆
                memory = search_results[0]
                
                # 构建共鸣记忆响应
                resonant_memory = {
                    "triggered_memory": memory.get("content", ""),
                    "memory_id": memory.get("memory_id", ""),
                    "relevance_score": memory.get("similarity_score", 0.0),
                    "risk_assessment": {
                        "level": "low",
                        "high_risk_factors": []
                    },
                    "recommended_actions": ["使用回忆的信息回应用户", "保持上下文连贯性"]
                }
                
                return resonant_memory
            
            return None
        except Exception as e:
            print(f"[记忆系统] 寻找共鸣记忆失败: {e}")
            return None
