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


class MemoryPriority(Enum):
    """记忆优先级"""
    CRITICAL = "critical"  # 关键记忆，永不删除
    HIGH = "high"  # 高频访问，重要记忆
    MEDIUM = "medium"  # 常规记忆
    LOW = "low"  # 低频访问，可压缩
    ARCHIVE = "archive"  # 归档记忆


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

    def to_dict(self) -> Dict:
        """转换为字典"""
        data = asdict(self)
        data['memory_type'] = self.memory_type.value
        data['priority'] = self.priority.value
        data['created_at'] = self.created_at.isoformat()
        if self.last_accessed:
            data['last_accessed'] = self.last_accessed.isoformat()
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
    """记忆缓存管理类"""

    def __init__(self, max_size: int = 1000, ttl_hours: int = 24):
        self.max_size = max_size
        self.ttl_seconds = ttl_hours * 3600
        self.cache: Dict[str, MemoryRecord] = {}
        self.access_times: Dict[str, float] = {}
        self.lock = threading.RLock()

    def get(self, memory_id: str) -> Optional[MemoryRecord]:
        """获取缓存项"""
        with self.lock:
            if memory_id in self.cache:
                record = self.cache[memory_id]
                # 检查是否过期
                if time.time() - self.access_times[memory_id] > self.ttl_seconds:
                    del self.cache[memory_id]
                    del self.access_times[memory_id]
                    return None

                # 更新访问时间
                record.last_accessed = datetime.now()
                record.access_count += 1
                self.access_times[memory_id] = time.time()
                return record
        return None

    def set(self, memory_id: str, record: MemoryRecord):
        """设置缓存项"""
        with self.lock:
            # 如果缓存已满，淘汰最少使用的
            if len(self.cache) >= self.max_size and memory_id not in self.cache:
                self._evict_least_used()

            self.cache[memory_id] = record
            self.access_times[memory_id] = time.time()

    def delete(self, memory_id: str):
        """删除缓存项"""
        with self.lock:
            if memory_id in self.cache:
                del self.cache[memory_id]
                del self.access_times[memory_id]

    def clear(self):
        """清空缓存"""
        with self.lock:
            self.cache.clear()
            self.access_times.clear()

    def _evict_least_used(self):
        """淘汰最少使用的缓存项"""
        if not self.cache:
            return

        # 找到访问时间最久的项目
        oldest_id = min(self.access_times.items(), key=lambda x: x[1])[0]
        del self.cache[oldest_id]
        del self.access_times[oldest_id]


class MemoryManager:
    """集中管理所有记忆操作（优化版）"""

    def __init__(self, config: Optional[MemoryConfig] = None):
        self.config = config or ConfigManager().get().memory
        self.session_id = ConfigManager().get().session_id
        self.user_id = ConfigManager().get().user_id

        # 初始化记忆系统
        self.memory_system = self._initialize_memory_system()

        # 缓存系统
        self.cache = MemoryCache(
            max_size=self.config.cache_size if hasattr(self.config, 'cache_size') else 1000,
            ttl_hours=self.config.cache_ttl_hours if hasattr(self.config, 'cache_ttl_hours') else 24
        )

        # 语义索引
        self.semantic_index: Dict[str, List[str]] = defaultdict(list)
        self.reverse_index: Dict[str, List[str]] = defaultdict(list)  # memory_id -> tags

        # 时间索引
        self.temporal_index: Dict[str, List[str]] = defaultdict(list)  # date_str -> memory_ids

        # 统计信息
        self.stats = {
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
        self._access_count = 0
        self._cache_hits = 0
        self._total_response_time = 0.0

        # 锁
        self._stats_lock = threading.Lock()
        self._index_lock = threading.RLock()

        # 后台清理线程
        self._cleanup_thread = threading.Thread(target=self._background_cleanup, daemon=True)
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
                   content: Union[str, List[Dict]],
                   memory_type: Union[str, MemoryType],
                   tags: Optional[List[str]] = None,
                   emotional_intensity: float = 0.5,
                   strategic_value: Optional[Dict] = None,
                   metadata: Optional[Dict] = None,
                   priority: Union[str, MemoryPriority] = MemoryPriority.MEDIUM,
                   ttl_hours: Optional[int] = None) -> str:
        """添加记忆（带TTL支持）"""
        start_time = time.time()

        try:
            # 处理枚举类型
            if isinstance(memory_type, str):
                try:
                    memory_type_enum = MemoryType(memory_type)
                except ValueError:
                    memory_type_enum = MemoryType.CONVERSATION
            else:
                memory_type_enum = memory_type

            if isinstance(priority, str):
                try:
                    priority_enum = MemoryPriority(priority)
                except ValueError:
                    priority_enum = MemoryPriority.MEDIUM
            else:
                priority_enum = priority

            # 准备元数据
            full_metadata = metadata or {}
            full_metadata.update({
                "priority": priority_enum.value,
                "added_by": "memory_manager",
                "estimated_size": len(str(content))
            })

            # 添加TTL信息
            if ttl_hours:
                expiry_time = datetime.now() + timedelta(hours=ttl_hours)
                full_metadata["expires_at"] = expiry_time.isoformat()

            # 添加到记忆系统
            memory_id = self.memory_system.add_memory(
                content=content,
                memory_type=memory_type_enum.value,
                tags=tags or [],
                emotional_intensity=emotional_intensity,
                strategic_value=strategic_value or {},
                metadata=full_metadata
            )

            # 创建记忆记录
            memory_record = MemoryRecord(
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

            response_time = (time.time() - start_time) * 1000
            self._record_response_time(response_time)

            return memory_id

        except Exception as e:
            print(f"[记忆管理器] 添加记忆失败: {e}")
            raise

    def _update_cache(self, memory_record: MemoryRecord):
        """更新缓存"""
        self.cache.set(memory_record.memory_id, memory_record)

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
        """搜索记忆（优化版）"""
        start_time = time.time()

        try:
            # 首先尝试从缓存中查找
            if use_cache and tags:
                cached_results = self._search_by_tags(tags, memory_type, priority, limit)
                if cached_results:
                    self._cache_hits += 1
                    return cached_results

            # 使用记忆系统搜索
            search_results = self.memory_system.search_memories(
                query=query,
                memory_type=memory_type.value if isinstance(memory_type, MemoryType) else memory_type,
                tags=tags,
                limit=limit * 2,
                rerank=True,
                similarity_threshold=similarity_threshold
            )

            # 转换为MemoryRecord并过滤
            records = []
            for result in search_results:
                record = self._create_or_update_record_from_search(result)
                if not record:
                    continue

                # 优先级过滤
                if priority and record.priority != priority:
                    continue

                # 类型过滤
                if memory_type and record.memory_type != memory_type:
                    continue

                records.append(record)

            # 按相关度排序
            records.sort(key=lambda x: x.relevance_score, reverse=True)

            response_time = (time.time() - start_time) * 1000
            self._record_response_time(response_time)
            self._access_count += 1

            return records[:limit]

        except Exception as e:
            print(f"[记忆管理器] 搜索记忆失败: {e}")
            return []

    def _search_by_tags(self,
                        tags: List[str],
                        memory_type: Optional[Union[str, MemoryType]],
                        priority: Optional[Union[str, MemoryPriority]],
                        limit: int) -> List[MemoryRecord]:
        """通过标签搜索记忆（优化版）"""
        with self._index_lock:
            memory_ids = set()

            # 计算每个记忆的标签匹配度
            memory_scores = defaultdict(int)
            for tag in tags:
                if tag in self.semantic_index:
                    for mem_id in self.semantic_index[tag]:
                        memory_scores[mem_id] += 1

            # 获取记录并过滤
            records = []
            for mem_id, score in sorted(memory_scores.items(), key=lambda x: x[1], reverse=True):
                record = self._get_memory_record(mem_id)
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
        record = self.cache.get(memory_id)
        if record:
            return record

        # 从记忆系统获取
        try:
            memory_data = self.memory_system.get_memory(memory_id, include_full_data=False)
            if memory_data:
                record = self._create_record_from_memory_data(memory_data)
                if record:
                    self.cache.set(memory_id, record)
                return record
        except Exception as e:
            print(f"[记忆管理器] 获取记忆失败 {memory_id}: {e}")

        return None

    def _create_or_update_record_from_search(self, result: Dict) -> Optional[MemoryRecord]:
        """从搜索结果创建或更新记忆记录"""
        try:
            metadata = result.get("metadata", {})
            memory_id = metadata.get("memory_id") or \
                        hashlib.md5(json.dumps(result, sort_keys=True).encode()).hexdigest()[:16]

            # 检查缓存
            existing_record = self.cache.get(memory_id)
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

    def _create_record_from_memory_data(self, memory_data: Dict) -> Optional[MemoryRecord]:
        """从记忆数据创建记忆记录"""
        try:
            metadata = memory_data.get("metadata", {})

            # 获取或生成记忆ID
            memory_id = metadata.get("memory_id") or \
                        hashlib.md5(json.dumps(memory_data, sort_keys=True).encode()).hexdigest()[:16]

            # 确定记忆类型
            memory_type_str = metadata.get("memory_type", "conversation")
            try:
                memory_type = MemoryType(memory_type_str)
            except ValueError:
                memory_type = MemoryType.CONVERSATION

            # 确定优先级
            priority_str = metadata.get("priority", "medium")
            try:
                priority = MemoryPriority(priority_str)
            except ValueError:
                priority = MemoryPriority.MEDIUM

            # 检查是否过期
            if "expires_at" in metadata:
                expires_at = datetime.fromisoformat(metadata["expires_at"])
                if datetime.now() > expires_at:
                    print(f"[记忆管理器] 记忆已过期: {memory_id}")
                    return None

            # 创建记录
            record = MemoryRecord(
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

    def find_resonant_memory(self, context: Dict) -> Optional[Dict]:
        """寻找共鸣记忆"""
        start_time = time.time()

        try:
            result = self.memory_system.find_resonant_memory(context)

            response_time = (time.time() - start_time) * 1000
            self._record_response_time(response_time)

            return result
        except Exception as e:
            print(f"[记忆管理器] 寻找共鸣记忆失败: {e}")
            return None

    def get_conversation_history(self, limit: int = 100,
                                 start_date: Optional[datetime] = None,
                                 end_date: Optional[datetime] = None) -> List[Dict]:
        """获取对话历史（支持时间范围）"""
        # 首先尝试从缓存中获取
        cache_key = f"conversation_history_{limit}_{start_date}_{end_date}"
        cached = self.cache.get(cache_key)
        if cached and isinstance(cached.content, list):
            return cached.content[:limit]

        # 从记忆系统加载
        conversations = self.memory_system.find_conversations(
            query="最近的对话",
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
            return content[:100] + "..." if len(content) > 100 else content
        elif isinstance(content, list):
            for item in content:
                if isinstance(item, dict) and item.get("role") == "user":
                    content_str = item.get("content", "")
                    return content_str[:100] + "..." if len(content_str) > 100 else content_str
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