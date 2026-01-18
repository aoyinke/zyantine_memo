# ============ 统一索引管理 ============
"""
统一的索引管理模块，整合原来分散在 memory_manager.py 和 memory_store.py 中的索引功能。
提供主题索引、关键信息索引、时间索引、类型索引等。
"""
import threading
from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Any, Optional, Set

from .models import MemoryRecord, MemoryType, MemoryPriority


class MemoryIndex:
    """
    统一的记忆索引管理器
    
    整合所有索引功能，提供一致的索引维护和查询接口。
    """

    def __init__(self):
        """初始化索引管理器"""
        # 主题索引
        self._topic_index: Dict[str, Set[str]] = defaultdict(set)  # topic -> memory_ids
        self._memory_topics: Dict[str, List[str]] = {}  # memory_id -> topics
        
        # 关键信息索引
        self._key_info_index: Dict[str, Set[str]] = defaultdict(set)  # key_info -> memory_ids
        self._memory_key_info: Dict[str, List[str]] = {}  # memory_id -> key_info_list
        
        # 重要性索引
        self._importance_index: Dict[int, Set[str]] = defaultdict(set)  # importance_level -> memory_ids
        
        # 类型索引
        self._type_index: Dict[str, Set[str]] = defaultdict(set)  # memory_type -> memory_ids
        
        # 标签索引（语义索引）
        self._tag_index: Dict[str, Set[str]] = defaultdict(set)  # tag -> memory_ids
        self._memory_tags: Dict[str, List[str]] = {}  # memory_id -> tags
        
        # 时间索引
        self._temporal_index: Dict[str, Set[str]] = defaultdict(set)  # date_str -> memory_ids
        
        # 优先级索引
        self._priority_index: Dict[str, Set[str]] = defaultdict(set)  # priority -> memory_ids
        
        # 会话索引
        self._session_index: Dict[str, Set[str]] = defaultdict(set)  # session_id -> memory_ids
        
        # 主题关键词库
        self._topic_keywords: Dict[str, List[str]] = {
            "fitness": ["体测", "体检", "健身", "运动", "锻炼", "测试", "成绩", "跑步", "跳远", "肺活量"],
            "work": ["工作", "上班", "任务", "项目", "会议", "报告", "邮件", "客户", "同事"],
            "study": ["学习", "学校", "考试", "作业", "课程", "复习", "考试", "成绩", "论文"],
            "life": ["生活", "日常", "周末", "假期", "旅行", "美食", "电影", "音乐"],
            "technology": ["编程", "代码", "软件", "开发", "技术", "系统", "数据", "算法"],
            "health": ["健康", "医院", "医生", "药物", "治疗", "症状", "检查"],
            "finance": ["金融", "投资", "股票", "基金", "理财", "银行", "贷款"],
        }
        
        self._lock = threading.RLock()

    # ============ 索引记忆 ============

    def index_memory(self, record: MemoryRecord) -> None:
        """
        为记忆建立所有索引
        
        Args:
            record: 记忆记录
        """
        with self._lock:
            memory_id = record.memory_id
            
            # 类型索引
            self._type_index[record.memory_type.value].add(memory_id)
            
            # 优先级索引
            self._priority_index[record.priority.value].add(memory_id)
            
            # 标签索引
            self._memory_tags[memory_id] = record.tags.copy()
            for tag in record.tags:
                self._tag_index[tag].add(memory_id)
            
            # 时间索引
            date_str = record.created_at.strftime("%Y-%m-%d")
            self._temporal_index[date_str].add(memory_id)
            
            # 会话索引
            session_id = record.metadata.get("session_id")
            if session_id:
                self._session_index[session_id].add(memory_id)
            
            # 主题索引（从内容检测）
            content_str = str(record.content) if not isinstance(record.content, str) else record.content
            topics = self._detect_topics(content_str)
            if topics:
                self._memory_topics[memory_id] = topics
                for topic in topics:
                    self._topic_index[topic].add(memory_id)
            
            # 关键信息索引
            key_info = record.metadata.get("key_information", [])
            if key_info:
                self._memory_key_info[memory_id] = key_info
                for info in key_info:
                    self._key_info_index[info].add(memory_id)
            
            # 重要性索引
            importance_score = record.metadata.get("importance_score", 5.0)
            importance_level = int(round(importance_score))
            self._importance_index[importance_level].add(memory_id)

    def remove_memory(self, memory_id: str) -> None:
        """
        从所有索引中移除记忆
        
        Args:
            memory_id: 记忆ID
        """
        with self._lock:
            # 从类型索引移除
            for type_memories in self._type_index.values():
                type_memories.discard(memory_id)
            
            # 从优先级索引移除
            for priority_memories in self._priority_index.values():
                priority_memories.discard(memory_id)
            
            # 从标签索引移除
            if memory_id in self._memory_tags:
                for tag in self._memory_tags[memory_id]:
                    self._tag_index[tag].discard(memory_id)
                del self._memory_tags[memory_id]
            
            # 从时间索引移除
            for date_memories in self._temporal_index.values():
                date_memories.discard(memory_id)
            
            # 从会话索引移除
            for session_memories in self._session_index.values():
                session_memories.discard(memory_id)
            
            # 从主题索引移除
            if memory_id in self._memory_topics:
                for topic in self._memory_topics[memory_id]:
                    self._topic_index[topic].discard(memory_id)
                del self._memory_topics[memory_id]
            
            # 从关键信息索引移除
            if memory_id in self._memory_key_info:
                for info in self._memory_key_info[memory_id]:
                    self._key_info_index[info].discard(memory_id)
                del self._memory_key_info[memory_id]
            
            # 从重要性索引移除
            for importance_memories in self._importance_index.values():
                importance_memories.discard(memory_id)

    def update_memory(self, record: MemoryRecord) -> None:
        """
        更新记忆的索引
        
        Args:
            record: 更新后的记忆记录
        """
        with self._lock:
            self.remove_memory(record.memory_id)
            self.index_memory(record)

    # ============ 查询方法 ============

    def get_by_type(self, memory_type: str) -> Set[str]:
        """根据类型获取记忆ID"""
        with self._lock:
            return self._type_index.get(memory_type, set()).copy()

    def get_by_priority(self, priority: str) -> Set[str]:
        """根据优先级获取记忆ID"""
        with self._lock:
            return self._priority_index.get(priority, set()).copy()

    def get_by_tag(self, tag: str) -> Set[str]:
        """根据标签获取记忆ID"""
        with self._lock:
            return self._tag_index.get(tag, set()).copy()

    def get_by_tags(self, tags: List[str], match_all: bool = False) -> Set[str]:
        """
        根据多个标签获取记忆ID
        
        Args:
            tags: 标签列表
            match_all: 是否要求匹配所有标签
        """
        with self._lock:
            if not tags:
                return set()
            
            result_sets = [self._tag_index.get(tag, set()) for tag in tags]
            
            if match_all:
                return set.intersection(*result_sets) if result_sets else set()
            else:
                return set.union(*result_sets) if result_sets else set()

    def get_by_date(self, date_str: str) -> Set[str]:
        """根据日期获取记忆ID"""
        with self._lock:
            return self._temporal_index.get(date_str, set()).copy()

    def get_by_date_range(self, start_date: str, end_date: str) -> Set[str]:
        """根据日期范围获取记忆ID"""
        with self._lock:
            result = set()
            for date_str, memory_ids in self._temporal_index.items():
                if start_date <= date_str <= end_date:
                    result.update(memory_ids)
            return result

    def get_by_session(self, session_id: str) -> Set[str]:
        """根据会话ID获取记忆ID"""
        with self._lock:
            return self._session_index.get(session_id, set()).copy()

    def get_by_topic(self, topic: str) -> Set[str]:
        """根据主题获取记忆ID"""
        with self._lock:
            return self._topic_index.get(topic, set()).copy()

    def get_by_topics(self, topics: List[str], match_all: bool = False) -> Set[str]:
        """根据多个主题获取记忆ID"""
        with self._lock:
            if not topics:
                return set()
            
            result_sets = [self._topic_index.get(topic, set()) for topic in topics]
            
            if match_all:
                return set.intersection(*result_sets) if result_sets else set()
            else:
                return set.union(*result_sets) if result_sets else set()

    def get_by_key_info(self, key_info: str) -> Set[str]:
        """根据关键信息获取记忆ID"""
        with self._lock:
            return self._key_info_index.get(key_info, set()).copy()

    def get_by_importance(self, min_level: int = 0, max_level: int = 10) -> Set[str]:
        """根据重要性级别范围获取记忆ID"""
        with self._lock:
            result = set()
            for level in range(min_level, max_level + 1):
                result.update(self._importance_index.get(level, set()))
            return result

    def get_topics_for_memory(self, memory_id: str) -> List[str]:
        """获取记忆的主题列表"""
        with self._lock:
            return self._memory_topics.get(memory_id, []).copy()

    def get_tags_for_memory(self, memory_id: str) -> List[str]:
        """获取记忆的标签列表"""
        with self._lock:
            return self._memory_tags.get(memory_id, []).copy()

    def get_key_info_for_memory(self, memory_id: str) -> List[str]:
        """获取记忆的关键信息列表"""
        with self._lock:
            return self._memory_key_info.get(memory_id, []).copy()

    # ============ 主题检测 ============

    def _detect_topics(self, content: str) -> List[str]:
        """检测内容的主题"""
        detected_topics = []
        content_lower = content.lower()
        
        for topic, keywords in self._topic_keywords.items():
            for keyword in keywords:
                if keyword in content_lower:
                    if topic not in detected_topics:
                        detected_topics.append(topic)
                    break
        
        return detected_topics

    def add_topic_keywords(self, topic: str, keywords: List[str]) -> None:
        """添加主题关键词"""
        with self._lock:
            if topic in self._topic_keywords:
                self._topic_keywords[topic].extend(keywords)
                self._topic_keywords[topic] = list(set(self._topic_keywords[topic]))
            else:
                self._topic_keywords[topic] = keywords

    def get_topic_keywords(self) -> Dict[str, List[str]]:
        """获取所有主题关键词"""
        with self._lock:
            return {k: v.copy() for k, v in self._topic_keywords.items()}

    # ============ 统计信息 ============

    def get_stats(self) -> Dict[str, Any]:
        """获取索引统计信息"""
        with self._lock:
            return {
                "total_indexed_memories": len(set.union(*self._type_index.values())) if self._type_index else 0,
                "type_distribution": {k: len(v) for k, v in self._type_index.items()},
                "priority_distribution": {k: len(v) for k, v in self._priority_index.items()},
                "topic_distribution": {k: len(v) for k, v in self._topic_index.items()},
                "tag_count": len(self._tag_index),
                "topic_count": len(self._topic_index),
                "key_info_count": len(self._key_info_index),
                "session_count": len(self._session_index),
                "date_count": len(self._temporal_index),
            }

    def clear(self) -> None:
        """清空所有索引"""
        with self._lock:
            self._topic_index.clear()
            self._memory_topics.clear()
            self._key_info_index.clear()
            self._memory_key_info.clear()
            self._importance_index.clear()
            self._type_index.clear()
            self._tag_index.clear()
            self._memory_tags.clear()
            self._temporal_index.clear()
            self._priority_index.clear()
            self._session_index.clear()


class ContextWindow:
    """
    上下文窗口管理器
    
    管理当前对话的上下文记忆，支持动态调整窗口大小。
    """

    def __init__(self,
                 max_size: int = 20,
                 min_size: int = 5,
                 dynamic_weight: float = 0.8):
        """
        初始化上下文窗口
        
        Args:
            max_size: 最大窗口大小
            min_size: 最小窗口大小
            dynamic_weight: 动态调整权重
        """
        self._window: List[str] = []
        self._max_size = max_size
        self._min_size = min_size
        self._current_size = (max_size + min_size) // 2
        self._dynamic_weight = dynamic_weight
        
        # 当前对话主题
        self._current_topics: Dict[str, float] = {}
        self._topic_decay_rate = 0.9
        
        # 短期记忆标记
        self._short_term_memories: Set[str] = set()
        
        self._lock = threading.RLock()

    def add(self, memory_id: str, topics: Optional[List[str]] = None) -> None:
        """
        添加记忆到上下文窗口
        
        Args:
            memory_id: 记忆ID
            topics: 相关主题
        """
        with self._lock:
            # 如果已存在，移到末尾
            if memory_id in self._window:
                self._window.remove(memory_id)
            
            self._window.append(memory_id)
            self._short_term_memories.add(memory_id)
            
            # 更新主题
            if topics:
                self._update_topics(topics)
            
            # 动态调整窗口大小
            self._adjust_size()
            
            # 如果超过当前大小，移除最旧的
            while len(self._window) > self._current_size:
                removed = self._window.pop(0)
                self._short_term_memories.discard(removed)

    def get_recent(self, limit: Optional[int] = None) -> List[str]:
        """获取最近的记忆ID"""
        with self._lock:
            if limit:
                return self._window[-limit:]
            return self._window.copy()

    def get_current_topics(self) -> Dict[str, float]:
        """获取当前对话主题及权重"""
        with self._lock:
            return self._current_topics.copy()

    def is_short_term(self, memory_id: str) -> bool:
        """检查是否是短期记忆"""
        with self._lock:
            return memory_id in self._short_term_memories

    def clear(self) -> None:
        """清空上下文窗口"""
        with self._lock:
            self._window.clear()
            self._current_topics.clear()
            self._short_term_memories.clear()

    def _update_topics(self, topics: List[str]) -> None:
        """更新当前对话主题"""
        # 衰减现有主题权重
        for topic in self._current_topics:
            self._current_topics[topic] *= self._topic_decay_rate
        
        # 移除权重过低的主题
        self._current_topics = {k: v for k, v in self._current_topics.items() if v > 0.1}
        
        # 更新新主题权重
        for topic in topics:
            if topic in self._current_topics:
                self._current_topics[topic] += 0.5
            else:
                self._current_topics[topic] = 1.0
        
        # 归一化
        total_weight = sum(self._current_topics.values())
        if total_weight > 0:
            for topic in self._current_topics:
                self._current_topics[topic] /= total_weight

    def _adjust_size(self) -> None:
        """动态调整窗口大小"""
        if len(self._window) > self._current_size * 1.5:
            # 对话活跃，增大窗口
            self._current_size = min(self._current_size + 2, self._max_size)
        elif len(self._window) < self._current_size * 0.5:
            # 对话不活跃，缩小窗口
            self._current_size = max(self._current_size - 1, self._min_size)

    def get_stats(self) -> Dict[str, Any]:
        """获取上下文窗口统计"""
        with self._lock:
            return {
                "window_size": len(self._window),
                "current_max_size": self._current_size,
                "max_size": self._max_size,
                "min_size": self._min_size,
                "short_term_count": len(self._short_term_memories),
                "topic_count": len(self._current_topics),
                "top_topics": sorted(
                    self._current_topics.items(),
                    key=lambda x: x[1],
                    reverse=True
                )[:5]
            }
