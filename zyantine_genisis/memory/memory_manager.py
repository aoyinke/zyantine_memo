"""
记忆管理器 - 管理所有记忆相关操作
"""
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
import json
from dataclasses import dataclass
from enum import Enum

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


class MemoryPriority(Enum):
    """记忆优先级"""
    HIGH = "high"  # 高频访问，重要记忆
    MEDIUM = "medium"  # 常规记忆
    LOW = "low"  # 低频访问，可压缩
    ARCHIVE = "archive"  # 归档记忆


@dataclass
class MemoryRecord:
    """记忆记录数据类"""
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


class MemoryManager:
    """集中管理所有记忆操作"""

    def __init__(self, config: Optional[MemoryConfig] = None):
        self.config = config or ConfigManager().get().memory
        self.session_id = ConfigManager().get().session_id
        self.user_id = ConfigManager().get().user_id

        # 初始化记忆系统
        self.memory_system = self._initialize_memory_system()

        # 缓存
        self.recent_memories: Dict[str, MemoryRecord] = {}
        self.semantic_index: Dict[str, List[str]] = {}  # 标签->记忆ID索引

        # 统计
        self.stats = {
            "total_memories": 0,
            "memory_by_type": {},
            "memory_by_priority": {},
            "average_emotional_intensity": 0.0,
            "average_strategic_score": 0.0
        }

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
                   priority: Union[str, MemoryPriority] = MemoryPriority.MEDIUM) -> str:
        """添加记忆"""
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
            "added_by": "memory_manager"
        })

        # 添加记忆到存储系统
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
            strategic_score=strategic_value.get("score", 0.0) if strategic_value else 0.0
        )

        # 更新缓存和索引
        self._update_cache(memory_record)
        self._update_semantic_index(memory_record)
        self._update_stats(memory_record)

        return memory_id

    def _update_cache(self, memory_record: MemoryRecord):
        """更新缓存"""
        # 保持缓存大小
        if len(self.recent_memories) > 1000:
            # 移除最不常访问的记忆
            oldest = min(self.recent_memories.items(),
                         key=lambda x: x[1].access_count)
            del self.recent_memories[oldest[0]]

        self.recent_memories[memory_record.memory_id] = memory_record

    def _update_semantic_index(self, memory_record: MemoryRecord):
        """更新语义索引"""
        for tag in memory_record.tags:
            if tag not in self.semantic_index:
                self.semantic_index[tag] = []

            # 保持索引有序（按战略分数）
            if memory_record.memory_id not in self.semantic_index[tag]:
                self.semantic_index[tag].append(memory_record.memory_id)
                # 按战略分数排序
                self.semantic_index[tag].sort(
                    key=lambda mem_id: self.recent_memories.get(mem_id,
                                                                MemoryRecord("", "", MemoryType.CONVERSATION,
                                                                             {}, [], MemoryPriority.MEDIUM,
                                                                             datetime.now())).strategic_score,
                    reverse=True
                )

    def _update_stats(self, memory_record: MemoryRecord):
        """更新统计信息"""
        self.stats["total_memories"] += 1

        # 按类型统计
        mem_type = memory_record.memory_type.value
        self.stats["memory_by_type"][mem_type] = \
            self.stats["memory_by_type"].get(mem_type, 0) + 1

        # 按优先级统计
        priority = memory_record.priority.value
        self.stats["memory_by_priority"][priority] = \
            self.stats["memory_by_priority"].get(priority, 0) + 1

        # 更新平均值
        current_avg_ei = self.stats["average_emotional_intensity"]
        current_avg_ss = self.stats["average_strategic_score"]

        self.stats["average_emotional_intensity"] = \
            (current_avg_ei * (self.stats["total_memories"] - 1) +
             memory_record.emotional_intensity) / self.stats["total_memories"]

        self.stats["average_strategic_score"] = \
            (current_avg_ss * (self.stats["total_memories"] - 1) +
             memory_record.strategic_score) / self.stats["total_memories"]

    def search_memories(self,
                        query: str,
                        memory_type: Optional[Union[str, MemoryType]] = None,
                        tags: Optional[List[str]] = None,
                        priority: Optional[Union[str, MemoryPriority]] = None,
                        limit: int = 5,
                        use_cache: bool = True) -> List[MemoryRecord]:
        """搜索记忆"""
        # 首先尝试从缓存和索引中查找
        if use_cache and tags:
            cached_results = self._search_by_tags(tags, memory_type, priority, limit)
            if cached_results:
                return cached_results

        # 使用记忆系统搜索
        search_results = self.memory_system.search_memories(
            query=query,
            memory_type=memory_type.value if isinstance(memory_type, MemoryType) else memory_type,
            tags=tags,
            limit=limit * 2,  # 获取更多结果用于过滤
            rerank=True
        )

        # 转换为MemoryRecord并过滤
        records = []
        for result in search_results:
            # 优先级过滤
            if priority:
                result_priority = result.get("metadata", {}).get("priority", "medium")
                if isinstance(priority, MemoryPriority):
                    priority_str = priority.value
                else:
                    priority_str = priority

                if result_priority != priority_str:
                    continue

            # 创建记录
            record = self._create_record_from_search_result(result)
            if record:
                records.append(record)

        # 按相关性排序
        records.sort(key=lambda x: x.metadata.get("similarity_score", 0), reverse=True)

        return records[:limit]

    def _search_by_tags(self,
                        tags: List[str],
                        memory_type: Optional[MemoryType],
                        priority: Optional[MemoryPriority],
                        limit: int) -> List[MemoryRecord]:
        """通过标签搜索记忆"""
        memory_ids = set()

        for tag in tags:
            if tag in self.semantic_index:
                memory_ids.update(self.semantic_index[tag][:limit * 2])

        # 获取记录并过滤
        records = []
        for mem_id in memory_ids:
            if mem_id in self.recent_memories:
                record = self.recent_memories[mem_id]

                # 类型过滤
                if memory_type and record.memory_type != memory_type:
                    continue

                # 优先级过滤
                if priority and record.priority != priority:
                    continue

                records.append(record)

        # 按战略分数排序
        records.sort(key=lambda x: x.strategic_score, reverse=True)
        return records[:limit]

    def _create_record_from_search_result(self, result: Dict) -> Optional[MemoryRecord]:
        """从搜索结果创建记忆记录"""
        try:
            metadata = result.get("metadata", {})

            # 获取或生成记忆ID
            memory_id = metadata.get("memory_id") or \
                        hashlib.md5(json.dumps(result).encode()).hexdigest()[:16]

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

            # 更新访问信息
            if memory_id in self.recent_memories:
                record = self.recent_memories[memory_id]
                record.access_count += 1
                record.last_accessed = datetime.now()
                return record

            # 创建新记录
            return MemoryRecord(
                memory_id=memory_id,
                content=result.get("content", ""),
                memory_type=memory_type,
                metadata=metadata,
                tags=metadata.get("tags", []),
                priority=priority,
                created_at=datetime.fromisoformat(metadata.get("created_at",
                                                               datetime.now().isoformat())),
                emotional_intensity=metadata.get("emotional_intensity", 0.5),
                strategic_score=metadata.get("strategic_score", 0.0)
            )
        except Exception as e:
            print(f"创建记忆记录失败: {e}")
            return None

    def find_resonant_memory(self, context: Dict) -> Optional[Dict]:
        """寻找共鸣记忆（包装原方法）"""
        return self.memory_system.find_resonant_memory(context)

    def get_conversation_history(self, limit: int = 100) -> List[Dict]:
        """获取对话历史"""
        # 尝试从记忆系统加载
        conversations = self.memory_system.find_conversations(
            query="最近的对话",
            session_id=self.session_id,
            limit=limit
        )

        # 转换为标准格式
        history = []
        for conv in conversations:
            history.append({
                "timestamp": conv.get("metadata", {}).get("created_at",
                                                          datetime.now().isoformat()),
                "user_input": self._extract_user_input(conv.get("content", "")),
                "system_response": self._extract_system_response(conv.get("content", "")),
                "context": {},
                "vector_state": {}
            })

        return history

    def record_interaction(self, interaction_data: Dict) -> bool:
        """记录交互"""
        try:
            # 安全地获取字典的键列表
            def get_keys_safely(data, key):
                value = data.get(key)
                if isinstance(value, dict):
                    return list(value.keys())
                elif isinstance(value, list):
                    return [f"item_{i}" for i in range(len(value))]
                elif value is not None:
                    return [str(value)]
                else:
                    return []

            # 添加到记忆系统
            memory_id = self.add_memory(
                content=[
                    {"role": "user", "content": interaction_data["user_input"]},
                    {"role": "assistant", "content": interaction_data["system_response"]}
                ],
                memory_type=MemoryType.CONVERSATION,
                tags=["对话", "交互"],
                emotional_intensity=interaction_data.get("emotional_intensity", 0.5),
                metadata={
                    "interaction_id": interaction_data.get("interaction_id"),
                    "context_keys": get_keys_safely(interaction_data, "context"),
                    "action_plan_keys": get_keys_safely(interaction_data, "action_plan"),
                    "growth_result_keys": get_keys_safely(interaction_data, "growth_result"),
                    "retrieved_memories_count": interaction_data.get("retrieved_memories_count", 0),
                    "has_resonant_memory": interaction_data.get("resonant_memory", False),
                    "has_cognitive_result": interaction_data.get("cognitive_result", False),
                    "has_growth_result": interaction_data.get("growth_result",
                                                              False) is not False and interaction_data.get(
                        "growth_result") is not None
                }
            )

            return True
        except Exception as e:
            print(f"记录交互失败: {e}")
            return False

    def cleanup_memory(self,
                       max_memories: int = 10000,
                       keep_high_priority: bool = True):
        """清理记忆"""
        # 获取所有记忆的统计信息
        stats = self.get_statistics()

        if stats["total_memories"] <= max_memories:
            return

        # 清理逻辑（这里简化为清除缓存）
        if len(self.recent_memories) > 1000:
            # 保留高频访问和高优先级的记忆
            to_keep = sorted(
                self.recent_memories.items(),
                key=lambda x: (x[1].access_count,
                               1 if x[1].priority == MemoryPriority.HIGH else 0),
                reverse=True
            )[:1000]

            self.recent_memories = dict(to_keep)

            # 重建语义索引
            self.semantic_index.clear()
            for record in self.recent_memories.values():
                self._update_semantic_index(record)

    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        # 获取记忆系统的统计信息
        system_stats = self.memory_system.get_statistics()

        # 合并统计信息
        return {
            **self.stats,
            "memory_system_stats": system_stats,
            "cache_size": len(self.recent_memories),
            "index_size": len(self.semantic_index),
            "session_id": self.session_id,
            "user_id": self.user_id
        }

    def export_memories(self,
                        file_path: str,
                        format_type: str = "json",
                        include_cache: bool = False) -> bool:
        """导出记忆"""
        return self.memory_system.export_memories(file_path, format_type)

    def test_connection(self) -> bool:
        """测试连接"""
        return self.memory_system.test_connection()

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
                    return item.get("content", "")[:100]
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