# ============ 基于memo0的简化记忆系统 ============
import os
import json
import hashlib
from datetime import datetime
from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum

# Mem0框架导入
from mem0 import Memory
from openai import OpenAI

# ============ 常量定义 ============
DEFAULT_BASE_URL = "https://openkey.cloud/v1"
DEFAULT_API_KEY = "sk-wiHpoarpNTHaep0t54852a32A75a4d6986108b3f6eF7B7B9"


# ============ 类型枚举 ============
class MemoryType(Enum):
    """记忆类型枚举"""
    CONVERSATION = "conversation"
    EXPERIENCE = "experience"
    VECTOR_STATE = "vector_state"
    USER_PROFILE = "user_profile"
    SYSTEM_PROFILE = "system_profile"
    INSIGHT = "insight"


@dataclass
class MemoryMetadata:
    """记忆元数据"""
    memory_id: str
    memory_type: str
    text: str
    tags: List[str] = field(default_factory=list)
    source: str = "unknown"
    emotional_intensity: float = 0.5
    strategic_value: Dict[str, Any] = field(default_factory=dict)
    linked_tool: Optional[str] = None
    access_count: int = 0
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())
    session_id: str = "default"
    user_id: str = "default"


# ============ 记忆系统核心类 ============
class ZyantineMemorySystem:
    """基于memo0的自衍体记忆系统"""

    def __init__(self,
                 base_url: str = DEFAULT_BASE_URL,
                 api_key: str = DEFAULT_API_KEY,
                 user_id: str = "default",
                 session_id: str = "default"):
        """
        初始化记忆系统

        Args:
            base_url: OpenAI API基础URL
            api_key: API密钥
            user_id: 用户ID
            session_id: 会话ID
        """
        self.base_url = base_url
        self.api_key = api_key
        self.user_id = user_id
        self.session_id = session_id

        # 初始化memo0记忆系统
        self.memory = self._initialize_memo0()

        # 语义记忆地图
        self.semantic_memory_map: Dict[str, Dict] = {}
        self.strategic_tags: List[str] = []

        # 统计信息
        self.stats = {
            "total_memories": 0,
            "by_type": {},
            "access_counts": {},
            "tags_distribution": {}
        }

        print(f"[记忆系统] 初始化完成，用户ID: {user_id}，会话ID: {session_id}")

    def _initialize_memo0(self) -> Memory:
        """初始化memo0框架"""
        config = {
            "vector_store": {
                "provider": "milvus",
                "config": {
                    "collection_name": "zyantine_memories",
                    "url": "http://localhost:19530",
                    "token": "",
                }
            },
            "llm": {
                "provider": "openai",
                "config": {
                    "openai_base_url": self.base_url,
                    "api_key": self.api_key
                }
            },
            "embedder": {
                "provider": "openai",
                "config": {
                    "model": "text-embedding-3-large",
                    "openai_base_url": self.base_url,
                    "api_key": self.api_key
                }
            }
        }

        return Memory.from_config(config)

    # ============ 记忆CRUD操作 ============

    def add_memory(self,
                   content: Union[str, List[Dict]],
                   memory_type: str = "conversation",
                   metadata: Optional[Dict] = None,
                   tags: Optional[List[str]] = None,
                   emotional_intensity: float = 0.5,
                   strategic_value: Optional[Dict] = None,
                   linked_tool: Optional[str] = None) -> str:
        """
        添加记忆

        Args:
            content: 记忆内容（字符串或对话列表）
            memory_type: 记忆类型
            metadata: 附加元数据
            tags: 标签列表
            emotional_intensity: 情感强度 (0-1)
            strategic_value: 战略价值评估
            linked_tool: 关联的认知工具

        Returns:
            记忆ID
        """
        # 生成记忆ID
        if isinstance(content, str):
            content_str = content
        else:
            content_str = json.dumps(content, ensure_ascii=False)

        memory_id = self._generate_memory_id(content_str, memory_type)

        # 准备完整内容
        if isinstance(content, list):
            # 对话格式
            full_content = content
        else:
            # 文本格式
            full_content = [{"role": "user", "content": content}]

        # 准备元数据
        memory_metadata = self._prepare_metadata(
            memory_id=memory_id,
            memory_type=memory_type,
            content=content_str,
            metadata=metadata or {},
            tags=tags or [],
            emotional_intensity=emotional_intensity,
            strategic_value=strategic_value or {},
            linked_tool=linked_tool
        )

        # 添加到memo0
        self.memory.add(
            full_content,
            user_id=self.user_id,
            metadata=memory_metadata
        )

        # 更新语义记忆地图
        self._update_semantic_memory(memory_id, memory_metadata)

        # 更新统计信息
        self._update_stats(memory_type, tags or [])

        print(f"[记忆系统] 添加记忆成功，ID: {memory_id}，类型: {memory_type}")
        return memory_id

    def _generate_memory_id(self, content: str, memory_type: str) -> str:
        """生成记忆ID"""
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        content_hash = hashlib.md5(content.encode()).hexdigest()[:8]
        return f"{memory_type}_{timestamp}_{content_hash}"

    def _prepare_metadata(self,
                          memory_id: str,
                          memory_type: str,
                          content: str,
                          metadata: Dict,
                          tags: List[str],
                          emotional_intensity: float,
                          strategic_value: Dict,
                          linked_tool: Optional[str]) -> Dict:
        """准备元数据"""
        base_metadata = {
            "memory_id": memory_id,
            "memory_type": memory_type,
            "session_id": self.session_id,
            "created_at": datetime.now().isoformat(),
            "tags": tags,
            "emotional_intensity": emotional_intensity,
            "strategic_value": strategic_value,
            "linked_tool": linked_tool,
            "source": "zyantine_system",
            "content_length": len(content)
        }

        # 合并用户提供的元数据
        base_metadata.update(metadata)

        return base_metadata

    def _update_semantic_memory(self, memory_id: str, metadata: Dict):
        """更新语义记忆地图"""
        self.semantic_memory_map[memory_id] = {
            "metadata": metadata,
            "access_count": 0,
            "last_accessed": None,
            "strategic_score": metadata.get("strategic_value", {}).get("score", 0)
        }

    def _update_stats(self, memory_type: str, tags: List[str]):
        """更新统计信息"""
        self.stats["total_memories"] += 1

        # 按类型统计
        if memory_type not in self.stats["by_type"]:
            self.stats["by_type"][memory_type] = 0
        self.stats["by_type"][memory_type] += 1

        # 标签分布
        for tag in tags:
            if tag not in self.stats["tags_distribution"]:
                self.stats["tags_distribution"][tag] = 0
            self.stats["tags_distribution"][tag] += 1

        # 更新战略标签
        for tag in tags:
            if tag not in self.strategic_tags:
                self.strategic_tags.append(tag)

    # ============ 记忆检索 ============

    def search_memories(self,
                        query: str,
                        memory_type: Optional[str] = None,
                        tags: Optional[List[str]] = None,
                        limit: int = 5,
                        similarity_threshold: float = 0.7,
                        rerank: bool = True) -> List[Dict]:
        """
        搜索记忆

        Args:
            query: 查询文本
            memory_type: 记忆类型过滤
            tags: 标签过滤
            limit: 返回数量限制
            similarity_threshold: 相似度阈值
            rerank: 是否重新排序

        Returns:
            记忆结果列表
        """
        # 构建元数据过滤器
        metadata_filter = {}
        if memory_type:
            metadata_filter["memory_type"] = memory_type
        if tags:
            metadata_filter["tags"] = tags

        # 执行搜索
        search_results = self.memory.search(
            query,
            user_id=self.user_id,
            limit=limit,
            rerank=rerank
        )

        # 处理结果
        processed_results = []
        for hit in search_results.get("results", []):
            memory_data = hit.get("memory", {})
            metadata = hit.get("metadata", {})
            score = hit.get("score", 0)

            # 应用相似度阈值
            if score < similarity_threshold:
                continue

            # 更新访问统计
            memory_id = metadata.get("memory_id")
            if memory_id and memory_id in self.semantic_memory_map:
                self.semantic_memory_map[memory_id]["access_count"] += 1
                self.semantic_memory_map[memory_id]["last_accessed"] = datetime.now().isoformat()

                # 更新全局访问统计
                if memory_id not in self.stats["access_counts"]:
                    self.stats["access_counts"][memory_id] = 0
                self.stats["access_counts"][memory_id] += 1

            # 构建结果
            result = {
                "memory_id": memory_id,
                "content": self._extract_content_from_memory(memory_data),
                "metadata": metadata,
                "similarity_score": score,
                "memory_type": metadata.get("memory_type"),
                "tags": metadata.get("tags", []),
                "emotional_intensity": metadata.get("emotional_intensity", 0.5),
                "strategic_value": metadata.get("strategic_value", {}),
                "linked_tool": metadata.get("linked_tool"),
                "created_at": metadata.get("created_at"),
                "access_count": self.semantic_memory_map.get(memory_id, {}).get("access_count", 0)
            }

            processed_results.append(result)

        # 按相似度排序
        processed_results.sort(key=lambda x: x["similarity_score"], reverse=True)

        return processed_results[:limit]

    def _extract_content_from_memory(self, memory_data: Any) -> str:
        """从记忆数据中提取内容"""
        if isinstance(memory_data, list):
            # 对话格式
            content_parts = []
            for msg in memory_data:
                if isinstance(msg, dict):
                    role = msg.get("role", "")
                    content = msg.get("content", "")
                    content_parts.append(f"{role}: {content}")
            return "\n".join(content_parts)
        elif isinstance(memory_data, str):
            return memory_data
        else:
            return str(memory_data)

    def find_conversations(self,
                           query: str,
                           session_id: Optional[str] = None,
                           limit: int = 5) -> List[Dict]:
        """
        查找相关对话

        Args:
            query: 查询文本
            session_id: 会话ID过滤
            limit: 返回数量限制

        Returns:
            对话结果列表
        """
        metadata_filter = {"memory_type": MemoryType.CONVERSATION.value}
        if session_id:
            metadata_filter["session_id"] = session_id
        else:
            metadata_filter["session_id"] = self.session_id

        return self.search_memories(
            query=query,
            memory_type=MemoryType.CONVERSATION.value,
            limit=limit
        )

    def find_experiences(self,
                         context: Dict,
                         limit: int = 3) -> List[Dict]:
        """
        查找相关经历记忆

        Args:
            context: 上下文信息
            limit: 返回数量限制

        Returns:
            经历记忆列表
        """
        # 构建查询
        query_parts = []
        if "user_input" in context:
            query_parts.append(context["user_input"])
        if "user_emotion" in context:
            query_parts.append(f"情绪 {context['user_emotion']}")
        if "topic" in context:
            query_parts.append(f"话题 {context['topic']}")

        query = " ".join(query_parts)

        return self.search_memories(
            query=query,
            memory_type=MemoryType.EXPERIENCE.value,
            limit=limit
        )

    # ============ 记忆管理 ============

    def get_memory(self, memory_id: str) -> Optional[Dict]:
        """
        获取特定记忆

        Args:
            memory_id: 记忆ID

        Returns:
            记忆信息或None
        """
        # 首先尝试从语义记忆地图获取
        if memory_id in self.semantic_memory_map:
            # 通过搜索找到具体记忆
            search_results = self.search_memories(
                query=memory_id,  # 使用ID作为查询
                limit=1
            )

            if search_results:
                memory_info = search_results[0]

                # 更新访问统计
                self.semantic_memory_map[memory_id]["access_count"] += 1
                self.semantic_memory_map[memory_id]["last_accessed"] = datetime.now().isoformat()

                return memory_info

        return None

    def update_memory(self,
                      memory_id: str,
                      new_content: Optional[str] = None,
                      new_tags: Optional[List[str]] = None,
                      new_metadata: Optional[Dict] = None) -> bool:
        """
        更新记忆

        Args:
            memory_id: 记忆ID
            new_content: 新内容
            new_tags: 新标签
            new_metadata: 新元数据

        Returns:
            是否成功
        """
        # 获取现有记忆
        memory_info = self.get_memory(memory_id)
        if not memory_info:
            return False

        # 构建更新内容
        current_metadata = memory_info.get("metadata", {})

        if new_tags:
            current_metadata["tags"] = new_tags

        if new_metadata:
            current_metadata.update(new_metadata)

        # 标记为更新
        current_metadata["updated_at"] = datetime.now().isoformat()

        # 如果需要更新内容，创建新记忆并标记旧记忆
        if new_content:
            # 创建新记忆
            self.add_memory(
                content=new_content,
                memory_type=current_metadata.get("memory_type", "conversation"),
                metadata=current_metadata,
                tags=current_metadata.get("tags", []),
                emotional_intensity=current_metadata.get("emotional_intensity", 0.5),
                strategic_value=current_metadata.get("strategic_value", {}),
                linked_tool=current_metadata.get("linked_tool")
            )

            # 标记旧记忆为已更新
            if memory_id in self.semantic_memory_map:
                self.semantic_memory_map[memory_id]["status"] = "updated"

        return True

    def delete_memory(self, memory_id: str) -> bool:
        """
        删除记忆

        Args:
            memory_id: 记忆ID

        Returns:
            是否成功
        """
        # memo0框架目前没有直接的删除API
        # 我们可以通过标记为删除来实现
        if memory_id in self.semantic_memory_map:
            self.semantic_memory_map[memory_id]["status"] = "deleted"
            self.semantic_memory_map[memory_id]["deleted_at"] = datetime.now().isoformat()

            # 更新统计信息
            self.stats["total_memories"] = max(0, self.stats["total_memories"] - 1)

            print(f"[记忆系统] 记忆标记为删除: {memory_id}")
            return True

        return False

    # ============ 批量操作 ============

    def add_conversation_batch(self, conversations: List[Dict]) -> List[str]:
        """
        批量添加对话

        Args:
            conversations: 对话列表

        Returns:
            记忆ID列表
        """
        memory_ids = []

        for conv in conversations:
            memory_id = self.add_memory(
                content=conv,
                memory_type=MemoryType.CONVERSATION.value,
                tags=["对话", "批量导入"],
                emotional_intensity=conv.get("emotional_intensity", 0.5)
            )
            memory_ids.append(memory_id)

        return memory_ids

    def import_user_profile(self, profile_data: Dict) -> List[str]:
        """
        导入用户档案数据

        Args:
            profile_data: 用户档案数据

        Returns:
            记忆ID列表
        """
        memory_ids = []

        # 导入用户记忆
        if "memories" in profile_data:
            for memory in profile_data["memories"]:
                memory_id = self.add_memory(
                    content=memory.get("content", ""),
                    memory_type=MemoryType.EXPERIENCE.value,
                    tags=memory.get("tags", ["用户记忆"]),
                    emotional_intensity=memory.get("emotional_intensity", 0.5),
                    strategic_value=memory.get("strategic_value", {}),
                    source="user_profile_import"
                )
                memory_ids.append(memory_id)

        # 导入用户特征
        if "personality_traits" in profile_data:
            traits_text = json.dumps(profile_data["personality_traits"], ensure_ascii=False)
            memory_id = self.add_memory(
                content=f"用户性格特征: {traits_text}",
                memory_type=MemoryType.USER_PROFILE.value,
                tags=["性格特征", "用户档案"],
                source="user_profile_import"
            )
            memory_ids.append(memory_id)

        return memory_ids

    # ============ 统计与分析 ============

    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        # 计算最常用的标签
        top_tags = sorted(
            self.stats["tags_distribution"].items(),
            key=lambda x: x[1],
            reverse=True
        )[:10]

        # 计算最常访问的记忆
        top_accessed = sorted(
            self.semantic_memory_map.items(),
            key=lambda x: x[1].get("access_count", 0),
            reverse=True
        )[:5]

        top_accessed_formatted = []
        for mem_id, mem_data in top_accessed:
            top_accessed_formatted.append({
                "memory_id": mem_id,
                "access_count": mem_data.get("access_count", 0),
                "last_accessed": mem_data.get("last_accessed"),
                "strategic_score": mem_data.get("strategic_score", 0)
            })

        return {
            "total_memories": self.stats["total_memories"],
            "memory_types": self.stats["by_type"],
            "top_tags": dict(top_tags),
            "top_accessed_memories": top_accessed_formatted,
            "strategic_tags_count": len(self.strategic_tags),
            "user_id": self.user_id,
            "session_id": self.session_id,
            "semantic_map_size": len(self.semantic_memory_map)
        }

    def analyze_memory_patterns(self) -> Dict[str, Any]:
        """
        分析记忆模式

        Returns:
            模式分析结果
        """
        # 按类型分析
        type_analysis = {}
        for mem_id, mem_data in self.semantic_memory_map.items():
            mem_type = mem_data.get("metadata", {}).get("memory_type")
            if mem_type not in type_analysis:
                type_analysis[mem_type] = {
                    "count": 0,
                    "total_access": 0,
                    "avg_emotional_intensity": 0
                }

            analysis = type_analysis[mem_type]
            analysis["count"] += 1
            analysis["total_access"] += mem_data.get("access_count", 0)

            # 情感强度累计
            emotional_intensity = mem_data.get("metadata", {}).get("emotional_intensity", 0.5)
            if "emotional_intensity_sum" not in analysis:
                analysis["emotional_intensity_sum"] = 0
                analysis["emotional_intensity_count"] = 0

            analysis["emotional_intensity_sum"] += emotional_intensity
            analysis["emotional_intensity_count"] += 1

        # 计算平均值
        for mem_type, analysis in type_analysis.items():
            if analysis["count"] > 0:
                analysis["avg_access"] = analysis["total_access"] / analysis["count"]
            if analysis.get("emotional_intensity_count", 0) > 0:
                analysis["avg_emotional_intensity"] = (
                        analysis["emotional_intensity_sum"] / analysis["emotional_intensity_count"]
                )

            # 移除临时字段
            analysis.pop("emotional_intensity_sum", None)
            analysis.pop("emotional_intensity_count", None)

        return {
            "type_analysis": type_analysis,
            "strategic_tags": self.strategic_tags,
            "high_value_memories": self._get_high_value_memories()
        }

    def _get_high_value_memories(self) -> List[Dict]:
        """获取高价值记忆"""
        high_value_memories = []

        for mem_id, mem_data in self.semantic_memory_map.items():
            strategic_score = mem_data.get("strategic_score", 0)
            access_count = mem_data.get("access_count", 0)

            # 高价值标准：战略分数高或访问次数多
            if strategic_score > 2 or access_count > 3:
                high_value_memories.append({
                    "memory_id": mem_id,
                    "strategic_score": strategic_score,
                    "access_count": access_count,
                    "tags": mem_data.get("metadata", {}).get("tags", []),
                    "memory_type": mem_data.get("metadata", {}).get("memory_type")
                })

        # 按战略分数排序
        high_value_memories.sort(key=lambda x: x["strategic_score"], reverse=True)
        return high_value_memories[:10]

    # ============ 导出与备份 ============

    def export_memories(self, file_path: str, format_type: str = "json") -> bool:
        """
        导出记忆

        Args:
            file_path: 文件路径
            format_type: 导出格式

        Returns:
            是否成功
        """
        try:
            export_data = {
                "metadata": {
                    "export_time": datetime.now().isoformat(),
                    "user_id": self.user_id,
                    "session_id": self.session_id,
                    "total_memories": self.stats["total_memories"]
                },
                "semantic_memory_map": self.semantic_memory_map,
                "statistics": self.stats,
                "strategic_tags": self.strategic_tags
            }

            with open(file_path, 'w', encoding='utf-8') as f:
                if format_type == "json":
                    json.dump(export_data, f, ensure_ascii=False, indent=2)
                else:
                    # 其他格式可扩展
                    f.write(str(export_data))

            print(f"[记忆系统] 记忆导出成功: {file_path}")
            return True

        except Exception as e:
            print(f"[记忆系统] 记忆导出失败: {e}")
            return False

    # ============ 记忆炼金术引擎 ============

    def find_resonant_memory(self, context: Dict) -> Optional[Dict]:
        """
        寻找共鸣记忆

        Args:
            context: 上下文信息

        Returns:
            共鸣记忆包
        """
        # 构建查询
        query_text = self._build_resonance_query(context)

        if not query_text:
            return None

        # 搜索相关经历记忆
        similar_experiences = self.find_experiences(context, limit=3)

        if not similar_experiences:
            # 如果没有直接经历，搜索相关对话
            similar_conversations = self.find_conversations(
                query=query_text,
                limit=2
            )

            if similar_conversations:
                best_match = similar_conversations[0]
            else:
                return None
        else:
            best_match = similar_experiences[0]

        # 构建战术信息包
        tactical_package = self._build_tactical_package(best_match, context)

        return tactical_package

    def _build_resonance_query(self, context: Dict) -> str:
        """构建共鸣查询"""
        query_parts = []

        if "user_input" in context:
            query_parts.append(context["user_input"])

        if "user_emotion" in context:
            query_parts.append(f"情绪: {context['user_emotion']}")

        if "topic" in context:
            query_parts.append(f"话题: {context['topic']}")

        return " ".join(query_parts) if query_parts else ""

    def _build_tactical_package(self, memory_match: Dict, context: Dict) -> Dict:
        """构建战术信息包"""
        metadata = memory_match.get("metadata", {})

        package = {
            "triggered_memory": memory_match.get("content", "未知记忆"),
            "memory_id": memory_match.get("memory_id"),
            "relevance_score": memory_match.get("similarity_score", 0),
            "source": metadata.get("source", "unknown"),
            "tags": metadata.get("tags", []),
            "strategic_value": metadata.get("strategic_value", {}),
            "linked_tool": metadata.get("linked_tool"),
            "emotional_intensity": metadata.get("emotional_intensity", 0.5),
            "risk_assessment": self._assess_memory_risk(metadata),
            "recommended_actions": self._generate_recommendations(metadata, context),
            "timestamp": datetime.now().isoformat(),
            "retrieval_method": "memo0_vector_search"
        }

        return package

    def _assess_memory_risk(self, metadata: Dict) -> Dict[str, Any]:
        """评估记忆风险"""
        risk_score = 0
        high_risk_factors = []

        # 高风险标签
        high_risk_tags = ["创伤", "背叛", "失败", "痛苦"]
        tags = metadata.get("tags", [])

        for tag in tags:
            if tag in high_risk_tags:
                risk_score += 3
                high_risk_factors.append(tag)

        # 情感强度影响
        emotional_intensity = metadata.get("emotional_intensity", 0.5)
        if emotional_intensity > 0.8:
            risk_score += 2

        # 确定风险级别
        if risk_score >= 5:
            level = "高"
        elif risk_score >= 3:
            level = "中"
        elif risk_score >= 1:
            level = "低"
        else:
            level = "极低"

        return {
            "level": level,
            "score": risk_score,
            "high_risk_factors": high_risk_factors
        }

    def _generate_recommendations(self, metadata: Dict, context: Dict) -> List[str]:
        """生成使用建议"""
        recommendations = []
        tags = metadata.get("tags", [])

        if "成就" in tags or "成功" in tags:
            recommendations.append("可安全提及以激活积极情绪")

        if "创伤" in tags or "痛苦" in tags:
            recommendations.append("高风险区域，谨慎使用")

        if "学习" in tags or "成长" in tags:
            recommendations.append("适合用于激励场景")

        return recommendations if recommendations else ["常规记忆，可灵活使用"]

    # ============ 工具方法 ============

    def clear_cache(self):
        """清理缓存"""
        # 重置语义记忆地图
        self.semantic_memory_map.clear()
        self.strategic_tags.clear()

        # 重置统计信息
        self.stats = {
            "total_memories": 0,
            "by_type": {},
            "access_counts": {},
            "tags_distribution": {}
        }

        print("[记忆系统] 缓存已清理")

    def test_connection(self) -> bool:
        """测试连接"""
        try:
            # 测试添加和搜索
            test_id = self.add_memory(
                content="连接测试",
                memory_type="test",
                tags=["测试"]
            )

            results = self.search_memories("连接测试", limit=1)

            if results and len(results) > 0:
                print("[记忆系统] 连接测试成功")
                return True
            else:
                print("[记忆系统] 连接测试失败")
                return False

        except Exception as e:
            print(f"[记忆系统] 连接测试异常: {e}")
            return False


# ============ 使用示例 ============
if __name__ == "__main__":
    # 1. 初始化记忆系统
    memory_system = ZyantineMemorySystem(
        user_id="demo-user",
        session_id="session-001"
    )

    # 2. 测试连接
    if memory_system.test_connection():
        print("✅ 记忆系统连接成功")
    else:
        print("❌ 记忆系统连接失败")

    # 3. 添加对话记忆
    conversation = [
        {"role": "user", "content": "你好，我叫小明"},
        {"role": "assistant", "content": "你好小明，很高兴认识你！"}
    ]

    memory_id = memory_system.add_memory(
        content=conversation,
        memory_type=MemoryType.CONVERSATION.value,
        tags=["初次见面", "自我介绍"],
        emotional_intensity=0.7
    )

    print(f"✅ 对话记忆添加成功，ID: {memory_id}")

    # 4. 添加经历记忆
    experience_id = memory_system.add_memory(
        content="我第一次学习编程是在大学时期，当时对Python产生了浓厚兴趣",
        memory_type=MemoryType.EXPERIENCE.value,
        tags=["学习", "编程", "Python", "大学"],
        emotional_intensity=0.8,
        strategic_value={"level": "高", "score": 4}
    )

    print(f"✅ 经历记忆添加成功，ID: {experience_id}")

    # 5. 搜索记忆
    search_results = memory_system.search_memories(
        query="用户叫什么名字",
        memory_type=MemoryType.CONVERSATION.value,
        limit=3
    )

    print(f"🔍 搜索结果 ({len(search_results)} 条):")
    for result in search_results:
        print(f"  - {result['memory_id']}: {result['content'][:50]}... (相似度: {result['similarity_score']:.3f})")

    # 6. 获取统计信息
    stats = memory_system.get_statistics()
    print(f"📊 统计信息: 总记忆数 {stats['total_memories']}")

    # 7. 寻找共鸣记忆
    context = {
        "user_input": "我对编程很感兴趣",
        "user_emotion": "兴奋",
        "topic": "学习编程"
    }

    resonant_memory = memory_system.find_resonant_memory(context)
    if resonant_memory:
        print(f"🎯 找到共鸣记忆: {resonant_memory['triggered_memory'][:50]}...")

    # 8. 导出记忆
    memory_system.export_memories("memory_export.json")