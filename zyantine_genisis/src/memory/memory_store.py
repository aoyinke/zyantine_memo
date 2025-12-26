# ============ 基于memo0的简化记忆系统 ============
import os
import json
import hashlib
from datetime import datetime
from typing import Dict, List, Any, Optional, Union

# Mem0框架导入
from mem0 import Memory

# ============ 常量定义 ============
DEFAULT_BASE_URL = "https://openkey.cloud/v1"
DEFAULT_API_KEY = "sk-wiHpoarpNTHaep0t54852a32A75a4d6986108b3f6eF7B7B9"


# ============ 记忆系统核心类 ============
class ZyantineMemorySystem:
    """基于memo0的自衍体记忆系统"""

    def __init__(self,
                 base_url: str = DEFAULT_BASE_URL,
                 api_key: str = DEFAULT_API_KEY,
                 user_id: str = "default",
                 session_id: str = "default"):
        """初始化记忆系统"""
        self.base_url = base_url
        self.api_key = api_key
        self.user_id = user_id
        self.session_id = session_id

        # 初始化memo0记忆系统
        self.memory = self._initialize_memo0()
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
        """添加记忆（精简metadata避免过大）"""
        content_str = content if isinstance(content, str) else json.dumps(content, ensure_ascii=False)
        memory_id = self._generate_memory_id(content_str, memory_type)

        full_content = content if isinstance(content, list) else [{"role": "user", "content": content}]

        # 准备精简的metadata
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

        # 检查并进一步压缩metadata
        metadata_size = len(json.dumps(memory_metadata))
        if metadata_size > 30000:
            print(f"[记忆系统] 警告：metadata大小({metadata_size})接近限制，进一步精简")
            memory_metadata = self._further_compress_metadata(memory_metadata)

        try:
            # 添加到memo0
            self.memory.add(full_content, user_id=self.user_id, metadata=memory_metadata)
        except Exception as e:
            if "exceeds max length" in str(e):
                print(f"[记忆系统] 错误：metadata仍然过大，使用最小化版本")
                minimal_metadata = {
                    "memory_id": memory_id,
                    "memory_type": memory_type,
                    "session_id": self.session_id,
                    "created_at": datetime.now().isoformat()
                }
                self.memory.add(full_content, user_id=self.user_id, metadata=minimal_metadata)
            else:
                raise e

        # 更新语义记忆地图
        self._update_semantic_memory(memory_id, {
            "full_metadata": memory_metadata,
            "content_preview": content_str[:500] if content_str else "",
            "tags": tags or [],
            "emotional_intensity": emotional_intensity,
            "strategic_value": strategic_value or {},
            "linked_tool": linked_tool
        })

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
        """准备精简的元数据"""
        # 精简strategic_value
        simplified_strategic = {
            "score": strategic_value.get("score", 0),
            "level": strategic_value.get("level", "低")
        } if strategic_value and isinstance(strategic_value, dict) else {"score": 0, "level": "低"}

        base_metadata = {
            "memory_id": memory_id,
            "memory_type": memory_type,
            "session_id": self.session_id,
            "created_at": datetime.now().isoformat(),
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

    def _further_compress_metadata(self, metadata: Dict) -> Dict:
        """进一步压缩metadata"""
        minimal = {
            "memory_id": metadata.get("memory_id", ""),
            "memory_type": metadata.get("memory_type", ""),
            "session_id": metadata.get("session_id", ""),
            "created_at": metadata.get("created_at", ""),
            "content_length": metadata.get("content_length", 0)
        }

        if "tags" in metadata and metadata["tags"]:
            minimal["tags"] = metadata["tags"][:2]

        return minimal

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
        self.stats["by_type"][memory_type] = self.stats["by_type"].get(memory_type, 0) + 1

        # 标签分布
        for tag in tags:
            self.stats["tags_distribution"][tag] = self.stats["tags_distribution"].get(tag, 0) + 1
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
        """搜索记忆"""
        # 构建元数据过滤器
        metadata_filter = {}
        if memory_type:
            metadata_filter["memory_type"] = memory_type
        if tags:
            metadata_filter["tags"] = tags

        # 执行搜索
        search_results = self.memory.search(query, user_id=self.user_id, limit=limit, rerank=rerank)

        processed_results = []
        for hit in search_results.get("results", []):
            memory_data = hit.get("memory", {})
            metadata = hit.get("metadata", {})
            score = hit.get("score", 0)

            if score < similarity_threshold:
                continue

            # 更新访问统计
            memory_id = metadata.get("memory_id")
            if memory_id and memory_id in self.semantic_memory_map:
                self.semantic_memory_map[memory_id]["access_count"] += 1
                self.semantic_memory_map[memory_id]["last_accessed"] = datetime.now().isoformat()
                self.stats["access_counts"][memory_id] = self.stats["access_counts"].get(memory_id, 0) + 1

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

        processed_results.sort(key=lambda x: x["similarity_score"], reverse=True)
        return processed_results[:limit]

    def _extract_content_from_memory(self, memory_data: Any) -> str:
        """从记忆数据中提取内容"""
        if isinstance(memory_data, list):
            return "\n".join(
                [f"{msg.get('role', '')}: {msg.get('content', '')}" for msg in memory_data if isinstance(msg, dict)])
        elif isinstance(memory_data, str):
            return memory_data
        else:
            return str(memory_data)

    def find_conversations(self,
                           query: str,
                           session_id: Optional[str] = None,
                           limit: int = 5) -> List[Dict]:
        """查找相关对话"""
        target_session = session_id or self.session_id
        return self.search_memories(query=query, memory_type="conversation", limit=limit)

    def find_experiences(self,
                         context: Dict,
                         limit: int = 3) -> List[Dict]:
        """查找相关经历记忆"""
        query_parts = []
        if "user_input" in context:
            query_parts.append(context["user_input"])
        if "user_emotion" in context:
            query_parts.append(f"情绪 {context['user_emotion']}")
        if "topic" in context:
            query_parts.append(f"话题 {context['topic']}")

        return self.search_memories(query=" ".join(query_parts), memory_type="experience", limit=limit)

    # ============ 记忆管理 ============

    def get_memory(self, memory_id: str, include_full_data: bool = False) -> Optional[Dict]:
        """获取特定记忆"""
        if memory_id in self.semantic_memory_map:
            search_results = self.search_memories(query=memory_id, limit=1)
            if search_results:
                memory_info = search_results[0]
                self.semantic_memory_map[memory_id]["access_count"] += 1
                self.semantic_memory_map[memory_id]["last_accessed"] = datetime.now().isoformat()

                if include_full_data:
                    full_data = self._load_full_memory_from_local(memory_id)
                    if full_data:
                        memory_info["full_data"] = full_data

                return memory_info
        return None

    def _load_full_memory_from_local(self, memory_id: str) -> Optional[Dict]:
        """从本地文件加载完整记忆数据"""
        try:
            local_file = f"./memory_backup/{self.user_id}/{self.session_id}/{memory_id}.json"
            if os.path.exists(local_file):
                with open(local_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
        except:
            pass
        return None

    # ============ 批量操作 ============

    def add_conversation_batch(self, conversations: List[Dict]) -> List[str]:
        """批量添加对话"""
        return [
            self.add_memory(content=conv, memory_type="conversation", tags=["对话", "批量导入"],
                            emotional_intensity=conv.get("emotional_intensity", 0.5))
            for conv in conversations
        ]

    def import_user_profile(self, profile_data: Dict) -> List[str]:
        """导入用户档案数据"""
        memory_ids = []

        # 导入用户记忆
        if "memories" in profile_data:
            for memory in profile_data["memories"]:
                memory_ids.append(self.add_memory(
                    content=memory.get("content", ""),
                    memory_type="experience",
                    tags=memory.get("tags", ["用户记忆"]),
                    emotional_intensity=memory.get("emotional_intensity", 0.5),
                    strategic_value=memory.get("strategic_value", {})
                ))

        # 导入用户特征
        if "personality_traits" in profile_data:
            traits_text = json.dumps(profile_data["personality_traits"], ensure_ascii=False)
            memory_ids.append(self.add_memory(
                content=f"用户性格特征: {traits_text}",
                memory_type="user_profile",
                tags=["性格特征", "用户档案"]
            ))

        return memory_ids

    # ============ 统计与分析 ============

    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        # 计算最常用的标签
        top_tags = sorted(self.stats["tags_distribution"].items(), key=lambda x: x[1], reverse=True)[:10]

        # 计算最常访问的记忆
        top_accessed = sorted(self.semantic_memory_map.items(),
                              key=lambda x: x[1].get("access_count", 0), reverse=True)[:5]

        top_accessed_formatted = [{
            "memory_id": mem_id,
            "access_count": mem_data.get("access_count", 0),
            "last_accessed": mem_data.get("last_accessed"),
            "strategic_score": mem_data.get("strategic_score", 0)
        } for mem_id, mem_data in top_accessed]

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

    # ============ 共鸣记忆功能 ============

    def find_resonant_memory(self, context: Dict) -> Optional[Dict]:
        """寻找共鸣记忆"""
        query_text = self._build_resonance_query(context)
        if not query_text:
            return None

        # 搜索相关经历记忆
        similar_experiences = self.find_experiences(context, limit=3)
        best_match = similar_experiences[0] if similar_experiences else None

        # 如果没有直接经历，搜索相关对话
        if not best_match:
            similar_conversations = self.find_conversations(query_text, limit=2)
            best_match = similar_conversations[0] if similar_conversations else None

        return self._build_tactical_package(best_match, context) if best_match else None

    def _build_resonance_query(self, context: Dict) -> str:
        """构建共鸣查询"""
        parts = []
        if "user_input" in context:
            parts.append(context["user_input"])
        if "user_emotion" in context:
            parts.append(f"情绪: {context['user_emotion']}")
        if "topic" in context:
            parts.append(f"话题: {context['topic']}")
        return " ".join(parts) if parts else ""

    def _build_tactical_package(self, memory_match: Dict, context: Dict) -> Dict:
        """构建战术信息包"""
        metadata = memory_match.get("metadata", {})

        return {
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
        if metadata.get("emotional_intensity", 0.5) > 0.8:
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

        return {"level": level, "score": risk_score, "high_risk_factors": high_risk_factors}

    def _generate_recommendations(self, metadata: Dict, context: Dict) -> List[str]:
        """生成使用建议"""
        tags = metadata.get("tags", [])
        recommendations = []

        if any(tag in tags for tag in ["成就", "成功"]):
            recommendations.append("可安全提及以激活积极情绪")
        if any(tag in tags for tag in ["创伤", "痛苦"]):
            recommendations.append("高风险区域，谨慎使用")
        if any(tag in tags for tag in ["学习", "成长"]):
            recommendations.append("适合用于激励场景")

        return recommendations if recommendations else ["常规记忆，可灵活使用"]

    # ============ 导出与工具方法 ============

    def export_memories(self, file_path: str, format_type: str = "json") -> bool:
        """导出记忆"""
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
                    f.write(str(export_data))

            print(f"[记忆系统] 记忆导出成功: {file_path}")
            return True
        except Exception as e:
            print(f"[记忆系统] 记忆导出失败: {e}")
            return False

    def clear_cache(self):
        """清理缓存"""
        self.semantic_memory_map.clear()
        self.strategic_tags.clear()
        self.stats = {"total_memories": 0, "by_type": {}, "access_counts": {}, "tags_distribution": {}}
        print("[记忆系统] 缓存已清理")

    def test_connection(self) -> bool:
        """测试连接"""
        try:
            test_id = self.add_memory(content="连接测试", memory_type="test", tags=["测试"])
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
