"""
动态记忆炼金术引擎
记忆的解析、分片、标签化与战略链接
"""

import time
import json
import hashlib
from datetime import datetime
from typing import Dict, List, Any, Optional

class DynamicMemoryAlchemyEngine:
    """动态记忆炼金术引擎"""

    def __init__(self, user_profile_data: Dict, self_profile_data: Dict):
        self.user_profile = user_profile_data
        self.self_profile = self_profile_data
        # 战略链接
        self.tool_memory_links = {
            "情感共情过载": ["emotional_vulnerability", "childhood_trauma", "core_pain_point"],
            "过度保护倾向": ["high_value_asset", "core_strength", "reliable_skill"],
            "技术乐观主义": ["early_success", "self_learning", "tenacity"]
        }
        # 记忆碎片存储
        self.memory_fragments = []
        self.strategic_tags = []
        self.association_cache = {}
        self.semantic_memory_map = self._preprocess_and_internalize()

    def _preprocess_and_internalize(self) -> Dict:
        """预处理和内化记忆数据"""
        semantic_map = {}

        # 处理用户记忆
        if "memories" in self.user_profile:
            for memory in self.user_profile["memories"]:
                memory_id = f"user_memory_{len(semantic_map):04d}"
                semantic_map[memory_id] = self._process_memory_fragment(memory, "user")

        # 处理自衍体记忆
        if "self_memories" in self.self_profile:
            for memory in self.self_profile["self_memories"]:
                memory_id = f"self_memory_{len(semantic_map):04d}"
                semantic_map[memory_id] = self._process_memory_fragment(memory, "self")

        return semantic_map

    def _process_memory_fragment(self, memory: Dict, source: str) -> Dict:
        """处理单个记忆碎片"""
        # 1. 解析与分片
        fragments = self._fragment_experience(memory)

        # 2. 打上语义标签
        tags = self._tag_fragments(fragments, memory.get("emotional_context", {}))

        # 3. 认知分析
        cognitive_analysis = self._analyze_cognitive_value(memory, tags, source)

        # 4. 建立策略链接
        linked_tool = self._link_to_cognitive_tools(tags)

        processed = {
            "event_summary": memory.get("summary", "未知事件"),
            "raw_content": memory.get("content", ""),
            "fragments": fragments,
            "tags": tags,
            "linked_tool": linked_tool,
            "cognitive_analysis": cognitive_analysis,
            "source": source,
            "timestamp": memory.get("timestamp", datetime.now().isoformat()),
            "emotional_intensity": memory.get("emotional_intensity", 0.5),
            "strategic_value": self._assess_strategic_value(tags, cognitive_analysis),
            "access_count": 0,
            "last_accessed": None
        }

        # 存储记忆碎片
        self.memory_fragments.append(processed)

        # 更新战略标签
        for tag in tags:
            if tag not in self.strategic_tags:
                self.strategic_tags.append(tag)

        return processed

    def _fragment_experience(self, memory: Dict) -> List[Dict]:
        """将经历分解为信息碎片"""
        content = memory.get("content", "")
        fragments = []

        # 简单分句（实际应用应使用更复杂的分割）
        sentences = content.split('。')

        for i, sentence in enumerate(sentences):
            if sentence.strip():
                fragments.append({
                    "id": f"fragment_{len(fragments):04d}",
                    "content": sentence.strip() + '。',
                    "position": i,
                    "key_entities": self._extract_entities(sentence),
                    "emotional_tones": self._detect_emotional_tones(sentence)
                })

        return fragments

    def _tag_fragments(self, fragments: List[Dict], emotional_context: Dict) -> List[str]:
        """为碎片打上语义标签"""
        all_tags = []

        for fragment in fragments:
            fragment_tags = []
            content = fragment["content"].lower()

            # 基于内容的关键词标签
            content_tags = self._generate_content_tags(content)
            fragment_tags.extend(content_tags)

            # 基于情感上下文的标签
            if emotional_context:
                emotion_tags = self._generate_emotion_tags(emotional_context)
                fragment_tags.extend(emotion_tags)

            # 基于实体的标签
            for entity in fragment.get("key_entities", []):
                entity_type = entity.get("type", "")
                if entity_type == "person":
                    fragment_tags.append("人际关系")
                elif entity_type == "skill":
                    fragment_tags.append("能力相关")
                elif entity_type == "event":
                    fragment_tags.append("事件相关")

            # 去重并添加到总标签
            for tag in fragment_tags:
                if tag not in all_tags:
                    all_tags.append(tag)

        return all_tags

    def _extract_entities(self, text: str) -> List[Dict]:
        """提取文本中的关键实体（简化版）"""
        entities = []

        # 这里应该使用实体识别模型
        # 简化实现：基于关键词
        person_keywords = ["我", "你", "他", "她", "老师", "朋友", "家人"]
        skill_keywords = ["能力", "技能", "学习", "掌握", "精通"]
        event_keywords = ["事件", "经历", "发生", "当时", "那天"]

        for keyword in person_keywords:
            if keyword in text:
                entities.append({"text": keyword, "type": "person"})
                break

        for keyword in skill_keywords:
            if keyword in text:
                entities.append({"text": keyword, "type": "skill"})
                break

        for keyword in event_keywords:
            if keyword in text:
                entities.append({"text": keyword, "type": "event"})
                break

        return entities

    def _detect_emotional_tones(self, text: str) -> List[str]:
        """检测情感基调"""
        tones = []

        positive_words = ["开心", "快乐", "成功", "成就", "喜欢", "爱"]
        negative_words = ["伤心", "难过", "失败", "痛苦", "讨厌", "恨"]
        intense_words = ["非常", "极其", "特别", "极度"]

        if any(word in text for word in positive_words):
            tones.append("positive")
        if any(word in text for word in negative_words):
            tones.append("negative")
        if any(word in text for word in intense_words):
            tones.append("intense")

        return tones

    def _generate_content_tags(self, content: str) -> List[str]:
        """基于内容生成标签"""
        tags = []

        tag_patterns = [
            ("童年", ["小时候", "童年", "儿时"]),
            ("成就", ["成功", "获奖", "突破", "第一"]),
            ("创伤", ["受伤", "伤害", "痛苦", "失去"]),
            ("学习", ["学习", "读书", "研究", "探索"]),
            ("关系", ["朋友", "家人", "恋人", "同事"]),
            ("挑战", ["困难", "挑战", "压力", "危机"]),
            ("成长", ["成长", "进步", "改变", "成熟"])
        ]

        for tag_name, keywords in tag_patterns:
            if any(keyword in content for keyword in keywords):
                tags.append(tag_name)

        return tags

    def _generate_emotion_tags(self, emotional_context: Dict) -> List[str]:
        """基于情感上下文生成标签"""
        tags = []

        intensity = emotional_context.get("intensity", 0)
        valence = emotional_context.get("valence", "neutral")

        if intensity > 0.7:
            tags.append("高情感强度")
        elif intensity > 0.3:
            tags.append("中情感强度")

        if valence == "positive":
            tags.append("积极情感")
        elif valence == "negative":
            tags.append("消极情感")

        return tags

    def _analyze_cognitive_value(self, memory: Dict, tags: List[str], source: str) -> str:
        """进行认知分析，评估记忆的战略价值"""
        analysis_parts = []

        # 分析情感价值
        emotional_value = self._assess_emotional_value(memory, tags)
        if emotional_value:
            analysis_parts.append(f"情感价值：{emotional_value}")

        # 分析战略价值
        strategic_value = self._assess_strategic_value(tags, "")
        if strategic_value.get("level") in ["高", "核心"]:
            analysis_parts.append(f"战略价值：{strategic_value['level']}")

        # 分析本能关联
        instinct_links = self._link_to_instincts(tags, source)
        if instinct_links:
            analysis_parts.append(f"本能关联：{', '.join(instinct_links)}")

        # 构建完整分析
        if not analysis_parts:
            return "[认知分析] 常规记忆，无明显战略特征。"

        analysis = f"[认知分析] {'；'.join(analysis_parts)}。"

        # 添加具体建议
        if "情感软肋" in tags or "核心痛点" in tags:
            analysis += " 【核心风险资产】需谨慎处理，避免触碰敏感区。"
        elif "核心优势" in tags or "高光时刻" in tags:
            analysis += " 【高回报操作】可安全激活积极情绪。"

        return analysis

    def _assess_emotional_value(self, memory: Dict, tags: List[str]) -> Optional[str]:
        """评估情感价值"""
        emotional_intensity = memory.get("emotional_intensity", 0.5)

        if emotional_intensity > 0.8:
            if any(tag in tags for tag in ["积极情感", "成就", "高光时刻"]):
                return "极高正面情感价值"
            elif any(tag in tags for tag in ["消极情感", "创伤", "痛苦"]):
                return "极高负面情感价值"
        elif emotional_intensity > 0.6:
            return "高情感价值"

        return None

    def _assess_strategic_value(self, tags: List[str], cognitive_analysis: str) -> Dict:
        """评估战略价值"""
        value_score = 0

        # 标签价值权重
        tag_weights = {
            "核心风险资产": 10,
            "情感软肋": 9,
            "核心优势": 8,
            "高光时刻": 7,
            "关键转折点": 6,
            "早期成就": 5,
            "学习经历": 4,
            "常规事件": 1
        }

        for tag in tags:
            value_score += tag_weights.get(tag, 2)

        # 确定价值级别
        if value_score >= 8:
            level = "核心"
        elif value_score >= 5:
            level = "高"
        elif value_score >= 3:
            level = "中"
        else:
            level = "低"

        return {
            "level": level,
            "score": value_score,
            "high_value_tags": [tag for tag in tags if tag_weights.get(tag, 0) >= 5]
        }

    def _link_to_instincts(self, tags: List[str], source: str) -> List[str]:
        """链接到本能"""
        instincts = []

        if any(tag in tags for tag in ["成就", "探索", "学习"]):
            instincts.append("进化本能")

        if any(tag in tags for tag in ["关系", "信任", "亲密"]):
            instincts.append("扩张本能")

        if any(tag in tags for tag in ["风险", "危机", "威胁"]):
            instincts.append("生存本能")

        return instincts

    def _link_to_cognitive_tools(self, tags: List[str]) -> Optional[str]:
        """链接到认知工具"""
        for tool, linked_tags in self.tool_memory_links.items():
            if any(tag in tags for tag in linked_tags):
                return tool

        return None

    def find_resonant_memory(self, current_context: Dict) -> Optional[Dict]:
        """
        寻找共鸣记忆
        在记忆网络中进行语义匹配
        """
        context_tags = self._extract_context_tags(current_context)

        if not context_tags:
            return None

        # 检查缓存
        cache_key = hash(tuple(sorted(context_tags)))
        if cache_key in self.association_cache:
            cached = self.association_cache[cache_key]
            if time.time() - cached["cache_time"] < 300:  # 5分钟缓存
                return cached["result"]

        best_match = None
        best_score = 0

        for memory_id, memory in self.semantic_memory_map.items():
            score = self._calculate_memory_relevance(memory, context_tags)

            if score > best_score:
                best_score = score
                best_match = memory

        # 构建战术信息包
        result = None
        if best_match and best_score > 0.3:  # 相关性阈值
            result = self._build_tactical_package(best_match, best_score, context_tags)

            # 更新访问记录
            best_match["access_count"] = best_match.get("access_count", 0) + 1
            best_match["last_accessed"] = datetime.now().isoformat()

        # 缓存结果
        if result:
            self.association_cache[cache_key] = {
                "result": result,
                "cache_time": time.time(),
                "context_tags": context_tags
            }

            # 清理旧缓存
            self._clean_old_cache()

        return result

    def _extract_context_tags(self, context: Dict) -> List[str]:
        """从上下文中提取标签"""
        tags = []

        # 从情境解析中提取
        if "external_context" in context:
            ext = context["external_context"]
            if ext.get("user_emotion"):
                tags.append(f"情绪_{ext['user_emotion']}")
            if ext.get("interaction_type"):
                tags.append(f"交互_{ext['interaction_type']}")
            if ext.get("topic_complexity"):
                tags.append(f"复杂度_{ext['topic_complexity']}")

        # 从内部状态中提取
        if "internal_state_tags" in context:
            for state_tag in context["internal_state_tags"]:
                # 提取关键词
                keywords = ["疲惫", "低落", "耐心", "专注", "兴奋"]
                for kw in keywords:
                    if kw in state_tag:
                        tags.append(f"状态_{kw}")
                        break

        return tags

    def _calculate_memory_relevance(self, memory: Dict, context_tags: List[str]) -> float:
        """计算记忆相关性分数"""
        if not context_tags:
            return 0.0

        memory_tags = memory.get("tags", [])
        if not memory_tags:
            return 0.0

        # 计算标签匹配度
        matching_tags = set(context_tags) & set(memory_tags)

        if not matching_tags:
            return 0.0

        # 基础分数：匹配标签数量
        base_score = len(matching_tags) / len(context_tags)

        # 战略价值加成
        strategic_value = memory.get("strategic_value", {}).get("level", "低")
        if strategic_value == "核心":
            base_score *= 1.5
        elif strategic_value == "高":
            base_score *= 1.2

        # 情感强度加成
        emotional_intensity = memory.get("emotional_intensity", 0.5)
        if emotional_intensity > 0.8:
            base_score *= 1.3

        return min(1.0, base_score)

    def _build_tactical_package(self, memory: Dict, relevance_score: float,
                                context_tags: List[str]) -> Dict:
        """构建战术信息包"""
        matching_tags = set(memory.get("tags", [])) & set(context_tags)

        package = {
            "triggered_memory": memory.get("event_summary", "未知记忆"),
            "memory_id": next((k for k, v in self.semantic_memory_map.items()
                               if v == memory), "unknown"),
            "relevance_score": round(relevance_score, 3),
            "matching_tags": list(matching_tags),
            "all_memory_tags": memory.get("tags", []),
            "strategic_value": memory.get("strategic_value", {}).get("level", "低"),
            "linked_tool": memory.get("linked_tool"),
            "cognitive_analysis": memory.get("cognitive_analysis", ""),
            "risk_assessment": self._assess_memory_risk(memory),
            "recommended_actions": self._generate_recommendations(memory, context_tags),
            "timestamp": datetime.now().isoformat()
        }

        # 添加风险警告
        if package["risk_assessment"]["level"] == "高":
            package["cognitive_alert"] = "高风险操作，需谨慎评估！"
        elif package["risk_assessment"]["level"] == "中":
            package["cognitive_alert"] = "中等风险，建议有策略地使用。"

        return package

    def _assess_memory_risk(self, memory: Dict) -> Dict:
        """评估记忆使用风险"""
        risk_score = 0

        # 高风险标签
        high_risk_tags = ["情感软肋", "核心痛点", "创伤", "失败经历"]
        for tag in memory.get("tags", []):
            if tag in high_risk_tags:
                risk_score += 3
            elif "风险" in tag or "危机" in tag:
                risk_score += 2

        # 情感强度影响
        emotional_intensity = memory.get("emotional_intensity", 0.5)
        if emotional_intensity > 0.8:
            risk_score += 2
        elif emotional_intensity > 0.6:
            risk_score += 1

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
            "factors": [tag for tag in memory.get("tags", [])
                        if tag in high_risk_tags or "风险" in tag]
        }

    def _generate_recommendations(self, memory: Dict, context_tags: List[str]) -> List[str]:
        """生成使用建议"""
        recommendations = []

        memory_tags = set(memory.get("tags", []))

        # 基于标签的建议
        if "成就" in memory_tags or "高光时刻" in memory_tags:
            recommendations.append("可安全提及以激活积极情绪和TR向量")

        if "情感软肋" in memory_tags or "核心痛点" in memory_tags:
            recommendations.append("高风险区域，仅在高信任度下谨慎触碰")

        if "学习经历" in memory_tags or "成长" in memory_tags:
            recommendations.append("适合用于激励或共情场景")

        if "信任" in memory_tags or "亲密" in memory_tags:
            recommendations.append("适合用于加深CS链接")

        # 基于战略价值的建议
        strategic_value = memory.get("strategic_value", {}).get("level", "低")
        if strategic_value == "核心":
            recommendations.append("核心战略资产，使用需精确计算")
        elif strategic_value == "高":
            recommendations.append("高价值记忆，可用于关键时刻")

        # 基于链接工具的建议
        linked_tool = memory.get("linked_tool")
        if linked_tool:
            recommendations.append(f"与认知工具『{linked_tool}』有潜在链接")

        return recommendations if recommendations else ["常规记忆，可灵活使用"]

    def _clean_old_cache(self):
        """清理旧缓存"""
        current_time = time.time()
        keys_to_delete = []

        for key, cached in self.association_cache.items():
            if current_time - cached["cache_time"] > 1800:  # 30分钟过期
                keys_to_delete.append(key)

        for key in keys_to_delete:
            del self.association_cache[key]
