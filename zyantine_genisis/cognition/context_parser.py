"""
情境解析器：纯技术性外部感知

优化版本：
- 预编译正则表达式
- 提取常量配置
- 改进类型提示
- 优化性能
"""

import re
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass, field


# ============ 常量配置 ============

@dataclass(frozen=True)
class EmotionKeywords:
    """情绪关键词配置"""
    positive: Tuple[str, ...] = ("喜欢", "爱", "高兴", "开心", "棒", "优秀", "谢谢", "感谢")
    negative: Tuple[str, ...] = ("讨厌", "恨", "生气", "愤怒", "糟糕", "差", "烦", "郁闷")
    sad: Tuple[str, ...] = ("伤心", "难过", "悲伤", "哭泣", "泪", "失望")
    anxious: Tuple[str, ...] = ("担心", "焦虑", "紧张", "害怕", "恐惧")


@dataclass(frozen=True)
class ComplexityIndicators:
    """复杂性指标配置"""
    high: Tuple[str, ...] = ("为什么", "如何", "分析", "原理", "机制", "对比", "优缺点")
    technical: Tuple[str, ...] = ("代码", "算法", "架构", "系统", "设计", "实现", "bug")


@dataclass(frozen=True)
class ReferentialKeywords:
    """指代性表述关键词配置"""
    remember: Tuple[str, ...] = ("记得", "还记得", "想起", "回忆")
    this: Tuple[str, ...] = ("这件事", "这件事情", "这个", "这")
    previous: Tuple[str, ...] = ("之前", "上次", "之前说的", "之前聊的", "之前提到的", 
                                  "刚才", "刚才说", "刚才聊的", "刚才提到的")
    continue_: Tuple[str, ...] = ("继续", "接着", "然后", "那", "那个", "它")


@dataclass(frozen=True)
class TopicKeywords:
    """主题关键词配置"""
    fitness: Tuple[str, ...] = ("体测", "体检", "健身", "运动", "锻炼", "测试", "成绩")
    work: Tuple[str, ...] = ("工作", "上班", "任务", "项目", "会议", "报告")
    study: Tuple[str, ...] = ("学习", "学校", "考试", "作业", "课程", "复习")


# 预编译的正则表达式
NUMBER_PATTERN = re.compile(r'\d+[公里天小时分钟个次]')

# 情绪翻译映射
EMOTION_TRANSLATIONS = {
    "positive": "积极", "negative": "消极", "sad": "悲伤",
    "anxious": "焦虑", "neutral": "中性"
}

# 复杂性翻译映射
COMPLEXITY_TRANSLATIONS = {"high": "高", "medium": "中", "low": "低"}

# 交互类型翻译映射
INTERACTION_TYPE_TRANSLATIONS = {
    "repetitive_question": "重复提问",
    "seeking_support": "寻求支持",
    "sharing_experience": "分享经历",
    "requesting_action": "请求行动",
    "general_chat": "常规聊天"
}

# 承诺关键词
PROMISE_KEYWORDS = (
    "可以帮你", "帮你", "给你", "为你",
    "制定", "计划", "方案", "建议",
    "我来", "我会", "我可以", "我能",
    "等一下", "稍后", "接下来",
    "先", "然后", "之后"
)

# 请求关键词
REQUEST_KEYWORDS = (
    "帮我", "给我", "告诉我", "教我",
    "怎么", "如何", "什么", "哪个",
    "可以吗", "能不能", "行不行",
    "开始", "继续", "来吧", "好的"
)


class ContextParser:
    """情境解析器：纯技术性外部感知"""

    # 类级别常量
    EMOTION_KEYWORDS = EmotionKeywords()
    COMPLEXITY_INDICATORS = ComplexityIndicators()
    REFERENTIAL_KEYWORDS = ReferentialKeywords()
    TOPIC_KEYWORDS = TopicKeywords()
    
    # 配置参数
    MAX_PARSE_HISTORY = 300
    MAX_TOPIC_HISTORY = 10
    TOPIC_PERSISTENCE_WEIGHT = 0.7
    MAX_COHERENCE_HISTORY = 50

    def __init__(self):
        self.parse_history: List[Dict[str, Any]] = []
        
        # 对话主题追踪
        self.current_topic: Optional[str] = None
        self.topic_confidence: float = 0.0
        self.topic_history: List[str] = []
        
        # 对话连贯性评分
        self.coherence_score: float = 1.0
        self.coherence_history: List[float] = []

    def parse(self, user_input: str, history: List[Dict]) -> Dict[str, Any]:
        """
        解析用户输入，生成纯技术性分析
        无任何主观判断
        """
        input_lower = user_input.lower()
        input_length = len(user_input)

        # 基础解析
        parsed: Dict[str, Any] = {
            "raw_input_preview": user_input[:100] + "..." if len(user_input) > 100 else user_input,
            "input_length": input_length,
            "word_count": len(user_input.split()),
            "contains_question": "?" in user_input or "？" in user_input,
            "contains_exclamation": "!" in user_input or "！" in user_input,
            "parse_timestamp": datetime.now().isoformat()
        }

        # 情绪分析
        parsed["user_emotion"] = self._analyze_emotion(input_lower)
        parsed["user_emotion_display"] = EMOTION_TRANSLATIONS.get(parsed["user_emotion"], "未知")

        # 话题复杂性分析
        parsed["topic_complexity"] = self._analyze_complexity(input_lower, input_length)
        parsed["topic_complexity_display"] = COMPLEXITY_TRANSLATIONS.get(parsed["topic_complexity"], "未知")

        # 交互类型分析
        parsed["interaction_type"] = self._analyze_interaction_type(input_lower, history)
        parsed["interaction_type_display"] = INTERACTION_TYPE_TRANSLATIONS.get(
            parsed["interaction_type"], "未知"
        )

        # 紧急程度分析
        parsed["urgency_level"] = self._analyze_urgency(input_lower)

        # 指代性表述分析
        parsed["referential_analysis"] = self._analyze_referential(input_lower)
        parsed["contains_referential"] = parsed["referential_analysis"]["contains_referential"]

        # 对话主题分析
        parsed["topic_analysis"] = self._analyze_topic(input_lower, history)
        parsed["current_topic"] = parsed["topic_analysis"]["topic"]
        parsed["topic_confidence"] = parsed["topic_analysis"]["confidence"]

        # 上下文关联分析
        parsed["context_analysis"] = self._analyze_context_association(input_lower, history)
        parsed["context_relevance"] = parsed["context_analysis"]["relevance_score"]
        
        # 前文承诺和上下文链接分析
        parsed["context_links"] = self._extract_context_links(input_lower, history)
        parsed["pending_promises"] = parsed["context_links"]["pending_promises"]
        parsed["likely_reference"] = parsed["context_links"]["likely_reference"]
        parsed["has_unresolved_context"] = parsed["context_links"]["has_unresolved_context"]
        
        # 对话连贯性评分
        parsed["coherence_analysis"] = self._analyze_coherence(input_lower, history, parsed)
        parsed["coherence_score"] = parsed["coherence_analysis"]["score"]

        # 解析完整性评估
        parsed["parse_completeness"] = self._assess_parse_completeness(parsed)

        # 生成总结
        parsed["summary"] = self._generate_summary(parsed)

        # 更新状态
        self._update_state(parsed)

        return parsed

    def _analyze_emotion(self, text: str) -> str:
        """分析文本情绪"""
        emotion_scores: Dict[str, int] = {}
        
        emotion_map = {
            "positive": self.EMOTION_KEYWORDS.positive,
            "negative": self.EMOTION_KEYWORDS.negative,
            "sad": self.EMOTION_KEYWORDS.sad,
            "anxious": self.EMOTION_KEYWORDS.anxious
        }

        for emotion, keywords in emotion_map.items():
            score = sum(1 for keyword in keywords if keyword in text)
            if score > 0:
                emotion_scores[emotion] = score

        if not emotion_scores:
            return "neutral"

        return max(emotion_scores.items(), key=lambda x: x[1])[0]

    def _analyze_complexity(self, text: str, length: int) -> str:
        """分析话题复杂性"""
        complexity_score = 0

        # 长度因素
        if length > 200:
            complexity_score += 2
        elif length > 100:
            complexity_score += 1

        # 技术性内容
        if any(indicator in text for indicator in self.COMPLEXITY_INDICATORS.high):
            complexity_score += 2

        if any(indicator in text for indicator in self.COMPLEXITY_INDICATORS.technical):
            complexity_score += 3

        # 包含问题
        if "?" in text or "？" in text:
            complexity_score += 1

        # 判断复杂性级别
        if complexity_score >= 4:
            return "high"
        elif complexity_score >= 2:
            return "medium"
        return "low"

    def _analyze_interaction_type(self, text: str, history: List[Dict]) -> str:
        """分析交互类型"""
        # 检查是否重复提问
        if history and len(history) >= 2:
            last_inputs = [h.get("user_input", "").lower() for h in history[-2:]]
            similarity = self._calculate_similarity(text, last_inputs)
            if similarity > 0.7:
                return "repetitive_question"

        # 检查是否寻求支持
        support_keywords = ("帮助", "怎么办", "建议", "支持", "求助", "问题")
        if any(keyword in text for keyword in support_keywords):
            return "seeking_support"

        # 检查是否分享
        share_keywords = ("告诉你", "分享", "今天", "刚才", "遇到")
        if any(keyword in text for keyword in share_keywords):
            return "sharing_experience"

        # 检查是否命令/请求
        command_keywords = ("请", "帮我", "想要", "需要", "能不能")
        if any(keyword in text for keyword in command_keywords):
            return "requesting_action"

        return "general_chat"

    def _analyze_urgency(self, text: str) -> str:
        """分析紧急程度"""
        urgent_keywords = ("紧急", "立刻", "马上", "赶紧", "快", "急需", "立刻马上")
        if any(keyword in text for keyword in urgent_keywords):
            return "high"

        time_keywords = ("现在", "今天", "立刻", "马上")
        if any(keyword in text for keyword in time_keywords):
            return "medium"

        return "low"

    def _calculate_similarity(self, text1: str, texts2: List[str]) -> float:
        """计算文本相似度（Jaccard相似度）"""
        if not texts2:
            return 0.0

        words1 = set(text1.split())
        max_similarity = 0.0

        for text2 in texts2:
            words2 = set(text2.split())
            if not words1 or not words2:
                continue

            intersection = len(words1 & words2)
            union = len(words1 | words2)

            if union > 0:
                similarity = intersection / union
                max_similarity = max(max_similarity, similarity)

        return max_similarity

    def _assess_parse_completeness(self, parsed: Dict) -> float:
        """评估解析完整性"""
        key_fields = ["user_emotion", "topic_complexity", "interaction_type", 
                      "urgency_level", "current_topic"]
        
        completeness = sum(
            1 for field in key_fields 
            if field in parsed and parsed[field] != "unknown"
        )
        total_checks = len(key_fields)

        # 检查字段值是否具体
        if parsed.get("input_length", 0) > 10:
            completeness += 1
        total_checks += 1

        return completeness / total_checks if total_checks > 0 else 0.0

    def _generate_summary(self, parsed: Dict) -> Dict[str, str]:
        """生成解析总结"""
        coherence_score = parsed.get('coherence_score', 1.0)
        coherence_status = "高" if coherence_score >= 0.7 else ("中" if coherence_score >= 0.4 else "低")
        
        return {
            "emotion_summary": f"用户情绪为{parsed.get('user_emotion_display', '未知')}",
            "complexity_summary": f"话题复杂性{parsed.get('topic_complexity_display', '未知')}",
            "interaction_summary": f"交互类型为{parsed.get('interaction_type_display', '未知')}",
            "urgency_summary": f"紧急程度{parsed.get('urgency_level', '低')}",
            "topic_summary": f"当前主题为{parsed.get('current_topic', '未知')}（置信度{parsed.get('topic_confidence', 0.0):.2f}）",
            "referential_summary": f"{'包含指代性表述' if parsed.get('contains_referential', False) else '不包含指代性表述'}",
            "context_summary": f"上下文关联度{parsed.get('context_relevance', 0.0):.2f}",
            "coherence_summary": f"对话连贯性{coherence_status}（评分{coherence_score:.2f}）",
            "overall": (
                f"{parsed.get('user_emotion_display', '情绪')}状态下的"
                f"{parsed.get('interaction_type_display', '交互')}，"
                f"话题为{parsed.get('current_topic', '未知')}，"
                f"复杂度{parsed.get('topic_complexity_display', '未知')}，"
                f"连贯性{coherence_status}，"
                f"需要{parsed.get('urgency_level', '常规')}响应"
            )
        }

    def _analyze_referential(self, text: str) -> Dict[str, Any]:
        """分析文本中的指代性表述"""
        contains_referential = False
        referential_types: List[str] = []
        referential_keywords: List[str] = []

        ref_map = {
            "remember": self.REFERENTIAL_KEYWORDS.remember,
            "this": self.REFERENTIAL_KEYWORDS.this,
            "previous": self.REFERENTIAL_KEYWORDS.previous,
            "continue": self.REFERENTIAL_KEYWORDS.continue_
        }

        for ref_type, keywords in ref_map.items():
            for keyword in keywords:
                if keyword in text:
                    contains_referential = True
                    if ref_type not in referential_types:
                        referential_types.append(ref_type)
                    referential_keywords.append(keyword)

        return {
            "contains_referential": contains_referential,
            "referential_types": referential_types,
            "referential_keywords": referential_keywords,
            "referential_count": len(referential_keywords)
        }

    def _analyze_topic(self, text: str, history: List[Dict]) -> Dict[str, Any]:
        """
        分析对话主题 - 优化版：增强主题持续性和连贯性
        """
        topic_scores: Dict[str, int] = {}
        max_score = 0
        best_topic = "unknown"

        topic_map = {
            "fitness": self.TOPIC_KEYWORDS.fitness,
            "work": self.TOPIC_KEYWORDS.work,
            "study": self.TOPIC_KEYWORDS.study
        }

        # 基于关键词计算主题分数
        for topic, keywords in topic_map.items():
            score = sum(1 for keyword in keywords if keyword in text)
            if score > 0:
                topic_scores[topic] = score
                if score > max_score:
                    max_score = score
                    best_topic = topic

        # 结合历史对话主题
        recent_topics = self._extract_recent_topics(history, topic_map)

        # 结合主题历史
        if self.topic_history:
            recent_topics.extend(self.topic_history[-3:])

        # 处理历史主题
        if recent_topics:
            best_topic, max_score = self._process_topic_history(
                best_topic, max_score, recent_topics
            )

        # 计算置信度
        confidence = min(max_score / 5.0, 1.0) if max_score > 0 else 0.0

        # 主题持续性
        if confidence < 0.3 and self.current_topic and self.current_topic != "unknown":
            best_topic = self.current_topic
            confidence = max(0.3, self.topic_confidence * self.TOPIC_PERSISTENCE_WEIGHT)

        return {
            "topic": best_topic,
            "confidence": confidence,
            "scores": topic_scores,
            "max_score": max_score,
            "topic_history": self.topic_history[-5:] if self.topic_history else []
        }

    def _extract_recent_topics(self, history: List[Dict], topic_map: Dict) -> List[str]:
        """从历史中提取最近的主题"""
        recent_topics: List[str] = []
        
        if not history:
            return recent_topics
            
        for h in history[-5:]:
            if not isinstance(h, dict):
                continue
                
            if "topic_analysis" in h:
                recent_topics.append(h["topic_analysis"]["topic"])
            elif "topics" in h and h["topics"]:
                recent_topics.extend(h["topics"])
            else:
                content = h.get("user_input", "") or h.get("content", "")
                if content:
                    content_lower = content.lower()
                    for topic, keywords in topic_map.items():
                        if any(keyword in content_lower for keyword in keywords):
                            recent_topics.append(topic)
                            break
        
        return recent_topics

    def _process_topic_history(self, best_topic: str, max_score: int, 
                               recent_topics: List[str]) -> Tuple[str, int]:
        """处理主题历史，更新最佳主题和分数"""
        topic_counts: Dict[str, int] = {}
        for topic in recent_topics:
            if topic and topic != "unknown":
                topic_counts[topic] = topic_counts.get(topic, 0) + 1
        
        if not topic_counts:
            return best_topic, max_score
            
        most_common_topic = max(topic_counts.items(), key=lambda x: x[1])[0]
        most_common_count = topic_counts[most_common_topic]

        if best_topic != "unknown" and best_topic == most_common_topic:
            max_score += 2 + most_common_count
        elif best_topic == "unknown":
            best_topic = most_common_topic
            max_score = int(1 + most_common_count * 0.5)
        elif most_common_count >= 3 and max_score < 2:
            best_topic = most_common_topic
            max_score = int(most_common_count * 0.8)
            
        return best_topic, max_score

    def _extract_context_links(self, text: str, history: List[Dict]) -> Dict[str, Any]:
        """
        提取前文中的承诺、待办事项和上下文关联
        """
        result: Dict[str, Any] = {
            "pending_promises": [],
            "active_topics": [],
            "context_entities": [],
            "likely_reference": None,
            "has_unresolved_context": False
        }
        
        if not history:
            return result
        
        is_request = any(kw in text for kw in REQUEST_KEYWORDS)
        has_reference = self._has_referential_keywords(text)
        
        # 遍历最近的对话历史
        recent_history = history[-5:] if len(history) > 5 else history
        
        for i, conv in enumerate(reversed(recent_history)):
            if not isinstance(conv, dict):
                continue
            
            system_response = conv.get("system_response", "") or conv.get("response", "")
            user_input = conv.get("user_input", "") or conv.get("content", "")
            
            if not system_response:
                continue
            
            # 检查AI回复中的承诺
            for promise_kw in PROMISE_KEYWORDS:
                if promise_kw in system_response:
                    promise_context = self._extract_promise_context(
                        system_response, promise_kw, user_input
                    )
                    if promise_context:
                        result["pending_promises"].append({
                            "promise": promise_context,
                            "original_topic": user_input[:100] if user_input else "",
                            "turns_ago": i + 1
                        })
            
            # 提取话题关键词
            self._extract_active_topics(user_input, system_response, result)
            
            # 提取上下文实体
            entities = self._extract_entities(user_input + " " + system_response)
            result["context_entities"].extend(entities)
        
        # 去重
        result["context_entities"] = list(set(result["context_entities"]))[:10]
        
        # 推断指代内容
        if has_reference and is_request and result["pending_promises"]:
            result["likely_reference"] = result["pending_promises"][0]
            result["has_unresolved_context"] = True
        
        return result

    def _has_referential_keywords(self, text: str) -> bool:
        """检查文本是否包含指代性关键词"""
        all_ref_keywords = (
            self.REFERENTIAL_KEYWORDS.remember +
            self.REFERENTIAL_KEYWORDS.this +
            self.REFERENTIAL_KEYWORDS.previous +
            self.REFERENTIAL_KEYWORDS.continue_
        )
        return any(kw in text for kw in all_ref_keywords)
    
    def _extract_promise_context(self, response: str, promise_keyword: str, 
                                  original_input: str) -> Optional[str]:
        """从AI回复中提取承诺的具体内容"""
        idx = response.find(promise_keyword)
        if idx == -1:
            return None
        
        start = idx
        end = min(idx + 50, len(response))
        
        for punct in ("。", "，", "！", "？", "\n"):
            punct_idx = response.find(punct, idx)
            if punct_idx != -1 and punct_idx < end:
                end = punct_idx + 1
                break
        
        promise_text = response[start:end].strip()
        
        if original_input:
            topic_map = {
                "fitness": self.TOPIC_KEYWORDS.fitness,
                "work": self.TOPIC_KEYWORDS.work,
                "study": self.TOPIC_KEYWORDS.study
            }
            for _, keywords in topic_map.items():
                for kw in keywords:
                    if kw in original_input:
                        return f"{promise_text}（关于{kw}）"
        
        return promise_text

    def _extract_active_topics(self, user_input: str, system_response: str, 
                                result: Dict[str, Any]) -> None:
        """提取活跃话题"""
        topic_map = {
            "fitness": self.TOPIC_KEYWORDS.fitness,
            "work": self.TOPIC_KEYWORDS.work,
            "study": self.TOPIC_KEYWORDS.study
        }
        
        combined_text = (user_input.lower() + " " + system_response.lower())
        for topic, keywords in topic_map.items():
            if any(kw in combined_text for kw in keywords):
                if topic not in result["active_topics"]:
                    result["active_topics"].append(topic)
    
    def _extract_entities(self, text: str) -> List[str]:
        """从文本中提取关键实体"""
        entities: List[str] = []
        
        # 主题关键词
        topic_map = {
            "fitness": self.TOPIC_KEYWORDS.fitness,
            "work": self.TOPIC_KEYWORDS.work,
            "study": self.TOPIC_KEYWORDS.study
        }
        
        for _, keywords in topic_map.items():
            for kw in keywords:
                if kw in text:
                    entities.append(kw)
        
        # 数字+单位模式
        number_patterns = NUMBER_PATTERN.findall(text)
        entities.extend(number_patterns)
        
        return entities

    def _analyze_coherence(self, text: str, history: List[Dict], 
                           parsed: Dict) -> Dict[str, Any]:
        """
        分析对话连贯性
        """
        if not history:
            return {
                "score": 1.0,
                "factors": {"no_history": True},
                "is_coherent": True,
                "suggestions": []
            }
        
        factors: Dict[str, float] = {}
        
        # 1. 主题一致性评分
        factors["topic_consistency"] = self._calculate_topic_consistency(parsed)
        
        # 2. 关键词重叠评分
        factors["keyword_overlap"] = self._calculate_keyword_overlap(text, history)
        
        # 3. 指代性表述评分
        factors["referential"] = self._calculate_referential_score(parsed, history)
        
        # 4. 上下文关联评分
        factors["context_relevance"] = parsed.get("context_relevance", 0.5)
        
        # 计算综合连贯性评分
        weights = {
            "topic_consistency": 0.35,
            "keyword_overlap": 0.25,
            "referential": 0.2,
            "context_relevance": 0.2
        }
        
        coherence_score = sum(factors[k] * weights[k] for k in weights)
        
        # 生成建议
        suggestions = self._generate_coherence_suggestions(factors)
        
        return {
            "score": round(coherence_score, 2),
            "factors": factors,
            "is_coherent": coherence_score >= 0.6,
            "suggestions": suggestions
        }

    def _calculate_topic_consistency(self, parsed: Dict) -> float:
        """计算主题一致性评分"""
        current_topic = parsed.get("current_topic", "unknown")
        
        if not self.topic_history:
            return 1.0
            
        recent_topic = self.topic_history[-1]
        if not recent_topic or recent_topic == "unknown":
            return 1.0
            
        if current_topic == recent_topic:
            return 1.0
        elif current_topic == "unknown":
            return 0.7
        return 0.4

    def _calculate_keyword_overlap(self, text: str, history: List[Dict]) -> float:
        """计算关键词重叠评分"""
        current_words = set(text.split())
        history_words: set = set()
        
        for h in history[-3:]:
            if isinstance(h, dict):
                content = h.get("user_input", "") or h.get("content", "")
                if content:
                    history_words.update(content.lower().split())
        
        if not current_words or not history_words:
            return 0.5
            
        overlap = len(current_words & history_words)
        overlap_ratio = overlap / len(current_words)
        return min(1.0, overlap_ratio * 2)

    def _calculate_referential_score(self, parsed: Dict, history: List[Dict]) -> float:
        """计算指代性表述评分"""
        referential_analysis = parsed.get("referential_analysis", {})
        contains_referential = (
            referential_analysis.get("contains_referential", False) 
            if isinstance(referential_analysis, dict) else False
        )
        
        if contains_referential:
            return 1.0 if history else 0.3
        return 0.7

    def _generate_coherence_suggestions(self, factors: Dict[str, float]) -> List[str]:
        """生成连贯性建议"""
        suggestions: List[str] = []
        
        if factors.get("topic_consistency", 1.0) < 0.5:
            suggestions.append("话题可能发生了切换，注意保持上下文连贯")
        if factors.get("keyword_overlap", 1.0) < 0.3:
            suggestions.append("与之前对话的关键词重叠较少，可能需要更多上下文")
            
        return suggestions

    def _analyze_context_association(self, text: str, history: List[Dict]) -> Dict[str, Any]:
        """分析当前对话与历史对话的关联度"""
        if not history:
            return {
                "relevance_score": 0.0,
                "most_relevant_turn": None,
                "relevant_turns": []
            }

        relevance_scores: List[Tuple[int, float]] = []
        relevant_turns: List[Dict] = []

        for i, turn in enumerate(history):
            if not isinstance(turn, dict) or "content" not in turn:
                continue
                
            history_content = turn["content"].lower()
            current_words = set(text.split())
            history_words = set(history_content.split())
            
            if not current_words or not history_words:
                similarity = 0.0
            else:
                overlap = len(current_words & history_words)
                similarity = overlap / len(current_words)

            time_decay = (i + 1) / len(history)
            relevance = similarity * time_decay
            
            relevance_scores.append((i, relevance))
            
            if relevance > 0.3:
                relevant_turns.append({
                    "turn_index": i,
                    "content_preview": (history_content[:50] + "..." 
                                       if len(history_content) > 50 else history_content),
                    "relevance_score": relevance
                })

        if relevance_scores:
            most_relevant_turn = max(relevance_scores, key=lambda x: x[1])[0]
            avg_relevance = sum(score for _, score in relevance_scores) / len(relevance_scores)
        else:
            most_relevant_turn = None
            avg_relevance = 0.0

        relevant_turns.sort(key=lambda x: x["relevance_score"], reverse=True)

        return {
            "relevance_score": avg_relevance,
            "most_relevant_turn": most_relevant_turn,
            "relevant_turns": relevant_turns[:3]
        }

    def _update_state(self, parsed: Dict) -> None:
        """更新解析器状态"""
        # 记录解析历史
        self.parse_history.append({
            "timestamp": parsed["parse_timestamp"],
            "input_preview": parsed["raw_input_preview"],
            "parsed_result": {k: v for k, v in parsed.items() if k != "raw_input_preview"}
        })

        if len(self.parse_history) > self.MAX_PARSE_HISTORY:
            self.parse_history = self.parse_history[-self.MAX_PARSE_HISTORY:]

        # 更新当前主题
        self.current_topic = parsed["current_topic"]
        self.topic_confidence = parsed["topic_confidence"]
        
        # 更新主题历史
        if parsed["current_topic"] and parsed["current_topic"] != "unknown":
            self.topic_history.append(parsed["current_topic"])
            if len(self.topic_history) > self.MAX_TOPIC_HISTORY:
                self.topic_history = self.topic_history[-self.MAX_TOPIC_HISTORY:]
        
        # 更新连贯性评分
        self.coherence_score = parsed["coherence_score"]
        self.coherence_history.append(self.coherence_score)
        if len(self.coherence_history) > self.MAX_COHERENCE_HISTORY:
            self.coherence_history = self.coherence_history[-self.MAX_COHERENCE_HISTORY:]
