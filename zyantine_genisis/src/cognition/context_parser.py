import time
import json
import hashlib
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import random

class ContextParser:
    """情境解析器：纯技术性外部感知"""

    def __init__(self):
        self.parse_history = []

        # 关键词库
        self.emotion_keywords = {
            "positive": ["喜欢", "爱", "高兴", "开心", "棒", "优秀", "谢谢", "感谢"],
            "negative": ["讨厌", "恨", "生气", "愤怒", "糟糕", "差", "烦", "郁闷"],
            "sad": ["伤心", "难过", "悲伤", "哭泣", "泪", "失望"],
            "anxious": ["担心", "焦虑", "紧张", "害怕", "恐惧"]
        }

        self.complexity_indicators = {
            "high": ["为什么", "如何", "分析", "原理", "机制", "对比", "优缺点"],
            "technical": ["代码", "算法", "架构", "系统", "设计", "实现", "bug"]
        }

    def parse(self, user_input: str, history: List[Dict]) -> Dict:
        """
        解析用户输入，生成纯技术性分析
        无任何主观判断
        """
        input_lower = user_input.lower()
        input_length = len(user_input)

        # 基础解析
        parsed = {
            "raw_input_preview": user_input[:100] + "..." if len(user_input) > 100 else user_input,
            "input_length": input_length,
            "word_count": len(user_input.split()),
            "contains_question": "?" in user_input or "？" in user_input,
            "contains_exclamation": "!" in user_input or "！" in user_input,
            "parse_timestamp": datetime.now().isoformat()
        }

        # 情绪分析
        parsed["user_emotion"] = self._analyze_emotion(input_lower)
        parsed["user_emotion_display"] = self._translate_emotion(parsed["user_emotion"])

        # 话题复杂性分析
        parsed["topic_complexity"] = self._analyze_complexity(input_lower, input_length)
        parsed["topic_complexity_display"] = self._translate_complexity(parsed["topic_complexity"])

        # 交互类型分析
        parsed["interaction_type"] = self._analyze_interaction_type(input_lower, history)
        parsed["interaction_type_display"] = self._translate_interaction_type(parsed["interaction_type"])

        # 紧急程度分析
        parsed["urgency_level"] = self._analyze_urgency(input_lower)

        # 解析完整性评估
        parsed["parse_completeness"] = self._assess_parse_completeness(parsed)

        # 生成总结
        parsed["summary"] = self._generate_summary(parsed)

        # 记录解析历史
        self.parse_history.append({
            "timestamp": parsed["parse_timestamp"],
            "input_preview": parsed["raw_input_preview"],
            "parsed_result": {k: v for k, v in parsed.items() if k != "raw_input_preview"}
        })

        # 保持历史长度
        if len(self.parse_history) > 300:
            self.parse_history = self.parse_history[-300:]

        return parsed

    def _analyze_emotion(self, text: str) -> str:
        """分析文本情绪"""
        emotion_scores = {}

        for emotion, keywords in self.emotion_keywords.items():
            score = sum(1 for keyword in keywords if keyword in text)
            if score > 0:
                emotion_scores[emotion] = score

        if not emotion_scores:
            return "neutral"

        # 返回得分最高的情绪
        return max(emotion_scores.items(), key=lambda x: x[1])[0]

    def _translate_emotion(self, emotion_code: str) -> str:
        """翻译情绪代码为可读文本"""
        translations = {
            "positive": "积极", "negative": "消极", "sad": "悲伤",
            "anxious": "焦虑", "neutral": "中性"
        }
        return translations.get(emotion_code, "未知")

    def _analyze_complexity(self, text: str, length: int) -> str:
        """分析话题复杂性"""
        complexity_score = 0

        # 长度因素
        if length > 200:
            complexity_score += 2
        elif length > 100:
            complexity_score += 1

        # 技术性内容
        for indicator in self.complexity_indicators["high"]:
            if indicator in text:
                complexity_score += 2
                break

        for indicator in self.complexity_indicators["technical"]:
            if indicator in text:
                complexity_score += 3
                break

        # 包含问题
        if "?" in text or "？" in text:
            complexity_score += 1

        # 判断复杂性级别
        if complexity_score >= 4:
            return "high"
        elif complexity_score >= 2:
            return "medium"
        else:
            return "low"

    def _translate_complexity(self, complexity_code: str) -> str:
        """翻译复杂性代码"""
        translations = {"high": "高", "medium": "中", "low": "低"}
        return translations.get(complexity_code, "未知")

    def _analyze_interaction_type(self, text: str, history: List[Dict]) -> str:
        """分析交互类型"""
        # 检查是否重复提问
        if history and len(history) >= 2:
            last_inputs = [h.get("user_input", "").lower() for h in history[-2:]]
            similarity = self._calculate_similarity(text, last_inputs)
            if similarity > 0.7:
                return "repetitive_question"

        # 检查是否寻求支持
        support_keywords = ["帮助", "怎么办", "建议", "支持", "求助", "问题"]
        if any(keyword in text for keyword in support_keywords):
            return "seeking_support"

        # 检查是否分享
        share_keywords = ["告诉你", "分享", "今天", "刚才", "遇到"]
        if any(keyword in text for keyword in share_keywords):
            return "sharing_experience"

        # 检查是否命令/请求
        command_keywords = ["请", "帮我", "想要", "需要", "能不能"]
        if any(keyword in text for keyword in command_keywords):
            return "requesting_action"

        return "general_chat"

    def _translate_interaction_type(self, type_code: str) -> str:
        """翻译交互类型代码"""
        translations = {
            "repetitive_question": "重复提问",
            "seeking_support": "寻求支持",
            "sharing_experience": "分享经历",
            "requesting_action": "请求行动",
            "general_chat": "常规聊天"
        }
        return translations.get(type_code, "未知")

    def _analyze_urgency(self, text: str) -> str:
        """分析紧急程度"""
        urgent_keywords = ["紧急", "立刻", "马上", "赶紧", "快", "急需", "立刻马上"]
        if any(keyword in text for keyword in urgent_keywords):
            return "high"

        time_keywords = ["现在", "今天", "立刻", "马上"]
        if any(keyword in text for keyword in time_keywords):
            return "medium"

        return "low"

    def _calculate_similarity(self, text1: str, texts2: List[str]) -> float:
        """计算文本相似度（简化版）"""
        if not texts2:
            return 0.0

        # 使用Jaccard相似度
        words1 = set(text1.split())
        max_similarity = 0.0

        for text2 in texts2:
            words2 = set(text2.split())
            if not words1 or not words2:
                continue

            intersection = len(words1.intersection(words2))
            union = len(words1.union(words2))

            if union > 0:
                similarity = intersection / union
                max_similarity = max(max_similarity, similarity)

        return max_similarity

    def _assess_parse_completeness(self, parsed: Dict) -> float:
        """评估解析完整性"""
        completeness = 0.0
        total_checks = 0

        # 检查关键字段是否存在
        key_fields = ["user_emotion", "topic_complexity", "interaction_type", "urgency_level"]
        for field in key_fields:
            if field in parsed and parsed[field] != "unknown":
                completeness += 1
            total_checks += 1

        # 检查字段值是否具体
        if parsed.get("input_length", 0) > 10:
            completeness += 1
        total_checks += 1

        return completeness / total_checks if total_checks > 0 else 0.0

    def _generate_summary(self, parsed: Dict) -> Dict:
        """生成解析总结"""
        return {
            "emotion_summary": f"用户情绪为{parsed.get('user_emotion_display', '未知')}",
            "complexity_summary": f"话题复杂性{parsed.get('topic_complexity_display', '未知')}",
            "interaction_summary": f"交互类型为{parsed.get('interaction_type_display', '未知')}",
            "urgency_summary": f"紧急程度{parsed.get('urgency_level', '低')}",
            "overall": (
                f"{parsed.get('user_emotion_display', '情绪')}状态下的"
                f"{parsed.get('interaction_type_display', '交互')}，"
                f"复杂度{parsed.get('topic_complexity_display', '未知')}，"
                f"需要{parsed.get('urgency_level', '常规')}响应"
            )
        }
