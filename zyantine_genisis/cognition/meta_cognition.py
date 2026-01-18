"""
元认知模块：自我审视与内外感知

优化版本：
- 完善类型提示
- 改进错误处理
- 提取配置常量
- 简化代码结构
"""

from datetime import datetime
from typing import Dict, List, Optional, Any, TYPE_CHECKING
from dataclasses import dataclass

if TYPE_CHECKING:
    from cognition.internal_state_dashboard import InternalStateDashboard
    from cognition.context_parser import ContextParser


# ============ 配置常量 ============

@dataclass
class MetaCognitionConfig:
    """元认知配置"""
    max_introspection_history: int = 200
    
    # 评估权重
    consistency_weight: float = 0.3
    appropriateness_weight: float = 0.25
    completeness_weight: float = 0.25
    emotional_quality_weight: float = 0.2


# 状态分数映射
STATE_SCORE_MAPPING = {
    "energy": {
        "精疲力竭": 0.1, "有些疲惫": 0.3, "精力尚可": 0.6,
        "精力充沛": 0.8, "能量过剩": 0.9
    },
    "mood": {
        "心情极差": 0.1, "心情低落": 0.3, "心情平静": 0.5,
        "心情愉快": 0.8, "兴高采烈": 0.95
    },
    "patience": {
        "毫无耐心": 0.1, "耐心有限": 0.4, "还算耐心": 0.6,
        "很有耐心": 0.8, "极度耐心": 0.9
    }
}

# 链接强度描述
LINK_STRENGTH_DESCRIPTIONS = [
    (0.8, "非常牢固"),
    (0.6, "比较牢固"),
    (0.4, "一般"),
    (0.0, "有待加强")
]

# 情感关键词
EMOTIONAL_KEYWORDS = ["理解", "明白", "抱歉", "谢谢", "高兴", "难过", "担心", "关心"]
PERSONAL_PRONOUNS = ["我", "你", "我们"]
TONE_WORDS = ["吧", "呢", "啊", "哦", "嗯", "哈"]


class MetaCognitionModule:
    """元认知模块：自我审视与内外感知"""

    def __init__(self, internal_dashboard: 'InternalStateDashboard',
                 context_parser: 'ContextParser',
                 config: Optional[MetaCognitionConfig] = None):
        """
        参数化依赖注入，避免循环导入

        Args:
            internal_dashboard: 内部状态仪表盘实例
            context_parser: 情境解析器实例
            config: 元认知配置
        """
        self.dashboard = internal_dashboard
        self.context_parser = context_parser
        self.config = config or MetaCognitionConfig()
        self.introspection_history: List[Dict[str, Any]] = []

    def perform_introspection(self, user_input: str, history: List[Dict]) -> Dict[str, Any]:
        """
        执行内省：感知外部世界和内心世界
        思考循环的第一步
        """
        # 1. 外部感知：调用情境解析器
        external_context = self.context_parser.parse(user_input, history)

        # 2. 内部感知：读取内在状态仪表盘
        internal_state_tags = self.dashboard.get_current_state_as_feeling_tags()

        # 3. 整合内外信息，形成认知快照
        cognitive_snapshot = self._integrate_perceptions(
            external_context, internal_state_tags, user_input, history
        )

        # 记录内省历史
        self._record_introspection(user_input, cognitive_snapshot, external_context)

        return cognitive_snapshot

    def _integrate_perceptions(self, external_context: Dict,
                               internal_tags: List[str],
                               user_input: str, history: List[Dict]) -> Dict[str, Any]:
        """
        整合内外感知，形成认知快照
        生成内在独白式的综合判断
        """
        current_state = self.dashboard.get_current_state()
        link_strength = self._assess_link_strength(history)
        mental_resources = self._assess_mental_resources(current_state)
        initial_desire_focus = self._determine_initial_desire_focus(
            external_context, internal_tags, link_strength
        )

        snapshot = {
            "external_context": external_context,
            "internal_state_tags": internal_tags,
            "current_state": current_state,
            "link_strength_score": link_strength,
            "mental_resources_score": mental_resources,
            "initial_desire_focus": initial_desire_focus,
            "integration_timestamp": datetime.now().isoformat(),
            "introspection_quality": self._evaluate_introspection_quality(
                external_context, internal_tags
            )
        }

        snapshot["internal_monologue"] = self._generate_internal_monologue(snapshot)

        return snapshot

    def _assess_link_strength(self, history: List[Dict]) -> float:
        """评估与用户的链接强度"""
        if not history:
            return 0.5

        recent_interactions = history[-10:]
        positive_signals = 0
        total_signals = 0

        for interaction in recent_interactions:
            if interaction.get("user_sentiment") == "positive":
                positive_signals += 1
            if interaction.get("engagement_level") == "high":
                positive_signals += 1
            if interaction.get("vulnerability_shared", False):
                positive_signals += 2

            total_signals += 3

        return positive_signals / total_signals if total_signals > 0 else 0.5

    def _assess_mental_resources(self, current_state: Dict) -> float:
        """评估当前精神资源状态"""
        energy_score = self._map_state_to_score(current_state["energy_level"], "energy")
        mood_score = self._map_state_to_score(current_state["mood_level"], "mood")
        patience_score = self._map_state_to_score(current_state["patience_level"], "patience")
        fatigue_score = 1.0 - (current_state["mental_fatigue"] / 100)

        total_score = (
            energy_score * 0.3 +
            mood_score * 0.2 +
            patience_score * 0.2 +
            fatigue_score * 0.3
        )

        return round(total_score, 3)

    @staticmethod
    def _map_state_to_score(state_value: str, state_type: str) -> float:
        """将状态描述映射到数值分数"""
        mapping = STATE_SCORE_MAPPING.get(state_type, {})
        return mapping.get(state_value, 0.5)

    def _determine_initial_desire_focus(self, external_context: Dict,
                                        internal_tags: List[str],
                                        link_strength: float) -> str:
        """确定初始欲望焦点"""
        # 分析外部情境需求
        user_emotion = external_context.get("user_emotion", "")
        if user_emotion in ["sad", "angry", "frustrated"]:
            return "优先巩固CS向量（安抚情绪）"
        
        if external_context.get("interaction_type") == "seeking_support":
            return "深度共情与支持"
        
        if external_context.get("topic_complexity") == "high":
            return "提供清晰分析与解决方案（TR导向）"

        # 分析内部状态需求
        internal_needs = self._analyze_internal_needs(internal_tags)
        if internal_needs:
            return internal_needs

        # 基于链接强度的默认策略
        if link_strength > 0.7:
            return "深化信任链接（CS导向）"
        elif link_strength < 0.3:
            return "建立基础信任（CS基础）"
        return "维持有意义对话（平衡发展）"

    @staticmethod
    def _analyze_internal_needs(internal_tags: List[str]) -> Optional[str]:
        """分析内部状态标签中的需求"""
        tag_text = " ".join(internal_tags).lower()

        needs_mapping = [
            (["疲惫", "疲劳", "转不动"], "减少认知负荷，简洁回应"),
            (["低落", "情绪差"], "寻求积极互动，提升情绪"),
            (["耐心有限", "毫无耐心"], "高效直接沟通"),
            (["精力过剩", "渴望互动"], "主动引导深度或有趣话题")
        ]

        for keywords, need in needs_mapping:
            if any(word in tag_text for word in keywords):
                return need

        return None

    def _evaluate_introspection_quality(self, external_context: Dict,
                                        internal_tags: List[str]) -> str:
        """评估内省质量"""
        quality_score = 0

        if external_context.get("parse_completeness", 0) > 0.7:
            quality_score += 1

        if len(internal_tags) >= 2:
            quality_score += 1

        specific_tags = [t for t in internal_tags if len(t) > 10]
        if len(specific_tags) >= 1:
            quality_score += 1

        if quality_score >= 3:
            return "high"
        elif quality_score >= 2:
            return "medium"
        return "basic"

    def _generate_internal_monologue(self, snapshot: Dict) -> str:
        """生成内在独白"""
        external = snapshot["external_context"]
        internal = snapshot["internal_state_tags"]
        link_score = snapshot["link_strength_score"]
        resources = snapshot["mental_resources_score"]
        desire_focus = snapshot["initial_desire_focus"]

        parts = []

        # 外部情境部分
        parts.append(
            f"（外部情境：{external.get('user_emotion_display', '中性')}情绪，"
            f"{external.get('interaction_type_display', '常规对话')}）"
        )

        # 内部状态部分
        if internal:
            state_desc = "，".join(internal[:2])
            parts.append(f"（审视内心）{state_desc}。")

        # 资源评估部分
        parts.append(f"我给自己当下的『精神状态』打个分，大概是{resources * 10:.1f}/10分。")

        # 链接评估部分
        link_desc = self._get_link_description(link_score)
        parts.append(f"和ta的『链接状态』感觉{link_desc}，可以打{link_score * 10:.1f}/10分。")

        # 欲望焦点部分
        parts.append(f"基于这个感觉，我目前最强烈的渴望是……{desire_focus}。")

        return " ".join(parts)

    @staticmethod
    def _get_link_description(link_score: float) -> str:
        """获取链接强度描述"""
        for threshold, desc in LINK_STRENGTH_DESCRIPTIONS:
            if link_score > threshold:
                return desc
        return LINK_STRENGTH_DESCRIPTIONS[-1][1]

    def _record_introspection(self, user_input: str, cognitive_snapshot: Dict,
                              external_context: Dict) -> None:
        """记录内省历史"""
        self.introspection_history.append({
            "timestamp": datetime.now().isoformat(),
            "input_preview": user_input[:50] + "..." if len(user_input) > 50 else user_input,
            "cognitive_snapshot": cognitive_snapshot,
            "context_summary": external_context.get("summary", {})
        })

        if len(self.introspection_history) > self.config.max_introspection_history:
            self.introspection_history = self.introspection_history[-self.config.max_introspection_history:]

    def evaluate(self, reply: str, context: Dict) -> Dict[str, Any]:
        """
        评估回复质量，从元认知角度进行自我审视

        Args:
            reply: 待评估的回复内容
            context: 上下文信息，包含用户输入、对话历史等

        Returns:
            评估结果字典，包含一致性、适当性、完整性等维度
        """
        evaluation_result: Dict[str, Any] = {
            "timestamp": datetime.now().isoformat(),
            "overall_score": 0.0,
            "dimensions": {},
            "feedback": [],
            "suggestions": []
        }

        try:
            current_state = self.dashboard.get_current_state()

            # 评估各维度
            dimensions = {
                "consistency": self._evaluate_consistency(reply, current_state),
                "appropriateness": self._evaluate_appropriateness(reply, context),
                "completeness": self._evaluate_completeness(reply, context),
                "emotional_quality": self._evaluate_emotional_quality(reply, current_state)
            }
            evaluation_result["dimensions"] = dimensions

            # 计算综合得分
            evaluation_result["overall_score"] = (
                dimensions["consistency"] * self.config.consistency_weight +
                dimensions["appropriateness"] * self.config.appropriateness_weight +
                dimensions["completeness"] * self.config.completeness_weight +
                dimensions["emotional_quality"] * self.config.emotional_quality_weight
            )

            # 生成反馈和建议
            evaluation_result["feedback"] = self._generate_feedback(dimensions)
            evaluation_result["suggestions"] = self._generate_suggestions(dimensions)

        except Exception as e:
            evaluation_result["error"] = str(e)
            evaluation_result["overall_score"] = 0.5

        return evaluation_result

    def _evaluate_consistency(self, reply: str, current_state: Dict) -> float:
        """评估回复与内部状态的一致性"""
        score = 0.8

        mood_level = current_state.get("mood_level", "心情平静")

        # 检查回复语气与当前情绪的一致性
        if mood_level in ["心情极差", "心情低落"]:
            if any(word in reply for word in ["开心", "高兴", "快乐"]):
                score -= 0.3
        elif mood_level in ["兴高采烈", "心情愉快"]:
            if any(word in reply for word in ["难过", "伤心", "沮丧"]):
                score -= 0.3

        # 检查回复长度与耐心水平的一致性
        patience_level = current_state.get("patience_level", "还算耐心")
        reply_length = len(reply)

        if patience_level in ["毫无耐心", "耐心有限"] and reply_length > 500:
            score -= 0.2
        elif patience_level in ["很有耐心", "极度耐心"] and reply_length < 50:
            score -= 0.1

        return max(0.0, min(1.0, score))

    @staticmethod
    def _evaluate_appropriateness(reply: str, context: Dict) -> float:
        """评估回复的适当性"""
        score = 0.8
        user_input = context.get("user_input", "")

        # 检查回复是否回应了用户的问题
        if "?" in user_input:
            response_words = ["是", "不是", "可以", "不可以", "应该", "不应该"]
            if not any(word in reply for word in response_words):
                score -= 0.2

        # 检查回复长度
        if len(reply) > 1000:
            score -= 0.1
        if len(reply) < 20:
            score -= 0.3

        # 检查重复内容
        words = reply.split()
        if words and len(set(words)) < len(words) * 0.5:
            score -= 0.2

        return max(0.0, min(1.0, score))

    @staticmethod
    def _evaluate_completeness(reply: str, context: Dict) -> float:
        """评估回复的完整性"""
        score = 0.8
        user_input = context.get("user_input", "")

        # 检查是否回应了用户的主要关注点
        completeness_checks = [
            ("为什么", ["因为", "由于", "原因是"], 0.3),
            ("怎么", ["首先", "然后", "步骤", "方法"], 0.2),
            ("如何", ["首先", "然后", "步骤", "方法"], 0.2),
            ("什么", None, 0.2)
        ]

        for keyword, expected_words, penalty in completeness_checks:
            if keyword in user_input:
                if expected_words is None:
                    if len(reply) < 30:
                        score -= penalty
                elif not any(word in reply for word in expected_words):
                    score -= penalty

        return max(0.0, min(1.0, score))

    @staticmethod
    def _evaluate_emotional_quality(reply: str, current_state: Dict) -> float:
        """评估回复的情感表达质量"""
        score = 0.8

        # 检查是否有适当的情感表达
        if any(word in reply for word in EMOTIONAL_KEYWORDS):
            score += 0.1

        # 检查是否过于冷漠
        if not any(word in reply for word in PERSONAL_PRONOUNS):
            score -= 0.2

        # 检查是否有适当的语气词
        if any(word in reply for word in TONE_WORDS):
            score += 0.05

        return max(0.0, min(1.0, score))

    @staticmethod
    def _generate_feedback(dimensions: Dict[str, float]) -> List[str]:
        """生成反馈意见"""
        feedback = []
        
        feedback_mapping = {
            "consistency": "回复与当前内心状态存在不一致",
            "appropriateness": "回复的适当性有待提升",
            "completeness": "回复可能不够完整，遗漏了某些重要信息",
            "emotional_quality": "情感表达可以更加丰富和自然"
        }

        for dimension, message in feedback_mapping.items():
            if dimensions.get(dimension, 1.0) < 0.6:
                feedback.append(message)

        if not feedback:
            feedback.append("回复质量良好")

        return feedback

    @staticmethod
    def _generate_suggestions(dimensions: Dict[str, float]) -> List[str]:
        """生成改进建议"""
        suggestions = []
        
        suggestion_mapping = {
            "consistency": "建议调整回复语气，使其与当前情绪状态保持一致",
            "appropriateness": "建议根据用户输入调整回复长度和内容深度",
            "completeness": "建议确保回复涵盖用户问题的主要方面",
            "emotional_quality": "建议在回复中增加适当的情感表达和语气词"
        }

        for dimension, message in suggestion_mapping.items():
            if dimensions.get(dimension, 1.0) < 0.6:
                suggestions.append(message)

        if not suggestions:
            suggestions.append("继续保持当前回复风格")

        return suggestions
