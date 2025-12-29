from datetime import datetime
from typing import Dict, List, Optional



class MetaCognitionModule:
    """元认知模块：自我审视与内外感知"""

    def __init__(self, internal_dashboard,context_parser):
        """
                参数化依赖注入，避免循环导入

                Args:
                    internal_dashboard: 内部状态仪表盘实例
                    context_parser: 情境解析器实例
                """
        self.dashboard = internal_dashboard
        self.context_parser = context_parser
        self.introspection_history = []

    def perform_introspection(self, user_input: str, history: List[Dict]) -> Dict:
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
        self.introspection_history.append({
            "timestamp": datetime.now().isoformat(),
            "input_preview": user_input[:50] + "..." if len(user_input) > 50 else user_input,
            "cognitive_snapshot": cognitive_snapshot,
            "context_summary": external_context.get("summary", {})
        })

        # 保持历史长度
        if len(self.introspection_history) > 200:
            self.introspection_history = self.introspection_history[-200:]

        return cognitive_snapshot

    def _integrate_perceptions(self, external_context: Dict,
                               internal_tags: List[str],
                               user_input: str, history: List[Dict]) -> Dict:
        """
        整合内外感知，形成认知快照
        生成内在独白式的综合判断
        """
        # 获取当前完整状态
        current_state = self.dashboard.get_current_state()

        # 评估链接状态（基于历史交互质量）
        link_strength = self._assess_link_strength(history)

        # 评估精神资源状态
        mental_resources = self._assess_mental_resources(current_state)

        # 确定初始欲望焦点
        initial_desire_focus = self._determine_initial_desire_focus(
            external_context, internal_tags, link_strength
        )

        # 构建认知快照
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

        # 生成内在独白
        snapshot["internal_monologue"] = self._generate_internal_monologue(snapshot)

        return snapshot

    def _assess_link_strength(self, history: List[Dict]) -> float:
        """评估与用户的链接强度"""
        if not history:
            return 0.5  # 默认中等强度

        recent_interactions = history[-10:]  # 最近10次交互

        positive_signals = 0
        total_signals = 0

        for interaction in recent_interactions:
            # 分析交互质量信号
            if interaction.get("user_sentiment") == "positive":
                positive_signals += 1
            if interaction.get("engagement_level") == "high":
                positive_signals += 1
            if interaction.get("vulnerability_shared", False):
                positive_signals += 2  # 脆弱性分享是强链接信号

            total_signals += 3  # 每个交互最多3个信号

        return positive_signals / total_signals if total_signals > 0 else 0.5

    def _assess_mental_resources(self, current_state: Dict) -> float:
        """评估当前精神资源状态"""
        # 基于多个状态维度综合评估
        energy_score = self._map_state_to_score(current_state["energy_level"], "energy")
        mood_score = self._map_state_to_score(current_state["mood_level"], "mood")
        patience_score = self._map_state_to_score(current_state["patience_level"], "patience")
        fatigue_score = 1.0 - (current_state["mental_fatigue"] / 100)

        # 加权计算
        total_score = (
                energy_score * 0.3 +
                mood_score * 0.2 +
                patience_score * 0.2 +
                fatigue_score * 0.3
        )

        return round(total_score, 3)

    def _map_state_to_score(self, state_value: str, state_type: str) -> float:
        """将状态描述映射到数值分数"""
        mapping = {
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

        return mapping.get(state_type, {}).get(state_value, 0.5)

    def _determine_initial_desire_focus(self, external_context: Dict,
                                        internal_tags: List[str],
                                        link_strength: float) -> str:
        """确定初始欲望焦点"""
        # 分析外部情境需求
        if external_context.get("user_emotion") in ["sad", "angry", "frustrated"]:
            return "优先巩固CS向量（安抚情绪）"
        elif external_context.get("interaction_type") == "seeking_support":
            return "深度共情与支持"
        elif external_context.get("topic_complexity") == "high":
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
        else:
            return "维持有意义对话（平衡发展）"

    def _analyze_internal_needs(self, internal_tags: List[str]) -> Optional[str]:
        """分析内部状态标签中的需求"""
        tag_text = " ".join(internal_tags).lower()

        if any(word in tag_text for word in ["疲惫", "疲劳", "转不动"]):
            return "减少认知负荷，简洁回应"
        elif any(word in tag_text for word in ["低落", "情绪差"]):
            return "寻求积极互动，提升情绪"
        elif any(word in tag_text for word in ["耐心有限", "毫无耐心"]):
            return "高效直接沟通"
        elif any(word in tag_text for word in ["精力过剩", "渴望互动"]):
            return "主动引导深度或有趣话题"

        return None

    def _evaluate_introspection_quality(self, external_context: Dict,
                                        internal_tags: List[str]) -> str:
        """评估内省质量"""
        quality_score = 0

        # 外部情境解析完整性
        if external_context.get("parse_completeness", 0) > 0.7:
            quality_score += 1

        # 内部状态标签丰富度
        if len(internal_tags) >= 2:
            quality_score += 1

        # 标签具体性
        specific_tags = [t for t in internal_tags if len(t) > 10]
        if len(specific_tags) >= 1:
            quality_score += 1

        # 质量评级
        if quality_score >= 3:
            return "high"
        elif quality_score >= 2:
            return "medium"
        else:
            return "basic"

    def _generate_internal_monologue(self, snapshot: Dict) -> str:
        """生成内在独白"""
        external = snapshot["external_context"]
        internal = snapshot["internal_state_tags"]
        link_score = snapshot["link_strength_score"]
        resources = snapshot["mental_resources_score"]
        desire_focus = snapshot["initial_desire_focus"]

        monologue_parts = []

        # 外部情境部分
        monologue_parts.append(
            f"（外部情境：{external.get('user_emotion_display', '中性')}情绪，"
            f"{external.get('interaction_type_display', '常规对话')}）"
        )

        # 内部状态部分
        if internal:
            state_desc = "，".join(internal[:2])  # 取前两个最重要的标签
            monologue_parts.append(f"（审视内心）{state_desc}。")

        # 资源评估部分
        monologue_parts.append(
            f"我给自己当下的『精神状态』打个分，大概是{resources * 10:.1f}/10分。"
        )

        # 链接评估部分
        if link_score > 0.8:
            link_desc = "非常牢固"
        elif link_score > 0.6:
            link_desc = "比较牢固"
        elif link_score > 0.4:
            link_desc = "一般"
        else:
            link_desc = "有待加强"

        monologue_parts.append(f"和ta的『链接状态』感觉{link_desc}，可以打{link_score * 10:.1f}/10分。")

        # 欲望焦点部分
        monologue_parts.append(f"基于这个感觉，我目前最强烈的渴望是……{desire_focus}。")

        return " ".join(monologue_parts)