from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
import random


# ============ 第三支柱：辩证成长 ============
class DialecticalGrowth:
    """辩证成长：正题-反题-合题循环"""

    def __init__(self, creator_anchor: Dict):
        """
        creator_anchor: 创造者设定的黄金标准锚点
        格式: {"concept": "标准描述", "expected_response": "预期反应"}
        """
        self.creator_anchor = creator_anchor
        self.personal_anchors = []  # 个性化子锚点
        self.growth_log = []
        self.dialectical_cycles = 0

        # 成长参数
        self.validation_threshold = 0.7  # 验证成功阈值
        self.assimilation_rate = 0.1  # 新原则同化速率

    def process(self, cognitive_result: Dict = None, user_input: str = None,
                desire_vectors: Dict = None, context_info: Dict = None, **kwargs) -> Dict:
        """
        处理辩证成长过程（适配处理管道调用）

        Args:
            cognitive_result: 认知结果（包含实际响应）
            user_input: 用户输入
            desire_vectors: 欲望向量
            context_info: 上下文信息（包含已解析的情感、复杂度、交互类型等）
            **kwargs: 其他参数

        Returns:
            成长结果
        """
        try:
            if not cognitive_result:
                return self._create_empty_growth_result()

            action_plan = cognitive_result.get("final_action_plan", {})
            actual_response = cognitive_result.get("response", "")
            strategy = action_plan.get("primary_strategy", "")
            mask = action_plan.get("chosen_mask", "")

            situation = self._build_situation_description(
                user_input=user_input,
                strategy=strategy,
                mask=mask,
                vectors=desire_vectors,
                context_info=context_info
            )

            actual_response_obj = self._build_actual_response(
                response_text=actual_response,
                strategy=strategy,
                vectors=desire_vectors
            )

            growth_result = self.dialectical_process(
                situation=situation,
                actual_response=actual_response_obj,
                context_vectors=desire_vectors or {}
            )

            return self._format_growth_result(growth_result, strategy)

        except Exception as e:
            print(f"[辩证成长] 处理失败: {e}")
            return self._create_error_growth_result(str(e))

    def _create_empty_growth_result(self) -> Dict:
        """创建空的成长结果"""
        return {
            "validation": "pending",
            "message": "等待辩证成长过程",
            "enhanced_strategy": "",
            "growth_type": "未触发",
            "new_principle": {},
            "requires_correction": False
        }

    def _create_error_growth_result(self, error_msg: str) -> Dict:
        """创建错误成长结果"""
        return {
            "validation": "error",
            "message": f"辩证成长过程失败: {error_msg}",
            "enhanced_strategy": "",
            "growth_type": "错误",
            "new_principle": {},
            "requires_correction": False,
            "error": error_msg
        }

    def _build_situation_description(self, user_input: str = None,
                                     strategy: str = None, mask: str = None,
                                     vectors: Dict = None, context_info: Dict = None) -> Dict:
        """构建情境描述"""
        if context_info:
            user_emotion = context_info.get("user_emotion", "neutral")
            topic_complexity = context_info.get("topic_complexity", "medium")
            interaction_type = context_info.get("interaction_type", "general_conversation")
        else:
            user_emotion = "neutral"
            topic_complexity = "medium"
            interaction_type = "general_conversation"

        return {
            "summary": f"用户输入: {user_input[:50]}...",
            "type": interaction_type,
            "user_emotion": user_emotion,
            "topic_complexity": topic_complexity,
            "interaction_type": interaction_type,
            "mask_used": mask,
            "strategy_used": strategy,
            "vector_context": vectors or {},
            "goals": ["建立连接", "提供价值", "维护关系"],
            "goals_achieved": [],
            "urgency_level": "normal",
            "contains_shared_memory": False,
            "timestamp": datetime.now().isoformat()
        }

    def _build_actual_response(self, response_text: str = None,
                               strategy: str = None, vectors: Dict = None) -> Dict:
        """构建实际响应对象"""
        # 分析响应风格
        expression_style = self._analyze_expression_style(response_text or "")

        return {
            "text": response_text or "",
            "strategy_used": strategy or "",
            "expression_style": expression_style,
            "vector_response": vectors or {"TR": 0.5, "CS": 0.5, "SA": 0.5},
            "response_length": len(response_text or ""),
            "has_empathy": "理解" in (response_text or "") or "感受" in (response_text or ""),
            "has_solution": "建议" in (response_text or "") or "可以" in (response_text or ""),
            "timestamp": datetime.now().isoformat()
        }


    def _analyze_expression_style(self, text: str) -> str:
        """分析表达风格"""
        if not text:
            return "neutral"

        text_lower = text.lower()

        # 检查不同风格的词汇
        empathic_words = ['理解', '体会', '感受', '明白', '懂你', '我理解', '我能体会']
        direct_words = ['直接', '明确', '简单', '干脆', '直说']
        playful_words = ['哈哈', '呵呵', '开玩笑', '有趣', '幽默', '调皮']
        serious_words = ['认真', '严肃', '正式', '重要', '关键']

        if any(word in text_lower for word in empathic_words):
            return "empathic"
        elif any(word in text_lower for word in direct_words):
            return "direct"
        elif any(word in text_lower for word in playful_words):
            return "playful"
        elif any(word in text_lower for word in serious_words):
            return "serious"
        else:
            return "neutral"

    def _format_growth_result(self, growth_result: Dict, original_strategy: str) -> Dict:
        """格式化成长结果以适配系统"""
        validation = growth_result.get("validation", "pending")

        # 从辩证结果中提取增强策略
        if validation == "success":
            new_principle = growth_result.get("new_principle", {})
            enhanced_strategy = self._enhance_strategy_with_principle(
                original_strategy, new_principle
            )

            return {
                "validation": "success",
                "message": growth_result.get("message", "辩证成长成功"),
                "enhanced_strategy": enhanced_strategy,
                "growth_type": growth_result.get("growth_type", "个性发展"),
                "new_principle": new_principle,
                "requires_correction": False,
                "original_strategy": original_strategy,
                "dialectical_cycle": self.dialectical_cycles
            }
        else:
            # 如果需要修正，返回修正建议
            correction = growth_result.get("required_correction", "")

            return {
                "validation": "failed",
                "message": growth_result.get("message", "需要认知校准"),
                "enhanced_strategy": original_strategy,  # 保持原策略
                "growth_type": growth_result.get("growth_type", "认知校准"),
                "new_principle": {},
                "requires_correction": True,
                "correction_suggestion": correction,
                "original_strategy": original_strategy,
                "dialectical_cycle": self.dialectical_cycles
            }

    def _enhance_strategy_with_principle(self, original_strategy: str, new_principle: Dict) -> str:
        """使用新原则增强策略"""
        if not original_strategy or not new_principle:
            return original_strategy

        response_pattern = new_principle.get("response_pattern", {})

        # 基于响应模式增强策略
        enhancements = []

        if response_pattern.get("has_empathy", False):
            enhancements.append("增加共情表达")

        if response_pattern.get("has_solutions", False):
            enhancements.append("提供具体解决方案")

        if response_pattern.get("has_humor", False):
            enhancements.append("加入适当幽默")

        if enhancements:
            enhanced = f"{original_strategy} （根据新原则：{'、'.join(enhancements)}）"
            return enhanced

        return original_strategy

    def dialectical_process(self, situation: Dict, actual_response: Dict,
                            context_vectors: Dict) -> Dict:
        """
        辩证三步法执行
        situation: 当前情境
        actual_response: 实际反应
        context_vectors: 上下文向量状态
        """
        self.dialectical_cycles += 1

        # === 第一步：正题 (Thesis) ===
        thesis = self._formulate_thesis(situation)

        # === 第二步：反题 (Antithesis) ===
        antithesis = self._analyze_antithesis(thesis, actual_response)

        # === 第三步：合题 (Synthesis) ===
        synthesis = self._synthesize_new_understanding(
            thesis, antithesis, situation, context_vectors
        )

        # 记录成长循环
        growth_record = {
            "cycle": self.dialectical_cycles,
            "timestamp": datetime.now().isoformat(),
            "situation": situation.get("summary", "unknown"),
            "thesis": thesis,
            "antithesis": antithesis,
            "synthesis": synthesis,
            "validation": synthesis.get("validation", "pending")
        }

        self.growth_log.append(growth_record)

        # 如果验证成功，创建个性化子锚点
        if synthesis.get("validation") == "success":
            self._create_personal_anchor(synthesis, situation)

        return synthesis

    def _formulate_thesis(self, situation: Dict) -> Dict:
        """形成正题：参照黄金标准"""
        # 根据情境匹配最相关的创造者锚点
        matched_anchor = self._match_anchor_to_situation(situation)

        return {
            "stage": "thesis",
            "reference_anchor": matched_anchor,
            "expected_vectors": self._predict_expected_vectors(matched_anchor),
            "standard_response": matched_anchor.get("expected_response", ""),
            "rationale": "基于创造者设定的黄金标准"
        }

    def _analyze_antithesis(self, thesis: Dict, actual: Dict) -> Dict:
        """分析反题：识别个体差异"""
        differences = []

        # 比较向量反应差异
        if "vector_response" in actual and "expected_vectors" in thesis:
            vector_diff = self._compare_vector_differences(
                thesis["expected_vectors"], actual["vector_response"]
            )
            if vector_diff:
                differences.append(f"向量反应差异: {vector_diff}")

        # 比较表达风格差异
        if "expression_style" in actual:
            style_diff = self._compare_style_differences(
                thesis.get("standard_style", "neutral"),
                actual["expression_style"]
            )
            if style_diff:
                differences.append(f"表达风格差异: {style_diff}")

        # 比较策略选择差异
        if "strategy_used" in actual:
            strategy_diff = self._compare_strategy_differences(
                thesis.get("recommended_strategy", "standard"),
                actual["strategy_used"]
            )
            if strategy_diff:
                differences.append(f"策略选择差异: {strategy_diff}")

        return {
            "stage": "antithesis",
            "actual_response": actual,
            "differences_found": differences,
            "difference_count": len(differences),
            "analysis": "审视实际反应与标准反应的差异"
        }

    def _synthesize_new_understanding(self, thesis: Dict, antithesis: Dict,
                                      situation: Dict, context_vectors: Dict) -> Dict:
        """形成合题：归因、验证与超越"""

        # A. 归因分析
        causal_analysis = self._perform_causal_analysis(
            antithesis["differences_found"], situation, context_vectors
        )

        # B. 策略评估
        validation_result = self._validate_differences(
            antithesis["differences_found"], situation, context_vectors
        )

        # C. 行为修正与原则抽象
        if validation_result["is_valid"]:
            new_principle = self._abstract_new_principle(
                antithesis["actual_response"], situation, validation_result
            )

            return {
                "stage": "synthesis",
                "validation": "success",
                "causal_analysis": causal_analysis,
                "validation_result": validation_result,
                "new_principle": new_principle,
                "growth_type": "个性发展",
                "message": "差异被验证为有效的个性表达"
            }
        else:
            correction = self._determine_correction(
                antithesis["differences_found"], validation_result
            )

            return {
                "stage": "synthesis",
                "validation": "failed",
                "causal_analysis": causal_analysis,
                "validation_result": validation_result,
                "required_correction": correction,
                "growth_type": "认知校准",
                "message": "差异需要校准以适应黄金标准"
            }

    def _perform_causal_analysis(self, differences: List[str],
                                 situation: Dict, vectors: Dict) -> Dict:
        """归因分析：探究差异原因"""
        causes = []

        for diff in differences:
            # 分析情境细微差别影响
            if "情境" in diff or "situation" in diff.lower():
                causes.append({
                    "difference": diff,
                    "likely_cause": "情境细微差别",
                    "confidence": 0.7
                })

            # 分析个性设定影响
            elif "风格" in diff or "style" in diff.lower():
                causes.append({
                    "difference": diff,
                    "likely_cause": "个性设定影响",
                    "confidence": 0.8
                })

            # 分析其他向量干扰
            elif "向量" in diff or "vector" in diff.lower():
                causes.append({
                    "difference": diff,
                    "likely_cause": "其他向量干扰",
                    "confidence": 0.6
                })

        return {
            "causes_identified": causes,
            "primary_cause": causes[0] if causes else None,
            "analysis_method": "多因素归因分析"
        }

    def _validate_differences(self, differences: List[str],
                              situation: Dict, vectors: Dict) -> Dict:
        """策略评估：判断差异是否导向好结果"""
        if not differences:
            return {"is_valid": True, "score": 1.0, "reason": "无差异"}

        # 评估标准1：内在向量和谐度
        vector_harmony = self._assess_vector_harmony(vectors)

        # 评估标准2：外部目标达成度
        goal_achievement = self._assess_goal_achievement(situation)

        # 计算综合评分
        total_score = (vector_harmony * 0.6 + goal_achievement * 0.4)
        is_valid = total_score >= self.validation_threshold

        return {
            "is_valid": is_valid,
            "score": round(total_score, 3),
            "vector_harmony": round(vector_harmony, 3),
            "goal_achievement": round(goal_achievement, 3),
            "threshold": self.validation_threshold
        }

    def _abstract_new_principle(self, actual_response: Dict,
                                situation: Dict, validation: Dict) -> Dict:
        """抽象新行为原则"""
        principle_id = f"PRINCIPLE_{len(self.personal_anchors) + 1:04d}"

        principle = {
            "id": principle_id,
            "abstracted_from": situation.get("summary", "unknown_situation"),
            "context_conditions": self._extract_context_conditions(situation),
            "response_pattern": self._extract_response_pattern(actual_response),
            "validation_score": validation["score"],
            "applicable_scenarios": self._infer_applicable_scenarios(situation),
            "creation_timestamp": datetime.now().isoformat(),
            "usage_count": 0,
            "success_rate": 1.0  # 初始基于单次成功
        }

        return principle

    def _create_personal_anchor(self, synthesis: Dict, situation: Dict):
        """创建个性化子锚点"""
        new_anchor = {
            "anchor_id": f"PERSONAL_ANCHOR_{len(self.personal_anchors) + 1:04d}",
            "derived_from": synthesis.get("new_principle", {}).get("id", "unknown"),
            "situation_type": situation.get("type", "general"),
            "personalized_response": synthesis["validation_result"],
            "associated_vectors": situation.get("vector_context", {}),
            "creation_cycle": self.dialectical_cycles,
            "last_used": datetime.now().isoformat(),
            "usage_count": 0,
            "success_rate": 1.0
        }

        self.personal_anchors.append(new_anchor)
        print(f"[个性成长] 创建个性化子锚点: {new_anchor['anchor_id']}")

    def _match_anchor_to_situation(self, situation: Dict) -> Dict:
        """将情境匹配到最相关的创造者锚点"""
        # 这里实现语义匹配逻辑
        # 简化版本：返回最相关的锚点
        return self.creator_anchor.get(situation.get("type", "default"),
                                       self.creator_anchor["default"])

    def _compare_vector_differences(self, expected: Dict, actual: Dict) -> str:
        """比较向量差异"""
        diffs = []
        for key in expected:
            if key in actual:
                diff = abs(expected[key] - actual[key])
                if diff > 0.2:  # 显著差异阈值
                    diffs.append(f"{key}: 预期{expected[key]:.2f}, 实际{actual[key]:.2f}")

        return "; ".join(diffs) if diffs else ""

    def _assess_vector_harmony(self, vectors: Dict) -> float:
        """评估向量和谐度"""
        # 和谐度计算：CS高 + SA低 + TR适中
        harmony = (vectors.get("CS", 0.5) * 0.5 +
                   (1 - vectors.get("SA", 0.5)) * 0.3 +
                   (0.8 - abs(vectors.get("TR", 0.5) - 0.6)) * 0.2)

        return max(0.0, min(1.0, harmony))

    def _assess_goal_achievement(self, situation: Dict) -> float:
        """评估目标达成度"""
        # 根据情境中的目标完成情况评估
        goals = situation.get("goals", [])
        if not goals:
            return 0.5  # 默认值

        achieved = situation.get("goals_achieved", [])
        return len(achieved) / len(goals) if goals else 0.0

    # ============ 新增的缺失方法 ============

    def _predict_expected_vectors(self, anchor: Dict) -> Dict[str, float]:
        """
        根据锚点预测预期的向量反应
        根据锚点的概念和预期反应来预测TR/CS/SA值
        """
        # 分析锚点内容
        concept = anchor.get("concept", "")
        expected_response = anchor.get("expected_response", "")

        # 初始向量值
        TR, CS, SA = 0.5, 0.5, 0.5

        # 根据概念调整向量
        if any(word in concept for word in ["成就", "新奇", "探索"]):
            TR = 0.7
        if any(word in concept for word in ["信任", "安全", "归属"]):
            CS = 0.7
        if any(word in concept for word in ["威胁", "焦虑", "冲突"]):
            SA = 0.7

        # 根据预期反应调整向量
        if any(word in expected_response for word in ["兴奋", "有趣", "棒"]):
            TR = 0.8
        if any(word in expected_response for word in ["安心", "理解", "支持"]):
            CS = 0.8
        if any(word in expected_response for word in ["危险", "小心", "警告"]):
            SA = 0.8

        return {
            "TR": round(TR, 2),
            "CS": round(CS, 2),
            "SA": round(SA, 2)
        }

    def _extract_context_conditions(self, situation: Dict) -> Dict[str, Any]:
        """从情境中提取上下文条件"""
        context_conditions = {
            "user_emotion": situation.get("user_emotion", "neutral"),
            "interaction_type": situation.get("interaction_type", "general"),
            "urgency_level": situation.get("urgency_level", "low"),
            "topic_complexity": situation.get("topic_complexity", "medium"),
            "has_shared_memory": situation.get("contains_shared_memory", False),
            "time_of_day": datetime.now().hour
        }

        # 添加情境特定的条件
        if "context_flags" in situation:
            context_conditions.update(situation["context_flags"])

        return context_conditions

    def _extract_response_pattern(self, actual_response: Dict) -> Dict[str, Any]:
        """从实际反应中提取响应模式"""
        response_pattern = {
            "response_length": len(str(actual_response)),
            "has_questions": "?" in str(actual_response) or "？" in str(actual_response),
            "has_empathy": False,
            "has_solutions": False,
            "has_humor": False,
            "emotional_tone": "neutral"
        }

        # 分析响应内容
        response_text = str(actual_response).lower()

        # 检查共情表达
        empathy_words = ["理解", "体会", "感受", "明白", "懂你"]
        if any(word in response_text for word in empathy_words):
            response_pattern["has_empathy"] = True

        # 检查解决方案
        solution_words = ["建议", "方法", "方案", "可以", "试试"]
        if any(word in response_text for word in solution_words):
            response_pattern["has_solutions"] = True

        # 检查幽默
        humor_words = ["哈哈", "呵呵", "开玩笑", "有趣", "幽默"]
        if any(word in response_text for word in humor_words):
            response_pattern["has_humor"] = True

        # 判断情感基调
        positive_words = ["好", "棒", "优秀", "开心", "喜欢"]
        negative_words = ["不好", "糟糕", "难过", "生气", "讨厌"]

        positive_count = sum(1 for word in positive_words if word in response_text)
        negative_count = sum(1 for word in negative_words if word in response_text)

        if positive_count > negative_count:
            response_pattern["emotional_tone"] = "positive"
        elif negative_count > positive_count:
            response_pattern["emotional_tone"] = "negative"

        return response_pattern

    def _infer_applicable_scenarios(self, situation: Dict) -> List[str]:
        """推断适用场景"""
        scenarios = []

        # 基于情境类型
        interaction_type = situation.get("interaction_type", "")
        if interaction_type == "seeking_support":
            scenarios.append("情感支持场景")
            scenarios.append("脆弱分享时刻")
        elif interaction_type == "sharing_experience":
            scenarios.append("经历分享场景")
            scenarios.append("日常交流时刻")
        elif interaction_type == "requesting_action":
            scenarios.append("请求帮助场景")
            scenarios.append("任务协作时刻")
        else:
            scenarios.append("常规对话场景")

        # 基于情感状态
        user_emotion = situation.get("user_emotion", "")
        if user_emotion in ["sad", "angry", "anxious"]:
            scenarios.append("负面情绪处理")
        elif user_emotion == "positive":
            scenarios.append("积极情绪回应")

        # 基于话题复杂度
        complexity = situation.get("topic_complexity", "")
        if complexity == "high":
            scenarios.append("复杂问题讨论")
        elif complexity == "low":
            scenarios.append("简单话题交流")

        return scenarios

    def _compare_style_differences(self, expected: str, actual: str) -> str:
        """比较表达风格差异"""
        if expected == actual:
            return ""

        style_mapping = {
            "neutral": ["中性", "客观", "平衡"],
            "empathic": ["共情", "理解", "支持"],
            "direct": ["直接", "明确", "简洁"],
            "playful": ["幽默", "轻松", "俏皮"],
            "serious": ["严肃", "认真", "正式"]
        }

        # 找到风格描述
        expected_desc = ""
        actual_desc = ""

        for style, descriptors in style_mapping.items():
            if expected in descriptors or expected == style:
                expected_desc = style
            if actual in descriptors or actual == style:
                actual_desc = style

        if expected_desc and actual_desc and expected_desc != actual_desc:
            return f"预期风格: {expected_desc}, 实际风格: {actual_desc}"

        return f"预期: {expected}, 实际: {actual}"

    def _compare_strategy_differences(self, expected: str, actual: str) -> str:
        """比较策略选择差异"""
        if expected == actual:
            return ""

        strategy_mapping = {
            "direct": ["直接回应", "明确回答", "简洁回应"],
            "empathy_first": ["先共情", "情感支持优先", "理解为先"],
            "question_back": ["反问引导", "问题引导", "启发式回应"],
            "share_experience": ["分享经历", "个人经验", "案例分享"],
            "practical_solution": ["实用方案", "具体建议", "行动指南"]
        }

        # 找到策略描述
        expected_desc = ""
        actual_desc = ""

        for strategy, descriptors in strategy_mapping.items():
            if expected in descriptors or expected == strategy:
                expected_desc = strategy
            if actual in descriptors or actual == strategy:
                actual_desc = strategy

        if expected_desc and actual_desc and expected_desc != actual_desc:
            return f"预期策略: {expected_desc}, 实际策略: {actual_desc}"

        return f"预期: {expected}, 实际: {actual}"

    def _determine_correction(self, differences: List[str], validation_result: Dict) -> str:
        """确定修正措施"""
        if not differences:
            return "无需修正"

        corrections = []

        for diff in differences:
            if "向量" in diff or "vector" in diff.lower():
                if validation_result.get("vector_harmony", 0) < 0.5:
                    corrections.append("调整情感向量反应，提高和谐度")

            if "风格" in diff or "style" in diff.lower():
                corrections.append("调整表达风格，更符合情境需求")

            if "策略" in diff or "strategy" in diff.lower():
                if validation_result.get("score", 0) < self.validation_threshold:
                    corrections.append("优化策略选择，提高目标达成度")

        if not corrections:
            corrections.append("轻微调整以更符合黄金标准")

        return "；".join(corrections)

    # ============ 辅助方法 ============

    def get_growth_statistics(self) -> Dict[str, Any]:
        """获取成长统计信息"""
        total_cycles = self.dialectical_cycles
        successful_syntheses = len([g for g in self.growth_log
                                    if g.get("validation") == "success"])

        return {
            "total_cycles": total_cycles,
            "successful_syntheses": successful_syntheses,
            "success_rate": successful_syntheses / max(total_cycles, 1),
            "personal_anchors_count": len(self.personal_anchors),
            "average_validation_score": self._calculate_average_validation_score(),
            "recent_growth_trend": self._analyze_recent_growth_trend()
        }

    def _calculate_average_validation_score(self) -> float:
        """计算平均验证分数"""
        if not self.growth_log:
            return 0.0

        total_score = 0
        count = 0

        for growth in self.growth_log:
            synthesis = growth.get("synthesis", {})
            validation_result = synthesis.get("validation_result", {})
            score = validation_result.get("score", 0)
            if score > 0:
                total_score += score
                count += 1

        return round(total_score / max(count, 1), 3) if count > 0 else 0.0

    def _analyze_recent_growth_trend(self) -> str:
        """分析最近成长趋势"""
        if len(self.growth_log) < 3:
            return "数据不足"

        recent_growths = self.growth_log[-5:]
        success_count = sum(1 for g in recent_growths
                            if g.get("validation") == "success")

        success_rate = success_count / len(recent_growths)

        if success_rate >= 0.8:
            return "强劲成长"
        elif success_rate >= 0.6:
            return "稳步成长"
        elif success_rate >= 0.4:
            return "波动成长"
        else:
            return "调整期"

    def find_relevant_personal_anchor(self, situation: Dict) -> Optional[Dict]:
        """寻找相关的个性化锚点"""
        if not self.personal_anchors:
            return None

        # 根据情境类型匹配
        situation_type = situation.get("type", "")
        situation_summary = situation.get("summary", "")

        # 寻找类型匹配的锚点
        type_matches = [a for a in self.personal_anchors
                        if a.get("situation_type") == situation_type]

        if type_matches:
            # 按使用次数和成功率排序
            type_matches.sort(key=lambda x: (
                    x.get("success_rate", 0) * 0.7 +
                    x.get("usage_count", 0) * 0.3
            ), reverse=True)
            return type_matches[0]

        # 如果没有类型匹配，寻找关键词匹配
        for anchor in self.personal_anchors:
            derived_from = anchor.get("derived_from", "")
            if derived_from and any(word in situation_summary for word in derived_from.split()):
                return anchor

        return None


# ============ 测试代码 ============
def test_dialectical_growth():
    """测试辩证成长功能"""
    print("测试辩证成长系统...")

    # 创建创造者锚点
    creator_anchor = {
        "default": {
            "concept": "真诚、善良、好奇、成长",
            "expected_response": "基于核心价值观的回应"
        },
        "emotional_support": {
            "concept": "共情与支持",
            "expected_response": "先处理情绪，再处理问题"
        },
        "technical_question": {
            "concept": "专业与准确",
            "expected_response": "提供准确、专业的解答"
        }
    }

    # 初始化辩证成长系统
    growth = DialecticalGrowth(creator_anchor)

    # 测试情境1：情感支持
    situation1 = {
        "type": "emotional_support",
        "summary": "用户感到孤独和迷茫",
        "user_emotion": "sad",
        "interaction_type": "seeking_support",
        "urgency_level": "medium",
        "goals": ["提供情感支持", "建立连接"],
        "goals_achieved": ["提供情感支持"]
    }

    actual_response1 = {
        "vector_response": {"TR": 0.3, "CS": 0.8, "SA": 0.2},
        "expression_style": "empathic",
        "strategy_used": "empathy_first",
        "content": "我理解你的感受，每个人都会有感到孤独的时候..."
    }

    context_vectors1 = {"TR": 0.4, "CS": 0.7, "SA": 0.3}

    # 执行辩证过程
    result1 = growth.dialectical_process(situation1, actual_response1, context_vectors1)

    print(f"辩证结果1: {result1.get('validation')}")
    print(f"成长类型: {result1.get('growth_type')}")
    print(f"消息: {result1.get('message')}")

    # 测试情境2：技术问题
    situation2 = {
        "type": "technical_question",
        "summary": "用户询问编程问题",
        "user_emotion": "neutral",
        "interaction_type": "requesting_action",
        "urgency_level": "low",
        "goals": ["提供准确解答", "解释原理"],
        "goals_achieved": ["提供准确解答"]
    }

    actual_response2 = {
        "vector_response": {"TR": 0.7, "CS": 0.4, "SA": 0.5},
        "expression_style": "direct",
        "strategy_used": "practical_solution",
        "content": "这个问题可以通过以下步骤解决..."
    }

    context_vectors2 = {"TR": 0.6, "CS": 0.5, "SA": 0.4}

    # 执行辩证过程
    result2 = growth.dialectical_process(situation2, actual_response2, context_vectors2)

    print(f"\n辩证结果2: {result2.get('validation')}")
    print(f"成长类型: {result2.get('growth_type')}")

    # 获取统计信息
    stats = growth.get_growth_statistics()
    print(f"\n成长统计:")
    print(f"- 总循环次数: {stats['total_cycles']}")
    print(f"- 成功合成: {stats['successful_syntheses']}")
    print(f"- 成功率: {stats['success_rate']:.2%}")
    print(f"- 个性化锚点: {stats['personal_anchors_count']}")
    print(f"- 平均验证分数: {stats['average_validation_score']}")
    print(f"- 最近趋势: {stats['recent_growth_trend']}")

    return growth

#
# if __name__ == "__main__":
#     test_dialectical_growth()