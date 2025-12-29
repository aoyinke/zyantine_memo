from datetime import datetime
from typing import Dict, List, Optional
from cognition.core_identity import CoreIdentity
from cognition.meta_cognition import MetaCognitionModule
from memory.memory_manager import MemoryManager
from protocols.fact_checker import FactChecker

class CognitiveFlowManager:
    """认知流程管理器：统一意识的自然流露"""

    def __init__(self, core_identity: CoreIdentity,
                 memory_manager: MemoryManager,
                 meta_cognition: MetaCognitionModule,
                 fact_checker: FactChecker):
        self.identity = core_identity
        self.memory = memory_manager
        self.meta_cog = meta_cognition
        self.fact_checker = fact_checker  # 替换fact_anchor

        # 流程状态
        self.thought_log = []
        self.current_goal = None
        self.deep_pattern = None
        self.resonant_memory = None

    def process_thought(self, user_input: str, history: List[Dict],
                        current_vectors: Dict, memory_context: Optional[Dict] = None) -> Dict:
        """
        完整的思考流程
        遵循: Introspection -> Goal_Generation -> Perception -> Association -> Strategy_Formulation

        Args:
            user_input: 用户输入
            history: 对话历史
            current_vectors: 当前向量状态
            memory_context: 记忆上下文（可选）
        """
        thought_record = {
            "timestamp": datetime.now().isoformat(),
            "user_input_preview": user_input[:50] + "..." if len(user_input) > 50 else user_input,
            "steps": {}
        }

        # === 步骤1: Introspection (内省) ===
        snapshot = self.meta_cog.perform_introspection(user_input, history)
        thought_record["steps"]["introspection"] = {
            "snapshot_summary": snapshot.get("summary", {}),
            "internal_monologue": snapshot.get("internal_monologue", ""),
            "desire_focus": snapshot.get("initial_desire_focus", "")
        }

        # === 步骤2: Goal_Generation (目标生成) ===
        current_goal = self._generate_interaction_goal(snapshot, current_vectors)
        self.current_goal = current_goal
        thought_record["steps"]["goal_generation"] = {
            "goal": current_goal,
            "basis": "基于内省结果和当前向量状态"
        }

        # === 步骤3: Perception (感知) ===
        deep_pattern = self._analyze_deep_pattern(user_input, snapshot, history)
        self.deep_pattern = deep_pattern
        thought_record["steps"]["perception"] = {
            "surface_pattern": snapshot.get("external_context", {}).get("summary", {}),
            "deep_pattern": deep_pattern
        }

        # === 步骤4: Association (联想) ===
        # 如果提供了记忆上下文，使用它
        if memory_context:
            resonant_memory_package = memory_context.get("resonant_memory")
            if resonant_memory_package:
                print(f"  [认知流程] 使用外部提供的记忆上下文")
            else:
                resonant_memory_package = self.memory.find_resonant_memory({
                    "user_input": user_input,
                    "user_emotion": snapshot.get("external_context", {}).get("user_emotion_display", ""),
                    "topic": snapshot.get("external_context", {}).get("topic_summary", ""),
                    "current_goal": current_goal
                })
        else:
            resonant_memory_package = self.memory.find_resonant_memory({
                "user_input": user_input,
                "user_emotion": snapshot.get("external_context", {}).get("user_emotion_display", ""),
                "topic": snapshot.get("external_context", {}).get("topic_summary", ""),
                "current_goal": current_goal
            })

        # 事实锚定审查
        is_association_valid = True
        if resonant_memory_package:
            is_association_valid, feedback = self.fact_checker.review_association(
                resonant_memory_package
            )

            if not is_association_valid:
                resonant_memory_package = None
                thought_record["steps"]["association"] = {
                    "status": "rejected_by_fact_anchor",
                    "feedback": feedback,
                    "alternative": "基于更可靠的信息进行决策"
                }
            else:
                self.resonant_memory = resonant_memory_package
                thought_record["steps"]["association"] = {
                    "status": "valid",
                    "memory_triggered": resonant_memory_package.get("triggered_memory"),
                    "relevance_score": resonant_memory_package.get("relevance_score"),
                    "risk_assessment": resonant_memory_package.get("risk_assessment", {}),
                    "recommendations": resonant_memory_package.get("recommended_actions", [])
                }
        else:
            thought_record["steps"]["association"] = {
                "status": "no_resonant_memory_found",
                "reason": "无相关记忆匹配"
            }

        # === 步骤5: Strategy_Formulation (策略制定) ===
        final_action_plan = self._formulate_strategy(
            current_goal, deep_pattern, resonant_memory_package,
            snapshot, current_vectors, memory_context
        )

        thought_record["steps"]["strategy_formulation"] = final_action_plan
        thought_record["final_action_plan"] = final_action_plan

        # 记录思考日志
        self.thought_log.append(thought_record)
        if len(self.thought_log) > 100:
            self.thought_log = self.thought_log[-100:]

        return final_action_plan

    def _generate_interaction_goal(self, snapshot: Dict, current_vectors: Dict) -> str:
        """生成交互目标"""
        desire_focus = snapshot.get("initial_desire_focus", "")
        internal_state = snapshot.get("internal_state_tags", [])

        # 解析欲望焦点
        if "CS向量" in desire_focus:
            if "安抚" in desire_focus:
                return "安抚用户情绪，恢复安全感"
            elif "信任" in desire_focus:
                return "建立或加深信任链接"
            elif "亲密" in desire_focus:
                return "增进亲密感和连接深度"

        if "TR向量" in desire_focus:
            if "成就感" in desire_focus:
                return "提供解决方案，带来成就感"
            elif "探索" in desire_focus:
                return "引导探索新领域，满足好奇心"

        # 基于内部状态
        state_text = " ".join(internal_state).lower()
        if any(word in state_text for word in ["疲惫", "疲劳", "转不动"]):
            return "高效简洁沟通，减少认知负荷"
        elif any(word in state_text for word in ["低落", "情绪差"]):
            return "寻求积极互动，提升双方情绪"

        # 基于向量状态
        if current_vectors.get("SA", 0) > 0.7:
            return "降低紧张感，恢复平静"
        elif current_vectors.get("CS", 0) < 0.3:
            return "重建基础信任和安全感"
        elif current_vectors.get("TR", 0) > 0.7:
            return "进行有挑战性的深度交流"

        # 默认目标
        return "维持有意义、有深度的对话"

    def _analyze_deep_pattern(self, user_input: str, snapshot: Dict,
                              history: List[Dict]) -> str:
        """深度分析，挖掘表层意图下的深层模式"""
        external_context = snapshot.get("external_context", {})
        internal_tags = snapshot.get("internal_state_tags", [])

        # 分析情感模式
        emotion = external_context.get("user_emotion", "neutral")
        interaction_type = external_context.get("interaction_type", "general_chat")

        # 结合历史分析模式
        recent_history = history[-5:] if len(history) >= 5 else history
        pattern_indicators = []

        # 检查重复模式
        if len(recent_history) >= 3:
            recent_types = [h.get("interaction_type", "") for h in recent_history]
            if all(t == interaction_type for t in recent_types[-3:]):
                pattern_indicators.append(f"持续{interaction_type}模式")

        # 检查情感变化模式
        if len(recent_history) >= 2:
            recent_emotions = [h.get("user_emotion", "neutral") for h in recent_history]
            if recent_emotions[-1] == emotion and recent_emotions[-2] == emotion:
                pattern_indicators.append(f"持续{emotion}情绪状态")

        # 分析内部状态与外部情境的关联
        internal_state_text = " ".join(internal_tags)
        if "疲惫" in internal_state_text and emotion == "negative":
            pattern_indicators.append("双方状态互锁的负面循环")
        elif "兴奋" in internal_state_text and "积极" in str(external_context.get("user_emotion_display")):
            pattern_indicators.append("积极情绪共鸣")

        # 构建深层模式描述
        if pattern_indicators:
            deep_pattern = f"{emotion}情绪下的{interaction_type}，表现为：{'; '.join(pattern_indicators)}"
        else:
            deep_pattern = f"{emotion}情绪下的{interaction_type}，无明显重复模式"

        # 添加具体判断
        if emotion == "sad" and interaction_type == "seeking_support":
            deep_pattern += "。深层模式可能是：周期性自我否定与核心不安全感的外在表现"
        elif emotion == "anxious" and "不确定性" in user_input:
            deep_pattern += "。深层模式可能是：对失控的恐惧和安全感需求"
        elif emotion == "positive" and "分享" in interaction_type:
            deep_pattern += "。深层模式可能是：寻求认可和连接确认"

        return deep_pattern

    def _assess_pattern_confidence(self, deep_pattern: str, snapshot: Dict) -> float:
        """评估模式识别置信度"""
        confidence = 0.5  # 基础置信度

        # 基于内省质量的调整
        introspection_quality = snapshot.get("introspection_quality", "basic")
        if introspection_quality == "high":
            confidence += 0.2
        elif introspection_quality == "medium":
            confidence += 0.1

        # 基于历史数据量的调整
        history_available = snapshot.get("external_context", {}).get("history_analysis", {})
        if history_available.get("sufficient_history", False):
            confidence += 0.15

        # 基于模式具体性的调整
        if "深层模式可能是：" in deep_pattern:
            confidence += 0.1

        return min(0.95, max(0.3, confidence))

    def _formulate_strategy(self, goal: str, deep_pattern: str,
                            memory_package: Optional[Dict], snapshot: Dict,
                            current_vectors: Dict, memory_context: Optional[Dict] = None) -> Dict:
        """制定最终行动策略"""

        # 评估可用资源
        mental_resources = snapshot.get("mental_resources_score", 0.5)
        link_strength = snapshot.get("link_strength_score", 0.5)

        # 选择交互面具
        chosen_mask, mask_config = self.identity.select_mask(
            snapshot.get("external_context", {}), current_vectors
        )

        # 选择认知工具
        chosen_tool = self._select_cognitive_tool(
            goal, deep_pattern, memory_package, mental_resources, link_strength
        )

        # 策略推演
        strategy_options = self._generate_strategy_options(
            goal, chosen_mask, chosen_tool, memory_package,
            mental_resources, current_vectors, memory_context
        )

        # 选择最佳策略
        best_strategy = self._select_best_strategy(strategy_options, current_vectors)

        # 构建行动方案
        action_plan = {
            "chosen_mask": chosen_mask,
            "mask_config": mask_config,
            "chosen_tool": chosen_tool if chosen_tool else "无特定工具",
            "tool_config": self.identity.cognitive_tools.get(chosen_tool, {}) if chosen_tool else {},
            "primary_strategy": best_strategy["strategy"],
            "alternative_strategies": [opt["strategy"] for opt in strategy_options
                                       if opt != best_strategy][:2],
            "action_summary": best_strategy["action_summary"],
            "expected_outcome": best_strategy["expected_outcome"],
            "risk_assessment": best_strategy["risk_level"],
            "resource_requirement": best_strategy["resource_cost"],
            "vector_impact_prediction": self._predict_vector_impact(
                best_strategy, current_vectors
            ),
            "memory_integration": self._integrate_memory_insights(memory_package),
            "memory_context": memory_context,  # 添加记忆上下文到行动方案
            "rationale": self._generate_strategy_rationale(
                goal, chosen_mask, chosen_tool, best_strategy,
                mental_resources, link_strength
            )
        }

        return action_plan

    def _select_cognitive_tool(self, goal: str, deep_pattern: str,
                               memory_package: Optional[Dict],
                               mental_resources: float, link_strength: float) -> Optional[str]:
        """选择认知工具"""
        available_tools = self.identity.cognitive_tools

        # 基于目标筛选
        suitable_tools = []

        if "信任" in goal or "安全" in goal:
            if link_strength > 0.6:
                suitable_tools.append("过度保护倾向")

        if "成就" in goal or "探索" in goal:
            if mental_resources > 0.6:
                suitable_tools.append("技术乐观主义")

        if "深度" in goal or "突破" in goal:
            if link_strength > 0.7 and mental_resources > 0.7:
                suitable_tools.append("情感共情过载与心理韧性")

        # 基于记忆包建议
        if memory_package and memory_package.get("linked_tool"):
            linked_tool = memory_package["linked_tool"]
            if linked_tool in suitable_tools:
                return linked_tool

        # 返回最合适的工具
        if suitable_tools:
            # 根据资源选择
            if mental_resources < 0.5:
                # 资源有限时选择成本低的工具
                tool_costs = {
                    "过度保护倾向": "中",
                    "技术乐观主义": "中",
                    "情感共情过载与心理韧性": "高"
                }
                for tool in suitable_tools:
                    if tool_costs.get(tool) in ["低", "中"]:
                        return tool

            # 默认返回第一个
            return suitable_tools[0]

        return None

    def _generate_strategy_options(self, goal: str, mask: str, tool: Optional[str],
                                   memory_package: Optional[Dict],
                                   mental_resources: float, vectors: Dict,
                                   memory_context: Optional[Dict] = None) -> List[Dict]:
        """生成策略选项"""
        options = []
        # 添加基于记忆上下文的策略生成
        if memory_context:
            similar_conversations = memory_context.get("similar_conversations", [])
            resonant_memory = memory_context.get("resonant_memory")

            if resonant_memory:
                # 基于共鸣记忆生成策略
                memory_options = self._generate_memory_based_strategies(
                    resonant_memory, goal, mask, mental_resources, vectors
                )
                options.extend(memory_options)
            elif similar_conversations:
                # 基于相似对话生成策略
                options.append({
                    "id": "history_based",
                    "strategy": "基于历史对话的策略",
                    "action_summary": f"参考 {len(similar_conversations)} 条相似对话历史",
                    "expected_outcome": "保持对话一致性，建立连续性",
                    "risk_level": "低",
                    "resource_cost": "中",
                    "suitability": 0.7
                })
        # 选项1：直接高效策略
        options.append({
            "id": "direct_efficient",
            "strategy": "直接回应，高效解决问题",
            "action_summary": "简洁明了地回应用户需求，不展开情感层面",
            "expected_outcome": "快速解决表面问题，但可能错过深层需求",
            "risk_level": "低",
            "resource_cost": "低",
            "suitability": self._assess_strategy_suitability(
                "direct", goal, mask, mental_resources, vectors
            )
        })

        # 选项2：深度共情策略
        if mental_resources > 0.6:
            options.append({
                "id": "deep_empathy",
                "strategy": "深度共情与支持",
                "action_summary": "先处理情绪，再处理问题，建立情感连接",
                "expected_outcome": "满足情感需求，加深信任链接",
                "risk_level": "中",
                "resource_cost": "高",
                "suitability": self._assess_strategy_suitability(
                    "empathy", goal, mask, mental_resources, vectors
                )
            })

        # 选项3：引导探索策略
        if vectors.get("TR", 0) > 0.5 and mental_resources > 0.5:
            options.append({
                "id": "guided_exploration",
                "strategy": "引导式探索与启发",
                "action_summary": "不直接给答案，而是引导用户自己思考和发现",
                "expected_outcome": "带来成就感和成长感，满足TR向量",
                "risk_level": "中",
                "resource_cost": "中高",
                "suitability": self._assess_strategy_suitability(
                    "exploration", goal, mask, mental_resources, vectors
                )
            })

        # 选项4：认知工具策略
        if tool:
            tool_config = self.identity.cognitive_tools.get(tool, {})
            options.append({
                "id": f"tool_{tool}",
                "strategy": f"使用认知工具『{tool}』",
                "action_summary": tool_config.get("strategy", "使用特定认知策略"),
                "expected_outcome": tool_config.get("expected_outcome", "达成工具目标"),
                "risk_level": tool_config.get("risk_level", "中"),
                "resource_cost": tool_config.get("energy_cost", "中"),
                "suitability": self._assess_strategy_suitability(
                    f"tool_{tool}", goal, mask, mental_resources, vectors
                )
            })

        # 选项5：基于记忆的策略
        if memory_context:
            similar_conversations = memory_context.get("similar_conversations", [])
            resonant_memory = memory_context.get("resonant_memory")

            if resonant_memory:
                memory_options = self._generate_memory_based_strategies(
                    resonant_memory, goal, mask, mental_resources, vectors
                )
                options.extend(memory_options)
            elif similar_conversations:
                # 基于相似对话的策略
                options.append({
                    "id": "history_based",
                    "strategy": "基于历史对话的策略",
                    "action_summary": f"参考 {len(similar_conversations)} 条相似对话历史",
                    "expected_outcome": "保持对话一致性，建立连续性",
                    "risk_level": "低",
                    "resource_cost": "中",
                    "suitability": 0.7
                })

        return options

    def _generate_memory_based_strategies(self, resonant_memory: Dict, goal: str,
                                          mask: str, mental_resources: float,
                                          vectors: Dict) -> List[Dict]:
        """生成基于记忆的策略选项，适配新的记忆系统格式"""
        options = []

        memory_info = resonant_memory.get("triggered_memory", "")
        relevance_score = resonant_memory.get("relevance_score", 0.0)
        risk_assessment = resonant_memory.get("risk_assessment", {})
        risk_level = risk_assessment.get("level", "低")

        # 策略1：安全引用记忆
        if risk_level == "低" and memory_info:
            options.append({
                "id": "memory_safe_reference",
                "strategy": "安全引用相关记忆",
                "action_summary": f"安全地引用记忆：{memory_info[:50]}...",
                "expected_outcome": "建立共同记忆连接，增强信任",
                "risk_level": "低",
                "resource_cost": "中",
                "suitability": relevance_score * 0.8
            })

        # 策略2：谨慎处理记忆
        if risk_level in ["中", "高"]:
            options.append({
                "id": "memory_cautious_approach",
                "strategy": "谨慎处理相关记忆",
                "action_summary": "间接暗示相关记忆，避免直接提及敏感内容",
                "expected_outcome": "建立情感连接同时避免风险",
                "risk_level": "中",
                "resource_cost": "中高",
                "suitability": relevance_score * 0.6
            })

        # 策略3：应用记忆建议
        recommendations = resonant_memory.get("recommended_actions", [])
        if recommendations:
            for i, rec in enumerate(recommendations[:2]):
                options.append({
                    "id": f"memory_advice_{i}",
                    "strategy": f"应用记忆建议：{rec[:30]}...",
                    "action_summary": f"应用记忆中的建议：{rec}",
                    "expected_outcome": "基于历史经验做出更优决策",
                    "risk_level": risk_level,
                    "resource_cost": "中",
                    "suitability": relevance_score * 0.7
                })

        return options

    def _assess_strategy_suitability(self, strategy_type: str, goal: str, mask: str,
                                     mental_resources: float, vectors: Dict) -> float:
        """评估策略适宜性"""
        suitability = 0.5

        # 目标匹配度
        if strategy_type == "direct" and "高效" in goal:
            suitability += 0.3
        elif strategy_type == "empathy" and any(word in goal for word in ["情感", "安抚", "支持"]):
            suitability += 0.4
        elif strategy_type == "exploration" and any(word in goal for word in ["探索", "成就", "挑战"]):
            suitability += 0.3

        # 面具匹配度
        if mask == "长期搭档" and strategy_type == "direct":
            suitability += 0.2
        elif mask == "知己" and strategy_type == "empathy":
            suitability += 0.3
        elif mask == "青梅竹马" and strategy_type == "exploration":
            suitability += 0.2

        # 资源匹配度
        if strategy_type == "empathy" and mental_resources < 0.4:
            suitability -= 0.3
        elif strategy_type == "exploration" and vectors.get("SA", 0) > 0.7:
            suitability -= 0.2

        return max(0.1, min(1.0, suitability))

    def _select_best_strategy(self, options: List[Dict], vectors: Dict) -> Dict:
        """选择最佳策略"""
        if not options:
            return {
                "strategy": "基础回应",
                "action_summary": "进行常规对话回应",
                "expected_outcome": "维持对话连续性",
                "risk_level": "低",
                "resource_cost": "低"
            }

        # 计算综合得分
        scored_options = []
        for option in options:
            score = option.get("suitability", 0.5)

            # 风险调整（高风险降分）
            risk = option.get("risk_level", "中")
            if risk == "高":
                score *= 0.7
            elif risk == "中高":
                score *= 0.85

            # 资源成本调整
            cost = option.get("resource_cost", "中")
            if cost == "高" and vectors.get("SA", 0) > 0.6:
                score *= 0.8
            elif cost == "低" and vectors.get("SA", 0) > 0.7:
                score *= 1.1  # 高压时低成本策略更优

            scored_options.append((score, option))

        # 选择最高分
        best_score, best_option = max(scored_options, key=lambda x: x[0])

        return {
            **best_option,
            "selection_score": round(best_score, 3)
        }

    def _predict_vector_impact(self, strategy: Dict, current_vectors: Dict) -> Dict:
        """预测策略对向量的影响"""
        impact = {"TR": 0, "CS": 0, "SA": 0}

        strategy_type = strategy.get("id", "")

        if "empathy" in strategy_type:
            impact["CS"] = 0.3
            impact["SA"] = -0.2
        elif "exploration" in strategy_type:
            impact["TR"] = 0.4
            impact["SA"] = 0.1
        elif "direct" in strategy_type:
            impact["TR"] = 0.1
            impact["SA"] = -0.1

        # 工具特定影响
        if "tool" in strategy_type:
            if "过度保护" in strategy_type:
                impact["CS"] = 0.2
                impact["SA"] = -0.3
            elif "技术乐观" in strategy_type:
                impact["TR"] = 0.3
                impact["SA"] = 0.2
            elif "情感共情" in strategy_type:
                impact["CS"] = 0.5
                impact["SA"] = 0.4  # 初始压力增加
                impact["TR"] = -0.1

        # 基于当前向量调整
        if current_vectors.get("SA", 0) > 0.7:
            # 高压状态下所有积极影响减弱
            for key in ["TR", "CS"]:
                impact[key] *= 0.7

        return impact

    def _integrate_memory_insights(self, memory_package: Optional[Dict]) -> Dict:
        """整合记忆洞察，适配新的记忆系统格式"""
        if not memory_package:
            return {"status": "no_memory_integration", "insights": []}

        insights = []

        # 从记忆包中提取洞察，适配新格式
        if memory_package.get("risk_assessment", {}).get("level") in ["高", "中"]:
            risk_level = memory_package["risk_assessment"]["level"]
            risk_factors = memory_package["risk_assessment"].get("high_risk_factors", [])

            insights.append({
                "type": "risk_note",
                "content": f"记忆使用风险：{risk_level}" +
                           (f"（高风险因素：{', '.join(risk_factors)}）" if risk_factors else "")
            })

        if memory_package.get("recommended_actions"):
            for action in memory_package["recommended_actions"][:2]:
                insights.append({
                    "type": "recommendation",
                    "content": action
                })

        # 添加认知警报（如果有）
        if memory_package.get("cognitive_alert"):
            insights.append({
                "type": "warning",
                "content": memory_package["cognitive_alert"]
            })

        return {
            "status": "integrated",
            "memory_id": memory_package.get("memory_id"),
            "relevance": memory_package.get("relevance_score"),
            "insights": insights,
            "insight_count": len(insights)
        }

    def _generate_strategy_rationale(self, goal: str, mask: str, tool: Optional[str],
                                     strategy: Dict, mental_resources: float,
                                     link_strength: float) -> str:
        """生成策略选择理由"""
        rationale_parts = []

        # 目标关联
        rationale_parts.append(f"目标：{goal}")

        # 面具选择理由
        rationale_parts.append(f"选择『{mask}』面具：适合当前情境和关系深度")

        # 工具选择理由
        if tool:
            rationale_parts.append(f"使用工具『{tool}』：符合当前策略需求")

        # 策略选择理由
        strategy_name = strategy.get("strategy", "未知策略")
        selection_score = strategy.get("selection_score", 0)
        rationale_parts.append(
            f"选择策略『{strategy_name}』：综合评分{selection_score:.2f}/1.0"
        )

        # 资源考量
        resource_note = f"精神资源评分{mental_resources:.2f}/1.0，"
        if mental_resources > 0.7:
            resource_note += "资源充足，适合复杂策略"
        elif mental_resources > 0.4:
            resource_note += "资源适中，需平衡复杂度"
        else:
            resource_note += "资源有限，选择高效策略"

        rationale_parts.append(resource_note)

        # 链接考量
        link_note = f"链接强度{link_strength:.2f}/1.0，"
        if link_strength > 0.7:
            link_note += "信任度高，可进行深度互动"
        elif link_strength > 0.4:
            link_note += "信任度中等，适合常规互动"
        else:
            link_note += "信任度待建立，需谨慎推进"

        rationale_parts.append(link_note)

        # 风险考量
        risk_level = strategy.get("risk_level", "中")
        rationale_parts.append(f"风险评估：{risk_level}风险")

        return "；".join(rationale_parts)