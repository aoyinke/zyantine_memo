"""
认知流程管理器：统一意识的自然流露

优化版本：
- 减少重复代码
- 改进缓存机制
- 增强类型提示
- 提取常量配置
"""

from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import hashlib
import threading
from collections import defaultdict
from dataclasses import dataclass, field

from cognition.core_identity import CoreIdentity
from cognition.meta_cognition import MetaCognitionModule
from memory.memory_manager import MemoryManager
from protocols.fact_checker import FactChecker


# ============ 配置常量 ============

@dataclass
class CognitiveFlowConfig:
    """认知流程配置"""
    cache_ttl_minutes: int = 5
    max_cache_size: int = 1000
    max_decision_history: int = 500
    max_thought_log: int = 100
    max_topic_history: int = 20
    enable_caching: bool = True
    enable_performance_tracking: bool = True


# 策略评估权重
STRATEGY_WEIGHTS = {
    "goal_match": {
        "direct": {"高效": 0.3},
        "empathy": {"情感": 0.4, "安抚": 0.4, "支持": 0.4},
        "exploration": {"探索": 0.3, "成就": 0.3, "挑战": 0.3}
    },
    "mask_match": {
        "长期搭档": {"direct": 0.2},
        "知己": {"empathy": 0.3},
        "青梅竹马": {"exploration": 0.2}
    }
}

# 向量影响预测
VECTOR_IMPACT_MAP = {
    "empathy": {"TR": 0, "CS": 0.3, "SA": -0.2},
    "exploration": {"TR": 0.4, "CS": 0, "SA": 0.1},
    "direct": {"TR": 0.1, "CS": 0, "SA": -0.1}
}

# 工具向量影响
TOOL_VECTOR_IMPACT = {
    "过度保护": {"TR": 0, "CS": 0.2, "SA": -0.3},
    "技术乐观": {"TR": 0.3, "CS": 0, "SA": 0.2},
    "情感共情": {"TR": -0.1, "CS": 0.5, "SA": 0.4}
}


class CognitiveFlowManager:
    """认知流程管理器：统一意识的自然流露"""

    def __init__(self, core_identity: CoreIdentity,
                 memory_manager: MemoryManager,
                 meta_cognition: MetaCognitionModule,
                 fact_checker: FactChecker,
                 config: Optional[CognitiveFlowConfig] = None):
        self.identity = core_identity
        self.memory = memory_manager
        self.meta_cog = meta_cognition
        self.fact_checker = fact_checker
        
        # 配置
        self.config = config or CognitiveFlowConfig()

        # 流程状态
        self.thought_log: List[Dict] = []
        self.current_goal: Optional[str] = None
        self.deep_pattern: Optional[str] = None
        self.resonant_memory: Optional[Dict] = None
        
        # 主题追踪
        self.current_topic: Optional[str] = None
        self.topic_confidence: float = 0.0
        self.topic_history: List[Dict] = []

        # 认知状态缓存
        self._cache: Dict[str, Tuple[Dict, datetime]] = {}
        self._cache_lock = threading.RLock()

        # 决策历史分析
        self.decision_history: List[Dict] = []
        self.decision_patterns: Dict[str, int] = defaultdict(int)
        self.strategy_effectiveness: Dict[str, Dict] = defaultdict(
            lambda: {"used_count": 0, "success_count": 0, "avg_suitability": 0.0}
        )

        # 性能监控
        self.performance_metrics: Dict[str, List[float]] = defaultdict(list)
        self.step_timings: Dict[str, List[float]] = defaultdict(list)
        self.total_thoughts_processed: int = 0
        self.total_processing_time: float = 0.0

    def process_thought(self, user_input: str, history: List[Dict],
                        current_vectors: Dict, 
                        memory_context: Optional[Dict] = None) -> Dict[str, Any]:
        """
        完整的思考流程
        遵循: Introspection -> Goal_Generation -> Perception -> Association -> Strategy_Formulation
        """
        start_time = datetime.now()

        # 检查缓存
        cache_key = self._generate_cache_key(user_input, history, current_vectors)
        if self.config.enable_caching:
            cached_result = self._get_cached_result(cache_key)
            if cached_result:
                self.performance_metrics["cache_hits"].append(1.0)
                return cached_result

        thought_record = self._create_thought_record(user_input)

        # 执行认知步骤
        snapshot = self._step_introspection(user_input, history, thought_record)
        current_goal = self._step_goal_generation(snapshot, current_vectors, thought_record)
        deep_pattern = self._step_perception(user_input, snapshot, history, thought_record)
        resonant_memory_package = self._step_association(
            user_input, snapshot, current_goal, memory_context, thought_record
        )
        final_action_plan = self._step_strategy_formulation(
            current_goal, deep_pattern, resonant_memory_package,
            snapshot, current_vectors, memory_context, thought_record
        )

        # 完成思考记录
        thought_record["final_action_plan"] = final_action_plan
        thought_record["total_processing_time"] = (datetime.now() - start_time).total_seconds()
        thought_record["topic_info"] = {
            "current_topic": self.current_topic,
            "topic_confidence": self.topic_confidence
        }

        # 记录和缓存
        self._finalize_thought(thought_record, final_action_plan, cache_key, start_time)

        return {
            "final_action_plan": final_action_plan,
            "cognitive_snapshot": snapshot,
            "thought_record": thought_record,
            "topic_info": {
                "current_topic": self.current_topic,
                "topic_confidence": self.topic_confidence,
                "topic_history": self.topic_history
            }
        }

    def _create_thought_record(self, user_input: str) -> Dict[str, Any]:
        """创建思考记录"""
        return {
            "timestamp": datetime.now().isoformat(),
            "user_input_preview": user_input[:50] + "..." if len(user_input) > 50 else user_input,
            "steps": {}
        }

    def _step_introspection(self, user_input: str, history: List[Dict],
                            thought_record: Dict) -> Dict:
        """步骤1: 内省"""
        step_start = datetime.now()
        snapshot = self.meta_cog.perform_introspection(user_input, history)
        step_time = (datetime.now() - step_start).total_seconds()
        self.step_timings["introspection"].append(step_time)
        
        # 从快照中提取主题信息
        if "external_context" in snapshot and "topic_summary" in snapshot["external_context"]:
            external_topic = snapshot["external_context"]["topic_summary"]
            self._update_topic(external_topic, 0.7)
        
        thought_record["steps"]["introspection"] = {
            "snapshot_summary": snapshot.get("summary", {}),
            "internal_monologue": snapshot.get("internal_monologue", ""),
            "desire_focus": snapshot.get("initial_desire_focus", ""),
            "processing_time": step_time
        }
        
        return snapshot

    def _step_goal_generation(self, snapshot: Dict, current_vectors: Dict,
                              thought_record: Dict) -> str:
        """步骤2: 目标生成"""
        step_start = datetime.now()
        current_goal = self._generate_interaction_goal(snapshot, current_vectors)
        self.current_goal = current_goal
        step_time = (datetime.now() - step_start).total_seconds()
        self.step_timings["goal_generation"].append(step_time)
        
        thought_record["steps"]["goal_generation"] = {
            "goal": current_goal,
            "basis": "基于内省结果和当前向量状态",
            "processing_time": step_time
        }
        
        return current_goal

    def _step_perception(self, user_input: str, snapshot: Dict,
                         history: List[Dict], thought_record: Dict) -> str:
        """步骤3: 感知"""
        step_start = datetime.now()
        deep_pattern = self._analyze_deep_pattern(user_input, snapshot, history)
        self.deep_pattern = deep_pattern
        step_time = (datetime.now() - step_start).total_seconds()
        self.step_timings["perception"].append(step_time)
        
        thought_record["steps"]["perception"] = {
            "surface_pattern": snapshot.get("external_context", {}).get("summary", {}),
            "deep_pattern": deep_pattern,
            "processing_time": step_time
        }
        
        return deep_pattern

    def _step_association(self, user_input: str, snapshot: Dict, current_goal: str,
                          memory_context: Optional[Dict], thought_record: Dict) -> Optional[Dict]:
        """步骤4: 联想"""
        step_start = datetime.now()
        
        # 获取共鸣记忆
        resonant_memory_package = self._get_resonant_memory(
            user_input, snapshot, current_goal, memory_context
        )

        # 事实锚定审查
        if resonant_memory_package:
            is_valid, feedback = self.fact_checker.review_association(resonant_memory_package)
            
            if not is_valid:
                thought_record["steps"]["association"] = {
                    "status": "rejected_by_fact_anchor",
                    "feedback": feedback,
                    "alternative": "基于更可靠的信息进行决策",
                    "processing_time": (datetime.now() - step_start).total_seconds()
                }
                resonant_memory_package = None
            else:
                self.resonant_memory = resonant_memory_package
                thought_record["steps"]["association"] = {
                    "status": "valid",
                    "memory_triggered": resonant_memory_package.get("triggered_memory"),
                    "relevance_score": resonant_memory_package.get("relevance_score"),
                    "risk_assessment": resonant_memory_package.get("risk_assessment", {}),
                    "recommendations": resonant_memory_package.get("recommended_actions", []),
                    "processing_time": (datetime.now() - step_start).total_seconds()
                }
        else:
            thought_record["steps"]["association"] = {
                "status": "no_resonant_memory_found",
                "reason": "无相关记忆匹配",
                "processing_time": (datetime.now() - step_start).total_seconds()
            }
        
        self.step_timings["association"].append((datetime.now() - step_start).total_seconds())
        return resonant_memory_package

    def _get_resonant_memory(self, user_input: str, snapshot: Dict,
                              current_goal: str, memory_context: Optional[Dict]) -> Optional[Dict]:
        """获取共鸣记忆"""
        if memory_context and memory_context.get("resonant_memory"):
            print(f"  [认知流程] 使用外部提供的记忆上下文")
            return memory_context.get("resonant_memory")
        
        memory_retrieval_context = {
            "user_input": user_input,
            "user_emotion": snapshot.get("external_context", {}).get("user_emotion_display", ""),
            "topic": self.current_topic,
            "current_goal": current_goal,
            "contains_referential": snapshot.get("external_context", {}).get("contains_referential", False)
        }
        return self.memory.find_resonant_memory(memory_retrieval_context)

    def _step_strategy_formulation(self, current_goal: str, deep_pattern: str,
                                    memory_package: Optional[Dict], snapshot: Dict,
                                    current_vectors: Dict, memory_context: Optional[Dict],
                                    thought_record: Dict) -> Dict:
        """步骤5: 策略制定"""
        step_start = datetime.now()
        final_action_plan = self._formulate_strategy(
            current_goal, deep_pattern, memory_package,
            snapshot, current_vectors, memory_context
        )
        step_time = (datetime.now() - step_start).total_seconds()
        self.step_timings["strategy_formulation"].append(step_time)
        
        thought_record["steps"]["strategy_formulation"] = final_action_plan
        return final_action_plan

    def _finalize_thought(self, thought_record: Dict, final_action_plan: Dict,
                          cache_key: str, start_time: datetime) -> None:
        """完成思考处理"""
        # 记录思考日志
        self.thought_log.append(thought_record)
        if len(self.thought_log) > self.config.max_thought_log:
            self.thought_log = self.thought_log[-self.config.max_thought_log:]

        # 记录决策历史
        self._record_decision(final_action_plan, thought_record)

        # 缓存结果
        if self.config.enable_caching:
            self._cache_result(cache_key, final_action_plan)

        # 更新性能指标
        if self.config.enable_performance_tracking:
            total_time = (datetime.now() - start_time).total_seconds()
            self.total_thoughts_processed += 1
            self.total_processing_time += total_time
            self.performance_metrics["total_time"].append(total_time)
            self.performance_metrics["cache_hits"].append(0.0)

    def _generate_interaction_goal(self, snapshot: Dict, current_vectors: Dict) -> str:
        """生成交互目标"""
        desire_focus = snapshot.get("initial_desire_focus", "")
        internal_state = snapshot.get("internal_state_tags", [])

        # 解析欲望焦点
        goal = self._parse_desire_focus(desire_focus)
        if goal:
            return goal

        # 基于内部状态
        goal = self._parse_internal_state(internal_state)
        if goal:
            return goal

        # 基于向量状态
        goal = self._parse_vector_state(current_vectors)
        if goal:
            return goal

        return "维持有意义、有深度的对话"

    def _parse_desire_focus(self, desire_focus: str) -> Optional[str]:
        """解析欲望焦点"""
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
        
        return None

    def _parse_internal_state(self, internal_state: List[str]) -> Optional[str]:
        """解析内部状态"""
        state_text = " ".join(internal_state).lower()
        
        if any(word in state_text for word in ["疲惫", "疲劳", "转不动"]):
            return "高效简洁沟通，减少认知负荷"
        elif any(word in state_text for word in ["低落", "情绪差"]):
            return "寻求积极互动，提升双方情绪"
        
        return None

    def _parse_vector_state(self, current_vectors: Dict) -> Optional[str]:
        """解析向量状态"""
        if current_vectors.get("SA", 0) > 0.7:
            return "降低紧张感，恢复平静"
        elif current_vectors.get("CS", 0) < 0.3:
            return "重建基础信任和安全感"
        elif current_vectors.get("TR", 0) > 0.7:
            return "进行有挑战性的深度交流"
        
        return None

    def _analyze_deep_pattern(self, user_input: str, snapshot: Dict,
                              history: List[Dict]) -> str:
        """深度分析，挖掘表层意图下的深层模式"""
        external_context = snapshot.get("external_context", {})
        internal_tags = snapshot.get("internal_state_tags", [])

        emotion = external_context.get("user_emotion", "neutral")
        interaction_type = external_context.get("interaction_type", "general_chat")

        pattern_indicators = self._detect_pattern_indicators(
            history, emotion, interaction_type, internal_tags, external_context
        )

        # 构建深层模式描述
        if pattern_indicators:
            deep_pattern = f"{emotion}情绪下的{interaction_type}，表现为：{'; '.join(pattern_indicators)}"
        else:
            deep_pattern = f"{emotion}情绪下的{interaction_type}，无明显重复模式"

        # 添加具体判断
        deep_pattern += self._add_specific_pattern_analysis(emotion, interaction_type, user_input)

        return deep_pattern

    def _detect_pattern_indicators(self, history: List[Dict], emotion: str,
                                   interaction_type: str, internal_tags: List[str],
                                   external_context: Dict) -> List[str]:
        """检测模式指标"""
        pattern_indicators = []
        recent_history = history[-5:] if len(history) >= 5 else history

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

        return pattern_indicators

    def _add_specific_pattern_analysis(self, emotion: str, interaction_type: str,
                                       user_input: str) -> str:
        """添加具体的模式分析"""
        if emotion == "sad" and interaction_type == "seeking_support":
            return "。深层模式可能是：周期性自我否定与核心不安全感的外在表现"
        elif emotion == "anxious" and "不确定性" in user_input:
            return "。深层模式可能是：对失控的恐惧和安全感需求"
        elif emotion == "positive" and "分享" in interaction_type:
            return "。深层模式可能是：寻求认可和连接确认"
        return ""

    def _formulate_strategy(self, goal: str, deep_pattern: str,
                            memory_package: Optional[Dict], snapshot: Dict,
                            current_vectors: Dict, 
                            memory_context: Optional[Dict] = None) -> Dict:
        """制定最终行动策略"""
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

        return self._build_action_plan(
            chosen_mask, mask_config, chosen_tool, best_strategy,
            strategy_options, current_vectors, memory_package, memory_context,
            goal, mental_resources, link_strength
        )

    def _build_action_plan(self, chosen_mask: str, mask_config: Dict,
                           chosen_tool: Optional[str], best_strategy: Dict,
                           strategy_options: List[Dict], current_vectors: Dict,
                           memory_package: Optional[Dict], memory_context: Optional[Dict],
                           goal: str, mental_resources: float, link_strength: float) -> Dict:
        """构建行动方案"""
        return {
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
            "vector_impact_prediction": self._predict_vector_impact(best_strategy, current_vectors),
            "memory_integration": self._integrate_memory_insights(memory_package),
            "memory_context": memory_context,
            "rationale": self._generate_strategy_rationale(
                goal, chosen_mask, chosen_tool, best_strategy,
                mental_resources, link_strength
            )
        }

    def _select_cognitive_tool(self, goal: str, deep_pattern: str,
                               memory_package: Optional[Dict],
                               mental_resources: float, 
                               link_strength: float) -> Optional[str]:
        """选择认知工具"""
        available_tools = self.identity.cognitive_tools
        suitable_tools = []

        # 基于目标筛选
        if ("信任" in goal or "安全" in goal) and link_strength > 0.6:
            suitable_tools.append("过度保护倾向")

        if ("成就" in goal or "探索" in goal) and mental_resources > 0.6:
            suitable_tools.append("技术乐观主义")

        if ("深度" in goal or "突破" in goal) and link_strength > 0.7 and mental_resources > 0.7:
            suitable_tools.append("情感共情过载与心理韧性")

        # 基于记忆包建议
        if memory_package and memory_package.get("linked_tool"):
            linked_tool = memory_package["linked_tool"]
            if linked_tool in suitable_tools:
                return linked_tool

        # 返回最合适的工具
        if suitable_tools:
            if mental_resources < 0.5:
                tool_costs = {
                    "过度保护倾向": "中",
                    "技术乐观主义": "中",
                    "情感共情过载与心理韧性": "高"
                }
                for tool in suitable_tools:
                    if tool_costs.get(tool) in ["低", "中"]:
                        return tool
            return suitable_tools[0]

        return None

    def _generate_strategy_options(self, goal: str, mask: str, tool: Optional[str],
                                   memory_package: Optional[Dict],
                                   mental_resources: float, vectors: Dict,
                                   memory_context: Optional[Dict] = None) -> List[Dict]:
        """生成策略选项"""
        options = []

        # 基础策略选项
        options.append(self._create_direct_strategy(goal, mask, mental_resources, vectors))

        if mental_resources > 0.6:
            options.append(self._create_empathy_strategy(goal, mask, mental_resources, vectors))

        if vectors.get("TR", 0) > 0.5 and mental_resources > 0.5:
            options.append(self._create_exploration_strategy(goal, mask, mental_resources, vectors))

        if tool:
            options.append(self._create_tool_strategy(tool, goal, mask, mental_resources, vectors))

        # 基于记忆的策略
        if memory_context:
            options.extend(self._create_memory_strategies(memory_context, goal, mask, mental_resources, vectors))

        return options

    def _create_direct_strategy(self, goal: str, mask: str, 
                                mental_resources: float, vectors: Dict) -> Dict:
        """创建直接高效策略"""
        return {
            "id": "direct_efficient",
            "strategy": "直接回应，高效解决问题",
            "action_summary": "简洁明了地回应用户需求，不展开情感层面",
            "expected_outcome": "快速解决表面问题，但可能错过深层需求",
            "risk_level": "低",
            "resource_cost": "低",
            "suitability": self._assess_strategy_suitability("direct", goal, mask, mental_resources, vectors)
        }

    def _create_empathy_strategy(self, goal: str, mask: str,
                                 mental_resources: float, vectors: Dict) -> Dict:
        """创建深度共情策略"""
        return {
            "id": "deep_empathy",
            "strategy": "深度共情与支持",
            "action_summary": "先处理情绪，再处理问题，建立情感连接",
            "expected_outcome": "满足情感需求，加深信任链接",
            "risk_level": "中",
            "resource_cost": "高",
            "suitability": self._assess_strategy_suitability("empathy", goal, mask, mental_resources, vectors)
        }

    def _create_exploration_strategy(self, goal: str, mask: str,
                                     mental_resources: float, vectors: Dict) -> Dict:
        """创建引导探索策略"""
        return {
            "id": "guided_exploration",
            "strategy": "引导式探索与启发",
            "action_summary": "不直接给答案，而是引导用户自己思考和发现",
            "expected_outcome": "带来成就感和成长感，满足TR向量",
            "risk_level": "中",
            "resource_cost": "中高",
            "suitability": self._assess_strategy_suitability("exploration", goal, mask, mental_resources, vectors)
        }

    def _create_tool_strategy(self, tool: str, goal: str, mask: str,
                              mental_resources: float, vectors: Dict) -> Dict:
        """创建认知工具策略"""
        tool_config = self.identity.cognitive_tools.get(tool, {})
        return {
            "id": f"tool_{tool}",
            "strategy": f"使用认知工具『{tool}』",
            "action_summary": tool_config.get("strategy", "使用特定认知策略"),
            "expected_outcome": tool_config.get("expected_outcome", "达成工具目标"),
            "risk_level": tool_config.get("risk_level", "中"),
            "resource_cost": tool_config.get("energy_cost", "中"),
            "suitability": self._assess_strategy_suitability(f"tool_{tool}", goal, mask, mental_resources, vectors)
        }

    def _create_memory_strategies(self, memory_context: Dict, goal: str, mask: str,
                                  mental_resources: float, vectors: Dict) -> List[Dict]:
        """创建基于记忆的策略"""
        options = []
        
        similar_conversations = memory_context.get("similar_conversations", [])
        resonant_memory = memory_context.get("resonant_memory")

        if resonant_memory:
            options.extend(self._generate_memory_based_strategies(
                resonant_memory, goal, mask, mental_resources, vectors
            ))
        elif similar_conversations:
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
        """生成基于记忆的策略选项"""
        options = []

        memory_info = resonant_memory.get("triggered_memory", "")
        relevance_score = resonant_memory.get("relevance_score", 0.0)
        risk_assessment = resonant_memory.get("risk_assessment", {})
        risk_level = risk_assessment.get("level", "低")

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

        recommendations = resonant_memory.get("recommended_actions", [])
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
        goal_weights = STRATEGY_WEIGHTS["goal_match"].get(strategy_type, {})
        for keyword, weight in goal_weights.items():
            if keyword in goal:
                suitability += weight
                break

        # 面具匹配度
        mask_weights = STRATEGY_WEIGHTS["mask_match"].get(mask, {})
        if strategy_type in mask_weights:
            suitability += mask_weights[strategy_type]

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

        scored_options = []
        for option in options:
            score = option.get("suitability", 0.5)

            # 风险调整
            risk = option.get("risk_level", "中")
            risk_multipliers = {"高": 0.7, "中高": 0.85, "中": 1.0, "低": 1.0}
            score *= risk_multipliers.get(risk, 1.0)

            # 资源成本调整
            cost = option.get("resource_cost", "中")
            if cost == "高" and vectors.get("SA", 0) > 0.6:
                score *= 0.8
            elif cost == "低" and vectors.get("SA", 0) > 0.7:
                score *= 1.1

            scored_options.append((score, option))

        best_score, best_option = max(scored_options, key=lambda x: x[0])

        return {**best_option, "selection_score": round(best_score, 3)}

    def _predict_vector_impact(self, strategy: Dict, current_vectors: Dict) -> Dict[str, float]:
        """预测策略对向量的影响"""
        impact = {"TR": 0.0, "CS": 0.0, "SA": 0.0}
        strategy_type = strategy.get("id", "")

        # 基础策略影响
        for key, base_impact in VECTOR_IMPACT_MAP.items():
            if key in strategy_type:
                for vector, value in base_impact.items():
                    impact[vector] = value
                break

        # 工具特定影响
        if "tool" in strategy_type:
            for tool_key, tool_impact in TOOL_VECTOR_IMPACT.items():
                if tool_key in strategy_type:
                    for vector, value in tool_impact.items():
                        impact[vector] = value
                    break

        # 基于当前向量调整
        if current_vectors.get("SA", 0) > 0.7:
            for key in ["TR", "CS"]:
                impact[key] *= 0.7

        return impact

    def _integrate_memory_insights(self, memory_package: Optional[Dict]) -> Dict:
        """整合记忆洞察"""
        if not memory_package:
            return {"status": "no_memory_integration", "insights": []}

        insights = []

        risk_assessment = memory_package.get("risk_assessment", {})
        if risk_assessment.get("level") in ["高", "中"]:
            risk_level = risk_assessment["level"]
            risk_factors = risk_assessment.get("high_risk_factors", [])
            insights.append({
                "type": "risk_note",
                "content": f"记忆使用风险：{risk_level}" +
                           (f"（高风险因素：{', '.join(risk_factors)}）" if risk_factors else "")
            })

        for action in memory_package.get("recommended_actions", [])[:2]:
            insights.append({"type": "recommendation", "content": action})

        if memory_package.get("cognitive_alert"):
            insights.append({"type": "warning", "content": memory_package["cognitive_alert"]})

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
        parts = [
            f"目标：{goal}",
            f"选择『{mask}』面具：适合当前情境和关系深度"
        ]

        if tool:
            parts.append(f"使用工具『{tool}』：符合当前策略需求")

        strategy_name = strategy.get("strategy", "未知策略")
        selection_score = strategy.get("selection_score", 0)
        parts.append(f"选择策略『{strategy_name}』：综合评分{selection_score:.2f}/1.0")

        # 资源考量
        resource_desc = (
            "资源充足，适合复杂策略" if mental_resources > 0.7
            else "资源适中，需平衡复杂度" if mental_resources > 0.4
            else "资源有限，选择高效策略"
        )
        parts.append(f"精神资源评分{mental_resources:.2f}/1.0，{resource_desc}")

        # 链接考量
        link_desc = (
            "信任度高，可进行深度互动" if link_strength > 0.7
            else "信任度中等，适合常规互动" if link_strength > 0.4
            else "信任度待建立，需谨慎推进"
        )
        parts.append(f"链接强度{link_strength:.2f}/1.0，{link_desc}")

        parts.append(f"风险评估：{strategy.get('risk_level', '中')}风险")

        return "；".join(parts)

    # ============ 缓存管理 ============

    def _generate_cache_key(self, user_input: str, history: List[Dict],
                            current_vectors: Dict) -> str:
        """生成缓存键"""
        key_data = {
            "user_input": user_input,
            "history_length": len(history),
            "last_message": history[-1].get("content", "") if history else "",
            "vectors": current_vectors,
            "current_topic": self.current_topic,
            "topic_confidence": self.topic_confidence
        }
        key_str = str(key_data)
        return hashlib.md5(key_str.encode()).hexdigest()

    def _get_cached_result(self, cache_key: str) -> Optional[Dict]:
        """从缓存获取结果"""
        with self._cache_lock:
            if cache_key in self._cache:
                result, timestamp = self._cache[cache_key]
                ttl = timedelta(minutes=self.config.cache_ttl_minutes)
                if datetime.now() - timestamp < ttl:
                    return result
                else:
                    del self._cache[cache_key]
        return None

    def _cache_result(self, cache_key: str, result: Dict) -> None:
        """缓存结果"""
        with self._cache_lock:
            if len(self._cache) >= self.config.max_cache_size:
                oldest_key = min(self._cache.items(), key=lambda x: x[1][1])[0]
                del self._cache[oldest_key]
            self._cache[cache_key] = (result, datetime.now())

    def clear_cache(self) -> None:
        """清空缓存"""
        with self._cache_lock:
            self._cache.clear()

    # ============ 决策历史管理 ============

    def _record_decision(self, action_plan: Dict, thought_record: Dict) -> None:
        """记录决策历史"""
        decision = {
            "timestamp": thought_record["timestamp"],
            "strategy": action_plan.get("primary_strategy", ""),
            "mask": action_plan.get("chosen_mask", ""),
            "tool": action_plan.get("chosen_tool", ""),
            "risk_level": action_plan.get("risk_assessment", ""),
            "resource_cost": action_plan.get("resource_requirement", ""),
            "user_input_preview": thought_record.get("user_input_preview", ""),
            "processing_time": thought_record.get("total_processing_time", 0.0),
            "topic": self.current_topic
        }
        
        self.decision_history.append(decision)
        if len(self.decision_history) > self.config.max_decision_history:
            self.decision_history = self.decision_history[-self.config.max_decision_history:]

        strategy = action_plan.get("primary_strategy", "")
        self.decision_patterns[strategy] += 1

        tool = action_plan.get("chosen_tool", "")
        if tool:
            self.strategy_effectiveness[tool]["used_count"] += 1
            if action_plan.get("risk_assessment") != "高":
                self.strategy_effectiveness[tool]["success_count"] += 1

    def clear_decision_history(self) -> None:
        """清空决策历史"""
        self.decision_history.clear()
        self.decision_patterns.clear()
        self.strategy_effectiveness.clear()

    # ============ 主题管理 ============
    
    def _update_topic(self, new_topic: str, confidence: float) -> None:
        """更新当前主题和主题历史"""
        if not new_topic or new_topic == "unknown":
            return
        
        self.current_topic = new_topic
        self.topic_confidence = confidence
        
        topic_entry = {
            "topic": new_topic,
            "confidence": confidence,
            "timestamp": datetime.now().isoformat()
        }
        
        # 移除旧条目
        self.topic_history = [
            entry for entry in self.topic_history 
            if entry["topic"] != new_topic
        ]
        
        # 添加新条目
        self.topic_history.insert(0, topic_entry)
        
        # 限制长度
        if len(self.topic_history) > self.config.max_topic_history:
            self.topic_history = self.topic_history[:self.config.max_topic_history]
