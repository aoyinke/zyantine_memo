"""
降级策略模块 - 当主要API不可用时的备用回复策略
"""
from typing import Dict, List, Optional, Any
from datetime import datetime
import random
import re
from dataclasses import dataclass

from cognition.core_identity import CoreIdentity
from utils.logger import get_logger
from utils.metrics import get_collector


@dataclass
class FallbackContext:
    """降级策略上下文"""
    action_plan: Dict
    growth_result: Dict
    memory_context: Optional[Dict] = None
    user_input: Optional[str] = None
    conversation_history: Optional[List[Dict]] = None
    timestamp: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()


class FallbackStrategy:
    """降级策略管理器"""

    def __init__(self, core_identity: Optional[CoreIdentity] = None):
        """
        初始化降级策略管理器

        Args:
            core_identity: 核心身份（可选）
        """
        self.logger = get_logger("fallback_strategy")
        self.metrics = get_collector("fallback_strategy")
        self.core_identity = core_identity

        # 使用统计
        self.usage_count = 0
        self.strategy_usage = {}

        # 初始化策略
        self._init_strategies()

        self.logger.info("降级策略管理器初始化完成")

    def _init_strategies(self):
        """初始化降级策略"""
        self.strategies = {
            "empathy_acknowledgment": {
                "name": "共情承认策略",
                "description": "对用户情绪表示理解和共情",
                "priority": 80,
                "conditions": ["emotional", "seeking_support"]
            },
            "simple_acknowledgment": {
                "name": "简单承认策略",
                "description": "简单确认收到消息",
                "priority": 60,
                "conditions": ["neutral", "general"]
            },
            "question_clarification": {
                "name": "问题澄清策略",
                "description": "通过提问澄清用户意图",
                "priority": 70,
                "conditions": ["unclear", "ambiguous"]
            },
            "topic_redirect": {
                "name": "话题重定向策略",
                "description": "引导到更安全或熟悉的话题",
                "priority": 50,
                "conditions": ["difficult", "sensitive"]
            },
            "memory_based": {
                "name": "记忆基础策略",
                "description": "基于记忆系统的回复",
                "priority": 90,
                "conditions": ["has_memory", "context_available"]
            },
            "delayed_response": {
                "name": "延迟响应策略",
                "description": "表示需要时间思考",
                "priority": 40,
                "conditions": ["complex", "requires_thought"]
            }
        }

        # 回复模板库
        self.response_templates = {
            "empathy_acknowledgment": [
                "我感受到你现在的情绪，这确实不容易。",
                "我能理解你此刻的心情，这种情况确实令人困扰。",
                "听起来你现在需要一些支持，我在这里陪着你。",
                "我能体会到你的感受，这样的经历确实很挑战。",
                "我能感受到你话语中的情绪，让我们一起面对这个问题。"
            ],
            "simple_acknowledgment": [
                "我收到你的消息了。",
                "我明白你的意思。",
                "谢谢分享，我理解了。",
                "我知道了，谢谢告诉我。",
                "好的，我记下了。"
            ],
            "question_clarification": [
                "能请你再具体说明一下你的问题吗？",
                "我想确认一下我的理解是否正确：你指的是...吗？",
                "关于这个问题，你能告诉我更多细节吗？",
                "为了更好帮助你，我需要了解一些具体情况...",
                "我不太确定是否完全理解，你能换个方式再说一次吗？"
            ],
            "topic_redirect": [
                "我们先换个轻松的话题如何？",
                "也许我们可以聊聊最近让你开心的事情？",
                "我记得我们之前聊过[相关话题]，那个话题现在怎么样？",
                "让我们暂时放下这个问题，聊聊别的吧。",
                "这个话题比较复杂，我们换个角度思考如何？"
            ],
            "delayed_response": [
                "我需要一点时间仔细思考这个问题。",
                "让我好好想想怎么回答你。",
                "这个问题值得深入思考，给我一点时间。",
                "我需要整理一下思绪，稍等一下。",
                "让我先思考一下这个问题的各个方面。"
            ]
        }

    def generate_fallback_reply(self, **kwargs) -> str:
        """
        生成降级回复

        Args:
            action_plan: 动作计划
            growth_result: 成长结果
            memory_context: 记忆上下文
            user_input: 用户输入（可选）
            conversation_history: 对话历史（可选）

        Returns:
            生成的回复文本
        """
        context = FallbackContext(**kwargs)

        self.usage_count += 1
        self.metrics.increment_counter("fallback.requests")

        start_time = datetime.now()

        try:
            # 选择最合适的策略
            selected_strategy = self._select_strategy(context)

            # 生成回复
            reply = self._generate_strategy_reply(selected_strategy, context)

            # 应用核心身份风格
            reply = self._apply_identity_style(reply, context)

            # 记录使用情况
            strategy_name = selected_strategy["name"]
            self.strategy_usage[strategy_name] = self.strategy_usage.get(strategy_name, 0) + 1

            # 记录指标
            generation_time = (datetime.now() - start_time).total_seconds()
            self.metrics.record_histogram("fallback.generation_time", generation_time)
            self.metrics.increment_counter(f"fallback.strategy.{strategy_name}")

            self.logger.info(f"使用降级策略: {strategy_name}, 耗时: {generation_time:.3f}秒")

            return reply

        except Exception as e:
            self.logger.error(f"降级回复生成失败: {e}")
            self.metrics.increment_counter("fallback.errors")

            # 返回最基础的回复
            return self._generate_emergency_reply()

    def _select_strategy(self, context: FallbackContext) -> Dict[str, Any]:
        """选择最合适的策略"""
        action_plan = context.action_plan
        memory_context = context.memory_context
        user_input = context.user_input or ""

        # 分析情境
        situational_analysis = self._analyze_situation(action_plan, user_input, memory_context)

        # 计算每个策略的得分
        strategy_scores = {}

        for strategy_id, strategy in self.strategies.items():
            score = self._calculate_strategy_score(strategy, situational_analysis)
            strategy_scores[strategy_id] = score

        # 选择最高分的策略
        best_strategy_id = max(strategy_scores, key=strategy_scores.get)

        return {
            "id": best_strategy_id,
            "name": self.strategies[best_strategy_id]["name"],
            "score": strategy_scores[best_strategy_id],
            "analysis": situational_analysis
        }

    def _analyze_situation(self, action_plan: Dict, user_input: str,
                           memory_context: Optional[Dict]) -> Dict[str, Any]:
        """分析当前情境"""
        analysis = {
            "is_emotional": False,
            "is_seeking_support": False,
            "is_unclear": False,
            "is_complex": False,
            "has_memory_context": memory_context is not None,
            "mask_type": action_plan.get("chosen_mask", ""),
            "strategy_type": action_plan.get("primary_strategy", ""),
            "emotional_intensity": 0.0,  # 新增：情绪强度
            "support_needs_level": 0.0  # 新增：支持需求等级
        }

        if not user_input:
            return analysis

        user_input_lower = user_input.lower()
        user_input_chars = len(user_input)

        # 更精细的情绪分析
        emotional_keywords = {
            "难过": 0.9, "伤心": 0.9, "悲伤": 0.9, "痛苦": 0.8,
            "生气": 0.8, "愤怒": 0.8, "恼火": 0.7,
            "沮丧": 0.7, "失落": 0.7, "绝望": 0.9,
            "焦虑": 0.6, "压力": 0.6, "紧张": 0.6,
            "抑郁": 0.9, "崩溃": 0.9, "哭了": 0.8, "泪": 0.7
        }

        # 计算情绪强度
        emotional_score = 0.0
        emotional_count = 0
        for keyword, weight in emotional_keywords.items():
            if keyword in user_input_lower:
                emotional_score += weight
                emotional_count += 1

        if emotional_count > 0:
            emotional_intensity = emotional_score / emotional_count
            analysis["is_emotional"] = emotional_intensity > 0.3
            analysis["emotional_intensity"] = min(1.0, emotional_intensity * 1.2)  # 归一化

        # 更精细的支持需求分析
        support_keywords = {
            "帮帮我": 1.0, "怎么办": 0.9, "建议": 0.7, "意见": 0.6,
            "想法": 0.5, "应该": 0.4, "怎么做": 0.9, "如何": 0.8,
            "为什么": 0.6, "原因": 0.5, "求助": 0.9, "指导": 0.7
        }

        support_score = 0.0
        for keyword, weight in support_keywords.items():
            if keyword in user_input_lower:
                support_score += weight

        # 考虑标点符号
        if "?" in user_input or "？" in user_input:
            support_score += 0.3

        analysis["is_seeking_support"] = support_score > 0.5
        analysis["support_needs_level"] = min(1.0, support_score / 3.0)

        # 改进的模糊问题判断
        question_indicators = ["?", "？", "吗", "呢", "什么", "怎样", "如何", "为什么"]
        unclear_indicators = ["有点", "可能", "大概", "也许", "似乎", "好像", "不太", "不太确定"]

        has_question = any(indicator in user_input for indicator in question_indicators)
        has_unclear = any(indicator in user_input_lower for indicator in unclear_indicators)
        is_short = user_input_chars < 15  # 增加长度阈值
        is_long = user_input_chars > 200

        analysis["is_unclear"] = (has_question and has_unclear) or (has_question and is_short) or is_long

        # 更精细的复杂性判断
        complex_keywords = ["解释", "分析", "讨论", "深入", "复杂", "困难", "挑战", "原理", "机制", "系统"]
        analysis["is_complex"] = any(keyword in user_input_lower for keyword in complex_keywords)

        # 检测重复内容（可能表示困惑）
        words = user_input_lower.split()
        if len(words) > 10:
            unique_words = set(words)
            repetition_ratio = len(unique_words) / len(words)
            if repetition_ratio < 0.5:  # 大量重复
                analysis["is_unclear"] = True

        return analysis

    def _calculate_strategy_score(self, strategy: Dict, analysis: Dict) -> float:
        """计算策略得分"""
        score = strategy["priority"] * 0.3  # 基础分

        conditions = strategy.get("conditions", [])

        # 根据条件匹配加分
        for condition in conditions:
            if condition == "emotional" and analysis["is_emotional"]:
                score += 30
            elif condition == "seeking_support" and analysis["is_seeking_support"]:
                score += 25
            elif condition == "unclear" and analysis["is_unclear"]:
                score += 20
            elif condition == "complex" and analysis["is_complex"]:
                score += 15
            elif condition == "has_memory" and analysis["has_memory_context"]:
                score += 25
            elif condition == "neutral" and not analysis["is_emotional"]:
                score += 10

        # 基于面具调整
        mask = analysis["mask_type"]
        if mask:
            if mask == "知己" and strategy["name"] == "共情承认策略":
                score += 20
            elif mask == "长期搭档" and strategy["name"] == "问题澄清策略":
                score += 15
            elif mask == "青梅竹马" and strategy["name"] == "话题重定向策略":
                score += 10

        return min(100, max(0, score))

    def _generate_strategy_reply(self, selected_strategy: Dict,
                                 context: FallbackContext) -> str:
        """生成策略回复"""
        strategy_id = selected_strategy["id"]

        # 如果有记忆上下文，尝试生成记忆相关的回复
        if strategy_id == "memory_based" and context.memory_context:
            reply = self._generate_memory_based_reply(context)
            if reply:
                return reply

        # 获取基础模板
        templates = self.response_templates.get(strategy_id, [])

        if not templates:
            # 如果该策略没有模板，使用简单承认
            templates = self.response_templates.get("simple_acknowledgment", [])

        # 随机选择一个模板
        base_reply = random.choice(templates) if templates else "我收到了你的消息。"

        # 根据情境优化回复
        optimized_reply = self._optimize_reply(base_reply, selected_strategy["analysis"], context)

        return optimized_reply

    def _generate_memory_based_reply(self, context: FallbackContext) -> Optional[str]:
        """生成基于记忆的回复"""
        memory_context = context.memory_context

        if not memory_context:
            return None

        memory_templates = [
            "关于这个话题，我记得我们之前讨论过...",
            "这让我想起我们之前的一次对话...",
            "根据我的记忆，之前我们提到过这个...",
            "我记得我们之前聊过类似的话题...",
            "从过去的对话来看，这个情况有点熟悉..."
        ]

        base_reply = random.choice(memory_templates)

        # 如果有具体的记忆内容，可以引用
        if "resonant_memory" in memory_context:
            memory = memory_context["resonant_memory"]
            if memory and isinstance(memory, dict):
                summary = memory.get("summary", "")[:50]
                if summary:
                    return f"{base_reply} {summary}..."

        return base_reply

    def _optimize_reply(self, base_reply: str, analysis: Dict,
                        context: FallbackContext) -> str:
        """优化回复内容"""
        reply = base_reply

        # 根据情绪状态优化
        if analysis["is_emotional"]:
            emotional_suffixes = [
                " 如果你愿意，可以和我多聊聊。",
                " 我会一直在这里倾听。",
                " 请不要独自承受，说出来会好受些。",
                " 无论发生什么，我都会支持你。"
            ]
            reply += random.choice(emotional_suffixes)

        # 根据面具类型优化语气
        mask = analysis.get("mask_type", "")  # 确保安全获取
        if mask == "知己":
            # 更亲密的语气
            reply = self._add_intimacy_markers(reply)
        elif mask == "长期搭档":
            # 更理性的语气
            reply = self._add_rationality_markers(reply)
        elif mask == "青梅竹马":
            # 更轻松的语气
            reply = self._add_casual_markers(reply)
        elif mask == "伴侣":
            # 更深情的语气
            reply = self._add_affection_markers(reply)

        # 根据用户输入个性化
        if context.user_input:
            reply = self._personalize_reply(reply, context.user_input)

        return reply

    def _add_intimacy_markers(self, reply: str) -> str:
        """添加亲密感标记"""
        markers = ["亲爱的", "你知道吗，", "说实话，", "我想告诉你，"]
        if random.random() > 0.5:
            return random.choice(markers) + reply
        return reply

    def _add_rationality_markers(self, reply: str) -> str:
        """添加理性标记"""
        markers = ["从逻辑上讲，", "客观来说，", "基于现有信息，", "分析一下，"]
        if random.random() > 0.5:
            return random.choice(markers) + reply
        return reply

    def _add_casual_markers(self, reply: str) -> str:
        """添加轻松标记"""
        markers = ["哈哈，", "说起来，", "对了，", "话说回来，"]
        if random.random() > 0.5:
            return random.choice(markers) + reply
        return reply

    def _add_affection_markers(self, reply: str) -> str:
        """添加深情标记"""
        markers = ["宝贝，", "亲爱的，", "你知道吗，我", "我一直觉得，"]
        if random.random() > 0.5:
            return random.choice(markers) + reply
        return reply

    def _personalize_reply(self, reply: str, user_input: str) -> str:
        """个性化回复"""
        # 提取用户输入中的关键词
        import re
        words = re.findall(r'[\w\u4e00-\u9fff]{2,}', user_input)

        if words and len(words) >= 2:
            # 使用用户提到的关键词
            keyword = random.choice(words[:3])
            if keyword not in reply:
                # 以不同方式融入关键词
                incorporations = [
                    f" 特别是关于'{keyword}'的部分。",
                    f" 尤其是'{keyword}'这一点。",
                    f" 关于'{keyword}'，我想了解更多。"
                ]
                if random.random() > 0.7:  # 30%的概率融入关键词
                    reply += random.choice(incorporations)

        return reply

    def _apply_identity_style(self, reply: str, context: FallbackContext) -> str:
        """应用核心身份风格"""
        if not self.core_identity:
            return reply

        # 获取核心身份的口头禅和风格
        catchphrase = self.core_identity.basic_profile.get("catchphrase", "呵~")

        # 随机添加口头禅
        if random.random() > 0.8:  # 20%的概率添加口头禅
            positions = [f"{catchphrase}{reply}", f"{reply}{catchphrase}"]
            reply = random.choice(positions)

        # 根据个性调整语气
        personality = self.core_identity.basic_profile.get("personality", "")

        if "活泼" in personality and random.random() > 0.7:
            reply = self._make_reply_more_lively(reply)
        elif "谨慎" in personality and random.random() > 0.7:
            reply = self._make_reply_more_cautious(reply)

        return reply

    def _make_reply_more_lively(self, reply: str) -> str:
        """使回复更活泼"""
        lively_additions = ["～", "！", "...", "呢", "呀"]
        if reply[-1] not in lively_additions and random.random() > 0.5:
            reply += random.choice(lively_additions)
        return reply

    def _make_reply_more_cautious(self, reply: str) -> str:
        """使回复更谨慎"""
        cautious_additions = ["我觉得", "可能", "也许", "大概"]
        if not any(addition in reply[:10] for addition in cautious_additions):
            if random.random() > 0.5:
                reply = random.choice(cautious_additions) + reply
        return reply

    def _generate_emergency_reply(self) -> str:
        """生成紧急回复"""
        emergency_responses = [
            "我的思维网络有点波动，让我重新连接一下。",
            "意识流暂时不稳定，我需要一点时间调整。",
            "思考回路遇到了些干扰，正在恢复中。",
            "信息处理出现了短暂的延迟，马上就好。"
        ]

        return random.choice(emergency_responses)

    def get_strategy_statistics(self) -> Dict[str, Any]:
        """获取策略统计信息"""
        total_usage = self.usage_count

        if total_usage == 0:
            return {
                "total_usage": 0,
                "strategies": {},
                "success_rate": 100.0
            }

        # 计算各策略使用率
        strategy_stats = {}
        for strategy_name, usage in self.strategy_usage.items():
            percentage = (usage / total_usage) * 100
            strategy_stats[strategy_name] = {
                "usage_count": usage,
                "usage_percentage": round(percentage, 1)
            }

        # 从metrics获取成功率和平均时间
        try:
            all_stats = self.metrics.get_all_statistics()
            success_rate = 100.0
            avg_time = 0.0

            if "fallback.errors" in all_stats:
                error_count = all_stats["fallback.errors"].get("sum", 0)
                if total_usage > 0:
                    success_rate = ((total_usage - error_count) / total_usage) * 100

            if "fallback.generation_time" in all_stats:
                avg_time = all_stats["fallback.generation_time"].get("mean", 0.0)
        except:
            success_rate = 100.0
            avg_time = 0.0

        return {
            "total_usage": total_usage,
            "strategies": strategy_stats,
            "success_rate": round(success_rate, 1),
            "average_generation_time": round(avg_time, 3),
            "last_updated": datetime.now().isoformat()
        }

    def add_custom_strategy(self, strategy_id: str, strategy_config: Dict[str, Any]):
        """
        添加自定义策略

        Args:
            strategy_id: 策略ID
            strategy_config: 策略配置
        """
        if strategy_id in self.strategies:
            self.logger.warning(f"策略 {strategy_id} 已存在，将被覆盖")

        self.strategies[strategy_id] = strategy_config
        self.logger.info(f"添加自定义策略: {strategy_id}")

    def add_custom_template(self, strategy_id: str, template: str):
        """
        添加自定义回复模板

        Args:
            strategy_id: 策略ID
            template: 回复模板
        """
        if strategy_id not in self.response_templates:
            self.response_templates[strategy_id] = []

        self.response_templates[strategy_id].append(template)
        self.logger.info(f"为策略 {strategy_id} 添加模板")

    def clear_statistics(self):
        """清除统计信息"""
        self.usage_count = 0
        self.strategy_usage.clear()
        self.logger.info("已清除策略统计信息")