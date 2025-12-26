"""
回复生成器 - 智能回复生成
"""
from typing import Dict, List, Optional, Any
from datetime import datetime
import random
import traceback
from dataclasses import dataclass

from .openai_service import OpenAIService
from .prompt_engine import PromptEngine
from .fallback_strategy import FallbackStrategy
# 修改导入路径
from cognition.core_identity import CoreIdentity
from utils.logger import SystemLogger
from utils.metrics import MetricsCollector


@dataclass
class GenerationContext:
    """生成上下文"""
    user_input: str
    action_plan: Dict
    growth_result: Dict
    context_analysis: Dict
    conversation_history: List[Dict]
    core_identity: CoreIdentity
    current_vectors: Dict
    memory_context: Optional[Dict] = None
    metadata: Optional[Dict] = None
    cognitive_flow_id: Optional[str] = None
    timestamp: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()


class APIBasedReplyGenerator:
    """基于API的智能回复生成器（增强版）"""

    def __init__(self,
                 api_service: Optional[OpenAIService],
                 prompt_engine: PromptEngine,
                 fallback_strategy: FallbackStrategy,
                 metrics_collector: Optional[MetricsCollector] = None,
                 core_identity: Optional[CoreIdentity] = None):  # 新增参数
        self.api = api_service
        self.prompt_engine = prompt_engine
        self.fallback_strategy = fallback_strategy
        self.metrics = metrics_collector or MetricsCollector("reply_generator")
        self.core_identity = core_identity  # 存储核心身份

        self.logger = SystemLogger().get_logger("reply_generator")

        # 生成日志
        self.generation_log: List[Dict] = []
        self.error_log: List[Dict] = []

        # 缓存
        self.prompt_cache: Dict[str, str] = {}

        self.logger.info("回复生成器初始化完成")

    def generate_reply(self, cognitive_result: Dict = None, **kwargs) -> str:
        """
        生成回复

        Args:
            cognitive_result: 认知流程结果（新版本）
            **kwargs: 其他参数，向后兼容旧版本

        Returns:
            生成的回复文本
        """
        if cognitive_result:
            # 新版本调用方式
            return self.generate_from_cognitive_flow(cognitive_result)
        else:
            # 旧版本兼容
            return self._generate_with_legacy_api(**kwargs)

    def _generate_with_legacy_api(self, **kwargs) -> str:
        """旧版本生成方式，保持兼容"""
        start_time = datetime.now()

        try:
            # 构建生成上下文
            context = GenerationContext(**kwargs)

            self.logger.info(f"开始生成回复，用户输入: {context.user_input[:50]}...")

            # 检查API是否可用
            if self.api and self.api.is_available():
                reply = self._generate_with_api(context)
            else:
                self.logger.warning("API不可用，使用降级策略")
                reply = self._generate_with_fallback(context)

            # 记录生成日志
            self._log_generation(context, reply, start_time)

            self.logger.info(f"回复生成成功，长度: {len(reply)}")
            return reply

        except Exception as e:
            self._log_error(None, e, start_time)
            return self._generate_emergency_reply(None)

    def generate_from_cognitive_flow(self, cognitive_result: Dict) -> str:
        """直接从认知流程结果生成回复"""
        start_time = datetime.now()

        try:
            # 提取必要信息
            action_plan = cognitive_result.get("final_action_plan", {})
            memory_context = cognitive_result.get("memory_context", {})
            user_input = cognitive_result.get("user_input", "")
            conversation_history = cognitive_result.get("conversation_history", [])

            # 获取核心身份
            core_identity = self._get_core_identity()

            # 构建上下文
            context = GenerationContext(
                user_input=user_input,
                action_plan=action_plan,
                growth_result=cognitive_result.get("growth_result", {}),
                context_analysis=cognitive_result.get("context_analysis", {}),
                conversation_history=conversation_history,
                core_identity=core_identity,
                current_vectors=cognitive_result.get("current_vectors", {}),
                memory_context=memory_context,
                metadata={
                    "cognitive_flow_id": cognitive_result.get("flow_id"),
                    "deep_pattern": cognitive_result.get("deep_pattern", "")
                },
                cognitive_flow_id=cognitive_result.get("flow_id")
            )

            # 根据面具和策略调整生成方式
            mask = action_plan.get("chosen_mask", "")
            strategy = action_plan.get("primary_strategy", "")

            if mask and strategy:
                self.logger.info(f"使用面具 '{mask}' 和策略 '{strategy}' 生成回复")

            # 生成回复
            if self.api and self.api.is_available():
                reply = self._generate_with_api(context)
            else:
                reply = self._generate_with_fallback(context)

            # 记录日志
            self._log_generation(context, reply, start_time)

            return reply

        except Exception as e:
            self.logger.error(f"认知流程生成失败: {e}")
            return self._generate_emergency_reply(None)

    def _build_system_prompt(self, context: GenerationContext) -> str:
        """构建系统提示词"""
        # 尝试从缓存获取
        cache_key = self._generate_prompt_cache_key(context)

        if cache_key in self.prompt_cache:
            self.logger.debug("使用缓存的提示词")
            return self.prompt_cache[cache_key]

        # 使用提示词引擎构建
        prompt = self.prompt_engine.build_prompt(
            action_plan=context.action_plan,
            growth_result=context.growth_result,
            context_analysis=context.context_analysis,
            core_identity=context.core_identity,
            current_vectors=context.current_vectors,
            memory_context=context.memory_context
        )

        # 缓存提示词
        self.prompt_cache[cache_key] = prompt

        # 保持缓存大小
        if len(self.prompt_cache) > 100:
            # 移除最旧的条目
            oldest_key = next(iter(self.prompt_cache))
            del self.prompt_cache[oldest_key]

        return prompt

    def _generate_prompt_cache_key(self, context: GenerationContext) -> str:
        """生成提示词缓存键"""
        # 使用关键信息生成哈希键
        import hashlib

        key_data = {
            "mask": context.action_plan.get("chosen_mask", ""),
            "strategy": context.action_plan.get("primary_strategy", ""),
            "tr": round(context.current_vectors.get("TR", 0), 2),
            "cs": round(context.current_vectors.get("CS", 0), 2),
            "sa": round(context.current_vectors.get("SA", 0), 2),
        }

        key_str = str(sorted(key_data.items()))
        return hashlib.md5(key_str.encode()).hexdigest()[:16]

    def _determine_max_tokens(self, context_analysis: Dict) -> int:
        """确定最大token数"""
        complexity = context_analysis.get("topic_complexity", "medium")

        token_map = {
            "high": 800,
            "medium": 500,
            "low": 300
        }

        return token_map.get(complexity, 500)

    def _determine_temperature(self, current_vectors: Dict) -> float:
        """确定温度参数"""
        tr = current_vectors.get("TR", 0.5)
        cs = current_vectors.get("CS", 0.5)
        sa = current_vectors.get("SA", 0.5)

        # 高压状态需要更稳定的回复
        if sa > 0.7:
            return 0.4
        # 高兴奋状态可以更有创造性
        elif tr > 0.7 and cs > 0.6:
            return 0.8
        # 默认状态
        else:
            return 0.7

    def _generate_with_fallback(self, context: GenerationContext) -> str:
        """使用降级策略生成回复"""
        return self.fallback_strategy.generate_fallback_reply(
            action_plan=context.action_plan,
            growth_result=context.growth_result,
            memory_context=context.memory_context
        )

    def _generate_emergency_reply(self, context: GenerationContext) -> str:
        """生成紧急回复"""
        emergency_responses = [
            "我的思考过程出现了一些混乱，能请你再问一次吗？",
            "刚才的思考链路好像打了个结，我们重新开始吧。",
            "意识流有点波动，让我重新整理一下思绪。"
        ]

        return random.choice(emergency_responses)

    def _log_generation(self, context: GenerationContext, reply: str, start_time: datetime):
        """记录生成日志"""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "user_input": context.user_input[:100],
            "reply_preview": reply[:100] + "..." if len(reply) > 100 else reply,
            "reply_length": len(reply),
            "generation_time": (datetime.now() - start_time).total_seconds(),
            "used_api": self.api is not None and self.api.is_available(),
            "mask": context.action_plan.get("chosen_mask", ""),
            "strategy": context.action_plan.get("primary_strategy", ""),
            "vectors": context.current_vectors
        }

        self.generation_log.append(log_entry)

        # 保持日志大小
        if len(self.generation_log) > 200:
            self.generation_log = self.generation_log[-200:]

        # 记录指标
        if self.metrics:
            self.metrics.record_reply_generation(
                success=True,
                length=len(reply),
                generation_time=(datetime.now() - start_time).total_seconds(),
                used_api=log_entry["used_api"]
            )

    def _log_error(self, context: GenerationContext, error: Exception, start_time: datetime):
        """记录错误日志"""
        error_entry = {
            "timestamp": datetime.now().isoformat(),
            "user_input": context.user_input[:100],
            "error_type": type(error).__name__,
            "error_message": str(error),
            "action_plan": str(context.action_plan)[:200],
            "vectors": context.current_vectors,
            "generation_time": (datetime.now() - start_time).total_seconds()
        }

        self.error_log.append(error_entry)

        # 保持日志大小
        if len(self.error_log) > 100:
            self.error_log = self.error_log[-100:]

        # 记录指标
        if self.metrics:
            self.metrics.record_reply_generation(
                success=False,
                length=0,
                generation_time=(datetime.now() - start_time).total_seconds(),
                used_api=False
            )

    def get_generation_log(self, limit: int = 10) -> List[Dict]:
        """获取生成日志"""
        return self.generation_log[-limit:] if self.generation_log else []

    def get_error_log(self, limit: int = 10) -> List[Dict]:
        """获取错误日志"""
        return self.error_log[-limit:] if self.error_log else []

    def clear_logs(self):
        """清空日志"""
        self.generation_log.clear()
        self.error_log.clear()
        self.logger.info("已清空生成日志")

    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        total_generations = len(self.generation_log)
        successful_generations = len([log for log in self.generation_log if log.get("used_api", False)])

        api_stats = self.api.get_statistics() if self.api else {}

        return {
            "total_generations": total_generations,
            "successful_generations": successful_generations,
            "error_count": len(self.error_log),
            "success_rate": (successful_generations / max(total_generations, 1)) * 100,
            "prompt_cache_size": len(self.prompt_cache),
            "api_service_stats": api_stats,
            "fallback_strategy_used": self.fallback_strategy.usage_count if hasattr(self.fallback_strategy,
                                                                                    'usage_count') else 0
        }

    def clear_cache(self):
        """清空缓存"""
        self.prompt_cache.clear()
        self.logger.info("已清空提示词缓存")

class TemplateReplyGenerator:
    """模板回复生成器（当API不可用时使用），适配新的记忆系统"""

    def __init__(self):
        self.templates = self._load_templates()

    def _load_templates(self) -> Dict:
        """加载回复模板"""
        return {
            "长期搭档": [
                "关于这个问题，我的分析是：{strategy}。你怎么看？",
                "从我的角度考虑，建议：{strategy}。",
                "根据我们之前的讨论，我认为：{strategy}。",
                "这个问题很有意思，我觉得可以这样考虑：{strategy}。"
            ],
            "知己": [
                "我理解你的感受。{strategy}",
                "其实我也有过类似的经历。{strategy}",
                "跟你说说我的想法：{strategy}",
                "我能体会到你的心情。{strategy}"
            ],
            "青梅竹马": [
                "哈哈，这让我想起以前...{strategy}",
                "你总是能提出有趣的问题！{strategy}",
                "记得你之前也说过类似的话...{strategy}",
                "哎呀，这个我熟！{strategy}"
            ],
            "伴侣": [
                "我深深感受到...{strategy}",
                "这对我很重要，因为...{strategy}",
                "我想和你分享的是...{strategy}",
                "你知道的，我总是...{strategy}"
            ]
        }

    def generate(self, mask: str, strategy: str, growth_result: Dict = None,
                 memory_context: Optional[Dict] = None) -> str:
        """使用模板生成回复，适配新的记忆系统"""
        template_list = self.templates.get(mask, self.templates["长期搭档"])
        template = random.choice(template_list)

        # 填充策略
        reply = template.format(strategy=strategy)

        # 融入辩证成长成果
        if growth_result and growth_result.get("validation") == "success":
            new_principle = growth_result.get("new_principle", {})
            if isinstance(new_principle, dict) and "abstracted_from" in new_principle:
                reply += f" （这让我想起了{new_principle['abstracted_from']}）"

        # 如果有记忆上下文，尝试增强回复
        if memory_context:
            reply = self._enhance_with_memory(reply, memory_context)

        return reply

    def _enhance_with_memory(self, base_reply: str, memory_context: Dict) -> str:
        """使用记忆信息增强模板回复，适配新的记忆系统"""
        resonant_memory = memory_context.get("resonant_memory")

        if resonant_memory:
            memory_info = resonant_memory.get("triggered_memory", "")
            risk_assessment = resonant_memory.get("risk_assessment", {})
            risk_level = risk_assessment.get("level", "低")

            if memory_info:
                # 根据风险级别调整回复
                if risk_level == "低":
                    # 安全记忆，可以大胆引用
                    memory_enhancement = f" 这让我想起：{memory_info}"
                    base_reply += memory_enhancement
                elif risk_level == "中":
                    # 中等风险，谨慎引用
                    memory_enhancement = f" 我记得类似的情况..."
                    base_reply += memory_enhancement
                else:
                    # 高风险，不直接引用记忆，但可以暗示
                    memory_enhancement = " 基于过去的经验..."
                    base_reply += memory_enhancement

                # 添加建议
                recommendations = resonant_memory.get('recommended_actions', [])
                if recommendations:
                    base_reply += f" 建议：{recommendations[0]}"

        return base_reply
