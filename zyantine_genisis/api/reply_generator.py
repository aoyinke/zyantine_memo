"""
回复生成器 - 智能回复生成
"""
from typing import Dict, List, Optional, Any, Protocol, Tuple, Union
from datetime import datetime
import random
import traceback
from dataclasses import dataclass

from .openai_service import OpenAIService
from .llm_service import BaseLLMService
from .prompt_engine import PromptEngine
from .fallback_strategy import FallbackStrategy
# 修改导入路径
from cognition.core_identity import CoreIdentity
from utils.logger import SystemLogger
from utils.metrics import MetricsCollector


class LLMServiceInterface(Protocol):
    """LLM服务接口协议"""

    def generate_reply(self,
                       system_prompt: str,
                       user_input: str,
                       conversation_history: Optional[List[Dict]] = None,
                       max_tokens: int = 500,
                       temperature: float = 1.0,
                       stream: bool = False) -> Tuple[Optional[str], Optional[Dict]]:
        """生成回复"""
        ...

    def is_available(self) -> bool:
        """检查服务是否可用"""
        ...

    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        ...


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
                 api_service: Optional[LLMServiceInterface],
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

    def generate_reply(self, cognitive_result: Dict = None, stream: bool = False, **kwargs) -> Union[Tuple[str, Optional[str]], Any]:
        """
        生成回复

        Args:
            cognitive_result: 认知流程结果（新版本）
            stream: 是否流式输出
            **kwargs: 其他参数，向后兼容旧版本

        Returns:
            非流式模式：返回(回复文本, 情绪信息)的元组
            流式模式：返回流式生成器
        """
        if cognitive_result:
            # 新版本调用方式
            return self.generate_from_cognitive_flow(cognitive_result, stream=stream)
        else:
            # 旧版本兼容
            return self._generate_with_legacy_api(stream=stream, **kwargs)

    def _generate_with_legacy_api(self, stream: bool = False, **kwargs) -> Union[Tuple[str, Optional[str]], Any]:
        """旧版本生成方式，保持兼容"""
        start_time = datetime.now()

        try:
            # 使用工厂方法创建上下文
            context = self._create_generation_context(**kwargs)

            self.logger.info(f"开始生成回复，用户输入: {context.user_input[:50]}...")

            # 检查API是否可用
            if self.api and self.api.is_available():
                result = self._generate_with_api(context, stream=stream)
            else:
                self.logger.warning("API不可用，使用降级策略")
                result = self._generate_with_fallback(context)

            if stream and hasattr(result, '__iter__') and not isinstance(result, str):
                # 流式模式：返回生成器，不记录日志直到生成完成
                return result
            else:
                # 非流式模式：记录生成日志
                self._log_generation(context, result, start_time)
                # 获取情绪信息
                emotion = "neutral"
                if hasattr(context, 'metadata') and context.metadata:
                    api_metadata = context.metadata.get('api_metadata')
                    if api_metadata:
                        emotion = api_metadata.get('emotion', 'neutral')
                self.logger.info(f"回复生成成功，长度: {len(result)}, emotion: {emotion}")
                return result, emotion

        except Exception as e:
            self._log_error(context if 'context' in locals() else None, e, start_time)
            return self._generate_emergency_reply(context if 'context' in locals() else None), "neutral"

    def generate_from_cognitive_flow(self, cognitive_result: Dict, stream: bool = False) -> Union[Tuple[str, Optional[str]], Any]:
        """直接从认知流程结果生成回复"""
        start_time = datetime.now()

        try:
            # 提取必要信息
            action_plan = cognitive_result.get("final_action_plan", {})
            memory_context = cognitive_result.get("memory_context", {})
            user_input = cognitive_result.get("user_input", "")
            conversation_history = cognitive_result.get("conversation_history", [])

            # 获取核心身份
            core_identity = self.core_identity
            if memory_context:
                # 如果有共鸣记忆，提取关键信息
                resonant_memory = memory_context.get("resonant_memory")
                if resonant_memory:
                    simplified_memory_context = {
                        "resonant_memory": {
                            "triggered_memory": resonant_memory.get("triggered_memory", "")[:200],
                            "relevance_score": resonant_memory.get("relevance_score", 0.0),
                            "emotional_intensity": resonant_memory.get("emotional_intensity", 0.5),
                            "tags": resonant_memory.get("tags", [])
                        }
                    }
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
                result = self._generate_with_api(context, stream=stream)
            else:
                result = self._generate_with_fallback(context)

            if stream and hasattr(result, '__iter__') and not isinstance(result, str):
                # 流式模式：返回生成器，不记录日志直到生成完成
                return result
            else:
                # 非流式模式：记录生成日志
                self._log_generation(context, result, start_time)
                # 获取情绪信息
                emotion = "neutral"
                if hasattr(context, 'metadata') and context.metadata:
                    api_metadata = context.metadata.get('api_metadata')
                    if api_metadata:
                        emotion = api_metadata.get('emotion', 'neutral')
                self.logger.info(f"回复生成成功，长度: {len(result)}, emotion: {emotion}")
                return result, emotion

        except Exception as e:
            self.logger.error(f"认知流程生成失败: {e}")
            return self._generate_emergency_reply(None), "neutral"

    def _build_system_prompt(self, context: GenerationContext) -> str:
        """构建系统提示词"""
        # 尝试从缓存获取
        cache_key = self._generate_prompt_cache_key(context)

        if cache_key in self.prompt_cache:
            self.logger.debug("使用缓存的提示词")
            return self.prompt_cache[cache_key]

        # 使用提示词引擎构建 - 新增conversation_history参数以保持话题连贯性
        prompt = self.prompt_engine.build_prompt(
            action_plan=context.action_plan,
            growth_result=context.growth_result,
            context_analysis=context.context_analysis,
            core_identity=context.core_identity,
            current_vectors=context.current_vectors,
            memory_context=context.memory_context,
            conversation_history=context.conversation_history  # 新增：传递对话历史
        )

        # 缓存提示词
        self.prompt_cache[cache_key] = prompt

        # 保持缓存大小
        if len(self.prompt_cache) > 100:
            # 移除最旧的条目
            oldest_key = next(iter(self.prompt_cache))
            del self.prompt_cache[oldest_key]

        return prompt

    def _generate_with_api(self, context: GenerationContext, stream: bool = False) -> Union[str, Any]:
        """
        使用API生成回复
        """
        try:
            # 增强API服务可用性检查
            if not self.api or not hasattr(self.api, 'generate_reply'):
                self.logger.error("API服务不可用或缺少generate_reply方法")
                return self._generate_with_fallback(context)

            # 检查API服务是否可用
            if hasattr(self.api, 'is_available') and not self.api.is_available():
                self.logger.error("API服务不可用")
                return self._generate_with_fallback(context)

            # 构建系统提示词
            system_prompt = self._build_system_prompt(context)
            if not system_prompt or not isinstance(system_prompt, str):
                self.logger.error("系统提示词构建失败")
                return self._generate_with_fallback(context)

            # 确定最大token数和温度
            max_tokens = self._determine_max_tokens(context.context_analysis)
            temperature = self._determine_temperature(context.current_vectors)

            self.logger.info(f"开始API调用，用户输入: {context.user_input[:50]}...")
            self.logger.debug(f"API调用参数: 模型={self.api.model if hasattr(self.api, 'model') else '未知'}, max_tokens={max_tokens}, temperature={temperature}, 流模式={stream}")
            
            # 调用API
            result = self.api.generate_reply(
                system_prompt=system_prompt,
                user_input=context.user_input,
                conversation_history=context.conversation_history,
                max_tokens=max_tokens,
                temperature=temperature,
                stream=stream
            )

            # 处理返回结果
            if stream:
                # 流式模式：返回生成器
                if not result or not isinstance(result, tuple) or len(result) != 2:
                    self.logger.error("API返回的流式结果格式错误")
                    return self._generate_with_fallback(context)
                
                reply_generator, metadata = result
                # 存储API使用元数据到上下文
                if hasattr(context, 'metadata'):
                    context.metadata = context.metadata or {}
                    context.metadata.update({
                        'api_used': True,
                        'api_metadata': metadata
                    })
                self.logger.debug("API流式调用成功，开始生成回复")
                return reply_generator
            else:
                # 非流式模式：返回完整字符串
                if not result or not isinstance(result, tuple) or len(result) != 2:
                    self.logger.error("API返回的非流式结果格式错误")
                    return self._generate_with_fallback(context)
                
                reply, metadata = result
                if reply and isinstance(reply, str):
                    # 存储API使用元数据到上下文
                    if hasattr(context, 'metadata'):
                        context.metadata = context.metadata or {}
                        context.metadata.update({
                            'api_used': True,
                            'api_metadata': metadata
                        })

                    self.logger.debug(
                        f"API生成成功，tokens: {metadata.get('tokens_used', 0) if metadata else 0}, latency: {metadata.get('latency', 0):.2f}s, emotion: {metadata.get('emotion', 'neutral') if metadata else 'neutral'}")
                    return reply
                else:
                    self.logger.warning("API返回空回复或格式错误，使用降级策略")
                    # 如果API返回空或格式错误，使用降级策略
                    return self._generate_with_fallback(context)

        except Exception as e:
            # 详细记录API调用失败信息
            import traceback
            error_msg = f"API生成失败: {str(e)}"
            self.logger.error(error_msg)
            self.logger.debug(f"API调用详细错误堆栈:\n{traceback.format_exc()}")
            
            # 记录API故障诊断信息
            if hasattr(self.api, 'get_statistics'):
                try:
                    api_stats = self.api.get_statistics()
                    self.logger.debug(f"API服务统计信息: {api_stats}")
                except Exception as stats_error:
                    self.logger.error(f"获取API统计信息失败: {str(stats_error)}")
            
            # 检查是否是特定类型的错误
            error_str = str(e).lower()
            if "invalid_api_key" in error_str or "authentication" in error_str:
                self.logger.error("API认证失败，请检查API密钥配置")
            elif "model_not_found" in error_str or "invalid_model" in error_str:
                self.logger.error(f"模型 {self.api.model if hasattr(self.api, 'model') else '未知'} 不存在或不可用")
            elif "connection" in error_str or "timeout" in error_str:
                self.logger.error("网络连接失败，请检查网络环境或API服务状态")
            elif "rate_limit" in error_str:
                self.logger.error("API请求频率过高，请检查配额或调整请求频率")
            elif "api_service_provider" in error_str:
                self.logger.error("API服务提供者不可用")
            
            # API调用失败时，使用降级策略
            self.logger.info("API调用失败，切换到降级策略")
            return self._generate_with_fallback(context)

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

    def _create_generation_context(self, **kwargs) -> GenerationContext:
        """创建生成上下文，确保所有必需参数都存在"""
        # 确保核心身份存在
        if 'core_identity' not in kwargs:
            kwargs['core_identity'] = self.core_identity

        # 为缺失的必需参数提供默认值
        defaults = {
            'user_input': kwargs.get('user_input', ''),
            'action_plan': kwargs.get('action_plan', {}),
            'growth_result': kwargs.get('growth_result', {}),
            'context_analysis': kwargs.get('context_analysis', {}),
            'conversation_history': kwargs.get('conversation_history', []),
            'current_vectors': kwargs.get('current_vectors', {}),
            'memory_context': kwargs.get('memory_context', None),
            'metadata': kwargs.get('metadata', None),
            'cognitive_flow_id': kwargs.get('cognitive_flow_id', None)
        }

        # 过滤掉 GenerationContext 不接受的参数
        valid_keys = set(['user_input', 'action_plan', 'growth_result', 'context_analysis', 
                         'conversation_history', 'core_identity', 'current_vectors', 
                         'memory_context', 'metadata', 'cognitive_flow_id', 'timestamp'])
        filtered_kwargs = {k: v for k, v in kwargs.items() if k in valid_keys}

        # 更新 filtered_kwargs 中的缺失参数
        for key, value in defaults.items():
            if key not in filtered_kwargs:
                filtered_kwargs[key] = value

        return GenerationContext(**filtered_kwargs)

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

        # 使用更高的基础温度以减少AI味
        base_temp = 0.8

        # TR影响：高兴奋需要更多创造性
        tr_adjustment = (tr - 0.5) * 0.3  # ±0.15

        # CS影响：高亲密感可以更自然
        cs_adjustment = (cs - 0.5) * 0.25  # ±0.125

        # SA影响：高压需要更稳定
        sa_adjustment = -sa * 0.2  # 最高降低0.2

        temperature = base_temp + tr_adjustment + cs_adjustment + sa_adjustment

        # 确保温度在合理范围内，提高下限以保持自然性
        temperature = max(0.5, min(1.0, temperature))

        # 特殊情况的微调
        if sa > 0.8 and tr > 0.7:
            # 高压高兴奋：稍微降低温度避免过激
            temperature *= 0.95
        elif sa < 0.3 and cs > 0.7:
            # 低压力高亲密：提高温度增加自然性
            temperature *= 1.05
        elif tr < 0.3 and cs < 0.3:
            # 低兴奋低亲密：提高温度避免过于机械
            temperature *= 1.1

        # 四舍五入到小数点后一位
        temperature = round(temperature, 1)

        self.logger.debug(f"计算温度: TR={tr:.2f}, CS={cs:.2f}, SA={sa:.2f} => temp={temperature}")

        return temperature

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

        # 记录指标 - 修改这里
        if self.metrics:
            # 使用正确的指标记录方法
            labels = {
                "mask": log_entry["mask"],
                "strategy": log_entry["strategy"],
                "api_used": str(log_entry["used_api"])
            }

            # 记录回复生成指标
            self.metrics.increment_counter("reply.generation.total")

            if log_entry["used_api"]:
                self.metrics.increment_counter("reply.generation.api")
            else:
                self.metrics.increment_counter("reply.generation.fallback")

            # 记录回复长度
            self.metrics.record_histogram("reply.length", len(reply), labels=labels)

            # 记录生成时间
            self.metrics.record_histogram("reply.generation.time",
                                          log_entry["generation_time"],
                                          labels=labels)

    def _log_error(self, context: GenerationContext, error: Exception, start_time: datetime):
        """记录错误日志"""
        # 确保上下文安全访问
        user_input = context.user_input[:100] if context and hasattr(context, 'user_input') else "无法获取用户输入"
        action_plan = str(context.action_plan)[:200] if context and hasattr(context, 'action_plan') else "无法获取行动方案"
        current_vectors = context.current_vectors if context and hasattr(context, 'current_vectors') else {}
        
        error_entry = {
            "timestamp": datetime.now().isoformat(),
            "user_input": user_input,
            "error_type": type(error).__name__,
            "error_message": str(error),
            "action_plan": action_plan,
            "vectors": current_vectors,
            "generation_time": (datetime.now() - start_time).total_seconds()
        }

        self.error_log.append(error_entry)

        # 保持日志大小
        if len(self.error_log) > 100:
            self.error_log = self.error_log[-100:]

        # 记录指标
        if self.metrics:
            labels = {
                "success": "False",
                "used_api": "False"
            }
            
            # 记录回复生成失败
            self.metrics.increment_counter("reply.generation.count", labels=labels)
            self.metrics.increment_counter("reply.generation.failure", labels=labels)
            
            # 记录生成时间
            generation_time = (datetime.now() - start_time).total_seconds()
            self.metrics.record_histogram("reply.generation.time", generation_time, labels=labels)

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
        self.prompt_engine.clear_cache()
        self.logger.info("已清空提示词缓存")

class TemplateReplyGenerator:
    """模板回复生成器（当API不可用时使用），适配新的记忆系统"""

    def __init__(self):
        self.templates = self._load_templates()

    def _load_templates(self) -> Dict:
        """加载回复模板"""
        return {
            "哲思伙伴": [
                "让我思考一下这个问题...{strategy}。你怎么看？",
                "从多个角度分析，我认为：{strategy}。不过我也在考虑...",
                "这个问题很有意思，它让我想到了：{strategy}。你是否有类似的思考？",
                "我一方面觉得{strategy}，但另一方面也考虑到可能存在的局限性...",
                "根据我的理解，{strategy}。当然，这只是我当前的思考，可能还有其他视角。"
            ],
            "创意同行": [
                "哇，这个想法激发了我的灵感！{strategy}",
                "我突然联想到：{strategy}，你觉得这个方向怎么样？",
                "让我脑洞大开一下：{strategy}。这个创意如何？",
                "这个问题可以从这样的角度切入：{strategy}，不过还有更有趣的可能性...",
                "我正在思考一个新的角度：{strategy}，你觉得可行吗？"
            ],
            "务实挚友": [
                "我理解你的感受，{strategy}",
                "根据我的经验，{strategy}。不过具体情况可能需要调整...",
                "让我们一起想想办法：{strategy}。你觉得这个方案可行吗？",
                "记得你之前提到过{strategy}，结合现在的情况，我认为...",
                "我正在考虑各种可能性，其中{strategy}看起来是最实际的选择。"
            ],
            "幽默知己": [
                "哈哈，这个问题很有意思！让我想想...{strategy}",
                "我有个有趣的想法：{strategy}，是不是有点意思？",
                "别担心，{strategy}，问题总能解决的！",
                "这让我想起一件趣事：{strategy}，是不是和你的情况有点像？",
                "我正在思考如何用轻松的方式解决这个问题：{strategy}，你觉得怎么样？"
            ]
        }

    def generate(self, mask: str, strategy: str, growth_result: Dict = None,
                 memory_context: Optional[Dict] = None) -> str:
        """使用模板生成回复，适配新的记忆系统"""
        template_list = self.templates.get(mask, self.templates["哲思伙伴"])
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
            relevance = resonant_memory.get("relevance_score", 0.0)

            # 只有相关性分数高于0.7才使用记忆
            if memory_info and relevance > 0.7:
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
