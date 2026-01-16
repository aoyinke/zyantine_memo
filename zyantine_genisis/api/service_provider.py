"""
API服务提供者 - 管理所有API相关服务
"""
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import time
import hashlib
import threading
import traceback
from collections import defaultdict

from .llm_service_factory import LLMServiceFactory
from .reply_generator import APIBasedReplyGenerator
from .prompt_engine import PromptEngine
from .fallback_strategy import FallbackStrategy
from config.config_manager import ConfigManager
from utils.logger import SystemLogger
from utils.metrics import MetricsCollector
from cognition.core_identity import CoreIdentity


class ServiceStatus(Enum):
    """服务状态"""
    ACTIVE = "active"
    DEGRADED = "degraded"
    FAILED = "failed"
    OFFLINE = "offline"


@dataclass
class ServiceMetrics:
    """服务指标"""
    request_count: int = 0
    success_count: int = 0
    error_count: int = 0
    total_latency: float = 0.0
    avg_latency: float = 0.0
    last_request_time: Optional[datetime] = None
    last_error_time: Optional[datetime] = None
    consecutive_errors: int = 0

    def record_request(self, success: bool, latency: float):
        """记录请求"""
        self.request_count += 1
        self.total_latency += latency

        if success:
            self.success_count += 1
            self.consecutive_errors = 0
        else:
            self.error_count += 1
            self.consecutive_errors += 1
            self.last_error_time = datetime.now()

        self.avg_latency = self.total_latency / self.request_count
        self.last_request_time = datetime.now()


class APIServiceProvider:
    """API服务提供者 - 管理所有API服务"""

    def __init__(self, config, core_identity=None):
        self.config = config
        self.core_identity = core_identity
        self.logger = SystemLogger().get_logger("api_service_provider")
        self.metrics = MetricsCollector("api_service")

        # 服务注册表
        self.services: Dict[str, Any] = {}
        self.service_metrics: Dict[str, ServiceMetrics] = {}
        self.active_service: Optional[str] = None

        # 请求缓存
        self.request_cache: Dict[str, Tuple[str, datetime]] = {}
        self.cache_ttl = timedelta(minutes=10)  # 缓存有效期10分钟
        self.max_cache_size = 1000
        self.cache_lock = threading.RLock()

        # 初始化服务和组件
        self._initialize_services()
        self._initialize_components()

        self.logger.info("API服务提供者初始化完成")

    def _initialize_services(self):
        """
        初始化API服务
        """
        api_config = self.config.api

        # 获取当前选择的提供商
        provider = api_config.provider or "openai"
        self.logger.info(f"=== 开始初始化API服务，提供商: {provider} ===")

        # 检查提供商配置
        providers_config = api_config.providers or {}
        provider_config = providers_config.get(provider, {})
        self.logger.debug(f"原始提供商配置: {provider_config}")

        # 如果提供商配置为空或API密钥为空，使用旧的配置方式（向后兼容）
        if not provider_config or not provider_config.get("api_key") or not provider_config.get("api_key").strip():
            self.logger.info("提供商配置不完整，尝试使用旧的配置方式")
            if api_config.enabled and api_config.api_key and api_config.api_key.strip():
                # 判断是否需要使用max_completion_tokens（DeepSeek需要）
                use_max_completion_tokens = (provider == "deepseek")

                provider_config = {
                    "enabled": True,
                    "api_key": api_config.api_key,
                    "base_url": api_config.base_url,
                    "chat_model": api_config.chat_model,
                    "timeout": api_config.timeout,
                    "max_retries": api_config.max_retries,
                    "use_max_completion_tokens": use_max_completion_tokens
                }
                self.logger.info(f"使用主配置初始化{provider}服务，模型: {provider_config.get('chat_model')}")
            else:
                self.logger.warning("API服务未启用或未配置有效API密钥，使用本地模式")
                self.logger.debug(f"API配置详情: enabled={api_config.enabled}, api_key={True if api_config.api_key and api_config.api_key.strip() else False}")
                self._setup_local_mode()
                return
        else:
            # 确保use_max_completion_tokens字段被设置
            if "use_max_completion_tokens" not in provider_config:
                provider_config["use_max_completion_tokens"] = (provider == "deepseek")
                self.logger.debug(f"自动设置use_max_completion_tokens为: {provider_config['use_max_completion_tokens']}")
            self.logger.info(f"使用提供商配置初始化{provider}服务，模型: {provider_config.get('chat_model')}")

        # 检查提供商是否启用
        if not provider_config.get("enabled", False):
            self.logger.warning(f"提供商 {provider} 未启用（enabled={provider_config.get('enabled')}），使用本地模式")
            self._setup_local_mode()
            return

        # 验证必要的配置参数
        required_fields = ["api_key", "chat_model"]
        missing_fields = []
        for field in required_fields:
            if not provider_config.get(field) or not provider_config.get(field).strip():
                missing_fields.append(field)
        
        if missing_fields:
            self.logger.error(f"提供商配置缺少必要字段: {', '.join(missing_fields)}")
            self._setup_local_mode()
            return

        # 使用工厂创建服务
        self.logger.debug(f"使用以下配置创建服务: {provider_config}")
        try:
            service = LLMServiceFactory.get_or_create_service(provider, provider_config)
        except Exception as e:
            self.logger.error(f"创建{provider}服务时发生异常: {str(e)}")
            self.logger.debug(f"异常详情: {traceback.format_exc()}")
            self._setup_local_mode()
            return

        if not service:
            self.logger.error(f"创建{provider}服务失败，可能是配置错误或提供商不支持")
            self.logger.error(f"请检查配置文件中的API密钥、模型名称和基础URL是否正确")
            self._setup_local_mode()
            return

        # 增强对API服务客户端的检查
        try:
            if not hasattr(service, 'client') or service.client is None:
                self.logger.error(f"{provider}服务客户端初始化失败，可能是API密钥或配置错误")
                self.logger.debug(f"服务对象: {service}, 客户端状态: {hasattr(service, 'client')}, 客户端值: {service.client if hasattr(service, 'client') else '未定义'}")
                self._setup_local_mode()
                return
        except Exception as e:
            self.logger.error(f"检查服务客户端时发生异常: {str(e)}")
            self.logger.debug(f"异常详情: {traceback.format_exc()}")
            self._setup_local_mode()
            return
        
        # 测试连接
        self.logger.info(f"正在测试{provider}服务连接...")
        try:
            success, message = service.test_connection()
            self.logger.debug(f"连接测试结果: success={success}, message={message}")
            if success:
                self.services[provider] = service
                self.service_metrics[provider] = ServiceMetrics()
                self.active_service = provider
                self.logger.info(f"{provider}服务初始化成功，模型: {provider_config.get('chat_model')}")
                self.logger.info(f"=== API服务初始化完成，活跃服务: {self.active_service} ===")
            else:
                # 记录详细的连接失败信息
                self.logger.warning(f"{provider}服务连接测试失败: {message}")
                self.logger.warning(f"错误类型可能为: API密钥无效、模型不存在、网络连接问题或API服务异常")
                
                # 检查是否是模型名称问题
                if "model_not_found" in message.lower() or "invalid_model" in message.lower():
                    self.logger.warning(f"请检查模型名称 '{provider_config.get('chat_model')}' 是否有效，建议使用更常见的模型如 'gpt-4o-mini' 或 'gpt-3.5-turbo'")
                
                # 检查是否是认证问题
                elif "invalid_api_key" in message.lower() or "authentication" in message.lower():
                    self.logger.warning("请检查API密钥是否正确配置")
                elif "connection" in message.lower() or "timeout" in message.lower():
                    self.logger.warning("网络连接失败，请检查网络环境或API服务状态")
                elif "rate_limit" in message.lower():
                    self.logger.warning("API请求频率过高，请检查配额或调整请求频率")
                
                # 即使测试失败，也尝试使用该服务（可能是临时网络问题）
                self.logger.info(f"仍尝试使用{provider}服务，后续请求可能会失败")
                self.services[provider] = service
                self.service_metrics[provider] = ServiceMetrics()
                self.active_service = provider
                self.logger.info(f"=== API服务初始化完成，活跃服务: {self.active_service}（连接测试失败） ===")
        except Exception as e:
            self.logger.error(f"测试{provider}服务连接时发生异常: {str(e)}")
            self.logger.debug(f"异常详情: {traceback.format_exc()}")
            # 即使连接测试失败，也尝试使用该服务
            self.services[provider] = service
            self.service_metrics[provider] = ServiceMetrics()
            self.active_service = provider
            self.logger.info(f"=== API服务初始化完成，活跃服务: {self.active_service}（连接测试异常） ===")

    def _initialize_components(self):
        """初始化API相关组件"""
        # 创建提示词引擎
        self.prompt_engine = PromptEngine(self.config)

        # 创建降级策略，传递核心身份
        self.fallback_strategy = FallbackStrategy(core_identity=self.core_identity)

        # 创建回复生成器
        self.reply_generator = APIBasedReplyGenerator(
            api_service=self.services.get(self.active_service),
            prompt_engine=self.prompt_engine,
            fallback_strategy=self.fallback_strategy,
            metrics_collector=self.metrics,
            core_identity=self.core_identity
        )

    def _setup_local_mode(self):
        """设置本地模式"""
        self.logger.info("设置本地模式")

        # 创建提示词引擎（如果还没有创建）
        if not hasattr(self, 'prompt_engine'):
            self.prompt_engine = PromptEngine(self.config)

        # 创建回复生成器（本地模式）
        self.reply_generator = APIBasedReplyGenerator(
            api_service=None,
            prompt_engine=self.prompt_engine,
            fallback_strategy=FallbackStrategy(core_identity=self.core_identity),
            metrics_collector=self.metrics
        )

    def generate_reply(self, **kwargs) -> str:
        """生成回复"""
        start_time = time.time()

        try:
            # 生成缓存键
            cache_key = self._generate_cache_key(**kwargs)

            # 检查缓存
            cached_reply = self._get_cached_reply(cache_key)
            if cached_reply:
                # 使用正确的指标收集方法
                self.metrics.increment_counter("api.requests.total", labels={"success": "true", "cached": "true", "endpoint": "generate_reply", "method": "POST"})
                self.metrics.record_histogram("api.request.duration", time.time() - start_time, labels={"success": "true", "cached": "true", "endpoint": "generate_reply", "method": "POST"})
                self.logger.debug(f"从缓存返回回复: {cache_key[:16]}...")
                return cached_reply

            # 如果kwargs中包含core_identity，使用它
            if 'core_identity' not in kwargs and self.core_identity:
                kwargs['core_identity'] = self.core_identity

            # 生成回复
            reply = self.reply_generator.generate_reply(**kwargs)

            # 记录指标
            latency = time.time() - start_time
            if self.active_service:
                self.service_metrics[self.active_service].record_request(True, latency)

            # 使用正确的指标收集方法
            self.metrics.increment_counter("api.requests.total", labels={"success": "true", "cached": "false", "endpoint": "generate_reply", "method": "POST"})
            self.metrics.record_histogram("api.request.duration", latency, labels={"success": "true", "cached": "false", "endpoint": "generate_reply", "method": "POST"})

            # 缓存结果
            self._cache_reply(cache_key, reply)

            return reply

        except Exception as e:
            # 记录错误
            latency = time.time() - start_time
            if self.active_service:
                self.service_metrics[self.active_service].record_request(False, latency)

            # 使用正确的指标收集方法
            self.metrics.increment_counter("api.requests.total", labels={"success": "false", "cached": "false", "endpoint": "generate_reply", "method": "POST"})
            self.metrics.record_histogram("api.request.duration", latency, labels={"success": "false", "cached": "false", "endpoint": "generate_reply", "method": "POST"})
            self.logger.error(f"生成回复失败: {e}")

            # 尝试降级
            try:
                # 确保传递必要的参数
                fallback_kwargs = {k: v for k, v in kwargs.items()
                                 if k in ['action_plan', 'growth_result', 'memory_context',
                                         'user_input', 'conversation_history']}
                fallback_reply = self.fallback_strategy.generate_fallback_reply(**fallback_kwargs)
                return fallback_reply
            except Exception as fallback_error:
                self.logger.error(f"降级策略也失败: {fallback_error}")
                return self._generate_emergency_response()

    def generate_reply_with_memory(self, context=None, strategy=None, template=None, mask_type=None, **kwargs):
        """适配原有的回复生成接口"""
        if context:
            return self.generate_reply(
                user_input=context.user_input,
                action_plan={
                    "chosen_mask": mask_type or "长期搭档",
                    "primary_strategy": strategy or ""
                },
                growth_result=getattr(context, 'growth_result', {}),
                context_analysis=getattr(context, 'context_info', {}),
                conversation_history=getattr(context, 'conversation_history', []),
                current_vectors=getattr(context, 'desire_vectors', {}),
                memory_context={
                    "retrieved_memories": getattr(context, 'retrieved_memories', []),
                    "resonant_memory": getattr(context, 'resonant_memory', None)
                },
                core_identity=self.core_identity
            )
        else:
            kwargs['core_identity'] = self.core_identity
            return self.generate_reply(**kwargs)

    def test_service_health(self) -> Dict[str, Any]:
        """测试服务健康状态"""
        results = {}

        for service_name, service in self.services.items():
            try:
                if hasattr(service, 'test_connection'):
                    success, message = service.test_connection()
                    results[service_name] = {
                        "status": "healthy" if success else "unhealthy",
                        "message": message
                    }
            except Exception as e:
                results[service_name] = {
                    "status": "error",
                    "message": str(e)
                }

        # 检查降级策略
        results["fallback_strategy"] = {
            "status": "ready",
            "message": "降级策略已就绪"
        }

        return results

    def get_overall_status(self) -> Dict[str, Any]:
        """获取整体状态"""
        api_metrics = {}
        for service_name, service_metric in self.service_metrics.items():
            api_metrics[service_name] = {
                "request_count": service_metric.request_count,
                "success_count": service_metric.success_count,
                "error_count": service_metric.error_count,
                "success_rate": (service_metric.success_count / max(service_metric.request_count, 1)) * 100,
                "avg_latency": service_metric.avg_latency,
                "consecutive_errors": service_metric.consecutive_errors,
                "last_request_time": service_metric.last_request_time.isoformat() if service_metric.last_request_time else None
            }

        return {
            "active_service": self.active_service,
            "available_services": list(self.services.keys()),
            "fallback_strategy_ready": self.fallback_strategy is not None,
            "metrics": api_metrics,
            "total_requests": sum(m.request_count for m in self.service_metrics.values()),
            "total_errors": sum(m.error_count for m in self.service_metrics.values()),
            "overall_success_rate": (
                    sum(m.success_count for m in self.service_metrics.values()) /
                    max(sum(m.request_count for m in self.service_metrics.values()), 1) * 100
            )
        }

    def _generate_emergency_response(self) -> str:
        """生成紧急响应"""
        emergency_responses = [
            "我的思考过程出现了一些混乱，能请你再问一次吗？",
            "刚才的思考链路好像打了个结，我们重新开始吧。",
            "意识流有点波动，让我重新整理一下思绪。"
        ]

        import random
        return random.choice(emergency_responses)

    def shutdown(self):
        """关闭服务"""
        self.logger.info("正在关闭API服务提供者...")

        # 清空工厂缓存
        LLMServiceFactory.clear_cache()

        # 关闭所有服务
        for service_name, service in self.services.items():
            if hasattr(service, 'shutdown'):
                try:
                    service.shutdown()
                    self.logger.info(f"已关闭服务: {service_name}")
                except Exception as e:
                    self.logger.error(f"关闭服务 {service_name} 失败: {e}")

        self.logger.info("API服务提供者已关闭")

    def _generate_cache_key(self, **kwargs) -> str:
        """生成缓存键"""
        key_data = {
            "user_input": kwargs.get("user_input", ""),
            "action_plan": kwargs.get("action_plan", {}),
            "conversation_history_length": len(kwargs.get("conversation_history", [])),
            "current_vectors": kwargs.get("current_vectors", {}),
            "mask": kwargs.get("action_plan", {}).get("chosen_mask", "")
        }
        key_str = str(key_data)
        return hashlib.md5(key_str.encode()).hexdigest()

    def _get_cached_reply(self, cache_key: str) -> Optional[str]:
        """从缓存获取回复"""
        with self.cache_lock:
            if cache_key in self.request_cache:
                reply, timestamp = self.request_cache[cache_key]
                if datetime.now() - timestamp < self.cache_ttl:
                    return reply
                else:
                    del self.request_cache[cache_key]
        return None

    def _cache_reply(self, cache_key: str, reply: str):
        """缓存回复"""
        with self.cache_lock:
            if len(self.request_cache) >= self.max_cache_size:
                oldest_key = min(self.request_cache.items(),
                                 key=lambda x: x[1][1])[0]
                del self.request_cache[oldest_key]
            self.request_cache[cache_key] = (reply, datetime.now())

    def clear_cache(self):
        """清空缓存"""
        with self.cache_lock:
            self.request_cache.clear()
        self.logger.info("请求缓存已清空")
