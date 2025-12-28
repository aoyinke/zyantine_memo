"""
API服务提供者 - 管理所有API相关服务
"""
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import asyncio
from datetime import datetime, timedelta
import time

from .openai_service import OpenAIService
from .reply_generator import APIBasedReplyGenerator, TemplateReplyGenerator
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


class ServiceType(Enum):
    """服务类型"""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    LOCAL = "local"
    HYBRID = "hybrid"


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

    def __init__(self, config, core_identity=None):  # 新增core_identity参数
        self.config = config
        self.core_identity = core_identity  # 存储核心身份
        self.logger = SystemLogger().get_logger("api_service_provider")
        self.metrics = MetricsCollector("api_service")

        # 服务注册表
        self.services: Dict[str, Any] = {}
        self.service_metrics: Dict[str, ServiceMetrics] = {}
        self.active_service: Optional[str] = None

        # 初始化服务
        self._initialize_services()

        # 初始化组件
        self._initialize_components()

        self.logger.info("API服务提供者初始化完成")

    def _initialize_services(self):
        """初始化API服务"""
        api_config = self.config.api
        # 安全打印配置，不暴露敏感信息
        self.logger.info(f"初始化API服务，模型: {api_config.chat_model}, URL: {api_config.base_url[:30]}...")

        if api_config.enabled and api_config.api_key:
            # 创建OpenAI服务
            openai_service = OpenAIService(
                api_key=api_config.api_key,
                base_url=api_config.base_url,
                model=api_config.chat_model,
                timeout=api_config.timeout,
                max_retries=api_config.max_retries
            )

            # 测试连接
            success, message = openai_service.test_connection()
            if success:
                self.services["openai"] = openai_service
                self.service_metrics["openai"] = ServiceMetrics()
                self.active_service = "openai"
                self.logger.info(f"OpenAI服务初始化成功，模型: {api_config.chat_model}")
            else:
                self.logger.warning(f"OpenAI服务连接测试失败: {message}，将使用降级策略")
                self._setup_fallback_strategy()
        else:
            self.logger.warning("API服务未启用或未配置API密钥，使用本地模式")
            self._setup_local_mode()

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
            core_identity=self.core_identity  # 传递核心身份
        )

    def _setup_fallback_strategy(self):
        """设置降级策略"""
        self.logger.info("设置降级策略")

        # 创建模板生成器作为备用
        template_generator = TemplateReplyGenerator()

        # 创建降级策略
        self.fallback_strategy = FallbackStrategy(

        )

        # 创建回复生成器（降级模式）
        self.reply_generator = APIBasedReplyGenerator(
            api_service=None,
            prompt_engine=self.prompt_engine,
            fallback_strategy=self.fallback_strategy,
            metrics_collector=self.metrics
        )

    def _setup_local_mode(self):
        """设置本地模式"""
        self.logger.info("设置本地模式")

        # 创建模板生成器
        template_generator = TemplateReplyGenerator()

        # 创建提示词引擎
        self.prompt_engine = PromptEngine(self.config)

        # 创建回复生成器（本地模式）
        self.reply_generator = APIBasedReplyGenerator(
            api_service=None,
            prompt_engine=self.prompt_engine,
            fallback_strategy=FallbackStrategy(),
            metrics_collector=self.metrics
        )

    def generate_reply(self, **kwargs) -> str:
        """生成回复"""
        start_time = time.time()

        try:
            # 如果kwargs中包含core_identity，使用它
            if 'core_identity' not in kwargs and self.core_identity:
                kwargs['core_identity'] = self.core_identity

            # 生成回复
            reply = self.reply_generator.generate_reply(**kwargs)

            # 记录指标
            latency = time.time() - start_time
            if self.active_service:
                self.service_metrics[self.active_service].record_request(True, latency)

            self.metrics.record_api_call(success=True, latency=latency)

            return reply

        except Exception as e:
            # 记录错误
            latency = time.time() - start_time
            if self.active_service:
                self.service_metrics[self.active_service].record_request(False, latency)

            self.metrics.record_api_call(success=False, latency=latency)
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
                core_identity=self.core_identity  # 添加 core_identity
            )
        else:
            kwargs['core_identity'] = self.core_identity  # 确保传递 core_identity
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

    def switch_service(self, service_name: str) -> bool:
        """切换服务"""
        if service_name in self.services:
            self.active_service = service_name
            self.logger.info(f"切换到服务: {service_name}")
            return True
        else:
            self.logger.error(f"服务不存在: {service_name}")
            return False

    def add_service(self, service_name: str, service: Any) -> bool:
        """添加新服务"""
        if service_name in self.services:
            self.logger.warning(f"服务已存在: {service_name}")
            return False

        self.services[service_name] = service
        self.service_metrics[service_name] = ServiceMetrics()
        self.logger.info(f"添加新服务: {service_name}")
        return True

    def get_service_metrics(self) -> Dict[str, Dict]:
        """获取服务指标"""
        metrics = {}

        for service_name, service_metric in self.service_metrics.items():
            metrics[service_name] = {
                "request_count": service_metric.request_count,
                "success_count": service_metric.success_count,
                "error_count": service_metric.error_count,
                "success_rate": (service_metric.success_count / max(service_metric.request_count, 1)) * 100,
                "avg_latency": service_metric.avg_latency,
                "consecutive_errors": service_metric.consecutive_errors,
                "last_request_time": service_metric.last_request_time.isoformat() if service_metric.last_request_time else None
            }

        return metrics

    def get_overall_status(self) -> Dict[str, Any]:
        """获取整体状态"""
        api_metrics = self.get_service_metrics()

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

        # 关闭所有服务
        for service_name, service in self.services.items():
            if hasattr(service, 'shutdown'):
                try:
                    service.shutdown()
                    self.logger.info(f"已关闭服务: {service_name}")
                except Exception as e:
                    self.logger.error(f"关闭服务 {service_name} 失败: {e}")

        self.logger.info("API服务提供者已关闭")