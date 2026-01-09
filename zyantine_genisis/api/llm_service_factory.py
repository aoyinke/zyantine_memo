"""
LLM服务工厂 - 根据配置创建不同的LLM服务实例
"""
from typing import Optional, Dict, Any, Tuple
from .llm_service import BaseLLMService, OpenAICompatibleService
from .llm_provider import LLMProvider, LLMModelConfig, LLMProviderPresets
from utils.logger import SystemLogger


class LLMServiceFactory:
    """LLM服务工厂类"""

    _instances: Dict[str, BaseLLMService] = {}
    _logger = SystemLogger().get_logger("llm_service_factory")

    @classmethod
    def create_service(cls, provider: str, config: Dict[str, Any]) -> Optional[BaseLLMService]:
        """
        创建LLM服务实例

        Args:
            provider: 提供商名称 (openai, deepseek, anthropic, zhipu, moonshot, alibaba, baidu)
            config: 配置字典

        Returns:
            LLM服务实例
        """
        try:
            # 转换为枚举
            provider_enum = LLMProvider(provider)

            # 创建模型配置
            model_config = LLMModelConfig(
                provider=provider_enum,
                model_name=config.get("chat_model", "gpt-5-nano-2025-08-07"),
                api_key=config.get("api_key", ""),
                base_url=config.get("base_url", ""),
                timeout=config.get("timeout", 30),
                max_retries=config.get("max_retries", 3),
                temperature=config.get("temperature", 1.0),
                max_tokens=config.get("max_tokens", 2000),
                enabled=config.get("enabled", True),
                extra_params=config.get("extra_params", {}),
                use_max_completion_tokens=config.get("use_max_completion_tokens", provider_enum == LLMProvider.DEEPSEEK),
                max_context_tokens=config.get("max_context_tokens", 3000)
            )

            # 根据提供商创建对应的服务
            if provider_enum in [LLMProvider.OPENAI, LLMProvider.DEEPSEEK,
                               LLMProvider.ZHIPU, LLMProvider.MOONSHOT,
                               LLMProvider.ALIBABA]:
                # OpenAI兼容服务
                service = OpenAICompatibleService(model_config)
            else:
                # 其他提供商（暂未实现）
                cls._logger.warning(f"提供商 {provider} 暂未实现，使用OpenAI兼容服务")
                service = OpenAICompatibleService(model_config)

            cls._logger.info(f"成功创建 {provider} 服务实例")
            return service

        except ValueError as e:
            cls._logger.error(f"不支持的提供商: {provider}, 错误: {e}")
            return None
        except Exception as e:
            cls._logger.error(f"创建服务失败: {e}")
            return None

    @classmethod
    def create_from_preset(cls, provider: str, api_key: str, model: Optional[str] = None) -> Optional[BaseLLMService]:
        """
        使用预设配置创建服务

        Args:
            provider: 提供商名称
            api_key: API密钥
            model: 模型名称（可选，使用默认值）

        Returns:
            LLM服务实例
        """
        try:
            provider_enum = LLMProvider(provider)

            # 根据提供商获取预设配置
            if provider_enum == LLMProvider.DEEPSEEK:
                model_config = LLMProviderPresets.get_deepseek_config(api_key)
            elif provider_enum == LLMProvider.OPENAI:
                model_config = LLMProviderPresets.get_openai_config(api_key, model or "gpt-4o-mini")
            elif provider_enum == LLMProvider.ANTHROPIC:
                model_config = LLMProviderPresets.get_anthropic_config(api_key, model or "claude-3-5-sonnet-20241022")
            elif provider_enum == LLMProvider.ZHIPU:
                model_config = LLMProviderPresets.get_zhipu_config(api_key, model or "glm-4")
            elif provider_enum == LLMProvider.MOONSHOT:
                model_config = LLMProviderPresets.get_moonshot_config(api_key, model or "moonshot-v1-8k")
            elif provider_enum == LLMProvider.ALIBABA:
                model_config = LLMProviderPresets.get_alibaba_config(api_key, model or "qwen-turbo")
            elif provider_enum == LLMProvider.BAIDU:
                model_config = LLMProviderPresets.get_baidu_config(api_key, model or "ernie-bot-4")
            else:
                cls._logger.error(f"不支持的提供商: {provider}")
                return None

            # 创建服务
            service = OpenAICompatibleService(model_config)
            cls._logger.info(f"使用预设配置成功创建 {provider} 服务实例")
            return service

        except ValueError as e:
            cls._logger.error(f"不支持的提供商: {provider}, 错误: {e}")
            return None
        except Exception as e:
            cls._logger.error(f"创建服务失败: {e}")
            return None

    @classmethod
    def get_or_create_service(cls, provider: str, config: Dict[str, Any]) -> Optional[BaseLLMService]:
        """
        获取或创建服务实例（单例模式）

        Args:
            provider: 提供商名称
            config: 配置字典

        Returns:
            LLM服务实例
        """
        # 生成唯一键
        key = f"{provider}_{config.get('chat_model', 'default')}"

        # 如果已存在，直接返回
        if key in cls._instances:
            cls._logger.debug(f"使用已存在的 {provider} 服务实例")
            return cls._instances[key]

        # 创建新实例
        service = cls.create_service(provider, config)
        if service:
            cls._instances[key] = service

        return service

    @classmethod
    def clear_cache(cls):
        """清空所有服务实例"""
        for key, service in cls._instances.items():
            try:
                service.shutdown()
            except Exception as e:
                cls._logger.warning(f"关闭服务 {key} 失败: {e}")

        cls._instances.clear()
        cls._logger.info("已清空所有服务实例缓存")

    @classmethod
    def remove_service(cls, provider: str, model: Optional[str] = None):
        """
        移除指定的服务实例

        Args:
            provider: 提供商名称
            model: 模型名称（可选）
        """
        # 生成可能的键
        keys_to_remove = []
        for key in cls._instances:
            if key.startswith(provider):
                if model is None or model in key:
                    keys_to_remove.append(key)

        # 移除服务
        for key in keys_to_remove:
            try:
                cls._instances[key].shutdown()
                del cls._instances[key]
                cls._logger.info(f"已移除服务实例: {key}")
            except Exception as e:
                cls._logger.warning(f"移除服务 {key} 失败: {e}")

    @classmethod
    def get_all_services(cls) -> Dict[str, BaseLLMService]:
        """获取所有服务实例"""
        return cls._instances.copy()

    @classmethod
    def get_service_statistics(cls) -> Dict[str, Any]:
        """获取所有服务的统计信息"""
        stats = {}
        for key, service in cls._instances.items():
            stats[key] = service.get_statistics()
        return stats

    @classmethod
    def test_all_services(cls) -> Dict[str, Tuple[bool, str]]:
        """测试所有服务的连接"""
        results = {}
        for key, service in cls._instances.items():
            try:
                success, message = service.test_connection()
                results[key] = (success, message)
            except Exception as e:
                results[key] = (False, str(e))
        return results
