"""
LLM服务提供商枚举和配置
"""
from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, List, Optional


class LLMProvider(Enum):
    """LLM服务提供商枚举"""
    OPENAI = "openai"
    DEEPSEEK = "deepseek"
    ANTHROPIC = "anthropic"
    AZURE = "azure"
    ALIBABA = "alibaba"
    BAIDU = "baidu"
    ZHIPU = "zhipu"
    MOONSHOT = "moonshot"


@dataclass
class LLMModelConfig:
    """LLM模型配置"""
    provider: LLMProvider
    model_name: str
    api_key: str
    base_url: str
    timeout: int = 30
    max_retries: int = 3
    temperature: float = 1.0
    max_tokens: int = 2000
    enabled: bool = True
    extra_params: Dict = field(default_factory=dict)
    use_max_completion_tokens: bool = False  # 是否使用max_completion_tokens参数
    max_context_tokens: int = 3000  # 最大上下文token数


class LLMProviderPresets:
    """LLM服务提供商预设配置"""

    @staticmethod
    def get_deepseek_config(api_key: str) -> LLMModelConfig:
        """获取DeepSeek配置"""
        return LLMModelConfig(
            provider=LLMProvider.DEEPSEEK,
            model_name="deepseek-chat",
            api_key=api_key,
            base_url="https://api.deepseek.com",
            timeout=30,
            max_retries=3,
            use_max_completion_tokens=True
        )

    @staticmethod
    def get_openai_config(api_key: str, model: str = "gpt-5-nano-2025-08-07") -> LLMModelConfig:
        """获取OpenAI配置"""
        return LLMModelConfig(
            provider=LLMProvider.OPENAI,
            model_name=model,
            api_key=api_key,
            base_url="https://api.openai.com/v1",
            timeout=30,
            max_retries=3
        )

    @staticmethod
    def get_anthropic_config(api_key: str, model: str = "claude-3-5-sonnet-20241022") -> LLMModelConfig:
        """获取Anthropic配置"""
        return LLMModelConfig(
            provider=LLMProvider.ANTHROPIC,
            model_name=model,
            api_key=api_key,
            base_url="https://api.anthropic.com",
            timeout=30,
            max_retries=3
        )

    @staticmethod
    def get_zhipu_config(api_key: str, model: str = "glm-4") -> LLMModelConfig:
        """获取智谱AI配置"""
        return LLMModelConfig(
            provider=LLMProvider.ZHIPU,
            model_name=model,
            api_key=api_key,
            base_url="https://open.bigmodel.cn/api/paas/v4",
            timeout=30,
            max_retries=3
        )

    @staticmethod
    def get_moonshot_config(api_key: str, model: str = "moonshot-v1-8k") -> LLMModelConfig:
        """获取月之暗面配置"""
        return LLMModelConfig(
            provider=LLMProvider.MOONSHOT,
            model_name=model,
            api_key=api_key,
            base_url="https://api.moonshot.cn/v1",
            timeout=30,
            max_retries=3
        )

    @staticmethod
    def get_alibaba_config(api_key: str, model: str = "qwen-turbo") -> LLMModelConfig:
        """获取阿里云配置"""
        return LLMModelConfig(
            provider=LLMProvider.ALIBABA,
            model_name=model,
            api_key=api_key,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            timeout=30,
            max_retries=3
        )

    @staticmethod
    def get_baidu_config(api_key: str, model: str = "ernie-bot-4") -> LLMModelConfig:
        """获取百度配置"""
        return LLMModelConfig(
            provider=LLMProvider.BAIDU,
            model_name=model,
            api_key=api_key,
            base_url="https://aip.baidubce.com/rpc/2.0/ai_custom/v1/wenxinworkshop/chat",
            timeout=30,
            max_retries=3
        )

    @staticmethod
    def get_custom_config(provider: LLMProvider,
                         model_name: str,
                         api_key: str,
                         base_url: str,
                         **kwargs) -> LLMModelConfig:
        """获取自定义配置"""
        return LLMModelConfig(
            provider=provider,
            model_name=model_name,
            api_key=api_key,
            base_url=base_url,
            **kwargs
        )
