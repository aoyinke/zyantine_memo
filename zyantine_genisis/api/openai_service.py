"""
OpenAI服务 - 封装OpenAI API调用
"""
from utils.logger import SystemLogger
from .llm_service import OpenAICompatibleService
from .llm_provider import LLMProvider, LLMModelConfig


class OpenAIService(OpenAICompatibleService):
    """OpenAI API服务封装（增强版）"""

    def __init__(self,
                 api_key: str,
                 base_url: str = "https://openkey.cloud/v1",
                 model: str = "gpt-5-nano-2025-08-07",
                 timeout: int = 30,
                 max_retries: int = 3,
                 use_max_completion_tokens: bool = False,
                 max_context_tokens: int = 3000):
        """
        初始化OpenAI服务
        
        Args:
            api_key: API密钥
            base_url: API基础URL
            model: 模型名称
            timeout: 超时时间（秒）
            max_retries: 最大重试次数
            use_max_completion_tokens: 是否使用max_completion_tokens参数
            max_context_tokens: 最大上下文token数
        """
        # 创建LLMModelConfig对象
        config = LLMModelConfig(
            provider=LLMProvider.OPENAI,
            model_name=model,
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
            max_retries=max_retries,
            use_max_completion_tokens=use_max_completion_tokens,
            max_context_tokens=max_context_tokens
        )

        # 调用父类初始化方法
        super().__init__(config)
        
        # 设置自定义日志器名称
        self.logger = SystemLogger().get_logger("openai_service")
        
        self.logger.info(f"OpenAI服务初始化完成，模型: {model}")





