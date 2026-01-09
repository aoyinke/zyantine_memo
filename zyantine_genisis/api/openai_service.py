"""
OpenAI服务 - 封装OpenAI API调用
"""
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
import time
import traceback
from dataclasses import dataclass, field
from enum import Enum

from openai import OpenAI
from openai.types.chat import ChatCompletion
import backoff

from utils.logger import SystemLogger
from .llm_service import OpenAICompatibleService, BaseLLMService, APIErrorType, APIRequest
from .llm_provider import LLMProvider, LLMModelConfig


class OpenAIModel(Enum):
    """OpenAI模型枚举"""
    GPT4_TURBO = "gpt-4-turbo"
    GPT4 = "gpt-4"
    GPT35_TURBO = "gpt-3.5-turbo"
    GPT4O = "gpt-4o"
    GPT4O_MINI = "gpt-4o-mini"


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





















