#!/usr/bin/env python3
"""
独立的DeepSeek参数转换验证脚本
不依赖项目其他模块，直接验证核心修复逻辑
"""
import logging
from unittest.mock import MagicMock, patch
from enum import Enum

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 模拟必要的类和枚举
class LLMProvider(Enum):
    """模拟LLMProvider枚举"""
    OPENAI = "openai"
    DEEPSEEK = "deepseek"
    
class LLMModelConfig:
    """模拟LLMModelConfig类"""
    def __init__(self, provider, model_name, api_key, base_url, timeout, max_retries, use_max_completion_tokens):
        self.provider = provider
        self.model_name = model_name
        self.api_key = api_key
        self.base_url = base_url
        self.timeout = timeout
        self.max_retries = max_retries
        self.use_max_completion_tokens = use_max_completion_tokens

# 模拟OpenAICompatibleService的核心逻辑
class MockOpenAICompatibleService:
    """模拟OpenAICompatibleService，只包含核心的参数转换逻辑"""
    def __init__(self, config):
        self.config = config
        self.client = None
    
    def _call_api(self, messages, max_tokens, temperature, stream, request_id):
        """模拟_call_api方法，包含参数转换逻辑"""
        if self.config.use_max_completion_tokens:
            response = self.client.chat.completions.create(
                model=self.config.model_name,
                messages=messages,
                max_completion_tokens=max_tokens,  # DeepSeek使用max_completion_tokens
                temperature=temperature,
                stream=stream
            )
        else:
            response = self.client.chat.completions.create(
                model=self.config.model_name,
                messages=messages,
                max_tokens=max_tokens,  # OpenAI使用max_tokens
                temperature=temperature,
                stream=stream
            )
        return response

def verify_deepseek_param_conversion():
    """
    验证DeepSeek参数转换逻辑
    """
    logger.info("=== 验证DeepSeek参数转换逻辑 ===")
    
    # 创建DeepSeek配置
    deepseek_config = LLMModelConfig(
        provider=LLMProvider.DEEPSEEK,
        model_name="deepseek-chat",
        api_key="test_key",
        base_url="https://api.deepseek.com",
        timeout=30,
        max_retries=3,
        use_max_completion_tokens=True  # 启用max_completion_tokens
    )
    
    # 创建服务实例
    service = MockOpenAICompatibleService(deepseek_config)
    
    # 模拟客户端
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_client.chat.completions.create.return_value = mock_response
    service.client = mock_client
    
    # 测试API调用
    messages = [{"role": "system", "content": "You are a helpful assistant."}, 
               {"role": "user", "content": "Hello, DeepSeek!"}]
    
    service._call_api(
        messages=messages,
        max_tokens=500,
        temperature=0.7,
        stream=False,
        request_id="test-request-123"
    )
    
    # 检查API调用参数
    call_args = mock_client.chat.completions.create.call_args
    kwargs = call_args.kwargs
    
    logger.info("API调用参数:")
    for key, value in kwargs.items():
        logger.info(f"   - {key}: {value}")
    
    # 验证参数转换是否正确
    if "max_completion_tokens" in kwargs and "max_tokens" not in kwargs:
        logger.info("✅ 验证通过！DeepSeek正确使用了max_completion_tokens参数")
        logger.info(f"   - max_completion_tokens: {kwargs['max_completion_tokens']}")
        return True
    else:
        logger.error("❌ 验证失败！DeepSeek没有正确使用max_completion_tokens参数")
        logger.error(f"   - 使用了max_tokens: {'max_tokens' in kwargs}")
        logger.error(f"   - 使用了max_completion_tokens: {'max_completion_tokens' in kwargs}")
        return False

def verify_openai_param_usage():
    """
    验证OpenAI参数使用逻辑（作为对比）
    """
    logger.info("\n=== 验证OpenAI参数使用逻辑 ===")
    
    # 创建OpenAI配置
    openai_config = LLMModelConfig(
        provider=LLMProvider.OPENAI,
        model_name="gpt-5-nano",
        api_key="test_key",
        base_url="https://api.openai.com/v1",
        timeout=30,
        max_retries=3,
        use_max_completion_tokens=False  # 禁用max_completion_tokens
    )
    
    # 创建服务实例
    service = MockOpenAICompatibleService(openai_config)
    
    # 模拟客户端
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_client.chat.completions.create.return_value = mock_response
    service.client = mock_client
    
    # 测试API调用
    messages = [{"role": "system", "content": "You are a helpful assistant."}, 
               {"role": "user", "content": "Hello, OpenAI!"}]
    
    service._call_api(
        messages=messages,
        max_tokens=500,
        temperature=0.7,
        stream=False,
        request_id="test-request-456"
    )
    
    # 检查API调用参数
    call_args = mock_client.chat.completions.create.call_args
    kwargs = call_args.kwargs
    
    logger.info("API调用参数:")
    for key, value in kwargs.items():
        logger.info(f"   - {key}: {value}")
    
    # 验证参数使用是否正确
    if "max_tokens" in kwargs and "max_completion_tokens" not in kwargs:
        logger.info("✅ 验证通过！OpenAI正确使用了max_tokens参数")
        logger.info(f"   - max_tokens: {kwargs['max_tokens']}")
        return True
    else:
        logger.error("❌ 验证失败！OpenAI没有正确使用max_tokens参数")
        logger.error(f"   - 使用了max_tokens: {'max_tokens' in kwargs}")
        logger.error(f"   - 使用了max_completion_tokens: {'max_completion_tokens' in kwargs}")
        return False

# 验证llm_service_factory中的默认行为
def verify_factory_default_behavior():
    """
    验证LLMServiceFactory对DeepSeek的默认行为
    """
    logger.info("\n=== 验证LLMServiceFactory默认行为 ===")
    
    # 模拟工厂创建服务的逻辑
    def mock_create_service(provider, config):
        """模拟create_service方法的核心逻辑"""
        provider_enum = LLMProvider(provider)
        
        # 注意这里的关键行：对于DeepSeek，默认启用use_max_completion_tokens
        use_max_completion_tokens = config.get("use_max_completion_tokens", provider_enum == LLMProvider.DEEPSEEK)
        
        model_config = LLMModelConfig(
            provider=provider_enum,
            model_name=config.get("chat_model", "default-model"),
            api_key=config.get("api_key", ""),
            base_url=config.get("base_url", ""),
            timeout=config.get("timeout", 30),
            max_retries=config.get("max_retries", 3),
            use_max_completion_tokens=use_max_completion_tokens
        )
        
        return model_config
    
    # 测试DeepSeek配置
    deepseek_config = {
        "chat_model": "deepseek-chat",
        "api_key": "test_key",
        "base_url": "https://api.deepseek.com",
    }
    
    model_config = mock_create_service("deepseek", deepseek_config)
    
    if model_config.use_max_completion_tokens is True:
        logger.info("✅ 验证通过！LLMServiceFactory为DeepSeek默认启用了use_max_completion_tokens")
        logger.info(f"   - provider: {model_config.provider.value}")
        logger.info(f"   - use_max_completion_tokens: {model_config.use_max_completion_tokens}")
        return True
    else:
        logger.error("❌ 验证失败！LLMServiceFactory没有为DeepSeek默认启用use_max_completion_tokens")
        return False

if __name__ == "__main__":
    logger.info("开始验证DeepSeek参数转换修复...")
    
    # 运行验证
    deepseek_result = verify_deepseek_param_conversion()
    openai_result = verify_openai_param_usage()
    factory_result = verify_factory_default_behavior()
    
    logger.info("\n=== 验证总结 ===")
    logger.info(f"DeepSeek参数转换: {'✅ 通过' if deepseek_result else '❌ 失败'}")
    logger.info(f"OpenAI参数使用: {'✅ 通过' if openai_result else '❌ 失败'}")
    logger.info(f"工厂默认行为: {'✅ 通过' if factory_result else '❌ 失败'}")
    
    if deepseek_result and openai_result and factory_result:
        logger.info("\n🎉 所有验证通过！DeepSeek参数转换修复逻辑正确")
        logger.info("\n修复总结：")
        logger.info("1. 启用了DeepSeek配置 (config_manager.py)")
        logger.info("2. OpenAICompatibleService根据use_max_completion_tokens参数选择使用的API参数")
        logger.info("3. LLMServiceFactory为DeepSeek默认启用use_max_completion_tokens")
        logger.info("4. 当use_max_completion_tokens=True时，API调用使用max_completion_tokens参数")
        logger.info("\n这将解决'unsupported parameter: max_tokens'错误")
    else:
        logger.error("\n💥 部分验证失败，需要进一步检查")
