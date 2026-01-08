#!/usr/bin/env python3
"""
测试脚本：验证DeepSeek API服务的实际调用情况
"""

import sys
import os

# 添加项目根目录到Python路径
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from zyantine_genisis.config.config_manager import ConfigManager
from zyantine_genisis.api.service_provider import APIServiceProvider
from zyantine_genisis.api.llm_service import OpenAICompatibleService
from zyantine_genisis.api.openai_service import OpenAIService


def test_service_initialization():
    """测试服务初始化情况"""
    print("=== 测试服务初始化 ===")
    
    # 加载配置
    config_manager = ConfigManager()
    llm_config = config_manager.load()
    
    print("1. 配置信息：")
    print(f"   API提供商: {llm_config.api.provider}")
    print(f"   模型名称: {llm_config.api.providers[llm_config.api.provider].chat_model}")
    print(f"   Base URL: {llm_config.api.providers[llm_config.api.provider].base_url}")
    print(f"   Use max_completion_tokens: {llm_config.api.providers[llm_config.api.provider].use_max_completion_tokens}")
    
    # 初始化服务提供商
    provider = APIServiceProvider(llm_config)
    
    print("\n2. 服务提供商信息：")
    print(f"   激活的服务: {provider.active_service}")
    print(f"   服务列表: {list(provider.services.keys())}")
    
    # 检查回复生成器使用的服务
    print("\n3. 回复生成器信息：")
    reply_generator = provider.reply_generator
    
    # 检查API服务类型
    api_service = reply_generator.api
    print(f"   API服务类型: {type(api_service).__name__}")
    print(f"   API服务类: {api_service.__class__}")
    
    # 检查服务的属性
    print(f"   服务提供商: {getattr(api_service, 'provider', 'N/A')}")
    print(f"   模型名称: {getattr(api_service, 'model', 'N/A')}")
    print(f"   Base URL: {getattr(api_service, 'base_url', 'N/A')}")
    
    # 检查max_completion_tokens设置
    if hasattr(api_service, 'use_max_completion_tokens'):
        print(f"   use_max_completion_tokens: {api_service.use_max_completion_tokens}")
    elif hasattr(api_service, 'config') and hasattr(api_service.config, 'use_max_completion_tokens'):
        print(f"   use_max_completion_tokens: {api_service.config.use_max_completion_tokens}")
    
    # 检查logger名称
    print(f"   Logger名称: {api_service.logger.name}")
    
    # 检查服务工厂创建的服务
    print("\n4. LLMServiceFactory测试：")
    from api.llm_service_factory import LLMServiceFactory
    provider_name = llm_config.api.provider
    provider_config = llm_config.api.providers[provider_name]
    service = LLMServiceFactory.create_service(provider_name, provider_config)
    print(f"   工厂创建的服务类型: {type(service).__name__}")
    print(f"   工厂创建的服务类: {service.__class__}")
    print(f"   工厂创建的服务提供商: {service.provider}")
    print(f"   工厂创建的服务use_max_completion_tokens: {service.config.use_max_completion_tokens}")


def test_api_call_params():
    """测试API调用参数"""
    print("\n=== 测试API调用参数 ===")
    
    # 加载配置
    config_manager = ConfigManager()
    llm_config = config_manager.load()
    
    # 创建服务
    from zyantine_genisis.api.llm_service_factory import LLMServiceFactory
    provider_name = llm_config.api.provider
    provider_config = llm_config.api.providers[provider_name]
    service = LLMServiceFactory.create_service(provider_name, provider_config)
    
    print("模拟API调用参数：")
    messages = [{"role": "user", "content": "测试连接"}]
    max_tokens = 10
    temperature = 0.7
    stream = False
    request_id = "test_params"
    
    # 检查_call_api方法的实现
    import inspect
    call_api_source = inspect.getsource(service._call_api)
    print(f"_call_api方法源代码：\n{call_api_source}")
    
    # 检查配置中的use_max_completion_tokens
    print(f"\n配置中的use_max_completion_tokens: {provider_config.use_max_completion_tokens}")
    print(f"服务实例中的use_max_completion_tokens: {service.config.use_max_completion_tokens}")


if __name__ == "__main__":
    print("DeepSeek API服务测试")
    print("=" * 50)
    
    test_service_initialization()
    test_api_call_params()
    
    print("\n" + "=" * 50)
    print("测试完成")
