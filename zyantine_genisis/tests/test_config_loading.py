#!/usr/bin/env python3
"""
测试配置加载和服务初始化
"""
import sys
import os
import json

# 添加项目根目录和zyantine_genisis目录到Python路径
project_root = os.path.dirname(os.path.abspath(__file__))
zyantine_path = os.path.join(project_root, 'zyantine_genisis')
sys.path.append(project_root)
sys.path.append(zyantine_path)

from zyantine_genisis.config.config_manager import ConfigManager
from zyantine_genisis.api.service_provider import APIServiceProvider
from zyantine_genisis.api.llm_service_factory import LLMServiceFactory
from zyantine_genisis.api.llm_service import OpenAICompatibleService
from zyantine_genisis.api.llm_provider import LLMProvider, LLMModelConfig


def test_config_loading():
    """测试配置加载"""
    print("=" * 60)
    print("测试配置加载")
    print("=" * 60)
    
    # 1. 测试配置管理器
    config_manager = ConfigManager()
    config_path = "./zyantine_genisis/config/llm_config.json"
    
    print(f"加载配置文件: {config_path}")
    config = config_manager.load(config_path)
    
    print(f"\n=== 基本配置信息 ===")
    print(f"会话ID: {config.session_id}")
    print(f"系统名称: {config.system_name}")
    print(f"版本: {config.version}")
    
    print(f"\n=== API配置信息 ===")
    print(f"API启用: {config.api.enabled}")
    print(f"当前提供商: {config.api.provider}")
    print(f"API密钥: {config.api.api_key[:8]}..." if config.api.api_key else "无API密钥")
    print(f"基础URL: {config.api.base_url}")
    print(f"聊天模型: {config.api.chat_model}")
    
    print(f"\n=== 提供商配置 ===")
    for provider_name, provider_config in config.api.providers.items():
        if provider_config.get("enabled"):
            print(f"\n{provider_name}:")
            print(f"  启用: {provider_config.get('enabled')}")
            print(f"  API密钥: {provider_config.get('api_key')[:8]}..." if provider_config.get('api_key') else "无API密钥")
            print(f"  基础URL: {provider_config.get('base_url')}")
            print(f"  聊天模型: {provider_config.get('chat_model')}")
            print(f"  使用max_completion_tokens: {provider_config.get('use_max_completion_tokens')}")
    
    # 2. 测试服务工厂
    print(f"\n" + "=" * 60)
    print("测试服务工厂")
    print("=" * 60)
    
    provider = config.api.provider
    provider_config = config.api.providers.get(provider, {})
    
    print(f"创建{provider}服务实例...")
    service = LLMServiceFactory.create_service(provider, provider_config)
    
    if service:
        print(f"服务类型: {type(service).__name__}")
        if hasattr(service, 'config'):
            print(f"服务配置 - 提供商: {service.config.provider}")
            print(f"服务配置 - 模型: {service.config.model_name}")
            print(f"服务配置 - 使用max_completion_tokens: {service.config.use_max_completion_tokens}")
            print(f"服务配置 - API密钥: {service.config.api_key[:8]}..." if service.config.api_key else "无API密钥")
            print(f"服务配置 - 基础URL: {service.config.base_url}")
    
    # 3. 测试服务提供者
    print(f"\n" + "=" * 60)
    print("测试服务提供者")
    print("=" * 60)
    
    print(f"初始化API服务提供者...")
    try:
        service_provider = APIServiceProvider(config)
        
        print(f"服务提供者初始化成功")
        print(f"活动服务: {service_provider.active_service}")
        print(f"可用服务: {list(service_provider.services.keys())}")
        
        if service_provider.active_service:
            active_service = service_provider.services[service_provider.active_service]
            print(f"\n活动服务详情:")
            print(f"  服务类型: {type(active_service).__name__}")
            if hasattr(active_service, 'config'):
                print(f"  提供商: {active_service.config.provider}")
                print(f"  使用max_completion_tokens: {active_service.config.use_max_completion_tokens}")
            
            # 检查回复生成器
            if hasattr(service_provider, 'reply_generator'):
                reply_generator = service_provider.reply_generator
                print(f"\n回复生成器详情:")
                print(f"  生成器类型: {type(reply_generator).__name__}")
                if hasattr(reply_generator, 'api_service'):
                    api_service = reply_generator.api_service
                    print(f"  API服务类型: {type(api_service).__name__}" if api_service else "无API服务")
                    if api_service and hasattr(api_service, 'config'):
                        print(f"  API服务提供商: {api_service.config.provider}")
                        print(f"  API服务使用max_completion_tokens: {api_service.config.use_max_completion_tokens}")
    except Exception as e:
        print(f"服务提供者初始化失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_config_loading()
