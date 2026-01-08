#!/usr/bin/env python3
"""
简单的配置测试脚本 - 验证配置加载和服务创建
"""
import os
import sys

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.abspath('.'))

from zyantine_genisis.config.config_manager import ConfigManager, SystemConfig
from zyantine_genisis.api.llm_service_factory import LLMServiceFactory

def test_config_loading():
    """测试配置文件加载"""
    print("=== 测试配置文件加载 ===")
    
    # 创建配置管理器实例
    config_manager = ConfigManager()
    
    # 加载配置
    config = config_manager.load()
    
    # 打印配置信息
    print(f"配置文件路径: {config_manager._config_file}")
    print(f"API提供商: {config.api.provider}")
    print(f"API基础URL: {config.api.base_url}")
    print(f"当前提供商: {config.api.provider}")
    
    # 检查是否正确加载了use_max_completion_tokens配置
    if config.api.provider and hasattr(config.api, 'providers'):
        provider_config = config.api.providers.get(config.api.provider, {})
        print(f"当前提供商配置: {provider_config}")
        if 'use_max_completion_tokens' in provider_config:
            print(f"use_max_completion_tokens: {provider_config['use_max_completion_tokens']}")
    
    print("配置加载测试完成\n")
    return config

def test_service_creation(config):
    """测试服务创建"""
    print("=== 测试服务创建 ===")
    
    # 获取当前选择的提供商
    provider = config.api.provider or "openai"
    
    # 获取提供商配置
    provider_config = config.api.providers.get(provider, {})
    
    print(f"正在为提供商 {provider} 创建服务...")
    print(f"提供商配置: {provider_config}")
    
    # 使用服务工厂创建服务
    service = LLMServiceFactory.create_service(provider, provider_config)
    
    if service:
        print(f"服务创建成功: {type(service).__name__}")
        print(f"服务配置: {service.config}")
        if hasattr(service.config, 'use_max_completion_tokens'):
            print(f"服务use_max_completion_tokens: {service.config.use_max_completion_tokens}")
    else:
        print(f"服务创建失败: {provider}")
    
    print("服务创建测试完成\n")
    return service

def test_deepseek_service():
    """测试DeepSeek服务创建"""
    print("=== 测试DeepSeek服务创建 ===")
    
    # 手动构造DeepSeek配置
    deepseek_config = {
        "enabled": True,
        "api_key": "sk-cd83b6411207408e8539b7623a1c5f35",
        "base_url": "https://api.deepseek.com",
        "chat_model": "deepseek-chat",
        "timeout": 30,
        "max_retries": 3,
        "use_max_completion_tokens": True
    }
    
    print(f"DeepSeek配置: {deepseek_config}")
    
    # 使用服务工厂创建DeepSeek服务
    service = LLMServiceFactory.create_service("deepseek", deepseek_config)
    
    if service:
        print(f"DeepSeek服务创建成功: {type(service).__name__}")
        print(f"服务配置: {service.config}")
        if hasattr(service.config, 'use_max_completion_tokens'):
            print(f"DeepSeek服务use_max_completion_tokens: {service.config.use_max_completion_tokens}")
    else:
        print("DeepSeek服务创建失败")
    
    print("DeepSeek服务测试完成\n")
    return service

if __name__ == "__main__":
    try:
        # 测试配置加载
        config = test_config_loading()
        
        # 测试服务创建
        test_service_creation(config)
        
        # 测试DeepSeek服务创建
        test_deepseek_service()
        
        print("所有测试完成!")
    except Exception as e:
        print(f"测试过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
