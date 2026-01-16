#!/usr/bin/env python3
# 测试API服务修复

import os
import sys
from zyantine_genisis.api.llm_service_factory import LLMServiceFactory
from zyantine_genisis.config.config_manager import ConfigManager

def test_api_service_creation():
    """测试API服务创建"""
    print("=== 测试API服务创建 ===")
    
    # 测试1：使用空API密钥创建服务
    print("\n1. 测试使用空API密钥创建服务：")
    config = {
        "enabled": True,
        "api_key": "",
        "base_url": "https://api.deepseek.com",
        "chat_model": "deepseek-chat",
        "timeout": 30,
        "max_retries": 3,
        "use_max_completion_tokens": True
    }
    service = LLMServiceFactory.create_service("deepseek", config)
    print(f"服务创建结果: {'成功' if service else '失败'}")
    if service:
        print(f"服务可用性: {'可用' if service.is_available() else '不可用'}")
    
    # 测试2：使用有效的API密钥占位符创建服务
    print("\n2. 测试使用有效API密钥占位符创建服务：")
    config = {
        "enabled": True,
        "api_key": "sk-valid-api-key",
        "base_url": "https://api.deepseek.com",
        "chat_model": "deepseek-chat",
        "timeout": 30,
        "max_retries": 3,
        "use_max_completion_tokens": True
    }
    service = LLMServiceFactory.create_service("deepseek", config)
    print(f"服务创建结果: {'成功' if service else '失败'}")
    if service:
        print(f"服务可用性: {'可用' if service.is_available() else '不可用'}")
    
    return service

def test_config_loading():
    """测试配置加载"""
    print("\n=== 测试配置加载 ===")
    
    # 加载配置
    config_manager = ConfigManager()
    config = config_manager.get()
    
    print(f"配置文件路径: {config_manager.get_config_file()}")
    print(f"主API密钥: {'已配置' if config.api.api_key and config.api.api_key.strip() else '未配置'}")
    print(f"当前提供商: {config.api.provider}")
    print(f"提供商配置: {config.api.providers}")
    
    return config

def test_service_provider_initialization():
    """测试服务提供者初始化"""
    print("\n=== 测试服务提供者初始化 ===")
    
    # 导入服务提供者
    from zyantine_genisis.api.service_provider import APIServiceProvider
    
    # 加载配置
    config_manager = ConfigManager()
    config = config_manager.get()
    
    try:
        # 创建服务提供者
        provider = APIServiceProvider(config)
        print("服务提供者初始化成功")
        print(f"活跃服务: {provider.active_service}")
        print(f"可用服务: {list(provider.services.keys())}")
        print(f"回复生成器API服务: {'可用' if provider.reply_generator.api else '不可用'}")
        
        return provider
    except Exception as e:
        print(f"服务提供者初始化失败: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    """主函数"""
    print("=== API服务修复测试 ===")
    
    # 测试API服务创建
    test_api_service_creation()
    
    # 测试配置加载
    test_config_loading()
    
    # 测试服务提供者初始化
    provider = test_service_provider_initialization()
    
    print("\n=== 测试总结 ===")
    if provider:
        if provider.active_service:
            print("✅ 服务提供者成功初始化并创建了活跃服务")
        else:
            print("✅ 服务提供者成功初始化并正确使用本地模式")
        return 0
    else:
        print("❌ 服务提供者初始化失败")
        return 1

if __name__ == "__main__":
    sys.exit(main())