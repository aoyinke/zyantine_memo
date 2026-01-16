#!/usr/bin/env python3
# 测试API服务可用性

import os
import sys
from zyantine_genisis.api.llm_service import OpenAICompatibleService
from zyantine_genisis.api.llm_provider import LLMProvider, LLMModelConfig
from zyantine_genisis.config.config_manager import ConfigManager

def test_api_config():
    """测试API配置"""
    print("=== 测试API配置 ===")
    
    # 加载配置
    config_manager = ConfigManager()
    config = config_manager.get()
    
    print(f"配置文件路径: {config_manager.get_config_file()}")
    print(f"主API密钥: {'已配置' if config.api.api_key else '未配置'}")
    print(f"OpenAI提供商API密钥: {'已配置' if config.api.providers.get('openai', {}).get('api_key') else '未配置'}")
    print(f"DeepSeek提供商API密钥: {'已配置' if config.api.providers.get('deepseek', {}).get('api_key') else '未配置'}")
    print(f"Mem0 LLM API密钥: {'已配置' if config.memory.memo0_config.get('llm', {}).get('config', {}).get('api_key') else '未配置'}")
    print(f"API基础URL: {config.api.base_url}")
    
    # 检查环境变量
    print("\n=== 检查环境变量 ===")
    print(f"OPENAI_API_KEY: {'已设置' if os.getenv('OPENAI_API_KEY') else '未设置'}")
    print(f"OPENAI_BASE_URL: {'已设置' if os.getenv('OPENAI_BASE_URL') else '未设置'}")
    print(f"ZHIPU_API_KEY: {'已设置' if os.getenv('ZHIPU_API_KEY') else '未设置'}")
    
    return config

def test_api_initialization(config):
    """测试API初始化"""
    print("\n=== 测试API初始化 ===")
    
    # 从配置中获取API参数
    api_key = config.api.api_key or config.api.providers.get('openai', {}).get('api_key') or os.getenv('OPENAI_API_KEY')
    base_url = config.api.base_url or os.getenv('OPENAI_BASE_URL', 'https://openkey.cloud/v1')
    model = config.api.chat_model or 'gpt-5-nano-2025-08-07'
    
    print(f"使用的API密钥: {'已获取' if api_key else '未获取'}")
    print(f"使用的API基础URL: {base_url}")
    print(f"使用的模型: {model}")
    
    # 创建LLMModelConfig
    llm_config = LLMModelConfig(
        provider=LLMProvider.OPENAI,
        model_name=model,
        api_key=api_key,
        base_url=base_url,
        timeout=30,
        max_retries=3
    )
    
    try:
        print("\n正在初始化API服务...")
        service = OpenAICompatibleService(llm_config)
        print(f"API服务初始化结果: {'成功' if service.client is not None else '失败'}")
        print(f"服务可用性: {'可用' if service.is_available() else '不可用'}")
        
        if service.client:
            print("\n正在测试连接...")
            success, message = service.test_connection()
            print(f"连接测试结果: {'成功' if success else '失败'}")
            print(f"连接测试消息: {message}")
        
        return service
    except Exception as e:
        print(f"API服务初始化失败: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    """主函数"""
    print("=== API服务可用性测试 ===")
    
    # 测试配置
    config = test_api_config()
    
    # 测试API初始化
    service = test_api_initialization(config)
    
    print("\n=== 测试总结 ===")
    if service and service.is_available():
        print("✅ API服务可用")
        return 0
    else:
        print("❌ API服务不可用")
        return 1

if __name__ == "__main__":
    sys.exit(main())