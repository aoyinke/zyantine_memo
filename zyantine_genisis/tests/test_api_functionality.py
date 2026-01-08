#!/usr/bin/env python3
"""
直接测试LLM服务的API调用功能
"""

import sys
import os

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.abspath('.'))

from zyantine_genisis.api.config_manager import ConfigManager
from zyantine_genisis.api.llm_service_factory import LLMServiceFactory

def test_api_call():
    """测试API调用功能"""
    print("=== 测试LLM服务API调用功能 ===")
    
    # 初始化配置管理器
    config_manager = ConfigManager()
    config = config_manager.load_config('./zyantine_genisis/config/llm_config.json')
    
    # 获取API配置
    api_config = config.get('api', {})
    
    print(f"配置信息：")
    print(f"  Provider: {api_config.get('provider')}")
    print(f"  Model: {api_config.get('chat_model')}")
    print(f"  Base URL: {api_config.get('base_url')}")
    
    # 创建LLM服务实例
    try:
        llm_service = LLMServiceFactory.create_service(api_config)
        print(f"\n✓ 成功创建{api_config.get('provider')}服务实例")
        
        # 测试API调用
        test_messages = [
            {"role": "user", "content": "你好，这是一个测试，简单介绍一下自己"}
        ]
        
        print("\n正在测试API调用...")
        response = llm_service.call_api(
            messages=test_messages,
            max_tokens=100,
            temperature=0.7
        )
        
        print(f"✓ API调用成功!")
        print(f"响应内容：\n{response['content']}")
        
        # 检查是否使用了正确的参数
        print(f"\n响应状态：")
        print(f"  模型：{response.get('model')}")
        print(f"  完成原因：{response.get('finish_reason')}")
        
        return True
        
    except Exception as e:
        print(f"\n✗ 测试失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_api_call()
    sys.exit(0 if success else 1)
