#!/usr/bin/env python3
"""
测试DeepSeek参数转换逻辑的脚本
"""

import os
import sys
from unittest.mock import Mock, patch

# 添加项目根目录到Python路径
project_root = os.path.abspath('/Users/gyc/Desktop/GYC_coding/zyantine/zyantine_memo')
sys.path.append(project_root)

from zyantine_genisis.api.llm_service_factory import LLMServiceFactory
from zyantine_genisis.api.llm_provider import LLMProvider
from zyantine_genisis.api.llm_service import OpenAICompatibleService


def test_deepseek_service_creation():
    """测试创建DeepSeek服务时是否正确设置了参数"""
    print("=== 测试DeepSeek服务创建 ===")
    
    # 配置DeepSeek参数
    deepseek_config = {
        "enabled": True,
        "api_key": "test_api_key",
        "base_url": "https://api.deepseek.com",
        "chat_model": "deepseek-chat",
        "timeout": 30,
        "max_retries": 3
    }
    
    # 使用工厂创建服务
    service = LLMServiceFactory.create_service("deepseek", deepseek_config)
    
    if service:
        print(f"✅ 成功创建DeepSeek服务: {type(service).__name__}")
        print(f"   - 提供商: {service.provider}")
        print(f"   - 模型: {service.model}")
        print(f"   - use_max_completion_tokens: {service.config.use_max_completion_tokens}")
        return service
    else:
        print("❌ 创建DeepSeek服务失败")
        return None


def test_parameter_conversion(service):
    """测试参数转换逻辑"""
    print("\n=== 测试参数转换逻辑 ===")
    
    if not service:
        print("❌ 服务未初始化，无法测试参数转换")
        return False
    
    # 检查是否是OpenAICompatibleService
    if not isinstance(service, OpenAICompatibleService):
        print(f"❌ 服务类型错误: {type(service).__name__}, 应该是 OpenAICompatibleService")
        return False
    
    # 创建模拟客户端和响应
    mock_response = Mock()
    mock_choice = Mock()
    mock_message = Mock()
    mock_message.content = "测试响应"
    mock_choice.message = mock_message
    mock_choice.delta = None
    mock_response.choices = [mock_choice]
    
    # 模拟API调用
    captured_params = None
    
    def mock_create(*args, **kwargs):
        nonlocal captured_params
        captured_params = kwargs
        print(f"   API调用参数: {kwargs}")
        return mock_response
    
    # 替换client.chat.completions.create
    with patch.object(service.client.chat.completions, 'create', side_effect=mock_create):
        try:
            # 调用生成回复方法
            response = service.generate_reply(
                system_prompt="你是一个助手",
                user_input="你好",
                max_tokens=500,
                temperature=0.7,
                stream=False
            )
            
            if captured_params:
                if service.config.use_max_completion_tokens:
                    if "max_completion_tokens" in captured_params:
                        print(f"✅ 成功: 使用了max_completion_tokens参数: {captured_params['max_completion_tokens']}")
                        if "max_tokens" not in captured_params:
                            print("✅ 成功: 没有使用max_tokens参数")
                            return True
                        else:
                            print("❌ 错误: 同时使用了max_tokens和max_completion_tokens参数")
                            return False
                    else:
                        print("❌ 错误: 应该使用max_completion_tokens但使用了其他参数")
                        return False
                else:
                    if "max_tokens" in captured_params:
                        print(f"✅ 成功: 使用了max_tokens参数: {captured_params['max_tokens']}")
                        return True
                    else:
                        print("❌ 错误: 应该使用max_tokens但使用了其他参数")
                        return False
            else:
                print("❌ 错误: 没有捕获到API调用参数")
                return False
                
        except Exception as e:
            print(f"❌ 测试过程中出错: {e}")
            import traceback
            traceback.print_exc()
            return False


def test_manual_param_setting():
    """手动测试参数设置"""
    print("\n=== 测试手动参数设置 ===")
    
    # 配置DeepSeek参数并显式设置use_max_completion_tokens
    deepseek_config = {
        "enabled": True,
        "api_key": "test_api_key",
        "base_url": "https://api.deepseek.com",
        "chat_model": "deepseek-chat",
        "timeout": 30,
        "max_retries": 3,
        "use_max_completion_tokens": True
    }
    
    # 使用工厂创建服务
    service = LLMServiceFactory.create_service("deepseek", deepseek_config)
    
    if service:
        print(f"✅ 成功创建DeepSeek服务")
        print(f"   - use_max_completion_tokens: {service.config.use_max_completion_tokens}")
        return service
    else:
        print("❌ 创建DeepSeek服务失败")
        return None


if __name__ == "__main__":
    print("开始测试DeepSeek参数转换逻辑")
    print("=" * 50)
    
    # 测试1: 使用默认配置创建服务
    service1 = test_deepseek_service_creation()
    result1 = test_parameter_conversion(service1)
    
    # 测试2: 手动设置参数
    service2 = test_manual_param_setting()
    result2 = test_parameter_conversion(service2)
    
    print("\n" + "=" * 50)
    print("测试总结:")
    print(f"测试1 (默认配置): {'通过' if result1 else '失败'}")
    print(f"测试2 (手动配置): {'通过' if result2 else '失败'}")
    
    if result1 and result2:
        print("\n✅ 所有测试都通过了！DeepSeek参数转换逻辑正常工作。")
        sys.exit(0)
    else:
        print("\n❌ 部分测试失败，请检查配置。")
        sys.exit(1)
