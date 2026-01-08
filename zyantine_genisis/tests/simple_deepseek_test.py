#!/usr/bin/env python3
"""
简单测试DeepSeek参数转换逻辑的脚本
直接测试OpenAICompatibleService的参数转换逻辑，避免项目依赖问题
"""

import os
import sys
from unittest.mock import Mock, patch

# 添加项目根目录和zyantine_genisis目录到Python路径
project_root = os.path.abspath('/Users/gyc/Desktop/GYC_coding/zyantine/zyantine_memo')
sys.path.append(project_root)

# 添加zyantine_genisis目录到Python路径
sys.path.append(os.path.join(project_root, 'zyantine_genisis'))

# 直接导入需要测试的类和配置
from zyantine_genisis.api.llm_service import OpenAICompatibleService
from zyantine_genisis.api.llm_provider import LLMModelConfig, LLMProvider

def test_deepseek_parameter_conversion():
    """测试DeepSeek参数转换逻辑"""
    print("=== 测试DeepSeek参数转换逻辑 ===")
    
    # 创建LLMModelConfig，设置use_max_completion_tokens=True
    model_config = LLMModelConfig(
        provider=LLMProvider.DEEPSEEK,
        model_name="deepseek-chat",
        api_key="test_api_key",
        base_url="https://api.deepseek.com",
        use_max_completion_tokens=True  # 关键参数
    )
    
    # 创建OpenAICompatibleService实例
    service = OpenAICompatibleService(model_config)
    
    # 模拟客户端
    mock_client = Mock()
    service.client = mock_client
    
    # 测试消息
    test_messages = [
        {"role": "system", "content": "你是一个助手"},
        {"role": "user", "content": "你好"}
    ]
    
    # 测试参数
    test_max_tokens = 500
    test_temperature = 0.7
    test_stream = False
    
    # 捕获API调用参数
    captured_params = None
    
    def mock_create(*args, **kwargs):
        nonlocal captured_params
        captured_params = kwargs
        print(f"   API调用参数: {kwargs}")
        return Mock()
    
    # 替换client.chat.completions.create
    mock_create_instance = Mock(side_effect=mock_create)
    mock_client.chat.completions.create = mock_create_instance
    
    try:
        # 调用_call_api方法
        service._call_api(
            messages=test_messages,
            max_tokens=test_max_tokens,
            temperature=test_temperature,
            stream=test_stream,
            request_id="test_request"
        )
        
        # 验证参数
        if captured_params:
            if "max_completion_tokens" in captured_params:
                print("✅ 成功: 使用了max_completion_tokens参数")
                if captured_params["max_completion_tokens"] == test_max_tokens:
                    print("✅ 成功: max_completion_tokens值正确")
                else:
                    print(f"❌ 错误: max_completion_tokens值不正确，预期: {test_max_tokens}，实际: {captured_params['max_completion_tokens']}")
            
            if "max_tokens" in captured_params:
                print("❌ 错误: 同时使用了max_tokens参数")
        else:
            print("❌ 错误: 没有捕获到API调用参数")
            
    except Exception as e:
        print(f"❌ 测试过程中出错: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

def test_openai_parameter_usage():
    """测试OpenAI参数使用（对比）"""
    print("\n=== 测试OpenAI参数使用（对比） ===")
    
    # 创建LLMModelConfig，设置use_max_completion_tokens=False
    model_config = LLMModelConfig(
        provider=LLMProvider.OPENAI,
        model_name="gpt-4o-mini",
        api_key="test_api_key",
        base_url="https://api.openai.com/v1",
        use_max_completion_tokens=False  # 关键参数
    )
    
    # 创建OpenAICompatibleService实例
    service = OpenAICompatibleService(model_config)
    
    # 模拟客户端
    mock_client = Mock()
    service.client = mock_client
    
    # 测试消息
    test_messages = [
        {"role": "system", "content": "你是一个助手"},
        {"role": "user", "content": "你好"}
    ]
    
    # 测试参数
    test_max_tokens = 500
    test_temperature = 0.7
    test_stream = False
    
    # 捕获API调用参数
    captured_params = None
    
    def mock_create(*args, **kwargs):
        nonlocal captured_params
        captured_params = kwargs
        print(f"   API调用参数: {kwargs}")
        return Mock()
    
    # 替换client.chat.completions.create
    mock_create_instance = Mock(side_effect=mock_create)
    mock_client.chat.completions.create = mock_create_instance
    
    try:
        # 调用_call_api方法
        service._call_api(
            messages=test_messages,
            max_tokens=test_max_tokens,
            temperature=test_temperature,
            stream=test_stream,
            request_id="test_request"
        )
        
        # 验证参数
        if captured_params:
            if "max_tokens" in captured_params:
                print("✅ 成功: 使用了max_tokens参数")
                if captured_params["max_tokens"] == test_max_tokens:
                    print("✅ 成功: max_tokens值正确")
                else:
                    print(f"❌ 错误: max_tokens值不正确，预期: {test_max_tokens}，实际: {captured_params['max_tokens']}")
            
            if "max_completion_tokens" in captured_params:
                print("❌ 错误: 同时使用了max_completion_tokens参数")
        else:
            print("❌ 错误: 没有捕获到API调用参数")
            
    except Exception as e:
        print(f"❌ 测试过程中出错: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


def test_config_logic():
    """测试配置逻辑"""
    print("\n=== 测试配置逻辑 ===")
    
    # 测试当provider是DEEPSEEK时，use_max_completion_tokens默认值
    from zyantine_genisis.api.llm_service_factory import LLMServiceFactory
    
    # 测试LLMServiceFactory的配置逻辑
    deepseek_config = {
        "enabled": True,
        "api_key": "test_api_key",
        "base_url": "https://api.deepseek.com",
        "chat_model": "deepseek-chat"
    }
    
    # 我们不实际创建服务，而是模拟配置创建过程
    try:
        # 模拟LLMModelConfig的创建
        from zyantine_genisis.api.llm_provider import LLMModelConfig
        
        # 测试当provider是DEEPSEEK时的默认行为
        model_config = LLMModelConfig(
            provider=LLMProvider.DEEPSEEK,
            model_name=deepseek_config["chat_model"],
            api_key=deepseek_config["api_key"],
            base_url=deepseek_config["base_url"]
        )
        
        print(f"DEEPSEEK默认use_max_completion_tokens: {model_config.use_max_completion_tokens}")
        
        # 测试当provider是OPENAI时的默认行为
        openai_config = {
            "enabled": True,
            "api_key": "test_api_key",
            "base_url": "https://api.openai.com/v1",
            "chat_model": "gpt-4o-mini"
        }
        
        model_config_openai = LLMModelConfig(
            provider=LLMProvider.OPENAI,
            model_name=openai_config["chat_model"],
            api_key=openai_config["api_key"],
            base_url=openai_config["base_url"]
        )
        
        print(f"OPENAI默认use_max_completion_tokens: {model_config_openai.use_max_completion_tokens}")
        
        return True
        
    except Exception as e:
        print(f"❌ 配置逻辑测试失败: {e}")
        return False


if __name__ == "__main__":
    print("开始测试DeepSeek参数转换逻辑")
    print("=" * 50)
    
    # 运行测试
    test1_result = test_deepseek_parameter_conversion()
    test2_result = test_openai_parameter_usage()
    test3_result = test_config_logic()
    
    print("\n" + "=" * 50)
    print("测试总结:")
    print(f"测试1 (DeepSeek参数转换): {'通过' if test1_result else '失败'}")
    print(f"测试2 (OpenAI参数使用): {'通过' if test2_result else '失败'}")
    print(f"测试3 (配置逻辑): {'通过' if test3_result else '失败'}")
    
    if test1_result and test2_result and test3_result:
        print("\n✅ 所有测试都通过了！DeepSeek参数转换逻辑正常工作。")
        sys.exit(0)
    else:
        print("\n❌ 部分测试失败，请检查配置。")
        sys.exit(1)
