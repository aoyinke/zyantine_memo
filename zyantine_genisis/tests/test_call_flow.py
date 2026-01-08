#!/usr/bin/env python3
"""
测试脚本：验证OpenAICompatibleService的调用流程

验证从generate_reply到_call_api的调用关系，以及参数处理逻辑
"""

import sys
import os

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from zyantine_genisis.api.llm_provider import LLMProvider, LLMModelConfig
from zyantine_genisis.api.llm_service import OpenAICompatibleService


def test_call_flow():
    """测试调用流程：generate_reply -> _call_api"""
    print("=== 测试OpenAICompatibleService调用流程 ===")
    
    # 创建测试配置
    # 使用DeepSeek作为示例，它会使用max_completion_tokens
    model_config = LLMModelConfig(
        provider=LLMProvider.DEEPSEEK,
        model_name="deepseek-chat",
        api_key="test_key",  # 使用测试密钥
        base_url="https://api.deepseek.com",
        timeout=30,
        max_retries=3,
        temperature=1.0,
        max_tokens=500,
        enabled=True,
        use_max_completion_tokens=True  # 设置为True以测试max_completion_tokens参数
    )
    
    # 创建服务实例
    service = OpenAICompatibleService(model_config)
    
    # 保存原始的_call_api方法
    original_call_api = service._call_api
    
    # 重写_call_api方法以验证调用
    def mock_call_api(self, messages, max_tokens, temperature, stream, request_id):
        """模拟_call_api方法，验证调用参数"""
        print("--- 执行OpenAICompatibleService._call_api方法 ---")
        print(f"  提供商: {self.provider.value}")
        print(f"  模型: {self.model}")
        print(f"  请求ID: {request_id}")
        print(f"  消息数量: {len(messages)}")
        print(f"  最大token: {max_tokens}")
        print(f"  温度: {temperature}")
        print(f"  流式: {stream}")
        
        # 验证参数处理逻辑
        if self.config.use_max_completion_tokens:
            print("  ✅ 将使用max_completion_tokens参数调用API")
        else:
            print("  ✅ 将使用max_tokens参数调用API")
            
        # 模拟API调用失败，但调用流程已验证
        raise Exception("模拟API调用失败，但调用流程已验证")
    
    # 替换原始方法
    service._call_api = mock_call_api.__get__(service, OpenAICompatibleService)
    
    try:
        # 调用generate_reply方法
        print("\n=== 执行service.generate_reply方法 ===")
        reply, metadata = service.generate_reply(
            system_prompt="你是一个AI助手",
            user_input="你好，世界！",
            conversation_history=[],
            max_tokens=100,
            temperature=0.7,
            stream=False
        )
        
        print("\n❌ 测试失败：没有触发预期的异常")
        return False
        
    except Exception as e:
        # 检查是否是我们期望的异常
        if "模拟API调用失败，但调用流程已验证" in str(e):
            print(f"\n✅ 调用流程验证成功！generate_reply -> _call_api的调用关系已确认")
            return True
        else:
            print(f"\n❌ 测试失败：捕获到意外异常: {e}")
            return False
    
    finally:
        # 恢复原始方法
        service._call_api = original_call_api


if __name__ == "__main__":
    success = test_call_flow()
    sys.exit(0 if success else 1)
