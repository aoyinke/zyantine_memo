#!/usr/bin/env python3
"""
测试LLM服务的消息优化功能
"""

from typing import Dict, List, Tuple
import sys
import os

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from api.llm_service import BaseLLMService, APIErrorType
from api.openai_service import OpenAIService
from api.llm_provider import LLMModelConfig, LLMProvider


def test_message_building():
    """测试消息构建功能"""
    print("=== 测试消息构建功能 ===")
    
    # 创建一个简单的LLM服务配置（不实际调用API）
    config = LLMModelConfig(
        provider=LLMProvider.OPENAI,
        model_name="gpt-4o-mini",
        api_key="dummy_key",
        base_url="https://api.openai.com/v1",
        max_context_tokens=3000
    )
    
    # 创建一个测试服务类，继承自BaseLLMService但不实际调用API
    class TestLLMService(BaseLLMService):
        def _initialize_client(self):
            self.client = None
            
        def _call_api(self, messages: List[Dict], max_tokens: int, temperature: float, stream: bool, request_id: str):
            return None
            
        def _extract_content(self, response) -> str:
            return "test response"
            
        def _extract_usage(self, response) -> Tuple[int, int, int]:
            return (10, 20, 30)
            
        def _extract_finish_reason(self, response) -> str:
            return "stop"
    
    # 创建测试服务实例
    service = TestLLMService(config)
    
    # 系统提示词
    system_prompt = "你是一个智能助手，帮助用户回答问题。"
    
    # 当前用户输入
    user_input = "请解释什么是人工智能？"
    
    # 测试用例1：正常对话历史
    print("\n测试用例1：正常对话历史")
    conversation_history1 = [
        {"user_input": "你好", "system_response": "你好！有什么可以帮助你的吗？"},
        {"user_input": "什么是机器学习？", "system_response": "机器学习是人工智能的一个分支，让计算机能够从数据中学习并改进性能。"}
    ]
    
    messages1 = service._build_messages(system_prompt, user_input, conversation_history1)
    print(f"构建的消息数量: {len(messages1)}")
    for i, msg in enumerate(messages1):
        print(f"  {i+1}. {msg['role']}: {msg['content'][:50]}...")
    
    # 验证消息顺序
    roles1 = [msg['role'] for msg in messages1]
    print(f"消息角色顺序: {roles1}")
    
    # 测试用例2：包含空消息的对话历史
    print("\n测试用例2：包含空消息的对话历史")
    conversation_history2 = [
        {"user_input": "你好", "system_response": "你好！有什么可以帮助你的吗？"},
        {"user_input": "", "system_response": "请输入您的问题。"},  # 空用户输入
        {"user_input": "什么是深度学习？", "system_response": ""}  # 空系统响应
    ]
    
    messages2 = service._build_messages(system_prompt, user_input, conversation_history2)
    print(f"构建的消息数量: {len(messages2)}")
    for i, msg in enumerate(messages2):
        print(f"  {i+1}. {msg['role']}: {msg['content'][:50]}...")
    
    roles2 = [msg['role'] for msg in messages2]
    print(f"消息角色顺序: {roles2}")
    
    # 测试用例3：超过token限制的长对话历史
    print("\n测试用例3：超过token限制的长对话历史")
    long_text = "a" * 2000  # 生成一个长文本
    conversation_history3 = []
    for i in range(5):  # 添加5轮长对话
        conversation_history3.append({
            "user_input": f"第{i+1}个问题：{long_text}",
            "system_response": f"第{i+1}个回答：{long_text}"
        })
    
    messages3 = service._build_messages(system_prompt, user_input, conversation_history3)
    print(f"构建的消息数量: {len(messages3)}")
    for i, msg in enumerate(messages3):
        print(f"  {i+1}. {msg['role']}: {msg['content'][:50]}...")
    
    roles3 = [msg['role'] for msg in messages3]
    print(f"消息角色顺序: {roles3}")
    
    # 测试用例4：消息顺序验证（确保没有连续相同角色）
    print("\n测试用例4：消息顺序验证")
    # 创建一个可能导致连续角色的历史
    conversation_history4 = [
        {"user_input": "问题1", "system_response": "回答1"},
        {"user_input": "问题2", "system_response": None},  # 缺少系统响应
        {"user_input": "问题3", "system_response": "回答3"}
    ]
    
    messages4 = service._build_messages(system_prompt, user_input, conversation_history4)
    print(f"构建的消息数量: {len(messages4)}")
    for i, msg in enumerate(messages4):
        print(f"  {i+1}. {msg['role']}: {msg['content'][:50]}...")
    
    roles4 = [msg['role'] for msg in messages4]
    print(f"消息角色顺序: {roles4}")
    
    # 检查是否有连续相同角色
    has_consecutive = False
    for i in range(1, len(roles4)):
        if roles4[i] == roles4[i-1]:
            has_consecutive = True
            print(f"  警告：发现连续相同角色在位置 {i} 和 {i+1}: {roles4[i]}")
    
    if not has_consecutive:
        print("  ✅ 没有发现连续相同角色的消息")
    
    # 测试用例5：验证对话历史的完整性
    print("\n测试用例5：验证对话历史的完整性")
    conversation_history5 = [
        {"user_input": "你好", "system_response": "你好！有什么可以帮助你的吗？"},
        {"user_input": "今天天气怎么样？", "system_response": "今天天气很好，阳光明媚。"},
        {"user_input": "那明天呢？", "system_response": "明天可能会下雨，请记得带伞。"}
    ]
    
    messages5 = service._build_messages(system_prompt, user_input, conversation_history5)
    print(f"构建的消息数量: {len(messages5)}")
    for i, msg in enumerate(messages5):
        print(f"  {i+1}. {msg['role']}: {msg['content'][:50]}...")
    
    # 验证历史对话的顺序是否正确
    if len(messages5) >= 8:  # system + 3轮对话 + current user
        if (messages5[1]['content'] == "你好" and 
            messages5[2]['content'] == "你好！有什么可以帮助你的吗？" and 
            messages5[3]['content'] == "今天天气怎么样？" and 
            messages5[4]['content'] == "今天天气很好，阳光明媚。" and 
            messages5[5]['content'] == "那明天呢？" and 
            messages5[6]['content'] == "明天可能会下雨，请记得带伞。"):
            print("  ✅ 历史对话顺序完全正确")
        else:
            print("  ❌ 历史对话顺序不正确")
    else:
        print("  ❌ 构建的消息数量不足，可能有消息被截断")
    
    print("\n=== 所有测试完成 ===")


if __name__ == "__main__":
    test_message_building()
