#!/usr/bin/env python3
"""
测试空响应修复功能
"""

import sys
import os
import json
from unittest.mock import Mock, patch

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from api.openai_service import OpenAIService
from api.reply_generator import APIBasedReplyGenerator
from api.prompt_engine import PromptEngine
from api.fallback_strategy import FallbackStrategy
from config.config_manager import ConfigManager
from utils.logger import SystemLogger
from utils.metrics import MetricsCollector


def test_empty_response_handling():
    """测试空响应处理"""
    print("=== 测试空响应修复功能 ===")
    
    # 加载配置
    config_manager = ConfigManager()
    config = config_manager.load("config/llm_config.json")
    
    # 创建OpenAIService实例
    api_service = OpenAIService(
        api_key="test_key",
        base_url="https://openkey.cloud/v1",
        model="gpt-5-nano-2025-08-07"
    )
    
    # 创建必要的组件
    prompt_engine = PromptEngine(config)
    fallback_strategy = FallbackStrategy()
    metrics_collector = MetricsCollector("test_metrics")
    
    # 创建ReplyGenerator实例
    reply_generator = APIBasedReplyGenerator(
        api_service=api_service,
        prompt_engine=prompt_engine,
        fallback_strategy=fallback_strategy,
        metrics_collector=metrics_collector
    )
    
    # 使用mock模拟API返回空响应
    mock_response = Mock()
    mock_response.choices = [Mock()]
    mock_response.choices[0].message.content = ""
    mock_response.usage = Mock()
    mock_response.usage.prompt_tokens = 10
    mock_response.usage.completion_tokens = 0
    mock_response.usage.total_tokens = 10
    
    with patch.object(api_service, 'client') as mock_client:
        mock_client.chat.completions.create.return_value = mock_response
        
        print("\n测试1: 模拟API返回空响应")
        try:
            reply, metadata = api_service.generate_reply(
                system_prompt="测试系统提示",
                user_input="测试用户输入"
            )
            
            print(f"✓ API服务返回值: reply={reply}, metadata={metadata}")
            if reply is None:
                print("✓ 成功检测到空响应并返回None")
            else:
                print(f"✗ 错误: 应该返回None但返回了'{reply}'")
                return False
        except Exception as e:
            print(f"✗ 测试失败: {e}")
            return False
    
    print("\n测试2: 验证reply_generator处理空响应")
    with patch.object(api_service, 'generate_reply') as mock_generate:
        mock_generate.return_value = ("", {"tokens_used": 10, "latency": 0.5})
        
        try:
            # 模拟生成上下文的调用
            context = type('obj', (object,), {
                'user_input': '测试用户输入',
                'action_plan': {"chosen_mask": "长期搭档", "primary_strategy": "直接回答"},
                'growth_result': {},
                'context_analysis': {"topic_complexity": "low"},
                'conversation_history': [],
                'core_identity': None,
                'current_vectors': {"TR": 0.5, "CS": 0.5, "SA": 0.5},
                'memory_context': None,
                'metadata': None
            })
            
            reply = reply_generator._generate_with_api(context)
            
            print(f"✓ ReplyGenerator返回值: '{reply}'")
            if reply is not None and reply != "":
                print("✓ 成功使用降级策略生成回复")
            else:
                print(f"✗ 错误: 降级策略没有生效，返回了'{reply}'")
                return False
                
        except Exception as e:
            print(f"✗ 测试失败: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    print("\n=== 所有测试通过! ===")
    return True


if __name__ == "__main__":
    success = test_empty_response_handling()
    sys.exit(0 if success else 1)
