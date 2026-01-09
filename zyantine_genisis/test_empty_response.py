#!/usr/bin/env python3
"""
测试API空响应处理
验证当API返回空响应时，系统是否能正确触发降级策略
"""
import sys
import os
import json
import logging
from unittest.mock import MagicMock, patch

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 导入相关模块
from api.openai_service import OpenAIService
from api.reply_generator import APIBasedReplyGenerator
from api.fallback_strategy import FallbackStrategy
from api.prompt_engine import PromptEngine
from utils.logger import get_logger

# 配置日志
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = get_logger("test_empty_response")

def test_openai_service_empty_response():
    """测试OpenAIService对空响应的处理"""
    logger.info("=== 测试OpenAIService空响应处理 ===")
    
    # 创建OpenAIService实例
    openai_service = OpenAIService(
        api_key="test_key",
        base_url="https://api.openai.com/v1",
        model="gpt-3.5-turbo"
    )
    
    # 模拟API返回空响应
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = ""
    mock_response.usage = MagicMock()
    mock_response.usage.total_tokens = 10
    mock_response.usage.prompt_tokens = 5
    mock_response.usage.completion_tokens = 5
    
    # 测试generate_reply方法
    with patch.object(openai_service.client.chat.completions, 'create', return_value=mock_response):
        reply, metadata = openai_service.generate_reply(
            system_prompt="测试系统提示",
            user_input="测试用户输入",
            conversation_history=None,
            max_tokens=500,
            temperature=0.7,
            stream=False
        )
    
    logger.info(f"OpenAIService返回: reply={reply}, metadata={metadata}")
    
    # 验证结果
    assert reply is None, f"期望返回None，但得到了: {reply}"
    assert metadata is None, f"期望返回None，但得到了: {metadata}"
    
    logger.info("✅ OpenAIService空响应处理测试通过")

def test_reply_generator_fallback():
    """测试ReplyGenerator在API返回空时的降级策略"""
    logger.info("\n=== 测试ReplyGenerator降级策略 ===")
    
    # 创建模拟的API服务
    mock_api_service = MagicMock()
    mock_api_service.is_available.return_value = True
    mock_api_service.generate_reply.return_value = (None, None)  # 模拟API返回空
    
    # 创建FallbackStrategy和PromptEngine实例
    fallback_strategy = FallbackStrategy()
    # 为PromptEngine提供必要的config参数
    mock_config = MagicMock()
    prompt_engine = PromptEngine(config=mock_config)
    
    # 创建ReplyGenerator实例
    reply_generator = APIBasedReplyGenerator(
        api_service=mock_api_service,
        prompt_engine=prompt_engine,
        fallback_strategy=fallback_strategy
    )
    
    # 测试生成回复
    test_context = {
        "user_input": "你好，今天天气怎么样？",
        "action_plan": {"chosen_mask": "知己", "primary_strategy": "empathy"},
        "growth_result": {},
        "context_analysis": {"topic_complexity": "low"},
        "conversation_history": [],
        "current_vectors": {"TR": 0.5, "CS": 0.5, "SA": 0.5},
        "memory_context": None
    }
    
    reply = reply_generator._generate_with_legacy_api(**test_context)
    
    logger.info(f"ReplyGenerator生成的回复: {reply}")
    
    # 验证结果
    assert reply is not None, "期望生成非空回复"
    assert reply != "", "期望生成非空字符串"
    
    logger.info(f"✅ ReplyGenerator降级策略测试通过，生成了有效回复: {reply}")
    return reply

def test_integration_empty_response():
    """测试完整流程对空响应的处理"""
    logger.info("\n=== 测试完整流程空响应处理 ===")
    
    # 创建模拟的API服务
    mock_api_service = MagicMock()
    mock_api_service.is_available.return_value = True
    mock_api_service.generate_reply.return_value = (None, None)  # 模拟API返回空
    
    # 创建FallbackStrategy和PromptEngine实例
    fallback_strategy = FallbackStrategy()
    # 为PromptEngine提供必要的config参数
    mock_config = MagicMock()
    prompt_engine = PromptEngine(config=mock_config)
    
    # 创建ReplyGenerator实例
    reply_generator = APIBasedReplyGenerator(
        api_service=mock_api_service,
        prompt_engine=prompt_engine,
        fallback_strategy=fallback_strategy
    )
    
    # 测试从认知流程生成回复
    cognitive_result = {
        "final_action_plan": {"chosen_mask": "长期搭档", "primary_strategy": "analysis"},
        "growth_result": {},
        "context_analysis": {"topic_complexity": "medium"},
        "user_input": "我遇到了一个技术问题，能帮我分析一下吗？",
        "conversation_history": [],
        "current_vectors": {"TR": 0.6, "CS": 0.4, "SA": 0.3},
        "memory_context": None,
        "flow_id": "test-flow-123"
    }
    
    reply = reply_generator.generate_reply(cognitive_result=cognitive_result)
    
    logger.info(f"完整流程生成的回复: {reply}")
    
    # 验证结果
    assert reply is not None, "期望生成非空回复"
    assert reply != "", "期望生成非空字符串"
    
    logger.info(f"✅ 完整流程空响应处理测试通过，生成了有效回复: {reply}")
    return reply

if __name__ == "__main__":
    try:
        logger.info("开始测试API空响应处理机制")
        
        test_openai_service_empty_response()
        test_reply_generator_fallback()
        test_integration_empty_response()
        
        logger.info("\n🎉 所有测试通过！API空响应处理机制工作正常")
        sys.exit(0)
        
    except AssertionError as e:
        logger.error(f"❌ 测试失败: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"❌ 测试过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
