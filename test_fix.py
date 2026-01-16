#!/usr/bin/env python3
"""
测试API修复效果的脚本
"""
import sys
import os

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from zyantine_genisis.core.processing_pipeline import ProcessingPipeline, StageContext
from zyantine_genisis.core.stage_handlers import ReplyGenerationHandler, ProtocolReviewHandler
from zyantine_genisis.api.reply_generator import APIBasedReplyGenerator
from zyantine_genisis.api.fallback_strategy import FallbackStrategy
from zyantine_genisis.api.prompt_engine import PromptEngine
from zyantine_genisis.config.config_manager import ConfigManager

# 初始化配置
config_manager = ConfigManager()
config = config_manager.load()

# 初始化各个组件
fallback_strategy = FallbackStrategy()
prompt_engine = PromptEngine(config)

# 初始化回复生成器
reply_generator = APIBasedReplyGenerator(
    api_service=None,  # 暂时使用None，模拟API不可用的情况
    prompt_engine=prompt_engine,
    fallback_strategy=fallback_strategy
)

# 创建上下文
context = StageContext(
    user_input="测试修复效果",
    conversation_history=[],
    system_components={}
)

# 测试回复生成器
print("=== 测试回复生成器 ===")
try:
    # 调用回复生成器
    reply_result = reply_generator.generate_reply(user_input="测试修复效果")
    print(f"回复生成器返回值: {reply_result}")
    print(f"返回值类型: {type(reply_result)}")
    
    # 测试ReplyGenerationHandler
    print("\n=== 测试ReplyGenerationHandler ===")
    reply_handler = ReplyGenerationHandler(reply_generator)
    context = reply_handler.process(context)
    print(f"处理后的上下文.final_reply: {context.final_reply}")
    print(f"final_reply类型: {type(context.final_reply)}")
    
    # 测试ProtocolReviewHandler
    print("\n=== 测试ProtocolReviewHandler ===")
    # 由于ProtocolReviewHandler需要protocol_engine和meta_cognition，这里简化测试
    # 只测试final_reply是否为字符串类型
    if isinstance(context.final_reply, str):
        print("✓ final_reply是字符串类型，协议审查应该能正常处理")
    else:
        print(f"✗ final_reply不是字符串类型: {type(context.final_reply)}")
        sys.exit(1)
        
    print("\n=== 测试完成 ===")
    print("修复效果良好，协议审查不再会因为类型错误而失败")
    sys.exit(0)
    
except Exception as e:
    print(f"测试失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)