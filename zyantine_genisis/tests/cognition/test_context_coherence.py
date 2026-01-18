#!/usr/bin/env python3
"""
测试上下文连贯性优化效果
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from zyantine_genisis.cognition.context_parser import ContextParser
from zyantine_genisis.memory.memory_store import ZyantineMemorySystem
from zyantine_genisis.memory.memory_manager import MemoryManager


def test_referential_expression_handling():
    """测试指代性表述处理"""
    print("=== 测试指代性表述处理 ===")
    
    # 初始化上下文解析器
    context_parser = ContextParser()
    
    # 模拟体测对话上下文
    conversation_history = [
        {"role": "user", "content": "我今天去做了体测"},
        {"role": "user", "content": "我的肺活量是4500毫升"},
        {"role": "user", "content": "50米跑了7.8秒"},
        {"role": "user", "content": "记得这件事情"},
        {"role": "user", "content": "我刚才说我的肺活量是多少来着？"}
    ]
    
    # 模拟当前输入
    current_input = "我刚才说我的肺活量是多少来着？"
    
    # 解析上下文
    parsed_context = context_parser.parse(
        user_input=current_input,
        history=conversation_history
    )
    
    print(f"当前输入: {current_input}")
    print(f"解析结果: {parsed_context}")
    print(f"检测到指代性表述: {parsed_context['contains_referential']}")
    print(f"检测到的主题: {parsed_context['current_topic']}")
    print(f"主题置信度: {parsed_context['topic_confidence']:.2f}")
    
    # 检查是否正确识别了指代性表述
    if parsed_context['contains_referential'] and parsed_context['current_topic'] == 'fitness':
        print("✅ 指代性表述处理测试通过")
        return True
    else:
        print("❌ 指代性表述处理测试失败")
        return False


def test_context_coherence():
    """测试上下文连贯性"""
    print("\n=== 测试上下文连贯性 ===")
    
    # 初始化上下文解析器
    context_parser = ContextParser()
    
    # 模拟多轮对话
    conversation_history = [
        {"role": "user", "content": "我喜欢阅读"},
        {"role": "assistant", "content": "很好！你喜欢读什么类型的书？"},
        {"role": "user", "content": "我喜欢科幻小说"},
        {"role": "assistant", "content": "科幻小说很有趣，你有什么推荐吗？"},
        {"role": "user", "content": "《三体》系列很不错"},
    ]
    
    # 当前输入
    current_input = "我刚才说的那本书的作者是谁？"
    
    # 解析上下文
    parsed_context = context_parser.parse(
        user_input=current_input,
        history=conversation_history
    )
    
    print(f"当前输入: {current_input}")
    print(f"解析结果: {parsed_context}")
    print(f"检测到指代性表述: {parsed_context['contains_referential']}")
    print(f"检测到的主题: {parsed_context['current_topic']}")
    print(f"主题置信度: {parsed_context['topic_confidence']:.2f}")
    
    # 检查是否正确识别了上下文
    if parsed_context['contains_referential'] and parsed_context['current_topic'] == 'reading':
        print("✅ 上下文连贯性测试通过")
        return True
    else:
        print("❌ 上下文连贯性测试失败")
        return False


def test_memory_retrieval():
    """测试记忆检索"""
    print("\n=== 测试记忆检索 ===")
    
    # 初始化记忆系统
    memory_system = ZyantineMemorySystem(user_id="test_user", session_id="test_session")
    
    # 添加记忆
    memory_id = memory_system.add_memory(
        content="用户喜欢阅读科幻小说，特别是《三体》系列",
        memory_type="preference",
        tags=["reading", "scifi"],
        metadata={"conversation_id": "test_conv_1"}
    )
    
    print(f"添加记忆ID: {memory_id}")
    
    # 搜索记忆
    search_results = memory_system.search_memories(
        query="用户喜欢什么类型的书",
        limit=5
    )
    
    print(f"搜索结果数量: {len(search_results)}")
    for result in search_results:
        print(f"  - {result.get('content', '')[:50]}...")
    
    if search_results and len(search_results) > 0:
        print("✅ 记忆检索测试通过")
        return True
    else:
        print("❌ 记忆检索测试失败")
        return False


if __name__ == "__main__":
    print("开始上下文连贯性测试\n")
    
    results = []
    results.append(test_referential_expression_handling())
    results.append(test_context_coherence())
    results.append(test_memory_retrieval())
    
    print("\n" + "=" * 50)
    print(f"测试结果: {sum(results)}/{len(results)} 通过")
    print("=" * 50)
