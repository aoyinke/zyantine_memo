#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
缓存禁用验证测试
测试在禁用缓存情况下，相同问题的回答是否不同
"""
import sys
import os

# 添加项目根目录到Python路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from zyantine_genisis.core.system_core import ZyantineCore


def test_no_cache_responses():
    """测试禁用缓存时，相同问题的回答是否不同"""
    print("=== 缓存禁用验证测试 ===")
    
    # 初始化系统，确保缓存禁用
    core = ZyantineCore()
    
    # 检查缓存配置
    status = core.get_system_status()
    cache_config = status['cache_stats']
    print(f"初始缓存状态: {cache_config}")
    
    # 测试问题
    test_question = "你好，今天天气怎么样？"
    responses = []
    
    # 连续发送相同问题3次
    for i in range(3):
        print(f"\n发送第 {i+1} 次请求: {test_question}")
        response = core.process_input(test_question)
        responses.append(response)
        print(f"收到响应 {i+1}: {response[:100]}...")
    
    # 比较3次回答
    all_same = True
    for i in range(1, 3):
        if responses[i] != responses[0]:
            all_same = False
            break
    
    # 检查缓存统计
    status = core.get_system_status()
    cache_stats = status['cache_stats']
    print(f"\n最终缓存状态: {cache_stats}")
    
    # 验证结果
    if not all_same:
        print("\n✅ 测试通过: 相同问题获得了不同的回答，缓存已禁用")
        return True
    else:
        print("\n❌ 测试失败: 相同问题获得了相同的回答，缓存可能未禁用")
        return False
    
    # 检查缓存命中率
    if cache_stats.get('hits', 0) == 0:
        print("✅ 缓存命中率为0，确认缓存已禁用")
    else:
        print(f"❌ 缓存命中率为 {cache_stats.get('hits', 0)}，缓存可能仍在工作")


def test_cache_config():
    """测试缓存配置是否正确禁用"""
    print("\n=== 缓存配置验证测试 ===")
    
    core = ZyantineCore()
    status = core.get_system_status()
    
    # 检查系统配置中的缓存设置
    config = core.config
    print(f"响应缓存启用状态: {config.processing.enable_response_cache}")
    print(f"记忆缓存启用状态: {config.processing.enable_memory_cache}")
    print(f"上下文缓存启用状态: {config.processing.enable_context_cache}")
    
    # 验证所有缓存都已禁用
    if (not config.processing.enable_response_cache and 
        not config.processing.enable_memory_cache and 
        not config.processing.enable_context_cache):
        print("✅ 所有缓存配置已正确禁用")
        return True
    else:
        print("❌ 缓存配置未完全禁用")
        return False


if __name__ == "__main__":
    # 运行测试
    test1_passed = test_no_cache_responses()
    test2_passed = test_cache_config()
    
    print("\n=== 测试总结 ===")
    if test1_passed and test2_passed:
        print("✅ 所有测试通过！缓存机制已成功禁用")
        sys.exit(0)
    else:
        print("❌ 部分测试失败，请检查缓存配置")
        sys.exit(1)