#!/usr/bin/env python3
"""
测试日志功能的脚本
"""
import sys
import os

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.abspath('/Users/gyc/Desktop/GYC_coding/zyantine/zyantine_memo'))

from zyantine_genisis.core.system_core import ZyantineCore

def test_logging():
    """测试日志功能"""
    try:
        # 初始化系统核心
        print("初始化系统核心...")
        core = ZyantineCore()
        
        # 测试不同类型的用户输入
        test_inputs = [
            "你好，我是测试用户",
            "今天天气怎么样？",
            "你能告诉我一些关于人工智能的信息吗？"
        ]
        
        for i, user_input in enumerate(test_inputs):
            print(f"\n=== 测试输入 {i+1}: {user_input} ===")
            response = core.process_input(user_input)
            print(f"系统响应: {response[:100]}...")
        
        print("\n测试完成！")
        
    except Exception as e:
        print(f"测试过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_logging()
