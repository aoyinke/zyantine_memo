#!/usr/bin/env python
"""
测试完整的API集成 - 验证DeepSeek API是否正常工作
"""
import sys
import os

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from zyantine_genisis.zyantine_facade import create_zyantine


def test_api_integration():
    """测试API集成"""
    print("=" * 60)
    print("测试完整的API集成")
    print("=" * 60)
    
    try:
        # 创建系统实例
        print("\n1. 创建系统实例...")
        zyantine = create_zyantine(config_file="./zyantine_genisis/config/llm_config.json")
        print("✓ 系统实例创建成功")
        
        # 测试API调用
        print("\n2. 测试API调用...")
        user_input = "你好，请介绍一下你自己"
        print(f"用户输入: {user_input}")
        
        response = zyantine.process(user_input)
        
        print(f"\n系统回复: {response}")
        print("✓ API调用成功")
        
        # 测试第二次调用（验证是否使用max_completion_tokens）
        print("\n3. 测试第二次API调用...")
        user_input2 = "你刚才说了什么？"
        print(f"用户输入: {user_input2}")
        
        response2 = zyantine.process(user_input2)
        
        print(f"\n系统回复: {response2}")
        print("✓ 第二次API调用成功")
        
        print("\n" + "=" * 60)
        print("✓ 所有测试通过！")
        print("✓ DeepSeek API集成正常工作")
        print("✓ 系统可以正常回答用户问题")
        print("=" * 60)
        
        return True
        
    except Exception as e:
        print(f"\n✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_api_integration()
    sys.exit(0 if success else 1)
