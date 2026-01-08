#!/usr/bin/env python3
"""
测试脚本：验证系统是否能正确调用DeepSeek API生成回复
"""
import os
import sys
import json

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from zyantine_genisis.facade import create_zyantine

def test_api_call():
    """测试API调用功能"""
    try:
        print("正在创建Zyantine实例...")
        
        # 创建Zyantine实例
        zyantine = create_zyantine(config_file="./zyantine_genisis/config/llm_config.json")
        
        print("Zyantine实例创建成功！")
        print(f"当前LLM提供商: {zyantine.config.api.provider}")
        print(f"当前模型: {zyantine.config.api.chat_model}")
        
        # 测试用户输入
        user_input = "你好，请介绍一下你自己"
        print(f"\n测试用户输入: {user_input}")
        print("正在生成回复...")
        
        # 调用API生成回复
        response = zyantine.process(user_input)
        
        print(f"\n系统回复: {response}")
        print("\n测试完成！")
        
        return True, response
        
    except Exception as e:
        print(f"\n测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False, None

if __name__ == "__main__":
    test_api_call()
