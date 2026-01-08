#!/usr/bin/env python3
"""
简单测试DeepSeek API调用功能，绕过项目导入链问题
"""

import sys
import json
import os
from openai import OpenAI

def test_deepseek_api():
    """直接测试DeepSeek API调用"""
    print("=== 直接测试DeepSeek API调用 ===")
    
    # 从配置文件获取API信息
    config_path = './zyantine_genisis/config/llm_config.json'
    if not os.path.exists(config_path):
        print(f"✗ 配置文件不存在: {config_path}")
        return False
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    api_config = config.get('api', {})
    provider = api_config.get('provider')
    api_key = api_config.get('api_key')
    base_url = api_config.get('base_url')
    model = api_config.get('chat_model')
    
    print(f"配置信息：")
    print(f"  Provider: {provider}")
    print(f"  Model: {model}")
    print(f"  Base URL: {base_url}")
    
    if not api_key:
        print("✗ 未找到API密钥")
        return False
    
    try:
        # 直接使用OpenAI客户端测试DeepSeek API
        client = OpenAI(
            api_key=api_key,
            base_url=base_url
        )
        
        print("\n正在测试API调用...")
        
        # 根据provider选择正确的参数
        if provider == "deepseek":
            # DeepSeek需要使用max_completion_tokens
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": "你好，这是一个测试，简单介绍一下自己"}],
                max_completion_tokens=100,
                temperature=0.7
            )
        else:
            # 其他provider使用max_tokens
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": "你好，这是一个测试，简单介绍一下自己"}],
                max_tokens=100,
                temperature=0.7
            )
        
        print(f"✓ API调用成功!")
        print(f"响应内容：\n{response.choices[0].message.content}")
        
        # 检查响应信息
        print(f"\n响应状态：")
        print(f"  模型：{response.model}")
        print(f"  完成原因：{response.choices[0].finish_reason}")
        
        return True
        
    except Exception as e:
        print(f"\n✗ 测试失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_deepseek_api()
    sys.exit(0 if success else 1)
