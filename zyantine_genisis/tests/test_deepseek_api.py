import sys
import os
from openai import OpenAI

# 测试DeepSeek API调用
def test_deepseek_api():
    # 配置DeepSeek API
    client = OpenAI(
        api_key="sk-cd83b6411207408e8539b7623a1c5f35",
        base_url="https://api.deepseek.com"
    )
    
    print("测试DeepSeek API调用...")
    
    try:
        # 使用max_completion_tokens参数
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "user", "content": "你好，这是一个测试，请简单介绍一下自己"}],
            max_completion_tokens=100,
            temperature=0.7
        )
        
        print(f"API调用成功！")
        print(f"响应内容: {response.choices[0].message.content}")
        print(f"使用的模型: {response.model}")
        print(f"消耗的tokens: 提示词 {response.usage.prompt_tokens}, 回复 {response.usage.completion_tokens}, 总计 {response.usage.total_tokens}")
        
        return True
    except Exception as e:
        print(f"API调用失败: {e}")
        return False

if __name__ == "__main__":
    test_deepseek_api()
