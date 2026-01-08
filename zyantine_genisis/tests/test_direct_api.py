#!/usr/bin/env python3
"""
直接测试脚本：绕过包初始化，直接导入必要的模块
"""
import os
import sys
import json

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

# 直接导入必要的模块，避免触发__init__.py中的导入链
sys.modules['zyantine_genisis'] = type('ZyantineGenisis', (), {})()

# 导入API相关模块
from zyantine_genisis.api.llm_service_factory import LLMServiceFactory
from zyantine_genisis.api.config_manager import ConfigManager

def test_direct_api():
    """直接测试LLMService的API调用功能"""
    try:
        print("正在加载配置...")
        
        # 加载配置
        config_manager = ConfigManager(config_file="./zyantine_genisis/config/llm_config.json")
        config = config_manager.load_config()
        
        print(f"配置加载成功！")
        print(f"当前提供商: {config.api.provider}")
        print(f"当前模型: {config.api.chat_model}")
        
        # 获取提供商配置
        provider = config.api.provider
        provider_config = config.api.providers.get(provider, {})
        
        # 确保use_max_completion_tokens设置正确
        if provider == "deepseek":
            provider_config["use_max_completion_tokens"] = True
        
        print(f"\n正在创建{provider}服务实例...")
        print(f"提供商配置: {json.dumps(provider_config, indent=2)}")
        
        # 创建服务实例
        service = LLMServiceFactory.create_service(provider, provider_config)
        
        if not service:
            print(f"创建{provider}服务实例失败！")
            return False
        
        print(f"{provider}服务实例创建成功！")
        
        # 测试API调用
        system_prompt = "你是一个AI助手，负责回答用户的问题。"
        user_input = "你好，请介绍一下你自己"
        
        print(f"\n测试用户输入: {user_input}")
        print("正在调用API生成回复...")
        
        # 调用API
        reply, metadata = service.generate_reply(
            system_prompt=system_prompt,
            user_input=user_input,
            max_tokens=500,
            temperature=0.7,
            stream=False
        )
        
        if reply:
            print(f"\nAPI调用成功！")
            print(f"回复内容: {reply}")
            if metadata:
                print(f"\n元数据: {json.dumps(metadata, indent=2, ensure_ascii=False)}")
            return True
        else:
            print(f"\nAPI调用失败，未返回回复内容")
            return False
            
    except Exception as e:
        print(f"\n测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_direct_api()
