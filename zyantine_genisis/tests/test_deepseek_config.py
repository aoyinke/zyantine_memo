#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DeepSeek配置测试脚本
用于验证DeepSeek API的参数转换逻辑是否正确实现
"""

import sys
import os
import json

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 添加zyantine_genisis目录到Python路径
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'zyantine_genisis'))

from zyantine_genisis.config.config_manager import ConfigManager
from zyantine_genisis.api.llm_service_factory import LLMServiceFactory
from zyantine_genisis.api.service_provider import APIServiceProvider
from zyantine_genisis.api.reply_generator import APIBasedReplyGenerator


def test_config_loading():
    """测试配置文件加载是否正确"""
    print("=== 测试配置文件加载 ===")
    
    try:
        config_manager = ConfigManager()
        llm_config = config_manager.load()
        
        print(f"加载的配置文件路径: {config_manager._config_file}")
        print(f"API提供者: {llm_config.api.provider}")
        print(f"DeepSeek启用状态: {llm_config.api.providers.get('deepseek', {}).get('enabled', False)}")
        print(f"use_max_completion_tokens设置: {llm_config.api.providers.get('deepseek', {}).get('use_max_completion_tokens', False)}")
        
        # 验证关键配置
        if llm_config.api.provider == "deepseek":
            print("✅ API提供者正确设置为deepseek")
        else:
            print(f"❌ API提供者设置错误，当前为: {llm_config.api.provider}")
            
        if llm_config.api.providers.get('deepseek', {}).get('use_max_completion_tokens', False):
            print("✅ DeepSeek已启用use_max_completion_tokens")
        else:
            print("❌ DeepSeek未启用use_max_completion_tokens")
            
        return llm_config
        
    except Exception as e:
        print(f"❌ 配置加载失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


def test_service_creation(llm_config):
    """测试服务创建逻辑"""
    print("\n=== 测试服务创建逻辑 ===")
    
    if not llm_config:
        print("没有可用的配置")
        return None
    
    try:
        factory = LLMServiceFactory(llm_config)
        service = factory.create_service()
        
        print(f"创建的服务类型: {type(service).__name__}")
        
        # 检查服务配置
        if hasattr(service, 'config'):
            print(f"服务配置: {service.config}")
            
            if hasattr(service.config, 'use_max_completion_tokens'):
                print(f"服务级use_max_completion_tokens: {service.config.use_max_completion_tokens}")
            
        return service
        
    except Exception as e:
        print(f"❌ 服务创建失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


def test_service_provider(llm_config):
    """测试服务提供者的初始化"""
    print("\n=== 测试服务提供者初始化 ===")
    
    try:
        service_provider = APIServiceProvider(config=llm_config)
        
        print(f"服务提供者类型: {type(service_provider).__name__}")
        print(f"活跃服务: {service_provider.active_service}")
        
        if hasattr(service_provider, 'reply_generator'):
            reply_generator = service_provider.reply_generator
            print(f"ReplyGenerator类型: {type(reply_generator).__name__}")
            
            if hasattr(reply_generator, 'api_service'):
                api_service = reply_generator.api_service
                print(f"注入的API服务类型: {type(api_service).__name__}")
                
                if hasattr(api_service, 'config'):
                    print(f"API服务配置: {api_service.config}")
                    
                    if hasattr(api_service.config, 'use_max_completion_tokens'):
                        print(f"API服务use_max_completion_tokens: {api_service.config.use_max_completion_tokens}")
        
        return service_provider
        
    except Exception as e:
        print(f"❌ 服务提供者初始化失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


def test_param_conversion(service):
    """测试参数转换逻辑"""
    print("\n=== 测试参数转换逻辑 ===")
    
    if not service:
        print("没有可用的服务")
        return
    
    if hasattr(service.config, 'use_max_completion_tokens'):
        use_max_completion = service.config.use_max_completion_tokens
        print(f"use_max_completion_tokens设置: {use_max_completion}")
        
        if use_max_completion:
            print("✅ 应该使用 max_completion_tokens 参数")
        else:
            print("❌ 应该使用 max_tokens 参数")
    else:
        print("❌ 无法确定参数使用方式，配置中没有use_max_completion_tokens属性")


def test_api_compatibility(service):
    """测试API兼容性"""
    print("\n=== 测试API兼容性 ===")
    
    if not service:
        print("没有可用的服务")
        return
    
    # 检查_call_api方法是否存在并能正确处理参数
    if hasattr(service, '_call_api'):
        print("✅ _call_api方法存在")
        
        # 检查_call_api方法的参数处理逻辑
        import inspect
        sig = inspect.signature(service._call_api)
        print(f"_call_api方法签名: {sig}")
        
        # 检查是否有处理max_tokens的逻辑
        method_source = inspect.getsource(service._call_api)
        if 'max_completion_tokens' in method_source:
            print("✅ _call_api方法中包含max_completion_tokens的处理逻辑")
        else:
            print("❌ _call_api方法中不包含max_completion_tokens的处理逻辑")
            print("方法源代码:")
            print(method_source)
    else:
        print("❌ _call_api方法不存在")


def main():
    """主测试函数"""
    print("开始DeepSeek配置测试...\n")
    
    # 测试1: 加载配置
    llm_config = test_config_loading()
    
    # 测试2: 创建服务
    service = test_service_creation(llm_config)
    
    # 测试3: 测试服务提供者
    service_provider = test_service_provider(llm_config)
    
    # 测试4: 验证参数转换逻辑
    if service:
        test_param_conversion(service)
        test_api_compatibility(service)
    
    # 测试5: 验证服务提供者中的服务
    if service_provider and hasattr(service_provider, 'reply_generator'):
        reply_gen = service_provider.reply_generator
        print("\n=== 详细检查API服务 ===")
        print(f"reply_generator属性: {dir(reply_gen)}")
        
        if hasattr(reply_gen, 'api_service'):
            api_service = reply_gen.api_service
            print(f"api_service类型: {type(api_service).__name__}")
            print(f"api_service存在: {api_service is not None}")
            
            if api_service:
                print(f"api_service属性: {dir(api_service)}")
                test_param_conversion(api_service)
                test_api_compatibility(api_service)
        else:
            print("❌ api_service属性不存在")
            
            # 检查是否有其他类似属性
            for attr in dir(reply_gen):
                if 'api' in attr.lower():
                    print(f"找到相关属性: {attr} = {getattr(reply_gen, attr)}")
    
    print("\n=== 测试完成 ===")


if __name__ == "__main__":
    main()
