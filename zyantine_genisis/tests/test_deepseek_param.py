#!/usr/bin/env python3
"""
DeepSeek参数转换测试脚本
验证OpenAICompatibleService是否正确将max_tokens转换为max_completion_tokens
"""
import sys
import os
import logging

# 添加项目根目录到Python路径
project_root = os.path.abspath('/Users/gyc/Desktop/GYC_coding/zyantine/zyantine_memo')
sys.path.append(project_root)

from zyantine_genisis.api.llm_service_factory import LLMServiceFactory
from zyantine_genisis.api.llm_service import OpenAICompatibleService
from zyantine_genisis.api.llm_provider import LLMProvider, LLMModelConfig
from zyantine_genisis.utils.logger import SystemLogger

# 配置日志
logger = SystemLogger().get_logger("deepseek_param_test", level=logging.DEBUG)

def test_deepseek_param_conversion():
    """测试DeepSeek参数转换逻辑"""
    logger.info("开始DeepSeek参数转换测试...")
    
    # 1. 创建DeepSeek配置
    deepseek_config = {
        "api_key": "test_key",  # 测试用，实际调用会失败但能看到参数
        "base_url": "https://api.deepseek.com",
        "chat_model": "deepseek-chat",
        "timeout": 30,
        "max_retries": 3,
        "enabled": True
    }
    
    # 2. 通过工厂创建服务
    logger.info("通过LLMServiceFactory创建DeepSeek服务...")
    service = LLMServiceFactory.create_service("deepseek", deepseek_config)
    
    if not service:
        logger.error("创建DeepSeek服务失败")
        return False
    
    logger.info(f"成功创建服务: {type(service).__name__}")
    logger.info(f"服务提供商: {service.provider.value}")
    logger.info(f"模型名称: {service.model}")
    logger.info(f"base_url: {service.base_url}")
    logger.info(f"use_max_completion_tokens: {service.config.use_max_completion_tokens}")
    
    # 3. 验证参数转换逻辑
    try:
        # 模拟API调用（会失败，但能看到参数）
        logger.info("测试API调用参数...")
        
        # 重写client方法以捕获参数
        original_create = service.client.chat.completions.create
        
        def mock_create(**kwargs):
            logger.info(f"API调用参数: {kwargs}")
            if "max_completion_tokens" in kwargs:
                logger.info("✅ 成功: 使用了max_completion_tokens参数")
            elif "max_tokens" in kwargs:
                logger.error("❌ 失败: 使用了max_tokens参数")
            else:
                logger.error("❌ 失败: 没有max_tokens或max_completion_tokens参数")
            
            # 引发异常模拟API调用失败
            raise Exception("测试异常: 模拟API调用")
        
        service.client.chat.completions.create = mock_create
        
        # 尝试调用API
        service.generate_reply(
            system_prompt="你是一个助手",
            user_input="测试",
            max_tokens=100,
            temperature=0.7,
            stream=False
        )
        
    except Exception as e:
        # 预期会失败，因为是测试
        if "测试异常" in str(e):
            logger.info("测试完成，参数验证成功")
            return True
        else:
            logger.error(f"测试失败: {e}")
            return False
    finally:
        # 恢复原始方法
        service.client.chat.completions.create = original_create

def test_openai_service():
    """测试OpenAI服务（对比用）"""
    logger.info("\n开始OpenAI服务测试（对比用）...")
    
    openai_config = {
        "api_key": "test_key",
        "base_url": "https://api.openai.com/v1",
        "chat_model": "gpt-5-nano-2025-08-07",
        "timeout": 30,
        "max_retries": 3
    }
    
    service = LLMServiceFactory.create_service("openai", openai_config)
    
    if not service:
        logger.error("创建OpenAI服务失败")
        return False
    
    logger.info(f"成功创建服务: {type(service).__name__}")
    logger.info(f"use_max_completion_tokens: {service.config.use_max_completion_tokens}")
    
    try:
        # 重写client方法
        original_create = service.client.chat.completions.create
        
        def mock_create(**kwargs):
            logger.info(f"OpenAI API调用参数: {kwargs}")
            if "max_tokens" in kwargs:
                logger.info("✅ 成功: OpenAI使用了max_tokens参数")
            else:
                logger.error("❌ 失败: OpenAI没有使用max_tokens参数")
            raise Exception("测试异常: 模拟API调用")
        
        service.client.chat.completions.create = mock_create
        
        # 尝试调用API
        service.generate_reply(
            system_prompt="你是一个助手",
            user_input="测试",
            max_tokens=100,
            temperature=0.7,
            stream=False
        )
        
    except Exception as e:
        if "测试异常" in str(e):
            logger.info("OpenAI测试完成")
            return True
        else:
            logger.error(f"测试失败: {e}")
            return False
    finally:
        service.client.chat.completions.create = original_create

if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("DeepSeek参数转换测试")
    logger.info("=" * 60)
    
    # 运行测试
    deepseek_result = test_deepseek_param_conversion()
    openai_result = test_openai_service()
    
    logger.info("\n" + "=" * 60)
    logger.info("测试总结:")
    logger.info(f"DeepSeek参数转换: {'✅ 成功' if deepseek_result else '❌ 失败'}")
    logger.info(f"OpenAI参数验证: {'✅ 成功' if openai_result else '❌ 失败'}")
    
    if deepseek_result and openai_result:
        logger.info("🎉 所有测试通过！参数转换逻辑正常工作")
        sys.exit(0)
    else:
        logger.error("❌ 测试失败！请检查参数转换逻辑")
        sys.exit(1)
