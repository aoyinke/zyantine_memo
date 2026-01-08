#!/usr/bin/env python3
"""
测试DeepSeek服务修复效果 - 验证是否正确使用max_completion_tokens参数
"""
import sys
import os
import logging
from unittest.mock import patch, MagicMock

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from zyantine_genisis.api.llm_service_factory import LLMServiceFactory
from zyantine_genisis.api.llm_service import OpenAICompatibleService
from zyantine_genisis.config.config_manager import ConfigManager
from zyantine_genisis.api.service_provider import APIServiceProvider

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_deepseek_service_creation():
    """
    测试DeepSeek服务是否正确创建
    """
    logger.info("=== 测试DeepSeek服务创建 ===")
    
    # 创建DeepSeek配置
    deepseek_config = {
        "enabled": True,
        "api_key": "test_key",
        "base_url": "https://api.deepseek.com",
        "chat_model": "deepseek-chat",
        "timeout": 30,
        "max_retries": 3
    }
    
    # 使用工厂创建服务
    service = LLMServiceFactory.create_service("deepseek", deepseek_config)
    
    assert service is not None, "DeepSeek服务创建失败"
    assert isinstance(service, OpenAICompatibleService), "DeepSeek服务不是OpenAICompatibleService类型"
    assert service.config.use_max_completion_tokens is True, "DeepSeek服务没有启用max_completion_tokens"
    
    logger.info("✅ DeepSeek服务创建测试通过")
    logger.info(f"   - 服务类型: {type(service).__name__}")
    logger.info(f"   - use_max_completion_tokens: {service.config.use_max_completion_tokens}")

def test_deepseek_api_call():
    """
    测试DeepSeek API调用是否使用max_completion_tokens参数
    """
    logger.info("=== 测试DeepSeek API调用 ===")
    
    # 创建DeepSeek配置
    deepseek_config = {
        "enabled": True,
        "api_key": "test_key",
        "base_url": "https://api.deepseek.com",
        "chat_model": "deepseek-chat",
        "timeout": 30,
        "max_retries": 3
    }
    
    # 创建服务
    service = LLMServiceFactory.create_service("deepseek", deepseek_config)
    
    # 模拟客户端和API调用
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.choices = [MagicMock(message=MagicMock(content="Test response"))]
    mock_client.chat.completions.create.return_value = mock_response
    
    # 替换服务的客户端
    service.client = mock_client
    
    # 调用API
    messages = [{"role": "user", "content": "Hello, DeepSeek!"}]
    service.generate_reply(
        system_prompt="You are a helpful assistant.",
        user_input="Hello, DeepSeek!",
        conversation_history=[],
        max_tokens=500,
        temperature=0.7,
        stream=False
    )
    
    # 检查API调用参数
    call_args = mock_client.chat.completions.create.call_args
    kwargs = call_args.kwargs
    
    assert "max_completion_tokens" in kwargs, "API调用没有使用max_completion_tokens参数"
    assert "max_tokens" not in kwargs, "API调用不应该使用max_tokens参数"
    assert kwargs["max_completion_tokens"] == 500, "max_completion_tokens参数值不正确"
    
    logger.info("✅ DeepSeek API调用测试通过")
    logger.info(f"   - 使用的参数: {list(kwargs.keys())}")
    logger.info(f"   - max_completion_tokens值: {kwargs['max_completion_tokens']}")

def test_service_provider_deepseek():
    """
    测试服务提供者是否正确使用DeepSeek服务
    """
    logger.info("=== 测试服务提供者DeepSeek集成 ===")
    
    # 创建配置管理器
    config_manager = ConfigManager()
    config = config_manager.get()
    
    # 修改配置使用DeepSeek
    config.api.provider = "deepseek"
    config.api.providers["deepseek"]["enabled"] = True
    config.api.providers["deepseek"]["api_key"] = "test_key"
    
    # 创建服务提供者
    with patch('zyantine_genisis.api.llm_service_factory.OpenAICompatibleService') as mock_service_class:
        # 设置mock返回值
        mock_service = MagicMock()
        mock_service.generate_reply.return_value = "Test response"
        mock_service_class.return_value = mock_service
        
        # 创建服务提供者
        service_provider = APIServiceProvider(config)
        
        # 验证服务创建
        assert "deepseek" in service_provider.services, "服务提供者没有创建DeepSeek服务"
        
        logger.info("✅ 服务提供者DeepSeek集成测试通过")
        logger.info(f"   - 活跃服务: {service_provider.active_service}")
        logger.info(f"   - 可用服务: {list(service_provider.services.keys())}")

def test_config_manager_deepseek_enabled():
    """
    测试配置管理器中的DeepSeek配置是否启用
    """
    logger.info("=== 测试配置管理器DeepSeek配置 ===")
    
    # 创建配置管理器
    config_manager = ConfigManager()
    config = config_manager.get()
    
    # 检查DeepSeek配置
    deepseek_config = config.api.providers.get("deepseek", {})
    
    assert deepseek_config.get("enabled") is True, "DeepSeek配置未启用"
    
    logger.info("✅ 配置管理器DeepSeek配置测试通过")
    logger.info(f"   - DeepSeek启用状态: {deepseek_config.get('enabled')}")
    logger.info(f"   - DeepSeek模型: {deepseek_config.get('chat_model')}")

def run_all_tests():
    """
    运行所有测试
    """
    logger.info("开始运行DeepSeek修复验证测试...")
    
    tests = [
        test_config_manager_deepseek_enabled,
        test_deepseek_service_creation,
        test_deepseek_api_call,
        test_service_provider_deepseek
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            passed += 1
        except AssertionError as e:
            logger.error(f"❌ {test.__name__} 失败: {e}")
            failed += 1
        except Exception as e:
            logger.error(f"❌ {test.__name__} 出错: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
    
    logger.info("\n=== 测试总结 ===")
    logger.info(f"总测试数: {len(tests)}")
    logger.info(f"通过: {passed}")
    logger.info(f"失败: {failed}")
    
    if failed == 0:
        logger.info("🎉 所有测试通过! DeepSeek修复验证成功")
        return True
    else:
        logger.error("💥 有测试失败，修复需要进一步检查")
        return False

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
