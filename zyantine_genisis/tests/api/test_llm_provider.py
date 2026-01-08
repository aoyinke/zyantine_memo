"""
测试 LLM 提供商配置
测试 DeepSeek 作为问答模型，OpenAI 作为 Embedding 模型
"""
import unittest
import os
import sys
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from api.llm_provider import LLMProvider, LLMModelConfig
from api.llm_service import BaseLLMService, OpenAICompatibleService
from api.llm_service_factory import LLMServiceFactory
from config.config_manager import ConfigManager, APIConfig


class TestLLMProvider(unittest.TestCase):
    """测试 LLM 提供商枚举和配置"""

    def test_llm_provider_enum(self):
        """测试 LLM 提供商枚举"""
        self.assertEqual(LLMProvider.OPENAI.value, "openai")
        self.assertEqual(LLMProvider.DEEPSEEK.value, "deepseek")
        self.assertEqual(LLMProvider.ANTHROPIC.value, "anthropic")
        self.assertEqual(LLMProvider.ZHIPU.value, "zhipu")
        self.assertEqual(LLMProvider.MOONSHOT.value, "moonshot")
        self.assertEqual(LLMProvider.ALIBABA.value, "alibaba")
        self.assertEqual(LLMProvider.BAIDU.value, "baidu")
        self.assertEqual(LLMProvider.AZURE.value, "azure")

    def test_llm_provider_from_string(self):
        """测试从字符串创建枚举"""
        self.assertEqual(LLMProvider("openai"), LLMProvider.OPENAI)
        self.assertEqual(LLMProvider("deepseek"), LLMProvider.DEEPSEEK)
        self.assertEqual(LLMProvider("anthropic"), LLMProvider.ANTHROPIC)

    def test_llm_model_config(self):
        """测试 LLM 模型配置"""
        config = LLMModelConfig(
            provider=LLMProvider.DEEPSEEK,
            model_name="deepseek-chat",
            api_key="test-api-key",
            base_url="https://api.deepseek.com",
            timeout=30,
            max_retries=3,
            temperature=0.7,
            max_tokens=2000,
            enabled=True
        )

        self.assertEqual(config.provider, LLMProvider.DEEPSEEK)
        self.assertEqual(config.model_name, "deepseek-chat")
        self.assertEqual(config.api_key, "test-api-key")
        self.assertEqual(config.base_url, "https://api.deepseek.com")
        self.assertEqual(config.timeout, 30)
        self.assertEqual(config.max_retries, 3)
        self.assertEqual(config.temperature, 0.7)
        self.assertEqual(config.max_tokens, 2000)
        self.assertTrue(config.enabled)


class TestLLMServiceFactory(unittest.TestCase):
    """测试 LLM 服务工厂"""

    def setUp(self):
        """测试前准备"""
        LLMServiceFactory.clear_cache()

    def tearDown(self):
        """测试后清理"""
        LLMServiceFactory.clear_cache()

    def test_create_deepseek_service(self):
        """测试创建 DeepSeek 服务"""
        config = {
            "enabled": True,
            "api_key": "test-deepseek-key",
            "base_url": "https://api.deepseek.com",
            "chat_model": "deepseek-chat",
            "timeout": 30,
            "max_retries": 3,
            "temperature": 0.7,
            "max_tokens": 2000
        }

        service = LLMServiceFactory.create_service("deepseek", config)

        self.assertIsNotNone(service)
        self.assertIsInstance(service, BaseLLMService)
        self.assertIsInstance(service, OpenAICompatibleService)
        self.assertEqual(service.provider, LLMProvider.DEEPSEEK)
        self.assertEqual(service.model, "deepseek-chat")
        self.assertEqual(service.api_key, "test-deepseek-key")
        self.assertEqual(service.base_url, "https://api.deepseek.com")

    def test_create_openai_service(self):
        """测试创建 OpenAI 服务"""
        config = {
            "enabled": True,
            "api_key": "test-openai-key",
            "base_url": "https://openkey.cloud/v1",
            "chat_model": "gpt-4o-mini",
            "timeout": 30,
            "max_retries": 3
        }

        service = LLMServiceFactory.create_service("openai", config)

        self.assertIsNotNone(service)
        self.assertIsInstance(service, BaseLLMService)
        self.assertIsInstance(service, OpenAICompatibleService)
        self.assertEqual(service.provider, LLMProvider.OPENAI)
        self.assertEqual(service.model, "gpt-4o-mini")
        self.assertEqual(service.api_key, "test-openai-key")
        self.assertEqual(service.base_url, "https://openkey.cloud/v1")

    def test_create_from_preset_deepseek(self):
        """测试从预设创建 DeepSeek 服务"""
        service = LLMServiceFactory.create_from_preset(
            provider="deepseek",
            api_key="test-deepseek-key",
            model="deepseek-chat"
        )

        self.assertIsNotNone(service)
        self.assertEqual(service.provider, LLMProvider.DEEPSEEK)
        self.assertEqual(service.model, "deepseek-chat")
        self.assertEqual(service.api_key, "test-deepseek-key")
        self.assertEqual(service.base_url, "https://api.deepseek.com")

    def test_create_from_preset_openai(self):
        """测试从预设创建 OpenAI 服务"""
        service = LLMServiceFactory.create_from_preset(
            provider="openai",
            api_key="test-openai-key",
            model="gpt-4o-mini"
        )

        self.assertIsNotNone(service)
        self.assertEqual(service.provider, LLMProvider.OPENAI)
        self.assertEqual(service.model, "gpt-4o-mini")
        self.assertEqual(service.api_key, "test-openai-key")
        self.assertEqual(service.base_url, "https://api.openai.com/v1")

    def test_service_caching(self):
        """测试服务缓存"""
        config = {
            "enabled": True,
            "api_key": "test-key",
            "base_url": "https://api.test.com",
            "chat_model": "test-model"
        }

        service1 = LLMServiceFactory.get_or_create_service("deepseek", config)
        service2 = LLMServiceFactory.get_or_create_service("deepseek", config)

        self.assertIs(service1, service2)

    def test_clear_cache(self):
        """测试清空缓存"""
        config = {
            "enabled": True,
            "api_key": "test-key",
            "base_url": "https://api.test.com",
            "chat_model": "test-model"
        }

        service1 = LLMServiceFactory.create_service("deepseek", config)
        LLMServiceFactory.clear_cache()
        service2 = LLMServiceFactory.create_service("deepseek", config)

        self.assertIsNot(service1, service2)

    def test_get_service_stats(self):
        """测试获取服务统计"""
        config = {
            "enabled": True,
            "api_key": "test-key",
            "base_url": "https://api.test.com",
            "chat_model": "test-model"
        }

        service = LLMServiceFactory.create_service("deepseek", config)
        stats = LLMServiceFactory.get_service_statistics()

        self.assertIsNotNone(stats)
        self.assertIsInstance(stats, dict)

    def test_create_invalid_provider(self):
        """测试创建无效提供商"""
        config = {
            "enabled": True,
            "api_key": "test-key",
            "base_url": "https://api.test.com",
            "chat_model": "test-model"
        }

        service = LLMServiceFactory.create_service("invalid_provider", config)
        self.assertIsNone(service)


class TestAPIConfig(unittest.TestCase):
    """测试 API 配置"""

    def test_default_config(self):
        """测试默认配置"""
        config = APIConfig()

        self.assertTrue(config.enabled)
        self.assertEqual(config.provider, "openai")
        self.assertEqual(config.base_url, "https://openkey.cloud/v1")
        self.assertEqual(config.embedding_model, "text-embedding-3-large")
        self.assertEqual(config.chat_model, "gpt-4.1-nano-2025-04-14")

    def test_deepseek_config(self):
        """测试 DeepSeek 配置"""
        config_dict = {
            "enabled": True,
            "provider": "deepseek",
            "api_key": "test-deepseek-key",
            "base_url": "https://api.deepseek.com",
            "chat_model": "deepseek-chat",
            "embedding_model": "text-embedding-3-large",
            "timeout": 30,
            "max_retries": 3
        }

        config = APIConfig(**config_dict)

        self.assertEqual(config.provider, "deepseek")
        self.assertEqual(config.api_key, "test-deepseek-key")
        self.assertEqual(config.base_url, "https://api.deepseek.com")
        self.assertEqual(config.chat_model, "deepseek-chat")
        self.assertEqual(config.embedding_model, "text-embedding-3-large")

    def test_openai_config_with_custom_base_url(self):
        """测试 OpenAI 配置使用自定义 base_url"""
        config_dict = {
            "enabled": True,
            "provider": "openai",
            "api_key": "test-openai-key",
            "base_url": "https://openkey.cloud/v1",
            "chat_model": "gpt-5-nano-2025-08-07",
            "embedding_model": "text-embedding-3-large"
        }

        config = APIConfig(**config_dict)

        self.assertEqual(config.provider, "openai")
        self.assertEqual(config.base_url, "https://openkey.cloud/v1")
        self.assertEqual(config.embedding_model, "text-embedding-3-large")

    def test_providers_config(self):
        """测试多提供商配置"""
        config = APIConfig()

        self.assertIn("openai", config.providers)
        self.assertIn("deepseek", config.providers)
        self.assertIn("anthropic", config.providers)

        self.assertEqual(config.providers["deepseek"]["base_url"], "https://api.deepseek.com")
        self.assertEqual(config.providers["deepseek"]["chat_model"], "deepseek-chat")


class TestConfigManager(unittest.TestCase):
    """测试配置管理器"""

    def setUp(self):
        """测试前准备"""
        self.config_file = "/tmp/test_llm_config.json"
        self.test_config = {
            "api": {
                "enabled": True,
                "provider": "deepseek",
                "api_key": "test-deepseek-key",
                "base_url": "https://api.deepseek.com",
                "chat_model": "deepseek-chat",
                "embedding_model": "text-embedding-3-large",
                "timeout": 30,
                "max_retries": 3,
                "providers": {
                    "openai": {
                        "enabled": True,
                        "api_key": "test-openai-key",
                        "base_url": "https://openkey.cloud/v1",
                        "chat_model": "gpt-5-nano-2025-08-07"
                    },
                    "deepseek": {
                        "enabled": True,
                        "api_key": "test-deepseek-key",
                        "base_url": "https://api.deepseek.com",
                        "chat_model": "deepseek-chat"
                    }
                }
            }
        }

        import json
        with open(self.config_file, 'w') as f:
            json.dump(self.test_config, f, indent=2)

    def tearDown(self):
        """测试后清理"""
        if os.path.exists(self.config_file):
            os.remove(self.config_file)

    def test_load_config(self):
        """测试加载配置"""
        config_manager = ConfigManager()
        config = config_manager.load(self.config_file)

        self.assertIsNotNone(config)
        self.assertEqual(config.api.provider, "deepseek")
        self.assertEqual(config.api.chat_model, "deepseek-chat")
        self.assertEqual(config.api.embedding_model, "text-embedding-3-large")

    def test_provider_selection(self):
        """测试提供商选择"""
        config_manager = ConfigManager()
        config = config_manager.load(self.config_file)

        self.assertEqual(config.api.provider, "deepseek")
        self.assertTrue(config.api.providers["deepseek"]["enabled"])
        self.assertTrue(config.api.providers["openai"]["enabled"])

    def test_base_url_configuration(self):
        """测试 base_url 配置"""
        config_manager = ConfigManager()
        config = config_manager.load(self.config_file)

        self.assertEqual(config.api.base_url, "https://api.deepseek.com")
        self.assertEqual(config.api.providers["openai"]["base_url"], "https://openkey.cloud/v1")
        self.assertEqual(config.api.providers["deepseek"]["base_url"], "https://api.deepseek.com")

    def test_embedding_model_configuration(self):
        """测试 Embedding 模型配置"""
        config_manager = ConfigManager()
        config = config_manager.load(self.config_file)

        self.assertEqual(config.api.embedding_model, "text-embedding-3-large")


class TestIntegration(unittest.TestCase):
    """集成测试"""

    def setUp(self):
        """测试前准备"""
        self.config_file = "/tmp/test_integration_config.json"
        self.test_config = {
            "api": {
                "enabled": True,
                "provider": "deepseek",
                "api_key": "test-deepseek-key",
                "base_url": "https://api.deepseek.com",
                "chat_model": "deepseek-chat",
                "embedding_model": "text-embedding-3-large",
                "timeout": 30,
                "max_retries": 3,
                "providers": {
                    "openai": {
                        "enabled": True,
                        "api_key": "test-openai-key",
                        "base_url": "https://openkey.cloud/v1",
                        "chat_model": "gpt-4o-mini"
                    },
                    "deepseek": {
                        "enabled": True,
                        "api_key": "test-deepseek-key",
                        "base_url": "https://api.deepseek.com",
                        "chat_model": "deepseek-chat"
                    }
                }
            }
        }

        import json
        with open(self.config_file, 'w') as f:
            json.dump(self.test_config, f, indent=2)

    def tearDown(self):
        """测试后清理"""
        if os.path.exists(self.config_file):
            os.remove(self.config_file)
        LLMServiceFactory.clear_cache()

    def test_full_integration(self):
        """测试完整集成流程"""
        config_manager = ConfigManager()
        config = config_manager.load(self.config_file)

        self.assertEqual(config.api.provider, "deepseek")

        provider_config = config.api.providers.get(config.api.provider)
        self.assertIsNotNone(provider_config)

        service = LLMServiceFactory.create_service(config.api.provider, provider_config)
        self.assertIsNotNone(service)
        self.assertEqual(service.provider, LLMProvider.DEEPSEEK)
        self.assertEqual(service.model, "deepseek-chat")

    def test_embedding_config_verification(self):
        """测试 Embedding 配置验证"""
        config_manager = ConfigManager()
        config = config_manager.load(self.config_file)

        self.assertEqual(config.api.embedding_model, "text-embedding-3-large")
        self.assertEqual(config.api.providers["openai"]["base_url"], "https://openkey.cloud/v1")

    @patch('api.llm_service.OpenAI')
    def test_service_initialization(self, mock_openai):
        """测试服务初始化"""
        mock_client = MagicMock()
        mock_openai.return_value = mock_client

        config = {
            "enabled": True,
            "api_key": "test-key",
            "base_url": "https://api.deepseek.com",
            "chat_model": "deepseek-chat",
            "timeout": 30,
            "max_retries": 3
        }

        service = LLMServiceFactory.create_service("deepseek", config)

        self.assertIsNotNone(service)
        self.assertEqual(service.provider, LLMProvider.DEEPSEEK)


def run_tests():
    """运行所有测试"""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    suite.addTests(loader.loadTestsFromTestCase(TestLLMProvider))
    suite.addTests(loader.loadTestsFromTestCase(TestLLMServiceFactory))
    suite.addTests(loader.loadTestsFromTestCase(TestAPIConfig))
    suite.addTests(loader.loadTestsFromTestCase(TestConfigManager))
    suite.addTests(loader.loadTestsFromTestCase(TestIntegration))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
