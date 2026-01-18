"""
使用示例 - 如何配置和使用不同的LLM提供商
"""
from config.config_manager import ConfigManager
from api.llm_service_factory import LLMServiceFactory
from api.llm_provider import LLMProviderPresets
from api.service_provider import APIServiceProvider
from cognition.core_identity import CoreIdentity


def example_1_use_config_file():
    """示例1: 使用配置文件"""
    print("=== 示例1: 使用配置文件 ===")

    # 加载配置
    config_manager = ConfigManager()
    config = config_manager.load("./config/llm_config.json")

    # 创建服务提供者
    provider = APIServiceProvider(config)

    # 生成回复
    reply = provider.generate_reply(
        user_input="你好",
        action_plan={"chosen_mask": "长期搭档", "primary_strategy": "友好交流"},
        growth_result={},
        context_analysis={},
        conversation_history=[],
        current_vectors={"TR": 0.5, "CS": 0.5, "SA": 0.5},
        memory_context={"retrieved_memories": [], "resonant_memory": None}
    )

    print(f"回复: {reply}")

    # 关闭服务
    provider.shutdown()


def example_2_use_preset_config():
    """示例2: 使用预设配置"""
    print("\n=== 示例2: 使用预设配置 ===")

    # 使用DeepSeek预设配置
    deepseek_service = LLMServiceFactory.create_from_preset(
        provider="deepseek",
        api_key="your-api-key-here",
        model="deepseek-chat"
    )

    if deepseek_service:
        # 测试连接
        success, message = deepseek_service.test_connection()
        print(f"连接测试: {success}, 消息: {message}")

        # 生成回复
        reply, metadata = deepseek_service.generate_reply(
            system_prompt="你是一个友好的助手",
            user_input="你好，请介绍一下你自己",
            max_tokens=100
        )

        print(f"回复: {reply}")
        print(f"元数据: {metadata}")

        # 获取统计信息
        stats = deepseek_service.get_statistics()
        print(f"统计信息: {stats}")

        # 关闭服务
        deepseek_service.shutdown()


def example_3_switch_provider():
    """示例3: 切换提供商"""
    print("\n=== 示例3: 切换提供商 ===")

    # 创建DeepSeek服务
    deepseek_service = LLMServiceFactory.create_from_preset(
        provider="deepseek",
        api_key="your-api-key-here"
    )

    # 创建OpenAI服务
    openai_service = LLMServiceFactory.create_from_preset(
        provider="openai",
        api_key="your-openai-api-key",
        model="gpt-4o-mini"
    )

    # 使用DeepSeek
    if deepseek_service:
        reply, _ = deepseek_service.generate_reply(
            system_prompt="你是一个助手",
            user_input="使用DeepSeek回答：1+1等于几？",
            max_tokens=50
        )
        print(f"DeepSeek回复: {reply}")

    # 使用OpenAI
    if openai_service:
        reply, _ = openai_service.generate_reply(
            system_prompt="你是一个助手",
            user_input="使用OpenAI回答：1+1等于几？",
            max_tokens=50
        )
        print(f"OpenAI回复: {reply}")

    # 关闭服务
    if deepseek_service:
        deepseek_service.shutdown()
    if openai_service:
        openai_service.shutdown()


def example_4_custom_config():
    """示例4: 自定义配置"""
    print("\n=== 示例4: 自定义配置 ===")

    # 创建自定义配置
    custom_config = {
        "provider": "deepseek",
        "chat_model": "deepseek-chat",
        "api_key": "your-api-key-here",
        "base_url": "https://api.deepseek.com",
        "timeout": 30,
        "max_retries": 3,
        "temperature": 0.8,
        "max_tokens": 2000,
        "enabled": True
    }

    # 使用自定义配置创建服务
    service = LLMServiceFactory.create_service("deepseek", custom_config)

    if service:
        # 生成回复
        reply, metadata = service.generate_reply(
            system_prompt="你是一个助手",
            user_input="你好",
            max_tokens=100
        )

        print(f"回复: {reply}")
        print(f"提供商: {metadata.get('provider')}")
        print(f"模型: {metadata.get('model')}")

        # 关闭服务
        service.shutdown()


def example_5_update_config():
    """示例5: 动态更新配置"""
    print("\n=== 示例5: 动态更新配置 ===")

    # 加载配置
    config_manager = ConfigManager()
    config = config_manager.load("./config/llm_config.json")

    print(f"当前提供商: {config.api.provider}")

    # 更新配置 - 切换到OpenAI
    config_manager.update({
        "api": {
            "provider": "openai",
            "providers": {
                "openai": {
                    "enabled": True,
                    "api_key": "your-openai-api-key",
                    "base_url": "https://api.openai.com/v1",
                    "chat_model": "gpt-5-nano-2025-08-07"
                },
                "deepseek": {
                    "enabled": False
                }
            }
        }
    })

    print(f"更新后提供商: {config.api.provider}")

    # 保存配置
    config_manager.save("./config/llm_config.json")
    print("配置已保存")


def example_6_get_statistics():
    """示例6: 获取服务统计信息"""
    print("\n=== 示例6: 获取服务统计信息 ===")

    # 创建多个服务
    services = []

    # DeepSeek服务
    deepseek_service = LLMServiceFactory.create_from_preset(
        provider="deepseek",
        api_key="your-api-key-here"
    )
    if deepseek_service:
        services.append(deepseek_service)

    # 生成一些请求
    for service in services:
        for i in range(3):
            service.generate_reply(
                system_prompt="你是一个助手",
                user_input=f"测试{i}",
                max_tokens=10
            )

    # 获取所有服务的统计信息
    all_stats = LLMServiceFactory.get_service_statistics()
    print("所有服务统计:")
    for key, stats in all_stats.items():
        print(f"  {key}:")
        print(f"    提供商: {stats.get('provider')}")
        print(f"    模型: {stats.get('model')}")
        print(f"    总请求数: {stats.get('total_requests')}")
        print(f"    成功数: {stats.get('success_count')}")
        print(f"    成功率: {stats.get('success_rate'):.2f}%")
        print(f"    平均延迟: {stats.get('avg_latency'):.2f}秒")

    # 测试所有服务连接
    test_results = LLMServiceFactory.test_all_services()
    print("\n连接测试结果:")
    for key, (success, message) in test_results.items():
        print(f"  {key}: {'成功' if success else '失败'} - {message}")

    # 关闭所有服务
    for service in services:
        service.shutdown()

    # 清空工厂缓存
    LLMServiceFactory.clear_cache()


if __name__ == "__main__":
    # 运行示例
    example_1_use_config_file()
    example_2_use_preset_config()
    example_3_switch_provider()
    example_4_custom_config()
    example_5_update_config()
    example_6_get_statistics()

    print("\n=== 所有示例运行完成 ===")
