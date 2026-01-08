#!/usr/bin/env python3
"""
调试配置加载 - 详细版本
"""
import sys
import os

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from config.config_manager import ConfigManager


def main():
    print("=" * 60)
    print("调试配置加载 - 详细版本")
    print("=" * 60)

    # 加载配置
    config = ConfigManager()
    config.load()
    system_config = config.get()

    print(f"\nAPI配置:")
    print(f"  - provider: {system_config.api.provider}")
    print(f"  - api_key: {system_config.api.api_key[:10]}...")
    print(f"  - base_url: {system_config.api.base_url}")
    print(f"  - chat_model: {system_config.api.chat_model}")

    print(f"\nProviders配置:")
    for provider_name, provider_config in system_config.api.providers.items():
        print(f"\n  {provider_name}:")
        print(f"    - enabled: {provider_config.get('enabled')}")
        print(f"    - api_key: {provider_config.get('api_key', '')[:10]}...")
        print(f"    - base_url: {provider_config.get('base_url')}")
        print(f"    - chat_model: {provider_config.get('chat_model')}")
        print(f"    - use_max_completion_tokens: {provider_config.get('use_max_completion_tokens')}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
