#!/usr/bin/env python3
"""
调试配置加载 - 带详细过程输出
"""
import sys
import os
import json

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from config.config_manager import ConfigManager


def main():
    print("=" * 60)
    print("调试配置加载 - 带详细过程输出")
    print("=" * 60)

    # 读取原始配置文件
    config_file = os.path.join(project_root, "config/llm_config.json")
    print(f"\n读取配置文件: {config_file}")
    with open(config_file, 'r', encoding='utf-8') as f:
        config_dict = json.load(f)

    print(f"\n原始配置文件中的api:")
    print(json.dumps(config_dict.get('api', {}), indent=2, ensure_ascii=False))

    print(f"\n原始配置文件中的api.providers.deepseek:")
    print(json.dumps(config_dict.get('api', {}).get('providers', {}).get('deepseek', {}), indent=2, ensure_ascii=False))

    # 加载配置
    print("\n" + "=" * 60)
    print("开始加载配置...")
    print("=" * 60)

    config = ConfigManager()
    config.load()
    system_config = config.get()

    print(f"\n加载后的API配置:")
    print(f"  - provider: {system_config.api.provider}")
    print(f"  - api_key: {system_config.api.api_key[:10]}...")
    print(f"  - base_url: {system_config.api.base_url}")
    print(f"  - chat_model: {system_config.api.chat_model}")

    print(f"\n加载后的Providers配置:")
    for provider_name, provider_config in system_config.api.providers.items():
        print(f"\n  {provider_name}:")
        print(f"    - enabled: {provider_config.get('enabled')}")
        print(f"    - api_key: {provider_config.get('api_key', '')[:10]}...")
        print(f"    - base_url: {provider_config.get('base_url')}")
        print(f"    - chat_model: {provider_config.get('chat_model')}")
        print(f"    - use_max_completion_tokens: {provider_config.get('use_max_completion_tokens')}")

    # 检查providers字典的id
    print(f"\n检查providers字典的id:")
    print(f"  - providers字典id: {id(system_config.api.providers)}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
