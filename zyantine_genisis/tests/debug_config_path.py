#!/usr/bin/env python3
"""
调试配置路径
"""
import sys
import os

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from config.config_manager import ConfigManager


def main():
    print("=" * 60)
    print("调试配置路径")
    print("=" * 60)

    print(f"\n当前工作目录: {os.getcwd()}")
    print(f"项目根目录: {project_root}")
    print(f"脚本所在目录: {os.path.dirname(os.path.abspath(__file__))}")

    print(f"\n默认配置路径:")
    default_paths = [
        "./config/llm_config.json",
    ]

    for path in default_paths:
        abs_path = os.path.abspath(path)
        exists = os.path.exists(path)
        print(f"  - {path}")
        print(f"    绝对路径: {abs_path}")
        print(f"    存在: {exists}")

    print(f"\n实际配置文件路径:")
    config_file = os.path.join(project_root, "config/llm_config.json")
    print(f"  - {config_file}")
    print(f"    存在: {os.path.exists(config_file)}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
