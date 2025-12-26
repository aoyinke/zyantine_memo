#!/usr/bin/env python3
"""
运行脚本 - 替代直接运行 main.py
"""
import sys
import os

# 添加当前目录到 Python 路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 导入并运行主函数
from main import main

if __name__ == "__main__":
    main()