#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
测试信息收集器 - 收集和展示所有测试文件的详细信息
"""

import os
import sys
from pathlib import Path
from typing import List, Dict, Any
import ast


class TestInfoCollector:
    """测试信息收集器"""

    def __init__(self):
        self.project_root = Path(__file__).parent
        self.test_files = self._discover_test_files()

    def _discover_test_files(self) -> List[Path]:
        """发现所有测试文件"""
        test_files = []

        # 查找 zyantine_genisis 目录下的测试文件
        for test_file in self.project_root.glob("test_*.py"):
            test_files.append(test_file)

        # 查找 tests 目录下的测试文件
        tests_dir = self.project_root / "tests"
        if tests_dir.exists():
            for test_file in tests_dir.glob("test_*.py"):
                test_files.append(test_file)

        return sorted(test_files)

    def _extract_test_functions(self, file_path: Path) -> List[str]:
        """从测试文件中提取测试函数"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            tree = ast.parse(content)
            test_functions = []

            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef) and node.name.startswith('test_'):
                    test_functions.append(node.name)

            return test_functions

        except Exception:
            return []

    def _extract_docstring(self, file_path: Path) -> str:
        """提取文件的文档字符串"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            tree = ast.parse(content)
            docstring = ast.get_docstring(tree)

            return docstring or "无描述"

        except Exception:
            return "无描述"

    def _count_lines(self, file_path: Path) -> int:
        """统计文件行数"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return sum(1 for _ in f)
        except Exception:
            return 0

    def collect_test_info(self) -> List[Dict[str, Any]]:
        """收集所有测试文件的信息"""
        test_info_list = []

        for test_file in self.test_files:
            test_functions = self._extract_test_functions(test_file)
            docstring = self._extract_docstring(test_file)
            line_count = self._count_lines(test_file)

            test_info = {
                "filename": test_file.name,
                "path": str(test_file),
                "description": docstring.split('\n')[0] if docstring else "无描述",
                "test_count": len(test_functions),
                "test_functions": test_functions,
                "line_count": line_count,
                "relative_path": str(test_file.relative_to(self.project_root))
            }

            test_info_list.append(test_info)

        return test_info_list

    def print_test_summary(self) -> None:
        """打印测试摘要"""
        test_info_list = self.collect_test_info()

        print("=" * 100)
        print("项目测试文件详细信息")
        print("=" * 100)

        total_tests = 0
        total_lines = 0

        for i, test_info in enumerate(test_info_list, 1):
            print(f"\n{i}. {test_info['filename']}")
            print(f"   路径: {test_info['relative_path']}")
            print(f"   描述: {test_info['description']}")
            print(f"   测试数量: {test_info['test_count']}")
            print(f"   代码行数: {test_info['line_count']}")

            if test_info['test_functions']:
                print(f"   测试函数:")
                for func in test_info['test_functions']:
                    print(f"     - {func}")
            else:
                print(f"   测试函数: 无")

            total_tests += test_info['test_count']
            total_lines += test_info['line_count']

        print("\n" + "=" * 100)
        print("统计信息")
        print("=" * 100)
        print(f"测试文件总数: {len(test_info_list)}")
        print(f"测试函数总数: {total_tests}")
        print(f"代码总行数: {total_lines}")
        print(f"平均每个文件的测试数: {total_tests/len(test_info_list):.1f}")
        print(f"平均每个文件的代码行数: {total_lines/len(test_info_list):.0f}")
        print("=" * 100)

    def export_to_markdown(self) -> None:
        """导出到 Markdown 文件"""
        test_info_list = self.collect_test_info()

        with open('TEST_DETAILS.md', 'w', encoding='utf-8') as f:
            f.write("# 测试文件详细信息\n\n")
            f.write(f"生成时间: {self._get_current_time()}\n\n")

            total_tests = 0
            total_lines = 0

            for i, test_info in enumerate(test_info_list, 1):
                f.write(f"## {i}. {test_info['filename']}\n\n")
                f.write(f"- **路径**: `{test_info['relative_path']}`\n")
                f.write(f"- **描述**: {test_info['description']}\n")
                f.write(f"- **测试数量**: {test_info['test_count']}\n")
                f.write(f"- **代码行数**: {test_info['line_count']}\n\n")

                if test_info['test_functions']:
                    f.write(f"**测试函数**:\n\n")
                    for func in test_info['test_functions']:
                        f.write(f"- `{func}`\n")
                    f.write("\n")

                total_tests += test_info['test_count']
                total_lines += test_info['line_count']

            f.write("---\n\n")
            f.write("## 统计信息\n\n")
            f.write(f"- **测试文件总数**: {len(test_info_list)}\n")
            f.write(f"- **测试函数总数**: {total_tests}\n")
            f.write(f"- **代码总行数**: {total_lines}\n")
            f.write(f"- **平均每个文件的测试数**: {total_tests/len(test_info_list):.1f}\n")
            f.write(f"- **平均每个文件的代码行数**: {total_lines/len(test_info_list):.0f}\n")

        print(f"\n测试详细信息已导出到: TEST_DETAILS.md")

    def _get_current_time(self) -> str:
        """获取当前时间"""
        from datetime import datetime
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description="测试信息收集器")
    parser.add_argument(
        "--export",
        action="store_true",
        help="导出到 Markdown 文件"
    )

    args = parser.parse_args()

    collector = TestInfoCollector()

    if args.export:
        collector.export_to_markdown()
    else:
        collector.print_test_summary()


if __name__ == "__main__":
    main()
