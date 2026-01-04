#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
测试运行器 - 统一管理和运行所有测试
"""

import os
import sys
import subprocess
from pathlib import Path
from typing import List, Dict, Tuple


class TestRunner:
    """测试运行器"""

    def __init__(self):
        self.project_root = Path(__file__).parent.parent.parent
        self.test_files = self._discover_test_files()

    def _discover_test_files(self) -> Dict[str, List[Path]]:
        """发现所有测试文件"""
        test_files = {
            "memory": [],
            "api": [],
            "system": [],
            "integration": [],
            "protocols": []
        }

        # 查找 tests 目录下各个子目录的测试文件
        tests_dir = self.project_root / "tests"
        if tests_dir.exists():
            # memory 子目录
            memory_dir = tests_dir / "memory"
            if memory_dir.exists():
                for test_file in memory_dir.glob("test_*.py"):
                    test_files["memory"].append(test_file)

            # api 子目录
            api_dir = tests_dir / "api"
            if api_dir.exists():
                for test_file in api_dir.glob("test_*.py"):
                    test_files["api"].append(test_file)

            # system 子目录
            system_dir = tests_dir / "system"
            if system_dir.exists():
                for test_file in system_dir.glob("test_*.py"):
                    test_files["system"].append(test_file)

            # integration 子目录
            integration_dir = tests_dir / "integration"
            if integration_dir.exists():
                for test_file in integration_dir.glob("test_*.py"):
                    test_files["integration"].append(test_file)

            # protocols 子目录（tests 根目录）
            for test_file in tests_dir.glob("test_*.py"):
                if "protocol" in test_file.name.lower():
                    test_files["protocols"].append(test_file)

        return test_files

    def list_tests(self) -> None:
        """列出所有测试"""
        print("=" * 80)
        print("项目测试文件列表")
        print("=" * 80)

        for category, files in self.test_files.items():
            if files:
                print(f"\n{category.upper()} 测试 ({len(files)} 个):")
                for file in sorted(files):
                    print(f"  - {file}")

        print("\n" + "=" * 80)
        print(f"总计: {sum(len(files) for files in self.test_files.values())} 个测试文件")
        print("=" * 80)

    def run_test(self, test_file: Path) -> Tuple[bool, str]:
        """运行单个测试文件"""
        print(f"\n运行测试: {test_file.name}")
        print("-" * 80)

        try:
            result = subprocess.run(
                [sys.executable, str(test_file)],
                cwd=self.project_root,
                capture_output=True,
                text=True,
                timeout=300
            )

            success = result.returncode == 0
            output = result.stdout + result.stderr

            return success, output

        except subprocess.TimeoutExpired:
            return False, "测试超时"
        except Exception as e:
            return False, f"测试执行失败: {str(e)}"

    def run_category(self, category: str) -> Dict[str, Tuple[bool, str]]:
        """运行指定类别的所有测试"""
        if category not in self.test_files:
            print(f"错误: 未知类别 '{category}'")
            print(f"可用类别: {', '.join(self.test_files.keys())}")
            return {}

        print(f"\n运行 {category.upper()} 类别的所有测试")
        print("=" * 80)

        results = {}
        for test_file in self.test_files[category]:
            results[test_file.name] = self.run_test(test_file)

        return results

    def run_all(self) -> Dict[str, Tuple[bool, str]]:
        """运行所有测试"""
        print("\n运行所有测试")
        print("=" * 80)

        results = {}
        for category, files in self.test_files.items():
            if files:
                for test_file in files:
                    results[test_file.name] = self.run_test(test_file)

        return results

    def print_summary(self, results: Dict[str, Tuple[bool, str]]) -> None:
        """打印测试结果摘要"""
        print("\n" + "=" * 80)
        print("测试结果摘要")
        print("=" * 80)

        passed = sum(1 for success, _ in results.values() if success)
        failed = sum(1 for success, _ in results.values() if not success)
        total = len(results)

        print(f"\n总计: {total} 个测试")
        print(f"通过: {passed} 个")
        print(f"失败: {failed} 个")
        print(f"通过率: {passed/total*100:.1f}%")

        if failed > 0:
            print("\n失败的测试:")
            for test_name, (success, _) in results.items():
                if not success:
                    print(f"  - {test_name}")

        print("\n" + "=" * 80)

    def run_quick_tests(self) -> None:
        """运行快速测试（核心功能）"""
        print("\n运行快速测试（核心功能）")
        print("=" * 80)

        quick_tests = [
            "test_health_checker.py",
            "test_dashboard_integration.py",
            "test_security.py"
        ]

        results = {}
        for test_name in quick_tests:
            for category, files in self.test_files.items():
                for test_file in files:
                    if test_file.name == test_name:
                        results[test_name] = self.run_test(test_file)
                        break

        self.print_summary(results)


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description="测试运行器")
    parser.add_argument(
        "--list",
        action="store_true",
        help="列出所有测试文件"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="运行所有测试"
    )
    parser.add_argument(
        "--category",
        type=str,
        choices=["memory", "api", "system", "integration", "protocols"],
        help="运行指定类别的测试"
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="运行快速测试（核心功能）"
    )
    parser.add_argument(
        "--test",
        type=str,
        help="运行指定的测试文件"
    )

    args = parser.parse_args()

    runner = TestRunner()

    if args.list:
        runner.list_tests()
    elif args.all:
        results = runner.run_all()
        runner.print_summary(results)
    elif args.category:
        results = runner.run_category(args.category)
        runner.print_summary(results)
    elif args.quick:
        runner.run_quick_tests()
    elif args.test:
        test_file = runner.project_root / args.test
        if test_file.exists():
            success, output = runner.run_test(test_file)
            print(output)
            sys.exit(0 if success else 1)
        else:
            print(f"错误: 测试文件 '{args.test}' 不存在")
            sys.exit(1)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
