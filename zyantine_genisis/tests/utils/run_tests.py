#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
测试运行器 - 统一管理和运行所有测试
"""

import os
import sys
import subprocess
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any
import time


class TestRunner:
    """测试运行器"""

    def __init__(self):
        self.project_root = Path(__file__).parent.parent.parent
        self.test_files = self._discover_test_files()
        self.test_results: Dict[str, Dict[str, Any]] = {}

    def _discover_test_files(self) -> Dict[str, List[Path]]:
        """发现所有测试文件，按目录结构自动分类"""
        test_files: Dict[str, List[Path]] = {}
        tests_dir = self.project_root / "tests"
        
        if not tests_dir.exists():
            return test_files
        
        # 自动发现 tests 目录下的所有子目录作为测试类别
        for subdir in tests_dir.iterdir():
            if subdir.is_dir() and subdir.name != "__pycache__":
                category = subdir.name
                test_files[category] = []
                
                # 添加子目录中的测试文件
                for test_file in subdir.glob("test_*.py"):
                    test_files[category].append(test_file)
        
        # 添加 tests 根目录下的测试文件（非协议相关）
        root_tests = []
        for test_file in tests_dir.glob("test_*.py"):
            if "protocol" not in test_file.name.lower():
                root_tests.append(test_file)
        
        if root_tests:
            test_files["root"] = root_tests
        
        # 专门处理协议测试
        protocol_tests = []
        for test_file in tests_dir.glob("test_*.py"):
            if "protocol" in test_file.name.lower():
                protocol_tests.append(test_file)
        
        if protocol_tests:
            test_files["protocols"] = protocol_tests
        
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

    def run_test(self, test_file: Path) -> Tuple[bool, str, float]:
        """运行单个测试文件"""
        print(f"\n运行测试: {test_file.name}")
        print("-" * 80)

        start_time = time.time()
        try:
            result = subprocess.run(
                [sys.executable, str(test_file)],
                cwd=self.project_root,
                capture_output=True,
                text=True,
                timeout=300
            )

            duration = time.time() - start_time
            success = result.returncode == 0
            output = result.stdout + result.stderr

            if success:
                print(f"✅ 测试通过 | 耗时: {duration:.2f}秒")
            else:
                print(f"❌ 测试失败 | 耗时: {duration:.2f}秒")

            return success, output, duration

        except subprocess.TimeoutExpired:
            duration = 300.0
            print(f"❌ 测试超时 | 耗时: {duration:.2f}秒")
            return False, "测试超时", duration
        except Exception as e:
            duration = time.time() - start_time
            print(f"❌ 测试执行失败 | 耗时: {duration:.2f}秒")
            return False, f"测试执行失败: {str(e)}", duration

    def run_category(self, category: str) -> Dict[str, Tuple[bool, str, float]]:
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

    def run_all(self) -> Dict[str, Tuple[bool, str, float]]:
        """运行所有测试"""
        print(f"\n运行所有测试")
        print("=" * 80)

        results = {}
        for category, files in self.test_files.items():
            if files:
                for test_file in files:
                    results[test_file.name] = self.run_test(test_file)

        return results

    def print_summary(self, results: Dict[str, Tuple[bool, str, float]]) -> None:
        """打印测试结果摘要"""
        print("\n" + "=" * 80)
        print("测试结果摘要")
        print("=" * 80)

        passed = sum(1 for success, _, _ in results.values() if success)
        failed = sum(1 for success, _, _ in results.values() if not success)
        total = len(results)
        total_duration = sum(duration for _, _, duration in results.values())

        print(f"\n总计: {total} 个测试")
        print(f"通过: {passed} 个")
        print(f"失败: {failed} 个")
        print(f"通过率: {passed/total*100:.1f}%")
        print(f"总耗时: {total_duration:.2f} 秒")
        print(f"平均耗时: {total_duration/total:.2f} 秒/个测试")

        if failed > 0:
            print("\n失败的测试:")
            for test_name, (success, _, duration) in results.items():
                if not success:
                    print(f"  - {test_name} | 耗时: {duration:.2f}秒")

        print("\n" + "=" * 80)

    def run_quick_tests(self) -> None:
        """运行快速测试（核心功能）"""
        print(f"\n运行快速测试（核心功能）")
        print("=" * 80)

        quick_tests = [
            "test_health_checker.py",
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
    
    # 更新category选项的choices
    if args.category:
        if args.category not in runner.test_files:
            print(f"错误: 未知类别 '{args.category}'")
            print(f"可用类别: {', '.join(runner.test_files.keys())}")
            sys.exit(1)

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
        # 支持相对路径和绝对路径
        test_file = Path(args.test)
        if not test_file.is_absolute():
            # 尝试在tests目录下查找
            test_file = runner.project_root / "tests" / args.test
            if not test_file.exists():
                # 尝试在当前目录下查找
                test_file = Path(args.test)
        
        if test_file.exists():
            success, output, _ = runner.run_test(test_file)
            if not success:
                print(output)
            sys.exit(0 if success else 1)
        else:
            print(f"错误: 测试文件 '{args.test}' 不存在")
            sys.exit(1)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
