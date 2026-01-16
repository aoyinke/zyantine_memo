#!/usr/bin/env python3
"""
异常处理模块测试脚本
"""
import pytest
from utils.exception_handler import (
    handle_error, APIException, ConfigException, ProcessingException, ValidationException,
    retry_on_error, safe_execute, get_error_statistics, clear_error_statistics
)


def test_basic_exception_handling():
    """测试基本异常处理"""
    print("\n" + "="*60)
    print("测试1: 基本异常处理")
    print("="*60)
    
    try:
        # 测试处理通用异常
        try:
            raise ValueError("测试值错误")
        except Exception as e:
            handled_error = handle_error(e, context="测试上下文")
            print(f"✓ 处理通用异常成功: {handled_error}")
            assert isinstance(handled_error, APIException) or isinstance(handled_error, ConfigException)
        
        # 测试处理API异常
        api_error = APIException("测试API错误", error_code="API-001")
        handled_api_error = handle_error(api_error)
        print(f"✓ 处理API异常成功: {handled_api_error}")
        assert isinstance(handled_api_error, APIException)
        assert handled_api_error.error_code == "API-001"
        
        # 测试处理配置异常
        config_error = ConfigException("测试配置错误", error_code="CFG-001")
        handled_config_error = handle_error(config_error)
        print(f"✓ 处理配置异常成功: {handled_config_error}")
        assert isinstance(handled_config_error, ConfigException)
        assert handled_config_error.error_code == "CFG-001"
        
        print("\n✓ 基本异常处理测试通过")
        return True
        
    except Exception as e:
        print(f"\n✗ 基本异常处理测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_retry_decorator():
    """测试重试装饰器"""
    print("\n" + "="*60)
    print("测试2: 重试装饰器")
    print("="*60)
    
    try:
        # 测试成功的函数
        @retry_on_error(max_retries=3)
        def success_function():
            return "成功"
        
        result = success_function()
        print(f"✓ 成功函数执行成功: {result}")
        assert result == "成功"
        
        # 测试会失败但重试后成功的函数
        retry_count = 0
        
        @retry_on_error(max_retries=3, delay=0.1)
        def flaky_function():
            nonlocal retry_count
            retry_count += 1
            if retry_count < 3:
                raise ValueError(f"测试失败 {retry_count}")
            return "最终成功"
        
        result = flaky_function()
        print(f"✓ 不稳定函数重试后成功: {result}, 重试次数: {retry_count}")
        assert result == "最终成功"
        assert retry_count == 3
        
        print("\n✓ 重试装饰器测试通过")
        return True
        
    except Exception as e:
        print(f"\n✗ 重试装饰器测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_safe_execute():
    """测试安全执行函数"""
    print("\n" + "="*60)
    print("测试3: 安全执行函数")
    print("="*60)
    
    try:
        # 测试成功执行
        def success_func():
            return "成功执行"
        
        result = safe_execute(success_func)
        print(f"✓ 成功函数执行结果: {result}")
        assert result == "成功执行"
        
        # 测试失败执行
        def fail_func():
            raise ValueError("测试失败")
        
        result = safe_execute(fail_func, default_return="默认值")
        print(f"✓ 失败函数执行结果: {result}")
        assert result == "默认值"
        
        # 测试带参数的函数
        def add_func(a, b):
            return a + b
        
        result = safe_execute(add_func, 1, 2)
        print(f"✓ 带参数函数执行结果: {result}")
        assert result == 3
        
        print("\n✓ 安全执行函数测试通过")
        return True
        
    except Exception as e:
        print(f"\n✗ 安全执行函数测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_error_statistics():
    """测试错误统计"""
    print("\n" + "="*60)
    print("测试4: 错误统计")
    print("="*60)
    
    try:
        # 清除之前的统计
        clear_error_statistics()
        
        # 触发一些错误
        for i in range(3):
            try:
                raise ValueError(f"测试错误 {i}")
            except Exception as e:
                handle_error(e, context=f"测试上下文 {i}")
        
        # 获取统计信息
        stats = get_error_statistics()
        print(f"✓ 获取错误统计成功: {stats}")
        assert stats["total_errors"] >= 3
        
        # 清除统计
        clear_error_statistics()
        stats_after_clear = get_error_statistics()
        print(f"✓ 清除错误统计成功")
        assert stats_after_clear["total_errors"] == 0
        
        print("\n✓ 错误统计测试通过")
        return True
        
    except Exception as e:
        print(f"\n✗ 错误统计测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_exception_types():
    """测试不同类型的异常"""
    print("\n" + "="*60)
    print("测试5: 异常类型")
    print("="*60)
    
    try:
        # 测试API异常
        api_error = APIException("API错误", error_code="API-404", level="error")
        print(f"✓ 创建API异常成功: {api_error}")
        assert api_error.error_code == "API-404"
        assert api_error.level == "error"
        
        # 测试配置异常
        config_error = ConfigException("配置错误", error_code="CFG-001", level="warning")
        print(f"✓ 创建配置异常成功: {config_error}")
        assert config_error.error_code == "CFG-001"
        assert config_error.level == "warning"
        
        # 测试处理异常
        processing_error = ProcessingException("处理错误", error_code="PRC-001")
        print(f"✓ 创建处理异常成功: {processing_error}")
        assert processing_error.error_code == "PRC-001"
        
        # 测试验证异常
        validation_error = ValidationException("验证错误", error_code="VAL-001", level="warning")
        print(f"✓ 创建验证异常成功: {validation_error}")
        assert validation_error.error_code == "VAL-001"
        assert validation_error.level == "warning"
        
        print("\n✓ 异常类型测试通过")
        return True
        
    except Exception as e:
        print(f"\n✗ 异常类型测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """主测试函数"""
    print("\n" + "="*60)
    print("异常处理模块测试")
    print("="*60)
    
    results = []
    
    # 运行所有测试
    results.append(("基本异常处理", test_basic_exception_handling()))
    results.append(("重试装饰器", test_retry_decorator()))
    results.append(("安全执行函数", test_safe_execute()))
    results.append(("错误统计", test_error_statistics()))
    results.append(("异常类型", test_exception_types()))
    
    # 打印测试结果
    print("\n" + "="*60)
    print("测试结果汇总")
    print("="*60)
    
    for test_name, result in results:
        status = "✓ 通过" if result else "✗ 失败"
        print(f"{test_name}: {status}")
    
    # 统计通过率
    passed = sum(1 for _, result in results if result)
    total = len(results)
    pass_rate = (passed / total) * 100
    
    print(f"\n通过率: {passed}/{total} ({pass_rate:.1f}%)")
    
    if pass_rate == 100:
        print("\n✓ 所有测试通过！异常处理模块测试成功！")
    else:
        print(f"\n✗ 有 {total - passed} 个测试失败")
    
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
