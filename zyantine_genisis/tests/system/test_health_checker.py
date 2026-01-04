#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
系统健康检查器集成测试
"""

import sys
import time
from datetime import datetime

from memory.memory_manager import MemoryManager
from cognition.cognitive_flow_manager import CognitiveFlowManager
from cognition.core_identity import CoreIdentity
from cognition.meta_cognition import MetaCognitionModule
from cognition.internal_state_dashboard import InternalStateDashboard
from cognition.context_parser import ContextParser
from protocols.protocol_engine import ProtocolEngine, ProtocolPriority
from protocols.fact_checker import FactChecker
from protocols.length_regulator import LengthRegulator
from protocols.expression_validator import ExpressionValidator
from api.service_provider import APIServiceProvider
from config.config_manager import ConfigManager
from system_health_checker import SystemHealthChecker, HealthStatus, TestResult


def test_health_checker_basic():
    """测试健康检查器基本功能"""
    print("\n" + "=" * 80)
    print("测试1: 健康检查器基本功能")
    print("=" * 80)

    # 创建健康检查器
    checker = SystemHealthChecker()
    print("  ✓ 健康检查器创建成功")

    # 注册一些测试检查项
    def test_check_pass():
        return True

    def test_check_fail():
        return False

    def test_check_error():
        raise Exception("测试错误")

    checker.register_check(
        name="test_pass",
        description="测试通过的检查",
        module="test",
        check_function=test_check_pass,
        critical=False
    )

    checker.register_check(
        name="test_fail",
        description="测试失败的检查",
        module="test",
        check_function=test_check_fail,
        critical=False
    )

    checker.register_check(
        name="test_error",
        description="测试错误的检查",
        module="test",
        check_function=test_check_error,
        critical=False
    )

    print("  ✓ 检查项注册成功")

    # 运行所有检查
    report = checker.run_all_checks()

    # 验证报告
    assert report.overall_status in [HealthStatus.HEALTHY, HealthStatus.DEGRADED, HealthStatus.UNHEALTHY]
    assert 0 <= report.health_score <= 100
    assert len(report.checks) == 3
    assert len(report.test_summary) > 0
    assert len(report.recommendations) > 0

    print("  ✓ 健康检查运行成功")
    print(f"    整体状态: {report.overall_status.value}")
    print(f"    健康分数: {report.health_score:.2f}")
    print(f"    检查项数: {len(report.checks)}")
    print(f"    测试摘要: {report.test_summary}")
    print(f"    建议数: {len(report.recommendations)}")

    # 获取健康摘要
    summary = checker.get_health_summary()
    assert summary["overall_status"] is not None
    assert summary["check_count"] == 3

    print("  ✓ 健康摘要获取成功")
    print(f"    摘要: {summary}")

    return True


def test_health_checker_with_memory_manager():
    """测试健康检查器与记忆管理器集成"""
    print("\n" + "=" * 80)
    print("测试2: 健康检查器与记忆管理器集成")
    print("=" * 80)

    # 创建记忆管理器
    config = ConfigManager().get()
    memory_manager = MemoryManager(config)
    print("  ✓ 记忆管理器初始化完成")

    # 创建健康检查器
    checker = SystemHealthChecker()

    # 注册记忆管理器健康检查
    def check_memory_availability():
        """检查记忆管理器可用性"""
        try:
            stats = memory_manager.get_performance_summary()
            return stats is not None
        except Exception:
            return False

    def check_memory_cache():
        """检查记忆缓存状态"""
        try:
            return memory_manager.cache is not None
        except Exception:
            return False

    def check_memory_storage():
        """检查记忆存储状态"""
        try:
            return memory_manager.storage_optimizer is not None
        except Exception:
            return False

    checker.register_check(
        name="memory_availability",
        description="记忆管理器可用性检查",
        module="memory",
        check_function=check_memory_availability,
        critical=True
    )

    checker.register_check(
        name="memory_cache",
        description="记忆缓存状态检查",
        module="memory",
        check_function=check_memory_cache,
        critical=False
    )

    checker.register_check(
        name="memory_storage",
        description="记忆存储状态检查",
        module="memory",
        check_function=check_memory_storage,
        critical=False
    )

    print("  ✓ 记忆管理器检查项注册成功")

    # 添加一些测试记忆
    for i in range(5):
        try:
            memory_manager.add_memory(
                content=f"测试健康检查记忆 {i}",
                memory_type="experience",
                metadata={"test": True}
            )
        except Exception as e:
            print(f"    警告: 添加记忆失败 (网络错误): {str(e)[:50]}")
            continue

    # 运行健康检查
    report = checker.run_all_checks()

    # 验证结果
    memory_checks = [c for c in report.checks if c.module == "memory"]
    assert len(memory_checks) == 3

    print("  ✓ 健康检查运行成功")
    print(f"    记忆模块状态: {report.module_status.get('memory', HealthStatus.HEALTHY).value}")
    print(f"    健康分数: {report.health_score:.2f}")

    # 获取检查详情
    details = checker.get_check_details()
    assert len(details) == 3

    print("  ✓ 检查详情获取成功")
    for detail in details:
        print(f"    {detail['name']}: {detail['last_result']}")

    return True


def test_health_checker_with_cognitive_flow():
    """测试健康检查器与认知流程集成"""
    print("\n" + "=" * 80)
    print("测试3: 健康检查器与认知流程集成")
    print("=" * 80)

    # 创建认知流程管理器
    config = ConfigManager().get()
    memory_manager = MemoryManager(config)
    core_identity = CoreIdentity()
    internal_state_dashboard = InternalStateDashboard()
    context_parser = ContextParser()
    meta_cognition = MetaCognitionModule(internal_state_dashboard, context_parser)
    fact_checker = FactChecker(memory_manager)
    cognitive_flow_manager = CognitiveFlowManager(
        core_identity,
        memory_manager,
        meta_cognition,
        fact_checker
    )
    print("  ✓ 认知流程管理器初始化完成")

    # 创建健康检查器
    checker = SystemHealthChecker()

    # 注册认知流程健康检查
    def check_cognitive_availability():
        """检查认知流程可用性"""
        try:
            metrics = cognitive_flow_manager.get_performance_stats()
            return metrics is not None
        except Exception:
            return False

    def check_cognitive_cache():
        """检查认知缓存状态"""
        try:
            return cognitive_flow_manager.cognitive_state_cache is not None
        except Exception:
            return False

    checker.register_check(
        name="cognitive_availability",
        description="认知流程可用性检查",
        module="cognitive",
        check_function=check_cognitive_availability,
        critical=True
    )

    checker.register_check(
        name="cognitive_cache",
        description="认知缓存状态检查",
        module="cognitive",
        check_function=check_cognitive_cache,
        critical=False
    )

    print("  ✓ 认知流程检查项注册成功")

    # 执行一些认知流程操作
    for i in range(3):
        cognitive_flow_manager.process_thought(
            user_input=f"测试健康检查认知流程 {i}",
            history=[],
            current_vectors={}
        )

    # 运行健康检查
    report = checker.run_all_checks()

    # 验证结果
    cognitive_checks = [c for c in report.checks if c.module == "cognitive"]
    assert len(cognitive_checks) == 2

    print("  ✓ 健康检查运行成功")
    print(f"    认知模块状态: {report.module_status.get('cognitive', HealthStatus.HEALTHY).value}")
    print(f"    健康分数: {report.health_score:.2f}")

    return True


def test_health_checker_with_protocol_engine():
    """测试健康检查器与协议引擎集成"""
    print("\n" + "=" * 80)
    print("测试4: 健康检查器与协议引擎集成")
    print("=" * 80)

    # 创建协议引擎
    config = ConfigManager().get()
    memory_manager = MemoryManager(config)
    fact_checker = FactChecker(memory_manager)
    length_regulator = LengthRegulator()
    expression_validator = ExpressionValidator()
    protocol_engine = ProtocolEngine(fact_checker, length_regulator, expression_validator)
    print("  ✓ 协议引擎初始化完成")

    # 创建健康检查器
    checker = SystemHealthChecker()

    # 注册协议引擎健康检查
    def check_protocol_availability():
        """检查协议引擎可用性"""
        try:
            performance = protocol_engine.get_protocol_performance()
            return performance is not None
        except Exception:
            return False

    def check_protocol_conflicts():
        """检查协议冲突状态"""
        try:
            conflict_stats = protocol_engine.get_conflict_statistics()
            return conflict_stats is not None
        except Exception:
            return False

    checker.register_check(
        name="protocol_availability",
        description="协议引擎可用性检查",
        module="protocol",
        check_function=check_protocol_availability,
        critical=True
    )

    checker.register_check(
        name="protocol_conflicts",
        description="协议冲突状态检查",
        module="protocol",
        check_function=check_protocol_conflicts,
        critical=False
    )

    print("  ✓ 协议引擎检查项注册成功")

    # 执行一些协议操作
    for i in range(3):
        protocol_engine.apply_all_protocols(
            draft=f"测试健康检查协议引擎 {i}",
            context={"user_id": "test_user"}
        )

    # 运行健康检查
    report = checker.run_all_checks()

    # 验证结果
    protocol_checks = [c for c in report.checks if c.module == "protocol"]
    assert len(protocol_checks) == 2

    print("  ✓ 健康检查运行成功")
    print(f"    协议模块状态: {report.module_status.get('protocol', HealthStatus.HEALTHY).value}")
    print(f"    健康分数: {report.health_score:.2f}")

    return True


def test_health_checker_alerts():
    """测试健康检查器告警功能"""
    print("\n" + "=" * 80)
    print("测试5: 健康检查器告警功能")
    print("=" * 80)

    # 创建健康检查器
    checker = SystemHealthChecker()

    # 注册一个会失败的检查
    fail_count = [0]

    def check_will_fail():
        """会失败的检查"""
        fail_count[0] += 1
        return False

    checker.register_check(
        name="critical_fail",
        description="关键失败检查",
        module="test",
        check_function=check_will_fail,
        critical=True
    )

    print("  ✓ 检查项注册成功")

    # 运行多次检查以触发告警
    for i in range(4):
        checker.run_all_checks()
        print(f"    第 {i+1} 次检查完成")

    # 验证告警
    summary = checker.get_health_summary()
    assert summary["active_alerts"] > 0 or summary["total_alerts"] > 0

    print("  ✓ 告警生成成功")
    print(f"    活跃告警数: {summary['active_alerts']}")
    print(f"    总告警数: {summary['total_alerts']}")

    # 显示告警详情
    alerts = checker.alerts
    for alert in alerts[:3]:
        print(f"    告警: {alert['message']}")

    # 解决告警
    if len(alerts) > 0:
        checker.resolve_alert(0)
        print("  ✓ 告警解决成功")

    # 清空告警
    checker.clear_alerts()
    summary = checker.get_health_summary()
    assert summary["active_alerts"] == 0
    assert summary["total_alerts"] == 0

    print("  ✓ 告警清空成功")

    return True


def test_health_checker_auto_check():
    """测试健康检查器自动检查功能"""
    print("\n" + "=" * 80)
    print("测试6: 健康检查器自动检查功能")
    print("=" * 80)

    # 创建健康检查器
    checker = SystemHealthChecker()
    checker.check_interval = 2  # 2秒间隔

    # 注册检查项
    check_count = [0]

    def check_counter():
        """计数检查"""
        check_count[0] += 1
        return True

    checker.register_check(
        name="counter_check",
        description="计数检查",
        module="test",
        check_function=check_counter,
        critical=False
    )

    print("  ✓ 检查项注册成功")

    # 启动自动检查
    checker.start_auto_check()
    print("  ✓ 自动检查已启动")

    # 等待几次自动检查
    print("  等待自动检查...")
    time.sleep(6)

    # 停止自动检查
    checker.stop_auto_check()
    print("  ✓ 自动检查已停止")

    # 验证检查次数
    assert check_count[0] >= 2  # 至少执行2次

    print(f"  ✓ 自动检查执行成功，执行次数: {check_count[0]}")

    return True


def test_health_checker_comprehensive():
    """测试健康检查器综合集成"""
    print("\n" + "=" * 80)
    print("测试7: 健康检查器综合集成")
    print("=" * 80)

    # 初始化所有模块
    print("  1. 初始化模块...")
    config = ConfigManager().get()
    memory_manager = MemoryManager(config)
    core_identity = CoreIdentity()
    internal_state_dashboard = InternalStateDashboard()
    context_parser = ContextParser()
    meta_cognition = MetaCognitionModule(internal_state_dashboard, context_parser)
    fact_checker = FactChecker(memory_manager)
    cognitive_flow_manager = CognitiveFlowManager(
        core_identity,
        memory_manager,
        meta_cognition,
        fact_checker
    )
    length_regulator = LengthRegulator()
    expression_validator = ExpressionValidator()
    protocol_engine = ProtocolEngine(fact_checker, length_regulator, expression_validator)
    print("    ✓ 模块初始化完成")

    # 创建健康检查器
    print("  2. 创建健康检查器...")
    checker = SystemHealthChecker()

    # 注册所有模块的检查项
    print("  3. 注册检查项...")

    # 记忆模块检查
    checker.register_check(
        name="memory_availability",
        description="记忆管理器可用性",
        module="memory",
        check_function=lambda: memory_manager.get_performance_summary() is not None,
        critical=True
    )

    checker.register_check(
        name="memory_cache",
        description="记忆缓存状态",
        module="memory",
        check_function=lambda: memory_manager.cache is not None,
        critical=False
    )

    # 认知模块检查
    checker.register_check(
        name="cognitive_availability",
        description="认知流程可用性",
        module="cognitive",
        check_function=lambda: cognitive_flow_manager.get_performance_stats() is not None,
        critical=True
    )

    checker.register_check(
        name="cognitive_cache",
        description="认知缓存状态",
        module="cognitive",
        check_function=lambda: cognitive_flow_manager.cognitive_state_cache is not None,
        critical=False
    )

    # 协议模块检查
    checker.register_check(
        name="protocol_availability",
        description="协议引擎可用性",
        module="protocol",
        check_function=lambda: protocol_engine.get_protocol_performance() is not None,
        critical=True
    )

    checker.register_check(
        name="protocol_conflicts",
        description="协议冲突状态",
        module="protocol",
        check_function=lambda: protocol_engine.get_conflict_statistics() is not None,
        critical=False
    )

    print("    ✓ 检查项注册完成")

    # 执行一些操作
    print("  4. 执行操作...")
    for i in range(5):
        try:
            memory_manager.add_memory(
                content=f"综合测试记忆 {i}",
                memory_type="experience",
                metadata={"test": True}
            )
        except Exception as e:
            print(f"    警告: 添加记忆失败 (网络错误): {str(e)[:50]}")
            continue

    for i in range(3):
        cognitive_flow_manager.process_thought(
            user_input=f"综合测试认知流程 {i}",
            history=[],
            current_vectors={}
        )

    for i in range(3):
        protocol_engine.apply_all_protocols(
            draft=f"综合测试协议引擎 {i}",
            context={"user_id": "test_user"}
        )

    print("    ✓ 操作执行完成")

    # 运行健康检查
    print("  5. 运行健康检查...")
    report = checker.run_all_checks()

    print("    ✓ 健康检查完成")

    # 显示报告
    print("\n  6. 健康检查报告:")
    print(f"    时间戳: {report.timestamp}")
    print(f"    整体状态: {report.overall_status.value}")
    print(f"    健康分数: {report.health_score:.2f}")
    print(f"    检查项数: {len(report.checks)}")
    print(f"    测试摘要: {report.test_summary}")
    print(f"    活跃告警数: {len(report.active_alerts)}")
    print(f"    建议数: {len(report.recommendations)}")

    print("\n    模块状态:")
    for module, status in report.module_status.items():
        print(f"      {module}: {status.value}")

    print("\n    检查项详情:")
    for check in report.checks:
        print(f"      {check.name}: {check.last_result.value if check.last_result else 'N/A'}")

    if report.recommendations:
        print("\n    建议:")
        for rec in report.recommendations[:3]:
            print(f"      - {rec}")

    # 获取健康摘要
    print("\n  7. 健康摘要:")
    summary = checker.get_health_summary()
    print(f"    状态: {summary['overall_status']}")
    print(f"    分数: {summary['health_score']:.2f}")
    print(f"    活跃告警: {summary['active_alerts']}")
    print(f"    总告警: {summary['total_alerts']}")
    print(f"    数据点: {summary['data_points']}")

    # 获取检查详情
    print("\n  8. 检查详情:")
    details = checker.get_check_details()
    for detail in details:
        print(f"    {detail['name']}: {detail['last_result']}")

    # 测试历史记录
    print("\n  9. 历史记录:")
    print(f"    历史记录数: {len(checker.check_history)}")

    # 再次运行检查以生成历史
    checker.run_all_checks()
    checker.run_all_checks()

    print(f"    更新后历史记录数: {len(checker.check_history)}")

    # 清空历史
    checker.clear_history()
    print(f"    清空后历史记录数: {len(checker.check_history)}")

    print("\n  ✓ 综合集成测试完成")

    return True


def main():
    """主测试函数"""
    print("=" * 80)
    print("系统健康检查器集成测试")
    print("=" * 80)

    tests = [
        ("健康检查器基本功能", test_health_checker_basic),
        ("健康检查器与记忆管理器集成", test_health_checker_with_memory_manager),
        ("健康检查器与认知流程集成", test_health_checker_with_cognitive_flow),
        ("健康检查器与协议引擎集成", test_health_checker_with_protocol_engine),
        ("健康检查器告警功能", test_health_checker_alerts),
        ("健康检查器自动检查功能", test_health_checker_auto_check),
        ("健康检查器综合集成", test_health_checker_comprehensive),
    ]

    passed = 0
    failed = 0

    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"  ✗ 测试异常: {str(e)}")
            import traceback
            traceback.print_exc()
            failed += 1

    print("\n" + "=" * 80)
    print(f"测试完成: 通过 {passed}/{len(tests)}, 失败 {failed}/{len(tests)}")
    print("=" * 80)

    return failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
