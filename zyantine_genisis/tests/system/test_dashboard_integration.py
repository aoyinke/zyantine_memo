#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
系统性能监控仪表板集成测试
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datetime import datetime
import json

from system_performance_dashboard import SystemPerformanceDashboard
from memory.memory_manager import MemoryManager
from memory.memory_store import ZyantineMemorySystem
from config.config_manager import ConfigManager
from cognition.core_identity import CoreIdentity
from cognition.meta_cognition import MetaCognitionModule
from cognition.internal_state_dashboard import InternalStateDashboard
from cognition.context_parser import ContextParser
from cognition.cognitive_flow_manager import CognitiveFlowManager
from protocols.fact_checker import FactChecker
from protocols.length_regulator import LengthRegulator
from protocols.expression_validator import ExpressionValidator
from protocols.protocol_engine import ProtocolEngine
from api.service_provider import APIServiceProvider


def test_dashboard_integration():
    """测试仪表板与各模块的集成"""
    print("\n" + "="*80)
    print("系统性能监控仪表板集成测试")
    print("="*80 + "\n")

    # 初始化配置
    config = ConfigManager()

    # 初始化记忆管理器
    print("1. 初始化记忆管理器...")
    memory_manager = MemoryManager(config)
    print("   ✓ 记忆管理器初始化完成\n")

    # 初始化认知流程管理器
    print("2. 初始化认知流程管理器...")
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
    print("   ✓ 认知流程管理器初始化完成\n")

    # 初始化协议引擎
    print("3. 初始化协议引擎...")
    length_regulator = LengthRegulator()
    expression_validator = ExpressionValidator()
    protocol_engine = ProtocolEngine(
        fact_checker,
        length_regulator,
        expression_validator
    )
    print("   ✓ 协议引擎初始化完成\n")

    # 初始化API服务提供者
    print("4. 初始化API服务提供者...")
    try:
        api_service_provider = APIServiceProvider(config, core_identity)
        print("   ✓ API服务提供者初始化完成\n")
    except Exception as e:
        print(f"   ! API服务提供者初始化失败（跳过）: {e}\n")
        api_service_provider = None

    # 初始化性能仪表板
    print("5. 初始化性能仪表板...")
    dashboard = SystemPerformanceDashboard()
    print("   ✓ 性能仪表板初始化完成\n")

    # 执行一些操作以生成性能数据
    print("6. 执行操作以生成性能数据...")

    # 记忆操作
    print("   - 执行记忆操作...")
    for i in range(10):
        memory_manager.add_memory(
            content=f"测试记忆内容 {i}",
            memory_type="experience",
            metadata={"test": True}
        )
        if i % 2 == 0:
            try:
                memory_manager.search_memories(f"测试记忆 {i}", limit=1)
            except Exception as e:
                pass

    # 认知流程操作
    print("   - 执行认知流程操作...")
    for i in range(5):
        cognitive_flow_manager.process_thought(
            user_input=f"测试用户输入 {i}",
            history=[],
            current_vectors={}
        )

    # 协议引擎操作
    print("   - 执行协议引擎操作...")
    for i in range(5):
        protocol_engine.apply_all_protocols(
            draft=f"测试草稿内容 {i}",
            context={"test": True}
        )

    # API服务操作
    print("   - 执行API服务操作...")
    if api_service_provider:
        try:
            for i in range(3):
                api_service_provider.generate_reply(
                    user_input=f"测试API请求 {i}",
                    context={"test": True}
                )
        except Exception as e:
            print(f"   - API操作跳过（可能需要API密钥）: {e}")
    else:
        print("   - API服务提供者未初始化，跳过操作")

    print("   ✓ 操作执行完成\n")

    # 收集性能指标
    print("7. 收集性能指标...")
    metrics = dashboard.collect_metrics(
        memory_manager=memory_manager,
        cognitive_flow_manager=cognitive_flow_manager,
        api_service_provider=None,
        protocol_engine=protocol_engine
    )
    print("   ✓ 性能指标收集完成\n")

    # 显示仪表板摘要
    print("8. 仪表板摘要:")
    summary = dashboard.get_dashboard_summary()
    print(f"   - 时间戳: {summary.get('timestamp')}")
    print(f"   - 整体健康分数: {summary.get('overall_health_score', 0):.2f}")
    print(f"   - 健康分数变化: {summary.get('health_score_change', 0):.2f}")
    print(f"   - 活跃告警数: {summary.get('active_alerts', 0)}")
    print(f"   - 总告警数: {summary.get('total_alerts', 0)}")
    print(f"   - 数据点数: {summary.get('data_points', 0)}")
    print(f"   - 模块状态:")
    for module, status in summary.get('module_status', {}).items():
        print(f"     * {module}: {status}")
    print()

    # 显示详细报告
    print("9. 详细报告:")
    detailed = dashboard.get_detailed_report()

    # 记忆模块
    memory = detailed.get('modules', {}).get('memory', {})
    if memory.get('status') == 'available':
        print("   - 记忆模块:")
        print(f"     * 总记忆数: {memory.get('total_memories', 0)}")
        print(f"     * 缓存命中率: {memory.get('cache_hit_rate', 0):.2%}")
        print(f"     * 平均延迟: {memory.get('avg_latency_ms', 0):.2f} ms")
        print(f"     * 成功率: {memory.get('success_rate', 0):.2%}")
        storage = memory.get('storage', {})
        print(f"     * 存储分布: HOT={storage.get('hot_items', 0)}, WARM={storage.get('warm_items', 0)}, COLD={storage.get('cold_items', 0)}")

    # 认知流程模块
    cognitive = detailed.get('modules', {}).get('cognitive', {})
    if cognitive.get('status') == 'available':
        print("   - 认知流程模块:")
        print(f"     * 处理思维数: {cognitive.get('total_thoughts_processed', 0)}")
        print(f"     * 平均处理时间: {cognitive.get('avg_processing_time', 0):.2f} 秒")
        print(f"     * 缓存大小: {cognitive.get('cache_size', 0)}")
        print(f"     * 缓存命中率: {cognitive.get('cache_hit_rate', 0):.2%}")

    # API服务模块
    api = detailed.get('modules', {}).get('api', {})
    if api.get('status') == 'available':
        print("   - API服务模块:")
        print(f"     * 总请求数: {api.get('total_requests', 0)}")
        print(f"     * 成功率: {api.get('success_rate', 0):.2%}")
        print(f"     * 平均延迟: {api.get('avg_latency', 0):.2f} 秒")
        print(f"     * 活跃服务: {api.get('active_service', 'none')}")
        cache = api.get('cache', {})
        print(f"     * 缓存大小: {cache.get('size', 0)}")
        print(f"     * 缓存命中率: {cache.get('hit_rate', 0):.2%}")

    # 协议引擎模块
    protocol = detailed.get('modules', {}).get('protocol', {})
    if protocol.get('status') == 'available':
        print("   - 协议引擎模块:")
        print(f"     * 总执行次数: {protocol.get('total_executions', 0)}")
        print(f"     * 事实检查通过率: {protocol.get('fact_check_pass_rate', 0):.2%}")
        print(f"     * 表达违规率: {protocol.get('expression_violation_rate', 0):.2%}")
        print(f"     * 平均缩减: {protocol.get('average_reduction', 0):.2f}")
        print(f"     * 冲突事件数: {protocol.get('conflict_incidents', 0)}")
        print(f"     * 冲突率: {protocol.get('conflict_rate', 0):.2%}")

    print()

    # 显示告警
    print("10. 告警信息:")
    alerts = detailed.get('alerts', {})
    active_alerts = alerts.get('active', [])
    if active_alerts:
        print(f"   - 活跃告警 ({len(active_alerts)}):")
        for alert in active_alerts:
            print(f"     * [{alert.get('module')}] {alert.get('message')}")
    else:
        print("   - 无活跃告警")

    recent_alerts = alerts.get('recent', [])
    if recent_alerts:
        print(f"   - 最近告警 ({len(recent_alerts)}):")
        for alert in recent_alerts:
            status = "已解决" if alert.get('resolved') else "活跃"
            print(f"     * [{status}] [{alert.get('module')}] {alert.get('message')}")

    print()

    # 显示趋势
    print("11. 性能趋势:")
    trends = detailed.get('trends', {})
    if trends.get('status') != 'insufficient_data':
        for module, module_trends in trends.items():
            print(f"   - {module}:")
            for metric, trend in module_trends.items():
                print(f"     * {metric}: {trend}")
    else:
        print("   - 数据不足，无法计算趋势")

    print()

    # 导出报告
    print("12. 导出报告...")
    report_path = "/tmp/system_performance_report.json"
    dashboard.export_report(report_path)
    print(f"   ✓ 报告已导出到: {report_path}\n")

    # 测试阈值设置
    print("13. 测试阈值设置...")
    dashboard.set_threshold("memory_cache_hit_rate", 0.6)
    dashboard.set_threshold("api_avg_latency", 2.0)
    print("   ✓ 阈值设置完成\n")

    # 再次收集指标以测试告警
    print("14. 再次收集指标以测试告警...")
    metrics = dashboard.collect_metrics(
        memory_manager=memory_manager,
        cognitive_flow_manager=cognitive_flow_manager,
        api_service_provider=api_service_provider,
        protocol_engine=protocol_engine
    )
    print("   ✓ 指标收集完成\n")

    # 显示更新后的摘要
    print("15. 更新后的仪表板摘要:")
    summary = dashboard.get_dashboard_summary()
    print(f"   - 整体健康分数: {summary.get('overall_health_score', 0):.2f}")
    print(f"   - 活跃告警数: {summary.get('active_alerts', 0)}")
    print(f"   - 总告警数: {summary.get('total_alerts', 0)}")
    print()

    # 测试清空功能
    print("16. 测试清空功能...")
    dashboard.clear_alerts()
    print("   ✓ 告警已清空")

    summary = dashboard.get_dashboard_summary()
    print(f"   - 活跃告警数: {summary.get('active_alerts', 0)}")
    print(f"   - 总告警数: {summary.get('total_alerts', 0)}")
    print()

    print("="*80)
    print("测试完成！")
    print("="*80 + "\n")

    return True


if __name__ == "__main__":
    try:
        success = test_dashboard_integration()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n测试失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
