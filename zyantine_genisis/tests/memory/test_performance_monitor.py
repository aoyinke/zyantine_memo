#!/usr/bin/env python3
"""
测试性能监控器功能
"""
import sys
import os
import time
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from memory.memory_manager import MemoryManager, MemoryType, MemoryPriority


def test_performance_monitor_basic() -> bool:
    """测试性能监控器基本功能"""
    print("\n" + "="*60)
    print("测试1: 性能监控器基本功能")
    print("="*60)
    
    try:
        manager = MemoryManager()
        
        # 添加一些记忆
        print("\n添加记忆...")
        for i in range(5):
            memory_id = manager.add_memory(
                content=f"测试记忆内容 {i}",
                memory_type=MemoryType.CONVERSATION,
                tags=["测试", f"标签{i}"],
                emotional_intensity=0.5 + i * 0.1,
                priority=MemoryPriority.MEDIUM
            )
            print(f"  ✓ 添加记忆 {i+1}: {memory_id}")
            time.sleep(0.1)  # 短暂延迟以产生不同的时间戳
        
        # 搜索记忆
        print("\n搜索记忆...")
        results = manager.search_memories(
            query="测试记忆",
            memory_type=MemoryType.CONVERSATION,
            limit=3
        )
        print(f"  ✓ 搜索到 {len(results)} 条记忆")
        
        # 获取性能统计
        print("\n获取性能统计...")
        performance_stats = manager.get_performance_stats()
        print(f"  ✓ 操作类型数量: {len(performance_stats)}")
        
        # 打印 add_memory 操作的统计
        if "add_memory" in performance_stats:
            add_stats = performance_stats["add_memory"]
            print(f"\n  add_memory 操作统计:")
            print(f"    - 总次数: {add_stats['count']}")
            print(f"    - 成功次数: {add_stats['success_count']}")
            print(f"    - 失败次数: {add_stats['failure_count']}")
            print(f"    - 平均耗时: {add_stats['avg_duration_ms']:.2f} ms")
            print(f"    - 最小耗时: {add_stats['min_duration_ms']:.2f} ms")
            print(f"    - 最大耗时: {add_stats['max_duration_ms']:.2f} ms")
            print(f"    - 成功率: {add_stats['success_rate']*100:.2f}%")
        
        # 打印 search_memories 操作的统计
        if "search_memories" in performance_stats:
            search_stats = performance_stats["search_memories"]
            print(f"\n  search_memories 操作统计:")
            print(f"    - 总次数: {search_stats['count']}")
            print(f"    - 成功次数: {search_stats['success_count']}")
            print(f"    - 失败次数: {search_stats['failure_count']}")
            print(f"    - 平均耗时: {search_stats['avg_duration_ms']:.2f} ms")
            print(f"    - 缓存命中次数: {search_stats['cache_hits']}")
            print(f"    - 缓存未命中次数: {search_stats['cache_misses']}")
            print(f"    - 缓存命中率: {search_stats['cache_hit_rate']*100:.2f}%")
        
        # 获取性能摘要
        print("\n获取性能摘要...")
        summary = manager.get_performance_summary()
        print(f"  ✓ 总操作次数: {summary['total_operations']}")
        print(f"  ✓ 总成功次数: {summary['total_success']}")
        print(f"  ✓ 总失败次数: {summary['total_failures']}")
        print(f"  ✓ 总体成功率: {summary['overall_success_rate']*100:.2f}%")
        print(f"  ✓ 总体缓存命中率: {summary['overall_cache_hit_rate']*100:.2f}%")
        print(f"  ✓ 操作类型: {', '.join(summary['operations'])}")
        
        print("\n✓ 性能监控器基本功能测试通过")
        return True
        
    except Exception as e:
        print(f"\n✗ 性能监控器基本功能测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_performance_monitor_recent_metrics() -> bool:
    """测试获取最近性能指标"""
    print("\n" + "="*60)
    print("测试2: 获取最近性能指标")
    print("="*60)
    
    try:
        manager = MemoryManager()
        
        # 添加一些记忆
        print("\n添加记忆...")
        for i in range(3):
            memory_id = manager.add_memory(
                content=f"最近测试记忆 {i}",
                memory_type=MemoryType.CONVERSATION,
                tags=["最近测试"]
            )
            print(f"  ✓ 添加记忆 {i+1}: {memory_id}")
            time.sleep(0.05)
        
        # 获取最近的性能指标
        print("\n获取最近的性能指标...")
        recent_metrics = manager.get_recent_performance_metrics(limit=5)
        print(f"  ✓ 获取到 {len(recent_metrics)} 条最近指标")
        
        # 打印最近几条指标
        print("\n  最近的性能指标:")
        for i, metric in enumerate(recent_metrics[:3]):
            print(f"    {i+1}. 操作: {metric['operation']}")
            print(f"       耗时: {metric['duration_ms']:.2f} ms")
            print(f"       成功: {metric['success']}")
            print(f"       时间: {metric['timestamp']}")
        
        # 获取特定操作的最近指标
        print("\n获取 add_memory 操作的最近指标...")
        add_memory_metrics = manager.get_recent_performance_metrics(operation="add_memory", limit=3)
        print(f"  ✓ 获取到 {len(add_memory_metrics)} 条 add_memory 指标")
        
        print("\n✓ 获取最近性能指标测试通过")
        return True
        
    except Exception as e:
        print(f"\n✗ 获取最近性能指标测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_performance_monitor_export() -> bool:
    """测试导出性能指标"""
    print("\n" + "="*60)
    print("测试3: 导出性能指标")
    print("="*60)
    
    try:
        manager = MemoryManager()
        
        # 添加一些记忆
        print("\n添加记忆...")
        for i in range(3):
            memory_id = manager.add_memory(
                content=f"导出测试记忆 {i}",
                memory_type=MemoryType.CONVERSATION,
                tags=["导出测试"]
            )
            print(f"  ✓ 添加记忆 {i+1}: {memory_id}")
            time.sleep(0.05)
        
        # 搜索记忆
        print("\n搜索记忆...")
        results = manager.search_memories(
            query="导出测试",
            limit=2
        )
        print(f"  ✓ 搜索到 {len(results)} 条记忆")
        
        # 导出性能指标
        print("\n导出性能指标...")
        export_path = "/tmp/memory_performance_metrics.json"
        manager.export_performance_metrics(export_path)
        print(f"  ✓ 性能指标已导出到: {export_path}")
        
        # 验证文件是否存在
        if os.path.exists(export_path):
            print(f"  ✓ 导出文件存在")
            file_size = os.path.getsize(export_path)
            print(f"  ✓ 文件大小: {file_size} bytes")
        else:
            print(f"  ✗ 导出文件不存在")
            return False
        
        # 读取并验证导出文件
        print("\n验证导出文件内容...")
        import json
        with open(export_path, 'r', encoding='utf-8') as f:
            exported_data = json.load(f)
        
        print(f"  ✓ 导出数据包含以下键: {list(exported_data.keys())}")
        
        if "metrics" in exported_data:
            print(f"  ✓ 包含 {len(exported_data['metrics'])} 条性能指标")
        
        if "stats" in exported_data:
            print(f"  ✓ 包含操作统计信息")
        
        if "summary" in exported_data:
            print(f"  ✓ 包含性能摘要")
        
        if "exported_at" in exported_data:
            print(f"  ✓ 导出时间: {exported_data['exported_at']}")
        
        print("\n✓ 导出性能指标测试通过")
        return True
        
    except Exception as e:
        print(f"\n✗ 导出性能指标测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_performance_monitor_clear() -> bool:
    """测试清除性能指标"""
    print("\n" + "="*60)
    print("测试4: 清除性能指标")
    print("="*60)
    
    try:
        manager = MemoryManager()
        
        # 添加一些记忆
        print("\n添加记忆...")
        for i in range(3):
            memory_id = manager.add_memory(
                content=f"清除测试记忆 {i}",
                memory_type=MemoryType.CONVERSATION,
                tags=["清除测试"]
            )
            print(f"  ✓ 添加记忆 {i+1}: {memory_id}")
            time.sleep(0.05)
        
        # 获取清除前的性能摘要
        print("\n获取清除前的性能摘要...")
        summary_before = manager.get_performance_summary()
        print(f"  ✓ 总操作次数: {summary_before['total_operations']}")
        print(f"  ✓ 存储的指标数: {summary_before['metrics_stored']}")
        
        # 清除性能指标
        print("\n清除性能指标...")
        manager.clear_performance_metrics()
        print(f"  ✓ 性能指标已清除")
        
        # 获取清除后的性能摘要
        print("\n获取清除后的性能摘要...")
        summary_after = manager.get_performance_summary()
        print(f"  ✓ 总操作次数: {summary_after['total_operations']}")
        print(f"  ✓ 存储的指标数: {summary_after['metrics_stored']}")
        
        # 验证清除是否成功
        if summary_after['total_operations'] == 0 and summary_after['metrics_stored'] == 0:
            print(f"\n  ✓ 性能指标已成功清除")
        else:
            print(f"\n  ✗ 性能指标未完全清除")
            return False
        
        print("\n✓ 清除性能指标测试通过")
        return True
        
    except Exception as e:
        print(f"\n✗ 清除性能指标测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_performance_monitor_error_handling() -> bool:
    """测试性能监控器错误处理"""
    print("\n" + "="*60)
    print("测试5: 性能监控器错误处理")
    print("="*60)
    
    try:
        manager = MemoryManager()
        
        # 测试添加记忆时的错误处理
        print("\n测试添加记忆时的错误处理...")
        try:
            # 尝试添加无效的记忆类型
            memory_id = manager.add_memory(
                content="测试内容",
                memory_type="invalid_type",  # 无效类型
                tags=["测试"]
            )
            print(f"  ✗ 应该抛出异常但没有")
            return False
        except Exception as e:
            print(f"  ✓ 正确捕获异常: {type(e).__name__}")
        
        # 检查性能统计是否记录了失败
        print("\n检查性能统计...")
        performance_stats = manager.get_performance_stats()
        
        if "add_memory" in performance_stats:
            add_stats = performance_stats["add_memory"]
            print(f"  ✓ add_memory 失败次数: {add_stats['failure_count']}")
            print(f"  ✓ add_memory 成功率: {add_stats['success_rate']*100:.2f}%")
            
            if add_stats['failure_count'] > 0:
                print(f"  ✓ 失败操作已被正确记录")
            else:
                print(f"  ✗ 失败操作未被记录")
                return False
        
        print("\n✓ 性能监控器错误处理测试通过")
        return True
        
    except Exception as e:
        print(f"\n✗ 性能监控器错误处理测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_detailed_statistics() -> bool:
    """测试获取详细统计信息"""
    print("\n" + "="*60)
    print("测试6: 获取详细统计信息")
    print("="*60)
    
    try:
        manager = MemoryManager()
        
        # 添加一些记忆
        print("\n添加记忆...")
        for i in range(3):
            memory_id = manager.add_memory(
                content=f"详细统计测试记忆 {i}",
                memory_type=MemoryType.CONVERSATION,
                tags=["详细统计"]
            )
            print(f"  ✓ 添加记忆 {i+1}: {memory_id}")
            time.sleep(0.05)
        
        # 搜索记忆
        print("\n搜索记忆...")
        results = manager.search_memories(
            query="详细统计测试",
            limit=2
        )
        print(f"  ✓ 搜索到 {len(results)} 条记忆")
        
        # 获取详细统计信息
        print("\n获取详细统计信息...")
        detailed_stats = manager.get_detailed_statistics()
        print(f"  ✓ 包含 {len(detailed_stats)} 个顶级键")
        
        # 验证统计信息结构
        print("\n验证统计信息结构...")
        required_keys = ["total_memories", "memory_by_type", "memory_by_priority", 
                        "average_emotional_intensity", "average_strategic_score",
                        "cache_hit_rate", "average_response_time_ms", "performance"]
        
        for key in required_keys:
            if key in detailed_stats:
                print(f"  ✓ 包含键: {key}")
            else:
                print(f"  ✗ 缺少键: {key}")
                return False
        
        # 验证性能统计信息
        if "performance" in detailed_stats:
            performance = detailed_stats["performance"]
            print(f"\n  性能统计信息:")
            print(f"    - 包含摘要: {'summary' in performance}")
            print(f"    - 包含操作统计: {'operations' in performance}")
            
            if "summary" in performance:
                summary = performance["summary"]
                print(f"    - 总操作次数: {summary.get('total_operations', 0)}")
                print(f"    - 总体成功率: {summary.get('overall_success_rate', 0)*100:.2f}%")
        
        print("\n✓ 获取详细统计信息测试通过")
        return True
        
    except Exception as e:
        print(f"\n✗ 获取详细统计信息测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """运行所有测试"""
    print("\n" + "="*60)
    print("记忆系统性能监控器测试")
    print("="*60)
    
    tests = [
        ("性能监控器基本功能", test_performance_monitor_basic),
        ("获取最近性能指标", test_performance_monitor_recent_metrics),
        ("导出性能指标", test_performance_monitor_export),
        ("清除性能指标", test_performance_monitor_clear),
        ("性能监控器错误处理", test_performance_monitor_error_handling),
        ("获取详细统计信息", test_detailed_statistics)
    ]
    
    results = []
    for test_name, test_func in tests:
        result = test_func()
        results.append((test_name, result))
    
    # 打印测试结果汇总
    print("\n" + "="*60)
    print("测试结果汇总")
    print("="*60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✓ 通过" if result else "✗ 失败"
        print(f"{status}: {test_name}")
    
    print(f"\n总计: {passed}/{total} 测试通过")
    
    if passed == total:
        print("\n🎉 所有测试通过！")
        return 0
    else:
        print(f"\n❌ {total - passed} 个测试失败")
        return 1


if __name__ == "__main__":
    sys.exit(main())