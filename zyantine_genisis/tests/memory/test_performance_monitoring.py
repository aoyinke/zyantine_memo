#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
记忆性能监控和统计测试脚本
"""

import sys
import time
from datetime import datetime, timedelta
from typing import Dict, List

from memory.memory_manager import (
    MemoryManager,
    MemoryType,
    MemoryPriority
)


def test_performance_monitoring():
    """测试性能监控"""
    print("\n" + "="*60)
    print("测试1: 性能监控")
    print("="*60)
    
    manager = MemoryManager()
    
    # 创建多个记忆
    memory_ids = []
    for i in range(10):
        memory_id = manager.add_memory(
            content=f"测试性能监控记忆 {i}",
            memory_type=MemoryType.CONVERSATION,
            tags=["测试", "性能"],
            priority=MemoryPriority.MEDIUM
        )
        memory_ids.append(memory_id)
    
    # 执行多次操作以生成性能数据
    for memory_id in memory_ids:
        manager.cache.get(memory_id)
        manager.search_memories("测试")
    
    # 获取性能统计
    stats = manager.get_performance_stats()
    
    if stats:
        print("  ✓ 性能监控测试成功")
        print(f"    操作类型数: {len(stats)}")
        for operation, op_stats in stats.items():
            print(f"    {operation}:")
            print(f"      总次数: {op_stats.get('count', 0)}")
            print(f"      成功率: {op_stats.get('success_rate', 0):.2%}")
            print(f"      平均耗时: {op_stats.get('avg_duration_ms', 0):.2f} ms")
        return True
    else:
        print("  ✗ 性能监控测试失败")
        return False


def test_storage_stats():
    """测试存储统计"""
    print("\n" + "="*60)
    print("测试2: 存储统计")
    print("="*60)
    
    manager = MemoryManager()
    
    # 创建多个记忆
    memory_ids = []
    for i in range(20):
        memory_id = manager.add_memory(
            content=f"测试存储统计记忆 {i} - " + "这是一个较长的内容。" * 5,
            memory_type=MemoryType.CONVERSATION,
            tags=["测试", "存储"],
            priority=MemoryPriority.MEDIUM
        )
        memory_ids.append(memory_id)
        
        # 获取记忆记录
        record = manager.cache.get(memory_id)
        
        # 模拟不同的访问频率
        access_count = 20 - i
        for _ in range(access_count):
            manager.cache.get(memory_id)
        
        # 手动将记忆存储到分层存储中
        if record:
            if access_count >= 10:
                manager.storage_optimizer._store_in_hot(memory_id, record)
            elif access_count >= 3:
                manager.storage_optimizer._store_in_warm(memory_id, record)
            else:
                manager.storage_optimizer._store_in_cold(memory_id, record)
    
    # 优化存储
    manager.storage_optimizer.optimize_storage()
    
    # 更新存储统计
    storage_stats = manager.storage_optimizer.get_stats()
    manager.performance_monitor.update_storage_stats(storage_stats)
    
    # 获取存储统计
    storage_stats = manager.performance_monitor.get_storage_stats()
    
    if storage_stats and storage_stats.get("total_items", 0) > 0:
        print("  ✓ 存储统计测试成功")
        print(f"    总记忆数: {storage_stats.get('total_items', 0)}")
        print(f"    热数据: {storage_stats.get('hot_items', 0)}")
        print(f"    温数据: {storage_stats.get('warm_items', 0)}")
        print(f"    冷数据: {storage_stats.get('cold_items', 0)}")
        print(f"    总大小: {storage_stats.get('total_size_bytes', 0)} 字节")
        print(f"    压缩比: {storage_stats.get('compression_ratio', 1.0):.2f}")
        return True
    else:
        print("  ✗ 存储统计测试失败")
        return False


def test_lifecycle_stats():
    """测试生命周期统计"""
    print("\n" + "="*60)
    print("测试3: 生命周期统计")
    print("="*60)
    
    manager = MemoryManager()
    
    # 创建多个记忆
    memory_ids = []
    for i in range(15):
        memory_id = manager.add_memory(
            content=f"测试生命周期统计记忆 {i}",
            memory_type=MemoryType.CONVERSATION,
            tags=["测试", "生命周期"],
            priority=MemoryPriority.MEDIUM
        )
        memory_ids.append(memory_id)
    
    # 批量更新生命周期阶段
    result = manager.batch_update_lifecycle_stages(memory_ids)
    
    # 获取生命周期统计
    lifecycle_stats = manager.performance_monitor.get_lifecycle_stats()
    
    if lifecycle_stats:
        print("  ✓ 生命周期统计测试成功")
        print(f"    更新数量: {result.get('updated', 0)}")
        print(f"    未变化数量: {result.get('unchanged', 0)}")
        print(f"    阶段分布: {result.get('stage_distribution', {})}")
        print(f"    监控统计: {lifecycle_stats}")
        return True
    else:
        print("  ✗ 生命周期统计测试失败")
        return False


def test_comprehensive_report():
    """测试综合性能报告"""
    print("\n" + "="*60)
    print("测试4: 综合性能报告")
    print("="*60)
    
    manager = MemoryManager()
    
    # 创建多个记忆
    memory_ids = []
    for i in range(25):
        memory_id = manager.add_memory(
            content=f"测试综合报告记忆 {i}",
            memory_type=MemoryType.CONVERSATION,
            tags=["测试", "综合"],
            priority=MemoryPriority.HIGH if i < 10 else MemoryPriority.MEDIUM
        )
        memory_ids.append(memory_id)
        
        # 模拟访问
        for _ in range(10 - i // 3):
            manager.cache.get(memory_id)
    
    # 优化存储
    manager.storage_optimizer.optimize_storage()
    
    # 更新生命周期
    manager.batch_update_lifecycle_stages(memory_ids)
    
    # 获取综合报告
    report = manager.performance_monitor.get_comprehensive_report()
    
    if report:
        print("  ✓ 综合性能报告测试成功")
        print(f"    性能摘要:")
        perf_summary = report.get("performance", {}).get("summary", {})
        print(f"      总操作数: {perf_summary.get('total_operations', 0)}")
        print(f"      成功率: {perf_summary.get('overall_success_rate', 0):.2%}")
        print(f"      缓存命中率: {perf_summary.get('overall_cache_hit_rate', 0):.2%}")
        
        print(f"    存储统计:")
        storage = report.get("storage", {})
        print(f"      总记忆数: {storage.get('total_items', 0)}")
        print(f"      热数据: {storage.get('hot_items', 0)}")
        print(f"      温数据: {storage.get('warm_items', 0)}")
        print(f"      冷数据: {storage.get('cold_items', 0)}")
        
        print(f"    生命周期统计:")
        lifecycle = report.get("lifecycle", {})
        print(f"      {lifecycle}")
        
        return True
    else:
        print("  ✗ 综合性能报告测试失败")
        return False


def test_real_time_monitoring():
    """测试实时监控"""
    print("\n" + "="*60)
    print("测试5: 实时监控")
    print("="*60)
    
    manager = MemoryManager()
    
    # 启动实时监控
    callback_called = []
    
    def monitoring_callback():
        callback_called.append(datetime.now())
        print("    监控回调已执行")
    
    manager.performance_monitor.start_monitoring(interval=2, callback=monitoring_callback)
    
    print("  实时监控已启动，等待回调...")
    time.sleep(3)
    
    # 停止监控
    manager.performance_monitor.stop_monitoring()
    
    if len(callback_called) > 0:
        print("  ✓ 实时监控测试成功")
        print(f"    回调执行次数: {len(callback_called)}")
        return True
    else:
        print("  ✗ 实时监控测试失败")
        return False


def test_export_metrics():
    """测试导出性能指标"""
    print("\n" + "="*60)
    print("测试6: 导出性能指标")
    print("="*60)
    
    manager = MemoryManager()
    
    # 创建多个记忆
    memory_ids = []
    for i in range(10):
        memory_id = manager.add_memory(
            content=f"测试导出指标记忆 {i}",
            memory_type=MemoryType.CONVERSATION,
            tags=["测试", "导出"],
            priority=MemoryPriority.MEDIUM
        )
        memory_ids.append(memory_id)
    
    # 执行操作
    for memory_id in memory_ids:
        manager.cache.get(memory_id)
    
    # 导出指标
    import tempfile
    import os
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        filepath = f.name
    
    try:
        manager.performance_monitor.export_metrics(filepath)
        
        # 检查文件是否存在
        if os.path.exists(filepath):
            print("  ✓ 导出性能指标测试成功")
            print(f"    导出路径: {filepath}")
            
            # 读取文件内容
            import json
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
                print(f"    导出数据包含: {list(data.keys())}")
            
            return True
        else:
            print("  ✗ 导出性能指标测试失败 - 文件不存在")
            return False
    finally:
        # 清理临时文件
        if os.path.exists(filepath):
            os.remove(filepath)


def test_detailed_statistics():
    """测试详细统计信息"""
    print("\n" + "="*60)
    print("测试7: 详细统计信息")
    print("="*60)
    
    manager = MemoryManager()
    
    # 创建多个记忆
    memory_ids = []
    for i in range(15):
        memory_id = manager.add_memory(
            content=f"测试详细统计记忆 {i}",
            memory_type=MemoryType.CONVERSATION,
            tags=["测试", "详细"],
            priority=MemoryPriority.HIGH if i < 5 else MemoryPriority.MEDIUM
        )
        memory_ids.append(memory_id)
    
    # 执行操作
    for memory_id in memory_ids:
        manager.cache.get(memory_id)
    
    # 获取详细统计
    stats = manager.get_detailed_statistics()
    
    if stats:
        print("  ✓ 详细统计信息测试成功")
        print(f"    总记忆数: {stats.get('total_memories', 0)}")
        print(f"    包含的统计类型: {list(stats.keys())}")
        
        if "performance" in stats:
            print(f"    性能统计: {list(stats['performance'].keys())}")
        if "storage" in stats:
            print(f"    存储统计: {list(stats['storage'].keys())}")
        if "lifecycle" in stats:
            print(f"    生命周期统计: {list(stats['lifecycle'].keys())}")
        if "deduplication" in stats:
            print(f"    去重统计: {list(stats['deduplication'].keys())}")
        
        return True
    else:
        print("  ✗ 详细统计信息测试失败")
        return False


def main():
    """主测试函数"""
    print("="*60)
    print("记忆性能监控和统计测试")
    print("="*60)
    
    tests = [
        ("性能监控", test_performance_monitoring),
        ("存储统计", test_storage_stats),
        ("生命周期统计", test_lifecycle_stats),
        ("综合性能报告", test_comprehensive_report),
        ("实时监控", test_real_time_monitoring),
        ("导出性能指标", test_export_metrics),
        ("详细统计信息", test_detailed_statistics),
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
    
    print("\n" + "="*60)
    print(f"测试完成: 通过 {passed}/{len(tests)}, 失败 {failed}/{len(tests)}")
    print("="*60)
    
    return failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
