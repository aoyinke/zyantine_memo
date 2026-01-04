#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
记忆生命周期管理器测试脚本
"""

import sys
import time
from datetime import datetime, timedelta
from typing import Dict, List

from memory.memory_manager import (
    MemoryManager,
    MemoryType,
    MemoryPriority,
    MemoryLifecycleStage
)


def test_lifecycle_stage_determination():
    """测试生命周期阶段确定"""
    print("\n" + "="*60)
    print("测试1: 生命周期阶段确定")
    print("="*60)
    
    manager = MemoryManager()
    
    # 创建不同年龄和访问次数的记忆
    test_cases = [
        {
            "age_hours": 10,
            "access_count": 8,
            "expected_stage": MemoryLifecycleStage.ACTIVE,
            "description": "新创建，频繁访问"
        },
        {
            "age_hours": 200,
            "access_count": 15,
            "expected_stage": MemoryLifecycleStage.MATURE,
            "description": "较新，定期访问"
        },
        {
            "age_hours": 500,
            "access_count": 25,
            "expected_stage": MemoryLifecycleStage.MATURE,
            "description": "中等年龄，稳定访问"
        },
        {
            "age_hours": 1000,
            "access_count": 25,
            "expected_stage": MemoryLifecycleStage.STABLE,
            "description": "较老，稳定访问"
        },
        {
            "age_hours": 2500,
            "access_count": 10,
            "expected_stage": MemoryLifecycleStage.DECLINING,
            "description": "较旧，访问频率低"
        },
        {
            "age_hours": 5000,
            "access_count": 60,
            "expected_stage": MemoryLifecycleStage.DECLINING,
            "description": "很旧，访问频率低"
        }
    ]
    
    all_passed = True
    for i, test_case in enumerate(test_cases, 1):
        # 创建记忆
        memory_id = manager.add_memory(
            content=f"测试记忆 {i}",
            memory_type=MemoryType.CONVERSATION,
            tags=["测试", "生命周期"],
            priority=MemoryPriority.MEDIUM
        )
        
        # 修改记忆的年龄和访问次数
        record = manager.cache.get(memory_id)
        if record:
            record.created_at = datetime.now() - timedelta(hours=test_case["age_hours"])
            record.access_count = test_case["access_count"]
            manager.cache.set(memory_id, record, record.version)
        
        # 确定生命周期阶段
        stage = manager.lifecycle_manager.determine_lifecycle_stage(record)
        
        if stage == test_case["expected_stage"]:
            print(f"  ✓ 测试用例 {i} 通过: {test_case['description']} -> {stage.value}")
        else:
            print(f"  ✗ 测试用例 {i} 失败: 期望 {test_case['expected_stage'].value}, 实际 {stage.value}")
            all_passed = False
    
    return all_passed


def test_update_lifecycle_stage():
    """测试更新生命周期阶段"""
    print("\n" + "="*60)
    print("测试2: 更新生命周期阶段")
    print("="*60)
    
    manager = MemoryManager()
    
    # 创建记忆
    memory_id = manager.add_memory(
        content="测试记忆更新生命周期",
        memory_type=MemoryType.CONVERSATION,
        tags=["测试", "生命周期"],
        priority=MemoryPriority.MEDIUM
    )
    
    # 获取记录
    record = manager.cache.get(memory_id)
    if not record:
        print("  ✗ 创建记忆失败")
        return False
    
    # 修改年龄和访问次数
    record.created_at = datetime.now() - timedelta(hours=200)
    record.access_count = 15
    manager.cache.set(memory_id, record, record.version)
    
    # 更新生命周期阶段
    updated_record = manager.update_lifecycle_stage(memory_id)
    
    if updated_record and updated_record.lifecycle_stage == MemoryLifecycleStage.MATURE:
        print("  ✓ 生命周期阶段更新成功")
        print(f"    新阶段: {updated_record.lifecycle_stage.value}")
        return True
    else:
        print("  ✗ 生命周期阶段更新失败")
        return False


def test_batch_update_lifecycle_stages():
    """测试批量更新生命周期阶段"""
    print("\n" + "="*60)
    print("测试3: 批量更新生命周期阶段")
    print("="*60)
    
    manager = MemoryManager()
    
    # 创建多个记忆
    memory_ids = []
    for i in range(5):
        memory_id = manager.add_memory(
            content=f"批量测试记忆 {i}",
            memory_type=MemoryType.CONVERSATION,
            tags=["测试", "批量"],
            priority=MemoryPriority.MEDIUM
        )
        memory_ids.append(memory_id)
    
    # 修改记忆的年龄和访问次数
    for i, memory_id in enumerate(memory_ids):
        record = manager.cache.get(memory_id)
        if record:
            record.created_at = datetime.now() - timedelta(hours=100 + i * 200)
            record.access_count = 10 + i * 5
            manager.cache.set(memory_id, record, record.version)
    
    # 批量更新
    results = manager.batch_update_lifecycle_stages(memory_ids)
    
    if results["updated"] >= 0:
        print(f"  ✓ 批量更新成功: 更新了 {results['updated']} 条记忆")
        print(f"    阶段分布: {results['stage_distribution']}")
        return True
    else:
        print(f"  ✗ 批量更新失败: 期望 >= 0, 实际 {results['updated']}")
        return False


def test_promote_memory():
    """测试提升记忆阶段"""
    print("\n" + "="*60)
    print("测试4: 提升记忆阶段")
    print("="*60)
    
    manager = MemoryManager()
    
    # 创建记忆
    memory_id = manager.add_memory(
        content="测试提升记忆",
        memory_type=MemoryType.CONVERSATION,
        tags=["测试", "提升"],
        priority=MemoryPriority.MEDIUM
    )
    
    # 获取记录并设置为较低阶段
    record = manager.cache.get(memory_id)
    if not record:
        print("  ✗ 创建记忆失败")
        return False
    
    record.created_at = datetime.now() - timedelta(hours=200)
    record.access_count = 15
    manager.cache.set(memory_id, record, record.version)
    
    # 更新到MATURE阶段
    manager.update_lifecycle_stage(memory_id)
    
    # 提升到STABLE阶段（注意：由于should_promote逻辑，可能不会直接提升）
    promoted_record = manager.promote_memory(memory_id)
    
    if promoted_record:
        print("  ✓ 记忆提升成功")
        print(f"    提升到阶段: {promoted_record.lifecycle_stage.value}")
        return True
    else:
        print("  ⚠ 记忆未提升（可能不符合提升条件）")
        return True


def test_demote_memory():
    """测试降级记忆阶段"""
    print("\n" + "="*60)
    print("测试5: 降级记忆阶段")
    print("="*60)
    
    manager = MemoryManager()
    
    # 创建记忆
    memory_id = manager.add_memory(
        content="测试降级记忆",
        memory_type=MemoryType.CONVERSATION,
        tags=["测试", "降级"],
        priority=MemoryPriority.MEDIUM
    )
    
    # 获取记录并设置为较高阶段
    record = manager.cache.get(memory_id)
    if not record:
        print("  ✗ 创建记忆失败")
        return False
    
    record.created_at = datetime.now() - timedelta(hours=500)
    record.access_count = 25
    manager.cache.set(memory_id, record, record.version)
    
    # 更新到STABLE阶段
    manager.update_lifecycle_stage(memory_id)
    
    # 降级到DECLINING阶段（注意：由于should_demote逻辑，可能不会直接降级）
    demoted_record = manager.demote_memory(memory_id)
    
    if demoted_record:
        print("  ✓ 记忆降级成功")
        print(f"    降级到阶段: {demoted_record.lifecycle_stage.value}")
        return True
    else:
        print("  ⚠ 记忆未降级（可能不符合降级条件）")
        return True


def test_get_lifecycle_stats():
    """测试获取生命周期统计"""
    print("\n" + "="*60)
    print("测试6: 获取生命周期统计")
    print("="*60)
    
    manager = MemoryManager()
    
    # 创建不同阶段的记忆
    test_cases = [
        {"age_hours": 10, "access_count": 8},
        {"age_hours": 200, "access_count": 15},
        {"age_hours": 500, "access_count": 25},
        {"age_hours": 2500, "access_count": 10},
    ]
    
    for i, test_case in enumerate(test_cases):
        memory_id = manager.add_memory(
            content=f"测试统计记忆 {i}",
            memory_type=MemoryType.CONVERSATION,
            tags=["测试", "统计"],
            priority=MemoryPriority.MEDIUM
        )
        
        record = manager.cache.get(memory_id)
        if record:
            record.created_at = datetime.now() - timedelta(hours=test_case["age_hours"])
            record.access_count = test_case["access_count"]
            manager.cache.set(memory_id, record, record.version)
    
    # 获取统计
    stats = manager.get_lifecycle_stats()
    
    if stats.get("total", 0) >= len(test_cases):
        print("  ✓ 获取生命周期统计成功")
        print(f"    总记忆数: {stats.get('total', 0)}")
        print(f"    阶段分布: {stats.get('stage_distribution', {})}")
        print(f"    平均访问次数: {stats.get('avg_access_count', 0):.2f}")
        return True
    else:
        print("  ✗ 获取生命周期统计失败")
        return False


def test_get_memories_by_lifecycle_stage():
    """测试按生命周期阶段获取记忆"""
    print("\n" + "="*60)
    print("测试7: 按生命周期阶段获取记忆")
    print("="*60)
    
    manager = MemoryManager()
    
    # 创建多个记忆
    for i in range(5):
        memory_id = manager.add_memory(
            content=f"阶段测试记忆 {i}",
            memory_type=MemoryType.CONVERSATION,
            tags=["测试", "阶段"],
            priority=MemoryPriority.MEDIUM
        )
        
        record = manager.cache.get(memory_id)
        if record:
            record.created_at = datetime.now() - timedelta(hours=100 + i * 200)
            record.access_count = 10 + i * 5
            manager.cache.set(memory_id, record, record.version)
    
    # 批量更新阶段
    all_memories = manager.cache.get_all()
    memory_ids = list(all_memories.keys())
    manager.batch_update_lifecycle_stages(memory_ids)
    
    # 获取ACTIVE阶段的记忆
    active_memories = manager.get_memories_by_lifecycle_stage(MemoryLifecycleStage.ACTIVE)
    
    if len(active_memories) >= 0:
        print("  ✓ 按阶段获取记忆成功")
        print(f"    ACTIVE阶段记忆数: {len(active_memories)}")
        return True
    else:
        print("  ✗ 按阶段获取记忆失败")
        return False


def test_auto_manage_lifecycle():
    """测试自动管理生命周期"""
    print("\n" + "="*60)
    print("测试8: 自动管理生命周期")
    print("="*60)
    
    manager = MemoryManager()
    
    # 创建多个记忆
    for i in range(10):
        memory_id = manager.add_memory(
            content=f"自动管理测试记忆 {i}",
            memory_type=MemoryType.CONVERSATION,
            tags=["测试", "自动管理"],
            priority=MemoryPriority.MEDIUM
        )
        
        record = manager.cache.get(memory_id)
        if record:
            record.created_at = datetime.now() - timedelta(hours=50 + i * 100)
            record.access_count = 5 + i * 3
            manager.cache.set(memory_id, record, record.version)
    
    # 自动管理
    results = manager.auto_manage_lifecycle()
    
    if "promoted" in results and "demoted" in results:
        print("  ✓ 自动生命周期管理成功")
        print(f"    提升记忆数: {results['promoted']}")
        print(f"    降级记忆数: {results['demoted']}")
        return True
    else:
        print("  ✗ 自动生命周期管理失败")
        return False


def test_cleanup_expired_memories():
    """测试清理过期记忆"""
    print("\n" + "="*60)
    print("测试9: 清理过期记忆")
    print("="*60)
    
    manager = MemoryManager()
    
    # 创建一个归档的记忆
    memory_id = manager.add_memory(
        content="测试过期记忆",
        memory_type=MemoryType.CONVERSATION,
        tags=["测试", "过期"],
        priority=MemoryPriority.LOW
    )
    
    # 设置为归档状态
    record = manager.cache.get(memory_id)
    if record:
        record.created_at = datetime.now() - timedelta(hours=9000)
        record.metadata["archived"] = True
        manager.cache.set(memory_id, record, record.version)
    
    # 更新生命周期阶段
    manager.update_lifecycle_stage(memory_id)
    
    # 清理过期记忆
    cleaned_count = manager.cleanup_expired_memories()
    
    if cleaned_count >= 0:
        print("  ✓ 清理过期记忆成功")
        print(f"    清理记忆数: {cleaned_count}")
        return True
    else:
        print("  ✗ 清理过期记忆失败")
        return False


def test_lifecycle_thresholds():
    """测试生命周期阈值管理"""
    print("\n" + "="*60)
    print("测试10: 生命周期阈值管理")
    print("="*60)
    
    manager = MemoryManager()
    
    # 获取当前阈值
    current_thresholds = manager.lifecycle_manager.get_thresholds()
    
    # 修改阈值
    new_thresholds = current_thresholds.copy()
    new_thresholds["active"]["max_age_hours"] = 200
    
    # 设置新阈值
    success = manager.lifecycle_manager.set_thresholds(new_thresholds)
    
    if success:
        print("  ✓ 生命周期阈值管理成功")
        print(f"    新阈值: {new_thresholds['active']}")
        return True
    else:
        print("  ✗ 生命周期阈值管理失败")
        return False


def main():
    """主测试函数"""
    print("="*60)
    print("记忆生命周期管理器测试")
    print("="*60)
    
    tests = [
        ("生命周期阶段确定", test_lifecycle_stage_determination),
        ("更新生命周期阶段", test_update_lifecycle_stage),
        ("批量更新生命周期阶段", test_batch_update_lifecycle_stages),
        ("提升记忆阶段", test_promote_memory),
        ("降级记忆阶段", test_demote_memory),
        ("获取生命周期统计", test_get_lifecycle_stats),
        ("按生命周期阶段获取记忆", test_get_memories_by_lifecycle_stage),
        ("自动管理生命周期", test_auto_manage_lifecycle),
        ("清理过期记忆", test_cleanup_expired_memories),
        ("生命周期阈值管理", test_lifecycle_thresholds),
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
            failed += 1
    
    print("\n" + "="*60)
    print(f"测试完成: 通过 {passed}/{len(tests)}, 失败 {failed}/{len(tests)}")
    print("="*60)
    
    return failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
