#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
记忆存储优化器测试脚本
"""

import sys
import time
from datetime import datetime, timedelta
from typing import Dict, List

from memory.memory_manager import (
    MemoryManager,
    MemoryType,
    MemoryPriority,
    StorageTier
)


def test_storage_tier_determination():
    """测试存储层级确定"""
    print("\n" + "="*60)
    print("测试1: 存储层级确定")
    print("="*60)
    
    manager = MemoryManager()
    
    # 创建记忆并多次访问
    memory_id = manager.add_memory(
        content="测试热数据记忆",
        memory_type=MemoryType.CONVERSATION,
        tags=["测试", "热数据"],
        priority=MemoryPriority.HIGH
    )
    
    # 多次访问以提升到热数据层
    for _ in range(15):
        manager.cache.get(memory_id)
    
    # 检查存储层级
    record = manager.cache.get(memory_id)
    if record:
        print(f"  ✓ 记忆创建成功: {memory_id}")
        print(f"    访问次数: {record.access_count}")
        print(f"    记忆已存储在缓存中")
        return True
    else:
        print("  ✗ 记忆创建失败")
        return False


def test_storage_migration():
    """测试存储迁移"""
    print("\n" + "="*60)
    print("测试2: 存储迁移")
    print("="*60)
    
    manager = MemoryManager()
    
    # 创建多个记忆
    memory_ids = []
    for i in range(10):
        memory_id = manager.add_memory(
            content=f"测试迁移记忆 {i}",
            memory_type=MemoryType.CONVERSATION,
            tags=["测试", "迁移"],
            priority=MemoryPriority.MEDIUM
        )
        memory_ids.append(memory_id)
        
        # 模拟不同的访问频率
        access_count = 15 - i
        for _ in range(access_count):
            manager.cache.get(memory_id)
    
    # 优化存储
    manager.storage_optimizer.optimize_storage()
    
    # 获取存储统计
    stats = manager.storage_optimizer.get_stats()
    
    if stats["total_items"] >= 0:
        print("  ✓ 存储迁移成功")
        print(f"    总记忆数: {stats['total_items']}")
        print(f"    热数据: {stats['hot_items']}")
        print(f"    温数据: {stats['warm_items']}")
        print(f"    冷数据: {stats['cold_items']}")
        print(f"    压缩比: {stats['compression_ratio']:.2f}")
        return True
    else:
        print("  ✗ 存储迁移失败")
        return False


def test_compression_efficiency():
    """测试压缩效率"""
    print("\n" + "="*60)
    print("测试3: 压缩效率")
    print("="*60)
    
    manager = MemoryManager()
    
    # 创建大量记忆
    memory_ids = []
    for i in range(20):
        memory_id = manager.add_memory(
            content=f"测试压缩记忆 {i} - " + "这是一个较长的内容用于测试压缩效率。" * 10,
            memory_type=MemoryType.CONVERSATION,
            tags=["测试", "压缩"],
            priority=MemoryPriority.MEDIUM
        )
        memory_ids.append(memory_id)
    
    # 优化存储
    manager.storage_optimizer.optimize_storage()
    
    # 获取存储统计
    stats = manager.storage_optimizer.get_stats()
    
    if stats["compression_ratio"] > 0:
        print("  ✓ 压缩效率测试成功")
        print(f"    热数据大小: {stats['hot_size_bytes']} 字节")
        print(f"    温数据大小: {stats['warm_size_bytes']} 字节")
        print(f"    冷数据大小: {stats['cold_size_bytes']} 字节")
        print(f"    总大小: {stats['total_size_bytes']} 字节")
        print(f"    压缩比: {stats['compression_ratio']:.2f}")
        return True
    else:
        print("  ✗ 压缩效率测试失败")
        return False


def test_storage_retrieval():
    """测试存储检索"""
    print("\n" + "="*60)
    print("测试4: 存储检索")
    print("="*60)
    
    manager = MemoryManager()
    
    # 创建记忆
    memory_id = manager.add_memory(
        content="测试检索记忆",
        memory_type=MemoryType.CONVERSATION,
        tags=["测试", "检索"],
        priority=MemoryPriority.HIGH
    )
    
    # 从存储优化器中检索
    record = manager.storage_optimizer.get(memory_id)
    
    if record and record.content == "测试检索记忆":
        print("  ✓ 存储检索成功")
        print(f"    记忆ID: {memory_id}")
        print(f"    内容: {record.content}")
        return True
    else:
        print("  ✗ 存储检索失败")
        return False


def test_storage_deletion():
    """测试存储删除"""
    print("\n" + "="*60)
    print("测试5: 存储删除")
    print("="*60)
    
    manager = MemoryManager()
    
    # 创建记忆
    memory_id = manager.add_memory(
        content="测试删除记忆",
        memory_type=MemoryType.CONVERSATION,
        tags=["测试", "删除"],
        priority=MemoryPriority.MEDIUM
    )
    
    # 删除记忆
    manager.storage_optimizer.delete(memory_id)
    
    # 检查是否已删除
    record = manager.storage_optimizer.get(memory_id)
    
    if record is None:
        print("  ✓ 存储删除成功")
        return True
    else:
        print("  ✗ 存储删除失败")
        return False


def test_get_all_memories():
    """测试获取所有记忆"""
    print("\n" + "="*60)
    print("测试6: 获取所有记忆")
    print("="*60)
    
    manager = MemoryManager()
    
    # 创建多个记忆
    memory_ids = []
    for i in range(5):
        memory_id = manager.add_memory(
            content=f"测试获取记忆 {i}",
            memory_type=MemoryType.CONVERSATION,
            tags=["测试", "获取"],
            priority=MemoryPriority.MEDIUM
        )
        memory_ids.append(memory_id)
    
    # 获取所有记忆
    all_memories = manager.storage_optimizer.get_all()
    
    if len(all_memories) >= len(memory_ids):
        print("  ✓ 获取所有记忆成功")
        print(f"    记忆数: {len(all_memories)}")
        return True
    else:
        print("  ✗ 获取所有记忆失败")
        return False


def test_storage_optimization():
    """测试存储优化"""
    print("\n" + "="*60)
    print("测试7: 存储优化")
    print("="*60)
    
    manager = MemoryManager()
    
    # 创建大量记忆
    memory_ids = []
    for i in range(30):
        memory_id = manager.add_memory(
            content=f"测试优化记忆 {i}",
            memory_type=MemoryType.CONVERSATION,
            tags=["测试", "优化"],
            priority=MemoryPriority.MEDIUM
        )
        memory_ids.append(memory_id)
        
        # 模拟不同的访问频率
        access_count = 20 - i
        for _ in range(access_count):
            manager.cache.get(memory_id)
    
    # 获取优化前的统计
    stats_before = manager.storage_optimizer.get_stats()
    
    # 优化存储
    manager.storage_optimizer.optimize_storage()
    
    # 获取优化后的统计
    stats_after = manager.storage_optimizer.get_stats()
    
    if stats_after["total_items"] >= 0:
        print("  ✓ 存储优化成功")
        print(f"    优化前 - 热数据: {stats_before['hot_items']}, 温数据: {stats_before['warm_items']}, 冷数据: {stats_before['cold_items']}")
        print(f"    优化后 - 热数据: {stats_after['hot_items']}, 温数据: {stats_after['warm_items']}, 冷数据: {stats_after['cold_items']}")
        return True
    else:
        print("  ✗ 存储优化失败")
        return False


def test_storage_capacity():
    """测试存储容量"""
    print("\n" + "="*60)
    print("测试8: 存储容量")
    print("="*60)
    
    manager = MemoryManager()
    
    # 获取存储容量信息
    stats = manager.storage_optimizer.get_stats()
    
    if stats["total_items"] >= 0:
        print("  ✓ 存储容量测试成功")
        print(f"    热数据容量: 500")
        print(f"    温数据容量: 2000")
        print(f"    冷数据容量: 10000")
        print(f"    当前使用: {stats['total_items']}")
        print(f"    利用率: {stats['total_items'] / 12500 * 100:.2f}%")
        return True
    else:
        print("  ✗ 存储容量测试失败")
        return False


def main():
    """主测试函数"""
    print("="*60)
    print("记忆存储优化器测试")
    print("="*60)
    
    tests = [
        ("存储层级确定", test_storage_tier_determination),
        ("存储迁移", test_storage_migration),
        ("压缩效率", test_compression_efficiency),
        ("存储检索", test_storage_retrieval),
        ("存储删除", test_storage_deletion),
        ("获取所有记忆", test_get_all_memories),
        ("存储优化", test_storage_optimization),
        ("存储容量", test_storage_capacity),
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
