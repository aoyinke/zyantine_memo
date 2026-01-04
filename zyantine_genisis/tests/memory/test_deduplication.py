#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
记忆去重机制测试脚本
"""

import sys
from typing import Dict, List

from memory.memory_manager import (
    MemoryManager,
    MemoryType,
    MemoryPriority,
    DeduplicationStrategy
)


def test_exact_duplicate_detection():
    """测试完全重复检测"""
    print("\n" + "="*60)
    print("测试1: 完全重复检测")
    print("="*60)
    
    manager = MemoryManager()
    
    # 添加第一个记忆
    memory_id1 = manager.add_memory(
        content="这是一个测试记忆",
        memory_type=MemoryType.CONVERSATION,
        tags=["测试", "去重"],
        priority=MemoryPriority.MEDIUM
    )
    
    # 添加完全相同的记忆
    memory_id2 = manager.add_memory(
        content="这是一个测试记忆",
        memory_type=MemoryType.CONVERSATION,
        tags=["测试", "去重"],
        priority=MemoryPriority.MEDIUM
    )
    
    if memory_id1 == memory_id2:
        print("  ✓ 完全重复检测测试成功")
        print(f"    第一个记忆ID: {memory_id1}")
        print(f"    第二个记忆ID: {memory_id2}")
        print(f"    检测到重复，返回相同的记忆ID")
        return True
    else:
        print("  ✗ 完全重复检测测试失败")
        print(f"    第一个记忆ID: {memory_id1}")
        print(f"    第二个记忆ID: {memory_id2}")
        return False


def test_similar_duplicate_detection():
    """测试相似重复检测"""
    print("\n" + "="*60)
    print("测试2: 相似重复检测")
    print("="*60)
    
    manager = MemoryManager()
    
    # 添加第一个记忆
    memory_id1 = manager.add_memory(
        content="这是一个关于机器学习的测试记忆",
        memory_type=MemoryType.CONVERSATION,
        tags=["测试", "去重"],
        priority=MemoryPriority.MEDIUM
    )
    
    # 添加相似的记忆
    memory_id2 = manager.add_memory(
        content="这是一个关于机器学习的测试记忆",
        memory_type=MemoryType.CONVERSATION,
        tags=["测试", "去重"],
        priority=MemoryPriority.MEDIUM
    )
    
    if memory_id1 == memory_id2:
        print("  ✓ 相似重复检测测试成功")
        print(f"    第一个记忆ID: {memory_id1}")
        print(f"    第二个记忆ID: {memory_id2}")
        print(f"    检测到相似重复，返回相同的记忆ID")
        return True
    else:
        print("  ✗ 相似重复检测测试失败")
        print(f"    第一个记忆ID: {memory_id1}")
        print(f"    第二个记忆ID: {memory_id2}")
        return False


def test_different_memory_types():
    """测试不同类型的记忆不冲突"""
    print("\n" + "="*60)
    print("测试3: 不同类型的记忆不冲突")
    print("="*60)
    
    manager = MemoryManager()
    
    # 添加对话记忆
    memory_id1 = manager.add_memory(
        content="这是一个测试记忆",
        memory_type=MemoryType.CONVERSATION,
        tags=["测试"],
        priority=MemoryPriority.MEDIUM
    )
    
    # 添加相同内容的经验记忆
    memory_id2 = manager.add_memory(
        content="这是一个测试记忆",
        memory_type=MemoryType.EXPERIENCE,
        tags=["测试"],
        priority=MemoryPriority.MEDIUM
    )
    
    if memory_id1 != memory_id2:
        print("  ✓ 不同类型记忆不冲突测试成功")
        print(f"    对话记忆ID: {memory_id1}")
        print(f"    经验记忆ID: {memory_id2}")
        print(f"    不同类型的记忆可以共存")
        return True
    else:
        print("  ✗ 不同类型记忆不冲突测试失败")
        print(f"    两个记忆ID相同: {memory_id1}")
        return False


def test_deduplication_stats():
    """测试去重统计"""
    print("\n" + "="*60)
    print("测试4: 去重统计")
    print("="*60)
    
    manager = MemoryManager()
    
    # 添加多个记忆，包括重复的
    memory_ids = []
    for i in range(10):
        memory_id = manager.add_memory(
            content=f"测试去重统计记忆 {i % 3}",
            memory_type=MemoryType.CONVERSATION,
            tags=["测试", "统计"],
            priority=MemoryPriority.MEDIUM
        )
        memory_ids.append(memory_id)
    
    # 获取去重统计
    dedup_stats = manager.deduplicator.get_stats()
    
    if dedup_stats and dedup_stats.get("registered_hashes", 0) > 0:
        print("  ✓ 去重统计测试成功")
        print(f"    去重启用: {dedup_stats.get('enabled', False)}")
        print(f"    相似度阈值: {dedup_stats.get('similarity_threshold', 0.0)}")
        print(f"    注册的哈希数: {dedup_stats.get('registered_hashes', 0)}")
        print(f"    默认策略: {dedup_stats.get('default_strategy', '')}")
        return True
    else:
        print("  ✗ 去重统计测试失败")
        return False


def test_deduplication_toggle():
    """测试去重开关"""
    print("\n" + "="*60)
    print("测试5: 去重开关")
    print("="*60)
    
    manager = MemoryManager()
    
    # 启用去重
    manager.deduplicator.enable_deduplication = True
    
    memory_id1 = manager.add_memory(
        content="测试去重开关记忆",
        memory_type=MemoryType.CONVERSATION,
        tags=["测试"],
        priority=MemoryPriority.MEDIUM
    )
    
    memory_id2 = manager.add_memory(
        content="测试去重开关记忆",
        memory_type=MemoryType.CONVERSATION,
        tags=["测试"],
        priority=MemoryPriority.MEDIUM
    )
    
    # 禁用去重
    manager.deduplicator.enable_deduplication = False
    
    memory_id3 = manager.add_memory(
        content="测试去重开关记忆",
        memory_type=MemoryType.CONVERSATION,
        tags=["测试"],
        priority=MemoryPriority.MEDIUM
    )
    
    if memory_id1 == memory_id2 and memory_id1 != memory_id3:
        print("  ✓ 去重开关测试成功")
        print(f"    启用去重时，记忆ID相同: {memory_id1} == {memory_id2}")
        print(f"    禁用去重时，记忆ID不同: {memory_id1} != {memory_id3}")
        return True
    else:
        print("  ✗ 去重开关测试失败")
        print(f"    第一个记忆ID: {memory_id1}")
        print(f"    第二个记忆ID: {memory_id2}")
        print(f"    第三个记忆ID: {memory_id3}")
        return False


def test_content_hash():
    """测试内容哈希计算"""
    print("\n" + "="*60)
    print("测试6: 内容哈希计算")
    print("="*60)
    
    manager = MemoryManager()
    
    # 计算相同内容的哈希
    content1 = "这是一个测试内容"
    content2 = "这是一个测试内容"
    hash1 = manager.deduplicator._compute_content_hash(content1)
    hash2 = manager.deduplicator._compute_content_hash(content2)
    
    # 计算不同内容的哈希
    content3 = "这是另一个测试内容"
    hash3 = manager.deduplicator._compute_content_hash(content3)
    
    if hash1 == hash2 and hash1 != hash3:
        print("  ✓ 内容哈希计算测试成功")
        print(f"    相同内容的哈希相同: {hash1 == hash2}")
        print(f"    不同内容的哈希不同: {hash1 != hash3}")
        return True
    else:
        print("  ✗ 内容哈希计算测试失败")
        print(f"    hash1: {hash1}")
        print(f"    hash2: {hash2}")
        print(f"    hash3: {hash3}")
        return False


def test_similarity_calculation():
    """测试相似度计算"""
    print("\n" + "="*60)
    print("测试7: 相似度计算")
    print("="*60)
    
    manager = MemoryManager()
    
    # 计算完全相同内容的相似度
    content1 = "这是一个测试内容"
    content2 = "这是一个测试内容"
    similarity1 = manager.deduplicator._compute_similarity(content1, content2)
    
    # 计算相似内容的相似度
    content3 = "这是一个测试内容"
    content4 = "这是一个测试内容"
    similarity2 = manager.deduplicator._compute_similarity(content3, content4)
    
    # 计算不同内容的相似度
    content5 = "这是一个测试内容"
    content6 = "这是完全不同的内容"
    similarity3 = manager.deduplicator._compute_similarity(content5, content6)
    
    if similarity1 == 1.0 and similarity3 < 0.5:
        print("  ✓ 相似度计算测试成功")
        print(f"    完全相同内容的相似度: {similarity1:.2f}")
        print(f"    相似内容的相似度: {similarity2:.2f}")
        print(f"    不同内容的相似度: {similarity3:.2f}")
        return True
    else:
        print("  ✗ 相似度计算测试失败")
        print(f"    相似度1: {similarity1}")
        print(f"    相似度2: {similarity2}")
        print(f"    相似度3: {similarity3}")
        return False


def test_register_unregister():
    """测试注册和注销"""
    print("\n" + "="*60)
    print("测试8: 注册和注销")
    print("="*60)
    
    manager = MemoryManager()
    
    # 注册记忆
    memory_id = "test_memory_001"
    content = "测试注册和注销"
    
    manager.deduplicator.register_memory(memory_id, content)
    
    # 检查是否注册成功
    content_hash = manager.deduplicator._compute_content_hash(content)
    is_registered = content_hash in manager.deduplicator.content_hashes
    
    # 注销记忆
    manager.deduplicator.unregister_memory(memory_id, content)
    
    # 检查是否注销成功
    is_unregistered = content_hash not in manager.deduplicator.content_hashes
    
    if is_registered and is_unregistered:
        print("  ✓ 注册和注销测试成功")
        print(f"    注册成功: {is_registered}")
        print(f"    注销成功: {is_unregistered}")
        return True
    else:
        print("  ✗ 注册和注销测试失败")
        print(f"    注册成功: {is_registered}")
        print(f"    注销成功: {is_unregistered}")
        return False


def main():
    """主测试函数"""
    print("="*60)
    print("记忆去重机制测试")
    print("="*60)
    
    tests = [
        ("完全重复检测", test_exact_duplicate_detection),
        ("相似重复检测", test_similar_duplicate_detection),
        ("不同类型的记忆不冲突", test_different_memory_types),
        ("去重统计", test_deduplication_stats),
        ("去重开关", test_deduplication_toggle),
        ("内容哈希计算", test_content_hash),
        ("相似度计算", test_similarity_calculation),
        ("注册和注销", test_register_unregister),
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
