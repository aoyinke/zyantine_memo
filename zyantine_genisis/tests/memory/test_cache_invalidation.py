#!/usr/bin/env python3
"""
缓存失效机制和版本控制测试脚本
"""
import asyncio
import json
from datetime import datetime
from memory.memory_manager import MemoryManager, MemoryType, MemoryPriority, MemoryCache, MemoryRecord


def test_cache_version_control() -> bool:
    """测试缓存版本控制"""
    print("\n" + "="*60)
    print("测试1: 缓存版本控制")
    print("="*60)
    
    try:
        cache = MemoryCache(max_size=100, ttl_hours=1)
        
        # 创建记忆记录
        record1 = MemoryRecord(
            memory_id="test_001",
            content="测试内容1",
            memory_type=MemoryType.CONVERSATION,
            metadata={},
            tags=["test"],
            priority=MemoryPriority.MEDIUM,
            created_at=datetime.now()
        )
        
        # 第一次添加（版本1）
        cache.set("test_001", record1, version=1)
        version1 = cache.get_version("test_001")
        print(f"✓ 第一次添加，版本号: {version1}")
        assert version1 == 1, f"版本号应为1，实际为{version1}"
        
        # 更新记忆记录
        record2 = MemoryRecord(
            memory_id="test_001",
            content="测试内容2（已更新）",
            memory_type=MemoryType.CONVERSATION,
            metadata={},
            tags=["test", "updated"],
            priority=MemoryPriority.MEDIUM,
            created_at=datetime.now()
        )
        
        # 第二次添加（版本2）
        cache.set("test_001", record2, version=2)
        version2 = cache.get_version("test_001")
        print(f"✓ 第二次添加，版本号: {version2}")
        assert version2 == 2, f"版本号应为2，实际为{version2}"
        
        # 验证缓存内容已更新
        cached_record = cache.get("test_001")
        assert cached_record.content == "测试内容2（已更新）", "缓存内容未更新"
        print(f"✓ 缓存内容已正确更新: {cached_record.content}")
        
        # 尝试用旧版本更新（不应更新）
        record3 = MemoryRecord(
            memory_id="test_001",
            content="测试内容3（旧版本）",
            memory_type=MemoryType.CONVERSATION,
            metadata={},
            tags=["test"],
            priority=MemoryPriority.MEDIUM,
            created_at=datetime.now()
        )
        
        cache.set("test_001", record3, version=1)  # 旧版本
        version_after_old = cache.get_version("test_001")
        cached_record_after_old = cache.get("test_001")
        
        assert version_after_old == 2, "旧版本不应更新缓存"
        assert cached_record_after_old.content == "测试内容2（已更新）", "旧版本不应覆盖缓存"
        print(f"✓ 旧版本未覆盖缓存，版本号仍为: {version_after_old}")
        
        print("\n✓ 缓存版本控制测试通过")
        return True
        
    except Exception as e:
        print(f"\n✗ 缓存版本控制测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_cache_invalidation_by_prefix() -> bool:
    """测试按前缀失效缓存"""
    print("\n" + "="*60)
    print("测试2: 按前缀失效缓存")
    print("="*60)
    
    try:
        cache = MemoryCache(max_size=100, ttl_hours=1)
        
        # 添加多个缓存项
        for i in range(5):
            record = MemoryRecord(
                memory_id=f"conversation_history_{i}",
                content=f"对话历史{i}",
                memory_type=MemoryType.CONVERSATION,
                metadata={},
                tags=["conversation"],
                priority=MemoryPriority.MEDIUM,
                created_at=datetime.now()
            )
            cache.set(f"conversation_history_{i}", record, version=1)
        
        # 添加其他缓存项
        other_record = MemoryRecord(
            memory_id="user_profile_001",
            content="用户资料",
            memory_type=MemoryType.USER_PROFILE,
            metadata={},
            tags=["user"],
            priority=MemoryPriority.HIGH,
            created_at=datetime.now()
        )
        cache.set("user_profile_001", other_record, version=1)
        
        print(f"✓ 添加了6个缓存项")
        
        # 按前缀失效
        cleared_count = cache.invalidate_by_prefix("conversation_history_")
        print(f"✓ 清除了 {cleared_count} 个对话历史缓存项")
        assert cleared_count == 5, f"应清除5项，实际清除{cleared_count}项"
        
        # 验证对话历史缓存已清除
        for i in range(5):
            cached = cache.get(f"conversation_history_{i}")
            assert cached is None, f"conversation_history_{i} 应已被清除"
        
        # 验证其他缓存项仍存在
        other_cached = cache.get("user_profile_001")
        assert other_cached is not None, "user_profile_001 应仍存在"
        print(f"✓ 其他缓存项（user_profile_001）仍存在")
        
        print("\n✓ 按前缀失效缓存测试通过")
        return True
        
    except Exception as e:
        print(f"\n✗ 按前缀失效缓存测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_cache_invalidation_by_tag() -> bool:
    """测试按标签失效缓存"""
    print("\n" + "="*60)
    print("测试3: 按标签失效缓存")
    print("="*60)
    
    try:
        cache = MemoryCache(max_size=100, ttl_hours=1)
        
        # 添加带不同标签的缓存项
        records = [
            ("test_001", ["conversation", "important"], "对话1"),
            ("test_002", ["conversation", "temp"], "对话2"),
            ("test_003", ["user", "profile"], "用户资料"),
            ("test_004", ["conversation"], "对话3"),
        ]
        
        for memory_id, tags, content in records:
            record = MemoryRecord(
                memory_id=memory_id,
                content=content,
                memory_type=MemoryType.CONVERSATION,
                metadata={},
                tags=tags,
                priority=MemoryPriority.MEDIUM,
                created_at=datetime.now()
            )
            cache.set(memory_id, record, version=1)
        
        print(f"✓ 添加了4个缓存项")
        
        # 按标签失效
        cleared_count = cache.invalidate_by_tag("conversation")
        print(f"✓ 清除了标签 'conversation' 的 {cleared_count} 个缓存项")
        assert cleared_count == 3, f"应清除3项，实际清除{cleared_count}项"
        
        # 验证带conversation标签的缓存已清除
        assert cache.get("test_001") is None, "test_001 应已被清除"
        assert cache.get("test_002") is None, "test_002 应已被清除"
        assert cache.get("test_004") is None, "test_004 应已被清除"
        
        # 验证不带conversation标签的缓存仍存在
        assert cache.get("test_003") is not None, "test_003 应仍存在"
        print(f"✓ 不带conversation标签的缓存项（test_003）仍存在")
        
        print("\n✓ 按标签失效缓存测试通过")
        return True
        
    except Exception as e:
        print(f"\n✗ 按标签失效缓存测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_cache_invalidation_by_memory_type() -> bool:
    """测试按记忆类型失效缓存"""
    print("\n" + "="*60)
    print("测试4: 按记忆类型失效缓存")
    print("="*60)
    
    try:
        cache = MemoryCache(max_size=100, ttl_hours=1)
        
        # 添加不同类型的缓存项
        records = [
            ("conv_001", MemoryType.CONVERSATION, "对话1"),
            ("conv_002", MemoryType.CONVERSATION, "对话2"),
            ("user_001", MemoryType.USER_PROFILE, "用户资料"),
            ("exp_001", MemoryType.EXPERIENCE, "经验"),
        ]
        
        for memory_id, memory_type, content in records:
            record = MemoryRecord(
                memory_id=memory_id,
                content=content,
                memory_type=memory_type,
                metadata={},
                tags=["test"],
                priority=MemoryPriority.MEDIUM,
                created_at=datetime.now()
            )
            cache.set(memory_id, record, version=1)
        
        print(f"✓ 添加了4个不同类型的缓存项")
        
        # 按记忆类型失效
        cleared_count = cache.invalidate_by_memory_type(MemoryType.CONVERSATION)
        print(f"✓ 清除了类型 'conversation' 的 {cleared_count} 个缓存项")
        assert cleared_count == 2, f"应清除2项，实际清除{cleared_count}项"
        
        # 验证CONVERSATION类型的缓存已清除
        assert cache.get("conv_001") is None, "conv_001 应已被清除"
        assert cache.get("conv_002") is None, "conv_002 应已被清除"
        
        # 验证其他类型的缓存仍存在
        assert cache.get("user_001") is not None, "user_001 应仍存在"
        assert cache.get("exp_001") is not None, "exp_001 应仍存在"
        print(f"✓ 其他类型的缓存项仍存在")
        
        print("\n✓ 按记忆类型失效缓存测试通过")
        return True
        
    except Exception as e:
        print(f"\n✗ 按记忆类型失效缓存测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_cache_stats() -> bool:
    """测试缓存统计信息"""
    print("\n" + "="*60)
    print("测试5: 缓存统计信息")
    print("="*60)
    
    try:
        cache = MemoryCache(max_size=100, ttl_hours=1)
        
        # 添加一些缓存项
        for i in range(10):
            record = MemoryRecord(
                memory_id=f"test_{i}",
                content=f"内容{i}",
                memory_type=MemoryType.CONVERSATION,
                metadata={},
                tags=["test"],
                priority=MemoryPriority.MEDIUM,
                created_at=datetime.now()
            )
            cache.set(f"test_{i}", record, version=1)
        
        # 获取统计信息
        stats = cache.get_stats()
        print(f"✓ 缓存统计信息:")
        print(f"  - 总项数: {stats['total_items']}")
        print(f"  - 最大容量: {stats['max_size']}")
        print(f"  - 利用率: {stats['utilization']:.2%}")
        print(f"  - TTL(秒): {stats['ttl_seconds']}")
        print(f"  - 版本数: {stats['total_versions']}")
        
        assert stats['total_items'] == 10, "总项数应为10"
        assert stats['max_size'] == 100, "最大容量应为100"
        assert stats['utilization'] == 0.1, "利用率应为0.1"
        assert stats['total_versions'] == 10, "版本数应为10"
        
        print("\n✓ 缓存统计信息测试通过")
        return True
        
    except Exception as e:
        print(f"\n✗ 缓存统计信息测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_memory_manager_cache_methods() -> bool:
    """测试MemoryManager的缓存方法"""
    print("\n" + "="*60)
    print("测试6: MemoryManager缓存方法")
    print("="*60)
    
    try:
        manager = MemoryManager()
        
        # 添加一些记忆
        memory_ids = []
        for i in range(5):
            memory_id = manager.add_memory(
                content=f"测试内容{i}",
                memory_type="conversation",
                tags=["test", f"tag_{i}"]
            )
            memory_ids.append(memory_id)
            print(f"✓ 添加记忆: {memory_id}")
        
        # 获取缓存统计
        stats = manager.get_cache_stats()
        print(f"✓ 缓存统计: {stats}")
        
        # 获取记忆版本
        for memory_id in memory_ids[:3]:
            version = manager.get_memory_version(memory_id)
            print(f"✓ 记忆 {memory_id} 版本: {version}")
            assert version >= 1, "版本号应>=1"
        
        # 按标签失效缓存
        cleared_count = manager.invalidate_cache_by_tag("test")
        print(f"✓ 按标签'test'清除了 {cleared_count} 个缓存项")
        
        # 按记忆类型失效缓存
        cleared_count = manager.invalidate_cache_by_memory_type("conversation")
        print(f"✓ 按类型'conversation'清除了 {cleared_count} 个缓存项")
        
        print("\n✓ MemoryManager缓存方法测试通过")
        return True
        
    except Exception as e:
        print(f"\n✗ MemoryManager缓存方法测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """运行所有测试"""
    print("="*60)
    print("缓存失效机制和版本控制测试")
    print("="*60)
    
    tests = [
        ("缓存版本控制", test_cache_version_control),
        ("按前缀失效缓存", test_cache_invalidation_by_prefix),
        ("按标签失效缓存", test_cache_invalidation_by_tag),
        ("按记忆类型失效缓存", test_cache_invalidation_by_memory_type),
        ("缓存统计信息", test_cache_stats),
        ("MemoryManager缓存方法", test_memory_manager_cache_methods),
    ]
    
    results = []
    for test_name, test_func in tests:
        result = test_func()
        results.append((test_name, result))
    
    # 打印测试结果汇总
    print("\n" + "="*60)
    print("测试结果汇总")
    print("="*60)
    
    for test_name, result in results:
        status = "✓ 通过" if result else "✗ 失败"
        print(f"{test_name}: {status}")
    
    passed_count = sum(1 for _, result in results if result)
    total_count = len(results)
    print(f"\n通过率: {passed_count}/{total_count} ({passed_count/total_count*100:.1f}%)")
    
    if passed_count == total_count:
        print("\n✓ 所有测试通过！缓存失效机制和版本控制实现成功！")
    else:
        print(f"\n✗ 有 {total_count - passed_count} 个测试失败")
    
    print("="*60)


if __name__ == "__main__":
    main()
