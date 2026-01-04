"""
测试记忆压缩和归档功能
"""
import time
from datetime import datetime, timedelta
from memory.memory_manager import (
    MemoryManager,
    MemoryType,
    MemoryPriority,
    MemoryRecord
)


def test_compression_and_archiving():
    """测试压缩和归档功能"""
    print("="*60)
    print("测试记忆压缩和归档功能")
    print("="*60)
    
    manager = MemoryManager()
    
    # 添加一些测试记忆
    print("\n添加测试记忆...")
    test_memories = [
        ("用户喜欢编程和人工智能", MemoryType.USER_PROFILE, ["用户", "编程"], MemoryPriority.HIGH),
        ("昨天讨论了机器学习算法", MemoryType.CONVERSATION, ["机器学习", "讨论"], MemoryPriority.MEDIUM),
        ("系统启动成功", MemoryType.SYSTEM_EVENT, ["系统", "启动"], MemoryPriority.LOW),
        ("Python是一种流行的编程语言", MemoryType.KNOWLEDGE, ["Python", "编程"], MemoryPriority.HIGH),
        ("用户最近在研究深度学习", MemoryType.USER_PROFILE, ["用户", "深度学习"], MemoryPriority.HIGH),
        ("今天天气很好", MemoryType.CONVERSATION, ["天气"], MemoryPriority.LOW),
        ("系统配置已完成", MemoryType.SYSTEM_EVENT, ["系统", "配置"], MemoryPriority.MEDIUM),
        ("JavaScript是Web开发的核心", MemoryType.KNOWLEDGE, ["JavaScript", "Web"], MemoryPriority.MEDIUM),
    ]
    
    memory_ids = []
    for content, mtype, tags, priority in test_memories:
        memory_id = manager.add_memory(
            content=content,
            memory_type=mtype,
            tags=tags,
            priority=priority
        )
        memory_ids.append(memory_id)
        print(f"  ✓ 添加记忆: {content[:20]}... (优先级: {priority.value})")
    
    time.sleep(1)
    
    # 测试1: 获取压缩统计信息
    print("\n" + "="*60)
    print("测试1: 获取压缩统计信息")
    print("="*60)
    
    stats = manager.get_compression_stats()
    print(f"  总压缩数: {stats['total_compressed']}")
    print(f"  总归档数: {stats['total_archived']}")
    print(f"  总节省空间: {stats['total_size_saved']} bytes")
    print(f"  压缩率: {stats['compression_ratio']:.2%}")
    print(f"  缓存中压缩数: {stats['compressed_in_cache']}")
    print(f"  缓存中归档数: {stats['archived_in_cache']}")
    print("  ✓ 压缩统计信息获取成功")
    
    # 测试2: 获取压缩和归档阈值配置
    print("\n" + "="*60)
    print("测试2: 获取压缩和归档阈值配置")
    print("="*60)
    
    thresholds = manager.get_compression_thresholds()
    print("  压缩阈值:")
    print(f"    - 内容大小: {thresholds['compression']['size_bytes']} bytes")
    print(f"    - 年龄: {thresholds['compression']['age_hours']} 小时")
    print(f"    - 访问次数: {thresholds['compression']['access_count']}")
    print(f"    - 优先级: {[p.value for p in thresholds['compression']['priority']]}")
    print("  归档阈值:")
    print(f"    - 年龄: {thresholds['archive']['age_hours']} 小时")
    print(f"    - 优先级: {[p.value for p in thresholds['archive']['priority']]}")
    print(f"    - 未访问时间: {thresholds['archive']['no_access_hours']} 小时")
    print("  ✓ 阈值配置获取成功")
    
    # 测试3: 更新压缩阈值
    print("\n" + "="*60)
    print("测试3: 更新压缩阈值")
    print("="*60)
    
    manager.update_compression_thresholds(size_bytes=5000, age_hours=360)
    
    updated_thresholds = manager.get_compression_thresholds()
    if updated_thresholds['compression']['size_bytes'] == 5000 and updated_thresholds['compression']['age_hours'] == 360:
        print("  ✓ 压缩阈值更新成功")
        print(f"    内容大小阈值已更新为: 5000 bytes")
        print(f"    年龄阈值已更新为: 360 小时")
    else:
        print("  ✗ 压缩阈值更新失败")
        return False
    
    # 测试4: 更新归档阈值
    print("\n" + "="*60)
    print("测试4: 更新归档阈值")
    print("="*60)
    
    manager.update_archive_thresholds(age_hours=1080, no_access_hours=720)
    
    updated_thresholds = manager.get_compression_thresholds()
    if updated_thresholds['archive']['age_hours'] == 1080 and updated_thresholds['archive']['no_access_hours'] == 720:
        print("  ✓ 归档阈值更新成功")
        print(f"    年龄阈值已更新为: 1080 小时")
        print(f"    未访问时间阈值已更新为: 720 小时")
    else:
        print("  ✗ 归档阈值更新失败")
        return False
    
    # 测试5: 压缩单个记忆
    print("\n" + "="*60)
    print("测试5: 压缩单个记忆")
    print("="*60)
    
    # 创建一个较大的记忆内容
    large_content = "这是一个较大的记忆内容。" * 500
    large_memory_id = manager.add_memory(
        content=large_content,
        memory_type=MemoryType.KNOWLEDGE,
        tags=["测试", "大内容"],
        priority=MemoryPriority.LOW
    )
    
    # 压缩记忆
    compressed_record = manager.compress_memory(large_memory_id)
    if compressed_record and compressed_record.metadata.get("compressed", False):
        print("  ✓ 记忆压缩成功")
        print(f"    原始大小: {compressed_record.metadata.get('original_size', 0)} bytes")
        print(f"    压缩后大小: {compressed_record.size_bytes} bytes")
    else:
        print("  ✗ 记忆压缩失败")
        return False
    
    # 测试6: 归档单个记忆
    print("\n" + "="*60)
    print("测试6: 归档单个记忆")
    print("="*60)
    
    # 创建一个较旧的记忆
    old_content = "这是一个较旧的记忆内容。"
    old_memory_id = manager.add_memory(
        content=old_content,
        memory_type=MemoryType.CONVERSATION,
        tags=["测试", "旧内容"],
        priority=MemoryPriority.LOW
    )
    
    # 手动设置记忆为较旧
    record = manager.cache.get(old_memory_id)
    if record:
        record.created_at = datetime.now() - timedelta(hours=2500)
        manager.cache.set(old_memory_id, record, record.version)
    
    # 归档记忆
    archived_record = manager.archive_memory(old_memory_id)
    if archived_record and archived_record.metadata.get("archived", False):
        print("  ✓ 记忆归档成功")
        print(f"    优先级: {archived_record.priority.value}")
        print(f"    已压缩: {archived_record.metadata.get('compressed', False)}")
    else:
        print("  ✗ 记忆归档失败")
        return False
    
    # 测试7: 解压缩记忆
    print("\n" + "="*60)
    print("测试7: 解压缩记忆")
    print("="*60)
    
    decompressed_record = manager.decompress_memory(large_memory_id)
    if decompressed_record and not decompressed_record.metadata.get("compressed", False):
        print("  ✓ 记忆解压成功")
        print(f"    大小: {decompressed_record.size_bytes} bytes")
    else:
        print("  ✗ 记忆解压失败")
        return False
    
    # 测试8: 批量压缩记忆
    print("\n" + "="*60)
    print("测试8: 批量压缩记忆")
    print("="*60)
    
    # 添加更多测试记忆
    for i in range(5):
        content = f"批量测试记忆 {i}。" * 200
        memory_id = manager.add_memory(
            content=content,
            memory_type=MemoryType.KNOWLEDGE,
            tags=["批量测试"],
            priority=MemoryPriority.LOW
        )
        memory_ids.append(memory_id)
    
    # 批量压缩
    compress_result = manager.batch_compress_memories()
    print(f"  批量压缩结果:")
    print(f"    压缩数量: {compress_result['compressed']}")
    print(f"    跳过数量: {compress_result['skipped']}")
    print(f"    总数量: {compress_result['total']}")
    print("  ✓ 批量压缩成功")
    
    # 测试9: 批量归档记忆
    print("\n" + "="*60)
    print("测试9: 批量归档记忆")
    print("="*60)
    
    # 添加更多较旧的测试记忆
    for i in range(5):
        content = f"批量归档测试记忆 {i}。"
        memory_id = manager.add_memory(
            content=content,
            memory_type=MemoryType.CONVERSATION,
            tags=["批量归档"],
            priority=MemoryPriority.LOW
        )
        
        # 设置为较旧
        record = manager.cache.get(memory_id)
        if record:
            record.created_at = datetime.now() - timedelta(hours=2500)
            manager.cache.set(memory_id, record, record.version)
        
        memory_ids.append(memory_id)
    
    # 批量归档
    archive_result = manager.batch_archive_memories()
    print(f"  批量归档结果:")
    print(f"    归档数量: {archive_result['archived']}")
    print(f"    跳过数量: {archive_result['skipped']}")
    print(f"    总数量: {archive_result['total']}")
    print("  ✓ 批量归档成功")
    
    # 测试10: 自动压缩和归档
    print("\n" + "="*60)
    print("测试10: 自动压缩和归档")
    print("="*60)
    
    auto_result = manager.auto_compress_and_archive()
    print(f"  自动压缩和归档结果:")
    print(f"    压缩数量: {auto_result['compressed']}")
    print(f"    归档数量: {auto_result['archived']}")
    print("  ✓ 自动压缩和归档成功")
    
    # 测试11: 搜索压缩的记忆
    print("\n" + "="*60)
    print("测试11: 搜索压缩的记忆")
    print("="*60)
    
    # 搜索包含"批量测试"的记忆
    search_results = manager.search_memories(
        query="批量测试",
        limit=5
    )
    
    print(f"  找到 {len(search_results)} 条记忆")
    for i, record in enumerate(search_results[:3], 1):
        content_preview = str(record.content)[:50] if record.content else "None"
        print(f"    {i}. {content_preview}... (压缩: {record.metadata.get('compressed', False)})")
    print("  ✓ 搜索压缩记忆成功")
    
    # 测试12: 获取最终压缩统计
    print("\n" + "="*60)
    print("测试12: 获取最终压缩统计")
    print("="*60)
    
    final_stats = manager.get_compression_stats()
    print(f"  总压缩数: {final_stats['total_compressed']}")
    print(f"  总归档数: {final_stats['total_archived']}")
    print(f"  总节省空间: {final_stats['total_size_saved']} bytes")
    print(f"  压缩率: {final_stats['compression_ratio']:.2%}")
    print(f"  缓存中压缩数: {final_stats['compressed_in_cache']}")
    print(f"  缓存中归档数: {final_stats['archived_in_cache']}")
    print("  ✓ 最终压缩统计获取成功")
    
    print("\n" + "="*60)
    print("🎉 所有压缩和归档测试通过！")
    print("="*60)
    
    return True


if __name__ == "__main__":
    try:
        success = test_compression_and_archiving()
        if success:
            print("\n✅ 所有测试通过")
        else:
            print("\n❌ 部分测试失败")
    except Exception as e:
        print(f"\n❌ 测试过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
