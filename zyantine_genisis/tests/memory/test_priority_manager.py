"""
测试记忆优先级管理功能
"""
import time
from datetime import datetime, timedelta
from memory.memory_manager import (
    MemoryManager,
    MemoryType,
    MemoryPriority,
    MemoryRecord
)


def test_priority_manager():
    """测试优先级管理器"""
    print("="*60)
    print("测试记忆优先级管理功能")
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
    for content, mem_type, tags, priority in test_memories:
        memory_id = manager.add_memory(
            content=content,
            memory_type=mem_type,
            tags=tags,
            priority=priority
        )
        memory_ids.append(memory_id)
        print(f"  ✓ 添加记忆: {content[:30]}... (优先级: {priority.value})")
    
    # 等待一下让记忆系统处理
    time.sleep(2)
    
    # 测试1: 获取优先级统计信息
    print("\n" + "="*60)
    print("测试1: 获取优先级统计信息")
    print("="*60)
    
    stats = manager.get_priority_stats()
    print(f"  总记忆数: {stats['total_count']}")
    print(f"  按优先级分布:")
    for priority, count in stats['by_priority'].items():
        print(f"    - {priority}: {count} 条")
    print(f"  平均优先级分数:")
    for priority, score in stats['average_scores'].items():
        print(f"    - {priority}: {score:.3f}")
    print(f"  可提升候选: {stats['promote_candidates']} 条")
    print(f"  可降低候选: {stats['demote_candidates']} 条")
    
    if stats['total_count'] > 0:
        print("  ✓ 优先级统计信息获取成功")
    else:
        print("  ✗ 优先级统计信息获取失败")
        return False
    
    # 测试2: 计算单个记忆的优先级分数
    print("\n" + "="*60)
    print("测试2: 计算单个记忆的优先级分数")
    print("="*60)
    
    if memory_ids:
        test_id = memory_ids[0]
        score = manager.calculate_priority_score(test_id)
        print(f"  记忆ID: {test_id[:8]}...")
        print(f"  优先级分数: {score:.3f}")
        
        if score is not None and 0.0 <= score <= 1.0:
            print("  ✓ 优先级分数计算成功")
        else:
            print("  ✗ 优先级分数计算失败")
            return False
    
    # 测试3: 按优先级排序记忆
    print("\n" + "="*60)
    print("测试3: 按优先级分数排序记忆")
    print("="*60)
    
    sorted_memories = manager.sort_memories_by_priority(include_score=True)
    print(f"  排序后的前5条记忆:")
    for i, (record, score) in enumerate(sorted_memories[:5]):
        print(f"    {i+1}. {record.content[:30]}... (分数: {score:.3f}, 优先级: {record.priority.value})")
    
    if len(sorted_memories) > 0:
        print("  ✓ 记忆排序成功")
    else:
        print("  ✗ 记忆排序失败")
        return False
    
    # 测试4: 获取高优先级记忆
    print("\n" + "="*60)
    print("测试4: 获取高优先级记忆")
    print("="*60)
    
    top_memories = manager.get_top_priority_memories(limit=5, min_score=0.5)
    print(f"  找到 {len(top_memories)} 条高优先级记忆:")
    for i, record in enumerate(top_memories):
        score = manager.calculate_priority_score(record.memory_id)
        print(f"    {i+1}. {record.content[:30]}... (分数: {score:.3f}, 优先级: {record.priority.value})")
    
    if len(top_memories) > 0:
        print("  ✓ 高优先级记忆获取成功")
    else:
        print("  ✗ 高优先级记忆获取失败")
        return False
    
    # 测试5: 获取优先级阈值配置
    print("\n" + "="*60)
    print("测试5: 获取优先级阈值配置")
    print("="*60)
    
    thresholds = manager.get_priority_thresholds()
    print(f"  自动提升阈值:")
    for priority, config in thresholds['auto_promote'].items():
        print(f"    - {priority}: 访问次数={config['access_count']}, 小时数={config['hours']}")
    print(f"  自动降低阈值:")
    for priority, config in thresholds['auto_demote'].items():
        print(f"    - {priority}: 未访问小时={config['no_access_hours']}, 年龄小时={config['age_hours']}")
    
    print("  ✓ 优先级阈值配置获取成功")
    
    # 测试6: 更新优先级阈值配置
    print("\n" + "="*60)
    print("测试6: 更新优先级阈值配置")
    print("="*60)
    
    new_promote_thresholds = {
        MemoryPriority.LOW: {"access_count": 5, "hours": 12}
    }
    
    manager.update_priority_thresholds(promote=new_promote_thresholds)
    
    updated_thresholds = manager.get_priority_thresholds()
    # 查找LOW优先级的阈值
    low_threshold_found = False
    for priority_key, thresholds in updated_thresholds['auto_promote'].items():
        if str(priority_key) == 'low' or priority_key == MemoryPriority.LOW:
            updated_low_threshold = thresholds
            low_threshold_found = True
            break
    
    if low_threshold_found and updated_low_threshold.get('access_count') == 5 and updated_low_threshold.get('hours') == 12:
        print(f"  ✓ 优先级阈值配置更新成功")
        print(f"    LOW优先级提升阈值已更新为: 访问次数=5, 小时数=12")
    else:
        print("  ✗ 优先级阈值配置更新失败")
        return False
    
    # 测试7: 模拟访问记忆以触发优先级调整
    print("\n" + "="*60)
    print("测试7: 模拟访问记忆以触发优先级调整")
    print("="*60)
    
    # 获取一个低优先级记忆
    low_priority_memories = [
        record for record in manager.cache.cache.values()
        if record.priority == MemoryPriority.LOW
    ]
    
    if low_priority_memories:
        test_record = low_priority_memories[0]
        print(f"  测试记忆: {test_record.content[:30]}...")
        print(f"  初始优先级: {test_record.priority.value}")
        
        # 模拟多次访问
        for i in range(10):
            manager.cache.get(test_record.memory_id)
        
        # 检查是否应该提升
        should_promote = manager.priority_manager.should_promote(test_record)
        print(f"  是否应该提升: {should_promote}")
        
        if should_promote:
            new_priority = manager.adjust_memory_priority(test_record.memory_id)
            print(f"  新优先级: {new_priority.value if new_priority else '无变化'}")
            
            if new_priority and new_priority != MemoryPriority.LOW:
                print("  ✓ 记忆优先级提升成功")
            else:
                print("  ✗ 记忆优先级提升失败")
                return False
        else:
            print("  ℹ 当前访问次数不足以触发提升")
    
    # 测试8: 批量调整优先级
    print("\n" + "="*60)
    print("测试8: 批量调整优先级")
    print("="*60)
    
    # 先获取调整前的统计
    stats_before = manager.get_priority_stats()
    print(f"  调整前统计:")
    print(f"    可提升候选: {stats_before['promote_candidates']} 条")
    print(f"    可降低候选: {stats_before['demote_candidates']} 条")
    
    # 执行批量调整
    adjust_stats = manager.batch_adjust_priorities()
    
    print(f"  调整结果:")
    print(f"    提升数量: {adjust_stats['promoted']} 条")
    print(f"    降低数量: {adjust_stats['demoted']} 条")
    
    if 'promoted' in adjust_stats and 'demoted' in adjust_stats:
        print("  ✓ 批量优先级调整成功")
    else:
        print("  ✗ 批量优先级调整失败")
        return False
    
    print("\n" + "="*60)
    print("🎉 所有优先级管理测试通过！")
    print("="*60)
    
    return True


if __name__ == "__main__":
    try:
        success = test_priority_manager()
        if success:
            print("\n✅ 所有测试通过")
        else:
            print("\n❌ 部分测试失败")
    except Exception as e:
        print(f"\n❌ 测试过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
