#!/usr/bin/env python3
"""
测试记忆检索优化功能
"""
import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from memory.memory_manager import MemoryManager, MemoryType, MemoryPriority, MemoryRetrievalStrategy

def test_retrieval_optimizer():
    """测试检索优化器"""
    print("="*60)
    print("测试记忆检索优化功能")
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
        print(f"  ✓ 添加记忆: {content[:30]}...")
    
    # 等待一下让记忆系统处理
    time.sleep(2)
    
    # 测试1: 基于标签的检索
    print("\n" + "="*60)
    print("测试1: 基于标签的检索")
    print("="*60)
    
    results = manager.search_memories(
        query="",
        tags=["编程"],
        limit=3
    )
    
    print(f"找到 {len(results)} 条记忆")
    for i, result in enumerate(results):
        print(f"  {i+1}. {result.content[:40]}... (标签: {result.tags})")
    
    if len(results) > 0:
        print("  ✓ 基于标签的检索成功")
    else:
        print("  ✗ 基于标签的检索失败")
        return False
    
    # 测试2: 基于优先级的检索
    print("\n" + "="*60)
    print("测试2: 基于优先级的检索")
    print("="*60)
    
    results = manager.search_memories(
        query="",
        priority=MemoryPriority.HIGH,
        limit=5
    )
    
    print(f"找到 {len(results)} 条高优先级记忆")
    for i, result in enumerate(results):
        print(f"  {i+1}. {result.content[:40]}... (优先级: {result.priority.value})")
    
    if len(results) > 0 and all(r.priority == MemoryPriority.HIGH for r in results):
        print("  ✓ 基于优先级的检索成功")
    else:
        print("  ✗ 基于优先级的检索失败")
        return False
    
    # 测试3: 语义检索
    print("\n" + "="*60)
    print("测试3: 语义检索")
    print("="*60)
    
    results = manager.search_memories(
        query="机器学习和深度学习",
        limit=3
    )
    
    print(f"找到 {len(results)} 条相关记忆")
    for i, result in enumerate(results):
        print(f"  {i+1}. {result.content[:40]}... (相关度: {result.relevance_score:.2f})")
    
    if len(results) > 0:
        print("  ✓ 语义检索成功")
    else:
        print("  ✗ 语义检索失败")
        return False
    
    # 测试4: 最近记忆检索
    print("\n" + "="*60)
    print("测试4: 最近记忆检索")
    print("="*60)
    
    results = manager.search_memories(
        query="用户",
        limit=3
    )
    
    print(f"找到 {len(results)} 条相关记忆")
    for i, result in enumerate(results):
        print(f"  {i+1}. {result.content[:40]}... (创建时间: {result.created_at.strftime('%H:%M:%S')})")
    
    if len(results) > 0:
        print("  ✓ 最近记忆检索成功")
    else:
        print("  ✗ 最近记忆检索失败")
        return False
    
    # 测试5: 混合检索
    print("\n" + "="*60)
    print("测试5: 混合检索（标签+查询）")
    print("="*60)
    
    results = manager.search_memories(
        query="学习",
        tags=["用户"],
        limit=3
    )
    
    print(f"找到 {len(results)} 条记忆")
    for i, result in enumerate(results):
        print(f"  {i+1}. {result.content[:40]}... (标签: {result.tags})")
    
    if len(results) > 0:
        print("  ✓ 混合检索成功")
    else:
        print("  ✗ 混合检索失败")
        return False
    
    # 测试6: 获取查询统计信息
    print("\n" + "="*60)
    print("测试6: 获取查询统计信息")
    print("="*60)
    
    stats = manager.retrieval_optimizer.get_query_stats()
    print(f"  总查询数: {stats['total_queries']}")
    print(f"  唯一查询数: {stats['unique_queries']}")
    print(f"  最常见查询:")
    for query, count in stats['most_common_queries']:
        print(f"    - {query}: {count} 次")
    
    print("  ✓ 查询统计信息获取成功")
    
    print("\n" + "="*60)
    print("🎉 所有检索优化测试通过！")
    print("="*60)
    
    return True

if __name__ == "__main__":
    try:
        success = test_retrieval_optimizer()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
