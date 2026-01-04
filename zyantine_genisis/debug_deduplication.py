#!/usr/bin/env python3
"""
调试记忆去重功能
"""
import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from memory.memory_manager import MemoryManager, MemoryType

def debug_memories():
    """调试记忆功能"""
    print("="*60)
    print("调试记忆去重功能")
    print("="*60)
    
    manager = MemoryManager()
    
    print(f"\n当前会话ID: {manager.session_id}")
    print(f"当前用户ID: {manager.user_id}")
    
    # 使用时间戳生成唯一内容
    timestamp = int(time.time())
    unique_content = f"这是一条新的测试记忆_{timestamp}"
    
    # 添加新记忆
    print("\n" + "="*60)
    print("添加新记忆...")
    print("="*60)
    
    new_memory_id = manager.add_memory(
        content=unique_content,
        memory_type=MemoryType.CONVERSATION,
        tags=["测试", "新记忆"]
    )
    print(f"添加成功: {new_memory_id}")
    
    # 直接从记忆系统获取记忆
    print("\n" + "="*60)
    print("直接从记忆系统获取记忆...")
    print("="*60)
    
    try:
        memory_data = manager.memory_system.get_memory(new_memory_id)
        print(f"找到记忆:")
        print(f"  ID: {memory_data.get('memory_id')}")
        print(f"  内容: {memory_data.get('content')}")
        print(f"  类型: {memory_data.get('memory_type')}")
        print(f"  Metadata:")
        for key, value in memory_data.get('metadata', {}).items():
            print(f"    {key}: {value}")
    except Exception as e:
        print(f"获取记忆失败: {e}")
        import traceback
        traceback.print_exc()
    
    # 使用搜索API获取记忆
    print("\n" + "="*60)
    print("使用搜索API获取记忆...")
    print("="*60)
    
    try:
        search_results = manager.memory_system.search_memories(
            query=unique_content,
            memory_type="conversation",
            limit=10,
            similarity_threshold=0.0
        )
        print(f"找到 {len(search_results)} 条记忆")
        
        for i, result in enumerate(search_results):
            print(f"\n记忆 {i+1}:")
            print(f"  ID: {result.get('memory_id')}")
            print(f"  内容: {result.get('content')}")
            print(f"  相似度: {result.get('similarity')}")
            print(f"  Metadata:")
            for key, value in result.get('metadata', {}).items():
                print(f"    {key}: {value}")
    except Exception as e:
        print(f"搜索记忆失败: {e}")
        import traceback
        traceback.print_exc()
    
    # 使用 get_all_memories 获取记忆
    print("\n" + "="*60)
    print("使用 get_all_memories 获取记忆...")
    print("="*60)
    
    all_memories = manager.get_all_memories(memory_type=MemoryType.CONVERSATION)
    print(f"找到 {len(all_memories)} 条对话记忆")
    
    # 查找我们添加的记忆
    found = False
    for memory in all_memories:
        if unique_content in str(memory.content):
            print(f"\n找到新记忆:")
            print(f"  ID: {memory.memory_id}")
            print(f"  内容: {memory.content}")
            print(f"  Metadata session_id: {memory.metadata.get('session_id')}")
            found = True
            break
    
    if not found:
        print("未找到新添加的记忆")

if __name__ == "__main__":
    debug_memories()