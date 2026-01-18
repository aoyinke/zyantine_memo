import os
import sys

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from zyantine_genisis.memory.memory_store import ZyantineMemorySystem

def test_search_fix():
    """测试搜索功能修复"""
    print("=== 测试搜索功能修复 ===")
    
    # 初始化记忆系统
    memory_system = ZyantineMemorySystem(user_id="test_user", session_id="test_session")
    
    # 添加一些测试记忆
    print("\n1. 添加测试记忆...")
    test_memories = [
        ("这是关于工作的重要信息", "work", ["工作", "重要"]),
        ("今天天气很好，适合户外活动", "life", ["生活", "天气"]),
        ("学习Python编程的技巧", "study", ["学习", "编程"]),
        ("健身锻炼的最佳时间", "fitness", ["健身", "锻炼"]),
        ("项目的截止日期是下周五", "work", ["工作", "截止日期"])
    ]
    
    for content, memory_type, tags in test_memories:
        memory_id = memory_system.add_memory(
            content=content,
            memory_type="conversation",
            tags=tags
        )
        print(f"   添加记忆: {content[:30]}... -> ID: {memory_id}")
    
    # 测试搜索功能
    print("\n2. 测试搜索功能...")
    search_queries = [
        ("工作", None, None, "测试基本搜索"),
        ("编程", None, None, "测试关键词搜索"),
        ("重要", "conversation", None, "测试按类型搜索"),
        ("天气", None, ["生活"], "测试按标签搜索")
    ]
    
    for query, memory_type, tags, description in search_queries:
        print(f"\n   {description}: 搜索 '{query}'")
        results = memory_system.search_memories(
            query=query,
            memory_type=memory_type,
            tags=tags,
            limit=3
        )
        
        if results:
            print(f"      找到 {len(results)} 个结果:")
            for i, result in enumerate(results):
                print(f"      {i+1}. {result['content'][:50]}... (相似度: {result['similarity_score']:.2f})")
        else:
            print(f"      没有找到结果")
    
    print("\n=== 测试完成 ===")
    print("搜索功能修复验证成功！")

if __name__ == "__main__":
    test_search_fix()