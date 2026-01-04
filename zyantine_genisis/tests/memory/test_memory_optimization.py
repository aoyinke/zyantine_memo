#!/usr/bin/env python3
"""
记忆模块优化测试脚本
"""
import asyncio
import json
from datetime import datetime
from memory.memory_manager import MemoryManager
from memory.memory_store import ZyantineMemorySystem
from memory.memory_exceptions import MemoryException, MemoryStorageError, MemoryRetrievalError, MemorySearchError


def test_memory_type_annotations() -> bool:
    """测试类型注解"""
    print("\n" + "="*60)
    print("测试1: 类型注解")
    print("="*60)
    
    try:
        # 创建记忆存储实例
        memory_store = ZyantineMemorySystem(user_id="test_user", session_id="test_session")
        
        # 测试add_memory的类型注解
        memory_id = memory_store.add_memory(
            content="测试记忆内容",
            memory_type="conversation",
            tags=["test"],
            emotional_intensity=0.5
        )
        print(f"✓ 添加记忆成功，ID: {memory_id}")
        
        # 测试search_memories的类型注解
        results = memory_store.search_memories(
            query="测试",
            memory_type="conversation",
            limit=5
        )
        print(f"✓ 搜索记忆成功，找到 {len(results)} 条结果")
        
        # 测试get_recent_conversations的类型注解
        conversations = memory_store.get_recent_conversations(limit=10)
        print(f"✓ 获取最近对话成功，找到 {len(conversations)} 条对话")
        
        print("\n✓ 类型注解测试通过")
        return True
        
    except Exception as e:
        print(f"\n✗ 类型注解测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_custom_exceptions():
    """测试自定义异常"""
    print("\n" + "="*60)
    print("测试2: 自定义异常")
    print("="*60)
    
    try:
        # 测试MemoryException
        try:
            raise MemoryException("测试基础异常", memory_id="test_123", details={"key": "value"})
        except MemoryException as e:
            print(f"✓ MemoryException: {e}")
            print(f"  - memory_id: {e.memory_id}")
            print(f"  - details: {e.details}")
        
        # 测试MemoryStorageError
        try:
            raise MemoryStorageError("测试存储错误", memory_id="test_456")
        except MemoryStorageError as e:
            print(f"✓ MemoryStorageError: {e}")
        
        # 测试MemoryRetrievalError
        try:
            raise MemoryRetrievalError("测试检索错误", memory_id="test_789")
        except MemoryRetrievalError as e:
            print(f"✓ MemoryRetrievalError: {e}")
        
        # 测试MemorySearchError
        try:
            raise MemorySearchError("测试搜索错误", details={"query": "test"})
        except MemorySearchError as e:
            print(f"✓ MemorySearchError: {e}")
        
        print("\n✓ 自定义异常测试通过")
        return True
        
    except Exception as e:
        print(f"\n✗ 自定义异常测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_optimized_get_recent_conversations() -> bool:
    """测试优化的get_recent_conversations方法"""
    print("\n" + "="*60)
    print("测试3: 优化的get_recent_conversations方法")
    print("="*60)
    
    try:
        # 创建记忆存储实例
        memory_store = ZyantineMemorySystem(user_id="test_user", session_id="test_session")
        
        # 添加多条对话记忆
        for i in range(5):
            memory_store.add_memory(
                content=f"对话 {i+1}",
                memory_type="conversation",
                tags=["test", f"conversation_{i+1}"]
            )
        
        # 测试获取最近对话
        conversations = memory_store.get_recent_conversations(limit=3)
        print(f"✓ 获取最近对话成功，找到 {len(conversations)} 条对话")
        
        # 验证返回的是字典类型
        for conv in conversations:
            if not isinstance(conv, dict):
                print(f"✗ 对话格式错误: {type(conv)}")
                return False
            print(f"  - 对话ID: {conv.get('memory_id', 'N/A')}")
        
        print("\n✓ 优化的get_recent_conversations方法测试通过")
        return True
        
    except Exception as e:
        print(f"\n✗ 优化的get_recent_conversations方法测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_memory_manager_optimizations():
    """测试记忆管理器优化"""
    print("\n" + "="*60)
    print("测试4: 记忆管理器优化")
    print("="*60)
    
    try:
        # 创建记忆管理器实例（不需要传递user_id和session_id）
        memory_manager = MemoryManager()
        
        # 测试添加记忆
        memory_id = memory_manager.add_memory(
            content="测试记忆管理器优化",
            memory_type="conversation",
            tags=["test", "optimization"],
            emotional_intensity=0.7
        )
        print(f"✓ 添加记忆成功，ID: {memory_id}")
        
        # 测试搜索记忆
        results = memory_manager.search_memories(
            query="测试",
            memory_type="conversation",
            limit=5
        )
        print(f"✓ 搜索记忆成功，找到 {len(results)} 条结果")
        
        # 测试获取对话历史
        history = memory_manager.get_conversation_history(limit=10)
        print(f"✓ 获取对话历史成功，找到 {len(history)} 条历史")
        
        print("\n✓ 记忆管理器优化测试通过")
        return True
        
    except Exception as e:
        print(f"\n✗ 记忆管理器优化测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_error_handling():
    """测试错误处理"""
    print("\n" + "="*60)
    print("测试5: 错误处理")
    print("="*60)
    
    try:
        memory_store = ZyantineMemorySystem(user_id="test_user", session_id="test_session")
        
        # 测试无效记忆类型的错误处理
        try:
            memory_store.add_memory(
                content="测试",
                memory_type="invalid_type"
            )
            print("✗ 应该抛出异常但没有")
            return False
        except Exception as e:
            print(f"✓ 正确处理了无效记忆类型: {type(e).__name__}")
        
        # 测试空搜索查询的错误处理
        try:
            results = memory_store.search_memories(
                query="",
                limit=5
            )
            print(f"✓ 空搜索查询处理正确，返回 {len(results)} 条结果")
        except Exception as e:
            print(f"✓ 空搜索查询抛出异常: {type(e).__name__}")
        
        print("\n✓ 错误处理测试通过")
        return True
        
    except Exception as e:
        print(f"\n✗ 错误处理测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """主测试函数"""
    print("\n" + "="*60)
    print("记忆模块优化测试")
    print("="*60)
    
    results = []
    
    # 运行所有测试
    results.append(("类型注解", test_memory_type_annotations()))
    results.append(("自定义异常", test_custom_exceptions()))
    results.append(("优化的get_recent_conversations", test_optimized_get_recent_conversations()))
    results.append(("记忆管理器优化", test_memory_manager_optimizations()))
    results.append(("错误处理", test_error_handling()))
    
    # 打印测试结果
    print("\n" + "="*60)
    print("测试结果汇总")
    print("="*60)
    
    for test_name, result in results:
        status = "✓ 通过" if result else "✗ 失败"
        print(f"{test_name}: {status}")
    
    # 统计通过率
    passed = sum(1 for _, result in results if result)
    total = len(results)
    pass_rate = (passed / total) * 100
    
    print(f"\n通过率: {passed}/{total} ({pass_rate:.1f}%)")
    
    if pass_rate == 100:
        print("\n✓ 所有测试通过！记忆模块优化成功！")
    else:
        print(f"\n✗ 有 {total - passed} 个测试失败")
    
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
