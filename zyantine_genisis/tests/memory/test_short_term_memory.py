#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
短期记忆单元测试和性能测试
"""

import unittest
import time
import random
from datetime import datetime
from zyantine_genisis.memory.memory_store import ZyantineMemorySystem


class TestShortTermMemory(unittest.TestCase):
    """短期记忆单元测试"""
    
    def setUp(self):
        """设置测试环境"""
        self.memory_system = ZyantineMemorySystem(user_id="test_user", session_id="test_session")
        self.conversation_id = "test_conversation_1"
    
    def tearDown(self):
        """清理测试环境"""
        # 重置短期记忆存储
        self.memory_system.short_term_store.clear()
        self.memory_system.conversation_index.clear()
    
    def test_add_short_term_memory(self):
        """测试添加短期记忆"""
        # 添加短期记忆
        content = "这是一个测试对话内容"
        memory_id = self.memory_system.add_short_term_memory(
            content=content,
            conversation_id=self.conversation_id,
            metadata={"test_key": "test_value"}
        )
        
        # 验证记忆是否添加成功
        self.assertIsNotNone(memory_id)
        self.assertIn(memory_id, self.memory_system.short_term_store)
        
        # 验证对话索引是否更新
        self.assertIn(self.conversation_id, self.memory_system.conversation_index)
        self.assertIn(memory_id, self.memory_system.conversation_index[self.conversation_id])
    
    def test_get_short_term_memory(self):
        """测试获取短期记忆"""
        # 添加短期记忆
        content = "这是一个测试对话内容"
        memory_id = self.memory_system.add_short_term_memory(
            content=content,
            conversation_id=self.conversation_id
        )
        
        # 获取短期记忆
        stm = self.memory_system.get_short_term_memory(memory_id)
        
        # 验证记忆是否获取成功
        self.assertIsNotNone(stm)
        self.assertEqual(stm.memory_id, memory_id)
        self.assertEqual(stm.content, content)
        self.assertEqual(stm.conversation_id, self.conversation_id)
    
    def test_get_nonexistent_memory(self):
        """测试获取不存在的记忆"""
        # 获取不存在的记忆
        stm = self.memory_system.get_short_term_memory("nonexistent_memory_id")
        
        # 验证结果
        self.assertIsNone(stm)
    
    def test_search_short_term_memories(self):
        """测试搜索短期记忆"""
        # 添加多个短期记忆
        test_contents = [
            "测试对话内容1",
            "测试对话内容2",
            "测试对话内容3"
        ]
        
        for content in test_contents:
            self.memory_system.add_short_term_memory(
                content=content,
                conversation_id=self.conversation_id
            )
        
        # 搜索短期记忆
        results = self.memory_system.search_short_term_memories(self.conversation_id, limit=2)
        
        # 验证搜索结果
        self.assertEqual(len(results), 2)
        # 验证结果按创建时间倒序排序
        self.assertTrue(results[0].created_at > results[1].created_at)
    
    def test_expired_memory_cleanup(self):
        """测试过期记忆自动清理"""
        # 添加一个短期记忆，设置1秒后过期
        content = "这是一个即将过期的测试对话内容"
        memory_id = self.memory_system.add_short_term_memory(
            content=content,
            conversation_id=self.conversation_id,
            ttl=1  # 1秒后过期
        )
        
        # 验证记忆添加成功
        self.assertIn(memory_id, self.memory_system.short_term_store)
        
        # 等待2秒，让记忆过期
        time.sleep(2)
        
        # 触发清理
        self.memory_system._cleanup_short_term_memory(force=True)
        
        # 验证记忆已被清理
        self.assertNotIn(memory_id, self.memory_system.short_term_store)
        # 验证对话索引已更新
        self.assertNotIn(memory_id, self.memory_system.conversation_index.get(self.conversation_id, []))
    
    def test_max_size_limit(self):
        """测试短期记忆最大容量限制"""
        # 设置较小的最大容量
        original_max_size = self.memory_system.short_term_max_size
        self.memory_system.short_term_max_size = 3
        
        try:
            # 添加4个短期记忆，应该触发清理
            for i in range(4):
                self.memory_system.add_short_term_memory(
                    content=f"测试对话内容{i}",
                    conversation_id=self.conversation_id
                )
            
            # 验证短期记忆数量不超过最大容量
            self.assertEqual(len(self.memory_system.short_term_store), 3)
        finally:
            # 恢复原始最大容量
            self.memory_system.short_term_max_size = original_max_size
    
    def test_evict_oldest_memory(self):
        """测试删除最旧的短期记忆"""
        # 设置最大容量为2
        original_max_size = self.memory_system.short_term_max_size
        self.memory_system.short_term_max_size = 2
        
        try:
            # 添加3个短期记忆
            memory_ids = []
            for i in range(3):
                memory_id = self.memory_system.add_short_term_memory(
                    content=f"测试对话内容{i}",
                    conversation_id=self.conversation_id
                )
                memory_ids.append(memory_id)
                # 等待一小段时间，确保创建时间不同
                time.sleep(0.1)
            
            # 验证第一个添加的记忆已被删除
            self.assertNotIn(memory_ids[0], self.memory_system.short_term_store)
            # 验证后两个添加的记忆仍然存在
            self.assertIn(memory_ids[1], self.memory_system.short_term_store)
            self.assertIn(memory_ids[2], self.memory_system.short_term_store)
        finally:
            # 恢复原始最大容量
            self.memory_system.short_term_max_size = original_max_size
    
    def test_statistics_include_short_term_memory(self):
        """测试统计信息包含短期记忆统计"""
        # 添加短期记忆
        self.memory_system.add_short_term_memory(
            content="测试对话内容",
            conversation_id=self.conversation_id
        )
        
        # 获取统计信息
        stats = self.memory_system.get_statistics()
        
        # 验证统计信息包含短期记忆统计
        self.assertIn("short_term_memory_stats", stats)
        self.assertEqual(stats["short_term_memory_stats"]["current_size"], 1)
        self.assertEqual(stats["short_term_memory_stats"]["max_size"], self.memory_system.short_term_max_size)


class TestShortTermMemoryPerformance(unittest.TestCase):
    """短期记忆性能测试"""
    
    def setUp(self):
        """设置测试环境"""
        self.memory_system = ZyantineMemorySystem(user_id="test_user", session_id="test_session")
        self.conversation_id = "test_conversation_1"
    
    def tearDown(self):
        """清理测试环境"""
        # 重置短期记忆存储
        self.memory_system.short_term_store.clear()
        self.memory_system.conversation_index.clear()
    
    def test_add_memory_performance(self):
        """测试添加短期记忆的性能"""
        # 测试添加1000个短期记忆的耗时
        total_time = 0
        count = 1000
        
        for i in range(count):
            start_time = time.time()
            self.memory_system.add_short_term_memory(
                content=f"测试对话内容{i}",
                conversation_id=self.conversation_id,
                metadata={"index": i}
            )
            end_time = time.time()
            total_time += (end_time - start_time) * 1000  # 转换为毫秒
        
        average_time = total_time / count
        print(f"\n添加短期记忆性能测试：")
        print(f"总耗时：{total_time:.2f}ms")
        print(f"平均耗时：{average_time:.2f}ms")
        
        # 验证平均耗时≤100ms
        self.assertLess(average_time, 100, f"添加短期记忆平均耗时{average_time:.2f}ms超过100ms")
    
    def test_get_memory_performance(self):
        """测试获取短期记忆的性能"""
        # 先添加1000个短期记忆
        memory_ids = []
        for i in range(1000):
            memory_id = self.memory_system.add_short_term_memory(
                content=f"测试对话内容{i}",
                conversation_id=self.conversation_id
            )
            memory_ids.append(memory_id)
        
        # 测试获取1000个短期记忆的耗时
        total_time = 0
        count = 1000
        
        for memory_id in memory_ids:
            start_time = time.time()
            self.memory_system.get_short_term_memory(memory_id)
            end_time = time.time()
            total_time += (end_time - start_time) * 1000  # 转换为毫秒
        
        average_time = total_time / count
        print(f"\n获取短期记忆性能测试：")
        print(f"总耗时：{total_time:.2f}ms")
        print(f"平均耗时：{average_time:.2f}ms")
        
        # 验证平均耗时≤100ms
        self.assertLess(average_time, 100, f"获取短期记忆平均耗时{average_time:.2f}ms超过100ms")
    
    def test_search_memory_performance(self):
        """测试搜索短期记忆的性能"""
        # 先添加1000个短期记忆
        for i in range(1000):
            self.memory_system.add_short_term_memory(
                content=f"测试对话内容{i}",
                conversation_id=self.conversation_id
            )
        
        # 测试搜索100次的耗时
        total_time = 0
        count = 100
        
        for i in range(count):
            start_time = time.time()
            self.memory_system.search_short_term_memories(self.conversation_id, limit=10)
            end_time = time.time()
            total_time += (end_time - start_time) * 1000  # 转换为毫秒
        
        average_time = total_time / count
        print(f"\n搜索短期记忆性能测试：")
        print(f"总耗时：{total_time:.2f}ms")
        print(f"平均耗时：{average_time:.2f}ms")
        
        # 验证平均耗时≤100ms
        self.assertLess(average_time, 100, f"搜索短期记忆平均耗时{average_time:.2f}ms超过100ms")
    
    def test_cleanup_performance(self):
        """测试清理过期记忆的性能"""
        # 添加1000个短期记忆，设置1秒后过期
        for i in range(1000):
            self.memory_system.add_short_term_memory(
                content=f"测试对话内容{i}",
                conversation_id=self.conversation_id,
                ttl=1
            )
        
        # 等待2秒，让所有记忆过期
        time.sleep(2)
        
        # 测试清理过期记忆的耗时
        start_time = time.time()
        self.memory_system._cleanup_short_term_memory(force=True)
        end_time = time.time()
        
        cleanup_time = (end_time - start_time) * 1000  # 转换为毫秒
        print(f"\n清理过期记忆性能测试：")
        print(f"清理耗时：{cleanup_time:.2f}ms")
        print(f"清理后短期记忆数量：{len(self.memory_system.short_term_store)}")
        
        # 验证清理是否成功
        self.assertEqual(len(self.memory_system.short_term_store), 0)
        # 验证清理耗时≤100ms
        self.assertLess(cleanup_time, 100, f"清理过期记忆耗时{cleanup_time:.2f}ms超过100ms")
    
    def test_stress_test(self):
        """压力测试：混合操作"""
        print(f"\n开始短期记忆压力测试...")
        
        # 测试参数
        total_operations = 10000
        add_count = 0
        get_count = 0
        search_count = 0
        
        # 存储添加的记忆ID
        added_memory_ids = []
        
        # 混合执行添加、获取和搜索操作
        start_time = time.time()
        
        for i in range(total_operations):
            # 随机选择操作类型
            operation = random.choice(["add", "get", "search"])
            
            if operation == "add":
                # 添加短期记忆
                memory_id = self.memory_system.add_short_term_memory(
                    content=f"压力测试对话内容{i}",
                    conversation_id=self.conversation_id,
                    metadata={"index": i}
                )
                added_memory_ids.append(memory_id)
                add_count += 1
            elif operation == "get" and added_memory_ids:
                # 获取随机添加的记忆
                memory_id = random.choice(added_memory_ids)
                self.memory_system.get_short_term_memory(memory_id)
                get_count += 1
            elif operation == "search":
                # 搜索短期记忆
                self.memory_system.search_short_term_memories(self.conversation_id, limit=10)
                search_count += 1
        
        end_time = time.time()
        total_time = (end_time - start_time) * 1000  # 转换为毫秒
        
        print(f"压力测试结果：")
        print(f"总操作数：{total_operations}")
        print(f"添加操作：{add_count}")
        print(f"获取操作：{get_count}")
        print(f"搜索操作：{search_count}")
        print(f"总耗时：{total_time:.2f}ms")
        print(f"平均操作耗时：{total_time/total_operations:.2f}ms")
        
        # 验证平均操作耗时≤100ms
        self.assertLess(total_time/total_operations, 100, f"压力测试平均操作耗时{total_time/total_operations:.2f}ms超过100ms")


if __name__ == "__main__":
    # 运行单元测试
    unittest.main(verbosity=2)
