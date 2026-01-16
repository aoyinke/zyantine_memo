#!/usr/bin/env python3
"""
测试对话信息存储功能
"""
import sys
import os
import json
from datetime import datetime

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from zyantine_genisis.memory.memory_store import ZyantineMemorySystem
from config.config_manager import ConfigManager

class ConversationStorageTester:
    """对话存储功能测试器"""
    
    def __init__(self):
        """初始化测试器"""
        self.config = ConfigManager().get()
        self.memory_system = None
        self.test_results = []
    
    def initialize_memory_system(self):
        """初始化记忆系统"""
        try:
            self.memory_system = ZyantineMemorySystem(
                user_id="test_user",
                session_id="test_session",
                memo0_config=self.config.memory.memo0_config
            )
            print("[测试] 记忆系统初始化成功")
            return True
        except Exception as e:
            print(f"[测试] 记忆系统初始化失败: {e}")
            return False
    
    def test_store_conversation(self):
        """测试存储对话功能"""
        try:
            test_conversations = [
                {
                    "user_input": "你好，我想了解一下如何学习编程",
                    "system_response": "你好！学习编程可以从Python开始，它语法简单易懂，适合初学者。建议你先学习基础语法，然后尝试做一些小项目来巩固知识。",
                    "emotional_intensity": 0.6,
                    "tags": ["编程学习", "初学者"]
                },
                {
                    "user_input": "Python有哪些好的学习资源？",
                    "system_response": "推荐你看《Python编程：从入门到实践》这本书，还有Codecademy和LeetCode这样的在线平台。YouTube上也有很多优质的Python教程视频。",
                    "emotional_intensity": 0.7,
                    "tags": ["编程学习", "资源推荐"]
                },
                {
                    "user_input": "谢谢！",
                    "system_response": "不客气！如果你在学习过程中遇到任何问题，随时可以问我。祝你学习顺利！",
                    "emotional_intensity": 0.8,
                    "tags": ["感谢", "结束对话"]
                }
            ]
            
            results = []
            for i, conversation in enumerate(test_conversations):
                print(f"\n[测试] 测试对话存储 {i+1}")
                print(f"用户输入: {conversation['user_input']}")
                print(f"系统回复: {conversation['system_response']}")
                
                # 调用存储对话方法
                result = self.memory_system.store_conversation(
                    user_input=conversation['user_input'],
                    system_response=conversation['system_response'],
                    emotional_intensity=conversation['emotional_intensity'],
                    tags=conversation['tags']
                )
                
                if not result.startswith("存储对话失败"):
                    print(f"[测试] 存储对话成功，ID: {result}")
                    results.append({
                        "conversation": conversation,
                        "memory_id": result,
                        "success": True
                    })
                else:
                    print(f"[测试] 存储对话失败: {result}")
                    results.append({
                        "conversation": conversation,
                        "error": result,
                        "success": False
                    })
            
            return results
        except Exception as e:
            print(f"[测试] 测试存储对话失败: {e}")
            return []
    
    def test_search_conversations(self):
        """测试搜索对话功能"""
        try:
            test_queries = ["编程学习", "Python"]
            
            for query in test_queries:
                print(f"\n[测试] 搜索对话: {query}")
                
                results = self.memory_system.search_memories(
                    query=query,
                    memory_type="conversation",
                    limit=3
                )
                
                if results:
                    print(f"[测试] 找到 {len(results)} 条对话")
                    for i, result in enumerate(results):
                        print(f"  {i+1}. 相似度: {result.get('similarity_score', 0):.2f}")
                        content = result.get('content', '')
                        if isinstance(content, list):
                            for msg in content:
                                if msg.get('role') == 'user':
                                    print(f"     用户: {msg.get('content', '')[:50]}...")
                                elif msg.get('role') == 'assistant':
                                    print(f"     系统: {msg.get('content', '')[:50]}...")
                        else:
                            print(f"     内容: {content[:100]}...")
                else:
                    print(f"[测试] 未找到相关对话")
            
            return True
        except Exception as e:
            print(f"[测试] 测试搜索对话失败: {e}")
            return False
    
    def run_all_tests(self):
        """运行所有测试"""
        print("=" * 60)
        print("开始测试对话信息存储功能")
        print("=" * 60)
        
        # 初始化记忆系统
        if not self.initialize_memory_system():
            return False
        
        # 测试存储对话
        print("\n" + "-" * 60)
        print("测试1: 存储对话功能")
        print("-" * 60)
        storage_results = self.test_store_conversation()
        
        # 测试搜索对话
        print("\n" + "-" * 60)
        print("测试2: 搜索对话功能")
        print("-" * 60)
        search_success = self.test_search_conversations()
        
        # 总结
        print("\n" + "=" * 60)
        print("测试总结")
        print("=" * 60)
        
        total_storage_tests = len(storage_results)
        successful_storage_tests = sum(1 for r in storage_results if r['success'])
        
        print(f"存储对话测试: {successful_storage_tests}/{total_storage_tests} 成功")
        print(f"搜索对话测试: {'成功' if search_success else '失败'}")
        
        overall_success = successful_storage_tests >= total_storage_tests * 0.7 and search_success
        print(f"\n整体测试结果: {'成功' if overall_success else '失败'}")
        
        return overall_success

if __name__ == "__main__":
    tester = ConversationStorageTester()
    success = tester.run_all_tests()
    sys.exit(0 if success else 1)
