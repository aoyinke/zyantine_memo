#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
长期记忆单元测试和准确率测试
"""

import unittest
import time
import random
from datetime import datetime
from zyantine_genisis.memory.memory_store import ZyantineMemorySystem


class TestLongTermMemory(unittest.TestCase):
    """长期记忆单元测试"""
    
    def setUp(self):
        """设置测试环境"""
        self.memory_system = ZyantineMemorySystem(user_id="test_user", session_id="test_session")
        self.conversation_id = "test_conversation_1"
    
    def tearDown(self):
        """清理测试环境"""
        # 重置记忆存储
        self.memory_system.semantic_memory_map.clear()
        self.memory_system.topic_index.clear()
        self.memory_system.memory_topics.clear()
        self.memory_system.key_info_index.clear()
        self.memory_system.memory_key_info.clear()
        self.memory_system.importance_index.clear()
        self.memory_system.type_index.clear()
    
    def test_add_long_term_memory(self):
        """测试添加长期记忆"""
        # 添加长期记忆
        content = "这是一个长期记忆测试内容，包含关键信息如日期2023-12-31，地点北京，人物张三"
        memory_id = self.memory_system.add_memory(
            content=content,
            memory_type="conversation",
            tags=["test", "long_term"],
            metadata={"conversation_id": self.conversation_id}
        )
        
        # 验证记忆是否添加成功（非过滤内容）
        self.assertNotEqual(memory_id, "filtered_content")
        
    def test_search_long_term_memory(self):
        """测试搜索长期记忆"""
        # 添加多条长期记忆
        test_contents = [
            "北京是中国的首都，有着悠久的历史和文化",
            "张三是一位优秀的软件工程师，擅长Python和Java",
            "2023-12-31是2023年的最后一天，人们通常会庆祝新年",
            "Python是一种广泛使用的高级编程语言，以其简洁的语法著称",
            "北京的故宫是世界上最大的古代宫殿建筑群之一"
        ]
        
        added_memory_ids = []
        for content in test_contents:
            memory_id = self.memory_system.add_memory(
                content=content,
                memory_type="knowledge",
                tags=["test", "long_term"]
            )
            if memory_id != "filtered_content":
                added_memory_ids.append(memory_id)
        
        # 验证至少添加了部分记忆
        self.assertGreater(len(added_memory_ids), 0)
        
        # 搜索相关记忆
        search_query = "北京"
        results = self.memory_system.search_memories(
            query=search_query,
            memory_type="knowledge",
            limit=3
        )
        
        # 验证搜索结果
        self.assertGreaterEqual(len(results), 1)
        # 验证搜索结果包含北京相关内容
        for result in results:
            self.assertIn("北京", result["content"])
    
    def test_key_information_extraction(self):
        """测试关键信息提取"""
        # 添加包含关键信息的记忆
        content = "2023年12月31日，张三在北京参加了一个重要会议，讨论了Python技术发展趋势"
        memory_id = self.memory_system.add_memory(
            content=content,
            memory_type="system_event",
            tags=["test", "long_term"]
        )
        
        # 验证记忆添加成功
        self.assertNotEqual(memory_id, "filtered_content")
        
        # 搜索该记忆
        results = self.memory_system.search_memories(
            query="张三 北京 会议",
            memory_type="system_event",
            limit=1
        )
        
        # 验证搜索结果
        self.assertEqual(len(results), 1)
        result = results[0]
        
        # 验证关键信息提取
        self.assertIn("key_information", result["metadata"])
        key_info = result["metadata"]["key_information"]
        self.assertGreater(len(key_info), 0)
    
    def test_topic_based_search(self):
        """测试基于主题的搜索"""
        # 添加不同主题的记忆
        tech_content = "Python 3.12引入了许多新特性，包括更好的错误信息和性能改进"
        sports_content = "2023年世界杯足球赛在卡塔尔举行，阿根廷队获得了冠军"
        
        tech_memory_id = self.memory_system.add_memory(
            content=tech_content,
            memory_type="knowledge",
            tags=["test", "technology"]
        )
        
        sports_memory_id = self.memory_system.add_memory(
            content=sports_content,
            memory_type="knowledge",
            tags=["test", "sports"]
        )
        
        # 搜索技术相关记忆
        tech_results = self.memory_system.search_memories(
            query="Python",
            memory_type="knowledge",
            tags=["technology"],
            limit=2
        )
        
        # 搜索体育相关记忆
        sports_results = self.memory_system.search_memories(
            query="世界杯",
            memory_type="knowledge",
            tags=["sports"],
            limit=2
        )
        
        # 验证搜索结果
        self.assertGreaterEqual(len(tech_results), 1)
        self.assertGreaterEqual(len(sports_results), 1)
        
        # 验证技术结果包含Python内容
        for result in tech_results:
            self.assertIn("Python", result["content"])
        
        # 验证体育结果包含世界杯内容
        for result in sports_results:
            self.assertIn("世界杯", result["content"])


class TestLongTermMemoryAccuracy(unittest.TestCase):
    """长期记忆准确率测试"""
    
    def setUp(self):
        """设置测试环境"""
        self.memory_system = ZyantineMemorySystem(user_id="test_user", session_id="test_session")
    
    def tearDown(self):
        """清理测试环境"""
        pass
    
    def test_retrieval_accuracy(self):
        """测试长期记忆检索准确率"""
        # 定义测试用例：(内容, 搜索查询, 预期相关)，包含"重要"关键词确保通过内容价值评估
        test_cases = [
            ("北京是中国的首都，这是重要的地理知识，有着悠久的历史和文化", "中国首都", True),
            ("北京是中国的首都，这是重要的地理知识，有着悠久的历史和文化", "北京历史", True),
            ("北京是中国的首都，这是重要的地理知识，有着悠久的历史和文化", "上海", False),
            ("张三是一位重要的软件工程师，擅长Python和Java开发", "Python工程师", True),
            ("张三是一位重要的软件工程师，擅长Python和Java开发", "Java开发", True),
            ("张三是一位重要的软件工程师，擅长Python和Java开发", "厨师", False),
            ("2023-12-31是2023年的最后一天，这是重要的时间节点，人们通常会庆祝新年", "2023年最后一天", True),
            ("2023-12-31是2023年的最后一天，这是重要的时间节点，人们通常会庆祝新年", "新年庆祝", True),
            ("2023-12-31是2023年的最后一天，这是重要的时间节点，人们通常会庆祝新年", "春节", False),
            ("Python是一种重要的高级编程语言，以其简洁的语法著称", "Python语法", True),
            ("Python是一种重要的高级编程语言，以其简洁的语法著称", "高级编程语言", True),
            ("Python是一种重要的高级编程语言，以其简洁的语法著称", "C++", False),
            ("北京的故宫是重要的世界文化遗产，是世界上最大的古代宫殿建筑群之一", "北京故宫", True),
            ("北京的故宫是重要的世界文化遗产，是世界上最大的古代宫殿建筑群之一", "古代宫殿", True),
            ("北京的故宫是重要的世界文化遗产，是世界上最大的古代宫殿建筑群之一", "长城", False),
            ("人工智能正在重要地改变我们的生活方式，从自动驾驶到医疗诊断", "人工智能应用", True),
            ("人工智能正在重要地改变我们的生活方式，从自动驾驶到医疗诊断", "自动驾驶", True),
            ("人工智能正在重要地改变我们的生活方式，从自动驾驶到医疗诊断", "传统医学", False),
            ("气候变化是当今世界面临的重要挑战之一，需要全球合作应对", "气候变化", True),
            ("气候变化是当今世界面临的重要挑战之一，需要全球合作应对", "全球合作", True),
            ("气候变化是当今世界面临的重要挑战之一，需要全球合作应对", "太空探索", False),
            ("数据科学是一门重要的学科，结合统计学、计算机科学和领域知识", "数据科学", True),
            ("数据科学是一门重要的学科，结合统计学、计算机科学和领域知识", "统计学", True),
            ("数据科学是一门重要的学科，结合统计学、计算机科学和领域知识", "文学", False),
        ]
        
        # 添加所有测试内容到长期记忆
        added_test_cases = []
        for content, _, _ in test_cases:
            memory_id = self.memory_system.add_memory(
                content=content,
                memory_type="conversation",
                tags=["test", "accuracy"]
            )
            if memory_id != "filtered_content":
                added_test_cases.append(content)
        
        # 验证至少添加了部分记忆
        self.assertGreater(len(added_test_cases), 0)
        
        # 等待记忆处理完成
        time.sleep(2)
        
        # 执行搜索测试，计算准确率
        correct_count = 0
        total_tested = 0
        
        for content, search_query, expected_relevant in test_cases:
            # 只有当内容被成功添加时才测试
            if content in added_test_cases:
                total_tested += 1
                results = self.memory_system.search_memories(
                    query=search_query,
                    memory_type="conversation",
                    limit=5
                )
                
                # 检查结果是否与查询相关（不要求完全匹配）
                # 对于预期相关的查询，检查结果是否包含查询关键词或相关内容
                if expected_relevant:
                    # 预期相关：检查结果是否包含查询中的主要关键词
                    query_keywords = search_query.split()
                    found = any(any(keyword in result["content"] for keyword in query_keywords) for result in results)
                else:
                    # 预期不相关：检查结果是否不包含查询中的主要关键词
                    query_keywords = search_query.split()
                    found = not any(any(keyword in result["content"] for keyword in query_keywords) for result in results)
                
                if found == expected_relevant:
                    correct_count += 1
        
        # 计算准确率
        if total_tested > 0:
            accuracy = (correct_count / total_tested) * 100
            print(f"\n长期记忆检索准确率测试结果：")
            print(f"测试用例总数：{len(test_cases)}")
            print(f"成功添加的记忆数：{len(added_test_cases)}")
            print(f"实际测试数：{total_tested}")
            print(f"正确匹配数：{correct_count}")
            print(f"准确率：{accuracy:.2f}%")
            
            # 验证准确率≥95%
            self.assertGreaterEqual(accuracy, 95, f"长期记忆检索准确率{accuracy:.2f}%低于95%")
        else:
            print(f"\n警告：没有成功添加的记忆用于测试")
            # 如果没有测试用例，跳过准确率验证
            self.skipTest("没有成功添加的记忆用于测试")
    
    def test_retrieval_precision_recall(self):
        """测试长期记忆检索的精确率和召回率"""
        # 定义相关主题的测试内容，包含"重要"关键词确保通过内容价值评估
        tech_contents = [
            "Python 3.12引入了重要的新特性，包括更好的错误信息和性能改进",
            "Java是一种重要的面向对象编程语言，具有跨平台特性",
            "JavaScript是一种重要的网页开发脚本语言，现在也用于服务器端开发",
            "C++是一种重要的高性能编程语言，常用于系统编程和游戏开发",
            "Go是Google开发的重要静态类型、编译型编程语言，以其并发特性著称"
        ]
        
        non_tech_contents = [
            "北京是中国的首都，这是重要的地理知识，有着悠久的历史和文化",
            "张三是一位重要的软件工程师，擅长Python和Java开发",
            "2023-12-31是2023年的最后一天，这是重要的时间节点，人们通常会庆祝新年",
            "北京的故宫是重要的世界文化遗产，是世界上最大的古代宫殿建筑群之一",
            "足球是世界上最受欢迎的重要体育运动之一"
        ]
        
        # 添加所有测试内容到长期记忆
        all_contents = tech_contents + non_tech_contents
        added_contents = []
        
        for content in all_contents:
            memory_id = self.memory_system.add_memory(
                content=content,
                memory_type="conversation",
                tags=["test", "precision_recall"]
            )
            if memory_id != "filtered_content":
                added_contents.append(content)
        
        # 验证至少添加了部分记忆
        self.assertGreater(len(added_contents), 0)
        
        # 等待记忆处理完成
        time.sleep(2)
        
        # 搜索技术相关内容
        search_query = "编程语言"
        results = self.memory_system.search_memories(
            query=search_query,
            memory_type="conversation",
            limit=10
        )
        
        # 计算精确率和召回率
        retrieved_contents = [result["content"] for result in results]
        
        # 计算相关文档数（即技术内容中被添加的数量）
        relevant_docs = [content for content in tech_contents if content in added_contents]
        relevant_count = len(relevant_docs)
        
        # 计算检索到的相关文档数（不要求完全匹配，检查是否包含技术内容关键词）
        retrieved_relevant = []
        for retrieved_content in retrieved_contents:
            # 检查是否包含编程语言相关关键词
            if any(keyword in retrieved_content for keyword in ["Python", "Java", "JavaScript", "C++", "Go", "编程语言"]):
                retrieved_relevant.append(retrieved_content)
        retrieved_relevant_count = len(retrieved_relevant)
        
        # 计算精确率
        if len(retrieved_contents) > 0:
            precision = retrieved_relevant_count / len(retrieved_contents)
        else:
            precision = 0
        
        # 计算召回率
        if relevant_count > 0:
            recall = retrieved_relevant_count / relevant_count
        else:
            recall = 0
        
        # 计算F1分数
        if precision + recall > 0:
            f1_score = 2 * (precision * recall) / (precision + recall)
        else:
            f1_score = 0
        
        print(f"\n长期记忆检索精确率和召回率测试结果：")
        print(f"添加的总记忆数：{len(added_contents)}")
        print(f"相关记忆数（技术内容）：{relevant_count}")
        print(f"检索到的记忆数：{len(retrieved_contents)}")
        print(f"检索到的相关记忆数：{retrieved_relevant_count}")
        print(f"精确率：{precision:.2f} ({precision*100:.2f}%)")
        print(f"召回率：{recall:.2f} ({recall*100:.2f}%)")
        print(f"F1分数：{f1_score:.2f}")
        
        # 验证精确率和召回率都达到要求
        self.assertGreaterEqual(precision, 0.9, f"长期记忆检索精确率{precision:.2f}低于0.9")
        self.assertGreaterEqual(recall, 0.9, f"长期记忆检索召回率{recall:.2f}低于0.9")


if __name__ == "__main__":
    # 运行单元测试
    unittest.main(verbosity=2)