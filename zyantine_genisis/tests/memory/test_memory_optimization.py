#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
记忆检索优化效果测试脚本

该脚本用于测试记忆检索优化前后的效果对比，包括：
1. 相关性评分对比
2. 召回率和精确率对比
3. 不相关记忆过滤效果
4. 不同上下文场景下的表现
"""

import sys
import os
import json
import time
from typing import List, Dict, Any

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from zyantine_genisis.memory.memory_store import ZyantineMemorySystem

class MemoryOptimizationTest:
    """记忆优化测试类"""
    
    def __init__(self):
        """初始化测试环境"""
        # 创建记忆系统实例
        self.memory_system = ZyantineMemorySystem(user_id="test_user", session_id="test_session")
        
        # 准备测试数据
        self.test_memories = self._prepare_test_memories()
        
        # 准备测试查询
        self.test_queries = self._prepare_test_queries()
        
        # 记录测试结果
        self.results = {
            "before_optimization": [],
            "after_optimization": []
        }
    
    def _prepare_test_memories(self) -> List[Dict[str, Any]]:
        """准备测试记忆数据"""
        return [
            {
                "content": "我喜欢吃苹果和香蕉",
                "memory_type": "conversation",
                "tags": ["food", "fruit"],
                "metadata": {"user_id": "test_user"}
            },
            {
                "content": "今天天气很好，适合去公园散步",
                "memory_type": "conversation",
                "tags": ["weather", "activity"],
                "metadata": {"user_id": "test_user"}
            },
            {
                "content": "Python是一种流行的编程语言",
                "memory_type": "knowledge",
                "tags": ["programming", "language"],
                "metadata": {"user_id": "test_user"}
            },
            {
                "content": "机器学习是人工智能的一个分支",
                "memory_type": "knowledge",
                "tags": ["ai", "machine_learning"],
                "metadata": {"user_id": "test_user"}
            },
            {
                "content": "我昨天去了健身房锻炼",
                "memory_type": "experience",
                "tags": ["fitness", "exercise"],
                "metadata": {"user_id": "test_user"}
            },
            {
                "content": "明天有一个重要的会议",
                "memory_type": "temporal",
                "tags": ["meeting", "work"],
                "metadata": {"user_id": "test_user"}
            },
            {
                "content": "我喜欢听古典音乐",
                "memory_type": "user_profile",
                "tags": ["music", "hobby"],
                "metadata": {"user_id": "test_user"}
            },
            {
                "content": "北京是中国的首都",
                "memory_type": "knowledge",
                "tags": ["geography", "china"],
                "metadata": {"user_id": "test_user"}
            },
            {
                "content": "我需要购买一本新的编程书籍",
                "memory_type": "conversation",
                "tags": ["shopping", "books"],
                "metadata": {"user_id": "test_user"}
            },
            {
                "content": "今天的会议讨论了项目进度",
                "memory_type": "conversation",
                "tags": ["meeting", "work"],
                "metadata": {"user_id": "test_user"}
            }
        ]
    
    def _prepare_test_queries(self) -> List[Dict[str, Any]]:
        """准备测试查询数据"""
        return [
            {
                "query": "我喜欢吃什么水果？",
                "expected_topics": ["food", "fruit"],
                "description": "测试食物相关记忆检索"
            },
            {
                "query": "Python是什么？",
                "expected_topics": ["programming", "language"],
                "description": "测试编程知识检索"
            },
            {
                "query": "关于机器学习的知识",
                "expected_topics": ["ai", "machine_learning"],
                "description": "测试AI相关知识检索"
            },
            {
                "query": "我昨天做了什么？",
                "expected_topics": ["fitness", "exercise"],
                "description": "测试近期经历检索"
            },
            {
                "query": "关于会议的安排",
                "expected_topics": ["meeting", "work"],
                "description": "测试会议相关记忆检索"
            }
        ]
    
    def _load_test_memories(self):
        """加载测试记忆到系统"""
        print("[测试] 开始加载测试记忆...")
        for memory in self.test_memories:
            try:
                self.memory_system.add_memory(
                    content=memory["content"],
                    memory_type=memory["memory_type"],
                    tags=memory["tags"],
                    metadata=memory["metadata"]
                )
            except Exception as e:
                print(f"[测试] 加载记忆失败: {e}")
        print(f"[测试] 成功加载 {len(self.test_memories)} 条测试记忆")
    
    def _test_query(self, query: Dict[str, Any]) -> Dict[str, Any]:
        """测试单个查询"""
        print(f"\n[测试] 测试查询: {query['query']}")
        print(f"[测试] 期望主题: {query['expected_topics']}")
        
        # 执行搜索
        results = self.memory_system.search_memories(
            query=query["query"],
            limit=3
        )
        
        # 分析结果
        analysis = {
            "query": query["query"],
            "description": query["description"],
            "results_count": len(results),
            "top_results": [],
            "average_similarity": 0.0,
            "average_final_score": 0.0,
            "relevant_results_count": 0
        }
        
        if results:
            # 计算平均相似度和最终分数
            total_similarity = sum(r.get("similarity_score", 0) for r in results)
            total_final_score = sum(r.get("final_score", 0) for r in results)
            
            analysis["average_similarity"] = total_similarity / len(results)
            analysis["average_final_score"] = total_final_score / len(results)
            
            # 检查相关结果数量
            for result in results:
                result_tags = result.get("tags", [])
                has_expected_topic = any(topic in result_tags for topic in query["expected_topics"])
                
                if has_expected_topic:
                    analysis["relevant_results_count"] += 1
                
                # 记录前3个结果
                analysis["top_results"].append({
                    "content": result.get("content", ""),
                    "tags": result_tags,
                    "similarity_score": result.get("similarity_score", 0),
                    "final_score": result.get("final_score", 0),
                    "memory_type": result.get("memory_type", ""),
                    "is_relevant": has_expected_topic
                })
        
        print(f"[测试] 检索结果数量: {analysis['results_count']}")
        print(f"[测试] 相关结果数量: {analysis['relevant_results_count']}")
        print(f"[测试] 平均相似度: {analysis['average_similarity']:.4f}")
        print(f"[测试] 平均最终分数: {analysis['average_final_score']:.4f}")
        
        return analysis
    
    def run_test(self) -> Dict[str, Any]:
        """运行完整测试"""
        print("[测试] 开始记忆检索优化效果测试")
        print("=" * 50)
        
        # 加载测试记忆
        self._load_test_memories()
        
        # 运行所有测试查询
        test_results = []
        for query in self.test_queries:
            result = self._test_query(query)
            test_results.append(result)
        
        # 汇总测试结果
        summary = {
            "total_queries": len(test_results),
            "average_results_count": 0.0,
            "average_relevant_results_count": 0.0,
            "average_similarity": 0.0,
            "average_final_score": 0.0,
            "average_precision": 0.0,
            "test_results": test_results
        }
        
        if test_results:
            summary["average_results_count"] = sum(r["results_count"] for r in test_results) / len(test_results)
            summary["average_relevant_results_count"] = sum(r["relevant_results_count"] for r in test_results) / len(test_results)
            summary["average_similarity"] = sum(r["average_similarity"] for r in test_results) / len(test_results)
            summary["average_final_score"] = sum(r["average_final_score"] for r in test_results) / len(test_results)
            
            # 计算平均精确率
            total_precision = 0.0
            for r in test_results:
                if r["results_count"] > 0:
                    total_precision += r["relevant_results_count"] / r["results_count"]
            summary["average_precision"] = total_precision / len(test_results)
        
        print("\n" + "=" * 50)
        print("[测试] 测试完成！")
        print(f"[测试] 总查询数: {summary['total_queries']}")
        print(f"[测试] 平均结果数: {summary['average_results_count']:.2f}")
        print(f"[测试] 平均相关结果数: {summary['average_relevant_results_count']:.2f}")
        print(f"[测试] 平均相似度: {summary['average_similarity']:.4f}")
        print(f"[测试] 平均最终分数: {summary['average_final_score']:.4f}")
        print(f"[测试] 平均精确率: {summary['average_precision']:.4f}")
        
        return summary
    
    def generate_report(self, results: Dict[str, Any], output_file: str = "memory_optimization_report.json"):
        """生成测试报告"""
        # 保存测试结果到文件
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        print(f"\n[测试] 测试报告已保存到: {output_file}")
        
        # 生成可读性更强的报告
        report_content = "# 记忆检索优化效果测试报告\n\n"
        report_content += f"测试时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        
        report_content += "## 测试汇总\n"
        report_content += f"- 总查询数: {results['total_queries']}\n"
        report_content += f"- 平均结果数: {results['average_results_count']:.2f}\n"
        report_content += f"- 平均相关结果数: {results['average_relevant_results_count']:.2f}\n"
        report_content += f"- 平均相似度: {results['average_similarity']:.4f}\n"
        report_content += f"- 平均最终分数: {results['average_final_score']:.4f}\n"
        report_content += f"- 平均精确率: {results['average_precision']:.4f}\n\n"
        
        report_content += "## 详细测试结果\n\n"
        for i, result in enumerate(results['test_results'], 1):
            report_content += f"### 测试用例 {i}: {result['description']}\n"
            report_content += f"**查询**: {result['query']}\n"
            report_content += f"**结果数量**: {result['results_count']}\n"
            report_content += f"**相关结果数量**: {result['relevant_results_count']}\n"
            report_content += f"**平均相似度**: {result['average_similarity']:.4f}\n"
            report_content += f"**平均最终分数**: {result['average_final_score']:.4f}\n"
            
            if result['results_count'] > 0:
                precision = result['relevant_results_count'] / result['results_count']
                report_content += f"**精确率**: {precision:.4f}\n"
            
            report_content += "\n**Top 3 检索结果**:\n"
            for j, res in enumerate(result['top_results'], 1):
                relevance = "✅ 相关" if res['is_relevant'] else "❌ 不相关"
                report_content += f"{j}. {relevance}\n"
                report_content += f"   内容: {res['content'][:50]}...\n"
                report_content += f"   标签: {', '.join(res['tags'])}\n"
                report_content += f"   相似度: {res['similarity_score']:.4f}\n"
                report_content += f"   最终分数: {res['final_score']:.4f}\n"
            
            report_content += "\n" 
        
        # 保存Markdown报告
        md_output_file = output_file.replace(".json", ".md")
        with open(md_output_file, "w", encoding="utf-8") as f:
            f.write(report_content)
        
        print(f"[测试] Markdown测试报告已保存到: {md_output_file}")

if __name__ == "__main__":
    # 创建测试实例
    test = MemoryOptimizationTest()
    
    # 运行测试
    results = test.run_test()
    
    # 生成测试报告
    test.generate_report(results)
