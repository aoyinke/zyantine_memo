"""
测试协议模块
"""

import unittest
from unittest.mock import Mock, MagicMock

from protocols.expression_validator import ExpressionValidator
from protocols.length_regulator import LengthRegulator


class TestExpressionValidator(unittest.TestCase):
    """测试表达验证器"""

    def setUp(self):
        """设置测试环境"""
        self.validator = ExpressionValidator()

    def test_validate_and_transform_removes_forbidden_symbols(self):
        """测试移除违禁符号"""
        test_text = "这是一个测试[违禁符号]文本（括号内容）"

        result, violations = self.validator.validate_and_transform(test_text)

        # 检查违禁符号是否被移除
        self.assertNotIn('[', result)
        self.assertNotIn(']', result)
        self.assertNotIn('(', result)
        self.assertNotIn(')', result)

        # 检查是否有违规记录
        self.assertTrue(len(violations) > 0)

    def test_implication_principle(self):
        """测试暗示原则转换"""
        test_text = "我感到开心，今天天气真好"

        result, _ = self.validator.validate_and_transform(test_text)

        # 检查是否进行了暗示转换
        self.assertIn("语气轻快地说", result)
        self.assertNotIn("我感到开心", result)


class TestLengthRegulator(unittest.TestCase):
    """测试长度规整器"""

    def setUp(self):
        """设置测试环境"""
        self.regulator = LengthRegulator()

    def test_regulate_critical_moment(self):
        """测试危机时刻的规整"""
        draft = "这是一个测试句子。这是第二个句子。这是第三个句子。这是第四个句子。"
        snapshot = {
            "external_context": {
                "user_emotion": "sad",
                "raw_input_preview": "救命，我很难过"
            },
            "internal_state_tags": []
        }

        result = self.regulator.regulate(draft, snapshot)

        # 检查结果是否非空
        self.assertTrue(len(result) > 0)

    def test_priority_sorting(self):
        """测试句子优先级排序"""
        sentences = ["这是一个重要句子", "这是一个普通句子", "这是另一个重要句子"]
        snapshot = {
            "initial_desire_focus": "优先巩固CS向量",
            "external_context": {
                "user_emotion": "neutral"
            }
        }

        prioritized = self.regulator._prioritize_sentences(sentences, snapshot)

        # 检查是否进行了排序
        self.assertEqual(len(prioritized), 3)
        # 检查是否按优先级排序（降序）
        for i in range(len(prioritized) - 1):
            self.assertGreaterEqual(prioritized[i]["priority_score"], prioritized[i + 1]["priority_score"])


if __name__ == '__main__':
    unittest.main()