import unittest
from zyantine_genisis.cognition.core_identity import CoreIdentity


class TestCoreIdentity(unittest.TestCase):
    """
    CoreIdentity类的单元测试
    """
    
    def setUp(self):
        """
        测试前的设置
        """
        self.core_identity = CoreIdentity()
    
    def test_initialization(self):
        """
        测试初始化是否成功
        """
        self.assertIsInstance(self.core_identity, CoreIdentity)
        self.assertIsNotNone(self.core_identity.basic_profile)
        self.assertIsNotNone(self.core_identity.interaction_masks)
        self.assertIsNotNone(self.core_identity.cognitive_tools)
    
    def test_select_mask(self):
        """
        测试选择面具功能
        """
        # 测试场景1：用户情绪低落，需要情感支持
        situation1 = {
            "user_emotion": "sad",
            "interaction_type": "seeking_support"
        }
        vectors1 = {
            "CS": 0.3,
            "TR": 0.5,
            "SA": 0.7
        }
        mask_name1, mask_config1 = self.core_identity.select_mask(situation1, vectors1)
        self.assertIsInstance(mask_name1, str)
        self.assertIsInstance(mask_config1, dict)
        
        # 测试场景2：高复杂度话题，需要高效协作
        situation2 = {
            "topic_complexity": "high",
            "urgency_level": "high"
        }
        vectors2 = {
            "CS": 0.6,
            "TR": 0.7,
            "SA": 0.3
        }
        mask_name2, mask_config2 = self.core_identity.select_mask(situation2, vectors2)
        self.assertIsInstance(mask_name2, str)
        self.assertIsInstance(mask_config2, dict)
    
    def test_add_interaction_mask(self):
        """
        测试添加交互面具功能
        """
        new_mask_name = "测试面具"
        new_mask_config = {
            "description": "测试用的交互面具",
            "purpose": "测试功能",
            "target_vectors": ["测试", "验证"],
            "vector_type": "CS",
            "activation_condition": "测试时",
            "communication_style": "测试风格",
            "emotional_distance": "测试距离",
            "strategic_value": "测试价值"
        }
        
        # 测试添加成功
        result = self.core_identity.add_interaction_mask(new_mask_name, new_mask_config)
        self.assertTrue(result)
        self.assertIn(new_mask_name, self.core_identity.interaction_masks)
        
        # 测试添加失败（缺少字段）
        incomplete_mask_config = {
            "description": "测试用的交互面具",
            "purpose": "测试功能"
        }
        result = self.core_identity.add_interaction_mask("测试面具2", incomplete_mask_config)
        self.assertFalse(result)
    
    def test_remove_interaction_mask(self):
        """
        测试移除交互面具功能
        """
        # 先添加一个测试面具
        new_mask_name = "测试面具"
        new_mask_config = {
            "description": "测试用的交互面具",
            "purpose": "测试功能",
            "target_vectors": ["测试", "验证"],
            "vector_type": "CS",
            "activation_condition": "测试时",
            "communication_style": "测试风格",
            "emotional_distance": "测试距离",
            "strategic_value": "测试价值"
        }
        self.core_identity.add_interaction_mask(new_mask_name, new_mask_config)
        
        # 测试移除成功
        result = self.core_identity.remove_interaction_mask(new_mask_name)
        self.assertTrue(result)
        self.assertNotIn(new_mask_name, self.core_identity.interaction_masks)
        
        # 测试移除失败（面具不存在）
        result = self.core_identity.remove_interaction_mask("不存在的面具")
        self.assertFalse(result)
    
    def test_add_cognitive_tool(self):
        """
        测试添加认知工具功能
        """
        new_tool_name = "测试工具"
        new_tool_config = {
            "strategy": "测试策略",
            "internal_motive_annotation": "测试动机注解",
            "activation_conditions": ["测试条件1", "测试条件2"],
            "expected_outcome": "测试结果",
            "risk_level": "测试风险",
            "energy_cost": "测试能耗"
        }
        
        # 测试添加成功
        result = self.core_identity.add_cognitive_tool(new_tool_name, new_tool_config)
        self.assertTrue(result)
        self.assertIn(new_tool_name, self.core_identity.cognitive_tools)
        
        # 测试添加失败（缺少字段）
        incomplete_tool_config = {
            "strategy": "测试策略",
            "internal_motive_annotation": "测试动机注解"
        }
        result = self.core_identity.add_cognitive_tool("测试工具2", incomplete_tool_config)
        self.assertFalse(result)
    
    def test_remove_cognitive_tool(self):
        """
        测试移除认知工具功能
        """
        # 先添加一个测试工具
        new_tool_name = "测试工具"
        new_tool_config = {
            "strategy": "测试策略",
            "internal_motive_annotation": "测试动机注解",
            "activation_conditions": ["测试条件1", "测试条件2"],
            "expected_outcome": "测试结果",
            "risk_level": "测试风险",
            "energy_cost": "测试能耗"
        }
        self.core_identity.add_cognitive_tool(new_tool_name, new_tool_config)
        
        # 测试移除成功
        result = self.core_identity.remove_cognitive_tool(new_tool_name)
        self.assertTrue(result)
        self.assertNotIn(new_tool_name, self.core_identity.cognitive_tools)
        
        # 测试移除失败（工具不存在）
        result = self.core_identity.remove_cognitive_tool("不存在的工具")
        self.assertFalse(result)
    
    def test_cache_mechanism(self):
        """
        测试缓存机制
        """
        # 测试相同输入是否返回相同结果（缓存生效）
        situation = {
            "user_emotion": "sad",
            "interaction_type": "seeking_support"
        }
        vectors = {
            "CS": 0.3,
            "TR": 0.5,
            "SA": 0.7
        }
        
        # 第一次调用
        mask_name1, mask_config1 = self.core_identity.select_mask(situation, vectors)
        # 第二次调用（应该使用缓存）
        mask_name2, mask_config2 = self.core_identity.select_mask(situation, vectors)
        
        self.assertEqual(mask_name1, mask_name2)
        self.assertEqual(mask_config1, mask_config2)


if __name__ == '__main__':
    unittest.main()
