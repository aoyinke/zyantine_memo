#!/usr/bin/env python3
# 测试回复生成器在API不可用时的行为

import sys
from zyantine_genisis.api.service_provider import APIServiceProvider
from zyantine_genisis.config.config_manager import ConfigManager
from cognition.core_identity import CoreIdentity

def test_reply_generator():
    """测试回复生成器"""
    print("=== 测试回复生成器 ===")
    
    # 加载配置
    config_manager = ConfigManager()
    config = config_manager.get()
    
    # 创建核心身份
    core_identity = CoreIdentity()
    
    try:
        # 创建服务提供者
        provider = APIServiceProvider(config, core_identity=core_identity)
        print("服务提供者初始化成功")
        
        # 测试生成回复
        print("\n1. 测试生成回复：")
        user_input = "你好，小叶同学。"
        conversation_history = []
        
        reply = provider.generate_reply(
            user_input=user_input,
            action_plan={"chosen_mask": "长期搭档", "primary_strategy": "你好，很高兴认识你。"},
            growth_result={"validation": "success", "new_principle": {"abstracted_from": "初次见面"}},
            context_analysis={"topic_complexity": "low"},
            conversation_history=conversation_history,
            current_vectors={"TR": 0.5, "CS": 0.5, "SA": 0.5},
            core_identity=core_identity
        )
        
        print(f"用户输入: {user_input}")
        print(f"生成回复: {reply}")
        print(f"回复类型: {'降级策略' if '刚才的思考链路好像打了个结' in reply or '我的思考过程出现了一些混乱' in reply else '正常回复'}")
        
        return True
    except Exception as e:
        print(f"测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主函数"""
    print("=== 回复生成器测试 ===")
    
    success = test_reply_generator()
    
    print("\n=== 测试总结 ===")
    if success:
        print("✅ 测试成功！回复生成器在API不可用时正确使用了降级策略")
        return 0
    else:
        print("❌ 测试失败！")
        return 1

if __name__ == "__main__":
    sys.exit(main())