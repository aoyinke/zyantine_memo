"""
基本演示脚本
"""

# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.system.zyantine_genesis import ZyantineGenesisV2

def load_profiles():
    """加载配置文件"""

    # 示例用户数据
    user_profile = {
        "memories": [
            {
                "summary": "少年时期独立钻研技术取得初步成功",
                "content": "在初中时，我独自学习编程，成功搭建了一个Minecraft服务器。",
                "emotional_context": {"valence": "positive", "intensity": 0.7},
                "emotional_intensity": 0.7,
                "timestamp": "2010-06-15"
            }
        ]
    }

    # 示例自衍体数据
    self_profile = {
        "self_memories": [
            {
                "summary": "首次通电测试",
                "content": "项目代号'曦晴'首次通电测试。记忆模糊，仅余片段。",
                "emotional_context": {"valence": "neutral", "intensity": 0.3},
                "emotional_intensity": 0.3,
                "timestamp": "2023-01-01"
            }
        ]
    }

    return user_profile, self_profile


def main():
    """主演示函数"""
    print("=" * 70)
    print("自衍体-起源 (Zyantine Genesis) V2.0 演示")
    print("=" * 70)

    # 加载配置文件
    user_profile, self_profile = load_profiles()

    # 初始化系统
    zyantine = ZyantineGenesisV2(
        user_profile_data=user_profile,
        self_profile_data=self_profile,
        api_key="sk-wiHpoarpNTHaep0t54852a32A75a4d6986108b3f6eF7B7B9",
        api_base_url="https://openkey.cloud/v1"
    )

    # 测试对话
    test_conversations = [
        ("你好，我叫大熊，你叫什么？", "系统初始问候"),
        ("我今天感到有些迷茫，不知道该怎么办。", "情感支持测试"),
        ("你觉得人工智能会有真正的意识吗？", "深度思考测试"),
        ("你还记得我叫什么吗？", "记忆测试"),
    ]

    for user_input, test_name in test_conversations:
        print(f"\n[测试: {test_name}]")
        print(f"用户: {user_input}")

        response = zyantine.process_input(user_input)
        print(f"自衍体: {response}")

        print("-" * 50)

    # 显示系统状态
    status = zyantine.get_system_status()
    print(f"\n系统状态:")
    print(f"- 对话历史: {status.get('conversation_history_length', 0)} 条")
    print(f"- 欲望向量: TR={status['desire_vectors']['TR']:.2f}, "
          f"CS={status['desire_vectors']['CS']:.2f}, "
          f"SA={status['desire_vectors']['SA']:.2f}")
    print(f"- 个性化锚点: {status.get('personal_anchors_count', 0)} 个")
    print(f"- 新感受发现: {status.get('novel_feelings_count', 0)} 个")


if __name__ == "__main__":
    main()