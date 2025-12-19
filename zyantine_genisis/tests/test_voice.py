# main.py
import os
from src.system.zyantine_genesis import ZyantineGenesisV2

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
# 初始化系统
zyantine = ZyantineGenesisV2(
        user_profile_data=user_profile,
        self_profile_data=self_profile,
        api_key="sk-wiHpoarpNTHaep0t54852a32A75a4d6986108b3f6eF7B7B9",
        api_base_url="https://openkey.cloud/v1"
)

# 交互循环
while True:
    user_input = input("\n你: ")
    if user_input.lower() in ['退出', 'exit', 'quit']:
        break

    response = zyantine.process_input(user_input)
    print(f"\n自衍体: {response}")