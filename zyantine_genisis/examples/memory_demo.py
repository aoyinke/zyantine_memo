# start_zyantine_memo0.py
# !/usr/bin/env python3
"""
自衍体-起源 (memo0记忆系统版) 启动脚本
"""

import os
import sys
import json
from datetime import datetime
from zyantine_old_version.config.config import ZyantineConfig
from zyantine_old_version.system.zyantine_memory import Memo0EnhancedZyantineGenesis
os.environ["OPENAI_API_KEY"] = "sk-wiHpoarpNTHaep0t54852a32A75a4d6986108b3f6eF7B7B9"
os.environ["OPENAI_BASE_URL"] = "https://openkey.cloud/v1"
def setup_environment():
    """设置环境变量"""
    # 检查环境变量
    required_env_vars = ["OPENAI_API_KEY"]

    missing_vars = []
    for var in required_env_vars:
        if not os.getenv(var):
            missing_vars.append(var)

    if missing_vars:
        print("⚠️  缺少必要的环境变量:")
        for var in missing_vars:
            print(f"  - {var}")

        # 尝试从配置文件读取
        config_file = "./zyantine_config.json"
        if os.path.exists(config_file):
            try:
                with open(config_file, 'r', encoding='utf-8') as f:
                    config = json.load(f)

                api_key = config.get("api", {}).get("openai_api_key", "")
                if api_key:
                    os.environ["OPENAI_API_KEY"] = api_key
                    print("✅ 从配置文件读取API密钥")
                    return True
            except:
                pass

        print("\n请设置环境变量:")
        print("  export OPENAI_API_KEY='sk-...'")
        print("  或编辑配置文件: zyantine_config.json")

        # 询问用户是否要输入
        response = input("\n是否要现在输入API密钥？(y/N): ")
        if response.lower() == 'y':
            api_key = input("请输入OpenAI API密钥: ").strip()
            if api_key:
                os.environ["OPENAI_API_KEY"] = api_key
                print("✅ API密钥已设置")

                # 保存到配置文件
                config = {
                    "api": {
                        "openai_api_key": api_key,
                        "openai_base_url": "https://openkey.cloud/v1",
                        "embedding_model": "text-embedding-3-large",
                        "embedding_dimensions": 1536,
                        "chat_model": "gpt-5-nano",
                        "enabled": True
                    },
                    "memory": {
                        "provider": "memo0",
                        "vector_store": "milvus",
                        "collection_name": "zyantine_memories"
                    }
                }

                with open(config_file, 'w', encoding='utf-8') as f:
                    json.dump(config, f, ensure_ascii=False, indent=2)

                print(f"✅ 配置已保存到: {config_file}")
                return True

        return False

    return True


def test_api_connection():
    """测试OpenAI API连接"""
    print("\n🔗 测试OpenAI API连接...")

    try:
        # 测试API连接
        from zyantine_old_version.api.service import test_api_connection

        api_key = os.getenv("OPENAI_API_KEY")
        base_url = "https://openkey.cloud/v1"
        model = "gpt-5-nano"

        success, message = test_api_connection(api_key, base_url, model)
        if success:
            print(f"✅ {message}")
            return True
        else:
            print(f"❌ {message}")
            return False

    except Exception as e:
        print(f"❌ 连接测试失败: {e}")
        return False


def main():
    """主函数"""
    print("=" * 70)
    print("自衍体-起源 (memo0记忆系统版)")
    print(f"启动时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    # 设置环境
    if not setup_environment():
        print("\n❌ 环境设置失败，程序退出")
        return 1

    # 测试连接
    if not test_api_connection():
        response = input("\n连接测试失败，是否继续？(y/N): ")
        if response.lower() != 'y':
            print("程序退出")
            return 1
    # 加载配置
    config = ZyantineConfig()

    # 初始化用户数据
    user_profile = {
        "memories": [
            {
                "summary": "首次使用memo0版自衍体",
                "content": f"于 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} 首次使用memo0记忆系统的自衍体系统。",
                "emotional_intensity": 0.5,
                "timestamp": datetime.now().isoformat()
            }
        ],
        "personality_traits": {
            "好奇": 0.8,
            "真诚": 0.9,
            "善良": 0.7,
            "喜欢学习": 0.85
        }
    }

    self_profile = {
        "self_memories": [
            {
                "summary": "系统启动",
                "content": f"memo0记忆系统的自衍体系统于 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} 启动。",
                "emotional_intensity": 0.3,
                "timestamp": datetime.now().isoformat()
            }
        ]
    }

    # 初始化自衍体系统
    try:
        print("\n🚀 正在初始化自衍体系统...")

        zyantine = Memo0EnhancedZyantineGenesis(
            user_profile_data=user_profile,
            self_profile_data=self_profile,
            config=config,
            session_id="default"
        )

        # 显示系统状态
        status = zyantine.get_system_status()
        print(f"\n📊 系统状态:")
        print(f"  会话ID: {status['session_id']}")
        print(f"  用户ID: {status.get('user_id', '未设置')}")
        print(f"  记忆系统: {status.get('memory_system', '未知')}")
        print(f"  聊天模型: {status.get('chat_model', '未知')}")

        # 显示记忆统计
        memory_stats = status.get('memory_stats', {})
        print(f"  记忆总数: {memory_stats.get('total_memories', 0)}")

        if 'memory_types' in memory_stats:
            print(f"  记忆类型分布:")
            for mem_type, count in memory_stats.get('memory_types', {}).items():
                print(f"    - {mem_type}: {count}")

        # 交互循环
        print(f"\n💬 开始交互 (输入 '退出'、'状态' 或 '帮助' 获取命令)")
        print("-" * 50)

        while True:
            try:
                user_input = input(f"\n你: ").strip()

                if not user_input:
                    continue

                if user_input.lower() in ['退出', 'exit', 'quit']:
                    print("\n👋 再见！正在保存记忆...")
                    zyantine.save_memory_system()
                    break

                elif user_input.lower() == '帮助':
                    print("\n📋 可用命令:")
                    print("  '状态' - 显示系统状态")
                    print("  '记忆统计' - 显示记忆系统统计")
                    print("  '记忆分析' - 分析记忆模式")
                    print("  '搜索 <关键词>' - 搜索记忆")
                    print("  '保存' - 手动保存记忆")
                    print("  '备份' - 创建记忆备份")
                    print("  '清除缓存' - 清除记忆缓存")
                    print("  '退出' - 退出程序")
                    continue

                elif user_input.lower() == '状态':
                    status = zyantine.get_system_status()
                    print(f"\n🔧 系统状态:")
                    print(f"  会话: {status['session_id']}")
                    print(f"  用户: {status.get('user_id', '未设置')}")
                    print(f"  向量状态: TR={status.get('desire_vectors', {}).get('TR', 0):.2f}, "
                          f"CS={status.get('desire_vectors', {}).get('CS', 0):.2f}, "
                          f"SA={status.get('desire_vectors', {}).get('SA', 0):.2f}")
                    print(f"  对话历史: {len(zyantine.conversation_history)} 条")
                    print(f"  组件加载: {status.get('components_loaded', 0)} 个")
                    continue

                elif user_input.lower() == '记忆统计':
                    stats = zyantine.get_memory_statistics()
                    print(f"\n📊 记忆系统统计:")
                    print(f"  总记忆数: {stats.get('total_memories', 0)}")

                    if 'memory_types' in stats:
                        print(f"  记忆类型分布:")
                        for mem_type, count in stats['memory_types'].items():
                            print(f"    - {mem_type}: {count}")

                    if 'top_tags' in stats and stats['top_tags']:
                        print(f"  热门标签:")
                        for tag, count in list(stats['top_tags'].items())[:5]:
                            print(f"    - {tag}: {count}")

                    if 'top_accessed_memories' in stats:
                        print(f"  最常访问的记忆: {len(stats['top_accessed_memories'])} 个")

                    if 'semantic_map_size' in stats:
                        print(f"  语义记忆地图大小: {stats['semantic_map_size']}")
                    continue

                elif user_input.lower() == '记忆分析':
                    analysis = zyantine.analyze_memory_patterns()
                    print(f"\n🧠 记忆模式分析:")

                    if 'type_analysis' in analysis:
                        print(f"  按类型分析:")
                        for mem_type, data in analysis['type_analysis'].items():
                            print(f"    - {mem_type}: {data.get('count', 0)}个记忆，"
                                  f"平均访问 {data.get('avg_access', 0):.1f}次，"
                                  f"情感强度 {data.get('avg_emotional_intensity', 0):.2f}")

                    if 'strategic_tags' in analysis:
                        print(f"  战略标签: {len(analysis['strategic_tags'])} 个")
                        if analysis['strategic_tags']:
                            print(f"    示例: {', '.join(analysis['strategic_tags'][:5])}")

                    if 'high_value_memories' in analysis:
                        print(f"  高价值记忆: {len(analysis['high_value_memories'])} 个")
                        if analysis['high_value_memories']:
                            print(f"    最高价值记忆: {analysis['high_value_memories'][0].get('memory_id', '未知')} "
                                  f"(分数: {analysis['high_value_memories'][0].get('strategic_score', 0)})")
                    continue

                elif user_input.lower().startswith('搜索 '):
                    query = user_input[3:].strip()
                    if query:
                        results = zyantine.search_memories(query, top_k=3)
                        print(f"\n🔍 搜索结果 ({len(results)} 个):")
                        for i, result in enumerate(results, 1):
                            print(f"{i}. 相似度: {result.get('similarity_score', 0):.3f}")
                            content = result.get('content', '')
                            if len(content) > 100:
                                content = content[:100] + "..."
                            print(f"   记忆: {content}")
                            print(f"   类型: {result.get('memory_type', '未知')}")
                    continue

                elif user_input.lower() == '保存':
                    success = zyantine.save_memory_system()
                    if success:
                        print("💾 记忆已保存")
                    else:
                        print("❌ 保存失败")
                    continue

                elif user_input.lower() == '备份':
                    backup_path = zyantine.backup_memory_system()
                    if backup_path:
                        print(f"💾 备份已创建: {backup_path}")
                    else:
                        print("❌ 备份失败")
                    continue

                elif user_input.lower() == '清除缓存':
                    success = zyantine.cleanup_memory(max_history=1000)
                    if success:
                        print("🧹 记忆缓存已清理")
                    else:
                        print("❌ 清理失败")
                    continue

                # 正常对话
                print(f"\n🤔 思考中...")
                response = zyantine.process_input(user_input)
                print(f"\n[自衍体] {response}")

            except KeyboardInterrupt:
                print("\n\n⏹️  中断请求，保存记忆中...")
                zyantine.save_memory_system()
                break
            except Exception as e:
                print(f"\n❌ 错误: {e}")
                import traceback
                traceback.print_exc()

    except Exception as e:
        print(f"\n❌ 系统初始化失败: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())