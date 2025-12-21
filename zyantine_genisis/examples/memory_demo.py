# start_zyantine_openai.py
# !/usr/bin/env python3
"""
自衍体-起源 (OpenAI嵌入版) 启动脚本
"""

import os
import sys
import json
from datetime import datetime


def setup_environment():
    """设置环境变量"""
    # 检查环境变量
    required_env_vars = ["OPENAI_API_KEY_OPENCLOUD"]

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
                    os.environ["OPENAI_API_KEY_OPENCLOUD"] = api_key
                    print("✅ 从配置文件读取API密钥")
                    return True
            except:
                pass

        print("\n请设置环境变量:")
        print("  export OPENAI_API_KEY_OPENCLOUD='sk-...'")
        print("  或编辑配置文件: zyantine_config.json")

        # 询问用户是否要输入
        response = input("\n是否要现在输入API密钥？(y/N): ")
        if response.lower() == 'y':
            api_key = input("请输入OpenAI API密钥: ").strip()
            if api_key:
                os.environ["OPENAI_API_KEY_OPENCLOUD"] = api_key
                print("✅ API密钥已设置")

                # 保存到配置文件
                config = {
                    "api": {
                        "openai_api_key": api_key,
                        "openai_base_url": "https://openkey.cloud/v1",
                        "embedding_model": "text-embedding-3-small",
                        "embedding_dimensions": 256,
                        "chat_model": "gpt-4",
                        "enabled": True
                    }
                }

                with open(config_file, 'w', encoding='utf-8') as f:
                    json.dump(config, f, ensure_ascii=False, indent=2)

                print(f"✅ 配置已保存到: {config_file}")
                return True

        return False

    return True


def test_openai_connection():
    """测试OpenAI连接"""
    print("\n🔗 测试OpenAI连接...")

    try:
        from openai import OpenAI

        api_key = os.getenv("OPENAI_API_KEY_OPENCLOUD")
        base_url = os.getenv("OPENAI_BASE_URL", "https://openkey.cloud/v1")

        client = OpenAI(
            api_key=api_key,
            base_url=base_url
        )

        # 测试嵌入API
        response = client.embeddings.create(
            input="测试连接",
            model="text-embedding-3-small",
            dimensions=256
        )

        embedding_vector = response.data[0].embedding
        print(f"✅ 连接测试成功，向量维度: {len(embedding_vector)}")
        return True

    except Exception as e:
        print(f"❌ 连接测试失败: {e}")
        return False


def main():
    """主函数"""
    print("=" * 70)
    print("自衍体-起源 V2.0 (OpenAI嵌入版)")
    print(f"启动时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    # 设置环境
    if not setup_environment():
        print("\n❌ 环境设置失败，程序退出")
        return 1

    # 测试连接
    if not test_openai_connection():
        response = input("\n连接测试失败，是否继续？(y/N): ")
        if response.lower() != 'y':
            print("程序退出")
            return 1

    # 导入需要的模块
    try:
        # 确保当前目录在Python路径中
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

        from src.system.zyantine_memory import OpenAIEnhancedZyantineGenesisV2
        from src.config.config import ZyantineConfig

    except ImportError as e:
        print(f"\n❌ 导入模块失败: {e}")
        print("请确保所有依赖已安装:")
        print("  pip install openai faiss-cpu numpy")
        return 1

    # 加载配置
    config = ZyantineConfig()

    # 初始化用户数据
    user_profile = {
        "memories": [
            {
                "summary": "首次使用自衍体",
                "content": f"于 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} 首次使用OpenAI嵌入版自衍体系统。",
                "emotional_intensity": 0.5,
                "timestamp": datetime.now().isoformat()
            }
        ]
    }

    self_profile = {
        "self_memories": [
            {
                "summary": "系统启动",
                "content": f"OpenAI嵌入版自衍体系统于 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} 启动。",
                "emotional_intensity": 0.3,
                "timestamp": datetime.now().isoformat()
            }
        ]
    }

    # 初始化自衍体系统
    try:
        print("\n🚀 正在初始化自衍体系统...")

        zyantine = OpenAIEnhancedZyantineGenesisV2(
            user_profile_data=user_profile,
            self_profile_data=self_profile,
            config=config,
            session_id="default"
        )

        # 显示系统状态
        status = zyantine.get_system_status()
        print(f"\n📊 系统状态:")
        print(f"  会话ID: {status['session_id']}")
        print(f"  嵌入模型: {status['embedding_model']} ({status['embedding_dimensions']}维)")
        print(f"  聊天模型: {status['chat_model']}")
        print(f"  记忆总数: {status['memory_stats']['faiss_memories']}")

        # 显示嵌入服务统计
        if hasattr(zyantine, 'memory_system') and hasattr(zyantine.memory_system, 'vector_store'):
            embed_service = zyantine.memory_system.vector_store.embedding_service
            if embed_service:
                embed_stats = embed_service.get_statistics()
                print(f"  嵌入请求: {embed_stats['total_requests']} (成功率: {embed_stats['success_rate']:.1f}%)")

        # 交互循环
        print(f"\n💬 开始交互 (输入 '退出'、'状态' 或 '帮助' 获取命令)")
        print("-" * 50)

        while True:
            try:
                user_input = input(f"\n[自衍体] 你: ").strip()

                if not user_input:
                    continue

                if user_input.lower() in ['退出', 'exit', 'quit']:
                    print("\n👋 再见！正在保存记忆...")
                    zyantine.save_memory_system()
                    break

                elif user_input.lower() == '帮助':
                    print("\n📋 可用命令:")
                    print("  '状态' - 显示系统状态")
                    print("  '记忆洞察' - 显示记忆系统洞察")
                    print("  '搜索 <关键词>' - 搜索记忆")
                    print("  '保存' - 手动保存记忆")
                    print("  '备份' - 创建记忆备份")
                    print("  '清除缓存' - 清除嵌入缓存")
                    print("  '退出' - 退出程序")
                    continue

                elif user_input.lower() == '状态':
                    status = zyantine.get_system_status()
                    print(f"\n🔧 系统状态:")
                    print(f"  会话: {status['session_id']}")
                    print(f"  向量状态: TR={status.get('desire_vectors', {}).get('TR', 0):.2f}, "
                          f"CS={status.get('desire_vectors', {}).get('CS', 0):.2f}, "
                          f"SA={status.get('desire_vectors', {}).get('SA', 0):.2f}")
                    print(f"  对话历史: {len(zyantine.conversation_history)} 条")
                    continue

                elif user_input.lower() == '记忆洞察':
                    insights = zyantine.get_memory_insights()
                    print(f"\n🧠 记忆系统洞察:")
                    print(f"  总记忆数: {insights['total_memories']}")
                    print(f"  对话数: {insights['total_conversations']}")
                    print(f"  向量维度: {insights['vector_dimension']}")

                    if insights.get('recent_patterns'):
                        print(f"  最近模式: {insights['recent_patterns'][0]['pattern']}")

                    if insights.get('common_tags'):
                        common_tags = list(insights['common_tags'].items())[:3]
                        print(f"  常见标签: {', '.join([f'{tag}({count})' for tag, count in common_tags])}")
                    continue

                elif user_input.lower().startswith('搜索 '):
                    query = user_input[3:].strip()
                    if query:
                        results = zyantine.search_memories(query, top_k=3)
                        print(f"\n🔍 搜索结果 ({len(results)} 个):")
                        for i, result in enumerate(results, 1):
                            print(f"{i}. 相似度: {result['similarity']:.3f}")
                            print(f"   记忆: {result['text'][:100]}...")
                    continue

                elif user_input.lower() == '保存':
                    zyantine.save_memory_system()
                    print("💾 记忆已保存")
                    continue

                elif user_input.lower() == '备份':
                    backup_path = zyantine.backup_memory_system()
                    print(f"💾 备份已创建: {backup_path}")
                    continue

                elif user_input.lower() == '清除缓存':
                    if hasattr(zyantine, 'memory_system') and hasattr(zyantine.memory_system, 'vector_store'):
                        embed_service = zyantine.memory_system.vector_store.embedding_service
                        if embed_service:
                            embed_service.clear_cache()
                            print("🧹 嵌入缓存已清除")
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