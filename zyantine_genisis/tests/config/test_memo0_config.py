"""
测试 memo0 配置是否从 llm_config.json 正确读取
"""
import sys
import os

# 添加项目根目录到 Python 路径
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, project_root)

from config.config_manager import ConfigManager


def test_memo0_config_loading():
    """测试 memo0 配置加载"""
    print("="*60)
    print("测试: memo0 配置从 llm_config.json 加载")
    print("="*60)

    try:
        # 加载配置
        config_manager = ConfigManager()
        config = config_manager.load("./config/llm_config.json")

        # 检查 memo0_config 是否存在
        assert hasattr(config.memory, 'memo0_config'), "memory 配置中缺少 memo0_config"
        print("✓ memo0_config 存在")

        # 检查 vector_store 配置
        assert "vector_store" in config.memory.memo0_config, "memo0_config 中缺少 vector_store"
        print("✓ vector_store 配置存在")

        vector_store = config.memory.memo0_config["vector_store"]
        assert vector_store["provider"] == "milvus", f"vector_store provider 应该是 milvus，实际是 {vector_store['provider']}"
        print(f"✓ vector_store provider: {vector_store['provider']}")

        # 检查 llm 配置
        assert "llm" in config.memory.memo0_config, "memo0_config 中缺少 llm"
        print("✓ llm 配置存在")

        llm_config = config.memory.memo0_config["llm"]
        assert llm_config["provider"] == "openai", f"llm provider 应该是 openai，实际是 {llm_config['provider']}"
        print(f"✓ llm provider: {llm_config['provider']}")

        llm_model_config = llm_config["config"]
        assert "model" in llm_model_config, "llm config 中缺少 model"
        assert llm_model_config["model"] == "gpt-5-nano-2025-08-07", f"llm model 应该是 gpt-5-nano-2025-08-07，实际是 {llm_model_config['model']}"
        print(f"✓ llm model: {llm_model_config['model']}")

        assert "api_key" in llm_model_config, "llm config 中缺少 api_key"
        print(f"✓ llm api_key: {llm_model_config['api_key'][:20]}...")

        assert "base_url" in llm_model_config, "llm config 中缺少 base_url"
        assert llm_model_config["base_url"] == "https://openkey.cloud/v1", f"llm base_url 应该是 https://openkey.cloud/v1，实际是 {llm_model_config['base_url']}"
        print(f"✓ llm base_url: {llm_model_config['base_url']}")

        # 检查 embedder 配置
        assert "embedder" in config.memory.memo0_config, "memo0_config 中缺少 embedder"
        print("✓ embedder 配置存在")

        embedder_config = config.memory.memo0_config["embedder"]
        assert embedder_config["provider"] == "openai", f"embedder provider 应该是 openai，实际是 {embedder_config['provider']}"
        print(f"✓ embedder provider: {embedder_config['provider']}")

        embedder_model_config = embedder_config["config"]
        assert "model" in embedder_model_config, "embedder config 中缺少 model"
        assert embedder_model_config["model"] == "text-embedding-3-large", f"embedder model 应该是 text-embedding-3-large，实际是 {embedder_model_config['model']}"
        print(f"✓ embedder model: {embedder_model_config['model']}")

        assert "api_key" in embedder_model_config, "embedder config 中缺少 api_key"
        print(f"✓ embedder api_key: {embedder_model_config['api_key'][:20]}...")

        assert "base_url" in embedder_model_config, "embedder config 中缺少 base_url"
        assert embedder_model_config["base_url"] == "https://openkey.cloud/v1", f"embedder base_url 应该是 https://openkey.cloud/v1，实际是 {embedder_model_config['base_url']}"
        print(f"✓ embedder base_url: {embedder_model_config['base_url']}")

        print("\n" + "="*60)
        print("✓ 所有测试通过！memo0 配置正确从 llm_config.json 加载")
        print("="*60)

        return True

    except Exception as e:
        print(f"\n✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_memory_system_initialization():
    """测试记忆系统初始化是否使用配置"""
    print("\n" + "="*60)
    print("测试: 记忆系统初始化使用配置")
    print("="*60)

    try:
        from memory.memory_store import ZyantineMemorySystem

        # 加载配置
        config_manager = ConfigManager()
        config = config_manager.load("./config/llm_config.json")

        # 创建记忆系统实例（不传递 base_url 和 api_key，应该从配置读取）
        memory_system = ZyantineMemorySystem(
            user_id="test_user",
            session_id="test_session"
        )

        # 检查配置是否正确应用
        assert memory_system.base_url == "https://openkey.cloud/v1", f"base_url 应该是 https://openkey.cloud/v1，实际是 {memory_system.base_url}"
        print(f"✓ base_url 正确: {memory_system.base_url}")

        assert memory_system.api_key == "sk-wiHpoarpNTHaep0t54852a32A75a4d6986108b3f6eF7B7B9", f"api_key 不匹配"
        print(f"✓ api_key 正确: {memory_system.api_key[:20]}...")

        # 检查 memo0_config 是否正确设置
        assert hasattr(memory_system, 'memo0_config'), "记忆系统缺少 memo0_config 属性"
        print("✓ memo0_config 属性存在")

        assert memory_system.memo0_config == config.memory.memo0_config, "memo0_config 与配置文件不一致"
        print("✓ memo0_config 与配置文件一致")

        print("\n" + "="*60)
        print("✓ 记忆系统初始化测试通过！")
        print("="*60)

        return True

    except Exception as e:
        print(f"\n✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = True

    # 运行测试
    success = test_memo0_config_loading() and success
    success = test_memory_system_initialization() and success

    # 退出
    sys.exit(0 if success else 1)
