# ============ 配置管理器 ============
import os
import json
from typing import Dict, Any, Optional
from ..memory.memory_store import OpenAIEmbeddingService

class ZyantineConfig:
    """自衍体配置管理器"""

    def __init__(self, config_path: str = "./zyantine_config.json"):
        self.config_path = config_path
        self.config = self._load_config()

    def _load_config(self) -> Dict[str, Any]:
        """加载配置文件"""
        default_config = {
            "api": {
                "openai_api_key": "",
                "openai_base_url": "https://openkey.cloud/v1",
                "embedding_model": "text-embedding-3-small",
                "embedding_dimensions": 256,
                "chat_model": "gpt-5-nano-2025-08-07",
                "enabled": False
            },
            "memory": {
                "storage_path": "./zyantine_memory",
                "session_id": "default",
                "auto_backup": True,
                "backup_interval": 100,
                "use_faiss": True,
                "faiss_index_type": "IVFFlat"
            },
            "system": {
                "debug_mode": False,
                "log_level": "INFO",
                "save_snapshots": True
            },
            "identity": {
                "name": "自衍体",
                "age": "19",
                "gender": "女",
                "personality_traits": ["活泼开朗", "谨慎细致", "科研精神", "好奇心强"]
            }
        }

        # 从环境变量获取API密钥
        env_api_key = os.getenv("OPENAI_API_KEY_OPENCLOUD")
        if env_api_key:
            default_config["api"]["openai_api_key"] = env_api_key
            default_config["api"]["enabled"] = True

        # 从环境变量获取基础URL
        env_base_url = os.getenv("OPENAI_BASE_URL")
        if env_base_url:
            default_config["api"]["openai_base_url"] = env_base_url

        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    user_config = json.load(f)
                    # 合并配置
                    self._merge_configs(default_config, user_config)
            except Exception as e:
                print(f"[配置] 加载配置文件失败: {e}")

        return default_config

    def _merge_configs(self, base: Dict, update: Dict):
        """递归合并配置"""
        for key, value in update.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._merge_configs(base[key], value)
            else:
                base[key] = value

    def get_openai_config(self) -> Dict:
        """获取OpenAI配置"""
        api_config = self.config.get("api", {})

        return {
            "api_key": api_config.get("openai_api_key", ""),
            "base_url": api_config.get("openai_base_url", "https://openkey.cloud/v1"),
            "embedding_model": api_config.get("embedding_model", "text-embedding-3-small"),
            "embedding_dimensions": api_config.get("embedding_dimensions", 256),
            "chat_model": api_config.get("chat_model", "gpt-5-nano-2025-08-07"),
            "enabled": api_config.get("enabled", False)
        }

    def get_memory_config(self) -> Dict:
        """获取记忆配置"""
        memory_config = self.config.get("memory", {})

        return {
            "storage_path": memory_config.get("storage_path", "./zyantine_memory"),
            "session_id": memory_config.get("session_id", "default"),
            "auto_backup": memory_config.get("auto_backup", True),
            "use_faiss": memory_config.get("use_faiss", True),
            "faiss_index_type": memory_config.get("faiss_index_type", "IVFFlat")
        }

    def get_embedding_service(self) -> Optional[OpenAIEmbeddingService]:
        """获取嵌入服务实例"""
        api_config = self.get_openai_config()

        if not api_config["api_key"]:
            print("[配置] 警告：未设置OpenAI API密钥")
            return None

        if not api_config["enabled"]:
            print("[配置] OpenAI API未启用")
            return None

        try:
            service = OpenAIEmbeddingService(
                api_key=api_config["api_key"],
                base_url=api_config["base_url"],
                model=api_config["embedding_model"],
                default_dimensions=api_config["embedding_dimensions"]
            )

            # 测试连接
            if service.test_connection():
                print(f"[配置] OpenAI嵌入服务初始化成功")
                return service
            else:
                print(f"[配置] OpenAI嵌入服务连接测试失败")
                return None

        except Exception as e:
            print(f"[配置] 创建嵌入服务失败: {e}")
            return None