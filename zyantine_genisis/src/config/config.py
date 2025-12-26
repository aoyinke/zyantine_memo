import os
import json
from typing import Dict, Any, Optional
from dataclasses import dataclass, field
from enum import Enum


class MemoryProvider(Enum):
    """记忆存储提供者枚举"""
    MEMO0 = "memo0"
    FAISS = "faiss"


class VectorStoreProvider(Enum):
    """向量存储提供者枚举"""
    MILVUS = "milvus"
    FAISS = "faiss"


@dataclass
class Memo0Config:
    """memo0框架配置"""
    provider: str = "memo0"
    vector_store: Dict[str, Any] = field(default_factory=lambda: {
        "provider": "milvus",
        "config": {
            "collection_name": "zyantine_memories",
            "url": "http://localhost:19530",
            "token": "",
        }
    })
    llm: Dict[str, Any] = field(default_factory=lambda: {
        "provider": "openai",
        "config": {
            "openai_base_url": "https://openkey.cloud/v1",
            "api_key": ""  # 从环境变量或配置文件读取
        }
    })
    embedder: Dict[str, Any] = field(default_factory=lambda: {
        "provider": "openai",
        "config": {
            "model": "text-embedding-3-large",
            "openai_base_url": "https://openkey.cloud/v1",
            "api_key": ""  # 从环境变量或配置文件读取
        }
    })


class ZyantineConfig:
    """自衍体配置管理器（适配memo0记忆系统）"""

    def __init__(self, config_path: str = "./zyantine_config.json"):
        self.config_path = config_path
        self.config = self._load_config()

        # 打印配置摘要
        self._print_config_summary()

    def _load_config(self) -> Dict[str, Any]:
        """加载配置文件（优先环境变量，其次配置文件，最后默认值）"""
        default_config = {
            "api": {
                "openai_api_key": "",
                "openai_base_url": "https://openkey.cloud/v1",
                "embedding_model": "text-embedding-3-large",
                "embedding_dimensions": 1536,
                "chat_model": "gpt-4.1-nano-2025-04-14",
                "enabled": False
            },
            "memory": {
                "provider": "memo0",  # 使用memo0框架
                "memo0_config": Memo0Config().__dict__,
                "session_id": "default",
                "auto_backup": True,
                "backup_interval": 100,
                "user_id": "default_user"
            },
            "system": {
                "debug_mode": False,
                "log_level": "INFO",
                "save_snapshots": True,
                "enable_analytics": True
            },
            "identity": {
                "name": "自衍体",
                "username": "信息源标识符",
                "age": "19",
                "gender": "女",
                "identity": "强势的二号人物、军师",
                "personality_traits": ["活泼开朗", "谨慎细致", "科研精神", "好奇心强"]
            },
            "vector_store": {
                "provider": "milvus",  # 默认使用Milvus
                "milvus": {
                    "url": "http://localhost:19530",
                    "collection_name": "zyantine_memories",
                    "token": ""
                },
                "faiss": {
                    "index_type": "FlatL2",
                    "dimensions": 1536
                }
            }
        }

        # 从环境变量获取API密钥
        env_api_key = os.getenv("OPENAI_API_KEY")
        if env_api_key:
            default_config["api"]["openai_api_key"] = env_api_key
            default_config["api"]["enabled"] = True
        else:
            # 尝试从旧的环境变量获取
            env_api_key_old = os.getenv("OPENAI_API_KEY_OPENCLOUD")
            if env_api_key_old:
                default_config["api"]["openai_api_key"] = env_api_key_old
                default_config["api"]["enabled"] = True

        # 从环境变量获取基础URL
        env_base_url = os.getenv("OPENAI_BASE_URL")
        if env_base_url:
            default_config["api"]["openai_base_url"] = env_base_url

        # 从环境变量获取用户ID
        env_user_id = os.getenv("ZYANTINE_USER_ID")
        if env_user_id:
            default_config["memory"]["user_id"] = env_user_id

        # 从环境变量获取会话ID
        env_session_id = os.getenv("ZYANTINE_SESSION_ID")
        if env_session_id:
            default_config["memory"]["session_id"] = env_session_id

        # 从环境变量获取Milvus配置
        env_milvus_url = os.getenv("MILVUS_URL")
        if env_milvus_url:
            default_config["vector_store"]["milvus"]["url"] = env_milvus_url

        # 加载并合并用户配置文件
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    user_config = json.load(f)
                    # 递归合并配置
                    self._merge_configs(default_config, user_config)
                print(f"[配置] 已从配置文件加载: {self.config_path}")
            except Exception as e:
                print(f"[配置] 加载配置文件失败: {e}")

        # 更新memo0配置中的API密钥和基础URL
        self._update_memo0_config(default_config)

        return default_config

    def _update_memo0_config(self, config: Dict[str, Any]):
        """更新memo0配置中的API密钥和基础URL，保持配置一致性"""
        api_key = config["api"]["openai_api_key"]
        base_url = config["api"]["openai_base_url"]

        # 更新memo0配置
        if "memo0_config" in config["memory"]:
            config["memory"]["memo0_config"]["llm"]["config"]["api_key"] = api_key
            config["memory"]["memo0_config"]["llm"]["config"]["openai_base_url"] = base_url
            config["memory"]["memo0_config"]["embedder"]["config"]["api_key"] = api_key
            config["memory"]["memo0_config"]["embedder"]["config"]["openai_base_url"] = base_url

    def _merge_configs(self, base: Dict, update: Dict):
        """递归合并配置（用户配置覆盖默认配置，保留层级结构）"""
        for key, value in update.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._merge_configs(base[key], value)
            else:
                base[key] = value

    def get_openai_config(self) -> Dict:
        """获取OpenAI相关配置（API密钥、模型、基础URL等）"""
        api_config = self.config.get("api", {})

        return {
            "api_key": api_config.get("openai_api_key", ""),
            "base_url": api_config.get("openai_base_url", "https://openkey.cloud/v1"),
            "embedding_model": api_config.get("embedding_model", "text-embedding-3-large"),
            "embedding_dimensions": api_config.get("embedding_dimensions", 1536),
            "chat_model": api_config.get("chat_model", "gpt-5-nano"),
            "enabled": api_config.get("enabled", False)
        }

    def get_memory_config(self) -> Dict:
        """获取记忆系统相关配置"""
        memory_config = self.config.get("memory", {})

        return {
            "provider": memory_config.get("provider", "memo0"),
            "memo0_config": memory_config.get("memo0_config", {}),
            "session_id": memory_config.get("session_id", "default"),
            "user_id": memory_config.get("user_id", "default_user"),
            "auto_backup": memory_config.get("auto_backup", True),
            "backup_interval": memory_config.get("backup_interval", 100),
            "use_memo0": True  # 强制使用memo0
        }

    def get_memo0_config(self) -> Dict:
        """获取memo0框架专属配置（兼容动态加载场景）"""
        memory_config = self.config.get("memory", {})

        if "memo0_config" in memory_config:
            return memory_config["memo0_config"]

        # 若配置中无memo0_config，生成默认配置并同步API信息
        memo0_default = Memo0Config()
        api_config = self.get_openai_config()

        memo0_default.llm["config"]["api_key"] = api_config["api_key"]
        memo0_default.llm["config"]["openai_base_url"] = api_config["base_url"]
        memo0_default.embedder["config"]["api_key"] = api_config["api_key"]
        memo0_default.embedder["config"]["openai_base_url"] = api_config["base_url"]
        memo0_default.embedder["config"]["model"] = api_config["embedding_model"]

        return memo0_default.__dict__

    def get_vector_store_config(self) -> Dict:
        """获取向量存储配置（Milvus/FAISS）"""
        vector_config = self.config.get("vector_store", {})

        return {
            "provider": vector_config.get("provider", "milvus"),
            "milvus": vector_config.get("milvus", {
                "url": "http://localhost:19530",
                "collection_name": "zyantine_memories",
                "token": ""
            }),
            "faiss": vector_config.get("faiss", {
                "index_type": "FlatL2",
                "dimensions": 1536
            })
        }

    def get_system_config(self) -> Dict:
        """获取系统级配置（调试模式、日志级别等）"""
        system_config = self.config.get("system", {})

        return {
            "debug_mode": system_config.get("debug_mode", False),
            "log_level": system_config.get("log_level", "INFO"),
            "save_snapshots": system_config.get("save_snapshots", True),
            "enable_analytics": system_config.get("enable_analytics", True)
        }

    def get_identity_config(self) -> Dict:
        """获取自衍体身份配置（名称、性格、人设等）"""
        identity_config = self.config.get("identity", {})

        return {
            "name": identity_config.get("name", "自衍体"),
            "username": identity_config.get("username", "信息源标识符"),
            "age": identity_config.get("age", "19"),
            "gender": identity_config.get("gender", "女"),
            "identity": identity_config.get("identity", "强势的二号人物、军师"),
            "personality_traits": identity_config.get(
                "personality_traits",
                ["活泼开朗", "谨慎细致", "科研精神", "好奇心强"]
            )
        }

    def save_config(self, file_path: Optional[str] = None) -> bool:
        """保存当前配置到文件（默认保存到初始化时指定的路径）"""
        if file_path is None:
            file_path = self.config_path

        try:
            # 确保目录存在
            os.makedirs(os.path.dirname(file_path), exist_ok=True)

            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(self.config, f, ensure_ascii=False, indent=2)

            print(f"[配置] 配置已保存到: {file_path}")
            return True
        except Exception as e:
            print(f"[配置] 保存配置失败: {e}")
            return False

    def update_config(self, section: str, key: str, value: Any) -> bool:
        """更新指定配置项（支持新增section/key）"""
        if section in self.config:
            if isinstance(self.config[section], dict):
                self.config[section][key] = value
                return True
        else:
            self.config[section] = {key: value}
            return True

        return False

    def validate_config(self) -> Dict[str, bool]:
        """验证配置的完整性和有效性"""
        validation_results = {}

        # 验证API配置
        api_config = self.get_openai_config()
        validation_results["api_key_set"] = bool(api_config["api_key"])
        validation_results["api_enabled"] = api_config["enabled"]

        # 验证记忆配置
        memory_config = self.get_memory_config()
        validation_results["memory_provider"] = memory_config["provider"] == "memo0"
        validation_results["memo0_config"] = "memo0_config" in self.config["memory"]

        # 验证向量存储配置
        vector_config = self.get_vector_store_config()
        validation_results["vector_store_provider"] = vector_config["provider"] in ["milvus", "faiss"]

        return validation_results

    def _print_config_summary(self):
        """打印配置摘要（便于调试和确认配置状态）"""
        validation = self.validate_config()

        print("\n" + "=" * 50)
        print("自衍体配置摘要")
        print("=" * 50)

        # API配置
        api_config = self.get_openai_config()
        print(f"📡 API配置:")
        print(f"  模型: {api_config['chat_model']}")
        print(f"  嵌入: {api_config['embedding_model']} ({api_config['embedding_dimensions']}维)")
        print(f"  状态: {'✅ 已启用' if validation['api_key_set'] else '❌ 未配置'}")

        # 记忆配置
        memory_config = self.get_memory_config()
        print(f"🧠 记忆配置:")
        print(f"  提供者: {memory_config['provider']}")
        print(f"  用户ID: {memory_config['user_id']}")
        print(f"  会话ID: {memory_config['session_id']}")
        print(f"  状态: {'✅ 已配置' if validation['memory_provider'] else '❌ 配置错误'}")

        # 向量存储
        vector_config = self.get_vector_store_config()
        print(f"🗃️  向量存储:")
        print(f"  提供者: {vector_config['provider']}")
        if vector_config['provider'] == 'milvus':
            print(f"  URL: {vector_config['milvus']['url']}")

        # 系统配置
        system_config = self.get_system_config()
        print(f"⚙️  系统配置:")
        print(f"  调试模式: {'开启' if system_config['debug_mode'] else '关闭'}")
        print(f"  日志级别: {system_config['log_level']}")

        print("=" * 50 + "\n")

    def generate_default_config(self) -> Dict[str, Any]:
        """生成默认配置模板（用于初始化配置文件）"""
        return {
            "api": {
                "openai_api_key": "YOUR_API_KEY_HERE",
                "openai_base_url": "https://openkey.cloud/v1",
                "embedding_model": "text-embedding-3-large",
                "embedding_dimensions": 1536,
                "chat_model": "gpt-5-nano",
                "enabled": True
            },
            "memory": {
                "provider": "memo0",
                "memo0_config": {
                    "provider": "memo0",
                    "vector_store": {
                        "provider": "milvus",
                        "config": {
                            "collection_name": "zyantine_memories",
                            "url": "http://localhost:19530",
                            "token": ""
                        }
                    },
                    "llm": {
                        "provider": "openai",
                        "config": {
                            "openai_base_url": "https://openkey.cloud/v1",
                            "api_key": "YOUR_API_KEY_HERE"
                        }
                    },
                    "embedder": {
                        "provider": "openai",
                        "config": {
                            "model": "text-embedding-3-large",
                            "openai_base_url": "https://openkey.cloud/v1",
                            "api_key": "YOUR_API_KEY_HERE"
                        }
                    }
                },
                "session_id": "default",
                "user_id": "default_user",
                "auto_backup": True,
                "backup_interval": 100
            },
            "system": {
                "debug_mode": False,
                "log_level": "INFO",
                "save_snapshots": True,
                "enable_analytics": True
            },
            "identity": {
                "name": "自衍体",
                "username": "信息源标识符",
                "age": "19",
                "gender": "女",
                "identity": "强势的二号人物、军师",
                "personality_traits": ["活泼开朗", "谨慎细致", "科研精神", "好奇心强"]
            },
            "vector_store": {
                "provider": "milvus",
                "milvus": {
                    "url": "http://localhost:19530",
                    "collection_name": "zyantine_memories",
                    "token": ""
                },
                "faiss": {
                    "index_type": "FlatL2",
                    "dimensions": 1536
                }
            }
        }

    @classmethod
    def create_default_config(cls, file_path: str = "./zyantine_config.json") -> bool:
        """创建默认配置文件（类方法，无需实例化即可调用）"""
        try:
            config = cls()
            default_config = config.generate_default_config()

            # 确保目录存在
            os.makedirs(os.path.dirname(file_path), exist_ok=True)

            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(default_config, f, ensure_ascii=False, indent=2)

            print(f"[配置] 默认配置文件已创建: {file_path}")
            print("[配置] 请编辑此文件，填入您的API密钥和其他配置")
            return True
        except Exception as e:
            print(f"[配置] 创建默认配置失败: {e}")
            return False


