"""
配置管理器 - 统一管理所有配置
"""
import os
import json
from dataclasses import dataclass, field
from typing import Dict, Any, Optional,List
from enum import Enum
import yaml


class MemorySystemType(Enum):
    """记忆系统类型"""
    MEMO0 = "memo0"
    VECTOR_DB = "vector_db"
    HYBRID = "hybrid"


class ProcessingMode(Enum):
    """处理模式"""
    STANDARD = "standard"
    FAST = "fast"
    PRECISE = "precise"
    MEMORY_INTENSIVE = "memory_intensive"


@dataclass
class MemoryConfig:
    """记忆系统配置"""
    system_type: MemorySystemType = MemorySystemType.MEMO0
    max_memories: int = 10000
    retrieval_limit: int = 5
    similarity_threshold: float = 0.7
    enable_semantic_cache: bool = True
    cache_ttl: int = 300  # 秒
    backup_interval: int = 3600  # 秒
    backup_path: str = "./memory_backups"

    # Mem0特定配置
    memo0_config: Dict = field(default_factory=lambda: {
        "vector_store": {
            "provider": "milvus",
            "config": {
                "collection_name": "zyantine_memories",
                "url": "http://localhost:19530",
                "token": "",
            }
        },
        "llm": {
            "provider": "openai",
            "config": {
                "model": "gpt-4.1-nano-2025-04-14"
            }
        },
        "embedder": {
            "provider": "openai",
            "config": {
                "model": "text-embedding-3-large"
            }
        }
    })


@dataclass
class APIConfig:
    """API配置"""
    enabled: bool = True
    api_key: str = field(default_factory=lambda: os.getenv("OPENAI_API_KEY", "sk-wiHpoarpNTHaep0t54852a32A75a4d6986108b3f6eF7B7B9"))
    base_url: str = "https://openkey.cloud/v1"
    chat_model: str = "gpt-4.1-nano-2025-04-14"
    embedding_model: str = "text-embedding-3-large"
    timeout: int = 30
    max_retries: int = 3
    rate_limit: int = 60  # 每分钟请求数


@dataclass
class ProcessingConfig:
    """处理配置"""
    mode: ProcessingMode = ProcessingMode.STANDARD
    enable_stage_parallelism: bool = False
    max_conversation_history: int = 1000
    enable_real_time_analysis: bool = True
    batch_size: int = 10

    # 阶段配置
    stage_configs: Dict[str, Any] = field(default_factory=lambda: {
        "preprocess": {"enabled": True, "timeout": 5},
        "memory_retrieval": {"enabled": True, "cache_results": True},
        "desire_update": {"enabled": True, "update_frequency": "always"},
        "cognitive_flow": {"enabled": True, "max_iterations": 3},
        "reply_generation": {"enabled": True, "fallback_to_template": True},
        "protocol_review": {"enabled": True, "strict_mode": False}
    })


@dataclass
class ProtocolConfig:
    """协议配置"""
    enable_fact_check: bool = True
    enable_length_regulation: bool = True
    enable_expression_protocol: bool = True
    fact_check_strictness: float = 0.8  # 0-1
    max_response_length: int = 2000
    min_response_length: int = 50
    allow_uncertainty_phrases: bool = True
    required_tags: List[str] = field(default_factory=lambda: [])


@dataclass
class SystemConfig:
    """完整系统配置"""
    # 基本配置
    session_id: str = "default"
    user_id: str = "default_user"
    system_name: str = "ZyantineAI"
    version: str = "1.0.0"

    # 组件配置
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    api: APIConfig = field(default_factory=APIConfig)
    processing: ProcessingConfig = field(default_factory=ProcessingConfig)
    protocols: ProtocolConfig = field(default_factory=ProtocolConfig)

    # 性能配置
    enable_monitoring: bool = True
    enable_logging: bool = True
    log_level: str = "INFO"
    log_file: str = "./logs/zyantine.log"

    # 安全配置
    enable_encryption: bool = False
    enable_backup: bool = True
    enable_validation: bool = True

    @classmethod
    def from_file(cls, config_path: str) -> 'SystemConfig':
        """从配置文件加载配置"""
        if not os.path.exists(config_path):
            return cls()

        with open(config_path, 'r', encoding='utf-8') as f:
            if config_path.endswith('.json'):
                config_dict = json.load(f)
            elif config_path.endswith(('.yaml', '.yml')):
                config_dict = yaml.safe_load(f)
            else:
                raise ValueError(f"不支持的配置文件格式: {config_path}")

        return cls.from_dict(config_dict)

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'SystemConfig':
        """从字典创建配置"""
        # 创建配置实例
        config = cls()

        # 更新基本配置
        for key in ['session_id', 'user_id', 'system_name', 'version']:
            if key in config_dict:
                setattr(config, key, config_dict[key])

        # 递归更新嵌套配置
        for section in ['memory', 'api', 'processing', 'protocols']:
            if section in config_dict:
                section_config = getattr(config, section)
                section_dict = config_dict[section]

                for key, value in section_dict.items():
                    if hasattr(section_config, key):
                        setattr(section_config, key, value)

        return config

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'session_id': self.session_id,
            'user_id': self.user_id,
            'system_name': self.system_name,
            'version': self.version,
            'memory': self._dataclass_to_dict(self.memory),
            'api': self._dataclass_to_dict(self.api),
            'processing': self._dataclass_to_dict(self.processing),
            'protocols': self._dataclass_to_dict(self.protocols),
            'enable_monitoring': self.enable_monitoring,
            'enable_logging': self.enable_logging,
            'log_level': self.log_level,
            'log_file': self.log_file,
            'enable_encryption': self.enable_encryption,
            'enable_backup': self.enable_backup,
            'enable_validation': self.enable_validation
        }

    def _dataclass_to_dict(self, obj) -> Dict:
        """将数据类转换为字典"""
        if hasattr(obj, '__dataclass_fields__'):
            return {k: getattr(obj, k) for k in obj.__dataclass_fields__}
        return obj


class ConfigManager:
    """配置管理器单例"""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._config = None
            cls._instance._config_file = None
        return cls._instance

    def load(self, config_file: Optional[str] = None) -> SystemConfig:
        """加载配置"""
        if config_file:
            self._config_file = config_file
            self._config = SystemConfig.from_file(config_file)
        else:
            # 从默认位置加载
            default_paths = [
                "./config/zyantine_config.json",
                "./config/zyantine_config.yaml",
                "./config/config.json"
            ]

            for path in default_paths:
                if os.path.exists(path):
                    self._config_file = path
                    self._config = SystemConfig.from_file(path)
                    break
            else:
                # 使用默认配置
                self._config = SystemConfig()

        return self._config

    def get(self) -> SystemConfig:
        """获取当前配置"""
        if self._config is None:
            self.load()
        return self._config

    def update(self, updates: Dict[str, Any]) -> SystemConfig:
        """更新配置"""
        config_dict = self._config.to_dict()

        # 深度合并更新
        self._merge_dicts(config_dict, updates)

        # 重新创建配置对象
        self._config = SystemConfig.from_dict(config_dict)

        return self._config

    def save(self, config_file: Optional[str] = None) -> bool:
        """保存配置到文件"""
        if self._config is None:
            return False

        save_path = config_file or self._config_file or "./config/zyantine_config.json"

        try:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)

            with open(save_path, 'w', encoding='utf-8') as f:
                if save_path.endswith('.json'):
                    json.dump(self._config.to_dict(), f, ensure_ascii=False, indent=2)
                elif save_path.endswith(('.yaml', '.yml')):
                    yaml.dump(self._config.to_dict(), f, default_flow_style=False, allow_unicode=True)
                else:
                    json.dump(self._config.to_dict(), f, ensure_ascii=False, indent=2)

            return True
        except Exception as e:
            print(f"保存配置失败: {e}")
            return False

    def _merge_dicts(self, target: Dict, source: Dict):
        """深度合并字典"""
        for key, value in source.items():
            if key in target and isinstance(target[key], dict) and isinstance(value, dict):
                self._merge_dicts(target[key], value)
            else:
                target[key] = value