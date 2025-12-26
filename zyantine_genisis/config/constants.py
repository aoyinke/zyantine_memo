"""
常量定义
"""
from enum import Enum

class ProcessingMode(Enum):
    """处理模式"""
    STANDARD = "standard"
    ENHANCED = "enhanced"
    LITE = "lite"

class MemorySystemType(Enum):
    """记忆系统类型"""
    BASIC = "basic"
    ADVANCED = "advanced"
    PERSISTENT = "persistent"

class LogLevel(Enum):
    """日志级别"""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"

# API相关常量
DEFAULT_API_TIMEOUT = 30
DEFAULT_MAX_TOKENS = 2000
DEFAULT_TEMPERATURE = 0.7

# 文件路径常量
DEFAULT_CONFIG_PATH = "config/default_config.yaml"
DEFAULT_LOG_DIR = "logs"
DEFAULT_DATA_DIR = "data"

# 系统常量
VERSION = "1.0.0"
AUTHOR = "自衍体项目组"
DESCRIPTION = "自衍体AI系统 - 智能对话与记忆系统"