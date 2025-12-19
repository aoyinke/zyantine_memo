"""
工具函数集
"""

import json
import hashlib
from datetime import datetime
from typing import Any, Dict, List
import random


def generate_id(prefix: str = "ID", length: int = 8) -> str:
    """生成唯一ID"""
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S%f")
    random_str = ''.join(random.choices('0123456789ABCDEF', k=4))
    return f"{prefix}_{timestamp}_{random_str}"


def calculate_hash(data: Any) -> str:
    """计算数据的哈希值"""
    if isinstance(data, dict):
        data_str = json.dumps(data, sort_keys=True, ensure_ascii=False)
    else:
        data_str = str(data)

    return hashlib.sha256(data_str.encode()).hexdigest()


def safe_get(data: Dict, keys: List[str], default: Any = None) -> Any:
    """安全获取嵌套字典的值"""
    current = data
    for key in keys:
        if isinstance(current, dict) and key in current:
            current = current[key]
        else:
            return default
    return current


def format_timestamp(timestamp: datetime = None) -> str:
    """格式化时间戳"""
    if timestamp is None:
        timestamp = datetime.now()
    return timestamp.isoformat()