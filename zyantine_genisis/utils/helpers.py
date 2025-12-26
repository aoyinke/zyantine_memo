"""
辅助函数模块 - 通用工具函数
"""
import os
import sys
import json
import hashlib
import random
import string
import time
import inspect
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union, Callable
from pathlib import Path
from functools import wraps
import asyncio


class TextHelper:
    """文本处理辅助类"""

    @staticmethod
    def truncate_text(text: str, max_length: int, suffix: str = "...") -> str:
        """截断文本"""
        if not text or len(text) <= max_length:
            return text

        if max_length <= len(suffix):
            return suffix

        return text[:max_length - len(suffix)] + suffix

    @staticmethod
    def split_into_chunks(text: str, chunk_size: int) -> List[str]:
        """将文本分割成块"""
        if not text:
            return []

        chunks = []
        for i in range(0, len(text), chunk_size):
            chunks.append(text[i:i + chunk_size])

        return chunks

    @staticmethod
    def count_words(text: str) -> int:
        """统计单词数（中文按字，英文按单词）"""
        if not text:
            return 0

        # 移除标点符号
        import re
        text = re.sub(r'[^\w\s]', ' ', text)

        # 分割单词
        words = text.split()

        # 处理中文字符
        chinese_chars = sum(1 for char in text if '\u4e00' <= char <= '\u9fff')

        return len(words) + chinese_chars

    @staticmethod
    def extract_keywords(text: str, top_n: int = 10) -> List[str]:
        """提取关键词（简单实现）"""
        if not text:
            return []

        # 移除停用词
        stop_words = {'的', '了', '在', '是', '我', '有', '和', '就', '不', '人', '都',
                      '一', '一个', '上', '也', '很', '到', '说', '要', '去', '你',
                      '会', '着', '没有', '看', '好', '自己', '这'}

        # 分割单词
        import re
        words = re.findall(r'[\w\u4e00-\u9fff]+', text.lower())

        # 统计词频
        word_count = {}
        for word in words:
            if word not in stop_words and len(word) > 1:
                word_count[word] = word_count.get(word, 0) + 1

        # 排序并返回前N个
        sorted_words = sorted(word_count.items(), key=lambda x: x[1], reverse=True)
        return [word for word, count in sorted_words[:top_n]]

    @staticmethod
    def normalize_text(text: str) -> str:
        """规范化文本"""
        if not text:
            return ""

        # 转换为小写
        normalized = text.lower()

        # 移除多余空格
        normalized = ' '.join(normalized.split())

        # 标准化标点
        import re
        normalized = re.sub(r'[。，；：？！]', '.', normalized)
        normalized = re.sub(r'\.+', '.', normalized)

        return normalized.strip()


class FileHelper:
    """文件操作辅助类"""

    @staticmethod
    def ensure_directory(directory: Union[str, Path]) -> Path:
        """确保目录存在"""
        if isinstance(directory, str):
            directory = Path(directory)

        directory.mkdir(parents=True, exist_ok=True)
        return directory

    @staticmethod
    def read_json(filepath: Union[str, Path], default: Any = None) -> Any:
        """读取JSON文件"""
        try:
            filepath = Path(filepath)
            if not filepath.exists():
                return default

            with open(filepath, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            return default

    @staticmethod
    def write_json(filepath: Union[str, Path], data: Any, indent: int = 2) -> bool:
        """写入JSON文件"""
        try:
            filepath = Path(filepath)

            # 确保目录存在
            filepath.parent.mkdir(parents=True, exist_ok=True)

            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=indent)
            return True
        except Exception:
            return False

    @staticmethod
    def read_lines(filepath: Union[str, Path], encoding: str = 'utf-8') -> List[str]:
        """读取文件所有行"""
        try:
            filepath = Path(filepath)
            with open(filepath, 'r', encoding=encoding) as f:
                return [line.strip() for line in f if line.strip()]
        except Exception:
            return []

    @staticmethod
    def write_lines(filepath: Union[str, Path], lines: List[str],
                    encoding: str = 'utf-8') -> bool:
        """写入文件行"""
        try:
            filepath = Path(filepath)

            # 确保目录存在
            filepath.parent.mkdir(parents=True, exist_ok=True)

            with open(filepath, 'w', encoding=encoding) as f:
                for line in lines:
                    f.write(line + '\n')
            return True
        except Exception:
            return False

    @staticmethod
    def get_file_hash(filepath: Union[str, Path], algorithm: str = 'md5') -> Optional[str]:
        """计算文件哈希值"""
        try:
            filepath = Path(filepath)
            if not filepath.exists() or not filepath.is_file():
                return None

            hash_func = hashlib.new(algorithm)

            with open(filepath, 'rb') as f:
                for chunk in iter(lambda: f.read(4096), b''):
                    hash_func.update(chunk)

            return hash_func.hexdigest()
        except Exception:
            return None


class TimeHelper:
    """时间处理辅助类"""

    @staticmethod
    def get_timestamp(fmt: str = "%Y%m%d_%H%M%S") -> str:
        """获取时间戳字符串"""
        return datetime.now().strftime(fmt)

    @staticmethod
    def format_duration(seconds: float) -> str:
        """格式化持续时间"""
        if seconds < 1:
            return f"{seconds * 1000:.1f}ms"

        minutes, seconds = divmod(int(seconds), 60)
        hours, minutes = divmod(minutes, 60)
        days, hours = divmod(hours, 24)

        parts = []
        if days > 0:
            parts.append(f"{days}d")
        if hours > 0:
            parts.append(f"{hours}h")
        if minutes > 0:
            parts.append(f"{minutes}m")
        if seconds > 0 or not parts:
            parts.append(f"{seconds}s")

        return ' '.join(parts)

    @staticmethod
    def is_expired(timestamp: datetime, ttl: timedelta) -> bool:
        """检查时间戳是否过期"""
        return datetime.now() - timestamp > ttl

    @staticmethod
    def parse_time_string(time_str: str) -> Optional[datetime]:
        """解析时间字符串"""
        formats = [
            "%Y-%m-%d %H:%M:%S",
            "%Y-%m-%d",
            "%H:%M:%S",
            "%Y%m%d_%H%M%S",
            "%Y/%m/%d %H:%M:%S"
        ]

        for fmt in formats:
            try:
                return datetime.strptime(time_str, fmt)
            except ValueError:
                continue

        return None


class SecurityHelper:
    """安全辅助类"""

    @staticmethod
    def generate_token(length: int = 32) -> str:
        """生成随机令牌"""
        chars = string.ascii_letters + string.digits
        return ''.join(random.choices(chars, k=length))

    @staticmethod
    def generate_api_key(prefix: str = "zy_") -> str:
        """生成API密钥"""
        timestamp = int(time.time())
        random_part = ''.join(random.choices(string.ascii_lowercase + string.digits, k=16))
        return f"{prefix}{timestamp}_{random_part}"

    @staticmethod
    def hash_password(password: str, salt: Optional[str] = None) -> Dict[str, str]:
        """哈希密码"""
        if salt is None:
            salt = SecurityHelper.generate_token(16)

        # 使用sha256
        hash_obj = hashlib.sha256()
        hash_obj.update((password + salt).encode('utf-8'))
        hashed = hash_obj.hexdigest()

        return {
            "hash": hashed,
            "salt": salt
        }

    @staticmethod
    def verify_password(password: str, hashed_password: str, salt: str) -> bool:
        """验证密码"""
        new_hash = SecurityHelper.hash_password(password, salt)["hash"]
        return new_hash == hashed_password


class PerformanceHelper:
    """性能辅助类"""

    @staticmethod
    def time_function(func: Callable) -> Callable:
        """函数计时装饰器"""

        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            result = func(*args, **kwargs)
            elapsed = time.time() - start_time

            print(f"函数 {func.__name__} 执行时间: {elapsed:.3f}秒")
            return result

        return wrapper

    @staticmethod
    def async_time_function(func: Callable) -> Callable:
        """异步函数计时装饰器"""

        @wraps(func)
        async def wrapper(*args, **kwargs):
            start_time = time.time()
            result = await func(*args, **kwargs)
            elapsed = time.time() - start_time

            print(f"异步函数 {func.__name__} 执行时间: {elapsed:.3f}秒")
            return result

        return wrapper

    @staticmethod
    def memoize(func: Callable) -> Callable:
        """记忆化装饰器"""
        cache = {}

        @wraps(func)
        def wrapper(*args, **kwargs):
            # 创建缓存键
            key = (args, frozenset(kwargs.items()))

            if key not in cache:
                cache[key] = func(*args, **kwargs)

            return cache[key]

        return wrapper


class AsyncHelper:
    """异步辅助类"""

    @staticmethod
    async def run_with_timeout(coro, timeout: float, default: Any = None) -> Any:
        """运行协程并设置超时"""
        try:
            return await asyncio.wait_for(coro, timeout)
        except asyncio.TimeoutError:
            return default

    @staticmethod
    async def gather_with_concurrency(n: int, *coros):
        """限制并发数执行协程"""
        semaphore = asyncio.Semaphore(n)

        async def sem_coro(coro):
            async with semaphore:
                return await coro

        return await asyncio.gather(*(sem_coro(c) for c in coros))

    @staticmethod
    def sync_to_async(func: Callable) -> Callable:
        """同步函数转换为异步函数"""

        @wraps(func)
        async def wrapper(*args, **kwargs):
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, lambda: func(*args, **kwargs))

        return wrapper


# 快捷函数实例
text_helper = TextHelper()
file_helper = FileHelper()
time_helper = TimeHelper()
security_helper = SecurityHelper()
performance_helper = PerformanceHelper()
async_helper = AsyncHelper()


# 快捷函数
def truncate_text(text: str, max_length: int, **kwargs) -> str:
    """截断文本（快捷函数）"""
    return text_helper.truncate_text(text, max_length, **kwargs)


def ensure_directory(directory: Union[str, Path]) -> Path:
    """确保目录存在（快捷函数）"""
    return file_helper.ensure_directory(directory)


def get_timestamp(fmt: str = "%Y%m%d_%H%M%S") -> str:
    """获取时间戳（快捷函数）"""
    return time_helper.get_timestamp(fmt)


def generate_token(length: int = 32) -> str:
    """生成随机令牌（快捷函数）"""
    return security_helper.generate_token(length)


def time_function(func: Callable) -> Callable:
    """函数计时装饰器（快捷函数）"""
    return performance_helper.time_function(func)