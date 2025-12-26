"""
验证器模块 - 数据验证和清理
"""
import re
import json
from typing import Any, Dict, List, Optional, Union, Tuple
from datetime import datetime, date
from email.utils import parseaddr
import phonenumbers


class DataValidator:
    """数据验证器"""

    @staticmethod
    def validate_email(email: str) -> bool:
        """验证邮箱地址"""
        if not email or not isinstance(email, str):
            return False

        # 使用email.utils解析
        parsed = parseaddr(email)
        if '@' not in parsed[1]:
            return False

        # 基本格式检查
        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        return bool(re.match(pattern, email))

    @staticmethod
    def validate_phone(phone: str, country_code: str = "CN") -> bool:
        """验证电话号码"""
        if not phone or not isinstance(phone, str):
            return False

        try:
            parsed = phonenumbers.parse(phone, country_code)
            return phonenumbers.is_valid_number(parsed)
        except:
            return False

    @staticmethod
    def validate_url(url: str) -> bool:
        """验证URL"""
        if not url or not isinstance(url, str):
            return False

        pattern = r'^https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+(?::\d+)?(?:/[-\w.%]*)*$'
        return bool(re.match(pattern, url))

    @staticmethod
    def validate_date(date_str: str, fmt: str = "%Y-%m-%d") -> bool:
        """验证日期格式"""
        try:
            datetime.strptime(date_str, fmt)
            return True
        except (ValueError, TypeError):
            return False

    @staticmethod
    def validate_json(json_str: str) -> bool:
        """验证JSON字符串"""
        try:
            json.loads(json_str)
            return True
        except (json.JSONDecodeError, TypeError):
            return False

    @staticmethod
    def validate_length(text: str, min_len: int = 0, max_len: int = None) -> bool:
        """验证文本长度"""
        if not isinstance(text, str):
            return False

        length = len(text)
        if length < min_len:
            return False

        if max_len is not None and length > max_len:
            return False

        return True

    @staticmethod
    def validate_range(value: Union[int, float],
                       min_val: Optional[Union[int, float]] = None,
                       max_val: Optional[Union[int, float]] = None) -> bool:
        """验证数值范围"""
        if not isinstance(value, (int, float)):
            return False

        if min_val is not None and value < min_val:
            return False

        if max_val is not None and value > max_val:
            return False

        return True


class InputSanitizer:
    """输入清理器"""

    @staticmethod
    def sanitize_text(text: str,
                      remove_html: bool = True,
                      remove_script: bool = True,
                      max_length: int = 5000) -> str:
        """
        清理文本

        Args:
            text: 原始文本
            remove_html: 是否移除HTML标签
            remove_script: 是否移除脚本
            max_length: 最大长度

        Returns:
            清理后的文本
        """
        if not text:
            return ""

        # 转换为字符串
        sanitized = str(text)

        # 移除HTML标签
        if remove_html:
            sanitized = re.sub(r'<[^>]*>', '', sanitized)

        # 移除脚本
        if remove_script:
            sanitized = re.sub(r'<script[^>]*>.*?</script>', '', sanitized, flags=re.DOTALL | re.IGNORECASE)
            sanitized = re.sub(r'on\w+="[^"]*"', '', sanitized, flags=re.IGNORECASE)
            sanitized = re.sub(r'on\w+=\'[^\']*\'', '', sanitized, flags=re.IGNORECASE)
            sanitized = re.sub(r'on\w+=[^ >]*', '', sanitized, flags=re.IGNORECASE)

        # 限制长度
        if max_length and len(sanitized) > max_length:
            sanitized = sanitized[:max_length]

        # 清理空白字符
        sanitized = ' '.join(sanitized.split())

        return sanitized.strip()

    @staticmethod
    def sanitize_filename(filename: str) -> str:
        """清理文件名"""
        if not filename:
            return ""

        # 移除非法字符
        sanitized = re.sub(r'[<>:"/\\|?*]', '_', filename)

        # 限制长度
        if len(sanitized) > 255:
            name, ext = sanitized.rsplit('.', 1) if '.' in sanitized else (sanitized, '')
            max_name_len = 255 - len(ext) - 1 if ext else 255
            name = name[:max_name_len]
            sanitized = f"{name}.{ext}" if ext else name

        return sanitized

    @staticmethod
    def sanitize_json(data: Union[str, Dict, List]) -> Dict:
        """
        清理JSON数据

        Args:
            data: JSON数据

        Returns:
            清理后的字典
        """
        # 如果是字符串，尝试解析
        if isinstance(data, str):
            try:
                data = json.loads(data)
            except:
                return {}

        # 递归清理
        def recursive_sanitize(obj):
            if isinstance(obj, dict):
                return {k: recursive_sanitize(v) for k, v in obj.items() if k and isinstance(k, str)}
            elif isinstance(obj, list):
                return [recursive_sanitize(item) for item in obj]
            elif isinstance(obj, str):
                return InputSanitizer.sanitize_text(obj)
            else:
                return obj

        result = recursive_sanitize(data)
        return result if isinstance(result, dict) else {}


class SchemaValidator:
    """模式验证器"""

    def __init__(self):
        self.schemas = {}

    def register_schema(self, name: str, schema: Dict):
        """注册验证模式"""
        self.schemas[name] = schema

    def validate(self, data: Dict, schema_name: str) -> Tuple[bool, List[str]]:
        """
        根据模式验证数据

        Args:
            data: 要验证的数据
            schema_name: 模式名称

        Returns:
            (是否有效, 错误消息列表)
        """
        if schema_name not in self.schemas:
            return False, [f"未找到模式: {schema_name}"]

        schema = self.schemas[schema_name]
        errors = []

        # 验证必需字段
        for field, rules in schema.get("required", {}).items():
            if field not in data:
                errors.append(f"缺少必需字段: {field}")
            else:
                field_errors = self._validate_field(data[field], rules, field)
                errors.extend(field_errors)

        # 验证可选字段
        for field, rules in schema.get("optional", {}).items():
            if field in data:
                field_errors = self._validate_field(data[field], rules, field)
                errors.extend(field_errors)

        return len(errors) == 0, errors

    def _validate_field(self, value: Any, rules: Dict, field_name: str) -> List[str]:
        """验证单个字段"""
        errors = []

        # 类型检查
        expected_type = rules.get("type")
        if expected_type:
            if expected_type == "string":
                if not isinstance(value, str):
                    errors.append(f"字段 {field_name} 应为字符串类型")
            elif expected_type == "integer":
                if not isinstance(value, int):
                    errors.append(f"字段 {field_name} 应为整数类型")
            elif expected_type == "number":
                if not isinstance(value, (int, float)):
                    errors.append(f"字段 {field_name} 应为数字类型")
            elif expected_type == "boolean":
                if not isinstance(value, bool):
                    errors.append(f"字段 {field_name} 应为布尔类型")
            elif expected_type == "array":
                if not isinstance(value, list):
                    errors.append(f"字段 {field_name} 应为数组类型")
            elif expected_type == "object":
                if not isinstance(value, dict):
                    errors.append(f"字段 {field_name} 应为对象类型")

        # 长度检查
        if "min_length" in rules and isinstance(value, str):
            if len(value) < rules["min_length"]:
                errors.append(f"字段 {field_name} 长度不能少于 {rules['min_length']} 个字符")

        if "max_length" in rules and isinstance(value, str):
            if len(value) > rules["max_length"]:
                errors.append(f"字段 {field_name} 长度不能超过 {rules['max_length']} 个字符")

        # 范围检查
        if "min" in rules and isinstance(value, (int, float)):
            if value < rules["min"]:
                errors.append(f"字段 {field_name} 不能小于 {rules['min']}")

        if "max" in rules and isinstance(value, (int, float)):
            if value > rules["max"]:
                errors.append(f"字段 {field_name} 不能大于 {rules['max']}")

        # 模式匹配
        if "pattern" in rules and isinstance(value, str):
            if not re.match(rules["pattern"], value):
                errors.append(f"字段 {field_name} 格式不正确")

        # 枚举值
        if "enum" in rules:
            if value not in rules["enum"]:
                errors.append(f"字段 {field_name} 必须是以下值之一: {rules['enum']}")

        return errors


# 快捷函数
validator = DataValidator()
sanitizer = InputSanitizer()
schema_validator = SchemaValidator()


def validate_email(email: str) -> bool:
    """验证邮箱（快捷函数）"""
    return validator.validate_email(email)


def sanitize_text(text: str, **kwargs) -> str:
    """清理文本（快捷函数）"""
    return sanitizer.sanitize_text(text, **kwargs)


def validate_schema(data: Dict, schema_name: str) -> Tuple[bool, List[str]]:
    """验证模式（快捷函数）"""
    return schema_validator.validate(data, schema_name)