"""
通用LLM服务 - 支持多个API提供商
"""
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
import time
import traceback
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod

from openai import OpenAI
from openai.types.chat import ChatCompletion
import backoff

from utils.logger import SystemLogger
from .llm_provider import LLMProvider, LLMModelConfig


class APIErrorType(Enum):
    """API错误类型"""
    RATE_LIMIT = "rate_limit"
    TOKEN_LIMIT = "token_limit"
    AUTHENTICATION = "authentication"
    NETWORK = "network"
    SERVER = "server"
    UNKNOWN = "unknown"


@dataclass
class APIRequest:
    """API请求记录"""
    request_id: str
    model: str
    provider: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    latency: float
    success: bool
    error_type: Optional[APIErrorType] = None
    error_message: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)


class BaseLLMService(ABC):
    """LLM服务基类"""

    def __init__(self, config: LLMModelConfig):
        self.config = config
        self.provider = config.provider
        self.model = config.model_name
        self.api_key = config.api_key
        self.base_url = config.base_url
        self.timeout = config.timeout
        self.max_retries = config.max_retries

        self.client = None
        self.logger = SystemLogger().get_logger(f"llm_service_{self.provider.value}")

        # 统计信息
        self.request_count = 0
        self.success_count = 0
        self.error_count = 0
        self.total_tokens = 0
        self.total_latency = 0.0

        # 请求历史
        self.request_history: List[APIRequest] = []
        self.max_history_size = 1000

        # 错误统计
        self.error_stats: Dict[APIErrorType, int] = {
            error_type: 0 for error_type in APIErrorType
        }

        # 初始化客户端
        self._initialize_client()

        self.logger.info(f"{self.provider.value}服务初始化完成，模型: {self.model}")

    @abstractmethod
    def _initialize_client(self):
        """初始化客户端（子类实现）"""
        pass

    @abstractmethod
    def _call_api(self, messages: List[Dict], max_tokens: int, temperature: float, stream: bool, request_id: str):
        """调用API（子类实现）"""
        pass

    def _test_initial_connection(self):
        """初始连接测试"""
        try:
            test_response = self._call_api(
                messages=[{"role": "user", "content": "测试连接"}],
                max_tokens=10,
                temperature=0.7,
                stream=False,
                request_id="test_connection"
            )

            if test_response and self._extract_content(test_response):
                self.logger.info("初始连接测试成功")
                self.success_count += 1
            else:
                self.logger.warning("初始连接测试响应为空")
        except Exception as e:
            self.logger.warning(f"初始连接测试失败: {e}")

    @abstractmethod
    def _extract_content(self, response) -> Optional[str]:
        """提取响应内容（子类实现）"""
        pass

    @abstractmethod
    def _extract_usage(self, response) -> Tuple[int, int, int]:
        """提取使用信息（子类实现）"""
        pass

    @backoff.on_exception(
        backoff.expo,
        Exception,
        max_tries=3,
        max_time=30
    )
    def generate_reply(self,
                       system_prompt: str,
                       user_input: str,
                       conversation_history: Optional[List[Dict]] = None,
                       max_tokens: int = 500,
                       temperature: float = 1.0,
                       stream: bool = False) -> Tuple[Optional[str], Optional[Dict]]:
        """
        调用API生成回复

        Args:
            system_prompt: 系统提示词
            user_input: 用户输入
            conversation_history: 对话历史
            max_tokens: 最大token数
            temperature: 温度参数
            stream: 是否流式传输

        Returns:
            Tuple[回复文本, 元数据]
        """
        start_time = time.time()
        request_id = f"req_{datetime.now().strftime('%Y%m%d%H%M%S')}_{self.request_count}"

        if not self.client:
            error_msg = "客户端未初始化"
            self.logger.error(error_msg)
            self._record_error(request_id, APIErrorType.NETWORK, error_msg, 0)
            return None, {"error": error_msg}

        try:
            # 构建消息列表
            messages = self._build_messages(system_prompt, user_input, conversation_history)

            # 调用API
            response = self._call_api(
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                stream=stream,
                request_id=request_id
            )

            if response:
                # 处理响应
                if stream:
                    reply = self._process_stream_response(response)
                else:
                    reply = self._extract_content(response)

                # 记录成功
                latency = time.time() - start_time
                prompt_tokens, completion_tokens, total_tokens = self._extract_usage(response)

                self._record_success(
                    request_id=request_id,
                    model=self.model,
                    provider=self.provider.value,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=total_tokens,
                    latency=latency
                )

                metadata = {
                    "request_id": request_id,
                    "provider": self.provider.value,
                    "model": self.model,
                    "tokens_used": total_tokens,
                    "latency": latency,
                    "finish_reason": self._extract_finish_reason(response)
                }

                return reply, metadata
            else:
                error_msg = "API响应为空"
                self.logger.error(error_msg)
                self._record_error(request_id, APIErrorType.SERVER, error_msg, 0)
                return None, {"error": error_msg}

        except Exception as e:
            latency = time.time() - start_time
            error_type = self._classify_error(e)
            error_msg = str(e)

            self.logger.error(f"生成回复失败: {error_msg}")
            self.logger.debug(f"详细错误堆栈:\n{traceback.format_exc()}")

            self._record_error(request_id, error_type, error_msg, latency)

            return None, {
                "error": error_msg,
                "error_type": error_type.value,
                "request_id": request_id
            }

    def _estimate_tokens(self, text: str) -> int:
        """估算文本的token数（简化版）"""
        # 简化的token估算：1个token ≈ 4个字符
        return len(text) // 4

    def _build_messages(self,
                        system_prompt: str,
                        user_input: str,
                        conversation_history: Optional[List[Dict]]) -> List[Dict]:
        """构建消息列表"""
        # 初始化消息列表
        messages = []
        total_tokens = 0
        
        # 从配置获取最大token数，默认3000
        max_total_tokens = getattr(self.config, 'max_context_tokens', 3000)
        original_history_length = len(conversation_history) if conversation_history else 0
        filtered_history_count = 0
        empty_message_count = 0

        # 系统提示词
        system_message = {"role": "system", "content": system_prompt}
        system_tokens = self._estimate_tokens(system_prompt)
        
        # 当前用户输入
        current_user_message = {"role": "user", "content": user_input}
        user_input_tokens = self._estimate_tokens(user_input)

        # 检查是否有足够的token空间
        if system_tokens + user_input_tokens > max_total_tokens:
            # 如果只有系统提示词和当前用户输入就超过了token限制，我们需要调整
            return [system_message, current_user_message]

        # 添加系统提示词
        messages.append(system_message)
        total_tokens += system_tokens

        # 处理历史对话
        if conversation_history:
            # 收集所有有效的历史轮次
            valid_turns = []
            for item in conversation_history:
                user_msg = None
                assistant_msg = None
                
                # 只处理非空消息
                if "user_input" in item and item["user_input"]:
                    user_msg = {"role": "user", "content": item["user_input"]}
                elif "user_input" in item:
                    empty_message_count += 1

                if "system_response" in item and item["system_response"]:
                    assistant_msg = {"role": "assistant", "content": item["system_response"]}
                elif "system_response" in item:
                    empty_message_count += 1
                
                # 检查是否有实际内容
                if user_msg or assistant_msg:
                    valid_turns.append((user_msg, assistant_msg))
            
            # 反向遍历有效的轮次，优先保留最近的对话
            for user_msg, assistant_msg in reversed(valid_turns):
                # 计算当前轮次的token数
                turn_tokens = 0
                if user_msg:
                    turn_tokens += self._estimate_tokens(user_msg["content"])
                if assistant_msg:
                    turn_tokens += self._estimate_tokens(assistant_msg["content"])
                
                # 检查是否超过token限制
                if total_tokens + turn_tokens + user_input_tokens > max_total_tokens:
                    break
                
                # 构建当前轮次的消息，确保正确的顺序
                turn_messages = []
                if user_msg:
                    turn_messages.append(user_msg)
                if assistant_msg:
                    turn_messages.append(assistant_msg)
                
                # 在系统提示词之后插入历史消息，保持对话顺序
                messages[1:1] = turn_messages
                total_tokens += turn_tokens
                filtered_history_count += 1

        # 检查是否可以添加当前用户输入
        if total_tokens + user_input_tokens <= max_total_tokens:
            # 添加当前用户输入前检查角色，避免连续相同角色
            if messages and messages[-1]['role'] == "user":
                # 如果最后一条是user消息，我们需要确保不会连续添加user消息
                # 我们检查是否有token空间添加一个简短的默认assistant消息
                default_assistant_msg = {"role": "assistant", "content": "..."}
                default_tokens = self._estimate_tokens(default_assistant_msg["content"])
                
                if total_tokens + default_tokens + user_input_tokens <= max_total_tokens:
                    messages.append(default_assistant_msg)
                    total_tokens += default_tokens
            
            # 添加当前用户输入
            messages.append(current_user_message)
            total_tokens += user_input_tokens

        # 最后检查整个消息列表，确保没有连续的相同角色
        cleaned_messages = []
        for msg in messages:
            if not cleaned_messages or cleaned_messages[-1]['role'] != msg['role']:
                cleaned_messages.append(msg)
        
        # 重新计算总token数
        final_total_tokens = sum(self._estimate_tokens(msg["content"]) for msg in cleaned_messages)

        # 记录优化日志
        self.logger.debug(
            f"消息构建完成: 原始历史长度={original_history_length}, "
            f"保留历史轮次={filtered_history_count}, "
            f"过滤空消息数={empty_message_count}, "
            f"总token数={final_total_tokens}, "
            f"系统提示词token={system_tokens}, "
            f"当前输入token={user_input_tokens}, "
            f"历史对话token={final_total_tokens - system_tokens - user_input_tokens}"
        )

        return cleaned_messages

    def _process_stream_response(self, response) -> str:
        """处理流式响应"""
        reply_parts = []

        for chunk in response:
            content = self._extract_content(chunk)
            if content:
                reply_parts.append(content)

        return "".join(reply_parts)

    @abstractmethod
    def _extract_finish_reason(self, response) -> Optional[str]:
        """提取完成原因（子类实现）"""
        pass

    def _classify_error(self, error: Exception) -> APIErrorType:
        """分类错误类型"""
        error_str = str(error).lower()

        if "rate limit" in error_str:
            return APIErrorType.RATE_LIMIT
        elif "token" in error_str and ("limit" in error_str or "exceeded" in error_str):
            return APIErrorType.TOKEN_LIMIT
        elif "auth" in error_str or "key" in error_str or "invalid" in error_str:
            return APIErrorType.AUTHENTICATION
        elif "network" in error_str or "timeout" in error_str or "connection" in error_str:
            return APIErrorType.NETWORK
        elif "server" in error_str or "internal" in error_str:
            return APIErrorType.SERVER
        else:
            return APIErrorType.UNKNOWN

    def _record_success(self,
                        request_id: str,
                        model: str,
                        provider: str,
                        prompt_tokens: int,
                        completion_tokens: int,
                        total_tokens: int,
                        latency: float):
        """记录成功请求"""
        self.success_count += 1
        self.total_tokens += total_tokens
        self.total_latency += latency

        # 记录请求历史
        request_record = APIRequest(
            request_id=request_id,
            model=model,
            provider=provider,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            latency=latency,
            success=True
        )

        self.request_history.append(request_record)

        # 保持历史大小
        if len(self.request_history) > self.max_history_size:
            self.request_history = self.request_history[-self.max_history_size:]

    def _record_error(self,
                      request_id: str,
                      error_type: APIErrorType,
                      error_message: str,
                      latency: float):
        """记录错误请求"""
        self.error_count += 1
        self.error_stats[error_type] = self.error_stats.get(error_type, 0) + 1

        # 记录错误请求历史
        request_record = APIRequest(
            request_id=request_id,
            model=self.model,
            provider=self.provider.value,
            prompt_tokens=0,
            completion_tokens=0,
            total_tokens=0,
            latency=latency,
            success=False,
            error_type=error_type,
            error_message=error_message
        )

        self.request_history.append(request_record)

        # 保持历史大小
        if len(self.request_history) > self.max_history_size:
            self.request_history = self.request_history[-self.max_history_size:]

    def is_available(self) -> bool:
        """检查服务是否可用"""
        return self.client is not None

    def test_connection(self) -> Tuple[bool, Optional[str]]:
        """测试连接"""
        if not self.client:
            return False, "客户端未初始化"

        try:
            test_response = self._call_api(
                messages=[{"role": "user", "content": "测试连接，请回复'连接成功'"}],
                max_tokens=10,
                temperature=0.7,
                stream=False,
                request_id="test_connection"
            )

            if test_response and self._extract_content(test_response):
                return True, "连接测试成功"
            else:
                return False, "测试响应为空"

        except Exception as e:
            return False, f"连接测试失败: {str(e)}"

    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        success_rate = (self.success_count / max(self.request_count, 1)) * 100

        avg_latency = self.total_latency / max(self.success_count, 1) if self.success_count > 0 else 0

        return {
            "provider": self.provider.value,
            "model": self.model,
            "base_url": self.base_url,
            "total_requests": self.request_count,
            "success_count": self.success_count,
            "error_count": self.error_count,
            "success_rate": success_rate,
            "total_tokens": self.total_tokens,
            "avg_latency": avg_latency,
            "error_stats": {k.value: v for k, v in self.error_stats.items()},
            "is_available": self.is_available(),
            "recent_requests": len(self.request_history),
            "client_status": "initialized" if self.client else "not_initialized"
        }

    def get_recent_requests(self, limit: int = 10) -> List[Dict]:
        """获取最近请求"""
        recent = self.request_history[-limit:] if self.request_history else []

        return [{
            "request_id": req.request_id,
            "provider": req.provider,
            "model": req.model,
            "tokens": req.total_tokens,
            "latency": req.latency,
            "success": req.success,
            "error_type": req.error_type.value if req.error_type else None,
            "timestamp": req.timestamp.isoformat()
        } for req in recent]

    def clear_history(self):
        """清空历史记录"""
        self.request_history.clear()
        self.logger.info("已清空请求历史")

    def switch_model(self, model: str) -> bool:
        """切换模型"""
        old_model = self.model
        self.model = model

        # 测试新模型
        success, message = self.test_connection()

        if success:
            self.logger.info(f"模型切换成功: {old_model} -> {model}")
            return True
        else:
            # 回滚
            self.model = old_model
            self.logger.error(f"模型切换失败: {message}")
            return False

    def shutdown(self):
        """关闭服务"""
        self.logger.info(f"正在关闭{self.provider.value}服务...")

        # 这里可以添加清理逻辑，如关闭连接池等

        self.logger.info(f"{self.provider.value}服务已关闭")


class OpenAICompatibleService(BaseLLMService):
    """OpenAI兼容服务（支持OpenAI、DeepSeek等）"""

    def _initialize_client(self):
        """初始化OpenAI兼容客户端"""
        try:
            self.client = OpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
                timeout=self.timeout
            )

            # 测试连接
            self._test_initial_connection()

        except ImportError as e:
            self.logger.error(f"导入失败: {str(e)}")
            self.logger.error("请运行: pip install openai")
            self.client = None
        except Exception as e:
            self.logger.error(f"初始化失败: {str(e)}")
            self.client = None

    def _call_api(self, messages: List[Dict], max_tokens: int, temperature: float, stream: bool, request_id: str):
        """调用OpenAI兼容API"""
        self.request_count += 1

        try:
            self.logger.debug(
                f"调用API，提供商: {self.provider.value}, "
                f"模型: {self.model}, "
                f"消息数量: {len(messages)}, "
                f"最大token: {max_tokens}, "
                f"请求ID: {request_id}"
            )

            # 根据配置选择使用 max_tokens 或 max_completion_tokens
            if self.config.use_max_completion_tokens:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    max_completion_tokens=max_tokens,
                    temperature=temperature,
                    stream=stream
                )
            else:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    stream=stream
                )

            return response

        except Exception as e:

            raise e

    def _extract_content(self, response) -> Optional[str]:
        """提取响应内容"""
        try:
            if response and hasattr(response, 'choices') and response.choices:
                if hasattr(response.choices[0], 'delta'):
                    # 流式响应
                    return response.choices[0].delta.content
                elif hasattr(response.choices[0], 'message'):
                    # 非流式响应
                    return response.choices[0].message.content
            return None
        except Exception:
            return None

    def _extract_usage(self, response) -> Tuple[int, int, int]:
        """提取使用信息"""
        try:
            if response and hasattr(response, 'usage') and response.usage:
                return (
                    response.usage.prompt_tokens,
                    response.usage.completion_tokens,
                    response.usage.total_tokens
                )
        except Exception:
            pass
        return 0, 0, 0

    def _extract_finish_reason(self, response) -> Optional[str]:
        """提取完成原因"""
        try:
            if response and hasattr(response, 'choices') and response.choices:
                return response.choices[0].finish_reason
        except Exception:
            pass
        return None
