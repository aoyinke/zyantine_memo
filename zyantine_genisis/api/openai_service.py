"""
OpenAI服务 - 封装OpenAI API调用
"""
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
import time
import traceback
from dataclasses import dataclass, field
from enum import Enum

from openai import OpenAI
from openai.types.chat import ChatCompletion
import backoff

from zyantine_genisis.utils.logger import SystemLogger


class OpenAIModel(Enum):
    """OpenAI模型枚举"""
    GPT4_TURBO = "gpt-4-turbo"
    GPT4 = "gpt-4"
    GPT35_TURBO = "gpt-3.5-turbo"
    GPT4O = "gpt-4o"
    GPT4O_MINI = "gpt-4o-mini"


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
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    latency: float
    success: bool
    error_type: Optional[APIErrorType] = None
    error_message: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)


class OpenAIService:
    """OpenAI API服务封装（增强版）"""

    def __init__(self,
                 api_key: str,
                 base_url: str = "https://openkey.cloud/v1",
                 model: str = "gpt-5-nano-2025-08-07",
                 timeout: int = 30,
                 max_retries: int = 3,
                 use_max_completion_tokens: bool = False):

        self.api_key = api_key
        self.base_url = base_url
        self.model = model
        self.timeout = timeout
        self.max_retries = max_retries
        self.use_max_completion_tokens = use_max_completion_tokens

        self.client = None
        self.logger = SystemLogger().get_logger("openai_service")

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

        self.logger.info(f"OpenAI服务初始化完成，模型: {model}")

    def _initialize_client(self):
        """初始化OpenAI客户端"""
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

    def _test_initial_connection(self):
        """初始连接测试"""
        try:
            test_response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": "测试连接"}],
                max_tokens=10
            )

            if test_response and test_response.choices:
                self.logger.info("初始连接测试成功")
                self.success_count += 1
            else:
                self.logger.warning("初始连接测试响应为空")
        except Exception as e:
            self.logger.warning(f"初始连接测试失败: {e}")

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
                    reply = response.choices[0].message.content

                # 记录成功
                latency = time.time() - start_time
                tokens_used = response.usage.total_tokens if response.usage else 0

                self._record_success(
                    request_id=request_id,
                    model=self.model,
                    prompt_tokens=response.usage.prompt_tokens if response.usage else 0,
                    completion_tokens=response.usage.completion_tokens if response.usage else 0,
                    total_tokens=tokens_used,
                    latency=latency
                )

                metadata = {
                    "request_id": request_id,
                    "model": self.model,
                    "tokens_used": tokens_used,
                    "latency": latency,
                    "finish_reason": response.choices[0].finish_reason if response.choices else None
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

    def _build_messages(self,
                        system_prompt: str,
                        user_input: str,
                        conversation_history: Optional[List[Dict]]) -> List[Dict]:
        """构建消息列表"""
        messages = []

        # 系统提示词
        messages.append({"role": "system", "content": system_prompt})

        # 添加历史对话
        if conversation_history:
            for item in conversation_history[-10:]:  # 只取最近10条历史
                if "user_input" in item:
                    messages.append({"role": "user", "content": item["user_input"]})
                if "system_response" in item:
                    messages.append({"role": "assistant", "content": item["system_response"]})

        # 当前用户输入
        messages.append({"role": "user", "content": user_input})

        return messages

    def _call_api(self,
                  messages: List[Dict],
                  max_tokens: int,
                  temperature: float,
                  stream: bool,
                  request_id: str) -> Optional[ChatCompletion]:
        """调用API"""
        self.request_count += 1

        try:
            self.logger.debug(
                f"调用API，模型: {self.model}, "
                f"消息数量: {len(messages)}, "
                f"最大token: {max_tokens}, "
                f"请求ID: {request_id}"
            )

            # 根据配置选择使用 max_tokens 或 max_completion_tokens
            if self.use_max_completion_tokens:
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

    def _process_stream_response(self, response) -> str:
        """处理流式响应"""
        reply_parts = []

        for chunk in response:
            if chunk.choices[0].delta.content:
                reply_parts.append(chunk.choices[0].delta.content)

        return "".join(reply_parts)

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
            test_response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": "测试连接，请回复'连接成功'"}],
                max_tokens=10
            )

            if test_response and test_response.choices:
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
        self.logger.info("正在关闭OpenAI服务...")

        # 这里可以添加清理逻辑，如关闭连接池等
        # 目前OpenAI客户端没有显式的关闭方法

        self.logger.info("OpenAI服务已关闭")