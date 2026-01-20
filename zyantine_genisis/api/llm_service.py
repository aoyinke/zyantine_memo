"""
通用LLM服务 - 支持多个API提供商
"""
from typing import Dict, List, Optional, Tuple, Any, Callable, Union
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
        self.consecutive_errors = 0  # 连续错误计数
        self.last_success_time = None  # 最后成功时间
        self.last_error_time = None  # 最后错误时间
        self.last_error_type = None  # 最后错误类型

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

    def generate_reply_parallel(self, 
                               system_prompt: str, 
                               user_input: str, 
                               conversation_history: Optional[List[Dict]] = None,
                               models: List[str] = None, 
                               max_tokens: int = 500, 
                               temperature: float = 1.0) -> Tuple[Optional[str], Optional[Dict]]:
        """
        并行调用多个模型，返回最快的响应

        Args:
            system_prompt: 系统提示词
            user_input: 用户输入
            conversation_history: 对话历史
            models: 要并行调用的模型列表
            max_tokens: 最大token数
            temperature: 温度参数

        Returns:
            Tuple[回复文本, 元数据]
        """
        if not models:
            models = [self.model]

        import concurrent.futures
        results = {}
        errors = {}

        def call_model(model):
            old_model = self.model
            try:
                self.model = model
                result, metadata = self.generate_reply(
                    system_prompt=system_prompt,
                    user_input=user_input,
                    conversation_history=conversation_history,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    stream=False
                )
                return model, result, metadata
            except Exception as e:
                return model, None, {"error": str(e)}
            finally:
                self.model = old_model

        # 并行调用模型
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(len(models), 3)) as executor:
            future_to_model = {executor.submit(call_model, model): model for model in models}
            for future in concurrent.futures.as_completed(future_to_model, timeout=30):
                model = future_to_model[future]
                try:
                    model_name, result, metadata = future.result()
                    if result:
                        results[model_name] = (result, metadata)
                    else:
                        errors[model_name] = metadata.get("error", "Unknown error")
                except Exception as e:
                    errors[model_name] = str(e)

        # 返回第一个成功的结果
        if results:
            # 按响应时间排序，返回最快的
            fastest_model = min(results.keys(), key=lambda m: results[m][1].get("latency", float('inf')))
            result, metadata = results[fastest_model]
            metadata["model_used"] = fastest_model
            self.logger.info(f"并行调用完成，使用模型: {fastest_model}")
            return result, metadata
        else:
            # 所有模型都失败
            error_msg = "所有模型调用失败: " + "; ".join([f"{m}: {e}" for m, e in errors.items()])
            self.logger.error(error_msg)
            return None, {"error": error_msg, "errors": errors}

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
                       stream: bool = False) -> Union[Tuple[Optional[str], Optional[Dict]], Any]:
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
                    # 优化流式响应生成器
                    def stream_generator():
                        reply_parts = []
                        all_content_parts = []  # 用于最终提取情绪
                        start_chunk_time = time.time()
                        chunk_count = 0
                        
                        for chunk in response:
                            content = self._extract_content(chunk)
                            if content:
                                chunk_count += 1
                                reply_parts.append(content)
                                all_content_parts.append(content)
                                # 优化：小 chunk 合并，减少网络往返
                                if chunk_count % 3 == 0 or len("".join(reply_parts)) > 100:
                                    # 清理回复：移除动作描述和情绪标签
                                    chunk_content = "".join(reply_parts)
                                    clean_chunk = self.clean_reply(chunk_content)
                                    if clean_chunk:
                                        yield clean_chunk
                                    reply_parts = []
                                    chunk_count = 0
                        
                        # 发送剩余部分
                        if reply_parts:
                            # 清理回复：移除动作描述和情绪标签
                            final_content = "".join(reply_parts)
                            clean_final = self.clean_reply(final_content)
                            if clean_final:
                                yield clean_final
                        
                        # 响应完成后记录统计信息
                        latency = time.time() - start_time
                        chunk_latency = time.time() - start_chunk_time
                        
                        # 提取情绪信息（从完整内容中提取）
                        full_reply = "".join(all_content_parts)
                        emotion = self.extract_emotion(full_reply)
                        
                        try:
                            prompt_tokens, completion_tokens, total_tokens = self._extract_usage(response)
                        except Exception:
                            prompt_tokens, completion_tokens, total_tokens = 0, 0, 0
                        
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
                            "chunk_latency": chunk_latency,
                            "finish_reason": self._extract_finish_reason(response),
                            "emotion": emotion
                        }
                    
                    return stream_generator(), None
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

                    # 提取情绪信息（在清理之前）
                    emotion = self.extract_emotion(reply)
                    # 清理回复：移除动作描述和情绪标签
                    clean_reply = self.clean_reply(reply)
                    
                    metadata = {
                        "request_id": request_id,
                        "provider": self.provider.value,
                        "model": self.model,
                        "tokens_used": total_tokens,
                        "latency": latency,
                        "finish_reason": self._extract_finish_reason(response),
                        "emotion": emotion
                    }

                    return clean_reply, metadata
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
        self.consecutive_errors = 0  # 重置连续错误计数
        self.last_success_time = datetime.now()  # 更新最后成功时间

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
        self.consecutive_errors += 1  # 递增连续错误计数
        self.last_error_time = datetime.now()  # 更新最后错误时间
        self.last_error_type = error_type  # 更新最后错误类型

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
        # 基本检查：客户端是否初始化
        if not self.client:
            self.logger.debug("服务不可用：客户端未初始化")
            return False
        
        # 检查API密钥是否有效
        if not self.api_key or len(self.api_key.strip()) < 10:
            self.logger.debug("服务不可用：API密钥无效")
            return False
        
        # 检查最近的错误情况：如果连续3个错误，标记为不可用
        if hasattr(self, 'consecutive_errors') and self.consecutive_errors >= 3:
            self.logger.debug(f"服务不可用：连续{self.consecutive_errors}个错误")
            return False
        
        # 检查最近的成功请求：如果最近有成功请求，认为可用
        if hasattr(self, 'last_success_time') and self.last_success_time is not None:
            from datetime import datetime, timedelta
            if datetime.now() - self.last_success_time < timedelta(minutes=5):
                self.logger.debug("服务可用：最近5分钟内有成功请求")
                return True
        
        # 检查是否有未解决的网络问题
        if hasattr(self, 'last_error_time') and self.last_error_time is not None and hasattr(self, 'last_error_type'):
            from datetime import datetime, timedelta
            if datetime.now() - self.last_error_time < timedelta(minutes=1):
                if self.last_error_type in [APIErrorType.NETWORK, APIErrorType.SERVER]:
                    self.logger.debug("服务不可用：最近1分钟内有网络或服务器错误")
                    return False
        
        # 默认可用
        self.logger.debug("服务可用：通过所有检查")
        return True

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

    def extract_emotion(self, text: str) -> str:
        """从文本中提取情绪标签"""
        import re
        match = re.search(r'\[EMOTION:(\w+)\]', text)
        if match:
            emotion = match.group(1).lower()
            # 验证情绪类型是否有效
            if self.validate_emotion(emotion):
                return emotion
        return "neutral"

    def validate_emotion(self, emotion: str) -> bool:
        """验证情绪类型是否有效"""
        valid_emotions = ["neutral", "happy", "sad", "angry", "excited", "calm", "surprised", "disgusted"]
        return emotion in valid_emotions

    def remove_emotion_tag(self, text: str) -> str:
        """从文本中移除情绪标签"""
        import re
        return re.sub(r'\s*\[EMOTION:\w+\]\s*$', '', text).strip()
    
    def extract_action(self, text: str) -> Optional[str]:
        """
        从文本中提取括号中的动作描述（包括表情动作）
        
        支持中文括号（）和英文括号()
        如果文本中有多个动作，返回第一个
        """
        import re
        
        if not text:
            return None
        
        # 匹配中文括号中的内容
        chinese_match = re.search(r'（([^）]+)）', text)
        if chinese_match:
            action = chinese_match.group(1).strip()
            # 排除系统内部动作（这些不应该作为表情动作）
            if not re.search(r'(?:正在|开始|尝试|查询|检索|搜索|分析|思考|处理|执行|调用)', action):
                return action
        
        # 匹配英文括号中的内容
        english_match = re.search(r'\(([^)]+)\)', text)
        if english_match:
            action = english_match.group(1).strip()
            # 排除系统内部动作
            if not re.search(r'(?:正在|开始|尝试|查询|检索|搜索|分析|思考|处理|执行|调用)', action):
                return action
        
        return None
    
    def remove_action_descriptions(self, text: str) -> str:
        """
        从文本中移除动作描述
        
        动作描述是指AI对自己内部行为的描述，如：
        - "让我查询一下记忆..."
        - "我正在思考..."
        - "根据我的分析..."
        - "*思考中*"
        - "[正在检索相关信息]"
        
        这些内容不应该出现在给用户的回复中。
        """
        import re
        
        if not text:
            return text
        
        original_text = text
        
        # 1. 移除方括号包裹的动作描述 [xxx]
        # 但保留情绪标签 [EMOTION:xxx]，因为会在其他地方处理
        text = re.sub(r'\[(?!EMOTION:)[^\]]*(?:正在|开始|尝试|查询|检索|搜索|分析|思考|处理|执行|调用)[^\]]*\]', '', text)
        
        # 2. 移除星号包裹的动作描述 *xxx*
        text = re.sub(r'\*[^*]*(?:思考|分析|查询|检索|搜索|处理|执行|调用|回忆|记忆)[^*]*\*', '', text)
        
        # 3. 移除括号包裹的动作描述 (xxx) 或 （xxx）
        # 先移除系统内部动作
        text = re.sub(r'[（(][^）)]*(?:正在|开始|尝试|查询|检索|搜索|分析|思考|处理|执行|调用)[^）)]*[）)]', '', text)
        # 再移除所有剩余的括号内容（包括表情动作）
        text = re.sub(r'（[^）]+）', '', text)
        text = re.sub(r'\([^)]+\)', '', text)
        
        # 4. 移除常见的动作描述开头句式
        action_patterns = [
            # 查询/检索相关
            r'^让我(?:先)?(?:查询|检索|搜索|查找|查看|翻阅)一下[^。，,\.]*[。，,\.]?\s*',
            r'^我(?:先)?(?:查询|检索|搜索|查找|查看|翻阅)一下[^。，,\.]*[。，,\.]?\s*',
            r'^(?:正在)?(?:查询|检索|搜索|查找)(?:相关)?(?:记忆|信息|内容|资料)[^。，,\.]*[。，,\.]?\s*',
            
            # 思考/分析相关
            r'^让我(?:先)?(?:思考|想想|分析|考虑)一下[^。，,\.]*[。，,\.]?\s*',
            r'^我(?:先)?(?:思考|想想|分析|考虑)一下[^。，,\.]*[。，,\.]?\s*',
            r'^(?:正在)?(?:思考|分析|处理)[^。，,\.]*[。，,\.]?\s*',
            
            # 记忆相关
            r'^让我(?:先)?(?:回忆|回想|想起)[^。，,\.]*[。，,\.]?\s*',
            r'^我(?:先)?(?:回忆|回想|想起)[^。，,\.]*[。，,\.]?\s*',
            r'^(?:正在)?(?:调用|访问|读取)(?:记忆|数据)[^。，,\.]*[。，,\.]?\s*',
            
            # 系统动作相关
            r'^(?:正在)?(?:执行|运行|启动|调用)[^。，,\.]*(?:流程|程序|模块|功能)[^。，,\.]*[。，,\.]?\s*',
            r'^(?:正在)?(?:使用|应用|采用)[^。，,\.]*(?:策略|方法|模式)[^。，,\.]*[。，,\.]?\s*',
            
            # 根据xxx相关（但保留正常的"根据你说的"等）
            r'^根据(?:我的|系统的|内部的)(?:分析|判断|评估|记忆)[^。，,\.]*[。，,\.]?\s*',
            r'^基于(?:我的|系统的|内部的)(?:分析|判断|评估|记忆)[^。，,\.]*[。，,\.]?\s*',
        ]
        
        for pattern in action_patterns:
            text = re.sub(pattern, '', text, flags=re.IGNORECASE)
        
        # 5. 移除文本中间的动作描述（更保守，只移除明显的）
        mid_action_patterns = [
            r'[。，,\.]\s*(?:让我|我(?:先)?)?(?:查询|检索|搜索)一下[^。，,\.]*[。，,\.]',
            r'[。，,\.]\s*(?:正在)?(?:思考|分析|处理)中[^。，,\.]*[。，,\.]',
        ]
        
        for pattern in mid_action_patterns:
            text = re.sub(pattern, '。', text, flags=re.IGNORECASE)
        
        # 6. 清理多余的空白和标点
        text = re.sub(r'\s+', ' ', text)  # 合并多个空格
        text = re.sub(r'^[。，,\.\s]+', '', text)  # 移除开头的标点和空格
        text = re.sub(r'[。，,\.]{2,}', '。', text)  # 合并多个标点
        text = text.strip()
        
        # 如果处理后文本为空或过短，返回原文本（避免过度过滤）
        if not text or len(text) < 5:
            return original_text
        
        return text
    
    def clean_reply(self, text: str) -> str:
        """
        清理回复文本，移除所有不应该出现在用户回复中的内容
        
        包括：
        1. 情绪标签
        2. 动作描述
        """
        if not text:
            return text
        
        # 先移除动作描述
        text = self.remove_action_descriptions(text)
        # 再移除情绪标签
        text = self.remove_emotion_tag(text)
        
        return text


class OpenAICompatibleService(BaseLLMService):
    """OpenAI兼容服务（支持OpenAI、DeepSeek等）"""

    def _initialize_client(self):
        """初始化OpenAI兼容客户端"""
        try:
            # 检查API密钥是否配置
            if not self.api_key:
                self.logger.error("未配置API密钥，无法初始化API客户端")
                self.client = None
                return
            
            # 验证API密钥格式
            if not isinstance(self.api_key, str) or len(self.api_key.strip()) < 10:
                self.logger.error("API密钥格式无效，请检查配置")
                self.client = None
                return
            
            # 记录初始化信息
            self.logger.info(f"正在初始化{self.provider.value}客户端，模型: {self.model}")
            self.logger.debug(f"基础URL: {self.base_url}")
            
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
            self.logger.debug(f"详细错误堆栈:\n{traceback.format_exc()}")
            # 提供更具体的错误诊断
            error_str = str(e).lower()
            if "invalid_api_key" in error_str or "authentication" in error_str:
                self.logger.error("API密钥无效，请检查配置")
            elif "model_not_found" in error_str or "invalid_model" in error_str:
                self.logger.error(f"模型 {self.model} 不存在或不可用")
            elif "connection" in error_str or "timeout" in error_str:
                self.logger.error("网络连接失败，请检查网络环境或基础URL配置")
            elif "rate_limit" in error_str:
                self.logger.error("API请求频率过高，请检查配额或调整请求频率")
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
