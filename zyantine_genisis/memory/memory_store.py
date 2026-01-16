# ============ 基于memo0的简化记忆系统 ============
import os
import json
import hashlib
import time
from datetime import datetime
from typing import Dict, List, Any, Optional, Union, Tuple

# Mem0框架导入
from mem0 import Memory

# 自定义异常类导入
from .memory_exceptions import (
    MemoryException,
    MemoryStorageError,
    MemoryRetrievalError,
    MemorySearchError,
    MemoryValidationError,
    MemoryNotFoundError,
    MemoryCacheError,
    MemoryIndexError,
    MemorySizeExceededError,
    MemoryMetadataError,
    MemoryTypeError,
    MemorySerializationError,
    MemoryDeserializationError,
    MemoryConfigError,
    handle_memory_error
)

# 统一异常处理导入
from utils.exception_handler import (
    handle_error, retry_on_error, safe_execute,
    APIException, ConfigException, ProcessingException
)

# ============ 常量定义 ============
import os
DEFAULT_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://openkey.cloud/v1")
DEFAULT_API_KEY = os.getenv("OPENAI_API_KEY", "")

# ============ 大模型API调用模块 ============
class LLMClient:
    """大模型API调用客户端"""
    
    def __init__(self, api_key: str = None):
        """初始化大模型客户端"""
        self.api_key = api_key or DEFAULT_API_KEY or os.getenv("ZHIPU_API_KEY", "")
        self.cache = {}  # 缓存评估结果
        self.cache_expiry = 3600  # 缓存过期时间（秒）
        self.max_cache_size = 1000  # 最大缓存大小
        self.batch_requests = {}  # 批量请求队列
        self.batch_timeout = 0.1  # 批量请求超时时间（秒）
        self.last_batch_time = time.time()
        self.client = None
        # 初始化智谱AI客户端
        self._init_zhipu_client()
    
    def _init_zhipu_client(self):
        """初始化智谱AI客户端，支持多种导入方式"""
        try:
            from zai import ZhipuAiClient
            if self.api_key:
                self.client = ZhipuAiClient(api_key=self.api_key)
                print(f"[LLMClient] 智谱AI客户端(ZhipuAiClient)初始化成功")
            else:
                print(f"[LLMClient] 警告：智谱AI API密钥为空，无法初始化ZhipuAiClient")
                self.client = None
        except ImportError as e:
            print(f"[LLMClient] 导入ZhipuAiClient失败: {e}")
            # 尝试其他导入方式
            try:
                from zhipuai import ZhipuAI
                if self.api_key:
                    self.client = ZhipuAI(api_key=self.api_key)
                    print(f"[LLMClient] 智谱AI SDK客户端(ZhipuAI)初始化成功")
                else:
                    print(f"[LLMClient] 警告：智谱AI API密钥为空，无法初始化ZhipuAI")
                    self.client = None
            except ImportError as e2:
                print(f"[LLMClient] 导入智谱AI SDK失败: {e2}")
                self.client = None
            except Exception as e3:
                print(f"[LLMClient] 初始化ZhipuAI客户端失败: {e3}")
                self.client = None
        except Exception as e:
            print(f"[LLMClient] 初始化智谱AI客户端失败: {e}")
            self.client = None
        
        # 如果所有初始化方式都失败，确保客户端为None并记录日志
        if not self.client:
            print(f"[LLMClient] 所有智谱AI客户端初始化方式均失败，将使用备用评估逻辑")
    
    @retry_on_error(max_retries=3, delay=1.0, backoff=2.0)
    def _call_api(self, prompt: str, model: str = "glm-4.7") -> str:
        """调用智谱AI API"""
        # 检查客户端是否初始化
        if not self.client:
            # 客户端未初始化，直接使用备用逻辑
            return self._fallback_evaluation(prompt)
        
        try:
            # 使用官方SDK调用智谱AI API
            response = self.client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "你是一个记忆系统的评估助手，负责评估对话内容的价值和重要性。"},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=65536,
                temperature=1.0
            )
            
            # 获取完整回复
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"[LLMClient] API调用失败: {e}")
            # 使用备用逻辑
            return self._fallback_evaluation(prompt)
    
    def _fallback_evaluation(self, prompt: str) -> str:
        """备用评估逻辑"""
        if "内容价值评估" in prompt or "评分" in prompt:
            # 简单的价值评估逻辑
            content = prompt.split("内容：")[1].split("评分：")[0].strip()
            if len(content) < 3:
                return "0"
            elif any(word in content for word in ["重要", "会议", "项目", "电话", "号码"]):
                return "8"
            elif any(word in content for word in ["你好", "天气", "忙", "工作"]):
                return "5"
            else:
                return "3"
        elif "搜索查询增强" in prompt:
            # 简单的查询增强逻辑
            query = prompt.split("原始查询：")[1].split("增强查询：")[0].strip()
            return query  # 简单返回原查询
        else:
            return ""
    
    def _get_cache_key(self, prompt: str) -> str:
        """生成缓存键"""
        return hashlib.md5(prompt.encode()).hexdigest()
    

    
    def evaluate_content_value(self, content: str) -> float:
        """评估内容价值"""
        prompt = f"请评估以下对话内容的价值，从0到10评分，0表示无意义，10表示非常重要。\n内容：{content}\n评分："
        response = self.get_response(prompt)
        try:
            return float(response)
        except:
            return 5.0
    
    def calculate_importance_score(self, content: str) -> float:
        """计算内容重要性"""
        prompt = f"请评估以下对话内容的重要性，从0到10评分，0表示不重要，10表示非常重要。\n内容：{content}\n评分："
        response = self.get_response(prompt)
        try:
            return float(response)
        except:
            return 5.0
    
    def enhance_search_query(self, query: str) -> str:
        """增强搜索查询"""
        prompt = f"请增强以下搜索查询，使其更准确地表达语义，保持核心含义不变。\n原始查询：{query}\n增强查询："
        response = self.get_response(prompt)
        return response or query
    
    def clear_cache(self):
        """清理缓存"""
        self.cache.clear()
    
    def _cleanup_cache(self):
        """清理过期缓存和限制缓存大小"""
        current_time = time.time()
        
        # 清理过期缓存
        expired_keys = []
        for key, data in self.cache.items():
            if current_time - data["timestamp"] > self.cache_expiry:
                expired_keys.append(key)
        
        for key in expired_keys:
            del self.cache[key]
        
        # 如果缓存仍然过大，删除最旧的缓存
        if len(self.cache) > self.max_cache_size:
            # 按时间排序缓存
            sorted_cache = sorted(self.cache.items(), key=lambda x: x[1]["timestamp"])
            # 删除最旧的缓存
            for key, _ in sorted_cache[:len(self.cache) - self.max_cache_size]:
                del self.cache[key]
    
    def get_response(self, prompt: str) -> str:
        """获取大模型响应，带缓存"""
        cache_key = self._get_cache_key(prompt)
        
        # 清理过期缓存
        self._cleanup_cache()
        
        # 检查缓存
        if cache_key in self.cache:
            cached_data = self.cache[cache_key]
            if time.time() - cached_data["timestamp"] < self.cache_expiry:
                return cached_data["response"]
        
        # 调用API
        try:
            response = self._call_api(prompt)
            # 更新缓存
            self.cache[cache_key] = {
                "response": response,
                "timestamp": time.time()
            }
            return response
        except Exception as e:
            # 只在真正的API调用失败时显示错误
            # 客户端未初始化的情况已经在_call_api中处理
            if self.client:
                print(f"[LLMClient] API调用失败: {e}")
            # 返回默认值
            return "5"  # 默认中等价值
    
    def batch_get_responses(self, prompts: List[str]) -> List[str]:
        """批量获取大模型响应"""
        responses = []
        uncached_prompts = []
        uncached_indices = []
        
        # 检查缓存
        for i, prompt in enumerate(prompts):
            cache_key = self._get_cache_key(prompt)
            if cache_key in self.cache:
                cached_data = self.cache[cache_key]
                if time.time() - cached_data["timestamp"] < self.cache_expiry:
                    responses.append(cached_data["response"])
                    continue
            # 未命中缓存
            uncached_prompts.append(prompt)
            uncached_indices.append(i)
            responses.append("5")  # 先填充默认值
        
        # 如果所有请求都命中缓存，直接返回
        if not uncached_prompts:
            return responses
        
        # 处理未命中缓存的请求
        # 由于智谱AI API可能不支持真正的批量请求，这里模拟批量处理
        for i, prompt in zip(uncached_indices, uncached_prompts):
            try:
                response = self._call_api(prompt)
                responses[i] = response
                # 更新缓存
                cache_key = self._get_cache_key(prompt)
                self.cache[cache_key] = {
                    "response": response,
                    "timestamp": time.time()
                }
            except Exception as e:
                if self.client:
                    print(f"[LLMClient] 批量API调用失败: {e}")
                # 保持默认值
        
        # 清理缓存
        self._cleanup_cache()
        
        return responses

# ============ 记忆系统核心类 ============
class ZyantineMemorySystem:
    """基于memo0的自衍体记忆系统"""

    def __init__(self,
                 base_url: Optional[str] = None,
                 api_key: Optional[str] = None,
                 zhipu_api_key: Optional[str] = None,
                 user_id: str = "default",
                 session_id: str = "default",
                 memo0_config: Optional[Dict[str, Any]] = None) -> None:
        """初始化记忆系统"""
        # 从配置管理器获取配置
        from config.config_manager import ConfigManager
        config_manager = ConfigManager()
        config = config_manager.get()

        # 使用传入的配置或从配置文件读取
        if memo0_config:
            self.memo0_config = memo0_config
        else:
            self.memo0_config = config.memory.memo0_config

        # 从配置中获取 LLM 和 Embedder 配置
        llm_config = self.memo0_config.get("llm", {}).get("config", {})
        embedder_config = self.memo0_config.get("embedder", {}).get("config", {})

        # 优先使用传入参数，否则使用配置文件中的值
        self.base_url: str = base_url or llm_config.get("base_url", DEFAULT_BASE_URL)
        self.api_key: str = api_key or llm_config.get("api_key", DEFAULT_API_KEY)
        self.user_id: str = user_id
        self.session_id: str = session_id

        # 获取智谱AI API配置
        self.zhipu_api_key: str = zhipu_api_key or config.api.providers.get("zhipu", {}).get("api_key", "")

        # 初始化大模型客户端（用于记忆评估，使用智谱AI）
        self.llm_client = LLMClient(api_key=self.zhipu_api_key)

        # 初始化memo0记忆系统，添加异常处理
        try:
            self.memory: Memory = self._initialize_memo0()
            print("[记忆系统] Mem0记忆系统初始化成功")
        except Exception as e:
            print(f"[记忆系统] Mem0记忆系统初始化失败: {e}")
            print("[记忆系统] 警告：将使用简化的内存记忆系统作为替代")
            # 创建一个简单的内存记忆系统作为替代
            self.memory = None
            self.memory_store = []

        self.semantic_memory_map: Dict[str, Dict[str, Any]] = {}
        self.strategic_tags: List[str] = []
        
        # 添加查询增强缓存
        self.query_enhancement_cache: Dict[str, str] = {}
        self.query_cache_ttl: int = 3600  # 缓存1小时
        self.query_cache_last_cleanup: float = time.time()
        
        # 语义记忆地图配置
        self.max_semantic_map_size: int = 5000  # 最大语义记忆地图大小
        self.semantic_map_cleanup_threshold: float = 0.3  # 清理阈值（低于此值的记忆会被清理）

        # 统计信息
        self.stats: Dict[str, Any] = {
            "total_memories": 0,
            "by_type": {},
            "access_counts": {},
            "tags_distribution": {}
        }

        print(f"[记忆系统] 初始化完成，用户ID: {user_id}，会话ID: {session_id}")
        print(f"[记忆系统] LLM配置: provider={self.memo0_config.get('llm', {}).get('provider')}, model={llm_config.get('model')}")
        print(f"[记忆系统] Embedder配置: provider={self.memo0_config.get('embedder', {}).get('provider')}, model={embedder_config.get('model')}")
        print(f"[记忆系统] LLM客户端初始化完成")

    def _initialize_memo0(self) -> Memory:
        """初始化memo0框架"""
        # 使用从配置文件读取的配置
        config: Dict[str, Any] = self.memo0_config.copy()

        # 验证和设置API密钥
        # 优先使用传入的api_key，否则从环境变量获取
        api_key_to_use = self.api_key
        if not api_key_to_use:
            api_key_to_use = os.getenv("OPENAI_API_KEY", "") or os.getenv("ZHIPU_API_KEY", "")
            if not api_key_to_use:
                print("[记忆系统] 警告：未提供API密钥，将使用虚拟密钥进行初始化")
                # 使用虚拟密钥通过验证，但实际调用时会失败
                # 不过这允许系统初始化，后续功能可能会受限
                api_key_to_use = "dummy-api-key"

        # 确保配置格式正确
        if "llm" not in config:
            config["llm"] = {
                "provider": "openai",
                "config": {
                    "model": "gpt-5-nano-2025-08-07",
                    "api_key": api_key_to_use,
                    "base_url": self.base_url
                }
            }
        else:
            # 确保 LLM 配置中有必要的字段
            llm_config = config["llm"]["config"]
            if "api_key" not in llm_config:
                llm_config["api_key"] = api_key_to_use
            if "base_url" not in llm_config:
                llm_config["base_url"] = self.base_url

        if "embedder" not in config:
            config["embedder"] = {
                "provider": "openai",
                "config": {
                    "model": "text-embedding-3-large",
                    "api_key": api_key_to_use,
                    "openai_base_url": self.base_url
                }
            }
        else:
            # 确保 Embedder 配置中有必要的字段
            embedder_config = config["embedder"]["config"]
            if "api_key" not in embedder_config:
                embedder_config["api_key"] = api_key_to_use
            if "openai_base_url" not in embedder_config:
                embedder_config["openai_base_url"] = self.base_url

        return Memory.from_config(config)

    # ============ 记忆CRUD操作 ============

    def add_memory(self,
                   content: Union[str, List[Dict[str, Any]]],
                   memory_type: str = "conversation",
                   metadata: Optional[Dict[str, Any]] = None,
                   tags: Optional[List[str]] = None,
                   emotional_intensity: float = 0.5,
                   strategic_value: Optional[Dict[str, Any]] = None,
                   linked_tool: Optional[str] = None) -> str:
        """添加记忆（精简metadata避免过大）"""
        # 验证记忆类型
        valid_memory_types: List[str] = [
            "conversation", "experience", "user_profile", "system_event",
            "knowledge", "emotion", "strategy", "temporal"
        ]
        if memory_type not in valid_memory_types:
            raise MemoryTypeError(
                f"无效的记忆类型: {memory_type}",
                details={"valid_types": valid_memory_types}
            )

        content_str: str = content if isinstance(content, str) else json.dumps(content, ensure_ascii=False)
        
        # 评估内容价值
        content_value = self._evaluate_content_value(content_str)
        
        # 根据记忆类型设置不同的价值阈值
        if memory_type == "conversation":
            # 对话内容设置较低的阈值，确保有意义的对话能够被存储
            if content_value < 4.0:
                print(f"[记忆系统] 过滤低价值对话内容: {content_str[:50]}...")
                return "filtered_content"
        else:
            # 其他类型的记忆保持原有的阈值
            if content_value < 6.0:
                print(f"[记忆系统] 过滤低价值内容: {content_str[:50]}...")
                return "filtered_content"
        
        # 检查重复内容
        duplicate_id = self._check_duplicate_content(content_str, memory_type)
        if duplicate_id:
            print(f"[记忆系统] 检测到重复内容，更新现有记忆: {duplicate_id}")
            # 这里可以实现更新逻辑，暂时返回重复ID
            return duplicate_id
        
        # 计算重要性分数
        importance_score = self._calculate_importance_score(content_str, memory_type)
        
        # 生成记忆ID
        memory_id: str = self._generate_memory_id(content_str, memory_type)

        full_content: Union[str, List[Dict[str, Any]]] = content if isinstance(content, list) else [{"role": "user", "content": content}]

        # 准备精简的metadata
        memory_metadata: Dict[str, Any] = self._prepare_metadata(
            memory_id=memory_id,
            memory_type=memory_type,
            content=content_str,
            metadata=metadata or {},
            tags=tags or [],
            emotional_intensity=emotional_intensity,
            strategic_value=strategic_value or {},
            linked_tool=linked_tool
        )
        
        # 添加重要性分数到metadata
        memory_metadata["importance_score"] = importance_score

        # 检查并进一步压缩metadata
        metadata_size: int = len(json.dumps(memory_metadata))
        if metadata_size > 30000:
            print(f"[记忆系统] 警告：metadata大小({metadata_size})接近限制，进一步精简")
            memory_metadata = self._further_compress_metadata(memory_metadata)

        try:
            # 添加到memo0或简化的内存存储
            if self.memory:
                # 添加到memo0
                self.memory.add(full_content, user_id=self.user_id, metadata=memory_metadata)
            else:
                # 添加到简化的内存存储
                memory_item = {
                    "content": full_content,
                    "user_id": self.user_id,
                    "metadata": memory_metadata
                }
                self.memory_store.append(memory_item)
        except Exception as e:
            if "exceeds max length" in str(e):
                print(f"[记忆系统] 错误：metadata仍然过大，使用最小化版本")
                minimal_metadata: Dict[str, Any] = {
                    "memory_id": memory_id,
                    "memory_type": memory_type,
                    "session_id": self.session_id,
                    "created_at": datetime.now().isoformat(),
                    "importance_score": importance_score
                }
                try:
                    if self.memory:
                        self.memory.add(full_content, user_id=self.user_id, metadata=minimal_metadata)
                    else:
                        memory_item = {
                            "content": full_content,
                            "user_id": self.user_id,
                            "metadata": minimal_metadata
                        }
                        self.memory_store.append(memory_item)
                except Exception as minimal_error:
                    raise MemorySizeExceededError(
                        f"无法添加记忆：metadata大小超过限制",
                        memory_id=memory_id,
                        size=metadata_size,
                        max_size=30000
                    ) from minimal_error
            else:
                raise MemoryStorageError(
                    f"添加记忆失败: {str(e)}",
                    memory_id=memory_id,
                    details={"content_length": len(content_str)}
                ) from e

        # 更新语义记忆地图
        self._update_semantic_memory(memory_id, {
            "full_metadata": memory_metadata,
            "content_preview": content_str[:500] if content_str else "",
            "tags": tags or [],
            "emotional_intensity": emotional_intensity,
            "strategic_value": strategic_value or {},
            "linked_tool": linked_tool,
            "importance_score": importance_score,
            "content_value": content_value
        })

        # 更新统计信息
        self._update_stats(memory_type, tags or [])

        print(f"[记忆系统] 添加记忆成功，ID: {memory_id}，类型: {memory_type}，重要性: {importance_score:.2f}，价值: {content_value:.2f}")
        return memory_id

    def _generate_memory_id(self, content: str, memory_type: str) -> str:
        """生成记忆ID"""
        timestamp: str = datetime.now().strftime("%Y%m%d%H%M%S")
        content_hash: str = hashlib.md5(content.encode()).hexdigest()[:8]
        return f"{memory_type}_{timestamp}_{content_hash}"

    def _prepare_metadata(self,
                          memory_id: str,
                          memory_type: str,
                          content: str,
                          metadata: Dict[str, Any],
                          tags: List[str],
                          emotional_intensity: float,
                          strategic_value: Dict[str, Any],
                          linked_tool: Optional[str]) -> Dict[str, Any]:
        """准备精简的元数据"""
        # 精简strategic_value
        simplified_strategic: Dict[str, Any] = {
            "score": strategic_value.get("score", 0),
            "level": strategic_value.get("level", "低")
        } if strategic_value and isinstance(strategic_value, dict) else {"score": 0, "level": "低"}

        base_metadata: Dict[str, Any] = {
            "memory_id": memory_id,
            "memory_type": memory_type,
            "session_id": self.session_id,
            "created_at": datetime.now().isoformat(),
            "tags": tags[:5] if tags else [],
            "emotional_intensity": round(emotional_intensity, 2),
            "strategic_score": simplified_strategic.get("score", 0),
            "strategic_level": simplified_strategic.get("level", "低"),
            "linked_tool": linked_tool,
            "source": "zyantine_system",
            "content_length": len(content),
            "content_preview": content[:200] if content else "",
            "user_id": self.user_id
        }

        # 合并必要的外部元数据
        for key in ["interaction_id", "context_keys", "action_plan_keys", "growth_result_keys"]:
            if key in metadata:
                if key in ["context_keys", "action_plan_keys", "growth_result_keys"] and isinstance(metadata[key],
                                                                                                    dict):
                    base_metadata[key] = list(metadata[key].keys())[:5]
                else:
                    base_metadata[key] = metadata[key]

        return base_metadata

    def _further_compress_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """进一步压缩metadata"""
        minimal: Dict[str, Any] = {
            "memory_id": metadata.get("memory_id", ""),
            "memory_type": metadata.get("memory_type", ""),
            "session_id": metadata.get("session_id", ""),
            "created_at": metadata.get("created_at", ""),
            "content_length": metadata.get("content_length", 0)
        }

        if "tags" in metadata and metadata["tags"]:
            minimal["tags"] = metadata["tags"][:2]

        return minimal

    def _update_semantic_memory(self, memory_id: str, memory_data: Dict[str, Any]) -> None:
        """更新语义记忆地图"""
        self.semantic_memory_map[memory_id] = {
            "metadata": memory_data,
            "access_count": 0,
            "last_accessed": None,
            "strategic_score": memory_data.get("strategic_score", 0),
            "importance_score": memory_data.get("importance_score", 5.0),
            "content_value": memory_data.get("content_value", 5.0)
        }
        print(f"[记忆系统] 更新语义记忆地图: {memory_id}")
        
        # 检查语义记忆地图大小，超过限制时清理
        if len(self.semantic_memory_map) > self.max_semantic_map_size:
            self._cleanup_semantic_memory()
    
    def _cleanup_semantic_memory(self):
        """清理语义记忆地图中的低价值记忆"""
        if len(self.semantic_memory_map) <= self.max_semantic_map_size:
            return
        
        # 计算需要清理的数量
        cleanup_count = len(self.semantic_memory_map) - int(self.max_semantic_map_size * 0.8)  # 清理到80%容量
        
        # 按价值排序记忆
        memory_values = []
        for memory_id, memory_data in self.semantic_memory_map.items():
            # 计算综合价值分数
            importance = memory_data.get("importance_score", 5.0)
            content_value = memory_data.get("content_value", 5.0)
            access_count = memory_data.get("access_count", 0)
            
            total_value = importance * 0.5 + content_value * 0.3 + (access_count / 10.0) * 0.2
            memory_values.append((memory_id, total_value))
        
        # 按价值从低到高排序
        memory_values.sort(key=lambda x: x[1])
        
        # 清理低价值记忆
        cleaned_count = 0
        for memory_id, value in memory_values:
            if cleaned_count >= cleanup_count or value >= self.semantic_map_cleanup_threshold:
                break
            
            if memory_id in self.semantic_memory_map:
                del self.semantic_memory_map[memory_id]
                # 同时从访问统计中删除
                if memory_id in self.stats.get("access_counts", {}):
                    del self.stats["access_counts"][memory_id]
                cleaned_count += 1
        
        print(f"[记忆系统] 清理语义记忆地图，删除 {cleaned_count} 个低价值记忆，剩余 {len(self.semantic_memory_map)} 个")

    def _update_stats(self, memory_type: str, tags: List[str]) -> None:
        """更新统计信息"""
        self.stats["total_memories"] += 1

        # 按类型统计
        self.stats["by_type"][memory_type] = self.stats["by_type"].get(memory_type, 0) + 1

        # 标签分布
        for tag in tags:
            self.stats["tags_distribution"][tag] = self.stats["tags_distribution"].get(tag, 0) + 1
            if tag not in self.strategic_tags:
                self.strategic_tags.append(tag)

    # ============ 记忆检索 ============

    def _cleanup_query_cache(self):
        """清理过期的查询缓存"""
        current_time = time.time()
        if current_time - self.query_cache_last_cleanup > self.query_cache_ttl:
            # 简单清理，实际项目中可以实现更复杂的过期机制
            self.query_enhancement_cache.clear()
            self.query_cache_last_cleanup = current_time

    def _enhance_search_query(self, query: str) -> str:
        """增强搜索查询"""
        # 清理过期缓存
        self._cleanup_query_cache()
        
        # 检查缓存
        if query in self.query_enhancement_cache:
            return self.query_enhancement_cache[query]
        
        try:
            enhanced_query = self.llm_client.enhance_search_query(query)
            if enhanced_query and enhanced_query != query:
                print(f"[记忆系统] 增强搜索查询: {query} -> {enhanced_query}")
                # 缓存结果
                self.query_enhancement_cache[query] = enhanced_query
                return enhanced_query
        except Exception as e:
            print(f"[记忆系统] 查询增强失败: {e}")
        
        # 缓存原始查询
        self.query_enhancement_cache[query] = query
        return query

    def _adjust_similarity_threshold(self, memory_type: Optional[str]) -> float:
        """根据记忆类型调整相似度阈值"""
        # 为不同类型的记忆设置不同的阈值
        thresholds = {
            "user_profile": 0.8,      # 用户档案需要更高的准确性
            "knowledge": 0.75,        # 知识记忆需要较高的准确性
            "strategy": 0.75,         # 策略记忆需要较高的准确性
            "experience": 0.7,         # 经历记忆
            "conversation": 0.6,       # 对话记忆
            "emotion": 0.65,           # 情感记忆
            "system_event": 0.5,       # 系统事件
            "temporal": 0.6            # 时间相关记忆
        }
        
        return thresholds.get(memory_type, 0.7)  # 默认阈值

    def search_memories(self,
                        query: str,
                        memory_type: Optional[str] = None,
                        tags: Optional[List[str]] = None,
                        limit: int = 5,
                        similarity_threshold: float = 0.7,
                        rerank: bool = True) -> List[Dict[str, Any]]:
        """搜索记忆"""
        try:
            # 增强搜索查询
            enhanced_query = self._enhance_search_query(query)
            
            # 如果使用简化的内存存储
            if not self.memory:
                # 简化的搜索逻辑
                results = []
                for memory_item in self.memory_store:
                    # 检查用户ID
                    if memory_item["user_id"] != self.user_id:
                        continue
                    
                    metadata = memory_item["metadata"]
                    
                    # 检查记忆类型
                    if memory_type and metadata.get("memory_type") != memory_type:
                        continue
                    
                    # 检查标签
                    if tags:
                        memory_tags = metadata.get("tags", [])
                        if not any(tag in memory_tags for tag in tags):
                            continue
                    
                    # 简单的文本匹配
                    content_str = str(memory_item["content"])
                    if enhanced_query.lower() in content_str.lower() or \
                       any(enhanced_query.lower() in str(value).lower() for value in metadata.values()):
                        results.append({
                            "memory_id": metadata.get("memory_id"),
                            "content": self._extract_content_from_memory(memory_item["content"]),
                            "metadata": metadata,
                            "similarity_score": 0.8,  # 默认相似度
                            "memory_type": metadata.get("memory_type"),
                            "tags": metadata.get("tags", []),
                            "emotional_intensity": metadata.get("emotional_intensity", 0.5),
                            "strategic_value": metadata.get("strategic_value", {}),
                            "linked_tool": metadata.get("linked_tool"),
                            "created_at": metadata.get("created_at"),
                            "access_count": self.semantic_memory_map.get(metadata.get("memory_id"), {}).get("access_count", 0),
                            "importance_score": metadata.get("importance_score", 5.0)
                        })
                
                # 限制结果数量
                return results[:limit]
            
            # 根据记忆类型调整相似度阈值
            adjusted_threshold = self._adjust_similarity_threshold(memory_type)
            
            # 构建元数据过滤器
            metadata_filter: Dict[str, Any] = {}
            if memory_type:
                metadata_filter["memory_type"] = memory_type
            if tags:
                metadata_filter["tags"] = tags

            # 执行搜索
            search_results: Dict[str, Any] = self.memory.search(
                enhanced_query,
                user_id=self.user_id,
                limit=limit * 2,  # 获取更多结果用于过滤
                rerank=rerank
            )

            processed_results: List[Dict[str, Any]] = []
            for hit in search_results.get("results", []):
                memory_data: Any = hit.get("memory", {})
                metadata: Dict[str, Any] = hit.get("metadata", {})
                score: float = hit.get("score", 0)

                # 使用调整后的阈值过滤
                if score < adjusted_threshold:
                    continue

                # 更新访问统计
                memory_id: Optional[str] = metadata.get("memory_id")
                if memory_id and memory_id in self.semantic_memory_map:
                    self.semantic_memory_map[memory_id]["access_count"] += 1
                    self.semantic_memory_map[memory_id]["last_accessed"] = datetime.now().isoformat()
                    self.stats["access_counts"][memory_id] = self.stats["access_counts"].get(memory_id, 0) + 1

                # 构建结果
                result: Dict[str, Any] = {
                    "memory_id": memory_id,
                    "content": self._extract_content_from_memory(memory_data),
                    "metadata": metadata,
                    "similarity_score": score,
                    "memory_type": metadata.get("memory_type"),
                    "tags": metadata.get("tags", []),
                    "emotional_intensity": metadata.get("emotional_intensity", 0.5),
                    "strategic_value": metadata.get("strategic_value", {}),
                    "linked_tool": metadata.get("linked_tool"),
                    "created_at": metadata.get("created_at"),
                    "access_count": self.semantic_memory_map.get(memory_id, {}).get("access_count", 0),
                    "importance_score": metadata.get("importance_score", 5.0)
                }

                processed_results.append(result)

            # 优化排序：结合相似度、重要性和访问频率
            processed_results.sort(key=lambda x: (
                x["similarity_score"] * 0.6 +  # 相似度权重
                (x.get("importance_score", 5.0) / 10.0) * 0.2 +  # 重要性权重
                (x.get("access_count", 0) / 10.0) * 0.1 +  # 访问频率权重
                (1.0 if x.get("memory_type") == "experience" else 0.0) * 0.1  # 类型权重
            ), reverse=True)
            
            return processed_results[:limit]
        except MemoryException:
            raise
        except Exception as e:
            raise MemorySearchError(
                f"搜索记忆失败: {str(e)}",
                details={"query": query, "memory_type": memory_type, "tags": tags, "limit": limit}
            ) from e

    def _extract_content_from_memory(self, memory_data: Any) -> str:
        """从记忆数据中提取内容"""
        if isinstance(memory_data, list):
            return "\n".join(
                [f"{msg.get('role', '')}: {msg.get('content', '')}" for msg in memory_data if isinstance(msg, dict)])
        elif isinstance(memory_data, str):
            return memory_data
        else:
            return str(memory_data)
    
    def _evaluate_content_value(self, content: str) -> float:
        """评估内容价值"""
        # 简单的长度检查
        if len(content.strip()) < 3:
            return 0.0
        
        # 检查是否为无意义内容
        meaningless_patterns = ["嗯...", "啊...", "哦...", "这个...", "那个..."]
        if any(pattern in content for pattern in meaningless_patterns):
            return 0.0
        
        # 调用大模型评估
        try:
            score = self.llm_client.evaluate_content_value(content)
            return score
        except Exception as e:
            print(f"[记忆系统] 内容价值评估失败: {e}")
            # 使用优化的备用算法
            content_stripped = content.strip()
            content_length = len(content_stripped)
            
            # 检查是否包含问题或命令
            question_patterns = ["?", "吗", "呢", "为什么", "怎么", "如何", "什么", "哪里", "何时"]
            command_patterns = ["请", "帮", "给", "做", "写", "创建", "生成", "分析"]
            
            # 检查是否包含关键词
            has_question = any(pattern in content_stripped for pattern in question_patterns)
            has_command = any(pattern in content_stripped for pattern in command_patterns)
            
            # 基于内容特征评分
            if content_length < 10:
                # 非常短的内容
                if has_question or has_command:
                    return 4.5  # 短问题或命令有一定价值
                else:
                    return 3.0
            elif content_length < 50:
                # 中等长度内容
                if has_question or has_command:
                    return 6.0  # 问题或命令价值较高
                else:
                    return 5.0
            else:
                # 长内容
                return 7.0
    
    def batch_evaluate_content_value(self, contents: List[str]) -> List[float]:
        """批量评估内容价值"""
        scores = []
        
        for content in contents:
            scores.append(self._evaluate_content_value(content))
        
        return scores
    
    def _calculate_importance_score(self, content: str, memory_type: str) -> float:
        """计算内容重要性"""
        # 基于记忆类型的基础分数
        type_scores = {
            "user_profile": 9.0,
            "knowledge": 8.0,
            "strategy": 7.0,
            "experience": 7.0,
            "conversation": 7.0,  # 提高对话记忆的基础分数
            "emotion": 6.0,
            "system_event": 4.0,
            "temporal": 3.0
        }
        base_score = type_scores.get(memory_type, 6.0)  # 默认分数也提高
        
        # 检查是否包含重要关键词
        important_keywords = ["重要", "会议", "项目", "电话", "号码", " deadline ", "截止日期", "负责人"]
        keyword_bonus = 0.0
        if any(keyword in content for keyword in important_keywords):
            keyword_bonus = 3.0
        
        # 调用大模型评估
        try:
            score = self.llm_client.calculate_importance_score(content)
            # 结合基础分数和关键词加成
            final_score = (score + base_score + keyword_bonus) / 3
            return min(10.0, final_score)
        except Exception as e:
            print(f"[记忆系统] 重要性评估失败: {e}")
            # 使用备用算法
            final_score = base_score + keyword_bonus
            return min(10.0, final_score)
    
    def _check_duplicate_content(self, content: str, memory_type: str) -> Optional[str]:
        """检查重复内容"""
        # 简单的重复检查实现
        # 实际项目中可以使用更复杂的相似度算法
        content_hash = hashlib.md5(content.encode()).hexdigest()
        
        # 检查语义记忆地图中是否有相似内容
        for memory_id, memory_data in self.semantic_memory_map.items():
            preview = memory_data.get("metadata", {}).get("content_preview", "")
            if preview:
                preview_hash = hashlib.md5(preview.encode()).hexdigest()
                # 简单的哈希比较，实际项目中应该使用更复杂的相似度算法
                if content_hash == preview_hash:
                    return memory_id
        
        return None

    def find_conversations(self,
                           query: str,
                           session_id: Optional[str] = None,
                           limit: int = 5) -> List[Dict[str, Any]]:
        """查找相关对话"""
        target_session: str = session_id or self.session_id
        return self.search_memories(query=query, memory_type="conversation", limit=limit)

    def get_recent_conversations(self,
                                 session_id: Optional[str] = None,
                                 limit: int = 5) -> List[Dict[str, Any]]:
        """按时间顺序获取最近对话（优化版 - 使用元数据过滤减少数据传输）"""
        target_session: str = session_id or self.session_id

        from datetime import datetime, timedelta

        # 如果使用简化的内存存储
        if not self.memory:
            # 简化的最近对话获取逻辑
            recent_conversations: List[Dict[str, Any]] = []
            for memory_item in self.memory_store:
                # 检查用户ID
                if memory_item["user_id"] != self.user_id:
                    continue
                
                metadata = memory_item["metadata"]
                
                # 检查记忆类型
                if metadata.get("memory_type") != "conversation":
                    continue
                
                # 检查会话ID
                if metadata.get("session_id") != target_session:
                    continue
                
                memory_data = memory_item["content"]
                created_at: str = metadata.get("created_at", datetime.now().isoformat())
                
                recent_conversations.append({
                    "memory_id": metadata.get("memory_id"),
                    "content": self._extract_content_from_memory(memory_data),
                    "metadata": metadata,
                    "memory_type": metadata.get("memory_type"),
                    "tags": metadata.get("tags", []),
                    "emotional_intensity": metadata.get("emotional_intensity", 0.5),
                    "strategic_value": metadata.get("strategic_value", {}),
                    "linked_tool": metadata.get("linked_tool"),
                    "created_at": created_at,
                    "access_count": self.semantic_memory_map.get(metadata.get("memory_id"), {}).get("access_count", 0)
                })
            
            # 按时间排序
            recent_conversations.sort(key=lambda x: x["created_at"], reverse=True)
            return recent_conversations[:limit]

        try:
            # 优化：使用搜索功能代替get_all，减少数据传输量
            # 添加时间范围限制（最近30天）
            time_threshold: str = (datetime.now() - timedelta(days=30)).isoformat()

            # 使用搜索API，带时间范围过滤
            search_results: Dict[str, Any] = self.memory.search(
                query="conversation",
                user_id=self.user_id,
                limit=limit * 3  # 获取更多结果用于过滤
            )

            recent_conversations: List[Dict[str, Any]] = []
            for hit in search_results.get("results", []):
                memory_data: Any = hit.get("memory", {})
                metadata: Dict[str, Any] = hit.get("metadata", {})

                # 元数据过滤
                if metadata.get("memory_type") != "conversation":
                    continue

                if metadata.get("session_id") != target_session:
                    continue

                created_at: str = metadata.get("created_at", datetime.now().isoformat())

                # 过滤时间范围
                if created_at < time_threshold:
                    continue

                # 添加类型检查
                if not isinstance(memory_data, dict) and not isinstance(memory_data, (list, str)):
                    print(f"[记忆系统] 警告：记忆数据类型异常: {type(memory_data)}")
                    continue

                recent_conversations.append({
                    "memory_id": metadata.get("memory_id"),
                    "content": self._extract_content_from_memory(memory_data),
                    "metadata": metadata,
                    "memory_type": metadata.get("memory_type"),
                    "tags": metadata.get("tags", []),
                    "emotional_intensity": metadata.get("emotional_intensity", 0.5),
                    "strategic_value": metadata.get("strategic_value", {}),
                    "linked_tool": metadata.get("linked_tool"),
                    "created_at": created_at,
                    "access_count": self.semantic_memory_map.get(metadata.get("memory_id"), {}).get("access_count", 0)
                })

            # 按时间排序
            recent_conversations.sort(key=lambda x: x["created_at"], reverse=True)
            return recent_conversations[:limit]

        except MemoryException:
            raise
        except Exception as e:
            print(f"[记忆系统] 获取最近对话失败: {e}")
            # 回退到原有实现
            try:
                all_memories: List[Any] = self.memory.get_all(user_id=self.user_id)
                recent_conversations: List[Dict[str, Any]] = []
                for memory_item in all_memories:
                    if not isinstance(memory_item, dict):
                        continue

                    memory_data: Any = memory_item.get("memory", {})
                    metadata: Dict[str, Any] = memory_item.get("metadata", {})

                    if metadata.get("memory_type") != "conversation":
                        continue

                    if metadata.get("session_id") != target_session:
                        continue

                    created_at: str = metadata.get("created_at", datetime.now().isoformat())

                    recent_conversations.append({
                        "memory_id": metadata.get("memory_id"),
                        "content": self._extract_content_from_memory(memory_data),
                        "metadata": metadata,
                        "memory_type": metadata.get("memory_type"),
                        "tags": metadata.get("tags", []),
                        "emotional_intensity": metadata.get("emotional_intensity", 0.5),
                        "strategic_value": metadata.get("strategic_value", {}),
                        "linked_tool": metadata.get("linked_tool"),
                        "created_at": created_at,
                        "access_count": self.semantic_memory_map.get(metadata.get("memory_id"), {}).get("access_count", 0)
                    })

                recent_conversations.sort(key=lambda x: x["created_at"], reverse=True)
                return recent_conversations[:limit]
            except MemoryException:
                raise
            except Exception as fallback_error:
                raise MemoryRetrievalError(
                    f"获取最近对话失败（主方法和回退方法都失败）: {str(fallback_error)}",
                    details={"session_id": target_session, "limit": limit}
                ) from fallback_error

    def find_experiences(self,
                         context: Dict[str, Any],
                         limit: int = 3) -> List[Dict[str, Any]]:
        """查找相关经历记忆"""
        query_parts: List[str] = []
        if "user_input" in context:
            query_parts.append(context["user_input"])
        if "user_emotion" in context:
            query_parts.append(f"情绪 {context['user_emotion']}")
        if "topic" in context:
            query_parts.append(f"话题 {context['topic']}")

        return self.search_memories(query=" ".join(query_parts), memory_type="experience", limit=limit)

    # ============ 记忆管理 ============

    def get_memory(self, memory_id: str, include_full_data: bool = False) -> Optional[Dict[str, Any]]:
        """获取特定记忆"""
        if memory_id in self.semantic_memory_map:
            search_results: List[Dict[str, Any]] = self.search_memories(query=memory_id, limit=1)
            if search_results:
                memory_info: Dict[str, Any] = search_results[0]
                # 确保更新访问统计
                if memory_id in self.semantic_memory_map:
                    self.semantic_memory_map[memory_id]["access_count"] += 1
                    self.semantic_memory_map[memory_id]["last_accessed"] = datetime.now().isoformat()
                    self.stats["access_counts"][memory_id] = self.stats["access_counts"].get(memory_id, 0) + 1
                    print(f"[记忆系统] 更新记忆访问统计: {memory_id} (访问次数: {self.semantic_memory_map[memory_id]['access_count']})")

                if include_full_data:
                    full_data: Optional[Dict[str, Any]] = self._load_full_memory_from_local(memory_id)
                    if full_data:
                        memory_info["full_data"] = full_data

                return memory_info
        return None

    def _load_full_memory_from_local(self, memory_id: str) -> Optional[Dict[str, Any]]:
        """从本地文件加载完整记忆数据"""
        try:
            local_file: str = f"./memory_backup/{self.user_id}/{self.session_id}/{memory_id}.json"
            if os.path.exists(local_file):
                with open(local_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
        except:
            pass
        return None
    
    def _sort_memories_by_value(self) -> List[Tuple[str, float]]:
        """按记忆价值排序"""
        memory_values = []
        
        for memory_id, memory_data in self.semantic_memory_map.items():
            # 计算记忆价值分数
            importance_score = memory_data.get("importance_score", 5.0)
            access_count = memory_data.get("access_count", 0)
            content_value = memory_data.get("content_value", 5.0)
            
            # 计算综合价值分数
            # 重要性占50%，访问次数占30%，内容价值占20%
            total_value = importance_score * 0.5 + access_count * 0.3 + content_value * 0.2
            memory_values.append((memory_id, total_value))
        
        # 按价值排序，从低到高
        memory_values.sort(key=lambda x: x[1])
        return memory_values
    
    def cleanup_memory(self, max_memories: int = 100, min_value_threshold: float = 3.0) -> int:
        """清理低价值记忆"""
        # 计算当前记忆数量
        current_count = len(self.semantic_memory_map)
        if current_count <= max_memories:
            print(f"[记忆系统] 记忆数量({current_count})未超过限制({max_memories})，无需清理")
            return 0
        
        # 按价值排序
        sorted_memories = self._sort_memories_by_value()
        
        # 计算需要清理的数量
        memories_to_clean = current_count - max_memories
        cleaned_count = 0
        
        print(f"[记忆系统] 开始清理低价值记忆，当前数量: {current_count}，目标数量: {max_memories}")
        
        # 清理低价值记忆
        for memory_id, value in sorted_memories:
            if cleaned_count >= memories_to_clean:
                break
            
            if value < min_value_threshold:
                print(f"[记忆系统] 清理低价值记忆: {memory_id} (价值: {value:.2f})")
                # 从语义记忆地图中删除
                if memory_id in self.semantic_memory_map:
                    del self.semantic_memory_map[memory_id]
                # 从访问统计中删除
                if memory_id in self.stats["access_counts"]:
                    del self.stats["access_counts"][memory_id]
                cleaned_count += 1
        
        # 更新统计信息
        self.stats["total_memories"] = len(self.semantic_memory_map)
        
        print(f"[记忆系统] 清理完成，共清理 {cleaned_count} 个低价值记忆，剩余 {len(self.semantic_memory_map)} 个记忆")
        return cleaned_count

    # ============ 批量操作 ============

    def add_conversation_batch(self, conversations: List[Dict[str, Any]]) -> List[str]:
        """批量添加对话"""
        return [
            self.add_memory(content=conv, memory_type="conversation", tags=["对话", "批量导入"],
                            emotional_intensity=conv.get("emotional_intensity", 0.5))
            for conv in conversations
        ]

    def store_conversation(self, user_input: str, system_response: str, emotional_intensity: float = 0.5, tags: Optional[List[str]] = None) -> str:
        """主动存储对话信息到向量库
        
        Args:
            user_input: 用户输入内容
            system_response: 系统回复内容
            emotional_intensity: 情感强度（0.0-1.0）
            tags: 标签列表
            
        Returns:
            存储的记忆ID，如果存储失败则返回失败原因
        """
        try:
            # 构建对话内容格式
            conversation_content = [
                {"role": "user", "content": user_input},
                {"role": "assistant", "content": system_response}
            ]
            
            # 合并标签
            conversation_tags = tags or []
            conversation_tags.extend(["对话", "主动存储"])
            
            # 存储对话
            memory_id = self.add_memory(
                content=conversation_content,
                memory_type="conversation",
                tags=conversation_tags,
                emotional_intensity=emotional_intensity,
                strategic_value={"score": 6.0, "level": "中"}
            )
            
            print(f"[记忆系统] 主动存储对话成功，ID: {memory_id}")
            return memory_id
        except Exception as e:
            error_message = f"存储对话失败: {e}"
            print(f"[记忆系统] {error_message}")
            return error_message

    def import_user_profile(self, profile_data: Dict[str, Any]) -> List[str]:
        """导入用户档案数据"""
        memory_ids: List[str] = []

        # 导入用户记忆
        if "memories" in profile_data:
            for memory in profile_data["memories"]:
                memory_ids.append(self.add_memory(
                    content=memory.get("content", ""),
                    memory_type="experience",
                    tags=memory.get("tags", ["用户记忆"]),
                    emotional_intensity=memory.get("emotional_intensity", 0.5),
                    strategic_value=memory.get("strategic_value", {})
                ))

        # 导入用户特征
        if "personality_traits" in profile_data:
            traits_text: str = json.dumps(profile_data["personality_traits"], ensure_ascii=False)
            memory_ids.append(self.add_memory(
                content=f"用户性格特征: {traits_text}",
                memory_type="user_profile",
                tags=["性格特征", "用户档案"]
            ))

        return memory_ids

    # ============ 统计与分析 ============

    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        # 计算最常用的标签
        top_tags: List[Tuple[str, int]] = sorted(self.stats["tags_distribution"].items(), key=lambda x: x[1], reverse=True)[:10]

        # 计算最常访问的记忆
        top_accessed: List[Tuple[str, Dict[str, Any]]] = sorted(self.semantic_memory_map.items(),
                              key=lambda x: x[1].get("access_count", 0), reverse=True)[:5]

        top_accessed_formatted: List[Dict[str, Any]] = [{
            "memory_id": mem_id,
            "access_count": mem_data.get("access_count", 0),
            "last_accessed": mem_data.get("last_accessed"),
            "strategic_score": mem_data.get("strategic_score", 0)
        } for mem_id, mem_data in top_accessed]

        return {
            "total_memories": self.stats["total_memories"],
            "memory_types": self.stats["by_type"],
            "top_tags": dict(top_tags),
            "top_accessed_memories": top_accessed_formatted,
            "strategic_tags_count": len(self.strategic_tags),
            "user_id": self.user_id,
            "session_id": self.session_id,
            "semantic_map_size": len(self.semantic_memory_map)
        }

    # ============ 共鸣记忆功能 ============

    def find_resonant_memory(self, context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """寻找共鸣记忆"""
        query_text: str = self._build_resonance_query(context)
        if not query_text:
            return None

        # 搜索相关经历记忆
        similar_experiences: List[Dict[str, Any]] = self.find_experiences(context, limit=5)
        
        # 搜索相关对话
        similar_conversations: List[Dict[str, Any]] = self.find_conversations(query_text, limit=3)
        
        # 合并并过滤结果
        all_matches: List[Dict[str, Any]] = []
        
        # 添加经历记忆
        for experience in similar_experiences:
            if experience.get("similarity_score", 0) > 0.7:
                all_matches.append(experience)
        
        # 添加对话记忆
        for conversation in similar_conversations:
            if conversation.get("similarity_score", 0) > 0.6:
                all_matches.append(conversation)
        
        # 按相似度排序
        all_matches.sort(key=lambda x: x.get("similarity_score", 0), reverse=True)
        
        # 限制返回数量
        top_matches: List[Dict[str, Any]] = all_matches[:3]
        
        if not top_matches:
            return None
        
        # 如果只有一个匹配，使用原有方法
        if len(top_matches) == 1:
            return self._build_tactical_package(top_matches[0], context)
        else:
            # 使用新方法构建组合记忆包
            return self._build_combined_tactical_package(top_matches, context)

    def _build_resonance_query(self, context: Dict[str, Any]) -> str:
        """构建共鸣查询"""
        parts: List[str] = []
        if "user_input" in context:
            parts.append(context["user_input"])
        if "user_emotion" in context:
            parts.append(f"情绪: {context['user_emotion']}")
        if "topic" in context:
            parts.append(f"话题: {context['topic']}")
        return " ".join(parts) if parts else ""

    def _build_tactical_package(self, memory_match: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """构建战术信息包"""
        metadata: Dict[str, Any] = memory_match.get("metadata", {})

        return {
            "triggered_memory": memory_match.get("content", "未知记忆"),
            "memory_id": memory_match.get("memory_id"),
            "relevance_score": memory_match.get("similarity_score", 0),
            "source": metadata.get("source", "unknown"),
            "tags": metadata.get("tags", []),
            "strategic_value": metadata.get("strategic_value", {}),
            "linked_tool": metadata.get("linked_tool"),
            "emotional_intensity": metadata.get("emotional_intensity", 0.5),
            "risk_assessment": self._assess_memory_risk(metadata),
            "recommended_actions": self._generate_recommendations(metadata, context),
            "timestamp": datetime.now().isoformat(),
            "retrieval_method": "memo0_vector_search"
        }

    def _build_combined_tactical_package(self, memory_matches: List[Dict[str, Any]], context: Dict[str, Any]) -> Dict[str, Any]:
        """构建组合战术信息包"""
        if not memory_matches:
            return None

        # 按创建时间和相似度排序
        def sort_key(memory):
            created_at = memory.get("metadata", {}).get("created_at", "1970-01-01T00:00:00")
            similarity_score = memory.get("similarity_score", 0)
            return (similarity_score, created_at)

        sorted_matches = sorted(memory_matches, key=sort_key, reverse=True)

        # 提取记忆内容
        memory_contents = []
        all_tags = set()
        emotional_intensities = []
        strategic_values = []
        risk_factors = []

        for memory in sorted_matches:
            content = memory.get("content", "")
            if content:
                memory_contents.append(content)
            
            # 收集标签
            tags = memory.get("tags", [])
            all_tags.update(tags)
            
            # 收集情感强度
            emotional_intensity = memory.get("emotional_intensity", 0.5)
            emotional_intensities.append(emotional_intensity)
            
            # 收集战略价值
            strategic_value = memory.get("strategic_value", {})
            if strategic_value:
                strategic_values.append(strategic_value)
            
            # 收集风险因素
            metadata = memory.get("metadata", {})
            risk_assessment = self._assess_memory_risk(metadata)
            if risk_assessment.get("high_risk_factors"):
                risk_factors.extend(risk_assessment.get("high_risk_factors", []))

        # 构建组合记忆内容
        combined_content = "\n---\n".join(memory_contents)
        
        # 计算平均情感强度
        avg_emotional_intensity = sum(emotional_intensities) / len(emotional_intensities) if emotional_intensities else 0.5
        
        # 计算平均相似度分数
        avg_relevance_score = sum(m.get("similarity_score", 0) for m in sorted_matches) / len(sorted_matches) if sorted_matches else 0
        
        # 生成组合标签
        combined_tags = list(all_tags)[:10]  # 限制标签数量
        
        # 生成组合风险评估
        combined_risk = self._assess_combined_memory_risk(risk_factors)
        
        # 生成组合建议
        combined_recommendations = self._generate_combined_recommendations(sorted_matches, context)

        return {
            "triggered_memory": combined_content,
            "memory_ids": [m.get("memory_id") for m in sorted_matches],
            "relevance_score": avg_relevance_score,
            "source": "combined_memories",
            "tags": combined_tags,
            "strategic_value": strategic_values,
            "linked_tool": sorted_matches[0].get("linked_tool"),
            "emotional_intensity": avg_emotional_intensity,
            "risk_assessment": combined_risk,
            "recommended_actions": combined_recommendations,
            "timestamp": datetime.now().isoformat(),
            "retrieval_method": "combined_memo0_search",
            "memory_count": len(sorted_matches),
            "memory_sources": [m.get("memory_type", "unknown") for m in sorted_matches]
        }

    def _assess_memory_risk(self, metadata: Dict) -> Dict[str, Any]:
        """评估记忆风险"""
        risk_score = 0
        high_risk_factors = []

        # 高风险标签
        high_risk_tags = ["创伤", "背叛", "失败", "痛苦"]
        tags = metadata.get("tags", [])

        for tag in tags:
            if tag in high_risk_tags:
                risk_score += 3
                high_risk_factors.append(tag)

        # 情感强度影响
        if metadata.get("emotional_intensity", 0.5) > 0.8:
            risk_score += 2

        # 确定风险级别
        if risk_score >= 5:
            level = "高"
        elif risk_score >= 3:
            level = "中"
        elif risk_score >= 1:
            level = "低"
        else:
            level = "极低"

        return {"level": level, "score": risk_score, "high_risk_factors": high_risk_factors}

    def _generate_recommendations(self, metadata: Dict[str, Any], context: Dict[str, Any]) -> List[str]:
        """生成使用建议"""
        tags: List[str] = metadata.get("tags", [])
        recommendations: List[str] = []

        if any(tag in tags for tag in ["成就", "成功"]):
            recommendations.append("可安全提及以激活积极情绪")
        if any(tag in tags for tag in ["创伤", "痛苦"]):
            recommendations.append("高风险区域，谨慎使用")
        if any(tag in tags for tag in ["学习", "成长"]):
            recommendations.append("适合用于激励场景")

        return recommendations if recommendations else ["常规记忆，可灵活使用"]

    def _assess_combined_memory_risk(self, risk_factors: List[str]) -> Dict[str, Any]:
        """评估组合记忆风险"""
        risk_score = 0
        unique_risk_factors = list(set(risk_factors))
        
        # 计算风险分数
        for factor in unique_risk_factors:
            if factor in ["创伤", "背叛", "失败", "痛苦"]:
                risk_score += 2
        
        # 确定风险级别
        if risk_score >= 4:
            level = "高"
        elif risk_score >= 2:
            level = "中"
        elif risk_score >= 1:
            level = "低"
        else:
            level = "极低"

        return {
            "level": level,
            "score": risk_score,
            "high_risk_factors": unique_risk_factors
        }

    def _generate_combined_recommendations(self, memories: List[Dict[str, Any]], context: Dict[str, Any]) -> List[str]:
        """生成组合记忆使用建议"""
        all_recommendations = []
        
        # 收集所有记忆的建议
        for memory in memories:
            metadata = memory.get("metadata", {})
            recommendations = self._generate_recommendations(metadata, context)
            all_recommendations.extend(recommendations)
        
        # 去重并排序
        unique_recommendations = list(set(all_recommendations))
        
        # 添加组合记忆特定建议
        if len(memories) > 2:
            unique_recommendations.append("多记忆组合，提供全面上下文")
        
        # 限制建议数量
        return unique_recommendations[:5]

    # ============ 导出与工具方法 ============

    def export_memories(self, file_path: str, format_type: str = "json") -> bool:
        """导出记忆"""
        try:
            export_data = {
                "metadata": {
                    "export_time": datetime.now().isoformat(),
                    "user_id": self.user_id,
                    "session_id": self.session_id,
                    "total_memories": self.stats["total_memories"]
                },
                "semantic_memory_map": self.semantic_memory_map,
                "statistics": self.stats,
                "strategic_tags": self.strategic_tags
            }

            with open(file_path, 'w', encoding='utf-8') as f:
                if format_type == "json":
                    json.dump(export_data, f, ensure_ascii=False, indent=2)
                else:
                    f.write(str(export_data))

            print(f"[记忆系统] 记忆导出成功: {file_path}")
            return True
        except Exception as e:
            print(f"[记忆系统] 记忆导出失败: {e}")
            return False

    def clear_cache(self):
        """清理缓存"""
        self.semantic_memory_map.clear()
        self.strategic_tags.clear()
        self.stats = {"total_memories": 0, "by_type": {}, "access_counts": {}, "tags_distribution": {}}
        print("[记忆系统] 缓存已清理")

    def test_connection(self) -> bool:
        """测试连接"""
        try:
            test_id = self.add_memory(content="连接测试", memory_type="test", tags=["测试"])
            results = self.search_memories("连接测试", limit=1)

            if results and len(results) > 0:
                print("[记忆系统] 连接测试成功")
                return True
            else:
                print("[记忆系统] 连接测试失败")
                return False
        except Exception as e:
            print(f"[记忆系统] 连接测试异常: {e}")
            return False
