# API文档

## 概述

本文档描述了自衍体AI系统的主要API接口，包括系统核心接口、记忆系统接口和大模型服务接口。

## 系统核心接口

### ZyantineFacade

`ZyantineFacade`是系统的主要入口点，提供了与系统交互的高级接口。

#### 创建系统实例

```python
from zyantine_facade import ZyantineFacade, create_zyantine

# 使用默认配置创建
facade = create_zyantine(
    config_file="config/llm_config.json",
    session_id="default"
)

# 或直接创建
facade = ZyantineFacade(
    config_path="config/llm_config.json",
    user_profile=None,
    self_profile=None,
    session_id="default"
)
```

#### 核心方法

##### `chat(user_input: str) -> str`

处理用户输入并返回系统响应。

**参数**：
- `user_input` - 用户输入文本

**返回值**：
- 系统生成的响应文本

**示例**：
```python
response = facade.chat("你好，今天天气怎么样？")
print(response)  # 输出系统响应
```

##### `get_status() -> dict`

获取系统当前状态。

**返回值**：
- 包含系统状态信息的字典

**示例**：
```python
status = facade.get_status()
print(status)
# 输出: {"session_id": "default", "memory_stats": {...}, "conversation_history_length": 5}
```

##### `save_memory() -> bool`

保存当前记忆系统。

**返回值**：
- 保存是否成功

**示例**：
```python
success = facade.save_memory()
print(f"保存记忆: {'成功' if success else '失败'}")
```

##### `cleanup() -> bool`

清理系统资源和低价值记忆。

**返回值**：
- 清理是否成功

**示例**：
```python
success = facade.cleanup()
print(f"清理记忆: {'成功' if success else '失败'}")
```

##### `shutdown() -> None`

关闭系统，释放资源。

**示例**：
```python
facade.shutdown()
```

## 记忆系统接口

### ZyantineMemorySystem

`ZyantineMemorySystem`提供了记忆管理的核心功能。

#### 创建记忆系统

```python
from memory.memory_store import ZyantineMemorySystem

memory_system = ZyantineMemorySystem(
    user_id="user_123",
    session_id="session_456"
)
```

#### 核心方法

##### `add_memory(content: Union[str, List[Dict[str, Any]]], memory_type: str = "conversation", **kwargs) -> str`

添加新记忆。

**参数**：
- `content` - 记忆内容
- `memory_type` - 记忆类型
- `tags` - 记忆标签
- `metadata` - 元数据
- `emotional_intensity` - 情感强度

**返回值**：
- 记忆ID

**示例**：
```python
memory_id = memory_system.add_memory(
    content="今天学习了Python编程",
    memory_type="knowledge",
    tags=["学习", "编程"],
    emotional_intensity=0.7
)
print(f"记忆ID: {memory_id}")
```

##### `search_memories(query: str, memory_type: Optional[str] = None, **kwargs) -> List[Dict[str, Any]]`

搜索记忆。

**参数**：
- `query` - 搜索查询
- `memory_type` - 记忆类型
- `tags` - 标签过滤
- `limit` - 结果数量限制
- `similarity_threshold` - 相似度阈值

**返回值**：
- 搜索结果列表

**示例**：
```python
results = memory_system.search_memories(
    query="Python编程",
    memory_type="knowledge",
    limit=5
)
for result in results:
    print(f"记忆: {result['content']}, 相似度: {result['similarity_score']}")
```

##### `get_memory(memory_id: str, include_full_data: bool = False) -> Optional[Dict[str, Any]]`

获取特定记忆。

**参数**：
- `memory_id` - 记忆ID
- `include_full_data` - 是否包含完整数据

**返回值**：
- 记忆信息字典

**示例**：
```python
memory = memory_system.get_memory("knowledge_20240101_abc123")
print(memory)
```

##### `cleanup_memory(max_memories: int = 100, min_value_threshold: float = 3.0) -> int`

清理低价值记忆。

**参数**：
- `max_memories` - 最大记忆数量
- `min_value_threshold` - 最小价值阈值

**返回值**：
- 清理的记忆数量

**示例**：
```python
cleaned_count = memory_system.cleanup_memory(max_memories=50)
print(f"清理了 {cleaned_count} 个记忆")
```

## 大模型服务接口

### LLMSerivce

`LLMSerivce`是大模型服务的抽象接口。

#### 实现类

- `OpenAIService` - OpenAI服务实现
- `DeepSeekService` - DeepSeek服务实现
- `ZhipuService` - 智谱AI服务实现

#### 核心方法

##### `generate(prompt: str, **kwargs) -> str`

生成文本响应。

**参数**：
- `prompt` - 提示文本
- `model` - 模型名称
- `temperature` - 温度参数
- `max_tokens` - 最大令牌数

**返回值**：
- 生成的文本

**示例**：
```python
from api.llm_service_factory import create_llm_service

# 创建大模型服务
llm_service = create_llm_service("openai")

# 生成响应
response = llm_service.generate(
    prompt="写一首关于春天的诗",
    model="gpt-4",
    temperature=0.7,
    max_tokens=500
)
print(response)
```

##### `batch_generate(prompts: List[str], **kwargs) -> List[str]`

批量生成文本响应。

**参数**：
- `prompts` - 提示文本列表
- `model` - 模型名称

**返回值**：
- 生成的文本列表

**示例**：
```python
prompts = ["写一首关于春天的诗", "写一首关于秋天的诗"]
responses = llm_service.batch_generate(prompts, model="gpt-4")
for i, response in enumerate(responses):
    print(f"响应 {i+1}: {response}")
```

## 提示词引擎接口

### PromptEngine

`PromptEngine`负责生成和管理提示词。

#### 核心方法

##### `generate_prompt(template_name: str, **kwargs) -> str`

根据模板生成提示词。

**参数**：
- `template_name` - 模板名称
- `**kwargs` - 模板参数

**返回值**：
- 生成的提示词

**示例**：
```python
from api.prompt_engine import PromptEngine

prompt_engine = PromptEngine()
prompt = prompt_engine.generate_prompt(
    "conversation",
    user_input="你好",
    context="这是一个友好的对话"
)
print(prompt)
```

## 异常处理

系统使用统一的异常处理机制，主要异常类包括：

- `ZyantineException` - 基础异常
- `APIException` - API调用异常
- `MemoryException` - 记忆系统异常
- `ConfigException` - 配置异常

**示例**：
```python
try:
    response = facade.chat("你好")
except APIException as e:
    print(f"API错误: {e}")
except MemoryException as e:
    print(f"记忆系统错误: {e}")
except Exception as e:
    print(f"未知错误: {e}")
```

## 配置API

### ConfigManager

`ConfigManager`负责管理系统配置。

#### 核心方法

##### `load(config_file: Optional[str] = None) -> SystemConfig`

加载配置。

**参数**：
- `config_file` - 配置文件路径

**返回值**：
- 系统配置对象

**示例**：
```python
from config.config_manager import ConfigManager

config_manager = ConfigManager()
config = config_manager.load("config/llm_config.json")
print(f"系统名称: {config.system_name}")
```

##### `update(updates: Dict[str, Any]) -> SystemConfig`

更新配置。

**参数**：
- `updates` - 配置更新字典

**返回值**：
- 更新后的配置对象

**示例**：
```python
updates = {
    "api": {
        "provider": "openai"
    }
}
updated_config = config_manager.update(updates)
print(f"更新后的提供商: {updated_config.api.provider}")
```

## 监控和日志API

### Logger

系统提供了结构化日志功能。

#### 核心方法

```python
from utils.logger import get_structured_logger, LogContext

logger = get_structured_logger("my_module")

# 记录信息
logger.info("系统启动", service="zyantine", version="1.0.0")

# 使用上下文
with LogContext(user_id="user_123", session_id="session_456"):
    logger.info("用户操作", action="login")
```

## 性能监控API

### PerformanceLogger

用于监控系统性能。

#### 核心方法

```python
from utils.logger import get_performance_logger

perf_logger = get_performance_logger("api")

# 记录操作性能
perf_logger.start_timer("api_call")
# 执行操作
response = llm_service.generate("你好")
elapsed = perf_logger.stop_timer("api_call")
print(f"API调用耗时: {elapsed:.3f}秒")

# 获取性能统计
metrics = perf_logger.get_metrics()
print(metrics)
```

## 输入输出示例

### 完整对话示例

**输入**：
```python
from zyantine_facade import create_zyantine

# 创建系统实例
facade = create_zyantine(session_id="user_123")

# 对话
responses = []
inputs = ["你好，我是小明", "今天天气怎么样？", "我想学习Python编程"]

for user_input in inputs:
    print(f"用户: {user_input}")
    response = facade.chat(user_input)
    responses.append(response)
    print(f"系统: {response}")
    print()

# 获取状态
status = facade.get_status()
print("系统状态:")
print(status)

# 保存记忆
facade.save_memory()

# 关闭系统
facade.shutdown()
```

**输出**：
```
用户: 你好，我是小明
系统: 你好，小明！很高兴认识你。我是你的AI助手，有什么可以帮助你的吗？

用户: 今天天气怎么样？
系统: 抱歉，我无法实时获取天气信息。不过你可以告诉我你所在的城市，我可以给你一些关于天气的一般性建议。

用户: 我想学习Python编程
系统: 学习Python编程是个很好的选择！Python是一种易于学习且功能强大的编程语言，广泛应用于数据科学、Web开发、人工智能等领域。我可以给你一些学习建议，你是完全的初学者吗？

系统状态:
{
    "session_id": "user_123",
    "memory_stats": {
        "total_memories": 3,
        "by_type": {
            "conversation": 3
        },
        "top_tags": {},
        "top_accessed_memories": [],
        "strategic_tags_count": 0
    },
    "conversation_history_length": 3,
    "processing_mode": "standard"
}
```

## 错误处理示例

**输入**：
```python
from zyantine_facade import create_zyantine
from utils.exception_handler import handle_error

try:
    # 创建系统实例（使用无效配置）
    facade = create_zyantine(config_file="invalid_config.json")
    
except Exception as e:
    # 处理错误
    handled_error = handle_error(e, context="创建系统实例")
    print(f"错误: {handled_error}")
    print(f"错误代码: {handled_error.error_code}")
    print(f"错误类别: {handled_error.category}")
```

**输出**：
```
错误: [CFG-000] 配置错误: 未找到配置文件，使用默认配置 (类别: config, 级别: error)
错误代码: CFG-000
错误类别: config
```
