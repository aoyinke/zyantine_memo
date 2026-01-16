# 自衍体AI系统 (Zyantine AI System)

## 系统概述

自衍体AI系统是一个基于大语言模型的智能对话系统，具有记忆能力、自我认知和持续学习能力。系统采用模块化设计，支持多种大模型提供商，并具有灵活的配置和扩展能力。

## 系统架构

### 核心组件

1. **API层** - 处理外部请求和大模型交互
   - `api/llm_service.py` - 大模型服务接口
   - `api/openai_service.py` - OpenAI服务实现
   - `api/prompt_engine.py` - 提示词引擎

2. **记忆系统** - 管理和检索记忆
   - `memory/memory_store.py` - 记忆存储实现
   - `memory/memory_manager.py` - 记忆管理

3. **认知系统** - 实现自我认知和决策
   - `cognition/cognitive_flow_manager.py` - 认知流程管理
   - `cognition/core_identity.py` - 核心身份
   - `cognition/desire_engine.py` - 欲望引擎

4. **配置管理** - 管理系统配置
   - `config/config_manager.py` - 配置管理器
   - `config/llm_config.json` - 配置文件

5. **核心系统** - 系统核心逻辑
   - `core/system_core.py` - 系统核心
   - `core/processing_pipeline.py` - 处理管道

6. **工具模块** - 通用工具
   - `utils/logger.py` - 日志系统
   - `utils/exception_handler.py` - 异常处理
   - `utils/metrics.py` -  metrics 指标

### 系统流程图

```
用户输入 → API层 → 处理管道 → 记忆检索 → 认知处理 → 大模型调用 → 响应生成 → 用户
```

## 快速开始

### 环境要求

- Python 3.8+
- 依赖包：`pip install -e .`

### 配置

1. **配置文件**：修改 `config/llm_config.json` 文件
2. **环境变量**：设置必要的环境变量

### 运行系统

```bash
# 交互模式
python main.py --interactive

# 批量处理
python main.py --batch input.txt --output output.json

# 查看系统状态
python main.py --status
```

## API文档

### 主要接口

#### ZyantineFacade

```python
from zyantine_facade import ZyantineFacade, create_zyantine

# 创建系统实例
facade = create_zyantine(
    config_file="config/llm_config.json",
    session_id="user_123"
)

# 对话
response = facade.chat("你好，我是用户")
print(response)

# 获取系统状态
status = facade.get_status()
print(status)

# 保存记忆
facade.save_memory()

# 清理记忆
facade.cleanup()
```

#### 记忆系统接口

```python
from memory.memory_store import ZyantineMemorySystem

# 创建记忆系统
memory_system = ZyantineMemorySystem(
    user_id="user_123",
    session_id="session_456"
)

# 添加记忆
memory_id = memory_system.add_memory(
    content="这是一条重要的信息",
    memory_type="knowledge",
    tags=["重要", "信息"]
)

# 搜索记忆
results = memory_system.search_memories(
    query="重要信息",
    memory_type="knowledge",
    limit=5
)
```

## 配置选项

### 环境变量

系统支持以下环境变量配置：

- `ZYANTINE_OPENAI_API_KEY` - OpenAI API密钥
- `ZYANTINE_OPENAI_BASE_URL` - OpenAI API基础URL
- `ZYANTINE_MEMORY_MAX_MEMORIES` - 最大记忆数量
- `ZYANTINE_LOG_LEVEL` - 日志级别

### 配置文件结构

配置文件采用JSON格式，主要包含以下部分：

- `api` - API配置
- `memory` - 记忆系统配置
- `processing` - 处理配置
- `protocols` - 协议配置

## 开发指南

### 扩展系统

1. **添加新的大模型提供商**：
   - 在 `api/` 目录下创建新的服务实现
   - 更新 `api/llm_service_factory.py`

2. **添加新的记忆存储后端**：
   - 实现 `memory/memory_store.py` 中的接口

3. **添加新的认知模块**：
   - 在 `cognition/` 目录下创建新的模块

### 代码风格

- 遵循 PEP 8 代码风格
- 使用类型提示
- 编写详细的文档字符串

## 部署指南

### 本地部署

1. 克隆代码库
2. 安装依赖：`pip install -e .`
3. 配置环境变量
4. 运行系统：`python main.py --interactive`

### 容器部署

```bash
# 构建镜像
docker build -t zyantine .

# 运行容器
docker run -it --env-file .env zyantine
```

## 监控与日志

### 日志系统

系统使用结构化日志，日志文件存储在 `logs/` 目录下。

### 监控指标

系统提供以下监控指标：

- 响应时间
- 记忆使用情况
- API调用频率
- 错误率

## 故障排除

### 常见问题

1. **API密钥错误**：检查环境变量或配置文件中的API密钥
2. **记忆存储错误**：检查向量存储配置
3. **响应缓慢**：检查网络连接和大模型提供商状态

### 日志分析

使用结构化日志分析工具查看 `logs/` 目录下的日志文件。

## 许可证

MIT License

## 贡献

欢迎贡献代码和提出问题！请遵循以下流程：

1. Fork 代码库
2. 创建功能分支
3. 提交更改
4. 创建 Pull Request
