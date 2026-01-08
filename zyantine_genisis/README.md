# ZyantineAI - 自衍体AI系统

一个具有自我进化能力的智能对话系统，支持多LLM提供商、记忆管理、认知流程和协议引擎。

## ✨ 核心特性

### 🤖 多LLM提供商支持
- **OpenAI**: GPT系列模型
- **DeepSeek**: 深度求索大模型
- **Anthropic**: Claude系列模型
- **智谱AI**: GLM系列模型
- **月之暗面**: Moonshot系列模型
- **阿里云**: 通义千问系列模型
- **百度文心**: 文心一言系列模型

### 🧠 智能记忆系统
- **Memo0记忆系统**: 高效的语义记忆存储和检索
- **分层存储优化**: 自动管理不同层级的记忆存储
- **记忆去重**: 智能检测和去除重复记忆
- **安全管理**: 支持加密和访问控制
- **性能监控**: 实时监控记忆系统性能

### 🔄 认知流程管理
- **核心身份识别**: 维持一致的系统人格
- **认知流程编排**: 多阶段认知处理流程
- **元认知能力**: 自我反思和调整
- **欲望引擎**: 动态调整系统目标和动机

### 🛡️ 协议引擎
- **事实检查**: 确保回答的准确性
- **长度控制**: 自动调整响应长度
- **表达协议**: 规范化表达方式
- **冲突解决**: 处理协议间的冲突

### 🌐 OpenAI兼容API
- **标准API格式**: 完全兼容OpenAI Chat Completions API
- **流式响应**: 支持实时流式输出
- **多轮对话**: 自动管理对话上下文
- **健康检查**: 完整的系统监控和状态报告

## 📦 安装

### 环境要求
- Python 3.11+
- pip

### 安装依赖

```bash
cd zyantine_genisis
pip install -r requirements.txt
```

## 🚀 快速开始

### 1. 配置LLM提供商

编辑 `config/llm_config.json` 文件，配置你想要使用的LLM提供商：

```json
{
  "api": {
    "enabled": true,
    "provider": "deepseek",
    "api_key": "your-api-key-here",
    "base_url": "https://api.deepseek.com",
    "chat_model": "deepseek-chat",
    "embedding_model": "text-embedding-3-large",
    "providers": {
      "openai": {
        "enabled": false,
        "api_key": "your-openai-key",
        "base_url": "https://api.openai.com/v1",
        "chat_model": "gpt-5-nano-2025-08-07"
      },
      "deepseek": {
        "enabled": true,
        "api_key": "your-deepseek-key",
        "base_url": "https://api.deepseek.com",
        "chat_model": "deepseek-chat"
      }
    }
  }
}
```

### 2. 启动系统

#### 交互模式

```bash
python main.py --interactive
```

#### API服务模式

```bash
python api_server.py
```

服务启动后，访问 http://localhost:8000/docs 查看API文档。

#### 批量处理模式

```bash
python main.py --batch input.txt --output output.json
```

### 3. 使用示例

#### Python客户端

```python
from zyantine_facade import create_zyantine

# 创建系统实例
facade = create_zyantine(
    api_key="your-api-key",
    session_id="user-123"
)

# 发送消息
response = facade.chat("你好，请介绍一下自己")
print(response)

# 保存记忆
facade.save_memory()

# 查看状态
status = facade.get_status()
print(status)
```

#### API调用

```python
import requests

# 非流式请求
response = requests.post(
    "http://localhost:8000/v1/chat/completions",
    json={
        "model": "zyantine-v1",
        "messages": [
            {"role": "user", "content": "你好"}
        ],
        "temperature": 0.7,
        "max_tokens": 1000
    }
)

result = response.json()
print(result["choices"][0]["message"]["content"])

# 流式请求
response = requests.post(
    "http://localhost:8000/v1/chat/completions/stream",
    json={
        "model": "zyantine-v1",
        "messages": [
            {"role": "user", "content": "你好"}
        ],
        "temperature": 0.7,
        "max_tokens": 1000
    },
    stream=True
)

for line in response.iter_lines():
    if line:
        print(line.decode('utf-8'))
```

## 📖 详细文档

### LLM提供商配置

#### 切换到OpenAI

```json
{
  "api": {
    "provider": "openai",
    "api_key": "your-openai-key",
    "base_url": "https://api.openai.com/v1",
    "chat_model": "gpt-5-nano-2025-08-07",
    "providers": {
      "openai": {
        "enabled": true,
        "api_key": "your-openai-key",
        "base_url": "https://api.openai.com/v1",
        "chat_model": "gpt-5-nano-2025-08-07"
      },
      "deepseek": {
        "enabled": false
      }
    }
  }
}
```

#### 使用自定义base_url

```json
{
  "api": {
    "provider": "openai",
    "base_url": "https://openkey.cloud/v1",
    "chat_model": "gpt-5-nano-2025-08-07"
  }
}
```

### 记忆系统配置

```json
{
  "memory": {
    "system_type": "memo0",
    "max_memories": 10000,
    "retrieval_limit": 5,
    "similarity_threshold": 0.7,
    "enable_semantic_cache": true,
    "cache_ttl": 300,
    "backup_interval": 3600,
    "backup_path": "./memory_backups"
  }
}
```

### 认知流程配置

```json
{
  "processing": {
    "mode": "standard",
    "enable_stage_parallelism": false,
    "max_conversation_history": 1000,
    "enable_real_time_analysis": true,
    "stage_configs": {
      "preprocess": {"enabled": true, "timeout": 5},
      "memory_retrieval": {"enabled": true, "cache_results": true},
      "desire_update": {"enabled": true, "update_frequency": "always"},
      "cognitive_flow": {"enabled": true, "max_iterations": 3},
      "reply_generation": {"enabled": true, "fallback_to_template": true},
      "protocol_review": {"enabled": true, "strict_mode": false}
    }
  }
}
```

### 协议引擎配置

```json
{
  "protocols": {
    "enable_fact_check": true,
    "enable_length_regulation": true,
    "enable_expression_protocol": true,
    "fact_check_strictness": 0.8,
    "max_response_length": 2000,
    "min_response_length": 50,
    "allow_uncertainty_phrases": true
  }
}
```

## 🧪 测试

### 运行所有测试

```bash
# 使用测试运行器
python run_tests.py --all

# 或直接运行测试文件
python tests/api/test_llm_provider.py
```

### 运行特定类别的测试

```bash
# 记忆模块测试
python run_tests.py --category memory

# 认知模块测试
python run_tests.py --category cognition

# 协议模块测试
python run_tests.py --category protocols

# API模块测试
python run_tests.py --category api

# 系统级测试
python run_tests.py --category system
```

### 运行快速测试

```bash
python run_tests.py --quick
```

## 📊 项目结构

```
zyantine_genisis/
├── api/                    # API和LLM服务
│   ├── llm_provider.py    # LLM提供商枚举和配置
│   ├── llm_service.py     # LLM服务抽象基类
│   ├── llm_service_factory.py  # LLM服务工厂
│   ├── openai_service.py  # OpenAI兼容服务
│   └── service_provider.py  # 服务提供商管理
├── cognition/             # 认知模块
│   ├── core_identity.py   # 核心身份识别
│   ├── cognitive_flow_manager.py  # 认知流程管理
│   ├── desire_engine.py   # 欲望引擎
│   └── meta_cognition.py  # 元认知
├── config/                # 配置管理
│   ├── config_manager.py  # 配置管理器
│   └── llm_config.json    # LLM配置文件
├── core/                  # 系统核心
│   ├── system_core.py     # 系统核心
│   ├── processing_pipeline.py  # 处理管道
│   └── stage_handlers.py  # 阶段处理器
├── memory/                # 记忆系统
│   ├── memory_manager.py  # 记忆管理器
│   └── memory_store.py    # 记忆存储
├── protocols/             # 协议引擎
│   ├── fact_checker.py    # 事实检查
│   ├── length_regulator.py # 长度控制
│   └── expression_validator.py # 表达验证
├── utils/                 # 工具函数
│   ├── logger.py          # 日志工具
│   ├── metrics.py         # 指标收集
│   └── error_handler.py   # 错误处理
├── examples/              # 示例代码
│   ├── llm_provider_usage.py
│   └── memory_demo.py
├── tests/                 # 测试文件
│   ├── api/               # API测试
│   ├── cognition/         # 认知测试
│   ├── memory/            # 记忆测试
│   └── protocols/         # 协议测试
├── main.py                # 主入口
├── api_server.py          # API服务器
├── zyantine_facade.py     # 外观模式入口
└── README.md              # 本文件
```

## 🔧 命令行参数

### main.py

```bash
python main.py [OPTIONS]

选项:
  --config, -c PATH        配置文件路径
  --api-key, -k KEY        OpenAI API密钥
  --session, -s ID         会话ID (默认: default)
  --interactive, -i        交互模式
  --batch, -b FILE         批量处理输入文件
  --output, -o FILE        批量处理输出文件
  --profile, -p FILE       用户配置文件
  --self-profile, -sp FILE 自衍体配置文件
  --status                 显示系统状态
  --save                   保存记忆系统
  --cleanup                清理记忆
  --log-level LEVEL        日志级别 (DEBUG/INFO/WARNING/ERROR/CRITICAL)
```

### api_server.py

```bash
python api_server.py [OPTIONS]

选项:
  --host HOST              监听地址 (默认: 0.0.0.0)
  --port PORT              监听端口 (默认: 8000)
  --api-key KEY            OpenAI API密钥
  --session ID             会话ID (默认: default)
```

## 🌟 核心概念

### 外观模式 (Facade Pattern)

`ZyantineFacade` 类提供了简化的系统接口，隐藏了复杂的内部实现：

```python
from zyantine_facade import create_zyantine

facade = create_zyantine(api_key="your-key", session_id="user-123")
response = facade.chat("你好")
```

### 工厂模式 (Factory Pattern)

`LLMServiceFactory` 负责创建不同提供商的LLM服务：

```python
from api.llm_service_factory import LLMServiceFactory

service = LLMServiceFactory.create_service("deepseek", config)
```

### 记忆生命周期

记忆系统支持完整的生命周期管理：
- 创建 → 存储 → 检索 → 更新 → 归档 → 删除

### 认知流程

标准认知流程包括以下阶段：
1. 预处理 (Preprocess)
2. 记忆检索 (Memory Retrieval)
3. 欲望更新 (Desire Update)
4. 认知流程 (Cognitive Flow)
5. 回复生成 (Reply Generation)
6. 协议审查 (Protocol Review)

## 🔍 监控和日志

### 日志文件

日志文件存储在 `logs/` 目录下：
- `zyantine.log`: 主系统日志
- `facade_*.log`: 外观模式日志
- `api_service_provider_*.log`: API服务日志
- `llm_service_*.log`: LLM服务日志
- 其他模块日志...

### 健康检查

```bash
curl http://localhost:8000/health
```

### 系统状态

```python
status = facade.get_status()
print(json.dumps(status, ensure_ascii=False, indent=2))
```

## 🤝 与语音RTC项目集成

### 集成流程

1. **语音识别**: RTC项目识别用户语音，转换为文本
2. **API调用**: 将识别的文本发送到 `/v1/chat/completions` 端点
3. **获取响应**: 接收AI生成的文本响应
4. **语音合成**: 将响应文本传递给RTC项目的语音合成模块

### 流式响应集成（推荐）

```python
import requests
import json

def process_voice_input_stream(text: str, callback):
    """处理语音输入并流式返回AI响应"""
    response = requests.post(
        "http://localhost:8000/v1/chat/completions/stream",
        json={
            "model": "zyantine-v1",
            "messages": [
                {"role": "user", "content": text}
            ],
            "temperature": 0.7,
            "max_tokens": 1000
        },
        stream=True,
        timeout=30
    )
    
    full_response = ""
    for line in response.iter_lines():
        if line:
            line_str = line.decode('utf-8')
            if line_str == "data: [DONE]":
                break
            
            if line_str.startswith("data: "):
                data = json.loads(line_str[6:])
                delta = data["choices"][0].get("delta", {})
                content = delta.get("content", "")
                
                if content:
                    full_response += content
                    callback(content)  # 将内容片段传递给回调函数
    
    return full_response

# 使用示例
def on_response_chunk(chunk: str):
    """处理响应片段的回调函数"""
    print(chunk, end='', flush=True)
    # 在这里可以调用语音合成模块

user_text = "今天天气怎么样？"
process_voice_input_stream(user_text, on_response_chunk)
```

## 📝 开发指南

### 添加新的LLM提供商

1. 在 `api/llm_provider.py` 中添加新的枚举值
2. 在 `api/llm_service.py` 中实现新的服务类
3. 在 `api/llm_service_factory.py` 中添加工厂方法
4. 在 `config/llm_config.json` 中添加配置

### 添加新的认知阶段

1. 在 `core/stage_handlers.py` 中实现新的处理器
2. 在 `config/llm_config.json` 中添加阶段配置
3. 更新处理管道以包含新阶段

### 添加新的协议

1. 在 `protocols/` 目录下创建新的协议类
2. 在 `protocols/protocol_engine.py` 中注册协议
3. 在配置文件中启用协议

## 🐛 故障排查

### 服务无法启动

1. 检查端口是否被占用：`lsof -i :8000`
2. 检查依赖是否完整：`pip install -r requirements.txt`
3. 查看日志输出

### API调用失败

1. 检查服务是否正常运行：访问 http://localhost:8000/health
2. 检查请求格式是否正确
3. 查看服务日志

### LLM提供商连接失败

1. 检查API密钥是否正确
2. 检查base_url是否正确
3. 检查网络连接
4. 查看LLM服务日志

## 📄 许可证

本项目遵循原有项目的许可证。

## 🙏 致谢

感谢所有为本项目做出贡献的开发者。

## 📧 联系方式

如有问题或建议，请通过以下方式联系：
- 提交Issue
- 发送Pull Request
- 参与讨论

---

**ZyantineAI** - 让AI真正具有自我进化能力 🚀
