# 自衍体 AI API 服务

符合 OpenAI API 格式的接口服务，为语音 RTC 交互项目提供标准化的 API 接口。

## 功能特性

- ✅ 完全兼容 OpenAI Chat Completions API 格式
- ✅ 支持标准版和增强版两种认知流程
- ✅ 支持非流式和流式响应
- ✅ 支持多轮对话
- ✅ 自动会话管理
- ✅ 健康检查和状态监控
- ✅ 完整的错误处理

## 快速开始

### 1. 安装依赖

```bash
cd zyantine_genisis
pip install -r requirements.txt
```

### 2. 启动 API 服务

```bash
# 使用默认配置启动
python api_server.py

# 指定端口启动
python api_server.py --port 8080

# 指定 API 密钥启动
python api_server.py --api-key "your-api-key"

# 指定会话 ID 启动
python api_server.py --session "user-123"
```

服务启动后，访问 http://localhost:8000/docs 查看完整的 API 文档。

### 3. 测试 API

```bash
# 运行所有测试
python test_api.py

# 测试特定接口
python test_api.py --test chat
python test_api.py --test stream
python test_api.py --test conversation
```

## API 端点

### 健康检查

```
GET /health
```

返回服务健康状态。

### 获取模型列表

```
GET /v1/models
```

返回可用的模型列表：
- `zyantine-v1`: 标准版认知流程
- `zyantine-enhanced`: 增强版认知流程

### 聊天完成（非流式）

```
POST /v1/chat/completions
```

**请求示例：**

```json
{
  "model": "zyantine-v1",
  "messages": [
    {
      "role": "user",
      "content": "你好，请介绍一下自己"
    }
  ],
  "temperature": 0.7,
  "max_tokens": 1000,
  "stream": false
}
```

**响应示例：**

```json
{
  "id": "chatcmpl-abc123",
  "object": "chat.completion",
  "created": 1234567890,
  "model": "zyantine-v1",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "你好！我是自衍体AI系统..."
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 10,
    "completion_tokens": 50,
    "total_tokens": 60
  }
}
```

### 聊天完成（流式）

```
POST /v1/chat/completions/stream
```

**请求示例：**

```json
{
  "model": "zyantine-v1",
  "messages": [
    {
      "role": "user",
      "content": "你好，请介绍一下自己"
    }
  ],
  "temperature": 0.7,
  "max_tokens": 1000
}
```

**响应格式：**

```
data: {"id":"chatcmpl-abc123","object":"chat.completion.chunk","created":1234567890,"model":"zyantine-v1","choices":[{"index":0,"delta":{"role":"assistant"},"finish_reason":null}]}

data: {"id":"chatcmpl-abc123","object":"chat.completion.chunk","created":1234567890,"model":"zyantine-v1","choices":[{"index":0,"delta":{"content":"你好"},"finish_reason":null}]}

data: {"id":"chatcmpl-abc123","object":"chat.completion.chunk","created":1234567890,"model":"zyantine-v1","choices":[{"index":0,"delta":{"content":"！"},"finish_reason":null}]}

data: {"id":"chatcmpl-abc123","object":"chat.completion.chunk","created":1234567890,"model":"zyantine-v1","choices":[{"index":0,"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":10,"completion_tokens":50,"total_tokens":60}}

data: [DONE]
```

## 使用示例

### Python 客户端

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

### cURL 示例

```bash
# 非流式请求
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "zyantine-v1",
    "messages": [
      {"role": "user", "content": "你好"}
    ],
    "temperature": 0.7,
    "max_tokens": 1000
  }'

# 流式请求
curl -X POST http://localhost:8000/v1/chat/completions/stream \
  -H "Content-Type: application/json" \
  -d '{
    "model": "zyantine-v1",
    "messages": [
      {"role": "user", "content": "你好"}
    ],
    "temperature": 0.7,
    "max_tokens": 1000
  }'
```

### JavaScript/Node.js 示例

```javascript
// 非流式请求
const response = await fetch('http://localhost:8000/v1/chat/completions', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
  },
  body: JSON.stringify({
    model: 'zyantine-v1',
    messages: [
      { role: 'user', content: '你好' }
    ],
    temperature: 0.7,
    max_tokens: 1000
  })
});

const result = await response.json();
console.log(result.choices[0].message.content);

// 流式请求
const response = await fetch('http://localhost:8000/v1/chat/completions/stream', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
  },
  body: JSON.stringify({
    model: 'zyantine-v1',
    messages: [
      { role: 'user', content: '你好' }
    ],
    temperature: 0.7,
    max_tokens: 1000
  })
});

const reader = response.body.getReader();
const decoder = new TextDecoder();

while (true) {
  const { done, value } = await reader.read();
  if (done) break;
  console.log(decoder.decode(value));
}
```

## 与语音 RTC 交互项目集成

### 集成流程

1. **语音识别**: RTC 项目识别用户语音，转换为文本
2. **API 调用**: 将识别的文本发送到 `/v1/chat/completions` 端点
3. **获取响应**: 接收 AI 生成的文本响应
4. **语音合成**: 将响应文本传递给 RTC 项目的语音合成模块

### 示例代码

```python
import requests

def process_voice_input(text: str) -> str:
    """
    处理语音输入并返回 AI 响应
    
    Args:
        text: 语音识别后的文本
        
    Returns:
        AI 生成的响应文本
    """
    response = requests.post(
        "http://localhost:8000/v1/chat/completions",
        json={
            "model": "zyantine-v1",
            "messages": [
                {"role": "user", "content": text}
            ],
            "temperature": 0.7,
            "max_tokens": 1000
        },
        timeout=30
    )
    
    result = response.json()
    return result["choices"][0]["message"]["content"]

# 使用示例
user_text = "今天天气怎么样？"  # 来自语音识别
ai_response = process_voice_input(user_text)
print(ai_response)  # 输出: "我无法获取实时天气信息，建议您查看天气预报..."
```

### 流式响应集成（推荐）

对于实时语音交互，建议使用流式响应：

```python
import requests

def process_voice_input_stream(text: str, callback):
    """
    处理语音输入并流式返回 AI 响应
    
    Args:
        text: 语音识别后的文本
        callback: 接收响应片段的回调函数
    """
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
    print(chunk, end='', flush=True)  # 实时输出
    # 在这里可以调用语音合成模块

user_text = "今天天气怎么样？"
process_voice_input_stream(user_text, on_response_chunk)
```

## 模型选择

### zyantine-v1（标准版）

- 适用于大多数对话场景
- 响应速度快
- 资源消耗较低

### zyantine-enhanced（增强版）

- 启用增强版认知流程
- 更深入的情感分析和策略制定
- 适合复杂对话场景
- 响应时间稍长

## 错误处理

API 返回标准的 HTTP 状态码和错误信息：

```json
{
  "error": {
    "message": "错误描述",
    "type": "invalid_request_error",
    "param": null,
    "code": null
  }
}
```

常见错误码：
- `400`: 请求参数错误
- `404`: 资源不存在
- `500`: 服务器内部错误
- `503`: 服务不可用

## 性能优化建议

1. **使用流式响应**: 对于长文本响应，使用流式接口可以提升用户体验
2. **合理设置 max_tokens**: 根据实际需求设置，避免不必要的资源消耗
3. **会话管理**: 为每个用户或会话使用独立的 session_id
4. **连接池**: 在高并发场景下，使用 HTTP 连接池复用连接

## 配置参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| --host | 0.0.0.0 | 监听地址 |
| --port | 8000 | 监听端口 |
| --api-key | None | OpenAI API 密钥 |
| --session | default | 会话 ID |

## 故障排查

### 服务无法启动

1. 检查端口是否被占用：`lsof -i :8000`
2. 检查依赖是否完整：`pip install -r requirements.txt`
3. 查看日志输出

### API 调用失败

1. 检查服务是否正常运行：访问 http://localhost:8000/health
2. 检查请求格式是否正确
3. 查看服务日志

## 许可证

本项目遵循原有项目的许可证。
