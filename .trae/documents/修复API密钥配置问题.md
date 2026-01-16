## 问题分析

根据错误日志，系统初始化失败是因为在mem0的Embedder初始化过程中，OpenAI客户端缺少API密钥配置。具体问题如下：

1. 错误信息：`The api_key client option must be set either by passing api_key to the client or by setting the OPENAI_API_KEY environment variable`
2. 错误发生在：`mem0/embeddings/openai.py` 中初始化OpenAI客户端时
3. 根本原因：在`_initialize_memo0`方法中，Embedder配置使用了错误的base_url键名

## 修复方案

### 1. 修正Embedder配置键名
**文件**：`memory_store.py`
**问题**：第342行和359行使用了`openai_base_url`作为键名，而mem0的Embedder实际需要的是`base_url`
**修复**：将这两处的`openai_base_url`改为`base_url`

### 2. 增强错误处理
**文件**：`memory_store.py`
**问题**：当API密钥缺失时，系统直接崩溃
**修复**：在`_initialize_memo0`方法中添加API密钥验证和默认值处理

### 3. 优化LLMClient初始化
**文件**：`memory_store.py`
**问题**：LLMClient尝试导入ZhipuAiClient失败，但没有影响系统运行
**修复**：添加更好的日志记录和错误处理，确保系统可以继续运行

## 修复步骤

1. 修改`_initialize_memo0`方法中的Embedder配置键名
2. 添加API密钥验证和默认值处理
3. 优化LLMClient初始化的错误处理
4. 测试修复后的系统是否能正常启动

## 预期结果

修复后，系统应该能够：
1. 正确初始化Embedder，即使没有提供API密钥
2. 使用默认值或备用方案继续运行
3. 提供清晰的日志信息，便于调试

## 影响范围

- 记忆系统初始化
- Embedder配置
- LLMClient初始化

这个修复解决了系统启动失败的核心问题，同时增强了系统的容错能力。