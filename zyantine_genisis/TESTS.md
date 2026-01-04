# 测试文档

本文档整理了项目中的所有测试代码，包括测试分类、运行方法和测试说明。

## 测试文件概览

### 1. 系统级测试 (System Tests)

#### test_health_checker.py
**描述**: 系统健康检查器集成测试

**测试内容**:
- 健康检查器基本功能
- 与记忆管理器集成
- 与认知流程集成
- 与协议引擎集成
- 告警功能
- 自动检查功能
- 综合集成测试

**运行方式**:
```bash
python test_health_checker.py
```

**测试结果**: 7/7 通过

---

#### test_dashboard_integration.py
**描述**: 系统性能监控仪表板集成测试

**测试内容**:
- 仪表板初始化
- 记忆模块指标收集
- 认知模块指标收集
- 协议模块指标收集
- API服务指标收集
- 综合报告生成

**运行方式**:
```bash
python test_dashboard_integration.py
```

---

### 2. 记忆模块测试 (Memory Tests)

#### test_lifecycle_manager.py
**描述**: 记忆生命周期管理器测试

**测试内容**:
- 记忆生命周期阶段转换
- 生命周期事件触发
- 生命周期统计信息

**运行方式**:
```bash
python test_lifecycle_manager.py
```

---

#### test_storage_optimizer.py
**描述**: 分层存储优化器测试

**测试内容**:
- 存储层级确定
- 存储迁移
- 存储统计信息

**运行方式**:
```bash
python test_storage_optimizer.py
```

---

#### test_deduplication.py
**描述**: 记忆去重测试

**测试内容**:
- 精确重复检测
- 相似重复检测
- 去重统计信息

**运行方式**:
```bash
python test_deduplication.py
```

---

#### test_security.py
**描述**: 记忆安全管理器测试

**测试内容**:
- 安全级别设置
- 隐私策略设置
- 内容加密/解密
- 访问控制
- 敏感数据检测

**运行方式**:
```bash
python test_security.py
```

**测试结果**: 8/8 通过

---

#### test_performance_monitor.py
**描述**: 记忆性能监控测试

**测试内容**:
- 性能指标收集
- 性能报告生成

**运行方式**:
```bash
python test_performance_monitor.py
```

---

#### test_cache_invalidation.py
**描述**: 缓存失效测试

**测试内容**:
- 缓存失效策略
- 缓存清理

**运行方式**:
```bash
python test_cache_invalidation.py
```

---

#### test_priority_manager.py
**描述**: 记忆优先级管理器测试

**测试内容**:
- 优先级设置
- 优先级调整
- 优先级统计

**运行方式**:
```bash
python test_priority_manager.py
```

---

#### test_retrieval_optimizer.py
**描述**: 记忆检索优化器测试

**测试内容**:
- 检索策略优化
- 检索性能提升

**运行方式**:
```bash
python test_retrieval_optimizer.py
```

---

#### test_compression.py
**描述**: 记忆压缩测试

**测试内容**:
- 内容压缩
- 内容解压
- 压缩率统计

**运行方式**:
```bash
python test_compression.py
```

---

#### test_memory_optimization.py
**描述**: 记忆优化综合测试

**测试内容**:
- 生命周期管理
- 分层存储
- 去重
- 安全管理
- 性能监控

**运行方式**:
```bash
python test_memory_optimization.py
```

---

### 3. 认知模块测试 (Cognition Tests)

#### tests/test_core.py
**描述**: 核心认知功能测试

**测试内容**:
- 核心身份识别
- 认知状态管理

**运行方式**:
```bash
python tests/test_core.py
```

---

### 4. 协议模块测试 (Protocol Tests)

#### tests/test_protocols.py
**描述**: 协议引擎测试

**测试内容**:
- 协议执行
- 协议优先级
- 协议冲突解决

**运行方式**:
```bash
python tests/test_protocols.py
```

---

### 5. API模块测试 (API Tests)

#### test_api.py
**描述**: API服务测试

**测试内容**:
- API请求处理
- API响应验证

**运行方式**:
```bash
python test_api.py
```

---

#### test_websocket.py
**描述**: WebSocket服务测试

**测试内容**:
- WebSocket连接
- 实时通信

**运行方式**:
```bash
python test_websocket.py
```

---

### 6. 集成测试 (Integration Tests)

#### test_multi_turn.py
**描述**: 多轮对话集成测试

**测试内容**:
- 多轮对话处理
- 上下文管理

**运行方式**:
```bash
python test_multi_turn.py
```

---

#### tests/test_memo0.py
**描述**: Memo0集成测试

**测试内容**:
- Memo0记忆系统
- 记忆存储和检索

**运行方式**:
```bash
python tests/test_memo0.py
```

---

#### tests/test_voice.py
**描述**: 语音功能测试

**测试内容**:
- 语音识别
- 语音合成

**运行方式**:
```bash
python tests/test_voice.py
```

---

## 测试运行器

项目提供了统一的测试运行器 `run_tests.py`，可以方便地管理和运行所有测试。

### 使用方法

#### 1. 列出所有测试文件
```bash
python run_tests.py --list
```

#### 2. 运行所有测试
```bash
python run_tests.py --all
```

#### 3. 运行指定类别的测试
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

# 其他测试
python run_tests.py --category other
```

#### 4. 运行快速测试（核心功能）
```bash
python run_tests.py --quick
```

#### 5. 运行指定的测试文件
```bash
python run_tests.py --test test_health_checker.py
```

---

## 测试分类说明

### 按模块分类

| 类别 | 说明 | 测试数量 |
|------|------|----------|
| memory | 记忆模块测试 | 10 |
| cognition | 认知模块测试 | 1 |
| protocols | 协议模块测试 | 1 |
| api | API模块测试 | 2 |
| system | 系统级测试 | 2 |
| other | 其他测试 | 2 |

### 按类型分类

| 类型 | 说明 |
|------|------|
| 单元测试 | 测试单个功能模块 |
| 集成测试 | 测试多个模块的集成 |
| 性能测试 | 测试系统性能 |
| 安全测试 | 测试系统安全性 |

---

## 测试最佳实践

### 1. 编写测试

- 每个测试文件应该有清晰的文档字符串
- 测试函数应该有描述性的名称
- 使用断言来验证结果
- 处理异常情况

### 2. 运行测试

- 在修改代码后运行相关测试
- 定期运行所有测试
- 查看测试输出，确保没有警告或错误

### 3. 维护测试

- 保持测试代码的更新
- 删除不再需要的测试
- 为新功能添加测试

---

## 测试覆盖率

当前项目的测试覆盖情况：

- **记忆模块**: 覆盖率高，包括生命周期、存储、去重、安全等
- **认知模块**: 基础覆盖
- **协议模块**: 基础覆盖
- **API模块**: 基础覆盖
- **系统级**: 完整覆盖，包括健康检查和性能监控

---

## 常见问题

### 1. 测试失败怎么办？

- 查看测试输出，了解失败原因
- 检查相关代码是否有问题
- 修复问题后重新运行测试

### 2. 如何添加新测试？

- 创建新的测试文件，命名为 `test_*.py`
- 编写测试函数
- 运行测试验证

### 3. 测试超时怎么办？

- 检查网络连接
- 检查外部依赖服务是否可用
- 增加超时时间

---

## 总结

项目共有 **18 个测试文件**，覆盖了记忆、认知、协议、API和系统级功能。使用统一的测试运行器可以方便地管理和运行所有测试。

建议定期运行测试以确保代码质量和系统稳定性。
