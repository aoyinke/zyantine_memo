# 测试目录结构说明

本目录包含 Zyantine Genesis 项目的所有测试代码，按照功能模块进行组织。

## 目录结构

```
tests/
├── __init__.py              # 测试包初始化
├── test_protocols.py        # 协议模块测试
├── memory/                  # 记忆模块测试
│   ├── __init__.py
│   ├── test_security.py              # 安全管理测试
│   ├── test_deduplication.py         # 去重机制测试
│   ├── test_compression.py           # 压缩归档测试
│   ├── test_priority_manager.py      # 优先级管理测试
│   ├── test_retrieval_optimizer.py   # 检索优化测试
│   ├── test_cache_invalidation.py    # 缓存失效测试
│   ├── test_memory_optimization.py  # 模块优化测试
│   ├── test_storage_optimizer.py     # 存储优化测试
│   ├── test_lifecycle_manager.py    # 生命周期管理测试
│   ├── test_performance_monitor.py   # 性能监控测试
│   └── test_performance_monitoring.py # 性能监控统计测试
├── api/                     # API 和通信测试
│   ├── __init__.py
│   ├── test_api.py                  # API 接口测试
│   └── test_websocket.py            # WebSocket 功能测试
├── system/                  # 系统测试
│   ├── __init__.py
│   ├── test_health_checker.py       # 健康检查测试
│   └── test_dashboard_integration.py # 仪表板集成测试
├── integration/             # 集成测试
│   ├── __init__.py
│   └── test_multi_turn.py          # 多轮对话集成测试
└── utils/                   # 测试工具
    ├── __init__.py
    ├── collect_test_info.py        # 测试信息收集工具
    └── run_tests.py                # 测试运行器
```

## 使用方法

### 运行所有测试

```bash
python tests/utils/run_tests.py --all
```

### 运行特定类别的测试

```bash
# 记忆模块测试
python tests/utils/run_tests.py --category memory

# API 测试
python tests/utils/run_tests.py --category api

# 系统测试
python tests/utils/run_tests.py --category system

# 集成测试
python tests/utils/run_tests.py --category integration

# 协议测试
python tests/utils/run_tests.py --category protocols
```

### 运行单个测试文件

```bash
python tests/utils/run_tests.py --test memory/test_security.py
```

### 列出所有测试文件

```bash
python tests/utils/run_tests.py --list
```

### 运行快速测试（核心功能）

```bash
python tests/utils/run_tests.py --quick
```

## 测试分类说明

### memory/ - 记忆模块测试
测试记忆管理系统的核心功能，包括：
- 安全管理和访问控制
- 记忆去重机制
- 压缩和归档功能
- 优先级管理
- 检索优化
- 缓存失效和版本控制
- 存储优化
- 生命周期管理
- 性能监控和统计

### api/ - API 和通信测试
测试 API 接口和通信功能，包括：
- RESTful API 接口
- WebSocket 实时通信

### system/ - 系统测试
测试系统级别的功能，包括：
- 系统健康检查
- 性能监控仪表板集成

### integration/ - 集成测试
测试多个模块的集成功能，包括：
- 多轮对话场景

### test_protocols.py - 协议模块测试
测试协议引擎和相关功能，包括：
- 表达验证器
- 长度规整器
- 事实检查器

## 注意事项

1. 部分测试需要外部服务支持（如 OpenAI API、Milvus 向量数据库等）
2. 如果外部服务不可用，测试会自动跳过并显示警告信息
3. 测试运行器会自动发现所有测试文件并按类别组织
4. 建议在运行测试前确保所有依赖服务已启动

## 添加新测试

1. 在对应的子目录下创建新的测试文件，文件名以 `test_` 开头
2. 测试运行器会自动发现新文件
3. 如果需要新的测试类别，请在 `run_tests.py` 中添加相应的目录扫描逻辑
