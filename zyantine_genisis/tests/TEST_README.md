# 测试模块使用指南

## 1. 测试目录结构

```
tests/
├── api/             # API接口测试
├── config/          # 配置文件测试
├── integration/     # 集成测试
├── memory/          # 记忆模块测试
├── protocols/       # 协议测试
├── system/          # 系统级测试
├── utils/           # 测试工具
├── __init__.py
└── TEST_README.md   # 本文件
```

## 2. 测试运行器

### 2.1 简介

测试运行器 (`tests/utils/run_tests.py`) 是一个统一的测试管理工具，用于发现、运行和报告测试结果。

### 2.2 基本用法

```bash
# 列出所有测试文件
python tests/utils/run_tests.py --list

# 运行所有测试
python tests/utils/run_tests.py --all

# 运行指定类别的测试
python tests/utils/run_tests.py --category <类别>

# 运行快速测试（核心功能）
python tests/utils/run_tests.py --quick

# 运行指定的测试文件
python tests/utils/run_tests.py --test <测试文件路径>
```

### 2.3 可用参数

- `--list`: 列出所有测试文件
- `--all`: 运行所有测试
- `--category <类别>`: 运行指定类别的测试
- `--quick`: 运行快速测试（核心功能）
- `--test <测试文件>`: 运行指定的测试文件

## 3. 测试类别

### 3.1 API 测试 (`api/`)

测试 API 接口的功能和性能：

- `test_api.py`: 测试基本 API 功能
- `test_empty_response.py`: 测试空响应处理
- `test_empty_response_fix.py`: 测试空响应修复
- `test_llm_provider.py`: 测试 LLM 提供商功能
- `test_message_optimization.py`: 测试消息优化
- `test_websocket.py`: 测试 WebSocket 功能

### 3.2 配置测试 (`config/`)

测试配置文件的加载和解析：

- `test_config_file.py`: 测试配置文件读取
- `test_env_api_key.py`: 测试环境变量 API 密钥
- `test_memo0_config.py`: 测试内存配置

### 3.3 集成测试 (`integration/`)

测试模块之间的集成功能：

- `test_multi_turn.py`: 测试多轮对话
- `test_real_conversation.py`: 测试真实对话
- `test_streaming.py`: 测试流式输出

### 3.4 记忆模块测试 (`memory/`)

测试记忆管理功能：

- `test_cache_invalidation.py`: 测试缓存失效
- `test_compression.py`: 测试压缩功能
- `test_deduplication.py`: 测试去重功能
- `test_lifecycle_manager.py`: 测试生命周期管理
- `test_memory_optimization.py`: 测试内存优化
- `test_performance_monitoring.py`: 测试性能监控
- `test_priority_manager.py`: 测试优先级管理
- `test_retrieval_optimizer.py`: 测试检索优化
- `test_security.py`: 测试安全功能
- `test_storage_optimizer.py`: 测试存储优化

### 3.5 协议测试 (`protocols/`)

测试系统协议功能：

- `test_protocols.py`: 测试协议功能

### 3.6 系统测试 (`system/`)

测试系统级功能：

- `test_dashboard_integration.py`: 测试仪表盘集成
- `test_health_checker.py`: 测试健康检查

## 4. 测试工具

### 4.1 测试运行器 (`run_tests.py`)

统一的测试管理工具，用于发现、运行和报告测试结果。

### 4.2 测试信息收集器 (`collect_test_info.py`)

收集和展示所有测试文件的详细信息。

## 5. 运行示例

### 5.1 列出所有测试

```bash
python tests/utils/run_tests.py --list
```

### 5.2 运行所有测试

```bash
python tests/utils/run_tests.py --all
```

### 5.3 运行API类别的测试

```bash
python tests/utils/run_tests.py --category api
```

### 5.4 运行单个测试文件

```bash
python tests/utils/run_tests.py --test tests/api/test_api.py
```

### 5.5 运行快速测试

```bash
python tests/utils/run_tests.py --quick
```

## 6. 添加新测试

### 6.1 测试文件命名

- 测试文件应以 `test_` 开头，以 `.py` 结尾
- 测试函数应以 `test_` 开头

### 6.2 测试文件结构

```python
import unittest
from your_module import your_function

class YourTest(unittest.TestCase):
    def test_functionality(self):
        # 测试代码
        result = your_function()
        self.assertEqual(result, expected_value)

if __name__ == "__main__":
    unittest.main()
```

### 6.3 运行新测试

添加新测试后，可以使用测试运行器运行：

```bash
python tests/utils/run_tests.py --test <新测试文件路径>
```

## 7. 测试报告

测试运行完成后，会生成详细的测试报告，包括：

- 测试总数
- 通过测试数
- 失败测试数
- 通过率
- 测试耗时统计

## 8. 最佳实践

1. **保持测试独立**: 每个测试应该独立运行，不依赖其他测试的结果
2. **测试命名清晰**: 测试函数和文件应该有清晰的命名，描述测试的功能
3. **覆盖边界情况**: 测试应该覆盖正常情况和边界情况
4. **保持测试简洁**: 测试应该专注于单一功能，避免测试过多内容
5. **定期运行测试**: 确保所有测试都能通过，特别是在代码变更后

## 9. 故障排除

### 9.1 测试失败

如果测试失败，测试运行器会显示详细的错误信息。查看错误信息，定位问题并修复。

### 9.2 测试运行超时

如果测试运行超时，可以检查：

- 网络连接是否正常
- API 服务是否可用
- 测试代码是否有死循环

### 9.3 配置问题

如果测试因配置问题失败，可以检查：

- 配置文件是否存在
- 配置文件格式是否正确
- API 密钥是否有效

## 10. 总结

测试模块是确保系统功能正确性和稳定性的重要工具。通过使用测试运行器，可以方便地管理和运行测试，确保系统的质量。

建议定期运行测试，特别是在代码变更后，以确保系统功能的正确性。
