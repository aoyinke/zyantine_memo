# 记忆模块架构文档

## 一、模块概述

Zyantine 记忆系统是一个完整的记忆管理解决方案，提供短期记忆、长期记忆、分层存储、多维度索引等功能。

## 二、目录结构

```
memory/
├── __init__.py           # 模块入口，定义公共接口
├── models.py             # 统一数据模型定义
├── memory_manager.py     # 记忆管理器（高层API）
├── memory_store.py       # 核心存储系统（基于memo0）
├── memory_exceptions.py  # 异常类定义
├── memory_utils.py       # 工具函数
├── memory_evaluator.py   # 记忆价值评估器
├── indexing.py           # 统一索引管理
├── storage/              # 存储子模块
│   ├── __init__.py
│   ├── base.py           # 存储抽象基类
│   └── tiered_storage.py # 分层存储实现
└── ARCHITECTURE.md       # 本文档
```

## 三、核心组件

### 3.1 数据模型 (`models.py`)

统一的数据模型定义，包括：

**枚举类型：**
- `MemoryType` - 记忆类型（对话、经验、用户档案等）
- `MemoryPriority` - 优先级（关键、高、中、低、归档）
- `MemoryLifecycleStage` - 生命周期阶段
- `StorageTier` - 存储层级（热/温/冷）
- `DeduplicationStrategy` - 去重策略
- `MemoryRetrievalStrategy` - 检索策略
- `MemorySecurityLevel` - 安全级别
- `MemoryPrivacyPolicy` - 隐私策略

**数据类：**
- `MemoryRecord` - 记忆记录
- `ShortTermMemory` - 短期记忆
- `PerformanceMetric` - 性能指标
- `SearchResult` - 搜索结果
- `MemoryStats` - 统计信息
- `StorageTierConfig` - 存储层配置
- `MemorySystemConfig` - 系统配置

### 3.2 存储系统 (`storage/`)

**`BaseStorage`** - 存储抽象基类
- 定义标准存储接口（get, set, delete, exists, clear等）
- 提供简单的内存存储实现
- 支持LFU淘汰策略

**`TieredStorage`** - 分层存储
- 热/温/冷三层数据存储
- 自动根据访问频率迁移数据
- 支持数据压缩

**`MemoryCache`** - 缓存管理
- 带版本控制的缓存
- TTL过期机制
- 支持按标签/类型/前缀无效化

### 3.3 索引系统 (`indexing.py`)

**`MemoryIndex`** - 统一索引管理器
- 主题索引
- 关键信息索引
- 重要性索引
- 类型索引
- 标签索引
- 时间索引
- 优先级索引
- 会话索引

**`ContextWindow`** - 上下文窗口管理
- 动态调整窗口大小
- 主题权重衰减
- 短期记忆标记

### 3.4 记忆管理器 (`memory_manager.py`)

高层API，整合所有组件：
- `MemoryManager` - 主要入口类
- `MemoryPerformanceMonitor` - 性能监控
- `MemoryStorageOptimizer` - 存储优化
- `MemoryDeduplicator` - 去重管理
- `MemoryRetrievalOptimizer` - 检索优化
- `MemoryPriorityManager` - 优先级管理
- `MemoryCompressionManager` - 压缩管理
- `MemoryLifecycleManager` - 生命周期管理
- `MemorySecurityManager` - 安全管理

### 3.5 记忆评估器 (`memory_evaluator.py`)

智能记忆价值评估系统：

**核心特性：**
- 多维度规则引擎评估（毫秒级响应）
- 减少对 LLM API 的依赖
- 保守策略：宁可多存储，不轻易过滤
- 支持增量学习和自适应调整

**评估维度：**
- `INFORMATION_DENSITY` - 信息密度
- `EMOTIONAL_VALUE` - 情感价值
- `TEMPORAL_RELEVANCE` - 时间相关性
- `USER_RELEVANCE` - 用户关联度
- `ACTIONABLE_VALUE` - 可执行价值
- `KNOWLEDGE_VALUE` - 知识价值
- `UNIQUENESS` - 独特性

**关键词检测：**
- 用户档案关键词（姓名、年龄、职业、偏好等）
- 事件/任务关键词（会议、提醒、计划等）
- 知识/观点关键词（定义、方法、建议等）
- 情感关键词（积极/消极/中性）

### 3.6 核心存储 (`memory_store.py`)

基于 memo0 的存储实现：
- `ZyantineMemorySystem` - 核心存储系统
- 集成智能评估器进行记忆筛选
- 支持 memo0 和文件持久化回退
- 短期记忆管理
- 记忆分级存储

## 四、使用示例

### 基础使用

```python
from memory import MemoryManager, MemoryType, MemoryPriority

# 创建管理器
manager = MemoryManager()

# 添加记忆
memory_id = manager.add_memory(
    content="用户喜欢Python编程",
    memory_type=MemoryType.USER_PROFILE,
    tags=["偏好", "编程"],
    priority=MemoryPriority.HIGH
)

# 搜索记忆
results = manager.search_memories("编程相关")
```

### 使用存储组件

```python
from memory import TieredStorage, MemoryRecord, MemoryType, MemoryPriority
from datetime import datetime

# 创建分层存储
storage = TieredStorage()

# 创建记忆记录
record = MemoryRecord(
    memory_id="mem_001",
    content="测试内容",
    memory_type=MemoryType.CONVERSATION,
    metadata={},
    tags=["test"],
    priority=MemoryPriority.MEDIUM,
    created_at=datetime.now()
)

# 存储
storage.set(record.memory_id, record)

# 获取
retrieved = storage.get(record.memory_id)
```

### 使用索引组件

```python
from memory import MemoryIndex, ContextWindow

# 创建索引
index = MemoryIndex()

# 索引记忆
index.index_memory(record)

# 查询
by_type = index.get_by_type("conversation")
by_topic = index.get_by_topic("work")

# 上下文窗口
ctx = ContextWindow(max_size=20)
ctx.add("memory_1", ["work", "meeting"])
recent = ctx.get_recent(limit=5)
```

## 五、设计原则

1. **单一职责** - 每个类只负责一个功能领域
2. **接口分离** - 通过抽象基类定义清晰的接口
3. **依赖倒置** - 高层模块不依赖低层模块的具体实现
4. **延迟导入** - 核心组件使用延迟导入，避免强制依赖
5. **向后兼容** - 保留原有API，确保现有代码可以继续工作

### 使用评估器

```python
from memory import MemoryEvaluator

# 创建评估器
evaluator = MemoryEvaluator(
    enable_llm_evaluation=False,  # 关闭LLM评估
    conservative_mode=True        # 保守模式
)

# 评估内容
result = evaluator.evaluate(
    content="我叫张三，今年25岁",
    memory_type="user_profile"
)

print(f"分数: {result.overall_score}")
print(f"是否存储: {result.should_store}")
print(f"优先级: {result.storage_priority}")
print(f"原因: {result.reason}")
```

## 六、优化历史

### v2.1.0 (2026-01-18)
- 新增智能记忆评估器 (`memory_evaluator.py`)
- 多维度规则引擎评估，减少API调用
- 保守策略优化，提高长期记忆存储率
- 丰富的特征提取（实体、意图、情感）

### v2.0.0 (2026-01-18)
- 重构模块结构，分离数据模型
- 创建统一的存储抽象层
- 整合分散的索引功能
- 实现延迟导入机制
- 添加完整的类型注解

### v1.x
- 原始实现，所有代码集中在两个大文件中
