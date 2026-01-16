"""
处理管道 - 实现7阶段处理流程的管道模式
"""
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
import asyncio
from concurrent.futures import ThreadPoolExecutor
import time
import copy


class ProcessingStage(Enum):
    """处理阶段枚举"""
    PREPROCESS = "preprocess"
    INSTINCT_CHECK = "instinct_check"
    MEMORY_RETRIEVAL = "memory_retrieval"
    DESIRE_UPDATE = "desire_update"
    COGNITIVE_FLOW = "cognitive_flow"
    DIALECTICAL_GROWTH = "dialectical_growth"
    REPLY_GENERATION = "reply_generation"
    PROTOCOL_REVIEW = "protocol_review"
    INTERACTION_RECORDING = "interaction_recording"
    WHITE_DOVE_CHECK = "white_dove_check"


class ProcessingResult(Enum):
    """处理结果"""
    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"
    OVERRIDDEN = "overridden"


@dataclass
class StageContext:
    """阶段上下文"""
    user_input: str
    conversation_history: List[Dict]
    system_components: Dict[str, Any]
    api_service_provider: Optional[Any] = None  # 新增：API服务提供者

    # 阶段输出
    context_info: Optional[Dict] = field(default_factory=dict)  # 修改：上下文信息
    instinct_override: Optional[Dict] = None
    retrieved_memories: List[Any] = field(default_factory=list)  # 新增：检索到的记忆
    resonant_memory: Optional[Dict] = None  # 新增：共鸣记忆
    desire_vectors: Dict[str, float] = field(default_factory=dict)  # 修改：欲望向量
    cognitive_snapshot: Optional[Dict] = None  # 新增：认知快照
    cognitive_result: Optional[Dict] = None  # 新增：认知结果
    strategy: Optional[str] = None  # 新增：策略
    emotional_context: Dict[str, Any] = field(default_factory=dict)  # 新增：情感上下文
    growth_result: Optional[Dict] = None  # 新增：成长结果
    mask_type: str = "长期搭档"  # 新增：面具类型
    final_reply: Optional[str] = None
    review_results: List[Dict] = field(default_factory=list)  # 新增：审查结果
    white_dove_check: Dict[str, Any] = field(default_factory=dict)  # 修改：白鸽检查结果

    # 状态跟踪
    current_stage: Optional[ProcessingStage] = None
    processing_log: List[Dict] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    start_time: float = field(default_factory=time.time)
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    interaction_recorded: bool = False  # 新增：交互是否已记录
    stage_results: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """初始化后验证所有字典字段"""
        dict_fields = ['context_info', 'desire_vectors', 'emotional_context', 'white_dove_check', 'stage_results']
        for field in dict_fields:
            if not isinstance(getattr(self, field), dict):
                # 如果字段不是字典，重置为默认字典
                setattr(self, field, {})
                # 记录警告
                import logging
                logger = logging.getLogger("StageContext")
                logger.warning(f"StageContext字段 {field} 不是字典类型，已重置为默认字典")
    
    def add_stage_result(self, stage: ProcessingStage, result: ProcessingResult):
        """添加阶段结果"""
        self.stage_results[stage] = result

    def add_error(self, error: str):
        """添加错误"""
        self.errors.append(error)

    def add_warning(self, warning: str):
        """添加警告"""
        self.warnings.append(warning)

    def add_performance_metric(self, stage: str, duration: float):
        """添加性能指标"""
        self.performance_metrics[stage] = duration

    def log_stage_completion(self, stage: ProcessingStage, metadata: Dict = None):
        """记录阶段完成"""
        self.processing_log.append({
            "stage": stage.value,
            "timestamp": time.time(),
            "metadata": metadata or {}
        })

    def should_continue(self) -> bool:
        """判断是否应该继续处理"""
        # 如果有本能接管，应该停止
        if self.instinct_override and self.instinct_override.get("skip_remaining_stages", False):
            return False

        # 如果有致命错误
        if any("致命错误" in error for error in self.errors):
            return False

        return True

    def to_dict(self) -> Dict:
        """转换为字典"""
        return {
            "user_input": self.user_input,
            "conversation_history_length": len(self.conversation_history),
            "stage_results": {stage.value: result.value for stage, result in self.stage_results.items()},
            "errors": self.errors,
            "warnings": self.warnings,
            "performance_metrics": self.performance_metrics,
            "processing_time": time.time() - self.start_time
        }


class StageHandler(ABC):
    """阶段处理器基类"""

    @abstractmethod
    def process(self, context: StageContext) -> StageContext:
        """处理阶段"""
        pass

    @property
    @abstractmethod
    def stage_name(self) -> ProcessingStage:
        """阶段名称"""
        pass

    @property
    def dependencies(self) -> List[ProcessingStage]:
        """依赖的阶段"""
        return []

    @property
    def is_optional(self) -> bool:
        """是否可选"""
        return False

    def pre_process(self, context: StageContext) -> StageContext:
        """预处理钩子"""
        return context

    def post_process(self, context: StageContext) -> StageContext:
        """后处理钩子"""
        return context


class BaseStageHandler(StageHandler):
    """基础阶段处理器（兼容stage_handlers中的处理器）"""

    def __init__(self, logger=None):
        self.logger = logger

    def process(self, context: StageContext) -> StageContext:
        """处理阶段 - 需要子类实现"""
        raise NotImplementedError("子类必须实现process方法")

    @property
    def stage_name(self) -> ProcessingStage:
        """阶段名称 - 需要子类实现"""
        raise NotImplementedError("子类必须实现stage_name属性")


class ProcessingPipeline:
    """可配置的处理管道"""

    def __init__(self, enable_parallelism: bool = False, max_workers: int = 4, enable_fast_path: bool = False):
        self.stages: Dict[ProcessingStage, StageHandler] = {}
        self.stage_order: List[ProcessingStage] = []
        self.pre_hooks: Dict[ProcessingStage, List[Callable]] = {}
        self.post_hooks: Dict[ProcessingStage, List[Callable]] = {}
        self.enable_parallelism = enable_parallelism
        self.enable_fast_path = enable_fast_path
        self.max_workers = max_workers
        self.executor = ThreadPoolExecutor(max_workers=max_workers) if enable_parallelism else None
        
        # 阶段依赖关系 - 只保留核心阶段的依赖
        self.stage_dependencies = {
            ProcessingStage.PREPROCESS: [],
            ProcessingStage.INSTINCT_CHECK: [ProcessingStage.PREPROCESS],
            ProcessingStage.MEMORY_RETRIEVAL: [ProcessingStage.PREPROCESS],
            ProcessingStage.DESIRE_UPDATE: [ProcessingStage.MEMORY_RETRIEVAL],
            ProcessingStage.COGNITIVE_FLOW: [ProcessingStage.DESIRE_UPDATE],
            ProcessingStage.DIALECTICAL_GROWTH: [ProcessingStage.COGNITIVE_FLOW],
            ProcessingStage.REPLY_GENERATION: [ProcessingStage.MEMORY_RETRIEVAL],  # 修改：只依赖记忆检索
            ProcessingStage.PROTOCOL_REVIEW: [ProcessingStage.REPLY_GENERATION],
            ProcessingStage.INTERACTION_RECORDING: [ProcessingStage.PROTOCOL_REVIEW],
            ProcessingStage.WHITE_DOVE_CHECK: [ProcessingStage.INTERACTION_RECORDING]
        }

    def register_stage(self, stage_handler: StageHandler):
        """注册阶段处理器"""
        stage_name = stage_handler.stage_name
        self.stages[stage_name] = stage_handler
        self.stage_order.append(stage_name)

    def add_pre_hook(self, stage: ProcessingStage, hook: Callable):
        """添加前置钩子"""
        if stage not in self.pre_hooks:
            self.pre_hooks[stage] = []
        self.pre_hooks[stage].append(hook)

    def add_post_hook(self, stage: ProcessingStage, hook: Callable):
        """添加后置钩子"""
        if stage not in self.post_hooks:
            self.post_hooks[stage] = []
        self.post_hooks[stage].append(hook)

    def execute(self, context: StageContext) -> StageContext:
        """执行管道处理"""
        print(f"[处理管道] 开始处理，共{len(self.stage_order)}个阶段")
        print(f"[处理管道] 配置: 并行={self.enable_parallelism}, 快速路径={self.enable_fast_path}")

        # 检查是否启用快速路径
        if self.enable_fast_path and self._should_use_fast_path(context):
            print("[处理管道] 使用快速路径处理")
            return self._execute_fast_path(context)

        # 检查是否启用并行处理
        if self.enable_parallelism:
            context = self._execute_parallel(context)
        else:
            # 按顺序执行各个阶段
            for stage in self.stage_order:
                if not self._should_continue(context, stage):
                    print(f"[处理管道] 提前终止，当前阶段: {stage.value}")
                    break

                # 执行前置钩子
                self._execute_hooks(stage, context, pre_hook=True)

                # 执行阶段处理器
                context = self._execute_stage(stage, context)

                # 执行后置钩子
                self._execute_hooks(stage, context, pre_hook=False)

        print(f"[处理管道] 处理完成，耗时: {time.time() - context.start_time:.2f}秒")
        return context

    def _should_use_fast_path(self, context: StageContext) -> bool:
        """判断是否应该使用快速路径"""
        # 简单请求判断逻辑
        user_input = context.user_input.strip()
        
        # 长度判断
        if len(user_input) > 100:
            return False
        
        # 关键词判断
        simple_keywords = ["你好", "Hello", "hi", "Hi", "再见", "Bye", "bye", "谢谢", "Thanks", "thanks"]
        for keyword in simple_keywords:
            if keyword in user_input:
                return True
        
        # 问题类型判断（过于简化的判断）
        if any(prefix in user_input for prefix in ["是什么", "什么是", "how", "How", "what", "What"]):
            return False
        
        return False

    def _execute_fast_path(self, context: StageContext) -> StageContext:
        """执行快速路径处理"""
        # 只执行4个核心阶段
        fast_path_stages = [
            ProcessingStage.PREPROCESS,
            ProcessingStage.MEMORY_RETRIEVAL,
            ProcessingStage.REPLY_GENERATION,
            ProcessingStage.PROTOCOL_REVIEW
        ]

        for stage in fast_path_stages:
            if stage not in self.stages:
                continue
                
            if not self._should_continue(context, stage):
                break

            # 执行前置钩子
            self._execute_hooks(stage, context, pre_hook=True)

            # 执行阶段处理器
            context = self._execute_stage(stage, context)

            # 执行后置钩子
            self._execute_hooks(stage, context, pre_hook=False)

        return context

    def _execute_parallel(self, context: StageContext) -> StageContext:
        """并行执行处理管道"""
        # 构建阶段执行图
        executed_stages = set()
        remaining_stages = set(self.stage_order)

        while remaining_stages:
            # 找出所有可以执行的阶段（依赖已满足）
            executable_stages = []
            for stage in remaining_stages:
                dependencies = self.stage_dependencies.get(stage, [])
                if all(dep in executed_stages for dep in dependencies):
                    executable_stages.append(stage)

            if not executable_stages:
                # 无法执行更多阶段，退出
                break

            # 并行执行这些阶段
            if len(executable_stages) > 1:
                print(f"[处理管道] 并行执行阶段: {[s.value for s in executable_stages]}")
                
                # 创建并行任务
                tasks = []
                for stage in executable_stages:
                    if self._should_continue(context, stage):
                        tasks.append((stage, context.copy() if hasattr(context, 'copy') else context))

                # 执行并行任务
                if tasks:
                    results = []
                    for stage, stage_context in tasks:
                        # 执行前置钩子
                        self._execute_hooks(stage, stage_context, pre_hook=True)
                        # 执行阶段
                        result_context = self._execute_stage(stage, stage_context)
                        # 执行后置钩子
                        self._execute_hooks(stage, result_context, pre_hook=False)
                        results.append((stage, result_context))

                    # 合并结果
                    for stage, result_context in results:
                        # 合并结果到主上下文
                        self._merge_contexts(context, result_context)
                        executed_stages.add(stage)
                        remaining_stages.remove(stage)
            else:
                # 只有一个阶段，串行执行
                stage = executable_stages[0]
                if self._should_continue(context, stage):
                    # 执行前置钩子
                    self._execute_hooks(stage, context, pre_hook=True)
                    # 执行阶段
                    context = self._execute_stage(stage, context)
                    # 执行后置钩子
                    self._execute_hooks(stage, context, pre_hook=False)
                    executed_stages.add(stage)
                    remaining_stages.remove(stage)

        return context

    def _merge_contexts(self, target: StageContext, source: StageContext) -> None:
        """合并上下文"""
        # 合并错误和警告
        target.errors.extend(source.errors)
        target.warnings.extend(source.warnings)
        
        # 合并性能指标
        for stage, duration in source.performance_metrics.items():
            if stage not in target.performance_metrics:
                target.performance_metrics[stage] = duration
        
        # 合并处理日志
        target.processing_log.extend(source.processing_log)
        
        # 合并阶段结果
        for stage, result in source.stage_results.items():
            if stage not in target.stage_results:
                target.stage_results[stage] = result
        
        # 合并其他字段
        if source.final_reply:
            target.final_reply = source.final_reply
        if source.instinct_override:
            target.instinct_override = source.instinct_override
        if source.retrieved_memories:
            target.retrieved_memories.extend(source.retrieved_memories)
        if source.resonant_memory:
            target.resonant_memory = source.resonant_memory
        if source.cognitive_snapshot:
            target.cognitive_snapshot = source.cognitive_snapshot
        if source.cognitive_result:
            target.cognitive_result = source.cognitive_result
        if source.growth_result:
            target.growth_result = source.growth_result

    def _should_continue(self, context: StageContext, current_stage: ProcessingStage) -> bool:
        """检查是否应该继续执行"""
        # 如果被本能响应接管，跳过后续阶段
        if context.instinct_override and context.instinct_override.get("skip_remaining_stages", False):
            return False

        # 更新当前阶段
        context.current_stage = current_stage

        return True

    def _execute_stage(self, stage: ProcessingStage, context: StageContext) -> StageContext:
        """执行单个阶段"""
        handler = self.stages.get(stage)
        if not handler:
            print(f"[处理管道] 警告：未找到阶段处理器: {stage.value}")
            return context

        stage_start = time.time()

        try:
            # 执行阶段处理
            print(f"[处理管道] 执行阶段: {stage.value}")

            # 执行处理器的预处理钩子
            pre_result = handler.pre_process(context)
            # 确保返回的是 StageContext 对象
            if pre_result is not None:
                context = pre_result

            # 执行主要处理逻辑
            process_result = handler.process(context)
            # 确保返回的是 StageContext 对象
            if process_result is not None:
                context = process_result

            # 执行处理器的后处理钩子
            post_result = handler.post_process(context)
            # 确保返回的是 StageContext 对象
            if post_result is not None:
                context = post_result

            duration = time.time() - stage_start
            context.add_performance_metric(stage.value, duration)

            # 记录阶段完成
            context.log_stage_completion(stage, {"duration": duration})

            print(f"[处理管道] 阶段完成: {stage.value}，耗时: {duration:.2f}秒")

        except Exception as e:
            duration = time.time() - stage_start
            context.add_performance_metric(stage.value, duration)
            context.add_error(f"阶段 {stage.value} 失败: {str(e)}")

            if not handler.is_optional:
                print(f"[处理管道] 错误：关键阶段 {stage.value} 失败: {e}")
            else:
                print(f"[处理管道] 警告：可选阶段 {stage.value} 失败，继续执行: {e}")

        return context

    def _execute_hooks(self, stage: ProcessingStage, context: StageContext, pre_hook: bool = True):
        """执行钩子函数"""
        hooks = self.pre_hooks.get(stage) if pre_hook else self.post_hooks.get(stage)
        if not hooks:
            return

        hook_type = "前置" if pre_hook else "后置"
        for hook in hooks:
            try:
                hook(context)
            except Exception as e:
                context.add_error(f"{hook_type}钩子执行失败 ({stage.value}): {str(e)}")

    def shutdown(self):
        """关闭管道"""
        if self.executor:
            self.executor.shutdown(wait=True)