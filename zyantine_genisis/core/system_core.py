"""
系统核心 - 整合所有模块的主系统（更新API集成部分）
"""
import os
import uuid
from typing import Dict, List, Any, Optional
from datetime import datetime
import random

from config.config_manager import ConfigManager
from memory.memory_manager import MemoryManager, MemoryType, MemoryPriority
from core.processing_pipeline import ProcessingPipeline, StageContext, ProcessingStage
from core.component_manager import ComponentManager
from protocols.protocol_engine import ProtocolEngine
from cognition.cognitive_flow_manager import CognitiveFlowManager
from utils.logger import SystemLogger
from utils.error_handler import ErrorHandler
from core.stage_handlers import (
            PreprocessHandler, InstinctCheckHandler, MemoryRetrievalHandler,
            DesireUpdateHandler, CognitiveFlowHandler, DialecticalGrowthHandler,
            ReplyGenerationHandler, ProtocolReviewHandler, InteractionRecordingHandler,
            WhiteDoveCheckHandler
        )

class ZyantineCore:
    """自衍体系统核心（增强版）"""

    def __init__(self,
                 config_file: Optional[str] = None,
                 user_profile_data: Optional[Dict] = None,
                 self_profile_data: Optional[Dict] = None,
                 session_id: Optional[str] = None):

        # 加载配置
        self.config_manager = ConfigManager()
        self.config = self.config_manager.load(config_file)

        # 更新会话ID
        if session_id:
            self.config.session_id = session_id

        # 初始化组件
        self.logger = SystemLogger().get_logger("ZyantineCore")
        self.error_handler = ErrorHandler()
        self.component_manager = ComponentManager(self.config)

        # 加载核心组件
        self._load_core_components()
        self.api_service_provider = self.components.get('api_service_provider')
        if not self.api_service_provider:
            raise ValueError("APIServiceProvider 初始化失败，缺少api_service_provider")

        # 初始化处理管道
        self.pipeline = self._initialize_pipeline()

        # 加载配置文件
        self.user_profile_data = user_profile_data or {}
        self.self_profile_data = self_profile_data or {}

        # 导入记忆
        self._import_initial_memories()

        # 系统状态
        self.system_status = self._create_system_status()

        self.logger.info(f"系统核心初始化完成，会话ID: {self.config.session_id}")

    def _load_core_components(self):
        """加载核心组件"""
        # 通过组件管理器获取所有组件
        self.components = self.component_manager.initialize_components()

        # 使用组件管理器中的内存管理器
        self.memory_manager = self.components['memory_manager']
        # 创建协议引擎
        self.protocol_engine = ProtocolEngine(
            fact_checker=self.components.get("fact_checker"),
            length_regulator=self.components.get("length_regulator"),
            expression_validator=self.components.get("expression_validator")
        )

        # 创建认知流程管理器
        self.cognitive_flow_manager = CognitiveFlowManager(
            core_identity=self.components.get("core_identity"),
            memory_manager=self.memory_manager,
            meta_cognition=self.components.get("meta_cognition"),
            fact_checker=self.components.get("fact_checker")
        )

    def _initialize_pipeline(self) -> ProcessingPipeline:
        """初始化处理管道"""

        pipeline = ProcessingPipeline(
            enable_parallelism=self.config.processing.enable_stage_parallelism,
            max_workers=4
        )

        # 注册阶段处理器
        pipeline.register_stage(PreprocessHandler(
            context_parser=self.components.get("context_parser"),
            logger=self.logger
        ))

        pipeline.register_stage(InstinctCheckHandler(
            instinct_core=self.components.get("instinct_core"),
            logger=self.logger
        ))

        pipeline.register_stage(MemoryRetrievalHandler(
            memory_manager=self.memory_manager,
            logger=self.logger
        ))

        pipeline.register_stage(DesireUpdateHandler(
            desire_engine=self.components.get("desire_engine"),
            dashboard=self.components.get("dashboard"),
            logger=self.logger
        ))

        pipeline.register_stage(CognitiveFlowHandler(
            cognitive_flow_manager=self.cognitive_flow_manager,
            logger=self.logger
        ))

        pipeline.register_stage(DialecticalGrowthHandler(
            dialectical_growth=self.components.get("dialectical_growth"),
            logger=self.logger
        ))

        pipeline.register_stage(ReplyGenerationHandler(
            reply_generator=self.components.get("reply_generator"),
            mask_templates=self._load_mask_templates(),
            logger=self.logger
        ))

        pipeline.register_stage(ProtocolReviewHandler(
            protocol_engine=self.protocol_engine,
            meta_cognition=self.components.get("meta_cognition"),
            logger=self.logger
        ))

        pipeline.register_stage(InteractionRecordingHandler(
            memory_manager=self.memory_manager,
            logger=self.logger
        ))

        pipeline.register_stage(WhiteDoveCheckHandler(
            desire_engine=self.components.get("desire_engine"),
            instinct_core=self.components.get("instinct_core"),
            logger=self.logger
        ))

        # 添加监控钩子
        pipeline.add_pre_hook(ProcessingStage.PREPROCESS,
                              lambda ctx: self.logger.info(f"开始处理: {ctx.user_input[:50]}..."))

        pipeline.add_post_hook(ProcessingStage.INTERACTION_RECORDING,
                               lambda ctx: self.logger.info(f"处理完成，响应长度: {len(ctx.final_reply or '')}"))

        return pipeline

    def _load_mask_templates(self) -> Dict[str, List[str]]:
        """加载面具模板"""
        return {
            "长期搭档": [
                "关于这个问题，我的分析是：{strategy}。你怎么看？",
                "从我的角度考虑，建议：{strategy}。",
                "根据我们之前的讨论，我认为：{strategy}。"
            ],
            "知己": [
                "我理解你的感受。{strategy}",
                "其实我也有过类似的经历。{strategy}",
                "跟你说说我的想法：{strategy}"
            ],
            "青梅竹马": [
                "哈哈，这让我想起以前...{strategy}",
                "你总是能提出有趣的问题！{strategy}",
                "记得你之前也说过类似的话...{strategy}"
            ],
            "伴侣": [
                "我深深感受到...{strategy}",
                "这对我很重要，因为...{strategy}",
                "我想和你分享的是...{strategy}"
            ]
        }

    def _import_initial_memories(self):
        """导入初始记忆"""
        if not self.user_profile_data and not self.self_profile_data:
            return

        self.logger.info("开始导入初始记忆...")
        imported_count = 0

        # 导入用户记忆
        if "memories" in self.user_profile_data:
            for memory in self.user_profile_data["memories"]:
                try:
                    self.memory_manager.add_memory(
                        content=memory.get("content", ""),
                        memory_type=MemoryType.EXPERIENCE,
                        tags=memory.get("tags", ["用户记忆", "导入"]),
                        emotional_intensity=memory.get("emotional_intensity", 0.5),
                        strategic_value=memory.get("strategic_value", {}),
                        priority=MemoryPriority.HIGH
                    )
                    imported_count += 1
                except Exception as e:
                    self.logger.error(f"导入用户记忆失败: {e}")

        # 导入自衍体记忆
        if "self_memories" in self.self_profile_data:
            for memory in self.self_profile_data["self_memories"]:
                try:
                    self.memory_manager.add_memory(
                        content=memory.get("content", ""),
                        memory_type=MemoryType.EXPERIENCE,
                        tags=memory.get("tags", ["自衍体记忆", "导入"]),
                        emotional_intensity=memory.get("emotional_intensity", 0.5),
                        strategic_value=memory.get("strategic_value", {}),
                        priority=MemoryPriority.HIGH
                    )
                    imported_count += 1
                except Exception as e:
                    self.logger.error(f"导入自衍体记忆失败: {e}")

        self.logger.info(f"成功导入 {imported_count} 条初始记忆")

    def _create_system_status(self) -> Dict[str, Any]:
        """创建系统状态报告"""
        memory_stats = self.memory_manager.get_statistics()

        return {
            "session_id": self.config.session_id,
            "user_id": self.config.user_id,
            "system_name": self.config.system_name,
            "version": self.config.version,
            "initialization_time": datetime.now().isoformat(),
            "memory_system": "memo0_framework",
            "chat_model": self.config.api.chat_model,
            "components_loaded": len(self.components),
            "memory_stats": memory_stats,
            "config_source": self.config_manager._config_file or "default",
            "processing_mode": self.config.processing.mode.value,
            "protocols_enabled": {
                "fact_check": self.config.protocols.enable_fact_check,
                "length_regulation": self.config.protocols.enable_length_regulation,
                "expression_protocol": self.config.protocols.enable_expression_protocol
            }
        }

    def _generate_error_response(self, error: Any, user_input: str) -> str:
        """生成错误响应"""
        error_responses = [
            "我的思考过程出现了一些混乱，能请你再问一次吗？",
            "刚才的思考链路好像打了个结，我们重新开始吧。",
            "意识流有点波动，让我重新整理一下思绪。"
        ]

        error_message = str(error) if isinstance(error, (Exception, str)) else "未知错误"
        self.logger.error(f"生成错误响应: {error_message}")

        response = random.choice(error_responses)

        # 添加调试信息（仅在开发模式）
        if self.config.log_level == "DEBUG":
            response += f" （系统日志: {error_message[:50]}...）"

        return response

    def process_input(self, user_input: str) -> str:
        """处理用户输入"""
        self.logger.info(f"开始处理用户输入: {user_input[:100]}...")

        try:
            # 创建处理上下文
            context = StageContext(
                user_input=user_input,
                conversation_history=self.memory_manager.get_conversation_history(limit=100),
                system_components=self.components,
                api_service_provider=self.api_service_provider  # 传递API服务提供者
            )

            # 执行处理管道
            context = self.pipeline.execute(context)

            # 调试：记录context对象的所有属性
            self.logger.info(f"Pipeline执行后的context属性:")
            for attr in ['final_reply', 'instinct_override', 'errors', 'conversation_history', 'retrieved_memories', 'resonant_memory', 'cognitive_result', 'growth_result']:
                if hasattr(context, attr):
                    value = getattr(context, attr)
                    self.logger.info(f"  {attr}: type={type(value)}, value={repr(value)[:100] if isinstance(value, (str, dict, list)) else value}")

            # 处理结果
            if context.final_reply:
                # 记录API使用情况
                if hasattr(context, 'api_metadata'):
                    self._record_api_usage(context.api_metadata)

                # 记录交互到记忆系统
                try:
                    interaction_data = {
                        "user_input": user_input,
                        "system_response": context.final_reply,
                        "interaction_id": str(uuid.uuid4()),
                        "context": {
                            "conversation_history_length": len(context.conversation_history) if isinstance(context.conversation_history, list) else 0
                        },
                        "retrieved_memories_count": len(context.retrieved_memories) if hasattr(context, 'retrieved_memories') and isinstance(context.retrieved_memories, list) else 0,
                        "resonant_memory": bool(context.resonant_memory) if hasattr(context, 'resonant_memory') and isinstance(context.resonant_memory, dict) else False,
                        "cognitive_result": bool(context.cognitive_result) if hasattr(context, 'cognitive_result') and isinstance(context.cognitive_result, dict) else False,
                        "growth_result": context.growth_result if hasattr(context, 'growth_result') and isinstance(context.growth_result, dict) else None
                    }
                    self.memory_manager.record_interaction(interaction_data)
                except Exception as e:
                    self.logger.error(f"记录交互失败: {e}")
                    import traceback
                    self.logger.error(traceback.format_exc())

                self.logger.info(f"处理成功，响应长度: {len(context.final_reply)}")
                return context.final_reply
            elif context.instinct_override and context.instinct_override.get("response"):
                self.logger.warning("处理被本能接管")
                return context.instinct_override["response"]
            else:
                # 生成错误响应
                error_response = self._generate_error_response(
                    context.errors[-1] if context.errors else "未知错误",
                    user_input
                )
                self.logger.error(f"处理失败，使用错误响应: {context.errors}")
                return error_response

        except Exception as e:
            self.logger.error(f"处理过程中发生异常: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return self._generate_error_response(e, user_input)

    def _record_api_usage(self, metadata: Dict):
        """记录API使用情况"""
        # 这里可以记录API使用指标，用于监控和计费
        if metadata and "tokens_used" in metadata:
            self.logger.debug(f"API使用: {metadata['tokens_used']} tokens, 延迟: {metadata.get('latency', 0):.2f}s")

    def get_system_status(self) -> Dict[str, Any]:
        """获取系统状态"""
        # 获取当前组件状态
        component_status = {}
        for name, component in self.components.items():
            if hasattr(component, 'get_status'):
                component_status[name] = component.get_status()

        # 更新记忆统计
        memory_stats = self.memory_manager.get_statistics()

        # 获取API服务状态
        api_status = self.api_service_provider.get_overall_status()

        # 获取欲望向量
        desire_vectors = {}
        desire_engine = self.components.get("desire_engine")
        if desire_engine and hasattr(desire_engine, 'get_vectors'):
            desire_vectors = desire_engine.get_vectors()

        return {
            **self.system_status,
            "current_time": datetime.now().isoformat(),
            "conversation_history_length": len(self.memory_manager.get_conversation_history()),
            "desire_vectors": desire_vectors,
            "dashboard_state": self.components.get("dashboard", {}).get_current_state()
            if hasattr(self.components.get("dashboard"), 'get_current_state') else {},
            "component_status": component_status,
            "memory_system_stats": memory_stats,
            "api_service_status": api_status,  # 新增API状态
            "pipeline_performance": self._get_pipeline_performance()
        }

    def _get_pipeline_performance(self) -> Dict[str, Any]:
        """获取管道性能数据"""
        # 这里可以记录和返回管道的性能指标
        return {
            "stages_executed": len(self.pipeline.stage_order),
            "parallelism_enabled": self.pipeline.enable_parallelism,
            "max_workers": self.pipeline.max_workers
        }

    def save_memory_system(self, file_path: Optional[str] = None) -> bool:
        """保存记忆系统"""
        if not file_path:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            file_path = f"./memory_backups/{self.config.session_id}_{timestamp}.json"

        # 确保目录存在，如果不存在则创建（安全方式）
        directory = os.path.dirname(file_path)
        os.makedirs(directory, exist_ok=True)  # exist_ok=True 确保目录不存在时创建，存在时不会报错 [[7]]

        self.logger.info(f"正在保存记忆系统到: {file_path}")
        return self.memory_manager.export_memories(file_path)

    def cleanup_memory(self, max_history: int = 1000):
        """清理记忆"""
        self.logger.info(f"开始清理记忆，最大历史: {max_history}")
        self.memory_manager.cleanup_memory(max_memories=max_history)

        # 清理缓存
        if hasattr(self.memory_manager, 'clear_cache'):
            self.memory_manager.clear_cache()

        self.logger.info("记忆清理完成")

    def shutdown(self):
        """关闭系统"""
        self.logger.info("正在关闭系统...")

        # 关闭处理管道
        self.pipeline.shutdown()

        # 关闭API服务提供者
        self.api_service_provider.shutdown()

        # 保存记忆
        self.save_memory_system()

        # 关闭其他资源
        for name, component in self.components.items():
            if hasattr(component, 'shutdown'):
                try:
                    component.shutdown()
                except Exception as e:
                    self.logger.error(f"关闭组件 {name} 失败: {e}")

        self.logger.info("系统关闭完成")