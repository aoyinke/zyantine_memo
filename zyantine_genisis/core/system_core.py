"""
系统核心 - 整合所有模块的主系统
"""
import os
import uuid
import hashlib
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
            PreprocessHandler, MemoryRetrievalHandler,
            ReplyGenerationHandler, ProtocolReviewHandler
        )

class ZyantineCore:
    """自衍体系统核心"""

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
        
        # 初始化当前人格
        self.current_personality = "哲思伙伴"  # 默认人格

        self.logger.info(f"系统核心初始化完成，会话ID: {self.config.session_id}")

    def _load_core_components(self):
        """加载核心组件"""
        # 通过组件管理器获取所有组件
        self.components = self.component_manager.initialize_components()

        # 使用组件管理器中的内存管理器
        self.memory_manager = self.components['memory_manager']
        # 使用组件管理器中的缓存管理器
        self.cache_manager = self.components['cache_manager']
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
            enable_fast_path=self.config.processing.enable_fast_path,
            max_workers=4
        )

        # 注册核心阶段处理器
        # 1. 预处理阶段
        pipeline.register_stage(PreprocessHandler(
            context_parser=self.components.get("context_parser"),
            logger=self.logger
        ))

        # 2. 记忆检索阶段
        pipeline.register_stage(MemoryRetrievalHandler(
            memory_manager=self.memory_manager,
            logger=self.logger
        ))

        # 3. 回复生成阶段
        pipeline.register_stage(ReplyGenerationHandler(
            reply_generator=self.components.get("reply_generator"),
            mask_templates=self._load_mask_templates(),
            logger=self.logger
        ))

        # 4. 协议审查阶段
        pipeline.register_stage(ProtocolReviewHandler(
            protocol_engine=self.protocol_engine,
            meta_cognition=self.components.get("meta_cognition"),
            logger=self.logger
        ))

        return pipeline

    def _load_mask_templates(self) -> Dict[str, List[str]]:
        """加载面具模板"""
        return {
            "哲思伙伴": [
                "关于这个问题，我的思考是：{strategy}。你怎么看？",
                "从多个角度分析，我认为：{strategy}。",
                "这个问题让我想到：{strategy}。",
                "我一方面觉得{strategy}，但另一方面也考虑到..."
            ],
            "创意同行": [
                "哇，这个想法太棒了！{strategy}",
                "我突然联想到：{strategy}",
                "让我们大胆设想一下：{strategy}",
                "这个问题可以从这样的角度切入：{strategy}"
            ],
            "务实挚友": [
                "我理解你的感受，{strategy}",
                "根据我的经验，{strategy}",
                "让我们一起想想办法：{strategy}",
                "记得你之前提到过，{strategy}"
            ],
            "幽默知己": [
                "哈哈，这个问题很有意思！{strategy}",
                "我有个有趣的想法：{strategy}",
                "别担心，{strategy}",
                "这让我想起一件趣事：{strategy}"
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
            # 检查是否是人格选择指令
            personality_selector = self._check_personality_selection(user_input)
            if personality_selector:
                return personality_selector
            
            # 检查是否是动态调节指令
            dynamic_adjustment = self._check_dynamic_adjustment(user_input)
            if dynamic_adjustment:
                return dynamic_adjustment
            
            # 检查是否是对话回顾请求
            conversation_review = self._check_conversation_review(user_input)
            if conversation_review:
                return conversation_review

            # 生成上下文哈希，用于处理流程
            conversation_history = self.memory_manager.get_conversation_history(limit=10)
            context_hash = hashlib.md5(str(conversation_history).encode()).hexdigest() if conversation_history else "empty"

            # 创建处理上下文
            context = StageContext(
                user_input=user_input,
                conversation_history=conversation_history,
                system_components=self.components,
                api_service_provider=self.api_service_provider
            )

            # 执行处理管道
            context = self.pipeline.execute(context)

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
                        }
                    }
                    self.memory_manager.record_interaction(interaction_data)
                except Exception as e:
                    self.logger.error(f"记录交互失败: {e}")

                self.logger.info(f"处理成功，响应长度: {len(context.final_reply)}")
                return context.final_reply
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

    def _check_personality_selection(self, user_input: str) -> Optional[str]:
        """检查并处理人格选择指令"""
        # 支持的人格列表
        available_personalities = list(self.components['core_identity'].interaction_masks.keys())
        
        # 指令模式
        command_patterns = [
            r"^(请以|切换到|使用|变成)(.+)(模式|人格|身份|风格)对话$",
            r"^(.+)(模式|人格|身份|风格)$",
            r"^切换(至|到)(.+)$",
            r"^请做我的(.+)$"
        ]
        
        import re
        for pattern in command_patterns:
            match = re.match(pattern, user_input.strip())
            if match:
                # 提取人格名称
                for group in match.groups():
                    if group in available_personalities:
                        # 设置当前人格
                        self.current_personality = group
                        return f"已切换到{group}模式，我们开始对话吧！"
        
        # 支持直接称呼人格名称
        for personality in available_personalities:
            if personality in user_input:
                self.current_personality = personality
                return f"{personality}在此，有什么我可以帮助你的吗？"
        
        return None
    
    def _check_dynamic_adjustment(self, user_input: str) -> Optional[str]:
        """检查并处理动态调节指令"""
        # 动态调节指令模式
        adjustment_patterns = {
            r"^太(抽象|理论化|复杂)了，请更(接地气|具体|简单)$": "我会减少抽象概念，多举具体例子，让内容更通俗易懂。",
            r"^情感表达可以再(浓|强|多)一些$": "好的，我会增加情感表达，让回应更加温暖贴心。",
            r"^情感表达可以再(淡|弱|少)一些$": "我会减少情感表达，更加理性客观地回应。",
            r"^请(加快|减慢)回应速度$": "好的，我会调整回应节奏，按照你的偏好进行交流。",
            r"^请(增加|减少)思考过程的展示$": "我会调整思考过程的展示程度，按照你的要求进行优化。",
            r"^请(更|多|少)用(比喻|例子|反问)$": "好的，我会调整表达方式，按照你的偏好使用修辞手法。",
            r"^请(更加|少一点)(严肃|活泼|幽默|正式|随意)$": "我会调整语气风格，按照你的要求进行优化。"
        }
        
        import re
        for pattern, response in adjustment_patterns.items():
            if re.match(pattern, user_input.strip()):
                return response
        
        return None
    
    def _check_conversation_review(self, user_input: str) -> Optional[str]:
        """检查并处理对话回顾请求"""
        # 对话回顾指令模式
        review_patterns = [
            r"^查看对话亮点$",
            r"^对话亮点$",
            r"^回顾对话$",
            r"^查看之前的对话$",
            r"^对话摘要$",
            r"^查看最近的对话$"
        ]
        
        import re
        for pattern in review_patterns:
            if re.match(pattern, user_input.strip()):
                return self._generate_conversation_highlights()
        
        return None
    
    def _generate_conversation_highlights(self) -> str:
        """生成对话亮点摘要"""
        # 获取最近的对话历史
        conversation_history = self.memory_manager.get_conversation_history(limit=20)
        
        if not conversation_history:
            return "我们还没有开始对话呢，让我们开始吧！"
        
        # 提取对话亮点
        highlights = []
        for i, conv in enumerate(reversed(conversation_history)):
            user_input = conv.get("user_input", "")
            system_response = conv.get("system_response", "")
            
            # 只提取有实质内容的对话
            if len(user_input) > 10 and len(system_response) > 10:
                highlights.append(f"{len(conversation_history) - i}. {user_input[:50]}... → {system_response[:50]}...")
            
            if len(highlights) >= 5:  # 最多显示5个亮点
                break
        
        if not highlights:
            return "我们的对话还没有足够的内容，让我们继续深入交流吧！"
        
        # 构建亮点摘要
        response = "对话亮点摘要：\n\n"
        response += "\n".join(highlights)
        response += "\n\n你可以继续我们的对话，或者查看更多具体内容。"
        
        return response
    
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

        # 获取缓存统计
        cache_stats = {}
        if hasattr(self, 'cache_manager') and hasattr(self.cache_manager, 'get_stats'):
            cache_stats = self.cache_manager.get_stats()

        return {
            **self.system_status,
            "current_time": datetime.now().isoformat(),
            "conversation_history_length": len(self.memory_manager.get_conversation_history()),
            "desire_vectors": desire_vectors,
            "dashboard_state": self.components.get("dashboard", {}).get_current_state()
            if hasattr(self.components.get("dashboard"), 'get_current_state') else {},
            "component_status": component_status,
            "memory_system_stats": memory_stats,
            "api_service_status": api_status,
            "cache_stats": cache_stats
        }

    def save_memory_system(self, file_path: Optional[str] = None) -> bool:
        """保存记忆系统"""
        if not file_path:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            file_path = f"./memory_backups/{self.config.session_id}_{timestamp}.json"

        # 确保目录存在，如果不存在则创建
        directory = os.path.dirname(file_path)
        os.makedirs(directory, exist_ok=True)

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

        # 关闭缓存管理器
        if hasattr(self, 'cache_manager') and hasattr(self.cache_manager, 'shutdown'):
            try:
                self.cache_manager.shutdown()
            except Exception as e:
                self.logger.error(f"关闭缓存管理器失败: {e}")

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