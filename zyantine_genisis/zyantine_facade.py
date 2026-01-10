"""
外观模式入口 - 提供简化的系统接口
"""
from typing import Dict, Any, Optional, List, Union
import json
import os
from datetime import datetime

from core.system_core import ZyantineCore
from config.config_manager import ConfigManager
# 修改导入路径
from utils.logger import get_logger, get_structured_logger
from utils.metrics import get_collector, increment_counter
# 新增导入
from cognition.core_identity import CoreIdentity
from cognition.cognitive_flow_manager import CognitiveFlowManager


class ZyantineFacade:
    """自衍体系统外观类 - 简化接口"""

    def __init__(self,
                 config_path: Optional[str] = None,
                 user_profile: Optional[Dict] = None,
                 self_profile: Optional[Dict] = None,
                 session_id: Optional[str] = None,
                 use_new_cognitive_flow: bool = False):
        """
        初始化系统

        Args:
            config_path: 配置文件路径
            user_profile: 用户配置文件
            self_profile: 自衍体配置文件
            session_id: 会话ID
            use_new_cognitive_flow: 是否使用新的认知流程（默认False）
        """
        # 初始化日志
        self.logger = get_logger("facade")
        self.structured_logger = get_structured_logger("facade_structured")
        self.metrics = get_collector("facade")

        self.logger.info(f"自衍体AI系统初始化中...")
        self.structured_logger.info("系统初始化开始",
                                    config_path=config_path,
                                    session_id=session_id)

        # 加载配置
        self.config_manager = ConfigManager()
        self.config = self.config_manager.load(config_path)

        # 更新会话ID
        if session_id:
            self.config.session_id = session_id

        # 是否使用新的认知流程
        self.use_new_cognitive_flow = use_new_cognitive_flow

        # 初始化核心系统
        self.core = ZyantineCore(
            config_file=config_path,
            user_profile_data=user_profile,
            self_profile_data=self_profile,
            session_id=session_id
        )

        # 如果使用新的认知流程，初始化相关组件
        if self.use_new_cognitive_flow:
            self._init_cognitive_flow_components()

        self.logger.info(f"自衍体AI系统 v{self.config.version} 初始化完成")
        self.logger.info(f"会话ID: {self.config.session_id}")
        self.logger.info(f"系统模式: {self.config.processing.mode.value}")
        self.logger.info(f"记忆系统: {self.config.memory.system_type.value}")

        if self.use_new_cognitive_flow:
            self.logger.info(f"认知流程: 增强版（新架构）")
        else:
            self.logger.info(f"认知流程: 标准版")

        # 记录指标
        increment_counter("system.startup")
        self.metrics.set_gauge("system.initialized", 1)

    def _init_cognitive_flow_components(self):
        """初始化认知流程组件"""
        self.logger.info("初始化增强版认知流程组件...")

        try:
            # 初始化核心身份
            self.core_identity = CoreIdentity()

            # 通过核心系统获取其他组件
            # 假设ZyantineCore有方法获取这些组件
            memory_manager = self.core.get_memory_manager()
            meta_cognition = self.core.get_meta_cognition()
            fact_checker = self.core.get_fact_checker()

            # 初始化认知流程管理器
            self.cognitive_flow_manager = CognitiveFlowManager(
                core_identity=self.core_identity,
                memory_manager=memory_manager,
                meta_cognition=meta_cognition,
                fact_checker=fact_checker
            )

            self.logger.info("增强版认知流程组件初始化完成")
            increment_counter("cognitive_flow.initialized")

        except Exception as e:
            self.logger.error(f"认知流程组件初始化失败: {e}")
            self.use_new_cognitive_flow = False
            increment_counter("cognitive_flow.init_failed")

    def chat(self, user_input: str, use_enhanced_flow: Optional[bool] = None, stream: bool = False) -> Union[str, Any]:
        """
        与系统对话

        Args:
            user_input: 用户输入
            use_enhanced_flow: 是否使用增强版认知流程（None时使用系统默认设置）
            stream: 是否流式输出

        Returns:
            系统响应（字符串或流式生成器）
        """
        if not user_input or not user_input.strip():
            return "我收到了空消息，有什么我可以帮助你的吗？"

        self.logger.info(f"接收用户输入: {user_input[:100]}...")
        self.structured_logger.info("处理用户输入",
                                    input_preview=user_input[:50],
                                    input_length=len(user_input))

        # 记录指标
        stop_timer = self.metrics.start_timer("chat.response_time")
        increment_counter("chat.requests")

        try:
            # 决定使用哪个流程
            if use_enhanced_flow is None:
                use_enhanced_flow = self.use_new_cognitive_flow

            if use_enhanced_flow and hasattr(self, 'cognitive_flow_manager'):
                self.logger.debug("使用增强版认知流程")
                response = self._chat_with_enhanced_flow(user_input, stream=stream)
                increment_counter("chat.enhanced_flow_used")
            else:
                self.logger.debug("使用标准流程")
                response = self.core.process_input(user_input)
                increment_counter("chat.standard_flow_used")

            if stream and hasattr(response, '__iter__') and not isinstance(response, str):
                # 流式模式：返回生成器，不记录日志直到生成完成
                return response
            else:
                # 记录响应时间和长度
                response_time = stop_timer()
                response_length = len(response)

                self.metrics.record_histogram("chat.response_length", response_length)
                self.metrics.set_gauge("chat.last_response_time", response_time)

                self.logger.info(f"响应生成成功，长度: {response_length}, 耗时: {response_time:.3f}秒")
                self.structured_logger.info("响应生成完成",
                                            response_preview=response[:50],
                                            response_length=response_length,
                                            response_time=response_time)

                return response

        except Exception as e:
            self.logger.error(f"处理消息时发生错误: {str(e)}")
            increment_counter("chat.errors")

            # 记录错误指标
            self.metrics.set_gauge("chat.error", 1)

            return "抱歉，我刚才遇到了一些技术问题，能请你再问一次吗？"

    def _chat_with_enhanced_flow(self, user_input: str, stream: bool = False) -> Union[str, Any]:
        """使用增强版认知流程进行对话"""
        try:
            # 获取对话历史
            conversation_history = self.core.get_conversation_history()

            # 获取当前向量状态（需要从核心系统获取）
            current_vectors = self.core.get_current_vectors()

            # 获取记忆上下文
            memory_context = self.core.get_memory_context(user_input, conversation_history)

            # 执行认知流程
            cognitive_result = self.cognitive_flow_manager.process_thought(
                user_input=user_input,
                history=conversation_history,
                current_vectors=current_vectors,
                memory_context=memory_context
            )

            # 使用回复生成器生成回复（假设核心系统能提供回复生成器）
            reply_generator = self.core.get_reply_generator()

            # 生成回复
            response = reply_generator.generate_from_cognitive_flow(cognitive_result, stream=stream)

            # 更新系统状态（如果需要）
            self.core.update_from_cognitive_result(cognitive_result)

            return response

        except Exception as e:
            self.logger.error(f"增强版认知流程失败: {e}")
            # 降级到标准流程
            return self.core.process_input(user_input)

    def get_status(self) -> Dict[str, Any]:
        """获取系统状态"""
        status = self.core.get_system_status()

        # 添加认知流程状态
        status["cognitive_flow"] = {
            "enhanced_version_enabled": self.use_new_cognitive_flow,
            "component_status": {
                "core_identity_initialized": hasattr(self, 'core_identity'),
                "cognitive_flow_manager_initialized": hasattr(self, 'cognitive_flow_manager'),
            }
        }

        if hasattr(self, 'cognitive_flow_manager'):
            status["cognitive_flow"]["manager_stats"] = {
                "total_thoughts_processed": len(self.cognitive_flow_manager.thought_log),
                "current_goal": self.cognitive_flow_manager.current_goal
            }

        return status

    def enable_enhanced_cognitive_flow(self, enable: bool = True) -> bool:
        """
        启用或禁用增强版认知流程

        Args:
            enable: 是否启用

        Returns:
            操作是否成功
        """
        if enable and not hasattr(self, 'cognitive_flow_manager'):
            # 初始化认知流程组件
            try:
                self._init_cognitive_flow_components()
                self.use_new_cognitive_flow = True
                return True
            except Exception as e:
                self.logger.error(f"启用增强版认知流程失败: {e}")
                return False
        else:
            self.use_new_cognitive_flow = enable
            return True

    def get_cognitive_flow_log(self, limit: int = 10) -> List[Dict]:
        """
        获取认知流程日志

        Args:
            limit: 日志数量限制

        Returns:
            认知流程日志
        """
        if hasattr(self, 'cognitive_flow_manager'):
            return self.cognitive_flow_manager.thought_log[-limit:] if self.cognitive_flow_manager.thought_log else []
        return []

    def save_memory(self, file_path: Optional[str] = None) -> bool:
        """保存记忆"""
        success = self.core.save_memory_system(file_path)

        if success:
            self.logger.info("记忆保存成功")
        else:
            self.logger.error("记忆保存失败")

        return success

    def cleanup(self, max_history: int = 1000) -> bool:
        """清理系统"""
        try:
            self.core.cleanup_memory(max_history)
            self.logger.info(f"系统清理完成，保留最近 {max_history} 条记忆")
            return True
        except Exception as e:
            self.logger.error(f"系统清理失败: {e}")
            return False

    def export_config(self, file_path: str) -> bool:
        """导出当前配置"""
        try:
            config_dict = self.config.to_dict()

            with open(file_path, 'w', encoding='utf-8') as f:
                if file_path.endswith('.json'):
                    json.dump(config_dict, f, ensure_ascii=False, indent=2)
                else:
                    import yaml
                    yaml.dump(config_dict, f, default_flow_style=False, allow_unicode=True)

            self.logger.info(f"配置已导出到: {file_path}")
            return True
        except Exception as e:
            self.logger.error(f"导出配置失败: {e}")
            return False

    def update_config(self, updates: Dict[str, Any]) -> bool:
        """更新配置"""
        try:
            self.config = self.config_manager.update(updates)
            self.logger.info("配置更新成功")
            return True
        except Exception as e:
            self.logger.error(f"配置更新失败: {e}")
            return False

    def shutdown(self):
        """关闭系统"""
        print("\n正在关闭系统...")
        self.core.shutdown()
        self.logger.info("系统已关闭")
        print("系统已安全关闭")


class SimpleZyantine:
    """简化版自衍体（适合快速集成）"""

    def __init__(self, api_key: Optional[str] = None, session_id: str = "default", use_enhanced_flow: bool = False):
        """快速初始化"""
        # 优先从环境变量获取API密钥
        api_key = api_key or os.getenv("ZYANTINE_API_KEY")
        
        if not api_key:
            raise ValueError("必须提供 api_key 参数或设置 ZYANTINE_API_KEY 环境变量")
            
        config = {
            "session_id": session_id,
            "api": {
                "api_key": api_key,
                "enabled": True
            },
            "processing": {
                "mode": "standard"
            }
        }

        self.facade = ZyantineFacade(
            config_path=None,
            session_id=session_id,
            use_new_cognitive_flow=use_enhanced_flow  # 新增参数
        )

    def reply(self, message: str, use_enhanced: Optional[bool] = None, stream: bool = False) -> Union[str, Any]:
        """简单回复"""
        return self.facade.chat(message, use_enhanced_flow=use_enhanced, stream=stream)

    def get_stats(self) -> Dict:
        """获取统计信息"""
        return self.facade.get_status()


# 快速使用示例
def create_zyantine(api_key: Optional[str] = None,
                    config_file: Optional[str] = None,
                    session_id: str = "default",
                    use_enhanced_flow: bool = False) -> ZyantineFacade:
    """
    快速创建自衍体实例

    Args:
        api_key: OpenAI API密钥（可选，如果不提供则使用配置文件）
        config_file: 配置文件路径（可选，默认使用config/llm_config.json）
        session_id: 会话ID
        use_enhanced_flow: 是否使用增强版认知流程

    Returns:
        ZyantineFacade实例
    """
    # 如果提供了配置文件，直接使用
    if config_file:
        return ZyantineFacade(
            config_path=config_file,
            session_id=session_id,
            use_new_cognitive_flow=use_enhanced_flow
        )

    # 如果没有提供配置文件，但提供了API密钥，则使用API密钥创建
    if api_key:
        config = {
            "api": {
                "api_key": api_key
            },
            "session_id": session_id
        }

        # 创建临时配置文件
        import tempfile
        import json

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(config, f)
            temp_config = f.name

        facade = ZyantineFacade(
            config_path=temp_config,
            session_id=session_id,
            use_new_cognitive_flow=use_enhanced_flow
        )

        # 删除临时文件
        os.unlink(temp_config)

        return facade
    else:
        # 如果既没有提供配置文件也没有提供API密钥，尝试使用默认配置文件
        default_config_file = "config/llm_config.json"
        if os.path.exists(default_config_file):
            return ZyantineFacade(
                config_path=default_config_file,
                session_id=session_id,
                use_new_cognitive_flow=use_enhanced_flow
            )
        else:
            raise ValueError("必须提供 api_key 参数或配置文件")

# if __name__ == "__main__":
#     # 使用标准流程
#     facade = ZyantineFacade(config_path="config.json", use_new_cognitive_flow=False)
#     response = facade.chat("你好")
#
#     # 使用增强版流程
#     facade_enhanced = ZyantineFacade(config_path="config.json", use_new_cognitive_flow=True)
#     response = facade_enhanced.chat("你好", use_enhanced_flow=True)
#
#     # 动态切换流程
#     facade.enable_enhanced_cognitive_flow(True)  # 启用增强版流程
#
#     # 查看认知流程日志
#     logs = facade.get_cognitive_flow_log(limit=5)