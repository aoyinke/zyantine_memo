# 修改所有相对导入为绝对导入
from cognition.core_identity import CoreIdentity
from cognition.meta_cognition import MetaCognitionModule
from cognition.cognitive_flow_manager import CognitiveFlowManager
from cognition.internal_state_dashboard import InternalStateDashboard
from memory.memory_manager import MemoryManager
from protocols.fact_checker import FactChecker
from protocols.expression_validator import ExpressionValidator
from protocols.length_regulator import LengthRegulator
from protocols.protocol_engine import ProtocolEngine
from api.reply_generator import APIBasedReplyGenerator
from typing import Dict


class ComponentManager:
    """组件管理器"""

    def __init__(self, config: Dict):
        self.config = config
        self.components = {}

    def initialize_components(self):
        """初始化所有组件"""
        # 初始化核心身份
        self.components['core_identity'] = CoreIdentity()

        # 初始化记忆系统
        self.components['memory_manager'] = MemoryManager(self.config)

        # 初始化内部状态仪表盘
        self.components['internal_state_dashboard'] = InternalStateDashboard()

        # 初始化元认知模块
        self.components['meta_cognition'] = MetaCognitionModule(
            self.components['internal_state_dashboard']
        )

        # 初始化协议层组件
        self._init_protocol_components()

        # 初始化认知流程管理器
        self.components['cognitive_flow'] = CognitiveFlowManager(
            self.components['core_identity'],
            self.components['memory_manager'],
            self.components['meta_cognition'],
            self.components['fact_checker']
        )

        return self.components

    def _init_protocol_components(self):
        """初始化协议层组件"""
        # 初始化表达验证器
        self.components['expression_validator'] = ExpressionValidator()

        # 初始化长度规整器
        self.components['length_regulator'] = LengthRegulator()

        # 初始化事实检查器（需要记忆管理器和API服务）
        memory_manager = self.components.get('memory_manager')
        api_service = self.components.get('api_service')
        self.components['fact_checker'] = FactChecker(
            memory_manager=memory_manager,
            api_service=api_service
        )

        # 初始化协议引擎
        self.components['protocol_engine'] = ProtocolEngine(
            fact_checker=self.components['fact_checker'],
            length_regulator=self.components['length_regulator'],
            expression_validator=self.components['expression_validator']
        )