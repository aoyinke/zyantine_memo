# 修改所有相对导入为绝对导入
from cognition.core_identity import CoreIdentity
from cognition.context_parser import ContextParser
from cognition.meta_cognition import MetaCognitionModule
from cognition.cognitive_flow_manager import CognitiveFlowManager
from cognition.internal_state_dashboard import InternalStateDashboard
from cognition.desire_engine import DesireEngine
from cognition.dialectical_growth import DialecticalGrowth
from memory.memory_manager import MemoryManager
from protocols.fact_checker import FactChecker
from protocols.expression_validator import ExpressionValidator
from protocols.length_regulator import LengthRegulator
from protocols.protocol_engine import ProtocolEngine
from api.service_provider import APIServiceProvider
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
        # 初始化基础服务
        self.components['api_service_provider'] = APIServiceProvider(self.config,core_identity=self.components['core_identity'])
        self.components['reply_generator'] = self.components['api_service_provider'].reply_generator
        # 初始化记忆系统 - 传递API服务
        self.components['memory_manager'] = MemoryManager(
            config=self.config,
        )

        # 初始化其他组件...
        self.components['desire_engine'] = DesireEngine()
        creator_anchor = {
            "default": {
                "concept": "真诚、善良、好奇、成长",
                "expected_response": "基于核心价值观的回应"
            },
            "emotional_support": {
                "concept": "共情与支持",
                "expected_response": "先处理情绪，再处理问题"
            },
            "technical_question": {
                "concept": "专业与准确",
                "expected_response": "提供准确、专业的解答"
            }
        }
        self.components['dialectical_growth'] = DialecticalGrowth(creator_anchor)
        self.components['internal_state_dashboard'] = InternalStateDashboard()
        self.components['context_parser'] = ContextParser()
        # 初始化元认知模块
        self.components['meta_cognition'] = MetaCognitionModule(
            self.components['internal_state_dashboard'],
            self.components['context_parser']
        )

        # 初始化协议层组件（现在有API服务了）
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
        # 获取API服务
        api_service = self.components.get('api_service_provider')

        # 初始化表达验证器
        self.components['expression_validator'] = ExpressionValidator()

        # 初始化长度规整器
        self.components['length_regulator'] = LengthRegulator()

        # 初始化事实检查器
        self.components['fact_checker'] = FactChecker(
            memory_manager=self.components.get('memory_manager'),
            api_service=api_service
        )

        # 初始化协议引擎
        self.components['protocol_engine'] = ProtocolEngine(
            fact_checker=self.components['fact_checker'],
            length_regulator=self.components['length_regulator'],
            expression_validator=self.components['expression_validator']
        )