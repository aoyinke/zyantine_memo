"""
认知模块 (Cognition Module)

该模块实现了自衍体的认知系统，包括：
- 认知流程管理
- 情境解析
- 核心身份
- 欲望引擎
- 辩证成长
- 内在状态仪表盘
- 元认知

使用示例：
    from cognition import CognitiveFlowManager, ContextParser, CoreIdentity
    
    # 初始化组件
    context_parser = ContextParser()
    core_identity = CoreIdentity()
"""

from cognition.cognitive_flow_manager import CognitiveFlowManager
from cognition.context_parser import ContextParser
from cognition.core_identity import CoreIdentity
from cognition.desire_engine import DesireEngine, EmotionalState, ChemicalState
from cognition.dialectical_growth import DialecticalGrowth
from cognition.internal_state_dashboard import InternalStateDashboard
from cognition.meta_cognition import MetaCognitionModule

__all__ = [
    # 核心组件
    "CognitiveFlowManager",
    "ContextParser", 
    "CoreIdentity",
    "DesireEngine",
    "DialecticalGrowth",
    "InternalStateDashboard",
    "MetaCognitionModule",
    # 辅助类
    "EmotionalState",
    "ChemicalState",
]

__version__ = "1.0.0"
