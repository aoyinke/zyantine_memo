"""
核心支柱模块
包含四大支柱：核心本能、欲望引擎、辩证成长、核心身份
"""
from .genesis_metadata import GenesisMetadata
from .instinctual_core import InstinctualCore
from .desire_engine import DesireEngine
from .dialectical_growth import DialecticalGrowth

__all__ = [
    'GenesisMetadata',
    'InstinctualCore',
    'DesireEngine',
    'DialecticalGrowth'
]