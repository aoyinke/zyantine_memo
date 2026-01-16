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
from core.cache_manager import CacheManager
from typing import Dict, List, Optional, Tuple
from utils.logger import SystemLogger
import time


class ComponentStatus:
    """组件状态枚举"""
    CREATED = "created"
    INITIALIZED = "initialized"
    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"
    FAILED = "failed"


class ComponentInfo:
    """组件信息类"""
    def __init__(self, name, cls, dependencies=None, kwargs=None):
        self.name = name
        self.cls = cls
        self.dependencies = dependencies or []
        self.kwargs = kwargs or {}
        self.status = ComponentStatus.CREATED
        self.instance = None
        self.error = None


class ComponentManager:
    """组件管理器"""

    def __init__(self, config: Dict):
        self.config = config
        self.components = {}
        self.component_status = {}
        self.logger = SystemLogger().get_logger("component_manager")
        self._component_registry = {
            # 基础组件
            'cache_manager': ComponentInfo(
                'cache_manager',
                CacheManager,
                kwargs={
                    'max_size': self.config.processing.cache_max_size,
                    'default_ttl': self.config.processing.cache_default_ttl,
                    'enable_response_cache': self.config.processing.enable_response_cache,
                    'enable_memory_cache': self.config.processing.enable_memory_cache,
                    'enable_context_cache': self.config.processing.enable_context_cache
                }
            ),
            'core_identity': ComponentInfo('core_identity', CoreIdentity),
            'api_service_provider': ComponentInfo(
                'api_service_provider',
                APIServiceProvider,
                dependencies=['core_identity'],
                kwargs={'config': self.config}
            ),
            'memory_manager': ComponentInfo(
                'memory_manager',
                MemoryManager,
                kwargs={'config': self.config}
            ),
            'reply_generator': ComponentInfo(
                'reply_generator',
                None,  # 特殊组件，从api_service_provider获取
                dependencies=['api_service_provider']
            ),
            
            # 认知组件
            'desire_engine': ComponentInfo('desire_engine', DesireEngine),
            'internal_state_dashboard': ComponentInfo('internal_state_dashboard', InternalStateDashboard),
            'context_parser': ComponentInfo('context_parser', ContextParser),
            'meta_cognition': ComponentInfo(
                'meta_cognition',
                MetaCognitionModule,
                dependencies=['internal_state_dashboard', 'context_parser']
            ),
            
            # 协议组件
            'expression_validator': ComponentInfo('expression_validator', ExpressionValidator),
            'length_regulator': ComponentInfo('length_regulator', LengthRegulator),
            'fact_checker': ComponentInfo(
                'fact_checker',
                FactChecker,
                dependencies=['memory_manager', 'api_service_provider']
            ),
            'protocol_engine': ComponentInfo(
                'protocol_engine',
                ProtocolEngine,
                dependencies=['fact_checker', 'length_regulator', 'expression_validator']
            ),
            
            # 认知流程组件
            'cognitive_flow': ComponentInfo(
                'cognitive_flow',
                CognitiveFlowManager,
                dependencies=['core_identity', 'memory_manager', 'meta_cognition', 'fact_checker']
            ),
            
            # 辩证成长组件
            'dialectical_growth': ComponentInfo(
                'dialectical_growth',
                DialecticalGrowth,
                kwargs={'creator_anchor': {
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
                }}
            )
        }

    def _topological_sort(self) -> List[ComponentInfo]:
        """拓扑排序组件依赖关系"""
        # Kahn's algorithm for topological sorting
        in_degree = {name: 0 for name in self._component_registry}
        adjacency = {name: [] for name in self._component_registry}
        
        # 构建入度表和邻接表
        for name, info in self._component_registry.items():
            for dep in info.dependencies:
                adjacency[dep].append(name)
                in_degree[name] += 1
        
        # 初始化队列
        queue = [name for name, degree in in_degree.items() if degree == 0]
        result = []
        
        # 执行拓扑排序
        while queue:
            name = queue.pop(0)
            result.append(name)
            
            for neighbor in adjacency[name]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)
        
        # 检查是否有循环依赖
        if len(result) != len(self._component_registry):
            raise ValueError("组件依赖关系存在循环")
        
        return [self._component_registry[name] for name in result]

    def _initialize_component(self, info: ComponentInfo) -> bool:
        """初始化单个组件"""
        try:
            if info.name == 'reply_generator':
                # 特殊处理：从api_service_provider获取
                api_service = self.components['api_service_provider']
                info.instance = api_service.reply_generator
                info.status = ComponentStatus.INITIALIZED
                self.logger.info(f"组件 {info.name} 初始化成功")
                return True
            
            # 准备依赖项
            kwargs = info.kwargs.copy()
            for dep in info.dependencies:
                if dep == 'internal_state_dashboard':
                    # MetaCognitionModule需要internal_dashboard参数
                    kwargs['internal_dashboard'] = self.components[dep]
                elif dep == 'context_parser':
                    # MetaCognitionModule需要context_parser参数
                    kwargs['context_parser'] = self.components[dep]
                elif dep == 'core_identity':
                    kwargs['core_identity'] = self.components[dep]
                elif dep == 'api_service_provider' and info.name == 'fact_checker':
                    # FactChecker需要api_service参数，而不是api_service_provider
                    kwargs['api_service'] = self.components[dep]
                elif dep == 'memory_manager' and info.name == 'fact_checker':
                    # FactChecker需要memory_manager参数
                    kwargs['memory_manager'] = self.components[dep]
                else:
                    # 对于其他依赖，使用依赖名称作为参数名
                    kwargs[dep] = self.components[dep]
            
            # 创建组件实例
            info.instance = info.cls(**kwargs)
            info.status = ComponentStatus.INITIALIZED
            self.logger.info(f"组件 {info.name} 初始化成功")
            return True
        except Exception as e:
            info.status = ComponentStatus.FAILED
            info.error = str(e)
            self.logger.error(f"组件 {info.name} 初始化失败: {e}")
            return False

    def _check_component_health(self, info: ComponentInfo) -> bool:
        """检查组件健康状态"""
        try:
            if info.name == 'reply_generator':
                # 特殊处理：检查API服务是否可用
                api_service = self.components['api_service_provider']
                if api_service.reply_generator.api:
                    return True
                return False
            
            # 检查组件是否有health_check方法
            if hasattr(info.instance, 'health_check'):
                return info.instance.health_check()
            
            # 检查组件是否有is_available方法
            if hasattr(info.instance, 'is_available'):
                return info.instance.is_available()
            
            # 默认健康
            return True
        except Exception as e:
            self.logger.error(f"组件 {info.name} 健康检查失败: {e}")
            return False

    def initialize_components(self):
        """初始化所有组件"""
        self.logger.info("开始初始化组件...")
        
        # 拓扑排序组件
        sorted_components = self._topological_sort()
        
        # 初始化组件
        for info in sorted_components:
            if self._initialize_component(info):
                self.components[info.name] = info.instance
                
                # 健康检查
                if self._check_component_health(info):
                    info.status = ComponentStatus.HEALTHY
                else:
                    info.status = ComponentStatus.UNHEALTHY
                    self.logger.warning(f"组件 {info.name} 健康检查未通过")
            else:
                self.logger.error(f"组件 {info.name} 初始化失败，无法继续")
                raise RuntimeError(f"组件 {info.name} 初始化失败")
        
        # 初始化协议层组件
        self._init_protocol_components()
        
        self.logger.info(f"成功初始化 {len(self.components)} 个组件")
        return self.components

    def _init_protocol_components(self):
        """初始化协议层组件"""
        # 这些组件已经在拓扑排序中初始化
        self.logger.info("协议层组件初始化完成")

    def get_component_status(self, component_name: Optional[str] = None) -> Dict:
        """获取组件状态"""
        if component_name:
            if component_name in self._component_registry:
                info = self._component_registry[component_name]
                return {
                    'name': info.name,
                    'status': info.status,
                    'error': info.error
                }
            return {'name': component_name, 'status': 'not_found'}
        
        # 返回所有组件状态
        return {
            info.name: {
                'status': info.status,
                'error': info.error
            }
            for info in self._component_registry.values()
        }

    def check_all_components_health(self) -> Dict[str, bool]:
        """检查所有组件健康状态"""
        results = {}
        for name, info in self._component_registry.items():
            if info.status in [ComponentStatus.HEALTHY, ComponentStatus.UNHEALTHY]:
                results[name] = self._check_component_health(info)
            else:
                results[name] = False
        return results

    def restart_component(self, component_name: str) -> bool:
        """重启单个组件"""
        if component_name not in self._component_registry:
            self.logger.error(f"组件 {component_name} 不存在")
            return False
        
        info = self._component_registry[component_name]
        self.logger.info(f"正在重启组件 {component_name}...")
        
        # 重新初始化组件
        if self._initialize_component(info):
            self.components[component_name] = info.instance
            
            # 健康检查
            if self._check_component_health(info):
                info.status = ComponentStatus.HEALTHY
            else:
                info.status = ComponentStatus.UNHEALTHY
            
            self.logger.info(f"组件 {component_name} 重启成功")
            return True
        else:
            self.logger.error(f"组件 {component_name} 重启失败")
            return False

    def restart_unhealthy_components(self) -> List[str]:
        """重启所有不健康的组件"""
        restarted = []
        health_results = self.check_all_components_health()
        
        for name, is_healthy in health_results.items():
            if not is_healthy:
                if self.restart_component(name):
                    restarted.append(name)
        
        return restarted

    def shutdown(self):
        """关闭所有组件"""
        self.logger.info("正在关闭组件...")
        
        for name, component in self.components.items():
            if hasattr(component, 'shutdown'):
                try:
                    component.shutdown()
                    self.logger.info(f"组件 {name} 关闭成功")
                except Exception as e:
                    self.logger.error(f"组件 {name} 关闭失败: {e}")
        
        self.logger.info("所有组件已关闭")