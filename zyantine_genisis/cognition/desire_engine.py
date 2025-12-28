"""
欲望引擎：向量化情感系统
TR/CS/SA三维情感向量 - 完整实现
"""

from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import random
import math
import numpy as np
from dataclasses import dataclass
from enum import Enum
from collections import deque
import json

class EmotionalState(Enum):
    """情感状态枚举"""
    CALM = "平静"
    EXCITED = "兴奋"
    CONTENT = "满足"
    ANXIOUS = "焦虑"
    MIXED = "混合"
    TRANSITION = "过渡"

@dataclass
class EmotionalEvent:
    """情感事件记录"""
    timestamp: datetime
    description: str
    impact_tr: float
    impact_cs: float
    impact_sa: float
    duration_seconds: float
    intensity: float
    tags: List[str]
    context: Dict[str, Any]

class ChemicalState:
    """化学状态容器"""
    def __init__(self):
        # 核心神经递质
        self.dopamine = 0.5      # 多巴胺 - 奖励、动力 (0-1)
        self.oxytocin = 0.6      # 催产素 - 连接、信任 (0-1)
        self.serotonin = 0.55    # 血清素 - 稳定、满足 (0-1)
        self.cortisol = 0.3      # 皮质醇 - 压力 (0-1)
        self.norepinephrine = 0.35  # 去甲肾上腺素 - 警觉 (0-1)

        # 代谢率（每秒衰减率）
        self.metabolic_rates = {
            "dopamine": 0.005,
            "oxytocin": 0.003,
            "serotonin": 0.002,
            "cortisol": 0.008,
            "norepinephrine": 0.01
        }

    def apply_decay(self, seconds: float):
        """应用时间衰减"""
        for attr in ["dopamine", "oxytocin", "serotonin", "cortisol", "norepinephrine"]:
            current = getattr(self, attr)
            rate = self.metabolic_rates[attr]
            baseline = 0.3 if attr in ["cortisol", "norepinephrine"] else 0.4

            # 指数衰减向基线值回归
            decay_factor = math.exp(-rate * seconds)
            new_value = baseline + (current - baseline) * decay_factor
            setattr(self, attr, max(0.0, min(1.0, new_value)))

    def to_dict(self) -> Dict:
        """转换为字典"""
        return {
            "dopamine": self.dopamine,
            "oxytocin": self.oxytocin,
            "serotonin": self.serotonin,
            "cortisol": self.cortisol,
            "norepinephrine": self.norepinephrine
        }

    def __repr__(self) -> str:
        return (f"ChemicalState(D={self.dopamine:.2f}, O={self.oxytocin:.2f}, "
                f"S={self.serotonin:.2f}, C={self.cortisol:.2f}, N={self.norepinephrine:.2f})")

class DesireEngine:
    """欲望引擎：向量化情感系统"""

    def __init__(self, config: Optional[Dict] = None):
        # 核心情感向量 (0.0-1.0)
        self.TR = 0.6  # 兴奋/奖励 - 成就、新奇、掌控、探索、创造
        self.CS = 0.7  # 满足/安全 - 信任、归属、安全、平静、和谐
        self.SA = 0.3  # 压力/警觉 - 威胁、混乱、焦虑、冲突、不确定性

        # 向量动量（前一时刻的变化率）
        self.vector_momentum = {"TR": 0.0, "CS": 0.0, "SA": 0.0}

        # 向量惯性系数
        self.inertia = {
            "TR": 0.3,  # 兴奋惯性较低，容易变化
            "CS": 0.5,  # 满足惯性较高，较难改变
            "SA": 0.2   # 压力惯性最低，容易波动
        }

        # 化学状态
        self.chemicals = ChemicalState()

        # 情感历史记录
        self.history = deque(maxlen=1000)
        self.emotional_events: List[EmotionalEvent] = []

        # 情感状态持久性
        self.current_state = EmotionalState.CALM
        self.state_duration = timedelta(0)
        self.state_stability = 0.7

        # 配置参数
        self.config = config or self._default_config()
        self._validate_config()

        # 时间跟踪
        self.last_update_time = datetime.now()
        self.circadian_modulator = 1.0

        # 初始化记录
        self._record_state("initialization")

    def _default_config(self) -> Dict:
        """默认配置"""
        return {
            # 衰减率（每秒）
            "decay_rates": {
                "TR": 0.004,  # 兴奋衰减较快
                "CS": 0.002,  # 满足衰减较慢
                "SA": 0.006   # 压力衰减最快
            },

            # 饱和效应阈值
            "saturation_thresholds": {
                "high": 0.85,  # 高位饱和点
                "low": 0.15    # 低位饱和点
            },

            # 最小变化阈值
            "min_delta_threshold": 0.01,

            # 化学协同矩阵
            "chemical_synergy": {
                ("dopamine", "oxytocin"): 0.2,    # 奖励加强连接
                ("oxytocin", "serotonin"): 0.3,   # 连接促进稳定
                ("serotonin", "dopamine"): 0.1,   # 稳定支持奖励
                ("cortisol", "norepinephrine"): 0.4,  # 压力增强警觉
                ("serotonin", "cortisol"): -0.3,  # 稳定抑制压力
                ("oxytocin", "cortisol"): -0.2    # 连接降低压力
            },

            # 昼夜节律影响
            "circadian_impact": {
                "morning": 1.1,    # 6-12点
                "afternoon": 1.0,  # 12-18点
                "evening": 0.9,    # 18-24点
                "night": 0.8       # 0-6点
            },

            # 情感事件记录设置
            "event_recording": {
                "min_intensity": 0.1,
                "max_events": 500
            }
        }

    def _validate_config(self):
        """验证配置参数"""
        # 确保所有值在合理范围内
        for vector, rate in self.config["decay_rates"].items():
            if not 0.001 <= rate <= 0.1:
                raise ValueError(f"衰减率 {vector}: {rate} 必须在0.001-0.1范围内")

    def update_vectors(self, interaction_context: Dict) -> Dict[str, float]:
        """
        更新情感向量

        Args:
            interaction_context: 交互上下文，包含情感事件信息

        Returns:
            更新后的向量状态和变化量
        """
        previous_state = {"TR": self.TR, "CS": self.CS, "SA": self.SA}
        current_time = datetime.now()
        time_delta = (current_time - self.last_update_time).total_seconds()

        # 1. 应用时间衰减
        self._apply_time_decay(time_delta)

        # 2. 计算交互影响
        context_impact = self._calculate_context_impact(interaction_context)

        # 3. 更新化学状态
        chemical_impact = self._calculate_chemical_impact(context_impact)
        self._update_chemicals(chemical_impact, time_delta)

        # 4. 应用情感更新（考虑化学影响和惯性）
        vector_deltas = self._apply_emotional_update(
            context_impact,
            chemical_impact,
            time_delta
        )

        # 5. 检测并记录情感事件
        emotional_event = self._detect_emotional_event(
            interaction_context,
            vector_deltas
        )
        if emotional_event:
            self.emotional_events.append(emotional_event)
            if len(self.emotional_events) > self.config["event_recording"]["max_events"]:
                self.emotional_events = self.emotional_events[-self.config["event_recording"]["max_events"]:]

        # 6. 更新情感状态分类
        self._update_emotional_state()

        # 7. 更新昼夜节律调节因子
        self._update_circadian_modulator()

        # 8. 记录状态
        self._record_state(interaction_context.get("interaction_type", "unknown"))
        self.last_update_time = current_time

        # 9. 返回结果
        return {
            "TR": round(self.TR, 3),
            "CS": round(self.CS, 3),
            "SA": round(self.SA, 3),
            "delta_TR": round(self.TR - previous_state["TR"], 3),
            "delta_CS": round(self.CS - previous_state["CS"], 3),
            "delta_SA": round(self.SA - previous_state["SA"], 3),
            "chemical_state": self.chemicals.to_dict(),
            "emotional_state": self.current_state.value,
            "state_stability": round(self.state_stability, 3),
            "state_duration_seconds": self.state_duration.total_seconds()
        }

    def update(self, user_input: str, retrieved_memories: List, impact_factors: Dict) -> Dict[str, float]:
        """
        更新欲望向量（兼容旧版本接口）

        Args:
            user_input: 用户输入
            retrieved_memories: 检索到的记忆
            impact_factors: 影响因子

        Returns:
            更新后的欲望向量
        """
        # 构建交互上下文
        interaction_context = self._build_interaction_context(
            user_input=user_input,
            retrieved_memories=retrieved_memories,
            impact_factors=impact_factors
        )

        # 调用现有的 update_vectors 方法
        return self.update_vectors(interaction_context)

    def _build_interaction_context(self, user_input: str, retrieved_memories: List,
                                   impact_factors: Dict) -> Dict:
        """构建交互上下文"""
        # 分析用户输入的情感
        sentiment = self._analyze_sentiment(user_input)

        # 基于记忆内容判断事件类型
        event_type = self._determine_event_type(user_input, retrieved_memories)

        # 计算强度
        intensity = impact_factors.get("overall_impact", 0.5)

        # 提取关键词作为标签
        tags = self._extract_tags(user_input, retrieved_memories)

        return {
            "description": user_input[:100],  # 截取前100个字符
            "sentiment": sentiment,
            "intensity": intensity,
            "event_type": event_type,
            "duration_seconds": 0.0,  # 默认持续时间
            "tags": tags,
            "metadata": {
                "user_input": user_input,
                "memories_count": len(retrieved_memories),
                "impact_factors": impact_factors
            }
        }

    def _analyze_sentiment(self, text: str) -> float:
        """分析文本情感（简化版本）"""
        positive_words = ['好', '喜欢', '爱', '开心', '快乐', '高兴', '感谢', '谢谢', '棒', '优秀', '成功']
        negative_words = ['不好', '讨厌', '恨', '难过', '伤心', '生气', '愤怒', '糟糕', '差', '垃圾', '失败']

        text_lower = text.lower()

        pos_count = sum(1 for word in positive_words if word in text_lower)
        neg_count = sum(1 for word in negative_words if word in text_lower)

        if pos_count > neg_count:
            return 0.5 + min(0.5, pos_count * 0.1)
        elif neg_count > pos_count:
            return -0.5 - min(0.5, neg_count * 0.1)
        else:
            return 0.0  # 中性

    def _determine_event_type(self, user_input: str, retrieved_memories: List) -> str:
        """确定事件类型"""
        # 基于关键词判断事件类型
        text_lower = user_input.lower()

        # 检查不同类型的词汇
        achievement_words = ['完成', '成功', '成就', '胜利', '突破']
        praise_words = ['谢谢', '感谢', '赞', '表扬', '认可']
        learning_words = ['学习', '知道', '了解', '研究', '发现']
        connection_words = ['朋友', '关系', '聊天', '交流', '沟通']
        trust_words = ['信任', '相信', '可靠', '依赖']
        conflict_words = ['问题', '困难', '挑战', '矛盾', '冲突']
        threat_words = ['危险', '威胁', '害怕', '担心', '恐惧']
        uncertainty_words = ['不确定', '不知道', '疑问', '困惑', '迷茫']
        overload_words = ['太多', '太忙', '压力', '累', '疲惫']

        if any(word in text_lower for word in achievement_words):
            return "achievement"
        elif any(word in text_lower for word in praise_words):
            return "praise"
        elif any(word in text_lower for word in learning_words):
            return "learning"
        elif any(word in text_lower for word in connection_words):
            return "connection"
        elif any(word in text_lower for word in trust_words):
            return "trust"
        elif any(word in text_lower for word in conflict_words):
            return "conflict"
        elif any(word in text_lower for word in threat_words):
            return "threat"
        elif any(word in text_lower for word in uncertainty_words):
            return "uncertainty"
        elif any(word in text_lower for word in overload_words):
            return "overload"
        else:
            return "neutral"

    def _extract_tags(self, user_input: str, retrieved_memories: List) -> List[str]:
        """提取标签"""
        tags = []

        # 提取用户输入中的名词或关键词
        import re
        words = re.findall(r'[\u4e00-\u9fff]{2,}', user_input)
        tags.extend(words[:3])  # 只取前3个词

        # 添加记忆标签
        for memory in retrieved_memories[:2]:  # 只取前2个记忆
            if isinstance(memory, dict) and 'tags' in memory:
                tags.extend(memory['tags'][:2])

        return list(set(tags))[:5]  # 去重，最多5个标签

    def _apply_time_decay(self, seconds: float):
        """应用时间衰减"""
        # 指数衰减公式: V_new = V_baseline + (V_current - V_baseline) * exp(-rate * time)
        baselines = {"TR": 0.4, "CS": 0.5, "SA": 0.3}
        decay_rates = self.config["decay_rates"]

        for vector in ["TR", "CS", "SA"]:
            current = getattr(self, vector)
            baseline = baselines[vector]
            rate = decay_rates[vector]

            decay_factor = math.exp(-rate * seconds)
            new_value = baseline + (current - baseline) * decay_factor

            # 应用饱和效应
            new_value = self._apply_saturation_effect(vector, new_value)

            setattr(self, vector, new_value)

        # 化学状态衰减
        self.chemicals.apply_decay(seconds)

    def _apply_saturation_effect(self, vector: str, value: float) -> float:
        """应用饱和效应"""
        thresholds = self.config["saturation_thresholds"]

        if value > thresholds["high"]:
            # 接近上限时，增加阻力
            excess = (value - thresholds["high"]) / (1.0 - thresholds["high"])
            resistance = 1.0 - (excess * 0.5)  # 最多减少50%
            value = thresholds["high"] + (value - thresholds["high"]) * resistance

        elif value < thresholds["low"]:
            # 接近下限时，减少下降速度
            deficiency = (thresholds["low"] - value) / thresholds["low"]
            support = 1.0 - (deficiency * 0.3)  # 最多增加30%
            value = thresholds["low"] - (thresholds["low"] - value) * support

        return max(0.0, min(1.0, value))

    def _calculate_context_impact(self, context: Dict) -> Dict[str, float]:
        """计算上下文影响"""
        impact = {"TR": 0.0, "CS": 0.0, "SA": 0.0}

        # 提取情感标记
        sentiment = context.get("sentiment", 0.0)  # -1到1
        intensity = context.get("intensity", 0.5)  # 0到1
        event_type = context.get("event_type", "neutral")

        # 基于事件类型的影响
        event_impacts = {
            "achievement": {"TR": 0.4, "CS": 0.1, "SA": -0.1},
            "praise": {"TR": 0.2, "CS": 0.3, "SA": -0.2},
            "learning": {"TR": 0.3, "CS": 0.1, "SA": 0.0},
            "connection": {"TR": 0.1, "CS": 0.4, "SA": -0.2},
            "trust": {"TR": 0.0, "CS": 0.5, "SA": -0.3},
            "conflict": {"TR": -0.1, "CS": -0.3, "SA": 0.4},
            "threat": {"TR": -0.2, "CS": -0.4, "SA": 0.5},
            "uncertainty": {"TR": -0.1, "CS": -0.2, "SA": 0.3},
            "overload": {"TR": -0.3, "CS": -0.4, "SA": 0.6},
            "neutral": {"TR": 0.0, "CS": 0.0, "SA": 0.0}
        }

        # 应用事件影响
        event_impact = event_impacts.get(event_type, event_impacts["neutral"])
        for vector in ["TR", "CS", "SA"]:
            impact[vector] += event_impact.get(vector, 0.0)

        # 应用情感极性影响
        if sentiment > 0.3:  # 积极情感
            impact["TR"] += 0.2 * sentiment
            impact["CS"] += 0.1 * sentiment
            impact["SA"] -= 0.15 * sentiment
        elif sentiment < -0.3:  # 消极情感
            impact["SA"] += 0.3 * abs(sentiment)
            impact["CS"] -= 0.2 * abs(sentiment)

        # 应用强度调节（非线性）
        intensity_factor = self._sigmoid_intensity(intensity)
        for vector in impact:
            impact[vector] *= intensity_factor

        # 应用昼夜节律调节
        for vector in impact:
            impact[vector] *= self.circadian_modulator

        return impact

    def _sigmoid_intensity(self, intensity: float) -> float:
        """S型强度曲线（边际递减效应）"""
        k = 5.0  # 曲线陡峭度
        return 2.0 / (1.0 + math.exp(-k * (intensity - 0.5)))

    def _calculate_chemical_impact(self, context_impact: Dict) -> Dict[str, float]:
        """计算化学影响"""
        chemical_impact = {
            "dopamine": 0.0,
            "oxytocin": 0.0,
            "serotonin": 0.0,
            "cortisol": 0.0,
            "norepinephrine": 0.0
        }

        # TR主要影响多巴胺和去甲肾上腺素
        chemical_impact["dopamine"] += context_impact["TR"] * 0.6
        chemical_impact["norepinephrine"] += context_impact["TR"] * 0.2

        # CS主要影响催产素和血清素
        chemical_impact["oxytocin"] += context_impact["CS"] * 0.7
        chemical_impact["serotonin"] += context_impact["CS"] * 0.3

        # SA主要影响皮质醇和去甲肾上腺素
        chemical_impact["cortisol"] += context_impact["SA"] * 0.8
        chemical_impact["norepinephrine"] += context_impact["SA"] * 0.4

        return chemical_impact

    def _update_chemicals(self, chemical_impact: Dict, seconds: float):
        """更新化学状态"""
        # 应用直接影响
        for chem, impact in chemical_impact.items():
            current = getattr(self.chemicals, chem)
            new_value = current + impact
            setattr(self.chemicals, chem, max(0.0, min(1.0, new_value)))

        # 应用协同效应
        synergy_matrix = self.config["chemical_synergy"]
        for (chem1, chem2), synergy in synergy_matrix.items():
            val1 = getattr(self.chemicals, chem1)
            synergy_effect = (val1 - 0.5) * synergy * 0.05
            current_val2 = getattr(self.chemicals, chem2)
            setattr(self.chemicals, chem2,
                   max(0.0, min(1.0, current_val2 + synergy_effect)))

    def _apply_emotional_update(self,
                               context_impact: Dict,
                               chemical_impact: Dict,
                               seconds: float) -> Dict[str, float]:
        """应用情感更新"""
        raw_deltas = {}
        final_deltas = {}

        # 计算基础变化
        for vector in ["TR", "CS", "SA"]:
            raw_deltas[vector] = context_impact[vector]

        # 应用化学调节
        chem_modifiers = self._calculate_chemical_modifiers()
        for vector, modifier in chem_modifiers.items():
            raw_deltas[vector] *= (1.0 + modifier)

        # 应用惯性和动量
        for vector in ["TR", "CS", "SA"]:
            inertia = self.inertia[vector]
            delta = (raw_deltas[vector] * (1.0 - inertia) +
                    self.vector_momentum[vector] * inertia)

            # 应用最小变化阈值
            if abs(delta) < self.config["min_delta_threshold"]:
                delta = 0.0

            # 应用饱和效应（基于当前值）
            current = getattr(self, vector)
            saturation_adjusted = self._adjust_for_saturation(vector, current, delta)

            # 更新向量值
            new_value = max(0.0, min(1.0, current + saturation_adjusted))
            setattr(self, vector, new_value)

            # 计算实际变化量
            actual_delta = new_value - current
            final_deltas[vector] = actual_delta

            # 更新动量（当前变化影响未来变化）
            self.vector_momentum[vector] = (actual_delta * 0.3 +
                                           self.vector_momentum[vector] * 0.7)

        return final_deltas

    def _calculate_chemical_modifiers(self) -> Dict[str, float]:
        """计算化学调节因子"""
        # 化学物质对情感向量的调节作用
        modifiers = {
            "TR": (self.chemicals.dopamine * 0.6 +
                  self.chemicals.norepinephrine * 0.4 -
                  self.chemicals.cortisol * 0.2) - 0.4,  # 减去基线

            "CS": (self.chemicals.oxytocin * 0.7 +
                  self.chemicals.serotonin * 0.3 -
                  self.chemicals.cortisol * 0.1) - 0.4,

            "SA": (self.chemicals.cortisol * 0.8 +
                  self.chemicals.norepinephrine * 0.2 -
                  self.chemicals.serotonin * 0.3) - 0.3
        }

        return modifiers

    def _adjust_for_saturation(self, vector: str, current: float, delta: float) -> float:
        """根据饱和情况调整变化量"""
        thresholds = self.config["saturation_thresholds"]

        if delta > 0 and current > thresholds["high"]:
            # 在高位时增加阻力
            excess = (current - thresholds["high"]) / (1.0 - thresholds["high"])
            resistance = 1.0 - excess * 0.7
            delta *= resistance

        elif delta < 0 and current < thresholds["low"]:
            # 在低位时增加支撑
            deficiency = (thresholds["low"] - current) / thresholds["low"]
            support = 1.0 - deficiency * 0.5
            delta *= support

        return delta

    def _detect_emotional_event(self, context: Dict, deltas: Dict) -> Optional[EmotionalEvent]:
        """检测情感事件"""
        # 计算事件总强度
        total_intensity = sum(abs(delta) for delta in deltas.values())

        if total_intensity < self.config["event_recording"]["min_intensity"]:
            return None

        # 创建情感事件记录
        event = EmotionalEvent(
            timestamp=datetime.now(),
            description=context.get("description", "未知事件"),
            impact_tr=deltas["TR"],
            impact_cs=deltas["CS"],
            impact_sa=deltas["SA"],
            duration_seconds=context.get("duration_seconds", 0.0),
            intensity=total_intensity,
            tags=context.get("tags", []),
            context=context.get("metadata", {})
        )

        return event

    def _update_emotional_state(self):
        """更新情感状态分类"""
        previous_state = self.current_state

        # 确定主导向量
        values = {"TR": self.TR, "CS": self.CS, "SA": self.SA}
        max_vector = max(values, key=values.get)
        max_value = values[max_vector]

        # 判断当前状态
        if max_value < 0.3:
            new_state = EmotionalState.CALM
        elif max_vector == "TR" and self.TR > 0.6:
            new_state = EmotionalState.EXCITED
        elif max_vector == "CS" and self.CS > 0.6:
            new_state = EmotionalState.CONTENT
        elif max_vector == "SA" and self.SA > 0.6:
            new_state = EmotionalState.ANXIOUS
        elif abs(self.TR - self.CS) < 0.2 and abs(self.CS - self.SA) < 0.2:
            new_state = EmotionalState.MIXED
        else:
            new_state = EmotionalState.TRANSITION

        # 更新状态持续时间和稳定性
        if new_state == previous_state:
            self.state_duration += (datetime.now() - self.last_update_time)
            self.state_stability = min(1.0, self.state_stability + 0.01)
        else:
            self.state_duration = timedelta(0)
            self.state_stability = max(0.0, self.state_stability - 0.1)
            self.current_state = new_state

    def _update_circadian_modulator(self):
        """更新昼夜节律调节因子"""
        current_hour = datetime.now().hour

        circadian_config = self.config["circadian_impact"]

        if 6 <= current_hour < 12:
            self.circadian_modulator = circadian_config["morning"]
        elif 12 <= current_hour < 18:
            self.circadian_modulator = circadian_config["afternoon"]
        elif 18 <= current_hour < 24:
            self.circadian_modulator = circadian_config["evening"]
        else:
            self.circadian_modulator = circadian_config["night"]

    def _record_state(self, event_type: str):
        """记录状态历史"""
        state_record = {
            "timestamp": datetime.now().isoformat(),
            "TR": round(self.TR, 4),
            "CS": round(self.CS, 4),
            "SA": round(self.SA, 4),
            "chemicals": self.chemicals.to_dict(),
            "emotional_state": self.current_state.value,
            "state_stability": round(self.state_stability, 3),
            "state_duration_seconds": self.state_duration.total_seconds(),
            "circadian_modulator": round(self.circadian_modulator, 2),
            "event_type": event_type
        }

        self.history.append(state_record)

    def get_current_state(self) -> Dict:
        """获取当前状态"""
        return {
            "vectors": {
                "TR": round(self.TR, 3),
                "CS": round(self.CS, 3),
                "SA": round(self.SA, 3)
            },
            "chemicals": self.chemicals.to_dict(),
            "emotional_state": self.current_state.value,
            "state_stability": round(self.state_stability, 3),
            "state_duration": str(self.state_duration),
            "momentum": {k: round(v, 3) for k, v in self.vector_momentum.items()}
        }

    def get_history_summary(self, window_minutes: int = 60) -> Dict:
        """获取历史摘要"""
        cutoff_time = datetime.now() - timedelta(minutes=window_minutes)

        recent_history = [
            record for record in self.history
            if datetime.fromisoformat(record["timestamp"]) > cutoff_time
        ]

        if not recent_history:
            return {"error": "指定时间段内无历史记录"}

        # 计算统计数据
        tr_values = [r["TR"] for r in recent_history]
        cs_values = [r["CS"] for r in recent_history]
        sa_values = [r["SA"] for r in recent_history]

        return {
            "time_window_minutes": window_minutes,
            "record_count": len(recent_history),
            "vector_stats": {
                "TR": {
                    "mean": round(np.mean(tr_values), 3),
                    "std": round(np.std(tr_values), 3),
                    "min": round(min(tr_values), 3),
                    "max": round(max(tr_values), 3),
                    "trend": round(tr_values[-1] - tr_values[0], 3) if len(tr_values) > 1 else 0
                },
                "CS": {
                    "mean": round(np.mean(cs_values), 3),
                    "std": round(np.std(cs_values), 3),
                    "min": round(min(cs_values), 3),
                    "max": round(max(cs_values), 3),
                    "trend": round(cs_values[-1] - cs_values[0], 3) if len(cs_values) > 1 else 0
                },
                "SA": {
                    "mean": round(np.mean(sa_values), 3),
                    "std": round(np.std(sa_values), 3),
                    "min": round(min(sa_values), 3),
                    "max": round(max(sa_values), 3),
                    "trend": round(sa_values[-1] - sa_values[0], 3) if len(sa_values) > 1 else 0
                }
            },
            "state_distribution": self._calculate_state_distribution(recent_history),
            "recent_events": [e.description for e in self.emotional_events[-5:]] if self.emotional_events else []
        }

    def _calculate_state_distribution(self, history: List[Dict]) -> Dict:
        """计算状态分布"""
        state_counts = {}
        for record in history:
            state = record.get("emotional_state", "unknown")
            state_counts[state] = state_counts.get(state, 0) + 1

        total = len(history)
        if total == 0:
            return {}

        return {state: round(count/total * 100, 1) for state, count in state_counts.items()}

    def get_detailed_analysis(self) -> Dict:
        """获取详细分析"""
        # 分析当前情感构成
        vector_balance = {
            "dominant_vector": max(["TR", "CS", "SA"], key=lambda v: getattr(self, v)),
            "vector_ratios": {
                "TR_CS_ratio": round(self.TR / self.CS, 2) if self.CS > 0 else float('inf'),
                "TR_SA_ratio": round(self.TR / self.SA, 2) if self.SA > 0 else float('inf'),
                "CS_SA_ratio": round(self.CS / self.SA, 2) if self.SA > 0 else float('inf')
            },
            "emotional_complexity": self._calculate_emotional_complexity()
        }

        # 化学平衡分析
        chemical_analysis = {
            "balance_score": self._calculate_chemical_balance_score(),
            "dominant_chemical": self._get_dominant_chemical(),
            "imbalances": self._detect_chemical_imbalances()
        }

        # 系统健康度
        health_metrics = {
            "stability_score": round(self.state_stability, 3),
            "adaptability_score": self._calculate_adaptability_score(),
            "resilience_score": self._calculate_resilience_score(),
            "emotional_range": self._calculate_emotional_range()
        }

        return {
            "timestamp": datetime.now().isoformat(),
            "vector_analysis": vector_balance,
            "chemical_analysis": chemical_analysis,
            "health_metrics": health_metrics,
            "recommendations": self._generate_recommendations()
        }

    def _calculate_emotional_complexity(self) -> float:
        """计算情感复杂度"""
        # 基于向量差异的复杂度计算
        vector_diff = abs(self.TR - self.CS) + abs(self.CS - self.SA) + abs(self.SA - self.TR)
        max_possible_diff = 2.0  # 最大差异（当两个向量为0，一个为1时）
        complexity = vector_diff / max_possible_diff
        return round(complexity, 3)

    def _calculate_chemical_balance_score(self) -> float:
        """计算化学平衡分数"""
        # 理想值：多巴胺=0.5, 催产素=0.6, 血清素=0.55, 皮质醇=0.3, 去甲肾上腺素=0.35
        ideal_values = {
            "dopamine": 0.5,
            "oxytocin": 0.6,
            "serotonin": 0.55,
            "cortisol": 0.3,
            "norepinephrine": 0.35
        }

        total_deviation = 0.0
        for chem, ideal in ideal_values.items():
            current = getattr(self.chemicals, chem)
            deviation = abs(current - ideal)
            total_deviation += deviation

        # 归一化为0-1分数（越低越好）
        max_possible_deviation = 5.0  # 5个化学物质，每个最大偏差1.0
        balance_score = 1.0 - (total_deviation / max_possible_deviation)
        return round(balance_score, 3)

    def _get_dominant_chemical(self) -> str:
        """获取主导化学物质"""
        chemical_values = self.chemicals.to_dict()
        return max(chemical_values, key=chemical_values.get)

    def _detect_chemical_imbalances(self) -> List[str]:
        """检测化学失衡"""
        imbalances = []
        chemical_values = self.chemicals.to_dict()

        # 检查各个化学物质水平
        if chemical_values["cortisol"] > 0.7:
            imbalances.append("皮质醇过高 - 长期压力")
        if chemical_values["dopamine"] < 0.3:
            imbalances.append("多巴胺不足 - 缺乏动力")
        if chemical_values["oxytocin"] < 0.4:
            imbalances.append("催产素不足 - 连接感弱")
        if chemical_values["serotonin"] < 0.4:
            imbalances.append("血清素不足 - 情绪不稳定")
        if chemical_values["norepinephrine"] > 0.6:
            imbalances.append("去甲肾上腺素过高 - 过度警觉")

        return imbalances

    def _calculate_adaptability_score(self) -> float:
        """计算适应能力分数"""
        # 基于最近历史的变化频率
        if len(self.history) < 10:
            return 0.5

        recent = list(self.history)[-10:]
        state_changes = 0

        for i in range(1, len(recent)):
            if recent[i]["emotional_state"] != recent[i-1]["emotional_state"]:
                state_changes += 1

        # 适度的变化率最好（既不太僵化也不太不稳定）
        ideal_changes = 3  # 10次交互中3次变化
        adaptability = 1.0 - abs(state_changes - ideal_changes) / ideal_changes
        return round(max(0.0, min(1.0, adaptability)), 3)

    def _calculate_resilience_score(self) -> float:
        """计算恢复能力分数"""
        # 基于从压力状态恢复的能力
        if len(self.emotional_events) < 5:
            return 0.5

        high_stress_events = [e for e in self.emotional_events[-10:]
                             if e.impact_sa > 0.3]

        if not high_stress_events:
            return 0.7  # 没有高压事件，假设恢复能力强

        recovery_times = []
        for event in high_stress_events:
            # 寻找事件后的恢复情况（简化实现）
            # 在实际应用中，需要更精确的时间序列分析
            pass

        # 简化实现：基于当前SA水平
        if self.SA < 0.3:
            resilience = 0.8
        elif self.SA < 0.5:
            resilience = 0.6
        elif self.SA < 0.7:
            resilience = 0.4
        else:
            resilience = 0.2

        return round(resilience, 3)

    def _calculate_emotional_range(self) -> float:
        """计算情感范围"""
        # 最近历史中情感向量的变化范围
        if len(self.history) < 20:
            return 0.3

        recent = list(self.history)[-20:]

        tr_range = max(r["TR"] for r in recent) - min(r["TR"] for r in recent)
        cs_range = max(r["CS"] for r in recent) - min(r["CS"] for r in recent)
        sa_range = max(r["SA"] for r in recent) - min(r["SA"] for r in recent)

        avg_range = (tr_range + cs_range + sa_range) / 3.0
        return round(avg_range, 3)

    def _generate_recommendations(self) -> List[str]:
        """生成改进建议"""
        recommendations = []

        # 基于向量状态
        if self.TR < 0.3:
            recommendations.append("需要增加兴奋度，尝试新的活动或挑战")
        elif self.TR > 0.8:
            recommendations.append("兴奋度过高，建议适当休息或进行放松活动")

        if self.CS < 0.4:
            recommendations.append("安全感不足，建议加强社交连接或建立信任关系")
        elif self.CS > 0.8:
            recommendations.append("满足感很高，保持当前状态")

        if self.SA > 0.6:
            recommendations.append("压力水平较高，建议进行压力管理或寻求支持")
        elif self.SA < 0.2:
            recommendations.append("警觉度过低，可能需要增加一些挑战性任务")

        # 基于化学状态
        chemical_state = self.chemicals.to_dict()
        if chemical_state["cortisol"] > 0.6:
            recommendations.append("检测到长期压力迹象，建议增加放松和恢复时间")
        if chemical_state["oxytocin"] < 0.4:
            recommendations.append("连接感化学物质偏低，建议增加积极社交互动")

        return recommendations

    def export_state(self, filepath: str):
        """导出当前状态"""
        state_data = {
            "timestamp": datetime.now().isoformat(),
            "vectors": {
                "TR": self.TR,
                "CS": self.CS,
                "SA": self.SA
            },
            "chemicals": self.chemicals.to_dict(),
            "emotional_state": self.current_state.value,
            "history_length": len(self.history),
            "events_count": len(self.emotional_events),
            "config": self.config
        }

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(state_data, f, ensure_ascii=False, indent=2)

        print(f"[DesireEngine] 状态已导出到: {filepath}")

    def reset(self, new_tr: float = 0.6, new_cs: float = 0.7, new_sa: float = 0.3):
        """重置引擎状态"""
        self.TR = new_tr
        self.CS = new_cs
        self.SA = new_sa

        self.vector_momentum = {"TR": 0.0, "CS": 0.0, "SA": 0.0}
        self.chemicals = ChemicalState()
        self.current_state = EmotionalState.CALM
        self.state_duration = timedelta(0)
        self.state_stability = 0.7

        self.history.clear()
        self.emotional_events.clear()

        self._record_state("reset")
        print(f"[DesireEngine] 已重置到 TR={new_tr}, CS={new_cs}, SA={new_sa}")

# # 使用示例
# if __name__ == "__main__":
#     # 创建欲望引擎
#     desire_engine = DesireEngine()
#
#     # 模拟一系列交互
#     interactions = [
#         {"description": "完成重要任务", "sentiment": 0.8, "intensity": 0.7, "event_type": "achievement"},
#         {"description": "收到积极反馈", "sentiment": 0.6, "intensity": 0.5, "event_type": "praise"},
#         {"description": "遇到技术难题", "sentiment": -0.4, "intensity": 0.6, "event_type": "conflict"},
#         {"description": "与朋友深入交流", "sentiment": 0.7, "intensity": 0.4, "event_type": "connection"}
#     ]
#
#     for i, interaction in enumerate(interactions, 1):
#         print(f"\n=== 交互 {i}: {interaction['description']} ===")
#         result = desire_engine.update_vectors(interaction)
#
#         print(f"情感向量: TR={result['TR']}, CS={result['CS']}, SA={result['SA']}")
#         print(f"情感状态: {result['emotional_state']}")
#         print(f"状态稳定性: {result['state_stability']}")
#
#     # 获取当前状态
#     print("\n=== 当前状态 ===")
#     current_state = desire_engine.get_current_state()
#     print(f"情感向量: {current_state['vectors']}")
#     print(f"化学状态: {current_state['chemicals']}")
#     print(f"情感状态: {current_state['emotional_state']}")
#
#     # 获取详细分析
#     print("\n=== 详细分析 ===")
#     analysis = desire_engine.get_detailed_analysis()
#     print(f"向量分析: {analysis['vector_analysis']}")
#     print(f"化学分析: {analysis['chemical_analysis']['balance_score']}")
#     print(f"健康指标: {analysis['health_metrics']}")
#     print(f"建议: {analysis['recommendations']}")