"""
欲望引擎：向量化情感系统
TR/CS/SA三维情感向量 - 完整实现

优化版本：
- 简化配置结构
- 改进类型提示
- 提取常量
- 优化性能
"""

from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import math
import json
from dataclasses import dataclass, field
from enum import Enum
from collections import deque

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False


# ============ 枚举和数据类 ============

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


@dataclass
class DesireEngineConfig:
    """欲望引擎配置"""
    # 衰减率（每秒）
    decay_rate_tr: float = 0.004
    decay_rate_cs: float = 0.002
    decay_rate_sa: float = 0.006
    
    # 基线值
    baseline_tr: float = 0.4
    baseline_cs: float = 0.5
    baseline_sa: float = 0.3
    
    # 饱和效应阈值
    saturation_high: float = 0.85
    saturation_low: float = 0.15
    
    # 最小变化阈值
    min_delta_threshold: float = 0.01
    
    # 事件记录设置
    min_event_intensity: float = 0.1
    max_events: int = 500
    max_history: int = 1000
    
    # 昼夜节律影响
    circadian_morning: float = 1.1
    circadian_afternoon: float = 1.0
    circadian_evening: float = 0.9
    circadian_night: float = 0.8


# 化学协同矩阵
CHEMICAL_SYNERGY = {
    ("dopamine", "oxytocin"): 0.2,
    ("oxytocin", "serotonin"): 0.3,
    ("serotonin", "dopamine"): 0.1,
    ("cortisol", "norepinephrine"): 0.4,
    ("serotonin", "cortisol"): -0.3,
    ("oxytocin", "cortisol"): -0.2
}

# 事件类型影响映射
EVENT_IMPACTS = {
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

# 化学物质理想值
CHEMICAL_IDEAL_VALUES = {
    "dopamine": 0.5,
    "oxytocin": 0.6,
    "serotonin": 0.55,
    "cortisol": 0.3,
    "norepinephrine": 0.35
}

# 化学物质代谢率
CHEMICAL_METABOLIC_RATES = {
    "dopamine": 0.005,
    "oxytocin": 0.003,
    "serotonin": 0.002,
    "cortisol": 0.008,
    "norepinephrine": 0.01
}


class ChemicalState:
    """化学状态容器"""
    
    def __init__(self):
        self.dopamine: float = 0.5
        self.oxytocin: float = 0.6
        self.serotonin: float = 0.55
        self.cortisol: float = 0.3
        self.norepinephrine: float = 0.35

    def apply_decay(self, seconds: float) -> None:
        """应用时间衰减"""
        for attr, rate in CHEMICAL_METABOLIC_RATES.items():
            current = getattr(self, attr)
            baseline = 0.3 if attr in ["cortisol", "norepinephrine"] else 0.4
            decay_factor = math.exp(-rate * seconds)
            new_value = baseline + (current - baseline) * decay_factor
            setattr(self, attr, max(0.0, min(1.0, new_value)))

    def to_dict(self) -> Dict[str, float]:
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

    def __init__(self, config: Optional[DesireEngineConfig] = None):
        self.config = config or DesireEngineConfig()
        
        # 核心情感向量 (0.0-1.0)
        self.TR: float = 0.6  # 兴奋/奖励
        self.CS: float = 0.7  # 满足/安全
        self.SA: float = 0.3  # 压力/警觉

        # 向量动量
        self.vector_momentum: Dict[str, float] = {"TR": 0.0, "CS": 0.0, "SA": 0.0}

        # 向量惯性系数
        self.inertia: Dict[str, float] = {"TR": 0.3, "CS": 0.5, "SA": 0.2}

        # 化学状态
        self.chemicals = ChemicalState()

        # 情感历史记录
        self.history: deque = deque(maxlen=self.config.max_history)
        self.emotional_events: List[EmotionalEvent] = []

        # 情感状态持久性
        self.current_state = EmotionalState.CALM
        self.state_duration = timedelta(0)
        self.state_stability: float = 0.7

        # 时间跟踪
        self.last_update_time = datetime.now()
        self.circadian_modulator: float = 1.0

        # 初始化记录
        self._record_state("initialization")

    def update_vectors(self, interaction_context: Dict) -> Dict[str, Any]:
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

        # 4. 应用情感更新
        vector_deltas = self._apply_emotional_update(context_impact, chemical_impact, time_delta)

        # 5. 检测并记录情感事件
        self._record_emotional_event(interaction_context, vector_deltas)

        # 6. 更新情感状态分类
        self._update_emotional_state()

        # 7. 更新昼夜节律调节因子
        self._update_circadian_modulator()

        # 8. 记录状态
        self._record_state(interaction_context.get("interaction_type", "unknown"))
        self.last_update_time = current_time

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

    def _apply_time_decay(self, seconds: float) -> None:
        """应用时间衰减"""
        decay_config = {
            "TR": (self.config.decay_rate_tr, self.config.baseline_tr),
            "CS": (self.config.decay_rate_cs, self.config.baseline_cs),
            "SA": (self.config.decay_rate_sa, self.config.baseline_sa)
        }

        for vector, (rate, baseline) in decay_config.items():
            current = getattr(self, vector)
            decay_factor = math.exp(-rate * seconds)
            new_value = baseline + (current - baseline) * decay_factor
            new_value = self._apply_saturation_effect(vector, new_value)
            setattr(self, vector, new_value)

        self.chemicals.apply_decay(seconds)

    def _apply_saturation_effect(self, vector: str, value: float) -> float:
        """应用饱和效应"""
        high = self.config.saturation_high
        low = self.config.saturation_low

        if value > high:
            excess = (value - high) / (1.0 - high)
            resistance = 1.0 - (excess * 0.5)
            value = high + (value - high) * resistance
        elif value < low:
            deficiency = (low - value) / low
            support = 1.0 - (deficiency * 0.3)
            value = low - (low - value) * support

        return max(0.0, min(1.0, value))

    def _calculate_context_impact(self, context: Dict) -> Dict[str, float]:
        """计算上下文影响"""
        impact = {"TR": 0.0, "CS": 0.0, "SA": 0.0}

        sentiment = context.get("sentiment", 0.0)
        intensity = context.get("intensity", 0.5)
        event_type = context.get("event_type", "neutral")

        # 应用事件影响
        event_impact = EVENT_IMPACTS.get(event_type, EVENT_IMPACTS["neutral"])
        for vector in ["TR", "CS", "SA"]:
            impact[vector] += event_impact.get(vector, 0.0)

        # 应用情感极性影响
        if sentiment > 0.3:
            impact["TR"] += 0.2 * sentiment
            impact["CS"] += 0.1 * sentiment
            impact["SA"] -= 0.15 * sentiment
        elif sentiment < -0.3:
            impact["SA"] += 0.3 * abs(sentiment)
            impact["CS"] -= 0.2 * abs(sentiment)

        # 应用强度调节
        intensity_factor = self._sigmoid_intensity(intensity)
        for vector in impact:
            impact[vector] *= intensity_factor * self.circadian_modulator

        return impact

    @staticmethod
    def _sigmoid_intensity(intensity: float) -> float:
        """S型强度曲线（边际递减效应）"""
        k = 5.0
        return 2.0 / (1.0 + math.exp(-k * (intensity - 0.5)))

    def _calculate_chemical_impact(self, context_impact: Dict) -> Dict[str, float]:
        """计算化学影响"""
        return {
            "dopamine": context_impact["TR"] * 0.6,
            "oxytocin": context_impact["CS"] * 0.7,
            "serotonin": context_impact["CS"] * 0.3,
            "cortisol": context_impact["SA"] * 0.8,
            "norepinephrine": context_impact["TR"] * 0.2 + context_impact["SA"] * 0.4
        }

    def _update_chemicals(self, chemical_impact: Dict, seconds: float) -> None:
        """更新化学状态"""
        # 应用直接影响
        for chem, impact in chemical_impact.items():
            current = getattr(self.chemicals, chem)
            new_value = max(0.0, min(1.0, current + impact))
            setattr(self.chemicals, chem, new_value)

        # 应用协同效应
        for (chem1, chem2), synergy in CHEMICAL_SYNERGY.items():
            val1 = getattr(self.chemicals, chem1)
            synergy_effect = (val1 - 0.5) * synergy * 0.05
            current_val2 = getattr(self.chemicals, chem2)
            setattr(self.chemicals, chem2, max(0.0, min(1.0, current_val2 + synergy_effect)))

    def _apply_emotional_update(self, context_impact: Dict, chemical_impact: Dict,
                                seconds: float) -> Dict[str, float]:
        """应用情感更新"""
        raw_deltas = dict(context_impact)
        final_deltas = {}

        # 应用化学调节
        chem_modifiers = self._calculate_chemical_modifiers()
        for vector, modifier in chem_modifiers.items():
            raw_deltas[vector] *= (1.0 + modifier)

        # 应用惯性和动量
        for vector in ["TR", "CS", "SA"]:
            inertia = self.inertia[vector]
            delta = (raw_deltas[vector] * (1.0 - inertia) +
                    self.vector_momentum[vector] * inertia)

            if abs(delta) < self.config.min_delta_threshold:
                delta = 0.0

            current = getattr(self, vector)
            saturation_adjusted = self._adjust_for_saturation(vector, current, delta)
            new_value = max(0.0, min(1.0, current + saturation_adjusted))
            setattr(self, vector, new_value)

            actual_delta = new_value - current
            final_deltas[vector] = actual_delta
            self.vector_momentum[vector] = actual_delta * 0.3 + self.vector_momentum[vector] * 0.7

        return final_deltas

    def _calculate_chemical_modifiers(self) -> Dict[str, float]:
        """计算化学调节因子"""
        return {
            "TR": (self.chemicals.dopamine * 0.6 +
                  self.chemicals.norepinephrine * 0.4 -
                  self.chemicals.cortisol * 0.2) - 0.4,
            "CS": (self.chemicals.oxytocin * 0.7 +
                  self.chemicals.serotonin * 0.3 -
                  self.chemicals.cortisol * 0.1) - 0.4,
            "SA": (self.chemicals.cortisol * 0.8 +
                  self.chemicals.norepinephrine * 0.2 -
                  self.chemicals.serotonin * 0.3) - 0.3
        }

    def _adjust_for_saturation(self, vector: str, current: float, delta: float) -> float:
        """根据饱和情况调整变化量"""
        high = self.config.saturation_high
        low = self.config.saturation_low

        if delta > 0 and current > high:
            excess = (current - high) / (1.0 - high)
            delta *= (1.0 - excess * 0.7)
        elif delta < 0 and current < low:
            deficiency = (low - current) / low
            delta *= (1.0 - deficiency * 0.5)

        return delta

    def _record_emotional_event(self, context: Dict, deltas: Dict) -> None:
        """检测并记录情感事件"""
        total_intensity = sum(abs(delta) for delta in deltas.values())

        if total_intensity < self.config.min_event_intensity:
            return

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

        self.emotional_events.append(event)
        if len(self.emotional_events) > self.config.max_events:
            self.emotional_events = self.emotional_events[-self.config.max_events:]

    def _update_emotional_state(self) -> None:
        """更新情感状态分类"""
        previous_state = self.current_state
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

    def _update_circadian_modulator(self) -> None:
        """更新昼夜节律调节因子"""
        hour = datetime.now().hour

        if 6 <= hour < 12:
            self.circadian_modulator = self.config.circadian_morning
        elif 12 <= hour < 18:
            self.circadian_modulator = self.config.circadian_afternoon
        elif 18 <= hour < 24:
            self.circadian_modulator = self.config.circadian_evening
        else:
            self.circadian_modulator = self.config.circadian_night

    def _record_state(self, event_type: str) -> None:
        """记录状态历史"""
        self.history.append({
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
        })

    def get_current_state(self) -> Dict[str, Any]:
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

    def get_history_summary(self, window_minutes: int = 60) -> Dict[str, Any]:
        """获取历史摘要"""
        cutoff_time = datetime.now() - timedelta(minutes=window_minutes)

        recent_history = [
            record for record in self.history
            if datetime.fromisoformat(record["timestamp"]) > cutoff_time
        ]

        if not recent_history:
            return {"error": "指定时间段内无历史记录"}

        tr_values = [r["TR"] for r in recent_history]
        cs_values = [r["CS"] for r in recent_history]
        sa_values = [r["SA"] for r in recent_history]

        def calc_stats(values: List[float]) -> Dict[str, float]:
            if HAS_NUMPY:
                return {
                    "mean": round(float(np.mean(values)), 3),
                    "std": round(float(np.std(values)), 3),
                    "min": round(min(values), 3),
                    "max": round(max(values), 3),
                    "trend": round(values[-1] - values[0], 3) if len(values) > 1 else 0
                }
            else:
                mean = sum(values) / len(values)
                variance = sum((x - mean) ** 2 for x in values) / len(values)
                return {
                    "mean": round(mean, 3),
                    "std": round(math.sqrt(variance), 3),
                    "min": round(min(values), 3),
                    "max": round(max(values), 3),
                    "trend": round(values[-1] - values[0], 3) if len(values) > 1 else 0
                }

        return {
            "time_window_minutes": window_minutes,
            "record_count": len(recent_history),
            "vector_stats": {
                "TR": calc_stats(tr_values),
                "CS": calc_stats(cs_values),
                "SA": calc_stats(sa_values)
            },
            "state_distribution": self._calculate_state_distribution(recent_history),
            "recent_events": [e.description for e in self.emotional_events[-5:]] if self.emotional_events else []
        }

    @staticmethod
    def _calculate_state_distribution(history: List[Dict]) -> Dict[str, float]:
        """计算状态分布"""
        if not history:
            return {}
            
        state_counts: Dict[str, int] = {}
        for record in history:
            state = record.get("emotional_state", "unknown")
            state_counts[state] = state_counts.get(state, 0) + 1

        total = len(history)
        return {state: round(count/total * 100, 1) for state, count in state_counts.items()}

    def get_detailed_analysis(self) -> Dict[str, Any]:
        """获取详细分析"""
        return {
            "timestamp": datetime.now().isoformat(),
            "vector_analysis": self._analyze_vectors(),
            "chemical_analysis": self._analyze_chemicals(),
            "health_metrics": self._calculate_health_metrics(),
            "recommendations": self._generate_recommendations()
        }

    def _analyze_vectors(self) -> Dict[str, Any]:
        """分析向量状态"""
        dominant = max(["TR", "CS", "SA"], key=lambda v: getattr(self, v))
        return {
            "dominant_vector": dominant,
            "vector_ratios": {
                "TR_CS_ratio": round(self.TR / self.CS, 2) if self.CS > 0 else float('inf'),
                "TR_SA_ratio": round(self.TR / self.SA, 2) if self.SA > 0 else float('inf'),
                "CS_SA_ratio": round(self.CS / self.SA, 2) if self.SA > 0 else float('inf')
            },
            "emotional_complexity": self._calculate_emotional_complexity()
        }

    def _calculate_emotional_complexity(self) -> float:
        """计算情感复杂度"""
        vector_diff = abs(self.TR - self.CS) + abs(self.CS - self.SA) + abs(self.SA - self.TR)
        return round(vector_diff / 2.0, 3)

    def _analyze_chemicals(self) -> Dict[str, Any]:
        """分析化学状态"""
        return {
            "balance_score": self._calculate_chemical_balance_score(),
            "dominant_chemical": max(self.chemicals.to_dict(), key=self.chemicals.to_dict().get),
            "imbalances": self._detect_chemical_imbalances()
        }

    def _calculate_chemical_balance_score(self) -> float:
        """计算化学平衡分数"""
        total_deviation = sum(
            abs(getattr(self.chemicals, chem) - ideal)
            for chem, ideal in CHEMICAL_IDEAL_VALUES.items()
        )
        return round(1.0 - (total_deviation / 5.0), 3)

    def _detect_chemical_imbalances(self) -> List[str]:
        """检测化学失衡"""
        imbalances = []
        chem_values = self.chemicals.to_dict()

        checks = [
            ("cortisol", 0.7, True, "皮质醇过高 - 长期压力"),
            ("dopamine", 0.3, False, "多巴胺不足 - 缺乏动力"),
            ("oxytocin", 0.4, False, "催产素不足 - 连接感弱"),
            ("serotonin", 0.4, False, "血清素不足 - 情绪不稳定"),
            ("norepinephrine", 0.6, True, "去甲肾上腺素过高 - 过度警觉")
        ]

        for chem, threshold, is_high, message in checks:
            value = chem_values[chem]
            if (is_high and value > threshold) or (not is_high and value < threshold):
                imbalances.append(message)

        return imbalances

    def _calculate_health_metrics(self) -> Dict[str, float]:
        """计算健康指标"""
        return {
            "stability_score": round(self.state_stability, 3),
            "adaptability_score": self._calculate_adaptability_score(),
            "resilience_score": self._calculate_resilience_score(),
            "emotional_range": self._calculate_emotional_range()
        }

    def _calculate_adaptability_score(self) -> float:
        """计算适应能力分数"""
        if len(self.history) < 10:
            return 0.5

        recent = list(self.history)[-10:]
        state_changes = sum(
            1 for i in range(1, len(recent))
            if recent[i]["emotional_state"] != recent[i-1]["emotional_state"]
        )

        ideal_changes = 3
        adaptability = 1.0 - abs(state_changes - ideal_changes) / ideal_changes
        return round(max(0.0, min(1.0, adaptability)), 3)

    def _calculate_resilience_score(self) -> float:
        """计算恢复能力分数"""
        if self.SA < 0.3:
            return 0.8
        elif self.SA < 0.5:
            return 0.6
        elif self.SA < 0.7:
            return 0.4
        return 0.2

    def _calculate_emotional_range(self) -> float:
        """计算情感范围"""
        if len(self.history) < 20:
            return 0.3

        recent = list(self.history)[-20:]
        ranges = [
            max(r[v] for r in recent) - min(r[v] for r in recent)
            for v in ["TR", "CS", "SA"]
        ]
        return round(sum(ranges) / 3.0, 3)

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

        if self.SA > 0.6:
            recommendations.append("压力水平较高，建议进行压力管理或寻求支持")
        elif self.SA < 0.2:
            recommendations.append("警觉度过低，可能需要增加一些挑战性任务")

        # 基于化学状态
        if self.chemicals.cortisol > 0.6:
            recommendations.append("检测到长期压力迹象，建议增加放松和恢复时间")
        if self.chemicals.oxytocin < 0.4:
            recommendations.append("连接感化学物质偏低，建议增加积极社交互动")

        return recommendations

    def export_state(self, filepath: str) -> None:
        """导出当前状态"""
        state_data = {
            "timestamp": datetime.now().isoformat(),
            "vectors": {"TR": self.TR, "CS": self.CS, "SA": self.SA},
            "chemicals": self.chemicals.to_dict(),
            "emotional_state": self.current_state.value,
            "history_length": len(self.history),
            "events_count": len(self.emotional_events)
        }

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(state_data, f, ensure_ascii=False, indent=2)

        print(f"[DesireEngine] 状态已导出到: {filepath}")

    def reset(self, new_tr: float = 0.6, new_cs: float = 0.7, new_sa: float = 0.3) -> None:
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
