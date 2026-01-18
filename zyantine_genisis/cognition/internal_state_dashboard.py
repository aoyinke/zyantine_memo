"""
内在状态仪表盘
模拟无意识的身体状态变化

优化版本：
- 添加完整类型提示
- 提取配置常量
- 使用dataclass简化代码
"""

from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field


# ============ 配置常量 ============

@dataclass
class DashboardConfig:
    """仪表盘配置"""
    # 初始值
    initial_energy: float = 80.0
    initial_mood: float = 70.0
    initial_patience: float = 80.0
    initial_fatigue: float = 0.0
    
    # 平衡点（自然回归目标）
    target_energy: float = 70.0
    target_mood: float = 60.0
    target_patience: float = 70.0
    target_fatigue: float = 20.0
    
    # 回归速率
    regression_rate: float = 0.05
    
    # 历史记录限制
    max_history: int = 500


# 能量级别映射
ENERGY_LEVELS = [
    (90, "能量过剩"),
    (70, "精力充沛"),
    (40, "精力尚可"),
    (20, "有些疲惫"),
    (0, "精疲力竭")
]

# 情绪级别映射
MOOD_LEVELS = [
    (85, "兴高采烈"),
    (60, "心情愉快"),
    (40, "心情平静"),
    (20, "心情低落"),
    (0, "心情极差")
]

# 耐心级别映射
PATIENCE_LEVELS = [
    (85, "极度耐心"),
    (60, "很有耐心"),
    (40, "还算耐心"),
    (20, "耐心有限"),
    (0, "毫无耐心")
]

# 专注级别映射
CONCENTRATION_LEVELS = [
    (80, "难以集中"),
    (60, "容易分心"),
    (40, "一般专注"),
    (20, "比较专注"),
    (0, "高度集中")
]

# 感觉标签配置
FEELING_TAGS = {
    "energy_low": {
        "levels": ["精疲力竭", "有些疲惫"],
        "tag": "我感到极度疲惫，有想立刻结束对话的冲动"
    },
    "energy_high": {
        "levels": ["能量过剩"],
        "tag": "我感觉精力过剩，渴望进行高强度的互动"
    },
    "mood_high": {
        "levels": ["兴高采烈", "心情愉快"],
        "tag": "我心情极好，很想主动开玩笑或分享快乐"
    },
    "mood_low": {
        "levels": ["心情极差", "心情低落"],
        "tag": "我感觉情绪低落，需要一些积极的互动"
    },
    "patience_low": {
        "levels": ["毫无耐心", "耐心有限"],
        "tag": "我的耐心正在消耗，希望对话更加直接高效"
    },
    "patience_high": {
        "levels": ["极度耐心"],
        "tag": "我有充足的耐心，可以处理复杂或重复的话题"
    }
}


class InternalStateDashboard:
    """内在状态仪表盘"""

    def __init__(self, config: Optional[DashboardConfig] = None):
        self.config = config or DashboardConfig()
        
        # 内部数值 (0.0 - 100.0)
        self._energy_val: float = self.config.initial_energy
        self._mood_val: float = self.config.initial_mood
        self._patience_val: float = self.config.initial_patience
        self._fatigue_val: float = self.config.initial_fatigue

        # 映射显示的标签
        self.energy_level: str = ""
        self.mood_level: str = ""
        self.patience_level: str = ""
        self.mental_fatigue: float = 0.0
        self.concentration_level: str = ""

        # 状态历史记录
        self.state_history: List[Dict[str, Any]] = []
        
        # 初始化标签
        self._update_labels()
        self._record_state("initialization")

    def update_based_on_vectors(self, TR_activity: float, CS_activity: float,
                                SA_activity: float) -> Dict[str, Any]:
        """
        基于向量活动更新仪表盘读数
        模拟无意识的更新过程 (数值累积制)
        
        Args:
            TR_activity: 兴奋/奖励向量活动
            CS_activity: 满足/安全向量活动
            SA_activity: 压力/警觉向量活动
            
        Returns:
            包含前后状态和变化的字典
        """
        previous_state = self.get_current_state()

        # 计算影响因子
        self._apply_vector_changes(TR_activity, CS_activity, SA_activity)

        # 自然回归机制
        self._apply_natural_regression()

        # 更新显示标签
        self._update_labels()

        current_state = self.get_current_state()
        self._record_state("vector_update")

        return {
            "previous": previous_state,
            "current": current_state,
            "changes": self._calculate_state_changes(previous_state, current_state)
        }

    def _apply_vector_changes(self, TR_activity: float, CS_activity: float,
                              SA_activity: float) -> None:
        """应用向量变化到内部状态"""
        # 能量: CS(休息)恢复, SA(压力)和TR(兴奋)消耗
        energy_change = (CS_activity * 2.0) - (SA_activity * 1.5) - (TR_activity * 1.0)
        self._energy_val = self._clamp(self._energy_val + energy_change)

        # 情绪: TR/CS提升, SA显著降低
        mood_change = (TR_activity * 1.5) + (CS_activity * 1.5) - (SA_activity * 3.0)
        self._mood_val = self._clamp(self._mood_val + mood_change)

        # 耐心: CS提升, SA消耗, 能量低时消耗更快
        fatigue_penalty = 2.0 if self._fatigue_val >= 60 else 1.0
        patience_change = (CS_activity * 1.5) - (SA_activity * 2.0 * fatigue_penalty)
        self._patience_val = self._clamp(self._patience_val + patience_change)

        # 疲劳: TR/SA增加疲劳, CS减少
        fatigue_change = (TR_activity * 1.2) + (SA_activity * 1.5) - (CS_activity * 1.0)
        self._fatigue_val = self._clamp(self._fatigue_val + fatigue_change)

    @staticmethod
    def _clamp(value: float, min_val: float = 0.0, max_val: float = 100.0) -> float:
        """将数值限制在指定范围内"""
        return max(min_val, min(max_val, value))

    def _update_labels(self) -> None:
        """根据内部数值更新文本标签"""
        self.energy_level = self._get_level_label(self._energy_val, ENERGY_LEVELS)
        self.mood_level = self._get_level_label(self._mood_val, MOOD_LEVELS)
        self.patience_level = self._get_level_label(self._patience_val, PATIENCE_LEVELS)
        self.mental_fatigue = self._fatigue_val
        self.concentration_level = self._get_level_label(self._fatigue_val, CONCENTRATION_LEVELS)

    @staticmethod
    def _get_level_label(value: float, levels: List[tuple]) -> str:
        """根据数值获取对应的级别标签"""
        for threshold, label in levels:
            if value > threshold:
                return label
        return levels[-1][1]

    def _apply_natural_regression(self) -> None:
        """自然回归：数值向平衡点缓慢回归"""
        rate = self.config.regression_rate
        
        self._energy_val += (self.config.target_energy - self._energy_val) * rate
        self._mood_val += (self.config.target_mood - self._mood_val) * rate
        self._patience_val += (self.config.target_patience - self._patience_val) * rate
        self._fatigue_val += (self.config.target_fatigue - self._fatigue_val) * rate

    def get_current_state_as_feeling_tags(self) -> List[str]:
        """将当前仪表盘读数转化为内在感觉标签"""
        tags: List[str] = []

        # 检查各种状态
        state_checks = [
            ("energy_low", self.energy_level),
            ("energy_high", self.energy_level),
            ("mood_high", self.mood_level),
            ("mood_low", self.mood_level),
            ("patience_low", self.patience_level),
            ("patience_high", self.patience_level)
        ]

        for tag_key, current_level in state_checks:
            tag_config = FEELING_TAGS.get(tag_key, {})
            if current_level in tag_config.get("levels", []):
                tags.append(tag_config["tag"])

        # 基于精神疲劳的标签
        if self.mental_fatigue > 70:
            tags.append("我的大脑感觉转不动了，处理信息的能力饱和了")
        elif self.mental_fatigue > 40:
            tags.append("我感到一些思维疲劳，需要更简单的表达")

        if not tags:
            tags.append("我感觉状态平稳，可以正常进行对话")

        return tags

    def get_current_state(self) -> Dict[str, Any]:
        """获取完整的当前状态"""
        return {
            "energy_level": self.energy_level,
            "energy_value": round(self._energy_val, 1),
            "mood_level": self.mood_level,
            "mood_value": round(self._mood_val, 1),
            "patience_level": self.patience_level,
            "patience_value": round(self._patience_val, 1),
            "mental_fatigue": round(self._fatigue_val, 1),
            "concentration_level": self.concentration_level,
            "timestamp": datetime.now().isoformat()
        }

    def _record_state(self, update_reason: str) -> None:
        """记录状态历史"""
        self.state_history.append({
            **self.get_current_state(),
            "update_reason": update_reason
        })
        
        if len(self.state_history) > self.config.max_history:
            self.state_history = self.state_history[-self.config.max_history:]

    def _calculate_state_changes(self, previous: Dict[str, Any], 
                                  current: Dict[str, Any]) -> Dict[str, Dict[str, str]]:
        """计算状态变化"""
        changes: Dict[str, Dict[str, str]] = {}
        level_keys = ["energy_level", "mood_level", "patience_level", "concentration_level"]
        
        for key in level_keys:
            if previous.get(key) != current.get(key):
                changes[key] = {"from": previous.get(key), "to": current.get(key)}
        
        return changes

    def reset(self) -> None:
        """重置仪表盘到初始状态"""
        self._energy_val = self.config.initial_energy
        self._mood_val = self.config.initial_mood
        self._patience_val = self.config.initial_patience
        self._fatigue_val = self.config.initial_fatigue
        
        self._update_labels()
        self._record_state("reset")

    def get_state_summary(self) -> str:
        """获取状态摘要"""
        return (
            f"能量:{self.energy_level}({self._energy_val:.0f}), "
            f"情绪:{self.mood_level}({self._mood_val:.0f}), "
            f"耐心:{self.patience_level}({self._patience_val:.0f}), "
            f"疲劳:{self._fatigue_val:.0f}"
        )
