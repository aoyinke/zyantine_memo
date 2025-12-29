"""
内在状态仪表盘
模拟无意识的身体状态变化
[优化版]：将瞬时映射改为数值累积，实现状态的连续性和惯性
"""

from datetime import datetime
from typing import Dict, List, Any
import random

class InternalStateDashboard:
    """内在状态仪表盘"""

    def __init__(self):
        # 内部数值 (0.0 - 100.0), 初始为健康平衡态
        self._energy_val = 80.0
        self._mood_val = 70.0
        self._patience_val = 80.0
        self._fatigue_val = 0.0

        # 映射显示的标签
        self.energy_level = "精力充沛"
        self.mood_level = "心情愉快"
        self.patience_level = "很有耐心"
        self.mental_fatigue = 0
        self.concentration_level = "高度集中"

        # 状态历史记录
        self.state_history = []
        self._update_labels() # 初始化标签
        self._record_state("initialization")

    def update_based_on_vectors(self, TR_activity: float, CS_activity: float,
                                SA_activity: float) -> Dict[str, Any]:
        """
        基于向量活动更新仪表盘读数
        模拟无意识的更新过程 (数值累积制)
        """
        previous_state = self.get_current_state()

        # === 计算影响因子 (Sensitivity) ===
        # 系数决定了状态变化的快慢

        # 1. 能量: CS(休息)恢复, SA(压力)和TR(兴奋)消耗
        energy_change = (CS_activity * 2.0) - (SA_activity * 1.5) - (TR_activity * 1.0)
        self._energy_val = self._clamp(self._energy_val + energy_change)

        # 2. 情绪: TR/CS提升, SA显著降低
        mood_change = (TR_activity * 1.5) + (CS_activity * 1.5) - (SA_activity * 3.0)
        self._mood_val = self._clamp(self._mood_val + mood_change)

        # 3. 耐心: CS提升, SA消耗, 能量低时消耗更快
        fatigue_penalty = 1.0 if self._fatigue_val < 60 else 2.0
        patience_change = (CS_activity * 1.5) - (SA_activity * 2.0 * fatigue_penalty)
        self._patience_val = self._clamp(self._patience_val + patience_change)

        # 4. 疲劳: TR/SA增加疲劳, CS减少
        fatigue_change = (TR_activity * 1.2) + (SA_activity * 1.5) - (CS_activity * 1.0)
        self._fatigue_val = self._clamp(self._fatigue_val + fatigue_change)

        # === 自然回归机制 (Homeostasis) ===
        self._apply_natural_regression()

        # === 更新显示标签 ===
        self._update_labels()

        current_state = self.get_current_state()
        self._record_state("vector_update")

        return {
            "previous": previous_state,
            "current": current_state,
            "changes": self._calculate_state_changes(previous_state, current_state)
        }

    def _clamp(self, value: float) -> float:
        """将数值限制在 0-100 之间"""
        return max(0.0, min(100.0, value))

    def _update_labels(self):
        """根据内部数值更新文本标签"""
        # 更新能量标签
        if self._energy_val > 90: self.energy_level = "能量过剩"
        elif self._energy_val > 70: self.energy_level = "精力充沛"
        elif self._energy_val > 40: self.energy_level = "精力尚可"
        elif self._energy_val > 20: self.energy_level = "有些疲惫"
        else: self.energy_level = "精疲力竭"

        # 更新情绪标签
        if self._mood_val > 85: self.mood_level = "兴高采烈"
        elif self._mood_val > 60: self.mood_level = "心情愉快"
        elif self._mood_val > 40: self.mood_level = "心情平静"
        elif self._mood_val > 20: self.mood_level = "心情低落"
        else: self.mood_level = "心情极差"

        # 更新耐心标签
        if self._patience_val > 85: self.patience_level = "极度耐心"
        elif self._patience_val > 60: self.patience_level = "很有耐心"
        elif self._patience_val > 40: self.patience_level = "还算耐心"
        elif self._patience_val > 20: self.patience_level = "耐心有限"
        else: self.patience_level = "毫无耐心"

        # 更新疲劳与专注
        self.mental_fatigue = self._fatigue_val
        if self._fatigue_val > 80: self.concentration_level = "难以集中"
        elif self._fatigue_val > 60: self.concentration_level = "容易分心"
        elif self._fatigue_val > 40: self.concentration_level = "一般专注"
        elif self._fatigue_val > 20: self.concentration_level = "比较专注"
        else: self.concentration_level = "高度集中"

    def _apply_natural_regression(self):
        """自然回归：数值向平衡点缓慢回归"""
        # 平衡点设置
        target_energy = 70.0
        target_mood = 60.0
        target_patience = 70.0
        target_fatigue = 20.0 # 休息状态下应有少量疲劳恢复

        rate = 0.05 # 回归速率

        self._energy_val += (target_energy - self._energy_val) * rate
        self._mood_val += (target_mood - self._mood_val) * rate
        self._patience_val += (target_patience - self._patience_val) * rate
        self._fatigue_val += (target_fatigue - self._fatigue_val) * rate

    def get_current_state_as_feeling_tags(self) -> List[str]:
        """将当前仪表盘读数转化为内在感觉标签"""
        tags = []

        # 基于能量级别的标签
        if self.energy_level in ["精疲力竭", "有些疲惫"]:
            tags.append("我感到极度疲惫，有想立刻结束对话的冲动")
        elif self.energy_level == "能量过剩":
            tags.append("我感觉精力过剩，渴望进行高强度的互动")

        # 基于情绪级别的标签
        if self.mood_level in ["兴高采烈", "心情愉快"]:
            tags.append("我心情极好，很想主动开玩笑或分享快乐")
        elif self.mood_level in ["心情极差", "心情低落"]:
            tags.append("我感觉情绪低落，需要一些积极的互动")

        # 基于耐心级别的标签
        if self.patience_level in ["毫无耐心", "耐心有限"]:
            tags.append("我的耐心正在消耗，希望对话更加直接高效")
        elif self.patience_level == "极度耐心":
            tags.append("我有充足的耐心，可以处理复杂或重复的话题")

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
            "patience_level": self.patience_level,
            "mental_fatigue": round(self._fatigue_val, 1),
            "concentration_level": self.concentration_level,
            "timestamp": datetime.now().isoformat()
        }

    def _record_state(self, update_reason: str):
        """记录状态历史"""
        self.state_history.append({
            **self.get_current_state(),
            "update_reason": update_reason
        })
        if len(self.state_history) > 500:
            self.state_history = self.state_history[-500:]

    def _calculate_state_changes(self, previous: Dict, current: Dict) -> Dict:
        changes = {}
        for key in ["energy_level", "mood_level", "patience_level", "concentration_level"]:
            if previous.get(key) != current.get(key):
                changes[key] = {"from": previous.get(key), "to": current.get(key)}
        return changes