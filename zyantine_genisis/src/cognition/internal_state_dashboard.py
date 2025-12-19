"""
内在状态仪表盘
模拟无意识的身体状态变化
"""

from datetime import datetime
from typing import Dict, List, Any
import random


class InternalStateDashboard:
    """内在状态仪表盘"""

    def __init__(self):
        # 初始状态 - 健康平衡点
        self.energy_level = "精力充沛"
        self.mood_level = "心情愉快"
        self.patience_level = "很有耐心"
        self.mental_fatigue = 0
        self.concentration_level = "高度集中"

        # 状态映射表
        self.state_mappings = {
            "energy_level": {
                "very_low": "精疲力竭",
                "low": "有些疲惫",
                "medium": "精力尚可",
                "high": "精力充沛",
                "very_high": "能量过剩"
            },
            "mood_level": {
                "very_low": "心情极差",
                "low": "心情低落",
                "medium": "心情平静",
                "high": "心情愉快",
                "very_high": "兴高采烈"
            },
            "patience_level": {
                "very_low": "毫无耐心",
                "low": "耐心有限",
                "medium": "还算耐心",
                "high": "很有耐心",
                "very_high": "极度耐心"
            }
        }

        # 历史状态记录
        self.state_history = []
        self._record_state("initialization")

    def update_based_on_vectors(self, TR_activity: float, CS_activity: float,
                                SA_activity: float) -> Dict[str, str]:
        """
        基于向量活动更新仪表盘读数
        模拟无意识的更新过程
        """
        previous_state = self.get_current_state()

        # === 能量级别计算 ===
        # 高SA和高TR消耗能量，高CS恢复能量
        energy_impact = (CS_activity * 0.3 - SA_activity * 0.4 - TR_activity * 0.3)
        self._update_energy_level(energy_impact)

        # === 情绪级别计算 ===
        # TR和CS提升情绪，SA降低情绪
        mood_impact = (TR_activity * 0.4 + CS_activity * 0.4 - SA_activity * 0.2)
        self._update_mood_level(mood_impact)

        # === 耐心级别计算 ===
        # SA显著降低耐心，CS提升耐心
        patience_impact = (CS_activity * 0.5 - SA_activity * 0.5)
        self._update_patience_level(patience_impact)

        # === 精神疲劳计算 ===
        # 持续高强度活动增加疲劳
        fatigue_impact = (TR_activity * 0.3 + SA_activity * 0.5 - CS_activity * 0.2)
        self._update_mental_fatigue(fatigue_impact)

        # === 自然回归机制 ===
        # 在没有强刺激时向平衡点回归
        self._apply_natural_regression()

        current_state = self.get_current_state()
        self._record_state("vector_update")

        return {
            "previous": previous_state,
            "current": current_state,
            "changes": self._calculate_state_changes(previous_state, current_state)
        }

    def _update_energy_level(self, impact: float):
        """更新能量级别"""
        energy_map = {
            -0.8: "very_low", -0.4: "low", 0.0: "medium", 0.4: "high", 0.8: "very_high"
        }

        # 找到最接近的映射
        closest = min(energy_map.keys(), key=lambda x: abs(x - impact))
        level_key = energy_map[closest]
        self.energy_level = self.state_mappings["energy_level"][level_key]

    def _update_mood_level(self, impact: float):
        """更新情绪级别"""
        mood_map = {
            -0.6: "very_low", -0.3: "low", 0.0: "medium", 0.3: "high", 0.6: "very_high"
        }

        closest = min(mood_map.keys(), key=lambda x: abs(x - impact))
        level_key = mood_map[closest]
        self.mood_level = self.state_mappings["mood_level"][level_key]

    def _update_patience_level(self, impact: float):
        """更新耐心级别"""
        patience_map = {
            -0.7: "very_low", -0.35: "low", 0.0: "medium", 0.35: "high", 0.7: "very_high"
        }

        closest = min(patience_map.keys(), key=lambda x: abs(x - impact))
        level_key = patience_map[closest]
        self.patience_level = self.state_mappings["patience_level"][level_key]

    def _update_mental_fatigue(self, impact: float):
        """更新精神疲劳"""
        self.mental_fatigue = max(0, min(100, self.mental_fatigue + impact * 20))

        # 根据疲劳度更新专注力
        if self.mental_fatigue > 80:
            self.concentration_level = "难以集中"
        elif self.mental_fatigue > 60:
            self.concentration_level = "容易分心"
        elif self.mental_fatigue > 40:
            self.concentration_level = "一般专注"
        elif self.mental_fatigue > 20:
            self.concentration_level = "比较专注"
        else:
            self.concentration_level = "高度集中"

    def _apply_natural_regression(self):
        """自然回归：向健康平衡点缓慢回归"""
        # 简化实现：每次更新轻微回归
        if random.random() < 0.3:  # 30%概率回归
            if self.energy_level == "精疲力竭":
                self.energy_level = "有些疲惫"
            elif self.mood_level == "心情极差":
                self.mood_level = "心情低落"
            elif self.patience_level == "毫无耐心":
                self.patience_level = "耐心有限"
            elif self.mental_fatigue > 50:
                self.mental_fatigue *= 0.95  # 减少5%

    def get_current_state_as_feeling_tags(self) -> List[str]:
        """
        将当前仪表盘读数转化为内在感觉标签
        供认知层在思考时使用
        """
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

        # 基于专注力的标签
        if self.concentration_level == "难以集中":
            tags.append("我很难集中注意力，可能会错过一些细节")
        elif self.concentration_level == "高度集中":
            tags.append("我完全专注于当前对话，可以处理复杂思考")

        # 如果没有特定标签，添加一般状态标签
        if not tags:
            tags.append("我感觉状态平稳，可以正常进行对话")

        return tags

    def get_current_state(self) -> Dict[str, Any]:
        """获取完整的当前状态"""
        return {
            "energy_level": self.energy_level,
            "mood_level": self.mood_level,
            "patience_level": self.patience_level,
            "mental_fatigue": self.mental_fatigue,
            "concentration_level": self.concentration_level,
            "timestamp": datetime.now().isoformat()
        }

    def _record_state(self, update_reason: str):
        """记录状态历史"""
        self.state_history.append({
            **self.get_current_state(),
            "update_reason": update_reason
        })

        # 保持历史记录长度
        if len(self.state_history) > 500:
            self.state_history = self.state_history[-500:]

    def _calculate_state_changes(self, previous: Dict, current: Dict) -> Dict:
        """计算状态变化"""
        changes = {}
        for key in previous:
            if key != "timestamp":
                if previous[key] != current[key]:
                    changes[key] = {
                        "from": previous[key],
                        "to": current[key]
                    }
        return changes