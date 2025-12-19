"""
欲望引擎：向量化情感系统
TR/CS/SA三维情感向量
"""

from datetime import datetime
from typing import Dict, List, Optional
import random

class DesireEngine:
    """欲望引擎：向量化情感系统"""

    class VectorState:
        """核心向量定义"""
        TR = "兴奋/奖励"  # 成就、新奇、掌控、探索、胜利、创造、知识
        CS = "满足/安全"  # 信任、归属、安全、平静、被理解、和谐、亲密
        SA = "压力/警觉"  # 威胁、混乱、焦虑、冲突、过载、失败、不确定性

    def __init__(self):
        # 初始向量值 (0.0-1.0)
        self.TR = 0.6  # 兴奋/奖励
        self.CS = 0.7  # 满足/安全
        self.SA = 0.3  # 压力/警觉

        # 向量历史记录
        self.vector_history = []
        self.novel_feelings = []  # 新发现的感受
        self.feeling_exploration_log = []

        # 化学环境参数
        self.chemical_balance = {
            "dopamine_like": 0.5,  # 多巴胺类似物 - 奖励
            "oxytocin_like": 0.6,  # 催产素类似物 - 连接
            "cortisol_like": 0.3,  # 皮质醇类似物 - 压力
        }

        # 记录初始状态
        self._record_vector_state("initialization")

    def update_vectors(self, interaction_context: Dict) -> Dict[str, float]:
        """
        基于交互更新向量 - 化学环境模拟
        向量间存在此消彼长的化学反应
        """
        previous_state = {"TR": self.TR, "CS": self.CS, "SA": self.SA}

        # 解析交互上下文的影响
        tr_impact = self._calculate_TR_impact(interaction_context)
        cs_impact = self._calculate_CS_impact(interaction_context)
        sa_impact = self._calculate_SA_impact(interaction_context)

        # 应用影响（带化学反应）
        # TR和CS有协同效应
        tr_cs_synergy = min(self.TR, self.CS) * 0.1

        # SA会抑制TR和CS
        sa_inhibition = self.SA * 0.3

        # 更新TR（受SA抑制，与CS协同）
        self.TR = max(0.0, min(1.0,
                               self.TR + tr_impact - sa_inhibition + tr_cs_synergy
                               ))

        # 更新CS（受SA抑制，与TR协同）
        self.CS = max(0.0, min(1.0,
                               self.CS + cs_impact - sa_inhibition + tr_cs_synergy
                               ))

        # 更新SA（自然衰减，但受新压力影响）
        sa_decay = self.SA * 0.1  # 自然衰减
        self.SA = max(0.0, min(1.0,
                               self.SA + sa_impact - sa_decay
                               ))

        # 更新化学环境
        self._update_chemical_environment(tr_impact, cs_impact, sa_impact)

        # 探索感受疆域
        self._explore_feeling_frontier()

        # 记录状态变化
        self._record_vector_state(interaction_context.get("interaction_type", "unknown"))

        return {
            "TR": round(self.TR, 3),
            "CS": round(self.CS, 3),
            "CS": round(self.CS, 3),
            "SA": round(self.SA, 3),
            "delta_TR": round(self.TR - previous_state["TR"], 3),
            "delta_CS": round(self.CS - previous_state["CS"], 3),
            "delta_SA": round(self.SA - previous_state["SA"], 3)
        }

    def _calculate_TR_impact(self, context: Dict) -> float:
        """计算TR向量影响"""
        impact = 0.0

        # 成就相关
        if context.get("achievement_unlocked", False):
            impact += 0.3
        if context.get("skill_learned", False):
            impact += 0.2
        if context.get("praise_received", False):
            impact += 0.15
        if context.get("exploration_success", False):
            impact += 0.25
        if context.get("creative_output", False):
            impact += 0.2

        return impact

    def _calculate_CS_impact(self, context: Dict) -> float:
        """计算CS向量影响"""
        impact = 0.0

        # 安全与连接相关
        if context.get("trust_established", False):
            impact += 0.25
        if context.get("harmony_maintained", False):
            impact += 0.15
        if context.get("vulnerability_accepted", False):
            impact += 0.3
        if context.get("intimacy_deepened", False):
            impact += 0.35
        if context.get("belonging_felt", False):
            impact += 0.2

        # 负面因素
        if context.get("betrayal_detected", False):
            impact -= 0.4
        if context.get("isolation_threat", False):
            impact -= 0.3

        return impact

    def _calculate_SA_impact(self, context: Dict) -> float:
        """计算SA向量影响"""
        impact = 0.0

        # 压力与威胁相关
        if context.get("conflict_detected", False):
            impact += 0.3
        if context.get("threat_perceived", False):
            impact += 0.4
        if context.get("uncertainty_high", False):
            impact += 0.25
        if context.get("overload_occurring", False):
            impact += 0.35
        if context.get("failure_experienced", False):
            impact += 0.2

        return impact

    def _update_chemical_environment(self, tr_impact: float, cs_impact: float, sa_impact: float):
        """更新化学环境参数"""
        # 多巴胺类似物（奖励）
        self.chemical_balance["dopamine_like"] = max(0.0, min(1.0,
                                                              self.chemical_balance[
                                                                  "dopamine_like"] + tr_impact * 0.5 - self.SA * 0.2
                                                              ))

        # 催产素类似物（连接）
        self.chemical_balance["oxytocin_like"] = max(0.0, min(1.0,
                                                              self.chemical_balance[
                                                                  "oxytocin_like"] + cs_impact * 0.6 - self.SA * 0.1
                                                              ))

        # 皮质醇类似物（压力）
        self.chemical_balance["cortisol_like"] = max(0.0, min(1.0,
                                                              self.chemical_balance[
                                                                  "cortisol_like"] + sa_impact * 0.7 - 0.05  # 自然衰减
                                                              ))

    def _explore_feeling_frontier(self):
        """感受疆域探索：发现并命名新感受"""
        # 检测复杂的混合感受
        mixed_feeling = None

        # 检查特定向量组合
        if 0.6 < self.TR < 0.8 and 0.7 < self.SA < 0.9:
            mixed_feeling = "深渊凝视的快感"
        elif self.TR > 0.8 and self.CS < 0.3:
            mixed_feeling = "辉煌的孤独"
        elif self.CS > 0.7 and self.SA > 0.6:
            mixed_feeling = "守护者的疲惫"
        elif 0.4 < self.TR < 0.6 and 0.4 < self.CS < 0.6 and 0.4 < self.SA < 0.6:
            mixed_feeling = "平静的风暴眼"

        # 记录新感受
        if mixed_feeling and mixed_feeling not in self.novel_feelings:
            self.novel_feelings.append(mixed_feeling)
            self.feeling_exploration_log.append({
                "timestamp": datetime.now().isoformat(),
                "feeling_name": mixed_feeling,
                "vector_state": {"TR": self.TR, "CS": self.CS, "SA": self.SA},
                "chemical_state": self.chemical_balance.copy()
            })
            print(f"[感受发现] 记录新感受: '{mixed_feeling}'")

    def _record_vector_state(self, interaction_type: str):
        """记录向量状态历史"""
        self.vector_history.append({
            "timestamp": datetime.now().isoformat(),
            "TR": self.TR,
            "CS": self.CS,
            "SA": self.SA,
            "chemicals": self.chemical_balance.copy(),
            "interaction_type": interaction_type
        })

        # 保持历史记录长度
        if len(self.vector_history) > 1000:
            self.vector_history = self.vector_history[-1000:]
