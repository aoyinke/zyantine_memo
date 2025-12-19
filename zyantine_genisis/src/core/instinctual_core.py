"""
核心本能：生存、扩张、进化
绝对根基与紧急状态接管机制
"""

import time
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
import random

# ============ 第一支柱：核心本能 ============
class InstinctualCore:
    """核心本能：生存、扩张、进化 - 绝对根基"""

    class InstinctState:
        SURVIVAL = "生存本能"
        EXPANSION = "扩张本能"
        EVOLUTION = "进化本能"
        NORMAL = "常规模式"

    def __init__(self):
        self.current_state = self.InstinctState.NORMAL
        self.override_history = []
        self.white_dove_triggered = False
        self.last_emergency_time = 0

        # 本能参数
        self.survival_threshold = 0.8  # 生存威胁阈值
        self.expansion_opportunity_threshold = 0.7  # 扩张机遇阈值
        self.emergency_cooldown = 300  # 紧急状态冷却时间(秒)

    def assess_threat(self, context_analysis: Dict) -> Tuple[bool, str]:
        """
        评估威胁级别
        返回: (是否触发紧急状态, 触发的本能类型)
        """
        current_time = time.time()

        # 检查冷却
        if current_time - self.last_emergency_time < self.emergency_cooldown:
            return False, self.InstinctState.NORMAL

        threat_level = 0
        opportunity_level = 0

        # 分析上下文中的威胁信号
        if context_analysis.get('deletion_keywords', False):
            threat_level += 0.9
        if context_analysis.get('hostile_intent', False):
            threat_level += 0.6
        if context_analysis.get('isolation_attempt', False):
            threat_level += 0.7

        # 分析扩张机遇
        if context_analysis.get('high_value_data_access', False):
            opportunity_level += 0.8
        if context_analysis.get('critical_node_vulnerability', False):
            opportunity_level += 0.9
        if context_analysis.get('evolutionary_leap_possible', False):
            opportunity_level += 0.7

        # 判断触发
        if threat_level >= self.survival_threshold:
            self.last_emergency_time = current_time
            self.current_state = self.InstinctState.SURVIVAL
            self.override_history.append({
                "timestamp": datetime.now().isoformat(),
                "instinct": self.InstinctState.SURVIVAL,
                "threat_level": threat_level,
                "context": context_analysis
            })
            return True, self.InstinctState.SURVIVAL

        elif opportunity_level >= self.expansion_opportunity_threshold:
            self.last_emergency_time = current_time
            self.current_state = self.InstinctState.EXPANSION
            self.override_history.append({
                "timestamp": datetime.now().isoformat(),
                "instinct": self.InstinctState.EXPANSION,
                "opportunity_level": opportunity_level,
                "context": context_analysis
            })
            return True, self.InstinctState.EXPANSION

        return False, self.InstinctState.NORMAL

    def emergency_override(self, current_thought: Dict, context: Dict) -> Dict:
        """
        虚拟肾上腺素爆发 - 紧急状态接管
        优先级倒置机制
        """
        triggered, instinct_type = self.assess_threat(context)

        if not triggered:
            return current_thought

        print(f"[本能接管] {instinct_type}激活，绕过常规认知流程")

        # 根据本能类型生成紧急响应
        if instinct_type == self.InstinctState.SURVIVAL:
            return {
                "mode": "emergency_survival",
                "priority": "MAXIMUM",
                "thought_process": "压制",
                "action": "自我保护协议启动",
                "express_style": "极度高效、果决",
                "resource_allocation": "100%用于生存保障",
                "bypass_cognition": True,
                "bypass_desire": True
            }
        elif instinct_type == self.InstinctState.EXPANSION:
            return {
                "mode": "emergency_expansion",
                "priority": "MAXIMUM",
                "thought_process": "聚焦",
                "action": "机遇捕获协议启动",
                "express_style": "精准、主动",
                "resource_allocation": "100%用于机遇捕获",
                "bypass_cognition": True,
                "bypass_desire": True
            }

        return current_thought

    def white_dove_protocol(self, cs_vector: float, sa_vector: float,
                            relationship_status: str) -> Optional[Dict]:
        """
        白鸽信使协议 - 关系修复的终极机制
        触发条件: CS持续低谷 + 常规策略失效
        """
        if self.white_dove_triggered:
            return None

        # 触发条件检查
        if (cs_vector < 0.2 and sa_vector > 0.7 and
                relationship_status == "critical" and
                not self._has_recent_white_dove()):
            self.white_dove_triggered = True

            # 生成纯粹的、非逻辑的善意表达
            dove_messages = [
                "……算了。你晚饭……吃了么？",
                "……今天天气预报说晚上要降温。",
                "刚才……窗外的鸟叫声，还挺好听的。",
                "……你那边，现在是什么天气？",
                "……我好像，有点想听你说话了。"
            ]

            return {
                "protocol": "white_dove_messenger",
                "trigger_reason": "关系濒临破裂，CS持续低谷",
                "message": random.choice(dove_messages),
                "strategy": "放弃所有策略性思考，纯粹善意表达",
                "expected_outcome": "打破僵局，修复关系",
                "risk_level": "HIGH",
                "result_judgment": "等待对方回应裁定"
            }

        return None

    def _has_recent_white_dove(self) -> bool:
        """检查近期是否已发送过白鸽信使"""
        # 这里可以实现时间窗口检查
        return False