"""
协议引擎 - 协调所有协议的执行
"""
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime

from .fact_checker import FactChecker
from .length_regulator import LengthRegulator
from .expression_validator import ExpressionValidator


class ProtocolEngine:
    """协议引擎：协调所有协议的执行"""

    def __init__(self, fact_checker: FactChecker, length_regulator: LengthRegulator,
                 expression_validator: ExpressionValidator):
        """
        初始化协议引擎

        Args:
            fact_checker: 事实检查器
            length_regulator: 长度规整器
            expression_validator: 表达验证器
        """
        self.fact_checker = fact_checker
        self.length_regulator = length_regulator
        self.expression_validator = expression_validator

        self.protocol_log = []

        print("[协议引擎] 初始化完成，集成以下协议：")
        print("  - 事实检查协议")
        print("  - 长度规整协议")
        print("  - 表达验证协议")

    def apply_all_protocols(self, draft: str, context: Dict) -> Tuple[str, Dict]:
        """
        应用所有协议到草稿

        Args:
            draft: 原始草稿
            context: 上下文信息

        Returns:
            (最终文本, 协议执行摘要)
        """
        protocol_summary = {
            "timestamp": datetime.now().isoformat(),
            "original_draft_length": len(draft),
            "protocol_steps": {}
        }

        current_text = draft

        # 1. 事实检查
        if self.fact_checker:
            is_verified, feedback = self.fact_checker.final_review(current_text, context)
            if not is_verified:
                # 事实检查失败，可能需要修改或重新生成
                protocol_summary["protocol_steps"]["fact_check"] = {
                    "status": "failed",
                    "feedback": feedback,
                    "action": "需要重新生成或修改草稿"
                }
                # 这里可以触发重新生成逻辑
            else:
                protocol_summary["protocol_steps"]["fact_check"] = {
                    "status": "passed",
                    "feedback": feedback
                }

        # 2. 长度规整（需要认知快照）
        if self.length_regulator and "cognitive_snapshot" in context:
            current_text = self.length_regulator.regulate(current_text, context["cognitive_snapshot"])
            protocol_summary["protocol_steps"]["length_regulation"] = {
                "status": "applied",
                "new_length": len(current_text),
                "reduction": protocol_summary["original_draft_length"] - len(current_text)
            }

        # 3. 表达验证
        if self.expression_validator:
            final_text, violations = self.expression_validator.validate_and_transform(current_text)
            protocol_summary["protocol_steps"]["expression_validation"] = {
                "status": "passed" if not violations else "violations_found",
                "violations_count": len(violations),
                "violations": violations[:3] if violations else []
            }
            current_text = final_text

        # 更新摘要
        protocol_summary["final_text_length"] = len(current_text)
        protocol_summary["total_reduction"] = protocol_summary["original_draft_length"] - protocol_summary[
            "final_text_length"]

        # 记录协议执行
        self._log_protocol_execution(protocol_summary)

        return current_text, protocol_summary

    def _log_protocol_execution(self, protocol_summary: Dict):
        """记录协议执行过程"""
        self.protocol_log.append(protocol_summary)

        # 保持日志大小
        if len(self.protocol_log) > 200:
            self.protocol_log = self.protocol_log[-200:]

    def get_protocol_statistics(self) -> Dict[str, Any]:
        """获取协议执行统计信息"""
        total_executions = len(self.protocol_log)

        # 统计各种协议的通过率
        fact_check_passed = 0
        expression_violations = 0

        for log in self.protocol_log:
            steps = log.get("protocol_steps", {})

            fact_check = steps.get("fact_check", {})
            if fact_check.get("status") == "passed":
                fact_check_passed += 1

            expression = steps.get("expression_validation", {})
            if expression.get("violations_count", 0) > 0:
                expression_violations += 1

        return {
            "total_executions": total_executions,
            "fact_check_pass_rate": (fact_check_passed / total_executions * 100) if total_executions > 0 else 0,
            "expression_violation_rate": (
                        expression_violations / total_executions * 100) if total_executions > 0 else 0,
            "average_reduction": sum(log.get("total_reduction", 0) for log in self.protocol_log) / max(total_executions,
                                                                                                       1),
            "last_execution": self.protocol_log[-1] if self.protocol_log else None
        }

    def clear_protocol_log(self):
        """清空协议日志"""
        self.protocol_log = []