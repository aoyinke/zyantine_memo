"""
协议引擎 - 协调所有协议的执行
"""
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime
from enum import Enum
import threading
from collections import defaultdict

from .fact_checker import FactChecker
from .length_regulator import LengthRegulator
from .expression_validator import ExpressionValidator


class ProtocolPriority(Enum):
    """协议优先级"""
    CRITICAL = 3
    HIGH = 2
    NORMAL = 1
    LOW = 0


class ProtocolConflictType(Enum):
    """协议冲突类型"""
    NONE = "none"
    LENGTH_VS_EXPRESSION = "length_vs_expression"
    FACT_VS_LENGTH = "fact_vs_length"
    FACT_VS_EXPRESSION = "fact_vs_expression"
    MULTIPLE = "multiple"


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

        # 协议优先级配置
        self.protocol_priorities = {
            "fact_check": ProtocolPriority.CRITICAL,
            "length_regulation": ProtocolPriority.NORMAL,
            "expression_validation": ProtocolPriority.HIGH
        }

        # 冲突解决策略
        self.conflict_resolution_strategies = {
            ProtocolConflictType.LENGTH_VS_EXPRESSION: "expression_first",
            ProtocolConflictType.FACT_VS_LENGTH: "fact_first",
            ProtocolConflictType.FACT_VS_EXPRESSION: "fact_first",
            ProtocolConflictType.MULTIPLE: "priority_based"
        }

        # 动态优先级调整
        self.protocol_performance: Dict[str, Dict] = defaultdict(lambda: {
            "success_count": 0,
            "failure_count": 0,
            "conflict_count": 0,
            "avg_processing_time": 0.0
        })

        # 冲突历史
        self.conflict_history: List[Dict] = []
        self.max_conflict_history = 100

        # 线程锁
        self.lock = threading.RLock()

        print("[协议引擎] 初始化完成，集成以下协议：")
        print("  - 事实检查协议 (优先级: CRITICAL)")
        print("  - 长度规整协议 (优先级: NORMAL)")
        print("  - 表达验证协议 (优先级: HIGH)")
        print("  - 协议冲突解决: 已启用")
        print("  - 动态优先级调整: 已启用")

    def apply_all_protocols(self, draft: str, context: Dict) -> Tuple[str, Dict]:
        """
        应用所有协议到草稿，包含冲突检测和解决

        Args:
            draft: 原始草稿
            context: 上下文信息，可能包含 skip_fact_check 标志

        Returns:
            (最终文本, 协议执行摘要)
        """
        # 初始化默认返回值，确保始终返回有效字典
        default_summary = {
            "timestamp": datetime.now().isoformat(),
            "original_draft_length": len(draft) if draft else 0,
            "final_text_length": len(draft) if draft else 0,
            "protocol_steps": {},
            "conflicts_detected": [],
            "conflicts_resolved": [],
            "total_reduction": 0,
            "total_processing_time": 0
        }
        
        start_time = datetime.now()
        protocol_summary = {
            "timestamp": datetime.now().isoformat(),
            "original_draft_length": len(draft) if draft else 0,
            "protocol_steps": {},
            "conflicts_detected": [],
            "conflicts_resolved": []
        }

        current_text = draft if draft else ""
        protocol_results = {}

        try:
            # 1. 事实检查（根据上下文决定是否跳过）
            skip_fact_check = context.get("skip_fact_check", False)
            
            if self.fact_checker and not skip_fact_check:
                try:
                    step_start = datetime.now()
                    is_verified, feedback = self.fact_checker.final_review(current_text, context)
                    step_time = (datetime.now() - step_start).total_seconds()
                    
                    # 确保feedback是字符串
                    feedback = str(feedback) if feedback else ""
                    
                    protocol_results["fact_check"] = {
                        "status": "passed" if is_verified else "failed",
                        "feedback": feedback,
                        "processing_time": step_time
                    }
                    
                    if not is_verified:
                        protocol_summary["protocol_steps"]["fact_check"] = {
                            "status": "failed",
                            "feedback": feedback,
                            "action": "需要重新生成或修改草稿"
                        }
                    else:
                        protocol_summary["protocol_steps"]["fact_check"] = {
                            "status": "passed",
                            "feedback": feedback
                        }
                except Exception as e:
                    # 事实检查失败，记录但不中断流程
                    error_msg = f"事实检查失败: {str(e)}"
                    print(f"[协议引擎] {error_msg}")
                    protocol_results["fact_check"] = {
                        "status": "error",
                        "feedback": error_msg,
                        "processing_time": 0
                    }
                    protocol_summary["protocol_steps"]["fact_check"] = {
                        "status": "error",
                        "feedback": error_msg
                    }
            elif skip_fact_check:
                # 跳过事实检查
                protocol_summary["protocol_steps"]["fact_check"] = {
                    "status": "skipped",
                    "feedback": "跳过事实检查（降级回复或标记跳过）"
                }

            # 2. 长度规整（需要认知快照）
            if self.length_regulator and context.get("cognitive_snapshot"):
                try:
                    step_start = datetime.now()
                    original_length = len(current_text)
                    regulated_text = self.length_regulator.regulate(current_text, context["cognitive_snapshot"])
                    # 确保返回的是字符串
                    current_text = regulated_text if regulated_text else current_text
                    step_time = (datetime.now() - step_start).total_seconds()
                    
                    protocol_results["length_regulation"] = {
                        "status": "applied",
                        "original_length": original_length,
                        "new_length": len(current_text),
                        "reduction": original_length - len(current_text),
                        "processing_time": step_time
                    }
                    
                    protocol_summary["protocol_steps"]["length_regulation"] = {
                        "status": "applied",
                        "new_length": len(current_text),
                        "reduction": protocol_summary["original_draft_length"] - len(current_text)
                    }
                except Exception as e:
                    error_msg = f"长度规整失败: {str(e)}"
                    print(f"[协议引擎] {error_msg}")
                    protocol_results["length_regulation"] = {
                        "status": "error",
                        "feedback": error_msg,
                        "processing_time": 0
                    }

            # 3. 表达验证
            if self.expression_validator:
                try:
                    step_start = datetime.now()
                    final_text, violations = self.expression_validator.validate_and_transform(current_text)
                    step_time = (datetime.now() - step_start).total_seconds()
                    
                    # 确保返回值有效
                    final_text = final_text if final_text else current_text
                    violations = violations if isinstance(violations, list) else []
                    
                    protocol_results["expression_validation"] = {
                        "status": "passed" if not violations else "violations_found",
                        "violations_count": len(violations),
                        "violations": violations[:3] if violations else [],
                        "processing_time": step_time
                    }
                    
                    protocol_summary["protocol_steps"]["expression_validation"] = {
                        "status": "passed" if not violations else "violations_found",
                        "violations_count": len(violations),
                        "violations": violations[:3] if violations else []
                    }
                    current_text = final_text
                except Exception as e:
                    error_msg = f"表达验证失败: {str(e)}"
                    print(f"[协议引擎] {error_msg}")
                    protocol_results["expression_validation"] = {
                        "status": "error",
                        "feedback": error_msg,
                        "processing_time": 0
                    }

            # 检测冲突
            try:
                conflicts = self._detect_conflicts(protocol_results)
                if conflicts and isinstance(conflicts, list):
                    protocol_summary["conflicts_detected"] = conflicts
                    
                    # 解决冲突
                    resolved_text, resolution_summary = self._resolve_conflicts(
                        conflicts, current_text, context, protocol_results
                    )
                    current_text = resolved_text if resolved_text else current_text
                    protocol_summary["conflicts_resolved"] = resolution_summary if isinstance(resolution_summary, list) else []
            except Exception as e:
                error_msg = f"冲突检测/解决失败: {str(e)}"
                print(f"[协议引擎] {error_msg}")
                protocol_summary["conflicts_detected"] = []

            # 更新性能统计
            total_time = (datetime.now() - start_time).total_seconds()
            try:
                self._update_protocol_performance(protocol_results, total_time, conflicts if 'conflicts' in locals() else [])
            except Exception:
                pass  # 性能统计失败不影响主流程

            # 更新摘要
            protocol_summary["final_text_length"] = len(current_text) if current_text else 0
            protocol_summary["total_reduction"] = protocol_summary["original_draft_length"] - protocol_summary["final_text_length"]
            protocol_summary["total_processing_time"] = total_time

            # 确保所有必需字段都存在
            if "protocol_steps" not in protocol_summary:
                protocol_summary["protocol_steps"] = {}
            if "conflicts_detected" not in protocol_summary:
                protocol_summary["conflicts_detected"] = []
            if "conflicts_resolved" not in protocol_summary:
                protocol_summary["conflicts_resolved"] = []

            # 记录协议执行
            try:
                self._log_protocol_execution(protocol_summary)
            except Exception:
                pass  # 日志记录失败不影响主流程

            return current_text, protocol_summary

        except Exception as e:
            # 发生严重错误，返回默认值
            error_msg = f"协议执行失败: {str(e)}"
            print(f"[协议引擎] {error_msg}")
            
            # 返回原始草稿和错误摘要
            default_summary["protocol_steps"]["error"] = {
                "status": "failed",
                "feedback": error_msg
            }
            default_summary["total_processing_time"] = (datetime.now() - start_time).total_seconds()
            
            return draft if draft else "", default_summary

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

    def _detect_conflicts(self, protocol_results: Dict) -> List[Dict]:
        """检测协议冲突"""
        conflicts = []

        fact_check = protocol_results.get("fact_check", {})
        length_reg = protocol_results.get("length_regulation", {})
        expression = protocol_results.get("expression_validation", {})

        # 检测长度规整与表达验证的冲突
        if length_reg.get("status") == "applied" and expression.get("violations_count", 0) > 0:
            conflicts.append({
                "type": ProtocolConflictType.LENGTH_VS_EXPRESSION,
                "protocols": ["length_regulation", "expression_validation"],
                "description": "长度规整可能导致表达违规",
                "severity": "medium"
            })

        # 检测事实检查与长度规整的冲突
        if fact_check.get("status") == "failed" and length_reg.get("status") == "applied":
            conflicts.append({
                "type": ProtocolConflictType.FACT_VS_LENGTH,
                "protocols": ["fact_check", "length_regulation"],
                "description": "事实检查失败，长度规整可能影响内容完整性",
                "severity": "high"
            })

        # 检测事实检查与表达验证的冲突
        if fact_check.get("status") == "failed" and expression.get("violations_count", 0) > 0:
            conflicts.append({
                "type": ProtocolConflictType.FACT_VS_EXPRESSION,
                "protocols": ["fact_check", "expression_validation"],
                "description": "事实检查失败且存在表达违规",
                "severity": "high"
            })

        # 检测多重冲突
        if len(conflicts) > 1:
            conflicts.append({
                "type": ProtocolConflictType.MULTIPLE,
                "protocols": [c["protocols"] for c in conflicts],
                "description": "检测到多个协议冲突",
                "severity": "high"
            })

        return conflicts

    def _resolve_conflicts(self, conflicts: List[Dict], current_text: str,
                           context: Dict, protocol_results: Dict) -> Tuple[str, List[Dict]]:
        """解决协议冲突"""
        resolution_summary = []
        resolved_text = current_text

        for conflict in conflicts:
            conflict_type = conflict["type"]
            strategy = self.conflict_resolution_strategies.get(conflict_type, "priority_based")

            if conflict_type == ProtocolConflictType.LENGTH_VS_EXPRESSION:
                if strategy == "expression_first":
                    # 优先保证表达正确，可能需要重新调整长度
                    if self.expression_validator:
                        resolved_text, violations = self.expression_validator.validate_and_transform(resolved_text)
                        resolution_summary.append({
                            "conflict": conflict_type.value,
                            "strategy": strategy,
                            "action": "优先保证表达正确性",
                            "result": "已重新验证表达"
                        })
                else:
                    resolution_summary.append({
                        "conflict": conflict_type.value,
                        "strategy": strategy,
                        "action": "保持长度规整结果",
                        "result": "接受可能的表达违规"
                    })

            elif conflict_type in [ProtocolConflictType.FACT_VS_LENGTH, ProtocolConflictType.FACT_VS_EXPRESSION]:
                if strategy == "fact_first":
                    # 事实检查失败，需要重新生成
                    resolution_summary.append({
                        "conflict": conflict_type.value,
                        "strategy": strategy,
                        "action": "事实检查优先",
                        "result": "需要重新生成内容"
                    })
                    # 这里可以触发重新生成逻辑
                else:
                    resolution_summary.append({
                        "conflict": conflict_type.value,
                        "strategy": strategy,
                        "action": "尝试其他策略",
                        "result": "使用替代方案"
                    })

            elif conflict_type == ProtocolConflictType.MULTIPLE:
                if strategy == "priority_based":
                    # 基于优先级解决
                    priority_order = sorted(
                        conflict["protocols"],
                        key=lambda p: self.protocol_priorities.get(p[0], ProtocolPriority.NORMAL).value,
                        reverse=True
                    )
                    resolution_summary.append({
                        "conflict": conflict_type.value,
                        "strategy": strategy,
                        "action": f"按优先级顺序: {priority_order}",
                        "result": "已按优先级解决"
                    })

        # 记录冲突历史
        self._record_conflict(conflicts, resolution_summary)

        return resolved_text, resolution_summary

    def _record_conflict(self, conflicts: List[Dict], resolutions: List[Dict]):
        """记录冲突历史"""
        with self.lock:
            record = {
                "timestamp": datetime.now().isoformat(),
                "conflicts": conflicts,
                "resolutions": resolutions
            }
            self.conflict_history.append(record)
            if len(self.conflict_history) > self.max_conflict_history:
                self.conflict_history = self.conflict_history[-self.max_conflict_history:]

    def _update_protocol_performance(self, protocol_results: Dict, total_time: float, conflicts: List[Dict]):
        """更新协议性能统计"""
        with self.lock:
            for protocol_name, result in protocol_results.items():
                perf = self.protocol_performance[protocol_name]
                if result.get("status") in ["passed", "applied"]:
                    perf["success_count"] += 1
                else:
                    perf["failure_count"] += 1
                
                processing_time = result.get("processing_time", 0.0)
                if perf["avg_processing_time"] == 0.0:
                    perf["avg_processing_time"] = processing_time
                else:
                    perf["avg_processing_time"] = (perf["avg_processing_time"] * 0.9 + processing_time * 0.1)

            # 更新冲突计数
            for conflict in conflicts:
                for protocol in conflict.get("protocols", []):
                    if isinstance(protocol, list):
                        for p in protocol:
                            self.protocol_performance[p]["conflict_count"] += 1
                    else:
                        self.protocol_performance[protocol]["conflict_count"] += 1

    def adjust_protocol_priority(self, protocol_name: str, new_priority: ProtocolPriority):
        """动态调整协议优先级"""
        if protocol_name in self.protocol_priorities:
            old_priority = self.protocol_priorities[protocol_name]
            self.protocol_priorities[protocol_name] = new_priority
            print(f"[协议引擎] 协议 '{protocol_name}' 优先级从 {old_priority.name} 调整为 {new_priority.name}")

    def get_conflict_statistics(self) -> Dict:
        """获取冲突统计信息"""
        if not self.conflict_history:
            return {"status": "no_data", "message": "暂无冲突记录"}

        total_conflicts = len(self.conflict_history)
        conflict_type_counts = defaultdict(int)

        for record in self.conflict_history:
            for conflict in record.get("conflicts", []):
                conflict_type_counts[conflict["type"].value] += 1

        return {
            "total_conflict_incidents": total_conflicts,
            "conflict_type_distribution": dict(conflict_type_counts),
            "most_common_conflict": max(conflict_type_counts.items(), key=lambda x: x[1]) if conflict_type_counts else None,
            "recent_conflicts": self.conflict_history[-5:] if total_conflicts >= 5 else self.conflict_history
        }

    def get_protocol_performance(self) -> Dict:
        """获取协议性能统计"""
        performance = {}
        for protocol_name, stats in self.protocol_performance.items():
            total_executions = stats["success_count"] + stats["failure_count"]
            performance[protocol_name] = {
                "total_executions": total_executions,
                "success_rate": stats["success_count"] / total_executions if total_executions > 0 else 0,
                "failure_rate": stats["failure_count"] / total_executions if total_executions > 0 else 0,
                "conflict_count": stats["conflict_count"],
                "avg_processing_time": stats["avg_processing_time"],
                "current_priority": self.protocol_priorities.get(protocol_name, ProtocolPriority.NORMAL).name
            }
        return performance

    def optimize_protocol_priorities(self):
        """基于性能统计优化协议优先级"""
        performance = self.get_protocol_performance()

        for protocol_name, perf in performance.items():
            if perf["total_executions"] < 10:
                continue

            # 如果失败率高，提高优先级
            if perf["failure_rate"] > 0.3:
                current_priority = self.protocol_priorities.get(protocol_name, ProtocolPriority.NORMAL)
                if current_priority != ProtocolPriority.CRITICAL:
                    new_priority = ProtocolPriority.HIGH if current_priority == ProtocolPriority.NORMAL else ProtocolPriority.CRITICAL
                    self.adjust_protocol_priority(protocol_name, new_priority)

            # 如果冲突频繁，提高优先级
            if perf["conflict_count"] > perf["total_executions"] * 0.2:
                current_priority = self.protocol_priorities.get(protocol_name, ProtocolPriority.NORMAL)
                if current_priority == ProtocolPriority.LOW:
                    self.adjust_protocol_priority(protocol_name, ProtocolPriority.NORMAL)

        print("[协议引擎] 优先级优化完成")