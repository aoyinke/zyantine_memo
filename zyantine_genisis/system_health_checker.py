#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
系统健康检查器 - 自动化测试和健康监控
"""

from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from enum import Enum
import threading
from dataclasses import dataclass, field
from collections import defaultdict


class HealthStatus(Enum):
    """健康状态"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    CRITICAL = "critical"


class TestResult(Enum):
    """测试结果"""
    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"
    ERROR = "error"


@dataclass
class HealthCheck:
    """健康检查项"""
    name: str
    description: str
    module: str
    check_function: callable
    critical: bool = False
    timeout: int = 30
    last_result: Optional[TestResult] = None
    last_execution_time: Optional[datetime] = None
    last_duration_ms: float = 0.0
    error_message: Optional[str] = None
    failure_count: int = 0
    consecutive_failures: int = 0


@dataclass
class SystemHealthReport:
    """系统健康报告"""
    timestamp: datetime
    overall_status: HealthStatus
    health_score: float
    module_status: Dict[str, HealthStatus]
    checks: List[HealthCheck]
    active_alerts: List[Dict]
    recommendations: List[str]
    test_summary: Dict[str, int]


class SystemHealthChecker:
    """系统健康检查器"""

    def __init__(self):
        self.checks: List[HealthCheck] = []
        self.check_history: List[Dict] = []
        self.max_history_size = 1000
        self.alerts: List[Dict] = []
        self.max_alerts_size = 100

        # 健康检查配置
        self.check_interval = 300  # 5分钟
        self.auto_check_enabled = False
        self.check_thread = None
        self.stop_event = threading.Event()
        self.lock = threading.RLock()

        # 健康阈值
        self.health_thresholds = {
            "critical_failure_count": 3,
            "degraded_failure_count": 1,
            "health_score_critical": 50.0,
            "health_score_degraded": 75.0
        }

        print("[健康检查器] 系统健康检查器初始化完成")

    def register_check(self, name: str, description: str, module: str,
                     check_function: callable, critical: bool = False,
                     timeout: int = 30):
        """注册健康检查项"""
        check = HealthCheck(
            name=name,
            description=description,
            module=module,
            check_function=check_function,
            critical=critical,
            timeout=timeout
        )
        with self.lock:
            self.checks.append(check)
        print(f"[健康检查器] 注册检查项: {name} (模块: {module}, 关键: {critical})")

    def run_all_checks(self) -> SystemHealthReport:
        """运行所有健康检查"""
        print(f"[健康检查器] 开始运行 {len(self.checks)} 个健康检查项...")
        start_time = datetime.now()

        results = []
        test_summary = defaultdict(int)
        module_status = defaultdict(lambda: HealthStatus.HEALTHY)

        for check in self.checks:
            try:
                result, duration_ms, error = self._run_check(check)
                check.last_result = result
                check.last_execution_time = datetime.now()
                check.last_duration_ms = duration_ms
                check.error_message = error

                test_summary[result.value] += 1

                if result == TestResult.PASSED:
                    check.consecutive_failures = 0
                elif result in [TestResult.FAILED, TestResult.ERROR]:
                    check.failure_count += 1
                    check.consecutive_failures += 1

                results.append(check)

                # 更新模块状态
                if check.critical and result in [TestResult.FAILED, TestResult.ERROR]:
                    module_status[check.module] = HealthStatus.CRITICAL
                elif result in [TestResult.FAILED, TestResult.ERROR]:
                    if module_status[check.module] != HealthStatus.CRITICAL:
                        module_status[check.module] = HealthStatus.UNHEALTHY

            except Exception as e:
                check.last_result = TestResult.ERROR
                check.error_message = str(e)
                check.failure_count += 1
                check.consecutive_failures += 1
                test_summary[TestResult.ERROR.value] += 1
                results.append(check)
                module_status[check.module] = HealthStatus.CRITICAL

        # 计算整体健康状态
        overall_status, health_score = self._calculate_overall_status(results, module_status)

        # 生成建议
        recommendations = self._generate_recommendations(results, module_status)

        # 生成告警
        self._generate_alerts(results, module_status, overall_status)

        # 创建报告
        report = SystemHealthReport(
            timestamp=datetime.now(),
            overall_status=overall_status,
            health_score=health_score,
            module_status=dict(module_status),
            checks=results,
            active_alerts=[a for a in self.alerts if not a.get("resolved", False)],
            recommendations=recommendations,
            test_summary=dict(test_summary)
        )

        # 保存历史
        self._save_report_to_history(report)

        duration = (datetime.now() - start_time).total_seconds()
        print(f"[健康检查器] 健康检查完成，耗时: {duration:.2f}秒，状态: {overall_status.value}，分数: {health_score:.2f}")

        return report

    def _run_check(self, check: HealthCheck) -> Tuple[TestResult, float, Optional[str]]:
        """运行单个健康检查"""
        start_time = datetime.now()

        try:
            result = check.check_function()
            duration_ms = (datetime.now() - start_time).total_seconds() * 1000

            if result is True:
                return TestResult.PASSED, duration_ms, None
            elif result is False:
                return TestResult.FAILED, duration_ms, "检查失败"
            elif result is None:
                return TestResult.SKIPPED, duration_ms, None
            else:
                return TestResult.PASSED, duration_ms, None

        except Exception as e:
            duration_ms = (datetime.now() - start_time).total_seconds() * 1000
            return TestResult.ERROR, duration_ms, str(e)

    def _calculate_overall_status(self, checks: List[HealthCheck],
                                  module_status: Dict[str, HealthStatus]) -> Tuple[HealthStatus, float]:
        """计算整体健康状态和分数"""
        total_checks = len(checks)
        passed_checks = sum(1 for c in checks if c.last_result == TestResult.PASSED)
        failed_checks = sum(1 for c in checks if c.last_result in [TestResult.FAILED, TestResult.ERROR])
        critical_failures = sum(1 for c in checks if c.critical and c.last_result in [TestResult.FAILED, TestResult.ERROR])

        # 计算健康分数 (0-100)
        if total_checks > 0:
            pass_rate = passed_checks / total_checks
            health_score = pass_rate * 100
        else:
            health_score = 100.0

        # 确定整体状态
        if critical_failures > 0:
            return HealthStatus.CRITICAL, health_score * 0.5
        elif HealthStatus.CRITICAL in module_status.values():
            return HealthStatus.CRITICAL, health_score * 0.7
        elif HealthStatus.UNHEALTHY in module_status.values():
            return HealthStatus.UNHEALTHY, health_score * 0.8
        elif failed_checks >= self.health_thresholds["degraded_failure_count"]:
            return HealthStatus.DEGRADED, health_score * 0.9
        elif health_score < self.health_thresholds["health_score_degraded"]:
            return HealthStatus.DEGRADED, health_score
        else:
            return HealthStatus.HEALTHY, health_score

    def _generate_recommendations(self, checks: List[HealthCheck],
                                  module_status: Dict[str, HealthStatus]) -> List[str]:
        """生成修复建议"""
        recommendations = []

        # 基于失败检查的建议
        failed_checks = [c for c in checks if c.last_result in [TestResult.FAILED, TestResult.ERROR]]
        for check in failed_checks:
            if check.consecutive_failures >= 3:
                recommendations.append(
                    f"[{check.module}] {check.name} 连续失败 {check.consecutive_failures} 次，建议立即检查"
                )
            elif check.critical:
                recommendations.append(
                    f"[{check.module}] 关键检查 {check.name} 失败，需要立即处理"
                )

        # 基于模块状态的建议
        for module, status in module_status.items():
            if status == HealthStatus.CRITICAL:
                recommendations.append(f"[{module}] 模块处于严重状态，建议立即重启或修复")
            elif status == HealthStatus.UNHEALTHY:
                recommendations.append(f"[{module}] 模块不健康，建议检查日志和配置")

        # 基于健康分数的建议
        overall_status, health_score = self._calculate_overall_status(checks, module_status)
        if health_score < self.health_thresholds["health_score_critical"]:
            recommendations.append("系统整体健康分数过低，建议进行全面检查")
        elif health_score < self.health_thresholds["health_score_degraded"]:
            recommendations.append("系统性能有所下降，建议优化配置或增加资源")

        if not recommendations:
            recommendations.append("系统运行正常，建议继续监控")

        return recommendations

    def _generate_alerts(self, checks: List[HealthCheck],
                        module_status: Dict[str, HealthStatus],
                        overall_status: HealthStatus):
        """生成告警"""
        timestamp = datetime.now()

        # 关键检查失败告警
        for check in checks:
            if check.critical and check.last_result in [TestResult.FAILED, TestResult.ERROR]:
                if check.consecutive_failures >= self.health_thresholds["critical_failure_count"]:
                    alert = {
                        "timestamp": timestamp.isoformat(),
                        "type": "critical_check_failure",
                        "severity": "critical",
                        "module": check.module,
                        "check": check.name,
                        "message": f"关键检查 {check.name} 连续失败 {check.consecutive_failures} 次",
                        "resolved": False
                    }
                    self._add_alert(alert)

        # 模块状态告警
        for module, status in module_status.items():
            if status == HealthStatus.CRITICAL:
                alert = {
                    "timestamp": timestamp.isoformat(),
                    "type": "module_critical",
                    "severity": "critical",
                    "module": module,
                    "message": f"模块 {module} 处于严重状态",
                    "resolved": False
                }
                self._add_alert(alert)

        # 整体状态告警
        if overall_status == HealthStatus.CRITICAL:
            alert = {
                "timestamp": timestamp.isoformat(),
                "type": "system_critical",
                "severity": "critical",
                "module": "system",
                "message": "系统整体处于严重状态",
                "resolved": False
            }
            self._add_alert(alert)

    def _add_alert(self, alert: Dict):
        """添加告警"""
        self.alerts.append(alert)

        if len(self.alerts) > self.max_alerts_size:
            self.alerts = self.alerts[-self.max_alerts_size:]

        print(f"[健康检查器] 告警: {alert['message']}")

    def _save_report_to_history(self, report: SystemHealthReport):
        """保存报告到历史"""
        report_dict = {
            "timestamp": report.timestamp.isoformat(),
            "overall_status": report.overall_status.value,
            "health_score": report.health_score,
            "module_status": {k: v.value for k, v in report.module_status.items()},
            "test_summary": report.test_summary,
            "check_count": len(report.checks)
        }

        self.check_history.append(report_dict)

        if len(self.check_history) > self.max_history_size:
            self.check_history = self.check_history[-self.max_history_size:]

    def start_auto_check(self):
        """启动自动健康检查"""
        if self.auto_check_enabled:
            print("[健康检查器] 自动健康检查已在运行")
            return

        self.auto_check_enabled = True
        self.stop_event.clear()
        self.check_thread = threading.Thread(target=self._auto_check_loop, daemon=True)
        self.check_thread.start()
        print(f"[健康检查器] 自动健康检查已启动，间隔: {self.check_interval}秒")

    def stop_auto_check(self):
        """停止自动健康检查"""
        if not self.auto_check_enabled:
            return

        self.auto_check_enabled = False
        self.stop_event.set()
        if self.check_thread:
            self.check_thread.join(timeout=5)
        print("[健康检查器] 自动健康检查已停止")

    def _auto_check_loop(self):
        """自动检查循环"""
        while not self.stop_event.is_set():
            try:
                self.run_all_checks()
            except Exception as e:
                print(f"[健康检查器] 自动检查异常: {e}")

            self.stop_event.wait(self.check_interval)

    def get_health_summary(self) -> Dict:
        """获取健康摘要"""
        if not self.check_history:
            return {"status": "no_data", "message": "暂无数据"}

        latest = self.check_history[-1]
        return {
            "timestamp": latest.get("timestamp"),
            "overall_status": latest.get("overall_status"),
            "health_score": latest.get("health_score"),
            "module_status": latest.get("module_status"),
            "test_summary": latest.get("test_summary"),
            "check_count": latest.get("check_count"),
            "active_alerts": len([a for a in self.alerts if not a.get("resolved", False)]),
            "total_alerts": len(self.alerts),
            "data_points": len(self.check_history)
        }

    def get_check_details(self, check_name: Optional[str] = None) -> List[Dict]:
        """获取检查详情"""
        with self.lock:
            if check_name:
                checks = [c for c in self.checks if c.name == check_name]
            else:
                checks = self.checks

            return [
                {
                    "name": c.name,
                    "description": c.description,
                    "module": c.module,
                    "critical": c.critical,
                    "last_result": c.last_result.value if c.last_result else None,
                    "last_execution_time": c.last_execution_time.isoformat() if c.last_execution_time else None,
                    "last_duration_ms": c.last_duration_ms,
                    "error_message": c.error_message,
                    "failure_count": c.failure_count,
                    "consecutive_failures": c.consecutive_failures
                }
                for c in checks
            ]

    def resolve_alert(self, alert_index: int):
        """解决告警"""
        if 0 <= alert_index < len(self.alerts):
            self.alerts[alert_index]["resolved"] = True
            self.alerts[alert_index]["resolved_at"] = datetime.now().isoformat()
            print(f"[健康检查器] 告警已解决: {self.alerts[alert_index]['message']}")

    def clear_alerts(self):
        """清空告警"""
        self.alerts.clear()
        print("[健康检查器] 告警已清空")

    def clear_history(self):
        """清空历史"""
        self.check_history.clear()
        print("[健康检查器] 历史记录已清空")
