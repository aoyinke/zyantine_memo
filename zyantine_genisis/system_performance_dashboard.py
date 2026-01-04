#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
系统性能监控仪表板
整合所有模块的性能数据，提供统一的监控界面
"""

from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from collections import defaultdict
import json


class SystemPerformanceDashboard:
    """系统性能监控仪表板"""

    def __init__(self):
        self.metrics_history: List[Dict] = []
        self.max_history_size = 1000
        self.alerts: List[Dict] = []
        self.max_alerts_size = 100

        # 性能阈值配置
        self.thresholds = {
            "memory_cache_hit_rate": 0.5,
            "memory_avg_latency_ms": 100.0,
            "cognitive_avg_processing_time": 2.0,
            "api_success_rate": 0.9,
            "api_avg_latency": 3.0,
            "protocol_conflict_rate": 0.2
        }

        print("[性能仪表板] 系统性能监控仪表板初始化完成")

    def collect_metrics(self, memory_manager=None, cognitive_flow_manager=None,
                       api_service_provider=None, protocol_engine=None) -> Dict:
        """收集所有模块的性能指标"""
        timestamp = datetime.now()
        metrics = {
            "timestamp": timestamp.isoformat(),
            "memory": self._collect_memory_metrics(memory_manager),
            "cognitive": self._collect_cognitive_metrics(cognitive_flow_manager),
            "api": self._collect_api_metrics(api_service_provider),
            "protocol": self._collect_protocol_metrics(protocol_engine)
        }

        # 计算整体健康分数
        metrics["overall_health_score"] = self._calculate_health_score(metrics)

        # 检查告警
        self._check_alerts(metrics)

        # 保存历史记录
        self._save_metrics_history(metrics)

        return metrics

    def _collect_memory_metrics(self, memory_manager) -> Dict:
        """收集记忆模块性能指标"""
        if not memory_manager:
            return {"status": "not_available"}

        try:
            stats = memory_manager.get_detailed_statistics()
            perf_stats = stats.get("performance", {})
            storage_stats = stats.get("storage", {})

            return {
                "status": "available",
                "total_memories": stats.get("total_memories", 0),
                "cache_hit_rate": perf_stats.get("cache_hit_rate", 0.0),
                "avg_latency_ms": perf_stats.get("avg_latency_ms", 0.0),
                "success_rate": perf_stats.get("success_rate", 0.0),
                "storage": {
                    "hot_items": storage_stats.get("hot_items", 0),
                    "warm_items": storage_stats.get("warm_items", 0),
                    "cold_items": storage_stats.get("cold_items", 0),
                    "total_size_bytes": storage_stats.get("total_size_bytes", 0)
                }
            }
        except Exception as e:
            return {"status": "error", "message": str(e)}

    def _collect_cognitive_metrics(self, cognitive_flow_manager) -> Dict:
        """收集认知流程性能指标"""
        if not cognitive_flow_manager:
            return {"status": "not_available"}

        try:
            perf_stats = cognitive_flow_manager.get_performance_stats()

            return {
                "status": "available",
                "total_thoughts_processed": perf_stats.get("total_thoughts_processed", 0),
                "avg_processing_time": perf_stats.get("avg_processing_time", 0.0),
                "cache_size": perf_stats.get("cache_size", 0),
                "decision_history_size": perf_stats.get("decision_history_size", 0),
                "cache_hit_rate": perf_stats.get("cache_hit_rate", 0.0),
                "step_timings": perf_stats.get("step_timings", {})
            }
        except Exception as e:
            return {"status": "error", "message": str(e)}

    def _collect_api_metrics(self, api_service_provider) -> Dict:
        """收集API服务性能指标"""
        if not api_service_provider:
            return {"status": "not_available"}

        try:
            service_metrics = api_service_provider.get_service_metrics()
            overall_status = api_service_provider.get_overall_status()
            cache_stats = api_service_provider.get_cache_stats()

            total_requests = sum(m.get("request_count", 0) for m in service_metrics.values())
            total_success = sum(m.get("success_count", 0) for m in service_metrics.values())

            return {
                "status": "available",
                "total_requests": total_requests,
                "total_success": total_success,
                "success_rate": total_success / max(total_requests, 1),
                "avg_latency": sum(m.get("avg_latency", 0) for m in service_metrics.values()) / max(len(service_metrics), 1),
                "active_service": overall_status.get("active_service", "none"),
                "cache": {
                    "size": cache_stats.get("cache_size", 0),
                    "hit_rate": cache_stats.get("hit_rate", 0.0)
                },
                "services": service_metrics
            }
        except Exception as e:
            return {"status": "error", "message": str(e)}

    def _collect_protocol_metrics(self, protocol_engine) -> Dict:
        """收集协议引擎性能指标"""
        if not protocol_engine:
            return {"status": "not_available"}

        try:
            protocol_stats = protocol_engine.get_protocol_statistics()
            conflict_stats = protocol_engine.get_conflict_statistics()
            performance = protocol_engine.get_protocol_performance()

            return {
                "status": "available",
                "total_executions": protocol_stats.get("total_executions", 0),
                "fact_check_pass_rate": protocol_stats.get("fact_check_pass_rate", 0.0),
                "expression_violation_rate": protocol_stats.get("expression_violation_rate", 0.0),
                "average_reduction": protocol_stats.get("average_reduction", 0.0),
                "conflict_incidents": conflict_stats.get("total_conflict_incidents", 0),
                "conflict_rate": conflict_stats.get("total_conflict_incidents", 0) / max(protocol_stats.get("total_executions", 1), 1),
                "protocol_performance": performance
            }
        except Exception as e:
            return {"status": "error", "message": str(e)}

    def _calculate_health_score(self, metrics: Dict) -> float:
        """计算整体健康分数（0-100）"""
        scores = []

        # 记忆模块健康分数
        memory = metrics.get("memory", {})
        if memory.get("status") == "available":
            cache_score = min(memory.get("cache_hit_rate", 0) / 0.8, 1.0)
            latency_score = 1.0 - min(memory.get("avg_latency_ms", 0) / 200.0, 1.0)
            scores.append((cache_score + latency_score) / 2 * 100)

        # 认知流程健康分数
        cognitive = metrics.get("cognitive", {})
        if cognitive.get("status") == "available":
            processing_score = 1.0 - min(cognitive.get("avg_processing_time", 0) / 5.0, 1.0)
            scores.append(processing_score * 100)

        # API服务健康分数
        api = metrics.get("api", {})
        if api.get("status") == "available":
            success_score = api.get("success_rate", 0)
            latency_score = 1.0 - min(api.get("avg_latency", 0) / 5.0, 1.0)
            scores.append((success_score + latency_score) / 2 * 100)

        # 协议引擎健康分数
        protocol = metrics.get("protocol", {})
        if protocol.get("status") == "available":
            pass_rate = protocol.get("fact_check_pass_rate", 0) / 100
            conflict_score = 1.0 - min(protocol.get("conflict_rate", 0) / 0.5, 1.0)
            scores.append((pass_rate + conflict_score) / 2 * 100)

        return sum(scores) / len(scores) if scores else 0.0

    def _check_alerts(self, metrics: Dict):
        """检查性能告警"""
        timestamp = datetime.now()

        # 检查记忆模块告警
        memory = metrics.get("memory", {})
        if memory.get("status") == "available":
            if memory.get("cache_hit_rate", 1.0) < self.thresholds["memory_cache_hit_rate"]:
                self._add_alert("memory", "cache_hit_rate_low",
                               f"记忆缓存命中率过低: {memory.get('cache_hit_rate', 0):.2%}",
                               timestamp)

            if memory.get("avg_latency_ms", 0) > self.thresholds["memory_avg_latency_ms"]:
                self._add_alert("memory", "high_latency",
                               f"记忆平均延迟过高: {memory.get('avg_latency_ms', 0):.2f} ms",
                               timestamp)

        # 检查认知流程告警
        cognitive = metrics.get("cognitive", {})
        if cognitive.get("status") == "available":
            if cognitive.get("avg_processing_time", 0) > self.thresholds["cognitive_avg_processing_time"]:
                self._add_alert("cognitive", "slow_processing",
                               f"认知处理时间过长: {cognitive.get('avg_processing_time', 0):.2f} 秒",
                               timestamp)

        # 检查API服务告警
        api = metrics.get("api", {})
        if api.get("status") == "available":
            if api.get("success_rate", 1.0) < self.thresholds["api_success_rate"]:
                self._add_alert("api", "low_success_rate",
                               f"API成功率过低: {api.get('success_rate', 0):.2%}",
                               timestamp)

            if api.get("avg_latency", 0) > self.thresholds["api_avg_latency"]:
                self._add_alert("api", "high_latency",
                               f"API平均延迟过高: {api.get('avg_latency', 0):.2f} 秒",
                               timestamp)

        # 检查协议引擎告警
        protocol = metrics.get("protocol", {})
        if protocol.get("status") == "available":
            if protocol.get("conflict_rate", 0) > self.thresholds["protocol_conflict_rate"]:
                self._add_alert("protocol", "high_conflict_rate",
                               f"协议冲突率过高: {protocol.get('conflict_rate', 0):.2%}",
                               timestamp)

    def _add_alert(self, module: str, alert_type: str, message: str, timestamp: datetime):
        """添加告警"""
        alert = {
            "timestamp": timestamp.isoformat(),
            "module": module,
            "type": alert_type,
            "message": message,
            "resolved": False
        }
        self.alerts.append(alert)

        # 保持告警列表大小
        if len(self.alerts) > self.max_alerts_size:
            self.alerts = self.alerts[-self.max_alerts_size:]

        print(f"[性能仪表板] 告警: [{module}] {message}")

    def _save_metrics_history(self, metrics: Dict):
        """保存指标历史记录"""
        self.metrics_history.append(metrics)

        # 保持历史记录大小
        if len(self.metrics_history) > self.max_history_size:
            self.metrics_history = self.metrics_history[-self.max_history_size:]

    def get_dashboard_summary(self) -> Dict:
        """获取仪表板摘要"""
        if not self.metrics_history:
            return {"status": "no_data", "message": "暂无数据"}

        latest = self.metrics_history[-1]
        previous = self.metrics_history[-2] if len(self.metrics_history) > 1 else latest

        return {
            "timestamp": latest.get("timestamp"),
            "overall_health_score": latest.get("overall_health_score", 0),
            "health_score_change": latest.get("overall_health_score", 0) - previous.get("overall_health_score", 0),
            "module_status": {
                "memory": latest.get("memory", {}).get("status"),
                "cognitive": latest.get("cognitive", {}).get("status"),
                "api": latest.get("api", {}).get("status"),
                "protocol": latest.get("protocol", {}).get("status")
            },
            "active_alerts": len([a for a in self.alerts if not a.get("resolved", False)]),
            "total_alerts": len(self.alerts),
            "data_points": len(self.metrics_history)
        }

    def get_detailed_report(self) -> Dict:
        """获取详细报告"""
        if not self.metrics_history:
            return {"status": "no_data", "message": "暂无数据"}

        latest = self.metrics_history[-1]

        return {
            "timestamp": latest.get("timestamp"),
            "overall_health_score": latest.get("overall_health_score", 0),
            "modules": {
                "memory": latest.get("memory", {}),
                "cognitive": latest.get("cognitive", {}),
                "api": latest.get("api", {}),
                "protocol": latest.get("protocol", {})
            },
            "alerts": {
                "active": [a for a in self.alerts if not a.get("resolved", False)],
                "recent": self.alerts[-10:] if len(self.alerts) >= 10 else self.alerts
            },
            "trends": self._calculate_trends()
        }

    def _calculate_trends(self) -> Dict:
        """计算性能趋势"""
        if len(self.metrics_history) < 2:
            return {"status": "insufficient_data"}

        recent_metrics = self.metrics_history[-10:] if len(self.metrics_history) >= 10 else self.metrics_history

        trends = {}

        for module in ["memory", "cognitive", "api", "protocol"]:
            module_data = [m.get(module, {}) for m in recent_metrics]
            if module_data and all(d.get("status") == "available" for d in module_data):
                if module == "memory":
                    trends[module] = {
                        "cache_hit_rate_trend": self._calculate_trend([d.get("cache_hit_rate", 0) for d in module_data]),
                        "avg_latency_trend": self._calculate_trend([d.get("avg_latency_ms", 0) for d in module_data])
                    }
                elif module == "cognitive":
                    trends[module] = {
                        "processing_time_trend": self._calculate_trend([d.get("avg_processing_time", 0) for d in module_data])
                    }
                elif module == "api":
                    trends[module] = {
                        "success_rate_trend": self._calculate_trend([d.get("success_rate", 0) for d in module_data]),
                        "avg_latency_trend": self._calculate_trend([d.get("avg_latency", 0) for d in module_data])
                    }
                elif module == "protocol":
                    trends[module] = {
                        "conflict_rate_trend": self._calculate_trend([d.get("conflict_rate", 0) for d in module_data])
                    }

        return trends

    def _calculate_trend(self, values: List[float]) -> str:
        """计算趋势方向"""
        if len(values) < 2:
            return "stable"

        first_half = values[:len(values)//2]
        second_half = values[len(values)//2:]

        avg_first = sum(first_half) / len(first_half)
        avg_second = sum(second_half) / len(second_half)

        change = (avg_second - avg_first) / max(avg_first, 0.001)

        if change > 0.1:
            return "increasing"
        elif change < -0.1:
            return "decreasing"
        else:
            return "stable"

    def export_report(self, filepath: str):
        """导出报告到文件"""
        report = self.get_detailed_report()

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        print(f"[性能仪表板] 报告已导出到: {filepath}")

    def set_threshold(self, metric_name: str, value: float):
        """设置性能阈值"""
        if metric_name in self.thresholds:
            self.thresholds[metric_name] = value
            print(f"[性能仪表板] 阈值已更新: {metric_name} = {value}")
        else:
            print(f"[性能仪表板] 未知指标: {metric_name}")

    def clear_alerts(self):
        """清空告警"""
        self.alerts.clear()
        print("[性能仪表板] 告警已清空")

    def clear_history(self):
        """清空历史记录"""
        self.metrics_history.clear()
        print("[性能仪表板] 历史记录已清空")