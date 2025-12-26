"""
指标收集器模块 - 系统性能指标收集和监控
"""
import time
import threading
import json
from datetime import datetime, timedelta
from collections import defaultdict, deque
from typing import Dict, List, Optional, Any, Union, Callable
from dataclasses import dataclass, asdict
from enum import Enum


class MetricType(Enum):
    """指标类型"""
    COUNTER = "counter"  # 计数器，只增不减
    GAUGE = "gauge"  # 仪表，可增可减
    HISTOGRAM = "histogram"  # 直方图，统计分布
    TIMER = "timer"  # 计时器


@dataclass
class MetricData:
    """指标数据"""
    name: str
    type: MetricType
    value: float
    timestamp: datetime
    labels: Dict[str, str]

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        data = asdict(self)
        data["type"] = self.type.value
        data["timestamp"] = self.timestamp.isoformat()
        return data


class MetricsCollector:
    """指标收集器"""

    def __init__(self, name: str = "default"):
        self.name = name
        self.metrics: Dict[str, List[MetricData]] = defaultdict(list)
        self.lock = threading.RLock()

        # 历史数据限制
        self.max_history_per_metric = 1000

        # 预定义指标
        self.predefined_metrics = {
            "system.start_time": datetime.now()
        }

    def increment_counter(self, name: str, value: float = 1.0,
                          labels: Optional[Dict[str, str]] = None):
        """
        增加计数器

        Args:
            name: 指标名称
            value: 增加值
            labels: 标签
        """
        with self.lock:
            metric_data = MetricData(
                name=name,
                type=MetricType.COUNTER,
                value=value,
                timestamp=datetime.now(),
                labels=labels or {}
            )

            self._add_metric(metric_data)

    def set_gauge(self, name: str, value: float,
                  labels: Optional[Dict[str, str]] = None):
        """
        设置仪表值

        Args:
            name: 指标名称
            value: 值
            labels: 标签
        """
        with self.lock:
            metric_data = MetricData(
                name=name,
                type=MetricType.GAUGE,
                value=value,
                timestamp=datetime.now(),
                labels=labels or {}
            )

            self._add_metric(metric_data)

    def record_histogram(self, name: str, value: float,
                         labels: Optional[Dict[str, str]] = None):
        """
        记录直方图

        Args:
            name: 指标名称
            value: 值
            labels: 标签
        """
        with self.lock:
            metric_data = MetricData(
                name=name,
                type=MetricType.HISTOGRAM,
                value=value,
                timestamp=datetime.now(),
                labels=labels or {}
            )

            self._add_metric(metric_data)

    def start_timer(self, name: str) -> Callable[[], float]:
        """
        开始计时器

        Args:
            name: 计时器名称

        Returns:
            停止函数，调用返回耗时
        """
        start_time = time.time()

        def stop_timer(labels: Optional[Dict[str, str]] = None) -> float:
            elapsed = time.time() - start_time

            with self.lock:
                metric_data = MetricData(
                    name=name,
                    type=MetricType.TIMER,
                    value=elapsed,
                    timestamp=datetime.now(),
                    labels=labels or {}
                )

                self._add_metric(metric_data)

            return elapsed

        return stop_timer

    def _add_metric(self, metric_data: MetricData):
        """添加指标数据"""
        metric_list = self.metrics[metric_data.name]
        metric_list.append(metric_data)

        # 限制历史数据数量
        if len(metric_list) > self.max_history_per_metric:
            self.metrics[metric_data.name] = metric_list[-self.max_history_per_metric:]

    def get_metrics(self, name: Optional[str] = None,
                    since: Optional[datetime] = None,
                    until: Optional[datetime] = None) -> List[MetricData]:
        """
        获取指标数据

        Args:
            name: 指标名称（None表示所有）
            since: 开始时间
            until: 结束时间

        Returns:
            指标数据列表
        """
        with self.lock:
            result = []

            if name:
                metrics_to_check = {name: self.metrics.get(name, [])}
            else:
                metrics_to_check = self.metrics

            for metric_name, metric_list in metrics_to_check.items():
                for metric in metric_list:
                    if since and metric.timestamp < since:
                        continue
                    if until and metric.timestamp > until:
                        continue
                    result.append(metric)

            return result

    def get_statistics(self, name: str,
                       since: Optional[datetime] = None,
                       until: Optional[datetime] = None) -> Dict[str, Any]:
        """
        获取指标统计信息

        Args:
            name: 指标名称
            since: 开始时间
            until: 结束时间

        Returns:
            统计信息
        """
        metrics = self.get_metrics(name, since, until)

        if not metrics:
            return {
                "count": 0,
                "mean": 0.0,
                "min": 0.0,
                "max": 0.0,
                "sum": 0.0
            }

        values = [metric.value for metric in metrics]

        return {
            "count": len(values),
            "mean": sum(values) / len(values),
            "min": min(values),
            "max": max(values),
            "sum": sum(values),
            "latest": values[-1] if values else 0.0,
            "metric_type": metrics[0].type.value if metrics else "unknown"
        }

    def get_all_statistics(self, since: Optional[datetime] = None,
                           until: Optional[datetime] = None) -> Dict[str, Dict[str, Any]]:
        """获取所有指标的统计信息"""
        with self.lock:
            stats = {}
            for metric_name in self.metrics:
                stats[metric_name] = self.get_statistics(metric_name, since, until)

            return stats

    def clear_metrics(self, name: Optional[str] = None):
        """清除指标数据"""
        with self.lock:
            if name:
                if name in self.metrics:
                    self.metrics[name] = []
            else:
                self.metrics.clear()

    def export_json(self, filepath: str):
        """导出指标数据到JSON文件"""
        with self.lock:
            export_data = {
                "collector_name": self.name,
                "export_time": datetime.now().isoformat(),
                "metrics": {}
            }

            for metric_name, metric_list in self.metrics.items():
                export_data["metrics"][metric_name] = [
                    metric.to_dict() for metric in metric_list[-100:]  # 只导出最近100条
                ]

            import json
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, ensure_ascii=False, indent=2)


class SystemMetrics:
    """系统指标监控"""

    def __init__(self, collector: MetricsCollector):
        self.collector = collector
        self.running = False
        self.monitor_thread: Optional[threading.Thread] = None

        # 监控间隔（秒）
        self.monitor_interval = 60

    def start_monitoring(self):
        """开始监控"""
        if self.running:
            return

        self.running = True
        self.monitor_thread = threading.Thread(
            target=self._monitor_loop,
            daemon=True,
            name="SystemMetricsMonitor"
        )
        self.monitor_thread.start()

    def stop_monitoring(self):
        """停止监控"""
        self.running = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)

    def _monitor_loop(self):
        """监控循环"""
        import psutil

        while self.running:
            try:
                # 收集CPU使用率
                cpu_percent = psutil.cpu_percent(interval=1)
                self.collector.set_gauge("system.cpu.percent", cpu_percent)

                # 收集内存使用情况
                memory = psutil.virtual_memory()
                self.collector.set_gauge("system.memory.percent", memory.percent)
                self.collector.set_gauge("system.memory.used_mb", memory.used / 1024 / 1024)
                self.collector.set_gauge("system.memory.available_mb", memory.available / 1024 / 1024)

                # 收集磁盘使用情况
                disk = psutil.disk_usage('/')
                self.collector.set_gauge("system.disk.percent", disk.percent)
                self.collector.set_gauge("system.disk.used_gb", disk.used / 1024 / 1024 / 1024)

                # 收集网络IO
                net_io = psutil.net_io_counters()
                self.collector.set_gauge("system.network.bytes_sent", net_io.bytes_sent)
                self.collector.set_gauge("system.network.bytes_recv", net_io.bytes_recv)

            except Exception:
                # 监控失败时不中断
                pass

            # 等待下次收集
            time.sleep(self.monitor_interval)

    def get_system_summary(self) -> Dict[str, Any]:
        """获取系统摘要"""
        import psutil

        try:
            return {
                "cpu_percent": psutil.cpu_percent(interval=0.1),
                "memory_percent": psutil.virtual_memory().percent,
                "disk_percent": psutil.disk_usage('/').percent,
                "boot_time": datetime.fromtimestamp(psutil.boot_time()).isoformat(),
                "process_count": len(psutil.pids()),
                "collector_name": self.collector.name,
                "metrics_count": sum(len(metrics) for metrics in self.collector.metrics.values())
            }
        except Exception:
            return {}


class APIMetrics:
    """API指标监控"""

    def __init__(self, collector: MetricsCollector):
        self.collector = collector

    def record_request(self, endpoint: str, method: str,
                       status_code: int, duration: float):
        """
        记录API请求

        Args:
            endpoint: 端点
            method: HTTP方法
            status_code: 状态码
            duration: 持续时间（秒）
        """
        labels = {
            "endpoint": endpoint,
            "method": method,
            "status_code": str(status_code)
        }

        # 记录请求次数
        self.collector.increment_counter("api.requests.total", labels=labels)

        # 记录请求耗时
        self.collector.record_histogram("api.request.duration", duration, labels=labels)

        # 按状态码分类
        status_category = "success" if 200 <= status_code < 300 else "error"
        self.collector.increment_counter(f"api.requests.{status_category}", labels={
            "endpoint": endpoint,
            "method": method
        })

    def get_api_statistics(self, since: Optional[datetime] = None) -> Dict[str, Any]:
        """获取API统计信息"""
        stats = self.collector.get_all_statistics(since)

        api_stats = {
            "total_requests": 0,
            "success_rate": 0.0,
            "avg_response_time": 0.0,
            "endpoints": {}
        }

        # 计算总请求数
        total_key = "api.requests.total"
        if total_key in stats:
            api_stats["total_requests"] = int(stats[total_key]["sum"])

        # 计算成功率
        success_key = "api.requests.success"
        error_key = "api.requests.error"

        success_count = stats.get(success_key, {}).get("sum", 0)
        error_count = stats.get(error_key, {}).get("sum", 0)

        total_api_calls = success_count + error_count
        if total_api_calls > 0:
            api_stats["success_rate"] = (success_count / total_api_calls) * 100

        # 计算平均响应时间
        duration_key = "api.request.duration"
        if duration_key in stats:
            api_stats["avg_response_time"] = stats[duration_key]["mean"]

        return api_stats


# 全局指标收集器实例
global_collector = MetricsCollector("global")


# 快捷函数
def get_collector(name: str = "global") -> MetricsCollector:
    """获取指标收集器"""
    if name == "global":
        return global_collector
    return MetricsCollector(name)


def increment_counter(name: str, **kwargs):
    """增加计数器（快捷函数）"""
    return global_collector.increment_counter(name, **kwargs)


def start_timer(name: str) -> Callable[[], float]:
    """开始计时器（快捷函数）"""
    return global_collector.start_timer(name)


def get_system_metrics() -> SystemMetrics:
    """获取系统指标监控器"""
    return SystemMetrics(global_collector)


def get_api_metrics() -> APIMetrics:
    """获取API指标监控器"""
    return APIMetrics(global_collector)