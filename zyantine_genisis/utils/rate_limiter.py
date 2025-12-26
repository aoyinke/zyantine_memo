"""
速率限制器模块 - 控制API调用频率和配额
"""
import time
import threading
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from collections import defaultdict, deque
from dataclasses import dataclass
from enum import Enum

from utils.logger import get_logger
from utils.metrics import get_collector


class RateLimitPolicy(Enum):
    """速率限制策略"""
    FIXED_WINDOW = "fixed_window"  # 固定窗口
    SLIDING_WINDOW = "sliding_window"  # 滑动窗口
    TOKEN_BUCKET = "token_bucket"  # 令牌桶
    LEAKY_BUCKET = "leaky_bucket"  # 漏桶


class RateLimitType(Enum):
    """速率限制类型"""
    REQUESTS_PER_MINUTE = "rpm"  # 每分钟请求数
    REQUESTS_PER_HOUR = "rph"  # 每小时请求数
    REQUESTS_PER_DAY = "rpd"  # 每天请求数
    TOKENS_PER_MINUTE = "tpm"  # 每分钟tokens数
    TOKENS_PER_HOUR = "tph"  # 每小时tokens数


@dataclass
class RateLimitConfig:
    """速率限制配置"""
    limit_type: RateLimitType
    limit_value: int
    policy: RateLimitPolicy = RateLimitPolicy.FIXED_WINDOW
    window_size: int = 60  # 窗口大小（秒）
    burst_capacity: Optional[int] = None  # 突发容量（令牌桶/漏桶）
    refill_rate: Optional[float] = None  # 填充速率（令牌桶/漏桶）
    user_specific: bool = False  # 是否用户特定


@dataclass
class RateLimitResult:
    """速率限制结果"""
    allowed: bool
    limit_type: RateLimitType
    current_usage: int
    limit_value: int
    remaining: int
    reset_time: float
    wait_time: Optional[float] = None
    reason: Optional[str] = None


class RateLimiter:
    """速率限制器"""

    def __init__(self, name: str = "default"):
        """
        初始化速率限制器

        Args:
            name: 限制器名称
        """
        self.name = name
        self.logger = get_logger(f"rate_limiter.{name}")
        self.metrics = get_collector(f"rate_limiter.{name}")

        # 存储配置和状态
        self.configs: Dict[str, RateLimitConfig] = {}
        self.states: Dict[str, Dict] = defaultdict(dict)
        self.lock = threading.RLock()

        # 默认配置
        self._init_default_configs()

        # 清理过期状态的线程
        self._cleanup_thread = None
        self._running = False

        self.logger.info(f"速率限制器 '{name}' 初始化完成")

    def _init_default_configs(self):
        """初始化默认配置"""
        # OpenAI API 默认限制
        self.add_config(
            key="openai_requests",
            config=RateLimitConfig(
                limit_type=RateLimitType.REQUESTS_PER_MINUTE,
                limit_value=60,  # 60 RPM
                policy=RateLimitPolicy.FIXED_WINDOW,
                window_size=60
            )
        )

        self.add_config(
            key="openai_tokens",
            config=RateLimitConfig(
                limit_type=RateLimitType.TOKENS_PER_MINUTE,
                limit_value=150000,  # 150K TPM
                policy=RateLimitPolicy.TOKEN_BUCKET,
                window_size=60,
                burst_capacity=200000,
                refill_rate=2500  # tokens/秒
            )
        )

        # 用户级别限制
        self.add_config(
            key="user_requests",
            config=RateLimitConfig(
                limit_type=RateLimitType.REQUESTS_PER_MINUTE,
                limit_value=30,
                policy=RateLimitPolicy.SLIDING_WINDOW,
                window_size=60,
                user_specific=True
            )
        )

    def add_config(self, key: str, config: RateLimitConfig):
        """
        添加速率限制配置

        Args:
            key: 配置键
            config: 速率限制配置
        """
        with self.lock:
            self.configs[key] = config
            self.logger.info(f"添加速率限制配置: {key} -> {config}")

    def remove_config(self, key: str):
        """移除速率限制配置"""
        with self.lock:
            if key in self.configs:
                del self.configs[key]
                self.logger.info(f"移除速率限制配置: {key}")

    def check_limit(self, key: str, value: int = 1,
                    user_id: Optional[str] = None) -> RateLimitResult:
        """
        检查是否超过限制

        Args:
            key: 配置键
            value: 请求值（如token数量）
            user_id: 用户ID（用于用户特定限制）

        Returns:
            速率限制结果
        """
        if key not in self.configs:
            # 没有配置限制，直接允许
            return RateLimitResult(
                allowed=True,
                limit_type=RateLimitType.REQUESTS_PER_MINUTE,
                current_usage=0,
                limit_value=0,
                remaining=0,
                reset_time=time.time() + 60
            )

        config = self.configs[key]
        state_key = self._get_state_key(key, user_id)

        with self.lock:
            # 获取或创建状态
            state = self.states[state_key]

            if config.policy == RateLimitPolicy.FIXED_WINDOW:
                result = self._check_fixed_window(config, state, value)
            elif config.policy == RateLimitPolicy.SLIDING_WINDOW:
                result = self._check_sliding_window(config, state, value)
            elif config.policy == RateLimitPolicy.TOKEN_BUCKET:
                result = self._check_token_bucket(config, state, value)
            elif config.policy == RateLimitPolicy.LEAKY_BUCKET:
                result = self._check_leaky_bucket(config, state, value)
            else:
                result = RateLimitResult(
                    allowed=True,
                    limit_type=config.limit_type,
                    current_usage=0,
                    limit_value=config.limit_value,
                    remaining=config.limit_value,
                    reset_time=time.time() + config.window_size,
                    reason=f"未知策略: {config.policy}"
                )

            # 记录指标
            self._record_metrics(key, result, user_id)

            return result

    def _get_state_key(self, key: str, user_id: Optional[str] = None) -> str:
        """获取状态键"""
        if user_id:
            return f"{key}:{user_id}"
        return key

    def _check_fixed_window(self, config: RateLimitConfig,
                            state: Dict, value: int) -> RateLimitResult:
        """检查固定窗口限制"""
        current_time = time.time()
        window_start = state.get("window_start", current_time)
        current_usage = state.get("usage", 0)

        # 检查是否在新窗口
        if current_time - window_start >= config.window_size:
            window_start = current_time
            current_usage = 0

        # 检查限制
        new_usage = current_usage + value
        allowed = new_usage <= config.limit_value

        # 计算等待时间
        wait_time = None
        if not allowed:
            wait_time = config.window_size - (current_time - window_start)

        # 更新状态（只在允许时更新）
        if allowed:
            state["window_start"] = window_start
            state["usage"] = new_usage

        return RateLimitResult(
            allowed=allowed,
            limit_type=config.limit_type,
            current_usage=current_usage,
            limit_value=config.limit_value,
            remaining=config.limit_value - current_usage,
            reset_time=window_start + config.window_size,
            wait_time=wait_time,
            reason="超出固定窗口限制" if not allowed else None
        )

    def _check_sliding_window(self, config: RateLimitConfig,
                              state: Dict, value: int) -> RateLimitResult:
        """检查滑动窗口限制"""
        current_time = time.time()

        # 获取或初始化请求时间队列
        requests = state.get("requests", deque())

        # 移除窗口外的请求
        window_start = current_time - config.window_size
        while requests and requests[0] < window_start:
            requests.popleft()

        # 检查限制
        current_usage = len(requests)
        allowed = (current_usage + value) <= config.limit_value

        # 计算等待时间
        wait_time = None
        if not allowed and requests:
            # 等待最早的请求离开窗口
            wait_time = requests[0] + config.window_size - current_time

        # 更新状态（只在允许时更新）
        if allowed:
            for _ in range(value):
                requests.append(current_time)
            state["requests"] = requests

        return RateLimitResult(
            allowed=allowed,
            limit_type=config.limit_type,
            current_usage=current_usage,
            limit_value=config.limit_value,
            remaining=config.limit_value - current_usage,
            reset_time=requests[0] + config.window_size if requests else current_time + config.window_size,
            wait_time=wait_time,
            reason="超出滑动窗口限制" if not allowed else None
        )

    def _check_token_bucket(self, config: RateLimitConfig,
                            state: Dict, value: int) -> RateLimitResult:
        """检查令牌桶限制"""
        if config.refill_rate is None:
            # 回退到固定窗口
            return self._check_fixed_window(config, state, value)

        current_time = time.time()

        # 获取或初始化状态
        tokens = state.get("tokens", config.limit_value)
        last_refill = state.get("last_refill", current_time)

        # 补充令牌
        time_passed = current_time - last_refill
        new_tokens = time_passed * config.refill_rate
        tokens = min(config.limit_value, tokens + new_tokens)

        # 检查是否有足够令牌
        allowed = tokens >= value

        # 计算等待时间
        wait_time = None
        if not allowed:
            # 需要等待的令牌数
            tokens_needed = value - tokens
            wait_time = tokens_needed / config.refill_rate

        # 更新状态（只在允许时更新）
        if allowed:
            tokens -= value
            state["tokens"] = tokens
            state["last_refill"] = current_time

        return RateLimitResult(
            allowed=allowed,
            limit_type=config.limit_type,
            current_usage=config.limit_value - tokens,
            limit_value=config.limit_value,
            remaining=int(tokens),
            reset_time=current_time + (config.limit_value - tokens) / config.refill_rate,
            wait_time=wait_time,
            reason="令牌不足" if not allowed else None
        )

    def _check_leaky_bucket(self, config: RateLimitConfig,
                            state: Dict, value: int) -> RateLimitResult:
        """检查漏桶限制"""
        if config.refill_rate is None:
            # 回退到固定窗口
            return self._check_fixed_window(config, state, value)

        current_time = time.time()

        # 获取或初始化状态
        water_level = state.get("water_level", 0)
        last_leak = state.get("last_leak", current_time)

        # 漏水
        time_passed = current_time - last_leak
        leaked = time_passed * config.refill_rate
        water_level = max(0, water_level - leaked)

        # 检查是否有容量
        allowed = (water_level + value) <= (config.burst_capacity or config.limit_value)

        # 计算等待时间
        wait_time = None
        if not allowed:
            # 需要等待的容量
            capacity_needed = water_level + value - (config.burst_capacity or config.limit_value)
            wait_time = capacity_needed / config.refill_rate

        # 更新状态（只在允许时更新）
        if allowed:
            water_level += value
            state["water_level"] = water_level
            state["last_leak"] = current_time

        return RateLimitResult(
            allowed=allowed,
            limit_type=config.limit_type,
            current_usage=water_level,
            limit_value=config.burst_capacity or config.limit_value,
            remaining=int((config.burst_capacity or config.limit_value) - water_level),
            reset_time=current_time + water_level / config.refill_rate,
            wait_time=wait_time,
            reason="漏桶容量不足" if not allowed else None
        )

    def _record_metrics(self, key: str, result: RateLimitResult,
                        user_id: Optional[str] = None):
        """记录指标"""
        # 记录请求
        self.metrics.increment_counter("rate_limit.requests", labels={
            "key": key,
            "allowed": str(result.allowed),
            "user_specific": str(user_id is not None)
        })

        # 记录拒绝
        if not result.allowed:
            self.metrics.increment_counter("rate_limit.rejected", labels={
                "key": key,
                "reason": result.reason or "unknown"
            })

        # 记录使用率
        usage_percentage = (result.current_usage / max(result.limit_value, 1)) * 100
        self.metrics.set_gauge("rate_limit.usage_percentage", usage_percentage, labels={
            "key": key
        })

    def wait_if_needed(self, key: str, value: int = 1,
                       user_id: Optional[str] = None,
                       max_wait: float = 60.0) -> bool:
        """
        如果需要等待，则等待直到可以执行

        Args:
            key: 配置键
            value: 请求值
            user_id: 用户ID
            max_wait: 最大等待时间（秒）

        Returns:
            是否成功获得许可
        """
        start_time = time.time()

        while True:
            result = self.check_limit(key, value, user_id)

            if result.allowed:
                return True

            # 检查是否超过最大等待时间
            elapsed = time.time() - start_time
            if elapsed >= max_wait:
                self.logger.warning(f"等待超时: {key}, 已等待 {elapsed:.1f}秒")
                return False

            # 等待一小段时间
            wait_time = min(result.wait_time or 0.1, 1.0)
            self.logger.debug(f"等待 {wait_time:.2f}秒: {key}")
            time.sleep(wait_time)

    def get_usage_statistics(self, key: Optional[str] = None,
                             user_id: Optional[str] = None) -> Dict[str, Any]:
        """
        获取使用统计

        Args:
            key: 配置键（None表示所有）
            user_id: 用户ID

        Returns:
            使用统计
        """
        with self.lock:
            stats = {}

            keys_to_check = [key] if key else list(self.configs.keys())

            for check_key in keys_to_check:
                if check_key not in self.configs:
                    continue

                config = self.configs[check_key]
                state_key = self._get_state_key(check_key, user_id)
                state = self.states.get(state_key, {})

                if config.policy == RateLimitPolicy.FIXED_WINDOW:
                    window_start = state.get("window_start", time.time())
                    current_usage = state.get("usage", 0)
                    time_remaining = max(0, window_start + config.window_size - time.time())

                    stats[check_key] = {
                        "policy": config.policy.value,
                        "current_usage": current_usage,
                        "limit": config.limit_value,
                        "remaining": config.limit_value - current_usage,
                        "usage_percentage": (current_usage / config.limit_value) * 100,
                        "window_remaining_seconds": time_remaining,
                        "window_size_seconds": config.window_size
                    }

                elif config.policy == RateLimitPolicy.SLIDING_WINDOW:
                    requests = state.get("requests", deque())
                    current_usage = len(requests)
                    time_remaining = 0
                    if requests:
                        time_remaining = max(0, requests[0] + config.window_size - time.time())

                    stats[check_key] = {
                        "policy": config.policy.value,
                        "current_usage": current_usage,
                        "limit": config.limit_value,
                        "remaining": config.limit_value - current_usage,
                        "usage_percentage": (current_usage / config.limit_value) * 100,
                        "window_remaining_seconds": time_remaining,
                        "window_size_seconds": config.window_size
                    }

                elif config.policy == RateLimitPolicy.TOKEN_BUCKET:
                    tokens = state.get("tokens", config.limit_value)
                    current_usage = config.limit_value - tokens

                    stats[check_key] = {
                        "policy": config.policy.value,
                        "current_usage": current_usage,
                        "limit": config.limit_value,
                        "remaining": int(tokens),
                        "usage_percentage": (current_usage / config.limit_value) * 100,
                        "burst_capacity": config.burst_capacity,
                        "refill_rate": config.refill_rate
                    }

                elif config.policy == RateLimitPolicy.LEAKY_BUCKET:
                    water_level = state.get("water_level", 0)
                    current_usage = water_level
                    limit = config.burst_capacity or config.limit_value

                    stats[check_key] = {
                        "policy": config.policy.value,
                        "current_usage": current_usage,
                        "limit": limit,
                        "remaining": int(limit - water_level),
                        "usage_percentage": (current_usage / limit) * 100,
                        "burst_capacity": config.burst_capacity,
                        "refill_rate": config.refill_rate
                    }

            return stats

    def reset_limits(self, key: Optional[str] = None,
                     user_id: Optional[str] = None):
        """
        重置限制状态

        Args:
            key: 配置键（None表示所有）
            user_id: 用户ID
        """
        with self.lock:
            if key:
                state_key = self._get_state_key(key, user_id)
                if state_key in self.states:
                    del self.states[state_key]
                    self.logger.info(f"重置限制状态: {state_key}")
            else:
                # 重置所有状态
                self.states.clear()
                self.logger.info("重置所有限制状态")

    def start_cleanup_thread(self, interval: float = 300.0):
        """
        启动清理线程（清理过期状态）

        Args:
            interval: 清理间隔（秒）
        """
        if self._cleanup_thread and self._cleanup_thread.is_alive():
            self.logger.warning("清理线程已在运行")
            return

        self._running = True
        self._cleanup_thread = threading.Thread(
            target=self._cleanup_loop,
            args=(interval,),
            daemon=True,
            name=f"RateLimiterCleanup-{self.name}"
        )
        self._cleanup_thread.start()
        self.logger.info(f"启动清理线程，间隔: {interval}秒")

    def stop_cleanup_thread(self):
        """停止清理线程"""
        self._running = False
        if self._cleanup_thread:
            self._cleanup_thread.join(timeout=5.0)
            self.logger.info("清理线程已停止")

    def _cleanup_loop(self, interval: float):
        """清理循环"""
        while self._running:
            time.sleep(interval)
            self._cleanup_expired_states()

    def _cleanup_expired_states(self):
        """清理过期状态"""
        with self.lock:
            current_time = time.time()
            keys_to_remove = []

            for key, state in self.states.items():
                config_key = key.split(":")[0]  # 移除用户ID部分

                if config_key not in self.configs:
                    # 配置已删除，清理状态
                    keys_to_remove.append(key)
                    continue

                config = self.configs[config_key]

                # 根据策略检查是否过期
                if config.policy == RateLimitPolicy.FIXED_WINDOW:
                    window_start = state.get("window_start", 0)
                    if current_time - window_start > config.window_size * 2:
                        keys_to_remove.append(key)

                elif config.policy == RateLimitPolicy.SLIDING_WINDOW:
                    requests = state.get("requests", deque())
                    if not requests:
                        keys_to_remove.append(key)

                elif config.policy in [RateLimitPolicy.TOKEN_BUCKET,
                                       RateLimitPolicy.LEAKY_BUCKET]:
                    last_update = state.get("last_refill", state.get("last_leak", 0))
                    if current_time - last_update > 3600:  # 1小时无活动
                        keys_to_remove.append(key)

            # 清理过期状态
            for key in keys_to_remove:
                del self.states[key]

            if keys_to_remove:
                self.logger.debug(f"清理了 {len(keys_to_remove)} 个过期状态")

    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        with self.lock:
            metrics_stats = self.metrics.get_all_statistics()

            # 计算总体统计
            total_requests = metrics_stats.get("rate_limit.requests", {}).get("sum", 0)
            total_rejected = metrics_stats.get("rate_limit.rejected", {}).get("sum", 0)

            rejection_rate = 0.0
            if total_requests > 0:
                rejection_rate = (total_rejected / total_requests) * 100

            return {
                "name": self.name,
                "config_count": len(self.configs),
                "state_count": len(self.states),
                "total_requests": total_requests,
                "total_rejected": total_rejected,
                "rejection_rate": round(rejection_rate, 2),
                "configs": {k: v.limit_value for k, v in self.configs.items()},
                "cleanup_thread_running": self._cleanup_thread is not None and self._cleanup_thread.is_alive(),
                "last_updated": datetime.now().isoformat()
            }


class MultiTenantRateLimiter:
    """多租户速率限制器"""

    def __init__(self):
        self.limiters: Dict[str, RateLimiter] = {}
        self.lock = threading.RLock()
        self.logger = get_logger("multi_tenant_rate_limiter")

    def get_limiter(self, tenant_id: str) -> RateLimiter:
        """
        获取或创建租户的速率限制器

        Args:
            tenant_id: 租户ID

        Returns:
            速率限制器
        """
        with self.lock:
            if tenant_id not in self.limiters:
                self.limiters[tenant_id] = RateLimiter(f"tenant_{tenant_id}")
                self.logger.info(f"为租户创建速率限制器: {tenant_id}")

            return self.limiters[tenant_id]

    def remove_limiter(self, tenant_id: str):
        """移除租户的速率限制器"""
        with self.lock:
            if tenant_id in self.limiters:
                self.limiters[tenant_id].stop_cleanup_thread()
                del self.limiters[tenant_id]
                self.logger.info(f"移除租户速率限制器: {tenant_id}")

    def get_all_statistics(self) -> Dict[str, Dict[str, Any]]:
        """获取所有租户的统计信息"""
        with self.lock:
            stats = {}
            for tenant_id, limiter in self.limiters.items():
                stats[tenant_id] = limiter.get_statistics()

            return stats

    def cleanup_expired_tenants(self, max_inactive_hours: float = 24.0):
        """
        清理不活动的租户

        Args:
            max_inactive_hours: 最大不活动时间（小时）
        """
        # 这里需要根据实际使用情况实现
        # 可能需要记录租户的最后活动时间
        pass


# 全局速率限制器实例
_global_rate_limiter = RateLimiter("global")


# 快捷函数
def get_rate_limiter(name: str = "global") -> RateLimiter:
    """获取速率限制器"""
    if name == "global":
        return _global_rate_limiter
    return RateLimiter(name)


def check_rate_limit(key: str, value: int = 1,
                     user_id: Optional[str] = None) -> RateLimitResult:
    """检查速率限制（快捷函数）"""
    return _global_rate_limiter.check_limit(key, value, user_id)


def wait_for_rate_limit(key: str, value: int = 1,
                        user_id: Optional[str] = None,
                        max_wait: float = 60.0) -> bool:
    """等待速率限制（快捷函数）"""
    return _global_rate_limiter.wait_if_needed(key, value, user_id, max_wait)


def get_rate_limit_stats(key: Optional[str] = None,
                         user_id: Optional[str] = None) -> Dict[str, Any]:
    """获取速率限制统计（快捷函数）"""
    return _global_rate_limiter.get_usage_statistics(key, user_id)