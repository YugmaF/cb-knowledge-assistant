"""Token-bucket rate limiting, per user.

Each key (a user id, or an IP for login) owns a bucket holding up to `capacity` tokens that refills
continuously at `refill_per_minute`. A request spends `cost` tokens. The bucket allows short bursts
but caps the sustained rate, which a fixed window cannot do without a burst at every boundary.

In-process state is enough for one API replica. With several replicas the same algorithm moves
into Redis (one Lua script per consume) so that all replicas share a bucket.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable
from dataclasses import dataclass

from kb_assistant.config import RateLimitRule
from kb_assistant.errors import RateLimitedError


@dataclass
class TokenBucket:
    capacity: float
    refill_per_second: float
    tokens: float
    updated_at: float

    def _refill(self, now: float) -> None:
        elapsed = max(0.0, now - self.updated_at)
        self.tokens = min(self.capacity, self.tokens + elapsed * self.refill_per_second)
        self.updated_at = now

    def try_consume(self, cost: float, now: float) -> float:
        """Spend `cost` tokens. Returns 0 on success, otherwise the seconds until it would succeed."""
        self._refill(now)
        if self.tokens >= cost:
            self.tokens -= cost
            return 0.0
        return (cost - self.tokens) / self.refill_per_second


class RateLimiter:
    def __init__(
        self, rules: dict[str, RateLimitRule], clock: Callable[[], float] = time.monotonic
    ) -> None:
        self._rules = rules
        self._clock = clock
        self._buckets: dict[tuple[str, str], TokenBucket] = {}
        self._lock = asyncio.Lock()

    def rule_for(self, rule_name: str) -> RateLimitRule:
        try:
            return self._rules[rule_name]
        except KeyError as exc:
            raise ValueError(f"no rate-limit rule named {rule_name!r}") from exc

    async def consume(self, key: str, rule_name: str, cost: float = 1.0) -> None:
        """Raise RateLimitedError (-> HTTP 429 with Retry-After) when the bucket is empty."""
        rule = self.rule_for(rule_name)
        async with self._lock:
            now = self._clock()
            bucket = self._buckets.get((rule_name, key))
            if bucket is None:
                bucket = TokenBucket(rule.capacity, rule.refill_per_minute / 60.0, rule.capacity, now)
                self._buckets[(rule_name, key)] = bucket
            wait_s = bucket.try_consume(cost, now)
        if wait_s > 0:
            raise RateLimitedError(retry_after_s=wait_s)

    def remaining(self, key: str, rule_name: str) -> float:
        bucket = self._buckets.get((rule_name, key))
        if bucket is None:
            return float(self.rule_for(rule_name).capacity)
        bucket._refill(self._clock())
        return bucket.tokens
