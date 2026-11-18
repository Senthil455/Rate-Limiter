"""In-memory rate limiter for an LLM API gateway.

Two strategies are supported:

* :class:`SlidingWindowPolicy` -- an exact sliding-window log.  Every accepted
  request is recorded and expires ``window_ms`` after it was admitted, so the
  limiter never allows more than ``limit`` requests in any window of that
  length (no fixed-window burst at the boundary).
* :class:`TokenBucketPolicy` -- a bucket with a burst ``capacity`` that refills
  continuously at ``refill_per_second``.  This allows short bursts while
  enforcing a long-run average rate.

State is kept per ``(client_id, resource)`` pair and every operation is
serialised through a single re-entrant lock, so one :class:`RateLimiter`
instance can safely be shared by many threads.

All timestamps are monotonic milliseconds (``float``).  The clock is
injectable, which keeps behaviour fully deterministic in tests.

Example
-------
>>> limiter = RateLimiter(SlidingWindowPolicy(limit=2, window_ms=1000))
>>> limiter.check("alice", "chat").allowed
True
>>> limiter.check("alice", "chat").allowed
True
>>> result = limiter.check("alice", "chat")
>>> result.allowed, result.remaining, round(result.retry_after_ms)
(False, 0, 1000)
"""

from __future__ import annotations

import math
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Callable, Hashable

__all__ = [
    "RateLimiter",
    "RateLimitResult",
    "RateLimitDecision",
    "Decision",
    "LimiterStats",
    "KeyStats",
    "SlidingWindowPolicy",
    "TokenBucketPolicy",
    "Policy",
    "sliding_window_limiter",
    "token_bucket_limiter",
]

Key = tuple[Hashable, Hashable]

Clock = Callable[[], float]


# ---------------------------------------------------------------------------
# Policies
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SlidingWindowPolicy:
    """Allow at most ``limit`` requests in any trailing ``window_ms`` window.

    Attributes:
        limit: Maximum number of request units admitted per window. Must be
            >= 1.
        window_ms: Length of the sliding window in milliseconds. Must be > 0.
    """

    limit: int
    window_ms: float

    def __post_init__(self) -> None:
        if isinstance(self.limit, bool) or not isinstance(self.limit, int):
            raise TypeError("limit must be an int")
        if self.limit < 1:
            raise ValueError("limit must be >= 1")
        if not isinstance(self.window_ms, (int, float)) or isinstance(
            self.window_ms, bool
        ):
            raise TypeError("window_ms must be a number")
        if not math.isfinite(self.window_ms) or self.window_ms <= 0:
            raise ValueError("window_ms must be a finite number > 0")

    @property
    def capacity(self) -> int:
        """Maximum number of immediately admissible request units."""
        return self.limit


@dataclass(frozen=True)
class TokenBucketPolicy:
    """A bucket of ``capacity`` tokens refilling at ``refill_per_second``.

    Attributes:
        capacity: Maximum (and initial) number of tokens. Must be > 0.
        refill_per_second: Tokens added per second. Must be a finite number
            > 0.
    """

    capacity: float
    refill_per_second: float

    def __post_init__(self) -> None:
        for name in ("capacity", "refill_per_second"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{name} must be a number")
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be a finite number > 0")

    @property
    def bucket_size(self) -> int:
        """Whole tokens the bucket can hold (``floor(capacity)``)."""
        return math.floor(self.capacity)


Policy = SlidingWindowPolicy | TokenBucketPolicy


# ---------------------------------------------------------------------------
# Result / stats value objects
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RateLimitResult:
    """Outcome of a single admission check.

    Attributes:
        allowed: ``True`` when the request was admitted.
        remaining: Whole request units still available for this key.  For an
            allowed request this is the capacity left *after* consuming
            ``cost``; for a rejected request it is what is available *now*.
        retry_after_ms: Milliseconds until a request with the same cost would
            be admitted.  ``0.0`` when ``allowed`` is true (and
            ``math.inf`` when the cost can never be satisfied, i.e. it exceeds
            the policy capacity).
        limit: The policy's total capacity for the key.
        strategy: ``"sliding_window"`` or ``"token_bucket"``.
    """

    allowed: bool
    remaining: int
    retry_after_ms: float
    limit: int
    strategy: str

    def __bool__(self) -> bool:
        """Allow ``if limiter.check(...):`` to read naturally."""
        return self.allowed

    @property
    def reset_after_ms(self) -> float:
        """Alias for :attr:`retry_after_ms` (0 when allowed)."""
        return self.retry_after_ms

    @property
    def accepted(self) -> bool:
        """Alias for :attr:`allowed`."""
        return self.allowed

    def as_dict(self) -> dict[str, object]:
        """Plain-dict representation suitable for JSON responses."""
        return {
            "allowed": self.allowed,
            "remaining": self.remaining,
            "retry_after_ms": self.retry_after_ms,
            "limit": self.limit,
            "strategy": self.strategy,
        }


# Backwards/forwards friendly aliases: the concept has several common names.
RateLimitDecision = RateLimitResult
Decision = RateLimitResult


@dataclass(frozen=True)
class KeyStats:
    """Counters for one ``(client_id, resource)`` pair."""

    client_id: Hashable
    resource: Hashable
    total: int
    allowed: int
    rejected: int

    def as_dict(self) -> dict[str, object]:
        return {
            "client_id": self.client_id,
            "resource": self.resource,
            "total": self.total,
            "allowed": self.allowed,
            "rejected": self.rejected,
        }


@dataclass(frozen=True)
class LimiterStats:
    """Aggregate counters for a limiter instance."""

    total: int
    allowed: int
    rejected: int
    keys: dict[Key, KeyStats]

    def as_dict(self) -> dict[str, object]:
        return {
            "total": self.total,
            "allowed": self.allowed,
            "rejected": self.rejected,
            "keys": {key: value.as_dict() for key, value in self.keys.items()},
        }


# ---------------------------------------------------------------------------
# Internal per-key state
# ---------------------------------------------------------------------------


class _SlidingWindowState:
    __slots__ = ("events", "last_seen_ms")

    def __init__(self, now_ms: float) -> None:
        self.events: deque[float] = deque()
        self.last_seen_ms = now_ms


class _TokenBucketState:
    __slots__ = ("tokens", "last_refill_ms", "last_seen_ms")

    def __init__(self, tokens: float, now_ms: float) -> None:
        self.tokens = tokens
        self.last_refill_ms = now_ms
        self.last_seen_ms = now_ms


_State = _SlidingWindowState | _TokenBucketState


# ---------------------------------------------------------------------------
# Limiter
# ---------------------------------------------------------------------------


class RateLimiter:
    """Thread-safe, in-memory rate limiter.

    Args:
        policy: Default policy applied to keys without an explicit override.
            It may be omitted if every key is configured with
            :meth:`configure`.
        clock: Callable returning monotonic milliseconds.  Defaults to
            ``time.monotonic_ns() / 1e6``.  Inject a fake clock for tests.
    """

    def __init__(
        self,
        policy: Policy | None = None,
        *,
        clock: Clock | None = None,
    ) -> None:
        if policy is not None and not isinstance(
            policy, (SlidingWindowPolicy, TokenBucketPolicy)
        ):
            raise TypeError(
                "policy must be a SlidingWindowPolicy or TokenBucketPolicy"
            )
        self._default_policy = policy
        self._policies: dict[Key, Policy] = {}
        self._states: dict[Key, _State] = {}
        self._lock = threading.RLock()
        self._clock: Clock = clock or (lambda: time.monotonic_ns() / 1_000_000.0)
        self._total = 0
        self._allowed = 0
        self._rejected = 0
        self._key_stats: dict[Key, list[int]] = {}

    # -- convenience constructors ------------------------------------------

    @classmethod
    def sliding_window(
        cls,
        limit: int,
        window_ms: float,
        *,
        clock: Clock | None = None,
    ) -> "RateLimiter":
        """Create a limiter backed by a sliding-window policy."""
        return cls(SlidingWindowPolicy(limit, window_ms), clock=clock)

    @classmethod
    def token_bucket(
        cls,
        capacity: float,
        refill_per_second: float,
        *,
        clock: Clock | None = None,
    ) -> "RateLimiter":
        """Create a limiter backed by a token-bucket policy."""
        return cls(TokenBucketPolicy(capacity, refill_per_second), clock=clock)

    # -- configuration ------------------------------------------------------

    @property
    def default_policy(self) -> Policy | None:
        return self._default_policy

    def configure(
        self,
        client_id: Hashable,
        resource: Hashable,
        policy: Policy,
        *,
        reset: bool = True,
    ) -> None:
        """Assign a policy to one key.

        Args:
            client_id: Client identifier (any hashable, typically ``str``).
            resource: Resource identifier (any hashable).
            policy: Policy used for this key.
            reset: When true (default), discard existing state for the key so
                the new policy starts from a clean slate.
        """
        if not isinstance(policy, (SlidingWindowPolicy, TokenBucketPolicy)):
            raise TypeError(
                "policy must be a SlidingWindowPolicy or TokenBucketPolicy"
            )
        key = self._normalize_key(client_id, resource)
        with self._lock:
            self._policies[key] = policy
            if reset:
                self._states.pop(key, None)

    def policy_for(self, client_id: Hashable, resource: Hashable) -> Policy:
        """Return the effective policy for a key (override or default)."""
        key = self._normalize_key(client_id, resource)
        with self._lock:
            return self._resolve_policy(key)

    # -- admission ----------------------------------------------------------

    def check(
        self,
        client_id: Hashable,
        resource: Hashable,
        cost: int = 1,
    ) -> RateLimitResult:
        """Atomically decide whether a request is admitted.

        On admission the capacity is consumed; on rejection the state is left
        untouched apart from bookkeeping (token-bucket refill still advances,
        which is required for a correct recovery estimate).

        Args:
            client_id: Client identifier.
            resource: Resource identifier (e.g. model or endpoint name).
            cost: Request units to consume. Must be an int >= 1.

        Returns:
            A :class:`RateLimitResult`.
        """
        self._validate_cost(cost)
        key = self._normalize_key(client_id, resource)
        with self._lock:
            now = self._clock()
            policy = self._resolve_policy(key)
            state = self._state_for(key, policy, now)
            result = self._evaluate(policy, state, now, cost, consume=True)
            self._record(key, result.allowed)
            return result

    def allow(
        self,
        client_id: Hashable,
        resource: Hashable,
        cost: int = 1,
    ) -> bool:
        """Boolean-only convenience wrapper around :meth:`check`."""
        return self.check(client_id, resource, cost).allowed

    # Common alternative names, kept as thin aliases so the implementation is
    # easy to drop into an existing gateway.
    allow_request = allow
    try_acquire = allow
    is_allowed = allow
    acquire = allow

    def peek(self, client_id: Hashable, resource: Hashable) -> RateLimitResult:
        """Evaluate a cost-1 request *without* consuming capacity."""
        key = self._normalize_key(client_id, resource)
        with self._lock:
            now = self._clock()
            policy = self._resolve_policy(key)
            state = self._state_for(key, policy, now)
            return self._evaluate(policy, state, now, 1, consume=False)

    def remaining(self, client_id: Hashable, resource: Hashable) -> int:
        """Whole request units currently available for a key."""
        return self.peek(client_id, resource).remaining

    # -- maintenance --------------------------------------------------------

    def reset(
        self,
        client_id: Hashable | None = None,
        resource: Hashable | None = None,
    ) -> None:
        """Drop state for one key, one dimension, or everything.

        ``reset()`` clears all per-key state but keeps configuration and
        cumulative statistics.  Passing only ``client_id`` clears every
        resource for that client; passing only ``resource`` clears that
        resource for every client.
        """
        with self._lock:
            if client_id is None and resource is None:
                self._states.clear()
                return
            doomed = [
                key
                for key in self._states
                if (client_id is None or key[0] == client_id)
                and (resource is None or key[1] == resource)
            ]
            for key in doomed:
                del self._states[key]

    def cleanup(self, max_idle_ms: float) -> int:
        """Evict idle keys to bound memory. Returns the number removed.

        A key is idle when no check touched it for ``max_idle_ms``
        milliseconds.  A sliding-window key with events still inside its
        window is never considered idle, so eviction cannot silently grant
        extra capacity.
        """
        if not math.isfinite(max_idle_ms) or max_idle_ms < 0:
            raise ValueError("max_idle_ms must be a finite number >= 0")
        removed = 0
        with self._lock:
            now = self._clock()
            for key in list(self._states):
                state = self._states[key]
                if key in self._policies:
                    continue  # explicitly configured keys are pinned
                idle_for = now - state.last_seen_ms
                if idle_for < max_idle_ms:
                    continue
                if isinstance(state, _SlidingWindowState) and state.events:
                    policy = self._policies.get(key, self._default_policy)
                    if isinstance(policy, SlidingWindowPolicy) and (
                        state.events[-1] + policy.window_ms > now
                    ):
                        continue
                del self._states[key]
                removed += 1
        return removed

    def stats(self) -> LimiterStats:
        """Snapshot of aggregate and per-key counters."""
        with self._lock:
            return LimiterStats(
                total=self._total,
                allowed=self._allowed,
                rejected=self._rejected,
                keys={
                    key: KeyStats(key[0], key[1], values[0], values[1], values[2])
                    for key, values in self._key_stats.items()
                },
            )

    def stats_dict(self) -> dict[str, object]:
        """JSON-friendly view of :meth:`stats`."""
        return self.stats().as_dict()

    def reset_stats(self) -> None:
        """Zero all statistics without touching limiter state."""
        with self._lock:
            self._total = 0
            self._allowed = 0
            self._rejected = 0
            self._key_stats.clear()

    # -- internals ----------------------------------------------------------

    @staticmethod
    def _normalize_key(client_id: Hashable, resource: Hashable) -> Key:
        if not isinstance(client_id, Hashable) or not isinstance(
            resource, Hashable
        ):
            raise TypeError("client_id and resource must be hashable")
        return (client_id, resource)

    @staticmethod
    def _validate_cost(cost: int) -> None:
        if isinstance(cost, bool) or not isinstance(cost, int):
            raise TypeError("cost must be an int")
        if cost < 1:
            raise ValueError("cost must be >= 1")

    def _resolve_policy(self, key: Key) -> Policy:
        policy = self._policies.get(key, self._default_policy)
        if policy is None:
            raise LookupError(
                f"no policy configured for client={key[0]!r}, resource={key[1]!r}"
            )
        return policy

    def _state_for(self, key: Key, policy: Policy, now: float) -> _State:
        state = self._states.get(key)
        if state is None:
            if isinstance(policy, SlidingWindowPolicy):
                state = _SlidingWindowState(now)
            else:
                state = _TokenBucketState(policy.capacity, now)
            self._states[key] = state
        elif isinstance(policy, SlidingWindowPolicy):
            if not isinstance(state, _SlidingWindowState):
                state = _SlidingWindowState(now)
                self._states[key] = state
        elif not isinstance(state, _TokenBucketState):
            state = _TokenBucketState(policy.capacity, now)
            self._states[key] = state
        return state

    def _evaluate(
        self,
        policy: Policy,
        state: _State,
        now: float,
        cost: int,
        *,
        consume: bool,
    ) -> RateLimitResult:
        if isinstance(policy, SlidingWindowPolicy):
            assert isinstance(state, _SlidingWindowState)
            return self._evaluate_sliding_window(policy, state, now, cost, consume)
        assert isinstance(state, _TokenBucketState)
        return self._evaluate_token_bucket(policy, state, now, cost, consume)

    @staticmethod
    def _evaluate_sliding_window(
        policy: SlidingWindowPolicy,
        state: _SlidingWindowState,
        now: float,
        cost: int,
        consume: bool,
    ) -> RateLimitResult:
        events = state.events
        state.last_seen_ms = now

        # Drop everything that has left the trailing window: (now - w, now].
        cutoff = now - policy.window_ms
        while events and events[0] <= cutoff:
            events.popleft()

        used = len(events)
        available = policy.limit - used

        if cost > policy.limit:
            # Can never succeed, whatever the window state.
            return RateLimitResult(
                allowed=False,
                remaining=max(0, available),
                retry_after_ms=math.inf,
                limit=policy.limit,
                strategy="sliding_window",
            )

        if cost <= available:
            if consume:
                events.extend([now] * cost)
                available -= cost
            return RateLimitResult(
                allowed=True,
                remaining=max(0, available),
                retry_after_ms=0.0,
                limit=policy.limit,
                strategy="sliding_window",
            )

        # Not enough room: the request becomes admissible once enough of the
        # oldest events expire.  ``need`` events must leave the window.
        need = cost - available
        expires_at = events[need - 1] + policy.window_ms
        retry_after = max(0.0, expires_at - now)
        return RateLimitResult(
            allowed=False,
            remaining=available,
            retry_after_ms=retry_after,
            limit=policy.limit,
            strategy="sliding_window",
        )

    @staticmethod
    def _evaluate_token_bucket(
        policy: TokenBucketPolicy,
        state: _TokenBucketState,
        now: float,
        cost: int,
        consume: bool,
    ) -> RateLimitResult:
        state.last_seen_ms = now

        elapsed = now - state.last_refill_ms
        if elapsed > 0:
            state.tokens = min(
                policy.capacity,
                state.tokens + elapsed * policy.refill_per_second / 1000.0,
            )
            state.last_refill_ms = now

        # Tolerance absorbs float drift so an exactly-full bucket is not
        # rejected by 1e-16.
        if cost <= policy.capacity + 1e-9 and state.tokens + 1e-9 >= cost:
            if consume:
                state.tokens = max(0.0, state.tokens - cost)
            return RateLimitResult(
                allowed=True,
                remaining=max(0, math.floor(state.tokens + 1e-9)),
                retry_after_ms=0.0,
                limit=policy.bucket_size,
                strategy="token_bucket",
            )

        if cost > policy.capacity + 1e-9:
            retry_after = math.inf  # the bucket can never hold this many
        else:
            missing = cost - state.tokens
            retry_after = max(
                0.0, missing / policy.refill_per_second * 1000.0
            )
        return RateLimitResult(
            allowed=False,
            remaining=max(0, math.floor(state.tokens + 1e-9)),
            retry_after_ms=retry_after,
            limit=policy.bucket_size,
            strategy="token_bucket",
        )

    def _record(self, key: Key, allowed: bool) -> None:
        self._total += 1
        if allowed:
            self._allowed += 1
        else:
            self._rejected += 1
        counters = self._key_stats.get(key)
        if counters is None:
            counters = [0, 0, 0]
            self._key_stats[key] = counters
        counters[0] += 1
        counters[1 if allowed else 2] += 1

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (
            f"RateLimiter(default_policy={self._default_policy!r}, "
            f"keys={len(self._states)})"
        )


# ---------------------------------------------------------------------------
# Factory helpers
# ---------------------------------------------------------------------------


def sliding_window_limiter(
    limit: int, window_ms: float, *, clock: Clock | None = None
) -> RateLimiter:
    """Build a :class:`RateLimiter` using a sliding-window policy."""
    return RateLimiter.sliding_window(limit, window_ms, clock=clock)


def token_bucket_limiter(
    capacity: float,
    refill_per_second: float,
    *,
    clock: Clock | None = None,
) -> RateLimiter:
    """Build a :class:`RateLimiter` using a token-bucket policy."""
    return RateLimiter.token_bucket(capacity, refill_per_second, clock=clock)
