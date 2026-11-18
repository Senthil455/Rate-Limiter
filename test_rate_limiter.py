"""Unit tests for the in-memory rate limiter."""

from __future__ import annotations

import math
import threading
import unittest

from rate_limiter import (
    Decision,
    KeyStats,
    LimiterStats,
    RateLimitDecision,
    RateLimitResult,
    RateLimiter,
    SlidingWindowPolicy,
    TokenBucketPolicy,
    sliding_window_limiter,
    token_bucket_limiter,
)


class FakeClock:
    """Deterministic monotonic clock in milliseconds."""

    def __init__(self) -> None:
        self.now = 0.0
        self._lock = threading.Lock()

    def __call__(self) -> float:
        with self._lock:
            return self.now

    def advance(self, ms: float) -> None:
        with self._lock:
            self.now += ms


class SlidingWindowTests(unittest.TestCase):
    def setUp(self) -> None:
        self.clock = FakeClock()
        self.limiter = RateLimiter.sliding_window(3, 1000, clock=self.clock)

    def test_rejects_after_limit(self) -> None:
        for _ in range(3):
            result = self.limiter.check("alice", "chat")
            self.assertTrue(result.allowed)
        self.assertEqual(result.remaining, 0)

        rejected = self.limiter.check("alice", "chat")
        self.assertFalse(rejected.allowed)
        self.assertEqual(rejected.remaining, 0)
        self.assertAlmostEqual(rejected.retry_after_ms, 1000.0)
        self.assertEqual(rejected.strategy, "sliding_window")
        self.assertEqual(rejected.limit, 3)

    def test_recovery_is_exact_at_window_boundary(self) -> None:
        for _ in range(3):
            self.limiter.check("alice", "chat")
        self.clock.advance(999.999)
        self.assertFalse(self.limiter.check("alice", "chat").allowed)
        self.clock.advance(0.001)
        self.assertTrue(self.limiter.check("alice", "chat").allowed)

    def test_partial_recovery_uses_oldest_event(self) -> None:
        limiter = RateLimiter.sliding_window(2, 1000, clock=self.clock)
        self.assertTrue(limiter.check("alice", "chat").allowed)  # t=0
        self.clock.advance(400)
        self.assertTrue(limiter.check("alice", "chat").allowed)  # t=400
        self.clock.advance(300)  # t=700
        result = limiter.check("alice", "chat")
        self.assertFalse(result.allowed)
        self.assertAlmostEqual(result.retry_after_ms, 300.0)

        self.clock.advance(300)  # t=1000, event from t=0 expires
        self.assertTrue(limiter.check("alice", "chat").allowed)

    def test_rejected_request_does_not_consume_capacity(self) -> None:
        self.limiter.check("alice", "chat")
        self.limiter.check("alice", "chat")
        self.limiter.check("alice", "chat")
        for _ in range(5):
            self.limiter.check("alice", "chat")
        self.assertEqual(self.limiter.remaining("alice", "chat"), 0)
        self.clock.advance(1000)
        self.assertEqual(self.limiter.remaining("alice", "chat"), 3)

    def test_cost_greater_than_one(self) -> None:
        self.clock.advance(1)
        self.assertTrue(self.limiter.check("a", "r", cost=2).allowed)
        result = self.limiter.check("a", "r", cost=2)
        self.assertFalse(result.allowed)
        self.assertEqual(result.remaining, 1)
        self.assertAlmostEqual(result.retry_after_ms, 1000.0)

    def test_cost_can_be_satisfied_by_multiple_expiries(self) -> None:
        limiter = RateLimiter.sliding_window(3, 1000, clock=self.clock)
        limiter.check("a", "r")  # t=0
        self.clock.advance(100)
        limiter.check("a", "r")  # t=100
        self.clock.advance(100)
        limiter.check("a", "r")  # t=200
        self.clock.advance(300)  # t=500
        result = limiter.check("a", "r", cost=2)
        self.assertFalse(result.allowed)
        # Need two slots: the events at t=0 and t=100 must both expire.
        self.assertAlmostEqual(result.retry_after_ms, 600.0)

    def test_cost_above_limit_is_permanently_impossible(self) -> None:
        result = self.limiter.check("a", "r", cost=4)
        self.assertFalse(result.allowed)
        self.assertTrue(math.isinf(result.retry_after_ms))
        self.assertEqual(self.limiter.remaining("a", "r"), 3)

    def test_peek_does_not_consume(self) -> None:
        first = self.limiter.peek("a", "r")
        self.assertTrue(first.allowed)
        self.assertEqual(first.remaining, 3)
        self.assertEqual(self.limiter.remaining("a", "r"), 3)
        self.assertTrue(self.limiter.check("a", "r").allowed)
        self.assertEqual(self.limiter.remaining("a", "r"), 2)

    def test_isolation_between_clients_and_resources(self) -> None:
        self.limiter.check("alice", "chat")
        self.limiter.check("alice", "chat")
        self.limiter.check("alice", "chat")
        self.assertFalse(self.limiter.check("alice", "chat").allowed)

        self.assertTrue(self.limiter.check("bob", "chat").allowed)
        self.assertTrue(self.limiter.check("alice", "embed").allowed)
        self.assertEqual(self.limiter.remaining("bob", "chat"), 2)

    def test_stats(self) -> None:
        self.limiter.check("a", "r")
        self.limiter.check("a", "r")
        self.limiter.check("a", "r")
        self.limiter.check("a", "r")
        self.limiter.check("b", "r")

        stats = self.limiter.stats()
        self.assertIsInstance(stats, LimiterStats)
        self.assertEqual((stats.total, stats.allowed, stats.rejected), (5, 4, 1))
        key_stats = stats.keys[("a", "r")]
        self.assertIsInstance(key_stats, KeyStats)
        self.assertEqual(
            (key_stats.total, key_stats.allowed, key_stats.rejected), (4, 3, 1)
        )
        self.assertEqual(stats.as_dict()["total"], 5)

        self.limiter.reset_stats()
        self.assertEqual(self.limiter.stats().total, 0)
        # Resetting counters must not restore capacity.
        self.assertEqual(self.limiter.remaining("a", "r"), 0)

    def test_reset(self) -> None:
        self.limiter.check("a", "r")
        self.limiter.check("a", "r")
        self.limiter.check("a", "r")
        self.limiter.reset("a", "r")
        self.assertEqual(self.limiter.remaining("a", "r"), 3)

        self.limiter.check("a", "r")
        self.limiter.check("a", "r")
        self.limiter.check("a", "r")
        self.limiter.reset("a")
        self.assertEqual(self.limiter.remaining("a", "r"), 3)

        self.limiter.check("a", "r")
        self.limiter.reset()
        self.assertEqual(self.limiter.remaining("a", "r"), 3)


class TokenBucketTests(unittest.TestCase):
    def setUp(self) -> None:
        self.clock = FakeClock()
        self.limiter = RateLimiter.token_bucket(
            capacity=3, refill_per_second=1, clock=self.clock
        )

    def test_burst_then_reject(self) -> None:
        for expected_remaining in (2, 1, 0):
            result = self.limiter.check("alice", "chat")
            self.assertTrue(result.allowed)
            self.assertEqual(result.remaining, expected_remaining)
        rejected = self.limiter.check("alice", "chat")
        self.assertFalse(rejected.allowed)
        self.assertAlmostEqual(rejected.retry_after_ms, 1000.0)
        self.assertEqual(rejected.strategy, "token_bucket")

    def test_continuous_refill(self) -> None:
        for _ in range(3):
            self.limiter.check("a", "r")
        self.clock.advance(250)
        rejected = self.limiter.check("a", "r")
        self.assertFalse(rejected.allowed)
        self.assertAlmostEqual(rejected.retry_after_ms, 750.0)

        self.clock.advance(750)
        admitted = self.limiter.check("a", "r")
        self.assertTrue(admitted.allowed)
        self.assertEqual(admitted.remaining, 0)

    def test_fractional_refill_accumulates(self) -> None:
        limiter = RateLimiter.token_bucket(1, 2, clock=self.clock)  # 2 tokens/s
        self.assertTrue(limiter.check("a", "r").allowed)  # t=0
        self.clock.advance(300)
        rejected = limiter.check("a", "r")
        self.assertFalse(rejected.allowed)
        self.assertAlmostEqual(rejected.retry_after_ms, 200.0)

        self.clock.advance(200)  # t=500, exactly one token accumulated
        self.assertTrue(limiter.check("a", "r").allowed)
        self.clock.advance(150)
        rejected = limiter.check("a", "r")
        self.assertFalse(rejected.allowed)
        self.assertAlmostEqual(rejected.retry_after_ms, 350.0)

    def test_refill_is_capped_at_capacity(self) -> None:
        limiter = RateLimiter.token_bucket(2, 10, clock=self.clock)
        limiter.check("a", "r")
        self.clock.advance(60_000)
        self.assertEqual(limiter.peek("a", "r").remaining, 2)
        self.assertEqual(limiter.remaining("a", "r"), 2)

    def test_rejected_request_does_not_consume_tokens(self) -> None:
        limiter = RateLimiter.token_bucket(1, 1, clock=self.clock)
        limiter.check("a", "r")
        self.clock.advance(400)
        for _ in range(10):
            self.assertFalse(limiter.check("a", "r").allowed)
        self.clock.advance(600)
        # 0.4 + 0.6 = exactly one token available; rejections took nothing.
        self.assertTrue(limiter.check("a", "r").allowed)

    def test_cost_greater_than_one(self) -> None:
        result = self.limiter.check("a", "r", cost=2)
        self.assertTrue(result.allowed)
        self.assertEqual(result.remaining, 1)
        result = self.limiter.check("a", "r", cost=2)
        self.assertFalse(result.allowed)
        self.assertAlmostEqual(result.retry_after_ms, 1000.0)

    def test_cost_above_capacity_is_permanently_impossible(self) -> None:
        result = self.limiter.check("a", "r", cost=4)
        self.assertFalse(result.allowed)
        self.assertTrue(math.isinf(result.retry_after_ms))
        self.assertEqual(self.limiter.remaining("a", "r"), 3)

    def test_isolation_between_clients_and_resources(self) -> None:
        for _ in range(3):
            self.limiter.check("alice", "chat")
        self.assertFalse(self.limiter.check("alice", "chat").allowed)
        self.assertTrue(self.limiter.check("bob", "chat").allowed)
        self.assertTrue(self.limiter.check("alice", "embed").allowed)


class ConfigurationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.clock = FakeClock()

    def test_requires_a_policy(self) -> None:
        limiter = RateLimiter()
        with self.assertRaises(LookupError):
            limiter.check("a", "r")

    def test_per_key_override(self) -> None:
        limiter = RateLimiter.sliding_window(1, 1000, clock=self.clock)
        limiter.configure(
            "premium", "chat", TokenBucketPolicy(capacity=5, refill_per_second=1)
        )
        self.assertIsInstance(
            limiter.policy_for("premium", "chat"), TokenBucketPolicy
        )
        for _ in range(5):
            self.assertTrue(limiter.check("premium", "chat").allowed)
        self.assertFalse(limiter.check("premium", "chat").allowed)
        # Other keys keep the default policy.
        self.assertTrue(limiter.check("free", "chat").allowed)
        self.assertFalse(limiter.check("free", "chat").allowed)

    def test_override_resets_existing_state(self) -> None:
        limiter = RateLimiter.sliding_window(1, 1000, clock=self.clock)
        limiter.check("a", "r")
        self.assertFalse(limiter.check("a", "r").allowed)
        limiter.configure("a", "r", SlidingWindowPolicy(2, 1000))
        self.assertEqual(limiter.remaining("a", "r"), 2)

    def test_override_can_keep_state(self) -> None:
        limiter = RateLimiter.sliding_window(1, 1000, clock=self.clock)
        limiter.check("a", "r")
        limiter.configure(
            "a", "r", SlidingWindowPolicy(1, 2000), reset=False
        )
        # The recorded event is still inside the (now longer) window.
        self.assertFalse(limiter.check("a", "r").allowed)

    def test_factories(self) -> None:
        self.assertIsInstance(
            sliding_window_limiter(1, 100).default_policy, SlidingWindowPolicy
        )
        self.assertIsInstance(
            token_bucket_limiter(1, 1).default_policy, TokenBucketPolicy
        )

    def test_cleanup_evicts_idle_keys(self) -> None:
        limiter = RateLimiter.sliding_window(2, 1000, clock=self.clock)
        limiter.check("a", "r")
        self.clock.advance(2000)
        self.assertEqual(limiter.cleanup(500), 1)
        self.assertEqual(limiter.remaining("a", "r"), 2)

    def test_cleanup_keeps_window_with_live_events(self) -> None:
        limiter = RateLimiter.sliding_window(2, 10_000, clock=self.clock)
        limiter.check("a", "r")
        self.clock.advance(5000)
        self.assertEqual(limiter.cleanup(1000), 0)
        self.assertEqual(limiter.remaining("a", "r"), 1)

    def test_cleanup_pins_explicitly_configured_keys(self) -> None:
        limiter = RateLimiter.sliding_window(2, 1000, clock=self.clock)
        limiter.configure("vip", "r", SlidingWindowPolicy(2, 20_000))
        limiter.check("vip", "r")
        self.clock.advance(10_000)
        self.assertEqual(limiter.cleanup(0), 0)
        self.assertEqual(limiter.remaining("vip", "r"), 1)


class ValidationTests(unittest.TestCase):
    def test_policy_validation(self) -> None:
        with self.assertRaises(ValueError):
            SlidingWindowPolicy(0, 1000)
        with self.assertRaises(ValueError):
            SlidingWindowPolicy(1, 0)
        with self.assertRaises(ValueError):
            SlidingWindowPolicy(1, math.inf)
        with self.assertRaises(TypeError):
            SlidingWindowPolicy(1.5, 1000)  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            TokenBucketPolicy(0, 1)
        with self.assertRaises(ValueError):
            TokenBucketPolicy(1, -1)
        with self.assertRaises(ValueError):
            TokenBucketPolicy(1, math.nan)

    def test_cost_validation(self) -> None:
        limiter = RateLimiter.sliding_window(1, 1000)
        with self.assertRaises(ValueError):
            limiter.check("a", "r", cost=0)
        with self.assertRaises(ValueError):
            limiter.check("a", "r", cost=-1)
        with self.assertRaises(TypeError):
            limiter.check("a", "r", cost=1.5)  # type: ignore[arg-type]
        with self.assertRaises(TypeError):
            limiter.check("a", "r", cost=True)  # type: ignore[arg-type]

    def test_key_validation(self) -> None:
        limiter = RateLimiter.sliding_window(1, 1000)
        with self.assertRaises(TypeError):
            limiter.check(["unhashable"], "r")  # type: ignore[arg-type]

    def test_cleanup_validation(self) -> None:
        limiter = RateLimiter.sliding_window(1, 1000)
        with self.assertRaises(ValueError):
            limiter.cleanup(-1)
        with self.assertRaises(ValueError):
            limiter.cleanup(math.inf)


class ResultTests(unittest.TestCase):
    def test_truthiness_and_aliases(self) -> None:
        result = RateLimitResult(True, 2, 0.0, 3, "sliding_window")
        self.assertTrue(result)
        self.assertTrue(result.accepted)
        self.assertEqual(result.reset_after_ms, 0.0)
        self.assertEqual(result.as_dict()["remaining"], 2)
        self.assertIs(RateLimitDecision, RateLimitResult)
        self.assertIs(Decision, RateLimitResult)

    def test_boolean_shortcuts(self) -> None:
        limiter = RateLimiter.sliding_window(1, 1000)
        self.assertTrue(limiter.allow("a", "r"))
        self.assertFalse(limiter.allow_request("a", "r"))
        self.assertFalse(limiter.try_acquire("a", "r"))
        self.assertFalse(limiter.is_allowed("a", "r"))
        self.assertFalse(limiter.acquire("a", "r"))


class ConcurrencyTests(unittest.TestCase):
    def test_sliding_window_never_over_admits(self) -> None:
        limiter = RateLimiter.sliding_window(100, 60_000)
        threads = 16
        per_thread = 50
        barrier = threading.Barrier(threads)
        admitted = []
        lock = threading.Lock()

        def worker() -> None:
            barrier.wait()
            local = 0
            for _ in range(per_thread):
                if limiter.check("shared", "model"):
                    local += 1
            with lock:
                admitted.append(local)

        workers = [threading.Thread(target=worker) for _ in range(threads)]
        for worker_thread in workers:
            worker_thread.start()
        for worker_thread in workers:
            worker_thread.join()

        self.assertEqual(sum(admitted), 100)
        stats = limiter.stats()
        self.assertEqual(stats.total, threads * per_thread)
        self.assertEqual(stats.allowed, 100)
        self.assertEqual(stats.rejected, threads * per_thread - 100)

    def test_token_bucket_never_goes_negative(self) -> None:
        limiter = RateLimiter.token_bucket(50, 10, clock=FakeClock())
        threads = 12
        per_thread = 40
        barrier = threading.Barrier(threads)
        admitted = []
        lock = threading.Lock()

        def worker() -> None:
            barrier.wait()
            local = 0
            for _ in range(per_thread):
                if limiter.allow("shared", "model"):
                    local += 1
            with lock:
                admitted.append(local)

        workers = [threading.Thread(target=worker) for _ in range(threads)]
        for worker_thread in workers:
            worker_thread.start()
        for worker_thread in workers:
            worker_thread.join()

        self.assertEqual(sum(admitted), 50)
        self.assertEqual(limiter.remaining("shared", "model"), 0)

    def test_independent_keys_stay_consistent(self) -> None:
        limiter = RateLimiter.sliding_window(10, 60_000)
        errors: list[BaseException] = []

        def worker(client: int) -> None:
            try:
                outcomes = [
                    limiter.check(f"client-{client}", "model").allowed
                    for _ in range(20)
                ]
                if sum(outcomes) != 10:
                    raise AssertionError(f"client {client} got {sum(outcomes)}")
            except BaseException as exc:  # pragma: no cover - failure path
                errors.append(exc)

        workers = [threading.Thread(target=worker, args=(i,)) for i in range(8)]
        for thread in workers:
            thread.start()
        for thread in workers:
            thread.join()

        self.assertEqual(errors, [])
        for client in range(8):
            self.assertEqual(
                limiter.stats().keys[(f"client-{client}", "model")].allowed, 10
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
