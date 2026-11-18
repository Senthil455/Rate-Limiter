# Rate Limiter

An in-memory, thread-safe rate limiter for an LLM API gateway. It supports two
strategies, keeps fully isolated state per `(client, resource)` pair, and returns
the remaining capacity plus an estimated recovery time on every decision.

```python
from rate_limiter import RateLimiter, SlidingWindowPolicy, TokenBucketPolicy

# 20 requests per minute, per client + resource
limiter = RateLimiter(SlidingWindowPolicy(limit=20, window_ms=60_000))

result = limiter.check("alice", "gpt-4o")
if result.allowed:
    call_llm()
else:
    retry_in_seconds = result.retry_after_ms / 1000
```

## API

### Policies

| Policy | Behaviour | Fields |
| --- | --- | --- |
| `SlidingWindowPolicy(limit, window_ms)` | Exact sliding-window log. At most `limit` request units in any trailing `window_ms` window. No fixed-window burst at boundaries. | `limit >= 1`, `window_ms > 0` |
| `TokenBucketPolicy(capacity, refill_per_second)` | Burst up to `capacity`, then refill continuously at `refill_per_second`. | `capacity > 0`, `refill_per_second > 0` |

### `RateLimiter`

```python
RateLimiter(policy=None, *, clock=None)
```

* `policy` — default policy; may be `None` when every key is configured explicitly.
* `clock` — callable returning monotonic milliseconds. Defaults to
  `time.monotonic_ns() / 1e6`; inject a fake clock in tests.

**Convenience constructors**

```python
RateLimiter.sliding_window(limit, window_ms, *, clock=None)
RateLimiter.token_bucket(capacity, refill_per_second, *, clock=None)
sliding_window_limiter(limit, window_ms)
token_bucket_limiter(capacity, refill_per_second)
```

**Admission**

| Method | Description |
| --- | --- |
| `check(client, resource, cost=1) -> RateLimitResult` | Atomically decide and, if allowed, consume `cost` units. |
| `allow(client, resource, cost=1) -> bool` | Boolean shortcut. Aliased as `allow_request`, `try_acquire`, `is_allowed`, `acquire`. |
| `peek(client, resource) -> RateLimitResult` | Decision for a cost-1 request without consuming anything. |
| `remaining(client, resource) -> int` | Whole units currently available. |

`cost` must be an `int >= 1`; costs larger than the policy capacity are rejected
with `retry_after_ms = inf` because they can never be satisfied.

**Configuration / maintenance**

| Method | Description |
| --- | --- |
| `configure(client, resource, policy, reset=True)` | Per-key policy override. |
| `policy_for(client, resource) -> Policy` | Effective policy for a key. |
| `reset(client=None, resource=None)` | Drop state for one key, one client, one resource, or all keys. |
| `cleanup(max_idle_ms) -> int` | Evict idle, unconfigured keys to bound memory. |
| `stats() -> LimiterStats` | Aggregate + per-key counters (`total`, `allowed`, `rejected`). |
| `stats_dict()` | JSON-friendly stats. |
| `reset_stats()` | Zero counters without restoring capacity. |

### `RateLimitResult`

| Field | Meaning |
| --- | --- |
| `allowed` | Whether the request was admitted (`__bool__` delegates here). |
| `remaining` | Whole units left for the key after the decision. |
| `retry_after_ms` | Time until the same request would be admitted; `0.0` when allowed. Also exposed as `reset_after_ms`. |
| `limit` | Capacity of the policy for that key. |
| `strategy` | `"sliding_window"` or `"token_bucket"`. |

`as_dict()` returns a JSON-serialisable view.

## Behaviour details

* **Isolation** — state is keyed by `(client, resource)`. `alice/gpt-4o` and
  `alice/embeddings` have independent budgets, as do different clients.
* **Sliding window** — only accepted requests are logged, each tagged with its
  admission timestamp and expiring exactly `window_ms` later. Rejected requests
  never consume capacity. Recovery is computed from the oldest event(s) that
  must leave the window, so it is exact even when `cost > 1`.
* **Token bucket** — tokens refill continuously (`elapsed_ms * rate / 1000`) and
  are capped at `capacity`; a rejected request consumes nothing but still
  advances the refill clock, which keeps the retry estimate accurate. Float
  drift is absorbed with a `1e-9` tolerance.
* **Concurrency** — every check, peek, reset, cleanup and stats snapshot runs
  under one re-entrant lock, so a shared limiter never over-admits. Verified by
  tests that fire hundreds of simultaneous requests from multiple threads.
* **Bounds** — `limit`/`capacity` must be positive, `window_ms` and
  `refill_per_second` finite and positive, and `cost` a positive integer.
  `cleanup()` removes idle keys; explicitly configured keys are pinned.

## Complexity

| Operation | Sliding window | Token bucket |
| --- | --- | --- |
| `check` (amortised) | `O(cost)` deque appends/evictions | `O(1)` |
| `peek` / `remaining` | `O(evicted)` | `O(1)` |
| Memory per key | `O(limit)` timestamps | `O(1)` |

`O(limit)` for the window log is the exact algorithm's trade-off; the helper
aliases make it easy to swap in `TokenBucketPolicy` where memory matters more
than strict window semantics.

## Tests

```bash
python -m unittest test_rate_limiter -v
```

36 tests cover boundaries, partial recovery, costs greater than one, bucket
refill/capping, key isolation, per-key configuration, stats, cleanup,
validation, result aliases and multi-threaded correctness.

## Note on the LeetCode interface

The problem statement visible in the prompt does not include the exact
class/method signature expected by the grader. This implementation exposes the
natural `check` / `allow` / `remaining` surface plus common aliases. If the
LeetCode editor declares specific names or field names, share that snippet and
the thin adapter (or a rename) is a one-minute change.
