"""Small end-to-end demo of the LLM gateway rate limiter.

Run with:  python demo.py
"""

from __future__ import annotations

from rate_limiter import RateLimiter, SlidingWindowPolicy, TokenBucketPolicy


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def advance(self, ms: float) -> None:
        self.now += ms


def main() -> None:
    clock = FakeClock()
    limiter = RateLimiter(
        SlidingWindowPolicy(limit=3, window_ms=10_000), clock=clock
    )
    limiter.configure(
        "premium-user",
        "gpt-4o",
        TokenBucketPolicy(capacity=5, refill_per_second=1),
    )

    print("free client, 3 requests / 10s")
    for i in range(1, 6):
        result = limiter.check("free-user", "gpt-4o")
        print(
            f"  req {i}: allowed={result.allowed} "
            f"remaining={result.remaining} retry={result.retry_after_ms:.0f}ms"
        )

    clock.advance(10_000)
    print("\nafter 10s the whole window is clear:")
    print(" ", limiter.check("free-user", "gpt-4o").as_dict())

    print("\npremium client, bucket of 5 refilling 1/s:")
    for i in range(1, 8):
        result = limiter.check("premium-user", "gpt-4o")
        print(
            f"  req {i}: allowed={result.allowed} "
            f"remaining={result.remaining} retry={result.retry_after_ms:.0f}ms"
        )

    print("\nstats:", limiter.stats_dict()["total"], "requests checked")
    print("free-user capacity left:", limiter.remaining("free-user", "gpt-4o"))


if __name__ == "__main__":
    main()
