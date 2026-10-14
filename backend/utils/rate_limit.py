"""In-process fixed-window rate limiting.

Sufficient for a locally-served single-process app. A multi-worker deployment
would need shared state (Redis) instead - documented in the README.
"""

from __future__ import annotations

import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Deque, Dict, Tuple


@dataclass
class RateLimiter:
    limit: int
    window_seconds: float
    _hits: Dict[Tuple[str, str], Deque[float]] = field(default_factory=lambda: defaultdict(deque))
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def check(self, bucket: str, key: str) -> Tuple[bool, int]:
        """Return (allowed, seconds_until_retry).

        `allowed` is False once the caller has exceeded `limit` attempts inside
        the window. The counter resets automatically as old entries age out, so
        there is no cleanup thread.
        """
        now = time.monotonic()
        k = (bucket, key)

        with self._lock:
            hits = self._hits[k]
            cutoff = now - self.window_seconds
            while hits and hits[0] < cutoff:
                hits.popleft()

            if len(hits) >= self.limit:
                retry_after = max(1, int(hits[0] + self.window_seconds - now))
                return False, retry_after

            hits.append(now)
            return True, 0

    def reset(self, bucket: str, key: str) -> None:
        with self._lock:
            self._hits.pop((bucket, key), None)


# Login is the only endpoint worth brute forcing: everything else needs a
# valid token. Five failures per 15 minutes per (email, client) pair.
login_limiter = RateLimiter(limit=5, window_seconds=15 * 60)
# Registration is cheap but hashes cost CPU; keep a modest ceiling.
register_limiter = RateLimiter(limit=10, window_seconds=60 * 60)


# Trusted proxy depth. X-Forwarded-For is only meaningful when a proxy you
# control sets it; trusting it unconditionally lets any caller forge a new
# identity per request and bypass the limit entirely.
TRUSTED_PROXY_COUNT = 0


def client_key(request) -> str:
    """Identity used for rate limiting.

    X-Forwarded-For is ignored unless TRUSTED_PROXY_COUNT is raised to match
    the number of proxies actually in front of the app. At the default of 0
    the header is attacker-controlled, so keying on it would let a brute-force
    attempt reset the counter on every request.
    """
    if TRUSTED_PROXY_COUNT:
        forwarded = request.headers.get("x-forwarded-for")
        if forwarded:
            parts = [p.strip() for p in forwarded.split(",") if p.strip()]
            if len(parts) > TRUSTED_PROXY_COUNT:
                return parts[-TRUSTED_PROXY_COUNT - 1]
    return request.client.host if request.client else "unknown"