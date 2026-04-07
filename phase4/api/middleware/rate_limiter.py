"""
Rate Limiter — Phase 4

Simple in-memory sliding window rate limiter.
No Redis required for Phase 4 — per-process limits are fine for single-instance.

Limits:
  - Global: RATE_LIMIT_RPM requests per minute per IP
  - Chat:   RATE_LIMIT_CHAT chat turns per minute per session

Usage (FastAPI dependency):
    @router.post("/chat")
    async def chat(request: Request, _=Depends(rate_limit_chat)):
        ...
"""

from __future__ import annotations

import time
from collections import defaultdict, deque
from typing import Optional

from fastapi import HTTPException, Request, status

from phase4.api.core.config import settings


class SlidingWindowRateLimiter:
    """
    Per-key sliding window rate limiter.
    Thread-safe via GIL (single-process, async-safe for FastAPI).

    Args:
        max_requests: Maximum requests allowed in the window
        window_secs:  Window duration in seconds
    """

    def __init__(self, max_requests: int, window_secs: int = 60) -> None:
        self.max_requests = max_requests
        self.window_secs  = window_secs
        self._windows: dict[str, deque] = defaultdict(deque)

    def is_allowed(self, key: str) -> tuple[bool, int]:
        """
        Check if a request from `key` is within limits.

        Returns:
            (allowed, retry_after_seconds)
            retry_after_seconds is 0 if allowed, else seconds until window resets
        """
        now = time.monotonic()
        cutoff = now - self.window_secs
        window = self._windows[key]

        # Remove timestamps outside the window
        while window and window[0] < cutoff:
            window.popleft()

        if len(window) >= self.max_requests:
            retry_after = int(self.window_secs - (now - window[0])) + 1
            return False, retry_after

        window.append(now)
        return True, 0

    def reset(self, key: str) -> None:
        """Clear the rate limit window for a specific key."""
        self._windows.pop(key, None)

    def get_remaining(self, key: str) -> int:
        """Return remaining requests in the current window."""
        now = time.monotonic()
        cutoff = now - self.window_secs
        window = self._windows[key]
        while window and window[0] < cutoff:
            window.popleft()
        return max(0, self.max_requests - len(window))


# ── Shared limiters ───────────────────────────────────────────────────────────

_global_limiter = SlidingWindowRateLimiter(
    max_requests=settings.RATE_LIMIT_RPM,
    window_secs=60,
)

_chat_limiter = SlidingWindowRateLimiter(
    max_requests=settings.RATE_LIMIT_CHAT,
    window_secs=60,
)


def _get_client_ip(request: Request) -> str:
    """Extract client IP, respecting X-Forwarded-For for proxies."""
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


async def rate_limit_global(request: Request) -> None:
    """
    FastAPI dependency — enforce global RPM limit per IP.
    Attach to any router that needs protection.
    """
    ip = _get_client_ip(request)
    allowed, retry_after = _global_limiter.is_allowed(ip)
    if not allowed:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Rate limit exceeded. Retry after {retry_after} seconds.",
            headers={"Retry-After": str(retry_after)},
        )


async def rate_limit_chat(request: Request) -> None:
    """
    FastAPI dependency — enforce chat-specific rate limit per session.
    Uses session_id from request body if available, falls back to IP.
    """
    # Try to get session_id from request body
    # FastAPI doesn't expose body in dependencies easily, so use IP as key here
    ip = _get_client_ip(request)
    allowed, retry_after = _chat_limiter.is_allowed(f"chat:{ip}")
    if not allowed:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Chat rate limit exceeded. Retry after {retry_after} seconds.",
            headers={"Retry-After": str(retry_after)},
        )