"""Rate limiting middleware for API protection."""

import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Callable

from fastapi import Request, Response
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from app.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


@dataclass
class RateLimitState:
    """Track rate limit state for a client."""

    requests: list[float] = field(default_factory=list)
    blocked_until: float = 0.0


class RateLimiter:
    """In-memory rate limiter using sliding window algorithm."""

    def __init__(
        self,
        requests_per_minute: int = 60,
        block_duration_seconds: int = 60,
    ):
        """Initialize rate limiter.

        Args:
            requests_per_minute: Maximum requests allowed per minute
            block_duration_seconds: How long to block after limit exceeded
        """
        self.requests_per_minute = requests_per_minute
        self.block_duration = block_duration_seconds
        self.window_size = 60.0  # 1 minute window
        self._clients: dict[str, RateLimitState] = defaultdict(RateLimitState)

    def _cleanup_old_requests(self, state: RateLimitState, now: float) -> None:
        """Remove requests older than the window."""
        cutoff = now - self.window_size
        state.requests = [ts for ts in state.requests if ts > cutoff]

    def is_allowed(self, client_id: str) -> tuple[bool, dict]:
        """Check if a request from this client is allowed.

        Args:
            client_id: Unique client identifier (usually IP address)

        Returns:
            Tuple of (allowed, info_dict with rate limit headers)
        """
        now = time.time()
        state = self._clients[client_id]

        # Check if client is blocked
        if state.blocked_until > now:
            retry_after = int(state.blocked_until - now)
            return False, {
                "X-RateLimit-Limit": str(self.requests_per_minute),
                "X-RateLimit-Remaining": "0",
                "X-RateLimit-Reset": str(int(state.blocked_until)),
                "Retry-After": str(retry_after),
            }

        # Cleanup old requests
        self._cleanup_old_requests(state, now)

        # Check if limit exceeded
        if len(state.requests) >= self.requests_per_minute:
            state.blocked_until = now + self.block_duration
            logger.warning(f"Rate limit exceeded for client {client_id}")
            return False, {
                "X-RateLimit-Limit": str(self.requests_per_minute),
                "X-RateLimit-Remaining": "0",
                "X-RateLimit-Reset": str(int(state.blocked_until)),
                "Retry-After": str(self.block_duration),
            }

        # Record this request
        state.requests.append(now)
        remaining = self.requests_per_minute - len(state.requests)

        return True, {
            "X-RateLimit-Limit": str(self.requests_per_minute),
            "X-RateLimit-Remaining": str(remaining),
            "X-RateLimit-Reset": str(int(now + self.window_size)),
        }


# Global rate limiters
_general_limiter: RateLimiter | None = None
_auth_limiter: RateLimiter | None = None


def get_general_limiter() -> RateLimiter:
    """Get or create the general rate limiter."""
    global _general_limiter
    if _general_limiter is None:
        _general_limiter = RateLimiter(
            requests_per_minute=settings.rate_limit_requests_per_minute,
            block_duration_seconds=60,
        )
    return _general_limiter


def get_auth_limiter() -> RateLimiter:
    """Get or create the auth rate limiter (stricter)."""
    global _auth_limiter
    if _auth_limiter is None:
        _auth_limiter = RateLimiter(
            requests_per_minute=settings.rate_limit_auth_requests_per_minute,
            block_duration_seconds=300,  # 5 minute block for auth abuse
        )
    return _auth_limiter


def get_client_ip(request: Request) -> str:
    """Extract client IP from request, handling proxies."""
    # Check X-Forwarded-For header (set by reverse proxies)
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        # Take the first IP (original client)
        return forwarded.split(",")[0].strip()

    # Check X-Real-IP header (nginx)
    real_ip = request.headers.get("X-Real-IP")
    if real_ip:
        return real_ip

    # Fall back to direct client IP
    if request.client:
        return request.client.host

    return "unknown"


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Middleware that applies rate limiting to all requests."""

    # Paths that use stricter auth rate limiting
    AUTH_PATHS = frozenset({
        "/api/v1/auth/login",
        "/api/v1/auth/register",
        "/api/v1/auth/refresh",
        "/api/v1/auth/change-password",
    })

    # Paths exempt from rate limiting
    EXEMPT_PATHS = frozenset({
        "/",
        "/docs",
        "/redoc",
        "/openapi.json",
        "/api/v1/health",
    })

    async def dispatch(
        self, request: Request, call_next: Callable
    ) -> Response:
        """Process request with rate limiting."""
        # Skip if rate limiting is disabled
        if not settings.rate_limit_enabled:
            return await call_next(request)

        path = request.url.path

        # Skip exempt paths
        if path in self.EXEMPT_PATHS:
            return await call_next(request)

        client_ip = get_client_ip(request)

        # Choose appropriate limiter
        if path in self.AUTH_PATHS:
            limiter = get_auth_limiter()
        else:
            limiter = get_general_limiter()

        # Check rate limit
        allowed, headers = limiter.is_allowed(client_ip)

        if not allowed:
            return JSONResponse(
                status_code=429,
                content={
                    "detail": {
                        "code": "RATE_LIMIT_EXCEEDED",
                        "message": "Too many requests. Please try again later.",
                    }
                },
                headers=headers,
            )

        # Process request and add rate limit headers to response
        response = await call_next(request)

        # Add rate limit headers to response
        for key, value in headers.items():
            response.headers[key] = value

        return response
