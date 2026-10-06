"""Tests for the in-memory rate limiter.

The suite runs with RATE_LIMIT_ENABLED=false (see conftest), so these tests
exercise RateLimiter and get_client_ip directly, plus the middleware on a
small app of its own.
"""

from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from starlette.requests import Request
from uvicorn.middleware.proxy_headers import ProxyHeadersMiddleware

from app.config import Settings
from app.middleware import rate_limit
from app.middleware.rate_limit import RateLimiter, RateLimitMiddleware, get_client_ip


@pytest.fixture
def clock(monkeypatch):
    """Freeze the limiter's clock; advance it with clock.now += seconds."""
    fake = SimpleNamespace(now=1_000_000.0)
    monkeypatch.setattr(rate_limit, "time", SimpleNamespace(time=lambda: fake.now))
    return fake


def make_request(headers: dict[str, str] | None = None, client=("203.0.113.7", 5555)):
    scope = {
        "type": "http",
        "method": "POST",
        "path": "/api/v1/auth/login",
        "headers": [
            (k.lower().encode("latin1"), v.encode("latin1"))
            for k, v in (headers or {}).items()
        ],
        "client": client,
    }
    return Request(scope)


class TestRateLimiter:
    def test_allows_up_to_limit_then_blocks(self, clock):
        limiter = RateLimiter(requests_per_minute=3, block_duration_seconds=300)

        remaining = []
        for _ in range(3):
            allowed, headers = limiter.is_allowed("1.2.3.4")
            assert allowed is True
            remaining.append(headers["X-RateLimit-Remaining"])
        assert remaining == ["2", "1", "0"]

        allowed, headers = limiter.is_allowed("1.2.3.4")
        assert allowed is False
        assert headers["X-RateLimit-Remaining"] == "0"
        assert headers["Retry-After"] == "300"

    def test_clients_have_separate_buckets(self, clock):
        limiter = RateLimiter(requests_per_minute=1, block_duration_seconds=60)
        assert limiter.is_allowed("1.1.1.1")[0] is True
        assert limiter.is_allowed("1.1.1.1")[0] is False
        assert limiter.is_allowed("2.2.2.2")[0] is True

    def test_block_lasts_for_block_duration(self, clock):
        limiter = RateLimiter(requests_per_minute=1, block_duration_seconds=300)
        limiter.is_allowed("1.2.3.4")
        assert limiter.is_allowed("1.2.3.4")[0] is False

        # The window has passed but the block has not.
        clock.now += 120
        allowed, headers = limiter.is_allowed("1.2.3.4")
        assert allowed is False
        assert headers["Retry-After"] == "180"

        clock.now += 181
        assert limiter.is_allowed("1.2.3.4")[0] is True

    def test_window_slides(self, clock):
        limiter = RateLimiter(requests_per_minute=2, block_duration_seconds=60)
        assert limiter.is_allowed("1.2.3.4")[0] is True
        clock.now += 30
        assert limiter.is_allowed("1.2.3.4")[0] is True
        # The first request leaves the 60s window, freeing one slot.
        clock.now += 31
        allowed, headers = limiter.is_allowed("1.2.3.4")
        assert allowed is True
        assert headers["X-RateLimit-Remaining"] == "0"


class TestClientIp:
    def test_uses_peer_address(self):
        assert get_client_ip(make_request()) == "203.0.113.7"

    @pytest.mark.parametrize(
        "headers",
        [
            {"X-Forwarded-For": "10.9.9.1"},
            {"X-Forwarded-For": "10.9.9.1, 198.51.100.4"},
            {"X-Real-IP": "10.9.9.2"},
            {"X-Forwarded-For": "10.9.9.1", "X-Real-IP": "10.9.9.2"},
        ],
    )
    def test_spoofed_forwarding_headers_ignored(self, headers):
        assert get_client_ip(make_request(headers)) == "203.0.113.7"

    def test_no_client(self):
        assert get_client_ip(make_request(client=None)) == "unknown"

    async def test_proxy_rewrite_only_for_trusted_peer(self):
        """Behind uvicorn's proxy-headers support, only a trusted proxy's
        X-Forwarded-For changes the key (FORWARDED_ALLOW_IPS)."""
        seen = []

        async def app(scope, receive, send):
            seen.append(get_client_ip(Request(scope)))

        proxied = ProxyHeadersMiddleware(app, trusted_hosts="10.0.0.2")

        async def call(client, xff):
            scope = make_request({"X-Forwarded-For": xff}, client=client).scope
            await proxied(scope, None, None)

        # Untrusted peer: its forged header is ignored.
        await call(("203.0.113.7", 5555), "10.9.9.1")
        # Trusted proxy that sets X-Forwarded-For to the real peer address.
        await call(("10.0.0.2", 5555), "198.51.100.4")
        assert seen == ["203.0.113.7", "198.51.100.4"]


class TestMiddleware:
    @pytest.fixture
    def limited_client(self, monkeypatch):
        monkeypatch.setattr(rate_limit.settings, "rate_limit_enabled", True)
        monkeypatch.setattr(
            rate_limit,
            "_auth_limiter",
            RateLimiter(requests_per_minute=3, block_duration_seconds=300),
        )

        app = FastAPI()
        app.add_middleware(RateLimitMiddleware)

        @app.post("/api/v1/auth/login")
        async def login():
            return {"ok": True}

        return TestClient(app)

    def test_rotating_forwarded_for_does_not_bypass_auth_limit(
        self, limited_client, clock
    ):
        statuses = [
            limited_client.post(
                "/api/v1/auth/login", headers={"X-Forwarded-For": f"10.8.8.{i}"}
            ).status_code
            for i in range(5)
        ]
        assert statuses == [200, 200, 200, 429, 429]

        blocked = limited_client.post("/api/v1/auth/login")
        assert blocked.status_code == 429
        assert blocked.json() == {
            "detail": {
                "code": "RATE_LIMIT_EXCEEDED",
                "message": "Too many requests. Please try again later.",
            }
        }
        assert blocked.headers["Retry-After"] == "300"


class TestDefaultLimits:
    """The shipped defaults must not throttle ordinary UI traffic."""

    @pytest.fixture
    def default_settings(self, monkeypatch):
        for name in (
            "RATE_LIMIT_ENABLED",
            "RATE_LIMIT_REQUESTS_PER_MINUTE",
            "RATE_LIMIT_AUTH_REQUESTS_PER_MINUTE",
        ):
            monkeypatch.delenv(name, raising=False)
        return Settings(_env_file=None, environment="test")

    def test_default_values(self, default_settings):
        assert default_settings.rate_limit_enabled is True
        assert default_settings.rate_limit_requests_per_minute == 300
        assert default_settings.rate_limit_auth_requests_per_minute == 10

    @pytest.fixture
    def default_limited_client(self, monkeypatch, default_settings):
        """The real middleware with limiters built from the default settings."""
        monkeypatch.setattr(rate_limit, "settings", default_settings)
        monkeypatch.setattr(rate_limit, "_general_limiter", None)
        monkeypatch.setattr(rate_limit, "_auth_limiter", None)

        app = FastAPI()
        app.add_middleware(RateLimitMiddleware)

        @app.post("/api/v1/learning/flashcards/review")
        async def review():
            return {"ok": True}

        @app.post("/api/v1/auth/login")
        async def login():
            return {"ok": True}

        @app.get("/api/v1/{path:path}")
        async def read(path: str):
            return {"ok": True}

        return TestClient(app)

    def test_fifty_card_review_session_not_throttled(
        self, default_limited_client, clock
    ):
        """A reader visit with a minute of status polling, then a 50-card
        review, all within one minute. With the old 60/min default the live
        run got a 429 at request 61."""
        client = default_limited_client
        statuses = []

        def get(*paths):
            for path in paths:
                statuses.append(client.get(f"/api/v1/{path}").status_code)

        # Reader: document, content, graph nodes, relations, profile.
        get(
            "documents/d",
            "documents/d/content",
            "graph/nodes",
            "graph/documents/d/relations",
            "learning/engagement/profile",
        )
        # The reader polls the document status every 3 s.
        get(*["documents/d"] * 20)
        # Flashcards page: due cards, stats, profile.
        page_loads = (
            "learning/flashcards/due",
            "learning/flashcards/stats",
            "learning/engagement/profile",
        )
        get(*page_loads)
        # One POST per reviewed card.
        for _ in range(50):
            statuses.append(
                client.post("/api/v1/learning/flashcards/review").status_code
            )
            clock.now += 0.5
        # "Done" reloads the page data.
        get(*page_loads)

        assert len(statuses) == 81
        assert statuses == [200] * 81

    def test_auth_limit_still_ten_per_minute(self, default_limited_client, clock):
        statuses = [
            default_limited_client.post("/api/v1/auth/login").status_code
            for _ in range(11)
        ]
        assert statuses == [200] * 10 + [429]
