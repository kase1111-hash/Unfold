"""Tests for the production checks and env parsing in Settings."""

import re
from pathlib import Path

import pytest

from app.config import ConfigurationError, Settings

DB_URL = "postgresql://unfold:db-pass@postgres:5432/unfold"
STRONG_JWT = "k3Y-" * 12  # 48 characters, no placeholder words
STRONG_NEO4J = "Neo4j-Str0ng-Pass"

# Values shipped in the example env files.
PLACEHOLDER_JWTS = [
    "change-me-in-production-use-a-secure-random-string",
    "CHANGE_ME_GENERATE_SECURE_RANDOM_STRING",
    "please-ChangeMe-before-deploying-this-app",
]
BAD_NEO4J_PASSWORDS = [
    "",
    "password",
    "changeme",
    "CHANGE_ME_SECURE_PASSWORD_HERE",
]


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    """Settings also reads the process env (the suite sets several vars)."""
    for name in (
        "ENVIRONMENT",
        "JWT_SECRET",
        "DATABASE_URL",
        "NEO4J_USER",
        "NEO4J_PASSWORD",
        "CORS_ORIGINS",
    ):
        monkeypatch.delenv(name, raising=False)


def make_settings(**overrides) -> Settings:
    values = {
        "environment": "production",
        "database_url": DB_URL,
        "jwt_secret": STRONG_JWT,
        "neo4j_password": STRONG_NEO4J,
    }
    values.update(overrides)
    return Settings(_env_file=None, **values)


class TestProductionSecrets:
    @pytest.mark.parametrize("environment", ["production", "staging"])
    def test_real_secrets_accepted(self, environment):
        settings = make_settings(environment=environment)
        assert settings.jwt_secret == STRONG_JWT
        assert settings.neo4j_password == STRONG_NEO4J

    @pytest.mark.parametrize("environment", ["production", "staging"])
    @pytest.mark.parametrize("jwt_secret", PLACEHOLDER_JWTS)
    def test_placeholder_jwt_secret_rejected(self, environment, jwt_secret):
        with pytest.raises(ConfigurationError, match="JWT_SECRET is still a placeholder"):
            make_settings(environment=environment, jwt_secret=jwt_secret)

    @pytest.mark.parametrize("environment", ["production", "staging"])
    @pytest.mark.parametrize("password", BAD_NEO4J_PASSWORDS)
    def test_default_or_placeholder_neo4j_password_rejected(
        self, environment, password
    ):
        with pytest.raises(ConfigurationError, match="NEO4J_PASSWORD must be changed"):
            make_settings(environment=environment, neo4j_password=password)

    def test_missing_neo4j_password_rejected(self):
        with pytest.raises(ConfigurationError, match="NEO4J_PASSWORD environment variable"):
            make_settings(neo4j_password=None)

    def test_missing_jwt_secret_rejected(self):
        with pytest.raises(ConfigurationError, match="JWT_SECRET environment variable"):
            make_settings(jwt_secret=None)

    def test_short_jwt_secret_rejected(self):
        with pytest.raises(ConfigurationError, match="at least 32 characters"):
            make_settings(jwt_secret="x" * 31)

    def test_placeholders_from_environment_variables_rejected(self, monkeypatch):
        """The same checks apply when values come from the process env."""
        monkeypatch.setenv("ENVIRONMENT", "production")
        monkeypatch.setenv("DATABASE_URL", DB_URL)
        monkeypatch.setenv("NEO4J_PASSWORD", STRONG_NEO4J)
        monkeypatch.setenv("JWT_SECRET", "CHANGE_ME_GENERATE_SECURE_RANDOM_STRING")
        with pytest.raises(ConfigurationError, match="JWT_SECRET is still a placeholder"):
            Settings(_env_file=None)


class TestNonProductionDefaults:
    @pytest.mark.parametrize("environment", ["development", "test"])
    def test_placeholders_allowed(self, environment):
        settings = make_settings(
            environment=environment,
            jwt_secret=PLACEHOLDER_JWTS[0],
            neo4j_password="changeme",
        )
        assert settings.jwt_secret == PLACEHOLDER_JWTS[0]
        assert settings.neo4j_password == "changeme"

    @pytest.mark.parametrize("environment", ["development", "test"])
    def test_unset_values_get_dev_defaults(self, environment):
        settings = Settings(_env_file=None, environment=environment)
        assert settings.neo4j_password == "changeme"
        assert len(settings.jwt_secret) >= 32
        assert str(settings.database_url) == (
            "postgresql://postgres:postgres@localhost:5432/unfold"
        )


# The repository root (backend/tests/unit/test_config.py -> repo root).
REPO_ROOT = Path(__file__).resolve().parents[3]


def _env_file_value(path: Path, key: str) -> str:
    """The value assigned to ``key`` in a KEY=value env file."""
    for line in path.read_text().splitlines():
        if line.startswith(f"{key}="):
            return line.split("=", 1)[1]
    raise AssertionError(f"{key} not found in {path}")


def _compose_default(path: Path, key: str) -> str:
    """The ``${KEY:-default}`` default used in a compose file."""
    match = re.search(r"\$\{" + key + r":-([^}]*)\}", path.read_text())
    assert match, f"no ${{{key}:-...}} default in {path}"
    return match.group(1)


def _assert_origin_list(origins, raw: str) -> None:
    assert isinstance(origins, list)
    assert origins, "expected at least one origin"
    for origin in origins:
        assert isinstance(origin, str)
        assert origin.startswith(("http://", "https://"))
        assert "," not in origin and origin == origin.strip()
    assert ",".join(origins) == raw.replace(" ", "")


class TestCorsOrigins:
    """CORS_ORIGINS must load in every form the docs and examples use.

    pydantic-settings 2.1 JSON-decodes list fields from the environment
    before validators run, so anything but a JSON list used to raise
    SettingsError and stop the app (and alembic) from starting.
    """

    @pytest.mark.parametrize(
        "raw, expected",
        [
            ("http://a.example,http://b.example", ["http://a.example", "http://b.example"]),
            ("a,b", ["a", "b"]),
            ("http://localhost:3000", ["http://localhost:3000"]),
            ('["http://a.example", "http://b.example"]', ["http://a.example", "http://b.example"]),
            (" http://a.example , ,http://b.example, ", ["http://a.example", "http://b.example"]),
        ],
    )
    def test_env_value_forms(self, monkeypatch, raw, expected):
        monkeypatch.setenv("CORS_ORIGINS", raw)
        settings = Settings(_env_file=None, environment="development")
        assert settings.cors_origins == expected

    def test_default(self):
        settings = Settings(_env_file=None, environment="development")
        assert settings.cors_origins == ["http://localhost:3000"]

    def test_list_passed_directly(self):
        settings = Settings(
            _env_file=None, environment="development", cors_origins=["http://x.example"]
        )
        assert settings.cors_origins == ["http://x.example"]

    @pytest.mark.parametrize(
        "source",
        [
            pytest.param(
                lambda: _env_file_value(REPO_ROOT / ".env.production.example", "CORS_ORIGINS"),
                id=".env.production.example",
            ),
            pytest.param(
                lambda: _compose_default(REPO_ROOT / "docker-compose.prod.yml", "CORS_ORIGINS"),
                id="docker-compose.prod.yml default",
            ),
            pytest.param(
                lambda: _env_file_value(REPO_ROOT / "backend" / ".env.example", "CORS_ORIGINS"),
                id="backend/.env.example",
            ),
        ],
    )
    def test_shipped_values_load_in_production(self, monkeypatch, source):
        """The prod compose stack passes these through the environment."""
        raw = source()
        monkeypatch.setenv("CORS_ORIGINS", raw)
        settings = make_settings()
        _assert_origin_list(settings.cors_origins, raw)

    def test_backend_env_example_loads_as_dotenv(self):
        """``cp backend/.env.example backend/.env`` must give working settings."""
        env_file = REPO_ROOT / "backend" / ".env.example"
        settings = Settings(_env_file=env_file)
        _assert_origin_list(
            settings.cors_origins, _env_file_value(env_file, "CORS_ORIGINS")
        )
