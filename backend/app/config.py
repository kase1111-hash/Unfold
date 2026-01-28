"""Application configuration management."""

import logging
import secrets
from functools import lru_cache
from typing import Literal

from pydantic import Field, PostgresDsn, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger(__name__)


class ConfigurationError(Exception):
    """Raised when configuration is invalid for the environment."""

    pass


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Application
    app_name: str = "Unfold API"
    app_version: str = "0.1.0"
    debug: bool = False
    environment: Literal["development", "staging", "production"] = "development"

    # Server
    host: str = "0.0.0.0"
    port: int = 8000

    # Database - PostgreSQL
    # In production, DATABASE_URL must be explicitly set
    database_url: PostgresDsn | None = Field(default=None)

    # Database - Neo4j
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    # In production, NEO4J_PASSWORD must be explicitly set
    neo4j_password: str | None = Field(default=None)

    # Vector Store - Pinecone
    pinecone_api_key: str | None = None
    pinecone_environment: str | None = None
    pinecone_index_name: str = "unfold-embeddings"

    # AI Models - OpenAI
    openai_api_key: str | None = None
    openai_embedding_model: str = "text-embedding-3-large"
    openai_chat_model: str = "gpt-4o"

    # AI Models - Anthropic
    anthropic_api_key: str | None = None

    # External APIs
    crossref_email: str | None = None
    orcid_client_id: str | None = None
    orcid_client_secret: str | None = None
    semantic_scholar_api_key: str | None = None

    # Security
    # JWT_SECRET must be explicitly set - no insecure defaults
    jwt_secret: str | None = Field(default=None)
    jwt_algorithm: str = "HS256"
    jwt_expiration_minutes: int = 30
    jwt_refresh_expiration_days: int = 7

    # Rate Limiting
    rate_limit_enabled: bool = True
    rate_limit_requests_per_minute: int = 60
    rate_limit_auth_requests_per_minute: int = 10  # Stricter for auth endpoints

    # CORS
    cors_origins: list[str] = ["http://localhost:3000"]

    # Redis Cache
    redis_url: str = "redis://localhost:6379/0"
    redis_max_connections: int = 50
    redis_socket_timeout: float = 5.0
    redis_socket_connect_timeout: float = 5.0
    cache_enabled: bool = True
    cache_default_ttl: int = 3600  # 1 hour

    # Database Pool Settings
    db_pool_size: int = 20
    db_max_overflow: int = 40
    db_pool_recycle: int = 1800  # 30 minutes
    db_pool_timeout: int = 30
    db_statement_timeout: int = 30000  # 30 seconds in ms

    @field_validator("cors_origins", mode="before")
    @classmethod
    def parse_cors_origins(cls, v: str | list[str]) -> list[str]:
        """Parse CORS origins from comma-separated string or list."""
        if isinstance(v, str):
            return [origin.strip() for origin in v.split(",")]
        return v

    @model_validator(mode="after")
    def validate_production_settings(self) -> "Settings":
        """Validate that production environment has secure settings."""
        is_production = self.environment in ("production", "staging")

        # JWT Secret validation
        if self.jwt_secret is None:
            if is_production:
                raise ConfigurationError(
                    "JWT_SECRET environment variable must be set in production/staging. "
                    "Generate one with: python -c \"import secrets; print(secrets.token_urlsafe(32))\""
                )
            # Generate a random secret for development (logged as warning)
            self.jwt_secret = secrets.token_urlsafe(32)
            logger.warning(
                "JWT_SECRET not set - using randomly generated secret. "
                "This is only acceptable in development mode."
            )

        if is_production and len(self.jwt_secret) < 32:
            raise ConfigurationError(
                "JWT_SECRET must be at least 32 characters in production"
            )

        # Database URL validation
        if self.database_url is None:
            if is_production:
                raise ConfigurationError(
                    "DATABASE_URL environment variable must be set in production/staging"
                )
            # Use development default
            self.database_url = PostgresDsn(
                "postgresql://postgres:postgres@localhost:5432/unfold"
            )

        # Neo4j password validation
        if self.neo4j_password is None:
            if is_production:
                raise ConfigurationError(
                    "NEO4J_PASSWORD environment variable must be set in production/staging"
                )
            # Use development default
            self.neo4j_password = "password"

        if is_production and self.neo4j_password == "password":
            raise ConfigurationError(
                "NEO4J_PASSWORD cannot be 'password' in production"
            )

        return self


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()


# Convenience singleton
settings = get_settings()
