"""
Caching services for performance optimization.
Redis backend (optional; nothing in the request path depends on it).
"""

from .redis_cache import (
    RedisCache,
    RedisCacheManager,
    init_redis,
    close_redis,
    is_redis_available,
    check_redis_health,
    get_redis_cache_manager,
)

__all__ = [
    "RedisCache",
    "RedisCacheManager",
    "init_redis",
    "close_redis",
    "is_redis_available",
    "check_redis_health",
    "get_redis_cache_manager",
]
