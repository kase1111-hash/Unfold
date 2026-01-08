"""
Caching services for performance optimization.
Supports both in-memory LRU cache and Redis backend.
"""

from .cache import (
    LRUCache,
    CacheManager,
    CacheEntry,
    CacheStats,
    get_cache_manager,
    cached,
    cache_invalidate,
    make_cache_key,
    start_cache_cleanup_task,
)

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
    # In-memory cache
    "LRUCache",
    "CacheManager",
    "CacheEntry",
    "CacheStats",
    "get_cache_manager",
    "cached",
    "cache_invalidate",
    "make_cache_key",
    "start_cache_cleanup_task",
    # Redis cache
    "RedisCache",
    "RedisCacheManager",
    "init_redis",
    "close_redis",
    "is_redis_available",
    "check_redis_health",
    "get_redis_cache_manager",
]
