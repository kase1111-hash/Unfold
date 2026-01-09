"""
Redis caching backend for production environments.
Provides distributed caching with persistence and high availability.
"""

import json
import logging
from typing import Optional, Any

logger = logging.getLogger(__name__)

# Redis client singleton
_redis_client = None
_redis_available = False


async def init_redis() -> bool:
    """
    Initialize Redis connection pool.

    Returns:
        True if Redis is available, False otherwise
    """
    global _redis_client, _redis_available

    try:
        import redis.asyncio as redis
        from app.config import get_settings

        settings = get_settings()

        if not settings.cache_enabled:
            logger.info("Cache disabled in settings")
            return False

        _redis_client = redis.from_url(
            settings.redis_url,
            max_connections=settings.redis_max_connections,
            socket_timeout=settings.redis_socket_timeout,
            socket_connect_timeout=settings.redis_socket_connect_timeout,
            decode_responses=True,
            retry_on_timeout=True,
        )

        # Test connection
        await _redis_client.ping()
        _redis_available = True
        logger.info("Redis connection established successfully")
        return True

    except ImportError:
        logger.warning("redis package not installed, falling back to in-memory cache")
        _redis_available = False
        return False
    except Exception as e:
        logger.warning(f"Redis connection failed: {e}, falling back to in-memory cache")
        _redis_available = False
        return False


async def close_redis() -> None:
    """Close Redis connection pool."""
    global _redis_client, _redis_available

    if _redis_client is not None:
        await _redis_client.close()
        _redis_client = None
        _redis_available = False
        logger.info("Redis connection closed")


def is_redis_available() -> bool:
    """Check if Redis is available."""
    return _redis_available and _redis_client is not None


class RedisCache:
    """
    Redis-backed cache implementation.
    Falls back to None operations if Redis is unavailable.
    """

    def __init__(self, namespace: str, default_ttl: int = 3600):
        """
        Initialize Redis cache for a namespace.

        Args:
            namespace: Key prefix for this cache
            default_ttl: Default TTL in seconds
        """
        self.namespace = namespace
        self.default_ttl = default_ttl
        self._stats = {
            "hits": 0,
            "misses": 0,
            "errors": 0,
        }

    def _make_key(self, key: str) -> str:
        """Create namespaced key."""
        return f"unfold:{self.namespace}:{key}"

    async def get(self, key: str) -> Optional[Any]:
        """
        Get value from Redis cache.

        Args:
            key: Cache key

        Returns:
            Cached value or None
        """
        if not is_redis_available():
            return None

        try:
            full_key = self._make_key(key)
            value = await _redis_client.get(full_key)

            if value is None:
                self._stats["misses"] += 1
                return None

            self._stats["hits"] += 1
            return json.loads(value)

        except Exception as e:
            self._stats["errors"] += 1
            logger.debug(f"Redis get error: {e}")
            return None

    async def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None,
    ) -> bool:
        """
        Set value in Redis cache.

        Args:
            key: Cache key
            value: Value to cache (must be JSON serializable)
            ttl: TTL in seconds

        Returns:
            True if successful
        """
        if not is_redis_available():
            return False

        try:
            full_key = self._make_key(key)
            ttl = ttl if ttl is not None else self.default_ttl

            serialized = json.dumps(value, default=str)
            await _redis_client.setex(full_key, ttl, serialized)
            return True

        except Exception as e:
            self._stats["errors"] += 1
            logger.debug(f"Redis set error: {e}")
            return False

    async def delete(self, key: str) -> bool:
        """
        Delete value from Redis cache.

        Args:
            key: Cache key

        Returns:
            True if key was deleted
        """
        if not is_redis_available():
            return False

        try:
            full_key = self._make_key(key)
            result = await _redis_client.delete(full_key)
            return result > 0

        except Exception as e:
            self._stats["errors"] += 1
            logger.debug(f"Redis delete error: {e}")
            return False

    async def delete_pattern(self, pattern: str) -> int:
        """
        Delete all keys matching pattern.

        Args:
            pattern: Key pattern (supports * wildcard)

        Returns:
            Number of keys deleted
        """
        if not is_redis_available():
            return 0

        try:
            full_pattern = self._make_key(pattern)
            keys = []

            async for key in _redis_client.scan_iter(match=full_pattern):
                keys.append(key)

            if keys:
                return await _redis_client.delete(*keys)
            return 0

        except Exception as e:
            self._stats["errors"] += 1
            logger.debug(f"Redis delete_pattern error: {e}")
            return 0

    async def exists(self, key: str) -> bool:
        """Check if key exists in cache."""
        if not is_redis_available():
            return False

        try:
            full_key = self._make_key(key)
            return await _redis_client.exists(full_key) > 0
        except Exception:
            return False

    async def ttl(self, key: str) -> int:
        """Get remaining TTL for key in seconds."""
        if not is_redis_available():
            return -1

        try:
            full_key = self._make_key(key)
            return await _redis_client.ttl(full_key)
        except Exception:
            return -1

    async def incr(self, key: str, amount: int = 1) -> Optional[int]:
        """
        Increment a counter.

        Args:
            key: Counter key
            amount: Amount to increment

        Returns:
            New counter value
        """
        if not is_redis_available():
            return None

        try:
            full_key = self._make_key(key)
            return await _redis_client.incrby(full_key, amount)
        except Exception as e:
            self._stats["errors"] += 1
            logger.debug(f"Redis incr error: {e}")
            return None

    async def get_stats(self) -> dict:
        """Get cache statistics."""
        stats = dict(self._stats)
        total = stats["hits"] + stats["misses"]
        stats["hit_rate"] = (
            f"{(stats['hits'] / total * 100):.2f}%" if total > 0 else "0.00%"
        )
        stats["redis_available"] = is_redis_available()
        return stats

    async def clear(self) -> int:
        """Clear all keys in this namespace."""
        return await self.delete_pattern("*")


class RedisCacheManager:
    """
    Manages Redis caches across multiple namespaces.
    """

    # Cache namespaces with TTL configuration
    NAMESPACES = {
        "documents": 3600,  # 1 hour
        "graphs": 1800,  # 30 minutes
        "embeddings": 7200,  # 2 hours
        "api_responses": 300,  # 5 minutes
        "user_sessions": 3600,  # 1 hour
        "citations": 3600,  # 1 hour
        "ethics": 600,  # 10 minutes
        "rate_limit": 60,  # 1 minute
        "search": 900,  # 15 minutes
    }

    def __init__(self):
        self._caches: dict[str, RedisCache] = {}

        for namespace, ttl in self.NAMESPACES.items():
            self._caches[namespace] = RedisCache(namespace, default_ttl=ttl)

    def get_cache(self, namespace: str) -> RedisCache:
        """Get or create cache for namespace."""
        if namespace not in self._caches:
            self._caches[namespace] = RedisCache(namespace)
        return self._caches[namespace]

    async def get(self, namespace: str, key: str) -> Optional[Any]:
        """Get value from namespaced cache."""
        return await self.get_cache(namespace).get(key)

    async def set(
        self,
        namespace: str,
        key: str,
        value: Any,
        ttl: Optional[int] = None,
    ) -> bool:
        """Set value in namespaced cache."""
        return await self.get_cache(namespace).set(key, value, ttl)

    async def delete(self, namespace: str, key: str) -> bool:
        """Delete from namespaced cache."""
        return await self.get_cache(namespace).delete(key)

    async def get_all_stats(self) -> dict:
        """Get stats for all namespaces."""
        stats = {}
        for namespace, cache in self._caches.items():
            stats[namespace] = await cache.get_stats()
        return stats


# Redis health check
async def check_redis_health() -> dict:
    """
    Check Redis connection health.

    Returns:
        Health status dict
    """
    if not is_redis_available():
        return {
            "status": "unavailable",
            "message": "Redis not connected",
        }

    try:
        info = await _redis_client.info("server")
        memory = await _redis_client.info("memory")

        return {
            "status": "healthy",
            "redis_version": info.get("redis_version"),
            "used_memory_human": memory.get("used_memory_human"),
            "connected_clients": info.get("connected_clients"),
        }
    except Exception as e:
        return {
            "status": "error",
            "message": str(e),
        }


# Singleton instance
_redis_cache_manager: Optional[RedisCacheManager] = None


def get_redis_cache_manager() -> RedisCacheManager:
    """Get or create singleton RedisCacheManager."""
    global _redis_cache_manager
    if _redis_cache_manager is None:
        _redis_cache_manager = RedisCacheManager()
    return _redis_cache_manager
