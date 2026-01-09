"""
Caching service for performance optimization.
Implements in-memory LRU cache with optional Redis backend.
"""

import hashlib
import json
import time
from dataclasses import dataclass, field
from typing import Optional, Any, Callable, TypeVar
from collections import OrderedDict
from functools import wraps
import asyncio

T = TypeVar("T")


@dataclass
class CacheEntry:
    """A single cache entry with metadata."""

    key: str
    value: Any
    created_at: float
    expires_at: float
    hit_count: int = 0
    last_accessed: float = field(default_factory=time.time)

    @property
    def is_expired(self) -> bool:
        return time.time() > self.expires_at

    @property
    def ttl_remaining(self) -> float:
        remaining = self.expires_at - time.time()
        return max(0, remaining)


@dataclass
class CacheStats:
    """Cache statistics."""

    hits: int = 0
    misses: int = 0
    evictions: int = 0
    expired: int = 0
    current_size: int = 0
    max_size: int = 0

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return (self.hits / total * 100) if total > 0 else 0.0

    def to_dict(self) -> dict:
        return {
            "hits": self.hits,
            "misses": self.misses,
            "evictions": self.evictions,
            "expired": self.expired,
            "current_size": self.current_size,
            "max_size": self.max_size,
            "hit_rate": f"{self.hit_rate:.2f}%",
        }


class LRUCache:
    """
    Thread-safe LRU cache with TTL support.
    """

    def __init__(self, max_size: int = 1000, default_ttl: int = 3600):
        """
        Initialize LRU cache.

        Args:
            max_size: Maximum number of entries
            default_ttl: Default TTL in seconds
        """
        self.max_size = max_size
        self.default_ttl = default_ttl
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._stats = CacheStats(max_size=max_size)
        self._lock = asyncio.Lock()

    async def get(self, key: str) -> Optional[Any]:
        """
        Get value from cache.

        Args:
            key: Cache key

        Returns:
            Cached value or None if not found/expired
        """
        async with self._lock:
            entry = self._cache.get(key)

            if entry is None:
                self._stats.misses += 1
                return None

            if entry.is_expired:
                del self._cache[key]
                self._stats.expired += 1
                self._stats.misses += 1
                self._stats.current_size = len(self._cache)
                return None

            # Move to end (most recently used)
            self._cache.move_to_end(key)
            entry.hit_count += 1
            entry.last_accessed = time.time()
            self._stats.hits += 1

            return entry.value

    async def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None,
    ) -> None:
        """
        Set value in cache.

        Args:
            key: Cache key
            value: Value to cache
            ttl: TTL in seconds (uses default if not provided)
        """
        async with self._lock:
            ttl = ttl if ttl is not None else self.default_ttl
            now = time.time()

            # Remove existing entry if present
            if key in self._cache:
                del self._cache[key]

            # Evict LRU entries if at capacity
            while len(self._cache) >= self.max_size:
                self._cache.popitem(last=False)
                self._stats.evictions += 1

            # Add new entry
            self._cache[key] = CacheEntry(
                key=key,
                value=value,
                created_at=now,
                expires_at=now + ttl,
            )
            self._stats.current_size = len(self._cache)

    async def delete(self, key: str) -> bool:
        """
        Delete entry from cache.

        Args:
            key: Cache key

        Returns:
            True if entry was deleted
        """
        async with self._lock:
            if key in self._cache:
                del self._cache[key]
                self._stats.current_size = len(self._cache)
                return True
            return False

    async def clear(self) -> int:
        """
        Clear all entries.

        Returns:
            Number of entries cleared
        """
        async with self._lock:
            count = len(self._cache)
            self._cache.clear()
            self._stats.current_size = 0
            return count

    async def clear_expired(self) -> int:
        """
        Remove all expired entries.

        Returns:
            Number of entries removed
        """
        async with self._lock:
            expired_keys = [
                key for key, entry in self._cache.items() if entry.is_expired
            ]
            for key in expired_keys:
                del self._cache[key]
                self._stats.expired += 1

            self._stats.current_size = len(self._cache)
            return len(expired_keys)

    async def get_stats(self) -> CacheStats:
        """Get cache statistics."""
        async with self._lock:
            self._stats.current_size = len(self._cache)
            return self._stats

    async def get_keys(self, pattern: Optional[str] = None) -> list[str]:
        """
        Get all cache keys, optionally filtered by pattern.

        Args:
            pattern: Optional prefix pattern

        Returns:
            List of matching keys
        """
        async with self._lock:
            if pattern:
                return [k for k in self._cache.keys() if k.startswith(pattern)]
            return list(self._cache.keys())


class CacheManager:
    """
    Manages multiple cache namespaces.
    """

    # Cache namespaces for different data types
    NAMESPACES = {
        "documents": {"max_size": 500, "ttl": 3600},  # 1 hour
        "graphs": {"max_size": 200, "ttl": 1800},  # 30 min
        "embeddings": {"max_size": 1000, "ttl": 7200},  # 2 hours
        "api_responses": {"max_size": 500, "ttl": 300},  # 5 min
        "user_sessions": {"max_size": 1000, "ttl": 3600},
        "citations": {"max_size": 500, "ttl": 3600},
        "ethics": {"max_size": 200, "ttl": 600},  # 10 min
    }

    def __init__(self):
        self._caches: dict[str, LRUCache] = {}

        # Initialize default namespaces
        for namespace, config in self.NAMESPACES.items():
            self._caches[namespace] = LRUCache(
                max_size=config["max_size"],
                default_ttl=config["ttl"],
            )

    def get_cache(self, namespace: str) -> LRUCache:
        """
        Get or create cache for namespace.

        Args:
            namespace: Cache namespace

        Returns:
            LRUCache instance
        """
        if namespace not in self._caches:
            self._caches[namespace] = LRUCache()
        return self._caches[namespace]

    async def get(self, namespace: str, key: str) -> Optional[Any]:
        """Get value from namespaced cache."""
        cache = self.get_cache(namespace)
        return await cache.get(key)

    async def set(
        self,
        namespace: str,
        key: str,
        value: Any,
        ttl: Optional[int] = None,
    ) -> None:
        """Set value in namespaced cache."""
        cache = self.get_cache(namespace)
        await cache.set(key, value, ttl)

    async def delete(self, namespace: str, key: str) -> bool:
        """Delete from namespaced cache."""
        cache = self.get_cache(namespace)
        return await cache.delete(key)

    async def clear_namespace(self, namespace: str) -> int:
        """Clear all entries in a namespace."""
        cache = self.get_cache(namespace)
        return await cache.clear()

    async def clear_all(self) -> dict[str, int]:
        """Clear all caches."""
        results = {}
        for namespace, cache in self._caches.items():
            results[namespace] = await cache.clear()
        return results

    async def get_all_stats(self) -> dict[str, dict]:
        """Get stats for all namespaces."""
        stats = {}
        for namespace, cache in self._caches.items():
            cache_stats = await cache.get_stats()
            stats[namespace] = cache_stats.to_dict()
        return stats

    async def cleanup_expired(self) -> dict[str, int]:
        """Clean up expired entries in all caches."""
        results = {}
        for namespace, cache in self._caches.items():
            results[namespace] = await cache.clear_expired()
        return results


def make_cache_key(*args, **kwargs) -> str:
    """
    Create a cache key from arguments.

    Args:
        *args: Positional arguments
        **kwargs: Keyword arguments

    Returns:
        SHA-256 hash of serialized arguments
    """
    key_data = json.dumps({"args": args, "kwargs": kwargs}, sort_keys=True, default=str)
    return hashlib.sha256(key_data.encode()).hexdigest()[:32]


def cached(
    namespace: str,
    ttl: Optional[int] = None,
    key_prefix: str = "",
):
    """
    Decorator for caching async function results.

    Args:
        namespace: Cache namespace
        ttl: Cache TTL in seconds
        key_prefix: Optional prefix for cache keys

    Usage:
        @cached("documents", ttl=3600)
        async def get_document(doc_id: str):
            ...
    """

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        async def wrapper(*args, **kwargs) -> T:
            cache = get_cache_manager()
            key = f"{key_prefix}{func.__name__}:{make_cache_key(*args, **kwargs)}"

            # Try to get from cache
            cached_value = await cache.get(namespace, key)
            if cached_value is not None:
                return cached_value

            # Call function and cache result
            result = await func(*args, **kwargs)
            if result is not None:
                await cache.set(namespace, key, result, ttl)

            return result

        return wrapper

    return decorator


def cache_invalidate(namespace: str, key_pattern: str):
    """
    Decorator to invalidate cache after function execution.

    Args:
        namespace: Cache namespace
        key_pattern: Key pattern to invalidate

    Usage:
        @cache_invalidate("documents", "get_document:")
        async def update_document(doc_id: str, data: dict):
            ...
    """

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        async def wrapper(*args, **kwargs) -> T:
            result = await func(*args, **kwargs)

            # Invalidate matching keys
            cache = get_cache_manager()
            lru_cache = cache.get_cache(namespace)
            keys = await lru_cache.get_keys(key_pattern)
            for key in keys:
                await lru_cache.delete(key)

            return result

        return wrapper

    return decorator


# Singleton instance
_cache_manager: Optional[CacheManager] = None


def get_cache_manager() -> CacheManager:
    """Get or create singleton CacheManager instance."""
    global _cache_manager
    if _cache_manager is None:
        _cache_manager = CacheManager()
    return _cache_manager


async def start_cache_cleanup_task(interval: int = 300):
    """
    Start background task to clean up expired cache entries.

    Args:
        interval: Cleanup interval in seconds (default: 5 minutes)
    """
    cache = get_cache_manager()
    while True:
        await asyncio.sleep(interval)
        await cache.cleanup_expired()
