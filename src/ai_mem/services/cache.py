"""Search Cache Service - Manage search result caching.

This service handles caching of search results to avoid
repeated expensive queries.
"""

import hashlib
import time
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

from ..logging_config import get_logger

if TYPE_CHECKING:
    from ..config import AppConfig

logger = get_logger("services.cache")


class SearchCacheService:
    """Manages search result caching.

    Provides time-based caching of search results with
    configurable TTL and max entries.

    Usage:
        service = SearchCacheService(config)
        key = service.build_key(query, limit, project)

        cached = service.get(key)
        if cached:
            return cached

        results = perform_search()
        service.set(key, results)
    """

    def __init__(self, config: "AppConfig"):
        """Initialize cache service.

        Args:
            config: Application configuration
        """
        self.config = config
        self._cache: Dict[str, Tuple[float, List[Dict[str, Any]]]] = {}
        self._hits = 0
        self._misses = 0
        self._last_hit: Optional[bool] = None

    def is_enabled(self) -> bool:
        """Check if caching is enabled.

        Returns:
            True if caching is active
        """
        return (
            self.config.search.cache_ttl_seconds > 0
            and self.config.search.cache_max_entries > 0
        )

    def build_key(
        self,
        query: str,
        limit: int,
        project: Optional[str] = None,
        obs_type: Optional[str] = None,
        session_id: Optional[str] = None,
        date_start: Optional[float] = None,
        date_end: Optional[float] = None,
        tags: Optional[List[str]] = None,
    ) -> str:
        """Build a cache key from search parameters.

        Args:
            query: Search query
            limit: Result limit
            project: Project filter
            obs_type: Observation type filter
            session_id: Session filter
            date_start: Date range start
            date_end: Date range end
            tags: Tag filters

        Returns:
            Hash key for the search parameters
        """
        parts = [
            query,
            str(limit),
            project or "",
            obs_type or "",
            session_id or "",
            str(date_start) if date_start else "",
            str(date_end) if date_end else "",
            ",".join(sorted(tags or [])),
        ]
        raw = "|".join(parts)
        return hashlib.sha256(raw.encode()).hexdigest()[:32]

    def get(self, key: str) -> Optional[List[Dict[str, Any]]]:
        """Retrieve cached results.

        Args:
            key: Cache key

        Returns:
            Cached results or None if not found/expired
        """
        if not self.is_enabled():
            return None

        if key not in self._cache:
            self._record_miss()
            return None

        timestamp, results = self._cache[key]
        if time.time() - timestamp > self.config.search.cache_ttl_seconds:
            # Expired
            del self._cache[key]
            self._record_miss()
            return None

        self._record_hit()
        return results

    def set(self, key: str, results: List[Dict[str, Any]]) -> None:
        """Store results in cache.

        Args:
            key: Cache key
            results: Search results to cache
        """
        if not self.is_enabled():
            return

        # Prune if at capacity
        if len(self._cache) >= self.config.search.cache_max_entries:
            self._prune_expired()

        # If still at capacity, remove oldest
        if len(self._cache) >= self.config.search.cache_max_entries:
            oldest_key = min(self._cache, key=lambda k: self._cache[k][0])
            del self._cache[oldest_key]

        self._cache[key] = (time.time(), results)

    def _prune_expired(self) -> int:
        """Remove expired entries from cache.

        Returns:
            Number of entries removed
        """
        now = time.time()
        ttl = self.config.search.cache_ttl_seconds
        expired = [k for k, (ts, _) in self._cache.items() if now - ts > ttl]
        for k in expired:
            del self._cache[k]
        return len(expired)

    def _record_hit(self) -> None:
        """Record a cache hit."""
        self._hits += 1
        self._last_hit = True

    def _record_miss(self) -> None:
        """Record a cache miss."""
        self._misses += 1
        self._last_hit = False

    @property
    def last_hit(self) -> Optional[bool]:
        """Get whether the last lookup was a hit."""
        return self._last_hit

    def get_summary(self) -> Dict[str, Any]:
        """Get cache statistics.

        Returns:
            Dictionary with cache stats
        """
        total = self._hits + self._misses
        return {
            "enabled": self.is_enabled(),
            "entries": len(self._cache),
            "max_entries": self.config.search.cache_max_entries,
            "ttl_seconds": self.config.search.cache_ttl_seconds,
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate_percent": round(self._hits / total * 100, 1) if total > 0 else 0,
        }

    def clear(self) -> int:
        """Clear all cached entries.

        Returns:
            Number of entries cleared
        """
        count = len(self._cache)
        self._cache.clear()
        self._hits = 0
        self._misses = 0
        self._last_hit = None
        logger.info(f"Cleared {count} cache entries")
        return count
