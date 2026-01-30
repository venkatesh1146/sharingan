"""
Redis Service - Reusable caching utilities.

Provides unified Redis client management and caching functions
for use across the application.
"""

import json
from typing import Any, Optional
from datetime import datetime, timedelta
import redis.asyncio as redis

from app.utils.logging import get_logger
from app.config import get_settings

logger = get_logger(__name__)


class RedisService:
    """Singleton Redis service for caching and data operations."""

    _instance: Optional["RedisService"] = None
    _redis_client: Optional[redis.Redis] = None

    def __new__(cls) -> "RedisService":
        """Ensure singleton instance."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    async def connect(self) -> None:
        """Initialize Redis connection."""
        if self._redis_client is None:
            try:
                settings = get_settings()
                self._redis_client = await redis.from_url(
                    settings.REDIS_URL,
                    encoding="utf-8",
                    decode_responses=True,
                )
                # Test connection
                await self._redis_client.ping()
                logger.info("redis_connected", url=settings.REDIS_URL)
            except Exception as e:
                logger.error("redis_connection_failed", error=str(e))
                raise

    async def disconnect(self) -> None:
        """Close Redis connection."""
        if self._redis_client:
            await self._redis_client.close()
            self._redis_client = None
            logger.info("redis_disconnected")

    def get_client(self) -> redis.Redis:
        """Get Redis client instance."""
        if self._redis_client is None:
            raise RuntimeError("Redis not connected. Call connect() first.")
        return self._redis_client

    async def set(
        self,
        key: str,
        value: Any,
        ttl_seconds: int = 3600,
    ) -> bool:
        """
        Set a key-value pair in Redis with TTL.

        Args:
            key: Cache key
            value: Value to cache (will be JSON serialized)
            ttl_seconds: Time to live in seconds (default: 1 hour)

        Returns:
            True if successful, False otherwise
        """
        try:
            client = self.get_client()
            serialized_value = json.dumps(value)
            await client.setex(key, ttl_seconds, serialized_value)
            logger.debug("cache_set", key=key, ttl_seconds=ttl_seconds)
            return True
        except Exception as e:
            logger.warning("cache_set_failed", key=key, error=str(e))
            return False

    async def get(self, key: str) -> Optional[Any]:
        """
        Get a value from Redis cache.

        Args:
            key: Cache key

        Returns:
            Deserialized value if found, None otherwise
        """
        try:
            client = self.get_client()
            value = await client.get(key)
            if value:
                logger.debug("cache_hit", key=key)
                return json.loads(value)
            logger.debug("cache_miss", key=key)
            return None
        except Exception as e:
            logger.warning("cache_get_failed", key=key, error=str(e))
            return None

    async def delete(self, key: str) -> bool:
        """
        Delete a key from Redis.

        Args:
            key: Cache key to delete

        Returns:
            True if successful, False otherwise
        """
        try:
            client = self.get_client()
            result = await client.delete(key)
            logger.debug("cache_deleted", key=key)
            return bool(result)
        except Exception as e:
            logger.warning("cache_delete_failed", key=key, error=str(e))
            return False

    async def exists(self, key: str) -> bool:
        """
        Check if a key exists in Redis.

        Args:
            key: Cache key to check

        Returns:
            True if key exists, False otherwise
        """
        try:
            client = self.get_client()
            result = await client.exists(key)
            return bool(result)
        except Exception as e:
            logger.warning("cache_exists_check_failed", key=key, error=str(e))
            return False

    async def get_ttl(self, key: str) -> Optional[int]:
        """
        Get remaining TTL for a key in seconds.

        Args:
            key: Cache key

        Returns:
            TTL in seconds, -1 if key has no expiry, -2 if key doesn't exist, None on error
        """
        try:
            client = self.get_client()
            ttl = await client.ttl(key)
            return ttl if ttl != -2 else None
        except Exception as e:
            logger.warning("cache_ttl_check_failed", key=key, error=str(e))
            return None

    async def clear_pattern(self, pattern: str) -> int:
        """
        Delete all keys matching a pattern.

        Args:
            pattern: Redis key pattern (e.g., "cache:*")

        Returns:
            Number of keys deleted
        """
        try:
            client = self.get_client()
            cursor = 0
            deleted_count = 0

            while True:
                cursor, keys = await client.scan(cursor, match=pattern)
                if keys:
                    deleted_count += await client.delete(*keys)
                if cursor == 0:
                    break

            logger.debug("cache_pattern_cleared", pattern=pattern, count=deleted_count)
            return deleted_count
        except Exception as e:
            logger.warning("cache_pattern_clear_failed", pattern=pattern, error=str(e))
            return 0


def get_redis_service() -> RedisService:
    """Get singleton Redis service instance."""
    return RedisService()


def build_cache_key(prefix: str, identifier: str) -> str:
    """
    Build a standardized cache key.

    Args:
        prefix: Key prefix (e.g., "allMarketNews")
        identifier: Key identifier (e.g., date string)

    Returns:
        Formatted cache key (e.g., "allMarketNews:2026-01-30")
    """
    return f"{prefix}:{identifier}"


def get_date_identifier() -> str:
    """
    Get today's date as identifier for daily cache keys.

    Returns:
        Date string in YYYY-MM-DD format
    """
    return datetime.utcnow().strftime("%Y-%m-%d")


def get_24hr_ttl() -> int:
    """Get TTL value for 2 hours in seconds."""
    return 2 * 60 * 60
