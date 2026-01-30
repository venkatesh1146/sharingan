"""
MongoDB Client - Connection management for Market Intelligence database.

Provides singleton MongoDB client with connection pooling and
lifecycle management for async operations.
"""

from typing import Optional
from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase

from app.config import get_settings
from app.utils.logging import get_logger

logger = get_logger(__name__)


class MongoDBClient:
    """
    Singleton MongoDB client manager.
    
    Handles connection lifecycle, database access, and ensures
    proper cleanup on application shutdown.
    """
    
    _instance: Optional["MongoDBClient"] = None
    _client: Optional[AsyncIOMotorClient] = None
    _db: Optional[AsyncIOMotorDatabase] = None

    def __new__(cls) -> "MongoDBClient":
        """Ensure singleton instance."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    async def connect(self) -> None:
        """
        Initialize MongoDB connection.
        
        Creates connection pool and verifies connectivity.
        
        Raises:
            ConnectionError: If unable to connect to MongoDB.
        """
        if self._client is not None:
            return

        settings = get_settings()

        try:
            self._client = AsyncIOMotorClient(
                settings.MONGODB_URL,
                maxPoolSize=settings.MONGODB_MAX_POOL_SIZE,
                minPoolSize=settings.MONGODB_MIN_POOL_SIZE,
                serverSelectionTimeoutMS=5000,
            )

            # Verify connection
            await self._client.admin.command("ping")
            
            self._db = self._client[settings.MONGODB_DATABASE]

            logger.info(
                "mongodb_connected",
                database=settings.MONGODB_DATABASE,
                max_pool=settings.MONGODB_MAX_POOL_SIZE,
            )

            # Create indexes on startup
            await self._create_indexes()

        except Exception as e:
            logger.error("mongodb_connection_failed", error=str(e))
            raise ConnectionError(f"Failed to connect to MongoDB: {e}") from e

    async def disconnect(self) -> None:
        """Close MongoDB connection and cleanup resources."""
        if self._client:
            self._client.close()
            self._client = None
            self._db = None
            logger.info("mongodb_disconnected")

    def get_database(self) -> AsyncIOMotorDatabase:
        """
        Get the database instance.
        
        Returns:
            AsyncIOMotorDatabase instance.
            
        Raises:
            RuntimeError: If not connected.
        """
        if self._db is None:
            raise RuntimeError("MongoDB not connected. Call connect() first.")
        return self._db

    def get_collection(self, name: str):
        """
        Get a collection by name.
        
        Args:
            name: Collection name.
            
        Returns:
            AsyncIOMotorCollection instance.
        """
        return self.get_database()[name]

    async def _create_indexes(self) -> None:
        """Create required indexes for all collections."""
        try:
            db = self.get_database()

            # news_articles indexes
            news_collection = db["news_articles"]
            await news_collection.create_index("news_id", unique=True, name="idx_news_id")
            await news_collection.create_index(
                [("published_at", -1)], 
                name="idx_published_at"
            )
            await news_collection.create_index(
                [("sentiment", 1), ("published_at", -1)],
                name="idx_sentiment_time"
            )
            await news_collection.create_index(
                "mentioned_stocks",
                name="idx_mentioned_stocks"
            )
            await news_collection.create_index(
                "mentioned_sectors",
                name="idx_mentioned_sectors"
            )
            await news_collection.create_index(
                [("is_breaking", 1), ("published_at", -1)],
                name="idx_breaking_news"
            )
            await news_collection.create_index(
                [("processed", 1), ("analyzed", 1)],
                name="idx_processing_status"
            )

            # market_snapshots indexes
            snapshots_collection = db["market_snapshots"]
            await snapshots_collection.create_index(
                [("generated_at", -1)],
                name="idx_generated_at"
            )
            await snapshots_collection.create_index(
                [("market_phase", 1), ("generated_at", -1)],
                name="idx_phase_time"
            )
            await snapshots_collection.create_index(
                "snapshot_id",
                unique=True,
                name="idx_snapshot_id"
            )
            await snapshots_collection.create_index(
                "expires_at",
                expireAfterSeconds=0,
                name="idx_ttl_expire"
            )

            # indices_timeseries indexes
            indices_collection = db["indices_timeseries"]
            await indices_collection.create_index(
                [("ticker", 1), ("timestamp", -1)],
                name="idx_ticker_time"
            )
            await indices_collection.create_index(
                [("timestamp", -1)],
                name="idx_timestamp"
            )
            # TTL index for 90 days retention
            await indices_collection.create_index(
                "created_at",
                expireAfterSeconds=7776000,  # 90 days
                name="idx_ttl_90days"
            )

            logger.info("mongodb_indexes_created")

        except Exception as e:
            logger.error("mongodb_index_creation_failed", error=str(e))
            # Don't raise - indexes can be created later

    async def health_check(self) -> bool:
        """
        Check MongoDB connectivity.
        
        Returns:
            True if healthy, False otherwise.
        """
        try:
            if self._client is None:
                return False
            await self._client.admin.command("ping")
            return True
        except Exception:
            return False


# Global client instance
_mongodb_client: Optional[MongoDBClient] = None


def get_mongodb_client() -> MongoDBClient:
    """Get singleton MongoDB client instance."""
    global _mongodb_client
    if _mongodb_client is None:
        _mongodb_client = MongoDBClient()
    return _mongodb_client
