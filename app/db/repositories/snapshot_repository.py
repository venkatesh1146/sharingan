"""
Snapshot Repository - Data access layer for market_snapshots collection.

Provides CRUD operations and queries for market snapshots.
"""

from datetime import datetime, timedelta
from typing import Any, Dict, List, Literal, Optional

from app.db.mongodb import get_mongodb_client
from app.db.models.snapshot_document import MarketSnapshotDocument
from app.utils.logging import get_logger

logger = get_logger(__name__)

COLLECTION_NAME = "market_snapshots"


class SnapshotRepository:
    """
    Repository for market_snapshots collection.
    
    Provides methods for:
    - Creating new snapshots
    - Retrieving latest snapshot by phase
    - TTL-based expiration management
    """

    def __init__(self):
        self._client = get_mongodb_client()

    @property
    def collection(self):
        """Get the market_snapshots collection."""
        return self._client.get_collection(COLLECTION_NAME)

    async def create(self, document: MarketSnapshotDocument) -> str:
        """
        Create or update a market snapshot.
        
        Uses upsert to handle duplicate snapshot_ids gracefully
        (e.g., when regenerating within the same minute).
        
        Args:
            document: MarketSnapshotDocument to insert/update
            
        Returns:
            snapshot_id of created/updated document
        """
        data = document.to_mongo_dict()
        
        # Use replace_one with upsert to handle duplicates gracefully
        await self.collection.replace_one(
            {"snapshot_id": document.snapshot_id},
            data,
            upsert=True,
        )
        
        logger.info(
            "market_snapshot_created",
            snapshot_id=document.snapshot_id,
            market_phase=document.market_phase,
        )
        
        return document.snapshot_id

    async def get_by_id(self, snapshot_id: str) -> Optional[MarketSnapshotDocument]:
        """
        Get a snapshot by its ID.
        
        Args:
            snapshot_id: Unique snapshot identifier
            
        Returns:
            MarketSnapshotDocument if found, None otherwise
        """
        doc = await self.collection.find_one({"snapshot_id": snapshot_id})
        if doc:
            return MarketSnapshotDocument.from_mongo_dict(doc)
        return None

    async def get_latest(
        self,
        market_phase: Optional[str] = None,
    ) -> Optional[MarketSnapshotDocument]:
        """
        Get the most recent snapshot.
        
        Args:
            market_phase: Optional filter by market phase
            
        Returns:
            Latest MarketSnapshotDocument if found
        """
        query: Dict[str, Any] = {}
        
        if market_phase:
            query["market_phase"] = market_phase
        
        doc = await self.collection.find_one(
            query,
            sort=[("generated_at", -1)],
        )
        
        if doc:
            return MarketSnapshotDocument.from_mongo_dict(doc)
        return None

    async def get_latest_valid(
        self,
        market_phase: Optional[str] = None,
    ) -> Optional[MarketSnapshotDocument]:
        """
        Get the most recent non-expired snapshot.
        
        Args:
            market_phase: Optional filter by market phase
            
        Returns:
            Latest valid MarketSnapshotDocument if found
        """
        now = datetime.utcnow()
        
        query: Dict[str, Any] = {
            "expires_at": {"$gt": now},
        }
        
        if market_phase:
            query["market_phase"] = market_phase
        
        doc = await self.collection.find_one(
            query,
            sort=[("generated_at", -1)],
        )
        
        if doc:
            snapshot = MarketSnapshotDocument.from_mongo_dict(doc)
            logger.debug(
                "found_valid_snapshot",
                snapshot_id=snapshot.snapshot_id,
                age_seconds=(now - snapshot.generated_at).total_seconds(),
            )
            return snapshot
        
        return None

    async def is_stale(
        self,
        market_phase: str,
        max_age_seconds: int = 900,
    ) -> bool:
        """
        Check if the latest snapshot is stale.
        
        Args:
            market_phase: Market phase to check
            max_age_seconds: Maximum age in seconds (default 15 min)
            
        Returns:
            True if stale or no snapshot exists
        """
        cutoff = datetime.utcnow() - timedelta(seconds=max_age_seconds)
        
        count = await self.collection.count_documents({
            "market_phase": market_phase,
            "generated_at": {"$gte": cutoff},
        }, limit=1)
        
        return count == 0

    async def get_recent(
        self,
        hours: int = 24,
        limit: int = 10,
    ) -> List[MarketSnapshotDocument]:
        """
        Get recent snapshots.
        
        Args:
            hours: How many hours back
            limit: Maximum snapshots
            
        Returns:
            List of MarketSnapshotDocument
        """
        cutoff = datetime.utcnow() - timedelta(hours=hours)
        
        cursor = self.collection.find(
            {"generated_at": {"$gte": cutoff}}
        ).sort("generated_at", -1).limit(limit)
        
        documents = []
        async for doc in cursor:
            documents.append(MarketSnapshotDocument.from_mongo_dict(doc))
        
        return documents

    async def delete_expired(self) -> int:
        """
        Manually delete expired snapshots.
        
        Note: TTL index handles this automatically, but this can be
        called for immediate cleanup.
        
        Returns:
            Number of deleted snapshots
        """
        now = datetime.utcnow()
        
        result = await self.collection.delete_many({
            "expires_at": {"$lt": now},
        })
        
        if result.deleted_count > 0:
            logger.info(
                "expired_snapshots_deleted",
                count=result.deleted_count,
            )
        
        return result.deleted_count

    async def count_by_phase(self, hours: int = 24) -> Dict[str, int]:
        """
        Count snapshots by market phase.
        
        Args:
            hours: How many hours back
            
        Returns:
            Dict with phase counts
        """
        cutoff = datetime.utcnow() - timedelta(hours=hours)
        
        pipeline = [
            {"$match": {"generated_at": {"$gte": cutoff}}},
            {"$group": {"_id": "$market_phase", "count": {"$sum": 1}}},
        ]
        
        result = {"pre": 0, "mid": 0, "post": 0}
        
        async for doc in self.collection.aggregate(pipeline):
            phase = doc["_id"]
            if phase in result:
                result[phase] = doc["count"]
        
        return result


# Singleton instance
_snapshot_repository: Optional[SnapshotRepository] = None


def get_snapshot_repository() -> SnapshotRepository:
    """Get singleton SnapshotRepository instance."""
    global _snapshot_repository
    if _snapshot_repository is None:
        _snapshot_repository = SnapshotRepository()
    return _snapshot_repository
