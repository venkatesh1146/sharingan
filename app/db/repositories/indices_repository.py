"""
Indices Repository - Data access layer for indices_timeseries collection.

Provides CRUD operations and queries for historical indices data.
"""

from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from pymongo import UpdateOne

from app.db.mongodb import get_mongodb_client
from app.db.models.indices_document import IndicesTimeseriesDocument
from app.utils.logging import get_logger

logger = get_logger(__name__)

COLLECTION_NAME = "indices_timeseries"


class IndicesRepository:
    """
    Repository for indices_timeseries collection.
    
    Provides methods for:
    - Storing indices data points
    - Querying historical data
    - Time-series analysis queries
    """

    def __init__(self):
        self._client = get_mongodb_client()

    @property
    def collection(self):
        """Get the indices_timeseries collection."""
        return self._client.get_collection(COLLECTION_NAME)

    async def create(self, document: IndicesTimeseriesDocument) -> str:
        """
        Create a new indices data point.
        
        Args:
            document: IndicesTimeseriesDocument to insert
            
        Returns:
            ticker of created document
        """
        data = document.to_mongo_dict()
        await self.collection.insert_one(data)
        
        logger.debug(
            "indices_datapoint_created",
            ticker=document.ticker,
            timestamp=document.timestamp.isoformat(),
        )
        
        return document.ticker

    async def bulk_insert(
        self,
        documents: List[IndicesTimeseriesDocument],
    ) -> int:
        """
        Bulk insert multiple indices data points.
        
        Args:
            documents: List of IndicesTimeseriesDocument
            
        Returns:
            Number of documents inserted
        """
        if not documents:
            return 0

        data_list = [doc.to_mongo_dict() for doc in documents]
        result = await self.collection.insert_many(data_list)
        
        count = len(result.inserted_ids)
        logger.info("indices_bulk_inserted", count=count)
        
        return count

    async def get_latest_for_ticker(
        self,
        ticker: str,
    ) -> Optional[IndicesTimeseriesDocument]:
        """
        Get the most recent data point for a ticker.
        
        Args:
            ticker: Index ticker symbol
            
        Returns:
            Latest IndicesTimeseriesDocument if found
        """
        doc = await self.collection.find_one(
            {"ticker": ticker},
            sort=[("timestamp", -1)],
        )
        
        if doc:
            return IndicesTimeseriesDocument.from_mongo_dict(doc)
        return None

    async def get_latest_all(self) -> Dict[str, IndicesTimeseriesDocument]:
        """
        Get the latest data point for each unique ticker.
        
        Returns:
            Dict mapping ticker to latest IndicesTimeseriesDocument
        """
        pipeline = [
            {"$sort": {"timestamp": -1}},
            {"$group": {
                "_id": "$ticker",
                "doc": {"$first": "$$ROOT"},
            }},
        ]
        
        result = {}
        async for item in self.collection.aggregate(pipeline):
            ticker = item["_id"]
            doc = item["doc"]
            doc.pop("_id", None)
            result[ticker] = IndicesTimeseriesDocument.from_mongo_dict(doc)
        
        return result

    async def get_history(
        self,
        ticker: str,
        hours: int = 24,
        limit: int = 100,
    ) -> List[IndicesTimeseriesDocument]:
        """
        Get historical data for a ticker.
        
        Args:
            ticker: Index ticker symbol
            hours: How many hours back
            limit: Maximum data points
            
        Returns:
            List of IndicesTimeseriesDocument sorted by timestamp desc
        """
        cutoff = datetime.utcnow() - timedelta(hours=hours)
        
        cursor = self.collection.find({
            "ticker": ticker,
            "timestamp": {"$gte": cutoff},
        }).sort("timestamp", -1).limit(limit)
        
        documents = []
        async for doc in cursor:
            documents.append(IndicesTimeseriesDocument.from_mongo_dict(doc))
        
        return documents

    async def get_change_over_period(
        self,
        ticker: str,
        hours: int = 24,
    ) -> Optional[Dict[str, Any]]:
        """
        Calculate change over a time period.
        
        Args:
            ticker: Index ticker symbol
            hours: Time period in hours
            
        Returns:
            Dict with start, end, and change data, or None if insufficient data
        """
        now = datetime.utcnow()
        cutoff = now - timedelta(hours=hours)
        
        # Get latest
        latest = await self.collection.find_one(
            {"ticker": ticker},
            sort=[("timestamp", -1)],
        )
        
        # Get oldest within period
        oldest = await self.collection.find_one(
            {"ticker": ticker, "timestamp": {"$gte": cutoff}},
            sort=[("timestamp", 1)],
        )
        
        if not latest or not oldest:
            return None
        
        latest_price = latest["data"]["current_price"]
        oldest_price = oldest["data"]["current_price"]
        
        change_absolute = latest_price - oldest_price
        change_percent = (change_absolute / oldest_price * 100) if oldest_price else 0
        
        return {
            "ticker": ticker,
            "period_hours": hours,
            "start_price": oldest_price,
            "end_price": latest_price,
            "change_absolute": round(change_absolute, 2),
            "change_percent": round(change_percent, 2),
            "start_timestamp": oldest["timestamp"],
            "end_timestamp": latest["timestamp"],
        }

    async def cleanup_old_data(self, days: int = 90) -> int:
        """
        Manually clean up old data.
        
        Note: TTL index handles this automatically, but this can be
        called for immediate cleanup or different retention period.
        
        Args:
            days: Data older than this will be deleted
            
        Returns:
            Number of deleted documents
        """
        cutoff = datetime.utcnow() - timedelta(days=days)
        
        result = await self.collection.delete_many({
            "created_at": {"$lt": cutoff},
        })
        
        if result.deleted_count > 0:
            logger.info(
                "old_indices_data_deleted",
                count=result.deleted_count,
                days=days,
            )
        
        return result.deleted_count

    async def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the indices collection.
        
        Returns:
            Dict with collection statistics
        """
        total_count = await self.collection.count_documents({})
        
        # Count distinct tickers
        tickers = await self.collection.distinct("ticker")
        
        # Get date range
        oldest = await self.collection.find_one(sort=[("timestamp", 1)])
        newest = await self.collection.find_one(sort=[("timestamp", -1)])
        
        return {
            "total_documents": total_count,
            "unique_tickers": len(tickers),
            "tickers": tickers,
            "oldest_timestamp": oldest["timestamp"] if oldest else None,
            "newest_timestamp": newest["timestamp"] if newest else None,
        }


# Singleton instance
_indices_repository: Optional[IndicesRepository] = None


def get_indices_repository() -> IndicesRepository:
    """Get singleton IndicesRepository instance."""
    global _indices_repository
    if _indices_repository is None:
        _indices_repository = IndicesRepository()
    return _indices_repository
