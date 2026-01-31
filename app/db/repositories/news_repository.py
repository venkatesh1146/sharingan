"""
News Repository - Data access layer for news_articles collection.

Provides CRUD operations and specialized queries for news articles.
"""

from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from pymongo import UpdateOne
from pymongo.errors import DuplicateKeyError

from app.db.mongodb import get_mongodb_client
from app.db.models.news_document import NewsArticleDocument
from app.utils.logging import get_logger

logger = get_logger(__name__)

COLLECTION_NAME = "news_articles"


class NewsRepository:
    """
    Repository for news_articles collection.

    Provides methods for:
    - Creating/updating news articles
    - Batch upserts for efficient processing
    - Querying by various criteria
    - Deduplication checks
    """

    @property
    def collection(self):
        """Get the news_articles collection (fresh client reference each time)."""
        return get_mongodb_client().get_collection(COLLECTION_NAME)

    async def create(self, document: NewsArticleDocument) -> str:
        """
        Create a new news article.

        Args:
            document: NewsArticleDocument to insert

        Returns:
            news_id of created document

        Raises:
            DuplicateKeyError: If news_id already exists
        """
        try:
            data = document.to_mongo_dict()
            await self.collection.insert_one(data)
            logger.info("news_article_created", news_id=document.news_id)
            return document.news_id
        except DuplicateKeyError:
            logger.warning("news_article_duplicate", news_id=document.news_id)
            raise

    async def upsert(self, document: NewsArticleDocument) -> bool:
        """
        Upsert a news article (create or update).

        Args:
            document: NewsArticleDocument to upsert

        Returns:
            True if new document created, False if updated
        """
        data = document.to_mongo_dict()
        data["updated_at"] = datetime.utcnow()

        result = await self.collection.update_one(
            {"news_id": document.news_id},
            {"$set": data},
            upsert=True,
        )

        is_new = result.upserted_id is not None
        logger.debug(
            "news_article_upserted",
            news_id=document.news_id,
            is_new=is_new,
        )
        return is_new

    async def bulk_upsert(
        self,
        documents: List[NewsArticleDocument],
    ) -> Dict[str, int]:
        """
        Bulk upsert multiple news articles.

        Efficiently processes large batches using bulk operations.

        Args:
            documents: List of NewsArticleDocument to upsert

        Returns:
            Dict with 'inserted' and 'modified' counts
        """
        if not documents:
            return {"inserted": 0, "modified": 0}

        operations = []
        now = datetime.utcnow()

        for doc in documents:
            data = doc.to_mongo_dict()
            data["updated_at"] = now

            operations.append(
                UpdateOne(
                    {"news_id": doc.news_id},
                    {"$set": data},
                    upsert=True,
                )
            )

        result = await self.collection.bulk_write(operations, ordered=False)

        stats = {
            "inserted": result.upserted_count,
            "modified": result.modified_count,
        }

        logger.info(
            "news_articles_bulk_upserted",
            total=len(documents),
            **stats,
        )

        return stats

    async def get_by_id(self, news_id: str) -> Optional[NewsArticleDocument]:
        """
        Get a news article by its ID.

        Args:
            news_id: Unique news identifier

        Returns:
            NewsArticleDocument if found, None otherwise
        """
        doc = await self.collection.find_one({"news_id": news_id})
        if doc:
            return NewsArticleDocument.from_mongo_dict(doc)
        return None

    async def get_by_ids(self, news_ids: List[str]) -> List[NewsArticleDocument]:
        """
        Get multiple news articles by IDs.

        Args:
            news_ids: List of news IDs

        Returns:
            List of NewsArticleDocument
        """
        cursor = self.collection.find({"news_id": {"$in": news_ids}})
        documents = []
        async for doc in cursor:
            documents.append(NewsArticleDocument.from_mongo_dict(doc))
        return documents

    async def exists(self, news_id: str) -> bool:
        """
        Check if a news article exists.

        Args:
            news_id: Unique news identifier

        Returns:
            True if exists, False otherwise
        """
        count = await self.collection.count_documents(
            {"news_id": news_id},
            limit=1,
        )
        return count > 0

    async def get_existing_ids(self, news_ids: List[str]) -> List[str]:
        """
        Get list of news IDs that already exist in database.

        Used for efficient deduplication during batch processing.

        Args:
            news_ids: List of news IDs to check

        Returns:
            List of IDs that already exist
        """
        cursor = self.collection.find(
            {"news_id": {"$in": news_ids}},
            {"news_id": 1},
        )
        existing = []
        async for doc in cursor:
            existing.append(doc["news_id"])
        return existing

    async def get_recent(
        self,
        hours: int = 4,
        limit: int = 50,
        sentiment: Optional[str] = None,
        analyzed_only: bool = False,
    ) -> List[NewsArticleDocument]:
        """
        Get recent news articles.

        Args:
            hours: How many hours back to fetch
            limit: Maximum number of articles
            sentiment: Filter by sentiment (optional)
            analyzed_only: Only return analyzed articles

        Returns:
            List of NewsArticleDocument sorted by published_at desc
        """
        cutoff = datetime.utcnow() - timedelta(hours=hours)

        query: Dict[str, Any] = {"published_at": {"$gte": cutoff}}

        if sentiment:
            query["sentiment"] = sentiment

        if analyzed_only:
            query["analyzed"] = True

        cursor = self.collection.find(query).sort(
            "published_at", -1
        ).limit(limit)

        documents = []
        async for doc in cursor:
            documents.append(NewsArticleDocument.from_mongo_dict(doc))

        return documents

    async def get_unprocessed(self, limit: int = 100) -> List[NewsArticleDocument]:
        """
        Get news articles that haven't been AI processed.

        Args:
            limit: Maximum number of articles

        Returns:
            List of unprocessed NewsArticleDocument
        """
        cursor = self.collection.find(
            {"analyzed": False}
        ).sort("published_at", -1).limit(limit)

        documents = []
        async for doc in cursor:
            documents.append(NewsArticleDocument.from_mongo_dict(doc))

        return documents

    async def mark_as_analyzed(
        self,
        news_id: str,
        summary: str,
        sentiment: str,
        sentiment_score: float,
        mentioned_stocks: List[str],
        mentioned_sectors: List[str],
        impacted_stocks: List[Dict[str, Any]] = None,
        sector_impacts: Dict[str, str] = None,
        causal_chain: str = "",
    ) -> bool:
        """
        Update a news article with AI analysis results.

        Args:
            news_id: News article ID
            summary: AI-generated summary
            sentiment: Analyzed sentiment
            sentiment_score: Sentiment score (-1 to 1)
            mentioned_stocks: Extracted stock tickers
            mentioned_sectors: Extracted sectors
            impacted_stocks: List of impacted stock dicts
            sector_impacts: Sector impact mapping
            causal_chain: Causal chain explanation

        Returns:
            True if updated successfully
        """
        update_data = {
            "summary": summary,
            "sentiment": sentiment,
            "sentiment_score": sentiment_score,
            "mentioned_stocks": mentioned_stocks,
            "mentioned_sectors": mentioned_sectors,
            "impacted_stocks": impacted_stocks or [],
            "sector_impacts": sector_impacts or {},
            "causal_chain": causal_chain,
            "analyzed": True,
            "updated_at": datetime.utcnow(),
        }

        result = await self.collection.update_one(
            {"news_id": news_id},
            {"$set": update_data},
        )

        return result.modified_count > 0

    async def add_to_snapshot(
        self,
        news_ids: List[str],
        snapshot_id: str,
    ) -> int:
        """
        Mark news articles as included in a snapshot.

        Args:
            news_ids: List of news IDs
            snapshot_id: Snapshot ID to add

        Returns:
            Number of articles updated
        """
        result = await self.collection.update_many(
            {"news_id": {"$in": news_ids}},
            {"$addToSet": {"included_in_snapshots": snapshot_id}},
        )
        return result.modified_count

    async def get_by_stocks(
        self,
        tickers: List[str],
        hours: int = 24,
        limit: int = 20,
    ) -> List[NewsArticleDocument]:
        """
        Get news mentioning specific stocks.

        Args:
            tickers: List of stock tickers
            hours: How many hours back
            limit: Maximum articles

        Returns:
            List of NewsArticleDocument
        """
        cutoff = datetime.utcnow() - timedelta(hours=hours)

        cursor = self.collection.find({
            "mentioned_stocks": {"$in": tickers},
            "published_at": {"$gte": cutoff},
        }).sort("published_at", -1).limit(limit)

        documents = []
        async for doc in cursor:
            documents.append(NewsArticleDocument.from_mongo_dict(doc))

        return documents

    async def count_by_sentiment(self, hours: int = 24) -> Dict[str, int]:
        """
        Get sentiment distribution for recent news.

        Args:
            hours: How many hours back

        Returns:
            Dict with sentiment counts
        """
        cutoff = datetime.utcnow() - timedelta(hours=hours)

        pipeline = [
            {"$match": {"published_at": {"$gte": cutoff}}},
            {"$group": {"_id": "$sentiment", "count": {"$sum": 1}}},
        ]

        result = {"bullish": 0, "bearish": 0, "neutral": 0}


        async for doc in self.collection.aggregate(pipeline):
            sentiment = doc["_id"]
            if sentiment in result:
                result[sentiment] = doc["count"]

        return result

    async def search_by_stocks_and_companies(
        self,
        stocks: Optional[List[str]] = None,
        companies: Optional[List[str]] = None,
        hours: int = 24,
        limit: int = 50,
    ) -> List[NewsArticleDocument]:
        """
        Search news mentioning specific stocks and/or companies using fuzzy matching.

        Performs OR search with case-insensitive partial matching - returns news mentioning
        ANY of the stocks OR ANY of the companies (supports partial matches).

        Args:
            stocks: List of stock tickers to search for (supports partial matches)
            companies: List of company names to search for (supports partial matches)
            hours: How many hours back to search
            limit: Maximum number of articles

        Returns:
            List of NewsArticleDocument matching the criteria

        Raises:
            ValueError: If both stocks and companies are None/empty
        """
        if not stocks and not companies:
            raise ValueError("At least one stock ticker or company name must be provided")

        # Build the query with OR logic for stocks and companies using regex for fuzzy matching
        or_conditions = []

        if stocks:
            # Filter out empty strings
            stocks = [s.strip().upper() for s in stocks if s.strip()]
            if stocks:
                # Use regex for case-insensitive partial matching on each stock
                stock_conditions = [{"mentioned_stocks": {"$regex": stock, "$options": "i"}} for stock in stocks]
                or_conditions.extend(stock_conditions)

        if companies:
            # Filter out empty strings
            companies = [c.strip() for c in companies if c.strip()]
            if companies:
                # Use regex for case-insensitive partial matching on each company
                company_conditions = [{"mentioned_companies": {"$regex": company, "$options": "i"}} for company in companies]
                or_conditions.extend(company_conditions)

        if not or_conditions:
            raise ValueError("No valid stocks or companies provided after filtering")

        # Build base query with date filter
        cutoff = datetime.utcnow() - timedelta(hours=hours)
        query: Dict[str, Any] = {
            "published_at": {"$gte": cutoff},
            "$or": or_conditions,
        }

        # Execute query sorted by published_at descending
        cursor = self.collection.find(query).sort(
            "published_at", -1
        ).limit(limit)

        documents = []
        async for doc in cursor:
            documents.append(NewsArticleDocument.from_mongo_dict(doc))

        return documents
# Singleton instance
_news_repository: Optional[NewsRepository] = None


def get_news_repository() -> NewsRepository:
    """Get singleton NewsRepository instance."""
    global _news_repository
    if _news_repository is None:
        _news_repository = NewsRepository()
    return _news_repository


def reset_news_repository() -> None:
    """Reset the news repository singleton (for Celery tasks)."""
    global _news_repository
    _news_repository = None

