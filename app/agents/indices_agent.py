"""
Indices Collection Agent - Background indices data collection and storage.

This agent is invoked by Celery tasks for:
- Fetching world indices data from CMOTS API
- Storing historical indices in MongoDB
- Market hours awareness

Part of the 3-agent background processing architecture:
1. NewsProcessingAgent - AI news analysis (called by fetch_news task)
2. SnapshotGenerationAgent - AI snapshot generation (called by gen_snapshot task)
3. IndicesCollectionAgent - Indices data collection (called by fetch_indices task)
"""

from datetime import datetime, time
from typing import Any, Dict, List, Optional

import pytz

from app.config import get_settings
from app.db.models.indices_document import IndicesTimeseriesDocument
from app.db.repositories.indices_repository import get_indices_repository
from app.services.cmots_news_service import fetch_world_indices
from app.utils.logging import get_logger

logger = get_logger(__name__)
settings = get_settings()

IST = pytz.timezone("Asia/Kolkata")


class IndicesCollectionAgent:
    """
    Background agent for indices data collection.
    
    Responsibilities:
    - Fetch world indices data from CMOTS API
    - Store indices data in MongoDB timeseries collection
    - Check market hours for scheduling optimization
    - Provide latest indices data for snapshots
    
    This agent manages the indices data pipeline for the
    background processing architecture.
    """
    
    def __init__(self):
        self.logger = get_logger("indices_collection_agent")
    
    def is_market_hours(self) -> bool:
        """
        Check if current time is within market hours (IST).
        
        Market hours: 9:15 AM to 3:30 PM IST, weekdays only.
        
        Returns:
            True if within market hours
        """
        now = datetime.now(IST)
        current_time = now.time()
        
        market_open = time(
            settings.MARKET_OPEN_HOUR,
            settings.MARKET_OPEN_MINUTE,
        )
        market_close = time(
            settings.MARKET_CLOSE_HOUR,
            settings.MARKET_CLOSE_MINUTE,
        )
        
        is_weekday = now.weekday() < 5
        
        return is_weekday and market_open <= current_time <= market_close
    
    async def fetch_and_store_indices(
        self,
        force: bool = False,
    ) -> Dict[str, Any]:
        """
        Fetch indices data from API and store in MongoDB.
        
        Args:
            force: Force fetch even outside market hours
            
        Returns:
            Dict with fetch statistics:
            - status: skipped/success/error
            - fetched: Number of indices fetched
            - stored: Number of documents stored
            - errors: Number of conversion errors
        """
        # Skip if not market hours (unless forced)
        if not force and not self.is_market_hours():
            self.logger.debug("fetch_skipped_non_market_hours")
            return {
                "status": "skipped",
                "reason": "non_market_hours",
            }
        
        self.logger.info("fetch_indices_started")
        
        stats = {
            "status": "success",
            "fetched": 0,
            "stored": 0,
            "errors": 0,
        }
        
        try:
            # Fetch from API
            response = await fetch_world_indices()
            raw_indices = response.get("data", [])
            
            stats["fetched"] = len(raw_indices)
            
            if not raw_indices:
                self.logger.warning("no_indices_data_fetched")
                return stats
            
            # Convert to documents
            documents = []
            for idx_data in raw_indices:
                try:
                    ticker = idx_data.get("indexname", "Unknown").upper()
                    # Normalize ticker names
                    ticker_map = {
                        "NIFTY": "NIFTY",
                        "BSE SENSEX": "SENSEX",
                        "GIFT NIFTY": "GIFT_NIFTY",
                    }
                    normalized = ticker_map.get(ticker, ticker)
                    
                    doc = IndicesTimeseriesDocument.from_world_indices_response(
                        ticker=normalized,
                        index_data=idx_data,
                    )
                    documents.append(doc)
                    
                except Exception as e:
                    self.logger.warning(
                        "indices_conversion_failed",
                        ticker=idx_data.get("indexname"),
                        error=str(e),
                    )
                    stats["errors"] += 1
            
            # Bulk insert to MongoDB
            if documents:
                indices_repo = get_indices_repository()
                stats["stored"] = await indices_repo.bulk_insert(documents)
            
            self.logger.info(
                "fetch_indices_completed",
                fetched=stats["fetched"],
                stored=stats["stored"],
            )
            
            return stats
            
        except Exception as e:
            self.logger.error("fetch_indices_error", error=str(e))
            stats["status"] = "error"
            stats["error"] = str(e)
            return stats
    
    async def get_latest_indices(self) -> Dict[str, Any]:
        """
        Get latest indices data from database.
        
        Returns:
            Dict mapping ticker to latest data
        """
        indices_repo = get_indices_repository()
        latest = await indices_repo.get_latest_all()
        
        result = {}
        for ticker, doc in latest.items():
            result[ticker] = {
                "ticker": doc.ticker,
                "timestamp": doc.timestamp.isoformat(),
                **doc.data.model_dump(),
            }
        
        return result
    
    async def get_index_history(
        self,
        ticker: str,
        hours: int = 24,
    ) -> List[Dict[str, Any]]:
        """
        Get historical data for a specific index.
        
        Args:
            ticker: Index ticker symbol
            hours: Hours of history
            
        Returns:
            List of historical data points
        """
        indices_repo = get_indices_repository()
        history = await indices_repo.get_history(ticker, hours=hours)
        
        return [
            {
                "timestamp": doc.timestamp.isoformat(),
                **doc.data.model_dump(),
            }
            for doc in history
        ]


# Singleton instance
_indices_collection_agent: Optional[IndicesCollectionAgent] = None


def get_indices_collection_agent() -> IndicesCollectionAgent:
    """Get singleton IndicesCollectionAgent instance."""
    global _indices_collection_agent
    if _indices_collection_agent is None:
        _indices_collection_agent = IndicesCollectionAgent()
    return _indices_collection_agent
