"""
Indices Tasks - Celery tasks for indices data collection.

Handles:
- Periodic indices fetching
- Historical data storage
- Market hours awareness
"""

import asyncio
from datetime import datetime, time
from typing import Any, Dict, List

from celery import shared_task
from celery.utils.log import get_task_logger
import pytz

from app.celery_app.celery_config import celery_app
from app.config import get_settings

logger = get_task_logger(__name__)
settings = get_settings()

IST = pytz.timezone("Asia/Kolkata")


def run_async(coro):
    """Helper to run async code in sync Celery tasks."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


def is_market_hours() -> bool:
    """Check if current time is within market hours (IST)."""
    now = datetime.now(IST)
    current_time = now.time()
    
    # Market hours: 9:15 AM to 3:30 PM IST
    market_open = time(
        settings.MARKET_OPEN_HOUR,
        settings.MARKET_OPEN_MINUTE,
    )
    market_close = time(
        settings.MARKET_CLOSE_HOUR,
        settings.MARKET_CLOSE_MINUTE,
    )
    
    # Also check if it's a weekday
    is_weekday = now.weekday() < 5
    
    return is_weekday and market_open <= current_time <= market_close


@celery_app.task(
    name="app.celery_app.tasks.indices_tasks.fetch_indices_data",
    bind=True,
    max_retries=2,
)
def fetch_indices_data(self, force: bool = False) -> Dict[str, Any]:
    """
    Fetch and store indices data.
    
    Runs every 5 minutes. During market hours, stores historical data.
    Outside market hours, still fetches but with lower priority.
    
    Args:
        force: Force fetch even outside market hours
        
    Returns:
        Dict with fetch statistics
    """
    # Skip if not market hours (unless forced)
    if not force and not is_market_hours():
        logger.debug("fetch_indices_skipped_non_market_hours")
        return {
            "status": "skipped",
            "reason": "non_market_hours",
        }
    
    logger.info("fetch_indices_data_started")
    
    try:
        result = run_async(_fetch_indices_async())
        logger.info("fetch_indices_data_completed", **result)
        return result
        
    except Exception as exc:
        logger.error(
            "fetch_indices_data_failed",
            error=str(exc),
            retry_count=self.request.retries,
        )
        raise self.retry(exc=exc)


async def _fetch_indices_async() -> Dict[str, Any]:
    """Async implementation of indices fetching using IndicesCollectionAgent."""
    from app.db.mongodb import get_mongodb_client
    from app.agents import get_indices_collection_agent
    
    # Connect to MongoDB
    mongo_client = get_mongodb_client()
    await mongo_client.connect()
    
    # Use IndicesCollectionAgent for data collection
    agent = get_indices_collection_agent()
    
    # Force fetch (market hours check done in task level)
    result = await agent.fetch_and_store_indices(force=True)
    
    return result


@celery_app.task(
    name="app.celery_app.tasks.indices_tasks.get_latest_indices",
    bind=True,
)
def get_latest_indices(self) -> Dict[str, Any]:
    """
    Get latest indices data from cache or database.
    
    Used by API endpoints for fast access to current indices.
    
    Returns:
        Dict with latest indices data
    """
    logger.info("get_latest_indices_started")
    
    try:
        result = run_async(_get_latest_indices_async())
        return result
    except Exception as exc:
        logger.error(f"get_latest_indices_failed: error={exc}")
        raise


async def _get_latest_indices_async() -> Dict[str, Any]:
    """Async implementation of getting latest indices using IndicesCollectionAgent."""
    from app.db.mongodb import get_mongodb_client
    from app.services.redis_service import RedisService
    from app.agents import get_indices_collection_agent
    
    # Try Redis cache first
    redis_service = RedisService()
    cache_key = "indices:latest"
    
    try:
        cached = await redis_service.get(cache_key)
        if cached:
            return {
                "status": "cached",
                "data": cached,
            }
    except Exception:
        pass  # Cache miss or error, continue to DB
    
    # Connect to MongoDB
    mongo_client = get_mongodb_client()
    await mongo_client.connect()
    
    # Use IndicesCollectionAgent
    agent = get_indices_collection_agent()
    result = await agent.get_latest_indices()
    
    if not result:
        return {
            "status": "empty",
            "data": {},
        }
    
    # Cache the result
    try:
        await redis_service.set(cache_key, result, ttl_seconds=300)
    except Exception:
        pass  # Cache failure is non-critical
    
    return {
        "status": "fresh",
        "data": result,
    }


@celery_app.task(
    name="app.celery_app.tasks.indices_tasks.get_index_history",
    bind=True,
)
def get_index_history(
    self,
    ticker: str,
    hours: int = 24,
) -> Dict[str, Any]:
    """
    Get historical data for a specific index.
    
    Args:
        ticker: Index ticker symbol
        hours: Hours of history to fetch
        
    Returns:
        Dict with historical data points
    """
    logger.info(f"get_index_history_started: ticker={ticker}, hours={hours}")
    
    try:
        result = run_async(_get_index_history_async(ticker, hours))
        return result
    except Exception as exc:
        logger.error(f"get_index_history_failed: ticker={ticker}, error={exc}")
        raise


async def _get_index_history_async(ticker: str, hours: int) -> Dict[str, Any]:
    """Async implementation of getting index history using IndicesCollectionAgent."""
    from app.db.mongodb import get_mongodb_client
    from app.db.repositories.indices_repository import get_indices_repository
    from app.agents import get_indices_collection_agent
    
    mongo_client = get_mongodb_client()
    await mongo_client.connect()
    
    # Use IndicesCollectionAgent for history
    agent = get_indices_collection_agent()
    data_points = await agent.get_index_history(ticker, hours=hours)
    
    if not data_points:
        return {
            "ticker": ticker,
            "data_points": [],
            "count": 0,
        }
    
    # Get change over period from repository
    indices_repo = get_indices_repository()
    change = await indices_repo.get_change_over_period(ticker, hours)
    
    return {
        "ticker": ticker,
        "data_points": data_points,
        "count": len(data_points),
        "period_change": change,
    }
