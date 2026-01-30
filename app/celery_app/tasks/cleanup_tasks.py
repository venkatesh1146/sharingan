"""
Cleanup Tasks - Celery tasks for data maintenance.

Handles:
- Expired snapshot cleanup
- Old news archival
- Indices data retention
- Redis cache cleanup
"""

import asyncio
from datetime import datetime, timedelta
from typing import Any, Dict

from celery import shared_task
from celery.utils.log import get_task_logger

from app.celery_app.celery_config import celery_app
from app.config import get_settings

logger = get_task_logger(__name__)
settings = get_settings()


def run_async(coro):
    """Helper to run async code in sync Celery tasks."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


@celery_app.task(
    name="app.celery_app.tasks.cleanup_tasks.cleanup_old_data",
    bind=True,
)
def cleanup_old_data(self) -> Dict[str, Any]:
    """
    Clean up old data from all collections.
    
    Runs daily at 2 AM via Celery Beat.
    
    Returns:
        Dict with cleanup statistics
    """
    # Check if cleanup tasks are paused
    if settings.PAUSE_CLEANUP_TASKS:
        logger.info("cleanup_old_data_skipped: PAUSE_CLEANUP_TASKS is enabled")
        return {"status": "skipped", "reason": "cleanup_tasks_paused"}
    
    logger.info("cleanup_old_data_started")
    
    try:
        result = run_async(_cleanup_async())
        logger.info(f"cleanup_old_data_completed: {result}")
        return result
        
    except Exception as exc:
        logger.error(f"cleanup_old_data_failed: error={exc}")
        # Don't retry cleanup tasks - they'll run again tomorrow
        return {"status": "error", "error": str(exc)}


async def _cleanup_async() -> Dict[str, Any]:
    """Async implementation of data cleanup."""
    from app.db.mongodb import get_mongodb_client
    from app.db.repositories.indices_repository import get_indices_repository
    from app.services.redis_service import RedisService
    
    # Connect to MongoDB
    mongo_client = get_mongodb_client()
    await mongo_client.connect()
    
    stats = {
        "indices_deleted": 0,
        "redis_keys_deleted": 0,
        "errors": [],
    }
    
    # Note: Snapshot cleanup removed - MongoDB TTL index handles expiration automatically
    
    # Clean up old indices data (TTL handles this, but manual backup)
    try:
        indices_repo = get_indices_repository()
        stats["indices_deleted"] = await indices_repo.cleanup_old_data(
            days=settings.INDICES_RETENTION_DAYS
        )
    except Exception as e:
        logger.warning(f"indices_cleanup_failed: error={e}")
        stats["errors"].append(f"indices: {str(e)}")
    
    # Clean up Redis cache (old patterns)
    try:
        redis_service = RedisService()
        # Clear old news cache patterns
        stats["redis_keys_deleted"] = await redis_service.clear_pattern(
            f"{settings.NEWS_CACHE_PREFIX}:*"
        )
    except Exception as e:
        logger.warning(f"redis_cleanup_failed: error={e}")
        stats["errors"].append(f"redis: {str(e)}")
    
    stats["status"] = "success" if not stats["errors"] else "partial"
    return stats


@celery_app.task(
    name="app.celery_app.tasks.cleanup_tasks.cleanup_redis_cache",
    bind=True,
)
def cleanup_redis_cache(self, pattern: str = "*") -> Dict[str, Any]:
    """
    Clean up Redis cache by pattern.
    
    Args:
        pattern: Redis key pattern to clean
        
    Returns:
        Dict with cleanup statistics
    """
    # Check if cleanup tasks are paused
    if settings.PAUSE_CLEANUP_TASKS:
        logger.info("cleanup_redis_cache_skipped: PAUSE_CLEANUP_TASKS is enabled")
        return {"status": "skipped", "reason": "cleanup_tasks_paused"}
    
    logger.info(f"cleanup_redis_cache_started: pattern={pattern}")
    
    try:
        result = run_async(_cleanup_redis_async(pattern))
        logger.info(f"cleanup_redis_cache_completed: {result}")
        return result
        
    except Exception as exc:
        logger.error(f"cleanup_redis_cache_failed: error={exc}")
        return {"status": "error", "error": str(exc)}


async def _cleanup_redis_async(pattern: str) -> Dict[str, Any]:
    """Async implementation of Redis cleanup."""
    from app.services.redis_service import RedisService
    
    redis_service = RedisService()
    deleted = await redis_service.clear_pattern(pattern)
    
    return {
        "status": "success",
        "pattern": pattern,
        "keys_deleted": deleted,
    }


@celery_app.task(
    name="app.celery_app.tasks.cleanup_tasks.archive_old_news",
    bind=True,
)
def archive_old_news(self, days: int = 90) -> Dict[str, Any]:
    """
    Archive old news articles.
    
    Moves old news to archive collection or marks for deletion.
    
    Args:
        days: News older than this will be archived
        
    Returns:
        Dict with archive statistics
    """
    # Check if cleanup tasks are paused
    if settings.PAUSE_CLEANUP_TASKS:
        logger.info("archive_old_news_skipped: PAUSE_CLEANUP_TASKS is enabled")
        return {"status": "skipped", "reason": "cleanup_tasks_paused"}
    
    logger.info(f"archive_old_news_started: days={days}")
    
    try:
        result = run_async(_archive_news_async(days))
        logger.info(f"archive_old_news_completed: {result}")
        return result
        
    except Exception as exc:
        logger.error(f"archive_old_news_failed: error={exc}")
        return {"status": "error", "error": str(exc)}


async def _archive_news_async(days: int) -> Dict[str, Any]:
    """Async implementation of news archival."""
    from app.db.mongodb import get_mongodb_client
    
    mongo_client = get_mongodb_client()
    await mongo_client.connect()
    
    db = mongo_client.get_database()
    news_collection = db["news_articles"]
    
    cutoff = datetime.utcnow() - timedelta(days=days)
    
    # For now, we'll just count old articles
    # In production, you might move to an archive collection
    old_count = await news_collection.count_documents({
        "published_at": {"$lt": cutoff},
    })
    
    # Option 1: Delete old news (if not needed)
    # result = await news_collection.delete_many({"published_at": {"$lt": cutoff}})
    
    # Option 2: Mark as archived (soft delete)
    result = await news_collection.update_many(
        {"published_at": {"$lt": cutoff}, "archived": {"$ne": True}},
        {"$set": {"archived": True, "archived_at": datetime.utcnow()}},
    )
    
    return {
        "status": "success",
        "cutoff_date": cutoff.isoformat(),
        "old_articles_found": old_count,
        "articles_archived": result.modified_count,
    }


@celery_app.task(
    name="app.celery_app.tasks.cleanup_tasks.get_cleanup_stats",
    bind=True,
)
def get_cleanup_stats(self) -> Dict[str, Any]:
    """
    Get statistics about data that needs cleanup.
    
    Returns:
        Dict with data statistics
    """
    logger.info("get_cleanup_stats_started")
    
    try:
        result = run_async(_get_stats_async())
        return result
        
    except Exception as exc:
        logger.error(f"get_cleanup_stats_failed: error={exc}")
        return {"status": "error", "error": str(exc)}


async def _get_stats_async() -> Dict[str, Any]:
    """Async implementation of getting cleanup stats."""
    from app.db.mongodb import get_mongodb_client
    from app.db.repositories.indices_repository import get_indices_repository
    
    mongo_client = get_mongodb_client()
    await mongo_client.connect()
    
    db = mongo_client.get_database()
    
    # Get collection stats
    news_count = await db["news_articles"].count_documents({})
    snapshots_count = await db["market_snapshots"].count_documents({})
    
    # Get indices stats
    indices_repo = get_indices_repository()
    indices_stats = await indices_repo.get_stats()
    
    # Get expired counts
    now = datetime.utcnow()
    expired_snapshots = await db["market_snapshots"].count_documents({
        "expires_at": {"$lt": now},
    })
    
    # Get old news count
    cutoff = now - timedelta(days=settings.NEWS_RETENTION_DAYS)
    old_news = await db["news_articles"].count_documents({
        "published_at": {"$lt": cutoff},
    })
    
    return {
        "status": "success",
        "collections": {
            "news_articles": {
                "total": news_count,
                "older_than_retention": old_news,
            },
            "market_snapshots": {
                "total": snapshots_count,
                "expired": expired_snapshots,
            },
            "indices_timeseries": indices_stats,
        },
        "timestamp": now.isoformat(),
    }
