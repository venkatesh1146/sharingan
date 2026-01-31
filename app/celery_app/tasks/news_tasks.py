"""
News Tasks - Celery tasks for news fetching and processing.

Handles:
- Periodic news fetching from CMOTS API
- AI-powered news analysis
- Deduplication and storage
"""

import asyncio
from typing import Any, Dict, List, Optional

from celery import shared_task
from celery.utils.log import get_task_logger

from app.celery_app.celery_config import celery_app
from app.config import get_settings

logger = get_task_logger(__name__)
settings = get_settings()


def run_async(coro):
    """Helper to run async code in sync Celery tasks."""
    # Reset MongoDB client to bind to fresh event loop
    from app.db.mongodb import reset_mongodb_client
    reset_mongodb_client()
    
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(coro)
    finally:
        # Clean up MongoDB connections before closing loop
        try:
            from app.db.mongodb import get_mongodb_client
            mongo_client = get_mongodb_client()
            if mongo_client._client is not None:
                loop.run_until_complete(mongo_client.disconnect())
        except Exception:
            pass
        loop.close()


@celery_app.task(
    name="app.celery_app.tasks.news_tasks.fetch_and_process_news",
    bind=True,
    max_retries=3,
    default_retry_delay=60,
)
def fetch_and_process_news(
    self, limit: int = 20, market_phase: Optional[str] = None
) -> Dict[str, Any]:
    """
    Fetch news from all sources, deduplicate, and process.
    
    Runs every 15 minutes via Celery Beat.
    
    Args:
        limit: Number of items per news type to fetch
        market_phase: Optional phase (pre/mid/post) to fetch phase-specific news only.
            If not provided, fetches from all sources.
        
    Returns:
        Dict with processing statistics
    """
    logger.info(f"fetch_and_process_news_started: limit={limit}, market_phase={market_phase}")
    
    try:
        result = run_async(_fetch_and_process_news_async(limit, market_phase))
        logger.info(f"fetch_and_process_news_completed: {result}")
        return result
        
    except Exception as exc:
        logger.error(
            f"fetch_and_process_news_failed: error={exc}, retry_count={self.request.retries}"
        )
        raise self.retry(exc=exc)


async def _fetch_and_process_news_async(
    limit: int, market_phase: Optional[str] = None
) -> Dict[str, Any]:
    """Async implementation of news fetching and processing."""
    from app.db.mongodb import get_mongodb_client
    from app.db.repositories.news_repository import get_news_repository
    from app.db.models.news_document import NewsArticleDocument
    from app.services.cmots_news_service import get_cmots_news_service
    from app.services.market_intelligence_service import get_market_phase
    
    # Connect to MongoDB
    mongo_client = get_mongodb_client()
    await mongo_client.connect()
    
    news_repo = get_news_repository()
    news_service = get_cmots_news_service()
    
    stats = {
        "fetched": 0,
        "new_articles": 0,
        "duplicates_skipped": 0,
        "processed": 0,
        "errors": 0,
    }
    
    # Use provided market_phase or calculate from current time
    if not market_phase:
        phase_data = await get_market_phase()
        market_phase = phase_data["phase"]
    
    try:
        # Fetch from all sources or phase-specific (market_phase: pre/mid/post)
        response = await news_service.fetch_unified_market_news(
            limit=limit,
            page=1,
            per_page=limit,
            market_phase=market_phase,
        )
        
        data_by_type = response.get("data", {})
        
        # Collect all news items
        all_items = []
        for type_key, items in data_by_type.items():
            for item in items:
                item["_news_type"] = type_key
                all_items.append(item)
        
        stats["fetched"] = len(all_items)
        
        if not all_items:
            return stats
        
        # Get existing IDs for deduplication
        news_ids = [str(item.get("sno", "")) for item in all_items]
        existing_ids = await news_repo.get_existing_ids(news_ids)
        existing_set = set(existing_ids)
        
        # Filter new items
        new_items = [
            item for item in all_items
            if str(item.get("sno", "")) not in existing_set
        ]
        
        stats["duplicates_skipped"] = len(all_items) - len(new_items)
        stats["new_articles"] = len(new_items)
        
        if not new_items:
            return stats
        
        # Convert to documents and store
        documents = []
        for item in new_items:
            try:
                doc = NewsArticleDocument.from_cmots_news(
                    item,
                    news_type=item.get("_news_type", ""),
                )
                # Set basic fields from the API response
                original_summary = item.get("summary", "")
                doc.summary = original_summary
                doc.full_text = original_summary  # Preserve original source summary
                doc.processed = True  # Basic processing done
                documents.append(doc)
            except Exception as e:
                logger.warning(
                    f"news_document_conversion_failed: sno={item.get('sno')}, error={e}"
                )
                stats["errors"] += 1
        
        # Bulk upsert to MongoDB
        if documents:
            result = await news_repo.bulk_upsert(documents)
            stats["processed"] = result.get("inserted", 0) + result.get("modified", 0)
        
        # Queue AI analysis for new articles
        if documents:
            process_news_batch.delay([doc.news_id for doc in documents])
        
        return stats
        
    except Exception as e:
        logger.error(f"news_fetch_async_error: error={e}")
        raise


@celery_app.task(
    name="app.celery_app.tasks.news_tasks.process_news_batch",
    bind=True,
    max_retries=2,
)
def process_news_batch(self, news_ids: List[str]) -> Dict[str, Any]:
    """
    Process a batch of news articles with AI analysis.
    
    Args:
        news_ids: List of news IDs to process
        
    Returns:
        Dict with processing statistics
    """
    logger.info(f"process_news_batch_started: count={len(news_ids)}")
    
    try:
        result = run_async(_process_news_batch_async(news_ids))
        logger.info(f"process_news_batch_completed: {result}")
        return result
        
    except Exception as exc:
        logger.error(f"process_news_batch_failed: error={exc}")
        raise self.retry(exc=exc)


async def _process_news_batch_async(news_ids: List[str]) -> Dict[str, Any]:
    """Async implementation of news batch processing with AI using NewsProcessingAgent."""
    from app.db.mongodb import get_mongodb_client
    from app.db.repositories.news_repository import get_news_repository
    from app.agents import get_news_processing_agent
    
    # Connect to MongoDB
    mongo_client = get_mongodb_client()
    await mongo_client.connect()
    
    news_repo = get_news_repository()
    
    stats = {
        "processed": 0,
        "errors": 0,
    }
    
    try:
        # Get unanalyzed articles
        documents = await news_repo.get_by_ids(news_ids)
        unanalyzed = [doc for doc in documents if not doc.analyzed]
        
        if not unanalyzed:
            return stats
        
        # Process with NewsProcessingAgent (AI-powered)
        agent = get_news_processing_agent()
        
        for doc in unanalyzed:
            try:
                result = await agent.analyze_article(doc)
                
                if result:
                    await news_repo.mark_as_analyzed(
                        news_id=doc.news_id,
                        summary=result.get("summary", doc.summary),
                        sentiment=result.get("sentiment", "neutral"),
                        sentiment_score=result.get("sentiment_score", 0.0),
                        mentioned_stocks=result.get("mentioned_stocks", []),
                        mentioned_sectors=result.get("mentioned_sectors", []),
                        impacted_stocks=result.get("impacted_stocks", []),
                        sector_impacts=result.get("sector_impacts", {}),
                        causal_chain=result.get("causal_chain", ""),
                    )
                    stats["processed"] += 1
                    
            except Exception as e:
                logger.warning(
                    f"news_analysis_failed: news_id={doc.news_id}, error={e}"
                )
                stats["errors"] += 1
        
        return stats
        
    except Exception as e:
        logger.error(f"process_batch_async_error: error={e}")
        raise


@celery_app.task(
    name="app.celery_app.tasks.news_tasks.process_single_article",
    bind=True,
    max_retries=2,
)
def process_single_article(self, news_id: str) -> Dict[str, Any]:
    """
    Process a single news article with AI analysis.
    
    Args:
        news_id: News article ID to process
        
    Returns:
        Dict with processing result
    """
    logger.info(f"process_single_article_started: news_id={news_id}")
    
    try:
        result = run_async(_process_single_article_async(news_id))
        logger.info(f"process_single_article_completed: news_id={news_id}")
        return result
        
    except Exception as exc:
        logger.error(
            f"process_single_article_failed: news_id={news_id}, error={exc}"
        )
        raise self.retry(exc=exc)


async def _process_single_article_async(news_id: str) -> Dict[str, Any]:
    """Async implementation of single article processing using NewsProcessingAgent."""
    from app.db.mongodb import get_mongodb_client
    from app.db.repositories.news_repository import get_news_repository
    from app.agents import get_news_processing_agent
    
    mongo_client = get_mongodb_client()
    await mongo_client.connect()
    
    news_repo = get_news_repository()
    
    # Get the article
    doc = await news_repo.get_by_id(news_id)
    if not doc:
        return {"status": "not_found", "news_id": news_id}
    
    if doc.analyzed:
        return {"status": "already_analyzed", "news_id": news_id}
    
    # Process with NewsProcessingAgent (AI-powered)
    agent = get_news_processing_agent()
    result = await agent.analyze_article(doc)
    
    if result:
        await news_repo.mark_as_analyzed(
            news_id=doc.news_id,
            summary=result.get("summary", doc.summary),
            sentiment=result.get("sentiment", "neutral"),
            sentiment_score=result.get("sentiment_score", 0.0),
            mentioned_stocks=result.get("mentioned_stocks", []),
            mentioned_sectors=result.get("mentioned_sectors", []),
            impacted_stocks=result.get("impacted_stocks", []),
            sector_impacts=result.get("sector_impacts", {}),
            causal_chain=result.get("causal_chain", ""),
        )
        return {"status": "success", "news_id": news_id}
    
    return {"status": "analysis_failed", "news_id": news_id}
