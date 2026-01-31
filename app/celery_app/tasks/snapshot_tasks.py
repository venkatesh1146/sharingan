"""
Snapshot Tasks - Celery tasks for market snapshot generation.

Handles:
- Periodic snapshot generation
- On-demand snapshot creation
- Snapshot expiration management
"""

import asyncio
from datetime import datetime, timedelta
from typing import Any, Dict, Optional

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
    name="app.celery_app.tasks.snapshot_tasks.generate_market_snapshot",
    bind=True,
    max_retries=2,
    default_retry_delay=30,
)
def generate_market_snapshot(
    self,
    market_phase: Optional[str] = None,
    force: bool = False,
) -> Dict[str, Any]:
    """
    Generate aggregated market snapshot.
    
    Runs every 30 minutes via Celery Beat or on-demand.
    
    Args:
        market_phase: Optional phase (pre/mid/post). If not provided, calculated from current time (IST).
        force: Force generation even if recent snapshot exists
        
    Returns:
        Dict with snapshot ID and generation stats
    """
    logger.info(
        f"generate_market_snapshot_started: market_phase={market_phase}, force={force}"
    )
    
    try:
        result = run_async(_generate_snapshot_async(market_phase, force))
        logger.info(f"generate_market_snapshot_completed: {result}")
        return result
        
    except Exception as exc:
        logger.error(
            f"generate_market_snapshot_failed: error={exc}, retry_count={self.request.retries}"
        )
        raise self.retry(exc=exc)


async def _generate_snapshot_async(
    market_phase: Optional[str],
    force: bool,
) -> Dict[str, Any]:
    """Async implementation of snapshot generation using SnapshotGenerationAgent."""
    from app.db.mongodb import get_mongodb_client
    from app.db.repositories.snapshot_repository import get_snapshot_repository
    from app.db.repositories.news_repository import get_news_repository
    from app.db.models.snapshot_document import (
        MarketSnapshotDocument,
        MarketOutlookDocument,
        IndexDataDocument,
        MarketSummaryBulletDocument,
        ThemedItemDocument,
    )
    from app.services.market_intelligence_service import (
        get_market_phase,
        fetch_market_indices,
        fetch_phase_specific_news,
        _filter_indices_by_phase,
        phase_news_to_documents,
        build_themed_from_news,
    )
    from app.agents import get_snapshot_generation_agent
    
    # Connect to MongoDB
    mongo_client = get_mongodb_client()
    await mongo_client.connect()
    
    snapshot_repo = get_snapshot_repository()
    news_repo = get_news_repository()
    
    # Use provided market_phase or calculate from current time (IST)
    if not market_phase:
        phase_data = await get_market_phase()
        market_phase = phase_data["phase"]
    
    # When phase changes: generate new snapshot if latest current-phase snapshot
    # does not exist (we generate below when no valid snapshot for this phase).
    latest_any = await snapshot_repo.get_latest()
    latest_valid_current = await snapshot_repo.get_latest_valid(market_phase)
    if latest_any and not latest_valid_current and latest_any.market_phase != market_phase:
        logger.info(
            f"phase_change_detected: previous_phase={latest_any.market_phase}, "
            f"current_phase={market_phase}, generating_new_snapshot"
        )
    
    # Check if we need to generate (unless forced)
    if not force:
        is_stale = await snapshot_repo.is_stale(
            market_phase,
            max_age_seconds=settings.SNAPSHOT_TTL_SECONDS,
        )
        if not is_stale:
            latest = await snapshot_repo.get_latest_valid(market_phase)
            if latest:
                return {
                    "status": "skipped",
                    "reason": "recent_snapshot_exists",
                    "snapshot_id": latest.snapshot_id,
                    "market_phase": market_phase,
                }
    
    # Fetch indices and filter by market phase (pre/mid/post)
    all_indices_data = await fetch_market_indices()
    indices_data = _filter_indices_by_phase(all_indices_data, market_phase)
    
    # Fetch phase-specific news (pre/mid/post market commentary) for snapshot
    phase_news_raw = await fetch_phase_specific_news(
        phase=market_phase,
        max_articles=50,
    )
    phase_news_docs = phase_news_to_documents(phase_news_raw)
    if phase_news_docs:
        recent_news = phase_news_docs
        logger.info(
            f"using_phase_specific_news: phase={market_phase}, count={len(recent_news)}"
        )
    else:
        # Fallback to recent news from DB if phase-specific fetch is empty
        recent_news = await news_repo.get_recent(hours=4, limit=50, analyzed_only=True)
        logger.info(
            f"phase_specific_news_empty_using_db: phase={market_phase}, count={len(recent_news)}"
        )
    
    # Fetch previous snapshot for context continuity
    previous_snapshot = None
    previous_snapshot_doc = await snapshot_repo.get_latest(market_phase)
    if previous_snapshot_doc:
        # Convert to dict format for the prompt
        previous_snapshot = {
            "market_outlook": previous_snapshot_doc.market_outlook.model_dump() 
                if previous_snapshot_doc.market_outlook else None,
            "market_summary": [
                bullet.model_dump() for bullet in (previous_snapshot_doc.market_summary or [])
            ],
            "executive_summary": previous_snapshot_doc.executive_summary,
        }
        logger.info(
            f"previous_snapshot_found: phase={market_phase}, "
            f"snapshot_id={previous_snapshot_doc.snapshot_id}"
        )
    
    # Generate snapshot content using SnapshotGenerationAgent (AI-powered)
    agent = get_snapshot_generation_agent()
    snapshot_content = await agent.generate_snapshot_content(
        market_phase=market_phase,
        indices_data=indices_data,
        news_items=recent_news,
        previous_snapshot=previous_snapshot,
    )
    
    # Create snapshot document
    snapshot_id = MarketSnapshotDocument.generate_snapshot_id(market_phase)
    
    # Convert indices to document format
    indices_docs = []
    for ticker, data in indices_data.items():
        if "error" not in data:
            indices_docs.append(IndexDataDocument(
                ticker=ticker,
                name=data.get("name", ticker),
                country=data.get("country", "Unknown"),
                current_price=data["current_price"],
                change_percent=data["change_percent"],
                change_absolute=data["change_absolute"],
                previous_close=data.get("previous_close", 0),
                intraday_high=data.get("intraday_high", data["current_price"]),
                intraday_low=data.get("intraday_low", data["current_price"]),
                volume=data.get("volume", 0),
                timestamp=datetime.fromisoformat(data["timestamp"])
                if isinstance(data.get("timestamp"), str)
                else datetime.utcnow(),
            ))
    
    # Build market outlook (all phases: pre, mid, post)
    market_outlook = None
    if snapshot_content.get("market_outlook"):
        outlook = snapshot_content["market_outlook"]
        market_outlook = MarketOutlookDocument(
            sentiment=outlook.get("sentiment", "neutral"),
            confidence=outlook.get("confidence", 0.5),
            reasoning=outlook.get("reasoning", ""),
            nifty_change_percent=outlook.get("nifty_change_percent", 0.0),
            key_drivers=outlook.get("key_drivers", []),
        )
    
    # Build market summary bullets
    summary_bullets = []
    for bullet in snapshot_content.get("market_summary", []):
        summary_bullets.append(MarketSummaryBulletDocument(
            text=bullet.get("text", ""),
            supporting_news_ids=bullet.get("supporting_news_ids", []),
            confidence=bullet.get("confidence", 0.7),
            sentiment=bullet.get("sentiment", "neutral"),
        ))
    
    # Themed: use AI-generated themed from snapshot content when present, else build from news
    themed_raw = snapshot_content.get("themed")
    if themed_raw and isinstance(themed_raw, list):
        themed_docs = []
        for t in themed_raw:
            if isinstance(t, dict) and t.get("sector"):
                themed_docs.append(ThemedItemDocument(
                    sector=str(t["sector"]),
                    relevant_companies=list(t.get("relevant_companies", [])) if isinstance(t.get("relevant_companies"), list) else [],
                    sentiment=t.get("sentiment", "neutral"),
                    sentiment_score=t.get("sentiment_score"),
                ))
    else:
        themed_raw = build_themed_from_news(recent_news)
        themed_docs = [ThemedItemDocument(**t) for t in themed_raw]
    
    # Create snapshot document
    snapshot = MarketSnapshotDocument(
        snapshot_id=snapshot_id,
        market_phase=market_phase,
        market_outlook=market_outlook,
        indices_data=indices_docs,
        market_summary=summary_bullets,
        executive_summary=snapshot_content.get("executive_summary"),
        trending_now=snapshot_content.get("trending_now") if market_phase == "mid" else None,
        themed=themed_docs,
        all_news_ids=[n.news_id for n in recent_news],
        expires_at=datetime.utcnow() + timedelta(seconds=settings.SNAPSHOT_TTL_SECONDS),
    )
    
    # Save to MongoDB
    await snapshot_repo.create(snapshot)
    
    # Update news articles with snapshot reference
    if recent_news:
        await news_repo.add_to_snapshot(
            news_ids=[n.news_id for n in recent_news],
            snapshot_id=snapshot_id,
        )
    
    return {
        "status": "success",
        "snapshot_id": snapshot_id,
        "market_phase": market_phase,
        "indices_count": len(indices_docs),
        "news_count": len(recent_news),
        "summary_bullets": len(summary_bullets),
    }


def _create_basic_snapshot_content(
    market_phase: str,
    indices_data: Dict[str, Any],
    news_items: list,
) -> Dict[str, Any]:
    """Create basic snapshot content without AI."""
    content = {
        "market_summary": [],
        "executive_summary": "Market data available. AI analysis temporarily unavailable.",
    }
    
    # Calculate basic outlook from NIFTY
    nifty = indices_data.get("NIFTY", indices_data.get("SENSEX", {}))
    if nifty:
        change = nifty.get("change_percent", 0)
        if change > 0.5:
            sentiment = "bullish"
        elif change < -0.5:
            sentiment = "bearish"
        else:
            sentiment = "neutral"
        
        content["market_outlook"] = {
            "sentiment": sentiment,
            "confidence": 0.6,
            "reasoning": f"Based on index movement of {change:.2f}%",
            "nifty_change_percent": change,
            "key_drivers": ["Index movement"],
        }
    
    if market_phase == "mid":
        content["trending_now"] = [n.news_id for n in news_items[:5]]
    
    return content


@celery_app.task(
    name="app.celery_app.tasks.snapshot_tasks.get_or_generate_snapshot",
    bind=True,
)
def get_or_generate_snapshot(
    self,
    market_phase: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Get latest valid snapshot or generate new one.
    
    Used by API endpoints for fast response with fallback to generation.
    
    Args:
        market_phase: Optional phase filter
        
    Returns:
        Dict with snapshot data
    """
    logger.info(f"get_or_generate_snapshot_started: market_phase={market_phase}")
    
    try:
        result = run_async(_get_or_generate_async(market_phase))
        return result
    except Exception as exc:
        logger.error(f"get_or_generate_snapshot_failed: error={exc}")
        raise


async def _get_or_generate_async(market_phase: Optional[str]) -> Dict[str, Any]:
    """Async implementation of get or generate."""
    from app.db.mongodb import get_mongodb_client
    from app.db.repositories.snapshot_repository import get_snapshot_repository
    from app.services.market_intelligence_service import get_market_phase
    
    mongo_client = get_mongodb_client()
    await mongo_client.connect()
    
    snapshot_repo = get_snapshot_repository()
    
    # Determine market phase if not provided
    if not market_phase:
        phase_data = await get_market_phase()
        market_phase = phase_data["phase"]
    
    # Try to get latest valid snapshot
    snapshot = await snapshot_repo.get_latest_valid(market_phase)
    
    if snapshot:
        return {
            "status": "cached",
            "snapshot": snapshot.model_dump(),
        }
    
    # No valid snapshot, trigger generation
    generate_market_snapshot.delay(market_phase=market_phase, force=True)
    
    # Return stale data if available
    stale_snapshot = await snapshot_repo.get_latest(market_phase)
    if stale_snapshot:
        return {
            "status": "stale",
            "snapshot": stale_snapshot.model_dump(),
            "message": "New snapshot being generated",
        }
    
    return {
        "status": "generating",
        "snapshot": None,
        "message": "Snapshot generation in progress",
    }
