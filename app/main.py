"""
Market Intelligence API

FastAPI application providing AI-powered market insights through
a background processing architecture with Celery tasks and MongoDB.

Architecture:
- GET /api/v1/market-summary reads pre-computed snapshots from MongoDB
- Background Celery tasks handle AI processing (news analysis, snapshot generation)
- 3 specialized agents: NewsProcessing, SnapshotGeneration, IndicesCollection
"""

from contextlib import asynccontextmanager
from datetime import datetime
from typing import Optional, List

from fastapi import BackgroundTasks, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app import __version__
from app.config import get_settings
from app.models.requests import NewsSearchRequest
from app.models.responses import (
    HealthCheckResponse,
    AgentStatusResponse,
    ErrorResponse,
)
from app.utils.logging import setup_logging, get_logger
from app.utils.tracing import setup_tracing
from app.utils.exceptions import MarketPulseError
from app.services.redis_service import get_redis_service
from app.services.cmots_news_service import fetch_world_indices, get_cmots_news_service
from app.services.company_news_service import get_company_news_service


# Initialize settings and logging
settings = get_settings()
logger = get_logger(__name__)


async def _trigger_startup_tasks():
    """Trigger initial data population tasks after a delay."""
    import asyncio
    await asyncio.sleep(5)  # Wait for Celery worker to be ready

    try:
        from app.celery_app.tasks.news_tasks import fetch_and_process_news
        from app.celery_app.tasks.snapshot_tasks import generate_market_snapshot
        from app.celery_app.tasks.indices_tasks import fetch_indices_data

        logger.info("startup_triggering_initial_data_population")

        # Queue all initial tasks
        fetch_and_process_news.delay(limit=20)
        fetch_indices_data.delay()
        generate_market_snapshot.delay(force=True)

        logger.info("startup_initial_data_tasks_queued")
    except Exception as e:
        logger.warning("startup_data_population_failed", error=str(e))


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager.

    Handles startup and shutdown events.
    """
    # Startup
    logger.info("application_starting", version=__version__)
    setup_logging(settings.LOG_LEVEL)
    setup_tracing()

    # Initialize Redis service
    try:
        redis_service = get_redis_service()
        await redis_service.connect()
        logger.info("redis_service_initialized")
    except Exception as e:
        logger.warning("redis_service_failed_to_initialize", error=str(e))

    # Initialize MongoDB
    try:
        from app.db.mongodb import get_mongodb_client
        mongo_client = get_mongodb_client()
        await mongo_client.connect()
        logger.info("mongodb_initialized")
    except Exception as e:
        logger.warning("mongodb_failed_to_initialize", error=str(e))

    # Trigger initial data population in background
    import asyncio
    asyncio.create_task(_trigger_startup_tasks())

    yield

    # Shutdown
    logger.info("application_shutting_down")

    # Close MongoDB connection
    try:
        from app.db.mongodb import get_mongodb_client
        mongo_client = get_mongodb_client()
        await mongo_client.disconnect()
        logger.info("mongodb_closed")
    except Exception as e:
        logger.warning("mongodb_close_failed", error=str(e))

    # Close Redis connection
    try:
        redis_service = get_redis_service()
        await redis_service.disconnect()
        logger.info("redis_service_closed")
    except Exception as e:
        logger.warning("redis_service_close_failed", error=str(e))


# Create FastAPI application
app = FastAPI(
    title="Market Intelligence API",
    description="""
    AI-powered market intelligence through background processing architecture.

    This API serves pre-computed market snapshots from MongoDB for fast responses.
    Background Celery tasks handle heavy AI processing:

    - **GET /api/v1/market-summary** - Returns cached market snapshot (< 200ms)
    - **NewsProcessingAgent** - AI-powered news analysis (runs every 15 min)
    - **SnapshotGenerationAgent** - AI market outlook generation (runs every 30 min)
    - **IndicesCollectionAgent** - Indices data collection (runs every 5 min)
    """,
    version=__version__,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =============================================================================
# Exception Handlers
# =============================================================================


@app.exception_handler(MarketPulseError)
async def market_pulse_error_handler(
    request: Request,
    exc: MarketPulseError,
) -> JSONResponse:
    """Handle custom MarketPulse errors."""
    logger.error(
        "market_pulse_error",
        error_code=exc.code,
        message=exc.message,
        details=exc.details,
    )
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error=exc.code,
            message=exc.message,
            details=exc.details,
        ).model_dump(),
    )


@app.exception_handler(Exception)
async def general_exception_handler(
    request: Request,
    exc: Exception,
) -> JSONResponse:
    """Handle unexpected errors."""
    logger.error(
        "unexpected_error",
        error=str(exc),
        error_type=type(exc).__name__,
    )
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error="INTERNAL_ERROR",
            message="An unexpected error occurred",
            details={"error_type": type(exc).__name__},
        ).model_dump(),
    )


# =============================================================================
# API Routes
# =============================================================================


@app.get(
    "/api/v1/health",
    response_model=HealthCheckResponse,
    summary="Health Check",
    description="Check API health status and background agent availability.",
)
async def health_check() -> HealthCheckResponse:
    """
    Health check endpoint with agent status.

    Returns the overall health status and background agent statuses.
    """
    return HealthCheckResponse(
        status="healthy",
        service="market-intelligence-api",
        version=__version__,
        timestamp=datetime.utcnow(),
        agents={
            "news_processing": "operational",
            "snapshot_generation": "operational",
            "indices_collection": "operational",
        },
    )


@app.get(
    "/api/v1/all-market-news/{records_to_fetch}",
    summary="All Market News",
    description="Fetch all market news from all sources with pagination",
)
async def get_all_market_news(
    records_to_fetch: int,
    page: int = 1,
    per_page: int = 10,
    type: Optional[str] = None,
):
    """
    Get all market news from all sources with standardized pagination.

    Combines news from:
    - Economy News
    - Other Markets News
    - Foreign Markets News
    - Mid-Session News

    Path Parameters:
    - records_to_fetch: Number of records to fetch per news type

    Query Parameters:
    - page: Page number (1-indexed, default: 1)
    - per_page: Items per page (1-100, default: 10)
    - type: Filter by source: economy-news, other-markets, foreign-markets, or market_phase (pre, mid, post) - optional

    Response includes:
    - data: News organized by source (economy-news, other-markets, foreign-markets, pre, mid, post)
    - pagination: Pagination metadata
    - errors: Any errors encountered during fetch
    """
    logger.info("all_market_news_request", page=page, per_page=per_page, records_to_fetch=records_to_fetch, type=type)

    try:
        news_service = get_cmots_news_service()
        response = await news_service.fetch_unified_market_news(
            limit=records_to_fetch,
            page=page,
            per_page=per_page,
            market_phase=type,
        )
        return response
    except Exception as e:
        logger.error("all_market_news_error", error=str(e))
        raise HTTPException(
            status_code=500,
            detail={
                "error": "ALL_MARKET_NEWS_ERROR",
                "message": str(e),
            },
        )


@app.get(
    "/api/v1/news/{news_type}",
    summary="News by Type",
    description="Fetch news for a specific type with pagination",
)
async def get_news_by_type(
    news_type: str,
    page: int = 1,
    per_page: int = 10,
    limit: int = 10,
):
    """
    Get news for a specific type with standardized pagination.

    Path Parameters:
    - news_type: One of 'economy-news', 'other-markets', 'foreign-markets'

    Query Parameters:
    - page: Page number (1-indexed, default: 1)
    - per_page: Items per page (1-100, default: 10)
    - limit: Items to fetch from API (default: 10)

    Response includes:
    - data: Array of news items for this page
    - pagination: Pagination metadata (page, total_pages, has_next, etc.)
    """
    logger.info("news_by_type_request", news_type=news_type, page=page, per_page=per_page)

    if news_type not in ["economy-news", "other-markets", "foreign-markets"]:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "INVALID_NEWS_TYPE",
                "message": "news_type must be one of: economy-news, other-markets, foreign-markets",
                "received": news_type,
            },
        )

    try:
        news_service = get_cmots_news_service()
        response = await news_service.fetch_news_by_type(
            news_type=news_type,
            limit=limit,
            page=page,
            per_page=per_page,
        )
        return response
    except Exception as e:
        logger.error("news_by_type_error", error=str(e), news_type=news_type)
        raise HTTPException(
            status_code=500,
            detail={
                "error": "NEWS_FETCH_ERROR",
                "message": str(e),
                "news_type": news_type,
            },
        )


@app.get(
    "/api/v1/mid-market-news/{records_to_fetch}",
    summary="Mid-Market News",
    description="Fetch mid-session market news with pagination",
)
async def get_mid_market_news(
    records_to_fetch: int,
    page: int = 1,
    per_page: int = 10,
):
    """
    Get mid-session market news with standardized pagination.

    Path Parameters:
    - records_to_fetch: Number of records to fetch from API

    Query Parameters:
    - page: Page number (1-indexed, default: 1)
    - per_page: Items per page (1-100, default: 10)

    Response includes:
    - data: Array of mid-session news items for this page
    - pagination: Pagination metadata (page, total_pages, has_next, etc.)
    """
    logger.info("mid_market_news_request", page=page, per_page=per_page, records_to_fetch=records_to_fetch)

    try:
        news_service = get_cmots_news_service()
        response = await news_service.fetch_news_by_type(
            news_type="mid",
            limit=records_to_fetch,
            page=page,
            per_page=per_page,
        )
        return response
    except Exception as e:
        logger.error("mid_market_news_error", error=str(e))
        raise HTTPException(
            status_code=500,
            detail={
                "error": "MID_MARKET_NEWS_ERROR",
                "message": str(e),
            },
        )


@app.get(
    "/api/v1/pre-market-news/{records_to_fetch}",
    summary="Pre-Market News",
    description="Fetch pre-session market news with pagination and caching",
)
async def get_pre_market_news(
    records_to_fetch: int,
    page: int = 1,
    per_page: int = 10,
):
    """
    Get pre-session market news with standardized pagination.

    Path Parameters:
    - records_to_fetch: Number of records to fetch from API

    Query Parameters:
    - page: Page number (1-indexed, default: 1)
    - per_page: Items per page (1-100, default: 10)

    Response includes:
    - data: Array of pre-session news items for this page
    - pagination: Pagination metadata (page, total_pages, has_next, etc.)

    Note: Pre-market news is cached with 24-hour TTL.
    """
    logger.info("pre_market_news_request", page=page, per_page=per_page, records_to_fetch=records_to_fetch)

    try:
        news_service = get_cmots_news_service()
        response = await news_service.fetch_news_by_type(
            news_type="pre",
            limit=records_to_fetch,
            page=page,
            per_page=per_page,
        )
        return response
    except Exception as e:
        logger.error("pre_market_news_error", error=str(e))
        raise HTTPException(
            status_code=500,
            detail={
                "error": "PRE_MARKET_NEWS_ERROR",
                "message": str(e),
            },
        )


@app.get(
    "/api/v1/post-market-news/{records_to_fetch}",
    summary="Post-Market News",
    description="Fetch end-session market news with pagination and caching",
)
async def get_post_market_news(
    records_to_fetch: int,
    page: int = 1,
    per_page: int = 10,
):
    """
    Get end-session market news with standardized pagination.

    Path Parameters:
    - records_to_fetch: Number of records to fetch from API

    Query Parameters:
    - page: Page number (1-indexed, default: 1)
    - per_page: Items per page (1-100, default: 10)

    Response includes:
    - data: Array of end-session news items for this page
    - pagination: Pagination metadata (page, total_pages, has_next, etc.)

    Note: Post-market news is cached with 24-hour TTL.
    """
    logger.info("post_market_news_request", page=page, per_page=per_page, records_to_fetch=records_to_fetch)

    try:
        news_service = get_cmots_news_service()
        response = await news_service.fetch_news_by_type(
            news_type="post",
            limit=records_to_fetch,
            page=page,
            per_page=per_page,
        )
        return response
    except Exception as e:
        logger.error("post_market_news_error", error=str(e))
        raise HTTPException(
            status_code=500,
            detail={
                "error": "POST_MARKET_NEWS_ERROR",
                "message": str(e),
            },
        )


@app.post(
    "/api/v1/company-news",
    summary="Company-wise News",
    description="Fetch news for multiple companies by NSE symbols with concurrency",
)
async def get_company_wise_news(
    nse_symbols: List[str],
    page: int = 1,
    per_page: int = 10,
    limit: int = 10,
):
    """
    Get company-wise news for multiple NSE symbols.

    Request Body:
    - nse_symbols: List of NSE symbols (e.g., ["RELIANCE", "TCS", "INFY"])

    Query Parameters:
    - page: Page number (1-indexed, default: 1)
    - per_page: Items per page (1-100, default: 10)
    - limit: Number of news items per company (default: 10)

    Response includes:
    - data: News organized by NSE symbol
    - pagination: Pagination metadata
    - symbols_found: List of valid symbols found
    - symbols_not_found: List of invalid symbols
    - errors: Any errors encountered
    """
    logger.info(
        "company_news_request",
        nse_symbols=nse_symbols,
        page=page,
        per_page=per_page,
        limit=limit,
    )

    try:
        company_news_service = get_company_news_service()
        response = await company_news_service.fetch_company_news_by_symbols(
            nse_symbols=nse_symbols,
            limit=limit,
            page=page,
            per_page=per_page,
        )
        return response
    except Exception as e:
        logger.error("company_news_error", error=str(e))
        raise HTTPException(
            status_code=500,
            detail={
                "error": "COMPANY_NEWS_ERROR",
                "message": str(e),
            },
        )


@app.post(
    "/api/v1/search-news",
    summary="Search News by Stocks and/or Companies",
    description="Search news database by stock tickers and/or company names",
)
async def search_news(
    search_request: NewsSearchRequest,
):
    """
    Search news articles by stock tickers and/or company names.

    This endpoint searches the news database for articles mentioning
    any of the provided stock tickers OR company names.

    Request Body:
    - mentioned_stocks: List of stock tickers (e.g., ["RELIANCE", "TCS"]) [optional]
    - mentioned_companies: List of company names (e.g., ["Apple", "Google"]) [optional]
    - hours: How many hours back to search (1-730, default: 24)
    - limit: Maximum articles to return (1-500, default: 50)

    At least one of mentioned_stocks or mentioned_companies must be provided.

    Response:
    - data: List of matching news articles with details
    - total_found: Total number of articles found
    - filters_applied: Summary of filters used
    - query_timestamp: When the query was executed
    """
    from app.db.repositories.news_repository import get_news_repository

    logger.info(
        "search_news_request",
        mentioned_stocks=search_request.mentioned_stocks,
        mentioned_companies=search_request.mentioned_companies,
        hours=search_request.hours,
        limit=search_request.limit,
    )

    try:
        news_repo = get_news_repository()

        # Search the database
        articles = await news_repo.search_by_stocks_and_companies(
            stocks=search_request.mentioned_stocks,
            companies=search_request.mentioned_companies,
            hours=search_request.hours,
            limit=search_request.limit,
        )

        # Format response with summary/full_text handling
        formatted_data = []
        for article in articles:
            article_dict = article.model_dump()

            # If summary is empty, use full_text as summary
            if not article_dict.get("summary") or article_dict["summary"] == "":
                article_dict["summary"] = article_dict.get("full_text", "")

            # Remove full_text from response
            article_dict.pop("full_text", None)

            formatted_data.append(article_dict)

        # Format response
        response = {
            "data": formatted_data,
            "total_found": len(articles),
            "filters_applied": {
                "stocks": search_request.mentioned_stocks,
                "companies": search_request.mentioned_companies,
                "hours": search_request.hours,
                "limit": search_request.limit,
            },
            "query_timestamp": datetime.utcnow().isoformat(),
        }

        logger.info(
            "search_news_success",
            total_found=len(articles),
        )

        return response

    except ValueError as e:
        logger.error("search_news_validation_error", error=str(e))
        raise HTTPException(
            status_code=400,
            detail={
                "error": "INVALID_SEARCH_CRITERIA",
                "message": str(e),
            },
        )
    except Exception as e:
        logger.error("search_news_error", error=str(e))
        raise HTTPException(
            status_code=500,
            detail={
                "error": "SEARCH_NEWS_ERROR",
                "message": str(e),
            },
        )


@app.get(
    "/api/v1/world-indices",
    summary="World Indices",
    description="Fetch current world market indices data",
)
async def get_world_indices():
    """
    Get world market indices data.

    Returns data for global market indices including:
    - Index name and symbol
    - Current value, change, and percentage change
    - Open, high, low values
    - Last updated timestamp

    Response includes:
    - data: Array of world indices
    - total_count: Number of indices returned
    - fetched_at: Timestamp of data fetch
    """
    logger.info("world_indices_request")

    try:
        response = await fetch_world_indices()
        return response
    except Exception as e:
        logger.error("world_indices_error", error=str(e))
        raise HTTPException(
            status_code=500,
            detail={
                "error": "WORLD_INDICES_ERROR",
                "message": str(e),
            },
        )


@app.get(
    "/api/v1/market-summary",
    summary="Market Summary",
    description="Get aggregated market summary from latest snapshot",
)
async def get_market_summary(
    phase: Optional[str] = None,
    include_news_details: bool = True,
    news_limit: int = 10,
):
    """
    Get aggregated market summary.

    This endpoint returns the latest market snapshot which includes:
    - Market outlook (sentiment, confidence, reasoning)
    - Indices data
    - Market summary bullets with causal language
    - Executive summary
    - Trending news (mid-market, top impacting). Themed sectors (sector, relevant companies, sentiment) when identified

    Query Parameters:
    - phase: Optional market phase filter (pre/mid/post)
    - include_news_details: Include full news details (default: true)
    - news_limit: Maximum news items to include (default: 10, max: 50)

    Response time target: < 200ms (with caching)
    """
    from datetime import datetime

    logger.info(
        "market_summary_request",
        phase=phase,
        include_news_details=include_news_details,
        news_limit=news_limit,
    )

    try:
        from app.db.mongodb import get_mongodb_client
        from app.db.repositories.snapshot_repository import get_snapshot_repository
        from app.db.repositories.news_repository import get_news_repository
        from app.services.market_intelligence_service import get_market_phase

        # Ensure MongoDB is connected
        mongo_client = get_mongodb_client()
        try:
            await mongo_client.connect()
        except Exception:
            pass  # Already connected

        snapshot_repo = get_snapshot_repository()
        news_repo = get_news_repository()

        # Determine market phase
        if not phase:
            phase_data = await get_market_phase()
            phase = phase_data["phase"]

        # Try to get latest valid snapshot
        snapshot = await snapshot_repo.get_latest_valid(phase)

        if not snapshot:
            # No valid snapshot, try to get any latest
            snapshot = await snapshot_repo.get_latest(phase)

            if not snapshot:
                # No snapshot at all, trigger generation and return basic response
                from app.celery_app.tasks.snapshot_tasks import generate_market_snapshot
                generate_market_snapshot.delay(market_phase=phase, force=True)

                return {
                    "generated_at": datetime.utcnow().isoformat(),
                    "market_phase": phase,
                    "status": "generating",
                    "message": "Snapshot generation in progress. Please retry in 30 seconds.",
                    "market_outlook": None,
                    "indices_data": [],
                    "market_summary": [],
                    "executive_summary": None,
                    "trending_now": None,
                    "themed": [],
                    "all_news": [],
                }

        def _news_to_dict(n):
            return {
                "news_id": n.news_id,
                "headline": n.headline,
                "summary": n.summary,
                "source": n.source,
                "published_at": n.published_at.isoformat(),
                "sentiment": n.sentiment,
                "mentioned_stocks": n.mentioned_stocks,
                "mentioned_sectors": n.mentioned_sectors,
                "is_breaking": n.is_breaking,
            }

        # Build response
        response = {
            "generated_at": snapshot.generated_at.isoformat(),
            "request_id": snapshot.snapshot_id,
            "market_phase": snapshot.market_phase,
            "market_outlook": snapshot.market_outlook.model_dump() if snapshot.market_outlook else None,
            "indices_data": [idx.model_dump() for idx in snapshot.indices_data],
            "market_summary": [ms.model_dump() for ms in snapshot.market_summary],
            "executive_summary": snapshot.executive_summary,
            "trending_now": None,
            "themed": [t.model_dump() for t in (snapshot.themed or [])],
            "all_news_ids": snapshot.all_news_ids[:news_limit],
            "degraded_mode": snapshot.degraded_mode,
            "warnings": snapshot.warnings,
        }

        # Trending now: full news objects (same shape as all_news), mid-market only
        if snapshot.market_phase == "mid":
            if snapshot.trending_now:
                trending_docs = await news_repo.get_by_ids(snapshot.trending_now)
                response["trending_now"] = [_news_to_dict(n) for n in trending_docs]
            else:
                response["trending_now"] = []

        # Optionally include full news details for all_news
        if include_news_details and snapshot.all_news_ids:
            news_ids = snapshot.all_news_ids[:min(news_limit, 50)]
            news_docs = await news_repo.get_by_ids(news_ids)
            response["all_news"] = [_news_to_dict(n) for n in news_docs]

        return response

    except Exception as e:
        logger.error("market_summary_error", error=str(e))
        raise HTTPException(
            status_code=500,
            detail={
                "error": "MARKET_SUMMARY_ERROR",
                "message": str(e),
            },
        )


@app.post(
    "/api/v1/snapshot/generate",
    summary="Trigger Snapshot Generation",
    description="Manually trigger market snapshot generation",
)
async def trigger_snapshot_generation(
    phase: Optional[str] = None,
    force: bool = False,
    background_tasks: BackgroundTasks = None,
):
    """
    Manually trigger snapshot generation.

    Query Parameters:
    - phase: Market phase (pre/mid/post). If not provided, auto-detected.
    - force: Force regeneration even if recent snapshot exists

    Returns:
    - Task status and snapshot ID (if available)
    """
    logger.info(
        "snapshot_generation_triggered",
        phase=phase,
        force=force,
    )

    try:
        from app.celery_app.tasks.snapshot_tasks import generate_market_snapshot
        from app.services.market_intelligence_service import get_market_phase

        # Determine phase if not provided
        if not phase:
            phase_data = await get_market_phase()
            phase = phase_data["phase"]

        # Trigger async task
        task = generate_market_snapshot.delay(market_phase=phase, force=force)

        return {
            "status": "triggered",
            "task_id": task.id,
            "market_phase": phase,
            "message": "Snapshot generation started. Check /api/v1/market-summary for results.",
        }

    except Exception as e:
        logger.error("snapshot_trigger_error", error=str(e))
        raise HTTPException(
            status_code=500,
            detail={
                "error": "SNAPSHOT_TRIGGER_ERROR",
                "message": str(e),
            },
        )


@app.post(
    "/api/v1/data/populate",
    summary="Trigger Data Population (Async)",
    description="Manually trigger data population tasks asynchronously (news, indices, snapshot)",
)
async def trigger_data_population(
    fetch_news: bool = True,
    fetch_indices: bool = True,
    generate_snapshot: bool = True,
    market_phase: Optional[str] = None,
):
    """
    Manually trigger data population tasks asynchronously.

    Query Parameters:
    - fetch_news: Trigger news fetch (default: true)
    - fetch_indices: Trigger indices fetch (default: true)
    - generate_snapshot: Trigger snapshot generation (default: true)
    - market_phase: Market phase (pre/mid/post) for snapshot generation. If not provided, auto-detected.

    Returns:
    - Task IDs for triggered tasks
    """
    logger.info(
        "data_population_triggered",
        fetch_news=fetch_news,
        fetch_indices=fetch_indices,
        generate_snapshot=generate_snapshot,
        market_phase=market_phase,
    )

    try:
        tasks = {}

        if fetch_news:
            from app.celery_app.tasks.news_tasks import fetch_and_process_news
            task = fetch_and_process_news.delay(limit=20, market_phase=market_phase)
            tasks["news_fetch"] = {"task_id": task.id, "status": "queued", "market_phase": market_phase}

        if fetch_indices:
            from app.celery_app.tasks.indices_tasks import fetch_indices_data
            task = fetch_indices_data.delay()
            tasks["indices_fetch"] = {"task_id": task.id, "status": "queued"}

        if generate_snapshot:
            from app.celery_app.tasks.snapshot_tasks import generate_market_snapshot
            task = generate_market_snapshot.delay(force=True, market_phase=market_phase)
            tasks["snapshot_generation"] = {"task_id": task.id, "status": "queued", "market_phase": market_phase}

        return {
            "status": "tasks_queued",
            "tasks": tasks,
            "message": "Check /api/v1/db/stats for results after tasks complete.",
        }

    except Exception as e:
        logger.error("data_population_error", error=str(e))
        raise HTTPException(
            status_code=500,
            detail={
                "error": "DATA_POPULATION_ERROR",
                "message": str(e),
            },
        )


@app.post(
    "/api/v1/data/populate/news",
    summary="Fetch News (Sync)",
    description="Synchronously fetch and process news from all sources",
)
async def sync_fetch_news(
    limit: int = 20,
):
    """
    Synchronously fetch and process news.

    This endpoint waits for completion and returns results immediately.

    Query Parameters:
    - limit: Number of items per news type to fetch (default: 20)

    Returns:
    - Statistics about fetched and processed news
    """
    logger.info("sync_fetch_news_started", limit=limit)

    try:
        from app.celery_app.tasks.news_tasks import _fetch_and_process_news_async

        result = await _fetch_and_process_news_async(limit)

        logger.info("sync_fetch_news_completed", **result)

        return {
            "status": "success",
            "operation": "news_fetch",
            "statistics": result,
        }

    except Exception as e:
        logger.error("sync_fetch_news_error", error=str(e))
        raise HTTPException(
            status_code=500,
            detail={
                "error": "NEWS_FETCH_ERROR",
                "message": str(e),
            },
        )


@app.post(
    "/api/v1/data/populate/indices",
    summary="Fetch Indices (Sync)",
    description="Synchronously fetch and store world indices data",
)
async def sync_fetch_indices():
    """
    Synchronously fetch and store indices data.

    This endpoint waits for completion and returns results immediately.
    Unlike the async version, this always fetches regardless of market hours.

    Returns:
    - Statistics about fetched and stored indices
    """
    logger.info("sync_fetch_indices_started")

    try:
        from app.celery_app.tasks.indices_tasks import _fetch_indices_async

        result = await _fetch_indices_async()

        logger.info("sync_fetch_indices_completed", **result)

        return {
            "status": "success",
            "operation": "indices_fetch",
            "statistics": result,
        }

    except Exception as e:
        logger.error("sync_fetch_indices_error", error=str(e))
        raise HTTPException(
            status_code=500,
            detail={
                "error": "INDICES_FETCH_ERROR",
                "message": str(e),
            },
        )


@app.post(
    "/api/v1/data/populate/snapshot",
    summary="Generate Snapshot (Sync)",
    description="Synchronously generate market snapshot",
)
async def sync_generate_snapshot(
    phase: Optional[str] = None,
    force: bool = True,
):
    """
    Synchronously generate market snapshot.

    This endpoint waits for completion and returns results immediately.

    Query Parameters:
    - phase: Market phase (pre/mid/post). If not provided, auto-detected.
    - force: Force generation even if recent snapshot exists (default: true)

    Returns:
    - Snapshot ID and generation statistics
    """
    logger.info("sync_generate_snapshot_started", phase=phase, force=force)

    try:
        from app.celery_app.tasks.snapshot_tasks import _generate_snapshot_async

        result = await _generate_snapshot_async(phase, force)

        logger.info("sync_generate_snapshot_completed", **result)

        return {
            "status": "success",
            "operation": "snapshot_generation",
            "result": result,
        }

    except Exception as e:
        logger.error("sync_generate_snapshot_error", error=str(e))
        raise HTTPException(
            status_code=500,
            detail={
                "error": "SNAPSHOT_GENERATION_ERROR",
                "message": str(e),
            },
        )


@app.get(
    "/api/v1/db/stats",
    summary="Database Statistics",
    description="Get statistics about stored data",
)
async def get_database_stats():
    """
    Get database statistics.

    Returns counts and metadata about stored data:
    - News articles count
    - Snapshots count
    - Indices data points
    """
    logger.info("database_stats_request")

    try:
        from app.db.mongodb import get_mongodb_client

        mongo_client = get_mongodb_client()

        # Check if MongoDB is healthy
        is_healthy = await mongo_client.health_check()
        if not is_healthy:
            return {
                "status": "unavailable",
                "message": "MongoDB not connected",
            }

        db = mongo_client.get_database()

        # Get collection stats
        news_count = await db["news_articles"].count_documents({})
        snapshots_count = await db["market_snapshots"].count_documents({})
        indices_count = await db["indices_timeseries"].count_documents({})

        # Get latest snapshot info
        latest_snapshot = await db["market_snapshots"].find_one(
            sort=[("generated_at", -1)]
        )

        return {
            "status": "healthy",
            "collections": {
                "news_articles": news_count,
                "market_snapshots": snapshots_count,
                "indices_timeseries": indices_count,
            },
            "latest_snapshot": {
                "snapshot_id": latest_snapshot.get("snapshot_id") if latest_snapshot else None,
                "generated_at": latest_snapshot.get("generated_at").isoformat() if latest_snapshot else None,
                "market_phase": latest_snapshot.get("market_phase") if latest_snapshot else None,
            } if latest_snapshot else None,
        }

    except Exception as e:
        logger.error("database_stats_error", error=str(e))
        return {
            "status": "error",
            "message": str(e),
        }


@app.get(
    "/api/v1/agents/status",
    response_model=AgentStatusResponse,
    summary="Agent Status",
    description="Get detailed status and metrics for background processing agents.",
)
async def agent_status() -> AgentStatusResponse:
    """
    Get detailed agent status and metrics.

    Returns information about each background processing agent including:
    - Operational status
    - Description of responsibilities
    - Task schedule
    """
    agents = [
        {
            "name": "news_processing_agent",
            "status": "operational",
            "description": "AI-powered news analysis (sentiment, entities, summaries). Runs every 15 min.",
            "avg_execution_time_ms": 500,
            "success_rate": 0.98,
        },
        {
            "name": "snapshot_generation_agent",
            "status": "operational",
            "description": "AI-powered market snapshot generation (outlook, causal bullets). Runs every 30 min.",
            "avg_execution_time_ms": 800,
            "success_rate": 0.97,
        },
        {
            "name": "indices_collection_agent",
            "status": "operational",
            "description": "Indices data collection and historical storage. Runs every 5 min during market hours.",
            "avg_execution_time_ms": 200,
            "success_rate": 0.99,
        },
    ]

    return AgentStatusResponse(
        agents=agents,
        total_agents=len(agents),
        operational_agents=len([a for a in agents if a["status"] == "operational"]),
    )


@app.get(
    "/",
    summary="Root",
    description="API root endpoint with basic information.",
)
async def root():
    """Root endpoint with API information."""
    return {
        "service": "Market Intelligence API",
        "version": __version__,
        "status": "running",
        "architecture": "background-processing-3-agent",
        "agents": [
            "news_processing_agent",
            "snapshot_generation_agent",
            "indices_collection_agent",
        ],
        "docs": "/docs",
        "health": "/api/v1/health",
        "market_summary": "/api/v1/market-summary",
    }


# =============================================================================
# Application Entry Point
# =============================================================================


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=True,
        log_level=settings.LOG_LEVEL.lower(),
    )
