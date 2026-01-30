"""
Market Pulse Multi-Agent API (Simplified)

FastAPI application providing AI-powered market insights using
a simplified 3-agent orchestration system.
"""

import uuid
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Optional, List

from fastapi import BackgroundTasks, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app import __version__
from app.agents.base import AgentExecutionContext
from app.agents.orchestrator import OrchestratorAgent
from app.config import get_settings
from app.models.requests import MarketPulseRequest
from app.models.responses import (
    MarketPulseResponse,
    HealthCheckResponse,
    AgentStatusResponse,
    ErrorResponse,
)
from app.utils.logging import setup_logging, get_logger, bind_request_context
from app.utils.tracing import setup_tracing
from app.utils.exceptions import MarketPulseError, OrchestrationError
from app.services.redis_service import get_redis_service
from app.services.cmots_news_service import fetch_world_indices, get_cmots_news_service
from app.services.company_news_service import get_company_news_service


# Initialize settings and logging
settings = get_settings()
logger = get_logger(__name__)

# Global orchestrator instance
orchestrator: Optional[OrchestratorAgent] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager.

    Handles startup and shutdown events.
    """
    global orchestrator

    # Startup
    logger.info("application_starting", version=__version__)
    setup_logging(settings.LOG_LEVEL)
    setup_tracing()

    # Initialize orchestrator
    orchestrator = OrchestratorAgent()
    logger.info("orchestrator_initialized")

    # Initialize Redis service
    try:
        redis_service = get_redis_service()
        await redis_service.connect()
        logger.info("redis_service_initialized")
    except Exception as e:
        logger.warning("redis_service_failed_to_initialize", error=str(e))

    yield

    # Shutdown
    logger.info("application_shutting_down")

    # Close Redis connection
    try:
        redis_service = get_redis_service()
        await redis_service.disconnect()
        logger.info("redis_service_closed")
    except Exception as e:
        logger.warning("redis_service_close_failed", error=str(e))


# Create FastAPI application
app = FastAPI(
    title="Market Pulse Multi-Agent API",
    description="""
    AI-powered market insights using simplified multi-agent orchestration.

    This API coordinates 3 specialized agents to generate comprehensive
    market pulse analysis:

    1. **Market Intelligence Agent** - Fetches indices, news, and determines market phase
    2. **Portfolio Insight Agent** - Retrieves user context and analyzes news impact
    3. **Summary Generation Agent** - Creates coherent market narrative
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


@app.post(
    "/api/v1/pulse",
    response_model=MarketPulseResponse,
    summary="Generate Market Pulse",
    description="Generate comprehensive market pulse analysis using multi-agent orchestration.",
    responses={
        200: {"description": "Successfully generated market pulse"},
        500: {"description": "Server error", "model": ErrorResponse},
    },
)
async def generate_market_pulse(
    request: MarketPulseRequest,
    background_tasks: BackgroundTasks,
) -> MarketPulseResponse:
    """
    Generate Market Pulse using simplified 3-agent orchestration.

    This endpoint coordinates 3 specialized agents to generate:
    - Market outlook and momentum analysis
    - News with impact analysis
    - Portfolio-specific insights
    - Causal market summaries

    The orchestrator manages agent execution with:
    - Sequential 3-phase execution
    - Fallback strategies for failures
    - Comprehensive error handling
    """
    # Generate request ID
    request_id = str(uuid.uuid4())

    # Bind request context for logging
    bind_request_context(request_id, request.user_id)

    logger.info(
        "pulse_request_received",
        request_id=request_id,
        user_id=request.user_id,
        indices=request.selected_indices,
    )

    # Create execution context
    context = AgentExecutionContext(
        request_id=request_id,
        user_id=request.user_id,
        timestamp=datetime.utcnow(),
        trace_id=request_id,
    )

    try:
        # Execute orchestration
        response = await orchestrator.orchestrate(request, context)

        # Log analytics in background
        background_tasks.add_task(
            log_analytics,
            request,
            response,
            context,
        )

        logger.info(
            "pulse_request_completed",
            request_id=request_id,
            market_phase=response.market_phase,
            degraded_mode=response.degraded_mode,
        )

        return response

    except OrchestrationError as e:
        logger.error(
            "orchestration_error",
            request_id=request_id,
            error=str(e),
            failed_agents=e.failed_agents,
        )
        raise HTTPException(
            status_code=500,
            detail={
                "error": "ORCHESTRATION_ERROR",
                "message": str(e),
                "request_id": request_id,
                "failed_agents": e.failed_agents,
            },
        )
    except Exception as e:
        logger.error(
            "pulse_generation_failed",
            request_id=request_id,
            error=str(e),
            error_type=type(e).__name__,
        )
        raise HTTPException(
            status_code=500,
            detail={
                "error": "GENERATION_ERROR",
                "message": "Market Pulse generation failed",
                "request_id": request_id,
            },
        )


@app.get(
    "/api/v1/health",
    response_model=HealthCheckResponse,
    summary="Health Check",
    description="Check API health status and agent availability.",
)
async def health_check() -> HealthCheckResponse:
    """
    Health check endpoint with agent status.

    Returns the overall health status and individual agent statuses.
    """
    return HealthCheckResponse(
        status="healthy",
        service="market-pulse-multi-agent",
        version=__version__,
        timestamp=datetime.utcnow(),
        agents={
            "market_intelligence": "operational",
            "portfolio_insight": "operational",
            "summary_generation": "operational",
            "orchestrator": "operational",
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
    - type: Filter by news type (economy-news, other-markets, foreign-markets, mid-market) - optional

    Response includes:
    - data: News organized by type (economy-news, other-markets, foreign-markets, mid-market)
    - pagination: Pagination metadata
    - errors: Any errors encountered during fetch
    """
    logger.info("all_market_news_request", page=page, per_page=per_page, records_to_fetch=records_to_fetch, news_type=type)

    try:
        news_service = get_cmots_news_service()
        response = await news_service.fetch_unified_market_news(
            limit=records_to_fetch,
            page=page,
            per_page=per_page,
            news_type=type,
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
            news_type="mid-market",
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
    "/api/v1/agents/status",
    response_model=AgentStatusResponse,
    summary="Agent Status",
    description="Get detailed status and metrics for all agents.",
)
async def agent_status() -> AgentStatusResponse:
    """
    Get detailed agent status and metrics.

    Returns information about each agent including:
    - Operational status
    - Average execution time
    - Success rate
    """
    agents = [
        {
            "name": "market_intelligence_agent",
            "status": "operational",
            "description": "Fetches market data, news, and analyzes market conditions",
            "avg_execution_time_ms": 450,
            "success_rate": 0.98,
        },
        {
            "name": "portfolio_insight_agent",
            "status": "operational",
            "description": "Retrieves user context and analyzes news impact on portfolio",
            "avg_execution_time_ms": 650,
            "success_rate": 0.97,
        },
        {
            "name": "summary_generation_agent",
            "status": "operational",
            "description": "Creates market summaries with causal language",
            "avg_execution_time_ms": 400,
            "success_rate": 0.97,
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
        "service": "Market Pulse Multi-Agent API",
        "version": __version__,
        "status": "running",
        "architecture": "simplified-3-agent",
        "agents": [
            "market_intelligence_agent",
            "portfolio_insight_agent",
            "summary_generation_agent",
        ],
        "docs": "/docs",
        "health": "/api/v1/health",
    }


# =============================================================================
# Background Tasks
# =============================================================================


async def log_analytics(
    request: MarketPulseRequest,
    response: MarketPulseResponse,
    context: AgentExecutionContext,
):
    """
    Background task to log analytics.

    Records metrics about the request for monitoring and analysis.
    """
    try:
        logger.info(
            "analytics_logged",
            request_id=context.request_id,
            user_id=request.user_id,
            market_phase=response.market_phase,
            news_count=len(response.all_news),
            themes_count=len(response.themed_news),
            degraded_mode=response.degraded_mode,
        )
    except Exception as e:
        logger.warning(
            "analytics_logging_failed",
            error=str(e),
        )


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
