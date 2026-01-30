"""
Services for the Market Pulse application.

Services:
1. CacheService - Redis-based caching
2. MarketIntelligenceService - Unified market data and news fetching
3. NewsProcessorService - AI-powered news analysis
4. SnapshotGeneratorService - AI-powered snapshot generation
"""

from app.services.cache_service import CacheService
from app.services.market_intelligence_service import (
    fetch_market_intelligence,
    fetch_market_indices,
    get_market_phase,
    fetch_market_news,
    fetch_stock_specific_news,
    cluster_news_by_topic,
    get_market_intelligence_tools,
    get_market_intelligence_tool_handlers,
)
from app.services.news_processor_service import (
    NewsProcessorService,
    get_news_processor_service,
)
from app.services.snapshot_generator_service import (
    SnapshotGeneratorService,
    get_snapshot_generator_service,
)

__all__ = [
    "CacheService",
    # Market Intelligence Service
    "fetch_market_intelligence",
    "fetch_market_indices",
    "get_market_phase",
    "fetch_market_news",
    "fetch_stock_specific_news",
    "cluster_news_by_topic",
    "get_market_intelligence_tools",
    "get_market_intelligence_tool_handlers",
    # News Processor Service
    "NewsProcessorService",
    "get_news_processor_service",
    # Snapshot Generator Service
    "SnapshotGeneratorService",
    "get_snapshot_generator_service",
]
