"""
Market Intelligence Agents - Background Processing Architecture.

This module provides 3 specialized agents for background data processing:

1. NewsProcessingAgent - AI-powered news analysis
   - Sentiment analysis (bullish/bearish/neutral)
   - Entity extraction (stocks, sectors, companies)
   - Summary generation
   - Impact analysis with causal chains

2. SnapshotGenerationAgent - AI-powered market snapshot creation
   - Market outlook generation (pre/post market)
   - Summary bullets with causal language
   - Executive summary generation
   - Trending news selection (mid-market)

3. IndicesCollectionAgent - Indices data collection
   - Fetch world indices from CMOTS API
   - Store historical data in MongoDB
   - Market hours awareness
   - Data normalization

Architecture Flow:
┌─────────────────────────────────────────────────────────────────┐
│                    Celery Tasks (Background)                     │
│  ┌─────────────────┐  ┌──────────────────┐  ┌────────────────┐ │
│  │fetch_news (15m) │  │gen_snapshot (30m)│  │fetch_indices   │ │
│  │                 │  │                  │  │(5m market hrs) │ │
│  └────────┬────────┘  └────────┬─────────┘  └───────┬────────┘ │
│           │                    │                     │          │
│           ▼                    ▼                     ▼          │
│  ┌─────────────────┐  ┌──────────────────┐  ┌────────────────┐ │
│  │NewsProcessing   │  │SnapshotGeneration│  │IndicesCollection│
│  │Agent (Gemini)   │  │Agent (Gemini)    │  │Agent           │ │
│  └─────────────────┘  └──────────────────┘  └────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                       MongoDB                                    │
│  ┌──────────────────┐  ┌────────────────┐  ┌─────────────────┐ │
│  │  market_snapshots │  │ news_articles  │  │indices_timeseries│ │
│  └──────────────────┘  └────────────────┘  └─────────────────┘ │
└─────────────────────────────────────────────────────────────────┘

Usage:
    from app.agents import (
        get_news_processing_agent,
        get_snapshot_generation_agent,
        get_indices_collection_agent,
    )
    
    # In Celery tasks:
    agent = get_news_processing_agent()
    result = await agent.analyze_article(article)
"""

# Background Processing Agents (3-Agent Architecture)
from app.agents.news_processing_agent import (
    NewsProcessingAgent,
    get_news_processing_agent,
)
from app.agents.snapshot_agent import (
    SnapshotGenerationAgent,
    get_snapshot_generation_agent,
)
from app.agents.indices_agent import (
    IndicesCollectionAgent,
    get_indices_collection_agent,
)

# Base classes (for extending agents if needed)
from app.agents.base import (
    BaseAgent,
    AgentConfig,
    AgentExecutionContext,
)

__all__ = [
    # Background Processing Agents
    "NewsProcessingAgent",
    "get_news_processing_agent",
    "SnapshotGenerationAgent",
    "get_snapshot_generation_agent",
    "IndicesCollectionAgent",
    "get_indices_collection_agent",
    # Base classes
    "BaseAgent",
    "AgentConfig",
    "AgentExecutionContext",
]
