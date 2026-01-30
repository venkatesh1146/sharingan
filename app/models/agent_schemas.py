"""
Agent Input/Output Schemas for the Simplified Multi-Agent System.

This simplified version has 3 agents:
1. MarketIntelligenceAgent - Combined market data + news analysis
2. PortfolioInsightAgent - Combined user context + impact analysis
3. SummaryGenerationAgent - Summary generation (unchanged)
"""

from datetime import datetime
from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field

from app.models.domain import (
    IndexData,
    MarketOutlook,
    NewsItem,
    PortfolioHolding,
    PreliminaryTheme,
    ThemeGroup,
    NewsWithImpact,
    PortfolioImpact,
    WatchlistAlert,
    MarketSummaryBullet,
)


# =============================================================================
# Market Intelligence Agent Schemas (Merged: MarketData + NewsAnalysis)
# =============================================================================


class MarketIntelligenceAgentInput(BaseModel):
    """Input schema for Market Intelligence Agent."""

    selected_indices: Optional[List[str]] = Field(
        default=None,
        description="List of index tickers (None = use phase-based indices: pre/mid/post market)",
    )
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="Timestamp for the analysis",
    )
    force_refresh: bool = Field(
        default=False,
        description="Force refresh of cached data",
    )
    time_window_hours: int = Field(
        default=24,
        ge=1,
        le=72,
        description="Hours of news to fetch",
    )
    max_articles: int = Field(
        default=50,
        ge=10,
        le=100,
        description="Maximum articles to analyze",
    )
    watchlist: Optional[List[str]] = Field(
        None, description="User's watchlist for filtering stock-specific news"
    )


class MarketIntelligenceAgentOutput(BaseModel):
    """Output schema for Market Intelligence Agent."""

    # Market data outputs
    market_phase: Literal["pre", "mid", "post"] = Field(
        ..., description="Current market phase"
    )
    indices_data: Dict[str, IndexData] = Field(
        ..., description="Market data for each requested index"
    )
    market_outlook: Optional[MarketOutlook] = Field(
        None, description="Market outlook (None during mid-market)"
    )
    market_momentum: Literal[
        "strong_up", "moderate_up", "sideways", "moderate_down", "strong_down"
    ] = Field(..., description="Overall market momentum")
    data_freshness: datetime = Field(
        default_factory=datetime.utcnow,
        description="When the data was fetched",
    )
    
    # News analysis outputs
    news_items: List[NewsItem] = Field(
        ..., description="All fetched and processed news items"
    )
    sentiment_distribution: Dict[str, int] = Field(
        ..., description="Count of news by sentiment {bullish: N, bearish: N, neutral: N}"
    )
    preliminary_themes: List[PreliminaryTheme] = Field(
        ..., description="Initial theme groupings"
    )
    key_topics: List[str] = Field(
        ..., description="Key topics identified in the news"
    )
    breaking_news: List[str] = Field(
        default_factory=list,
        description="IDs of breaking news items",
    )


# =============================================================================
# Portfolio Insight Agent Schemas (Merged: UserContext + ImpactAnalysis)
# =============================================================================


class PortfolioInsightAgentInput(BaseModel):
    """Input schema for Portfolio Insight Agent."""

    user_id: str = Field(..., description="User identifier")
    include_watchlist: bool = Field(
        default=True, description="Whether to fetch watchlist"
    )
    include_portfolio: bool = Field(
        default=True, description="Whether to fetch portfolio"
    )
    news_filter: Literal["all", "watchlist", "portfolio"] = Field(
        default="all", description="News filtering preference"
    )
    # Impact analysis inputs
    news_items: List[NewsItem] = Field(
        ..., description="News items to analyze for impact"
    )
    indices_data: Dict[str, IndexData] = Field(
        ..., description="Market indices data for context"
    )
    preliminary_themes: List[PreliminaryTheme] = Field(
        default_factory=list, description="Preliminary themes from Market Intelligence"
    )


class PortfolioInsightAgentOutput(BaseModel):
    """Output schema for Portfolio Insight Agent."""

    # User context outputs
    user_id: str = Field(..., description="User identifier")
    watchlist: List[str] = Field(
        default_factory=list, description="User's watchlist tickers"
    )
    portfolio: List[PortfolioHolding] = Field(
        default_factory=list, description="User's portfolio holdings"
    )
    sector_exposure: Dict[str, float] = Field(
        default_factory=dict,
        description="Sector exposure as percentages (sector -> weight)",
    )
    total_portfolio_value: float = Field(
        default=0.0, description="Total portfolio value"
    )
    user_preferences: Dict = Field(
        default_factory=dict, description="User preferences and settings"
    )
    risk_profile: Optional[Literal["conservative", "moderate", "aggressive"]] = Field(
        None, description="User's risk profile if available"
    )
    
    # Impact analysis outputs
    news_with_impacts: List[NewsWithImpact] = Field(
        ..., description="News items with impact analysis"
    )
    portfolio_level_impact: PortfolioImpact = Field(
        ..., description="Aggregate portfolio impact"
    )
    watchlist_alerts: List[WatchlistAlert] = Field(
        default_factory=list, description="Alerts for watchlist stocks"
    )
    refined_themes: List[ThemeGroup] = Field(
        ..., description="Refined theme groups with full analysis"
    )
    sector_impact_summary: Dict[str, str] = Field(
        default_factory=dict,
        description="Summary of impact by sector",
    )
    causal_chains: List[str] = Field(
        default_factory=list,
        description="Key causal chains identified",
    )


# =============================================================================
# Summary Generation Agent Schemas (Unchanged)
# =============================================================================


class SummaryGenerationAgentInput(BaseModel):
    """Input schema for Summary Generation Agent."""

    market_outlook: Optional[MarketOutlook] = Field(
        None, description="Market outlook (None during mid-market)"
    )
    market_phase: Literal["pre", "mid", "post"] = Field(
        ..., description="Current market phase"
    )
    news_with_impacts: List[NewsWithImpact] = Field(
        ..., description="News items with impact analysis"
    )
    indices_data: Dict[str, IndexData] = Field(
        ..., description="Market indices data"
    )
    portfolio_impact: PortfolioImpact = Field(
        ..., description="Portfolio impact analysis"
    )
    refined_themes: List[ThemeGroup] = Field(
        ..., description="Refined theme groups"
    )
    max_bullets: int = Field(
        default=3, ge=1, le=5, description="Maximum summary bullets"
    )


class SummaryGenerationAgentOutput(BaseModel):
    """Output schema for Summary Generation Agent."""

    market_summary_bullets: Optional[List[MarketSummaryBullet]] = Field(
        None,
        description="Market summary bullets (None during mid-market)",
        max_length=5,
    )
    trending_now_section: Optional[List[NewsItem]] = Field(
        None,
        description="Trending news (only during mid-market)",
    )
    executive_summary: str = Field(
        ..., description="2-3 sentence executive summary"
    )
    key_takeaways: List[str] = Field(
        default_factory=list,
        description="Key takeaways for the user",
    )
    generation_metadata: Dict = Field(
        default_factory=dict,
        description="Metadata about the generation process",
    )


# =============================================================================
# Orchestrator Schemas (Simplified for 3 agents)
# =============================================================================


class AgentExecutionResult(BaseModel):
    """Result of a single agent execution."""

    agent_name: str = Field(..., description="Name of the agent")
    status: Literal["success", "partial", "failed"] = Field(
        ..., description="Execution status"
    )
    output: Optional[BaseModel] = Field(
        None, description="Agent output (if successful)"
    )
    error: Optional[str] = Field(None, description="Error message (if failed)")
    execution_time_ms: int = Field(..., description="Execution time in milliseconds")
    retry_count: int = Field(default=0, description="Number of retries attempted")

    class Config:
        arbitrary_types_allowed = True


class OrchestratorDecision(BaseModel):
    """Decision made by orchestrator after evaluating agent results."""

    can_proceed: bool = Field(..., description="Whether orchestration can continue")
    fallback_strategy: Optional[str] = Field(
        None, description="Fallback strategy to use"
    )
    degraded_mode: bool = Field(
        default=False, description="Whether running in degraded mode"
    )
    reasoning: str = Field(..., description="Explanation of the decision")
    skip_agents: List[str] = Field(
        default_factory=list, description="Agents to skip due to failures"
    )


class OrchestrationMetrics(BaseModel):
    """Metrics for the full orchestration run."""

    total_execution_time_ms: int = Field(
        ..., description="Total orchestration time"
    )
    agent_execution_times: Dict[str, int] = Field(
        ..., description="Execution time per agent"
    )
    agents_succeeded: List[str] = Field(
        default_factory=list, description="Successfully completed agents"
    )
    agents_failed: List[str] = Field(
        default_factory=list, description="Failed agents"
    )
    degraded_mode: bool = Field(
        default=False, description="Whether response is degraded"
    )
    cache_hits: int = Field(default=0, description="Number of cache hits")


# =============================================================================
# Legacy Schemas (kept for backward compatibility during migration)
# =============================================================================


class MarketDataAgentInput(BaseModel):
    """Legacy: Input schema for Market Data Agent."""
    selected_indices: List[str] = Field(...)
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    force_refresh: bool = Field(default=False)


class MarketDataAgentOutput(BaseModel):
    """Legacy: Output schema for Market Data Agent."""
    market_phase: Literal["pre", "mid", "post"] = Field(...)
    indices_data: Dict[str, IndexData] = Field(...)
    market_outlook: Optional[MarketOutlook] = Field(None)
    market_momentum: Literal[
        "strong_up", "moderate_up", "sideways", "moderate_down", "strong_down"
    ] = Field(...)
    data_freshness: datetime = Field(default_factory=datetime.utcnow)


class NewsAnalysisAgentInput(BaseModel):
    """Legacy: Input schema for News Analysis Agent."""
    market_phase: Literal["pre", "mid", "post"] = Field(...)
    selected_indices: List[str] = Field(...)
    time_window_hours: int = Field(default=24)
    max_articles: int = Field(default=50)
    watchlist: Optional[List[str]] = Field(None)
    portfolio_tickers: Optional[List[str]] = Field(None)


class NewsAnalysisAgentOutput(BaseModel):
    """Legacy: Output schema for News Analysis Agent."""
    raw_news: List[NewsItem] = Field(...)
    sentiment_distribution: Dict[str, int] = Field(...)
    preliminary_themes: List[PreliminaryTheme] = Field(...)
    key_topics: List[str] = Field(...)
    breaking_news: List[str] = Field(default_factory=list)
    processing_stats: Dict[str, int] = Field(default_factory=dict)


class UserContextAgentInput(BaseModel):
    """Legacy: Input schema for User Context Agent."""
    user_id: str = Field(...)
    include_watchlist: bool = Field(default=True)
    include_portfolio: bool = Field(default=True)
    news_filter: Literal["all", "watchlist", "portfolio"] = Field(default="all")


class UserContextAgentOutput(BaseModel):
    """Legacy: Output schema for User Context Agent."""
    user_id: str = Field(...)
    watchlist: List[str] = Field(default_factory=list)
    portfolio: List[PortfolioHolding] = Field(default_factory=list)
    sector_exposure: Dict[str, float] = Field(default_factory=dict)
    total_portfolio_value: float = Field(default=0.0)
    user_preferences: Dict = Field(default_factory=dict)
    risk_profile: Optional[Literal["conservative", "moderate", "aggressive"]] = Field(None)


class ImpactAnalysisAgentInput(BaseModel):
    """Legacy: Input schema for Impact Analysis Agent."""
    news_items: List[NewsItem] = Field(...)
    watchlist: List[str] = Field(default_factory=list)
    portfolio: List[PortfolioHolding] = Field(default_factory=list)
    indices_data: Dict[str, IndexData] = Field(...)
    sector_exposure: Dict[str, float] = Field(default_factory=dict)
    preliminary_themes: List[PreliminaryTheme] = Field(default_factory=list)


class ImpactAnalysisAgentOutput(BaseModel):
    """Legacy: Output schema for Impact Analysis Agent."""
    news_with_impacts: List[NewsWithImpact] = Field(...)
    portfolio_level_impact: PortfolioImpact = Field(...)
    watchlist_alerts: List[WatchlistAlert] = Field(default_factory=list)
    refined_themes: List[ThemeGroup] = Field(...)
    sector_impact_summary: Dict[str, str] = Field(default_factory=dict)
    causal_chains: List[str] = Field(default_factory=list)
