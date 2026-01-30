"""
API Request Models for Market Pulse endpoints.
"""

from typing import List, Literal, Optional

from pydantic import BaseModel, Field


class MarketPulseRequest(BaseModel):
    """
    Request model for the Market Pulse API endpoint.
    
    This is the main entry point for generating market pulse insights.
    """

    user_id: str = Field(
        ...,
        description="Unique user identifier",
        examples=["user_123"],
    )
    selected_indices: List[str] = Field(
        default=["NIFTY", "SENSEX", "GIFT NIFTY"],
        description="Market indices to analyze (use NIFTY, SENSEX, S&P 500, DJIA, etc.)",
        examples=[["NIFTY", "SENSEX", "S&P 500"]],
    )
    include_watchlist: bool = Field(
        default=True,
        description="Include user's watchlist in analysis",
    )
    include_portfolio: bool = Field(
        default=True,
        description="Include user's portfolio in analysis",
    )
    news_filter: Literal["all", "watchlist", "portfolio"] = Field(
        default="all",
        description="Filter news by relevance to watchlist/portfolio",
    )
    max_news_items: int = Field(
        default=50,
        ge=10,
        le=100,
        description="Maximum number of news items to process",
    )
    max_themes: int = Field(
        default=5,
        ge=1,
        le=10,
        description="Maximum number of themes to return",
    )
    include_portfolio_impact: bool = Field(
        default=True,
        description="Calculate and include portfolio impact analysis",
    )
    time_window_hours: int = Field(
        default=24,
        ge=1,
        le=72,
        description="Time window for news (in hours)",
    )
    force_refresh: bool = Field(
        default=False,
        description="Force refresh of cached data",
    )

    class Config:
        json_schema_extra = {
            "example": {
                "user_id": "user_123",
                "selected_indices": ["NIFTY", "SENSEX", "S&P 500", "DJIA"],
                "include_watchlist": True,
                "include_portfolio": True,
                "news_filter": "all",
                "max_news_items": 50,
                "max_themes": 5,
                "include_portfolio_impact": True,
                "time_window_hours": 24,
                "force_refresh": False,
            }
        }


class HealthCheckRequest(BaseModel):
    """Request model for health check with optional deep check."""

    deep_check: bool = Field(
        default=False,
        description="Perform deep health check including external dependencies",
    )


class AgentStatusRequest(BaseModel):
    """Request model for agent status endpoint."""

    agent_name: Optional[str] = Field(
        None,
        description="Specific agent to check (None for all agents)",
    )
    include_metrics: bool = Field(
        default=True,
        description="Include performance metrics",
    )
