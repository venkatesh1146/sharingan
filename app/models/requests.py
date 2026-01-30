"""
API Request Models for Market Pulse endpoints.
"""

from typing import Literal, Optional

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
    force_refresh: bool = Field(
        default=False,
        description="Force refresh of cached data",
    )

    class Config:
        json_schema_extra = {
            "example": {
                "user_id": "user_123",
                "news_filter": "all",
                "max_news_items": 10,
                "max_themes": 5,
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
