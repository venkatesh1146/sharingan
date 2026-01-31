"""
API Request Models for Market Intelligence API endpoints.
"""

from typing import Optional, List

from pydantic import BaseModel, Field


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


class NewsSearchRequest(BaseModel):
    """Request model for searching news by stocks and/or companies."""

    mentioned_stocks: Optional[List[str]] = Field(
        default=None,
        description="List of stock tickers to search for (e.g., ['RELIANCE', 'TCS'])",
    )
    mentioned_companies: Optional[List[str]] = Field(
        default=None,
        description="List of company names to search for",
    )
    hours: int = Field(
        default=24,
        description="How many hours back to search (default: 24)",
        ge=1,
        le=730,
    )
    limit: int = Field(
        default=50,
        description="Maximum number of news articles to return (default: 50)",
        ge=1,
        le=500,
    )
