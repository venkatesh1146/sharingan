"""
API Request Models for Market Intelligence API endpoints.
"""

from typing import Optional

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
