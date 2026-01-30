"""
API Response Models for Market Intelligence API endpoints.

The main endpoint /api/v1/market-summary returns data directly from
MongoDB snapshots, not through Pydantic response models.
"""

from datetime import datetime
from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field


class HealthCheckResponse(BaseModel):
    """Response model for health check endpoint."""

    status: Literal["healthy", "degraded", "unhealthy"] = Field(
        ..., description="Overall health status"
    )
    service: str = Field(
        default="market-intelligence-api",
        description="Service name",
    )
    version: str = Field(..., description="API version")
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="Health check timestamp",
    )
    agents: Dict[str, Literal["operational", "degraded", "down"]] = Field(
        default_factory=dict,
        description="Status of each background agent",
    )
    dependencies: Optional[Dict[str, bool]] = Field(
        None,
        description="Status of external dependencies (for deep check)",
    )


class AgentStatusResponse(BaseModel):
    """Response model for agent status endpoint."""

    agents: List[Dict] = Field(
        ..., description="List of agent status information"
    )
    total_agents: int = Field(..., description="Total number of agents")
    operational_agents: int = Field(..., description="Number of operational agents")


class ErrorResponse(BaseModel):
    """Standard error response model."""

    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Human-readable error message")
    request_id: Optional[str] = Field(
        None, description="Request ID for tracing"
    )
    details: Optional[Dict] = Field(
        None, description="Additional error details"
    )
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="Error timestamp",
    )

    class Config:
        json_schema_extra = {
            "example": {
                "error": "MARKET_SUMMARY_ERROR",
                "message": "Failed to fetch market summary",
                "request_id": "req_abc123",
                "details": {"reason": "MongoDB connection failed"},
                "timestamp": "2026-01-30T10:30:00Z",
            }
        }
