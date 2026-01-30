"""
Pydantic models for Market Intelligence API.

This module contains all data models organized as:
- domain.py: Core domain models shared across the system
- agent_schemas.py: Schemas for background processing agents
- requests.py: API request models
- responses.py: API response models
"""

from app.models.domain import (
    IndexData,
    MarketOutlook,
    NewsItem,
    ThemeGroup,
    ImpactedStock,
)
from app.models.agent_schemas import (
    AgentExecutionResult,
    TaskStatistics,
    NewsAnalysisResult,
    SnapshotContent,
    IndicesCollectionResult,
)
from app.models.responses import (
    HealthCheckResponse,
    AgentStatusResponse,
    ErrorResponse,
)

__all__ = [
    # Domain models
    "IndexData",
    "MarketOutlook",
    "NewsItem",
    "ThemeGroup",
    "ImpactedStock",
    # Agent schemas
    "AgentExecutionResult",
    "TaskStatistics",
    "NewsAnalysisResult",
    "SnapshotContent",
    "IndicesCollectionResult",
    # Response models
    "HealthCheckResponse",
    "AgentStatusResponse",
    "ErrorResponse",
]
