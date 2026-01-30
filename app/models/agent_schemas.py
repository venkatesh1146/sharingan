"""
Agent Schemas for Background Processing Architecture.

These schemas are used by the 3 background processing agents:
1. NewsProcessingAgent - AI news analysis
2. SnapshotGenerationAgent - AI snapshot generation
3. IndicesCollectionAgent - Indices data collection

All agents are invoked by Celery tasks, not real-time API requests.
"""

from datetime import datetime
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


# =============================================================================
# Common Schemas
# =============================================================================


class AgentExecutionResult(BaseModel):
    """Result of a background agent execution (used by Celery tasks)."""

    agent_name: str = Field(..., description="Name of the agent")
    status: Literal["success", "partial", "failed", "skipped"] = Field(
        ..., description="Execution status"
    )
    output: Optional[Dict[str, Any]] = Field(
        None, description="Agent output data"
    )
    error: Optional[str] = Field(None, description="Error message (if failed)")
    execution_time_ms: int = Field(
        default=0, description="Execution time in milliseconds"
    )

    class Config:
        arbitrary_types_allowed = True


class TaskStatistics(BaseModel):
    """Statistics returned by Celery tasks."""

    fetched: int = Field(default=0, description="Items fetched from source")
    processed: int = Field(default=0, description="Items successfully processed")
    stored: int = Field(default=0, description="Items stored in database")
    skipped: int = Field(default=0, description="Items skipped (duplicates, etc.)")
    errors: int = Field(default=0, description="Processing errors")


# =============================================================================
# News Processing Agent Schemas
# =============================================================================


class NewsAnalysisResult(BaseModel):
    """Result of AI news analysis."""

    sentiment: Literal["bullish", "bearish", "neutral"] = Field(
        default="neutral", description="Analyzed sentiment"
    )
    sentiment_score: float = Field(
        default=0.0, ge=-1.0, le=1.0, description="Sentiment score (-1 to 1)"
    )
    summary: str = Field(default="", description="AI-generated summary")
    mentioned_stocks: List[str] = Field(
        default_factory=list, description="Stock tickers mentioned"
    )
    mentioned_sectors: List[str] = Field(
        default_factory=list, description="Sectors mentioned"
    )
    impacted_stocks: List[Dict[str, Any]] = Field(
        default_factory=list, description="Stocks impacted by this news"
    )
    sector_impacts: Dict[str, str] = Field(
        default_factory=dict, description="Impact on each sector"
    )
    causal_chain: str = Field(
        default="", description="Causal explanation of impact"
    )


# =============================================================================
# Snapshot Generation Agent Schemas
# =============================================================================


class SnapshotContent(BaseModel):
    """Content generated for a market snapshot."""

    market_outlook: Optional[Dict[str, Any]] = Field(
        None, description="Market outlook (pre/post only)"
    )
    market_summary: List[Dict[str, Any]] = Field(
        default_factory=list, description="Summary bullets with causal language"
    )
    executive_summary: Optional[str] = Field(
        None, description="Brief executive summary"
    )
    trending_now: Optional[List[str]] = Field(
        None, description="Trending news IDs (mid-market only)"
    )


# =============================================================================
# Indices Collection Agent Schemas
# =============================================================================


class IndicesCollectionResult(BaseModel):
    """Result of indices collection."""

    status: Literal["success", "skipped", "error"] = Field(
        ..., description="Collection status"
    )
    reason: Optional[str] = Field(None, description="Reason if skipped")
    fetched: int = Field(default=0, description="Indices fetched")
    stored: int = Field(default=0, description="Indices stored")
    errors: int = Field(default=0, description="Conversion errors")
