"""
Market Snapshot Document - MongoDB schema for aggregated market views.

Stores periodically generated market snapshots with AI analysis,
indices data, and news summaries.
"""

from datetime import datetime, timedelta
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


class MarketOutlookDocument(BaseModel):
    """Market outlook embedded in snapshot."""
    
    sentiment: Literal["bullish", "bearish", "neutral"]
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str
    nifty_change_percent: float
    key_drivers: List[str] = Field(default_factory=list)


class IndexDataDocument(BaseModel):
    """Index data embedded in snapshot."""
    
    ticker: str
    name: str
    country: str = "Unknown"
    current_price: float
    change_percent: float
    change_absolute: float
    previous_close: float
    intraday_high: float
    intraday_low: float
    volume: int = 0
    timestamp: datetime


class MarketSummaryBulletDocument(BaseModel):
    """Market summary bullet embedded in snapshot."""
    
    text: str = Field(..., description="Summary text with causal language")
    supporting_news_ids: List[str] = Field(default_factory=list)
    confidence: float = Field(default=0.8, ge=0.0, le=1.0)
    sentiment: Literal["bullish", "bearish", "neutral"] = "neutral"


class MarketSnapshotDocument(BaseModel):
    """
    MongoDB document schema for market_snapshots collection.
    
    Stores aggregated market views generated periodically (every 15-30 min).
    """
    
    # Primary identifier
    snapshot_id: str = Field(
        ..., 
        description="Unique snapshot ID (e.g., snapshot_2026-01-30_15:21_post)"
    )
    
    # Timestamps
    generated_at: datetime = Field(default_factory=datetime.utcnow)
    request_id: Optional[str] = Field(None, description="Request ID if triggered manually")
    
    # Market status
    market_phase: Literal["pre", "mid", "post"] = Field(
        ..., 
        description="Market phase when snapshot was generated"
    )
    
    # Market analysis
    market_outlook: Optional[MarketOutlookDocument] = Field(
        None, 
        description="Market outlook (null during mid-market)"
    )
    
    # Indices data
    indices_data: List[IndexDataDocument] = Field(
        default_factory=list,
        description="Current indices data"
    )
    
    # Summary content
    market_summary: List[MarketSummaryBulletDocument] = Field(
        default_factory=list,
        description="Market summary bullets with causal language"
    )
    executive_summary: Optional[str] = Field(
        None, 
        description="Brief executive summary"
    )
    
    # Trending content (mid-market only)
    trending_now: Optional[List[str]] = Field(
        None,
        description="Trending news IDs (mid-market only)"
    )
    
    # All news referenced
    all_news_ids: List[str] = Field(
        default_factory=list,
        description="All news IDs included in this snapshot"
    )
    
    # Metadata
    degraded_mode: bool = Field(default=False)
    warnings: List[str] = Field(default_factory=list)
    expires_at: datetime = Field(
        default_factory=lambda: datetime.utcnow() + timedelta(minutes=15)
    )
    created_at: datetime = Field(default_factory=datetime.utcnow)
    version: str = Field(default="1.0")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

    def to_mongo_dict(self) -> Dict[str, Any]:
        """Convert to MongoDB-compatible dictionary."""
        data = self.model_dump()
        
        # Convert nested models
        if data.get("market_outlook"):
            data["market_outlook"] = dict(data["market_outlook"])
        
        data["indices_data"] = [dict(idx) for idx in data.get("indices_data", [])]
        data["market_summary"] = [dict(ms) for ms in data.get("market_summary", [])]
        
        return data

    @classmethod
    def from_mongo_dict(cls, data: Dict[str, Any]) -> "MarketSnapshotDocument":
        """Create instance from MongoDB document."""
        data.pop("_id", None)
        return cls(**data)

    @classmethod
    def generate_snapshot_id(cls, market_phase: str) -> str:
        """
        Generate a unique snapshot ID.
        
        Format: snapshot_YYYY-MM-DD_HH:MM_phase
        
        Args:
            market_phase: Current market phase (pre/mid/post)
            
        Returns:
            Unique snapshot ID string
        """
        now = datetime.utcnow()
        return f"snapshot_{now.strftime('%Y-%m-%d_%H:%M')}_{market_phase}"
