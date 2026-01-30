"""
Indices Timeseries Document - MongoDB schema for historical indices data.

Stores time-series data for market indices for historical tracking
and trend analysis.
"""

from datetime import datetime
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field


class IndicesDataDocument(BaseModel):
    """Embedded indices data structure."""
    
    name: str
    country: str = "Unknown"
    current_price: float
    change_percent: float
    change_absolute: float
    previous_close: float
    intraday_high: float
    intraday_low: float
    volume: int = 0


class IndicesTimeseriesDocument(BaseModel):
    """
    MongoDB document schema for indices_timeseries collection.
    
    Stores historical indices data for trend analysis and backtesting.
    TTL-indexed for automatic cleanup after 90 days.
    """
    
    # Index identifier
    ticker: str = Field(..., description="Index ticker symbol (e.g., NIFTY, SENSEX)")
    
    # Timestamp
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="Data point timestamp"
    )
    
    # Index data
    data: IndicesDataDocument = Field(..., description="Index data snapshot")
    
    # Metadata
    created_at: datetime = Field(default_factory=datetime.utcnow)
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

    def to_mongo_dict(self) -> Dict[str, Any]:
        """Convert to MongoDB-compatible dictionary."""
        result = self.model_dump()
        result["data"] = dict(result["data"])
        return result

    @classmethod
    def from_mongo_dict(cls, data: Dict[str, Any]) -> "IndicesTimeseriesDocument":
        """Create instance from MongoDB document."""
        data.pop("_id", None)
        return cls(**data)

    @classmethod
    def from_world_indices_response(
        cls,
        ticker: str,
        index_data: Dict[str, Any],
    ) -> "IndicesTimeseriesDocument":
        """
        Create from World Indices API response.
        
        Args:
            ticker: Normalized ticker symbol
            index_data: Raw index data from API
            
        Returns:
            IndicesTimeseriesDocument instance
        """
        # Parse timestamp from API
        timestamp = datetime.utcnow()
        date_str = index_data.get("date", "")
        if date_str:
            try:
                timestamp = datetime.fromisoformat(date_str)
            except (ValueError, TypeError):
                pass
        
        return cls(
            ticker=ticker,
            timestamp=timestamp,
            data=IndicesDataDocument(
                name=index_data.get("indexname", ticker),
                country=index_data.get("Country", "Unknown"),
                current_price=float(index_data.get("close", 0)),
                change_percent=float(index_data.get("PChg", 0)),
                change_absolute=float(index_data.get("Chg", 0)),
                previous_close=float(index_data.get("PrevClose", 0)),
                intraday_high=float(index_data.get("close", 0)),
                intraday_low=float(index_data.get("close", 0)),
                volume=0,
            ),
        )
