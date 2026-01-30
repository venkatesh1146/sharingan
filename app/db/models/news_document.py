"""
News Article Document - MongoDB schema for news articles.

Stores processed news articles with AI-generated analysis,
sentiment, and entity extraction results.
"""

from datetime import datetime
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


class ImpactedStockDocument(BaseModel):
    """Stock impacted by a news article."""
    
    ticker: str
    impact_type: Literal["positive", "negative", "neutral"]
    impact_magnitude: Literal["high", "medium", "low"] = "medium"
    reasoning: str = ""


class NewsArticleDocument(BaseModel):
    """
    MongoDB document schema for news_articles collection.
    
    Stores individual news articles with AI-generated summaries,
    sentiment analysis, and entity extraction.
    """
    
    # Primary identifier (from CMOTS 'sno' field)
    news_id: str = Field(..., description="Unique news identifier (sno from API)")
    
    # Content fields
    headline: str = Field(..., description="News headline")
    summary: str = Field(default="", description="AI-generated concise summary")
    full_text: Optional[str] = Field(None, description="Original source summary/text before AI processing")
    
    # Source information
    source: str = Field(default="Capital Market", description="News source")
    source_url: Optional[str] = Field(None, description="URL to original article")
    published_at: datetime = Field(..., description="Publication timestamp")
    fetched_at: datetime = Field(default_factory=datetime.utcnow, description="When fetched")
    
    # AI Analysis results
    sentiment: Literal["bullish", "bearish", "neutral"] = Field(
        default="neutral", 
        description="AI-analyzed sentiment"
    )
    sentiment_score: float = Field(
        default=0.0, 
        ge=-1.0, 
        le=1.0, 
        description="Sentiment score (-1 to 1)"
    )
    is_breaking: bool = Field(default=False, description="Breaking news flag")
    
    # Entity extraction
    mentioned_stocks: List[str] = Field(
        default_factory=list, 
        description="Stock tickers mentioned"
    )
    mentioned_sectors: List[str] = Field(
        default_factory=list, 
        description="Sectors mentioned"
    )
    mentioned_companies: List[str] = Field(
        default_factory=list, 
        description="Company names mentioned"
    )
    
    # Impact analysis
    impacted_stocks: List[ImpactedStockDocument] = Field(
        default_factory=list,
        description="Stocks impacted by this news"
    )
    sector_impacts: Dict[str, Literal["positive", "negative", "neutral"]] = Field(
        default_factory=dict,
        description="Impact on each sector"
    )
    causal_chain: str = Field(
        default="",
        description="Causal chain explaining impact"
    )
    
    # Categorization
    category: str = Field(default="general", description="News category")
    subcategory: Optional[str] = Field(None, description="News subcategory")
    news_type: str = Field(default="", description="News type from API")
    
    # Processing status
    processed: bool = Field(default=False, description="Whether basic processing is done")
    analyzed: bool = Field(default=False, description="Whether AI analysis is complete")
    included_in_snapshots: List[str] = Field(
        default_factory=list,
        description="Snapshot IDs that include this news"
    )
    
    # Metadata
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

    def to_mongo_dict(self) -> Dict[str, Any]:
        """Convert to MongoDB-compatible dictionary."""
        data = self.model_dump()
        # Ensure datetime fields are proper datetime objects
        for field in ["published_at", "fetched_at", "created_at", "updated_at"]:
            if isinstance(data.get(field), str):
                data[field] = datetime.fromisoformat(data[field])
        return data

    @classmethod
    def from_mongo_dict(cls, data: Dict[str, Any]) -> "NewsArticleDocument":
        """Create instance from MongoDB document."""
        # Remove MongoDB's _id field if present
        data.pop("_id", None)
        return cls(**data)

    @classmethod
    def from_cmots_news(
        cls,
        news_item: Dict[str, Any],
        news_type: str = "",
    ) -> "NewsArticleDocument":
        """
        Create NewsArticleDocument from CMOTS API news item.
        
        Args:
            news_item: Raw news item from CMOTS API
            news_type: News type category
            
        Returns:
            NewsArticleDocument instance
        """
        # Parse published_at from date and time fields
        published_at = datetime.utcnow()
        date_str = news_item.get("date", "")
        time_str = news_item.get("time", "")
        
        if date_str:
            try:
                from datetime import datetime as dt
                base_dt = dt.strptime(date_str, "%m/%d/%Y %I:%M:%S %p")
                if time_str:
                    time_dt = dt.strptime(time_str, "%H:%M")
                    published_at = base_dt.replace(
                        hour=time_dt.hour,
                        minute=time_dt.minute,
                        second=0,
                    )
                else:
                    published_at = base_dt
            except (ValueError, TypeError):
                pass
        
        # Get original summary from source
        original_summary = news_item.get("summary", "")
        
        return cls(
            news_id=str(news_item.get("sno", "")),
            headline=news_item.get("heading", ""),
            summary=original_summary,  # Will be replaced by AI summary after analysis
            full_text=original_summary,  # Preserve original source summary
            source=news_item.get("section_name", "Capital Market"),
            published_at=published_at,
            news_type=news_type,
            processed=False,
            analyzed=False,
        )
