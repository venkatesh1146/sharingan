"""
Core domain models shared across the Market Pulse Multi-Agent system.

These models represent the fundamental data structures used throughout
the application, including market data, news items, portfolio holdings,
and analysis results.
"""

from datetime import datetime
from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field, field_validator


# =============================================================================
# Market Data Models
# =============================================================================


class IndexData(BaseModel):
    """Represents data for a single market index."""

    ticker: str = Field(..., description="Index ticker symbol (e.g., 'NIFTY', 'SENSEX')")
    name: str = Field(..., description="Full name of the index (e.g., 'Nifty', 'BSE Sensex')")
    country: str = Field(default="Unknown", description="Country where the index is based (e.g., 'India', 'United States')")
    current_price: float = Field(..., description="Current price/value of the index")
    change_percent: float = Field(..., description="Percentage change from previous close")
    change_absolute: float = Field(..., description="Absolute change from previous close")
    previous_close: float = Field(..., description="Previous trading day's closing value")
    intraday_high: float = Field(..., description="Highest value during current session")
    intraday_low: float = Field(..., description="Lowest value during current session")
    volume: int = Field(default=0, description="Trading volume (if applicable)")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Data timestamp")

    @field_validator("change_percent")
    @classmethod
    def round_change_percent(cls, v: float) -> float:
        """Round change percentage to 2 decimal places."""
        return round(v, 2)


class MarketOutlook(BaseModel):
    """
    Market outlook assessment based on NIFTY 50 performance.
    
    Only generated during pre-market and post-market phases.
    Hidden during mid-market trading hours.
    """

    sentiment: Literal["bullish", "bearish", "neutral"] = Field(
        ..., description="Overall market sentiment"
    )
    confidence: float = Field(
        ..., ge=0.0, le=1.0, description="Confidence score (0-1)"
    )
    reasoning: str = Field(..., description="Explanation for the outlook assessment")
    nifty_change_percent: float = Field(
        ..., description="NIFTY 50 change percentage used for determination"
    )
    key_drivers: List[str] = Field(
        default_factory=list, description="Key factors driving the outlook"
    )

    @field_validator("sentiment", mode="before")
    @classmethod
    def determine_sentiment(cls, v: str) -> str:
        """Ensure sentiment is lowercase."""
        return v.lower() if isinstance(v, str) else v


# =============================================================================
# News Models
# =============================================================================


class NewsItem(BaseModel):
    """Represents a single news article."""

    id: str = Field(..., description="Unique identifier for the news item")
    headline: str = Field(..., description="News headline/title")
    summary: str = Field(..., description="Brief summary of the news")
    source: str = Field(..., description="News source (e.g., 'Economic Times')")
    url: Optional[str] = Field(None, description="URL to the full article")
    published_at: datetime = Field(..., description="Publication timestamp")
    sentiment: Literal["bullish", "bearish", "neutral"] = Field(
        ..., description="Sentiment classification"
    )
    sentiment_score: float = Field(
        default=0.0, ge=-1.0, le=1.0, description="Sentiment score (-1 to 1)"
    )
    mentioned_stocks: List[str] = Field(
        default_factory=list, description="Stock tickers mentioned in the news"
    )
    mentioned_sectors: List[str] = Field(
        default_factory=list, description="Sectors mentioned in the news"
    )
    relevance_score: float = Field(
        default=0.5, ge=0.0, le=1.0, description="Relevance score for the user"
    )
    is_breaking: bool = Field(default=False, description="Whether this is breaking news")


class PreliminaryTheme(BaseModel):
    """Initial theme grouping from news analysis."""

    theme_name: str = Field(..., description="Name of the theme")
    news_ids: List[str] = Field(..., description="IDs of news items in this theme")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Clustering confidence")
    mentioned_stocks: List[str] = Field(
        default_factory=list, description="Stocks related to this theme"
    )
    sentiment: Literal["bullish", "bearish", "neutral", "mixed"] = Field(
        default="neutral", description="Overall theme sentiment"
    )


class ThemeGroup(BaseModel):
    """
    Refined theme group with full analysis.
    
    Contains clustered news with impact analysis and causal reasoning.
    """

    theme_id: str = Field(..., description="Unique identifier for the theme")
    theme_name: str = Field(..., description="Human-readable theme name")
    theme_description: str = Field(..., description="Brief description of the theme")
    news_items: List[NewsItem] = Field(..., description="News items in this theme")
    overall_sentiment: Literal["bullish", "bearish", "neutral", "mixed"] = Field(
        ..., description="Aggregated sentiment for the theme"
    )
    impacted_sectors: List[str] = Field(
        default_factory=list, description="Sectors impacted by this theme"
    )
    impacted_stocks: List[str] = Field(
        default_factory=list, description="Stocks impacted by this theme"
    )
    causal_summary: str = Field(
        ..., description="Causal explanation of the theme's impact"
    )
    relevance_to_user: float = Field(
        default=0.5, ge=0.0, le=1.0, description="Relevance to user's portfolio/watchlist"
    )


# =============================================================================
# Portfolio & User Models
# =============================================================================


class PortfolioHolding(BaseModel):
    """Represents a single holding in user's portfolio."""

    ticker: str = Field(..., description="Stock ticker symbol")
    company_name: str = Field(..., description="Full company name")
    quantity: int = Field(..., ge=0, description="Number of shares held")
    average_price: float = Field(..., gt=0, description="Average purchase price")
    current_price: float = Field(..., gt=0, description="Current market price")
    unrealized_pnl: float = Field(..., description="Unrealized profit/loss")
    unrealized_pnl_percent: float = Field(..., description="Unrealized P&L percentage")
    sector: str = Field(..., description="Stock's sector classification")
    weight_in_portfolio: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Weight in total portfolio"
    )

    @property
    def current_value(self) -> float:
        """Calculate current value of the holding."""
        return self.quantity * self.current_price

    @property
    def invested_value(self) -> float:
        """Calculate invested value of the holding."""
        return self.quantity * self.average_price


# =============================================================================
# Impact Analysis Models
# =============================================================================


class ImpactedStock(BaseModel):
    """Stock with impact analysis from news."""

    ticker: str = Field(..., description="Stock ticker symbol")
    company_name: str = Field(..., description="Company name")
    impact_type: Literal["positive", "negative", "neutral"] = Field(
        ..., description="Type of impact"
    )
    impact_magnitude: Literal["high", "medium", "low"] = Field(
        ..., description="Magnitude of impact"
    )
    reasoning: str = Field(..., description="Why this stock is impacted")
    related_news_ids: List[str] = Field(
        default_factory=list, description="News items causing this impact"
    )
    in_portfolio: bool = Field(default=False, description="Whether stock is in user's portfolio")
    in_watchlist: bool = Field(default=False, description="Whether stock is in user's watchlist")


class NewsWithImpact(BaseModel):
    """News item enriched with impact analysis."""

    news_id: str = Field(..., description="Reference to original news item")
    news_item: NewsItem = Field(..., description="The original news item")
    impacted_stocks: List[ImpactedStock] = Field(
        default_factory=list, description="Stocks impacted by this news"
    )
    sector_impacts: Dict[str, Literal["positive", "negative", "neutral"]] = Field(
        default_factory=dict, description="Impact on each sector"
    )
    causal_chain: str = Field(
        ..., description="Causal chain explaining the impact (e.g., 'Oil ↑ → Paints ↓')"
    )
    impact_confidence: float = Field(
        default=0.5, ge=0.0, le=1.0, description="Confidence in impact analysis"
    )


class PortfolioImpact(BaseModel):
    """Aggregate impact on user's portfolio."""

    overall_sentiment: Literal["positive", "negative", "neutral", "mixed"] = Field(
        ..., description="Overall portfolio sentiment"
    )
    estimated_impact_percent: float = Field(
        ..., description="Estimated portfolio impact percentage"
    )
    top_affected_holdings: List[str] = Field(
        ..., max_length=5, description="Top affected holdings (tickers)"
    )
    positive_drivers: List[str] = Field(
        default_factory=list, description="Positive impact drivers"
    )
    negative_drivers: List[str] = Field(
        default_factory=list, description="Negative impact drivers"
    )
    reasoning: str = Field(..., description="Explanation of portfolio impact")


class WatchlistAlert(BaseModel):
    """Alert for a watchlist stock."""

    ticker: str = Field(..., description="Stock ticker symbol")
    company_name: str = Field(..., description="Company name")
    alert_type: Literal["opportunity", "risk", "neutral"] = Field(
        ..., description="Type of alert"
    )
    alert_priority: Literal["high", "medium", "low"] = Field(
        default="medium", description="Alert priority"
    )
    related_news_ids: List[str] = Field(
        default_factory=list, description="Related news items"
    )
    reasoning: str = Field(..., description="Why this alert was generated")
    suggested_action: Optional[str] = Field(
        None, description="Suggested action (informational only)"
    )


# =============================================================================
# Summary Models
# =============================================================================


class MarketSummaryBullet(BaseModel):
    """
    A single market summary bullet point.
    
    MUST contain causal language explaining market movements.
    """

    text: str = Field(..., description="Summary text with causal language")
    supporting_news_ids: List[str] = Field(
        default_factory=list, description="News IDs supporting this summary"
    )
    confidence: float = Field(
        default=0.8, ge=0.0, le=1.0, description="Confidence in the summary"
    )
    sentiment: Literal["bullish", "bearish", "neutral"] = Field(
        default="neutral", description="Sentiment of this bullet"
    )

    @field_validator("text")
    @classmethod
    def validate_causal_language(cls, v: str) -> str:
        """
        Validate that summary contains causal language.
        
        Required keywords: due to, after, following, as, because, driven by, on account of
        """
        causal_keywords = [
            "due to",
            "after",
            "following",
            "as ",
            "because",
            "driven by",
            "on account of",
            "amid",
            "on the back of",
            "triggered by",
            "led by",
            "supported by",
            "weighed by",
        ]
        text_lower = v.lower()
        if not any(keyword in text_lower for keyword in causal_keywords):
            raise ValueError(
                f"Summary must contain causal language (e.g., 'due to', 'driven by'). Got: {v}"
            )
        return v
