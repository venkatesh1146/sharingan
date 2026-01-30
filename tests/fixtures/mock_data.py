"""
Mock data fixtures for testing.

Provides sample data for unit and integration tests.
"""

from datetime import datetime
from typing import List

from app.agents.base import AgentExecutionContext
from app.models.requests import MarketPulseRequest
from app.models.agent_schemas import MarketDataAgentOutput
from app.models.domain import (
    IndexData,
    MarketOutlook,
    NewsItem,
    PortfolioHolding,
)


def mock_market_pulse_request() -> MarketPulseRequest:
    """Create a mock MarketPulseRequest."""
    return MarketPulseRequest(
        user_id="test_user_123",
        news_filter="all",
        max_news_items=10,
        max_themes=5,
        force_refresh=False,
    )


def mock_execution_context() -> AgentExecutionContext:
    """Create a mock AgentExecutionContext."""
    return AgentExecutionContext(
        request_id="test_request_001",
        user_id="test_user_123",
        timestamp=datetime.utcnow(),
        trace_id="test_trace_001",
    )


def mock_market_data_output() -> MarketDataAgentOutput:
    """Create mock MarketDataAgentOutput."""
    return MarketDataAgentOutput(
        market_phase="pre",
        indices_data={
            "NIFTY 50": IndexData(
                ticker="NIFTY 50",
                name="NIFTY 50 Index",
                current_price=22150.50,
                change_percent=0.85,
                change_absolute=186.75,
                previous_close=21963.75,
                intraday_high=22200.00,
                intraday_low=21950.00,
                volume=250000000,
                timestamp=datetime.utcnow(),
            ),
            "SENSEX": IndexData(
                ticker="SENSEX",
                name="S&P BSE SENSEX",
                current_price=72850.25,
                change_percent=0.72,
                change_absolute=520.50,
                previous_close=72329.75,
                intraday_high=72950.00,
                intraday_low=72200.00,
                volume=180000000,
                timestamp=datetime.utcnow(),
            ),
        },
        market_outlook=MarketOutlook(
            sentiment="bullish",
            confidence=0.85,
            reasoning="NIFTY 50 is up 0.85%, indicating bullish sentiment",
            nifty_change_percent=0.85,
            key_drivers=["IT sector gains", "Banking strength"],
        ),
        market_momentum="moderate_up",
        data_freshness=datetime.utcnow(),
    )


def mock_news_items() -> List[NewsItem]:
    """Create mock news items."""
    return [
        NewsItem(
            id="news_001",
            headline="IT stocks rally as US tech earnings beat expectations",
            summary="Indian IT stocks including TCS, Infosys, and Wipro surged after major US tech companies reported better-than-expected quarterly earnings.",
            source="Economic Times",
            url="https://example.com/news/001",
            published_at=datetime.utcnow(),
            sentiment="bullish",
            sentiment_score=0.75,
            mentioned_stocks=["TCS", "INFY", "WIPRO"],
            mentioned_sectors=["IT", "Technology"],
            relevance_score=0.85,
            is_breaking=False,
        ),
        NewsItem(
            id="news_002",
            headline="RBI maintains repo rate, signals accommodative stance",
            summary="The Reserve Bank of India kept the repo rate unchanged at 6.5%.",
            source="Mint",
            url="https://example.com/news/002",
            published_at=datetime.utcnow(),
            sentiment="bullish",
            sentiment_score=0.65,
            mentioned_stocks=["HDFC", "ICICI", "SBI"],
            mentioned_sectors=["Banking", "Finance"],
            relevance_score=0.90,
            is_breaking=True,
        ),
        NewsItem(
            id="news_003",
            headline="Crude oil prices surge to $85 on OPEC+ supply concerns",
            summary="Brent crude oil prices jumped to $85 per barrel amid concerns about OPEC+ production cuts.",
            source="Reuters",
            url="https://example.com/news/003",
            published_at=datetime.utcnow(),
            sentiment="bearish",
            sentiment_score=-0.55,
            mentioned_stocks=["IOCL", "BPCL", "ASIANPAINT"],
            mentioned_sectors=["Oil & Gas", "Paints"],
            relevance_score=0.75,
            is_breaking=False,
        ),
    ]


def mock_portfolio_holdings() -> List[PortfolioHolding]:
    """Create mock portfolio holdings."""
    return [
        PortfolioHolding(
            ticker="RELIANCE",
            company_name="Reliance Industries Ltd",
            quantity=50,
            average_price=2450.00,
            current_price=2520.50,
            unrealized_pnl=3525.00,
            unrealized_pnl_percent=2.88,
            sector="Oil & Gas",
            weight_in_portfolio=0.35,
        ),
        PortfolioHolding(
            ticker="TCS",
            company_name="Tata Consultancy Services",
            quantity=30,
            average_price=3800.00,
            current_price=3950.25,
            unrealized_pnl=4507.50,
            unrealized_pnl_percent=3.96,
            sector="IT",
            weight_in_portfolio=0.33,
        ),
        PortfolioHolding(
            ticker="HDFC",
            company_name="HDFC Bank Ltd",
            quantity=100,
            average_price=1580.00,
            current_price=1620.75,
            unrealized_pnl=4075.00,
            unrealized_pnl_percent=2.58,
            sector="Banking",
            weight_in_portfolio=0.32,
        ),
    ]
