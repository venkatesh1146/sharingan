"""
Market Intelligence Agent - Combined market data and news analysis.

This agent merges the functionality of:
- MarketDataAgent: Fetches market indices, determines phase, calculates outlook
- NewsAnalysisAgent: Fetches news, analyzes sentiment, clusters into themes

Responsibilities:
- Fetch and analyze market indices
- Determine market phase (pre/mid/post)
- Fetch and analyze market news
- Cluster news into themes
- Calculate market outlook and momentum
"""

from datetime import datetime
from typing import Dict, List, Optional

from app.agents.base import BaseAgent, AgentConfig, AgentExecutionContext
from app.config import get_settings
from app.models.agent_schemas import (
    MarketIntelligenceAgentInput,
    MarketIntelligenceAgentOutput,
)
from app.models.domain import IndexData, MarketOutlook, NewsItem, PreliminaryTheme
from app.prompts.market_intelligence_prompts import MARKET_INTELLIGENCE_SYSTEM_PROMPT
from app.services.market_intelligence_service import (
    fetch_market_intelligence,
    get_market_intelligence_tools,
    get_market_intelligence_tool_handlers,
)


class MarketIntelligenceAgent(BaseAgent[MarketIntelligenceAgentInput, MarketIntelligenceAgentOutput]):
    """
    Combined agent for market data and news analysis.
    
    This agent consolidates market data fetching and news analysis:
    - Fetches current market index data
    - Determines market phase (pre-market, mid-market, post-market)
    - Calculates market outlook based on NIFTY movement
    - Fetches and analyzes market news
    - Clusters news into thematic groups
    """

    input_schema = MarketIntelligenceAgentInput
    output_schema = MarketIntelligenceAgentOutput

    def __init__(self):
        settings = get_settings()
        config = AgentConfig(
            name="market_intelligence_agent",
            description="Fetches market data and analyzes news in a unified workflow",
            model_name=settings.GEMINI_FAST_MODEL,
            temperature=0.1,  # Mostly deterministic with slight flexibility for analysis
            max_output_tokens=4096,
            timeout_seconds=settings.MARKET_DATA_AGENT_TIMEOUT + 10,  # Combined timeout
            retry_attempts=2,
        )
        super().__init__(config)

        # Register tool handlers
        for name, handler in get_market_intelligence_tool_handlers().items():
            self.register_tool_handler(name, handler)

    def get_system_prompt(self) -> str:
        return MARKET_INTELLIGENCE_SYSTEM_PROMPT

    def get_tools(self):
        return get_market_intelligence_tools()

    async def execute(
        self,
        input_data: MarketIntelligenceAgentInput,
        context: AgentExecutionContext,
    ) -> MarketIntelligenceAgentOutput:
        """
        Execute market intelligence gathering.
        
        Steps:
        1. Fetch comprehensive market intelligence (indices + news)
        2. Process indices data and calculate outlook
        3. Process news and create sentiment distribution
        4. Cluster news into preliminary themes
        """
        self.logger.info(
            "executing_market_intelligence",
            selected_indices=input_data.selected_indices,
            time_window_hours=input_data.time_window_hours,
            request_id=context.request_id,
        )

        # Step 1: Fetch all market intelligence in one call
        intelligence_data = await fetch_market_intelligence(
            indices=input_data.selected_indices,
            watchlist=input_data.watchlist,
            time_window_hours=input_data.time_window_hours,
            max_articles=input_data.max_articles,
        )

        # Step 2: Process market phase
        phase_data = intelligence_data["market_phase"]
        market_phase = phase_data["phase"]

        self.logger.info(
            "market_phase_determined",
            phase=market_phase,
            is_trading_day=phase_data.get("is_trading_day"),
        )

        # Step 3: Process indices data (from World Indices API)
        raw_indices = intelligence_data["indices_data"]
        indices_data = {}
        for ticker, data in raw_indices.items():
            if "error" not in data:
                indices_data[ticker] = IndexData(
                    ticker=data["ticker"],
                    name=data.get("name", ticker),
                    country=data.get("country", "Unknown"),
                    current_price=data["current_price"],
                    change_percent=data["change_percent"],
                    change_absolute=data["change_absolute"],
                    previous_close=data.get("previous_close", 0),
                    intraday_high=data.get("intraday_high", data["current_price"]),
                    intraday_low=data.get("intraday_low", data["current_price"]),
                    volume=data.get("volume", 0),
                    timestamp=datetime.fromisoformat(data["timestamp"])
                    if isinstance(data.get("timestamp"), str)
                    else datetime.utcnow(),
                )

        # Step 4: Calculate market outlook (only for pre/post market)
        market_outlook: Optional[MarketOutlook] = None
        if market_phase != "mid":
            # Use NIFTY (or SENSEX as fallback) for outlook calculation
            nifty_data = indices_data.get("NIFTY", indices_data.get("SENSEX"))
            if nifty_data:
                nifty_change = nifty_data.change_percent

                if nifty_change > 0.5:
                    sentiment = "bullish"
                    confidence = min(0.5 + (nifty_change / 2), 0.95)
                elif nifty_change < -0.5:
                    sentiment = "bearish"
                    confidence = min(0.5 + (abs(nifty_change) / 2), 0.95)
                else:
                    sentiment = "neutral"
                    confidence = 0.7

                market_outlook = MarketOutlook(
                    sentiment=sentiment,
                    confidence=round(confidence, 2),
                    reasoning=self._generate_outlook_reasoning(
                        sentiment, nifty_change, indices_data
                    ),
                    nifty_change_percent=nifty_change,
                    key_drivers=self._identify_key_drivers(indices_data),
                )

        # Step 5: Get market momentum
        momentum_data = intelligence_data["momentum"]
        market_momentum = momentum_data["momentum"]

        # Step 6: Process news items
        raw_news = intelligence_data["news"]
        news_items: List[NewsItem] = []
        for news in raw_news:
            news_items.append(
                NewsItem(
                    id=news["id"],
                    headline=news["headline"],
                    summary=news.get("summary", ""),
                    source=news.get("source", "Unknown"),
                    url=news.get("url"),
                    published_at=datetime.fromisoformat(news["published_at"])
                    if isinstance(news.get("published_at"), str)
                    else datetime.utcnow(),
                    sentiment=news.get("sentiment", "neutral"),
                    sentiment_score=news.get("sentiment_score", 0.0),
                    mentioned_stocks=news.get("mentioned_stocks", []),
                    mentioned_sectors=news.get("mentioned_sectors", []),
                    relevance_score=news.get("relevance_score", 0.5),
                    is_breaking=news.get("is_breaking", False),
                )
            )

        # Step 7: Calculate sentiment distribution
        sentiment_distribution = self._calculate_sentiment_distribution(news_items)

        # Step 8: Process themes
        raw_themes = intelligence_data["themes"]
        preliminary_themes: List[PreliminaryTheme] = []
        for theme in raw_themes:
            preliminary_themes.append(
                PreliminaryTheme(
                    theme_name=theme["theme_name"],
                    news_ids=theme["news_ids"],
                    confidence=theme.get("confidence", 0.7),
                    mentioned_stocks=theme.get("mentioned_stocks", []),
                    sentiment=theme.get("sentiment", "neutral"),
                )
            )

        # Step 9: Extract key topics
        key_topics = self._extract_key_topics(preliminary_themes)

        # Step 10: Identify breaking news
        breaking_news = [n.id for n in news_items if n.is_breaking]

        self.logger.info(
            "market_intelligence_complete",
            market_phase=market_phase,
            outlook=market_outlook.sentiment if market_outlook else "hidden",
            momentum=market_momentum,
            total_news=len(news_items),
            themes=len(preliminary_themes),
            breaking_count=len(breaking_news),
        )

        return MarketIntelligenceAgentOutput(
            # Market data outputs
            market_phase=market_phase,
            indices_data=indices_data,
            market_outlook=market_outlook,
            market_momentum=market_momentum,
            data_freshness=datetime.utcnow(),
            # News analysis outputs
            news_items=news_items,
            sentiment_distribution=sentiment_distribution,
            preliminary_themes=preliminary_themes,
            key_topics=key_topics,
            breaking_news=breaking_news,
        )

    def _generate_outlook_reasoning(
        self,
        sentiment: str,
        nifty_change: float,
        indices_data: dict,
    ) -> str:
        """Generate reasoning for market outlook."""
        direction = "up" if nifty_change > 0 else "down"
        abs_change = abs(nifty_change)

        changes = [
            (ticker, data.change_percent)
            for ticker, data in indices_data.items()
        ]
        changes.sort(key=lambda x: x[1], reverse=True)

        top_performer = changes[0] if changes else ("N/A", 0)
        worst_performer = changes[-1] if changes else ("N/A", 0)

        if sentiment == "bullish":
            return (
                f"NIFTY 50 is {direction} {abs_change:.2f}%, indicating bullish sentiment. "
                f"{top_performer[0]} leads with {top_performer[1]:.2f}% gain."
            )
        elif sentiment == "bearish":
            return (
                f"NIFTY 50 is {direction} {abs_change:.2f}%, indicating bearish sentiment. "
                f"{worst_performer[0]} is down {abs(worst_performer[1]):.2f}%."
            )
        else:
            return (
                f"NIFTY 50 is relatively flat at {nifty_change:.2f}%, "
                f"indicating neutral/consolidating market conditions."
            )

    def _identify_key_drivers(self, indices_data: dict) -> List[str]:
        """Identify key drivers for market movement."""
        drivers = []

        for ticker, data in indices_data.items():
            if abs(data.change_percent) > 1.0:
                direction = "gains" if data.change_percent > 0 else "losses"
                drivers.append(f"{ticker} {direction}")

        return drivers[:3] if drivers else ["Broad-based movement"]

    def _calculate_sentiment_distribution(
        self,
        news_items: List[NewsItem],
    ) -> Dict[str, int]:
        """Calculate distribution of sentiments across news."""
        distribution = {"bullish": 0, "bearish": 0, "neutral": 0}

        for news in news_items:
            sentiment = news.sentiment.lower()
            if sentiment in distribution:
                distribution[sentiment] += 1

        return distribution

    def _extract_key_topics(
        self,
        themes: List[PreliminaryTheme],
    ) -> List[str]:
        """Extract key topics from themes."""
        topics = []

        for theme in themes[:5]:
            topic = theme.theme_name.replace(" Update", "").strip()
            if topic and topic not in topics:
                topics.append(topic)

        return topics
