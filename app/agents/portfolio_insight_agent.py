"""
Portfolio Insight Agent - Combined user context and impact analysis.

This agent merges the functionality of:
- UserContextAgent: Fetches user watchlist, portfolio, preferences
- ImpactAnalysisAgent: Connects news to stocks, causal reasoning, portfolio impact

Responsibilities:
- Fetch user watchlist and portfolio
- Calculate sector exposure
- Analyze news impact on user's holdings
- Generate portfolio-level impact analysis
- Create watchlist alerts
- Build causal chains
"""

from typing import Dict, List, Optional

from app.agents.base import BaseAgent, AgentConfig, AgentExecutionContext
from app.config import get_settings
from app.models.agent_schemas import (
    PortfolioInsightAgentInput,
    PortfolioInsightAgentOutput,
)
from app.models.domain import (
    NewsItem,
    NewsWithImpact,
    ImpactedStock,
    PortfolioHolding,
    PortfolioImpact,
    WatchlistAlert,
    ThemeGroup,
    PreliminaryTheme,
    IndexData,
)
from app.prompts.portfolio_insight_prompts import PORTFOLIO_INSIGHT_SYSTEM_PROMPT
# TODO: Create user_data_tools.py with fetch_user_watchlist, fetch_user_portfolio, etc.
# from app.tools.user_data_tools import (
#     fetch_user_watchlist,
#     fetch_user_portfolio,
#     calculate_sector_exposure,
#     get_user_preferences,
# )
from app.tools.analysis_tools import (
    analyze_supply_chain_impact,
    get_analysis_tools,
    get_analysis_tool_handlers,
)


class PortfolioInsightAgent(BaseAgent[PortfolioInsightAgentInput, PortfolioInsightAgentOutput]):
    """
    Combined agent for user context and impact analysis.
    
    This agent consolidates:
    - User data retrieval (watchlist, portfolio, preferences)
    - Impact analysis on user's holdings
    - Causal chain building
    - Alert generation
    """

    input_schema = PortfolioInsightAgentInput
    output_schema = PortfolioInsightAgentOutput

    def __init__(self):
        settings = get_settings()
        config = AgentConfig(
            name="portfolio_insight_agent",
            description="Fetches user context and analyzes news impact on portfolio",
            model_name=settings.GEMINI_FAST_MODEL,
            temperature=0.2,  # Some creativity for causal reasoning
            max_output_tokens=8192,
            timeout_seconds=settings.IMPACT_ANALYSIS_AGENT_TIMEOUT + 10,
            retry_attempts=2,
        )
        super().__init__(config)

        # Register tool handlers
        for name, handler in get_analysis_tool_handlers().items():
            self.register_tool_handler(name, handler)

    def get_system_prompt(self) -> str:
        return PORTFOLIO_INSIGHT_SYSTEM_PROMPT

    def get_tools(self):
        return get_analysis_tools()

    async def execute(
        self,
        input_data: PortfolioInsightAgentInput,
        context: AgentExecutionContext,
    ) -> PortfolioInsightAgentOutput:
        """
        Execute portfolio insight analysis.
        
        Steps:
        1. Fetch user watchlist and portfolio
        2. Calculate sector exposure
        3. Analyze news impact on holdings
        4. Calculate portfolio-level impact
        5. Generate watchlist alerts
        6. Refine themes with user context
        """
        self.logger.info(
            "executing_portfolio_insight",
            user_id=input_data.user_id,
            news_count=len(input_data.news_items),
            request_id=context.request_id,
        )

        # =========================================
        # PHASE 1: Fetch User Context
        # =========================================
        
        watchlist: List[str] = []
        portfolio: List[PortfolioHolding] = []
        sector_exposure: Dict[str, float] = {}
        total_portfolio_value: float = 0.0

        # Fetch watchlist
        if input_data.include_watchlist:
            watchlist_data = await fetch_user_watchlist(input_data.user_id)
            watchlist = watchlist_data.get("watchlist", [])
            self.logger.info("fetched_watchlist", count=len(watchlist))

        # Fetch portfolio
        if input_data.include_portfolio:
            portfolio_data = await fetch_user_portfolio(input_data.user_id)
            raw_holdings = portfolio_data.get("holdings", [])
            total_portfolio_value = portfolio_data.get("total_current_value", 0.0)

            for holding in raw_holdings:
                portfolio.append(
                    PortfolioHolding(
                        ticker=holding["ticker"],
                        company_name=holding.get("company_name", holding["ticker"]),
                        quantity=holding["quantity"],
                        average_price=holding["average_price"],
                        current_price=holding["current_price"],
                        unrealized_pnl=holding.get("unrealized_pnl", 0.0),
                        unrealized_pnl_percent=holding.get("unrealized_pnl_percent", 0.0),
                        sector=holding.get("sector", "Unknown"),
                        weight_in_portfolio=holding.get("weight_in_portfolio", 0.0) / 100,
                    )
                )

            self.logger.info(
                "fetched_portfolio",
                holdings=len(portfolio),
                total_value=total_portfolio_value,
            )

            # Calculate sector exposure
            if portfolio:
                sector_exposure = await calculate_sector_exposure(raw_holdings)

        # Fetch user preferences
        prefs_data = await get_user_preferences(input_data.user_id)
        user_preferences = prefs_data.get("preferences", {})
        risk_profile = user_preferences.get("risk_profile")

        # =========================================
        # PHASE 2: Impact Analysis
        # =========================================
        
        # Create lookup sets
        watchlist_set = set(t.upper() for t in watchlist)
        portfolio_tickers = {h.ticker.upper() for h in portfolio}
        portfolio_map = {h.ticker.upper(): h for h in portfolio}

        # Analyze each news item
        news_with_impacts: List[NewsWithImpact] = []
        all_impacted_stocks: Dict[str, List[str]] = {}

        for news in input_data.news_items:
            impacted_stocks = await self._analyze_news_impact(
                news,
                watchlist_set,
                portfolio_tickers,
            )

            # Build causal chain
            causal_chain = self._build_causal_chain(news, impacted_stocks)

            # Determine sector impacts
            sector_impacts = self._determine_sector_impacts(impacted_stocks)

            news_impact = NewsWithImpact(
                news_id=news.id,
                news_item=news,
                impacted_stocks=impacted_stocks,
                sector_impacts=sector_impacts,
                causal_chain=causal_chain,
                impact_confidence=self._calculate_confidence(impacted_stocks),
            )
            news_with_impacts.append(news_impact)

            # Track impacted stocks
            for stock in impacted_stocks:
                if stock.ticker not in all_impacted_stocks:
                    all_impacted_stocks[stock.ticker] = []
                all_impacted_stocks[stock.ticker].append(news.id)

        # Calculate portfolio-level impact
        portfolio_impact = self._calculate_portfolio_impact(
            news_with_impacts,
            portfolio,
            sector_exposure,
        )

        # Generate watchlist alerts
        watchlist_alerts = self._generate_watchlist_alerts(
            all_impacted_stocks,
            watchlist_set,
            news_with_impacts,
        )

        # Refine themes
        refined_themes = self._refine_themes(
            input_data.preliminary_themes,
            news_with_impacts,
            watchlist_set,
            portfolio_tickers,
        )

        # Generate causal chains summary
        causal_chains = list(set(
            nwi.causal_chain for nwi in news_with_impacts
            if nwi.causal_chain and len(nwi.causal_chain) > 10
        ))[:5]

        self.logger.info(
            "portfolio_insight_complete",
            watchlist_count=len(watchlist),
            portfolio_count=len(portfolio),
            news_analyzed=len(news_with_impacts),
            alerts_generated=len(watchlist_alerts),
            themes_refined=len(refined_themes),
        )

        return PortfolioInsightAgentOutput(
            # User context outputs
            user_id=input_data.user_id,
            watchlist=watchlist,
            portfolio=portfolio,
            sector_exposure=sector_exposure,
            total_portfolio_value=total_portfolio_value,
            user_preferences=user_preferences,
            risk_profile=risk_profile,
            # Impact analysis outputs
            news_with_impacts=news_with_impacts,
            portfolio_level_impact=portfolio_impact,
            watchlist_alerts=watchlist_alerts,
            refined_themes=refined_themes,
            sector_impact_summary=self._summarize_sector_impacts(news_with_impacts),
            causal_chains=causal_chains,
        )

    async def _analyze_news_impact(
        self,
        news: NewsItem,
        watchlist: set,
        portfolio: set,
    ) -> List[ImpactedStock]:
        """Analyze a single news item for stock impacts."""
        impacted_stocks: List[ImpactedStock] = []

        # Direct impacts from mentioned stocks
        for ticker in news.mentioned_stocks:
            ticker_upper = ticker.upper()
            impact_type = "positive" if news.sentiment == "bullish" else (
                "negative" if news.sentiment == "bearish" else "neutral"
            )

            impacted_stocks.append(
                ImpactedStock(
                    ticker=ticker_upper,
                    company_name=ticker_upper,
                    impact_type=impact_type,
                    impact_magnitude="medium",
                    reasoning=f"Directly mentioned in news: {news.headline[:50]}...",
                    related_news_ids=[news.id],
                    in_portfolio=ticker_upper in portfolio,
                    in_watchlist=ticker_upper in watchlist,
                )
            )

        # Check for supply chain impacts
        supply_chain_events = ["oil", "crude", "steel", "rupee", "dollar", "rate"]
        headline_lower = news.headline.lower()

        for event in supply_chain_events:
            if event in headline_lower:
                if any(word in headline_lower for word in ["rise", "surge", "jump", "up"]):
                    event_key = f"{event}_up"
                elif any(word in headline_lower for word in ["fall", "drop", "decline", "down"]):
                    event_key = f"{event}_down"
                else:
                    continue

                try:
                    supply_impact = await analyze_supply_chain_impact(event_key)
                    if supply_impact.get("event_recognized"):
                        for chain in supply_impact.get("causal_chains", []):
                            ticker = chain["stock"]
                            if ticker not in [s.ticker for s in impacted_stocks]:
                                impacted_stocks.append(
                                    ImpactedStock(
                                        ticker=ticker,
                                        company_name=ticker,
                                        impact_type=chain["impact"],
                                        impact_magnitude="medium",
                                        reasoning=chain["chain"],
                                        related_news_ids=[news.id],
                                        in_portfolio=ticker in portfolio,
                                        in_watchlist=ticker in watchlist,
                                    )
                                )
                except Exception as e:
                    self.logger.warning(
                        "supply_chain_analysis_failed",
                        event=event_key,
                        error=str(e),
                    )

        return impacted_stocks

    def _build_causal_chain(
        self,
        news: NewsItem,
        impacted_stocks: List[ImpactedStock],
    ) -> str:
        """Build a causal chain explanation."""
        if not impacted_stocks:
            return "No direct stock impact identified"

        primary = impacted_stocks[0]

        if primary.reasoning and "→" in primary.reasoning:
            return primary.reasoning

        sentiment_desc = {
            "positive": "positive impact",
            "negative": "negative pressure",
            "neutral": "neutral effect",
        }

        return (
            f"{news.headline[:40]}... → "
            f"{sentiment_desc.get(primary.impact_type, 'impact')} → "
            f"{primary.ticker}"
        )

    def _determine_sector_impacts(
        self,
        impacted_stocks: List[ImpactedStock],
    ) -> Dict[str, str]:
        """Determine sector-level impacts from stock impacts."""
        sector_impacts: Dict[str, Dict[str, int]] = {}

        for stock in impacted_stocks:
            sector = "Unknown"
            if sector not in sector_impacts:
                sector_impacts[sector] = {"positive": 0, "negative": 0, "neutral": 0}
            sector_impacts[sector][stock.impact_type] += 1

        result = {}
        for sector, counts in sector_impacts.items():
            if counts["positive"] > counts["negative"]:
                result[sector] = "positive"
            elif counts["negative"] > counts["positive"]:
                result[sector] = "negative"
            else:
                result[sector] = "neutral"

        return result

    def _calculate_confidence(self, impacted_stocks: List[ImpactedStock]) -> float:
        """Calculate confidence score for impact analysis."""
        if not impacted_stocks:
            return 0.3

        direct_count = sum(
            1 for s in impacted_stocks
            if "Directly mentioned" in s.reasoning
        )

        base_confidence = 0.5
        direct_bonus = min(direct_count * 0.1, 0.3)

        return min(base_confidence + direct_bonus, 0.95)

    def _calculate_portfolio_impact(
        self,
        news_with_impacts: List[NewsWithImpact],
        portfolio: list,
        sector_exposure: Dict[str, float],
    ) -> PortfolioImpact:
        """Calculate aggregate portfolio impact."""
        if not portfolio:
            return PortfolioImpact(
                overall_sentiment="neutral",
                estimated_impact_percent=0.0,
                top_affected_holdings=[],
                positive_drivers=[],
                negative_drivers=[],
                reasoning="No portfolio holdings to analyze",
            )

        portfolio_tickers = {h.ticker.upper() for h in portfolio}
        portfolio_weights = {h.ticker.upper(): h.weight_in_portfolio for h in portfolio}

        positive_impact = 0.0
        negative_impact = 0.0
        affected_holdings: Dict[str, float] = {}
        positive_drivers: List[str] = []
        negative_drivers: List[str] = []

        for nwi in news_with_impacts:
            for stock in nwi.impacted_stocks:
                if stock.ticker in portfolio_tickers:
                    weight = portfolio_weights.get(stock.ticker, 0)
                    magnitude_factor = {"high": 1.5, "medium": 1.0, "low": 0.5}.get(
                        stock.impact_magnitude, 1.0
                    )

                    if stock.impact_type == "positive":
                        positive_impact += weight * magnitude_factor
                        positive_drivers.append(f"{stock.ticker} ({stock.reasoning[:30]}...)")
                    elif stock.impact_type == "negative":
                        negative_impact += weight * magnitude_factor
                        negative_drivers.append(f"{stock.ticker} ({stock.reasoning[:30]}...)")

                    affected_holdings[stock.ticker] = abs(weight * magnitude_factor)

        net_impact = positive_impact - negative_impact
        if net_impact > 0.02:
            overall_sentiment = "positive"
        elif net_impact < -0.02:
            overall_sentiment = "negative"
        elif positive_impact > 0 and negative_impact > 0:
            overall_sentiment = "mixed"
        else:
            overall_sentiment = "neutral"

        top_affected = sorted(
            affected_holdings.items(),
            key=lambda x: x[1],
            reverse=True,
        )[:3]

        return PortfolioImpact(
            overall_sentiment=overall_sentiment,
            estimated_impact_percent=round(net_impact * 100, 2),
            top_affected_holdings=[t[0] for t in top_affected],
            positive_drivers=positive_drivers[:3],
            negative_drivers=negative_drivers[:3],
            reasoning=self._generate_portfolio_reasoning(
                overall_sentiment,
                net_impact,
                positive_drivers,
                negative_drivers,
            ),
        )

    def _generate_portfolio_reasoning(
        self,
        sentiment: str,
        net_impact: float,
        positive_drivers: List[str],
        negative_drivers: List[str],
    ) -> str:
        """Generate human-readable portfolio impact reasoning."""
        if sentiment == "positive":
            drivers = ", ".join([d.split()[0] for d in positive_drivers[:2]])
            return f"Portfolio likely to benefit from current news flow. Key drivers: {drivers}."
        elif sentiment == "negative":
            drivers = ", ".join([d.split()[0] for d in negative_drivers[:2]])
            return f"Portfolio faces headwinds from current news. Pressure points: {drivers}."
        elif sentiment == "mixed":
            return "Mixed impact expected. Some holdings benefit while others face pressure."
        else:
            return "Current news has limited direct impact on portfolio holdings."

    def _generate_watchlist_alerts(
        self,
        all_impacted_stocks: Dict[str, List[str]],
        watchlist: set,
        news_with_impacts: List[NewsWithImpact],
    ) -> List[WatchlistAlert]:
        """Generate alerts for watchlist stocks."""
        alerts: List[WatchlistAlert] = []

        for ticker in watchlist:
            if ticker not in all_impacted_stocks:
                continue

            news_ids = all_impacted_stocks[ticker]
            impact_type = "neutral"
            reasoning_parts = []

            for nwi in news_with_impacts:
                for stock in nwi.impacted_stocks:
                    if stock.ticker == ticker:
                        if stock.impact_type == "positive":
                            impact_type = "opportunity"
                        elif stock.impact_type == "negative":
                            impact_type = "risk"
                        reasoning_parts.append(stock.reasoning)

            if reasoning_parts:
                alerts.append(
                    WatchlistAlert(
                        ticker=ticker,
                        company_name=ticker,
                        alert_type=impact_type,
                        alert_priority="medium",
                        related_news_ids=news_ids,
                        reasoning=reasoning_parts[0],
                        suggested_action=self._suggest_action(impact_type),
                    )
                )

        return alerts

    def _suggest_action(self, alert_type: str) -> str:
        """Suggest action based on alert type."""
        if alert_type == "opportunity":
            return "Consider adding to position on pullbacks"
        elif alert_type == "risk":
            return "Monitor closely, consider reviewing position size"
        return "No immediate action required"

    def _refine_themes(
        self,
        preliminary_themes: list,
        news_with_impacts: List[NewsWithImpact],
        watchlist: set,
        portfolio: set,
    ) -> List[ThemeGroup]:
        """Refine themes with impact analysis."""
        refined: List[ThemeGroup] = []

        for idx, theme in enumerate(preliminary_themes[:5]):
            theme_news = [
                nwi.news_item for nwi in news_with_impacts
                if nwi.news_id in theme.news_ids
            ]

            if not theme_news:
                continue

            impacted_stocks = set()
            impacted_sectors = set()
            for nwi in news_with_impacts:
                if nwi.news_id in theme.news_ids:
                    for stock in nwi.impacted_stocks:
                        impacted_stocks.add(stock.ticker)
                    impacted_sectors.update(nwi.sector_impacts.keys())

            user_stocks = watchlist | portfolio
            overlap = impacted_stocks & user_stocks
            relevance = min(len(overlap) / max(len(user_stocks), 1), 1.0)

            refined.append(
                ThemeGroup(
                    theme_id=f"theme_{idx:03d}",
                    theme_name=theme.theme_name,
                    theme_description=f"Theme covering {len(theme_news)} related news items",
                    news_items=theme_news,
                    overall_sentiment=theme.sentiment,
                    impacted_sectors=list(impacted_sectors),
                    impacted_stocks=list(impacted_stocks),
                    causal_summary=f"{theme.theme_name}: {theme.sentiment} sentiment",
                    relevance_to_user=round(relevance, 2),
                )
            )

        return refined

    def _summarize_sector_impacts(
        self,
        news_with_impacts: List[NewsWithImpact],
    ) -> Dict[str, str]:
        """Summarize impacts by sector."""
        sector_sentiment: Dict[str, Dict[str, int]] = {}

        for nwi in news_with_impacts:
            for sector, impact in nwi.sector_impacts.items():
                if sector not in sector_sentiment:
                    sector_sentiment[sector] = {"positive": 0, "negative": 0, "neutral": 0}
                sector_sentiment[sector][impact] += 1

        summaries = {}
        for sector, counts in sector_sentiment.items():
            if counts["positive"] > counts["negative"]:
                summaries[sector] = "Positive momentum"
            elif counts["negative"] > counts["positive"]:
                summaries[sector] = "Facing headwinds"
            else:
                summaries[sector] = "Mixed/Neutral"

        return summaries
