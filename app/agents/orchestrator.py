"""
Orchestrator Agent for coordinating the simplified 3-agent system.

Simplified Architecture:
- Phase 1: Market Intelligence (market data + news)
- Phase 2: Portfolio Insight (user context + impact analysis)  
- Phase 3: Summary Generation
"""

import asyncio
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

from app.agents.base import AgentExecutionContext
from app.agents.market_intelligence_agent import MarketIntelligenceAgent
from app.agents.portfolio_insight_agent import PortfolioInsightAgent
from app.agents.summary_generation_agent import SummaryGenerationAgent
from app.config import get_settings
from app.constants.themes import MAX_THEMED_NEWS_ITEMS, normalize_theme_to_allowed
from app.models.requests import MarketPulseRequest
from app.models.responses import (
    MarketPulseResponse,
    ThemedNewsItem,
    _news_item_to_response,
    _news_with_impact_to_response,
)
from app.models.agent_schemas import (
    MarketIntelligenceAgentInput,
    PortfolioInsightAgentInput,
    SummaryGenerationAgentInput,
    AgentExecutionResult,
    OrchestratorDecision,
    OrchestrationMetrics,
)
from app.models.domain import PortfolioImpact
from app.utils.logging import get_logger
from app.utils.tracing import trace_orchestration
from app.utils.exceptions import OrchestrationError


logger = get_logger(__name__)


def _build_themed_news_list(
    refined_themes: list,
    market_phase: str,
) -> List[ThemedNewsItem]:
    """
    Build themed_news for API: only pre/post market, max 5 items, allowed themes only.

    Themed news represents themes impacted in post-market and will be impacted in pre-market.
    """
    if market_phase not in ("pre", "post"):
        return []
    items: List[ThemedNewsItem] = []
    for theme in refined_themes[:MAX_THEMED_NEWS_ITEMS]:
        allowed = normalize_theme_to_allowed(theme.theme_name)
        if not allowed:
            continue
        reason = (theme.causal_summary or "").strip() or f"{allowed}: {theme.overall_sentiment} sentiment"
        items.append(
            ThemedNewsItem(
                theme_name=allowed,
                sentiment=theme.overall_sentiment,
                theme=allowed,
                reason=reason,
            )
        )
        if len(items) >= MAX_THEMED_NEWS_ITEMS:
            break
    return items


class OrchestratorAgent:
    """
    Simplified orchestrator coordinating 3 specialized agents.
    
    Execution phases:
    1. Phase 1: Market Intelligence (market data + news analysis)
    2. Phase 2: Portfolio Insight (user context + impact analysis)
    3. Phase 3: Summary Generation
    4. Assembly: Create final MarketPulseResponse
    """

    def __init__(self):
        self.market_intelligence_agent = MarketIntelligenceAgent()
        self.portfolio_insight_agent = PortfolioInsightAgent()
        self.summary_generation_agent = SummaryGenerationAgent()
        self.logger = get_logger("orchestrator")
        self.settings = get_settings()

    async def orchestrate(
        self,
        request: MarketPulseRequest,
        context: AgentExecutionContext,
    ) -> MarketPulseResponse:
        """
        Main orchestration logic with simplified 3-phase flow.
        
        Args:
            request: Market Pulse request
            context: Execution context
        
        Returns:
            Complete MarketPulseResponse
        """
        start_time = time.time()
        agent_times: Dict[str, int] = {}
        agents_succeeded: List[str] = []
        agents_failed: List[str] = []
        warnings: List[str] = []
        degraded_mode = False

        self.logger.info(
            "orchestration_started",
            request_id=context.request_id,
            user_id=request.user_id,
        )

        with trace_orchestration(context.request_id, context.user_id):
            try:
                # ===== Phase 1: Market Intelligence =====
                self.logger.info("phase1_start", phase="market_intelligence")
                phase1_start = time.time()

                phase1_result = await self._execute_phase1(request, context)
                agent_times["market_intelligence"] = phase1_result.execution_time_ms

                # Evaluate Phase 1 result
                if phase1_result.status == "success":
                    agents_succeeded.append("market_intelligence_agent")
                else:
                    agents_failed.append("market_intelligence_agent")
                    # Market intelligence is critical - cannot proceed without it
                    self.logger.warning(
                        "market_intelligence_failed",
                        error=phase1_result.error,
                    )
                    return self._generate_fallback_response(
                        OrchestratorDecision(
                            can_proceed=False,
                            degraded_mode=True,
                            reasoning="Market intelligence failed, cannot proceed",
                            skip_agents=[],
                        ),
                        context,
                        warnings + ["Market data unavailable"],
                    )

                self.logger.info(
                    "phase1_complete",
                    duration_ms=int((time.time() - phase1_start) * 1000),
                )

                # ===== Phase 2: Portfolio Insight =====
                self.logger.info("phase2_start", phase="portfolio_insight")
                phase2_start = time.time()

                phase2_result = await self._execute_phase2(
                    request, phase1_result, context
                )
                agent_times["portfolio_insight"] = phase2_result.execution_time_ms

                if phase2_result.status == "success":
                    agents_succeeded.append("portfolio_insight_agent")
                else:
                    agents_failed.append("portfolio_insight_agent")
                    warnings.append("Portfolio insight unavailable, using defaults")
                    degraded_mode = True

                self.logger.info(
                    "phase2_complete",
                    duration_ms=int((time.time() - phase2_start) * 1000),
                    status=phase2_result.status,
                )

                # ===== Phase 3: Summary Generation =====
                self.logger.info("phase3_start", phase="summary_generation")
                phase3_start = time.time()

                phase3_result = await self._execute_phase3(
                    request, phase1_result, phase2_result, context
                )
                agent_times["summary_generation"] = phase3_result.execution_time_ms

                if phase3_result.status == "success":
                    agents_succeeded.append("summary_generation_agent")
                else:
                    agents_failed.append("summary_generation_agent")
                    warnings.append("Summary generation failed, using basic summary")
                    degraded_mode = True

                self.logger.info(
                    "phase3_complete",
                    duration_ms=int((time.time() - phase3_start) * 1000),
                    status=phase3_result.status,
                )

                # ===== Assembly =====
                self.logger.info("assembly_start", phase="assembly")

                total_time = int((time.time() - start_time) * 1000)

                metrics = OrchestrationMetrics(
                    total_execution_time_ms=total_time,
                    agent_execution_times=agent_times,
                    agents_succeeded=agents_succeeded,
                    agents_failed=agents_failed,
                    degraded_mode=degraded_mode,
                    cache_hits=0,
                )

                response = self._assemble_response(
                    phase1_result,
                    phase2_result,
                    phase3_result,
                    context,
                    metrics,
                    warnings,
                    degraded_mode,
                )

                self.logger.info(
                    "orchestration_completed",
                    request_id=context.request_id,
                    total_time_ms=total_time,
                    degraded_mode=degraded_mode,
                    agents_succeeded=len(agents_succeeded),
                    agents_failed=len(agents_failed),
                )

                return response

            except Exception as e:
                self.logger.error(
                    "orchestration_failed",
                    error=str(e),
                    request_id=context.request_id,
                )
                raise OrchestrationError(
                    message=f"Orchestration failed: {str(e)}",
                    failed_agents=agents_failed,
                    phase="orchestration",
                )

    async def _execute_phase1(
        self,
        request: MarketPulseRequest,
        context: AgentExecutionContext,
    ) -> AgentExecutionResult:
        """
        Phase 1: Market Intelligence (market data + news analysis).
        """
        market_intelligence_input = MarketIntelligenceAgentInput(
            selected_indices=None,  # Phase-based indices (pre/mid/post market)
            timestamp=datetime.utcnow(),
            force_refresh=request.force_refresh,
            time_window_hours=24,
            max_articles=request.max_news_items,
            watchlist=None,  # Will be populated if needed
        )

        try:
            return await self.market_intelligence_agent.execute_with_retry(
                market_intelligence_input, context
            )
        except Exception as e:
            return AgentExecutionResult(
                agent_name="market_intelligence_agent",
                status="failed",
                error=str(e),
                execution_time_ms=0,
            )

    async def _execute_phase2(
        self,
        request: MarketPulseRequest,
        phase1_result: AgentExecutionResult,
        context: AgentExecutionContext,
    ) -> AgentExecutionResult:
        """
        Phase 2: Portfolio Insight (user context + impact analysis).
        """
        market_intelligence_output = phase1_result.output

        # Prepare input with data from Phase 1
        portfolio_insight_input = PortfolioInsightAgentInput(
            user_id=request.user_id,
            include_watchlist=False,
            include_portfolio=False,
            news_filter=request.news_filter,
            news_items=market_intelligence_output.news_items if market_intelligence_output else [],
            indices_data=market_intelligence_output.indices_data if market_intelligence_output else {},
            preliminary_themes=market_intelligence_output.preliminary_themes if market_intelligence_output else [],
        )

        try:
            return await self.portfolio_insight_agent.execute_with_retry(
                portfolio_insight_input, context
            )
        except Exception as e:
            return AgentExecutionResult(
                agent_name="portfolio_insight_agent",
                status="failed",
                error=str(e),
                execution_time_ms=0,
            )

    async def _execute_phase3(
        self,
        request: MarketPulseRequest,
        phase1_result: AgentExecutionResult,
        phase2_result: AgentExecutionResult,
        context: AgentExecutionContext,
    ) -> AgentExecutionResult:
        """
        Phase 3: Summary Generation.
        """
        market_intelligence_output = phase1_result.output
        portfolio_insight_output = phase2_result.output

        # Handle missing data
        market_outlook = market_intelligence_output.market_outlook if market_intelligence_output else None
        market_phase = market_intelligence_output.market_phase if market_intelligence_output else "mid"
        news_with_impacts = (
            portfolio_insight_output.news_with_impacts if portfolio_insight_output else []
        )
        indices_data = market_intelligence_output.indices_data if market_intelligence_output else {}
        portfolio_impact = (
            portfolio_insight_output.portfolio_level_impact if portfolio_insight_output else None
        )
        refined_themes = (
            portfolio_insight_output.refined_themes if portfolio_insight_output else []
        )

        # Create default portfolio impact if missing
        if not portfolio_impact:
            portfolio_impact = PortfolioImpact(
                overall_sentiment="neutral",
                estimated_impact_percent=0.0,
                top_affected_holdings=[],
                positive_drivers=[],
                negative_drivers=[],
                reasoning="Impact analysis unavailable",
            )

        summary_input = SummaryGenerationAgentInput(
            market_outlook=market_outlook,
            market_phase=market_phase,
            news_with_impacts=news_with_impacts,
            indices_data=indices_data,
            portfolio_impact=portfolio_impact,
            refined_themes=refined_themes,
            max_bullets=3,
        )

        try:
            return await self.summary_generation_agent.execute_with_retry(
                summary_input, context
            )
        except Exception as e:
            return AgentExecutionResult(
                agent_name="summary_generation_agent",
                status="failed",
                error=str(e),
                execution_time_ms=0,
            )

    def _assemble_response(
        self,
        phase1_result: AgentExecutionResult,
        phase2_result: AgentExecutionResult,
        phase3_result: AgentExecutionResult,
        context: AgentExecutionContext,
        metrics: OrchestrationMetrics,
        warnings: List[str],
        degraded_mode: bool,
    ) -> MarketPulseResponse:
        """
        Assemble final MarketPulseResponse from all agent outputs.
        """
        market_intelligence_output = phase1_result.output
        portfolio_insight_output = phase2_result.output
        summary_output = phase3_result.output

        # Build response based on market phase
        market_phase = market_intelligence_output.market_phase if market_intelligence_output else "mid"
        is_mid_market = market_phase == "mid"

        # Log metrics (not sent in API response)
        self._log_metrics(context, metrics)

        return MarketPulseResponse(
            generated_at=datetime.utcnow(),
            request_id=context.request_id,
            # Market Status
            market_phase=market_phase,
            market_outlook=(
                market_intelligence_output.market_outlook
                if market_intelligence_output and not is_mid_market
                else None
            ),
            indices_data=(
                market_intelligence_output.indices_data if market_intelligence_output else {}
            ),
            # Summary Section
            market_summary=(
                summary_output.market_summary_bullets
                if summary_output and not is_mid_market
                else None
            ),
            executive_summary=(
                summary_output.executive_summary if summary_output else None
            ),
            # Mid-Market Section (single-layer; exclude url, sentiment_score, relevance_score)
            trending_now=(
                [_news_item_to_response(n) for n in (summary_output.trending_now_section or [])]
                if summary_output and is_mid_market
                else None
            ),
            # News & Themes (pre/post only; max 5 allowed themes)
            themed_news=(
                _build_themed_news_list(
                    portfolio_insight_output.refined_themes if portfolio_insight_output else [],
                    market_phase,
                )
            ),
            # Single-layer; exclude url, sentiment_score, relevance_score, impact_confidence
            all_news=(
                [_news_with_impact_to_response(nwi) for nwi in portfolio_insight_output.news_with_impacts]
                if portfolio_insight_output
                else []
            ),
            # User-Specific
            watchlist_impacted=(
                [a.ticker for a in portfolio_insight_output.watchlist_alerts]
                if portfolio_insight_output
                else []
            ),
            watchlist_alerts=(
                portfolio_insight_output.watchlist_alerts
                if portfolio_insight_output
                else []
            ),
            portfolio_impact_summary=(
                portfolio_insight_output.portfolio_level_impact.reasoning
                if portfolio_insight_output and portfolio_insight_output.portfolio_level_impact
                else None
            ),
            portfolio_sentiment=(
                portfolio_insight_output.portfolio_level_impact.overall_sentiment
                if portfolio_insight_output and portfolio_insight_output.portfolio_level_impact
                else None
            ),
            # Metadata (metrics logged below, not in response)
            degraded_mode=degraded_mode,
            warnings=warnings,
        )

    def _log_metrics(self, context: AgentExecutionContext, metrics: OrchestrationMetrics) -> None:
        """Log orchestration metrics for monitoring; not sent in API response."""
        logger.info(
            "orchestration_metrics",
            request_id=context.request_id,
            total_execution_time_ms=metrics.total_execution_time_ms,
            agent_execution_times=metrics.agent_execution_times,
            agents_succeeded=metrics.agents_succeeded,
            agents_failed=metrics.agents_failed,
            degraded_mode=metrics.degraded_mode,
            cache_hits=metrics.cache_hits,
        )

    def _generate_fallback_response(
        self,
        decision: OrchestratorDecision,
        context: AgentExecutionContext,
        warnings: List[str],
    ) -> MarketPulseResponse:
        """Generate degraded response when critical agents fail."""
        self.logger.warning(
            "generating_fallback_response",
            strategy=decision.fallback_strategy,
            reasoning=decision.reasoning,
        )

        return MarketPulseResponse(
            generated_at=datetime.utcnow(),
            request_id=context.request_id,
            market_phase="mid",  # Safe default
            all_news=[],
            themed_news=[],
            portfolio_impact_summary="Market data temporarily unavailable",
            degraded_mode=True,
            warnings=warnings + [decision.reasoning],
        )
