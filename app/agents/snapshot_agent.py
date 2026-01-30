"""
Snapshot Generation Agent - Background AI-powered market snapshot creation.

This agent is invoked by Celery tasks for:
- Market outlook generation (sentiment, confidence, reasoning)
- Summary bullets with causal language
- Executive summary generation
- Trending news selection

Part of the 3-agent background processing architecture:
1. NewsProcessingAgent - AI news analysis (called by fetch_news task)
2. SnapshotGenerationAgent - AI snapshot generation (called by gen_snapshot task)
3. IndicesCollectionAgent - Indices data collection (called by fetch_indices task)
"""

from datetime import datetime
from typing import Any, Dict, List, Optional

from app.config import get_settings
from app.db.models.news_document import NewsArticleDocument
from app.services.snapshot_generator_service import SnapshotGeneratorService
from app.utils.logging import get_logger

logger = get_logger(__name__)
settings = get_settings()


class SnapshotGenerationAgent:
    """
    Background agent for AI-powered market snapshot generation.
    
    Responsibilities:
    - Generate market outlook (pre/post market)
    - Create summary bullets with mandatory causal language
    - Generate executive summary
    - Select trending news (mid-market)
    
    This agent wraps SnapshotGeneratorService and provides a clean
    interface for Celery tasks.
    """
    
    def __init__(self):
        self.generator = SnapshotGeneratorService()
        self.logger = get_logger("snapshot_generation_agent")
    
    async def generate_snapshot_content(
        self,
        market_phase: str,
        indices_data: Dict[str, Any],
        news_items: List[NewsArticleDocument],
    ) -> Dict[str, Any]:
        """
        Generate complete snapshot content using AI.
        
        Args:
            market_phase: Current market phase (pre/mid/post)
            indices_data: Current indices data from API
            news_items: Recent analyzed news articles
            
        Returns:
            Dict with snapshot content:
            - market_outlook: Sentiment, confidence, reasoning (pre/post only)
            - market_summary: List of causal summary bullets
            - executive_summary: Brief overview
            - trending_now: News IDs (mid-market only)
        """
        self.logger.info(
            "generating_snapshot_content",
            market_phase=market_phase,
            indices_count=len(indices_data),
            news_count=len(news_items),
        )
        
        try:
            result = await self.generator.generate_snapshot(
                market_phase=market_phase,
                indices_data=indices_data,
                news_items=news_items,
            )
            
            self.logger.info(
                "snapshot_content_generated",
                market_phase=market_phase,
                has_outlook=result.get("market_outlook") is not None,
                bullets_count=len(result.get("market_summary", [])),
            )
            
            return result
            
        except Exception as e:
            self.logger.error(
                "snapshot_generation_failed",
                market_phase=market_phase,
                error=str(e),
            )
            # Return fallback content
            return self._create_fallback_content(market_phase, indices_data, news_items)
    
    async def generate_executive_summary(
        self,
        market_phase: str,
        indices_data: Dict[str, Any],
        news_items: List[NewsArticleDocument],
    ) -> str:
        """
        Generate just the executive summary.
        
        Args:
            market_phase: Current phase
            indices_data: Indices data
            news_items: News items
            
        Returns:
            Executive summary string
        """
        return await self.generator.generate_executive_summary(
            market_phase=market_phase,
            indices_data=indices_data,
            news_items=news_items,
        )
    
    def _create_fallback_content(
        self,
        market_phase: str,
        indices_data: Dict[str, Any],
        news_items: List[NewsArticleDocument],
    ) -> Dict[str, Any]:
        """Create rule-based fallback content when AI fails."""
        content = {
            "market_summary": [],
            "executive_summary": "Market data available. AI analysis temporarily unavailable.",
        }
        
        # Calculate basic outlook from NIFTY
        nifty = indices_data.get("NIFTY", indices_data.get("SENSEX", {}))
        if nifty and isinstance(nifty, dict):
            change = nifty.get("change_percent", 0)
            if change > 0.5:
                sentiment = "bullish"
            elif change < -0.5:
                sentiment = "bearish"
            else:
                sentiment = "neutral"
            
            content["market_outlook"] = {
                "sentiment": sentiment,
                "confidence": 0.6,
                "reasoning": f"Based on index movement of {change:.2f}%",
                "nifty_change_percent": change,
                "key_drivers": ["Index movement"],
            }
        
        if market_phase == "mid":
            content["trending_now"] = [n.news_id for n in news_items[:5]]
        
        return content


# Singleton instance
_snapshot_generation_agent: Optional[SnapshotGenerationAgent] = None


def get_snapshot_generation_agent() -> SnapshotGenerationAgent:
    """Get singleton SnapshotGenerationAgent instance."""
    global _snapshot_generation_agent
    if _snapshot_generation_agent is None:
        _snapshot_generation_agent = SnapshotGenerationAgent()
    return _snapshot_generation_agent
