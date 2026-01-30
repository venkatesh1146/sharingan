"""
Snapshot Generator Service - AI-powered market snapshot generation.

Provides:
- Market outlook generation
- Summary bullet creation with causal language
- Executive summary generation
- Trending news selection
"""

from typing import Any, Dict, List, Optional
import json

from app.config import get_settings
from app.db.models.news_document import NewsArticleDocument
from app.utils.logging import get_logger
from app.utils.vertex_ai_client import VertexAIClient

logger = get_logger(__name__)
settings = get_settings()


# Causal language keywords that MUST be present in summaries
CAUSAL_KEYWORDS = [
    "due to", "after", "following", "driven by", "as", "because",
    "on account of", "amid", "on the back of", "triggered by",
    "led by", "supported by", "weighed by",
]


class SnapshotGeneratorService:
    """
    Service for generating market snapshots with AI.
    
    Creates:
    - Market outlook (pre/post market)
    - Summary bullets with causal language
    - Executive summaries
    - Trending news selection (mid market)
    """
    
    def __init__(self):
        self.client = VertexAIClient(
            model_name=settings.GEMINI_FAST_MODEL,
            temperature=0.3,  # Slightly creative for narrative
            max_output_tokens=4096,
        )
        self.logger = get_logger("snapshot_generator")

    async def generate_snapshot(
        self,
        market_phase: str,
        indices_data: Dict[str, Any],
        news_items: List[NewsArticleDocument],
    ) -> Dict[str, Any]:
        """
        Generate complete snapshot content.
        
        Args:
            market_phase: Current market phase (pre/mid/post)
            indices_data: Current indices data
            news_items: Recent news articles
            
        Returns:
            Dict with snapshot content
        """
        result = {
            "market_outlook": None,
            "market_summary": [],
            "executive_summary": None,
            "trending_now": None,
        }
        
        # Get NIFTY/SENSEX data for context
        nifty = indices_data.get("NIFTY", indices_data.get("SENSEX", {}))
        nifty_change = nifty.get("change_percent", 0) if nifty else 0
        
        if market_phase == "mid":
            # Mid-market: Generate trending news
            result["trending_now"] = self._select_trending(news_items)
            result["executive_summary"] = "Market activity ongoing. Key developments being monitored."
        else:
            # Pre/Post market: Generate full analysis
            try:
                ai_result = await self._generate_ai_snapshot(
                    market_phase, indices_data, news_items
                )
                if ai_result:
                    result.update(ai_result)
            except Exception as e:
                self.logger.warning("ai_snapshot_failed", error=str(e))
                # Fall back to rule-based
                result.update(self._generate_rule_based_snapshot(
                    market_phase, indices_data, news_items
                ))
        
        return result

    async def _generate_ai_snapshot(
        self,
        market_phase: str,
        indices_data: Dict[str, Any],
        news_items: List[NewsArticleDocument],
    ) -> Optional[Dict[str, Any]]:
        """Generate snapshot using AI."""
        prompt = self._build_snapshot_prompt(market_phase, indices_data, news_items)
        
        try:
            response = await self.client.generate_content(prompt)
            if not response:
                return None
            
            return self._parse_snapshot_response(response, indices_data)
            
        except Exception as e:
            self.logger.warning("ai_snapshot_error", error=str(e))
            return None

    def _build_snapshot_prompt(
        self,
        market_phase: str,
        indices_data: Dict[str, Any],
        news_items: List[NewsArticleDocument],
    ) -> str:
        """Build prompt for snapshot generation."""
        # Format indices data
        indices_text = ""
        for ticker, data in list(indices_data.items())[:5]:
            if isinstance(data, dict) and "error" not in data:
                change = data.get("change_percent", 0)
                price = data.get("current_price", 0)
                indices_text += f"- {ticker}: {price:.2f} ({change:+.2f}%)\n"
        
        # Format news headlines
        news_text = ""
        for item in news_items[:10]:
            sentiment = item.sentiment if hasattr(item, 'sentiment') else 'neutral'
            news_text += f"- [{sentiment}] {item.headline}\n"
        
        phase_context = {
            "pre": "Markets are about to open. Focus on overnight developments and opening expectations.",
            "post": "Markets have closed. Summarize the day's key movements and drivers.",
        }
        
        return f"""Generate a market snapshot for {market_phase}-market phase.

{phase_context.get(market_phase, '')}

Current Indices:
{indices_text}

Recent News:
{news_text}

Return a JSON object with:
1. "market_outlook": {{
     "sentiment": "bullish/bearish/neutral",
     "confidence": 0.0-1.0,
     "reasoning": "2-3 sentence explanation",
     "nifty_change_percent": number,
     "key_drivers": ["driver1", "driver2"]
   }}

2. "market_summary": Array of 3 summary bullets, each with:
   - "text": Summary with MANDATORY causal language (use: "due to", "driven by", "following", "amid", "on the back of")
   - "supporting_news_ids": [list of relevant news IDs if known]
   - "confidence": 0.0-1.0
   - "sentiment": "bullish/bearish/neutral"
   
   IMPORTANT: Each summary MUST contain causal language explaining WHY something happened.
   Bad: "Markets closed higher"
   Good: "Markets closed higher driven by positive global cues following US market rally"

3. "executive_summary": A 2-3 sentence overview of the market

Return ONLY valid JSON, no other text."""

    def _parse_snapshot_response(
        self,
        response: str,
        indices_data: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        """Parse AI response for snapshot."""
        try:
            # Clean response
            text = response.strip()
            if text.startswith("```json"):
                text = text[7:]
            if text.startswith("```"):
                text = text[3:]
            if text.endswith("```"):
                text = text[:-3]
            
            data = json.loads(text.strip())
            
            result = {}
            
            # Parse market outlook
            if data.get("market_outlook"):
                outlook = data["market_outlook"]
                # Get actual NIFTY change
                nifty = indices_data.get("NIFTY", indices_data.get("SENSEX", {}))
                nifty_change = nifty.get("change_percent", 0) if nifty else 0
                
                result["market_outlook"] = {
                    "sentiment": outlook.get("sentiment", "neutral"),
                    "confidence": float(outlook.get("confidence", 0.5)),
                    "reasoning": outlook.get("reasoning", ""),
                    "nifty_change_percent": nifty_change,
                    "key_drivers": outlook.get("key_drivers", []),
                }
            
            # Parse market summary
            if data.get("market_summary"):
                result["market_summary"] = []
                for bullet in data["market_summary"]:
                    text = bullet.get("text", "")
                    # Validate causal language
                    if self._has_causal_language(text):
                        result["market_summary"].append({
                            "text": text,
                            "supporting_news_ids": bullet.get("supporting_news_ids", []),
                            "confidence": float(bullet.get("confidence", 0.7)),
                            "sentiment": bullet.get("sentiment", "neutral"),
                        })
            
            # Parse executive summary
            if data.get("executive_summary"):
                result["executive_summary"] = data["executive_summary"]
            
            return result
            
        except (json.JSONDecodeError, ValueError, TypeError) as e:
            self.logger.warning("snapshot_parse_error", error=str(e))
            return None

    def _generate_rule_based_snapshot(
        self,
        market_phase: str,
        indices_data: Dict[str, Any],
        news_items: List[NewsArticleDocument],
    ) -> Dict[str, Any]:
        """Generate snapshot using rules when AI fails."""
        result = {}
        
        # Get NIFTY data
        nifty = indices_data.get("NIFTY", indices_data.get("SENSEX", {}))
        nifty_change = nifty.get("change_percent", 0) if nifty else 0
        
        # Determine sentiment
        if nifty_change > 0.5:
            sentiment = "bullish"
            direction = "higher"
        elif nifty_change < -0.5:
            sentiment = "bearish"
            direction = "lower"
        else:
            sentiment = "neutral"
            direction = "flat"
        
        # Market outlook
        result["market_outlook"] = {
            "sentiment": sentiment,
            "confidence": 0.6,
            "reasoning": f"Markets trading {direction} with NIFTY at {nifty_change:.2f}%.",
            "nifty_change_percent": nifty_change,
            "key_drivers": ["Index movement"],
        }
        
        # Generate basic summary bullets
        result["market_summary"] = []
        
        # Bullet 1: Index movement
        bullet1 = f"Markets trading {direction} driven by "
        if sentiment == "bullish":
            bullet1 += "positive sentiment amid favorable market conditions."
        elif sentiment == "bearish":
            bullet1 += "selling pressure following cautious market sentiment."
        else:
            bullet1 += "mixed cues amid consolidation phase."
        
        result["market_summary"].append({
            "text": bullet1,
            "supporting_news_ids": [],
            "confidence": 0.6,
            "sentiment": sentiment,
        })
        
        # Bullet 2: From news if available
        if news_items:
            top_news = news_items[0]
            headline = top_news.headline[:60]
            news_sentiment = top_news.sentiment if hasattr(top_news, 'sentiment') else 'neutral'
            
            if news_sentiment == "bullish":
                bullet2 = f"Positive sentiment supported by {headline.lower()}."
            elif news_sentiment == "bearish":
                bullet2 = f"Caution amid {headline.lower()}."
            else:
                bullet2 = f"Markets focused on {headline.lower()}."
            
            result["market_summary"].append({
                "text": bullet2,
                "supporting_news_ids": [top_news.news_id],
                "confidence": 0.5,
                "sentiment": news_sentiment,
            })
        
        # Executive summary
        result["executive_summary"] = (
            f"Markets trading {direction} with NIFTY at {nifty_change:.1f}%. "
            "Key developments being monitored."
        )
        
        return result

    def _select_trending(
        self,
        news_items: List[NewsArticleDocument],
    ) -> List[str]:
        """Select trending news IDs for mid-market."""
        # Sort by published_at (most recent first)
        sorted_news = sorted(
            news_items,
            key=lambda x: x.published_at,
            reverse=True,
        )
        
        return [n.news_id for n in sorted_news[:5]]

    def _has_causal_language(self, text: str) -> bool:
        """Check if text contains causal language."""
        text_lower = text.lower()
        return any(keyword in text_lower for keyword in CAUSAL_KEYWORDS)

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
        nifty = indices_data.get("NIFTY", indices_data.get("SENSEX", {}))
        nifty_change = nifty.get("change_percent", 0) if nifty else 0
        
        prompt = f"""Generate a brief 2-3 sentence executive summary for {market_phase}-market.

NIFTY change: {nifty_change:.2f}%
Top news: {news_items[0].headline if news_items else 'No recent news'}

Return only the summary text, no JSON or formatting."""
        
        try:
            response = await self.client.generate_content(prompt)
            if response:
                return response.strip()[:300]
        except Exception as e:
            self.logger.warning("executive_summary_error", error=str(e))
        
        # Fallback
        if nifty_change > 0:
            return f"Markets trading higher with NIFTY up {nifty_change:.1f}%."
        elif nifty_change < 0:
            return f"Markets under pressure with NIFTY down {abs(nifty_change):.1f}%."
        else:
            return "Markets trading flat. Key developments being monitored."


# Singleton instance
_snapshot_generator: Optional[SnapshotGeneratorService] = None


def get_snapshot_generator_service() -> SnapshotGeneratorService:
    """Get singleton SnapshotGeneratorService instance."""
    global _snapshot_generator
    if _snapshot_generator is None:
        _snapshot_generator = SnapshotGeneratorService()
    return _snapshot_generator
