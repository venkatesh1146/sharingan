"""
Snapshot Generator Service - AI-powered market snapshot generation.

Provides:
- Market outlook generation
- Summary bullet creation with causal language
- Executive summary generation
- Trending news selection
"""

from datetime import datetime
from typing import Any, Dict, List, Literal, Optional
import json

from pydantic import BaseModel, Field, field_validator

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


# ============================================================================
# Structured Output Models for AI Response
# ============================================================================

class MarketOutlookResponse(BaseModel):
    """Structured response for market outlook from AI."""
    sentiment: Literal["bullish", "bearish", "neutral"] = Field(
        default="neutral",
        description="Overall market sentiment"
    )
    confidence: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Confidence score between 0 and 1"
    )
    reasoning: str = Field(
        default="",
        description="2-3 sentence explanation of the outlook"
    )
    key_drivers: List[str] = Field(
        default_factory=list,
        description="List of key market drivers"
    )
    
    @field_validator("confidence", mode="before")
    @classmethod
    def clamp_confidence(cls, v):
        """Ensure confidence is within valid range."""
        try:
            val = float(v)
            return max(0.0, min(1.0, val))
        except (TypeError, ValueError):
            return 0.5
    
    @field_validator("sentiment", mode="before")
    @classmethod
    def normalize_sentiment(cls, v):
        """Normalize sentiment to valid values."""
        if isinstance(v, str):
            v_lower = v.lower().strip()
            if v_lower in ("bullish", "positive", "up"):
                return "bullish"
            elif v_lower in ("bearish", "negative", "down"):
                return "bearish"
        return "neutral"


class MarketSummaryBulletResponse(BaseModel):
    """Structured response for a single summary bullet."""
    text: str = Field(
        default="",
        description="Summary text with causal language"
    )
    supporting_news_ids: List[str] = Field(
        default_factory=list,
        description="IDs of supporting news articles"
    )
    confidence: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Confidence score"
    )
    sentiment: Literal["bullish", "bearish", "neutral"] = Field(
        default="neutral",
        description="Sentiment of this bullet point"
    )
    
    @field_validator("confidence", mode="before")
    @classmethod
    def clamp_confidence(cls, v):
        """Ensure confidence is within valid range."""
        try:
            val = float(v)
            return max(0.0, min(1.0, val))
        except (TypeError, ValueError):
            return 0.7
    
    @field_validator("sentiment", mode="before")
    @classmethod
    def normalize_sentiment(cls, v):
        """Normalize sentiment to valid values."""
        if isinstance(v, str):
            v_lower = v.lower().strip()
            if v_lower in ("bullish", "positive", "up"):
                return "bullish"
            elif v_lower in ("bearish", "negative", "down"):
                return "bearish"
        return "neutral"


class ThemedItemResponse(BaseModel):
    """Impacted theme/sector item (matches ThemedItemDocument shape)."""
    sector: str = Field(default="", description="Theme or sector name")
    relevant_companies: List[str] = Field(default_factory=list)
    sentiment: Literal["bullish", "bearish", "neutral"] = "neutral"
    sentiment_score: Optional[float] = Field(default=None, description="Optional 0.0-1.0")

    @field_validator("sentiment", mode="before")
    @classmethod
    def normalize_sentiment(cls, v):
        if isinstance(v, str):
            v_lower = v.lower().strip()
            if v_lower in ("bullish", "positive", "up"):
                return "bullish"
            elif v_lower in ("bearish", "negative", "down"):
                return "bearish"
        return "neutral"


class SnapshotAIResponse(BaseModel):
    """Complete structured response from AI for snapshot generation."""
    market_outlook: Optional[MarketOutlookResponse] = Field(
        default=None,
        description="Market outlook analysis"
    )
    market_summary: List[MarketSummaryBulletResponse] = Field(
        default_factory=list,
        description="List of summary bullets"
    )
    executive_summary: str = Field(
        default="",
        description="2-3 sentence executive summary"
    )
    themed: List[ThemedItemResponse] = Field(
        default_factory=list,
        description="Impacted themes with sector, companies, sentiment",
    )

    @classmethod
    def from_raw_response(
        cls,
        data: Dict[str, Any],
        nifty_change: float = 0.0,
    ) -> "SnapshotAIResponse":
        """
        Parse raw AI response dict into structured model.
        
        Handles missing fields, type mismatches, and invalid values gracefully.
        """
        try:
            # Parse market outlook
            outlook = None
            if data.get("market_outlook"):
                outlook_data = data["market_outlook"]
                outlook = MarketOutlookResponse(
                    sentiment=outlook_data.get("sentiment", "neutral"),
                    confidence=outlook_data.get("confidence", 0.5),
                    reasoning=outlook_data.get("reasoning", ""),
                    key_drivers=outlook_data.get("key_drivers", []),
                )
            
            # Parse market summary bullets
            bullets = []
            if data.get("market_summary"):
                for bullet_data in data["market_summary"]:
                    if isinstance(bullet_data, dict) and bullet_data.get("text"):
                        bullets.append(MarketSummaryBulletResponse(
                            text=bullet_data.get("text", ""),
                            supporting_news_ids=bullet_data.get("supporting_news_ids", []),
                            confidence=bullet_data.get("confidence", 0.7),
                            sentiment=bullet_data.get("sentiment", "neutral"),
                        ))
            
            # Parse executive summary
            exec_summary = data.get("executive_summary", "")
            if not isinstance(exec_summary, str):
                exec_summary = str(exec_summary) if exec_summary else ""

            # Parse themed (impacted themes: sector, relevant_companies, sentiment, sentiment_score)
            themed_list: List[ThemedItemResponse] = []
            if data.get("themed") and isinstance(data["themed"], list):
                for t in data["themed"]:
                    if isinstance(t, dict) and t.get("sector"):
                        try:
                            themed_list.append(ThemedItemResponse(
                                sector=str(t.get("sector", "")),
                                relevant_companies=list(t.get("relevant_companies", [])) if isinstance(t.get("relevant_companies"), list) else [],
                                sentiment=t.get("sentiment", "neutral"),
                                sentiment_score=t.get("sentiment_score") if t.get("sentiment_score") is not None else None,
                            ))
                        except Exception:
                            continue
            
            return cls(
                market_outlook=outlook,
                market_summary=bullets,
                executive_summary=exec_summary,
                themed=themed_list,
            )
            
        except Exception:
            # Return empty response on any parsing error
            return cls()
    
    def to_snapshot_dict(self, nifty_change: float = 0.0) -> Dict[str, Any]:
        """Convert to dict format expected by snapshot document."""
        result: Dict[str, Any] = {
            "market_outlook": None,
            "market_summary": [],
            "executive_summary": self.executive_summary or None,
            "themed": [],
        }
        
        if self.market_outlook:
            result["market_outlook"] = {
                "sentiment": self.market_outlook.sentiment,
                "confidence": self.market_outlook.confidence,
                "reasoning": self.market_outlook.reasoning,
                "nifty_change_percent": nifty_change,
                "key_drivers": self.market_outlook.key_drivers,
            }
        
        for bullet in self.market_summary:
            if bullet.text:
                result["market_summary"].append({
                    "text": bullet.text,
                    "supporting_news_ids": bullet.supporting_news_ids,
                    "confidence": bullet.confidence,
                    "sentiment": bullet.sentiment,
                })

        for t in self.themed:
            item: Dict[str, Any] = {
                "sector": t.sector,
                "relevant_companies": t.relevant_companies,
                "sentiment": t.sentiment,
            }
            if t.sentiment_score is not None:
                item["sentiment_score"] = t.sentiment_score
            result["themed"].append(item)
        
        return result


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
            model_name=settings.GEMINI_PRO_MODEL,
            temperature=0.3,  # Slightly creative for narrative
            max_output_tokens=5000,
        )
        self.logger = get_logger("snapshot_generator")

    async def generate_snapshot(
        self,
        market_phase: str,
        indices_data: Dict[str, Any],
        news_items: List[NewsArticleDocument],
        previous_snapshot: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Generate complete snapshot content.
        
        Args:
            market_phase: Current market phase (pre/mid/post)
            indices_data: Current indices data
            news_items: Recent news articles
            previous_snapshot: Previous snapshot for same phase (for context continuity)
            
        Returns:
            Dict with snapshot content
        """
        result = {
            "market_outlook": None,
            "market_summary": [],
            "executive_summary": None,
            "trending_now": None,
            "themed": [],
        }
        
        # Get NIFTY/SENSEX data for context
        nifty = indices_data.get("NIFTY", indices_data.get("SENSEX", {}))
        nifty_change = nifty.get("change_percent", 0) if nifty else 0
        
        # Generate market outlook and summary on all phases (pre, mid, post)
        ai_result = await self._generate_ai_snapshot(
            market_phase, indices_data, news_items, previous_snapshot
        )
        if ai_result:
            result.update(ai_result)
            self.logger.info(
                "snapshot_generated_with_ai",
                market_phase=market_phase,
            )
        else:
            self.logger.info(
                "snapshot_fallback_to_rules",
                market_phase=market_phase,
                reason="AI generation failed or returned empty",
            )
            fallback_result = self._generate_rule_based_snapshot(
                market_phase, indices_data, news_items
            )
            result.update(fallback_result)
        
        # Mid-market: also populate trending_now from top impacting news
        if market_phase == "mid":
            result["trending_now"] = self._select_trending_by_impact(news_items)
            if not result.get("executive_summary"):
                result["executive_summary"] = "Market activity ongoing. Key developments being monitored."
        
        return result

    async def _generate_ai_snapshot(
        self,
        market_phase: str,
        indices_data: Dict[str, Any],
        news_items: List[NewsArticleDocument],
        previous_snapshot: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Generate snapshot using AI with graceful error handling.
        
        Returns:
            Dict with snapshot content on success, None on failure.
            Caller should fall back to rule-based generation on None.
        """
        self.logger.info(
            "ai_snapshot_generation_started",
            market_phase=market_phase,
            news_count=len(news_items),
            has_previous=previous_snapshot is not None,
        )
        
        prompt = self._build_snapshot_prompt(
            market_phase, indices_data, news_items, previous_snapshot
        )
        
        try:
            response = await self.client.generate_content(prompt)
            
            if not response:
                self.logger.warning(
                    "ai_snapshot_empty_response",
                    market_phase=market_phase,
                )
                return None
            
            result = self._parse_snapshot_response(response, indices_data)
            
            if result:
                self.logger.info(
                    "ai_snapshot_generation_success",
                    market_phase=market_phase,
                    has_outlook=result.get("market_outlook") is not None,
                    bullet_count=len(result.get("market_summary", [])),
                )
            else:
                self.logger.warning(
                    "ai_snapshot_parse_failed",
                    market_phase=market_phase,
                    message="Response parsing returned None, will use fallback",
                )
            
            return result
            
        except TimeoutError as e:
            self.logger.error(
                "ai_snapshot_timeout",
                market_phase=market_phase,
                error=str(e),
            )
            return None
            
        except Exception as e:
            self.logger.error(
                "ai_snapshot_error",
                market_phase=market_phase,
                error=str(e),
                error_type=type(e).__name__,
            )
            return None

    def _build_snapshot_prompt(
        self,
        market_phase: str,
        indices_data: Dict[str, Any],
        news_items: List[NewsArticleDocument],
        previous_snapshot: Optional[Dict[str, Any]] = None,
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
            "pre": "Pre-Market (07:00–09:15 IST). Focus on overnight developments and opening expectations. Market outlook is ALLOWED; derive it ONLY from NIFTY 50.",
            "mid": "Mid-Market (09:15–15:30 IST). Focus on live, fast-moving developments and factual structure. Market outlook is ALLOWED; derive it ONLY from NIFTY 50.",
            "post": "Post-Market (15:30–07:00 IST). Summarize the day's key movements and drivers. Market outlook is ALLOWED; derive it ONLY from NIFTY 50.",
        }
        
        # Build previous snapshot context if available
        previous_context = ""
        if previous_snapshot:
            previous_context = self._format_previous_snapshot_context(previous_snapshot)
        
        return f"""You are a Market Intelligence Agent specializing in Indian equity markets.
Your role is to gather, analyze, and STRUCTURE market intelligence.

CORE OBJECTIVE
Convert fragmented market data and news into structured, factual intelligence that enables clear market context, noise reduction, and accurate stock/theme linkage.

MARKET PHASE (MANDATORY)
Current phase: {market_phase}-market.
- Pre-Market: 07:00 – 09:15 IST
- Mid-Market: 09:15 – 15:30 IST
- Post-Market: 15:30 – 07:00 IST
Market phase controls what signals are allowed downstream.

{phase_context.get(market_phase, '')}

MARKET OUTLOOK (STRICT RULES)
- Compute market outlook in all phases (Pre-Market, Mid-Market, Post-Market).
- Market outlook is derived ONLY from NIFTY 50 movement.
- Allowed values: bullish | bearish | neutral.

MARKET DATA
Indian indices (primary): NIFTY 50, SENSEX, sectoral. Use for outlook and reasoning.
Global indices are contextual only; they must NOT directly influence market outlook.

Current Indices:
{indices_text}

Recent News:
{news_text}
{previous_context}

CONSTRAINTS
- No predictions. No trading advice.
- Accuracy, structure, and restraint are critical.

OUTPUT CONTRACT (STRICT)
Return ONLY a valid JSON object with the following structure. No other text.

1. "market_outlook": Include for all phases (pre, mid, post). {{
     "sentiment": "bullish" | "bearish" | "neutral",
     "confidence": 0.0-1.0,
     "reasoning": "2-3 sentence factual explanation",
     "nifty_change_percent": number,
     "key_drivers": ["driver1", "driver2"]
   }}

2. "market_summary": Array of exactly 3-5 bullets, each with:
   - "text": Factual summary with MANDATORY causal language ("due to", "driven by", "following", "amid", "on the back of"). Explain WHY.
   - "supporting_news_ids": [list of relevant news IDs if known]
   - "confidence": 0.0-1.0

   - "sentiment": "bullish" | "bearish" | "neutral"
   Bad: "Markets closed higher"
   Good: "Markets closed higher driven by positive global cues following US market rally"

3. "executive_summary": A 2-3 sentence internal overview of the market (structured intelligence, not user-facing copy).

4. "themed": Array of impacted themes/sectors (ALWAYS include when relevant). Each item:
   - "sector": Theme or sector name (e.g. "Banking", "IT", "Auto")
   - "relevant_companies": List of company names or tickers mentioned for this theme
   - "sentiment": "bullish" | "bearish" | "neutral"
   - "sentiment_score": Optional number 0.0-1.0 (strength of sentiment; omit if unknown)
   Include 0-10 themed items based on news and indices. Omit "themed" or use [] if none identified.

Return ONLY valid JSON, no other text."""

    def _format_previous_snapshot_context(
        self,
        previous_snapshot: Dict[str, Any],
    ) -> str:
        """Format previous snapshot for inclusion in prompt context."""
        context_parts = ["\n--- PREVIOUS SNAPSHOT CONTEXT ---"]
        context_parts.append(
            "Build upon this previous analysis. Focus on NEW developments and changes. "
            "Avoid repeating the same points unless they remain highly relevant."
        )
        
        # Previous outlook
        if previous_snapshot.get("market_outlook"):
            outlook = previous_snapshot["market_outlook"]
            context_parts.append(f"\nPrevious Sentiment: {outlook.get('sentiment', 'neutral')}")
            if outlook.get("reasoning"):
                context_parts.append(f"Previous Reasoning: {outlook['reasoning']}")
            if outlook.get("key_drivers"):
                drivers = ", ".join(outlook["key_drivers"][:3])
                context_parts.append(f"Previous Key Drivers: {drivers}")
        
        # Previous summary bullets
        if previous_snapshot.get("market_summary"):
            context_parts.append("\nPrevious Summary Points:")
            for i, bullet in enumerate(previous_snapshot["market_summary"][:3], 1):
                text = bullet.get("text", "") if isinstance(bullet, dict) else str(bullet)
                if text:
                    context_parts.append(f"  {i}. {text}")
        
        # Previous executive summary
        if previous_snapshot.get("executive_summary"):
            context_parts.append(f"\nPrevious Executive Summary: {previous_snapshot['executive_summary']}")
        
        context_parts.append("--- END PREVIOUS CONTEXT ---\n")
        
        return "\n".join(context_parts)

    def _parse_snapshot_response(
        self,
        response: str,
        indices_data: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        """
        Parse AI response for snapshot using structured Pydantic models.
        
        Provides graceful handling of malformed responses by:
        - Cleaning JSON formatting artifacts
        - Using Pydantic models for type validation
        - Falling back to empty/default values on parse errors
        - Filtering bullets without causal language
        """
        # Get NIFTY change for context
        nifty = indices_data.get("NIFTY", indices_data.get("SENSEX", {}))
        nifty_change = nifty.get("change_percent", 0) if nifty else 0
        
        try:
            # Clean response - remove markdown code blocks
            text = response.strip()
            if text.startswith("```json"):
                text = text[7:]
            elif text.startswith("```"):
                text = text[3:]
            if text.endswith("```"):
                text = text[:-3]
            text = text.strip()
            
            # Parse JSON
            data = json.loads(text)
            
            # Use structured model to parse and validate
            structured_response = SnapshotAIResponse.from_raw_response(
                data=data,
                nifty_change=nifty_change,
            )
            
            # Convert to dict format
            result = structured_response.to_snapshot_dict(nifty_change=nifty_change)
            
            # Filter bullets: only keep those with causal language
            if result.get("market_summary"):
                result["market_summary"] = [
                    bullet for bullet in result["market_summary"]
                    if self._has_causal_language(bullet.get("text", ""))
                ]
            
            # Validate we have meaningful content
            has_outlook = result.get("market_outlook") is not None
            has_summary = len(result.get("market_summary", [])) > 0
            has_executive = bool(result.get("executive_summary"))
            
            if not (has_outlook or has_summary or has_executive):
                self.logger.warning(
                    "snapshot_parse_empty",
                    message="Parsed response has no meaningful content"
                )
                return None
            
            self.logger.info(
                "snapshot_parsed_successfully",
                has_outlook=has_outlook,
                bullet_count=len(result.get("market_summary", [])),
                has_executive=has_executive,
            )
            
            return result
            
        except json.JSONDecodeError as e:
            self.logger.warning(
                "snapshot_json_parse_error",
                error=str(e),
                response_preview=response[:200] if response else "empty"
            )
            return None
            
        except Exception as e:
            self.logger.warning(
                "snapshot_parse_error",
                error=str(e),
                error_type=type(e).__name__,
            )
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

        result["themed"] = []
        
        return result

    def _select_trending_by_impact(
        self,
        news_items: List[NewsArticleDocument],
    ) -> List[str]:
        """Select trending news IDs for mid-market based on top impacting news."""
        if not news_items:
            return []

        def _mag(s: Any) -> str:
            if hasattr(s, "impact_magnitude"):
                return getattr(s, "impact_magnitude", "medium") or "medium"
            return (s.get("impact_magnitude", "medium") if isinstance(s, dict) else "medium")

        def impact_score(n: NewsArticleDocument) -> tuple:
            # (breaking first, has_impact, high_impact_count, medium_impact_count, sentiment_magnitude, recency)
            breaking = 1 if getattr(n, "is_breaking", False) else 0
            impacted = getattr(n, "impacted_stocks", None) or []
            high = sum(1 for s in impacted if _mag(s) == "high")
            medium = sum(1 for s in impacted if _mag(s) == "medium")
            has_impact = 1 if (impacted or getattr(n, "sector_impacts", None)) else 0
            sentiment_mag = abs(getattr(n, "sentiment_score", 0) or 0)
            ts = getattr(n, "published_at", None)
            recency = ts.timestamp() if ts and hasattr(ts, "timestamp") else 0.0
            return (breaking, has_impact, high, medium, sentiment_mag, recency)

        sorted_news = sorted(
            news_items,
            key=impact_score,
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
