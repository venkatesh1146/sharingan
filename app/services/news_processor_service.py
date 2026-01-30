"""
News Processor Service - AI-powered news analysis.

Provides:
- Sentiment analysis
- Entity extraction (stocks, sectors, companies)
- Summary generation
- Impact analysis
"""

from typing import Any, Dict, List, Optional
import json
import re

from app.config import get_settings
from app.constants.tickers import COMMON_NSE_TICKERS
from app.db.models.news_document import NewsArticleDocument
from app.utils.logging import get_logger
from app.utils.vertex_ai_client import VertexAIClient

logger = get_logger(__name__)
settings = get_settings()


# Sentiment keywords for rule-based fallback
BULLISH_KEYWORDS = [
    "rally", "surge", "jump", "gain", "rise", "climb", "advance", "bullish",
    "positive", "growth", "record high", "outperform", "upgrade", "beat",
    "strong", "robust", "optimism", "recovery", "boost", "expand",
]

BEARISH_KEYWORDS = [
    "fall", "drop", "decline", "slump", "plunge", "crash", "bearish",
    "negative", "loss", "concern", "fear", "warning", "downgrade", "miss",
    "weak", "slowdown", "pessimism", "retreat", "contract", "cut",
]

# Sector mapping
SECTOR_KEYWORDS = {
    "Banking": ["bank", "npa", "loan", "credit", "deposit", "rbi", "monetary"],
    "IT": ["it", "software", "tech", "digital", "ai", "cloud", "saas"],
    "Pharma": ["pharma", "drug", "fda", "medicine", "healthcare", "hospital"],
    "Auto": ["auto", "vehicle", "car", "ev", "electric vehicle", "automobile"],
    "Energy": ["oil", "gas", "energy", "power", "electricity", "renewable"],
    "Metals": ["steel", "metal", "iron", "copper", "aluminium", "mining"],
    "FMCG": ["fmcg", "consumer", "retail", "food", "beverage"],
    "Realty": ["real estate", "realty", "property", "housing", "infrastructure"],
    "Telecom": ["telecom", "5g", "spectrum", "broadband", "mobile"],
    "Economy": ["gdp", "inflation", "fiscal", "budget", "economy", "rbi", "policy"],
}


class NewsProcessorService:
    """
    Service for AI-powered news analysis.
    
    Uses Google Gemini for:
    - Sentiment classification
    - Summary generation
    - Entity extraction
    - Impact analysis
    """
    
    def __init__(self):
        self.client = VertexAIClient(
            model_name=settings.GEMINI_FAST_MODEL,
            temperature=0.1,
            max_output_tokens=2048,
        )
        self.logger = get_logger("news_processor")

    async def analyze_news_article(
        self,
        article: NewsArticleDocument,
    ) -> Optional[Dict[str, Any]]:
        """
        Analyze a news article with AI.
        
        Args:
            article: NewsArticleDocument to analyze
            
        Returns:
            Dict with analysis results or None on failure
        """
        try:
            # Try AI analysis first
            result = await self._ai_analyze(article)
            if result:
                return result
        except Exception as e:
            self.logger.warning(
                "ai_analysis_failed",
                news_id=article.news_id,
                error=str(e),
            )
        
        # Fall back to rule-based analysis
        return self._rule_based_analyze(article)

    async def _ai_analyze(
        self,
        article: NewsArticleDocument,
    ) -> Optional[Dict[str, Any]]:
        """AI-powered analysis using Gemini."""
        prompt = self._build_analysis_prompt(article)
        
        try:
            response = await self.client.generate_content(prompt)
            
            if not response:
                return None
            
            # Parse JSON response
            return self._parse_ai_response(response, article)
            
        except Exception as e:
            self.logger.warning("ai_analysis_error", error=str(e))
            return None

    def _build_analysis_prompt(self, article: NewsArticleDocument) -> str:
        """Build the analysis prompt for Gemini."""
        return f"""Analyze this financial news article and provide structured analysis.

Headline: {article.headline}
Summary: {article.summary or article.full_text or 'No summary available'}
Source: {article.source}

Analyze and return a JSON object with:
1. "sentiment": One of "bullish", "bearish", or "neutral"
2. "sentiment_score": A score from -1.0 (very bearish) to 1.0 (very bullish)
3. "summary": A concise 1-2 sentence summary (max 100 words)
4. "mentioned_stocks": List of NSE stock tickers mentioned (e.g., ["RELIANCE", "TCS"])
5. "mentioned_sectors": List of sectors affected (e.g., ["Banking", "IT", "Economy"])
6. "impacted_stocks": List of objects with {{"ticker": "XXX", "impact_type": "positive/negative/neutral", "reasoning": "why"}}
7. "sector_impacts": Object mapping sector to impact type (e.g., {{"Banking": "positive"}})
8. "causal_chain": A brief explanation of the impact chain (e.g., "Oil prices ↑ → Paints costs ↑ → Asian Paints margins ↓")

Focus on Indian market context. Return ONLY valid JSON, no other text."""

    def _parse_ai_response(
        self,
        response: str,
        article: NewsArticleDocument,
    ) -> Optional[Dict[str, Any]]:
        """Parse and validate AI response."""
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
            
            # Validate and normalize
            result = {
                "sentiment": data.get("sentiment", "neutral"),
                "sentiment_score": float(data.get("sentiment_score", 0.0)),
                "summary": data.get("summary", article.summary)[:500],
                "mentioned_stocks": data.get("mentioned_stocks", []),
                "mentioned_sectors": data.get("mentioned_sectors", []),
                "impacted_stocks": data.get("impacted_stocks", []),
                "sector_impacts": data.get("sector_impacts", {}),
                "causal_chain": data.get("causal_chain", ""),
            }
            
            # Validate sentiment
            if result["sentiment"] not in ["bullish", "bearish", "neutral"]:
                result["sentiment"] = "neutral"
            
            # Clamp sentiment score
            result["sentiment_score"] = max(-1.0, min(1.0, result["sentiment_score"]))
            
            return result
            
        except (json.JSONDecodeError, ValueError, TypeError) as e:
            self.logger.warning("ai_response_parse_error", error=str(e))
            return None

    def _rule_based_analyze(
        self,
        article: NewsArticleDocument,
    ) -> Dict[str, Any]:
        """Rule-based analysis fallback."""
        text = f"{article.headline} {article.summary}".lower()
        
        # Sentiment analysis
        bullish_count = sum(1 for kw in BULLISH_KEYWORDS if kw in text)
        bearish_count = sum(1 for kw in BEARISH_KEYWORDS if kw in text)
        
        total = bullish_count + bearish_count
        if total > 0:
            score = (bullish_count - bearish_count) / total
        else:
            score = 0.0
        
        if score > 0.2:
            sentiment = "bullish"
        elif score < -0.2:
            sentiment = "bearish"
        else:
            sentiment = "neutral"
        
        # Extract stocks
        mentioned_stocks = self._extract_tickers(text)
        
        # Extract sectors
        mentioned_sectors = self._extract_sectors(text)
        
        # Generate summary if needed
        summary = article.summary
        if not summary or len(summary) < 10:
            summary = article.headline
        elif len(summary.split()) > 100:
            words = summary.split()[:100]
            summary = " ".join(words) + "..."
        
        return {
            "sentiment": sentiment,
            "sentiment_score": round(score, 2),
            "summary": summary,
            "mentioned_stocks": mentioned_stocks,
            "mentioned_sectors": mentioned_sectors,
            "impacted_stocks": [],
            "sector_impacts": {},
            "causal_chain": "",
        }

    def _extract_tickers(self, text: str) -> List[str]:
        """Extract stock tickers from text."""
        text_upper = text.upper()
        found = []
        
        # Sort by length descending
        sorted_tickers = sorted(COMMON_NSE_TICKERS, key=len, reverse=True)
        
        for ticker in sorted_tickers:
            pattern = r"\b" + re.escape(ticker) + r"\b"
            if re.search(pattern, text_upper):
                found.append(ticker)
        
        return found[:10]  # Limit to 10

    def _extract_sectors(self, text: str) -> List[str]:
        """Extract sectors from text."""
        text_lower = text.lower()
        found = []
        
        for sector, keywords in SECTOR_KEYWORDS.items():
            if any(kw in text_lower for kw in keywords):
                found.append(sector)
        
        return found if found else ["General"]

    async def analyze_batch(
        self,
        articles: List[NewsArticleDocument],
        max_concurrent: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Analyze multiple articles concurrently.
        
        Args:
            articles: List of articles to analyze
            max_concurrent: Maximum concurrent analyses
            
        Returns:
            List of analysis results
        """
        import asyncio
        
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def analyze_with_semaphore(article):
            async with semaphore:
                return await self.analyze_news_article(article)
        
        tasks = [analyze_with_semaphore(a) for a in articles]
        results = await asyncio.gather(*tasks)
        
        return [r for r in results if r is not None]

    async def summarize_text(
        self,
        text: str,
        max_words: int = 100,
    ) -> str:
        """
        Generate a concise summary of text.
        
        Args:
            text: Text to summarize
            max_words: Maximum words in summary
            
        Returns:
            Summarized text
        """
        if len(text.split()) <= max_words:
            return text
        
        prompt = f"""Summarize this text in under {max_words} words, preserving key information:

{text}

Return only the summary, no other text."""
        
        try:
            response = await self.client.generate_content(prompt)
            if response:
                return response.strip()[:500]
        except Exception as e:
            self.logger.warning("summarize_error", error=str(e))
        
        # Fallback: truncate
        words = text.split()[:max_words]
        return " ".join(words) + "..."


# Singleton instance
_news_processor: Optional[NewsProcessorService] = None


def get_news_processor_service() -> NewsProcessorService:
    """Get singleton NewsProcessorService instance."""
    global _news_processor
    if _news_processor is None:
        _news_processor = NewsProcessorService()
    return _news_processor
