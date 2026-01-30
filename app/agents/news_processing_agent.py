"""
News Processing Agent - Background AI-powered news analysis.

This agent is invoked by Celery tasks for:
- Sentiment analysis of news articles
- Entity extraction (stocks, sectors, companies)
- Summary generation
- Impact analysis

Part of the 3-agent background processing architecture:
1. NewsProcessingAgent - AI news analysis (called by fetch_news task)
2. SnapshotGenerationAgent - AI snapshot generation (called by gen_snapshot task)
3. IndicesCollectionAgent - Indices data collection (called by fetch_indices task)
"""

from typing import Any, Dict, List, Optional

from app.config import get_settings
from app.db.models.news_document import NewsArticleDocument
from app.services.news_processor_service import NewsProcessorService
from app.utils.logging import get_logger

logger = get_logger(__name__)
settings = get_settings()


class NewsProcessingAgent:
    """
    Background agent for AI-powered news analysis.
    
    Responsibilities:
    - Analyze news sentiment (bullish/bearish/neutral)
    - Extract entities (stocks, sectors, companies)
    - Generate concise summaries
    - Identify stock and sector impacts
    - Generate causal chains
    
    This agent wraps NewsProcessorService and provides a clean
    interface for Celery tasks.
    """
    
    def __init__(self):
        self.processor = NewsProcessorService()
        self.logger = get_logger("news_processing_agent")
    
    async def analyze_article(
        self,
        article: NewsArticleDocument,
    ) -> Optional[Dict[str, Any]]:
        """
        Analyze a single news article with AI.
        
        Args:
            article: NewsArticleDocument to analyze
            
        Returns:
            Dict with analysis results:
            - sentiment: bullish/bearish/neutral
            - sentiment_score: -1.0 to 1.0
            - summary: Concise summary
            - mentioned_stocks: List of stock tickers
            - mentioned_sectors: List of sectors
            - impacted_stocks: List of impact objects
            - sector_impacts: Dict of sector to impact
            - causal_chain: Causal explanation
        """
        self.logger.info(
            "analyzing_article",
            news_id=article.news_id,
            headline=article.headline[:50],
        )
        
        try:
            result = await self.processor.analyze_news_article(article)
            
            if result:
                self.logger.info(
                    "article_analyzed",
                    news_id=article.news_id,
                    sentiment=result.get("sentiment"),
                    stocks_found=len(result.get("mentioned_stocks", [])),
                )
            
            return result
            
        except Exception as e:
            self.logger.error(
                "article_analysis_failed",
                news_id=article.news_id,
                error=str(e),
            )
            return None
    
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
            List of analysis results (excludes failures)
        """
        self.logger.info(
            "batch_analysis_started",
            article_count=len(articles),
            max_concurrent=max_concurrent,
        )
        
        results = await self.processor.analyze_batch(articles, max_concurrent)
        
        self.logger.info(
            "batch_analysis_completed",
            input_count=len(articles),
            success_count=len(results),
        )
        
        return results
    
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
        return await self.processor.summarize_text(text, max_words)


# Singleton instance
_news_processing_agent: Optional[NewsProcessingAgent] = None


def get_news_processing_agent() -> NewsProcessingAgent:
    """Get singleton NewsProcessingAgent instance."""
    global _news_processing_agent
    if _news_processing_agent is None:
        _news_processing_agent = NewsProcessingAgent()
    return _news_processing_agent
