"""
Market Intelligence Service - Unified service for market data and news fetching.

This service consolidates all data fetching operations:
- Market index data (from CMOTS World Indices API)
- Market phase determination
- News fetching (general and stock-specific)
- News clustering
"""

from datetime import datetime, timedelta, time
from typing import Any, Callable, Dict, List, Optional
import asyncio
import json
import pytz

from app.config import get_settings
from app.constants.themes import (
    MAX_THEMED_NEWS_ITEMS,
    normalize_theme_to_allowed,
)
from app.services.cmots_news_service import (
    fetch_world_indices,
    get_market_news_service,
)
from app.utils.logging import get_logger
from app.agents.summary_generation_agent import SummaryGenerationAgent

logger = get_logger(__name__)

# Indian Standard Time
IST = pytz.timezone("Asia/Kolkata")


# =============================================================================
# Index Name Mapping (API names to normalized names)
# =============================================================================

# Map API index names to normalized keys for lookups
INDEX_NAME_MAP = {
    "Nifty": "NIFTY",
    "BSE Sensex": "SENSEX",
    "GIFT NIFTY": "GIFT NIFTY",
    "Hang Seng": "HANG SENG",
    "Nikkei 225": "NIKKEI 225",
    "DAX": "DAX",
    "CAC 40": "CAC 40",
    "FTSE 100": "FTSE 100",
    "DJIA": "DJIA",
    "S&P 500": "S&P 500",
    "Shanghai Composite": "SHANGHAI COMPOSITE",
    "Taiwan Weighted": "TAIWAN WEIGHTED",
    "ASX 200": "ASX 200",
    "KOSPI": "KOSPI",
    "US Tech 100": "US TECH 100",
}

# Primary indices for Indian market analysis
INDIAN_PRIMARY_INDICES = ["NIFTY", "SENSEX", "GIFT NIFTY"]

# Phase-specific indices to show in the response
# Pre-market: Show global indices that trade before Indian markets open
PRE_MARKET_INDICES = [
    "GIFT NIFTY",   # GIFT Nifty (pre-market indicator)
    "NIKKEI 225",   # Japan - Nikkei
    "FTSE 100",     # UK - FTSE 100
    "SHANGHAI COMPOSITE",  # China - Shanghai Composite
    "DAX",          # Germany - DAX
]

# Post-market: Show indices relevant after Indian markets close
POST_MARKET_INDICES = [
    "SENSEX",       # BSE Sensex (Indian)
    "NIFTY",        # Nifty 50 (Indian)
    "SHANGHAI COMPOSITE",  # China - Shanghai Composite
    "NIKKEI 225",   # Japan - Nikkei
    "FTSE 100",     # UK - FTSE 100
    "DJIA",         # US - Dow Jones
    "S&P 500",      # US - S&P 500
]

# Mid-market: Show Indian indices primarily
MID_MARKET_INDICES = [
    "SENSEX",       # BSE Sensex
    "NIFTY",        # Nifty 50
]


# News type to sector mapping for categorization
NEWS_TYPE_SECTOR_MAP = {
    "Economy": ["Economy", "Macro", "Policy"],
    "Other Markets": ["Commodities", "Forex", "Bullion"],
    "Foreign Markets": ["Global Markets", "International"],
}

# Keywords for sentiment analysis from headlines/summary
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

# Maximum words for news summary
MAX_SUMMARY_WORDS = 100


async def _summarize_news_batch(news_items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Summarize news item summaries using the SummaryGenerationAgent to be concise (under 100 words).

    Delegates to the SummaryGenerationAgent for consistent summarization across the system.

    Args:
        news_items: List of news items with 'summary' field

    Returns:
        Same list with summaries condensed to under 100 words
    """
    if not news_items:
        return news_items

    # Filter items that need summarization (over 100 words)
    items_to_summarize = []
    indices_to_summarize = []

    for i, item in enumerate(news_items):
        summary = item.get("summary", "")
        word_count = len(summary.split())
        if word_count > MAX_SUMMARY_WORDS:
            items_to_summarize.append({"id": item.get("id", str(i)), "summary": summary})
            indices_to_summarize.append(i)

    if not items_to_summarize:
        # All summaries are already concise
        return news_items

    logger.info(
        "summarizing_news_batch",
        total_items=len(news_items),
        items_to_summarize=len(items_to_summarize),
    )

    try:
        # Use SummaryGenerationAgent for summarization
        agent = SummaryGenerationAgent()
        summarized_items = await agent.summarize_news_batch(
            news_items=items_to_summarize,
            max_words=MAX_SUMMARY_WORDS,
        )

        # Create a mapping of id to summarized text
        summary_map = {item["id"]: item["summary"] for item in summarized_items}

        # Update the original news items with summarized content
        for idx in indices_to_summarize:
            item_id = news_items[idx].get("id", str(idx))
            if item_id in summary_map:
                news_items[idx]["summary"] = summary_map[item_id]

        logger.info(
            "news_summarization_complete",
            summarized_count=len(summarized_items),
        )

    except json.JSONDecodeError as e:
        logger.warning(
            "news_summarization_json_error",
            error=str(e),
        )
        # Fall back to simple truncation if LLM response is malformed
        for idx in indices_to_summarize:
            summary = news_items[idx].get("summary", "")
            words = summary.split()[:MAX_SUMMARY_WORDS]
            news_items[idx]["summary"] = " ".join(words) + "..." if len(summary.split()) > MAX_SUMMARY_WORDS else summary

    except Exception as e:
        logger.warning(
            "news_summarization_error",
            error=str(e),
        )
        # Fall back to simple truncation on any error
        for idx in indices_to_summarize:
            summary = news_items[idx].get("summary", "")
            words = summary.split()[:MAX_SUMMARY_WORDS]
            news_items[idx]["summary"] = " ".join(words) + "..." if len(summary.split()) > MAX_SUMMARY_WORDS else summary

    return news_items


# =============================================================================
# Market Data Functions
# =============================================================================


async def fetch_market_indices(
    indices: Optional[List[str]] = None,
) -> Dict[str, Dict[str, Any]]:
    """
    Fetch current market data for all world indices from CMOTS API.

    Args:
        indices: Optional list of index names to filter (if None, returns all)

    Returns:
        Dictionary mapping normalized index name to index data

    API Response Format:
        {
            "indexname": "BSE Sensex",
            "Country": "India",
            "date": "2026-01-30T14:17:00",
            "close": 82365.62,
            "Chg": -200.75,
            "PChg": -0.24,
            "PrevClose": 82566.37
        }
    """
    result = {}

    try:
        # Fetch all world indices from API
        api_response = await fetch_world_indices()
        raw_indices = api_response.get("data", [])

        logger.info("fetched_world_indices", count=len(raw_indices))

        for idx_data in raw_indices:
            index_name = idx_data.get("indexname", "Unknown")
            normalized_name = INDEX_NAME_MAP.get(index_name, index_name.upper())

            # Parse timestamp from API
            date_str = idx_data.get("date", "")
            try:
                timestamp = datetime.fromisoformat(date_str) if date_str else datetime.now(IST)
            except ValueError:
                timestamp = datetime.now(IST)

            # Map API response to our standard format
            result[normalized_name] = {
                "ticker": normalized_name,
                "name": index_name,
                "country": idx_data.get("Country", "Unknown"),
                "current_price": float(idx_data.get("close", 0)),
                "change_percent": float(idx_data.get("PChg", 0)),
                "change_absolute": float(idx_data.get("Chg", 0)),
                "previous_close": float(idx_data.get("PrevClose", 0)),
                "intraday_high": float(idx_data.get("close", 0)),  # API doesn't provide high
                "intraday_low": float(idx_data.get("close", 0)),   # API doesn't provide low
                "volume": 0,  # API doesn't provide volume
                "timestamp": timestamp.isoformat() if hasattr(timestamp, 'isoformat') else str(timestamp),
            }

        # Filter by requested indices if provided
        if indices:
            # Normalize requested indices for comparison
            normalized_requested = {idx.upper(): idx for idx in indices}
            filtered_result = {}

            for key, data in result.items():
                if key in normalized_requested:
                    filtered_result[key] = data
                # Also check original name match
                elif data["name"].upper() in normalized_requested:
                    filtered_result[key] = data

            # Add placeholder for any requested indices not found
            for requested_idx in indices:
                normalized_req = requested_idx.upper()
                if normalized_req not in filtered_result and requested_idx.upper() not in [d["name"].upper() for d in filtered_result.values()]:
                    filtered_result[normalized_req] = {
                        "ticker": normalized_req,
                        "name": requested_idx,
                        "country": "Unknown",
                        "current_price": 0.0,
                        "change_percent": 0.0,
                        "change_absolute": 0.0,
                        "previous_close": 0.0,
                        "intraday_high": 0.0,
                        "intraday_low": 0.0,
                        "volume": 0,
                        "timestamp": datetime.now(IST).isoformat(),
                        "error": f"Index {requested_idx} not found in API response",
                    }

            return filtered_result

        return result

    except Exception as e:
        logger.error("fetch_market_indices_error", error=str(e))
        # Return empty result with error for requested indices
        timestamp = datetime.now(IST)
        if indices:
            for idx in indices:
                result[idx.upper()] = {
                    "ticker": idx.upper(),
                    "name": idx,
                    "country": "Unknown",
                    "current_price": 0.0,
                    "change_percent": 0.0,
                    "change_absolute": 0.0,
                    "previous_close": 0.0,
                    "intraday_high": 0.0,
                    "intraday_low": 0.0,
                    "volume": 0,
                    "timestamp": timestamp.isoformat(),
                    "error": f"API error: {str(e)}",
                }
        return result


async def fetch_all_world_indices() -> Dict[str, Any]:
    """
    Fetch all world indices without filtering.

    Returns:
        Dictionary with all indices data organized by region
    """
    all_indices = await fetch_market_indices(indices=None)

    # Organize by country/region
    by_region = {
        "india": {},
        "asia": {},
        "europe": {},
        "americas": {},
    }

    region_map = {
        "India": "india",
        "Hong Kong": "asia",
        "Japan": "asia",
        "China": "asia",
        "Taiwan": "asia",
        "Australia": "asia",
        "South Korea": "asia",
        "Germany": "europe",
        "France": "europe",
        "United Kingdom": "europe",
        "United States": "americas",
    }

    for ticker, data in all_indices.items():
        country = data.get("country", "Unknown")
        region = region_map.get(country, "asia")
        by_region[region][ticker] = data

    return {
        "all_indices": all_indices,
        "by_region": by_region,
        "total_count": len(all_indices),
        "timestamp": datetime.now(IST).isoformat(),
    }


async def get_market_phase() -> Dict[str, Any]:
    """
    Determine the current market phase based on IST time.

    Market phases:
    - pre: 08:00 - 09:15 (Pre-market)
    - mid: 09:15 - 15:30 (Trading hours)
    - post: 15:30 - 08:00 next day (Post-market)

    Returns:
        Dictionary with phase and timing information
    """
    settings = get_settings()
    now = datetime.now(IST)
    current_time = now.time()

    pre_market_start = time(
        settings.PRE_MARKET_START_HOUR,
        settings.PRE_MARKET_START_MINUTE,
    )
    market_open = time(
        settings.MARKET_OPEN_HOUR,
        settings.MARKET_OPEN_MINUTE,
    )
    market_close = time(
        settings.MARKET_CLOSE_HOUR,
        settings.MARKET_CLOSE_MINUTE,
    )

    if pre_market_start <= current_time < market_open:
        phase = "pre"
        phase_description = "Pre-Market"
        next_event = "Market opens"
        next_event_time = market_open.isoformat()
    elif market_open <= current_time < market_close:
        phase = "mid"
        phase_description = "Market Hours"
        next_event = "Market closes"
        next_event_time = market_close.isoformat()
    else:
        phase = "post"
        phase_description = "Post-Market"
        next_event = "Pre-market starts"
        next_event_time = pre_market_start.isoformat()

    return {
        "phase": phase,
        "phase_description": phase_description,
        "current_time_ist": now.isoformat(),
        "is_trading_day": now.weekday() < 5,
        "next_event": next_event,
        "next_event_time": next_event_time,
    }


async def calculate_index_momentum(
    indices_data: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Calculate market momentum based on index movements.

    Uses NIFTY (or SENSEX as fallback) as the primary index for momentum calculation.

    Args:
        indices_data: Dictionary of index data

    Returns:
        Momentum analysis results
    """
    # Try NIFTY first, then SENSEX as fallback
    nifty_data = indices_data.get("NIFTY", indices_data.get("SENSEX", {}))
    primary_index = "NIFTY" if "NIFTY" in indices_data else "SENSEX"
    nifty_change = nifty_data.get("change_percent", 0.0)

    if nifty_change > 1.0:
        momentum = "strong_up"
        description = "Strong bullish momentum"
    elif nifty_change > 0.5:
        momentum = "moderate_up"
        description = "Moderate bullish momentum"
    elif nifty_change > -0.5:
        momentum = "sideways"
        description = "Sideways/consolidating"
    elif nifty_change > -1.0:
        momentum = "moderate_down"
        description = "Moderate bearish momentum"
    else:
        momentum = "strong_down"
        description = "Strong bearish momentum"

    positive_count = sum(
        1 for data in indices_data.values()
        if data.get("change_percent", 0) > 0
    )
    total_count = len(indices_data)
    breadth = positive_count / total_count if total_count > 0 else 0.5

    # Add global market sentiment
    global_indices = ["S&P 500", "DJIA", "US TECH 100", "FTSE 100", "DAX"]
    global_positive = sum(
        1 for idx in global_indices
        if idx in indices_data and indices_data[idx].get("change_percent", 0) > 0
    )
    global_count = sum(1 for idx in global_indices if idx in indices_data)
    global_sentiment = "positive" if global_positive > global_count / 2 else "negative" if global_positive < global_count / 2 else "mixed"

    return {
        "momentum": momentum,
        "description": description,
        "primary_index": primary_index,
        "primary_change_percent": nifty_change,
        "market_breadth": breadth,
        "advancing_indices": positive_count,
        "declining_indices": total_count - positive_count,
        "global_sentiment": global_sentiment,
    }


# =============================================================================
# News Functions
# =============================================================================


def _analyze_sentiment(headline: str, summary: str) -> tuple[str, float]:
    """
    Analyze sentiment from headline and summary text.

    Returns:
        Tuple of (sentiment label, sentiment score)
    """
    text = f"{headline} {summary}".lower()

    bullish_count = sum(1 for keyword in BULLISH_KEYWORDS if keyword in text)
    bearish_count = sum(1 for keyword in BEARISH_KEYWORDS if keyword in text)

    # Calculate sentiment score (-1 to 1)
    total = bullish_count + bearish_count
    if total == 0:
        return "neutral", 0.0

    score = (bullish_count - bearish_count) / max(total, 1)
    score = max(-1.0, min(1.0, score))  # Clamp to [-1, 1]

    if score > 0.2:
        sentiment = "bullish"
    elif score < -0.2:
        sentiment = "bearish"
    else:
        sentiment = "neutral"

    return sentiment, round(score, 2)


def _transform_news_item(item: Dict[str, Any]) -> Dict[str, Any]:
    """
    Transform CMOTS API news item to internal format.

    Maps:
        - sno -> id
        - heading -> headline
        - section_name -> source
        - news_type -> mentioned_sectors (category)
        - published_at -> published_at
        - summary -> summary

    Args:
        item: Raw news item from CMOTS API

    Returns:
        Transformed news item in internal format
    """
    headline = item.get("heading", "")
    summary = item.get("summary", "")
    news_type = item.get("news_type", "")

    # Analyze sentiment
    sentiment, sentiment_score = _analyze_sentiment(headline, summary)

    # Determine sectors based on news type
    mentioned_sectors = NEWS_TYPE_SECTOR_MAP.get(news_type, [news_type]) if news_type else ["General"]

    # Check if breaking news (most recent 3 articles are considered breaking)
    is_breaking = False  # Will be set later based on recency

    return {
        "id": str(item.get("sno", "")),
        "headline": headline,
        "summary": summary,
        "source": item.get("section_name", "Capital Market"),
        "url": None,  # CMOTS API doesn't provide URLs
        "published_at": item.get("published_at", datetime.now(IST).isoformat()),
        "sentiment": sentiment,
        "sentiment_score": sentiment_score,
        "mentioned_stocks": [],  # Could be extracted from summary if needed
        "mentioned_sectors": mentioned_sectors,
        "relevance_score": 0.5 + (0.25 * abs(sentiment_score)),  # Higher score for stronger sentiment
        "is_breaking": is_breaking,
        # Additional fields from API
        "news_type": news_type,
        "caption": item.get("caption", ""),
    }


async def fetch_market_news(
    time_window_hours: int = 24,
    max_articles: int = 50,
    categories: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """
    Fetch general market news articles from CMOTS API.

    Fetches news from three sources:
    - Economy News (economic reports, policy updates)
    - Other Markets (commodities, forex, bullion)
    - Foreign Markets (global market updates)

    Args:
        time_window_hours: How far back to fetch news (in hours)
        max_articles: Maximum number of articles to return
        categories: Optional list of news types to filter:
            - "economy-news": Economic and policy news
            - "other-markets": Commodities, forex, bullion
            - "foreign-markets": Global market updates

    Returns:
        List of news article dictionaries with standardized format
    """
    cutoff_time = datetime.now(IST) - timedelta(hours=time_window_hours)

    # Map categories to news_type filter
    news_type_filter = None
    if categories and len(categories) == 1:
        news_type_filter = categories[0]

    try:
        # Fetch from CMOTS API
        news_service = get_market_news_service()
        api_response = await news_service.fetch_unified_market_news(
            limit=max_articles,
            page=1,
            per_page=max_articles,
            news_type=news_type_filter,
        )

        logger.info(
            "fetched_market_news",
            total_items=api_response.get("pagination", {}).get("total_items", 0),
            fetched_at=api_response.get("fetched_at"),
        )

        # Collect all news items from all categories
        all_news = []
        data_by_type = api_response.get("data", {})

        for news_type_key, items in data_by_type.items():
            # Filter by categories if provided
            if categories and news_type_key not in categories:
                continue

            for item in items:
                transformed = _transform_news_item(item)
                all_news.append(transformed)

        # Filter by time window
        filtered_news = []
        for article in all_news:
            try:
                pub_time_str = article.get("published_at", "")
                if pub_time_str:
                    pub_time = datetime.fromisoformat(pub_time_str.replace("Z", "+00:00"))
                    # Make naive datetime for comparison if needed
                    if pub_time.tzinfo is None:
                        pub_time = IST.localize(pub_time)
                    if pub_time >= cutoff_time:
                        filtered_news.append(article)
                else:
                    # Include articles without timestamp
                    filtered_news.append(article)
            except (ValueError, TypeError) as e:
                logger.warning("news_time_parse_error", error=str(e), article_id=article.get("id"))
                filtered_news.append(article)  # Include anyway

        # Sort by published_at (most recent first)
        filtered_news.sort(
            key=lambda x: x.get("published_at", ""),
            reverse=True,
        )

        # Mark most recent 3 articles as breaking
        for i, article in enumerate(filtered_news[:3]):
            article["is_breaking"] = True

        # Limit to max_articles
        result = filtered_news[:max_articles]

        # Summarize verbose summaries using LLM (under 100 words)
        result = await _summarize_news_batch(result)

        logger.info(
            "processed_market_news",
            total_fetched=len(all_news),
            after_time_filter=len(filtered_news),
            returned=len(result),
        )

        return result

    except Exception as e:
        logger.error("fetch_market_news_error", error=str(e))
        # Return empty list on error
        return []


async def fetch_stock_specific_news(
    tickers: List[str],
    time_window_hours: int = 24,
    max_articles: int = 20,
) -> List[Dict[str, Any]]:
    """
    Fetch news mentioning specific stocks.

    Searches through market news for articles that mention the given tickers
    in their headline or summary.

    Args:
        tickers: List of stock tickers to search for
        time_window_hours: How far back to fetch news
        max_articles: Maximum articles to return

    Returns:
        List of news articles mentioning the specified stocks
    """
    if not tickers:
        return []

    # Normalize tickers for search
    tickers_upper = [t.upper() for t in tickers]

    # Fetch all market news first
    all_news = await fetch_market_news(
        time_window_hours=time_window_hours,
        max_articles=100,  # Fetch more to search through
    )

    matching_news = []
    for article in all_news:
        # Search for ticker mentions in headline and summary
        text = f"{article.get('headline', '')} {article.get('summary', '')}".upper()

        matched_tickers = []
        for ticker in tickers_upper:
            # Look for ticker as whole word
            if ticker in text:
                matched_tickers.append(ticker)

        if matched_tickers:
            article_copy = article.copy()
            article_copy["matched_tickers"] = matched_tickers
            article_copy["mentioned_stocks"] = matched_tickers
            matching_news.append(article_copy)

    # Sort by relevance (number of matches) and recency
    matching_news.sort(
        key=lambda x: (len(x.get("matched_tickers", [])), x.get("published_at", "")),
        reverse=True,
    )

    return matching_news[:max_articles]


async def cluster_news_by_topic(
    news_items: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Cluster news items into themes using only allowed themes (max 5).

    Groups news by news_type and sector, maps each to an allowed theme via
    normalize_theme_to_allowed, aggregates by allowed theme, and returns
    up to MAX_THEMED_NEWS_ITEMS themes (impacted in post-market / pre-market).
    """
    # Raw groups: news_type -> news_ids, sector -> news_ids
    type_groups: Dict[str, List[str]] = {}
    sector_groups: Dict[str, List[str]] = {}

    for article in news_items:
        article_id = article.get("id", "")
        news_type = article.get("news_type", "General")
        if news_type:
            type_groups.setdefault(news_type, []).append(article_id)
        sectors = article.get("mentioned_sectors", ["General"])
        primary_sector = sectors[0] if sectors else "General"
        sector_groups.setdefault(primary_sector, []).append(article_id)

    # Build candidate themes and map to allowed theme
    candidates: List[tuple] = []

    def add_candidate(raw_name: str, news_ids: List[str], cluster_articles: List[Dict]) -> None:
        if not cluster_articles:
            return
        allowed = normalize_theme_to_allowed(raw_name)
        if not allowed:
            return
        avg_sentiment = sum(a.get("sentiment_score", 0) for a in cluster_articles) / len(cluster_articles)
        if avg_sentiment > 0.2:
            sentiment = "bullish"
        elif avg_sentiment < -0.2:
            sentiment = "bearish"
        else:
            sentiment = "neutral"
        mentioned_stocks = set()
        for a in cluster_articles:
            mentioned_stocks.update(a.get("mentioned_stocks", []))
        confidence = min(0.7 + (0.05 * len(news_ids)), 0.95)
        candidates.append((allowed, news_ids, sentiment, mentioned_stocks, confidence))

    for news_type, news_ids in type_groups.items():
        cluster_articles = [a for a in news_items if a.get("id") in news_ids]
        raw_name = f"{news_type} News"
        if news_type == "Economy":
            raw_name = "Economic & Policy Updates"
        elif news_type == "Other Markets":
            raw_name = "Commodities & Forex"
        elif news_type == "Foreign Markets":
            raw_name = "Global Market Updates"
        add_candidate(raw_name, news_ids, cluster_articles)

    skip_sectors = {"Economy", "Macro", "Policy", "Commodities", "Forex", "Bullion", "Global Markets", "International"}
    for sector, news_ids in sector_groups.items():
        if sector in skip_sectors:
            continue
        cluster_articles = [a for a in news_items if a.get("id") in news_ids]
        add_candidate(f"{sector} Update", news_ids, cluster_articles)

    # Aggregate by allowed theme: merge news_ids and average sentiment
    by_allowed: Dict[str, Dict[str, Any]] = {}
    for allowed_name, news_ids, sentiment, mentioned_stocks, confidence in candidates:
        if allowed_name not in by_allowed:
            by_allowed[allowed_name] = {
                "theme_name": allowed_name,
                "news_ids": [],
                "mentioned_stocks": set(),
                "sentiment_scores": [],
                "confidence_sum": 0.0,
                "count": 0,
            }
        rec = by_allowed[allowed_name]
        rec["news_ids"] = list(set(rec["news_ids"]) | set(news_ids))
        rec["mentioned_stocks"] |= mentioned_stocks
        rec["sentiment_scores"].append(sentiment)
        rec["confidence_sum"] += confidence
        rec["count"] += 1

    themes = []
    for allowed_name, rec in by_allowed.items():
        news_ids = rec["news_ids"]
        sentiment_scores = rec["sentiment_scores"]
        bullish = sum(1 for s in sentiment_scores if s == "bullish")
        bearish = sum(1 for s in sentiment_scores if s == "bearish")
        if bullish > bearish:
            sentiment = "bullish"
        elif bearish > bullish:
            sentiment = "bearish"
        else:
            sentiment = "neutral"
        confidence = min(rec["confidence_sum"] / max(rec["count"], 1), 0.95)
        themes.append({
            "theme_name": allowed_name,
            "news_ids": news_ids,
            "confidence": round(confidence, 2),
            "mentioned_stocks": list(rec["mentioned_stocks"]),
            "sentiment": sentiment,
            "avg_sentiment_score": 0.0,
        })

    # Sort by number of news items (most populated first), return max 5
    themes = sorted(themes, key=lambda x: len(x["news_ids"]), reverse=True)
    return themes[:MAX_THEMED_NEWS_ITEMS]


# =============================================================================
# Combined Service Function - Main Entry Point
# =============================================================================


def _get_phase_specific_indices(phase: str) -> List[str]:
    """
    Get the list of indices to show based on market phase.

    Args:
        phase: Market phase ('pre', 'mid', 'post')

    Returns:
        List of index tickers to include in response
    """
    if phase == "pre":
        return PRE_MARKET_INDICES
    elif phase == "post":
        return POST_MARKET_INDICES
    else:  # mid-market
        return MID_MARKET_INDICES


def _filter_indices_by_phase(
    all_indices: Dict[str, Dict[str, Any]],
    phase: str,
) -> Dict[str, Dict[str, Any]]:
    """
    Filter indices data based on market phase.

    Args:
        all_indices: Dictionary of all fetched indices
        phase: Market phase ('pre', 'mid', 'post')

    Returns:
        Filtered dictionary containing only phase-appropriate indices
    """
    phase_indices = _get_phase_specific_indices(phase)

    # Normalize phase indices for comparison
    normalized_phase_indices = {idx.upper(): idx for idx in phase_indices}

    filtered = {}
    for ticker, data in all_indices.items():
        ticker_upper = ticker.upper()
        # Check if this ticker should be included for this phase
        if ticker_upper in normalized_phase_indices:
            filtered[ticker] = data
        # Also check by name match
        elif data.get("name", "").upper() in normalized_phase_indices:
            filtered[ticker] = data

    return filtered


async def fetch_market_intelligence(
    indices: Optional[List[str]] = None,
    watchlist: Optional[List[str]] = None,
    time_window_hours: int = 24,
    max_articles: int = 50,
) -> Dict[str, Any]:
    """
    Fetch all market intelligence data in one call.

    This is the main service function that combines:
    - Market indices data (from CMOTS World Indices API)
    - Market phase
    - Market news
    - Stock-specific news (if watchlist provided)
    - News clustering

    The indices returned are filtered based on market phase:
    - Pre-market: GIFT NIFTY, Nikkei, FTSE 100, Shanghai Composite, DAX
    - Post-market: SENSEX, NIFTY, Shanghai Composite, Nikkei, FTSE 100, DJIA, S&P 500
    - Mid-market: SENSEX, NIFTY

    Args:
        indices: Optional list of index tickers to filter (None = phase-based filtering)
        watchlist: Optional user watchlist for stock-specific news
        time_window_hours: News time window
        max_articles: Maximum news articles

    Returns:
        Combined market intelligence data
    """
    # Fetch market phase first (needed for phase-based index filtering)
    phase_data = await get_market_phase()
    market_phase = phase_data["phase"]

    # Fetch all indices data (from CMOTS World Indices API)
    all_indices_data = await fetch_market_indices(indices)

    # Filter indices based on market phase
    # If specific indices were requested, use those; otherwise filter by phase
    if indices:
        # User requested specific indices, respect that
        indices_data = all_indices_data
    else:
        # Apply phase-based filtering
        indices_data = _filter_indices_by_phase(all_indices_data, market_phase)

    logger.info(
        "indices_filtered_by_phase",
        phase=market_phase,
        total_fetched=len(all_indices_data),
        phase_filtered=len(indices_data),
        included_indices=list(indices_data.keys()),
    )

    # Calculate momentum (use all data for accurate calculation)
    momentum_data = await calculate_index_momentum(all_indices_data)

    # Fetch general market news
    news = await fetch_market_news(time_window_hours, max_articles)

    # Fetch stock-specific news if watchlist provided
    if watchlist:
        stock_news = await fetch_stock_specific_news(
            watchlist, time_window_hours, max_articles=20
        )
        # Merge without duplicates
        existing_ids = {n["id"] for n in news}
        for article in stock_news:
            if article["id"] not in existing_ids:
                news.append(article)

    # Cluster news into themes
    themes = await cluster_news_by_topic(news)

    return {
        "market_phase": phase_data,
        "indices_data": indices_data,
        "momentum": momentum_data,
        "news": news,
        "themes": themes,
        "timestamp": datetime.now(IST).isoformat(),
    }


def get_market_intelligence_tools() -> List[Any]:
    """
    Get tool definitions for market intelligence functions.

    Returns:
        List of tool definitions (currently empty as we use direct function calls)
    """
    return []


def get_market_intelligence_tool_handlers() -> Dict[str, Callable]:
    """
    Get mapping of tool names to handler functions.

    Returns:
        Dictionary mapping function names to async handlers
    """
    return {
        "fetch_market_intelligence": fetch_market_intelligence,
        "fetch_market_indices": fetch_market_indices,
        "fetch_all_world_indices": fetch_all_world_indices,
        "get_market_phase": get_market_phase,
        "fetch_market_news": fetch_market_news,
        "cluster_news_by_topic": cluster_news_by_topic,
        "calculate_index_momentum": calculate_index_momentum,
        "fetch_stock_specific_news": fetch_stock_specific_news,
    }