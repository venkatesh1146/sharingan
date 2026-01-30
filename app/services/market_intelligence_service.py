"""
Market Intelligence Service - Unified service for market data and news fetching.

This service consolidates all data fetching operations:
- Market index data
- Market phase determination
- News fetching (general and stock-specific)
- News clustering
"""

from datetime import datetime, timedelta, time
from typing import Any, Callable, Dict, List, Optional
import hashlib
import pytz

from app.config import get_settings


# Indian Standard Time
IST = pytz.timezone("Asia/Kolkata")


# =============================================================================
# Mock Data (Replace with actual API calls in production)
# =============================================================================


MOCK_INDICES_DATA = {
    "NIFTY 50": {
        "ticker": "NIFTY 50",
        "name": "NIFTY 50 Index",
        "current_price": 22150.50,
        "change_percent": 0.85,
        "change_absolute": 186.75,
        "previous_close": 21963.75,
        "intraday_high": 22200.00,
        "intraday_low": 21950.00,
        "volume": 250000000,
    },
    "SENSEX": {
        "ticker": "SENSEX",
        "name": "S&P BSE SENSEX",
        "current_price": 72850.25,
        "change_percent": 0.72,
        "change_absolute": 520.50,
        "previous_close": 72329.75,
        "intraday_high": 72950.00,
        "intraday_low": 72200.00,
        "volume": 180000000,
    },
    "BANK NIFTY": {
        "ticker": "BANK NIFTY",
        "name": "NIFTY Bank Index",
        "current_price": 47250.75,
        "change_percent": 1.15,
        "change_absolute": 538.25,
        "previous_close": 46712.50,
        "intraday_high": 47400.00,
        "intraday_low": 46600.00,
        "volume": 120000000,
    },
    "NIFTY IT": {
        "ticker": "NIFTY IT",
        "name": "NIFTY IT Index",
        "current_price": 35500.00,
        "change_percent": -0.45,
        "change_absolute": -160.50,
        "previous_close": 35660.50,
        "intraday_high": 35750.00,
        "intraday_low": 35400.00,
        "volume": 80000000,
    },
    "NIFTY MIDCAP": {
        "ticker": "NIFTY MIDCAP",
        "name": "NIFTY Midcap 100",
        "current_price": 45800.00,
        "change_percent": 0.95,
        "change_absolute": 432.25,
        "previous_close": 45367.75,
        "intraday_high": 45900.00,
        "intraday_low": 45200.00,
        "volume": 95000000,
    },
}


MOCK_NEWS = [
    {
        "id": "news_001",
        "headline": "IT stocks rally as US tech earnings beat expectations",
        "summary": "Indian IT stocks including TCS, Infosys, and Wipro surged after major US tech companies reported better-than-expected quarterly earnings, boosting sentiment for the sector.",
        "source": "Economic Times",
        "url": "https://example.com/news/001",
        "published_at": (datetime.now(IST) - timedelta(hours=2)).isoformat(),
        "sentiment": "bullish",
        "sentiment_score": 0.75,
        "mentioned_stocks": ["TCS", "INFY", "WIPRO"],
        "mentioned_sectors": ["IT", "Technology"],
        "is_breaking": False,
    },
    {
        "id": "news_002",
        "headline": "RBI maintains repo rate, signals accommodative stance",
        "summary": "The Reserve Bank of India kept the repo rate unchanged at 6.5% and maintained an accommodative policy stance, providing relief to rate-sensitive sectors like banking and real estate.",
        "source": "Mint",
        "url": "https://example.com/news/002",
        "published_at": (datetime.now(IST) - timedelta(hours=4)).isoformat(),
        "sentiment": "bullish",
        "sentiment_score": 0.65,
        "mentioned_stocks": ["HDFC", "ICICI", "SBI", "DLF"],
        "mentioned_sectors": ["Banking", "Finance", "Real Estate"],
        "is_breaking": True,
    },
    {
        "id": "news_003",
        "headline": "Crude oil prices surge to $85 on OPEC+ supply concerns",
        "summary": "Brent crude oil prices jumped to $85 per barrel amid concerns about OPEC+ production cuts, potentially impacting oil marketing companies and increasing input costs for paint and chemical manufacturers.",
        "source": "Reuters",
        "url": "https://example.com/news/003",
        "published_at": (datetime.now(IST) - timedelta(hours=6)).isoformat(),
        "sentiment": "bearish",
        "sentiment_score": -0.55,
        "mentioned_stocks": ["IOCL", "BPCL", "HINDPETRO", "ASIANPAINT"],
        "mentioned_sectors": ["Oil & Gas", "Energy", "Paints"],
        "is_breaking": False,
    },
    {
        "id": "news_004",
        "headline": "Auto sales hit record high in January, driven by SUV demand",
        "summary": "Automobile manufacturers reported record sales in January 2026, led by strong SUV and EV demand. Maruti, Tata Motors, and M&M emerged as top performers.",
        "source": "Business Standard",
        "url": "https://example.com/news/004",
        "published_at": (datetime.now(IST) - timedelta(hours=8)).isoformat(),
        "sentiment": "bullish",
        "sentiment_score": 0.70,
        "mentioned_stocks": ["MARUTI", "TATAMOTORS", "M&M"],
        "mentioned_sectors": ["Auto", "Consumer"],
        "is_breaking": False,
    },
    {
        "id": "news_005",
        "headline": "FIIs turn net buyers after 5 months of selling",
        "summary": "Foreign Institutional Investors (FIIs) turned net buyers in Indian markets, pumping in Rs 5,000 crore in the last week after five consecutive months of selling.",
        "source": "Moneycontrol",
        "url": "https://example.com/news/005",
        "published_at": (datetime.now(IST) - timedelta(hours=3)).isoformat(),
        "sentiment": "bullish",
        "sentiment_score": 0.80,
        "mentioned_stocks": [],
        "mentioned_sectors": ["Market"],
        "is_breaking": True,
    },
    {
        "id": "news_006",
        "headline": "Pharma sector faces headwinds from US FDA observations",
        "summary": "Several Indian pharmaceutical companies including Sun Pharma and Dr. Reddy's received warning letters from US FDA, raising concerns about their export revenue.",
        "source": "Economic Times",
        "url": "https://example.com/news/006",
        "published_at": (datetime.now(IST) - timedelta(hours=10)).isoformat(),
        "sentiment": "bearish",
        "sentiment_score": -0.60,
        "mentioned_stocks": ["SUNPHARMA", "DRREDDY", "CIPLA"],
        "mentioned_sectors": ["Pharma", "Healthcare"],
        "is_breaking": False,
    },
    {
        "id": "news_007",
        "headline": "Steel prices rise 5% as China reduces production",
        "summary": "Domestic steel prices increased by 5% following China's announcement to cut steel production, benefiting Indian steel manufacturers like Tata Steel and JSW Steel.",
        "source": "Financial Express",
        "url": "https://example.com/news/007",
        "published_at": (datetime.now(IST) - timedelta(hours=5)).isoformat(),
        "sentiment": "bullish",
        "sentiment_score": 0.55,
        "mentioned_stocks": ["TATASTEEL", "JSWSTEEL", "SAIL"],
        "mentioned_sectors": ["Metals", "Steel"],
        "is_breaking": False,
    },
    {
        "id": "news_008",
        "headline": "Government announces new PLI scheme for electronics manufacturing",
        "summary": "The government unveiled a new Production Linked Incentive (PLI) scheme worth Rs 20,000 crore for electronics manufacturing, boosting companies like Dixon Technologies and Amber Enterprises.",
        "source": "Hindu Business Line",
        "url": "https://example.com/news/008",
        "published_at": (datetime.now(IST) - timedelta(hours=7)).isoformat(),
        "sentiment": "bullish",
        "sentiment_score": 0.85,
        "mentioned_stocks": ["DIXON", "AMBER"],
        "mentioned_sectors": ["Electronics", "Manufacturing"],
        "is_breaking": True,
    },
]


# =============================================================================
# Market Data Functions
# =============================================================================


async def fetch_market_indices(
    indices: List[str],
) -> Dict[str, Dict[str, Any]]:
    """
    Fetch current market data for specified indices.
    
    Args:
        indices: List of index tickers to fetch
    
    Returns:
        Dictionary mapping ticker to index data
    """
    result = {}
    timestamp = datetime.now(IST)

    for ticker in indices:
        if ticker in MOCK_INDICES_DATA:
            data = MOCK_INDICES_DATA[ticker].copy()
            data["timestamp"] = timestamp.isoformat()
            result[ticker] = data
        else:
            result[ticker] = {
                "ticker": ticker,
                "name": ticker,
                "current_price": 0.0,
                "change_percent": 0.0,
                "change_absolute": 0.0,
                "previous_close": 0.0,
                "intraday_high": 0.0,
                "intraday_low": 0.0,
                "volume": 0,
                "timestamp": timestamp.isoformat(),
                "error": f"Index {ticker} not found",
            }

    return result


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
    
    Args:
        indices_data: Dictionary of index data
    
    Returns:
        Momentum analysis results
    """
    nifty_data = indices_data.get("NIFTY 50", {})
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

    return {
        "momentum": momentum,
        "description": description,
        "primary_index": "NIFTY 50",
        "primary_change_percent": nifty_change,
        "market_breadth": breadth,
        "advancing_indices": positive_count,
        "declining_indices": total_count - positive_count,
    }


# =============================================================================
# News Functions
# =============================================================================


async def fetch_market_news(
    time_window_hours: int = 24,
    max_articles: int = 50,
    categories: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """
    Fetch general market news articles.
    
    Args:
        time_window_hours: How far back to fetch news (in hours)
        max_articles: Maximum number of articles to return
        categories: Optional list of categories to filter
    
    Returns:
        List of news article dictionaries
    """
    cutoff_time = datetime.now(IST) - timedelta(hours=time_window_hours)

    news = []
    for article in MOCK_NEWS[:max_articles]:
        pub_time = datetime.fromisoformat(article["published_at"])
        if pub_time >= cutoff_time:
            article_copy = article.copy()
            article_copy["relevance_score"] = 0.5 + (0.5 * article.get("sentiment_score", 0))
            news.append(article_copy)

    return news


async def fetch_stock_specific_news(
    tickers: List[str],
    time_window_hours: int = 24,
    max_articles: int = 20,
) -> List[Dict[str, Any]]:
    """
    Fetch news specifically about given stocks.
    
    Args:
        tickers: List of stock tickers to search for
        time_window_hours: How far back to fetch news
        max_articles: Maximum articles to return
    
    Returns:
        List of news articles mentioning the specified stocks
    """
    tickers_set = set(t.upper() for t in tickers)
    cutoff_time = datetime.now(IST) - timedelta(hours=time_window_hours)

    matching_news = []
    for article in MOCK_NEWS:
        mentioned = set(s.upper() for s in article.get("mentioned_stocks", []))
        if mentioned & tickers_set:
            pub_time = datetime.fromisoformat(article["published_at"])
            if pub_time >= cutoff_time:
                article_copy = article.copy()
                article_copy["matched_tickers"] = list(mentioned & tickers_set)
                matching_news.append(article_copy)

    return matching_news[:max_articles]


async def cluster_news_by_topic(
    news_items: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Cluster news items into themes/topics.
    
    Args:
        news_items: List of news articles to cluster
    
    Returns:
        List of theme clusters
    """
    sector_groups: Dict[str, List[str]] = {}
    
    for article in news_items:
        sectors = article.get("mentioned_sectors", ["General"])
        primary_sector = sectors[0] if sectors else "General"
        
        if primary_sector not in sector_groups:
            sector_groups[primary_sector] = []
        sector_groups[primary_sector].append(article["id"])

    themes = []
    for sector, news_ids in sector_groups.items():
        cluster_articles = [a for a in news_items if a["id"] in news_ids]
        avg_sentiment = sum(a.get("sentiment_score", 0) for a in cluster_articles) / len(cluster_articles)
        
        if avg_sentiment > 0.2:
            sentiment = "bullish"
        elif avg_sentiment < -0.2:
            sentiment = "bearish"
        else:
            sentiment = "neutral"

        mentioned_stocks = set()
        for article in cluster_articles:
            mentioned_stocks.update(article.get("mentioned_stocks", []))

        themes.append({
            "theme_name": f"{sector} Update",
            "news_ids": news_ids,
            "confidence": 0.7 + (0.1 * len(news_ids)),
            "mentioned_stocks": list(mentioned_stocks),
            "sentiment": sentiment,
            "avg_sentiment_score": round(avg_sentiment, 2),
        })

    return sorted(themes, key=lambda x: len(x["news_ids"]), reverse=True)


# =============================================================================
# Combined Service Function - Main Entry Point
# =============================================================================


async def fetch_market_intelligence(
    indices: List[str],
    watchlist: Optional[List[str]] = None,
    time_window_hours: int = 24,
    max_articles: int = 50,
) -> Dict[str, Any]:
    """
    Fetch all market intelligence data in one call.
    
    This is the main service function that combines:
    - Market indices data
    - Market phase
    - Market news
    - Stock-specific news (if watchlist provided)
    - News clustering
    
    Args:
        indices: List of index tickers to fetch
        watchlist: Optional user watchlist for stock-specific news
        time_window_hours: News time window
        max_articles: Maximum news articles
    
    Returns:
        Combined market intelligence data
    """
    # Fetch market phase
    phase_data = await get_market_phase()
    
    # Fetch indices data
    indices_data = await fetch_market_indices(indices)
    
    # Calculate momentum
    momentum_data = await calculate_index_momentum(indices_data)
    
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
        "get_market_phase": get_market_phase,
        "fetch_market_news": fetch_market_news,
        "cluster_news_by_topic": cluster_news_by_topic,
        "calculate_index_momentum": calculate_index_momentum,
        "fetch_stock_specific_news": fetch_stock_specific_news,
    }
