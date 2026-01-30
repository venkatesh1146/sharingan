"""
Market News Service - Unified service for fetching news from multiple sources.

Fetches economy, other markets, and foreign markets news from the CMOTS API
and combines them into a single unified response with standardized pagination.
"""

from typing import Any, Dict, List, Optional
from datetime import datetime

from app.utils.funds_api_client import call_funds_proxy
from app.utils.logging import get_logger
from app.utils.exceptions import DataFetchError
from app.utils.pagination import create_paginated_response

logger = get_logger(__name__)


# News type categories
NEWS_TYPES = {
    "economy-news": "Economy",
    "other-markets": "Other Markets",
    "foreign-markets": "Foreign Markets",
}


async def fetch_unified_market_news(
    limit: int = 10,
    page: int = 1,
    per_page: int = 10,
    news_type: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Fetch unified market news from all sources with pagination.

    Calls 3 endpoints:
    - Economy News
    - Other Markets News
    - Foreign Markets News

    Combines all results with a news_type field and applies standardized pagination.

    Args:
        limit: Number of items per news type to fetch from API (default: 10)
        page: Page number for pagination (1-indexed, default: 1)
        per_page: Items per page (1-100, default: 10)
        news_type: Filter by specific news type (optional)

    Returns:
        Dictionary with paginated data and pagination metadata
    """
    all_news = []
    errors = []

    # Fetch from all 3 sources
    news_sources = [
        ("economy-news", f"https://wealthyapis.cmots.com/api/CapitalMarketLiveNews/economy-news/{limit}"),
        ("other-markets", f"https://wealthyapis.cmots.com/api/CapitalMarketLiveNews/other-markets/{limit}"),
        ("foreign-markets", f"https://wealthyapis.cmots.com/api/CapitalMarketLiveNews/foreign-markets/{limit}"),
    ]

    # Filter news sources if a specific type is requested
    if news_type:
        news_sources = [(t, u) for t, u in news_sources if t == news_type]

    for news_type_key, url in news_sources:
        try:
            logger.info("fetching_news", news_type=news_type_key, url=url)

            response = await call_funds_proxy(
                method="get",
                url=url,
                payload=None,
            )

            # Extract news items from response
            items = response.get("data", [])
            logger.info("fetched_news_count", news_type=news_type_key, count=len(items))

            # Add news_type to each item
            for item in items:
                enriched_item = {
                    **item,
                    "news_type": NEWS_TYPES.get(news_type_key, news_type_key),
                    "raw_news_type": news_type_key,
                }
                all_news.append(enriched_item)

        except DataFetchError as e:
            logger.error(
                "news_fetch_error",
                news_type=news_type_key,
                error=str(e),
            )
            errors.append({
                "news_type": news_type_key,
                "error": str(e),
            })
        except Exception as e:
            logger.error(
                "unexpected_news_error",
                news_type=news_type_key,
                error=str(e),
            )
            errors.append({
                "news_type": news_type_key,
                "error": f"Unexpected error: {str(e)}",
            })

    # Sort by date and time (most recent first)
    # Note: You may need to adjust this based on actual date format
    try:
        all_news.sort(
            key=lambda x: (
                x.get("date", ""),
                x.get("time", ""),
            ),
            reverse=True,
        )
    except Exception as e:
        logger.warning("news_sorting_failed", error=str(e))

    # Organize news by type
    data_by_type = {}
    for news_type_key in NEWS_TYPES.keys():
        data_by_type[news_type_key] = [
            n for n in all_news if n.get("raw_news_type") == news_type_key
        ]

    # Calculate pagination for combined data
    total_items = len(all_news)
    total_pages = (total_items + per_page - 1) // per_page if per_page > 0 else 1
    start_idx = (page - 1) * per_page
    end_idx = start_idx + per_page
    paginated_data = all_news[start_idx:end_idx]

    return {
        "data": data_by_type,
        "pagination": {
            "page": page,
            "per_page": per_page,
            "total_items": total_items,
            "total_pages": total_pages,
            "has_next": page < total_pages,
            "has_prev": page > 1,
            "next_page": page + 1 if page < total_pages else None,
            "prev_page": page - 1 if page > 1 else None,
        },
        "errors": errors if errors else None,
        "fetched_at": datetime.utcnow().isoformat(),
    }


async def fetch_news_by_type(
    news_type: str,
    limit: int = 10,
    page: int = 1,
    per_page: int = 10,
) -> Dict[str, Any]:
    """
    Fetch news for a specific type with pagination.

    Args:
        news_type: "economy-news", "other-markets", or "foreign-markets"
        limit: Number of items to fetch from API
        page: Page number for pagination (1-indexed)
        per_page: Items per page (1-100)

    Returns:
        Dictionary with paginated news items and pagination metadata
    """
    if news_type not in NEWS_TYPES:
        raise ValueError(
            f"Invalid news_type. Must be one of: {', '.join(NEWS_TYPES.keys())}"
        )

    url = f"https://wealthyapis.cmots.com/api/CapitalMarketLiveNews/{news_type}/{limit}"

    try:
        logger.info("fetching_specific_news", news_type=news_type, url=url)

        response = await call_funds_proxy(
            method="get",
            url=url,
            payload=None,
        )

        items = response.get("data", [])

        # Add news_type to each item
        for item in items:
            item["news_type"] = NEWS_TYPES.get(news_type, news_type)
            item["raw_news_type"] = news_type

        # Apply pagination
        return create_paginated_response(
            items=items,
            page=page,
            per_page=per_page,
            news_type=NEWS_TYPES.get(news_type, news_type),
            raw_news_type=news_type,
            fetched_at=datetime.utcnow().isoformat(),
        )

    except Exception as e:
        logger.error(
            "fetch_news_by_type_error",
            news_type=news_type,
            error=str(e),
        )
        raise DataFetchError(
            source=f"cmots_api/{news_type}",
            message=str(e),
        ) from e
