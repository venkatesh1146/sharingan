"""
Market News Service - Unified service for fetching news from multiple sources.

Fetches economy, other markets, and foreign markets news from the CMOTS API
and combines them into a single unified response with standardized pagination.
"""

from typing import Any, Dict, List, Optional
from datetime import datetime
import re
import asyncio

from app.utils.funds_api_client import call_funds_proxy
from app.utils.logging import get_logger
from app.utils.exceptions import DataFetchError

logger = get_logger(__name__)


# News type categories
NEWS_TYPES = {
    "economy-news": "Economy",
    "other-markets": "Other Markets",
    "foreign-markets": "Foreign Markets",
}


def _clean_html(text: str) -> str:
    if not text:
        return ""
    cleaned = re.sub(r"<[^>]+>", " ", text)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def _build_published_at(date_str: Optional[str], time_str: Optional[str]) -> Optional[str]:
    if not date_str and not time_str:
        return None
    date_str = date_str or ""
    time_str = time_str or ""
    try:
        base_dt = datetime.strptime(date_str, "%m/%d/%Y %I:%M:%S %p")
        if time_str:
            time_dt = datetime.strptime(time_str, "%H:%M")
            combined = base_dt.replace(
                hour=time_dt.hour,
                minute=time_dt.minute,
                second=0,
                microsecond=0,
            )
            return combined.isoformat()
        return base_dt.isoformat()
    except Exception:
        combined = f"{date_str} {time_str}".strip()
        return combined or None


async def _fetch_news_details(sno: Optional[str]) -> Optional[str]:
    if not sno:
        return None
    url = f"https://wealthyapis.cmots.com/api/NewsDetails/{sno}"
    try:
        response = await call_funds_proxy(
            method="get",
            url=url,
            payload=None,
        )
        items = response.get("data", [])
        if not items:
            return None
        detail = items[0]
        summary = _clean_html(detail.get("arttext", ""))
        return summary
    except Exception as e:
        logger.warning("news_details_failed", sno=sno, error=str(e))
        return None


async def _enrich_item(
    item: Dict[str, Any],
    news_type_key: str,
    semaphore: asyncio.Semaphore,
) -> Dict[str, Any]:
    async with semaphore:
        published_at = _build_published_at(item.get("date"), item.get("time"))
        summary = await _fetch_news_details(item.get("sno"))
    enriched_item = {
        **item,
        "news_type": NEWS_TYPES.get(news_type_key, news_type_key),
        "published_at": published_at,
        "summary": summary,
        "_type_key": news_type_key,
    }
    enriched_item.pop("date", None)
    enriched_item.pop("time", None)
    return enriched_item


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

    semaphore = asyncio.Semaphore(10)

    async def _fetch_source(news_type_key: str, url: str) -> None:
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

            # Enrich items concurrently (details fetch)
            tasks = [
                _enrich_item(item, news_type_key, semaphore)
                for item in items
            ]
            enriched_items = await asyncio.gather(*tasks)
            all_news.extend(enriched_items)

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

    await asyncio.gather(*[_fetch_source(t, u) for t, u in news_sources])

    # Sort by published_at (most recent first)
    try:
        all_news.sort(
            key=lambda x: (
                x.get("published_at", ""),
            ),
            reverse=True,
        )
    except Exception as e:
        logger.warning("news_sorting_failed", error=str(e))

    # Organize news by type
    data_by_type = {}
    for news_type_key in NEWS_TYPES.keys():
        data_by_type[news_type_key] = [
            n for n in all_news if n.get("_type_key") == news_type_key
        ]

    # Remove internal grouping key from output items
    for items in data_by_type.values():
        for item in items:
            item.pop("_type_key", None)

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

        # Enrich items concurrently (details fetch)
        semaphore = asyncio.Semaphore(10)
        tasks = [
            _enrich_item(item, news_type, semaphore)
            for item in items
        ]
        items = await asyncio.gather(*tasks)

        # Apply pagination
        for item in items:
            item.pop("_type_key", None)

        return {
            "data": items,
            "pagination": {
                "page": page,
                "per_page": per_page,
                "total_items": len(items),
                "total_pages": (len(items) + per_page - 1) // per_page if per_page > 0 else 1,
                "has_next": page < ((len(items) + per_page - 1) // per_page if per_page > 0 else 1),
                "has_prev": page > 1,
                "next_page": page + 1 if page < ((len(items) + per_page - 1) // per_page if per_page > 0 else 1) else None,
                "prev_page": page - 1 if page > 1 else None,
            },
            "news_type": NEWS_TYPES.get(news_type, news_type),
            "fetched_at": datetime.utcnow().isoformat(),
        }

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
