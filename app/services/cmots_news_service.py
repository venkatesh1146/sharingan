"""
Market News Service - Unified service for fetching news from multiple sources.

Fetches economy, other markets, and foreign markets news from the CMOTS API
and combines them into a single unified response with standardized pagination.
Includes Redis caching for improved performance and reduced API calls.
All API calls and caching logic are handled within this service.
"""

from typing import Any, Dict, List, Optional
from datetime import datetime
import re
import asyncio

import httpx

from app.utils.logging import get_logger
from app.utils.exceptions import DataFetchError
from app.config import get_settings

logger = get_logger(__name__)


# News type categories. Session news uses market_phase enum: pre, mid, post.
NEWS_TYPES = {
    "economy-news": "Economy",
    "other-markets": "Other Markets",
    "foreign-markets": "Foreign Markets",
    "pre": "Pre-Session",
    "mid": "Mid-Session",
    "post": "End-Session",
}


class CMOTSNewsService:
    """Service for fetching and caching market news from CMOTS API with integrated proxy calls."""

    def __init__(self):
        """Initialize CMOTS news service with settings."""
        self.settings = get_settings()
        self.news_types = NEWS_TYPES

    def _build_proxy_headers(self) -> Dict[str, str]:
        """Build headers for proxy API request."""
        return {
            "X-TOKEN": self.settings.FUNDS_API_X_TOKEN,
            "Content-Type": "application/json",
        }

    def _build_target_headers(
        self,
        headers: Optional[Dict[str, str]] = None,
    ) -> Dict[str, str]:
        """
        Build headers for target API including CMOTS authorization.

        Args:
            headers: Optional additional headers

        Returns:
            Merged headers with Authorization
        """
        merged = dict(headers or {})
        if self.settings.CMOTS_TOKEN and "Authorization" not in merged:
            merged["Authorization"] = f'Bearer {self.settings.CMOTS_TOKEN}'
        return merged

    async def _call_proxy_api(
        self,
        method: str,
        url: str,
        payload: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
        timeout_seconds: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Call the Wealthy proxy API to reach CMOTS endpoints.

        Args:
            method: HTTP method to use for the upstream call (e.g., "get")
            url: Upstream URL to call via the proxy
            payload: Optional payload to send to upstream
            headers: Optional upstream headers
            timeout_seconds: Override timeout for this call

        Returns:
            Parsed JSON response from the proxy

        Raises:
            DataFetchError: If proxy URL/token not set or request fails
        """
        proxy_url = self.settings.FUNDS_API_PROXY_URL

        if not proxy_url:
            raise DataFetchError("funds_proxy", "FUNDS_API_PROXY_URL is not set")
        if not self.settings.FUNDS_API_X_TOKEN:
            raise DataFetchError("funds_proxy", "FUNDS_API_X_TOKEN is not set")

        request_body = {
            "method": method,
            "url": url,
            "payload": payload,
            "headers": self._build_target_headers(headers),
        }

        timeout = timeout_seconds or self.settings.FUNDS_API_TIMEOUT_SECONDS
        print(request_body)
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await client.post(
                    proxy_url,
                    headers=self._build_proxy_headers(),
                    json=request_body,
                )
                response.raise_for_status()
                return response.json()
        except httpx.HTTPStatusError as exc:
            logger.error(
                "funds_proxy_http_error",
                status_code=exc.response.status_code,
                response_text=exc.response.text,
            )
            raise DataFetchError(
                source="funds_proxy",
                message="HTTP error from proxy",
                status_code=exc.response.status_code,
            ) from exc
        except httpx.RequestError as exc:
            logger.error("funds_proxy_request_error", error=str(exc))
            raise DataFetchError(
                source="funds_proxy",
                message="Request error while calling proxy",
            ) from exc

    @staticmethod
    def _extract_items(response: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Extract a list of items from varied CMOTS response shapes.

        Returns:
            List of items or an empty list if none found
        """
        if not isinstance(response, dict):
            return []
        items = response.get("data") or response.get("Data")
        if isinstance(items, dict):
            items = items.get("data") or items.get("Data")
        if not isinstance(items, list):
            return []
        return items

    @staticmethod
    def _clean_html(text: str) -> str:
        """Remove HTML tags from text using regex."""
        if not text:
            return ""
        cleaned = re.sub(r"<[^>]+>", " ", text)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        return cleaned

    @staticmethod
    def _build_published_at(date_str: Optional[str], time_str: Optional[str]) -> Optional[str]:
        """
        Build ISO format published_at timestamp from date and time strings.

        Args:
            date_str: Date string in format "m/d/Y I:M:S p"
            time_str: Time string in format "H:M"

        Returns:
            ISO format datetime string or None
        """
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

    async def _fetch_news_details(self, sno: Optional[str]) -> Optional[str]:
        """
        Fetch detailed news content from CMOTS NewsDetails endpoint.

        Args:
            sno: Serial number identifier for the news item

        Returns:
            Cleaned HTML summary text or None
        """
        if not sno:
            return None
        url = f"https://wealthyapis.cmots.com/api/NewsDetails/{sno}"
        try:
            response = await self._call_proxy_api(
                method="get",
                url=url,
                payload=None,
            )
            items = response.get("data", [])
            if not items:
                return None
            detail = items[0]
            summary = self._clean_html(detail.get("arttext", ""))
            return summary
        except Exception as e:
            logger.warning("news_details_failed", sno=sno, error=str(e))
            return None

    async def _enrich_item(
        self,
        item: Dict[str, Any],
        news_type_key: str,
        semaphore: asyncio.Semaphore,
    ) -> Dict[str, Any]:
        """
        Enrich a news item with publication timestamp and detailed summary.

        Args:
            item: Base news item from API
            news_type_key: Category key (economy-news, other-markets, foreign-markets)
            semaphore: Asyncio semaphore for concurrency limiting

        Returns:
            Enriched news item with summary and published_at
        """
        async with semaphore:
            published_at = self._build_published_at(item.get("date"), item.get("time"))
            summary = await self._fetch_news_details(item.get("sno"))

        enriched_item = {
            **item,
            "news_type": self.news_types.get(news_type_key, news_type_key),
            "published_at": published_at,
            "summary": summary,
            "_type_key": news_type_key,
        }
        enriched_item.pop("date", None)
        enriched_item.pop("time", None)
        return enriched_item

    async def _fetch_source(
        self,
        news_type_key: str,
        url: str,
        semaphore: asyncio.Semaphore,
        all_news: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Fetch news from a single source endpoint and enrich items.

        Args:
            news_type_key: Category key for this news type
            url: API endpoint URL
            semaphore: Asyncio semaphore for detail fetching concurrency
            all_news: List to accumulate news items

        Returns:
            List of enriched items from this source
        """
        source_items = []
        try:
            logger.info("fetching_news", news_type=news_type_key, url=url)

            response = await self._call_proxy_api(
                method="get",
                url=url,
                payload=None,
            )

            # Extract news items from response
            items = self._extract_items(response)
            logger.info("fetched_news_count", news_type=news_type_key, count=len(items))

            # Enrich items concurrently (details fetch)
            tasks = [
                self._enrich_item(item, news_type_key, semaphore)
                for item in items
            ]
            enriched_items = await asyncio.gather(*tasks)
            source_items.extend(enriched_items)
            all_news.extend(enriched_items)

        except DataFetchError as e:
            logger.error(
                "news_fetch_error",
                news_type=news_type_key,
                error=str(e),
            )
        except Exception as e:
            logger.error(
                "unexpected_news_error",
                news_type=news_type_key,
                error=str(e),
            )

        return source_items

    async def fetch_unified_market_news(
        self,
        limit: int = 10,
        page: int = 1,
        per_page: int = 10,
        market_phase: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Fetch unified market news from all sources with pagination.

        Calls 6 endpoints: economy-news, other-markets, foreign-markets, and session news by market_phase (pre, mid, post).

        Args:
            limit: Number of items per source to fetch from API (default: 10)
            page: Page number for pagination (1-indexed, default: 1)
            per_page: Items per page (1-100, default: 10)
            market_phase: Optional filter by market phase (pre/mid/post). If set, only that session is fetched.

        Returns:
            Dictionary with paginated data. Data keys: economy-news, other-markets, foreign-markets, pre, mid, post.
        """
        all_news = []

        # All 6 sources; session sources use market_phase enum (pre, mid, post)
        news_sources = [
            ("economy-news", f"https://wealthyapis.cmots.com/api/CapitalMarketLiveNews/economy-news/{limit}"),
            ("other-markets", f"https://wealthyapis.cmots.com/api/CapitalMarketLiveNews/other-markets/{limit}"),
            ("foreign-markets", f"https://wealthyapis.cmots.com/api/CapitalMarketLiveNews/foreign-markets/{limit}"),
            ("pre", f"https://wealthyapis.cmots.com/api/CapitalMarketLiveNews/pre-session/{limit}"),
            ("mid", f"https://wealthyapis.cmots.com/api/CapitalMarketLiveNews/mid-session/{limit}"),
            ("post", f"https://wealthyapis.cmots.com/api/CapitalMarketLiveNews/end-session/{limit}"),
        ]

        if market_phase:
            news_sources = [(k, u) for k, u in news_sources if k == market_phase]

        semaphore = asyncio.Semaphore(10)

        await asyncio.gather(
            *[
                self._fetch_source(t, u, semaphore, all_news)
                for t, u in news_sources
            ]
        )

        try:
            all_news.sort(
                key=lambda x: (x.get("published_at", "")),
                reverse=True,
            )
        except Exception as e:
            logger.warning("news_sorting_failed", error=str(e))

        return self._apply_pagination_and_filter(
            all_news,
            page,
            per_page,
            market_phase,
        )

    def _apply_pagination_and_filter(
        self,
        all_news: List[Dict[str, Any]],
        page: int,
        per_page: int,
        market_phase: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Apply pagination and optional filter by market_phase.

        Args:
            all_news: Full list of news items
            page: Page number (1-indexed)
            per_page: Items per page
            market_phase: Optional filter by market phase (pre/mid/post) or source key

        Returns:
            Paginated and filtered response dictionary
        """
        if market_phase:
            all_news = [n for n in all_news if n.get("_type_key") == market_phase]

        data_by_type = {}
        for type_key in self.news_types.keys():
            data_by_type[type_key] = [
                n for n in all_news if n.get("_type_key") == type_key
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
            "errors": None,
            "fetched_at": datetime.utcnow().isoformat(),
        }

    async def fetch_news_by_type(
        self,
        news_type: str,
        limit: int = 10,
        page: int = 1,
        per_page: int = 10,
    ) -> Dict[str, Any]:
        """
        Fetch news for a specific type with pagination.

        Args:
            news_type: economy-news, other-markets, foreign-markets, or market_phase (pre, mid, post)
            limit: Number of items to fetch from API
            page: Page number for pagination (1-indexed)
            per_page: Items per page (1-100)

        Returns:
            Dictionary with paginated news items and pagination metadata
        """
        if news_type not in self.news_types:
            raise ValueError(
                f"Invalid news_type. Must be one of: {', '.join(self.news_types.keys())}"
            )

        # Session endpoints by market_phase (pre, mid, post)
        if news_type == "mid":
            url = f"https://wealthyapis.cmots.com/api/CapitalMarketLiveNews/mid-session/{limit}"
        elif news_type == "pre":
            url = f"https://wealthyapis.cmots.com/api/CapitalMarketLiveNews/pre-session/{limit}"
        elif news_type == "post":
            url = f"https://wealthyapis.cmots.com/api/CapitalMarketLiveNews/end-session/{limit}"
        else:
            url = f"https://wealthyapis.cmots.com/api/CapitalMarketLiveNews/{news_type}/{limit}"

        try:
            logger.info("fetching_specific_news", news_type=news_type, url=url)

            response = await self._call_proxy_api(
                method="get",
                url=url,
                payload=None,
            )

            items = self._extract_items(response)

            # Enrich items concurrently (details fetch)
            semaphore = asyncio.Semaphore(10)
            tasks = [
                self._enrich_item(item, news_type, semaphore)
                for item in items
            ]
            items = await asyncio.gather(*tasks)

            # Remove internal grouping key
            for item in items:
                item.pop("_type_key", None)

            return self._apply_type_pagination(items, page, per_page, news_type)

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

    def _apply_type_pagination(
        self,
        items: List[Dict[str, Any]],
        page: int,
        per_page: int,
        news_type: str,
    ) -> Dict[str, Any]:
        """
        Apply pagination to a single news type result set.

        Args:
            items: List of news items
            page: Page number (1-indexed)
            per_page: Items per page
            news_type: The news type category

        Returns:
            Paginated response dictionary
        """
        total_items = len(items)
        total_pages = (total_items + per_page - 1) // per_page if per_page > 0 else 1
        start_idx = (page - 1) * per_page
        end_idx = start_idx + per_page
        paginated_items = items[start_idx:end_idx]

        return {
            "data": paginated_items,
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
            "news_type": self.news_types.get(news_type, news_type),
            "fetched_at": datetime.utcnow().isoformat(),
        }

def get_cmots_news_service() -> CMOTSNewsService:
    """Create a new CMOTS news service instance."""
    return CMOTSNewsService()


async def fetch_world_indices() -> Dict[str, Any]:
    """
    Fetch world indices data from CMOTS API.

    Returns data for global market indices including:
    - Index name and symbol
    - Current value, change, and percentage change
    - Open, high, low values
    - Last updated timestamp

    Returns:
        Dictionary with world indices data and metadata
    """
    url = "https://wealthyapis.cmots.com/api/WorldIndices"

    try:
        logger.info("fetching_world_indices", url=url)

        # Create service instance to call proxy API
        service = CMOTSNewsService()
        response = await service._call_proxy_api(
            method="get",
            url=url,
            payload=None,
        )

        # Extract indices data from response
        indices = response.get("data", [])
        logger.info("fetched_world_indices_count", count=len(indices))

        return {
            "data": indices,
            "total_count": len(indices),
            "fetched_at": datetime.utcnow().isoformat(),
        }

    except DataFetchError as e:
        logger.error(
            "world_indices_fetch_error",
            error=str(e),
        )
        raise
    except Exception as e:
        logger.error(
            "unexpected_world_indices_error",
            error=str(e),
        )
        raise DataFetchError(
            source="cmots_api/world_indices",
            message=str(e),
        ) from e
