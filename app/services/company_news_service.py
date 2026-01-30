"""
Company-wise News Service - Fetches news for specific companies by NSE symbol.

Caches CompanyMaster data and fetches company-specific news from CMOTS API.
Uses concurrency for efficient multi-company news fetching.
"""

from typing import Any, Dict, List, Optional
from datetime import datetime
import asyncio

import httpx

from app.utils.logging import get_logger
from app.utils.exceptions import DataFetchError
from app.services.redis_service import (
    RedisService,
    build_cache_key,
    get_date_identifier,
    get_24hr_ttl,
)
from app.config import get_settings

logger = get_logger(__name__)


class CompanyNewsService:
    """Service for fetching company-wise news with CompanyMaster caching."""

    def __init__(self):
        """Initialize service with settings and Redis client."""
        self.settings = get_settings()
        self.redis_service = RedisService()

    def _build_proxy_headers(self, headers: Optional[Dict[str, str]] = None) -> Dict[str, str]:
        """
        Build headers for proxy API call including X-TOKEN.

        Args:
            headers: Optional additional headers

        Returns:
            Headers dict with X-TOKEN
        """
        merged = dict(headers or {})
        if self.settings.FUNDS_API_X_TOKEN and "X-TOKEN" not in merged:
            merged["X-TOKEN"] = self.settings.FUNDS_API_X_TOKEN
        return merged

    def _build_target_headers(self, headers: Optional[Dict[str, str]] = None) -> Dict[str, str]:
        """
        Build headers for target API including CMOTS authorization.

        Args:
            headers: Optional additional headers

        Returns:
            Merged headers with Authorization
        """
        merged = dict(headers or {})
        if self.settings.CMOTS_TOKEN and "Authorization" not in merged:
            merged["Authorization"] = self.settings.CMOTS_TOKEN
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
            method: HTTP method to use for the upstream call
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

    async def _fetch_company_master(self, use_cache: bool = True) -> Dict[str, Any]:
        """
        Fetch and cache CompanyMaster data.

        Args:
            use_cache: Enable/disable caching (default: True)

        Returns:
            Dictionary with company master data indexed by NSE symbol
        """
        cache_key = build_cache_key(
            "companyMaster",
            get_date_identifier(),
        )

        # Try cache first
        if use_cache and self.settings.ENABLE_CACHING:
            cached = await self.redis_service.get(cache_key)
            if cached:
                logger.info("company_master_cache_hit", cache_key=cache_key)
                return cached

        url = "https://wealthyapis.cmots.com/api/CompanyMaster"

        try:
            logger.info("fetching_company_master", url=url)
            response = await self._call_proxy_api(
                method="get",
                url=url,
                payload=None,
            )

            items = self._extract_items(response)
            logger.info("fetched_company_master_count", count=len(items))

            # Index by NSE symbol for fast lookup
            indexed = {}
            for company in items:
                nsesymbol = company.get("nsesymbol")
                if nsesymbol:
                    indexed[nsesymbol] = company

            # Cache the indexed data
            if self.settings.ENABLE_CACHING:
                ttl = get_24hr_ttl()
                await self.redis_service.set(cache_key, indexed, ttl_seconds=ttl)
                logger.info("company_master_cached", cache_key=cache_key, count=len(indexed))

            return indexed

        except Exception as e:
            logger.error("fetch_company_master_error", error=str(e))
            raise DataFetchError(
                source="company_master_api",
                message=str(e),
            ) from e

    async def _fetch_company_news(
        self,
        co_code: float,
        nsesymbol: str,
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        """
        Fetch news for a single company.

        Args:
            co_code: Company code from master data
            nsesymbol: NSE symbol for logging
            limit: Number of news items to fetch

        Returns:
            List of news items for this company
        """
        url = f"https://wealthyapis.cmots.com/api/Company-News/{int(co_code)}"

        try:
            logger.info("fetching_company_news", nsesymbol=nsesymbol, co_code=int(co_code), url=url)

            response = await self._call_proxy_api(
                method="get",
                url=url,
                payload=None,
            )

            items = self._extract_items(response)
            logger.info("fetched_company_news_count", nsesymbol=nsesymbol, count=len(items))

            # Limit results
            return items[:limit]

        except Exception as e:
            logger.error("fetch_company_news_error", nsesymbol=nsesymbol, error=str(e))
            return []

    async def fetch_company_news_by_symbols(
        self,
        nse_symbols: List[str],
        limit: int = 10,
        page: int = 1,
        per_page: int = 10,
    ) -> Dict[str, Any]:
        """
        Fetch company-wise news for multiple NSE symbols with concurrency.

        Args:
            nse_symbols: List of NSE symbols to fetch news for
            limit: Number of news items per company
            page: Page number for pagination (1-indexed)
            per_page: Items per page

        Returns:
            Dictionary with company news organized by symbol and pagination metadata
        """
        # Fetch company master
        company_master = await self._fetch_company_master()

        # Filter valid symbols
        valid_symbols = [s for s in nse_symbols if s in company_master]
        invalid_symbols = [s for s in nse_symbols if s not in company_master]

        if not valid_symbols:
            return {
                "data": {},
                "pagination": {
                    "page": page,
                    "per_page": per_page,
                    "total_items": 0,
                    "total_pages": 0,
                    "has_next": False,
                    "has_prev": False,
                },
                "errors": [f"No valid symbols found. Invalid: {invalid_symbols}"],
                "fetched_at": datetime.utcnow().isoformat(),
            }

        # Fetch all company news concurrently
        tasks = [
            self._fetch_company_news(
                company_master[symbol].get("co_code"),
                symbol,
                limit,
            )
            for symbol in valid_symbols
        ]

        news_results = await asyncio.gather(*tasks)

        # Organize by symbol
        company_news = {
            symbol: news_results[i]
            for i, symbol in enumerate(valid_symbols)
        }

        # Track which symbols have data and which don't
        symbols_with_data = [s for s in valid_symbols if len(company_news[s]) > 0]
        symbols_with_no_data = [s for s in valid_symbols if len(company_news[s]) == 0]

        # Count total items
        total_items = sum(len(news) for news in company_news.values())
        total_pages = (total_items + per_page - 1) // per_page if per_page > 0 else 1

        # Apply pagination (simple pagination across all news)
        start_idx = (page - 1) * per_page
        end_idx = start_idx + per_page

        # Flatten, paginate, then reorganize
        all_news_flat = []
        for symbol in valid_symbols:
            for news in company_news[symbol]:
                # Deep copy to avoid mutating cached data
                news_copy = dict(news)
                news_copy["_symbol"] = symbol
                all_news_flat.append(news_copy)

        paginated_flat = all_news_flat[start_idx:end_idx]

        # Reorganize back by symbol
        paginated_by_symbol = {}
        for news in paginated_flat:
            symbol = news.pop("_symbol")
            if symbol not in paginated_by_symbol:
                paginated_by_symbol[symbol] = []
            paginated_by_symbol[symbol].append(news)

        # Include empty lists for symbols that have no data
        for symbol in symbols_with_no_data:
            if symbol not in paginated_by_symbol:
                paginated_by_symbol[symbol] = []

        errors = []
        if invalid_symbols:
            errors.append(f"Invalid symbols not found in CompanyMaster: {invalid_symbols}")
        if symbols_with_no_data:
            errors.append(f"No company news found for symbols: {symbols_with_no_data}")

        return {
            "data": paginated_by_symbol,
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
            "symbols_requested": nse_symbols,
            "symbols_found": valid_symbols,
            "symbols_not_found": invalid_symbols,
            "symbols_with_data": symbols_with_data,
            "symbols_with_no_data": symbols_with_no_data,
            "errors": errors if errors else None,
            "fetched_at": datetime.utcnow().isoformat(),
        }


def get_company_news_service() -> CompanyNewsService:
    """Create a new company news service instance."""
    return CompanyNewsService()
