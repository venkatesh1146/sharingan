"""
Reusable client for the Wealthy Funds Proxy API.

This helper posts to the proxy with a payload like:
{
  "method": "get",
  "url": "https://wealthyapis.cmots.com/api/Company-News/33",
  "payload": null,
  "headers": {
    "Authorization": "{{cmots_token}}"
  }
}
"""

from typing import Any, Dict, Optional

import httpx

from app.config import get_settings
from app.utils.exceptions import DataFetchError
from app.utils.logging import get_logger

logger = get_logger(__name__)


def _build_proxy_headers(x_token: str) -> Dict[str, str]:
    return {
        "X-TOKEN": x_token,
        "Content-Type": "application/json",
    }


def _build_target_headers(
    cmots_token: str,
    headers: Optional[Dict[str, str]] = None,
) -> Dict[str, str]:
    merged = dict(headers or {})
    if cmots_token and "Authorization" not in merged:
        merged["Authorization"] = cmots_token
    return merged


async def call_funds_proxy(
    *,
    method: str,
    url: str,
    payload: Optional[Dict[str, Any]] = None,
    headers: Optional[Dict[str, str]] = None,
    timeout_seconds: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Call the Wealthy proxy API with a reusable helper.

    Args:
        method: HTTP method to use for the upstream call (e.g., "get")
        url: Upstream URL to call via the proxy
        payload: Optional payload to send to upstream
        headers: Optional upstream headers (e.g., Authorization)
        timeout_seconds: Override timeout for this call

    Returns:
        Parsed JSON response from the proxy
    """
    settings = get_settings()
    proxy_url = settings.FUNDS_API_PROXY_URL
    x_token = settings.FUNDS_API_X_TOKEN
    cmots_token = settings.CMOTS_TOKEN

    if not proxy_url:
        raise DataFetchError("funds_proxy", "FUNDS_API_PROXY_URL is not set")
    if not x_token:
        raise DataFetchError("funds_proxy", "FUNDS_API_X_TOKEN is not set")

    request_body = {
        "method": method,
        "url": url,
        "payload": payload,
        "headers": _build_target_headers(cmots_token, headers),
    }

    timeout = timeout_seconds or settings.FUNDS_API_TIMEOUT_SECONDS

    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(
                proxy_url,
                headers=_build_proxy_headers(x_token),
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
