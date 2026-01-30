"""
Pagination utilities for standardized API responses.

Provides consistent pagination format across all list APIs
optimized for mobile and web clients.
"""

from typing import Any, Dict, List, TypeVar, Generic, Optional
from math import ceil
from pydantic import BaseModel, Field

T = TypeVar("T")


class PaginationMeta(BaseModel):
    """Standardized pagination metadata."""

    page: int = Field(
        default=1,
        description="Current page number (1-indexed)",
        ge=1,
    )
    per_page: int = Field(
        default=10,
        description="Number of items per page",
        ge=1,
        le=100,
    )
    total_items: int = Field(
        default=0,
        description="Total number of items across all pages",
        ge=0,
    )
    total_pages: int = Field(
        default=0,
        description="Total number of pages",
        ge=0,
    )
    has_next: bool = Field(
        default=False,
        description="Whether there is a next page",
    )
    has_prev: bool = Field(
        default=False,
        description="Whether there is a previous page",
    )
    next_page: Optional[int] = Field(
        default=None,
        description="Next page number, or null if at last page",
    )
    prev_page: Optional[int] = Field(
        default=None,
        description="Previous page number, or null if at first page",
    )


class PaginatedResponse(BaseModel, Generic[T]):
    """
    Standardized paginated response wrapper.

    Used across all list/collection APIs for consistency.
    Mobile-friendly with compact field names.
    """

    data: List[T] = Field(
        default_factory=list,
        description="List of items for this page",
    )
    pagination: PaginationMeta = Field(
        description="Pagination metadata",
    )

    class Config:
        # Allow generic types
        arbitrary_types_allowed = True


def paginate_list(
    items: List[T],
    page: int = 1,
    per_page: int = 10,
) -> Dict[str, Any]:
    """
    Paginate a list and return standardized response.

    Args:
        items: List of items to paginate
        page: Page number (1-indexed)
        per_page: Items per page (1-100)

    Returns:
        Dictionary with paginated data and metadata
    """
    # Validate and normalize parameters
    page = max(1, page)
    per_page = max(1, min(100, per_page))  # Clamp to 1-100

    total_items = len(items)
    total_pages = ceil(total_items / per_page) if total_items > 0 else 0

    # Validate page number
    if total_pages > 0:
        page = min(page, total_pages)

    # Calculate slice indices
    start_idx = (page - 1) * per_page
    end_idx = start_idx + per_page
    paginated_items = items[start_idx:end_idx]

    # Calculate next/prev
    has_next = page < total_pages
    has_prev = page > 1
    next_page = page + 1 if has_next else None
    prev_page = page - 1 if has_prev else None

    pagination = PaginationMeta(
        page=page,
        per_page=per_page,
        total_items=total_items,
        total_pages=total_pages,
        has_next=has_next,
        has_prev=has_prev,
        next_page=next_page,
        prev_page=prev_page,
    )

    return {
        "data": paginated_items,
        "pagination": pagination.model_dump(),
    }


def create_paginated_response(
    items: List[Dict[str, Any]],
    page: int = 1,
    per_page: int = 10,
    **extra_fields,
) -> Dict[str, Any]:
    """
    Create a standardized paginated response with optional extra fields.

    Args:
        items: List of items to paginate
        page: Page number (1-indexed)
        per_page: Items per page
        **extra_fields: Additional fields to include in response (errors, etc.)

    Returns:
        Complete response dictionary with data, pagination, and extra fields
    """
    paginated = paginate_list(items, page, per_page)

    response = {
        "data": paginated["data"],
        "pagination": paginated["pagination"],
    }

    # Add extra fields if provided
    if extra_fields:
        response.update(extra_fields)

    return response
