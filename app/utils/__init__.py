"""
Utility modules for Market Pulse Multi-Agent API.

Contains:
- logging: Structured logging setup
- exceptions: Custom exception classes
- tracing: OpenTelemetry distributed tracing
- vertex_ai_client: Vertex AI client wrapper
- pagination: Standardized pagination utilities
- funds_api_client: Reusable Wealthy proxy API client
"""

from app.utils.exceptions import (
    MarketPulseError,
    AgentExecutionError,
    AgentTimeoutError,
    AgentReasoningError,
    DataValidationError,
    DataFetchError,
    OrchestrationError,
)
from app.utils.logging import setup_logging, get_logger
from app.utils.pagination import (
    PaginationMeta,
    PaginatedResponse,
    paginate_list,
    create_paginated_response,
)

__all__ = [
    # Exceptions
    "MarketPulseError",
    "AgentExecutionError",
    "AgentTimeoutError",
    "AgentReasoningError",
    "DataValidationError",
    "DataFetchError",
    "OrchestrationError",
    # Logging
    "setup_logging",
    "get_logger",
    # Pagination
    "PaginationMeta",
    "PaginatedResponse",
    "paginate_list",
    "create_paginated_response",
]
