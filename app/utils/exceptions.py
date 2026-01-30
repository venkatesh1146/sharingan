"""
Custom exceptions for Market Intelligence API.

Provides a hierarchy of exceptions for different error scenarios:
- MarketPulseError: Base exception for all custom errors
- AgentExecutionError: Errors during agent execution
- DataValidationError: Input/output validation errors
- DataFetchError: External data fetching errors
- CacheError: Cache operation errors
"""

from typing import Optional, Dict, Any


class MarketPulseError(Exception):
    """
    Base exception for all Market Intelligence errors.
    
    All custom exceptions should inherit from this class.
    """

    def __init__(
        self,
        message: str,
        code: str = "MARKET_PULSE_ERROR",
        details: Optional[Dict[str, Any]] = None,
    ):
        self.message = message
        self.code = code
        self.details = details or {}
        super().__init__(self.message)

    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary for API responses."""
        return {
            "error": self.code,
            "message": self.message,
            "details": self.details,
        }


# =============================================================================
# Agent Errors
# =============================================================================


class AgentExecutionError(MarketPulseError):
    """
    Raised when a background agent fails to execute successfully.
    
    This is a general error for agent execution failures that don't
    fall into more specific categories.
    """

    def __init__(
        self,
        agent_name: str,
        message: str,
        details: Optional[Dict[str, Any]] = None,
    ):
        self.agent_name = agent_name
        super().__init__(
            message=f"Agent '{agent_name}' execution failed: {message}",
            code="AGENT_EXECUTION_ERROR",
            details={"agent_name": agent_name, **(details or {})},
        )


class AgentTimeoutError(AgentExecutionError):
    """
    Raised when an agent exceeds its execution timeout.
    
    This indicates the agent's operation took too long and was terminated.
    """

    def __init__(
        self,
        agent_name: str,
        timeout_seconds: int,
        details: Optional[Dict[str, Any]] = None,
    ):
        self.timeout_seconds = timeout_seconds
        super().__init__(
            agent_name=agent_name,
            message=f"Execution timed out after {timeout_seconds} seconds",
            details={"timeout_seconds": timeout_seconds, **(details or {})},
        )
        self.code = "AGENT_TIMEOUT_ERROR"


class AgentReasoningError(AgentExecutionError):
    """
    Raised when an agent's AI reasoning or response is invalid.
    
    This indicates the AI model produced output that doesn't meet
    the expected format or quality requirements.
    """

    def __init__(
        self,
        agent_name: str,
        message: str,
        raw_response: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ):
        self.raw_response = raw_response
        super().__init__(
            agent_name=agent_name,
            message=message,
            details={"raw_response": raw_response, **(details or {})},
        )
        self.code = "AGENT_REASONING_ERROR"


# =============================================================================
# Data Errors
# =============================================================================


class DataValidationError(MarketPulseError):
    """
    Raised when data validation fails.
    
    This can occur for input validation, output validation,
    or schema validation failures.
    """

    def __init__(
        self,
        message: str,
        field: Optional[str] = None,
        value: Optional[Any] = None,
        details: Optional[Dict[str, Any]] = None,
    ):
        self.field = field
        self.value = value
        super().__init__(
            message=message,
            code="DATA_VALIDATION_ERROR",
            details={"field": field, "value": str(value), **(details or {})},
        )


class DataFetchError(MarketPulseError):
    """
    Raised when external data fetching fails.
    
    This includes failures to fetch market data, news,
    or any other external data source.
    """

    def __init__(
        self,
        source: str,
        message: str,
        status_code: Optional[int] = None,
        details: Optional[Dict[str, Any]] = None,
    ):
        self.source = source
        self.status_code = status_code
        super().__init__(
            message=f"Failed to fetch data from {source}: {message}",
            code="DATA_FETCH_ERROR",
            details={
                "source": source,
                "status_code": status_code,
                **(details or {}),
            },
        )


# =============================================================================
# Infrastructure Errors
# =============================================================================


class CacheError(MarketPulseError):
    """Raised when cache operations fail."""

    def __init__(
        self,
        operation: str,
        message: str,
        details: Optional[Dict[str, Any]] = None,
    ):
        self.operation = operation
        super().__init__(
            message=f"Cache {operation} failed: {message}",
            code="CACHE_ERROR",
            details={"operation": operation, **(details or {})},
        )


class ConfigurationError(MarketPulseError):
    """Raised when configuration is invalid or missing."""

    def __init__(
        self,
        config_key: str,
        message: str,
        details: Optional[Dict[str, Any]] = None,
    ):
        self.config_key = config_key
        super().__init__(
            message=f"Configuration error for '{config_key}': {message}",
            code="CONFIGURATION_ERROR",
            details={"config_key": config_key, **(details or {})},
        )


class OrchestrationError(MarketPulseError):
    """Raised when agent orchestration fails."""

    def __init__(
        self,
        message: str,
        failed_agents: Optional[list] = None,
        details: Optional[Dict[str, Any]] = None,
    ):
        self.failed_agents = failed_agents or []
        super().__init__(
            message=f"Orchestration failed: {message}",
            code="ORCHESTRATION_ERROR",
            details={"failed_agents": self.failed_agents, **(details or {})},
        )
