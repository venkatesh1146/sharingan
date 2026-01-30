"""
Configuration management for Market Pulse Multi-Agent API.

Uses Pydantic Settings for type-safe configuration with environment variable support.
"""

from functools import lru_cache
from typing import List

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=(".env", ".env.local"),  # .env.local takes precedence
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
    )

    # Google AI Configuration
    GOOGLE_AI_API_KEY: str = ""
    GEMINI_FAST_MODEL: str = "gemini-2.0-flash-exp"
    GEMINI_PRO_MODEL: str = "gemini-1.5-pro"

    # Agent Timeouts (seconds)
    MARKET_DATA_AGENT_TIMEOUT: int = 10
    NEWS_ANALYSIS_AGENT_TIMEOUT: int = 30
    USER_CONTEXT_AGENT_TIMEOUT: int = 15
    IMPACT_ANALYSIS_AGENT_TIMEOUT: int = 45
    SUMMARY_AGENT_TIMEOUT: int = 20
    ORCHESTRATOR_TIMEOUT: int = 120

    # Agent Retry Configuration
    AGENT_RETRY_ATTEMPTS: int = 2
    AGENT_RETRY_DELAY_SECONDS: float = 1.0

    # API Configuration
    API_PORT: int = 8000
    API_HOST: str = "0.0.0.0"
    LOG_LEVEL: str = "INFO"
    CORS_ORIGINS: List[str] = ["http://localhost:3000", "http://localhost:5173"]

    # Feature Flags
    ENABLE_CACHING: bool = True
    ENABLE_TRACING: bool = True
    CACHE_TTL_SECONDS: int = 300

    # Redis Configuration
    REDIS_URL: str = "redis://localhost:6379"

    # Data Source URLs (Placeholders)
    MARKET_DATA_API_URL: str = "https://placeholder.api/market"
    NEWS_API_URL: str = "https://placeholder.api/news"
    USER_DATA_SERVICE_URL: str = "https://placeholder.api/users"

    # Funds Proxy API Configuration
    FUNDS_API_PROXY_URL: str = "https://fundsapi.wealthy.in/proxy-api/"
    FUNDS_API_X_TOKEN: str = ""
    CMOTS_TOKEN: str = ""
    FUNDS_API_TIMEOUT_SECONDS: int = 30

    # Market Phase Times (IST - Indian Standard Time)
    # Pre-market: 08:00 - 09:15
    # Mid-market: 09:15 - 15:30
    # Post-market: 15:30 - 23:59
    PRE_MARKET_START_HOUR: int = 8
    PRE_MARKET_START_MINUTE: int = 0
    MARKET_OPEN_HOUR: int = 9
    MARKET_OPEN_MINUTE: int = 15
    MARKET_CLOSE_HOUR: int = 15
    MARKET_CLOSE_MINUTE: int = 30


@lru_cache()
def get_settings() -> Settings:
    """
    Get cached settings instance.

    Uses lru_cache to ensure settings are only loaded once.
    """
    return Settings()
