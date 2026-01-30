"""
User Data Tools for the User Context Agent.

These tools provide functionality to:
- Fetch user watchlist
- Fetch user portfolio
- Calculate sector exposure
"""

from typing import Any, Callable, Dict, List, Optional


# =============================================================================
# Mock Data (Replace with actual database/API calls in production)
# =============================================================================


# Mock user data store
MOCK_USERS = {
    "user_123": {
        "watchlist": ["RELIANCE", "TCS", "HDFC", "INFY", "ICICI", "BHARTIARTL"],
        "portfolio": [
            {
                "ticker": "RELIANCE",
                "company_name": "Reliance Industries Ltd",
                "quantity": 50,
                "average_price": 2450.00,
                "current_price": 2520.50,
                "sector": "Oil & Gas",
            },
            {
                "ticker": "TCS",
                "company_name": "Tata Consultancy Services",
                "quantity": 30,
                "average_price": 3800.00,
                "current_price": 3950.25,
                "sector": "IT",
            },
            {
                "ticker": "HDFC",
                "company_name": "HDFC Bank Ltd",
                "quantity": 100,
                "average_price": 1580.00,
                "current_price": 1620.75,
                "sector": "Banking",
            },
            {
                "ticker": "INFY",
                "company_name": "Infosys Ltd",
                "quantity": 40,
                "average_price": 1650.00,
                "current_price": 1580.50,
                "sector": "IT",
            },
            {
                "ticker": "TATASTEEL",
                "company_name": "Tata Steel Ltd",
                "quantity": 75,
                "average_price": 145.00,
                "current_price": 152.30,
                "sector": "Metals",
            },
        ],
        "preferences": {
            "news_frequency": "real-time",
            "risk_profile": "moderate",
            "preferred_sectors": ["IT", "Banking"],
        },
    },
    "user_456": {
        "watchlist": ["MARUTI", "TATAMOTORS", "M&M", "BAJAJ-AUTO"],
        "portfolio": [
            {
                "ticker": "MARUTI",
                "company_name": "Maruti Suzuki India Ltd",
                "quantity": 10,
                "average_price": 10500.00,
                "current_price": 11200.00,
                "sector": "Auto",
            },
            {
                "ticker": "ASIANPAINT",
                "company_name": "Asian Paints Ltd",
                "quantity": 25,
                "average_price": 3200.00,
                "current_price": 3050.75,
                "sector": "Paints",
            },
        ],
        "preferences": {
            "news_frequency": "daily",
            "risk_profile": "aggressive",
            "preferred_sectors": ["Auto", "Consumer"],
        },
    },
}

# Default data for unknown users
DEFAULT_USER_DATA = {
    "watchlist": ["NIFTY 50", "SENSEX"],
    "portfolio": [],
    "preferences": {
        "news_frequency": "daily",
        "risk_profile": "moderate",
        "preferred_sectors": [],
    },
}


# =============================================================================
# Tool Functions
# =============================================================================


async def fetch_user_watchlist(user_id: str) -> Dict[str, Any]:
    """
    Fetch user's watchlist.
    
    In production, this would query a database or user service.
    
    Args:
        user_id: User identifier
    
    Returns:
        Dictionary with watchlist data
    """
    user_data = MOCK_USERS.get(user_id, DEFAULT_USER_DATA)
    
    return {
        "user_id": user_id,
        "watchlist": user_data["watchlist"],
        "watchlist_count": len(user_data["watchlist"]),
        "last_updated": "2026-01-30T10:00:00+05:30",
    }


async def fetch_user_portfolio(user_id: str) -> Dict[str, Any]:
    """
    Fetch user's portfolio holdings.
    
    In production, this would query a portfolio service.
    
    Args:
        user_id: User identifier
    
    Returns:
        Dictionary with portfolio data including calculated metrics
    """
    user_data = MOCK_USERS.get(user_id, DEFAULT_USER_DATA)
    portfolio = user_data["portfolio"]
    
    # Calculate metrics for each holding
    enriched_portfolio = []
    total_invested = 0.0
    total_current = 0.0
    
    for holding in portfolio:
        quantity = holding["quantity"]
        avg_price = holding["average_price"]
        current_price = holding["current_price"]
        
        invested_value = quantity * avg_price
        current_value = quantity * current_price
        unrealized_pnl = current_value - invested_value
        unrealized_pnl_percent = (unrealized_pnl / invested_value * 100) if invested_value > 0 else 0
        
        total_invested += invested_value
        total_current += current_value
        
        enriched_portfolio.append({
            **holding,
            "invested_value": invested_value,
            "current_value": current_value,
            "unrealized_pnl": round(unrealized_pnl, 2),
            "unrealized_pnl_percent": round(unrealized_pnl_percent, 2),
        })
    
    # Calculate weights
    for holding in enriched_portfolio:
        holding["weight_in_portfolio"] = round(
            holding["current_value"] / total_current * 100 if total_current > 0 else 0,
            2
        )
    
    total_pnl = total_current - total_invested
    total_pnl_percent = (total_pnl / total_invested * 100) if total_invested > 0 else 0
    
    return {
        "user_id": user_id,
        "holdings": enriched_portfolio,
        "total_invested_value": round(total_invested, 2),
        "total_current_value": round(total_current, 2),
        "total_unrealized_pnl": round(total_pnl, 2),
        "total_unrealized_pnl_percent": round(total_pnl_percent, 2),
        "holdings_count": len(enriched_portfolio),
        "last_updated": "2026-01-30T10:00:00+05:30",
    }


async def calculate_sector_exposure(
    portfolio: List[Dict[str, Any]],
) -> Dict[str, float]:
    """
    Calculate sector-wise exposure from portfolio holdings.
    
    Args:
        portfolio: List of portfolio holdings with sector and weight info
    
    Returns:
        Dictionary mapping sector names to exposure percentages
    """
    sector_weights: Dict[str, float] = {}
    
    for holding in portfolio:
        sector = holding.get("sector", "Unknown")
        weight = holding.get("weight_in_portfolio", 0) or holding.get("current_value", 0)
        
        if sector in sector_weights:
            sector_weights[sector] += weight
        else:
            sector_weights[sector] = weight
    
    # Normalize to percentages if using absolute values
    total = sum(sector_weights.values())
    if total > 100:  # Likely using absolute values
        sector_weights = {
            sector: round(weight / total * 100, 2)
            for sector, weight in sector_weights.items()
        }
    
    return dict(sorted(sector_weights.items(), key=lambda x: x[1], reverse=True))


async def get_user_preferences(user_id: str) -> Dict[str, Any]:
    """
    Fetch user preferences and settings.
    
    Args:
        user_id: User identifier
    
    Returns:
        Dictionary of user preferences
    """
    user_data = MOCK_USERS.get(user_id, DEFAULT_USER_DATA)
    
    return {
        "user_id": user_id,
        "preferences": user_data["preferences"],
    }


# =============================================================================
# Tool Registration
# =============================================================================


def get_user_data_tools() -> List[Any]:
    """
    Get tool definitions for user data functions.
    
    Returns:
        List of tool definitions (currently empty as we use direct function calls)
    """
    return []


def get_user_data_tool_handlers() -> Dict[str, Callable]:
    """
    Get mapping of tool names to handler functions.
    
    Returns:
        Dictionary mapping function names to async handlers
    """
    return {
        "fetch_user_watchlist": fetch_user_watchlist,
        "fetch_user_portfolio": fetch_user_portfolio,
        "calculate_sector_exposure": calculate_sector_exposure,
        "get_user_preferences": get_user_preferences,
    }
