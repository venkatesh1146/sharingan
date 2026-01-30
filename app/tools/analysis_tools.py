"""
Analysis Tools for the Impact Analysis Agent.

These tools provide functionality to:
- Identify sectors from stock tickers
- Calculate stock correlations
- Get company fundamentals
- Analyze supply chain impacts
"""

from typing import Any, Callable, Dict, List, Optional


# =============================================================================
# Mock Data (Replace with actual API/database calls in production)
# =============================================================================


# Stock to Sector mapping
STOCK_SECTORS = {
    # IT
    "TCS": "IT",
    "INFY": "IT",
    "WIPRO": "IT",
    "HCLTECH": "IT",
    "TECHM": "IT",
    # Banking
    "HDFC": "Banking",
    "ICICI": "Banking",
    "SBI": "Banking",
    "KOTAK": "Banking",
    "AXIS": "Banking",
    # Oil & Gas
    "RELIANCE": "Oil & Gas",
    "IOCL": "Oil & Gas",
    "BPCL": "Oil & Gas",
    "HINDPETRO": "Oil & Gas",
    "ONGC": "Oil & Gas",
    # Auto
    "MARUTI": "Auto",
    "TATAMOTORS": "Auto",
    "M&M": "Auto",
    "BAJAJ-AUTO": "Auto",
    "HEROMOTOCO": "Auto",
    # Metals
    "TATASTEEL": "Metals",
    "JSWSTEEL": "Metals",
    "SAIL": "Metals",
    "HINDALCO": "Metals",
    "VEDL": "Metals",
    # Pharma
    "SUNPHARMA": "Pharma",
    "DRREDDY": "Pharma",
    "CIPLA": "Pharma",
    "DIVISLAB": "Pharma",
    "BIOCON": "Pharma",
    # Consumer
    "HINDUNILVR": "Consumer",
    "ITC": "Consumer",
    "NESTLEIND": "Consumer",
    "BRITANNIA": "Consumer",
    "DABUR": "Consumer",
    # Paints
    "ASIANPAINT": "Paints",
    "BERGEPAINT": "Paints",
    "PIDILITIND": "Paints",
    # Electronics
    "DIXON": "Electronics",
    "AMBER": "Electronics",
    # Telecom
    "BHARTIARTL": "Telecom",
    "IDEA": "Telecom",
    # Real Estate
    "DLF": "Real Estate",
    "GODREJPROP": "Real Estate",
}

# Company fundamentals
COMPANY_FUNDAMENTALS = {
    "TCS": {
        "name": "Tata Consultancy Services",
        "sector": "IT",
        "market_cap": "Large Cap",
        "revenue_sources": ["IT Services", "Consulting", "Digital Solutions"],
        "key_clients": ["US Banks", "Retail", "Manufacturing"],
        "export_revenue_percent": 95,
        "input_costs": ["Employee Costs", "Infrastructure"],
    },
    "RELIANCE": {
        "name": "Reliance Industries Ltd",
        "sector": "Oil & Gas",
        "market_cap": "Large Cap",
        "revenue_sources": ["Petrochemicals", "Refining", "Retail", "Telecom"],
        "key_clients": ["B2B", "B2C"],
        "export_revenue_percent": 35,
        "input_costs": ["Crude Oil", "Natural Gas"],
    },
    "ASIANPAINT": {
        "name": "Asian Paints Ltd",
        "sector": "Paints",
        "market_cap": "Large Cap",
        "revenue_sources": ["Decorative Paints", "Industrial Coatings"],
        "key_clients": ["Retail", "Real Estate", "Auto OEMs"],
        "export_revenue_percent": 10,
        "input_costs": ["Crude Oil Derivatives", "Titanium Dioxide", "Packaging"],
    },
    "MARUTI": {
        "name": "Maruti Suzuki India Ltd",
        "sector": "Auto",
        "market_cap": "Large Cap",
        "revenue_sources": ["Passenger Vehicles", "Spare Parts", "Services"],
        "key_clients": ["Retail Consumers"],
        "export_revenue_percent": 15,
        "input_costs": ["Steel", "Aluminum", "Semiconductors", "Rubber"],
    },
    "TATASTEEL": {
        "name": "Tata Steel Ltd",
        "sector": "Metals",
        "market_cap": "Large Cap",
        "revenue_sources": ["Steel Products", "Mining"],
        "key_clients": ["Auto", "Construction", "Infrastructure"],
        "export_revenue_percent": 25,
        "input_costs": ["Iron Ore", "Coking Coal", "Energy"],
    },
}

# Sector correlations
SECTOR_CORRELATIONS = {
    ("IT", "Banking"): 0.45,
    ("IT", "Consumer"): 0.30,
    ("Oil & Gas", "Paints"): -0.55,  # Oil up → Paints down (input cost)
    ("Oil & Gas", "Auto"): -0.40,
    ("Metals", "Auto"): -0.50,  # Steel up → Auto down (input cost)
    ("Banking", "Real Estate"): 0.65,
    ("Pharma", "Healthcare"): 0.80,
}

# Supply chain relationships
SUPPLY_CHAIN = {
    "crude_oil_up": {
        "direct_negative": ["IOCL", "BPCL", "HINDPETRO"],  # OMCs face margin pressure
        "indirect_negative": ["ASIANPAINT", "BERGEPAINT"],  # Paint input costs
        "indirect_negative_2": ["MARUTI", "TATAMOTORS"],  # Fuel costs for consumers
        "positive": ["ONGC", "RELIANCE"],  # Oil producers benefit
    },
    "steel_price_up": {
        "direct_positive": ["TATASTEEL", "JSWSTEEL", "SAIL"],
        "direct_negative": ["MARUTI", "TATAMOTORS", "M&M"],  # Auto input costs
        "indirect_negative": ["DLF", "GODREJPROP"],  # Construction costs
    },
    "rupee_depreciation": {
        "positive": ["TCS", "INFY", "WIPRO"],  # IT exporters
        "negative": ["IOCL", "BPCL"],  # Oil importers
    },
    "interest_rate_cut": {
        "positive": ["HDFC", "ICICI", "DLF", "GODREJPROP"],  # Banks, Real Estate
        "negative": [],
    },
}


# =============================================================================
# Tool Functions
# =============================================================================


async def identify_sector_from_stocks(
    tickers: List[str],
) -> Dict[str, Any]:
    """
    Identify sectors for given stock tickers.
    
    Args:
        tickers: List of stock ticker symbols
    
    Returns:
        Dictionary mapping tickers to sectors and sector summary
    """
    ticker_sectors = {}
    sector_counts: Dict[str, int] = {}

    for ticker in tickers:
        ticker_upper = ticker.upper()
        sector = STOCK_SECTORS.get(ticker_upper, "Unknown")
        ticker_sectors[ticker_upper] = sector

        if sector in sector_counts:
            sector_counts[sector] += 1
        else:
            sector_counts[sector] = 1

    return {
        "ticker_sectors": ticker_sectors,
        "sector_distribution": sector_counts,
        "unique_sectors": list(sector_counts.keys()),
        "total_stocks": len(tickers),
    }


async def calculate_stock_correlation(
    ticker1: str,
    ticker2: str,
) -> Dict[str, Any]:
    """
    Calculate correlation between two stocks based on sectors.
    
    In production, this would use historical price data.
    Currently uses sector-based approximations.
    
    Args:
        ticker1: First stock ticker
        ticker2: Second stock ticker
    
    Returns:
        Correlation analysis results
    """
    sector1 = STOCK_SECTORS.get(ticker1.upper(), "Unknown")
    sector2 = STOCK_SECTORS.get(ticker2.upper(), "Unknown")

    # Same sector = high correlation
    if sector1 == sector2:
        correlation = 0.75
        relationship = "Same sector, high positive correlation"
    else:
        # Check sector correlation
        key = (sector1, sector2)
        reverse_key = (sector2, sector1)
        
        if key in SECTOR_CORRELATIONS:
            correlation = SECTOR_CORRELATIONS[key]
        elif reverse_key in SECTOR_CORRELATIONS:
            correlation = SECTOR_CORRELATIONS[reverse_key]
        else:
            correlation = 0.20  # Default low correlation
        
        if correlation > 0.5:
            relationship = "Positive correlation - tend to move together"
        elif correlation < -0.3:
            relationship = "Negative correlation - tend to move opposite"
        else:
            relationship = "Low correlation - largely independent"

    return {
        "ticker1": ticker1.upper(),
        "ticker2": ticker2.upper(),
        "sector1": sector1,
        "sector2": sector2,
        "correlation": round(correlation, 2),
        "relationship": relationship,
    }


async def get_company_fundamentals(
    ticker: str,
) -> Dict[str, Any]:
    """
    Get fundamental information about a company.
    
    Args:
        ticker: Stock ticker symbol
    
    Returns:
        Company fundamentals and business model information
    """
    ticker_upper = ticker.upper()
    
    if ticker_upper in COMPANY_FUNDAMENTALS:
        fundamentals = COMPANY_FUNDAMENTALS[ticker_upper].copy()
        fundamentals["ticker"] = ticker_upper
        fundamentals["data_available"] = True
        return fundamentals
    
    # Return basic info for unknown stocks
    return {
        "ticker": ticker_upper,
        "name": ticker_upper,
        "sector": STOCK_SECTORS.get(ticker_upper, "Unknown"),
        "market_cap": "Unknown",
        "data_available": False,
        "message": f"Detailed fundamentals not available for {ticker_upper}",
    }


async def analyze_supply_chain_impact(
    event: str,
    affected_sector: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Analyze supply chain impact of an economic event.
    
    Args:
        event: Economic event (e.g., "crude_oil_up", "steel_price_up")
        affected_sector: Optional sector to focus analysis on
    
    Returns:
        Supply chain impact analysis with affected stocks
    """
    event_lower = event.lower().replace(" ", "_")
    
    # Try to match event
    matched_event = None
    for key in SUPPLY_CHAIN.keys():
        if key in event_lower or event_lower in key:
            matched_event = key
            break
    
    if matched_event:
        impacts = SUPPLY_CHAIN[matched_event]
        
        # Build causal chains
        causal_chains = []
        
        if "positive" in impacts and impacts["positive"]:
            for stock in impacts["positive"]:
                causal_chains.append({
                    "stock": stock,
                    "impact": "positive",
                    "chain": f"{matched_event.replace('_', ' ').title()} → Direct benefit → {stock}",
                })
        
        if "direct_negative" in impacts:
            for stock in impacts["direct_negative"]:
                causal_chains.append({
                    "stock": stock,
                    "impact": "negative",
                    "chain": f"{matched_event.replace('_', ' ').title()} → Higher costs → {stock} margin pressure",
                })
        
        if "indirect_negative" in impacts:
            for stock in impacts["indirect_negative"]:
                causal_chains.append({
                    "stock": stock,
                    "impact": "negative",
                    "chain": f"{matched_event.replace('_', ' ').title()} → Input cost increase → {stock} affected",
                })
        
        return {
            "event": matched_event,
            "event_recognized": True,
            "causal_chains": causal_chains,
            "positive_stocks": impacts.get("positive", []),
            "negative_stocks": (
                impacts.get("direct_negative", []) +
                impacts.get("indirect_negative", []) +
                impacts.get("indirect_negative_2", [])
            ),
            "summary": f"Event '{matched_event}' impacts {len(causal_chains)} stocks across supply chain",
        }
    
    return {
        "event": event,
        "event_recognized": False,
        "causal_chains": [],
        "message": f"Supply chain impact for '{event}' not in knowledge base",
    }


async def rank_news_by_importance(
    news_items: List[Dict[str, Any]],
    portfolio_tickers: Optional[List[str]] = None,
    watchlist_tickers: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """
    Rank news items by importance considering portfolio/watchlist relevance.
    
    Args:
        news_items: List of news items to rank
        portfolio_tickers: User's portfolio stocks
        watchlist_tickers: User's watchlist stocks
    
    Returns:
        News items sorted by importance score
    """
    portfolio_set = set(t.upper() for t in (portfolio_tickers or []))
    watchlist_set = set(t.upper() for t in (watchlist_tickers or []))
    
    ranked_news = []
    for news in news_items:
        score = 0.5  # Base score
        
        # Breaking news boost
        if news.get("is_breaking"):
            score += 0.3
        
        # Sentiment strength
        sentiment_score = abs(news.get("sentiment_score", 0))
        score += sentiment_score * 0.2
        
        # Portfolio relevance
        mentioned_stocks = set(s.upper() for s in news.get("mentioned_stocks", []))
        if mentioned_stocks & portfolio_set:
            score += 0.3
        
        # Watchlist relevance
        if mentioned_stocks & watchlist_set:
            score += 0.15
        
        # Recency (newer = higher score)
        # In production, calculate based on published_at
        
        ranked_news.append({
            **news,
            "importance_score": round(min(score, 1.0), 2),
        })
    
    return sorted(ranked_news, key=lambda x: x["importance_score"], reverse=True)


# =============================================================================
# Tool Registration
# =============================================================================


def get_analysis_tools() -> List[Any]:
    """
    Get tool definitions for analysis functions.
    
    Returns:
        List of tool definitions (currently empty as we use direct function calls)
    """
    return []


def get_analysis_tool_handlers() -> Dict[str, Callable]:
    """
    Get mapping of tool names to handler functions.
    
    Returns:
        Dictionary mapping function names to async handlers
    """
    return {
        "identify_sector_from_stocks": identify_sector_from_stocks,
        "calculate_stock_correlation": calculate_stock_correlation,
        "get_company_fundamentals": get_company_fundamentals,
        "analyze_supply_chain_impact": analyze_supply_chain_impact,
        "rank_news_by_importance": rank_news_by_importance,
    }
