"""
Common NSE tickers used to extract mentioned stocks from news when no watchlist is provided.

Used for word-boundary matching in headline + summary to populate mentioned_stocks.
Representative set of liquid / frequently mentioned names (Nifty 50 style + others).
"""

from typing import List

# Common NSE symbols (uppercase) - used when watchlist is not provided.
# Sorted by length descending at match time so longer symbols match first
# (e.g. RELIANCE before RELI). Extend as needed or load from CompanyMaster.
COMMON_NSE_TICKERS: List[str] = [
    "BHARTIARTL", "HINDUNILVR", "BAJFINANCE", "ICICIBANK", "HDFCBANK",
    "RELIANCE", "KOTAKBANK", "INFY", "TCS", "SBIN", "AXISBANK", "BAJAJFINSV",
    "ASIANPAINT", "MARUTI", "HCLTECH", "ITC", "LT", "WIPRO", "TITAN",
    "ULTRACEMCO", "TATAMOTORS", "SUNPHARMA", "NESTLEIND", "TATASTEEL",
    "POWERGRID", "INDUSINDBK", "NTPC", "HINDALCO", "BRITANNIA", "ONGC",
    "CIPLA", "TECHM", "ADANIPORTS", "JSWSTEEL", "DIVISLAB", "DRREDDY",
    "GRASIM", "APOLLOHOSP", "HEROMOTOCO", "EICHERMOT", "COALINDIA", "BPCL",
    "TATACONSUM", "SBILIFE", "HDFCLIFE", "ADANIENT", "LTIM", "APOLLOTYRE",
    "BANKBARODA", "PNB", "IDEA", "IOC", "VEDL", "GAIL", "BEL", "HDFCAMC",
    "PIDILITIND", "SIEMENS", "ABB", "BOSCHLTD", "TATAPOWER", "AMBUJACEM",
    "ACC", "ZEEL", "UPL", "DABUR", "COLPAL", "HAVELLS", "INDIGO", "SRF",
    "BIOCON", "GODREJCP", "JINDALSTEL", "SAIL", "HAL", "IRCTC", "PVR",
    "RELAXO", "DALBHARAT", "PIIND", "AUBANK", "AARTIIND", "ESCORTS",
    "MPHASIS", "LALPATHLAB", "PERSISTENT", "CUMMINSIND", "TORNTPHARM",
    "GLENMARK", "LUPIN", "AUROPHARMA", "MUTHOOTFIN", "CHOLAFIN", "SRTRANSFIN",
    "SBICARD", "BAJAJHLDNG", "MOTHERSUMI", "BALKRISIND", "FEDERALBNK",
    "BANDHANBNK", "IDFC", "IDFCFIRSTB", "RBLBANK", "YESBANK",
]
