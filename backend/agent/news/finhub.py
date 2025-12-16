import os
from datetime import date, timedelta
import finnhub

# =====================================================
# Finnhub client setup
# =====================================================

FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY")
if not FINNHUB_API_KEY:
    raise RuntimeError("FINNHUB_API_KEY not set")

finnhub_client = finnhub.Client(api_key=FINNHUB_API_KEY)

# =====================================================
# Quote & validation
# =====================================================

def finnhub_quote(ticker: str) -> dict:
    """
    Latest quote from Finnhub
    (c=current, d=change, dp=% change, h/l/o/pc).
    """
    try:
        return finnhub_client.quote(ticker.upper())
    except Exception as e:
        return {"error": str(e)}


def is_valid_ticker_finnhub(t: str) -> bool:
    """
    Check if a ticker is a real tradable symbol using Finnhub.
    Prevents words like STOCK / TONE / etc.
    """
    try:
        q = finnhub_quote(t)
        return (
            isinstance(q, dict)
            and "error" not in q
            and q.get("c") is not None
        )
    except Exception:
        return False

# =====================================================
# News fetch
# =====================================================

def finnhub_company_news(
    ticker: str,
    days: int = 7,
    max_items: int = 8,
    strict: bool = True
):
    """
    Fetch recent company news.
    strict=True keeps only ticker-relevant items.
    """
    try:
        end = date.today()
        start = end - timedelta(days=days)

        items = finnhub_client.company_news(
            ticker.upper(),
            _from=start.isoformat(),
            to=end.isoformat()
        ) or []

        if not strict:
            return items[:max_items]

        return items[:max_items]

    except Exception as e:
        return [{"error": str(e)}]


def finnhub_company_news_range(
    ticker: str,
    start_d: date,
    end_d: date,
    max_items: int = 30
):
    """
    Fetch company news in a custom date range.
    """
    try:
        items = finnhub_client.company_news(
            ticker.upper(),
            _from=start_d.isoformat(),
            to=end_d.isoformat()
        ) or []

        return items[:max_items]

    except Exception as e:
        return [{"error": str(e)}]

# =====================================================
# Fundamentals
# =====================================================

def finnhub_fundamentals_basic(ticker: str):
    """
    Selected basic financial metrics from Finnhub.
    """
    t = ticker.upper()
    try:
        resp = finnhub_client.company_basic_financials(t, "all") or {}
        metric = resp.get("metric", {}) if isinstance(resp, dict) else {}

        keep = [
            "marketCapitalization",
            "peTTM",
            "pb",
            "epsTTM",
            "dividendYieldIndicatedAnnual",
            "52WeekHigh",
            "52WeekLow",
            "52WeekPriceReturnDaily",
            "beta",
        ]

        out = {k: metric.get(k) for k in keep}
        out["ticker"] = t
        return out

    except Exception as e:
        return {"ticker": t, "error": str(e)}
