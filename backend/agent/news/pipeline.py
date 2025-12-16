import re
from datetime import date, datetime
from .finnhub_client import is_valid_ticker_finnhub

# =====================================================
# Relevance & company specificity
# =====================================================

def is_ticker_relevant(item: dict, ticker: str) -> bool:
    T = ticker.upper()
    related = (item.get("related") or "").upper()
    rel_tokens = {x.strip() for x in related.split(",") if x.strip()}

    if T in rel_tokens:
        return True

    text = " ".join([
        str(item.get("headline","")),
        str(item.get("summary","")),
        str(item.get("source",""))
    ]).upper()

    return re.search(rf"\b{re.escape(T)}\b", text) is not None


def is_company_specific(item: dict, ticker: str) -> bool:
    if not item or "error" in item:
        return False

    T = ticker.upper()
    headline = (item.get("headline") or "").upper()

    if re.search(rf"\b{re.escape(T)}\b", headline):
        return True

    COMPANY_NAMES = {
        "AAPL": "APPLE",
        "NVDA": "NVIDIA",
        "MSFT": "MICROSOFT",
        "AMZN": "AMAZON",
        "META": "META",
        "GOOGL": "ALPHABET",
        "TSLA": "TESLA",
        "AMD": "ADVANCED MICRO DEVICES",
    }

    cname = COMPANY_NAMES.get(T)
    return cname in headline if cname else False

# =====================================================
# Date filtering
# =====================================================

def filter_items_to_date(items: list, target_d: date):
    if not items:
        return items

    out = []
    for it in items:
        ts = it.get("datetime")
        if ts is None:
            continue
        if datetime.utcfromtimestamp(ts).date() == target_d:
            out.append(it)
    return out

def extract_news_date(user_query: str):
    m = re.search(r"\b(20\d{2})-(\d{2})-(\d{2})\b", user_query)
    if not m:
        return None
    y, mth, d = map(int, m.groups())
    try:
        return date(y, mth, d)
    except ValueError:
        return None

# =====================================================
# Price relevance scoring
# =====================================================

def score_price_relevance(item: dict, ticker: str) -> int:
    text = f"{item.get('headline','')} {item.get('summary','')}".lower()
    score = 0

    high = [
        "earnings", "guidance", "revenue", "profit", "beat", "miss",
        "lawsuit", "antitrust", "regulator", "sec", "doj",
        "upgrade", "downgrade", "price target",
        "buyback", "dividend", "split"
    ]

    medium = [
        "forecast", "margin", "valuation", "analyst", "etf",
        "buffett", "stake"
    ]

    for w in high:
        if w in text:
            score += 3
    for w in medium:
        if w in text:
            score += 1

    if ticker.lower() in text:
        score += 2

    return score


def filter_price_relevant_news(
    items: list,
    ticker: str,
    max_items: int = 8,
    min_score: int = 3
):
    scored = []
    for it in items:
        if "error" in it:
            continue
        scored.append((score_price_relevance(it, ticker), it))

    scored.sort(reverse=True, key=lambda x: x[0])

    strong = [it for s, it in scored if s >= min_score]
    return (strong or [it for _, it in scored])[:max_items]

# =====================================================
# User intent helpers
# =====================================================

def wants_price_relevant_news(user_query: str) -> bool:
    q = user_query.lower()
    triggers = [
        "price", "stock", "shares", "move", "moving",
        "down", "up", "drop", "rally", "impact", "catalyst"
    ]
    return any(t in q for t in triggers)


def wants_why_move(user_query: str) -> bool:
    q = user_query.lower()
    return "why" in q and ("move" in q or "moving" in q)

# =====================================================
# Unified pipeline
# =====================================================

def news_pipeline(items, ticker, user_query, asked_date=None):
    items = [it for it in items if isinstance(it, dict) and "error" not in it]

    if asked_date:
        items = filter_items_to_date(items, asked_date)

    items = [it for it in items if is_ticker_relevant(it, ticker)]
    items = [it for it in items if is_company_specific(it, ticker)]

    if wants_price_relevant_news(user_query) or wants_why_move(user_query):
        items = filter_price_relevant_news(items, ticker, max_items=12, min_score=1)
    else:
        items = items[:12]

    return items

