import re
from collections import Counter

def fix_mojibake(s: str) -> str:
    try:
        return s.encode("latin1").decode("utf-8")
    except Exception:
        return s

def extract_daily_event_hint(items):
    if not items:
        return ""
    it = items[0]
    text = f"{it.get('headline','')} {it.get('summary','')}"
    return fix_mojibake(text)[:180].rstrip(".") + "."

def date_signature_sentence(items, k=6):
    text = " ".join(
        fix_mojibake(it.get("headline","")) + " " +
        fix_mojibake(it.get("summary",""))
        for it in items
    ).lower()

    words = re.findall(r"[a-z]{3,}", text)
    counts = Counter(words)
    top = [w for w, c in counts.most_common() if c >= 2][:k]
    return "Key topics mentioned include: " + ", ".join(top) + "." if top else ""

def dominant_event_sentence(items, ticker: str) -> str:
    text = " ".join(
        fix_mojibake(it.get("headline","")) + " " +
        fix_mojibake(it.get("summary",""))
        for it in items
    ).lower()

    rules = [
        (["earnings","guidance"], "earnings and guidance"),
        (["antitrust","regulator","doj"], "regulatory developments"),
        (["upgrade","downgrade","rating"], "analyst actions"),
        (["iphone","demand","sales"], "product demand signals"),
    ]

    for kws, desc in rules:
        if any(k in text for k in kws):
            return f"On this date, {ticker} news was dominated by {desc}."

    return f"On this date, coverage was mixed with no single dominant catalyst."

def summarize_news_paragraph(items, ticker: str, user_query: str) -> str:
    if not items:
        return f"No recent news available for {ticker}."

    sig = date_signature_sentence(items)
    dom = dominant_event_sentence(items, ticker)
    hint = extract_daily_event_hint(items)

    return (
        f"{sig}\n\n"
        f"{dom}\n\n"
        f"Most concrete signal: {hint}"
    )

