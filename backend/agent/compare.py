
from typing import List, Dict

from agent.technical import technical_summary_daily
from agent.sentiment import sentiment_analyzer
from agent.news import (
    finnhub_company_news,
    filter_price_relevant_news,
)
from agent.price import get_today_price_change


# =========================================================
# Scoring logic
# =========================================================

def score_technical_sentiment(tech: Dict, sent: Dict) -> float:
    """
    Simple PM-style composite score.
    Positive = relatively better.
    """

    score = 0.0

    # ---- Trend ----
    trend = tech.get("trend")
    if trend == "Bullish":
        score += 1.0
    elif trend == "Improving":
        score += 0.5
    elif trend == "Weak":
        score -= 0.5

    # ---- RSI ----
    rsi = tech.get("rsi14")
    if rsi is not None:
        if rsi >= 60:
            score += 0.5
        elif rsi <= 40:
            score -= 0.5

    # ---- Drawdown ----
    mdd = tech.get("max_drawdown")
    if mdd is not None:
        if mdd <= -0.25:
            score -= 0.5
        elif mdd >= -0.10:
            score += 0.3

    # ---- Sentiment ----
    sent_score = sent.get("score", 0.0)
    score += float(sent_score)

    return round(score, 2)


# =========================================================
# Formatting helpers
# =========================================================

def format_compare_table(rows: List[Dict]) -> str:
    """
    Render a clean ASCII comparison table.
    """

    if not rows:
        return ""

    headers = list(rows[0].keys())
    col_w = {
        h: max(len(h), max(len(str(r[h])) for r in rows))
        for h in headers
    }

    def fmt_row(r):
        return " | ".join(str(r[h]).ljust(col_w[h]) for h in headers)

    sep = "-+-".join("-" * col_w[h] for h in headers)

    lines = [
        fmt_row({h: h for h in headers}),
        sep,
    ]
    for r in rows:
        lines.append(fmt_row(r))

    return "\n".join(lines)


# =========================================================
# PM-style relative verdict
# =========================================================

def generate_relative_verdict(
    ticker_a: str, a_tech: Dict, a_sent: Dict,
    ticker_b: str, b_tech: Dict, b_sent: Dict
) -> str:
    """
    Deterministic PM-style verdict.
    NO LLM. Explainable.
    """

    a_score = score_technical_sentiment(a_tech, a_sent)
    b_score = score_technical_sentiment(b_tech, b_sent)

    if a_score == b_score:
        return (
            "📌 Relative Verdict\n\n"
            "No clear relative preference at this time.\n"
            "Both stocks show comparable technical and sentiment profiles."
        )

    if a_score > b_score:
        winner, loser = ticker_a, ticker_b
        w_score, l_score = a_score, b_score
    else:
        winner, loser = ticker_b, ticker_a
        w_score, l_score = b_score, a_score

    return f"""📌 Relative Verdict (PM-style)

Preferred: {winner}
Relative Laggard: {loser}

Rationale:
- {winner} exhibits stronger technical structure and/or sentiment support.
- Relative score advantage suggests better risk-adjusted positioning.

Score Summary:
- {winner}: {w_score:.2f}
- {loser}: {l_score:.2f}

What would change the view:
- Trend deterioration or sentiment shock for {winner}
- Confirmed technical improvement for {loser}
""".strip()


# =========================================================
# Compare type detection
# =========================================================

def _detect_compare_type(query: str) -> str:
    q = query.lower()

    if any(k in q for k in ["price", "performance", "return", "change", "moved"]):
        return "price"

    if any(k in q for k in ["risk", "drawdown", "volatility"]):
        return "risk"

    if any(k in q for k in ["sentiment", "bullish", "bearish", "tone"]):
        return "sentiment"

    return "technical"


# =========================================================
# Public entry point
# =========================================================

def run_compare(tickers: List[str], user_query: str) -> str:
    """
    Main compare entry.
    Called ONLY by finance_agent.
    """

    if len(tickers) < 2:
        return "❌ Please provide two tickers to compare."

    tickers = tickers[:2]
    t1, t2 = tickers
    compare_type = _detect_compare_type(user_query)

    # =====================================================
    # PRICE-ONLY COMPARISON
    # =====================================================
    if compare_type == "price":
        rows = []

        for t in tickers:
            info = get_today_price_change(t)
            if not info:
                continue

            rows.append({
                "Ticker": t,
                "Price": f"${info['price']:.2f}",
                "Change": f"{info['pct_change']:+.2f}%"
            })

        if not rows:
            return "⚠️ Price data unavailable."

        out = []
        out.append("📊 Price Performance Comparison (Today)\n")
        out.append(format_compare_table(rows))

        if len(rows) == 2:
            a, b = rows
            a_chg = float(a["Change"].replace("%", ""))
            b_chg = float(b["Change"].replace("%", ""))

            winner = a if a_chg > b_chg else b
            loser = b if winner is a else a

            out.append("\n📌 Relative Price Performance")
            out.append(f"Outperformed: {winner['Ticker']}")
            out.append(f"Underperformed: {loser['Ticker']}")

        return "\n".join(out)

    # =====================================================
    # TECHNICAL / SENTIMENT COMPARISON
    # =====================================================
    rows = []
    tech_map = {}
    sent_map = {}

    for t in tickers:
        tech = technical_summary_daily(t)
        sent = {}

        if compare_type in {"sentiment", "technical"}:
            items = finnhub_company_news(
                t, days=7, max_items=20, strict=True
            )
            items = filter_price_relevant_news(
                items, t, max_items=10, min_score=2
            )
            sent = sentiment_analyzer.analyze(items, ticker=t)

        tech_map[t] = tech
        sent_map[t] = sent

        score = score_technical_sentiment(tech or {}, sent or {})

        rows.append({
            "Ticker": t,
            "Trend": tech.get("trend", "N/A") if tech else "N/A",
            "RSI": f"{tech['rsi14']:.0f}" if tech and tech.get("rsi14") is not None else "N/A",
            "Max DD": f"{tech['max_drawdown']*100:.1f}%" if tech else "N/A",
            "Daily Bias": tech.get("daily_bias", "N/A") if tech else "N/A",
            "Sentiment": sent.get("sentiment", "N/A").capitalize() if sent else "N/A",
            "Score": score,
        })

    out = []
    out.append("📊 Technical & Sentiment Comparison\n")
    out.append(format_compare_table(rows))

    if len(rows) == 2:
        verdict = generate_relative_verdict(
            t1, tech_map[t1], sent_map[t1],
            t2, tech_map[t2], sent_map[t2],
        )
        out.append("\n" + verdict)

    return "\n".join(out)

