# backend/agent_core.py
# -*- coding: utf-8 -*-

import os
import re
import json
import textwrap
import logging
from datetime import datetime, timedelta, date, timezone
from zoneinfo import ZoneInfo

# --- Data Libraries ---
try:
    import yfinance as yf
    import pandas as pd
    import numpy as np
    import finnhub
    from dotenv import load_dotenv
except ImportError as e:
    raise ImportError(f"Missing core library: {e}. Run 'pip install yfinance pandas numpy finnhub-python python-dotenv'")

# --- AI Libraries (Graceful Fallback) ---
try:
    import torch
    from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
except ImportError:
    torch = None
    pipeline = None
    AutoTokenizer = None
    AutoModelForCausalLM = None
    print("⚠️ transformers/torch not installed. AI features will be limited.")

# =====================================================
# Configuration & Setup
# =====================================================

load_dotenv()

# Use the key from environment, or fallback to the one in your script
VALID_KEY = "d514ts1r01qjia5bbe50d514ts1r01qjia5bbe5"
FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY", VALID_KEY)

finnhub_client = None
if FINNHUB_API_KEY:
    try:
        finnhub_client = finnhub.Client(api_key=FINNHUB_API_KEY)
        # print(f"✅ Finnhub client connected.")
    except Exception as e:
        print(f"⚠️ Finnhub connection failed: {e}")

# =====================================================
# LLM Loading (Local ./model_final)
# =====================================================

tokenizer = None
model = None
finbert = None

if AutoTokenizer is not None:
    # 1. Load Fine-Tuned Model
    if os.path.exists("./model_final"):
        try:
            # print("🔄 Loading fine-tuned model... (this may take a moment)")
            tokenizer = AutoTokenizer.from_pretrained("./model_final")
            # model = AutoModelForCausalLM.from_pretrained(
            #     "./model_final",
            #     device_map="auto",
            #     dtype="auto"
            # )
            model = AutoModelForCausalLM.from_pretrained(
                "./model_final",
                device_map="auto",
                torch_dtype="auto",
                offload_folder="offload",  # <--- Essential for low-RAM Macs
                offload_state_dict=True
            )
            print("✅ Fine-tuned model loaded from ./model_final")
        except Exception as e:
            print(f"⚠️ Error loading local model: {e}")
            tokenizer = None
            model = None
    else:
        print("⚠️ ./model_final folder not found. Running in rule-based mode.")

    # 2. Load FinBERT
    try:
        finbert = pipeline(
            "text-classification",
            model="ProsusAI/finbert",
            top_k=None
        )
        # print("✅ FinBERT loaded.")
    except Exception:
        pass

# =====================================================
# Finnhub Tools
# =====================================================

def finnhub_quote(ticker: str) -> dict:
    if finnhub_client is None: return {"error": "finnhub_not_configured"}
    try:
        return finnhub_client.quote(ticker.upper())
    except Exception as e:
        return {"error": str(e)}

def is_valid_ticker_finnhub(t: str) -> bool:
    try:
        q = finnhub_quote(t)
        return isinstance(q, dict) and "error" not in q and q.get("c") is not None
    except Exception:
        return False

def is_ticker_relevant(item: dict, ticker: str) -> bool:
    T = ticker.upper()
    related = (item.get("related") or "").upper()
    rel_tokens = {x.strip() for x in related.split(",") if x.strip()}
    text = " ".join([
        str(item.get("headline","")),
        str(item.get("summary","")),
        str(item.get("source",""))
    ]).upper()
    return (T in rel_tokens) or (re.search(rf"\b{re.escape(T)}\b", text) is not None)

def is_company_specific(item: dict, ticker: str) -> bool:
    if not item or "error" in item: return False
    T = ticker.upper()
    headline = (item.get("headline") or "").upper()
    if re.search(rf"\b{re.escape(T)}\b", headline): return True
    
    COMPANY_NAMES = {
        "AAPL": "APPLE", "NVDA": "NVIDIA", "MSFT": "MICROSOFT",
        "AMZN": "AMAZON", "META": "META", "GOOGL": "ALPHABET",
        "TSLA": "TESLA", "AMD": "ADVANCED MICRO DEVICES",
    }
    cname = COMPANY_NAMES.get(T)
    if cname and cname in headline: return True
    return False

def finnhub_company_news(ticker: str, days=7, max_items=8, strict=True):
    if finnhub_client is None: return []
    try:
        end = date.today()
        start = end - timedelta(days=days)
        items = finnhub_client.company_news(
            ticker, _from=start.isoformat(), to=end.isoformat()
        ) or []
        if not strict: return items[:max_items]
        filtered = [it for it in items if is_ticker_relevant(it, ticker)]
        if not filtered: return []
        return filtered[:max_items]
    except Exception as e:
        return [{"error": str(e)}]

def fix_mojibake(s: str) -> str:
    if not s: return s
    try: return s.encode("latin1").decode("utf-8")
    except Exception: return s

def format_news_items(items):
    lines = []
    for it in items:
        if "error" in it: break
        ts = it.get("datetime", None)
        date_str = datetime.fromtimestamp(ts, timezone.utc).strftime("%Y-%m-%d %H:%M UTC") if ts else "N/A"
        headline = fix_mojibake(it.get("headline", "").strip())
        source = it.get("source", "").strip()
        url = it.get("url","").strip()
        lines.append(f"- [{date_str}] {headline} ({source})\n  {url}")
    return "\n".join(lines) if lines else "No recent news returned."

def score_price_relevance(item: dict, ticker: str) -> int:
    text = " ".join([str(item.get("headline","")), str(item.get("summary",""))]).lower()
    score = 0
    high = ["earnings", "guidance", "revenue", "profit", "sec", "antitrust", "upgrade"]
    for w in high:
        if w in text: score += 3
    if ticker.lower() in text: score += 2
    return score

def filter_price_relevant_news(items: list, ticker: str, max_items: int = 8, min_score: int = 3):
    scored = []
    for it in items:
        if "error" in it: continue
        s = score_price_relevance(it, ticker)
        scored.append((s, it))
    scored.sort(key=lambda x: x[0], reverse=True)
    strong = [it for s, it in scored if s >= min_score]
    if strong: return strong[:max_items]
    return [it for s, it in scored][:max_items]

def finnhub_fundamentals_basic(ticker: str):
    t = ticker.upper()
    try:
        resp = finnhub_client.company_basic_financials(t, "all") or {}
        metric = resp.get("metric", {}) if isinstance(resp, dict) else {}
        keep = ["marketCapitalization", "peTTM", "epsTTM", "52WeekHigh", "52WeekLow"]
        out = {k: metric.get(k, None) for k in keep}
        out["ticker"] = t
        return out
    except Exception as e:
        return {"ticker": t, "error": str(e)}

def format_fundamentals(d: dict):
    if not d or "error" in d: return "No fundamentals returned."
    lines = [f"📊 Fundamentals for {d.get('ticker','')}"]
    for k, v in d.items():
        if k!="ticker": lines.append(f"- {k}: {v}")
    return "\n".join(lines)

# =====================================================
# Sentiment Class (Hybrid)
# =====================================================

class SentimentAnalyzer:
    def __init__(self, finbert_pipeline, llm_model, tokenizer):
        self.finbert = finbert_pipeline
        self.llm = llm_model
        self.tokenizer = tokenizer

    def analyze(self, news_items, ticker=None, mode="label"):
        # 1. Base Sentiment (FinBERT)
        if not self.finbert or not news_items:
            return {"sentiment": "neutral", "score": 0.0, "positive_count": 0, "negative_count": 0, "article_count": 0}

        scores = []
        for it in news_items:
            text = (it.get("headline") or "") + " " + (it.get("summary") or "")
            text = text.strip()[:512]
            if not text: continue
            
            preds = self.finbert(text)[0]
            best = max(preds, key=lambda d: d["score"])
            lab = best["label"].lower()
            sc = best["score"] if lab == "positive" else -best["score"] if lab == "negative" else 0.0
            scores.append(sc)

        avg = np.mean(scores) if scores else 0.0
        pos = sum(1 for s in scores if s > 0.1)
        neg = sum(1 for s in scores if s < -0.1)
        
        base = {
            "sentiment": "positive" if avg > 0.15 else "negative" if avg < -0.15 else "neutral",
            "score": avg,
            "positive_count": pos,
            "negative_count": neg,
            "article_count": len(scores)
        }

        # 2. Reasoning (LLM) if requested
        if mode == "reasoning" and self.llm:
            base["reasoning"] = self._analyze_reason(news_items, ticker, base["sentiment"])
            
        return base

    def _analyze_reason(self, news_items, ticker, sentiment):
        snippets = []
        for it in news_items[:5]:
            h = it.get("headline", "")
            snippets.append(f"- {h}")
        news_block = "\n".join(snippets)
        
        prompt = f"""
        Analyze the sentiment for {ticker} (Current status: {sentiment.upper()}).
        
        News:
        {news_block}
        
        Explain WHY in 2 sentences:
        """
        
        try:
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.llm.device)
            outputs = self.llm.generate(**inputs, max_new_tokens=100, do_sample=False)
            full_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            if "Explain WHY" in full_text:
                return full_text.split("Explain WHY")[-1].strip()
            return full_text
        except Exception as e:
            return f"Reasoning unavailable: {e}"

sentiment_analyzer = SentimentAnalyzer(finbert, model, tokenizer)

def format_sentiment_human(sent: dict, ticker: str = "", when: str = "recent") -> str:
    if not sent or "sentiment" not in sent: return "Sentiment data unavailable."
    
    tone = sent['sentiment'].capitalize()
    score = sent.get('score', 0)
    
    text = f"🧠 Sentiment for {ticker} ({when})\n"
    text += f"Tone: {tone} (Score: {score:.2f})\n"
    text += f"Breakdown: {sent.get('positive_count')} Pos / {sent.get('negative_count')} Neg\n"
    
    if "reasoning" in sent:
        text += f"\nReasoning:\n{sent['reasoning']}"
        
    return text

# =====================================================
# Technical Analysis
# =====================================================

def fetch_stock_data(ticker, period="6mo"):
    try:
        df = yf.download(ticker, period=period, interval="1d", progress=False, auto_adjust=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        return df.dropna()
    except Exception:
        return None

def compute_rsi(close, period=14):
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def compute_max_drawdown(close):
    peak = close.cummax()
    return (close / peak - 1.0).iloc[-1]

def technical_summary_daily(ticker: str):
    df = fetch_stock_data(ticker)
    if df is None or df.empty: return None
    
    close = df["Close"]
    last_close = close.iloc[-1]
    ma20 = close.rolling(20).mean().iloc[-1]
    ma50 = close.rolling(50).mean().iloc[-1]
    rsi = compute_rsi(close).iloc[-1]
    mdd = compute_max_drawdown(close)
    
    trend = "Bullish" if last_close > ma20 and last_close > ma50 else "Weak"
    risk = "High" if mdd < -0.20 else "Low"
    
    return {
        "status": "ok",
        "last_close": last_close,
        "trend": trend,
        "rsi14": rsi,
        "max_drawdown": mdd,
        "risk": risk
    }

def get_today_price_change(ticker: str):
    try:
        df = yf.download(ticker, period="2d", progress=False)
        if len(df) < 2: return None
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
        
        close = df["Close"]
        prev = close.iloc[-2]
        curr = close.iloc[-1]
        pct = ((curr - prev) / prev) * 100
        return {"price": curr, "pct_change": pct}
    except Exception:
        return None

# =====================================================
# Intent & Extraction
# =====================================================

INTENT_LABELS = [
    "price_now", "news", "sentiment", "fundamentals",
    "technical_full", "compare", "bull", "bear", "risk", "analysis"
]

def extract_tickers(query: str) -> list[str]:
    # Basic regex for tickers
    return re.findall(r"\b[A-Z]{2,5}\b", query.upper())

def pick_primary_ticker(tickers, query):
    return tickers[0] if tickers else None

def predict_intent_with_epoch5(query: str):
    # Use Fine-Tuned Model if available
    if model and tokenizer:
        prompt = f"Classify intent:\n{query}\n\nOptions: {', '.join(INTENT_LABELS)}\nIntent:"
        try:
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            outputs = model.generate(**inputs, max_new_tokens=5, do_sample=False)
            res = tokenizer.decode(outputs[0], skip_special_tokens=True)
            if "Intent:" in res:
                return res.split("Intent:")[-1].strip().lower()
        except Exception:
            pass
    
    # Fallback Rules
    q = query.lower()
    if "news" in q: return "news"
    if "sentiment" in q: return "sentiment"
    if "bull" in q: return "bull"
    if "bear" in q: return "bear"
    if "risk" in q: return "risk"
    if "technical" in q or "rsi" in q: return "technical_full"
    if "price" in q or "quote" in q: return "price_now"
    if "compare" in q or " vs " in q: return "compare"
    if "fundamental" in q: return "fundamentals"
    
    return "analysis" # default

# =====================================================
# Agents
# =====================================================

def bull_agent_pm(ticker, tech):
    return (f"{ticker} Bull Case:\n"
            f"- Trend: {tech['trend']}\n"
            f"- RSI: {tech['rsi14']:.1f} (Momentum supports upside)")

def bear_agent_pm(ticker, tech):
    return (f"{ticker} Bear Case:\n"
            f"- Drawdown: {tech['max_drawdown']:.1%}\n"
            f"- Risk Level: {tech['risk']}")

def risk_agent_pm(ticker, tech):
    return f"{ticker} Risk Profile: {tech['risk']} (Max DD: {tech['max_drawdown']:.1%})"

def coordinator_agent(ticker, tech, bull, bear, risk):
    rec = "Buy" if tech['trend'] == "Bullish" and tech['risk'] == "Low" else "Hold"
    return f"Recommendation: {rec}. {bull.splitlines()[1]}. {bear.splitlines()[1]}."

# =====================================================
# MAIN ENTRY POINT
# =====================================================

def finance_agent(user_query: str):
    """
    Main function called by backend/main.py
    """
    q = user_query.strip()
    
    # 1. Extract Ticker
    tickers = extract_tickers(q)
    ticker = pick_primary_ticker(tickers, q)
    
    if not ticker:
        return "❌ Please mention a valid stock ticker (e.g. AAPL, NVDA)."

    # 2. Detect Intent
    intent = predict_intent_with_epoch5(q)
    
    # 3. Route & Execute
    if intent == "price_now":
        data = get_today_price_change(ticker)
        if data:
            return f"💰 {ticker} Price: ${data['price']:.2f} ({data['pct_change']:+.2f}%)"
        return f"Could not fetch price for {ticker}."

    if intent == "news":
        items = finnhub_company_news(ticker, strict=True)
        return format_news_items(items)

    if intent == "fundamentals":
        f = finnhub_fundamentals_basic(ticker)
        return format_fundamentals(f)

    if intent == "sentiment":
        items = finnhub_company_news(ticker, strict=True)
        # Check if user wants reasoning
        mode = "reasoning" if "why" in q.lower() else "label"
        sent = sentiment_analyzer.analyze(items, ticker, mode=mode)
        return format_sentiment_human(sent, ticker)

    if intent == "technical_full":
        tech = technical_summary_daily(ticker)
        if not tech: return f"Technical data unavailable for {ticker}."
        return (f"📈 Technicals for {ticker}:\n"
                f"- Trend: {tech['trend']}\n"
                f"- RSI: {tech['rsi14']:.1f}\n"
                f"- Risk: {tech['risk']}")

    if intent == "compare":
        if len(tickers) < 2: return "Please provide two tickers to compare (e.g. AAPL vs MSFT)."
        t1, t2 = tickers[:2]
        p1 = get_today_price_change(t1)
        p2 = get_today_price_change(t2)
        if p1 and p2:
            return (f"📊 Comparison:\n"
                    f"{t1}: ${p1['price']:.2f} ({p1['pct_change']:+.2f}%)\n"
                    f"{t2}: ${p2['price']:.2f} ({p2['pct_change']:+.2f}%)")
        return "Could not fetch data for comparison."

    # Default: Full Multi-Agent Analysis
    tech = technical_summary_daily(ticker)
    if not tech: return f"Could not analyze {ticker}."
    
    bull = bull_agent_pm(ticker, tech)
    bear = bear_agent_pm(ticker, tech)
    risk = risk_agent_pm(ticker, tech)
    decision = coordinator_agent(ticker, tech, bull, bear, risk)

    return f"""📊 Multi-Agent Analysis for {ticker}

🟢 Bull:
{bull}

🔴 Bear:
{bear}

⚠️ Risk:
{risk}

📌 Final Recommendation:
{decision}
""".strip()

# --- Test Block (Runs only if executed directly) ---
if __name__ == "__main__":
    # Simulate a query to test local model loading
    print(finance_agent("Analyze AAPL"))