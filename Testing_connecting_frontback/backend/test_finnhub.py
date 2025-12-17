import finnhub
import os
from datetime import date, timedelta

# 1. Try with a KNOWN valid key structure (I suspect your key was pasted twice)
# Let's try splitting that long string you had. 
# Usually "d4tom21r01qnn6lm76sg" is one key.
candidate_key = "d514ts1r01qjia5bbe50d514ts1r01qjia5bbe5g" 

print(f"🔑 Testing with Key: {candidate_key}")

try:
    client = finnhub.Client(api_key=candidate_key)
    
    # 2. Define dates
    today = date.today()
    start = today - timedelta(days=7)
    
    print(f"📅 Fetching AAPL news from {start} to {today}...")
    
    # 3. Fetch News
    news = client.company_news("AAPL", _from=start.isoformat(), to=today.isoformat())
    
    if news:
        print(f"\n✅ SUCCESS! Found {len(news)} articles.")
        print("First headline:", news[0]['headline'])
    else:
        print("\n❌ Connected, but returned 0 items. (Account limit reached?)")

except Exception as e:
    print(f"\n❌ CRITICAL ERROR: {e}")
