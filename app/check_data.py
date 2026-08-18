"""Data provider diagnostic. Run: python check_data.py

Tests each provider one at a time and tells you exactly which are working,
which are not, and how fresh the prices are.
"""
import datetime as dt
import os

from app.market import PROVIDERS, SYMBOL_ROUTE, TWELVEDATA_KEY, get_candles
from app.config import OANDA_TOKEN

SYMBOL = os.getenv("CHECK_SYMBOL", "XAUUSD")
TIMEFRAME = os.getenv("CHECK_TF", "15m")


def line(char="-"):
    print(char * 72)


print()
line("=")
print(f"  Data check: {SYMBOL} {TIMEFRAME}")
print(f"  Now (UTC): {dt.datetime.now(dt.timezone.utc):%Y-%m-%d %H:%M:%S}")
line("=")

# --- What is configured? -----------------------------------------------------
print("\nCredentials:")
print(f"  OANDA_TOKEN      {'set' if OANDA_TOKEN else 'NOT SET'}")
print(f"  TWELVEDATA_KEY   {'set' if TWELVEDATA_KEY else 'NOT SET'}")
try:
    import yfinance  # noqa: F401
    print("  yfinance         installed")
except ImportError:
    print("  yfinance         NOT INSTALLED  ->  pip install yfinance")

# --- Test each provider individually ----------------------------------------
chain = SYMBOL_ROUTE.get(SYMBOL.upper(), ["yfinance"])
print(f"\nProvider chain for {SYMBOL}: {' -> '.join(chain)}")
line()

now = dt.datetime.now(dt.timezone.utc)

for name in chain:
    print(f"\n{name}")
    try:
        df = PROVIDERS[name].fetch(SYMBOL, TIMEFRAME, 300)
        tkr = df.attrs.get("ticker")
        if df.empty:
            print("   FAILED: returned an empty frame")
            continue
        last_t = df.index[-1].to_pydatetime()
        age_min = (now - last_t).total_seconds() / 60
        price = float(df["close"].iloc[-1])

        print(f"   OK        {len(df)} candles" + (f"   ticker: {tkr}" if tkr else ""))
        print(f"   price     {price:,.2f}")
        print(f"   last bar  {last_t:%Y-%m-%d %H:%M} UTC")
        print(f"   age       {age_min:.0f} minutes")

        if age_min < 20:
            print("   -> fresh")
        elif age_min < 90:
            print("   -> delayed feed, or the candle is still forming")
        else:
            print("   -> STALE. Market is probably closed.")
    except Exception as e:
        print(f"   FAILED: {type(e).__name__}: {e}")

# --- What the app actually serves -------------------------------------------
line()
df, source = get_candles(SYMBOL, TIMEFRAME)
last_t = df.index[-1].to_pydatetime()
age_min = (now - last_t).total_seconds() / 60

print(f"\nYour app is using: {source.upper()}")
print(f"  price {float(df['close'].iloc[-1]):,.2f}   last bar {last_t:%H:%M} UTC "
      f"({age_min:.0f} min ago)")

if source == "synthetic":
    print("\n  >> These prices are FAKE. Every provider above failed.")
    print("     Fix the failures listed above before going further.")
line("=")
print()
