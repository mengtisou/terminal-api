"""Candle cache and provider request accounting.

Two problems this solves:

1. Free data tiers are capped per day (Twelve Data: 800). A 5-second poll
   would exhaust that in under an hour of market time.
2. Several browser tabs, or a background scanner, would each hit the provider
   independently for identical data.

So: cache candles briefly, serve everyone from it, and count real upstream
calls so the UI can show what is left.
"""
from __future__ import annotations

import datetime as dt
import threading
from collections import defaultdict

import pandas as pd

_lock = threading.Lock()
_cache: dict[tuple, tuple[dt.datetime, pd.DataFrame, str]] = {}
_calls: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))

# How long a cached frame stays fresh, per timeframe. Roughly a fifth of the
# candle period - long enough to absorb a fast poll, short enough that the
# forming candle still updates visibly.
DEFAULT_TTL = {
    "1m": 12, "5m": 30, "15m": 45, "30m": 60,
    "1h": 90, "4h": 300, "1d": 900,
}

# Runtime override, set from the UI. Lower = fresher prices, more upstream
# calls. This is the only real lever on how "live" the chart feels.
_ttl_override: int | None = None
TTL_SECONDS = dict(DEFAULT_TTL)


def set_ttl(seconds: int | None) -> None:
    """None restores the per-timeframe defaults."""
    global _ttl_override
    with _lock:
        _ttl_override = seconds


_budget_mode = False


def set_budget_mode(on: bool) -> None:
    """Spread the remaining daily quota over the hours left in the day.

    Without this, any fixed refresh rate either wastes quota early or runs out
    before the session ends. With it the chart stays live all day and simply
    updates less often when the budget is tight.
    """
    global _budget_mode
    with _lock:
        _budget_mode = on


def _metered_provider() -> str | None:
    today = _calls.get(_today(), {})
    for name, limit in DAILY_LIMIT.items():
        if limit and name in today:
            return name
    return None


def budget_ttl() -> int | None:
    """Seconds per request that would exactly spend the remaining quota."""
    provider = _metered_provider()
    if not provider:
        return None
    limit = DAILY_LIMIT[provider]
    used = _calls[_today()][provider]
    remaining = max(0, limit - used)

    now = dt.datetime.now(dt.timezone.utc)
    seconds_left = max(60, (24 * 3600) - (now.hour * 3600 + now.minute * 60 + now.second))

    if remaining <= 0:
        return 900          # quota gone - back right off, do not hammer a 429
    # Keep 10% in reserve for signals, news tagging and manual refreshes.
    usable = remaining * 0.9
    return max(5, round(seconds_left / usable))


def ttl_for(timeframe: str, source: str | None = None) -> int:
    # Unmetered providers: cache briefly so browser requests return from cache
    # instantly while the server keeps data fresh in the background.
    if source and source.split(":")[0] in UNMETERED:
        # MT5 is a local IPC call (~0ms) - 1s is fine
        # OANDA takes 2-3s per request - cache 2s so polls are instant
        src = source.split(":")[0]
        return 1 if src == "mt5" else 2
    if _budget_mode:
        b = budget_ttl()
        if b is not None:
            return b
    if _ttl_override is not None:
        return max(1, _ttl_override)
    return DEFAULT_TTL.get(timeframe, 45)


def projected_daily(timeframe: str, hours_open: float = 24.0) -> int:
    """Upstream calls per day at the current TTL, if the chart runs the whole
    session. This is the number that blows a free tier, not the poll rate."""
    return round(hours_open * 3600 / ttl_for(timeframe))

# Documented free-tier daily limits, for the UI.
DAILY_LIMIT = {"twelvedata": 800, "finnhub": 86400, "oanda": None,
               "yfinance": None, "binance": None, "synthetic": None,
               "mt5": None}

# Providers with no request cost. Cached only briefly, so the chart is as live
# as the terminal is.
# No request limits on these providers - cache briefly so the browser
# gets instant responses while the server stays close to real-time.
UNMETERED = {"mt5", "oanda", "binance", "synthetic"}


def _today() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d")


def get(symbol: str, timeframe: str) -> tuple[pd.DataFrame, str] | None:
    key = (symbol.upper(), timeframe)
    with _lock:
        entry = _cache.get(key)
        if not entry:
            return None
        stored, df, source = entry
        age = (dt.datetime.now(dt.timezone.utc) - stored).total_seconds()
        if age > ttl_for(timeframe, source):
            return None
        return df, source


def put(symbol: str, timeframe: str, df: pd.DataFrame, source: str) -> None:
    with _lock:
        _cache[(symbol.upper(), timeframe)] = (
            dt.datetime.now(dt.timezone.utc), df, source)


def record(provider: str) -> None:
    """Count one real upstream request."""
    with _lock:
        _calls[_today()][provider.split(":")[0]] += 1


def usage() -> dict:
    day = _today()
    with _lock:
        today = dict(_calls.get(day, {}))
    out = {}
    for provider, used in today.items():
        limit = DAILY_LIMIT.get(provider)
        out[provider] = {
            "used": used,
            "limit": limit,
            "remaining": (limit - used) if limit else None,
            "pct": round(used / limit * 100, 1) if limit else None,
        }
    return {"date": day, "providers": out,
            "cached_series": len(_cache),
            "ttl_override": _ttl_override,
            "budget_mode": _budget_mode,
            "budget_ttl": budget_ttl(),
            "ttl_seconds": {tf: ttl_for(tf) for tf in DEFAULT_TTL},
            "ttl_defaults": DEFAULT_TTL}


def clear() -> None:
    with _lock:
        _cache.clear()


def invalidate(symbol: str, timeframe: str) -> bool:
    """Drop one frame so the next read refetches. Returns True if it existed.

    Used when the user changes timeframe: they are about to stare at a chart
    they have not seen for a while, and serving them a frame that is up to a
    full TTL old means the candle they are looking at can be missing a spike
    that already happened.
    """
    with _lock:
        return _cache.pop((symbol.upper(), timeframe), None) is not None
