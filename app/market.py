"""OHLCV fetching plus session/staleness detection.

Every provider returns the same DataFrame shape:
    index: UTC DatetimeIndex (candle open time)
    columns: open, high, low, close, volume

Providers, easiest first:
  yfinance    - no API key, no signup. Gold, forex, crypto, oil. Start here.
  twelvedata  - free key, 800 req/day. Cleaner FX/metals data than Yahoo.
  oanda       - broker feed, needs a free practice account. Best for FX/metals.
  binance     - crypto only, blocked in some countries.
  synthetic   - fake data for offline development.
"""
from __future__ import annotations

import datetime as dt
import logging
import os
from pathlib import Path
from typing import Protocol

import httpx
import numpy as np
import pandas as pd

from . import cache
from .config import BINANCE_BASE, OANDA_BASE, OANDA_TOKEN, settings

TF_MINUTES = {
    "1m": 1, "2m": 2, "3m": 3, "5m": 5, "10m": 10, "15m": 15, "20m": 20,
    "30m": 30, "45m": 45,
    "1h": 60, "2h": 120, "3h": 180, "4h": 240, "6h": 360, "8h": 480, "12h": 720,
    "1d": 1440, "3d": 4320, "1w": 10080, "1M": 43200,
}

# Timeframes shown as pills in the UI; the rest live in the dropdown.
QUICK_TF = ["1m", "5m", "15m", "30m", "1h", "4h", "1d"]

# What each provider serves natively. Anything else is built by resampling a
# smaller native timeframe, so the UI can offer the full list regardless.
NATIVE_TF = {
    "mt5": {"1m", "2m", "3m", "5m", "10m", "15m", "20m", "30m",
            "1h", "2h", "3h", "4h", "6h", "8h", "12h", "1d", "1w", "1M"},
    "twelvedata": {"1m", "5m", "15m", "30m", "45m", "1h", "2h", "4h", "8h", "1d", "1w", "1M"},
    "binance": {"1m", "3m", "5m", "15m", "30m", "1h", "2h", "4h", "6h", "8h",
                "12h", "1d", "3d", "1w", "1M"},
    "oanda": {"1m", "5m", "15m", "30m", "1h", "2h", "3h", "4h", "6h", "8h", "12h", "1d", "1w", "1M"},
    "yfinance": {"1m", "2m", "5m", "15m", "30m", "1h", "1d", "1w", "1M"},
    "synthetic": set(TF_MINUTES),
}


def _resample_source(provider: str, timeframe: str) -> str | None:
    """Largest native timeframe that divides the requested one exactly."""
    native = NATIVE_TF.get(provider, set())
    want = TF_MINUTES[timeframe]
    options = [tf for tf in native
               if TF_MINUTES[tf] < want and want % TF_MINUTES[tf] == 0]
    return max(options, key=lambda t: TF_MINUTES[t]) if options else None


PANDAS_RULE = {"1M": "MS", "1w": "W-MON", "1d": "1D"}


def _to_rule(timeframe: str) -> str:
    return PANDAS_RULE.get(timeframe, f"{TF_MINUTES[timeframe]}min")

ALLOW_SYNTHETIC = os.getenv("ALLOW_SYNTHETIC", "1") == "1"
TWELVEDATA_KEY = os.getenv("TWELVEDATA_KEY", "")

log = logging.getLogger(__name__)


class DataUnavailable(RuntimeError):
    """Raised when no real provider can serve this symbol and synthetic data
    is disallowed. Callers must surface this, never swallow it."""


class Provider(Protocol):
    def fetch(self, symbol: str, timeframe: str, limit: int) -> pd.DataFrame: ...


def _resample(df: pd.DataFrame, minutes_or_rule) -> pd.DataFrame:
    """Aggregate into a larger timeframe the provider does not serve natively."""
    rule = (minutes_or_rule if isinstance(minutes_or_rule, str)
            else f"{minutes_or_rule}min")
    return (
        df.resample(rule)
        .agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"})
        .dropna()
    )


class YFinanceProvider:
    """Yahoo Finance. No key, no signup, works everywhere.

    Caveat: FX and metals quotes are delayed roughly 15 minutes and volume on
    FX pairs is unreliable. Crypto is close to real time. Good enough to build
    against; move to OANDA or Twelve Data before trading anything.
    """

    SYMBOLS = {
        "XAUUSD": "GC=F",      # COMEX gold futures - better intraday depth
        "XAGUSD": "SI=F",
        "USOIL": "CL=F",
        "BTCUSDT": "BTC-USD",
        "ETHUSDT": "ETH-USD",
        "EURUSD": "EURUSD=X",
        "GBPUSD": "GBPUSD=X",
        "USDJPY": "JPY=X",
    }
    INTERVALS = {"1m": "1m", "5m": "5m", "15m": "15m", "30m": "30m", "1h": "60m", "1d": "1d"}
    # Yahoo caps intraday history by interval. 1h allows ~730 days, which is
    # what the event-reaction engine needs to cover 8+ FOMC meetings.
    PERIODS = {"1m": "7d", "5m": "60d", "15m": "60d", "30m": "60d",
               "1h": "730d", "1d": "5y"}

    # Yahoo does NOT carry spot XAU/USD - XAUUSD=X and XAU=X both 404.
    # GC=F is COMEX futures, which trades at a premium to spot (the basis
    # covers interest and storage to delivery). Expect a $30-80 gap versus a
    # broker's spot feed. For true spot, use OANDA or Twelve Data.
    ALTERNATES = {
        "XAUUSD": ["GC=F"],
        "XAGUSD": ["SI=F"],
        "USOIL": ["CL=F"],
        "BTCUSDT": ["BTC-USD"],
        "ETHUSDT": ["ETH-USD"],
        "EURUSD": ["EURUSD=X", "EUR=X"],
        "GBPUSD": ["GBPUSD=X", "GBP=X"],
        "USDJPY": ["JPY=X", "USDJPY=X"],
    }
    # Instruments where the Yahoo proxy is not the same thing the user asked
    # for. Surfaced in the UI so nobody mistakes futures for spot.
    PROXY_NOTE = {
        "XAUUSD": "futures (GC=F), not spot",
        "XAGUSD": "futures (SI=F), not spot",
        "USOIL": "WTI futures (CL=F)",
    }

    def fetch(self, symbol: str, timeframe: str, limit: int) -> pd.DataFrame:
        import yfinance as yf

        # yfinance prints its own 404 noise straight to stdout; quiet it.
        logging.getLogger("yfinance").setLevel(logging.CRITICAL)

        # 4h is not a native Yahoo interval - pull 1h and resample.
        native_tf = "1h" if timeframe == "4h" else timeframe
        if native_tf not in self.INTERVALS:
            raise ValueError(f"yfinance cannot serve {timeframe}")

        interval = self.INTERVALS[native_tf]
        period = self.PERIODS[native_tf]
        candidates = self.ALTERNATES.get(
            symbol.upper(), [self.SYMBOLS.get(symbol.upper(), symbol)]
        )

        raw, used, errors = None, None, []
        for ticker in candidates:
            try:
                # Ticker.history() returns single-level columns on every
                # yfinance version. yf.download() changed shape in 1.x, so
                # avoid it.
                got = yf.Ticker(ticker).history(
                    period=period, interval=interval, auto_adjust=False
                )
                if got is not None and not got.empty:
                    raw, used = got, ticker
                    break
                errors.append(f"{ticker}: empty")
            except Exception as e:
                errors.append(f"{ticker}: {type(e).__name__}: {e}")

        if raw is None:
            raise DataUnavailable(f"yfinance found no data - {'; '.join(errors)}")

        log.info("yfinance served %s via ticker %s", symbol, used)

        # Defensive: some versions still hand back MultiIndex columns.
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)

        raw.columns = [str(c).strip().lower() for c in raw.columns]

        missing = {"open", "high", "low", "close"} - set(raw.columns)
        if missing:
            raise DataUnavailable(
                f"yfinance response missing {missing}; got {list(raw.columns)}"
            )
        if "volume" not in raw.columns:
            raw["volume"] = 0.0

        df = raw[["open", "high", "low", "close", "volume"]].copy()

        # Yahoo returns tz-aware for intraday, naive for daily.
        df.index = pd.to_datetime(df.index, utc=True)
        df = df.dropna().sort_index()

        if timeframe == "4h":
            df = _resample(df, 240)

        out = df.tail(limit).astype(float)
        out.attrs["ticker"] = used
        note = self.PROXY_NOTE.get(symbol.upper())
        if note:
            out.attrs["proxy_note"] = note
            log.warning("%s served as %s - %s", symbol, used, note)
        return out


class TwelveDataProvider:
    """Twelve Data. Free tier: 800 requests/day, 8 per minute.

    Get a key at twelvedata.com - about a minute, no card needed.
    Cleaner metals and FX data than Yahoo.
    """

    SYMBOLS = {
        "XAUUSD": "XAU/USD", "XAGUSD": "XAG/USD",
        "BTCUSDT": "BTC/USD", "ETHUSDT": "ETH/USD",
        "EURUSD": "EUR/USD", "GBPUSD": "GBP/USD", "USDJPY": "USD/JPY",
    }
    INTERVALS = {"1m": "1min", "5m": "5min", "15m": "15min", "30m": "30min",
                 "1h": "1h", "4h": "4h", "1d": "1day"}

    def fetch(self, symbol: str, timeframe: str, limit: int) -> pd.DataFrame:
        if not TWELVEDATA_KEY:
            raise DataUnavailable("TWELVEDATA_KEY not set")

        r = httpx.get(
            "https://api.twelvedata.com/time_series",
            params={
                "symbol": self.SYMBOLS.get(symbol.upper(), symbol),
                "interval": self.INTERVALS[timeframe],
                "outputsize": min(limit, 5000),
                "apikey": TWELVEDATA_KEY,
                "timezone": "UTC",
            },
            timeout=20,
        )
        r.raise_for_status()
        payload = r.json()

        if payload.get("status") == "error":
            raise DataUnavailable(f"twelvedata: {payload.get('message')}")

        df = pd.DataFrame(payload["values"])
        df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
        df = df.set_index("datetime").sort_index()

        if "volume" not in df.columns:
            df["volume"] = 0.0  # FX and metals often have no volume field

        return df[["open", "high", "low", "close", "volume"]].astype(float).tail(limit)


class BinanceProvider:
    """Public spot klines. No API key. Blocked in some countries."""

    def fetch(self, symbol: str, timeframe: str, limit: int) -> pd.DataFrame:
        r = httpx.get(
            f"{BINANCE_BASE}/api/v3/klines",
            params={"symbol": symbol.upper(), "interval": timeframe, "limit": limit},
            timeout=15,
        )
        r.raise_for_status()
        df = pd.DataFrame(
            r.json(),
            columns=["open_time", "open", "high", "low", "close", "volume",
                     "close_time", "qav", "trades", "tbb", "tbq", "ignore"],
        )
        df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
        return df.set_index("open_time")[["open", "high", "low", "close", "volume"]].astype(float)


class OandaProvider:
    """Forex, metals and indices. Needs OANDA_TOKEN from a practice account."""

    GRAN = {"1m": "M1", "2m": "M2", "3m": "M3", "5m": "M5",
            "10m": "M10", "15m": "M15", "30m": "M30",
            "1h": "H1", "2h": "H2", "3h": "H3", "4h": "H4",
            "6h": "H6", "8h": "H8", "12h": "H12",
            "1d": "D", "1w": "W", "1M": "M"}
    SYMBOLS = {"XAUUSD": "XAU_USD", "XAGUSD": "XAG_USD", "EURUSD": "EUR_USD",
               "GBPUSD": "GBP_USD", "USDJPY": "USD_JPY", "USOIL": "WTICO_USD"}

    def live_price(self, symbol: str) -> dict | None:
        """Current bid/ask from OANDA's pricing endpoint.

        This is the RIGHT endpoint for real-time price - it returns the
        current tradeable quote, not the last completed candle. Candles
        only update on close; this updates continuously.
        """
        instrument = self.SYMBOLS.get(symbol.upper(), symbol)
        base = OANDA_BASE.rstrip("/").removesuffix("/v3")
        account = os.getenv("OANDA_ACCOUNT", "")
        if not account:
            return None
        try:
            r = httpx.get(
                f"{base}/v3/accounts/{account}/pricing",
                headers={"Authorization": f"Bearer {OANDA_TOKEN}"},
                params={"instruments": instrument},
                timeout=8,
            )
            r.raise_for_status()
            prices = r.json().get("prices", [])
            if not prices:
                return None
            p = prices[0]
            bid = float(p["bids"][0]["price"]) if p.get("bids") else None
            ask = float(p["asks"][0]["price"]) if p.get("asks") else None
            if bid is None or ask is None:
                return None
            return {
                "bid": bid, "ask": ask,
                "mid": round((bid + ask) / 2, 5),
                "spread": round(ask - bid, 5),
                "time": p.get("time"),
                "tradeable": p.get("tradeable", True),
            }
        except Exception as e:
            log.debug("oanda live price failed: %s", e)
            return None

    def fetch(self, symbol: str, timeframe: str, limit: int) -> pd.DataFrame:
        instrument = self.SYMBOLS.get(symbol.upper(), symbol)
        # Strip /v3 suffix if the user included it in OANDA_BASE already.
        base = OANDA_BASE.rstrip("/").removesuffix("/v3")
        r = httpx.get(
            f"{base}/v3/instruments/{instrument}/candles",
            headers={"Authorization": f"Bearer {OANDA_TOKEN}"},
            params={"granularity": self.GRAN[timeframe], "count": limit, "price": "M"},
            timeout=15,
        )
        r.raise_for_status()
        rows = [
            {
                "open_time": pd.to_datetime(c["time"], utc=True),
                "open": float(c["mid"]["o"]), "high": float(c["mid"]["h"]),
                "low": float(c["mid"]["l"]), "close": float(c["mid"]["c"]),
                "volume": float(c["volume"]),
            }
            for c in r.json()["candles"] if c["complete"]
        ]
        if not rows:
            raise DataUnavailable(f"oanda returned no complete candles for {instrument}")
        return pd.DataFrame(rows).set_index("open_time")


class SyntheticProvider:
    """Deterministic fake data for offline development.
    Never let this reach a user as if it were real."""

    def fetch(self, symbol: str, timeframe: str, limit: int) -> pd.DataFrame:
        rng = np.random.default_rng(abs(hash((symbol, timeframe))) % (2**32))
        step = dt.timedelta(minutes=TF_MINUTES[timeframe])
        end = dt.datetime.now(dt.timezone.utc).replace(second=0, microsecond=0)
        idx = pd.DatetimeIndex([end - step * i for i in range(limit)][::-1])

        base = {"XAUUSD": 4375.0, "BTCUSDT": 63000.0}.get(symbol.upper(), 100.0)
        close = base + np.cumsum(rng.normal(0, base * 0.0006, limit))
        spread = np.abs(rng.normal(0, base * 0.0004, limit))
        df = pd.DataFrame(
            {
                "open": close - rng.normal(0, base * 0.0002, limit),
                "high": close + spread,
                "low": close - spread,
                "close": close,
                "volume": rng.lognormal(6, 0.5, limit),
            },
            index=idx,
        )
        df["high"] = df[["high", "open", "close"]].max(axis=1)
        df["low"] = df[["low", "open", "close"]].min(axis=1)
        return df


try:
    from .mt5_provider import MT5Provider
except Exception as _e:  # pragma: no cover - non-Windows
    MT5Provider = None
    log.debug("MT5 provider unavailable: %s", _e)


PROVIDERS: dict[str, Provider] = {
    "yfinance": YFinanceProvider(),
    "twelvedata": TwelveDataProvider(),
    "binance": BinanceProvider(),
    "oanda": OandaProvider(),
    "synthetic": SyntheticProvider(),
}
if MT5Provider is not None:
    PROVIDERS["mt5"] = MT5Provider()

# Ordered fallback chain per symbol. First one that works wins.
# MT5 first everywhere it can serve: it is a real broker feed, has no request
# limit, and reaches the broker over a path that is not blocked here.
SYMBOL_ROUTE: dict[str, list[str]] = {
    "XAUUSD": ["oanda", "mt5", "twelvedata", "yfinance"],
    "XAGUSD": ["oanda", "mt5", "twelvedata", "yfinance"],
    "EURUSD": ["oanda", "mt5", "twelvedata", "yfinance"],
    "GBPUSD": ["oanda", "mt5", "twelvedata", "yfinance"],
    "USDJPY": ["oanda", "mt5", "twelvedata", "yfinance"],
    "USOIL": ["oanda", "mt5", "yfinance"],
    "BTCUSDT": ["mt5", "binance", "yfinance", "twelvedata"],
    "ETHUSDT": ["mt5", "binance", "yfinance", "twelvedata"],
}
DEFAULT_ROUTE = ["oanda", "mt5", "yfinance"]

# --- runtime routing overrides ----------------------------------------------
# Set from the UI. Persisted so a restart keeps your choice.
# Precedence: UI override > DATA_CHAIN_<SYMBOL> in .env > built-in default.

_ROUTE_FILE = Path(os.getenv("STATE_DIR",
                            Path(__file__).resolve().parent.parent)) / "data_routes.json"
_route_overrides: dict[str, list[str]] = {}


def _load_routes() -> None:
    global _route_overrides
    try:
        import json
        _route_overrides = json.loads(_ROUTE_FILE.read_text())
        log.info("loaded data route overrides: %s", _route_overrides)
    except (OSError, ValueError):
        _route_overrides = {}


def _save_routes() -> None:
    try:
        import json
        _ROUTE_FILE.write_text(json.dumps(_route_overrides, indent=1))
    except OSError as e:
        log.warning("could not persist data routes: %s", e)


_load_routes()


def chain_for(symbol: str) -> list[str]:
    key = symbol.upper()
    if key in _route_overrides:
        return list(_route_overrides[key])
    env = os.getenv(f"DATA_CHAIN_{key}", "")
    if env.strip():
        return [p.strip() for p in env.split(",") if p.strip()]
    return SYMBOL_ROUTE.get(key, DEFAULT_ROUTE)


def set_route(symbol: str, providers_list: list[str]) -> dict:
    """Reorder or restrict which providers serve a symbol. Empty list clears
    the override."""
    key = symbol.upper()
    for name in providers_list:
        if name not in PROVIDERS:
            raise ValueError(f"unknown provider '{name}'. options: {list(PROVIDERS)}")
    if providers_list:
        _route_overrides[key] = providers_list
    else:
        _route_overrides.pop(key, None)
    _save_routes()
    cache.clear()      # old provider's candles must not linger
    return routes()


def clear_routes() -> dict:
    _route_overrides.clear()
    _save_routes()
    cache.clear()
    return routes()


def routes() -> dict:
    """Current chain per symbol, plus which providers are usable right now."""
    symbols = sorted(set(SYMBOL_ROUTE) | set(_route_overrides))
    return {
        "providers": {n: ("ready" if _configured(n) else "not configured")
                      for n in PROVIDERS if n != "synthetic"},
        "chains": {s: chain_for(s) for s in symbols},
        "overrides": _route_overrides,
        "source": "ui" if _route_overrides else "default",
    }


def _configured(name: str) -> bool:
    if name == "mt5":
        return "mt5" in PROVIDERS and PROVIDERS["mt5"].available()
    if name == "oanda":
        return bool(OANDA_TOKEN)
    if name == "twelvedata":
        return bool(TWELVEDATA_KEY)
    return True


def get_candles(
    symbol: str, timeframe: str, limit: int | None = None
) -> tuple[pd.DataFrame, str]:
    """Returns (candles, data_source).

    Walks the provider chain for this symbol and returns the first success.
    data_source is carried into the snapshot and the UI, so a synthetic candle
    can never be mistaken for a real one. Set ALLOW_SYNTHETIC=0 in production
    to make total failure loud instead of silent.
    """
    limit = limit or settings.candle_limit
    if timeframe not in TF_MINUTES:
        raise ValueError(f"unsupported timeframe: {timeframe}")

    # Serve from cache when fresh. This is what makes a fast UI poll safe on a
    # metered free tier - the browser can ask every 5s while the provider is
    # only hit once per TTL window.
    hit = cache.get(symbol, timeframe)
    if hit is not None:
        return hit

    chain = chain_for(symbol)
    errors = []

    for name in chain:
        if not _configured(name):
            log.debug("%s skipped for %s: not configured", name, symbol)
            continue
        try:
            if timeframe in NATIVE_TF.get(name, set()):
                df = PROVIDERS[name].fetch(symbol, timeframe, limit)
                built_from = None
            else:
                src = _resample_source(name, timeframe)
                if src is None:
                    raise DataUnavailable(f"{name} cannot build {timeframe}")
                factor = TF_MINUTES[timeframe] // TF_MINUTES[src]
                raw = PROVIDERS[name].fetch(symbol, src, min(limit * factor + factor, 5000))
                df = _resample(raw, _to_rule(timeframe)).tail(limit)
                df.attrs.update(raw.attrs)
                built_from = src

            if df.empty:
                raise DataUnavailable("empty frame")
            label = name
            if built_from:
                log.info("%s %s built from %s candles", symbol, timeframe, built_from)
            if df.attrs.get("ticker") and df.attrs["ticker"] != symbol:
                label = f"{name}:{df.attrs['ticker']}"
            if df.attrs.get("proxy_note"):
                label += " (proxy)"
            cache.record(name)
            cache.put(symbol, timeframe, df, label)
            log.info("%s %s served by %s (%d candles)", symbol, timeframe, label, len(df))
            return df, label
        except Exception as e:
            errors.append(f"{name}: {type(e).__name__}: {e}")
            log.warning("provider %s failed for %s: %s", name, symbol, e)

    detail = " | ".join(errors) if errors else "no provider configured"
    if not ALLOW_SYNTHETIC:
        raise DataUnavailable(f"all providers failed for {symbol} - {detail}")

    log.error("falling back to SYNTHETIC data for %s - %s", symbol, detail)
    df = PROVIDERS["synthetic"].fetch(symbol, timeframe, limit)
    cache.put(symbol, timeframe, df, "synthetic")
    return df, "synthetic"


def background_refresh(symbol: str, timeframe: str) -> None:
    """Proactively refresh an unmetered provider so the cache is always warm.

    Called after serving a cached response. By the time the browser polls
    again (1s later) the new data is already in the cache, so the user
    sees a fresh price with zero waiting.
    """
    import threading
    def _work():
        try:
            chain = chain_for(symbol)
            for name in chain:
                if not _configured(name):
                    continue
                src = cache.get(symbol, timeframe)
                if src and src[1].split(":")[0] in cache.UNMETERED:
                    # Force a fresh fetch by clearing just this entry
                    with cache._lock:
                        cache._cache.pop((symbol.upper(), timeframe), None)
                    get_candles(symbol, timeframe)
                    break
        except Exception:
            pass
    threading.Thread(target=_work, daemon=True).start()


def session_state(df: pd.DataFrame, timeframe: str) -> dict:
    """Detect stale / frozen data - the weekend-gold case.

    Runs BEFORE any model call. If the market is closed there is nothing to
    reason about and you should not spend a token on it.
    """
    if df.empty:
        return {"state": "no_data", "data_stale": True, "age_seconds": None}

    now = dt.datetime.now(dt.timezone.utc)
    last = df.index[-1].to_pydatetime()
    age = (now - last).total_seconds()

    tolerance = TF_MINUTES[timeframe] * 60 * 2
    flat = float(df["close"].tail(20).std()) == 0.0

    stale = age > max(tolerance, settings.risk.max_data_age_seconds) or flat
    return {
        "state": "closed" if stale else "open",
        "data_stale": stale,
        "age_seconds": round(age),
        "last_candle_utc": last.isoformat(),
        "recent_volume": round(float(df["volume"].tail(10).sum()), 2),
    }
