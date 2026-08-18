"""MetaTrader 5 data provider.

The best option when broker APIs are unreachable: MT5 talks to your broker's
servers, and this module talks to the MT5 terminal running on the same machine.
Nothing crosses a blocked network path.

  - Real broker spot feed (the same prices you would trade on)
  - No request limits - it is a local IPC call
  - Years of intraday history, which the event-reaction engine needs
  - Tick-level data available

Requirements: Windows, MetaTrader 5 installed and LOGGED IN, and
`pip install MetaTrader5`. The package is Windows-only; on Linux and macOS
this provider reports unavailable and the chain falls through.
"""
from __future__ import annotations

import datetime as dt
import logging
import os
import threading

import pandas as pd

log = logging.getLogger(__name__)

_lock = threading.Lock()
_ready = False
_symbol_cache: dict[str, str | None] = {}

# --- server time offset ------------------------------------------------------
# MT5 reports candle and tick times in the BROKER's server timezone, not UTC.
# Most brokers run UTC+2 or UTC+3 (EET, with DST). Treating those stamps as UTC
# shifts every candle by hours - which then shows up as a wrong clock on the
# chart, a wrong session state, and event-reaction windows landing on the wrong
# bars. So measure the offset from a live tick and subtract it.
_offset_seconds: int | None = None
_offset_checked: float = 0.0
_OFFSET_TTL = 3600.0        # re-measure hourly; brokers shift with DST


def server_offset(force: bool = False) -> int:
    """Seconds the broker's clock runs ahead of UTC. Rounded to a half hour."""
    global _offset_seconds, _offset_checked
    import time as _time

    if (not force and _offset_seconds is not None
            and _time.time() - _offset_checked < _OFFSET_TTL):
        return _offset_seconds

    mt5 = _mt5()
    if mt5 is None or not connect()[0]:
        return _offset_seconds or 0

    # A live tick carries the server's own clock. Compare it to real UTC.
    for want in ("EURUSD", "XAUUSD", "BTCUSD"):
        name = resolve_symbol(want)
        if not name:
            continue
        tick = mt5.symbol_info_tick(name)
        if not tick or not tick.time:
            continue
        raw = tick.time - _time.time()
        # Only trust it while the market is live; a stale weekend tick would
        # otherwise be read as a huge offset.
        if abs(raw) > 24 * 3600:
            continue
        _offset_seconds = int(round(raw / 1800.0) * 1800)
        _offset_checked = _time.time()
        log.info("MT5 server clock is UTC%+d:%02d",
                 _offset_seconds // 3600, abs(_offset_seconds % 3600) // 60)
        return _offset_seconds

    return _offset_seconds or 0

# Brokers decorate symbol names: XAUUSD, XAUUSDm, XAUUSD.a, XAUUSD_i, GOLD...
# Candidates are tried in order and the first one the terminal knows wins.
CANDIDATES = {
    "XAUUSD": ["XAUUSD", "XAUUSDm", "XAUUSD.a", "XAUUSD_i", "XAUUSD.", "GOLD", "GOLDmicro"],
    "XAGUSD": ["XAGUSD", "XAGUSDm", "XAGUSD.a", "SILVER"],
    "EURUSD": ["EURUSD", "EURUSDm", "EURUSD.a", "EURUSD_i"],
    "GBPUSD": ["GBPUSD", "GBPUSDm", "GBPUSD.a"],
    "USDJPY": ["USDJPY", "USDJPYm", "USDJPY.a"],
    "USOIL": ["USOIL", "WTI", "XTIUSD", "CrudeOIL", "USOUSD"],
    "BTCUSDT": ["BTCUSD", "BTCUSDm", "BTCUSDT", "BTCUSD.a"],
    "ETHUSDT": ["ETHUSD", "ETHUSDm", "ETHUSDT"],
}


def _mt5():
    """Import lazily so non-Windows machines are not broken by the import."""
    try:
        import MetaTrader5 as mt5
        return mt5
    except ImportError:
        return None


def _timeframes(mt5):
    return {
        "1m": mt5.TIMEFRAME_M1, "5m": mt5.TIMEFRAME_M5,
        "15m": mt5.TIMEFRAME_M15, "30m": mt5.TIMEFRAME_M30,
        "1h": mt5.TIMEFRAME_H1, "4h": mt5.TIMEFRAME_H4,
        "1d": mt5.TIMEFRAME_D1,
    }


def connect() -> tuple[bool, str]:
    """Attach to the running terminal. Idempotent."""
    global _ready
    mt5 = _mt5()
    if mt5 is None:
        return False, "MetaTrader5 package not installed (pip install MetaTrader5, Windows only)"

    with _lock:
        if _ready and mt5.terminal_info() is not None:
            return True, "already connected"

        path = os.getenv("MT5_PATH") or None
        login = os.getenv("MT5_LOGIN")
        kwargs = {"path": path} if path else {}
        if login and os.getenv("MT5_PASSWORD") and os.getenv("MT5_SERVER"):
            kwargs.update(login=int(login),
                          password=os.getenv("MT5_PASSWORD"),
                          server=os.getenv("MT5_SERVER"))

        if not mt5.initialize(**kwargs):
            code, msg = mt5.last_error()
            _ready = False
            return False, f"initialize failed ({code}): {msg}. Is MT5 open and logged in?"

        _ready = True
        info = mt5.terminal_info()
        acct = mt5.account_info()
        return True, (f"{info.company if info else 'terminal'} · "
                      f"account {acct.login if acct else 'n/a'} "
                      f"({acct.server if acct else ''})")


def resolve_symbol(symbol: str) -> str | None:
    """Find what this broker calls the instrument, and make it selectable."""
    key = symbol.upper()
    if key in _symbol_cache:
        return _symbol_cache[key]

    mt5 = _mt5()
    if mt5 is None:
        return None

    for name in CANDIDATES.get(key, [key]):
        if mt5.symbol_info(name) is not None:
            mt5.symbol_select(name, True)   # must be in Market Watch to serve data
            _symbol_cache[key] = name
            log.info("MT5 resolved %s -> %s", key, name)
            return name

    # Last resort: scan everything the broker offers for a prefix match.
    stem = key[:6]
    for s in (mt5.symbols_get() or []):
        if s.name.upper().startswith(stem):
            mt5.symbol_select(s.name, True)
            _symbol_cache[key] = s.name
            log.info("MT5 resolved %s -> %s (scan)", key, s.name)
            return s.name

    _symbol_cache[key] = None
    return None


class MT5Provider:
    name = "mt5"

    def available(self) -> bool:
        if _mt5() is None:
            return False
        ok, _ = connect()
        return ok

    def fetch(self, symbol: str, timeframe: str, limit: int) -> pd.DataFrame:
        mt5 = _mt5()
        if mt5 is None:
            raise RuntimeError("MetaTrader5 package not installed")

        ok, msg = connect()
        if not ok:
            raise RuntimeError(msg)

        broker_symbol = resolve_symbol(symbol)
        if not broker_symbol:
            raise RuntimeError(
                f"{symbol} not offered by this broker. Check Market Watch "
                f"(right-click, Show All) for the exact name.")

        tf = _timeframes(mt5).get(timeframe)
        if tf is None:
            raise ValueError(f"MT5 cannot serve {timeframe}")

        rates = mt5.copy_rates_from_pos(broker_symbol, tf, 0, limit)
        if rates is None or len(rates) == 0:
            code, err = mt5.last_error()
            raise RuntimeError(f"no rates for {broker_symbol} ({code}: {err})")

        df = pd.DataFrame(rates)
        # Convert broker server time to real UTC before anything else touches it.
        df["time"] = pd.to_datetime(df["time"] - server_offset(), unit="s", utc=True)
        df = df.set_index("time")

        # tick_volume is the reliable one on FX; real_volume is often zero.
        df["volume"] = df.get("real_volume", 0)
        if df["volume"].sum() == 0:
            df["volume"] = df.get("tick_volume", 0)

        out = df[["open", "high", "low", "close", "volume"]].astype(float)
        out.attrs["ticker"] = broker_symbol
        return out


def live_tick(symbol: str) -> dict | None:
    """Current bid/ask. Free and instant - no candle fetch needed."""
    mt5 = _mt5()
    if mt5 is None or not connect()[0]:
        return None
    name = resolve_symbol(symbol)
    if not name:
        return None
    t = mt5.symbol_info_tick(name)
    if t is None:
        return None
    return {
        "symbol": name,
        "bid": t.bid, "ask": t.ask,
        "spread": round(t.ask - t.bid, 5),
        "at": dt.datetime.fromtimestamp(t.time - server_offset(),
                                        dt.timezone.utc).isoformat(),
    }


def status() -> dict:
    mt5 = _mt5()
    if mt5 is None:
        return {"installed": False,
                "hint": "pip install MetaTrader5  (Windows only)"}
    ok, msg = connect()
    out = {"installed": True, "connected": ok, "detail": msg}
    if ok:
        off = server_offset()
        out["server_offset_seconds"] = off
        out["server_timezone"] = f"UTC{off // 3600:+d}:{abs(off % 3600) // 60:02d}"
        acct = mt5.account_info()
        if acct:
            out["account"] = {"login": acct.login, "server": acct.server,
                              "company": acct.company, "currency": acct.currency}
        out["resolved_symbols"] = {k: v for k, v in _symbol_cache.items() if v}
    return out
