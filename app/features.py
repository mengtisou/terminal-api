"""Deterministic feature engine.

Everything numeric happens here, in code. The model never sees a raw candle
array — it sees the ~1k-token summary produced by build_snapshot().
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .market import get_candles, session_state


# --- Indicators --------------------------------------------------------------

def ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False).mean()


def rsi(s: pd.Series, n: int = 14) -> pd.Series:
    delta = s.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).fillna(50)


def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    prev_close = df["close"].shift()
    tr = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False).mean()


def macd(s: pd.Series, fast=12, slow=26, signal=9) -> pd.DataFrame:
    line = ema(s, fast) - ema(s, slow)
    sig = ema(line, signal)
    return pd.DataFrame({"macd": line, "signal": sig, "hist": line - sig})


def adx(df: pd.DataFrame, n: int = 14) -> pd.Series:
    up = df["high"].diff()
    down = -df["low"].diff()
    plus_dm = np.where((up > down) & (up > 0), up, 0.0)
    minus_dm = np.where((down > up) & (down > 0), down, 0.0)
    tr = atr(df, n)
    plus_di = 100 * pd.Series(plus_dm, index=df.index).ewm(alpha=1 / n, adjust=False).mean() / tr
    minus_di = 100 * pd.Series(minus_dm, index=df.index).ewm(alpha=1 / n, adjust=False).mean() / tr
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    return dx.ewm(alpha=1 / n, adjust=False).mean().fillna(0)


def bollinger_width(s: pd.Series, n: int = 20, k: float = 2.0) -> pd.Series:
    ma = s.rolling(n).mean()
    sd = s.rolling(n).std()
    return ((ma + k * sd) - (ma - k * sd)) / ma


# --- Market structure --------------------------------------------------------

def swing_points(df: pd.DataFrame, left: int = 3, right: int = 3) -> tuple[list, list]:
    """Fractal swing highs and lows. A pivot needs `left` lower bars before it
    and `right` lower bars after it."""
    highs, lows = [], []
    h, l = df["high"].values, df["low"].values
    for i in range(left, len(df) - right):
        window_h = h[i - left : i + right + 1]
        window_l = l[i - left : i + right + 1]
        if h[i] == window_h.max() and (window_h == h[i]).sum() == 1:
            highs.append((df.index[i], float(h[i])))
        if l[i] == window_l.min() and (window_l == l[i]).sum() == 1:
            lows.append((df.index[i], float(l[i])))
    return highs, lows


def cluster_levels(prices: list[float], tolerance: float) -> list[dict]:
    """Group nearby pivots into levels. More touches = stronger level."""
    if not prices:
        return []
    out: list[list[float]] = []
    for p in sorted(prices):
        if out and abs(p - np.mean(out[-1])) <= tolerance:
            out[-1].append(p)
        else:
            out.append([p])
    return [
        {"price": round(float(np.mean(g)), 5), "touches": len(g)}
        for g in out
    ]


def structure(df: pd.DataFrame, atr_val: float) -> dict:
    price = float(df["close"].iloc[-1])
    highs, lows = swing_points(df)
    recent_h = [p for _, p in highs[-25:]]
    recent_l = [p for _, p in lows[-25:]]
    tol = atr_val * 0.6

    resistance = [lv for lv in cluster_levels(recent_h, tol) if lv["price"] > price]
    support = [lv for lv in cluster_levels(recent_l, tol) if lv["price"] < price]

    # Higher highs / lower lows over the last few pivots.
    trend = "range"
    if len(recent_h) >= 2 and len(recent_l) >= 2:
        hh = recent_h[-1] > recent_h[-2]
        hl = recent_l[-1] > recent_l[-2]
        if hh and hl:
            trend = "uptrend"
        elif not hh and not hl:
            trend = "downtrend"

    return {
        "structure_trend": trend,
        "swing_high": round(max(recent_h), 5) if recent_h else None,
        "swing_low": round(min(recent_l), 5) if recent_l else None,
        "resistance": sorted(resistance, key=lambda x: x["price"])[:3],
        "support": sorted(support, key=lambda x: -x["price"])[:3],
    }


# --- Snapshot ----------------------------------------------------------------

def build_snapshot(symbol: str, timeframe: str, news: list[dict] | None = None,
                   smc_on: bool = True) -> dict:
    """The single object the model is allowed to reason over."""
    df, source = get_candles(symbol, timeframe)
    session = session_state(df, timeframe)

    close = df["close"]
    price = float(close.iloc[-1])
    a = float(atr(df).iloc[-1])
    m = macd(close).iloc[-1]

    ema20, ema50, ema200 = (float(ema(close, n).iloc[-1]) for n in (20, 50, 200))
    # Normalised slope: EMA change over 10 bars, expressed in ATRs.
    slope = float((ema(close, 20).iloc[-1] - ema(close, 20).iloc[-11]) / a) if a else 0.0

    vol = df["volume"]
    rel_vol = float(vol.iloc[-1] / vol.tail(50).mean()) if vol.tail(50).mean() else 0.0

    snap = {
        "symbol": symbol.upper(),
        "timeframe": timeframe,
        "data_source": source,
        "price": round(price, 5),
        "session": session,
        "trend": {
            "ema20": round(ema20, 5),
            "ema50": round(ema50, 5),
            "ema200": round(ema200, 5),
            "price_vs_ema200": "above" if price > ema200 else "below",
            "ema_stack": ("bullish" if ema20 > ema50 > ema200
                          else "bearish" if ema20 < ema50 < ema200 else "mixed"),
            "slope_20_in_atr": round(slope, 3),
        },
        "momentum": {
            "rsi14": round(float(rsi(close).iloc[-1]), 2),
            "macd_hist": round(float(m["hist"]), 5),
            "macd_cross": "bullish" if m["macd"] > m["signal"] else "bearish",
            "adx14": round(float(adx(df).iloc[-1]), 2),
        },
        "volatility": {
            "atr14": round(a, 5),
            "atr_pct": round(a / price * 100, 3),
            "bb_width": round(float(bollinger_width(close).iloc[-1]), 4),
            "regime": "low" if a / price * 100 < 0.15 else "normal" if a / price * 100 < 0.4 else "high",
        },
        "volume": {"rel_vol": round(rel_vol, 2)},
        "recent_bars": _recent_bars(df, 6),
        "news": news or [],
    }
    snap.update(structure(df, a))

    # SMC / ICT context. This is what turns a generic indicator read into an
    # actual methodology - the model gets order blocks, liquidity, PD array
    # and structure events rather than just RSI and a moving average.
    if smc_on:
        try:
            from .smc import analyse
            full = analyse(df, timeframe=timeframe)
            snap["smc"] = {
                "read": full["read"],
                "structure": full["structure"]["trend"],
                "last_structure_event": full["structure"]["last_event"],
                "pd_array": full["pd_array"],
                "ktr": {"position": full["ktr"]["position"],
                        "trend": full["ktr_trend"],
                        "open": full["ktr"]["levels"]["OP"],
                        "levels": full["ktr"]["levels"]},
                "nearest_demand_ob": full["order_blocks"]["nearest_demand"],
                "nearest_supply_ob": full["order_blocks"]["nearest_supply"],
                "open_fvg": {"bullish": full["fvg"]["bullish_open"][-2:],
                             "bearish": full["fvg"]["bearish_open"][-2:]},
                "liquidity": full["liquidity"],
                "cisd": full["cisd"],
                "crt": full["crt"],
                "order_flow": full["order_flow"],
            }
            from .pa_toolkit import analyse as pa
            t = pa(df)
            snap["smc"]["supply_demand"] = {
                "demand": t["supply_demand"]["nearest_demand"],
                "supply": t["supply_demand"]["nearest_supply"],
            }
            snap["smc"]["breaker_blocks"] = t["breaker_blocks"]["blocks"][-1:] or None
            snap["smc"]["liquidity_grab"] = t["liquidity_grabs"]["latest"]
            snap["smc"]["structure_events"] = t["structure"]["events"][-3:]

            from .ktr_signals import signals as ktr_sig
            k = ktr_sig(df)
            snap["smc"]["ktr_signals"] = {
                "latest_alert": k["latest_alert"],
                "latest_entry": k["latest_entry"],
                "alert_on_last_bar": k["alert_on_last_bar"],
                "entry_on_last_bar": k["entry_on_last_bar"],
                "pending": k["pending"],
                "fast_trend": k["trend"], "slow_trend": k["slow_trend"],
            }
        except Exception as e:  # never let SMC break the snapshot
            import logging
            logging.getLogger(__name__).warning("SMC analysis failed: %s", e)
            snap["smc"] = {"error": str(e)}

    return snap


def _recent_bars(df: pd.DataFrame, n: int) -> list[dict]:
    """A handful of raw bars for context. Six, not two hundred."""
    tail = df.tail(n)
    return [
        {
            "t": ts.strftime("%m-%d %H:%M"),
            "o": round(float(r.open), 5), "h": round(float(r.high), 5),
            "l": round(float(r.low), 5), "c": round(float(r.close), 5),
        }
        for ts, r in tail.iterrows()
    ]
