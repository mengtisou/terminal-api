"""Indicator library.

Pine Script cannot run outside TradingView - it is proprietary and there is no
export path. So indicators are re-implemented natively here. Each one declares
what it needs and what it draws, and the frontend renders whatever is enabled.

Adding your own: write a function returning {series_name: pd.Series}, then
register it in CATALOG. It appears in the UI automatically.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .features import atr, bollinger_width, ema, macd, rsi


# --- overlays (drawn on the price pane) --------------------------------------

def bollinger(df: pd.DataFrame, period: int = 20, mult: float = 2.0) -> dict:
    mid = df["close"].rolling(period).mean()
    sd = df["close"].rolling(period).std()
    return {"upper": mid + mult * sd, "middle": mid, "lower": mid - mult * sd}


def supertrend(df: pd.DataFrame, period: int = 10, mult: float = 3.0) -> dict:
    """Trend-following ATR band. Flips side when price closes through it."""
    a = atr(df, period)
    hl2 = (df["high"] + df["low"]) / 2
    upper, lower = hl2 + mult * a, hl2 - mult * a

    close = df["close"].values
    up, lo = upper.values.copy(), lower.values.copy()
    trend = np.ones(len(df))

    for i in range(1, len(df)):
        up[i] = min(up[i], up[i - 1]) if close[i - 1] <= up[i - 1] else up[i]
        lo[i] = max(lo[i], lo[i - 1]) if close[i - 1] >= lo[i - 1] else lo[i]
        if close[i] > up[i - 1]:
            trend[i] = 1
        elif close[i] < lo[i - 1]:
            trend[i] = -1
        else:
            trend[i] = trend[i - 1]

    line = pd.Series(np.where(trend == 1, lo, up), index=df.index)
    return {"supertrend": line, "direction": pd.Series(trend, index=df.index)}


def vwap(df: pd.DataFrame) -> dict:
    """Session VWAP, reset daily. Volume-weighted, so it needs real volume -
    MT5 tick volume works, most FX feeds do not."""
    tp = (df["high"] + df["low"] + df["close"]) / 3
    day = df.index.floor("D")
    pv = (tp * df["volume"]).groupby(day).cumsum()
    vol = df["volume"].groupby(day).cumsum().replace(0, np.nan)
    return {"vwap": pv / vol}


def ichimoku(df: pd.DataFrame, conv: int = 9, base: int = 26, span_b: int = 52) -> dict:
    def mid(n):
        return (df["high"].rolling(n).max() + df["low"].rolling(n).min()) / 2
    tenkan, kijun = mid(conv), mid(base)
    return {
        "tenkan": tenkan,
        "kijun": kijun,
        "senkou_a": ((tenkan + kijun) / 2).shift(base),
        "senkou_b": mid(span_b).shift(base),
        "chikou": df["close"].shift(-base),
    }


def emas(df: pd.DataFrame, periods=(20, 50, 200)) -> dict:
    return {f"ema{p}": ema(df["close"], p) for p in periods}


def sma(df: pd.DataFrame, periods=(50, 200)) -> dict:
    return {f"sma{p}": df["close"].rolling(p).mean() for p in periods}


def donchian(df: pd.DataFrame, period: int = 20) -> dict:
    hi = df["high"].rolling(period).max()
    lo = df["low"].rolling(period).min()
    return {"dc_upper": hi, "dc_lower": lo, "dc_mid": (hi + lo) / 2}


def psar(df: pd.DataFrame, step: float = 0.02, cap: float = 0.2) -> dict:
    """Parabolic SAR - dots that flip with trend."""
    high, low = df["high"].values, df["low"].values
    out = np.zeros(len(df))
    bull, af, ep, sar = True, step, high[0], low[0]

    for i in range(1, len(df)):
        sar = sar + af * (ep - sar)
        if bull:
            if low[i] < sar:
                bull, sar, ep, af = False, ep, low[i], step
            elif high[i] > ep:
                ep, af = high[i], min(af + step, cap)
        else:
            if high[i] > sar:
                bull, sar, ep, af = True, ep, high[i], step
            elif low[i] < ep:
                ep, af = low[i], min(af + step, cap)
        out[i] = sar
    out[0] = np.nan
    return {"psar": pd.Series(out, index=df.index)}


# --- SMC / ICT overlays ------------------------------------------------------
# These draw flat lines at computed levels. A horizontal level is just a series
# holding one value across every bar, so it plugs into the same renderer.

def _flat(df: pd.DataFrame, value: float | None, since: str | None = None) -> pd.Series:
    """A horizontal level.

    `since` matters more than it looks: drawing a level across the entire chart
    implies it existed before it formed, and turns the chart into a spiderweb.
    TradingView draws from the origin bar rightward, so we do the same.
    """
    if value is None or value != value:
        return pd.Series(np.nan, index=df.index)
    s = pd.Series(float(value), index=df.index)
    if since:
        try:
            start = pd.Timestamp(since)
            s[s.index < start] = np.nan
        except (ValueError, TypeError):
            pass
    return s


def ktr_levels(df: pd.DataFrame) -> dict:
    """Daily open anchor and the +-0.4% step ladder.

    Silently empty on daily and above, where a daily-open anchor is meaningless.
    """
    from .smc import ktr, ktr_days

    span = (df.index[-1] - df.index[-2]).total_seconds() / 60 if len(df) > 1 else 15
    if span >= 1440:
        return {"OP": _flat(df, None)}

    lv = ktr(df)["levels"]
    # KTR is anchored to the trading day, so it should not stretch back over
    # yesterday's candles. Must use the SAME boundary the levels were computed
    # on, or the line starts at a different bar than the maths assumed.
    days = ktr_days(df.index)
    day_start = df.index[days == days[-1]][0].isoformat()
    _f = lambda v: _flat(df, v, day_start)
    return {
        "OP": _f(lv["OP"]), "MLP": _f(lv["MLP"]),
        "KTR+1": _f(lv["KTR+1"]), "KTR+2": _f(lv["KTR+2"]), "KTR+3": _f(lv["KTR+3"]),
        "KTR-1": _f(lv["KTR-1"]), "KTR-2": _f(lv["KTR-2"]), "KTR-3": _f(lv["KTR-3"]),
    }


def liquidity_levels(df: pd.DataFrame) -> dict:
    """BSL / SSL, the pools price tends to reach for."""
    from .smc import liquidity
    lq = liquidity(df)
    out = {"BSL": _flat(df, lq["bsl"]), "SSL": _flat(df, lq["ssl"])}
    for i, v in enumerate(lq.get("eqh") or []):
        out[f"EQH{i+1}"] = _flat(df, v)
    for i, v in enumerate(lq.get("eql") or []):
        out[f"EQL{i+1}"] = _flat(df, v)
    return out


def order_block_zones(df: pd.DataFrame) -> dict:
    """Nearest unmitigated demand and supply, drawn from where they formed."""
    from .smc import order_blocks, structure
    obs = order_blocks(df, structure(df))
    out = {}
    d, sup = obs["nearest_demand"], obs["nearest_supply"]
    if d:
        out["demand_top"] = _flat(df, d["top"], d["at"])
        out["demand_bottom"] = _flat(df, d["bottom"], d["at"])
    if sup:
        out["supply_top"] = _flat(df, sup["top"], sup["at"])
        out["supply_bottom"] = _flat(df, sup["bottom"], sup["at"])
    return out or {"demand_top": _flat(df, None)}


def fvg_zones(df: pd.DataFrame) -> dict:
    """Most recent unfilled gaps, with their consequent encroachment midline."""
    from .smc import fvg
    g = fvg(df)
    out = {}
    for i, z in enumerate(g["bullish_open"][-2:]):
        out[f"bull_fvg{i+1}_top"] = _flat(df, z["top"], z["at"])
        out[f"bull_fvg{i+1}_ce"] = _flat(df, z["ce"], z["at"])
        out[f"bull_fvg{i+1}_bot"] = _flat(df, z["bottom"], z["at"])
    for i, z in enumerate(g["bearish_open"][-2:]):
        out[f"bear_fvg{i+1}_top"] = _flat(df, z["top"], z["at"])
        out[f"bear_fvg{i+1}_ce"] = _flat(df, z["ce"], z["at"])
        out[f"bear_fvg{i+1}_bot"] = _flat(df, z["bottom"], z["at"])
    return out or {"bull_fvg1_top": _flat(df, None)}


def structure_levels(df: pd.DataFrame) -> dict:
    """Last swing high / low and the current dealing range midpoint."""
    from .smc import structure
    st = structure(df)
    hi, lo = st.get("last_swing_high"), st.get("last_swing_low")
    out = {"swing_high": _flat(df, hi), "swing_low": _flat(df, lo)}
    if hi and lo:
        out["equilibrium"] = _flat(df, (hi + lo) / 2)
        out["premium_62"] = _flat(df, lo + (hi - lo) * 0.62)
        out["discount_38"] = _flat(df, lo + (hi - lo) * 0.38)
    return out


def sd_zones(df: pd.DataFrame) -> dict:
    """Supply and demand from the PA toolkit: proximal and distal edges."""
    from .pa_toolkit import supply_demand
    sd = supply_demand(df)
    out = {}
    d, s_ = sd["nearest_demand"], sd["nearest_supply"]
    if d:
        out["demand_prox"] = _flat(df, d["proximal"], d["since"])
        out["demand_distal"] = _flat(df, d["distal"], d["since"])
    if s_:
        out["supply_prox"] = _flat(df, s_["proximal"], s_["since"])
        out["supply_distal"] = _flat(df, s_["distal"], s_["since"])
    return out or {"demand_prox": _flat(df, None)}


def breaker_zones(df: pd.DataFrame) -> dict:
    from .pa_toolkit import breaker_blocks
    bb = breaker_blocks(df)["blocks"][-2:]
    out = {}
    for i, b in enumerate(bb):
        out[f"breaker{i+1}_top"] = _flat(df, b["top"], b["since"])
        out[f"breaker{i+1}_bottom"] = _flat(df, b["bottom"], b["since"])
    return out or {"breaker1_top": _flat(df, None)}


def vi_zones(df: pd.DataFrame) -> dict:
    from .pa_toolkit import volume_imbalance
    zs = volume_imbalance(df)["zones"][-2:]
    out = {}
    for i, z in enumerate(zs):
        out[f"vi{i+1}_top"] = _flat(df, z["top"], z["since"])
        out[f"vi{i+1}_bottom"] = _flat(df, z["bottom"], z["since"])
    return out or {"vi1_top": _flat(df, None)}


def ktr_ma(df: pd.DataFrame, length: int = 30) -> dict:
    """The 30MA. On the KTR chart this is the line price pulls back to."""
    return {"ma30": df["close"].rolling(length).mean()}


def pivot_01(df: pd.DataFrame, length: int = 10) -> dict:
    """Most recent confirmed swing high - the level the method watches."""
    from .smc import pivot_high
    ph = pivot_high(df["high"], length, length).dropna()
    if ph.empty:
        return {"pivot01": _flat(df, None)}
    return {"pivot01": _flat(df, float(ph.iloc[-1]), ph.index[-1].isoformat())}


def ktr_supertrend(df: pd.DataFrame) -> dict:
    """The Supertrend line that drives KTR candle colouring."""
    from .smc import supertrend_line
    return {"ktr_trend": supertrend_line(df, 2.0, 10)}


# --- oscillators (drawn in a separate pane) ----------------------------------

def rsi_pane(df: pd.DataFrame, period: int = 14) -> dict:
    return {"rsi": rsi(df["close"], period)}


def macd_pane(df: pd.DataFrame) -> dict:
    m = macd(df["close"])
    return {"macd": m["macd"], "signal": m["signal"], "histogram": m["hist"]}


def stochastic(df: pd.DataFrame, k: int = 14, d: int = 3, smooth: int = 3) -> dict:
    lo = df["low"].rolling(k).min()
    hi = df["high"].rolling(k).max()
    raw = 100 * (df["close"] - lo) / (hi - lo).replace(0, np.nan)
    k_line = raw.rolling(smooth).mean()
    return {"k": k_line, "d": k_line.rolling(d).mean()}


def atr_pane(df: pd.DataFrame, period: int = 14) -> dict:
    return {"atr": atr(df, period)}


def volume_pane(df: pd.DataFrame) -> dict:
    return {"volume": df["volume"],
            "vol_ma": df["volume"].rolling(20).mean()}


# --- catalog -----------------------------------------------------------------
# pane: "price" draws over candles, anything else gets its own panel.

CATALOG = {
    "ema":        {"fn": emas,        "pane": "price", "label": "EMA 20/50/200",
                   "colors": {"ema20": "#4a9eff", "ema50": "#e3a008", "ema200": "#a371f7"},
                   "default": True},
    "sma":        {"fn": sma,         "pane": "price", "label": "SMA 50/200",
                   "colors": {"sma50": "#f78166", "sma200": "#7ee787"}},
    "bollinger":  {"fn": bollinger,   "pane": "price", "label": "Bollinger Bands",
                   "colors": {"upper": "#8b98a8", "middle": "#4a9eff", "lower": "#8b98a8"}},
    "supertrend": {"fn": supertrend,  "pane": "price", "label": "Supertrend",
                   "colors": {"supertrend": "#26a69a"}, "skip": ["direction"]},
    "vwap":       {"fn": vwap,        "pane": "price", "label": "VWAP (session)",
                   "colors": {"vwap": "#f0b84a"}},
    "ichimoku":   {"fn": ichimoku,    "pane": "price", "label": "Ichimoku Cloud",
                   "colors": {"tenkan": "#4a9eff", "kijun": "#f85149",
                              "senkou_a": "#26a69a", "senkou_b": "#ef5350",
                              "chikou": "#8b98a8"}},
    "donchian":   {"fn": donchian,    "pane": "price", "label": "Donchian Channel",
                   "colors": {"dc_upper": "#ef5350", "dc_mid": "#8b98a8",
                              "dc_lower": "#26a69a"}},
    "psar":       {"fn": psar,        "pane": "price", "label": "Parabolic SAR",
                   "colors": {"psar": "#a371f7"}, "style": "dots"},

    # --- SMC / ICT ---
    "ktr_levels": {"fn": ktr_levels, "pane": "price", "label": "KTR levels (OP/MLP/±3)",
                   "colors": {"OP": "#ffffff", "MLP": "#d4af37",
                              "KTR+1": "#00e676", "KTR+2": "#ffeb3b", "KTR+3": "#ff9800",
                              "KTR-1": "#00e676", "KTR-2": "#ffeb3b", "KTR-3": "#ff9800"},
                   "style": "level"},
    "ktr_trend":  {"fn": ktr_supertrend, "pane": "price", "label": "KTR Supertrend",
                   "colors": {"ktr_trend": "#ffeb3b"}},
    "smc_ob":     {"fn": order_block_zones, "pane": "price", "label": "Order Blocks",
                   "colors": {"demand_top": "#26a69a", "demand_bottom": "#26a69a",
                              "supply_top": "#ef5350", "supply_bottom": "#ef5350"},
                   "style": "level"},
    "smc_fvg":    {"fn": fvg_zones, "pane": "price", "label": "Fair Value Gaps",
                   "colors": {"bull_fvg1_top": "#26a69a", "bull_fvg1_ce": "#26a69a",
                              "bull_fvg1_bot": "#26a69a", "bull_fvg2_top": "#26a69a",
                              "bull_fvg2_ce": "#26a69a", "bull_fvg2_bot": "#26a69a",
                              "bear_fvg1_top": "#ef5350", "bear_fvg1_ce": "#ef5350",
                              "bear_fvg1_bot": "#ef5350", "bear_fvg2_top": "#ef5350",
                              "bear_fvg2_ce": "#ef5350", "bear_fvg2_bot": "#ef5350"},
                   "style": "level"},
    "smc_liq":    {"fn": liquidity_levels, "pane": "price", "label": "Liquidity BSL/SSL/EQ",
                   "colors": {"BSL": "#ff9800", "SSL": "#00bcd4",
                              "EQH1": "#ff9800", "EQH2": "#ff9800",
                              "EQL1": "#00bcd4", "EQL2": "#00bcd4"},
                   "style": "level"},
    "smc_pd":     {"fn": structure_levels, "pane": "price", "label": "PD Array / Swings",
                   "colors": {"swing_high": "#ef5350", "swing_low": "#26a69a",
                              "equilibrium": "#8b98a8", "premium_62": "#ef5350",
                              "discount_38": "#26a69a"},
                   "style": "level"},

    "ktr_ma":     {"fn": ktr_ma, "pane": "price", "label": "30 MA",
                   "colors": {"ma30": "#00e676"}},
    "pivot01":    {"fn": pivot_01, "pane": "price", "label": "Pivot 01 (swing high)",
                   "colors": {"pivot01": "#00e5ff"}, "style": "level"},
    "pa_sd":      {"fn": sd_zones, "pane": "price", "label": "Supply / Demand zones",
                   "colors": {"demand_prox": "#2196f3", "demand_distal": "#2196f3",
                              "supply_prox": "#ff9800", "supply_distal": "#ff9800"},
                   "style": "level"},
    "pa_breaker": {"fn": breaker_zones, "pane": "price", "label": "Breaker Blocks",
                   "colors": {"breaker1_top": "#9c27b0", "breaker1_bottom": "#9c27b0",
                              "breaker2_top": "#9c27b0", "breaker2_bottom": "#9c27b0"},
                   "style": "level"},
    "pa_vi":      {"fn": vi_zones, "pane": "price", "label": "Volume Imbalances",
                   "colors": {"vi1_top": "#8b98a8", "vi1_bottom": "#8b98a8",
                              "vi2_top": "#8b98a8", "vi2_bottom": "#8b98a8"},
                   "style": "level"},

    "rsi":        {"fn": rsi_pane,    "pane": "rsi", "label": "RSI (14)",
                   "colors": {"rsi": "#4a9eff"}, "bands": [30, 70], "range": [0, 100]},
    "macd":       {"fn": macd_pane,   "pane": "macd", "label": "MACD",
                   "colors": {"macd": "#4a9eff", "signal": "#e3a008",
                              "histogram": "#8b98a8"}, "histogram": ["histogram"]},
    "stochastic": {"fn": stochastic,  "pane": "stoch", "label": "Stochastic",
                   "colors": {"k": "#4a9eff", "d": "#e3a008"},
                   "bands": [20, 80], "range": [0, 100]},
    "atr":        {"fn": atr_pane,    "pane": "atr", "label": "ATR (14)",
                   "colors": {"atr": "#f0b84a"}},
    "volume":     {"fn": volume_pane, "pane": "volume", "label": "Volume",
                   "colors": {"volume": "#3d4a5c", "vol_ma": "#e3a008"},
                   "histogram": ["volume"]},
}


def catalog() -> list[dict]:
    """What the UI lists, without the function objects."""
    return [
        {"id": k, "label": v["label"], "pane": v["pane"],
         "default": v.get("default", False),
         "group": ("smc" if k.startswith(("smc_", "ktr_", "pa_"))
                   else "price" if v["pane"] == "price" else "pane"),
         "series": list(v.get("colors", {}))}
        for k, v in CATALOG.items()
    ]


def compute(df: pd.DataFrame, wanted: list[str]) -> dict:
    """Run the requested indicators and shape them for Lightweight Charts."""
    out = {}
    for name in wanted:
        spec = CATALOG.get(name)
        if not spec:
            continue
        try:
            series = spec["fn"](df)
        except Exception as e:  # one bad indicator must not kill the chart
            out[name] = {"error": f"{type(e).__name__}: {e}"}
            continue

        skip = set(spec.get("skip", []))
        data = {}
        for key, s in series.items():
            if key in skip:
                continue
            points = [
                {"time": int(ts.timestamp()), "value": round(float(v), 5)}
                for ts, v in s.items() if pd.notna(v)
            ]
            data[key] = points

        out[name] = {
            "pane": spec["pane"],
            "label": spec["label"],
            "colors": spec.get("colors", {}),
            "series": data,
            "bands": spec.get("bands"),
            "range": spec.get("range"),
            "histogram": spec.get("histogram", []),
            "style": spec.get("style", "line"),
        }
    return out
