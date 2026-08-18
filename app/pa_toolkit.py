"""Price Action Toolkit — ported from Pine Script v6.

Adds what smc.py did not cover:

  Supply / Demand   base consolidation followed by an impulsive departure,
                    classified RBR / DBR / RBD / DBD
  Breaker Blocks    an order block that gets violated flips polarity
  Liquidity Grabs   wick through a swing that closes back inside
  CHoCH+            a change of character that also clears the prior swing
  Volume Imbalance  gap between one candle's open and the previous close
  Zone volume       volume traded inside a zone as a share of the lookback

Zones carry a `since` timestamp so the frontend can draw them from where they
formed rather than across the whole chart - which is what makes a chart look
like TradingView instead of a spiderweb.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .features import atr
from .smc import pivot_high, pivot_low


def _fmt_vol(v: float) -> str:
    for div, sfx in ((1e9, "B"), (1e6, "M"), (1e3, "K")):
        if v >= div:
            return f"{v / div:.3g}{sfx}"
    return f"{v:.0f}"


# ------------------------------------------------------- supply and demand

def supply_demand(df: pd.DataFrame, atr_len: int = 50, base_mult: float = 1.0,
                  depart_mult: float = 1.2, body_share: float = 0.5,
                  min_base: int = 1, max_base: int = 6,
                  keep: int = 5, vol_lookback: int = 100) -> dict:
    """Base then departure. The base range becomes the zone.

    proximal = the edge price returns to, distal = invalidation.
    """
    a = atr(df, atr_len)
    rng = df["high"] - df["low"]
    body = (df["close"] - df["open"]).abs()
    vol_sum = df["volume"].rolling(vol_lookback).sum()

    demand, supply = [], []
    count = 0
    hi = lo = vol_acc = 0.0
    start = 0
    pre = 0

    for i in range(1, len(df)):
        av = a.iloc[i]
        if av != av:
            continue
        is_base = rng.iloc[i] <= av * base_mult
        big_up = (df["close"].iloc[i] > df["open"].iloc[i]
                  and rng.iloc[i] >= av * depart_mult
                  and body.iloc[i] >= rng.iloc[i] * body_share)
        big_dn = (df["close"].iloc[i] < df["open"].iloc[i]
                  and rng.iloc[i] >= av * depart_mult
                  and body.iloc[i] >= rng.iloc[i] * body_share)

        if is_base:
            if count == 0:
                hi, lo = float(df["high"].iloc[i]), float(df["low"].iloc[i])
                vol_acc = float(df["volume"].iloc[i])
                start = i
                pre = 1 if df["close"].iloc[i-1] > df["open"].iloc[i-1] else \
                      -1 if df["close"].iloc[i-1] < df["open"].iloc[i-1] else 0
                count = 1
            else:
                hi = max(hi, float(df["high"].iloc[i]))
                lo = min(lo, float(df["low"].iloc[i]))
                vol_acc += float(df["volume"].iloc[i])
                count += 1
            continue

        ok = min_base <= count <= max_base
        if ok and big_up and df["close"].iloc[i] > hi:
            total = vol_acc + float(df["volume"].iloc[i])
            vs = vol_sum.iloc[i]
            demand.append(_zone(df, i, start, hi, lo, total, vs, 1,
                                "DBR" if pre == -1 else "RBR", "Demand"))
        elif ok and big_dn and df["close"].iloc[i] < lo:
            total = vol_acc + float(df["volume"].iloc[i])
            vs = vol_sum.iloc[i]
            supply.append(_zone(df, i, start, hi, lo, total, vs, -1,
                                "RBD" if pre == 1 else "DBD", "Supply"))
        count = 0

    price = float(df["close"].iloc[-1])
    for z in demand + supply:
        after = df.iloc[z["_i"] + 1:]
        z["tested"] = bool(((after["low"] <= z["top"]) & (after["high"] >= z["bottom"])).any())
        probe = after["close"]
        z["broken"] = bool((probe < z["bottom"]).any()) if z["dir"] == 1 \
            else bool((probe > z["top"]).any())
        z["distance_pct"] = round((z["mid"] - price) / price * 100, 3)

    live_d = [z for z in demand if not z["broken"]][-keep:]
    live_s = [z for z in supply if not z["broken"]][-keep:]
    for z in live_d + live_s:
        z.pop("_i", None)

    return {
        "demand": live_d, "supply": live_s,
        "nearest_demand": max((z for z in live_d if z["top"] < price),
                              key=lambda z: z["top"], default=None),
        "nearest_supply": min((z for z in live_s if z["bottom"] > price),
                              key=lambda z: z["bottom"], default=None),
        "counts": {"demand": len(live_d), "supply": len(live_s)},
    }


def _zone(df, i, start, hi, lo, vol, vol_sum, direction, pattern, kind) -> dict:
    pct = (vol / vol_sum * 100) if vol_sum and vol_sum == vol_sum else None
    return {
        "_i": i, "kind": kind, "pattern": pattern, "dir": direction,
        "top": round(hi, 5), "bottom": round(lo, 5), "mid": round((hi + lo) / 2, 5),
        "proximal": round(lo if direction == 1 else hi, 5),
        "distal": round(hi if direction == 1 else lo, 5),
        "since": df.index[start].isoformat(),
        "volume": round(vol, 0), "volume_label": _fmt_vol(vol),
        "volume_pct": round(min(100.0, pct), 1) if pct else None,
        "label": f"{kind} · {pattern}",
    }


# ------------------------------------------------------------ breaker blocks

def breaker_blocks(df: pd.DataFrame, swing: int = 10, atr_mult: float = 0.5,
                   keep: int = 5) -> dict:
    """An order block that price closes through flips polarity: bullish OB
    broken to the downside becomes bearish resistance."""
    a = atr(df, 200 if len(df) > 200 else 20)
    price = float(df["close"].iloc[-1])
    out = []

    for i in range(3, len(df)):
        k3o, k3c = df["open"].iloc[i-2], df["close"].iloc[i-2]
        k3h, k3l = df["high"].iloc[i-2], df["low"].iloc[i-2]
        k2rng = df["high"].iloc[i-1] - df["low"].iloc[i-1]
        av = a.iloc[i]
        if av != av or k2rng < av * atr_mult:
            continue

        bull = k3c < k3o and df["close"].iloc[i] > k3h
        bear = k3c > k3o and df["close"].iloc[i] < k3l
        if not (bull or bear):
            continue

        top, bot = float(k3h), float(k3l)
        after = df.iloc[i + 1:]
        if after.empty:
            continue
        violated = bool((after["close"] < bot).any()) if bull else bool((after["close"] > top).any())
        if not violated:
            continue

        out.append({
            "top": round(top, 5), "bottom": round(bot, 5),
            "mid": round((top + bot) / 2, 5),
            "dir": -1 if bull else 1,           # polarity flipped
            "was": "bullish OB" if bull else "bearish OB",
            "now": "bearish breaker (resistance)" if bull else "bullish breaker (support)",
            "since": df.index[i - 2].isoformat(),
            "distance_pct": round(((top + bot) / 2 - price) / price * 100, 3),
        })

    out = out[-keep:]
    return {"blocks": out, "count": len(out)}


# ----------------------------------------------------------- liquidity grabs

def liquidity_grabs(df: pd.DataFrame, swing: int = 10, wick_share: float = 0.5,
                    keep: int = 6) -> dict:
    """Wick through a swing that closes back inside. The wick must be at least
    `wick_share` of the candle's range - otherwise it is just a break."""
    ph = pivot_high(df["high"], swing, swing)
    pl = pivot_low(df["low"], swing, swing)
    grabs = []
    last_h = last_l = np.nan
    h_used = l_used = True

    for i in range(len(df)):
        if ph.iloc[i] == ph.iloc[i]:
            last_h, h_used = ph.iloc[i], False
        if pl.iloc[i] == pl.iloc[i]:
            last_l, l_used = pl.iloc[i], False

        o, c = df["open"].iloc[i], df["close"].iloc[i]
        h, l = df["high"].iloc[i], df["low"].iloc[i]
        rng = max(float(h - l), 1e-9)
        up_wick = float(h - max(o, c))
        dn_wick = float(min(o, c) - l)

        if last_h == last_h and not h_used and h > last_h and c < last_h \
                and up_wick / rng >= wick_share:
            grabs.append({"dir": "bearish", "level": round(float(last_h), 5),
                          "at": df.index[i].isoformat(),
                          "wick_share": round(up_wick / rng, 2)})
            h_used = True

        if last_l == last_l and not l_used and l < last_l and c > last_l \
                and dn_wick / rng >= wick_share:
            grabs.append({"dir": "bullish", "level": round(float(last_l), 5),
                          "at": df.index[i].isoformat(),
                          "wick_share": round(dn_wick / rng, 2)})
            l_used = True

    recent = grabs[-keep:]
    return {"grabs": recent, "count": len(recent),
            "latest": recent[-1] if recent else None,
            "on_last_bar": bool(recent and recent[-1]["at"] == df.index[-1].isoformat())}


# --------------------------------------------------------- volume imbalance

def volume_imbalance(df: pd.DataFrame, keep: int = 5) -> dict:
    """Gap between one candle's open and the previous close - a thin area
    price often revisits."""
    out = []
    price = float(df["close"].iloc[-1])
    for i in range(1, len(df)):
        o, prev_c = df["open"].iloc[i], df["close"].iloc[i-1]
        lo, hi = df["low"].iloc[i], df["high"].iloc[i]
        if o > prev_c and lo <= prev_c:
            out.append({"dir": "bullish", "top": round(float(o), 5),
                        "bottom": round(float(prev_c), 5),
                        "since": df.index[i].isoformat()})
        elif o < prev_c and hi >= prev_c:
            out.append({"dir": "bearish", "top": round(float(prev_c), 5),
                        "bottom": round(float(o), 5),
                        "since": df.index[i].isoformat()})
    out = out[-keep:]
    for z in out:
        z["distance_pct"] = round(((z["top"] + z["bottom"]) / 2 - price) / price * 100, 3)
    return {"zones": out, "count": len(out)}


# ------------------------------------------------- structure with CHoCH plus

def structure_plus(df: pd.DataFrame, swing: int = 10) -> dict:
    """BOS / CHoCH / CHoCH+ - the plus variant also clears the prior swing,
    which is a stronger reversal than a bare CHoCH."""
    ph = pivot_high(df["high"], swing, swing)
    pl = pivot_low(df["low"], swing, swing)
    close = df["close"].values

    last_h = prev_h = last_l = prev_l = np.nan
    h_taken = l_taken = True
    trend = 0
    events = []

    for i in range(len(df)):
        if ph.iloc[i] == ph.iloc[i]:
            prev_h, last_h, h_taken = last_h, ph.iloc[i], False
        if pl.iloc[i] == pl.iloc[i]:
            prev_l, last_l, l_taken = last_l, pl.iloc[i], False

        if last_h == last_h and not h_taken and close[i] > last_h:
            h_taken = True
            is_bos = trend >= 0
            plus = (not is_bos and prev_h == prev_h
                    and close[i] > max(float(prev_h), float(last_h)))
            events.append({"type": "BOS" if is_bos else "CHoCH+" if plus else "CHoCH",
                           "dir": "bullish", "level": round(float(last_h), 5),
                           "at": df.index[i].isoformat()})
            trend = 1

        if last_l == last_l and not l_taken and close[i] < last_l:
            l_taken = True
            is_bos = trend <= 0
            plus = (not is_bos and prev_l == prev_l
                    and close[i] < min(float(prev_l), float(last_l)))
            events.append({"type": "BOS" if is_bos else "CHoCH+" if plus else "CHoCH",
                           "dir": "bearish", "level": round(float(last_l), 5),
                           "at": df.index[i].isoformat()})
            trend = -1

    return {
        "trend": "bullish" if trend == 1 else "bearish" if trend == -1 else "neutral",
        "events": events[-8:],
        "last_event": events[-1] if events else None,
        "swing_high": round(float(last_h), 5) if last_h == last_h else None,
        "swing_low": round(float(last_l), 5) if last_l == last_l else None,
    }


# ------------------------------------------------------------------ assemble

def analyse(df: pd.DataFrame, swing: int = 10) -> dict:
    sd = supply_demand(df)
    bb = breaker_blocks(df, swing)
    lg = liquidity_grabs(df, swing)
    vi = volume_imbalance(df)
    st = structure_plus(df, swing)

    out = {"structure": st, "supply_demand": sd, "breaker_blocks": bb,
           "liquidity_grabs": lg, "volume_imbalance": vi}
    out["read"] = summarise(out)
    return out


def summarise(t: dict) -> str:
    bits = [f"Structure {t['structure']['trend']}"]
    ev = t["structure"]["last_event"]
    if ev:
        bits.append(f"last {ev['type']} {ev['dir']} at {ev['level']}")

    d, s = t["supply_demand"]["nearest_demand"], t["supply_demand"]["nearest_supply"]
    if d:
        bits.append(f"demand {d['bottom']}-{d['top']} ({d['pattern']}"
                    + (f", {d['volume_pct']}% of volume" if d.get("volume_pct") else "") + ")")
    if s:
        bits.append(f"supply {s['bottom']}-{s['top']} ({s['pattern']})")

    if t["breaker_blocks"]["count"]:
        b = t["breaker_blocks"]["blocks"][-1]
        bits.append(f"breaker {b['bottom']}-{b['top']} now {b['now']}")

    lg = t["liquidity_grabs"]["latest"]
    if lg:
        bits.append(f"last liquidity grab {lg['dir']} at {lg['level']}")
    return ". ".join(bits) + "."
