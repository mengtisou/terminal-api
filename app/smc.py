"""SMC / ICT engine — ported from Pine Script.

Covers the analytical core of the "SMC 2026 ICT + KTR" indicator:

  KTR      daily OP/MLP anchors and KTR+-1/2/3 step levels
  Structure BOS / CHoCH from fractal pivots, market trend state
  Order Blocks  last opposing candle before a break, with mitigation
                tracking and relative volume
  FVG/IFVG  3-candle imbalance, fill tracking, inversion
  Liquidity BSL / SSL swing levels and sweep detection with BS%/SS%
  CISD      change in state of delivery
  CRT       3-candle range theory
  Order flow  volume-weighted delta
  PD array  premium / discount / equilibrium position
  EQH/EQL   equal highs and lows

Deliberately omitted: AMD session boxes, BTC panel and the dashboard, which
are presentation rather than analysis. SMT needs a second symbol and lives in
smt() below.

Everything returns plain dicts so it can go straight into the model snapshot -
that is the point. Drawing it on a chart is secondary.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .features import atr, ema


# ---------------------------------------------------------------- pivots

def pivot_high(h: pd.Series, left: int, right: int) -> pd.Series:
    """Pine's ta.pivothigh: value is placed at the pivot bar, NaN elsewhere."""
    out = pd.Series(np.nan, index=h.index)
    v = h.values
    for i in range(left, len(v) - right):
        w = v[i - left:i + right + 1]
        if v[i] == w.max() and (w == v[i]).sum() == 1:
            out.iloc[i] = v[i]
    return out


def pivot_low(l: pd.Series, left: int, right: int) -> pd.Series:
    out = pd.Series(np.nan, index=l.index)
    v = l.values
    for i in range(left, len(v) - right):
        w = v[i - left:i + right + 1]
        if v[i] == w.min() and (w == v[i]).sum() == 1:
            out.iloc[i] = v[i]
    return out


# ------------------------------------------------------------------- KTR

def ktr(df: pd.DataFrame, step_pct: float = 0.40,
        timeframe: str | None = None) -> dict:
    """Daily open anchor plus symmetric percentage steps.

    OP  = today's open, MLP = yesterday's close.
    KTR+n = OP * (1 + n*step%), KTR-n likewise below.

    The method is intraday by construction: it measures how far price has
    travelled from TODAY'S open. On a daily chart each candle is its own day,
    and on weekly or monthly charts the idea has no meaning at all - so we say
    so rather than emitting numbers that look authoritative and are not.
    """
    from .market import TF_MINUTES

    if timeframe and TF_MINUTES.get(timeframe, 0) >= 1440:
        price = float(df["close"].iloc[-1])
        return {
            "applicable": False,
            "reason": f"KTR anchors to the daily open, so it is not meaningful "
                      f"on a {timeframe} chart. Use 4h or lower.",
            "levels": {}, "step": None,
            "position": "n/a", "distance_from_open_pct": None,
            "side": "n/a",
        }

    days = df.index.normalize()
    op = float(df["open"][days == days[-1]].iloc[0])
    prev = df[days < days[-1]]
    mlp = float(prev["close"].iloc[-1]) if len(prev) else np.nan

    stp = op * step_pct / 100.0
    price = float(df["close"].iloc[-1])

    levels = {"OP": round(op, 5), "MLP": round(mlp, 5) if mlp == mlp else None}
    for n in (1, 2, 3):
        levels[f"KTR+{n}"] = round(op + n * stp, 5)
        levels[f"KTR-{n}"] = round(op - n * stp, 5)

    # Which band price currently sits in - the actionable read.
    band, above = "OP", price - op
    for n in (3, 2, 1):
        if price >= op + n * stp:
            band = f"above KTR+{n}"
            break
        if price <= op - n * stp:
            band = f"below KTR-{n}"
            break
    else:
        band = "between OP and KTR+1" if price > op else "between OP and KTR-1"

    return {
        "applicable": True,
        "levels": levels,
        "step": round(stp, 5),
        "position": band,
        "distance_from_open_pct": round((price - op) / op * 100, 3),
        "side": "above open" if above > 0 else "below open",
    }


def ktr_variants(df: pd.DataFrame, step_pct: float = 0.40) -> dict:
    """The same KTR ladder under every day-boundary and anchor convention.

    This port and the Pine original disagree on two independent choices, and
    from the numbers alone you cannot tell which one is off:

      1. the day boundary - 00:00 UTC here, but spot metals conventionally
         roll at 17:00 New York (21:00/22:00 UTC), which changes what
         "today's open" and "yesterday's close" resolve to;
      2. the anchor the +-n steps are measured from - OP here, MLP in Pine.

    Rather than guess and quietly emit wrong levels, compute all of them and
    let the chart decide: whichever row reproduces TradingView is the one the
    script uses.
    """
    if df.empty:
        return {"error": "no candles"}

    price = float(df["close"].iloc[-1])
    out = {"price": round(price, 5), "step_pct": step_pct, "variants": {}}

    for bname, hour in (("utc_midnight", 0), ("ny_1700_edt", 21), ("ny_1700_est", 22)):
        # Shift so the chosen boundary lands on midnight, group, then shift back.
        shifted = df.index + pd.Timedelta(hours=24 - hour if hour else 0)
        days = shifted.normalize()
        today = df[days == days[-1]]
        prev = df[days < days[-1]]
        if today.empty:
            continue

        op = float(today["open"].iloc[0])
        mlp = float(prev["close"].iloc[-1]) if len(prev) else None

        row = {"OP": round(op, 5), "MLP": round(mlp, 5) if mlp else None,
               "bars_today": int(len(today)), "anchored": {}}

        for aname, anchor in (("from_OP", op), ("from_MLP", mlp)):
            if anchor is None:
                continue
            stp = anchor * step_pct / 100.0
            row["anchored"][aname] = {
                "step": round(stp, 5),
                **{f"KTR+{n}": round(anchor + n * stp, 2) for n in (1, 2, 3)},
                **{f"KTR-{n}": round(anchor - n * stp, 2) for n in (1, 2, 3)},
            }
        out["variants"][bname] = row

    return out


def supertrend(df: pd.DataFrame, factor: float = 2.0, period: int = 10) -> pd.Series:
    """Faithful port of Pine's ta.supertrend.

    Returns +1 for uptrend, -1 for downtrend. Note Pine's own direction output
    is INVERTED (-1 means up), which is why the Pine script reads
    `ktrUp = ktrDir < 0`. We flip it here so +1 means bullish everywhere.

    Two details that must match or every flip lands a bar early:
      - Bands are clamped against the previous bar, then the close is compared
        against the CLAMPED band, not the previous one.
      - The band in force is chosen by which band the previous supertrend sat
        on, not by the previous direction value.
    """
    a = atr(df, period)
    hl2 = (df["high"] + df["low"]) / 2
    upper = (hl2 + factor * a).to_numpy(copy=True)
    lower = (hl2 - factor * a).to_numpy(copy=True)
    close = df["close"].to_numpy()
    n = len(df)

    direction = np.ones(n)          # Pine convention internally: 1 = down
    st = np.full(n, np.nan)

    for i in range(1, n):
        if np.isnan(a.iloc[i - 1]):
            direction[i] = 1
            st[i] = upper[i]
            continue

        prev_lower = lower[i - 1]
        prev_upper = upper[i - 1]

        # Clamp: a band only loosens once price has closed through it.
        if not (lower[i] > prev_lower or close[i - 1] < prev_lower):
            lower[i] = prev_lower
        if not (upper[i] < prev_upper or close[i - 1] > prev_upper):
            upper[i] = prev_upper

        prev_st = st[i - 1]
        if prev_st == prev_upper:                       # was in downtrend
            direction[i] = -1 if close[i] > upper[i] else 1
        else:                                           # was in uptrend
            direction[i] = 1 if close[i] < lower[i] else -1

        st[i] = lower[i] if direction[i] == -1 else upper[i]

    # Flip to the convention used across this codebase: +1 bullish.
    return pd.Series(-direction, index=df.index)


def supertrend_line(df: pd.DataFrame, factor: float = 2.0,
                    period: int = 10) -> pd.Series:
    """The Supertrend line itself, for plotting."""
    a = atr(df, period)
    hl2 = (df["high"] + df["low"]) / 2
    d = supertrend(df, factor, period)
    return pd.Series(np.where(d > 0, hl2 - factor * a, hl2 + factor * a),
                     index=df.index)


# ------------------------------------------------------- structure / BOS

def structure(df: pd.DataFrame, swing: int = 10) -> dict:
    """BOS, CHoCH and the resulting trend state."""
    ph = pivot_high(df["high"], swing, swing)
    pl = pivot_low(df["low"], swing, swing)
    close = df["close"].values

    last_h = last_l = np.nan
    trend = 0
    events: list[dict] = []

    ch_len = max(swing // 2, 2)
    ch_h = pivot_high(df["high"], ch_len, ch_len)
    ch_l = pivot_low(df["low"], ch_len, ch_len)
    last_ch_h = last_ch_l = np.nan

    for i in range(len(df)):
        if ph.iloc[i] == ph.iloc[i]:
            last_h = ph.iloc[i]
        if pl.iloc[i] == pl.iloc[i]:
            last_l = pl.iloc[i]
        if ch_h.iloc[i] == ch_h.iloc[i]:
            last_ch_h = ch_h.iloc[i]
        if ch_l.iloc[i] == ch_l.iloc[i]:
            last_ch_l = ch_l.iloc[i]
        if i == 0:
            continue

        bull_bos = last_h == last_h and close[i] > last_h and close[i - 1] <= last_h
        bear_bos = last_l == last_l and close[i] < last_l and close[i - 1] >= last_l
        bull_ch = (last_ch_h == last_ch_h and close[i] > last_ch_h
                   and close[i - 1] <= last_ch_h and trend == -1) or (bull_bos and trend == -1)
        bear_ch = (last_ch_l == last_ch_l and close[i] < last_ch_l
                   and close[i - 1] >= last_ch_l and trend == 1) or (bear_bos and trend == 1)

        if bull_ch:
            events.append({"i": i, "type": "CHoCH", "dir": "bullish",
                           "at": df.index[i].isoformat(), "level": float(last_ch_h)})
            trend = 1
        elif bull_bos:
            events.append({"i": i, "type": "BOS", "dir": "bullish",
                           "at": df.index[i].isoformat(), "level": float(last_h)})
            trend = 1
        if bear_ch:
            events.append({"i": i, "type": "CHoCH", "dir": "bearish",
                           "at": df.index[i].isoformat(), "level": float(last_ch_l)})
            trend = -1
        elif bear_bos:
            events.append({"i": i, "type": "BOS", "dir": "bearish",
                           "at": df.index[i].isoformat(), "level": float(last_l)})
            trend = -1

    recent = events[-6:]
    return {
        "trend": "bullish" if trend == 1 else "bearish" if trend == -1 else "ranging",
        "last_event": recent[-1] if recent else None,
        "recent_events": [{k: v for k, v in e.items() if k != "i"} for e in recent],
        "last_swing_high": round(float(last_h), 5) if last_h == last_h else None,
        "last_swing_low": round(float(last_l), 5) if last_l == last_l else None,
        "_events": events,
        "_ph": ph, "_pl": pl,
    }


# ---------------------------------------------------------- order blocks

def order_blocks(df: pd.DataFrame, struct: dict, swing: int = 10,
                 max_blocks: int = 3) -> dict:
    """Last opposing candle before a break. Tracked until mitigated."""
    vol_avg = df["volume"].rolling(20).mean()
    price = float(df["close"].iloc[-1])
    bull, bear = [], []

    for ev in struct["_events"]:
        if ev["type"] not in ("BOS", "CHoCH"):
            continue
        i = ev["i"]
        want_down = ev["dir"] == "bullish"   # bullish break -> last DOWN candle
        for j in range(1, swing + 1):
            k = i - j
            if k < 1:
                break
            o, c = df["open"].iloc[k], df["close"].iloc[k]
            if (c < o) if want_down else (c > o):
                top, bot = max(o, c), min(o, c)
                if top <= bot:
                    break
                va = vol_avg.iloc[k]
                vp = float(df["volume"].iloc[k] / va * 100) if va and va == va else 100.0
                # Mitigated once price has traded back inside the block.
                after = df.iloc[k + 1:]
                mit = bool(((after["low"] < top) & (after["high"] > bot)).any())
                block = {
                    "top": round(float(top), 5), "bottom": round(float(bot), 5),
                    "mid": round(float((top + bot) / 2), 5),
                    "at": df.index[k].isoformat(),
                    "volume_pct": round(vp, 0),
                    "strength": "high" if vp >= 150 else "strong" if vp >= 120 else "normal",
                    "mitigated": mit,
                    "distance_pct": round((float((top + bot) / 2) - price) / price * 100, 3),
                }
                (bull if want_down else bear).append(block)
                break

    bull = [b for b in bull if not b["mitigated"]][-max_blocks:]
    bear = [b for b in bear if not b["mitigated"]][-max_blocks:]
    return {
        "bullish": sorted(bull, key=lambda b: -b["top"]),
        "bearish": sorted(bear, key=lambda b: b["bottom"]),
        "nearest_demand": max((b for b in bull if b["top"] < price),
                              key=lambda b: b["top"], default=None),
        "nearest_supply": min((b for b in bear if b["bottom"] > price),
                              key=lambda b: b["bottom"], default=None),
    }


# ------------------------------------------------------------ FVG / IFVG

def fvg(df: pd.DataFrame, atr_mult: float = 0.1, keep: int = 6) -> dict:
    """3-candle imbalance. Inverted once filled and closed through."""
    a = atr(df)
    price = float(df["close"].iloc[-1])
    bull, bear, inverted = [], [], []

    for i in range(2, len(df)):
        lo_i, hi_2 = df["low"].iloc[i], df["high"].iloc[i - 2]
        hi_i, lo_2 = df["high"].iloc[i], df["low"].iloc[i - 2]
        thr = a.iloc[i] * atr_mult

        if lo_i > hi_2 and (lo_i - hi_2) > thr:
            top, bot = float(lo_i), float(hi_2)
            after = df.iloc[i + 1:]
            filled = bool((after["low"] < top).any())
            flipped = filled and bool((after["close"] < bot).any())
            entry = {"top": round(top, 5), "bottom": round(bot, 5),
                     "ce": round((top + bot) / 2, 5), "at": df.index[i].isoformat(),
                     "filled": filled,
                     "distance_pct": round(((top + bot) / 2 - price) / price * 100, 3)}
            (inverted if flipped else bull).append({**entry, "type":
                "IFVG bearish (was bullish)" if flipped else "bullish"})

        if hi_i < lo_2 and (lo_2 - hi_i) > thr:
            top, bot = float(lo_2), float(hi_i)
            after = df.iloc[i + 1:]
            filled = bool((after["high"] > bot).any())
            flipped = filled and bool((after["close"] > top).any())
            entry = {"top": round(top, 5), "bottom": round(bot, 5),
                     "ce": round((top + bot) / 2, 5), "at": df.index[i].isoformat(),
                     "filled": filled,
                     "distance_pct": round(((top + bot) / 2 - price) / price * 100, 3)}
            (inverted if flipped else bear).append({**entry, "type":
                "IFVG bullish (was bearish)" if flipped else "bearish"})

    open_bull = [g for g in bull if not g["filled"]][-keep:]
    open_bear = [g for g in bear if not g["filled"]][-keep:]
    return {
        "bullish_open": open_bull,
        "bearish_open": open_bear,
        "inverted": inverted[-3:],
        "counts": {"bullish_open": len(open_bull), "bearish_open": len(open_bear),
                   "inverted": len(inverted)},
    }


# ------------------------------------------------------------- liquidity

def liquidity(df: pd.DataFrame, length: int = 10, tol_pct: float = 0.05) -> dict:
    """BSL / SSL, sweep detection, and equal highs / lows."""
    ph = pivot_high(df["high"], length, length).dropna()
    pl = pivot_low(df["low"], length, length).dropna()
    price = float(df["close"].iloc[-1])

    bsl = float(ph.iloc[-1]) if len(ph) else None
    ssl = float(pl.iloc[-1]) if len(pl) else None

    last = df.iloc[-1]
    swept_bsl = bool(bsl and last["high"] > bsl and last["close"] < bsl)
    swept_ssl = bool(ssl and last["low"] < ssl and last["close"] > ssl)

    rng = max(float(last["high"] - last["low"]), 1e-9)
    bv = float(last["volume"]) if last["close"] >= last["open"] else \
        float(last["volume"]) * float(last["close"] - last["low"]) / rng
    sv = float(last["volume"]) if last["close"] < last["open"] else \
        float(last["volume"]) * float(last["high"] - last["close"]) / rng
    tot = max(bv + sv, 1e-9)

    def equals(series, tol):
        out = []
        vals = series.values
        for i in range(1, len(vals)):
            if abs(vals[i] - vals[i - 1]) <= abs(vals[i]) * tol / 100:
                out.append(round(float((vals[i] + vals[i - 1]) / 2), 5))
        return out[-2:]

    return {
        "bsl": round(bsl, 5) if bsl else None,
        "ssl": round(ssl, 5) if ssl else None,
        "bsl_distance_pct": round((bsl - price) / price * 100, 3) if bsl else None,
        "ssl_distance_pct": round((ssl - price) / price * 100, 3) if ssl else None,
        "bsl_swept": swept_bsl,
        "ssl_swept": swept_ssl,
        "sweep_buy_pct": round(bv / tot * 100, 1) if (swept_bsl or swept_ssl) else None,
        "sweep_sell_pct": round(sv / tot * 100, 1) if (swept_bsl or swept_ssl) else None,
        "eqh": equals(ph, tol_pct),
        "eql": equals(pl, tol_pct),
    }


# ------------------------------------------------------- CISD / CRT / PD

def cisd(df: pd.DataFrame, run: int = 2) -> dict:
    """Change in state of delivery: a run of one-sided candles, then a close
    back through the origin candle's close."""
    c, o = df["close"], df["open"]
    origin = 3 if run >= 2 else 2
    if len(df) < origin + 2:
        return {"signal": None}

    bear_run = all(c.iloc[-k] < o.iloc[-k] for k in range(2, 2 + run))
    bull_run = all(c.iloc[-k] > o.iloc[-k] for k in range(2, 2 + run))
    oc = float(c.iloc[-(origin + 1)])
    oo = float(o.iloc[-(origin + 1)])

    if bear_run and oc < oo and float(c.iloc[-1]) > oc:
        return {"signal": "bullish", "level": round(oc, 5), "role": "support"}
    if bull_run and oc > oo and float(c.iloc[-1]) < oc:
        return {"signal": "bearish", "level": round(oc, 5), "role": "resistance"}
    return {"signal": None}


def crt(df: pd.DataFrame) -> dict:
    """3-candle range theory: sweep of the range candle, then close back inside."""
    if len(df) < 3:
        return {"signal": None}
    h2, l2 = float(df["high"].iloc[-3]), float(df["low"].iloc[-3])
    h1, l1 = float(df["high"].iloc[-2]), float(df["low"].iloc[-2])
    o, c = float(df["open"].iloc[-1]), float(df["close"].iloc[-1])

    if l1 < l2 and c > o and l2 < c < h2 and h1 < h2:
        return {"signal": "bullish", "range_high": round(h2, 5), "range_low": round(l2, 5)}
    if h1 > h2 and c < o and l2 < c < h2 and l1 > l2:
        return {"signal": "bearish", "range_high": round(h2, 5), "range_low": round(l2, 5)}
    return {"signal": None, "range_high": round(h2, 5), "range_low": round(l2, 5)}


def order_flow(df: pd.DataFrame, length: int = 10) -> dict:
    """Volume-weighted delta over the lookback."""
    rng = (df["high"] - df["low"]).clip(lower=1e-9)
    bv = np.where(df["close"] >= df["open"], df["volume"],
                  df["volume"] * (df["close"] - df["low"]) / rng)
    sv = np.where(df["close"] < df["open"], df["volume"],
                  df["volume"] * (df["high"] - df["close"]) / rng)
    b, s = float(pd.Series(bv).tail(length).sum()), float(pd.Series(sv).tail(length).sum())
    delta = (b - s) / max(b + s, 1e-9) * 100
    return {"delta_pct": round(delta, 1),
            "bias": "buying" if delta > 0 else "selling",
            "lookback": length}


def pd_array(df: pd.DataFrame, struct: dict) -> dict:
    """Premium / discount position within the last dealing range."""
    hi, lo = struct.get("last_swing_high"), struct.get("last_swing_low")
    price = float(df["close"].iloc[-1])
    if not hi or not lo or hi <= lo:
        return {"position_pct": 50.0, "zone": "unknown"}
    pos = (price - lo) / (hi - lo) * 100

    # Price outside the last dealing range means the range has been broken and
    # is no longer a valid premium/discount reference. Saying "205% premium" is
    # meaningless - say the range is stale instead.
    if pos > 100 or pos < 0:
        return {
            "position_pct": round(pos, 1),
            "zone": "range broken - no valid PD reference",
            "range_high": hi, "range_low": lo,
            "broken": True,
            "direction": "above range" if pos > 100 else "below range",
            "ok_for_long": pos < 0, "ok_for_short": pos > 100,
            "note": "price has expanded beyond the last swing range; wait for a "
                    "new range to form before using premium/discount",
        }

    zone = ("premium - sell zone" if pos >= 62 else
            "discount - buy zone" if pos <= 38 else "equilibrium - wait")
    return {"position_pct": round(pos, 1), "zone": zone,
            "range_high": hi, "range_low": lo, "broken": False,
            "ok_for_long": pos <= 62, "ok_for_short": pos >= 38}


# ----------------------------------------------------------------- SMT

def smt(df: pd.DataFrame, other: pd.DataFrame, length: int = 10) -> dict:
    """Divergence against a correlated instrument (gold vs silver)."""
    try:
        gh, gl = pivot_high(df["high"], length, length).dropna(), pivot_low(df["low"], length, length).dropna()
        oh, ol = pivot_high(other["high"], length, length).dropna(), pivot_low(other["low"], length, length).dropna()
        if min(len(gh), len(gl), len(oh), len(ol)) < 2:
            return {"signal": None}
        if gh.iloc[-1] > gh.iloc[-2] and oh.iloc[-1] < oh.iloc[-2]:
            return {"signal": "bearish", "note": "this made a higher high, the correlated symbol did not"}
        if gl.iloc[-1] < gl.iloc[-2] and ol.iloc[-1] > ol.iloc[-2]:
            return {"signal": "bullish", "note": "this made a lower low, the correlated symbol did not"}
        return {"signal": None}
    except Exception:
        return {"signal": None}


# ------------------------------------------------------------- assemble

def analyse(df: pd.DataFrame, swing: int = 10, ktr_step: float = 0.40,
            cisd_run: int = 2, timeframe: str | None = None) -> dict:
    """The full SMC read, shaped for the model snapshot."""
    st = structure(df, swing)
    obs = order_blocks(df, st, swing)
    gaps = fvg(df)
    liq = liquidity(df, swing)
    pda = pd_array(df, st)
    trend = supertrend(df)

    out = {
        "ktr": ktr(df, ktr_step, timeframe),
        "ktr_trend": "bullish" if trend.iloc[-1] > 0 else "bearish",
        "structure": {k: v for k, v in st.items() if not k.startswith("_")},
        "order_blocks": obs,
        "fvg": gaps,
        "liquidity": liq,
        "cisd": cisd(df, cisd_run),
        "crt": crt(df),
        "order_flow": order_flow(df),
        "pd_array": pda,
    }
    out["read"] = summarise(out)
    return out


def summarise(s: dict) -> str:
    """One paragraph a trader would actually say out loud."""
    bits = [f"Structure {s['structure']['trend']}"]
    ev = s["structure"].get("last_event")
    if ev:
        bits.append(f"last {ev['type']} {ev['dir']} at {ev['level']}")
    bits.append(f"price {s['pd_array']['zone']} ({s['pd_array']['position_pct']}%)")
    if s["ktr"].get("applicable", True):
        bits.append(f"KTR {s['ktr']['position']}, trend {s['ktr_trend']}")
    else:
        bits.append(f"trend {s['ktr_trend']} (KTR levels n/a on this timeframe)")
    bits.append(f"order flow {s['order_flow']['bias']} {abs(s['order_flow']['delta_pct'])}%")

    d, sup = s["order_blocks"]["nearest_demand"], s["order_blocks"]["nearest_supply"]
    if d:
        bits.append(f"demand OB {d['bottom']}-{d['top']} ({d['strength']} vol)")
    if sup:
        bits.append(f"supply OB {sup['bottom']}-{sup['top']} ({sup['strength']} vol)")

    liq = s["liquidity"]
    if liq["bsl_swept"]:
        bits.append(f"BSL swept at {liq['bsl']}")
    if liq["ssl_swept"]:
        bits.append(f"SSL swept at {liq['ssl']}")
    if s["cisd"]["signal"]:
        bits.append(f"CISD {s['cisd']['signal']} from {s['cisd']['level']}")
    if s["crt"]["signal"]:
        bits.append(f"CRT {s['crt']['signal']}")
    return ". ".join(bits) + "."
