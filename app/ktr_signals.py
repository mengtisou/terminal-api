"""KTR entry engine — the diamonds and ❖ confirmations.

This is the part of the SMC 2026 script that actually tells you to act. The
levels are context; these are triggers.

Two stages, exactly as in Pine:

  Stage 1 - alert diamond.  Three ways to fire:
      reversal      counter-trend thrust with stretched RSI or a fresh extreme
      continuation  a pullback against the SLOW trend failing (softer RSI gate)
      micro         2+ counter-trend closes then a thrust back with the trend

  Stage 2 - entry ❖.  Within the confirm window, either the fast trend flips
      or price breaks the alert bar's extreme. Requires above-average volume
      ("follow the Big Boys"). Fires once per alert.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .features import atr, rsi
from .smc import supertrend


def signals(df: pd.DataFrame, *, st_factor: float = 2.0, st_len: int = 10,
            rsi_lo: float = 35, rsi_hi: float = 65, cooldown: int = 3,
            window: int = 12, continuation: bool = True,
            require_volume: bool = True, min_rel_vol: float = 1.0,
            keep: int = 12) -> dict:
    n = len(df)
    if n < 30:
        return {"alerts": [], "entries": [], "latest_alert": None, "latest_entry": None}

    close, open_, high, low = (df[c].values for c in ("close", "open", "high", "low"))
    fast = supertrend(df, st_factor, st_len).values
    slow = supertrend(df, st_factor * 2.0, st_len).values

    r = rsi(df["close"], 7)
    rsi_min5 = r.rolling(5).min().values
    rsi_max5 = r.rolling(5).max().values

    low12 = df["low"].rolling(12).min().values
    high12 = df["high"].rolling(12).max().values
    rel_vol = (df["volume"] / df["volume"].rolling(20).mean()).values
    a = atr(df).values

    alerts, entries = [], []
    last_sig = -10**9
    up_run = dn_run = 0

    # Pending alert state, mirroring the Pine vars.
    buy_bar = sell_bar = -10**9
    buy_high = sell_low = np.nan
    buy_done = sell_done = True

    for i in range(2, n):
        up = fast[i] > 0
        slow_up = slow[i] > 0

        bull_thrust = close[i] > open_[i] and close[i] > high[i-1]
        bear_thrust = close[i] < open_[i] and close[i] < low[i-1]

        os_ = rsi_min5[i] <= rsi_lo
        ob = rsi_max5[i] >= rsi_hi
        os_soft = rsi_min5[i] <= 45
        ob_soft = rsi_max5[i] >= 55

        # At a fresh extreme - lets diamonds fire on a slow grind that never
        # stretches RSI to the strict zone.
        at_low = bool(np.any(low[max(0, i-3):i+1] == low12[i]))
        at_high = bool(np.any(high[max(0, i-3):i+1] == high12[i]))

        rev_buy = (not up) and bull_thrust and (os_ or at_low)
        rev_sell = up and bear_thrust and (ob or at_high)

        con_buy = continuation and slow_up and (not up) and bull_thrust and os_soft
        con_sell = continuation and (not slow_up) and up and bear_thrust and ob_soft

        mic_sell = continuation and (not slow_up) and (not up) and up_run >= 2 and bear_thrust
        mic_buy = continuation and slow_up and up and dn_run >= 2 and bull_thrust

        up_run = up_run + 1 if close[i] > open_[i] else 0
        dn_run = dn_run + 1 if close[i] < open_[i] else 0

        cool_ok = i - last_sig >= cooldown
        a_buy = (rev_buy or con_buy or mic_buy) and cool_ok
        a_sell = (rev_sell or con_sell or mic_sell) and cool_ok

        if a_buy or a_sell:
            last_sig = i
            kind = ("reversal" if (rev_buy or rev_sell)
                    else "continuation" if (con_buy or con_sell) else "micro pullback")
            alerts.append({
                "dir": "buy" if a_buy else "sell", "kind": kind,
                "at": df.index[i].isoformat(),
                "price": round(float(close[i]), 5),
                "marker_price": round(float(low[i] - a[i] * 0.25 if a_buy
                                            else high[i] + a[i] * 0.25), 5),
            })
        if a_buy:
            buy_bar, buy_high, buy_done = i, high[i], False
        if a_sell:
            sell_bar, sell_low, sell_done = i, low[i], False

        flip_up = up and fast[i-1] <= 0
        flip_dn = (not up) and fast[i-1] > 0
        vol_ok = (not require_volume) or (rel_vol[i] >= min_rel_vol
                                          if rel_vol[i] == rel_vol[i] else False)

        e_buy = (not buy_done and i - buy_bar <= window and vol_ok
                 and (flip_up or (i > buy_bar and close[i] > buy_high)))
        e_sell = (not sell_done and i - sell_bar <= window and vol_ok
                  and (flip_dn or (i > sell_bar and close[i] < sell_low)))

        if e_buy:
            buy_done = True
            entries.append(_entry(df, i, "buy", close, low, a, rel_vol,
                                  "trend flip" if flip_up else "broke alert high"))
        if e_sell:
            sell_done = True
            entries.append(_entry(df, i, "sell", close, high, a, rel_vol,
                                  "trend flip" if flip_dn else "broke alert low"))

    last_t = df.index[-1].isoformat()
    return {
        "alerts": alerts[-keep:],
        "entries": entries[-keep:],
        "latest_alert": alerts[-1] if alerts else None,
        "latest_entry": entries[-1] if entries else None,
        "alert_on_last_bar": bool(alerts and alerts[-1]["at"] == last_t),
        "entry_on_last_bar": bool(entries and entries[-1]["at"] == last_t),
        "trend": "bullish" if fast[-1] > 0 else "bearish",
        "slow_trend": "bullish" if slow[-1] > 0 else "bearish",
        "pending": {
            "buy": not buy_done, "sell": not sell_done,
            "buy_trigger_above": round(float(buy_high), 5) if not buy_done else None,
            "sell_trigger_below": round(float(sell_low), 5) if not sell_done else None,
        },
    }


def _entry(df, i, direction, close, extreme, a, rel_vol, trigger) -> dict:
    return {
        "dir": direction, "trigger": trigger,
        "at": df.index[i].isoformat(),
        "price": round(float(close[i]), 5),
        "rel_volume": round(float(rel_vol[i]), 2) if rel_vol[i] == rel_vol[i] else None,
        "marker_price": round(float(extreme[i] - a[i] * 0.6 if direction == "buy"
                                    else extreme[i] + a[i] * 0.6), 5),
    }


def boys_bullishness(df: pd.DataFrame, vol_len: int = 20,
                     bull_len: int = 14) -> dict:
    """The two pane histograms.

    Boys        relative volume - are institutions actually participating
    Bullishness signed volume pressure; deep readings mark exhaustion
    """
    vol_ma = df["volume"].rolling(vol_len).mean()
    rel = (df["volume"] / vol_ma * 100).fillna(0)

    signed = np.where(df["close"] > df["open"], df["volume"],
                      np.where(df["close"] < df["open"], -df["volume"], 0.0))
    net = pd.Series(signed, index=df.index).rolling(bull_len).sum()
    press = (net / df["volume"].rolling(100).mean() * 50).fillna(0)
    extreme = press < press.rolling(200).min() * 0.75

    return {
        "relative_volume": rel,
        "pressure": press,
        "extreme": extreme,
        "current": {
            "relative_volume": round(float(rel.iloc[-1]), 1),
            "pressure": round(float(press.iloc[-1]), 1),
            "label": "boys buying" if press.iloc[-1] > 0 else "boys selling",
            "at_extreme": bool(extreme.iloc[-1]),
        },
    }
