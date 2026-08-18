"""Top-down cascade: 4H bias → 1H confirm → entry timeframe trigger.

The single most useful thing in the SMC script, because it stops the engine
taking a 15m long while the 4H is bearish. Three gates, all of which must pass:

  1 BIAS     higher timeframe CISD or CHoCH sets direction, and records the
             level that cancels it
  2 CONFIRM  middle timeframe agrees with the same direction
  3 ENTRY    chart timeframe fires CRT, CISD or CHoCH - and the PD array must
             not be hostile (no longs in premium, no shorts in discount)

Bias is cancelled the moment price closes back through the origin level.
"""
from __future__ import annotations

import logging

from .market import get_candles
from .smc import cisd, crt, pd_array, structure

log = logging.getLogger(__name__)

# The ladder used to pick bias and confirm relative to the entry timeframe.
# Roughly 4x steps, which is the conventional top-down spacing.
LADDER = ["1m", "5m", "15m", "1h", "4h", "1d", "1w", "1M"]


def auto_timeframes(entry_tf: str) -> tuple[str | None, str | None]:
    """Bias is two rungs above entry, confirm is one.

    A fixed 4h/1h pair is wrong the moment the entry timeframe is 1d or higher -
    the "bias" would sit BELOW the chart you are trading, which inverts the
    whole point of a top-down check.
    """
    from .market import TF_MINUTES

    mins = TF_MINUTES.get(entry_tf)
    if mins is None:
        return "4h", "1h"

    above = [tf for tf in LADDER if TF_MINUTES[tf] > mins]
    if not above:
        return None, None            # already at the top - nothing above it
    confirm = above[0]
    bias = above[1] if len(above) > 1 else above[0]
    return bias, confirm


def _read(symbol: str, timeframe: str, cisd_run: int) -> dict | None:
    try:
        df, _ = get_candles(symbol, timeframe)
    except Exception as e:
        log.warning("cascade could not load %s %s: %s", symbol, timeframe, e)
        return None
    if len(df) < 30:
        return None

    st = structure(df, 10)
    c, r = cisd(df, cisd_run), crt(df)
    ev = st.get("last_event")

    bull = c["signal"] == "bullish" or (ev and ev["type"] == "CHoCH" and ev["dir"] == "bullish")
    bear = c["signal"] == "bearish" or (ev and ev["type"] == "CHoCH" and ev["dir"] == "bearish")

    return {
        "timeframe": timeframe,
        "price": round(float(df["close"].iloc[-1]), 5),
        "structure": st["trend"],
        "cisd": c["signal"], "crt": r["signal"],
        "last_event": ev,
        "bull_signal": bool(bull), "bear_signal": bool(bear),
        "source": ("CISD" if c["signal"] else "CHoCH" if ev and ev["type"].startswith("CHoCH")
                   else "structure"),
        "invalidation": c.get("level") or (ev["level"] if ev else None),
        "pd": pd_array(df, st),
    }


def evaluate(symbol: str, entry_tf: str = "15m", bias_tf: str | None = None,
             confirm_tf: str | None = None, cisd_run: int = 2,
             sl_points: float | None = None,
             tp_points: float | None = None) -> dict:
    from .market import TF_MINUTES

    auto_bias, auto_confirm = auto_timeframes(entry_tf)
    bias_tf = bias_tf or auto_bias
    confirm_tf = confirm_tf or auto_confirm

    if bias_tf is None or confirm_tf is None:
        return {"stage": "not_applicable",
                "status": "top of the ladder",
                "message": f"{entry_tf} is the highest timeframe available - "
                           f"there is nothing above it to take a bias from.",
                "ready": False, "direction": None}

    # Guard against a caller passing a bias timeframe at or below the entry.
    e = TF_MINUTES.get(entry_tf, 15)
    if TF_MINUTES.get(bias_tf, 240) <= e or TF_MINUTES.get(confirm_tf, 60) <= e:
        bias_tf, confirm_tf = auto_bias, auto_confirm
        if bias_tf is None:
            return {"stage": "not_applicable", "status": "top of the ladder",
                    "message": f"No timeframe sits above {entry_tf}.",
                    "ready": False, "direction": None}

    bias = _read(symbol, bias_tf, cisd_run)
    conf = _read(symbol, confirm_tf, cisd_run)
    entry = _read(symbol, entry_tf, cisd_run)

    if not all((bias, conf, entry)):
        return {"stage": "no_data",
                "message": "one or more timeframes unavailable"}

    # --- 1 bias -----------------------------------------------------------
    direction = None
    if bias["bull_signal"] or bias["structure"] == "bullish":
        direction = "long"
    if bias["bear_signal"] or bias["structure"] == "bearish":
        direction = "short" if not bias["bull_signal"] else direction

    if direction is None:
        return _stage(1, "no bias", bias, conf, entry,
                      f"{bias_tf} has no directional signal. Nothing to confirm.")

    inval = bias["invalidation"]
    if inval:
        broken = (entry["price"] < inval) if direction == "long" else (entry["price"] > inval)
        if broken:
            return _stage(1, "bias cancelled", bias, conf, entry,
                          f"{bias_tf} {direction} bias cancelled - price closed "
                          f"through {inval}.", direction=direction)

    # --- 2 confirm --------------------------------------------------------
    agrees = ((direction == "long" and (conf["bull_signal"] or conf["structure"] == "bullish"))
              or (direction == "short" and (conf["bear_signal"] or conf["structure"] == "bearish")))
    if not agrees:
        return _stage(2, "waiting confirm", bias, conf, entry,
                      f"{bias_tf} says {direction}, but {confirm_tf} has not agreed. "
                      f"{confirm_tf} structure is {conf['structure']}.", direction=direction)

    # --- 3 entry trigger --------------------------------------------------
    trig = None
    if direction == "long":
        if entry["crt"] == "bullish":
            trig = "CRT"
        elif entry["cisd"] == "bullish":
            trig = "CISD"
        elif entry["last_event"] and entry["last_event"]["type"].startswith("CHoCH") \
                and entry["last_event"]["dir"] == "bullish":
            trig = entry["last_event"]["type"]
    else:
        if entry["crt"] == "bearish":
            trig = "CRT"
        elif entry["cisd"] == "bearish":
            trig = "CISD"
        elif entry["last_event"] and entry["last_event"]["type"].startswith("CHoCH") \
                and entry["last_event"]["dir"] == "bearish":
            trig = entry["last_event"]["type"]

    if not trig:
        return _stage(3, "watching entry", bias, conf, entry,
                      f"{bias_tf} and {confirm_tf} both {direction}. Waiting for a "
                      f"CRT, CISD or CHoCH on {entry_tf}.", direction=direction)

    # PD gate - do not buy in premium or sell in discount.
    pd_e = entry["pd"]
    pos = pd_e.get("position_pct", 50)
    if not pd_e.get("broken"):
        if direction == "long" and pos > 62:
            return _stage(3, "blocked by PD", bias, conf, entry,
                          f"Trigger fired but price is in premium ({pos}%). "
                          f"Wait for a pullback.", direction=direction, trigger=trig)
        if direction == "short" and pos < 38:
            return _stage(3, "blocked by PD", bias, conf, entry,
                          f"Trigger fired but price is in discount ({pos}%). "
                          f"Wait for a bounce.", direction=direction, trigger=trig)

    out = _stage(3, "entry", bias, conf, entry,
                 f"{bias_tf} {direction} + {confirm_tf} confirm + {entry_tf} {trig}. "
                 f"All three align.", direction=direction, trigger=trig)

    px = entry["price"]
    if sl_points and tp_points:
        out["levels"] = {
            "entry": px,
            "stop_loss": round(px - sl_points if direction == "long" else px + sl_points, 5),
            "take_profit": round(px + tp_points if direction == "long" else px - tp_points, 5),
            "risk_reward": round(tp_points / sl_points, 2),
        }
    else:
        out["levels"] = {"entry": px,
                         "note": "size the stop off structure, not a fixed distance"}
    return out


def _stage(n, status, bias, conf, entry, message, direction=None, trigger=None) -> dict:
    return {
        "stage": n, "status": status, "direction": direction,
        "trigger": trigger, "message": message,
        "bias": {"timeframe": bias["timeframe"], "structure": bias["structure"],
                 "signal": bias["source"], "invalidation": bias["invalidation"]},
        "confirm": {"timeframe": conf["timeframe"], "structure": conf["structure"],
                    "signal": conf["source"]},
        "entry": {"timeframe": entry["timeframe"], "price": entry["price"],
                  "cisd": entry["cisd"], "crt": entry["crt"],
                  "pd_zone": entry["pd"].get("zone"),
                  "pd_pct": entry["pd"].get("position_pct")},
        "ready": status == "entry",
    }
