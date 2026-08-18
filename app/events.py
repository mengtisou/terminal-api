"""Event reaction history.

Answers the question that actually matters before a release: "how did this
instrument move the LAST eight times this event happened?"

Everything here is measured from real candles. The model is never asked to
recall what gold did after the June FOMC - it is handed the numbers.

Date sources, in order of reliability:
  1. Rule-derived  - NFP is the first Friday of the month, 08:30 ET. Exact.
  2. Seeded list   - FOMC dates are published years ahead.
  3. Auto-captured - anything the live calendar feed has already shown us.
"""
from __future__ import annotations

import datetime as dt
import logging
import re
import statistics
from zoneinfo import ZoneInfo

import pandas as pd

from .features import atr
from .market import get_candles

log = logging.getLogger(__name__)

ET = ZoneInfo("America/New_York")
UTC = dt.timezone.utc

# Reaction windows in minutes after the release.
WINDOWS = [15, 60, 240]

# FOMC decision days. The Fed publishes these years ahead, so a static list is
# accurate - but it goes stale. Anything the live calendar feed reports is
# merged in automatically via capture(), and _seen_dates() persists what we
# have learned so history survives a restart.
# Statement lands 14:00 ET, presser 14:30 ET.
FOMC_DATES = [
    "2025-01-29", "2025-03-19", "2025-05-07", "2025-06-18",
    "2025-07-30", "2025-09-17", "2025-10-29", "2025-12-10",
    "2026-01-28", "2026-03-18", "2026-04-29", "2026-06-17",
    "2026-07-29", "2026-09-16", "2026-11-04", "2026-12-16",
]

# Events we can generate dates for without any external data.
KNOWN = {
    "fomc": {"label": "FOMC rate decision", "hour_et": 14, "minute": 0},
    "nfp": {"label": "Non-farm payrolls", "hour_et": 8, "minute": 30},
    "cpi": {"label": "US CPI", "hour_et": 8, "minute": 30},
}

# Extra dates learned from the live calendar feed as events pass. Persisted so
# the list keeps growing past the hardcoded window without any code edit.
_CAPTURED: dict[str, set[str]] = {}
_STORE_FILE = __import__("pathlib").Path(
    __import__("os").getenv("STATE_DIR",
        __import__("pathlib").Path(__file__).resolve().parent.parent)) / "event_dates.json"


def _load_captured() -> None:
    global _CAPTURED
    try:
        import json
        raw = json.loads(_STORE_FILE.read_text())
        _CAPTURED = {k: set(v) for k, v in raw.items()}
        log.info("loaded %d captured event dates",
                 sum(len(v) for v in _CAPTURED.values()))
    except (OSError, ValueError):
        _CAPTURED = {}


def _save_captured() -> None:
    try:
        import json
        _STORE_FILE.write_text(
            json.dumps({k: sorted(v) for k, v in _CAPTURED.items()}, indent=1))
    except OSError as e:
        log.warning("could not persist event dates: %s", e)


_load_captured()


def classify(title: str) -> str | None:
    """Map a calendar title onto one of our known event families."""
    t = title.lower()
    if "fomc" in t or "federal funds" in t or "fed interest rate" in t:
        return "fomc"
    if "non-farm" in t or "nonfarm" in t or t.startswith("nfp"):
        return "nfp"
    if re.search(r"\bcpi\b", t) and "core" not in t:
        return "cpi"
    return None


def capture(events: list[dict]) -> None:
    """Learn real release times from the calendar feed, so the history gets
    more accurate over time rather than relying on rules forever."""
    added = 0
    for e in events:
        kind = classify(e.get("title", ""))
        if kind and e.get("currency") == "USD":
            bucket = _CAPTURED.setdefault(kind, set())
            if e["at"] not in bucket:
                bucket.add(e["at"])
                added += 1
    if added:
        _save_captured()
        log.info("captured %d new event dates", added)


# --- date generation ---------------------------------------------------------

def _first_friday(year: int, month: int) -> dt.date:
    d = dt.date(year, month, 1)
    return d + dt.timedelta(days=(4 - d.weekday()) % 7)


def event_dates(kind: str, lookback_days: int = 400) -> list[dt.datetime]:
    """Past occurrences, newest first, as UTC datetimes."""
    kind = kind.lower()
    spec = KNOWN.get(kind)
    if not spec:
        return []

    now = dt.datetime.now(UTC)
    start = now - dt.timedelta(days=lookback_days)
    out: list[dt.datetime] = []

    if kind == "fomc":
        for ds in FOMC_DATES:
            d = dt.date.fromisoformat(ds)
            at = dt.datetime(d.year, d.month, d.day, spec["hour_et"],
                             spec["minute"], tzinfo=ET).astimezone(UTC)
            if start <= at <= now:
                out.append(at)
    else:
        # NFP: first Friday. CPI: mid-month, roughly the 2nd Wednesday - less
        # exact, so captured dates take precedence when we have them.
        cur = start.date().replace(day=1)
        while cur <= now.date():
            if kind == "nfp":
                d = _first_friday(cur.year, cur.month)
            else:
                first = dt.date(cur.year, cur.month, 1)
                d = first + dt.timedelta(days=(2 - first.weekday()) % 7 + 7)
            at = dt.datetime(d.year, d.month, d.day, spec["hour_et"],
                             spec["minute"], tzinfo=ET).astimezone(UTC)
            if start <= at <= now:
                out.append(at)
            cur = (cur.replace(day=28) + dt.timedelta(days=4)).replace(day=1)

    for iso in _CAPTURED.get(kind, set()):
        try:
            at = dt.datetime.fromisoformat(iso)
            if start <= at <= now and not any(
                    abs((at - o).total_seconds()) < 7200 for o in out):
                out.append(at)
        except ValueError:
            continue

    return sorted(out, reverse=True)


# --- reaction measurement ----------------------------------------------------

def measure(df: pd.DataFrame, at: dt.datetime, atr_val: float) -> dict | None:
    """Price behaviour around one release, measured off real candles."""
    if df.empty:
        return None

    idx = df.index
    before = idx[idx <= at]
    if len(before) < 2:
        return None

    anchor = before[-1]
    base = float(df.loc[anchor, "close"])

    moves = {}
    for w in WINDOWS:
        end = at + dt.timedelta(minutes=w)
        window = df.loc[(idx > at) & (idx <= end)]
        if window.empty:
            continue
        close = float(window["close"].iloc[-1])
        hi, lo = float(window["high"].max()), float(window["low"].min())
        moves[f"{w}m"] = {
            "change": round(close - base, 3),
            "pct": round((close - base) / base * 100, 3),
            "atr": round((close - base) / atr_val, 2) if atr_val else None,
            "range": round(hi - lo, 3),
            "range_atr": round((hi - lo) / atr_val, 2) if atr_val else None,
            "direction": "up" if close > base else "down" if close < base else "flat",
        }

    if not moves:
        return None
    return {"at": at.isoformat(), "price_before": round(base, 3), "moves": moves}


def history(symbol: str, kind: str, count: int = 8,
            timeframe: str = "auto") -> dict:
    """Past reactions plus aggregate stats.

    Uses the finest data available per event. Providers cap intraday history by
    interval - 15m is typically 60 days, 1h about 2 years - so recent events are
    measured on 15m candles (giving a real 15-minute reaction number) and older
    ones fall back to 1h.
    """
    kind = kind.lower()
    if kind not in KNOWN:
        return {"error": f"unknown event '{kind}'. options: {list(KNOWN)}"}

    dates = event_dates(kind)[:count * 3]
    if not dates:
        return {"error": f"no past {kind} dates in range"}

    frames: dict[str, tuple] = {}

    def load(tf: str):
        if tf not in frames:
            try:
                frames[tf] = get_candles(symbol, tf, 5000)
            except Exception as e:
                log.warning("could not load %s %s: %s", symbol, tf, e)
                frames[tf] = (pd.DataFrame(), "unavailable")
        return frames[tf]

    tfs = ["15m", "1h"] if timeframe == "auto" else [timeframe]
    source, a = None, 0.0
    reactions = []

    for at in dates:
        for tf in tfs:
            df, src = load(tf)
            if df.empty or at < df.index[0]:
                continue          # this event predates the available history
            if not a:
                a = float(atr(df).iloc[-1]) if len(df) > 20 else 0.0
            r = measure(df, at, a)
            if r:
                r["measured_on"] = tf
                reactions.append(r)
                source = source or src
                break
        if len(reactions) >= count:
            break

    if not reactions:
        return {"error": f"no candle data covering past {kind} dates",
                "hint": "data history is shorter than the event lookback",
                "data_source": source}

    return {
        "symbol": symbol.upper(),
        "event": kind,
        "label": KNOWN[kind]["label"],
        "timeframe": "mixed 15m/1h" if timeframe == "auto" else timeframe,
        "oldest_event": reactions[-1]["at"][:10] if reactions else None,
        "data_source": source,
        "atr_reference": round(a, 3),
        "sample_size": len(reactions),
        "stats": _stats(reactions),
        "reactions": reactions,
    }


def _stats(reactions: list[dict]) -> dict:
    out = {}
    for w in WINDOWS:
        key = f"{w}m"
        rows = [r["moves"][key] for r in reactions if key in r["moves"]]
        if not rows:
            continue
        changes = [x["change"] for x in rows]
        ranges = [x["range"] for x in rows]
        ups = sum(1 for x in rows if x["direction"] == "up")
        out[key] = {
            "n": len(rows),
            "up": ups,
            "down": len(rows) - ups,
            "up_rate": round(ups / len(rows), 2),
            "avg_change": round(statistics.fmean(changes), 3),
            "median_abs_change": round(statistics.median(abs(c) for c in changes), 3),
            "avg_range": round(statistics.fmean(ranges), 3),
            "max_range": round(max(ranges), 3),
        }
    return out


def summarise(h: dict) -> str:
    """One-line plain-English read, for the UI and the model context."""
    if "error" in h:
        return h["error"]
    s = h["stats"].get("60m") or next(iter(h["stats"].values()), None)
    if not s:
        return "no measurable reaction data"
    bias = ("mostly up" if s["up_rate"] >= 0.65 else
            "mostly down" if s["up_rate"] <= 0.35 else "no directional edge")
    return (f"{h['label']}: last {s['n']} events moved {h['symbol']} a median "
            f"{s['median_abs_change']} in the first hour, average range "
            f"{s['avg_range']}. Direction {bias} ({s['up']} up / {s['down']} down).")


def coverage() -> dict:
    """What date data we hold, and how far the hardcoded list reaches.
    Surfaces staleness rather than letting it fail silently."""
    now = dt.datetime.now(UTC)
    last_seeded = dt.date.fromisoformat(FOMC_DATES[-1])
    return {
        "fomc_seeded_until": last_seeded.isoformat(),
        "seeded_list_stale": last_seeded < now.date(),
        "captured": {k: len(v) for k, v in _CAPTURED.items()},
        "note": ("FOMC dates past the seeded window are learned from the live "
                 "calendar feed as they appear."),
    }
