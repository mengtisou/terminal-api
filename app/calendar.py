"""Economic calendar.

The highest-value "news" for gold and forex is scheduled, not breaking. FOMC,
CPI and NFP dates are known weeks ahead and move price far more than any
headline. This module keeps them loaded and lets the risk layer refuse to
trade into them.

Primary source is ForexFactory's weekly JSON, which is public and needs no
key. Finnhub's economic calendar is used as a fallback when a key is present.
"""
from __future__ import annotations

import datetime as dt
import logging
import threading

import httpx

from .config import FINNHUB_KEY

log = logging.getLogger(__name__)

FF_THIS_WEEK = "https://nfs.faireconomy.media/ff_calendar_thisweek.json"
FF_NEXT_WEEK = "https://nfs.faireconomy.media/ff_calendar_nextweek.json"

_EVENTS: list[dict] = []
_lock = threading.Lock()
_last_load: dt.datetime | None = None

IMPACT = {"High": 3, "Medium": 2, "Low": 1, "Holiday": 1}

# Releases that reliably move gold and the dollar. Used to flag the ones worth
# blacking out for, rather than treating every "High" tag as equal.
MAJOR = (
    "fomc", "federal funds", "interest rate", "cpi", "core cpi", "ppi",
    "non-farm", "nonfarm", "nfp", "unemployment rate", "gdp",
    "powell", "fed chair", "pce", "retail sales", "ism",
)


def _parse_ff(rows: list[dict]) -> list[dict]:
    out = []
    for r in rows:
        raw = r.get("date")
        if not raw:
            continue
        try:
            at = dt.datetime.fromisoformat(raw.replace("Z", "+00:00"))
            if at.tzinfo is None:
                at = at.replace(tzinfo=dt.timezone.utc)
        except ValueError:
            continue

        title = (r.get("title") or "").strip()
        impact = IMPACT.get(r.get("impact", "Low"), 1)
        # Promote the releases that actually matter, demote the rest.
        if impact == 3 and not any(m in title.lower() for m in MAJOR):
            impact = 2

        out.append({
            "at": at.astimezone(dt.timezone.utc).isoformat(),
            "title": title,
            "currency": (r.get("country") or "").upper(),
            "impact": impact,
            "forecast": r.get("forecast") or None,
            "previous": r.get("previous") or None,
            "actual": r.get("actual") or None,
        })
    return out


def refresh(force: bool = False) -> dict:
    """Pull this week and next. Cheap - no model call, no API key."""
    global _last_load
    with _lock:
        if not force and _last_load:
            age = (dt.datetime.now(dt.timezone.utc) - _last_load).total_seconds()
            if age < 3600:
                return {"cached": True, "events": len(_EVENTS),
                        "age_seconds": round(age)}

        events, errors = [], []
        for url in (FF_THIS_WEEK, FF_NEXT_WEEK):
            try:
                r = httpx.get(url, timeout=20, follow_redirects=True,
                              headers={"User-Agent": "Mozilla/5.0 (terminal-ai)"})
                r.raise_for_status()
                events.extend(_parse_ff(r.json()))
            except Exception as e:
                errors.append(f"{url.rsplit('/', 1)[-1]}: {type(e).__name__}")
                log.warning("calendar fetch failed %s: %s", url, e)

        if not events and FINNHUB_KEY:
            events = _finnhub()
            if events:
                errors.append("used finnhub fallback")

        if events:
            # Dedupe on (time, title, currency) - the two weekly files overlap.
            seen, unique = set(), []
            for e in sorted(events, key=lambda x: x["at"]):
                k = (e["at"], e["title"], e["currency"])
                if k not in seen:
                    seen.add(k)
                    unique.append(e)
            _EVENTS[:] = unique
            _last_load = dt.datetime.now(dt.timezone.utc)

        return {"events": len(_EVENTS), "errors": errors or None,
                "loaded_at": _last_load.isoformat() if _last_load else None}


def _finnhub() -> list[dict]:
    today = dt.date.today()
    try:
        r = httpx.get(
            "https://finnhub.io/api/v1/calendar/economic",
            params={"from": today.isoformat(),
                    "to": (today + dt.timedelta(days=14)).isoformat(),
                    "token": FINNHUB_KEY},
            timeout=20)
        r.raise_for_status()
        out = []
        for e in r.json().get("economicCalendar", []):
            out.append({
                "at": f"{e['time'].replace(' ', 'T')}+00:00",
                "title": e.get("event", ""),
                "currency": (e.get("country") or "").upper(),
                "impact": min(3, max(1, int(e.get("impact") or 1))),
                "forecast": e.get("estimate"), "previous": e.get("prev"),
                "actual": e.get("actual"),
            })
        return out
    except Exception as e:
        log.warning("finnhub calendar failed: %s", e)
        return []


def all_events() -> list[dict]:
    return list(_EVENTS)


def next_major(currencies: set[str], hours: int = 336) -> dict | None:
    """The next impact-3 release. Drives the chart badge."""
    now = dt.datetime.now(dt.timezone.utc)
    horizon = now + dt.timedelta(hours=hours)
    for e in _EVENTS:
        at = dt.datetime.fromisoformat(e["at"])
        if e["impact"] >= 3 and now <= at <= horizon and e["currency"] in currencies:
            mins = (at - now).total_seconds() / 60
            return {**e, "minutes_away": round(mins),
                    "human": _human(mins)}
    return None


def _human(mins: float) -> str:
    if mins < 60:
        return f"{round(mins)}m"
    if mins < 1440:
        h, m = divmod(round(mins), 60)
        return f"{h}h {m}m" if m else f"{h}h"
    d, rem = divmod(round(mins), 1440)
    return f"{d}d {rem // 60}h"
