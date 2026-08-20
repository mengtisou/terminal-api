"""Automatic entry alerts, pushed to Telegram and recorded for scoring.

Two things are deliberately fused here, because they are the same work:

  push     - tell me when an entry appears, while I am not watching the screen
  record   - write down what was claimed, so it can be scored later

You cannot push without recording. To avoid alerting twice on the same setup
you need a durable fingerprint of what you already sent, and the moment you
store that you have also stored the trade idea. Recording entry/SL/TP at
signal time is exactly the outcome-tracking table - so the alert loop pays for
itself twice, and the terminal stops being an opinion generator with no
evidence attached.

Cost control matters. generate_signal() runs a reasoning model, a council and
a reviewer - far too expensive to poll on a timer. So the loop gates on the
deterministic KTR entry from ktr_signals.signals(), which is pure pandas and
free. The model only runs when that cheap trigger actually fires.
"""

from __future__ import annotations

import datetime as dt
import json
import logging
import os
import sqlite3
import threading
import time
from pathlib import Path

import httpx

log = logging.getLogger(__name__)

DB_PATH = Path(os.getenv("ALERTS_DB", "/root/terminal-api/data/signals.db"))
TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
TELEGRAM_CHAT = os.getenv("TELEGRAM_CHAT_ID", "").strip()

ALERTS_ON = os.getenv("ALERTS_ENABLED", "0") == "1"
ALERT_SYMBOLS = [s.strip().upper() for s in
                 os.getenv("ALERT_SYMBOLS", "XAUUSD").split(",") if s.strip()]
ALERT_TIMEFRAME = os.getenv("ALERT_TIMEFRAME", "15m")
SCAN_SECONDS = int(os.getenv("ALERT_SCAN_SECONDS", "60"))
MIN_CONFIDENCE = float(os.getenv("ALERT_MIN_CONFIDENCE", "0.55"))

_lock = threading.Lock()
_state = {"running": False, "last_scan": None, "last_error": None,
          "scans": 0, "triggers": 0, "pushed": 0}


# ---------------------------------------------------------------- storage
def _connect() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    c = sqlite3.connect(DB_PATH, timeout=10)
    c.row_factory = sqlite3.Row
    return c


def init_db() -> None:
    """Create the table if it is missing. Safe to call on every boot."""
    with _connect() as c:
        c.execute("""
            CREATE TABLE IF NOT EXISTS signals (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                fingerprint   TEXT UNIQUE,      -- symbol|tf|bar time|direction
                symbol        TEXT NOT NULL,
                timeframe     TEXT NOT NULL,
                direction     TEXT NOT NULL,    -- long / short
                bar_time      TEXT NOT NULL,    -- the bar that triggered it
                created_at    TEXT NOT NULL,
                entry         REAL,
                stop_loss     REAL,
                take_profit   TEXT,             -- JSON list
                confidence    REAL,
                trigger       TEXT,             -- KTR trigger name
                reasoning     TEXT,
                -- Resolution, filled in later. Never overwrite the columns
                -- above: an idea edited after the fact proves nothing.
                status        TEXT DEFAULT 'open',
                resolved_at   TEXT,
                exit_price    REAL,
                result_r      REAL,
                pushed        INTEGER DEFAULT 0
            )""")
        c.execute("CREATE INDEX IF NOT EXISTS ix_status ON signals(status)")


def already_seen(fingerprint: str) -> bool:
    with _connect() as c:
        return c.execute("SELECT 1 FROM signals WHERE fingerprint=?",
                         (fingerprint,)).fetchone() is not None


def record(sig: dict) -> int | None:
    """Insert one signal. Returns the row id, or None if it was a duplicate."""
    with _connect() as c:
        try:
            cur = c.execute("""
                INSERT INTO signals (fingerprint, symbol, timeframe, direction,
                    bar_time, created_at, entry, stop_loss, take_profit,
                    confidence, trigger, reasoning, pushed)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,0)""", (
                sig["fingerprint"], sig["symbol"], sig["timeframe"],
                sig["direction"], sig["bar_time"],
                dt.datetime.now(dt.timezone.utc).isoformat(),
                sig.get("entry"), sig.get("stop_loss"),
                json.dumps(sig.get("take_profit") or []),
                sig.get("confidence"), sig.get("trigger"),
                (sig.get("reasoning") or "")[:1200],
            ))
            return cur.lastrowid
        except sqlite3.IntegrityError:
            return None            # raced with another scan; not an error


def mark_pushed(row_id: int) -> None:
    with _connect() as c:
        c.execute("UPDATE signals SET pushed=1 WHERE id=?", (row_id,))


def stats() -> dict:
    """Counts for the status endpoint. Cheap enough to call from a browser."""
    with _connect() as c:
        row = c.execute("""
            SELECT COUNT(*) total,
                   SUM(status='open')   open,
                   SUM(status='tp')     wins,
                   SUM(status='sl')     losses,
                   SUM(pushed)          pushed
            FROM signals""").fetchone()
        recent = c.execute("""
            SELECT symbol, direction, entry, stop_loss, confidence,
                   created_at, status
            FROM signals ORDER BY id DESC LIMIT 5""").fetchall()
    out = {k: (row[k] or 0) for k in row.keys()}
    decided = out["wins"] + out["losses"]
    out["win_rate"] = round(out["wins"] / decided * 100, 1) if decided else None
    out["recent"] = [dict(r) for r in recent]
    return out


# ---------------------------------------------------------------- push channels
# Three ways to reach a phone, all plain HTTP. Whatever is configured gets used;
# configure more than one and the alert goes to all of them.
#
#   ntfy      free, no signup, real Android/iOS app. Closest thing to MT5's
#             push. Topic names on the public server are effectively passwords -
#             anyone who guesses yours can read your alerts - so use a long
#             random one.
#   pushover  $5 once, more polished app, per-device targeting.
#   telegram  no extra app if you already live in Telegram, but it is a chat
#             message, not a real notification channel.
NTFY_URL = os.getenv("NTFY_URL", "https://ntfy.sh").rstrip("/")
NTFY_TOPIC = os.getenv("NTFY_TOPIC", "").strip()
NTFY_TOKEN = os.getenv("NTFY_TOKEN", "").strip()      # only for private servers

PUSHOVER_TOKEN = os.getenv("PUSHOVER_TOKEN", "").strip()
PUSHOVER_USER = os.getenv("PUSHOVER_USER", "").strip()


def channels() -> list[str]:
    out = []
    try:
        from .webpush import count as _push_count
        if _push_count():
            out.append("browser")
    except Exception:
        pass
    if NTFY_TOPIC:
        out.append("ntfy")
    if PUSHOVER_TOKEN and PUSHOVER_USER:
        out.append("pushover")
    if TELEGRAM_TOKEN and TELEGRAM_CHAT:
        out.append("telegram")
    return out


def telegram_configured() -> bool:
    return bool(TELEGRAM_TOKEN and TELEGRAM_CHAT)


def _send_ntfy(title: str, body: str, high: bool) -> tuple[bool, str]:
    headers = {
        "Title": title.encode("utf-8"),
        "Priority": "high" if high else "default",
        "Tags": "chart_with_upwards_trend",
    }
    if NTFY_TOKEN:
        headers["Authorization"] = f"Bearer {NTFY_TOKEN}"
    try:
        r = httpx.post(f"{NTFY_URL}/{NTFY_TOPIC}",
                       content=body.encode("utf-8"), headers=headers, timeout=15)
        return (r.status_code < 300,
                "sent" if r.status_code < 300 else f"HTTP {r.status_code}: {r.text[:160]}")
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


def _send_pushover(title: str, body: str, high: bool) -> tuple[bool, str]:
    try:
        r = httpx.post("https://api.pushover.net/1/messages.json",
                       data={"token": PUSHOVER_TOKEN, "user": PUSHOVER_USER,
                             "title": title, "message": body,
                             "priority": 1 if high else 0},
                       timeout=15)
        return (r.status_code == 200,
                "sent" if r.status_code == 200 else f"HTTP {r.status_code}: {r.text[:160]}")
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


def _send_browser(title: str, body: str, high: bool) -> tuple[bool, str]:
    from .webpush import send
    ok, detail = send(title, body, url="/")
    return ok, ("sent" if ok else str(detail)[:200])


def _send_telegram(title: str, body: str, high: bool) -> tuple[bool, str]:
    try:
        r = httpx.post(
            f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage",
            json={"chat_id": TELEGRAM_CHAT, "text": f"<b>{title}</b>\n{body}",
                  "parse_mode": "HTML", "disable_web_page_preview": True},
            timeout=15,
        )
        return (r.status_code == 200,
                "sent" if r.status_code == 200 else f"HTTP {r.status_code}: {r.text[:200]}")
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


_SENDERS = {"browser": _send_browser, "ntfy": _send_ntfy,
            "pushover": _send_pushover, "telegram": _send_telegram}


def notify(title: str, body: str, high: bool = True) -> tuple[bool, dict]:
    """Push to every configured channel. True if at least one landed."""
    active = channels()
    if not active:
        return False, {"error": "no push channel configured - subscribe a "
                                "browser with the bell button, or set "
                                "NTFY_TOPIC / PUSHOVER_* / TELEGRAM_*"}
    results, any_ok = {}, False
    for name in active:
        ok, detail = _SENDERS[name](title, body, high)
        results[name] = detail
        any_ok = any_ok or ok
    return any_ok, results


def send_telegram(text: str) -> tuple[bool, str]:
    """Kept so older callers keep working."""
    return _send_telegram("Terminal", text, False)


def format_alert(sig: dict) -> tuple[str, str]:
    """(title, body). Plain text - a push notification is not a chat message,
    so no HTML, and the title has to carry the trade on its own for the case
    where that is all you see on the lock screen."""
    side = "BUY" if sig["direction"] == "long" else "SELL"
    title = f"{side} {sig['symbol']} @ {sig.get('entry')}"
    lines = [f"SL {sig.get('stop_loss')}"]
    tps = sig.get("take_profit") or []
    if tps:
        lines.append("TP " + " / ".join(str(t) for t in tps))
    rr = _risk_reward(sig)
    if rr:
        lines.append(f"R:R ~{rr}")
    if sig.get("confidence") is not None:
        lines.append(f"Confidence {int(sig['confidence'] * 100)}%  ({sig['timeframe']})")
    if sig.get("trigger"):
        lines.append(f"Trigger {sig['trigger']}")
    if sig.get("reasoning"):
        lines.append("")
        lines.append(sig["reasoning"][:300])
    return title, "\n".join(lines)


def _risk_reward(sig: dict) -> float | None:
    e, sl = sig.get("entry"), sig.get("stop_loss")
    tps = sig.get("take_profit") or []
    if not e or not sl or not tps:
        return None
    risk = abs(e - sl)
    if risk <= 0:
        return None
    return round(abs(tps[0] - e) / risk, 2)


# ---------------------------------------------------------------- scanning
def scan_once(symbol: str, timeframe: str) -> dict:
    """One cheap pass. Runs the model only when the KTR trigger has fired."""
    from .ktr_signals import signals as ktr_signals
    from .market import get_candles

    df, _src = get_candles(symbol, timeframe, 300)
    entry = ktr_signals(df).get("latest_entry")
    if not entry:
        return {"triggered": False, "reason": "no KTR entry on the latest bars"}

    # Only the most recent closed bar counts. An entry from twenty bars ago is
    # history, not an alert - and would fire once per scan forever.
    last_bar = df.index[-1]
    entry_bar = dt.datetime.fromisoformat(entry["at"])
    bars_ago = len(df.index[df.index > entry_bar])
    if bars_ago > 1:
        return {"triggered": False, "reason": f"KTR entry is {bars_ago} bars old"}

    direction = "long" if entry["dir"] == "buy" else "short"
    fingerprint = f"{symbol}|{timeframe}|{entry['at']}|{direction}"
    if already_seen(fingerprint):
        return {"triggered": True, "sent": False, "reason": "already alerted"}

    _state["triggers"] += 1

    # Cheap gate passed - now it is worth paying for the model.
    from .signals import generate_signal
    full = generate_signal(symbol, timeframe)

    if full.get("bias") in (None, "no_trade"):
        # Record the rejection too, so the KTR trigger's own hit rate can be
        # measured separately from the model's.
        record({**_base(symbol, timeframe, direction, entry, fingerprint),
                "reasoning": f"model declined: {full.get('reason') or full.get('bias')}"})
        return {"triggered": True, "sent": False, "reason": "model returned no_trade"}

    if full.get("bias") != direction:
        record({**_base(symbol, timeframe, direction, entry, fingerprint),
                "reasoning": f"model disagreed with KTR ({full.get('bias')})"})
        return {"triggered": True, "sent": False, "reason": "model disagreed with KTR"}

    conf = full.get("confidence") or 0
    sig = {**_base(symbol, timeframe, direction, entry, fingerprint),
           "entry": full.get("entry"), "stop_loss": full.get("stop_loss"),
           "take_profit": full.get("take_profit"), "confidence": conf,
           "reasoning": full.get("reasoning")}

    row_id = record(sig)
    if row_id is None:
        return {"triggered": True, "sent": False, "reason": "duplicate"}

    if conf < MIN_CONFIDENCE:
        return {"triggered": True, "sent": False,
                "reason": f"confidence {conf} below {MIN_CONFIDENCE}", "recorded": row_id}

    title, body = format_alert(sig)
    ok, detail = notify(title, body, high=True)
    if ok:
        mark_pushed(row_id)
        _state["pushed"] += 1
    return {"triggered": True, "sent": ok, "detail": detail, "recorded": row_id}


def _base(symbol, timeframe, direction, entry, fingerprint) -> dict:
    return {"fingerprint": fingerprint, "symbol": symbol, "timeframe": timeframe,
            "direction": direction, "bar_time": entry["at"],
            "trigger": entry.get("trigger"), "entry": entry.get("price"),
            "stop_loss": None, "take_profit": [], "confidence": None,
            "reasoning": ""}


def _loop() -> None:
    log.info("alert scanner started: %s %s every %ss",
             ALERT_SYMBOLS, ALERT_TIMEFRAME, SCAN_SECONDS)
    while True:
        for sym in ALERT_SYMBOLS:
            try:
                res = scan_once(sym, ALERT_TIMEFRAME)
                _state["last_error"] = None
                if res.get("triggered"):
                    log.info("alert scan %s: %s", sym, res)
            except Exception as e:
                _state["last_error"] = f"{type(e).__name__}: {e}"
                log.warning("alert scan failed for %s: %s", sym, e)
        _state["scans"] += 1
        _state["last_scan"] = dt.datetime.now(dt.timezone.utc).isoformat()
        time.sleep(SCAN_SECONDS)


def start() -> None:
    """Start the scanner once, from the app's startup hook."""
    with _lock:
        if _state["running"]:
            return
        init_db()
        if not ALERTS_ON:
            log.info("alerts disabled (set ALERTS_ENABLED=1 to turn on)")
            return
        _state["running"] = True
        threading.Thread(target=_loop, daemon=True, name="alert-scanner").start()


def status() -> dict:
    return {
        "enabled": ALERTS_ON,
        "running": _state["running"],
        "push_channels": channels() or ["none configured"],
        "symbols": ALERT_SYMBOLS,
        "timeframe": ALERT_TIMEFRAME,
        "scan_seconds": SCAN_SECONDS,
        "min_confidence": MIN_CONFIDENCE,
        "scans": _state["scans"],
        "triggers": _state["triggers"],
        "pushed": _state["pushed"],
        "last_scan": _state["last_scan"],
        "last_error": _state["last_error"],
        "db": str(DB_PATH),
        **stats(),
    }
