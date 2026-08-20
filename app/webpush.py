"""Browser push notifications for the terminal itself.

This is the same mechanism TradingView uses in a browser tab: the page asks
permission once, registers a service worker, and the server pushes to it
through the browser vendor's own push service. Notifications then arrive on
the phone's lock screen even with Chrome closed - no extra app to install.

Two things are non-negotiable and worth stating plainly:
  - HTTPS only. Service workers refuse to register over plain http, which is
    fine here since both origins already have certificates.
  - The service worker must be served from the SAME origin as the page. A
    worker at api.realflylink.com cannot receive pushes for a page loaded from
    vercel.app. Whichever address you actually open has to serve /sw.js.

VAPID keys identify this server to the push service. They are generated once on
first use and kept in the data directory, which is gitignored - regenerating
them silently invalidates every existing subscription.
"""

from __future__ import annotations

import base64
import json
import logging
import os
import sqlite3
from pathlib import Path

log = logging.getLogger(__name__)

DATA_DIR = Path(os.getenv("ALERTS_DB", "/root/terminal-api/data/signals.db")).parent
VAPID_PEM = DATA_DIR / "vapid_private.pem"
VAPID_PUB = DATA_DIR / "vapid_public.txt"
VAPID_SUBJECT = os.getenv("VAPID_SUBJECT", "mailto:admin@realflylink.com")


def _ensure_keys() -> tuple[str, str]:
    """(public_key_b64url, private_pem_path). Generates once, then reuses."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if VAPID_PEM.exists() and VAPID_PUB.exists():
        return VAPID_PUB.read_text().strip(), str(VAPID_PEM)

    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric import ec

    key = ec.generate_private_key(ec.SECP256R1())
    VAPID_PEM.write_bytes(key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption()))
    # The browser wants the raw uncompressed EC point, base64url, unpadded.
    raw = key.public_key().public_bytes(
        encoding=serialization.Encoding.X962,
        format=serialization.PublicFormat.UncompressedPoint)
    pub = base64.urlsafe_b64encode(raw).rstrip(b"=").decode()
    VAPID_PUB.write_text(pub)
    log.info("generated new VAPID keypair in %s", DATA_DIR)
    return pub, str(VAPID_PEM)


def public_key() -> str:
    return _ensure_keys()[0]


# ---------------------------------------------------------------- storage
def _db() -> sqlite3.Connection:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    c = sqlite3.connect(DATA_DIR / "signals.db", timeout=10)
    c.row_factory = sqlite3.Row
    return c


def init_db() -> None:
    with _db() as c:
        c.execute("""
            CREATE TABLE IF NOT EXISTS push_subs (
                endpoint   TEXT PRIMARY KEY,
                sub_json   TEXT NOT NULL,
                created_at TEXT NOT NULL,
                label      TEXT
            )""")


def add_subscription(sub: dict, label: str = "") -> bool:
    """Idempotent: re-subscribing the same browser replaces its row."""
    import datetime as dt
    endpoint = sub.get("endpoint")
    if not endpoint:
        return False
    init_db()
    with _db() as c:
        c.execute("""INSERT INTO push_subs (endpoint, sub_json, created_at, label)
                     VALUES (?,?,?,?)
                     ON CONFLICT(endpoint) DO UPDATE SET sub_json=excluded.sub_json""",
                  (endpoint, json.dumps(sub),
                   dt.datetime.now(dt.timezone.utc).isoformat(), label[:60]))
    return True


def remove_subscription(endpoint: str) -> None:
    with _db() as c:
        c.execute("DELETE FROM push_subs WHERE endpoint=?", (endpoint,))


def subscriptions() -> list[dict]:
    init_db()
    with _db() as c:
        return [json.loads(r["sub_json"])
                for r in c.execute("SELECT sub_json FROM push_subs")]


def count() -> int:
    init_db()
    with _db() as c:
        return c.execute("SELECT COUNT(*) n FROM push_subs").fetchone()["n"]


# ---------------------------------------------------------------- sending
def send(title: str, body: str, url: str = "/") -> tuple[bool, dict]:
    """Push to every registered browser. Returns (any_succeeded, per-endpoint)."""
    subs = subscriptions()
    if not subs:
        return False, {"error": "no browser is subscribed yet"}

    try:
        from pywebpush import WebPushException, webpush
    except ImportError:
        return False, {"error": "pywebpush not installed "
                                "(pip install pywebpush --break-system-packages)"}

    pub, pem = _ensure_keys()
    payload = json.dumps({"title": title, "body": body, "url": url})
    results, any_ok = {}, False

    for sub in subs:
        endpoint = sub.get("endpoint", "?")
        short = endpoint.split("/")[-1][:12]
        try:
            webpush(subscription_info=sub, data=payload,
                    vapid_private_key=pem,
                    vapid_claims={"sub": VAPID_SUBJECT})
            results[short] = "sent"
            any_ok = True
        except WebPushException as e:
            code = getattr(e.response, "status_code", None)
            # 404/410 mean the browser threw the subscription away - uninstalled,
            # permission revoked, or profile cleared. Drop it rather than
            # retrying it forever on every alert.
            if code in (404, 410):
                remove_subscription(endpoint)
                results[short] = f"expired ({code}), removed"
            else:
                results[short] = f"failed: {code or e}"
        except Exception as e:
            results[short] = f"{type(e).__name__}: {e}"

    return any_ok, results
