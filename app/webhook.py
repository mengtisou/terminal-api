"""TradingView alert webhook receiver.

TradingView cannot be used as a data source - there is no public API and their
feeds are licensed from brokers. But alerts can push INTO your backend, which
is officially supported.

Flow:
  Pine Script detects a setup
    -> TradingView fires an alert with a JSON payload
    -> this endpoint receives it
    -> your feature engine adds context
    -> Claude reviews it and produces entry/SL/TP with reasoning
    -> validated, then pushed to the UI / Telegram

Set the alert's "Webhook URL" to:
    https://your-domain.com/webhook/tradingview?secret=YOUR_SECRET

TradingView only sends webhooks to port 80/443 on a public domain. For local
testing use ngrok:  ngrok http 8000
"""
from __future__ import annotations

import datetime as dt
import hmac
import logging
import os

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from .features import build_snapshot
from .llm import LLMError, json_call
from .news import relevant_news, upcoming_events
from .signals import SIGNAL_SCHEMA, _blocked, validate

log = logging.getLogger(__name__)
router = APIRouter()

WEBHOOK_SECRET = os.getenv("TRADINGVIEW_WEBHOOK_SECRET", "")

# Recent alerts, newest first. Swap for a real table in production.
ALERT_LOG: list[dict] = []


class TVAlert(BaseModel):
    """Shape of the JSON you put in the alert message box."""

    symbol: str
    timeframe: str = "15m"
    action: str = Field(description="buy, sell, or close")
    price: float | None = None
    strategy: str | None = None
    note: str | None = None


REVIEW_SYSTEM = """A TradingView alert has fired. Your job is to REVIEW it, not \
to trust it.

The alert tells you what a Pine Script indicator detected. The snapshot tells \
you the actual market state. Decide whether the alert is worth acting on.

- If the snapshot contradicts the alert (alert says buy, but the ema_stack is \
bearish, ADX is under 20 and price is under resistance), return no_trade and \
explain the conflict. Disagreeing with the alert is the whole point of this step.
- If the alert is confirmed, produce entry, stop and targets anchored to levels \
present in the snapshot.
- Entry must be within 1.5x ATR of current price. Stop between 0.5x and 4x ATR. \
Risk:reward at least 1.5 to the first target.
- Mention the alert's strategy name in your reasoning so the user knows what fired.

This is informational analysis, not financial advice, and no order is placed."""


@router.post("/webhook/tradingview")
async def tradingview_webhook(request: Request, secret: str = ""):
    """Receive, verify, enrich, review, validate."""
    if WEBHOOK_SECRET and not hmac.compare_digest(secret, WEBHOOK_SECRET):
        log.warning("rejected webhook with bad secret from %s", request.client.host)
        raise HTTPException(401, "invalid secret")

    raw = await request.body()
    try:
        alert = TVAlert.model_validate_json(raw)
    except Exception as e:
        raise HTTPException(400, f"could not parse alert payload: {e}")

    log.info("tradingview alert: %s %s %s", alert.action, alert.symbol, alert.strategy)

    record = {
        "received_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "alert": alert.model_dump(),
    }

    if alert.action == "close":
        record["result"] = {"status": "close_signal", "bias": "no_trade"}
        ALERT_LOG.insert(0, record)
        return record["result"]

    result = review_alert(alert)
    record["result"] = result
    ALERT_LOG.insert(0, record)
    del ALERT_LOG[200:]
    return result


def review_alert(alert: TVAlert) -> dict:
    """Enrich the alert with real market context, then have Claude judge it."""
    snap = build_snapshot(
        alert.symbol, alert.timeframe, news=relevant_news(alert.symbol)
    )

    if snap["session"]["data_stale"]:
        return _blocked(
            snap, "market_closed",
            f"Alert received but the {alert.symbol} session is closed "
            f"(last candle {snap['session']['age_seconds']}s ago). Ignored.",
        )

    # Price sanity: if TradingView's price is far from ours, the feeds disagree
    # and one of them is wrong. Do not guess which.
    if alert.price and snap["volatility"]["atr14"]:
        drift = abs(alert.price - snap["price"]) / snap["volatility"]["atr14"]
        if drift > 3:
            return _blocked(
                snap, "price_mismatch",
                f"Alert price {alert.price} is {drift:.1f} ATR from our feed's "
                f"{snap['price']}. Feeds disagree - not acting on this.",
            )

    import json as _json
    user = (
        f"TradingView alert:\n{_json.dumps(alert.model_dump(), indent=1)}\n\n"
        f"Market snapshot:\n{_json.dumps(snap, indent=1)}\n\n"
        f"Upcoming events:\n{_json.dumps(upcoming_events(alert.symbol), indent=1)}\n\n"
        "Confirm or reject this alert."
    )

    try:
        sig = json_call(
            role="reasoning", system=REVIEW_SYSTEM, user=user,
            schema=SIGNAL_SCHEMA, max_tokens=1200,
        )
    except LLMError as e:
        return _blocked(snap, "model_error", str(e))

    check = validate(sig, snap)
    if not check.ok:
        return _blocked(snap, "validation_failed", "; ".join(check.errors))

    sig.update({
        "symbol": snap["symbol"],
        "timeframe": alert.timeframe,
        "source": "tradingview_alert",
        "strategy": alert.strategy,
        "price_at_generation": snap["price"],
        "data_source": snap.get("data_source"),
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "status": "ok",
        "provider": sig.pop("_provider", None),
        "disclaimer": "Informational analysis, not financial advice.",
    })
    return sig


@router.get("/webhook/alerts")
def recent_alerts(limit: int = 20):
    """What has fired recently, and what the review decided."""
    return {"alerts": ALERT_LOG[:limit]}
