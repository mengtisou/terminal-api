"""Signal generation.

Flow: gate -> snapshot -> model (structured output) -> validate -> maybe retry.
The validator is the important part. A model will happily return an entry 400
points away from the current price with a stop on the wrong side, and it will
sound completely confident while doing it. Never show an unvalidated signal.
"""
from __future__ import annotations

import datetime as dt
import logging
import os
from dataclasses import dataclass

from .config import settings
from .features import build_snapshot
from .llm import (LLMError, council, council_enabled, json_call,
                  review_enabled, reviewer_for)
from .events import classify, history as event_history, summarise
from .news import in_news_blackout, relevant_news, upcoming_events

log = logging.getLogger(__name__)

SIGNAL_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "bias": {"type": "string", "enum": ["long", "short", "no_trade"]},
        "entry": {"type": ["number", "null"]},
        "stop_loss": {"type": ["number", "null"]},
        "take_profit": {"type": "array", "items": {"type": "number"}},
        "confidence": {"type": "number", "description": "0.0 to 1.0"},
        "invalidation": {"type": "string", "description": "One line: what proves this wrong."},
        "reasoning": {"type": "string", "description": "3-5 sentences citing specific snapshot values."},
        "key_levels_used": {"type": "array", "items": {"type": "number"}},
    },
    "required": ["bias", "entry", "stop_loss", "take_profit", "confidence", "invalidation", "reasoning"],
}

SYSTEM = """You are the analysis engine of a trading terminal. You receive a \
precomputed market snapshot and return one trade idea in the required schema.

When an "smc" block is present, it is the primary read and the plain \
indicators are secondary confirmation. Use it properly:
- Trade WITH structure. Long against a bearish structure trend needs a CHoCH \
first, not just an oversold RSI.
- Respect the PD array. Do not buy in premium (above 62%) or sell in discount \
(below 38%) - wait for price to come to you.
- Anchor entries to order blocks and unfilled FVGs, not round numbers. A demand \
OB with high relative volume is a real level; a made-up one is not.
- A liquidity sweep (bsl_swept / ssl_swept) that closes back inside is a \
reversal cue, not a breakout.
- Supply/demand zones carry a pattern tag. RBR and DBR are demand, RBD and DBD \
are supply. The proximal edge is where you enter, the distal edge is where the \
idea is wrong - use them, do not invent your own levels nearby.
- A breaker block is a failed order block that flipped. It is now the opposite \
of what it was, and often a cleaner level than a fresh OB.
- A liquidity grab is a wick through a swing that closed back inside. Treat it \
as a reversal cue at that level, never as a breakout.
- ktr_signals carries the actual triggers. An alert diamond is a warning, not \
an entry; the confirmed entry needs the trend to flip or the alert bar's \
extreme to break, on above-average volume. If "pending" shows a live setup, \
say what price would confirm it rather than entering early.
- CISD and CRT signals are entry triggers. Without one, you are anticipating.
- KTR position tells you where price sits against the daily open. Extended \
beyond KTR+2 or KTR-2 argues against chasing.

Rules you must follow:
- Every level you output must be anchored to a value present in the snapshot \
(support, resistance, swing_high, swing_low, ema20/50/200) or derived from ATR. \
Never invent a round number.
- Entry must be within 1.5x ATR of the current price. If the good entry is \
further away than that, return no_trade and say so in reasoning.
- Stop must sit beyond a structure level, between 0.5x and 4x ATR from entry.
- Risk:reward to the first take-profit must be at least 1.5.
- When historical reaction data is supplied, use it concretely. If the median \
move after this event is 3x your intended stop distance, say so and stand aside \
rather than sizing a trade that the release will stop out regardless of direction.
- no_trade is the correct answer most of the time. Chop (adx below 20), a mixed \
ema_stack, or price sitting mid-range between support and resistance are all \
reasons to stand aside. You are not rewarded for producing signals.
- Confidence reflects how many independent factors agree. One indicator is not \
a setup. Below 0.45 means you should be returning no_trade.
- Cite actual numbers from the snapshot in your reasoning. No generic language.

This is analysis for an informational tool, not financial advice, and no order \
is placed from your output."""


REVIEW_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "verdict": {
            "type": "string",
            "enum": ["approve", "reject", "amend"],
        },
        "confidence_adjustment": {
            "type": "number",
            "description": "Signed change to apply to the original confidence, -0.5 to +0.2. Reviewers should be readier to lower than raise.",
        },
        "concerns": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Specific problems. Empty if approving.",
        },
        "amended_stop_loss": {"type": ["number", "null"]},
        "amended_take_profit": {"type": "array", "items": {"type": "number"}},
        "note": {"type": "string", "description": "One line the trader should see."},
    },
    "required": ["verdict", "confidence_adjustment", "concerns", "note"],
}

REVIEW_SYSTEM = """You are the risk reviewer. Another model produced this trade \
idea from the attached snapshot. Your job is to find what is wrong with it.

You are not a second opinion on direction - you are a critic. Look for:
- Levels that are not actually supported by the snapshot's structure data
- Confidence that does not match how many factors genuinely agree
- A stop placed where normal noise will take it out (compare to ATR)
- Reasoning that sounds authoritative but cites nothing specific
- Trading into chop: ADX under 20, mixed ema_stack, price mid-range
- Ignoring an upcoming high-impact event or the historical reaction data

Verdicts:
- approve  the idea holds up. Say so briefly and stop.
- amend    the direction is sound but a level is wrong. Supply the corrected \
stop or targets.
- reject   the setup should not be taken.

Approving a weak setup is a worse failure than rejecting a decent one. But do \
not manufacture objections to look useful - if it is a clean setup, approve it."""


@dataclass
class ValidationResult:
    ok: bool
    errors: list[str]


def validate(sig: dict, snap: dict) -> ValidationResult:
    """Every rule here exists because a model broke it at some point."""
    r = settings.risk
    errs: list[str] = []

    if sig["bias"] == "no_trade":
        return ValidationResult(True, [])

    entry, sl = sig.get("entry"), sig.get("stop_loss")
    tps = sig.get("take_profit") or []
    price = snap["price"]
    a = snap["volatility"]["atr14"]

    if entry is None or sl is None or not tps:
        return ValidationResult(False, ["directional signal missing entry, stop or target"])

    # 1. Stop on the correct side.
    if sig["bias"] == "long" and sl >= entry:
        errs.append(f"long stop_loss {sl} must be below entry {entry}")
    if sig["bias"] == "short" and sl <= entry:
        errs.append(f"short stop_loss {sl} must be above entry {entry}")

    # 2. Targets on the correct side, ordered outward.
    for tp in tps:
        if sig["bias"] == "long" and tp <= entry:
            errs.append(f"long take_profit {tp} must be above entry {entry}")
        if sig["bias"] == "short" and tp >= entry:
            errs.append(f"short take_profit {tp} must be below entry {entry}")

    # 3. Entry near the live price — catches hallucinated levels.
    if a and abs(entry - price) > r.max_entry_distance_atr * a:
        errs.append(
            f"entry {entry} is {abs(entry - price) / a:.1f} ATR from price {price}, "
            f"max is {r.max_entry_distance_atr}"
        )

    # 4. Stop distance sane relative to volatility.
    if a:
        d = abs(entry - sl) / a
        if d < r.min_stop_distance_atr:
            errs.append(f"stop {d:.2f} ATR away, too tight (min {r.min_stop_distance_atr})")
        if d > r.max_stop_distance_atr:
            errs.append(f"stop {d:.2f} ATR away, too wide (max {r.max_stop_distance_atr})")

    # 5. Risk:reward.
    risk = abs(entry - sl)
    reward = abs(tps[0] - entry)
    if risk > 0:
        rr = reward / risk
        if rr < r.min_risk_reward:
            errs.append(f"risk:reward {rr:.2f} below minimum {r.min_risk_reward}")
        sig["risk_reward"] = round(rr, 2)

    # 6. Confidence floor.
    if sig.get("confidence", 0) < r.min_confidence:
        errs.append(f"confidence {sig.get('confidence')} below floor {r.min_confidence}")

    return ValidationResult(not errs, errs)


def generate_signal(symbol: str, timeframe: str = "15m") -> dict:
    """Public entry point. Always returns a dict; never raises on a bad model
    response — it degrades to no_trade."""
    news = relevant_news(symbol)
    snap = build_snapshot(symbol, timeframe, news=news)
    events = upcoming_events(symbol)

    # --- Pre-model gates. Cheapest possible rejection. ---
    if snap.get("data_source") == "synthetic" and not _dev_signals_allowed():
        return _blocked(
            snap, "synthetic_data",
            "This snapshot came from the synthetic development provider, not a live "
            "feed. Configure a real provider before generating signals.",
        )

    if snap["session"]["data_stale"]:
        return _blocked(
            snap, "market_closed",
            f"Data is stale ({snap['session']['state']}, last candle "
            f"{snap['session']['age_seconds']}s ago). Nothing to analyse until the session reopens.",
        )

    blackout = in_news_blackout(symbol)
    if blackout:
        return _blocked(
            snap, "news_blackout",
            f"{blackout['title']} in {blackout['minutes_away']} minutes. "
            "Spreads widen and stops get run around high-impact events.",
        )

    # --- Model call ---
    user = _prompt(snap, events)
    try:
        if council_enabled():
            sig, _votes = council(system=SYSTEM, user=user,
                                  schema=SIGNAL_SCHEMA, max_tokens=1200)
        else:
            sig = json_call(role="reasoning", system=SYSTEM, user=user,
                            schema=SIGNAL_SCHEMA, max_tokens=1200)
    except LLMError as e:
        return _blocked(snap, "model_error", str(e))

    result = validate(sig, snap)

    # --- One repair attempt, feeding the errors back ---
    if not result.ok:
        try:
            sig = json_call(
                role="reasoning", system=SYSTEM,
                user=user + "\n\nYour previous answer was rejected by the risk validator:\n"
                + "\n".join(f"- {e}" for e in result.errors)
                + "\n\nFix these or return no_trade. Do not force a setup.",
                schema=SIGNAL_SCHEMA, max_tokens=1200,
            )
            result = validate(sig, snap)
        except LLMError as e:
            return _blocked(snap, "model_error", str(e))

    if not result.ok:
        return _blocked(snap, "validation_failed", "; ".join(result.errors))

    # --- reviewer pass -------------------------------------------------
    # A second model, from a different provider, critiques the idea.
    review = None
    if review_enabled() and sig["bias"] != "no_trade":
        review = _review(sig, snap, user)
        if review and review.get("verdict") == "reject":
            blocked = _blocked(snap, "review_rejected",
                               review.get("note") or "; ".join(review.get("concerns", [])))
            blocked["review"] = review
            return blocked
        if review:
            if review.get("verdict") == "amend":
                if review.get("amended_stop_loss"):
                    sig["stop_loss"] = review["amended_stop_loss"]
                if review.get("amended_take_profit"):
                    sig["take_profit"] = review["amended_take_profit"]
            adj = max(-0.5, min(0.2, review.get("confidence_adjustment", 0)))
            sig["confidence"] = round(max(0.0, min(1.0, sig["confidence"] + adj)), 2)

            # Amended levels must clear the same validator as the original.
            recheck = validate(sig, snap)
            if not recheck.ok:
                blocked = _blocked(snap, "review_amendment_invalid",
                                   "; ".join(recheck.errors))
                blocked["review"] = review
                return blocked
            if sig["confidence"] < settings.risk.min_confidence:
                blocked = _blocked(
                    snap, "review_lowered_confidence",
                    f"Reviewer cut confidence to {sig['confidence']}. "
                    + (review.get("note") or ""))
                blocked["review"] = review
                return blocked

    sig.update({
        "review": review,
        "symbol": snap["symbol"], "timeframe": timeframe,
        "price_at_generation": snap["price"],
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "status": "ok",
        "provider": sig.pop("_provider", None),
        "disclaimer": "Informational analysis, not financial advice.",
    })
    return sig


def _dev_signals_allowed() -> bool:
    """Explicit opt-in for generating signals from synthetic data, so you can
    exercise the model path locally without a data provider."""
    return os.getenv("DEV_ALLOW_SYNTHETIC_SIGNALS", "0") == "1"


def _review(sig: dict, snap: dict, original_prompt: str) -> dict | None:
    """Second-model critique. Returns None if no distinct reviewer is available."""
    import json as _json

    generator = (sig.get("_provider") or "").split("/")[0]
    pair = reviewer_for(generator)
    if not pair:
        log.debug("no reviewer available distinct from %s", generator)
        return None
    name, model = pair

    clean = {k: v for k, v in sig.items() if not k.startswith("_")}
    try:
        out = json_call(
            role="review", model=model,
            system=REVIEW_SYSTEM,
            user=(f"{original_prompt}\n\nProposed trade idea:\n"
                  f"{_json.dumps(clean, indent=1)}\n\nReview it."),
            schema=REVIEW_SCHEMA, max_tokens=900,
        )
        out["reviewer"] = f"{name}/{model}"
        out.pop("_provider", None)
        return out
    except LLMError as e:
        log.warning("review step failed, passing through: %s", e)
        return None


def _prompt(snap: dict, events: list[dict]) -> str:
    import json as _json
    parts = [f"Market snapshot:\n{_json.dumps(snap, indent=1)}"]
    if events:
        parts.append(f"Upcoming events:\n{_json.dumps(events, indent=1)}")

        # If a major release lands within 24h, show how this instrument
        # actually behaved the last several times - measured, not recalled.
        for e in events[:3]:
            if e.get("impact", 1) < 3 or e.get("minutes_away", 0) > 1440:
                continue
            kind = classify(e.get("title", ""))
            if not kind:
                continue
            h = event_history(snap["symbol"], kind, count=8)
            if "error" not in h:
                parts.append(
                    f"Historical reaction to {h['label']} "
                    f"(n={h['sample_size']}):\n"
                    f"{summarise(h)}\n"
                    f"{_json.dumps(h['stats'], indent=1)}")
            break
    parts.append("Return one trade idea, or no_trade if nothing qualifies.")
    return "\n\n".join(parts)


def _blocked(snap: dict, reason: str, message: str) -> dict:
    return {
        "symbol": snap["symbol"], "timeframe": snap["timeframe"],
        "bias": "no_trade", "entry": None, "stop_loss": None, "take_profit": [],
        "confidence": 0.0, "status": reason, "reasoning": message,
        "invalidation": "", "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "disclaimer": "Informational analysis, not financial advice.",
    }
