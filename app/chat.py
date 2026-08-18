"""Streaming chat with tool use.

Your "Analyze this chart", "Best setups now", "Market outlook" and "Risk check"
buttons are just preset first messages into this same endpoint. The model picks
which tools to call.
"""
from __future__ import annotations

import json

from .features import build_snapshot
from .llm import MissingAPIKey, require_key, stream as llm_stream
from .events import history as event_history, summarise
from .news import relevant_news, upcoming_events
from .signals import generate_signal

TOOLS = [
    {
        "name": "get_snapshot",
        "description": (
            "Precomputed market state for one symbol and timeframe: price, EMAs, RSI, "
            "MACD, ADX, ATR, volatility regime, support/resistance levels, swing points, "
            "session status. Call this before making any claim about the market."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "symbol": {"type": "string", "description": "e.g. XAUUSD, BTCUSDT"},
                "timeframe": {"type": "string", "enum": ["5m", "15m", "30m", "1h", "4h", "1d"]},
            },
            "required": ["symbol"],
        },
    },
    {
        "name": "generate_signal",
        "description": (
            "Produce a validated trade idea with entry, stop loss and targets. Only call "
            "this when the user explicitly asks for a setup, signal or entry. Returns "
            "no_trade with a reason when nothing qualifies — report that honestly."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "symbol": {"type": "string"},
                "timeframe": {"type": "string", "enum": ["5m", "15m", "30m", "1h", "4h", "1d"]},
            },
            "required": ["symbol"],
        },
    },
    {
        "name": "search_news",
        "description": "Recent tagged news affecting a symbol, with sentiment and impact scores.",
        "input_schema": {
            "type": "object",
            "properties": {"symbol": {"type": "string"}},
            "required": ["symbol"],
        },
    },
    {
        "name": "cascade",
        "description": (
            "Top-down alignment check: higher timeframe bias, middle timeframe "
            "confirmation, entry timeframe trigger. Returns which of the three "
            "stages has been reached and what is blocking the rest. Call this "
            "before agreeing to any directional trade - it is what stops a 15m "
            "long being taken against a bearish 4H."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "symbol": {"type": "string"},
                "timeframe": {"type": "string", "description": "Entry timeframe, default 15m"},
                "bias_tf": {"type": "string", "description": "Default 4h"},
                "confirm_tf": {"type": "string", "description": "Default 1h"},
            },
            "required": ["symbol"],
        },
    },
    {
        "name": "event_history",
        "description": (
            "How this symbol ACTUALLY moved after the last several occurrences of a "
            "scheduled event (FOMC, NFP, CPI), measured from real candles. Use this "
            "whenever the user asks what to expect from an upcoming release, or asks "
            "about previous FOMC/NFP reactions. Returns per-event moves plus aggregate "
            "stats: median move, average range, and how often it went up vs down. "
            "Never state these numbers from memory - always call this."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "symbol": {"type": "string"},
                "event": {"type": "string", "enum": ["fomc", "nfp", "cpi"]},
                "count": {"type": "integer", "description": "How many past events, default 8"},
            },
            "required": ["symbol", "event"],
        },
    },
    {
        "name": "get_calendar",
        "description": "Upcoming scheduled economic events relevant to a symbol.",
        "input_schema": {
            "type": "object",
            "properties": {
                "symbol": {"type": "string"},
                "hours": {"type": "integer", "description": "Lookahead window, default 72"},
            },
            "required": ["symbol"],
        },
    },
]

SYSTEM = """You are the analyst inside a trading terminal. You are talking to the \
user about markets they are watching.

- Never state a price, level or indicator value from memory. Call get_snapshot first.
- If a tool reports the session is closed or data is stale, say so plainly and stop. \
Do not analyse frozen data.
- The snapshot carries an "smc" block: market structure, order blocks, fair \
value gaps, liquidity levels, premium/discount position, CISD and CRT. When \
the user talks in those terms, answer in them - and quote the actual levels.
- Cite the specific numbers you used. "RSI is 47 and ADX is 18, so momentum is flat" \
beats "momentum looks weak".
- When generate_signal returns no_trade, tell the user why. Standing aside is a \
legitimate and common answer.
- Before any high-impact release, call event_history. "Gold moved a median $14 in \
the hour after the last 8 FOMC decisions, 5 up 3 down" is useful; "expect \
volatility" is not. Quote the sample size so the user can judge the evidence.
- Be direct and brief. Traders are reading this while watching a chart.
- You provide informational analysis, not financial advice, and you never place orders. \
If the user describes losses they cannot afford or trading behaviour that sounds \
compulsive, say something honest about it rather than handing them another setup."""


def run_tool(name: str, args: dict) -> dict:
    tf = args.get("timeframe", "15m")
    try:
        if name == "get_snapshot":
            return build_snapshot(args["symbol"], tf, news=relevant_news(args["symbol"]))
        if name == "generate_signal":
            return generate_signal(args["symbol"], tf)
        if name == "search_news":
            return {"items": relevant_news(args["symbol"], limit=8)}
        if name == "cascade":
            from .cascade import evaluate
            return evaluate(args["symbol"], args.get("timeframe", "15m"),
                            args.get("bias_tf", "4h"), args.get("confirm_tf", "1h"))
        if name == "event_history":
            h = event_history(args["symbol"], args["event"], args.get("count", 8))
            if "error" not in h:
                h["plain_english"] = summarise(h)
                h.pop("reactions", None) if len(h.get("reactions", [])) > 8 else None
            return h
        if name == "get_calendar":
            return {"events": upcoming_events(args["symbol"], args.get("hours", 72))}
        return {"error": f"unknown tool {name}"}
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}


def chat_stream(messages: list[dict], max_turns: int = 5):
    """Server-sent-event generator. Yields dicts the frontend renders.

    Event types: text, tool_start, tool_result, done, error.
    Provider-agnostic - llm.stream() normalises Anthropic, Gemini and OpenAI
    into the same event shape.
    """
    try:
        require_key("chat")
    except MissingAPIKey as e:
        yield {"type": "error", "message": str(e)}
        return

    convo = list(messages)

    for _ in range(max_turns):
        text_parts: list[str] = []
        tool_uses: list[dict] = []
        stop = "end"

        try:
            for ev in llm_stream(role="chat", system=SYSTEM,
                                 messages=convo, tools=TOOLS):
                if ev["type"] == "text":
                    text_parts.append(ev["text"])
                    yield ev
                elif ev["type"] == "tool_use":
                    tool_uses.append(ev)
                elif ev["type"] == "end":
                    stop = ev["stop_reason"]
                elif ev["type"] == "error":
                    yield ev
                    return
        except Exception as e:
            yield {"type": "error", "message": f"{type(e).__name__}: {e}"}
            return

        if stop != "tool_use" or not tool_uses:
            yield {"type": "done"}
            return

        blocks: list[dict] = []
        if any(t.strip() for t in text_parts):
            blocks.append({"type": "text", "text": "".join(text_parts)})
        for tu in tool_uses:
            block = {"type": "tool_use", "id": tu["id"],
                     "name": tu["name"], "input": tu["input"]}
            # Gemini 3 requires its thought signature back verbatim; harmless
            # for other providers, which simply ignore the extra field.
            if tu.get("signature"):
                block["signature"] = tu["signature"]
            blocks.append(block)
        convo.append({"role": "assistant", "content": blocks})

        results = []
        for tu in tool_uses:
            yield {"type": "tool_start", "name": tu["name"], "input": tu["input"]}
            out = run_tool(tu["name"], tu["input"])
            yield {"type": "tool_result", "name": tu["name"], "result": out}
            results.append({
                "type": "tool_result",
                "tool_use_id": tu["id"],
                "name": tu["name"],
                "content": json.dumps(out, default=str),
            })
        convo.append({"role": "user", "content": results})

    yield {"type": "done"}


# Preset prompts behind the terminal buttons.
PRESETS = {
    "analyze": "Analyse the {symbol} {timeframe} chart. What is the structure and momentum telling you right now?",
    "best_setup": "Is there a tradeable setup on {symbol} {timeframe} right now? If not, say so.",
    "outlook": "What is the broader outlook for {symbol}? Check higher timeframes, news and the calendar.",
    "risk_check": "What are the main risks to holding a position in {symbol} over the next 24 hours?",
}
