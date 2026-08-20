"""Model routing.

Three roles, each independently routed to a provider + model:

    cheap      news tagging, dedupe. High volume, must be cheap.
    reasoning  signal generation. Needs actual reasoning.
    chat       the terminal assistant. Latency matters.

Each role has an ordered chain. The first provider with a key that succeeds
wins; the rest are failover. Change the chain in .env, no code edits:

    REASONING_CHAIN=gemini,anthropic
    CHAT_CHAIN=anthropic,gemini

Set COUNCIL=1 and the reasoning role queries every configured provider and
only returns a directional signal when they agree - the "analyst council"
pattern. Disagreement becomes no_trade.
"""
from __future__ import annotations

import json
import logging
import os
import threading
import time
from pathlib import Path

from . import providers
from .providers import MissingKey, ProviderError

log = logging.getLogger(__name__)

# --- rate limiting -----------------------------------------------------------
# Free tiers cap requests per MINUTE, not just per day. Firing six tagging
# calls at once trips that instantly, and every one comes back 429. A small
# token bucket per provider keeps a batch inside the limit instead.

_RPM = {
    "gemini": int(os.getenv("GEMINI_RPM", "8")),      # free tier is ~10/min
    "anthropic": int(os.getenv("ANTHROPIC_RPM", "50")),
    "openai": int(os.getenv("OPENAI_RPM", "50")),
    "deepseek": int(os.getenv("DEEPSEEK_RPM", "50")),
    "xai": int(os.getenv("XAI_RPM", "50")),
    # Free OpenRouter models are throttled hard (roughly 20/min, and a shared
    # daily cap). Paid ones are far looser - raise this if you point it at one.
    "openrouter": int(os.getenv("OPENROUTER_RPM", "15")),
}
_last_call: dict[str, float] = {}
_rate_lock = threading.Lock()


def _throttle(provider: str) -> None:
    """Space calls so a provider's per-minute cap is not exceeded."""
    rpm = _RPM.get(provider)
    if not rpm:
        return
    gap = 60.0 / rpm
    with _rate_lock:
        now = time.monotonic()
        wait = gap - (now - _last_call.get(provider, 0.0))
        if wait > 0:
            time.sleep(wait)
            now = time.monotonic()
        _last_call[provider] = now


class LLMError(RuntimeError):
    pass


class MissingAPIKey(LLMError):
    """Raised when no provider in the chain has a usable key."""


_KEY_HELP = (
    "No LLM provider is configured.\n"
    "  Easiest fix - this writes .env and live-tests the key:\n"
    "    python setup_key.py gemini <your-key>\n"
    "\n"
    "  Get a free Gemini key: https://aistudio.google.com/apikey\n"
    "  (New keys start with 'AQ.' - that is the current format, not an error.)\n"
    "\n"
    "  Then restart the server. --reload does not watch .env."
)

# model per (provider, role). Override any of these in .env.
DEFAULT_MODELS = {
    ("anthropic", "cheap"): "claude-haiku-4-5-20251001",
    ("anthropic", "reasoning"): "claude-sonnet-5",
    ("anthropic", "chat"): "claude-sonnet-5",
    # Google retires model names regularly. If these 404, run
    # `python setup_key.py gemini <key>` - it lists what your key can use
    # and pins a working name into .env.
    ("gemini", "cheap"): "gemini-3.6-flash",
    ("gemini", "reasoning"): "gemini-3.6-flash",
    ("gemini", "chat"): "gemini-3.6-flash",
    ("openai", "cheap"): "gpt-4o-mini",
    ("openai", "reasoning"): "gpt-4o",
    ("openai", "chat"): "gpt-4o",
    ("xai", "cheap"): "grok-4-fast",
    ("xai", "reasoning"): "grok-4",
    ("xai", "chat"): "grok-4",
    # The reviewer role. Deliberately a DIFFERENT provider from the generator
    # by default - a model reviewing its own output mostly agrees with itself.
    ("anthropic", "review"): "claude-sonnet-5",
    ("openai", "review"): "gpt-4o",
    ("gemini", "review"): "gemini-3.6-flash",
    ("xai", "review"): "grok-4",
    ("deepseek", "cheap"): "deepseek-chat",
    ("deepseek", "reasoning"): "deepseek-reasoner",
    ("deepseek", "chat"): "deepseek-chat",
    ("deepseek", "review"): "deepseek-reasoner",
    # OpenRouter models carry a vendor prefix. Overridable per role with
    # OPENROUTER_MODEL_CHEAP / _REASONING / _CHAT / _REVIEW in .env, since which
    # model is the best value there changes month to month.
    ("openrouter", "cheap"): os.getenv("OPENROUTER_MODEL_CHEAP",
                                       "google/gemini-2.0-flash-exp:free"),
    ("openrouter", "reasoning"): os.getenv("OPENROUTER_MODEL_REASONING",
                                           "openai/gpt-4o"),
    ("openrouter", "chat"): os.getenv("OPENROUTER_MODEL_CHAT", "openai/gpt-4o"),
    ("openrouter", "review"): os.getenv("OPENROUTER_MODEL_REVIEW", "openai/gpt-4o"),
}


# --- runtime overrides -------------------------------------------------------
# Set from the UI at /providers. Persisted so a restart keeps your choice.
# Precedence: runtime override > .env chain > built-in default.

_STATE_DIR = Path(os.getenv("STATE_DIR",
                            Path(__file__).resolve().parent.parent))
_OVERRIDE_FILE = _STATE_DIR / "runtime_config.json"
_lock = threading.Lock()
_overrides: dict = {}


def _load_overrides() -> None:
    global _overrides
    try:
        if _OVERRIDE_FILE.exists():
            _overrides = json.loads(_OVERRIDE_FILE.read_text())
            log.info("loaded runtime overrides: %s", _overrides)
    except Exception as e:
        log.warning("could not read %s: %s", _OVERRIDE_FILE, e)
        _overrides = {}


def _save_overrides() -> None:
    try:
        _OVERRIDE_FILE.write_text(json.dumps(_overrides, indent=1))
    except Exception as e:
        log.warning("could not write %s: %s", _OVERRIDE_FILE, e)


_load_overrides()


def set_routing(chains: dict[str, list[str]] | None = None,
                council: bool | None = None,
                review: bool | None = None,
                models: dict[str, str] | None = None) -> dict:
    """Change routing at runtime. Pass an empty list for a role to clear its
    override and fall back to .env."""
    with _lock:
        if chains is not None:
            for role, providers_list in chains.items():
                if role not in ("cheap", "reasoning", "chat", "review"):
                    raise ValueError(f"unknown role '{role}'")
                for name in providers_list:
                    providers.get(name)  # raises on unknown provider
                if providers_list:
                    _overrides.setdefault("chains", {})[role] = providers_list
                else:
                    _overrides.get("chains", {}).pop(role, None)
        if council is not None:
            _overrides["council"] = bool(council)
        if review is not None:
            _overrides["review"] = bool(review)
        if models is not None:
            for k, v in models.items():
                if v:
                    _overrides.setdefault("models", {})[k] = v
                else:
                    _overrides.get("models", {}).pop(k, None)
        _save_overrides()
    return status()


def clear_routing() -> dict:
    """Drop all UI overrides and go back to whatever .env says."""
    with _lock:
        _overrides.clear()
        _save_overrides()
    return status()


def _chain(role: str) -> list[str]:
    """Ordered provider list for a role. Free tier first by default, so a
    fresh install works without a paid key."""
    ui = _overrides.get("chains", {}).get(role)
    if ui:
        return list(ui)
    env = os.getenv(f"{role.upper()}_CHAIN", "")
    if env.strip():
        return [p.strip() for p in env.split(",") if p.strip()]
    return ["anthropic", "gemini", "openai", "xai", "deepseek"]


def _model(provider: str, role: str) -> str:
    key = f"{provider}_{role}"
    return (
        _overrides.get("models", {}).get(key)
        or os.getenv(f"{provider.upper()}_{role.upper()}_MODEL")
        or DEFAULT_MODELS.get((provider, role), "")
    )


def active(role: str) -> list[tuple[str, str]]:
    """(provider, model) pairs that are actually usable for this role."""
    return [
        (n, _model(n, role))
        for n in _chain(role)
        if providers.get(n).available() and _model(n, role)
    ]


def require_key(role: str = "chat") -> None:
    if not active(role):
        raise MissingAPIKey(_KEY_HELP)


def json_call(*, role="reasoning", system, user, schema, max_tokens=1500,
              retries=1, model=None) -> dict:
    """Structured output, walking the provider chain on failure."""
    chain = active(role)
    if not chain:
        raise MissingAPIKey(_KEY_HELP)

    errors = []
    for name, default_model in chain:
        p = providers.get(name)
        use = model or default_model
        for attempt in range(retries + 1):
            try:
                _throttle(name)
                out = p.json_call(model=use, system=system, user=user,
                                  schema=schema, max_tokens=max_tokens)
                log.info("%s role served by %s/%s", role, name, use)
                out["_provider"] = f"{name}/{use}"
                return out
            except (ProviderError, json.JSONDecodeError, KeyError) as e:
                msg = str(e)
                rate_limited = "429" in msg or "quota" in msg.lower() \
                    or "rate" in msg.lower()
                errors.append(f"{name}/{use}: {type(e).__name__}: {msg[:120]}")
                log.warning("%s failed for role %s: %s", name, role, msg[:200])
                if attempt < retries:
                    # Back off harder on a rate limit than on a transient error.
                    time.sleep((6.0 if rate_limited else 1.5) * (attempt + 1))
    raise LLMError("all providers failed - " + " | ".join(errors))


def stream(*, role="chat", system, messages, tools=None, max_tokens=2000):
    """Normalised streaming events. Falls through the chain on connect errors.

    Once a provider has started emitting text we do not switch - swapping
    mid-answer would produce a spliced, incoherent reply.
    """
    chain = active(role)
    if not chain:
        raise MissingAPIKey(_KEY_HELP)

    errors = []
    for name, default_model in chain:
        p = providers.get(name)
        started = False
        try:
            for ev in p.stream(model=default_model, system=system, messages=messages,
                               tools=tools, max_tokens=max_tokens):
                started = True
                yield ev
            return
        except (ProviderError, MissingKey) as e:
            errors.append(f"{name}: {e}")
            log.warning("stream provider %s failed: %s", name, e)
            if started:
                yield {"type": "error", "message": f"{name} dropped mid-response: {e}"}
                return
    raise LLMError("all providers failed - " + " | ".join(errors))


def council(*, system, user, schema, max_tokens=1500) -> tuple[dict, list[dict]]:
    """Ask every configured reasoning provider. Returns (verdict, all_votes).

    Agreement on direction raises confidence. Disagreement returns no_trade -
    if two models read the same snapshot and reach opposite conclusions, the
    setup is not clear enough to trade.
    """
    votes, failures = [], []
    members = active("reasoning")
    if not members:
        raise LLMError("no reasoning provider configured - set a key and "
                       "REASONING_CHAIN in .env")

    for name, model in members:
        try:
            v = providers.get(name).json_call(
                model=model, system=system, user=user,
                schema=schema, max_tokens=max_tokens)
            v["_provider"] = f"{name}/{model}"
            votes.append(v)
        except Exception as e:
            # Keep the reason. "no council member responded" on its own is
            # unactionable - the caller cannot tell a bad key from a wrong
            # model name from a rate limit.
            failures.append(f"{name}/{model}: {type(e).__name__}: {e}")
            log.warning("council member %s failed: %s", name, e)

    if not votes:
        raise LLMError("no council member responded | " + " | ".join(failures))
    if len(votes) == 1:
        return votes[0], votes

    biases = {v.get("bias") for v in votes}
    if len(biases) > 1:
        dissent = ", ".join(f"{v['_provider'].split('/')[0]}={v.get('bias')}" for v in votes)
        return {
            "bias": "no_trade", "entry": None, "stop_loss": None, "take_profit": [],
            "confidence": 0.0,
            "invalidation": "",
            "reasoning": f"Council disagreed ({dissent}). The setup is not clear "
                         f"enough to act on.",
            "_provider": "council",
        }, votes

    # Unanimous: take the most conservative numbers on the table.
    best = max(votes, key=lambda v: v.get("confidence", 0))
    best = dict(best)
    best["confidence"] = round(min(v.get("confidence", 0) for v in votes), 2)
    best["_provider"] = "council:" + ",".join(
        v["_provider"].split("/")[0] for v in votes)
    return best, votes


def review_enabled() -> bool:
    if "review" in _overrides:
        return bool(_overrides["review"])
    return os.getenv("REVIEW", "0") == "1"


def reviewer_for(generator: str) -> tuple[str, str] | None:
    """Pick a reviewer that is NOT the generator. Self-review is close to
    worthless - a model asked to critique its own answer usually endorses it."""
    for name, model in active("review"):
        if name != generator:
            return name, model
    return None


def council_enabled() -> bool:
    if "council" in _overrides:
        return bool(_overrides["council"])
    return os.getenv("COUNCIL", "0") == "1"


def status() -> dict:
    return {
        "providers": providers.status(),
        "roles": {r: [f"{n}/{m}" for n, m in active(r)]
                  for r in ("cheap", "reasoning", "chat", "review")},
        "chains": {r: _chain(r) for r in ("cheap", "reasoning", "chat", "review")},
        "review_enabled": review_enabled(),
        "council": council_enabled(),
        "overrides": _overrides,
        "source": "ui" if _overrides.get("chains") else "env",
        "defaults": {f"{p}/{r}": m for (p, r), m in DEFAULT_MODELS.items()},
    }
