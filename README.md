# Terminal AI

Backend for an AI trading terminal: signal generation, news analysis, and a chat
assistant with tool use.

The design rule throughout: **the model never does arithmetic.** Your code
computes every indicator and level; the model reads a compact summary and
decides. Then your code checks its answer before anything reaches a user.

---

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env          # add your ANTHROPIC_API_KEY
python test_offline.py        # runs with no key and no network
uvicorn app.main:app --reload
```

Open https://api.realflylink.com/docs for the interactive API.

Without a market data provider configured, the app serves **synthetic** candles
so you can develop offline. Signals are blocked on synthetic data — the gate
lives in `signals.py`. To exercise the model path anyway:
`DEV_ALLOW_SYNTHETIC_SIGNALS=1`. In production set `ALLOW_SYNTHETIC=0` so a
provider outage raises instead of silently serving fake prices.

---

## Endpoints

| Method | Path | Purpose |
|---|---|---|
| GET | `/snapshot/{symbol}?timeframe=15m` | The computed market state (~350 tokens) |
| POST | `/signal/{symbol}?timeframe=15m` | Validated trade idea, or `no_trade` with a reason |
| GET | `/news/{symbol}` | Tagged news + upcoming calendar events |
| POST | `/news/ingest` | Pull the feed and tag new articles |
| POST | `/calendar` | Load economic events |
| POST | `/chat` | Streaming chat (SSE) with tool use |
| POST | `/chat/preset/{name}?symbol=XAUUSD` | The terminal buttons: `analyze`, `best_setup`, `outlook`, `risk_check` |

---

## The four layers

**1. Feature engine** (`features.py`) — pandas only, no model. EMA/RSI/MACD/ADX/ATR,
Bollinger width, fractal swing points clustered into support and resistance with
touch counts. Output is one JSON object of about 350 tokens.

**2. News pipeline** (`news.py`) — two stages, because volume is the cost driver.
A cheap model tags every incoming article once (assets, sentiment −1..1, impact
1–3, 25-word summary), deduplicated by normalised-headline fingerprint. Only the
top few relevant items are injected into the reasoning context. The economic
calendar is structured data and never touches a model.

**3. Signal generation** (`signals.py`) — structured outputs constrain the model
to the schema, so malformed JSON is impossible. Then the validator runs.

**4. Chat** (`chat.py`) — one streaming endpoint, four tools
(`get_snapshot`, `generate_signal`, `search_news`, `get_calendar`). The preset
buttons are just canned first messages.

---

## The validator

This is the part that matters. Models hallucinate price levels fluently and
confidently. Every rule in `validate()` exists because a model broke it:

- stop loss on the correct side of entry
- take profits on the correct side, in order
- entry within 1.5× ATR of live price — catches invented levels
- stop between 0.5× and 4× ATR (not inside noise, not absurdly wide)
- risk:reward ≥ 1.5 to the first target
- confidence floor of 0.45

On failure the errors are fed back for one repair attempt, then it degrades to
`no_trade`. Tune the thresholds in `config.py:RiskRules`.

There are also three gates that run **before** any model call, so a rejected
request costs nothing:

- synthetic/unverified data source
- stale or closed session (the weekend-gold case: volume 0, flat closes)
- high-impact event inside the blackout window

---

## Model routing

| Job | Model | Why |
|---|---|---|
| News tagging, dedupe | `claude-haiku-4-5` | thousands of calls/day |
| Signal reasoning | `claude-sonnet-5` | needs real reasoning |
| Chat | `claude-sonnet-5` | latency matters |

Set via `CHEAP_MODEL` / `REASONING_MODEL` / `CHAT_MODEL`. `llm.with_failover()`
retries on a fallback model when the primary errors — in production that fallback
is often a different provider entirely, so one outage doesn't take the terminal
down. That is the whole reason teams run two providers; there is nothing magic
about the mix.

---

## Cost control

6 assets × 5 timeframes every 15 minutes is roughly 2,900 reasoning calls a day.
Don't do that. Two things keep the bill sane:

1. **Only regenerate on candle close**, and only when a feature moved materially
   (price crossed an EMA, ADX changed regime, a level was broken). A snapshot
   identical to the last one does not need a new opinion.
2. **Cache by snapshot hash.** Same inputs, same answer.

Prompt caching on the system prompt helps too, since it is identical across
every signal call.

---

## Before you launch

Publishing trade signals is a regulated activity in most jurisdictions — it can
count as investment advice or a financial recommendation regardless of what your
disclaimer says. Check the rules where you're incorporated and where your users
are before going live. Common mitigations are educational framing, no execution,
and no personalised recommendations, but they aren't a substitute for actual
advice from a lawyer in the relevant jurisdiction.

Every response from this backend carries a `disclaimer` field. Keep it in the UI.
