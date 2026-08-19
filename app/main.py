"""FastAPI surface. Run: uvicorn app.main:app --reload"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel

from .chat import PRESETS, chat_stream
from .features import atr, build_snapshot, ema, structure
from .indicators import catalog as indicator_catalog, compute as compute_indicators
from .calendar import all_events, next_major, refresh as refresh_calendar
from .config import FINNHUB_KEY
from .events import (KNOWN as EVENT_KINDS, capture, coverage as event_coverage,
                     history as event_history, summarise)
from .market import (DataUnavailable, clear_routes, get_candles,
                     routes as data_routes, session_state, set_route)
from .news import (feed, ingest, ingest_finnhub, load_calendar,
                   relevant_news, sentiment, upcoming_events)
from .signals import generate_signal
from .webhook import router as webhook_router

app = FastAPI(title="Terminal AI")

# --- access control ---------------------------------------------------------
# Unset locally, so nothing changes on your own machine. Set it the moment the
# backend is public: without it, anyone who finds the URL spends your API
# credits on every chat message and signal.
ACCESS_KEY = os.getenv("ACCESS_KEY", "")

# Endpoints that cost nothing and reveal nothing stay open, so the frontend can
# tell "wrong key" apart from "server down".
_OPEN_PATHS = {"/health", "/", "/docs", "/openapi.json", "/redoc",
               "/docs/oauth2-redirect"}


@app.middleware("http")
async def require_access_key(request: Request, call_next):
    if not ACCESS_KEY or request.method == "OPTIONS" \
            or request.url.path in _OPEN_PATHS:
        return await call_next(request)

    supplied = (request.headers.get("x-access-key")
                or request.query_params.get("key", ""))
    if supplied != ACCESS_KEY:
        return JSONResponse(
            {"detail": "Missing or invalid access key. Append ?key=... to the "
                       "frontend URL, or send an X-Access-Key header."},
            status_code=401)
    return await call_next(request)
# ALLOWED_ORIGINS lets a hosted frontend (GitHub Pages, a CDN) talk to a
# backend running elsewhere. "*" is fine while the backend is local-only and
# holds no user data, but set it explicitly the moment it is public.
_origins = os.getenv("ALLOWED_ORIGINS", "*")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if _origins.strip() == "*"
                  else [o.strip() for o in _origins.split(",") if o.strip()],
    allow_methods=["*"],
    allow_headers=["*"],
)


app.include_router(webhook_router)

_STATIC = Path(__file__).resolve().parent.parent / "static"


@app.get("/")
def terminal():
    """Serve the chart UI at https://api.realflylink.com/"""
    index = _STATIC / "index.html"
    if not index.exists():
        raise HTTPException(404, "static/index.html not found")
    return FileResponse(index)


@app.get("/smc/{symbol}")
def smc_analysis(symbol: str, timeframe: str = "15m", swing: int = 10,
                 ktr_step: float = 0.40, cisd_run: int = 2):
    """Full SMC/ICT read: structure, order blocks, FVG, liquidity, PD array,
    KTR levels, CISD and CRT. Computed from candles - no model involved."""
    from .smc import analyse
    try:
        df, source = get_candles(symbol, timeframe)
    except DataUnavailable as e:
        raise HTTPException(503, str(e))
    out = analyse(df, swing=swing, ktr_step=ktr_step, cisd_run=cisd_run,
                  timeframe=timeframe)
    out["symbol"] = symbol.upper()
    out["timeframe"] = timeframe
    out["data_source"] = source
    return out


@app.get("/dashboard/{symbol}")
def dashboard(symbol: str, timeframe: str = "15m", btc_symbol: str = "BTCUSDT"):
    """Everything the on-chart dashboard table shows, in one call.

    Mirrors the Pine dashboard: KTR state, EMA bias, BOS trend, CHoCH, order
    flow, PD array, liquidity levels, OB/FVG counts, CISD, cascade stages, and
    the BTC correlation panel.
    """
    from .cascade import evaluate
    from .features import ema as _ema
    from .ktr_signals import boys_bullishness, signals as ktr_sig
    from .smc import analyse as smc_analyse

    try:
        df, source = get_candles(symbol, timeframe)
    except DataUnavailable as e:
        raise HTTPException(503, str(e))

    sm = smc_analyse(df, timeframe=timeframe)
    k = ktr_sig(df)
    boys = boys_bullishness(df)["current"]
    price = float(df["close"].iloc[-1])

    ef = float(_ema(df["close"], 20).iloc[-1])
    es = float(_ema(df["close"], 50).iloc[-1])
    ema_bias = ("long" if ef > es and price > ef else
                "short" if ef < es and price < ef else "wait")

    ktr_state = ("BUY entry" if k["entry_on_last_bar"] and k["latest_entry"]["dir"] == "buy"
                 else "SELL entry" if k["entry_on_last_bar"]
                 else "Buy alert" if k["alert_on_last_bar"] and k["latest_alert"]["dir"] == "buy"
                 else "Sell alert" if k["alert_on_last_bar"]
                 else f"{k['trend']} trend")

    obs, gaps = sm["order_blocks"], sm["fvg"]
    casc = evaluate(symbol, timeframe, "4h", "1h")

    # BTC correlation - gold and crypto share a risk appetite driver.
    btc = None
    try:
        bdf, _ = get_candles(btc_symbol, timeframe)
        b_close = float(bdf["close"].iloc[-1])
        b_open = float(bdf["open"].iloc[-1])
        b_prev = float(bdf["close"].resample("1D").last().dropna().iloc[-2])
        b20 = float(_ema(bdf["close"], 20).iloc[-1])
        b50 = float(_ema(bdf["close"], 50).iloc[-1])
        bull3 = bool((bdf["close"].tail(3).values > bdf["open"].tail(3).values).all())
        bear3 = bool((bdf["close"].tail(3).values < bdf["open"].tail(3).values).all())
        b_bull = b_close >= b_open
        aligned = (b_bull and price >= float(df["open"].iloc[-1])) or \
                  (not b_bull and price < float(df["open"].iloc[-1]))
        up, dn = b20 > b50, b20 < b50
        btc = {
            "symbol": btc_symbol, "price": round(b_close, 2),
            "day_change_pct": round((b_close - b_prev) / b_prev * 100, 2),
            "candle": "bull" if b_bull else "bear",
            "trend": "up" if up else "down" if dn else "flat",
            "momentum": "strong bull" if bull3 else "strong bear" if bear3 else "mixed",
            "alignment": "same direction" if aligned else "diverging",
            "hint": ("BTC bullish - favours risk-on" if up and bull3 else
                     "BTC bearish - favours risk-off" if dn and bear3 else
                     "aligned" if aligned else "BTC and this symbol disagree - wait"),
        }
    except Exception as e:
        log_ = __import__("logging").getLogger(__name__)
        log_.debug("btc panel unavailable: %s", e)

    return {
        "symbol": symbol.upper(), "timeframe": timeframe,
        "data_source": source, "price": round(price, 5),
        "rows": {
            "KTR": ktr_state,
            "EMA bias": ema_bias,
            "BOS trend": sm["structure"]["trend"],
            "CHoCH": (sm["structure"]["last_event"]["dir"] + " reversal"
                      if sm["structure"]["last_event"]
                      and sm["structure"]["last_event"]["type"].startswith("CHoCH") else "-"),
            "Order flow": f"{sm['order_flow']['bias']} {abs(sm['order_flow']['delta_pct'])}%",
            "PD array": f"{sm['pd_array']['zone']} ({sm['pd_array']['position_pct']}%)",
            "BSL": sm["liquidity"]["bsl"],
            "SSL": sm["liquidity"]["ssl"],
            "Order blocks": f"{len(obs['bullish'])} bull / {len(obs['bearish'])} bear",
            "FVG": f"{gaps['counts']['bullish_open']} bull / {gaps['counts']['bearish_open']} bear open",
            "CISD": sm["cisd"]["signal"] or "-",
            "CRT": sm["crt"]["signal"] or "-",
            "Boys": f"{boys['label']} · rel vol {boys['relative_volume']}%",
        },
        "cascade": {
            "bias": f"{casc['bias']['timeframe']} {casc.get('direction') or 'flat'}",
            "confirm": casc["confirm"]["structure"] if casc["stage"] != "no_data" else "-",
            "entry": casc.get("status", "-"),
            "ready": casc.get("ready", False),
            "direction": casc.get("direction"),
        },
        "btc": btc,
    }


@app.get("/cascade/{symbol}")
def cascade_read(symbol: str, timeframe: str = "15m", bias_tf: str = "4h",
                 confirm_tf: str = "1h", sl_points: float | None = None,
                 tp_points: float | None = None):
    """Top-down alignment: bias timeframe, confirm timeframe, entry trigger."""
    from .cascade import evaluate
    return evaluate(symbol, timeframe, bias_tf, confirm_tf,
                    sl_points=sl_points, tp_points=tp_points)


@app.get("/ktr/{symbol}")
def ktr_read(symbol: str, timeframe: str = "15m"):
    """KTR alert diamonds and confirmed entry signals."""
    from .ktr_signals import boys_bullishness, signals
    try:
        df, source = get_candles(symbol, timeframe)
    except DataUnavailable as e:
        raise HTTPException(503, str(e))
    out = signals(df)
    out["boys"] = boys_bullishness(df)["current"]
    out.update(symbol=symbol.upper(), timeframe=timeframe, data_source=source)
    return out


@app.get("/overlay/{symbol}")
def chart_overlay(symbol: str, timeframe: str = "15m"):
    """Zones and event markers for canvas rendering.

    Lightweight Charts has no box or label primitive, so the frontend paints
    these onto a canvas layered over the chart, converting price and time to
    pixels through the chart's own coordinate functions.
    """
    from .pa_toolkit import breaker_blocks, liquidity_grabs, structure_plus, supply_demand
    from .smc import cisd, crt, fvg, order_blocks, structure as smc_structure

    try:
        df, source = get_candles(symbol, timeframe)
    except DataUnavailable as e:
        raise HTTPException(503, str(e))

    zones, markers = [], []

    def add_zone(top, bottom, since, color, label, dashed=False):
        zones.append({"top": top, "bottom": bottom,
                      "since": int(pd.Timestamp(since).timestamp()),
                      "color": color, "label": label, "dashed": dashed})

    st = smc_structure(df)
    obs = order_blocks(df, st)
    for b in obs["bullish"]:
        add_zone(b["top"], b["bottom"], b["at"], "38,166,154",
                 f"OB demand · {b['volume_pct']:.0f}%")
    for b in obs["bearish"]:
        add_zone(b["top"], b["bottom"], b["at"], "239,83,80",
                 f"OB supply · {b['volume_pct']:.0f}%")

    g = fvg(df)
    for z in g["bullish_open"][-3:]:
        add_zone(z["top"], z["bottom"], z["at"], "38,166,154", "FVG", True)
    for z in g["bearish_open"][-3:]:
        add_zone(z["top"], z["bottom"], z["at"], "239,83,80", "FVG", True)

    sd = supply_demand(df)
    for z in sd["demand"]:
        vp = f" · {z['volume_pct']}%" if z.get("volume_pct") else ""
        add_zone(z["top"], z["bottom"], z["since"], "33,150,243",
                 f"{z['volume_label']}{vp}\nDemand · {z['pattern']}")
    for z in sd["supply"]:
        vp = f" · {z['volume_pct']}%" if z.get("volume_pct") else ""
        add_zone(z["top"], z["bottom"], z["since"], "255,152,0",
                 f"{z['volume_label']}{vp}\nSupply · {z['pattern']}")

    for b in breaker_blocks(df)["blocks"][-2:]:
        add_zone(b["top"], b["bottom"], b["since"], "156,39,176", "Breaker")

    for ev in structure_plus(df)["events"]:
        markers.append({
            "time": int(pd.Timestamp(ev["at"]).timestamp()),
            "position": "belowBar" if ev["dir"] == "bullish" else "aboveBar",
            "color": "#26a69a" if ev["dir"] == "bullish" else "#ef5350",
            "shape": "arrowUp" if ev["dir"] == "bullish" else "arrowDown",
            "text": ev["type"],
        })

    for lg in liquidity_grabs(df)["grabs"]:
        markers.append({
            "time": int(pd.Timestamp(lg["at"]).timestamp()),
            "position": "belowBar" if lg["dir"] == "bullish" else "aboveBar",
            "color": "#f0b84a", "shape": "circle", "text": "LG",
        })

    from .ktr_signals import signals as ktr_sig
    ks = ktr_sig(df)
    for al in ks["alerts"]:
        markers.append({
            "time": int(pd.Timestamp(al["at"]).timestamp()),
            "position": "belowBar" if al["dir"] == "buy" else "aboveBar",
            "color": "#00e5ff" if al["dir"] == "buy" else "#ff5252",
            "shape": "circle", "text": "",
        })
    for en in ks["entries"]:
        markers.append({
            "time": int(pd.Timestamp(en["at"]).timestamp()),
            "position": "belowBar" if en["dir"] == "buy" else "aboveBar",
            "color": "#ffc878", "shape": "square",
            "text": "BUY" if en["dir"] == "buy" else "SELL",
        })

    c, r = cisd(df), crt(df)
    last_t = int(df.index[-1].timestamp())
    if c["signal"]:
        markers.append({"time": last_t,
                        "position": "belowBar" if c["signal"] == "bullish" else "aboveBar",
                        "color": "#4a9eff", "shape": "square", "text": "CISD"})
    if r["signal"]:
        markers.append({"time": last_t,
                        "position": "belowBar" if r["signal"] == "bullish" else "aboveBar",
                        "color": "#a371f7", "shape": "square", "text": "CRT"})

    markers.sort(key=lambda m: m["time"])
    return {"symbol": symbol.upper(), "timeframe": timeframe,
            "data_source": source, "zones": zones, "markers": markers}


@app.get("/pa/{symbol}")
def pa_toolkit(symbol: str, timeframe: str = "15m", swing: int = 10):
    """Price Action Toolkit: supply/demand with RBR-DBR-RBD-DBD patterns,
    breaker blocks, liquidity grabs, volume imbalances, CHoCH+."""
    from .pa_toolkit import analyse as pa_analyse
    try:
        df, source = get_candles(symbol, timeframe)
    except DataUnavailable as e:
        raise HTTPException(503, str(e))
    out = pa_analyse(df, swing=swing)
    out.update(symbol=symbol.upper(), timeframe=timeframe, data_source=source)
    return out


@app.get("/timeframes")
def timeframes():
    """Every timeframe the app can serve, and which are pinned as quick pills."""
    from .market import QUICK_TF, TF_MINUTES
    return {
        "all": [{"id": tf, "minutes": m} for tf, m in
                sorted(TF_MINUTES.items(), key=lambda kv: kv[1])],
        "quick": QUICK_TF,
    }


@app.get("/indicators")
def indicators_list():
    """Everything available, and which are on by default."""
    return {"indicators": indicator_catalog()}


@app.get("/price/{symbol}")
def price_tick(symbol: str, timeframe: str = "5m"):
    """Live price — the current tradeable quote, not the last candle.

    Tries the provider's dedicated pricing endpoint first (OANDA /pricing,
    MT5 symbol_info_tick). These update continuously between candle closes,
    which is what makes the display genuinely real-time. Falls back to the
    last candle close when no tick endpoint is available.
    """
    from .market import PROVIDERS, _configured, chain_for

    # Try a real tick feed first
    for name in chain_for(symbol):
        if not _configured(name):
            continue
        prov = PROVIDERS.get(name)

        if name == "oanda" and hasattr(prov, "live_price"):
            tick = prov.live_price(symbol)
            if tick:
                return {
                    "symbol": symbol.upper(),
                    "data_source": "oanda:tick",
                    "price": tick["mid"],
                    "bid": tick["bid"],
                    "ask": tick["ask"],
                    "spread": tick["spread"],
                    "time": tick["time"],
                    "tradeable": tick["tradeable"],
                    "live": True,
                }

        if name == "mt5":
            try:
                from .mt5_provider import live_tick
                t = live_tick(symbol)
                if t:
                    return {
                        "symbol": symbol.upper(),
                        "data_source": "mt5:tick",
                        "price": round((t["bid"] + t["ask"]) / 2, 5),
                        "bid": t["bid"], "ask": t["ask"],
                        "spread": t["spread"], "time": t["at"],
                        "live": True,
                    }
            except Exception:
                pass
        break   # only try the primary provider for ticks

    # Fallback: last candle close
    try:
        df, source = get_candles(symbol, timeframe, 2)
    except DataUnavailable as e:
        raise HTTPException(503, str(e))

    last = df.iloc[-1]
    prev = df.iloc[-2] if len(df) > 1 else last
    close = float(last["close"])
    prev_close = float(prev["close"])

    return {
        "symbol": symbol.upper(),
        "timeframe": timeframe,
        "data_source": source,
        "price": round(close, 5),
        "change": round(close - prev_close, 5),
        "change_pct": round((close - prev_close) / prev_close * 100, 3),
        "time": last.name.isoformat(),
        "live": False,
    }


@app.get("/candles/{symbol}")
def candles(symbol: str, timeframe: str = "15m", limit: int = 300,
            indicators: str = "ema", candle_mode: str = "off"):
    """Full OHLCV for charting, plus the levels to draw on top.

    Times are unix seconds, which is what Lightweight Charts expects.
    """
    try:
        df, source = get_candles(symbol, timeframe, limit)
    except DataUnavailable as e:
        raise HTTPException(503, str(e))
    except ValueError as e:
        raise HTTPException(400, str(e))

    close = df["close"]
    bars, ema20, ema50, ema200 = [], [], [], []
    e20, e50, e200 = (ema(close, n) for n in (20, 50, 200))

    # Candle colouring. "ktr" paints by Supertrend direction (yellow / red),
    # which is the visual signature of the KTR method - you read trend from the
    # candles themselves rather than hunting for a line.
    colors = None
    if candle_mode == "ktr":
        from .smc import supertrend
        d = supertrend(df).values
        colors = ["#ffd93b" if v > 0 else "#ef5350" for v in d]
    elif candle_mode == "orderflow":
        rng = (df["high"] - df["low"]).clip(lower=1e-9)
        bv = np.where(df["close"] >= df["open"], df["volume"],
                      df["volume"] * (df["close"] - df["low"]) / rng)
        sv = np.where(df["close"] < df["open"], df["volume"],
                      df["volume"] * (df["high"] - df["close"]) / rng)
        delta = pd.Series(bv - sv).rolling(10).sum().fillna(0).values
        colors = ["#26a69a" if v > 0 else "#ef5350" for v in delta]

    for i, (ts, row) in enumerate(df.iterrows()):
        t = int(ts.timestamp())
        bar = {
            "time": t,
            "open": round(float(row.open), 5),
            "high": round(float(row.high), 5),
            "low": round(float(row.low), 5),
            "close": round(float(row.close), 5),
        }
        if colors:
            bar["color"] = bar["borderColor"] = bar["wickColor"] = colors[i]
        bars.append(bar)
        if i >= 20:
            ema20.append({"time": t, "value": round(float(e20.iloc[i]), 5)})
        if i >= 50:
            ema50.append({"time": t, "value": round(float(e50.iloc[i]), 5)})
        if i >= 200:
            ema200.append({"time": t, "value": round(float(e200.iloc[i]), 5)})

    a = float(atr(df).iloc[-1])
    levels = structure(df, a)
    wanted = [x.strip() for x in indicators.split(",") if x.strip()]

    # For unlimited providers (OANDA, MT5), start a background refresh so
    # the NEXT request gets fresh data without waiting for a network round-trip.
    # This is what makes 1-second refresh feel genuinely real-time.
    from .cache import UNMETERED
    if source.split(":")[0] in UNMETERED:
        import threading
        def _prefetch():
            try:
                from .cache import _cache, _lock
                import datetime as _dt
                with _lock:
                    entry = _cache.get((symbol.upper(), timeframe))
                    if entry:
                        stored, _df, _src = entry
                        age = (_dt.datetime.now(_dt.timezone.utc) - stored).total_seconds()
                        if age < 1:  # only if we just refreshed
                            return
                get_candles(symbol, timeframe, limit)
            except Exception:
                pass
        threading.Thread(target=_prefetch, daemon=True).start()

    return {
        "symbol": symbol.upper(),
        "timeframe": timeframe,
        "data_source": source,
        "session": session_state(df, timeframe),
        "bars": bars,
        "ema20": ema20,
        "ema50": ema50,
        "ema200": ema200,
        "support": levels["support"],
        "resistance": levels["resistance"],
        "structure_trend": levels["structure_trend"],
        "next_event": next_major(_ccy(symbol)),
        "indicators": compute_indicators(df, wanted),
    }


@app.on_event("startup")
def _boot():
    """Warm the calendar so the first chart load already has the badge."""
    try:
        refresh_calendar()
        capture(all_events())   # learn real release times for the history engine
    except Exception as e:
        import logging
        logging.getLogger(__name__).warning("calendar warmup failed: %s", e)


@app.get("/events/{symbol}")
def events_history(symbol: str, event: str = "fomc", count: int = 8,
                   timeframe: str = "1h"):
    """How this symbol moved after the last N occurrences of a scheduled event.

    Measured from candles - no model involved.
    """
    if event.lower() not in EVENT_KINDS:
        raise HTTPException(400, f"unknown event. options: {list(EVENT_KINDS)}")
    h = event_history(symbol, event, count, timeframe)
    if "error" not in h:
        h["plain_english"] = summarise(h)
    return h


@app.get("/events")
def events_coverage():
    """What event-date data we hold and whether the seeded list is stale."""
    return event_coverage()


@app.get("/data-providers")
def data_providers_get():
    """Which data provider serves each symbol, in fallback order."""
    return data_routes()


class RoutePayload(BaseModel):
    symbol: str
    providers: list[str]


@app.post("/data-providers")
def data_providers_set(payload: RoutePayload):
    """Reorder or restrict a symbol's provider chain. Persists across restarts.

    Example: {"symbol": "XAUUSD", "providers": ["mt5", "twelvedata"]}
    Empty list clears the override.
    """
    try:
        return set_route(payload.symbol, payload.providers)
    except ValueError as e:
        raise HTTPException(400, str(e))


@app.delete("/data-providers")
def data_providers_reset():
    return clear_routes()


@app.get("/mt5")
def mt5_status():
    """Is the local MetaTrader 5 terminal connected, and what does this broker
    call each instrument?"""
    from .mt5_provider import status
    return status()


@app.post("/mt5/resync-clock")
def mt5_resync():
    """Re-measure the broker's clock offset. Run after a DST change."""
    from .mt5_provider import server_offset
    from .cache import clear
    off = server_offset(force=True)
    clear()   # cached candles were stamped with the old offset
    return {"server_offset_seconds": off,
            "server_timezone": f"UTC{off // 3600:+d}:{abs(off % 3600) // 60:02d}"}


@app.get("/mt5/tick/{symbol}")
def mt5_tick(symbol: str):
    """Live bid/ask straight from the terminal. No candle fetch, no cost."""
    from .mt5_provider import live_tick
    t = live_tick(symbol)
    if not t:
        raise HTTPException(503, "MT5 not connected or symbol not found")
    return t


@app.get("/quota")
def quota():
    """Upstream requests used today, per provider, against free-tier limits."""
    from .cache import projected_daily, usage
    u = usage()
    u["projected_daily"] = {tf: projected_daily(tf) for tf in ("1m", "5m", "15m", "1h")}
    return u


class TTLPayload(BaseModel):
    seconds: int | None = None
    budget_mode: bool | None = None


@app.post("/quota/ttl")
def set_cache_ttl(payload: TTLPayload):
    """How long the server holds candles before refetching.

    Lower means fresher prices and more upstream requests. Free tiers are
    capped per day, so this is a direct trade against your quota.
    """
    from .cache import projected_daily, set_budget_mode, set_ttl, usage
    if payload.seconds is not None and not 1 <= payload.seconds <= 900:
        raise HTTPException(400, "seconds must be between 1 and 900, or null to reset")
    if payload.budget_mode is not None:
        set_budget_mode(payload.budget_mode)
    if payload.seconds is not None or payload.budget_mode is False:
        set_ttl(payload.seconds)
    u = usage()
    u["projected_daily"] = {tf: projected_daily(tf) for tf in ("1m", "5m", "15m", "1h")}
    return u


@app.post("/quota/clear-cache")
def clear_cache():
    from .cache import clear
    clear()
    return {"cleared": True}


@app.get("/health")
def health():
    """Reports which providers are usable and how each role is routed."""
    from .llm import status
    st = status()
    st["ok"] = any(v == "ready" for v in st["providers"].values())
    st["auth_required"] = bool(ACCESS_KEY)
    return st


@app.get("/providers")
def providers_get():
    """Current routing: which provider serves each role, and what is available."""
    from .llm import status
    return status()


class RoutingPayload(BaseModel):
    chains: dict[str, list[str]] | None = None
    council: bool | None = None
    review: bool | None = None
    models: dict[str, str] | None = None


@app.post("/providers")
def providers_set(payload: RoutingPayload):
    """Change routing at runtime. Persists across restarts.

    Example: {"chains": {"reasoning": ["anthropic", "gemini"]}, "council": false}
    Pass an empty list for a role to fall back to .env.
    """
    from .llm import set_routing
    try:
        return set_routing(chains=payload.chains, council=payload.council,
                           review=payload.review, models=payload.models)
    except (ValueError, Exception) as e:
        raise HTTPException(400, str(e))


@app.delete("/providers")
def providers_reset():
    """Drop UI overrides and go back to .env."""
    from .llm import clear_routing
    return clear_routing()


@app.get("/snapshot/{symbol}")
def snapshot(symbol: str, timeframe: str = "15m"):
    try:
        return build_snapshot(symbol, timeframe, news=relevant_news(symbol))
    except ValueError as e:
        raise HTTPException(400, str(e))


@app.post("/signal/{symbol}")
def signal(symbol: str, timeframe: str = "15m"):
    return generate_signal(symbol, timeframe)


@app.get("/news")
def news_all(limit: int = 25, min_impact: str = "low"):
    """Full tagged feed for the news panel."""
    return {"items": feed(limit=limit, min_impact=min_impact)}


@app.get("/news/poll")
def news_poll(since: str | None = None, symbol: str = "XAUUSD"):
    """What changed since the client last checked.

    Returns only NEW headlines so the client can diff rather than re-render
    everything. Also triggers a background ingest if enough time has passed.
    """
    import threading, datetime as dt, time as _time
    from .news import _STORE, feed_status, ingest, looks_relevant

    now = dt.datetime.now(dt.timezone.utc)

    # Kick off a background ingest every 3 minutes so news arrives automatically
    # without the user pressing anything.
    global _last_bg_ingest
    if not hasattr(app.state, "last_bg_ingest"):
        app.state.last_bg_ingest = 0.0
    if _time.time() - app.state.last_bg_ingest > 180:
        app.state.last_bg_ingest = _time.time()
        threading.Thread(target=ingest, daemon=True).start()

    cutoff = None
    if since:
        try:
            cutoff = dt.datetime.fromisoformat(since.replace("Z", "+00:00"))
        except ValueError:
            pass

    new_items = []
    for item in _STORE.values():
        if not item or not item.get("relevant", True):
            continue
        ts = item.get("ingested_at") or item.get("published_at", "")
        try:
            item_t = dt.datetime.fromisoformat(ts.replace("Z", "+00:00"))
        except (ValueError, AttributeError):
            continue
        if cutoff and item_t <= cutoff:
            continue
        new_items.append(item)

    new_items.sort(key=lambda x: x.get("ingested_at",""), reverse=True)
    high = [i for i in new_items if i.get("impact") == "high"]

    return {
        "timestamp": now.isoformat(),
        "new_count": len(new_items),
        "high_impact": len(high),
        "items": new_items[:20],
        "alert": high[0]["title"] if high else None,
    }


@app.get("/news/live")
def news_live(symbol: str = "XAUUSD", limit: int = 30, ai_tag: int = 40):
    """Fetch, score and AI-tag headlines in one request, storing nothing.

    Keyword rules score everything instantly; up to `ai_tag` items then get a
    model pass, in batches, for a real directional read and a concrete
    takeaway. Set ai_tag=0 to skip the model entirely.
    """
    from .news import fetch_live
    return fetch_live(symbol, limit, ai_tag=ai_tag)


@app.get("/news/feeds")
def news_feeds():
    """Per-feed reachability. Run this when the news panel comes back empty.

    Declared before /news/{symbol} - FastAPI matches routes in order, and the
    parameterised route would otherwise treat "feeds" as a symbol.
    """
    from .news import feed_status
    return {"feeds": feed_status()}


@app.get("/news/{symbol}")
def news_symbol(symbol: str, limit: int = 20):
    """Feed filtered to this symbol's asset class, plus an aggregate read."""
    return {
        "items": feed(symbol, limit=limit),
        "sentiment": sentiment(symbol),
        "events": upcoming_events(symbol),
    }


@app.delete("/news")
def news_clear():
    """Purge cached headlines. Use after changing the feed list."""
    from .news import clear_store
    return clear_store()


@app.get("/news/latest")
def news_latest(symbol: str = "XAUUSD", since: str | None = None):
    """Poll for headlines newer than `since` (ISO timestamp).

    The frontend polls this every 60s. Only fresh articles are returned, so the
    response is tiny when nothing is new and the page updates automatically.
    """
    from .news import feed, relevant_news
    import datetime as dt
    items = feed(symbol, limit=60)
    if since:
        try:
            cutoff = dt.datetime.fromisoformat(since)
            items = [i for i in items
                     if dt.datetime.fromisoformat(i["published_at"]) > cutoff]
        except (ValueError, KeyError):
            pass
    return {"items": items, "count": len(items)}


@app.post("/news/tag")
def news_tag(limit: int = 8):
    """Score headlines that were stored but not yet tagged."""
    from .news import tag_pending
    return tag_pending(limit)


@app.post("/news/ingest")
def news_ingest(max_tag: int = 24):
    """Pull every configured source, then score what is worth scoring.

    APIs run first when configured - they are faster and better structured than
    scraped RSS, and Marketaux even arrives pre-scored.
    """
    from .news import MARKETAUX_KEY, ingest_marketaux

    result = {}
    if MARKETAUX_KEY:
        result["marketaux"] = ingest_marketaux()
    # Finnhub is now called inside ingest() as the primary source.
    # Passing finnhub_first=True means it runs even before RSS feeds.
    result.update(ingest(max_tag=max_tag, finnhub_first=True))
    return result


class CalendarPayload(BaseModel):
    events: list[dict]


@app.get("/calendar")
def calendar_get(symbol: str | None = None, hours: int = 336):
    """Upcoming economic events. Free ForexFactory data, no key needed."""
    refresh_calendar()
    if symbol:
        return {"events": upcoming_events(symbol, hours),
                "next_major": next_major(_ccy(symbol), hours)}
    return {"events": all_events()}


@app.post("/calendar/refresh")
def calendar_refresh():
    return refresh_calendar(force=True)


@app.post("/calendar")
def calendar_load(payload: CalendarPayload):
    """Inject events manually, for testing."""
    load_calendar(payload.events)
    return {"loaded": len(payload.events)}


def _ccy(symbol: str) -> set[str]:
    s = symbol.upper()
    if s in ("XAUUSD", "XAGUSD", "BTCUSDT", "ETHUSDT", "USOIL"):
        return {"USD"}
    return {s[:3], s[3:6]} if len(s) >= 6 else {"USD"}


class ChatPayload(BaseModel):
    messages: list[dict]


@app.post("/chat")
def chat(payload: ChatPayload):
    def sse():
        for event in chat_stream(payload.messages):
            yield f"data: {json.dumps(event, default=str)}\n\n"

    return StreamingResponse(
        sse(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/chat/preset/{name}")
def chat_preset(name: str, symbol: str, timeframe: str = "15m"):
    if name not in PRESETS:
        raise HTTPException(404, f"unknown preset. options: {list(PRESETS)}")
    text = PRESETS[name].format(symbol=symbol.upper(), timeframe=timeframe)

    def sse():
        for event in chat_stream([{"role": "user", "content": text}]):
            yield f"data: {json.dumps(event, default=str)}\n\n"

    return StreamingResponse(sse(), media_type="text/event-stream")
