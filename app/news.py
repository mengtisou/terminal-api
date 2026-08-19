"""News pipeline.

Two stages, because volume is the cost driver:
  1. Ingest headlines from free RSS feeds (no API key needed).
  2. One cheap-model call per article scores it across every asset class we
     track, with an impact rating and a one-line actionable takeaway.

Deduped by normalised-headline fingerprint, because the same wire story
arrives from five feeds with slightly different wording.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import logging
import os
import re
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor

import httpx

from .config import FINNHUB_KEY, settings
from .llm import LLMError, json_call

log = logging.getLogger(__name__)

# In-memory store. Swap for Postgres in production.
_STORE: dict[str, dict] = {}

# Asset buckets we score every story against.
BUCKETS = ["gold", "usd", "crypto", "oil", "stocks"]

# Which bucket drives which symbol, for the terminal's per-symbol news feed.
SYMBOL_BUCKET = {
    "XAUUSD": "gold", "XAGUSD": "gold",
    "BTCUSDT": "crypto", "ETHUSDT": "crypto",
    "USOIL": "oil",
    "EURUSD": "usd", "GBPUSD": "usd", "USDJPY": "usd",
}

# Free RSS feeds, chosen for macro and metals rather than general business.
# A general finance feed is mostly single-company stories that move nothing on
# a gold chart - the earlier mix produced headlines about copper history and
# stock picks, which is noise for this use.
# News feeds only. Analysis and opinion feeds were producing "7 Bold
# Predictions for 2026" and "Will X Support Y?" - columns, not events. A
# trading desk wants what happened, not what a columnist thinks might.
# RSS feeds are geo-blocked from many Asian networks. Finnhub HTTPS API is
# the reliable primary source; RSS stays as optional supplementary feeds.
# News feeds only - analysis and opinion feeds produce columns, not events.
# Investing.com's per-category feeds are the most reliable free source that is
# actually about macro and metals rather than single-company business news.
FEEDS = [
    ("Investing Forex", "https://www.investing.com/rss/news_1.rss"),
    ("Investing Commodities", "https://www.investing.com/rss/commodities.rss"),
    ("Investing Economy", "https://www.investing.com/rss/news_14.rss"),
    ("Investing Economic Indicators", "https://www.investing.com/rss/news_95.rss"),
    ("Investing Breaking", "https://www.investing.com/rss/news_285.rss"),
    ("CoinDesk", "https://www.coindesk.com/arc/outboundfeeds/rss/"),
]

# Finnhub categories to pull from in order
FINNHUB_CATEGORIES = ["general", "forex", "merger"]

# A slow feed should not hold up the others. Eight seconds is generous for RSS;
# anything slower is broken, not busy.
FEED_TIMEOUT = 8

# Opinion and listicle patterns. These are the shapes columns take, and none of
# them report an event.
# Sources that publish almost exclusively off-topic content for a gold terminal.
# Letting them through wastes quota and adds noise.
OFF_TOPIC_SOURCES = {
    "cnbc", "disney", "abc", "nbc", "cbs", "bbc sport",
    "techcrunch", "wired", "engadget", "the verge",
}

OPINION_PATTERNS = (
    r"^\d+\s+(bold|top|best|worst|key|things|reasons|stocks|ways)",
    r"\b(bold predictions|predictions for|outlook for \d{4}|year ahead)\b",
    r"^(will|is|are|should|can|why|how|what|does|do|did|has|could|would)\b.*\?$",
    r"\?$",                                  # any question headline
    r"\b(opinion|column|commentary|analysis|explainer|explained|guide)\b",
    r"\b(here'?s (why|how|what)|what to (watch|know)|things to watch)\b",
    r"\b(my |i think|i'?m |we think)\b",
    r"\b(top \d+|best \d+|\d+ stocks|stock picks|watchlist)\b",
    r"\b(could|might|may) (be|see|mean|signal)\b",
    r"\b(what it means|takeaways|lessons)\b",
    # Editorial framing: an assertion about significance, not a report.
    r"\b(is|are|remains?) an? (warning|sign|signal|red flag|problem|risk|concern)\b",
    r"\b(what'?s next|means for|matters for|to watch for)\b",
    r"\b(the case for|the case against|time to (buy|sell))\b",
)
_OPINION_RE = [re.compile(p, re.I) for p in OPINION_PATTERNS]


def is_opinion(title: str, source: str = "") -> bool:
    """True for columns, listicles, and clearly off-topic sources."""
    if source and any(s in source.lower() for s in OFF_TOPIC_SOURCES):
        return True
    t = title.strip()
    return any(rx.search(t) for rx in _OPINION_RE)

# Cheap keyword prefilter. Tagging every headline burns quota on stories that
# obviously do not touch gold, oil or the dollar - so screen first, spend after.
RELEVANT_WORDS = {
    "gold", "xau", "bullion", "silver", "metal",
    "dollar", "usd", "dxy", "euro", "yen", "pound", "currency", "forex", "fx",
    "fed", "fomc", "powell", "rate", "rates", "hike", "cut", "inflation", "cpi",
    "ppi", "pce", "payroll", "nfp", "jobs", "unemployment", "gdp", "recession",
    "treasury", "yield", "bond", "central bank", "ecb", "boj", "boe",
    "oil", "crude", "brent", "wti", "opec", "energy",
    "war", "strike", "attack", "sanction", "tariff", "trade deal", "conflict",
    "iran", "israel", "russia", "ukraine", "china", "hormuz", "middle east",
    "bitcoin", "crypto", "risk-off", "risk off", "safe haven", "stocks", "equities",
}


def looks_relevant(title: str, summary: str = "") -> bool:
    if is_opinion(title):
        return False
    text = f"{title} {summary}".lower()
    return any(w in text for w in RELEVANT_WORDS)

_SENT = {"type": "string", "enum": ["bullish", "bearish", "neutral"]}

TAG_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "relevant": {
            "type": "boolean",
            "description": "False for sports, lifestyle, single-company news with no macro read-through, and anything else a macro trader would skip.",
        },
        "impact": {
            "type": "string",
            "enum": ["high", "medium", "low"],
            "description": "high is rare: central bank decisions, war, major surprise data. Most news is low.",
        },
        "assets": {
            "type": "object",
            "additionalProperties": False,
            "properties": {b: _SENT for b in BUCKETS},
            "required": BUCKETS,
            "description": "Direction this story implies for each asset class over the next 24-48h.",
        },
        "summary": {"type": "string", "description": "At most 20 words, plain and factual."},
        "takeaway": {
            "type": "string",
            "description": "One short actionable line, e.g. 'Buy oil and gold on confirmed disruption; fade equity rebounds.' Empty string if there is no clear action.",
        },
    },
    "required": ["relevant", "impact", "assets", "summary", "takeaway"],
}

TAG_SYSTEM = """You tag financial news for a trading terminal.

Be conservative. Most headlines move nothing - mark them low impact and \
neutral across the board rather than inventing a signal. Reserve high impact \
for central bank decisions, war and conflict escalation, and major surprise \
economic data.

Think about the actual transmission mechanism before assigning a direction. \
Risk-off tends to be gold bullish and stocks bearish. A stronger dollar is \
usually gold bearish. Supply disruption in the Gulf is oil bullish. If you \
cannot name the mechanism, the answer is neutral.

The takeaway must be concrete and short. If there is no clear action, return \
an empty string rather than filler."""


# Publishers append their own name to syndicated copy, so the same wire story
# arrives as "Dollar feeble as rate bets dwindle" and
# "Dollar feeble as rate bets dwindle - Reuters". Strip that before hashing.
_SOURCE_SUFFIX = re.compile(
    r"\s*[-–—|]\s*(reuters|bloomberg|ap|afp|cnbc|marketwatch|investing\.?com|"
    r"fxstreet|kitco|coindesk|barron'?s|wsj|ft|forbes|yahoo|the times|"
    r"business insider|seeking ?alpha)\s*$", re.I)

_STOPWORDS = {
    "the", "a", "an", "of", "in", "on", "to", "for", "as", "at", "by", "is",
    "are", "and", "or", "with", "from", "after", "amid", "over", "its", "it",
    "that", "this", "says", "said",
}


def _fingerprint(title: str) -> str:
    """Order-insensitive hash of a headline's meaningful words.

    Sorting the words means a reordered or lightly reworded headline still
    collides, which is what catches syndicated duplicates.
    """
    clean = _SOURCE_SUFFIX.sub("", title.strip())
    norm = re.sub(r"[^a-z0-9 ]", " ", clean.lower())
    words = sorted(set(norm.split()) - _STOPWORDS)
    return hashlib.sha1(" ".join(words[:12]).encode()).hexdigest()[:16]


NEUTRAL = {b: "neutral" for b in BUCKETS}

# Rule-based impact and direction. Not as good as a model, but it costs nothing
# and it means the High/Medium filters are useful even when the AI quota is gone.
HIGH_WORDS = (
    "fomc", "fed decision", "rate decision", "interest rate decision",
    "powell", "fed chair", "the fed", "fed holds", "fed cuts", "fed raises",
    "rate cut", "rate hike", "cuts rates", "raises rates", "holds rates",
    "rates steady", "dovish", "hawkish",
    "non-farm", "nonfarm", "nfp", "payrolls", "unemployment rate",
    "recession", "default",
    "war", "invasion", "missile", "airstrike", "attack", "strike on",
    "sanctions", "embargo", "ceasefire", "nuclear",
    "opec", "production cut", "output cut",
    "tariff", "trade war",
)

# Data releases only move gold when they are US data - the Fed reacts to those.
# Malaysian CPI is real news and almost never a gold event.
US_DATA = ("cpi", "inflation", "pce", "gdp", "retail sales", "ppi",
           "jobless claims", "consumer confidence", "ism")
US_CONTEXT = ("us ", "u.s.", "american", "fed", "dollar", "treasury",
              "washington", "united states")
MEDIUM_WORDS = (
    "retail sales", "ism", "pmi", "jobless claims", "consumer confidence",
    "trade balance", "industrial output", "industrial production",
    "central bank", "ecb", "boj", "boe", "treasury", "yield", "bond",
    "stockpiles", "inventories", "supply", "shipment", "export", "import",
    "election", "summit", "talks", "deal",
)

# Words that push markets risk-off. Gold gains, equities and crypto suffer.
RISK_OFF = ("war", "attack", "missile", "strike", "invasion", "escalat",
            "conflict", "tension", "sanction", "nuclear", "crisis", "threat",
            "recession", "default", "slump", "plunge", "selloff", "sell-off")
RISK_ON = ("ceasefire", "peace", "deal reached", "agreement", "de-escalat",
           "resolve", "recovery", "rebound", "optimis", "rally", "surge")


def rule_score(title: str, summary: str = "") -> dict:
    """Impact and direction from keywords. A floor, not a replacement."""
    text = f"{title} {summary}".lower()

    impact = "low"
    if any(w in text for w in HIGH_WORDS):
        impact = "high"
    elif any(w in text for w in US_DATA):
        # US data is a gold event; the same release elsewhere usually is not.
        impact = "high" if any(c in text for c in US_CONTEXT) else "medium"
    elif any(w in text for w in MEDIUM_WORDS):
        impact = "medium"

    off = sum(w in text for w in RISK_OFF)
    on = sum(w in text for w in RISK_ON)
    assets = dict(NEUTRAL)
    if off > on:
        assets.update(gold="bullish", stocks="bearish", crypto="bearish",
                      oil="bullish", usd="bullish")
    elif on > off:
        assets.update(gold="bearish", stocks="bullish", crypto="bullish",
                      oil="bearish", usd="neutral")

    # A direct read beats the risk proxy when the headline names the asset.
    if "gold" in text or "bullion" in text:
        if any(w in text for w in ("rise", "gain", "climb", "jump", "higher", "record")):
            assets["gold"] = "bullish"
        elif any(w in text for w in ("fall", "drop", "slip", "lower", "decline", "sink")):
            assets["gold"] = "bearish"
    if any(w in text for w in ("rate cut", "cuts rates", "dovish")):
        assets.update(gold="bullish", usd="bearish")
    if any(w in text for w in ("rate hike", "raises rates", "hawkish")):
        assets.update(gold="bearish", usd="bullish")

    return {"impact": impact, "assets": assets,
            "scored_by": "rules" if impact != "low" or off or on else None}


def store_raw(title: str, summary: str = "", source: str = "", url: str = "",
              published: str | None = None) -> dict | None:
    """Save a headline with no AI involved.

    The news panel must work whether or not a model is reachable - a headline
    with no sentiment tag is still worth reading. Tagging upgrades it later.
    """
    fp = _fingerprint(title)
    if fp in _STORE and _STORE[fp]:
        # Keep the shorter, un-suffixed headline if this copy is cleaner.
        existing = _STORE[fp]
        clean = _SOURCE_SUFFIX.sub("", title.strip())
        if len(clean) < len(existing["title"]):
            existing["title"] = clean[:160]
        return existing

    rules = rule_score(title, summary)
    item = {
        "fingerprint": fp,
        "title": title[:160],
        "summary": (summary or "")[:220],
        "source": source,
        "url": url,
        # Missing dates are the exception; flag them rather than silently
        "published_at": published or dt.datetime.now(dt.timezone.utc).isoformat(),
        "date_known": bool(published),
        "ingested_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "tagged": False,
        "scored_by": rules["scored_by"],
        "relevant": True,
        "impact": rules["impact"],
        "assets": rules["assets"],
        "takeaway": "",
    }
    _STORE[fp] = item
    return item


def tag_article(title: str, body: str = "", source: str = "", url: str = "",
                published: str | None = None) -> dict | None:
    """One cheap-model call per article, upgrading a stored headline in place."""
    fp = _fingerprint(title)
    existing = _STORE.get(fp)
    if existing and existing.get("tagged"):
        return existing

    try:
        tags = json_call(
            role="cheap", system=TAG_SYSTEM,
            user=f"Headline: {title}\n\n{body[:1200]}",
            schema=TAG_SCHEMA, max_tokens=500,
        )
    except Exception as e:      # never let one bad article break a batch
        log.warning("could not tag %r: %s", title[:50], e)
        return None

    tags.pop("_provider", None)
    base = existing or store_raw(title, body, source, url, published) or {}
    base.update({
        "tagged": True,
        "scored_by": "ai",
        "relevant": bool(tags.get("relevant")),
        "impact": tags.get("impact", "low"),
        "assets": tags.get("assets", dict(NEUTRAL)),
        "summary": tags.get("summary") or base.get("summary", ""),
        "takeaway": tags.get("takeaway", ""),
    })
    _STORE[fp] = base
    return base if base["relevant"] else None


# --- ingestion ---------------------------------------------------------------

def _parse_rss(xml_text: str) -> list[dict]:
    """Handles both RSS <item> and Atom <entry>."""
    out = []
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        return out

    for item in root.iter():
        tag = item.tag.split("}")[-1]
        if tag not in ("item", "entry"):
            continue

        def field(*names):
            for n in names:
                el = item.find(n)
                if el is None:
                    el = next((c for c in item if c.tag.split("}")[-1] == n), None)
                if el is not None:
                    return (el.text or el.get("href") or "").strip()
            return ""

        title = field("title")
        if title:
            out.append({
                "title": re.sub(r"<[^>]+>", "", title),
                "summary": re.sub(r"<[^>]+>", "", field("description", "summary"))[:600],
                "url": field("link", "id"),
                "published": field("pubDate", "published", "updated"),
            })
    return out


def _fetch_feed(name: str, url: str) -> list[dict]:
    try:
        r = httpx.get(url, timeout=FEED_TIMEOUT, follow_redirects=True,
                      headers={"User-Agent": "Mozilla/5.0 (terminal-ai)"})
        r.raise_for_status()
        items = _parse_rss(r.text)
        for i in items:
            i["source"] = name
        log.info("%s: %d headlines", name, len(items))
        return items
    except Exception as e:
        log.warning("feed %s failed: %s", name, e)
        return []


def clear_store() -> dict:
    """Drop every cached headline.

    Needed when the feed list changes - old articles linger in memory and make
    it look as though nothing improved.
    """
    n = len(_STORE)
    _STORE.clear()
    log.info("cleared %d cached headlines", n)
    return {"cleared": n}


def prune_old(hours: int = 48) -> int:
    """Drop headlines past the lookback window so the panel stays current."""
    cutoff = dt.datetime.now(dt.timezone.utc) - dt.timedelta(hours=hours)
    stale = []
    for fp, item in _STORE.items():
        if not item:
            continue
        ts = item.get("published_at") or item.get("ingested_at")
        try:
            if dt.datetime.fromisoformat(ts) < cutoff:
                stale.append(fp)
        except (TypeError, ValueError):
            continue
    for fp in stale:
        del _STORE[fp]
    return len(stale)


def fetch_live(symbol: str = "XAUUSD", limit: int = 30,
               limit_per_feed: int = 10) -> dict:
    """Fetch and score headlines in one request, storing nothing.

    Serverless platforms give each request a fresh process, so the in-memory
    store is always empty there. This path does the whole job in one call:
    pull the feeds in parallel, drop opinion and off-topic pieces, score with
    the keyword rules (no AI, so no latency and no quota), and return.

    Slower per request than the cached path, but it works anywhere.
    """
    with ThreadPoolExecutor(max_workers=len(FEEDS)) as pool:
        batches = list(pool.map(lambda f: _fetch_feed(*f)[:limit_per_feed], FEEDS))

    bucket = SYMBOL_BUCKET.get(symbol.upper(), "usd")
    seen, items = set(), []

    for batch in batches:
        for raw in batch:
            title = raw.get("title", "")
            if not title:
                continue
            fp = _fingerprint(title)
            if fp in seen:
                continue
            seen.add(fp)

            source = raw.get("source", "")
            if is_opinion(title, source):
                continue
            summary = raw.get("summary", "")
            if not looks_relevant(title, summary):
                continue

            rules = rule_score(title, summary)
            published = _iso(raw.get("published"))
            items.append({
                "fingerprint": fp,
                "title": title[:160],
                "summary": summary[:220],
                "source": source,
                "url": raw.get("url", ""),
                "published_at": published or dt.datetime.now(dt.timezone.utc).isoformat(),
                "date_known": bool(published),
                "tagged": False,
                "scored_by": rules["scored_by"] or "rules",
                "relevant": True,
                "impact": rules["impact"],
                "assets": rules["assets"],
                "takeaway": "",
            })

    rank = {"low": 1, "medium": 2, "high": 3}
    items.sort(key=lambda i: (-rank[i["impact"]], i["published_at"]), reverse=False)
    items.sort(key=lambda i: (rank[i["impact"]], i["published_at"]), reverse=True)
    items = items[:limit]

    # Aggregate sentiment for the symbol, same weighting as the cached path.
    score, high = 0.0, 0
    for i in items:
        w = {"high": 3, "medium": 2, "low": 1}[i["impact"]]
        lean = i["assets"].get(bucket, "neutral")
        score += w * (1 if lean == "bullish" else -1 if lean == "bearish" else 0)
        high += i["impact"] == "high"
    norm = round(score / max(len(items) * 3, 1), 2) if items else 0.0

    return {
        "items": items,
        "sentiment": {
            "symbol": symbol.upper(), "bucket": bucket, "score": norm,
            "label": "bullish" if norm > 0.15 else "bearish" if norm < -0.15 else "neutral",
            "articles": len(items), "high_impact": high,
        },
        "events": upcoming_events(_ccy_for(symbol), hours=48),
        "stateless": True,
    }


def _ccy_for(symbol: str) -> str:
    s = symbol.upper()
    if s.startswith("XAU") or s.startswith("XAG") or "USD" in s:
        return "USD"
    return "USD"


def feed_status() -> list[dict]:
    """Which feeds are reachable from this machine. Pure diagnostics - some
    networks block some publishers, and a silent zero is unhelpful."""
    def probe(name, url):
        t0 = dt.datetime.now()
        try:
            r = httpx.get(url, timeout=FEED_TIMEOUT, follow_redirects=True,
                          headers={"User-Agent": "Mozilla/5.0 (terminal-ai)"})
            items = _parse_rss(r.text) if r.status_code == 200 else []
            return {"name": name, "url": url, "status": r.status_code,
                    "headlines": len(items),
                    "ms": round((dt.datetime.now() - t0).total_seconds() * 1000),
                    "ok": r.status_code == 200 and bool(items)}
        except Exception as e:
            return {"name": name, "url": url, "status": None, "headlines": 0,
                    "ok": False, "error": f"{type(e).__name__}: {e}"}

    with ThreadPoolExecutor(max_workers=len(FEEDS)) as pool:
        return list(pool.map(lambda f: probe(*f), FEEDS))


def ingest(limit_per_feed: int = 12, max_tag: int = 8,
           workers: int = 2, finnhub_first: bool = True) -> dict:
    """Pull every feed, then tag whatever is new.

    Tagging is throttled in app/llm.py to respect per-minute provider caps, so
    workers stays low deliberately - two parallel calls against an 8/min limit
    is sustainable, six is not.
    """
    pruned = prune_old()

    # Finnhub first - it is an HTTPS API, not a scraped feed, so it works
    # regardless of network geo-blocking.
    finnhub_result = {}
    if finnhub_first and FINNHUB_KEY:
        finnhub_result = ingest_finnhub_all()

    # RSS feeds - supplementary, often blocked in some regions
    fresh = []
    if FEEDS:
        with ThreadPoolExecutor(max_workers=max(len(FEEDS), 1)) as pool:
            batches = list(pool.map(lambda f: _fetch_feed(*f)[:limit_per_feed], FEEDS))
        seen = set()
        for batch in batches:
            for item in batch:
                fp = _fingerprint(item["title"])
                if fp in seen or fp in _STORE:
                    continue
                seen.add(fp)
                fresh.append(item)

    # Store everything first, so the panel has content even if the AI is down.
    opinion = 0
    for item in fresh:
        if is_opinion(item["title"], item.get("source", "")):
            opinion += 1
            continue
        store_raw(item["title"], item.get("summary", ""), item.get("source", ""),
                  item.get("url", ""), _iso(item.get("published")))

    # Spend the tagging budget on headlines that could plausibly matter.
    candidates = [i for i in fresh
                  if not is_opinion(i["title"], i.get("source", ""))
                  and looks_relevant(i["title"], i.get("summary", ""))]
    batch_to_tag = candidates[:max_tag]
    tagged, tag_errors = 0, 0
    if batch_to_tag:
        def work(item):
            try:
                return tag_article(item["title"], item.get("summary", ""),
                                   item.get("source", ""), item.get("url", ""),
                                   _iso(item.get("published")))
            except Exception as e:
                log.warning("tagging worker failed: %s", e)
                return None
        with ThreadPoolExecutor(max_workers=workers) as pool:
            results = list(pool.map(work, batch_to_tag))
        tagged = sum(1 for r in results if r)
        tag_errors = sum(1 for r in results if r is None)

    fetched = sum(len(b) for b in batches)
    return {
        "fetched": fetched,
        "prefiltered": len(candidates),
        "opinion_dropped": opinion,
        "pruned_old": pruned,
        "finnhub": finnhub_result,
        "feeds_ok": sum(1 for b in (batches if FEEDS else []) if b),
        "feeds_total": len(FEEDS),
        "new": len(fresh),
        "tagged": tagged,
        "skipped_irrelevant": len(batch_to_tag) - tagged,
        "remaining_untagged": max(0, len(fresh) - len(batch_to_tag)),
        "total_stored": len([v for v in _STORE.values() if v]),
        "tag_failures": tag_errors,
        "hint": _hint(fetched, batch_to_tag, tagged),
    }


def _hint(fetched: int, batch, tagged: int) -> str | None:
    if fetched == 0:
        return "No feeds reachable - check the network or see /news/feeds"
    if batch and not tagged:
        return ("Headlines saved, but AI scoring failed. Most often this is the "
                "Gemini free-tier daily cap (250 requests). Scoring resumes "
                "tomorrow, or add credits to another provider. The headlines "
                "themselves are unaffected.")
    return None


MARKETAUX_KEY = os.getenv("MARKETAUX_KEY", "")


def ingest_marketaux(symbols: str = "XAUUSD,GLD,USO,BTC") -> dict:
    """Marketaux. Free tier: 100 requests/day.

    Worth having because it returns its OWN sentiment score per entity, so
    headlines arrive pre-scored and do not consume LLM quota at all.
    """
    if not MARKETAUX_KEY:
        return {"error": "MARKETAUX_KEY not set", "tagged": 0}
    try:
        r = httpx.get("https://api.marketaux.com/v1/news/all",
                      params={"api_token": MARKETAUX_KEY, "language": "en",
                              "filter_entities": "true", "limit": 25,
                              "search": "gold OR dollar OR oil OR fed OR inflation"},
                      timeout=20)
        r.raise_for_status()
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}", "tagged": 0}

    added = 0
    for a in r.json().get("data", []):
        title = a.get("title", "")
        if not title or is_opinion(title):
            continue
        item = store_raw(title, a.get("description", ""), a.get("source", "Marketaux"),
                         a.get("url", ""), a.get("published_at"))
        if not item:
            continue

        # Fold their entity sentiment straight into our buckets - free scoring.
        scores = [e.get("sentiment_score") for e in a.get("entities", [])
                  if e.get("sentiment_score") is not None]
        if scores:
            avg = sum(scores) / len(scores)
            lean = "bullish" if avg > 0.15 else "bearish" if avg < -0.15 else "neutral"
            # Risk-off lifts gold and hurts equities; risk-on does the reverse.
            item["assets"] = {
                "gold": "bearish" if lean == "bullish" else "bullish" if lean == "bearish" else "neutral",
                "stocks": lean, "crypto": lean,
                "usd": "neutral", "oil": lean,
            }
            item["impact"] = "medium" if abs(avg) > 0.35 else "low"
            item["tagged"] = True
            item["takeaway"] = ""
            item["source"] = (a.get("source") or "Marketaux") + " · scored by feed"
            added += 1
    return {"tagged": added, "source": "marketaux"}


def ingest_finnhub_all() -> dict:
    """Pull all Finnhub categories and store everything."""
    if not FINNHUB_KEY:
        return {"error": "FINNHUB_KEY not set", "stored": 0}
    total = 0
    for cat in FINNHUB_CATEGORIES:
        r = ingest_finnhub(cat)
        total += r.get("stored", 0)
    return {"stored": total, "source": "finnhub"}


def ingest_finnhub(category: str = "general") -> dict:
    """Optional richer feed. Needs FINNHUB_KEY."""
    if not FINNHUB_KEY:
        return {"error": "FINNHUB_KEY not set"}
    try:
        r = httpx.get("https://finnhub.io/api/v1/news",
                      params={"category": category, "token": FINNHUB_KEY}, timeout=20)
        r.raise_for_status()
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}", "stored": 0}

    stored = 0
    for item in r.json()[:60]:
        headline = item.get("headline", "")
        if not headline or is_opinion(headline):
            continue
        ts = dt.datetime.fromtimestamp(item.get("datetime", 0), dt.timezone.utc)
        if store_raw(headline, item.get("summary", ""),
                     item.get("source", "Finnhub"), item.get("url", ""),
                     ts.isoformat()):
            stored += 1
    return {"stored": stored, "source": "finnhub"}


def _iso(raw: str | None) -> str | None:
    """Parse an RSS publication date.

    RSS uses RFC 2822 dates ("Mon, 18 Aug 2026 06:22:00 +0700"), Atom uses
    ISO 8601, and publishers vary within both. The stdlib RFC parser handles
    the messy real-world cases that a strptime list misses - and getting this
    wrong makes every headline read "just now", which is worse than useless
    on a news panel.
    """
    if not raw:
        return None
    raw = raw.strip()

    try:
        from email.utils import parsedate_to_datetime
        d = parsedate_to_datetime(raw)
        if d:
            if d.tzinfo is None:
                d = d.replace(tzinfo=dt.timezone.utc)
            return d.astimezone(dt.timezone.utc).isoformat()
    except (TypeError, ValueError):
        pass

    try:
        d = dt.datetime.fromisoformat(raw.replace("Z", "+00:00"))
        if d.tzinfo is None:
            d = d.replace(tzinfo=dt.timezone.utc)
        return d.astimezone(dt.timezone.utc).isoformat()
    except ValueError:
        pass

    for fmt in ("%a, %d %b %Y %H:%M:%S %z", "%a, %d %b %Y %H:%M:%S %Z",
                "%Y-%m-%dT%H:%M:%S%z", "%Y-%m-%dT%H:%M:%SZ",
                "%Y-%m-%d %H:%M:%S", "%d %b %Y %H:%M:%S %z"):
        try:
            d = dt.datetime.strptime(raw, fmt)
            if d.tzinfo is None:
                d = d.replace(tzinfo=dt.timezone.utc)
            return d.astimezone(dt.timezone.utc).isoformat()
        except ValueError:
            continue

    log.debug("unparsed date: %r", raw)
    return None


# --- retrieval ---------------------------------------------------------------

_RANK = {"high": 3, "medium": 2, "low": 1}


def feed(symbol: str | None = None, limit: int = 20,
         min_impact: str = "low") -> list[dict]:
    """Newest first, filtered to a symbol's asset bucket when given."""
    cutoff = dt.datetime.now(dt.timezone.utc) - dt.timedelta(
        hours=settings.news_lookback_hours)
    bucket = SYMBOL_BUCKET.get((symbol or "").upper())
    floor = _RANK.get(min_impact, 1)

    out = []
    for item in _STORE.values():
        if not item or _RANK.get(item.get("impact", "low"), 1) < floor:
            continue
        ts = item.get("published_at") or item["ingested_at"]
        try:
            if dt.datetime.fromisoformat(ts) < cutoff:
                continue
        except ValueError:
            pass
        # Untagged headlines have no sentiment yet - show them anyway.
        if (item.get("tagged") and bucket
                and item["assets"].get(bucket) == "neutral"
                and item["impact"] == "low"):
            continue  # tagged, and genuinely says nothing about this symbol
        out.append(item)

    out.sort(key=lambda x: (x.get("published_at") or x["ingested_at"]), reverse=True)
    return out[:limit]


def tag_pending(limit: int = 8, workers: int = 2) -> dict:
    """Tag stored headlines that have not been scored yet."""
    pending = [i for i in _STORE.values()
               if i and not i.get("tagged")
               and looks_relevant(i["title"], i.get("summary", ""))][:limit]
    if not pending:
        return {"tagged": 0, "pending": 0}

    def work(i):
        try:
            return tag_article(i["title"], i.get("summary", ""),
                               i.get("source", ""), i.get("url", ""),
                               i.get("published_at"))
        except Exception:
            return None

    with ThreadPoolExecutor(max_workers=workers) as pool:
        done = sum(1 for r in pool.map(work, pending) if r)
    return {"tagged": done,
            "pending": len([i for i in _STORE.values() if i and not i.get("tagged")])}


def relevant_news(symbol: str, limit: int | None = None) -> list[dict]:
    """Compact form for the model context - not the full UI payload."""
    limit = limit or settings.max_news_in_context
    bucket = SYMBOL_BUCKET.get(symbol.upper(), "usd")
    items = feed(symbol, limit=limit * 2)
    items.sort(key=lambda x: (_RANK[x["impact"]],
                              x.get("published_at") or x["ingested_at"]), reverse=True)
    return [
        {
            "title": i["title"][:110],
            "summary": i["summary"],
            "direction": i["assets"].get(bucket, "neutral"),
            "impact": i["impact"],
            "at": i.get("published_at"),
        }
        for i in items[:limit]
    ]


def sentiment(symbol: str) -> dict:
    """Aggregate read for one symbol - drives the header strip."""
    bucket = SYMBOL_BUCKET.get(symbol.upper(), "usd")
    items = feed(symbol, limit=40)
    score, weight = 0.0, 0.0
    for i in items:
        w = _RANK[i["impact"]]
        d = i["assets"].get(bucket, "neutral")
        score += w * (1 if d == "bullish" else -1 if d == "bearish" else 0)
        weight += w
    net = score / weight if weight else 0.0
    return {
        "symbol": symbol.upper(), "bucket": bucket,
        "score": round(net, 2),
        "label": "bullish" if net > 0.15 else "bearish" if net < -0.15 else "neutral",
        "articles": len(items),
        "high_impact": sum(1 for i in items if i["impact"] == "high"),
    }


# --- economic calendar -------------------------------------------------------
# Events come from app/calendar.py (free ForexFactory JSON). Manual events can
# still be injected via load_calendar for testing.

_MANUAL: list[dict] = []


def load_calendar(events: list[dict]) -> None:
    _MANUAL.clear()
    _MANUAL.extend(events)


def upcoming_events(symbol: str, hours: int = 72) -> list[dict]:
    from .calendar import all_events

    now = dt.datetime.now(dt.timezone.utc)
    horizon = now + dt.timedelta(hours=hours)
    currencies = _currencies_for(symbol)
    out = []
    for e in all_events() + _MANUAL:
        try:
            at = dt.datetime.fromisoformat(e["at"])
        except (KeyError, ValueError):
            continue
        if now <= at <= horizon and e.get("currency") in currencies:
            out.append({**e, "minutes_away": round((at - now).total_seconds() / 60)})
    return sorted(out, key=lambda x: x["minutes_away"])


def _currencies_for(symbol: str) -> set[str]:
    s = symbol.upper()
    if s in ("XAUUSD", "XAGUSD", "BTCUSDT", "ETHUSDT", "USOIL"):
        return {"USD"}
    return {s[:3], s[3:6]} if len(s) >= 6 else {"USD"}


def in_news_blackout(symbol: str) -> dict | None:
    window = settings.risk.news_blackout_minutes
    for e in upcoming_events(symbol, hours=6):
        if e.get("impact", 1) >= 3 and e["minutes_away"] <= window:
            return e
    return None
