"""
Telegram push for gold news - drop this file into your terminal-api backend
(next to your other route/util modules) and wire it into news ingestion.

Setup
-----
1. Message @BotFather on Telegram -> /newbot -> follow the prompts.
   You get a token that looks like: 123456789:AAExxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

2. Send /start to your new bot from whichever Telegram account/chat should
   receive alerts (works for your personal DM or a group you add the bot to).

3. Find your chat_id:
   https://api.telegram.org/bot<YOUR_TOKEN>/getUpdates
   Open that URL in a browser after step 2. Look for "chat":{"id": ...}.
   For a group chat this id is negative (e.g. -1001234567890) - that's normal.

4. Add to your backend's .env:
   TELEGRAM_BOT_TOKEN=123456789:AAExxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
   TELEGRAM_CHAT_ID=123456789
   # Optional: only push high-impact / gold-tagged items (default true).
   # Set to "0" to push every headline instead.
   TELEGRAM_NEWS_FILTER=1

5. Import and call notify_news_items(items) from wherever your ingestion
   pipeline finishes tagging a fresh batch (see the bottom of this file for
   the exact integration point).
"""

import os
import asyncio
import httpx

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")
TELEGRAM_NEWS_FILTER = os.getenv("TELEGRAM_NEWS_FILTER", "1") != "0"

# Items we've already pushed, so a re-poll of the same feed doesn't spam the
# chat with the same headline every time /news/ingest runs. Swap this set for
# a persisted store (a small sqlite table, a Redis set, a column on your news
# table) if the process restarts often - in memory it resets on deploy.
_notified_ids: set[str] = set()

def _item_key(item: dict) -> str:
    # url is the stable identifier when present; fall back to title+source so
    # feeds that omit a url still dedupe correctly.
    return item.get("url") or f"{item.get('source','')}|{item.get('title','')}"

def _is_notifiable(item: dict) -> bool:
    if not TELEGRAM_NEWS_FILTER:
        return True
    if item.get("impact") == "high":
        return True
    assets = item.get("assets") or {}
    return assets.get("gold") in ("bullish", "bearish")

def _format_message(item: dict) -> str:
    assets = item.get("assets") or {}
    gold = assets.get("gold")
    tag = f"🟢 GOLD BULLISH" if gold == "bullish" else \
          f"🔴 GOLD BEARISH" if gold == "bearish" else \
          f"⚪ {(item.get('impact') or 'news').upper()}"
    lines = [
        tag,
        f"<b>{_escape(item.get('title',''))}</b>",
    ]
    if item.get("takeaway"):
        lines.append(_escape(item["takeaway"]))
    src = item.get("source")
    if src:
        lines.append(f"— {_escape(src)}")
    if item.get("url"):
        lines.append(item["url"])
    return "\n".join(lines)

def _escape(text: str) -> str:
    # HTML parse_mode only needs these three escaped.
    return (text or "").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

async def _send(text: str) -> bool:
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        return False
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": text,
        "parse_mode": "HTML",
        "disable_web_page_preview": False,
    }
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            r = await client.post(url, json=payload)
            return r.status_code == 200
    except Exception:
        return False

async def notify_news_items(items: list[dict]) -> int:
    """Call this with the freshly-tagged batch from your ingestion step.
    Sends one Telegram message per new, notifiable item. Returns how many
    were sent, so a manual trigger endpoint can report back something useful.
    """
    sent = 0
    for item in items:
        key = _item_key(item)
        if key in _notified_ids:
            continue
        _notified_ids.add(key)
        if not _is_notifiable(item):
            continue
        if await _send(_format_message(item)):
            sent += 1
            await asyncio.sleep(0.4)  # stay well under Telegram's rate limit
    return sent

async def send_test_message() -> bool:
    return await _send("✅ Gold terminal is connected. News alerts will arrive here.")
