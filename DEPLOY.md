# Hosting the backend

The frontend is static and can live anywhere. The backend is a normal Docker
container, so most platforms will take it as-is.

**Before you start, understand what changes in the cloud:**

| | Local (now) | Hosted |
|---|---|---|
| MT5 | works | **not available** (Windows only) |
| Gold data | your broker feed | Twelve Data (800/day) |
| Cost | free | free tier, with limits |
| Who can use it | you | anyone with the URL |
| API bill | yours, one user | yours, **every** user |

That last row is the one that bites. Every visitor's chat message spends your
Anthropic credit. Add auth before sharing the URL publicly.

---

## Render (simplest free option)

1. Push this folder to a GitHub repo. `.env` is gitignored — check it stays out.
2. [render.com](https://render.com) → **New → Blueprint** → pick the repo.
   It reads `render.yaml` automatically.
3. Add your keys under **Environment**:
   ```
   GEMINI_API_KEY, ANTHROPIC_API_KEY, TWELVEDATA_KEY, FINNHUB_KEY
   ```
4. Deploy. You get `https://terminal-ai-xxxx.onrender.com`.

**The free tier sleeps after 15 minutes idle.** The first request then takes
roughly 50 seconds to wake it. The frontend allows 60s before reporting a
failure, so it looks slow rather than broken — but it is not suitable for a
chart you want to glance at.

---

## Hugging Face Spaces (free, no card)

1. New Space → SDK **Docker** → Public or Private.
2. Push these files. It builds the Dockerfile directly.
3. Keys go under **Settings → Repository secrets**.
4. Add to the Dockerfile: `ENV PORT=7860` (Spaces expects that port).

No sleep on the free tier, which makes it better than Render for this. CPU is
modest but the workload is small — pandas over 300 candles.

---

## Fly.io / Koyeb

Both take the Dockerfile. Fly needs a card even for the free allowance; Koyeb
has one free service without one.

```bash
fly launch --dockerfile Dockerfile
fly secrets set GEMINI_API_KEY=... ANTHROPIC_API_KEY=...
```

---

## Own VPS (~$4/month)

Hetzner, Vultr or DigitalOcean if you want no sleep, no request caps and full
control. Not free, but it is the only option that can also run MT5 — on a
**Windows** VPS with the terminal open and logged in.

---

## After deploying

**Point the frontend at it:**

```
https://yourname.github.io/repo/?api=https://your-backend.onrender.com
```

The URL is remembered, so you only need the parameter once.

**Lock down CORS** — set `ALLOWED_ORIGINS` to your frontend URL exactly:

```
ALLOWED_ORIGINS=https://yourname.github.io
```

**Configure data.** Without MT5 you need `TWELVEDATA_KEY` for spot gold; Yahoo
is the fallback but serves futures, not spot. `ALLOW_SYNTHETIC=0` is set in the
blueprint so a provider failure raises loudly instead of quietly serving
invented prices.

---

## What resets on restart

The news store, candle cache, event dates and routing overrides live in process
memory or local files. A container restart clears them — headlines re-fetch on
the next pull, and routing falls back to `.env`.

For anything that must survive, move it to Postgres. Render and Fly both offer
a free tier.

---

## One worker, on purpose

The Dockerfile pins `--workers 1`. The cache, news store and routing overrides
are per-process, so two workers would each hold a different copy and you would
get inconsistent answers depending on which one served the request. Scaling
past one worker means moving that state to Redis or Postgres first.
