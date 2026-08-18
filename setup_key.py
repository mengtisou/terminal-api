"""Write a key into .env and test it for real.

    python setup_key.py gemini AQ.Ab8RN6...
    python setup_key.py anthropic sk-ant-api03-...
    python setup_key.py openai sk-proj-...

Writes clean UTF-8 with no BOM, no quotes, no stray spaces, then makes an
actual API call so you find out immediately whether the key works - rather
than guessing from a "no provider configured" message.

    python setup_key.py            # just test whatever is already in .env
"""
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ENV = ROOT / ".env"

VARS = {
    # LLM providers
    "gemini": "GEMINI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
    "xai": "XAI_API_KEY",
    "deepseek": "DEEPSEEK_API_KEY",
    # Market data providers
    "oanda": "OANDA_TOKEN",
    "twelvedata": "TWELVEDATA_KEY",
    # News + economic calendar
    "finnhub": "FINNHUB_KEY",
    "marketaux": "MARKETAUX_KEY",
}


def read_env() -> dict:
    out = {}
    if ENV.is_file():
        for line in ENV.read_text(encoding="utf-8-sig").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                out[k.strip()] = v.strip().strip("\"'")
    return out


def write_env(updates: dict) -> None:
    """Preserve existing lines, replace only the keys we are setting."""
    lines = []
    if ENV.is_file():
        lines = ENV.read_text(encoding="utf-8-sig").splitlines()

    remaining = dict(updates)
    out = []
    for line in lines:
        stripped = line.strip()
        if stripped and not stripped.startswith("#") and "=" in stripped:
            k = stripped.split("=", 1)[0].strip()
            if k in remaining:
                out.append(f"{k}={remaining.pop(k)}")
                continue
        out.append(line)

    for k, v in remaining.items():
        out.append(f"{k}={v}")

    # newline="\n" and plain utf-8: no BOM, no CRLF surprises.
    with open(ENV, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(out).rstrip() + "\n")


def get(url, headers, timeout=30):
    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.loads(r.read().decode())
    except urllib.error.HTTPError as e:
        return e.code, {"raw": e.read().decode()[:300]}
    except Exception as e:
        return 0, {"error": str(e)}


def gemini_models(key):
    """Ask Google which models THIS key can use. Model availability differs
    per account and Google retires names without warning, so never hard-code."""
    status, body = get(
        "https://generativelanguage.googleapis.com/v1beta/models",
        {"x-goog-api-key": key})
    if status != 200:
        return []
    out = []
    for m in body.get("models", []):
        if "generateContent" in m.get("supportedGenerationMethods", []):
            out.append(m["name"].replace("models/", ""))
    return out


def post(url, payload, headers, timeout=45):
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode(), method="POST",
        headers={"Content-Type": "application/json", **headers})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.loads(r.read().decode())
    except urllib.error.HTTPError as e:
        body = e.read().decode()
        try:
            return e.code, json.loads(body)
        except json.JSONDecodeError:
            return e.code, {"raw": body[:400]}
    except Exception as e:
        return 0, {"error": f"{type(e).__name__}: {e}"}


def test_gemini(key, model="gemini-3.6-flash"):
    return post(
        f"https://generativelanguage.googleapis.com/v1beta/models/"
        f"{model}:generateContent",
        {"contents": [{"role": "user", "parts": [{"text": "Say OK"}]}],
         "generationConfig": {"maxOutputTokens": 20}},
        {"x-goog-api-key": key})


def test_anthropic(key):
    return post(
        "https://api.anthropic.com/v1/messages",
        {"model": "claude-haiku-4-5-20251001", "max_tokens": 20,
         "messages": [{"role": "user", "content": "Say OK"}]},
        {"x-api-key": key, "anthropic-version": "2023-06-01"})


def test_openai(key):
    return post(
        "https://api.openai.com/v1/chat/completions",
        {"model": "gpt-4o-mini", "max_tokens": 20,
         "messages": [{"role": "user", "content": "Say OK"}]},
        {"Authorization": f"Bearer {key}"})


def test_oanda(key):
    """Practice account by default. Live accounts use api-fxtrade.oanda.com."""
    base = read_env().get("OANDA_BASE") or "https://api-fxpractice.oanda.com"
    status, body = get(f"{base}/v3/accounts", {"Authorization": f"Bearer {key}"})
    if status != 200:
        return status, body

    accounts = body.get("accounts", [])
    if not accounts:
        return 200, {"note": "token works but the account list is empty"}

    acct = accounts[0]["id"]
    st, candles = get(
        f"{base}/v3/instruments/XAU_USD/candles?granularity=M15&count=1&price=M",
        {"Authorization": f"Bearer {key}"})
    if st == 200 and candles.get("candles"):
        c = candles["candles"][-1]
        print(f"         account {acct}")
        print(f"         live XAU/USD close: {c['mid']['c']}  at {c['time'][:19]}Z")
    return st, candles


def test_twelvedata(key):
    status, body = get(
        f"https://api.twelvedata.com/time_series?symbol=XAU/USD"
        f"&interval=15min&outputsize=1&apikey={key}", {})
    if isinstance(body, dict) and body.get("status") == "error":
        return 401, {"error": {"message": body.get("message", "")}}
    if status == 200 and body.get("values"):
        v = body["values"][0]
        print(f"         live XAU/USD close: {v['close']}  at {v['datetime']}")
    return status, body


def test_deepseek(key):
    return post("https://api.deepseek.com/v1/chat/completions",
                {"model": "deepseek-chat", "max_tokens": 20,
                 "messages": [{"role": "user", "content": "Say OK"}]},
                {"Authorization": f"Bearer {key}"})


def test_xai(key):
    return post("https://api.x.ai/v1/chat/completions",
                {"model": "grok-4", "max_tokens": 20,
                 "messages": [{"role": "user", "content": "Say OK"}]},
                {"Authorization": f"Bearer {key}"})


def test_finnhub(key):
    status, body = get(
        f"https://finnhub.io/api/v1/news?category=general&token={key}", {})
    if status == 200 and isinstance(body, list) and body:
        print(f"         {len(body)} headlines available")
        print(f"         latest: {body[0].get('headline', '')[:60]}")
    elif status == 401:
        return 401, {"error": {"message": "key rejected"}}
    return status, body


def test_marketaux(key):
    status, body = get(
        f"https://api.marketaux.com/v1/news/all?api_token={key}&language=en&limit=3", {})
    if status == 200 and body.get("data"):
        print(f"         {body.get('meta', {}).get('found', '?')} articles available")
        print(f"         latest: {body['data'][0].get('title', '')[:60]}")
    return status, body


TESTS = {
    "gemini": test_gemini, "anthropic": test_anthropic, "openai": test_openai,
    "finnhub": test_finnhub, "marketaux": test_marketaux,
    "xai": test_xai, "deepseek": test_deepseek,
    "oanda": test_oanda, "twelvedata": test_twelvedata,
}


def run_test(provider, key):
    print(f"\n  Testing {provider} ({key[:8]}... {len(key)} chars)")
    status, body = TESTS[provider](key)

    if status == 200:
        print("  [PASS] the API accepted this key")
        return True

    # 404 on Gemini means the key is FINE but the model name is retired.
    # Discover what this key can actually use and pick one.
    if status == 404 and provider == "gemini":
        print("  [ok]   key is valid - the default model name is retired")
        print("\n  Asking Google which models your key can use...")
        models = gemini_models(key)
        if not models:
            print("  could not list models")
            return False

        flash = [m for m in models if "flash" in m and "lite" not in m
                 and "image" not in m and "tts" not in m]
        pick = (flash or models)[0]
        print(f"  found {len(models)}: {', '.join(models[:6])}"
              + (" ..." if len(models) > 6 else ""))
        print(f"\n  Retesting with {pick}")

        st2, body2 = test_gemini(key, pick)
        if st2 == 200:
            print(f"  [PASS] {pick} works")
            write_env({"GEMINI_CHEAP_MODEL": pick,
                       "GEMINI_REASONING_MODEL": pick,
                       "GEMINI_CHAT_MODEL": pick})
            print(f"  [ok]   pinned {pick} in .env for all three roles")
            return True
        print(f"  [FAIL] {pick} also failed: HTTP {st2}")
        return False

    print(f"  [FAIL] HTTP {status}")
    msg = ""
    if isinstance(body, dict):
        msg = (body.get("error", {}).get("message")
               if isinstance(body.get("error"), dict)
               else str(body.get("error") or body))[:250]
    print(f"         {msg}")

    if status == 400 and provider == "anthropic" and "credit" in msg.lower():
        print("\n  ->  Key is valid but the account has no credits.")
        print("      Add $5 at console.anthropic.com -> Billing")
    elif status in (401, 403):
        print("\n  ->  Key rejected. Generate a fresh one and run this again.")
        if provider == "gemini":
            print("      https://aistudio.google.com/apikey -> Create API key")
        elif provider == "finnhub":
            print("      finnhub.io/register -> free tier, no card")
        elif provider == "marketaux":
            print("      marketaux.com -> free tier, 100 requests/day")
        elif provider == "oanda":
            print("      oanda.com -> log in -> Manage API Access -> Generate")
            print("      If you made a LIVE (not practice) account, also run:")
            print("      python setup_key.py oanda_base https://api-fxtrade.oanda.com")
    elif status == 429:
        print("\n  ->  Rate limited. The key works; just wait a minute.")
        return True
    elif status == 0:
        print("\n  ->  Could not reach the API. Check your internet or firewall.")
    return False


def main():
    args = sys.argv[1:]

    if args:
        if len(args) < 2:
            print(f"usage: python setup_key.py <{'|'.join(VARS)}> <your-key>")
            return 1
        provider, key = args[0].lower(), args[1].strip().strip("\"'")

        # Plain config values, not credentials to test against an API.
        SETTINGS = {
            "mt5_login": "MT5_LOGIN", "mt5_password": "MT5_PASSWORD",
            "mt5_server": "MT5_SERVER", "mt5_path": "MT5_PATH",
            "oanda_base": "OANDA_BASE",
        }
        if provider in SETTINGS and provider != "oanda_base":
            write_env({SETTINGS[provider]: key})
            print(f"\n  Set {SETTINGS[provider]}")
            if provider == "mt5_server":
                print("\n  Now run:  python check_mt5.py")
            return 0

        if provider == "oanda_base":
            write_env({"OANDA_BASE": key})
            print(f"\n  Set OANDA_BASE={key}")
            print("  Now run:  python setup_key.py oanda <your-token>")
            return 0

        if provider not in VARS:
            print(f"unknown provider '{provider}'.")
            print(f"  API keys : {', '.join(VARS)}")
            print(f"  Settings : mt5_login, mt5_password, mt5_server, mt5_path, oanda_base")
            return 1

        write_env({VARS[provider]: key})
        print(f"\n  Wrote {VARS[provider]} to {ENV}")

        back = read_env().get(VARS[provider], "")
        if back != key:
            print(f"  [FAIL] read-back mismatch: got {back[:12]!r}")
            return 1
        print("  [ok]   read back correctly")

        ok = run_test(provider, key)
    else:
        env = read_env()
        present = {p: env[v] for p, v in VARS.items() if env.get(v)}
        if not present:
            print(f"\n  No keys found in {ENV}")
            print("  Run:  python setup_key.py gemini <your-key>")
            return 1
        ok = any(run_test(p, k) for p, k in present.items())

    if ok:
        print("\n  Now restart the server:")
        print("    python -m uvicorn app.main:app --reload")
        print("  Then check https://terminal-ai.onrender.com/providers\n")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
