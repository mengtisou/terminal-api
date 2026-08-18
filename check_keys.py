"""Key + routing diagnostic. Run: python check_keys.py

Traces the whole chain: does .env exist, is it parseable, did python-dotenv
load it, and what does each provider actually see.
"""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ENV = ROOT / ".env"
KEYS = ["ANTHROPIC_API_KEY", "GEMINI_API_KEY", "OPENAI_API_KEY"]


def show(label, ok, detail=""):
    print(f"  [{'ok ' if ok else 'XX '}] {label:34} {detail}")


print("\n" + "=" * 68)
print("  Key diagnostic")
print("=" * 68)

# --- 1. dotenv installed? ---
print("\n1. python-dotenv")
try:
    import dotenv
    show("installed", True, getattr(dotenv, "__version__", ""))
except ImportError:
    show("installed", False, "->  pip install python-dotenv")

# --- 2. .env file ---
print("\n2. .env file")
show("exists", ENV.exists(), str(ENV))

raw_keys = {}
if ENV.exists():
    text = ENV.read_text(encoding="utf-8-sig")
    show("readable", True, f"{len(text)} bytes")

    for i, line in enumerate(text.splitlines(), 1):
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        k, v = k.strip(), v.strip()
        if k not in KEYS:
            continue
        raw_keys[k] = v

        problems = []
        if v.startswith(("'", '"')) or v.endswith(("'", '"')):
            problems.append("has quotes - remove them")
        if not v:
            problems.append("empty value")
        if " " in v:
            problems.append("contains a space")
        # No prefix checks. Google changed Gemini keys from "AIza" to "AQ."
        # in 2026 - hard-coding a credential's shape only creates false alarms.

        show(f"line {i}: {k}", not problems,
             "; ".join(problems) if problems else f"{v[:10]}... ({len(v)} chars)")

    if not raw_keys:
        show("any provider key found", False, "no ANTHROPIC/GEMINI/OPENAI line")
else:
    print("\n     Fix:  Copy-Item .env.example .env")
    print("     then open .env and paste a key.")

# --- 3. loaded into the process? ---
print("\n3. Loaded into environment (via app/__init__.py)")
import app  # noqa: F401  - triggers load_dotenv
for k in KEYS:
    v = os.getenv(k, "")
    show(k, bool(v), f"{v[:10]}... ({len(v)} chars)" if v else "not set")
    if k in raw_keys and raw_keys[k] and not v:
        print(f"         .env has it but os.getenv does not -> check for BOM / odd encoding")

# --- 4. what the app will actually do ---
print("\n4. Provider routing")
from app import llm

st = llm.status()
for name, state in st["providers"].items():
    show(name, state == "ready", state)

print()
for role, chain in st["roles"].items():
    show(f"role: {role}", bool(chain), " -> ".join(chain) if chain else "NO PROVIDER")

print(f"\n  config source: {st['source']}   council: {st['council']}")

if not any(v == "ready" for v in st["providers"].values()):
    print("\n  >> No usable provider. Chat and signals will fail.")
    print("     Free key: https://aistudio.google.com/apikey -> Create API key")
    print("     then put this in .env:   GEMINI_API_KEY=<paste it>")
else:
    print("\n  >> Ready. Restart the server if it is already running.")

print("=" * 68 + "\n")
