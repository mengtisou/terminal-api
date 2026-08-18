"""Standalone key check. Run:  python whereami.py

Uses ONLY the standard library and imports nothing from app/, so it works
even when the project itself is misconfigured.
"""
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
KEYS = ["GEMINI_API_KEY", "ANTHROPIC_API_KEY", "OPENAI_API_KEY"]
# Deliberately NO prefix validation. Google changed Gemini keys from "AIza"
# to "AQ." in 2026; anything that hard-codes a key's shape breaks when the
# provider rebrands. Let the API decide if a key is valid.

print("\n" + "=" * 70)
print("  Standalone diagnostic")
print("=" * 70)

print(f"\nRunning from : {HERE}")
print(f"Python       : {sys.version.split()[0]}")

# --- what is actually in this folder? ---
print("\nFiles here:")
names = sorted(p.name for p in HERE.iterdir() if p.is_file())
for n in names:
    mark = "  <-- " if n.lower().startswith(".env") else "      "
    print(f"  {n}{mark}")
if not any(n.lower().startswith(".env") for n in names):
    print("  (no .env-like file found at all)")

print("\napp/ folder:")
app = HERE / "app"
if app.is_dir():
    for p in sorted(app.iterdir()):
        if p.suffix == ".py":
            size = p.stat().st_size
            note = "  <-- EMPTY, must contain load_dotenv" if p.name == "__init__.py" and size < 50 else ""
            print(f"  {p.name:16} {size:6} bytes{note}")
else:
    print("  MISSING - you are in the wrong directory")

# --- exact filename check (Windows hides extensions) ---
print("\n.env filename check:")
exact = HERE / ".env"
if exact.is_file():
    print(f"  [ok ] exactly '.env' exists ({exact.stat().st_size} bytes)")
else:
    print("  [XX ] there is no file named exactly '.env'")
    for p in HERE.iterdir():
        if p.is_file() and p.name.lower().startswith(".env") and p.name != ".env.example":
            print(f"         found '{p.name}' instead - rename it to exactly .env")
            print(f"         PowerShell:  Rename-Item '{p.name}' '.env'")

# --- parse it ---
found = {}
if exact.is_file():
    raw = exact.read_bytes()
    print(f"\n  first bytes: {raw[:12]!r}")
    if raw.startswith(b"\xef\xbb\xbf"):
        print("  note: file has a UTF-8 BOM (Notepad adds this). Handled, but "
              "VS Code saves cleaner.")

    text = raw.decode("utf-8-sig", errors="replace")
    print(f"\n  Lines with a key:")
    any_line = False
    for i, line in enumerate(text.splitlines(), 1):
        st = line.strip()
        if not st or st.startswith("#") or "=" not in st:
            continue
        k, v = st.split("=", 1)
        k, v = k.strip(), v.strip()
        if k not in KEYS:
            continue
        any_line = True
        found[k] = v.strip("\"'")

        issues = []
        if not v:
            issues.append("EMPTY")
        if v.startswith(("'", '"')):
            issues.append("has quotes")
        status = "XX " if issues else "ok "
        detail = "; ".join(issues) if issues else f"{found[k][:8]}... ({len(found[k])} chars)"
        print(f"    [{status}] line {i:2} {k:20} {detail}")
    if not any_line:
        print("    (none - the file has no GEMINI/ANTHROPIC/OPENAI line)")

# --- environment ---
print("\nos.environ (before any app code runs):")
for k in KEYS:
    v = os.getenv(k, "")
    print(f"  [{'ok ' if v else 'XX '}] {k:20} {(v[:8] + '...') if v else 'not set'}")

# --- verdict ---
print("\n" + "-" * 70)
usable = [k for k, v in found.items() if v]
if usable:
    print(f"  VERDICT: {', '.join(usable)} look valid.")
    print("           If the app still says 'no provider configured',")
    print("           you did not restart the server. Ctrl+C, then:")
    print("             python -m uvicorn app.main:app --reload")
else:
    print("  VERDICT: no usable key found.")
    print("           Add one line to .env:  GEMINI_API_KEY=<your key>")
print("=" * 70 + "\n")
