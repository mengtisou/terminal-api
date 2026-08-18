"""MetaTrader 5 connection check. Run: python check_mt5.py

Verifies the terminal is reachable, works out what your broker calls each
instrument, and prints live prices.
"""
import sys

print("\n" + "=" * 66)
print("  MetaTrader 5 check")
print("=" * 66)

try:
    import MetaTrader5 as mt5
except ImportError:
    print("\n  [XX] MetaTrader5 package not installed")
    print("       pip install MetaTrader5")
    print("       (Windows only - the package does not exist for Linux/macOS)")
    raise SystemExit(1)

print(f"\n  [ok] package installed (v{mt5.__version__})")

import os
sys.path.insert(0, ".")
try:
    import app  # loads .env
except Exception:
    pass

# Try a plain attach first, then with explicit credentials if .env has them.
attempts = [({}, "attach to running terminal")]
login, pw, server = (os.getenv("MT5_LOGIN"), os.getenv("MT5_PASSWORD"),
                     os.getenv("MT5_SERVER"))
path = os.getenv("MT5_PATH")
if path:
    attempts.insert(0, ({"path": path}, f"terminal at {path}"))
if login and pw and server:
    kw = {"login": int(login), "password": pw, "server": server}
    if path:
        kw["path"] = path
    attempts.append((kw, f"explicit login {login} on {server}"))

connected, last = False, ("", "")
for kwargs, label in attempts:
    if mt5.initialize(**kwargs):
        print(f"  [ok] {label}")
        connected = True
        break
    last = mt5.last_error()
    print(f"  [--] {label}: {last[0]} {last[1]}")

if not connected:
    code, msg = last
    print(f"\n  [XX] could not connect ({code}: {msg})")
    if code == -6:
        print("""
       -6 means Python FOUND your terminal but it is not logged in
       to a trading account.

       Fix A - log in through the MT5 window:
         File > Open an Account > choose a broker > Next
         > Open a demo account > fill the form > Next
         Note the login number, password and server it gives you.
         Bottom-right of MT5 should then show a green connection
         indicator with a data rate, not "No connection".

       Fix B - pass the credentials to this app:
         python setup_key.py mt5_login    12345678
         python setup_key.py mt5_password YourPassword
         python setup_key.py mt5_server   Exness-MT5Trial
         python check_mt5.py

       The server name must match EXACTLY what MT5 shows in the
       Navigator panel under Accounts.""")
    else:
        print("""
       Checklist:
         1. Is MetaTrader 5 running and logged in?
         2. Tools > Options > Expert Advisors >
            tick "Allow algorithmic trading"
         3. If you have several MT5 installs, point at the right one:
            python setup_key.py mt5_path "C:\\Program Files\\MetaTrader 5\\terminal64.exe" """)
    raise SystemExit(1)

info, acct = mt5.terminal_info(), mt5.account_info()
print(f"  [ok] connected to {info.company if info else 'terminal'}")
if acct:
    print(f"       account {acct.login} · {acct.server} · {acct.currency}"
          f" · balance {acct.balance:,.2f}")

sys.path.insert(0, ".")
from app.mt5_provider import CANDIDATES, resolve_symbol  # noqa: E402

print(f"\n  Symbols ({len(mt5.symbols_get() or [])} offered by this broker):")
found = 0
for want in ["XAUUSD", "EURUSD", "GBPUSD", "BTCUSDT", "USOIL"]:
    name = resolve_symbol(want)
    if not name:
        print(f"    [XX] {want:8} not found")
        continue
    found += 1
    tick = mt5.symbol_info_tick(name)
    rates = mt5.copy_rates_from_pos(name, mt5.TIMEFRAME_M15, 0, 1)
    bars = len(mt5.copy_rates_from_pos(name, mt5.TIMEFRAME_H1, 0, 20000) or [])
    price = f"{tick.bid:,.3f} / {tick.ask:,.3f}" if tick else "no tick"
    print(f"    [ok] {want:8} -> {name:12} {price:24} {bars:,} 1h bars")

print("\n" + "-" * 66)
if found:
    print(f"  {found} symbols ready. MT5 is first in the routing chain, so restart")
    print("  the server and the badge will change to 'mt5'.")
else:
    print("  No symbols matched. In MT5, right-click Market Watch > Show All,")
    print("  then note the exact gold symbol name and tell me what it is.")
print("=" * 66 + "\n")
mt5.shutdown()
