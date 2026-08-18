"""Offline tests. No API key, no network. Run: python test_offline.py

These cover the parts that must not break: the feature engine and the
validator. The model is not tested here — it is not deterministic.
"""
import json

from app.features import build_snapshot
from app.signals import validate

SNAP = {"price": 4375.70, "volatility": {"atr14": 3.85}}

CASES = [
    ("good long", {"bias": "long", "entry": 4374.0, "stop_loss": 4368.0,
                   "take_profit": [4384.0], "confidence": 0.62}, True),
    ("good short", {"bias": "short", "entry": 4377.0, "stop_loss": 4383.0,
                    "take_profit": [4366.0], "confidence": 0.55}, True),
    ("no_trade", {"bias": "no_trade", "entry": None, "stop_loss": None,
                  "take_profit": [], "confidence": 0.0}, True),
    ("stop on wrong side", {"bias": "long", "entry": 4374.0, "stop_loss": 4380.0,
                            "take_profit": [4384.0], "confidence": 0.7}, False),
    ("target on wrong side", {"bias": "short", "entry": 4376.0, "stop_loss": 4382.0,
                              "take_profit": [4390.0], "confidence": 0.7}, False),
    ("hallucinated entry", {"bias": "short", "entry": 4600.0, "stop_loss": 4610.0,
                            "take_profit": [4560.0], "confidence": 0.8}, False),
    ("risk:reward too low", {"bias": "long", "entry": 4374.0, "stop_loss": 4366.0,
                             "take_profit": [4378.0], "confidence": 0.6}, False),
    ("stop too tight", {"bias": "long", "entry": 4375.0, "stop_loss": 4374.5,
                        "take_profit": [4380.0], "confidence": 0.6}, False),
    ("stop too wide", {"bias": "long", "entry": 4375.0, "stop_loss": 4353.0,
                       "take_profit": [4420.0], "confidence": 0.6}, False),
    ("confidence too low", {"bias": "short", "entry": 4376.0, "stop_loss": 4382.0,
                            "take_profit": [4362.0], "confidence": 0.2}, False),
    ("missing stop", {"bias": "long", "entry": 4374.0, "stop_loss": None,
                      "take_profit": [4384.0], "confidence": 0.7}, False),
]


def test_validator():
    failures = 0
    for name, sig, expect_ok in CASES:
        result = validate(dict(sig), SNAP)
        mark = "ok " if result.ok == expect_ok else "FAIL"
        if result.ok != expect_ok:
            failures += 1
        detail = result.errors[0] if result.errors else ""
        print(f"  [{mark}] {name:24} {'accept' if result.ok else 'reject':7} {detail}")
    return failures


def test_snapshot():
    failures = 0
    for symbol, tf in [("XAUUSD", "15m"), ("BTCUSDT", "1h")]:
        snap = build_snapshot(symbol, tf)
        size = len(json.dumps(snap)) // 4
        required = ["price", "trend", "momentum", "volatility", "support", "resistance", "session"]
        missing = [k for k in required if k not in snap]
        ok = not missing and size < 1500
        if not ok:
            failures += 1
        print(f"  [{'ok ' if ok else 'FAIL'}] {symbol} {tf:4} ~{size} tokens"
              + (f"  missing: {missing}" if missing else ""))
    return failures


if __name__ == "__main__":
    print("validator:")
    f1 = test_validator()
    print("\nsnapshot:")
    f2 = test_snapshot()
    total = f1 + f2
    print(f"\n{'all passed' if total == 0 else f'{total} failures'}")
    raise SystemExit(1 if total else 0)
