"""Synthetic tests for first_candle_sr_poll_live.py. No FYERS login needed."""
from datetime import datetime
import importlib.util
from pathlib import Path

HERE = Path(__file__).resolve().parent
TARGET = HERE / "first_candle_sr_poll_live.py"
spec = importlib.util.spec_from_file_location("fcsr_live", TARGET)
mod = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(mod)
IST = mod.IST


def run_case(bars):
    engine = mod.StrategyEngine()
    signals = []
    for minute, o, h, l, c in bars:
        ts = IST.localize(datetime(2026, 10, 19, 9, minute))
        signal = engine.add_closed_bar({"timestamp": ts, "open": o, "high": h, "low": l, "close": c, "volume": 100})
        if signal:
            signals.append(signal)
    return signals


def check(name, bars, expected_direction=None, expected_confirmation=None):
    signals = run_case(bars)
    ok = (not signals) if expected_direction is None else bool(signals and signals[-1]["direction"] == expected_direction and signals[-1]["confirmation"] == expected_confirmation)
    if not ok:
        raise AssertionError(f"{name}: unexpected signal result: {[(s['direction'], s['confirmation']) for s in signals]}")
    print(f"PASS: {name}")

ORH100_ORL50 = [(15, 75, 100, 50, 90)]
ORH150_ORL100 = [(15, 125, 150, 100, 140)]
check("strong C1/C2 bullish at ORH", ORH100_ORL50 + [(16,112,117,111,114),(17,105,107,93,94),(18,94,113,93,112)], "BUY", "BULLISH_C12_STRONGER_SECOND")
check("strong C1/C2 bearish at ORH", ORH100_ORL50 + [(16,112,117,111,114),(17,94,106,93,105),(18,105,106,84,85)], "SELL", "BEARISH_C12_STRONGER_SECOND")
check("strong C1/C2 bullish at ORL", ORH150_ORL100 + [(16,86,89,84,86),(17,105,107,94,95),(18,95,115,93,112)], "BUY", "BULLISH_C12_STRONGER_SECOND")
check("strong C1/C2 bearish at ORL", ORH150_ORL100 + [(16,86,89,84,86),(17,94,104,93,100),(18,100,101,84,85)], "SELL", "BEARISH_C12_STRONGER_SECOND")
check("smaller C2 uses C1/C2/C3 fallback", ORH100_ORL50 + [(16,112,117,111,114),(17,105,107,93,94),(18,94,100,93,98),(19,98,115,97,113)], "BUY", "BULLISH_C123_FALLBACK")
check("move away on wrong side is not a breakout/retest", ORH100_ORL50 + [(16,80,85,75,82),(17,105,107,93,94),(18,94,113,93,112)])
check("equal C1/C2 body is strong confirmation", ORH100_ORL50 + [(16,112,117,111,114),(17,105,107,93,95),(18,95,108,94,105)], "BUY", "BULLISH_C12_STRONGER_SECOND")
check("strong C2 that closes on wrong side cannot use C3 fallback", ORH100_ORL50 + [(16,112,117,111,114),(17,96,106,89,90),(18,90,101,89,98),(19,98,115,97,113)])
print("All 8 synthetic strategy tests passed. This does not test live API connectivity or real order execution.")
