"""Synthetic tests for the uploaded-backtest live StrategyEngine. No credentials/orders."""
from datetime import datetime, timedelta, date
import pytz
from first_candle_sr_backtest_live import StrategyEngine, Config
IST = pytz.timezone("Asia/Kolkata")

def make_bar(day, hhmm, o, h, l, c):
    t = datetime.strptime(f"{day} {hhmm}", "%Y-%m-%d %H:%M")
    return {"timestamp": IST.localize(t), "open": float(o), "high": float(h), "low": float(l), "close": float(c), "volume": 100.0}

def run_bars(bars):
    engine = StrategyEngine()
    out = []
    for b in bars:
        sig = engine.add_closed_bar(b)
        if sig:
            out.append(sig)
    return out

def opening(day):
    return make_bar(day, "09:15", 95, 100, 90, 95)

def bullish_base(day, c1, c2, c3=None, retest=(100.2, 99.8, 100.1)):
    arr = [opening(day),
           make_bar(day, "09:16", 100, 102, 100.5, 101.5),   # breakout close above ORH
           make_bar(day, "09:17", 101.5, 103, 101.1, 102.0), # clearly away from ORH
           make_bar(day, "09:18", retest[0], max(100, retest[0]), min(100, retest[1]), retest[2])]
    arr += c1
    arr += c2
    if c3:
        arr += c3
    return arr

def bearish_base(day, c1, c2, c3=None, retest=(90.2, 89.8, 89.9)):
    arr = [opening(day),
           make_bar(day, "09:16", 90, 90, 88, 89),            # breakdown close below ORL
           make_bar(day, "09:17", 89, 89, 87.5, 88),         # clearly away below ORL
           make_bar(day, "09:18", retest[0], max(90, retest[0]), min(90, retest[1]), retest[2])]
    arr += c1
    arr += c2
    if c3:
        arr += c3
    return arr

def test_bullish_strong_c2():
    d="2026-10-19"
    bars=bullish_base(d,
      [make_bar(d,"09:19",101,101.3,99.9,100.2)],
      [make_bar(d,"09:20",100.2,101.6,100.1,101.5)])
    sig=run_bars(bars)
    assert len(sig)==1, sig
    assert sig[0]["direction"]=="BUY" and sig[0]["confirmation"]=="BULLISH_C12_STRONGER_SECOND", sig
    assert sig[0]["c3_time"]=="", sig

def test_bearish_strong_c2():
    d="2026-10-19"
    bars=bearish_base(d,
      [make_bar(d,"09:19",89,90.1,88.9,89.8)],
      [make_bar(d,"09:20",89.8,89.9,88.2,88.3)])
    sig=run_bars(bars)
    assert len(sig)==1, sig
    assert sig[0]["direction"]=="SELL" and sig[0]["confirmation"]=="BEARISH_C12_STRONGER_SECOND", sig

def test_bullish_c3_fallback():
    d="2026-10-19"
    bars=bullish_base(d,
      [make_bar(d,"09:19",101.5,101.6,99.9,100.0)],
      [make_bar(d,"09:20",100.1,100.7,100.0,100.5)],
      [make_bar(d,"09:21",100.5,102.2,100.4,102.0)])
    sig=run_bars(bars)
    assert len(sig)==1, sig
    assert sig[0]["confirmation"]=="BULLISH_C123_FALLBACK", sig

def test_bearish_c3_fallback():
    d="2026-10-19"
    bars=bearish_base(d,
      [make_bar(d,"09:19",88.5,90.1,88.4,90.0)],
      [make_bar(d,"09:20",89.9,90.0,89.4,89.5)],
      [make_bar(d,"09:21",89.5,89.6,87.9,88.0)])
    sig=run_bars(bars)
    assert len(sig)==1, sig
    assert sig[0]["confirmation"]=="BEARISH_C123_FALLBACK", sig

def test_near_but_not_exact_retest_rejected():
    d="2026-10-19"
    bars=[opening(d),
          make_bar(d,"09:16",100,102,100.5,101.5),
          make_bar(d,"09:17",101.5,103,101.1,102.0),
          make_bar(d,"09:18",100.3,100.6,100.1,100.4),  # gap/near but does not intersect ORH=100
          make_bar(d,"09:19",100.3,101.3,100.1,100.8),  # remains entirely above ORH
          make_bar(d,"09:20",100.8,101.6,100.2,101.5)]
    sig=run_bars(bars)
    assert not sig, sig

def main():
    assert Config.QUANTITY==65 and Config.MAX_TRADES_PER_DAY==1
    assert Config.STOP_LOSS_SPOT_POINTS==40 and Config.TAKE_PROFIT_SPOT_POINTS==130
    tests=[test_bullish_strong_c2,test_bearish_strong_c2,test_bullish_c3_fallback,test_bearish_c3_fallback,test_near_but_not_exact_retest_rejected]
    for t in tests:
        t(); print("PASS", t.__name__)
    print(f"Passed {len(tests)} synthetic tests.")
if __name__=="__main__": main()
