"""
Signal-edge diagnostic (self-contained, no look-ahead).
Does NIFTY actually move the signalled way after a raw VAM signal?

Tests EVERY causal vam_buy_sig / vam_sell_sig in the trade window (no HTF
filter, no pivot, no sequencing) and measures index-direction hit-rate at
several horizons. Isolates the predictive power of the entry signal itself.

Run in the folder that has fyers_appid.txt / fyers_token.txt:
    python VAM_SignalEdgeTest.py
"""
import os, warnings, urllib3
from datetime import datetime, timedelta, time as dtime
import numpy as np
import pandas as pd
import pytz
import requests

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
warnings.filterwarnings("ignore")
IST = pytz.timezone("Asia/Kolkata")

# ---- config (mirrors the backtest's signal-relevant settings) ----
UNDERLYING = "NSE:NIFTY50-INDEX"
TIMEFRAME  = "10S"
HIST_DAYS  = 30
SESSION_START, SESSION_END = dtime(9, 20), dtime(15, 30)
TRADE_START,  TRADE_END    = dtime(9, 30), dtime(15, 0)
VAM_BUY, VAM_SELL = 0.10, -0.10
LENS  = [8, 10, 14, 20, 28]
MULTS = [1.05, 1.10, 1.15, 1.20, 1.25]
BANDS = [1.15, 1.20, 1.25, 1.30, 1.35]
SMOO  = [2, 2, 3, 3, 4]
HORIZONS_BARS = [3, 6, 12, 18, 30, 60, 90]   # 30s,1,2,3,5,10,15 min

def ema(s, span): return s.ewm(span=span, adjust=False).mean()
def tr(df):
    hl = df["high"] - df["low"]
    hc = (df["high"] - df["close"].shift()).abs()
    lc = (df["low"]  - df["close"].shift()).abs()
    return pd.Series(np.maximum(hl, np.maximum(hc, lc)), index=df.index)

def r1_side(df, length, r1_mult, band_mult, smooth):
    src = df["close"]; r1 = ema(ema(tr(df), length) * r1_mult, smooth)
    s = src.to_numpy(); r = r1.to_numpy(); n = len(df)
    c = np.empty(n); c[0] = s[0]
    for i in range(1, n):
        p = c[i-1]; d = s[i] - p
        c[i] = p + (d - r[i]) if d > r[i] else (p + (d + r[i]) if d < -r[i] else p)
    center = pd.Series(c, index=df.index)
    up = ema(center + r1 * band_mult, smooth); lo = ema(center - r1 * band_mult, smooth)
    bl = (src > up.shift(1)) & (src.shift(1) <= up.shift(2))
    bs = (src < lo.shift(1)) & (src.shift(1) >= lo.shift(2))
    bln = bl.to_numpy(); bsn = bs.to_numpy(); side = np.ones(n, dtype=int)
    for i in range(1, n):
        side[i] = 1 if bln[i] else (-1 if bsn[i] else side[i-1])
    return side

def vam_signals(df):
    sides = [r1_side(df, LENS[k], MULTS[k], BANDS[k], SMOO[k]) for k in range(5)]
    score = np.mean(sides, axis=0) / 5.0
    df["vam_score"] = score
    df["buy"]  = (pd.Series(score).shift(1).to_numpy() <= VAM_BUY)  & (score > VAM_BUY)
    df["sell"] = (pd.Series(score).shift(1).to_numpy() >= VAM_SELL) & (score < VAM_SELL)
    return df

def fetch_index():
    app = open("fyers_appid.txt").read().strip()
    tok = open("fyers_token.txt").read().strip()
    auth = tok if ":" in tok else f"{app}:{tok}"
    sess = requests.Session(); sess.verify = False
    sess.headers.update({"Authorization": auth})
    prof = sess.get("https://api-t1.fyers.in/api/v3/profile").json()
    if prof.get("s") != "ok":
        raise SystemExit(f"Fyers auth failed: {prof}")
    print(f"Connected: {prof['data']['name']}")
    end = datetime.now(IST); start = end - timedelta(days=HIST_DAYS)
    r = sess.get("https://api-t1.fyers.in/data/history", params={
        "symbol": UNDERLYING, "resolution": TIMEFRAME, "date_format": "1",
        "range_from": start.strftime("%Y-%m-%d"), "range_to": end.strftime("%Y-%m-%d"),
        "cont_flag": "0"})
    candles = r.json().get("candles", [])
    if not candles: raise SystemExit("No index data returned.")
    df = pd.DataFrame(candles, columns=["timestamp","open","high","low","close","volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s", utc=True).dt.tz_convert(IST)
    df = df.sort_values("timestamp").reset_index(drop=True).set_index("timestamp")
    m = df.index.indexer_between_time(SESSION_START, SESSION_END)
    return df.iloc[m].reset_index()

def main():
    df = vam_signals(fetch_index())
    ts = pd.to_datetime(df["timestamp"]); close = df["close"].to_numpy(dtype=float)
    day = ts.dt.date.to_numpy(); n = len(df)
    inwin = ((ts.dt.time >= TRADE_START) & (ts.dt.time <= TRADE_END)).to_numpy()
    buy = df["buy"].to_numpy(); sell = df["sell"].to_numpy()
    sig = [(i, +1) for i in range(n) if buy[i] and inwin[i]] + \
          [(i, -1) for i in range(n) if sell[i] and inwin[i]]
    print(f"\nData: {day[0]} -> {day[-1]} | {n} bars")
    print(f"Raw causal VAM signals (trade window): "
          f"{sum(1 for _,d in sig if d>0)} buys, {sum(1 for _,d in sig if d<0)} sells\n")
    COSTS = [0.5, 1.0, 2.0, 5.0]   # round-trip futures cost, in index points
    print("gross = mean favorable move per trade (pts, already nets winners/losers).")
    print("net@c = gross - c  = per-trade expectancy in points AFTER a round-trip cost of c.\n")
    hdr = f"{'horizon':>6} {'n':>6} {'hit%':>6} {'gross':>7} " + " ".join(f"{'net@'+format(c,'g'):>8}" for c in COSTS)
    print(hdr); print("-"*len(hdr))
    for H in HORIZONS_BARS:
        moves = []; hits = 0
        for i, d in sig:
            j = i + H
            if j >= n or day[j] != day[i]: continue
            fav = (close[j] - close[i]) * d
            moves.append(fav); hits += 1 if fav > 0 else 0
        if not moves: continue
        tot = len(moves); hp = 100*hits/tot; mv = float(np.mean(moves))
        nets = " ".join(f"{mv-c:>8.2f}" for c in COSTS)
        print(f"{H*10/60:>5.1f}m {tot:>6} {hp:>5.1f}% {mv:>7.2f} {nets}")
    print("\nnet@c > 0  -> the per-trade edge survives that round-trip cost.")
    print("Realistic NIFTY FUTURES round-trip is ~5-6 pts (STT alone ~4.8 at 24000,")
    print("plus exchange fees, stamp, brokerage, GST) -> look hardest at net@5.")
    print("Options are worse still (bid/ask + theta on top). If net@5 <= 0 everywhere,")
    print("the signal cannot pay its own transaction costs on any liquid instrument.")

if __name__ == "__main__":
    main()
