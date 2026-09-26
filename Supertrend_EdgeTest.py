"""
Supertrend ENTRY edge test (self-contained, causal, no look-ahead).
Strategy = enter on each Supertrend flip, exit on the next opposite flip
(or session end). Measures index points captured per trade, net of cost,
across several timeframes. Answers: does Supertrend capture moves that beat
the ~5-pt cost floor?  Run where fyers_appid.txt / fyers_token.txt live:
    python Supertrend_EdgeTest.py
"""
import warnings, urllib3
from datetime import datetime, timedelta, time as dtime
import numpy as np, pandas as pd, pytz, requests
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning); warnings.filterwarnings("ignore")
IST=pytz.timezone("Asia/Kolkata")

UNDERLYING="NSE:NIFTY50-INDEX"; TIMEFRAME="10S"; HIST_DAYS=30
SESSION_START,SESSION_END=dtime(9,20),dtime(15,30)
TRADE_START,TRADE_END=dtime(9,30),dtime(15,0)
ST_PERIOD=10; ST_MULT=3.0
TIMEFRAMES=["1min","3min","5min","10min","15min"]
COSTS=[0.5,1.0,2.0,5.0]

def fetch_index():
    app=open("fyers_appid.txt").read().strip();tok=open("fyers_token.txt").read().strip()
    auth=tok if ":" in tok else f"{app}:{tok}"
    s=requests.Session();s.verify=False;s.headers.update({"Authorization":auth})
    if s.get("https://api-t1.fyers.in/api/v3/profile").json().get("s")!="ok": raise SystemExit("auth failed")
    end=datetime.now(IST);start=end-timedelta(days=HIST_DAYS)
    r=s.get("https://api-t1.fyers.in/data/history",params={"symbol":UNDERLYING,"resolution":TIMEFRAME,
        "date_format":"1","range_from":start.strftime("%Y-%m-%d"),"range_to":end.strftime("%Y-%m-%d"),"cont_flag":"0"})
    c=r.json().get("candles",[])
    if not c: raise SystemExit("no data")
    df=pd.DataFrame(c,columns=["timestamp","open","high","low","close","volume"])
    df["timestamp"]=pd.to_datetime(df["timestamp"],unit="s",utc=True).dt.tz_convert(IST)
    df=df.sort_values("timestamp").reset_index(drop=True).set_index("timestamp")
    m=df.index.indexer_between_time(SESSION_START,SESSION_END);return df.iloc[m]

def supertrend(ohlc,rule,period,mult):
    r=ohlc.resample(rule).agg({"open":"first","high":"max","low":"min","close":"last","volume":"sum"}).dropna()
    h=r["high"].to_numpy();l=r["low"].to_numpy();cl=r["close"].to_numpy();n=len(r)
    if n<2: return r, np.ones(n,int)
    pc=np.empty(n);pc[0]=cl[0];pc[1:]=cl[:-1]
    trv=np.maximum(h-l,np.maximum(np.abs(h-pc),np.abs(l-pc)))
    atr=pd.Series(trv).ewm(alpha=1/period,adjust=False).mean().to_numpy()
    hl2=(h+l)/2;ub=hl2+mult*atr;db=hl2-mult*atr;fu=np.empty(n);fl=np.empty(n);fu[0]=ub[0];fl[0]=db[0]
    for i in range(1,n):
        fu[i]=ub[i] if(ub[i]<fu[i-1] or cl[i-1]>fu[i-1])else fu[i-1]
        fl[i]=db[i] if(db[i]>fl[i-1] or cl[i-1]<fl[i-1])else fl[i-1]
    t=np.ones(n,int)
    for i in range(1,n): t[i]=(-1 if cl[i]<fl[i] else 1) if t[i-1]==1 else (1 if cl[i]>fu[i] else -1)
    return r, t

def test_tf(ohlc, rule):
    r,t=supertrend(ohlc,rule,ST_PERIOD,ST_MULT)
    if len(r)<3: return None
    cl=r["close"].to_numpy(); idx=r.index
    day=pd.Series(idx).dt.date.to_numpy(); tm=pd.Series(idx).dt.time.to_numpy()
    n=len(r); trades=[]
    i=1
    while i<n:
        if t[i]!=t[i-1] and TRADE_START<=tm[i]<=TRADE_END:   # flip = entry
            d=t[i]; entry=cl[i]; j=i+1
            while j<n and t[j]==d and day[j]==day[i] and tm[j]<=TRADE_END:
                j+=1
            # exit at opposite flip / day-end / trade-end (last valid bar in leg)
            ex=min(j, n-1)
            if day[ex]!=day[i]: ex=j-1
            move=(cl[ex]-entry)*d
            trades.append(move); i=j
        else:
            i+=1
    if not trades: return None
    a=np.array(trades); return a

def main():
    ohlc=fetch_index()
    print(f"\nData bars(10s): {len(ohlc)} | Supertrend({ST_PERIOD},{ST_MULT}) flip-to-flip, intraday\n")
    hdr=f"{'TF':>7} {'trades':>7} {'win%':>6} {'avgGross':>9} "+" ".join(f"{'net@'+format(c,'g'):>8}" for c in COSTS)+f" {'totNet@5':>9}"
    print(hdr); print("-"*len(hdr))
    for tf in TIMEFRAMES:
        a=test_tf(ohlc, tf)
        if a is None: print(f"{tf:>7}   (no trades)"); continue
        win=100*(a>0).mean(); g=a.mean()
        nets=" ".join(f"{g-c:>8.2f}" for c in COSTS)
        totnet5=(a-5.0).sum()
        print(f"{tf:>7} {len(a):>7} {win:>5.1f}% {g:>9.2f} {nets} {totnet5:>9.0f}")
    print("\navgGross = avg index points captured per trend leg (before cost).")
    print("net@c = avgGross - c (per-trade expectancy after round-trip cost c).")
    print("totNet@5 = total index points over all trades after 5-pt cost each.")
    print("If avgGross >> 5 and net@5 > 0, Supertrend captures real directional")
    print("moves on the INDEX -> worth a full cost/option-aware backtest next.")

if __name__=="__main__":
    main()
