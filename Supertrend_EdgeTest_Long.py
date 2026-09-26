"""
Supertrend ENTRY edge test over LONG history (self-contained, causal).
Fetches 1-min index data in chunks over up to LOOKBACK_DAYS (default 365),
resamples to each timeframe, and measures index points captured per Supertrend
flip-to-flip trade, net of cost. Goal: hundreds of trades so the 10/15-min
result from the 30-day run can be trusted (or not).
Run where fyers_appid.txt / fyers_token.txt live:
    python Supertrend_EdgeTest_Long.py
"""
import time, warnings, urllib3
from datetime import datetime, timedelta, time as dtime
import numpy as np, pandas as pd, pytz, requests
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning); warnings.filterwarnings("ignore")
IST=pytz.timezone("Asia/Kolkata")

UNDERLYING="NSE:NIFTY50-INDEX"; BASE_RES="1"; LOOKBACK_DAYS=365; CHUNK_DAYS=90
SESSION_START,SESSION_END=dtime(9,20),dtime(15,30)
TRADE_START,TRADE_END=dtime(9,30),dtime(15,0)
ST_PERIOD=10; ST_MULT=3.0
TIMEFRAMES=["3min","5min","10min","15min","30min"]
COSTS=[0.5,1.0,2.0,5.0]

def _session():
    app=open("fyers_appid.txt").read().strip();tok=open("fyers_token.txt").read().strip()
    auth=tok if ":" in tok else f"{app}:{tok}"
    s=requests.Session();s.verify=False;s.headers.update({"Authorization":auth})
    if s.get("https://api-t1.fyers.in/api/v3/profile").json().get("s")!="ok": raise SystemExit("auth failed")
    return s

def fetch_long():
    s=_session(); end=datetime.now(IST); start=end-timedelta(days=LOOKBACK_DAYS)
    frames=[]; cur=start
    while cur < end:
        ce=min(cur+timedelta(days=CHUNK_DAYS), end)
        r=s.get("https://api-t1.fyers.in/data/history",params={"symbol":UNDERLYING,"resolution":BASE_RES,
            "date_format":"1","range_from":cur.strftime("%Y-%m-%d"),"range_to":ce.strftime("%Y-%m-%d"),"cont_flag":"0"})
        try: c=r.json().get("candles",[])
        except Exception: c=[]
        if c:
            f=pd.DataFrame(c,columns=["timestamp","open","high","low","close","volume"])
            frames.append(f); print(f"  {cur.date()} -> {ce.date()}: {len(c)} bars")
        cur=ce+timedelta(days=1); time.sleep(0.4)
    if not frames: raise SystemExit("no data returned")
    df=pd.concat(frames,ignore_index=True)
    df["timestamp"]=pd.to_datetime(df["timestamp"],unit="s",utc=True).dt.tz_convert(IST)
    df=df.drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True).set_index("timestamp")
    m=df.index.indexer_between_time(SESSION_START,SESSION_END); return df.iloc[m]

def supertrend(ohlc,rule,period,mult):
    r=ohlc.resample(rule).agg({"open":"first","high":"max","low":"min","close":"last","volume":"sum"}).dropna()
    h=r["high"].to_numpy();l=r["low"].to_numpy();cl=r["close"].to_numpy();n=len(r)
    if n<2: return r,np.ones(n,int)
    pc=np.empty(n);pc[0]=cl[0];pc[1:]=cl[:-1]
    trv=np.maximum(h-l,np.maximum(np.abs(h-pc),np.abs(l-pc)))
    atr=pd.Series(trv).ewm(alpha=1/period,adjust=False).mean().to_numpy()
    hl2=(h+l)/2;ub=hl2+mult*atr;db=hl2-mult*atr;fu=np.empty(n);fl=np.empty(n);fu[0]=ub[0];fl[0]=db[0]
    for i in range(1,n):
        fu[i]=ub[i] if(ub[i]<fu[i-1] or cl[i-1]>fu[i-1])else fu[i-1]
        fl[i]=db[i] if(db[i]>fl[i-1] or cl[i-1]<fl[i-1])else fl[i-1]
    t=np.ones(n,int)
    for i in range(1,n): t[i]=(-1 if cl[i]<fl[i] else 1) if t[i-1]==1 else (1 if cl[i]>fu[i] else -1)
    return r,t

def test_tf(ohlc,rule):
    r,t=supertrend(ohlc,rule,ST_PERIOD,ST_MULT)
    if len(r)<3: return None
    cl=r["close"].to_numpy(); idx=r.index
    day=pd.Series(idx).dt.date.to_numpy(); tm=pd.Series(idx).dt.time.to_numpy()
    n=len(r); trades=[]; i=1
    while i<n:
        if t[i]!=t[i-1] and TRADE_START<=tm[i]<=TRADE_END:
            d=t[i]; entry=cl[i]; j=i+1
            while j<n and t[j]==d and day[j]==day[i] and tm[j]<=TRADE_END: j+=1
            ex=min(j,n-1)
            if day[ex]!=day[i]: ex=j-1
            trades.append((cl[ex]-entry)*d); i=j
        else: i+=1
    return np.array(trades) if trades else None

def main():
    print(f"\nFetching {LOOKBACK_DAYS}d of {BASE_RES}-min data in {CHUNK_DAYS}d chunks...")
    ohlc=fetch_long()
    span=f"{ohlc.index[0].date()} -> {ohlc.index[-1].date()}"
    days=pd.Series(ohlc.index).dt.date.nunique()
    print(f"\nLoaded {len(ohlc):,} 1-min bars | {span} | {days} trading days")
    print(f"Supertrend({ST_PERIOD},{ST_MULT}) flip-to-flip, intraday\n")
    hdr=f"{'TF':>6} {'trades':>7} {'win%':>6} {'avgGross':>9} "+" ".join(f"{'net@'+format(c,'g'):>8}" for c in COSTS)+f" {'totNet@5':>9}"
    print(hdr); print("-"*len(hdr))
    for tf in TIMEFRAMES:
        a=test_tf(ohlc,tf)
        if a is None: print(f"{tf:>6}  (no trades)"); continue
        g=a.mean(); nets=" ".join(f"{g-c:>8.2f}" for c in COSTS)
        print(f"{tf:>6} {len(a):>7} {100*(a>0).mean():>5.1f}% {g:>9.2f} {nets} {(a-5).sum():>9.0f}")
    print("\nWith hundreds of trades: if net@5 stays clearly positive on 10/15/30min,")
    print("the edge is real -> next = futures/option P&L test + parameter robustness.")
    print("If it collapses vs the 30-day run, the earlier result was small-sample luck.")

if __name__=="__main__":
    main()
