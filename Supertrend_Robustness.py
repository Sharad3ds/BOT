"""
Supertrend robustness + distribution sweep (self-contained, causal).
Fetches ~1yr of 1-min index data ONCE, then:
  1) sweeps ST_PERIOD x ST_MULT across timeframes -> net@5 grid (is the edge
     robust to parameters, or only one lucky cell?)
  2) for focus configs, reports trade distribution + monthly consistency
     (is the profit broad, or a few trend-day outliers / one lucky quarter?)
Run where fyers_appid.txt / fyers_token.txt live:
    python Supertrend_Robustness.py
"""
import time, warnings, urllib3
from collections import defaultdict
from datetime import datetime, timedelta, time as dtime
import numpy as np, pandas as pd, pytz, requests
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning); warnings.filterwarnings("ignore")
IST=pytz.timezone("Asia/Kolkata")

UNDERLYING="NSE:NIFTY50-INDEX"; BASE_RES="1"; LOOKBACK_DAYS=365; CHUNK_DAYS=90
SESSION_START,SESSION_END=dtime(9,20),dtime(15,30)
TRADE_START,TRADE_END=dtime(9,30),dtime(15,0)
COST=5.0
PERIODS=[7,10,14]; MULTS=[2.0,3.0,4.0]; TFS=["5min","10min","15min"]
FOCUS=[("10min",10,3.0),("15min",10,3.0)]   # detailed distribution for these

def _sess():
    app=open("fyers_appid.txt").read().strip();tok=open("fyers_token.txt").read().strip()
    auth=tok if ":" in tok else f"{app}:{tok}"
    s=requests.Session();s.verify=False;s.headers.update({"Authorization":auth})
    if s.get("https://api-t1.fyers.in/api/v3/profile").json().get("s")!="ok": raise SystemExit("auth failed")
    return s
def fetch_long():
    s=_sess(); end=datetime.now(IST); start=end-timedelta(days=LOOKBACK_DAYS); frames=[]; cur=start
    while cur<end:
        ce=min(cur+timedelta(days=CHUNK_DAYS),end)
        r=s.get("https://api-t1.fyers.in/data/history",params={"symbol":UNDERLYING,"resolution":BASE_RES,
            "date_format":"1","range_from":cur.strftime("%Y-%m-%d"),"range_to":ce.strftime("%Y-%m-%d"),"cont_flag":"0"})
        try:c=r.json().get("candles",[])
        except Exception:c=[]
        if c: frames.append(pd.DataFrame(c,columns=["timestamp","open","high","low","close","volume"]))
        cur=ce+timedelta(days=1); time.sleep(0.4)
    if not frames: raise SystemExit("no data")
    df=pd.concat(frames,ignore_index=True)
    df["timestamp"]=pd.to_datetime(df["timestamp"],unit="s",utc=True).dt.tz_convert(IST)
    df=df.drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True).set_index("timestamp")
    m=df.index.indexer_between_time(SESSION_START,SESSION_END); return df.iloc[m]

def st_trend(r,period,mult):
    h=r["high"].to_numpy();l=r["low"].to_numpy();cl=r["close"].to_numpy();n=len(r)
    if n<2: return np.ones(n,int)
    pc=np.empty(n);pc[0]=cl[0];pc[1:]=cl[:-1]
    trv=np.maximum(h-l,np.maximum(np.abs(h-pc),np.abs(l-pc)))
    atr=pd.Series(trv).ewm(alpha=1/period,adjust=False).mean().to_numpy()
    hl2=(h+l)/2;ub=hl2+mult*atr;db=hl2-mult*atr;fu=np.empty(n);fl=np.empty(n);fu[0]=ub[0];fl[0]=db[0]
    for i in range(1,n):
        fu[i]=ub[i] if(ub[i]<fu[i-1] or cl[i-1]>fu[i-1])else fu[i-1]
        fl[i]=db[i] if(db[i]>fl[i-1] or cl[i-1]<fl[i-1])else fl[i-1]
    t=np.ones(n,int)
    for i in range(1,n): t[i]=(-1 if cl[i]<fl[i] else 1) if t[i-1]==1 else (1 if cl[i]>fu[i] else -1)
    return t

_resample_cache={}
def resampled(ohlc,rule):
    if rule not in _resample_cache:
        _resample_cache[rule]=ohlc.resample(rule).agg({"open":"first","high":"max","low":"min","close":"last","volume":"sum"}).dropna()
    return _resample_cache[rule]

def trades_for(ohlc,rule,period,mult):
    r=resampled(ohlc,rule); t=st_trend(r,period,mult)
    if len(r)<3: return []
    cl=r["close"].to_numpy(); idx=r.index
    day=pd.Series(idx).dt.date.to_numpy(); tm=pd.Series(idx).dt.time.to_numpy()
    n=len(r); out=[]; i=1
    while i<n:
        if t[i]!=t[i-1] and TRADE_START<=tm[i]<=TRADE_END:
            d=t[i]; entry=cl[i]; j=i+1
            while j<n and t[j]==d and day[j]==day[i] and tm[j]<=TRADE_END: j+=1
            ex=min(j,n-1)
            if day[ex]!=day[i]: ex=j-1
            out.append((idx[i],(cl[ex]-entry)*d)); i=j
        else: i+=1
    return out

def main():
    print(f"\nFetching {LOOKBACK_DAYS}d of {BASE_RES}-min data...")
    ohlc=fetch_long()
    print(f"Loaded {len(ohlc):,} bars | {ohlc.index[0].date()} -> {ohlc.index[-1].date()} | "
          f"{pd.Series(ohlc.index).dt.date.nunique()} days")

    print(f"\n=== ROBUSTNESS: net@{COST:g} per trade  (n trades)  ===")
    for tf in TFS:
        print(f"\nTF={tf}")
        print("  per\\mult " + "".join(f"{m:>13.1f}" for m in MULTS))
        for p in PERIODS:
            cells=[]
            for m in MULTS:
                tr=trades_for(ohlc,tf,p,m)
                if tr:
                    a=np.array([x for _,x in tr]); cells.append(f"{a.mean()-COST:+6.1f}/{len(a):<4}")
                else: cells.append("   -   ")
            print(f"  {p:>6}   " + "".join(f"{c:>13}" for c in cells))

    print(f"\n=== DISTRIBUTION + MONTHLY (net@{COST:g}) for focus configs ===")
    for tf,p,m in FOCUS:
        tr=trades_for(ohlc,tf,p,m)
        if not tr: print(f"\n{tf} ST({p},{m}): no trades"); continue
        a=np.array([x for _,x in tr]); net=a-COST
        top5=np.sort(a)[-5:].sum()-5*COST
        print(f"\n-- {tf} ST({p},{m}) | {len(a)} trades --")
        print(f"   win%={100*(a>0).mean():.1f}  mean={a.mean():.2f}  median={np.median(a):.2f}  "
              f"best={a.max():.1f}  worst={a.min():.1f}")
        print(f"   total net@{COST:g} = {net.sum():.0f} pts | excl. top-5 winners = {net.sum()-top5:.0f} pts")
        mo=defaultdict(float); moc=defaultdict(int)
        for ts,x in tr: k=ts.strftime("%Y-%m"); mo[k]+=x-COST; moc[k]+=1
        months=sorted(mo); pos=sum(1 for k in months if mo[k]>0)
        print(f"   months positive: {pos}/{len(months)}")
        print("   " + "  ".join(f"{k[2:]}:{mo[k]:+.0f}" for k in months))
    print("\nVerdict: edge is REAL if net@5 stays positive across most param cells")
    print("AND most months are positive AND removing top-5 winners keeps it positive.")

if __name__=="__main__":
    main()
