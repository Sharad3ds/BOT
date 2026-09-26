"""
Signal-edge diagnostic with a VOLUME filter (self-contained, no look-ahead).
Compares RAW VAM signals vs VOLUME-SURGE signals (bar volume >= VOL_MULT x mean
of prior VOL_LOOKBACK bars) on index-direction hit-rate + net-of-cost expectancy.
First checks whether the underlying even HAS volume (index often = 0).
Run where fyers_appid.txt / fyers_token.txt live:  python VAM_SignalEdgeTest_Volume.py
"""
import warnings, urllib3
from datetime import datetime, timedelta, time as dtime
import numpy as np, pandas as pd, pytz, requests
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning); warnings.filterwarnings("ignore")
IST = pytz.timezone("Asia/Kolkata")

UNDERLYING="NSE:NIFTY50-INDEX"; TIMEFRAME="10S"; HIST_DAYS=30
SESSION_START,SESSION_END=dtime(9,20),dtime(15,30); TRADE_START,TRADE_END=dtime(9,30),dtime(15,0)
VAM_BUY,VAM_SELL=0.10,-0.10
LENS=[8,10,14,20,28]; MULTS=[1.05,1.10,1.15,1.20,1.25]; BANDS=[1.15,1.20,1.25,1.30,1.35]; SMOO=[2,2,3,3,4]
HORIZONS=[3,6,12,18,30]; COSTS=[0.5,1.0,2.0,5.0]
VOL_LOOKBACK=20; VOL_MULT=1.5

def ema(s,sp): return s.ewm(span=sp,adjust=False).mean()
def tr(df):
    hl=df["high"]-df["low"];hc=(df["high"]-df["close"].shift()).abs();lc=(df["low"]-df["close"].shift()).abs()
    return pd.Series(np.maximum(hl,np.maximum(hc,lc)),index=df.index)
def r1_side(df,L,rm,bm,sm):
    src=df["close"];r1=ema(ema(tr(df),L)*rm,sm);s=src.to_numpy();r=r1.to_numpy();n=len(df)
    c=np.empty(n);c[0]=s[0]
    for i in range(1,n):
        p=c[i-1];d=s[i]-p
        c[i]=p+(d-r[i]) if d>r[i] else (p+(d+r[i]) if d<-r[i] else p)
    center=pd.Series(c,index=df.index);up=ema(center+r1*bm,sm);lo=ema(center-r1*bm,sm)
    bl=((src>up.shift(1))&(src.shift(1)<=up.shift(2))).to_numpy();bs=((src<lo.shift(1))&(src.shift(1)>=lo.shift(2))).to_numpy()
    side=np.ones(n,int)
    for i in range(1,n): side[i]=1 if bl[i] else (-1 if bs[i] else side[i-1])
    return side
def vam_signals(df):
    sc=np.mean([r1_side(df,LENS[k],MULTS[k],BANDS[k],SMOO[k]) for k in range(5)],axis=0)/5
    df["buy"]=(pd.Series(sc).shift(1).to_numpy()<=VAM_BUY)&(sc>VAM_BUY)
    df["sell"]=(pd.Series(sc).shift(1).to_numpy()>=VAM_SELL)&(sc<VAM_SELL)
    return df
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
    m=df.index.indexer_between_time(SESSION_START,SESSION_END);return df.iloc[m].reset_index()

def report(label, sig, close, day, n):
    print(f"== {label}  (signals: {len(sig)}) ==")
    hdr=f"   {'horizon':>7} {'n':>6} {'hit%':>6} {'gross':>7} "+" ".join(f"{'net@'+format(c,'g'):>7}" for c in COSTS)
    print(hdr)
    for H in HORIZONS:
        mv=[];hit=0
        for i,d in sig:
            j=i+H
            if j>=n or day[j]!=day[i]: continue
            fav=(close[j]-close[i])*d; mv.append(fav); hit+=fav>0
        if not mv: 
            print(f"   {H*10/60:>5.1f}m   (no samples)"); continue
        g=float(np.mean(mv)); nets=" ".join(f"{g-c:>7.2f}" for c in COSTS)
        print(f"   {H*10/60:>5.1f}m {len(mv):>6} {100*hit/len(mv):>5.1f}% {g:>7.2f} {nets}")
    print()

def main():
    df=vam_signals(fetch_index())
    ts=pd.to_datetime(df["timestamp"]);close=df["close"].to_numpy(float);day=ts.dt.date.to_numpy();n=len(df)
    inwin=((ts.dt.time>=TRADE_START)&(ts.dt.time<=TRADE_END)).to_numpy()
    buy=df["buy"].to_numpy();sell=df["sell"].to_numpy()
    vol=df["volume"].to_numpy(float)

    # volume diagnostics
    nz=float((vol>0).mean())*100
    print(f"\nData: {day[0]} -> {day[-1]} | {n} bars")
    print(f"VOLUME check: total={vol.sum():,.0f} | bars with vol>0: {nz:.1f}%")
    if vol.sum()==0 or nz<1:
        print(">>> Underlying volume is ZERO (index has no volume). The volume filter")
        print(">>> is meaningless here — feed NIFTY FUTURES volume to test it properly.\n")

    vol_ma=pd.Series(vol).rolling(VOL_LOOKBACK).mean().shift(1).to_numpy()
    vol_ok=np.where(np.isnan(vol_ma),False,vol>=VOL_MULT*vol_ma)

    raw=[(i,+1) for i in range(n) if buy[i] and inwin[i]]+[(i,-1) for i in range(n) if sell[i] and inwin[i]]
    volf=[(i,d) for (i,d) in raw if vol_ok[i]]
    print(f"Raw signals: {len(raw)} | volume-surge (>= {VOL_MULT}x prior {VOL_LOOKBACK}-bar avg): {len(volf)}\n")
    print("gross = mean favorable move/trade (pts); net@c = gross - c.\n")
    report("RAW (no filter)", raw, close, day, n)
    report(f"VOLUME-surge filtered", volf, close, day, n)
    print("If volume-surge 'gross' isn't much larger than RAW (and net@5 still <0),")
    print("the volume filter adds nothing over the ~5-pt cost floor.")

if __name__=="__main__":
    main()
