"""
Signal-edge diagnostic WITH causal trend filters (self-contained, no look-ahead).

For each causal filter, keeps only signals whose direction matches the filter's
trend (read from the previous CLOSED higher-TF bar -> no look-ahead / no delay),
and measures the per-trade index move + net-of-cost expectancy at several
horizons. Answers: can ANY trend filter lift the gross move above the ~5-pt
cost floor?  Run where fyers_appid.txt / fyers_token.txt live:
    python VAM_SignalEdgeTest_Filters.py
"""
import os, warnings, urllib3
from datetime import datetime, timedelta, time as dtime
import numpy as np, pandas as pd, pytz, requests

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
warnings.filterwarnings("ignore")
IST = pytz.timezone("Asia/Kolkata")

UNDERLYING="NSE:NIFTY50-INDEX"; TIMEFRAME="10S"; HIST_DAYS=30
SESSION_START,SESSION_END=dtime(9,20),dtime(15,30)
TRADE_START,TRADE_END=dtime(9,30),dtime(15,0)
VAM_BUY,VAM_SELL=0.10,-0.10
LENS=[8,10,14,20,28]; MULTS=[1.05,1.10,1.15,1.20,1.25]
BANDS=[1.15,1.20,1.25,1.30,1.35]; SMOO=[2,2,3,3,4]
HORIZONS=[3,6,12,18,30]          # 30s,1,2,3,5 min
COST=5.0                          # realistic NIFTY futures round-trip (pts)

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
    bl=((src>up.shift(1))&(src.shift(1)<=up.shift(2))).to_numpy()
    bs=((src<lo.shift(1))&(src.shift(1)>=lo.shift(2))).to_numpy()
    side=np.ones(n,int)
    for i in range(1,n): side[i]=1 if bl[i] else (-1 if bs[i] else side[i-1])
    return side
def vam_signals(df):
    sc=np.mean([r1_side(df,LENS[k],MULTS[k],BANDS[k],SMOO[k]) for k in range(5)],axis=0)/5
    df["buy"]=(pd.Series(sc).shift(1).to_numpy()<=VAM_BUY)&(sc>VAM_BUY)
    df["sell"]=(pd.Series(sc).shift(1).to_numpy()>=VAM_SELL)&(sc<VAM_SELL)
    return df

# ---- causal trend builders (shifted to previous CLOSED bar) ----
def _shift(r,rule):
    r=r.reset_index(); r["timestamp"]=r["timestamp"]+pd.tseries.frequencies.to_offset(rule)
    return r[["timestamp","trend"]]
def htf_ema(ohlc,rule,elen=6):
    r=ohlc.resample(rule).agg({"open":"first","high":"max","low":"min","close":"last","volume":"sum"}).dropna()
    e=r["close"].ewm(span=elen,adjust=False).mean()
    r["trend"]=np.where(r["close"]>e,"up",np.where(r["close"]<e,"down","flat"));return _shift(r,rule)
def supertrend(ohlc,rule,period=10,mult=3.0):
    r=ohlc.resample(rule).agg({"open":"first","high":"max","low":"min","close":"last","volume":"sum"}).dropna()
    h=r["high"].to_numpy();l=r["low"].to_numpy();cl=r["close"].to_numpy();n=len(r)
    pc=np.empty(n);pc[0]=cl[0];pc[1:]=cl[:-1]
    trv=np.maximum(h-l,np.maximum(np.abs(h-pc),np.abs(l-pc)))
    atr=pd.Series(trv).ewm(alpha=1/period,adjust=False).mean().to_numpy()
    hl2=(h+l)/2;ub=hl2+mult*atr;db=hl2-mult*atr;fu=np.empty(n);fl=np.empty(n);fu[0]=ub[0];fl[0]=db[0]
    for i in range(1,n):
        fu[i]=ub[i] if(ub[i]<fu[i-1] or cl[i-1]>fu[i-1])else fu[i-1]
        fl[i]=db[i] if(db[i]>fl[i-1] or cl[i-1]<fl[i-1])else fl[i-1]
    t=np.ones(n,int)
    for i in range(1,n): t[i]=(-1 if cl[i]<fl[i] else 1) if t[i-1]==1 else (1 if cl[i]>fu[i] else -1)
    r["trend"]=np.where(t==1,"up","down");return _shift(r,rule)
def emaslope_adx(ohlc,rule,elen=20,adxlen=14,adxmin=20.0):
    r=ohlc.resample(rule).agg({"open":"first","high":"max","low":"min","close":"last","volume":"sum"}).dropna()
    h=r["high"].to_numpy();l=r["low"].to_numpy();cl=r["close"].to_numpy();n=len(r)
    rma=lambda x: pd.Series(x).ewm(alpha=1/adxlen,adjust=False).mean().to_numpy()
    e=pd.Series(cl).ewm(span=elen,adjust=False).mean().to_numpy();slope=np.zeros(n);slope[1:]=np.sign(e[1:]-e[:-1])
    pc=np.empty(n);pc[0]=cl[0];pc[1:]=cl[:-1];trv=np.maximum(h-l,np.maximum(np.abs(h-pc),np.abs(l-pc)))
    um=np.zeros(n);dm=np.zeros(n);um[1:]=h[1:]-h[:-1];dm[1:]=l[:-1]-l[1:]
    pdm=np.where((um>dm)&(um>0),um,0.0);mdm=np.where((dm>um)&(dm>0),dm,0.0);atr=rma(trv)
    with np.errstate(divide="ignore",invalid="ignore"):
        pdi=100*rma(pdm)/atr;mdi=100*rma(mdm)/atr;dx=100*np.abs(pdi-mdi)/(pdi+mdi)
    adx=rma(np.nan_to_num(dx));strong=adx>=adxmin
    r["trend"]=np.where(strong&(slope>0),"up",np.where(strong&(slope<0),"down","flat"));return _shift(r,rule)

def fetch_index():
    app=open("fyers_appid.txt").read().strip();tok=open("fyers_token.txt").read().strip()
    auth=tok if ":" in tok else f"{app}:{tok}"
    s=requests.Session();s.verify=False;s.headers.update({"Authorization":auth})
    if s.get("https://api-t1.fyers.in/api/v3/profile").json().get("s")!="ok": raise SystemExit("auth failed")
    end=datetime.now(IST);start=end-timedelta(days=HIST_DAYS)
    r=s.get("https://api-t1.fyers.in/data/history",params={"symbol":UNDERLYING,"resolution":TIMEFRAME,
        "date_format":"1","range_from":start.strftime("%Y-%m-%d"),"range_to":end.strftime("%Y-%m-%d"),"cont_flag":"0"})
    c=r.json().get("candles",[]);
    if not c: raise SystemExit("no data")
    df=pd.DataFrame(c,columns=["timestamp","open","high","low","close","volume"])
    df["timestamp"]=pd.to_datetime(df["timestamp"],unit="s",utc=True).dt.tz_convert(IST)
    df=df.sort_values("timestamp").reset_index(drop=True).set_index("timestamp")
    m=df.index.indexer_between_time(SESSION_START,SESSION_END);return df.iloc[m].reset_index()

def trend_on_bars(df,tdf):
    mg=pd.merge_asof(df[["timestamp"]].sort_values("timestamp"),tdf.sort_values("timestamp"),
                     on="timestamp",direction="backward")
    return mg["trend"].to_numpy()

def main():
    df=vam_signals(fetch_index())
    ohlc=df.set_index("timestamp")[["open","high","low","close","volume"]]
    ts=pd.to_datetime(df["timestamp"]);close=df["close"].to_numpy(float);day=ts.dt.date.to_numpy();n=len(df)
    inwin=((ts.dt.time>=TRADE_START)&(ts.dt.time<=TRADE_END)).to_numpy()
    buy=df["buy"].to_numpy();sell=df["sell"].to_numpy()
    base=[(i,+1) for i in range(n) if buy[i] and inwin[i]]+[(i,-1) for i in range(n) if sell[i] and inwin[i]]

    filters={"NONE (raw)":None,
             "HTF-EMA 5min":htf_ema(ohlc,"5min"),
             "HTF-EMA 15min":htf_ema(ohlc,"15min"),
             "Supertrend 1min":supertrend(ohlc,"1min"),
             "EMAslope+ADX 1m":emaslope_adx(ohlc,"1min")}
    trends={k:(None if v is None else trend_on_bars(df,v)) for k,v in filters.items()}

    print(f"\nData {day[0]} -> {day[-1]} | {n} bars | round-trip cost assumed = {COST} pts")
    print("gross = mean favorable move/trade (pts). net = gross - cost.\n")
    for name,tr_arr in trends.items():
        if tr_arr is None:
            sig=base
        else:
            sig=[(i,d) for (i,d) in base if (d>0 and tr_arr[i]=="up") or (d<0 and tr_arr[i]=="down")]
        print(f"== {name}  (aligned signals: {len(sig)}) ==")
        print(f"   {'horizon':>7} {'n':>6} {'hit%':>6} {'gross':>7} {'net@'+format(COST,'g'):>8}")
        for H in HORIZONS:
            mv=[]; hit=0
            for i,d in sig:
                j=i+H
                if j>=n or day[j]!=day[i]: continue
                fav=(close[j]-close[i])*d; mv.append(fav); hit+=fav>0
            if not mv: continue
            g=float(np.mean(mv))
            print(f"   {H*10/60:>5.1f}m {len(mv):>6} {100*hit/len(mv):>5.1f}% {g:>7.2f} {g-COST:>8.2f}")
        print()
    print("If 'gross' stays ~1-2 pts and net stays deeply negative for EVERY filter,")
    print("no trend timeframe can lift the edge over the cost floor -> retire the signal.")

if __name__=="__main__":
    main()
