"""
Supertrend 10-min futures P&L + drawdown (self-contained, causal, realistic).
Validated config: 10-min Supertrend(10,3.0). Enters/exits at the NEXT bar's
open after a flip (not the flip-bar close), applies slippage + round-trip
fees, converts to rupees, and reports equity curve / max drawdown / worst
losing streak / monthly P&L. Also shows ideal-vs-realistic degradation.
Run where fyers_appid.txt / fyers_token.txt live:
    python Supertrend_FuturesPnL.py
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
RULE="10min"; ST_PERIOD=10; ST_MULT=3.0
LOT=75                 # NIFTY futures lot size -> SET TO YOUR ACTUAL LOT
FEES_RT=5.0            # round-trip fees+taxes in index points (STT-dominated)
SLIP_PER_SIDE=0.5      # adverse fill per side, points

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
    return df.iloc[df.index.indexer_between_time(SESSION_START,SESSION_END)]

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

def run(ohlc, realistic):
    r=ohlc.resample(RULE).agg({"open":"first","high":"max","low":"min","close":"last","volume":"sum"}).dropna()
    t=st_trend(r,ST_PERIOD,ST_MULT)
    op=r["open"].to_numpy(); cl=r["close"].to_numpy(); idx=r.index
    day=pd.Series(idx).dt.date.to_numpy(); tm=pd.Series(idx).dt.time.to_numpy(); n=len(r)
    fric=(FEES_RT+2*SLIP_PER_SIDE) if realistic else 0.0
    trades=[]; i=1
    while i<n:
        if t[i]!=t[i-1] and TRADE_START<=tm[i]<=TRADE_END:
            d=t[i]
            # entry
            if realistic:
                ei=i+1
                if ei>=n or day[ei]!=day[i] or tm[ei]>TRADE_END: i+=1; continue
                entry=op[ei]
            else:
                ei=i; entry=cl[i]
            # find opposite flip
            j=i+1
            while j<n and t[j]==d and day[j]==day[i] and tm[j]<=TRADE_END: j+=1
            if realistic:
                if j<n and t[j]!=d and day[j]==day[i] and (j+1)<n and day[j+1]==day[i] and tm[j+1]<=TRADE_END:
                    exit_p=op[j+1]           # next bar after opposite flip
                else:
                    ex=min(j,n-1);  ex=ex-1 if day[ex]!=day[i] else ex
                    exit_p=cl[ex]            # EOD / last valid bar
            else:
                ex=min(j,n-1); ex=ex-1 if day[ex]!=day[i] else ex; exit_p=cl[ex]
            gross=(exit_p-entry)*d
            trades.append((idx[i], gross-fric))
            i=j
        else: i+=1
    return trades

def stats(trades, label):
    a=np.array([x for _,x in trades]); rup=a*LOT; n=len(a)
    eq=np.cumsum(rup); dd=eq-np.maximum.accumulate(eq); mdd=dd.min()
    # worst losing streak (count)
    st=cur=0
    for x in a: cur=cur+1 if x<0 else 0; st=max(st,cur)
    print(f"\n== {label} ==  ({n} trades, lot={LOT})")
    print(f"  win%={100*(a>0).mean():.1f}  avg/trade={a.mean():.2f} pts (Rs {a.mean()*LOT:,.0f})  median={np.median(a):.2f} pts")
    print(f"  total = {a.sum():.0f} pts  = Rs {rup.sum():,.0f}")
    print(f"  max drawdown = Rs {mdd:,.0f}   |  return/DD = {rup.sum()/abs(mdd):.2f}" if mdd<0 else "  no drawdown")
    print(f"  best={a.max():.0f} pts  worst={a.min():.0f} pts  longest losing streak={st} trades")
    return trades

def monthly(trades):
    mo=defaultdict(float)
    for ts,x in trades: mo[ts.strftime("%Y-%m")]+=x*LOT
    ks=sorted(mo); pos=sum(1 for k in ks if mo[k]>0)
    print(f"  months positive: {pos}/{len(ks)}")
    print("  " + "  ".join(f"{k[2:]}:{mo[k]/1000:+.0f}k" for k in ks))

def main():
    print(f"\nFetching {LOOKBACK_DAYS}d of {BASE_RES}-min data...")
    ohlc=fetch_long()
    print(f"Loaded {len(ohlc):,} bars | {ohlc.index[0].date()} -> {ohlc.index[-1].date()} | "
          f"{pd.Series(ohlc.index).dt.date.nunique()} days")
    print(f"Config: {RULE} Supertrend({ST_PERIOD},{ST_MULT}) | fees_rt={FEES_RT} slip/side={SLIP_PER_SIDE} pts | lot={LOT}")
    ideal=run(ohlc, realistic=False)
    real =run(ohlc, realistic=True)
    stats(ideal, "IDEAL (flip-close fills, ZERO cost)")
    stats(real,  "REALISTIC (next-bar-open fills + slippage + fees)")
    print("\nMonthly P&L (REALISTIC):"); monthly(real)
    print("\nCheck: is REALISTIC still clearly positive with a return/DD you can stomach?")
    print("If yes -> forward/paper-test before capital. If the realism gap wipes it, reconsider.")

if __name__=="__main__":
    main()
