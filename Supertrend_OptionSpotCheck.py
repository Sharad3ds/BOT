"""
Supertrend 10-min -> OPTION P&L spot-check on the recent window (self-contained).
Validated signal = 10-min Supertrend(10,3.0) flips on the index. For each flip
trade it buys the ATM (and, separately, 1-strike-ITM) weekly option for the
correct per-signal expiry, using REAL option prices, and reports option P&L.
Limited to ~LOOKBACK_DAYS because expired weekly-option history is sparse.
Run where fyers_appid.txt / fyers_token.txt live:
    python Supertrend_OptionSpotCheck.py
"""
import time, warnings, urllib3, calendar
from datetime import datetime, timedelta, time as dtime
import numpy as np, pandas as pd, pytz, requests
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning); warnings.filterwarnings("ignore")
IST=pytz.timezone("Asia/Kolkata")

UNDERLYING="NSE:NIFTY50-INDEX"; LOOKBACK_DAYS=15
SESSION_START,SESSION_END=dtime(9,20),dtime(15,30); TRADE_START,TRADE_END=dtime(9,30),dtime(15,0)
RULE="10min"; ST_PERIOD=10; ST_MULT=3.0; STRIKE_STEP=100
OPT_COST_PTS=1.5      # option round-trip spread+fees, in premium points

def _sess():
    app=open("fyers_appid.txt").read().strip();tok=open("fyers_token.txt").read().strip()
    auth=tok if ":" in tok else f"{app}:{tok}"
    s=requests.Session();s.verify=False;s.headers.update({"Authorization":auth})
    if s.get("https://api-t1.fyers.in/api/v3/profile").json().get("s")!="ok": raise SystemExit("auth failed")
    return s
S=None
def hist(symbol,days,res="1"):
    end=datetime.now(IST);start=end-timedelta(days=days)
    r=S.get("https://api-t1.fyers.in/data/history",params={"symbol":symbol,"resolution":res,"date_format":"1",
        "range_from":start.strftime("%Y-%m-%d"),"range_to":end.strftime("%Y-%m-%d"),"cont_flag":"0"})
    try:c=r.json().get("candles",[])
    except Exception:c=[]
    if not c: return pd.DataFrame()
    df=pd.DataFrame(c,columns=["timestamp","open","high","low","close","volume"])
    df["timestamp"]=pd.to_datetime(df["timestamp"],unit="s",utc=True).dt.tz_convert(IST)
    return df.drop_duplicates("timestamp").sort_values("timestamp").set_index("timestamp")

def weekly_expiry_for(d):
    x=datetime(d.year,d.month,d.day)
    while x.weekday()!=1: x+=timedelta(days=1)   # Tuesday
    return x
def monthly_expiry(y,m):
    last=calendar.monthrange(y,m)[1]; x=datetime(y,m,last)
    while x.weekday()!=1: x-=timedelta(days=1)
    return x
def expiry_code(dt):
    yy=dt.year%100; me=monthly_expiry(dt.year,dt.month)
    if dt.date()==me.date(): return f"{yy:02d}{dt.strftime('%b').upper()}"
    mc=str(dt.month) if dt.month<=9 else {10:'O',11:'N',12:'D'}[dt.month]
    return f"{yy:02d}{mc}{dt.day:02d}"

def st_trend(r,period,mult):
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
    return t

def flip_trades(idxbars):
    r=idxbars.resample(RULE).agg({"open":"first","high":"max","low":"min","close":"last","volume":"sum"}).dropna()
    t=st_trend(r,ST_PERIOD,ST_MULT); cl=r["close"].to_numpy(); idx=r.index
    day=pd.Series(idx).dt.date.to_numpy(); tm=pd.Series(idx).dt.time.to_numpy(); n=len(r); out=[]; i=1
    while i<n:
        if t[i]!=t[i-1] and TRADE_START<=tm[i]<=TRADE_END:
            d=t[i]; j=i+1
            while j<n and t[j]==d and day[j]==day[i] and tm[j]<=TRADE_END: j+=1
            ex=min(j,n-1); ex=ex-1 if day[ex]!=day[i] else ex
            out.append((idx[i],idx[ex],d,cl[i],cl[ex])); i=j
        else: i+=1
    return out

def opt_symbol(entry_dt, spot, direction, itm=False):
    code=expiry_code(weekly_expiry_for(entry_dt))
    atm=int(round(spot/STRIKE_STEP)*STRIKE_STEP)
    if direction>0:  # BUY -> CE ; ITM = lower strike
        strike=atm-STRIKE_STEP if itm else atm; opt="CE"
    else:            # SELL -> PE ; ITM = higher strike
        strike=atm+STRIKE_STEP if itm else atm; opt="PE"
    return f"NSE:NIFTY{code}{strike:05d}{opt}"

def price_at(odf, ts):
    if odf.empty: return None
    pos=odf.index.searchsorted(ts, side="left")
    if pos>=len(odf): return None
    # require the bar to be within a few minutes of ts
    if abs((odf.index[pos]-ts).total_seconds())>600: return None
    return float(odf.iloc[pos]["open"])

def evaluate(trades, itm, cache):
    rows=[]
    for et,xt,d,ei,xi in trades:
        sym=opt_symbol(et, ei, d, itm)
        if sym not in cache: cache[sym]=hist(sym, LOOKBACK_DAYS+3); time.sleep(0.35)
        odf=cache[sym]
        pe=price_at(odf,et); px=price_at(odf,xt)
        if pe is None or px is None or pe<=0: continue
        pnl=(px-pe)-OPT_COST_PTS
        rows.append((et,sym,pe,px,pnl,pnl/pe*100))
    return rows

def report(rows,label):
    if not rows: print(f"\n{label}: no valid option trades (option history unavailable for the window)"); return
    p=np.array([r[4] for r in rows]); pct=np.array([r[5] for r in rows])
    print(f"\n== {label} == ({len(rows)} trades)")
    print(f"   win%={100*(p>0).mean():.1f}  avg P&L/trade={p.mean():.2f} pts ({pct.mean():.1f}% of premium)")
    print(f"   total={p.sum():.1f} option pts  median={np.median(p):.2f}  best={p.max():.1f}  worst={p.min():.1f}")

def main():
    global S; S=_sess()
    print(f"Fetching {LOOKBACK_DAYS}d index + option data (option cost {OPT_COST_PTS} pts round-trip)...")
    idxbars=hist(UNDERLYING, LOOKBACK_DAYS)
    if idxbars.empty: raise SystemExit("no index data")
    idxbars=idxbars.iloc[idxbars.index.indexer_between_time(SESSION_START,SESSION_END)]
    trades=flip_trades(idxbars)
    print(f"Index bars: {len(idxbars):,} | 10-min Supertrend flip trades: {len(trades)} "
          f"({idxbars.index[0].date()} -> {idxbars.index[-1].date()})")
    cache={}
    report(evaluate(trades, itm=False, cache=cache), "ATM options")
    report(evaluate(trades, itm=True,  cache=cache), "1-strike ITM options")
    print("\nThis is a SMALL recent-window spot-check (option history is sparse).")
    print("Compare avg P&L%/premium and win% to the futures result. A clean forward")
    print("dry-run is still the real option validation.")

if __name__=="__main__":
    main()
