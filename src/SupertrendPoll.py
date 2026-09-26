"""
Supertrend 10-min OPTION dry-run bot (forward validation of the validated edge).
Strategy: hold the option in the direction of the CURRENT closed 10-min
Supertrend(10,3.0); flip when the closed-bar trend flips. Causal (only closed
bars) so entry is immediate on the next poll after a bar closes -- no gate.
Buys 1-strike ITM options by default. DRY-RUN by default: logs simulated P&L
on live premiums, never sends real orders.
"""
import pandas as pd, numpy as np, requests, logging, os, time, json, warnings, calendar, pytz, urllib3
from datetime import datetime, timedelta, time as dtime
from typing import Optional
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
warnings.filterwarnings("ignore")
IST = pytz.timezone("Asia/Kolkata")
STATE_FILE = "supertrend_position_state.json"

def ist_time(*a): return datetime.now(IST).timetuple()
logging.Formatter.converter = ist_time
try:
    from logger_config import get_logger; logger = get_logger()
except Exception:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - STPoll - %(levelname)s - %(message)s",
                        datefmt="%Y-%m-%d %H:%M:%S"); logger = logging.getLogger("STPoll")

# ---------- expiry helpers ----------
def get_next_tuesday(now=None):
    now = now or datetime.now(IST)
    x = datetime(now.year, now.month, now.day)
    while x.weekday()!=1: x += timedelta(days=1)
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

# ---------- config ----------
class Config:
    UNDERLYING="NSE:NIFTY50-INDEX"; BASE_RES="1"; HIST_DAYS=15
    ST_RULE="10min"; ST_PERIOD=10; ST_MULT=3.0
    STRIKE_STEP=100; LOT=65; MONEYNESS="ITM"      # "ATM" or "ITM"
    SESSION_START=dtime(9,20); SESSION_END=dtime(15,30)
    TRADE_START=dtime(9,30); TRADE_END=dtime(15,0); EOD_EXIT_TIME=dtime(15,20)
    POLL_INTERVAL_SECS=20; DRY_RUN=True
    def __init__(self):
        self.EXPIRY_CODE=expiry_code(get_next_tuesday())
        logger.info(f"Weekly expiry code {self.EXPIRY_CODE} | ST({self.ST_PERIOD},{self.ST_MULT}) {self.ST_RULE} | "
                    f"moneyness={self.MONEYNESS} | DRY_RUN={self.DRY_RUN}")

# ---------- fyers ----------
class Fyers:
    T1="https://api-t1.fyers.in/api/v3"; T2="https://api-t1.fyers.in/data"
    def __init__(self, app, tok):
        auth = tok if ":" in tok else f"{app}:{tok}"
        self.s=requests.Session(); self.s.verify=False; self.s.headers.update({"Authorization":auth})
    def profile(self): return self.s.get(f"{self.T1}/profile").json()
    def history(self, symbol, res, d_from, d_to):
        r=self.s.get(f"{self.T2}/history", params={"symbol":symbol,"resolution":res,"date_format":"1",
            "range_from":d_from,"range_to":d_to,"cont_flag":"0"})
        try: return r.json()
        except Exception: return {}
    def quotes(self, symbol):
        try: return self.s.get(f"{self.T2}/quotes", params={"symbols":symbol}).json()
        except Exception: return {}
    def place_order(self, data):
        try: return self.s.post(f"{self.T1}/orders/sync", json=data).json()
        except Exception as e: return {"s":"error","message":str(e)}

# ---------- supertrend ----------
def supertrend_trend(r, period, mult):
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

# ---------- history cache ----------
class Hist:
    def __init__(self, fy, cfg): self.fy=fy; self.cfg=cfg; self._df=None; self._day=None
    def _fetch(self, days):
        end=datetime.now(IST); start=end-timedelta(days=days)
        resp=self.fy.history(self.cfg.UNDERLYING,self.cfg.BASE_RES,start.strftime("%Y-%m-%d"),end.strftime("%Y-%m-%d"))
        if resp.get("s")!="ok": logger.warning(f"history err: {resp}"); return pd.DataFrame()
        c=resp.get("candles",[])
        if not c: return pd.DataFrame()
        df=pd.DataFrame(c,columns=["timestamp","open","high","low","close","volume"])
        df["timestamp"]=pd.to_datetime(df["timestamp"],unit="s",utc=True).dt.tz_convert(IST)
        df=df.drop_duplicates("timestamp").sort_values("timestamp").set_index("timestamp")
        return df.iloc[df.index.indexer_between_time(self.cfg.SESSION_START,self.cfg.SESSION_END)]
    def get(self):
        today=datetime.now(IST).date()
        if self._df is None or self._day!=today:
            self._df=self._fetch(self.cfg.HIST_DAYS); self._day=today
            logger.info(f"[HIST] warmup {len(self._df)} bars"); return self._df
        fresh=self._fetch(1)
        if fresh.empty: return self._df
        base=self._df[pd.to_datetime(self._df.index).date != today]
        self._df=pd.concat([base,fresh]); self._df=self._df[~self._df.index.duplicated(keep="last")].sort_index()
        return self._df

# ---------- order manager (dry-run + live) ----------
class OM:
    def __init__(self, fy, cfg):
        self.fy=fy; self.cfg=cfg
        self.has=False; self.symbol=None; self.dir=None; self.entry=None; self.oid=None; self.entry_time=None
    def _save(self):
        try: json.dump({"has":self.has,"symbol":self.symbol,"dir":self.dir,"entry":self.entry,
                        "entry_time":self.entry_time.isoformat() if self.entry_time else None,
                        "saved":datetime.now(IST).isoformat()}, open(STATE_FILE+".tmp","w")); os.replace(STATE_FILE+".tmp",STATE_FILE)
        except Exception as e: logger.error(f"save state: {e}")
    def _clear(self):
        try:
            if os.path.exists(STATE_FILE): os.remove(STATE_FILE)
        except Exception: pass
    def restore(self):
        if not os.path.exists(STATE_FILE): return
        try:
            st=json.load(open(STATE_FILE))
            if datetime.fromisoformat(st["saved"]).date()!=datetime.now(IST).date(): self._clear(); return
            if st.get("has"):
                self.has=True; self.symbol=st["symbol"]; self.dir=st["dir"]; self.entry=st["entry"]
                self.entry_time=datetime.fromisoformat(st["entry_time"]) if st.get("entry_time") else None
            logger.info(f"[OM] restored {self.symbol} {self.dir} @ {self.entry}")
        except Exception as e: logger.error(f"restore: {e}")
    def open(self, symbol, direction, ltp):
        self.has=True; self.symbol=symbol; self.dir=direction; self.entry=float(ltp); self.entry_time=datetime.now(IST)
        if self.cfg.DRY_RUN:
            self.oid=f"DRY-{datetime.now(IST).strftime('%H%M%S')}"
            logger.info(f"[OM][DRY] BUY {symbol} ({direction}) @ {ltp:.2f}")
        else:
            resp=self.fy.place_order({"symbol":symbol,"qty":self.cfg.LOT,"type":2,"side":1,
                "productType":"INTRADAY","limitPrice":0,"stopPrice":0,"validity":"DAY","offlineOrder":False})
            self.oid=resp.get("id"); logger.info(f"[OM][LIVE] BUY {symbol} resp={resp}")
        self._save()
    def close(self, ltp, reason):
        pnl=(float(ltp)-self.entry)*self.cfg.LOT
        if not self.cfg.DRY_RUN:
            resp=self.fy.place_order({"symbol":self.symbol,"qty":self.cfg.LOT,"type":2,"side":-1,
                "productType":"INTRADAY","limitPrice":0,"stopPrice":0,"validity":"DAY","offlineOrder":False})
            logger.info(f"[OM][LIVE] SELL {self.symbol} resp={resp}")
        logger.info(f"[OM]{'[DRY]' if self.cfg.DRY_RUN else ''} EXIT {self.symbol} @ {ltp:.2f} reason={reason} "
                    f"entry={self.entry:.2f} PnL=Rs {pnl:,.0f}")
        rec={"symbol":self.symbol,"dir":self.dir,
             "entry_time":self.entry_time.strftime("%Y-%m-%d %H:%M:%S") if self.entry_time else "",
             "entry":round(self.entry,2),"exit":round(float(ltp),2),
             "reason":reason,"pnl_pts":round(float(ltp)-self.entry,2),"pnl_rs":round(pnl,2),
             "exit_time":datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S")}
        self.has=False; self.symbol=None; self.dir=None; self.entry=None; self.oid=None; self.entry_time=None; self._clear()
        return rec

# ---------- engine ----------
class Engine:
    def __init__(self):
        self.cfg=Config(); self.fy=self._init()
        self.hist=Hist(self.fy,self.cfg); self.om=OM(self.fy,self.cfg); self.om.restore()
        self.trades=[]; self._baseline=None
        logger.info(f"[ENG] ready. interval={self.cfg.POLL_INTERVAL_SECS}s")
        if self.cfg.DRY_RUN: logger.warning("*** DRY-RUN — no real orders ***")
    def _init(self):
        for f in ["fyers_appid.txt","fyers_token.txt"]:
            if not os.path.exists(f): raise FileNotFoundError(f)
        app=open("fyers_appid.txt").read().strip(); tok=open("fyers_token.txt").read().strip()
        fy=Fyers(app,tok); p=fy.profile()
        if p.get("s")!="ok": raise ConnectionError(f"auth failed: {p}")
        logger.info(f"[ENG] connected {p['data']['name']}"); return fy
    def _opt_symbol(self, spot, direction):
        atm=int(round(spot/self.cfg.STRIKE_STEP)*self.cfg.STRIKE_STEP); itm=self.cfg.MONEYNESS=="ITM"
        if direction=="BUY": strike=atm-self.cfg.STRIKE_STEP if itm else atm; typ="CE"
        else: strike=atm+self.cfg.STRIKE_STEP if itm else atm; typ="PE"
        return f"NSE:NIFTY{self.cfg.EXPIRY_CODE}{strike:05d}{typ}"
    def _ltp(self, symbol):
        r=self.fy.quotes(symbol)
        try:
            v=r["d"][0]["v"].get("lp")
            return float(v) if v is not None else None
        except Exception: return None
    def _closed_trend(self):
        df=self.hist.get()
        if df is None or df.empty: return None
        r=df.resample(self.cfg.ST_RULE).agg({"open":"first","high":"max","low":"min","close":"last","volume":"sum"}).dropna()
        if len(r)<3: return None
        now=datetime.now(IST); step=int(self.cfg.ST_RULE.replace("min",""))
        last_start=r.index[-1]
        if last_start.tzinfo is None: last_start=IST.localize(last_start)
        if last_start+timedelta(minutes=step) > now:      # drop forming bar
            r=r.iloc[:-1]
        if len(r)<3: return None
        t=supertrend_trend(r,self.cfg.ST_PERIOD,self.cfg.ST_MULT)
        return {"dir":"BUY" if t[-1]==1 else "SELL","spot":float(r["close"].iloc[-1]),"bar":r.index[-1]}
    def _poll(self):
        now=datetime.now(IST)
        # EOD
        eod=IST.localize(datetime.combine(now.date(), self.cfg.EOD_EXIT_TIME))
        if now>=eod and self.om.has:
            ltp=self._ltp(self.om.symbol) or self.om.entry
            self.trades.append(self.om.close(ltp,"eod_exit")); return
        if not (self.cfg.SESSION_START<=now.time()<=self.cfg.SESSION_END): return
        st=self._closed_trend()
        if st is None: return
        desired=st["dir"]
        # baseline on first sight: adopt current trend WITHOUT trading (wait for next real flip)
        if self._baseline is None and not self.om.has:
            self._baseline=desired; logger.info(f"[ENG] baseline trend={desired} (waiting for next flip)"); return
        held=self.om.dir if self.om.has else None
        if held==desired: return                          # already aligned, hold
        # trend differs from what we hold -> flip
        in_win = self.cfg.TRADE_START<=now.time()<=self.cfg.TRADE_END
        if self.om.has:                                   # close current first
            ltp=self._ltp(self.om.symbol) or self.om.entry
            self.trades.append(self.om.close(ltp,"flip_exit"))
        if in_win:
            sym=self._opt_symbol(st["spot"],desired); ltp=self._ltp(sym)
            if ltp is None: logger.warning(f"[ENG] no LTP {sym}, skip entry"); self._baseline=desired; return
            self.om.open(sym,desired,ltp); logger.info(f"[ENG] FLIP -> {desired} on bar {st['bar']} spot={st['spot']:.1f}")
        self._baseline=desired
    def run(self):
        logger.info("[ENG] starting Supertrend poll loop")
        try:
            while True:
                t0=time.monotonic()
                try: self._poll()
                except Exception as e: logger.error(f"poll err: {e}", exc_info=True)
                time.sleep(max(0.0, self.cfg.POLL_INTERVAL_SECS-(time.monotonic()-t0)))
        except KeyboardInterrupt: logger.info("[ENG] interrupted")
        finally: self._shutdown()
    def _shutdown(self):
        if self.om.has:
            ltp=self._ltp(self.om.symbol) or self.om.entry
            self.trades.append(self.om.close(ltp,"shutdown"))
        self._export()
        logger.info("[ENG] shutdown complete")
    def _export(self):
        if not self.trades: logger.info("[ENG] no trades to export"); return
        try:
            df=pd.DataFrame(self.trades); df["cum_pnl_rs"]=df["pnl_rs"].cumsum()
            d=datetime.now(IST).strftime("%Y%m%d")
            path=os.path.join(os.path.dirname(os.path.abspath(__file__)),"logs"); os.makedirs(path,exist_ok=True)
            fp=os.path.join(path,f"supertrend_trades_{d}.xlsx"); df.to_excel(fp,index=False)
            net=df["pnl_rs"].sum(); wins=(df["pnl_rs"]>0).sum()
            logger.info(f"[ENG] exported {len(df)} trades -> {fp} | net Rs {net:,.0f} | wins {wins}/{len(df)}")
        except Exception as e: logger.error(f"export: {e}")

if __name__=="__main__":
    Engine().run()
