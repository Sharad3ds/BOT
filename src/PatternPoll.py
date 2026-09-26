"""
Sequential 3-candle REVERSAL option dry-run bot (near-9-EMA), user's rules.

SELL setup (short via ITM PE):
  C1 = GREEN  (its OPEN is the trigger level)
  C2 = RED
  C3 = RED, CLOSES BELOW C1's open, is really close to the 9-EMA, and is a
       "healthy" candle (body >= HEALTHY_BODY_FRAC of its range).
  If C3 doesn't satisfy this -> setup expires, no trade. (CONFIRM_END=3)

BUY setup (long via ITM CE) — exact MIRROR of the SELL rules:
  C1 = RED, C2 = GREEN, C3 = GREEN closing ABOVE C1's open, really close to the
  9-EMA, and healthy. (Enable/disable with ENABLE_BUY.)

Causal: only CLOSED bars are used (the forming bar is dropped). Each poll re-checks
the last closed bar as the confirmation candle, so the "wait for the next candle"
is handled naturally as bars close.

Risk: primary stop is a STRUCTURAL stop on the underlying index (C1-based:
PE=C1 open, CE=C1 close); optional premium take-profit; opposite-setup flip; and
EOD exit. DRY-RUN by default: logs simulated P&L on live premiums, never sends
real orders.

Runs ALONGSIDE the VAM / Supertrend bots, SHARING the same Fyers token files
(fyers_appid.txt / fyers_token.txt) in this dir. Read-only + dry-run, so multiple
clients on one token is safe.
"""
import pandas as pd, numpy as np, requests, logging, os, time, json, warnings, calendar, pytz, urllib3
from datetime import datetime, timedelta, time as dtime
from typing import Optional
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
warnings.filterwarnings("ignore")
IST = pytz.timezone("Asia/Kolkata")
STATE_FILE = "pattern_position_state.json"

def ist_time(*a): return datetime.now(IST).timetuple()
logging.Formatter.converter = ist_time
try:
    from logger_config import get_logger; logger = get_logger()
except Exception:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - PATPoll - %(levelname)s - %(message)s",
                        datefmt="%Y-%m-%d %H:%M:%S"); logger = logging.getLogger("PATPoll")

# ---------- expiry helpers (identical convention to the other bots) ----------
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
    TF_RULE="1min"                                  # timeframe the candles + EMA run on
    EMA_LEN=50                                       # the EMA the confirmation candle must cross AND close past
    USE_EMA_CROSS_FILTER=False                        # entry candle must cross the EMA and finish on the right side
    HEALTHY_BODY_FRAC=0.8                            # C3 "healthy": body >= this fraction of its high-low range
    CONFIRM_START=3                                  # confirmation may be candle 3...
    CONFIRM_END=3                                    # ...up to candle 3 (C3 is the entry candle)
    CONFIRM_MUST_BE_COLORED=True                     # SELL conf must be red / BUY conf must be green
    ENABLE_BUY=True                                  # mirror BUY side (set False for SELL-only)
    # ---- fast entry: skip waiting for C3 if C2 is a big enough reversal ----
    USE_BIG_C2_FAST_ENTRY=True
    BIG_C2_BODY_MULT=1.05      # C2's body must be >= this many times C1's body (C1 must also be "healthy")
    # ---- volume filter: recent participation required behind the move ----
    USE_VOLUME_FILTER=True
    VOLUME_LOOKBACK=4          # check up to this many recent candles (entry candle inclusive)
    VOL_SMA_LEN=20             # volume moving-average length; a candle "spikes" if its volume > this SMA
    STRIKE_STEP=100; LOT=65; MONEYNESS="ITM"        # "ATM" or "ITM" (1-strike ITM default)
    SESSION_START=dtime(9,20); SESSION_END=dtime(15,30)
    SIGNAL_START_TIME=dtime(9,35)                    # no signal (normal or fast-entry) evaluated before this
    TRADE_START=dtime(9,30); TRADE_END=dtime(15,0); EOD_EXIT_TIME=dtime(15,20)
    POLL_INTERVAL_SECS=20; DRY_RUN=True

    # ---- structural stop on the UNDERLYING index (candle-1 based) ----
    USE_CANDLE_SL=True
    SL_REF_SELL="c1_high"     # PE buy (SELL setup): stop at C1's HIGH (farthest/widest point)
    SL_REF_BUY ="c1_low"      # CE buy (BUY  setup): stop at C1's LOW (farthest/widest point)
    #   each option: "c1_open" | "c1_close" | "c1_high" | "c1_low"

    # ---- risk on option premium (secondary / backstop) ----
    USE_PREMIUM_SL=False                            # premium stop; OFF because candle SL is primary
    USE_PREMIUM_TP=True                             # premium take-profit
    STOP_LOSS_PCT=25.0                              # premium SL % (only if USE_PREMIUM_SL)
    TAKE_PROFIT_PCT=40.0                            # premium TP % (only if USE_PREMIUM_TP)

    # ---- profit lock + trail on option premium ----
    # Two phases, driven by the BEST premium gain % seen since entry (so the
    # floor only ever rises, never falls):
    #   gain < PROFIT_LOCK_TRIGGER_PCT (12%)          -> no floor yet, rides freely
    #   PROFIT_LOCK_TRIGGER_PCT <= gain < TRAIL_START_PCT (12%-20%) -> floor locked flat at 12%
    #   gain >= TRAIL_START_PCT (20%+)                -> floor trails TRAIL_BEHIND_PCT (5%)
    #                                                     behind the best gain seen (e.g. best=25% -> floor=20%)
    # Never caps the upside -- still rides to TAKE_PROFIT_PCT if price keeps moving.
    USE_PROFIT_LOCK=True
    PROFIT_LOCK_TRIGGER_PCT=12.0   # arm point: floor locks in once gain reaches this
    PROFIT_LOCK_FLOOR_PCT=12.0     # flat floor value while armed but below TRAIL_START_PCT
    TRAIL_START_PCT=20.0           # once best gain reaches this, switch to trailing
    TRAIL_BEHIND_PCT=5.0           # once trailing, keep floor this far behind the best gain

    def __init__(self):
        self.EXPIRY_CODE=expiry_code(get_next_tuesday())
        logger.info(f"Weekly expiry code {self.EXPIRY_CODE} | seq {self.CONFIRM_START}-{self.CONFIRM_END} candle "
                    f"reversal on {self.TF_RULE} (EMA{self.EMA_LEN} cross+close filter={'on' if self.USE_EMA_CROSS_FILTER else 'off'}) "
                    f"healthy>={self.HEALTHY_BODY_FRAC} | "
                    f"vol={('spike>SMA'+str(self.VOL_SMA_LEN)+' in last '+str(self.VOLUME_LOOKBACK)) if self.USE_VOLUME_FILTER else 'off'} | "
                    f"fastC2={'on(>='+str(self.BIG_C2_BODY_MULT)+'x)' if self.USE_BIG_C2_FAST_ENTRY else 'off'} | "
                    f"BUY={'on' if self.ENABLE_BUY else 'off'} | moneyness={self.MONEYNESS} | "
                    f"signalsFrom={self.SIGNAL_START_TIME} | "
                    f"candleSL PE={self.SL_REF_SELL} CE={self.SL_REF_BUY} | "
                    f"premSL={'on' if self.USE_PREMIUM_SL else 'off'} "
                    f"premTP={str(self.TAKE_PROFIT_PCT)+'%' if self.USE_PREMIUM_TP else 'off'} "
                    f"profitLock={(str(self.PROFIT_LOCK_TRIGGER_PCT)+'%->floor'+str(self.PROFIT_LOCK_FLOOR_PCT)+'%,trail'+str(self.TRAIL_BEHIND_PCT)+'%after'+str(self.TRAIL_START_PCT)+'%') if self.USE_PROFIT_LOCK else 'off'} | "
                    f"DRY_RUN={self.DRY_RUN}")

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

# ---------- sequential candle-reversal detection ----------
def detect_signal(o, h, l, c, ema, vol, vol_sma, cfg):
    """
    o,h,l,c,ema,vol,vol_sma: numpy arrays of the resampled bars. CONFIRMATION = the LAST bar.
    Implements the user's SELL rules and the mirror BUY rules.
    Returns (direction, name, c1_idx, c2_idx) or None.
    """
    n=len(c)
    if n < 2: return None                   # minimum bars needed for the fast-entry path (C1,C2)
    i=n-1                                   # confirmation candle = last CLOSED bar

    def healthy(k):
        rng=h[k]-l[k]
        return rng>0 and abs(c[k]-o[k]) >= cfg.HEALTHY_BODY_FRAC*rng
    def body(k): return abs(c[k]-o[k])
    green=lambda k: c[k]>o[k]
    red  =lambda k: c[k]<o[k]

    def crossed_and_closed(k, direction):
        # EMA_LEN-EMA filter, re-expressed as a genuine crossover instead of a
        # fixed-point proximity band: the EMA must fall WITHIN this candle's
        # high-low range (i.e. price actually traded through/across it during
        # the bar) AND the candle must finish (close) on the expected side.
        #   BUY  -> crossed above and finished above the EMA
        #   SELL -> crossed below and finished below the EMA
        if not cfg.USE_EMA_CROSS_FILTER: return True
        crossed = l[k] <= ema[k] <= h[k]
        if not crossed: return False
        return c[k] > ema[k] if direction=="BUY" else c[k] < ema[k]

    def volume_ok(k):
        # At least ONE of the last VOLUME_LOOKBACK candles (ending at k, the
        # entry candle, inclusive) must show a volume spike -- i.e. that
        # bar's volume above its own VOL_SMA_LEN-period volume SMA. Doesn't
        # require the entry candle itself to be the spike, just that one
        # happened recently, confirming real participation behind the move.
        if not cfg.USE_VOLUME_FILTER: return True
        lo=max(0, k-(cfg.VOLUME_LOOKBACK-1))
        for j in range(lo, k+1):
            sma=vol_sma[j]
            if sma==sma and vol[j] > sma:   # sma==sma is a fast NaN check (NaN != NaN)
                return True
        return False

    # ---- Fast entry: C2 is a big reversal candle right after a healthy C1 --
    # skip waiting for C3 entirely and enter as soon as C2 closes. Still
    # requires C2's close to actually break C1's open (same requirement as
    # the normal path) because the candle-SL is set at c1_open/c1_close --
    # without the break, entry would start out already past its own stop.
    if cfg.USE_BIG_C2_FAST_ENTRY and i>=1:
        c1=i-1; c2=i
        b1=body(c1)
        if healthy(c1) and b1>0 and (body(c2)/b1)>=cfg.BIG_C2_BODY_MULT:
            if green(c1) and red(c2) and c[c2]<o[c1] and crossed_and_closed(c2,"SELL") and volume_ok(c2):
                return ("SELL", "sell_bigc2", c1, c2)
            if cfg.ENABLE_BUY and red(c1) and green(c2) and c[c2]>o[c1] and crossed_and_closed(c2,"BUY") and volume_ok(c2):
                return ("BUY", "buy_bigc2", c1, c2)

    # ---- SELL: C1 green, C2 red, later candle closes BELOW C1 open ----
    for cn in range(cfg.CONFIRM_START, cfg.CONFIRM_END+1):
        c1=i-(cn-1)
        if c1<0: break
        c2=c1+1
        if not (green(c1) and red(c2)):            continue   # C1 green, C2 red
        if not (c[i] < o[c1]):                     continue   # confirm closes below C1 open
        if cfg.CONFIRM_MUST_BE_COLORED and not red(i): continue
        # earlier candles (3rd/4th here) must NOT have already closed below C1 open,
        # else THEY would have been the confirmation (we take the FIRST one).
        if any(c[j] < o[c1] for j in range(c2+1, i)): continue
        if not healthy(i):                         continue   # C3 healthy body
        if not crossed_and_closed(i,"SELL"):       continue   # C3 crossed below EMA and closed below
        if not volume_ok(i):                       continue   # recent volume spike required
        return ("SELL", f"sell_break_c{cn}", c1, c2)

    # ---- BUY (mirror): C1 red, C2 green, later candle closes ABOVE C1 open ----
    if cfg.ENABLE_BUY:
        for cn in range(cfg.CONFIRM_START, cfg.CONFIRM_END+1):
            c1=i-(cn-1)
            if c1<0: break
            c2=c1+1
            if not (red(c1) and green(c2)):            continue
            if not (c[i] > o[c1]):                     continue
            if cfg.CONFIRM_MUST_BE_COLORED and not green(i): continue
            if any(c[j] > o[c1] for j in range(c2+1, i)): continue
            if not healthy(i):                         continue   # C3 healthy body
            if not crossed_and_closed(i,"BUY"):        continue   # C3 crossed above EMA and closed above
            if not volume_ok(i):                       continue   # recent volume spike required
            return ("BUY", f"buy_break_c{cn}", c1, c2)
    return None

def profit_lock_floor_pct(best_gain_pct, cfg):
    """
    Profit floor driven by the BEST premium gain % seen since entry (so it
    only ever rises, never falls):
      best_gain_pct < PROFIT_LOCK_TRIGGER_PCT           -> None (not armed yet)
      PROFIT_LOCK_TRIGGER_PCT <= best_gain_pct < TRAIL_START_PCT
                                                         -> flat floor at PROFIT_LOCK_FLOOR_PCT
      best_gain_pct >= TRAIL_START_PCT                  -> trailing floor,
                                                            TRAIL_BEHIND_PCT behind best_gain_pct
    e.g. with defaults (12/12/20/5): gain=15% -> floor=12%; gain=25% -> floor=20%.
    """
    if best_gain_pct < cfg.PROFIT_LOCK_TRIGGER_PCT:
        return None
    if best_gain_pct < cfg.TRAIL_START_PCT:
        return cfg.PROFIT_LOCK_FLOOR_PCT
    return best_gain_pct - cfg.TRAIL_BEHIND_PCT

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
        self.has=False; self.symbol=None; self.dir=None; self.entry=None; self.oid=None
        self.entry_time=None; self.pattern=None; self.sl_level=None
        self.best_gain_pct=0.0        # best premium gain % seen since entry (drives the staircase)
        self.profit_floor_pct=None    # current locked floor %, None until first armed
    def _save(self):
        try: json.dump({"has":self.has,"symbol":self.symbol,"dir":self.dir,"entry":self.entry,
                        "pattern":self.pattern,"sl_level":self.sl_level,
                        "best_gain_pct":self.best_gain_pct,"profit_floor_pct":self.profit_floor_pct,
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
                self.pattern=st.get("pattern"); self.sl_level=st.get("sl_level")
                self.best_gain_pct=st.get("best_gain_pct", 0.0); self.profit_floor_pct=st.get("profit_floor_pct")
                self.entry_time=datetime.fromisoformat(st["entry_time"]) if st.get("entry_time") else None
            logger.info(f"[OM] restored {self.symbol} {self.dir} @ {self.entry} ({self.pattern})")
        except Exception as e: logger.error(f"restore: {e}")
    def open(self, symbol, direction, ltp, pattern, sl_level=None):
        self.has=True; self.symbol=symbol; self.dir=direction; self.entry=float(ltp)
        self.entry_time=datetime.now(IST); self.pattern=pattern; self.sl_level=sl_level
        self.best_gain_pct=0.0; self.profit_floor_pct=None
        _sl=f" SL@{sl_level:.1f}" if sl_level is not None else ""
        if self.cfg.DRY_RUN:
            self.oid=f"DRY-{datetime.now(IST).strftime('%H%M%S')}"
            logger.info(f"[OM][DRY] BUY {symbol} ({direction}/{pattern}) @ {ltp:.2f}{_sl}")
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
        rec={"symbol":self.symbol,"dir":self.dir,"pattern":self.pattern,
             "sl_level":round(self.sl_level,2) if self.sl_level is not None else None,
             "entry_time":self.entry_time.strftime("%Y-%m-%d %H:%M:%S") if self.entry_time else "",
             "entry":round(self.entry,2),"exit":round(float(ltp),2),
             "reason":reason,"pnl_pts":round(float(ltp)-self.entry,2),"pnl_rs":round(pnl,2),
             "exit_time":datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S")}
        self.has=False; self.symbol=None; self.dir=None; self.entry=None; self.oid=None
        self.entry_time=None; self.pattern=None; self.sl_level=None
        self.best_gain_pct=0.0; self.profit_floor_pct=None; self._clear()
        return rec

# ---------- engine ----------
class Engine:
    def __init__(self):
        self.cfg=Config(); self.fy=self._init()
        self.hist=Hist(self.fy,self.cfg); self.om=OM(self.fy,self.cfg); self.om.restore()
        self.trades=[]; self._fired_bar=None          # timestamp of the bar we last acted on
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
    def _closed_bars(self):
        """Return the resampled TF frame with the still-forming bar dropped."""
        df=self.hist.get()
        if df is None or df.empty: return None
        r=df.resample(self.cfg.TF_RULE).agg({"open":"first","high":"max","low":"min",
            "close":"last","volume":"sum"}).dropna()
        if len(r)<self.cfg.CONFIRM_START: return None
        now=datetime.now(IST); step=int(self.cfg.TF_RULE.replace("min",""))
        last_start=r.index[-1]
        if last_start.tzinfo is None: last_start=IST.localize(last_start)
        if last_start+timedelta(minutes=step) > now:      # drop the forming bar
            r=r.iloc[:-1]
        return r if len(r)>=self.cfg.CONFIRM_START else None
    def _signal(self):
        """Detect the user's SELL/BUY sequence confirmed on the last CLOSED bar."""
        r=self._closed_bars()
        if r is None: return None
        ema=r["close"].ewm(span=self.cfg.EMA_LEN, adjust=False).mean().to_numpy()
        o=r["open"].to_numpy(); h=r["high"].to_numpy(); l=r["low"].to_numpy(); c=r["close"].to_numpy()
        vol=r["volume"].to_numpy()
        vol_sma=r["volume"].rolling(self.cfg.VOL_SMA_LEN, min_periods=self.cfg.VOL_SMA_LEN).mean().to_numpy()
        sig=detect_signal(o,h,l,c,ema,vol,vol_sma,self.cfg)
        if sig is None: return None
        direction,name,c1,c2=sig
        return {"dir":direction,"name":name,"spot":float(c[-1]),"bar":r.index[-1],
                "ema":float(ema[-1]),
                "c1_open":float(o[c1]),"c1_close":float(c[c1]),
                "c1_high":float(h[c1]),"c1_low":float(l[c1])}
    def _sl_level(self, st, direction):
        ref = self.cfg.SL_REF_SELL if direction=="SELL" else self.cfg.SL_REF_BUY
        return {"c1_open":st["c1_open"],"c1_close":st["c1_close"],
                "c1_high":st["c1_high"],"c1_low":st["c1_low"]}[ref]
    def _poll(self):
        now=datetime.now(IST)
        # ---- EOD forced exit ----
        eod=IST.localize(datetime.combine(now.date(), self.cfg.EOD_EXIT_TIME))
        if now>=eod and self.om.has:
            ltp=self._ltp(self.om.symbol) or self.om.entry
            self.trades.append(self.om.close(ltp,"eod_exit")); return
        if not (self.cfg.SESSION_START<=now.time()<=self.cfg.SESSION_END): return

        # ---- risk management while holding ----
        if self.om.has:
            # (a) structural candle stop on the UNDERLYING index (primary)
            if self.cfg.USE_CANDLE_SL and self.om.sl_level is not None:
                spot=self._ltp(self.cfg.UNDERLYING)
                if spot is not None:
                    hit = (self.om.dir=="SELL" and spot>=self.om.sl_level) or \
                          (self.om.dir=="BUY"  and spot<=self.om.sl_level)
                    if hit:
                        ltp=self._ltp(self.om.symbol) or self.om.entry
                        logger.info(f"[ENG] candle-SL hit: spot={spot:.1f} vs SL={self.om.sl_level:.1f} "
                                    f"({self.om.dir})")
                        self.trades.append(self.om.close(ltp,"candle_sl")); return
            # (b) premium SL / TP / profit-lock (secondary, on the option premium itself)
            if self.cfg.USE_PREMIUM_SL or self.cfg.USE_PREMIUM_TP or self.cfg.USE_PROFIT_LOCK:
                ltp=self._ltp(self.om.symbol)
                if ltp is not None:
                    if self.cfg.USE_PREMIUM_SL and ltp<=self.om.entry*(1-self.cfg.STOP_LOSS_PCT/100.0):
                        self.trades.append(self.om.close(ltp,"stop_loss")); return
                    if self.cfg.USE_PREMIUM_TP and ltp>=self.om.entry*(1+self.cfg.TAKE_PROFIT_PCT/100.0):
                        self.trades.append(self.om.close(ltp,"target")); return
                    if self.cfg.USE_PROFIT_LOCK:
                        gain_pct=(ltp-self.om.entry)/self.om.entry*100.0
                        if gain_pct>self.om.best_gain_pct: self.om.best_gain_pct=gain_pct
                        new_floor=profit_lock_floor_pct(self.om.best_gain_pct, self.cfg)
                        if new_floor is not None and (self.om.profit_floor_pct is None or new_floor>self.om.profit_floor_pct):
                            logger.info(f"[ENG] profit-lock {'ARMED' if self.om.profit_floor_pct is None else 'stepped up'}: "
                                        f"{self.om.symbol} best_gain={self.om.best_gain_pct:.1f}% -> floor={new_floor:.1f}%")
                            self.om.profit_floor_pct=new_floor; self.om._save()
                        if self.om.profit_floor_pct is not None:
                            floor_price=self.om.entry*(1+self.om.profit_floor_pct/100.0)
                            if ltp<=floor_price:
                                logger.info(f"[ENG] profit-lock hit: ltp={ltp:.2f} floor={floor_price:.2f} "
                                            f"(locked +{self.om.profit_floor_pct:.1f}%)")
                                self.trades.append(self.om.close(ltp,"profit_lock")); return

        if now.time() < self.cfg.SIGNAL_START_TIME: return   # too early for reliable signal detection

        st=self._signal()
        if st is None: return
        bar=st["bar"]
        if self._fired_bar is not None and bar==self._fired_bar: return   # already acted on this bar
        desired=st["dir"]
        held=self.om.dir if self.om.has else None
        if held==desired:                                 # same-direction setup -> mark & hold
            self._fired_bar=bar; return
        in_win = self.cfg.TRADE_START<=now.time()<=self.cfg.TRADE_END
        if self.om.has:                                   # opposite setup -> flip: close first
            ltp=self._ltp(self.om.symbol) or self.om.entry
            self.trades.append(self.om.close(ltp,"flip_exit"))
        if in_win:
            sym=self._opt_symbol(st["spot"],desired); ltp=self._ltp(sym)
            if ltp is None:
                logger.warning(f"[ENG] no LTP {sym}, skip entry"); self._fired_bar=bar; return
            sl_level=self._sl_level(st,desired)
            self.om.open(sym,desired,ltp,st["name"],sl_level)
            logger.info(f"[ENG] {st['name'].upper()} -> {desired} on bar {bar} "
                        f"spot={st['spot']:.1f} ema={st['ema']:.1f} SL@{sl_level:.1f}")
        self._fired_bar=bar
    def run(self):
        logger.info("[ENG] starting sequential-reversal poll loop")
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
            fp=os.path.join(path,f"pattern_trades_{d}.xlsx"); df.to_excel(fp,index=False)
            net=df["pnl_rs"].sum(); wins=(df["pnl_rs"]>0).sum()
            logger.info(f"[ENG] exported {len(df)} trades -> {fp} | net Rs {net:,.0f} | wins {wins}/{len(df)}")
        except Exception as e: logger.error(f"export: {e}")

if __name__=="__main__":
    Engine().run()
