"""
Sequential 3-candle REVERSAL LIVE polling bot -- ported to match
pattern_backtest_swingzone.py exactly (this is a full rewrite from the
earlier version; the pattern definition, risk management, and filters below
are ALL different from what used to be here).

PATTERN (fixed 3-candle check, C1/C2/C3=confirmation):
  SELL: C1 green, C2 red, C3 (confirmation, the last bar) red. Fires if
        EITHER (a) C1 Close > C2 Close > C3 Close (three declining closes),
        OR (b) C1 Body < BODY_RATIO_MAX x C2 Body (default 1.2x).
  BUY:  exact mirror -- C1 red, C2 green, C3 green. Fires if EITHER
        (a) C1 Close < C2 Close < C3 Close (three rising closes), OR
        (b) C1 Body < BODY_RATIO_MAX x C2 Body.
PLUS a swing-zone (support/resistance) filter: C1 must sit near the most
recently CONFIRMED swing zone, using detect_fractal_pivots() copied
VERBATIM from the Break/Retest bot / PatternPoll.py / the backtest -- all
four files must stay identical on this function; do not edit one without
updating the others.

RISK MANAGEMENT: fixed premium SL/TP only, checked every poll against live
option LTP -- NO candle SL anymore (no more C1 high/low reference on the
underlying), no profit-lock trailing, no flip-on-opposite-signal. A position
only closes via stop_loss (-10%), target (+20%), or eod_exit (15:20).

DIRECTION ALTERNATION: no CE directly after CE, no PE directly after PE.
Resets every trading day (this process restarts daily via cron), matching
the backtest's per-day-fresh zone computation. NOTE: the backtest's own
direction-alternation tracker persists across its ENTIRE multi-day run
(not reset daily) -- this is a KNOWN, confirmed point of divergence from
this live version. With EOD exit restored on both sides, trades almost
always resolve same-day in practice, so this matters less than it used to,
but it can still cause a live/backtest disagreement on which signal is
"first" right after a day boundary. Keep that in mind when comparing.

Architecture follows breakretest_poll_live.py / smc_poll_live.py: FyersHTTP
client, OrderManager with state persistence, dry-run SL/TP simulation via
live LTP polling, EOD forced exit, ExpiryResolver for the current week's
expiry. DRY_RUN=True by default.

Signal generation is CAUSAL and re-derived from scratch every poll using only
TODAY's closed bars, exactly mirroring generate_signals()'s per-day logic in
the backtest (swing zones reset daily too -- no carry-over from prior days,
same as the backtest's per-day groupby starting r_zone=None/s_zone=None).
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, time as dtime
import requests
import logging
import os
from typing import Dict, Optional, List
import warnings
import pytz
import calendar
import time
import json
import urllib3
from expiry_resolver import ExpiryResolver

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
IST = pytz.timezone("Asia/Kolkata")

STATE_FILE = "pattern_live_position_state.json"

def ist_time(*args):
    return datetime.now(IST).timetuple()
logging.Formatter.converter = ist_time

try:
    from logger_config import get_logger
    logger = get_logger()
except ImportError:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - PatternPoll - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logger = logging.getLogger("PatternPoll")

# --------------------------- EXPIRY HELPERS (identical convention) ---------------------------

def get_next_tuesday(now=None):
    now = now or datetime.now(IST)
    x = datetime(now.year, now.month, now.day)
    while x.weekday() != 1: x += timedelta(days=1)
    return x

def monthly_expiry(y, m):
    last = calendar.monthrange(y, m)[1]; x = datetime(y, m, last)
    while x.weekday() != 1: x -= timedelta(days=1)
    return x

def expiry_code(dt):
    yy = dt.year % 100; me = monthly_expiry(dt.year, dt.month)
    if dt.date() == me.date(): return f"{yy:02d}{dt.strftime('%b').upper()}"
    mc = str(dt.month) if dt.month <= 9 else {10: 'O', 11: 'N', 12: 'D'}[dt.month]
    return f"{yy:02d}{mc}{dt.day:02d}"

# --------------------------- CONFIG (matches pattern_backtest_swingzone.py) ---------------------------

class Config:
    UNDERLYING = "NSE:NIFTY50-INDEX"; BASE_RES = "1"
    TF_RULE = "1min"

    # ── candle reversal pattern: fixed 3-candle check (C1,C2,C3=confirm).
    #    CONFIRM_START kept only as the "need at least this many bars" gate. ──
    CONFIRM_START = 3
    ENABLE_BUY = True
    BODY_RATIO_MAX = 1.2         # SELL/BUY fires if C1 body < this many times C2's body (OR declining/rising closes)

    # ── swing zone (support/resistance) entry filter ──
    USE_SWING_FILTER = True
    SWING_LOOKBACK = 10          # bars needed on both sides to confirm a pivot
    SWING_PROXIMITY_PCT = 0.5    # how close (% of zone price) C1 must be to count as "near"

    STRIKE_STEP = 100; LOT = 65; MONEYNESS = "ITM"
    SESSION_START = dtime(9, 20); SESSION_END = dtime(15, 30)
    SIGNAL_START_TIME = dtime(9, 35)
    TRADE_START = dtime(9, 30); TRADE_END = dtime(15, 0); EOD_EXIT_TIME = dtime(15, 20)

    # ── risk management: fixed premium SL + fixed premium TP only.
    #    No candle SL, no profit-lock trailing, no flip-on-opposite-signal --
    #    a position only ever closes via stop_loss, target, or eod_exit. ──
    USE_PREMIUM_SL = True
    STOP_LOSS_PCT = 10.0
    USE_PREMIUM_TP = True
    TAKE_PROFIT_PCT = 20.0

    POLL_INTERVAL_SECS = 20
    DRY_RUN = True

    def __init__(self):
        exp_dt = get_next_tuesday()
        self.EXPIRY_CODE = expiry_code(exp_dt)
        logger.info(f"Weekly expiry code {self.EXPIRY_CODE} | "
                    f"C1/C2/C3 pattern: declining/rising closes OR body<{self.BODY_RATIO_MAX}xC2, "
                    f"{self.TF_RULE} bars | "
                    f"swing={'lookback='+str(self.SWING_LOOKBACK)+' prox='+str(self.SWING_PROXIMITY_PCT)+'%' if self.USE_SWING_FILTER else 'off'} | "
                    f"SL={'-'+str(self.STOP_LOSS_PCT)+'%' if self.USE_PREMIUM_SL else 'off'} "
                    f"TP={'+'+str(self.TAKE_PROFIT_PCT)+'%' if self.USE_PREMIUM_TP else 'off'} | "
                    f"direction-alternating (resets daily) | DRY_RUN={self.DRY_RUN}")

# --------------------------- FYERS HTTP CLIENT ---------------------------

class FyersHTTP:
    T1 = "https://api-t1.fyers.in/api/v3"; T2 = "https://api-t1.fyers.in/data"

    def __init__(self, app, tok):
        auth = tok if ":" in tok else f"{app}:{tok}"
        self.s = requests.Session(); self.s.verify = False
        self.s.headers.update({"Authorization": auth})

    def profile(self): return self.s.get(f"{self.T1}/profile").json()

    def history(self, symbol, res, d_from, d_to):
        r = self.s.get(f"{self.T2}/history", params={
            "symbol": symbol, "resolution": res, "date_format": "1",
            "range_from": d_from, "range_to": d_to, "cont_flag": "0"})
        try: return r.json()
        except Exception: return {}

    def quotes(self, symbol: str) -> dict:
        try:
            return self.s.get(f"{self.T2}/quotes", params={"symbols": symbol}).json()
        except Exception:
            return {}

    def place_order(self, data: dict) -> dict:
        try:
            return self.s.post(f"{self.T1}/orders/sync", json=data).json()
        except Exception as e:
            return {"s": "error", "message": str(e)}

# --------------------------- FRACTAL PIVOT DETECTION (VERBATIM -- see module docstring) ---------------------------

def detect_fractal_pivots(h, l, lookback: int):
    """Direct port of Pine's ta.pivotlow(low, bb, bb) / ta.pivothigh(high, bb, bb).
    MUST stay identical across breakretest_poll_live.py, PatternPoll.py,
    pattern_backtest_swingzone.py, and this file. A pivot at raw index piv_i
    is only confirmed once piv_i+lookback bars exist after it, so pivots
    within `lookback` bars of the array's end are never returned -- this is
    by design, not a bug."""
    n = len(h)
    piv_lows, piv_highs = [], []
    for i in range(lookback, n - lookback):
        window_l = l[i - lookback: i + lookback + 1]
        window_h = h[i - lookback: i + lookback + 1]
        if l[i] == np.min(window_l):
            piv_lows.append((i + lookback, i))
        if h[i] == np.max(window_h):
            piv_highs.append((i + lookback, i))
    return piv_highs, piv_lows

def most_recent_confirmed_zone(h, l, up_to_idx, lookback, kind):
    """
    Returns (zone_top, zone_bottom) for the most recently CONFIRMED pivot of
    the requested kind ("high" or "low") among bars [0, up_to_idx], or None
    if none confirmed yet. Zone is defined the same way the Break/Retest bot
    defines it: the pivot's own high/low plus its immediate neighbors.
    """
    if up_to_idx < 2*lookback + 1:
        return None
    piv_highs, piv_lows = detect_fractal_pivots(h[:up_to_idx+1], l[:up_to_idx+1], lookback)
    pivots = piv_highs if kind == "high" else piv_lows
    if not pivots:
        return None
    _, pi = pivots[-1]   # most recent CONFIRMED pivot (by confirm-index order)
    if not (0 < pi < up_to_idx):
        return None
    if kind == "high":
        return (float(h[pi]), float(max(h[pi-1], h[pi+1])))   # (top, bottom)
    else:
        return (float(min(l[pi-1], l[pi+1])), float(l[pi]))   # (top, bottom)

def near_zone(rng_low, rng_high, zone, proximity_pct):
    """Does [rng_low, rng_high] (e.g. C1's own high/low) overlap the zone,
    extended by a proximity band on each side?"""
    if zone is None:
        return False
    zone_top, zone_bottom = zone
    band = zone_top * proximity_pct / 100.0
    return not (rng_low > zone_top + band or rng_high < zone_bottom - band)

# --------------------------- SEQUENTIAL CANDLE-REVERSAL DETECTION (VERBATIM) ---------------------------

def detect_signal(o, h, l, c, cfg):
    """
    C1 green, C2 red, C3 (confirmation, bar i) red -- fires if EITHER:
      (a) C1 Close > C2 Close > C3 Close (three declining closes), OR
      (b) C1 Body < BODY_RATIO_MAX x C2 Body
    BUY is the exact mirror: C1 red, C2 green, C3 green, fires if EITHER
      (a) C1 Close < C2 Close < C3 Close (three rising closes), OR
      (b) C1 Body < BODY_RATIO_MAX x C2 Body
    PLUS C1 must sit near a genuine, recently-confirmed support/resistance
    swing zone (SELL -> near a swing high, BUY -> near a swing low).
    MUST stay identical to pattern_backtest_swingzone.py's detect_signal()."""
    n = len(c)
    if n < 3: return None
    i = n - 1
    c1, c2 = i - 2, i - 1
    green = lambda k: c[k] > o[k]
    red   = lambda k: c[k] < o[k]
    body  = lambda k: abs(c[k] - o[k])

    def swing_ok(c1_idx, direction):
        if not cfg.USE_SWING_FILTER: return True
        kind = "high" if direction == "SELL" else "low"
        zone = most_recent_confirmed_zone(h, l, i, cfg.SWING_LOOKBACK, kind)
        return near_zone(l[c1_idx], h[c1_idx], zone, cfg.SWING_PROXIMITY_PCT)

    # ---- SELL: C1 green, C2 red, C3 (confirm) red ----
    if green(c1) and red(c2) and red(i):
        cond_a = c[c1] > c[c2] > c[i]                       # three declining closes
        cond_b = body(c1) < cfg.BODY_RATIO_MAX * body(c2)     # C1 body < 1.2x C2 body
        if (cond_a or cond_b) and swing_ok(c1, "SELL"):
            return ("SELL", "sell_pattern", c1, c2)

    # ---- BUY (mirror): C1 red, C2 green, C3 (confirm) green ----
    if cfg.ENABLE_BUY and red(c1) and green(c2) and green(i):
        cond_a = c[c1] < c[c2] < c[i]                       # three rising closes
        cond_b = body(c1) < cfg.BODY_RATIO_MAX * body(c2)
        if (cond_a or cond_b) and swing_ok(c1, "BUY"):
            return ("BUY", "buy_pattern", c1, c2)
    return None

def generate_today_events(day_df: pd.DataFrame, cfg: Config) -> List[dict]:
    """Rebuilds signal events for a single day's bars -- IDENTICAL logic to
    pattern_backtest_swingzone.py's generate_signals() inner loop (one day
    at a time, zones reset fresh each day). Called fresh every poll with
    only TODAY's closed bars."""
    r = day_df.resample(cfg.TF_RULE).agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    ).dropna()
    if len(r) < cfg.CONFIRM_START:
        return []

    o = r["open"].to_numpy(); h = r["high"].to_numpy()
    l = r["low"].to_numpy(); cl = r["close"].to_numpy()

    events: List[dict] = []
    for i in range(cfg.CONFIRM_START - 1, len(r)):
        bar_close_time = r.index[i] + pd.Timedelta(cfg.TF_RULE)
        ts_time = bar_close_time.time()
        if not (cfg.SESSION_START <= ts_time <= cfg.SESSION_END):
            continue
        if ts_time < cfg.SIGNAL_START_TIME:
            continue

        sig = detect_signal(o[:i+1], h[:i+1], l[:i+1], cl[:i+1], cfg)
        if sig is None:
            continue
        direction, name, c1, c2 = sig
        events.append({
            "timestamp": bar_close_time,
            "dir": direction,
            "name": name,
            "spot": float(cl[i]),
        })

    return events

# --------------------------- ORDER MANAGER (dry-run + live) ---------------------------

class OrderManager:
    def __init__(self, fyers: FyersHTTP, cfg: Config):
        self.fyers = fyers; self.cfg = cfg
        self.has = False; self.symbol = None; self.direction = None
        self.entry_price = None; self.entry_time = None
        self.pattern = None; self.oid = None

    def _save_state(self):
        try:
            state = {
                "has": self.has, "symbol": self.symbol, "direction": self.direction,
                "entry_price": self.entry_price,
                "entry_time": self.entry_time.isoformat() if self.entry_time else None,
                "pattern": self.pattern,
                "saved": datetime.now(IST).isoformat(),
            }
            tmp = STATE_FILE + ".tmp"
            json.dump(state, open(tmp, "w"))
            os.replace(tmp, STATE_FILE)
        except Exception as e:
            logger.error(f"[OM] save state: {e}")

    def _clear_state(self):
        try:
            if os.path.exists(STATE_FILE): os.remove(STATE_FILE)
        except Exception: pass

    def restore(self):
        if not os.path.exists(STATE_FILE): return
        try:
            st = json.load(open(STATE_FILE))
            saved = datetime.fromisoformat(st["saved"])
            if saved.date() != datetime.now(IST).date():
                self._clear_state(); return
            if st.get("has"):
                self.has = True; self.symbol = st["symbol"]; self.direction = st["direction"]
                self.entry_price = st["entry_price"]
                self.entry_time = datetime.fromisoformat(st["entry_time"]) if st.get("entry_time") else None
                self.pattern = st.get("pattern")
                logger.info(f"[OM] Restored {self.symbol} {self.direction} @ {self.entry_price}")
        except Exception as e:
            logger.error(f"[OM] restore: {e}")

    def open(self, symbol: str, direction: str, ltp: float, pattern: str):
        self.has = True; self.symbol = symbol; self.direction = direction
        self.entry_price = float(ltp); self.entry_time = datetime.now(IST)
        self.pattern = pattern
        if self.cfg.DRY_RUN:
            self.oid = f"DRY-{datetime.now(IST).strftime('%H%M%S')}"
            logger.info(f"[OM][DRY] BUY {symbol} ({direction}) @ {ltp:.2f} | pattern={pattern}")
        else:
            resp = self.fyers.place_order({
                "symbol": symbol, "qty": self.cfg.LOT, "type": 2, "side": 1,
                "productType": "INTRADAY", "limitPrice": 0, "stopPrice": 0,
                "validity": "DAY", "offlineOrder": False,
            })
            self.oid = resp.get("id")
            logger.info(f"[OM][LIVE] BUY {symbol} resp={resp}")
        self._save_state()

    def close(self, ltp: float, reason: str) -> dict:
        pnl_pts = float(ltp) - self.entry_price
        pnl_rs = pnl_pts * self.cfg.LOT
        if not self.cfg.DRY_RUN:
            resp = self.fyers.place_order({
                "symbol": self.symbol, "qty": self.cfg.LOT, "type": 2, "side": -1,
                "productType": "INTRADAY", "limitPrice": 0, "stopPrice": 0,
                "validity": "DAY", "offlineOrder": False,
            })
            logger.info(f"[OM][LIVE] SELL {self.symbol} resp={resp}")
        logger.info(f"[OM]{'[DRY]' if self.cfg.DRY_RUN else ''} EXIT {self.symbol} @ {ltp:.2f} "
                    f"reason={reason} entry={self.entry_price:.2f} PnL=Rs {pnl_rs:,.0f}")
        rec = {
            "direction": self.direction, "option_symbol": self.symbol, "pattern": self.pattern,
            "entry_time": self.entry_time.strftime("%Y-%m-%d %H:%M:%S") if self.entry_time else "",
            "entry_price": round(self.entry_price, 2),
            "exit_price": round(float(ltp), 2), "exit_reason": reason,
            "pnl_pts": round(pnl_pts, 2), "pnl_rs": round(pnl_rs, 2),
            "exit_time": datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S"),
            "result": "profit" if pnl_pts > 0 else "loss",
        }
        self.has = False; self.symbol = None; self.direction = None
        self.entry_price = None; self.entry_time = None
        self.pattern = None; self.oid = None
        self._clear_state()
        return rec

# --------------------------- ENGINE ---------------------------

class Engine:
    def __init__(self):
        self.cfg = Config()
        self.fy = self._init_fyers()
        self.expiry_resolver = ExpiryResolver()
        self.om = OrderManager(self.fy, self.cfg)
        self.om.restore()
        self.trades: List[dict] = []
        self._fired_events: set = set()          # timestamps already acted on today
        self._last_trade_dir: Optional[str] = None  # DIRECTION ALTERNATION tracker (resets daily)
        self._today: Optional[object] = None
        logger.info(f"[ENG] ready. interval={self.cfg.POLL_INTERVAL_SECS}s")
        if self.cfg.DRY_RUN:
            logger.warning("*** DRY-RUN — no real orders ***")

    def _init_fyers(self) -> FyersHTTP:
        for f in ["fyers_appid.txt", "fyers_token.txt"]:
            if not os.path.exists(f): raise FileNotFoundError(f)
        app = open("fyers_appid.txt").read().strip()
        tok = open("fyers_token.txt").read().strip()
        fy = FyersHTTP(app, tok); p = fy.profile()
        if p.get("s") != "ok":
            raise ConnectionError(f"Fyers auth failed: {p}")
        logger.info(f"[ENG] connected {p['data']['name']}")
        return fy

    def _fetch_today_bars(self) -> pd.DataFrame:
        now = datetime.now(IST)
        resp = self.fy.history(self.cfg.UNDERLYING, self.cfg.BASE_RES,
                                now.strftime("%Y-%m-%d"), now.strftime("%Y-%m-%d"))
        if resp.get("s") != "ok":
            logger.warning(f"[ENG] history err: {resp}")
            return pd.DataFrame()
        candles = resp.get("candles", [])
        if not candles:
            return pd.DataFrame()
        df = pd.DataFrame(candles, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s", utc=True).dt.tz_convert(IST)
        df = df.drop_duplicates("timestamp").sort_values("timestamp").set_index("timestamp")
        return df.iloc[df.index.indexer_between_time(self.cfg.SESSION_START, self.cfg.SESSION_END)]

    def _opt_symbol(self, spot: float, direction: str) -> str:
        c = self.cfg
        exp_code, _ = self.expiry_resolver.resolve_expiry_code(
            datetime.now(IST).date(), expiry_code, get_next_tuesday)
        atm = int(round(spot / c.STRIKE_STEP) * c.STRIKE_STEP); itm = c.MONEYNESS == "ITM"
        if direction == "BUY":
            strike = atm - c.STRIKE_STEP if itm else atm; typ = "CE"
        else:
            strike = atm + c.STRIKE_STEP if itm else atm; typ = "PE"
        return f"NSE:NIFTY{exp_code}{strike:05d}{typ}"

    def _ltp(self, symbol: str) -> Optional[float]:
        r = self.fy.quotes(symbol)
        try:
            v = r["d"][0]["v"].get("lp")
            return float(v) if v is not None else None
        except Exception:
            return None

    def _drop_forming_bar(self, df: pd.DataFrame) -> pd.DataFrame:
        """Trims the still-forming bar so only CLOSED bars feed the signal
        engine. Bar duration is derived from BASE_RES (in minutes, since no
        'S' suffix -- matches Fyers' convention)."""
        if df.empty:
            return df
        try:
            bar_minutes = int(self.cfg.BASE_RES)
        except ValueError:
            bar_minutes = 1
        now = datetime.now(IST)
        last_start = df.index[-1]
        if last_start.tzinfo is None:
            last_start = IST.localize(last_start)
        if last_start + timedelta(minutes=bar_minutes) > now:
            df = df.iloc[:-1]
        return df

    def _check_dryrun_exits(self) -> bool:
        """Fixed premium SL (-10%) and TP (+20%) checked against live option
        LTP -- NO candle SL anymore, matching pattern_backtest_swingzone.py's
        actual execution logic."""
        if not self.cfg.DRY_RUN or not self.om.has:
            return False

        opt_ltp = self._ltp(self.om.symbol)
        if opt_ltp is None:
            return False

        if self.cfg.USE_PREMIUM_SL and opt_ltp <= self.om.entry_price * (1 - self.cfg.STOP_LOSS_PCT / 100.0):
            self.trades.append(self.om.close(opt_ltp, "stop_loss"))
            return True

        if self.cfg.USE_PREMIUM_TP and opt_ltp >= self.om.entry_price * (1 + self.cfg.TAKE_PROFIT_PCT / 100.0):
            self.trades.append(self.om.close(opt_ltp, "target"))
            return True

        return False

    def _eod_exit_if_needed(self) -> bool:
        now = datetime.now(IST)
        eod = IST.localize(datetime.combine(now.date(), self.cfg.EOD_EXIT_TIME))
        if now >= eod and self.om.has:
            ltp = self._ltp(self.om.symbol) or self.om.entry_price
            self.trades.append(self.om.close(ltp, "eod_exit"))
            return True
        return False

    def _poll(self):
        now = datetime.now(IST)

        if self._today != now.date():
            self._today = now.date()
            self._fired_events = set()
            self._last_trade_dir = None   # direction alternation resets each new trading day
            logger.info(f"[ENG] New day {self._today} — state reset.")

        if self._eod_exit_if_needed():
            return

        if not (self.cfg.SESSION_START <= now.time() <= self.cfg.SESSION_END):
            return

        if self._check_dryrun_exits():
            return

        df = self._fetch_today_bars()
        df = self._drop_forming_bar(df)
        if df.empty:
            return

        events = generate_today_events(df, self.cfg)
        if not events:
            return

        next_ev = None
        for ev in events:
            if ev["timestamp"] not in self._fired_events:
                next_ev = ev
                break
        if next_ev is None:
            return

        self._fired_events.add(next_ev["timestamp"])

        # single-position engine: ignore all new signals while in a trade
        if self.om.has:
            logger.info(f"[ENG] Signal {next_ev['dir']} @ {next_ev['timestamp']} ignored — already in position.")
            return

        # DIRECTION ALTERNATION: no CE after CE, no PE after PE
        if self._last_trade_dir is not None and next_ev["dir"] == self._last_trade_dir:
            logger.info(f"[ENG] Signal {next_ev['dir']} @ {next_ev['timestamp']} skipped — "
                        f"same direction as last trade ({self._last_trade_dir}).")
            return

        if not (self.cfg.TRADE_START <= now.time() <= self.cfg.TRADE_END):
            logger.info(f"[ENG] Signal {next_ev['dir']} @ {next_ev['timestamp']} outside trade window — skipping.")
            return

        sym = self._opt_symbol(next_ev["spot"], next_ev["dir"])
        ltp = self._ltp(sym)
        if ltp is None:
            logger.warning(f"[ENG] No LTP for {sym} — skipping this signal.")
            return

        logger.info(f"[ENG] Signal {next_ev['dir']} @ {next_ev['timestamp']} "
                    f"pattern={next_ev['name']} -> entering {sym} @ {ltp:.2f}")
        self.om.open(sym, next_ev["dir"], ltp, next_ev["name"])
        self._last_trade_dir = next_ev["dir"]

    def run(self):
        logger.info("[ENG] starting Sequential Candle Reversal poll loop")
        try:
            while True:
                t0 = time.monotonic()
                try:
                    self._poll()
                except Exception as e:
                    logger.error(f"poll err: {e}", exc_info=True)
                time.sleep(max(0.0, self.cfg.POLL_INTERVAL_SECS - (time.monotonic() - t0)))
        except KeyboardInterrupt:
            logger.info("[ENG] interrupted")
        finally:
            self._shutdown()

    def _shutdown(self):
        if self.om.has:
            ltp = self._ltp(self.om.symbol) or self.om.entry_price
            self.trades.append(self.om.close(ltp, "shutdown"))
        self._export()
        logger.info("[ENG] shutdown complete")

    def _export(self):
        if not self.trades:
            logger.info("[ENG] no trades to export")
            return
        try:
            df = pd.DataFrame(self.trades)
            df["cum_pnl_rs"] = df["pnl_rs"].cumsum()
            d = datetime.now(IST).strftime("%Y%m%d")
            path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
            os.makedirs(path, exist_ok=True)
            fp = os.path.join(path, f"pattern_trades_{d}.xlsx")
            df.to_excel(fp, index=False)
            net = df["pnl_rs"].sum(); wins = (df["pnl_rs"] > 0).sum()
            logger.info(f"[ENG] exported {len(df)} trades -> {fp} | net Rs {net:,.0f} | wins {wins}/{len(df)}")
        except Exception as e:
            logger.error(f"export: {e}")


if __name__ == "__main__":
    Engine().run()