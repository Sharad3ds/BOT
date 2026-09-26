"""
Breaks and Retests [HG] LIVE polling bot -- deploys the exact zone + break +
retest logic from the working breakretest_backtest.py config (LOOKBACK=10,
BARS_SINCE_BREAK=2, RETEST_VALID_BARS=2, TF_RULE="1min", EXPIRY_TYPE=weekly,
SL=20%, TP=20%, same-direction-per-day guard).

Architecture follows smc_poll_live.py: FyersHTTP client, OrderManager with
state persistence across restarts, dry-run SL/TP simulation via live LTP
polling, EOD forced exit. DRY_RUN=True by default.

Signal generation is CAUSAL and re-derived from scratch every poll using only
TODAY's closed bars (zones reset daily, same as the backtest's per-day
groupby) -- cheap at 1-min resolution, and guarantees live signals are
bit-identical to what the backtest would produce for the same historical
data. A dedup set prevents re-acting on the same confirmed retest twice.

detect_fractal_pivots() and the break/retest zone state machine are copied
VERBATIM from breakretest_backtest.py -- do not modify one without updating
the other in lockstep.
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

STATE_FILE = "breakretest_position_state.json"

def ist_time(*args):
    return datetime.now(IST).timetuple()
logging.Formatter.converter = ist_time

try:
    from logger_config import get_logger
    logger = get_logger()
except ImportError:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - BRPoll - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logger = logging.getLogger("BRPoll")

# --------------------------- EXPIRY HELPERS (identical convention) ---------------------------

def get_next_tuesday(now=None):
    now = now or datetime.now(IST)
    x = datetime(now.year, now.month, now.day)
    while x.weekday() != 1: x += timedelta(days=1)
    return x

def get_nifty_monthly_expiry(year: int, month: int) -> datetime:
    last_day = calendar.monthrange(year, month)[1]
    dt = datetime(year, month, last_day)
    while dt.weekday() != 1:
        dt -= timedelta(days=1)
    return dt

def get_current_or_next_monthly_expiry(now=None) -> datetime:
    now = now or datetime.now(IST)
    y, m = now.year, now.month
    exp_dt = get_nifty_monthly_expiry(y, m)
    if exp_dt.date() < now.date():
        if m == 12:
            y += 1; m = 1
        else:
            m += 1
        exp_dt = get_nifty_monthly_expiry(y, m)
    return exp_dt

def monthly_expiry(y, m):
    last = calendar.monthrange(y, m)[1]; x = datetime(y, m, last)
    while x.weekday() != 1: x -= timedelta(days=1)
    return x

def expiry_code(dt):
    yy = dt.year % 100; me = monthly_expiry(dt.year, dt.month)
    if dt.date() == me.date(): return f"{yy:02d}{dt.strftime('%b').upper()}"
    mc = str(dt.month) if dt.month <= 9 else {10: 'O', 11: 'N', 12: 'D'}[dt.month]
    return f"{yy:02d}{mc}{dt.day:02d}"

# --------------------------- CONFIG (identical values to the working backtest) ---------------------------

class Config:
    UNDERLYING = "NSE:NIFTY50-INDEX"; BASE_RES = "1"
    TF_RULE = "1min"

    LOOKBACK          = 10
    BARS_SINCE_BREAK  = 2
    RETEST_VALID_BARS = 2

    STRIKE_STEP = 100; LOT = 65; MONEYNESS = "ITM"
    EXPIRY_TYPE = "WEEKLY"   # "WEEKLY" or "MONTHLY"
    SESSION_START = dtime(9, 20); SESSION_END = dtime(15, 30)
    TRADE_START = dtime(9, 30); TRADE_END = dtime(15, 0); EOD_EXIT_TIME = dtime(15, 20)

    STOP_LOSS_PCT = 20.0
    TAKE_PROFIT_PCT = 20.0

    POLL_INTERVAL_SECS = 20
    DRY_RUN = True

    def __init__(self):
        now = datetime.now(IST)
        exp_dt = (get_current_or_next_monthly_expiry(now) if self.EXPIRY_TYPE.upper() == "MONTHLY"
                  else get_next_tuesday(now))
        self.EXPIRY_CODE = expiry_code(exp_dt)
        logger.info(f"{self.EXPIRY_TYPE} expiry code {self.EXPIRY_CODE} | Break&Retest on {self.TF_RULE} "
                    f"lookback={self.LOOKBACK} bars_since_break={self.BARS_SINCE_BREAK} "
                    f"retest_valid={self.RETEST_VALID_BARS} | SL={self.STOP_LOSS_PCT}% TP={self.TAKE_PROFIT_PCT}% "
                    f"| DRY_RUN={self.DRY_RUN}")

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

# --------------------------- FRACTAL PIVOT DETECTION (VERBATIM from breakretest_backtest.py) ---------------------------

def detect_fractal_pivots(h, l, lookback: int):
    """Direct port of Pine's ta.pivotlow(low, bb, bb) / ta.pivothigh(high, bb, bb).
    See breakretest_backtest.py for full derivation notes. MUST stay
    identical between the two files."""
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

def generate_today_events(day_df: pd.DataFrame, cfg: Config) -> List[dict]:
    """Rebuilds zones + fires break/retest-confirmation events for a single
    day's bars -- IDENTICAL logic to breakretest_backtest.py's
    generate_signals() inner loop (one day at a time). Called fresh every
    poll with only TODAY's closed bars, so it's deterministic and matches
    what the backtest would have produced for this same data."""
    r = day_df.resample(cfg.TF_RULE).agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    ).dropna()
    n = len(r)
    if n < cfg.LOOKBACK * 2 + 2:
        return []

    o = r["open"].to_numpy(); h = r["high"].to_numpy()
    l = r["low"].to_numpy(); cl = r["close"].to_numpy()

    piv_highs, piv_lows = detect_fractal_pivots(h, l, cfg.LOOKBACK)
    highs_by_confirm = {ci: pi for ci, pi in piv_highs}
    lows_by_confirm = {ci: pi for ci, pi in piv_lows}

    events: List[dict] = []
    r_zone = None
    s_zone = None

    for i in range(cfg.LOOKBACK, n):
        if i in highs_by_confirm:
            piv_i = highs_by_confirm[i]
            if 0 < piv_i < n - 1:
                r_zone = {"top": h[piv_i], "bottom": max(h[piv_i - 1], h[piv_i + 1]),
                          "broken": False, "break_idx": None, "retest": None}
        if i in lows_by_confirm:
            piv_i = lows_by_confirm[i]
            if 0 < piv_i < n - 1:
                s_zone = {"top": min(l[piv_i - 1], l[piv_i + 1]), "bottom": l[piv_i],
                          "broken": False, "break_idx": None, "retest": None}

        if r_zone is not None:
            if not r_zone["broken"]:
                if i > 0 and cl[i - 1] <= r_zone["top"] < cl[i]:
                    r_zone["broken"] = True
                    r_zone["break_idx"] = i
            else:
                bars_since = i - r_zone["break_idx"]
                if bars_since >= cfg.BARS_SINCE_BREAK:
                    rTop, rBot = r_zone["top"], r_zone["bottom"]
                    touch = (
                        (l[i] <= rBot and cl[i] >= rTop) or
                        (l[i] <= rBot and rBot <= cl[i] <= rTop) or
                        (rBot <= l[i] <= rTop) or
                        (rBot <= l[i] <= rTop and cl[i] > rTop)
                    )
                    if r_zone["retest"] is None and touch:
                        r_zone["retest"] = {"touch_idx": i, "ref_high": h[i]}
                    elif r_zone["retest"] is not None:
                        rt = r_zone["retest"]
                        within_window = rt["touch_idx"] < i <= rt["touch_idx"] + cfg.RETEST_VALID_BARS
                        if within_window and cl[i] >= rt["ref_high"]:
                            events.append({
                                "timestamp": r.index[i] + pd.Timedelta(cfg.TF_RULE),
                                "dir": "BUY", "zone_kind": "resistance",
                                "zone_top": rTop, "zone_bottom": rBot,
                                "retest_ref": rt["ref_high"], "index_price": float(cl[i]),
                            })
                            r_zone["retest"] = None
                        elif i > rt["touch_idx"] + cfg.RETEST_VALID_BARS:
                            r_zone["retest"] = None

        if s_zone is not None:
            if not s_zone["broken"]:
                if i > 0 and cl[i - 1] >= s_zone["bottom"] > cl[i]:
                    s_zone["broken"] = True
                    s_zone["break_idx"] = i
            else:
                bars_since = i - s_zone["break_idx"]
                if bars_since >= cfg.BARS_SINCE_BREAK:
                    sTop, sBot = s_zone["top"], s_zone["bottom"]
                    touch = (
                        (h[i] >= sTop and cl[i] <= sBot) or
                        (h[i] >= sTop and sBot <= cl[i] <= sTop) or
                        (sBot <= h[i] <= sTop) or
                        (sBot <= h[i] <= sTop and cl[i] < sBot)
                    )
                    if s_zone["retest"] is None and touch:
                        s_zone["retest"] = {"touch_idx": i, "ref_low": l[i]}
                    elif s_zone["retest"] is not None:
                        rt = s_zone["retest"]
                        within_window = rt["touch_idx"] < i <= rt["touch_idx"] + cfg.RETEST_VALID_BARS
                        if within_window and cl[i] <= rt["ref_low"]:
                            events.append({
                                "timestamp": r.index[i] + pd.Timedelta(cfg.TF_RULE),
                                "dir": "SELL", "zone_kind": "support",
                                "zone_top": sTop, "zone_bottom": sBot,
                                "retest_ref": rt["ref_low"], "index_price": float(cl[i]),
                            })
                            s_zone["retest"] = None
                        elif i > rt["touch_idx"] + cfg.RETEST_VALID_BARS:
                            s_zone["retest"] = None

    return events

# --------------------------- ORDER MANAGER (dry-run + live) ---------------------------

class OrderManager:
    def __init__(self, fyers: FyersHTTP, cfg: Config):
        self.fyers = fyers; self.cfg = cfg
        self.has = False; self.symbol = None; self.direction = None
        self.entry_price = None; self.entry_time = None
        self.zone_kind = None; self.oid = None

    def _save_state(self):
        try:
            state = {
                "has": self.has, "symbol": self.symbol, "direction": self.direction,
                "entry_price": self.entry_price,
                "entry_time": self.entry_time.isoformat() if self.entry_time else None,
                "zone_kind": self.zone_kind,
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
                self.zone_kind = st.get("zone_kind")
                logger.info(f"[OM] Restored {self.symbol} {self.direction} @ {self.entry_price}")
        except Exception as e:
            logger.error(f"[OM] restore: {e}")

    def open(self, symbol: str, direction: str, ltp: float, zone_kind: str):
        self.has = True; self.symbol = symbol; self.direction = direction
        self.entry_price = float(ltp); self.entry_time = datetime.now(IST)
        self.zone_kind = zone_kind
        if self.cfg.DRY_RUN:
            self.oid = f"DRY-{datetime.now(IST).strftime('%H%M%S')}"
            logger.info(f"[OM][DRY] BUY {symbol} ({direction}) @ {ltp:.2f} | zone={zone_kind}")
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
            "direction": self.direction, "option_symbol": self.symbol, "zone_kind": self.zone_kind,
            "entry_time": self.entry_time.strftime("%Y-%m-%d %H:%M:%S") if self.entry_time else "",
            "entry_price": round(self.entry_price, 2),
            "exit_price": round(float(ltp), 2), "exit_reason": reason,
            "pnl_pts": round(pnl_pts, 2), "pnl_rs": round(pnl_rs, 2),
            "exit_time": datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S"),
            "result": "profit" if pnl_pts > 0 else "loss",
        }
        self.has = False; self.symbol = None; self.direction = None
        self.entry_price = None; self.entry_time = None
        self.zone_kind = None; self.oid = None
        self._clear_state()
        return rec

# --------------------------- ENGINE ---------------------------

class Engine:
    def __init__(self):
        self.cfg = Config()
        self.fy = self._init_fyers()
        self.om = OrderManager(self.fy, self.cfg)
        self.om.restore()
        self.trades: List[dict] = []
        self._fired_events: set = set()   # (timestamp, direction, retest_ref) fired today
        self._traded_directions_today: set = set()   # same-direction-per-day guard
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

    def _fetch_today_1min(self) -> pd.DataFrame:
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
        exp_dt = (get_current_or_next_monthly_expiry() if c.EXPIRY_TYPE.upper() == "MONTHLY"
                  else get_next_tuesday())
        exp_code = expiry_code(exp_dt)
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
        if df.empty:
            return df
        now = datetime.now(IST)
        last_start = df.index[-1]
        if last_start.tzinfo is None:
            last_start = IST.localize(last_start)
        if last_start + timedelta(minutes=1) > now:
            df = df.iloc[:-1]
        return df

    def _check_dryrun_sl_tp(self) -> bool:
        if not self.cfg.DRY_RUN or not self.om.has:
            return False
        ltp = self._ltp(self.om.symbol)
        if ltp is None:
            return False
        sl_price = self.om.entry_price * (1 - self.cfg.STOP_LOSS_PCT / 100.0)
        tp_price = self.om.entry_price * (1 + self.cfg.TAKE_PROFIT_PCT / 100.0)
        if ltp <= sl_price:
            self.trades.append(self.om.close(ltp, "stop_loss"))
            return True
        if ltp >= tp_price:
            self.trades.append(self.om.close(ltp, "target"))
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
            self._traded_directions_today = set()
            logger.info(f"[ENG] New day {self._today} — state reset.")

        if self._eod_exit_if_needed():
            return

        if not (self.cfg.SESSION_START <= now.time() <= self.cfg.SESSION_END):
            return

        if self._check_dryrun_sl_tp():
            return

        df = self._fetch_today_1min()
        df = self._drop_forming_bar(df)
        if df.empty:
            return

        events = generate_today_events(df, self.cfg)
        if not events:
            return

        # find first unfired event
        next_ev = None
        for ev in events:
            key = (ev["timestamp"], ev["dir"], round(ev["retest_ref"], 2))
            if key not in self._fired_events:
                next_ev = ev
                break
        if next_ev is None:
            return

        key = (next_ev["timestamp"], next_ev["dir"], round(next_ev["retest_ref"], 2))
        self._fired_events.add(key)

        # same-direction-per-day guard
        if next_ev["dir"] in self._traded_directions_today:
            logger.info(f"[ENG] Signal {next_ev['dir']} @ {next_ev['timestamp']} skipped — "
                        f"{next_ev['dir']} already traded today.")
            return

        # single-position engine
        if self.om.has:
            logger.info(f"[ENG] Signal {next_ev['dir']} @ {next_ev['timestamp']} ignored — already in position.")
            return

        if not (self.cfg.TRADE_START <= now.time() <= self.cfg.TRADE_END):
            logger.info(f"[ENG] Signal {next_ev['dir']} @ {next_ev['timestamp']} outside trade window — skipping.")
            return

        sym = self._opt_symbol(next_ev["index_price"], next_ev["dir"])
        ltp = self._ltp(sym)
        if ltp is None:
            logger.warning(f"[ENG] No LTP for {sym} — skipping this signal.")
            return

        logger.info(f"[ENG] Signal {next_ev['dir']} @ {next_ev['timestamp']} "
                    f"zone={next_ev['zone_kind']} -> entering {sym} @ {ltp:.2f}")
        self.om.open(sym, next_ev["dir"], ltp, next_ev["zone_kind"])
        self._traded_directions_today.add(next_ev["dir"])

    def run(self):
        logger.info("[ENG] starting Break & Retest poll loop")
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
            fp = os.path.join(path, f"breakretest_trades_{d}.xlsx")
            df.to_excel(fp, index=False)
            net = df["pnl_rs"].sum(); wins = (df["pnl_rs"] > 0).sum()
            logger.info(f"[ENG] exported {len(df)} trades -> {fp} | net Rs {net:,.0f} | wins {wins}/{len(df)}")
        except Exception as e:
            logger.error(f"export: {e}")


if __name__ == "__main__":
    Engine().run()
