"""
Support/Resistance Reversal LIVE polling bot -- deploys the exact zone +
reversal-confirmation logic from the working smc_backtest.py (LOT=650,
TF_RULE="1min", SWING_LEN=15, ATR_PERIOD=10, ZONE_TOL_MULT=0.1,
ZONE_INVALIDATE_MULT=0.1, HEALTHY_BODY_FRAC=0.5, SL=20%, TP=5%).

Architecture follows vam_poll_5S.py: FyersHTTP client, OrderManager with
state persistence across restarts, TradeLogger, dry-run SL/TP simulation via
live LTP polling, EOD forced exit. DRY_RUN=True by default.

Signal generation is CAUSAL and re-derived from scratch every poll using only
TODAY's closed bars (zones reset daily, same as the backtest's per-day
groupby) -- cheap at 1-min resolution, and guarantees live signals are
bit-identical to what the backtest would produce for the same historical
data. A dedup set (mirroring vam_poll_5S.py's _fired_signals) prevents
re-acting on the same zone-fire event twice.

detect_pivots() is copied VERBATIM from smc_backtest.py -- do not modify
without updating both files in lockstep.
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

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
IST = pytz.timezone("Asia/Kolkata")

STATE_FILE = "smc_position_state.json"

def ist_time(*args):
    return datetime.now(IST).timetuple()
logging.Formatter.converter = ist_time

try:
    from logger_config import get_logger
    logger = get_logger()
except ImportError:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - SMCPoll - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logger = logging.getLogger("SMCPoll")

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

# --------------------------- CONFIG (identical values to the working backtest) ---------------------------

class Config:
    UNDERLYING = "NSE:NIFTY50-INDEX"; BASE_RES = "1"; HIST_DAYS = 30
    TF_RULE = "1min"

    SWING_LEN = 20
    ATR_PERIOD = 15
    ZONE_TOL_MULT = 0.1
    ZONE_INVALIDATE_MULT = 0.1
    HEALTHY_BODY_FRAC = 0.001
    MAX_ACTIVE_ZONES_PER_SIDE = 5

    STRIKE_STEP = 100; LOT = 6500; MONEYNESS = "ITM"
    SESSION_START = dtime(9, 20); SESSION_END = dtime(15, 30)
    SIGNAL_START_TIME = dtime(9, 35)
    TRADE_START = dtime(9, 30); TRADE_END = dtime(15, 0); EOD_EXIT_TIME = dtime(15, 20)

    STOP_LOSS_PCT = 20.0
    TAKE_PROFIT_PCT = 30.0

    POLL_INTERVAL_SECS = 20
    DRY_RUN = True

    def __init__(self):
        self.EXPIRY_CODE = expiry_code(get_next_tuesday())
        logger.info(f"Weekly expiry code {self.EXPIRY_CODE} | S/R reversal on {self.TF_RULE} "
                    f"swing_len={self.SWING_LEN} zone_tol={self.ZONE_TOL_MULT}xATR | "
                    f"SL={self.STOP_LOSS_PCT}% TP={self.TAKE_PROFIT_PCT}% | DRY_RUN={self.DRY_RUN}")

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

# --------------------------- SWING PIVOT DETECTION (VERBATIM from smc_backtest.py) ---------------------------

def detect_pivots(o, h, l, c, swing_len: int):
    """
    Direct port of LuxAlgo's leg()/getCurrentStructure() pivot logic.
    See smc_backtest.py for full derivation notes. MUST stay identical
    between the two files.
    """
    n = len(c)
    pivot_highs, pivot_lows = [], []
    leg = 0  # 0 = bearish leg, 1 = bullish leg

    for i in range(swing_len, n):
        window_high = np.max(h[i - swing_len + 1: i + 1])
        window_low  = np.min(l[i - swing_len + 1: i + 1])
        piv_idx = i - swing_len

        new_leg_high = h[piv_idx] > window_high
        new_leg_low  = l[piv_idx] < window_low

        new_leg = leg
        if new_leg_high:
            new_leg = 0
        elif new_leg_low:
            new_leg = 1

        if new_leg != leg:
            if new_leg == 1:
                pivot_lows.append((i, piv_idx, float(l[piv_idx])))
            else:
                pivot_highs.append((i, piv_idx, float(h[piv_idx])))
            leg = new_leg

    return pivot_highs, pivot_lows

def generate_today_events(day_df: pd.DataFrame, cfg: Config) -> List[dict]:
    """
    Rebuilds zones + fires reversal-confirmation events for a single day's
    bars -- IDENTICAL logic/order of operations to smc_backtest.py's
    generate_signals() inner loop (one day at a time). Called fresh every
    poll with only TODAY's closed bars, so it's deterministic and matches
    what the backtest would have produced for this same data.
    """
    r = day_df.resample(cfg.TF_RULE).agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    ).dropna()
    if len(r) < cfg.SWING_LEN + cfg.ATR_PERIOD + 2:
        return []

    o = r["open"].to_numpy(); h = r["high"].to_numpy()
    l = r["low"].to_numpy(); cl = r["close"].to_numpy()

    prev_close = np.empty(len(r)); prev_close[0] = cl[0]; prev_close[1:] = cl[:-1]
    tr = np.maximum(h - l, np.maximum(np.abs(h - prev_close), np.abs(l - prev_close)))
    atr = pd.Series(tr).ewm(alpha=1 / cfg.ATR_PERIOD, adjust=False).mean().to_numpy()

    pivot_highs, pivot_lows = detect_pivots(o, h, l, cl, cfg.SWING_LEN)

    events: List[dict] = []
    zones: List[dict] = []
    highs_by_confirm = {}
    for confirm_i, piv_i, level in pivot_highs:
        highs_by_confirm.setdefault(confirm_i, []).append((piv_i, level))
    lows_by_confirm = {}
    for confirm_i, piv_i, level in pivot_lows:
        lows_by_confirm.setdefault(confirm_i, []).append((piv_i, level))

    def healthy(k):
        rng = h[k] - l[k]
        return rng > 0 and abs(cl[k] - o[k]) >= cfg.HEALTHY_BODY_FRAC * rng
    green = lambda k: cl[k] > o[k]
    red   = lambda k: cl[k] < o[k]

    for i in range(cfg.SWING_LEN, len(r)):
        bar_close_time = r.index[i] + pd.Timedelta(cfg.TF_RULE)
        ts_time = bar_close_time.time()
        if not (cfg.SESSION_START <= ts_time <= cfg.SESSION_END):
            continue

        tol = cfg.ZONE_TOL_MULT * atr[i] if not np.isnan(atr[i]) else None
        if tol is not None:
            for piv_i, level in highs_by_confirm.get(i, []):
                n_active = sum(1 for z in zones if z["kind"] == "resistance" and z["active"])
                if n_active < cfg.MAX_ACTIVE_ZONES_PER_SIDE:
                    zones.append({"level": level, "tol": tol, "kind": "resistance",
                                  "state": "watching", "active": True})
            for piv_i, level in lows_by_confirm.get(i, []):
                n_active = sum(1 for z in zones if z["kind"] == "support" and z["active"])
                if n_active < cfg.MAX_ACTIVE_ZONES_PER_SIDE:
                    zones.append({"level": level, "tol": tol, "kind": "support",
                                  "state": "watching", "active": True})

        if ts_time < cfg.SIGNAL_START_TIME:
            continue

        for z in zones:
            if not z["active"]:
                continue
            lo, hi = z["level"] - z["tol"], z["level"] + z["tol"]
            in_zone = lo <= cl[i] <= hi

            if z["kind"] == "support":
                invalidated = cl[i] < z["level"] - cfg.ZONE_INVALIDATE_MULT * atr[i]
            else:
                invalidated = cl[i] > z["level"] + cfg.ZONE_INVALIDATE_MULT * atr[i]

            if z["state"] == "watching":
                if invalidated:
                    z["active"] = False
                    continue
                if in_zone:
                    z["state"] = "in_zone"
                continue

            if z["state"] == "in_zone":
                if invalidated:
                    z["active"] = False
                    continue
                if z["kind"] == "support" and green(i) and healthy(i) and cl[i] > hi:
                    events.append({
                        "timestamp": bar_close_time, "dir": "BUY",
                        "zone_level": z["level"], "zone_kind": "support",
                        "index_price": float(cl[i]),
                    })
                    z["active"] = False
                elif z["kind"] == "resistance" and red(i) and healthy(i) and cl[i] < lo:
                    events.append({
                        "timestamp": bar_close_time, "dir": "SELL",
                        "zone_level": z["level"], "zone_kind": "resistance",
                        "index_price": float(cl[i]),
                    })
                    z["active"] = False

    return events

# --------------------------- ORDER MANAGER (dry-run + live, same pattern as VAM/Supertrend bots) ---------------------------

class OrderManager:
    def __init__(self, fyers: FyersHTTP, cfg: Config):
        self.fyers = fyers; self.cfg = cfg
        self.has = False; self.symbol = None; self.direction = None
        self.entry_price = None; self.entry_time = None
        self.zone_level = None; self.zone_kind = None; self.oid = None

    def _save_state(self):
        try:
            state = {
                "has": self.has, "symbol": self.symbol, "direction": self.direction,
                "entry_price": self.entry_price,
                "entry_time": self.entry_time.isoformat() if self.entry_time else None,
                "zone_level": self.zone_level, "zone_kind": self.zone_kind,
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
                self.zone_level = st.get("zone_level"); self.zone_kind = st.get("zone_kind")
                logger.info(f"[OM] Restored {self.symbol} {self.direction} @ {self.entry_price}")
        except Exception as e:
            logger.error(f"[OM] restore: {e}")

    def open(self, symbol: str, direction: str, ltp: float, zone_level: float, zone_kind: str):
        self.has = True; self.symbol = symbol; self.direction = direction
        self.entry_price = float(ltp); self.entry_time = datetime.now(IST)
        self.zone_level = zone_level; self.zone_kind = zone_kind
        if self.cfg.DRY_RUN:
            self.oid = f"DRY-{datetime.now(IST).strftime('%H%M%S')}"
            logger.info(f"[OM][DRY] BUY {symbol} ({direction}) @ {ltp:.2f} | zone={zone_kind}@{zone_level:.2f}")
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
            "direction": self.direction, "option_symbol": self.symbol,
            "zone_kind": self.zone_kind, "zone_level": round(self.zone_level, 2) if self.zone_level else None,
            "entry_time": self.entry_time.strftime("%Y-%m-%d %H:%M:%S") if self.entry_time else "",
            "entry_price": round(self.entry_price, 2),
            "exit_price": round(float(ltp), 2), "exit_reason": reason,
            "pnl_pts": round(pnl_pts, 2), "pnl_rs": round(pnl_rs, 2),
            "exit_time": datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S"),
            "result": "profit" if pnl_pts > 0 else "loss",
        }
        self.has = False; self.symbol = None; self.direction = None
        self.entry_price = None; self.entry_time = None
        self.zone_level = None; self.zone_kind = None; self.oid = None
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
        self._fired_events: set = set()  # (bar_close_time, direction, zone_level) tuples fired today
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
        atm = int(round(spot / c.STRIKE_STEP) * c.STRIKE_STEP); itm = c.MONEYNESS == "ITM"
        if direction == "BUY":
            strike = atm - c.STRIKE_STEP if itm else atm; typ = "CE"
        else:
            strike = atm + c.STRIKE_STEP if itm else atm; typ = "PE"
        return f"NSE:NIFTY{c.EXPIRY_CODE}{strike:05d}{typ}"

    def _ltp(self, symbol: str) -> Optional[float]:
        r = self.fy.quotes(symbol)
        try:
            v = r["d"][0]["v"].get("lp")
            return float(v) if v is not None else None
        except Exception:
            return None

    def _drop_forming_bar(self, df: pd.DataFrame) -> pd.DataFrame:
        """Trim the still-forming 1-min bar so only CLOSED bars feed the
        signal engine -- same causal guarantee as every other bot here."""
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

        # reset per-day dedup state at the start of a new session day
        if self._today != now.date():
            self._today = now.date()
            self._fired_events = set()
            logger.info(f"[ENG] New day {self._today} — zone/dedup state reset.")

        if self._eod_exit_if_needed():
            return

        if not (self.cfg.SESSION_START <= now.time() <= self.cfg.SESSION_END):
            return

        # dry-run SL/TP simulated against live LTP every poll
        if self._check_dryrun_sl_tp():
            return

        df = self._fetch_today_1min()
        df = self._drop_forming_bar(df)
        if df.empty:
            return

        events = generate_today_events(df, self.cfg)
        if not events:
            return

        # find first unfired event (mirrors vam_poll_5S.py's dedup pattern)
        next_ev = None
        for ev in events:
            key = (ev["timestamp"], ev["dir"], round(ev["zone_level"], 2))
            if key not in self._fired_events:
                next_ev = ev
                break
        if next_ev is None:
            return

        key = (next_ev["timestamp"], next_ev["dir"], round(next_ev["zone_level"], 2))
        self._fired_events.add(key)

        # single-position engine, same as backtest: ignore new events while in a trade
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
                    f"zone={next_ev['zone_kind']}@{next_ev['zone_level']:.2f} -> entering {sym} @ {ltp:.2f}")
        self.om.open(sym, next_ev["dir"], ltp, next_ev["zone_level"], next_ev["zone_kind"])

    def run(self):
        logger.info("[ENG] starting S/R reversal poll loop")
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
            fp = os.path.join(path, f"smc_trades_{d}.xlsx")
            df.to_excel(fp, index=False)
            net = df["pnl_rs"].sum(); wins = (df["pnl_rs"] > 0).sum()
            logger.info(f"[ENG] exported {len(df)} trades -> {fp} | net Rs {net:,.0f} | wins {wins}/{len(df)}")
        except Exception as e:
            logger.error(f"export: {e}")


if __name__ == "__main__":
    Engine().run()