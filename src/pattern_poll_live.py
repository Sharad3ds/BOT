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

RISK MANAGEMENT: fixed premium SL/TP only.
  - DRY-RUN: SL/TP are simulated by polling live option LTP.
  - LIVE:    orders are placed as productType="INTRADAY" using the documented
             POST /api/v3/orders/sync market-order schema. After the position is
             visible, PATCH /api/v3/positions attaches broker-side SL/TP using
             legType=2 (percentage), so the broker manages the exits.
No candle SL, no profit-lock trailing, no flip-on-opposite-signal.
A position only closes via broker SL/TP, explicit position exit, or eod_exit.

MANUAL CLOSE PATH (LIVE): OrderManager.close() uses the documented
DELETE /api/v3/positions endpoint with the actual position id
(e.g. <symbol>-INTRADAY).

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
client, OrderManager with state persistence, EOD forced exit, ExpiryResolver
for the current week's expiry. DRY_RUN=False by default in this LIVE version.

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
from typing import Optional, List
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

    # ── candle reversal pattern: fixed 3-candle check (C1,C2,C3=confirm). ──
    HEALTHY_BODY_FRAC = 0.8       # C3 "healthy": body >= this fraction of its high-low range
    CONFIRM_START = 3              # confirmation may be candle 3...
    CONFIRM_END = 3                # ...up to candle 3 (C3 is the entry candle)
    CONFIRM_MUST_BE_COLORED = True # SELL conf must be red / BUY conf must be green
    ENABLE_BUY = True              # BUY signals enabled
    BODY_RATIO_MAX = 1.2           # SELL/BUY fires if C1 body < this many times C2's body (OR declining/rising closes)

    # ── swing zone (support/resistance) entry filter ──
    USE_SWING_FILTER = True
    SWING_LOOKBACK = 20        # bars needed on both sides to confirm a pivot
    SWING_PROXIMITY_PCT = 0.2   # how close (% of zone price) C1 must be to count as "near"

    STRIKE_STEP = 100; LOT = 65; MONEYNESS = "ITM"
    SESSION_START = dtime(9, 20); SESSION_END = dtime(15, 30)
    SIGNAL_START_TIME = dtime(9, 35)
    TRADE_START = dtime(9, 30); TRADE_END = dtime(15, 0); EOD_EXIT_TIME = dtime(15, 20)

    # ── risk management: fixed premium SL + fixed premium TP only.
    #    No candle SL, no profit-lock trailing, no flip-on-opposite-signal.
    #    LIVE mode uses INTRADAY plus broker-side percentage SL/TP via
    #    PATCH /api/v3/positions with legType=2. ──
    USE_PREMIUM_SL = True
    STOP_LOSS_PCT = 10.0
    USE_PREMIUM_TP = True
    TAKE_PROFIT_PCT = 15.0

    POLL_INTERVAL_SECS = 20
    DRY_RUN = False

    # Fyers V3 broker-side protection is attached after the INTRADAY market
    # order is filled/visible in the Positions API.
    PROTECTION_RETRIES = 10
    PROTECTION_RETRY_DELAY_SECS = 0.5

    def __init__(self):
        exp_dt = get_next_tuesday()
        self.EXPIRY_CODE = expiry_code(exp_dt)
        logger.info(f"Weekly expiry code {self.EXPIRY_CODE} | "
                    f"C1/C2/C3 pattern: declining/rising closes OR body<{self.BODY_RATIO_MAX}xC2, "
                    f"{self.TF_RULE} bars | "
                    f"swing={'lookback='+str(self.SWING_LOOKBACK)+' prox='+str(self.SWING_PROXIMITY_PCT)+'%' if self.USE_SWING_FILTER else 'off'} | "
                    f"SL={'-'+str(self.STOP_LOSS_PCT)+'%' if self.USE_PREMIUM_SL else 'off'} "
                    f"TP={'+'+str(self.TAKE_PROFIT_PCT)+'%' if self.USE_PREMIUM_TP else 'off'} | "
                    f"live_exit={'DELETE /positions + broker SL/TP' if not self.DRY_RUN else 'dry-run sim'} | "
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
        """POST /api/v3/orders/sync -- documented Fyers V3 sync order API."""
        try:
            return self.s.post(f"{self.T1}/orders/sync", json=data).json()
        except Exception as e:
            return {"s": "error", "message": str(e)}

    def exit_order(self, data: dict) -> dict:
        """Cancel a pending order via DELETE /api/v3/orders/sync/{id}."""
        try:
            r = self.s.delete(f"{self.T1}/orders/sync/{data['id']}")
            return r.json()
        except Exception as e:
            return {"s": "error", "message": str(e)}

    def exit_position(self, position_id: str) -> dict:
        """Close a specific open position via DELETE /api/v3/positions."""
        try:
            r = self.s.delete(f"{self.T1}/positions", json={"id": position_id})
            return r.json()
        except Exception as e:
            return {"s": "error", "message": str(e)}

    def set_stop_loss_take_profit(
        self,
        position_id: str,
        stop_loss: float,
        take_profit: float,
        qty: int,
        leg_type: int = 2,
    ) -> dict:
        """PATCH /api/v3/positions -- attach SL/TP to an open position.

        Fyers V3 documents legType=1 as offset and legType=2 as percentage.
        This bot uses percentage protection so the configured STOP_LOSS_PCT and
        TAKE_PROFIT_PCT can be applied without depending on the exact fill price.
        """
        try:
            payload = {
                "positionId": position_id,
                "takeProfit": float(take_profit),
                "stopLoss": float(stop_loss),
                "qty": int(qty),
                "legType": int(leg_type),
            }
            r = self.s.patch(f"{self.T1}/positions", json=payload)
            return r.json()
        except Exception as e:
            return {"s": "error", "message": str(e)}

    def positions(self) -> dict:
        try:
            return self.s.get(f"{self.T1}/positions").json()
        except Exception as e:
            return {"s": "error", "message": str(e)}

    def get_position(self, symbol: str) -> Optional[dict]:
        """Return the current INTRADAY position for symbol, if present."""
        resp = self.positions()
        if not isinstance(resp, dict) or resp.get("s") != "ok":
            return None
        for pos in resp.get("netPositions", []):
            if pos.get("symbol") == symbol and pos.get("productType") == "INTRADAY":
                return pos
        return None

    def get_order(self, order_id: str) -> Optional[dict]:
        """Return a current-day order by exchange order id."""
        if not order_id:
            return None
        resp = self.orderbook()
        if not isinstance(resp, dict) or resp.get("s") != "ok":
            return None
        for order in resp.get("orderBook", []):
            if order.get("id") == order_id:
                return order
        return None

    def orderbook(self) -> dict:
        try:
            return self.s.get(f"{self.T1}/orders").json()
        except Exception as e:
            return {"s": "error", "message": str(e)}

    def tradebook(self) -> dict:
        try:
            return self.s.get(f"{self.T1}/trades").json()
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

    # C3 confirmation/entry candle must have a healthy real body: the body
    # occupies at least HEALTHY_BODY_FRAC of the candle's full high-low range.
    c3_range = h[i] - l[i]
    healthy_c3 = c3_range > 0 and body(i) >= cfg.HEALTHY_BODY_FRAC * c3_range

    def swing_ok(c1_idx, direction):
        if not cfg.USE_SWING_FILTER: return True
        kind = "high" if direction == "SELL" else "low"
        zone = most_recent_confirmed_zone(h, l, i, cfg.SWING_LOOKBACK, kind)
        return near_zone(l[c1_idx], h[c1_idx], zone, cfg.SWING_PROXIMITY_PCT)

    # ---- SELL: C1 green, C2 red, C3 confirmation ----
    # When CONFIRM_MUST_BE_COLORED=True, C3 must be red.
    sell_confirm_ok = (not cfg.CONFIRM_MUST_BE_COLORED) or red(i)
    if green(c1) and red(c2) and sell_confirm_ok and healthy_c3:
        cond_a = c[c1] > c[c2] > c[i]                       # three declining closes
        cond_b = body(c1) < cfg.BODY_RATIO_MAX * body(c2)   # C1 body < BODY_RATIO_MAX x C2 body
        if (cond_a or cond_b) and swing_ok(c1, "SELL"):
            return ("SELL", "sell_pattern", c1, c2)

    # ---- BUY (mirror): C1 red, C2 green, C3 confirmation ----
    # When CONFIRM_MUST_BE_COLORED=True, C3 must be green.
    buy_confirm_ok = (not cfg.CONFIRM_MUST_BE_COLORED) or green(i)
    if cfg.ENABLE_BUY and red(c1) and green(c2) and buy_confirm_ok and healthy_c3:
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

# --------------------------- ORDER MANAGER (dry-run + live INTRADAY) ---------------------------

class OrderManager:
    def __init__(self, fyers: FyersHTTP, cfg: Config):
        self.fyers = fyers; self.cfg = cfg
        self.has = False; self.symbol = None; self.direction = None
        self.entry_price = None; self.entry_time = None
        self.pattern = None; self.oid = None; self.position_id = None
        self.protection_attached = False
        self.entry_confirmed = False
        self.exit_pending_reason = None

    # ── state persistence ──

    def _save_state(self):
        try:
            state = {
                "has": self.has, "symbol": self.symbol, "direction": self.direction,
                "entry_price": self.entry_price,
                "entry_time": self.entry_time.isoformat() if self.entry_time else None,
                "pattern": self.pattern, "oid": self.oid,
                "position_id": self.position_id,
                "protection_attached": self.protection_attached,
                "entry_confirmed": self.entry_confirmed,
                "exit_pending_reason": self.exit_pending_reason,
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

    def _reset_state(self):
        self.has = False; self.symbol = None; self.direction = None
        self.entry_price = None; self.entry_time = None
        self.pattern = None; self.oid = None; self.position_id = None
        self.protection_attached = False
        self.entry_confirmed = False
        self.exit_pending_reason = None

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
                self.pattern = st.get("pattern"); self.oid = st.get("oid")
                self.position_id = st.get("position_id")
                self.protection_attached = bool(st.get("protection_attached", False))
                self.entry_confirmed = bool(st.get("entry_confirmed", False))
                self.exit_pending_reason = st.get("exit_pending_reason")
                if not self.position_id and self.symbol:
                    self.position_id = f"{self.symbol}-INTRADAY"
                logger.info(f"[OM] Restored {self.symbol} {self.direction} @ {self.entry_price} "
                            f"(oid={self.oid}, position_id={self.position_id})")
        except Exception as e:
            logger.error(f"[OM] restore: {e}")

    # ── helpers ──

    def _build_trade_rec(self, exit_ltp: float, reason: str) -> dict:
        pnl_pts = float(exit_ltp) - self.entry_price
        pnl_rs  = pnl_pts * self.cfg.LOT
        return {
            "direction": self.direction, "option_symbol": self.symbol, "pattern": self.pattern,
            "entry_time": self.entry_time.strftime("%Y-%m-%d %H:%M:%S") if self.entry_time else "",
            "entry_price": round(self.entry_price, 2),
            "exit_price": round(float(exit_ltp), 2), "exit_reason": reason,
            "pnl_pts": round(pnl_pts, 2), "pnl_rs": round(pnl_rs, 2),
            "exit_time": datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S"),
            "result": "profit" if pnl_pts > 0 else "loss",
        }

    def _current_ltp(self) -> float:
        """Best-effort current option LTP used for exit logging."""
        if not self.symbol:
            return 0.0
        try:
            resp = self.fyers.quotes(self.symbol)
            v = resp["d"][0]["v"].get("lp")
            return float(v) if v is not None else 0.0
        except Exception:
            return 0.0

    # ── open (dry-run + live INTRADAY market order) ──

    def open(self, symbol: str, direction: str, ltp: float, pattern: str) -> bool:
        if self.has:
            logger.warning("[OM] position already open, open() ignored")
            return False

        entry_price = float(ltp)

        if self.cfg.DRY_RUN:
            # Dry-run: simulate fill at LTP; caller simulates SL/TP using live option LTP.
            self.has = True; self.symbol = symbol; self.direction = direction
            self.entry_price = entry_price; self.entry_time = datetime.now(IST)
            self.pattern = pattern
            self.oid = f"DRY-{datetime.now(IST).strftime('%H%M%S')}"
            self.entry_confirmed = True
            logger.info(f"[OM][TRADE] BUY {symbol} ({direction}) @ {ltp:.2f} | pattern={pattern}")
            logger.info(f"[OM][TRADE] Entry details: qty={self.cfg.LOT} "
                        f"order_type=MARKET productType=INTRADAY "
                        f"SL_pts={entry_price * self.cfg.STOP_LOSS_PCT / 100.0:.0f} "
                        f"TP_pts={entry_price * self.cfg.TAKE_PROFIT_PCT / 100.0:.0f}")
            self._save_state()
            return True

        # LIVE: documented Fyers V3 market order + broker-side percentage SL/TP.
        # BO/CO are deprecated; INTRADAY is the supported product type.
        data = {
            "symbol": symbol,
            "qty": int(self.cfg.LOT),
            "type": 2,               # Market order
            "side": 1,               # Buy
            "productType": "INTRADAY",
            "limitPrice": 0,
            "stopPrice": 0,
            "validity": "DAY",
            "disclosedQty": 0,
            "offlineOrder": False,
            "stopLoss": 0,
            "takeProfit": 0,
            "orderTag": "PatRev",
            "isSliceOrder": False,
        }
        logger.info(f"[OM][LIVE] Placing INTRADAY MARKET BUY: symbol={symbol} dir={direction} "
                    f"qty={self.cfg.LOT} quote_ltp={entry_price:.2f}")
        resp = self.fyers.place_order(data=data)
        logger.info(f"[OM][LIVE] place_order response: {resp}")

        if not (isinstance(resp, dict) and resp.get("s") == "ok"):
            logger.error(f"[OM][LIVE] INTRADAY order rejected: {resp}")
            return False

        self.has = True; self.symbol = symbol; self.direction = direction
        self.entry_price = entry_price; self.entry_time = datetime.now(IST)
        self.pattern = pattern
        self.oid = resp.get("id")
        self.position_id = None
        self.protection_attached = False
        self.entry_confirmed = False
        self.exit_pending_reason = None

        # The market order response gives us the order id. The position id is
        # obtained from the Positions API once the fill becomes visible.
        pos = None
        for attempt in range(self.cfg.PROTECTION_RETRIES):
            pos = self.fyers.get_position(symbol)
            if pos and int(pos.get("netQty", 0)) != 0:
                break
            time.sleep(self.cfg.PROTECTION_RETRY_DELAY_SECS)

        if not pos or int(pos.get("netQty", 0)) == 0:
            logger.error(f"[OM][LIVE] Position not visible after order {self.oid}; "
                         "broker-side SL/TP could not be attached.")
            # Keep local state so the next poll can reconcile the open position.
            self.position_id = f"{symbol}-INTRADAY"
            self.protection_attached = False
            self.entry_confirmed = False
            self._save_state()
            return True

        self.position_id = pos.get("id") or f"{symbol}-INTRADAY"
        self.entry_confirmed = True
        actual_entry = pos.get("buyAvg") or pos.get("netAvg") or entry_price
        try:
            self.entry_price = float(actual_entry)
        except (TypeError, ValueError):
            self.entry_price = entry_price

        # Fyers V3 PATCH /positions: legType=2 means percentage.
        protection = self.fyers.set_stop_loss_take_profit(
            position_id=self.position_id,
            stop_loss=self.cfg.STOP_LOSS_PCT,
            take_profit=self.cfg.TAKE_PROFIT_PCT,
            qty=int(pos.get("qty", self.cfg.LOT)),
            leg_type=2,
        )
        logger.info(f"[OM][LIVE] SL/TP attach response: {protection}")

        if not (isinstance(protection, dict) and protection.get("s") == "ok"):
            logger.error(f"[OM][LIVE] Failed to attach broker SL/TP to {self.position_id}: {protection}")
            logger.error("[OM][LIVE] Position remains open; reconciliation will retry protection on the next poll.")
        else:
            self.protection_attached = True
            logger.info(f"[OM][LIVE] Broker protection attached: position={self.position_id} "
                        f"SL={self.cfg.STOP_LOSS_PCT}% TP={self.cfg.TAKE_PROFIT_PCT}%")

        logger.info(f"[OM][LIVE] INTRADAY order placed successfully, order_id={self.oid}")
        logger.info(f"[OM][TRADE] BUY {symbol} ({direction}) @ ~{self.entry_price:.2f} | pattern={pattern}")
        logger.info(f"[OM][TRADE] Entry details: qty={self.cfg.LOT} order_type=MARKET "
                    f"productType=INTRADAY position_id={self.position_id} "
                    f"SL={self.cfg.STOP_LOSS_PCT}% TP={self.cfg.TAKE_PROFIT_PCT}%")
        self._save_state()
        return True

    # ── close (dry-run uses provided ltp; live uses DELETE /positions) ──

    def close(self, ltp: float, reason: str) -> Optional[dict]:
        """Close position. LIVE uses the same INTRADAY market-order schema as the
        working S/R bot and then records the exit using the latest available LTP."""
        if not self.has or not self.symbol:
            logger.warning("[OM] no open position, close() ignored")
            return None

        if self.cfg.DRY_RUN:
            exit_ltp = float(ltp)
            rec = self._build_trade_rec(exit_ltp, reason)
            logger.info(f"[OM][TRADE] EXIT {self.symbol} @ {exit_ltp:.2f} reason={reason} "
                        f"entry={self.entry_price:.2f}")
            logger.info(f"[OM][TRADE] PnL Rs {rec['pnl_rs']:,.0f} | "
                        f"points={rec['pnl_pts']:.2f} | result={rec['result']}")
            self._reset_state(); self._clear_state()
            return rec

        # LIVE: verify the broker still reports an open INTRADAY position first.
        # Never send another exit request after netQty reaches 0.
        pos_resp = self.fyers.positions()
        if not isinstance(pos_resp, dict) or pos_resp.get("s") != "ok":
            logger.error(f"[OM][LIVE] Cannot verify position before exit; /positions response: {pos_resp}")
            return None

        pos = None
        for candidate in pos_resp.get("netPositions", []):
            if self.position_id and candidate.get("id") == self.position_id:
                pos = candidate
                break
            if candidate.get("symbol") == self.symbol and candidate.get("productType") == "INTRADAY":
                pos = candidate
                break

        if pos is None:
            logger.info(f"[OM][LIVE] No matching INTRADAY position record; netQty=0 cannot be confirmed. "
                        f"No exit order sent. symbol={self.symbol} reason={reason}")
            return None

        if int(pos.get("netQty", 0)) == 0:
            logger.info(f"[OM][LIVE] Position already closed (netQty=0); no exit order sent. "
                        f"symbol={self.symbol} reason={reason}")
            self._reset_state(); self._clear_state()
            return None

        self.position_id = pos.get("id") or self.position_id or f"{self.symbol}-INTRADAY"
        self.entry_confirmed = True
        actual_entry = pos.get("buyAvg") or pos.get("netAvg") or self.entry_price
        try:
            self.entry_price = float(actual_entry)
        except (TypeError, ValueError):
            pass

        self.exit_pending_reason = reason
        self._save_state()

        logger.info(f"[OM][LIVE] Exiting open INTRADAY position: position_id={self.position_id} "
                    f"symbol={self.symbol} netQty={pos.get('netQty')} reason={reason}")
        resp = self.fyers.exit_position(self.position_id)
        logger.info(f"[OM][LIVE] exit_position response: {resp}")

        if not (isinstance(resp, dict) and resp.get("s") == "ok"):
            logger.error(f"[OM][LIVE] position exit rejected: {resp}")
            self.exit_pending_reason = None
            self._save_state()
            return None

        # Verify the broker position after requesting the exit. A trade is treated
        # as closed only when /positions reports netQty == 0.
        pos_after_resp = self.fyers.positions()
        if not isinstance(pos_after_resp, dict) or pos_after_resp.get("s") != "ok":
            logger.warning(f"[OM][LIVE] Could not verify post-exit netQty; keeping state for reconciliation: "
                           f"{pos_after_resp}")
            return None

        pos_after = None
        for candidate in pos_after_resp.get("netPositions", []):
            if candidate.get("id") == self.position_id:
                pos_after = candidate
                break
            if candidate.get("symbol") == self.symbol and candidate.get("productType") == "INTRADAY":
                pos_after = candidate
                break

        if pos_after is None:
            logger.info(f"[OM][LIVE] Post-exit position record not available; netQty=0 not confirmed. "
                        f"No additional exit order will be sent; reconciliation will continue.")
            return None

        if int(pos_after.get("netQty", 0)) != 0:
            logger.info(f"[OM][LIVE] Exit request accepted but position is still open "
                        f"(netQty={pos_after.get('netQty')}). No additional exit order will be sent "
                        f"until a later reconciliation confirms netQty=0.")
            return None

        exit_ltp = self._current_ltp() or float(ltp)
        rec = self._build_trade_rec(exit_ltp, reason)
        logger.info(f"[OM][TRADE] EXIT {self.symbol} @ ~{exit_ltp:.2f} reason={reason} "
                    f"entry={self.entry_price:.2f}")
        logger.info(f"[OM][TRADE] PnL Rs {rec['pnl_rs']:,.0f} | "
                    f"points={rec['pnl_pts']:.2f} | result={rec['result']}")
        self._reset_state(); self._clear_state()
        return rec

    # ── live broker-position reconciliation ──

    def broker_exit_reason(self, exit_ltp: float) -> str:
        """Best-effort classification when broker-side SL/TP has closed the position."""
        if not self.entry_price:
            return "broker_exit"
        sl_price = self.entry_price * (1 - self.cfg.STOP_LOSS_PCT / 100.0)
        tp_price = self.entry_price * (1 + self.cfg.TAKE_PROFIT_PCT / 100.0)
        if exit_ltp <= sl_price:
            return "stop_loss"
        if exit_ltp >= tp_price:
            return "target"
        return "broker_exit"

    def sync_with_broker(self) -> Optional[dict]:
        """LIVE-only. Treat the INTRADAY position as closed only when /positions reports netQty == 0."""
        if not self.has or not self.symbol or self.cfg.DRY_RUN:
            return None

        resp = self.fyers.positions()
        if not isinstance(resp, dict) or resp.get("s") != "ok":
            logger.warning(f"[OM] sync_with_broker: positions API error: {resp}")
            return None

        found = None
        for pos in resp.get("netPositions", []):
            if self.position_id and pos.get("id") == self.position_id:
                found = pos
                break
            if pos.get("symbol") == self.symbol and pos.get("productType") == "INTRADAY":
                found = pos
                break

        # The order may be accepted before the fill becomes visible in Positions.
        # Do not treat that transient state as an exit.
        if not self.entry_confirmed:
            if found is not None and int(found.get("netQty", 0)) != 0:
                self.position_id = found.get("id") or f"{self.symbol}-INTRADAY"
                self.entry_confirmed = True
                actual_entry = found.get("buyAvg") or found.get("netAvg") or self.entry_price
                try:
                    self.entry_price = float(actual_entry)
                except (TypeError, ValueError):
                    pass
                logger.info(f"[OM][LIVE] Entry fill confirmed: position_id={self.position_id} "
                            f"entry~{self.entry_price:.2f}")
            else:
                order = self.fyers.get_order(self.oid)
                if order:
                    status = int(order.get("status", 0))
                    # 1=Canceled, 2=Traded/Filled, 5=Rejected, 6=Pending, 7=Expired.
                    if status in (1, 5, 7):
                        logger.error(f"[OM][LIVE] Entry order {self.oid} is not open: {order}")
                        self._reset_state(); self._clear_state()
                        return {"entry_failed": True}
                    logger.info(f"[OM][LIVE] Entry order {self.oid} status={status}; waiting for position visibility.")
                return None

        if found is not None and int(found.get("netQty", 0)) != 0:
            if self.position_id is None:
                self.position_id = found.get("id") or f"{self.symbol}-INTRADAY"

            if self.exit_pending_reason:
                logger.info(f"[OM][LIVE] Exit pending for {self.symbol}; broker still reports "
                            f"netQty={found.get('netQty')}. Waiting for netQty=0 before finalizing the exit.")
                return None

            actual_entry = found.get("buyAvg") or found.get("netAvg") or self.entry_price
            try:
                self.entry_price = float(actual_entry)
            except (TypeError, ValueError):
                pass

            if not self.protection_attached:
                protection = self.fyers.set_stop_loss_take_profit(
                    position_id=self.position_id,
                    stop_loss=self.cfg.STOP_LOSS_PCT,
                    take_profit=self.cfg.TAKE_PROFIT_PCT,
                    qty=int(found.get("qty", self.cfg.LOT)),
                    leg_type=2,
                )
                logger.info(f"[OM][LIVE] Retry SL/TP attach response: {protection}")
                if isinstance(protection, dict) and protection.get("s") == "ok":
                    self.protection_attached = True
                    self._save_state()
                    logger.info(f"[OM][LIVE] Broker protection attached on retry: position={self.position_id}")
            return None

        if found is None:
            logger.info(f"[OM][LIVE] No matching INTRADAY position record for {self.symbol}; "
                        f"waiting until /positions confirms netQty=0.")
            return None

        # Matching position is present with netQty=0 -> broker has closed it.
        exit_ltp = self._current_ltp()
        reason = self.exit_pending_reason or self.broker_exit_reason(exit_ltp)
        logger.info(f"[OM][LIVE] Broker position closed: position_id={self.position_id} "
                    f"symbol={self.symbol} ltp~{exit_ltp:.2f} reason={reason}")
        rec = self._build_trade_rec(exit_ltp, reason)
        logger.info(f"[OM][TRADE] EXIT {self.symbol} @ ~{exit_ltp:.2f} reason={reason} "
                    f"entry={self.entry_price:.2f}")
        logger.info(f"[OM][TRADE] PnL Rs {rec['pnl_rs']:,.0f} | "
                    f"points={rec['pnl_pts']:.2f} | result={rec['result']}")
        self._reset_state(); self._clear_state()
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
        else:
            logger.warning("*** LIVE MODE — INTRADAY market entry + broker-side SL/TP + DELETE /positions exit ***")

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
        """DRY-RUN only: fetch current option LTP and simulate SL/TP exit."""
        if not self.cfg.DRY_RUN or not self.om.has:
            return False

        opt_ltp = self._ltp(self.om.symbol)
        if opt_ltp is None:
            return False

        if self.cfg.USE_PREMIUM_SL and opt_ltp <= self.om.entry_price * (1 - self.cfg.STOP_LOSS_PCT / 100.0):
            rec = self.om.close(opt_ltp, "stop_loss")
            if rec: self.trades.append(rec)
            return True

        if self.cfg.USE_PREMIUM_TP and opt_ltp >= self.om.entry_price * (1 + self.cfg.TAKE_PROFIT_PCT / 100.0):
            rec = self.om.close(opt_ltp, "target")
            if rec: self.trades.append(rec)
            return True

        return False

    def _check_live_broker_exit(self) -> bool:
        """LIVE only: detect broker-side SL/TP or other automatic closure."""
        if self.cfg.DRY_RUN or not self.om.has:
            return False
        rec = self.om.sync_with_broker()
        if isinstance(rec, dict) and rec.get("entry_failed"):
            return True
        if rec is not None:
            self.trades.append(rec)
            return True
        return False

    def _eod_exit_if_needed(self) -> bool:
        now = datetime.now(IST)
        eod = IST.localize(datetime.combine(now.date(), self.cfg.EOD_EXIT_TIME))
        if now >= eod and self.om.has:
            # close() re-checks /positions and sends DELETE /positions only while
            # the broker still reports an open INTRADAY position (netQty != 0).
            ltp = self._ltp(self.om.symbol) or self.om.entry_price
            rec = self.om.close(ltp, "eod_exit")
            if rec: self.trades.append(rec)
            return True
        return False

    def _poll(self):
        now = datetime.now(IST)

        if self._today != now.date():
            self._today = now.date()
            self._fired_events = set()
            self._last_trade_dir = None   # direction alternation resets each new trading day
            logger.info(f"[ENG] New day {self._today} — state reset.")

        if not (self.cfg.SESSION_START <= now.time() <= self.cfg.SESSION_END):
            return

        # Position monitoring happens before EOD/manual exit. LIVE is considered
        # closed only when GET /positions reports netQty == 0.
        if self.cfg.DRY_RUN:
            if self._check_dryrun_exits():
                return
        else:
            if self._check_live_broker_exit():
                return

        # EOD exit is attempted only after the latest broker-position check.
        # OrderManager.close() performs another /positions check immediately before
        # sending DELETE /positions.
        if self._eod_exit_if_needed():
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
        opened = self.om.open(sym, next_ev["dir"], ltp, next_ev["name"])
        if opened:
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
            rec = self.om.close(ltp, "shutdown")
            if rec: self.trades.append(rec)
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
