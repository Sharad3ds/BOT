"""
Sequential 3-candle REVERSAL LIVE polling bot -- HEIKIN ASHI VARIANT,
FULLY INDEPENDENT (no import of pattern_poll_live.py).

This is the LIVE counterpart of pattern_poll_live_HA_backtest.py, carrying
over the EXACT config that backtested at ~43% win rate with decent net
profit (60-day window):
    TF_RULE=5min, BASE_RES=5
    BODY_RATIO_MAX=0.5
    USE_SWING_FILTER=False   (swing-zone filter OFF)
    USE_C2_DOJI_FILTER=True, C2_MAX_BODY_RANGE_RATIO=0.12
    STOP_LOSS_PCT=10.0, TAKE_PROFIT_PCT=30.0
    POLL_INTERVAL_SECS=60
NOTE: the backtest run that produced the 43%-win-rate result did NOT have
flip_exit enabled (it used the plain "ignore any signal while holding, no
flips, no re-entries" engine). This live file matches that EXACTLY --
flip_exit is intentionally NOT included here. Say the word if you want it
added; it's a small, isolated change to the Engine loop below.

STRATEGY LOGIC (Config, detect_fractal_pivots, most_recent_confirmed_zone,
near_zone, detect_signal [+ C2 doji filter], generate_today_events) is
copied VERBATIM from pattern_poll_live.py (as pasted 2026-08-30) with only
the tuned Config values above and the C2 doji addition applied -- same
convention as pattern_poll_live_HA_backtest.py. If pattern_poll_live.py's
core logic changes later, this file will NOT pick it up automatically.

THE ONE STRATEGIC DIFFERENCE FROM pattern_poll_live.py: every poll, this
bot converts TODAY's real 1-min (well, BASE_RES-min) closed bars to Heikin
Ashi (reset fresh each trading day -- HA_open[0] = (O+C)/2 of today's first
real bar, no carry-over from yesterday, matching the backtest's daily
reset) BEFORE calling generate_today_events. Signal detection therefore
runs on HA candles; OPTION ORDER PRICES (entry LTP, stop/take-profit
levels, exit fills) are always REAL market prices -- Heikin Ashi has no
tradable price and only ever shapes which bar produced a signal.

ARCHITECTURE: same as pattern_poll_live.py -- FyersHTTP client,
OrderManager with state persistence, EOD forced exit, ExpiryResolver for
the current week's expiry, direction alternation (resets daily). DRY_RUN=
True by default -- flip to False only once you've watched this run in
dry-run for real market sessions and are satisfied with its behavior.

MIGRATED OFF DEPRECATED BO (2026-08-31): per Fyers' own changelog --
    02 AUG 2026 -- "CO & BO Deprecated, Added modify the Stop Loss and
                    Take Profit for an existing open position."
    07 AUG 2026 -- "Added Manage Positions option to cancel Stop Loss and
                    Take Profit on an open position via PATCH
                    /api/v3/positions by passing null."
-- productType="BO" is gone from the current place-order spec (only CNC,
INTRADAY, MARGIN, MTF remain), and the original OrderManager.open()/close()
here (and in pattern_poll_live.py) built its entries around BO orders with
broker-held SL/TP legs. THIS FILE NOW USES THE DOCUMENTED REPLACEMENT
instead of that deprecated flow:
  - Entry: OrderManager.open() places a plain productType="INTRADAY" limit
    order (Config.PRODUCT_TYPE) with stopLoss/takeProfit fields directly on
    the order body (same field names Fyers' "Place Take Profit and Stop
    Loss orders" doc section describes for INTRADAY/CNC/MARGIN/MTF) --
    Fyers auto-creates the linked SL/TP child orders exactly like BO used
    to, just without the deprecated productType. The resulting open
    position is identified as "<symbol>-<PRODUCT_TYPE>" (e.g.
    "NSE:NIFTY...CE-INTRADAY"), stored as self.position_id.
  - Exit: OrderManager.close() calls the documented "Exit Position - By Id"
    endpoint -- DELETE /api/v3/positions with {"id": position_id} -- instead
    of the old BO-specific {"id": f"{symbol}-BO"}.
  - Profit lock (LIVE): raises the position's stop-loss via the documented
    "Modify stop-loss and target" endpoint -- PATCH /api/v3/positions with
    {"positionId": position_id, "stopLoss": <pct>, "legType": 2} (legType
    2 = percentage, straight from the doc) -- instead of the old guessed
    orderbook-scan-for-a-BO-child-leg approach. This is now built directly
    from Fyers' documented API rather than reverse-engineered, but it has
    STILL not been exercised against a live account from this bot -- do a
    small real-money validation run (place, watch profit cross a tier,
    confirm the position's stop-loss actually moved in the Fyers app)
    before trusting it at full size, the same way test_exit_order.py
    validated the old exit_position() call before it went into
    pattern_poll_live.py.
  - Order-placement stopLoss/takeProfit UNITS: the doc's "Place Take Profit
    and Stop Loss orders" section says only "provide valid price" for these
    fields without a worked non-zero example -- this file assumes they are
    POINT OFFSETS from entry (matching the pre-deprecation BO convention,
    same field names carried over). Worth confirming on a small live order
    if you want full certainty.

DEPENDENCY: like pattern_poll_live.py, this file expects expiry_resolver.py
(ExpiryResolver class) alongside it in the same production folder, plus
fyers_appid.txt / fyers_token.txt for auth. Uses its own state file
(pattern_live_ha_position_state.json) so it won't collide with
pattern_poll_live.py's state if both ever exist side by side.

PROFIT LOCK (added 2026-08-31): a stepped trailing-SL ladder --
  profit >= 10% -> lock SL at +2% profit
  profit >= 15% -> lock SL at +5% profit
  profit >= 20% -> lock SL at +10% profit
(configurable via Config.PROFIT_LOCK_LADDER, a list of (profit_pct,
lock_pct) tuples -- add more tiers past 20% the same way). The lock only
ever ratchets UP, never down, even if profit later retraces. TAKE_PROFIT_PCT
(30%) is untouched and still the hard cap.
  - DRY-RUN: fully implemented and tested -- OrderManager tracks
    locked_sl_pct locally and the dry-run SL check in Engine compares
    against the ratcheted level instead of the fixed STOP_LOSS_PCT.
  - LIVE: implemented against Fyers' documented "Modify stop-loss and
    target" position endpoint (see MIGRATED OFF DEPRECATED BO above) --
    no more guessed child-leg lookups, but still not exercised against a
    live account from this bot. Validate before trusting at full size.

IMPORTANT: pattern_poll_live.py (the non-HA live bot) and any other bot in
your crontab that still places productType="BO" orders (or calls
exit_position with id=f"{symbol}-BO") is targeting a deprecated order type
per the changelog above -- worth checking against your live account
urgently, independent of anything in this file.
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

STATE_FILE = "pattern_live_ha_position_state.json"

def ist_time(*args):
    return datetime.now(IST).timetuple()
logging.Formatter.converter = ist_time

try:
    from logger_config import get_logger
    logger = get_logger()
except ImportError:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - PatternPollHA - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logger = logging.getLogger("PatternPollHA")

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

# --------------------------- CONFIG (tuned Heikin Ashi settings) ---------------------------

class Config:
    UNDERLYING = "NSE:NIFTY50-INDEX"; BASE_RES = "5"
    TF_RULE = "5min"

    # ── candle reversal pattern: fixed 3-candle check (C1,C2,C3=confirm).
    #    CONFIRM_START kept only as the "need at least this many bars" gate. ──
    CONFIRM_START = 3
    ENABLE_BUY = True
    BODY_RATIO_MAX = 0.5         # SELL/BUY fires if C1 body < this many times C2's body (OR declining/rising closes)

    # ── swing zone (support/resistance) entry filter -- OFF in this tuned config ──
    USE_SWING_FILTER = False
    SWING_LOOKBACK = 10          # bars needed on both sides to confirm a pivot
    SWING_PROXIMITY_PCT = 0.5    # how close (% of zone price) C1 must be to count as "near"

    # ── C2 doji/small-body filter: C2 must be a small, indecision-style
    #    candle sitting between C1 and the C3 confirmation. Body is measured
    #    against C2's OWN high-low range: body(C2)/(high(C2)-low(C2)). ──
    USE_C2_DOJI_FILTER = True
    C2_MAX_BODY_RANGE_RATIO = 0.12

    STRIKE_STEP = 100; LOT = 65; MONEYNESS = "ITM"
    SESSION_START = dtime(9, 20); SESSION_END = dtime(15, 30)
    SIGNAL_START_TIME = dtime(9, 35)
    TRADE_START = dtime(9, 30); TRADE_END = dtime(15, 0); EOD_EXIT_TIME = dtime(15, 20)

    # ── risk management: fixed premium SL + fixed premium TP.
    #    A position closes via stop_loss, target, eod_exit, OR flip_exit
    #    (added 2026-09-01, see USE_FLIP_EXIT below -- NOTE this is a
    #    deviation from the exact backtested 43%-winrate config, which did
    #    NOT have flip_exit enabled; added on request to stop missing
    #    reversal signals like the 09-01 10:10 BUY that got ignored while
    #    holding a losing SELL). ──
    USE_PREMIUM_SL = True
    STOP_LOSS_PCT = 10.0
    USE_PREMIUM_TP = True
    TAKE_PROFIT_PCT = 30.0

    # ── flip_exit: if an OPPOSITE-direction signal fires while a position
    #    is open, close the current position immediately (at current option
    #    LTP -- live fills happen in real time, unlike the backtest's
    #    "real option open price at the signal bar" convention) and open
    #    the new, opposite-direction position right away in the same poll.
    #    Same-direction signals while holding are still ignored (unchanged).
    #    This is UNVALIDATED against this exact tuned config -- the 43%
    #    win-rate backtest result was produced WITHOUT flip_exit. Turn this
    #    off (False) to fall back to the originally validated behavior. ──
    USE_FLIP_EXIT = True

    # ── profit lock (trailing SL ladder). Ratchets the effective SL up as
    #    profit crosses each threshold -- never moves back down. List of
    #    (profit_threshold_pct, lock_sl_at_profit_pct) tuples; add more
    #    tiers past 20% the same way. TAKE_PROFIT_PCT above is still the
    #    hard cap regardless of how the lock has ratcheted. LIVE-mode
    #    enforcement uses the documented PATCH /api/v3/positions "Modify
    #    stop-loss and target" endpoint -- see module docstring. ──
    USE_PROFIT_LOCK = True
    PROFIT_LOCK_LADDER = [
        (10.0, 2.0),
        (15.0, 5.0),
        (20.0, 10.0),
    ]

    # ── LIVE order product type. BO/CO are deprecated (Fyers changelog,
    #    02 AUG 2026) -- entries now go through a plain productType order
    #    (INTRADAY applies to all segments incl. NFO options) carrying
    #    stopLoss/takeProfit fields directly, per Fyers' "Place Take Profit
    #    and Stop Loss orders" doc section. ──
    PRODUCT_TYPE = "INTRADAY"

    # ── LIVE entry limit-price offset: must be a limit order (type=1).
    #    We overshoot LTP by this fraction so the entry leg fills instantly. ──
    ENTRY_LIMIT_OFFSET_PCT = 0.5   # 0.5% above LTP for BUY-side option orders

    # ── statutory + brokerage charges (Fyers, NSE options, as of the
    #    Budget 2026-27 STT hike) -- used to compute NET (post-cost) PnL on
    #    every trade record, alongside the raw premium PnL. Update these if
    #    Fyers' pricing or STT rates change. Source: fyers.in/charges-list
    #    and the STT rate confirmed effective 01-Apr-2026. ──
    BROKERAGE_PER_ORDER = 20.0        # flat Rs per executed order (buy leg + sell leg)
    STT_SELL_PCT = 0.15               # % of SELL-side premium value only
    EXCH_TXN_PCT = 0.0355299          # % of (buy value + sell value)
    STAMP_DUTY_BUY_PCT = 0.003        # % of BUY value only
    SEBI_FEE_PER_CRORE = 10.0         # Rs per crore of (buy value + sell value)
    IPFT_PER_CRORE = 50.0             # Rs per crore of (buy value + sell value)
    GST_PCT = 18.0                    # % on (brokerage + exchange charge + SEBI fee + IPFT)

    POLL_INTERVAL_SECS = 60
    DRY_RUN = True

    def __init__(self):
        exp_dt = get_next_tuesday()
        self.EXPIRY_CODE = expiry_code(exp_dt)
        logger.info(f"Weekly expiry code {self.EXPIRY_CODE} | "
                    f"HEIKIN ASHI signals (daily reset) | "
                    f"C1/C2/C3 pattern: declining/rising closes OR body<{self.BODY_RATIO_MAX}xC2, "
                    f"{self.TF_RULE} bars | "
                    f"swing={'lookback='+str(self.SWING_LOOKBACK)+' prox='+str(self.SWING_PROXIMITY_PCT)+'%' if self.USE_SWING_FILTER else 'off'} | "
                    f"C2_doji_filter={self.USE_C2_DOJI_FILTER} (max_body/range={self.C2_MAX_BODY_RANGE_RATIO:.0%}) | "
                    f"SL={'-'+str(self.STOP_LOSS_PCT)+'%' if self.USE_PREMIUM_SL else 'off'} "
                    f"TP={'+'+str(self.TAKE_PROFIT_PCT)+'%' if self.USE_PREMIUM_TP else 'off'} | "
                    f"profit_lock={self.USE_PROFIT_LOCK} {self.PROFIT_LOCK_LADDER if self.USE_PROFIT_LOCK else ''} "
                    f"(live via PATCH /positions, unexercised) | "
                    f"live_exit={'productType='+self.PRODUCT_TYPE+'+DELETE /positions' if not self.DRY_RUN else 'dry-run sim'} | "
                    f"direction-alternating (resets daily) | flip_exit={self.USE_FLIP_EXIT} | DRY_RUN={self.DRY_RUN}")

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

    def exit_order(self, data: dict) -> dict:
        """Cancel an unfilled order by order_id (DELETE /orders/sync/{id}).
        Unrelated to the BO deprecation -- still the correct call for a
        pending order that hasn't filled."""
        try:
            r = self.s.delete(f"{self.T1}/orders/sync/{data['id']}")
            return r.json()
        except Exception as e:
            return {"s": "error", "message": str(e)}

    def modify_position_sl_tp(self, position_id: str, stop_loss: Optional[float] = None,
                               take_profit: Optional[float] = None, qty: Optional[int] = None,
                               leg_type: Optional[int] = None) -> dict:
        """Set/modify the Stop Loss and/or Take Profit on an OPEN POSITION --
        Fyers' documented replacement for BO leg management (changelog
        02 AUG 2026: 'CO & BO Deprecated, Added modify the Stop Loss and
        Take Profit for an existing open position').
        Endpoint: PATCH /api/v3/positions
        Body: {"positionId": position_id, "stopLoss": ..., "takeProfit": ...,
               "qty": ..., "legType": ...}   (1=offset points, 2=percentage)
        Only the fields you pass are changed; omitted fields keep their
        existing value (same call also handles the initial 'Set stop-loss
        and target' -- it's documented as the same API). Passing null for
        stopLoss/takeProfit cancels that leg; this wrapper only sends
        fields that aren't None, so pass 0.0 explicitly if you truly mean
        zero rather than omit."""
        data = {"positionId": position_id}
        if stop_loss is not None:   data["stopLoss"] = stop_loss
        if take_profit is not None: data["takeProfit"] = take_profit
        if qty is not None:         data["qty"] = qty
        if leg_type is not None:    data["legType"] = leg_type
        try:
            r = self.s.patch(f"{self.T1}/positions", json=data)
            return r.json()
        except Exception as e:
            return {"s": "error", "message": str(e)}

    def exit_position(self, position_id: str) -> dict:
        """Close an open position via the documented 'Exit Position - By Id'
        endpoint: DELETE /api/v3/positions, body {"id": "<symbol>-<productType>"}
        e.g. "NSE:NIFTY...CE-INTRADAY". Replaces the old BO-specific
        {"id": f"{symbol}-BO"} convention (BO deprecated 02 AUG 2026)."""
        try:
            data = {"id": position_id}
            r = self.s.delete(f"{self.T1}/positions", json=data)
            return r.json()
        except Exception as e:
            return {"s": "error", "message": str(e)}

    def positions(self) -> dict:
        try:
            return self.s.get(f"{self.T1}/positions").json()
        except Exception as e:
            return {"s": "error", "message": str(e)}

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
    A pivot at raw index piv_i is only confirmed once piv_i+lookback bars
    exist after it, so pivots within `lookback` bars of the array's end are
    never returned -- this is by design, not a bug."""
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
    if none confirmed yet.
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

# --------------------------- SEQUENTIAL CANDLE-REVERSAL DETECTION (+ C2 doji filter) ---------------------------

def detect_signal(o, h, l, c, cfg):
    """
    C1 green, C2 red, C3 (confirmation, bar i) red -- fires if EITHER:
      (a) C1 Close > C2 Close > C3 Close (three declining closes), OR
      (b) C1 Body < BODY_RATIO_MAX x C2 Body
    BUY is the exact mirror: C1 red, C2 green, C3 green, fires if EITHER
      (a) C1 Close < C2 Close < C3 Close (three rising closes), OR
      (b) C1 Body < BODY_RATIO_MAX x C2 Body
    PLUS C1 must sit near a genuine, recently-confirmed support/resistance
    swing zone (SELL -> near a swing high, BUY -> near a swing low) --
    disabled in this tuned config (USE_SWING_FILTER=False).
    PLUS C2 must be a small/doji-style candle:
    body(C2) / (high(C2) - low(C2)) <= cfg.C2_MAX_BODY_RANGE_RATIO."""
    n = len(c)
    if n < 3: return None
    i = n - 1
    c1, c2 = i - 2, i - 1
    green = lambda k: c[k] > o[k]
    red   = lambda k: c[k] < o[k]
    body  = lambda k: abs(c[k] - o[k])
    rng   = lambda k: h[k] - l[k]

    def swing_ok(c1_idx, direction):
        if not cfg.USE_SWING_FILTER: return True
        kind = "high" if direction == "SELL" else "low"
        zone = most_recent_confirmed_zone(h, l, i, cfg.SWING_LOOKBACK, kind)
        return near_zone(l[c1_idx], h[c1_idx], zone, cfg.SWING_PROXIMITY_PCT)

    def c2_doji_ok(c2_idx):
        if not cfg.USE_C2_DOJI_FILTER: return True
        r = rng(c2_idx)
        if r <= 0: return False   # zero-range C2 can't satisfy a body/range ratio
        return body(c2_idx) <= cfg.C2_MAX_BODY_RANGE_RATIO * r

    # ---- SELL: C1 green, C2 red, C3 (confirm) red ----
    if green(c1) and red(c2) and red(i):
        cond_a = c[c1] > c[c2] > c[i]                       # three declining closes
        cond_b = body(c1) < cfg.BODY_RATIO_MAX * body(c2)     # C1 body < BODY_RATIO_MAX x C2 body
        if (cond_a or cond_b) and swing_ok(c1, "SELL") and c2_doji_ok(c2):
            return ("SELL", "sell_pattern", c1, c2)

    # ---- BUY (mirror): C1 red, C2 green, C3 (confirm) green ----
    if cfg.ENABLE_BUY and red(c1) and green(c2) and green(i):
        cond_a = c[c1] < c[c2] < c[i]                       # three rising closes
        cond_b = body(c1) < cfg.BODY_RATIO_MAX * body(c2)
        if (cond_a or cond_b) and swing_ok(c1, "BUY") and c2_doji_ok(c2):
            return ("BUY", "buy_pattern", c1, c2)
    return None

def generate_today_events(day_df: pd.DataFrame, cfg: Config) -> List[dict]:
    """Rebuilds signal events for a single day's bars -- zones reset fresh
    each day. Called fresh every poll with only TODAY's closed bars, which
    the Engine has ALREADY converted to Heikin Ashi before calling this."""
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

# --------------------------- HEIKIN ASHI CONVERSION (reset every trading day) ---------------------------

def to_heikin_ashi(df: pd.DataFrame) -> pd.DataFrame:
    """
    Converts a real OHLC dataframe (DatetimeIndex, columns open/high/low/close
    [/volume]) into Heikin Ashi OHLC, resetting the recursion at the start of
    EVERY calendar day found in the index. In live use the Engine only ever
    passes TODAY's bars in, so this naturally seeds HA_open[0] from today's
    first real candle with no carry-over from yesterday -- same convention
    as the validated backtest.

    Output has the SAME index and the SAME column names (open/high/low/close,
    volume passed through unchanged if present) so it's a drop-in replacement
    for the real dataframe wherever generate_today_events expects one.
    """
    if df.empty:
        return df.copy()

    out = df.copy()
    ha_open  = pd.Series(index=df.index, dtype="float64")
    ha_high  = pd.Series(index=df.index, dtype="float64")
    ha_low   = pd.Series(index=df.index, dtype="float64")
    ha_close = pd.Series(index=df.index, dtype="float64")

    for _day, day_df in df.groupby(df.index.date):
        o = day_df["open"].to_numpy()
        h = day_df["high"].to_numpy()
        l = day_df["low"].to_numpy()
        c = day_df["close"].to_numpy()
        n = len(day_df)

        d_close = (o + h + l + c) / 4.0
        d_open  = pd.Series(index=day_df.index, dtype="float64")
        d_open.iloc[0] = (o[0] + c[0]) / 2.0
        for i in range(1, n):
            d_open.iloc[i] = (d_open.iloc[i - 1] + d_close[i - 1]) / 2.0

        d_open_np = d_open.to_numpy()
        d_high = pd.Series(
            [max(h[i], d_open_np[i], d_close[i]) for i in range(n)],
            index=day_df.index,
        )
        d_low = pd.Series(
            [min(l[i], d_open_np[i], d_close[i]) for i in range(n)],
            index=day_df.index,
        )

        ha_open.loc[day_df.index]  = d_open.values
        ha_high.loc[day_df.index]  = d_high.values
        ha_low.loc[day_df.index]   = d_low.values
        ha_close.loc[day_df.index] = d_close

    out["open"]  = ha_open
    out["high"]  = ha_high
    out["low"]   = ha_low
    out["close"] = ha_close
    return out

# --------------------------- ORDER MANAGER (dry-run + live, post-BO-deprecation) ---------------------------

class OrderManager:
    def __init__(self, fyers: FyersHTTP, cfg: Config):
        self.fyers = fyers; self.cfg = cfg
        self.has = False; self.symbol = None; self.direction = None
        self.entry_price = None; self.entry_time = None
        self.pattern = None; self.oid = None
        self.position_id = None   # "<symbol>-<PRODUCT_TYPE>", used for LIVE modify/exit (post-BO)
        self.locked_sl_pct: Optional[float] = None   # profit-lock ratchet state (None = not yet locked)

    # ── state persistence ──

    def _save_state(self):
        try:
            state = {
                "has": self.has, "symbol": self.symbol, "direction": self.direction,
                "entry_price": self.entry_price,
                "entry_time": self.entry_time.isoformat() if self.entry_time else None,
                "pattern": self.pattern, "oid": self.oid, "position_id": self.position_id,
                "locked_sl_pct": self.locked_sl_pct,
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
        self.locked_sl_pct = None

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
                self.locked_sl_pct = st.get("locked_sl_pct")
                logger.info(f"[OM] Restored {self.symbol} {self.direction} @ {self.entry_price} "
                            f"(oid={self.oid}, position_id={self.position_id}, "
                            f"locked_sl_pct={self.locked_sl_pct})")
        except Exception as e:
            logger.error(f"[OM] restore: {e}")

    # ── profit lock (trailing SL ladder) -- see module docstring ──

    def _profit_pct(self, current_ltp: float) -> float:
        if not self.entry_price:
            return 0.0
        return (current_ltp - self.entry_price) / self.entry_price * 100.0

    def _target_lock_pct(self, profit_pct: float) -> Optional[float]:
        """Highest ladder tier whose threshold profit_pct has reached, or
        None if no tier reached yet. Works with any number of tiers, so
        extending Config.PROFIT_LOCK_LADDER past 20% just works."""
        if not self.cfg.USE_PROFIT_LOCK:
            return None
        locked = None
        for threshold, lock in sorted(self.cfg.PROFIT_LOCK_LADDER):
            if profit_pct >= threshold:
                locked = lock
        return locked

    def effective_sl_price(self) -> float:
        """Current stop-loss price: the profit-locked level if the ladder
        has ratcheted one in, otherwise the original fixed STOP_LOSS_PCT."""
        if self.locked_sl_pct is not None:
            return self.entry_price * (1 + self.locked_sl_pct / 100.0)
        return self.entry_price * (1 - self.cfg.STOP_LOSS_PCT / 100.0)

    def update_profit_lock(self, current_ltp: float) -> bool:
        """DRY-RUN (and general bookkeeping): ratchet locked_sl_pct upward
        as profit crosses PROFIT_LOCK_LADDER thresholds. Never moves down.
        Returns True if the lock was just raised."""
        if not self.has or not self.cfg.USE_PROFIT_LOCK:
            return False
        profit_pct = self._profit_pct(current_ltp)
        target = self._target_lock_pct(profit_pct)
        if target is not None and (self.locked_sl_pct is None or target > self.locked_sl_pct):
            old = self.locked_sl_pct
            self.locked_sl_pct = target
            logger.info(f"[OM] Profit lock ratcheted: {self.symbol} profit={profit_pct:.1f}% "
                        f"-> SL locked at +{target:.1f}% "
                        f"(was {'none' if old is None else f'+{old:.1f}%'})")
            self._save_state()
            return True
        return False

    def update_profit_lock_live(self, current_ltp: float) -> None:
        """
        LIVE-ONLY. Ratchets the position's stop-loss upward as profit
        crosses PROFIT_LOCK_LADDER thresholds, via Fyers' documented
        "Modify stop-loss and target" call -- PATCH /api/v3/positions with
        {"positionId": ..., "stopLoss": <pct>, "legType": 2} (legType 2 =
        percentage, so no manual price math needed). This replaces BO's
        deprecated child-leg-order approach; no orderbook scanning, no
        guessed order IDs -- built directly from Fyers' documented API.
        Still not exercised against a live account from this bot (see
        module docstring) -- validate on a small real position first. On
        any failure this leaves the broker's current SL/TP untouched
        (fail-safe) and retries next poll.
        """
        if not self.has or not self.cfg.USE_PROFIT_LOCK:
            return
        if not self.position_id:
            logger.warning(f"[OM][LIVE] profit_lock: no position_id recorded for {self.symbol} "
                            f"— skipping this poll.")
            return
        profit_pct = self._profit_pct(current_ltp)
        target = self._target_lock_pct(profit_pct)
        if target is None or (self.locked_sl_pct is not None and target <= self.locked_sl_pct):
            return

        logger.info(f"[OM][LIVE] profit_lock: raising SL on {self.position_id} "
                    f"to lock +{target:.1f}% profit (profit now {profit_pct:.1f}%)")
        resp = self.fyers.modify_position_sl_tp(
            position_id=self.position_id, stop_loss=target, leg_type=2)
        logger.info(f"[OM][LIVE] profit_lock: modify_position_sl_tp response: {resp}")

        if isinstance(resp, dict) and resp.get("s") == "ok":
            self.locked_sl_pct = target
            self._save_state()
            logger.info(f"[OM][LIVE] profit_lock: SL raised — locked +{target:.1f}% profit "
                        f"on {self.symbol}")
        else:
            logger.error(f"[OM][LIVE] profit_lock: modify_position_sl_tp FAILED for "
                         f"{self.position_id} — original SL/TP remain in effect. resp={resp}")

    # ── helpers ──

    def _compute_sl_tp_points(self, entry_price: float) -> tuple:
        """Convert STOP_LOSS_PCT / TAKE_PROFIT_PCT into option-premium POINT
        offsets for the order's stopLoss/takeProfit fields. Same convention
        as the pre-deprecation BO legs (same field names carried over into
        the current plain-order + stopLoss/takeProfit flow) -- see module
        docstring's note on order-placement SL/TP units."""
        sl_points = int(round(entry_price * self.cfg.STOP_LOSS_PCT   / 100.0))
        tp_points = int(round(entry_price * self.cfg.TAKE_PROFIT_PCT / 100.0))
        return float(max(1, sl_points)), float(max(1, tp_points))

    def _calc_charges(self, entry_price: float, exit_price: float) -> dict:
        """Round-trip statutory + brokerage charges for one lot (buy + sell),
        using Config's rate constants (see Config docstring block). Returns a
        breakdown dict plus 'total'. All charges are on premium value, i.e.
        entry_price/exit_price * LOT, matching how Fyers actually bills
        options (never on the underlying spot)."""
        qty = self.cfg.LOT
        buy_value  = entry_price * qty
        sell_value = exit_price  * qty
        turnover   = buy_value + sell_value

        brokerage = self.cfg.BROKERAGE_PER_ORDER * 2   # one buy order + one sell order
        stt       = sell_value * self.cfg.STT_SELL_PCT / 100.0
        exch      = turnover   * self.cfg.EXCH_TXN_PCT / 100.0
        stamp     = buy_value  * self.cfg.STAMP_DUTY_BUY_PCT / 100.0
        sebi_fee  = turnover   * self.cfg.SEBI_FEE_PER_CRORE / 1e7
        ipft      = turnover   * self.cfg.IPFT_PER_CRORE / 1e7
        gst       = (brokerage + exch + sebi_fee + ipft) * self.cfg.GST_PCT / 100.0

        total = brokerage + stt + exch + stamp + sebi_fee + ipft + gst
        return {
            "brokerage": round(brokerage, 2), "stt": round(stt, 2), "exch": round(exch, 2),
            "stamp": round(stamp, 2), "sebi_ipft": round(sebi_fee + ipft, 2),
            "gst": round(gst, 2), "total": round(total, 2),
        }

    def _build_trade_rec(self, exit_ltp: float, reason: str) -> dict:
        pnl_pts = float(exit_ltp) - self.entry_price
        pnl_rs  = pnl_pts * self.cfg.LOT
        charges = self._calc_charges(self.entry_price, float(exit_ltp))
        net_pnl_rs = pnl_rs - charges["total"]
        return {
            "direction": self.direction, "option_symbol": self.symbol, "pattern": self.pattern,
            "entry_time": self.entry_time.strftime("%Y-%m-%d %H:%M:%S") if self.entry_time else "",
            "entry_price": round(self.entry_price, 2),
            "exit_price": round(float(exit_ltp), 2), "exit_reason": reason,
            "pnl_pts": round(pnl_pts, 2), "pnl_rs": round(pnl_rs, 2),
            "charges_rs": charges["total"], "charges_breakdown": charges,
            "net_pnl_rs": round(net_pnl_rs, 2),
            "exit_time": datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S"),
            "result": "profit" if net_pnl_rs > 0 else "loss",   # now cost-aware
        }

    def _current_ltp(self) -> float:
        """Best-effort current option LTP (used to approximate exit fill in LIVE
        mode, since exit_position doesn't return a fill price in its response)."""
        if not self.symbol:
            return 0.0
        try:
            resp = self.fyers.quotes(self.symbol)
            v = resp["d"][0]["v"].get("lp")
            return float(v) if v is not None else 0.0
        except Exception:
            return 0.0

    # ── open (dry-run + live, post-BO-deprecation plain order) ──

    def open(self, symbol: str, direction: str, ltp: float, pattern: str) -> bool:
        if self.has:
            logger.warning("[OM] position already open, open() ignored")
            return False

        entry_price = float(ltp)

        if self.cfg.DRY_RUN:
            # Dry-run: simulate fill at LTP; caller (poll loop) will simulate
            # SL/TP by checking live option LTP each tick.
            self.has = True; self.symbol = symbol; self.direction = direction
            self.entry_price = entry_price; self.entry_time = datetime.now(IST)
            self.pattern = pattern
            self.oid = f"DRY-{datetime.now(IST).strftime('%H%M%S')}"
            self.position_id = f"{symbol}-{self.cfg.PRODUCT_TYPE}"   # not used in dry-run, kept for parity
            logger.info(f"[OM][DRY] BUY {symbol} ({direction}) @ {ltp:.2f} | pattern={pattern}")
            self._save_state()
            return True

        # LIVE: place a plain productType order (BO/CO deprecated 02 AUG 2026)
        # with stopLoss/takeProfit fields directly on the order body -- Fyers
        # auto-creates the linked SL/TP legs the same way BO used to.
        sl_points, tp_points = self._compute_sl_tp_points(entry_price)
        limit_price = round(entry_price * (1.0 + self.cfg.ENTRY_LIMIT_OFFSET_PCT / 100.0), 1)

        data = {
            "symbol":       symbol,
            "qty":          int(self.cfg.LOT),
            "type":         1,               # limit
            "side":         1,               # buy
            "productType":  self.cfg.PRODUCT_TYPE,   # "INTRADAY" -- BO/CO no longer valid
            "limitPrice":   limit_price,
            "stopPrice":    0,
            "validity":     "DAY",
            "disclosedQty": 0,
            "offlineOrder": False,
            "stopLoss":     sl_points,
            "takeProfit":   tp_points,
        }
        logger.info(f"[OM][LIVE] Placing order: symbol={symbol} dir={direction} qty={self.cfg.LOT} "
                    f"productType={self.cfg.PRODUCT_TYPE} ltp={entry_price:.2f} "
                    f"limitPrice={limit_price:.1f} SL_pts={sl_points:.0f} TP_pts={tp_points:.0f}")
        resp = self.fyers.place_order(data=data)
        logger.info(f"[OM][LIVE] place_order response: {resp}")

        if isinstance(resp, dict) and resp.get("s") == "ok":
            self.has = True; self.symbol = symbol; self.direction = direction
            self.entry_price = entry_price; self.entry_time = datetime.now(IST)
            self.pattern = pattern
            self.oid = resp.get("id")
            self.position_id = f"{symbol}-{self.cfg.PRODUCT_TYPE}"
            logger.info(f"[OM][LIVE] Order placed successfully, order_id={self.oid}, "
                        f"position_id={self.position_id}")
            self._save_state()
            return True

        logger.error(f"[OM][LIVE] Order rejected: {resp}")
        return False

    # ── close (dry-run uses provided ltp; live uses documented Exit Position) ──

    def close(self, ltp: float, reason: str) -> Optional[dict]:
        """Close position. In dry-run, exit_price = passed ltp. In live,
        calls the documented 'Exit Position - By Id' endpoint (DELETE
        /api/v3/positions, id=position_id -- replaces the old BO-specific
        id=f"{symbol}-BO" convention, BO deprecated 02 AUG 2026) and uses
        the current option LTP as an approximation of the actual fill
        price."""
        if not self.has or not self.symbol:
            logger.warning("[OM] no open position, close() ignored")
            return None

        if self.cfg.DRY_RUN:
            exit_ltp = float(ltp)
            logger.info(f"[OM][DRY] EXIT {self.symbol} @ {exit_ltp:.2f} reason={reason} "
                        f"entry={self.entry_price:.2f}")
            rec = self._build_trade_rec(exit_ltp, reason)
            logger.info(f"[OM][DRY] PnL Rs {rec['pnl_rs']:,.0f} (gross) | charges Rs {rec['charges_rs']:,.0f} | NET Rs {rec['net_pnl_rs']:,.0f}")
            self._reset_state(); self._clear_state()
            return rec

        # LIVE: send Exit Position - By Id (DELETE /api/v3/positions)
        logger.info(f"[OM][LIVE] Exiting position. reason={reason} symbol={self.symbol} "
                    f"position_id={self.position_id} oid={self.oid}")
        resp = self.fyers.exit_position(position_id=self.position_id)
        logger.info(f"[OM][LIVE] exit_position response: {resp}")

        if not (isinstance(resp, dict) and resp.get("s") == "ok"):
            logger.error(f"[OM][LIVE] exit_position failed: {resp}")
            return None

        # Approximate exit fill with current LTP (exit_position doesn't return
        # a fill price). Real fill is available in Fyers tradebook if needed.
        exit_ltp = self._current_ltp()
        logger.info(f"[OM][LIVE] EXIT {self.symbol} @ ~{exit_ltp:.2f} reason={reason} "
                    f"entry={self.entry_price:.2f}")
        rec = self._build_trade_rec(exit_ltp, reason)
        logger.info(f"[OM][LIVE] PnL Rs {rec['pnl_rs']:,.0f} (gross) | charges Rs {rec['charges_rs']:,.0f} | NET Rs {rec['net_pnl_rs']:,.0f}")
        self._reset_state(); self._clear_state()
        return rec

    # ── sync_with_broker: detect broker-side auto-close (SL/TP leg fired) ──

    def sync_with_broker(self) -> Optional[dict]:
        """LIVE-only. Checks broker positions; if our symbol is netQty=0 (or
        absent), the broker auto-closed the position (its SL or TP leg
        fired -- these still exist post-BO-deprecation, just attached to a
        plain productType order instead of a BO). Builds a trade record
        with current option LTP and inferred reason, then clears local
        state. Returns the trade record if auto-closed, None if still open
        or on error. Matches VAM's sync_with_broker convention."""
        if not self.has or not self.symbol:
            return None
        try:
            pos_resp = self.fyers.positions()
            if not isinstance(pos_resp, dict) or pos_resp.get("s") != "ok":
                logger.warning(f"[OM] sync_with_broker: positions API error: {pos_resp}")
                return None

            found = False
            auto_closed = False
            for pos in pos_resp.get("netPositions", []):
                if pos.get("symbol") == self.symbol:
                    found = True
                    if int(pos.get("netQty", 0)) == 0:
                        auto_closed = True
                    break
            if not found:
                # Symbol absent -> also treat as closed
                auto_closed = True

            if not auto_closed:
                return None

            # Broker closed the position. Infer reason from current LTP vs
            # entry SL/TP thresholds (sl_price accounts for a profit-lock
            # ratchet, if one was applied via update_profit_lock_live()).
            exit_ltp = self._current_ltp()
            sl_price = self.effective_sl_price()
            tp_price = self.entry_price * (1 + self.cfg.TAKE_PROFIT_PCT / 100.0)
            if   exit_ltp <= sl_price: reason = "profit_lock_sl" if self.locked_sl_pct is not None else "stop_loss"
            elif exit_ltp >= tp_price: reason = "target"
            else:                      reason = "broker_exit"   # closed but outside SL/TP bands (rare)

            logger.info(f"[OM][LIVE] Broker auto-closed {self.symbol} at ~{exit_ltp:.2f} "
                        f"(entry={self.entry_price:.2f} sl_thresh={sl_price:.2f} tp_thresh={tp_price:.2f}) "
                        f"reason={reason}")
            rec = self._build_trade_rec(exit_ltp, reason)
            logger.info(f"[OM][LIVE] PnL Rs {rec['pnl_rs']:,.0f} (gross) | charges Rs {rec['charges_rs']:,.0f} | NET Rs {rec['net_pnl_rs']:,.0f}")
            self._reset_state(); self._clear_state()
            return rec
        except Exception as e:
            logger.error(f"[OM] sync_with_broker error: {e}")
            return None

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
        logger.info(f"[ENG] ready. interval={self.cfg.POLL_INTERVAL_SECS}s | HEIKIN ASHI signals")
        if self.cfg.DRY_RUN:
            logger.warning("*** DRY-RUN — no real orders ***")
        else:
            logger.warning(f"*** LIVE MODE — {self.cfg.PRODUCT_TYPE} entries + PATCH/DELETE /positions "
                           f"on modify/close (BO deprecated) ***")

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
        """DRY-RUN only: fetch current option LTP, ratchet the profit-lock
        ladder if a new tier was reached, then simulate SL/TP exit against
        the (possibly raised) effective SL."""
        if not self.cfg.DRY_RUN or not self.om.has:
            return False

        opt_ltp = self._ltp(self.om.symbol)
        if opt_ltp is None:
            return False

        self.om.update_profit_lock(opt_ltp)

        sl_price = self.om.effective_sl_price()
        if self.cfg.USE_PREMIUM_SL and opt_ltp <= sl_price:
            reason = "profit_lock_sl" if self.om.locked_sl_pct is not None else "stop_loss"
            rec = self.om.close(opt_ltp, reason)
            if rec: self.trades.append(rec)
            return True

        if self.cfg.USE_PREMIUM_TP and opt_ltp >= self.om.entry_price * (1 + self.cfg.TAKE_PROFIT_PCT / 100.0):
            rec = self.om.close(opt_ltp, "target")
            if rec: self.trades.append(rec)
            return True

        return False

    def _check_live_broker_exits(self) -> bool:
        """LIVE only: detect broker-side auto-close (SL or TP leg fired)."""
        if self.cfg.DRY_RUN or not self.om.has:
            return False
        rec = self.om.sync_with_broker()
        if rec is not None:
            self.trades.append(rec)
            return True
        return False

    def _eod_exit_if_needed(self) -> bool:
        now = datetime.now(IST)
        eod = IST.localize(datetime.combine(now.date(), self.cfg.EOD_EXIT_TIME))
        if now >= eod and self.om.has:
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

        if self._eod_exit_if_needed():
            return

        if not (self.cfg.SESSION_START <= now.time() <= self.cfg.SESSION_END):
            return

        # Position monitoring: dry-run simulates SL/TP; live checks broker for auto-close.
        if self.cfg.DRY_RUN:
            if self._check_dryrun_exits():
                return
        else:
            if self._check_live_broker_exits():
                return
            # Profit lock (LIVE -- see module docstring): try to raise the
            # position's SL via PATCH /positions if a new ladder tier was reached.
            if self.om.has:
                lock_ltp = self._ltp(self.om.symbol)
                if lock_ltp is not None:
                    self.om.update_profit_lock_live(lock_ltp)

        df = self._fetch_today_bars()
        df = self._drop_forming_bar(df)
        if df.empty:
            return

        # Convert TODAY's real closed bars to Heikin Ashi (resets fresh each
        # day since df only ever contains today's bars) BEFORE signal scan.
        ha_df = to_heikin_ashi(df)

        events = generate_today_events(ha_df, self.cfg)
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

        # Already in a position: same-direction signals are still ignored.
        # An OPPOSITE-direction signal triggers flip_exit (if enabled) --
        # close now at current LTP, then fall through to open the new
        # position below in this same poll. (added 2026-09-01 -- see
        # Config.USE_FLIP_EXIT docstring; unvalidated against this tuned
        # config's backtest, which did not use flip_exit.)
        if self.om.has:
            if not (self.cfg.USE_FLIP_EXIT and next_ev["dir"] != self.om.direction):
                logger.info(f"[ENG] Signal {next_ev['dir']} @ {next_ev['timestamp']} ignored — already in position.")
                return

            flip_ltp = self._ltp(self.om.symbol)
            if flip_ltp is None:
                logger.warning(f"[ENG] flip_exit: no LTP for {self.om.symbol} — "
                                f"can't close this poll, signal skipped (will re-check next poll).")
                return

            logger.info(f"[ENG] flip_exit: opposite signal {next_ev['dir']} @ {next_ev['timestamp']} "
                        f"while holding {self.om.direction} {self.om.symbol} — closing then flipping.")
            rec = self.om.close(flip_ltp, "flip_exit")
            if rec:
                self.trades.append(rec)
            else:
                logger.error(f"[ENG] flip_exit: close() failed (broker rejected / error) — position "
                             f"may still be open. Aborting the flip this poll; will retry on the next "
                             f"opposite signal or normal SL/TP/eod_exit path.")
                return
            # falls through to open the new (opposite-direction) position below

        # DIRECTION ALTERNATION: no CE after CE, no PE after PE
        # (a flip_exit re-entry is always the opposite direction of the
        # trade that was just closed, so it never collides with this rule)
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
                    f"pattern={next_ev['name']} (HA candle) -> entering {sym} @ {ltp:.2f}")
        opened = self.om.open(sym, next_ev["dir"], ltp, next_ev["name"])
        if opened:
            self._last_trade_dir = next_ev["dir"]

    def run(self):
        logger.info("[ENG] starting Sequential Candle Reversal (Heikin Ashi) poll loop")
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
            # flatten the nested charges_breakdown dict (brokerage/stt/exch/
            # stamp/sebi_ipft/gst) into its own columns, drop the dict column
            # itself (Excel can't hold a dict cell), keep charges_rs as-is.
            if "charges_breakdown" in df.columns:
                brk = pd.json_normalize(df["charges_breakdown"]).add_prefix("chg_")
                df = pd.concat([df.drop(columns=["charges_breakdown"]), brk], axis=1)
            df["cum_pnl_rs"] = df["pnl_rs"].cumsum()
            df["cum_net_pnl_rs"] = df["net_pnl_rs"].cumsum()
            d = datetime.now(IST).strftime("%Y%m%d")
            path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
            os.makedirs(path, exist_ok=True)
            fp = os.path.join(path, f"pattern_ha_trades_{d}.xlsx")
            df.to_excel(fp, index=False)
            gross = df["pnl_rs"].sum(); charges = df["charges_rs"].sum(); net = df["net_pnl_rs"].sum()
            wins = (df["net_pnl_rs"] > 0).sum()   # win/loss now judged NET of costs
            logger.info(f"[ENG] exported {len(df)} trades -> {fp} | "
                        f"gross Rs {gross:,.0f} | charges Rs {charges:,.0f} | NET Rs {net:,.0f} | "
                        f"wins(net) {wins}/{len(df)}")
        except Exception as e:
            logger.error(f"export: {e}")


if __name__ == "__main__":
    Engine().run()
