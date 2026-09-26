"""
Sequential 3-candle REVERSAL option backtest.

PATTERN (replaces the earlier opposite-color-C1/C2 + break-of-C1-open rule
entirely -- this is now the only pattern definition):
  SELL: C1 green, C2 red, C3 (confirmation, the last bar) red. Fires if
        EITHER (a) C1 Close > C2 Close > C3 Close (three declining closes),
        OR (b) C1 Body < BODY_RATIO_MAX x C2 Body (default 1.2x).
  BUY:  exact mirror -- C1 red, C2 green, C3 green. Fires if EITHER
        (a) C1 Close < C2 Close < C3 Close (three rising closes), OR
        (b) C1 Body < BODY_RATIO_MAX x C2 Body.
PLUS ONE added filter: entries must occur AT a genuine support/resistance
swing zone, using detect_fractal_pivots() copied VERBATIM from the
Break/Retest bot (and from PatternPoll.py's own swing filter, added in
lockstep -- all three files must stay identical on this function; do not
edit one without updating the others).

Still NOT layered on top:
  - EMA cross-and-close filter (USE_EMA_CROSS_FILTER)
  - Volume filter (USE_VOLUME_FILTER / VOLUME_LOOKBACK / VOL_SMA_LEN)
  - Healthy-body requirement (HEALTHY_BODY_FRAC)
  - Big-C2 fast-entry alternate mode (USE_BIG_C2_FAST_ENTRY / BIG_C2_BODY_MULT)

SWING ZONE FILTER: C1 must sit near the most recently CONFIRMED swing
zone (SWING_LOOKBACK bars on both sides to confirm, per the standard fractal
pivot definition):
  SELL (bearish reversal) -> C1 must be near a confirmed swing HIGH
                              (i.e. reversing off resistance)
  BUY  (bullish reversal) -> C1 must be near a confirmed swing LOW
                              (i.e. reversing off support)
A pivot at raw index piv_i is only confirmed once piv_i+SWING_LOOKBACK bars
exist after it. Since C1 is only 1-2 bars before the confirmation candle,
C1 itself is essentially never old enough to BE a freshly-confirmed pivot --
so this checks proximity to the most recent zone confirmed using bars up to
the CONFIRMATION candle, not requiring C1 to be the pivot bar itself.

RISK MANAGEMENT is deliberately minimal -- fixed premium SL/TP, checked
minute-by-minute on 1-min option data:
  1. Fixed 10% premium stop-loss: STOP_LOSS_PCT=10.0.
  2. Fixed 20% premium target: TAKE_PROFIT_PCT=20.0.
  Plus EOD forced exit. No candle SL (no more C1 high/low reference on the
  underlying), no profit-lock trailing, no flip-exit -- a position no longer
  reverses on an opposite-direction signal; while a position is open, ALL
  new signals (same or opposite direction) are simply ignored. A trade only
  ever closes via stop_loss, target, eod_exit, or (rarely) data_exhausted --
  a safety net for the case where an option's own data runs out (contract
  past its usable history) before SL/TP/EOD ever fires, so a position can't
  silently sit "open" forever and block every future entry.

DIRECTION ALTERNATION: consecutive trades may not share the same direction --
no CE (BUY) directly after another CE, no PE (SELL) directly after another
PE. This tracking is global across the whole backtest run (not reset daily).
NOTE: this is a KNOWN, confirmed point of divergence from the live version of
this strategy, which resets this tracker every trading day. Keep that in
mind when comparing this backtest's numbers to live trade logs -- they will
not necessarily agree on which signal is "first" after a day boundary.
"""
import pandas as pd, numpy as np, requests, logging, os, calendar, pytz, urllib3, warnings, time
from datetime import datetime, timedelta, time as dtime
from typing import Optional, Dict
from expiry_resolver import ExpiryResolver

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
warnings.filterwarnings("ignore")
IST = pytz.timezone("Asia/Kolkata")

logging.basicConfig(level=logging.INFO,
                     format="%(asctime)s - PATBacktest - %(levelname)s - %(message)s",
                     datefmt="%Y-%m-%d %H:%M:%S")
logger = logging.getLogger("PATBacktest")

# ---------- expiry helpers (identical to PatternPoll.py) ----------
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

# ---------- config ----------
class Config:
    UNDERLYING = "NSE:NIFTY50-INDEX"; BASE_RES = "1"; HIST_DAYS = 10
    TF_RULE = "1min"

    # ── candle reversal pattern: fixed 3-candle check (C1,C2,C3=confirm).
    #    CONFIRM_START/END no longer define a search range (the old
    #    break-of-open pattern did) -- they're kept only as the "need at
    #    least this many bars" gate, always 3 for this fixed-width pattern. ──
    CONFIRM_START = 3
    CONFIRM_END = 3
    ENABLE_BUY = True
    BODY_RATIO_MAX = 1.2         # SELL/BUY fires if C1 body < this many times C2's body (OR declining/rising closes)

    # ── swing zone (support/resistance) entry filter -- NEW ──
    USE_SWING_FILTER = True   # True = C1 must sit near a confirmed swing zone (high for SELL, low for BUY)
    SWING_LOOKBACK = 10          # bars needed on both sides to confirm a pivot
    SWING_PROXIMITY_PCT = 0.5    # how close (% of zone price) C1 must be to count as "near"

    STRIKE_STEP = 100; LOT = 65; MONEYNESS = "ITM"
    SESSION_START = dtime(9, 20); SESSION_END = dtime(15, 30)
    SIGNAL_START_TIME = dtime(9, 35)
    TRADE_START = dtime(9, 30); TRADE_END = dtime(15, 0); EOD_EXIT_TIME = dtime(15, 20)

    # ── risk management: fixed premium SL + fixed premium TP, both checked
    #    minute-by-minute on option data. No candle SL, no profit-lock
    #    trailing, no flip-on-opposite-signal -- a position only ever closes
    #    via stop_loss, target, or eod_exit. ──
    USE_PREMIUM_SL = True
    STOP_LOSS_PCT = 10.0

    USE_PREMIUM_TP = True
    TAKE_PROFIT_PCT = 20.0

    USE_PROFIT_LOCK = True
    PROFIT_LOCK_TRIGGER_PCT = 12.0
    PROFIT_LOCK_FLOOR_PCT = 12.0
    TRAIL_START_PCT = 20.0
    TRAIL_BEHIND_PCT = 5.0

    def __init__(self):
        self.EXPIRY_CODE = expiry_code(get_next_tuesday())
        logger.info(f"Weekly expiry code {self.EXPIRY_CODE} | "
                    f"C1/C2/C3 pattern: declining/rising closes OR body<{self.BODY_RATIO_MAX}xC2, "
                    f"{self.TF_RULE} bars | "
                    f"swing={'lookback='+str(self.SWING_LOOKBACK)+' prox='+str(self.SWING_PROXIMITY_PCT)+'%' if self.USE_SWING_FILTER else 'off'} | "
                    f"SL={'-'+str(self.STOP_LOSS_PCT)+'%' if self.USE_PREMIUM_SL else 'off'} "
                    f"TP={'+'+str(self.TAKE_PROFIT_PCT)+'%' if self.USE_PREMIUM_TP else 'off'} | "
                    f"direction-alternating")

# ---------- fyers HTTP client ----------
class Fyers:
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

# ---------- fractal pivot detection (VERBATIM from breakretest_poll_live.py / PatternPoll.py) ----------
def detect_fractal_pivots(h, l, lookback: int):
    """Direct port of Pine's ta.pivotlow(low, bb, bb) / ta.pivothigh(high, bb, bb).
    MUST stay identical across breakretest_poll_live.py, PatternPoll.py, and
    this file. A pivot at raw index piv_i is only confirmed once piv_i+lookback
    bars exist after it, so pivots within `lookback` bars of the array's end
    are never returned -- this is by design, not a bug."""
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

# ---------- sequential candle-reversal detection (RAW pattern + swing zone filter) ----------
def detect_signal(o, h, l, c, cfg):
    """
    C1 green, C2 red, C3 (confirmation, bar i) red -- fires if EITHER:
      (a) C1 Close > C2 Close > C3 Close (three declining closes), OR
      (b) C1 Body < 1.2x C2 Body
    BUY is the exact mirror: C1 red, C2 green, C3 green, fires if EITHER
      (a) C1 Close < C2 Close < C3 Close (three rising closes), OR
      (b) C1 Body < 1.2x C2 Body
    PLUS C1 must sit near a genuine, recently-confirmed support/resistance
    swing zone (SELL -> near a swing high, BUY -> near a swing low).
    No EMA filter, no volume filter, no healthy-body requirement, no
    "must be colored beyond C1/C2/C3" requirement, no fast-entry mode.
    """
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
        cond_a = c[c1] > c[c2] > c[i]                    # three declining closes
        cond_b = body(c1) < cfg.BODY_RATIO_MAX * body(c2)  # C1 body < 1.2x C2 body
        if (cond_a or cond_b) and swing_ok(c1, "SELL"):
            return ("SELL", "sell_pattern", c1, c2)

    # ---- BUY (mirror): C1 red, C2 green, C3 (confirm) green ----
    if cfg.ENABLE_BUY and red(c1) and green(c2) and green(i):
        cond_a = c[c1] < c[c2] < c[i]                    # three rising closes
        cond_b = body(c1) < cfg.BODY_RATIO_MAX * body(c2)
        if (cond_a or cond_b) and swing_ok(c1, "BUY"):
            return ("BUY", "buy_pattern", c1, c2)
    return None

# ---------- backtest bot ----------
class PatternBacktestBot:
    def __init__(self):
        self.cfg = Config()
        self.fy = self._init_fyers()
        self.expiry_resolver = ExpiryResolver()
        self._fallback_expiry_count = 0
        self._skipped_no_data = 0
        self._skipped_same_direction = 0

    def _init_fyers(self) -> Fyers:
        for f in ["fyers_appid.txt", "fyers_token.txt"]:
            if not os.path.exists(f): raise FileNotFoundError(f)
        app = open("fyers_appid.txt").read().strip()
        tok = open("fyers_token.txt").read().strip()
        fy = Fyers(app, tok); p = fy.profile()
        if p.get("s") != "ok":
            raise ConnectionError(f"Fyers auth failed — re-run GenerateToken.py.\nResponse: {p}")
        logger.info(f"Fyers connected — {p['data']['name']} OK")
        return fy

    def _fetch_history(self, symbol: str, days: int, res: str = None) -> pd.DataFrame:
        res = res or self.cfg.BASE_RES
        end = datetime.now(IST); start = end - timedelta(days=days)
        resp = self.fy.history(symbol, res, start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"))
        if resp.get("s") != "ok":
            raise ValueError(f"History API error for {symbol}: {resp}")
        candles = resp.get("candles", [])
        if not candles:
            raise ValueError(f"No historical data for {symbol}")
        df = pd.DataFrame(candles, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s", utc=True).dt.tz_convert(IST)
        df = df.drop_duplicates("timestamp").sort_values("timestamp").set_index("timestamp")
        return df

    def get_index_data(self) -> pd.DataFrame:
        df = self._fetch_history(self.cfg.UNDERLYING, self.cfg.HIST_DAYS)
        df = df.iloc[df.index.indexer_between_time(self.cfg.SESSION_START, self.cfg.SESSION_END)]
        logger.info(f"Loaded NIFTY50 1-min index data: {len(df)} rows")
        return df

    def _fetch_option_1min(self, symbol: str) -> pd.DataFrame:
        return self._fetch_history(symbol, self.cfg.HIST_DAYS, res="1").sort_index()

    def _opt_symbol_for_date(self, spot: float, direction: str, signal_date) -> str:
        """Resolves the weekly expiry ACTIVE ON THE SIGNAL'S OWN DATE via
        ExpiryResolver (matching Fyers' live symbol master), instead of a
        single fixed expiry computed once at bot init."""
        c = self.cfg
        exp_code, source = self.expiry_resolver.resolve_expiry_code(
            signal_date, expiry_code, get_next_tuesday)
        if source == "fallback_tuesday":
            self._fallback_expiry_count += 1
        atm = int(round(spot / c.STRIKE_STEP) * c.STRIKE_STEP); itm = c.MONEYNESS == "ITM"
        if direction == "BUY":
            strike = atm - c.STRIKE_STEP if itm else atm; typ = "CE"
        else:
            strike = atm + c.STRIKE_STEP if itm else atm; typ = "PE"
        return f"NSE:NIFTY{exp_code}{strike:05d}{typ}"

    # NOTE: _sl_level() (candle-based SL reference lookup) removed -- this
    # backtest now uses fixed premium SL/TP instead, so C1's high/low is no
    # longer needed as a stop reference. c1_high/c1_low are still recorded
    # in signal events purely as diagnostic info.

    # ── signal generation: replays the pattern bar-by-bar ──

    def generate_signals(self, df_1min: pd.DataFrame) -> list:
        """Returns a list of raw signal events (dicts) in chronological order,
        one per NEW closed bar where detect_signal fires."""
        c = self.cfg
        events = []

        for day, day_df in df_1min.groupby(df_1min.index.date):
            r = day_df.resample(c.TF_RULE).agg(
                {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
            ).dropna()
            if r.index.duplicated().any():
                n_dupes = int(r.index.duplicated().sum())
                logger.warning(f"[{day}] {n_dupes} duplicate timestamp(s) in resampled bars -- "
                                f"check raw Fyers fetch for overlapping/duplicate candles")
            if len(r) < c.CONFIRM_START:
                continue

            o = r["open"].to_numpy(); h = r["high"].to_numpy()
            l = r["low"].to_numpy(); cl = r["close"].to_numpy()

            for i in range(c.CONFIRM_START - 1, len(r)):
                bar_close_time = r.index[i] + pd.Timedelta(c.TF_RULE)
                ts_time = bar_close_time.time()
                if not (c.SESSION_START <= ts_time <= c.SESSION_END):
                    continue
                if ts_time < c.SIGNAL_START_TIME:
                    continue

                sig = detect_signal(o[:i+1], h[:i+1], l[:i+1], cl[:i+1], c)
                if sig is None:
                    continue
                direction, name, c1, c2 = sig
                events.append({
                    "timestamp": bar_close_time,
                    "bar_idx": i,
                    "dir": direction,
                    "name": name,
                    "spot": float(cl[i]),
                    "c1_open": float(o[c1]), "c1_close": float(cl[c1]),
                    "c1_high": float(h[c1]), "c1_low": float(l[c1]),
                    "c1_time": r.index[c1].strftime("%Y-%m-%d %H:%M"),
                    "c1_color": "green" if cl[c1] > o[c1] else "red",
                    "c2_open": float(o[c2]), "c2_close": float(cl[c2]),
                    "c2_time": r.index[c2].strftime("%Y-%m-%d %H:%M"),
                    "c2_color": "green" if cl[c2] > o[c2] else "red",
                    "confirm_open": float(o[i]), "confirm_close": float(cl[i]),
                    "confirm_time": r.index[i].strftime("%Y-%m-%d %H:%M"),
                })

        logger.info(f"Generated {len(events)} raw signal events across all days")
        return events

    # ── simulate the position + risk management ──

    def run_backtest(self) -> pd.DataFrame:
        c = self.cfg
        print("Starting Sequential 3-Candle Reversal Backtest -- SWING ZONE FILTER, DIRECTION-ALTERNATING (ITM contracts)...")
        df_1min = self.get_index_data()
        events = self.generate_signals(df_1min)
        if not events:
            print("No signals generated")
            return pd.DataFrame()

        events.sort(key=lambda e: e["timestamp"])

        option_cache: Dict[str, pd.DataFrame] = {}

        def get_option_df(sym):
            if sym not in option_cache:
                time.sleep(0.25)
                try:
                    option_cache[sym] = self._fetch_option_1min(sym)
                except ValueError as e:
                    msg = str(e)
                    if "429" in msg or "request limit" in msg.lower():
                        logger.warning(f"Rate limited on {sym}, waiting 3s and retrying once...")
                        time.sleep(3.0)
                        try:
                            option_cache[sym] = self._fetch_option_1min(sym)
                        except Exception as e2:
                            logger.warning(f"Skipping {sym} (retry failed): {e2}")
                            option_cache[sym] = pd.DataFrame()
                    else:
                        logger.warning(f"Skipping {sym} (no data / invalid symbol): {e}")
                        option_cache[sym] = pd.DataFrame()
                except Exception as e:
                    logger.warning(f"Skipping {sym} (unexpected fetch error): {e}")
                    option_cache[sym] = pd.DataFrame()
            return option_cache[sym]

        # state (mirrors OM)
        has = False; symbol = None; direction = None; entry_price = None
        entry_time = None; pattern = None
        sig_debug = None   # holds the triggering event's C1/C2/confirmation diagnostics
        last_trade_dir = None   # DIRECTION ALTERNATION: direction of the most recently OPENED trade

        trades = []
        fired_timestamps = {}   # date -> last acted event timestamp

        idx_1min = df_1min

        def close_trade(exit_time, exit_price, reason):
            nonlocal has, symbol, direction, entry_price, entry_time, pattern
            nonlocal sig_debug
            pnl_pts = exit_price - entry_price
            pnl_rs = pnl_pts * c.LOT
            trades.append({
                "direction": direction, "pattern": pattern, "option_symbol": symbol,
                "entry_time": entry_time, "entry_price": round(entry_price, 2),
                "exit_time": exit_time, "exit_price": round(exit_price, 2),
                "exit_reason": reason,
                "pnl_pts": round(pnl_pts, 2), "pnl_rs": round(pnl_rs, 2),
                "result": "profit" if pnl_pts > 0 else "loss",
                "c1_time": sig_debug["c1_time"] if sig_debug else None,
                "c1_open": sig_debug["c1_open"] if sig_debug else None,
                "c1_close": sig_debug["c1_close"] if sig_debug else None,
                "c1_color": sig_debug["c1_color"] if sig_debug else None,
                "c2_time": sig_debug["c2_time"] if sig_debug else None,
                "c2_open": sig_debug["c2_open"] if sig_debug else None,
                "c2_close": sig_debug["c2_close"] if sig_debug else None,
                "c2_color": sig_debug["c2_color"] if sig_debug else None,
                "confirm_time": sig_debug["confirm_time"] if sig_debug else None,
                "confirm_open": sig_debug["confirm_open"] if sig_debug else None,
                "confirm_close": sig_debug["confirm_close"] if sig_debug else None,
            })
            has = False; symbol = None; direction = None; entry_price = None
            entry_time = None; pattern = None
            sig_debug = None

        def opt_price_at_or_after(opt_df, ts, field="close"):
            if opt_df.empty: return None, None
            pos = opt_df.index.searchsorted(ts, side="right" if field == "open" else "left")
            if field == "open":
                if pos >= len(opt_df): return None, None
                return float(opt_df.iloc[int(pos)]["open"]), opt_df.index[int(pos)]
            else:
                if pos >= len(opt_df):
                    pos = len(opt_df) - 1
                return float(opt_df.iloc[int(pos)]["close"]), opt_df.index[int(pos)]

        for ev in events:
            ts = ev["timestamp"]
            day = ts.date()
            eod_ts = IST.localize(datetime.combine(day, c.EOD_EXIT_TIME))

            # ── forced EOD exit for any position still open from a previous day ──
            if has and entry_time is not None and entry_time.date() < day:
                opt_df = get_option_df(symbol)
                prev_eod = IST.localize(datetime.combine(entry_time.date(), c.EOD_EXIT_TIME))
                px, pxt = opt_price_at_or_after(opt_df, prev_eod)
                if px is not None:
                    close_trade(pxt, px, "eod_exit")

            # ── risk management: walk underlying+option bars between last check and this event ──
            # Priority each bar: stop_loss/target first (most specific), then
            # EOD forced exit at 15:20 if neither has fired yet that day, then
            # the data_exhausted safety net (options expire weekly -- if this
            # option's own data runs out before SL/TP/EOD, force-close at the
            # last known price so the position doesn't sit open forever and
            # silently block every future entry for the rest of the run).
            if has:
                opt_df = get_option_df(symbol)
                if opt_df.empty or ts > opt_df.index[-1]:
                    last_ts = opt_df.index[-1] if not opt_df.empty else ts
                    last_px = float(opt_df.iloc[-1]["close"]) if not opt_df.empty else entry_price
                    close_trade(last_ts, last_px, "data_exhausted")
                else:
                    window = idx_1min[(idx_1min.index > entry_time) & (idx_1min.index <= ts)]
                    for bar_ts, bar in window.iterrows():
                        if not has: break
                        if bar_ts >= eod_ts:
                            px, pxt = opt_price_at_or_after(opt_df, eod_ts)
                            if px is not None: close_trade(pxt, px, "eod_exit")
                            break
                        if has and c.USE_PREMIUM_SL:
                            px, pxt = opt_price_at_or_after(opt_df, bar_ts)
                            if px is not None and px <= entry_price * (1 - c.STOP_LOSS_PCT / 100.0):
                                close_trade(pxt, px, "stop_loss"); continue
                        if has and c.USE_PREMIUM_TP:
                            px, pxt = opt_price_at_or_after(opt_df, bar_ts)
                            if px is not None and px >= entry_price * (1 + c.TAKE_PROFIT_PCT / 100.0):
                                close_trade(pxt, px, "target"); continue

            if ts >= eod_ts and has:
                opt_df = get_option_df(symbol)
                px, pxt = opt_price_at_or_after(opt_df, eod_ts)
                if px is not None: close_trade(pxt, px, "eod_exit")

            # ── act on the signal event ──
            if fired_timestamps.get(day) == ev["timestamp"]:
                continue  # already acted on this bar/timestamp

            # NO FLIP LOGIC: while a position is open, ALL new signals are ignored.
            if has:
                fired_timestamps[day] = ev["timestamp"]
                continue

            desired = ev["dir"]

            # DIRECTION ALTERNATION: no CE after CE, no PE after PE. If the
            # last OPENED trade was the same direction as this signal, skip
            # it -- only an opposite-direction signal may open the next trade.
            if last_trade_dir is not None and desired == last_trade_dir:
                self._skipped_same_direction += 1
                fired_timestamps[day] = ev["timestamp"]
                continue

            in_win = c.TRADE_START <= ts.time() <= c.TRADE_END

            if in_win:
                sym = self._opt_symbol_for_date(ev["spot"], desired, ts.date())
                opt_df = get_option_df(sym)
                entry_px, entry_ts_actual = opt_price_at_or_after(opt_df, ts, field="open")
                if entry_px is None:
                    self._skipped_no_data += 1
                    fired_timestamps[day] = ev["timestamp"]
                    continue
                entry_px *= 1.005  # small slippage, matches other backtests' convention

                has = True; symbol = sym; direction = desired; entry_price = entry_px
                sig_debug = {k: ev[k] for k in (
                    "c1_time", "c1_open", "c1_close", "c1_color",
                    "c2_time", "c2_open", "c2_close", "c2_color",
                    "confirm_time", "confirm_open", "confirm_close",
                )}
                entry_time = entry_ts_actual; pattern = ev["name"]
                last_trade_dir = desired   # update alternation tracker on entry

            fired_timestamps[day] = ev["timestamp"]

        # ── final flush: close anything still open at the very end ──
        if has:
            opt_df = get_option_df(symbol)
            last_ts = opt_df.index[-1] if not opt_df.empty else entry_time
            px, pxt = opt_price_at_or_after(opt_df, last_ts)
            if px is not None:
                close_trade(pxt, px, "backtest_end")

        trades_df = pd.DataFrame(trades)
        logger.info(f"Signals skipped due to missing/invalid option data: {self._skipped_no_data}")
        logger.info(f"Signals skipped (same direction as last trade): {self._skipped_same_direction}")
        logger.info(f"Signals with UNVERIFIED expiry (not found in Fyers' live master): {self._fallback_expiry_count}")
        if trades_df.empty:
            print("No trades executed")
            print(f"  Signals skipped (no option data): {self._skipped_no_data}")
            return trades_df

        trades_df["cum_pnl_rs"] = trades_df["pnl_rs"].cumsum()

        wins = int((trades_df["result"] == "profit").sum())
        losses = int((trades_df["result"] == "loss").sum())
        total = len(trades_df)
        net = trades_df["pnl_rs"].sum()
        wr = round(wins / total * 100, 1) if total > 0 else 0.0

        csv_path = "Pattern_SwingZone_Alternating_NIFTY_ITM_Backtest.csv"
        trades_df.to_csv(csv_path, index=False, encoding="utf-8")

        with open(csv_path, "a", encoding="utf-8") as f:
            for line in [
                "", "METRICS,VALUE",
                f"Total Trades,{total}", f"Wins,{wins}", f"Losses,{losses}",
                f"Win Rate (%),{wr}", f"Net PnL (Rs),{net:.2f}",
                f"Signals Skipped (no option data),{self._skipped_no_data}",
                f"Signals Skipped (same direction as last trade),{self._skipped_same_direction}",
                f"Signals Using Fallback Expiry Guess,{self._fallback_expiry_count}",
                f"TF Rule,{c.TF_RULE}", f"Confirm Range,{c.CONFIRM_START}-{c.CONFIRM_END}",
                f"Stop Loss %,{c.STOP_LOSS_PCT if c.USE_PREMIUM_SL else 'off'}",
                f"Take Profit %,{c.TAKE_PROFIT_PCT if c.USE_PREMIUM_TP else 'off'}",
                f"Swing Filter,{c.USE_SWING_FILTER}", f"Swing Lookback,{c.SWING_LOOKBACK}",
                f"Swing Proximity %,{c.SWING_PROXIMITY_PCT}",
                f"Lot Size,{c.LOT}",
            ]:
                f.write(line + "\n")

        print(f"\n{'='*55}")
        print(f"  Contract Type       : {c.MONEYNESS}")
        print(f"  Total Trades        : {total}")
        print(f"  Signals Skipped     : {self._skipped_no_data} (no option data)")
        print(f"  Same-Dir Skipped    : {self._skipped_same_direction} (no CE-after-CE/PE-after-PE)")
        print(f"  Unverified Expiries : {self._fallback_expiry_count}")
        print(f"  Win Rate            : {wr}%")
        print(f"  Net PnL             : Rs {net:,.2f}")
        print(f"{'='*55}")
        print(f"Results saved -> {csv_path}")
        return trades_df


# --------------------------- MAIN ---------------------------

if __name__ == "__main__":
    bot    = PatternBacktestBot()
    trades = bot.run_backtest()