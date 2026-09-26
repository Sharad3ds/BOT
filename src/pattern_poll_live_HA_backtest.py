"""
Backtest for pattern_poll_live.py (Sequential 3-Candle Reversal + Swing-Zone)
-- HEIKIN ASHI VARIANT, FULLY INDEPENDENT (no import of pattern_poll_live.py).

This file is self-contained: Config, detect_fractal_pivots,
most_recent_confirmed_zone, near_zone, detect_signal, and
generate_today_events below are copied VERBATIM from pattern_poll_live.py
(as pasted by the user on 2026-08-30) so the pattern + zone rules are
byte-identical to the live bot's. If pattern_poll_live.py's logic changes
later, this file will NOT pick it up automatically -- re-sync those
functions here by hand.

The only strategic difference from a real-candle backtest is WHAT DATA is
fed into generate_today_events: this file converts the real 1-min index
bars to Heikin Ashi (reset every trading day, so HA opens are never
distorted by the previous day's gap) BEFORE running the unchanged
signal/zone logic on them.

    real-candle backtest : generate_today_events(real_1min_day_df, cfg)
    this HA backtest     : generate_today_events(heikin_ashi_day_df, cfg)

Everything downstream of signal generation -- option symbol selection,
entry fills, premium SL/TP, EOD exit, direction alternation -- is
UNCHANGED and still uses REAL option premiums (Heikin Ashi has no real
traded price, so it can only ever shape the signal, never the P&L).

HEIKIN ASHI FORMULA (reset every trading day):
    HA_close[0] = (O+H+L+C)/4  of the day's first real candle
    HA_open[0]  = (O+C)/2      of the day's first real candle   (== real O)
    HA_high[0]  = real high[0]
    HA_low[0]   = real low[0]
    for i > 0 (same day):
        HA_close[i] = (O[i]+H[i]+L[i]+C[i]) / 4          (real OHLC)
        HA_open[i]  = (HA_open[i-1] + HA_close[i-1]) / 2
        HA_high[i]  = max(H[i], HA_open[i], HA_close[i])
        HA_low[i]   = min(L[i],  HA_open[i], HA_close[i])

Config OVERRIDES: edit BacktestConfig below without touching anything else.

KNOWN LIMITATION: EXPIRY_CODE is computed once at init using
get_next_tuesday(today), so ALL trades in the backtest are simulated
against the CURRENT week's option symbols -- fine for short recent windows,
invalid for older backtest dates.
"""
import os, sys, calendar, warnings, logging, requests, urllib3, pytz
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, time as dtime
from typing import Dict, Optional, List

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
warnings.filterwarnings("ignore")
IST = pytz.timezone("Asia/Kolkata")

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - PatternLiveBacktestHA - %(levelname)s - %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S")
logger = logging.getLogger("PatternLiveBacktestHA")


# ══════════════════════════════════════════════════════════════════════════════
#  EXPIRY HELPERS -- verbatim from pattern_poll_live.py
# ══════════════════════════════════════════════════════════════════════════════
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


# ══════════════════════════════════════════════════════════════════════════════
#  CONFIG -- verbatim from pattern_poll_live.py (single source of strategy
#  parameters). BacktestConfig below adds only backtest-window overrides.
# ══════════════════════════════════════════════════════════════════════════════
class Config:
    UNDERLYING = "NSE:NIFTY50-INDEX"; BASE_RES = "5"
    TF_RULE = "5min"

    # ── candle reversal pattern: fixed 3-candle check (C1,C2,C3=confirm).
    #    CONFIRM_START kept only as the "need at least this many bars" gate. ──
    CONFIRM_START = 3
    ENABLE_BUY = True
    BODY_RATIO_MAX = 0.5      # SELL/BUY fires if C1 body < this many times C2's body (OR declining/rising closes)

    # ── swing zone (support/resistance) entry filter ──
    USE_SWING_FILTER = False
    SWING_LOOKBACK = 10          # bars needed on both sides to confirm a pivot
    SWING_PROXIMITY_PCT = 0.5    # how close (% of zone price) C1 must be to count as "near"

    # ── C2 doji/small-body filter (HA-backtest-only addition, per pasted
    #    reference image: C2 must be a small, indecision-style candle sitting
    #    between C1 and the C3 confirmation). Body is measured against C2's
    #    OWN high-low range: body(C2) / (high(C2) - low(C2)) <= this ratio.
    #    Strict doji threshold requested = 15%. ──
    USE_C2_DOJI_FILTER = True
    C2_MAX_BODY_RANGE_RATIO = 0.12

    STRIKE_STEP = 100; LOT = 65; MONEYNESS = "ITM"
    SESSION_START = dtime(9, 20); SESSION_END = dtime(15, 30)
    SIGNAL_START_TIME = dtime(9, 35)
    TRADE_START = dtime(9, 30); TRADE_END = dtime(15, 0); EOD_EXIT_TIME = dtime(15, 20)

    # ── risk management: fixed premium SL + fixed premium TP only. ──
    USE_PREMIUM_SL = True
    STOP_LOSS_PCT = 10.0
    USE_PREMIUM_TP = True
    TAKE_PROFIT_PCT = 30.0

    ENTRY_LIMIT_OFFSET_PCT = 0.5   # not used in backtest execution, kept for parity with live Config

    POLL_INTERVAL_SECS = 60
    DRY_RUN = True

    def __init__(self):
        exp_dt = get_next_tuesday()
        self.EXPIRY_CODE = expiry_code(exp_dt)


# ══════════════════════════════════════════════════════════════════════════════
#  BACKTEST-SPECIFIC CONFIG OVERRIDES
#  --------------------------------------------------------------------------
#  Change any value below to tune the BACKTEST without touching the Config
#  block above (which mirrors the live bot).
# ══════════════════════════════════════════════════════════════════════════════
class BacktestConfig(Config):
    # ── Backtest window ──
    HIST_DAYS = 60

    # ── Common A/B knobs (uncomment to override live defaults) ──
    # USE_SWING_FILTER    = True    # False -> take every pattern signal
    # SWING_LOOKBACK      = 10
    # SWING_PROXIMITY_PCT = 0.5     # % of zone price
    # BODY_RATIO_MAX      = 1.2     # C1 body < this * C2 body
    # ENABLE_BUY          = True    # False -> SELL-only
    # USE_C2_DOJI_FILTER      = True   # False -> skip the C2 small-body check
    # C2_MAX_BODY_RANGE_RATIO = 0.15   # C2 body <= 15% of its own high-low range

    # ── Risk (matches live defaults) ──
    # USE_PREMIUM_SL   = True
    # STOP_LOSS_PCT    = 10.0       # -10% on option premium
    # USE_PREMIUM_TP   = True
    # TAKE_PROFIT_PCT  = 20.0       # +20% on option premium

    # ── Instrument (usually leave alone) ──
    # TF_RULE   = "1min"
    # LOT       = 65
    # MONEYNESS = "ITM"

    def __init__(self):
        super().__init__()
        logger.info(
            f"BacktestConfig (HEIKIN ASHI signals) | TF={self.TF_RULE} | "
            f"body_ratio<{self.BODY_RATIO_MAX}xC2 | "
            f"swing_filter={self.USE_SWING_FILTER} "
            f"(lookback={self.SWING_LOOKBACK}, prox={self.SWING_PROXIMITY_PCT}%) | "
            f"SL={'-'+str(self.STOP_LOSS_PCT)+'%' if self.USE_PREMIUM_SL else 'off'} "
            f"TP={'+'+str(self.TAKE_PROFIT_PCT)+'%' if self.USE_PREMIUM_TP else 'off'} "
            f"(premium, REAL option prices) | BUY={self.ENABLE_BUY} | "
            f"C2_doji_filter={self.USE_C2_DOJI_FILTER} "
            f"(max_body/range={self.C2_MAX_BODY_RANGE_RATIO:.0%}) | "
            f"direction-alternating (resets daily) | HA reset daily | "
            f"expiry={self.EXPIRY_CODE} | hist_days={self.HIST_DAYS}"
        )


# ══════════════════════════════════════════════════════════════════════════════
#  FRACTAL PIVOT DETECTION -- verbatim from pattern_poll_live.py
# ══════════════════════════════════════════════════════════════════════════════
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


# ══════════════════════════════════════════════════════════════════════════════
#  SEQUENTIAL CANDLE-REVERSAL DETECTION -- verbatim from pattern_poll_live.py
# ══════════════════════════════════════════════════════════════════════════════
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
    PLUS (HA-backtest-only addition) C2 must be a small/doji-style candle:
    body(C2) / (high(C2) - low(C2)) <= cfg.C2_MAX_BODY_RANGE_RATIO -- matches
    the reference image where the middle reversal candle is a tiny-bodied
    indecision candle between C1 and the C3 confirmation."""
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
        cond_b = body(c1) < cfg.BODY_RATIO_MAX * body(c2)     # C1 body < 1.2x C2 body
        if (cond_a or cond_b) and swing_ok(c1, "SELL") and c2_doji_ok(c2):
            return ("SELL", "sell_pattern", c1, c2)

    # ---- BUY (mirror): C1 red, C2 green, C3 (confirm) green ----
    if cfg.ENABLE_BUY and red(c1) and green(c2) and green(i):
        cond_a = c[c1] < c[c2] < c[i]                       # three rising closes
        cond_b = body(c1) < cfg.BODY_RATIO_MAX * body(c2)
        if (cond_a or cond_b) and swing_ok(c1, "BUY") and c2_doji_ok(c2):
            return ("BUY", "buy_pattern", c1, c2)
    return None

def generate_today_events(day_df: pd.DataFrame, cfg) -> List[dict]:
    """Rebuilds signal events for a single day's bars -- zones reset fresh
    each day. IDENTICAL to pattern_poll_live.py's generate_today_events
    except this file may hand it Heikin Ashi OHLC instead of real OHLC."""
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


# ══════════════════════════════════════════════════════════════════════════════
#  HEIKIN ASHI CONVERSION (reset every trading day) -- backtest-only addition
# ══════════════════════════════════════════════════════════════════════════════
def to_heikin_ashi(df: pd.DataFrame) -> pd.DataFrame:
    """
    Converts a real OHLC dataframe (DatetimeIndex, columns open/high/low/close
    [/volume]) into Heikin Ashi OHLC, resetting the recursion at the start of
    EVERY calendar day found in the index (day 1 HA candle seeds from that
    day's first real candle only -- no carry-over from the previous day).

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


# ══════════════════════════════════════════════════════════════════════════════
#  FYERS HTTP CLIENT (history + profile only -- backtest doesn't place orders)
# ══════════════════════════════════════════════════════════════════════════════
class Fyers:
    T1 = "https://api-t1.fyers.in/api/v3"; T2 = "https://api-t1.fyers.in/data"

    def __init__(self, app, tok):
        auth = tok if ":" in tok else f"{app}:{tok}"
        self.s = requests.Session(); self.s.verify = False
        self.s.headers.update({"Authorization": auth})

    def profile(self):
        return self.s.get(f"{self.T1}/profile").json()

    def history(self, symbol, res, d_from, d_to):
        r = self.s.get(f"{self.T2}/history", params={
            "symbol": symbol, "resolution": res, "date_format": "1",
            "range_from": d_from, "range_to": d_to, "cont_flag": "0"})
        try: return r.json()
        except Exception: return {}


# ══════════════════════════════════════════════════════════════════════════════
#  BACKTEST BOT
# ══════════════════════════════════════════════════════════════════════════════
class PatternLiveBacktestHABot:
    def __init__(self):
        self.cfg = BacktestConfig()
        self.fy  = self._init_fyers()

    def _init_fyers(self) -> Fyers:
        for f in ["fyers_appid.txt", "fyers_token.txt"]:
            if not os.path.exists(f): raise FileNotFoundError(f)
        app = open("fyers_appid.txt").read().strip()
        tok = open("fyers_token.txt").read().strip()
        fy = Fyers(app, tok); p = fy.profile()
        if p.get("s") != "ok":
            raise ConnectionError(f"Fyers auth failed -- re-run GenerateToken.py.\nResponse: {p}")
        logger.info(f"Fyers connected -- {p['data']['name']}")
        return fy

    def _fetch_history(self, symbol: str, days: int) -> pd.DataFrame:
        end = datetime.now(IST); start = end - timedelta(days=days)
        resp = self.fy.history(symbol, self.cfg.BASE_RES,
                               start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"))
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
        """Real 1-min NIFTY index bars -- used only to DERIVE Heikin Ashi bars.
        Real bars are NOT passed to generate_today_events; the HA-converted
        frame returned by get_index_data_ha() is."""
        df = self._fetch_history(self.cfg.UNDERLYING, self.cfg.HIST_DAYS)
        df = df.iloc[df.index.indexer_between_time(self.cfg.SESSION_START, self.cfg.SESSION_END)]
        logger.info(f"Loaded NIFTY50 1-min REAL index data: {len(df)} rows across {self.cfg.HIST_DAYS} days")
        return df

    def get_index_data_ha(self) -> pd.DataFrame:
        real_df = self.get_index_data()
        ha_df = to_heikin_ashi(real_df)
        logger.info(f"Converted to Heikin Ashi (daily reset): {len(ha_df)} rows")
        return ha_df

    def _fetch_option_1min(self, symbol: str) -> pd.DataFrame:
        # Option premiums are ALWAYS real -- Heikin Ashi is never applied here.
        return self._fetch_history(symbol, self.cfg.HIST_DAYS).sort_index()

    def _opt_symbol(self, spot: float, direction: str) -> str:
        c = self.cfg
        atm = int(round(spot / c.STRIKE_STEP) * c.STRIKE_STEP); itm = c.MONEYNESS == "ITM"
        if direction == "BUY":
            strike = atm - c.STRIKE_STEP if itm else atm; typ = "CE"
        else:
            strike = atm + c.STRIKE_STEP if itm else atm; typ = "PE"
        return f"NSE:NIFTY{c.EXPIRY_CODE}{strike:05d}{typ}"

    # ── Signal generation: per-day using generate_today_events ──
    def generate_signals(self, df_1min_ha: pd.DataFrame) -> List[dict]:
        """
        Walks day-by-day and calls generate_today_events for each day's
        HEIKIN ASHI bars. Zones/pivots/patterns reset per day exactly as in
        the live bot -- only the OHLC values fed in are Heikin Ashi instead
        of real.
        """
        c = self.cfg
        all_events: List[dict] = []
        for _day, day_df in df_1min_ha.groupby(df_1min_ha.index.date):
            day_events = generate_today_events(day_df, c)
            all_events.extend(day_events)
        logger.info(f"Generated {len(all_events)} signal events across all days "
                    f"(fed Heikin Ashi bars)")
        return all_events

    # ── backtest engine (premium SL/TP, direction alternation, no flips) ──
    # Option prices used for entry/SL/TP/EOD are always REAL, regardless of
    # HA being used for signals.

    def run_backtest(self) -> pd.DataFrame:
        c = self.cfg
        print("Starting pattern_poll_live HEIKIN ASHI backtest (premium SL/TP, direction-alternating)...")
        df_1min_ha = self.get_index_data_ha()
        events     = self.generate_signals(df_1min_ha)
        if not events:
            print("No signals generated")
            return pd.DataFrame()

        events.sort(key=lambda e: e["timestamp"])
        for ev in events:
            ev["option_symbol"] = self._opt_symbol(ev["spot"], ev["dir"])

        option_cache: Dict[str, pd.DataFrame] = {}

        def get_option_df(sym):
            if sym not in option_cache:
                try:
                    option_cache[sym] = self._fetch_option_1min(sym)
                except ValueError as e:
                    logger.warning(f"Skipping {sym}: {e}")
                    option_cache[sym] = pd.DataFrame()
            return option_cache[sym]

        def opt_price_after(opt_df, ts, field="close"):
            """side='left' returns the option bar AT ts (not one strictly after) --
            avoids the 1-min entry delay."""
            if opt_df.empty:
                return None, None
            pos = opt_df.index.searchsorted(ts, side="left")
            if field == "open":
                if pos >= len(opt_df): return None, None
                return float(opt_df.iloc[int(pos)]["open"]), opt_df.index[int(pos)]
            if pos >= len(opt_df):
                pos = len(opt_df) - 1
            return float(opt_df.iloc[int(pos)]["close"]), opt_df.index[int(pos)]

        trades = []
        # position state
        has = False
        symbol = direction = entry_price = entry_time = pattern = None
        # direction-alternation state (resets per day)
        current_day = None
        last_trade_dir: Optional[str] = None
        # stats
        skipped_dir_alt = 0
        skipped_window  = 0
        skipped_holding = 0

        def reset_pos():
            nonlocal has, symbol, direction, entry_price, entry_time, pattern
            has = False
            symbol = direction = entry_price = entry_time = pattern = None

        def record_trade(exit_time, exit_price, reason):
            pnl_pts = exit_price - entry_price
            pnl_rs  = pnl_pts * c.LOT
            trades.append({
                "direction":     direction,
                "option_symbol": symbol,
                "pattern":       pattern,
                "entry_time":    entry_time,
                "entry_price":   round(entry_price, 2),
                "exit_time":     exit_time,
                "exit_price":    round(exit_price, 2),
                "exit_reason":   reason,
                "pnl_pts":       round(pnl_pts, 2),
                "pnl_rs":        round(pnl_rs, 2),
                "result":        "profit" if pnl_pts > 0 else "loss",
            })

        for ev in events:
            ts = ev["timestamp"]
            ev_day = ts.date()

            # Day boundary -> reset direction alternation (matches live behavior)
            if current_day is None or ev_day != current_day:
                current_day = ev_day
                last_trade_dir = None

            # ── If holding, walk option 1-min bars for SL/TP/EOD ──
            if has:
                opt_df       = get_option_df(symbol)
                entry_day    = entry_time.date()
                entry_eod_ts = IST.localize(datetime.combine(entry_day, c.EOD_EXIT_TIME))
                window_end   = min(ts, entry_eod_ts)
                opt_window   = opt_df[(opt_df.index > entry_time) & (opt_df.index <= window_end)]

                for bar_ts, obar in opt_window.iterrows():
                    if not has: break

                    # EOD force close
                    if bar_ts >= entry_eod_ts:
                        px, pxt = opt_price_after(opt_df, entry_eod_ts, field="close")
                        if px is not None:
                            record_trade(pxt, px, "eod_exit")
                            reset_pos()
                        break

                    opt_high = float(obar["high"])
                    opt_low  = float(obar["low"])
                    sl_price = entry_price * (1 - c.STOP_LOSS_PCT   / 100.0)
                    tp_price = entry_price * (1 + c.TAKE_PROFIT_PCT / 100.0)

                    # SL priority if both hit inside same bar (conservative)
                    if c.USE_PREMIUM_SL and opt_low <= sl_price:
                        record_trade(bar_ts, sl_price, "stop_loss")
                        reset_pos()
                        continue
                    if c.USE_PREMIUM_TP and opt_high >= tp_price:
                        record_trade(bar_ts, tp_price, "target")
                        reset_pos()
                        continue

                # Gap safeguard: past entry_eod_ts with no covering bar
                if has and ts >= entry_eod_ts:
                    px, pxt = opt_price_after(opt_df, entry_eod_ts, field="close")
                    if px is not None:
                        record_trade(pxt, px, "eod_exit")
                        reset_pos()

            # While holding, ignore every new signal (no flips, no re-entries)
            if has:
                skipped_holding += 1
                continue

            # Direction alternation gate
            if last_trade_dir is not None and ev["dir"] == last_trade_dir:
                skipped_dir_alt += 1
                continue

            # Trade window gate
            if not (c.TRADE_START <= ts.time() <= c.TRADE_END):
                skipped_window += 1
                continue

            # Open position
            sym = ev["option_symbol"]
            opt_df = get_option_df(sym)
            px, pxt = opt_price_after(opt_df, ts, field="open")
            if px is None:
                continue
            entry_px_slipped = px * 1.005   # 0.5% slippage, matches other backtests
            has         = True
            symbol      = sym
            direction   = ev["dir"]
            entry_price = entry_px_slipped
            entry_time  = pxt
            pattern     = ev["name"]
            last_trade_dir = ev["dir"]

            delay_secs = (pxt - ts).total_seconds()
            if delay_secs > 60:
                logger.warning(
                    f"Entry {sym} {ev['dir']} signal@{ts} -> fill@{pxt} "
                    f"(delay {delay_secs:.0f}s -- likely thin option data)"
                )

        # Final flush at entry-day EOD if still open
        if has:
            opt_df = get_option_df(symbol)
            entry_day    = entry_time.date()
            entry_eod_ts = IST.localize(datetime.combine(entry_day, c.EOD_EXIT_TIME))
            px, pxt = opt_price_after(opt_df, entry_eod_ts, field="close")
            if px is None:
                last_ts = opt_df.index[-1] if not opt_df.empty else entry_time
                px, pxt = opt_price_after(opt_df, last_ts, field="close")
            if px is not None:
                record_trade(pxt, px, "backtest_end")
                reset_pos()

        trades_df = pd.DataFrame(trades)
        if trades_df.empty:
            print("No trades executed")
            print(f"\n(For diagnostics: signals generated={len(events)}, "
                  f"skipped_dir_alt={skipped_dir_alt}, skipped_window={skipped_window}, "
                  f"skipped_holding={skipped_holding})")
            return trades_df

        trades_df["cum_pnl_rs"] = trades_df["pnl_rs"].cumsum()

        total  = len(trades_df)
        wins   = int((trades_df["result"] == "profit").sum())
        losses = int((trades_df["result"] == "loss").sum())
        net    = trades_df["pnl_rs"].sum()
        wr     = round(wins / total * 100, 1) if total > 0 else 0.0

        csv_path = "Pattern_Live_HA_NIFTY_ITM_Backtest.csv"
        trades_df.to_csv(csv_path, index=False, encoding="utf-8")

        with open(csv_path, "a", encoding="utf-8") as f:
            for line in [
                "", "METRICS,VALUE",
                f"Total Trades,{total}", f"Wins,{wins}", f"Losses,{losses}",
                f"Win Rate (%),{wr}", f"Net PnL (Rs),{net:.2f}",
                f"Candle Basis,Heikin Ashi (daily reset)",
                f"TF Rule,{c.TF_RULE}",
                f"Body Ratio Max,{c.BODY_RATIO_MAX}",
                f"Swing Filter,{c.USE_SWING_FILTER}",
                f"Swing Lookback,{c.SWING_LOOKBACK}",
                f"Swing Proximity %,{c.SWING_PROXIMITY_PCT}",
                f"C2 Doji Filter,{c.USE_C2_DOJI_FILTER}",
                f"C2 Max Body/Range,{c.C2_MAX_BODY_RANGE_RATIO}",
                f"BUY Enabled,{c.ENABLE_BUY}",
                f"Premium SL %,{c.STOP_LOSS_PCT}",
                f"Premium TP %,{c.TAKE_PROFIT_PCT}",
                "Direction Alternation,daily reset",
                f"Signals Generated,{len(events)}",
                f"Skipped Dir-Alt,{skipped_dir_alt}",
                f"Skipped Window,{skipped_window}",
                f"Skipped Holding,{skipped_holding}",
                f"Lot Size,{c.LOT}",
                f"Hist Days,{c.HIST_DAYS}",
            ]:
                f.write(line + "\n")

        by_reason = trades_df.groupby("exit_reason")["pnl_rs"].agg(["count", "sum"]).reset_index()
        by_dir    = trades_df.groupby("direction")["pnl_rs"].agg(["count", "sum"]).reset_index()
        print(f"\n{'='*62}")
        print(f"  Candle Basis        : HEIKIN ASHI (daily reset)")
        print(f"  Contract Type       : {c.MONEYNESS}  |  TF: {c.TF_RULE}")
        print(f"  Premium SL / TP     : -{c.STOP_LOSS_PCT}% / +{c.TAKE_PROFIT_PCT}%  (real option prices)")
        print(f"  Swing Filter        : {c.USE_SWING_FILTER}  "
              f"(lookback={c.SWING_LOOKBACK}, prox={c.SWING_PROXIMITY_PCT}%)")
        print(f"  Total Trades        : {total}   (W {wins} / L {losses})")
        print(f"  Win Rate            : {wr}%")
        print(f"  Net PnL             : Rs {net:,.2f}")
        print(f"  Best / Worst        : Rs {trades_df['pnl_rs'].max():,.2f} / Rs {trades_df['pnl_rs'].min():,.2f}")
        print(f"{'-'*62}")
        print("  Signal funnel:")
        print(f"    Signals generated       : {len(events)}")
        print(f"    Skipped dir-alternation : {skipped_dir_alt}")
        print(f"    Skipped trade window    : {skipped_window}")
        print(f"    Skipped (in-position)   : {skipped_holding}")
        print(f"    Actual entries          : {total}")
        print("  Exit-reason breakdown:")
        for _, row in by_reason.iterrows():
            print(f"    {row['exit_reason']:<15} count={int(row['count']):<3}  sum=Rs {row['sum']:>12,.2f}")
        print("  Direction breakdown:")
        for _, row in by_dir.iterrows():
            print(f"    {str(row['direction']):<8} count={int(row['count']):<3}  sum=Rs {row['sum']:>12,.2f}")
        print(f"{'='*62}")
        print(f"Results saved -> {csv_path}")
        return trades_df


# --------------------------- MAIN ---------------------------

if __name__ == "__main__":
    bot    = PatternLiveBacktestHABot()
    trades = bot.run_backtest()
