"""
Breaks and Retests [HG] backtest -- direct port of the "Off: Candle
Confirmation" (non-repainting) mode of the original Pine indicator. Unlike
the SMC and Killzone indicators ported earlier, this one already has a
complete, causal trade trigger built in -- no invented trading logic layered
on top, just the indicator's own break -> retest -> continuation sequence.

MECHANIC (ported from the Pine source):
  1. PIVOT -> ZONE: classic fractal pivot (ta.pivotlow/pivothigh, symmetric
     `lookback` bars each side). The zone is NOT just the pivot price -- its
     other edge is the nearer of the pivot's two immediate neighbor bars
     (mirrors s_yLoc/r_yLoc in the Pine source), giving the zone real width.
     Only the MOST RECENT support zone and MOST RECENT resistance zone are
     tracked at any time -- a new pivot retires the old zone (mirrors the
     Pine source's single sBox/rBox with ta.change(pl)/ta.change(ph) reset).
  2. BREAK (one-shot per zone, bar-confirmation mode): a CLOSE crossing
     through the zone's outer edge marks the zone "broken." Re-armed only
     when a newer pivot supersedes the zone.
  3. RETEST TOUCH: after a break, once `bars_since_breakout` have passed,
     watch each bar's high/low/close against the zone's top/bottom for any
     of the 4 touch patterns from the Pine source (s1-s4 for a broken
     support zone, r1-r4 for a broken resistance zone) -- these cover wick-
     only touches, close-inside-zone touches, and full-zone-crossing wicks.
     The first touch bar's extreme (low for a support retest, high for a
     resistance retest) becomes the confirmation reference level.
  4. RETEST CONFIRMATION = ENTRY: within `retest_valid_bars` of the touch,
     if a later bar's CLOSE breaks back through that reference level in the
     direction of the original break, the retest is confirmed -- this IS
     the entry trigger:
       - Resistance broken -> retested -> close breaks back ABOVE the touch
         bar's high -> BUY (continuation).
       - Support broken -> retested -> close breaks back BELOW the touch
         bar's low -> SELL (continuation).
     If no confirmation happens within the window, the potential retest
     expires with no signal, but a NEW touch can still start a fresh
     potential-retest cycle against the SAME broken zone (mirrors the Pine
     source resetting retOccurred whenever a new retEvent fires) -- so one
     broken zone can produce multiple retest signals over time.

SIMPLIFICATIONS vs the original (both documented, neither is a logic error):
  1. The Pine source has a minor refinement where an already-active potential
     retest can be "upgraded" to a better touch point if a subsequent bar's
     extreme is more favorable before ever confirming. Not ported -- the
     first touch bar's extreme is used as-is.
  2. Pine's retEvent only fires on a RISING EDGE of the touch condition
     (retActive and not retActive[1] -- i.e. the touch condition must go
     false then true again to start a new potential-retest cycle). This
     port starts a new cycle as soon as `retest` is cleared (via confirm or
     expiry) and touch is true again, without requiring an intervening
     false bar. In practice this only differs when price sits continuously
     inside the zone across a confirm/expiry boundary, a narrow edge case.

VERIFIED AGAINST THE PINE SOURCE (two off-by-one bugs found and fixed in
this version):
  - BARS_SINCE_BREAK gate: Pine's `ta.barssince(na(breakout)) > input_retSince`
    first becomes true at bar (break_idx + BARS_SINCE_BREAK), confirmed by
    tracing barssince's value bar-by-bar from the break. The gate now uses
    `bars_since >= BARS_SINCE_BREAK` -- an earlier version used `>`, which
    gated one bar too late.
  - Retest confirmation window: Pine's `retValid` requires
    `barssince(retEvent) <= input_retValid` AS PART OF the confirmation
    check itself, not just as a separate expiry condition. An earlier
    version had no upper bound on the confirmation check, only on the
    expiry check -- meaning a close breaking back through the reference
    level arbitrarily many bars after the touch would still have incorrectly
    fired a signal. Now bounded via `within_window`.

TRADING RULE (added on top of the ported indicator logic, since the
indicator itself has no position-management rules): once a direction
(BUY or SELL) has been traded on a given calendar day, further signals in
that SAME direction that day are skipped -- avoids repeated same-direction
entries (e.g. SELL -> target, then another SELL minutes later -> stop_loss).

Same conventions as the other backtests in this project: same-day EOD
enforcement, throttled option-history fetches, entry at the confirming
bar's own close (not a following bar's open) to avoid unnecessary lag.
"""
import pandas as pd, numpy as np, requests, logging, os, calendar, pytz, urllib3, warnings, time
from datetime import datetime, timedelta, time as dtime
from typing import Optional, Dict, List
from expiry_resolver import ExpiryResolver

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
warnings.filterwarnings("ignore")
IST = pytz.timezone("Asia/Kolkata")

logging.basicConfig(level=logging.INFO,
                     format="%(asctime)s - BRBacktest - %(levelname)s - %(message)s",
                     datefmt="%Y-%m-%d %H:%M:%S")
logger = logging.getLogger("BRBacktest")

# ---------- expiry helpers (identical convention to the other bots) ----------
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
    """Returns the CURRENT month's expiry if it hasn't passed yet, else next
    month's -- same convention as vam_poll_5S.py / VAMStrategy.py."""
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

# ---------- config ----------
class Config:
    UNDERLYING = "NSE:NIFTY50-INDEX"; BASE_RES = "1"
    # HIST_DAYS shrunk from 30 -> 10: Fyers' live symbol master only lists
    # CURRENTLY TRADABLE contracts, so ExpiryResolver can only verify recent
    # weeks (see expiry_resolver.py docstring). At 30 days, ~99% of signals
    # (160/161 in testing) fell back to the unreliable Tuesday-guess and
    # mostly failed as invalid symbols. 10 days keeps the backtest within
    # the resolver's verifiable window -- almost every signal should now
    # resolve against the real master instead of guessing.
    HIST_DAYS = 10
    # TF_RULE matched to 1min: the reference TradingView chart runs the
    # indicator on a 1-minute chart with LOOKBACK=20 (20 MINUTES of pivot
    # window) -- the previous 5min setting made LOOKBACK=20 span 100
    # minutes, a much coarser/less frequent signal cadence than what's
    # visible on the chart.
    TF_RULE = "1min"   # timeframe pivots/breaks/retests run on

    LOOKBACK          = 10   # Pine: input_lookback -- bars each side for pivot confirmation
    BARS_SINCE_BREAK  = 2    # Pine: input_retSince -- min bars after break before a retest can register
    RETEST_VALID_BARS = 2    # Pine: input_retValid -- bars a potential retest stays active before expiring

    STRIKE_STEP = 100; LOT = 65; MONEYNESS = "ITM"
    EXPIRY_TYPE = "weekly"   # "WEEKLY" or "MONTHLY"
    SESSION_START = dtime(9, 20); SESSION_END = dtime(15, 30)
    TRADE_START = dtime(9, 30); TRADE_END = dtime(15, 0); EOD_EXIT_TIME = dtime(15, 20)

    STOP_LOSS_PCT = 20
    TAKE_PROFIT_PCT = 20

    def __init__(self):
        now = datetime.now(IST)
        exp_dt = (get_current_or_next_monthly_expiry(now) if self.EXPIRY_TYPE.upper() == "MONTHLY"
                  else get_next_tuesday(now))
        self.EXPIRY_CODE = expiry_code(exp_dt)
        logger.info(f"{self.EXPIRY_TYPE} expiry code {self.EXPIRY_CODE} | Break&Retest on {self.TF_RULE} "
                    f"lookback={self.LOOKBACK} bars_since_break={self.BARS_SINCE_BREAK} "
                    f"retest_valid={self.RETEST_VALID_BARS} | SL={self.STOP_LOSS_PCT}% TP={self.TAKE_PROFIT_PCT}%")

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

# ---------- fractal pivot detection ----------
def detect_fractal_pivots(h, l, lookback: int):
    """
    Direct port of Pine's ta.pivotlow(low, bb, bb) / ta.pivothigh(high, bb, bb):
    a pivot low at bar i is confirmed once bar i+bb has closed, if low[i] is
    the minimum over the window [i-bb, i+bb] (symmetric). Mirrors fixnan() by
    only returning the CONFIRM index (when the pivot becomes knowable) plus
    the pivot's own bar index -- callers walk forward using confirm index as
    the causal point.
    Returns two lists of (confirm_bar_idx, pivot_bar_idx).
    """
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

# ---------- backtest bot ----------
class BreakRetestBacktestBot:
    def __init__(self):
        self.cfg = Config()
        self.fy = self._init_fyers()
        self.expiry_resolver = ExpiryResolver()
        self._fallback_expiry_count = 0

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
        df = self._fetch_history(self.cfg.UNDERLYING, self.cfg.HIST_DAYS)
        df = df.iloc[df.index.indexer_between_time(self.cfg.SESSION_START, self.cfg.SESSION_END)]
        logger.info(f"Loaded NIFTY50 1-min index data: {len(df)} rows")
        return df

    def _fetch_option_1min(self, symbol: str) -> pd.DataFrame:
        return self._fetch_history(symbol, self.cfg.HIST_DAYS).sort_index()

    def _opt_symbol_for_date(self, spot: float, direction: str, signal_date) -> str:
        """Resolves the option's expiry for MONTHLY contracts.
        Unlike the weekly case, monthly expiry is computed DIRECTLY (same
        logic as vam_poll_5S.py) rather than via ExpiryResolver's nearest-
        any-expiry day scan -- that scan would find the NEXT WEEKLY expiry
        first, since it's always chronologically closer than the monthly
        one. Fyers' live symbol master is still used to VERIFY the computed
        code (tracked via _fallback_expiry_count) but is not used to search."""
        c = self.cfg
        signal_dt = datetime.combine(signal_date, dtime(9, 20))
        if c.EXPIRY_TYPE.upper() == "MONTHLY":
            exp_dt = get_current_or_next_monthly_expiry(signal_dt)
        else:
            exp_dt = get_next_tuesday(signal_dt)
        exp_code = expiry_code(exp_dt)

        self.expiry_resolver._ensure_loaded()
        if self.expiry_resolver._valid_codes and exp_code not in self.expiry_resolver._valid_codes:
            self._fallback_expiry_count += 1  # unverified against Fyers' live master (likely aged out)

        atm = int(round(spot / c.STRIKE_STEP) * c.STRIKE_STEP); itm = c.MONEYNESS == "ITM"
        if direction == "BUY":
            strike = atm - c.STRIKE_STEP if itm else atm; typ = "CE"
        else:
            strike = atm + c.STRIKE_STEP if itm else atm; typ = "PE"
        return f"NSE:NIFTY{exp_code}{strike:05d}{typ}"

    # ── zone + break + retest signal generation, replayed bar-by-bar per day ──

    def generate_signals(self, df_1min: pd.DataFrame) -> list:
        c = self.cfg
        events = []

        for day, day_df in df_1min.groupby(df_1min.index.date):
            r = day_df.resample(c.TF_RULE).agg(
                {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
            ).dropna()
            n = len(r)
            if n < c.LOOKBACK * 2 + 2:
                continue

            o = r["open"].to_numpy(); h = r["high"].to_numpy()
            l = r["low"].to_numpy(); cl = r["close"].to_numpy()

            piv_highs, piv_lows = detect_fractal_pivots(h, l, c.LOOKBACK)
            highs_by_confirm = {ci: pi for ci, pi in piv_highs}
            lows_by_confirm = {ci: pi for ci, pi in piv_lows}

            # single active zone per side, mirrors the Pine source's one sBox/rBox
            r_zone = None  # {"top": ph, "bottom": r_yLoc, "broken": bool, "break_idx": int, "retest": {...} or None}
            s_zone = None  # {"top": s_yLoc, "bottom": pl, "broken": bool, "break_idx": int, "retest": {...} or None}

            for i in range(c.LOOKBACK, n):
                # ── new pivot confirmed at this bar -> retire old zone, start new one ──
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

                # ── resistance: break, retest touch, retest confirmation (BUY) ──
                if r_zone is not None:
                    if not r_zone["broken"]:
                        if i > 0 and cl[i - 1] <= r_zone["top"] < cl[i]:
                            r_zone["broken"] = True
                            r_zone["break_idx"] = i
                    else:
                        bars_since = i - r_zone["break_idx"]
                        if bars_since >= c.BARS_SINCE_BREAK:
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
                                within_window = rt["touch_idx"] < i <= rt["touch_idx"] + c.RETEST_VALID_BARS
                                if within_window and cl[i] >= rt["ref_high"]:
                                    events.append({
                                        "timestamp": r.index[i] + pd.Timedelta(c.TF_RULE),
                                        "dir": "BUY", "zone_kind": "resistance",
                                        "zone_top": rTop, "zone_bottom": rBot,
                                        "retest_ref": rt["ref_high"], "index_price": float(cl[i]),
                                    })
                                    r_zone["retest"] = None  # allow a fresh cycle against the same zone
                                elif i > rt["touch_idx"] + c.RETEST_VALID_BARS:
                                    r_zone["retest"] = None  # expired, no signal

                # ── support: break, retest touch, retest confirmation (SELL) ──
                if s_zone is not None:
                    if not s_zone["broken"]:
                        if i > 0 and cl[i - 1] >= s_zone["bottom"] > cl[i]:
                            s_zone["broken"] = True
                            s_zone["break_idx"] = i
                    else:
                        bars_since = i - s_zone["break_idx"]
                        if bars_since >= c.BARS_SINCE_BREAK:
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
                                within_window = rt["touch_idx"] < i <= rt["touch_idx"] + c.RETEST_VALID_BARS
                                if within_window and cl[i] <= rt["ref_low"]:
                                    events.append({
                                        "timestamp": r.index[i] + pd.Timedelta(c.TF_RULE),
                                        "dir": "SELL", "zone_kind": "support",
                                        "zone_top": sTop, "zone_bottom": sBot,
                                        "retest_ref": rt["ref_low"], "index_price": float(cl[i]),
                                    })
                                    s_zone["retest"] = None
                                elif i > rt["touch_idx"] + c.RETEST_VALID_BARS:
                                    s_zone["retest"] = None

        logger.info(f"Generated {len(events)} break-and-retest signal events across all days")
        return events

    # ── backtest engine: entries at confirming bar's own close, premium SL/TP, EOD ──

    def run_backtest(self) -> pd.DataFrame:
        c = self.cfg
        print("Starting Break & Retest Backtest (ITM contracts)...")
        df_1min = self.get_index_data()
        events = self.generate_signals(df_1min)
        if not events:
            print("No signals generated")
            return pd.DataFrame()

        events.sort(key=lambda e: e["timestamp"])
        for ev in events:
            ev["option_symbol"] = self._opt_symbol_for_date(ev["index_price"], ev["dir"], ev["timestamp"].date())

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

        def opt_price_after(opt_df, ts, field="close"):
            if opt_df.empty: return None, None
            pos = opt_df.index.searchsorted(ts, side="right" if field == "open" else "left")
            if field == "open":
                if pos >= len(opt_df): return None, None
                return float(opt_df.iloc[int(pos)]["open"]), opt_df.index[int(pos)]
            if pos >= len(opt_df):
                pos = len(opt_df) - 1
            return float(opt_df.iloc[int(pos)]["close"]), opt_df.index[int(pos)]

        trades = []
        skipped_no_data = 0
        skipped_same_direction = 0
        has = False; symbol = None; direction = None; entry_price = None; entry_time = None
        zone_kind = None
        # Tracks which directions have ALREADY been traded on each calendar
        # day, e.g. {date(2026,8,12): {"SELL"}}. Once a direction has fired
        # once on a given day, further signals in that SAME direction that
        # day are skipped -- avoids repeated same-direction entries (e.g.
        # SELL -> target, then another SELL minutes later -> stop_loss).
        traded_directions_today: Dict[object, set] = {}

        for ev in events:
            ts = ev["timestamp"]

            if has:
                opt_df = get_option_df(symbol)
                entry_day = entry_time.date()
                entry_eod_ts = IST.localize(datetime.combine(entry_day, c.EOD_EXIT_TIME))
                window_end = min(ts, entry_eod_ts)
                window = opt_df[(opt_df.index > entry_time) & (opt_df.index <= window_end)]
                for bar_ts, bar in window.iterrows():
                    if not has: break
                    if bar_ts >= entry_eod_ts:
                        px, pxt = opt_price_after(opt_df, entry_eod_ts)
                        if px is not None:
                            trades.append(self._close(direction, symbol, zone_kind,
                                                       entry_time, entry_price, pxt, px, "eod_exit", c.LOT))
                            has = False
                        break
                    px = float(bar["close"])
                    if px <= entry_price * (1 - c.STOP_LOSS_PCT / 100.0):
                        trades.append(self._close(direction, symbol, zone_kind,
                                                   entry_time, entry_price, bar_ts, px, "stop_loss", c.LOT))
                        has = False; continue
                    if px >= entry_price * (1 + c.TAKE_PROFIT_PCT / 100.0):
                        trades.append(self._close(direction, symbol, zone_kind,
                                                   entry_time, entry_price, bar_ts, px, "target", c.LOT))
                        has = False; continue

                if has and ts >= entry_eod_ts:
                    px, pxt = opt_price_after(opt_df, entry_eod_ts)
                    if px is not None:
                        trades.append(self._close(direction, symbol, zone_kind,
                                                   entry_time, entry_price, pxt, px, "eod_exit", c.LOT))
                    has = False

            if has:
                continue

            in_win = c.TRADE_START <= ts.time() <= c.TRADE_END
            if not in_win:
                continue

            sig_day = ts.date()
            sig_dir = ev["dir"]
            if sig_dir in traded_directions_today.get(sig_day, set()):
                skipped_same_direction += 1
                continue

            sym = ev["option_symbol"]
            opt_df = get_option_df(sym)
            entry_px, entry_ts_actual = opt_price_after(opt_df, ts, field="close")
            if entry_px is None:
                skipped_no_data += 1
                continue
            entry_px *= 1.005

            has = True; symbol = sym; direction = ev["dir"]
            entry_price = entry_px; entry_time = entry_ts_actual
            zone_kind = ev["zone_kind"]
            traded_directions_today.setdefault(sig_day, set()).add(sig_dir)

        if has:
            opt_df = get_option_df(symbol)
            entry_day = entry_time.date()
            entry_eod_ts = IST.localize(datetime.combine(entry_day, c.EOD_EXIT_TIME))
            px, pxt = opt_price_after(opt_df, entry_eod_ts)
            if px is None:
                last_ts = opt_df.index[-1] if not opt_df.empty else entry_time
                px, pxt = opt_price_after(opt_df, last_ts)
            if px is not None:
                trades.append(self._close(direction, symbol, zone_kind,
                                           entry_time, entry_price, pxt, px, "backtest_end", c.LOT))

        logger.info(f"Signals skipped due to missing/invalid option data: {skipped_no_data}")
        logger.info(f"Signals skipped (direction already traded that day): {skipped_same_direction}")
        logger.info(f"Signals with UNVERIFIED expiry (not found in Fyers' live master): {self._fallback_expiry_count}")

        trades_df = pd.DataFrame(trades)
        if trades_df.empty:
            print("No trades executed")
            print(f"  Signals skipped (no option data): {skipped_no_data}")
            return trades_df

        trades_df["cum_pnl_rs"] = trades_df["pnl_rs"].cumsum()

        wins = int((trades_df["result"] == "profit").sum())
        losses = int((trades_df["result"] == "loss").sum())
        total = len(trades_df)
        net = trades_df["pnl_rs"].sum()
        wr = round(wins / total * 100, 1) if total > 0 else 0.0

        csv_path = "BreakRetest_NIFTY_ITM_Backtest.csv"
        trades_df.to_csv(csv_path, index=False, encoding="utf-8")

        with open(csv_path, "a", encoding="utf-8") as f:
            for line in [
                "", "METRICS,VALUE",
                f"Total Trades,{total}", f"Wins,{wins}", f"Losses,{losses}",
                f"Win Rate (%),{wr}", f"Net PnL (Rs),{net:.2f}",
                f"Signals Skipped (no option data),{skipped_no_data}",
                f"Signals Skipped (same direction already traded that day),{skipped_same_direction}",
                f"Signals Using Fallback Expiry Guess,{self._fallback_expiry_count}",
                f"TF Rule,{c.TF_RULE}", f"Lookback,{c.LOOKBACK}",
                f"Lot Size,{c.LOT}",
            ]:
                f.write(line + "\n")

        print(f"\n{'='*55}")
        print(f"  Contract Type       : {c.MONEYNESS}")
        print(f"  Total Trades        : {total}")
        print(f"  Signals Skipped     : {skipped_no_data} (no option data)")
        print(f"  Same-Dir Skipped    : {skipped_same_direction} (already traded that day)")
        print(f"  Unverified Expiries : {self._fallback_expiry_count}")
        print(f"  Win Rate            : {wr}%")
        print(f"  Net PnL             : Rs {net:,.2f}")
        print(f"{'='*55}")
        print(f"Results saved -> {csv_path}")
        return trades_df

    @staticmethod
    def _close(direction, symbol, zone_kind, entry_time, entry_price,
               exit_time, exit_price, reason, lot):
        pnl_pts = exit_price - entry_price
        pnl_rs = pnl_pts * lot
        return {
            "direction": direction, "option_symbol": symbol, "zone_kind": zone_kind,
            "entry_time": entry_time, "entry_price": round(entry_price, 2),
            "exit_time": exit_time, "exit_price": round(exit_price, 2),
            "exit_reason": reason,
            "pnl_pts": round(pnl_pts, 2), "pnl_rs": round(pnl_rs, 2),
            "result": "profit" if pnl_pts > 0 else "loss",
        }


# --------------------------- MAIN ---------------------------

if __name__ == "__main__":
    bot    = BreakRetestBacktestBot()
    trades = bot.run_backtest()