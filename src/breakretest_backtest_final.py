"""
Breaks and Retests [HG] backtest -- paired with breakretest_poll_live.py's
LIVE Config (LOOKBACK=10, BARS_SINCE_BREAK=2, RETEST_VALID_BARS=2,
TF_RULE="1min", EXPIRY_TYPE=WEEKLY, LOT=65, SL=20%, TP=20%,
same-direction-per-day guard). This backtest exists to validate that exact
configuration against history.

MECHANIC (ported from the "Breaks and Retests [HG]" Pine indicator's
"Off: Candle Confirmation" non-repainting mode):
  1. PIVOT -> ZONE: classic fractal pivot (ta.pivotlow/pivothigh, symmetric
     `lookback` bars each side). The zone's other edge is the nearer of the
     pivot's two immediate neighbor bars, giving the zone real width. Only
     the MOST RECENT support zone and MOST RECENT resistance zone are
     tracked at any time -- a new pivot retires the old zone.
  2. BREAK (one-shot per zone): a CLOSE crossing through the zone's outer
     edge marks it "broken." Re-armed only when a newer pivot supersedes it.
  3. RETEST TOUCH: after `bars_since_breakout` bars have passed, watch each
     bar's high/low/close against the zone's top/bottom for any of the 4
     touch patterns (wick-only touches, close-inside-zone touches, full-
     zone-crossing wicks). The first touch bar's extreme becomes the
     confirmation reference level.
  4. RETEST CONFIRMATION = ENTRY: within `retest_valid_bars` of the touch, a
     later bar's CLOSE breaking back through that reference level in the
     original break direction confirms the retest -- BUY (resistance) or
     SELL (support). If unconfirmed within the window, the potential retest
     expires but a NEW touch can start a fresh cycle against the same zone.

TRADING RULE (position management, since the indicator has no trade rules of
its own): same-direction-per-day guard -- once a direction has been traded
on a given calendar day, further signals in that SAME direction that day are
skipped. Resets each new day. No flip-exit -- a position only closes via
candle-level SL... actually this config has NO candle SL; only premium
STOP_LOSS_PCT/TAKE_PROFIT_PCT and EOD close a position.

Same conventions as the other backtests: ExpiryResolver for per-signal-date
weekly expiry (expiry_resolver.py), same-day EOD enforcement, throttled
option-history fetches (0.25s + 429 retry), entry at the confirming bar's
own close (not a following bar's open) to avoid unnecessary lag.
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

# ---------- expiry helpers (identical convention to the live bot) ----------
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

# ---------- config (identical to breakretest_poll_live.py) ----------
class Config:
    UNDERLYING = "NSE:NIFTY50-INDEX"; BASE_RES = "5"; HIST_DAYS = 15
    TF_RULE = "5min"                       # timeframe candles/pivots/reversal run on

    LOOKBACK          = 30
    BARS_SINCE_BREAK  = 2
    RETEST_VALID_BARS = 2

    STRIKE_STEP = 100; LOT = 65; MONEYNESS = "ITM"
    EXPIRY_TYPE = "WEEKLY"   # "WEEKLY" or "MONTHLY"
    SESSION_START = dtime(9, 20); SESSION_END = dtime(15, 30)
    TRADE_START = dtime(9, 30); TRADE_END = dtime(15, 0); EOD_EXIT_TIME = dtime(15, 20)

    STOP_LOSS_PCT = 20.0
    TAKE_PROFIT_PCT = 20.0

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

# ---------- fractal pivot detection (VERBATIM, must match the live bot) ----------
def detect_fractal_pivots(h, l, lookback: int):
    """Direct port of Pine's ta.pivotlow(low, bb, bb) / ta.pivothigh(high, bb, bb):
    a pivot low at bar i is confirmed once bar i+bb has closed, if low[i] is
    the minimum over the window [i-bb, i+bb] (symmetric). Returns two lists
    of (confirm_bar_idx, pivot_bar_idx)."""
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
        """Resolves the expiry ACTIVE ON THE SIGNAL'S OWN DATE via
        ExpiryResolver (matching Fyers' live symbol master), instead of a
        single fixed expiry computed once at bot init."""
        c = self.cfg
        if c.EXPIRY_TYPE.upper() == "MONTHLY":
            exp_dt = get_current_or_next_monthly_expiry(datetime.combine(signal_date, dtime(9, 20)))
            exp_code = expiry_code(exp_dt)
        else:
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

    # ── zone + break + retest signal generation, replayed bar-by-bar per day ──

    def generate_signals(self, df_1min: pd.DataFrame) -> list:
        """IDENTICAL logic to breakretest_poll_live.py's generate_today_events(),
        replayed for every historical day via groupby instead of just today."""
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

            r_zone = None
            s_zone = None

            for i in range(c.LOOKBACK, n):
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
                                    r_zone["retest"] = None
                                elif i > rt["touch_idx"] + c.RETEST_VALID_BARS:
                                    r_zone["retest"] = None

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
        has = False; symbol = None; direction = None; entry_price = None; entry_time = None
        zone_kind = None
        # Same-direction-per-day guard, matching breakretest_poll_live.py's
        # _traded_directions_today: resets every calendar day.
        traded_directions_today: Dict[object, set] = {}

        for ev in events:
            ts = ev["timestamp"]

            # ── risk management: EOD boundary is the ENTRY's own day ──
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
                continue  # single-position engine, same as the live bot

            in_win = c.TRADE_START <= ts.time() <= c.TRADE_END
            if not in_win:
                continue

            sig_day = ts.date()
            sig_dir = ev["dir"]
            if sig_dir in traded_directions_today.get(sig_day, set()):
                continue  # same direction already traded today

            sym = ev["option_symbol"]
            opt_df = get_option_df(sym)
            # Enter using the option's price at/immediately after the
            # confirming bar's own close (approximates an immediate market
            # order the instant the signal fires), not the following bar's
            # open, to avoid unnecessary lag.
            entry_px, entry_ts_actual = opt_price_after(opt_df, ts, field="close")
            if entry_px is None:
                skipped_no_data += 1
                continue
            entry_px *= 1.005  # small slippage, matches other backtests' convention

            has = True; symbol = sym; direction = ev["dir"]
            entry_price = entry_px; entry_time = entry_ts_actual
            zone_kind = ev["zone_kind"]
            traded_directions_today.setdefault(sig_day, set()).add(sig_dir)

        # ── final flush: force-close at the entry day's EOD price ──
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

        csv_path = "BreakRetest_Final_NIFTY_ITM_Backtest.csv"
        trades_df.to_csv(csv_path, index=False, encoding="utf-8")

        with open(csv_path, "a", encoding="utf-8") as f:
            for line in [
                "", "METRICS,VALUE",
                f"Total Trades,{total}", f"Wins,{wins}", f"Losses,{losses}",
                f"Win Rate (%),{wr}", f"Net PnL (Rs),{net:.2f}",
                f"Signals Skipped (no option data),{skipped_no_data}",
                f"Signals Using Fallback Expiry Guess,{self._fallback_expiry_count}",
                f"TF Rule,{c.TF_RULE}", f"Lookback,{c.LOOKBACK}",
                f"Lot Size,{c.LOT}",
            ]:
                f.write(line + "\n")

        print(f"\n{'='*55}")
        print(f"  Contract Type       : {c.MONEYNESS}")
        print(f"  Total Trades        : {total}")
        print(f"  Signals Skipped     : {skipped_no_data} (no option data)")
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
