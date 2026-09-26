"""
ICT Killzones + Pivots [TFO] backtest -- distilled to its one tradeable
mechanic: session high/low tracking with breakout-continuation entries.

The original Pine script is almost entirely a charting/display tool (boxes,
labels, midpoint lines, range levels, D/W/M opens, timestamps, a hit-rate
stats table) with NO trading rules of its own. The only structural mechanic
worth porting is:

  1. KILLZONE SESSIONS: configurable IST time windows (e.g. "Opening Range
     09:15-09:45"). While price is inside a session window, the bar-by-bar
     high/low is tracked (mirrors adjust_in_kz()).
  2. SESSION CLOSE -> PIVOT PAIR: once the session ends, its high and low
     become a fixed support/resistance pair, active until price closes
     through either one ("mitigated" in the original's terminology --
     mirrors adjust_out_kz()'s high > hi_line / low < lo_line checks).
  3. BREAKOUT CONTINUATION (the trading rule, since the original has none):
     a CLOSE beyond the session high fires a BUY; a close beyond the session
     low fires a SELL. Each pivot fires at most once (mirrors the original
     setting kz._hi_valid/_lo_valid := false on the first break -- once
     "mitigated" it's retired, matching ext_pivots="Until Mitigated").
  4. Multiple killzones can be configured; each tracks its own independent
     pivot pair per day.

Causal -- everything evaluated on CLOSED bars only, walked forward bar by bar.

Option symbol resolution uses ExpiryResolver (expiry_resolver.py), which
matches each signal's date against Fyers' LIVE symbol master to find the
actual weekly expiry -- not a hardcoded "next Tuesday" guess, which breaks
whenever NSE shifts the expiry weekday or a Tuesday falls on a holiday.
Falls back to the Tuesday guess (and counts it) only when the master has no
match, typically because that expiry has already aged out of Fyers' listing.

Every position is force-closed at EOD_EXIT_TIME on its own ENTRY day.

Entries execute at the option's price at/immediately after the breakout
bar's own close (approximates an immediate market order right when the
signal fires) -- not at the following bar's open, which was adding an extra
minute of unnecessary lag on top of the unavoidable 1-bar close-confirmation
delay.

Option-history fetches are throttled (0.25s between calls, retry-once on
HTTP 429) since this bot can generate many more signals across many more
distinct weeks/strikes than the SMC backtest, which was triggering Fyers'
rate limit on unthrottled bursts.
"""
import pandas as pd, numpy as np, requests, logging, os, calendar, pytz, urllib3, warnings, time
from datetime import datetime, timedelta, time as dtime
from typing import Optional, Dict, List
from expiry_resolver import ExpiryResolver

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
warnings.filterwarnings("ignore")
IST = pytz.timezone("Asia/Kolkata")

logging.basicConfig(level=logging.INFO,
                     format="%(asctime)s - KZBacktest - %(levelname)s - %(message)s",
                     datefmt="%Y-%m-%d %H:%M:%S")
logger = logging.getLogger("KZBacktest")

# ---------- expiry helpers (identical convention to the other bots) ----------
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
    UNDERLYING = "NSE:NIFTY50-INDEX"; BASE_RES = "10S"; HIST_DAYS = 30

    # Configurable killzone sessions in IST, (name, start, end) -- keep it
    # list-based like the original's 6 session inputs. Defaults are a
    # reasonable starting set for NSE cash-market hours (09:15-15:30):
    #   Opening Range: captures the initial volatility burst after open.
    #   Mid-Morning:   a calmer trend-establishment window.
    #   Post-Lunch:    the pre-close session where afternoon moves develop.
    KILLZONES = [
        ("ORB",       dtime(9, 15),  dtime(9, 45)),
        ("MIDMORN",   dtime(9, 45),  dtime(11, 30)),
        ("POSTLUNCH", dtime(13, 0),  dtime(14, 30)),
    ]

    STRIKE_STEP = 100; LOT = 65; MONEYNESS = "ITM"
    SESSION_START = dtime(9, 20); SESSION_END = dtime(15, 30)
    TRADE_START = dtime(9, 30); TRADE_END = dtime(15, 0); EOD_EXIT_TIME = dtime(15, 20)

    STOP_LOSS_PCT =20
    TAKE_PROFIT_PCT = 20

    def __init__(self):
        self.EXPIRY_CODE = expiry_code(get_next_tuesday())
        kz_str = ", ".join(f"{n}({s.strftime('%H:%M')}-{e.strftime('%H:%M')})" for n, s, e in self.KILLZONES)
        logger.info(f"Weekly expiry code {self.EXPIRY_CODE} | Killzones: {kz_str} | "
                    f"breakout-continuation | SL={self.STOP_LOSS_PCT}% TP={self.TAKE_PROFIT_PCT}%")

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

# ---------- backtest bot ----------
class KillzoneBacktestBot:
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
        """Resolves the ACTUAL weekly expiry against Fyers' live symbol
        master (see expiry_resolver.py) instead of guessing 'next Tuesday' --
        the guess breaks whenever NSE shifts the expiry weekday or a Tuesday
        falls on a trading holiday. Falls back to the old guess (and counts
        it) only when the master has no match for this date -- typically
        because the expiry has already aged out of Fyers' live listing."""
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

    # ── killzone session high/low tracking + breakout signal generation ──

    def generate_signals(self, df_1min: pd.DataFrame) -> list:
        c = self.cfg
        events = []

        for day, day_df in df_1min.groupby(df_1min.index.date):
            # track each killzone's own state independently for this day
            kz_state = []
            for name, start_t, end_t in c.KILLZONES:
                kz_state.append({
                    "name": name, "start": start_t, "end": end_t,
                    "hi": None, "lo": None,      # running session extremes
                    "hi_pivot": None, "lo_pivot": None,  # fixed once session ends
                    "hi_active": False, "lo_active": False,
                    "in_session_seen": False,
                })

            for ts, bar in day_df.iterrows():
                ts_time = ts.time()
                h = float(bar["high"]); l = float(bar["low"]); cl = float(bar["close"])

                for kz in kz_state:
                    in_session = kz["start"] <= ts_time < kz["end"]

                    if in_session:
                        kz["in_session_seen"] = True
                        kz["hi"] = h if kz["hi"] is None else max(kz["hi"], h)
                        kz["lo"] = l if kz["lo"] is None else min(kz["lo"], l)
                        continue

                    # session has ended (or hasn't started) -- if we just left
                    # an active session, finalize its high/low into pivots
                    if kz["in_session_seen"] and kz["hi_pivot"] is None:
                        kz["hi_pivot"] = kz["hi"]
                        kz["lo_pivot"] = kz["lo"]
                        kz["hi_active"] = True
                        kz["lo_active"] = True

                    # check breakout against active pivots (post-session only)
                    if kz["hi_active"] and cl > kz["hi_pivot"]:
                        events.append({
                            "timestamp": ts, "dir": "BUY",
                            "kz_name": kz["name"], "pivot_level": kz["hi_pivot"],
                            "pivot_kind": "session_high", "index_price": cl,
                        })
                        kz["hi_active"] = False  # one trade per pivot ("until mitigated")

                    if kz["lo_active"] and cl < kz["lo_pivot"]:
                        events.append({
                            "timestamp": ts, "dir": "SELL",
                            "kz_name": kz["name"], "pivot_level": kz["lo_pivot"],
                            "pivot_kind": "session_low", "index_price": cl,
                        })
                        kz["lo_active"] = False

        logger.info(f"Generated {len(events)} killzone breakout signal events across all days")
        return events

    # ── backtest engine: entries at next option-bar open, premium SL/TP, EOD ──

    def run_backtest(self) -> pd.DataFrame:
        c = self.cfg
        print("Starting ICT Killzone Breakout Backtest (ITM contracts)...")
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
            # Any fetch failure (invalid symbol, holiday-shifted expiry
            # mismatch, delisted strike, rate limit) is cached as an empty
            # DataFrame so we never retry the same bad symbol twice. A short
            # sleep between EVERY fetch throttles the burst that was hitting
            # Fyers' rate limit (code 429) when many distinct strikes/weeks
            # are touched in one run (this bot generates far more signals
            # than the SMC backtest, so more unique symbols get fetched).
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
        kz_name = None; pivot_level = None; pivot_kind = None

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
                            trades.append(self._close(direction, symbol, kz_name, pivot_kind, pivot_level,
                                                       entry_time, entry_price, pxt, px, "eod_exit", c.LOT))
                            has = False
                        break
                    px = float(bar["close"])
                    if px <= entry_price * (1 - c.STOP_LOSS_PCT / 100.0):
                        trades.append(self._close(direction, symbol, kz_name, pivot_kind, pivot_level,
                                                   entry_time, entry_price, bar_ts, px, "stop_loss", c.LOT))
                        has = False; continue
                    if px >= entry_price * (1 + c.TAKE_PROFIT_PCT / 100.0):
                        trades.append(self._close(direction, symbol, kz_name, pivot_kind, pivot_level,
                                                   entry_time, entry_price, bar_ts, px, "target", c.LOT))
                        has = False; continue

                if has and ts >= entry_eod_ts:
                    px, pxt = opt_price_after(opt_df, entry_eod_ts)
                    if px is not None:
                        trades.append(self._close(direction, symbol, kz_name, pivot_kind, pivot_level,
                                                   entry_time, entry_price, pxt, px, "eod_exit", c.LOT))
                    has = False

            if has:
                continue  # single-position engine, same as other backtests here

            in_win = c.TRADE_START <= ts.time() <= c.TRADE_END
            if not in_win:
                continue

            sym = ev["option_symbol"]
            opt_df = get_option_df(sym)
            # Enter using the option's price at/immediately AFTER the
            # breakout bar's own close (approximates an immediate market
            # order the instant the signal fires) -- NOT the following
            # bar's open, which was adding a full extra minute of lag on
            # top of the unavoidable 1-bar close-confirmation delay, and
            # was the source of entries feeling "late" after a breakout.
            entry_px, entry_ts_actual = opt_price_after(opt_df, ts, field="close")
            if entry_px is None:
                skipped_no_data += 1
                continue
            entry_px *= 1.005  # small slippage, matches other backtests' convention

            has = True; symbol = sym; direction = ev["dir"]
            entry_price = entry_px; entry_time = entry_ts_actual
            kz_name = ev["kz_name"]; pivot_level = ev["pivot_level"]; pivot_kind = ev["pivot_kind"]

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
                trades.append(self._close(direction, symbol, kz_name, pivot_kind, pivot_level,
                                           entry_time, entry_price, pxt, px, "backtest_end", c.LOT))

        logger.info(f"Signals skipped due to missing/invalid option data: {skipped_no_data}")
        logger.info(f"Signals using FALLBACK Tuesday-guess expiry (unverified against Fyers master): "
                    f"{self._fallback_expiry_count}")

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

        csv_path = "Killzone_Breakout_NIFTY_ITM_Backtest.csv"
        trades_df.to_csv(csv_path, index=False, encoding="utf-8")

        with open(csv_path, "a", encoding="utf-8") as f:
            for line in [
                "", "METRICS,VALUE",
                f"Total Trades,{total}", f"Wins,{wins}", f"Losses,{losses}",
                f"Win Rate (%),{wr}", f"Net PnL (Rs),{net:.2f}",
                f"Signals Skipped (no option data),{skipped_no_data}",
                f"Signals Using Fallback Expiry Guess,{self._fallback_expiry_count}",
                f"Lot Size,{c.LOT}",
            ]:
                f.write(line + "\n")

        print(f"\n{'='*55}")
        print(f"  Contract Type       : {c.MONEYNESS}")
        print(f"  Total Trades        : {total}")
        print(f"  Signals Skipped     : {skipped_no_data} (no option data)")
        print(f"  Fallback Expiries   : {self._fallback_expiry_count} (unverified guess)")
        print(f"  Win Rate            : {wr}%")
        print(f"  Net PnL             : Rs {net:,.2f}")
        print(f"{'='*55}")
        print(f"Results saved -> {csv_path}")
        return trades_df

    @staticmethod
    def _close(direction, symbol, kz_name, pivot_kind, pivot_level, entry_time, entry_price,
               exit_time, exit_price, reason, lot):
        pnl_pts = exit_price - entry_price
        pnl_rs = pnl_pts * lot
        return {
            "direction": direction, "option_symbol": symbol,
            "killzone": kz_name, "pivot_kind": pivot_kind, "pivot_level": round(pivot_level, 2),
            "entry_time": entry_time, "entry_price": round(entry_price, 2),
            "exit_time": exit_time, "exit_price": round(exit_price, 2),
            "exit_reason": reason,
            "pnl_pts": round(pnl_pts, 2), "pnl_rs": round(pnl_rs, 2),
            "result": "profit" if pnl_pts > 0 else "loss",
        }


# --------------------------- MAIN ---------------------------

if __name__ == "__main__":
    bot    = KillzoneBacktestBot()
    trades = bot.run_backtest()