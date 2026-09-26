"""
UT Bot (ATR Trailing Stop) OPTION backtest.
Direct Python port of the Pine Script v4 "UT Bot" indicator (credit: HPotter).
Causal — only CLOSED bars are used, so there's no repaint. Buys 1-strike ITM
options in the direction of the trailing-stop flip; holds until the next
flip or EOD. No premium SL/TP (the strategy's own risk control IS the
underlying trailing stop) — matches the SupertrendPoll.py backtest convention.

Two signal sources are available (SIGNAL_SOURCE config):
  "pos"       -> Pine's `pos` state variable (drives bar coloring / xcolor).
                 Both comparisons use the PREVIOUS bar's stop value
                 (xATRTrailingStop[1]) for src[1] AND src -- this is the
                 literal Pine translation, verified against the source.
  "crossover" -> Pine's plotted buy/sell labels, computed via TradingView's
                 built-in crossover()/crossunder(), which compares src vs stop
                 at MATCHING indices each step (src[i] vs stop[i], src[i-1] vs
                 stop[i-1]) rather than a fixed reference. This is what
                 actually drives the alertcondition()s in the original script.
  These two can very occasionally disagree by a bar near a tight threshold;
  "pos" is the default since it's what the strategy's directional STATE is
  actually based on (bar coloring), not just the plotted markers.
"""
import pandas as pd, numpy as np, requests, logging, os, calendar, pytz, urllib3, warnings
from datetime import datetime, timedelta, time as dtime
from typing import Optional, Dict

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
warnings.filterwarnings("ignore")
IST = pytz.timezone("Asia/Kolkata")

logging.basicConfig(level=logging.INFO,
                     format="%(asctime)s - UTBotBacktest - %(levelname)s - %(message)s",
                     datefmt="%Y-%m-%d %H:%M:%S")
logger = logging.getLogger("UTBotBacktest")

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
    UNDERLYING = "NSE:NIFTY50-INDEX"; BASE_RES = "1"; HIST_DAYS = 15
    UT_RULE = "10s"           # timeframe the ATR trailing stop runs on
    KEY_VALUE = 10         # Pine "keyvalue" — sensitivity multiplier
    ATR_PERIOD = 20              # Pine "atrperiod"
    SIGNAL_SOURCE = "pos"          # "pos" (bar-color state) or "crossover" (buy/sell labels)
    STRIKE_STEP = 100; LOT = 65; MONEYNESS = "ITM"
    SESSION_START = dtime(9, 20); SESSION_END = dtime(15, 30)
    TRADE_START = dtime(9, 30); TRADE_END = dtime(15, 0); EOD_EXIT_TIME = dtime(15, 20)

    def __init__(self):
        self.EXPIRY_CODE = expiry_code(get_next_tuesday())
        logger.info(f"Weekly expiry code {self.EXPIRY_CODE} | UT Bot(key={self.KEY_VALUE},"
                    f"atr={self.ATR_PERIOD}) {self.UT_RULE} | signal={self.SIGNAL_SOURCE} | "
                    f"moneyness={self.MONEYNESS}")

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

# ---------- UT Bot (direct port of the Pine v4 script) ----------
def compute_ut_bot(r: pd.DataFrame, key_value: float, atr_period: int):
    """
    r: OHLC DataFrame (one row per bar, e.g. resampled 10min bars).
    Returns (stop, pos, buy, sell):
      stop -> xATRTrailingStop (np.ndarray, float)
      pos  -> Pine `pos` state: 1 (long), -1 (short), 0 (undefined/warmup)
      buy  -> bool array, TradingView crossover(src, stop) equivalent
      sell -> bool array, TradingView crossunder(src, stop) equivalent
    """
    src = r["close"].to_numpy()
    high = r["high"].to_numpy(); low = r["low"].to_numpy()
    n = len(r)

    # Pine's atr() = RMA (Wilder's smoothing) of True Range.
    prev_close = np.empty(n); prev_close[0] = src[0]; prev_close[1:] = src[:-1]
    tr = np.maximum(high - low, np.maximum(np.abs(high - prev_close), np.abs(low - prev_close)))
    atr = pd.Series(tr).ewm(alpha=1 / atr_period, adjust=False).mean().to_numpy()
    n_loss = key_value * atr

    stop = np.zeros(n)
    pos = np.zeros(n, dtype=int)
    buy = np.zeros(n, dtype=bool)
    sell = np.zeros(n, dtype=bool)

    for i in range(n):
        prev_stop = stop[i - 1] if i > 0 else 0.0
        prev_src = src[i - 1] if i > 0 else src[i]

        # xATRTrailingStop
        if src[i] > prev_stop and prev_src > prev_stop:
            stop[i] = max(prev_stop, src[i] - n_loss[i])
        elif src[i] < prev_stop and prev_src < prev_stop:
            stop[i] = min(prev_stop, src[i] + n_loss[i])
        else:
            stop[i] = src[i] - n_loss[i] if src[i] > prev_stop else src[i] + n_loss[i]

        if i == 0:
            pos[i] = 0
            continue

        # pos -- FIXED: both comparisons use prev_stop (xATRTrailingStop[1]),
        # exactly matching the Pine line:
        #   pos := iff(src[1] < stop[1] and src > stop[1], 1,
        #          iff(src[1] > stop[1] and src < stop[1], -1, pos[1]))
        if prev_src < prev_stop and src[i] > prev_stop:
            pos[i] = 1
        elif prev_src > prev_stop and src[i] < prev_stop:
            pos[i] = -1
        else:
            pos[i] = pos[i - 1]

        # buy/sell -- TradingView crossover()/crossunder(): matching-index
        # comparison (src[i] vs stop[i], src[i-1] vs stop[i-1]), NOT the
        # fixed-prev-stop comparison pos uses. This is what actually drives
        # the plotted labels / alertcondition()s in the original script.
        buy[i] = (prev_src <= prev_stop) and (src[i] > stop[i])
        sell[i] = (prev_src >= prev_stop) and (src[i] < stop[i])

    return stop, pos, buy, sell

# ---------- backtest bot ----------
class UTBotBacktestBot:
    def __init__(self):
        self.cfg = Config()
        self.fy = self._init_fyers()

    def _init_fyers(self) -> Fyers:
        for f in ["fyers_appid.txt", "fyers_token.txt"]:
            if not os.path.exists(f): raise FileNotFoundError(f)
        app = open("fyers_appid.txt").read().strip()
        tok = open("fyers_token.txt").read().strip()
        fy = Fyers(app, tok); p = fy.profile()
        if p.get("s") != "ok":
            raise ConnectionError(f"Fyers auth failed — re-run GenerateToken.py.\nResponse: {p}")
        logger.info(f"Fyers connected — {p['data']['name']} ✅")
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
        logger.info(f"Loaded NIFTY50 index data: {len(df)} rows")
        return df

    def _fetch_option_history(self, symbol: str) -> pd.DataFrame:
        return self._fetch_history(symbol, self.cfg.HIST_DAYS).sort_index()

    def _opt_symbol(self, spot: float, direction: str) -> str:
        c = self.cfg
        atm = int(round(spot / c.STRIKE_STEP) * c.STRIKE_STEP); itm = c.MONEYNESS == "ITM"
        if direction == "BUY":
            strike = atm - c.STRIKE_STEP if itm else atm; typ = "CE"
        else:
            strike = atm + c.STRIKE_STEP if itm else atm; typ = "PE"
        return f"NSE:NIFTY{c.EXPIRY_CODE}{strike:05d}{typ}"

    # ── signal generation: replays a UT-Bot poll loop day by day, closed-bars only ──

    def generate_signals(self, df_1min: pd.DataFrame) -> pd.DataFrame:
        c = self.cfg
        events = []
        use_crossover = (c.SIGNAL_SOURCE == "crossover")

        for day, day_df in df_1min.groupby(df_1min.index.date):
            r = day_df.resample(c.UT_RULE).agg(
                {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
            ).dropna()
            if len(r) < c.ATR_PERIOD + 2:
                continue

            stop, pos, buy, sell = compute_ut_bot(r, c.KEY_VALUE, c.ATR_PERIOD)

            held_dir = None
            for i in range(1, len(r)):
                bar_close_time = r.index[i] + pd.Timedelta(c.UT_RULE)
                ts_time = bar_close_time.time()
                if not (c.SESSION_START <= ts_time <= c.SESSION_END):
                    continue

                if use_crossover:
                    if buy[i]:
                        desired = "BUY"
                    elif sell[i]:
                        desired = "SELL"
                    else:
                        continue  # no flip this bar
                else:
                    if pos[i] == 0:
                        continue
                    desired = "BUY" if pos[i] == 1 else "SELL"

                if held_dir is None:
                    held_dir = desired  # adopt baseline silently, same as Supertrend bot
                    continue
                if held_dir == desired:
                    continue

                in_win = c.TRADE_START <= ts_time <= c.TRADE_END

                events.append({
                    "timestamp": bar_close_time, "action": "EXIT", "reason": "flip_exit",
                    "direction": held_dir, "index_price": float(r["close"].iloc[i]),
                })

                if in_win:
                    events.append({
                        "timestamp": bar_close_time, "action": "ENTRY", "reason": "flip_entry",
                        "direction": desired, "index_price": float(r["close"].iloc[i]),
                        "option_symbol": self._opt_symbol(float(r["close"].iloc[i]), desired),
                    })
                    held_dir = desired
                else:
                    held_dir = None

            if held_dir is not None:
                eod_ts = IST.localize(datetime.combine(day, c.EOD_EXIT_TIME))
                events.append({
                    "timestamp": eod_ts, "action": "EXIT", "reason": "eod_exit",
                    "direction": held_dir, "index_price": float(r["close"].iloc[-1]),
                })

        events_df = pd.DataFrame(events)
        logger.info(f"Generated {len(events_df)} signal events "
                    f"({(events_df['action']=='ENTRY').sum() if not events_df.empty else 0} entries) "
                    f"[source={c.SIGNAL_SOURCE}]")
        return events_df

    # ── backtest engine: pairs ENTRY/EXIT events, fetches option prices, computes P&L ──

    def run_backtest(self) -> pd.DataFrame:
        print(f"Starting UT Bot Backtest (ITM contracts, signal={self.cfg.SIGNAL_SOURCE}, "
              f"no SL/TP — trend-flip + EOD exit only)...")
        df_1min = self.get_index_data()
        events = self.generate_signals(df_1min)

        if events.empty:
            print("No signals generated")
            return events

        entries = events[events["action"] == "ENTRY"].reset_index(drop=True)
        exits   = events[events["action"] == "EXIT"].reset_index(drop=True).copy()

        if entries.empty:
            print("No entries generated")
            return events

        option_cache: Dict[str, pd.DataFrame] = {}
        for sym in entries["option_symbol"].unique():
            try:
                option_cache[sym] = self._fetch_option_history(sym)
            except ValueError as e:
                logger.warning(f"Skipping {sym}: {e}")

        trades = []
        exit_ptr = 0

        for i, entry in entries.iterrows():
            opt_symbol = entry["option_symbol"]
            direction  = entry["direction"]
            entry_ts   = entry["timestamp"]

            exit_row = None
            for j in range(exit_ptr, len(exits)):
                cand = exits.iloc[j]
                if cand["timestamp"] >= entry_ts and cand["direction"] == direction:
                    exit_row = cand
                    exit_ptr = j + 1
                    break
            if exit_row is None:
                logger.warning(f"No matching exit found for entry @ {entry_ts} ({direction}) — skipping.")
                continue

            if opt_symbol not in option_cache:
                trades.append(self._skip(entry_ts, direction, opt_symbol, "no_option_data"))
                continue

            opt_df = option_cache[opt_symbol]

            entry_pos = opt_df.index.searchsorted(entry_ts, side="right")
            if entry_pos >= len(opt_df):
                trades.append(self._skip(entry_ts, direction, opt_symbol, "no_entry_bar"))
                continue
            entry_bar   = opt_df.iloc[int(entry_pos)]
            entry_price = float(entry_bar["open"]) * 1.005
            entry_time  = opt_df.index[int(entry_pos)]

            exit_ts  = exit_row["timestamp"]
            exit_pos = opt_df.index.searchsorted(exit_ts, side="right")
            if exit_pos >= len(opt_df):
                exit_pos = len(opt_df) - 1
            exit_bar   = opt_df.iloc[int(exit_pos)]
            exit_price = float(exit_bar["close"])
            exit_time  = opt_df.index[int(exit_pos)]

            pnl_pts = exit_price - entry_price
            pnl_rs  = pnl_pts * self.cfg.LOT

            trades.append({
                "direction": direction, "option_symbol": opt_symbol,
                "entry_time": entry_time, "entry_price": round(entry_price, 2),
                "exit_time": exit_time, "exit_price": round(exit_price, 2),
                "exit_reason": exit_row["reason"],
                "pnl_pts": round(pnl_pts, 2), "pnl_rs": round(pnl_rs, 2),
                "result": "profit" if pnl_pts > 0 else "loss",
            })

        trades_df = pd.DataFrame(trades)
        if trades_df.empty:
            print("No trades executed")
            return trades_df

        trades_df["cum_pnl_rs"] = trades_df["pnl_rs"].cumsum()

        traded = trades_df[~trades_df["result"].isin(["skipped"])]
        total_trades = len(traded)
        wins   = int((traded["result"] == "profit").sum())
        losses = int((traded["result"] == "loss").sum())
        net    = traded["pnl_rs"].sum()
        wr     = round(wins / total_trades * 100, 1) if total_trades > 0 else 0.0

        csv_path = f"UTBot_{self.cfg.SIGNAL_SOURCE}_NIFTY_ITM_Backtest.csv"
        trades_df.to_csv(csv_path, index=False, encoding="utf-8")

        with open(csv_path, "a", encoding="utf-8") as f:
            for line in [
                "", "METRICS,VALUE",
                f"Total Trades,{total_trades}", f"Wins,{wins}", f"Losses,{losses}",
                f"Win Rate (%),{wr}", f"Net PnL (Rs),{net:.2f}",
                f"Signal Source,{self.cfg.SIGNAL_SOURCE}",
                f"Key Value,{self.cfg.KEY_VALUE}", f"ATR Period,{self.cfg.ATR_PERIOD}",
                f"UT Rule,{self.cfg.UT_RULE}", f"Lot Size,{self.cfg.LOT}",
            ]:
                f.write(line + "\n")

        print(f"\n{'='*55}")
        print(f"  Contract Type       : {self.cfg.MONEYNESS}")
        print(f"  Signal Source       : {self.cfg.SIGNAL_SOURCE}")
        print(f"  Total Trades        : {total_trades}")
        print(f"  Win Rate            : {wr}%")
        print(f"  Net PnL             : Rs {net:,.2f}")
        print(f"{'='*55}")
        print(f"Results saved → {csv_path}")
        return trades_df

    @staticmethod
    def _skip(ts, direction, symbol, reason):
        return {
            "direction": direction, "option_symbol": symbol,
            "entry_time": ts, "entry_price": np.nan,
            "exit_time": pd.NaT, "exit_price": np.nan,
            "exit_reason": reason, "pnl_pts": 0.0, "pnl_rs": 0.0,
            "result": "skipped",
        }


# --------------------------- MAIN ---------------------------

if __name__ == "__main__":
    bot    = UTBotBacktestBot()
    trades = bot.run_backtest()