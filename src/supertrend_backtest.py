"""
Supertrend 10-min OPTION backtest — offline replay of SupertrendPoll.Engine's
exact live logic: hold the option in the direction of the CURRENT closed
10-min Supertrend(10,3.0); flip when the closed-bar trend flips. No SL/TP
(matches the live bot, which relies purely on trend flips + EOD exit).
Buys 1-strike ITM options by default.
"""
import pandas as pd, numpy as np, requests, logging, os, calendar, pytz, urllib3, warnings
from datetime import datetime, timedelta, time as dtime
from typing import Optional, Dict

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
warnings.filterwarnings("ignore")
IST = pytz.timezone("Asia/Kolkata")

logging.basicConfig(level=logging.INFO,
                     format="%(asctime)s - STBacktest - %(levelname)s - %(message)s",
                     datefmt="%Y-%m-%d %H:%M:%S")
logger = logging.getLogger("STBacktest")

# ---------- expiry helpers (identical to SupertrendPoll.py) ----------
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

# ---------- config (identical to SupertrendPoll.py) ----------
class Config:
    UNDERLYING = "NSE:NIFTY50-INDEX"; BASE_RES = "1"; HIST_DAYS = 60
    ST_RULE = "10min"; ST_PERIOD = 7; ST_MULT = 1.0
    STRIKE_STEP = 100; LOT = 65; MONEYNESS = "ITM"      # "ATM" or "ITM"
    SESSION_START = dtime(9, 20); SESSION_END = dtime(15, 30)
    TRADE_START = dtime(9, 30); TRADE_END = dtime(15, 0); EOD_EXIT_TIME = dtime(15, 20)

    def __init__(self):
        self.EXPIRY_CODE = expiry_code(get_next_tuesday())
        logger.info(f"Weekly expiry code {self.EXPIRY_CODE} | ST({self.ST_PERIOD},{self.ST_MULT}) "
                    f"{self.ST_RULE} | moneyness={self.MONEYNESS}")

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

# ---------- supertrend (identical to SupertrendPoll.py) ----------
def supertrend_trend(r, period, mult):
    h = r["high"].to_numpy(); l = r["low"].to_numpy(); cl = r["close"].to_numpy(); n = len(r)
    if n < 2: return np.ones(n, int)
    pc = np.empty(n); pc[0] = cl[0]; pc[1:] = cl[:-1]
    trv = np.maximum(h - l, np.maximum(np.abs(h - pc), np.abs(l - pc)))
    atr = pd.Series(trv).ewm(alpha=1 / period, adjust=False).mean().to_numpy()
    hl2 = (h + l) / 2; ub = hl2 + mult * atr; db = hl2 - mult * atr
    fu = np.empty(n); fl = np.empty(n); fu[0] = ub[0]; fl[0] = db[0]
    for i in range(1, n):
        fu[i] = ub[i] if (ub[i] < fu[i - 1] or cl[i - 1] > fu[i - 1]) else fu[i - 1]
        fl[i] = db[i] if (db[i] > fl[i - 1] or cl[i - 1] < fl[i - 1]) else fl[i - 1]
    t = np.ones(n, int)
    for i in range(1, n):
        t[i] = (-1 if cl[i] < fl[i] else 1) if t[i - 1] == 1 else (1 if cl[i] > fu[i] else -1)
    return t

# ---------- backtest bot ----------
class SupertrendBacktestBot:
    def __init__(self):
        self.cfg = Config()
        self.fy = self._init_fyers()

    def _init_fyers(self) -> Fyers:
        for f in ["fyers_appid.txt", "fyers_token.txt"]:
            if not os.path.exists(f): raise FileNotFoundError(f"Missing {f}")
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

    # ── signal generation: replays SupertrendPoll.Engine._poll exactly, day by day ──

    def generate_signals(self, df_1min: pd.DataFrame) -> pd.DataFrame:
        c = self.cfg
        signals = []

        for day, day_df in df_1min.groupby(df_1min.index.date):
            r = day_df.resample(c.ST_RULE).agg(
                {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
            ).dropna()
            if len(r) < 3:
                continue

            t = supertrend_trend(r, c.ST_PERIOD, c.ST_MULT)

            baseline = None
            held_dir = None

            # Walk bar-by-bar as CLOSED bars become available — bar i's trend
            # is only "known" once bar i has fully closed, i.e. at time
            # r.index[i] + ST_RULE. This mirrors the live engine's
            # "drop forming bar" behaviour without needing a live clock.
            for i in range(2, len(r)):
                bar_close_time = r.index[i] + pd.Timedelta(c.ST_RULE)
                ts_time = bar_close_time.time()

                if not (c.SESSION_START <= ts_time <= c.SESSION_END):
                    continue

                desired = "BUY" if t[i] == 1 else "SELL"

                if baseline is None and held_dir is None:
                    baseline = desired
                    continue

                if held_dir == desired:
                    continue  # already aligned, hold

                in_win = c.TRADE_START <= ts_time <= c.TRADE_END

                # flip_exit for the currently held position, if any
                if held_dir is not None:
                    signals.append({
                        "timestamp": bar_close_time,
                        "action": "EXIT",
                        "reason": "flip_exit",
                        "direction": held_dir,
                        "index_price": float(r["close"].iloc[i]),
                    })

                if in_win:
                    signals.append({
                        "timestamp": bar_close_time,
                        "action": "ENTRY",
                        "reason": "flip_entry",
                        "direction": desired,
                        "index_price": float(r["close"].iloc[i]),
                        "option_symbol": self._opt_symbol(float(r["close"].iloc[i]), desired),
                    })
                    held_dir = desired
                else:
                    held_dir = None  # flat outside trade window

                baseline = desired

            # EOD forced exit for the day, if still holding
            if held_dir is not None:
                eod_ts = IST.localize(datetime.combine(day, c.EOD_EXIT_TIME))
                signals.append({
                    "timestamp": eod_ts,
                    "action": "EXIT",
                    "reason": "eod_exit",
                    "direction": held_dir,
                    "index_price": float(r["close"].iloc[-1]),
                })

        signals_df = pd.DataFrame(signals)
        logger.info(f"Generated {len(signals_df)} signal events "
                    f"({(signals_df['action']=='ENTRY').sum() if not signals_df.empty else 0} entries)")
        return signals_df

    # ── backtest engine: pairs ENTRY/EXIT events, fetches option prices, computes P&L ──

    def run_backtest(self) -> pd.DataFrame:
        print("Starting Supertrend 10-min Backtest (ITM contracts, no SL/TP — trend-flip + EOD exit only)...")
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
        exit_ptr = 0  # exits are chronological and 1:1 paired with entries in order

        for i, entry in entries.iterrows():
            opt_symbol = entry["option_symbol"]
            direction  = entry["direction"]
            entry_ts   = entry["timestamp"]

            # matching exit is the next exit event at/after this entry with the same direction
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
            entry_price = float(entry_bar["open"]) * 1.005  # small slippage, matches VAM backtest convention
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

        csv_path = "Supertrend_NIFTY_ITM_Backtest.csv"
        trades_df.to_csv(csv_path, index=False, encoding="utf-8")

        with open(csv_path, "a", encoding="utf-8") as f:
            for line in [
                "", "METRICS,VALUE",
                f"Total Trades,{total_trades}", f"Wins,{wins}", f"Losses,{losses}",
                f"Win Rate (%),{wr}", f"Net PnL (Rs),{net:.2f}",
                f"ST Period,{self.cfg.ST_PERIOD}", f"ST Mult,{self.cfg.ST_MULT}",
                f"ST Rule,{self.cfg.ST_RULE}", f"Lot Size,{self.cfg.LOT}",
            ]:
                f.write(line + "\n")

        print(f"\n{'='*55}")
        print(f"  Contract Type       : {self.cfg.MONEYNESS}")
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
    bot    = SupertrendBacktestBot()
    trades = bot.run_backtest()