"""
SwiftTrend index-flip backtest — offline replay of the live SwiftTrendBot +
LiveEngine confirmation logic, WITHOUT the HTF directional filter (removed —
it was repainting live and too slow/coarse at 120min to be useful). Trades
every ST flip that the position-state guard allows. compute_swift_trend() is
reused verbatim so signal math matches exactly. Live entries execute at the
OPEN of the bar AFTER the signal bar (1-bar confirmation) — this backtest
replicates that by entering at the option's open price at/after the next bar's
timestamp. SL/TP are simulated on 1-min option data as % of entry premium.
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, time as dtime
import requests
import logging
import os
from typing import Dict, Optional
import warnings
import pytz
import calendar
import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
IST = pytz.timezone("Asia/Kolkata")

logging.basicConfig(level=logging.INFO,
                     format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# --------------------------- EXPIRY HELPERS (identical to live) ---------------------------

def get_next_tuesday(now_ist: Optional[datetime] = None) -> datetime:
    now_ist = now_ist or datetime.now(IST)
    if now_ist.weekday() < 1:
        days_ahead = 1 - now_ist.weekday()
    else:
        days_ahead = 7 - (now_ist.weekday() - 1)
    return (now_ist + timedelta(days=days_ahead)).replace(
        hour=0, minute=0, second=0, microsecond=0
    )

def get_nifty_monthly_expiry(year: int, month: int) -> datetime:
    last_day = calendar.monthrange(year, month)[1]
    dt = datetime(year, month, last_day)
    while dt.weekday() != 1:
        dt -= timedelta(days=1)
    return dt

def get_current_or_next_monthly_expiry(now_ist: Optional[datetime] = None) -> datetime:
    now_ist = now_ist or datetime.now(IST)
    y, m = now_ist.year, now_ist.month
    exp_dt = get_nifty_monthly_expiry(y, m)
    if exp_dt.date() < now_ist.date():
        if m == 12:
            y += 1; m = 1
        else:
            m += 1
        exp_dt = get_nifty_monthly_expiry(y, m)
    return exp_dt

def format_expiry_code(dt: datetime) -> str:
    yy = dt.year % 100
    mon = dt.strftime("%b").upper()
    return f"{yy:02d}{mon}"

# --------------------------- CONFIG ---------------------------

class TradingConfig:
    UNDERLYING_SYMBOL = "NSE:NIFTY50-INDEX"
    TIMEFRAME = "1"
    HIST_DAYS = 60

    STRIKE_STEP = 100
    NIFTY_LOT_SIZE = 65

    EXPIRY_TYPE = "WEEKLY"

    ST_PERIOD    = 100
    ST_BAND_MULT = 0.5
    ST_TOL_MULT  = 0.5

    # ── HTF filter removed entirely (was repainting live, too slow at 120min) ──

    SESSION_START = dtime(9, 20)
    SESSION_END   = dtime(15, 30)

    TRADE_START = dtime(9, 30)
    TRADE_END   = dtime(15, 0)
    EOD_EXIT_TIME = dtime(15, 20)

    STOP_LOSS_PCT   = -20.0
    TAKE_PROFIT_PCT = 20.0

    OPTION_TICK_SIZE = 0.05
    BO_QTY = 65

    def __init__(self):
        now = datetime.now(IST)
        exp_type = self.EXPIRY_TYPE.upper()
        if exp_type == "WEEKLY":
            exp_dt = get_next_tuesday(now); kind = "WEEKLY"
        else:
            exp_dt = get_current_or_next_monthly_expiry(now); kind = "MONTHLY"
        self.EXPIRY_CODE = format_expiry_code(exp_dt)
        logger.info(f"Using NIFTY {kind} expiry {exp_dt.date()} with code {self.EXPIRY_CODE}")

# --------------------------- FYERS HTTP CLIENT ---------------------------

class FyersHTTP:
    BASE_T1 = "https://api-t1.fyers.in/api/v3"
    BASE_T2 = "https://api-t1.fyers.in/data"

    def __init__(self, app_id: str, token: str):
        combined = f"{app_id}:{token}" if ":" not in token else token
        self.session = requests.Session()
        self.session.verify = False
        self.session.headers.update({"Authorization": combined})
        logger.info(f"FyersHTTP init | app_id={app_id} | auth={combined[:40]}...")

    def get_profile(self) -> dict:
        return self.session.get(f"{self.BASE_T1}/profile").json()

    def history(self, data: dict) -> dict:
        params = {
            "symbol":      data["symbol"],
            "resolution":  data["resolution"],
            "date_format": data.get("date_format", "1"),
            "range_from":  data["range_from"],
            "range_to":    data["range_to"],
            "cont_flag":   data.get("cont_flag", "0"),
        }
        r = self.session.get(f"{self.BASE_T2}/history", params=params)
        try:
            return r.json()
        except Exception as e:
            raise ValueError(f"History API non-JSON (status={r.status_code}): {r.text[:300]}") from e

# --------------------------- SWIFTTREND CORE (VERBATIM from live bot) ---------------------------

class SwiftTrendParams:
    def __init__(self, period=100, band_multiplier=2.5, tolerance_multiplier=2.5):
        self.period = period
        self.band_multiplier = band_multiplier
        self.tolerance_multiplier = tolerance_multiplier

def compute_swift_trend(df: pd.DataFrame,
                        params: SwiftTrendParams,
                        open_col="open", high_col="high",
                        low_col="low", close_col="close") -> pd.DataFrame:
    o = df[open_col].values
    h = df[high_col].values
    l = df[low_col].values
    c = df[close_col].values
    n = len(df)

    period = params.period
    band_multiplier = params.band_multiplier
    tolerance_multiplier = params.tolerance_multiplier

    body = np.abs(o - c)
    avg_body = pd.Series(body).rolling(window=period, min_periods=period).mean().values
    body_mid = (o + c) / 2.0

    basic_upper = np.full(n, np.nan)
    basic_lower = np.full(n, np.nan)
    tolerance = np.full(n, np.nan)
    avg_line = np.full(n, np.nan)
    margin_line_base = np.full(n, np.nan)
    is_uptrend_arr = np.full(n, True, dtype=bool)
    trend_changed = np.full(n, False, dtype=bool)
    signal = [None] * n

    for i in range(n):
        if i > 0 and not np.isnan(avg_body[i - 1]):
            basic_upper[i] = body_mid[i - 1] + band_multiplier * avg_body[i - 1]
            basic_lower[i] = body_mid[i - 1] - band_multiplier * avg_body[i - 1]

        if not np.isnan(avg_body[i]):
            tolerance[i] = tolerance_multiplier * avg_body[i]

        prev_avg_line = avg_line[i - 1] if i > 0 else np.nan
        prev_is_uptrend = is_uptrend_arr[i - 1] if i > 0 else True

        if np.isnan(prev_avg_line):
            avg_line[i] = basic_lower[i]
            is_uptrend_arr[i] = True
        else:
            if prev_is_uptrend:
                avg_line[i] = np.nanmax([basic_lower[i], prev_avg_line]) if not np.isnan(basic_lower[i]) else prev_avg_line
            else:
                avg_line[i] = np.nanmin([basic_upper[i], prev_avg_line]) if not np.isnan(basic_upper[i]) else prev_avg_line
            is_uptrend_arr[i] = prev_is_uptrend

        if not np.isnan(avg_line[i]) and not np.isnan(tolerance[i]):
            margin_line_base[i] = avg_line[i] - tolerance[i] if is_uptrend_arr[i] else avg_line[i] + tolerance[i]

        trend_changed[i] = False
        if not np.isnan(margin_line_base[i]):
            if is_uptrend_arr[i] and c[i] < margin_line_base[i]:
                is_uptrend_arr[i] = False
                trend_changed[i] = True
                signal[i] = "Sell"
            elif (not is_uptrend_arr[i]) and c[i] > margin_line_base[i]:
                is_uptrend_arr[i] = True
                trend_changed[i] = True
                signal[i] = "Buy"

        if not trend_changed[i]:
            signal[i] = None

    out = df.copy()
    out["st_avg_line"] = avg_line
    out["st_margin"] = margin_line_base
    out["st_is_uptrend"] = is_uptrend_arr
    out["st_trend_changed"] = trend_changed
    out["st_signal"] = signal
    return out

# --------------------------- SWIFTTREND BACKTEST BOT ---------------------------

class SwiftTrendBacktestBot:
    def __init__(self):
        self.config = TradingConfig()
        self.fyers = self._init_fyers()
        self.position_size: int = 0
        self.last_trade: Optional[str] = None

    def _init_fyers(self) -> FyersHTTP:
        for f in ["fyers_appid.txt", "fyers_token.txt"]:
            if not os.path.exists(f):
                raise FileNotFoundError(f"Missing {f}")
        app_id = open("fyers_appid.txt").read().strip()
        token  = open("fyers_token.txt").read().strip()
        if not app_id or not token:
            raise ValueError("Empty app_id or token")
        client  = FyersHTTP(app_id, token)
        profile = client.get_profile()
        if profile.get("s") != "ok":
            raise ConnectionError(f"Fyers auth failed — re-run GenerateToken.py.\nResponse: {profile}")
        logger.info(f"Fyers connected — {profile['data']['name']} ✅")
        return client

    def _fetch_history(self, symbol: str, days: int) -> pd.DataFrame:
        end_dt   = datetime.now(IST)
        start_dt = end_dt - timedelta(days=days)
        data = dict(symbol=symbol, resolution=self.config.TIMEFRAME,
                    date_format="1", range_from=start_dt.strftime("%Y-%m-%d"),
                    range_to=end_dt.strftime("%Y-%m-%d"), cont_flag="0")
        resp = self.fyers.history(data=data)
        if isinstance(resp, dict):
            if resp.get("s") != "ok":
                raise ValueError(f"API error for {symbol}: {resp}")
            candles = resp.get("candles", [])
        else:
            candles = []
        if not candles:
            raise ValueError(f"No historical data for {symbol}")
        df = pd.DataFrame(candles, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s", utc=True).dt.tz_convert(IST)
        return df.sort_values("timestamp").reset_index(drop=True)

    def get_index_data(self, days: int = None) -> pd.DataFrame:
        days = days or self.config.HIST_DAYS
        df = self._fetch_history(self.config.UNDERLYING_SYMBOL, days)
        df.set_index("timestamp", inplace=True)
        mask = df.index.indexer_between_time(self.config.SESSION_START, self.config.SESSION_END)
        df = df.iloc[mask]
        logger.info(f"Loaded NIFTY50 index data: {len(df)} rows")
        return df

    def _fetch_option_history(self, symbol: str, days: int) -> pd.DataFrame:
        df = self._fetch_history(symbol, days)
        df.set_index("timestamp", inplace=True)
        return df

    def _pick_itm_option_symbol(self, underlying_price: float, direction: str) -> str:
        c = self.config
        spot = float(underlying_price)
        step = c.STRIKE_STEP
        if direction == "BUY":
            atm_strike = (int(spot) // step) * step
            strike = atm_strike - step
            opt_type = "CE"
        else:
            atm_strike = ((int(spot) + step - 1) // step) * step
            strike = atm_strike + step
            opt_type = "PE"
        strike = max(step, (int(strike) // step) * step)
        return f"NSE:NIFTY{c.EXPIRY_CODE}{int(strike):05d}{opt_type}"

    # ── signal generation — no HTF filter, pure ST flip + position-state guard ──

    def generate_swifttrend_signals(self, df_index: pd.DataFrame) -> pd.DataFrame:
        df = df_index.copy().reset_index()

        st_params = SwiftTrendParams(
            period=self.config.ST_PERIOD,
            band_multiplier=self.config.ST_BAND_MULT,
            tolerance_multiplier=self.config.ST_TOL_MULT
        )
        df_st = compute_swift_trend(df, st_params)

        signals = []
        warmup = self.config.ST_PERIOD

        for i in range(warmup, len(df_st)):
            row = df_st.iloc[i]
            prev_pos_ok_long  = (self.position_size <= 0) and (self.last_trade != "long")
            prev_pos_ok_short = (self.position_size >= 0) and (self.last_trade != "short")

            ts = row["timestamp"]
            ts_time = ts.time()
            in_trade_window = (ts_time >= self.config.TRADE_START) and (ts_time <= self.config.TRADE_END)

            if bool(row["st_trend_changed"]) and in_trade_window:
                if row["st_is_uptrend"] and prev_pos_ok_long:
                    sig = self._create_signal(row, "BUY")
                    sig["strategy_type"] = "swifttrend_flip"
                    signals.append(sig)
                elif (not row["st_is_uptrend"]) and prev_pos_ok_short:
                    sig = self._create_signal(row, "SELL")
                    sig["strategy_type"] = "swifttrend_flip"
                    signals.append(sig)

        signals_df = pd.DataFrame(signals) if signals else pd.DataFrame()
        logger.info(f"Generated {len(signals_df)} SwiftTrend index signals (no HTF filter)")
        return signals_df

    def _create_signal(self, row: pd.Series, direction: str) -> Dict:
        und_price = row["close"]
        opt_symbol = self._pick_itm_option_symbol(underlying_price=und_price, direction=direction)
        signal = {
            "timestamp": row["timestamp"],
            "index_symbol": self.config.UNDERLYING_SYMBOL,
            "index_price": float(und_price),
            "direction": direction,
            "option_symbol": opt_symbol,
        }
        if direction == "BUY":
            self.position_size = 1; self.last_trade = "long"
        else:
            self.position_size = -1; self.last_trade = "short"
        return signal

    # ── backtest engine: 1-bar confirmation entry + SL/TP simulation ──

    def run_backtest(self) -> pd.DataFrame:
        print("Starting SwiftTrend Backtest — NO HTF filter (ITM contracts, 1-bar confirmation entry)...")
        df_index = self.get_index_data()
        signals  = self.generate_swifttrend_signals(df_index)

        if signals.empty:
            print("No SwiftTrend signals generated")
            return signals

        signals_sorted = signals.sort_values("timestamp").reset_index(drop=True)

        option_cache: Dict[str, pd.DataFrame] = {}
        for sym in signals_sorted["option_symbol"].unique():
            try:
                option_cache[sym] = self._fetch_option_history(sym, self.config.HIST_DAYS).sort_index()
            except ValueError as e:
                logger.warning(f"Skipping {sym}: {e}")

        step_min = int(self.config.TIMEFRAME)
        sl_pct = abs(self.config.STOP_LOSS_PCT)
        tp_pct = self.config.TAKE_PROFIT_PCT
        EOD_EXIT = self.config.EOD_EXIT_TIME

        cols = dict(
            entry_price_option=[], entry_time=[], exit_time=[], exit_price_option=[],
            pnl_pct=[], pnl_rupees_1lot=[], result=[], exit_reason=[],
        )

        for i, s in signals_sorted.iterrows():
            signal_ts  = s["timestamp"]
            opt_symbol = s["option_symbol"]
            direction  = s["direction"]

            confirm_ts = signal_ts + pd.Timedelta(minutes=step_min)

            eod_boundary = IST.localize(datetime.combine(signal_ts.date(), EOD_EXIT))
            if confirm_ts >= eod_boundary:
                self._append_skip(cols, "confirm_after_eod")
                continue

            next_flip_ts = None
            for j in range(i + 1, len(signals_sorted)):
                nxt = signals_sorted.iloc[j]
                if nxt["direction"] != direction:
                    next_flip_ts = nxt["timestamp"]
                    break

            if opt_symbol not in option_cache:
                self._append_skip(cols, "no_option_data")
                continue

            opt_df = option_cache[opt_symbol]
            idx_pos = opt_df.index.searchsorted(confirm_ts, side="right")
            if idx_pos >= len(opt_df):
                self._append_skip(cols, "no_entry_bar_after_confirm")
                continue

            entry_bar   = opt_df.iloc[int(idx_pos)]
            entry_price = float(entry_bar["open"])
            entry_time  = opt_df.index[int(idx_pos)]

            sl_price = entry_price * (1.0 - sl_pct / 100.0)
            tp_price = entry_price * (1.0 + tp_pct / 100.0)

            exit_idx = exit_price = exit_time = exit_reason = None

            for k in range(int(idx_pos) + 1, len(opt_df)):
                bar = opt_df.iloc[k]
                bar_time = opt_df.index[k]

                if bar_time >= eod_boundary:
                    exit_idx = k; exit_price = float(bar["close"])
                    exit_time = bar_time; exit_reason = "eod_exit"; break

                hit_sl = float(bar["low"])  <= sl_price
                hit_tp = float(bar["high"]) >= tp_price

                if hit_sl and hit_tp:
                    exit_idx = k; exit_price = sl_price
                    exit_time = bar_time; exit_reason = "sl_and_tp_same_bar"; break
                elif hit_sl:
                    exit_idx = k; exit_price = sl_price
                    exit_time = bar_time; exit_reason = "stop_loss"; break
                elif hit_tp:
                    exit_idx = k; exit_price = tp_price
                    exit_time = bar_time; exit_reason = "target"; break

                if next_flip_ts is not None and bar_time >= next_flip_ts:
                    exit_idx = k; exit_price = float(bar["close"])
                    exit_time = bar_time; exit_reason = "flip_exit"; break

            if exit_idx is None:
                self._append_skip(cols, "no_exit_found", entry_price=entry_price, entry_time=entry_time)
                continue

            pnl_pct    = (exit_price - entry_price) / entry_price * 100.0
            pnl_rupees = entry_price * self.config.NIFTY_LOT_SIZE * (pnl_pct / 100.0)

            cols["entry_price_option"].append(entry_price)
            cols["entry_time"].append(entry_time)
            cols["exit_time"].append(exit_time)
            cols["exit_price_option"].append(exit_price)
            cols["pnl_pct"].append(pnl_pct)
            cols["pnl_rupees_1lot"].append(pnl_rupees)
            cols["result"].append("profit" if pnl_pct > 0 else "loss")
            cols["exit_reason"].append(exit_reason)

        for col, vals in cols.items():
            signals_sorted[col] = vals

        signals_sorted["cum_pnl_rupees_1lot"] = signals_sorted["pnl_rupees_1lot"].cumsum()

        traded = signals_sorted[~signals_sorted["result"].isin(["skipped", "open", "unknown"])]
        total_trades = len(traded)
        total_profit = traded.loc[traded["result"] == "profit", "pnl_rupees_1lot"].sum()
        total_loss   = traded.loc[traded["result"] == "loss",   "pnl_rupees_1lot"].sum()
        net_pnl      = traded["pnl_rupees_1lot"].sum()
        win_rate     = (len(traded[traded["result"] == "profit"]) / total_trades * 100.0
                        if total_trades > 0 else 0.0)
        eod_exits    = len(traded[traded["exit_reason"] == "eod_exit"])

        csv_path = "SwiftTrend_NoHTF_NIFTY_ITM_Backtest.csv"
        signals_sorted.to_csv(csv_path, index=False, encoding="utf-8")

        with open(csv_path, "a", encoding="utf-8") as f:
            for line in [
                "", "METRICS,VALUE",
                f"Total Signals,{len(signals_sorted)}",
                f"Total Trades Executed,{total_trades}",
                f"EOD Exits,{eod_exits}",
                f"Total Profit (Rs),{total_profit:.2f}",
                f"Total Loss (Rs),{total_loss:.2f}",
                f"Net PnL (Rs),{net_pnl:.2f}",
                f"Win Rate (%),{win_rate:.2f}",
                f"ST Period,{self.config.ST_PERIOD}",
                f"ST Band Mult,{self.config.ST_BAND_MULT}",
                f"ST Tol Mult,{self.config.ST_TOL_MULT}",
                f"SL %,{self.config.STOP_LOSS_PCT}",
                f"TP %,{self.config.TAKE_PROFIT_PCT}",
            ]:
                f.write(line + "\n")

        print(f"\n{'='*55}")
        print(f"  Contract Type       : ITM")
        print(f"  Total Signals       : {len(signals_sorted)}")
        print(f"  Trades Executed     : {total_trades}")
        print(f"  EOD Exits           : {eod_exits}")
        print(f"  Win Rate            : {win_rate:.1f}%")
        print(f"  Net PnL             : Rs {net_pnl:,.2f}")
        print(f"  Total Profit        : Rs {total_profit:,.2f}")
        print(f"  Total Loss          : Rs {total_loss:,.2f}")
        print(f"{'='*55}")
        print(f"Results saved → {csv_path}")
        return signals_sorted

    def _append_skip(self, cols: dict, reason: str,
                      entry_price: float = np.nan, entry_time=pd.NaT):
        cols["entry_price_option"].append(entry_price)
        cols["entry_time"].append(entry_time)
        cols["exit_time"].append(pd.NaT)
        cols["exit_price_option"].append(np.nan)
        cols["pnl_pct"].append(0.0)
        cols["pnl_rupees_1lot"].append(0.0)
        cols["result"].append("skipped")
        cols["exit_reason"].append(reason)


# --------------------------- MAIN ---------------------------

if __name__ == "__main__":
    bot     = SwiftTrendBacktestBot()
    signals = bot.run_backtest()