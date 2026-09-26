import pandas as pd
import numpy as np
from datetime import datetime, timedelta, time
import requests
import logging
import os
import pytz
import calendar
from typing import Dict, Optional, List
import warnings
import urllib3

# ─── SSL FIX ──────────────────────────────────────────────────────────────────
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
# ──────────────────────────────────────────────────────────────────────────────

IST = pytz.timezone("Asia/Kolkata")
warnings.filterwarnings("ignore", category=FutureWarning)

# --------------------------- EXPIRY HELPERS ---------------------------

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
    """
    Fyers option symbol expiry code:
    - Last Tuesday of month (monthly expiry) → YYMON   e.g. '26JUN'
    - Any other Tuesday    (weekly expiry)   → YY+M+DD e.g. '26616'
      where M is 1-9 for Jan-Sep, O/N/D for Oct/Nov/Dec (Fyers convention).
    """
    yy          = dt.year % 100
    monthly_exp = get_nifty_monthly_expiry(dt.year, dt.month)
    if dt.date() == monthly_exp.date():
        return f"{yy:02d}{dt.strftime('%b').upper()}"
    else:
        month_code = str(dt.month) if dt.month <= 9 else {10: "O", 11: "N", 12: "D"}[dt.month]
        return f"{yy:02d}{month_code}{dt.day:02d}"

# --------------------------- CONFIG ---------------------------

class TradingConfig:
    UNDERLYING_SYMBOL = "NSE:NIFTY50-INDEX"
    TIMEFRAME         = "10S"
    HIST_DAYS         = 30

    STRIKE_STEP    = 100
    NIFTY_LOT_SIZE = 65

    EXPIRY_TYPE = "MONTHLY"

    VAM_BUY_SCORE  =  0.10
    VAM_SELL_SCORE = -0.10

    LEN1 = 8;  LEN2 = 10;  LEN3 = 14;  LEN4 = 20;  LEN5 = 28
    MULT1 = 1.05; MULT2 = 1.10; MULT3 = 1.15; MULT4 = 1.20; MULT5 = 1.25
    BAND1 = 1.15; BAND2 = 1.20; BAND3 = 1.25; BAND4 = 1.30; BAND5 = 1.35
    SMOOTH1 = 2;  SMOOTH2 = 2;  SMOOTH3 = 3;  SMOOTH4 = 3;  SMOOTH5 = 4

    ATR_LEN       = 14
    ATR_LAYERMULT = 0.90
    TRAIL_SMOOTH  = 4

    HTF_RULE    = "15min"
    HTF_EMA_LEN = 6

    SESSION_START = time(9, 20)
    SESSION_END   = time(15, 30)
    TRADE_START   = time(9, 30)
    TRADE_END     = time(15, 0)

    EOD_EXIT_TIME = time(15, 20)

    PIVOT_BARS          = 10
    PIVOT_PROXIMITY_PCT = 0.5

    def __init__(self):
        now      = datetime.now(IST)
        exp_type = self.EXPIRY_TYPE.upper()
        if exp_type == "WEEKLY":
            exp_dt = get_next_tuesday(now);                kind = "WEEKLY"
        else:
            exp_dt = get_current_or_next_monthly_expiry(now); kind = "MONTHLY"
        self.EXPIRY_CODE = format_expiry_code(exp_dt)
        logging.getLogger(__name__).info(
            f"Using NIFTY {kind} expiry {exp_dt.date()} → code {self.EXPIRY_CODE}"
        )

# --------------------------- LOGGING ---------------------------

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

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
            raise ValueError(
                f"History API non-JSON (status={r.status_code}): {r.text[:300]}"
            ) from e

# --------------------------- VAM CORE ---------------------------

def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()

def tr(df: pd.DataFrame) -> pd.Series:
    hl = df["high"] - df["low"]
    hc = (df["high"] - df["close"].shift()).abs()
    lc = (df["low"]  - df["close"].shift()).abs()
    return pd.Series(np.maximum(hl, np.maximum(hc, lc)), index=df.index)

def compute_r1_side(df, length, r1_mult, band_mult, smooth):
    src    = df["close"]
    tr_val = tr(df)
    r1     = ema(ema(tr_val, length) * r1_mult, smooth)

    center = pd.Series(index=df.index, dtype=float)
    center.iloc[0] = src.iloc[0]
    for i in range(1, len(df)):
        prev  = center.iloc[i - 1]
        diff  = src.iloc[i] - prev
        if   diff >  r1.iloc[i]: center.iloc[i] = prev + (diff - r1.iloc[i])
        elif diff < -r1.iloc[i]: center.iloc[i] = prev + (diff + r1.iloc[i])
        else:                    center.iloc[i] = prev

    upper = ema(center + r1 * band_mult, smooth)
    lower = ema(center - r1 * band_mult, smooth)

    break_long  = (src > upper.shift(1)) & (src.shift(1) <= upper.shift(2))
    break_short = (src < lower.shift(1)) & (src.shift(1) >= lower.shift(2))

    side = np.ones(len(df), dtype=int)
    for i in range(1, len(df)):
        if   break_long.iloc[i]:  side[i] =  1
        elif break_short.iloc[i]: side[i] = -1
        else:                     side[i] =  side[i - 1]

    return side, upper, lower

def compute_vam_ensemble(df: pd.DataFrame, cfg: TradingConfig) -> pd.DataFrame:
    df_local = df.copy()
    params   = dict(
        lens  =[cfg.LEN1,   cfg.LEN2,   cfg.LEN3,   cfg.LEN4,   cfg.LEN5],
        mults =[cfg.MULT1,  cfg.MULT2,  cfg.MULT3,  cfg.MULT4,  cfg.MULT5],
        bands =[cfg.BAND1,  cfg.BAND2,  cfg.BAND3,  cfg.BAND4,  cfg.BAND5],
        smooth=[cfg.SMOOTH1,cfg.SMOOTH2,cfg.SMOOTH3,cfg.SMOOTH4,cfg.SMOOTH5],
    )
    sides, uppers, lowers = [], {}, {}
    for i in range(5):
        s, u, l = compute_r1_side(df_local, params["lens"][i], params["mults"][i],
                                   params["bands"][i], params["smooth"][i])
        sides.append(s); uppers[i+1] = u; lowers[i+1] = l

    score = np.mean(sides, axis=0) / 5.0
    df_local["vam_score"]    = score
    df_local["vam_is_bull"]  = score >  cfg.VAM_BUY_SCORE
    df_local["vam_is_bear"]  = score <  cfg.VAM_SELL_SCORE
    df_local["vam_is_flat"]  = ~df_local["vam_is_bull"] & ~df_local["vam_is_bear"]
    df_local["vam_buy_sig"]  = (df_local["vam_score"].shift(1) <= cfg.VAM_BUY_SCORE)  & (score >  cfg.VAM_BUY_SCORE)
    df_local["vam_sell_sig"] = (df_local["vam_score"].shift(1) >= cfg.VAM_SELL_SCORE) & (score <  cfg.VAM_SELL_SCORE)

    tr_val       = tr(df_local)
    atr_layer    = ema(tr_val, cfg.ATR_LEN)
    tl1          = ema(lowers[3], cfg.TRAIL_SMOOTH)
    tu1          = ema(uppers[3], cfg.TRAIL_SMOOTH)
    df_local["vam_trail_lower1"] = tl1
    df_local["vam_trail_upper1"] = tu1
    df_local["vam_trail_lower2"] = ema(tl1 - atr_layer * cfg.ATR_LAYERMULT, cfg.TRAIL_SMOOTH)
    df_local["vam_trail_upper2"] = ema(tu1 + atr_layer * cfg.ATR_LAYERMULT, cfg.TRAIL_SMOOTH)
    return df_local

# --------------------------- PRICE ACTION PIVOT HELPER ---------------------------

def compute_pivot_zone(
    df_idx_reset: pd.DataFrame,
    signal_bar_pos: int,
    pivot_bars: int,
    direction: str,
) -> Optional[float]:
    scan_end = signal_bar_pos - pivot_bars
    if scan_end <= pivot_bars:
        return None

    if direction == "BUY":
        for i in range(scan_end, pivot_bars - 1, -1):
            low_i = float(df_idx_reset.iloc[i]["low"])
            left  = df_idx_reset.iloc[i - pivot_bars : i]["low"]
            right = df_idx_reset.iloc[i + 1 : i + pivot_bars + 1]["low"]
            if len(left) < pivot_bars or len(right) < pivot_bars:
                continue
            if low_i < float(left.min()) and low_i < float(right.min()):
                return low_i
    else:
        for i in range(scan_end, pivot_bars - 1, -1):
            high_i = float(df_idx_reset.iloc[i]["high"])
            left   = df_idx_reset.iloc[i - pivot_bars : i]["high"]
            right  = df_idx_reset.iloc[i + 1 : i + pivot_bars + 1]["high"]
            if len(left) < pivot_bars or len(right) < pivot_bars:
                continue
            if high_i > float(left.max()) and high_i > float(right.max()):
                return high_i

    return None


# --------------------------- VAM BACKTEST BOT ---------------------------

class VAMBacktestBot:
    def __init__(self):
        self.config        = TradingConfig()
        self.fyers         = self._init_fyers()
        self.position_size = 0
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
            raise ConnectionError(
                f"Fyers auth failed — re-run GenerateToken.py.\nResponse: {profile}"
            )
        logger.info(f"Fyers connected — {profile['data']['name']} ({profile['data']['fy_id']}) ✅")
        return client

    def _check_ssl(self):
        try:
            r = requests.get("https://api-t1.fyers.in", timeout=5, verify=False)
            logger.info(f"Connectivity check passed (status={r.status_code})")
        except Exception as e:
            logger.warning(f"Connectivity check inconclusive: {e}")

    def _fetch_history(self, symbol: str, days: int) -> pd.DataFrame:
        end_dt   = datetime.now(IST)
        start_dt = end_dt - timedelta(days=days)
        data = dict(symbol=symbol, resolution=self.config.TIMEFRAME,
                    date_format="1",
                    range_from=start_dt.strftime("%Y-%m-%d"),
                    range_to=end_dt.strftime("%Y-%m-%d"),
                    cont_flag="0")
        logger.info(f"Fetching history → {data}")
        resp = self.fyers.history(data=data)
        if isinstance(resp, dict):
            if resp.get("s") != "ok":
                raise ValueError(f"API error for {symbol}: {resp}")
            candles = resp.get("candles", [])
        else:
            candles = []
        if not candles:
            raise ValueError(f"No historical data for {symbol}")
        df = pd.DataFrame(candles,
                          columns=["timestamp","open","high","low","close","volume"])
        df["timestamp"] = (pd.to_datetime(df["timestamp"], unit="s", utc=True)
                           .dt.tz_convert(IST))
        return df.sort_values("timestamp").reset_index(drop=True)

    def get_index_data(self, days: int = None) -> pd.DataFrame:
        days = days or self.config.HIST_DAYS
        df   = self._fetch_history(self.config.UNDERLYING_SYMBOL, days)
        df.set_index("timestamp", inplace=True)
        mask = df.index.indexer_between_time(
            self.config.SESSION_START, self.config.SESSION_END)
        df = df.iloc[mask]
        logger.info(f"Loaded NIFTY50 index data: {len(df)} rows")
        return df

    def _fetch_option_history(self, symbol: str, days: int) -> pd.DataFrame:
        df = self._fetch_history(symbol, days)
        df.set_index("timestamp", inplace=True)
        return df

    def _build_htf_trend(self, df_idx: pd.DataFrame) -> pd.DataFrame:
        c      = self.config
        df_htf = df_idx.resample(c.HTF_RULE).agg(
            {"open":"first","high":"max","low":"min","close":"last","volume":"sum"}
        ).dropna()
        df_htf["htf_ema"]   = df_htf["close"].ewm(span=c.HTF_EMA_LEN, adjust=False).mean()
        df_htf["htf_trend"] = np.where(
            df_htf["close"] > df_htf["htf_ema"], "up",
            np.where(df_htf["close"] < df_htf["htf_ema"], "down", "flat")
        )
        return df_htf.reset_index()

    def _pick_atm_option_symbol(self, underlying_price: float, direction: str) -> str:
        """Picks 1-strike ITM option (matches the live bots)."""
        c      = self.config
        spot   = float(underlying_price)
        step   = c.STRIKE_STEP
        atm    = int(round(spot / step) * step)
        if direction == "BUY":
            strike   = atm - step
            opt_type = "CE"
        else:
            strike   = atm + step
            opt_type = "PE"
        strike = int(max(step, strike))
        sym = f"NSE:NIFTY{c.EXPIRY_CODE}{strike:05d}{opt_type}"
        logger.info(f"ITM option symbol: {sym} (spot={spot:.2f}, dir={direction})")
        return sym

    # ── signal generation ──

    def generate_vam_signals(self, df_index: pd.DataFrame) -> pd.DataFrame:
        df_idx  = df_index.copy().reset_index()
        df_vam  = compute_vam_ensemble(df_idx, self.config)
        df_htf  = self._build_htf_trend(df_index.copy())

        df_merged = pd.merge_asof(
            df_vam.sort_values("timestamp"),
            df_htf[["timestamp","htf_trend"]].sort_values("timestamp"),
            on="timestamp", direction="backward"
        )

        warmup  = max(self.config.LEN1, self.config.LEN2, self.config.LEN3,
                      self.config.LEN4, self.config.LEN5, 50)
        signals = []

        for i in range(warmup, len(df_merged)):
            row      = df_merged.iloc[i]
            ts_time  = row["timestamp"].time()
            in_win   = self.config.TRADE_START <= ts_time <= self.config.TRADE_END
            htf      = row.get("htf_trend", "flat")

            ok_long  = (self.position_size <= 0) and (self.last_trade != "long")
            ok_short = (self.position_size >= 0) and (self.last_trade != "short")

            if not in_win:
                continue

            if bool(row["vam_buy_sig"]) and ok_long and htf == "up":
                pivot_lvl = compute_pivot_zone(
                    df_merged, i, self.config.PIVOT_BARS, "BUY"
                )
                sig = {
                    "timestamp":      row["timestamp"],
                    "signal_bar_pos": i,
                    "index_symbol":   self.config.UNDERLYING_SYMBOL,
                    "index_price":    float(row["close"]),
                    "direction":      "BUY",
                    "option_symbol":  self._pick_atm_option_symbol(row["close"], "BUY"),
                    "strategy_type":  "vam_htf_pivot",
                    "htf_trend":      htf,
                    "pivot_level":    pivot_lvl,
                }
                signals.append(sig)
                self.position_size = 1
                self.last_trade    = "long"

            elif bool(row["vam_sell_sig"]) and ok_short and htf == "down":
                pivot_lvl = compute_pivot_zone(
                    df_merged, i, self.config.PIVOT_BARS, "SELL"
                )
                sig = {
                    "timestamp":      row["timestamp"],
                    "signal_bar_pos": i,
                    "index_symbol":   self.config.UNDERLYING_SYMBOL,
                    "index_price":    float(row["close"]),
                    "direction":      "SELL",
                    "option_symbol":  self._pick_atm_option_symbol(row["close"], "SELL"),
                    "strategy_type":  "vam_htf_pivot",
                    "htf_trend":      htf,
                    "pivot_level":    pivot_lvl,
                }
                signals.append(sig)
                self.position_size = -1
                self.last_trade    = "short"

        signals_df = pd.DataFrame(signals) if signals else pd.DataFrame()
        logger.info(f"Generated {len(signals_df)} VAM signals (HTF-filtered)")
        return signals_df

    # ── backtest engine ──

    def run_backtest(self) -> pd.DataFrame:
        try:
            print("Checking connectivity...")
            self._check_ssl()

            print("Starting VAM + HTF + Price Action Pivot Backtest (ATM contracts)...")
            df_index      = self.get_index_data()
            df_idx_reset  = df_index.reset_index()
            signals       = self.generate_vam_signals(df_index)

            if signals.empty:
                print("No VAM signals generated")
                return signals

            option_cache: Dict[str, pd.DataFrame] = {}
            for sym in signals["option_symbol"].unique():
                try:
                    option_cache[sym] = self._fetch_option_history(
                        sym, self.config.HIST_DAYS).sort_index()
                except ValueError as e:
                    logger.warning(f"Skipping {sym}: {e}")

            cols = dict(
                entry_price_option=[], exit_time=[], exit_price_option=[],
                pnl_pct=[], pnl_rupees_1lot=[], result=[], exit_reason=[],
                pivot_level=[], entry_method=[], entry_time=[],
            )

            signals_sorted = signals.sort_values("timestamp").reset_index(drop=True)
            stop_loss_pct  = -20.0
            target_pct     =  20
            EOD_EXIT       = self.config.EOD_EXIT_TIME

            for i, s in signals_sorted.iterrows():
                signal_ts    = s["timestamp"]
                opt_symbol   = s["option_symbol"]
                direction    = s["direction"]
                pivot_lvl    = s.get("pivot_level")
                signal_price = float(s["index_price"])

                if signal_ts.time() >= EOD_EXIT:
                    self._append_skip(cols, "signal_after_eod", pivot_lvl)
                    continue

                next_flip_ts = None
                for j in range(i + 1, len(signals_sorted)):
                    nxt = signals_sorted.iloc[j]
                    if nxt["direction"] != direction:
                        next_flip_ts = nxt["timestamp"]
                        break

                eod_boundary = IST.localize(
                    datetime.combine(signal_ts.date(), EOD_EXIT)
                )

                sig_idx_pos = df_idx_reset.index[
                    df_idx_reset["timestamp"] == signal_ts
                ]
                if len(sig_idx_pos) == 0:
                    self._append_skip(cols, "signal_ts_not_in_index", pivot_lvl)
                    continue
                sig_idx_pos = int(sig_idx_pos[0])

                if pivot_lvl is None:
                    self._append_skip(cols, "no_pivot_found", pivot_lvl)
                    continue

                prox_band      = pivot_lvl * (self.config.PIVOT_PROXIMITY_PCT / 100.0)
                entry_time_idx = None
                entry_method   = None

                if direction == "BUY":
                    already_near = (signal_price - pivot_lvl) <= prox_band
                else:
                    already_near = (pivot_lvl - signal_price) <= prox_band

                if already_near:
                    entry_time_idx = signal_ts
                    entry_method   = "already_near_pivot"
                else:
                    for k in range(sig_idx_pos + 1, len(df_idx_reset)):
                        idx_bar      = df_idx_reset.iloc[k]
                        idx_bar_time = idx_bar["timestamp"]

                        if idx_bar_time >= eod_boundary:
                            break
                        if next_flip_ts is not None and idx_bar_time >= next_flip_ts:
                            break

                        check_price = (float(idx_bar["low"]) if direction == "BUY"
                                       else float(idx_bar["high"]))
                        if abs(check_price - pivot_lvl) <= prox_band:
                            entry_time_idx = idx_bar_time
                            entry_method   = "pulled_to_pivot"
                            break

                if entry_time_idx is None:
                    self._append_skip(cols, "pivot_never_reached", pivot_lvl, triggered=False)
                    continue

                if opt_symbol not in option_cache:
                    self._append_skip(cols, "no_option_data", pivot_lvl, triggered=True)
                    continue

                opt_df  = option_cache[opt_symbol]
                idx_pos = opt_df.index.searchsorted(entry_time_idx, side="right")

                if idx_pos >= len(opt_df):
                    self._append_skip(cols, "no_entry_bar_after_pivot", pivot_lvl, triggered=True)
                    continue

                entry_bar   = opt_df.iloc[int(idx_pos)]
                entry_price = float(entry_bar["open"]) * 1.005
                entry_time  = opt_df.index[int(idx_pos)]

                sl_price = entry_price * (1.0 + stop_loss_pct / 100.0)
                tp_price = entry_price * (1.0 + target_pct    / 100.0)

                exit_idx = exit_price = exit_time = exit_reason = None

                for k in range(int(idx_pos) + 1, len(opt_df)):
                    bar      = opt_df.iloc[k]
                    bar_time = opt_df.index[k]

                    if bar_time >= eod_boundary:
                        exit_idx    = k
                        exit_price  = float(bar["close"])
                        exit_time   = bar_time
                        exit_reason = "eod_exit"
                        break

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
                    self._append_skip(cols, "no_exit_found", pivot_lvl, triggered=True,
                                      entry_price=entry_price, entry_time=entry_time)
                    continue

                pnl_pct    = (exit_price - entry_price) / entry_price * 100.0
                pnl_rupees = entry_price * self.config.NIFTY_LOT_SIZE * (pnl_pct / 100.0)

                cols["entry_price_option"].append(entry_price)
                cols["exit_time"].append(exit_time)
                cols["exit_price_option"].append(exit_price)
                cols["pnl_pct"].append(pnl_pct)
                cols["pnl_rupees_1lot"].append(pnl_rupees)
                cols["result"].append("profit" if pnl_pct > 0 else "loss")
                cols["exit_reason"].append(exit_reason)
                cols["pivot_level"].append(pivot_lvl)
                cols["entry_method"].append(entry_method)
                cols["entry_time"].append(entry_time)

            for col, vals in cols.items():
                signals_sorted[col] = vals

            signals_sorted["cum_pnl_rupees_1lot"] = (
                signals_sorted["pnl_rupees_1lot"].cumsum()
            )

            traded = signals_sorted[
                ~signals_sorted["result"].isin(["skipped","open","unknown"])
            ]
            total_trades       = len(traded)
            total_profit       = traded.loc[traded["result"]=="profit","pnl_rupees_1lot"].sum()
            total_loss         = traded.loc[traded["result"]=="loss",  "pnl_rupees_1lot"].sum()
            net_pnl            = traded["pnl_rupees_1lot"].sum()
            win_rate           = (len(traded[traded["result"]=="profit"]) / total_trades * 100.0
                                  if total_trades > 0 else 0.0)
            eod_exits          = len(traded[traded["exit_reason"]=="eod_exit"])
            pivot_skipped      = len(signals_sorted[signals_sorted["exit_reason"]=="pivot_never_reached"])
            already_near_pivot = len(signals_sorted[signals_sorted["entry_method"]=="already_near_pivot"])
            pulled_to_pivot_ct = len(signals_sorted[signals_sorted["entry_method"]=="pulled_to_pivot"])

            csv_path = "VAM_NIFTY_ATM_Backtest.csv"
            signals_sorted.to_csv(csv_path, index=False, encoding="utf-8")

            with open(csv_path, "a", encoding="utf-8") as f:
                for line in [
                    "", "METRICS,VALUE",
                    f"Total Signals,{len(signals_sorted)}",
                    f"Pivot Never Reached (skipped),{pivot_skipped}",
                    f"Already Near Pivot,{already_near_pivot}",
                    f"Pulled to Pivot,{pulled_to_pivot_ct}",
                    f"Total Trades Executed,{total_trades}",
                    f"EOD Exits (3:20 PM),{eod_exits}",
                    f"Total Profit (Rs),{total_profit:.2f}",
                    f"Total Loss (Rs),{total_loss:.2f}",
                    f"Net PnL (Rs),{net_pnl:.2f}",
                    f"Win Rate (%),{win_rate:.2f}",
                ]:
                    f.write(line + "\n")

            print(f"\n{'='*55}")
            print(f"  Contract Type         : ATM")
            print(f"  Total Signals         : {len(signals_sorted)}")
            print(f"  Pivot Skipped         : {pivot_skipped}  (price never reached pivot)")
            print(f"  Already Near Pivot    : {already_near_pivot}  (entered at signal bar)")
            print(f"  Pulled to Pivot       : {pulled_to_pivot_ct}  (waited for pullback)")
            print(f"  Trades Executed       : {total_trades}")
            print(f"  EOD Exits (3:20 PM)   : {eod_exits}")
            print(f"  Win Rate              : {win_rate:.1f}%")
            print(f"  Net PnL               : Rs {net_pnl:,.2f}")
            print(f"  Total Profit          : Rs {total_profit:,.2f}")
            print(f"  Total Loss            : Rs {total_loss:,.2f}")
            print(f"{'='*55}")
            print(f"Results saved → {csv_path}")
            return signals_sorted

        except Exception as e:
            logger.error(f"Backtest failed: {e}")
            raise

    def _append_skip(
        self, cols: dict, reason: str, pivot_lvl,
        triggered: bool = False,
        entry_price: float = np.nan,
        entry_time=pd.NaT,
    ):
        cols["entry_price_option"].append(entry_price)
        cols["exit_time"].append(pd.NaT)
        cols["exit_price_option"].append(np.nan)
        cols["pnl_pct"].append(0.0)
        cols["pnl_rupees_1lot"].append(0.0)
        cols["result"].append("skipped")
        cols["exit_reason"].append(reason)
        cols["pivot_level"].append(pivot_lvl)
        cols["entry_method"].append("skipped")
        cols["entry_time"].append(entry_time)


# --------------------------- MAIN ---------------------------

if __name__ == "__main__":
    bot     = VAMBacktestBot()
    signals = bot.run_backtest()
