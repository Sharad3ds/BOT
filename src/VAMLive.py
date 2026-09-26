import pandas as pd
import numpy as np
from datetime import datetime, timedelta, time as dtime
from fyers_apiv3 import fyersModel
from fyers_apiv3.FyersWebsocket import data_ws
import logging
import os
from typing import Dict, Optional
import warnings
import pytz
import calendar
import time
import threading
import json

# --------------------------- TIMEZONE ---------------------------

IST = pytz.timezone("Asia/Kolkata")

# --------------------------- STATE FILE ---------------------------

STATE_FILE = "vam_position_state.json"


# --------------------------- LOGGING (FORCE IST) ---------------------------

def ist_time(*args):
    return datetime.now(IST).timetuple()

logging.Formatter.converter = ist_time

try:
    from logger_config import get_logger
    logger = get_logger()
except ImportError:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - SMBot_V2 - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logger = logging.getLogger("SMBot_V2")

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
            y += 1
            m = 1
        else:
            m += 1
        exp_dt = get_nifty_monthly_expiry(y, m)
    return exp_dt


def format_expiry_code(dt: datetime) -> str:
    yy = dt.year % 100
    mon = dt.strftime("%b").upper()
    return f"{yy:02d}{mon}"


# --------------------------- CONFIG ---------------------------
# ✅ All values matched to backtest version

class TradingConfig:
    UNDERLYING_SYMBOL = "NSE:NIFTY50-INDEX"
    TIMEFRAME         = "10S"        # 10-second bars — matches polling bot
    HIST_DAYS         = 90          # ✅ matched backtest

    STRIKE_STEP    = 100
    NIFTY_LOT_SIZE = 65            # ✅ matched backtest

    EXPIRY_TYPE = "WEEKLY"          # matches polling bot

    VAM_BUY_SCORE  =  0.10
    VAM_SELL_SCORE = -0.10

    LEN1 = 8;  LEN2 = 10;  LEN3 = 14;  LEN4 = 20;  LEN5 = 28
    MULT1 = 1.05; MULT2 = 1.10; MULT3 = 1.15; MULT4 = 1.20; MULT5 = 1.25
    BAND1 = 1.15; BAND2 = 1.20; BAND3 = 1.25; BAND4 = 1.30; BAND5 = 1.35
    SMOOTH1 = 2;  SMOOTH2 = 2;  SMOOTH3 = 3;  SMOOTH4 = 3;  SMOOTH5 = 4

    ATR_LEN       = 14
    ATR_LAYERMULT = 0.90
    TRAIL_SMOOTH  = 4

    HTF_RULE    = "15min"             # matches polling bot / backtest
    HTF_EMA_LEN = 6                  # matches polling bot / backtest

    SESSION_START = dtime(9, 20)
    SESSION_END   = dtime(15, 30)
    TRADE_START   = dtime(9, 30)
    TRADE_END     = dtime(15, 0)

    # ✅ Matched backtest
    EOD_EXIT_TIME   = dtime(15, 20)
    STOP_LOSS_PCT   = 20.0          # matches backtest
    TAKE_PROFIT_PCT = 25.0          # matches backtest (polling bot had 15 — wrong)

    # Staleness guard — skip pending signal if bar closed more than N seconds ago
    SIGNAL_STALE_SECS = 30

    # ✅ Pivot zone — matched backtest
    PIVOT_BARS          = 10        # bars each side to confirm pivot
    PIVOT_PROXIMITY_PCT = 0.5       # % within pivot to trigger entry

    # How often (seconds) to refresh today's bars from the API
    API_REFRESH_INTERVAL_SECS = 300

    def __init__(self):
        now = datetime.now(IST)
        exp_type = self.EXPIRY_TYPE.upper()
        if exp_type == "WEEKLY":
            exp_dt = get_next_tuesday(now)
            kind = "WEEKLY"
        else:
            exp_dt = get_current_or_next_monthly_expiry(now)
            kind = "MONTHLY"
        self.EXPIRY_CODE = format_expiry_code(exp_dt)
        logging.getLogger(__name__).info(
            f"Using NIFTY {kind} expiry {exp_dt.date()} with code {self.EXPIRY_CODE}"
        )


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
        prev = center.iloc[i - 1]
        diff = src.iloc[i] - prev
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
    params = {
        "lens":   [cfg.LEN1, cfg.LEN2, cfg.LEN3, cfg.LEN4, cfg.LEN5],
        "mults":  [cfg.MULT1, cfg.MULT2, cfg.MULT3, cfg.MULT4, cfg.MULT5],
        "bands":  [cfg.BAND1, cfg.BAND2, cfg.BAND3, cfg.BAND4, cfg.BAND5],
        "smooth": [cfg.SMOOTH1, cfg.SMOOTH2, cfg.SMOOTH3, cfg.SMOOTH4, cfg.SMOOTH5],
    }
    sides, uppers, lowers = [], {}, {}
    for i in range(5):
        s, u, l = compute_r1_side(
            df_local, params["lens"][i], params["mults"][i],
            params["bands"][i], params["smooth"][i],
        )
        sides.append(s); uppers[i + 1] = u; lowers[i + 1] = l

    score = np.mean(sides, axis=0) / 5.0
    df_local["vam_score"]    = score
    df_local["vam_is_bull"]  = score >  cfg.VAM_BUY_SCORE
    df_local["vam_is_bear"]  = score <  cfg.VAM_SELL_SCORE
    df_local["vam_is_flat"]  = ~df_local["vam_is_bull"] & ~df_local["vam_is_bear"]
    df_local["vam_buy_sig"]  = (
        (df_local["vam_score"].shift(1) <= cfg.VAM_BUY_SCORE)
        & (score > cfg.VAM_BUY_SCORE)
    )
    df_local["vam_sell_sig"] = (
        (df_local["vam_score"].shift(1) >= cfg.VAM_SELL_SCORE)
        & (score < cfg.VAM_SELL_SCORE)
    )
    tr_val       = tr(df_local)
    atr_layer    = ema(tr_val, cfg.ATR_LEN)
    tl1          = ema(lowers[3], cfg.TRAIL_SMOOTH)
    tu1          = ema(uppers[3], cfg.TRAIL_SMOOTH)
    df_local["vam_trail_lower1"] = tl1
    df_local["vam_trail_upper1"] = tu1
    df_local["vam_trail_lower2"] = ema(tl1 - atr_layer * cfg.ATR_LAYERMULT, cfg.TRAIL_SMOOTH)
    df_local["vam_trail_upper2"] = ema(tu1 + atr_layer * cfg.ATR_LAYERMULT, cfg.TRAIL_SMOOTH)
    return df_local


# --------------------------- PIVOT ZONE HELPER ---------------------------
# ✅ Identical to backtest version

def compute_pivot_zone(
    df_idx_reset: pd.DataFrame,
    signal_bar_pos: int,
    pivot_bars: int,
    direction: str,
) -> Optional[float]:
    """
    Finds the most recent confirmed Price Action Pivot before the signal bar.

    Pivot Low  (BUY) : bar[i].low  < all lows  in [i-pivot_bars:i] AND [i+1:i+pivot_bars+1]
    Pivot High (SELL): bar[i].high > all highs in [i-pivot_bars:i] AND [i+1:i+pivot_bars+1]

    Scans backwards from (signal_bar_pos - pivot_bars) so pivot is fully confirmed.
    Returns pivot price or None.
    """
    scan_end = signal_bar_pos - pivot_bars
    if scan_end <= pivot_bars:
        return None

    if direction == "BUY":
        for i in range(scan_end, pivot_bars - 1, -1):
            low_i  = float(df_idx_reset.iloc[i]["low"])
            left   = df_idx_reset.iloc[i - pivot_bars : i]["low"]
            right  = df_idx_reset.iloc[i + 1 : i + pivot_bars + 1]["low"]
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


# --------------------------- VAM LIVE BOT (SIGNAL ENGINE) ---------------------------

class VAMLiveBot:
    def __init__(self):
        self.config = TradingConfig()
        self.fyers  = self._init_fyers()
        self.position_size: int          = 0
        self.last_trade:    Optional[str] = None

        self.app_id:       Optional[str] = None
        self.access_token: Optional[str] = None
        if os.path.exists("fyers_appid.txt"):
            with open("fyers_appid.txt") as f:
                self.app_id = f.read().strip()
        if os.path.exists("fyers_token.txt"):
            with open("fyers_token.txt") as f:
                self.access_token = f.read().strip()

        logger.info("VAMLiveBot initialized.")

    def _init_fyers(self):
        for f in ["fyers_appid.txt", "fyers_token.txt"]:
            if not os.path.exists(f):
                raise FileNotFoundError(f"Missing {f}")
        app_id = open("fyers_appid.txt").read().strip()
        token  = open("fyers_token.txt").read().strip()
        if not app_id or not token:
            raise ValueError("Empty app_id or token")
        fy = fyersModel.FyersModel(client_id=app_id, token=token, is_async=False, log_path="logs")
        logger.info("Fyers V3 API initialized for VAMLiveBot")
        return fy

    def _fetch_history(self, symbol: str, days: int) -> pd.DataFrame:
        end_dt_ist   = datetime.now(IST)
        start_dt_ist = end_dt_ist - timedelta(days=days)
        data = {
            "symbol":      symbol,
            "resolution":  self.config.TIMEFRAME,
            "date_format": "1",
            "range_from":  start_dt_ist.strftime("%Y-%m-%d"),
            "range_to":    end_dt_ist.strftime("%Y-%m-%d"),
            "cont_flag":   "1",
        }
        resp = self.fyers.history(data=data)
        if isinstance(resp, dict):
            if resp.get("s") != "ok":
                logger.warning(f"API error for {symbol}: {resp}")
                return pd.DataFrame(columns=["timestamp","open","high","low","close","volume"])
            candles = resp.get("candles", [])
        else:
            candles = []
        if not candles:
            logger.warning(f"No historical data for {symbol}")
            return pd.DataFrame(columns=["timestamp","open","high","low","close","volume"])
        df = pd.DataFrame(candles, columns=["timestamp","open","high","low","close","volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s", utc=True).dt.tz_convert(IST)
        return df.sort_values("timestamp").reset_index(drop=True)

    def get_index_data(self, days: int = None) -> pd.DataFrame:
        days = days or self.config.HIST_DAYS
        df   = self._fetch_history(self.config.UNDERLYING_SYMBOL, days)
        if df.empty:
            return df
        df.set_index("timestamp", inplace=True)
        mask = df.index.indexer_between_time(
            self.config.SESSION_START, self.config.SESSION_END,
            include_start=True, include_end=True,
        )
        df = df.iloc[mask]
        logger.info(f"Loaded NIFTY50 index data: {len(df)} rows in session")
        return df

    def _build_htf_trend(self, df_idx: pd.DataFrame) -> pd.DataFrame:
        c      = self.config
        df_htf = df_idx.resample(c.HTF_RULE).agg(
            {"open":"first","high":"max","low":"min","close":"last","volume":"sum"}
        ).dropna()
        # Drop the current incomplete HTF bar only if it just opened (< 60s old).
        # A brand-new bar has too few ticks for a reliable EMA and causes spurious
        # signals at the very first bar of a new HTF period (e.g. 09:30:00 open).
        # Mid-bar signals (bar > 60s old) keep the partial bar — matches backtest.
        now_ist = datetime.now(IST)
        if not df_htf.empty:
            last_bar_start = df_htf.index[-1]
            if last_bar_start.tzinfo is None:
                last_bar_start = IST.localize(last_bar_start)
            htf_offset   = pd.tseries.frequencies.to_offset(c.HTF_RULE)
            last_bar_end = last_bar_start + htf_offset
            if last_bar_end.tzinfo is None:
                last_bar_end = IST.localize(last_bar_end)
            bar_age_secs = (now_ist - last_bar_start).total_seconds()
            if last_bar_end > now_ist and bar_age_secs < 60:
                df_htf = df_htf.iloc[:-1]
        df_htf["htf_ema"]   = df_htf["close"].ewm(span=c.HTF_EMA_LEN, adjust=False).mean()
        df_htf["htf_trend"] = np.where(
            df_htf["close"] > df_htf["htf_ema"], "up",
            np.where(df_htf["close"] < df_htf["htf_ema"], "down", "flat")
        )
        return df_htf.reset_index()

    def _pick_itm_option_symbol(self, underlying_price: float, direction: str) -> str:
        c    = self.config
        spot = float(underlying_price)
        step = c.STRIKE_STEP
        # ATM strike — round to nearest step (matches backtest exactly)
        strike   = int(round(spot / step)) * step
        strike   = int(max(step, strike))
        opt_type = "CE" if direction == "BUY" else "PE"
        sym      = f"NSE:NIFTY{c.EXPIRY_CODE}{strike:05d}{opt_type}"
        logger.info(f"[VAM LIVE] ATM option: {sym} (spot={spot:.2f}, dir={direction})")
        return sym

    def _create_signal(self, row: pd.Series, direction: str) -> Dict:
        und_price  = row["close"]
        # option_symbol intentionally omitted here — computed only for the
        # LAST (current) signal after the replay loop to avoid ~47 API
        # symbol lookups per bar (one per historical signal).
        return {
            "timestamp":     row["timestamp"],
            "index_symbol":  self.config.UNDERLYING_SYMBOL,
            "index_price":   float(und_price),
            "direction":     direction,
            "option_symbol": None,
        }

    def generate_vam_signals_live(
        self,
        df_index: pd.DataFrame,
        confirmed_signals: Optional[list] = None,
    ) -> pd.DataFrame:
        """
        Replays history to produce VAM + HTF signals.
        Each signal now also carries pivot_level so the engine can
        apply the same pivot wait logic as the backtest.
        confirmed_signals are injected to prevent state drift on EMA recomputation.
        """
        if df_index.empty:
            return pd.DataFrame()

        self.position_size = 0
        self.last_trade    = None

        confirmed_signals = confirmed_signals or []
        confirmed_map: Dict[pd.Timestamp, str] = {
            pd.Timestamp(cs["bar_ts"]).floor("min"): cs["direction"]
            for cs in confirmed_signals
        }

        df_idx    = df_index.copy().reset_index()
        df_vam    = compute_vam_ensemble(df_idx, self.config)
        df_htf    = self._build_htf_trend(df_index.copy())

        df_merged = pd.merge_asof(
            df_vam.sort_values("timestamp"),
            df_htf[["timestamp","htf_trend"]].sort_values("timestamp"),
            on="timestamp", direction="backward",
        )

        warmup  = max(
            self.config.LEN1, self.config.LEN2, self.config.LEN3,
            self.config.LEN4, self.config.LEN5, 50,
        )
        signals = []

        for i in range(warmup, len(df_merged)):
            row      = df_merged.iloc[i]
            ts       = row["timestamp"]
            ts_floor = pd.Timestamp(ts).floor("min")
            ts_time  = ts.time()

            in_win    = self.config.TRADE_START <= ts_time <= self.config.TRADE_END
            htf       = row.get("htf_trend", "flat")
            buy_sig   = bool(row["vam_buy_sig"])
            sell_sig  = bool(row["vam_sell_sig"])

            ok_long  = (self.position_size <= 0) and (self.last_trade != "long")
            ok_short = (self.position_size >= 0) and (self.last_trade != "short")

            # ── inject confirmed signal state to prevent EMA-recompute drift ──
            if ts_floor in confirmed_map:
                confirmed_dir = confirmed_map[ts_floor]
                if confirmed_dir == "BUY" and self.last_trade != "long":
                    self.position_size = 1; self.last_trade = "long"
                elif confirmed_dir == "SELL" and self.last_trade != "short":
                    self.position_size = -1; self.last_trade = "short"
                ok_long  = (self.position_size <= 0) and (self.last_trade != "long")
                ok_short = (self.position_size >= 0) and (self.last_trade != "short")

            if not in_win:
                continue

            if buy_sig and ok_long and htf == "up":
                # ✅ compute pivot level — same logic as backtest
                pivot_lvl = compute_pivot_zone(
                    df_merged, i, self.config.PIVOT_BARS, "BUY"
                )
                sig = self._create_signal(row, "BUY")
                sig["strategy_type"] = "vam_htf_pivot_live"
                sig["htf_trend"]     = htf
                sig["pivot_level"]   = pivot_lvl   # ✅ carry pivot into engine
                signals.append(sig)
                self.position_size = 1; self.last_trade = "long"

            elif sell_sig and ok_short and htf == "down":
                pivot_lvl = compute_pivot_zone(
                    df_merged, i, self.config.PIVOT_BARS, "SELL"
                )
                sig = self._create_signal(row, "SELL")
                sig["strategy_type"] = "vam_htf_pivot_live"
                sig["htf_trend"]     = htf
                sig["pivot_level"]   = pivot_lvl   # ✅ carry pivot into engine
                signals.append(sig)
                self.position_size = -1; self.last_trade = "short"

        # Compute option_symbol ONLY for the last (most recent) signal —
        # not for every historical signal in the replay.
        if signals:
            last = signals[-1]
            last["option_symbol"] = self._pick_itm_option_symbol(
                last["index_price"], last["direction"]
            )

        logger.info(
            f"[VAM LIVE] Replay complete — {len(signals)} signals. "
            f"last_trade={self.last_trade}, position_size={self.position_size}"
        )
        return pd.DataFrame(signals) if signals else pd.DataFrame()


# --------------------------- ORDER MANAGER ---------------------------

class VAMOrderManager:
    def __init__(self, fyers_client: fyersModel.FyersModel, config: TradingConfig):
        self.fyers  = fyers_client
        self.config = config
        self.has_position:   bool            = False
        self.option_symbol:  Optional[str]   = None
        self.direction:      Optional[str]   = None
        self.entry_price:    Optional[float] = None
        self.qty:            int             = self.config.NIFTY_LOT_SIZE
        self.entry_order_id: Optional[str]   = None

    def _save_state(self) -> None:
        state = {
            "has_position":   self.has_position,
            "option_symbol":  self.option_symbol,
            "direction":      self.direction,
            "entry_price":    self.entry_price,
            "entry_order_id": self.entry_order_id,
            "saved_at":       datetime.now(IST).isoformat(),
        }
        tmp = STATE_FILE + ".tmp"
        try:
            with open(tmp, "w") as f:
                json.dump(state, f, indent=2)
            os.replace(tmp, STATE_FILE)
            logger.info(f"[VAM OM] State saved: {state}")
        except Exception as e:
            logger.error(f"[VAM OM] Failed to save state: {e}")

    def _load_state_from_file(self) -> bool:
        if not os.path.exists(STATE_FILE):
            return False
        try:
            with open(STATE_FILE) as f:
                state = json.load(f)
            saved_at = datetime.fromisoformat(state.get("saved_at", ""))
            if saved_at.date() != datetime.now(IST).date():
                os.remove(STATE_FILE); return False
            if not state.get("has_position"):
                return False
            self.has_position   = True
            self.option_symbol  = state.get("option_symbol")
            self.direction      = state.get("direction")
            self.entry_price    = state.get("entry_price")
            self.entry_order_id = state.get("entry_order_id")
            logger.info(
                f"[VAM OM] Restored from file: symbol={self.option_symbol}, "
                f"direction={self.direction}, entry_price={self.entry_price}"
            )
            return True
        except Exception as e:
            logger.error(f"[VAM OM] Failed to load state: {e}"); return False

    def _clear_state_file(self) -> None:
        try:
            if os.path.exists(STATE_FILE):
                os.remove(STATE_FILE)
        except Exception as e:
            logger.error(f"[VAM OM] Failed to remove state file: {e}")

    def _find_bo_order_id(self, symbol: str) -> Optional[str]:
        try:
            ob_resp = self.fyers.orderbook()
            if not isinstance(ob_resp, dict) or ob_resp.get("s") != "ok":
                return None
            for order in ob_resp.get("orderBook", []):
                if (
                    order.get("symbol")      == symbol
                    and order.get("productType") == "BO"
                    and order.get("side")        == 1
                    and order.get("status")      == 2
                ):
                    return order.get("id")
            return None
        except Exception as e:
            logger.error(f"[VAM OM] _find_bo_order_id error: {e}"); return None

    def _restore_state_from_broker(self) -> bool:
        try:
            pos_resp = self.fyers.positions()
            logger.info(f"[VAM OM] Live positions on startup: {pos_resp}")
            if not isinstance(pos_resp, dict) or pos_resp.get("s") != "ok":
                return False
            for pos in pos_resp.get("netPositions", []):
                if pos.get("netQty", 0) == 0:
                    continue
                symbol = pos.get("symbol", "")
                if symbol.endswith("CE"):
                    direction   = "BUY"
                    entry_price = float(pos.get("buyAvg", 0.0))
                elif symbol.endswith("PE"):
                    direction   = "SELL"
                    entry_price = float(pos.get("buyAvg", 0.0))
                else:
                    continue
                self.has_position   = True
                self.option_symbol  = symbol
                self.direction      = direction
                self.entry_price    = entry_price
                self.entry_order_id = self._find_bo_order_id(symbol)
                self._save_state()
                logger.info(
                    f"[VAM OM] Restored from broker: symbol={symbol}, "
                    f"direction={direction}, entry_price={entry_price:.2f}"
                )
                return True
            logger.info("[VAM OM] No open positions at broker. Starting fresh.")
            return False
        except Exception as e:
            logger.error(f"[VAM OM] _restore_state_from_broker error: {e}"); return False

    def restore_state_on_startup(self) -> None:
        logger.info("[VAM OM] Checking for existing position on startup...")
        if not self._load_state_from_file():
            self._restore_state_from_broker()

    def _compute_bo_sl_tp_points(self, entry_price: float) -> tuple[float, float]:
        sl_points = int(round(entry_price * self.config.STOP_LOSS_PCT   / 100.0))
        tp_points = int(round(entry_price * self.config.TAKE_PROFIT_PCT / 100.0))
        return float(sl_points), float(tp_points)

    def open_position(self, signal_row: pd.Series, opt_ltp: float) -> bool:
        symbol    = signal_row.get("option_symbol")
        direction = signal_row.get("direction", "BUY")
        if not symbol:
            logger.warning("[VAM OM] Missing option_symbol"); return False
        if self.has_position:
            logger.warning("[VAM OM] Position already open, ignoring."); return False

        entry_price          = float(opt_ltp)
        sl_points, tp_points = self._compute_bo_sl_tp_points(entry_price)
        limit_price          = round(entry_price * 1.005, 1)

        data = {
            "symbol":       symbol,
            "qty":          int(self.qty),
            "type":         1,
            "side":         1,
            "productType":  "BO",
            "limitPrice":   limit_price,
            "stopPrice":    0,
            "validity":     "DAY",
            "disclosedQty": 0,
            "offlineOrder": False,
            "stopLoss":     float(sl_points),
            "takeProfit":   float(tp_points),
        }
        try:
            logger.info(
                f"[VAM OM] Placing BO: symbol={symbol}, dir={direction}, "
                f"qty={self.qty}, ltp={entry_price:.2f}, limitPrice={limit_price:.1f}, "
                f"SL_pts={sl_points:.0f}, TP_pts={tp_points:.0f}"
            )
            resp = self.fyers.place_order(data=data)
            logger.info(f"[VAM OM] place_order response: {resp}")
            if isinstance(resp, dict) and resp.get("s") == "ok":
                self.has_position   = True
                self.option_symbol  = symbol
                self.direction      = direction
                self.entry_price    = entry_price
                self.entry_order_id = resp.get("id")
                self._save_state()
                logger.info(f"[VAM OM] BO placed, order_id={self.entry_order_id}")
                return True
            else:
                logger.error(f"[VAM OM] BO rejected: {resp}"); return False
        except Exception as e:
            logger.error(f"[VAM OM] place_order error: {e}"); return False

    def close_position(self, exit_reason: str, exit_price: Optional[float] = None) -> bool:
        if not self.has_position or not self.option_symbol:
            logger.warning("[VAM OM] No open position, ignoring."); return False
        if not self.entry_order_id:
            logger.warning("[VAM OM] No entry_order_id, cannot exit BO."); return False
        try:
            # Per Fyers docs: DELETE /positions with {"id": "<symbol>-BO"}
            # exit_order (cancel by order_id) does NOT close a filled BO —
            # only exit_positions on the position itself does.
            position_id = f"{self.option_symbol}-BO"
            logger.info(
                f"[VAM OM] Exiting BO via positions. reason={exit_reason}, "
                f"position_id={position_id}"
            )
            resp = self.fyers.exit_positions(data={"id": position_id})
            logger.info(f"[VAM OM] exit_positions response: {resp}")
            if isinstance(resp, dict) and resp.get("s") == "ok":
                self.has_position   = False
                self.option_symbol  = None
                self.direction      = None
                self.entry_price    = None
                self.entry_order_id = None
                self._clear_state_file()
                logger.info("[VAM OM] BO exited, state cleared.")
                return True
            else:
                logger.error(f"[VAM OM] exit_positions failed: {resp}"); return False
        except Exception as e:
            logger.error(f"[VAM OM] exit_positions error: {e}"); return False


# --------------------------- TRADE LOGGER ---------------------------

class TradeLogger:
    def __init__(self, fyers_client: fyersModel.FyersModel, config: TradingConfig):
        self.fyers  = fyers_client
        self.config = config
        self.trades: list            = []
        self._open:  Optional[dict]  = None

    def log_entry(
        self,
        sig:         pd.Series,
        entry_price: float,
        order_id:    str,
        entry_method: str = "",        # ✅ "already_near_pivot" | "pulled_to_pivot"
        pivot_level:  Optional[float] = None,
    ) -> None:
        self._open = {
            "signal_time":    str(sig.get("timestamp", "")),
            "entry_time":     datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S"),
            "index_price":    sig.get("index_price"),
            "direction":      sig.get("direction"),
            "option_symbol":  sig.get("option_symbol"),
            "strategy_type":  sig.get("strategy_type", "vam_htf_pivot_live"),
            "htf_trend":      sig.get("htf_trend", ""),
            "pivot_level":    pivot_level,       # ✅ log pivot level
            "entry_method":   entry_method,      # ✅ log how entry was triggered
            "entry_price":    round(float(entry_price), 2),
            "entry_order_id": order_id,
            "exit_time":      None,
            "exit_price":     None,
            "exit_reason":    None,
            "pnl_pct":        None,
            "pnl_rupees":     None,
            "result":         None,
        }
        logger.info(
            f"[TRADE LOG] Entry: symbol={self._open['option_symbol']}, "
            f"dir={self._open['direction']}, entry={self._open['entry_price']}, "
            f"pivot={pivot_level}, method={entry_method}"
        )

    def log_flip_exit(self, exit_price: float) -> None:
        if self._open is None:
            logger.warning("[TRADE LOG] log_flip_exit called but no open trade.")
            return
        self._finalize_trade(float(exit_price), "flip_exit")

    def _finalize_trade(self, exit_price: float, exit_reason: str, exit_time: str = None) -> None:
        if self._open is None:
            return
        trade = self._open.copy()
        trade["exit_time"]   = exit_time or datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S")
        trade["exit_price"]  = round(exit_price, 2)
        trade["exit_reason"] = exit_reason
        entry = trade.get("entry_price") or 0.0
        if entry > 0 and exit_price > 0:
            pnl_pct             = (exit_price - entry) / entry * 100.0
            trade["pnl_pct"]    = round(pnl_pct, 2)
            trade["pnl_rupees"] = round(entry * self.config.NIFTY_LOT_SIZE * pnl_pct / 100.0, 2)
            trade["result"]     = "profit" if pnl_pct > 0 else "loss"
        else:
            trade["pnl_pct"] = 0.0; trade["pnl_rupees"] = 0.0; trade["result"] = "unknown"
        self.trades.append(trade)
        self._open = None
        logger.info(
            f"[TRADE LOG] Finalized: {trade['option_symbol']} | "
            f"entry={trade['entry_price']} exit={trade['exit_price']} "
            f"reason={exit_reason} pnl=₹{trade['pnl_rupees']}"
        )

    def resolve_open_trade_at_eod(self) -> None:
        if self._open is None:
            logger.info("[TRADE LOG] No open trade to resolve at EOD."); return
        opt_symbol = self._open["option_symbol"]
        logger.info(f"[TRADE LOG] Resolving EOD trade for {opt_symbol}...")
        try:
            resp = self.fyers.tradebook()
            if not isinstance(resp, dict) or resp.get("s") != "ok":
                self._finalize_trade(0.0, "eod_unresolved"); return
            sell_fills = [
                t for t in resp.get("tradebook", [])
                if t.get("symbol") == opt_symbol and t.get("side") == -1
            ]
            if not sell_fills:
                self._finalize_trade(0.0, "eod_no_fill_found"); return
            latest     = sell_fills[-1]
            exit_price = float(latest.get("tradedPrice", 0))
            exit_time  = latest.get("orderDateTime", datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S"))
            entry      = self._open["entry_price"]
            sl_price   = entry * (1 - self.config.STOP_LOSS_PCT   / 100.0)
            tp_price   = entry * (1 + self.config.TAKE_PROFIT_PCT / 100.0)
            exit_reason = (
                "stop_loss" if abs(exit_price - sl_price) < abs(exit_price - tp_price)
                else "target"
            )
            self._finalize_trade(exit_price, exit_reason, exit_time)
        except Exception as e:
            logger.error(f"[TRADE LOG] EOD resolution error: {e}")
            self._finalize_trade(0.0, "eod_resolution_error")

    def export_to_excel(self) -> Optional[str]:
        if not self.trades:
            logger.info("[TRADE LOG] No trades — nothing to export."); return None
        try:
            import openpyxl
            from openpyxl.styles import PatternFill, Font, Alignment
            from openpyxl.utils import get_column_letter

            df = pd.DataFrame(self.trades)
            df["pnl_pct"]        = pd.to_numeric(df["pnl_pct"],    errors="coerce").fillna(0)
            df["pnl_rupees"]     = pd.to_numeric(df["pnl_rupees"], errors="coerce").fillna(0)
            df["cum_pnl_rupees"] = df["pnl_rupees"].cumsum().round(2)

            today    = datetime.now(IST).strftime("%Y%m%d")
            log_dir  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "logs")
            os.makedirs(log_dir, exist_ok=True)
            xlsx_path = os.path.join(log_dir, f"trades_{today}.xlsx")

            wb = openpyxl.Workbook()
            ws = wb.active; ws.title = "Trades"

            header_fill = PatternFill("solid", fgColor="1F4E79")
            header_font = Font(color="FFFFFF", bold=True)
            green_fill  = PatternFill("solid", fgColor="C6EFCE")
            red_fill    = PatternFill("solid", fgColor="FFC7CE")
            grey_fill   = PatternFill("solid", fgColor="F2F2F2")

            cols = list(df.columns)
            for ci, col in enumerate(cols, 1):
                cell = ws.cell(row=1, column=ci, value=col.replace("_", " ").title())
                cell.fill = header_fill; cell.font = header_font
                cell.alignment = Alignment(horizontal="center")

            for ri, row in df.iterrows():
                result = row.get("result", "")
                fill = green_fill if result == "profit" else red_fill if result == "loss" else grey_fill
                for ci, col in enumerate(cols, 1):
                    cell = ws.cell(row=ri + 2, column=ci, value=row[col])
                    cell.fill = fill; cell.alignment = Alignment(horizontal="center")

            for ci, col in enumerate(cols, 1):
                max_len = max(len(str(col)), *(len(str(df.iloc[r][col])) for r in range(len(df))))
                ws.column_dimensions[get_column_letter(ci)].width = min(max_len + 4, 35)

            ws2 = wb.create_sheet("Summary")
            ws2.column_dimensions["A"].width = 22
            ws2.column_dimensions["B"].width = 20

            total = len(df)
            wins  = int((df["result"] == "profit").sum())
            losses= int((df["result"] == "loss").sum())
            net   = df["pnl_rupees"].sum()
            wr    = round(wins / total * 100, 1) if total > 0 else 0.0

            summary_rows = [
                ("Date",             datetime.now(IST).strftime("%d-%b-%Y")),
                ("Total Trades",     total), ("Wins", wins), ("Losses", losses),
                ("Win Rate (%)",     wr), ("Net PnL (Rs)", round(net, 2)),
                ("Best Trade (Rs)",  round(float(df["pnl_rupees"].max()), 2)),
                ("Worst Trade (Rs)", round(float(df["pnl_rupees"].min()), 2)),
                ("Avg PnL (Rs)",     round(float(df["pnl_rupees"].mean()), 2)),
                ("Stop Loss %",      self.config.STOP_LOSS_PCT),
                ("Take Profit %",    self.config.TAKE_PROFIT_PCT),
                ("Lot Size",         self.config.NIFTY_LOT_SIZE),
                ("Pivot Bars",       self.config.PIVOT_BARS),     # ✅
                ("Pivot Proximity%", self.config.PIVOT_PROXIMITY_PCT),  # ✅
            ]
            label_fill  = PatternFill("solid", fgColor="1F4E79")
            label_font  = Font(color="FFFFFF", bold=True)
            profit_fill = PatternFill("solid", fgColor="C6EFCE")
            loss_fill   = PatternFill("solid", fgColor="FFC7CE")

            for ri, (label, val) in enumerate(summary_rows, 1):
                lc = ws2.cell(row=ri, column=1, value=label)
                vc = ws2.cell(row=ri, column=2, value=val)
                lc.fill = label_fill; lc.font = label_font
                lc.alignment = Alignment(horizontal="left")
                vc.alignment = Alignment(horizontal="center")
                if label == "Net PnL (Rs)":
                    vc.fill = profit_fill if net >= 0 else loss_fill
                    vc.font = Font(bold=True)

            wb.save(xlsx_path)
            logger.info(f"[TRADE LOG] Excel exported → {xlsx_path}")
            return xlsx_path
        except Exception as e:
            logger.error(f"[TRADE LOG] Excel export failed: {e}"); return None


# --------------------------- LIVE ENGINE ---------------------------

class VAMLiveEngine:
    def __init__(self, bot: VAMLiveBot, access_token: str):
        self.bot          = bot
        self.access_token = access_token
        self.index_symbol = bot.config.UNDERLYING_SYMBOL

        self.index_candles     = bot.get_index_data(days=bot.config.HIST_DAYS).reset_index()
        self.current_index_bar = None

        self.last_tick_time:  Optional[datetime] = None
        self.logged_raw_once: bool               = False

        # Parse timeframe — supports "10S" (seconds) or "1" (minutes)
        tf = str(self.bot.config.TIMEFRAME)
        if tf.upper().endswith("S"):
            self.timeframe_secs = int(tf[:-1])
            self.timeframe_mins = None
        else:
            self.timeframe_secs = None
            self.timeframe_mins = int(tf)
        bar_duration_secs   = self.timeframe_secs if self.timeframe_secs else (self.timeframe_mins * 60)
        self.heartbeat_secs = bar_duration_secs * 20   # reconnect after 20 missed bars

        self._all_ws_sockets: list                              = []
        self.ws:              Optional[data_ws.FyersDataSocket] = None

        self.order_manager = VAMOrderManager(self.bot.fyers, self.bot.config)
        self.order_manager.restore_state_on_startup()

        # ✅ Pivot wait state — mirrors backtest Phase 1 logic
        self.pending_signal:    Optional[pd.Series]  = None
        self.pending_bar_ts:    Optional[datetime]   = None
        self.pivot_wait_active: bool                 = False   # True while waiting for pivot touch
        self.pivot_wait_signal: Optional[pd.Series]  = None   # signal awaiting pivot touch
        self.pivot_wait_level:  Optional[float]      = None   # pivot price to watch
        self.pivot_wait_dir:    Optional[str]        = None   # BUY or SELL

        self.option_ltp_cache:      dict[str, float]    = {}
        self._execution_lock:       threading.Lock      = threading.Lock()
        self._last_executed_bar_ts: Optional[datetime]  = None
        self._confirmed_signals_today: list             = []
        self._last_api_refresh:     Optional[datetime]  = None
        # Tracks position_size from last replay — used to detect new signals
        # instead of fragile timestamp-floor matching (which was always empty).
        self._last_replay_position: int                 = 0

        self.trade_logger = TradeLogger(self.bot.fyers, self.bot.config)

    def _bar_key(self, ts: datetime) -> datetime:
        if self.timeframe_secs is not None:
            # Second-based timeframe (e.g. 10S)
            s_bucket = (ts.second // self.timeframe_secs) * self.timeframe_secs
            return ts.replace(microsecond=0, second=s_bucket)
        else:
            # Minute-based timeframe (e.g. 1)
            mb = (ts.minute // self.timeframe_mins) * self.timeframe_mins
            return ts.replace(second=0, microsecond=0, minute=mb)

    # ── WebSocket callbacks ──────────────────────────────────────────────

    def _on_open(self):
        logger.info("[VAM LIVE] WS connected — subscribing")
        if self.ws is not None:
            self.ws.subscribe(symbols=[self.index_symbol], data_type="SymbolUpdate")

    def _on_close(self, msg):
        logger.warning(f"[VAM LIVE] WS closed: {msg}")

    def _on_error(self, msg):
        logger.error(f"[VAM LIVE] WS error: {msg}")

    def _on_message(self, msg):
        try:
            if not self.logged_raw_once:
                logger.info(f"[VAM LIVE] RAW WS MSG (once): {msg}")
                self.logged_raw_once = True

            if isinstance(msg, list) and msg:
                data = msg[0]
            elif isinstance(msg, dict):
                data = msg.get("symbolData") or msg
            else:
                return
            if not isinstance(data, dict):
                return

            symbol = data.get("symbol")
            v_dict = data.get("v", {}) or {}
            ltp    = (
                v_dict.get("lp") or data.get("ltp")
                or data.get("price") or data.get("close")
            )
            if ltp is None or not symbol:
                return

            now      = datetime.now(IST)
            self.last_tick_time = now
            bar_time = self._bar_key(now)
            ltp      = float(ltp)

            open_p = float(v_dict.get("open_price") or v_dict.get("op") or ltp)
            high_p = float(v_dict.get("high_price") or v_dict.get("high") or ltp)
            low_p  = float(v_dict.get("low_price")  or v_dict.get("low")  or ltp)
            vol    = float(v_dict.get("volume")      or v_dict.get("vol")  or
                           data.get("volume", 0)     or 0)

            if symbol == self.index_symbol:
                self._handle_index_tick(bar_time, ltp, open_p, high_p, low_p, vol)

        except Exception as e:
            logger.error(f"[VAM LIVE] on_message error: {e} | raw: {msg}")

    def _handle_index_tick(
        self, bar_time: datetime, ltp: float,
        open_p: float, high_p: float, low_p: float, vol: float,
    ):
        if self.current_index_bar is None or self.current_index_bar["timestamp"] != bar_time:
            if self.current_index_bar is not None:
                self._finalize_index_bar(self.current_index_bar)
            if (
                self.pending_signal is not None
                and self.pending_bar_ts is not None
                and bar_time > self.pending_bar_ts
            ):
                self._execute_pending_signal_at_next_bar(bar_time)

            self.current_index_bar = {
                "timestamp": bar_time,
                "open": open_p, "high": high_p,
                "low": low_p,   "close": ltp, "volume": vol,
            }
        else:
            self.current_index_bar["high"]   = max(self.current_index_bar["high"], high_p)
            self.current_index_bar["low"]    = min(self.current_index_bar["low"],  low_p)
            self.current_index_bar["close"]  = ltp
            self.current_index_bar["volume"] += vol

            # ✅ PIVOT WAIT — check each tick while waiting for zone touch
            if self.pivot_wait_active:
                self._check_pivot_touch_on_tick(low_p, high_p, ltp, bar_time)

    # ── ✅ Pivot wait logic (backtest Phase 1 equivalent) ─────────────────

    def _check_pivot_touch_on_tick(
        self, low_p: float, high_p: float, ltp: float, bar_time: datetime
    ):
        """
        Called on every tick while waiting for price to touch the pivot zone.
        Mirrors backtest Phase 1 Case B.
        """
        if not self.pivot_wait_active or self.pivot_wait_signal is None:
            return

        pivot_lvl  = self.pivot_wait_level
        direction  = self.pivot_wait_dir
        prox_band  = pivot_lvl * (self.bot.config.PIVOT_PROXIMITY_PCT / 100.0)

        check_price = low_p if direction == "BUY" else high_p

        if abs(check_price - pivot_lvl) <= prox_band:
            logger.info(
                f"[VAM PIVOT] Price {check_price:.2f} touched pivot zone "
                f"{pivot_lvl:.2f} ± {prox_band:.2f} — queuing entry"
            )
            # queue for execution on next bar open (same as backtest)
            self.pending_signal    = self.pivot_wait_signal.copy()
            self.pending_bar_ts    = bar_time
            self.pivot_wait_active = False
            self.pivot_wait_signal = None
            self.pivot_wait_level  = None
            self.pivot_wait_dir    = None

    def _cancel_pivot_wait(self, reason: str):
        logger.info(f"[VAM PIVOT] Wait cancelled: {reason}")
        self.pivot_wait_active = False
        self.pivot_wait_signal = None
        self.pivot_wait_level  = None
        self.pivot_wait_dir    = None

    # ── Bar finalisation — signal detection + pivot entry logic ──────────

    def _finalize_index_bar(self, bar: dict):
        bar_ts = bar["timestamp"]
        today  = datetime.now(IST).date()

        if bar_ts.date() != today:
            return
        if not (self.bot.config.SESSION_START <= bar_ts.time() <= self.bot.config.SESSION_END):
            return

        # ✅ EOD forced exit at 3:20 PM
        if bar_ts.time() >= self.bot.config.EOD_EXIT_TIME:
            if self.order_manager.has_position:
                logger.info("[VAM LIVE] EOD 3:20 PM — force closing position.")
                opt_symbol = self.order_manager.option_symbol
                ltp = self._get_ltp(opt_symbol) or self.order_manager.entry_price
                closed = self.order_manager.close_position("eod_exit", exit_price=ltp)
                if closed:
                    self.trade_logger.log_flip_exit(ltp or 0.0)
            if self.pivot_wait_active:
                self._cancel_pivot_wait("EOD reached")
            return

        # ✅ Cancel pivot wait if a flip signal arrives before touch
        if self.pivot_wait_active:
            # will be checked below after signals are generated
            pass

        logger.info(f"[VAM LIVE] Finalized index bar: {bar}")

        self.index_candles = (
            pd.concat([self.index_candles, pd.DataFrame([bar])], ignore_index=True)
            .drop_duplicates("timestamp")
            .sort_values("timestamp")
            .tail(22500)
            .reset_index(drop=True)
        )

        df_for_signals = self.index_candles.copy()
        df_for_signals.set_index("timestamp", inplace=True)

        signals = self.bot.generate_vam_signals_live(
            df_for_signals,
            confirmed_signals=self._confirmed_signals_today,
        )
        new_position = self.bot.position_size  # captured after replay

        # ── Signal detection: compare replay position to last known position ──
        # The old approach (filter signals by bar_ts.floor("min")) was always
        # returning empty because signal timestamps rarely fall on the exact
        # same minute boundary as the bar being finalized.
        if signals.empty or new_position == self._last_replay_position:
            self._last_replay_position = new_position
            return

        self._last_replay_position = new_position
        last_sig     = signals.iloc[-1]   # last signal = the one that changed position
        pivot_lvl    = last_sig.get("pivot_level")
        direction    = last_sig["direction"]
        signal_price = float(last_sig["index_price"])
        bar_ts_floor = pd.Timestamp(bar_ts).floor("min")

        # ✅ If a new signal arrives while waiting, cancel the old pivot wait
        if self.pivot_wait_active and self.pivot_wait_dir != direction:
            self._cancel_pivot_wait("Opposing signal — pivot wait cancelled")

        # Record confirmed signal for replay state injection
        confirmed_entry = {
            "bar_ts":    bar_ts_floor,
            "direction": direction,
        }
        if not any(c["bar_ts"] == bar_ts_floor for c in self._confirmed_signals_today):
            self._confirmed_signals_today.append(confirmed_entry)
            logger.info(
                f"[VAM LIVE] Confirmed signal: {direction} at {bar_ts_floor}"
            )

        # ── ✅ PIVOT ENTRY LOGIC (mirrors backtest Phase 1) ──────────────

        if pivot_lvl is None:
            logger.warning(f"[VAM LIVE] No pivot found for signal at {bar_ts} — skipping")
            return

        prox_band = pivot_lvl * (self.bot.config.PIVOT_PROXIMITY_PCT / 100.0)

        # Case A: already within proximity of pivot → enter immediately
        if direction == "BUY":
            already_near = (signal_price - pivot_lvl) <= prox_band
        else:
            already_near = (pivot_lvl - signal_price) <= prox_band

        if already_near:
            logger.info(
                f"[VAM LIVE] Price {signal_price:.2f} already near pivot "
                f"{pivot_lvl:.2f} (±{prox_band:.2f}) — entering now"
            )
            self.pending_signal = last_sig.copy()
            self.pending_bar_ts = bar_ts
            self._store_pending_metadata("already_near_pivot", pivot_lvl)
        else:
            # Case B: price not near pivot — start watching for touch
            logger.info(
                f"[VAM LIVE] Price {signal_price:.2f} not near pivot "
                f"{pivot_lvl:.2f} — waiting for pullback"
            )
            self.pivot_wait_active = True
            self.pivot_wait_signal = last_sig.copy()
            self.pivot_wait_level  = pivot_lvl
            self.pivot_wait_dir    = direction

    def _store_pending_metadata(self, entry_method: str, pivot_lvl: Optional[float]):
        """Attach entry_method and pivot_level to the pending signal for trade logging."""
        if self.pending_signal is not None:
            self.pending_signal = self.pending_signal.copy()
            self.pending_signal["_entry_method"] = entry_method
            self.pending_signal["_pivot_level"]  = pivot_lvl

    def _get_ltp(self, symbol: Optional[str]) -> Optional[float]:
        if not symbol:
            return None
        try:
            resp = self.bot.fyers.quotes(data={"symbols": symbol})
            if isinstance(resp, dict) and resp.get("d"):
                d0 = resp["d"][0]
                if isinstance(d0, dict):
                    return float(d0.get("v", {}).get("lp") or 0) or None
        except Exception as e:
            logger.warning(f"[VAM LIVE] _get_ltp error for {symbol}: {e}")
        return self.option_ltp_cache.get(symbol)

    # ── Execute pending signal at next bar open ───────────────────────────

    def _execute_pending_signal_at_next_bar(self, bar_time: datetime):
        if not self._execution_lock.acquire(blocking=False):
            logger.info("[VAM CONFIRM] Execution already in progress, skipping.")
            return

        try:
            if self._last_executed_bar_ts == self.pending_bar_ts:
                logger.info(f"[VAM CONFIRM] Signal for bar {self.pending_bar_ts} already executed.")
                self.pending_signal = None; self.pending_bar_ts = None
                return

            # ── Staleness guard — skip if signal bar closed too long ago ──────
            pending_ts = self.pending_bar_ts
            if pending_ts.tzinfo is None:
                pending_ts = IST.localize(pending_ts)
            stale_secs = (datetime.now(IST) - pending_ts).total_seconds()
            if stale_secs > self.bot.config.SIGNAL_STALE_SECS:
                logger.info(
                    f"[VAM CONFIRM] ⏩ Signal too old ({stale_secs:.0f}s "
                    f"> {self.bot.config.SIGNAL_STALE_SECS}s) — skipping entry."
                )
                self.pending_signal = None; self.pending_bar_ts = None
                return

            sig        = self.pending_signal.copy()
            opt_symbol = sig["option_symbol"]
            direction  = sig["direction"]
            entry_method = sig.get("_entry_method", "unknown")
            pivot_lvl    = sig.get("_pivot_level")

            opt_ltp = self._get_ltp(opt_symbol)
            if opt_ltp is None:
                cached = self.option_ltp_cache.get(opt_symbol)
                if cached is None:
                    logger.info(f"[VAM CONFIRM] No LTP for {opt_symbol}, skipping entry.")
                    self.pending_signal = None; self.pending_bar_ts = None
                    return
                opt_ltp = float(cached)
                logger.info(f"[VAM CONFIRM] Using cached LTP {opt_ltp:.2f}")
            else:
                self.option_ltp_cache[opt_symbol] = opt_ltp

            if self.order_manager.has_position:
                if direction != self.order_manager.direction:
                    logger.info("[VAM LIVE] Opposite signal — flipping position.")
                    closed = self.order_manager.close_position("flip_exit", exit_price=opt_ltp)
                    if not closed:
                        logger.error("[VAM LIVE] Flip failed — skipping new entry.")
                        self.pending_signal = None; self.pending_bar_ts = None
                        return
                    self.trade_logger.log_flip_exit(opt_ltp)

            if not self.order_manager.has_position:
                ok = self.order_manager.open_position(sig, opt_ltp)
                if ok:
                    logger.info(f"[VAM LIVE] BO opened for {opt_symbol} via {entry_method}")
                    self._last_executed_bar_ts = self.pending_bar_ts
                    self.trade_logger.log_entry(
                        sig, opt_ltp,
                        self.order_manager.entry_order_id,
                        entry_method=entry_method,
                        pivot_level=pivot_lvl,
                    )

            self.pending_signal = None; self.pending_bar_ts = None

        except Exception as e:
            logger.warning(f"[VAM CONFIRM] Execute error: {e}")
            self.pending_signal = None; self.pending_bar_ts = None
        finally:
            self._execution_lock.release()

    # ── WS / threading ───────────────────────────────────────────────────

    def _create_and_connect_ws(self):
        logger.info(f"[VAM LIVE] Closing {len(self._all_ws_sockets)} existing WS socket(s).")
        for old_ws in list(self._all_ws_sockets):
            try:
                old_ws.close()
            except Exception as e:
                logger.debug(f"[VAM LIVE] WS close error: {e}")
        self._all_ws_sockets.clear()
        time.sleep(2)

        logger.info("[VAM LIVE] Creating new FyersDataSocket...")
        new_ws = data_ws.FyersDataSocket(
            access_token  = self.access_token,
            log_path      = "",
            litemode      = False,
            write_to_file = False,
            reconnect     = True,
            on_connect    = self._on_open,
            on_close      = self._on_close,
            on_error      = self._on_error,
            on_message    = self._on_message,
        )
        self.ws = new_ws
        self._all_ws_sockets.append(new_ws)
        new_ws.connect()

    def _heartbeat_loop(self):
        while True:
            try:
                if self.last_tick_time is not None:
                    delta = (datetime.now(IST) - self.last_tick_time).total_seconds()
                    if delta > self.heartbeat_secs:
                        logger.warning(f"[VAM LIVE] No ticks for {delta:.0f}s — reconnecting.")
                        try:
                            self._create_and_connect_ws()
                        except Exception as e:
                            logger.error(f"[VAM LIVE] Reconnect failed: {e}")
                        self.last_tick_time = datetime.now(IST)
                time.sleep(5)
            except Exception as e:
                logger.error(f"[VAM LIVE] Heartbeat error: {e}")
                time.sleep(5)

    def _refresh_todays_bars_from_api(self) -> None:
        logger.info("[VAM LIVE] API bar refresh...")
        try:
            df_fresh = self.bot.get_index_data(days=2)
            if df_fresh.empty:
                return
            df_fresh = df_fresh.reset_index()
            today    = datetime.now(IST).date()

            if df_fresh["timestamp"].dt.tz is None:
                df_fresh["timestamp"] = df_fresh["timestamp"].dt.tz_localize(IST)
            else:
                df_fresh["timestamp"] = df_fresh["timestamp"].dt.tz_convert(IST)

            df_today = df_fresh[df_fresh["timestamp"].dt.date == today].copy()
            if df_today.empty:
                return

            if self.index_candles["timestamp"].dt.tz is None:
                self.index_candles["timestamp"] = self.index_candles["timestamp"].dt.tz_localize(IST)

            prev_rows  = (self.index_candles["timestamp"].dt.date != today)
            historical = self.index_candles[prev_rows]
            self.index_candles = (
                pd.concat([historical, df_today], ignore_index=True)
                .drop_duplicates("timestamp")
                .sort_values("timestamp")
                .reset_index(drop=True)
            )
            logger.info(f"[VAM LIVE] API refresh: replaced today with {len(df_today)} bars.")
            self._last_api_refresh = datetime.now(IST)
        except Exception as e:
            logger.error(f"[VAM LIVE] API refresh failed: {e}")

    def _api_refresh_loop(self) -> None:
        interval = self.bot.config.API_REFRESH_INTERVAL_SECS
        while True:
            try:
                time.sleep(interval)
                now = datetime.now(IST)
                if self.bot.config.SESSION_START <= now.time() <= self.bot.config.SESSION_END:
                    self._refresh_todays_bars_from_api()
            except Exception as e:
                logger.error(f"[VAM LIVE] API refresh loop error: {e}")

    def stop(self) -> None:
        logger.info("[VAM LIVE] stop() — EOD cleanup...")
        for ws in list(self._all_ws_sockets):
            try:
                ws.close()
            except Exception as e:
                logger.error(f"[VAM LIVE] WS close error: {e}")
        self._all_ws_sockets.clear()
        self.trade_logger.resolve_open_trade_at_eod()
        self.trade_logger.export_to_excel()
        logger.info("[VAM LIVE] EOD cleanup complete.")

    def start(self):
        logger.info("Starting VAM live engine with pivot wait + EOD exit.")
        try:
            self._create_and_connect_ws()

            threading.Thread(target=self._heartbeat_loop, daemon=True).start()
            threading.Thread(target=self._api_refresh_loop, daemon=True).start()
            logger.info(
                f"[VAM LIVE] Threads started "
                f"(API refresh every {self.bot.config.API_REFRESH_INTERVAL_SECS}s)."
            )
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            logger.info("KeyboardInterrupt — stopping.")
        finally:
            for ws in list(self._all_ws_sockets):
                try:
                    ws.close()
                except Exception:
                    pass


# --------------------------- MAIN ---------------------------

if __name__ == "__main__":
    bot       = VAMLiveBot()
    app_id    = bot.app_id
    raw_token = bot.access_token

    if not app_id or not raw_token:
        raise RuntimeError("app_id or token missing; check fyers_appid.txt / fyers_token.txt")

    ws_access_token = f"{app_id}:{raw_token}"
    engine = VAMLiveEngine(bot, access_token=ws_access_token)
    engine.start()
