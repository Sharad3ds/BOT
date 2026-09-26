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
    """
    NIFTY weekly expiry: next Tuesday.
    If today is Tuesday, use next week's Tuesday.
    """
    now_ist = now_ist or datetime.now(IST)
    if now_ist.weekday() < 1:
        days_ahead = 1 - now_ist.weekday()
    else:
        days_ahead = 7 - (now_ist.weekday() - 1)
    return (now_ist + timedelta(days=days_ahead)).replace(
        hour=0, minute=0, second=0, microsecond=0
    )


def get_nifty_monthly_expiry(year: int, month: int) -> datetime:
    """
    NIFTY monthly expiry: last Tuesday of the month.
    """
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


# --------------------------- CONFIG (VAM) ---------------------------

class TradingConfig:
    UNDERLYING_SYMBOL = "NSE:NIFTY50-INDEX"
    TIMEFRAME = "1"
    HIST_DAYS = 60

    STRIKE_STEP = 100
    NIFTY_LOT_SIZE = 65

    # Expiry control: "WEEKLY" or "MONTHLY"
    EXPIRY_TYPE = "WEEKLY"

    # VAM ensemble thresholds
    VAM_BUY_SCORE  =  0.10
    VAM_SELL_SCORE = -0.10

    # VAM 5-speed params
    LEN1 = 8
    LEN2 = 10
    LEN3 = 14
    LEN4 = 20
    LEN5 = 28

    MULT1 = 1.05
    MULT2 = 1.10
    MULT3 = 1.15
    MULT4 = 1.20
    MULT5 = 1.25

    BAND1 = 1.15
    BAND2 = 1.20
    BAND3 = 1.25
    BAND4 = 1.30
    BAND5 = 1.35

    SMOOTH1 = 2
    SMOOTH2 = 2
    SMOOTH3 = 3
    SMOOTH4 = 3
    SMOOTH5 = 4

    # ATR trail
    ATR_LEN       = 14
    ATR_LAYERMULT = 0.90
    TRAIL_SMOOTH  = 4

    # Higher timeframe trend filter — "120T" matches backtest exactly (120-minute bars)
    HTF_RULE    = "120T"
    HTF_EMA_LEN = 5

    SESSION_START = dtime(9, 20)
    SESSION_END   = dtime(15, 30)
    TRADE_START   = dtime(9, 30)
    TRADE_END     = dtime(15, 0)

    # SL/TP — single source of truth, used by both signal engine and order manager
    STOP_LOSS_PCT   = 20.0   # 20% loss from entry
    TAKE_PROFIT_PCT = 60.0   # 60% gain from entry

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
    tr_val = np.maximum(hl, np.maximum(hc, lc))
    return pd.Series(tr_val, index=df.index)


def compute_r1_side(
    df: pd.DataFrame,
    length: int,
    r1_mult: float,
    band_mult: float,
    smooth: int
) -> (np.ndarray, pd.Series, pd.Series):
    src = df["close"]
    tr_val = tr(df)

    r1_raw = ema(tr_val, length) * r1_mult
    r1 = ema(r1_raw, smooth)

    center = pd.Series(index=df.index, dtype=float)
    center.iloc[0] = src.iloc[0]

    for i in range(1, len(df)):
        prev_center = center.iloc[i - 1]
        diff = src.iloc[i] - prev_center
        if diff > r1.iloc[i]:
            center.iloc[i] = prev_center + (diff - r1.iloc[i])
        elif diff < -r1.iloc[i]:
            center.iloc[i] = prev_center + (diff + r1.iloc[i])
        else:
            center.iloc[i] = prev_center

    upper_raw = center + r1 * band_mult
    lower_raw = center - r1 * band_mult

    upper = ema(upper_raw, smooth)
    lower = ema(lower_raw, smooth)

    break_long  = (src > upper.shift(1)) & (src.shift(1) <= upper.shift(2))
    break_short = (src < lower.shift(1)) & (src.shift(1) >= lower.shift(2))

    side = np.ones(len(df), dtype=int)
    for i in range(1, len(df)):
        if break_long.iloc[i]:
            side[i] = 1
        elif break_short.iloc[i]:
            side[i] = -1
        else:
            side[i] = side[i - 1]

    return side, upper, lower


def compute_vam_ensemble(df: pd.DataFrame, cfg: TradingConfig) -> pd.DataFrame:
    df_local = df.copy()

    params = {
        "lens":   [cfg.LEN1, cfg.LEN2, cfg.LEN3, cfg.LEN4, cfg.LEN5],
        "mults":  [cfg.MULT1, cfg.MULT2, cfg.MULT3, cfg.MULT4, cfg.MULT5],
        "bands":  [cfg.BAND1, cfg.BAND2, cfg.BAND3, cfg.BAND4, cfg.BAND5],
        "smooth": [cfg.SMOOTH1, cfg.SMOOTH2, cfg.SMOOTH3, cfg.SMOOTH4, cfg.SMOOTH5],
    }

    sides = []
    uppers = {}
    lowers = {}

    for i in range(5):
        side_i, upper_i, lower_i = compute_r1_side(
            df_local,
            params["lens"][i],
            params["mults"][i],
            params["bands"][i],
            params["smooth"][i],
        )
        sides.append(side_i)
        uppers[i + 1] = upper_i
        lowers[i + 1] = lower_i

    score_raw = np.mean(sides, axis=0)
    score = score_raw / 5.0
    df_local["vam_score"] = score

    df_local["vam_is_bull"] = df_local["vam_score"] > cfg.VAM_BUY_SCORE
    df_local["vam_is_bear"] = df_local["vam_score"] < cfg.VAM_SELL_SCORE
    df_local["vam_is_flat"] = ~df_local["vam_is_bull"] & ~df_local["vam_is_bear"]

    df_local["vam_buy_sig"] = (
        (df_local["vam_score"].shift(1) <= cfg.VAM_BUY_SCORE)
        & (df_local["vam_score"] > cfg.VAM_BUY_SCORE)
    )
    df_local["vam_sell_sig"] = (
        (df_local["vam_score"].shift(1) >= cfg.VAM_SELL_SCORE)
        & (df_local["vam_score"] < cfg.VAM_SELL_SCORE)
    )

    tr_val = tr(df_local)
    atr_layer = ema(tr_val, cfg.ATR_LEN)

    trail_lower1 = ema(lowers[3], cfg.TRAIL_SMOOTH)
    trail_upper1 = ema(uppers[3], cfg.TRAIL_SMOOTH)
    trail_lower2 = ema(trail_lower1 - atr_layer * cfg.ATR_LAYERMULT, cfg.TRAIL_SMOOTH)
    trail_upper2 = ema(trail_upper1 + atr_layer * cfg.ATR_LAYERMULT, cfg.TRAIL_SMOOTH)

    df_local["vam_trail_lower1"] = trail_lower1
    df_local["vam_trail_upper1"] = trail_upper1
    df_local["vam_trail_lower2"] = trail_lower2
    df_local["vam_trail_upper2"] = trail_upper2

    return df_local


# --------------------------- VAM LIVE BOT (SIGNAL ENGINE) ---------------------------

class VAMLiveBot:
    def __init__(self):
        self.config = TradingConfig()
        self.fyers = self._init_fyers()

        # Signal state — tracks last signal direction to prevent duplicate entries.
        # Reset to neutral on startup; rebuilt by replaying history before each
        # live signal check, exactly as the backtest does in its forward loop.
        self.position_size: int = 0
        self.last_trade: Optional[str] = None

        self.app_id: Optional[str] = None
        self.access_token: Optional[str] = None
        if os.path.exists("fyers_appid.txt"):
            with open("fyers_appid.txt", "r") as f:
                self.app_id = f.read().strip()
        if os.path.exists("fyers_token.txt"):
            with open("fyers_token.txt", "r") as f:
                self.access_token = f.read().strip()

        logger.info("VAMLiveBot initialized (signals for live).")

    def _init_fyers(self):
        if not all(os.path.exists(f) for f in ["fyers_appid.txt", "fyers_token.txt"]):
            raise FileNotFoundError("Missing fyers_appid.txt or fyers_token.txt")

        app_id = open("fyers_appid.txt", "r").read().strip()
        token  = open("fyers_token.txt",  "r").read().strip()
        if not app_id or not token:
            raise ValueError("Empty app_id or token in credential files")

        fy = fyersModel.FyersModel(
            client_id=app_id,
            token=token,
            is_async=False,
            log_path="logs"
        )
        logger.info("Fyers V3 API initialized for VAMLiveBot")
        return fy

    # -------- history helpers --------

    def _fetch_history(self, symbol: str, days: int) -> pd.DataFrame:
        end_dt_ist   = datetime.now(IST)
        start_dt_ist = end_dt_ist - timedelta(days=days)

        data = {
            "symbol":     symbol,
            "resolution": self.config.TIMEFRAME,
            "date_format": "1",
            "range_from": start_dt_ist.strftime("%Y-%m-%d"),
            "range_to":   end_dt_ist.strftime("%Y-%m-%d"),
            "cont_flag":  "1",
        }
        resp = self.fyers.history(data=data)

        if isinstance(resp, dict):
            if resp.get("s") != "ok":
                logger.warning(f"API error for {symbol}: {resp}")
                return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])
            candles = resp.get("candles", [])
        else:
            candles = resp.get("candles", []) if hasattr(resp, "get") else []

        if not candles:
            logger.warning(f"No historical data for {symbol}")
            return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])

        df = pd.DataFrame(candles, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s", utc=True).dt.tz_convert(IST)
        df = df.sort_values("timestamp").reset_index(drop=True)
        return df

    def get_index_data(self, days: int = None) -> pd.DataFrame:
        days = days or self.config.HIST_DAYS
        df = self._fetch_history(self.config.UNDERLYING_SYMBOL, days)
        if df.empty:
            return df
        df.set_index("timestamp", inplace=True)
        session_mask = df.index.indexer_between_time(
            self.config.SESSION_START, self.config.SESSION_END,
            include_start=True, include_end=True
        )
        df = df.iloc[session_mask]
        logger.info(f"Loaded NIFTY50 index data with {len(df)} rows [IST] in session (VAM)")
        return df

    # -------- higher timeframe trend filter --------

    def _build_htf_trend(self, df_idx: pd.DataFrame) -> pd.DataFrame:
        c = self.config
        df_htf = df_idx.resample(c.HTF_RULE).agg({
            "open":   "first",
            "high":   "max",
            "low":    "min",
            "close":  "last",
            "volume": "sum",
        }).dropna()

        df_htf["htf_ema"]   = df_htf["close"].ewm(span=c.HTF_EMA_LEN, adjust=False).mean()
        df_htf["htf_trend"] = np.where(
            df_htf["close"] > df_htf["htf_ema"], "up",
            np.where(df_htf["close"] < df_htf["htf_ema"], "down", "flat")
        )
        df_htf = df_htf.reset_index()
        return df_htf

    # -------- option helper --------

    def _pick_itm_option_symbol(self, underlying_price: float, direction: str) -> str:
        c    = self.config
        spot = float(underlying_price)
        step = c.STRIKE_STEP

        if direction == "BUY":
            atm_strike = (int(spot) // step) * step
            strike     = atm_strike - step
            opt_type   = "CE"
        else:
            atm_strike = ((int(spot) + step - 1) // step) * step
            strike     = atm_strike + step
            opt_type   = "PE"

        strike     = max(step, (int(strike) // step) * step)
        strike_int = int(strike)
        symbol     = f"NSE:NIFTY{c.EXPIRY_CODE}{strike_int:05d}{opt_type}"
        logger.info(f"[VAM LIVE] Option symbol: {symbol} (spot={spot:.2f}, dir={direction})")
        return symbol

    def _create_signal(self, row: pd.Series, direction: str) -> Dict:
        """
        Build signal dict only. Does NOT mutate position_size / last_trade.
        State is updated explicitly in the signal loop, exactly as in the backtest.
        """
        und_price  = row["close"]
        opt_symbol = self._pick_itm_option_symbol(underlying_price=und_price, direction=direction)

        signal = {
            "timestamp":     row["timestamp"],
            "index_symbol":  self.config.UNDERLYING_SYMBOL,
            "index_price":   float(und_price),
            "direction":     direction,
            "option_symbol": opt_symbol,
        }
        return signal

    # -------- VAM signals on index + HTF filter --------

    def generate_vam_signals_live(self, df_index: pd.DataFrame) -> pd.DataFrame:
        """
        Signal logic mirrors the backtest exactly:
          - Reset signal state before each full replay.
          - Loop from warmup onwards, check trade window + HTF filter.
          - Update position_size / last_trade explicitly after each signal,
            NOT inside _create_signal.
        After replay, self.last_trade and self.position_size correctly reflect
        the last historical signal direction — used to filter the next live entry.
        """
        if df_index.empty:
            return pd.DataFrame()

        # ---- Reset signal state before replay (mirrors backtest __init__) ----
        self.position_size = 0
        self.last_trade    = None

        df_idx  = df_index.copy().reset_index()
        df_vam  = compute_vam_ensemble(df_idx, self.config)
        df_htf  = self._build_htf_trend(df_index.copy())

        df_merged = pd.merge_asof(
            df_vam.sort_values("timestamp"),
            df_htf[["timestamp", "htf_trend"]].sort_values("timestamp"),
            on="timestamp",
            direction="backward"
        )

        signals = []
        warmup  = max(
            self.config.LEN1, self.config.LEN2, self.config.LEN3,
            self.config.LEN4, self.config.LEN5, 50,
        )

        for i in range(warmup, len(df_merged)):
            row = df_merged.iloc[i]

            ts: datetime = row["timestamp"]
            ts_time = ts.time()
            in_trade_window = (
                ts_time >= self.config.TRADE_START
                and ts_time <= self.config.TRADE_END
            )

            htf_trend = row.get("htf_trend", "flat")
            buy_sig   = bool(row["vam_buy_sig"])
            sell_sig  = bool(row["vam_sell_sig"])

            # Exactly as backtest: no same-direction repeat
            prev_pos_ok_long  = (self.position_size <= 0) and (self.last_trade != "long")
            prev_pos_ok_short = (self.position_size >= 0) and (self.last_trade != "short")

            if in_trade_window:
                if buy_sig and prev_pos_ok_long and (htf_trend == "up"):
                    sig = self._create_signal(row, "BUY")
                    sig["strategy_type"] = "vam_htf_live"
                    sig["htf_trend"]     = htf_trend
                    signals.append(sig)
                    # Update state explicitly, same as backtest loop
                    self.position_size = 1
                    self.last_trade    = "long"

                elif sell_sig and prev_pos_ok_short and (htf_trend == "down"):
                    sig = self._create_signal(row, "SELL")
                    sig["strategy_type"] = "vam_htf_live"
                    sig["htf_trend"]     = htf_trend
                    signals.append(sig)
                    # Update state explicitly, same as backtest loop
                    self.position_size = -1
                    self.last_trade    = "short"

        logger.info(
            f"[VAM LIVE] Replay complete — {len(signals)} historical signals. "
            f"last_trade={self.last_trade}, position_size={self.position_size}"
        )
        return pd.DataFrame(signals) if signals else pd.DataFrame()


# --------------------------- ORDER MANAGER ---------------------------

class VAMOrderManager:
    """
    Bracket Order manager.
    - Entry   : type=1 (limit), limitPrice = LTP * 1.005, SL and TP from config.
    - Exit    : exit_order(id=entry_order_id) — cancels all BO legs atomically.
    - Flip    : close existing BO first (confirmed via API response), then open new BO.
    - Persist : state is saved to STATE_FILE (JSON) after every open/close so a restart
                can recover without losing track of an open position.
    - Restore : on startup, load from JSON first; if stale or missing, query Fyers live.
    """

    def __init__(self, fyers_client: fyersModel.FyersModel, config: TradingConfig):
        self.fyers  = fyers_client
        self.config = config

        self.has_position:   bool            = False
        self.option_symbol:  Optional[str]   = None
        self.direction:      Optional[str]   = None
        self.entry_price:    Optional[float] = None
        self.qty:            int             = self.config.NIFTY_LOT_SIZE
        self.entry_order_id: Optional[str]   = None

    # ------------------------------------------------------------------ #
    #  STATE PERSISTENCE                                                   #
    # ------------------------------------------------------------------ #

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
            logger.info(f"[VAM OM] No state file found at {STATE_FILE}.")
            return False

        try:
            with open(STATE_FILE, "r") as f:
                state = json.load(f)

            saved_at = datetime.fromisoformat(state.get("saved_at", ""))
            today    = datetime.now(IST).date()

            if saved_at.date() != today:
                logger.info(
                    f"[VAM OM] State file from {saved_at.date()}, today is {today}. "
                    f"Discarding stale state."
                )
                os.remove(STATE_FILE)
                return False

            if not state.get("has_position"):
                logger.info("[VAM OM] State file shows no open position.")
                return False

            self.has_position   = True
            self.option_symbol  = state.get("option_symbol")
            self.direction      = state.get("direction")
            self.entry_price    = state.get("entry_price")
            self.entry_order_id = state.get("entry_order_id")

            logger.info(
                f"[VAM OM] Restored from file: symbol={self.option_symbol}, "
                f"direction={self.direction}, entry_price={self.entry_price}, "
                f"order_id={self.entry_order_id}"
            )
            return True

        except Exception as e:
            logger.error(f"[VAM OM] Failed to load state file: {e}")
            return False

    def _clear_state_file(self) -> None:
        try:
            if os.path.exists(STATE_FILE):
                os.remove(STATE_FILE)
                logger.info(f"[VAM OM] State file {STATE_FILE} removed.")
        except Exception as e:
            logger.error(f"[VAM OM] Failed to remove state file: {e}")

    # ------------------------------------------------------------------ #
    #  BROKER STATE RESTORE                                                #
    # ------------------------------------------------------------------ #

    def _find_bo_order_id(self, symbol: str) -> Optional[str]:
        try:
            ob_resp = self.fyers.orderbook()
            if not isinstance(ob_resp, dict) or ob_resp.get("s") != "ok":
                logger.warning("[VAM OM] Could not fetch orderbook for order_id recovery.")
                return None

            for order in ob_resp.get("orderBook", []):
                if (
                    order.get("symbol")      == symbol
                    and order.get("productType") == "BO"
                    and order.get("side")        == 1
                    and order.get("status")      == 2
                ):
                    order_id = order.get("id")
                    logger.info(f"[VAM OM] Recovered order_id from orderbook: {order_id}")
                    return order_id

            logger.warning(f"[VAM OM] No matching BO order in orderbook for {symbol}.")
            return None

        except Exception as e:
            logger.error(f"[VAM OM] _find_bo_order_id error: {e}")
            return None

    def _restore_state_from_broker(self) -> bool:
        try:
            pos_resp = self.fyers.positions()   # V3: no data= argument
            logger.info(f"[VAM OM] Live positions on startup: {pos_resp}")

            if not isinstance(pos_resp, dict) or pos_resp.get("s") != "ok":
                logger.warning("[VAM OM] Could not fetch positions from broker.")
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

                order_id = self._find_bo_order_id(symbol)

                self.has_position   = True
                self.option_symbol  = symbol
                self.direction      = direction
                self.entry_price    = entry_price
                self.entry_order_id = order_id

                self._save_state()

                logger.info(
                    f"[VAM OM] Restored from broker: symbol={symbol}, "
                    f"direction={direction}, entry_price={entry_price:.2f}, "
                    f"order_id={order_id}"
                )
                return True

            logger.info("[VAM OM] No open positions at broker. Starting fresh.")
            return False

        except Exception as e:
            logger.error(f"[VAM OM] _restore_state_from_broker error: {e}")
            return False

    def restore_state_on_startup(self) -> None:
        logger.info("[VAM OM] Checking for existing position on startup...")
        restored = self._load_state_from_file()
        if not restored:
            self._restore_state_from_broker()

    # ------------------------------------------------------------------ #
    #  TRADING                                                             #
    # ------------------------------------------------------------------ #

    def _compute_bo_sl_tp_points(self, entry_price: float) -> tuple[float, float]:
        """
        SL and TP read from config — single source of truth shared with backtest.
        """
        sl_points = int(round(entry_price * self.config.STOP_LOSS_PCT   / 100.0))
        tp_points = int(round(entry_price * self.config.TAKE_PROFIT_PCT / 100.0))
        return float(sl_points), float(tp_points)

    def open_position(self, signal_row: pd.Series, opt_ltp: float) -> bool:
        symbol    = signal_row.get("option_symbol")
        direction = signal_row.get("direction", "BUY")

        if not symbol:
            logger.warning(f"[VAM OM] Missing option_symbol in signal: {dict(signal_row)}")
            return False

        if self.has_position:
            logger.warning("[VAM OM] Position already open, open_position() ignored.")
            return False

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
                f"[VAM OM] Placing BO: symbol={symbol}, dir={direction}, qty={self.qty}, "
                f"ltp={entry_price:.2f}, limitPrice={limit_price:.1f}, "
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
                logger.info(f"[VAM OM] BO placed successfully, order_id={self.entry_order_id}")
                return True
            else:
                logger.error(f"[VAM OM] BO rejected: {resp}")
                return False

        except Exception as e:
            logger.error(f"[VAM OM] place_order error: {e}")
            return False

    def close_position(self, exit_reason: str, exit_price: Optional[float] = None) -> bool:
        if not self.has_position or not self.option_symbol:
            logger.warning("[VAM OM] No open position, close_position() ignored.")
            return False

        if not self.entry_order_id:
            logger.warning("[VAM OM] No entry_order_id stored, cannot exit BO.")
            return False

        data = {"id": self.entry_order_id}

        try:
            logger.info(
                f"[VAM OM] Exiting BO. reason={exit_reason}, "
                f"order_id={self.entry_order_id}, symbol={self.option_symbol}"
            )
            resp = self.fyers.exit_order(data=data)
            logger.info(f"[VAM OM] exit_order response: {resp}")

            if isinstance(resp, dict) and resp.get("s") == "ok":
                self.has_position   = False
                self.option_symbol  = None
                self.direction      = None
                self.entry_price    = None
                self.entry_order_id = None
                self._clear_state_file()
                logger.info("[VAM OM] BO exited successfully, state cleared.")
                return True
            else:
                logger.error(f"[VAM OM] exit_order failed: {resp}")
                return False

        except Exception as e:
            logger.error(f"[VAM OM] exit_order error: {e}")
            return False


# --------------------------- TRADE LOGGER ---------------------------

class TradeLogger:
    """
    Records every live trade in the same format as the backtest output.
    - log_entry()            : called when a BO is successfully placed
    - log_flip_exit()        : called when position is closed via flip signal
    - resolve_open_trade_at_eod() : queries Fyers tradebook at 3:10 PM to find
                                    SL/TP hit exits managed by the broker
    - export_to_excel()      : exports all trades + summary to a dated .xlsx file
    """

    def __init__(self, fyers_client: fyersModel.FyersModel, config: "TradingConfig"):
        self.fyers   = fyers_client
        self.config  = config
        self.trades:  list             = []    # completed trade dicts
        self._open:   Optional[dict]   = None  # currently open trade

    # ------------------------------------------------------------------ #

    def log_entry(self, sig: pd.Series, entry_price: float, order_id: str) -> None:
        """Record trade entry when BO is successfully placed."""
        self._open = {
            "signal_time":    str(sig.get("timestamp", "")),
            "entry_time":     datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S"),
            "index_price":    sig.get("index_price"),
            "direction":      sig.get("direction"),
            "option_symbol":  sig.get("option_symbol"),
            "strategy_type":  sig.get("strategy_type", "vam_htf_live"),
            "htf_trend":      sig.get("htf_trend", ""),
            "entry_price":    round(float(entry_price), 2),
            "entry_order_id": order_id,
            "exit_time":      None,
            "exit_price":     None,
            "exit_reason":    None,
            "pnl_pct":        None,
            "pnl_rupees":     None,
            "result":         None,
        }
        logger.info(f"[TRADE LOG] Entry recorded: symbol={self._open['option_symbol']}, "
                    f"dir={self._open['direction']}, entry_price={self._open['entry_price']}")

    def log_flip_exit(self, exit_price: float) -> None:
        """Record trade exit when position is closed by an opposing signal (flip)."""
        if self._open is None:
            logger.warning("[TRADE LOG] log_flip_exit called but no open trade.")
            return
        self._finalize_trade(float(exit_price), "flip_exit")

    def _finalize_trade(self, exit_price: float, exit_reason: str,
                        exit_time: str = None) -> None:
        """Attach exit details, compute PnL, move to completed trades list."""
        if self._open is None:
            return
        trade = self._open.copy()
        trade["exit_time"]   = exit_time or datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S")
        trade["exit_price"]  = round(exit_price, 2)
        trade["exit_reason"] = exit_reason

        entry = trade.get("entry_price") or 0.0
        if entry > 0 and exit_price > 0:
            pnl_pct              = (exit_price - entry) / entry * 100.0
            trade["pnl_pct"]     = round(pnl_pct, 2)
            trade["pnl_rupees"]  = round(
                entry * self.config.NIFTY_LOT_SIZE * pnl_pct / 100.0, 2
            )
            trade["result"] = "profit" if pnl_pct > 0 else "loss"
        else:
            trade["pnl_pct"] = 0.0; trade["pnl_rupees"] = 0.0
            trade["result"]  = "unknown"

        self.trades.append(trade)
        self._open = None
        logger.info(
            f"[TRADE LOG] Trade finalized: {trade['option_symbol']} | "
            f"entry={trade['entry_price']} exit={trade['exit_price']} "
            f"reason={exit_reason} pnl=₹{trade['pnl_rupees']}"
        )

    # ------------------------------------------------------------------ #

    def resolve_open_trade_at_eod(self) -> None:
        """
        At EOD (3:10 PM), query Fyers tradebook to find the broker-managed
        SL/TP exit for any still-open trade.
        """
        if self._open is None:
            logger.info("[TRADE LOG] No open trade to resolve at EOD.")
            return

        opt_symbol = self._open["option_symbol"]
        logger.info(f"[TRADE LOG] Resolving EOD trade for {opt_symbol}...")

        try:
            resp = self.fyers.tradebook()
            if not isinstance(resp, dict) or resp.get("s") != "ok":
                logger.warning("[TRADE LOG] Tradebook fetch failed — marking as eod_unresolved.")
                self._finalize_trade(0.0, "eod_unresolved")
                return

            # BO exit fills appear as side=-1 (sell) trades for the same symbol
            sell_fills = [
                t for t in resp.get("tradebook", [])
                if t.get("symbol") == opt_symbol and t.get("side") == -1
            ]

            if not sell_fills:
                logger.warning(f"[TRADE LOG] No exit fill found in tradebook for {opt_symbol}.")
                self._finalize_trade(0.0, "eod_no_fill_found")
                return

            latest      = sell_fills[-1]
            exit_price  = float(latest.get("tradedPrice", 0))
            exit_time   = latest.get("orderDateTime",
                                     datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S"))

            # Infer exit reason by proximity to SL/TP price levels
            entry      = self._open["entry_price"]
            sl_price   = entry * (1 - self.config.STOP_LOSS_PCT   / 100.0)
            tp_price   = entry * (1 + self.config.TAKE_PROFIT_PCT / 100.0)
            exit_reason = (
                "stop_loss" if abs(exit_price - sl_price) < abs(exit_price - tp_price)
                else "target"
            )

            logger.info(
                f"[TRADE LOG] EOD exit resolved: price={exit_price:.2f}, "
                f"reason={exit_reason}"
            )
            self._finalize_trade(exit_price, exit_reason, exit_time)

        except Exception as e:
            logger.error(f"[TRADE LOG] EOD resolution error: {e}")
            self._finalize_trade(0.0, "eod_resolution_error")

    # ------------------------------------------------------------------ #

    def export_to_excel(self) -> Optional[str]:
        """Export all trades and a summary to a dated Excel file."""
        if not self.trades:
            logger.info("[TRADE LOG] No trades today — nothing to export.")
            return None

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

            # ── TRADES SHEET ──────────────────────────────────────────
            ws = wb.active
            ws.title = "Trades"

            header_fill = PatternFill("solid", fgColor="1F4E79")
            header_font = Font(color="FFFFFF", bold=True)
            green_fill  = PatternFill("solid", fgColor="C6EFCE")
            red_fill    = PatternFill("solid", fgColor="FFC7CE")
            grey_fill   = PatternFill("solid", fgColor="F2F2F2")

            cols = list(df.columns)
            for ci, col in enumerate(cols, 1):
                cell = ws.cell(row=1, column=ci,
                               value=col.replace("_", " ").title())
                cell.fill      = header_fill
                cell.font      = header_font
                cell.alignment = Alignment(horizontal="center")

            for ri, row in df.iterrows():
                result = row.get("result", "")
                fill   = (green_fill if result == "profit" else
                          red_fill   if result == "loss"   else grey_fill)
                for ci, col in enumerate(cols, 1):
                    val  = row[col]
                    cell = ws.cell(row=ri + 2, column=ci, value=val)
                    cell.fill      = fill
                    cell.alignment = Alignment(horizontal="center")

            # Auto-width
            for ci, col in enumerate(cols, 1):
                max_len = max(
                    len(str(col)),
                    *(len(str(df.iloc[r][col])) for r in range(len(df)))
                )
                ws.column_dimensions[get_column_letter(ci)].width = min(max_len + 4, 35)

            # ── SUMMARY SHEET ─────────────────────────────────────────
            ws2 = wb.create_sheet("Summary")
            ws2.column_dimensions["A"].width = 22
            ws2.column_dimensions["B"].width = 20

            total  = len(df)
            wins   = int((df["result"] == "profit").sum())
            losses = int((df["result"] == "loss").sum())
            net    = df["pnl_rupees"].sum()
            wr     = round(wins / total * 100, 1) if total > 0 else 0.0

            summary_rows = [
                ("Date",             datetime.now(IST).strftime("%d-%b-%Y")),
                ("Total Trades",     total),
                ("Wins",             wins),
                ("Losses",           losses),
                ("Win Rate (%)",     wr),
                ("Net PnL (Rs)",     round(net, 2)),
                ("Best Trade (Rs)",  round(float(df["pnl_rupees"].max()), 2)),
                ("Worst Trade (Rs)", round(float(df["pnl_rupees"].min()), 2)),
                ("Avg PnL (Rs)",     round(float(df["pnl_rupees"].mean()), 2)),
                ("Stop Loss %",      self.config.STOP_LOSS_PCT),
                ("Take Profit %",    self.config.TAKE_PROFIT_PCT),
                ("Lot Size",         self.config.NIFTY_LOT_SIZE),
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
            print(f"\n[TRADE LOG] Trade log saved: {xlsx_path}")
            return xlsx_path

        except Exception as e:
            logger.error(f"[TRADE LOG] Excel export failed: {e}")
            return None


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

        self.timeframe_mins  = int(self.bot.config.TIMEFRAME)
        self.heartbeat_secs  = int(self.timeframe_mins * 60 * 2)

        logger.info(
            f"[VAM LIVE] Timeframe={self.timeframe_mins}m, "
            f"heartbeat threshold={self.heartbeat_secs}s"
        )

        self.ws:           Optional[data_ws.FyersDataSocket] = None
        self.order_manager = VAMOrderManager(self.bot.fyers, self.bot.config)

        self.order_manager.restore_state_on_startup()

        self.pending_signal:   Optional[pd.Series] = None
        self.pending_bar_ts:   Optional[datetime]  = None
        self.option_ltp_cache: dict[str, float]    = {}

        # Guard against duplicate order attempts from rapid WS reconnects
        self._execution_lock:       threading.Lock     = threading.Lock()
        self._last_executed_bar_ts: Optional[datetime] = None

        # Trade logger — records every trade in backtest format, exports Excel at EOD
        self.trade_logger = TradeLogger(self.bot.fyers, self.bot.config)

    def _bar_key(self, ts: datetime) -> datetime:
        minute_bucket = (ts.minute // self.timeframe_mins) * self.timeframe_mins
        return ts.replace(second=0, microsecond=0, minute=minute_bucket)

    def _on_open(self):
        logger.info("[VAM LIVE] WS connected — subscribing to index symbol")
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
                v_dict.get("lp")
                or data.get("ltp")
                or data.get("price")
                or data.get("close")
            )
            if ltp is None or not symbol:
                return

            now      = datetime.now(IST)
            self.last_tick_time = now
            bar_time = self._bar_key(now)
            ltp      = float(ltp)
            vol      = float(v_dict.get("volume") or data.get("volume", 0) or 0)

            if symbol == self.index_symbol:
                self._handle_index_tick(bar_time, ltp, vol)

        except Exception as e:
            logger.error(f"[VAM LIVE] WS on_message error: {e} | raw: {msg}")

    def _handle_index_tick(self, bar_time: datetime, ltp: float, vol: float):
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
                "open":  ltp,
                "high":  ltp,
                "low":   ltp,
                "close": ltp,
                "volume": vol,
            }
        else:
            self.current_index_bar["high"]   = max(self.current_index_bar["high"], ltp)
            self.current_index_bar["low"]    = min(self.current_index_bar["low"],  ltp)
            self.current_index_bar["close"]  = ltp
            self.current_index_bar["volume"] += vol

    def _finalize_index_bar(self, bar: dict):
        bar_ts = bar["timestamp"]
        today  = datetime.now(IST).date()

        if bar_ts.date() != today:
            logger.info(f"[VAM LIVE] Skipping index bar from previous day: {bar_ts}")
            return

        if not (self.bot.config.SESSION_START <= bar_ts.time() <= self.bot.config.SESSION_END):
            logger.info(f"[VAM LIVE] Skipping index bar outside session: {bar_ts}")
            return

        logger.info(f"[VAM LIVE] Finalized index bar: {bar}")

        # Keep ~60 days of 1-min bars (375 bars/day × 60 days).
        # Must match HIST_DAYS so EMA/center initialisation is identical to backtest.
        self.index_candles = (
            pd.concat([self.index_candles, pd.DataFrame([bar])], ignore_index=True)
            .drop_duplicates("timestamp")
            .sort_values("timestamp")
            .tail(22500)
            .reset_index(drop=True)
        )

        df_for_signals = self.index_candles.copy()
        df_for_signals.set_index("timestamp", inplace=True)

        # generate_vam_signals_live resets and replays from scratch each call,
        # then leaves self.last_trade / self.position_size at the last signal
        # seen in history — identical behaviour to backtest forward loop.
        signals = self.bot.generate_vam_signals_live(df_for_signals)
        if signals.empty:
            logger.info("[VAM LIVE] No VAM signals on this completed index bar.")
            return

        if isinstance(signals["timestamp"].iloc[0], pd.Timestamp):
            if signals["timestamp"].dt.tz is None:
                signals["timestamp"] = signals["timestamp"].dt.tz_localize(IST)
            else:
                signals["timestamp"] = signals["timestamp"].dt.tz_convert(IST)

        # Floor to minute precision to guard against tz/microsecond mismatches
        bar_ts_floor = pd.Timestamp(bar_ts).floor("min")
        sig_on_bar = signals[signals["timestamp"].dt.floor("min") == bar_ts_floor]
        if sig_on_bar.empty:
            logger.info(f"[VAM LIVE] No VAM signal on completed bar {bar_ts}.")
            return

        last_sig            = sig_on_bar.iloc[-1]
        self.pending_signal = last_sig
        self.pending_bar_ts = bar_ts

        logger.info(
            f"[VAM LIVE] Stored pending signal from bar {bar_ts} "
            f"to execute at next bar open: {dict(last_sig)}"
        )

    def _execute_pending_signal_at_next_bar(self, bar_time: datetime):
        # Acquire lock — prevents duplicate execution from simultaneous WS reconnects
        if not self._execution_lock.acquire(blocking=False):
            logger.info("[VAM CONFIRM] Execution already in progress, skipping duplicate call.")
            return

        try:
            # Per-bar guard — if this bar was already executed, skip
            if self._last_executed_bar_ts == self.pending_bar_ts:
                logger.info(
                    f"[VAM CONFIRM] Signal for bar {self.pending_bar_ts} already executed, skipping."
                )
                self.pending_signal = None
                self.pending_bar_ts = None
                return

            sig        = self.pending_signal.copy()
            opt_symbol = sig["option_symbol"]
            direction  = sig["direction"]

            try:
                quote_resp = self.bot.fyers.quotes(data={"symbols": opt_symbol})
                opt_ltp    = None
                if isinstance(quote_resp, dict) and quote_resp.get("d"):
                    d0 = quote_resp["d"][0]
                    if isinstance(d0, dict):
                        opt_ltp = d0.get("v", {}).get("lp")

                if opt_ltp is None:
                    cached = self.option_ltp_cache.get(opt_symbol)
                    if cached is None:
                        logger.info(
                            f"[VAM CONFIRM] No LTP for {opt_symbol} from API or cache, skip entry."
                        )
                        self.pending_signal = None
                        self.pending_bar_ts = None
                        return
                    opt_ltp = float(cached)
                    logger.info(f"[VAM CONFIRM] Using cached LTP {opt_ltp:.2f} for {opt_symbol}.")
                else:
                    opt_ltp = float(opt_ltp)
                    self.option_ltp_cache[opt_symbol] = opt_ltp

                if self.order_manager.has_position:
                    if direction != self.order_manager.direction:
                        logger.info("[VAM LIVE] Opposite signal detected -> flipping position.")
                        closed = self.order_manager.close_position(
                            exit_reason="flip_exit", exit_price=opt_ltp
                        )
                        if not closed:
                            logger.error(
                                "[VAM LIVE] Flip failed — existing BO exit not confirmed. "
                                "Skipping new entry to avoid double position."
                            )
                            self.pending_signal = None
                            self.pending_bar_ts = None
                            return
                        # Log the flip exit
                        self.trade_logger.log_flip_exit(opt_ltp)

                if not self.order_manager.has_position:
                    ok = self.order_manager.open_position(sig, opt_ltp)
                    if ok:
                        logger.info(f"[VAM LIVE] New BO opened for {opt_symbol}")
                        # Mark this bar as executed so reconnect retries are blocked
                        self._last_executed_bar_ts = self.pending_bar_ts
                        # Log the entry
                        self.trade_logger.log_entry(
                            sig, opt_ltp, self.order_manager.entry_order_id
                        )

            except Exception as e:
                logger.warning(f"[VAM CONFIRM] Failed to execute pending signal: {e}")

            self.pending_signal = None
            self.pending_bar_ts = None

        finally:
            self._execution_lock.release()

    def _heartbeat_loop(self):
        while True:
            try:
                if self.last_tick_time is not None:
                    delta = (datetime.now(IST) - self.last_tick_time).total_seconds()
                    if delta > self.heartbeat_secs:
                        logger.warning(
                            f"[VAM LIVE] No ticks for {delta:.0f}s (>{self.heartbeat_secs}s). "
                            f"Reconnecting WS."
                        )
                        try:
                            if self.ws is not None:
                                try:
                                    self.ws.close()
                                except Exception:
                                    pass
                            self._create_and_connect_ws()
                        except Exception as e:
                            logger.error(f"[VAM LIVE] Reconnect failed: {e}")
                        self.last_tick_time = datetime.now(IST)
                time.sleep(5)
            except Exception as e:
                logger.error(f"[VAM LIVE] Heartbeat loop error: {e}")
                time.sleep(5)

    def _create_and_connect_ws(self):
        logger.info("[VAM LIVE] Creating new FyersDataSocket and connecting...")
        self.ws = data_ws.FyersDataSocket(
            access_token=self.access_token,
            log_path="",
            litemode=True,
            write_to_file=False,
            reconnect=True,
            on_connect=self._on_open,
            on_close=self._on_close,
            on_error=self._on_error,
            on_message=self._on_message,
        )
        self.ws.connect()

    def stop(self) -> None:
        """
        Graceful shutdown:
        1. Close WebSocket
        2. Resolve any open trade via Fyers tradebook
        3. Export trade log to Excel
        """
        logger.info("[VAM LIVE] stop() called — running EOD cleanup...")

        # 1. Close WebSocket
        if self.ws is not None:
            try:
                self.ws.close()
            except Exception as e:
                logger.error(f"[VAM LIVE] WS close error: {e}")

        # 2. Resolve open trade from broker tradebook
        self.trade_logger.resolve_open_trade_at_eod()

        # 3. Export to Excel
        self.trade_logger.export_to_excel()

        logger.info("[VAM LIVE] EOD cleanup complete.")

    def start(self):
        logger.info("Starting VAM WebSocket live engine with heartbeat watchdog.")
        try:
            self._create_and_connect_ws()
            hb_thread = threading.Thread(target=self._heartbeat_loop, daemon=True)
            hb_thread.start()
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            logger.info("KeyboardInterrupt received, stopping VAM engine.")
        finally:
            if self.ws is not None:
                try:
                    self.ws.close()
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