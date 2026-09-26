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
import time
import json
import urllib3

# ─── SSL FIX (same as backtest) ───────────────────────────────────────────────
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

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
        format="%(asctime)s - VAMPoll - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logger = logging.getLogger("VAMPoll")

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
# All values copied exactly from VAMBacktestBot (backtest file)

class TradingConfig:
    UNDERLYING_SYMBOL = "NSE:NIFTY50-INDEX"

    # ── Exactly as in backtest ──
    TIMEFRAME  = "10S"
    HIST_DAYS  = 3

    STRIKE_STEP    = 100
    NIFTY_LOT_SIZE = 65

    EXPIRY_TYPE = "MONTHLY"

    VAM_BUY_SCORE  =  0.10
    VAM_SELL_SCORE = -0.10

    # ── VAM ensemble params — unchanged from backtest ──
    LEN1 = 8;  LEN2 = 10;  LEN3 = 14;  LEN4 = 20;  LEN5 = 28
    MULT1 = 1.05; MULT2 = 1.10; MULT3 = 1.15; MULT4 = 1.20; MULT5 = 1.25
    BAND1 = 1.15; BAND2 = 1.20; BAND3 = 1.25; BAND4 = 1.30; BAND5 = 1.35
    SMOOTH1 = 2;  SMOOTH2 = 2;  SMOOTH3 = 3;  SMOOTH4 = 3;  SMOOTH5 = 4

    ATR_LEN       = 14
    ATR_LAYERMULT = 0.90
    TRAIL_SMOOTH  = 4

    # ── HTF — exactly as in backtest ──
    HTF_RULE    = "30min"
    HTF_EMA_LEN = 5

    SESSION_START = dtime(9, 20)
    SESSION_END   = dtime(15, 30)
    TRADE_START   = dtime(9, 30)
    TRADE_END     = dtime(15, 0)
    EOD_EXIT_TIME = dtime(15, 20)

    # ── SL/TP — exactly as in backtest ──
    STOP_LOSS_PCT   = 20.0   # -20% on option premium
    TAKE_PROFIT_PCT = 20.0   # +20% on option premium

    # ── Pivot — exactly as in backtest ──
    PIVOT_BARS          = 10
    PIVOT_PROXIMITY_PCT = 0.5

    # ── Poll interval: 5s (half a 10s bar) ──
    POLL_INTERVAL_SECS = 5

    # ── Signal confirmation: number of consecutive scans a signal must persist
    #    before an order is placed. LTP is captured on the FIRST appearance and
    #    used as the limit price when the order fires.
    #    2 scans × 5s = ~10s confirmation window.
    #    Poll 1: signal first seen on bar N, ltp captured.
    #    Poll 2: bar N+1 exists and signal on bar N still present → fire.
    #    This ensures the signal survived at least one new bar forming,
    #    filtering phantom signals that vanish as EMAs stabilise. ──
    SIGNAL_CONFIRM_BARS = 2

    # ── Dry-run mode: log all order activity but never send real orders ──
    #    Set to False only when ready for live trading.
    DRY_RUN = False


    def __init__(self):
        now      = datetime.now(IST)
        exp_type = self.EXPIRY_TYPE.upper()
        if exp_type == "WEEKLY":
            exp_dt = get_next_tuesday(now)
            kind   = "WEEKLY"
        else:
            exp_dt = get_current_or_next_monthly_expiry(now)
            kind   = "MONTHLY"
        self.EXPIRY_CODE = format_expiry_code(exp_dt)
        logger.info(f"Using NIFTY {kind} expiry {exp_dt.date()} → code {self.EXPIRY_CODE}")


# --------------------------- FYERS HTTP CLIENT (same as backtest) ---------------------------

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
            "cont_flag":   data.get("cont_flag", "1"),
        }
        r = self.session.get(f"{self.BASE_T2}/history", params=params)
        logger.debug(f"History HTTP {r.status_code} | raw(300): {r.text[:300]}")
        try:
            return r.json()
        except Exception as e:
            raise ValueError(
                f"History API non-JSON (status={r.status_code}): {r.text[:300]}"
            ) from e

    def quotes(self, symbol: str) -> dict:
        r = self.session.get(
            f"{self.BASE_T2}/quotes",
            params={"symbols": symbol},
        )
        try:
            return r.json()
        except Exception as e:
            raise ValueError(f"Quotes API non-JSON: {r.text[:200]}") from e

    def place_order(self, data: dict) -> dict:
        r = self.session.post(f"{self.BASE_T1}/orders/sync", json=data)
        try:
            return r.json()
        except Exception as e:
            raise ValueError(f"place_order non-JSON: {r.text[:200]}") from e

    def exit_order(self, data: dict) -> dict:
        # Kept for reference — use exit_position for filled BO positions.
        r = self.session.delete(f"{self.BASE_T1}/orders/sync/{data['id']}")
        try:
            return r.json()
        except Exception as e:
            raise ValueError(f"exit_order non-JSON: {r.text[:200]}") from e

    def exit_position(self, symbol: str, qty: int) -> dict:
        """Exit a filled BO position via the positions endpoint (market order).
        Equivalent to pressing 'Exit' on the position in the Fyers app.
        This executes the TP leg as a market order and cancels the SL leg.
        """
        data = {"id": symbol, "qty": qty, "type": 1}  # type=1 market order
        r = self.session.delete(f"{self.BASE_T1}/positions", json=data)
        logger.debug(f"exit_position HTTP {r.status_code} | raw: {r.text[:200]}")
        try:
            return r.json()
        except Exception as e:
            raise ValueError(f"exit_position non-JSON: {r.text[:200]}") from e

    def positions(self) -> dict:
        r = self.session.get(f"{self.BASE_T1}/positions")
        try:
            return r.json()
        except Exception as e:
            raise ValueError(f"positions non-JSON: {r.text[:200]}") from e

    def orderbook(self) -> dict:
        r = self.session.get(f"{self.BASE_T1}/orders")
        try:
            return r.json()
        except Exception as e:
            raise ValueError(f"orderbook non-JSON: {r.text[:200]}") from e

    def tradebook(self) -> dict:
        r = self.session.get(f"{self.BASE_T1}/trades")
        try:
            return r.json()
        except Exception as e:
            raise ValueError(f"tradebook non-JSON: {r.text[:200]}") from e


# --------------------------- VAM CORE (identical to backtest) ---------------------------

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
    params   = dict(
        lens  =[cfg.LEN1,    cfg.LEN2,    cfg.LEN3,    cfg.LEN4,    cfg.LEN5],
        mults =[cfg.MULT1,   cfg.MULT2,   cfg.MULT3,   cfg.MULT4,   cfg.MULT5],
        bands =[cfg.BAND1,   cfg.BAND2,   cfg.BAND3,   cfg.BAND4,   cfg.BAND5],
        smooth=[cfg.SMOOTH1, cfg.SMOOTH2, cfg.SMOOTH3, cfg.SMOOTH4, cfg.SMOOTH5],
    )
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


# --------------------------- PIVOT HELPER (identical to backtest) ---------------------------

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


# --------------------------- ORDER MANAGER ---------------------------

class VAMOrderManager:
    def __init__(self, fyers_client: FyersHTTP, config: TradingConfig):
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
            logger.info(f"[OM] State saved: {state}")
        except Exception as e:
            logger.error(f"[OM] Failed to save state: {e}")

    def _load_state_from_file(self) -> bool:
        if not os.path.exists(STATE_FILE):
            return False
        try:
            with open(STATE_FILE, "r") as f:
                state = json.load(f)
            saved_at = datetime.fromisoformat(state.get("saved_at", ""))
            today    = datetime.now(IST).date()
            if saved_at.date() != today:
                os.remove(STATE_FILE)
                return False
            if not state.get("has_position"):
                return False
            self.has_position   = True
            self.option_symbol  = state.get("option_symbol")
            self.direction      = state.get("direction")
            self.entry_price    = state.get("entry_price")
            self.entry_order_id = state.get("entry_order_id")
            logger.info(
                f"[OM] Restored from file: symbol={self.option_symbol}, "
                f"direction={self.direction}, entry_price={self.entry_price}"
            )
            return True
        except Exception as e:
            logger.error(f"[OM] Failed to load state file: {e}")
            return False

    def _clear_state_file(self) -> None:
        try:
            if os.path.exists(STATE_FILE):
                os.remove(STATE_FILE)
        except Exception as e:
            logger.error(f"[OM] Failed to remove state file: {e}")

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
            logger.error(f"[OM] _find_bo_order_id error: {e}")
            return None

    def _restore_state_from_broker(self) -> bool:
        try:
            pos_resp = self.fyers.positions()
            logger.info(f"[OM] Live positions on startup: {pos_resp}")
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
                    f"[OM] Restored from broker: symbol={symbol}, "
                    f"direction={direction}, entry_price={entry_price:.2f}"
                )
                return True
            logger.info("[OM] No open positions at broker. Starting fresh.")
            return False
        except Exception as e:
            logger.error(f"[OM] _restore_state_from_broker error: {e}")
            return False

    def restore_state_on_startup(self) -> None:
        logger.info("[OM] Checking for existing position on startup...")
        if not self._load_state_from_file():
            self._restore_state_from_broker()

    def _compute_bo_sl_tp_points(self, entry_price: float) -> tuple:
        sl_points = int(round(entry_price * self.config.STOP_LOSS_PCT   / 100.0))
        tp_points = int(round(entry_price * self.config.TAKE_PROFIT_PCT / 100.0))
        return float(sl_points), float(tp_points)

    def open_position(self, signal: dict, opt_ltp: float) -> bool:
        symbol    = signal.get("option_symbol")
        direction = signal.get("direction", "BUY")
        if not symbol:
            logger.warning("[OM] Missing option_symbol in signal")
            return False
        if self.has_position:
            logger.warning("[OM] Position already open, open_position() ignored.")
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
                f"[OM] {'[DRY-RUN] ' if self.config.DRY_RUN else ''}Placing BO: "
                f"symbol={symbol}, dir={direction}, qty={self.qty}, "
                f"ltp={entry_price:.2f}, limitPrice={limit_price:.1f}, "
                f"SL_pts={sl_points:.0f}, TP_pts={tp_points:.0f}"
            )
            if self.config.DRY_RUN:
                # Simulate a successful fill without touching the broker.
                self.has_position   = True
                self.option_symbol  = symbol
                self.direction      = direction
                self.entry_price    = entry_price
                self.entry_order_id = f"DRY-{datetime.now(IST).strftime('%H%M%S%f')}"
                self._save_state()
                logger.info(f"[OM] [DRY-RUN] Simulated fill, fake order_id={self.entry_order_id}")
                return True

            resp = self.fyers.place_order(data=data)
            logger.info(f"[OM] place_order response: {resp}")
            if isinstance(resp, dict) and resp.get("s") == "ok":
                self.has_position   = True
                self.option_symbol  = symbol
                self.direction      = direction
                self.entry_price    = entry_price
                self.entry_order_id = resp.get("id")
                self._save_state()
                logger.info(f"[OM] BO placed successfully, order_id={self.entry_order_id}")
                return True
            else:
                logger.error(f"[OM] BO rejected: {resp}")
                return False
        except Exception as e:
            logger.error(f"[OM] place_order error: {e}")
            return False

    def sync_with_broker(self) -> bool:
        """
        Verify our position state against live broker positions.
        If we think we have a position but the broker shows netQty=0 for that
        symbol (SL or TP was auto-triggered), clear local state and return True
        so the caller can log the exit and resume signal scanning.
        Returns False if nothing changed.
        """
        if not self.has_position or not self.option_symbol:
            return False
        try:
            pos_resp = self.fyers.positions()
            if not isinstance(pos_resp, dict) or pos_resp.get("s") != "ok":
                logger.warning(f"[OM] sync_with_broker: positions API error: {pos_resp}")
                return False
            for pos in pos_resp.get("netPositions", []):
                if pos.get("symbol") == self.option_symbol:
                    net_qty = int(pos.get("netQty", 0))
                    if net_qty != 0:
                        # Position still open at broker — all good.
                        return False
                    # netQty == 0 → broker closed it (SL or TP hit)
                    logger.info(
                        f"[OM] Broker-side exit detected for {self.option_symbol} "
                        f"(netQty=0). Clearing local state."
                    )
                    self.has_position   = False
                    self.option_symbol  = None
                    self.direction      = None
                    self.entry_price    = None
                    self.entry_order_id = None
                    self._clear_state_file()
                    return True
            # Symbol not in positions at all → also means it's closed.
            logger.info(
                f"[OM] {self.option_symbol} not found in broker positions — "
                "assuming auto-closed. Clearing local state."
            )
            self.has_position   = False
            self.option_symbol  = None
            self.direction      = None
            self.entry_price    = None
            self.entry_order_id = None
            self._clear_state_file()
            return True
        except Exception as e:
            logger.error(f"[OM] sync_with_broker error: {e}")
            return False

    def close_position(self, exit_reason: str, exit_price: Optional[float] = None) -> bool:
        if not self.has_position or not self.option_symbol:
            logger.warning("[OM] No open position, close_position() ignored.")
            return False
        if not self.entry_order_id:
            logger.warning("[OM] No entry_order_id stored, cannot exit BO.")
            return False
        try:
            logger.info(
                f"[OM] {'[DRY-RUN] ' if self.config.DRY_RUN else ''}Exiting BO. "
                f"reason={exit_reason}, order_id={self.entry_order_id}, symbol={self.option_symbol}"
            )
            if self.config.DRY_RUN:
                self.has_position   = False
                self.option_symbol  = None
                self.direction      = None
                self.entry_price    = None
                self.entry_order_id = None
                self._clear_state_file()
                logger.info("[OM] [DRY-RUN] Simulated exit, state cleared.")
                return True

            resp = self.fyers.exit_position(
                symbol=self.option_symbol, qty=self.qty
            )
            logger.info(f"[OM] exit_position response: {resp}")
            if isinstance(resp, dict) and resp.get("s") == "ok":
                self.has_position   = False
                self.option_symbol  = None
                self.direction      = None
                self.entry_price    = None
                self.entry_order_id = None
                self._clear_state_file()
                logger.info("[OM] BO exited successfully via positions endpoint, state cleared.")
                return True
            else:
                logger.error(f"[OM] exit_position failed: {resp}")
                return False
        except Exception as e:
            logger.error(f"[OM] exit_position error: {e}")
            return False


# --------------------------- TRADE LOGGER ---------------------------

class TradeLogger:
    def __init__(self, fyers_client: FyersHTTP, config: TradingConfig):
        self.fyers  = fyers_client
        self.config = config
        self.trades: list           = []
        self._open:  Optional[dict] = None

    def log_entry(self, sig: dict, entry_price: float, order_id: str) -> None:
        self._open = {
            "signal_time":    str(sig.get("timestamp", "")),
            "entry_time":     datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S"),
            "index_price":    sig.get("index_price"),
            "direction":      sig.get("direction"),
            "option_symbol":  sig.get("option_symbol"),
            "strategy_type":  sig.get("strategy_type", "vam_poll"),
            "htf_trend":      sig.get("htf_trend", ""),
            "pivot_level":    sig.get("pivot_level"),
            "entry_method":   sig.get("entry_method", ""),
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
            f"[TL] Entry: symbol={self._open['option_symbol']}, "
            f"dir={self._open['direction']}, price={self._open['entry_price']}, "
            f"pivot={self._open['pivot_level']}, method={self._open['entry_method']}"
        )

    def log_flip_exit(self, exit_price: float) -> None:
        if self._open is None:
            logger.warning("[TL] log_flip_exit called but no open trade.")
            return
        self._finalize_trade(float(exit_price), "flip_exit")

    def log_sl_tp_exit(self, exit_price: float, reason: str) -> None:
        """Used by dry-run SL/TP simulation to record exits with a known price."""
        if self._open is None:
            logger.warning("[TL] log_sl_tp_exit called but no open trade.")
            return
        self._finalize_trade(float(exit_price), reason)

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
            f"[TL] Finalized: {trade['option_symbol']} | "
            f"entry={trade['entry_price']} exit={trade['exit_price']} "
            f"reason={exit_reason} pnl=₹{trade['pnl_rupees']}"
        )

    def resolve_open_trade_at_eod(self) -> None:
        if self._open is None:
            logger.info("[TL] No open trade to resolve at EOD.")
            return
        opt_symbol = self._open["option_symbol"]
        logger.info(f"[TL] Resolving EOD trade for {opt_symbol}...")
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
            logger.error(f"[TL] EOD resolution error: {e}")
            self._finalize_trade(0.0, "eod_resolution_error")

    def export_to_excel(self) -> Optional[str]:
        if not self.trades:
            logger.info("[TL] No trades today — nothing to export.")
            return None
        try:
            import openpyxl
            from openpyxl.styles import PatternFill, Font, Alignment
            from openpyxl.utils import get_column_letter

            df = pd.DataFrame(self.trades)
            df["pnl_pct"]        = pd.to_numeric(df["pnl_pct"],    errors="coerce").fillna(0)
            df["pnl_rupees"]     = pd.to_numeric(df["pnl_rupees"], errors="coerce").fillna(0)
            df["cum_pnl_rupees"] = df["pnl_rupees"].cumsum().round(2)

            today     = datetime.now(IST).strftime("%Y%m%d")
            log_dir   = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "logs")
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
                fill   = green_fill if result == "profit" else red_fill if result == "loss" else grey_fill
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
                ("Win Rate (%)",     wr),    ("Net PnL (Rs)", round(net, 2)),
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
            logger.info(f"[TL] Excel exported → {xlsx_path}")
            return xlsx_path
        except Exception as e:
            logger.error(f"[TL] Excel export failed: {e}")
            return None


# --------------------------- HISTORY CACHE ---------------------------
# Avoids fetching 1 day of 10s data (≈ 2,200 bars) on every 5s tick.
# Warmup fetched once at startup; only today's bars are refreshed each tick.

class HistoryCache:
    def __init__(self, fyers_client: FyersHTTP, config: TradingConfig):
        self.fyers      = fyers_client
        self.config     = config
        self._cache:    Optional[pd.DataFrame] = None
        self._cache_day: Optional[object]      = None   # date

    def _fetch_raw(self, days: int) -> pd.DataFrame:
        end_dt   = datetime.now(IST)
        start_dt = end_dt - timedelta(days=days)
        data = dict(
            symbol     = self.config.UNDERLYING_SYMBOL,
            resolution = self.config.TIMEFRAME,
            date_format= "1",
            range_from = start_dt.strftime("%Y-%m-%d"),
            range_to   = end_dt.strftime("%Y-%m-%d"),
            cont_flag  = "0",
        )
        resp = self.fyers.history(data=data)
        if not isinstance(resp, dict) or resp.get("s") != "ok":
            logger.warning(f"[CACHE] History API error: {resp}")
            return pd.DataFrame()
        candles = resp.get("candles", [])
        if not candles:
            return pd.DataFrame()
        df = pd.DataFrame(candles, columns=["timestamp","open","high","low","close","volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s", utc=True).dt.tz_convert(IST)
        df = df.sort_values("timestamp").reset_index(drop=True)
        df.set_index("timestamp", inplace=True)
        # session filter
        mask = df.index.indexer_between_time(
            self.config.SESSION_START, self.config.SESSION_END,
            include_start=True, include_end=True,
        )
        return df.iloc[mask]

    def get(self) -> pd.DataFrame:
        today = datetime.now(IST).date()

        # ── Full warmup fetch — once per calendar day ──
        if self._cache is None or self._cache_day != today:
            logger.info("[CACHE] Fetching 1-day warmup history...")
            self._cache     = self._fetch_raw(self.config.HIST_DAYS)
            self._cache_day = today
            logger.info(f"[CACHE] Warmup complete: {len(self._cache)} bars")
            return self._cache

        # ── Incremental: fetch only today, merge with cached history ──
        today_fresh = self._fetch_raw(days=1)
        if today_fresh.empty:
            return self._cache

        # Drop any stale today rows from cache, append fresh today rows
        cache_excl_today = self._cache[
            pd.to_datetime(self._cache.index).date != today
        ]
        merged = pd.concat([cache_excl_today, today_fresh])
        merged = merged[~merged.index.duplicated(keep="last")]
        merged.sort_index(inplace=True)
        self._cache = merged
        return self._cache


# --------------------------- POLLING ENGINE ---------------------------

class VAMPollingEngine:
    """
    Every POLL_INTERVAL_SECS (5s):
      1. Get fresh index data from HistoryCache (full fetch once/day, incremental after)
      2. Run VAM + HTF + pivot scan — identical logic to backtest
      3. Check last signal on today's bars; skip if already fired
      4. Apply pivot proximity logic (enter now OR wait for pullback)
      5. EOD forced exit at 15:20

    Key differences vs 1-min bot:
      - Uses FyersHTTP (requests) instead of fyersModel — same as backtest
      - TIMEFRAME = "10S", HTF_RULE = "10min", TAKE_PROFIT_PCT = 20
      - sig_ts floored to "10s" not "min" for dedup
      - HistoryCache prevents multi-day fetch on every 5s tick
    """

    def __init__(self):
        self.config = TradingConfig()
        self.fyers  = self._init_fyers()

        self.history_cache = HistoryCache(self.fyers, self.config)

        self.order_manager = VAMOrderManager(self.fyers, self.config)
        self.order_manager.restore_state_on_startup()

        self.trade_logger = TradeLogger(self.fyers, self.config)

        # dedup: (bar_ts_floor_10s, direction) tuples fired today
        self._fired_signals: set = set()

        # pivot-wait state
        self._pivot_pending:    Optional[dict]     = None
        self._pivot_pending_ts: Optional[datetime] = None

        # confirmation-delay state: tracks the signal currently accumulating
        # confirmations before an order is placed.
        # Structure: {"key": dedup_key, "sig": dict, "first_ltp": float, "count": int}
        self._confirm_pending: Optional[dict] = None

        if self.config.DRY_RUN:
            logger.warning(
                "[POLL] *** DRY-RUN MODE ENABLED — no real orders will be placed ***"
            )
        logger.info(
            f"[POLL] Engine ready. Interval={self.config.POLL_INTERVAL_SECS}s, "
            f"TF={self.config.TIMEFRAME}, HTF={self.config.HTF_RULE}, "
            f"SL={self.config.STOP_LOSS_PCT}% TP={self.config.TAKE_PROFIT_PCT}%"
        )

    # ── init ──────────────────────────────────────────────────────────────────

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
            raise ConnectionError(f"Fyers auth failed: {profile}")
        logger.info(f"[POLL] Fyers connected — {profile['data']['name']} ✅")
        return client

    # ── option LTP ─────────────────────────────────────────────────────────────

    def _get_option_ltp(self, symbol: str) -> Optional[float]:
        try:
            resp = self.fyers.quotes(symbol)
            if isinstance(resp, dict) and resp.get("d"):
                d0 = resp["d"][0]
                if isinstance(d0, dict):
                    ltp = d0.get("v", {}).get("lp")
                    if ltp is not None:
                        return float(ltp)
            logger.warning(f"[POLL] Could not get LTP for {symbol}: {resp}")
            return None
        except Exception as e:
            logger.error(f"[POLL] _get_option_ltp error: {e}")
            return None

    # ── option symbol ──────────────────────────────────────────────────────────

    def _pick_itm_option_symbol(self, underlying_price: float, direction: str) -> str:
        c    = self.config
        spot = float(underlying_price)
        step = c.STRIKE_STEP
        if direction == "BUY":
            strike   = (int(spot) // step) * step - step
            opt_type = "CE"
        else:
            strike   = ((int(spot) + step - 1) // step) * step + step
            opt_type = "PE"
        strike = int(max(step, (int(strike) // step) * step))
        return f"NSE:NIFTY{c.EXPIRY_CODE}{strike:05d}{opt_type}"

    # ── HTF trend ──────────────────────────────────────────────────────────────

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

    # ── core signal scan — identical logic to backtest ─────────────────────────

    def _run_backtest_scan(self, df_idx: pd.DataFrame) -> list:
        df_idx_r = df_idx.copy().reset_index()
        df_vam   = compute_vam_ensemble(df_idx_r, self.config)
        df_htf   = self._build_htf_trend(df_idx.copy())

        df_merged = pd.merge_asof(
            df_vam.sort_values("timestamp"),
            df_htf[["timestamp","htf_trend"]].sort_values("timestamp"),
            on="timestamp", direction="backward",
        )

        warmup = max(
            self.config.LEN1, self.config.LEN2, self.config.LEN3,
            self.config.LEN4, self.config.LEN5, 50,
        )

        signals       = []
        position_size = 0
        last_trade    = None

        for i in range(warmup, len(df_merged)):
            row      = df_merged.iloc[i]
            ts       = row["timestamp"]
            ts_time  = ts.time()
            htf      = row.get("htf_trend", "flat")
            buy_sig  = bool(row["vam_buy_sig"])
            sell_sig = bool(row["vam_sell_sig"])

            in_window = self.config.TRADE_START <= ts_time <= self.config.TRADE_END

            ok_long  = (position_size <= 0) and (last_trade != "long")
            ok_short = (position_size >= 0) and (last_trade != "short")

            if not in_window:
                continue

            if buy_sig and ok_long and htf == "up":
                pivot_lvl = compute_pivot_zone(df_merged, i, self.config.PIVOT_BARS, "BUY")
                signals.append({
                    "timestamp":     ts,
                    "bar_pos":       i,
                    "index_price":   float(row["close"]),
                    "direction":     "BUY",
                    "option_symbol": self._pick_itm_option_symbol(float(row["close"]), "BUY"),
                    "strategy_type": "vam_poll",
                    "htf_trend":     htf,
                    "pivot_level":   pivot_lvl,
                })
                position_size = 1
                last_trade    = "long"

            elif sell_sig and ok_short and htf == "down":
                pivot_lvl = compute_pivot_zone(df_merged, i, self.config.PIVOT_BARS, "SELL")
                signals.append({
                    "timestamp":     ts,
                    "bar_pos":       i,
                    "index_price":   float(row["close"]),
                    "direction":     "SELL",
                    "option_symbol": self._pick_itm_option_symbol(float(row["close"]), "SELL"),
                    "strategy_type": "vam_poll",
                    "htf_trend":     htf,
                    "pivot_level":   pivot_lvl,
                })
                position_size = -1
                last_trade    = "short"

        logger.info(f"[POLL] Scan complete — {len(signals)} total signals")
        return signals

    # ── pivot proximity check ──────────────────────────────────────────────────

    def _is_near_pivot(self, current_price: float, pivot_lvl: float, direction: str) -> bool:
        prox_band = pivot_lvl * (self.config.PIVOT_PROXIMITY_PCT / 100.0)
        if direction == "BUY":
            return (current_price - pivot_lvl) <= prox_band
        else:
            return (pivot_lvl - current_price) <= prox_band

    # ── execute a signal (open or flip) ───────────────────────────────────────

    def _execute_signal(self, sig: dict, opt_ltp: float, entry_method: str) -> None:
        direction = sig["direction"]

        if self.order_manager.has_position:
            if direction != self.order_manager.direction:
                logger.info(f"[POLL] Opposite signal → flipping position.")
                # Fetch the EXISTING option's LTP for accurate flip exit PnL.
                existing_symbol = self.order_manager.option_symbol
                existing_ltp = self._get_option_ltp(existing_symbol) if existing_symbol else None
                if existing_ltp is None:
                    logger.warning("[POLL] Could not fetch existing option LTP for flip exit — falling back to new option LTP.")
                    existing_ltp = opt_ltp
                closed = self.order_manager.close_position("flip_exit", existing_ltp)
                if not closed:
                    logger.error("[POLL] Flip close failed — skipping new entry.")
                    return
                self.trade_logger.log_flip_exit(existing_ltp)
            else:
                logger.info("[POLL] Same-direction signal, already in position — skipping.")
                return

        sig["entry_method"] = entry_method
        ok = self.order_manager.open_position(sig, opt_ltp)
        if ok:
            self.trade_logger.log_entry(sig, opt_ltp, self.order_manager.entry_order_id)
            logger.info(f"[POLL] ✅ BO placed: {sig['option_symbol']} {direction} ({entry_method})")

    # ── EOD forced exit ────────────────────────────────────────────────────────

    def _check_dryrun_sl_tp(self) -> bool:
        """
        Dry-run only: fetch current option LTP and simulate SL/TP exit if hit.
        Returns True if an exit was triggered (caller should skip further processing).
        """
        if not self.config.DRY_RUN or not self.order_manager.has_position:
            return False
        symbol      = self.order_manager.option_symbol
        entry_price = self.order_manager.entry_price
        if not symbol or not entry_price:
            return False

        current_ltp = self._get_option_ltp(symbol)
        if current_ltp is None:
            return False

        sl_price = entry_price * (1 - self.config.STOP_LOSS_PCT   / 100.0)
        tp_price = entry_price * (1 + self.config.TAKE_PROFIT_PCT / 100.0)

        if current_ltp <= sl_price:
            logger.info(
                f"[DRY-RUN] SL hit: {symbol} ltp={current_ltp:.2f} "
                f"<= sl={sl_price:.2f} (entry={entry_price:.2f})"
            )
            self.order_manager.close_position("stop_loss")
            self.trade_logger.log_sl_tp_exit(current_ltp, "stop_loss")
            return True

        if current_ltp >= tp_price:
            logger.info(
                f"[DRY-RUN] TP hit: {symbol} ltp={current_ltp:.2f} "
                f">= tp={tp_price:.2f} (entry={entry_price:.2f})"
            )
            self.order_manager.close_position("target")
            self.trade_logger.log_sl_tp_exit(tp_price, "target")  # cap at limit price; real BO fills exactly at TP
            return True

        logger.debug(
            f"[DRY-RUN] Position live: {symbol} ltp={current_ltp:.2f} "
            f"entry={entry_price:.2f} sl={sl_price:.2f} tp={tp_price:.2f}"
        )
        return False

    def _eod_exit_if_needed(self) -> bool:
        now = datetime.now(IST)
        eod = IST.localize(datetime.combine(now.date(), self.config.EOD_EXIT_TIME))
        if now >= eod and self.order_manager.has_position:
            logger.info("[POLL] EOD reached — force-closing position.")
            if self.config.DRY_RUN:
                # Capture symbol BEFORE close_position clears it.
                symbol  = self.order_manager.option_symbol or (
                    self.trade_logger._open or {}
                ).get("option_symbol")
                self.order_manager.close_position("eod_exit")
                eod_ltp = self._get_option_ltp(symbol) if symbol else None
                if eod_ltp is not None:
                    logger.info(f"[DRY-RUN] EOD exit at live ltp={eod_ltp:.2f}")
                    self.trade_logger.log_sl_tp_exit(eod_ltp, "eod_exit")
                else:
                    logger.warning("[DRY-RUN] Could not fetch EOD LTP — logging exit at 0.")
                    self.trade_logger.log_sl_tp_exit(0.0, "eod_exit")
            else:
                self.order_manager.close_position("eod_exit")
                self.trade_logger.resolve_open_trade_at_eod()
            return True
        return False

    # ── single poll tick ──────────────────────────────────────────────────────

    def _poll_once(self) -> None:
        now = datetime.now(IST)

        if self._eod_exit_if_needed():
            return

        if not (self.config.SESSION_START <= now.time() <= self.config.SESSION_END):
            logger.info(f"[POLL] Outside session ({now.strftime('%H:%M:%S')} IST), next check in 5s.")
            return

        # ── 1. Position sync / SL-TP simulation ───────────────────────────────
        if self.config.DRY_RUN:
            # Simulate SL/TP exits by checking live option LTP each poll.
            if self._check_dryrun_sl_tp():
                return   # position just closed — skip signal scan this tick
        else:
            # Live: detect broker-side auto-exits (SL or TP hit).
            if self.order_manager.has_position:
                was_closed = self.order_manager.sync_with_broker()
                if was_closed:
                    logger.info("[POLL] Position was auto-closed by broker (SL/TP). Resolving trade log.")
                    self.trade_logger.resolve_open_trade_at_eod()

        # ── 2. Get fresh data (cached warmup + incremental today) ──
        df_idx = self.history_cache.get()
        if df_idx is None or df_idx.empty:
            logger.warning("[POLL] Empty dataframe — skipping this tick.")
            return

        today = now.date()

        # ── 3. Pivot-wait: check if price has now reached pivot ──
        if self._pivot_pending is not None:
            eod_boundary = IST.localize(datetime.combine(today, self.config.EOD_EXIT_TIME))
            if now >= eod_boundary:
                logger.info("[POLL] EOD — abandoning pivot wait and any pending confirmation.")
                self._pivot_pending    = None
                self._pivot_pending_ts = None
                self._confirm_pending  = None
            else:
                sig       = self._pivot_pending
                direction = sig["direction"]
                pivot_lvl = sig.get("pivot_level")

                if pivot_lvl is None:
                    self._pivot_pending = None
                else:
                    last_bar    = df_idx.iloc[-1]
                    check_price = float(last_bar["low"]) if direction == "BUY" else float(last_bar["high"])
                    prox_band   = pivot_lvl * (self.config.PIVOT_PROXIMITY_PCT / 100.0)
                    logger.info(
                        f"[POLL] Pivot wait: dir={direction}, pivot={pivot_lvl:.2f}, "
                        f"check={check_price:.2f}"
                    )
                    if abs(check_price - pivot_lvl) <= prox_band:
                        # Use the confirmed_ltp captured at signal-first-sight if
                        # available; otherwise fetch a fresh quote as fallback.
                        opt_symbol    = sig["option_symbol"]
                        confirmed_ltp = sig.get("confirmed_ltp")
                        if confirmed_ltp is not None:
                            logger.info(
                                f"[POLL] Price reached pivot {pivot_lvl:.2f} — "
                                f"executing entry at confirmed_ltp={confirmed_ltp:.2f}."
                            )
                            self._execute_signal(sig, confirmed_ltp, "pulled_to_pivot")
                        else:
                            opt_ltp = self._get_option_ltp(opt_symbol)
                            if opt_ltp is not None:
                                logger.info(
                                    f"[POLL] Price reached pivot {pivot_lvl:.2f} — "
                                    f"executing entry at fresh ltp={opt_ltp:.2f}."
                                )
                                self._execute_signal(sig, opt_ltp, "pulled_to_pivot")
                            else:
                                logger.warning(f"[POLL] No LTP for {opt_symbol}, skipping entry.")
                        self._pivot_pending    = None
                        self._pivot_pending_ts = None
            return

        # ── 4. Run full backtest scan ──
        all_signals = self._run_backtest_scan(df_idx)
        if not all_signals:
            return

        # ── 5. Filter to today's signals only ──
        today_signals = [
            s for s in all_signals
            if pd.Timestamp(s["timestamp"]).date() == today
        ]
        if not today_signals:
            logger.debug("[POLL] No signals on today's bars.")
            return

        # ── 6. Find the FIRST unfired signal in sequence (not just the last) ──
        # Processing in order ensures intermediate SELL signals are not skipped.
        # e.g. scan produces BUY(11:22)→SELL(11:28)→BUY(11:30):
        #   tick 1: fires BUY(11:22), marks it fired
        #   tick 2: BUY(11:22) already fired → fires SELL(11:28)
        #   tick 3: fires BUY(11:30)
        next_sig = None
        for s in today_signals:
            key = (pd.Timestamp(s["timestamp"]).floor("10s"), s["direction"])
            if key not in self._fired_signals:
                next_sig = s
                break

        if next_sig is None:
            logger.debug("[POLL] All today's signals already fired.")
            return

        last_sig  = next_sig
        sig_ts    = pd.Timestamp(last_sig["timestamp"]).floor("10s")
        sig_dir   = last_sig["direction"]
        dedup_key = (sig_ts, sig_dir)

        logger.info(
            f"[POLL] 🔔 Signal detected: dir={sig_dir}, ts={sig_ts}, "
            f"index={last_sig['index_price']:.2f}, pivot={last_sig.get('pivot_level')}, "
            f"option={last_sig['option_symbol']}"
        )

        # ── 6. Confirmation delay ──────────────────────────────────────────────
        # The signal must persist for SIGNAL_CONFIRM_BARS consecutive polls.
        # Option LTP is captured on the FIRST sighting and used as the limit
        # price when the order eventually fires (so we fill at the original price,
        # not the current price after confirmation completes).
        n_required = self.config.SIGNAL_CONFIRM_BARS

        if self._confirm_pending is None or self._confirm_pending["key"] != dedup_key:
            # New signal (or different from the one we were tracking) — start fresh.
            first_ltp = self._get_option_ltp(last_sig["option_symbol"])
            if first_ltp is None:
                logger.warning(
                    f"[POLL] No LTP for {last_sig['option_symbol']} on first sighting "
                    "— will retry next tick."
                )
                return
            self._confirm_pending = {
                "key":       dedup_key,
                "sig":       last_sig,
                "first_ltp": first_ltp,
                "count":     1,
            }
            logger.info(
                f"[POLL] Confirmation 1/{n_required}: {sig_dir} @ {sig_ts}, "
                f"first_ltp={first_ltp:.2f} — waiting for {n_required - 1} more poll(s)."
            )
            return
        else:
            # Same signal still present — increment counter.
            self._confirm_pending["count"] += 1
            count = self._confirm_pending["count"]
            logger.info(
                f"[POLL] Confirmation {count}/{n_required}: {sig_dir} @ {sig_ts}"
            )
            if count < n_required:
                return
            # Fully confirmed — extract stored LTP and clear pending state.
            confirmed_ltp = self._confirm_pending["first_ltp"]
            self._confirm_pending = None
            logger.info(
                f"[POLL] ✅ Signal confirmed ({n_required}/{n_required}) — "
                f"proceeding with order at original ltp={confirmed_ltp:.2f}"
            )

        # ── 7. Skip if no pivot found (same as backtest) ──
        pivot_lvl = last_sig.get("pivot_level")
        if pivot_lvl is None:
            logger.info("[POLL] No pivot found — skipping signal.")
            self._fired_signals.add(dedup_key)   # mark fired so we don't retry
            return

        # ── 8. Pivot proximity logic (Case A / Case B) ──
        current_price = float(df_idx.iloc[-1]["close"])

        if self._is_near_pivot(current_price, pivot_lvl, sig_dir):
            # Case A: already near pivot → enter immediately at confirmed ltp
            logger.info(
                f"[POLL] Price {current_price:.2f} already near pivot "
                f"{pivot_lvl:.2f} — entering now at confirmed_ltp={confirmed_ltp:.2f}."
            )
            self._fired_signals.add(dedup_key)
            self._execute_signal(last_sig, confirmed_ltp, "already_near_pivot")
        else:
            # Case B: wait for pullback to pivot zone.
            # Store confirmed_ltp in the signal dict so it's used when pivot is hit.
            logger.info(
                f"[POLL] Price {current_price:.2f} not yet near pivot "
                f"{pivot_lvl:.2f} — entering pivot-wait mode "
                f"(will fill at confirmed_ltp={confirmed_ltp:.2f})."
            )
            last_sig["confirmed_ltp"] = confirmed_ltp   # carried into pivot-wait
            self._fired_signals.add(dedup_key)
            self._pivot_pending    = last_sig
            self._pivot_pending_ts = now

    # ── main loop ──────────────────────────────────────────────────────────────

    def start(self) -> None:
        logger.info(
            f"[POLL] Starting VAM 10S Polling Engine "
            f"(interval={self.config.POLL_INTERVAL_SECS}s)"
        )
        try:
            while True:
                loop_start = time.monotonic()
                try:
                    self._poll_once()
                except Exception as e:
                    logger.error(f"[POLL] _poll_once error: {e}", exc_info=True)

                elapsed   = time.monotonic() - loop_start
                sleep_for = max(0.0, self.config.POLL_INTERVAL_SECS - elapsed)
                logger.debug(f"[POLL] Tick took {elapsed:.2f}s, sleeping {sleep_for:.2f}s")
                time.sleep(sleep_for)

        except KeyboardInterrupt:
            logger.info("[POLL] KeyboardInterrupt — running EOD cleanup.")
        finally:
            self._shutdown()

    def _shutdown(self) -> None:
        logger.info("[POLL] Shutdown: resolving open trades and exporting Excel.")
        if self.order_manager.has_position:
            self.order_manager.close_position("shutdown_exit")
        self.trade_logger.resolve_open_trade_at_eod()
        self.trade_logger.export_to_excel()
        logger.info("[POLL] Shutdown complete.")


# --------------------------- MAIN ---------------------------

if __name__ == "__main__":
    engine = VAMPollingEngine()
    engine.start()
