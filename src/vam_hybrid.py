"""
vam_hybrid.py  —  VAM Hybrid Bot
=================================
Signal generation : polling (identical to vam_poll_5S.py, every 5 s)
HTF trend         : live WebSocket ticks → 15-min bar builder (no API resample)

Why hybrid?
  - Polling keeps proven signal logic unchanged.
  - WebSocket HTF removes the need for the 60-s age-filter hack: a 15-min bar
    is only "completed" when the first tick of the *next* bar arrives, so
    htf_trend is always based on fully closed bars — exactly like the backtest.
  - No spurious 09:30 signals, no mid-bar EMA noise.

Architecture:
  HTFBarBuilder  — background WebSocket thread
    ├── seeded with API 15-min history at startup (EMA warm from day 1)
    ├── on every NIFTY50-INDEX tick → updates current 15-min bar
    ├── on bar-close tick → finalises bar, recomputes EMA, writes self.trend
    └── get_htf_df() → DataFrame[timestamp, htf_trend] for merge_asof

  VAMHybridEngine — polling loop (every 5 s)
    └── identical to VAMPollingEngine except _build_htf_trend() reads
        htf_builder.get_htf_df() instead of resampling API data
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
import time
import json
import threading
import urllib3

from fyers_apiv3.FyersWebsocket import data_ws

# ─── SSL FIX ──────────────────────────────────────────────────────────────────
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# --------------------------- TIMEZONE ---------------------------
IST = pytz.timezone("Asia/Kolkata")

# --------------------------- STATE FILE ---------------------------
STATE_FILE = "vam_hybrid_position_state.json"   # separate from polling bot

# --------------------------- LOGGING ---------------------------

def ist_time(*args):
    return datetime.now(IST).timetuple()

logging.Formatter.converter = ist_time

try:
    from logger_config import get_logger
    logger = get_logger()
except ImportError:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - VAMHybrid - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logger = logging.getLogger("VAMHybrid")

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
    yy = dt.year % 100
    monthly_exp = get_nifty_monthly_expiry(dt.year, dt.month)
    if dt.date() == monthly_exp.date():
        return f"{yy:02d}{dt.strftime('%b').upper()}"
    else:
        month_code = str(dt.month) if dt.month <= 9 else {10: "O", 11: "N", 12: "D"}[dt.month]
        return f"{yy:02d}{month_code}{dt.day:02d}"


# --------------------------- CONFIG ---------------------------

class TradingConfig:
    UNDERLYING_SYMBOL = "NSE:NIFTY50-INDEX"

    TIMEFRAME  = "10S"
    HIST_DAYS  = 3       # days of 10S history for signal scan warmup

    HTF_RULE    = "15min"
    HTF_MINS    = 15      # numeric version of HTF_RULE for bar bucketing
    HTF_EMA_LEN = 6
    HTF_HIST_DAYS = 5    # days of 15-min history to seed the HTF builder

    STRIKE_STEP    = 100
    NIFTY_LOT_SIZE = 65
    EXPIRY_TYPE    = "WEEKLY"

    VAM_BUY_SCORE  =  0.10
    VAM_SELL_SCORE = -0.10

    LEN1 = 8;  LEN2 = 10;  LEN3 = 14;  LEN4 = 20;  LEN5 = 28
    MULT1 = 1.05; MULT2 = 1.10; MULT3 = 1.15; MULT4 = 1.20; MULT5 = 1.25
    BAND1 = 1.15; BAND2 = 1.20; BAND3 = 1.25; BAND4 = 1.30; BAND5 = 1.35
    SMOOTH1 = 2;  SMOOTH2 = 2;  SMOOTH3 = 3;  SMOOTH4 = 3;  SMOOTH5 = 4

    ATR_LEN       = 14
    ATR_LAYERMULT = 0.90
    TRAIL_SMOOTH  = 4

    SESSION_START = dtime(9, 20)
    SESSION_END   = dtime(15, 30)
    TRADE_START   = dtime(9, 30)
    TRADE_END     = dtime(15, 0)
    EOD_EXIT_TIME = dtime(15, 20)

    STOP_LOSS_PCT   = 20.0
    TAKE_PROFIT_PCT = 25.0   # matches backtest

    PIVOT_BARS          = 10
    PIVOT_PROXIMITY_PCT = 0.5

    POLL_INTERVAL_SECS  = 5
    SIGNAL_CONFIRM_BARS = 1
    SIGNAL_STALE_SECS   = 15

    DRY_RUN = True

    def __init__(self):
        now = datetime.now(IST)
        exp_type = self.EXPIRY_TYPE.upper()
        if exp_type == "WEEKLY":
            exp_dt = get_next_tuesday(now)
            kind   = "WEEKLY"
        else:
            exp_dt = get_current_or_next_monthly_expiry(now)
            kind   = "MONTHLY"
        self.EXPIRY_CODE = format_expiry_code(exp_dt)
        logger.info(f"[HYBRID] NIFTY {kind} expiry {exp_dt.date()} → code {self.EXPIRY_CODE}")


# --------------------------- FYERS HTTP CLIENT ---------------------------

class FyersHTTP:
    BASE_T1 = "https://api-t1.fyers.in/api/v3"
    BASE_T2 = "https://api-t1.fyers.in/data"

    def __init__(self, app_id: str, token: str):
        combined = f"{app_id}:{token}" if ":" not in token else token
        self.session = requests.Session()
        self.session.verify = False
        self.session.headers.update({"Authorization": combined})
        logger.info(f"[HYBRID] FyersHTTP init | app_id={app_id} | auth={combined[:40]}...")

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
        try:
            return r.json()
        except Exception as e:
            raise ValueError(f"History API non-JSON (status={r.status_code}): {r.text[:300]}") from e

    def quotes(self, symbol: str) -> dict:
        r = self.session.get(f"{self.BASE_T2}/quotes", params={"symbols": symbol})
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

    def exit_position(self, symbol: str, qty: int = 0) -> dict:
        position_id = f"{symbol}-BO"
        data = {"id": position_id}
        r = self.session.delete(f"{self.BASE_T1}/positions", json=data)
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
    params = dict(
        lens  =[cfg.LEN1, cfg.LEN2, cfg.LEN3, cfg.LEN4, cfg.LEN5],
        mults =[cfg.MULT1, cfg.MULT2, cfg.MULT3, cfg.MULT4, cfg.MULT5],
        bands =[cfg.BAND1, cfg.BAND2, cfg.BAND3, cfg.BAND4, cfg.BAND5],
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

    tr_val    = tr(df_local)
    atr_layer = ema(tr_val, cfg.ATR_LEN)
    tl1       = ema(lowers[3], cfg.TRAIL_SMOOTH)
    tu1       = ema(uppers[3], cfg.TRAIL_SMOOTH)
    df_local["vam_trail_lower1"] = tl1
    df_local["vam_trail_upper1"] = tu1
    df_local["vam_trail_lower2"] = ema(tl1 - atr_layer * cfg.ATR_LAYERMULT, cfg.TRAIL_SMOOTH)
    df_local["vam_trail_upper2"] = ema(tu1 + atr_layer * cfg.ATR_LAYERMULT, cfg.TRAIL_SMOOTH)
    return df_local


# --------------------------- PIVOT HELPER ---------------------------

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


# --------------------------- HTF BAR BUILDER (WebSocket) ---------------------------

class HTFBarBuilder:
    """
    Builds 15-min OHLCV bars from live NIFTY50-INDEX WebSocket ticks.

    Lifecycle:
      1. __init__  — seeds _completed_bars from API 15-min history, computes
                     initial EMA so trend is valid before the first live tick.
      2. start()   — launches background WebSocket thread.
      3. on_tick() — called by WS callback; updates current bar; on bar-close,
                     finalises bar, recomputes EMA, writes self.trend.
      4. get_htf_df() — returns DataFrame[timestamp, htf_ema, htf_trend]
                        for use in merge_asof (same shape as the old resample output).
    """

    def __init__(self, config: TradingConfig, fyers_http: FyersHTTP, ws_token: str):
        self.config    = config
        self._fyers    = fyers_http
        self._ws_token = ws_token
        self._lock     = threading.Lock()

        # List of finalised bar dicts: {ts, open, high, low, close}
        self._completed_bars: list = []

        # Current open (incomplete) bar
        self._current_bar: Optional[dict] = None

        # Exposed state (read by polling engine under GIL — no lock needed for reads)
        self.trend:   str   = "flat"
        self._htf_df: pd.DataFrame = pd.DataFrame()

        # Seed with historical completed bars so EMA starts warm
        self._seed_from_api()

        # WebSocket internals
        self._ws = None
        self._ws_thread: Optional[threading.Thread] = None
        self._running = False

    # ── bar bucketing ──────────────────────────────────────────────────────────

    def _bar_key(self, ts: datetime) -> datetime:
        """Floor ts to the nearest HTF_MINS boundary."""
        mins = self.config.HTF_MINS
        bucket = (ts.minute // mins) * mins
        return ts.replace(minute=bucket, second=0, microsecond=0)

    # ── historical seed ────────────────────────────────────────────────────────

    def _seed_from_api(self) -> None:
        """
        Fetch HTF_HIST_DAYS of 15-min history from Fyers and populate
        _completed_bars with all bars EXCEPT the current incomplete one.
        """
        logger.info("[HTF] Seeding 15-min bar history from API...")
        try:
            end_dt   = datetime.now(IST)
            start_dt = end_dt - timedelta(days=self.config.HTF_HIST_DAYS)
            resp = self._fyers.history({
                "symbol":      self.config.UNDERLYING_SYMBOL,
                "resolution":  "15",   # 15-minute bars
                "date_format": "1",
                "range_from":  start_dt.strftime("%Y-%m-%d"),
                "range_to":    end_dt.strftime("%Y-%m-%d"),
                "cont_flag":   "0",
            })
            if not isinstance(resp, dict) or resp.get("s") != "ok":
                logger.warning(f"[HTF] Seed API error: {resp}")
                return

            candles = resp.get("candles", [])
            if not candles:
                logger.warning("[HTF] No candles returned for seed.")
                return

            df = pd.DataFrame(candles, columns=["timestamp","open","high","low","close","volume"])
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s", utc=True).dt.tz_convert(IST)
            df = df.sort_values("timestamp").reset_index(drop=True)

            # Drop the last bar if it is still open (its end > now)
            if not df.empty:
                last_ts    = df.iloc[-1]["timestamp"]
                if last_ts.tzinfo is None:
                    last_ts = IST.localize(last_ts)
                bar_end = last_ts + timedelta(minutes=self.config.HTF_MINS)
                if bar_end > end_dt:
                    df = df.iloc[:-1]

            # Populate _completed_bars
            with self._lock:
                self._completed_bars = [
                    {
                        "ts":    row["timestamp"],
                        "open":  float(row["open"]),
                        "high":  float(row["high"]),
                        "low":   float(row["low"]),
                        "close": float(row["close"]),
                    }
                    for _, row in df.iterrows()
                ]
                self._recompute_ema_locked()

            logger.info(
                f"[HTF] Seed complete: {len(self._completed_bars)} completed 15-min bars, "
                f"trend={self.trend}"
            )
        except Exception as e:
            logger.error(f"[HTF] Seed failed: {e}", exc_info=True)

    # ── EMA recompute (called inside lock) ────────────────────────────────────

    def _recompute_ema_locked(self) -> None:
        """Recompute EMA(6) on all completed bars and update self.trend + self._htf_df."""
        if not self._completed_bars:
            return
        closes = pd.Series([b["close"] for b in self._completed_bars], dtype=float)
        ema_series = closes.ewm(span=self.config.HTF_EMA_LEN, adjust=False).mean()

        last_close = closes.iloc[-1]
        last_ema   = ema_series.iloc[-1]
        if   last_close > last_ema: self.trend = "up"
        elif last_close < last_ema: self.trend = "down"
        else:                       self.trend = "flat"

        # Rebuild the DataFrame exposed to the polling engine
        timestamps = [b["ts"] for b in self._completed_bars]
        self._htf_df = pd.DataFrame({
            "timestamp": timestamps,
            "htf_ema":   ema_series.values,
            "htf_trend": np.where(
                closes.values > ema_series.values, "up",
                np.where(closes.values < ema_series.values, "down", "flat")
            ),
        })

    # ── public accessor ────────────────────────────────────────────────────────

    def get_htf_df(self) -> pd.DataFrame:
        """
        Thread-safe snapshot of [timestamp, htf_ema, htf_trend].
        Same shape as the old _build_htf_trend() output — drop-in replacement
        for the merge_asof in _run_backtest_scan.
        """
        with self._lock:
            return self._htf_df.copy() if not self._htf_df.empty else pd.DataFrame(
                columns=["timestamp", "htf_ema", "htf_trend"]
            )

    # ── tick handler ──────────────────────────────────────────────────────────

    def on_tick(self, tick: dict) -> None:
        """
        Called by the WebSocket callback for every NIFTY50-INDEX tick.
        tick expected: {"ltp": float, "timestamp": int (epoch seconds), ...}
        """
        try:
            ltp = float(tick.get("ltp", 0))
            ts_raw = tick.get("timestamp")
            if ltp <= 0 or ts_raw is None:
                return

            ts = datetime.fromtimestamp(float(ts_raw), tz=IST)
            # Only process during session
            if not (self.config.SESSION_START <= ts.time() <= self.config.SESSION_END):
                return

            bar_key = self._bar_key(ts)

            with self._lock:
                if self._current_bar is None:
                    # First tick ever
                    self._current_bar = {
                        "ts_key": bar_key, "open": ltp, "high": ltp,
                        "low": ltp, "close": ltp,
                    }
                elif bar_key != self._current_bar["ts_key"]:
                    # Bar boundary crossed → finalise previous bar
                    closed = {
                        "ts":    self._current_bar["ts_key"],
                        "open":  self._current_bar["open"],
                        "high":  self._current_bar["high"],
                        "low":   self._current_bar["low"],
                        "close": self._current_bar["close"],
                    }
                    self._completed_bars.append(closed)
                    # Keep at most 2 days of 15-min bars (≈ 96 bars/day)
                    if len(self._completed_bars) > 200:
                        self._completed_bars = self._completed_bars[-200:]
                    self._recompute_ema_locked()
                    logger.info(
                        f"[HTF] Bar closed: ts={closed['ts'].strftime('%H:%M')}, "
                        f"close={closed['close']:.2f} → trend={self.trend}"
                    )
                    # Open new bar
                    self._current_bar = {
                        "ts_key": bar_key, "open": ltp, "high": ltp,
                        "low": ltp, "close": ltp,
                    }
                else:
                    # Update current bar
                    cb = self._current_bar
                    if ltp > cb["high"]: cb["high"] = ltp
                    if ltp < cb["low"]:  cb["low"]  = ltp
                    cb["close"] = ltp
        except Exception as e:
            logger.error(f"[HTF] on_tick error: {e}")

    # ── WebSocket management ───────────────────────────────────────────────────

    def _ws_callback(self, msg: dict) -> None:
        """data_ws message handler."""
        if not isinstance(msg, dict):
            return
        msg_type = msg.get("type", "")
        if msg_type == "symbolData":
            ticks = msg.get("ticks", [])
            for tick in (ticks if isinstance(ticks, list) else [ticks]):
                sym = tick.get("symbol", "")
                if sym == self.config.UNDERLYING_SYMBOL:
                    self.on_tick(tick)
        elif msg_type == "error":
            logger.error(f"[HTF WS] Error message: {msg}")

    def _ws_error(self, msg) -> None:
        logger.error(f"[HTF WS] Socket error: {msg}")

    def _ws_close(self, msg) -> None:
        logger.warning(f"[HTF WS] Connection closed: {msg}")
        if self._running:
            logger.info("[HTF WS] Reconnecting in 5s...")
            time.sleep(5)
            self._connect_ws()

    def _connect_ws(self) -> None:
        try:
            self._ws = data_ws.FyersDataSocket(
                access_token=self._ws_token,
                log_path="",
                litemode=False,
                write_to_file=False,
                reconnect=True,
                on_connect=lambda ws, msg: logger.info("[HTF WS] Connected ✅"),
                on_close=self._ws_close,
                on_error=self._ws_error,
                on_message=self._ws_callback,
            )
            self._ws.subscribe(
                symbols=[self.config.UNDERLYING_SYMBOL],
                data_type="symbolData",
            )
            self._ws.keep_running()    # blocks until close
        except Exception as e:
            logger.error(f"[HTF WS] _connect_ws error: {e}", exc_info=True)

    def start(self) -> None:
        """Start the WebSocket tick-ingestion thread."""
        self._running = True
        self._ws_thread = threading.Thread(
            target=self._connect_ws,
            daemon=True,
            name="HTFBarBuilder-WS",
        )
        self._ws_thread.start()
        logger.info("[HTF] WebSocket thread started.")

    def stop(self) -> None:
        self._running = False
        try:
            if self._ws:
                self._ws.close()
        except Exception:
            pass


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
        if self.config.DRY_RUN:
            logger.warning("[OM] *** DRY-RUN MODE ENABLED — no real orders will be placed ***")

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
                f"[OM] Restored from file: symbol={self.option_symbol}, "
                f"direction={self.direction}, entry_price={self.entry_price}"
            )
            return True
        except Exception as e:
            logger.error(f"[OM] Failed to load state file: {e}"); return False

    def _clear_state_file(self) -> None:
        try:
            if os.path.exists(STATE_FILE):
                os.remove(STATE_FILE)
        except Exception as e:
            logger.error(f"[OM] Failed to remove state file: {e}")

    def _find_bo_order_id(self, symbol: str) -> Optional[str]:
        try:
            ob = self.fyers.orderbook()
            if not isinstance(ob, dict) or ob.get("s") != "ok":
                return None
            for order in ob.get("orderBook", []):
                if (order.get("symbol") == symbol and order.get("productType") == "BO"
                        and order.get("side") == 1 and order.get("status") == 2):
                    return order.get("id")
            return None
        except Exception as e:
            logger.error(f"[OM] _find_bo_order_id error: {e}"); return None

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
                    direction = "BUY"; entry_price = float(pos.get("buyAvg", 0.0))
                elif symbol.endswith("PE"):
                    direction = "SELL"; entry_price = float(pos.get("buyAvg", 0.0))
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
            logger.error(f"[OM] _restore_state_from_broker error: {e}"); return False

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
            logger.warning("[OM] Missing option_symbol in signal"); return False
        if self.has_position:
            logger.warning("[OM] Position already open, ignoring."); return False

        entry_price          = float(opt_ltp)
        sl_points, tp_points = self._compute_bo_sl_tp_points(entry_price)
        limit_price          = round(entry_price * 1.005, 1)
        data = {
            "symbol": symbol, "qty": int(self.qty), "type": 1, "side": 1,
            "productType": "BO", "limitPrice": limit_price, "stopPrice": 0,
            "validity": "DAY", "disclosedQty": 0, "offlineOrder": False,
            "stopLoss": float(sl_points), "takeProfit": float(tp_points),
        }
        try:
            logger.info(
                f"[OM] {'[DRY-RUN] ' if self.config.DRY_RUN else ''}Placing BO: "
                f"symbol={symbol}, dir={direction}, qty={self.qty}, "
                f"ltp={entry_price:.2f}, limitPrice={limit_price:.1f}, "
                f"SL_pts={sl_points:.0f}, TP_pts={tp_points:.0f}"
            )
            if self.config.DRY_RUN:
                self.has_position   = True
                self.option_symbol  = symbol
                self.direction      = direction
                self.entry_price    = entry_price
                self.entry_order_id = f"DRY-{datetime.now(IST).strftime('%H%M%S%f')}"
                self._save_state()
                logger.info(f"[OM] [DRY-RUN] Simulated fill, order_id={self.entry_order_id}")
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
                logger.info(f"[OM] BO placed, order_id={self.entry_order_id}")
                return True
            else:
                logger.error(f"[OM] BO rejected: {resp}"); return False
        except Exception as e:
            logger.error(f"[OM] place_order error: {e}"); return False

    def sync_with_broker(self) -> bool:
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
                        return False
                    logger.info(
                        f"[OM] Broker-side exit detected for {self.option_symbol} "
                        f"(netQty=0). Clearing local state."
                    )
                    self.has_position = False; self.option_symbol = None
                    self.direction = None; self.entry_price = None
                    self.entry_order_id = None
                    self._clear_state_file(); return True
            logger.info(f"[OM] {self.option_symbol} not in broker positions — assuming auto-closed.")
            self.has_position = False; self.option_symbol = None
            self.direction = None; self.entry_price = None
            self.entry_order_id = None
            self._clear_state_file(); return True
        except Exception as e:
            logger.error(f"[OM] sync_with_broker error: {e}"); return False

    def close_position(self, exit_reason: str, exit_price: Optional[float] = None) -> bool:
        if not self.has_position or not self.option_symbol:
            logger.warning("[OM] No open position, close_position() ignored."); return False
        if not self.entry_order_id:
            logger.warning("[OM] No entry_order_id stored, cannot exit BO."); return False
        try:
            logger.info(
                f"[OM] {'[DRY-RUN] ' if self.config.DRY_RUN else ''}Exiting BO. "
                f"reason={exit_reason}, order_id={self.entry_order_id}, symbol={self.option_symbol}"
            )
            if self.config.DRY_RUN:
                self.has_position = False; self.option_symbol = None
                self.direction = None; self.entry_price = None
                self.entry_order_id = None
                self._clear_state_file()
                logger.info("[OM] [DRY-RUN] Simulated exit, state cleared.")
                return True
            resp = self.fyers.exit_position(symbol=self.option_symbol, qty=self.qty)
            logger.info(f"[OM] exit_position response: {resp}")
            if isinstance(resp, dict) and resp.get("s") == "ok":
                self.has_position = False; self.option_symbol = None
                self.direction = None; self.entry_price = None
                self.entry_order_id = None
                self._clear_state_file()
                logger.info("[OM] BO exited successfully, state cleared.")
                return True
            else:
                logger.error(f"[OM] exit_position failed: {resp}"); return False
        except Exception as e:
            logger.error(f"[OM] exit_position error: {e}"); return False


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
            "strategy_type":  "vam_hybrid",
            "htf_trend":      sig.get("htf_trend", ""),
            "htf_source":     "websocket",
            "pivot_level":    sig.get("pivot_level"),
            "entry_method":   sig.get("entry_method", ""),
            "entry_price":    round(float(entry_price), 2),
            "entry_order_id": order_id,
            "exit_time": None, "exit_price": None,
            "exit_reason": None, "pnl_pct": None,
            "pnl_rupees": None, "result": None,
        }
        logger.info(
            f"[TL] Entry: symbol={self._open['option_symbol']}, "
            f"dir={self._open['direction']}, price={self._open['entry_price']}, "
            f"pivot={self._open['pivot_level']}, method={self._open['entry_method']}, "
            f"htf_source=websocket"
        )

    def log_flip_exit(self, exit_price: float) -> None:
        if self._open is None:
            logger.warning("[TL] log_flip_exit called but no open trade."); return
        self._finalize_trade(float(exit_price), "flip_exit")

    def log_sl_tp_exit(self, exit_price: float, reason: str) -> None:
        if self._open is None:
            logger.warning("[TL] log_sl_tp_exit called but no open trade."); return
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
            logger.info("[TL] No open trade to resolve at EOD."); return
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
            entry = self._open["entry_price"]
            sl_price = entry * (1 - self.config.STOP_LOSS_PCT   / 100.0)
            tp_price = entry * (1 + self.config.TAKE_PROFIT_PCT / 100.0)
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
            logger.info("[TL] No trades today — nothing to export."); return None
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
            xlsx_path = os.path.join(log_dir, f"trades_hybrid_{today}.xlsx")

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

            total = len(df); wins = int((df["result"] == "profit").sum())
            losses = int((df["result"] == "loss").sum())
            net = df["pnl_rupees"].sum()
            wr  = round(wins / total * 100, 1) if total > 0 else 0.0
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
                ("HTF Source",       "websocket"),
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
            logger.error(f"[TL] Excel export failed: {e}"); return None


# --------------------------- HISTORY CACHE (10S bars for signal scan) ---------------------------

class HistoryCache:
    def __init__(self, fyers_client: FyersHTTP, config: TradingConfig):
        self.fyers       = fyers_client
        self.config      = config
        self._cache:     Optional[pd.DataFrame] = None
        self._cache_day: Optional[object]       = None

    def _fetch_raw(self, days: int) -> pd.DataFrame:
        end_dt   = datetime.now(IST)
        start_dt = end_dt - timedelta(days=days)
        resp = self.fyers.history({
            "symbol":      self.config.UNDERLYING_SYMBOL,
            "resolution":  self.config.TIMEFRAME,
            "date_format": "1",
            "range_from":  start_dt.strftime("%Y-%m-%d"),
            "range_to":    end_dt.strftime("%Y-%m-%d"),
            "cont_flag":   "0",
        })
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
        mask = df.index.indexer_between_time(
            self.config.SESSION_START, self.config.SESSION_END,
            include_start=True, include_end=True,
        )
        return df.iloc[mask]

    def get(self) -> pd.DataFrame:
        today = datetime.now(IST).date()
        if self._cache is None or self._cache_day != today:
            logger.info("[CACHE] Fetching warmup history...")
            self._cache     = self._fetch_raw(self.config.HIST_DAYS)
            self._cache_day = today
            logger.info(f"[CACHE] Warmup complete: {len(self._cache)} bars")
            return self._cache
        today_fresh = self._fetch_raw(days=1)
        if today_fresh.empty:
            return self._cache
        cache_excl = self._cache[pd.to_datetime(self._cache.index).date != today]
        merged = pd.concat([cache_excl, today_fresh])
        merged = merged[~merged.index.duplicated(keep="last")]
        merged.sort_index(inplace=True)
        self._cache = merged
        return self._cache


# --------------------------- HYBRID POLLING ENGINE ---------------------------

class VAMHybridEngine:
    """
    Polling engine identical to VAMPollingEngine, with one difference:
      _build_htf_trend() is replaced by htf_builder.get_htf_df()

    This means HTF trend is driven by live WebSocket ticks rather than
    resampling API-fetched 10S bars, eliminating the 60-s age-filter hack.
    """

    def __init__(self, ws_token: str):
        self.config = TradingConfig()
        self.fyers  = self._init_fyers()

        # ── HTF bar builder — must start before first poll ──
        self.htf_builder = HTFBarBuilder(self.config, self.fyers, ws_token)
        self.htf_builder.start()

        self.history_cache = HistoryCache(self.fyers, self.config)
        self.order_manager = VAMOrderManager(self.fyers, self.config)
        self.order_manager.restore_state_on_startup()
        self.trade_logger  = TradeLogger(self.fyers, self.config)

        self._fired_signals:    set              = set()
        self._pivot_pending:    Optional[dict]   = None
        self._pivot_pending_ts: Optional[datetime] = None
        self._confirm_pending:  Optional[dict]   = None
        self._signal_cache:     dict             = {}

        if self.config.DRY_RUN:
            logger.warning("[HYBRID] *** DRY-RUN MODE — no real orders will be placed ***")
        logger.info(
            f"[HYBRID] Engine ready. Interval={self.config.POLL_INTERVAL_SECS}s, "
            f"TF={self.config.TIMEFRAME}, HTF={self.config.HTF_RULE} (WebSocket), "
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
        logger.info(f"[HYBRID] Fyers connected — {profile['data']['name']} ✅")
        return client

    # ── option helpers ─────────────────────────────────────────────────────────

    def _get_option_ltp(self, symbol: str) -> Optional[float]:
        try:
            resp = self.fyers.quotes(symbol)
            if isinstance(resp, dict) and resp.get("d"):
                d0 = resp["d"][0]
                if isinstance(d0, dict):
                    ltp = d0.get("v", {}).get("lp")
                    if ltp is not None:
                        return float(ltp)
            logger.warning(f"[HYBRID] Could not get LTP for {symbol}: {resp}")
            return None
        except Exception as e:
            logger.error(f"[HYBRID] _get_option_ltp error: {e}"); return None

    def _pick_itm_option_symbol(self, underlying_price: float, direction: str) -> str:
        c = self.config
        spot = float(underlying_price)
        step = c.STRIKE_STEP
        strike   = int(round(spot / step)) * step
        strike   = int(max(step, strike))
        opt_type = "CE" if direction == "BUY" else "PE"
        return f"NSE:NIFTY{c.EXPIRY_CODE}{strike:05d}{opt_type}"

    # ── HTF trend — read from WebSocket builder ────────────────────────────────

    def _get_htf_df(self) -> pd.DataFrame:
        """
        Returns DataFrame[timestamp, htf_ema, htf_trend] from the live
        WebSocket bar builder.  Falls back gracefully to empty df if the
        builder has no bars yet (e.g. first few seconds after startup).
        """
        df = self.htf_builder.get_htf_df()
        if df.empty:
            logger.warning("[HYBRID] HTF builder has no bars yet — trend unavailable.")
        else:
            logger.debug(
                f"[HYBRID] HTF df: {len(df)} bars, "
                f"last={df.iloc[-1]['timestamp']}, trend={df.iloc[-1]['htf_trend']}"
            )
        return df

    # ── core signal scan ───────────────────────────────────────────────────────

    def _run_backtest_scan(self, df_idx: pd.DataFrame) -> list:
        df_idx_r = df_idx.copy().reset_index()
        df_vam   = compute_vam_ensemble(df_idx_r, self.config)

        # ── KEY DIFFERENCE: HTF comes from WebSocket builder, not API resample ──
        df_htf = self._get_htf_df()
        if df_htf.empty:
            return []   # No HTF yet — skip scan until builder warms up

        df_merged = pd.merge_asof(
            df_vam.sort_values("timestamp"),
            df_htf[["timestamp", "htf_trend"]].sort_values("timestamp"),
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
            row     = df_merged.iloc[i]
            ts      = row["timestamp"]
            ts_time = ts.time()
            htf     = row.get("htf_trend", "flat")
            buy_sig = bool(row["vam_buy_sig"])
            sell_sig = bool(row["vam_sell_sig"])

            in_window = self.config.TRADE_START <= ts_time <= self.config.TRADE_END
            ok_long   = (position_size <= 0) and (last_trade != "long")
            ok_short  = (position_size >= 0) and (last_trade != "short")

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
                    "strategy_type": "vam_hybrid",
                    "htf_trend":     htf,
                    "pivot_level":   pivot_lvl,
                })
                position_size = 1; last_trade = "long"

            elif sell_sig and ok_short and htf == "down":
                pivot_lvl = compute_pivot_zone(df_merged, i, self.config.PIVOT_BARS, "SELL")
                signals.append({
                    "timestamp":     ts,
                    "bar_pos":       i,
                    "index_price":   float(row["close"]),
                    "direction":     "SELL",
                    "option_symbol": self._pick_itm_option_symbol(float(row["close"]), "SELL"),
                    "strategy_type": "vam_hybrid",
                    "htf_trend":     htf,
                    "pivot_level":   pivot_lvl,
                })
                position_size = -1; last_trade = "short"

        logger.info(f"[HYBRID] Scan: {len(signals)} total signals, HTF trend={self.htf_builder.trend}")
        return signals

    # ── pivot proximity ────────────────────────────────────────────────────────

    def _is_near_pivot(self, current_price: float, pivot_lvl: float, direction: str) -> bool:
        prox_band = pivot_lvl * (self.config.PIVOT_PROXIMITY_PCT / 100.0)
        if direction == "BUY":
            return (current_price - pivot_lvl) <= prox_band
        else:
            return (pivot_lvl - current_price) <= prox_band

    # ── execute a signal ───────────────────────────────────────────────────────

    def _execute_signal(self, sig: dict, opt_ltp: float, entry_method: str) -> None:
        direction = sig["direction"]
        if self.order_manager.has_position:
            if direction != self.order_manager.direction:
                logger.info("[HYBRID] Opposite signal → flipping position.")
                existing_ltp = self._get_option_ltp(self.order_manager.option_symbol) or opt_ltp
                closed = self.order_manager.close_position("flip_exit", existing_ltp)
                if not closed:
                    logger.error("[HYBRID] Flip close failed — skipping new entry."); return
                self.trade_logger.log_flip_exit(existing_ltp)
            else:
                logger.info("[HYBRID] Same-direction signal, already in position — skipping."); return
        sig["entry_method"] = entry_method
        ok = self.order_manager.open_position(sig, opt_ltp)
        if ok:
            self.trade_logger.log_entry(sig, opt_ltp, self.order_manager.entry_order_id)
            logger.info(f"[HYBRID] ✅ BO placed: {sig['option_symbol']} {direction} ({entry_method})")

    # ── SL/TP simulation (dry-run) ─────────────────────────────────────────────

    def _check_dryrun_sl_tp(self) -> bool:
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
            logger.info(f"[DRY-RUN] SL hit: {symbol} ltp={current_ltp:.2f} <= sl={sl_price:.2f}")
            self.order_manager.close_position("stop_loss")
            self.trade_logger.log_sl_tp_exit(current_ltp, "stop_loss")
            return True
        if current_ltp >= tp_price:
            logger.info(f"[DRY-RUN] TP hit: {symbol} ltp={current_ltp:.2f} >= tp={tp_price:.2f}")
            self.order_manager.close_position("target")
            self.trade_logger.log_sl_tp_exit(tp_price, "target")
            return True
        logger.debug(
            f"[DRY-RUN] Position live: {symbol} ltp={current_ltp:.2f} "
            f"entry={entry_price:.2f} sl={sl_price:.2f} tp={tp_price:.2f}"
        )
        return False

    # ── EOD exit ──────────────────────────────────────────────────────────────

    def _eod_exit_if_needed(self) -> bool:
        now = datetime.now(IST)
        eod = IST.localize(datetime.combine(now.date(), self.config.EOD_EXIT_TIME))
        if now >= eod and self.order_manager.has_position:
            logger.info("[HYBRID] EOD reached — force-closing position.")
            if self.config.DRY_RUN:
                symbol = self.order_manager.option_symbol or (self.trade_logger._open or {}).get("option_symbol")
                self.order_manager.close_position("eod_exit")
                eod_ltp = self._get_option_ltp(symbol) if symbol else None
                if eod_ltp is not None:
                    self.trade_logger.log_sl_tp_exit(eod_ltp, "eod_exit")
                else:
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
            logger.debug(f"[HYBRID] Outside session ({now.strftime('%H:%M:%S')}), sleeping.")
            return

        # ── 1. Position sync / SL-TP simulation ──
        if self.config.DRY_RUN:
            if self._check_dryrun_sl_tp():
                return
        else:
            if self.order_manager.has_position:
                was_closed = self.order_manager.sync_with_broker()
                if was_closed:
                    self.trade_logger.resolve_open_trade_at_eod()

        # ── 2. Fresh 10S bar data ──
        df_idx = self.history_cache.get()
        if df_idx is None or df_idx.empty:
            logger.warning("[HYBRID] Empty dataframe — skipping tick."); return

        today = now.date()

        # ── 3. Pivot-wait ──
        if self._pivot_pending is not None:
            eod_boundary  = IST.localize(datetime.combine(today, self.config.EOD_EXIT_TIME))
            htf_minutes   = self.config.HTF_MINS
            pivot_timeout = timedelta(minutes=htf_minutes * 2)
            timed_out = (
                self._pivot_pending_ts is not None
                and (now - self._pivot_pending_ts) >= pivot_timeout
            )
            if now >= eod_boundary or timed_out:
                reason = "EOD" if now >= eod_boundary else f"timeout ({htf_minutes*2}min)"
                logger.info(f"[HYBRID] Abandoning pivot wait ({reason}) — resuming scan.")
                self._pivot_pending = None; self._pivot_pending_ts = None
                self._confirm_pending = None
                if now >= eod_boundary:
                    self._signal_cache.clear()
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
                        f"[HYBRID] Pivot wait: dir={direction}, pivot={pivot_lvl:.2f}, "
                        f"check={check_price:.2f}"
                    )
                    if abs(check_price - pivot_lvl) <= prox_band:
                        confirmed_ltp = sig.get("confirmed_ltp") or self._get_option_ltp(sig["option_symbol"])
                        if confirmed_ltp is not None:
                            logger.info(f"[HYBRID] Pivot reached — executing at ltp={confirmed_ltp:.2f}.")
                            self._execute_signal(sig, confirmed_ltp, "pulled_to_pivot")
                        else:
                            logger.warning("[HYBRID] No LTP at pivot hit — skipping entry.")
                        self._pivot_pending = None; self._pivot_pending_ts = None
                return

        # ── 4. Signal scan ──
        all_signals = self._run_backtest_scan(df_idx)
        if not all_signals:
            return

        # ── 5. Today's signals only ──
        today_signals = [
            s for s in all_signals
            if pd.Timestamp(s["timestamp"]).date() == today
        ]
        if not today_signals:
            return

        # ── 6. First unfired signal ──
        next_sig = None
        for s in today_signals:
            key = (pd.Timestamp(s["timestamp"]).floor("10s"), s["direction"])
            if key not in self._fired_signals:
                next_sig = s; break

        if next_sig is None:
            logger.debug("[HYBRID] All today's signals already fired."); return

        last_sig  = next_sig
        sig_ts    = pd.Timestamp(last_sig["timestamp"]).floor("10s")
        sig_dir   = last_sig["direction"]
        dedup_key = (sig_ts, sig_dir)

        # Lock pivot + index price on first sight
        if dedup_key not in self._signal_cache:
            self._signal_cache[dedup_key] = {
                "pivot_level": last_sig.get("pivot_level"),
                "index_price": last_sig.get("index_price"),
            }
            logger.info(
                f"[HYBRID] 🔔 Signal (first sight): dir={sig_dir}, ts={sig_ts}, "
                f"index={last_sig['index_price']:.2f}, pivot={last_sig.get('pivot_level')}, "
                f"option={last_sig['option_symbol']}, HTF={self.htf_builder.trend}"
            )
        else:
            last_sig["pivot_level"] = self._signal_cache[dedup_key]["pivot_level"]
            last_sig["index_price"] = self._signal_cache[dedup_key]["index_price"]

        # ── 7. Confirmation delay ──
        n_required = self.config.SIGNAL_CONFIRM_BARS
        if self._confirm_pending is None or self._confirm_pending["key"] != dedup_key:
            first_ltp = self._get_option_ltp(last_sig["option_symbol"])
            if first_ltp is None:
                logger.warning(f"[HYBRID] No LTP for {last_sig['option_symbol']} — retrying next tick.")
                return
            self._confirm_pending = {
                "key": dedup_key, "sig": last_sig,
                "first_ltp": first_ltp, "count": 1,
            }
            logger.info(
                f"[HYBRID] Confirmation 1/{n_required}: {sig_dir} @ {sig_ts}, "
                f"first_ltp={first_ltp:.2f}"
            )
            if n_required > 1:
                return
            confirmed_ltp = first_ltp
            self._confirm_pending = None
        else:
            self._confirm_pending["count"] += 1
            count = self._confirm_pending["count"]
            logger.info(f"[HYBRID] Confirmation {count}/{n_required}: {sig_dir} @ {sig_ts}")
            if count < n_required:
                return
            confirmed_ltp = self._confirm_pending["first_ltp"]
            self._confirm_pending = None
            logger.info(
                f"[HYBRID] ✅ Signal confirmed ({n_required}/{n_required}) "
                f"— proceeding at ltp={confirmed_ltp:.2f}"
            )

        # ── 8. Staleness guard ──
        sig_dt = sig_ts.to_pydatetime()
        if sig_dt.tzinfo is None:
            sig_dt = IST.localize(sig_dt)
        sig_age_secs = (now - sig_dt).total_seconds()
        if sig_age_secs > self.config.SIGNAL_STALE_SECS:
            logger.info(
                f"[HYBRID] ⏩ Signal too old ({sig_age_secs:.0f}s > "
                f"{self.config.SIGNAL_STALE_SECS}s) — skipping."
            )
            self._fired_signals.add(dedup_key); self._confirm_pending = None; return

        # ── 9. Pivot required ──
        pivot_lvl = last_sig.get("pivot_level")
        if pivot_lvl is None:
            logger.info("[HYBRID] No pivot found — skipping signal.")
            self._fired_signals.add(dedup_key); return

        # ── 10. Pivot proximity (Case A / Case B) ──
        signal_price = float(last_sig["index_price"])
        if self._is_near_pivot(signal_price, pivot_lvl, sig_dir):
            logger.info(
                f"[HYBRID] Signal price {signal_price:.2f} already near pivot "
                f"{pivot_lvl:.2f} — entering now at ltp={confirmed_ltp:.2f}."
            )
            self._fired_signals.add(dedup_key)
            self._execute_signal(last_sig, confirmed_ltp, "already_near_pivot")
        else:
            logger.info(
                f"[HYBRID] Signal price {signal_price:.2f} not near pivot "
                f"{pivot_lvl:.2f} — entering pivot-wait (fill at ltp={confirmed_ltp:.2f})."
            )
            last_sig["confirmed_ltp"] = confirmed_ltp
            self._fired_signals.add(dedup_key)
            self._pivot_pending    = last_sig
            self._pivot_pending_ts = now

    # ── main loop ──────────────────────────────────────────────────────────────

    def start(self) -> None:
        logger.info(
            f"[HYBRID] Starting VAM Hybrid Engine "
            f"(poll={self.config.POLL_INTERVAL_SECS}s, HTF via WebSocket)"
        )
        try:
            while True:
                loop_start = time.monotonic()
                try:
                    self._poll_once()
                except Exception as e:
                    logger.error(f"[HYBRID] _poll_once error: {e}", exc_info=True)
                elapsed   = time.monotonic() - loop_start
                sleep_for = max(0.0, self.config.POLL_INTERVAL_SECS - elapsed)
                logger.debug(f"[HYBRID] Tick took {elapsed:.2f}s, sleeping {sleep_for:.2f}s")
                time.sleep(sleep_for)
        except KeyboardInterrupt:
            logger.info("[HYBRID] KeyboardInterrupt — shutting down.")
        finally:
            self._shutdown()

    def _shutdown(self) -> None:
        logger.info("[HYBRID] Shutdown: stopping WebSocket and exporting trades.")
        self.htf_builder.stop()
        if self.order_manager.has_position:
            self.order_manager.close_position("shutdown_exit")
        self.trade_logger.resolve_open_trade_at_eod()
        self.trade_logger.export_to_excel()
        logger.info("[HYBRID] Shutdown complete.")


# --------------------------- MAIN ---------------------------

if __name__ == "__main__":
    for f in ["fyers_appid.txt", "fyers_token.txt"]:
        if not os.path.exists(f):
            raise FileNotFoundError(f"Missing {f} — run runBot.py first to generate token.")

    app_id    = open("fyers_appid.txt").read().strip()
    raw_token = open("fyers_token.txt").read().strip()
    ws_token  = f"{app_id}:{raw_token}"

    engine = VAMHybridEngine(ws_token=ws_token)
    engine.start()
