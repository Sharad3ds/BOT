"""Live engine for the uploaded First Candle High-Low S/R backtest strategy.

Strategy derived from the supplied backtest file:
- First NIFTY 1-minute candle at 09:15 IST fixes ORH and ORL for the day.
- Bullish/CE setup: a closed candle closes above ORH; a later candle's OHLC
  range exactly touches ORH; then post-retest candle confirmation is checked.
- Bearish/PE setup: a closed candle closes below ORL; a later candle's OHLC
  range exactly touches ORL; then post-retest candle confirmation is checked.
- Latest user-confirmed confirmation rule: C1 red/C2 green for bullish or C1
  green/C2 red for bearish. If C2 real body >= C1 real body, C2 is confirmation,
  C3 is ignored, and entry is requested after C2 closes if C2 closes beyond ORH/ORL.
  If C2 body < C1 body, the uploaded backtest's strict C1/C2/C3 fallback applies:
  matching C3 candle colour, C3 body/range >= 80%, and close progression OR the
  original C1/C2 body-ratio condition; C3 must close beyond the breakout level.
- No EMA or swing-zone filter.
- Exact level touch (no tolerance), because the uploaded backtest uses low <= level <= high.
- ITM option from the current FYERS NSE_FO symbol master; do not synthesize symbols.
- Spot SL/TP: 40 / 130 NIFTY points. Force exit at 15:20 IST. Entry cutoff 15:00.
- Maximum one executed trade per day, matching the uploaded backtest configuration.

SHARED-TOKEN MODE:
- The launcher reads fyers_appid.txt / fyers_token.txt shared with the VAM bot.
- No token generation occurs unless separately enabled in the launcher.
- DRY_RUN=True and LIVE_TRADING_ENABLED=False by default. Real orders require both
  flags changed and FCSR_BACKTEST_ENABLE_LIVE_ORDERS=YES.
- SL/TP are monitored against NIFTY spot by this process and are NOT broker-side
  protective orders. If this process or its data feed fails, exits can be delayed.
- Order status is reconciled; ambiguous order requests are never blindly retried.
"""

from __future__ import annotations

import csv
import json
import logging
import os
import queue
import re
import threading
import time
from datetime import date, datetime, time as dtime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import pytz
import requests

IST = pytz.timezone("Asia/Kolkata")
BASE_API = "https://api-t1.fyers.in/api/v3"
BASE_DATA = "https://api-t1.fyers.in/data"


class Config:
    UNDERLYING = "NSE:NIFTY50-INDEX"
    RESOLUTION = "1"
    SESSION_START = dtime(9, 15)
    SESSION_END = dtime(15, 30)
    OPENING_CANDLE_TIME = dtime(9, 15)
    ENTRY_START = dtime(9, 16)
    ENTRY_END = dtime(15, 0)
    EOD_EXIT_TIME = dtime(15, 20)

    STRIKE_STEP = 100
    QUANTITY = 650           # Matches LOT=65 in the uploaded backtest; total option quantity.
    MAX_TRADES_PER_DAY = 1
    MONEYNESS = "ITM"

    LEVEL_TOUCH_TOLERANCE_PTS = 0.0  # Exact intersection, matching the uploaded backtest.
    MAX_C1_START_BARS_AFTER_RETEST = 1  # C1 is the candle immediately after retest.
    HEALTHY_BODY_FRAC = 0.80    # Uploaded backtest C3 quality filter, fallback only.
    BODY_RATIO_MAX = 1.20       # Only used by the C1/C2/C3 fallback.

    STOP_LOSS_SPOT_POINTS = 40.0
    TAKE_PROFIT_SPOT_POINTS = 130.0
    SLICE_ORDER = False

    DRY_RUN = False
    LIVE_TRADING_ENABLED = True
    LIVE_CONFIRM_ENV = "FCSR_BACKTEST_ENABLE_LIVE_ORDERS"

    API_TIMEOUT_SECS = 15
    ORDER_STATUS_WAIT_SECS = 8.0
    ORDER_STATUS_POLL_SECS = 0.5
    POSITION_QUOTE_FALLBACK_SECS = 3.0
    STALE_SIGNAL_MAX_SECS = 15.0

    SYMBOL_MASTER_URL = "https://public.fyers.in/sym_details/NSE_FO_sym_master.json"
    STATE_FILE = "First_Candle_SR_Backtest_Live_State.json"
    EVENTS_CSV = "First_Candle_SR_Backtest_Live_Events.csv"
    LOG_FILE = "First_Candle_SR_Backtest_Live.log"


def now_ist() -> datetime:
    return datetime.now(IST)


def as_float(value: Any) -> Optional[float]:
    try:
        result = float(value)
        return result if pd.notna(result) else None
    except (TypeError, ValueError):
        return None


def parse_expiry(value: Any) -> Optional[date]:
    if value in (None, "", 0, "0"):
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if isinstance(value, (int, float)) or (isinstance(value, str) and re.fullmatch(r"\d{9,13}", value.strip())):
        try:
            n = float(value)
            if n > 10_000_000_000:
                n /= 1000.0
            if n > 100_000_000:
                return datetime.fromtimestamp(n, tz=timezone.utc).date()
        except (OverflowError, OSError, ValueError):
            return None
    text = str(value).strip()
    for fmt in ("%Y-%m-%d", "%d-%m-%Y", "%d/%m/%Y", "%Y/%m/%d", "%d %b %Y", "%d-%b-%Y"):
        try:
            return datetime.strptime(text[:11], fmt).date()
        except ValueError:
            continue
    try:
        ts = pd.to_datetime(text, errors="coerce", utc=True)
        if pd.notna(ts):
            return ts.date()
    except Exception:
        pass
    return None


def parse_master_contract(key: str, item: Any) -> Optional[dict]:
    if not isinstance(item, dict):
        return None
    symbol = str(item.get("symTicker") or item.get("symbol") or key).upper().strip()
    if not symbol.startswith("NSE:NIFTY") or not symbol.endswith(("CE", "PE")):
        return None
    expiry = None
    for field in ("expiryDate", "expiry_date", "expiry", "expDate"):
        expiry = parse_expiry(item.get(field))
        if expiry:
            break
    strike = None
    for field in ("strikePrice", "strike_price", "strike"):
        strike = as_float(item.get(field))
        if strike and strike > 0:
            break
    opt_type = str(item.get("optType") or item.get("optionType") or item.get("option_type") or symbol[-2:]).upper()
    lot_size = None
    for field in ("minLotSize", "lotSize", "lot_size", "qtyMultiplier", "qty_multiplier"):
        lot_size = as_float(item.get(field))
        if lot_size and lot_size > 0:
            lot_size = int(lot_size)
            break
    if not expiry or not strike or opt_type not in ("CE", "PE"):
        return None
    return {"symbol": symbol, "expiry_date": expiry, "strike": float(strike),
            "option_type": opt_type, "lot_size": lot_size, "tick_size": as_float(item.get("tickSize"))}


class FyersREST:
    """Small REST wrapper around documented FYERS V3 endpoints."""
    def __init__(self, app_id: str, token: str, timeout: int = 15):
        auth_token = token if ":" in token else f"{app_id}:{token}"
        self.session = requests.Session()
        self.session.headers.update({"Authorization": auth_token, "Content-Type": "application/json"})
        self.timeout = timeout

    def request(self, method: str, url: str, **kwargs) -> dict:
        response = self.session.request(method, url, timeout=self.timeout, **kwargs)
        try:
            payload = response.json()
        except ValueError:
            payload = {"s": "error", "message": f"Non-JSON HTTP {response.status_code}: {response.text[:300]}"}
        if response.status_code >= 400:
            payload.setdefault("http_status", response.status_code)
        return payload

    def profile(self) -> dict:
        return self.request("GET", f"{BASE_API}/profile")

    def positions(self) -> dict:
        return self.request("GET", f"{BASE_API}/positions")

    def history_today(self, day: date) -> dict:
        return self.request("GET", f"{BASE_DATA}/history", params={
            "symbol": Config.UNDERLYING,
            "resolution": Config.RESOLUTION,
            "date_format": "1",
            "range_from": day.strftime("%Y-%m-%d"),
            "range_to": day.strftime("%Y-%m-%d"),
            "cont_flag": "0",
        })

    def quote_ltp(self, symbol: str) -> Optional[float]:
        response = self.request("GET", f"{BASE_DATA}/quotes", params={"symbols": symbol})
        for row in response.get("d", []) if isinstance(response, dict) else []:
            if row.get("n") == symbol and row.get("s") == "ok":
                val = row.get("v", {})
                return as_float(val.get("lp", val.get("ltp")))
        return None

    def place_market_order(self, symbol: str, qty: int, side: int, tag: str) -> dict:
        body = {
            "symbol": symbol,
            "qty": int(qty),
            "type": 2,              # Market order (FYERS may apply Market Price Protection)
            "side": int(side),      # 1 buy, -1 sell
            "productType": "INTRADAY",
            "limitPrice": 0,
            "stopPrice": 0,
            "validity": "DAY",
            "disclosedQty": 0,
            "offlineOrder": False,
            "orderTag": tag[:30],
            "isSliceOrder": bool(Config.SLICE_ORDER),
        }
        return self.request("POST", f"{BASE_API}/orders/sync", json=body)

    def order_details(self, order_id: str) -> dict:
        return self.request("GET", f"{BASE_API}/orders", params={"id": order_id})

    def all_orders(self) -> dict:
        return self.request("GET", f"{BASE_API}/orders")

    def cancel_order(self, order_id: str) -> dict:
        # FYERS V3 sync cancel path is /orders/{order_id}/sync.
        return self.request("DELETE", f"{BASE_API}/orders/{order_id}/sync")


def candles_to_bars(response: dict) -> List[dict]:
    rows = response.get("candles", []) if isinstance(response, dict) else []
    bars: List[dict] = []
    for row in rows:
        if not isinstance(row, (list, tuple)) or len(row) < 5:
            continue
        try:
            stamp = datetime.fromtimestamp(float(row[0]), tz=timezone.utc).astimezone(IST).replace(second=0, microsecond=0)
            vals = [float(row[1]), float(row[2]), float(row[3]), float(row[4])]
            if not all(pd.notna(x) for x in vals):
                continue
            volume = float(row[5]) if len(row) >= 6 and as_float(row[5]) is not None else 0.0
            bars.append({"timestamp": stamp, "open": vals[0], "high": vals[1], "low": vals[2], "close": vals[3], "volume": volume})
        except (TypeError, ValueError, OverflowError, OSError):
            continue
    return sorted({b["timestamp"]: b for b in bars}.values(), key=lambda x: x["timestamp"])


class StrategyEngine:
    """Causal per-bar implementation of the uploaded backtest's breakout/retest rules."""

    def __init__(self):
        self.bars: List[dict] = []
        self.seen = set()
        self.opening_high: Optional[float] = None
        self.opening_low: Optional[float] = None
        self.opening_index: Optional[int] = None
        self.state = "WAIT_BREAKOUT"
        self.direction: Optional[str] = None  # BUY for ORH breakout; SELL for ORL breakdown
        self.level_name: Optional[str] = None
        self.level: Optional[float] = None
        self.breakout_index: Optional[int] = None
        self.retest_index: Optional[int] = None
        self.c1_index: Optional[int] = None
        self.consumed_retests = set()

    def _reset_setup(self) -> None:
        self.state = "WAIT_BREAKOUT"
        self.direction = None
        self.level_name = None
        self.level = None
        self.breakout_index = None
        self.retest_index = None
        self.c1_index = None

    def add_closed_bar(self, bar: dict, evaluate: bool = True) -> Optional[dict]:
        ts = bar["timestamp"].astimezone(IST).replace(second=0, microsecond=0)
        key = ts.isoformat()
        if key in self.seen or ts.time() < Config.SESSION_START or ts.time() > Config.SESSION_END:
            return None
        # This engine is intended to run for one market session per process. Guard against
        # accidental day carry-over if a caller keeps it alive over a date boundary.
        if self.bars and self.bars[-1]["timestamp"].date() != ts.date():
            self.bars.clear()
            self.opening_high = self.opening_low = None
            self.opening_index = None
            self._reset_setup()
            self.consumed_retests.clear()
        bar = dict(bar)
        bar["timestamp"] = ts
        self.seen.add(key)
        self.bars.append(bar)
        self.bars.sort(key=lambda b: b["timestamp"])
        idx = next(i for i, b in enumerate(self.bars) if b["timestamp"] == ts)
        if ts.time() == Config.OPENING_CANDLE_TIME:
            self.opening_high = float(bar["high"])
            self.opening_low = float(bar["low"])
            self.opening_index = idx
            self._reset_setup()
            logging.info("Opening range fixed from 09:15 candle: ORH=%.2f ORL=%.2f",
                         self.opening_high, self.opening_low)
        if not evaluate or self.opening_high is None or self.opening_low is None or self.opening_index is None:
            return None
        # Signals must be processed in chronological order; late historical bars are handled by warmup/resync.
        if self.bars[-1]["timestamp"] != ts:
            return None
        return self._evaluate_latest()

    @staticmethod
    def _body(bar: dict) -> float:
        return abs(float(bar["close"]) - float(bar["open"]))

    @staticmethod
    def _is_red(bar: dict) -> bool:
        return float(bar["close"]) < float(bar["open"])

    @staticmethod
    def _is_green(bar: dict) -> bool:
        return float(bar["close"]) > float(bar["open"])

    def _make_signal(self, confirm_index: int, c1_index: int, c2_index: int,
                     c3_index: Optional[int], confirmation: str) -> Optional[dict]:
        if self.retest_index is None or self.breakout_index is None or self.direction is None or self.level is None:
            return None
        bar = self.bars[confirm_index]
        signal_dt = bar["timestamp"] + timedelta(minutes=1)
        if signal_dt.time() < Config.ENTRY_START or signal_dt.time() > Config.ENTRY_END:
            return None
        if self.direction == "BUY" and float(bar["close"]) <= self.level:
            return None
        if self.direction == "SELL" and float(bar["close"]) >= self.level:
            return None
        retest_key = (self.level_name, self.bars[self.retest_index]["timestamp"].isoformat())
        if retest_key in self.consumed_retests:
            return None
        self.consumed_retests.add(retest_key)
        return {
            "date": signal_dt.date().isoformat(),
            "signal_time": signal_dt.isoformat(),
            "bar_time": bar["timestamp"].isoformat(),
            "direction": self.direction,
            "option_type": "CE" if self.direction == "BUY" else "PE",
            "level_name": self.level_name,
            "level": round(float(self.level), 2),
            "orh": round(float(self.opening_high), 2),
            "orl": round(float(self.opening_low), 2),
            "retest_time": self.bars[self.retest_index]["timestamp"].isoformat(),
            "departure_time": self.bars[self.breakout_index]["timestamp"].isoformat(),
            "breakout_time": self.bars[self.breakout_index]["timestamp"].isoformat(),
            "confirmation": confirmation,
            "c1_time": self.bars[c1_index]["timestamp"].isoformat(),
            "c2_time": self.bars[c2_index]["timestamp"].isoformat(),
            "c3_time": self.bars[c3_index]["timestamp"].isoformat() if c3_index is not None else "",
            "detail": (f"{confirmation}; level={self.level_name} {self.level:.2f}; "
                       f"breakout={self.bars[self.breakout_index]['timestamp'].strftime('%H:%M')}; "
                       f"retest={self.bars[self.retest_index]['timestamp'].strftime('%H:%M')}; "
                       f"C1={self.bars[c1_index]['timestamp'].strftime('%H:%M')}; "
                       f"C2={self.bars[c2_index]['timestamp'].strftime('%H:%M')}")
        }

    def _fallback_c123_valid(self, c1_idx: int, c2_idx: int, c3_idx: int) -> bool:
        c1, c2, c3 = self.bars[c1_idx], self.bars[c2_idx], self.bars[c3_idx]
        body1, body2, body3 = self._body(c1), self._body(c2), self._body(c3)
        c3_range = float(c3["high"]) - float(c3["low"])
        if c3_range <= 0 or body3 < Config.HEALTHY_BODY_FRAC * c3_range:
            return False
        if self.direction == "BUY":
            if not (self._is_red(c1) and self._is_green(c2) and self._is_green(c3)):
                return False
            progression = float(c1["close"]) < float(c2["close"]) < float(c3["close"])
        else:
            if not (self._is_green(c1) and self._is_red(c2) and self._is_red(c3)):
                return False
            progression = float(c1["close"]) > float(c2["close"]) > float(c3["close"])
        body_ratio_ok = body1 < Config.BODY_RATIO_MAX * body2
        return bool(progression or body_ratio_ok)

    def _evaluate_latest(self) -> Optional[dict]:
        i = len(self.bars) - 1
        if i < 1 or self.opening_index is None or i <= self.opening_index:
            return None
        bar = self.bars[i]
        ts = bar["timestamp"]
        close, high, low = float(bar["close"]), float(bar["high"]), float(bar["low"])
        orh, orl = float(self.opening_high), float(self.opening_low)

        # 1) Wait for a completed candle to close outside the opening range.
        if self.state == "WAIT_BREAKOUT":
            if ts.time() < Config.ENTRY_START or ts.time() > Config.ENTRY_END:
                return None
            if close > orh:
                self.direction, self.level_name, self.level = "BUY", "ORH", orh
                self.breakout_index = i
                self.state = "WAIT_RETEST"
                logging.info("ORH breakout close at %s: close=%.2f ORH=%.2f", ts.strftime("%H:%M"), close, orh)
            elif close < orl:
                self.direction, self.level_name, self.level = "SELL", "ORL", orl
                self.breakout_index = i
                self.state = "WAIT_RETEST"
                logging.info("ORL breakdown close at %s: close=%.2f ORL=%.2f", ts.strftime("%H:%M"), close, orl)
            return None

        # 2) Wait for a distinct later candle whose OHLC intersects the exact level.
        if self.state == "WAIT_RETEST":
            if self.breakout_index is None or i <= self.breakout_index:
                return None
            assert self.level is not None and self.direction is not None
            touched = low <= self.level <= high
            if touched:
                self.retest_index = i
                self.state = "WAIT_C1"
                logging.info("Exact %s retest at %s: level=%.2f; close=%.2f",
                             self.level_name, ts.strftime("%H:%M"), self.level, close)
            elif self.direction == "BUY" and close < orh:
                logging.info("ORH breakout invalidated without exact touch at %s; waiting for a new breakout.", ts.strftime("%H:%M"))
                self._reset_setup()
            elif self.direction == "SELL" and close > orl:
                logging.info("ORL breakdown invalidated without exact touch at %s; waiting for a new breakdown.", ts.strftime("%H:%M"))
                self._reset_setup()
            return None

        # Retest candle must finish before C1 starts, matching the uploaded backtest.
        if self.state == "WAIT_C1":
            if self.retest_index is None or i != self.retest_index + 1:
                self._reset_setup()
                return None
            self.c1_index = i
            self.state = "WAIT_C2"
            return None

        # 3) If C2 body >= C1 body, confirm at C2 close; C3 is ignored completely.
        if self.state == "WAIT_C2":
            if self.c1_index is None or self.retest_index is None or i != self.c1_index + 1:
                self._reset_setup()
                return None
            c1_idx, c2_idx = self.c1_index, i
            c1, c2 = self.bars[c1_idx], self.bars[c2_idx]
            body1, body2 = self._body(c1), self._body(c2)
            expected_pair = (self._is_red(c1) and self._is_green(c2)) if self.direction == "BUY" else (self._is_green(c1) and self._is_red(c2))
            if not expected_pair:
                self._reset_setup()
                return None
            if body2 >= body1:
                # The user requested that C3 be ignored for a strong/equal C2.
                valid_close = float(c2["close"]) > float(self.level) if self.direction == "BUY" else float(c2["close"]) < float(self.level)
                signal = self._make_signal(c2_idx, c1_idx, c2_idx, None,
                    "BULLISH_C12_STRONGER_SECOND" if self.direction == "BUY" else "BEARISH_C12_STRONGER_SECOND") if valid_close else None
                self._reset_setup()
                return signal
            self.state = "WAIT_C3"
            return None

        # 4) C2 body is smaller than C1: apply the uploaded file's strict C1/C2/C3 fallback.
        if self.state == "WAIT_C3":
            if self.c1_index is None or i != self.c1_index + 2:
                self._reset_setup()
                return None
            c1_idx, c2_idx, c3_idx = self.c1_index, self.c1_index + 1, i
            valid = self._fallback_c123_valid(c1_idx, c2_idx, c3_idx)
            valid_close = float(bar["close"]) > float(self.level) if self.direction == "BUY" else float(bar["close"]) < float(self.level)
            signal = self._make_signal(c3_idx, c1_idx, c2_idx, c3_idx,
                "BULLISH_C123_FALLBACK" if self.direction == "BUY" else "BEARISH_C123_FALLBACK") if valid and valid_close else None
            self._reset_setup()
            return signal

        self._reset_setup()
        return None

class SpotFeed:
    """Aggregate live NIFTY index ticks into minute bars; reconnect/gap recovery via REST."""
    def __init__(self, api: FyersREST, engine: StrategyEngine):
        self.api = api
        self.engine = engine
        self.events: queue.Queue = queue.Queue()
        self.lock = threading.RLock()
        self.latest_spot: Optional[float] = None
        self.last_tick_wall: Optional[float] = None
        self.current_bar: Optional[dict] = None
        self.risk_high: Optional[float] = None
        self.risk_low: Optional[float] = None
        self.socket = None
        self.connected = False

    def seed_current_bar(self, bar: dict) -> None:
        with self.lock:
            if self.current_bar is None or bar["timestamp"] >= self.current_bar["timestamp"]:
                self.current_bar = dict(bar)

    def _tick_time(self, payload: dict) -> datetime:
        for field in ("exch_feed_time", "last_traded_time", "timestamp", "tt"):
            value = payload.get(field)
            try:
                if value not in (None, "", 0, "0"):
                    stamp = float(value)
                    if stamp > 10_000_000_000:
                        stamp /= 1000.0
                    return datetime.fromtimestamp(stamp, timezone.utc).astimezone(IST)
            except (TypeError, ValueError, OverflowError, OSError):
                pass
        return now_ist()

    def on_message(self, message: Any) -> None:
        if isinstance(message, str):
            try:
                message = json.loads(message)
            except ValueError:
                return
        if isinstance(message, list):
            for item in message:
                self.on_message(item)
            return
        if not isinstance(message, dict):
            return
        symbol = message.get("symbol")
        msg_type = message.get("type")
        if symbol and symbol != Config.UNDERLYING:
            return
        if msg_type not in ("if", "sf", None):
            return
        ltp = as_float(message.get("ltp"))
        if ltp is None or ltp <= 0:
            return
        ts = self._tick_time(message).replace(second=0, microsecond=0)
        if ts.time() < Config.SESSION_START or ts.time() > Config.SESSION_END:
            with self.lock:
                self.latest_spot = ltp
                self.last_tick_wall = time.time()
            return
        with self.lock:
            self.latest_spot = ltp
            self.last_tick_wall = time.time()
            self.risk_high = ltp if self.risk_high is None else max(self.risk_high, ltp)
            self.risk_low = ltp if self.risk_low is None else min(self.risk_low, ltp)
            if self.current_bar is None:
                self.current_bar = {"timestamp": ts, "open": ltp, "high": ltp, "low": ltp, "close": ltp, "volume": 0.0}
                return
            cur_ts = self.current_bar["timestamp"]
            if ts < cur_ts:
                return  # Ignore out-of-order ticks.
            if ts == cur_ts:
                self.current_bar["high"] = max(float(self.current_bar["high"]), ltp)
                self.current_bar["low"] = min(float(self.current_bar["low"]), ltp)
                self.current_bar["close"] = ltp
                return
            # A clean next-minute transition closes the current candle.
            gap_minutes = int((ts - cur_ts).total_seconds() // 60)
            if gap_minutes == 1:
                self.events.put(("bar", dict(self.current_bar)))
            else:
                # Avoid fabricating bars after a disconnect; recover them from REST history.
                self.events.put(("resync", ts))
            self.current_bar = {"timestamp": ts, "open": ltp, "high": ltp, "low": ltp, "close": ltp, "volume": 0.0}

    def on_connect(self) -> None:
        self.connected = True
        logging.info("FYERS market-data socket connected; subscribing to %s", Config.UNDERLYING)
        try:
            self.socket.subscribe(symbols=[Config.UNDERLYING], data_type="SymbolUpdate")
            self.socket.keep_running()
        except Exception as exc:
            logging.exception("Socket subscription failed: %s", exc)

    def on_error(self, message: Any) -> None:
        self.connected = False
        logging.error("FYERS market-data socket error: %s", message)

    def on_close(self, message: Any) -> None:
        self.connected = False
        logging.warning("FYERS market-data socket closed: %s", message)

    def start(self, app_id: str, token: str) -> None:
        try:
            from fyers_apiv3.FyersWebsocket import data_ws
        except ImportError as exc:
            raise RuntimeError("Install FYERS SDK first: pip install --upgrade fyers-apiv3") from exc
        auth_token = token if ":" in token else f"{app_id}:{token}"
        self.socket = data_ws.FyersDataSocket(
            access_token=auth_token,
            log_path="",
            litemode=False,
            write_to_file=False,
            reconnect=True,
            on_connect=self.on_connect,
            on_close=self.on_close,
            on_error=self.on_error,
            on_message=self.on_message,
        )
        threading.Thread(target=self.socket.connect, daemon=True, name="FyersDataSocket").start()

    def risk_snapshot(self) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[float]]:
        with self.lock:
            latest = self.latest_spot
            high = self.risk_high
            low = self.risk_low
            age = (time.time() - self.last_tick_wall) if self.last_tick_wall else None
            self.risk_high = latest
            self.risk_low = latest
            return latest, high, low, age


class LiveBot:
    def __init__(self):
        self.cfg = Config()
        self.script_dir = Path(__file__).resolve().parent
        credential_dirs = [self.script_dir, self.script_dir.parent]
        credential_dir = next(
            (folder for folder in credential_dirs
             if (folder / "fyers_appid.txt").is_file() and (folder / "fyers_token.txt").is_file()),
            None,
        )
        if credential_dir is None:
            expected = "\n".join(f"  - {folder}" for folder in credential_dirs)
            raise FileNotFoundError(
                "Could not find both fyers_appid.txt and fyers_token.txt. "
                f"Place them beside this engine or one folder above:\n{expected}. "
                "Never paste credentials into chat."
            )
        app_path = credential_dir / "fyers_appid.txt"
        token_path = credential_dir / "fyers_token.txt"
        self.app_id = app_path.read_text(encoding="utf-8").strip()
        self.token = token_path.read_text(encoding="utf-8").strip()
        if not self.app_id or not self.token:
            raise ValueError("FYERS App ID or access token file is empty.")

        env_ok = os.getenv(self.cfg.LIVE_CONFIRM_ENV, "").strip().upper() == "YES"
        self.live_mode = bool(not self.cfg.DRY_RUN and self.cfg.LIVE_TRADING_ENABLED and env_ok)
        if not self.live_mode and not self.cfg.DRY_RUN:
            logging.warning("Live flags/environment confirmation are incomplete; running in DRY-RUN mode.")

        self.api = FyersREST(self.app_id, self.token, self.cfg.API_TIMEOUT_SECS)
        self.engine = StrategyEngine()
        self.feed = SpotFeed(self.api, self.engine)
        self.contracts: List[dict] = []
        self.open_trade: Optional[dict] = None
        self.trades_today = 0
        self.day = now_ist().date()
        self.halted = False
        self.pending_order: Optional[dict] = None
        self.last_quote_fallback = 0.0
        self.last_pending_check = 0.0
        self.sequence = 0
        self.force_exit_reason: Optional[str] = None
        self.exit_retry_at = 0.0
        self.exit_attempts = 0
        self.events_path = self.script_dir / self.cfg.EVENTS_CSV
        self.state_path = self.script_dir / self.cfg.STATE_FILE
        self.log_path = self.script_dir / self.cfg.LOG_FILE
        self.stop_requested = threading.Event()
        self._load_state()

    def request_stop(self) -> None:
        """Ask the polling loop to stop safely, exiting any open trade first."""
        self.stop_requested.set()
        logging.warning("Graceful stop requested by launcher: no new entries will be opened; any open position will be closed/reconciled before stopping.")

    def _close_market_data_socket(self) -> None:
        """Close the market-data socket if the SDK exposes a close method."""
        sock = getattr(self.feed, "socket", None)
        if sock is None:
            return
        for method_name in ("close_connection", "close"):
            method = getattr(sock, method_name, None)
            if callable(method):
                try:
                    method()
                    logging.info("Market-data socket closed (%s).", method_name)
                except Exception as exc:
                    logging.warning("Could not close market-data socket via %s: %s", method_name, exc)
                return

    def record(self, status: str, **fields: Any) -> None:
        row = {"time": now_ist().isoformat(timespec="seconds"), "mode": "LIVE" if self.live_mode else "DRY_RUN", "status": status}
        row.update(fields)
        exists = self.events_path.exists()
        columns = ["time", "mode", "status", "direction", "option_symbol", "qty", "spot", "orh", "orl", "level_name", "level", "confirmation", "order_id", "detail"]
        with self.events_path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
            if not exists:
                writer.writeheader()
            writer.writerow(row)
        logging.info("%s | %s", status, json.dumps(row, default=str, ensure_ascii=False))

    def _load_state(self) -> None:
        if not self.state_path.exists():
            return
        try:
            data = json.loads(self.state_path.read_text(encoding="utf-8"))
            if data.get("date") == self.day.isoformat():
                self.trades_today = int(data.get("trades_today", 0))
                if not self.live_mode:
                    # A virtual dry-run position cannot be trusted after process restart.
                    self.open_trade = None
                else:
                    self.open_trade = data.get("open_trade")
                    self.pending_order = data.get("pending_order")
                    self.force_exit_reason = data.get("force_exit_reason")
                    self.exit_attempts = int(data.get("exit_attempts", 0))
        except Exception as exc:
            logging.warning("Could not load live state file; starting without saved state: %s", exc)

    def _save_state(self) -> None:
        payload = {
            "date": self.day.isoformat(),
            "trades_today": self.trades_today,
            "open_trade": self.open_trade,
            "pending_order": self.pending_order,
            "force_exit_reason": self.force_exit_reason,
            "exit_attempts": self.exit_attempts,
            "updated_at": now_ist().isoformat(timespec="seconds"),
        }
        tmp = self.state_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        os.replace(tmp, self.state_path)

    def _startup_reconcile(self) -> None:
        profile = self.api.profile()
        if not isinstance(profile, dict) or profile.get("s") != "ok":
            raise ConnectionError(f"FYERS profile/authentication check failed: {profile}")
        logging.info("FYERS authentication validated for profile. Do not log or share tokens.")

        if not self.live_mode:
            return
        # Under the post-April-2026 setup, API order placement requires the newly
        # activated order-enabled app and its registered/whitelisted static IP.
        suffix = self.app_id.split("-")[-1]
        if suffix != "200":
            raise RuntimeError("Live mode safety check: FYERS now requires the activated API trading app for the post-01-Apr-2026 framework (FYERS documents new App IDs ending in 200). Review your app/static-IP/permissions before enabling live orders.")
        if self.pending_order:
            self.halted = True
            logging.critical("A previous order is unresolved. Bot will start only in reconciliation mode and will not submit new entries.")
        resp = self.api.positions()
        if not isinstance(resp, dict) or resp.get("s") != "ok":
            raise RuntimeError(f"Could not verify current FYERS positions before live trading: {resp}")
        open_index_options = []
        for pos in resp.get("netPositions", []):
            symbol = str(pos.get("symbol", "")).upper()
            net_qty = int(pos.get("netQty", 0) or 0)
            if net_qty and symbol.startswith("NSE:NIFTY") and symbol.endswith(("CE", "PE")):
                open_index_options.append(pos)
        if not open_index_options:
            self.open_trade = None
            self._save_state()
            return
        saved = self.open_trade if isinstance(self.open_trade, dict) else None
        pending_symbol = self.pending_order.get("symbol") if isinstance(self.pending_order, dict) else None
        matched = next((p for p in open_index_options if saved and p.get("symbol") == saved.get("option_symbol")), None)
        if not matched and pending_symbol:
            matched = next((p for p in open_index_options if p.get("symbol") == pending_symbol), None)
            if matched and self.pending_order.get("kind") == "ENTRY":
                sig = self.pending_order.get("signal", {})
                self.open_trade = {
                    "date": self.day.isoformat(), "direction": sig.get("direction", "BUY"),
                    "option_symbol": pending_symbol, "qty": abs(int(matched.get("netQty", 0))),
                    "entry_spot": float(self.pending_order.get("spot_entry", 0.0)),
                    "stop_level": float(self.pending_order.get("stop_level", 0.0)),
                    "target_level": float(self.pending_order.get("target_level", 0.0)),
                    "entry_time": now_ist().isoformat(), "entry_order_id": str(self.pending_order.get("order_id") or "UNRESOLVED"),
                    "level_name": sig.get("level_name", ""), "level": sig.get("level", 0.0),
                    "confirmation": sig.get("confirmation", ""), "signal_time": sig.get("signal_time", "")
                }
        if not matched:
            raise RuntimeError(f"Existing NIFTY option position(s) found but not matched to this bot's saved trade/order: {open_index_options}. Bot will not trade until reconciled.")
        if self.open_trade:
            self.open_trade["qty"] = abs(int(matched.get("netQty", 0)))
            self._save_state()
            logging.warning("Restored saved open position %s; spot SL/TP remains client-side and needs continuous monitoring.", self.open_trade.get("option_symbol"))

    def load_symbol_master(self) -> None:
        response = requests.get(self.cfg.SYMBOL_MASTER_URL, timeout=self.cfg.API_TIMEOUT_SECS)
        response.raise_for_status()
        master = response.json()
        if not isinstance(master, dict):
            raise ValueError("FYERS NSE_FO symbol master response is not a JSON object.")
        today = now_ist().date()
        contracts = []
        for key, value in master.items():
            contract = parse_master_contract(str(key), value)
            if contract and today <= contract["expiry_date"] <= today + timedelta(days=10):
                contracts.append(contract)
        if not contracts:
            raise RuntimeError("No future NIFTY CE/PE contracts could be resolved from the FYERS symbol master.")
        self.contracts = contracts
        logging.info("Loaded %s active NIFTY option contracts from the FYERS symbol master.", len(contracts))

    def resolve_option(self, spot: float, direction: str) -> dict:
        today = now_ist().date()
        available = [x for x in self.contracts if x["expiry_date"] >= today]
        if not available:
            raise RuntimeError("No unexpired option contracts in the loaded FYERS symbol master.")
        nearest_expiry = min(x["expiry_date"] for x in available)
        chain = [x for x in available if x["expiry_date"] == nearest_expiry]
        option_type = "CE" if direction == "BUY" else "PE"
        chain = [x for x in chain if x["option_type"] == option_type]
        if not chain:
            raise RuntimeError(f"No {option_type} contracts for expiry {nearest_expiry}.")
        atm = int(round(float(spot) / self.cfg.STRIKE_STEP) * self.cfg.STRIKE_STEP)
        target = atm - self.cfg.STRIKE_STEP if direction == "BUY" else atm + self.cfg.STRIKE_STEP
        exact = [x for x in chain if abs(x["strike"] - target) < 0.001]
        if exact:
            chosen = exact[0]
        elif direction == "BUY":
            eligible = [x for x in chain if x["strike"] < spot]
            deeper = [x for x in eligible if x["strike"] <= target]
            if not (deeper or eligible):
                raise RuntimeError(f"No ITM CE available below spot={spot}.")
            chosen = max(deeper or eligible, key=lambda x: x["strike"])
        else:
            eligible = [x for x in chain if x["strike"] > spot]
            deeper = [x for x in eligible if x["strike"] >= target]
            if not (deeper or eligible):
                raise RuntimeError(f"No ITM PE available above spot={spot}.")
            chosen = min(deeper or eligible, key=lambda x: x["strike"])
        if chosen.get("lot_size") and self.cfg.QUANTITY % int(chosen["lot_size"]) != 0:
            raise RuntimeError(f"Configured quantity {self.cfg.QUANTITY} is not a multiple of master lot size {chosen['lot_size']} for {chosen['symbol']}.")
        return chosen

    def warmup_from_history(self) -> None:
        today = now_ist().date()
        response = self.api.history_today(today)
        if not isinstance(response, dict) or response.get("s") != "ok":
            raise RuntimeError(f"Could not load today's NIFTY history to initialize opening range: {response}")
        now_minute = now_ist().replace(second=0, microsecond=0)
        past = []
        current = None
        for bar in candles_to_bars(response):
            ts = bar["timestamp"]
            if ts.date() != today or ts.time() < self.cfg.SESSION_START or ts.time() > self.cfg.SESSION_END:
                continue
            if ts < now_minute:
                past.append(bar)
            elif ts == now_minute:
                current = bar
        # Replay historical closed bars to seed range/state. Historical signals are logged
        # but not traded; they consume retests so the bot won't enter a stale setup on start.
        for bar in past:
            event = self.engine.add_closed_bar(bar, evaluate=True)
            if event:
                self.record("WARMUP_SIGNAL_IGNORED", direction=event["direction"], orh=event["orh"], orl=event["orl"],
                            level_name=event["level_name"], level=event["level"], confirmation=event["confirmation"],
                            detail="Signal occurred before this process finished starting; not traded.")
        if current:
            self.feed.seed_current_bar(current)
        if self.engine.opening_high is not None:
            logging.info("Today's opening range initialized from 09:15 candle: ORH=%.2f ORL=%.2f",
                         self.engine.opening_high, self.engine.opening_low)
        else:
            logging.info("No 09:15 candle available yet; opening range will be fixed when the first live 09:15 bar closes.")

    def _make_tag(self) -> str:
        self.sequence += 1
        return f"FCSR{now_ist().strftime('%y%m%d%H%M%S')}{self.sequence:02d}"[:30]

    def _order_by_id(self, order_id: str) -> Optional[dict]:
        response = self.api.order_details(order_id)
        rows = response.get("orderBook", []) if isinstance(response, dict) else []
        if rows:
            return next((row for row in rows if str(row.get("id")) == str(order_id)), rows[0])
        return None

    def _find_order_by_tag(self, tag: str) -> Optional[dict]:
        try:
            response = self.api.all_orders()
        except Exception:
            return None
        rows = response.get("orderBook", []) if isinstance(response, dict) else []
        for row in rows:
            row_tag = str(row.get("orderTag", ""))
            if row_tag == tag or row_tag.endswith(tag) or tag in row_tag:
                return row
        return None

    def _place_and_wait(self, symbol: str, qty: int, side: int, tag: str) -> dict:
        try:
            response = self.api.place_market_order(symbol, qty, side, tag)
        except Exception as exc:
            # The server may have received an order even if the client timed out.
            order = None
            for _ in range(3):
                order = self._find_order_by_tag(tag)
                if order:
                    break
                time.sleep(0.5)
            if order and order.get("id"):
                order_id = str(order["id"])
            else:
                raise OrderUnresolved(f"Order request raised an exception and cannot yet be reconciled: {exc}; do not retry blindly.",
                                      order_id=None, symbol=symbol, qty=qty, side=side, tag=tag) from exc
        else:
            if not isinstance(response, dict):
                raise OrderUnresolved(f"Unexpected order response: {response}; do not retry blindly.",
                                      order_id=None, symbol=symbol, qty=qty, side=side, tag=tag)
            order_id = response.get("id")
            code = response.get("code")
            # FYERS documents code 201 as sent but not acknowledged; find by tag, never duplicate it.
            ambiguous = code == 201 or (response.get("s") == "ok" and not order_id)
            if not order_id and ambiguous:
                order = None
                for _ in range(3):
                    order = self._find_order_by_tag(tag)
                    if order and order.get("id"):
                        order_id = str(order["id"])
                        break
                    time.sleep(0.5)
                if not order_id:
                    raise OrderUnresolved(f"FYERS order code={code} but order ID was not returned; unable to resolve by orderTag. Do not retry.",
                                          order_id=None, symbol=symbol, qty=qty, side=side, tag=tag, details=response)
            if response.get("s") != "ok" and not (code == 201 and order_id):
                raise RuntimeError(f"FYERS rejected order request: {response}")
            if not order_id:
                raise RuntimeError(f"FYERS rejected order request without an order ID: {response}")
            order_id = str(order_id)
            logging.info("Order request acknowledged: id=%s code=%s message=%s", order_id, code, response.get("message", ""))

        deadline = time.time() + self.cfg.ORDER_STATUS_WAIT_SECS
        last_order = None
        while time.time() < deadline:
            last_order = self._order_by_id(str(order_id))
            if last_order:
                status = int(last_order.get("status", -1))
                filled = int(last_order.get("filledQty", 0) or 0)
                qty_req = int(last_order.get("qty", qty) or qty)
                remaining = int(last_order.get("remainingQuantity", max(0, qty_req - filled)) or 0)
                if status == 2 and filled >= qty_req and remaining == 0:
                    return {"order_id": str(order_id), "filled_qty": filled,
                            "traded_price": as_float(last_order.get("tradedPrice")), "order": last_order}
                if status in (1, 5, 7):
                    raise RuntimeError(f"Order {order_id} ended without full fill (status={status}, filled={filled}/{qty_req}, message={last_order.get('message')}).")
            time.sleep(self.cfg.ORDER_STATUS_POLL_SECS)
        raise OrderUnresolved(f"Order {order_id} not confirmed fully filled within {self.cfg.ORDER_STATUS_WAIT_SECS}s; last_order={last_order}",
                              order_id=str(order_id), symbol=symbol, qty=qty, side=side, tag=tag, details=last_order)

    def _current_position_qty(self, symbol: str) -> Optional[int]:
        response = self.api.positions()
        if not isinstance(response, dict) or response.get("s") != "ok":
            return None
        for pos in response.get("netPositions", []):
            if pos.get("symbol") == symbol and pos.get("productType") == "INTRADAY":
                return int(pos.get("netQty", 0) or 0)
        return 0

    def handle_signal(self, signal: dict) -> None:
        if self.stop_requested.is_set():
            self.record("SKIP_LAUNCHER_STOP_REQUESTED", direction=signal.get("direction"),
                        level_name=signal.get("level_name"), level=signal.get("level"),
                        detail="Launcher requested shutdown; new entries are disabled.")
            return
        now = now_ist()
        signal_dt = datetime.fromisoformat(signal["signal_time"])
        if signal_dt.tzinfo is None:
            signal_dt = IST.localize(signal_dt)
        age = (now - signal_dt.astimezone(IST)).total_seconds()
        if age > self.cfg.STALE_SIGNAL_MAX_SECS:
            self.record("SKIP_STALE_SIGNAL", direction=signal["direction"], orh=signal["orh"], orl=signal["orl"],
                        level_name=signal["level_name"], level=signal["level"], confirmation=signal["confirmation"],
                        detail=f"Signal {age:.1f}s old; not entering late.")
            return
        # Validate the intended entry-bar timestamp, not wall-clock seconds after its open.
        # A 15:00:00 entry signal may arrive a few seconds into the 15:00 minute.
        if signal_dt.time() < self.cfg.ENTRY_START or signal_dt.time() > self.cfg.ENTRY_END:
            self.record("SKIP_OUTSIDE_ENTRY_WINDOW", direction=signal["direction"], level_name=signal["level_name"],
                        level=signal["level"], confirmation=signal["confirmation"], detail="Signal's intended entry-bar timestamp is outside the entry window.")
            return
        if self.halted or self.pending_order:
            self.record("SKIP_BOT_HALTED", direction=signal["direction"], level_name=signal["level_name"], level=signal["level"],
                        detail="A previous order/position state needs reconciliation.")
            return
        if self.open_trade:
            self.record("SKIP_POSITION_ALREADY_OPEN", direction=signal["direction"], level_name=signal["level_name"], level=signal["level"],
                        detail="One position at a time; wait for previous position to exit.")
            return
        if self.trades_today >= self.cfg.MAX_TRADES_PER_DAY:
            self.record("SKIP_DAILY_TRADE_LIMIT", direction=signal["direction"], level_name=signal["level_name"], level=signal["level"],
                        detail=f"Reached daily max {self.cfg.MAX_TRADES_PER_DAY}.")
            return
        spot = self.feed.latest_spot
        # A REST quote is used if the data socket has not delivered a fresh tick yet.
        if spot is None or (self.feed.last_tick_wall and time.time() - self.feed.last_tick_wall > 3.0):
            spot = self.api.quote_ltp(self.cfg.UNDERLYING)
        if spot is None or spot <= 0:
            self.record("SKIP_NO_SPOT_PRICE", direction=signal["direction"], detail="No usable NIFTY spot LTP for risk levels.")
            return
        try:
            contract = self.resolve_option(float(spot), signal["direction"])
        except Exception as exc:
            self.record("SKIP_NO_OPTION_CONTRACT", direction=signal["direction"], spot=spot, orh=signal["orh"], orl=signal["orl"],
                        level_name=signal["level_name"], level=signal["level"], confirmation=signal["confirmation"], detail=str(exc))
            return

        if signal["direction"] == "BUY":
            stop = float(spot) - self.cfg.STOP_LOSS_SPOT_POINTS
            target = float(spot) + self.cfg.TAKE_PROFIT_SPOT_POINTS
        else:
            stop = float(spot) + self.cfg.STOP_LOSS_SPOT_POINTS
            target = float(spot) - self.cfg.TAKE_PROFIT_SPOT_POINTS

        self.record("SIGNAL_CONFIRMED", direction=signal["direction"], option_symbol=contract["symbol"],
                    qty=self.cfg.QUANTITY, spot=spot, orh=signal["orh"], orl=signal["orl"], level_name=signal["level_name"],
                    level=signal["level"], confirmation=signal["confirmation"], detail=signal["detail"])

        if self.live_mode:
            tag = self._make_tag()
            try:
                result = self._place_and_wait(contract["symbol"], self.cfg.QUANTITY, 1, tag)
            except OrderUnresolved as exc:
                self.pending_order = {"kind": "ENTRY", "order_id": exc.order_id, "symbol": contract["symbol"],
                                      "qty": self.cfg.QUANTITY, "side": 1, "tag": tag, "signal": signal,
                                      "spot_entry": float(spot), "stop_level": stop, "target_level": target,
                                      "started_at": now_ist().isoformat()}
                self.halted = True
                self._save_state()
                self.record("CRITICAL_ORDER_UNRESOLVED", direction=signal["direction"], option_symbol=contract["symbol"],
                            qty=self.cfg.QUANTITY, spot=spot, level_name=signal["level_name"], level=signal["level"],
                            confirmation=signal["confirmation"], order_id=exc.order_id, detail=str(exc))
                return
            except Exception as exc:
                self.record("ORDER_REJECTED_OR_FAILED", direction=signal["direction"], option_symbol=contract["symbol"],
                            qty=self.cfg.QUANTITY, spot=spot, level_name=signal["level_name"], level=signal["level"],
                            confirmation=signal["confirmation"], detail=str(exc))
                return
            actual_qty = int(result["filled_qty"])
            order_id = result["order_id"]
        else:
            actual_qty = self.cfg.QUANTITY
            order_id = "DRY_RUN"

        self.open_trade = {
            "date": now_ist().date().isoformat(), "direction": signal["direction"],
            "option_symbol": contract["symbol"], "qty": actual_qty,
            "entry_spot": float(spot), "stop_level": float(stop), "target_level": float(target),
            "entry_time": now_ist().isoformat(), "entry_order_id": order_id,
            "level_name": signal["level_name"], "level": signal["level"],
            "confirmation": signal["confirmation"], "signal_time": signal["signal_time"],
        }
        self.trades_today += 1
        self._save_state()
        self.feed.risk_snapshot()  # Reset the tick excursion window at entry.
        self.record("ENTRY_FILLED" if self.live_mode else "DRY_RUN_ENTRY", direction=signal["direction"],
                    option_symbol=contract["symbol"], qty=actual_qty, spot=spot, orh=signal["orh"], orl=signal["orl"],
                    level_name=signal["level_name"], level=signal["level"], confirmation=signal["confirmation"],
                    order_id=order_id, detail=f"spot SL={stop:.2f}, target={target:.2f}; entry acknowledged and fill status verified" if self.live_mode else f"virtual position; spot SL={stop:.2f}, target={target:.2f}")

    def _close_position(self, reason: str, spot: float) -> None:
        if not self.open_trade:
            return
        if self.pending_order:
            return
        if self.exit_retry_at and time.time() < self.exit_retry_at:
            return
        if self.exit_attempts >= 3:
            logging.critical("Exit attempts exhausted for %s. Position remains open; close it manually in FYERS if necessary.", self.open_trade.get("option_symbol"))
            return
        trade = self.open_trade
        symbol = trade["option_symbol"]
        qty = int(trade["qty"])
        order_id = "DRY_RUN"
        if self.live_mode:
            broker_qty = self._current_position_qty(symbol)
            if broker_qty is None:
                self.record("EXIT_BLOCKED_POSITION_UNKNOWN", direction=trade["direction"], option_symbol=symbol, qty=qty,
                            spot=spot, detail="Could not verify broker net quantity. No duplicate exit order sent.")
                self.halted = True
                self._save_state()
                return
            if broker_qty == 0:
                self.record("POSITION_ALREADY_FLAT", direction=trade["direction"], option_symbol=symbol, qty=qty,
                            spot=spot, detail="Broker reports no open net position; state reconciled.")
            else:
                side = -1 if broker_qty > 0 else 1
                tag = self._make_tag()
                try:
                    result = self._place_and_wait(symbol, abs(broker_qty), side, tag)
                    order_id = result["order_id"]
                except OrderUnresolved as exc:
                    self.pending_order = {"kind": "EXIT", "order_id": exc.order_id, "symbol": symbol,
                                          "qty": abs(broker_qty), "side": side, "tag": tag,
                                          "reason": reason, "spot": spot, "started_at": now_ist().isoformat()}
                    self.halted = True
                    self._save_state()
                    self.record("CRITICAL_EXIT_UNRESOLVED", direction=trade["direction"], option_symbol=symbol,
                                qty=abs(broker_qty), spot=spot, order_id=exc.order_id, detail=str(exc))
                    return
                except Exception as exc:
                    self.exit_attempts += 1
                    self.exit_retry_at = time.time() + 3.0
                    self.force_exit_reason = reason
                    self.record("EXIT_ORDER_FAILED", direction=trade["direction"], option_symbol=symbol, qty=abs(broker_qty),
                                spot=spot, detail=f"Exit reason={reason}; attempt={self.exit_attempts}; retry scheduled after order rejection/error: {exc}")
                    self.halted = True
                    self._save_state()
                    return
        self.record("EXIT_FILLED" if self.live_mode else "DRY_RUN_EXIT", direction=trade["direction"],
                    option_symbol=symbol, qty=qty, spot=spot, level_name=trade.get("level_name"), level=trade.get("level"),
                    confirmation=trade.get("confirmation"), order_id=order_id, detail=f"reason={reason}; entry_spot={trade['entry_spot']:.2f}; stop={trade['stop_level']:.2f}; target={trade['target_level']:.2f}")
        self.open_trade = None
        self.force_exit_reason = None
        self.exit_attempts = 0
        self.exit_retry_at = 0.0
        self.halted = False if not self.pending_order else self.halted
        self._save_state()

    def monitor_position(self) -> None:
        if not self.open_trade:
            return
        if self.pending_order and self.pending_order.get("kind") == "EXIT":
            return  # Don't submit a duplicate exit while the prior exit order is unresolved.
        spot, hi_seen, lo_seen, age = self.feed.risk_snapshot()
        now = now_ist()
        # Quote fallback only while a trade is open; avoid repeated REST calls when the socket is healthy.
        if (spot is None or age is None or age > 3.0) and time.time() - self.last_quote_fallback >= self.cfg.POSITION_QUOTE_FALLBACK_SECS:
            self.last_quote_fallback = time.time()
            fallback = self.api.quote_ltp(self.cfg.UNDERLYING)
            if fallback is not None:
                spot = fallback
                hi_seen = fallback if hi_seen is None else max(hi_seen, fallback)
                lo_seen = fallback if lo_seen is None else min(lo_seen, fallback)
                logging.warning("Using REST quote fallback for open-position risk monitor: NIFTY=%.2f", fallback)
            else:
                logging.error("Cannot obtain current NIFTY LTP while a position is open. Spot-based SL/TP monitoring is impaired.")
        if spot is None:
            return
        t = self.open_trade
        exit_reason = self.force_exit_reason
        # Conservative priority if the tick range crossed both SL and TP between checks.
        if t["direction"] == "BUY":
            hit_sl = lo_seen is not None and lo_seen <= float(t["stop_level"])
            hit_tp = hi_seen is not None and hi_seen >= float(t["target_level"])
        else:
            hit_sl = hi_seen is not None and hi_seen >= float(t["stop_level"])
            hit_tp = lo_seen is not None and lo_seen <= float(t["target_level"])
        if hit_sl:
            exit_reason = "spot_stop_loss"
        elif hit_tp:
            exit_reason = "spot_take_profit"
        elif now.time() >= self.cfg.EOD_EXIT_TIME:
            exit_reason = "eod_exit"
        if exit_reason:
            self._close_position(exit_reason, float(spot))

    def _resync_missing_bars(self, before: datetime) -> None:
        response = self.api.history_today(now_ist().date())
        if not isinstance(response, dict) or response.get("s") != "ok":
            logging.error("Socket gap detected but REST candle resync failed: %s. Strategy will wait for live bars.", response)
            return
        last_ts = self.engine.bars[-1]["timestamp"] if self.engine.bars else None
        recovered = [b for b in candles_to_bars(response) if b["timestamp"].date() == now_ist().date()
                     and b["timestamp"].time() >= self.cfg.SESSION_START
                     and b["timestamp"].time() <= self.cfg.SESSION_END
                     and b["timestamp"] < before
                     and (last_ts is None or b["timestamp"] > last_ts)]
        for bar in recovered:
            event = self.engine.add_closed_bar(bar, evaluate=True)
            if event:
                # No retroactive entries after a connection/data gap.
                event_dt = datetime.fromisoformat(event["signal_time"])
                age = (now_ist() - event_dt).total_seconds()
                if age <= self.cfg.STALE_SIGNAL_MAX_SECS:
                    self.handle_signal(event)
                else:
                    self.record("SKIP_SIGNAL_DURING_FEED_GAP", direction=event["direction"], level_name=event["level_name"],
                                level=event["level"], confirmation=event["confirmation"], detail=f"Recovered from feed gap; signal age={age:.1f}s.")

    def resolve_pending_order(self) -> None:
        if not self.pending_order:
            return
        pending = self.pending_order
        order_id = pending.get("order_id")
        if not order_id:
            recovered = self._find_order_by_tag(str(pending.get("tag", ""))) if pending.get("tag") else None
            if recovered and recovered.get("id"):
                order_id = str(recovered["id"])
                pending["order_id"] = order_id
                self._save_state()
                logging.warning("Recovered previously ambiguous order by tag: %s", order_id)
            else:
                logging.critical("Unresolved order has no FYERS order id yet. Trading remains halted; checking order tag/positions again.")
                symbol = pending.get("symbol")
                if symbol:
                    actual = self._current_position_qty(symbol)
                    if actual is not None and actual != 0 and pending.get("kind") == "ENTRY" and not self.open_trade:
                        sig = pending.get("signal", {})
                        self.open_trade = {"date": self.day.isoformat(), "direction": sig.get("direction", "BUY"),
                                           "option_symbol": symbol, "qty": abs(actual),
                                           "entry_spot": float(pending.get("spot_entry", 0.0)),
                                           "stop_level": float(pending.get("stop_level", 0.0)),
                                           "target_level": float(pending.get("target_level", 0.0)),
                                           "entry_time": now_ist().isoformat(), "entry_order_id": "UNRESOLVED",
                                           "level_name": sig.get("level_name", ""), "level": sig.get("level", 0.0),
                                           "confirmation": sig.get("confirmation", ""), "signal_time": sig.get("signal_time", "")}
                        self._save_state()
                        self.record("POSITION_FOUND_WITH_UNRESOLVED_ORDER", direction=self.open_trade["direction"],
                                    option_symbol=symbol, qty=abs(actual), spot=self.open_trade["entry_spot"],
                                    detail="Broker position found; automated entries remain halted pending order reconciliation.")
                return
        order = self._order_by_id(str(order_id))
        if not order:
            logging.critical("Order %s not present in order lookup. Trading remains halted; do not retry blindly.", order_id)
            return
        status = int(order.get("status", -1))
        filled = int(order.get("filledQty", 0) or 0)
        requested = int(order.get("qty", pending.get("qty", 0)) or pending.get("qty", 0))
        remaining = int(order.get("remainingQuantity", max(0, requested - filled)) or 0)
        if status == 2 and filled > 0 and remaining == 0:
            if pending["kind"] == "ENTRY":
                signal = pending["signal"]
                self.open_trade = {
                    "date": now_ist().date().isoformat(), "direction": signal["direction"],
                    "option_symbol": pending["symbol"], "qty": filled,
                    "entry_spot": float(pending["spot_entry"]), "stop_level": float(pending["stop_level"]),
                    "target_level": float(pending["target_level"]), "entry_time": now_ist().isoformat(),
                    "entry_order_id": str(order_id), "level_name": signal["level_name"], "level": signal["level"],
                    "confirmation": signal["confirmation"], "signal_time": signal["signal_time"],
                }
                self.trades_today += 1
                self.record("ENTRY_FILLED_AFTER_DELAY", direction=signal["direction"], option_symbol=pending["symbol"],
                            qty=filled, spot=pending["spot_entry"], level_name=signal["level_name"], level=signal["level"],
                            confirmation=signal["confirmation"], order_id=order_id, detail="Filled after initial order status timeout.")
            else:
                self.record("EXIT_FILLED_AFTER_DELAY", option_symbol=pending["symbol"], qty=filled, spot=pending.get("spot"),
                            order_id=order_id, detail=f"Exit reason={pending.get('reason', 'unknown')}")
                self.open_trade = None
            self.pending_order = None
            self.halted = False
            self._save_state()
            return
        if status in (1, 5, 7):
            # Reconcile any partial position before removing the pending state.
            symbol = pending["symbol"]
            actual = self._current_position_qty(symbol)
            if actual is None:
                logging.critical("Order %s ended but its position state cannot be reconciled. Trading remains halted.", order_id)
                return
            if pending["kind"] == "ENTRY" and actual != 0:
                sig = pending["signal"]
                spot_entry = float(pending["spot_entry"])
                self.open_trade = {"date": self.day.isoformat(), "direction": sig["direction"], "option_symbol": symbol,
                                   "qty": abs(actual), "entry_spot": spot_entry, "stop_level": float(pending["stop_level"]),
                                   "target_level": float(pending["target_level"]), "entry_time": now_ist().isoformat(),
                                   "entry_order_id": str(order_id), "level_name": sig["level_name"], "level": sig["level"],
                                   "confirmation": sig["confirmation"], "signal_time": sig["signal_time"]}
                self.trades_today += 1
                self.record("ENTRY_PARTIAL_RECONCILED", direction=sig["direction"], option_symbol=symbol, qty=abs(actual),
                            spot=spot_entry, order_id=order_id, detail="Broker position reconciled after terminal/partial order status.")
            elif pending["kind"] == "EXIT" and actual != 0:
                logging.critical("Exit order ended but %s still has netQty=%s. Trading remains halted; manual action may be needed.", symbol, actual)
                return
            else:
                self.record("ORDER_TERMINAL_NOT_FILLED", option_symbol=symbol, qty=requested, order_id=order_id,
                            detail=f"status={status}; filled={filled}; message={order.get('message', '')}")
            self.pending_order = None
            self.halted = False
            self._save_state()
            return
        # If an order sits pending/transit, try cancelling the remainder after several seconds,
        # but never submit a duplicate replacement order.
        started = pending.get("started_at")
        if started:
            try:
                started_dt = datetime.fromisoformat(started)
                if (now_ist() - started_dt.astimezone(IST)).total_seconds() > 8 and not pending.get("cancel_requested"):
                    cancel_resp = self.api.cancel_order(str(order_id))
                    pending["cancel_requested"] = True
                    pending["cancel_response"] = cancel_resp
                    self._save_state()
                    logging.critical("Requested cancellation of unresolved order %s; awaiting final order/position reconciliation: %s", order_id, cancel_resp)
            except Exception as exc:
                logging.critical("Cannot cancel/reconcile unresolved order %s: %s", order_id, exc)

    def run(self) -> None:
        self._startup_reconcile()
        self.load_symbol_master()
        self.warmup_from_history()
        self.feed.start(self.app_id, self.token)
        logging.warning("Bot started in %s mode. DRY_RUN=%s LIVE_TRADING_ENABLED=%s", "LIVE" if self.live_mode else "DRY-RUN",
                        self.cfg.DRY_RUN, self.cfg.LIVE_TRADING_ENABLED)
        logging.info("For best opening-range capture, start before 09:15 IST. OR levels are fixed from the 09:15 one-minute candle.")
        logging.info("Strategy: move-away/retest -> C1/C2 strong-body confirm; otherwise strict C1/C2/C3 fallback. SL/TP use NIFTY spot.")

        try:
            while True:
                now = now_ist()
                if now.date() != self.day:
                    logging.critical("Date changed while engine was active. Do not assume any open position was closed; broker reconciliation is required.")
                    if self.open_trade or self.pending_order:
                        self.force_exit_reason = "date_change_shutdown"
                    else:
                        break

                if self.pending_order and time.time() - self.last_pending_check >= 1.5:
                    self.last_pending_check = time.time()
                    self.resolve_pending_order()

                # At launcher cutoff, prevent further signal processing. Continue
                # order reconciliation and position monitoring until flat.
                if self.stop_requested.is_set():
                    if self.open_trade and not self.pending_order and not self.force_exit_reason:
                        self.force_exit_reason = "launcher_shutdown"
                    self.monitor_position()
                    if not self.open_trade and not self.pending_order:
                        logging.info("Graceful shutdown complete: no open trade or unresolved order remains.")
                        self._save_state()
                        self._close_market_data_socket()
                        return
                    # Keep the process alive to reconcile/monitor an unresolved position.
                    time.sleep(0.5)
                    continue

                # Finalize candles queued by the websocket thread, or repair missed minutes from REST history.
                try:
                    kind, payload = self.feed.events.get(timeout=0.25)
                except queue.Empty:
                    kind, payload = None, None
                if kind == "bar":
                    event = self.engine.add_closed_bar(payload, evaluate=True)
                    if event:
                        self.handle_signal(event)
                elif kind == "resync":
                    self.record("MARKET_DATA_GAP", detail=f"WebSocket minute gap; REST resync before {payload.isoformat()}.")
                    self._resync_missing_bars(payload)

                self.monitor_position()
                if now.time() >= self.cfg.SESSION_END and not self.open_trade and not self.pending_order:
                    logging.info("Session finished and no position is open. Bot stopping.")
                    self._save_state()
                    self._close_market_data_socket()
                    break
                time.sleep(0.1)
        except KeyboardInterrupt:
            logging.warning("Keyboard interrupt received by engine. Requesting graceful stop and position reconciliation.")
            self.request_stop()
            # Keep reconciling if an open trade/order remains; don't silently abandon it.
            while self.open_trade or self.pending_order:
                if self.pending_order and time.time() - self.last_pending_check >= 1.5:
                    self.last_pending_check = time.time()
                    self.resolve_pending_order()
                if self.open_trade and not self.pending_order and not self.force_exit_reason:
                    self.force_exit_reason = "keyboard_interrupt"
                self.monitor_position()
                time.sleep(0.5)
            self._save_state()
            self._close_market_data_socket()


def setup_logging() -> None:
    cfg = Config()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.FileHandler(cfg.LOG_FILE, encoding="utf-8"), logging.StreamHandler()],
    )


class OrderUnresolved(RuntimeError):
    def __init__(self, message: str, order_id: Optional[str], symbol: str, qty: int, side: int, tag: str, details: Any = None):
        super().__init__(message)
        self.order_id = order_id
        self.symbol = symbol
        self.qty = qty
        self.side = side
        self.tag = tag
        self.details = details


# Import-compatible name used by runBot_first_candle_sr_backtest_live.py.
Engine = LiveBot


def main() -> None:
    setup_logging()
    cfg = Config()
    live_flag_ok = (not cfg.DRY_RUN and cfg.LIVE_TRADING_ENABLED and
                    os.getenv(cfg.LIVE_CONFIRM_ENV, "").strip().upper() == "YES")
    logging.info("Uploaded Backtest First Candle SR LIVE engine starting. Mode=%s | quantity=%s | daily trade limit=%s | spot SL/TP=%s/%s",
                 "LIVE" if live_flag_ok else "DRY_RUN", cfg.QUANTITY, cfg.MAX_TRADES_PER_DAY,
                 cfg.STOP_LOSS_SPOT_POINTS, cfg.TAKE_PROFIT_SPOT_POINTS)
    if now_ist().weekday() >= 5:
        logging.warning("Today is a weekend in IST. The bot will exit; run it on a trading day.")
        return
    try:
        Engine().run()
    except KeyboardInterrupt:
        logging.warning("Stopped by user. Verify FYERS positions before assuming any open position is closed.")
    except Exception as exc:
        logging.exception("Bot stopped due to error: %s", exc)
        raise


if __name__ == "__main__":
    main()
