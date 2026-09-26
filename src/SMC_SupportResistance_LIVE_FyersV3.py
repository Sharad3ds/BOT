"""
Live Support/Resistance Reversal bot based directly on the accompanying
SMC_SupportResistance backtest.

Strategy:
  1. Detect LuxAlgo-style swing pivots using detect_pivots().
  2. Turn confirmed pivots into ATR-sized support/resistance zones.
  3. Keep zones alive until invalidated, fired once, or aged out.
  4. When price re-enters a zone, wait for a healthy reversal candle:
       support + green close outside zone -> BUY ITM CE
       resistance + red close outside zone -> BUY ITM PE
  5. One position at a time.

IMPORTANT:
  - Signals use CLOSED 1-minute bars only.
  - The live engine rebuilds zones from recent history each poll so the
    2-day zone-age rule can carry levels across day boundaries.
  - Entry is a normal Fyers V3 INTRADAY MARKET order.
  - SL/TP are based on NIFTY SPOT POINTS, matching the backtest. The live
    engine therefore monitors NIFTY spot and closes the option position with
    DELETE /api/v3/positions when the spot SL/TP level is reached.
  - This is intentionally NOT BO/CO and does not attach premium-based broker
    SL/TP because that would change the backtest's risk model.

DRY_RUN=False by default for the live version.
"""

import os
import json
import time
import calendar
import logging
import warnings
from datetime import datetime, timedelta, time as dtime
from typing import Optional, Dict, List

import numpy as np
import pandas as pd
import pytz
import requests
import urllib3

from expiry_resolver import ExpiryResolver

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

IST = pytz.timezone("Asia/Kolkata")
STATE_FILE = "smc_sr_live_position_state.json"


def ist_time(*args):
    return datetime.now(IST).timetuple()


logging.Formatter.converter = ist_time

try:
    from logger_config import get_logger
    logger = get_logger()
except ImportError:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - SMCSRLive - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logger = logging.getLogger("SMCSRLive")


# ---------------------------------------------------------------------------
# Expiry helpers
# ---------------------------------------------------------------------------

def get_next_tuesday(now=None):
    now = now or datetime.now(IST)
    x = datetime(now.year, now.month, now.day)
    while x.weekday() != 1:
        x += timedelta(days=1)
    return x


def monthly_expiry(y, m):
    last = calendar.monthrange(y, m)[1]
    x = datetime(y, m, last)
    while x.weekday() != 1:
        x -= timedelta(days=1)
    return x


def expiry_code(dt):
    yy = dt.year % 100
    me = monthly_expiry(dt.year, dt.month)
    if dt.date() == me.date():
        return f"{yy:02d}{dt.strftime('%b').upper()}"
    mc = str(dt.month) if dt.month <= 9 else {10: "O", 11: "N", 12: "D"}[dt.month]
    return f"{yy:02d}{mc}{dt.day:02d}"


# ---------------------------------------------------------------------------
# Config - kept aligned with the supplied backtest
# ---------------------------------------------------------------------------

class Config:
    UNDERLYING = "NSE:NIFTY50-INDEX"
    BASE_RES = "1"
    TF_RULE = "1min"

    # Need enough history to retain zones up to MAX_ZONE_AGE_DAYS across days.
    LIVE_HISTORY_DAYS = 7

    SWING_LEN = 15
    ATR_PERIOD = 15
    ZONE_TOL_MULT = 0.1
    ZONE_INVALIDATE_MULT = 0.1
    HEALTHY_BODY_FRAC = 0.001
    MAX_ACTIVE_ZONES_PER_SIDE = 5
    MAX_ZONE_AGE_DAYS = 2

    STRIKE_STEP = 100
    LOT = 650
    MONEYNESS = "ITM"

    SESSION_START = dtime(9, 20)
    SESSION_END = dtime(15, 30)
    SIGNAL_START_TIME = dtime(9, 35)
    TRADE_START = dtime(9, 30)
    TRADE_END = dtime(15, 0)
    EOD_EXIT_TIME = dtime(15, 20)

    # NIFTY spot-point SL/TP. Matches supplied backtest.
    STOP_LOSS_PTS = 40
    TAKE_PROFIT_PTS = 130

    POLL_INTERVAL_SECS = 5
    DRY_RUN = False

    # After order placement, wait for the position to become visible.
    FILL_RETRIES = 20
    FILL_RETRY_DELAY_SECS = 0.5

    def __init__(self):
        self.EXPIRY_CODE = expiry_code(get_next_tuesday())
        logger.info(
            f"[CFG] S/R live | TF={self.TF_RULE} swing={self.SWING_LEN} "
            f"tol={self.ZONE_TOL_MULT}xATR invalid={self.ZONE_INVALIDATE_MULT}xATR "
            f"age={self.MAX_ZONE_AGE_DAYS}d active/side={self.MAX_ACTIVE_ZONES_PER_SIDE} "
            f"SL={self.STOP_LOSS_PTS} TP={self.TAKE_PROFIT_PTS} spot points "
            f"| ITM lot={self.LOT} | DRY_RUN={self.DRY_RUN}"
        )


# ---------------------------------------------------------------------------
# Fyers V3 HTTP client
# ---------------------------------------------------------------------------

class FyersHTTP:
    T1 = "https://api-t1.fyers.in/api/v3"
    T2 = "https://api-t1.fyers.in/data"

    def __init__(self, app, tok):
        auth = tok if ":" in tok else f"{app}:{tok}"
        self.s = requests.Session()
        self.s.verify = False
        self.s.headers.update({"Authorization": auth})

    def profile(self):
        return self.s.get(f"{self.T1}/profile", timeout=10).json()

    def history(self, symbol, res, d_from, d_to):
        try:
            r = self.s.get(
                f"{self.T2}/history",
                params={
                    "symbol": symbol,
                    "resolution": res,
                    "date_format": "1",
                    "range_from": d_from,
                    "range_to": d_to,
                    "cont_flag": "0",
                },
                timeout=15,
            )
            return r.json()
        except Exception as e:
            return {"s": "error", "message": str(e)}

    def quotes(self, symbol: str) -> dict:
        try:
            return self.s.get(
                f"{self.T2}/quotes",
                params={"symbols": symbol},
                timeout=5,
            ).json()
        except Exception:
            return {}

    def place_order(self, data: dict) -> dict:
        try:
            return self.s.post(
                f"{self.T1}/orders/sync", json=data, timeout=10
            ).json()
        except Exception as e:
            return {"s": "error", "message": str(e)}

    def exit_position(self, position_id: str) -> dict:
        try:
            r = self.s.delete(
                f"{self.T1}/positions", json={"id": position_id}, timeout=10
            )
            return r.json()
        except Exception as e:
            return {"s": "error", "message": str(e)}

    def positions(self) -> dict:
        try:
            return self.s.get(f"{self.T1}/positions", timeout=10).json()
        except Exception as e:
            return {"s": "error", "message": str(e)}

    def orderbook(self) -> dict:
        try:
            return self.s.get(f"{self.T1}/orders", timeout=10).json()
        except Exception as e:
            return {"s": "error", "message": str(e)}

    def get_position(self, symbol: str) -> Optional[dict]:
        resp = self.positions()
        if not isinstance(resp, dict) or resp.get("s") != "ok":
            return None
        for pos in resp.get("netPositions", []):
            if pos.get("symbol") == symbol and pos.get("productType") == "INTRADAY":
                return pos
        return None

    def get_order(self, order_id: str) -> Optional[dict]:
        if not order_id:
            return None
        resp = self.orderbook()
        if not isinstance(resp, dict) or resp.get("s") != "ok":
            return None
        for order in resp.get("orderBook", []):
            if order.get("id") == order_id:
                return order
        return None


# ---------------------------------------------------------------------------
# Exact S/R signal logic from the supplied backtest
# ---------------------------------------------------------------------------

def detect_pivots(o, h, l, c, swing_len: int):
    """Causal port of the supplied backtest's LuxAlgo-style leg logic."""
    pivot_highs, pivot_lows = [], []
    leg = 0

    for i in range(swing_len, len(c)):
        window_high = np.max(h[i - swing_len + 1: i + 1])
        window_low = np.min(l[i - swing_len + 1: i + 1])
        piv_idx = i - swing_len

        new_leg_high = h[piv_idx] > window_high
        new_leg_low = l[piv_idx] < window_low

        new_leg = leg
        if new_leg_high:
            new_leg = 0
        elif new_leg_low:
            new_leg = 1

        if new_leg != leg:
            if new_leg == 1:
                pivot_lows.append((i, piv_idx, float(l[piv_idx])))
            else:
                pivot_highs.append((i, piv_idx, float(h[piv_idx])))
            leg = new_leg

    return pivot_highs, pivot_lows


def generate_recent_signals(df_1min: pd.DataFrame, cfg: Config, target_date) -> List[dict]:
    """Rebuild the S/R state over recent history and return only today's events."""
    if df_1min.empty:
        return []

    r = df_1min.resample(cfg.TF_RULE).agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    ).dropna()

    if len(r) < cfg.SWING_LEN + cfg.ATR_PERIOD + 2:
        return []

    o = r["open"].to_numpy()
    h = r["high"].to_numpy()
    l = r["low"].to_numpy()
    cl = r["close"].to_numpy()

    prev_close = np.empty(len(r))
    prev_close[0] = cl[0]
    prev_close[1:] = cl[:-1]
    tr = np.maximum(
        h - l,
        np.maximum(np.abs(h - prev_close), np.abs(l - prev_close)),
    )
    atr = pd.Series(tr).ewm(
        alpha=1 / cfg.ATR_PERIOD, adjust=False
    ).mean().to_numpy()

    pivot_highs, pivot_lows = detect_pivots(o, h, l, cl, cfg.SWING_LEN)

    zones: List[dict] = []
    highs_by_confirm: Dict[int, list] = {}
    lows_by_confirm: Dict[int, list] = {}

    for confirm_i, piv_i, level in pivot_highs:
        highs_by_confirm.setdefault(confirm_i, []).append((piv_i, level))
    for confirm_i, piv_i, level in pivot_lows:
        lows_by_confirm.setdefault(confirm_i, []).append((piv_i, level))

    def healthy(k):
        rng = h[k] - l[k]
        return rng > 0 and abs(cl[k] - o[k]) >= cfg.HEALTHY_BODY_FRAC * rng

    green = lambda k: cl[k] > o[k]
    red = lambda k: cl[k] < o[k]

    events: List[dict] = []

    for i in range(cfg.SWING_LEN, len(r)):
        bar_close_time = r.index[i] + pd.Timedelta(cfg.TF_RULE)
        ts_time = bar_close_time.time()
        if not (cfg.SESSION_START <= ts_time <= cfg.SESSION_END):
            continue

        current_atr = atr[i]
        if np.isnan(current_atr):
            continue

        # Expire old zones using the same calendar-elapsed-time rule as backtest.
        for z in zones:
            if not z["active"]:
                continue
            age_days = (bar_close_time - z["created_at"]).total_seconds() / 86400.0
            if age_days > cfg.MAX_ZONE_AGE_DAYS:
                z["active"] = False
                z["inactive_reason"] = "age_expired"

        # Add newly confirmed pivots, capped per side.
        tol = cfg.ZONE_TOL_MULT * current_atr

        for piv_i, level in highs_by_confirm.get(i, []):
            n_active = sum(
                1 for z in zones if z["kind"] == "resistance" and z["active"]
            )
            if n_active < cfg.MAX_ACTIVE_ZONES_PER_SIDE:
                zones.append({
                    "level": level,
                    "tol": tol,
                    "kind": "resistance",
                    "state": "watching",
                    "active": True,
                    "created_at": bar_close_time,
                    "pivot_idx": piv_i,
                })

        for piv_i, level in lows_by_confirm.get(i, []):
            n_active = sum(
                1 for z in zones if z["kind"] == "support" and z["active"]
            )
            if n_active < cfg.MAX_ACTIVE_ZONES_PER_SIDE:
                zones.append({
                    "level": level,
                    "tol": tol,
                    "kind": "support",
                    "state": "watching",
                    "active": True,
                    "created_at": bar_close_time,
                    "pivot_idx": piv_i,
                })

        if ts_time < cfg.SIGNAL_START_TIME:
            continue

        for z in zones:
            if not z["active"]:
                continue

            lo = z["level"] - z["tol"]
            hi = z["level"] + z["tol"]
            in_zone = lo <= cl[i] <= hi

            if z["kind"] == "support":
                invalidated = cl[i] < z["level"] - cfg.ZONE_INVALIDATE_MULT * current_atr
            else:
                invalidated = cl[i] > z["level"] + cfg.ZONE_INVALIDATE_MULT * current_atr

            if z["state"] == "watching":
                if invalidated:
                    z["active"] = False
                    z["inactive_reason"] = "invalidated"
                    continue
                if in_zone:
                    z["state"] = "in_zone"
                continue

            if z["state"] == "in_zone":
                if invalidated:
                    z["active"] = False
                    z["inactive_reason"] = "invalidated"
                    continue

                if z["kind"] == "support" and green(i) and healthy(i) and cl[i] > hi:
                    if bar_close_time.date() == target_date:
                        events.append({
                            "timestamp": bar_close_time,
                            "dir": "BUY",
                            "zone_level": z["level"],
                            "zone_kind": "support",
                            "index_price": float(cl[i]),
                        })
                    z["active"] = False
                    z["inactive_reason"] = "fired"

                elif z["kind"] == "resistance" and red(i) and healthy(i) and cl[i] < lo:
                    if bar_close_time.date() == target_date:
                        events.append({
                            "timestamp": bar_close_time,
                            "dir": "SELL",
                            "zone_level": z["level"],
                            "zone_kind": "resistance",
                            "index_price": float(cl[i]),
                        })
                    z["active"] = False
                    z["inactive_reason"] = "fired"

    return events


# ---------------------------------------------------------------------------
# Order / position manager
# ---------------------------------------------------------------------------

class OrderManager:
    def __init__(self, fyers: FyersHTTP, cfg: Config):
        self.fyers = fyers
        self.cfg = cfg
        self.has = False
        self.symbol = None
        self.direction = None
        self.entry_price = None
        self.entry_time = None
        self.entry_index_price = None
        self.zone_kind = None
        self.zone_level = None
        self.order_id = None
        self.position_id = None
        self.entry_confirmed = False
        self.exit_pending_reason = None

    def _save_state(self):
        try:
            state = {
                "has": self.has,
                "symbol": self.symbol,
                "direction": self.direction,
                "entry_price": self.entry_price,
                "entry_time": self.entry_time.isoformat() if self.entry_time else None,
                "entry_index_price": self.entry_index_price,
                "zone_kind": self.zone_kind,
                "zone_level": self.zone_level,
                "order_id": self.order_id,
                "position_id": self.position_id,
                "entry_confirmed": self.entry_confirmed,
                "exit_pending_reason": self.exit_pending_reason,
                "saved": datetime.now(IST).isoformat(),
            }
            tmp = STATE_FILE + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(state, f)
            os.replace(tmp, STATE_FILE)
        except Exception as e:
            logger.error(f"[OM] save state: {e}")

    def _clear_state(self):
        try:
            if os.path.exists(STATE_FILE):
                os.remove(STATE_FILE)
        except Exception:
            pass

    def _reset_state(self):
        self.has = False
        self.symbol = None
        self.direction = None
        self.entry_price = None
        self.entry_time = None
        self.entry_index_price = None
        self.zone_kind = None
        self.zone_level = None
        self.order_id = None
        self.position_id = None
        self.entry_confirmed = False
        self.exit_pending_reason = None

    def restore(self):
        if not os.path.exists(STATE_FILE):
            return
        try:
            with open(STATE_FILE, encoding="utf-8") as f:
                st = json.load(f)
            saved = datetime.fromisoformat(st["saved"])
            if saved.date() != datetime.now(IST).date():
                self._clear_state()
                return
            if st.get("has"):
                self.has = True
                self.symbol = st.get("symbol")
                self.direction = st.get("direction")
                self.entry_price = st.get("entry_price")
                self.entry_time = datetime.fromisoformat(st["entry_time"]) if st.get("entry_time") else None
                self.entry_index_price = st.get("entry_index_price")
                self.zone_kind = st.get("zone_kind")
                self.zone_level = st.get("zone_level")
                self.order_id = st.get("order_id")
                self.position_id = st.get("position_id") or (
                    f"{self.symbol}-INTRADAY" if self.symbol else None
                )
                self.entry_confirmed = bool(st.get("entry_confirmed"))
                self.exit_pending_reason = st.get("exit_pending_reason")
                logger.info(
                    f"[OM] Restored {self.symbol} {self.direction} "
                    f"entry_option={self.entry_price} entry_spot={self.entry_index_price} "
                    f"position={self.position_id}"
                )
        except Exception as e:
            logger.error(f"[OM] restore: {e}")

    def _find_position(self):
        resp = self.fyers.positions()
        if not isinstance(resp, dict) or resp.get("s") != "ok":
            return None, resp
        for pos in resp.get("netPositions", []):
            if self.position_id and pos.get("id") == self.position_id:
                return pos, resp
            if self.symbol and pos.get("symbol") == self.symbol and pos.get("productType") == "INTRADAY":
                return pos, resp
        return None, resp

    def _current_ltp(self):
        if not self.symbol:
            return 0.0
        try:
            resp = self.fyers.quotes(self.symbol)
            return float(resp["d"][0]["v"]["lp"])
        except Exception:
            return 0.0

    def _trade_rec(self, exit_ltp, reason):
        pnl_pts = float(exit_ltp) - float(self.entry_price)
        pnl_rs = pnl_pts * self.cfg.LOT
        return {
            "direction": self.direction,
            "option_symbol": self.symbol,
            "zone_kind": self.zone_kind,
            "zone_level": round(float(self.zone_level), 2) if self.zone_level is not None else None,
            "entry_time": self.entry_time.strftime("%Y-%m-%d %H:%M:%S") if self.entry_time else "",
            "entry_index_price": round(float(self.entry_index_price), 2) if self.entry_index_price is not None else None,
            "entry_price": round(float(self.entry_price), 2),
            "exit_time": datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S"),
            "exit_price": round(float(exit_ltp), 2),
            "exit_reason": reason,
            "pnl_pts": round(pnl_pts, 2),
            "pnl_rs": round(pnl_rs, 2),
            "result": "profit" if pnl_pts > 0 else "loss",
        }

    def open(self, symbol, direction, option_ltp, entry_index_price, zone_kind, zone_level):
        if self.has:
            return False

        if self.cfg.DRY_RUN:
            self.has = True
            self.symbol = symbol
            self.direction = direction
            self.entry_price = float(option_ltp)
            self.entry_time = datetime.now(IST)
            self.entry_index_price = float(entry_index_price)
            self.zone_kind = zone_kind
            self.zone_level = float(zone_level)
            self.order_id = f"DRY-{datetime.now(IST).strftime('%H%M%S')}"
            self.entry_confirmed = True
            logger.info(
                f"[OM][TRADE] ENTRY {direction} {symbol} @ option {option_ltp:.2f} "
                f"| spot={entry_index_price:.2f} | zone={zone_kind}@{zone_level:.2f}"
            )
            logger.info(
                f"[OM][TRADE] Entry details: qty={self.cfg.LOT} MARKET INTRADAY "
                f"| spot SL={entry_index_price - self.cfg.STOP_LOSS_PTS:.2f} / "
                f"{entry_index_price + self.cfg.STOP_LOSS_PTS:.2f} "
                f"TP={entry_index_price - self.cfg.TAKE_PROFIT_PTS:.2f} / "
                f"{entry_index_price + self.cfg.TAKE_PROFIT_PTS:.2f}"
            )
            self._save_state()
            return True

        data = {
            "symbol": symbol,
            "qty": int(self.cfg.LOT),
            "type": 2,
            "side": 1,
            "productType": "INTRADAY",
            "limitPrice": 0,
            "stopPrice": 0,
            "validity": "DAY",
            "disclosedQty": 0,
            "offlineOrder": False,
            "stopLoss": 0,
            "takeProfit": 0,
            "orderTag": "SMCSR",
            "isSliceOrder": False,
        }

        logger.info(
            f"[OM][LIVE] ENTRY BUY {symbol} | dir={direction} qty={self.cfg.LOT} "
            f"option_ltp={option_ltp:.2f} spot={entry_index_price:.2f}"
        )
        resp = self.fyers.place_order(data)
        logger.info(f"[OM][LIVE] place_order response: {resp}")

        if not isinstance(resp, dict) or resp.get("s") != "ok":
            logger.error(f"[OM][LIVE] order rejected: {resp}")
            return False

        self.has = True
        self.symbol = symbol
        self.direction = direction
        self.entry_price = float(option_ltp)
        self.entry_time = datetime.now(IST)
        self.entry_index_price = float(entry_index_price)
        self.zone_kind = zone_kind
        self.zone_level = float(zone_level)
        self.order_id = resp.get("id")
        self.position_id = f"{symbol}-INTRADAY"
        self.entry_confirmed = False
        self.exit_pending_reason = None
        self._save_state()

        # Wait for /positions to expose the fill and replace quote entry with
        # broker-reported average price when available.
        pos = None
        for attempt in range(self.cfg.FILL_RETRIES):
            pos = self.fyers.get_position(symbol)
            if pos and int(pos.get("netQty", 0)) != 0:
                break
            time.sleep(self.cfg.FILL_RETRY_DELAY_SECS)

        if pos and int(pos.get("netQty", 0)) != 0:
            self.position_id = pos.get("id") or self.position_id
            actual_entry = pos.get("buyAvg") or pos.get("netAvg")
            if actual_entry is not None:
                try:
                    self.entry_price = float(actual_entry)
                except (TypeError, ValueError):
                    pass
            self.entry_confirmed = True
            self._save_state()
            logger.info(
                f"[OM][LIVE] Fill confirmed: order={self.order_id} "
                f"position={self.position_id} option_entry~{self.entry_price:.2f}"
            )
        else:
            logger.warning(
                f"[OM][LIVE] Position not visible yet for order={self.order_id}; "
                f"reconciliation will continue."
            )

        logger.info(
            f"[OM][TRADE] ENTRY {direction} {symbol} @ option ~{self.entry_price:.2f} "
            f"| spot={self.entry_index_price:.2f} | zone={zone_kind}@{zone_level:.2f}"
        )
        logger.info(
            f"[OM][TRADE] Spot risk: SL={self.cfg.STOP_LOSS_PTS} pts "
            f"TP={self.cfg.TAKE_PROFIT_PTS} pts | entry spot={self.entry_index_price:.2f}"
        )
        return True

    def close(self, ltp, reason):
        if not self.has or not self.symbol:
            return None

        if self.cfg.DRY_RUN:
            rec = self._trade_rec(float(ltp), reason)
            logger.info(
                f"[OM][TRADE] EXIT {self.symbol} option={ltp:.2f} reason={reason} "
                f"entry={self.entry_price:.2f} spot_entry={self.entry_index_price:.2f}"
            )
            logger.info(
                f"[OM][TRADE] PnL Rs {rec['pnl_rs']:,.0f} | points={rec['pnl_pts']:.2f} | result={rec['result']}"
            )
            self._reset_state()
            self._clear_state()
            return rec

        # Do not send a second exit while an earlier exit is pending.
        if self.exit_pending_reason:
            logger.info(
                f"[OM][LIVE] exit already pending for {self.symbol} "
                f"reason={self.exit_pending_reason}; waiting for netQty=0"
            )
            return None

        pos, resp = self._find_position()
        if not isinstance(resp, dict) or resp.get("s") != "ok":
            logger.warning(f"[OM][LIVE] /positions failed before exit: {resp}")
            return None

        if pos is None:
            logger.info(
                f"[OM][LIVE] no matching INTRADAY position for {self.symbol}; "
                f"no exit sent"
            )
            return None

        net_qty = int(pos.get("netQty", 0))
        if net_qty == 0:
            return self._finalize_broker_close(float(ltp), reason)

        self.position_id = pos.get("id") or self.position_id or f"{self.symbol}-INTRADAY"
        actual_entry = pos.get("buyAvg") or pos.get("netAvg")
        if actual_entry is not None:
            try:
                self.entry_price = float(actual_entry)
            except (TypeError, ValueError):
                pass

        self.exit_pending_reason = reason
        self._save_state()

        logger.info(
            f"[OM][LIVE] EXIT {self.symbol} position={self.position_id} "
            f"netQty={net_qty} option_ltp~{ltp:.2f} reason={reason}"
        )
        resp = self.fyers.exit_position(self.position_id)
        logger.info(f"[OM][LIVE] exit_position response: {resp}")

        if not isinstance(resp, dict) or resp.get("s") != "ok":
            logger.error(f"[OM][LIVE] exit rejected: {resp}")
            self.exit_pending_reason = None
            self._save_state()
            return None

        return None

    def _finalize_broker_close(self, exit_ltp, reason):
        rec = self._trade_rec(exit_ltp, reason)
        logger.info(
            f"[OM][TRADE] EXIT {self.symbol} option~{exit_ltp:.2f} reason={reason} "
            f"entry={self.entry_price:.2f}"
        )
        logger.info(
            f"[OM][TRADE] PnL Rs {rec['pnl_rs']:,.0f} | points={rec['pnl_pts']:.2f} | result={rec['result']}"
        )
        self._reset_state()
        self._clear_state()
        return rec

    def sync_with_broker(self):
        """Reconcile local state with current INTRADAY /positions."""
        if not self.has or self.cfg.DRY_RUN:
            return None

        pos, resp = self._find_position()
        if not isinstance(resp, dict) or resp.get("s") != "ok":
            logger.warning(f"[OM] positions sync failed: {resp}")
            return None

        if not self.entry_confirmed:
            if pos is not None and int(pos.get("netQty", 0)) != 0:
                self.position_id = pos.get("id") or self.position_id
                actual_entry = pos.get("buyAvg") or pos.get("netAvg")
                if actual_entry is not None:
                    try:
                        self.entry_price = float(actual_entry)
                    except (TypeError, ValueError):
                        pass
                self.entry_confirmed = True
                self._save_state()
                logger.info(
                    f"[OM][LIVE] Entry fill confirmed during reconciliation: "
                    f"position={self.position_id} option_entry~{self.entry_price:.2f}"
                )
            else:
                order = self.fyers.get_order(self.order_id)
                if order:
                    try:
                        status = int(order.get("status", 0))
                    except Exception:
                        status = 0
                    if status in (1, 5, 7):
                        logger.error(
                            f"[OM][LIVE] Entry order {self.order_id} failed/cancelled: {order}"
                        )
                        self._reset_state()
                        self._clear_state()
                        return {"entry_failed": True}
            return None

        if pos is not None and int(pos.get("netQty", 0)) != 0:
            return None

        if pos is None:
            logger.info(
                f"[OM][LIVE] No matching position record for {self.symbol}; "
                f"waiting for /positions to confirm closure."
            )
            return None

        # Matching position with netQty=0: broker considers it closed.
        exit_ltp = self._current_ltp()
        reason = self.exit_pending_reason or "broker_exit"
        return self._finalize_broker_close(exit_ltp, reason)


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class Engine:
    def __init__(self):
        self.cfg = Config()
        self.fy = self._init_fyers()
        self.expiry_resolver = ExpiryResolver()
        self.om = OrderManager(self.fy, self.cfg)
        self.om.restore()
        self.trades: List[dict] = []
        self._fired_events = set()
        self._today = None
        self._last_logged_event_signature = None

        logger.info(
            f"[ENG] ready | poll={self.cfg.POLL_INTERVAL_SECS}s "
            f"history={self.cfg.LIVE_HISTORY_DAYS}d"
        )
        if self.cfg.DRY_RUN:
            logger.warning("*** DRY-RUN: no real orders ***")
        else:
            logger.warning(
                "*** LIVE MODE: INTRADAY MARKET entry + spot-point SL/TP + DELETE /positions ***"
            )

    def _init_fyers(self):
        for f in ("fyers_appid.txt", "fyers_token.txt"):
            if not os.path.exists(f):
                raise FileNotFoundError(f)
        with open("fyers_appid.txt", encoding="utf-8") as f:
            app = f.read().strip()
        with open("fyers_token.txt", encoding="utf-8") as f:
            tok = f.read().strip()
        fy = FyersHTTP(app, tok)
        p = fy.profile()
        if p.get("s") != "ok":
            raise ConnectionError(f"Fyers auth failed: {p}")
        logger.info(f"[ENG] Fyers connected: {p['data']['name']}")
        return fy

    def _fetch_recent_bars(self):
        now = datetime.now(IST)
        start = now.date() - timedelta(days=self.cfg.LIVE_HISTORY_DAYS)
        resp = self.fy.history(
            self.cfg.UNDERLYING,
            self.cfg.BASE_RES,
            start.strftime("%Y-%m-%d"),
            now.strftime("%Y-%m-%d"),
        )
        if resp.get("s") != "ok":
            logger.warning(f"[ENG] history error: {resp}")
            return pd.DataFrame()
        candles = resp.get("candles", [])
        if not candles:
            return pd.DataFrame()
        df = pd.DataFrame(
            candles,
            columns=["timestamp", "open", "high", "low", "close", "volume"],
        )
        df["timestamp"] = pd.to_datetime(
            df["timestamp"], unit="s", utc=True
        ).dt.tz_convert(IST)
        df = (
            df.drop_duplicates("timestamp")
            .sort_values("timestamp")
            .set_index("timestamp")
        )
        df = df.iloc[df.index.indexer_between_time(self.cfg.SESSION_START, self.cfg.SESSION_END)]
        return df

    def _drop_forming_bar(self, df):
        if df.empty:
            return df
        try:
            minutes = int(self.cfg.BASE_RES)
        except ValueError:
            minutes = 1
        now = datetime.now(IST)
        last_start = df.index[-1]
        if last_start + timedelta(minutes=minutes) > now:
            return df.iloc[:-1]
        return df

    def _ltp(self, symbol):
        r = self.fy.quotes(symbol)
        try:
            v = r["d"][0]["v"].get("lp")
            return float(v) if v is not None else None
        except Exception:
            return None

    def _opt_symbol(self, spot, direction):
        exp_code, _ = self.expiry_resolver.resolve_expiry_code(
            datetime.now(IST).date(), expiry_code, get_next_tuesday
        )
        atm = int(round(float(spot) / self.cfg.STRIKE_STEP) * self.cfg.STRIKE_STEP)
        itm = self.cfg.MONEYNESS == "ITM"
        if direction == "BUY":
            strike = atm - self.cfg.STRIKE_STEP if itm else atm
            typ = "CE"
        else:
            strike = atm + self.cfg.STRIKE_STEP if itm else atm
            typ = "PE"
        return f"NSE:NIFTY{exp_code}{strike:05d}{typ}"

    def _spot_ltp(self):
        return self._ltp(self.cfg.UNDERLYING)

    def _spot_exit_reason(self, spot_ltp):
        if not self.om.has or self.om.entry_index_price is None:
            return None
        entry = float(self.om.entry_index_price)
        sl = self.cfg.STOP_LOSS_PTS
        tp = self.cfg.TAKE_PROFIT_PTS
        if self.om.direction == "BUY":
            if spot_ltp <= entry - sl:
                return "stop_loss"
            if spot_ltp >= entry + tp:
                return "target"
        else:
            if spot_ltp >= entry + sl:
                return "stop_loss"
            if spot_ltp <= entry - tp:
                return "target"
        return None

    def _check_risk(self):
        if not self.om.has:
            return False
        spot = self._spot_ltp()
        if spot is None:
            return False
        reason = self._spot_exit_reason(spot)
        if reason is None:
            return False

        option_ltp = self._ltp(self.om.symbol) or self.om.entry_price
        logger.info(
            f"[ENG][RISK] {reason.upper()} | spot={spot:.2f} "
            f"entry_spot={self.om.entry_index_price:.2f} "
            f"option={self.om.symbol} ltp~{option_ltp:.2f}"
        )
        rec = self.om.close(option_ltp, reason)
        if rec:
            self.trades.append(rec)
        return True

    def _eod_exit(self):
        if not self.om.has:
            return False
        now = datetime.now(IST)
        if now.time() < self.cfg.EOD_EXIT_TIME:
            return False
        option_ltp = self._ltp(self.om.symbol) or self.om.entry_price
        rec = self.om.close(option_ltp, "eod_exit")
        if rec:
            self.trades.append(rec)
        return True

    def _poll(self):
        now = datetime.now(IST)

        if self._today != now.date():
            self._today = now.date()
            self._fired_events.clear()
            logger.info(f"[ENG] New trading day {self._today}")

        if now.time() < self.cfg.SESSION_START or now.time() > self.cfg.SESSION_END:
            return

        # Always reconcile broker first.
        if not self.cfg.DRY_RUN:
            rec = self.om.sync_with_broker()
            if rec and rec.get("entry_failed"):
                return
            if rec and "pnl_rs" in rec:
                self.trades.append(rec)
                return

        # Risk / EOD checks take precedence over fresh entries.
        if self._check_risk():
            return
        if self._eod_exit():
            return

        # New entries are disabled after TRADE_END.
        if now.time() > self.cfg.TRADE_END:
            return

        df = self._drop_forming_bar(self._fetch_recent_bars())
        if df.empty:
            return

        events = generate_recent_signals(df, self.cfg, now.date())
        if not events:
            return

        events.sort(key=lambda x: x["timestamp"])

        # Handle the first not-yet-acted event in chronological order.
        next_ev = None
        for ev in events:
            key = (ev["timestamp"].isoformat(), ev["dir"], round(ev["zone_level"], 2))
            if key not in self._fired_events:
                next_ev = ev
                break

        if next_ev is None:
            return

        # A signal is historical once its closed candle has fired. Do not let it
        # repeatedly trigger every 5 seconds.
        event_key = (
            next_ev["timestamp"].isoformat(),
            next_ev["dir"],
            round(next_ev["zone_level"], 2),
        )
        self._fired_events.add(event_key)

        if self.om.has:
            logger.info(
                f"[ENG] Signal {next_ev['dir']} @ {next_ev['timestamp']} "
                f"ignored - position already open"
            )
            return

        sym = self._opt_symbol(next_ev["index_price"], next_ev["dir"])
        opt_ltp = self._ltp(sym)
        if opt_ltp is None:
            logger.warning(
                f"[ENG] No option LTP for {sym}; event not lost from strategy state "
                f"but entry was not possible this poll."
            )
            # Allow retry on next poll if quote was unavailable.
            self._fired_events.discard(event_key)
            return

        logger.info(
            f"[ENG] SIGNAL {next_ev['dir']} | time={next_ev['timestamp']} "
            f"zone={next_ev['zone_kind']} level={next_ev['zone_level']:.2f} "
            f"spot={next_ev['index_price']:.2f} -> {sym} option_ltp={opt_ltp:.2f}"
        )

        opened = self.om.open(
            sym,
            next_ev["dir"],
            opt_ltp,
            next_ev["index_price"],
            next_ev["zone_kind"],
            next_ev["zone_level"],
        )
        if not opened:
            logger.warning("[ENG] Entry was not opened")

    def run(self):
        logger.info("[ENG] starting S/R reversal live poll loop")
        try:
            while True:
                started = time.monotonic()
                try:
                    self._poll()
                except Exception as e:
                    logger.error(f"[ENG] poll error: {e}", exc_info=True)
                elapsed = time.monotonic() - started
                time.sleep(max(0.0, self.cfg.POLL_INTERVAL_SECS - elapsed))
        except KeyboardInterrupt:
            logger.info("[ENG] interrupted")
        finally:
            self._shutdown()

    def _shutdown(self):
        if self.om.has:
            ltp = self._ltp(self.om.symbol) or self.om.entry_price
            rec = self.om.close(ltp, "shutdown")
            if rec:
                self.trades.append(rec)
        self._export()
        logger.info("[ENG] shutdown complete")

    def _export(self):
        if not self.trades:
            logger.info("[ENG] no trades to export")
            return
        try:
            df = pd.DataFrame(self.trades)
            df["cum_pnl_rs"] = df["pnl_rs"].cumsum()
            logs_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
            os.makedirs(logs_dir, exist_ok=True)
            day = datetime.now(IST).strftime("%Y%m%d")
            path = os.path.join(logs_dir, f"smc_sr_live_trades_{day}.xlsx")
            df.to_excel(path, index=False)
            logger.info(
                f"[ENG] exported {len(df)} trades -> {path} | "
                f"net Rs {df['pnl_rs'].sum():,.0f} | "
                f"wins {(df['pnl_rs'] > 0).sum()}/{len(df)}"
            )
        except Exception as e:
            logger.error(f"[ENG] export error: {e}")


if __name__ == "__main__":
    Engine().run()
