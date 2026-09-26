"""
Sequential 3-candle Reversal + S/R Zone Confluence backtest for PatternPoll.py.

Layers the core C1/C2/C3 candle logic (imported verbatim from PatternPoll)
on top of an S/R zone filter: patterns only fire when they occur INSIDE a
matching active zone (SELL near resistance, BUY near support).

CANDLE LOGIC (unchanged, imported from PatternPoll.detect_signal):
  SELL: C1 green, C2 red, C3 red closing BELOW C1's open
  BUY : C1 red,   C2 green, C3 green closing ABOVE C1's open

ALL PATTERN FILTERS DISABLED via BacktestConfig overrides:
  USE_EMA_CROSS_FILTER=False, USE_VOLUME_FILTER=False,
  USE_BIG_C2_FAST_ENTRY=False, HEALTHY_BODY_FRAC=0.0

S/R ZONE CONFLUENCE (added by this backtest):
  - Swing pivots detected via LuxAlgo leg()/getCurrentStructure() port on the
    SAME timeframe as the pattern (TF_RULE, default 5min).
  - Pivot HIGH confirmed ZONE_SWING_LEN bars back -> RESISTANCE zone opens.
  - Pivot LOW  confirmed ZONE_SWING_LEN bars back -> SUPPORT    zone opens.
  - Zone tolerance = ZONE_TOL_MULT * ATR (half-width). Zone is [level-tol, level+tol].
  - Zone invalidated when a close moves ZONE_INVALIDATE_MULT * ATR beyond it.
  - Zones RESET at each new trading day (no overnight carry).
  - Cap of ZONE_MAX_ACTIVE_PER_SIDE zones per direction, oldest are held.

CONFLUENCE RULE:
  SELL fires only if C3.close is inside an active resistance zone.
  BUY  fires only if C3.close is inside an active support    zone.
  Signals that don't match are dropped and counted (logged at end).

RISK: fixed-point spot SL/TP (10 pts adverse / 30 pts favourable), intrabar
detection on 1-min index bars, SL priority if both hit same bar, exit fill =
option premium at trigger bar's close.

ENTRY GATING: while a position is open, ALL new signals are ignored. Position
closes only on SL, TP, or EOD. No flips. No re-entries until exit.

Config OVERRIDES: edit BacktestConfig below without touching live PatternPoll.py.
"""
import os, sys, warnings, logging, requests, urllib3, pytz
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Optional, List, Tuple

# ── Import signal logic + LiveConfig VERBATIM from the live bot ────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from PatternPoll import Config as LiveConfig, detect_signal

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
warnings.filterwarnings("ignore")
IST = pytz.timezone("Asia/Kolkata")

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - PatternBacktest - %(levelname)s - %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S")
logger = logging.getLogger("PatternBacktest")


# ══════════════════════════════════════════════════════════════════════════════
#  BACKTEST-SPECIFIC CONFIG OVERRIDES
# ══════════════════════════════════════════════════════════════════════════════
class BacktestConfig(LiveConfig):
    # ── Pattern filters: all OFF, only core C1/C2/C3 candle logic ──
    USE_EMA_CROSS_FILTER  = False
    USE_VOLUME_FILTER     = False
    USE_BIG_C2_FAST_ENTRY = False
    HEALTHY_BODY_FRAC     = 0.0

    # ── Fixed-point SL/TP on NIFTY SPOT ──
    STOP_LOSS_PTS   = 30
    TAKE_PROFIT_PTS = 60

    # ── S/R zone confluence filter ──
    # Only take pattern signals when C3.close is inside a matching active zone
    # (SELL -> resistance, BUY -> support). Set USE_ZONE_FILTER=False to bypass
    # the gate and get every pattern signal (useful for A/B comparison).
    USE_ZONE_FILTER          = True
    ZONE_SWING_LEN           = 10    # bars back to confirm a pivot (LuxAlgo "size")
    ZONE_ATR_PERIOD          = 5   # ATR window for tolerance sizing
    ZONE_TOL_MULT            = 1   # zone half-width = this * ATR
    ZONE_INVALIDATE_MULT     = 0.5   # close beyond zone by this * ATR -> invalidate
    ZONE_MAX_ACTIVE_PER_SIDE = 2    # cap tracked zones per direction

    # ── Optional tunables (uncomment to override): ──
    TF_RULE    = "10s"
    HIST_DAYS  = 30
    # LOT        = 65
    # MONEYNESS  = "ITM"
    # ENABLE_BUY = True


# ---------- LuxAlgo swing-pivot port (verbatim from SMC backtest) ----------
def detect_pivots(o, h, l, c, swing_len: int):
    """
    Port of LuxAlgo's leg()/getCurrentStructure() logic:
      newLegHigh at bar i  <=>  high[i-size] > highest(high, size) over the
                                window ending at bar i (inclusive).
      newLegLow  at bar i  <=>  low[i-size]  < lowest(low, size)  (mirror).
    A pivot HIGH is confirmed `size` bars back the first time newLegHigh
    becomes true (leg turns bearish); pivot LOW mirror.
    Returns (pivot_highs, pivot_lows), each a list of (confirm_i, piv_i, price).
    """
    n = len(c)
    pivot_highs, pivot_lows = [], []
    leg = 0  # 0 = bearish leg, 1 = bullish leg
    for i in range(swing_len, n):
        window_high = np.max(h[i - swing_len + 1: i + 1])
        window_low  = np.min(l[i - swing_len + 1: i + 1])
        piv_idx = i - swing_len
        new_leg_high = h[piv_idx] > window_high
        new_leg_low  = l[piv_idx] < window_low
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


# ---------- Fyers HTTP client ----------
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


# ---------- backtest bot ----------
class PatternBacktestBot:
    def __init__(self):
        self.cfg = BacktestConfig()
        self.fy  = self._init_fyers()
        logger.info(
            f"BacktestConfig | TF={self.cfg.TF_RULE} | filters=OFF | "
            f"SL={self.cfg.STOP_LOSS_PTS}pts TP={self.cfg.TAKE_PROFIT_PTS}pts (spot) | "
            f"zone_filter={self.cfg.USE_ZONE_FILTER} "
            f"(swing={self.cfg.ZONE_SWING_LEN}, tol={self.cfg.ZONE_TOL_MULT}xATR, "
            f"invalidate={self.cfg.ZONE_INVALIDATE_MULT}xATR)"
        )

    def _init_fyers(self) -> Fyers:
        for f in ["fyers_appid.txt", "fyers_token.txt"]:
            if not os.path.exists(f): raise FileNotFoundError(f)
        app = open("fyers_appid.txt").read().strip()
        tok = open("fyers_token.txt").read().strip()
        fy = Fyers(app, tok); p = fy.profile()
        if p.get("s") != "ok":
            raise ConnectionError(f"Fyers auth failed -- re-run GenerateToken.py.\nResponse: {p}")
        logger.info(f"Fyers connected -- {p['data']['name']}")
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
        logger.info(f"Loaded NIFTY50 1-min index data: {len(df)} rows across {self.cfg.HIST_DAYS} days")
        return df

    def _fetch_option_1min(self, symbol: str) -> pd.DataFrame:
        return self._fetch_history(symbol, self.cfg.HIST_DAYS).sort_index()

    def _opt_symbol(self, spot: float, direction: str) -> str:
        c = self.cfg
        atm = int(round(spot / c.STRIKE_STEP) * c.STRIKE_STEP); itm = c.MONEYNESS == "ITM"
        if direction == "BUY":
            strike = atm - c.STRIKE_STEP if itm else atm; typ = "CE"
        else:
            strike = atm + c.STRIKE_STEP if itm else atm; typ = "PE"
        return f"NSE:NIFTY{c.EXPIRY_CODE}{strike:05d}{typ}"

    # ── signal walk: continuous TF_RULE resample, per-day zone tracking, ──
    # ── detect_signal at each bar, filter by zone confluence.              ──

    def generate_signals(self, df_1min: pd.DataFrame) -> List[dict]:
        c = self.cfg
        r = df_1min.resample(c.TF_RULE).agg(
            {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
        ).dropna()
        warmup_needed = max(c.CONFIRM_END, c.ZONE_SWING_LEN + c.ZONE_ATR_PERIOD, 3) + 1
        if len(r) < warmup_needed:
            logger.warning(f"Insufficient {c.TF_RULE} bars: {len(r)} < {warmup_needed}")
            return []

        o  = r["open"].to_numpy(); h  = r["high"].to_numpy()
        l  = r["low"].to_numpy();  cl = r["close"].to_numpy()
        vol     = r["volume"].to_numpy()
        # EMA + vol_sma passed to detect_signal but IGNORED (filters off)
        ema     = r["close"].ewm(span=c.EMA_LEN, adjust=False).mean().to_numpy()
        vol_sma = r["volume"].rolling(c.VOL_SMA_LEN, min_periods=c.VOL_SMA_LEN).mean().to_numpy()

        # ── Per-day pivots + ATR (zones reset at each new day) ────────────────
        days = [ts.date() for ts in r.index]
        day_ranges: List[Tuple[int, int, object]] = []
        day_start = 0
        for i in range(1, len(r)):
            if days[i] != days[day_start]:
                day_ranges.append((day_start, i, days[day_start]))
                day_start = i
        day_ranges.append((day_start, len(r), days[day_start]))

        atr_values = np.full(len(r), np.nan)
        # bar_idx -> list of ("resistance"|"support", level, tol) to open at that bar
        zone_registrations: Dict[int, List[Tuple[str, float, float]]] = {}

        for start, end, _day in day_ranges:
            day_h = h[start:end]; day_l = l[start:end]; day_c = cl[start:end]
            n_day = end - start
            if n_day < c.ZONE_SWING_LEN + 1:
                continue
            prev_close = np.empty(n_day); prev_close[0] = day_c[0]; prev_close[1:] = day_c[:-1]
            tr = np.maximum(day_h - day_l, np.maximum(np.abs(day_h - prev_close), np.abs(day_l - prev_close)))
            day_atr = pd.Series(tr).ewm(alpha=1 / c.ZONE_ATR_PERIOD, adjust=False).mean().to_numpy()
            atr_values[start:end] = day_atr

            day_o = o[start:end]
            pivot_highs, pivot_lows = detect_pivots(day_o, day_h, day_l, day_c, c.ZONE_SWING_LEN)
            for confirm_i, _piv_i, level in pivot_highs:
                g = start + confirm_i
                tol = c.ZONE_TOL_MULT * day_atr[confirm_i]
                if not np.isnan(tol):
                    zone_registrations.setdefault(g, []).append(("resistance", level, tol))
            for confirm_i, _piv_i, level in pivot_lows:
                g = start + confirm_i
                tol = c.ZONE_TOL_MULT * day_atr[confirm_i]
                if not np.isnan(tol):
                    zone_registrations.setdefault(g, []).append(("support", level, tol))

        tf_delta = pd.Timedelta(c.TF_RULE)
        events = []
        zones: List[dict] = []   # active zones for the current day
        current_day = None
        zone_rejected = 0
        pattern_hits  = 0

        for i in range(2, len(r)):
            # Day change -> reset zone book (no overnight zones)
            if days[i] != current_day:
                current_day = days[i]
                zones = []

            # Register any newly-confirmed pivots as zones at this bar
            for kind, level, tol in zone_registrations.get(i, []):
                active_side = sum(1 for z in zones if z["kind"] == kind and z["active"])
                if active_side < c.ZONE_MAX_ACTIVE_PER_SIDE:
                    zones.append({"kind": kind, "level": level, "tol": tol, "active": True})

            # Zone invalidation: this bar's close moved decisively past the zone
            atr_i = atr_values[i]
            if not np.isnan(atr_i):
                inv_thr = c.ZONE_INVALIDATE_MULT * atr_i
                for z in zones:
                    if not z["active"]:
                        continue
                    if z["kind"] == "support" and cl[i] < z["level"] - inv_thr:
                        z["active"] = False
                    elif z["kind"] == "resistance" and cl[i] > z["level"] + inv_thr:
                        z["active"] = False

            bar_close_time = r.index[i] + tf_delta
            ts_time = bar_close_time.time()
            if not (c.SESSION_START <= ts_time <= c.SESSION_END):
                continue
            if ts_time < c.SIGNAL_START_TIME:
                continue

            sig = detect_signal(
                o[:i+1], h[:i+1], l[:i+1], cl[:i+1],
                ema[:i+1], vol[:i+1], vol_sma[:i+1],
                c,
            )
            if sig is None:
                continue
            pattern_hits += 1
            direction, name, _c1_idx, _c2_idx = sig
            spot = float(cl[i])

            # ── Zone confluence filter ────────────────────────────────────────
            zone_level: Optional[float] = None
            zone_kind:  Optional[str]   = None
            if c.USE_ZONE_FILTER:
                needed_kind = "resistance" if direction == "SELL" else "support"
                match = None
                for z in zones:
                    if not z["active"] or z["kind"] != needed_kind:
                        continue
                    lo, hi = z["level"] - z["tol"], z["level"] + z["tol"]
                    if lo <= spot <= hi:
                        match = z; break
                if match is None:
                    zone_rejected += 1
                    continue
                zone_level = match["level"]; zone_kind = match["kind"]

            events.append({
                "timestamp":  bar_close_time,
                "dir":        direction,
                "name":       name,
                "spot":       spot,
                "zone_kind":  zone_kind,
                "zone_level": zone_level,
            })

        logger.info(
            f"Pattern hits: {pattern_hits} | Zone-filtered: kept {len(events)}, "
            f"rejected {zone_rejected} (no matching active zone at C3.close)"
        )
        return events

    # ── backtest engine (spot-point risk model, no flips) ──

    def run_backtest(self) -> pd.DataFrame:
        c = self.cfg
        print("Starting Pattern + S/R Zone Confluence Backtest...")
        df_1min = self.get_index_data()
        events  = self.generate_signals(df_1min)
        if not events:
            print("No signals generated")
            return pd.DataFrame()

        events.sort(key=lambda e: e["timestamp"])
        for ev in events:
            ev["option_symbol"] = self._opt_symbol(ev["spot"], ev["dir"])

        option_cache: Dict[str, pd.DataFrame] = {}

        def get_option_df(sym):
            if sym not in option_cache:
                try:
                    option_cache[sym] = self._fetch_option_1min(sym)
                except ValueError as e:
                    logger.warning(f"Skipping {sym}: {e}")
                    option_cache[sym] = pd.DataFrame()
            return option_cache[sym]

        def opt_price_after(opt_df, ts, field="close"):
            """Both fields use side='left' -> pick the bar AT ts (not one after)."""
            if opt_df.empty:
                return None, None
            pos = opt_df.index.searchsorted(ts, side="left")
            if field == "open":
                if pos >= len(opt_df): return None, None
                return float(opt_df.iloc[int(pos)]["open"]), opt_df.index[int(pos)]
            if pos >= len(opt_df):
                pos = len(opt_df) - 1
            return float(opt_df.iloc[int(pos)]["close"]), opt_df.index[int(pos)]

        def spot_at(ts):
            pos = df_1min.index.searchsorted(ts, side="right") - 1
            if pos < 0: pos = 0
            if pos >= len(df_1min): pos = len(df_1min) - 1
            return float(df_1min.iloc[int(pos)]["close"])

        trades = []
        has = False
        symbol = direction = entry_price = entry_time = pattern = None
        entry_spot: Optional[float] = None
        entry_zone_kind: Optional[str] = None
        entry_zone_level: Optional[float] = None

        def reset_pos():
            nonlocal has, symbol, direction, entry_price, entry_time, pattern
            nonlocal entry_spot, entry_zone_kind, entry_zone_level
            has = False
            symbol = direction = entry_price = entry_time = pattern = None
            entry_spot = None
            entry_zone_kind = None; entry_zone_level = None

        def record_trade(exit_time, exit_price, reason, exit_spot):
            pnl_pts = exit_price - entry_price
            pnl_rs  = pnl_pts * c.LOT
            idx_move = None
            if entry_spot is not None and exit_spot is not None:
                idx_move = round(exit_spot - entry_spot, 2)
            trades.append({
                "direction":       direction,
                "option_symbol":   symbol,
                "pattern":         pattern,
                "zone_kind":       entry_zone_kind,
                "zone_level":      round(entry_zone_level, 2) if entry_zone_level is not None else None,
                "entry_time":      entry_time,
                "entry_spot":      round(entry_spot, 2) if entry_spot is not None else None,
                "entry_price":     round(entry_price, 2),
                "exit_time":       exit_time,
                "exit_spot":       round(exit_spot, 2) if exit_spot is not None else None,
                "exit_price":      round(exit_price, 2),
                "index_move_pts":  idx_move,
                "exit_reason":     reason,
                "pnl_pts":         round(pnl_pts, 2),
                "pnl_rs":          round(pnl_rs, 2),
                "result":          "profit" if pnl_pts > 0 else "loss",
            })

        for ev in events:
            ts = ev["timestamp"]

            # ── risk management: walk INDEX 1-min bars until SL/TP/EOD ──
            if has:
                opt_df       = get_option_df(symbol)
                entry_day    = entry_time.date()
                entry_eod_ts = IST.localize(datetime.combine(entry_day, c.EOD_EXIT_TIME))
                window_end   = min(ts, entry_eod_ts)
                idx_window   = df_1min[(df_1min.index > entry_time) & (df_1min.index <= window_end)]

                for bar_ts, ibar in idx_window.iterrows():
                    if not has: break

                    if bar_ts >= entry_eod_ts:
                        px, pxt = opt_price_after(opt_df, entry_eod_ts, field="close")
                        if px is not None:
                            record_trade(pxt, px, "eod_exit", spot_at(entry_eod_ts))
                            reset_pos()
                        break

                    bar_high = float(ibar["high"]); bar_low = float(ibar["low"])
                    if direction == "BUY":
                        sl_trigger = entry_spot - c.STOP_LOSS_PTS
                        tp_trigger = entry_spot + c.TAKE_PROFIT_PTS
                        sl_hit = bar_low  <= sl_trigger
                        tp_hit = bar_high >= tp_trigger
                    else:  # SELL
                        sl_trigger = entry_spot + c.STOP_LOSS_PTS
                        tp_trigger = entry_spot - c.TAKE_PROFIT_PTS
                        sl_hit = bar_high >= sl_trigger
                        tp_hit = bar_low  <= tp_trigger

                    if sl_hit:
                        exit_px, exit_ts = opt_price_after(opt_df, bar_ts, field="close")
                        if exit_px is not None:
                            record_trade(exit_ts, exit_px, "stop_loss", sl_trigger)
                            reset_pos()
                        continue
                    if tp_hit:
                        exit_px, exit_ts = opt_price_after(opt_df, bar_ts, field="close")
                        if exit_px is not None:
                            record_trade(exit_ts, exit_px, "target", tp_trigger)
                            reset_pos()
                        continue

                if has and ts >= entry_eod_ts:
                    px, pxt = opt_price_after(opt_df, entry_eod_ts, field="close")
                    if px is not None:
                        record_trade(pxt, px, "eod_exit", spot_at(entry_eod_ts))
                        reset_pos()

            # While holding, ignore every new signal until SL/TP/EOD closes us
            if has:
                continue

            if not (c.TRADE_START <= ts.time() <= c.TRADE_END):
                continue
            sym = ev["option_symbol"]
            opt_df = get_option_df(sym)
            px, pxt = opt_price_after(opt_df, ts, field="open")
            if px is None:
                continue
            entry_px_slipped = px * 1.005
            has              = True
            symbol           = sym
            direction        = ev["dir"]
            entry_price      = entry_px_slipped
            entry_time       = pxt
            pattern          = ev["name"]
            entry_spot       = float(ev["spot"])
            entry_zone_kind  = ev["zone_kind"]
            entry_zone_level = ev["zone_level"]

            delay_secs = (pxt - ts).total_seconds()
            if delay_secs > 60:
                logger.warning(
                    f"Entry {sym} {ev['dir']} signal@{ts} -> fill@{pxt} "
                    f"(delay {delay_secs:.0f}s -- likely thin option data)"
                )

        # Final flush at entry-day EOD if still open
        if has:
            opt_df = get_option_df(symbol)
            entry_day    = entry_time.date()
            entry_eod_ts = IST.localize(datetime.combine(entry_day, c.EOD_EXIT_TIME))
            px, pxt = opt_price_after(opt_df, entry_eod_ts, field="close")
            if px is None:
                last_ts = opt_df.index[-1] if not opt_df.empty else entry_time
                px, pxt = opt_price_after(opt_df, last_ts, field="close")
            if px is not None:
                record_trade(pxt, px, "backtest_end", spot_at(entry_eod_ts))
                reset_pos()

        trades_df = pd.DataFrame(trades)
        if trades_df.empty:
            print("No trades executed")
            return trades_df

        trades_df["cum_pnl_rs"] = trades_df["pnl_rs"].cumsum()

        total  = len(trades_df)
        wins   = int((trades_df["result"] == "profit").sum())
        losses = int((trades_df["result"] == "loss").sum())
        net    = trades_df["pnl_rs"].sum()
        wr     = round(wins / total * 100, 1) if total > 0 else 0.0

        csv_path = "Pattern_Zone_NIFTY_ITM_Backtest.csv"
        trades_df.to_csv(csv_path, index=False, encoding="utf-8")

        with open(csv_path, "a", encoding="utf-8") as f:
            for line in [
                "", "METRICS,VALUE",
                f"Total Trades,{total}", f"Wins,{wins}", f"Losses,{losses}",
                f"Win Rate (%),{wr}", f"Net PnL (Rs),{net:.2f}",
                f"TF Rule,{c.TF_RULE}",
                "Pattern Filters,ALL OFF",
                f"Zone Filter,{c.USE_ZONE_FILTER}",
                f"Zone Swing Len,{c.ZONE_SWING_LEN}",
                f"Zone Tol (xATR),{c.ZONE_TOL_MULT}",
                f"Zone Invalidate (xATR),{c.ZONE_INVALIDATE_MULT}",
                f"Zone Max Active/Side,{c.ZONE_MAX_ACTIVE_PER_SIDE}",
                f"BUY Enabled,{c.ENABLE_BUY}",
                f"SL Points (spot),{c.STOP_LOSS_PTS}",
                f"TP Points (spot),{c.TAKE_PROFIT_PTS}",
                f"Lot Size,{c.LOT}",
                f"Hist Days,{c.HIST_DAYS}",
            ]:
                f.write(line + "\n")

        by_reason = trades_df.groupby("exit_reason")["pnl_rs"].agg(["count", "sum"]).reset_index()
        by_zone   = trades_df.groupby("zone_kind")["pnl_rs"].agg(["count", "sum"]).reset_index()
        print(f"\n{'='*60}")
        print(f"  Contract Type       : {c.MONEYNESS}  |  TF: {c.TF_RULE}")
        print(f"  SL / TP (spot pts)  : {c.STOP_LOSS_PTS} / {c.TAKE_PROFIT_PTS}")
        print(f"  Zone Filter         : {c.USE_ZONE_FILTER}  "
              f"(swing={c.ZONE_SWING_LEN}, tol={c.ZONE_TOL_MULT}xATR)")
        print(f"  Total Trades        : {total}   (W {wins} / L {losses})")
        print(f"  Win Rate            : {wr}%")
        print(f"  Net PnL             : Rs {net:,.2f}")
        print(f"  Best / Worst        : Rs {trades_df['pnl_rs'].max():,.2f} / Rs {trades_df['pnl_rs'].min():,.2f}")
        print(f"{'-'*60}")
        print("  Exit-reason breakdown:")
        for _, row in by_reason.iterrows():
            print(f"    {row['exit_reason']:<15} count={int(row['count']):<3}  sum=Rs {row['sum']:>12,.2f}")
        print("  Zone-side breakdown:")
        for _, row in by_zone.iterrows():
            print(f"    {str(row['zone_kind']):<15} count={int(row['count']):<3}  sum=Rs {row['sum']:>12,.2f}")
        print(f"{'='*60}")
        print(f"Results saved -> {csv_path}")
        return trades_df


# --------------------------- MAIN ---------------------------

if __name__ == "__main__":
    bot    = PatternBacktestBot()
    trades = bot.run_backtest()