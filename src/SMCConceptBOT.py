"""
Support/Resistance Reversal backtest — built on a simplified port of LuxAlgo's
"Smart Money Concepts" swing-structure detection (leg()/getCurrentStructure()).

Full SMC (order blocks, FVGs, equal highs/lows, premium/discount zones) is NOT
ported — only the core swing-pivot logic that everything else in that
indicator is built on. That's used here as support/resistance zone detection.

Trading logic (my own design on top of the S/R zones, since the original
indicator has no trade rules):
  1. SWING PIVOTS -> ZONES: a pivot low/high is confirmed `SWING_LEN` bars back
     using LuxAlgo's leg() rule (mirrors getCurrentStructure()). Each confirmed
     pivot becomes a zone: [level - tolerance, level + tolerance], tolerance
     scaled off ATR (same spirit as the indicator's volatility filter).
  2. ZONE RE-ENTRY: zones stay active until price closes decisively through
     them (invalidated, no trade) or price re-enters the zone.
  3. REVERSAL CONFIRMATION: once price re-enters an active zone, wait for a
     reversal candle -- a "healthy"-bodied candle closing back OUT of the zone
     in the expected direction (green closing above zone = BUY at support;
     red closing below zone = SELL at resistance). This is the same
     healthy-body reversal concept used in PatternPoll.py, applied at S/R.
  4. ONE TRADE PER ZONE: a zone is retired after it fires or is invalidated,
     so the same level can't re-trigger repeatedly.

Causal — everything is evaluated on CLOSED bars only, walking forward bar by
bar exactly like the other backtests in this project.

INTRADAY ENFORCEMENT: every position's EOD boundary is computed from its own
ENTRY day (entry_time.date()), not from whatever day the next signal happens
to land on. A position is always force-closed at EOD_EXIT_TIME on the same
day it was opened, regardless of how long it takes for another signal to
appear -- so no trade can span multiple calendar days.
"""
import pandas as pd, numpy as np, requests, logging, os, calendar, pytz, urllib3, warnings
from datetime import datetime, timedelta, time as dtime
from typing import Optional, Dict, List

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
warnings.filterwarnings("ignore")
IST = pytz.timezone("Asia/Kolkata")

logging.basicConfig(level=logging.INFO,
                     format="%(asctime)s - SMCBacktest - %(levelname)s - %(message)s",
                     datefmt="%Y-%m-%d %H:%M:%S")
logger = logging.getLogger("SMCBacktest")

# ---------- expiry helpers (identical convention to the other bots) ----------
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

# ---------- config ----------
class Config:
    UNDERLYING = "NSE:NIFTY50-INDEX"; BASE_RES = "1"; HIST_DAYS = 10
    TF_RULE = "1min"                       # timeframe candles/pivots/reversal run on

    SWING_LEN = 20                          # bars back to confirm a pivot (leg() "size")
    ATR_PERIOD = 15                          # for zone tolerance + healthy-body reference
    ZONE_TOL_MULT = 0.1                    # zone half-width = this * ATR
    ZONE_INVALIDATE_MULT = 0.1               # close beyond zone by this * ATR -> invalidate
    HEALTHY_BODY_FRAC = 0.001                 # reversal candle body >= this fraction of range
    MAX_ACTIVE_ZONES_PER_SIDE = 5            # cap on tracked support / resistance zones

    STRIKE_STEP = 100; LOT = 65; MONEYNESS = "ITM"
    SESSION_START = dtime(9, 20); SESSION_END = dtime(15, 30)
    SIGNAL_START_TIME = dtime(9, 35)
    TRADE_START = dtime(9, 30); TRADE_END = dtime(15, 0); EOD_EXIT_TIME = dtime(15, 20)

    # premium risk (option-side, simple % since the S/R logic has no native SL/TP)
    STOP_LOSS_PCT =20
    TAKE_PROFIT_PCT = 60

    def __init__(self):
        self.EXPIRY_CODE = expiry_code(get_next_tuesday())
        logger.info(f"Weekly expiry code {self.EXPIRY_CODE} | S/R reversal on {self.TF_RULE} "
                    f"swing_len={self.SWING_LEN} zone_tol={self.ZONE_TOL_MULT}xATR | "
                    f"SL={self.STOP_LOSS_PCT}% TP={self.TAKE_PROFIT_PCT}%")

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

# ---------- swing pivot detection (simplified port of LuxAlgo's leg()) ----------
def detect_pivots(o, h, l, c, swing_len: int):
    """
    Direct port of LuxAlgo's leg()/getCurrentStructure() pivot logic:
      newLegHigh at bar i  <=>  high[i-size] > highest(high, size) measured
                                over the window ending at bar i (inclusive).
      newLegLow  at bar i  <=>  low[i-size]  < lowest(low, size)  (mirror).
    A pivot HIGH is confirmed `size` bars back the first time newLegHigh flips
    true (i.e. leg turns bearish); a pivot LOW is confirmed the first time
    newLegLow flips true (leg turns bullish). Only the FIRST flip counts (leg
    only updates on a genuine change), matching startOfNewLeg().

    Returns two lists of (confirm_bar_idx, pivot_bar_idx, price):
      pivot_highs, pivot_lows
    confirm_bar_idx = the bar at which the pivot becomes known (causal point);
    pivot_bar_idx = the actual bar `size` back where the extreme occurred.
    """
    n = len(c)
    pivot_highs, pivot_lows = [], []
    leg = 0  # 0 = bearish leg, 1 = bullish leg (LuxAlgo convention)

    for i in range(swing_len, n):
        window_high = np.max(h[i - swing_len + 1: i + 1])
        window_low  = np.min(l[i - swing_len + 1: i + 1])
        piv_idx = i - swing_len

        new_leg_high = h[piv_idx] > window_high
        new_leg_low  = l[piv_idx] < window_low

        new_leg = leg
        if new_leg_high:
            new_leg = 0  # BEARISH_LEG
        elif new_leg_low:
            new_leg = 1  # BULLISH_LEG

        if new_leg != leg:  # startOfNewLeg
            if new_leg == 1:   # startOfBullishLeg -> pivot LOW confirmed
                pivot_lows.append((i, piv_idx, float(l[piv_idx])))
            else:              # startOfBearishLeg -> pivot HIGH confirmed
                pivot_highs.append((i, piv_idx, float(h[piv_idx])))
            leg = new_leg

    return pivot_highs, pivot_lows

# ---------- backtest bot ----------
class SMCBacktestBot:
    def __init__(self):
        self.cfg = Config()
        self.fy = self._init_fyers()

    def _init_fyers(self) -> Fyers:
        for f in ["fyers_appid.txt", "fyers_token.txt"]:
            if not os.path.exists(f): raise FileNotFoundError(f)
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
        logger.info(f"Loaded NIFTY50 1-min index data: {len(df)} rows")
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

    # ── zone + reversal signal generation, replayed bar-by-bar per day ──

    def generate_signals(self, df_1min: pd.DataFrame) -> list:
        c = self.cfg
        events = []

        for day, day_df in df_1min.groupby(df_1min.index.date):
            r = day_df.resample(c.TF_RULE).agg(
                {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
            ).dropna()
            if len(r) < c.SWING_LEN + c.ATR_PERIOD + 2:
                continue

            o = r["open"].to_numpy(); h = r["high"].to_numpy()
            l = r["low"].to_numpy(); cl = r["close"].to_numpy()

            prev_close = np.empty(len(r)); prev_close[0] = cl[0]; prev_close[1:] = cl[:-1]
            tr = np.maximum(h - l, np.maximum(np.abs(h - prev_close), np.abs(l - prev_close)))
            atr = pd.Series(tr).ewm(alpha=1 / c.ATR_PERIOD, adjust=False).mean().to_numpy()

            pivot_highs, pivot_lows = detect_pivots(o, h, l, cl, c.SWING_LEN)

            # zones known at confirm_idx: {"level":..., "tol":..., "kind": "support"/"resistance",
            #                              "state": "watching"/"in_zone", "active": True}
            zones: List[dict] = []
            # index confirmed pivots by confirm_bar_idx for O(1) insertion during the walk
            highs_by_confirm = {}
            for confirm_i, piv_i, level in pivot_highs:
                highs_by_confirm.setdefault(confirm_i, []).append((piv_i, level))
            lows_by_confirm = {}
            for confirm_i, piv_i, level in pivot_lows:
                lows_by_confirm.setdefault(confirm_i, []).append((piv_i, level))

            def healthy(k):
                rng = h[k] - l[k]
                return rng > 0 and abs(cl[k] - o[k]) >= c.HEALTHY_BODY_FRAC * rng
            green = lambda k: cl[k] > o[k]
            red   = lambda k: cl[k] < o[k]

            for i in range(c.SWING_LEN, len(r)):
                bar_close_time = r.index[i] + pd.Timedelta(c.TF_RULE)
                ts_time = bar_close_time.time()
                if not (c.SESSION_START <= ts_time <= c.SESSION_END):
                    continue

                # register any newly-confirmed pivots as zones at this bar
                tol = c.ZONE_TOL_MULT * atr[i] if not np.isnan(atr[i]) else None
                if tol is not None:
                    for piv_i, level in highs_by_confirm.get(i, []):
                        n_active = sum(1 for z in zones if z["kind"] == "resistance" and z["active"])
                        if n_active < c.MAX_ACTIVE_ZONES_PER_SIDE:
                            zones.append({"level": level, "tol": tol, "kind": "resistance",
                                          "state": "watching", "active": True})
                    for piv_i, level in lows_by_confirm.get(i, []):
                        n_active = sum(1 for z in zones if z["kind"] == "support" and z["active"])
                        if n_active < c.MAX_ACTIVE_ZONES_PER_SIDE:
                            zones.append({"level": level, "tol": tol, "kind": "support",
                                          "state": "watching", "active": True})

                if ts_time < c.SIGNAL_START_TIME:
                    continue

                # walk active zones: check invalidation / re-entry / reversal fire
                for z in zones:
                    if not z["active"]:
                        continue
                    lo, hi = z["level"] - z["tol"], z["level"] + z["tol"]
                    in_zone = lo <= cl[i] <= hi

                    if z["kind"] == "support":
                        invalidated = cl[i] < z["level"] - c.ZONE_INVALIDATE_MULT * atr[i]
                    else:
                        invalidated = cl[i] > z["level"] + c.ZONE_INVALIDATE_MULT * atr[i]

                    if z["state"] == "watching":
                        if invalidated:
                            z["active"] = False
                            continue
                        if in_zone:
                            z["state"] = "in_zone"
                        continue

                    if z["state"] == "in_zone":
                        if invalidated:
                            z["active"] = False
                            continue
                        # reversal confirmation candle
                        if z["kind"] == "support" and green(i) and healthy(i) and cl[i] > hi:
                            events.append({
                                "timestamp": bar_close_time, "dir": "BUY",
                                "zone_level": z["level"], "zone_kind": "support",
                                "index_price": float(cl[i]),
                            })
                            z["active"] = False
                        elif z["kind"] == "resistance" and red(i) and healthy(i) and cl[i] < lo:
                            events.append({
                                "timestamp": bar_close_time, "dir": "SELL",
                                "zone_level": z["level"], "zone_kind": "resistance",
                                "index_price": float(cl[i]),
                            })
                            z["active"] = False
                        # else: still sitting in zone, keep watching next bar

        logger.info(f"Generated {len(events)} S/R reversal signal events across all days")
        return events

    # ── backtest engine: entries at next option-bar open, premium SL/TP, EOD ──

    def run_backtest(self) -> pd.DataFrame:
        c = self.cfg
        print("Starting Support/Resistance Reversal Backtest (ITM contracts)...")
        df_1min = self.get_index_data()
        events = self.generate_signals(df_1min)
        if not events:
            print("No signals generated")
            return pd.DataFrame()

        events.sort(key=lambda e: e["timestamp"])
        for ev in events:
            ev["option_symbol"] = self._opt_symbol(ev["index_price"], ev["dir"])

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
            if opt_df.empty: return None, None
            pos = opt_df.index.searchsorted(ts, side="right" if field == "open" else "left")
            if field == "open":
                if pos >= len(opt_df): return None, None
                return float(opt_df.iloc[int(pos)]["open"]), opt_df.index[int(pos)]
            if pos >= len(opt_df):
                pos = len(opt_df) - 1
            return float(opt_df.iloc[int(pos)]["close"]), opt_df.index[int(pos)]

        trades = []
        has = False; symbol = None; direction = None; entry_price = None; entry_time = None
        zone_level = None; zone_kind = None

        for ev in events:
            ts = ev["timestamp"]

            # ── risk management: walk option bars between entry and this event ──
            # EOD boundary MUST be based on the ENTRY's own day (intraday trade),
            # not on ts (the day of whatever the next signal happens to be) --
            # previously this let positions ride for days if no new signal fired
            # in between, instead of force-closing same-day like a real intraday
            # bot must.
            if has:
                opt_df = get_option_df(symbol)
                entry_day = entry_time.date()
                entry_eod_ts = IST.localize(datetime.combine(entry_day, c.EOD_EXIT_TIME))
                window_end = min(ts, entry_eod_ts)
                window = opt_df[(opt_df.index > entry_time) & (opt_df.index <= window_end)]
                for bar_ts, bar in window.iterrows():
                    if not has: break
                    if bar_ts >= entry_eod_ts:
                        px, pxt = opt_price_after(opt_df, entry_eod_ts)
                        if px is not None:
                            trades.append(self._close(direction, symbol, zone_kind, zone_level,
                                                       entry_time, entry_price, pxt, px, "eod_exit", c.LOT))
                            has = False
                        break
                    px = float(bar["close"])
                    if px <= entry_price * (1 - c.STOP_LOSS_PCT / 100.0):
                        trades.append(self._close(direction, symbol, zone_kind, zone_level,
                                                   entry_time, entry_price, bar_ts, px, "stop_loss", c.LOT))
                        has = False; continue
                    if px >= entry_price * (1 + c.TAKE_PROFIT_PCT / 100.0):
                        trades.append(self._close(direction, symbol, zone_kind, zone_level,
                                                   entry_time, entry_price, bar_ts, px, "target", c.LOT))
                        has = False; continue

                # window didn't include a bar exactly at/after EOD (data gap),
                # but we've clearly passed the entry day's EOD -- force-close.
                if has and ts >= entry_eod_ts:
                    px, pxt = opt_price_after(opt_df, entry_eod_ts)
                    if px is not None:
                        trades.append(self._close(direction, symbol, zone_kind, zone_level,
                                                   entry_time, entry_price, pxt, px, "eod_exit", c.LOT))
                    has = False

            if has:
                continue  # already in a trade, this backtest is single-position (no flips modeled)

            in_win = c.TRADE_START <= ts.time() <= c.TRADE_END
            if not in_win:
                continue

            sym = ev["option_symbol"]
            opt_df = get_option_df(sym)
            entry_px, entry_ts_actual = opt_price_after(opt_df, ts, field="open")
            if entry_px is None:
                continue
            entry_px *= 1.005  # small slippage, matches other backtests' convention

            has = True; symbol = sym; direction = ev["dir"]
            entry_price = entry_px; entry_time = entry_ts_actual
            zone_level = ev["zone_level"]; zone_kind = ev["zone_kind"]

        # ── final flush: if STILL open here, force-close at the entry day's
        # EOD price (not at whatever the last available option bar happens to
        # be, days later, which would misrepresent an intraday trade). ──
        if has:
            opt_df = get_option_df(symbol)
            entry_day = entry_time.date()
            entry_eod_ts = IST.localize(datetime.combine(entry_day, c.EOD_EXIT_TIME))
            px, pxt = opt_price_after(opt_df, entry_eod_ts)
            if px is None:
                last_ts = opt_df.index[-1] if not opt_df.empty else entry_time
                px, pxt = opt_price_after(opt_df, last_ts)
            if px is not None:
                trades.append(self._close(direction, symbol, zone_kind, zone_level,
                                           entry_time, entry_price, pxt, px, "backtest_end", c.LOT))

        trades_df = pd.DataFrame(trades)
        if trades_df.empty:
            print("No trades executed")
            return trades_df

        trades_df["cum_pnl_rs"] = trades_df["pnl_rs"].cumsum()

        wins = int((trades_df["result"] == "profit").sum())
        losses = int((trades_df["result"] == "loss").sum())
        total = len(trades_df)
        net = trades_df["pnl_rs"].sum()
        wr = round(wins / total * 100, 1) if total > 0 else 0.0

        csv_path = "SMC_SupportResistance_NIFTY_ITM_Backtest.csv"
        trades_df.to_csv(csv_path, index=False, encoding="utf-8")

        with open(csv_path, "a", encoding="utf-8") as f:
            for line in [
                "", "METRICS,VALUE",
                f"Total Trades,{total}", f"Wins,{wins}", f"Losses,{losses}",
                f"Win Rate (%),{wr}", f"Net PnL (Rs),{net:.2f}",
                f"TF Rule,{c.TF_RULE}", f"Swing Len,{c.SWING_LEN}",
                f"Zone Tol (xATR),{c.ZONE_TOL_MULT}", f"Lot Size,{c.LOT}",
            ]:
                f.write(line + "\n")

        print(f"\n{'='*55}")
        print(f"  Contract Type       : {c.MONEYNESS}")
        print(f"  Total Trades        : {total}")
        print(f"  Win Rate            : {wr}%")
        print(f"  Net PnL             : Rs {net:,.2f}")
        print(f"{'='*55}")
        print(f"Results saved → {csv_path}")
        return trades_df

    @staticmethod
    def _close(direction, symbol, zone_kind, zone_level, entry_time, entry_price,
               exit_time, exit_price, reason, lot):
        pnl_pts = exit_price - entry_price
        pnl_rs = pnl_pts * lot
        return {
            "direction": direction, "option_symbol": symbol,
            "zone_kind": zone_kind, "zone_level": round(zone_level, 2),
            "entry_time": entry_time, "entry_price": round(entry_price, 2),
            "exit_time": exit_time, "exit_price": round(exit_price, 2),
            "exit_reason": reason,
            "pnl_pts": round(pnl_pts, 2), "pnl_rs": round(pnl_rs, 2),
            "result": "profit" if pnl_pts > 0 else "loss",
        }


# --------------------------- MAIN ---------------------------

if __name__ == "__main__":
    bot    = SMCBacktestBot()
    trades = bot.run_backtest()