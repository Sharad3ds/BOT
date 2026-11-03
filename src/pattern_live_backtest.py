"""

Backtest for pattern_poll_live.py (Sequential 3-Candle Reversal + Swing-Zone).



Signal + zone logic is IMPORTED VERBATIM from pattern_poll_live to prevent

drift -- detect_signal, detect_fractal_pivots, most_recent_confirmed_zone,

near_zone, generate_today_events all come from the live bot. Only the

execution machinery (bar-walking, option-side risk simulation, EOD

boundary, direction-alternation tracking, CSV export) lives here.



CANDLE LOGIC (from pattern_poll_live.detect_signal):

  SELL: C1 green, C2 red, C3 (confirm) red. Fires if EITHER

        (a) C1.close > C2.close > C3.close, OR

        (b) C1 body < BODY_RATIO_MAX x C2 body (default 1.2x).

  BUY : exact mirror -- C1 red, C2 green, C3 green.



SWING ZONE FILTER (from pattern_poll_live.most_recent_confirmed_zone +

near_zone, backed by detect_fractal_pivots):

  - Fractal pivot detection: Pine ta.pivotlow/high port. Confirmed

    SWING_LOOKBACK bars after the pivot.

  - SELL requires C1 near a recent swing HIGH; BUY near a swing LOW.

  - "Near" = C1's [low, high] overlaps the zone extended by

    SWING_PROXIMITY_PCT% of the zone price on each side.



RISK MANAGEMENT (matches live pattern_poll_live behavior):

  - Fixed premium SL (-STOP_LOSS_PCT%) and premium TP (+TAKE_PROFIT_PCT%),

    checked intrabar against option 1-min high/low. Exit fill = the exact

    SL/TP price (assumes stop/limit order semantics). SL priority if both

    hit inside the same bar.

  - EOD forced exit at EOD_EXIT_TIME, anchored to ENTRY day (never carries

    across trading days).

  - No candle SL, no profit-lock, no flips, no re-entries while holding.



DIRECTION ALTERNATION: no CE directly after CE, no PE directly after PE.

Resets at each new trading day, matching the live bot exactly.



Config OVERRIDES: edit BacktestConfig below without touching the live bot.



KNOWN LIMITATION: EXPIRY_CODE is computed once at init using

get_next_tuesday(today), so ALL trades in the backtest are simulated

against the CURRENT week's option symbols. This is fine for short recent

windows (a few days) but produces invalid symbols for older backtest dates

where those weekly contracts didn't exist yet. To backtest a longer window

faithfully, wire in ExpiryResolver + per-date _opt_symbol_for_date() as

noted in the project's expiry_resolver.py.

"""

import os, sys, warnings, logging, requests, urllib3, pytz

import pandas as pd

from datetime import datetime, timedelta

from typing import Dict, Optional, List



# ── Import EVERYTHING signal-related VERBATIM from the live bot ────────────────
# Zero drift: change pattern_poll_live.py and this file automatically picks
# up the new candle rule, new zone logic, new proximity math -- whatever.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pattern_poll_live import (

    Config as LiveConfig,

    generate_today_events,

)



urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

warnings.filterwarnings("ignore")

IST = pytz.timezone("Asia/Kolkata")



logging.basicConfig(level=logging.INFO,

                    format="%(asctime)s - PatternLiveBacktest - %(levelname)s - %(message)s",

                    datefmt="%Y-%m-%d %H:%M:%S")

logger = logging.getLogger("PatternLiveBacktest")





# ══════════════════════════════════════════════════════════════════════════════
#  BACKTEST-SPECIFIC CONFIG OVERRIDES
#  --------------------------------------------------------------------------
#  Everything not listed here is inherited from pattern_poll_live.Config
#  (single source of truth for the live strategy). Change any value below to
#  tune the BACKTEST WITHOUT touching the live bot.
# ══════════════════════════════════════════════════════════════════════════════
class BacktestConfig(LiveConfig):

    # ── Backtest window ──
    HIST_DAYS = 30

    # EMA settings are inherited from pattern_poll_live.Config.
    # This backtest only adds the filter logic; it does not override live settings.

    # ── Common A/B knobs (uncomment to override live defaults) ──
    # USE_SWING_FILTER    = True    # False -> take every pattern signal
    # SWING_LOOKBACK      = 10
    # SWING_PROXIMITY_PCT = 0.5     # % of zone price
    # BODY_RATIO_MAX      = 1.2     # C1 body < this * C2 body
    # ENABLE_BUY          = True    # False -> SELL-only
    # ── Risk (matches live defaults) ──
    # USE_PREMIUM_SL   = True
    # STOP_LOSS_PCT    = 10.0       # -10% on option premium
    # USE_PREMIUM_TP   = True
    # TAKE_PROFIT_PCT  = 20.0       # +20% on option premium
    # ── Instrument (usually leave alone) ──
    # TF_RULE   = "1min"
    # LOT       = 65
    # MONEYNESS = "ITM"
# ---------- Fyers HTTP client ----------
class Fyers:

    T1 = "https://api-t1.fyers.in/api/v3"; T2 = "https://api-t1.fyers.in/data"



    def __init__(self, app, tok):

        auth = tok if ":" in tok else f"{app}:{tok}"

        self.s = requests.Session(); self.s.verify = False

        self.s.headers.update({"Authorization": auth})



    def profile(self):

        return self.s.get(f"{self.T1}/profile").json()



    def history(self, symbol, res, d_from, d_to):

        r = self.s.get(f"{self.T2}/history", params={

            "symbol": symbol, "resolution": res, "date_format": "1",

            "range_from": d_from, "range_to": d_to, "cont_flag": "0"})

        try: return r.json()

        except Exception: return {}





# ---------- backtest bot ----------
class PatternLiveBacktestBot:

    def __init__(self):

        self.cfg = BacktestConfig()

        self.fy  = self._init_fyers()
        self.skipped_ema = 0

        logger.info(

            f"BacktestConfig | TF={self.cfg.TF_RULE} | "

            f"body_ratio<{self.cfg.BODY_RATIO_MAX}xC2 | "

            f"swing_filter={self.cfg.USE_SWING_FILTER} "

            f"(lookback={self.cfg.SWING_LOOKBACK}, prox={self.cfg.SWING_PROXIMITY_PCT}%) | "
            f"EMA={'EMA'+str(self.cfg.EMA_PERIOD) if self.cfg.USE_EMA_FILTER else 'off'} | "
            f"SL={'-'+str(self.cfg.STOP_LOSS_PCT)+'%' if self.cfg.USE_PREMIUM_SL else 'off'} "

            f"TP={'+'+str(self.cfg.TAKE_PROFIT_PCT)+'%' if self.cfg.USE_PREMIUM_TP else 'off'} "

            f"(premium) | BUY={self.cfg.ENABLE_BUY} | direction-alternating (resets daily)"

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

        # Fetch extra NIFTY history before the requested backtest window so
        # EMA200 has a warm-up period instead of starting on the first test bar.
        fetch_days = self.cfg.HIST_DAYS + (
            self.cfg.EMA_LOOKBACK_DAYS if self.cfg.USE_EMA_FILTER else 0
        )
        end = datetime.now(IST)
        backtest_start = end - timedelta(days=self.cfg.HIST_DAYS)

        df = self._fetch_history(self.cfg.UNDERLYING, fetch_days)
        df = df.iloc[df.index.indexer_between_time(
            self.cfg.SESSION_START, self.cfg.SESSION_END
        )]

        if self.cfg.USE_EMA_FILTER:
            # EMA is calculated causally from NIFTY index closes.
            df[f"ema{self.cfg.EMA_PERIOD}"] = df["close"].ewm(
                span=self.cfg.EMA_PERIOD, adjust=False
            ).mean()

        # Only the requested HIST_DAYS are used for actual signal/trade testing.
        df = df.loc[df.index >= backtest_start].copy()

        logger.info(
            f"Loaded NIFTY50 1-min index data: {len(df)} rows across "
            f"{self.cfg.HIST_DAYS} days | "
            f"EMA{self.cfg.EMA_PERIOD}={'ON' if self.cfg.USE_EMA_FILTER else 'OFF'}"
        )
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



    # ── Signal generation: per-day using live's generate_today_events ──
    def generate_signals(self, df_1min: pd.DataFrame) -> List[dict]:

        """Generate the existing pattern/swing signals and apply the 200 EMA gate.

        The EMA rule matches the live strategy:
          SELL / PE -> NIFTY signal-bar close must be BELOW EMA200
          BUY / CE  -> NIFTY signal-bar close must be ABOVE EMA200

        The existing 3-candle pattern and swing-zone logic is not changed.
        """

        c = self.cfg
        all_events: List[dict] = []
        skipped_ema = 0

        for _day, day_df in df_1min.groupby(df_1min.index.date):
            # Let the backtest own the EMA gate. This prevents double-filtering
            # when the imported live generate_today_events() already contains
            # the same EMA200 rule. The candle pattern and swing-zone logic are
            # otherwise unchanged.
            ema_enabled = getattr(c, "USE_EMA_FILTER", False)
            c.USE_EMA_FILTER = False
            try:
                day_events = generate_today_events(day_df, c)
            finally:
                c.USE_EMA_FILTER = ema_enabled

            for ev in day_events:
                if c.USE_EMA_FILTER:
                    # The event timestamp is the candle CLOSE time.
                    # NIFTY/EMA rows are indexed by candle START time, so use
                    # the exact 1-minute bar that produced the signal.
                    signal_bar_ts = ev["timestamp"] - pd.Timedelta(c.TF_RULE)

                    try:
                        row = day_df.loc[signal_bar_ts]
                        ema = float(row[f"ema{c.EMA_PERIOD}"])
                        spot = float(row["close"])
                    except (KeyError, TypeError, ValueError):
                        skipped_ema += 1
                        continue

                    if ev["dir"] == "SELL" and spot >= ema:
                        skipped_ema += 1
                        continue

                    if ev["dir"] == "BUY" and spot <= ema:
                        skipped_ema += 1
                        continue

                all_events.append(ev)

        self.skipped_ema = skipped_ema
        logger.info(
            f"Generated {len(all_events)} valid signal events across all days "
            f"| EMA{c.EMA_PERIOD}={'ON' if c.USE_EMA_FILTER else 'OFF'} "
            f"| EMA-skipped={skipped_ema}"
        )
        return all_events


    # ── backtest engine (premium SL/TP, direction alternation, no flips) ──
    def run_backtest(self) -> pd.DataFrame:

        c = self.cfg

        print("Starting pattern_poll_live backtest (premium SL/TP, direction-alternating)...")

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

            """side='left' returns the option bar AT ts (not one strictly after) --

            same fix applied to the other backtests to avoid the 1-min entry delay."""

            if opt_df.empty:

                return None, None

            pos = opt_df.index.searchsorted(ts, side="left")

            if field == "open":

                if pos >= len(opt_df): return None, None

                return float(opt_df.iloc[int(pos)]["open"]), opt_df.index[int(pos)]

            if pos >= len(opt_df):

                pos = len(opt_df) - 1

            return float(opt_df.iloc[int(pos)]["close"]), opt_df.index[int(pos)]



        def opt_bar_at(opt_df, ts):

            """Returns the option OHLC bar at exactly ts (bar_start == ts).

            Used for intrabar SL/TP triggers per minute."""

            if opt_df.empty:

                return None

            pos = opt_df.index.searchsorted(ts, side="left")

            if pos >= len(opt_df):

                return None

            bar_ts = opt_df.index[int(pos)]

            if bar_ts != ts:

                return None

            row = opt_df.iloc[int(pos)]

            return float(row["open"]), float(row["high"]), float(row["low"]), float(row["close"])



        trades = []

        # position state
        has = False

        symbol = direction = entry_price = entry_time = pattern = None

        # direction-alternation state (resets per day)
        current_day = None

        last_trade_dir: Optional[str] = None

        # stats
        skipped_dir_alt = 0

        skipped_window  = 0

        skipped_holding = 0



        def reset_pos():

            nonlocal has, symbol, direction, entry_price, entry_time, pattern

            has = False

            symbol = direction = entry_price = entry_time = pattern = None



        def record_trade(exit_time, exit_price, reason):

            pnl_pts = exit_price - entry_price

            pnl_rs  = pnl_pts * c.LOT

            trades.append({

                "direction":     direction,

                "option_symbol": symbol,

                "pattern":       pattern,

                "entry_time":    entry_time,

                "entry_price":   round(entry_price, 2),

                "exit_time":     exit_time,

                "exit_price":    round(exit_price, 2),

                "exit_reason":   reason,

                "pnl_pts":       round(pnl_pts, 2),

                "pnl_rs":        round(pnl_rs, 2),

                "result":        "profit" if pnl_pts > 0 else "loss",

            })



        for ev in events:

            ts = ev["timestamp"]

            ev_day = ts.date()



            # Day boundary -> reset direction alternation (matches live behavior)
            if current_day is None or ev_day != current_day:

                current_day = ev_day

                last_trade_dir = None



            # ── If holding, walk option 1-min bars for SL/TP/EOD ──
            if has:

                opt_df       = get_option_df(symbol)

                entry_day    = entry_time.date()

                entry_eod_ts = IST.localize(datetime.combine(entry_day, c.EOD_EXIT_TIME))

                window_end   = min(ts, entry_eod_ts)

                opt_window   = opt_df[(opt_df.index > entry_time) & (opt_df.index <= window_end)]



                for bar_ts, obar in opt_window.iterrows():

                    if not has: break



                    # EOD force close
                    if bar_ts >= entry_eod_ts:

                        px, pxt = opt_price_after(opt_df, entry_eod_ts, field="close")

                        if px is not None:

                            record_trade(pxt, px, "eod_exit")

                            reset_pos()

                        break



                    opt_high = float(obar["high"])

                    opt_low  = float(obar["low"])

                    sl_price = entry_price * (1 - c.STOP_LOSS_PCT   / 100.0)

                    tp_price = entry_price * (1 + c.TAKE_PROFIT_PCT / 100.0)



                    # SL priority if both hit inside same bar (conservative)
                    if c.USE_PREMIUM_SL and opt_low <= sl_price:

                        record_trade(bar_ts, sl_price, "stop_loss")

                        reset_pos()

                        continue

                    if c.USE_PREMIUM_TP and opt_high >= tp_price:

                        record_trade(bar_ts, tp_price, "target")

                        reset_pos()

                        continue



                # Gap safeguard: past entry_eod_ts with no covering bar
                if has and ts >= entry_eod_ts:

                    px, pxt = opt_price_after(opt_df, entry_eod_ts, field="close")

                    if px is not None:

                        record_trade(pxt, px, "eod_exit")

                        reset_pos()



            # While holding, ignore every new signal (no flips, no re-entries)
            if has:

                skipped_holding += 1

                continue



            # Direction alternation gate
            if last_trade_dir is not None and ev["dir"] == last_trade_dir:

                skipped_dir_alt += 1

                continue



            # Trade window gate
            if not (c.TRADE_START <= ts.time() <= c.TRADE_END):

                skipped_window += 1

                continue



            # Open position
            sym = ev["option_symbol"]

            opt_df = get_option_df(sym)

            px, pxt = opt_price_after(opt_df, ts, field="open")

            if px is None:

                continue

            entry_px_slipped = px * 1.005   # 0.5% slippage, matches other backtests

            has         = True

            symbol      = sym

            direction   = ev["dir"]

            entry_price = entry_px_slipped

            entry_time  = pxt

            pattern     = ev["name"]

            last_trade_dir = ev["dir"]



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

                record_trade(pxt, px, "backtest_end")

                reset_pos()



        trades_df = pd.DataFrame(trades)

        if trades_df.empty:

            print("No trades executed")

            print(f"\n(For diagnostics: signals generated={len(events)}, "

                  f"skipped_ema={self.skipped_ema}, skipped_dir_alt={skipped_dir_alt}, "

                  f"skipped_holding={skipped_holding})")

            return trades_df



        trades_df["cum_pnl_rs"] = trades_df["pnl_rs"].cumsum()



        total  = len(trades_df)

        wins   = int((trades_df["result"] == "profit").sum())

        losses = int((trades_df["result"] == "loss").sum())

        net    = trades_df["pnl_rs"].sum()

        wr     = round(wins / total * 100, 1) if total > 0 else 0.0



        csv_path = "Pattern_Live_NIFTY_ITM_Backtest.csv"

        trades_df.to_csv(csv_path, index=False, encoding="utf-8")



        with open(csv_path, "a", encoding="utf-8") as f:

            for line in [

                "", "METRICS,VALUE",

                f"Total Trades,{total}", f"Wins,{wins}", f"Losses,{losses}",

                f"Win Rate (%),{wr}", f"Net PnL (Rs),{net:.2f}",

                f"TF Rule,{c.TF_RULE}",

                f"Body Ratio Max,{c.BODY_RATIO_MAX}",

                f"Swing Filter,{c.USE_SWING_FILTER}",

                f"Swing Lookback,{c.SWING_LOOKBACK}",

                f"Swing Proximity %,{c.SWING_PROXIMITY_PCT}",

                f"EMA Filter,{c.USE_EMA_FILTER}",

                f"EMA Period,{c.EMA_PERIOD}",

                f"EMA Lookback Days,{c.EMA_LOOKBACK_DAYS}",

                f"BUY Enabled,{c.ENABLE_BUY}",

                f"Premium SL %,{c.STOP_LOSS_PCT}",

                f"Premium TP %,{c.TAKE_PROFIT_PCT}",

                "Direction Alternation,daily reset",

                f"Signals Generated,{len(events)}",

                f"Skipped EMA,{self.skipped_ema}",

                f"Skipped Dir-Alt,{skipped_dir_alt}",

                f"Skipped Window,{skipped_window}",

                f"Skipped Holding,{skipped_holding}",

                f"Lot Size,{c.LOT}",

                f"Hist Days,{c.HIST_DAYS}",

            ]:

                f.write(line + "\n")



        by_reason = trades_df.groupby("exit_reason")["pnl_rs"].agg(["count", "sum"]).reset_index()

        by_dir    = trades_df.groupby("direction")["pnl_rs"].agg(["count", "sum"]).reset_index()

        print(f"\n{'='*62}")

        print(f"  Contract Type       : {c.MONEYNESS}  |  TF: {c.TF_RULE}")

        print(f"  Premium SL / TP     : -{c.STOP_LOSS_PCT}% / +{c.TAKE_PROFIT_PCT}%")

        print(f"  Swing Filter        : {c.USE_SWING_FILTER}  "

              f"(lookback={c.SWING_LOOKBACK}, prox={c.SWING_PROXIMITY_PCT}%)")

        print(f"  EMA Filter          : {c.USE_EMA_FILTER}  (period={c.EMA_PERIOD})")
        print(f"  Total Trades        : {total}   (W {wins} / L {losses})")

        print(f"  Win Rate            : {wr}%")

        print(f"  Net PnL             : Rs {net:,.2f}")

        print(f"  Best / Worst        : Rs {trades_df['pnl_rs'].max():,.2f} / Rs {trades_df['pnl_rs'].min():,.2f}")

        print(f"{'-'*62}")

        print("  Signal funnel:")

        print(f"    Signals generated       : {len(events)}")

        print(f"    Skipped EMA             : {self.skipped_ema}")

        print(f"    Skipped dir-alternation : {skipped_dir_alt}")

        print(f"    Skipped trade window    : {skipped_window}")

        print(f"    Skipped (in-position)   : {skipped_holding}")

        print(f"    Actual entries          : {total}")

        print("  Exit-reason breakdown:")

        for _, row in by_reason.iterrows():

            print(f"    {row['exit_reason']:<15} count={int(row['count']):<3}  sum=Rs {row['sum']:>12,.2f}")

        print("  Direction breakdown:")

        for _, row in by_dir.iterrows():

            print(f"    {str(row['direction']):<8} count={int(row['count']):<3}  sum=Rs {row['sum']:>12,.2f}")

        print(f"{'='*62}")

        print(f"Results saved -> {csv_path}")

        return trades_df





# --------------------------- MAIN ---------------------------
if __name__ == "__main__":

    bot    = PatternLiveBacktestBot()

    trades = bot.run_backtest()
