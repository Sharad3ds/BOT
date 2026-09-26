"""
Target sweep on the NO-LOOK-AHEAD backtest.
Fetches data ONCE (via VAMBacktest_ATM_NoLookahead), then replays the SAME
signals/entries against several take-profit targets. Only tp_price varies;
entries, flips, stop-loss and EOD logic are identical across targets, so the
comparison is apples-to-apples.

Run in the same folder as fyers_appid.txt / fyers_token.txt, like the backtest:
    python VAM_TargetSweep_NoLookahead.py
"""
import pandas as pd
import numpy as np
from datetime import datetime
import VAMBacktest_ATM_NoLookahead as B   # reuse the no-look-ahead bot + data

IST = B.IST
STOP_LOSS_PCT = -20.0
TARGETS = [2, 5, 10, 15, 20, 25]          # % take-profit values to compare


def simulate(signals_sorted, df_idx_reset, option_cache, cfg, target_pct):
    EOD_EXIT = cfg.EOD_EXIT_TIME
    lot = cfg.NIFTY_LOT_SIZE
    pnls = []
    for i, s in signals_sorted.iterrows():
        signal_ts    = s["timestamp"]
        opt_symbol   = s["option_symbol"]
        direction    = s["direction"]
        pivot_lvl    = s.get("pivot_level")
        signal_price = float(s["index_price"])

        if signal_ts.time() >= EOD_EXIT:
            continue
        next_flip_ts = None
        for j in range(i + 1, len(signals_sorted)):
            if signals_sorted.iloc[j]["direction"] != direction:
                next_flip_ts = signals_sorted.iloc[j]["timestamp"]; break
        eod_boundary = IST.localize(datetime.combine(signal_ts.date(), EOD_EXIT))

        pos = df_idx_reset.index[df_idx_reset["timestamp"] == signal_ts]
        if len(pos) == 0 or pivot_lvl is None:
            continue
        sig_idx_pos = int(pos[0])
        prox = pivot_lvl * (cfg.PIVOT_PROXIMITY_PCT / 100.0)
        if direction == "BUY":
            near = (signal_price - pivot_lvl) <= prox
        else:
            near = (pivot_lvl - signal_price) <= prox
        entry_time_idx = None
        if near:
            entry_time_idx = signal_ts
        else:
            for k in range(sig_idx_pos + 1, len(df_idx_reset)):
                bt = df_idx_reset.iloc[k]["timestamp"]
                if bt >= eod_boundary: break
                if next_flip_ts is not None and bt >= next_flip_ts: break
                cp = float(df_idx_reset.iloc[k]["low"]) if direction == "BUY" else float(df_idx_reset.iloc[k]["high"])
                if abs(cp - pivot_lvl) <= prox:
                    entry_time_idx = bt; break
        if entry_time_idx is None or opt_symbol not in option_cache:
            continue
        opt_df = option_cache[opt_symbol]
        ip = opt_df.index.searchsorted(entry_time_idx, side="right")
        if ip >= len(opt_df):
            continue
        entry_price = float(opt_df.iloc[int(ip)]["open"]) * 1.005
        sl_price = entry_price * (1.0 + STOP_LOSS_PCT / 100.0)
        tp_price = entry_price * (1.0 + target_pct / 100.0)

        exit_price = None
        for k in range(int(ip) + 1, len(opt_df)):
            bar = opt_df.iloc[k]; bt = opt_df.index[k]
            if bt >= eod_boundary:
                exit_price = float(bar["close"]); break
            hit_sl = float(bar["low"]) <= sl_price
            hit_tp = float(bar["high"]) >= tp_price
            if hit_sl and hit_tp: exit_price = sl_price; break
            elif hit_sl:          exit_price = sl_price; break
            elif hit_tp:          exit_price = tp_price; break
            if next_flip_ts is not None and bt >= next_flip_ts:
                exit_price = float(bar["close"]); break
        if exit_price is None:
            continue
        pnl_pct = (exit_price - entry_price) / entry_price * 100.0
        pnls.append(entry_price * lot * pnl_pct / 100.0)
    return pnls


def main():
    bot = B.VAMBacktestBot()
    df_index = bot.get_index_data()
    df_idx_reset = df_index.reset_index()
    signals = bot.generate_vam_signals(df_index)
    if signals.empty:
        print("No signals."); return
    signals_sorted = signals.sort_values("timestamp").reset_index(drop=True)

    option_cache = {}
    for sym in signals_sorted["option_symbol"].unique():
        try:
            option_cache[sym] = bot._fetch_option_history(sym, bot.config.HIST_DAYS).sort_index()
        except Exception as e:
            print(f"  skip {sym}: {e}")

    lot = bot.config.NIFTY_LOT_SIZE
    print(f"\n1 lot = {lot} | data span: {df_idx_reset['timestamp'].min().date()} -> {df_idx_reset['timestamp'].max().date()}")
    print(f"\n{'TP%':>4} {'trades':>7} {'wins':>5} {'win%':>7} {'netPnL(Rs)':>12} {'avg/trade':>11} {'maxDD(Rs)':>11}")
    print("-" * 62)
    for tp in TARGETS:
        arr = np.array(simulate(signals_sorted, df_idx_reset, option_cache, bot.config, tp))
        n = len(arr)
        if n == 0:
            print(f"{tp:>4} {0:>7}"); continue
        wins = int((arr > 0).sum())
        net  = arr.sum()
        cum  = np.cumsum(arr); dd = (cum - np.maximum.accumulate(cum)).min()
        print(f"{tp:>4} {n:>7} {wins:>5} {100*wins/n:>6.1f}% {net:>12,.0f} {net/n:>11,.0f} {dd:>11,.0f}")


if __name__ == "__main__":
    main()
