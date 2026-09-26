"""
Monthly cumulative-PnL rollup. Combines the daily trade Excel files the bots
write into one workbook per bot: Daily (per-day net + cumulative + drawdown),
Trades (all trades w/ running cum PnL), Summary.

Searches BOTH log folders because the bots write to different places:
  - Supertrend bot -> src/logs/
  - VAM bot        -> ../logs/  (parent)

Usage (from src/):
    python monthly_pnl.py                 # BOTH bots, current month
    python monthly_pnl.py all 202607
    python monthly_pnl.py supertrend
    python monthly_pnl.py vam 202607
Run with the bot's interpreter for formatted xlsx, e.g. /root/trading_env/bin/python3
"""
import os, sys, glob, re
from datetime import datetime
import pandas as pd, pytz
IST = pytz.timezone("Asia/Kolkata")

_HERE = os.path.dirname(os.path.abspath(__file__))
LOG_DIRS = [os.path.join(_HERE, "logs"), os.path.join(_HERE, "..", "logs")]  # search both
PREFIX = {"supertrend": "supertrend_trades_", "vam": "trades_"}

def _pnl_col(df):
    for c in ("pnl_rs", "pnl_rupees"):
        if c in df.columns:
            return c
    return None

def run_bot(bot, month):
    prefix = PREFIX.get(bot)
    if prefix is None:
        print(f"Unknown bot '{bot}' (use supertrend / vam / all)"); return
    files = []
    for ld in LOG_DIRS:
        files += glob.glob(os.path.join(ld, f"{prefix}{month}*.xlsx"))
    files = sorted(set(f for f in files if not os.path.basename(f).startswith("monthly_")))
    if not files:
        searched = " ; ".join(os.path.abspath(d) for d in LOG_DIRS)
        print(f"[{bot}] No daily files for {month}. Searched: {searched}")
        return
    out_dir = os.path.dirname(files[0])

    all_rows, daily = [], []
    for fp in files:
        m = re.search(r'(\d{8})\.xlsx$', fp)
        if not m:
            continue
        d = datetime.strptime(m.group(1), "%Y%m%d").date()
        try:
            df = pd.read_excel(fp, sheet_name=0)
        except Exception as e:
            print(f"[{bot}] skip {fp}: {e}"); continue
        pc = _pnl_col(df)
        if pc is None or df.empty:
            continue
        df = df.copy(); df.insert(0, "date", str(d))
        all_rows.append(df)
        net = float(df[pc].sum()); n = len(df); wins = int((df[pc] > 0).sum())
        daily.append({"date": str(d), "trades": n, "wins": wins, "losses": n - wins,
                      "win_pct": round(100*wins/n, 1) if n else 0.0, "net_pnl": round(net, 2)})
    if not all_rows:
        print(f"[{bot}] No trades found for {month}."); return

    trades = pd.concat(all_rows, ignore_index=True)
    pc = _pnl_col(trades)
    trades["cum_pnl"] = trades[pc].cumsum().round(2)
    dd = pd.DataFrame(daily)
    dd["cum_pnl"]  = dd["net_pnl"].cumsum().round(2)
    dd["drawdown"] = (dd["cum_pnl"] - dd["cum_pnl"].cummax()).round(2)

    total = len(trades); wins = int((trades[pc] > 0).sum()); net = float(trades[pc].sum())
    summary = [
        ("Bot", bot), ("Month", month), ("Trading days", len(dd)),
        ("Total trades", total), ("Wins", wins), ("Losses", total - wins),
        ("Win rate (%)", round(100*wins/total, 1) if total else 0.0),
        ("Net PnL (Rs)", round(net, 2)), ("Avg / trade (Rs)", round(net/total, 2) if total else 0.0),
        ("Best day (Rs)", round(dd["net_pnl"].max(), 2)), ("Worst day (Rs)", round(dd["net_pnl"].min(), 2)),
        ("Max drawdown (Rs)", round(dd["drawdown"].min(), 2)),
        ("Best trade (Rs)", round(trades[pc].max(), 2)), ("Worst trade (Rs)", round(trades[pc].min(), 2)),
    ]
    summary_df = pd.DataFrame(summary, columns=["Metric", "Value"])
    out = os.path.join(out_dir, f"monthly_pnl_{bot}_{month}.xlsx")

    try:
        import openpyxl
        from openpyxl.styles import PatternFill, Font, Alignment
        from openpyxl.utils import get_column_letter
        wb = openpyxl.Workbook()
        hf = PatternFill("solid", fgColor="1F4E79"); hfont = Font(color="FFFFFF", bold=True)
        green = PatternFill("solid", fgColor="C6EFCE"); red = PatternFill("solid", fgColor="FFC7CE")
        def sheet(ws, df, color_by=None):
            cols = list(df.columns)
            for ci, c in enumerate(cols, 1):
                cell = ws.cell(1, ci, str(c).replace("_", " ").title()); cell.fill = hf; cell.font = hfont
                cell.alignment = Alignment(horizontal="center")
            for ri, (_, row) in enumerate(df.iterrows(), 2):
                fill = None
                if color_by and color_by in df.columns:
                    fill = green if row[color_by] > 0 else (red if row[color_by] < 0 else None)
                for ci, c in enumerate(cols, 1):
                    cell = ws.cell(ri, ci, row[c]); cell.alignment = Alignment(horizontal="center")
                    if fill: cell.fill = fill
            for ci, c in enumerate(cols, 1):
                w = max([len(str(c))] + [len(str(df.iloc[r][c])) for r in range(len(df))]) if len(df) else len(str(c))
                ws.column_dimensions[get_column_letter(ci)].width = min(w + 3, 34)
        ws1 = wb.active; ws1.title = "Daily";  sheet(ws1, dd, color_by="net_pnl")
        ws2 = wb.create_sheet("Trades");        sheet(ws2, trades, color_by=pc)
        ws3 = wb.create_sheet("Summary")
        ws3.column_dimensions["A"].width = 22; ws3.column_dimensions["B"].width = 18
        for ri, (k, v) in enumerate(summary, 1):
            lc = ws3.cell(ri, 1, k); vc = ws3.cell(ri, 2, v)
            lc.fill = hf; lc.font = hfont; lc.alignment = Alignment(horizontal="left")
            vc.alignment = Alignment(horizontal="center")
            if k == "Net PnL (Rs)": vc.fill = green if net >= 0 else red; vc.font = Font(bold=True)
        wb.save(out)
        print(f"[{bot}] Wrote {out}")
    except ImportError:
        base = os.path.join(out_dir, f"monthly_pnl_{bot}_{month}")
        dd.to_csv(base + "_daily.csv", index=False)
        trades.to_csv(base + "_trades.csv", index=False)
        summary_df.to_csv(base + "_summary.csv", index=False)
        print(f"[{bot}] openpyxl missing — wrote CSVs at {base}_daily/_trades/_summary.csv "
              f"(use /root/trading_env/bin/python3 for formatted .xlsx)")
    print(f"[{bot}] {month}: {total} trades, {len(dd)} days, win% "
          f"{round(100*wins/total,1) if total else 0}, NET Rs {round(net,2):,}, "
          f"maxDD Rs {round(dd['drawdown'].min(),2):,}")

def main():
    arg   = sys.argv[1] if len(sys.argv) > 1 else "all"
    month = sys.argv[2] if len(sys.argv) > 2 else datetime.now(IST).strftime("%Y%m")
    bots  = ["supertrend", "vam"] if arg in ("all", "both") else [arg]
    for b in bots:
        run_bot(b, month)

if __name__ == "__main__":
    main()