"""
test_exit_order.py
------------------
Verifies that FyersHTTP.exit_order() actually closes a filled BO.

Steps:
  1. Place a 1-lot deep-OTM NIFTY BO (minimal capital at risk)
  2. Wait for entry leg to fill
  3. Call exit_order on the parent order ID
  4. Confirm netQty == 0 in positions

Run ONLY during market hours (9:15–15:30 IST).
Place this file in the same folder as vam_poll_5S.py.
"""

import sys, os, time, argparse
from datetime import datetime
import pytz

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from vam_poll_5S import FyersHTTP, TradingConfig, format_expiry_code, get_next_tuesday

IST = pytz.timezone("Asia/Kolkata")

# ── Config ────────────────────────────────────────────────────────────────────
OTM_STEPS        = 5     # 5 × 100 = 500 pts OTM — cheap but liquid enough to fill
SL_PCT           = 50.0  # wide SL so it won't trigger during the test
TP_PCT           = 100.0 # unreachable TP
FILL_TIMEOUT_SEC = 60    # seconds to wait for fill before giving up
# ─────────────────────────────────────────────────────────────────────────────


def init_fyers() -> FyersHTTP:
    for f in ["fyers_appid.txt", "fyers_token.txt"]:
        if not os.path.exists(f):
            raise FileNotFoundError(f"Missing {f} — run from the bot folder")
    app_id = open("fyers_appid.txt").read().strip()
    token  = open("fyers_token.txt").read().strip()
    client = FyersHTTP(app_id, token)
    profile = client.get_profile()
    if profile.get("s") != "ok":
        raise ConnectionError(f"Auth failed: {profile}")
    print(f"✅ Connected: {profile['data']['name']}")
    return client


def get_spot(fyers: FyersHTTP) -> float:
    resp = fyers.quotes("NSE:NIFTY50-INDEX")
    lp   = float(resp["d"][0]["v"]["lp"])
    print(f"   NIFTY spot: {lp:.2f}")
    return lp


def build_otm_ce(spot: float, cfg: TradingConfig) -> str:
    # Always use WEEKLY expiry for better liquidity
    weekly_exp  = get_next_tuesday()
    weekly_code = format_expiry_code(weekly_exp)
    step        = cfg.STRIKE_STEP
    strike      = ((int(spot) // step) + OTM_STEPS) * step
    print(f"   Weekly expiry: {weekly_exp.date()} → code {weekly_code}")
    return f"NSE:NIFTY{weekly_code}{strike:05d}CE"


def get_ltp(fyers: FyersHTTP, symbol: str) -> float:
    resp = fyers.quotes(symbol)
    print(f"   Raw quote response: {resp}")
    try:
        v   = resp["d"][0]["v"]
        ltp = v.get("lp") or v.get("ask") or v.get("bid")
        if not ltp:
            raise ValueError(f"No tradeable price in response: {v}")
        return float(ltp)
    except (KeyError, IndexError) as e:
        raise ValueError(f"Unexpected quote response structure: {resp}") from e


def place_bo(fyers: FyersHTTP, symbol: str, ltp: float, cfg: TradingConfig) -> str:
    sl_pts    = max(1, int(round(ltp * SL_PCT  / 100.0)))
    tp_pts    = max(1, int(round(ltp * TP_PCT  / 100.0)))
    limit_px  = round(ltp * 1.02, 1)   # 2% above LTP for fast fill
    qty       = cfg.NIFTY_LOT_SIZE      # 1 lot

    data = {
        "symbol":       symbol,
        "qty":          qty,
        "type":         1,       # limit
        "side":         1,       # buy
        "productType":  "BO",
        "limitPrice":   limit_px,
        "stopPrice":    0,
        "validity":     "DAY",
        "disclosedQty": 0,
        "offlineOrder": False,
        "stopLoss":     float(sl_pts),
        "takeProfit":   float(tp_pts),
    }
    print(f"\n📤 Placing BO: {symbol}")
    print(f"   qty={qty}  limitPrice={limit_px:.1f}  SL_pts={sl_pts}  TP_pts={tp_pts}")

    resp = fyers.place_order(data=data)
    print(f"   Response: {resp}")

    if not (isinstance(resp, dict) and resp.get("s") == "ok"):
        raise RuntimeError(f"BO placement failed: {resp}")

    order_id = resp["id"]
    print(f"✅ Order placed — id={order_id}")
    return order_id


def wait_for_fill(fyers: FyersHTTP, order_id: str) -> bool:
    print(f"\n⏳ Polling for fill (timeout={FILL_TIMEOUT_SEC}s)...")
    deadline = time.monotonic() + FILL_TIMEOUT_SEC
    while time.monotonic() < deadline:
        try:
            ob = fyers.orderbook()
            if isinstance(ob, dict) and ob.get("s") == "ok":
                for order in ob.get("orderBook", []):
                    if order.get("id") == order_id and order.get("status") == 2:
                        print(f"✅ Entry filled at ₹{order.get('tradedPrice', '?')}")
                        return True
        except Exception as e:
            print(f"   orderbook poll error: {e}")
        time.sleep(2)

    print("⚠️  Timeout — order not filled within limit.")
    return False


def call_exit(fyers: FyersHTTP, symbol: str, qty: int) -> dict:
    print(f"\n📤 Calling exit_position (DELETE /positions) for {symbol}...")
    resp = fyers.exit_position(symbol=symbol, qty=qty)
    print(f"   Response: {resp}")
    return resp


def verify_closed(fyers: FyersHTTP, symbol: str) -> None:
    print(f"\n🔍 Checking broker positions for {symbol}...")
    time.sleep(2)   # give broker a moment to reflect
    resp = fyers.positions()
    if not isinstance(resp, dict) or resp.get("s") != "ok":
        print(f"   positions API error: {resp}")
        return

    for pos in resp.get("netPositions", []):
        if pos.get("symbol") == symbol:
            net_qty = pos.get("netQty", "?")
            if net_qty == 0:
                print(f"✅ PASS — netQty=0. exit_order correctly closes a filled BO.")
            else:
                print(f"❌ FAIL — netQty={net_qty}. Position still open at broker!")
                print("   ▶ Manually close it in the Fyers app NOW.")
                print("   ▶ Then fix close_position() to use the correct exit endpoint.")
            return

    print(f"✅ PASS — {symbol} absent from positions (fully closed).")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--yes", action="store_true",
                        help="Skip confirmation prompt (for cron/non-interactive use)")
    args = parser.parse_args()

    print("=" * 60)
    print("  exit_order verification — run during market hours only")
    print("=" * 60)
    print(f"  Time: {datetime.now(IST).strftime('%Y-%m-%d %H:%M:%S IST')}\n")

    fyers  = init_fyers()
    cfg    = TradingConfig()
    spot   = get_spot(fyers)
    symbol = build_otm_ce(spot, cfg)
    ltp    = get_ltp(fyers, symbol)

    print(f"   Option : {symbol}")
    print(f"   LTP    : ₹{ltp:.2f}")

    if ltp < 1.0:
        print("\n⚠️  LTP < ₹1 — option too illiquid. Reduce OTM_STEPS at the top of the script.")
        return

    if args.yes:
        print(f"\n[--yes] Skipping confirmation. Placing 1-lot BO on {symbol} at ~₹{ltp:.2f} "
              f"(cost ≈ ₹{ltp * cfg.NIFTY_LOT_SIZE:.0f})")
    else:
        confirm = input(
            f"\nPlace 1-lot BO on {symbol} at ~₹{ltp:.2f} "
            f"(cost ≈ ₹{ltp * cfg.NIFTY_LOT_SIZE:.0f})? [y/N]: "
        ).strip().lower()
        if confirm != "y":
            print("Aborted.")
            return

    order_id = place_bo(fyers, symbol, ltp, cfg)
    filled   = wait_for_fill(fyers, order_id)

    if not filled:
        print("\nOrder not filled — attempting cancel...")
        cancel = fyers.exit_order(data={"id": order_id})
        print(f"   Cancel response: {cancel}")
        print("Test inconclusive. Try again when market is more active, or reduce OTM_STEPS.")
        return

    print("\n⏳ Waiting for position to appear at broker (poll 1s, max 10s)...")
    settled = False
    for attempt in range(10):
        resp = fyers.positions()
        net_positions = resp.get("netPositions", []) if isinstance(resp, dict) else []
        for pos in net_positions:
            if pos.get("symbol") == symbol and int(pos.get("netQty", 0)) != 0:
                print(f"   ✅ Position confirmed at broker after {attempt + 1}s")
                settled = True
                break
        if settled:
            break
        print(f"   [{attempt + 1}/10] Position not yet visible — retrying...")
        time.sleep(1)

    if not settled:
        print("   ⚠️  Position not visible after 10s — attempting exit anyway.")

    exit_resp = call_exit(fyers, symbol, cfg.NIFTY_LOT_SIZE)

    if isinstance(exit_resp, dict) and exit_resp.get("s") == "ok":
        verify_closed(fyers, symbol)
    else:
        print(f"\n❌ exit_position returned non-ok for a FILLED BO.")
        print("   ▶ Manually close the position in Fyers app NOW.")
        print("   ▶ Response above should indicate the correct error from Fyers API.")


if __name__ == "__main__":
    main()