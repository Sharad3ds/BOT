"""Backtest: First Candle High/Low Level Retest -> C1/C2/C3 Reversal

Rules:
- The first NIFTY 1-minute candle at 09:15 IST defines fixed ORH/ORL for the day.
- Price must first move away from a level, then return to its tolerance zone. A
  return/retest is valid from either side; a full candle close beyond ORH/ORL is
  not mandatory. This is deliberate so both breakout retests and in-range
  support/resistance re-tests can be captured like the user's chart examples.
- The retest candle may itself be C1. C1 may also begin up to the configured
  delay after the retest, but the C1/C2/C3 window must remain near that same level.
- If C1/C2 colors reverse directionally and C2 real body >= C1 real body, confirm on C2,
  ignore C3 completely, and enter at the next bar open (C2 must close on the trade side
  of the tested level).
- Otherwise use the strict three-candle fallback: bullish C1 red/C2 green/C3 green or
  bearish C1 green/C2 red/C3 red; C3 must close on the trade side of the tested level.
- The three-candle fallback retains the C3 body/range and close-progression/body-ratio filters.
- A retest is used at most once per level, to avoid duplicate entries from overlapping windows.
- Maximum 5 non-overlapping trades/day by default; spot SL/TP 40/100 points, EOD
  exit 15:20 IST, entries through 15:00 IST.

Options are resolved from FYERS contract metadata. This is a backtest only; it does
not place live orders. Validate actual historical performance before use.
"""

import os
import re
import json
import hashlib
import logging
import warnings
from datetime import datetime, timedelta, time as dtime, date, timezone
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import pytz
import requests
import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

IST = pytz.timezone("Asia/Kolkata")


class Config:
    UNDERLYING = "NSE:NIFTY50-INDEX"
    BASE_RES = "1"
    TF_RULE = "1min"

    HIST_DAYS = 30
    SESSION_START = dtime(9, 15)
    SESSION_END = dtime(15, 30)
    ENTRY_START = dtime(9, 16)
    ENTRY_END = dtime(15, 0)
    EOD_EXIT_TIME = dtime(15, 20)

    OPENING_CANDLE_TIME = dtime(9, 15)
    STRIKE_STEP = 100
    MONEYNESS = "ITM"
    LOT = 650

    STOP_LOSS_PTS = 40.0
    TAKE_PROFIT_PTS = 100.0

    # Applies only to the three-candle fallback. Stronger C1/C2 reversals ignore C3.
    HEALTHY_BODY_FRAC = 0.50
    BODY_RATIO_MAX = 1.20
    CONFIRM_MUST_BE_COLORED = True

    # A candle within this distance of ORH/ORL counts as a level test/retest.
    LEVEL_TOUCH_TOLERANCE_PTS = 10.0
    MAX_C1_START_BARS_AFTER_RETEST = 5  # C1 may be the retest candle or start within 5 bars afterward.
    MAX_TRADES_PER_DAY = 5  # Only take a new trade after the previous trade has exited.

    SLIPPAGE_PCT = 0.5  # Option entry premium slippage, applied to modeled entry price.
    API_TIMEOUT_SECS = 20

    # Official FYERS NSE_FO master provides exact active option symbols.
    SYMBOL_MASTER_URL = "https://public.fyers.in/sym_details/NSE_FO_sym_master.json"

    # FYERS Standard plans may limit expired F&O intraday history requests to
    # 10/day. Successful expired-history downloads are cached and reused.
    # Set to a higher value only if your FYERS plan allows more requests.
    MAX_EXPIRED_HISTORY_REQUESTS_PER_DAY = 10
    MAX_EXPIRY_DAYS_AFTER_TRADE = 10  # Guard against selecting a later expiry if metadata is missing.
    OPTION_CACHE_DIR = "First_Candle_SR_OptionCache"
    EXPIRED_USAGE_FILE = "First_Candle_SR_ExpiredApiUsage.json"

    TRADES_CSV = "First_Candle_High_Low_SR_C12StrongBodyConfirmation_Backtest.csv"
    SUMMARY_CSV = "First_Candle_High_Low_SR_C12StrongBodyConfirmation_Summary.csv"
    DIAGNOSTICS_CSV = "First_Candle_High_Low_SR_C12StrongBodyConfirmation_Diagnostics.csv"


# ---------- FYERS symbol/date parsing helpers ----------
def parse_date_value(value) -> Optional[date]:
    """Parse dates or FYERS expiry timestamps without guessing on unknown values."""
    if value is None or value == "":
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if isinstance(value, (int, float)):
        try:
            number = float(value)
            if number <= 0:
                return None
            if number > 10_000_000_000:  # milliseconds, if supplied
                number /= 1000.0
            if number > 100_000_000:
                return datetime.fromtimestamp(number, tz=timezone.utc).date()
        except (ValueError, OverflowError, OSError):
            return None
    text = str(value).strip()
    if not text:
        return None
    if re.fullmatch(r"\d{9,13}", text):
        try:
            number = int(text)
            if number > 10_000_000_000:
                number //= 1000
            return datetime.fromtimestamp(number, tz=timezone.utc).date()
        except (ValueError, OverflowError, OSError):
            return None
    for fmt in ("%Y-%m-%d", "%d-%m-%Y", "%d/%m/%Y", "%Y/%m/%d", "%d %b %Y", "%d-%b-%Y"):
        try:
            return datetime.strptime(text[:11], fmt).date()
        except ValueError:
            continue
    try:
        parsed = pd.to_datetime(text, errors="coerce", utc=True)
        if pd.notna(parsed):
            return parsed.date()
    except Exception:
        pass
    return None


def _expiry_prefixes(expiry_day: date) -> List[str]:
    """Return known FYERS NIFTY weekly/monthly symbol prefixes for one expiry date.

    This is used only to parse the strike from an exact symbol returned by FYERS;
    the backtest no longer invents a symbol to request option history.
    """
    yy = f"{expiry_day.year % 100:02d}"
    month_code = str(expiry_day.month) if expiry_day.month <= 9 else {10: "O", 11: "N", 12: "D"}[expiry_day.month]
    prefixes = [f"{yy}{month_code}{expiry_day.day:02d}", f"{yy}{expiry_day.strftime('%b').upper()}"]
    return sorted(set(prefixes), key=len, reverse=True)


def parse_nifty_option_symbol(symbol: str, expiry_day: date) -> Optional[dict]:
    """Parse strike/type from an exact symbol supplied by a FYERS data source."""
    if not isinstance(symbol, str) or not symbol.upper().endswith(("CE", "PE")):
        return None
    upper = symbol.upper().strip()
    ticker = upper.split(":", 1)[-1]
    if not ticker.startswith("NIFTY"):
        return None
    body = ticker[len("NIFTY"):-2]
    for prefix in _expiry_prefixes(expiry_day):
        if body.startswith(prefix):
            strike_text = body[len(prefix):]
            if strike_text.isdigit():
                return {"symbol": upper, "expiry_date": expiry_day,
                        "strike": float(strike_text), "option_type": upper[-2:]}
    return None


def extract_expiry_dates(response: dict) -> List[date]:
    """Extract option expiry dates from the documented FYERS response shape."""
    if not isinstance(response, dict):
        return []
    data = response.get("data", response)
    root = data.get("expiry_dates", data) if isinstance(data, dict) else data
    if isinstance(root, dict):
        items = root.get("options", root.get("option_expiry_dates", []))
    else:
        items = root
    if isinstance(items, dict):
        items = list(items.values())
    if not isinstance(items, list):
        return []
    dates_found = set()
    for item in items:
        candidates = []
        if isinstance(item, dict):
            for key in ("expiry_date", "expiryDate", "date", "expiry", "expiry_date_epoch"):
                if key in item:
                    candidates.append(item[key])
        else:
            candidates.append(item)
        for candidate in candidates:
            parsed = parse_date_value(candidate)
            if parsed:
                dates_found.add(parsed)
                break
    return sorted(dates_found)


def _walk_option_items(value):
    """Yield possible option symbol records from nested FYERS contracts JSON."""
    if isinstance(value, str):
        if value.upper().startswith("NSE:NIFTY") and value.upper().endswith(("CE", "PE")):
            yield value
        return
    if isinstance(value, list):
        for item in value:
            yield from _walk_option_items(item)
        return
    if not isinstance(value, dict):
        return

    symbol_keys = ("symbol", "symTicker", "ticker", "trading_symbol", "tradingSymbol", "option_symbol")
    for key in symbol_keys:
        symbol = value.get(key)
        if isinstance(symbol, str) and symbol.upper().startswith("NSE:NIFTY") and symbol.upper().endswith(("CE", "PE")):
            yield value
            return
    for key, nested in value.items():
        if isinstance(key, str) and key.upper().startswith("NSE:NIFTY") and key.upper().endswith(("CE", "PE")):
            if isinstance(nested, dict):
                record = dict(nested)
                record.setdefault("symbol", key)
                yield record
            else:
                yield key
        else:
            yield from _walk_option_items(nested)


def extract_option_contract_items(response: dict) -> List[object]:
    """Extract CE/PE contract records from the FYERS underlying-symbols response."""
    if not isinstance(response, dict):
        return []
    data = response.get("data", response)
    if not isinstance(data, dict):
        return []
    contracts = data.get("contracts", data)
    options = contracts.get("options") if isinstance(contracts, dict) else None
    source = options if options is not None else data
    found = list(_walk_option_items(source))
    # Some response versions use a flat list of symbol strings and some nest
    # option records more deeply. Deduplicate by exact symbol while preserving order.
    result, seen = [], set()
    for item in found:
        if isinstance(item, str):
            symbol = item
        elif isinstance(item, dict):
            symbol = next((item.get(k) for k in ("symbol", "symTicker", "ticker", "trading_symbol", "tradingSymbol", "option_symbol")
                           if isinstance(item.get(k), str)), None)
        else:
            symbol = None
        if symbol and symbol.upper() not in seen:
            seen.add(symbol.upper())
            result.append(item)
    return result


# ---------- FYERS historical-data client ----------
class FyersHistory:
    T1 = "https://api-t1.fyers.in/api/v3"
    T2 = "https://api-t1.fyers.in/data"

    def __init__(self, app_id: str, token: str, timeout: int = 20):
        auth = token if ":" in token else f"{app_id}:{token}"
        self.session = requests.Session()
        self.session.verify = False  # Mirrors the current project's FYERS client setup.
        self.session.headers.update({"Authorization": auth})
        self.timeout = timeout

    def profile(self) -> dict:
        try:
            response = self.session.get(f"{self.T1}/profile", timeout=self.timeout)
            return response.json()
        except Exception as exc:
            return {"s": "error", "message": str(exc)}

    def history(self, symbol: str, resolution: str, start: date, end: date) -> dict:
        try:
            response = self.session.get(
                f"{self.T2}/history",
                params={
                    "symbol": symbol,
                    "resolution": resolution,
                    "date_format": "1",
                    "range_from": start.strftime("%Y-%m-%d"),
                    "range_to": end.strftime("%Y-%m-%d"),
                    "cont_flag": "0",
                },
                timeout=self.timeout,
            )
            try:
                return response.json()
            except Exception:
                return {"s": "error", "message": f"Non-JSON history response (HTTP {response.status_code})"}
        except Exception as exc:
            return {"s": "error", "message": str(exc)}

    def expired_expiry_dates(self, start: date, end: date) -> dict:
        return self._data_get("/history/fno/expired/expiry-dates", {
            "symbol": "NSE:NIFTY50-INDEX",
            "range_from": start.strftime("%Y-%m-%d"),
            "range_to": end.strftime("%Y-%m-%d"),
            "date_format": "1",
        })

    def expired_underlying_symbols(self, expiry_day: date) -> dict:
        return self._data_get("/history/fno/expired/underlying-symbols", {
            "symbol": "NSE:NIFTY50-INDEX",
            "expiry_date": expiry_day.strftime("%Y-%m-%d"),
        })

    def expired_history(self, symbol: str, start: date, end: date) -> dict:
        return self._data_get("/history/fno/expired/historical-data", {
            "symbol": symbol,
            "resolution": "1",
            "date_format": "1",
            "range_from": start.strftime("%Y-%m-%d"),
            "range_to": end.strftime("%Y-%m-%d"),
            "include_greeks": "0",
            "include_oi": "0",
        })

    def _data_get(self, path: str, params: dict) -> dict:
        try:
            response = self.session.get(f"{self.T2}{path}", params=params, timeout=self.timeout)
            try:
                return response.json()
            except Exception:
                return {"s": "error", "message": f"Non-JSON response (HTTP {response.status_code})"}
        except Exception as exc:
            return {"s": "error", "message": str(exc)}


def parse_candles(resp: dict) -> pd.DataFrame:
    candles = resp.get("candles", []) if isinstance(resp, dict) else []
    if not candles:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
    df = pd.DataFrame(candles, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s", utc=True).dt.tz_convert(IST)
    df = df.drop_duplicates("timestamp").sort_values("timestamp").set_index("timestamp")
    for col in ("open", "high", "low", "close", "volume"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.dropna(subset=["open", "high", "low", "close"])


class FirstCandleSRBacktest:
    def __init__(self):
        self.cfg = Config()
        for name in ("fyers_appid.txt", "fyers_token.txt"):
            if not os.path.exists(name):
                raise FileNotFoundError(f"Required credential file not found: {name}")
        with open("fyers_appid.txt", "r", encoding="utf-8") as f:
            app_id = f.read().strip()
        with open("fyers_token.txt", "r", encoding="utf-8") as f:
            token = f.read().strip()
        self.fyers = FyersHistory(app_id, token, self.cfg.API_TIMEOUT_SECS)
        self.option_cache: Dict[str, pd.DataFrame] = {}
        self.diagnostics: List[dict] = []
        self.contracts_by_expiry: Dict[date, Dict[str, dict]] = {}
        self.catalog_expiries = set()
        self.backtest_start = datetime.now(IST).date() - timedelta(days=self.cfg.HIST_DAYS)
        self.backtest_end = datetime.now(IST).date()

        p = self.fyers.profile()
        if not (isinstance(p, dict) and p.get("s") == "ok"):
            raise ConnectionError(f"FYERS authentication failed: {p}")
        logging.info("Connected to FYERS account profile.")
        logging.info(
            "Config | ORH/ORL move-away -> retest -> strong C1/C2 or C1/C2/C3 reversal strategy | opening candle=%s | SL=%s spot points | TP=%s spot points | "
            "ITM=%s | qty=%s | max %s non-overlapping trades/day | exact FYERS symbol catalog | expired-F&O API enabled",
            self.cfg.OPENING_CANDLE_TIME, self.cfg.STOP_LOSS_PTS, self.cfg.TAKE_PROFIT_PTS,
            self.cfg.MONEYNESS, self.cfg.LOT, self.cfg.MAX_TRADES_PER_DAY,
        )

    def fetch_history(self, symbol: str, start: date, end: date) -> pd.DataFrame:
        resp = self.fyers.history(symbol, self.cfg.BASE_RES, start, end)
        if not (isinstance(resp, dict) and resp.get("s") == "ok"):
            msg = resp.get("message", str(resp)) if isinstance(resp, dict) else str(resp)
            raise ValueError(f"History API error for {symbol}: {msg} (code={resp.get('code') if isinstance(resp, dict) else 'n/a'})")
        df = parse_candles(resp)
        if df.empty:
            raise ValueError(f"No historical candles returned for {symbol}")
        return df

    def get_index_data(self) -> pd.DataFrame:
        """Fetch NIFTY history in daily chunks to reduce risk of API range truncation."""
        end_dt = datetime.now(IST)
        start_day = (end_dt - timedelta(days=self.cfg.HIST_DAYS)).date()
        end_day = end_dt.date()
        daily_frames = []
        api_failures = 0

        day = start_day
        while day <= end_day:
            # Skip weekends; weekday exchange holidays will simply return no bars.
            if day.weekday() < 5:
                resp = self.fyers.history(self.cfg.UNDERLYING, self.cfg.BASE_RES, day, day)
                if isinstance(resp, dict) and resp.get("s") == "ok":
                    chunk = parse_candles(resp)
                    if not chunk.empty:
                        daily_frames.append(chunk)
                else:
                    api_failures += 1
                    logging.warning(
                        "NIFTY history unavailable for %s: %s",
                        day,
                        resp.get("message", resp) if isinstance(resp, dict) else resp,
                    )
            day += timedelta(days=1)

        if not daily_frames:
            raise ValueError("No NIFTY candles returned by FYERS in the requested history window.")
        df = pd.concat(daily_frames).sort_index()
        df = df[~df.index.duplicated(keep="last")]
        df = df.between_time(self.cfg.SESSION_START, self.cfg.SESSION_END, inclusive="both")
        if df.empty:
            raise ValueError("No NIFTY index candles found inside the configured market session.")
        logging.info(
            "Loaded %s NIFTY 1-minute candles across %s calendar days (daily-chunk fetch, API failures=%s).",
            len(df), self.cfg.HIST_DAYS, api_failures,
        )
        return df.sort_index()

    def _script_dir(self) -> str:
        return os.path.dirname(os.path.abspath(__file__))

    def _cache_dir(self) -> str:
        path = os.path.join(self._script_dir(), self.cfg.OPTION_CACHE_DIR)
        os.makedirs(path, exist_ok=True)
        return path

    def _option_cache_path(self, symbol: str) -> str:
        digest = hashlib.sha1(symbol.upper().encode("utf-8")).hexdigest()[:12]
        safe = re.sub(r"[^A-Za-z0-9_-]+", "_", symbol)
        return os.path.join(self._cache_dir(), f"{safe[:60]}_{digest}.csv")

    def _load_option_cache(self, symbol: str) -> pd.DataFrame:
        path = self._option_cache_path(symbol)
        if not os.path.exists(path):
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
        try:
            df = pd.read_csv(path)
            if "timestamp" not in df.columns:
                return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
            df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True).dt.tz_convert(IST)
            df = df.dropna(subset=["timestamp"]).drop_duplicates("timestamp").set_index("timestamp").sort_index()
            for col in ("open", "high", "low", "close", "volume"):
                if col not in df.columns:
                    df[col] = np.nan
                df[col] = pd.to_numeric(df[col], errors="coerce")
            return df[["open", "high", "low", "close", "volume"]].dropna(subset=["open", "high", "low", "close"])
        except Exception as exc:
            logging.warning("Could not read option cache for %s: %s", symbol, exc)
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

    def _save_option_cache(self, symbol: str, df: pd.DataFrame) -> None:
        if df.empty:
            return
        path = self._option_cache_path(symbol)
        try:
            out = df.copy().reset_index()
            out = out.rename(columns={out.columns[0]: "timestamp"})
            tmp = path + ".tmp"
            out.to_csv(tmp, index=False, encoding="utf-8")
            os.replace(tmp, path)
        except Exception as exc:
            logging.warning("Could not save option cache for %s: %s", symbol, exc)

    def _load_daily_expired_usage(self) -> dict:
        path = os.path.join(self._script_dir(), self.cfg.EXPIRED_USAGE_FILE)
        today = datetime.now(IST).date().isoformat()
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if data.get("date") == today:
                return data
        except Exception:
            pass
        return {"date": today, "requests": 0}

    def _save_daily_expired_usage(self, data: dict) -> None:
        path = os.path.join(self._script_dir(), self.cfg.EXPIRED_USAGE_FILE)
        tmp = path + ".tmp"
        try:
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(data, f)
            os.replace(tmp, path)
        except Exception as exc:
            logging.warning("Could not persist expired-F&O API usage counter: %s", exc)

    def _expired_history_request_allowed(self) -> bool:
        usage = self._load_daily_expired_usage()
        return int(usage.get("requests", 0)) < int(self.cfg.MAX_EXPIRED_HISTORY_REQUESTS_PER_DAY)

    def _increment_expired_history_usage(self) -> None:
        usage = self._load_daily_expired_usage()
        usage["requests"] = int(usage.get("requests", 0)) + 1
        self._save_daily_expired_usage(usage)

    @staticmethod
    def _symbol_from_record(item) -> Optional[str]:
        if isinstance(item, str):
            return item.upper()
        if isinstance(item, dict):
            for key in ("symbol", "symTicker", "ticker", "trading_symbol", "tradingSymbol", "option_symbol"):
                value = item.get(key)
                if isinstance(value, str) and value.upper().startswith("NSE:NIFTY"):
                    return value.upper()
        return None

    @staticmethod
    def _contract_from_master(symbol: str, item: dict) -> Optional[dict]:
        if not isinstance(item, dict):
            return None
        symbol = str(item.get("symTicker") or symbol).upper()
        if not symbol.startswith("NSE:NIFTY") or not symbol.endswith(("CE", "PE")):
            return None
        expiry_day = None
        for key in ("expiryDate", "expiry_date", "expiry", "expDate"):
            if key in item:
                expiry_day = parse_date_value(item.get(key))
                if expiry_day:
                    break
        try:
            strike = float(item.get("strikePrice", item.get("strike_price", item.get("strike", 0))))
        except (TypeError, ValueError):
            strike = 0.0
        option_type = str(item.get("optType", item.get("optionType", item.get("option_type", symbol[-2:])))).upper()
        if not expiry_day or strike <= 0 or option_type not in ("CE", "PE"):
            return None
        return {"symbol": symbol, "expiry_date": expiry_day, "strike": strike,
                "option_type": option_type, "source": "symbol_master"}

    def _contract_from_expired_item(self, item, expiry_day: date) -> Optional[dict]:
        symbol = self._symbol_from_record(item)
        if not symbol:
            return None
        strike = None
        option_type = symbol[-2:]
        if isinstance(item, dict):
            try:
                candidate = item.get("strikePrice", item.get("strike_price", item.get("strike")))
                if candidate not in (None, "", -1, "-1"):
                    strike = float(candidate)
            except (TypeError, ValueError):
                strike = None
            option_type = str(item.get("optType", item.get("optionType", item.get("option_type", option_type)))).upper()
        parsed = parse_nifty_option_symbol(symbol, expiry_day)
        if strike is None and parsed:
            strike = parsed["strike"]
        if not strike or option_type not in ("CE", "PE"):
            return None
        return {"symbol": symbol, "expiry_date": expiry_day, "strike": float(strike),
                "option_type": option_type, "source": "expired_api"}

    def prepare_contract_catalog(self, start_day: date, end_day: date) -> None:
        """Build exact option symbols from FYERS symbol master + expired contract metadata.

        Symbols are never synthesized and sent to History. The backtest uses an
        exact contract returned by a FYERS-authoritative source, preventing the
        previous -300 errors caused by assuming an expiry/strike symbol exists.
        """
        self.contracts_by_expiry: Dict[date, Dict[str, dict]] = {}
        self.catalog_expiries = set()
        today = datetime.now(IST).date()
        master_count = 0

        try:
            response = requests.get(self.cfg.SYMBOL_MASTER_URL, timeout=self.cfg.API_TIMEOUT_SECS)
            response.raise_for_status()
            master = response.json()
            if isinstance(master, dict):
                for key, item in master.items():
                    contract = self._contract_from_master(str(key), item)
                    if contract:
                        self._add_contract(contract)
                        master_count += 1
            logging.info("Loaded %s NIFTY CE/PE contracts from current FYERS NSE_FO symbol master.", master_count)
        except Exception as exc:
            logging.warning("Could not load FYERS NSE_FO symbol master: %s", exc)
            self.diagnostics.append({"date": "", "event_time": "", "direction": "", "symbol": "",
                                     "status": "SYMBOL_MASTER_UNAVAILABLE", "details": str(exc)})

        # FYERS rejects range_to=today (and future dates) on the expired-expiry
        # lookup endpoint. Query only through the latest fully elapsed calendar day.
        # Current/near-future contracts are sourced from the current symbol master above.
        expiry_lookup_end = min(end_day, today - timedelta(days=1))
        expired_dates = []
        if expiry_lookup_end < start_day:
            logging.info(
                "Skipping expired-expiry lookup: backtest window has no past dates "
                "(start=%s, lookup_end=%s); current contracts come from symbol master.",
                start_day, expiry_lookup_end,
            )
        else:
            try:
                expiry_resp = self.fyers.expired_expiry_dates(start_day, expiry_lookup_end)
                expired_dates = extract_expiry_dates(expiry_resp) if expiry_resp.get("s") == "ok" else []
                if expired_dates:
                    logging.info(
                        "FYERS expired F&O metadata returned %s expiry dates in %s..%s.",
                        len(expired_dates), start_day, expiry_lookup_end,
                    )
                elif expiry_resp.get("s") != "ok":
                    self.diagnostics.append({
                        "date": "", "event_time": "", "direction": "", "symbol": "",
                        "status": "EXPIRED_EXPIRY_LOOKUP_FAILED",
                        "details": f"FYERS expired-expiry lookup failed for range {start_day}..{expiry_lookup_end}: {str(expiry_resp)[:900]}",
                    })
                    logging.warning("Expired F&O expiry-date lookup failed: %s", expiry_resp)
                else:
                    self.diagnostics.append({
                        "date": "", "event_time": "", "direction": "", "symbol": "",
                        "status": "EXPIRED_EXPIRY_LIST_EMPTY",
                        "details": (
                            f"FYERS returned no expired expiry dates for {start_day}..{expiry_lookup_end}. "
                            "Historical trades can only use contracts resolved from the current symbol master."
                        ),
                    })
            except Exception as exc:
                expired_dates = []
                logging.warning("Expired F&O expiry-date lookup failed: %s", exc)
                self.diagnostics.append({
                    "date": "", "event_time": "", "direction": "", "symbol": "",
                    "status": "EXPIRED_EXPIRY_LOOKUP_FAILED", "details": str(exc),
                })

        # Current master may include this/next expiry; expired endpoint supplies dates
        # and exact symbols for older contracts. Never pass today/future to this API.
        relevant_expiries = sorted(
            d for d in set(expired_dates)
            if start_day <= d <= expiry_lookup_end
        )
        for expiry_day in relevant_expiries:
            # If this expiry's symbols are already available from the current master,
            # still fetch historical contracts because the active master is not a
            # historical symbol master and may not contain the old chain.
            try:
                resp = self.fyers.expired_underlying_symbols(expiry_day)
                if not (isinstance(resp, dict) and resp.get("s") == "ok"):
                    self.diagnostics.append({"date": expiry_day.isoformat(), "event_time": "", "direction": "", "symbol": "",
                                             "status": "EXPIRED_CONTRACT_LOOKUP_FAILED", "details": str(resp)[:1000]})
                    logging.warning("Expired contract-list lookup failed for %s: %s", expiry_day, resp)
                    continue
                items = extract_option_contract_items(resp)
                found = 0
                for item in items:
                    contract = self._contract_from_expired_item(item, expiry_day)
                    if contract:
                        self._add_contract(contract)
                        found += 1
                logging.info("Loaded %s expired CE/PE contract symbols for expiry %s.", found, expiry_day)
                if found == 0:
                    self.diagnostics.append({"date": expiry_day.isoformat(), "event_time": "", "direction": "", "symbol": "",
                                             "status": "EXPIRED_CONTRACT_LIST_EMPTY",
                                             "details": "FYERS returned no parseable NSE:NIFTY CE/PE contract symbols for this expiry."})
            except Exception as exc:
                logging.warning("Expired contract-list lookup failed for %s: %s", expiry_day, exc)
                self.diagnostics.append({"date": expiry_day.isoformat(), "event_time": "", "direction": "", "symbol": "",
                                         "status": "EXPIRED_CONTRACT_LOOKUP_FAILED", "details": str(exc)})

        self.catalog_expiries = set(self.contracts_by_expiry.keys())
        logging.info("Contract catalog ready: %s expiries, %s exact symbols.",
                     len(self.catalog_expiries), sum(len(v) for v in self.contracts_by_expiry.values()))
        if not self.catalog_expiries:
            logging.warning("No option contracts could be resolved from FYERS metadata; the backtest will record diagnostics and will not invent symbols.")

    def _add_contract(self, contract: dict) -> None:
        expiry_day = contract["expiry_date"]
        bucket = self.contracts_by_expiry.setdefault(expiry_day, {})
        symbol = contract["symbol"].upper()
        existing = bucket.get(symbol)
        # Prefer explicit current-master metadata when duplicate symbols exist; otherwise
        # keep the exact returned symbol/strike information from expired contract data.
        if existing is None or contract.get("source") == "symbol_master":
            bucket[symbol] = contract

    def resolve_option_contract(self, spot: float, direction: str, trade_day: date) -> Optional[dict]:
        """Resolve the nearest actual expiry and exact ITM option symbol from catalog."""
        expiry_dates = sorted(d for d in self.catalog_expiries if d >= trade_day)
        if not expiry_dates:
            return None
        expiry_day = expiry_dates[0]
        if (expiry_day - trade_day).days > self.cfg.MAX_EXPIRY_DAYS_AFTER_TRADE:
            return None
        option_type = "CE" if direction == "BUY" else "PE"
        contracts = [rec for rec in self.contracts_by_expiry.get(expiry_day, {}).values()
                     if rec.get("option_type") == option_type and rec.get("strike", 0) > 0]
        if not contracts:
            return None

        # Preserve the configured one-step ITM target from the original backtest,
        # but validate it against the real contracts returned by FYERS.
        atm = int(round(float(spot) / self.cfg.STRIKE_STEP) * self.cfg.STRIKE_STEP)
        target_strike = atm - self.cfg.STRIKE_STEP if direction == "BUY" else atm + self.cfg.STRIKE_STEP
        exact = [rec for rec in contracts if abs(float(rec["strike"]) - target_strike) < 0.001]
        if exact:
            return min(exact, key=lambda rec: rec["symbol"])

        # If the exact target strike is absent, choose the closest available strike
        # that remains ITM and is not less ITM than the requested target where possible.
        if direction == "BUY":
            eligible = [rec for rec in contracts if float(rec["strike"]) < float(spot)]
            deeper = [rec for rec in eligible if float(rec["strike"]) <= target_strike]
            pool = deeper or eligible
            if pool:
                return max(pool, key=lambda rec: float(rec["strike"]))
        else:
            eligible = [rec for rec in contracts if float(rec["strike"]) > float(spot)]
            deeper = [rec for rec in eligible if float(rec["strike"]) >= target_strike]
            pool = deeper or eligible
            if pool:
                return min(pool, key=lambda rec: float(rec["strike"]))
        return None

    def option_data(self, contract: dict, trade_day: date) -> pd.DataFrame:
        symbol = contract["symbol"].upper()
        if symbol in self.option_cache:
            cached = self.option_cache[symbol]
            mask = (cached.index.date >= self.backtest_start) & (cached.index.date <= self.backtest_end)
            return cached.loc[mask]

        expiry_day = contract["expiry_date"]
        source = contract.get("source", "expired_api")
        # Query the contract's life-window in a single call, then use that data for
        # all matching dates. This reduces requests and allows a persistent cache.
        history_start = max(self.backtest_start, expiry_day - timedelta(days=45))
        history_end = min(self.backtest_end, expiry_day)
        if history_end < history_start:
            history_end = trade_day
            history_start = trade_day

        cache_path = self._option_cache_path(symbol)
        is_historical_expired_contract = source != "symbol_master"
        if is_historical_expired_contract and os.path.exists(cache_path):
            full_df = self._load_option_cache(symbol)
            if not full_df.empty:
                self.option_cache[symbol] = full_df
                return full_df[(full_df.index.date >= self.backtest_start) & (full_df.index.date <= self.backtest_end)]

        if is_historical_expired_contract:
            if not self._expired_history_request_allowed():
                status = "EXPIRED_HISTORY_DAILY_LIMIT_REACHED"
                details = ("FYERS expired F&O intraday request budget for this local day is exhausted by this script. "
                           "Successful contract histories are cached; run again after the FYERS quota resets.")
                self.diagnostics.append({"date": trade_day.isoformat(), "event_time": "", "direction": "",
                                         "symbol": symbol, "status": status, "details": details})
                empty = pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
                empty.attrs["fetch_status"] = status
                return empty
            self._increment_expired_history_usage()
            resp = self.fyers.expired_history(symbol, history_start, history_end)
            endpoint_name = "expired-F&O historical-data"
        else:
            resp = self.fyers.history(symbol, self.cfg.BASE_RES, history_start, history_end)
            endpoint_name = "regular history"

        if not (isinstance(resp, dict) and resp.get("s") == "ok"):
            message = resp.get("message", str(resp)) if isinstance(resp, dict) else str(resp)
            status_code = resp.get("code", "n/a") if isinstance(resp, dict) else "n/a"
            details = f"FYERS {endpoint_name} failed for exact catalog symbol: {message} (code={status_code})"
            status = "EXPIRED_HISTORY_API_FAILED" if is_historical_expired_contract else "ACTIVE_HISTORY_API_FAILED"
            self.diagnostics.append({"date": trade_day.isoformat(), "event_time": "", "direction": "",
                                     "symbol": symbol, "status": status, "details": details})
            logging.warning("Option history failed for %s via %s: %s", symbol, endpoint_name, details)
            empty = pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
            empty.attrs["fetch_status"] = status
            return empty

        full_df = parse_candles(resp).sort_index()
        full_df = full_df.between_time(self.cfg.SESSION_START, self.cfg.SESSION_END, inclusive="both")
        if full_df.empty:
            status = "OPTION_HISTORY_EMPTY"
            details = f"FYERS returned no intraday option candles for {symbol} via {endpoint_name}."
            self.diagnostics.append({"date": trade_day.isoformat(), "event_time": "", "direction": "",
                                     "symbol": symbol, "status": status, "details": details})
            empty = pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
            empty.attrs["fetch_status"] = status
            return empty

        # Only persist expired contracts, whose candle history is now complete.
        if is_historical_expired_contract:
            self._save_option_cache(symbol, full_df)
        self.option_cache[symbol] = full_df
        return full_df[(full_df.index.date >= self.backtest_start) & (full_df.index.date <= self.backtest_end)]

    def generate_day_signals(self, day_df: pd.DataFrame) -> List[dict]:
        """Scan retests and use either strong C1/C2 or the original C1/C2/C3 pattern.

        Rule requested by the user:
        - After a move away and a retest of ORH/ORL, bullish C1 red + C2 green
          or bearish C1 green + C2 red is a valid two-candle reversal when
          abs(C2 close - C2 open) >= abs(C1 close - C1 open).
        - In that case C3 is completely ignored and entry is at the next bar open
          after C2. No C3 color/body/close condition is evaluated for that setup.
        - If C2 body is smaller than C1 body, use the existing strict C1/C2/C3
          confirmation: C3 continues the reversal color, meets the body/range
          threshold, and closes on the reversal side of the tested level.
        - A retest is consumed at most once, and all entries remain non-lookahead.
        """
        c = self.cfg
        df = day_df.sort_index().between_time(c.SESSION_START, c.SESSION_END, inclusive="both")
        if df.empty:
            return []

        opening_rows = df[df.index.time == c.OPENING_CANDLE_TIME]
        if opening_rows.empty:
            day = df.index[0].date()
            self.diagnostics.append({
                "date": day.isoformat(), "event_time": "", "direction": "", "symbol": "",
                "status": "SKIP_DAY_NO_0915_CANDLE",
                "details": "No candle starting at 09:15 IST; ORH/ORL cannot be calculated."
            })
            logging.warning("Skipping %s: no 09:15 opening candle in NIFTY data.", day)
            return []

        first_ts = opening_rows.index[0]
        first_idx = int(df.index.get_indexer([first_ts])[0])
        orh = float(df.iloc[first_idx]["high"])
        orl = float(df.iloc[first_idx]["low"])
        if not np.isfinite(orh) or not np.isfinite(orl) or orh <= orl:
            self.diagnostics.append({
                "date": df.index[0].date().isoformat(), "event_time": "", "direction": "", "symbol": "",
                "status": "SKIP_INVALID_OPENING_RANGE",
                "details": f"Invalid opening range: ORH={orh}, ORL={orl}."
            })
            return []

        tol = float(c.LEVEL_TOUCH_TOLERANCE_PTS)
        max_c1_delay = int(c.MAX_C1_START_BARS_AFTER_RETEST)
        events: List[dict] = []
        rejection_counts = {
            "non_consecutive_candles": 0,
            "invalid_ohlc": 0,
            "c2_body_smaller_waiting_for_c3": 0,
            "c3_body_below_threshold": 0,
            "wrong_candle_color_sequence": 0,
            "close_progression_or_body_ratio_failed": 0,
            "pattern_not_near_level": 0,
            "pattern_touched_both_levels": 0,
            "no_prior_move_away_then_retest": 0,
            "c2_did_not_close_on_signal_side_of_level": 0,
            "c3_did_not_close_on_signal_side_of_level": 0,
            "retest_already_used": 0,
            "c3_ignored_c2_body_ge_c1": 0,
        }

        highs = df["high"].to_numpy(dtype=float)
        lows = df["low"].to_numpy(dtype=float)
        closes = df["close"].to_numpy(dtype=float)
        opens = df["open"].to_numpy(dtype=float)
        index = df.index
        bodies = np.abs(closes - opens)

        def touches_level(bar_idx: int, level: float) -> bool:
            return bool(lows[bar_idx] <= level + tol and highs[bar_idx] >= level - tol)

        def level_distance(bar_idx: int, level: float) -> float:
            if lows[bar_idx] > level:
                return float(lows[bar_idx] - level)
            if highs[bar_idx] < level:
                return float(level - highs[bar_idx])
            return 0.0

        def find_retest(level: float, c1_idx: int) -> Optional[dict]:
            """Find a fresh retest at/before C1 following at least one bar away from the zone."""
            earliest = max(first_idx + 2, c1_idx - max_c1_delay)
            for r in range(c1_idx, earliest - 1, -1):
                if not touches_level(r, level) or r <= first_idx + 1:
                    continue
                # The immediately preceding bar must have been fully outside the level zone.
                departure_idx = r - 1
                if touches_level(departure_idx, level):
                    continue
                if level == orh and closes[departure_idx] > orh:
                    breakout_idx = departure_idx
                elif level == orl and closes[departure_idx] < orl:
                    breakout_idx = departure_idx
                else:
                    breakout_idx = None
                return {
                    "retest_index": r,
                    "departure_index": departure_idx,
                    "breakout_index": breakout_idx,
                }
            return None

        def determine_level(start_idx: int, end_idx: int, event_ts: pd.Timestamp) -> Optional[tuple]:
            distances = {
                "ORH": min(level_distance(j, orh) for j in range(start_idx, end_idx + 1)),
                "ORL": min(level_distance(j, orl) for j in range(start_idx, end_idx + 1)),
            }
            near_orh = distances["ORH"] <= tol
            near_orl = distances["ORL"] <= tol
            if not near_orh and not near_orl:
                rejection_counts["pattern_not_near_level"] += 1
                return None
            if near_orh and near_orl:
                rejection_counts["pattern_touched_both_levels"] += 1
                self.diagnostics.append({
                    "date": event_ts.date().isoformat(),
                    "event_time": event_ts.strftime("%Y-%m-%d %H:%M:%S"),
                    "direction": "", "symbol": "", "status": "SKIP_PATTERN_NEAR_BOTH_LEVELS",
                    "details": (f"Pattern candles are within {tol:.1f} points of both ORH={orh:.2f} "
                                f"and ORL={orl:.2f}; skipped as ambiguous.")
                })
                return None
            level_name = "ORH" if near_orh else "ORL"
            return level_name, (orh if level_name == "ORH" else orl)

        def add_event(start_idx: int, end_idx: int, signal_bar_idx: int, level_name: str,
                      level: float, retest: dict, direction: str, pattern_kind: str,
                      close_on_signal_side: str, detail: str) -> bool:
            """Append a confirmed setup. signal_bar_idx determines next-bar entry time."""
            retest_idx = int(retest["retest_index"])
            if (level_name, retest_idx) in consumed_retests:
                rejection_counts["retest_already_used"] += 1
                return False

            signal_ts = index[signal_bar_idx] + pd.Timedelta(minutes=1)
            if signal_ts.time() > c.ENTRY_END or signal_ts.time() < c.ENTRY_START:
                return False

            departure_idx = int(retest["departure_index"])
            breakout_idx = retest["breakout_index"]
            breakout_ts = index[int(breakout_idx)] if breakout_idx is not None else None
            retest_ts = index[retest_idx]
            departure_ts = index[departure_idx]
            suffix = "BUY_CE" if direction == "BUY" else "BUY_PE"
            setup_prefix = "BREAKOUT_RETEST" if breakout_ts is not None else "LEVEL_RETEST"
            setup_type = f"{level_name}_{setup_prefix}_{pattern_kind}_{suffix}"
            touched_labels = ",".join(
                f"C{pos + 1}@{index[j].strftime('%H:%M')}"
                for pos, j in enumerate(range(start_idx, end_idx + 1))
                if touches_level(j, level)
            )
            c1_idx = start_idx
            c2_idx = start_idx + 1
            c3_idx = end_idx if end_idx == start_idx + 2 else None
            event = {
                "date": index[signal_bar_idx].date(),
                "timestamp": signal_ts,
                "signal_bar_start": index[signal_bar_idx],
                "direction": direction,
                "pattern": setup_type,
                "setup_type": setup_type,
                "opening_range_high": round(orh, 2),
                "opening_range_low": round(orl, 2),
                "signal_level": round(level, 2),
                "level_touch_time": retest_ts,
                "departure_time": departure_ts,
                "breakout_time": breakout_ts,
                "retest_time": retest_ts,
                "touch_candles": f"{level_name}: {touched_labels or 'near-level window'}",
                "signal_spot_close": float(closes[signal_bar_idx]),
                "signal_index": signal_bar_idx,
                "c1_time": index[c1_idx],
                "c2_time": index[c2_idx],
                "c3_time": index[c3_idx] if c3_idx is not None else None,
                "confirmation_type": pattern_kind,
            }
            events.append(event)
            consumed_retests.add((level_name, retest_idx))
            self.diagnostics.append({
                "date": index[signal_bar_idx].date().isoformat(),
                "event_time": signal_ts.strftime("%Y-%m-%d %H:%M:%S"),
                "direction": direction,
                "symbol": "",
                "status": "VALID_LEVEL_RETEST_C12_STRONGER_SECOND" if c3_idx is None else "VALID_LEVEL_RETEST_C123_REVERSAL",
                "details": (
                    f"{setup_type}; confirmation={pattern_kind}; prior_departure={departure_ts.strftime('%H:%M')}; "
                    f"strict_breakout_close_before_retest={'YES' if breakout_ts is not None else 'NO'}; "
                    f"retest={retest_ts.strftime('%H:%M')}; ORH={orh:.2f}, ORL={orl:.2f}; "
                    f"tolerance=+/-{tol:.1f}; C1 O/C={opens[c1_idx]:.2f}/{closes[c1_idx]:.2f}; "
                    f"C1 body={bodies[c1_idx]:.2f}; C2 O/C={opens[c2_idx]:.2f}/{closes[c2_idx]:.2f}; "
                    f"C2 body={bodies[c2_idx]:.2f}; {detail}; {close_on_signal_side}"
                )
            })
            return True

        consumed_retests = set()

        # Start at the earliest bar that can be C2 after the opening candle and a departure bar.
        for end_idx in range(first_idx + 3, len(df)):
            # ---------------------------------------------------------------
            # Rule A: two-candle confirmation; C3 must not affect this signal.
            # ---------------------------------------------------------------
            c1_idx = end_idx - 1
            c2_idx = end_idx
            if index[c2_idx] - index[c1_idx] != pd.Timedelta(minutes=1):
                rejection_counts["non_consecutive_candles"] += 1
            else:
                pair_opens = opens[[c1_idx, c2_idx]]
                pair_closes = closes[[c1_idx, c2_idx]]
                pair_bodies = bodies[[c1_idx, c2_idx]]
                if not np.all(np.isfinite(np.r_[pair_opens, pair_closes, pair_bodies])):
                    rejection_counts["invalid_ohlc"] += 1
                else:
                    pair_bullish = bool(pair_closes[0] < pair_opens[0] and pair_closes[1] > pair_opens[1])
                    pair_bearish = bool(pair_closes[0] > pair_opens[0] and pair_closes[1] < pair_opens[1])
                    if (pair_bullish or pair_bearish) and pair_bodies[1] < pair_bodies[0]:
                        rejection_counts["c2_body_smaller_waiting_for_c3"] += 1
                    if (pair_bullish or pair_bearish) and pair_bodies[1] >= pair_bodies[0]:
                        signal_ts = index[c2_idx] + pd.Timedelta(minutes=1)
                        if signal_ts.time() <= c.ENTRY_END and signal_ts.time() >= c.ENTRY_START:
                            level_info = determine_level(c1_idx, c2_idx, signal_ts)
                            if level_info is not None:
                                level_name, level = level_info
                                retest = find_retest(level, c1_idx)
                                if retest is None:
                                    rejection_counts["no_prior_move_away_then_retest"] += 1
                                elif (level_name, int(retest["retest_index"])) in consumed_retests:
                                    rejection_counts["retest_already_used"] += 1
                                else:
                                    bullish = pair_bullish
                                    direction = "BUY" if bullish else "SELL"
                                    if (bullish and pair_closes[1] <= level) or ((not bullish) and pair_closes[1] >= level):
                                        rejection_counts["c2_did_not_close_on_signal_side_of_level"] += 1
                                    else:
                                        add_event(
                                            c1_idx, c2_idx, c2_idx, level_name, level, retest, direction,
                                            "BULLISH_C12_STRONGER_SECOND" if bullish else "BEARISH_C12_STRONGER_SECOND",
                                            "C2 closed above tested level" if bullish else "C2 closed below tested level",
                                            f"C2 body >= C1 body ({pair_bodies[1]:.2f} >= {pair_bodies[0]:.2f}); C3 ignored"
                                        )

            # ---------------------------------------------------------------
            # Rule B: if C2 was not large enough, fall back to strict C1/C2/C3.
            # A valid reversal-color pair with C2 >= C1 always ignores C3.
            # ---------------------------------------------------------------
            if end_idx < first_idx + 4:
                continue
            c1_idx = end_idx - 2
            c2_idx = end_idx - 1
            c3_idx = end_idx
            rows = df.iloc[c1_idx:c3_idx + 1]
            if (index[c2_idx] - index[c1_idx] != pd.Timedelta(minutes=1) or
                    index[c3_idx] - index[c2_idx] != pd.Timedelta(minutes=1)):
                rejection_counts["non_consecutive_candles"] += 1
                continue

            o = opens[c1_idx:c3_idx + 1]
            h = highs[c1_idx:c3_idx + 1]
            l = lows[c1_idx:c3_idx + 1]
            close = closes[c1_idx:c3_idx + 1]
            body = bodies[c1_idx:c3_idx + 1]
            if not np.all(np.isfinite(np.r_[o, h, l, close, body])):
                rejection_counts["invalid_ohlc"] += 1
                continue

            red = close < o
            green = close > o
            pair_bullish = bool(red[0] and green[1])
            pair_bearish = bool(green[0] and red[1])
            if (pair_bullish or pair_bearish) and body[1] >= body[0]:
                rejection_counts["c3_ignored_c2_body_ge_c1"] += 1
                continue

            bullish_colors = bool(red[0] and green[1] and green[2])
            bearish_colors = bool(green[0] and red[1] and red[2])
            if not (bullish_colors or bearish_colors):
                rejection_counts["wrong_candle_color_sequence"] += 1
                continue

            c3_range = float(h[2] - l[2])
            c3_body = float(body[2])
            if c3_range <= 0 or c3_body < float(c.HEALTHY_BODY_FRAC) * c3_range:
                rejection_counts["c3_body_below_threshold"] += 1
                continue

            close_progression_bull = bool(close[0] < close[1] < close[2])
            close_progression_bear = bool(close[0] > close[1] > close[2])
            body_ratio_ok = bool(body[0] < float(c.BODY_RATIO_MAX) * body[1])
            if bullish_colors and not (close_progression_bull or body_ratio_ok):
                rejection_counts["close_progression_or_body_ratio_failed"] += 1
                continue
            if bearish_colors and not (close_progression_bear or body_ratio_ok):
                rejection_counts["close_progression_or_body_ratio_failed"] += 1
                continue

            signal_ts = index[c3_idx] + pd.Timedelta(minutes=1)
            if signal_ts.time() > c.ENTRY_END or signal_ts.time() < c.ENTRY_START:
                continue
            level_info = determine_level(c1_idx, c3_idx, signal_ts)
            if level_info is None:
                continue
            level_name, level = level_info
            retest = find_retest(level, c1_idx)
            if retest is None:
                rejection_counts["no_prior_move_away_then_retest"] += 1
                continue
            if (level_name, int(retest["retest_index"])) in consumed_retests:
                rejection_counts["retest_already_used"] += 1
                continue

            if bullish_colors:
                direction = "BUY"
                if close[2] <= level:
                    rejection_counts["c3_did_not_close_on_signal_side_of_level"] += 1
                    continue
                detail = (f"C3 body/range={c3_body / c3_range:.2f}; "
                          f"close_progression={close_progression_bull}; body_ratio={body[0] / body[1] if body[1] > 0 else float('inf'):.2f}")
                pattern_kind = "BULLISH_C123"
            else:
                direction = "SELL"
                if close[2] >= level:
                    rejection_counts["c3_did_not_close_on_signal_side_of_level"] += 1
                    continue
                detail = (f"C3 body/range={c3_body / c3_range:.2f}; "
                          f"close_progression={close_progression_bear}; body_ratio={body[0] / body[1] if body[1] > 0 else float('inf'):.2f}")
                pattern_kind = "BEARISH_C123"

            add_event(
                c1_idx, c3_idx, c3_idx, level_name, level, retest, direction,
                pattern_kind, "C3 closed on signal side of tested level", detail
            )

        events.sort(key=lambda e: e["timestamp"])
        self.diagnostics.append({
            "date": index[0].date().isoformat(),
            "event_time": "",
            "direction": "",
            "symbol": "",
            "status": "SIGNAL_SCAN_SUMMARY",
            "details": "Valid candidates={}; {}".format(
                len(events), "; ".join(f"{key}={value}" for key, value in rejection_counts.items())
            ),
        })
        return events

    @staticmethod
    def get_bar_at(df: pd.DataFrame, ts: pd.Timestamp) -> Optional[pd.Series]:
        if df.empty or ts not in df.index:
            return None
        return df.loc[ts]

    def entry_option_price(self, opt_df: pd.DataFrame, entry_ts: pd.Timestamp) -> Optional[float]:
        """Use the option candle open at the exact entry minute; do not silently shift forward."""
        row = self.get_bar_at(opt_df, entry_ts)
        if row is None:
            return None
        price = float(row["open"])
        if not np.isfinite(price) or price <= 0:
            return None
        return price * (1.0 + self.cfg.SLIPPAGE_PCT / 100.0)

    def simulate_trade(self, index_df: pd.DataFrame, opt_df: pd.DataFrame, event: dict,
                       option_symbol: str, entry_premium: float, spot_entry: float) -> dict:
        c = self.cfg
        trade_day = event["date"]
        direction = event["direction"]
        if direction == "BUY":
            stop_level = spot_entry - c.STOP_LOSS_PTS
            target_level = spot_entry + c.TAKE_PROFIT_PTS
        else:
            stop_level = spot_entry + c.STOP_LOSS_PTS
            target_level = spot_entry - c.TAKE_PROFIT_PTS

        eod_ts = IST.localize(datetime.combine(trade_day, c.EOD_EXIT_TIME))
        future = index_df[(index_df.index.date == trade_day) & (index_df.index >= event["timestamp"]) & (index_df.index.time < c.EOD_EXIT_TIME)]
        exit_ts = None
        exit_reason = None
        exit_option_price = None
        exit_spot = None

        for bar_ts, bar in future.iterrows():
            hi = float(bar["high"])
            lo = float(bar["low"])
            if direction == "BUY":
                hit_sl = lo <= stop_level
                hit_tp = hi >= target_level
            else:
                hit_sl = hi >= stop_level
                hit_tp = lo <= target_level

            # Conservative priority if both index SL and TP are touched in the same candle.
            if hit_sl or hit_tp:
                exit_reason = "stop_loss" if hit_sl else "target"
                exit_ts = bar_ts + pd.Timedelta(minutes=1)
                exit_spot = stop_level if hit_sl else target_level
                opt_bar = self.get_bar_at(opt_df, bar_ts)
                if opt_bar is not None:
                    exit_option_price = float(opt_bar["close"])
                break

        if exit_ts is None:
            day_bars = index_df[(index_df.index.date == trade_day) & (index_df.index.time < c.EOD_EXIT_TIME)]
            if day_bars.empty:
                exit_reason = "eod_exit_no_index_bar"
                exit_ts = event["timestamp"]
                exit_spot = float(spot_entry)
            else:
                last_bar_ts = day_bars.index[-1]
                exit_ts = last_bar_ts + pd.Timedelta(minutes=1)
                exit_spot = float(day_bars.iloc[-1]["close"])
                exit_reason = "eod_exit"
                opt_bar = self.get_bar_at(opt_df, last_bar_ts)
                if opt_bar is not None:
                    exit_option_price = float(opt_bar["close"])

        if exit_option_price is None or not np.isfinite(exit_option_price) or exit_option_price <= 0:
            # Without an aligned option exit candle, do not invent a premium/PnL.
            raise ValueError(f"No aligned option candle for exit timestamp around {exit_ts} ({option_symbol})")

        pnl_points = exit_option_price - entry_premium
        pnl_rupees = pnl_points * c.LOT
        return {
            "date": trade_day.isoformat(),
            "direction": direction,
            "option_symbol": option_symbol,
            "pattern": event["pattern"],
            "opening_range_high": event["opening_range_high"],
            "opening_range_low": event["opening_range_low"],
            "signal_level": event["signal_level"],
            "breakout_time": event["breakout_time"].strftime("%Y-%m-%d %H:%M:%S") if event.get("breakout_time") is not None else "",
            "departure_time": event["departure_time"].strftime("%Y-%m-%d %H:%M:%S"),
            "retest_time": event["retest_time"].strftime("%Y-%m-%d %H:%M:%S"),
            "level_touch_time": event["level_touch_time"].strftime("%Y-%m-%d %H:%M:%S"),
            "c1_time": event["c1_time"].strftime("%Y-%m-%d %H:%M:%S"),
            "c2_time": event["c2_time"].strftime("%Y-%m-%d %H:%M:%S"),
            "c3_time": event["c3_time"].strftime("%Y-%m-%d %H:%M:%S") if event.get("c3_time") is not None else "",
            "signal_time": event["timestamp"].strftime("%Y-%m-%d %H:%M:%S"),
            "entry_time": event["timestamp"].strftime("%Y-%m-%d %H:%M:%S"),
            "spot_entry": round(float(spot_entry), 2),
            "spot_stop_level": round(float(stop_level), 2),
            "spot_target_level": round(float(target_level), 2),
            "entry_premium": round(float(entry_premium), 2),
            "exit_time": exit_ts.strftime("%Y-%m-%d %H:%M:%S"),
            "exit_spot": round(float(exit_spot), 2),
            "exit_premium": round(float(exit_option_price), 2),
            "exit_reason": exit_reason,
            "pnl_premium_points": round(float(pnl_points), 2),
            "pnl_rs": round(float(pnl_rupees), 2),
            "result": "profit" if pnl_rupees > 0 else "loss",
            "quantity": c.LOT,
        }

    def run(self) -> pd.DataFrame:
        index_df = self.get_index_data()
        if index_df.empty:
            raise ValueError("No NIFTY index candles were returned for this backtest window.")
        self.backtest_start = index_df.index[0].date()
        self.backtest_end = index_df.index[-1].date()
        self.prepare_contract_catalog(self.backtest_start, self.backtest_end)
        trade_rows: List[dict] = []
        executed_counts: Dict[date, int] = {}
        last_exit_by_day: Dict[date, pd.Timestamp] = {}

        for day, day_df in index_df.groupby(index_df.index.date):
            candidates = self.generate_day_signals(day_df)
            if not candidates:
                self.diagnostics.append({"date": day.isoformat(), "event_time": "", "direction": "", "symbol": "",
                                         "status": "NO_VALID_SETUP", "details": "No strong C1/C2 or strict C1/C2/C3 reversal qualified after a prior move away and a fresh ORH/ORL retest."})
                continue

            for event in candidates:
                if executed_counts.get(day, 0) >= int(self.cfg.MAX_TRADES_PER_DAY):
                    self.diagnostics.append({
                        "date": day.isoformat(), "event_time": str(event["timestamp"]),
                        "direction": event["direction"], "symbol": "",
                        "status": "SKIP_DAILY_TRADE_LIMIT",
                        "details": f"Already executed {executed_counts.get(day, 0)} trades; configured daily maximum is {self.cfg.MAX_TRADES_PER_DAY}."
                    })
                    break
                if event["timestamp"].time() < self.cfg.ENTRY_START or event["timestamp"].time() > self.cfg.ENTRY_END:
                    continue
                previous_exit = last_exit_by_day.get(day)
                if previous_exit is not None and event["timestamp"] < previous_exit:
                    self.diagnostics.append({
                        "date": day.isoformat(), "event_time": str(event["timestamp"]),
                        "direction": event["direction"], "symbol": "",
                        "status": "SKIP_TRADE_OVERLAP",
                        "details": f"Previous trade does not exit until {previous_exit}; skipping overlapping entry."
                    })
                    continue

                # Entry is the open of the minute after the confirmation bar (C2 for a strong-body
                # two-candle setup, or C3 for the three-candle fallback).
                entry_ts = event["timestamp"]
                entry_row = self.get_bar_at(index_df, entry_ts)
                if entry_row is None:
                    self.diagnostics.append({"date": day.isoformat(), "event_time": str(entry_ts), "direction": event["direction"],
                                             "symbol": "", "status": "SKIP_NO_SPOT_ENTRY_BAR", "details": "No NIFTY bar at signal-close timestamp."})
                    continue
                spot_entry = float(entry_row["open"])
                contract = self.resolve_option_contract(spot_entry, event["direction"], day)
                if contract is None:
                    self.diagnostics.append({"date": day.isoformat(), "event_time": str(entry_ts), "direction": event["direction"],
                                             "symbol": "", "status": "NO_VALID_CONTRACT_SYMBOL",
                                             "details": "FYERS metadata did not provide a matching ITM option at the nearest available expiry; no symbol was constructed."})
                    logging.warning("Skipping %s %s: no exact FYERS-listed ITM contract was found.", day, event["direction"])
                    continue
                option_symbol = contract["symbol"]
                opt_df = self.option_data(contract, day)
                if opt_df.empty:
                    fetch_status = opt_df.attrs.get("fetch_status", "SKIP_NO_OPTION_DATA")
                    self.diagnostics.append({"date": day.isoformat(), "event_time": str(entry_ts), "direction": event["direction"],
                                             "symbol": option_symbol, "status": "SKIP_NO_OPTION_DATA",
                                             "details": f"No usable option candles for this exact contract. Fetch status: {fetch_status}."})
                    continue

                entry_premium = self.entry_option_price(opt_df, entry_ts)
                if entry_premium is None:
                    self.diagnostics.append({"date": day.isoformat(), "event_time": str(entry_ts), "direction": event["direction"],
                                             "symbol": option_symbol, "status": "SKIP_NO_ALIGNED_OPTION_ENTRY_BAR", "details": "No option candle at exact entry timestamp."})
                    continue

                try:
                    trade = self.simulate_trade(index_df, opt_df, event, option_symbol, entry_premium, spot_entry)
                except ValueError as exc:
                    self.diagnostics.append({"date": day.isoformat(), "event_time": str(entry_ts), "direction": event["direction"],
                                             "symbol": option_symbol, "status": "SKIP_NO_ALIGNED_OPTION_EXIT_BAR", "details": str(exc)})
                    continue

                trade_rows.append(trade)
                executed_counts[day] = executed_counts.get(day, 0) + 1
                exit_stamp = pd.Timestamp(trade["exit_time"])
                if exit_stamp.tzinfo is None:
                    exit_stamp = IST.localize(exit_stamp.to_pydatetime())
                last_exit_by_day[day] = exit_stamp
                self.diagnostics.append({"date": day.isoformat(), "event_time": str(entry_ts), "direction": event["direction"],
                                         "symbol": option_symbol, "status": "TRADE_EXECUTED", "details": trade["exit_reason"]})
                logging.info("TRADE %s %s | entry spot %.2f | SL %.2f | TP %.2f | option=%s | PnL Rs %.2f | trade %s/%s",
                             day, event["direction"], spot_entry, trade["spot_stop_level"], trade["spot_target_level"],
                             option_symbol, trade["pnl_rs"], executed_counts[day], self.cfg.MAX_TRADES_PER_DAY)

        # Keep a stable trade schema even when the strategy produces no executed trades.
        trade_columns = [
            "date", "direction", "option_symbol", "pattern",
            "opening_range_high", "opening_range_low", "signal_level", "level_touch_time",
            "c1_time", "c2_time", "c3_time", "signal_time", "entry_time", "spot_entry",
            "spot_stop_level", "spot_target_level",
            "entry_premium", "exit_time", "exit_spot", "exit_premium", "exit_reason",
            "pnl_premium_points", "pnl_rs", "result", "quantity", "cum_pnl_rs",
        ]
        trades_df = pd.DataFrame(trade_rows)
        if not trades_df.empty:
            trades_df["cum_pnl_rs"] = pd.to_numeric(trades_df["pnl_rs"], errors="coerce").fillna(0).cumsum()
            for col in trade_columns:
                if col not in trades_df.columns:
                    trades_df[col] = ""
            trades_df = trades_df[trade_columns]
        else:
            trades_df = pd.DataFrame(columns=trade_columns)

        diagnostics_df = pd.DataFrame(self.diagnostics)
        diagnostics_df.to_csv(self.cfg.DIAGNOSTICS_CSV, index=False, encoding="utf-8")

        summary_rows = self._build_summary_rows(trades_df, diagnostics_df, index_df)
        summary_df = pd.DataFrame(summary_rows, columns=["metric", "value"])
        summary_df.to_csv(self.cfg.SUMMARY_CSV, index=False, encoding="utf-8")

        # Put the summary inside the PnL CSV too, using a record_type column so
        # every row has the same columns and the CSV remains parseable.
        pnl_columns = ["record_type"] + trade_columns + ["metric", "value"]
        combined_rows = []
        for trade in trades_df.to_dict(orient="records"):
            row = {col: "" for col in pnl_columns}
            row["record_type"] = "TRADE"
            for col in trade_columns:
                value = trade.get(col, "")
                if pd.isna(value):
                    value = ""
                row[col] = value
            combined_rows.append(row)
        for item in summary_rows:
            row = {col: "" for col in pnl_columns}
            row["record_type"] = "SUMMARY"
            row["metric"] = item[0]
            row["value"] = item[1]
            combined_rows.append(row)
        pd.DataFrame(combined_rows, columns=pnl_columns).to_csv(
            self.cfg.TRADES_CSV, index=False, encoding="utf-8"
        )

        summary_lookup = {str(metric): value for metric, value in summary_rows}
        print("\n" + "=" * 72)
        print("FIRST CANDLE HIGH-LOW SUPPORT AND RESISTANCE BACKTEST")
        print(f"Opening candle       : {self.cfg.OPENING_CANDLE_TIME}")
        print("Entry setup          : prior move away -> ORH/ORL retest -> C1/C2 stronger-body confirmation, else C1/C2/C3 fallback")
        print(f"Spot SL / TP         : {self.cfg.STOP_LOSS_PTS:.0f} / {self.cfg.TAKE_PROFIT_PTS:.0f} index points")
        print(f"Option / quantity    : {self.cfg.MONEYNESS} / {self.cfg.LOT}")
        print(f"Trades               : {summary_lookup['Total trades']}")
        print(f"Wins / losses / BE   : {summary_lookup['Winning trades']} / {summary_lookup['Losing trades']} / {summary_lookup['Breakeven trades']}")
        print(f"Win rate             : {summary_lookup['Win rate (%)']}")
        print(f"Gross profit         : Rs {float(summary_lookup['Gross profit (Rs)']):,.2f}")
        print(f"Gross loss           : Rs {float(summary_lookup['Gross loss amount (Rs)']):,.2f}")
        print(f"Net option PnL       : Rs {float(summary_lookup['Net PnL (Rs)']):,.2f}")
        print(f"Average PnL / trade  : Rs {float(summary_lookup['Average PnL per trade (Rs)']):,.2f}")
        print(f"Profit factor        : {summary_lookup['Profit factor']}")
        print(f"Max drawdown         : Rs {float(summary_lookup['Maximum drawdown (Rs)']):,.2f}")
        print(f"Trades CSV + summary : {self.cfg.TRADES_CSV}")
        print(f"Summary-only CSV     : {self.cfg.SUMMARY_CSV}")
        print(f"Diagnostics CSV      : {self.cfg.DIAGNOSTICS_CSV}")
        print(f"Expired option cache : {self.cfg.OPTION_CACHE_DIR}")
        usage = self._load_daily_expired_usage()
        print(f"Expired API calls today by this script: {usage.get('requests', 0)} / {self.cfg.MAX_EXPIRED_HISTORY_REQUESTS_PER_DAY}")
        if not diagnostics_df.empty and "status" in diagnostics_df:
            print("\nDiagnostic counts:")
            print(diagnostics_df["status"].value_counts().to_string())
        if trades_df.empty:
            print("\nNo trades were executed. Summary and diagnostics were still saved.")
        print("=" * 72)
        return trades_df

    def _build_summary_rows(self, trades_df: pd.DataFrame, diagnostics_df: pd.DataFrame,
                            index_df: pd.DataFrame) -> List[tuple]:
        """Build comprehensive run metrics for both the summary and PnL CSVs."""
        c = self.cfg
        summary: List[tuple] = []

        def add(metric: str, value):
            if isinstance(value, (np.integer,)):
                value = int(value)
            elif isinstance(value, (np.floating, float)):
                value = round(float(value), 4) if np.isfinite(value) else str(value)
            summary.append((metric, value))

        pnl = pd.to_numeric(trades_df.get("pnl_rs", pd.Series(dtype=float)), errors="coerce").dropna()
        total = int(len(pnl))
        wins = int((pnl > 0).sum())
        losses = int((pnl < 0).sum())
        breakeven = int((pnl == 0).sum())
        gross_profit = float(pnl[pnl > 0].sum()) if total else 0.0
        gross_loss_amount = float(-pnl[pnl < 0].sum()) if total else 0.0
        net_pnl = float(pnl.sum()) if total else 0.0
        win_rate = (wins / total * 100.0) if total else 0.0
        loss_rate = (losses / total * 100.0) if total else 0.0
        avg_pnl = net_pnl / total if total else 0.0
        avg_win = float(pnl[pnl > 0].mean()) if wins else 0.0
        avg_loss_amount = float(-pnl[pnl < 0].mean()) if losses else 0.0
        profit_factor = (gross_profit / gross_loss_amount) if gross_loss_amount > 0 else (
            "N/A (no losing trades)" if wins else "N/A (no trades or no gross profit)"
        )
        best_trade = float(pnl.max()) if total else 0.0
        worst_trade = float(pnl.min()) if total else 0.0

        if total:
            equity = pd.concat([pd.Series([0.0]), pnl.reset_index(drop=True)], ignore_index=True).cumsum()
            running_peak = equity.cummax()
            max_drawdown = float((equity - running_peak).min() * -1.0)
        else:
            max_drawdown = 0.0

        def max_streak(values, condition):
            longest = streak = 0
            for value in values:
                if condition(float(value)):
                    streak += 1
                    longest = max(longest, streak)
                else:
                    streak = 0
            return longest

        max_win_streak = max_streak(pnl.tolist(), lambda x: x > 0)
        max_loss_streak = max_streak(pnl.tolist(), lambda x: x < 0)

        avg_holding_minutes = 0.0
        if total and {"entry_time", "exit_time"}.issubset(trades_df.columns):
            entry_times = pd.to_datetime(trades_df["entry_time"], errors="coerce")
            exit_times = pd.to_datetime(trades_df["exit_time"], errors="coerce")
            holding = (exit_times - entry_times).dt.total_seconds() / 60.0
            holding = holding.dropna()
            if not holding.empty:
                avg_holding_minutes = float(holding.mean())

        # Strategy configuration and backtest coverage.
        add("Strategy", "First Candle High-Low Support and Resistance")
        add("Backtest start date", str(getattr(self, "backtest_start", "")))
        add("Backtest end date", str(getattr(self, "backtest_end", "")))
        add("Index candles loaded", len(index_df))
        add("Trading dates with index data", int(pd.Series(index_df.index.date).nunique()) if not index_df.empty else 0)
        add("Opening candle time", str(c.OPENING_CANDLE_TIME))
        add("Session start", str(c.SESSION_START))
        add("Session end", str(c.SESSION_END))
        add("Entry window start", str(c.ENTRY_START))
        add("Entry window end", str(c.ENTRY_END))
        add("Entry rule", "Prior move away -> fresh ORH/ORL retest; if C2 body >= C1 body, confirm after C2 and ignore C3; otherwise use strict C1/C2/C3 fallback; entry at next 1-minute bar open")
        add("Two-candle confirmation", "Bullish C1 red/C2 green or bearish C1 green/C2 red; if real body C2 >= real body C1, C3 is ignored and C2 must close on trade side of tested level")
        add("Three-candle fallback", "Used only when a directional C1/C2 pair has C2 body < C1 body; C3 must continue direction, pass body/range and close-progression/body-ratio filters, and close on trade side of tested level")
        add("Scenario families", "Bullish reversal near ORH or ORL -> BUY ITM CE; bearish reversal near ORH or ORL -> BUY ITM PE")
        add("Retest rule", f"Candle range returns within +/- {c.LEVEL_TOUCH_TOLERANCE_PTS} points after a prior bar moved fully away; prior close beyond ORH/ORL is optional, and retest can be C1")
        add("Maximum C1 delay after retest (bars; 0 means C1 is retest)", c.MAX_C1_START_BARS_AFTER_RETEST)
        add("Ambiguous dual-level rule", "Skip if the same 3-candle window intersects both ORH and ORL zones")
        add("Candle confirmation", "Strict BUY: red/green/green; strict SELL: green/red/red; no engulfing-only or two-candle substitute")
        add("C3 minimum body/range", c.HEALTHY_BODY_FRAC)
        add("C1/C2 body quality", f"Directional close progression OR C1 body < {c.BODY_RATIO_MAX:.2f} x C2 body")
        add("Direction mapping", "BUY=ITM CE; SELL=ITM PE")
        add("EMA filter", "Off")
        add("Swing-zone filter", "Off; opening range is used instead")
        add("Option moneyness", c.MONEYNESS)
        add("Option quantity", c.LOT)
        add("Strike step", c.STRIKE_STEP)
        add("Stop-loss (NIFTY spot points)", c.STOP_LOSS_PTS)
        add("Take-profit (NIFTY spot points)", c.TAKE_PROFIT_PTS)
        add("Risk-to-reward ratio", c.TAKE_PROFIT_PTS / c.STOP_LOSS_PTS if c.STOP_LOSS_PTS else "N/A")
        add("Premium entry slippage (%)", c.SLIPPAGE_PCT)
        add("Maximum trades per day", c.MAX_TRADES_PER_DAY)
        add("Trade overlap rule", "No new position before the previous trade's modeled exit time")
        add("EOD exit time", str(c.EOD_EXIT_TIME))

        # Trade performance.
        add("Total trades", total)
        add("Winning trades", wins)
        add("Losing trades", losses)
        add("Breakeven trades", breakeven)
        add("Win rate (%)", win_rate)
        add("Loss rate (%)", loss_rate)
        add("Gross profit (Rs)", gross_profit)
        add("Gross loss amount (Rs)", gross_loss_amount)
        add("Net PnL (Rs)", net_pnl)
        add("Average PnL per trade (Rs)", avg_pnl)
        add("Average winning trade (Rs)", avg_win)
        add("Average losing trade amount (Rs)", avg_loss_amount)
        add("Profit factor", profit_factor)
        add("Best trade (Rs)", best_trade)
        add("Worst trade (Rs)", worst_trade)
        add("Maximum drawdown (Rs)", max_drawdown)
        add("Maximum consecutive wins", max_win_streak)
        add("Maximum consecutive losses", max_loss_streak)
        add("Average holding time (minutes)", avg_holding_minutes)

        if total:
            for reason, count in trades_df["exit_reason"].fillna("unknown").value_counts().sort_index().items():
                add(f"Exit reason count - {reason}", int(count))
            if "direction" in trades_df.columns:
                for direction, group in trades_df.groupby("direction", dropna=False):
                    group_pnl = pd.to_numeric(group["pnl_rs"], errors="coerce").fillna(0)
                    add(f"{direction} trades", int(len(group)))
                    add(f"{direction} wins", int((group_pnl > 0).sum()))
                    add(f"{direction} losses", int((group_pnl < 0).sum()))
                    add(f"{direction} net PnL (Rs)", float(group_pnl.sum()))
        else:
            add("Exit reason count - stop_loss", 0)
            add("Exit reason count - target", 0)
            add("Exit reason count - eod_exit", 0)
            for direction in ("BUY", "SELL"):
                add(f"{direction} trades", 0)
                add(f"{direction} wins", 0)
                add(f"{direction} losses", 0)
                add(f"{direction} net PnL (Rs)", 0.0)

        # Diagnostic funnel counts explain why a day did not result in a trade.
        if not diagnostics_df.empty and "status" in diagnostics_df.columns:
            for status, count in diagnostics_df["status"].fillna("unknown").value_counts().sort_index().items():
                add(f"Diagnostic count - {status}", int(count))
        else:
            add("Diagnostic rows", 0)

        usage = self._load_daily_expired_usage()
        add("Expired history API requests used today", usage.get("requests", 0))
        add("Expired history API request budget per day", c.MAX_EXPIRED_HISTORY_REQUESTS_PER_DAY)
        return summary


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - FirstCandleSRBacktest - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    bot = FirstCandleSRBacktest()
    bot.run()


if __name__ == "__main__":
    main()
