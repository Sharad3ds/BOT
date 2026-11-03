"""Backtest: First Candle High-Low Support and Resistance - strict C1/C2/C3 reversal

The first NIFTY 1-minute candle starting at 09:15 IST defines fixed ORH/ORL.
The script tracks opening-range tests and later reactions over time, so a level event
need not be in the same three-candle window as the final confirmation.

CANDLE REVERSAL RULE IS INTENTIONALLY STRICT AND CONSISTENT FOR ALL SCENARIOS:
- Bullish reversal: C1 RED, C2 GREEN, C3 GREEN.
- Bearish reversal: C1 GREEN, C2 RED, C3 RED.
- C3 real body must be at least 80% of its high-low range.
- Additionally, closes must progress in the trade direction OR C1 body must be less
  than 1.2 times C2 body (same rule as the reference WithMetrics strategy).
- Engulfing-only and two-candle follow-through patterns are NOT standalone signals.

Scenario families still cover both outcomes at ORH and ORL, including reversals after
price has extended beyond an opening-range boundary. Only the strict three-candle
pattern above can confirm a candidate. If both levels were interacted with, the code
uses its level-context precedence logic; at most one trade is executed per day.

Entry is at the next 1-minute bar open after C3 closes; no same-bar look-ahead.
No EMA or prior swing-zone filter is used. Spot-based SL/TP are 40/100 NIFTY points;
EOD exit is 15:20 IST and entries are allowed through 15:00 IST.

Option contracts are resolved from FYERS NSE_FO symbol master and expired F&O contract
metadata. Expired option history uses the FYERS expired-F&O endpoint with local caching
and a configurable daily request budget. This is a backtest only; it does not place live
orders. Validate actual performance on the same dates/contracts before treating the
strategy as profitable.
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

    # Restore the reference strategy's strict candle quality/reversal rules.
    HEALTHY_BODY_FRAC = 0.80
    BODY_RATIO_MAX = 1.20
    CONFIRM_MUST_BE_COLORED = True

    # Multi-bar opening-range context. Points are NIFTY spot points.
    LEVEL_TOUCH_TOLERANCE_PTS = 5.0
    LEVEL_CONTEXT_BARS = 8
    BREAKOUT_CONTEXT_BARS = 10
    EXTENSION_CONTEXT_BARS = 8
    EXTENSION_MIN_RANGE_FRAC = 0.10

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

    TRADES_CSV = "First_Candle_High_Low_SR_PPTXScenarios_Backtest.csv"
    SUMMARY_CSV = "First_Candle_High_Low_SR_PPTXScenarios_Summary.csv"
    DIAGNOSTICS_CSV = "First_Candle_High_Low_SR_PPTXScenarios_Diagnostics.csv"


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
            "Config | PPTX scenario families for opening range | opening candle=%s | SL=%s spot points | TP=%s spot points | "
            "ITM=%s | qty=%s | max one trade/day | exact FYERS symbol catalog | expired-F&O API enabled",
            self.cfg.OPENING_CANDLE_TIME, self.cfg.STOP_LOSS_PTS, self.cfg.TAKE_PROFIT_PTS,
            self.cfg.MONEYNESS, self.cfg.LOT,
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
        """Detect all level-reaction families shown in the supplied chart-only PPTX.

        Important distinction from the earlier implementation:
        * A boundary test is tracked as state across bars, not only inside C1/C2/C3.
        * The 09:15 candle initializes both boundary contexts for a short immediate-reaction
          window, which allows an immediate rejection after the opening candle.
        * Breakout/breakdown failures can trigger after price has moved away from the level.
        * An extension reversal can trigger while price is still above ORH / below ORL.
        * Candidate signals are returned chronologically; `run()` still executes at most one/day.
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
                "details": "No candle starting at 09:15 IST; fixed ORH/ORL cannot be calculated."
            })
            logging.warning("Skipping %s: no 09:15 opening candle in NIFTY data.", day)
            return []

        first_ts = opening_rows.index[0]
        first_idx = int(df.index.get_indexer([first_ts])[0])
        orh = float(df.iloc[first_idx]["high"])
        orl = float(df.iloc[first_idx]["low"])
        or_range = orh - orl
        if not np.isfinite(orh) or not np.isfinite(orl) or or_range <= 0:
            self.diagnostics.append({
                "date": df.index[0].date().isoformat(), "event_time": "", "direction": "", "symbol": "",
                "status": "SKIP_INVALID_OPENING_RANGE",
                "details": f"Invalid opening range: ORH={orh}, ORL={orl}."
            })
            return []

        tol = float(c.LEVEL_TOUCH_TOLERANCE_PTS)
        level_context = int(c.LEVEL_CONTEXT_BARS)
        break_context = int(c.BREAKOUT_CONTEXT_BARS)
        extension_bars = int(c.EXTENSION_CONTEXT_BARS)
        min_extension = max(5.0, or_range * float(c.EXTENSION_MIN_RANGE_FRAC))

        # The opening candle defines and touches both levels. This initial touch context
        # expires quickly unless a later bar re-tests a level.
        last_orh_touch_idx = first_idx
        last_orl_touch_idx = first_idx
        last_close_above_orh_idx: Optional[int] = None
        last_close_below_orl_idx: Optional[int] = None
        events: List[dict] = []
        last_emitted: Dict[str, int] = {}

        for i in range(first_idx + 1, len(df)):
            candle_time = df.index[i]
            signal_close_ts = candle_time + pd.Timedelta(minutes=1)
            if signal_close_ts.time() > c.ENTRY_END:
                break
            if signal_close_ts.time() < c.ENTRY_START:
                continue

            # Maintain boundary-test and breakout state from the first post-opening
            # minute onward, even before a complete 3-candle confirmation exists.
            current_high = float(df.iloc[i]["high"])
            current_low = float(df.iloc[i]["low"])
            current_close = float(df.iloc[i]["close"])
            if current_low <= orh + tol and current_high >= orh - tol:
                last_orh_touch_idx = i
            if current_low <= orl + tol and current_high >= orl - tol:
                last_orl_touch_idx = i
            if i < first_idx + 3:
                if current_close > orh:
                    last_close_above_orh_idx = i
                if current_close < orl:
                    last_close_below_orl_idx = i
                continue

            rows = df.iloc[i - 2:i + 1]
            if (rows.index[1] - rows.index[0] != pd.Timedelta(minutes=1) or
                    rows.index[2] - rows.index[1] != pd.Timedelta(minutes=1)):
                # Do not let state jump across a missing intraday candle.
                continue

            o = rows["open"].to_numpy(dtype=float)
            h = rows["high"].to_numpy(dtype=float)
            l = rows["low"].to_numpy(dtype=float)
            close = rows["close"].to_numpy(dtype=float)
            if not np.all(np.isfinite(np.r_[o, h, l, close])):
                continue

            # Strict C1/C2/C3 reversal confirmation is shared by every scenario.
            # Do not let an engulfing candle or two candles of continuation replace
            # the required 3-candle reversal sequence.
            c3_open = float(o[2])
            c3_close = float(close[2])
            c3_range = float(h[2] - l[2])
            c3_body = abs(c3_close - c3_open)
            body = np.abs(close - o)
            green = close > o
            red = close < o

            c3_quality_ok = bool(
                c3_range > 0 and c3_body >= float(c.HEALTHY_BODY_FRAC) * c3_range
            )
            body_ratio_ok = bool(body[0] < float(c.BODY_RATIO_MAX) * body[1])
            bullish_color_sequence = bool(
                red[0] and green[1] and (green[2] or not c.CONFIRM_MUST_BE_COLORED)
            )
            bearish_color_sequence = bool(
                green[0] and red[1] and (red[2] or not c.CONFIRM_MUST_BE_COLORED)
            )
            bullish_close_progression = bool(close[0] < close[1] < close[2])
            bearish_close_progression = bool(close[0] > close[1] > close[2])

            bullish_confirm = bool(
                c3_quality_ok and bullish_color_sequence and
                (bullish_close_progression or body_ratio_ok)
            )
            bearish_confirm = bool(
                c3_quality_ok and bearish_color_sequence and
                (bearish_close_progression or body_ratio_ok)
            )

            orh_age = i - last_orh_touch_idx
            orl_age = i - last_orl_touch_idx
            orh_recent = orh_age <= level_context
            orl_recent = orl_age <= level_context
            prior_above_recent = (last_close_above_orh_idx is not None and
                                  0 < i - last_close_above_orh_idx <= break_context)
            prior_below_recent = (last_close_below_orl_idx is not None and
                                  0 < i - last_close_below_orl_idx <= break_context)

            # Detect price extensions from the opening range and later swing reversals.
            extension_start = max(first_idx + 1, i - extension_bars)
            prior_window = df.iloc[extension_start:i]
            prior_swing_high = float(prior_window["high"].max()) if not prior_window.empty else float("nan")
            prior_swing_low = float(prior_window["low"].min()) if not prior_window.empty else float("nan")
            upper_extension_recent = bool(np.isfinite(prior_swing_high) and
                                          prior_swing_high >= orh + min_extension)
            lower_extension_recent = bool(np.isfinite(prior_swing_low) and
                                          prior_swing_low <= orl - min_extension)

            direction: Optional[str] = None
            setup_type: Optional[str] = None
            level: Optional[float] = None
            context_idx: Optional[int] = None

            # Long outcomes from ORL: a recovery after a flush, a reclaim after a failed
            # breakdown, a support bounce, or bullish breakout continuation above ORH.
            if bullish_confirm:
                if c3_close < orl and lower_extension_recent:
                    direction, setup_type, level = "BUY", "ORL_EXTENDED_SWING_LOW_REVERSAL_BUY", orl
                    context_idx = last_orl_touch_idx
                elif c3_close > orh and (orh_recent or prior_above_recent or close[1] <= orh):
                    direction, setup_type, level = "BUY", "ORH_BREAKOUT_CONTINUATION_BUY", orh
                    context_idx = last_orh_touch_idx
                elif c3_close > orl and prior_below_recent:
                    direction, setup_type, level = "BUY", "ORL_FAILED_BREAKDOWN_RECLAIM_BUY", orl
                    context_idx = last_orl_touch_idx
                elif c3_close > orl and orl_recent:
                    # If both levels are recent and C3 is inside the opening range, use
                    # the level tested most recently; when tied, use the closer boundary.
                    choose_orl = (not orh_recent or last_orl_touch_idx > last_orh_touch_idx or
                                  (last_orl_touch_idx == last_orh_touch_idx and
                                   abs(c3_close - orl) <= abs(c3_close - orh)))
                    if choose_orl:
                        direction, setup_type, level = "BUY", "ORL_SUPPORT_BOUNCE_BUY", orl
                        context_idx = last_orl_touch_idx

            # Short outcomes from ORH: a rejection, failed breakout, extension reversal,
            # or bearish continuation after a low-side breakdown.
            if direction is None and bearish_confirm:
                if c3_close > orh and upper_extension_recent:
                    direction, setup_type, level = "SELL", "ORH_EXTENDED_SWING_HIGH_REVERSAL_SELL", orh
                    context_idx = last_orh_touch_idx
                elif c3_close < orl:
                    direction, setup_type, level = "SELL", "ORL_BREAKDOWN_CONTINUATION_SELL", orl
                    context_idx = last_orl_touch_idx
                elif c3_close < orh and prior_above_recent:
                    direction, setup_type, level = "SELL", "ORH_FAILED_BREAKOUT_REJECTION_SELL", orh
                    context_idx = last_orh_touch_idx
                elif c3_close < orh and orh_recent:
                    choose_orh = (not orl_recent or last_orh_touch_idx > last_orl_touch_idx or
                                  (last_orh_touch_idx == last_orl_touch_idx and
                                   abs(c3_close - orh) <= abs(c3_close - orl)))
                    if choose_orh:
                        direction, setup_type, level = "SELL", "ORH_RESISTANCE_REJECTION_SELL", orh
                        context_idx = last_orh_touch_idx

            # Remember breakouts only after evaluating the current candle so that the
            # current close cannot masquerade as a previous failed breakout/breakdown.
            if c3_close > orh:
                last_close_above_orh_idx = i
            if c3_close < orl:
                last_close_below_orl_idx = i

            if direction is None or setup_type is None or level is None:
                continue

            # Deduplicate repeated rolling windows for the same scenario within 2 bars;
            # different scenarios can still be recorded later on the same day.
            if setup_type in last_emitted and i - last_emitted[setup_type] <= 2:
                continue
            last_emitted[setup_type] = i

            context_label = "level-context unavailable"
            if context_idx is not None and 0 <= context_idx < len(df):
                level_name = "ORH" if abs(level - orh) < abs(level - orl) else "ORL"
                context_label = f"{level_name}@{df.index[context_idx].strftime('%H:%M')}"
            event = {
                "date": candle_time.date(),
                "timestamp": signal_close_ts,
                "signal_bar_start": candle_time,
                "direction": direction,
                "pattern": setup_type,
                "setup_type": setup_type,
                "opening_range_high": round(orh, 2),
                "opening_range_low": round(orl, 2),
                "signal_level": round(float(level), 2),
                "level_touch_time": df.index[context_idx] if context_idx is not None else rows.index[0],
                "touch_candles": context_label,
                "signal_spot_close": c3_close,
                "signal_index": i,
                "c1_time": rows.index[0],
                "c2_time": rows.index[1],
                "c3_time": rows.index[2],
            }
            events.append(event)
            self.diagnostics.append({
                "date": candle_time.date().isoformat(),
                "event_time": signal_close_ts.strftime("%Y-%m-%d %H:%M:%S"),
                "direction": direction, "symbol": "", "status": "VALID_PPTX_SCENARIO_CONFIRMATION",
                "details": (
                    f"{setup_type}; ORH={orh:.2f}, ORL={orl:.2f}; context={context_label}; "
                    f"C1={rows.index[0].strftime('%H:%M')} O/C={o[0]:.2f}/{close[0]:.2f}; "
                    f"C2={rows.index[1].strftime('%H:%M')} O/C={o[1]:.2f}/{close[1]:.2f}; "
                    f"C3={rows.index[2].strftime('%H:%M')} O/C={o[2]:.2f}/{close[2]:.2f}; "
                    f"C3 body/range={c3_body / c3_range:.2f}."
                )
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
            "level_touch_time": event["level_touch_time"].strftime("%Y-%m-%d %H:%M:%S"),
            "c1_time": event["c1_time"].strftime("%Y-%m-%d %H:%M:%S"),
            "c2_time": event["c2_time"].strftime("%Y-%m-%d %H:%M:%S"),
            "c3_time": event["c3_time"].strftime("%Y-%m-%d %H:%M:%S"),
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
        executed_days = set()

        for day, day_df in index_df.groupby(index_df.index.date):
            candidates = self.generate_day_signals(day_df)
            if not candidates:
                self.diagnostics.append({"date": day.isoformat(), "event_time": "", "direction": "", "symbol": "",
                                         "status": "NO_VALID_SETUP", "details": "No level-context, breakout/failure, or extension-reversal setup passed candle confirmation and quality checks."})
                continue

            for event in candidates:
                if day in executed_days:
                    break
                if event["timestamp"].time() < self.cfg.ENTRY_START or event["timestamp"].time() > self.cfg.ENTRY_END:
                    continue

                # Spot entry is the open of the minute after C3 confirmation if available.
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
                executed_days.add(day)
                self.diagnostics.append({"date": day.isoformat(), "event_time": str(entry_ts), "direction": event["direction"],
                                         "symbol": option_symbol, "status": "TRADE_EXECUTED", "details": trade["exit_reason"]})
                logging.info("TRADE %s %s | entry spot %.2f | SL %.2f | TP %.2f | option=%s | PnL Rs %.2f",
                             day, event["direction"], spot_entry, trade["spot_stop_level"], trade["spot_target_level"],
                             option_symbol, trade["pnl_rs"])
                break  # one trade per day

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
        print("Entry setup          : stateful ORH/ORL scenarios + strict C1/C2/C3 reversal confirmation")
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
        add("Entry rule", "First executable candidate from multi-bar boundary context, failed breakouts/breakdowns, or extension reversals; entry at next 1-minute bar open")
        add("Scenario families", "ORH breakout BUY; ORH rejection/failed breakout SELL; extended swing-high reversal SELL; ORL support bounce/reclaim BUY; ORL breakdown SELL; extended swing-low reversal BUY")
        add("Reversal rule source", "Restored reference C1/C2/C3 colors, C3 body >= 80%, close progression OR C1 body < 1.2 x C2 body")
        add("Level-touch rule", f"OHLC test with {c.LEVEL_TOUCH_TOLERANCE_PTS} NIFTY-point tolerance; opening candle seeds short immediate-reaction context; later tests persist for {c.LEVEL_CONTEXT_BARS} bars")
        add("Breakout/failure memory", f"Closing excursions tracked for {c.BREAKOUT_CONTEXT_BARS} bars")
        add("Extension reversal rule", f"Swing high/low in last {c.EXTENSION_CONTEXT_BARS} bars; extension at least max(5 points, {c.EXTENSION_MIN_RANGE_FRAC:.0%} of opening range)")
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
        add("Maximum trades per day", 1)
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
