"""
Resolves the ACTUAL weekly/monthly expiry code for a NIFTY option, by
matching against Fyers' public NSE F&O symbol master instead of guessing
"next Tuesday" (which breaks whenever NSE shifts the expiry weekday, or a
Tuesday falls on a trading holiday and gets moved to the prior day).

HOW IT WORKS (schema-agnostic, so it survives Fyers changing the CSV's
column layout): the master file contains the exact tradable symbol string
for every listed instrument, e.g. "NSE:NIFTY2481124500PE". Rather than
trying to locate and parse a specific "expiry date" column (whose name/
position isn't guaranteed stable), this scans the RAW file text for every
substring matching the NIFTY-option symbol pattern and extracts the
embedded expiry code directly from it. That code is exactly what
expiry_code() would need to produce -- so resolving a date's expiry becomes:
"scan forward day by day from the signal date, and take the first calendar
day whose computed expiry_code() appears in the set of codes Fyers actually
lists." No expiry-weekday assumption required at all.

CRITICAL LIMITATION: Fyers only lists CURRENTLY TRADABLE instruments in this
master file -- once a weekly contract expires, it typically disappears from
it. That means this resolver is reliable for the CURRENT and near-future
expiries (which is what matters for live trading), but for historical dates
whose expiry has already passed and rolled out of the live master, there is
nothing to match against. For those, resolve_expiry_code() falls back to the
old "next Tuesday" guess and logs a warning -- callers should treat those
particular historical trades with reduced confidence, the same way they
always could have been wrong before this fix.
"""
import os
import re
import time
import logging
import requests
import urllib3
from datetime import datetime, timedelta, time as dtime
import pytz

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
IST = pytz.timezone("Asia/Kolkata")

try:
    logger = logging.getLogger("ExpiryResolver")
    if not logger.handlers:
        logging.basicConfig(level=logging.INFO,
                             format="%(asctime)s - ExpiryResolver - %(levelname)s - %(message)s",
                             datefmt="%Y-%m-%d %H:%M:%S")
except Exception:
    pass

MASTER_URL = "https://public.fyers.in/sym_details/NSE_FO.csv"
CACHE_FILE = "nse_fo_symbol_master_cache.csv"
CACHE_MAX_AGE_HOURS = 12  # symbol master doesn't change intra-day; re-fetch at most twice/day

# Matches full NIFTY (not BANKNIFTY/FINNIFTY -- those don't contain the
# literal "NSE:NIFTY" substring, so no exclusion regex is needed) option
# symbols anywhere in the raw CSV text, e.g. "NSE:NIFTY2481124500PE",
# "NSE:NIFTY24JUL24500CE".
NIFTY_OPT_PATTERN = re.compile(r"NSE:NIFTY(\d{2}(?:[A-Z]{3}|[1-9OND]\d{2}))(\d{5})(CE|PE)\b")


def _fetch_master_raw() -> str:
    r = requests.get(MASTER_URL, timeout=20, verify=False)
    r.raise_for_status()
    return r.text


def _load_master_text() -> str:
    """Returns raw master file text, using a local cache to avoid
    re-downloading (~multi-MB file) on every run. Cache refreshes if older
    than CACHE_MAX_AGE_HOURS or missing."""
    use_cache = False
    if os.path.exists(CACHE_FILE):
        age_hrs = (time.time() - os.path.getmtime(CACHE_FILE)) / 3600.0
        if age_hrs < CACHE_MAX_AGE_HOURS:
            use_cache = True

    if use_cache:
        try:
            with open(CACHE_FILE, "r", encoding="utf-8", errors="ignore") as f:
                return f.read()
        except Exception:
            pass  # fall through to re-fetch

    text = _fetch_master_raw()
    try:
        with open(CACHE_FILE, "w", encoding="utf-8") as f:
            f.write(text)
    except Exception as e:
        logger.warning(f"Could not write symbol master cache: {e}")
    return text


class ExpiryResolver:
    """
    Usage:
        resolver = ExpiryResolver()
        exp_code = resolver.resolve_expiry_code(signal_date)  # e.g. "26811" or "26AUG"
    """

    def __init__(self):
        self._valid_codes = None  # set[str], populated lazily on first use

    def _ensure_loaded(self):
        if self._valid_codes is not None:
            return
        try:
            text = _load_master_text()
            codes = set()
            for m in NIFTY_OPT_PATTERN.finditer(text):
                codes.add(m.group(1))
            self._valid_codes = codes
            logger.info(f"Loaded {len(codes)} distinct NIFTY expiry codes from Fyers symbol master")
        except Exception as e:
            logger.warning(f"Could not load Fyers symbol master ({e}) -- "
                            f"resolve_expiry_code() will fall back to the Tuesday guess for all dates.")
            self._valid_codes = set()

    def resolve_expiry_code(self, signal_date, expiry_code_fn, get_next_tuesday_fn,
                             search_days: int = 14) -> tuple:
        """
        Returns (expiry_code: str, source: "master" | "fallback_tuesday").

        Scans forward day-by-day from signal_date (inclusive) up to
        search_days, computing expiry_code_fn(candidate_date) for each and
        checking it against the set of codes Fyers' live symbol master
        actually lists. The first match is the real expiry -- this makes no
        assumption about which weekday NSE's expiry falls on.

        If nothing matches within the window (most likely because the
        expiry has already passed and aged out of Fyers' live master, or the
        master couldn't be loaded at all), falls back to the old
        get_next_tuesday_fn() guess and flags it so callers can track how
        many trades used the unreliable fallback.
        """
        self._ensure_loaded()

        if self._valid_codes:
            base = datetime.combine(signal_date, dtime(0, 0))
            for i in range(search_days):
                candidate = base + timedelta(days=i)
                code = expiry_code_fn(candidate)
                if code in self._valid_codes:
                    return code, "master"

        # fallback: old behavior
        exp_dt = get_next_tuesday_fn(datetime.combine(signal_date, dtime(9, 20)))
        return expiry_code_fn(exp_dt), "fallback_tuesday"