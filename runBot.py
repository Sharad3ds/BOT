import sys
import os
import time
import threading
from datetime import datetime
import pytz

# ── Force UTF-8 throughout — fixes UnicodeEncodeError on Windows cp1252 consoles ──
os.environ.setdefault("PYTHONIOENCODING", "utf-8")
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

from logger_config import get_logger

# ── CHANGED: import VAMPollingEngine instead of VAMLiveBot + VAMLiveEngine ──
from vam_poll_5S import VAMPollingEngine

from generate_TOTP_Token import GenerateTOTPToken

logger = get_logger()
IST = pytz.timezone("Asia/Kolkata")

# ── CHANGED: stop time moved to 15:25 to allow EOD cleanup after 15:20 exit ──
STOP_TIME_HOUR   = 15
STOP_TIME_MINUTE = 25

def should_stop():
    now = datetime.now(IST)
    return (
        now.hour > STOP_TIME_HOUR
        or (now.hour == STOP_TIME_HOUR and now.minute >= STOP_TIME_MINUTE)
    )

def stop_watcher(engine: VAMPollingEngine):
    """
    Watch clock and stop engine at 15:25 IST.
    The polling engine already handles EOD exit at 15:20 internally,
    so this watcher just ensures the process exits cleanly shortly after.
    """
    logger.info("Stop watcher started — will stop at 15:25 IST")
    while True:
        time.sleep(30)
        if should_stop():
            logger.info("15:25 IST reached — running EOD cleanup and stopping")
            try:
                # ── CHANGED: polling engine uses _eod_cleanup() not stop() ──
                engine.Shutdown()
            except Exception as e:
                logger.error(f"Error during engine._eod_cleanup(): {e}")
            sys.exit(0)

def main():
    logger.info("=== VAM Polling Bot Starting ===")

    # Step 1: Generate TOTP Token
    logger.info("Generating TOTP Token...")
    flag = GenerateTOTPToken()
    if not flag:
        logger.error("TOTP generation failed — exiting")
        sys.exit(1)
    logger.info("TOTP generated successfully ✅")

    # Step 2: Init polling engine (handles Fyers auth internally)
    # ── CHANGED: no separate VAMLiveBot needed — engine is self-contained ──
    engine = VAMPollingEngine()

    # Step 3: Start stop watcher in background
    watcher = threading.Thread(
        target=stop_watcher,
        args=(engine,),
        daemon=True
    )
    watcher.start()
    logger.info("Stop watcher running in background ✅")

    # Step 4: Start trading (blocking — polls every 30s internally)
    logger.info("Starting VAMPollingEngine...")
    engine.start()
    logger.info("=== VAM Polling Bot Stopped ===")

if __name__ == "__main__":
    main()
