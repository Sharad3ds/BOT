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

# ── Fix CWD to src/ so relative paths (token files, state files) resolve correctly ──
os.chdir(BASE_DIR)

# ── Per-bot log file: logs/live_YYYYMMDD.log ──
os.environ.setdefault("BOT_NAME", "live")

from logger_config import get_logger
from VAMLive import VAMLiveBot, VAMLiveEngine

logger = get_logger()
IST = pytz.timezone("Asia/Kolkata")

STOP_TIME_HOUR   = 15
STOP_TIME_MINUTE = 25


def should_stop():
    now = datetime.now(IST)
    return (
        now.hour > STOP_TIME_HOUR
        or (now.hour == STOP_TIME_HOUR and now.minute >= STOP_TIME_MINUTE)
    )


def stop_watcher(engine: VAMLiveEngine):
    """
    Watch clock and stop engine at 15:25 IST.
    VAMLiveEngine already handles EOD exit at 15:20 internally;
    this watcher ensures the process exits cleanly shortly after.
    """
    logger.info("[runBotLive] Stop watcher started — will stop at 15:25 IST")
    while True:
        time.sleep(30)
        if should_stop():
            logger.info("[runBotLive] 15:25 IST reached — stopping WebSocket engine")
            try:
                engine.stop()
            except Exception as e:
                logger.error(f"[runBotLive] Error during engine.stop(): {e}")
            time.sleep(2)
            os._exit(0)


def main():
    logger.info("=== VAM Live (WebSocket) Bot Starting ===")

    # Step 1: Init VAMLiveBot (loads app_id + access_token from files already written by runBot.py)
    bot = VAMLiveBot()
    if not bot.app_id or not bot.access_token:
        logger.error("[runBotLive] app_id or access_token missing — check fyers_appid.txt / fyers_token.txt")
        sys.exit(1)

    ws_access_token = f"{bot.app_id}:{bot.access_token}"
    engine = VAMLiveEngine(bot, access_token=ws_access_token)

    # Step 3: Start stop watcher in background
    watcher = threading.Thread(target=stop_watcher, args=(engine,), daemon=True)
    watcher.start()
    logger.info("[runBotLive] Stop watcher running in background ✅")

    # Step 4: Start WebSocket engine (blocking — driven by tick callbacks)
    logger.info("[runBotLive] Starting VAMLiveEngine...")
    try:
        engine.start()
    except KeyboardInterrupt:
        logger.info("[runBotLive] KeyboardInterrupt received — stopping")
        try:
            engine.stop()
        except Exception:
            pass

    logger.info("=== VAM Live (WebSocket) Bot Stopped ===")


if __name__ == "__main__":
    main()
