import sys
import os
import time
import threading
from datetime import datetime
import pytz

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

from logger_config import get_logger
from VAMLive import VAMLiveBot, VAMLiveEngine
from generate_TOTP_Token import GenerateTOTPToken

logger = get_logger()
IST = pytz.timezone("Asia/Kolkata")

STOP_TIME_HOUR = 15    # 3 PM
STOP_TIME_MINUTE = 10  # 3:10 PM

def should_stop():
    """Check if current IST time is past 3:10 PM"""
    now = datetime.now(IST)
    return (now.hour > STOP_TIME_HOUR or 
            (now.hour == STOP_TIME_HOUR and now.minute >= STOP_TIME_MINUTE))

def stop_watcher(engine):
    """Watch clock and stop engine at 3:10 PM IST"""
    logger.info("Stop watcher started - will stop at 3:10 PM IST")
    while True:
        if should_stop():
            logger.info("3:10 PM IST reached - stopping VAMLiveEngine")
            try:
                if engine.ws is not None:
                    engine.ws.close()
            except Exception as e:
                logger.error(f"Error closing websocket: {e}")
            sys.exit(0)
        time.sleep(30)  # check every 30 seconds

def main():
    logger.info("=== SMBot Starting via Render Cron Job ===")

    # Step 1: Generate TOTP Token
    logger.info("Generating TOTP Token...")
    flag = GenerateTOTPToken()
    if not flag:
        logger.error("TOTP generation failed — exiting")
        sys.exit(1)
    logger.info("TOTP generated successfully ✅")

    # Step 2: Initialize bot
    bot = VAMLiveBot()
    app_id = bot.app_id
    raw_token = bot.access_token

    if not app_id or not raw_token:
        logger.error("Missing app_id or token — exiting")
        sys.exit(1)

    ws_access_token = f"{app_id}:{raw_token}"

    # Step 3: Start engine
    engine = VAMLiveEngine(bot, access_token=ws_access_token)

    # Step 4: Start stop watcher in background
    watcher = threading.Thread(
        target=stop_watcher, 
        args=(engine,), 
        daemon=True
    )
    watcher.start()
    logger.info("Stop watcher running in background ✅")

    # Step 5: Start trading (blocking)
    logger.info("Starting VAMLiveEngine...")
    engine.start()
    logger.info("=== SMBot Stopped ===")

if __name__ == "__main__":
    main()