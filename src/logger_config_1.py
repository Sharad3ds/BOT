import logging
import pytz
from datetime import datetime

IST = pytz.timezone("Asia/Kolkata")

def ist_time(*args):
    return datetime.now(IST).timetuple()

def get_logger(name: str = "VAMPoll") -> logging.Logger:
    logging.Formatter.converter = ist_time
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        ))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger
