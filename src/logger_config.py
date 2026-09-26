import logging
import logging.handlers
import os
from datetime import datetime


class Logger:
    """Singleton logger for the application """
    _instance = None
    _logger = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Logger, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        if self._logger is None:
            self._setup_logger()

    def _setup_logger(self):
        """Setup the logger configuration"""
        self._logger = logging.getLogger('SMBot_V2')
        
        # Set log level based on environment
        log_level = os.getenv('LOG_LEVEL', 'INFO').upper()
        self._logger.setLevel(getattr(logging, log_level))
        
        # Prevent duplicate handlers
        if self._logger.handlers:
            return

        # Create logs directory if it doesn't exist
        os.makedirs('logs', exist_ok=True)
        
        # File handler with rotation
        # BOT_NAME env var lets each bot write its own log file (poll / live / hybrid)
        bot_name = os.getenv('BOT_NAME', 'smbot')
        log_file = f'logs/{bot_name}_{datetime.now().strftime("%Y%m%d")}.log'
        file_handler = logging.handlers.RotatingFileHandler(
            log_file, maxBytes=10*1024*1024, backupCount=5
        )
        
        # Console handler
        console_handler = logging.StreamHandler()
        
        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        # Add handlers
        self._logger.addHandler(file_handler)
        self._logger.addHandler(console_handler)

    def get_logger(self):
        """Get the logger instance"""
        return self._logger

    @classmethod
    def get_instance(cls):
        """Get the singleton logger instance"""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance.get_logger()


def get_logger():
    """Get the application logger"""
    return Logger.get_instance()
