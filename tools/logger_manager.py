import os
import logging
from datetime import datetime

"""
Singleton class for logging
"""
class Logger:
    _logger = None

    @classmethod
    def get_logger(cls, level=logging.INFO):
        if cls._logger is None:
            # Ensure the logs directory exists
            log_dir = 'logs'
            if not os.path.exists(log_dir):
                os.makedirs(log_dir)

            # Format the current date and time
            current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            log_file = os.path.join(log_dir, f'{current_time}.log')

            # Configure logging
            cls._logger = logging.getLogger(__name__)
            cls._logger.setLevel(level)

            # Create file handler with time-dependent log file name
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(level)

            # Create console handler
            console_handler = logging.StreamHandler()
            console_handler.setLevel(level)

            # Create formatter and add it to handlers
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s (%(filename)s:%(lineno)d)')
            file_handler.setFormatter(formatter)
            console_handler.setFormatter(formatter)

            # Add handlers to logger
            cls._logger.addHandler(file_handler)
            cls._logger.addHandler(console_handler)

            # Clean up old log files to keep only the 5 most recent
            cls.cleanup_logs(log_dir)

        return cls._logger

    @classmethod
    def cleanup_logs(cls, log_dir, keep=5):
        # Get all log files in the directory
        files = [os.path.join(log_dir, f) for f in os.listdir(log_dir) if f.endswith('.log')]
        # Sort files by modified time
        files.sort(key=os.path.getmtime)

        # Remove the oldest files, keeping only the `keep` most recent
        for f in files[:-keep]:
            os.remove(f)

    # Logging methods
    def info(self, message):
        Logger.get_logger().info(message)

    def warning(self, message):
        Logger.get_logger().warning(message)

    def error(self, message, exc_info=False):
        Logger.get_logger().error(message, exc_info=exc_info)
