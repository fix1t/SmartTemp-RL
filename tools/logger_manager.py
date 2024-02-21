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
            # Check if the logs folder exists, create it if it does not
            if not os.path.exists('logs'):
                os.makedirs('logs')

            # Format the current date and time
            current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            log_file = f'logs/{current_time}.log'  # time-dependent log file name

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

        return cls._logger

    def info(self, message):
        Logger.get_logger().info(message)

    def warning(self, message):
        Logger.get_logger().warning(message)

    def error(self, message, exc_info=False):
        Logger.get_logger().error(message, exc_info=exc_info)

