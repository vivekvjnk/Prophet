import logging
from logging.handlers import RotatingFileHandler

def configure_rotating_file_handler(log_file_path, max_bytes=5*1024*1024, backup_count=5):
    """
    Configures a RotatingFileHandler for local filesystem logging.

    Args:
        log_file_path (str): Full path to the log file.
        max_bytes (int): Max file size before rotation (default: 5MB).
        backup_count (int): Number of rotated logs to keep.

    Returns:
        RotatingFileHandler: Configured log handler.
    """
    rotating_handler = RotatingFileHandler(
        log_file_path, maxBytes=max_bytes, backupCount=backup_count
    )
    rotating_handler.setLevel(logging.DEBUG)

    # Define log format
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    rotating_handler.setFormatter(formatter)

    return rotating_handler


def get_pipeline_logger(pipeline_name, file_handler):
    """
    Returns a logger for the given pipeline using a provided file handler.
    - pipeline_name: Name of the pipeline (used as logger name).
    - file_handler: Pre-configured file handler.
    """
    logger = logging.getLogger(pipeline_name)
    logger.setLevel(logging.DEBUG)

    if not logger.hasHandlers():  # Prevent duplicate handlers
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)

        # Formatter for console logs
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        console_handler.setFormatter(formatter)

        # Attach handlers
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

    return logger