"""
Central logging configuration for the Smart Research Assistant application.
"""

import os
import logging
from logging.handlers import RotatingFileHandler
from src.constants import DEFAULT_LOG_PATH, LOG_FORMAT

def setup_logger(
    name: str = "smart_research_assistant",
    log_level: str = "INFO",
    log_file: str = DEFAULT_LOG_PATH
) -> logging.Logger:
    """
    Sets up a logger with a console handler and a rotating file handler.
    
    Args:
        name: Name of the logger.
        log_level: String representation of the logging level.
        log_file: Path to the log file.
        
    Returns:
        logging.Logger: Configured logger instance.
    """
    # Convert string level to logging levels
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        numeric_level = logging.INFO

    logger = logging.getLogger(name)
    logger.setLevel(numeric_level)

    # Avoid duplicate handlers if logger is already configured
    if logger.handlers:
        return logger

    # Ensure log directory exists
    log_dir = os.path.dirname(log_file)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)

    formatter = logging.Formatter(LOG_FORMAT)

    # Console Handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(numeric_level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # Rotating File Handler (10 MB files, keeping up to 5 backups)
    try:
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=10 * 1024 * 1024,
            backupCount=5,
            encoding="utf-8"
        )
        file_handler.setLevel(numeric_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    except Exception as e:
        print(f"Failed to initialize file logger: {e}. Writing to console only.")

    logger.info(f"Logger '{name}' initialized at level {log_level}.")
    return logger
