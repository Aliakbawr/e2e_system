"""Application logging configuration."""

import logging
from logging.handlers import RotatingFileHandler

from config.settings import LOG_BACKUP_COUNT, LOG_FILE, LOG_LEVEL, LOG_MAX_BYTES


def setup_logging() -> None:
    """Configure idempotent console and rotating-file application logs."""
    root_logger = logging.getLogger()
    if getattr(root_logger, "_persian_assistant_configured", False):
        return

    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    level = getattr(logging, LOG_LEVEL.upper(), logging.INFO)
    formatter = logging.Formatter(
        fmt="%(asctime)s %(levelname)s %(name)s %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S%z",
    )

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    file_handler = RotatingFileHandler(
        LOG_FILE,
        maxBytes=LOG_MAX_BYTES,
        backupCount=LOG_BACKUP_COUNT,
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)

    root_logger.setLevel(level)
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)
    root_logger._persian_assistant_configured = True

    logging.getLogger(__name__).info(
        "logging_started level=%s file=%s",
        logging.getLevelName(level),
        LOG_FILE,
    )
