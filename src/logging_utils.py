"""Project-wide logging: one format, console + rotating log file.

Usage in any module:
    from src.logging_utils import get_logger
    logger = get_logger(__name__)
    logger.info("Loaded %d rows", n)
"""
from __future__ import annotations

import logging
import sys
from logging.handlers import RotatingFileHandler

from src.config import settings

LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
LOG_FILE_NAME = "complaintiq.log"

_configured = False


def setup_logging(level: str | None = None) -> None:
    """Configure the root logger once. Safe to call many times."""
    global _configured
    if _configured:
        return

    formatter = logging.Formatter(LOG_FORMAT, DATE_FORMAT)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)

    settings.paths.logs.mkdir(parents=True, exist_ok=True)
    file_handler = RotatingFileHandler(
        settings.paths.logs / LOG_FILE_NAME,
        maxBytes=5_000_000,  # rotate after ~5 MB
        backupCount=3,       # keep 3 old files
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)

    root = logging.getLogger()
    root.setLevel(level or settings.log_level)
    root.addHandler(console_handler)
    root.addHandler(file_handler)

    # Third-party libraries can be very chatty; keep them quiet
    for noisy in ("urllib3", "matplotlib", "PIL", "filelock", "httpx"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    _configured = True


def get_logger(name: str) -> logging.Logger:
    """Return a named logger, configuring logging on first use."""
    setup_logging()
    return logging.getLogger(name)


if __name__ == "__main__":
    log = get_logger("logging_check")
    log.debug("This DEBUG line only shows if CIQ_LOG_LEVEL=DEBUG")
    log.info("Logging works. Log file: %s", settings.paths.logs / LOG_FILE_NAME)
    log.warning("This is what a warning looks like")