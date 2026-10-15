"""Shared logging utilities."""

import logging

from src.common.config import get_settings

LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"


def get_logger(name: str) -> logging.Logger:
    """Create or return a configured logger."""

    settings = get_settings()
    level_name = settings.log_level.upper()
    level = getattr(logging, level_name, logging.INFO)

    logging.basicConfig(level=level, format=LOG_FORMAT)
    return logging.getLogger(name)
