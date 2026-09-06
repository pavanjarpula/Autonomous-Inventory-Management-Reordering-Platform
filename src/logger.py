"""
Structured logging for SellerSense. Provides JSON-formatted logs
with context metadata for LangSmith integration and debugging.

Usage:
    from logger import get_logger
    logger = get_logger(__name__, extra_data={"item_id": "I001"})
    logger.info("Assessing inventory item")
"""

import json
import logging
from datetime import datetime, timezone
from typing import Optional


class StructuredFormatter(logging.Formatter):
    """JSON-formatted log output for observability."""

    def format(self, record: logging.LogRecord) -> str:
        log_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
            "message": record.getMessage(),
        }

        if hasattr(record, "extra_data") and record.extra_data:
            log_entry["metadata"] = record.extra_data

        return json.dumps(log_entry)


class ContextFilter(logging.Filter):
    """Injects extra_data into log records."""

    def __init__(self, extra_data: Optional[dict] = None):
        super().__init__()
        self.extra_data = extra_data or {}

    def filter(self, record: logging.LogRecord) -> bool:
        record.extra_data = self.extra_data
        return True


def get_logger(
    name: str,
    level: int = logging.INFO,
    extra_data: Optional[dict] = None,
) -> logging.Logger:
    """
    Get a structured logger for a module.

    Args:
        name: Logger name (typically __name__)
        level: Logging level
        extra_data: Context metadata attached to every log line

    Returns:
        Configured logger with structured JSON output
    """
    logger = logging.getLogger(name)

    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(StructuredFormatter())
        logger.addHandler(handler)
        logger.setLevel(level)

    if extra_data:
        context_filter = ContextFilter(extra_data)
        logger.addFilter(context_filter)

    return logger
