"""
Structured Logging Module.
FAANG Pattern: Centralized logging with context, metrics, and traceability.
"""
import logging
import sys
from datetime import datetime
from typing import Optional, Dict, Any
from dataclasses import dataclass
from functools import wraps
import time
import traceback


@dataclass
class LogContext:
    """Context information for structured logging."""
    request_id: Optional[str] = None
    user_id: Optional[str] = None
    component: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in vars(self).items() if v is not None}


class StructuredLogger:
    """
    Production-grade logger with structured output.

    Features:
    - Structured JSON-like output for log aggregation
    - Performance metrics tracking
    - Error context preservation
    - Component-level isolation
    """

    def __init__(self, name: str, level: str = "INFO"):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(getattr(logging, level))

        # Prevent duplicate handlers
        if not self.logger.handlers:
            handler = logging.StreamHandler(sys.stdout)
            formatter = logging.Formatter(
                '%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

        self.context: Optional[LogContext] = None

    def set_context(self, context: LogContext) -> None:
        """Set logging context for request tracing."""
        self.context = context

    def _format_message(self, message: str, extra: Optional[Dict] = None) -> str:
        """Format message with context and extra data."""
        parts = [message]

        if self.context:
            parts.append(f"context={self.context.to_dict()}")

        if extra:
            parts.append(f"data={extra}")

        return " | ".join(parts)

    def debug(self, message: str, **kwargs) -> None:
        self.logger.debug(self._format_message(message, kwargs if kwargs else None))

    def info(self, message: str, **kwargs) -> None:
        self.logger.info(self._format_message(message, kwargs if kwargs else None))

    def warning(self, message: str, **kwargs) -> None:
        self.logger.warning(self._format_message(message, kwargs if kwargs else None))

    def error(self, message: str, exc_info: bool = False, **kwargs) -> None:
        if exc_info:
            kwargs['traceback'] = traceback.format_exc()
        self.logger.error(self._format_message(message, kwargs if kwargs else None))

    def critical(self, message: str, **kwargs) -> None:
        self.logger.critical(self._format_message(message, kwargs if kwargs else None))


def log_performance(logger: StructuredLogger):
    """
    Decorator to log function performance metrics.

    FAANG Pattern: Automatic latency tracking for SLA monitoring.
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.perf_counter()
            func_name = func.__name__

            try:
                result = func(*args, **kwargs)
                elapsed = (time.perf_counter() - start_time) * 1000
                logger.info(
                    f"Function completed",
                    function=func_name,
                    latency_ms=round(elapsed, 2),
                    status="success"
                )
                return result
            except Exception as e:
                elapsed = (time.perf_counter() - start_time) * 1000
                logger.error(
                    f"Function failed",
                    function=func_name,
                    latency_ms=round(elapsed, 2),
                    status="error",
                    error_type=type(e).__name__,
                    exc_info=True
                )
                raise
        return wrapper
    return decorator


# Factory function for module-level loggers
def get_logger(name: str, level: str = "INFO") -> StructuredLogger:
    """Get a structured logger instance."""
    return StructuredLogger(name, level)
