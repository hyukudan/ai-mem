"""
Logging configuration for ai-mem.

Provides structured logging with JSON output option and performance metrics.
"""

import logging
import json
import time
import functools
from typing import Any, Callable, Optional
from contextlib import contextmanager


class JSONFormatter(logging.Formatter):
    """JSON formatter for structured logging output."""

    def format(self, record: logging.LogRecord) -> str:
        log_data = {
            "timestamp": self.formatTime(record),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        # Add extra fields if present
        if hasattr(record, "duration_ms"):
            log_data["duration_ms"] = record.duration_ms
        if hasattr(record, "operation"):
            log_data["operation"] = record.operation
        if hasattr(record, "details"):
            log_data["details"] = record.details

        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_data)


def setup_logging(
    level: str = "INFO",
    json_format: bool = False,
    log_file: Optional[str] = None,
) -> None:
    """
    Configure logging for ai-mem.

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        json_format: If True, output logs as JSON
        log_file: Optional file path to write logs to
    """
    root_logger = logging.getLogger("ai_mem")
    root_logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    # Remove existing handlers
    root_logger.handlers.clear()

    # Create formatter
    if json_format:
        formatter = JSONFormatter()
    else:
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # File handler if specified
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)


def get_logger(name: str) -> logging.Logger:
    """Get a logger with the ai_mem prefix."""
    return logging.getLogger(f"ai_mem.{name}")


@contextmanager
def log_duration(logger: logging.Logger, operation: str, level: int = logging.DEBUG):
    """
    Context manager to log operation duration.

    Usage:
        with log_duration(logger, "database query"):
            result = await db.query(...)
    """
    start = time.perf_counter()
    try:
        yield
    finally:
        duration_ms = (time.perf_counter() - start) * 1000
        logger.log(
            level,
            f"{operation} completed in {duration_ms:.2f}ms",
            extra={"duration_ms": duration_ms, "operation": operation},
        )


def log_async_duration(operation: str, level: int = logging.DEBUG):
    """
    Decorator to log async function duration.

    Usage:
        @log_async_duration("search observations")
        async def search(...):
            ...
    """
    def decorator(func: Callable) -> Callable:
        logger = get_logger(func.__module__.split(".")[-1])

        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            start = time.perf_counter()
            try:
                result = await func(*args, **kwargs)
                duration_ms = (time.perf_counter() - start) * 1000
                logger.log(
                    level,
                    f"{operation} completed in {duration_ms:.2f}ms",
                    extra={"duration_ms": duration_ms, "operation": operation},
                )
                return result
            except Exception as e:
                duration_ms = (time.perf_counter() - start) * 1000
                logger.error(
                    f"{operation} failed after {duration_ms:.2f}ms: {e}",
                    extra={"duration_ms": duration_ms, "operation": operation},
                    exc_info=True,
                )
                raise

        return wrapper
    return decorator


def log_sync_duration(operation: str, level: int = logging.DEBUG):
    """
    Decorator to log sync function duration.

    Usage:
        @log_sync_duration("embed text")
        def embed(...):
            ...
    """
    def decorator(func: Callable) -> Callable:
        logger = get_logger(func.__module__.split(".")[-1])

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start = time.perf_counter()
            try:
                result = func(*args, **kwargs)
                duration_ms = (time.perf_counter() - start) * 1000
                logger.log(
                    level,
                    f"{operation} completed in {duration_ms:.2f}ms",
                    extra={"duration_ms": duration_ms, "operation": operation},
                )
                return result
            except Exception as e:
                duration_ms = (time.perf_counter() - start) * 1000
                logger.error(
                    f"{operation} failed after {duration_ms:.2f}ms: {e}",
                    extra={"duration_ms": duration_ms, "operation": operation},
                    exc_info=True,
                )
                raise

        return wrapper
    return decorator


class PerformanceMetrics:
    """Collect and report performance metrics."""

    def __init__(self):
        self._metrics: dict[str, list[float]] = {}
        self._logger = get_logger("metrics")

    def record(self, operation: str, duration_ms: float) -> None:
        """Record a duration for an operation."""
        if operation not in self._metrics:
            self._metrics[operation] = []
        self._metrics[operation].append(duration_ms)

    def get_stats(self, operation: str) -> dict[str, float]:
        """Get statistics for an operation."""
        durations = self._metrics.get(operation, [])
        if not durations:
            return {}

        return {
            "count": len(durations),
            "total_ms": sum(durations),
            "avg_ms": sum(durations) / len(durations),
            "min_ms": min(durations),
            "max_ms": max(durations),
        }

    def report(self) -> dict[str, dict[str, float]]:
        """Get all metrics statistics."""
        return {op: self.get_stats(op) for op in self._metrics}

    def log_report(self) -> None:
        """Log all metrics."""
        report = self.report()
        for operation, stats in report.items():
            self._logger.info(
                f"Performance: {operation} - "
                f"count={stats['count']}, avg={stats['avg_ms']:.2f}ms, "
                f"min={stats['min_ms']:.2f}ms, max={stats['max_ms']:.2f}ms"
            )


# Global metrics instance
metrics = PerformanceMetrics()
