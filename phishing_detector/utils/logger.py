"""
PhishGuard AI - Logging Configuration
Colored console output + rotating file logs.
"""
import logging
import logging.handlers
import sys
from pathlib import Path

try:
    import colorlog
    HAS_COLOR = True
except ImportError:
    HAS_COLOR = False

_initialized = False


def get_logger(name: str = "phishguard") -> logging.Logger:
    """Get a configured logger. Safe to call multiple times."""
    global _initialized

    logger = logging.getLogger(name)

    if _initialized:
        return logger

    logger.setLevel(logging.DEBUG)

    # ── Console handler ────────────────────────────────────────────────────────
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)

    if HAS_COLOR:
        fmt = colorlog.ColoredFormatter(
            "%(log_color)s%(asctime)s [%(levelname)-8s]%(reset)s %(name)s: %(message)s",
            datefmt="%H:%M:%S",
            log_colors={
                "DEBUG":    "cyan",
                "INFO":     "green",
                "WARNING":  "yellow",
                "ERROR":    "red",
                "CRITICAL": "red,bg_white",
            },
        )
    else:
        fmt = logging.Formatter(
            "%(asctime)s [%(levelname)-8s] %(name)s: %(message)s",
            datefmt="%H:%M:%S",
        )

    console_handler.setFormatter(fmt)
    logger.addHandler(console_handler)

    # ── File handler ───────────────────────────────────────────────────────────
    try:
        from config.settings import LOG_FILE, LOG_MAX_BYTES, LOG_BACKUPS
        log_path = Path(LOG_FILE)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.handlers.RotatingFileHandler(
            log_path, maxBytes=LOG_MAX_BYTES, backupCount=LOG_BACKUPS,
            encoding="utf-8",
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(logging.Formatter(
            "%(asctime)s [%(levelname)-8s] %(name)s: %(message)s"
        ))
        logger.addHandler(file_handler)
    except Exception:
        pass  # File logging is optional

    _initialized = True
    return logger


# Default logger instance
log = get_logger("phishguard")
