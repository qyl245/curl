"""
日志工具
"""
import logging
import sys

_logger_initialized = False


def setup_logger(name: str = "biceps", level=logging.INFO) -> logging.Logger:
    global _logger_initialized
    logger = logging.getLogger(name)
    if not _logger_initialized:
        logger.setLevel(level)
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(level)
        fmt = logging.Formatter(
            "[%(asctime)s %(levelname)s] %(message)s", datefmt="%H:%M:%S"
        )
        handler.setFormatter(fmt)
        logger.addHandler(handler)
        _logger_initialized = True
    return logger
