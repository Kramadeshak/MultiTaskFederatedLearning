import logging
import sys

_LOGGER = None

def get_logger():
    global _LOGGER
    if _LOGGER is not None:
        return _LOGGER

    logger = logging.getLogger("fedsim")
    logger.setLevel(logging.DEBUG)

    if not logger.hasHandlers():
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter(
            "[%(asctime)s] %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.propagate = False

    _LOGGER = logger
    return logger
