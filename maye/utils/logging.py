import logging
from functools import lru_cache
from pathlib import Path

import colorlog
from torch import distributed as dist


def get_logger(path: Path):
    logger = logging.getLogger(path.as_posix())
    logger.propagate = False  # Disable propagation up to root logger
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        file_handler = logging.FileHandler(path)
        file_handler.setLevel(logging.INFO)

        console_handler = colorlog.StreamHandler()
        console_handler.setLevel(logging.INFO)

        formatter = logging.Formatter(
            "%(asctime)s - %(levelname)s - %(message)s", datefmt="%m-%d %H:%M:%S"
        )
        color_formatter = colorlog.ColoredFormatter(
            "%(log_color)s%(asctime)-8s%(reset)s %(log_color)s%(levelname)-8s%(reset)s %(blue)s%(message)s",
            datefmt="%m-%d %H:%M:%S",
            reset=True,
            log_colors={
                "DEBUG": "cyan",
                "INFO": "green",
                "WARNING": "yellow",
                "ERROR": "red",
                "CRITICAL": "red,bg_white",
            },
            secondary_log_colors={},
            style="%",
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(color_formatter)

        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

    return logger


logger = get_logger(Path(__file__).parent.parent.parent / "log.log")


@lru_cache(None)
def log_once(msg: str, logger=logger, level: int = logging.INFO) -> None:
    log_rank_zero(logger=logger, msg=msg, level=level)


def log_rank_zero(msg: str, logger=logger, level: int = logging.INFO) -> None:
    rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
    if rank != 0:
        return
    logger.log(level, msg)
