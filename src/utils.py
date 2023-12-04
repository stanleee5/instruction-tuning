"""
utils
"""

import logging
import os
from itertools import chain
from types import FrameType
from typing import cast

from loguru import logger


def get_local_rank_from_launcher():
    # DeepSpeed launcher will set it so get from there
    rank = os.environ.get("LOCAL_RANK")

    if rank is None:
        rank = os.environ.get("OMPI_COMM_WORLD_LOCAL_RANK")

    # Make it a single process job and set rank to 0
    if rank is None:
        rank = 0

    return int(rank)


def is_main_process() -> bool:
    return get_local_rank_from_launcher() == 0


class InterceptHandler(logging.Handler):
    """Logs to loguru from Python logging module"""

    def emit(self, record: logging.LogRecord) -> None:
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = str(record.levelno)

        frame, depth = logging.currentframe(), 2
        while frame.f_code.co_filename == logging.__file__:  # noqa: WPS609
            frame = cast(FrameType, frame.f_back)
            depth += 1
        logger_with_opts = logger.opt(depth=depth, exception=record.exc_info)
        try:
            logger_with_opts.log(level, "{}", record.getMessage())
        except Exception as e:
            safe_msg = getattr(record, "msg", None) or str(record)
            logger_with_opts.warning(
                "Exception logging the following native logger message: {}, {!r}",
                safe_msg,
                e,
            )


def setup_loguru_logging_intercept(level=logging.DEBUG, modules=()):
    """intercept logging to loguru"""
    logging.basicConfig(handlers=[InterceptHandler()], level=level)  # noqa
    for logger_name in chain(("",), modules):
        logger.info(f"logger_name: {logger_name}")
        mod_logger = logging.getLogger(logger_name)
        mod_logger.handlers = [InterceptHandler(level=level)]
        mod_logger.propagate = False
