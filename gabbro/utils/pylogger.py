import logging
import os

# from rich.logging import RichHandler


def get_pylogger(name=__name__, rank=None) -> logging.Logger:
    """Initializes multi-GPU-friendly python command line logger.

    Parameters
    ----------
    name : str, optional
        Name of the logger. Default is __name__.
    rank : int, optional
        Rank of the current process. If not provided, it will be retrieved from
        torch.distributed.get_rank().

    Returns
    -------
    logging.Logger
        Logger object.
    """
    if rank is None:
        rank = "unknown"
    rank_string = f"rank:{rank}"

    hostname = os.getenv("HOSTNAME", default="unknown-host")

    # logging.basicConfig(level="INFO", datefmt="[%X]", handlers=[RichHandler()])

    logger = logging.getLogger(f"{hostname}|{rank_string}|{name}")

    # reset the logging basic config to avoid duplicated logs
    # logging.basicConfig(level="INFO", datefmt="[%X]", handlers=[RichHandler()])

    # this ensures all logging levels get marked with the rank zero decorator
    # otherwise logs would get multiplied for each GPU process in multi-GPU setup
    # logging_levels = ("debug", "info", "warning", "error", "exception", "fatal", "critical")
    # for level in logging_levels:
    #     setattr(logger, level, rank_zero_only(getattr(logger, level)))

    return logger
