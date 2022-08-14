import logging
import os

from pathlib import Path

########################################################################################################################
# LOGGING UTILITY                                                                                                      #
########################################################################################################################


def get_logger(file_name, log_path=None) -> logging.Logger:
    """
    Get a logger with particular formatting
    :param file_name: log file name
    :param log_path: path to log directory (if None, make one at the root level)
    :return:
        logger object
    """

    fmt = "%(levelname)s :: %(asctime)s :: Process ID %(process)s :: %(module)s :: " + \
          "%(funcName)s() :: Line %(lineno)d :: %(message)s"

    formatter = logging.Formatter(fmt)
    root_logger = logging.getLogger()

    # Log path
    if not log_path:
        log_path = Path(__file__).parent.parent.parent / "logs"  # /src/utils/logger.py
    log_path = Path(log_path)

    # Make sure the path exists
    if not log_path.exists():
        os.makedirs(log_path)

    # Print to file
    file_handler = logging.FileHandler(log_path / f"{file_name}.log")
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)

    # Print to console
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    return root_logger
