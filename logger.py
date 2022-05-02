import logging


LEVEL = "INFO"
"""str: The log level to use."""

FORMAT = "%(asctime)s@%(thread)d %(levelname)s %(module)s(%(lineno)d):%(funcName)s|: %(message)s"
"""str: The log format to use."""

# DATEFMT = "%H:%M:%S.%f"

NAME = "dataset_parsing"

logging.basicConfig(
        format=FORMAT,
        level=LEVEL)

LOGGER = logging.getLogger(NAME)

