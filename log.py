import logging


def configure_logging(default_level=logging.CRITICAL):
    logging.basicConfig(
        level=default_level,
        format="%(asctime)s - %(levelname)s - [ %(module)s ] - %(message)s",
    )
    logger = logging.getLogger(__name__)
    logger.setLevel(default_level)
    return logger
