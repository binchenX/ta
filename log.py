import logging


def configure_logging():
    logging.basicConfig(
        level=logging.CRITICAL,
        format="%(asctime)s - %(levelname)s - [ %(module)s ] - %(message)s",
    )
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.CRITICAL)

    # turn on when debugging
    # logging.getLogger("requests").setLevel(logging.DEBUG)
    # logging.getLogger("urllib3").setLevel(logging.DEBUG)
    # logging.getLogger("langchain").setLevel(logging.DEBUG)

    return logger
