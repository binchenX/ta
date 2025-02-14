import logging


def configure_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - [ %(name)s - %(module)s ] - %(message)s",
    )
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    # turn on when debugging
    # logging.getLogger("requests").setLevel(logging.DEBUG)
    # logging.getLogger("urllib3").setLevel(logging.DEBUG)
    # logging.getLogger("langchain").setLevel(logging.DEBUG)

    return logger
