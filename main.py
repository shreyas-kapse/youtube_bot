import logging

from utils.logger import setup_logger

logger = logging.getLogger(__name__)

def main():
    setup_logger()
    logger.info("Starting application")
    logger.warning("check IO")

main()
