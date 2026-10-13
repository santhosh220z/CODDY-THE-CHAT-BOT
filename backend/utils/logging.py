import logging
import sys

from backend.utils.config import settings


def configure_logging() -> logging.Logger:
    level = getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO)

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        logging.Formatter("%(asctime)s [%(levelname)-7s] %(name)s: %(message)s", "%H:%M:%S")
    )

    root = logging.getLogger()
    root.setLevel(level)
    # Replace rather than append, so repeated imports do not duplicate output.
    root.handlers = [handler]

    # Third-party chatter.
    for noisy in (
        "chromadb",
        "sentence_transformers",
        "urllib3",
        "httpx",
        "httpcore",
        "filelock",
        "transformers",
        "pypdf",
        "markdown_it",
    ):
        logging.getLogger(noisy).setLevel(max(level, logging.WARNING))

    logger = logging.getLogger("coddy")
    logger.setLevel(level)
    return logger


logger = configure_logging()