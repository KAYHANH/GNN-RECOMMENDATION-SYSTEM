from __future__ import annotations

import logging


def configure_logging(log_level: str) -> None:
    root_logger = logging.getLogger()
    resolved_level = getattr(logging, log_level.upper(), logging.INFO)

    if root_logger.handlers:
        root_logger.setLevel(resolved_level)
        return

    logging.basicConfig(
        level=resolved_level,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )
