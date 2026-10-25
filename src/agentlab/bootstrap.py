"""Shared start-up for the level scripts: settings, logging, and the LLM client."""

from __future__ import annotations

import logging
import sys

from agentlab.config import ConfigError, Settings, load_settings
from agentlab.llm import LLMClient
from agentlab.log import setup_logging

logger = logging.getLogger(__name__)


def start() -> tuple[Settings, LLMClient]:
    """Load config, configure logging and build the LLM client (exits cleanly on bad config)."""
    try:
        settings = load_settings()
    except ConfigError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
    setup_logging(settings.log_level, settings.data_dir / "agentlab.log")
    logger.info("Using provider=%s model=%s", settings.provider, settings.model)
    return settings, LLMClient(settings)
