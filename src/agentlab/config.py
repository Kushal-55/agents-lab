"""Central, validated configuration.

All settings come from environment variables (optionally loaded from a git-ignored
`.env` file). Secrets are held in a field excluded from ``repr`` so they cannot leak
through logs or tracebacks.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class ProviderPreset:
    """Connection defaults for one free, OpenAI-compatible LLM backend."""

    base_url: str
    key_env: str | None  # env var holding the API key; None = no key needed
    default_model: str


# Every preset speaks the OpenAI chat-completions protocol, so one client covers all.
# Free-tier availability and model names change over time: override with LLM_MODEL.
PROVIDERS: dict[str, ProviderPreset] = {
    "ollama": ProviderPreset("http://localhost:11434/v1", None, "llama3.1:8b"),
    "openrouter": ProviderPreset(
        "https://openrouter.ai/api/v1",
        "OPENROUTER_API_KEY",
        "nvidia/nemotron-3-super-120b-a12b:free",
    ),
    "groq": ProviderPreset(
        "https://api.groq.com/openai/v1", "GROQ_API_KEY", "llama-3.3-70b-versatile"
    ),
    "gemini": ProviderPreset(
        "https://generativelanguage.googleapis.com/v1beta/openai/",
        "GEMINI_API_KEY",
        "gemini-2.5-flash",
    ),
}


class ConfigError(ValueError):
    """Raised when configuration is missing or invalid."""


@dataclass(frozen=True)
class Settings:
    provider: str
    model: str
    base_url: str
    api_key: str = field(repr=False)  # never printed
    timeout_s: float = 60.0
    max_retries: int = 3
    temperature: float = 0.2
    log_level: str = "WARNING"
    data_dir: Path = PROJECT_ROOT / "data" / "runtime"
    knowledge_dir: Path = PROJECT_ROOT / "data" / "knowledge"


def _env_number(name: str, default: float, low: float, high: float) -> float:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        value = float(raw)
    except ValueError as exc:
        raise ConfigError(f"{name} must be a number, got {raw!r}") from exc
    if not low <= value <= high:
        raise ConfigError(f"{name} must be between {low} and {high}, got {value}")
    return value


def load_settings(*, load_env_file: bool = True, require_key: bool = True) -> Settings:
    """Build validated :class:`Settings` from the environment.

    Args:
        require_key: set False for commands that never call the LLM (e.g. offline forecasting).

    Raises:
        ConfigError: unknown provider, missing API key (if required), or out-of-range number.
    """
    if load_env_file:
        # Real environment variables win over the .env file (override=False).
        load_dotenv(PROJECT_ROOT / ".env", override=False)

    provider = os.getenv("LLM_PROVIDER", "ollama").strip().lower()
    preset = PROVIDERS.get(provider)
    if preset is None:
        raise ConfigError(f"Unknown LLM_PROVIDER {provider!r}. Choose one of: {sorted(PROVIDERS)}")

    api_key = "not-needed"
    if preset.key_env:
        api_key = os.getenv(preset.key_env, "").strip()
        if not api_key and require_key:
            raise ConfigError(
                f"Provider {provider!r} needs {preset.key_env}. Add it to your .env file "
                "(see .env.example) or use LLM_PROVIDER=ollama for a key-free local model."
            )

    # Where journals, caches, memory and logs are written. Overridable so tests and experiments
    # never touch the real data.
    data_dir = Path(os.getenv("AGENTLAB_DATA_DIR") or PROJECT_ROOT / "data" / "runtime")
    settings = Settings(
        provider=provider,
        model=os.getenv("LLM_MODEL", "").strip() or preset.default_model,
        base_url=preset.base_url,
        api_key=api_key,
        timeout_s=_env_number("LLM_TIMEOUT_S", 60.0, 1.0, 600.0),
        max_retries=int(_env_number("LLM_MAX_RETRIES", 3, 0, 10)),
        temperature=_env_number("LLM_TEMPERATURE", 0.2, 0.0, 2.0),
        log_level=os.getenv("LOG_LEVEL", "WARNING").strip().upper(),
        data_dir=data_dir.expanduser().resolve(),
    )
    # Safe to log the whole object: api_key is excluded from Settings.__repr__.
    logger.debug("Loaded settings: %s", settings)
    return settings
