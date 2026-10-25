"""A thin, provider-agnostic chat client over the OpenAI-compatible protocol.

``ChatModel`` is a structural type, so tests (and later levels) can swap in a fake model
without touching the network.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Protocol

import openai
from openai import OpenAI

from agentlab.config import Settings

logger = logging.getLogger(__name__)


class LLMError(RuntimeError):
    """A user-presentable failure talking to the model (never contains secrets)."""


@dataclass(frozen=True)
class ToolCall:
    id: str
    name: str
    arguments: str  # raw JSON string exactly as produced by the model


@dataclass(frozen=True)
class ChatResult:
    content: str
    tool_calls: tuple[ToolCall, ...] = ()
    prompt_tokens: int = 0
    completion_tokens: int = 0
    finish_reason: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    def as_message(self) -> dict[str, Any]:
        """The assistant message to append to the conversation history."""
        message: dict[str, Any] = {"role": "assistant", "content": self.content or ""}
        if self.tool_calls:
            message["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.name, "arguments": tc.arguments},
                }
                for tc in self.tool_calls
            ]
        return message


class ChatModel(Protocol):
    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        *,
        temperature: float | None = None,
        max_tokens: int = 1024,
    ) -> ChatResult: ...


class LLMClient:
    """Chat client for Ollama / OpenRouter / Groq / Gemini (all OpenAI-compatible)."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        # The SDK retries connection errors, 408/409/429 and 5xx with exponential backoff.
        self._client = OpenAI(
            base_url=settings.base_url,
            api_key=settings.api_key,
            timeout=settings.timeout_s,
            max_retries=settings.max_retries,
        )

    @property
    def model(self) -> str:
        return self._settings.model

    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        *,
        temperature: float | None = None,
        max_tokens: int = 1024,
    ) -> ChatResult:
        kwargs: dict[str, Any] = {
            "model": self._settings.model,
            "messages": messages,
            "temperature": self._settings.temperature if temperature is None else temperature,
            "max_tokens": max_tokens,
        }
        if tools:
            kwargs["tools"] = tools
        logger.debug("LLM request: model=%s messages=%d tools=%d", self.model, len(messages),
                     len(tools or []))
        # Each branch below turns an SDK exception into a short message the user can act on.
        # Only the exception type is logged, never the request or response body, because
        # those can echo back credentials or the user's text.
        try:
            response = self._client.chat.completions.create(**kwargs)
        except openai.APIConnectionError as exc:
            logger.error("LLM connection failed (%s)", self._settings.provider)
            raise LLMError(self._connection_hint()) from exc
        except openai.AuthenticationError as exc:
            logger.error("LLM authentication failed (%s)", self._settings.provider)
            raise LLMError(
                f"Authentication failed for provider {self._settings.provider!r}. "
                "Check the API key in your .env file."
            ) from exc
        except openai.RateLimitError as exc:
            logger.warning("LLM rate limited (%s, model %s)", self._settings.provider, self.model)
            raise LLMError(
                "Rate limited by the free tier. Wait a minute, or switch model/provider."
            ) from exc
        except openai.NotFoundError as exc:
            logger.error("LLM model not found: %s on %s", self.model, self._settings.provider)
            raise LLMError(
                f"Model {self.model!r} was not found on {self._settings.provider!r}. "
                "Set LLM_MODEL to an available model."
            ) from exc
        except openai.APIStatusError as exc:
            logger.error("LLM API error: HTTP %s", exc.status_code)
            raise LLMError(f"Model API error (HTTP {exc.status_code}).") from exc
        except openai.OpenAIError as exc:
            logger.error("Unexpected LLM client error: %s", type(exc).__name__)
            raise LLMError(f"Unexpected LLM client error: {type(exc).__name__}") from exc

        # Some free providers return HTTP 200 with an empty choices list when overloaded.
        if not response.choices:
            logger.warning("LLM returned no choices (%s)", self._settings.provider)
            raise LLMError("The model returned no choices (the provider may be overloaded).")
        choice = response.choices[0]
        message = choice.message
        # Providers sometimes omit the tool-call id, so make one up; the id is only used to
        # pair a tool result with its request in the next message.
        calls = tuple(
            ToolCall(
                id=tc.id or f"call_{i}", name=tc.function.name, arguments=tc.function.arguments
            )
            for i, tc in enumerate(message.tool_calls or [])
            if tc.type == "function"
        )
        usage = response.usage
        result = ChatResult(
            content=message.content or "",
            tool_calls=calls,
            prompt_tokens=getattr(usage, "prompt_tokens", 0) or 0,
            completion_tokens=getattr(usage, "completion_tokens", 0) or 0,
            finish_reason=choice.finish_reason or "",
        )
        logger.debug("LLM response: tokens in/out=%d/%d tool_calls=%d", result.prompt_tokens,
                     result.completion_tokens, len(calls))
        return result

    def _connection_hint(self) -> str:
        if self._settings.provider == "ollama":
            return (
                "Cannot reach Ollama at localhost:11434. Install it from https://ollama.com, "
                f"run `ollama pull {self.model}`, or set LLM_PROVIDER=openrouter (free models)."
            )
        return f"Cannot reach {self._settings.provider!r}. Check your internet connection."
