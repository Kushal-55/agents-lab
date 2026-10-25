"""Tool definitions and a safe execution registry.

Model output is untrusted input. The registry therefore:
  * parses arguments as JSON and validates them against the declared schema,
  * never lets a tool exception escape (the model gets a short error instead),
  * caps the size of every tool result so one call cannot flood the context window.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

MAX_TOOL_OUTPUT_CHARS = 4000
MAX_ARGUMENT_JSON_CHARS = 8000

_JSON_TYPES: dict[str, tuple[type, ...]] = {
    "string": (str,),
    "integer": (int,),
    "number": (int, float),
    "boolean": (bool,),
}


class ToolError(Exception):
    """An expected, user-presentable tool failure. The message is shown to the model."""


def param(type_: str, description: str, **extra: Any) -> dict[str, Any]:
    """Build one JSON-schema property, e.g. ``param("string", "Stock ticker")``."""
    return {"type": type_, "description": description, **extra}


def object_schema(properties: dict[str, dict[str, Any]], required: list[str]) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": properties,
        "required": required,
        "additionalProperties": False,
    }


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    parameters: dict[str, Any]
    func: Callable[..., str]

    def spec(self) -> dict[str, Any]:
        """OpenAI-style function-calling spec."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


def _validate(schema: dict[str, Any], args: dict[str, Any]) -> None:
    properties: dict[str, dict[str, Any]] = schema.get("properties", {})
    for name in schema.get("required", []):
        if name not in args:
            raise ToolError(f"Missing required argument '{name}'.")
    for name, value in args.items():
        spec = properties.get(name)
        if spec is None:
            raise ToolError(f"Unknown argument '{name}'.")
        expected = _JSON_TYPES.get(spec.get("type", ""))
        if expected is None:
            continue
        # bool is a subclass of int in Python; reject it where a number is expected.
        if isinstance(value, bool) and spec["type"] != "boolean":
            raise ToolError(f"Argument '{name}' must be of type {spec['type']}.")
        if not isinstance(value, expected):
            raise ToolError(f"Argument '{name}' must be of type {spec['type']}.")
        if "enum" in spec and value not in spec["enum"]:
            raise ToolError(f"Argument '{name}' must be one of {spec['enum']}.")
        if isinstance(value, (int, float)):
            if "minimum" in spec and value < spec["minimum"]:
                raise ToolError(f"Argument '{name}' must be >= {spec['minimum']}.")
            if "maximum" in spec and value > spec["maximum"]:
                raise ToolError(f"Argument '{name}' must be <= {spec['maximum']}.")


class ToolRegistry:
    def __init__(self, tools: list[Tool] | None = None) -> None:
        self._tools: dict[str, Tool] = {}
        for tool in tools or []:
            self.register(tool)

    def register(self, tool: Tool) -> None:
        if tool.name in self._tools:
            raise ValueError(f"Tool {tool.name!r} is already registered.")
        self._tools[tool.name] = tool

    def names(self) -> list[str]:
        return list(self._tools)

    def specs(self) -> list[dict[str, Any]]:
        return [t.spec() for t in self._tools.values()]

    def execute(self, name: str, raw_arguments: str) -> str:
        """Run a tool and always return a string (errors included)."""
        tool = self._tools.get(name)
        if tool is None:
            # name comes from the model: %r escapes newlines (log injection) and [:60] caps it.
            logger.warning("Model requested unknown tool %r", name[:60])
            return f"Error: unknown tool '{name}'. Available tools: {', '.join(self._tools)}."
        if len(raw_arguments or "") > MAX_ARGUMENT_JSON_CHARS:
            return "Error: tool arguments are too large."
        try:
            args = json.loads(raw_arguments) if raw_arguments and raw_arguments.strip() else {}
        except json.JSONDecodeError:
            return "Error: tool arguments were not valid JSON."
        if not isinstance(args, dict):
            return "Error: tool arguments must be a JSON object."

        try:
            _validate(tool.parameters, args)
            output = tool.func(**args)
        except ToolError as exc:
            logger.info("Tool %s rejected the call: %s", name, exc)
            return f"Error: {exc}"
        except Exception:  # noqa: BLE001 - last line of defence; details go to the log only
            logger.exception("Tool %s crashed", name)
            return f"Error: tool '{name}' failed unexpectedly."

        text = output if isinstance(output, str) else str(output)
        if len(text) > MAX_TOOL_OUTPUT_CHARS:
            logger.info("Tool %s output truncated (%d chars)", name, len(text))
            text = text[:MAX_TOOL_OUTPUT_CHARS] + "\n...[truncated]"
        return text
