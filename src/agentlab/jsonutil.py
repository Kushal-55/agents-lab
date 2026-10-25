"""Pull a JSON object out of messy model output (prose, markdown fences, trailing text).

Models often wrap JSON in commentary or code fences, so ``json.loads`` on the whole reply
fails. Used by Level 3's hand-made tool protocol and the Level 7 planner.
"""

from __future__ import annotations

import json
import logging
from typing import Any

logger = logging.getLogger(__name__)


def extract_json_object(text: str) -> dict[str, Any] | None:
    """Return the first balanced ``{...}`` in ``text`` that parses as a JSON object, else None."""
    # Try each "{" as a possible start. Scan forward tracking brace depth, ignoring braces that
    # sit inside JSON strings, until the matching "}" closes the object.
    start = text.find("{")
    while start != -1:
        depth, in_string, escaped = 0, False, False
        for i in range(start, len(text)):
            ch = text[i]
            if in_string:
                if escaped:
                    escaped = False
                elif ch == "\\":
                    escaped = True
                elif ch == '"':
                    in_string = False
            elif ch == '"':
                in_string = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    try:
                        parsed = json.loads(text[start : i + 1])
                    except json.JSONDecodeError:
                        break
                    if isinstance(parsed, dict):
                        return parsed
                    break
        start = text.find("{", start + 1)
    logger.debug("No JSON object found in model output (%d chars)", len(text))
    return None
