"""Safe arithmetic evaluation for the calculator tool.

Text written by a model must never reach ``eval``. This parses the expression into an AST and
only allows numbers, arithmetic operators and a few pure math functions; anything else (names,
attribute access, imports, comprehensions) is rejected. Numeric constants are converted to
``float`` so an expression like ``9**9**9`` overflows quickly instead of eating memory and CPU.
"""

from __future__ import annotations

import ast
import math
import operator
from collections.abc import Callable

MAX_EXPRESSION_LENGTH = 200
MAX_AST_NODES = 60

_BINARY: dict[type, Callable[[float, float], float]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY: dict[type, Callable[[float], float]] = {ast.UAdd: operator.pos, ast.USub: operator.neg}
_FUNCTIONS: dict[str, Callable[..., float]] = {
    "sqrt": math.sqrt,
    "log": math.log,
    "log10": math.log10,
    "exp": math.exp,
    "abs": abs,
    "round": round,
    "min": min,
    "max": max,
}
_CONSTANTS = {"pi": math.pi, "e": math.e}


class UnsafeExpressionError(ValueError):
    """The expression is malformed, too complex, or uses a disallowed construct."""


def safe_eval(expression: str) -> float:
    """Evaluate an arithmetic expression such as ``"2 * (3 + 4) ** 2 / sqrt(16)"``.

    Raises:
        UnsafeExpressionError: for anything outside the allowed grammar or on math errors.
    """
    if not isinstance(expression, str) or not expression.strip():
        raise UnsafeExpressionError("Expression is empty.")
    if len(expression) > MAX_EXPRESSION_LENGTH:
        raise UnsafeExpressionError(f"Expression longer than {MAX_EXPRESSION_LENGTH} characters.")
    try:
        tree = ast.parse(expression.strip(), mode="eval")
    except SyntaxError as exc:
        raise UnsafeExpressionError("Not a valid arithmetic expression.") from exc
    if sum(1 for _ in ast.walk(tree)) > MAX_AST_NODES:
        raise UnsafeExpressionError("Expression is too complex.")
    try:
        result = _eval(tree.body)
    except (ZeroDivisionError, OverflowError, ValueError) as exc:
        raise UnsafeExpressionError(f"Math error: {exc}") from exc
    if isinstance(result, complex) or math.isnan(result) or math.isinf(result):
        raise UnsafeExpressionError("Result is not a finite real number.")
    return result


def _eval(node: ast.AST) -> float:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) \
            and not isinstance(node.value, bool):
        return float(node.value)
    if isinstance(node, ast.Name) and node.id in _CONSTANTS:
        return _CONSTANTS[node.id]
    if isinstance(node, ast.BinOp) and type(node.op) in _BINARY:
        return _BINARY[type(node.op)](_eval(node.left), _eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY:
        return _UNARY[type(node.op)](_eval(node.operand))
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in _FUNCTIONS
        and not node.keywords
    ):
        return _FUNCTIONS[node.func.id](*(_eval(arg) for arg in node.args))
    raise UnsafeExpressionError(f"Disallowed syntax: {type(node).__name__}")
