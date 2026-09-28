"""A restricted Python executor for model-written code.

Two features run code the model wrote: the research agent's search plans (RLM) and the Python
analysis tool. Model-written code is untrusted input, so before it runs it must pass an AST
allow-list, and while it runs it has:

  * no imports, no dunder attribute access, no `while`, no function or class definitions
  * an allow-list of builtins and of attribute names (str/list/dict methods only)
  * a line-event budget (stops runaway loops) and a wall-clock timeout
  * only the functions and data we pass in

This is a guard against a confused or manipulated model, not a boundary against a determined
attacker with arbitrary code: CPython cannot be made safe in-process. In production this runs in
a separate container with no network and a read-only filesystem (gVisor / Firecracker).
"""

from __future__ import annotations

import ast
import asyncio
import json
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any

from kb_assistant.errors import SandboxError

_ALLOWED_NODES: tuple[type[ast.AST], ...] = (
    ast.Module, ast.Expr, ast.Assign, ast.AugAssign, ast.Name, ast.Load, ast.Store, ast.Constant,
    ast.List, ast.Tuple, ast.Dict, ast.Set, ast.ListComp, ast.DictComp, ast.SetComp, ast.GeneratorExp,
    ast.comprehension, ast.For, ast.If, ast.IfExp, ast.BoolOp, ast.And, ast.Or, ast.UnaryOp, ast.Not,
    ast.USub, ast.UAdd, ast.BinOp, ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod,
    ast.Compare, ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE, ast.In, ast.NotIn, ast.Is,
    ast.IsNot, ast.Call, ast.keyword, ast.Attribute, ast.Subscript, ast.Slice, ast.Break,
    ast.Continue, ast.Pass, ast.JoinedStr, ast.FormattedValue, ast.Lambda, ast.arguments, ast.arg,
    ast.Starred,
)

_ALLOWED_ATTRS = frozenset({
    # str
    "lower", "upper", "strip", "split", "startswith", "endswith", "replace", "join", "title",
    "count", "find",
    # list / dict / set / Counter
    "append", "extend", "sort", "get", "items", "keys", "values", "setdefault", "update", "add",
    "most_common", "index", "pop",
})

_SAFE_BUILTINS: dict[str, Any] = {
    "len": len, "sorted": sorted, "min": min, "max": max, "sum": sum, "any": any, "all": all,
    "enumerate": enumerate, "zip": zip, "set": set, "list": list, "dict": dict, "tuple": tuple,
    "str": str, "int": int, "float": float, "round": round, "abs": abs, "bool": bool,
    "reversed": reversed, "Counter": Counter, "defaultdict": defaultdict,
    "range": lambda *a: range(*a) if len(range(*a)) <= 10_000 else _raise("range too large"),
    "True": True, "False": False, "None": None,
}


def _raise(msg: str) -> Any:
    raise SandboxError(msg)


@dataclass
class SandboxResult:
    result: Any
    output: list[str] = field(default_factory=list)
    lines_executed: int = 0


def validate(code: str, max_chars: int = 4000) -> ast.Module:
    if len(code) > max_chars:
        raise SandboxError(f"code longer than {max_chars} characters")
    try:
        tree = ast.parse(code, mode="exec")
    except SyntaxError as exc:
        raise SandboxError(f"syntax error: {exc.msg} (line {exc.lineno})") from exc
    for node in ast.walk(tree):
        if not isinstance(node, _ALLOWED_NODES):
            raise SandboxError(f"'{type(node).__name__}' is not allowed")
        if isinstance(node, ast.Attribute):
            if node.attr.startswith("_") or node.attr not in _ALLOWED_ATTRS:
                raise SandboxError(f"attribute '.{node.attr}' is not allowed")
        if isinstance(node, ast.Name) and node.id.startswith("__"):
            raise SandboxError(f"name '{node.id}' is not allowed")
    return tree


def _run(tree: ast.Module, env: dict[str, Any], max_lines: int) -> SandboxResult:
    output: list[str] = []
    lines = 0

    def tracer(frame, event, arg):  # noqa: ANN001
        nonlocal lines
        if event == "line":
            lines += 1
            if lines > max_lines:
                raise SandboxError(f"execution budget of {max_lines} steps exceeded")
        return tracer

    def _print(*args: Any) -> None:
        if len(output) < 50:
            output.append(" ".join(str(a) for a in args)[:500])

    scope: dict[str, Any] = {"__builtins__": {**_SAFE_BUILTINS, "print": _print}, **env}
    sys.settrace(tracer)
    try:
        exec(compile(tree, "<sandbox>", "exec"), scope)  # noqa: S102 - validated AST, see module doc
    except SandboxError:
        raise
    except Exception as exc:
        raise SandboxError(f"{type(exc).__name__}: {exc}") from exc
    finally:
        sys.settrace(None)
    if "result" not in scope:
        raise SandboxError("code must assign its answer to a variable named `result`")
    return SandboxResult(_jsonable(scope["result"]), output, lines)


def _jsonable(value: Any) -> Any:
    try:
        return json.loads(json.dumps(value, default=str))
    except (TypeError, ValueError) as exc:
        raise SandboxError(f"result is not serialisable: {exc}") from exc


async def run_sandboxed(
    code: str, env: dict[str, Any], timeout_s: float = 5.0, max_lines: int = 200_000
) -> SandboxResult:
    """Validate, then execute in a worker thread so the event loop stays free. Functions in `env`
    that need async services call back into the loop with run_coroutine_threadsafe."""
    tree = validate(code)
    try:
        return await asyncio.wait_for(asyncio.to_thread(_run, tree, env, max_lines), timeout=timeout_s)
    except TimeoutError as exc:
        raise SandboxError(f"timed out after {timeout_s}s") from exc
