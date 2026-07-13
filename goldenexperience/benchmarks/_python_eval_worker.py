"""Resource-limited worker for publication code-task evaluation."""

from __future__ import annotations

import ast
import builtins
import collections
import functools
import heapq
import itertools
import json
import math
import resource
import statistics
import sys
from typing import Any

_ALLOWED_MODULES = {
    "collections": collections,
    "functools": functools,
    "heapq": heapq,
    "itertools": itertools,
    "math": math,
    "statistics": statistics,
}
_FORBIDDEN_NAMES = {
    "breakpoint",
    "compile",
    "delattr",
    "eval",
    "exec",
    "getattr",
    "globals",
    "input",
    "locals",
    "open",
    "setattr",
    "vars",
    "__import__",
}


def _limit_process() -> None:
    resource.setrlimit(resource.RLIMIT_CPU, (3, 3))
    resource.setrlimit(resource.RLIMIT_AS, (512 * 1024 * 1024, 512 * 1024 * 1024))
    resource.setrlimit(resource.RLIMIT_FSIZE, (1024 * 1024, 1024 * 1024))
    resource.setrlimit(resource.RLIMIT_NOFILE, (16, 16))
    resource.setrlimit(resource.RLIMIT_NPROC, (1, 1))


def _validate_candidate(source: str) -> None:
    tree = ast.parse(source, mode="exec")
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            if any(alias.name.split(".", 1)[0] not in _ALLOWED_MODULES for alias in node.names):
                raise ValueError("candidate imports an unavailable module")
        elif isinstance(node, ast.ImportFrom):
            if node.module is None or node.module.split(".", 1)[0] not in _ALLOWED_MODULES:
                raise ValueError("candidate imports an unavailable module")
        elif isinstance(node, ast.Name) and node.id in _FORBIDDEN_NAMES:
            raise ValueError("candidate uses a forbidden builtin")
        elif isinstance(node, ast.Attribute) and node.attr.startswith("__"):
            raise ValueError("candidate uses forbidden dunder introspection")


def _safe_import(
    name: str,
    _globals: Any = None,
    _locals: Any = None,
    fromlist: Any = (),
    level: int = 0,
) -> Any:
    if level != 0 or name.split(".", 1)[0] not in _ALLOWED_MODULES:
        raise ImportError("module is unavailable in publication evaluation")
    return _ALLOWED_MODULES[name.split(".", 1)[0]]


def _audit(event: str, _args: tuple[Any, ...]) -> None:
    if event == "open" or event.startswith(
        ("socket.", "subprocess.", "os.system", "os.spawn", "os.exec", "ctypes.")
    ):
        raise PermissionError("operation blocked in publication evaluation")


def main() -> int:
    _limit_process()
    payload = json.loads(sys.stdin.read())
    candidate = payload["candidate_code"]
    test_code = payload["test_code"]
    entry_point = payload["entry_point"]
    test_mode = payload["test_mode"]
    if not all(isinstance(value, str) for value in (candidate, test_code, entry_point, test_mode)):
        return 2
    _validate_candidate(candidate)
    safe_builtins = dict(vars(builtins))
    for name in _FORBIDDEN_NAMES:
        safe_builtins.pop(name, None)
    safe_builtins["__import__"] = _safe_import
    namespace: dict[str, Any] = {"__builtins__": safe_builtins, "__name__": "__candidate__"}
    sys.addaudithook(_audit)
    exec(compile(candidate, "<candidate>", "exec"), namespace)  # noqa: S102
    entry = namespace.get(entry_point)
    if not callable(entry):
        return 3
    exec(compile(test_code, "<publication-tests>", "exec"), namespace)  # noqa: S102
    if test_mode == "check":
        check = namespace.get("check")
        if not callable(check):
            return 4
        check(entry)
    elif test_mode != "exec":
        return 5
    sys.stdout.write('{"passed": true}\n')
    return 0


if __name__ == "__main__":
    try:
        exit_code = main()
    except BaseException:
        sys.stdout.write('{"passed": false}\n')
        raise SystemExit(1) from None
    raise SystemExit(exit_code)
