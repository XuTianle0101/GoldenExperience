#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "$REPO_ROOT"

python_bin="${PYTHON_BIN:-python3}"

runtime_libs="$("$python_bin" - <<'PY' || true
from __future__ import annotations

import importlib.util
import site
import sys
import sysconfig
from pathlib import Path

paths: list[str] = []


def add(path: Path) -> None:
    if path.is_dir():
        value = str(path)
        if value not in paths:
            paths.append(value)


torch_spec = importlib.util.find_spec("torch")
if torch_spec and torch_spec.submodule_search_locations:
    for location in torch_spec.submodule_search_locations:
        add(Path(location) / "lib")

site_roots = {Path(sys.prefix) / "lib"}
for key in ("purelib", "platlib"):
    if value := sysconfig.get_paths().get(key):
        site_roots.add(Path(value))
try:
    site_roots.update(Path(value) for value in site.getsitepackages())
except AttributeError:
    pass

for root in site_roots:
    add(root)
    nvidia_root = root / "nvidia"
    if nvidia_root.is_dir():
        for lib_dir in nvidia_root.glob("*/lib"):
            add(lib_dir)

mooncake_spec = importlib.util.find_spec("mooncake")
if mooncake_spec and mooncake_spec.submodule_search_locations:
    for location in mooncake_spec.submodule_search_locations:
        mooncake_dir = Path(location)
        add(mooncake_dir)
        add(mooncake_dir.parent / "mooncake_transfer_engine.libs")

add(Path("/usr/local/cuda/lib64"))
add(Path("/usr/local/nvidia/lib64"))
print(":".join(paths))
PY
)"
if [ -n "$runtime_libs" ]; then
  export LD_LIBRARY_PATH="${runtime_libs}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
fi

exec "$python_bin" -m goldenexperience.runtime.kv_baseline "$@"
