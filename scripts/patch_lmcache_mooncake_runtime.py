#!/usr/bin/env python3
"""Patch LMCache/Mooncake runtime compatibility for the local baseline.

This script makes the Mooncake Store path reproducible for package installs that
ship Mooncake as ``mooncake/store.so`` while LMCache expects
``libmooncake_store.so``. It also patches LMCache 0.4.x to use Mooncake's Python
``MooncakeDistributedStore`` SET/GET API by default and avoids the native
``batchIsExist`` crash path by using an in-process key index for lookup. The Python
adapter also refreshes an optional ``GE_MOONCAKE_EXTERNAL_INDEX`` JSONL sidecar so
materialized cross-model chunks inserted by a sidecar process can be consumed by
vLLM through LMCache MP.
"""

from __future__ import annotations

import argparse
import importlib.util
import shutil
import site
import sys
import sysconfig
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

PYTHON_ADAPTER_MARKER = "# --- GoldenExperience Mooncake Python adapter begin ---"
PYTHON_ADAPTER_END_MARKER = "# --- GoldenExperience Mooncake Python adapter end ---"
PYTHON_ADAPTER_VERSION_MARKER = "GOLDENEXPERIENCE_MOONCAKE_ADAPTER_VERSION = 2"
NATIVE_LOOKUP_MARKER = "# --- GoldenExperience Mooncake native lookup patch begin ---"
NATIVE_LOOKUP_END_MARKER = "# --- GoldenExperience Mooncake native lookup patch end ---"

MOONCAKE_ADAPTER_REQUIRED = (
    "class MooncakePythonL2Adapter",
    "def _env_enabled",
    "LMCACHE_MOONCAKE_PYTHON_ADAPTER",
    "MooncakeDistributedStore",
    "MooncakeStore SET",
    "MooncakeStore GET",
    "GE_MOONCAKE_EXTERNAL_INDEX",
    "_refresh_external_index",
    "external_index_hits",
    PYTHON_ADAPTER_VERSION_MARKER,
    "bytes_read == requested_size == indexed_size",
)

NATIVE_ADAPTER_REQUIRED = (
    "_use_local_lookup_index",
    "_submit_local_lookup_and_lock_task",
    "LMCACHE_MOONCAKE_NATIVE_EXISTS",
)


def is_complete_mooncake_read(
    bytes_read: int,
    requested_size: int,
    indexed_size: int,
) -> bool:
    return bytes_read == requested_size == indexed_size


MOONCAKE_PYTHON_ADAPTER_BLOCK = f'''{PYTHON_ADAPTER_MARKER}
{PYTHON_ADAPTER_VERSION_MARKER}


def _env_enabled(name: str, default: str) -> bool:
    return os.environ.get(name, default).lower() in {{"1", "true", "yes", "on"}}


def _memoryview_ptr_and_size(obj: MemoryObj) -> tuple[int, int]:
    mv = _obj_to_memoryview(obj)
    return ctypes.addressof(ctypes.c_char.from_buffer(mv)), mv.nbytes


class MooncakePythonL2Adapter(L2AdapterInterface):
    """Mooncake Store adapter using the Python MooncakeDistributedStore API."""

    def __init__(self, config: MooncakeStoreL2AdapterConfig):
        super().__init__(max_capacity_bytes=0)
        from mooncake.store import MooncakeDistributedStore

        setup_config = dict(config.setup_config)
        setup_config.pop("storage_root_dir", None)
        self._store = MooncakeDistributedStore()
        rc = self._store.setup(setup_config)
        if rc != 0:
            raise RuntimeError(f"MooncakeDistributedStore.setup failed with rc={{rc}}")

        self._store_efd = create_event_notifier()
        self._lookup_efd = create_event_notifier()
        self._load_efd = create_event_notifier()
        self._completed_stores: dict[L2TaskId, L2StoreResult] = {{}}
        self._completed_lookups: dict[L2TaskId, Bitmap] = {{}}
        self._completed_loads: dict[L2TaskId, Bitmap] = {{}}
        self._key_sizes: dict[ObjectKey, int] = {{}}
        self._locked_keys: dict[ObjectKey, int] = {{}}
        self._external_index_path = os.environ.get("GE_MOONCAKE_EXTERNAL_INDEX", "")
        self._external_index_offset = 0
        self._external_key_sizes: dict[str, int] = {{}}
        self._next_task_id: L2TaskId = 0
        self._lock = threading.Lock()
        self._closed = False

    def get_store_event_fd(self) -> int:
        return self._store_efd.fileno()

    def get_lookup_and_lock_event_fd(self) -> int:
        return self._lookup_efd.fileno()

    def get_load_event_fd(self) -> int:
        return self._load_efd.fileno()

    def submit_store_task(
        self,
        keys: list[ObjectKey],
        objects: list[MemoryObj],
    ) -> L2TaskId:
        with self._lock:
            task_id = self._get_next_task_id()

        ok = True
        stored_keys: list[ObjectKey] = []
        stored_sizes: list[int] = []
        task_bytes = 0
        for key, obj in zip(keys, objects, strict=True):
            key_string = _object_key_to_string(key)
            ptr, size = _memoryview_ptr_and_size(obj)
            rc = self._store.put_from(key_string, ptr, size)
            logger.info("MooncakeStore SET key=%s bytes=%d rc=%d", key_string, size, rc)
            if rc != 0:
                ok = False
                break
            stored_keys.append(key)
            stored_sizes.append(size)

        notify_keys: list[ObjectKey] = []
        notify_sizes: list[int] = []
        with self._lock:
            if ok:
                for key, size in zip(stored_keys, stored_sizes, strict=True):
                    if key not in self._key_sizes:
                        self._key_sizes[key] = size
                        notify_keys.append(key)
                        notify_sizes.append(size)
                        task_bytes += size
                    else:
                        notify_keys.append(key)
                        notify_sizes.append(0)
            self._completed_stores[task_id] = L2StoreResult(ok, task_bytes)
            self._store_efd.notify()

        if notify_keys:
            self._notify_keys_stored(notify_keys, notify_sizes)
        return task_id

    def pop_completed_store_tasks(self) -> dict[L2TaskId, L2StoreResult]:
        with self._lock:
            completed = self._completed_stores
            self._completed_stores = {{}}
        return completed

    def submit_lookup_and_lock_task(self, keys: list[ObjectKey]) -> L2TaskId:
        bitmap = Bitmap(len(keys))
        found = 0
        external_index_hits = 0
        notify_keys: list[ObjectKey] = []
        notify_sizes: list[int] = []
        with self._lock:
            task_id = self._get_next_task_id()
            self._refresh_external_index()
            for i, key in enumerate(keys):
                key_string = _object_key_to_string(key)
                external_size = self._external_key_sizes.get(key_string)
                if key not in self._key_sizes and external_size is not None:
                    self._key_sizes[key] = external_size
                    notify_keys.append(key)
                    notify_sizes.append(external_size)
                    external_index_hits += 1
                if key in self._key_sizes:
                    bitmap.set(i)
                    self._locked_keys[key] = self._locked_keys.get(key, 0) + 1
                    found += 1
            self._completed_lookups[task_id] = bitmap
            self._lookup_efd.notify()
        if notify_keys:
            self._notify_keys_stored(notify_keys, notify_sizes)
        logger.info(
            "MooncakeStore EXISTS local_index_hits=%d external_index_hits=%d "
            "external_index_path=%s total=%d",
            found,
            external_index_hits,
            self._external_index_path or "",
            len(keys),
        )
        return task_id

    def query_lookup_and_lock_result(self, task_id: L2TaskId) -> Bitmap | None:
        with self._lock:
            return self._completed_lookups.pop(task_id, None)

    def submit_unlock(self, keys: list[ObjectKey]) -> None:
        with self._lock:
            for key in keys:
                count = self._locked_keys.get(key, 0)
                if count <= 1:
                    self._locked_keys.pop(key, None)
                else:
                    self._locked_keys[key] = count - 1

    def submit_load_task(
        self,
        keys: list[ObjectKey],
        objects: list[MemoryObj],
    ) -> L2TaskId:
        with self._lock:
            task_id = self._get_next_task_id()

        bitmap = Bitmap(len(keys))
        loaded_keys: list[ObjectKey] = []
        for i, (key, obj) in enumerate(zip(keys, objects, strict=True)):
            with self._lock:
                indexed_size = self._key_sizes.get(key)
            if indexed_size is None:
                continue
            key_string = _object_key_to_string(key)
            ptr, requested_size = _memoryview_ptr_and_size(obj)
            bytes_read = self._store.get_into(key_string, ptr, requested_size)
            logger.info(
                "MooncakeStore GET key=%s requested_bytes=%d indexed_bytes=%d read_bytes=%d",
                key_string,
                requested_size,
                indexed_size,
                bytes_read,
            )
            if bytes_read == requested_size == indexed_size:
                bitmap.set(i)
                loaded_keys.append(key)
            else:
                logger.warning(
                    "MooncakeStore GET rejected incomplete read key=%s "
                    "requested_bytes=%d indexed_bytes=%d read_bytes=%d",
                    key_string,
                    requested_size,
                    indexed_size,
                    bytes_read,
                )

        with self._lock:
            self._completed_loads[task_id] = bitmap
            self._load_efd.notify()
        if loaded_keys:
            self._notify_keys_accessed(loaded_keys)
        return task_id

    def query_load_result(self, task_id: L2TaskId) -> Bitmap | None:
        with self._lock:
            return self._completed_loads.pop(task_id, None)

    def delete(self, keys: list[ObjectKey]) -> None:
        deleted_keys: list[ObjectKey] = []
        deleted_sizes: list[int] = []
        for key in keys:
            key_string = _object_key_to_string(key)
            rc = self._store.remove(key_string, True)
            logger.info("MooncakeStore DELETE key=%s rc=%d", key_string, rc)
            if rc != 0:
                continue
            with self._lock:
                size = self._key_sizes.pop(key, None)
            if size is not None:
                deleted_keys.append(key)
                deleted_sizes.append(size)
        if deleted_keys:
            self._notify_keys_deleted(deleted_keys, deleted_sizes)

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
        self._store.close()
        self._store_efd.close()
        self._lookup_efd.close()
        self._load_efd.close()

    def report_status(self) -> dict[str, object]:
        return {{
            "is_healthy": not self._closed,
            "type": "mooncake_store_python",
            "known_keys": len(self._key_sizes),
            "external_index_path": self._external_index_path,
            "external_index_keys": len(self._external_key_sizes),
        }}

    def _get_next_task_id(self) -> L2TaskId:
        task_id = self._next_task_id
        self._next_task_id += 1
        return task_id

    def _refresh_external_index(self) -> None:
        if not self._external_index_path:
            return
        try:
            size = os.path.getsize(self._external_index_path)
        except OSError:
            return
        if size < self._external_index_offset:
            self._external_index_offset = 0
            self._external_key_sizes.clear()
        try:
            with open(self._external_index_path, "r", encoding="utf-8") as handle:
                handle.seek(self._external_index_offset)
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError:
                        logger.warning(
                            "Skipping invalid Mooncake external index line: %s", line[:200]
                        )
                        continue
                    key = record.get("key")
                    if not isinstance(key, str) or not key:
                        continue
                    try:
                        value_size = int(record.get("bytes", 0))
                    except (TypeError, ValueError):
                        value_size = 0
                    if value_size > 0:
                        self._external_key_sizes[key] = value_size
                self._external_index_offset = handle.tell()
        except OSError as exc:
            logger.warning(
                "Could not refresh Mooncake external index %s: %s",
                self._external_index_path,
                exc,
            )
{PYTHON_ADAPTER_END_MARKER}
'''

MOONCAKE_FACTORY_BRANCH = """    if _env_enabled("LMCACHE_MOONCAKE_PYTHON_ADAPTER", "1"):
        if not isinstance(config, MooncakeStoreL2AdapterConfig):
            raise ValueError(f"Expected MooncakeStoreL2AdapterConfig, got {type(config)}")
        adapter = MooncakePythonL2Adapter(config)
        logger.info("Created Mooncake Store Python L2 adapter")
        return adapter

"""

NATIVE_LOOKUP_HELPERS = f'''    {NATIVE_LOOKUP_MARKER}
    def _use_local_lookup_index(self) -> bool:
        """Avoid Mooncake native batch-exists crashes on missing keys."""
        return (
            self._type_name == "LMCacheMooncakeClient"
            and os.environ.get("LMCACHE_MOONCAKE_NATIVE_EXISTS", "0").lower()
            not in {{"1", "true", "yes", "on"}}
        )

    def _submit_local_lookup_and_lock_task(
        self,
        keys: list[ObjectKey],
    ) -> L2TaskId:
        bitmap = Bitmap(len(keys))
        with self._lock:
            task_id = self._get_next_task_id()
            for i, key in enumerate(keys):
                if key in self._key_sizes:
                    bitmap.set(i)
                    self._locked_keys[key] += 1
            self._completed_lookups[task_id] = bitmap
            self._lookup_efd.notify()
        return task_id
    {NATIVE_LOOKUP_END_MARKER}

'''


@dataclass(frozen=True)
class PatchResult:
    name: str
    path: Path | None
    changed: bool
    message: str


class PatchError(RuntimeError):
    """Raised when a required runtime patch cannot be applied or verified."""


def _site_package_roots(extra_roots: Iterable[Path] | None = None) -> list[Path]:
    roots: list[Path] = []

    def add(path: Path | str | None) -> None:
        if path is None:
            return
        root = Path(path).resolve()
        if root.is_dir() and root not in roots:
            roots.append(root)

    if extra_roots:
        for root in extra_roots:
            add(root)
        return roots

    for key in ("purelib", "platlib"):
        add(sysconfig.get_paths().get(key))
    try:
        for root in site.getsitepackages():
            add(root)
    except AttributeError:
        pass
    add(Path(sys.prefix) / "lib")
    return roots


def _package_dirs(package_name: str, roots: Iterable[Path] | None = None) -> list[Path]:
    found: list[Path] = []

    def add(path: Path) -> None:
        path = path.resolve()
        if path.is_dir() and path not in found:
            found.append(path)

    root_list = list(roots or [])
    if root_list:
        for root in root_list:
            add(root / package_name)
        return found

    spec = importlib.util.find_spec(package_name)
    if spec and spec.submodule_search_locations:
        for location in spec.submodule_search_locations:
            add(Path(location))

    for root in _site_package_roots():
        add(root / package_name)
    return found


def _write_text(path: Path, text: str, dry_run: bool) -> None:
    if dry_run:
        return
    backup = path.with_name(path.name + ".goldenexperience.bak")
    if not backup.exists():
        shutil.copy2(path, backup)
    path.write_text(text, encoding="utf-8")


def _require_anchor(text: str, anchor: str, path: Path) -> None:
    if anchor not in text:
        raise PatchError(f"Could not find expected anchor in {path}: {anchor!r}")


def _insert_after_once(text: str, anchor: str, insertion: str, path: Path) -> str:
    _require_anchor(text, anchor, path)
    return text.replace(anchor, anchor + insertion, 1)


def _insert_before_once(text: str, anchor: str, insertion: str, path: Path) -> str:
    _require_anchor(text, anchor, path)
    return text.replace(anchor, insertion + anchor, 1)


def _verify_strings(path: Path, text: str, required: Iterable[str]) -> None:
    missing = [value for value in required if value not in text]
    if missing:
        raise PatchError(f"{path} is missing required Mooncake patch strings: {missing}")


def ensure_mooncake_store_library(
    mooncake_dir: Path,
    *,
    check: bool = False,
    dry_run: bool = False,
) -> PatchResult:
    source = mooncake_dir / "store.so"
    target = mooncake_dir / "libmooncake_store.so"
    if target.exists() or target.is_symlink():
        return PatchResult(
            "mooncake-library",
            target,
            False,
            "libmooncake_store.so already exists",
        )
    if not source.exists():
        message = f"Mooncake store.so was not found in {mooncake_dir}"
        if check:
            raise PatchError(message)
        return PatchResult("mooncake-library", source, False, message)
    if check:
        raise PatchError(f"{target} is missing; run this script without --check")
    if not dry_run:
        try:
            target.symlink_to(source.name)
        except OSError:
            shutil.copy2(source, target)
    return PatchResult(
        "mooncake-library",
        target,
        not dry_run,
        "created libmooncake_store.so alias for store.so",
    )


def patch_mooncake_store_adapter(
    path: Path,
    *,
    check: bool = False,
    dry_run: bool = False,
) -> PatchResult:
    text = path.read_text(encoding="utf-8")
    if check:
        _verify_strings(path, text, MOONCAKE_ADAPTER_REQUIRED)
        return PatchResult("lmcache-mooncake-adapter", path, False, "patch verified")

    original = text
    if "import json" not in text:
        if "import ctypes\nimport os\nimport threading\n" in text:
            text = text.replace(
                "import ctypes\nimport os\nimport threading\n",
                "import ctypes\nimport json\nimport os\nimport threading\n",
                1,
            )
        elif "import ctypes" in text:
            text = _insert_after_once(text, "import ctypes\n", "import json\n", path)

    if "class MooncakePythonL2Adapter" in text and PYTHON_ADAPTER_VERSION_MARKER not in text:
        if PYTHON_ADAPTER_MARKER in text:
            start = text.index(PYTHON_ADAPTER_MARKER)
            end = text.index(PYTHON_ADAPTER_END_MARKER, start) + len(PYTHON_ADAPTER_END_MARKER)
        else:
            # Early patch versions had no markers. Replace the complete adapter section
            # rather than leaving an old class that happens to share the same name.
            start = text.index("def _env_enabled(")
            end = text.index("def _create_mooncake_store_l2_adapter(", start)
        text = text[:start] + MOONCAKE_PYTHON_ADAPTER_BLOCK + "\n" + text[end:]

    if "class MooncakePythonL2Adapter" not in text:
        typing_anchor = "from typing import (\n    TYPE_CHECKING,\n    cast,\n)\n"
        if "import ctypes" not in text:
            text = _insert_after_once(
                text,
                typing_anchor,
                "import ctypes\nimport json\nimport os\nimport threading\n\n",
                path,
            )

        base_import = "from lmcache.v1.distributed.l2_adapters.base import (\n"
        if "L2TaskId" not in text:
            text = _insert_after_once(text, base_import, "    L2TaskId,\n", path)

        factory_import = (
            "from lmcache.v1.distributed.l2_adapters.factory import (\n"
            "    register_l2_adapter_factory,\n"
            ")\n"
        )
        extra_imports = (
            "from lmcache.native_storage_ops import Bitmap\n"
            "from lmcache.v1.distributed.api import ObjectKey\n"
            "from lmcache.v1.distributed.internal_api import L2StoreResult\n"
            "from lmcache.v1.distributed.l2_adapters.native_connector_l2_adapter import (\n"
            "    _obj_to_memoryview,\n"
            "    _object_key_to_string,\n"
            ")\n"
            "from lmcache.v1.memory_management import MemoryObj\n"
            "from lmcache.v1.platform import create_event_notifier\n"
        )
        if "from lmcache.native_storage_ops import Bitmap" not in text:
            text = _insert_after_once(text, factory_import, extra_imports, path)

        text = _insert_before_once(
            text,
            "\ndef _create_mooncake_store_l2_adapter(\n",
            "\n\n" + MOONCAKE_PYTHON_ADAPTER_BLOCK,
            path,
        )

    if "Created Mooncake Store Python L2 adapter" not in text:
        text = _insert_before_once(
            text,
            "    try:\n        # First Party\n        from lmcache.lmcache_mooncake import (\n",
            MOONCAKE_FACTORY_BRANCH,
            path,
        )
    elif "class MooncakePythonL2Adapter" in text and "LMCACHE_MOONCAKE_PYTHON_ADAPTER" not in text:
        raise PatchError(f"{path} has partial Mooncake Python adapter patch")

    _verify_strings(path, text, MOONCAKE_ADAPTER_REQUIRED)
    if text == original:
        return PatchResult("lmcache-mooncake-adapter", path, False, "already patched")
    _write_text(path, text, dry_run)
    return PatchResult(
        "lmcache-mooncake-adapter",
        path,
        not dry_run,
        "patched Mooncake Store adapter to use Python SET/GET path",
    )


def patch_native_connector_adapter(
    path: Path,
    *,
    check: bool = False,
    dry_run: bool = False,
) -> PatchResult:
    text = path.read_text(encoding="utf-8")
    if check:
        _verify_strings(path, text, NATIVE_ADAPTER_REQUIRED)
        return PatchResult("lmcache-native-lookup", path, False, "patch verified")

    original = text
    if "import os" not in text:
        text = _insert_after_once(text, "from typing import Any\n", "import os\n", path)

    if "self._use_local_lookup_index()" not in text:
        lookup_anchor = (
            "    def submit_lookup_and_lock_task(\n"
            "        self,\n"
            "        keys: list[ObjectKey],\n"
            "    ) -> L2TaskId:\n"
            "        key_strings = [_object_key_to_string(k) for k in keys]\n"
        )
        lookup_replacement = (
            "    def submit_lookup_and_lock_task(\n"
            "        self,\n"
            "        keys: list[ObjectKey],\n"
            "    ) -> L2TaskId:\n"
            "        if self._use_local_lookup_index():\n"
            "            return self._submit_local_lookup_and_lock_task(keys)\n\n"
            "        key_strings = [_object_key_to_string(k) for k in keys]\n"
        )
        _require_anchor(text, lookup_anchor, path)
        text = text.replace(lookup_anchor, lookup_replacement, 1)

    if "def _use_local_lookup_index" not in text:
        text = _insert_before_once(
            text,
            "    def query_lookup_and_lock_result(self, task_id: L2TaskId) -> Bitmap | None:\n",
            NATIVE_LOOKUP_HELPERS,
            path,
        )

    _verify_strings(path, text, NATIVE_ADAPTER_REQUIRED)
    if text == original:
        return PatchResult("lmcache-native-lookup", path, False, "already patched")
    _write_text(path, text, dry_run)
    return PatchResult(
        "lmcache-native-lookup",
        path,
        not dry_run,
        "patched native Mooncake lookup to use local key index by default",
    )


def patch_runtime(
    *,
    site_packages: Iterable[Path] | None = None,
    check: bool = False,
    dry_run: bool = False,
) -> list[PatchResult]:
    roots = _site_package_roots(site_packages)
    results: list[PatchResult] = []

    mooncake_dirs = _package_dirs("mooncake", roots if site_packages else None)
    if mooncake_dirs:
        for mooncake_dir in mooncake_dirs:
            results.append(
                ensure_mooncake_store_library(mooncake_dir, check=check, dry_run=dry_run)
            )
    else:
        message = "Python package 'mooncake' was not found; library alias was not patched"
        if check:
            raise PatchError(message)
        results.append(PatchResult("mooncake-library", None, False, message))

    lmcache_dirs = _package_dirs("lmcache", roots if site_packages else None)
    if not lmcache_dirs:
        message = "Python package 'lmcache' was not found; LMCache adapters were not patched"
        if check:
            raise PatchError(message)
        results.append(PatchResult("lmcache", None, False, message))
        return results

    for lmcache_dir in lmcache_dirs:
        adapter_dir = lmcache_dir / "v1" / "distributed" / "l2_adapters"
        mooncake_adapter = adapter_dir / "mooncake_store_l2_adapter.py"
        native_adapter = adapter_dir / "native_connector_l2_adapter.py"
        if mooncake_adapter.exists():
            results.append(
                patch_mooncake_store_adapter(mooncake_adapter, check=check, dry_run=dry_run)
            )
        elif check:
            raise PatchError(f"Missing LMCache Mooncake adapter file: {mooncake_adapter}")
        else:
            results.append(
                PatchResult(
                    "lmcache-mooncake-adapter",
                    mooncake_adapter,
                    False,
                    "adapter file not found",
                )
            )

        if native_adapter.exists():
            results.append(
                patch_native_connector_adapter(native_adapter, check=check, dry_run=dry_run)
            )
        elif check:
            raise PatchError(f"Missing LMCache native connector adapter file: {native_adapter}")
        else:
            results.append(
                PatchResult(
                    "lmcache-native-lookup",
                    native_adapter,
                    False,
                    "adapter file not found",
                )
            )

    return results


def _print_results(results: Iterable[PatchResult]) -> None:
    for result in results:
        state = "changed" if result.changed else "ok"
        path = "" if result.path is None else f" {result.path}"
        print(f"[{state}] {result.name}:{path} - {result.message}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--site-packages",
        action="append",
        type=Path,
        default=None,
        help="Patch this site-packages root instead of auto-discovering the active env.",
    )
    parser.add_argument("--check", action="store_true", help="Verify patches without writing.")
    parser.add_argument("--dry-run", action="store_true", help="Show planned changes only.")
    args = parser.parse_args(argv)

    try:
        results = patch_runtime(
            site_packages=args.site_packages,
            check=args.check,
            dry_run=args.dry_run,
        )
    except PatchError as exc:
        print(f"Mooncake runtime patch failed: {exc}", file=sys.stderr)
        return 1
    _print_results(results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
