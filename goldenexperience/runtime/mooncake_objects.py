"""Strict Mooncake object I/O for cached-KV materialization."""

from __future__ import annotations

import ctypes
import json
import os
import time
from collections.abc import Callable, Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class MooncakeObjectError(RuntimeError):
    """Raised when Mooncake cannot prove a complete object operation."""


@dataclass(frozen=True)
class ExactObjectRead:
    key: str
    data: bytearray
    expected_bytes: int
    remote_bytes: int
    read_bytes: int


@dataclass(frozen=True)
class ExactObjectWrite:
    key: str
    bytes: int
    put_rc: int
    remote_bytes: int


class ExactMooncakeStore:
    """Context-managed Mooncake client with all-or-nothing batch semantics."""

    def __init__(
        self,
        setup_config: Mapping[str, Any],
        *,
        store_factory: Callable[[], Any] | None = None,
    ) -> None:
        self.setup_config = {str(key): str(value) for key, value in setup_config.items()}
        self._store_factory = store_factory or _default_store_factory
        self._store: Any | None = None

    def __enter__(self) -> ExactMooncakeStore:
        if self._store is not None:
            raise MooncakeObjectError("Mooncake store is already open")
        if metadata_server := self.setup_config.get("metadata_server"):
            os.environ["MOONCAKE_TE_META_DATA_SERVER"] = metadata_server
        store = self._store_factory()
        rc = int(store.setup(dict(self.setup_config)))
        if rc != 0:
            with suppress(Exception):
                store.close()
            raise MooncakeObjectError(f"Mooncake setup failed with rc={rc}")
        self._store = store
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        del exc_type, exc, traceback
        store = self._store
        self._store = None
        if store is not None:
            with suppress(Exception):
                store.close()

    def read_many_exact(
        self,
        keys: Sequence[str],
        expected_sizes: Sequence[int],
    ) -> list[ExactObjectRead]:
        store = self._require_store()
        normalized_keys, normalized_sizes = _validate_batch(keys, expected_sizes)
        remote_sizes = [int(store.get_size(key)) for key in normalized_keys]
        for key, expected, remote in zip(
            normalized_keys, normalized_sizes, remote_sizes, strict=True
        ):
            if remote != expected:
                raise MooncakeObjectError(
                    f"source object size mismatch for {key}: expected={expected}, remote={remote}"
                )

        buffers = [bytearray(size) for size in normalized_sizes]
        pointers = [_buffer_pointer(buffer) for buffer in buffers]
        batch_get = getattr(store, "batch_get_into", None)
        if callable(batch_get):
            raw_results = batch_get(normalized_keys, pointers, normalized_sizes)
            results = _integer_results(raw_results, len(normalized_keys), "batch_get_into")
        else:
            results = [
                int(store.get_into(key, pointer, size))
                for key, pointer, size in zip(
                    normalized_keys, pointers, normalized_sizes, strict=True
                )
            ]
        for key, expected, remote, read in zip(
            normalized_keys,
            normalized_sizes,
            remote_sizes,
            results,
            strict=True,
        ):
            if read != expected or remote != expected:
                raise MooncakeObjectError(
                    f"incomplete source read for {key}: expected={expected}, "
                    f"remote={remote}, read={read}"
                )
        final_sizes = [int(store.get_size(key)) for key in normalized_keys]
        if final_sizes != remote_sizes:
            raise MooncakeObjectError("source object sizes changed during read")
        return [
            ExactObjectRead(
                key=key,
                data=data,
                expected_bytes=expected,
                remote_bytes=remote,
                read_bytes=read,
            )
            for key, data, expected, remote, read in zip(
                normalized_keys,
                buffers,
                normalized_sizes,
                remote_sizes,
                results,
                strict=True,
            )
        ]

    def write_many_exact(
        self,
        keys: Sequence[str],
        payloads: Sequence[bytes | bytearray | memoryview],
    ) -> list[ExactObjectWrite]:
        store = self._require_store()
        if not keys or len(keys) != len(payloads):
            raise MooncakeObjectError("target keys and payloads must be non-empty and aligned")
        normalized_keys = [str(key) for key in keys]
        if any(not key for key in normalized_keys) or len(set(normalized_keys)) != len(
            normalized_keys
        ):
            raise MooncakeObjectError("target keys must be non-empty and unique")
        buffers = [bytearray(payload) for payload in payloads]
        sizes = [len(buffer) for buffer in buffers]
        if any(size <= 0 for size in sizes):
            raise MooncakeObjectError("target payloads must be non-empty")

        existence = self._batch_exists(normalized_keys)
        if any(result != 0 for result in existence):
            raise MooncakeObjectError(
                f"target objects must all be absent before materialization: {existence}"
            )
        pointers = [_buffer_pointer(buffer) for buffer in buffers]
        batch_put = getattr(store, "batch_put_from", None)
        try:
            if callable(batch_put):
                raw_results = batch_put(normalized_keys, pointers, sizes)
                results = _integer_results(raw_results, len(normalized_keys), "batch_put_from")
            else:
                results = [
                    int(store.put_from(key, pointer, size))
                    for key, pointer, size in zip(normalized_keys, pointers, sizes, strict=True)
                ]
            if any(result != 0 for result in results):
                raise MooncakeObjectError(f"Mooncake target put failed: {results}")
            remote_sizes = [int(store.get_size(key)) for key in normalized_keys]
            if remote_sizes != sizes:
                raise MooncakeObjectError(
                    "target object size verification failed: "
                    f"expected={sizes}, remote={remote_sizes}"
                )
        except Exception:
            self.rollback(normalized_keys)
            raise
        return [
            ExactObjectWrite(key=key, bytes=size, put_rc=result, remote_bytes=remote)
            for key, size, result, remote in zip(
                normalized_keys, sizes, results, remote_sizes, strict=True
            )
        ]

    def rollback(self, keys: Sequence[str]) -> dict[str, int]:
        store = self._require_store()
        results: dict[str, int] = {}
        for key in keys:
            try:
                results[str(key)] = int(store.remove(str(key), True))
            except Exception:  # noqa: BLE001 - best-effort removal after a failed batch.
                results[str(key)] = -1
        return results

    def _batch_exists(self, keys: list[str]) -> list[int]:
        store = self._require_store()
        batch_exists = getattr(store, "batch_is_exist", None)
        if callable(batch_exists):
            return _integer_results(batch_exists(keys), len(keys), "batch_is_exist")
        return [int(store.is_exist(key)) for key in keys]

    def _require_store(self) -> Any:
        if self._store is None:
            raise MooncakeObjectError("Mooncake store is not open")
        return self._store


def publish_external_index(path: str | Path, records: Sequence[Mapping[str, Any]]) -> None:
    """Atomically publish a complete batch to the LMCache external index."""

    import fcntl

    if not records:
        raise MooncakeObjectError("external index batch must be non-empty")
    lines: list[str] = []
    for record in records:
        if not isinstance(record.get("key"), str) or not record["key"]:
            raise MooncakeObjectError("external index record key is required")
        try:
            size = int(record.get("bytes", 0))
        except (TypeError, ValueError) as exc:
            raise MooncakeObjectError("external index record bytes must be an integer") from exc
        if size <= 0:
            raise MooncakeObjectError("external index record bytes must be positive")
        lines.append(json.dumps(dict(record), sort_keys=True, separators=(",", ":")) + "\n")

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    lock_path = output.with_name(output.name + ".lock")
    temp_path = output.with_name(f".{output.name}.{os.getpid()}.{time.time_ns()}.tmp")
    with lock_path.open("a+b") as lock_handle:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
        existing = output.read_bytes() if output.exists() else b""
        if existing and not existing.endswith(b"\n"):
            raise MooncakeObjectError("existing external index ends with a partial record")
        try:
            with temp_path.open("xb") as temp_handle:
                temp_handle.write(existing)
                temp_handle.write("".join(lines).encode("utf-8"))
                temp_handle.flush()
                os.fsync(temp_handle.fileno())
            os.replace(temp_path, output)
            # The index is already visible after replace; a later fsync error must not
            # trigger target rollback while LMCache can observe the published keys.
            with suppress(OSError):
                directory_fd = os.open(output.parent, os.O_RDONLY)
                try:
                    os.fsync(directory_fd)
                finally:
                    os.close(directory_fd)
        finally:
            temp_path.unlink(missing_ok=True)
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)


def _validate_batch(
    keys: Sequence[str],
    expected_sizes: Sequence[int],
) -> tuple[list[str], list[int]]:
    if not keys or len(keys) != len(expected_sizes):
        raise MooncakeObjectError("source keys and expected sizes must be non-empty and aligned")
    normalized_keys = [str(key) for key in keys]
    normalized_sizes = [int(size) for size in expected_sizes]
    if any(not key for key in normalized_keys) or len(set(normalized_keys)) != len(normalized_keys):
        raise MooncakeObjectError("source keys must be non-empty and unique")
    if any(size <= 0 for size in normalized_sizes):
        raise MooncakeObjectError("source expected sizes must be positive")
    return normalized_keys, normalized_sizes


def _buffer_pointer(buffer: bytearray) -> int:
    return ctypes.addressof(ctypes.c_char.from_buffer(buffer))


def _integer_results(raw: Any, count: int, operation: str) -> list[int]:
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes, bytearray)):
        raise MooncakeObjectError(f"{operation} returned a non-sequence result")
    if len(raw) != count:
        raise MooncakeObjectError(f"{operation} returned {len(raw)} results for {count} objects")
    try:
        return [int(item) for item in raw]
    except (TypeError, ValueError) as exc:
        raise MooncakeObjectError(f"{operation} returned a non-integer result") from exc


def _default_store_factory() -> Any:
    from mooncake.store import MooncakeDistributedStore

    return MooncakeDistributedStore()
