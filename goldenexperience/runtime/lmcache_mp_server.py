"""Managed LMCache MP server used by the isolated publication audit."""

from __future__ import annotations

import importlib.metadata
import math
import os
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, TextIO

LMCACHE_MP_AUDIT_SERVER_SCHEMA = "goldenexperience.lmcache_mp_audit_server.v1"
LMCACHE_MP_AUDIT_CHUNK_SIZE = 128
LMCACHE_MP_AUDIT_TRANSFER_MODE = "non_gpu"
LMCACHE_MP_AUDIT_EVICTION_POLICY = "LRU"
LMCACHE_MP_AUDIT_EXPECTED_VERSION = "0.4.6"


class LMCacheMPServerError(RuntimeError):
    """Raised when the isolated LMCache MP server cannot be verified."""


@dataclass(frozen=True)
class LMCacheMPServerConfig:
    host: str = "127.0.0.1"
    port: int = 0
    chunk_size: int = LMCACHE_MP_AUDIT_CHUNK_SIZE
    l1_size_gb: float = 4.0
    l1_init_size_gb: int = 1
    max_cpu_workers: int = 1
    startup_timeout_s: float = 30.0
    shutdown_timeout_s: float = 20.0
    schema_version: str = LMCACHE_MP_AUDIT_SERVER_SCHEMA

    def validate(self) -> list[str]:
        errors: list[str] = []
        if self.schema_version != LMCACHE_MP_AUDIT_SERVER_SCHEMA:
            errors.append("unsupported LMCache MP audit server schema")
        if self.host != "127.0.0.1":
            errors.append("LMCache MP audit server must bind IPv4 loopback")
        if type(self.port) is not int or not 0 <= self.port <= 65535:
            errors.append("LMCache MP audit server port is invalid")
        if self.chunk_size != LMCACHE_MP_AUDIT_CHUNK_SIZE:
            errors.append("LMCache MP audit chunk size must remain 128")
        if not _finite_positive(self.l1_size_gb):
            errors.append("LMCache MP audit L1 size must be finite and positive")
        if (
            type(self.l1_init_size_gb) is not int
            or self.l1_init_size_gb <= 0
            or self.l1_init_size_gb > self.l1_size_gb
        ):
            errors.append("LMCache MP audit initial L1 size is invalid")
        if type(self.max_cpu_workers) is not int or self.max_cpu_workers <= 0:
            errors.append("LMCache MP audit worker count must be positive")
        if not _finite_positive(self.startup_timeout_s) or not _finite_positive(
            self.shutdown_timeout_s
        ):
            errors.append("LMCache MP audit process timeouts must be finite and positive")
        return errors

    def publication_parameters(self) -> dict[str, Any]:
        payload = asdict(self)
        payload.update(
            {
                "port": self.port if self.port else "automatic_loopback_port",
                "transfer_mode": LMCACHE_MP_AUDIT_TRANSFER_MODE,
                "eviction_policy": LMCACHE_MP_AUDIT_EVICTION_POLICY,
                "l2_adapters": [],
                "filesystem_backing_configured": False,
            }
        )
        return payload


class LMCacheMPServerProcess:
    """Own one bounded, memory-only LMCache MP subprocess."""

    def __init__(self, config: LMCacheMPServerConfig) -> None:
        errors = config.validate()
        if errors:
            raise LMCacheMPServerError("; ".join(errors))
        self.config = config
        self._process: subprocess.Popen[str] | None = None
        self._port: int | None = None
        self._runtime_directory: tempfile.TemporaryDirectory[str] | None = None
        self._log_thread: threading.Thread | None = None
        self._log_lines: deque[str] = deque(maxlen=200)
        self._last_backing_files_remaining = 0

    @property
    def running(self) -> bool:
        return self._process is not None and self._process.poll() is None

    @property
    def port(self) -> int:
        if self._port is None:
            raise LMCacheMPServerError("LMCache MP audit server has not started")
        return self._port

    @property
    def server_url(self) -> str:
        return f"tcp://{self.config.host}:{self.port}"

    @property
    def log_tail(self) -> tuple[str, ...]:
        return tuple(self._log_lines)

    @property
    def backing_files_remaining(self) -> int:
        if self._runtime_directory is None:
            return self._last_backing_files_remaining
        root = Path(self._runtime_directory.name)
        return sum(1 for path in root.rglob("*") if path.is_file() or path.is_symlink())

    def command(self, port: int) -> tuple[str, ...]:
        if type(port) is not int or not 0 < port <= 65535:
            raise LMCacheMPServerError("LMCache MP audit launch port is invalid")
        return (
            sys.executable,
            "-m",
            "lmcache.v1.multiprocess.server",
            "--host",
            self.config.host,
            "--port",
            str(port),
            "--chunk-size",
            str(self.config.chunk_size),
            "--max-workers",
            "1",
            "--max-gpu-workers",
            "1",
            "--max-cpu-workers",
            str(self.config.max_cpu_workers),
            "--transfer-mode",
            LMCACHE_MP_AUDIT_TRANSFER_MODE,
            "--l1-size-gb",
            str(self.config.l1_size_gb),
            "--l1-use-lazy",
            "--l1-init-size-gb",
            str(self.config.l1_init_size_gb),
            "--eviction-policy",
            LMCACHE_MP_AUDIT_EVICTION_POLICY,
        )

    def start(self) -> LMCacheMPServerProcess:
        if self._process is not None or self._runtime_directory is not None:
            raise LMCacheMPServerError("LMCache MP audit server was already started")
        try:
            lmcache_version = importlib.metadata.version("lmcache")
        except importlib.metadata.PackageNotFoundError as exc:
            raise LMCacheMPServerError("LMCache MP audit runtime is not installed") from exc
        if lmcache_version != LMCACHE_MP_AUDIT_EXPECTED_VERSION:
            raise LMCacheMPServerError("LMCache MP audit server requires LMCache 0.4.6")
        self._port = self.config.port or _available_loopback_port(self.config.host)
        self._runtime_directory = tempfile.TemporaryDirectory(prefix="golden-lmcache-mp-audit-")
        environment = os.environ.copy()
        environment["PYTHONUNBUFFERED"] = "1"
        try:
            self._process = subprocess.Popen(
                self.command(self._port),
                cwd=self._runtime_directory.name,
                env=environment,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                close_fds=True,
            )
            if self._process.stdout is None:
                raise LMCacheMPServerError("LMCache MP audit server log pipe is unavailable")
            self._log_thread = threading.Thread(
                target=self._drain_logs,
                args=(self._process.stdout,),
                name="golden-lmcache-mp-log",
                daemon=True,
            )
            self._log_thread.start()
            self._wait_until_ready()
            self.assert_no_backing_files()
            return self
        except Exception:
            self._stop(suppress_errors=True)
            raise

    def assert_no_backing_files(self) -> None:
        count = self.backing_files_remaining
        if count:
            raise LMCacheMPServerError(
                f"LMCache MP audit server created {count} filesystem backing files"
            )

    def stop(self) -> None:
        self._stop(suppress_errors=False)

    def __enter__(self) -> LMCacheMPServerProcess:
        return self.start()

    def __exit__(self, exc_type: object, *_args: object) -> None:
        self._stop(suppress_errors=exc_type is not None)

    def _wait_until_ready(self) -> None:
        assert self._process is not None
        deadline = time.monotonic() + self.config.startup_timeout_s
        last_error: Exception | None = None
        while time.monotonic() < deadline:
            return_code = self._process.poll()
            if return_code is not None:
                raise LMCacheMPServerError(
                    f"LMCache MP audit server exited with code {return_code}: "
                    + " | ".join(self.log_tail[-20:])
                )
            remaining = deadline - time.monotonic()
            try:
                observed_chunk_size = _probe_chunk_size(
                    self.server_url,
                    timeout_s=min(1.0, max(0.05, remaining)),
                )
            except Exception as exc:
                last_error = exc
                time.sleep(min(0.05, max(0.0, deadline - time.monotonic())))
                continue
            if observed_chunk_size != self.config.chunk_size:
                raise LMCacheMPServerError("LMCache MP audit server returned another chunk size")
            return
        detail = f": {last_error!r}" if last_error is not None else ""
        raise LMCacheMPServerError("LMCache MP audit server startup timed out" + detail)

    def _drain_logs(self, stream: TextIO) -> None:
        try:
            for line in stream:
                self._log_lines.append(line.rstrip())
        finally:
            stream.close()

    def _stop(self, *, suppress_errors: bool) -> None:
        errors: list[Exception] = []
        process = self._process
        initial_return_code = process.poll() if process is not None else None
        if process is not None and initial_return_code is None:
            try:
                process.send_signal(signal.SIGINT)
                process.wait(timeout=self.config.shutdown_timeout_s)
            except subprocess.TimeoutExpired:
                errors.append(
                    LMCacheMPServerError("LMCache MP audit server did not shut down gracefully")
                )
                process.terminate()
                try:
                    process.wait(timeout=5.0)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5.0)
            except Exception as exc:
                errors.append(exc)
        elif process is not None and initial_return_code != 0:
            errors.append(
                LMCacheMPServerError(
                    f"LMCache MP audit server exited unexpectedly with code {initial_return_code}"
                )
            )
        if self._log_thread is not None:
            self._log_thread.join(timeout=5.0)
            if self._log_thread.is_alive():
                errors.append(LMCacheMPServerError("LMCache MP audit log thread did not stop"))
        if process is not None and process.stdout is not None:
            process.stdout.close()
        self._last_backing_files_remaining = self.backing_files_remaining
        if self._last_backing_files_remaining:
            errors.append(
                LMCacheMPServerError("LMCache MP audit server left filesystem backing files")
            )
        if self._runtime_directory is not None:
            self._runtime_directory.cleanup()
        self._process = None
        self._runtime_directory = None
        self._log_thread = None
        if errors and not suppress_errors:
            raise LMCacheMPServerError("; ".join(str(error) for error in errors))


def _available_loopback_port(host: str) -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind((host, 0))
        return int(listener.getsockname()[1])


def _probe_chunk_size(server_url: str, *, timeout_s: float) -> int:
    import zmq
    from lmcache.integration.vllm.vllm_multi_process_adapter import (  # type: ignore[import-untyped]
        get_lmcache_chunk_size,
    )
    from lmcache.v1.multiprocess.mq import (  # type: ignore[import-untyped]
        MessageQueueClient,
    )

    client = MessageQueueClient(server_url, zmq.Context.instance())
    try:
        return int(get_lmcache_chunk_size(client, timeout=timeout_s))
    finally:
        client.close()


def _finite_positive(value: Any) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(value)
        and value > 0
    )
