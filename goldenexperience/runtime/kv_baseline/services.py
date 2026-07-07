"""Process and readiness helpers for KV baseline services."""

from __future__ import annotations

import importlib.util
import json
import os
import shutil
import signal
import site
import socket
import subprocess
import sys
import sysconfig
import time
from pathlib import Path
from typing import Any

from goldenexperience.runtime.kv_baseline.config import BaselineConfig, REPO_ROOT


def ensure_command(command_name: str) -> None:
    if shutil.which(command_name) is None:
        raise FileNotFoundError(f"Required command {command_name!r} was not found in PATH.")


def _lmcache_mooncake_extension_found() -> bool:
    lmcache_spec = importlib.util.find_spec("lmcache")
    if lmcache_spec is None or lmcache_spec.submodule_search_locations is None:
        return False
    return any(
        candidate.exists()
        for location in lmcache_spec.submodule_search_locations
        for candidate in Path(location).glob("lmcache_mooncake*")
    )


def _prepend_library_paths(env: dict[str, str]) -> None:
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
    add(Path("/usr/local/cuda/lib64"))
    add(Path("/usr/local/nvidia/lib64"))

    if paths:
        existing = env.get("LD_LIBRARY_PATH")
        env["LD_LIBRARY_PATH"] = ":".join(paths + ([existing] if existing else []))


def validate_runtime_requirements(config: BaselineConfig) -> None:
    if not config.use_mooncake_store:
        return
    if _lmcache_mooncake_extension_found():
        return
    raise RuntimeError(
        "GE_LMCACHE_MP_L2_ADAPTER_TYPE=mooncake_store requires the LMCache Mooncake "
        "C++ extension, but Python module 'lmcache.lmcache_mooncake' is missing. "
        "Reinstall LMCache from source with Mooncake support, for example: "
        "MOONCAKE_INCLUDE_DIR=/path/to/mooncake-store/include BUILD_MOONCAKE=1 "
        "python3 -m pip install -e /path/to/LMCache. "
        "For a non-Mooncake filesystem L2 run, set GE_LMCACHE_MP_L2_ADAPTER_TYPE=fs."
    )


def tail_lines(path: Path, limit: int = 120) -> str:
    if not path.exists():
        return ""
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    return "\n".join(lines[-limit:])


def wait_for_tcp_port(host: str, port: int, timeout: float) -> bool:
    if host.startswith("tcp://"):
        host = host[len("tcp://") :]
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with socket.create_connection((host, port), timeout=2):
                return True
        except OSError:
            time.sleep(1)
    return False


def _metadata_path(config: BaselineConfig) -> Path:
    return config.run_dir / "metadata.json"


def record_service_pid(config: BaselineConfig, service_name: str, pid: int, log_path: Path) -> None:
    path = _metadata_path(config)
    metadata: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    services = metadata.setdefault("services", {})
    service = services.setdefault(service_name, {})
    service["pid"] = pid
    service["log_path"] = str(log_path)
    path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")


class ProcessGroup:
    """Track and clean up external services started by a baseline run."""

    def __init__(self, config: BaselineConfig) -> None:
        self.config = config
        self.server: subprocess.Popen[bytes] | None = None
        self.lmcache_mp: subprocess.Popen[bytes] | None = None
        self.mooncake_master: subprocess.Popen[bytes] | None = None
        self.mooncake_metadata: subprocess.Popen[bytes] | None = None

    def _open_log(self, path: Path) -> Any:
        path.parent.mkdir(parents=True, exist_ok=True)
        return path.open("ab")

    def _start(
        self,
        command: list[str],
        log_path: Path,
        env: dict[str, str] | None = None,
    ) -> subprocess.Popen[bytes]:
        log_handle = self._open_log(log_path)
        proc = subprocess.Popen(
            command,
            cwd=REPO_ROOT,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            env=self._child_env(env),
            start_new_session=True,
        )
        log_handle.close()
        return proc

    def _child_env(self, env: dict[str, str] | None = None) -> dict[str, str]:
        child_env = dict(os.environ if env is None else env)
        _prepend_library_paths(child_env)
        return child_env

    def start_mooncake_services(self) -> None:
        if not self.config.use_mooncake_store:
            return

        self._start_mooncake_master_embedded()
        embedded_ready = False
        for _ in range(15):
            master_ready = wait_for_tcp_port(
                self.config.mooncake_master_host,
                self.config.mooncake_master_port,
                1,
            )
            metadata_ready = wait_for_tcp_port(
                self.config.mooncake_metadata_host,
                self.config.mooncake_metadata_port,
                1,
            )
            if master_ready and metadata_ready:
                embedded_ready = True
                break
            if self.mooncake_master and self.mooncake_master.poll() is not None:
                break
            time.sleep(1)

        if embedded_ready:
            self.mooncake_metadata = None
            print("Mooncake embedded metadata server is ready")
            return

        if self.mooncake_master and self.mooncake_master.poll() is None:
            print("Mooncake embedded metadata server was not ready quickly; continuing to wait")
            self.wait_for_mooncake_ready()
            return

        print("Mooncake master did not accept embedded metadata flags; falling back")
        print(tail_lines(self.config.log_dir / "mooncake_master.log", 40))
        self.mooncake_master = None
        self._start_mooncake_metadata_server()
        self._start_mooncake_master_plain()

    def _mooncake_env(self) -> dict[str, str]:
        env = os.environ.copy()
        env["MOONCAKE_TE_META_DATA_SERVER"] = self.config.mooncake_metadata_server
        return env

    def _start_mooncake_master_embedded(self) -> None:
        ensure_command(self.config.mooncake_master_bin)
        log_path = self.config.log_dir / "mooncake_master.log"
        command = [
            self.config.mooncake_master_bin,
            "--port",
            str(self.config.mooncake_master_port),
            "--enable_http_metadata_server=1",
            "--http_metadata_server_host",
            self.config.mooncake_metadata_bind_host,
            "--http_metadata_server_port",
            str(self.config.mooncake_metadata_port),
        ]
        if extra := os.environ.get("GE_MOONCAKE_MASTER_EXTRA_ARGS"):
            command.extend(extra.split())
        print(f"Starting Mooncake master with embedded metadata; log: {log_path}")
        self.mooncake_master = self._start(command, log_path, self._mooncake_env())
        (self.config.run_dir / "mooncake_master.pid").write_text(
            f"{self.mooncake_master.pid}\n", encoding="utf-8"
        )
        record_service_pid(
            self.config, "mooncake_master", self.mooncake_master.pid, log_path
        )

    def _start_mooncake_master_plain(self) -> None:
        ensure_command(self.config.mooncake_master_bin)
        log_path = self.config.log_dir / "mooncake_master.log"
        command = [
            self.config.mooncake_master_bin,
            "--port",
            str(self.config.mooncake_master_port),
        ]
        if extra := os.environ.get("GE_MOONCAKE_MASTER_EXTRA_ARGS"):
            command.extend(extra.split())
        print(f"Starting Mooncake master; log: {log_path}")
        self.mooncake_master = self._start(command, log_path, self._mooncake_env())
        (self.config.run_dir / "mooncake_master.pid").write_text(
            f"{self.mooncake_master.pid}\n", encoding="utf-8"
        )
        record_service_pid(
            self.config, "mooncake_master", self.mooncake_master.pid, log_path
        )

    def _start_mooncake_metadata_server(self) -> None:
        ensure_command(self.config.mooncake_http_metadata_server_bin)
        log_path = self.config.log_dir / "mooncake_metadata_server.log"
        command = [
            self.config.mooncake_http_metadata_server_bin,
            "--host",
            self.config.mooncake_metadata_bind_host,
            "--port",
            str(self.config.mooncake_metadata_port),
        ]
        if extra := os.environ.get("GE_MOONCAKE_METADATA_EXTRA_ARGS"):
            command.extend(extra.split())
        print(
            "Starting Mooncake HTTP metadata server on "
            f"{self.config.mooncake_metadata_bind_host}:{self.config.mooncake_metadata_port}; "
            f"log: {log_path}"
        )
        self.mooncake_metadata = self._start(command, log_path)
        (self.config.run_dir / "mooncake_metadata.pid").write_text(
            f"{self.mooncake_metadata.pid}\n", encoding="utf-8"
        )
        record_service_pid(
            self.config, "mooncake_metadata", self.mooncake_metadata.pid, log_path
        )

    def wait_for_mooncake_ready(self) -> None:
        if not self.config.use_mooncake_store:
            return
        master_log = self.config.log_dir / "mooncake_master.log"
        metadata_log = self.config.log_dir / "mooncake_metadata_server.log"
        if not wait_for_tcp_port(
            self.config.mooncake_master_host,
            self.config.mooncake_master_port,
            self.config.start_timeout,
        ):
            raise RuntimeError(
                "Timed out waiting for Mooncake master; last log lines:\n"
                + tail_lines(master_log)
            )
        if self.mooncake_master and self.mooncake_master.poll() is not None:
            raise RuntimeError(
                "Mooncake master exited after port became reachable; last log lines:\n"
                + tail_lines(master_log)
            )
        if not wait_for_tcp_port(
            self.config.mooncake_metadata_host,
            self.config.mooncake_metadata_port,
            self.config.start_timeout,
        ):
            raise RuntimeError(
                "Timed out waiting for Mooncake metadata server; last log lines:\n"
                + tail_lines(metadata_log)
                + "\n"
                + tail_lines(master_log)
            )
        if self.mooncake_metadata and self.mooncake_metadata.poll() is not None:
            raise RuntimeError(
                "Mooncake metadata server exited after port became reachable; last log lines:\n"
                + tail_lines(metadata_log)
            )
        print(
            "Mooncake is ready: "
            f"master={self.config.mooncake_master_addr}; "
            f"metadata={self.config.mooncake_metadata_server}"
        )

    def start_lmcache_mp_server(self) -> None:
        if self.config.kv_backend != "mp":
            return
        ensure_command(self.config.lmcache_bin)
        log_path = self.config.log_dir / "lmcache_mp_server.log"
        command = [
            self.config.lmcache_bin,
            "server",
            "--host",
            self.config.lmcache_mp_bind_host,
            "--port",
            str(self.config.lmcache_mp_port),
            "--http-host",
            self.config.lmcache_mp_http_host,
            "--http-port",
            str(self.config.lmcache_mp_http_port),
            "--prometheus-port",
            str(self.config.lmcache_mp_prometheus_port),
            "--chunk-size",
            str(self.config.chunk_size),
            "--hash-algorithm",
            self.config.hash_algorithm,
            "--l1-size-gb",
            str(self.config.lmcache_mp_l1_gb),
            "--l1-init-size-gb",
            str(self.config.lmcache_mp_l1_init_gb),
            "--eviction-policy",
            self.config.lmcache_mp_eviction_policy,
            "--l2-store-policy",
            self.config.lmcache_mp_l2_store_policy,
            "--l2-adapter",
            self.config.l2_adapter_json(),
        ]
        if self.config.lmcache_mp_lookup_hash_log:
            command.extend(
                ["--lookup-hash-log-dir", str(self.config.lmcache_mp_lookup_hash_log_dir)]
            )
        if extra := os.environ.get("GE_LMCACHE_MP_EXTRA_ARGS"):
            command.extend(extra.split())

        env = os.environ.copy()
        env["PYTHONHASHSEED"] = os.environ.get("PYTHONHASHSEED", "0")
        env["LMCACHE_LOG_LEVEL"] = self.config.lmcache_log_level
        if self.config.use_mooncake_store:
            env["MOONCAKE_TE_META_DATA_SERVER"] = self.config.mooncake_metadata_server
            env["MOONCAKE_MASTER"] = self.config.mooncake_master_addr
            env["MOONCAKE_PROTOCOL"] = self.config.mooncake_protocol
            env["LOCAL_HOSTNAME"] = self.config.mooncake_local_hostname

        print(
            "Starting LMCache MP server on "
            f"{self.config.lmcache_mp_bind_host}:{self.config.lmcache_mp_port}; "
            f"log: {log_path}"
        )
        self.lmcache_mp = self._start(command, log_path, env)
        (self.config.run_dir / "lmcache_mp.pid").write_text(
            f"{self.lmcache_mp.pid}\n", encoding="utf-8"
        )
        record_service_pid(self.config, "lmcache_mp", self.lmcache_mp.pid, log_path)

    def wait_for_lmcache_mp_ready(self) -> None:
        if self.config.kv_backend != "mp":
            return
        log_path = self.config.log_dir / "lmcache_mp_server.log"
        if not wait_for_tcp_port(
            self.config.lmcache_mp_host,
            self.config.lmcache_mp_port,
            self.config.start_timeout,
        ):
            raise RuntimeError(
                "Timed out waiting for LMCache MP server; last log lines:\n"
                + tail_lines(log_path)
            )
        if self.lmcache_mp and self.lmcache_mp.poll() is not None:
            raise RuntimeError(
                "LMCache MP server exited after port became reachable; last log lines:\n"
                + tail_lines(log_path)
            )
        print(
            f"LMCache MP server is ready on "
            f"{self.config.lmcache_mp_host}:{self.config.lmcache_mp_port}"
        )

    def start_engine_server(self, phase: str) -> None:
        log_path = self.config.log_dir / f"{phase}_server.log"
        print(
            f"Starting {phase} {self.config.engine} server on "
            f"{self.config.base_url}; log: {log_path}"
        )
        ensure_command(self.config.vllm_bin)
        env = os.environ.copy()
        env["PYTHONHASHSEED"] = os.environ.get("PYTHONHASHSEED", "0")
        env["LMCACHE_LOG_LEVEL"] = self.config.lmcache_log_level
        command = [
            self.config.vllm_bin,
            "serve",
            self.config.model_path,
            "--host",
            self.config.host,
            "--port",
            self.config.port,
            "--served-model-name",
            self.config.model_name,
            "--kv-transfer-config",
            self.config.vllm_kv_transfer_config_json(),
            *self.config.engine_args,
        ]
        self.server = self._start(command, log_path, env)
        (self.config.run_dir / f"{phase}.pid").write_text(
            f"{self.server.pid}\n", encoding="utf-8"
        )

    def wait_for_engine_ready(self, phase: str) -> None:
        log_path = self.config.log_dir / f"{phase}_server.log"
        deadline = time.monotonic() + self.config.start_timeout
        while time.monotonic() < deadline:
            if self.server and self.server.poll() is not None:
                raise RuntimeError(
                    f"{phase} server exited before becoming ready; last log lines:\n"
                    + tail_lines(log_path, 80)
                )
            result = subprocess.run(
                [
                    self.config.python_bin,
                    str(REPO_ROOT / "scripts" / "kv_baseline" / "kv_baseline_client.py"),
                    "wait",
                    "--base-url",
                    self.config.base_url,
                    "--timeout",
                    "2",
                    "--interval",
                    "1",
                ],
                cwd=REPO_ROOT,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )
            if result.returncode == 0:
                self._wait_for_ready_log(phase)
                print(f"{phase} server is ready at {self.config.base_url}")
                return
            time.sleep(2)
        raise RuntimeError(
            f"Timed out waiting for {phase} server; last log lines:\n"
            + tail_lines(log_path, 80)
        )

    def _wait_for_ready_log(self, phase: str) -> None:
        if not self.config.wait_for_ready_log:
            return
        import re

        log_path = self.config.log_dir / f"{phase}_server.log"
        deadline = time.monotonic() + self.config.start_timeout
        pattern = re.compile(self.config.ready_log_pattern)
        while time.monotonic() < deadline:
            if log_path.exists() and pattern.search(
                log_path.read_text(encoding="utf-8", errors="replace")
            ):
                print(f"{phase} server emitted ready log: {self.config.ready_log_pattern}")
                return
            if self.server and self.server.poll() is not None:
                raise RuntimeError(
                    f"{phase} server exited before ready log; last log lines:\n"
                    + tail_lines(log_path, 80)
                )
            time.sleep(1)
        raise RuntimeError(
            f"Timed out waiting for {phase} ready log; last log lines:\n"
            + tail_lines(log_path, 80)
        )

    def stop_server(self, label: str = "server") -> None:
        self._stop_process("engine " + label, self.server)
        self.server = None

    def stop_lmcache_mp(self) -> None:
        self._stop_process("LMCache MP server", self.lmcache_mp)
        self.lmcache_mp = None

    def stop_mooncake(self) -> None:
        self._stop_process("Mooncake master", self.mooncake_master)
        self._stop_process("Mooncake metadata server", self.mooncake_metadata)
        self.mooncake_master = None
        self.mooncake_metadata = None

    def _stop_process(self, label: str, proc: subprocess.Popen[bytes] | None) -> None:
        if proc is None or proc.poll() is not None:
            return
        print(f"Stopping {label} pid={proc.pid}")
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        except ProcessLookupError:
            return
        try:
            proc.wait(timeout=60)
        except subprocess.TimeoutExpired:
            print(f"{label} pid={proc.pid} did not exit after SIGTERM; sending SIGKILL")
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except ProcessLookupError:
                return
            proc.wait(timeout=10)

    def cleanup(self) -> None:
        if not self.config.keep_server_after_reuse:
            self.stop_server("active")
        if self.config.kv_backend == "mp" and not self.config.keep_lmcache_mp_after_run:
            self.stop_lmcache_mp()
        if self.config.use_mooncake_store and not self.config.keep_mooncake_after_run:
            self.stop_mooncake()
