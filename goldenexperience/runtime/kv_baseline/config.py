"""Configuration for the vLLM + LMCache MP + Mooncake KV baseline."""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]


def _env_bool(name: str, default: str = "0") -> bool:
    return os.environ.get(name, default).lower() in {"1", "true", "yes", "on"}


def _tcp_url(host: str) -> str:
    return host if "://" in host else f"tcp://{host}"


def _env_int(name: str, default: str) -> int:
    value = os.environ.get(name, default)
    try:
        return int(value)
    except ValueError:
        pass

    try:
        parsed = Decimal(value)
    except InvalidOperation as exc:
        raise ValueError(f"{name} must be an integer, got {value!r}") from exc
    if parsed != parsed.to_integral_value():
        raise ValueError(f"{name} must be an integer, got {value!r}")
    return int(parsed)


def _abs_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def _json_object(value: str, name: str) -> dict[str, Any]:
    parsed = json.loads(value)
    if not isinstance(parsed, dict):
        raise ValueError(f"{name} must be a JSON object")
    return parsed


@dataclass
class BaselineConfig:
    """All state needed by the same-model KV baseline runner."""

    python_bin: str
    kv_backend: str
    engine: str
    vllm_bin: str
    lmcache_bin: str
    mooncake_master_bin: str
    mooncake_http_metadata_server_bin: str
    model_path: str
    model_name: str
    host: str
    client_host: str
    port: str
    run_id: str
    run_dir: Path
    cache_dir: Path
    log_dir: Path
    request_dir: Path
    metrics_dir: Path
    config_file: Path
    prompt_file: Path
    prompt_file_was_set: bool
    prompt_id: str
    chunk_size: int
    force_disk_offload: bool
    local_cpu_enabled: bool
    local_cpu_gb: float
    local_disk_gb: float
    hash_algorithm: str
    disk_prompt_id: str
    disk_prompt_repeat: int
    disk_prompt_max_tokens: int
    start_timeout: int
    request_timeout: int
    after_request_sleep: float
    after_warmup_sleep: float
    baseline_mode: str
    disable_engine_prefix_cache: bool
    enable_metrics: bool
    save_decode_cache: bool
    include_usage: bool
    wait_for_ready_log: bool
    ready_log_pattern: str
    warmup_before_measure: bool
    warmup_prompt_id: str
    require_reuse_evidence: bool
    keep_server_after_reuse: bool
    keep_lmcache_mp_after_run: bool
    keep_mooncake_after_run: bool
    dry_run: bool
    lmcache_mp_host: str
    lmcache_mp_bind_host: str
    lmcache_mp_connect_host: str
    lmcache_mp_port: int
    lmcache_mp_http_host: str
    lmcache_mp_http_port: int
    lmcache_mp_prometheus_port: int
    lmcache_mp_l1_gb: float
    lmcache_mp_l1_init_gb: int
    lmcache_mp_eviction_policy: str
    lmcache_mp_l2_store_policy: str
    lmcache_mp_l2_adapter_type: str
    lmcache_mp_l2_dir: Path
    lmcache_mp_l2_use_odirect: bool
    lmcache_mp_l2_num_workers: str
    lmcache_mp_transfer_mode: str
    lmcache_mp_lookup_hash_log: bool
    lmcache_mp_lookup_hash_log_dir: Path
    lmcache_log_level: str
    mooncake_master_host: str
    mooncake_master_port: int
    mooncake_metadata_host: str
    mooncake_metadata_bind_host: str
    mooncake_metadata_port: int
    mooncake_protocol: str
    mooncake_storage_root: Path
    mooncake_local_hostname: str
    mooncake_global_segment_size: str
    mooncake_local_buffer_size: str
    mooncake_num_workers: int
    mooncake_per_op_workers: dict[str, int]
    mooncake_master_addr: str
    mooncake_metadata_server: str
    mooncake_extra_setup: dict[str, Any]
    raw_engine_args: list[str] = field(default_factory=list)
    engine_args: list[str] = field(default_factory=list)

    @classmethod
    def from_env(cls, raw_engine_args: list[str] | None = None) -> "BaselineConfig":
        args = list(raw_engine_args or [])
        if args and args[0] == "--":
            args = args[1:]

        kv_backend = os.environ.get("GE_KV_BACKEND", "mp")
        engine = os.environ.get("GE_ENGINE", "vllm")
        if kv_backend != "mp":
            raise ValueError(
                "GoldenExperience baseline uses GE_KV_BACKEND=mp with LMCacheMPConnector"
            )
        if engine != "vllm":
            raise ValueError("GoldenExperience baseline uses GE_ENGINE=vllm")

        run_id = os.environ.get("GE_RUN_ID") or time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
        run_dir = _abs_path(os.environ.get("GE_RUN_DIR", f"artifacts/kv_baseline/{run_id}"))
        cache_dir = _abs_path(os.environ.get("GE_KV_CACHE_DIR", str(run_dir / "cache")))
        config_file = _abs_path(
            os.environ.get("GE_LMCACHE_CONFIG_FILE", str(run_dir / "lmc_config.yaml"))
        )
        prompt_file_was_set = "GE_PROMPT_FILE" in os.environ
        prompt_file = _abs_path(
            os.environ.get("GE_PROMPT_FILE", "configs/kv_baseline_prompts.json")
        )
        prompt_id = os.environ.get("GE_PROMPT_ID", "gsm8k_natalia_clips")
        force_disk_offload = _env_bool("GE_FORCE_DISK_OFFLOAD", "1")
        disk_prompt_id = os.environ.get("GE_DISK_PROMPT_ID", "kv_disk_long_prefix")
        if force_disk_offload and not prompt_file_was_set:
            prompt_file = run_dir / "disk_prompt.json"
            prompt_id = disk_prompt_id

        local_cpu_enabled = _env_bool("GE_KV_LOCAL_CPU_ENABLED", "true")
        local_cpu_gb = float(os.environ.get("GE_KV_LOCAL_CPU_GB", "10"))
        if force_disk_offload:
            local_cpu_enabled = _env_bool("GE_KV_LOCAL_CPU_ENABLED", "false")
            local_cpu_gb = float(os.environ.get("GE_KV_LOCAL_CPU_GB", "0"))

        keep_server_after_reuse = _env_bool("GE_KEEP_SERVER_AFTER_REUSE", "0")
        keep_lmcache = _env_bool(
            "GE_KEEP_LMCACHE_MP_AFTER_RUN", "1" if keep_server_after_reuse else "0"
        )
        keep_mooncake = _env_bool("GE_KEEP_MOONCAKE_AFTER_RUN", "1" if keep_lmcache else "0")

        wait_for_ready_log = _env_bool("GE_WAIT_FOR_READY_LOG", "0")
        require_reuse_evidence = _env_bool(
            "GE_REQUIRE_REUSE_EVIDENCE", "1" if force_disk_offload else "0"
        )

        lmcache_mp_host = os.environ.get("GE_LMCACHE_MP_HOST", "127.0.0.1")
        lmcache_mp_connect_host = os.environ.get(
            "GE_LMCACHE_MP_CONNECT_HOST", _tcp_url(lmcache_mp_host)
        )
        lmcache_mp_l2_adapter_type = os.environ.get(
            "GE_LMCACHE_MP_L2_ADAPTER_TYPE", "mooncake_store"
        )
        mooncake_master_host = os.environ.get("GE_MOONCAKE_MASTER_HOST", "127.0.0.1")
        mooncake_master_port = int(os.environ.get("GE_MOONCAKE_MASTER_PORT", "50051"))
        mooncake_metadata_host = os.environ.get(
            "GE_MOONCAKE_METADATA_HOST", mooncake_master_host
        )
        mooncake_metadata_port = int(os.environ.get("GE_MOONCAKE_METADATA_PORT", "8080"))
        mooncake_master_addr = os.environ.get(
            "GE_MOONCAKE_MASTER_ADDR", f"{mooncake_master_host}:{mooncake_master_port}"
        )
        mooncake_metadata_server = os.environ.get(
            "GE_MOONCAKE_METADATA_SERVER",
            f"http://{mooncake_metadata_host}:{mooncake_metadata_port}/metadata",
        )
        per_op_workers = {
            str(k): int(v)
            for k, v in _json_object(
                os.environ.get(
                    "GE_MOONCAKE_PER_OP_WORKERS_JSON",
                    '{"lookup":2,"retrieve":8,"store":4}',
                ),
                "GE_MOONCAKE_PER_OP_WORKERS_JSON",
            ).items()
        }

        cfg = cls(
            python_bin=os.environ.get("PYTHON_BIN", "python3"),
            kv_backend=kv_backend,
            engine=engine,
            vllm_bin=os.environ.get("VLLM_BIN", "vllm"),
            lmcache_bin=os.environ.get("LMCACHE_BIN", "lmcache"),
            mooncake_master_bin=os.environ.get("MOONCAKE_MASTER_BIN", "mooncake_master"),
            mooncake_http_metadata_server_bin=os.environ.get(
                "MOONCAKE_HTTP_METADATA_SERVER_BIN", "mooncake_http_metadata_server"
            ),
            model_path=os.environ.get("GE_MODEL_PATH", "Qwen/Qwen3-8B"),
            model_name=os.environ.get(
                "GE_MODEL_NAME", os.environ.get("GE_MODEL_PATH", "Qwen/Qwen3-8B")
            ),
            host=os.environ.get("GE_HOST", "0.0.0.0"),
            client_host=os.environ.get("GE_CLIENT_HOST", "127.0.0.1"),
            port=os.environ.get("GE_PORT", "30000"),
            run_id=run_id,
            run_dir=run_dir,
            cache_dir=cache_dir,
            log_dir=run_dir / "logs",
            request_dir=run_dir / "requests",
            metrics_dir=run_dir / "metrics",
            config_file=config_file,
            prompt_file=prompt_file,
            prompt_file_was_set=prompt_file_was_set,
            prompt_id=prompt_id,
            chunk_size=int(os.environ.get("GE_KV_CHUNK_SIZE", "16")),
            force_disk_offload=force_disk_offload,
            local_cpu_enabled=local_cpu_enabled,
            local_cpu_gb=local_cpu_gb,
            local_disk_gb=float(os.environ.get("GE_KV_LOCAL_DISK_GB", "100")),
            hash_algorithm=os.environ.get("GE_LMCACHE_HASH_ALGORITHM", "blake3"),
            disk_prompt_id=disk_prompt_id,
            disk_prompt_repeat=int(os.environ.get("GE_DISK_PROMPT_REPEAT", "256")),
            disk_prompt_max_tokens=int(os.environ.get("GE_DISK_PROMPT_MAX_TOKENS", "128")),
            start_timeout=int(os.environ.get("GE_SERVER_START_TIMEOUT_SEC", "900")),
            request_timeout=int(os.environ.get("GE_REQUEST_TIMEOUT_SEC", "600")),
            after_request_sleep=float(os.environ.get("GE_AFTER_REQUEST_SLEEP_SEC", "10")),
            after_warmup_sleep=float(os.environ.get("GE_AFTER_WARMUP_SLEEP_SEC", "1")),
            baseline_mode=os.environ.get("GE_BASELINE_MODE", "restart"),
            disable_engine_prefix_cache=_env_bool("GE_DISABLE_ENGINE_PREFIX_CACHE", "0"),
            enable_metrics=_env_bool("GE_ENABLE_ENGINE_METRICS", "1"),
            save_decode_cache=_env_bool("GE_SAVE_DECODE_CACHE", "false"),
            include_usage=_env_bool("GE_INCLUDE_USAGE", "1"),
            wait_for_ready_log=wait_for_ready_log,
            ready_log_pattern=os.environ.get(
                "GE_READY_LOG_PATTERN", "Application startup complete|Uvicorn running"
            ),
            warmup_before_measure=_env_bool("GE_WARMUP_BEFORE_MEASURE", "1"),
            warmup_prompt_id=os.environ.get("GE_WARMUP_PROMPT_ID", "kv_baseline_warmup"),
            require_reuse_evidence=require_reuse_evidence,
            keep_server_after_reuse=keep_server_after_reuse,
            keep_lmcache_mp_after_run=keep_lmcache,
            keep_mooncake_after_run=keep_mooncake,
            dry_run=_env_bool("GE_DRY_RUN", "0"),
            lmcache_mp_host=lmcache_mp_host,
            lmcache_mp_bind_host=os.environ.get("GE_LMCACHE_MP_BIND_HOST", lmcache_mp_host),
            lmcache_mp_connect_host=lmcache_mp_connect_host,
            lmcache_mp_port=int(os.environ.get("GE_LMCACHE_MP_PORT", "6555")),
            lmcache_mp_http_host=os.environ.get("GE_LMCACHE_MP_HTTP_HOST", "127.0.0.1"),
            lmcache_mp_http_port=int(os.environ.get("GE_LMCACHE_MP_HTTP_PORT", "8081")),
            lmcache_mp_prometheus_port=int(
                os.environ.get("GE_LMCACHE_MP_PROMETHEUS_PORT", "9090")
            ),
            lmcache_mp_l1_gb=float(os.environ.get("GE_LMCACHE_MP_L1_GB", "4")),
            lmcache_mp_l1_init_gb=_env_int("GE_LMCACHE_MP_L1_INIT_GB", "1"),
            lmcache_mp_eviction_policy=os.environ.get("GE_LMCACHE_MP_EVICTION_POLICY", "noop"),
            lmcache_mp_l2_store_policy=os.environ.get(
                "GE_LMCACHE_MP_L2_STORE_POLICY", "skip_l1"
            ),
            lmcache_mp_l2_adapter_type=lmcache_mp_l2_adapter_type,
            lmcache_mp_l2_dir=_abs_path(os.environ.get("GE_LMCACHE_MP_L2_DIR", str(cache_dir))),
            lmcache_mp_l2_use_odirect=_env_bool("GE_LMCACHE_MP_L2_USE_ODIRECT", "false"),
            lmcache_mp_l2_num_workers=os.environ.get("GE_LMCACHE_MP_L2_NUM_WORKERS", ""),
            lmcache_mp_transfer_mode=os.environ.get("GE_LMCACHE_MP_TRANSFER_MODE", "auto"),
            lmcache_mp_lookup_hash_log=_env_bool("GE_LMCACHE_MP_LOOKUP_HASH_LOG", "1"),
            lmcache_mp_lookup_hash_log_dir=_abs_path(
                os.environ.get("GE_LMCACHE_MP_LOOKUP_HASH_LOG_DIR", str(run_dir / "lookup_hashes"))
            ),
            lmcache_log_level=os.environ.get("LMCACHE_LOG_LEVEL", "DEBUG"),
            mooncake_master_host=mooncake_master_host,
            mooncake_master_port=mooncake_master_port,
            mooncake_metadata_host=mooncake_metadata_host,
            mooncake_metadata_bind_host=os.environ.get(
                "GE_MOONCAKE_METADATA_BIND_HOST", mooncake_metadata_host
            ),
            mooncake_metadata_port=mooncake_metadata_port,
            mooncake_protocol=os.environ.get("GE_MOONCAKE_PROTOCOL", "tcp"),
            mooncake_storage_root=_abs_path(
                os.environ.get("GE_MOONCAKE_STORAGE_ROOT", str(cache_dir / "mooncake"))
            ),
            mooncake_local_hostname=os.environ.get("GE_MOONCAKE_LOCAL_HOSTNAME", "127.0.0.1"),
            mooncake_global_segment_size=os.environ.get(
                "GE_MOONCAKE_GLOBAL_SEGMENT_SIZE", "4294967296"
            ),
            mooncake_local_buffer_size=os.environ.get(
                "GE_MOONCAKE_LOCAL_BUFFER_SIZE", "4294967296"
            ),
            mooncake_num_workers=int(os.environ.get("GE_MOONCAKE_NUM_WORKERS", "4")),
            mooncake_per_op_workers=per_op_workers,
            mooncake_master_addr=mooncake_master_addr,
            mooncake_metadata_server=mooncake_metadata_server,
            mooncake_extra_setup=_json_object(
                os.environ.get("GE_MOONCAKE_EXTRA_SETUP_JSON", "{}"),
                "GE_MOONCAKE_EXTRA_SETUP_JSON",
            ),
            raw_engine_args=args,
        )
        if cfg.baseline_mode not in {"restart", "same-process"}:
            raise ValueError(
                f"GE_BASELINE_MODE must be restart or same-process, got {cfg.baseline_mode}"
            )
        cfg.engine_args = cfg.normalized_engine_args()
        return cfg

    @property
    def base_url(self) -> str:
        return f"http://{self.client_host}:{self.port}"

    @property
    def use_mooncake_store(self) -> bool:
        return self.kv_backend == "mp" and self.lmcache_mp_l2_adapter_type == "mooncake_store"

    @property
    def kv_cache_dir(self) -> Path:
        if self.kv_backend == "mp" and not self.use_mooncake_store:
            return self.lmcache_mp_l2_dir
        return self.cache_dir

    def normalized_engine_args(self) -> list[str]:
        normalized: list[str] = []
        args = list(self.raw_engine_args)
        while args:
            arg = args.pop(0)
            if self.engine == "vllm" and arg == "--tp":
                if not args:
                    raise ValueError("--tp requires a value")
                normalized.extend(["--tensor-parallel-size", args.pop(0)])
            elif self.engine == "vllm" and arg in {"--disable-engine-prefix-cache", "--enable-metrics"}:
                continue
            else:
                normalized.append(arg)
        return normalized

    def l2_adapter(self) -> dict[str, Any]:
        if self.use_mooncake_store:
            adapter: dict[str, Any] = {
                "type": "mooncake_store",
                "num_workers": self.mooncake_num_workers,
                "per_op_workers": self.mooncake_per_op_workers,
                "local_hostname": self.mooncake_local_hostname,
                "metadata_server": self.mooncake_metadata_server,
                "master_server_addr": self.mooncake_master_addr,
                "protocol": self.mooncake_protocol,
                "storage_root_dir": str(self.mooncake_storage_root),
                "global_segment_size": self.mooncake_global_segment_size,
                "local_buffer_size": self.mooncake_local_buffer_size,
            }
            adapter.update(self.mooncake_extra_setup)
            return adapter

        adapter = {
            "type": self.lmcache_mp_l2_adapter_type,
            "base_path": str(self.lmcache_mp_l2_dir),
        }
        if self.lmcache_mp_l2_use_odirect:
            adapter["use_odirect"] = True
        if self.lmcache_mp_l2_num_workers:
            adapter["num_thread"] = int(self.lmcache_mp_l2_num_workers)
        return adapter

    def l2_adapter_json(self) -> str:
        return json.dumps(self.l2_adapter(), separators=(",", ":"))

    def vllm_kv_transfer_config(self) -> dict[str, Any]:
        return {
            "kv_connector": "LMCacheMPConnector",
            "kv_role": "kv_both",
            "kv_connector_extra_config": {
                "lmcache.mp.host": self.lmcache_mp_connect_host,
                "lmcache.mp.port": self.lmcache_mp_port,
                "lmcache.mp.mp_transfer_mode": self.lmcache_mp_transfer_mode,
            },
        }

    def vllm_kv_transfer_config_json(self) -> str:
        return json.dumps(self.vllm_kv_transfer_config(), separators=(",", ":"))

    def ensure_dirs(self) -> None:
        for path in [
            self.cache_dir,
            self.lmcache_mp_l2_dir,
            self.log_dir,
            self.request_dir,
            self.metrics_dir,
        ]:
            path.mkdir(parents=True, exist_ok=True)
        self.config_file.parent.mkdir(parents=True, exist_ok=True)
        if self.use_mooncake_store:
            self.mooncake_storage_root.mkdir(parents=True, exist_ok=True)
        if self.lmcache_mp_lookup_hash_log:
            self.lmcache_mp_lookup_hash_log_dir.mkdir(parents=True, exist_ok=True)

    def write_lmcache_config(self) -> None:
        text = f"""# Generated by goldenexperience.runtime.kv_baseline
kv_backend: mp
engine: vllm
chunk_size: {self.chunk_size}
hash_algorithm: {self.hash_algorithm}
lmcache_mp:
  host: {self.lmcache_mp_host}
  bind_host: {self.lmcache_mp_bind_host}
  port: {self.lmcache_mp_port}
  http_host: {self.lmcache_mp_http_host}
  http_port: {self.lmcache_mp_http_port}
  prometheus_port: {self.lmcache_mp_prometheus_port}
  l1_size_gb: {self.lmcache_mp_l1_gb}
  l1_init_size_gb: {self.lmcache_mp_l1_init_gb}
  eviction_policy: {self.lmcache_mp_eviction_policy}
  l2_store_policy: {self.lmcache_mp_l2_store_policy}
  l2_adapter_json: '{self.l2_adapter_json()}'
mooncake:
  enabled: {str(self.use_mooncake_store).lower()}
  master_server_addr: {self.mooncake_master_addr}
  metadata_server: {self.mooncake_metadata_server}
  protocol: {self.mooncake_protocol}
  storage_root_dir: {self.mooncake_storage_root}
  local_hostname: {self.mooncake_local_hostname}
vllm:
  kv_transfer_config: '{self.vllm_kv_transfer_config_json()}'
"""
        self.config_file.write_text(text, encoding="utf-8")

    def metadata(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "run_dir": str(self.run_dir),
            "mode": self.baseline_mode,
            "kv_backend": self.kv_backend,
            "engine": self.engine,
            "model_path": self.model_path,
            "model_name": self.model_name,
            "base_url": self.base_url,
            "prompt_file": str(self.prompt_file),
            "prompt_id": self.prompt_id,
            "lmcache_config_file": str(self.config_file),
            "kv_cache_dir": str(self.kv_cache_dir),
            "chunk_size": self.chunk_size,
            "force_disk_offload": self.force_disk_offload,
            "local_cpu_enabled": self.local_cpu_enabled,
            "local_cpu_gb": self.local_cpu_gb,
            "local_disk_gb": self.local_disk_gb,
            "hash_algorithm": self.hash_algorithm,
            "disk_prompt_id": self.disk_prompt_id,
            "disk_prompt_repeat": self.disk_prompt_repeat,
            "disk_prompt_max_tokens": self.disk_prompt_max_tokens,
            "generated_disk_prompt": self.force_disk_offload and not self.prompt_file_was_set,
            "lmcache_mp": {
                "enabled": self.kv_backend == "mp",
                "host": self.lmcache_mp_host,
                "bind_host": self.lmcache_mp_bind_host,
                "connect_host": self.lmcache_mp_connect_host,
                "port": self.lmcache_mp_port,
                "http_host": self.lmcache_mp_http_host,
                "http_port": self.lmcache_mp_http_port,
                "prometheus_port": self.lmcache_mp_prometheus_port,
                "l1_size_gb": self.lmcache_mp_l1_gb,
                "l1_init_size_gb": self.lmcache_mp_l1_init_gb,
                "eviction_policy": self.lmcache_mp_eviction_policy,
                "l2_store_policy": self.lmcache_mp_l2_store_policy,
                "l2_adapter_json": self.l2_adapter_json(),
                "l2_adapter": self.l2_adapter(),
                "l2_adapter_type": self.lmcache_mp_l2_adapter_type,
                "l2_dir": str(self.lmcache_mp_l2_dir),
                "transfer_mode": self.lmcache_mp_transfer_mode,
                "lookup_hash_log": self.lmcache_mp_lookup_hash_log,
                "lookup_hash_log_dir": str(self.lmcache_mp_lookup_hash_log_dir),
                "pid_file": str(self.run_dir / "lmcache_mp.pid"),
                "log_path": str(self.log_dir / "lmcache_mp_server.log"),
            },
            "mooncake": {
                "enabled": self.use_mooncake_store,
                "master_bin": self.mooncake_master_bin,
                "http_metadata_server_bin": self.mooncake_http_metadata_server_bin,
                "master_host": self.mooncake_master_host,
                "master_port": self.mooncake_master_port,
                "master_server_addr": self.mooncake_master_addr,
                "metadata_host": self.mooncake_metadata_host,
                "metadata_bind_host": self.mooncake_metadata_bind_host,
                "metadata_port": self.mooncake_metadata_port,
                "metadata_server": self.mooncake_metadata_server,
                "protocol": self.mooncake_protocol,
                "storage_root": str(self.mooncake_storage_root),
                "local_hostname": self.mooncake_local_hostname,
                "global_segment_size": self.mooncake_global_segment_size,
                "local_buffer_size": self.mooncake_local_buffer_size,
                "num_workers": self.mooncake_num_workers,
                "per_op_workers": self.mooncake_per_op_workers,
                "adapter_json": self.l2_adapter_json() if self.use_mooncake_store else None,
                "master_pid_file": str(self.run_dir / "mooncake_master.pid"),
                "metadata_pid_file": str(self.run_dir / "mooncake_metadata.pid"),
                "master_log_path": str(self.log_dir / "mooncake_master.log"),
                "metadata_log_path": str(self.log_dir / "mooncake_metadata_server.log"),
            },
            "services": {
                "lmcache_mp": {
                    "enabled": self.kv_backend == "mp",
                    "pid_file": str(self.run_dir / "lmcache_mp.pid"),
                    "log_path": str(self.log_dir / "lmcache_mp_server.log"),
                },
                "mooncake_master": {
                    "enabled": self.use_mooncake_store,
                    "pid_file": str(self.run_dir / "mooncake_master.pid"),
                    "log_path": str(self.log_dir / "mooncake_master.log"),
                },
                "mooncake_metadata": {
                    "enabled": self.use_mooncake_store,
                    "pid_file": str(self.run_dir / "mooncake_metadata.pid"),
                    "log_path": str(self.log_dir / "mooncake_metadata_server.log"),
                },
            },
            "vllm_kv_transfer_config": self.vllm_kv_transfer_config_json(),
            "disable_engine_prefix_cache": self.disable_engine_prefix_cache,
            "enable_engine_metrics": self.enable_metrics,
            "include_usage": self.include_usage,
            "wait_for_ready_log": self.wait_for_ready_log,
            "ready_log_pattern": self.ready_log_pattern,
            "warmup_before_measure": self.warmup_before_measure,
            "warmup_prompt_id": self.warmup_prompt_id,
            "require_reuse_evidence": self.require_reuse_evidence,
            "created_unix": time.time(),
        }

    def write_metadata(self) -> None:
        (self.run_dir / "metadata.json").write_text(
            json.dumps(self.metadata(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
