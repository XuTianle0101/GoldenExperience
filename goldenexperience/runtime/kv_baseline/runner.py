"""Top-level orchestration for the same-model KV baseline."""

from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path

from goldenexperience.runtime.kv_baseline.config import REPO_ROOT, BaselineConfig
from goldenexperience.runtime.kv_baseline.prompts import write_generated_disk_prompt
from goldenexperience.runtime.kv_baseline.services import (
    ProcessGroup,
    validate_runtime_requirements,
)

CLIENT = REPO_ROOT / "scripts" / "kv_baseline" / "kv_baseline_client.py"


def _run_client(config: BaselineConfig, args: list[str]) -> None:
    subprocess.run([config.python_bin, str(CLIENT), *args], cwd=REPO_ROOT, check=True)


def _mark_lmcache_log_start(config: BaselineConfig, phase: str) -> None:
    if config.kv_backend != "mp":
        return
    source = config.log_dir / "lmcache_mp_server.log"
    offset = source.stat().st_size if source.exists() else 0
    (config.run_dir / f"{phase}_lmcache_log_offset.txt").write_text(f"{offset}\n", encoding="utf-8")


def _capture_lmcache_log_delta(config: BaselineConfig, phase: str) -> None:
    if config.kv_backend != "mp":
        return
    source = config.log_dir / "lmcache_mp_server.log"
    if not source.exists():
        return
    offset_path = config.run_dir / f"{phase}_lmcache_log_offset.txt"
    offset = int(offset_path.read_text(encoding="utf-8").strip() or "0")
    with source.open("rb") as handle:
        handle.seek(max(0, offset))
        data = handle.read()
    (config.log_dir / f"{phase}_lmcache_mp_server.log").write_bytes(data)


def _send_request(
    config: BaselineConfig,
    phase: str,
    prompt_id: str,
    output_path: Path,
) -> None:
    args = [
        "request",
        "--base-url",
        config.base_url,
        "--model",
        config.model_name,
        "--prompt-file",
        str(config.prompt_file),
        "--prompt-id",
        prompt_id,
        "--phase",
        phase,
        "--output",
        str(output_path),
        "--timeout",
        str(config.request_timeout),
    ]
    import os

    if max_tokens := os.environ.get("GE_MAX_TOKENS"):
        args.extend(["--max-tokens", max_tokens])
    if temperature := os.environ.get("GE_TEMPERATURE"):
        args.extend(["--temperature", temperature])
    if not config.include_usage:
        args.append("--no-include-usage")
    _run_client(config, args)


def _run_phase_request(config: BaselineConfig, phase: str) -> None:
    request_output = config.request_dir / f"{phase}.json"
    if config.warmup_before_measure:
        print(
            f"Sending {phase} warmup request ({config.warmup_prompt_id}); "
            "output is excluded from timing deltas"
        )
        _send_request(
            config,
            f"{phase}_warmup",
            config.warmup_prompt_id,
            config.request_dir / f"{phase}_warmup.json",
        )
        time.sleep(config.after_warmup_sleep)

    _mark_lmcache_log_start(config, phase)
    print(f"Sending {phase} request")
    _send_request(config, phase, config.prompt_id, request_output)

    if config.enable_metrics:
        _run_client(
            config,
            [
                "fetch-metrics",
                "--base-url",
                config.base_url,
                "--output",
                str(config.metrics_dir / f"{phase}.prom"),
                "--allow-missing",
            ],
        )

    time.sleep(config.after_request_sleep)
    _capture_lmcache_log_delta(config, phase)


def _write_summary(config: BaselineConfig) -> None:
    args = [
        "summarize",
        "--run-dir",
        str(config.run_dir),
        "--output",
        str(config.run_dir / "summary.json"),
    ]
    if config.require_reuse_evidence:
        args.append("--require-reuse-evidence")
    if config.force_disk_offload:
        args.append("--require-disk-offload")
    _run_client(config, args)


def _print_header(config: BaselineConfig) -> None:
    print(f"KV baseline run directory: {config.run_dir}")
    print(f"KV backend: {config.kv_backend}; engine: {config.engine}")
    print(f"Recorded config: {config.config_file}")
    print(f"Persistent KV cache dir: {config.kv_cache_dir}")
    if config.kv_backend == "mp":
        print(
            "LMCache MP: "
            f"{config.lmcache_mp_bind_host}:{config.lmcache_mp_port}; "
            f"L2 adapter: {config.l2_adapter_json()}"
        )
        print(f"vLLM KV transfer config: {config.vllm_kv_transfer_config_json()}")
    if config.use_mooncake_store:
        print(
            "Mooncake: "
            f"master={config.mooncake_master_addr}; "
            f"metadata={config.mooncake_metadata_server}; "
            f"storage={config.mooncake_storage_root}"
        )
    print(f"Force disk offload: {int(config.force_disk_offload)}")
    print(f"Prompt: {config.prompt_file}#{config.prompt_id}")
    if config.force_disk_offload and not config.prompt_file_was_set:
        print(
            f"Generated disk prompt repeat: {config.disk_prompt_repeat}; "
            f"max_tokens={config.disk_prompt_max_tokens}"
        )
    extra_args = " ".join(config.engine_args) if config.engine_args else "(none)"
    print(f"Extra engine args: {extra_args}")


def run_baseline(config: BaselineConfig) -> int:
    """Run a two-phase offload/reuse baseline."""

    config.ensure_dirs()
    write_generated_disk_prompt(config)
    config.write_lmcache_config()
    config.write_metadata()
    _print_header(config)

    if config.dry_run:
        print("GE_DRY_RUN=1: generated config and metadata without starting services.")
        return 0

    validate_runtime_requirements(config)

    processes = ProcessGroup(config)
    try:
        processes.start_mooncake_services()
        processes.wait_for_mooncake_ready()
        processes.start_lmcache_mp_server()
        processes.wait_for_lmcache_mp_ready()

        processes.start_engine_server("offload")
        processes.wait_for_engine_ready("offload")
        _run_phase_request(config, "offload")

        if config.baseline_mode == "restart":
            processes.stop_server("offload")
            time.sleep(float(os.environ.get("GE_AFTER_ENGINE_STOP_SLEEP_SEC", "10")))
            processes.start_engine_server("reuse")
            processes.wait_for_engine_ready("reuse")

        _run_phase_request(config, "reuse")

        if not config.keep_server_after_reuse:
            processes.stop_server("reuse")

        _write_summary(config)

        if config.kv_backend == "mp" and not config.keep_lmcache_mp_after_run:
            processes.stop_lmcache_mp()
        if config.use_mooncake_store and not config.keep_mooncake_after_run:
            processes.stop_mooncake()
    finally:
        processes.cleanup()

    print("Done. Key outputs:")
    for relative in [
        "metadata.json",
        "lmc_config.yaml",
        "requests/offload.json",
        "requests/reuse.json",
        "summary.json",
    ]:
        print(f"  {config.run_dir / relative}")
    return 0
