#!/usr/bin/env python3
"""Run quality-gated Qwen3 8B -> 14B hidden-bridge KV reuse against vLLM/LMCache."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from goldenexperience.runtime.cross_model_reuse import (
    load_lookup_hash_records,
    mooncake_key_exists,
    object_key_string,
    select_lookup_candidate,
    token_ids_from_prompt,
)
from goldenexperience.runtime.kv_baseline.config import BaselineConfig
from goldenexperience.runtime.kv_baseline.prompts import write_generated_disk_prompt
from goldenexperience.runtime.kv_baseline.runner import _run_phase_request
from goldenexperience.runtime.kv_baseline.services import (
    ProcessGroup,
    validate_runtime_requirements,
)
from scripts.run_cross_model_runtime import _cache_dir_summary, _log_evidence, _metrics, _timing


def _env_bool(name: str, default: str = "0") -> bool:
    return os.environ.get(name, default).lower() in {"1", "true", "yes", "on"}


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _phase_request(config: BaselineConfig, phase: str) -> dict[str, Any]:
    return _json(config.request_dir / f"{phase}.json")


def _default_run_id() -> str:
    return time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())


def _set_default_run_env() -> None:
    run_id = os.environ.get("GE_RUN_ID") or _default_run_id()
    os.environ.setdefault("GE_RUN_ID", run_id)
    os.environ.setdefault(
        "GE_RUN_DIR",
        str(REPO_ROOT / "artifacts" / "cross_model_runtime" / f"qwen3_8b_to_14b_hidden_bridge_{run_id}"),
    )


def _update_metadata(
    config: BaselineConfig,
    source: BaselineConfig,
    target: BaselineConfig,
    external_index_path: Path,
) -> None:
    path = config.run_dir / "metadata.json"
    metadata = _json(path)
    metadata["cross_model_runtime"] = {
        "scenario": "same_model_different_parameter_size",
        "source_model_path": source.model_path,
        "source_model_name": source.model_name,
        "target_model_path": target.model_path,
        "target_model_name": target.model_name,
        "materializer_mode": "hidden_bridge",
        "external_index_path": str(external_index_path),
        "quality_gate_required": True,
        "allow_unsafe": _env_bool("GE_ALLOW_UNSAFE_CROSS_MODEL_REUSE", "0"),
    }
    _write_json(path, metadata)


def _candidate_keys(model_name: str, chunk_hashes: list[str]) -> list[str]:
    return [object_key_string(model_name=model_name, chunk_hash=item) for item in chunk_hashes]


def _lookup_source_candidate(config: BaselineConfig, source_config: BaselineConfig) -> dict[str, Any]:
    records = load_lookup_hash_records(config.lmcache_mp_lookup_hash_log_dir)
    candidate = select_lookup_candidate(records, model_name=source_config.model_name)
    if candidate is None:
        return {
            "success": False,
            "fallback_reason": "source_candidate_not_found",
            "records_scanned": len(records),
        }
    chunk_hashes = [str(item) for item in candidate.get("chunk_hashes", [])]
    source_keys = _candidate_keys(source_config.model_name, chunk_hashes)
    return {
        "success": True,
        "records_scanned": len(records),
        "record": candidate,
        "chunk_hashes": chunk_hashes,
        "source_key_strings": source_keys,
    }


def _run_materializer(
    *,
    config: BaselineConfig,
    source_config: BaselineConfig,
    target_config: BaselineConfig,
    candidate: dict[str, Any],
    token_ids: list[int],
    external_index_path: Path,
) -> dict[str, Any]:
    request_path = config.run_dir / "materializer_request.json"
    output_path = config.run_dir / "materializer_result.json"
    request = {
        "source_model_path": source_config.model_path,
        "target_model_path": target_config.model_path,
        "target_model_name": target_config.model_name,
        "token_ids": token_ids,
        "chunk_hashes": candidate["chunk_hashes"],
        "chunk_size": config.chunk_size,
        "max_chunks": int(os.environ.get("GE_MATERIALIZER_MAX_CHUNKS", "0")),
        "output_dir": str(config.run_dir / "materialized_chunks"),
        "bridge_summary_path": os.environ.get(
            "GE_BRIDGE_SUMMARY_PATH",
            str(REPO_ROOT / "artifacts" / "hidden_bridge" / "qwen3_hidden_bridge_qwen3_8b_to_14b_prefix32.json"),
        ),
        "bridge_weights_path": os.environ.get(
            "GE_BRIDGE_WEIGHTS_PATH",
            str(REPO_ROOT / "artifacts" / "hidden_bridge" / "qwen3_hidden_bridge_qwen3_8b_to_14b_prefix32.pt"),
        ),
        "allow_unsafe": _env_bool("GE_ALLOW_UNSAFE_CROSS_MODEL_REUSE", "0"),
        "device": os.environ.get("GE_MATERIALIZER_DEVICE", "cuda:0"),
        "inject_to_mooncake": True,
        "mooncake_setup_config": config.l2_adapter(),
        "external_index_path": str(external_index_path),
    }
    _write_json(request_path, request)
    started = time.perf_counter()
    completed = subprocess.run(
        [
            config.python_bin,
            "-m",
            "goldenexperience.runtime.cross_model_materializer",
            "--request",
            str(request_path),
            "--output",
            str(output_path),
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    result = _json(output_path)
    result["process"] = {
        "returncode": completed.returncode,
        "stdout": completed.stdout[-4000:],
        "stderr": completed.stderr[-4000:],
        "elapsed_ms": (time.perf_counter() - started) * 1000,
        "request_path": str(request_path),
        "output_path": str(output_path),
    }
    _write_json(output_path, result)
    return result


def _summarize(
    *,
    config: BaselineConfig,
    source_config: BaselineConfig,
    target_config: BaselineConfig,
    phases: list[str],
    lookup: dict[str, Any],
    materializer: dict[str, Any],
    tokenization: dict[str, Any],
    target_key_status_before: dict[str, Any],
    source_key_status: dict[str, Any],
    target_key_status_after_materializer: dict[str, Any] | None,
    external_index_path: Path,
) -> dict[str, Any]:
    requests = {phase: _phase_request(config, phase) for phase in phases}
    metrics = {phase: _metrics(config.metrics_dir / f"{phase}.prom") for phase in phases}
    logs = {
        phase: _log_evidence(config.log_dir / f"{phase}_lmcache_mp_server.log")
        for phase in phases
    }
    target_phase = phases[-1]
    target_hits = metrics[target_phase].get("external_prefix_cache_hits_total")
    target_external_tokens = metrics[target_phase].get("prompt_tokens_external_kv_transfer")
    target_local_tokens = metrics[target_phase].get("prompt_tokens_local_compute")
    materializer_injected = bool(materializer.get("injected"))
    consumed_materialized = bool(
        materializer_injected
        and isinstance(target_external_tokens, (int, float))
        and target_external_tokens > 0
    )
    fallback = not consumed_materialized
    summary = {
        "schema_version": "goldenexperience.cross_model_hidden_bridge_runtime.v1",
        "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "run_id": config.run_id,
        "run_dir": str(config.run_dir),
        "scenario": "qwen3_8b_to_14b_cross_parameter_kv_reuse",
        "source": {
            "model_path": source_config.model_path,
            "model_name": source_config.model_name,
        },
        "target": {
            "model_path": target_config.model_path,
            "model_name": target_config.model_name,
        },
        "runtime": {
            "engine": config.engine,
            "kv_backend": config.kv_backend,
            "l2_adapter_type": config.lmcache_mp_l2_adapter_type,
            "mooncake_enabled": config.use_mooncake_store,
            "chunk_size": config.chunk_size,
            "hash_algorithm": config.hash_algorithm,
            "external_index_path": str(external_index_path),
        },
        "lmcache_cross_model_lookup": {
            "source_candidate": lookup,
            "source_key_status": source_key_status,
            "target_key_status_before_materializer": target_key_status_before,
        },
        "tokenization": tokenization,
        "goldenexperience_materializer": materializer,
        "target_key_status_after_materializer": target_key_status_after_materializer,
        "policy": {
            "quality_gate_required": True,
            "allow_unsafe": _env_bool("GE_ALLOW_UNSAFE_CROSS_MODEL_REUSE", "0"),
            "automatic_reuse_allowed": materializer_injected and not materializer.get("allow_unsafe", False),
        },
        "result": {
            "materializer_injected": materializer_injected,
            "vllm_consumed_materialized_kv": consumed_materialized,
            "fallback_used": fallback,
            "success": consumed_materialized and not materializer.get("allow_unsafe", False),
            "status": (
                "cross_model_reuse_success"
                if consumed_materialized and not materializer.get("allow_unsafe", False)
                else "unsafe_shadow_reuse"
                if consumed_materialized
                else "fallback"
            ),
        },
        "phases": phases,
        "requests": requests,
        "metrics": metrics,
        "logs": logs,
        "cache": _cache_dir_summary(config.kv_cache_dir),
        "mooncake_storage": _cache_dir_summary(config.mooncake_storage_root),
        "evidence": {
            "target_phase": target_phase,
            "target_external_prefix_cache_hits_total": target_hits,
            "target_prompt_tokens_external_kv_transfer": target_external_tokens,
            "target_prompt_tokens_local_compute": target_local_tokens,
            "target_mooncake_get_events": int(logs[target_phase].get("mooncake_store_get") or 0),
            "target_mooncake_set_events": int(logs[target_phase].get("mooncake_store_set") or 0),
        },
        "timing": {
            "source_offload_ttft_ms": _timing(requests.get("source_offload", {}), "ttft_ms"),
            "target_phase_ttft_ms": _timing(requests.get(target_phase, {}), "ttft_ms"),
            "target_phase_e2e_ms": _timing(requests.get(target_phase, {}), "e2e_ms"),
        },
    }
    output = config.run_dir / "cross_model_hidden_bridge_summary.json"
    _write_json(output, summary)
    manifest = REPO_ROOT / "artifacts" / "cross_model_runtime" / "manifests" / f"{config.run_id}.json"
    _write_json(manifest, summary)
    return summary


def main() -> int:
    _set_default_run_env()
    source_model_path = os.environ.get(
        "GE_SOURCE_MODEL_PATH", "/workspace/volume/softdata/models/Qwen3-8B"
    )
    target_model_path = os.environ.get(
        "GE_TARGET_MODEL_PATH", "/workspace/volume/softdata/models/Qwen3-14B"
    )
    source_model_name = os.environ.get("GE_SOURCE_MODEL_NAME", source_model_path)
    target_model_name = os.environ.get("GE_TARGET_MODEL_NAME", target_model_path)

    os.environ["GE_MODEL_PATH"] = source_model_path
    os.environ["GE_MODEL_NAME"] = source_model_name
    config = BaselineConfig.from_env(sys.argv[1:])
    source_config = replace(config, model_path=source_model_path, model_name=source_model_name)
    target_config = replace(config, model_path=target_model_path, model_name=target_model_name)
    external_index_path = Path(
        os.environ.get("GE_MOONCAKE_EXTERNAL_INDEX", str(config.run_dir / "mooncake_external_index.jsonl"))
    )
    os.environ["GE_MOONCAKE_EXTERNAL_INDEX"] = str(external_index_path)

    config.ensure_dirs()
    write_generated_disk_prompt(config)
    config.write_lmcache_config()
    config.write_metadata()
    _update_metadata(config, source_config, target_config, external_index_path)

    print(f"Cross-model hidden-bridge run directory: {config.run_dir}")
    print(f"Source: {source_model_path}")
    print(f"Target: {target_model_path}")
    print(f"External Mooncake index: {external_index_path}")
    print(f"Allow unsafe reuse: {int(_env_bool('GE_ALLOW_UNSAFE_CROSS_MODEL_REUSE', '0'))}")

    if config.dry_run:
        print("GE_DRY_RUN=1: generated config and metadata only.")
        return 0

    validate_runtime_requirements(config)
    processes = ProcessGroup(config)
    phases = ["source_offload"]
    target_phase = "target_fallback"
    try:
        processes.start_mooncake_services()
        processes.wait_for_mooncake_ready()
        processes.start_lmcache_mp_server()
        processes.wait_for_lmcache_mp_ready()

        processes.config = source_config
        processes.start_engine_server("source_offload")
        processes.wait_for_engine_ready("source_offload")
        _run_phase_request(source_config, "source_offload")
        processes.stop_server("source_offload")
        time.sleep(float(os.environ.get("GE_AFTER_ENGINE_STOP_SLEEP_SEC", "10")))

        lookup = _lookup_source_candidate(config, source_config)
        source_key_status = {"skipped": True}
        target_key_status_before = {"skipped": True}
        target_key_status_after_materializer = None
        tokenization: dict[str, Any] = {"success": False}
        materializer: dict[str, Any] = {
            "success": False,
            "fallback_reason": lookup.get("fallback_reason", "not_started"),
            "injected": False,
        }
        if lookup.get("success"):
            chunk_hashes = lookup["chunk_hashes"]
            source_key_status = mooncake_key_exists(
                setup_config=config.l2_adapter(),
                key_strings=lookup["source_key_strings"],
            )
            target_keys = _candidate_keys(target_config.model_name, chunk_hashes)
            target_key_status_before = mooncake_key_exists(
                setup_config=config.l2_adapter(),
                key_strings=target_keys,
            )
            token_ids = token_ids_from_prompt(
                tokenizer_path=target_config.model_path,
                prompt_file=config.prompt_file,
                prompt_id=config.prompt_id,
            )
            seq_len = int(lookup["record"].get("seq_len") or 0)
            tokenization = {
                "success": len(token_ids) == seq_len,
                "token_count": len(token_ids),
                "lookup_seq_len": seq_len,
            }
            if len(token_ids) == seq_len:
                materializer = _run_materializer(
                    config=config,
                    source_config=source_config,
                    target_config=target_config,
                    candidate=lookup,
                    token_ids=token_ids,
                    external_index_path=external_index_path,
                )
                target_key_status_after_materializer = mooncake_key_exists(
                    setup_config=config.l2_adapter(),
                    key_strings=target_keys,
                )
            else:
                materializer = {
                    "success": False,
                    "fallback_reason": "tokenization_mismatch",
                    "injected": False,
                }

        if materializer.get("injected"):
            target_phase = "target_reuse"
        phases.append(target_phase)
        processes.config = target_config
        processes.start_engine_server(target_phase)
        processes.wait_for_engine_ready(target_phase)
        _run_phase_request(target_config, target_phase)
        processes.stop_server(target_phase)

        summary = _summarize(
            config=config,
            source_config=source_config,
            target_config=target_config,
            phases=phases,
            lookup=lookup,
            materializer=materializer,
            tokenization=tokenization,
            target_key_status_before=target_key_status_before,
            source_key_status=source_key_status,
            target_key_status_after_materializer=target_key_status_after_materializer,
            external_index_path=external_index_path,
        )
        print(f"Result: {summary['result']['status']}")
    finally:
        processes.config = config
        processes.stop_server("active")
        processes.stop_lmcache_mp()
        processes.stop_mooncake()

    print("Done. Key outputs:")
    for relative in [
        "metadata.json",
        "materializer_request.json",
        "materializer_result.json",
        "requests/source_offload.json",
        f"requests/{target_phase}.json",
        "cross_model_hidden_bridge_summary.json",
    ]:
        print(f"  {config.run_dir / relative}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
