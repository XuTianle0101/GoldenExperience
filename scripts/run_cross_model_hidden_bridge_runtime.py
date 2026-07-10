#!/usr/bin/env python3
"""Run quality-gated Qwen3 cached-KV translation against vLLM/LMCache."""

# ruff: noqa: E402

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
    evaluate_runtime_reuse,
    load_lookup_hash_records,
    mooncake_key_exists,
    object_key_string,
    select_shared_prefix_candidate,
    token_ids_from_prompt,
    token_ids_sha256,
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


def _env_path(name: str, default: Path) -> Path:
    path = Path(os.environ.get(name, str(default)))
    return path if path.is_absolute() else REPO_ROOT / path


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
    direction = os.environ.get("GE_CACHED_KV_DIRECTION", "8b_to_14b")
    os.environ.setdefault("GE_RUN_ID", run_id)
    os.environ.setdefault(
        "GE_RUN_DIR",
        str(
            REPO_ROOT
            / "artifacts"
            / "cross_model_runtime"
            / f"qwen3_cached_kv_{direction}_{run_id}"
        ),
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
        "source_prompt_id": source.prompt_id,
        "source_prompt_file": str(source.prompt_file),
        "target_prompt_id": target.prompt_id,
        "target_prompt_file": str(target.prompt_file),
        "materializer_mode": "cached_kv",
        "external_index_path": str(external_index_path),
        "quality_gate_required": True,
        "allow_unsafe": _env_bool("GE_ALLOW_UNSAFE_CROSS_MODEL_REUSE", "0"),
    }
    _write_json(path, metadata)


def _candidate_keys(model_name: str, chunk_hashes: list[str]) -> list[str]:
    return [object_key_string(model_name=model_name, chunk_hash=item) for item in chunk_hashes]


def _lookup_source_candidate(
    config: BaselineConfig,
    source_config: BaselineConfig,
    source_token_ids: list[int],
    target_token_ids: list[int],
) -> dict[str, Any]:
    records = load_lookup_hash_records(config.lmcache_mp_lookup_hash_log_dir)
    source_request = _phase_request(config, "source_offload")
    source_response_id = source_request.get("response", {}).get("id")
    if not isinstance(source_response_id, str) or not source_response_id:
        return {
            "success": False,
            "fallback_reason": "source_request_identity_missing",
            "records_scanned": len(records),
        }
    try:
        candidate = select_shared_prefix_candidate(
            records,
            model_name=source_config.model_name,
            source_request_id=source_response_id,
            source_token_ids=source_token_ids,
            target_token_ids=target_token_ids,
            chunk_size=config.chunk_size,
            hash_algorithm=config.hash_algorithm,
        )
    except ValueError as exc:
        return {
            "success": False,
            "fallback_reason": "deterministic_prompt_hashing_unavailable",
            "records_scanned": len(records),
            "error": str(exc),
        }
    if candidate is None:
        return {
            "success": False,
            "fallback_reason": "prompt_bound_shared_prefix_not_found",
            "records_scanned": len(records),
            "source_response_id": source_response_id,
            "source_token_ids_sha256": token_ids_sha256(source_token_ids),
            "target_token_ids_sha256": token_ids_sha256(target_token_ids),
        }
    chunk_hashes = candidate["chunk_hashes"]
    source_keys = _candidate_keys(source_config.model_name, chunk_hashes)
    return {
        "success": True,
        "records_scanned": len(records),
        "record": candidate["record"],
        "chunk_hashes": chunk_hashes,
        "source_key_strings": source_keys,
        "binding": candidate["binding"],
    }


def _run_materializer(
    *,
    config: BaselineConfig,
    source_config: BaselineConfig,
    target_config: BaselineConfig,
    candidate: dict[str, Any],
    external_index_path: Path,
) -> dict[str, Any]:
    request_path = config.run_dir / "materializer_request.json"
    output_path = config.run_dir / "materializer_result.json"
    native_target = _phase_request(config, "target_native")
    native_target_prefill_ms = native_target.get("timing", {}).get("ttft_ms")
    direction = os.environ.get("GE_CACHED_KV_DIRECTION", "8b_to_14b")
    request = {
        "mode": "cached_kv",
        "source_model_path": source_config.model_path,
        "source_model_name": source_config.model_name,
        "target_model_path": target_config.model_path,
        "target_model_name": target_config.model_name,
        "chunk_hashes": candidate["chunk_hashes"],
        "source_lookup_record": candidate["record"],
        "prompt_binding": candidate["binding"],
        "hash_algorithm": config.hash_algorithm,
        "chunk_size": config.chunk_size,
        "max_chunks": int(os.environ.get("GE_MATERIALIZER_MAX_CHUNKS", "0")),
        "bridge_manifest_path": os.environ.get(
            "GE_CACHED_KV_BRIDGE_MANIFEST",
            str(REPO_ROOT / "artifacts" / "cached_kv" / f"qwen3_{direction}.json"),
        ),
        "direction": direction,
        "allow_unsafe": _env_bool("GE_ALLOW_UNSAFE_CROSS_MODEL_REUSE", "0"),
        "device": os.environ.get("GE_MATERIALIZER_DEVICE", "cuda:0"),
        "world_size": int(os.environ.get("GE_TENSOR_PARALLEL_SIZE", "1")),
        "cache_salt": os.environ.get("GE_CACHE_SALT", os.environ.get("LMCACHE_CACHE_SALT", "")),
        "native_target_prefill_ms": native_target_prefill_ms,
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
        phase: _log_evidence(config.log_dir / f"{phase}_lmcache_mp_server.log") for phase in phases
    }
    target_phase = phases[-1]
    target_hits = metrics[target_phase].get("external_prefix_cache_hits_total")
    target_external_tokens = metrics[target_phase].get("prompt_tokens_external_kv_transfer")
    target_local_tokens = metrics[target_phase].get("prompt_tokens_local_compute")
    materializer_injected = bool(materializer.get("injected"))
    target_keys = _candidate_keys(target_config.model_name, lookup.get("chunk_hashes", []))
    validation = evaluate_runtime_reuse(
        materializer=materializer,
        target_key_strings=target_keys,
        source_key_status=source_key_status,
        target_key_status_before=target_key_status_before,
        target_key_status_after=target_key_status_after_materializer,
        target_external_tokens=target_external_tokens,
        chunk_size=config.chunk_size,
        native_request=requests.get("target_native", {}),
        reuse_request=requests.get(target_phase, {}),
    )
    consumed_materialized = validation["consumed_materialized_kv"]
    fallback = target_phase == "target_fallback"
    direction = str(
        materializer.get("direction") or os.environ.get("GE_CACHED_KV_DIRECTION", "8b_to_14b")
    )
    summary = {
        "schema_version": "goldenexperience.cross_model_cached_kv_runtime.v1",
        "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "run_id": config.run_id,
        "run_dir": str(config.run_dir),
        "scenario": f"qwen3_{direction}_cross_parameter_kv_reuse",
        "direction": direction,
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
            "automatic_reuse_allowed": validation["success"],
        },
        "result": {
            "materializer_injected": materializer_injected,
            "vllm_consumed_materialized_kv": consumed_materialized,
            "fallback_used": fallback,
            "success": validation["success"],
            "status": validation["status"],
            "validation": validation,
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
            "target_native_ttft_ms": _timing(requests.get("target_native", {}), "ttft_ms"),
            "target_native_e2e_ms": _timing(requests.get("target_native", {}), "e2e_ms"),
        },
    }
    output = config.run_dir / "cross_model_cached_kv_summary.json"
    _write_json(output, summary)
    manifest = (
        REPO_ROOT / "artifacts" / "cross_model_runtime" / "manifests" / f"{config.run_id}.json"
    )
    _write_json(manifest, summary)
    return summary


def main() -> int:
    _set_default_run_env()
    direction = os.environ.get("GE_CACHED_KV_DIRECTION", "8b_to_14b")
    if direction == "8b_to_14b":
        default_source = "/workspace/volume/softdata/models/Qwen3-8B"
        default_target = "/workspace/volume/softdata/models/Qwen3-14B"
    elif direction == "14b_to_8b":
        default_source = "/workspace/volume/softdata/models/Qwen3-14B"
        default_target = "/workspace/volume/softdata/models/Qwen3-8B"
    else:
        raise ValueError("GE_CACHED_KV_DIRECTION must be 8b_to_14b or 14b_to_8b")
    source_model_path = os.environ.get("GE_SOURCE_MODEL_PATH", default_source)
    target_model_path = os.environ.get("GE_TARGET_MODEL_PATH", default_target)
    source_model_name = os.environ.get("GE_SOURCE_MODEL_NAME", source_model_path)
    target_model_name = os.environ.get("GE_TARGET_MODEL_NAME", target_model_path)

    os.environ["GE_MODEL_PATH"] = source_model_path
    os.environ["GE_MODEL_NAME"] = source_model_name
    config = BaselineConfig.from_env(sys.argv[1:])
    source_config = replace(
        config,
        model_path=source_model_path,
        model_name=source_model_name,
        prompt_file=_env_path("GE_SOURCE_PROMPT_FILE", config.prompt_file),
        prompt_file_was_set=True,
        prompt_id=os.environ.get("GE_SOURCE_PROMPT_ID", config.prompt_id),
    )
    target_config = replace(
        config,
        model_path=target_model_path,
        model_name=target_model_name,
        prompt_file=_env_path("GE_TARGET_PROMPT_FILE", config.prompt_file),
        prompt_file_was_set=True,
        prompt_id=os.environ.get("GE_TARGET_PROMPT_ID", config.prompt_id),
    )
    external_index_path = Path(
        os.environ.get(
            "GE_MOONCAKE_EXTERNAL_INDEX", str(config.run_dir / "mooncake_external_index.jsonl")
        )
    )
    os.environ["GE_MOONCAKE_EXTERNAL_INDEX"] = str(external_index_path)

    config.ensure_dirs()
    write_generated_disk_prompt(config)
    config.write_lmcache_config()
    config.write_metadata()
    _update_metadata(config, source_config, target_config, external_index_path)

    print(f"Cross-model cached-KV run directory: {config.run_dir}")
    print(f"Source: {source_model_path}")
    print(f"Target: {target_model_path}")
    print(f"Source prompt: {source_config.prompt_file}#{source_config.prompt_id}")
    print(f"Target prompt: {target_config.prompt_file}#{target_config.prompt_id}")
    print(f"External Mooncake index: {external_index_path}")
    print(f"Allow unsafe reuse: {int(_env_bool('GE_ALLOW_UNSAFE_CROSS_MODEL_REUSE', '0'))}")

    if config.dry_run:
        print("GE_DRY_RUN=1: generated config and metadata only.")
        return 0

    validate_runtime_requirements(config)
    processes = ProcessGroup(config)
    phases = ["source_offload", "target_native"]
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

        processes.config = target_config
        processes.start_engine_server("target_native", use_kv_transfer=False)
        processes.wait_for_engine_ready("target_native")
        _run_phase_request(target_config, "target_native")
        processes.stop_server("target_native")
        time.sleep(float(os.environ.get("GE_AFTER_ENGINE_STOP_SLEEP_SEC", "10")))

        try:
            source_token_ids = token_ids_from_prompt(
                tokenizer_path=source_config.model_path,
                prompt_file=source_config.prompt_file,
                prompt_id=source_config.prompt_id,
            )
            target_token_ids = token_ids_from_prompt(
                tokenizer_path=target_config.model_path,
                prompt_file=target_config.prompt_file,
                prompt_id=target_config.prompt_id,
            )
            tokenization: dict[str, Any] = {
                "success": True,
                "source": {
                    "token_count": len(source_token_ids),
                    "token_ids_sha256": token_ids_sha256(source_token_ids),
                },
                "target": {
                    "token_count": len(target_token_ids),
                    "token_ids_sha256": token_ids_sha256(target_token_ids),
                },
            }
        except Exception as exc:  # noqa: BLE001 - runtime fallback boundary.
            source_token_ids = []
            target_token_ids = []
            tokenization = {
                "success": False,
                "error": repr(exc),
            }
        lookup = (
            _lookup_source_candidate(
                config,
                source_config,
                source_token_ids,
                target_token_ids,
            )
            if tokenization["success"]
            else {"success": False, "fallback_reason": "tokenization_failed"}
        )
        source_key_status = {"skipped": True}
        target_key_status_before = {"skipped": True}
        target_key_status_after_materializer = None
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
            seq_len = int(lookup["record"].get("seq_len") or 0)
            shared_prefix_tokens = len(chunk_hashes) * config.chunk_size
            tokenization["lookup_seq_len"] = seq_len
            tokenization["shared_prefix_tokens"] = shared_prefix_tokens
            tokenization["success"] = (
                len(source_token_ids) == seq_len
                and len(target_token_ids) >= shared_prefix_tokens
            )
            if tokenization["success"]:
                materializer = _run_materializer(
                    config=config,
                    source_config=source_config,
                    target_config=target_config,
                    candidate=lookup,
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
        "requests/target_native.json",
        f"requests/{target_phase}.json",
        "cross_model_cached_kv_summary.json",
    ]:
        print(f"  {config.run_dir / relative}")
    if (
        summary["result"]["materializer_injected"]
        and not summary["result"]["success"]
        and not _env_bool("GE_ALLOW_UNSAFE_CROSS_MODEL_REUSE", "0")
    ):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
