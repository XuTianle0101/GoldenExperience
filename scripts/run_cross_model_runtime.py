#!/usr/bin/env python3
"""Run a cross-size vLLM + LMCache MP runtime proof.

This is an end-to-end runtime harness, not an offline planner smoke test. It keeps
one LMCache MP + Mooncake L2 service alive while sequentially starting source and
target vLLM servers.

The current materializer mode is intentionally explicit:

* ``native_target_seed`` uses a target-model prefill to create target-shaped KV
  entries in the shared LMCache runtime, then verifies that a fresh target vLLM
  process retrieves those entries. This proves the cross-model runtime plumbing
  and target-shaped cache reuse path without pretending the hidden bridge is
  production-ready.
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
from contextlib import suppress
from dataclasses import replace
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from goldenexperience.runtime.kv_baseline.config import BaselineConfig  # noqa: E402
from goldenexperience.runtime.kv_baseline.prompts import (  # noqa: E402
    write_generated_disk_prompt,
)
from goldenexperience.runtime.kv_baseline.runner import (  # noqa: E402
    _run_phase_request,
)
from goldenexperience.runtime.kv_baseline.services import (  # noqa: E402
    ProcessGroup,
    validate_runtime_requirements,
)


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _phase_request(config: BaselineConfig, phase: str) -> dict[str, Any]:
    return _json(config.request_dir / f"{phase}.json")


def _metric_values(text: str, metric_name: str) -> list[float]:
    number = r"([-+]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][-+]?[0-9]+)?|NaN|Inf|-Inf)"
    pattern = re.compile(
        rf"^{re.escape(metric_name)}(?:\{{[^}}]*\}})?\s+{number}\s*$",
        flags=re.MULTILINE,
    )
    values: list[float] = []
    for match in pattern.finditer(text):
        with suppress(ValueError):
            values.append(float(match.group(1)))
    return values


def _metrics(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"path": str(path), "exists": False}
    text = path.read_text(encoding="utf-8", errors="replace")

    def prompt_tokens_by_source(source: str) -> float:
        number = (
            r"([-+]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)"
            r"(?:[eE][-+]?[0-9]+)?|NaN|Inf|-Inf)"
        )
        pattern = re.compile(
            rf'^vllm:prompt_tokens_by_source_total\{{[^}}]*source="{re.escape(source)}"[^}}]*\}}\s+{number}\s*$',
            flags=re.MULTILINE,
        )
        total = 0.0
        for match in pattern.finditer(text):
            with suppress(ValueError):
                total += float(match.group(1))
        return total

    return {
        "path": str(path),
        "exists": True,
        "external_prefix_cache_hits_total": max(
            _metric_values(text, "vllm:external_prefix_cache_hits_total") or [0.0]
        ),
        "prompt_tokens_external_kv_transfer": prompt_tokens_by_source("external_kv_transfer"),
        "prompt_tokens_local_compute": prompt_tokens_by_source("local_compute"),
        "cache_hit_rate_max": max(_metric_values(text, "vllm:cache_hit_rate") or [0.0]),
        "prompt_tokens_histogram_sum": sum(
            _metric_values(text, "vllm:prompt_tokens_histogram_sum")
        ),
        "uncached_prompt_tokens_histogram_sum": sum(
            _metric_values(text, "vllm:uncached_prompt_tokens_histogram_sum")
        ),
    }


def _count(pattern: str, text: str) -> int:
    return len(re.findall(pattern, text, flags=re.IGNORECASE))


def _log_evidence(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"path": str(path), "exists": False}
    text = path.read_text(encoding="utf-8", errors="replace")
    return {
        "path": str(path),
        "exists": True,
        "bytes": path.stat().st_size,
        "mooncake_store_set": _count(r"\bSET\b|\bmooncake[_ -]?store\b.*\bset\b", text),
        "mooncake_store_get": _count(r"\bGET\b|\bmooncake[_ -]?store\b.*\bget\b", text),
        "l2_prefetch_load_completed": _count(
            r"L2.*prefetch.*complete|prefetch.*load.*complete", text
        ),
        "retrieve_mentions": _count(
            r"\b(retrieve|retrieved|prefetch|prefetched|loaded|hit)\b", text
        ),
        "store_mentions": _count(r"\b(store|stored|offload|saved|SET)\b", text),
    }


def _cache_dir_summary(path: Path) -> dict[str, Any]:
    files: list[Path] = []
    total = 0
    if path.exists():
        for item in path.rglob("*"):
            if item.is_file():
                files.append(item)
                total += item.stat().st_size
    return {
        "path": str(path),
        "exists": path.exists(),
        "file_count": len(files),
        "total_bytes": total,
        "sample_files": [str(item) for item in sorted(files)[:10]],
    }


def _timing(request: dict[str, Any], key: str) -> float | None:
    value = request.get("timing", {}).get(key)
    return float(value) if isinstance(value, (int, float)) else None


def _write_cross_summary(
    config: BaselineConfig,
    source_config: BaselineConfig,
    target_config: BaselineConfig,
    phases: list[str],
) -> dict[str, Any]:
    requests = {phase: _phase_request(config, phase) for phase in phases}
    metrics = {phase: _metrics(config.metrics_dir / f"{phase}.prom") for phase in phases}
    logs = {
        phase: _log_evidence(config.log_dir / f"{phase}_lmcache_mp_server.log") for phase in phases
    }
    cache = _cache_dir_summary(config.kv_cache_dir)
    mooncake = _cache_dir_summary(config.mooncake_storage_root)

    target_materialize_ttft = _timing(requests["target_materialize"], "ttft_ms")
    target_reuse_ttft = _timing(requests["target_reuse"], "ttft_ms")
    target_reuse_gets = int(logs["target_reuse"].get("mooncake_store_get") or 0)
    target_reuse_retrieves = int(logs["target_reuse"].get("retrieve_mentions") or 0)
    target_reuse_external_hits = metrics["target_reuse"].get("external_prefix_cache_hits_total")
    target_reuse_has_cache_evidence = bool(
        target_reuse_gets > 0
        or target_reuse_retrieves > 0
        or (isinstance(target_reuse_external_hits, (int, float)) and target_reuse_external_hits > 0)
    )

    summary = {
        "schema_version": "goldenexperience.cross_model_runtime.v1",
        "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "run_id": config.run_id,
        "run_dir": str(config.run_dir),
        "scenario": "same_model_different_parameter_size",
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
            "lmcache_config_file": str(config.config_file),
            "vllm_kv_transfer_config": config.vllm_kv_transfer_config(),
        },
        "materializer": {
            "mode": "native_target_seed",
            "status": "runtime_proof_only",
            "note": (
                "Target-shaped KV is seeded by a target-model prefill, then reused "
                "by a fresh target vLLM process. Hidden-bridge KV injection remains "
                "a separate quality-gated implementation step."
            ),
        },
        "phases": phases,
        "requests": requests,
        "metrics": metrics,
        "logs": logs,
        "cache": cache,
        "mooncake_storage": mooncake,
        "evidence": {
            "source_offload_store_events": int(
                logs["source_offload"].get("mooncake_store_set") or 0
            ),
            "target_materialize_store_events": int(
                logs["target_materialize"].get("mooncake_store_set") or 0
            ),
            "target_reuse_get_events": target_reuse_gets,
            "target_reuse_retrieve_mentions": target_reuse_retrieves,
            "target_reuse_external_prefix_cache_hits_total": target_reuse_external_hits,
            "target_reuse_has_cache_evidence": target_reuse_has_cache_evidence,
            "mooncake_storage_has_files": bool(
                mooncake["file_count"] > 0 and mooncake["total_bytes"] > 0
            ),
        },
        "deltas": {
            "target_reuse_minus_materialize_ttft_ms": (
                target_reuse_ttft - target_materialize_ttft
                if target_reuse_ttft is not None and target_materialize_ttft is not None
                else None
            ),
            "target_reuse_minus_materialize_e2e_ms": (
                _timing(requests["target_reuse"], "e2e_ms")
                - _timing(requests["target_materialize"], "e2e_ms")
                if _timing(requests["target_reuse"], "e2e_ms") is not None
                and _timing(requests["target_materialize"], "e2e_ms") is not None
                else None
            ),
        },
    }
    output = config.run_dir / "cross_model_runtime_summary.json"
    output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def _update_metadata(
    config: BaselineConfig, source: BaselineConfig, target: BaselineConfig
) -> None:
    path = config.run_dir / "metadata.json"
    metadata = _json(path)
    metadata["cross_model_runtime"] = {
        "scenario": "same_model_different_parameter_size",
        "source_model_path": source.model_path,
        "source_model_name": source.model_name,
        "target_model_path": target.model_path,
        "target_model_name": target.model_name,
        "materializer_mode": "native_target_seed",
    }
    path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    source_model_path = os.environ.get(
        "GE_SOURCE_MODEL_PATH", "/workspace/volume/softdata/models/Qwen3-8B"
    )
    target_model_path = os.environ.get(
        "GE_TARGET_MODEL_PATH", "/workspace/volume/softdata/models/Qwen3-14B"
    )
    source_model_name = os.environ.get("GE_SOURCE_MODEL_NAME", source_model_path)
    target_model_name = os.environ.get("GE_TARGET_MODEL_NAME", target_model_path)

    # Build the shared runtime config from the source model. Individual vLLM
    # phases use dataclass copies with source/target model paths.
    os.environ["GE_MODEL_PATH"] = source_model_path
    os.environ["GE_MODEL_NAME"] = source_model_name
    config = BaselineConfig.from_env(sys.argv[1:])
    source_config = replace(config, model_path=source_model_path, model_name=source_model_name)
    target_config = replace(config, model_path=target_model_path, model_name=target_model_name)

    config.ensure_dirs()
    write_generated_disk_prompt(config)
    config.write_lmcache_config()
    config.write_metadata()
    _update_metadata(config, source_config, target_config)

    print(f"Cross-model runtime run directory: {config.run_dir}")
    print(f"Source: {source_model_path}")
    print(f"Target: {target_model_path}")
    print("Materializer mode: native_target_seed")
    print(f"LMCache config: {config.config_file}")

    if config.dry_run:
        print("GE_DRY_RUN=1: generated config and metadata only.")
        return 0

    validate_runtime_requirements(config)
    processes = ProcessGroup(config)
    phases = ["source_offload", "target_materialize", "target_reuse"]
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
        processes.start_engine_server("target_materialize")
        processes.wait_for_engine_ready("target_materialize")
        _run_phase_request(target_config, "target_materialize")
        processes.stop_server("target_materialize")
        time.sleep(float(os.environ.get("GE_AFTER_ENGINE_STOP_SLEEP_SEC", "10")))

        processes.config = target_config
        processes.start_engine_server("target_reuse")
        processes.wait_for_engine_ready("target_reuse")
        _run_phase_request(target_config, "target_reuse")
        processes.stop_server("target_reuse")

        summary = _write_cross_summary(config, source_config, target_config, phases)
        if not summary["evidence"]["target_reuse_has_cache_evidence"]:
            raise RuntimeError(
                "target_reuse did not show LMCache reuse evidence; see "
                f"{config.run_dir / 'cross_model_runtime_summary.json'}"
            )
    finally:
        # Make teardown deterministic for this E2E proof.
        processes.config = config
        processes.stop_server("active")
        processes.stop_lmcache_mp()
        processes.stop_mooncake()

    print("Done. Key outputs:")
    for relative in [
        "metadata.json",
        "lmc_config.yaml",
        "requests/source_offload.json",
        "requests/target_materialize.json",
        "requests/target_reuse.json",
        "cross_model_runtime_summary.json",
    ]:
        print(f"  {config.run_dir / relative}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
