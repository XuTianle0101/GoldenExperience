"""Non-publishing Mooncake cost benchmark for cached-KV bridge candidates."""

from __future__ import annotations

import hashlib
import json
import math
import os
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from goldenexperience.runtime.mooncake_objects import ExactMooncakeStore, MooncakeObjectError
from goldenexperience.size_variant.cached_kv_manifest import sha256_file

CACHED_KV_COST_SCHEMA_VERSION = "goldenexperience.cached_kv_cost.v1"
NATIVE_PREFILL_COST_SCHEMA_VERSION = "goldenexperience.native_prefill_cost.v1"
MIN_COST_SAMPLES = 20


@dataclass(frozen=True)
class NativePrefillEvidence:
    samples_ms: tuple[float, ...]
    backend: str
    eligible_for_approval: bool
    report_sha256: str


def run_cached_kv_cost_benchmark(
    bridge: Any,
    *,
    candidate_manifest_path: str | Path,
    setup_config: Mapping[str, Any],
    source_keys: Sequence[str],
    chunk_size: int,
    native_prefill_samples_ms: Sequence[float],
    iterations: int = MIN_COST_SAMPLES,
    warmup_iterations: int = 3,
    store_factory: Callable[[], Any] | None = None,
    native_prefill_backend: str = "unverified",
    native_prefill_eligible: bool = False,
    native_prefill_report_sha256: str | None = None,
) -> dict[str, Any]:
    """Measure exact read-transform-write and always remove temporary target objects."""

    import torch

    if iterations <= 0 or warmup_iterations < 0:
        raise ValueError("benchmark iteration counts are invalid")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    normalized_source_keys = [str(key) for key in source_keys]
    if not normalized_source_keys or any(not key for key in normalized_source_keys):
        raise ValueError("source_keys must be non-empty")
    if len(set(normalized_source_keys)) != len(normalized_source_keys):
        raise ValueError("source_keys must be unique")
    native_samples = _positive_finite_samples(native_prefill_samples_ms, "native prefill")
    manifest_path = Path(candidate_manifest_path).resolve()
    if not manifest_path.is_file():
        raise ValueError("candidate manifest does not exist")
    if bridge.manifest.approved:
        raise ValueError("cost benchmark expects an unapproved candidate artifact")
    if bridge.manifest.artifact_errors():
        raise ValueError("candidate artifact structure is invalid")

    source_dtype = _torch_dtype(bridge.manifest.source.dtype)
    target_dtype = _torch_dtype(bridge.manifest.target.dtype)
    source_shape = (
        2,
        bridge.manifest.source.num_layers,
        chunk_size,
        bridge.manifest.source.kv_width,
    )
    target_shape = (
        2,
        bridge.manifest.target.num_layers,
        chunk_size,
        bridge.manifest.target.kv_width,
    )
    source_bytes = math.prod(source_shape) * torch.empty((), dtype=source_dtype).element_size()
    target_bytes = math.prod(target_shape) * torch.empty((), dtype=target_dtype).element_size()
    read_samples: list[float] = []
    transform_samples: list[float] = []
    write_samples: list[float] = []
    operation_samples: list[float] = []
    rollback_failures: list[dict[str, int]] = []
    total_iterations = warmup_iterations + iterations
    run_nonce = f"{os.getpid()}-{time.time_ns()}"

    with ExactMooncakeStore(setup_config, store_factory=store_factory) as store:
        for iteration in range(total_iterations):
            target_keys = [
                f"ge-cost/{bridge.manifest.bridge_id}/{run_nonce}/{iteration}/{index}"
                for index in range(len(normalized_source_keys))
            ]
            operation_started = time.perf_counter()
            try:
                read_started = time.perf_counter()
                reads = store.read_many_exact(
                    normalized_source_keys,
                    [source_bytes] * len(normalized_source_keys),
                )
                read_ms = (time.perf_counter() - read_started) * 1000

                transform_started = time.perf_counter()
                target_payloads: list[bytearray] = []
                for chunk_index, read in enumerate(reads):
                    source_object = (
                        torch.frombuffer(read.data, dtype=source_dtype)
                        .reshape(source_shape)
                        .clone()
                    )
                    target_object = bridge.transform(
                        source_object,
                        position_start=chunk_index * chunk_size,
                    )
                    if tuple(target_object.shape) != target_shape:
                        raise ValueError("bridge output shape does not match candidate manifest")
                    if target_object.dtype != target_dtype:
                        raise ValueError("bridge output dtype does not match candidate manifest")
                    target_cpu = target_object.detach().to("cpu").contiguous()
                    payload = bytearray(target_cpu.view(torch.uint8).numpy().tobytes())
                    if len(payload) != target_bytes:
                        raise ValueError("serialized bridge output has an invalid size")
                    target_payloads.append(payload)
                if bridge.device.type == "cuda":
                    torch.cuda.synchronize(bridge.device)
                transform_ms = (time.perf_counter() - transform_started) * 1000

                write_started = time.perf_counter()
                writes = store.write_many_exact(target_keys, target_payloads)
                write_ms = (time.perf_counter() - write_started) * 1000
                if any(write.bytes != write.remote_bytes for write in writes):
                    raise MooncakeObjectError("temporary target write was not exact")
                operation_ms = (time.perf_counter() - operation_started) * 1000
                if iteration >= warmup_iterations:
                    read_samples.append(read_ms)
                    transform_samples.append(transform_ms)
                    write_samples.append(write_ms)
                    operation_samples.append(operation_ms)
            finally:
                rollback = store.rollback(target_keys)
                failed = {key: result for key, result in rollback.items() if result != 0}
                if failed:
                    rollback_failures.append(failed)

    if rollback_failures:
        raise MooncakeObjectError(f"temporary target rollback failed: {rollback_failures}")
    if len(operation_samples) != iterations:
        raise RuntimeError("cost benchmark did not produce the requested samples")

    p95_operation = _percentile(operation_samples, 0.95)
    p95_native = _percentile(native_samples, 0.95)
    manifest = bridge.manifest
    real_mooncake_backend = store_factory is None
    eligible = (
        real_mooncake_backend
        and native_prefill_eligible
        and _is_sha256(native_prefill_report_sha256)
        and iterations >= MIN_COST_SAMPLES
        and len(native_samples) >= MIN_COST_SAMPLES
    )
    return {
        "schema_version": CACHED_KV_COST_SCHEMA_VERSION,
        "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "eligible_for_approval": eligible,
        "store_backend": "mooncake_store" if real_mooncake_backend else "test_double",
        "native_prefill_backend": native_prefill_backend,
        "native_prefill_report_sha256": native_prefill_report_sha256,
        "non_publishing": True,
        "external_index_published": False,
        "all_temporary_targets_rolled_back": True,
        "candidate_manifest_sha256": sha256_file(manifest_path),
        "bridge_id": manifest.bridge_id,
        "direction": manifest.direction,
        "weights_sha256": manifest.weights_sha256,
        "source_model_weights_sha256": manifest.source.weights_sha256,
        "target_model_weights_sha256": manifest.target.weights_sha256,
        "validation_dataset_sha256": manifest.validation_dataset_sha256,
        "source_keys_sha256": _json_sha256(normalized_source_keys),
        "setup_config_sha256": _json_sha256(
            {str(key): str(value) for key, value in setup_config.items()}
        ),
        "chunk_size": chunk_size,
        "chunk_count": len(normalized_source_keys),
        "warmup_iterations": warmup_iterations,
        "iterations": iterations,
        "native_prefill_samples": len(native_samples),
        "p95_source_read_transform_put_ms": p95_operation,
        "p95_target_prefill_ms": p95_native,
        "p95_materialization_to_prefill_ratio": p95_operation / p95_native,
        "measurements_ms": {
            "source_read": read_samples,
            "transform": transform_samples,
            "target_put": write_samples,
            "read_transform_put": operation_samples,
            "native_target_prefill": native_samples,
        },
    }


def load_native_prefill_evidence(
    path: str | Path,
    *,
    bridge: Any,
    expected_tokens: int,
) -> NativePrefillEvidence:
    report_path = Path(path)
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != NATIVE_PREFILL_COST_SCHEMA_VERSION:
        raise ValueError("native prefill report schema_version is invalid")
    if payload.get("direction") != bridge.manifest.direction:
        raise ValueError("native prefill report direction mismatch")
    if payload.get("target_model_weights_sha256") != bridge.manifest.target.weights_sha256:
        raise ValueError("native prefill report target model identity mismatch")
    if int(payload.get("token_count", -1)) != expected_tokens:
        raise ValueError("native prefill report token count mismatch")
    samples = payload.get("samples_ms")
    if not isinstance(samples, list):
        raise ValueError("native prefill report samples_ms must be a list")
    parsed_samples = _positive_finite_samples(samples, "native prefill")
    backend = str(payload.get("backend", ""))
    eligible = bool(payload.get("eligible_for_approval"))
    eligible = (
        eligible and backend == "vllm_native_target" and len(parsed_samples) >= MIN_COST_SAMPLES
    )
    return NativePrefillEvidence(
        samples_ms=tuple(parsed_samples),
        backend=backend,
        eligible_for_approval=eligible,
        report_sha256=sha256_file(report_path),
    )


def _positive_finite_samples(values: Sequence[float], name: str) -> list[float]:
    samples = [float(value) for value in values]
    if not samples or any(not math.isfinite(value) or value <= 0 for value in samples):
        raise ValueError(f"{name} samples must be finite and positive")
    return samples


def _percentile(values: Sequence[float], quantile: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, math.ceil(quantile * len(ordered)) - 1))
    return float(ordered[index])


def _json_sha256(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _is_sha256(value: str | None) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True


def _torch_dtype(name: str) -> Any:
    import torch

    try:
        return {"bfloat16": torch.bfloat16, "float16": torch.float16}[name]
    except KeyError as exc:
        raise ValueError(f"unsupported cached KV dtype: {name}") from exc
