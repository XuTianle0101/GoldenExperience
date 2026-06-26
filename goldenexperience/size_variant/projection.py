"""Projection and materialization helpers for size-variant KV reuse."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Mapping

from goldenexperience.size_variant.models import (
    CalibrationManifest,
    FallbackReason,
    ProjectionSpec,
    SizeVariantDirection,
    kv_width,
    stable_artifact_id,
)
from goldenexperience.utils.tensors import infer_shape, identity_projection, project_last_dim


@dataclass(frozen=True)
class KVChunk:
    """Engine-neutral KV chunk used by tests and patch adapters."""

    layer_id: int
    key: Any
    value: Any
    token_start: int = 0
    token_end: int = 0
    dtype: str = "float16"
    metadata: dict[str, str | int | float | bool] | None = None


@dataclass(frozen=True)
class MaterializedKVChunk:
    """Projected target-model KV chunk."""

    layer_id: int
    key: Any
    value: Any
    token_start: int
    token_end: int
    dtype: str
    source_layer_ids: tuple[int, ...]
    transform_id: str


@dataclass(frozen=True)
class MaterializationResult:
    """Result of materializing a target chunk set."""

    success: bool
    chunks: tuple[MaterializedKVChunk, ...]
    elapsed_ms: float
    fallback_reason: FallbackReason = FallbackReason.NONE
    error: str | None = None


def build_projection_spec(
    pair_id: str,
    direction: SizeVariantDirection,
    source_kv_heads: int,
    target_kv_heads: int,
    source_head_dim: int,
    target_head_dim: int,
    method: str = "identity_pad_truncate",
) -> ProjectionSpec:
    projection_id = stable_artifact_id(
        "projection",
        pair_id,
        direction.value,
        source_kv_heads,
        target_kv_heads,
        source_head_dim,
        target_head_dim,
        method,
    )
    return ProjectionSpec(
        projection_id=projection_id,
        pair_id=pair_id,
        direction=direction,
        source_width=source_kv_heads * source_head_dim,
        target_width=target_kv_heads * target_head_dim,
        source_kv_heads=source_kv_heads,
        target_kv_heads=target_kv_heads,
        source_head_dim=source_head_dim,
        target_head_dim=target_head_dim,
        method=method,
    )


class SizeVariantMaterializer:
    """Materialize target-shaped KV chunks from source-model chunks."""

    def __init__(self, manifest: CalibrationManifest, timeout_ms: float | None = None) -> None:
        self.manifest = manifest
        self.timeout_ms = timeout_ms
        self._weight = identity_projection(
            manifest.projection.source_width,
            manifest.projection.target_width,
        )

    def materialize(self, source_chunks: Mapping[int, KVChunk]) -> MaterializationResult:
        start = time.perf_counter()
        errors = self.manifest.validate()
        if errors:
            return MaterializationResult(
                success=False,
                chunks=(),
                elapsed_ms=0.0,
                fallback_reason=FallbackReason.QUALITY_GATE_FAILED,
                error="; ".join(errors),
            )

        output: list[MaterializedKVChunk] = []
        for entry in self.manifest.layer_map.entries:
            if self.timeout_ms is not None and (time.perf_counter() - start) * 1000.0 > self.timeout_ms:
                return MaterializationResult(
                    success=False,
                    chunks=tuple(output),
                    elapsed_ms=(time.perf_counter() - start) * 1000.0,
                    fallback_reason=FallbackReason.MATERIALIZATION_TIMEOUT,
                    error="projection timeout",
                )
            missing = [layer_id for layer_id in entry.source_layer_ids if layer_id not in source_chunks]
            if missing:
                return MaterializationResult(
                    success=False,
                    chunks=tuple(output),
                    elapsed_ms=(time.perf_counter() - start) * 1000.0,
                    fallback_reason=FallbackReason.SOURCE_LAYER_MISSING,
                    error=f"missing source layer(s): {missing}",
                )
            source = self._blend_source_chunks([source_chunks[layer_id] for layer_id in entry.source_layer_ids], entry.weights)
            key = project_last_dim(source.key, self._weight)
            value = project_last_dim(source.value, self._weight)
            key_shape = infer_shape(key)
            value_shape = infer_shape(value)
            if (key_shape and key_shape[-1] != self.manifest.projection.target_width) or (
                value_shape and value_shape[-1] != self.manifest.projection.target_width
            ):
                return MaterializationResult(
                    success=False,
                    chunks=tuple(output),
                    elapsed_ms=(time.perf_counter() - start) * 1000.0,
                    fallback_reason=FallbackReason.PROJECTION_SHAPE_MISMATCH,
                    error="projected KV final dimension does not match target width",
                )
            output.append(
                MaterializedKVChunk(
                    layer_id=entry.target_layer_id,
                    key=key,
                    value=value,
                    token_start=source.token_start,
                    token_end=source.token_end,
                    dtype=self.manifest.target.kv_shape.dtype,
                    source_layer_ids=entry.source_layer_ids,
                    transform_id=self.manifest.projection_id,
                )
            )
        return MaterializationResult(
            success=True,
            chunks=tuple(output),
            elapsed_ms=(time.perf_counter() - start) * 1000.0,
        )

    def _blend_source_chunks(self, chunks: list[KVChunk], weights: tuple[float, ...]) -> KVChunk:
        if len(chunks) == 1:
            return chunks[0]
        key = _weighted_sum([chunk.key for chunk in chunks], weights)
        value = _weighted_sum([chunk.value for chunk in chunks], weights)
        first = chunks[0]
        return KVChunk(
            layer_id=first.layer_id,
            key=key,
            value=value,
            token_start=first.token_start,
            token_end=first.token_end,
            dtype=first.dtype,
            metadata=first.metadata,
        )


def validate_projection_cost(
    estimated_materialization_ms: float | None,
    estimated_target_prefill_ms: float | None,
    max_materialization_ratio: float = 0.70,
) -> bool:
    if estimated_materialization_ms is None or estimated_target_prefill_ms is None:
        return True
    if estimated_target_prefill_ms <= 0:
        return False
    return estimated_materialization_ms <= max_materialization_ratio * estimated_target_prefill_ms


def expected_projection_width(spec: ProjectionSpec) -> int:
    return spec.target_width


def width_from_manifest(manifest: CalibrationManifest) -> tuple[int, int]:
    return kv_width(manifest.source.kv_shape), kv_width(manifest.target.kv_shape)


def _weighted_sum(values: list[Any], weights: tuple[float, ...]) -> Any:
    if not values:
        return []
    if len(values) != len(weights):
        raise ValueError("values and weights must have the same length")
    first = values[0]
    if isinstance(first, (int, float)):
        return sum(float(value) * weight for value, weight in zip(values, weights))
    if isinstance(first, list):
        return [
            _weighted_sum([value[idx] for value in values], weights)
            for idx in range(len(first))
        ]
    if isinstance(first, tuple):
        return tuple(
            _weighted_sum([value[idx] for value in values], weights)
            for idx in range(len(first))
        )
    return first
