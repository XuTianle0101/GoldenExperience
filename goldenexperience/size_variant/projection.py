"""Projection and materialization helpers for size-variant KV reuse."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Mapping

from goldenexperience.size_variant.models import (
    CalibrationManifest,
    FallbackReason,
    HiddenBridgeSpec,
    KVRestoreSpec,
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
class HiddenStateChunk:
    """Engine-neutral hidden-state chunk captured before target KV projection."""

    layer_id: int
    hidden: Any
    token_start: int = 0
    token_end: int = 0
    position_ids: tuple[int, ...] | None = None
    dtype: str = "float16"
    capture_point: str = "pre_kv_hidden"
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
class MaterializedHiddenChunk:
    """Target-width hidden chunk produced by the cross-scale bridge."""

    layer_id: int
    hidden: Any
    token_start: int
    token_end: int
    position_ids: tuple[int, ...] | None
    dtype: str
    capture_point: str
    source_layer_ids: tuple[int, ...]
    transform_id: str


@dataclass(frozen=True)
class MaterializationResult:
    """Result of materializing a target chunk set."""

    success: bool
    chunks: tuple[MaterializedKVChunk | MaterializedHiddenChunk, ...]
    elapsed_ms: float
    fallback_reason: FallbackReason = FallbackReason.NONE
    error: str | None = None


def build_hidden_bridge_spec(
    pair_id: str,
    direction: SizeVariantDirection,
    source_hidden_size: int,
    target_hidden_size: int,
    source_num_layers: int,
    target_num_layers: int,
    method: str = "low_rank_linear",
    capture_point: str = "pre_kv_hidden",
    rank: int | None = 256,
    weight_uri: str | None = None,
    weight_sha256: str | None = None,
) -> HiddenBridgeSpec:
    bridge_id = stable_artifact_id(
        "hidden-bridge",
        pair_id,
        direction.value,
        source_hidden_size,
        target_hidden_size,
        source_num_layers,
        target_num_layers,
        method,
        capture_point,
        rank,
    )
    return HiddenBridgeSpec(
        bridge_id=bridge_id,
        pair_id=pair_id,
        direction=direction,
        source_hidden_size=source_hidden_size,
        target_hidden_size=target_hidden_size,
        source_num_layers=source_num_layers,
        target_num_layers=target_num_layers,
        method=method,
        capture_point=capture_point,
        weight_uri=weight_uri,
        weight_sha256=weight_sha256,
        rank=rank,
    )


def build_kv_restore_spec(
    pair_id: str,
    direction: SizeVariantDirection,
    target_model_id: str,
    target_hidden_size: int,
    target_kv_heads: int,
    target_head_dim: int,
    method: str = "target_kv_projection",
    target_kv_layout: str = "gqa_paged_attention",
) -> KVRestoreSpec:
    restore_id = stable_artifact_id(
        "kv-restore",
        pair_id,
        direction.value,
        target_model_id,
        target_hidden_size,
        target_kv_heads,
        target_head_dim,
        method,
        target_kv_layout,
    )
    return KVRestoreSpec(
        restore_id=restore_id,
        pair_id=pair_id,
        direction=direction,
        target_model_id=target_model_id,
        target_hidden_size=target_hidden_size,
        target_kv_width=target_kv_heads * target_head_dim,
        target_kv_heads=target_kv_heads,
        target_head_dim=target_head_dim,
        method=method,
        target_kv_layout=target_kv_layout,
    )


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


class HiddenBridgeMaterializer:
    """Materialize target-model hidden states from source-model hidden states."""

    def __init__(
        self,
        manifest: CalibrationManifest,
        timeout_ms: float | None = None,
        layer_projectors: Mapping[int, Any] | None = None,
        layer_weights: Mapping[int, Any] | None = None,
        layer_biases: Mapping[int, Any] | None = None,
        enforce_quality_gate: bool = True,
        allow_identity_fallback: bool = False,
    ) -> None:
        if manifest.hidden_bridge is None:
            raise ValueError("HiddenBridgeMaterializer requires a hidden_bridge manifest.")
        self.manifest = manifest
        self.timeout_ms = timeout_ms
        self.layer_projectors = dict(layer_projectors or {})
        self.layer_weights = dict(layer_weights or {})
        self.layer_biases = dict(layer_biases or {})
        self.enforce_quality_gate = enforce_quality_gate
        self.allow_identity_fallback = allow_identity_fallback
        self._weight = identity_projection(
            manifest.hidden_bridge.source_hidden_size,
            manifest.hidden_bridge.target_hidden_size,
        )

    def materialize(self, source_chunks: Mapping[int, HiddenStateChunk]) -> MaterializationResult:
        start = time.perf_counter()
        errors = self.manifest.validate(include_quality=self.enforce_quality_gate)
        if errors:
            return MaterializationResult(
                success=False,
                chunks=(),
                elapsed_ms=0.0,
                fallback_reason=FallbackReason.QUALITY_GATE_FAILED,
                error="; ".join(errors),
            )

        assert self.manifest.hidden_bridge is not None
        if (
            self.manifest.hidden_bridge.method == "low_rank_linear"
            and not self.allow_identity_fallback
            and not self._has_all_layer_projectors()
        ):
            return MaterializationResult(
                success=False,
                chunks=(),
                elapsed_ms=0.0,
                fallback_reason=FallbackReason.MISSING_CALIBRATION,
                error="low-rank hidden bridge weights are not loaded for every target layer",
            )
        output: list[MaterializedHiddenChunk] = []
        for entry in self.manifest.layer_map.entries:
            if self.timeout_ms is not None and (time.perf_counter() - start) * 1000.0 > self.timeout_ms:
                return MaterializationResult(
                    success=False,
                    chunks=tuple(output),
                    elapsed_ms=(time.perf_counter() - start) * 1000.0,
                    fallback_reason=FallbackReason.MATERIALIZATION_TIMEOUT,
                    error="hidden bridge timeout",
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
            if source.capture_point != self.manifest.hidden_bridge.capture_point:
                return MaterializationResult(
                    success=False,
                    chunks=tuple(output),
                    elapsed_ms=(time.perf_counter() - start) * 1000.0,
                    fallback_reason=FallbackReason.QUALITY_GATE_FAILED,
                    error="hidden capture point mismatch",
                )
            hidden = self._project_hidden(entry.target_layer_id, source.hidden)
            hidden_shape = infer_shape(hidden)
            if hidden_shape and hidden_shape[-1] != self.manifest.hidden_bridge.target_hidden_size:
                return MaterializationResult(
                    success=False,
                    chunks=tuple(output),
                    elapsed_ms=(time.perf_counter() - start) * 1000.0,
                    fallback_reason=FallbackReason.PROJECTION_SHAPE_MISMATCH,
                    error="bridged hidden final dimension does not match target width",
                )
            output.append(
                MaterializedHiddenChunk(
                    layer_id=entry.target_layer_id,
                    hidden=hidden,
                    token_start=source.token_start,
                    token_end=source.token_end,
                    position_ids=source.position_ids,
                    dtype=self.manifest.target.kv_shape.dtype,
                    capture_point=self.manifest.hidden_bridge.capture_point,
                    source_layer_ids=entry.source_layer_ids,
                    transform_id=self.manifest.hidden_bridge.bridge_id,
                )
            )
        return MaterializationResult(
            success=True,
            chunks=tuple(output),
            elapsed_ms=(time.perf_counter() - start) * 1000.0,
        )

    def _blend_source_chunks(self, chunks: list[HiddenStateChunk], weights: tuple[float, ...]) -> HiddenStateChunk:
        if len(chunks) == 1:
            return chunks[0]
        hidden = _weighted_sum([chunk.hidden for chunk in chunks], weights)
        first = chunks[0]
        return HiddenStateChunk(
            layer_id=first.layer_id,
            hidden=hidden,
            token_start=first.token_start,
            token_end=first.token_end,
            position_ids=first.position_ids,
            dtype=first.dtype,
            capture_point=first.capture_point,
            metadata=first.metadata,
        )

    def _project_hidden(self, target_layer_id: int, hidden: Any) -> Any:
        projector = self.layer_projectors.get(target_layer_id)
        if projector is not None:
            return projector(hidden)
        weight = self.layer_weights.get(target_layer_id, self._weight)
        bias = self.layer_biases.get(target_layer_id)
        return project_last_dim(hidden, weight, bias)

    def _has_all_layer_projectors(self) -> bool:
        target_layers = {entry.target_layer_id for entry in self.manifest.layer_map.entries}
        loaded_layers = set(self.layer_projectors) | set(self.layer_weights)
        return target_layers <= loaded_layers


class TargetKVRestorer:
    """Restore target-shaped KV chunks from bridged hidden states.

    Runtime integrations should pass real target W_K/W_V/RoPE callables. The default
    deterministic weights keep the contract testable without model weights.
    """

    def __init__(
        self,
        manifest: CalibrationManifest,
        key_weight: Any | None = None,
        value_weight: Any | None = None,
        key_projector: Any | None = None,
        value_projector: Any | None = None,
        rope_fn: Any | None = None,
        layer_key_projectors: Mapping[int, Any] | None = None,
        layer_value_projectors: Mapping[int, Any] | None = None,
        layer_rope_fns: Mapping[int, Any] | None = None,
    ) -> None:
        if manifest.kv_restore is None:
            raise ValueError("TargetKVRestorer requires a kv_restore manifest.")
        self.manifest = manifest
        self.restore = manifest.kv_restore
        self.key_projector = key_projector
        self.value_projector = value_projector
        self.rope_fn = rope_fn
        self.layer_key_projectors = dict(layer_key_projectors or {})
        self.layer_value_projectors = dict(layer_value_projectors or {})
        self.layer_rope_fns = dict(layer_rope_fns or {})
        self.key_weight = key_weight or identity_projection(
            self.restore.target_hidden_size,
            self.restore.target_kv_width,
        )
        self.value_weight = value_weight or identity_projection(
            self.restore.target_hidden_size,
            self.restore.target_kv_width,
        )

    def restore_chunk(self, hidden_chunk: MaterializedHiddenChunk) -> MaterializationResult:
        start = time.perf_counter()
        key_projector = self.layer_key_projectors.get(hidden_chunk.layer_id, self.key_projector)
        value_projector = self.layer_value_projectors.get(hidden_chunk.layer_id, self.value_projector)
        rope_fn = self.layer_rope_fns.get(hidden_chunk.layer_id, self.rope_fn)
        key = (
            key_projector(hidden_chunk.hidden)
            if key_projector is not None
            else project_last_dim(hidden_chunk.hidden, self.key_weight)
        )
        value = (
            value_projector(hidden_chunk.hidden)
            if value_projector is not None
            else project_last_dim(hidden_chunk.hidden, self.value_weight)
        )
        if rope_fn is not None:
            key, value = rope_fn(key, value, hidden_chunk.position_ids)
        key_shape = infer_shape(key)
        value_shape = infer_shape(value)
        if (key_shape and not _matches_target_kv_layout(
            key_shape,
            self.restore.target_kv_heads,
            self.restore.target_head_dim,
            self.restore.target_kv_width,
        )) or (
            value_shape
            and not _matches_target_kv_layout(
                value_shape,
                self.restore.target_kv_heads,
                self.restore.target_head_dim,
                self.restore.target_kv_width,
            )
        ):
            return MaterializationResult(
                success=False,
                chunks=(),
                elapsed_ms=(time.perf_counter() - start) * 1000.0,
                fallback_reason=FallbackReason.PROJECTION_SHAPE_MISMATCH,
                error="restored KV final dimension does not match target width",
            )
        chunk = MaterializedKVChunk(
            layer_id=hidden_chunk.layer_id,
            key=key,
            value=value,
            token_start=hidden_chunk.token_start,
            token_end=hidden_chunk.token_end,
            dtype=self.manifest.target.kv_shape.dtype,
            source_layer_ids=hidden_chunk.source_layer_ids,
            transform_id=self.restore.restore_id,
        )
        return MaterializationResult(
            success=True,
            chunks=(chunk,),
            elapsed_ms=(time.perf_counter() - start) * 1000.0,
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
        return False
    if estimated_materialization_ms < 0 or estimated_target_prefill_ms <= 0:
        return False
    return estimated_materialization_ms <= max_materialization_ratio * estimated_target_prefill_ms


def expected_projection_width(spec: ProjectionSpec) -> int:
    return spec.target_width


def width_from_manifest(manifest: CalibrationManifest) -> tuple[int, int]:
    return kv_width(manifest.source.kv_shape), kv_width(manifest.target.kv_shape)


def _matches_target_kv_layout(shape: tuple[int, ...], target_kv_heads: int, target_head_dim: int, target_kv_width: int) -> bool:
    if not shape:
        return True
    if shape[-1] == target_kv_width:
        return True
    if len(shape) >= 4 and shape[1] == target_kv_heads and shape[-1] == target_head_dim:
        return True
    return False


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
    if hasattr(first, "__mul__") and hasattr(first, "__add__"):
        total = first * weights[0]
        for value, weight in zip(values[1:], weights[1:], strict=True):
            total = total + (value * weight)
        return total
    return first
