"""Records for same-model, different-parameter-size KV reuse."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import Any

from goldenexperience.reuse.models import KVShape, ModelRef


class SizeVariantDirection(str, Enum):
    """Directional artifact identity for model-size variants."""

    SMALL_TO_LARGE = "small_to_large"
    LARGE_TO_SMALL = "large_to_small"
    UNKNOWN = "unknown"


class FallbackReason(str, Enum):
    """Stable fallback labels used by patch accounting."""

    NONE = "none"
    MISSING_CALIBRATION = "missing_calibration"
    ARTIFACT_HASH_MISMATCH = "artifact_hash_mismatch"
    TOKENIZER_MISMATCH = "tokenizer_mismatch"
    ROPE_MISMATCH = "rope_mismatch"
    INCOMPLETE_LAYER_MAP = "incomplete_layer_map"
    SOURCE_LAYER_MISSING = "source_layer_missing"
    PROJECTION_SHAPE_MISMATCH = "projection_shape_mismatch"
    QUALITY_GATE_FAILED = "quality_gate_failed"
    MATERIALIZATION_TIMEOUT = "materialization_timeout"
    COST_GATE_FAILED = "cost_gate_failed"
    ARTIFACT_NOT_APPROVED = "artifact_not_approved"
    RISK_GATE_REJECTED = "risk_gate_rejected"
    MISSING_SIDECAR = "missing_sidecar"
    OUT_OF_DISTRIBUTION = "out_of_distribution"
    MODEL_HASH_CHANGED = "model_hash_changed"
    DIRECT_INJECTION_FAILED = "direct_injection_failed"


@dataclass(frozen=True)
class LayerMapEntry:
    """Mapping from one target layer to one or more source layers."""

    target_layer_id: int
    source_layer_ids: tuple[int, ...]
    weights: tuple[float, ...]

    def validate(self, source_num_layers: int) -> list[str]:
        errors: list[str] = []
        if not self.source_layer_ids:
            errors.append(f"target layer {self.target_layer_id} has no source layer")
        if len(self.source_layer_ids) != len(self.weights):
            errors.append(f"target layer {self.target_layer_id} has mismatched layer weights")
        for source_layer_id in self.source_layer_ids:
            if source_layer_id < 0 or source_layer_id >= source_num_layers:
                errors.append(f"source layer {source_layer_id} is outside source depth")
        if self.weights and abs(sum(self.weights) - 1.0) > 1e-6:
            errors.append(f"target layer {self.target_layer_id} weights do not sum to 1")
        return errors


@dataclass(frozen=True)
class LayerMap:
    """Complete target-depth layer mapping for one direction."""

    layer_map_id: str
    pair_id: str
    direction: SizeVariantDirection
    source_num_layers: int
    target_num_layers: int
    entries: tuple[LayerMapEntry, ...]
    method: str = "linear_interpolation"
    score: float = 1.0

    def entry_for(self, target_layer_id: int) -> LayerMapEntry | None:
        for entry in self.entries:
            if entry.target_layer_id == target_layer_id:
                return entry
        return None

    def validate(self) -> list[str]:
        errors: list[str] = []
        target_ids = [entry.target_layer_id for entry in self.entries]
        expected = list(range(self.target_num_layers))
        if sorted(target_ids) != expected:
            errors.append("layer map must cover every target layer exactly once")
        if len(target_ids) != len(set(target_ids)):
            errors.append("layer map contains duplicate target layers")
        for entry in self.entries:
            errors.extend(entry.validate(self.source_num_layers))
        return errors


@dataclass(frozen=True)
class ProjectionSpec:
    """KV projection shape contract for one direction.

    The MVP stores projection metadata and uses deterministic pad/truncate projection.
    Learned per-layer weights can be attached later through weight_uri without changing
    the manifest contract.
    """

    projection_id: str
    pair_id: str
    direction: SizeVariantDirection
    source_width: int
    target_width: int
    source_kv_heads: int
    target_kv_heads: int
    source_head_dim: int
    target_head_dim: int
    method: str = "identity_pad_truncate"
    weight_uri: str | None = None
    rank: int | None = None

    def validate(self) -> list[str]:
        errors = []
        if self.source_width != self.source_kv_heads * self.source_head_dim:
            errors.append("source_width must equal source_kv_heads * source_head_dim")
        if self.target_width != self.target_kv_heads * self.target_head_dim:
            errors.append("target_width must equal target_kv_heads * target_head_dim")
        if self.source_width <= 0 or self.target_width <= 0:
            errors.append("projection widths must be positive")
        return errors


@dataclass(frozen=True)
class HiddenBridgeSpec:
    """Hidden-state bridge contract for cross-scale KV restoration."""

    bridge_id: str
    pair_id: str
    direction: SizeVariantDirection
    source_hidden_size: int
    target_hidden_size: int
    source_num_layers: int
    target_num_layers: int
    method: str = "low_rank_linear"
    capture_point: str = "pre_kv_hidden"
    weight_uri: str | None = None
    weight_sha256: str | None = None
    rank: int | None = 256

    def validate(self, *, require_weights: bool = False) -> list[str]:
        errors = []
        if self.source_hidden_size <= 0 or self.target_hidden_size <= 0:
            errors.append("hidden bridge widths must be positive")
        if self.source_num_layers <= 0 or self.target_num_layers <= 0:
            errors.append("hidden bridge layer counts must be positive")
        if self.capture_point != "pre_kv_hidden":
            errors.append("only pre_kv_hidden capture is supported")
        if self.method not in {"low_rank_linear", "identity_pad_truncate"}:
            errors.append("unsupported hidden bridge method")
        if self.rank is not None and self.rank <= 0:
            errors.append("hidden bridge rank must be positive")
        if require_weights and self.method == "low_rank_linear":
            if not self.weight_uri:
                errors.append("low-rank hidden bridge weight_uri is required")
            if not self.weight_sha256 or len(self.weight_sha256) != 64:
                errors.append("low-rank hidden bridge weight_sha256 is required")
        return errors


@dataclass(frozen=True)
class KVRestoreSpec:
    """Target-model KV restore contract from bridged hidden states."""

    restore_id: str
    pair_id: str
    direction: SizeVariantDirection
    target_model_id: str
    target_hidden_size: int
    target_kv_width: int
    target_kv_heads: int
    target_head_dim: int
    method: str = "target_kv_projection"
    target_kv_layout: str = "gqa_paged_attention"
    weight_source: str = "target_model"
    rope_applied: bool = True

    def validate(self) -> list[str]:
        errors = []
        if self.target_hidden_size <= 0:
            errors.append("restore target hidden size must be positive")
        if self.target_kv_width != self.target_kv_heads * self.target_head_dim:
            errors.append("restore target_kv_width must equal target_kv_heads * target_head_dim")
        if self.target_kv_width <= 0:
            errors.append("restore target KV width must be positive")
        if self.method != "target_kv_projection":
            errors.append("unsupported KV restore method")
        if self.weight_source != "target_model":
            errors.append("KV restore must use target_model weights")
        return errors


@dataclass(frozen=True)
class QualityGateResult:
    """Offline or shadow-mode quality gate result."""

    passed: bool
    hidden_cosine: float = 0.0
    kv_cosine: float = 0.0
    attention_proxy_cosine: float = 0.0
    perplexity_drift_pct: float = 0.0
    task_score_drop_pct: float = 0.0
    reasons: tuple[str, ...] = ()

    @classmethod
    def uncalibrated(cls, reason: str = "calibration_metrics_missing") -> QualityGateResult:
        """Return a fail-closed result for manifests without measured quality evidence."""

        return cls(passed=False, reasons=(reason,))

    @classmethod
    def from_metrics(
        cls,
        kv_cosine: float,
        attention_proxy_cosine: float,
        perplexity_drift_pct: float,
        task_score_drop_pct: float,
        hidden_cosine: float = 0.0,
        min_hidden_cosine: float | None = None,
        min_kv_cosine: float = 0.90,
        min_attention_proxy_cosine: float = 0.95,
        max_perplexity_drift_pct: float = 5.0,
        max_task_score_drop_pct: float = 2.0,
    ) -> QualityGateResult:
        reasons: list[str] = []
        if min_hidden_cosine is not None and hidden_cosine < min_hidden_cosine:
            reasons.append("hidden_cosine_below_threshold")
        if kv_cosine < min_kv_cosine:
            reasons.append("kv_cosine_below_threshold")
        if attention_proxy_cosine < min_attention_proxy_cosine:
            reasons.append("attention_proxy_below_threshold")
        if perplexity_drift_pct > max_perplexity_drift_pct:
            reasons.append("perplexity_drift_above_threshold")
        if task_score_drop_pct > max_task_score_drop_pct:
            reasons.append("task_score_drop_above_threshold")
        return cls(
            passed=not reasons,
            hidden_cosine=hidden_cosine,
            kv_cosine=kv_cosine,
            attention_proxy_cosine=attention_proxy_cosine,
            perplexity_drift_pct=perplexity_drift_pct,
            task_score_drop_pct=task_score_drop_pct,
            reasons=tuple(reasons),
        )


@dataclass(frozen=True)
class CalibrationManifest:
    """Serializable artifact manifest for one GoldenScale direction."""

    calibration_id: str
    pair_id: str
    direction: SizeVariantDirection
    source: ModelRef
    target: ModelRef
    layer_map: LayerMap
    projection: ProjectionSpec
    quality: QualityGateResult
    hidden_bridge: HiddenBridgeSpec | None = None
    kv_restore: KVRestoreSpec | None = None
    artifact_root: str = "artifacts/golden_scale"
    prompts_count: int = 0
    created_by: str = "golden-scale-fit"
    references: tuple[str, ...] = ()
    metadata: dict[str, str | int | float | bool] = field(default_factory=dict)
    scope: str = "unscoped"
    prefix_hash_allowlist: tuple[str, ...] = ()
    evaluation_dataset_hash: str | None = None
    held_out_prompts_count: int = 0

    @property
    def layer_map_id(self) -> str:
        return self.layer_map.layer_map_id

    @property
    def projection_id(self) -> str:
        return self.projection.projection_id

    @property
    def hidden_bridge_id(self) -> str | None:
        return self.hidden_bridge.bridge_id if self.hidden_bridge is not None else None

    @property
    def restore_id(self) -> str | None:
        return self.kv_restore.restore_id if self.kv_restore is not None else None

    @property
    def state_kind(self) -> str:
        return "hidden" if self.hidden_bridge is not None else "kv"

    @property
    def passed(self) -> bool:
        return self.quality.passed and not self.validate()

    def validate(self, *, include_quality: bool = True) -> list[str]:
        errors: list[str] = []
        if not self.source.same_family_architecture(self.target):
            errors.append("source and target must share family and architecture")
        if not self.source.shares_tokenizer_with(self.target):
            errors.append("source and target tokenizers differ")
        if self.source.kv_shape.rope_theta != self.target.kv_shape.rope_theta:
            errors.append("source and target rope_theta differ")
        if self.source.kv_shape.rope_scaling != self.target.kv_shape.rope_scaling:
            errors.append("source and target rope_scaling differ")
        if self.layer_map.source_num_layers != self.source.kv_shape.num_layers:
            errors.append("layer map source depth does not match source model")
        if self.layer_map.target_num_layers != self.target.kv_shape.num_layers:
            errors.append("layer map target depth does not match target model")
        if self.layer_map.pair_id != self.pair_id or self.projection.pair_id != self.pair_id:
            errors.append("layer map/projection pair_id must match manifest pair_id")
        if (
            self.layer_map.direction != self.direction
            or self.projection.direction != self.direction
        ):
            errors.append("layer map/projection direction must match manifest direction")
        if self.hidden_bridge is not None:
            errors.extend(self.hidden_bridge.validate(require_weights=include_quality))
            if self.hidden_bridge.pair_id != self.pair_id:
                errors.append("hidden bridge pair_id must match manifest pair_id")
            if self.hidden_bridge.direction != self.direction:
                errors.append("hidden bridge direction must match manifest direction")
            if self.hidden_bridge.source_num_layers != self.source.kv_shape.num_layers:
                errors.append("hidden bridge source depth does not match source model")
            if self.hidden_bridge.target_num_layers != self.target.kv_shape.num_layers:
                errors.append("hidden bridge target depth does not match target model")
            if (
                self.source.kv_shape.hidden_size
                and self.hidden_bridge.source_hidden_size != self.source.kv_shape.hidden_size
            ):
                errors.append("hidden bridge source width does not match source model")
            if (
                self.target.kv_shape.hidden_size
                and self.hidden_bridge.target_hidden_size != self.target.kv_shape.hidden_size
            ):
                errors.append("hidden bridge target width does not match target model")
        if self.kv_restore is not None:
            errors.extend(self.kv_restore.validate())
            if self.kv_restore.pair_id != self.pair_id:
                errors.append("KV restore pair_id must match manifest pair_id")
            if self.kv_restore.direction != self.direction:
                errors.append("KV restore direction must match manifest direction")
            if self.kv_restore.target_model_id != self.target.model_id:
                errors.append("KV restore target model differs from manifest target")
            if (
                self.target.kv_shape.hidden_size
                and self.kv_restore.target_hidden_size != self.target.kv_shape.hidden_size
            ):
                errors.append("KV restore hidden width does not match target model")
            if self.kv_restore.target_kv_width != kv_width(self.target.kv_shape):
                errors.append("KV restore target width does not match target KV shape")
        if (self.hidden_bridge is None) != (self.kv_restore is None):
            errors.append("hidden bridge and KV restore specs must be provided together")
        errors.extend(self.layer_map.validate())
        errors.extend(self.projection.validate())
        if include_quality and self.prompts_count <= 0:
            errors.append("calibration prompt count must be positive")
        if include_quality and self.scope not in {"global", "prefix_allowlist"}:
            errors.append("calibration scope must be global or prefix_allowlist")
        if include_quality and self.scope == "prefix_allowlist" and not self.prefix_hash_allowlist:
            errors.append("prefix-scoped calibration requires a prefix hash allowlist")
        if include_quality and self.held_out_prompts_count <= 0:
            errors.append("held-out prompt count must be positive")
        if include_quality and not _is_sha256(self.evaluation_dataset_hash):
            errors.append("evaluation dataset SHA-256 is required")
        if include_quality and self.hidden_bridge is not None:
            errors.extend(self._validate_bridge_weight())
        if include_quality and not self.quality.passed:
            errors.append("quality gate failed")
        return errors

    def _validate_bridge_weight(self) -> list[str]:
        bridge = self.hidden_bridge
        if bridge is None or bridge.method != "low_rank_linear" or not bridge.weight_uri:
            return []
        path = Path(bridge.weight_uri)
        if not path.is_absolute():
            path = Path(self.artifact_root) / path
        if not path.is_file():
            return [f"hidden bridge weight file does not exist: {path}"]
        if bridge.weight_sha256 is None:
            return []
        stat = path.stat()
        digest = _file_sha256(str(path.resolve()), stat.st_size, stat.st_mtime_ns)
        if digest != bridge.weight_sha256:
            return ["hidden bridge weight checksum mismatch"]
        return []

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["direction"] = self.direction.value
        payload["layer_map"]["direction"] = self.layer_map.direction.value
        payload["projection"]["direction"] = self.projection.direction.value
        if payload.get("hidden_bridge") is not None:
            payload["hidden_bridge"]["direction"] = self.hidden_bridge.direction.value  # type: ignore[union-attr]
        if payload.get("kv_restore") is not None:
            payload["kv_restore"]["direction"] = self.kv_restore.direction.value  # type: ignore[union-attr]
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> CalibrationManifest:
        source = model_ref_from_dict(payload["source"])
        target = model_ref_from_dict(payload["target"])
        layer_map_payload = payload["layer_map"]
        projection_payload = payload["projection"]
        quality_payload = payload["quality"]
        layer_map = LayerMap(
            layer_map_id=layer_map_payload["layer_map_id"],
            pair_id=layer_map_payload["pair_id"],
            direction=SizeVariantDirection(layer_map_payload["direction"]),
            source_num_layers=int(layer_map_payload["source_num_layers"]),
            target_num_layers=int(layer_map_payload["target_num_layers"]),
            entries=tuple(
                LayerMapEntry(
                    target_layer_id=int(item["target_layer_id"]),
                    source_layer_ids=tuple(int(value) for value in item["source_layer_ids"]),
                    weights=tuple(float(value) for value in item["weights"]),
                )
                for item in layer_map_payload["entries"]
            ),
            method=layer_map_payload.get("method", "linear_interpolation"),
            score=float(layer_map_payload.get("score", 1.0)),
        )
        projection = ProjectionSpec(
            projection_id=projection_payload["projection_id"],
            pair_id=projection_payload["pair_id"],
            direction=SizeVariantDirection(projection_payload["direction"]),
            source_width=int(projection_payload["source_width"]),
            target_width=int(projection_payload["target_width"]),
            source_kv_heads=int(projection_payload["source_kv_heads"]),
            target_kv_heads=int(projection_payload["target_kv_heads"]),
            source_head_dim=int(projection_payload["source_head_dim"]),
            target_head_dim=int(projection_payload["target_head_dim"]),
            method=projection_payload.get("method", "identity_pad_truncate"),
            weight_uri=projection_payload.get("weight_uri"),
            rank=projection_payload.get("rank"),
        )
        hidden_bridge = None
        hidden_bridge_payload = payload.get("hidden_bridge")
        if hidden_bridge_payload is not None:
            hidden_bridge = HiddenBridgeSpec(
                bridge_id=hidden_bridge_payload["bridge_id"],
                pair_id=hidden_bridge_payload["pair_id"],
                direction=SizeVariantDirection(hidden_bridge_payload["direction"]),
                source_hidden_size=int(hidden_bridge_payload["source_hidden_size"]),
                target_hidden_size=int(hidden_bridge_payload["target_hidden_size"]),
                source_num_layers=int(hidden_bridge_payload["source_num_layers"]),
                target_num_layers=int(hidden_bridge_payload["target_num_layers"]),
                method=hidden_bridge_payload.get("method", "low_rank_linear"),
                capture_point=hidden_bridge_payload.get("capture_point", "pre_kv_hidden"),
                weight_uri=hidden_bridge_payload.get("weight_uri"),
                weight_sha256=hidden_bridge_payload.get("weight_sha256"),
                rank=hidden_bridge_payload.get("rank"),
            )
        kv_restore = None
        kv_restore_payload = payload.get("kv_restore")
        if kv_restore_payload is not None:
            kv_restore = KVRestoreSpec(
                restore_id=kv_restore_payload["restore_id"],
                pair_id=kv_restore_payload["pair_id"],
                direction=SizeVariantDirection(kv_restore_payload["direction"]),
                target_model_id=kv_restore_payload["target_model_id"],
                target_hidden_size=int(kv_restore_payload["target_hidden_size"]),
                target_kv_width=int(kv_restore_payload["target_kv_width"]),
                target_kv_heads=int(kv_restore_payload["target_kv_heads"]),
                target_head_dim=int(kv_restore_payload["target_head_dim"]),
                method=kv_restore_payload.get("method", "target_kv_projection"),
                target_kv_layout=kv_restore_payload.get("target_kv_layout", "gqa_paged_attention"),
                weight_source=kv_restore_payload.get("weight_source", "target_model"),
                rope_applied=bool(kv_restore_payload.get("rope_applied", True)),
            )
        quality = QualityGateResult(
            passed=bool(quality_payload["passed"]),
            hidden_cosine=float(quality_payload.get("hidden_cosine", 0.0)),
            kv_cosine=float(quality_payload.get("kv_cosine", 0.0)),
            attention_proxy_cosine=float(quality_payload.get("attention_proxy_cosine", 0.0)),
            perplexity_drift_pct=float(quality_payload.get("perplexity_drift_pct", 0.0)),
            task_score_drop_pct=float(quality_payload.get("task_score_drop_pct", 0.0)),
            reasons=tuple(quality_payload.get("reasons", ())),
        )
        return cls(
            calibration_id=payload["calibration_id"],
            pair_id=payload["pair_id"],
            direction=SizeVariantDirection(payload["direction"]),
            source=source,
            target=target,
            layer_map=layer_map,
            projection=projection,
            quality=quality,
            hidden_bridge=hidden_bridge,
            kv_restore=kv_restore,
            artifact_root=payload.get("artifact_root", "artifacts/golden_scale"),
            prompts_count=int(payload.get("prompts_count", 0)),
            created_by=payload.get("created_by", "golden-scale-fit"),
            references=tuple(payload.get("references", ())),
            metadata=dict(payload.get("metadata", {})),
            scope=payload.get("scope", "unscoped"),
            prefix_hash_allowlist=tuple(payload.get("prefix_hash_allowlist", ())),
            evaluation_dataset_hash=payload.get("evaluation_dataset_hash"),
            held_out_prompts_count=int(payload.get("held_out_prompts_count", 0)),
        )

    def save(self, path: str | Path) -> None:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True), encoding="utf-8"
        )

    @classmethod
    def load(cls, path: str | Path) -> CalibrationManifest:
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def kv_width(shape: KVShape) -> int:
    return shape.num_key_value_heads * shape.head_dim


def infer_direction(source: ModelRef, target: ModelRef) -> SizeVariantDirection:
    if source.parameter_count_b is None or target.parameter_count_b is None:
        return SizeVariantDirection.UNKNOWN
    if source.parameter_count_b < target.parameter_count_b:
        return SizeVariantDirection.SMALL_TO_LARGE
    if source.parameter_count_b > target.parameter_count_b:
        return SizeVariantDirection.LARGE_TO_SMALL
    return SizeVariantDirection.UNKNOWN


def pair_id_for(source: ModelRef, target: ModelRef) -> str:
    family = source.family if source.family == target.family else f"{source.family}_{target.family}"
    left = source.model_id.replace("/", "_").replace(" ", "_")
    right = target.model_id.replace("/", "_").replace(" ", "_")
    return f"{family}:{left}->{right}"


def stable_artifact_id(prefix: str, *parts: object) -> str:
    raw = "|".join(str(part) for part in parts)
    return f"{prefix}-" + hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def _is_sha256(value: str | None) -> bool:
    return bool(
        value
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value.lower())
    )


@lru_cache(maxsize=32)
def _file_sha256(path: str, size: int, mtime_ns: int) -> str:
    del size, mtime_ns
    hasher = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def model_ref_to_dict(model: ModelRef) -> dict[str, Any]:
    payload = asdict(model)
    return payload


def model_ref_from_dict(payload: dict[str, Any]) -> ModelRef:
    kv_shape = KVShape(**payload["kv_shape"])
    return ModelRef(
        model_id=payload["model_id"],
        family=payload["family"],
        architecture=payload["architecture"],
        tokenizer_id=payload["tokenizer_id"],
        kv_shape=kv_shape,
        parameter_count_b=payload.get("parameter_count_b"),
        base_model_id=payload.get("base_model_id"),
        lora_adapter_id=payload.get("lora_adapter_id"),
        revision=payload.get("revision"),
        metadata=dict(payload.get("metadata", {})),
    )
