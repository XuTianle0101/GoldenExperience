"""Model and plan records for cross-model KV cache reuse."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class ReuseScenario(str, Enum):
    """The three product scenarios GoldenExperience is built around."""

    LORA_ADAPTER = "model_lora_mutual_reuse"
    SAME_MODEL_SIZE_VARIANT = "same_model_different_parameter_size"
    CROSS_BASE_MODEL = "different_base_model"


class ReuseStrategy(str, Enum):
    """High-level materialization strategy for a reuse plan."""

    ADAPTER_DELTA_GATED_ALIAS = "adapter_delta_gated_alias"
    DIRECT_SHAPE_ALIAS = "direct_shape_alias"
    LAYERWISE_PROJECTION = "layerwise_projection"
    LEARNED_CROSS_BASE_TRANSLATOR = "learned_cross_base_translator"
    FALLBACK_RECOMPUTE = "fallback_recompute"


class PlanStatus(str, Enum):
    """Whether a plan may be executed by an LMCache patch path."""

    READY = "ready"
    NEEDS_CALIBRATION = "needs_calibration"
    BLOCKED = "blocked"


@dataclass(frozen=True)
class KVShape:
    """Minimal shape surface needed before trying to reuse a KV payload."""

    num_layers: int
    num_key_value_heads: int
    head_dim: int
    dtype: str = "float16"
    rope_theta: float | None = None
    sliding_window: int | None = None

    def same_layout(self, other: "KVShape") -> bool:
        return (
            self.num_layers == other.num_layers
            and self.num_key_value_heads == other.num_key_value_heads
            and self.head_dim == other.head_dim
            and self.dtype == other.dtype
            and self.rope_theta == other.rope_theta
            and self.sliding_window == other.sliding_window
        )

    def projection_required(self, other: "KVShape") -> bool:
        return not self.same_layout(other)


@dataclass(frozen=True)
class ModelRef:
    """Stable identity for a model, a LoRA adapter, or a model size variant."""

    model_id: str
    family: str
    architecture: str
    tokenizer_id: str
    kv_shape: KVShape
    parameter_count_b: float | None = None
    base_model_id: str | None = None
    lora_adapter_id: str | None = None
    revision: str | None = None
    metadata: dict[str, str | int | float | bool] = field(default_factory=dict)

    @property
    def is_lora(self) -> bool:
        return self.lora_adapter_id is not None

    @property
    def canonical_base_model_id(self) -> str:
        return self.base_model_id or self.model_id

    def shares_tokenizer_with(self, other: "ModelRef") -> bool:
        return self.tokenizer_id == other.tokenizer_id

    def same_family_architecture(self, other: "ModelRef") -> bool:
        return self.family == other.family and self.architecture == other.architecture


@dataclass(frozen=True)
class ReuseRequest:
    """A control-plane request made before asking LMCache to reuse KV."""

    source: ModelRef
    target: ModelRef
    prefix_hash: str
    prompt_tokens: int | None = None
    calibration_id: str | None = None
    allow_cross_base: bool = False
    quality_floor: float = 0.95
    metadata: dict[str, str | int | float | bool] = field(default_factory=dict)


@dataclass(frozen=True)
class ReusePlan:
    """A non-invasive plan that the LMCache patch can either execute or skip."""

    request: ReuseRequest
    scenario: ReuseScenario
    strategy: ReuseStrategy
    status: PlanStatus
    confidence: float
    transform_id: str
    lmcache_lookup_model_id: str
    required_gates: tuple[str, ...]
    patch_hooks: tuple[str, ...]
    notes: tuple[str, ...] = ()

    @property
    def executable(self) -> bool:
        return self.status == PlanStatus.READY

    def as_metadata(self) -> dict[str, str | float | bool]:
        """Metadata fields that can be attached to an LMCache lookup/store path."""

        return {
            "ge_scenario": self.scenario.value,
            "ge_strategy": self.strategy.value,
            "ge_status": self.status.value,
            "ge_confidence": self.confidence,
            "ge_transform_id": self.transform_id,
            "ge_source_model_id": self.request.source.model_id,
            "ge_target_model_id": self.request.target.model_id,
            "ge_prefix_hash": self.request.prefix_hash,
            "ge_executable": self.executable,
        }
