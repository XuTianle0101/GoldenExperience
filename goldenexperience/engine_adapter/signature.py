"""Model architecture signatures used for compatibility decisions."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class CompatibilityLevel(str, Enum):
    EXACT = "exact"
    SHAPE_COMPATIBLE = "shape_compatible"
    SHAPE_MISMATCH = "shape_mismatch"
    INCOMPATIBLE = "incompatible"


@dataclass(frozen=True, slots=True)
class ArchitectureSignature:
    """Minimal model signature needed to reason about KV reuse."""

    model_id: str
    family: str
    architecture: str
    num_layers: int
    hidden_size: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    rope_theta: float | None = None
    rope_scaling: str | None = None
    sliding_window: int | None = None
    tokenizer_id: str | None = None
    tokenizer_hash: str | None = None
    dtype: str = "float16"
    vocab_size: int | None = None
    revision: str | None = None
    model_config_hash: str | None = None
    weights_hash: str | None = None
    extra: dict[str, str | int | float | bool] = field(default_factory=dict)

    def compatibility_with(self, target: "ArchitectureSignature") -> CompatibilityLevel:
        if self is target or self._same_verified_weights(target):
            return CompatibilityLevel.EXACT
        if self.family != target.family or self.architecture != target.architecture:
            return CompatibilityLevel.INCOMPATIBLE
        if not self._same_runtime_contract(target):
            return CompatibilityLevel.INCOMPATIBLE
        same_shape = (
            self.num_layers == target.num_layers
            and self.num_key_value_heads == target.num_key_value_heads
            and self.head_dim == target.head_dim
        )
        if same_shape:
            return CompatibilityLevel.SHAPE_COMPATIBLE
        if self.num_layers > 0 and target.num_layers > 0:
            return CompatibilityLevel.SHAPE_MISMATCH
        return CompatibilityLevel.INCOMPATIBLE

    def _same_runtime_contract(self, target: "ArchitectureSignature") -> bool:
        source_tokenizer = self.tokenizer_hash or self.tokenizer_id
        target_tokenizer = target.tokenizer_hash or target.tokenizer_id
        if source_tokenizer != target_tokenizer:
            return False
        if self.model_id != target.model_id and not source_tokenizer:
            return False
        return (
            self.dtype == target.dtype
            and self.rope_theta == target.rope_theta
            and self.rope_scaling == target.rope_scaling
            and self.sliding_window == target.sliding_window
        )

    def _same_verified_weights(self, target: "ArchitectureSignature") -> bool:
        if not _is_sha256(self.weights_hash) or self.weights_hash != target.weights_hash:
            return False
        if (
            not _is_sha256(self.model_config_hash)
            or self.model_config_hash != target.model_config_hash
        ):
            return False
        return (
            self.family == target.family
            and self.architecture == target.architecture
            and self.num_layers == target.num_layers
            and self.hidden_size == target.hidden_size
            and self.num_attention_heads == target.num_attention_heads
            and self.num_key_value_heads == target.num_key_value_heads
            and self.head_dim == target.head_dim
            and self._same_runtime_contract(target)
        )


def _is_sha256(value: str | None) -> bool:
    return bool(
        value
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value.lower())
    )
