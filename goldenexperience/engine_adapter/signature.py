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
    tokenizer_id: str | None = None
    dtype: str = "float16"
    vocab_size: int | None = None
    extra: dict[str, str | int | float | bool] = field(default_factory=dict)

    def compatibility_with(self, target: "ArchitectureSignature") -> CompatibilityLevel:
        if self == target:
            return CompatibilityLevel.EXACT
        if self.family != target.family or self.architecture != target.architecture:
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

