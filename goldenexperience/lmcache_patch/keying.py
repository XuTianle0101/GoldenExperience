"""Key metadata helpers for cross-model LMCache lookups."""

from __future__ import annotations

from dataclasses import dataclass

from goldenexperience.reuse.models import ReusePlan


@dataclass(frozen=True)
class CrossModelCacheKey:
    """Sidecar key fields for LMCache entries considered across models."""

    source_model_id: str
    target_model_id: str
    prefix_hash: str
    scenario: str
    transform_id: str
    calibration_id: str | None = None
    pair_id: str | None = None
    direction: str | None = None
    source_config_hash: str | None = None
    target_config_hash: str | None = None
    layer_map_id: str | None = None
    projection_id: str | None = None
    fallback_reason: str | None = None

    @classmethod
    def from_plan(cls, plan: ReusePlan) -> "CrossModelCacheKey":
        return cls(
            source_model_id=plan.request.source.model_id,
            target_model_id=plan.request.target.model_id,
            prefix_hash=plan.request.prefix_hash,
            scenario=plan.scenario.value,
            transform_id=plan.transform_id,
            calibration_id=plan.request.calibration_id,
            pair_id=plan.pair_id,
            direction=plan.direction,
            source_config_hash=plan.request.source.kv_shape.model_config_hash,
            target_config_hash=plan.request.target.kv_shape.model_config_hash,
            layer_map_id=plan.layer_map_id,
            projection_id=plan.projection_id,
            fallback_reason=plan.fallback_reason,
        )

    def to_sidecar_fields(self) -> dict[str, str]:
        fields = {
            "ge_source_model_id": self.source_model_id,
            "ge_target_model_id": self.target_model_id,
            "ge_prefix_hash": self.prefix_hash,
            "ge_scenario": self.scenario,
            "ge_transform_id": self.transform_id,
        }
        if self.calibration_id is not None:
            fields["ge_calibration_id"] = self.calibration_id
        optional = {
            "ge_pair_id": self.pair_id,
            "ge_direction": self.direction,
            "ge_source_config_hash": self.source_config_hash,
            "ge_target_config_hash": self.target_config_hash,
            "ge_layer_map_id": self.layer_map_id,
            "ge_projection_id": self.projection_id,
            "ge_fallback_reason": self.fallback_reason,
        }
        for key, value in optional.items():
            if value is not None:
                fields[key] = value
        return fields
