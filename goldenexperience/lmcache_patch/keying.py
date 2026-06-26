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

    @classmethod
    def from_plan(cls, plan: ReusePlan) -> "CrossModelCacheKey":
        return cls(
            source_model_id=plan.request.source.model_id,
            target_model_id=plan.request.target.model_id,
            prefix_hash=plan.request.prefix_hash,
            scenario=plan.scenario.value,
            transform_id=plan.transform_id,
            calibration_id=plan.request.calibration_id,
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
        return fields
