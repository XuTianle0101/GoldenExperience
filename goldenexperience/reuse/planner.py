"""Planner for the three GoldenExperience cross-model reuse scenarios."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

from goldenexperience.reuse.models import (
    ModelRef,
    PlanStatus,
    ReusePlan,
    ReuseRequest,
    ReuseScenario,
    ReuseStrategy,
)


@dataclass(frozen=True)
class ScenarioDescriptor:
    """Human-readable contract for one reuse scenario."""

    scenario: ReuseScenario
    goal: str
    default_strategy: ReuseStrategy
    required_evidence: tuple[str, ...]
    development_focus: tuple[str, ...]


class CrossModelReusePlanner:
    """Classify source/target pairs and produce an LMCache-safe reuse plan.

    The planner only decides whether GoldenExperience should ask the LMCache patch path
    to look for a compatible entry. It does not mutate SGLang scheduling, LMCache offload,
    or engine-owned KV tensors.
    """

    def __init__(
        self,
        lora_confidence: float = 0.97,
        same_shape_confidence: float = 0.94,
        projection_confidence: float = 0.86,
        cross_base_confidence: float = 0.55,
    ) -> None:
        self.lora_confidence = lora_confidence
        self.same_shape_confidence = same_shape_confidence
        self.projection_confidence = projection_confidence
        self.cross_base_confidence = cross_base_confidence

    def plan(self, request: ReuseRequest) -> ReusePlan:
        if self._is_lora_pair(request.source, request.target):
            return self._plan_lora(request)
        if self._is_same_model_size_variant(request.source, request.target):
            return self._plan_size_variant(request)
        return self._plan_cross_base(request)

    def scenario_matrix(self) -> list[ScenarioDescriptor]:
        return [
            ScenarioDescriptor(
                scenario=ReuseScenario.LORA_ADAPTER,
                goal="Reuse KV between a base model and its LoRA-adapted model in both directions.",
                default_strategy=ReuseStrategy.ADAPTER_DELTA_GATED_ALIAS,
                required_evidence=(
                    "same base_model_id",
                    "same tokenizer_id",
                    "same KV layout",
                    "adapter delta or probe-logit quality gate",
                ),
                development_focus=(
                    "carry LoRA adapter metadata in LMCache keys",
                    "add a cheap LoRA drift gate before direct injection",
                    "record fallback reasons when adapter drift is high",
                ),
            ),
            ScenarioDescriptor(
                scenario=ReuseScenario.SAME_MODEL_SIZE_VARIANT,
                goal="Reuse KV across parameter-size variants of the same model line.",
                default_strategy=ReuseStrategy.LAYERWISE_PROJECTION,
                required_evidence=(
                    "same family and architecture",
                    "tokenizer alignment",
                    "layer/head mapping",
                    "projection calibration when KV shapes differ",
                ),
                development_focus=(
                    "layer alignment tables",
                    "head_dim projection materializers",
                    "partial-depth reuse policy",
                ),
            ),
            ScenarioDescriptor(
                scenario=ReuseScenario.CROSS_BASE_MODEL,
                goal="Explore reuse between different base models under an explicit calibration gate.",
                default_strategy=ReuseStrategy.LEARNED_CROSS_BASE_TRANSLATOR,
                required_evidence=(
                    "offline calibration set",
                    "tokenizer bridge or shared prompt canonicalization",
                    "probe quality gate",
                    "per-task allowlist",
                ),
                development_focus=(
                    "learned cross-base translator interface",
                    "strict fallback-to-recompute defaults",
                    "evaluation harness for quality regression",
                ),
            ),
        ]

    def _plan_lora(self, request: ReuseRequest) -> ReusePlan:
        gates = [
            "same_base_model_id",
            "same_tokenizer_id",
            "same_kv_shape",
            "lora_delta_quality_gate",
        ]
        status = PlanStatus.READY
        notes = ["LoRA pair detected; inference and offload remain owned by SGLang and LMCache."]
        if not request.source.shares_tokenizer_with(request.target):
            status = PlanStatus.BLOCKED
            notes.append("Tokenizers differ, so direct KV aliasing is unsafe.")
        if not request.source.kv_shape.same_layout(request.target.kv_shape):
            status = PlanStatus.BLOCKED
            notes.append("KV layout differs for the base/LoRA pair.")
        return self._make_plan(
            request=request,
            scenario=ReuseScenario.LORA_ADAPTER,
            strategy=ReuseStrategy.ADAPTER_DELTA_GATED_ALIAS,
            status=status,
            confidence=self.lora_confidence if status == PlanStatus.READY else 0.0,
            required_gates=tuple(gates),
            notes=tuple(notes),
        )

    def _plan_size_variant(self, request: ReuseRequest) -> ReusePlan:
        same_layout = request.source.kv_shape.same_layout(request.target.kv_shape)
        strategy = ReuseStrategy.DIRECT_SHAPE_ALIAS if same_layout else ReuseStrategy.LAYERWISE_PROJECTION
        status = PlanStatus.READY if same_layout or request.calibration_id else PlanStatus.NEEDS_CALIBRATION
        confidence = self.same_shape_confidence if same_layout else self.projection_confidence
        gates = [
            "same_family_architecture",
            "tokenizer_alignment",
            "layer_mapping",
        ]
        if not same_layout:
            gates.append("projection_calibration")
        notes = [
            "Same model line with a parameter-size change; reuse is limited to mapped prefix KV.",
        ]
        if not request.source.shares_tokenizer_with(request.target):
            status = PlanStatus.BLOCKED
            confidence = 0.0
            notes.append("Tokenizers differ inside the same model line.")
        elif not same_layout and request.calibration_id is None:
            notes.append("Projection path is scaffolded but needs calibration before execution.")
        return self._make_plan(
            request=request,
            scenario=ReuseScenario.SAME_MODEL_SIZE_VARIANT,
            strategy=strategy,
            status=status,
            confidence=confidence,
            required_gates=tuple(gates),
            notes=tuple(notes),
        )

    def _plan_cross_base(self, request: ReuseRequest) -> ReusePlan:
        status = PlanStatus.READY if request.allow_cross_base and request.calibration_id else PlanStatus.NEEDS_CALIBRATION
        confidence = self.cross_base_confidence if status == PlanStatus.READY else 0.0
        notes = [
            "Different base model detected; default behavior is conservative and must fallback cleanly.",
        ]
        if not request.allow_cross_base:
            notes.append("Set allow_cross_base only for experiments with an explicit task allowlist.")
        if request.calibration_id is None:
            notes.append("A calibration_id is required before the LMCache patch executes cross-base reuse.")
        return self._make_plan(
            request=request,
            scenario=ReuseScenario.CROSS_BASE_MODEL,
            strategy=ReuseStrategy.LEARNED_CROSS_BASE_TRANSLATOR,
            status=status,
            confidence=confidence,
            required_gates=(
                "calibration_dataset",
                "tokenizer_bridge",
                "learned_translator",
                "probe_quality_gate",
                "task_allowlist",
            ),
            notes=tuple(notes),
        )

    def _make_plan(
        self,
        request: ReuseRequest,
        scenario: ReuseScenario,
        strategy: ReuseStrategy,
        status: PlanStatus,
        confidence: float,
        required_gates: tuple[str, ...],
        notes: tuple[str, ...],
    ) -> ReusePlan:
        transform_id = self._transform_id(request, scenario, strategy)
        return ReusePlan(
            request=request,
            scenario=scenario,
            strategy=strategy,
            status=status,
            confidence=round(confidence, 4),
            transform_id=transform_id,
            lmcache_lookup_model_id=request.source.model_id,
            required_gates=required_gates,
            patch_hooks=(
                "sglang_request_metadata",
                "lmcache_cross_model_lookup",
                "goldenexperience_materializer",
                "quality_gate_accounting",
            ),
            notes=notes,
        )

    def _is_lora_pair(self, source: ModelRef, target: ModelRef) -> bool:
        same_base = source.canonical_base_model_id == target.canonical_base_model_id
        return same_base and (source.is_lora or target.is_lora)

    def _is_same_model_size_variant(self, source: ModelRef, target: ModelRef) -> bool:
        if source.is_lora or target.is_lora:
            return False
        if not source.same_family_architecture(target):
            return False
        if source.model_id == target.model_id:
            return False
        if source.parameter_count_b is None or target.parameter_count_b is None:
            return True
        return source.parameter_count_b != target.parameter_count_b

    def _transform_id(
        self,
        request: ReuseRequest,
        scenario: ReuseScenario,
        strategy: ReuseStrategy,
    ) -> str:
        raw = "|".join(
            [
                request.source.model_id,
                request.target.model_id,
                request.prefix_hash,
                scenario.value,
                strategy.value,
                request.calibration_id or "uncalibrated",
            ]
        )
        return "ge-" + hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]
