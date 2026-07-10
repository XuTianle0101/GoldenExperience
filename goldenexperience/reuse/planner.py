"""Planner for the three GoldenExperience cross-model reuse scenarios."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from pathlib import Path

from goldenexperience.reuse.models import (
    ModelRef,
    PlanStatus,
    ReusePlan,
    ReuseRequest,
    ReuseScenario,
    ReuseStrategy,
)
from goldenexperience.size_variant.models import (
    CalibrationManifest,
    FallbackReason,
    infer_direction,
    pair_id_for,
)
from goldenexperience.size_variant.projection import validate_projection_cost


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

    The planner only decides whether GoldenExperience should ask the LMCache MP patch path
    to look for a compatible entry. It does not mutate vLLM scheduling, LMCache MP offload,
    Mooncake Store persistence, or engine-owned KV tensors.
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
                default_strategy=ReuseStrategy.HIDDEN_STATE_BRIDGE,
                required_evidence=(
                    "same family and architecture",
                    "tokenizer alignment",
                    "layer/head mapping",
                    "hidden-state bridge calibration when KV shapes differ",
                ),
                development_focus=(
                    "layer alignment tables",
                    "pre-KV hidden-state bridge materializers",
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
        has_quality_evidence = (
            request.calibration_id is not None and request.lora_quality_score is not None
        )
        status = PlanStatus.READY if has_quality_evidence else PlanStatus.NEEDS_CALIBRATION
        confidence = float(request.lora_quality_score or 0.0)
        fallback_reason = None if has_quality_evidence else FallbackReason.MISSING_CALIBRATION.value
        notes = [
            "LoRA pair detected; inference remains owned by vLLM and shared KV by LMCache MP."
        ]
        if not request.source.shares_tokenizer_with(request.target):
            status = PlanStatus.BLOCKED
            confidence = 0.0
            fallback_reason = FallbackReason.TOKENIZER_MISMATCH.value
            notes.append("Tokenizers differ, so direct KV aliasing is unsafe.")
        if not request.source.kv_shape.same_layout(request.target.kv_shape):
            status = PlanStatus.BLOCKED
            confidence = 0.0
            fallback_reason = FallbackReason.PROJECTION_SHAPE_MISMATCH.value
            notes.append("KV layout differs for the base/LoRA pair.")
        if not has_quality_evidence:
            notes.append("LoRA reuse requires measured adapter-drift or probe quality evidence.")
        return self._make_plan(
            request=request,
            scenario=ReuseScenario.LORA_ADAPTER,
            strategy=ReuseStrategy.ADAPTER_DELTA_GATED_ALIAS,
            status=status,
            confidence=confidence,
            required_gates=tuple(gates),
            notes=tuple(notes),
            fallback_reason=fallback_reason,
        )

    def _plan_size_variant(self, request: ReuseRequest) -> ReusePlan:
        same_layout = request.source.kv_shape.same_layout(request.target.kv_shape)
        manifest = self._load_size_variant_manifest(request)
        manifest_errors: list[str] = []
        has_hidden_bridge = manifest is not None and manifest.hidden_bridge is not None
        strategy = ReuseStrategy.HIDDEN_STATE_BRIDGE
        if manifest is not None and not has_hidden_bridge:
            strategy = ReuseStrategy.KV_PROJECTION_BASELINE
        status = PlanStatus.NEEDS_CALIBRATION
        confidence = 0.0
        fallback_reason: str | None = None
        gates = [
            "same_family_architecture",
            "tokenizer_alignment",
            "layer_mapping",
        ]
        gates.extend(("hidden_bridge_calibration", "target_kv_restore"))
        notes = [
            "Same model line with a parameter-size change; reuse is limited to mapped prefix state.",
        ]
        if not request.source.shares_tokenizer_with(request.target):
            status = PlanStatus.BLOCKED
            confidence = 0.0
            fallback_reason = FallbackReason.TOKENIZER_MISMATCH.value
            notes.append("Tokenizers differ inside the same model line.")
        elif not request.source.kv_shape.same_runtime_contract(request.target.kv_shape):
            status = PlanStatus.BLOCKED
            confidence = 0.0
            fallback_reason = FallbackReason.ROPE_MISMATCH.value
            notes.append("Runtime KV contract differs; dtype, RoPE, or sliding-window settings are incompatible.")
        elif manifest is None:
            fallback_reason = FallbackReason.MISSING_CALIBRATION.value
            notes.append("Hidden-state bridge path needs a validated calibration artifact before execution.")
        elif not validate_projection_cost(
            request.estimated_materialization_ms,
            request.estimated_target_prefill_ms,
        ):
            status = PlanStatus.WARM_START_RECOMPUTE
            confidence = 0.0
            fallback_reason = FallbackReason.COST_GATE_FAILED.value
            notes.append("Projection cost gate failed; target prefill is cheaper than materialization.")

        if manifest is not None:
            manifest_errors = self._manifest_errors(request, manifest)
            if manifest_errors:
                status = PlanStatus.BLOCKED
                confidence = 0.0
                fallback_reason = FallbackReason.ARTIFACT_HASH_MISMATCH.value
                notes.extend(manifest_errors)
            else:
                gates.append("artifact_hash_match")
                if has_hidden_bridge:
                    status = PlanStatus.READY if status != PlanStatus.WARM_START_RECOMPUTE else status
                    gates.append("pre_kv_hidden_contract")
                    confidence = self._manifest_confidence(manifest)
                    notes.append("Using hidden-state bridge: h_small -> h_large_hat -> target W_K/W_V/RoPE -> KV.")
                else:
                    status = PlanStatus.READY if status != PlanStatus.WARM_START_RECOMPUTE else status
                    gates.append("kv_projection_baseline")
                    confidence = self._manifest_confidence(manifest)
                    notes.append("Using legacy KV projection baseline because artifact has no hidden bridge spec.")

        direction = infer_direction(request.source, request.target).value
        pair_id = pair_id_for(request.source, request.target)
        layer_map_id = manifest.layer_map_id if manifest is not None and not manifest_errors else None
        projection_id = (
            manifest.projection_id
            if manifest is not None and not manifest_errors and not has_hidden_bridge
            else None
        )
        hidden_bridge_id = manifest.hidden_bridge_id if manifest is not None and not manifest_errors else None
        restore_id = manifest.restore_id if manifest is not None and not manifest_errors else None
        state_kind = manifest.state_kind if manifest is not None and not manifest_errors else ("kv" if same_layout else "hidden")
        hidden_contract = (
            manifest.hidden_bridge.capture_point
            if manifest is not None and manifest.hidden_bridge is not None and not manifest_errors
            else None
        )
        target_kv_layout = (
            manifest.kv_restore.target_kv_layout
            if manifest is not None and manifest.kv_restore is not None and not manifest_errors
            else None
        )
        estimated_prefill_saved_ms = None
        if request.estimated_target_prefill_ms is not None and request.estimated_materialization_ms is not None:
            estimated_prefill_saved_ms = max(
                0.0,
                request.estimated_target_prefill_ms - request.estimated_materialization_ms,
            )
        return self._make_plan(
            request=request,
            scenario=ReuseScenario.SAME_MODEL_SIZE_VARIANT,
            strategy=strategy,
            status=status,
            confidence=confidence,
            required_gates=tuple(gates),
            notes=tuple(notes),
            direction=direction,
            pair_id=pair_id,
            artifact_uri=request.artifact_uri,
            layer_map_id=layer_map_id,
            projection_id=projection_id,
            hidden_bridge_id=hidden_bridge_id,
            restore_id=restore_id,
            state_kind=state_kind,
            hidden_contract=hidden_contract,
            target_kv_layout=target_kv_layout,
            estimated_prefill_saved_ms=estimated_prefill_saved_ms,
            estimated_materialization_ms=request.estimated_materialization_ms,
            fallback_reason=fallback_reason,
        )

    def _plan_cross_base(self, request: ReuseRequest) -> ReusePlan:
        status = PlanStatus.NEEDS_CALIBRATION
        confidence = 0.0
        notes = [
            "Different base model detected; default behavior is conservative and must fallback cleanly.",
        ]
        if not request.allow_cross_base:
            notes.append("Set allow_cross_base only for experiments with an explicit task allowlist.")
        if request.calibration_id is None:
            notes.append("A calibration_id is required before the LMCache patch executes cross-base reuse.")
        else:
            notes.append(
                "Cross-base execution remains disabled until translator, tokenizer bridge, "
                "probe, and task-allowlist artifacts have a verifiable schema."
            )
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
        direction: str | None = None,
        pair_id: str | None = None,
        artifact_uri: str | None = None,
        layer_map_id: str | None = None,
        projection_id: str | None = None,
        hidden_bridge_id: str | None = None,
        restore_id: str | None = None,
        state_kind: str | None = None,
        hidden_contract: str | None = None,
        target_kv_layout: str | None = None,
        estimated_prefill_saved_ms: float | None = None,
        estimated_materialization_ms: float | None = None,
        fallback_reason: str | None = None,
    ) -> ReusePlan:
        if status == PlanStatus.READY and (
            not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0
        ):
            status = PlanStatus.BLOCKED
            confidence = 0.0
            fallback_reason = FallbackReason.QUALITY_GATE_FAILED.value
            notes = (*notes, "measured confidence must be finite and between 0 and 1")
        elif status == PlanStatus.READY and not 0.0 <= request.quality_floor <= 1.0:
            status = PlanStatus.BLOCKED
            confidence = 0.0
            fallback_reason = FallbackReason.QUALITY_GATE_FAILED.value
            notes = (*notes, "quality_floor must be between 0 and 1")
        elif status == PlanStatus.READY and confidence < request.quality_floor:
            status = PlanStatus.BLOCKED
            fallback_reason = FallbackReason.QUALITY_GATE_FAILED.value
            notes = (
                *notes,
                f"Measured confidence {confidence:.4f} is below quality floor "
                f"{request.quality_floor:.4f}.",
            )
        transform_id = hidden_bridge_id or projection_id or self._transform_id(request, scenario, strategy)
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
                "engine_request_metadata",
                "lmcache_cross_model_lookup",
                "goldenexperience_materializer",
                "quality_gate_accounting",
            ),
            direction=direction,
            pair_id=pair_id,
            artifact_uri=artifact_uri,
            layer_map_id=layer_map_id,
            projection_id=projection_id,
            hidden_bridge_id=hidden_bridge_id,
            restore_id=restore_id,
            state_kind=state_kind,
            hidden_contract=hidden_contract,
            target_kv_layout=target_kv_layout,
            estimated_prefill_saved_ms=estimated_prefill_saved_ms,
            estimated_materialization_ms=estimated_materialization_ms,
            fallback_reason=fallback_reason,
            notes=notes,
        )

    def _load_size_variant_manifest(self, request: ReuseRequest) -> CalibrationManifest | None:
        if request.artifact_uri is None:
            return None
        path = Path(request.artifact_uri)
        if not path.exists():
            return None
        try:
            return CalibrationManifest.load(path)
        except (KeyError, TypeError, ValueError, OSError):
            return None

    def _manifest_errors(
        self,
        request: ReuseRequest,
        manifest: CalibrationManifest,
    ) -> list[str]:
        errors = manifest.validate()
        if manifest.calibration_id != request.calibration_id:
            errors.append("calibration_id differs from artifact")
        if manifest.source.model_id != request.source.model_id:
            errors.append("source model differs from artifact")
        if manifest.target.model_id != request.target.model_id:
            errors.append("target model differs from artifact")
        if (
            manifest.scope == "prefix_allowlist"
            and request.prefix_hash not in manifest.prefix_hash_allowlist
        ):
            errors.append("request prefix hash is outside artifact allowlist")
        source_hash = request.source.kv_shape.model_config_hash
        target_hash = request.target.kv_shape.model_config_hash
        if source_hash and manifest.source.kv_shape.model_config_hash and source_hash != manifest.source.kv_shape.model_config_hash:
            errors.append("source model_config_hash differs from artifact")
        if target_hash and manifest.target.kv_shape.model_config_hash and target_hash != manifest.target.kv_shape.model_config_hash:
            errors.append("target model_config_hash differs from artifact")
        return errors

    def _manifest_confidence(self, manifest: CalibrationManifest) -> float:
        values = [manifest.quality.kv_cosine, manifest.quality.attention_proxy_cosine]
        if manifest.hidden_bridge is not None:
            values.append(manifest.quality.hidden_cosine)
        positive = [value for value in values if value > 0.0]
        return min(positive) if positive else 0.0

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
