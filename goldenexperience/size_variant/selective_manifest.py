"""Manifest v5 contracts for calibrated, selective cross-scale KV reuse."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, field, replace
from enum import Enum
from pathlib import Path
from typing import Any

from goldenexperience.size_variant.cached_kv_manifest import (
    CACHED_KV_V5_SCHEMA_VERSION,
    CachedKVModelSpec,
    _is_sha256,
)

V5_LAYOUT = "kv_layer_head_token_head_dim"
V5_METHOD = "head_aware_attention_preserving_transport"
V5_FEATURE_SCHEMA = "goldenexperience.source_kv_risk_features.v1"


class ArtifactState(str, Enum):
    """Publication and runtime authority carried by a v5 artifact."""

    VALIDATION_CANDIDATE = "validation_candidate"
    SEMANTIC_APPROVED = "semantic_approved"
    APPROVED = "approved"


@dataclass(frozen=True)
class TransportLossContract:
    native_generation: float = 1.0
    prompt_tail_distillation: float = 0.25
    attention_logit_kl: float = 0.5
    attention_output_mse: float = 0.5
    transformed_kv_anchor: float = 0.1

    def validate(self) -> list[str]:
        errors: list[str] = []
        for name, value in asdict(self).items():
            if not math.isfinite(value) or value < 0:
                errors.append(f"transport loss {name} must be finite and non-negative")
        if self.native_generation != 1.0:
            errors.append("native_generation loss weight must remain 1.0")
        return errors


@dataclass(frozen=True)
class TransportSpec:
    """Frozen head-aware transport architecture and content identity."""

    weights_uri: str
    weights_sha256: str
    rank: int = 64
    source_window: int = 3
    layer_mixer: str = "per_target_head_window_softmax"
    head_mixer: str = "per_target_head_source_softmax"
    projection: str = "independent_per_head_low_rank_kv"
    residual: str = "gated_silu"
    normalizer: str = "layer_head_ood_zscore"
    structure_id: str = "head_aware_transport_v1"
    loss: TransportLossContract = field(default_factory=TransportLossContract)

    def validate(self, source: CachedKVModelSpec) -> list[str]:
        errors: list[str] = []
        if not self.weights_uri.endswith(".safetensors"):
            errors.append("transport weights must use safetensors")
        if not _is_sha256(self.weights_sha256):
            errors.append("transport weights_sha256 must be a SHA-256 digest")
        if self.rank not in {32, 64, 128}:
            errors.append("transport rank must be one of 32, 64, or 128")
        if self.rank > source.head_dim:
            errors.append("transport rank exceeds source head dimension")
        if self.source_window not in {1, 3} or self.source_window > source.num_layers:
            errors.append("transport source_window must be 1 or 3 within source depth")
        expected = {
            "layer_mixer": "per_target_head_window_softmax",
            "head_mixer": "per_target_head_source_softmax",
            "projection": "independent_per_head_low_rank_kv",
            "residual": "gated_silu",
            "normalizer": "layer_head_ood_zscore",
            "structure_id": "head_aware_transport_v1",
        }
        for name, value in expected.items():
            if getattr(self, name) != value:
                errors.append(f"unsupported transport {name}")
        errors.extend(self.loss.validate())
        return errors


@dataclass(frozen=True)
class RiskGateSpec:
    """Direction-specific predictor and one-shot calibration evidence."""

    predictor_uri: str
    predictor_sha256: str
    threshold: float | None
    calibration_dataset_sha256: str | None
    calibration_method: str
    candidate_threshold_count: int
    accepted_count: int = 0
    total_count: int = 0
    error_count: int = 0
    coverage: float = 0.0
    regression_risk_upper_bound: float = 1.0
    confidence_level: float = 0.95
    feature_schema_version: str = V5_FEATURE_SCHEMA
    hidden_size: int = 64
    ood_threshold: float = 6.0
    min_shadow_samples: int = 1

    @property
    def calibrated(self) -> bool:
        return self.threshold is not None and self.calibration_dataset_sha256 is not None

    def artifact_errors(self) -> list[str]:
        from goldenexperience.size_variant.risk_gate import RISK_CALIBRATION_METHOD

        errors: list[str] = []
        if not self.predictor_uri.endswith(".safetensors"):
            errors.append("risk predictor must use safetensors")
        if not _is_sha256(self.predictor_sha256):
            errors.append("risk predictor_sha256 must be a SHA-256 digest")
        if self.feature_schema_version != V5_FEATURE_SCHEMA:
            errors.append("unsupported risk feature schema")
        if self.hidden_size != 64:
            errors.append("risk predictor hidden_size must be 64")
        if not math.isfinite(self.ood_threshold) or self.ood_threshold <= 0:
            errors.append("risk OOD threshold must be finite and positive")
        if self.min_shadow_samples < 1:
            errors.append("risk min_shadow_samples must be positive")
        if self.calibrated:
            if self.calibration_method != RISK_CALIBRATION_METHOD:
                errors.append("risk calibration must use Bonferroni-corrected Clopper-Pearson")
            if self.candidate_threshold_count < 1:
                errors.append("risk calibration candidate threshold count is invalid")
        return errors

    def calibration_errors(self, *, min_accepted: int = 300) -> list[str]:
        from goldenexperience.size_variant.risk_gate import (
            RISK_CALIBRATION_METHOD,
            bonferroni_adjusted_confidence,
            clopper_pearson_upper_bound,
        )

        errors = self.artifact_errors()
        if (
            self.threshold is None
            or not math.isfinite(self.threshold)
            or not 0 <= self.threshold <= 1
        ):
            errors.append("calibrated risk threshold must be between zero and one")
        if not _is_sha256(self.calibration_dataset_sha256):
            errors.append("risk calibration_dataset_sha256 must be a SHA-256 digest")
        if self.calibration_method != RISK_CALIBRATION_METHOD:
            errors.append("risk calibration must use Bonferroni-corrected Clopper-Pearson")
        candidate_count_valid = 1 <= self.candidate_threshold_count <= max(1, self.total_count)
        if not candidate_count_valid:
            errors.append("risk calibration candidate threshold count is invalid")
        if (
            self.total_count <= 0
            or self.accepted_count < 0
            or self.accepted_count > self.total_count
        ):
            errors.append("risk calibration counts are invalid")
        if self.error_count < 0 or self.error_count > self.accepted_count:
            errors.append("risk calibration error_count is invalid")
        if self.accepted_count < min_accepted:
            errors.append("risk calibration accepted count is below 300")
        expected_coverage = self.accepted_count / self.total_count if self.total_count > 0 else 0.0
        if not math.isfinite(self.coverage) or abs(self.coverage - expected_coverage) > 1e-9:
            errors.append("risk calibration coverage is inconsistent with counts")
        confidence_valid = math.isfinite(self.confidence_level) and self.confidence_level == 0.95
        if not confidence_valid:
            errors.append("risk calibration confidence_level must be 0.95")
        if (
            self.accepted_count > 0
            and 0 <= self.error_count <= self.accepted_count
            and candidate_count_valid
            and confidence_valid
        ):
            adjusted_confidence = bonferroni_adjusted_confidence(
                self.confidence_level,
                self.candidate_threshold_count,
            )
            expected_upper = clopper_pearson_upper_bound(
                self.error_count,
                self.accepted_count,
                confidence=adjusted_confidence,
            )
            if (
                not math.isfinite(self.regression_risk_upper_bound)
                or abs(self.regression_risk_upper_bound - expected_upper) > 1e-8
            ):
                errors.append("risk upper bound is inconsistent with calibration counts")
            elif self.regression_risk_upper_bound > 0.01:
                errors.append("risk upper bound exceeds one percent")
        return errors


@dataclass(frozen=True)
class SelectiveQualityThresholds:
    min_coverage: float = 0.30
    min_bridge_task_score: float = 0.95
    min_greedy_agreement: float = 0.98
    max_task_score_drop_pct: float = 1.0
    max_perplexity_drift_pct: float = 2.0
    max_regression_risk_upper_bound: float = 0.01
    min_accepted: int = 300

    def validate(self) -> list[str]:
        errors: list[str] = []
        for name, value in (
            ("min_coverage", self.min_coverage),
            ("min_bridge_task_score", self.min_bridge_task_score),
            ("min_greedy_agreement", self.min_greedy_agreement),
            ("max_regression_risk_upper_bound", self.max_regression_risk_upper_bound),
        ):
            if not _finite_number(value) or not 0 <= value <= 1:
                errors.append(f"{name} must be between zero and one")
        for name, value in (
            ("max_task_score_drop_pct", self.max_task_score_drop_pct),
            ("max_perplexity_drift_pct", self.max_perplexity_drift_pct),
        ):
            if not _finite_number(value) or value < 0:
                errors.append(f"{name} must be finite and non-negative")
        if type(self.min_accepted) is not int or self.min_accepted < 300:
            errors.append("selective quality min_accepted cannot be below 300")
        return errors


@dataclass(frozen=True)
class AcceptedSubsetQualityEvidence:
    evaluation_dataset_sha256: str
    total_count: int
    accepted_count: int
    unsafe_count: int
    coverage: float
    native_task_score: float
    bridge_task_score: float
    task_score_drop_pct: float
    greedy_agreement: float
    perplexity_drift_pct: float
    regression_risk_upper_bound: float
    key_cosine: float | None = None
    value_cosine: float | None = None

    def gate_errors(self, thresholds: SelectiveQualityThresholds) -> list[str]:
        from goldenexperience.size_variant.risk_gate import clopper_pearson_upper_bound

        errors = thresholds.validate()
        if not _is_sha256(self.evaluation_dataset_sha256):
            errors.append("quality evaluation_dataset_sha256 must be a SHA-256 digest")
        if self.total_count <= 0 or not 0 <= self.accepted_count <= self.total_count:
            errors.append("quality sample counts are invalid")
        if not 0 <= self.unsafe_count <= self.accepted_count:
            errors.append("quality unsafe_count is invalid")
        expected = self.accepted_count / self.total_count if self.total_count else 0.0
        if not math.isfinite(self.coverage) or abs(self.coverage - expected) > 1e-9:
            errors.append("quality coverage is inconsistent with counts")
        for name, value in (
            ("native_task_score", self.native_task_score),
            ("bridge_task_score", self.bridge_task_score),
            ("greedy_agreement", self.greedy_agreement),
            ("regression_risk_upper_bound", self.regression_risk_upper_bound),
        ):
            if not math.isfinite(value) or not 0 <= value <= 1:
                errors.append(f"quality {name} must be between zero and one")
        for name in ("key_cosine", "value_cosine"):
            value = getattr(self, name)
            if value is not None and (not math.isfinite(value) or not -1 <= value <= 1):
                errors.append(f"diagnostic {name} must be between minus one and one")
        for name, value in (
            ("task_score_drop_pct", self.task_score_drop_pct),
            ("perplexity_drift_pct", self.perplexity_drift_pct),
        ):
            if not math.isfinite(value) or value < 0:
                errors.append(f"quality {name} must be finite and non-negative")
        if self.accepted_count < thresholds.min_accepted:
            errors.append("accepted sample count is below threshold")
        if self.accepted_count > 0:
            expected_upper = clopper_pearson_upper_bound(
                self.unsafe_count,
                self.accepted_count,
            )
            if abs(self.regression_risk_upper_bound - expected_upper) > 1e-8:
                errors.append("quality risk upper bound is inconsistent with unsafe_count")
        if self.native_task_score > 0:
            expected_drop = max(
                0.0,
                (self.native_task_score - self.bridge_task_score) / self.native_task_score * 100,
            )
            if abs(self.task_score_drop_pct - expected_drop) > 1e-8:
                errors.append("quality task score drop is inconsistent")
        if self.coverage < thresholds.min_coverage:
            errors.append("accepted coverage is below threshold")
        if self.bridge_task_score < thresholds.min_bridge_task_score:
            errors.append("accepted bridge task score is below threshold")
        if self.greedy_agreement < thresholds.min_greedy_agreement:
            errors.append("accepted greedy agreement is below threshold")
        if self.task_score_drop_pct > thresholds.max_task_score_drop_pct:
            errors.append("accepted task score drop is above threshold")
        if self.perplexity_drift_pct > thresholds.max_perplexity_drift_pct:
            errors.append("accepted perplexity drift is above threshold")
        if self.regression_risk_upper_bound > thresholds.max_regression_risk_upper_bound:
            errors.append("accepted regression risk upper bound is above threshold")
        return errors


@dataclass(frozen=True)
class TransportQualityEvidence:
    evaluation_dataset_sha256: str
    prompt_count: int
    task_score: float
    oracle_safe_coverage: float
    greedy_agreement: float

    def gate_errors(self, expected_dataset_sha256: str) -> list[str]:
        errors: list[str] = []
        if self.evaluation_dataset_sha256 != expected_dataset_sha256:
            errors.append("transport quality refers to the wrong method-dev dataset")
        if self.prompt_count != 1024:
            errors.append("transport quality prompt_count must be 1024")
        for name, value in (
            ("task_score", self.task_score),
            ("oracle_safe_coverage", self.oracle_safe_coverage),
            ("greedy_agreement", self.greedy_agreement),
        ):
            if not math.isfinite(value) or not 0 <= value <= 1:
                errors.append(f"transport quality {name} must be between zero and one")
        if self.task_score < 0.95:
            errors.append("transport task score is below 0.95")
        if self.oracle_safe_coverage < 0.45:
            errors.append("transport oracle safe coverage is below 0.45")
        return errors


@dataclass(frozen=True)
class SemanticSealedEvidence:
    dataset_sha256: str
    report_sha256: str
    sample_count: int
    code_sha256: str
    transport_weights_sha256: str
    predictor_sha256: str
    threshold: float
    quality: AcceptedSubsetQualityEvidence
    opened_once: bool = True
    immutable_report: bool = True

    def validate(
        self,
        *,
        expected_dataset_sha256: str,
        transport: TransportSpec,
        risk_gate: RiskGateSpec,
        thresholds: SelectiveQualityThresholds,
    ) -> list[str]:
        errors: list[str] = []
        for name in ("dataset_sha256", "report_sha256", "code_sha256"):
            if not _is_sha256(getattr(self, name)):
                errors.append(f"semantic sealed {name} must be a SHA-256 digest")
        if self.dataset_sha256 != expected_dataset_sha256:
            errors.append("semantic sealed evidence refers to the wrong dataset")
        if self.quality.evaluation_dataset_sha256 != self.dataset_sha256:
            errors.append("semantic sealed quality refers to the wrong dataset")
        if self.sample_count != 2048:
            errors.append("semantic sealed sample count must be 2048")
        if self.quality.total_count != 2048:
            errors.append("semantic sealed quality must contain 2048 samples")
        if self.transport_weights_sha256 != transport.weights_sha256:
            errors.append("semantic sealed transport hash changed")
        if self.predictor_sha256 != risk_gate.predictor_sha256:
            errors.append("semantic sealed predictor hash changed")
        if risk_gate.threshold is None or self.threshold != risk_gate.threshold:
            errors.append("semantic sealed threshold changed")
        if not self.opened_once or not self.immutable_report:
            errors.append("semantic sealed evidence is not one-shot and immutable")
        errors.extend(self.quality.gate_errors(thresholds))
        return errors


@dataclass(frozen=True)
class RuntimeCostEvidence:
    report_sha256: str
    runtime_audit_dataset_sha256: str
    audit_requests: int
    warmup_iterations: int
    measured_iterations: int
    p95_materialization_ms: float
    p95_native_prefill_ms: float
    p95_materialization_to_prefill_ratio: float
    accepted_p95_ttft_reduction_pct: float
    rejected_p95_fallback_overhead_pct: float

    def validate(self, expected_audit_sha256: str) -> list[str]:
        errors: list[str] = []
        if not _is_sha256(self.report_sha256):
            errors.append("runtime report_sha256 must be a SHA-256 digest")
        if self.runtime_audit_dataset_sha256 != expected_audit_sha256:
            errors.append("runtime evidence refers to the wrong audit dataset")
        if self.audit_requests != 512:
            errors.append("runtime audit must contain exactly 512 requests")
        if self.warmup_iterations < 20 or self.measured_iterations < 100:
            errors.append("runtime latency evidence has too few iterations")
        values = (
            self.p95_materialization_ms,
            self.p95_native_prefill_ms,
            self.p95_materialization_to_prefill_ratio,
            self.accepted_p95_ttft_reduction_pct,
            self.rejected_p95_fallback_overhead_pct,
        )
        if any(not math.isfinite(value) or value < 0 for value in values):
            errors.append("runtime measurements must be finite and non-negative")
            return errors
        if self.p95_native_prefill_ms <= 0:
            errors.append("runtime native prefill P95 must be positive")
        else:
            expected_ratio = self.p95_materialization_ms / self.p95_native_prefill_ms
            if abs(self.p95_materialization_to_prefill_ratio - expected_ratio) > 1e-9:
                errors.append("runtime materialization ratio is inconsistent")
        if self.p95_materialization_to_prefill_ratio > 0.70:
            errors.append("runtime materialization P95 exceeds 0.70x native prefill")
        if self.accepted_p95_ttft_reduction_pct < 30.0:
            errors.append("accepted request TTFT reduction is below 30 percent")
        if self.rejected_p95_fallback_overhead_pct > 5.0:
            errors.append("rejected request fallback overhead exceeds five percent")
        return errors


@dataclass(frozen=True)
class DirectInjectionEvidence:
    report_sha256: str
    paged_slot_mapping_verified: bool
    load_complete_after_all_layers: bool
    partial_failure_invalidates_blocks: bool
    native_prefill_overwrites_invalid_blocks: bool
    accepted_target_mooncake_puts: int
    backing_files_remaining: int
    runtime_audit_passed: bool

    def validate(self) -> list[str]:
        errors: list[str] = []
        if not _is_sha256(self.report_sha256):
            errors.append("direct injection report_sha256 must be a SHA-256 digest")
        checks = {
            "paged slot mapping": self.paged_slot_mapping_verified,
            "atomic load-complete publication": self.load_complete_after_all_layers,
            "partial failure invalidation": self.partial_failure_invalidates_blocks,
            "native overwrite fallback": self.native_prefill_overwrites_invalid_blocks,
            "runtime audit": self.runtime_audit_passed,
        }
        errors.extend(
            f"direct injection {name} is not verified" for name, ok in checks.items() if not ok
        )
        if self.accepted_target_mooncake_puts != 0:
            errors.append("direct injection produced target Mooncake puts")
        if self.backing_files_remaining != 0:
            errors.append("direct injection left backing files")
        return errors


@dataclass(frozen=True)
class SelectiveKVBridgeManifest:
    """Content-addressed v5 artifact with monotonic publication authority."""

    artifact_id: str
    direction: str
    source: CachedKVModelSpec
    target: CachedKVModelSpec
    transport: TransportSpec
    risk_gate: RiskGateSpec
    benchmark_manifest_sha256: str
    transport_train_dataset_sha256: str
    selector_train_dataset_sha256: str
    method_dev_dataset_sha256: str
    risk_calibration_dataset_sha256: str
    validation_dataset_sha256: str
    semantic_sealed_dataset_sha256: str
    runtime_audit_dataset_sha256: str
    transport_quality: TransportQualityEvidence | None = None
    accepted_quality: AcceptedSubsetQualityEvidence | None = None
    semantic_sealed: SemanticSealedEvidence | None = None
    runtime_cost: RuntimeCostEvidence | None = None
    direct_injection: DirectInjectionEvidence | None = None
    thresholds: SelectiveQualityThresholds = field(default_factory=SelectiveQualityThresholds)
    state: ArtifactState = ArtifactState.VALIDATION_CANDIDATE
    schema_version: str = CACHED_KV_V5_SCHEMA_VERSION
    scope: str = "global"
    layout: str = V5_LAYOUT
    method: str = V5_METHOD
    rope_convention: str = "qwen_half_split"

    def __post_init__(self) -> None:
        if isinstance(self.state, str):
            object.__setattr__(self, "state", ArtifactState(self.state))

    @property
    def bridge_id(self) -> str:
        return self.artifact_id

    @property
    def approved(self) -> bool:
        return self.state is ArtifactState.APPROVED and not self.validate()

    @property
    def semantic_approved(self) -> bool:
        return (
            self.state in {ArtifactState.SEMANTIC_APPROVED, ArtifactState.APPROVED}
            and not self.semantic_errors()
        )

    def artifact_errors(self) -> list[str]:
        errors: list[str] = []
        if self.schema_version != CACHED_KV_V5_SCHEMA_VERSION:
            errors.append("unsupported selective cached KV schema_version")
        if not self.artifact_id:
            errors.append("artifact_id is required")
        if not self.direction:
            errors.append("direction is required")
        errors.extend(f"source: {item}" for item in self.source.validate())
        errors.extend(f"target: {item}" for item in self.target.validate())
        if self.source.model_id == self.target.model_id:
            errors.append("source and target model identities must differ")
        if self.source.architecture != self.target.architecture:
            errors.append("source and target architectures differ")
        if self.source.tokenizer_sha256 != self.target.tokenizer_sha256:
            errors.append("source and target tokenizer identities differ")
        if self.source.head_dim != self.target.head_dim:
            errors.append("source and target head dimensions differ")
        if self.source.rope_scaling != self.target.rope_scaling:
            errors.append("source and target RoPE scaling contracts differ")
        if self.source.sliding_window != self.target.sliding_window:
            errors.append("source and target sliding-window contracts differ")
        if self.layout != V5_LAYOUT or self.method != V5_METHOD:
            errors.append("unsupported v5 transport layout or method")
        if self.rope_convention != "qwen_half_split":
            errors.append("unsupported RoPE convention")
        if self.scope != "global":
            errors.append("selective transport must have global scope")
        errors.extend(self.transport.validate(self.source))
        errors.extend(self.risk_gate.artifact_errors())
        dataset_hashes = (
            self.transport_train_dataset_sha256,
            self.selector_train_dataset_sha256,
            self.method_dev_dataset_sha256,
            self.risk_calibration_dataset_sha256,
            self.validation_dataset_sha256,
            self.semantic_sealed_dataset_sha256,
            self.runtime_audit_dataset_sha256,
        )
        if not _is_sha256(self.benchmark_manifest_sha256) or any(
            not _is_sha256(value) for value in dataset_hashes
        ):
            errors.append("benchmark and split SHA-256 digests are required")
        if len(set(dataset_hashes)) != len(dataset_hashes):
            errors.append("benchmark split identities must be distinct")
        if self.risk_gate.calibration_dataset_sha256 not in {
            None,
            self.risk_calibration_dataset_sha256,
        }:
            errors.append("risk gate calibration hash differs from manifest split")
        if self.artifact_id and self.artifact_id != selective_artifact_id_for(self):
            errors.append("artifact_id does not match the content-addressed v5 manifest")
        return errors

    def validation_errors(self) -> list[str]:
        from goldenexperience.benchmarks.publication import SPLIT_COUNTS

        errors = self.risk_gate.calibration_errors(min_accepted=self.thresholds.min_accepted)
        if self.risk_gate.total_count != SPLIT_COUNTS["risk_calibration"]:
            errors.append("risk calibration must contain exactly 2048 samples")
        if self.transport_quality is None:
            errors.append("transport method-dev quality evidence is required")
        else:
            errors.extend(self.transport_quality.gate_errors(self.method_dev_dataset_sha256))
        if self.accepted_quality is None:
            errors.append("accepted validation quality evidence is required")
        else:
            if self.accepted_quality.evaluation_dataset_sha256 != self.validation_dataset_sha256:
                errors.append("accepted quality refers to the wrong validation dataset")
            if self.accepted_quality.total_count != SPLIT_COUNTS["validation"]:
                errors.append("accepted validation quality must contain 2048 samples")
            errors.extend(self.accepted_quality.gate_errors(self.thresholds))
        return errors

    def semantic_errors(self) -> list[str]:
        errors = self.artifact_errors() + self.validation_errors()
        if self.semantic_sealed is None:
            errors.append("semantic sealed evidence is required")
        else:
            errors.extend(
                self.semantic_sealed.validate(
                    expected_dataset_sha256=self.semantic_sealed_dataset_sha256,
                    transport=self.transport,
                    risk_gate=self.risk_gate,
                    thresholds=self.thresholds,
                )
            )
        return errors

    def validate(self) -> list[str]:
        errors = self.artifact_errors()
        if self.state is ArtifactState.VALIDATION_CANDIDATE:
            return errors
        errors = self.semantic_errors()
        if self.state is ArtifactState.SEMANTIC_APPROVED:
            if self.runtime_cost is not None or self.direct_injection is not None:
                errors.append("semantic_approved artifact cannot carry runtime approval evidence")
            return errors
        if self.runtime_cost is None:
            errors.append("approved artifact requires runtime cost evidence")
        else:
            errors.extend(self.runtime_cost.validate(self.runtime_audit_dataset_sha256))
        if self.direct_injection is None:
            errors.append("approved artifact requires direct injection evidence")
        else:
            errors.extend(self.direct_injection.validate())
        return errors

    def with_content_id(self) -> SelectiveKVBridgeManifest:
        return replace(self, artifact_id=selective_artifact_id_for(replace(self, artifact_id="")))

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["state"] = self.state.value
        for side in ("source", "target"):
            if payload[side].get("chat_template_sha256") is None:
                payload[side].pop("chat_template_sha256", None)
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> SelectiveKVBridgeManifest:
        transport_payload = dict(payload["transport"])
        transport_payload["loss"] = TransportLossContract(**transport_payload.get("loss", {}))
        risk_payload = dict(payload["risk_gate"])
        risk_payload.setdefault("calibration_method", "")
        risk_payload.setdefault("candidate_threshold_count", 0)
        quality_payload = payload.get("accepted_quality")
        transport_quality_payload = payload.get("transport_quality")
        sealed_payload = payload.get("semantic_sealed")
        if sealed_payload is not None:
            sealed_payload = dict(sealed_payload)
            sealed_payload["quality"] = AcceptedSubsetQualityEvidence(**sealed_payload["quality"])
        return cls(
            artifact_id=payload["artifact_id"],
            direction=payload["direction"],
            source=CachedKVModelSpec(**payload["source"]),
            target=CachedKVModelSpec(**payload["target"]),
            transport=TransportSpec(**transport_payload),
            risk_gate=RiskGateSpec(**risk_payload),
            benchmark_manifest_sha256=payload["benchmark_manifest_sha256"],
            transport_train_dataset_sha256=payload["transport_train_dataset_sha256"],
            selector_train_dataset_sha256=payload["selector_train_dataset_sha256"],
            method_dev_dataset_sha256=payload["method_dev_dataset_sha256"],
            risk_calibration_dataset_sha256=payload["risk_calibration_dataset_sha256"],
            validation_dataset_sha256=payload["validation_dataset_sha256"],
            semantic_sealed_dataset_sha256=payload["semantic_sealed_dataset_sha256"],
            runtime_audit_dataset_sha256=payload["runtime_audit_dataset_sha256"],
            transport_quality=(
                TransportQualityEvidence(**transport_quality_payload)
                if transport_quality_payload is not None
                else None
            ),
            accepted_quality=(
                AcceptedSubsetQualityEvidence(**quality_payload)
                if quality_payload is not None
                else None
            ),
            semantic_sealed=(SemanticSealedEvidence(**sealed_payload) if sealed_payload else None),
            runtime_cost=(
                RuntimeCostEvidence(**payload["runtime_cost"])
                if payload.get("runtime_cost") is not None
                else None
            ),
            direct_injection=(
                DirectInjectionEvidence(**payload["direct_injection"])
                if payload.get("direct_injection") is not None
                else None
            ),
            thresholds=SelectiveQualityThresholds(**payload.get("thresholds", {})),
            state=ArtifactState(payload.get("state", ArtifactState.VALIDATION_CANDIDATE.value)),
            schema_version=payload.get("schema_version", ""),
            scope=payload.get("scope", ""),
            layout=payload.get("layout", ""),
            method=payload.get("method", ""),
            rope_convention=payload.get("rope_convention", ""),
        )

    @classmethod
    def load(cls, path: str | Path) -> SelectiveKVBridgeManifest:
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    def save(self, path: str | Path) -> None:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

    def resolve_transport_weights_path(self, manifest_path: str | Path) -> Path:
        return _resolve_uri(self.transport.weights_uri, manifest_path)

    def resolve_predictor_path(self, manifest_path: str | Path) -> Path:
        return _resolve_uri(self.risk_gate.predictor_uri, manifest_path)


CachedKVBridgeManifestV5 = SelectiveKVBridgeManifest


def selective_artifact_id_for(manifest: SelectiveKVBridgeManifest) -> str:
    payload = manifest.to_dict()
    payload["artifact_id"] = ""
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return "selective-kv-" + hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]


def _resolve_uri(uri: str, manifest_path: str | Path) -> Path:
    path = Path(uri)
    if not path.is_absolute():
        path = Path(manifest_path).resolve().parent / path
    return path


def _finite_number(value: Any) -> bool:
    return not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(value)
