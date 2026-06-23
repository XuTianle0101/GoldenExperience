"""Quality and latency aware cross-model reuse decisions."""

from __future__ import annotations

from dataclasses import dataclass

from goldenexperience.cache_core.block import CacheBlockMetadata
from goldenexperience.cache_core.enums import ReuseAction
from goldenexperience.engine_adapter.signature import ArchitectureSignature, CompatibilityLevel


@dataclass(slots=True)
class ReuseDecision:
    action: ReuseAction
    reason: str
    expected_latency_savings_ms: float
    quality_score: float
    compatibility: CompatibilityLevel


@dataclass(slots=True)
class ReusePolicy:
    """Policy that chooses between reuse, warm start, and recompute."""

    direct_quality_threshold: float = 0.97
    partial_quality_threshold: float = 0.90
    min_prefix_similarity: float = 0.90
    min_latency_savings_ms: float = 1.0

    def decide(
        self,
        source_metadata: CacheBlockMetadata,
        source_signature: ArchitectureSignature,
        target_signature: ArchitectureSignature,
        mapper_confidence: float,
        prefix_similarity: float,
        expected_latency_savings_ms: float,
    ) -> ReuseDecision:
        compatibility = source_signature.compatibility_with(target_signature)
        quality_score = min(source_metadata.quality_score, mapper_confidence)
        if compatibility == CompatibilityLevel.INCOMPATIBLE:
            return ReuseDecision(
                action=ReuseAction.FALLBACK_RECOMPUTE,
                reason="incompatible architecture or model family",
                expected_latency_savings_ms=0.0,
                quality_score=quality_score,
                compatibility=compatibility,
            )
        if prefix_similarity < self.min_prefix_similarity:
            return ReuseDecision(
                action=ReuseAction.FALLBACK_RECOMPUTE,
                reason="prefix similarity below threshold",
                expected_latency_savings_ms=0.0,
                quality_score=quality_score,
                compatibility=compatibility,
            )
        if expected_latency_savings_ms < self.min_latency_savings_ms:
            return ReuseDecision(
                action=ReuseAction.WARM_START_RECOMPUTE,
                reason="reuse benefit is too small for direct injection",
                expected_latency_savings_ms=expected_latency_savings_ms,
                quality_score=quality_score,
                compatibility=compatibility,
            )
        if quality_score >= self.direct_quality_threshold:
            return ReuseDecision(
                action=ReuseAction.DIRECT_REUSE,
                reason="quality and latency gates passed",
                expected_latency_savings_ms=expected_latency_savings_ms,
                quality_score=quality_score,
                compatibility=compatibility,
            )
        if quality_score >= self.partial_quality_threshold:
            return ReuseDecision(
                action=ReuseAction.PARTIAL_REUSE,
                reason="quality supports partial prefix or layer reuse",
                expected_latency_savings_ms=expected_latency_savings_ms,
                quality_score=quality_score,
                compatibility=compatibility,
            )
        return ReuseDecision(
            action=ReuseAction.FALLBACK_RECOMPUTE,
            reason="quality gate rejected mapped KV",
            expected_latency_savings_ms=0.0,
            quality_score=quality_score,
            compatibility=compatibility,
        )

