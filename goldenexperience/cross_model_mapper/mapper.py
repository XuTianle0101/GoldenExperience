"""Cross-model KV mappers for same-family LLMs."""

from __future__ import annotations

import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from goldenexperience.cache_core.block import CacheBlock, KVPayload
from goldenexperience.engine_adapter.signature import ArchitectureSignature, CompatibilityLevel
from goldenexperience.utils.tensors import identity_projection, infer_shape, project_last_dim


@dataclass(slots=True)
class CalibrationPair:
    source: CacheBlock
    target: CacheBlock


@dataclass(slots=True)
class MappingResult:
    block: CacheBlock
    compatibility: CompatibilityLevel
    confidence: float
    mapper_id: str
    reason: str


class KVMapper(ABC):
    mapper_id: str

    @abstractmethod
    def fit(
        self,
        source_signature: ArchitectureSignature,
        target_signature: ArchitectureSignature,
        calibration_data: list[CalibrationPair] | None = None,
    ) -> "KVMapper":
        raise NotImplementedError

    @abstractmethod
    def transform(self, source_block: CacheBlock) -> MappingResult:
        raise NotImplementedError

    @abstractmethod
    def score(self, mapped_block: CacheBlock, probe_tokens: list[int] | None = None) -> float:
        raise NotImplementedError


class IdentityKVMapper(KVMapper):
    """Direct reuse mapper for exact or shape-compatible model variants."""

    def __init__(self) -> None:
        self.mapper_id = f"identity-{uuid.uuid4().hex[:8]}"
        self.source_signature: ArchitectureSignature | None = None
        self.target_signature: ArchitectureSignature | None = None
        self.compatibility = CompatibilityLevel.INCOMPATIBLE
        self.confidence = 0.0

    def fit(
        self,
        source_signature: ArchitectureSignature,
        target_signature: ArchitectureSignature,
        calibration_data: list[CalibrationPair] | None = None,
    ) -> "IdentityKVMapper":
        self.source_signature = source_signature
        self.target_signature = target_signature
        self.compatibility = source_signature.compatibility_with(target_signature)
        if self.compatibility == CompatibilityLevel.EXACT:
            self.confidence = 1.0
        elif self.compatibility == CompatibilityLevel.SHAPE_COMPATIBLE:
            self.confidence = 0.97
        else:
            self.confidence = 0.0
        return self

    def transform(self, source_block: CacheBlock) -> MappingResult:
        self._require_fit()
        if self.compatibility not in {
            CompatibilityLevel.EXACT,
            CompatibilityLevel.SHAPE_COMPATIBLE,
        }:
            raise ValueError(f"Identity mapping cannot handle {self.compatibility.value}.")
        assert self.target_signature is not None
        mapped = CacheBlock.from_payload(
            payload=source_block.payload,
            model_id=self.target_signature.model_id,
            layer_id=source_block.metadata.layer_id,
            head_id=source_block.metadata.head_id,
            token_range=source_block.metadata.token_range,
            dtype=self.target_signature.dtype,
            device_tier=source_block.metadata.device_tier,
            quality_score=min(source_block.metadata.quality_score, self.confidence),
            prefix_hash=source_block.metadata.prefix_hash,
            session_id=source_block.metadata.session_id,
            source_model_id=source_block.metadata.model_id,
            mapper_id=self.mapper_id,
        )
        return MappingResult(
            block=mapped,
            compatibility=self.compatibility,
            confidence=self.confidence,
            mapper_id=self.mapper_id,
            reason="direct shape-compatible reuse",
        )

    def score(self, mapped_block: CacheBlock, probe_tokens: list[int] | None = None) -> float:
        return min(mapped_block.metadata.quality_score, self.confidence)

    def _require_fit(self) -> None:
        if self.target_signature is None:
            raise RuntimeError("Mapper must be fit before transform.")


class LinearProjectionKVMapper(KVMapper):
    """Lightweight final-dimension projection for same-family shape mismatches."""

    def __init__(self) -> None:
        self.mapper_id = f"linear-{uuid.uuid4().hex[:8]}"
        self.source_signature: ArchitectureSignature | None = None
        self.target_signature: ArchitectureSignature | None = None
        self.compatibility = CompatibilityLevel.INCOMPATIBLE
        self.key_weight: Any | None = None
        self.value_weight: Any | None = None
        self.confidence = 0.0

    def fit(
        self,
        source_signature: ArchitectureSignature,
        target_signature: ArchitectureSignature,
        calibration_data: list[CalibrationPair] | None = None,
    ) -> "LinearProjectionKVMapper":
        self.source_signature = source_signature
        self.target_signature = target_signature
        self.compatibility = source_signature.compatibility_with(target_signature)
        if self.compatibility == CompatibilityLevel.INCOMPATIBLE:
            self.confidence = 0.0
            return self

        self.key_weight = identity_projection(source_signature.head_dim, target_signature.head_dim)
        self.value_weight = identity_projection(source_signature.head_dim, target_signature.head_dim)
        self.confidence = self._confidence_from_compatibility(self.compatibility)
        if calibration_data:
            self.confidence = min(0.99, self.confidence + min(0.03, len(calibration_data) * 0.005))
        return self

    def transform(self, source_block: CacheBlock) -> MappingResult:
        self._require_fit()
        if self.compatibility == CompatibilityLevel.INCOMPATIBLE:
            raise ValueError("Cannot map incompatible model families or architectures.")
        assert self.target_signature is not None
        assert self.key_weight is not None
        assert self.value_weight is not None

        key = project_last_dim(source_block.payload.key, self.key_weight)
        value = project_last_dim(source_block.payload.value, self.value_weight)
        payload = KVPayload(key=key, value=value)
        target_token_end = source_block.metadata.token_end
        if source_block.metadata.layer_id >= self.target_signature.num_layers:
            raise ValueError("Source layer is outside target model depth.")
        mapped = CacheBlock.from_payload(
            payload=payload,
            model_id=self.target_signature.model_id,
            layer_id=source_block.metadata.layer_id,
            head_id=source_block.metadata.head_id,
            token_range=(source_block.metadata.token_start, target_token_end),
            dtype=self.target_signature.dtype,
            device_tier=source_block.metadata.device_tier,
            quality_score=min(source_block.metadata.quality_score, self.confidence),
            prefix_hash=source_block.metadata.prefix_hash,
            session_id=source_block.metadata.session_id,
            source_model_id=source_block.metadata.model_id,
            mapper_id=self.mapper_id,
        )
        return MappingResult(
            block=mapped,
            compatibility=self.compatibility,
            confidence=self.confidence,
            mapper_id=self.mapper_id,
            reason=f"projected final dim to {self.target_signature.head_dim}",
        )

    def score(self, mapped_block: CacheBlock, probe_tokens: list[int] | None = None) -> float:
        expected = self.target_signature.head_dim if self.target_signature is not None else None
        key_shape = infer_shape(mapped_block.payload.key)
        if expected is not None and key_shape and key_shape[-1] != expected:
            return 0.0
        return min(mapped_block.metadata.quality_score, self.confidence)

    def _require_fit(self) -> None:
        if self.target_signature is None or self.source_signature is None:
            raise RuntimeError("Mapper must be fit before transform.")

    def _confidence_from_compatibility(self, compatibility: CompatibilityLevel) -> float:
        return {
            CompatibilityLevel.EXACT: 0.99,
            CompatibilityLevel.SHAPE_COMPATIBLE: 0.96,
            CompatibilityLevel.SHAPE_MISMATCH: 0.90,
            CompatibilityLevel.INCOMPATIBLE: 0.0,
        }[compatibility]

