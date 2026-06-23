"""Cache block metadata and payload containers."""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from goldenexperience.cache_core.enums import DeviceTier
from goldenexperience.utils.tensors import infer_shape, stable_digest, tensor_nbytes


@dataclass(slots=True)
class KVPayload:
    """Engine-neutral key/value payload.

    The key and value fields may be PyTorch tensors, NumPy arrays, or nested Python lists.
    The core store treats them as opaque payloads and uses utility functions for shape,
    bytes, and digest calculation.
    """

    key: Any
    value: Any

    @property
    def shape(self) -> tuple[tuple[int, ...], tuple[int, ...]]:
        return infer_shape(self.key), infer_shape(self.value)

    @property
    def nbytes(self) -> int:
        return tensor_nbytes(self.key) + tensor_nbytes(self.value)


@dataclass(slots=True)
class CacheBlockMetadata:
    """Metadata for a KV cache block.

    Payload bytes are stored separately in a tier backend. The metadata is small enough to
    keep in memory and can be indexed by model, prefix, and session.
    """

    block_id: str
    model_id: str
    layer_id: int
    head_id: int | None
    token_start: int
    token_end: int
    dtype: str
    device_tier: DeviceTier
    shape: tuple[Any, ...]
    checksum: str
    quality_score: float = 1.0
    prefix_hash: str | None = None
    session_id: str | None = None
    bytes_size: int = 0
    ref_count: int = 0
    pinned: bool = False
    created_at: float = field(default_factory=time.time)
    last_accessed: float = field(default_factory=time.time)
    access_count: int = 0
    source_model_id: str | None = None
    mapper_id: str | None = None

    @property
    def token_range(self) -> tuple[int, int]:
        return self.token_start, self.token_end

    def touch(self) -> None:
        self.last_accessed = time.time()
        self.access_count += 1

    def retain(self) -> None:
        self.ref_count += 1

    def release(self) -> None:
        if self.ref_count > 0:
            self.ref_count -= 1

    def overlaps_tokens(self, token_range: tuple[int, int] | None) -> bool:
        if token_range is None:
            return True
        start, end = token_range
        return self.token_start < end and start < self.token_end


@dataclass(slots=True)
class CacheQuery:
    """Query fields for locating a cache block or a group of compatible blocks."""

    model_id: str | None = None
    layer_id: int | None = None
    head_id: int | None = None
    token_range: tuple[int, int] | None = None
    prefix_hash: str | None = None
    session_id: str | None = None
    min_quality_score: float = 0.0
    tiers: tuple[DeviceTier, ...] | None = None

    def matches(self, metadata: CacheBlockMetadata) -> bool:
        if self.model_id is not None and metadata.model_id != self.model_id:
            return False
        if self.layer_id is not None and metadata.layer_id != self.layer_id:
            return False
        if self.head_id is not None and metadata.head_id != self.head_id:
            return False
        if self.prefix_hash is not None and metadata.prefix_hash != self.prefix_hash:
            return False
        if self.session_id is not None and metadata.session_id != self.session_id:
            return False
        if metadata.quality_score < self.min_quality_score:
            return False
        if self.tiers is not None and metadata.device_tier not in self.tiers:
            return False
        return metadata.overlaps_tokens(self.token_range)


@dataclass(slots=True)
class CacheBlock:
    """A KV payload plus metadata."""

    metadata: CacheBlockMetadata
    payload: KVPayload

    @classmethod
    def from_payload(
        cls,
        payload: KVPayload,
        model_id: str,
        layer_id: int,
        head_id: int | None,
        token_range: tuple[int, int],
        dtype: str,
        device_tier: DeviceTier,
        quality_score: float = 1.0,
        prefix_hash: str | None = None,
        session_id: str | None = None,
        source_model_id: str | None = None,
        mapper_id: str | None = None,
        block_id: str | None = None,
    ) -> "CacheBlock":
        checksum = stable_digest(payload)
        bytes_size = payload.nbytes
        metadata = CacheBlockMetadata(
            block_id=block_id or uuid.uuid4().hex,
            model_id=model_id,
            layer_id=layer_id,
            head_id=head_id,
            token_start=token_range[0],
            token_end=token_range[1],
            dtype=dtype,
            device_tier=device_tier,
            shape=payload.shape,
            checksum=checksum,
            quality_score=quality_score,
            prefix_hash=prefix_hash,
            session_id=session_id,
            bytes_size=bytes_size,
            source_model_id=source_model_id,
            mapper_id=mapper_id,
        )
        return cls(metadata=metadata, payload=payload)

    def refresh_integrity(self) -> None:
        self.metadata.shape = self.payload.shape
        self.metadata.bytes_size = self.payload.nbytes
        self.metadata.checksum = stable_digest(self.payload)

