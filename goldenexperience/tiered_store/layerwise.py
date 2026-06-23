"""Layerwise KV cache offload plans and result records."""

from __future__ import annotations

from dataclasses import dataclass, field

from goldenexperience.cache_core.block import CacheBlock, CacheQuery
from goldenexperience.cache_core.enums import DeviceTier


@dataclass(slots=True)
class LayerwiseOffloadPlan:
    """Plan for moving existing cache blocks one layer at a time."""

    query: CacheQuery
    target_tier: DeviceTier = DeviceTier.CPU
    layer_ids: list[int] | None = None
    asynchronous: bool = False
    pipeline_depth: int = 1
    reason: str = "layerwise_offload"
    metadata: dict[str, str] = field(default_factory=dict)


@dataclass(slots=True)
class LayerTransferResult:
    """Result for one layer transfer."""

    layer_id: int
    block_ids: list[str]
    target_tier: DeviceTier
    success: bool
    bytes_moved: int
    elapsed_ms: float
    failures: list[str] = field(default_factory=list)


@dataclass(slots=True)
class LayerRetrievalResult:
    """Result for retrieving one layer into a target tier."""

    layer_id: int
    blocks: list[CacheBlock]
    target_tier: DeviceTier
    hit: bool
    bytes_loaded: int
    elapsed_ms: float


@dataclass(slots=True)
class LayerGroup:
    """Engine-neutral grouping for shape-compatible KV layers."""

    layer_ids: list[int]
    shape: tuple[object, ...]
    dtype: str
    bytes_per_layer: int

