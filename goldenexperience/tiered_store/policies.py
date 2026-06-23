"""Eviction and prefetch policies for tiered KV cache placement."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from goldenexperience.cache_core.block import CacheBlockMetadata
from goldenexperience.cache_core.enums import DeviceTier


class EvictionPolicy(Protocol):
    def select_victims(
        self,
        candidates: list[CacheBlockMetadata],
        required_bytes: int,
        protected_ids: set[str],
    ) -> list[str]:
        """Return block ids to evict or demote."""


@dataclass(slots=True)
class LRUEvictionPolicy:
    """Least-recently-used policy with pinned/ref-count protection."""

    prefer_low_quality: bool = True

    def select_victims(
        self,
        candidates: list[CacheBlockMetadata],
        required_bytes: int,
        protected_ids: set[str],
    ) -> list[str]:
        eligible = [
            item
            for item in candidates
            if item.block_id not in protected_ids and not item.pinned and item.ref_count == 0
        ]
        if self.prefer_low_quality:
            eligible.sort(key=lambda item: (item.quality_score, item.last_accessed))
        else:
            eligible.sort(key=lambda item: item.last_accessed)

        victims: list[str] = []
        freed = 0
        for item in eligible:
            victims.append(item.block_id)
            freed += item.bytes_size
            if freed >= required_bytes:
                break
        return victims


@dataclass(slots=True)
class PrefetchPlan:
    """A placement request for warming blocks into a faster tier."""

    block_ids: list[str]
    target_tier: DeviceTier = DeviceTier.HBM
    asynchronous: bool = True
    reason: str = "manual"
    metadata: dict[str, str] = field(default_factory=dict)

