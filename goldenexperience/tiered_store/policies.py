"""Eviction, offload, and prefetch policies for tiered KV cache placement."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from goldenexperience.cache_core.block import CacheBlockMetadata, CacheQuery
from goldenexperience.cache_core.enums import DeviceTier
from goldenexperience.tiered_store.cost import TierCostModel, TierState


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
class LFUEvictionPolicy:
    """Least-frequently-used policy with age as the tie breaker."""

    prefer_low_quality: bool = True

    def select_victims(
        self,
        candidates: list[CacheBlockMetadata],
        required_bytes: int,
        protected_ids: set[str],
    ) -> list[str]:
        eligible = _eligible_candidates(candidates, protected_ids)
        if self.prefer_low_quality:
            eligible.sort(key=lambda item: (item.access_count, item.quality_score, item.last_accessed))
        else:
            eligible.sort(key=lambda item: (item.access_count, item.last_accessed))
        return _take_until_bytes(eligible, required_bytes)


@dataclass(slots=True)
class CostAwareEvictionPolicy:
    """Policy that keeps expensive-to-reload and high-quality blocks resident."""

    cost_model: TierCostModel = field(default_factory=TierCostModel)
    quality_weight: float = 4.0
    frequency_weight: float = 1.0
    recency_weight: float = 1.0
    reload_cost_weight: float = 1.0

    def select_victims(
        self,
        candidates: list[CacheBlockMetadata],
        required_bytes: int,
        protected_ids: set[str],
    ) -> list[str]:
        eligible = _eligible_candidates(candidates, protected_ids)
        if not eligible:
            return []
        newest_access = max(item.last_accessed for item in eligible)

        def score(item: CacheBlockMetadata) -> float:
            age = max(0.0, newest_access - item.last_accessed)
            reload_cost = self.cost_model.transfer_time_ms(
                item.bytes_size,
                item.device_tier,
                DeviceTier.HBM,
            )
            return (
                self.quality_weight * item.quality_score
                + self.frequency_weight * item.access_count
                - self.recency_weight * age
                + self.reload_cost_weight * reload_cost
            )

        eligible.sort(key=score)
        return _take_until_bytes(eligible, required_bytes)


@dataclass(slots=True)
class OffloadPlan:
    """A block-level offload request."""

    block_ids: list[str]
    target_tier: DeviceTier
    asynchronous: bool = False
    reason: str = "manual_offload"
    metadata: dict[str, str] = field(default_factory=dict)


@dataclass(slots=True)
class OffloadResult:
    """Result for moving one block between tiers."""

    block_id: str
    source_tier: DeviceTier | None
    target_tier: DeviceTier
    success: bool
    bytes_moved: int
    elapsed_ms: float
    reason: str = ""
    error: str | None = None


class OffloadPolicy(Protocol):
    def plan(
        self,
        candidates: list[CacheBlockMetadata],
        tier_states: dict[DeviceTier, TierState],
        demotion_path: dict[DeviceTier, DeviceTier | None],
        protected_ids: set[str],
    ) -> list[OffloadPlan]:
        """Return block-level offload plans."""


@dataclass(slots=True)
class WatermarkOffloadPolicy:
    """Demote blocks when a tier exceeds its high watermark.

    The policy chooses enough victims to move the tier back toward its low watermark. It
    never selects pinned or retained blocks.
    """

    high_watermark: float = 0.90
    low_watermark: float = 0.75
    eviction_policy: EvictionPolicy = field(default_factory=LRUEvictionPolicy)

    def plan(
        self,
        candidates: list[CacheBlockMetadata],
        tier_states: dict[DeviceTier, TierState],
        demotion_path: dict[DeviceTier, DeviceTier | None],
        protected_ids: set[str],
    ) -> list[OffloadPlan]:
        plans: list[OffloadPlan] = []
        for tier in (DeviceTier.HBM, DeviceTier.CPU):
            state = tier_states[tier]
            if state.capacity_bytes <= 0 or state.utilization <= self.high_watermark:
                continue
            target_tier = demotion_path.get(tier)
            if target_tier is None:
                continue
            target_used = int(state.capacity_bytes * self.low_watermark)
            required_bytes = max(0, state.used_bytes - target_used)
            tier_candidates = [item for item in candidates if item.device_tier == tier]
            victims = self.eviction_policy.select_victims(tier_candidates, required_bytes, protected_ids)
            if victims:
                plans.append(
                    OffloadPlan(
                        block_ids=victims,
                        target_tier=target_tier,
                        reason=f"watermark:{tier.value}>{self.high_watermark}",
                    )
                )
        return plans


@dataclass(slots=True)
class PrefetchPlan:
    """A placement request for warming blocks into a faster tier."""

    block_ids: list[str]
    target_tier: DeviceTier = DeviceTier.HBM
    asynchronous: bool = True
    reason: str = "manual"
    metadata: dict[str, str] = field(default_factory=dict)


@dataclass(slots=True)
class PrefetchContext:
    """Context passed to a prefetch policy."""

    query: CacheQuery = field(default_factory=CacheQuery)
    target_tier: DeviceTier = DeviceTier.HBM
    current_layer_id: int | None = None
    lookahead_layers: int = 1
    max_blocks: int | None = None
    max_bytes: int | None = None
    asynchronous: bool = True
    reason: str = "policy_prefetch"


class PrefetchPolicy(Protocol):
    def plan(self, candidates: list[CacheBlockMetadata], context: PrefetchContext) -> PrefetchPlan:
        """Return a prefetch plan for matching candidates."""


@dataclass(slots=True)
class DecodeWindowPrefetchPolicy:
    """Prefetch upcoming decoder layers for a matching prefix/session."""

    prefer_hot_prefixes: bool = True
    prefer_higher_quality: bool = True

    def plan(self, candidates: list[CacheBlockMetadata], context: PrefetchContext) -> PrefetchPlan:
        selected = [
            item
            for item in candidates
            if item.device_tier != context.target_tier
            and _inside_layer_window(item, context.current_layer_id, context.lookahead_layers)
        ]
        selected.sort(key=self._priority_key)

        block_ids: list[str] = []
        bytes_selected = 0
        for item in selected:
            if context.max_blocks is not None and len(block_ids) >= context.max_blocks:
                break
            if context.max_bytes is not None and bytes_selected + item.bytes_size > context.max_bytes:
                break
            block_ids.append(item.block_id)
            bytes_selected += item.bytes_size
        return PrefetchPlan(
            block_ids=block_ids,
            target_tier=context.target_tier,
            asynchronous=context.asynchronous,
            reason=context.reason,
            metadata={
                "policy": self.__class__.__name__,
                "current_layer_id": "" if context.current_layer_id is None else str(context.current_layer_id),
                "lookahead_layers": str(context.lookahead_layers),
            },
        )

    def _priority_key(self, item: CacheBlockMetadata) -> tuple[int, float, int, float]:
        quality = -item.quality_score if self.prefer_higher_quality else 0.0
        return (item.layer_id, quality, -item.access_count, -item.last_accessed)


@dataclass(slots=True)
class PrefixHotnessPrefetchPolicy:
    """Prefetch the hottest matching blocks regardless of layer window."""

    def plan(self, candidates: list[CacheBlockMetadata], context: PrefetchContext) -> PrefetchPlan:
        selected = [item for item in candidates if item.device_tier != context.target_tier]
        selected.sort(key=lambda item: (item.access_count, item.quality_score, item.last_accessed), reverse=True)
        block_ids: list[str] = []
        bytes_selected = 0
        for item in selected:
            if context.max_blocks is not None and len(block_ids) >= context.max_blocks:
                break
            if context.max_bytes is not None and bytes_selected + item.bytes_size > context.max_bytes:
                break
            block_ids.append(item.block_id)
            bytes_selected += item.bytes_size
        return PrefetchPlan(
            block_ids=block_ids,
            target_tier=context.target_tier,
            asynchronous=context.asynchronous,
            reason=context.reason,
            metadata={"policy": self.__class__.__name__},
        )


def _eligible_candidates(
    candidates: list[CacheBlockMetadata],
    protected_ids: set[str],
) -> list[CacheBlockMetadata]:
    return [
        item
        for item in candidates
        if item.block_id not in protected_ids and not item.pinned and item.ref_count == 0
    ]


def _take_until_bytes(candidates: list[CacheBlockMetadata], required_bytes: int) -> list[str]:
    victims: list[str] = []
    freed = 0
    for item in candidates:
        victims.append(item.block_id)
        freed += item.bytes_size
        if freed >= required_bytes:
            break
    return victims


def _inside_layer_window(
    item: CacheBlockMetadata,
    current_layer_id: int | None,
    lookahead_layers: int,
) -> bool:
    if current_layer_id is None:
        return True
    return current_layer_id < item.layer_id <= current_layer_id + max(1, lookahead_layers)
