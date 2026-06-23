"""Tiered engine-decoupled KV cache store."""

from __future__ import annotations

import threading
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any

from goldenexperience.cache_core.block import CacheBlock, CacheBlockMetadata, CacheQuery
from goldenexperience.cache_core.enums import DeviceTier
from goldenexperience.cache_core.index import CacheIndex
from goldenexperience.tiered_store.backend import MemoryTierBackend, NvmeTierBackend, TierBackend
from goldenexperience.tiered_store.cost import TierCostModel, TierState
from goldenexperience.tiered_store.policies import EvictionPolicy, LRUEvictionPolicy, PrefetchPlan
from goldenexperience.utils.tensors import move_payload_to_tier, stable_digest


DEMOTION_TARGET: dict[DeviceTier, DeviceTier | None] = {
    DeviceTier.HBM: DeviceTier.CPU,
    DeviceTier.CPU: DeviceTier.NVME,
    DeviceTier.NVME: None,
}


class TieredKVStore:
    """KV store with HBM, CPU, and NVMe tiers.

    The store is intentionally engine-neutral. Engine adapters export KVPayload objects,
    and this store handles metadata indexing, offload, prefetch, pin/release, and
    placement decisions.
    """

    def __init__(
        self,
        capacities: dict[DeviceTier, int] | None = None,
        nvme_path: str | Path = "artifacts/cache/nvme",
        eviction_policy: EvictionPolicy | None = None,
        cost_model: TierCostModel | None = None,
        max_workers: int = 2,
    ) -> None:
        self.capacities = capacities or {
            DeviceTier.HBM: 8 * 1024**3,
            DeviceTier.CPU: 64 * 1024**3,
            DeviceTier.NVME: 512 * 1024**3,
        }
        self.backends: dict[DeviceTier, TierBackend] = {
            DeviceTier.HBM: MemoryTierBackend(DeviceTier.HBM),
            DeviceTier.CPU: MemoryTierBackend(DeviceTier.CPU),
            DeviceTier.NVME: NvmeTierBackend(nvme_path),
        }
        self.index = CacheIndex()
        self.eviction_policy = eviction_policy or LRUEvictionPolicy()
        self.cost_model = cost_model or TierCostModel()
        self._lock = threading.RLock()
        self._executor = ThreadPoolExecutor(max_workers=max_workers)

    def put(self, block: CacheBlock) -> str:
        """Insert a cache block and return its block id."""

        with self._lock:
            block.refresh_integrity()
            existing = self.index.get(block.metadata.block_id)
            if existing is not None:
                self.backends[existing.device_tier].remove(block.metadata.block_id)
                self.index.remove(block.metadata.block_id)
            tier = block.metadata.device_tier
            self._ensure_capacity(tier, block.metadata.bytes_size, protected_ids={block.metadata.block_id})
            payload = self._prepare_payload_for_tier(block.payload, tier)
            self.backends[tier].put(block.metadata.block_id, payload)
            self.index.add(block.metadata)
            return block.metadata.block_id

    def get(self, query: CacheQuery, promote_to: DeviceTier | None = None) -> CacheBlock | None:
        with self._lock:
            matches = self.index.find(query)
            if not matches:
                return None
            return self._load(matches[0].block_id, promote_to=promote_to)

    def get_many(self, query: CacheQuery, limit: int | None = None) -> list[CacheBlock]:
        with self._lock:
            matches = self.index.find(query)
            if limit is not None:
                matches = matches[:limit]
            blocks = []
            for metadata in matches:
                block = self._load(metadata.block_id)
                if block is not None:
                    blocks.append(block)
            return blocks

    def get_by_id(self, block_id: str, promote_to: DeviceTier | None = None) -> CacheBlock | None:
        with self._lock:
            return self._load(block_id, promote_to=promote_to)

    def offload(self, block_id: str, target_tier: DeviceTier) -> bool:
        """Move a block to a different tier."""

        with self._lock:
            metadata = self.index.get(block_id)
            if metadata is None:
                return False
            if metadata.device_tier == target_tier:
                return True
            payload = self.backends[metadata.device_tier].get(block_id)
            if payload is None:
                return False
            self._ensure_capacity(target_tier, metadata.bytes_size, protected_ids={block_id})
            payload = self._prepare_payload_for_tier(payload, target_tier)
            self.backends[target_tier].put(block_id, payload)
            self.backends[metadata.device_tier].remove(block_id)
            metadata.device_tier = target_tier
            metadata.touch()
            return True

    def prefetch(self, plan: PrefetchPlan) -> list[Future[bool]] | list[bool]:
        """Promote blocks according to a prefetch plan."""

        if plan.asynchronous:
            return [self._executor.submit(self.offload, block_id, plan.target_tier) for block_id in plan.block_ids]
        return [self.offload(block_id, plan.target_tier) for block_id in plan.block_ids]

    def evict(self, query: CacheQuery | None = None, required_bytes: int = 0) -> list[str]:
        """Remove blocks selected by the eviction policy."""

        with self._lock:
            candidates = self.index.find(query or CacheQuery())
            victim_ids = self.eviction_policy.select_victims(candidates, required_bytes, protected_ids=set())
            for block_id in victim_ids:
                self.remove(block_id)
            return victim_ids

    def remove(self, block_id: str) -> bool:
        with self._lock:
            metadata = self.index.remove(block_id)
            if metadata is None:
                return False
            self.backends[metadata.device_tier].remove(block_id)
            return True

    def pin(self, block_id: str) -> bool:
        metadata = self.index.get(block_id)
        if metadata is None:
            return False
        metadata.pinned = True
        metadata.retain()
        return True

    def release(self, block_id: str) -> bool:
        metadata = self.index.get(block_id)
        if metadata is None:
            return False
        metadata.release()
        if metadata.ref_count == 0:
            metadata.pinned = False
        return True

    def tier_states(self) -> dict[DeviceTier, TierState]:
        return {
            tier: TierState(
                tier=tier,
                capacity_bytes=self.capacities.get(tier, 0),
                used_bytes=backend.bytes_used(),
                bandwidth_gbps=self.cost_model.bandwidth_gbps(tier),
                latency_us=self.cost_model.latency_us(tier),
            )
            for tier, backend in self.backends.items()
        }

    def shutdown(self) -> None:
        self._executor.shutdown(wait=True)

    def _load(self, block_id: str, promote_to: DeviceTier | None = None) -> CacheBlock | None:
        metadata = self.index.get(block_id)
        if metadata is None:
            return None
        if promote_to is not None and promote_to != metadata.device_tier:
            self.offload(block_id, promote_to)
            metadata = self.index.get(block_id)
            if metadata is None:
                return None
        payload = self.backends[metadata.device_tier].get(block_id)
        if payload is None:
            return None
        if stable_digest(payload) != metadata.checksum:
            raise ValueError(f"Checksum mismatch for cache block {block_id}")
        metadata.touch()
        return CacheBlock(metadata=metadata, payload=payload)

    def _prepare_payload_for_tier(self, payload: Any, tier: DeviceTier) -> Any:
        return move_payload_to_tier(payload, tier.value, pin_cpu=(tier == DeviceTier.CPU))

    def _ensure_capacity(self, tier: DeviceTier, incoming_bytes: int, protected_ids: set[str]) -> None:
        capacity = self.capacities.get(tier, 0)
        if capacity <= 0:
            return
        backend = self.backends[tier]
        if backend.bytes_used() + incoming_bytes <= capacity:
            return

        required = backend.bytes_used() + incoming_bytes - capacity
        candidates = [item for item in self.index.all() if item.device_tier == tier]
        victims = self.eviction_policy.select_victims(candidates, required, protected_ids)
        for victim_id in victims:
            metadata = self.index.get(victim_id)
            if metadata is None:
                continue
            lower_tier = DEMOTION_TARGET[metadata.device_tier]
            if lower_tier is None:
                self.remove(victim_id)
            else:
                self.offload(victim_id, lower_tier)
