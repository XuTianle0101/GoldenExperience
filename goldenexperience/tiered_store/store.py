"""Tiered engine-decoupled KV cache store."""

from __future__ import annotations

import threading
import time
from collections.abc import Iterator, Mapping, Sequence
from concurrent.futures import Future, ThreadPoolExecutor, wait
from dataclasses import replace
from pathlib import Path
from typing import Any

from goldenexperience.cache_core.block import CacheBlock, CacheBlockMetadata, CacheQuery
from goldenexperience.cache_core.enums import DeviceTier
from goldenexperience.cache_core.index import CacheIndex
from goldenexperience.tiered_store.backend import MemoryTierBackend, NvmeTierBackend, TierBackend
from goldenexperience.tiered_store.cost import TierCostModel, TierState
from goldenexperience.tiered_store.layerwise import (
    LayerGroup,
    LayerRetrievalResult,
    LayerTransferResult,
    LayerwiseOffloadPlan,
)
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

    def layer_ids(self, query: CacheQuery) -> list[int]:
        """Return sorted layer ids that have blocks matching the query."""

        with self._lock:
            return self.index.layer_ids(query)

    def layer_groups(self, query: CacheQuery) -> list[LayerGroup]:
        """Group matched layers by KV shape and dtype.

        This mirrors the LMCache idea of grouping transfer-compatible layers by a
        metadata identity, while remaining independent of a specific serving engine.
        """

        with self._lock:
            groups: dict[tuple[tuple[object, ...], str, int], list[int]] = {}
            for layer_id, items in self.index.group_by_layer(query).items():
                if not items:
                    continue
                representative = items[0]
                key = (representative.shape, representative.dtype, representative.bytes_size)
                groups.setdefault(key, []).append(layer_id)
            return [
                LayerGroup(
                    layer_ids=layer_ids,
                    shape=shape,
                    dtype=dtype,
                    bytes_per_layer=bytes_per_layer,
                )
                for (shape, dtype, bytes_per_layer), layer_ids in sorted(
                    groups.items(),
                    key=lambda item: min(item[1]),
                )
            ]

    def get_layer(
        self,
        query: CacheQuery,
        layer_id: int,
        promote_to: DeviceTier | None = None,
    ) -> list[CacheBlock]:
        """Load all blocks from a single layer, optionally promoting the layer."""

        layer_query = replace(query, layer_id=layer_id)
        with self._lock:
            matches = self.index.find(layer_query)
        blocks: list[CacheBlock] = []
        for metadata in matches:
            block = self.get_by_id(metadata.block_id, promote_to=promote_to)
            if block is not None:
                blocks.append(block)
        blocks.sort(key=lambda block: (block.metadata.token_start, block.metadata.head_id or -1))
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

    def put_layers(
        self,
        blocks_by_layer: Mapping[int, Sequence[CacheBlock]] | Sequence[Sequence[CacheBlock]],
        target_tier: DeviceTier | None = None,
    ) -> Iterator[LayerTransferResult]:
        """Store KV blocks layer by layer with a one-layer pipeline.

        The caller can feed blocks in layer-major order from an engine adapter. Each
        yielded result means the previous layer has been materialized in the requested
        tier, which is the engine-neutral counterpart of LMCache's layerwise store loop.
        """

        layer_items = self._normalize_layer_input(blocks_by_layer)
        pending: Future[LayerTransferResult] | None = None
        for layer_id, blocks in layer_items:
            next_future = self._executor.submit(self._put_layer, layer_id, list(blocks), target_tier)
            if pending is not None:
                yield pending.result()
            pending = next_future
        if pending is not None:
            yield pending.result()

    def offload_layer(
        self,
        query: CacheQuery,
        layer_id: int,
        target_tier: DeviceTier,
    ) -> LayerTransferResult:
        """Move every matching block from one layer to the target tier."""

        start = time.perf_counter()
        layer_query = replace(query, layer_id=layer_id)
        with self._lock:
            metadata = list(self.index.find(layer_query))
        block_ids = [item.block_id for item in metadata]
        bytes_to_move = sum(item.bytes_size for item in metadata if item.device_tier != target_tier)
        failures = []
        for block_id in block_ids:
            if not self.offload(block_id, target_tier):
                failures.append(block_id)
        return LayerTransferResult(
            layer_id=layer_id,
            block_ids=block_ids,
            target_tier=target_tier,
            success=not failures,
            bytes_moved=bytes_to_move,
            elapsed_ms=(time.perf_counter() - start) * 1000.0,
            failures=failures,
        )

    def offload_layers(
        self,
        plan: LayerwiseOffloadPlan,
    ) -> list[LayerTransferResult] | list[Future[LayerTransferResult]]:
        """Move matching layers to a target tier.

        Synchronous mode returns ordered results. Asynchronous mode returns futures while
        respecting the requested pipeline depth.
        """

        layer_ids = plan.layer_ids or self.layer_ids(plan.query)
        if not plan.asynchronous:
            return [
                self.offload_layer(plan.query, layer_id, plan.target_tier)
                for layer_id in layer_ids
            ]

        depth = max(1, plan.pipeline_depth)
        futures: list[Future[LayerTransferResult]] = []
        in_flight: set[Future[LayerTransferResult]] = set()
        for layer_id in layer_ids:
            future = self._executor.submit(self.offload_layer, plan.query, layer_id, plan.target_tier)
            futures.append(future)
            in_flight.add(future)
            if len(in_flight) >= depth:
                done, in_flight = wait(in_flight, return_when="FIRST_COMPLETED")
                for item in done:
                    item.result()
        return futures

    def retrieve_layers(
        self,
        query: CacheQuery,
        target_tier: DeviceTier = DeviceTier.HBM,
        layer_ids: Sequence[int] | None = None,
    ) -> Iterator[LayerRetrievalResult]:
        """Retrieve matching KV cache one layer at a time.

        The next layer is scheduled before the current layer result is yielded, so an
        adapter can overlap device injection with storage retrieval.
        """

        ordered_layers = list(layer_ids) if layer_ids is not None else self.layer_ids(query)
        pending: Future[LayerRetrievalResult] | None = None
        for layer_id in ordered_layers:
            next_future = self._executor.submit(self._retrieve_layer, query, layer_id, target_tier)
            if pending is not None:
                yield pending.result()
            pending = next_future
        if pending is not None:
            yield pending.result()

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

    def _put_layer(
        self,
        layer_id: int,
        blocks: list[CacheBlock],
        target_tier: DeviceTier | None,
    ) -> LayerTransferResult:
        start = time.perf_counter()
        block_ids = []
        bytes_moved = 0
        failures = []
        for block in blocks:
            if block.metadata.layer_id != layer_id:
                failures.append(block.metadata.block_id)
                continue
            if target_tier is not None:
                block.metadata.device_tier = target_tier
            bytes_moved += block.metadata.bytes_size or block.payload.nbytes
            try:
                self.put(block)
                block_ids.append(block.metadata.block_id)
            except Exception:
                failures.append(block.metadata.block_id)
        resolved_tier = target_tier or (blocks[0].metadata.device_tier if blocks else DeviceTier.CPU)
        return LayerTransferResult(
            layer_id=layer_id,
            block_ids=block_ids,
            target_tier=resolved_tier,
            success=not failures,
            bytes_moved=bytes_moved,
            elapsed_ms=(time.perf_counter() - start) * 1000.0,
            failures=failures,
        )

    def _retrieve_layer(
        self,
        query: CacheQuery,
        layer_id: int,
        target_tier: DeviceTier,
    ) -> LayerRetrievalResult:
        start = time.perf_counter()
        blocks = self.get_layer(query, layer_id, promote_to=target_tier)
        return LayerRetrievalResult(
            layer_id=layer_id,
            blocks=blocks,
            target_tier=target_tier,
            hit=bool(blocks),
            bytes_loaded=sum(block.metadata.bytes_size for block in blocks),
            elapsed_ms=(time.perf_counter() - start) * 1000.0,
        )

    def _normalize_layer_input(
        self,
        blocks_by_layer: Mapping[int, Sequence[CacheBlock]] | Sequence[Sequence[CacheBlock]],
    ) -> list[tuple[int, Sequence[CacheBlock]]]:
        if isinstance(blocks_by_layer, Mapping):
            return sorted(blocks_by_layer.items(), key=lambda item: item[0])
        items: list[tuple[int, Sequence[CacheBlock]]] = []
        for layer_id, blocks in enumerate(blocks_by_layer):
            items.append((layer_id, blocks))
        return items

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
