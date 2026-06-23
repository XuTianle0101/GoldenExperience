"""In-memory metadata index for KV cache blocks."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable

from goldenexperience.cache_core.block import CacheBlockMetadata, CacheQuery


class CacheIndex:
    """Small metadata index optimized for prototype and artifact reproducibility."""

    def __init__(self) -> None:
        self._by_id: dict[str, CacheBlockMetadata] = {}
        self._by_model: dict[str, set[str]] = defaultdict(set)
        self._by_layer: dict[tuple[str, int], set[str]] = defaultdict(set)
        self._by_prefix: dict[str, set[str]] = defaultdict(set)
        self._by_session: dict[str, set[str]] = defaultdict(set)

    def __len__(self) -> int:
        return len(self._by_id)

    def add(self, metadata: CacheBlockMetadata) -> None:
        self.remove(metadata.block_id)
        self._by_id[metadata.block_id] = metadata
        self._by_model[metadata.model_id].add(metadata.block_id)
        self._by_layer[(metadata.model_id, metadata.layer_id)].add(metadata.block_id)
        if metadata.prefix_hash is not None:
            self._by_prefix[metadata.prefix_hash].add(metadata.block_id)
        if metadata.session_id is not None:
            self._by_session[metadata.session_id].add(metadata.block_id)

    def remove(self, block_id: str) -> CacheBlockMetadata | None:
        metadata = self._by_id.pop(block_id, None)
        if metadata is None:
            return None
        self._by_model[metadata.model_id].discard(block_id)
        self._by_layer[(metadata.model_id, metadata.layer_id)].discard(block_id)
        if metadata.prefix_hash is not None:
            self._by_prefix[metadata.prefix_hash].discard(block_id)
        if metadata.session_id is not None:
            self._by_session[metadata.session_id].discard(block_id)
        return metadata

    def get(self, block_id: str) -> CacheBlockMetadata | None:
        return self._by_id.get(block_id)

    def all(self) -> list[CacheBlockMetadata]:
        return list(self._by_id.values())

    def find(self, query: CacheQuery) -> list[CacheBlockMetadata]:
        candidates = self._candidate_ids(query)
        matches = [self._by_id[block_id] for block_id in candidates if query.matches(self._by_id[block_id])]
        matches.sort(key=lambda meta: (meta.quality_score, meta.last_accessed), reverse=True)
        return matches

    def layer_ids(self, query: CacheQuery) -> list[int]:
        return sorted({metadata.layer_id for metadata in self.find(query)})

    def group_by_layer(self, query: CacheQuery) -> dict[int, list[CacheBlockMetadata]]:
        grouped: dict[int, list[CacheBlockMetadata]] = defaultdict(list)
        for metadata in self.find(query):
            grouped[metadata.layer_id].append(metadata)
        return dict(sorted(grouped.items(), key=lambda item: item[0]))

    def _candidate_ids(self, query: CacheQuery) -> Iterable[str]:
        if query.model_id is not None and query.layer_id is not None:
            return set(self._by_layer.get((query.model_id, query.layer_id), set()))
        if query.prefix_hash is not None:
            return set(self._by_prefix.get(query.prefix_hash, set()))
        if query.session_id is not None:
            return set(self._by_session.get(query.session_id, set()))
        if query.model_id is not None:
            return set(self._by_model.get(query.model_id, set()))
        return set(self._by_id)
