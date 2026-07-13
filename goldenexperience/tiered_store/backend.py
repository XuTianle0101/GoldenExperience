"""Storage backends for cache tiers."""

from __future__ import annotations

import pickle
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from goldenexperience.cache_core.enums import DeviceTier
from goldenexperience.utils.tensors import tensor_nbytes


class TierBackend(ABC):
    """Abstract payload backend for one cache tier."""

    def __init__(self, tier: DeviceTier) -> None:
        self.tier = tier

    @abstractmethod
    def put(self, block_id: str, payload: Any) -> None:
        raise NotImplementedError

    @abstractmethod
    def get(self, block_id: str) -> Any | None:
        raise NotImplementedError

    @abstractmethod
    def remove(self, block_id: str) -> Any | None:
        raise NotImplementedError

    @abstractmethod
    def contains(self, block_id: str) -> bool:
        raise NotImplementedError

    @abstractmethod
    def bytes_used(self) -> int:
        raise NotImplementedError


class MemoryTierBackend(TierBackend):
    """In-process backend for HBM and CPU tiers.

    In prototype mode HBM is represented as an in-memory reference. If the payload is a
    CUDA tensor the store can move it to CUDA before insertion.
    """

    def __init__(self, tier: DeviceTier) -> None:
        super().__init__(tier)
        self._payloads: dict[str, Any] = {}
        self._sizes: dict[str, int] = {}

    def put(self, block_id: str, payload: Any) -> None:
        self._payloads[block_id] = payload
        self._sizes[block_id] = tensor_nbytes(payload)

    def get(self, block_id: str) -> Any | None:
        return self._payloads.get(block_id)

    def remove(self, block_id: str) -> Any | None:
        self._sizes.pop(block_id, None)
        return self._payloads.pop(block_id, None)

    def contains(self, block_id: str) -> bool:
        return block_id in self._payloads

    def bytes_used(self) -> int:
        return sum(self._sizes.values())


class NvmeTierBackend(TierBackend):
    """Pickle-backed NVMe tier for artifact-friendly prototyping."""

    def __init__(self, root: str | Path) -> None:
        super().__init__(DeviceTier.NVME)
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._sizes: dict[str, int] = {}

    def put(self, block_id: str, payload: Any) -> None:
        path = self._path(block_id)
        with path.open("wb") as handle:
            pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
        self._sizes[block_id] = path.stat().st_size

    def get(self, block_id: str) -> Any | None:
        path = self._path(block_id)
        if not path.exists():
            return None
        with path.open("rb") as handle:
            return pickle.load(handle)

    def remove(self, block_id: str) -> Any | None:
        payload = self.get(block_id)
        path = self._path(block_id)
        if path.exists():
            path.unlink()
        self._sizes.pop(block_id, None)
        return payload

    def contains(self, block_id: str) -> bool:
        return self._path(block_id).exists()

    def bytes_used(self) -> int:
        return sum(self._sizes.values())

    def _path(self, block_id: str) -> Path:
        return self.root / f"{block_id}.pkl"
