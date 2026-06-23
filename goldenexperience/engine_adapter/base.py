"""Base interface for engine-neutral model adapters."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from goldenexperience.cache_core.block import CacheBlock
from goldenexperience.engine_adapter.signature import ArchitectureSignature


class ModelAdapter(ABC):
    """Thin boundary between inference engines and GoldenExperience."""

    @property
    @abstractmethod
    def architecture_signature(self) -> ArchitectureSignature:
        raise NotImplementedError

    @abstractmethod
    def extract_kv(self, engine_state: Any, **kwargs: Any) -> list[CacheBlock]:
        """Export engine-owned KV state into engine-neutral cache blocks."""

    @abstractmethod
    def inject_kv(self, blocks: list[CacheBlock], engine_state: Any | None = None, **kwargs: Any) -> Any:
        """Inject cache blocks back into an engine-specific format."""

    def supports_model(self, signature: ArchitectureSignature) -> bool:
        return self.architecture_signature.compatibility_with(signature).value != "incompatible"

