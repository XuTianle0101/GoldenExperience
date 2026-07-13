"""vLLM thin adapter placeholders.

vLLM internals evolve quickly, so the v1 adapter exposes an explicit boundary for
packing and unpacking cache blocks without making GoldenExperience depend on vLLM.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from goldenexperience.cache_core.block import CacheBlock
from goldenexperience.engine_adapter.base import ModelAdapter
from goldenexperience.engine_adapter.signature import ArchitectureSignature


class VLLMAdapter(ModelAdapter):
    """Adapter shell for vLLM integrations supplied by benchmark scripts."""

    def __init__(
        self,
        signature: ArchitectureSignature,
        extractor: Callable[[Any], list[CacheBlock]] | None = None,
        injector: Callable[[list[CacheBlock], Any | None], Any] | None = None,
    ) -> None:
        self._signature = signature
        self._extractor = extractor
        self._injector = injector

    @property
    def architecture_signature(self) -> ArchitectureSignature:
        return self._signature

    def extract_kv(self, engine_state: Any, **kwargs: Any) -> list[CacheBlock]:
        if self._extractor is None:
            raise NotImplementedError(
                "Provide a vLLM extractor callable for the installed vLLM version."
            )
        return self._extractor(engine_state)

    def inject_kv(
        self, blocks: list[CacheBlock], engine_state: Any | None = None, **kwargs: Any
    ) -> Any:
        if self._injector is None:
            raise NotImplementedError(
                "Provide a vLLM injector callable for the installed vLLM version."
            )
        return self._injector(blocks, engine_state)
