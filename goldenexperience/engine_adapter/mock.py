"""Mock adapter used by tests and synthetic benchmarks."""

from __future__ import annotations

from typing import Any

from goldenexperience.cache_core.block import CacheBlock, KVPayload
from goldenexperience.cache_core.enums import DeviceTier
from goldenexperience.engine_adapter.base import ModelAdapter
from goldenexperience.engine_adapter.signature import ArchitectureSignature


class MockModelAdapter(ModelAdapter):
    """Deterministic adapter for artifact tests without heavyweight model deps."""

    def __init__(self, signature: ArchitectureSignature) -> None:
        self._signature = signature

    @property
    def architecture_signature(self) -> ArchitectureSignature:
        return self._signature

    def extract_kv(self, engine_state: Any, **kwargs: Any) -> list[CacheBlock]:
        token_ids = kwargs.get("token_ids", [0])
        session_id = kwargs.get("session_id")
        prefix_hash = kwargs.get("prefix_hash")
        blocks = []
        for layer_id in range(self._signature.num_layers):
            payload = KVPayload(
                key=self._make_tensor(layer_id, token_ids, scale=1.0),
                value=self._make_tensor(layer_id, token_ids, scale=0.5),
            )
            block = CacheBlock.from_payload(
                payload=payload,
                model_id=self._signature.model_id,
                layer_id=layer_id,
                head_id=None,
                token_range=(0, len(token_ids)),
                dtype=self._signature.dtype,
                device_tier=DeviceTier.HBM,
                prefix_hash=prefix_hash,
                session_id=session_id,
            )
            blocks.append(block)
        return blocks

    def inject_kv(self, blocks: list[CacheBlock], engine_state: Any | None = None, **kwargs: Any) -> Any:
        return {
            "model_id": self._signature.model_id,
            "blocks": blocks,
            "engine_state": engine_state,
        }

    def _make_tensor(self, layer_id: int, token_ids: list[int], scale: float) -> list[list[list[float]]]:
        result = []
        for _head in range(self._signature.num_key_value_heads):
            head_rows = []
            for token in token_ids:
                head_rows.append([
                    scale * float(layer_id + token + dim + 1)
                    for dim in range(self._signature.head_dim)
                ])
            result.append(head_rows)
        return result
