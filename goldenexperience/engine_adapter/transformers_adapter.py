"""Hugging Face Transformers thin adapter."""

from __future__ import annotations

import hashlib
import json
from typing import Any

from goldenexperience.cache_core.block import CacheBlock, KVPayload
from goldenexperience.cache_core.enums import DeviceTier
from goldenexperience.engine_adapter.base import ModelAdapter
from goldenexperience.engine_adapter.signature import ArchitectureSignature


class TransformersAdapter(ModelAdapter):
    """Adapter for Hugging Face style `past_key_values`.

    This adapter intentionally does not own generation. It only converts between HF
    cache tuples and GoldenExperience cache blocks.
    """

    def __init__(
        self,
        model: Any,
        tokenizer: Any | None = None,
        model_id: str | None = None,
        family: str | None = None,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self._signature = self._signature_from_model(model, tokenizer, model_id, family)

    @property
    def architecture_signature(self) -> ArchitectureSignature:
        return self._signature

    def extract_kv(self, engine_state: Any, **kwargs: Any) -> list[CacheBlock]:
        past_key_values = kwargs.get("past_key_values", engine_state)
        if past_key_values is None:
            raise ValueError("TransformersAdapter.extract_kv requires past_key_values.")
        prefix_hash = kwargs.get("prefix_hash")
        session_id = kwargs.get("session_id")
        token_start = int(kwargs.get("token_start", 0))
        blocks = []
        for layer_id, layer_kv in enumerate(past_key_values):
            key, value = layer_kv[:2]
            token_end = token_start + int(key.shape[-2])
            block = CacheBlock.from_payload(
                payload=KVPayload(key=key, value=value),
                model_id=self._signature.model_id,
                layer_id=layer_id,
                head_id=None,
                token_range=(token_start, token_end),
                dtype=str(getattr(key, "dtype", self._signature.dtype)).replace("torch.", ""),
                device_tier=DeviceTier.HBM,
                prefix_hash=prefix_hash,
                session_id=session_id,
            )
            blocks.append(block)
        return blocks

    def inject_kv(self, blocks: list[CacheBlock], engine_state: Any | None = None, **kwargs: Any) -> Any:
        ordered = sorted(blocks, key=lambda block: block.metadata.layer_id)
        return tuple((block.payload.key, block.payload.value) for block in ordered)

    def _signature_from_model(
        self,
        model: Any,
        tokenizer: Any | None,
        model_id: str | None,
        family: str | None,
    ) -> ArchitectureSignature:
        config = getattr(model, "config", model)
        resolved_model_id = model_id or getattr(config, "_name_or_path", None) or config.__class__.__name__
        architecture = (getattr(config, "architectures", None) or [config.__class__.__name__])[0]
        hidden_size = int(getattr(config, "hidden_size"))
        num_attention_heads = int(getattr(config, "num_attention_heads"))
        num_key_value_heads = int(getattr(config, "num_key_value_heads", num_attention_heads))
        head_dim = int(getattr(config, "head_dim", hidden_size // num_attention_heads))
        tokenizer_id = getattr(tokenizer, "name_or_path", None) if tokenizer is not None else None
        config_payload = config.to_dict() if hasattr(config, "to_dict") else {}
        config_hash = hashlib.sha256(
            json.dumps(config_payload, sort_keys=True, default=str).encode("utf-8")
        ).hexdigest()
        rope_scaling = getattr(config, "rope_scaling", None)
        return ArchitectureSignature(
            model_id=str(resolved_model_id),
            family=family or str(resolved_model_id).split("-")[0].lower(),
            architecture=str(architecture),
            num_layers=int(getattr(config, "num_hidden_layers")),
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            head_dim=head_dim,
            rope_theta=getattr(config, "rope_theta", None),
            rope_scaling=(
                json.dumps(rope_scaling, sort_keys=True, default=str)
                if rope_scaling is not None
                else None
            ),
            sliding_window=getattr(config, "sliding_window", None),
            tokenizer_id=tokenizer_id,
            dtype=str(getattr(config, "torch_dtype", "float16")).replace("torch.", ""),
            vocab_size=getattr(config, "vocab_size", None),
            revision=getattr(config, "_commit_hash", None),
            model_config_hash=config_hash,
        )
