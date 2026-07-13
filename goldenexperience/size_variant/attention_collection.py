"""Bounded target-query and attention-output collection for transport training."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from goldenexperience.size_variant.head_aware_transport import (
    _apply_rope_heads,
    sample_attention_positions,
)


@dataclass(frozen=True)
class TargetAttentionTrace:
    queries: Any
    attention_outputs: Any
    query_positions: Any
    key_positions: Any
    causal_mask: Any


class TargetAttentionCollector:
    """Collect at most 32 queries and 256 key positions per prompt."""

    def __init__(
        self,
        model: Any,
        *,
        token_count: int,
        rope_theta: float,
        max_queries: int = 32,
        max_keys: int = 256,
        offload_to_cpu: bool = True,
    ) -> None:
        self.model = model
        self.layers = tuple(model.model.layers)
        self.rope_theta = float(rope_theta)
        self.query_positions, self.key_positions = sample_attention_positions(
            token_count,
            max_queries=max_queries,
            max_keys=max_keys,
        )
        self.offload_to_cpu = offload_to_cpu
        self._queries: dict[int, Any] = {}
        self._outputs: dict[int, Any] = {}
        self._handles: list[Any] = []

    def __enter__(self) -> TargetAttentionCollector:
        for layer_id, layer in enumerate(self.layers):
            attention = layer.self_attn
            query_module = getattr(attention, "q_norm", None) or attention.q_proj
            self._handles.append(
                query_module.register_forward_hook(self._query_hook(layer_id, attention))
            )
            self._handles.append(
                attention.o_proj.register_forward_pre_hook(self._output_hook(layer_id, attention))
            )
        return self

    def __exit__(self, *_: object) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def trace(self) -> TargetAttentionTrace:
        import torch

        expected = set(range(len(self.layers)))
        if set(self._queries) != expected or set(self._outputs) != expected:
            raise ValueError("target attention trace is incomplete")
        queries = torch.stack([self._queries[index] for index in range(len(self.layers))])
        outputs = torch.stack([self._outputs[index] for index in range(len(self.layers))])
        query_positions = self.query_positions.to(queries.device)
        key_positions = self.key_positions.to(queries.device)
        mask = causal_sample_mask(query_positions, key_positions)
        return TargetAttentionTrace(
            queries=queries,
            attention_outputs=outputs,
            query_positions=query_positions,
            key_positions=key_positions,
            causal_mask=mask,
        )

    def _query_hook(self, layer_id: int, attention: Any):
        def hook(_module: Any, _inputs: Any, output: Any) -> None:
            import torch

            value = output[0] if isinstance(output, tuple) else output
            if not isinstance(value, torch.Tensor) or value.shape[0] != 1:
                raise ValueError("query collector requires a batch-one tensor")
            heads = int(attention.config.num_attention_heads)
            head_dim = int(attention.head_dim)
            if value.ndim == 3:
                value = value.reshape(1, value.shape[1], heads, head_dim)
            if value.ndim != 4:
                raise ValueError("query collector received an unsupported tensor layout")
            if int(value.shape[2]) == heads:
                value = value[0].permute(1, 0, 2)
            elif int(value.shape[1]) == heads:
                value = value[0]
            else:
                raise ValueError("query collector could not identify the head axis")
            positions = self.query_positions.to(value.device)
            sampled = value.index_select(1, positions)
            sampled = _apply_rope_heads(
                sampled,
                positions,
                theta=self.rope_theta,
                inverse=False,
            )
            self._queries[layer_id] = self._finish(sampled)

        return hook

    def _output_hook(self, layer_id: int, attention: Any):
        def hook(_module: Any, inputs: Any) -> None:
            import torch

            if not inputs or not isinstance(inputs[0], torch.Tensor):
                raise ValueError("attention output collector received no tensor")
            value = inputs[0]
            if value.shape[0] != 1:
                raise ValueError("attention output collector requires batch size one")
            heads = int(attention.config.num_attention_heads)
            head_dim = int(attention.head_dim)
            if value.ndim == 3:
                value = value.reshape(1, value.shape[1], heads, head_dim)
            if value.ndim != 4:
                raise ValueError("attention output collector received an unsupported layout")
            if int(value.shape[2]) == heads:
                value = value[0].permute(1, 0, 2)
            elif int(value.shape[1]) == heads:
                value = value[0]
            else:
                raise ValueError("attention output collector could not identify the head axis")
            positions = self.query_positions.to(value.device)
            self._outputs[layer_id] = self._finish(value.index_select(1, positions))

        return hook

    def _finish(self, value: Any) -> Any:
        result = value.detach()
        if self.offload_to_cpu:
            result = result.to("cpu")
        return result.contiguous()


def causal_sample_mask(query_positions: Any, key_positions: Any) -> Any:
    """Return `[1, 1, query, key]` causal visibility for sampled positions."""

    import torch

    queries = torch.as_tensor(query_positions, dtype=torch.long)
    keys = torch.as_tensor(key_positions, dtype=torch.long, device=queries.device)
    if queries.ndim != 1 or keys.ndim != 1:
        raise ValueError("sampled attention positions must be rank one")
    return (keys.unsqueeze(0) <= queries.unsqueeze(1)).unsqueeze(0).unsqueeze(0)
