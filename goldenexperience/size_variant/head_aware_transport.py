"""Head-aware, attention-preserving cached-KV transport for manifest v5."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from goldenexperience.size_variant.cached_kv_manifest import (
    CachedKVModelSpec,
    sha256_file,
)
from goldenexperience.size_variant.selective_manifest import (
    ArtifactState,
    SelectiveKVBridgeManifest,
    TransportLossContract,
    TransportSpec,
)


class HeadAwareTransportError(RuntimeError):
    """Raised when a v5 transport or input KV object fails closed."""


REQUIRED_TRANSPORT_TENSORS = frozenset(
    {
        "source_layer_ids",
        "layer_mix",
        "head_mix",
        "key_normalizer_mean",
        "key_normalizer_scale",
        "key_down",
        "key_up",
        "key_gate",
        "key_scale",
        "key_bias",
        "value_normalizer_mean",
        "value_normalizer_scale",
        "value_down",
        "value_up",
        "value_gate",
        "value_scale",
        "value_bias",
    }
)
TRANSPORT_TRAINING_SEEDS = (17, 29, 43)


class HeadAwareKVTransport:
    """Translate head-structured KV without loading source or target models."""

    def __init__(
        self,
        source: CachedKVModelSpec,
        target: CachedKVModelSpec,
        spec: TransportSpec,
        tensors: Mapping[str, Any],
        *,
        device: str = "cpu",
        compute_dtype: Any | None = None,
    ) -> None:
        import torch

        errors = _compatibility_errors(source, target) + spec.validate(source)
        if errors:
            raise HeadAwareTransportError("; ".join(errors))
        self.source = source
        self.target = target
        self.spec = spec
        self.device = torch.device(device)
        self.compute_dtype = compute_dtype or (
            torch.bfloat16 if self.device.type == "cuda" else torch.float32
        )
        self.tensors = {
            name: tensor.detach().to(device=self.device) for name, tensor in tensors.items()
        }
        self._validate_tensors()

    @classmethod
    def from_artifact(
        cls,
        manifest_path: str | Path,
        *,
        device: str = "cpu",
        compute_dtype: Any | None = None,
        offline: bool = False,
    ) -> HeadAwareKVTransport:
        """Load v5 weights, requiring final approval except in explicit offline mode."""

        path = Path(manifest_path).resolve()
        manifest = SelectiveKVBridgeManifest.load(path)
        return cls.from_manifest(
            manifest,
            path,
            device=device,
            compute_dtype=compute_dtype,
            offline=offline,
        )

    @classmethod
    def from_manifest(
        cls,
        manifest: SelectiveKVBridgeManifest,
        manifest_path: str | Path,
        *,
        device: str = "cpu",
        compute_dtype: Any | None = None,
        offline: bool = False,
    ) -> HeadAwareKVTransport:
        """Load transport tensors bound to one already-validated manifest snapshot."""

        from safetensors import safe_open

        path = Path(manifest_path).resolve()
        errors = manifest.artifact_errors() if offline else manifest.validate()
        if not offline and manifest.state is not ArtifactState.APPROVED:
            errors.append("v5 artifact is not approved for runtime reuse")
        if errors:
            raise HeadAwareTransportError("; ".join(errors))
        weights_path = manifest.resolve_transport_weights_path(path)
        if weights_path.suffix != ".safetensors" or not weights_path.is_file():
            raise HeadAwareTransportError("transport weights are missing")
        if sha256_file(weights_path) != manifest.transport.weights_sha256:
            raise HeadAwareTransportError("transport weight checksum mismatch")
        tensors: dict[str, Any] = {}
        with safe_open(str(weights_path), framework="pt", device="cpu") as handle:
            expected_metadata = transport_safetensors_metadata(manifest)
            if (handle.metadata() or {}) != expected_metadata:
                raise HeadAwareTransportError("transport safetensors metadata is invalid")
            for name in handle.keys():  # noqa: SIM118
                tensors[name] = handle.get_tensor(name)
        if sha256_file(weights_path) != manifest.transport.weights_sha256:
            raise HeadAwareTransportError("transport weights changed while loading")
        return cls(
            manifest.source,
            manifest.target,
            manifest.transport,
            tensors,
            device=device,
            compute_dtype=compute_dtype,
        )

    @property
    def source_object_shape(self) -> tuple[int, int, int, int]:
        return (
            2,
            self.source.num_layers,
            self.source.num_key_value_heads,
            self.source.head_dim,
        )

    @property
    def target_object_shape(self) -> tuple[int, int, int, int]:
        return (
            2,
            self.target.num_layers,
            self.target.num_key_value_heads,
            self.target.head_dim,
        )

    def transform(
        self,
        source_kv: Any,
        *,
        position_start: int = 0,
        position_ids: Any | None = None,
    ) -> Any:
        """Translate `[2, source_layer, source_head, token, head_dim]` KV."""

        import torch
        import torch.nn.functional as functional

        self._validate_source(source_kv)
        token_count = int(source_kv.shape[3])
        positions = self._positions(token_count, position_start, position_ids)
        source = source_kv.to(device=self.device, dtype=self.compute_dtype)
        source_key = _apply_rope_heads(
            source[0],
            positions,
            theta=self.source.rope_theta,
            inverse=True,
        )
        base_key = self._mix_source(source_key)
        base_value = self._mix_source(source[1])

        key_input = (base_key - self._tensor("key_normalizer_mean").unsqueeze(2)) / self._tensor(
            "key_normalizer_scale"
        ).unsqueeze(2)
        value_input = (
            base_value - self._tensor("value_normalizer_mean").unsqueeze(2)
        ) / self._tensor("value_normalizer_scale").unsqueeze(2)
        key_latent = torch.einsum("lhtd,lhdr->lhtr", key_input, self._tensor("key_down"))
        value_latent = torch.einsum("lhtd,lhdr->lhtr", value_input, self._tensor("value_down"))
        key_residual = torch.einsum(
            "lhtr,lhrd->lhtd", functional.silu(key_latent), self._tensor("key_up")
        )
        value_residual = torch.einsum(
            "lhtr,lhrd->lhtd", functional.silu(value_latent), self._tensor("value_up")
        )
        target_key_unrotated = (
            base_key * self._tensor("key_scale").unsqueeze(2)
            + torch.sigmoid(self._tensor("key_gate")).unsqueeze(2) * key_residual
            + self._tensor("key_bias").unsqueeze(2)
        )
        target_value = (
            base_value * self._tensor("value_scale").unsqueeze(2)
            + torch.sigmoid(self._tensor("value_gate")).unsqueeze(2) * value_residual
            + self._tensor("value_bias").unsqueeze(2)
        )
        target_key = _apply_rope_heads(
            target_key_unrotated,
            positions,
            theta=self.target.rope_theta,
            inverse=False,
        )
        target = torch.stack((target_key, target_value), dim=0).to(
            dtype=_torch_dtype(self.target.dtype)
        )
        if not bool(torch.isfinite(target).all()):
            raise HeadAwareTransportError("head-aware transport produced non-finite KV")
        return target.contiguous()

    def ood_distance(self, source_kv: Any, *, position_start: int = 0) -> float:
        """Return the maximum per-head RMS z-distance used by source sidecars."""

        import torch

        self._validate_source(source_kv)
        positions = self._positions(int(source_kv.shape[3]), position_start, None)
        source = source_kv.to(device=self.device, dtype=self.compute_dtype)
        key = _apply_rope_heads(source[0], positions, theta=self.source.rope_theta, inverse=True)
        mixed = (self._mix_source(key), self._mix_source(source[1]))
        distances: list[Any] = []
        for prefix, value in zip(("key", "value"), mixed, strict=True):
            normalized = (
                value - self._tensor(f"{prefix}_normalizer_mean").unsqueeze(2)
            ) / self._tensor(f"{prefix}_normalizer_scale").unsqueeze(2)
            distances.append(normalized.square().mean(dim=(-1, -2)).sqrt())
        return float(torch.stack(distances).max().item())

    def parameter_count(self) -> int:
        return sum(
            int(tensor.numel())
            for name, tensor in self.tensors.items()
            if name != "source_layer_ids"
        )

    def _mix_source(self, source: Any) -> Any:
        import torch

        layer_ids = self.tensors["source_layer_ids"].long()
        selected = source[layer_ids]
        # selected: target_layer, target_head, window, source_head, token, head_dim
        head_mixed = torch.einsum(
            "lhwstd,lhws->lhwtd",
            selected,
            self._tensor("head_mix"),
        )
        return torch.einsum("lhwtd,lhw->lhtd", head_mixed, self._tensor("layer_mix"))

    def _tensor(self, name: str) -> Any:
        return self.tensors[name].to(dtype=self.compute_dtype)

    def _positions(self, token_count: int, position_start: int, position_ids: Any | None) -> Any:
        import torch

        if token_count <= 0:
            raise HeadAwareTransportError("source KV contains no tokens")
        if position_ids is None:
            if position_start < 0:
                raise HeadAwareTransportError("position_start must be non-negative")
            positions = torch.arange(
                position_start,
                position_start + token_count,
                device=self.device,
                dtype=torch.long,
            )
        else:
            positions = torch.as_tensor(position_ids, device=self.device, dtype=torch.long)
            if positions.ndim != 1 or positions.numel() != token_count:
                raise HeadAwareTransportError("position_ids must have one entry per token")
            if bool((positions < 0).any()):
                raise HeadAwareTransportError("position_ids must be non-negative")
        max_position = min(
            self.source.max_position_embeddings,
            self.target.max_position_embeddings,
        )
        if int(positions.max().item()) >= max_position:
            raise HeadAwareTransportError("position_ids exceed the source/target RoPE contract")
        return positions

    def _validate_source(self, source_kv: Any) -> None:
        import torch

        expected = (
            2,
            self.source.num_layers,
            self.source.num_key_value_heads,
            self.source.head_dim,
        )
        if not isinstance(source_kv, torch.Tensor) or source_kv.ndim != 5:
            raise HeadAwareTransportError(
                "source KV must have [2, layer, head, token, head_dim] layout"
            )
        observed = (
            int(source_kv.shape[0]),
            int(source_kv.shape[1]),
            int(source_kv.shape[2]),
            int(source_kv.shape[4]),
        )
        if observed != expected:
            raise HeadAwareTransportError(f"source KV static dimensions must be {expected}")
        if source_kv.dtype != _torch_dtype(self.source.dtype):
            raise HeadAwareTransportError(f"source KV dtype must be {self.source.dtype}")
        if not bool(torch.isfinite(source_kv).all()):
            raise HeadAwareTransportError("source KV contains non-finite values")

    def _validate_tensors(self) -> None:
        import torch

        if set(self.tensors) != REQUIRED_TRANSPORT_TENSORS:
            missing = sorted(REQUIRED_TRANSPORT_TENSORS - set(self.tensors))
            unknown = sorted(set(self.tensors) - REQUIRED_TRANSPORT_TENSORS)
            raise HeadAwareTransportError(
                f"transport tensor set mismatch; missing={missing}, unknown={unknown}"
            )
        target_layers = self.target.num_layers
        target_heads = self.target.num_key_value_heads
        source_heads = self.source.num_key_value_heads
        window = self.spec.source_window
        dim = self.source.head_dim
        rank = self.spec.rank
        shapes = {
            "source_layer_ids": (target_layers, target_heads, window),
            "layer_mix": (target_layers, target_heads, window),
            "head_mix": (target_layers, target_heads, window, source_heads),
        }
        for prefix in ("key", "value"):
            shapes.update(
                {
                    f"{prefix}_normalizer_mean": (target_layers, target_heads, dim),
                    f"{prefix}_normalizer_scale": (target_layers, target_heads, dim),
                    f"{prefix}_down": (target_layers, target_heads, dim, rank),
                    f"{prefix}_up": (target_layers, target_heads, rank, dim),
                    f"{prefix}_gate": (target_layers, target_heads, dim),
                    f"{prefix}_scale": (target_layers, target_heads, dim),
                    f"{prefix}_bias": (target_layers, target_heads, dim),
                }
            )
        for name, shape in shapes.items():
            tensor = self.tensors[name]
            if tuple(tensor.shape) != shape:
                raise HeadAwareTransportError(
                    f"transport {name} shape {tuple(tensor.shape)} does not match {shape}"
                )
            if name == "source_layer_ids":
                if tensor.dtype not in {torch.int32, torch.int64}:
                    raise HeadAwareTransportError("source_layer_ids must be integer")
            elif not tensor.is_floating_point() or not bool(torch.isfinite(tensor).all()):
                raise HeadAwareTransportError(f"transport {name} must be finite floating point")
        layer_ids = self.tensors["source_layer_ids"]
        if bool((layer_ids < 0).any()) or bool((layer_ids >= self.source.num_layers).any()):
            raise HeadAwareTransportError("source_layer_ids contains an out-of-range layer")
        for name in ("layer_mix", "head_mix"):
            weights = self.tensors[name].float()
            if bool((weights < 0).any()):
                raise HeadAwareTransportError(f"{name} must be non-negative")
            if not torch.allclose(
                weights.sum(dim=-1),
                torch.ones_like(weights.sum(dim=-1)),
                atol=1e-5,
                rtol=0,
            ):
                raise HeadAwareTransportError(f"{name} must sum to one")
        for name in ("key_normalizer_scale", "value_normalizer_scale"):
            if bool((self.tensors[name] <= 0).any()):
                raise HeadAwareTransportError(f"{name} must be positive")


def initialize_head_aware_state(
    source: CachedKVModelSpec,
    target: CachedKVModelSpec,
    spec: TransportSpec,
) -> dict[str, Any]:
    """Create a normalized depth/head interpolation baseline for training."""

    import torch

    errors = _compatibility_errors(source, target) + spec.validate(source)
    if errors:
        raise HeadAwareTransportError("; ".join(errors))
    target_layers = target.num_layers
    target_heads = target.num_key_value_heads
    source_heads = source.num_key_value_heads
    window = spec.source_window
    dim = source.head_dim
    rank = spec.rank
    layer_ids = torch.empty(target_layers, target_heads, window, dtype=torch.int64)
    layer_mix = torch.zeros(target_layers, target_heads, window)
    for target_layer in range(target_layers):
        position = (
            0.0
            if target_layers == 1
            else target_layer * (source.num_layers - 1) / (target_layers - 1)
        )
        center = round(position)
        start = max(0, min(source.num_layers - window, center - window // 2))
        ids = torch.arange(start, start + window)
        layer_ids[target_layer] = ids.view(1, window).expand(target_heads, -1)
        distances = (ids.float() - position).abs()
        weights = torch.softmax(-distances, dim=0)
        layer_mix[target_layer] = weights.view(1, window).expand(target_heads, -1)
    head_mix = torch.zeros(target_layers, target_heads, window, source_heads)
    for target_head in range(target_heads):
        position = (
            0.0 if target_heads == 1 else target_head * (source_heads - 1) / (target_heads - 1)
        )
        low = math.floor(position)
        high = min(source_heads - 1, low + 1)
        high_weight = position - low
        head_mix[:, target_head, :, low] = 1.0 - high_weight
        head_mix[:, target_head, :, high] += high_weight
    state: dict[str, Any] = {
        "source_layer_ids": layer_ids,
        "layer_mix": layer_mix,
        "head_mix": head_mix,
    }
    for prefix in ("key", "value"):
        state.update(
            {
                f"{prefix}_normalizer_mean": torch.zeros(target_layers, target_heads, dim),
                f"{prefix}_normalizer_scale": torch.ones(target_layers, target_heads, dim),
                f"{prefix}_down": torch.zeros(target_layers, target_heads, dim, rank),
                f"{prefix}_up": torch.zeros(target_layers, target_heads, rank, dim),
                f"{prefix}_gate": torch.zeros(target_layers, target_heads, dim),
                f"{prefix}_scale": torch.ones(target_layers, target_heads, dim),
                f"{prefix}_bias": torch.zeros(target_layers, target_heads, dim),
            }
        )
    return state


def build_trainable_head_aware_transport(
    source: CachedKVModelSpec,
    target: CachedKVModelSpec,
    spec: TransportSpec,
    *,
    device: str = "cpu",
    seed: int = 17,
) -> Any:
    """Build the differentiable form of the exact runtime transport."""

    import torch

    errors = _compatibility_errors(source, target) + spec.validate(source)
    if errors:
        raise HeadAwareTransportError("; ".join(errors))
    initial = initialize_head_aware_state(source, target, spec)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)

    class _TrainableTransport(torch.nn.Module):
        source_layer_ids: Any

        def __init__(self) -> None:
            super().__init__()
            self.source = source
            self.target = target
            self.spec = spec
            self.register_buffer("source_layer_ids", initial["source_layer_ids"])
            self.layer_mix_logits = torch.nn.Parameter(initial["layer_mix"].clamp_min(1e-6).log())
            self.head_mix_logits = torch.nn.Parameter(initial["head_mix"].clamp_min(1e-6).log())
            for prefix in ("key", "value"):
                self.register_buffer(
                    f"{prefix}_normalizer_mean",
                    initial[f"{prefix}_normalizer_mean"],
                )
                self.register_buffer(
                    f"{prefix}_normalizer_scale",
                    initial[f"{prefix}_normalizer_scale"],
                )
                down = initial[f"{prefix}_down"]
                torch.nn.init.normal_(down, mean=0.0, std=0.02, generator=generator)
                setattr(self, f"{prefix}_down", torch.nn.Parameter(down))
                setattr(self, f"{prefix}_up", torch.nn.Parameter(initial[f"{prefix}_up"]))
                setattr(self, f"{prefix}_gate", torch.nn.Parameter(initial[f"{prefix}_gate"]))
                setattr(self, f"{prefix}_scale", torch.nn.Parameter(initial[f"{prefix}_scale"]))
                setattr(self, f"{prefix}_bias", torch.nn.Parameter(initial[f"{prefix}_bias"]))

        def _mix(self, value: Any) -> Any:
            selected = value[self.source_layer_ids]
            head_mix = self.head_mix_logits.softmax(dim=-1)
            layer_mix = self.layer_mix_logits.softmax(dim=-1)
            head_mixed = torch.einsum("lhwstd,lhws->lhwtd", selected, head_mix)
            return torch.einsum("lhwtd,lhw->lhtd", head_mixed, layer_mix)

        def forward(self, source_kv: Any, position_ids: Any) -> Any:
            positions = torch.as_tensor(position_ids, dtype=torch.long, device=source_kv.device)
            source_value = source_kv.float()
            source_key = _apply_rope_heads(
                source_value[0],
                positions,
                theta=source.rope_theta,
                inverse=True,
            )
            base_key = self._mix(source_key)
            base_value = self._mix(source_value[1])
            outputs: list[Any] = []
            for prefix, base in (("key", base_key), ("value", base_value)):
                mean = getattr(self, f"{prefix}_normalizer_mean").unsqueeze(2)
                scale = getattr(self, f"{prefix}_normalizer_scale").unsqueeze(2)
                normalized = (base - mean) / scale
                latent = torch.einsum(
                    "lhtd,lhdr->lhtr", normalized, getattr(self, f"{prefix}_down")
                )
                residual = torch.einsum(
                    "lhtr,lhrd->lhtd",
                    torch.nn.functional.silu(latent),
                    getattr(self, f"{prefix}_up"),
                )
                output = (
                    base * getattr(self, f"{prefix}_scale").unsqueeze(2)
                    + torch.sigmoid(getattr(self, f"{prefix}_gate")).unsqueeze(2) * residual
                    + getattr(self, f"{prefix}_bias").unsqueeze(2)
                )
                outputs.append(output)
            outputs[0] = _apply_rope_heads(
                outputs[0],
                positions,
                theta=target.rope_theta,
                inverse=False,
            )
            return torch.stack(outputs)

        def runtime_state(self) -> dict[str, Any]:
            state = {
                "source_layer_ids": self.source_layer_ids.detach().cpu(),
                "layer_mix": self.layer_mix_logits.softmax(dim=-1).detach().cpu(),
                "head_mix": self.head_mix_logits.softmax(dim=-1).detach().cpu(),
            }
            for prefix in ("key", "value"):
                for suffix in (
                    "normalizer_mean",
                    "normalizer_scale",
                    "down",
                    "up",
                    "gate",
                    "scale",
                    "bias",
                ):
                    state[f"{prefix}_{suffix}"] = getattr(self, f"{prefix}_{suffix}").detach().cpu()
            return state

    return _TrainableTransport().to(device)


def head_aware_training_objective(
    module: Any,
    source_kv: Any,
    native_target_kv: Any,
    position_ids: Any,
    target_query: Any,
    *,
    native_generation_loss: Any,
    prompt_tail_distillation_loss: Any,
    attention_mask: Any | None = None,
    native_attention_output: Any | None = None,
    contract: TransportLossContract | None = None,
) -> tuple[Any, AttentionLossTerms]:
    """Apply all five frozen loss terms to one sampled training batch."""

    import torch.nn.functional as functional

    transformed = module(source_kv, position_ids)
    logit_kl, output_mse = attention_distillation_terms(
        target_query,
        native_target_kv[0],
        native_target_kv[1],
        transformed[0],
        transformed[1],
        attention_mask=attention_mask,
        native_attention_output=native_attention_output,
    )
    anchor = functional.mse_loss(transformed.float(), native_target_kv.float())
    terms = attention_preserving_loss(
        native_generation_loss=native_generation_loss,
        prompt_tail_distillation_loss=prompt_tail_distillation_loss,
        attention_logit_kl=logit_kl,
        attention_output_mse=output_mse,
        transformed_kv_anchor_loss=anchor,
        contract=contract,
    )
    return transformed, terms


def fit_head_aware_normalizers(
    module: Any,
    batches: Sequence[tuple[Any, Any]],
) -> None:
    """Fit train-split-only per-layer/head normalizers used by transport and OOD."""

    import torch

    if not batches:
        raise ValueError("transport normalizer fitting requires batches")
    sums: dict[str, Any] = {}
    square_sums: dict[str, Any] = {}
    samples = 0
    with torch.no_grad():
        for source_kv, position_ids in batches:
            positions = torch.as_tensor(
                position_ids,
                dtype=torch.long,
                device=source_kv.device,
            )
            if source_kv.ndim != 5 or source_kv.shape[0] != 2:
                raise ValueError("normalizer source KV has an invalid layout")
            if positions.numel() != source_kv.shape[3]:
                raise ValueError("normalizer positions do not match source tokens")
            source = source_kv.float()
            key = _apply_rope_heads(
                source[0],
                positions,
                theta=module.source.rope_theta,
                inverse=True,
            )
            mixed = {"key": module._mix(key), "value": module._mix(source[1])}
            for prefix, value in mixed.items():
                reduced = value.sum(dim=2)
                square_reduced = value.square().sum(dim=2)
                sums[prefix] = sums.get(prefix, torch.zeros_like(reduced)) + reduced
                square_sums[prefix] = (
                    square_sums.get(prefix, torch.zeros_like(square_reduced)) + square_reduced
                )
            samples += int(source_kv.shape[3])
    if samples <= 0:
        raise ValueError("transport normalizer fitting observed no tokens")
    for prefix in ("key", "value"):
        mean = sums[prefix] / samples
        variance = (square_sums[prefix] / samples - mean.square()).clamp_min(1e-6)
        getattr(module, f"{prefix}_normalizer_mean").copy_(mean)
        getattr(module, f"{prefix}_normalizer_scale").copy_(variance.sqrt())


def dynamic_cache_to_head_object(past_key_values: Any) -> Any:
    """Convert a Transformers DynamicCache to `[2, layer, head, token, dim]`."""

    import torch

    layers = getattr(past_key_values, "layers", None)
    if not layers:
        raise ValueError("past_key_values does not contain cache layers")
    keys: list[Any] = []
    values: list[Any] = []
    for layer in layers:
        key = layer.keys
        value = layer.values
        if key is None or value is None or key.ndim != 4 or key.shape != value.shape:
            raise ValueError("cache layer must contain matching rank-four K/V")
        if key.shape[0] != 1:
            raise ValueError("head-aware cache collection requires batch size one")
        keys.append(key[0])
        values.append(value[0])
    return torch.stack((torch.stack(keys), torch.stack(values)))


def head_object_to_dynamic_cache(kv_object: Any, config: Any) -> Any:
    """Build a Transformers DynamicCache from head-structured KV."""

    from transformers.cache_utils import DynamicCache

    if kv_object.ndim != 5 or kv_object.shape[0] != 2:
        raise ValueError("KV object must have [2, layer, head, token, dim] layout")
    heads = int(config.num_key_value_heads)
    head_dim = int(getattr(config, "head_dim", config.hidden_size // config.num_attention_heads))
    if int(kv_object.shape[2]) != heads or int(kv_object.shape[4]) != head_dim:
        raise ValueError("KV object does not match target head layout")
    cache = DynamicCache(config=config)
    for layer in range(int(kv_object.shape[1])):
        cache.update(kv_object[0, layer].unsqueeze(0), kv_object[1, layer].unsqueeze(0), layer)
    return cache


@dataclass(frozen=True)
class AttentionLossTerms:
    native_generation: Any
    prompt_tail_distillation: Any
    attention_logit_kl: Any
    attention_output_mse: Any
    transformed_kv_anchor: Any
    total: Any


@dataclass(frozen=True)
class TransportScreeningCandidate:
    candidate_id: str
    direction: str
    rank: int
    attention_loss_weight: float
    task_score: float
    oracle_safe_coverage: float
    greedy_agreement: float
    transform_cost_ms: float


def select_transport_candidate(
    candidates: Sequence[TransportScreeningCandidate],
) -> TransportScreeningCandidate:
    """Freeze structure using the pre-registered Qwen3 4B->8B ordering only."""

    if not candidates:
        raise ValueError("transport screening requires candidates")
    for candidate in candidates:
        if candidate.direction != "qwen3_4b_to_8b":
            raise ValueError("transport structure may only be screened on Qwen3 4B->8B")
        if candidate.rank not in {32, 64, 128}:
            raise ValueError("transport screening rank is outside the fixed set")
        values = (
            candidate.attention_loss_weight,
            candidate.task_score,
            candidate.oracle_safe_coverage,
            candidate.greedy_agreement,
            candidate.transform_cost_ms,
        )
        if any(not math.isfinite(value) for value in values):
            raise ValueError("transport screening metrics must be finite")
    return max(
        candidates,
        key=lambda item: (
            item.task_score,
            item.oracle_safe_coverage,
            item.greedy_agreement,
            -item.transform_cost_ms,
        ),
    )


def attention_distillation_terms(
    target_query: Any,
    native_key: Any,
    native_value: Any,
    transformed_key: Any,
    transformed_value: Any,
    *,
    attention_mask: Any | None = None,
    native_attention_output: Any | None = None,
) -> tuple[Any, Any]:
    """Compute teacher/student attention KL and output MSE for sampled positions."""

    import torch
    import torch.nn.functional as functional

    if target_query.ndim != 4:
        raise ValueError("target_query must have [layer, query_head, query, dim] layout")
    for value in (native_key, native_value, transformed_key, transformed_value):
        if value.ndim != 4:
            raise ValueError("attention KV samples must have [layer, kv_head, key, dim] layout")
    query_heads = int(target_query.shape[1])
    native_key = _expand_kv_heads(native_key, query_heads)
    native_value = _expand_kv_heads(native_value, query_heads)
    transformed_key = _expand_kv_heads(transformed_key, query_heads)
    transformed_value = _expand_kv_heads(transformed_value, query_heads)
    scale = 1.0 / math.sqrt(int(target_query.shape[-1]))
    native_logits = (
        torch.einsum("lhqd,lhkd->lhqk", target_query.float(), native_key.float()) * scale
    )
    transformed_logits = (
        torch.einsum("lhqd,lhkd->lhqk", target_query.float(), transformed_key.float()) * scale
    )
    if attention_mask is not None:
        mask = torch.as_tensor(attention_mask, dtype=torch.bool, device=native_logits.device)
        native_logits = native_logits.masked_fill(~mask, float("-inf"))
        transformed_logits = transformed_logits.masked_fill(~mask, float("-inf"))
    teacher = native_logits.softmax(dim=-1)
    log_student = transformed_logits.log_softmax(dim=-1)
    # F.kl_div evaluates 0 * -inf at causally masked positions, which produces
    # NaNs even though those positions have zero probability under both models.
    # Reduce the categorical KL only over visible keys, then average examples.
    log_teacher = teacher.clamp_min(torch.finfo(teacher.dtype).tiny).log()
    kl_elements = teacher * (log_teacher - log_student)
    if attention_mask is not None:
        kl_elements = kl_elements.masked_fill(~mask, 0.0)
    logit_kl = kl_elements.sum(dim=-1).mean()
    if native_attention_output is None:
        native_output = torch.einsum("lhqk,lhkd->lhqd", teacher, native_value.float())
    else:
        native_output = torch.as_tensor(
            native_attention_output,
            dtype=torch.float32,
            device=teacher.device,
        )
        if native_output.shape != target_query.shape:
            raise ValueError("native attention output shape does not match target queries")
    transformed_output = torch.einsum(
        "lhqk,lhkd->lhqd", transformed_logits.softmax(dim=-1), transformed_value.float()
    )
    output_mse = functional.mse_loss(transformed_output, native_output)
    return logit_kl, output_mse


def attention_preserving_loss(
    *,
    native_generation_loss: Any,
    prompt_tail_distillation_loss: Any,
    attention_logit_kl: Any,
    attention_output_mse: Any,
    transformed_kv_anchor_loss: Any,
    contract: TransportLossContract | None = None,
) -> AttentionLossTerms:
    weights = contract or TransportLossContract()
    errors = weights.validate()
    if errors:
        raise ValueError("; ".join(errors))
    total = (
        weights.native_generation * native_generation_loss
        + weights.prompt_tail_distillation * prompt_tail_distillation_loss
        + weights.attention_logit_kl * attention_logit_kl
        + weights.attention_output_mse * attention_output_mse
        + weights.transformed_kv_anchor * transformed_kv_anchor_loss
    )
    return AttentionLossTerms(
        native_generation=native_generation_loss,
        prompt_tail_distillation=prompt_tail_distillation_loss,
        attention_logit_kl=attention_logit_kl,
        attention_output_mse=attention_output_mse,
        transformed_kv_anchor=transformed_kv_anchor_loss,
        total=total,
    )


def sample_attention_positions(
    token_count: int,
    *,
    max_queries: int = 32,
    max_keys: int = 256,
) -> tuple[Any, Any]:
    """Deterministically stratify query/key positions without prompt-tail bias."""

    import torch

    if token_count <= 0 or max_queries <= 0 or max_keys <= 0:
        raise ValueError("attention sampling counts must be positive")
    query_count = min(token_count, max_queries)
    key_count = min(token_count, max_keys)
    queries = torch.linspace(0, token_count - 1, steps=query_count).round().long().unique()
    keys = torch.linspace(0, token_count - 1, steps=key_count).round().long().unique()
    return queries, keys


def transport_safetensors_metadata(manifest: SelectiveKVBridgeManifest) -> dict[str, str]:
    return {
        "schema_version": manifest.schema_version,
        "direction": manifest.direction,
        "source_config_sha256": manifest.source.config_sha256,
        "source_tokenizer_sha256": manifest.source.tokenizer_sha256,
        "source_weights_sha256": manifest.source.weights_sha256,
        "target_config_sha256": manifest.target.config_sha256,
        "target_tokenizer_sha256": manifest.target.tokenizer_sha256,
        "target_weights_sha256": manifest.target.weights_sha256,
        "structure_id": manifest.transport.structure_id,
    }


def _compatibility_errors(source: CachedKVModelSpec, target: CachedKVModelSpec) -> list[str]:
    errors: list[str] = []
    errors.extend(f"source: {item}" for item in source.validate())
    errors.extend(f"target: {item}" for item in target.validate())
    if source.architecture != target.architecture:
        errors.append("source and target architectures differ")
    if source.tokenizer_sha256 != target.tokenizer_sha256:
        errors.append("source and target tokenizers differ")
    if source.head_dim != target.head_dim:
        errors.append("source and target head dimensions differ")
    if source.rope_scaling != target.rope_scaling:
        errors.append("source and target RoPE scaling differs")
    if source.sliding_window != target.sliding_window:
        errors.append("source and target sliding-window contracts differ")
    return errors


def _apply_rope_heads(value: Any, positions: Any, *, theta: float, inverse: bool) -> Any:
    """Apply half-split Qwen RoPE to `[..., head, token, head_dim]`."""

    import torch

    if value.ndim < 3 or int(value.shape[-1]) % 2:
        raise HeadAwareTransportError("RoPE input must have an even head dimension")
    token_count = int(value.shape[-2])
    if positions.ndim != 1 or positions.numel() != token_count:
        raise HeadAwareTransportError("RoPE positions do not match the token axis")
    head_dim = int(value.shape[-1])
    frequencies = torch.arange(0, head_dim, 2, dtype=torch.float32, device=value.device)
    inv_freq = 1.0 / (theta ** (frequencies / head_dim))
    angles = torch.outer(positions.float(), inv_freq)
    embedding = torch.cat((angles, angles), dim=-1)
    broadcast = (1,) * (value.ndim - 2) + (token_count, head_dim)
    cosine = embedding.cos().reshape(broadcast).to(value.dtype)
    sine = embedding.sin().reshape(broadcast).to(value.dtype)
    first, second = value[..., : head_dim // 2], value[..., head_dim // 2 :]
    rotated_half = torch.cat((-second, first), dim=-1)
    sign = -1.0 if inverse else 1.0
    return value * cosine + sign * rotated_half * sine


def _expand_kv_heads(value: Any, query_heads: int) -> Any:
    kv_heads = int(value.shape[1])
    if query_heads % kv_heads:
        raise ValueError("query head count must be divisible by KV head count")
    return value.repeat_interleave(query_heads // kv_heads, dim=1)


def _torch_dtype(name: str) -> Any:
    import torch

    try:
        return {"bfloat16": torch.bfloat16, "float16": torch.float16}[name]
    except KeyError as exc:
        raise HeadAwareTransportError(f"unsupported KV dtype: {name}") from exc
