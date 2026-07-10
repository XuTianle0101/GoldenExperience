"""Direction-independent Qwen3 cached-KV tensor translation."""

from __future__ import annotations

import os
import threading
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from goldenexperience.size_variant.cached_kv_manifest import (
    CACHED_KV_SCHEMA_VERSION,
    CachedKVBridgeManifest,
    model_identity_paths,
    sha256_file,
    verify_model_path,
)

REQUIRED_WEIGHT_TENSORS = frozenset(
    {
        "source_layer_ids",
        "source_layer_weights",
        "feature_mean",
        "key_base_scale",
        "key_down",
        "key_up",
        "key_bias",
        "value_base_scale",
        "value_down",
        "value_up",
        "value_bias",
    }
)


class CachedKVBridgeError(RuntimeError):
    """Raised when a cached-KV artifact or source object is unsafe to use."""


@dataclass(frozen=True)
class _ResidentBridgeEntry:
    bridge: Qwen3CachedKVBridge
    dependency_paths: tuple[Path, ...]
    dependency_snapshot: tuple[tuple[str, int, int, int, int, int], ...]


class Qwen3CachedKVBridge:
    """Translate real cached Qwen3 KV objects without loading either model."""

    def __init__(
        self,
        manifest: CachedKVBridgeManifest,
        tensors: Mapping[str, Any],
        *,
        device: str = "cpu",
        compute_dtype: Any | None = None,
    ) -> None:
        manifest_errors = manifest.validate()
        if manifest_errors:
            raise CachedKVBridgeError("; ".join(manifest_errors))
        self._initialize(manifest, tensors, device=device, compute_dtype=compute_dtype)

    def _initialize(
        self,
        manifest: CachedKVBridgeManifest,
        tensors: Mapping[str, Any],
        *,
        device: str,
        compute_dtype: Any | None,
    ) -> None:
        import torch

        self.manifest = manifest
        self.device = torch.device(device)
        self.compute_dtype = compute_dtype or (
            torch.bfloat16 if self.device.type == "cuda" else torch.float32
        )
        self._tensors = {
            name: tensor.detach().to(device=self.device) for name, tensor in tensors.items()
        }
        self._validate_tensors()

    @classmethod
    def from_artifact(
        cls,
        manifest_path: str | Path,
        *,
        source_model_path: str | Path,
        target_model_path: str | Path,
        device: str = "cpu",
        compute_dtype: Any | None = None,
    ) -> Qwen3CachedKVBridge:
        """Load an approved artifact after model and weight content verification."""

        return cls._from_artifact_path(
            manifest_path,
            source_model_path=source_model_path,
            target_model_path=target_model_path,
            device=device,
            compute_dtype=compute_dtype,
            benchmark_candidate=False,
        )

    @classmethod
    def from_validation_candidate_for_benchmark(
        cls,
        manifest_path: str | Path,
        *,
        source_model_path: str | Path,
        target_model_path: str | Path,
        device: str = "cpu",
        compute_dtype: Any | None = None,
    ) -> Qwen3CachedKVBridge:
        """Load an unapproved validation artifact for non-publishing cost benchmarks."""

        return cls._from_artifact_path(
            manifest_path,
            source_model_path=source_model_path,
            target_model_path=target_model_path,
            device=device,
            compute_dtype=compute_dtype,
            benchmark_candidate=True,
        )

    @classmethod
    def _from_artifact_path(
        cls,
        manifest_path: str | Path,
        *,
        source_model_path: str | Path,
        target_model_path: str | Path,
        device: str,
        compute_dtype: Any | None,
        benchmark_candidate: bool,
    ) -> Qwen3CachedKVBridge:

        from safetensors import safe_open

        path = Path(manifest_path).resolve()
        manifest = CachedKVBridgeManifest.load(path)
        errors = manifest.artifact_errors() if benchmark_candidate else manifest.validate()
        if errors:
            raise CachedKVBridgeError("; ".join(errors))
        source_errors = verify_model_path(manifest.source, source_model_path)
        target_errors = verify_model_path(manifest.target, target_model_path)
        if source_errors or target_errors:
            messages = [f"source model: {item}" for item in source_errors]
            messages.extend(f"target model: {item}" for item in target_errors)
            raise CachedKVBridgeError("; ".join(messages))

        weights_path = manifest.resolve_weights_path(path)
        if weights_path.suffix != ".safetensors" or not weights_path.is_file():
            raise CachedKVBridgeError("cached KV weights must be an existing safetensors file")
        before = weights_path.stat()
        if sha256_file(weights_path) != manifest.weights_sha256:
            raise CachedKVBridgeError("cached KV weight checksum mismatch")
        tensors: dict[str, Any] = {}
        with safe_open(str(weights_path), framework="pt", device="cpu") as handle:
            metadata = handle.metadata() or {}
            _validate_safetensors_metadata(manifest, metadata)
            for name in handle.keys():  # noqa: SIM118 - safe_open is not iterable.
                tensors[name] = handle.get_tensor(name)
        after = weights_path.stat()
        if (before.st_size, before.st_mtime_ns, before.st_ino) != (
            after.st_size,
            after.st_mtime_ns,
            after.st_ino,
        ):
            raise CachedKVBridgeError("cached KV weights changed while loading")
        if sha256_file(weights_path) != manifest.weights_sha256:
            raise CachedKVBridgeError("cached KV weights changed while loading")
        final = weights_path.stat()
        if (after.st_size, after.st_mtime_ns, after.st_ino) != (
            final.st_size,
            final.st_mtime_ns,
            final.st_ino,
        ):
            raise CachedKVBridgeError("cached KV weights changed while loading")
        instance = cls.__new__(cls)
        instance._initialize(
            manifest,
            tensors,
            device=device,
            compute_dtype=compute_dtype,
        )
        return instance

    @property
    def source_object_shape(self) -> tuple[int, int, int]:
        """Static `[kv, layers, width]` dimensions around the token axis."""

        return (2, self.manifest.source.num_layers, self.manifest.source.kv_width)

    @property
    def target_object_shape(self) -> tuple[int, int, int]:
        return (2, self.manifest.target.num_layers, self.manifest.target.kv_width)

    def transform(
        self,
        source_kv: Any,
        *,
        position_start: int = 0,
        position_ids: Any | None = None,
    ) -> Any:
        """Translate `[2, source_layers, tokens, width]` to the target layout."""

        import torch

        self._validate_source(source_kv)
        token_count = int(source_kv.shape[2])
        positions = self._positions(
            token_count=token_count,
            position_start=position_start,
            position_ids=position_ids,
        )
        source = source_kv.to(device=self.device, dtype=self.compute_dtype)
        layer_ids = self._tensors["source_layer_ids"].long()
        layer_weights = self._tensors["source_layer_weights"].to(self.compute_dtype)

        selected_key = source[0][layer_ids]
        selected_value = source[1][layer_ids]
        unrotated_key = _apply_rope_flat(
            selected_key,
            positions,
            num_heads=self.manifest.source.num_key_value_heads,
            head_dim=self.manifest.source.head_dim,
            theta=self.manifest.source.rope_theta,
            inverse=True,
        )
        key_features = unrotated_key.permute(0, 2, 1, 3).reshape(
            self.manifest.target.num_layers,
            token_count,
            -1,
        )
        value_features = selected_value.permute(0, 2, 1, 3).reshape(
            self.manifest.target.num_layers,
            token_count,
            -1,
        )
        features = torch.cat((key_features, value_features), dim=-1)
        centered = features - self._tensors["feature_mean"].to(self.compute_dtype).unsqueeze(1)

        base_key = torch.einsum("ls,lstw->ltw", layer_weights, unrotated_key)
        base_key = base_key * self._tensors["key_base_scale"].to(self.compute_dtype).unsqueeze(1)
        base_value = torch.einsum("ls,lstw->ltw", layer_weights, selected_value)
        base_value = base_value * self._tensors["value_base_scale"].to(
            self.compute_dtype
        ).unsqueeze(1)
        key_delta = torch.bmm(
            torch.bmm(centered, self._tensors["key_down"].to(self.compute_dtype)),
            self._tensors["key_up"].to(self.compute_dtype),
        )
        value_delta = torch.bmm(
            torch.bmm(centered, self._tensors["value_down"].to(self.compute_dtype)),
            self._tensors["value_up"].to(self.compute_dtype),
        )
        target_key_unrotated = (
            base_key + key_delta + self._tensors["key_bias"].to(self.compute_dtype).unsqueeze(1)
        )
        target_value = (
            base_value
            + value_delta
            + self._tensors["value_bias"].to(self.compute_dtype).unsqueeze(1)
        )
        target_key = _apply_rope_flat(
            target_key_unrotated,
            positions,
            num_heads=self.manifest.target.num_key_value_heads,
            head_dim=self.manifest.target.head_dim,
            theta=self.manifest.target.rope_theta,
            inverse=False,
        )
        target = torch.stack((target_key, target_value), dim=0).to(dtype=source_kv.dtype)
        if not bool(torch.isfinite(target).all()):
            raise CachedKVBridgeError("cached KV bridge produced non-finite values")
        return target.contiguous()

    def _positions(
        self,
        *,
        token_count: int,
        position_start: int,
        position_ids: Any | None,
    ) -> Any:
        import torch

        if position_ids is None:
            if position_start < 0:
                raise CachedKVBridgeError("position_start must be non-negative")
            positions = torch.arange(
                position_start,
                position_start + token_count,
                dtype=torch.long,
                device=self.device,
            )
        else:
            positions = torch.as_tensor(position_ids, dtype=torch.long, device=self.device)
            if positions.ndim != 1 or int(positions.shape[0]) != token_count:
                raise CachedKVBridgeError("position_ids must have one entry per token")
            if bool((positions < 0).any()):
                raise CachedKVBridgeError("position_ids must be non-negative")
        if token_count <= 0:
            raise CachedKVBridgeError("source cached KV object has no tokens")
        if int(positions.max().item()) >= self.manifest.source.max_position_embeddings:
            raise CachedKVBridgeError("position_ids exceed the model RoPE contract")
        return positions

    def _validate_source(self, source_kv: Any) -> None:
        import torch

        if not isinstance(source_kv, torch.Tensor):
            raise CachedKVBridgeError("source cached KV object must be a torch.Tensor")
        expected = (
            2,
            self.manifest.source.num_layers,
            self.manifest.source.kv_width,
        )
        if (
            source_kv.ndim != 4
            or (
                int(source_kv.shape[0]),
                int(source_kv.shape[1]),
                int(source_kv.shape[3]),
            )
            != expected
        ):
            raise CachedKVBridgeError(
                f"source cached KV shape must be [2, {expected[1]}, tokens, {expected[2]}]"
            )
        expected_dtype = _torch_dtype(self.manifest.source.dtype)
        if source_kv.dtype != expected_dtype:
            raise CachedKVBridgeError(
                f"source cached KV dtype must be {self.manifest.source.dtype}"
            )
        if not bool(torch.isfinite(source_kv).all()):
            raise CachedKVBridgeError("source cached KV object contains non-finite values")

    def _validate_tensors(self) -> None:
        import torch

        names = set(self._tensors)
        if names != REQUIRED_WEIGHT_TENSORS:
            missing = sorted(REQUIRED_WEIGHT_TENSORS - names)
            unknown = sorted(names - REQUIRED_WEIGHT_TENSORS)
            raise CachedKVBridgeError(
                f"cached KV weight tensor set mismatch; missing={missing}, unknown={unknown}"
            )
        target_layers = self.manifest.target.num_layers
        source_window = self.manifest.source_window
        width = self.manifest.source.kv_width
        feature_width = source_window * width * 2
        rank = self.manifest.rank
        if rank > min(feature_width, width):
            raise CachedKVBridgeError("bridge rank exceeds the low-rank matrix dimensions")
        shapes = {
            "source_layer_ids": (target_layers, source_window),
            "source_layer_weights": (target_layers, source_window),
            "feature_mean": (target_layers, feature_width),
            "key_base_scale": (target_layers, width),
            "key_down": (target_layers, feature_width, rank),
            "key_up": (target_layers, rank, width),
            "key_bias": (target_layers, width),
            "value_base_scale": (target_layers, width),
            "value_down": (target_layers, feature_width, rank),
            "value_up": (target_layers, rank, width),
            "value_bias": (target_layers, width),
        }
        for name, expected in shapes.items():
            tensor = self._tensors[name]
            if tuple(tensor.shape) != expected:
                raise CachedKVBridgeError(
                    f"{name} shape {tuple(tensor.shape)} does not match {expected}"
                )
            if name == "source_layer_ids":
                if tensor.dtype not in {torch.int32, torch.int64}:
                    raise CachedKVBridgeError("source_layer_ids must be an integer tensor")
            elif not tensor.is_floating_point():
                raise CachedKVBridgeError(f"{name} must be a floating-point tensor")
            elif not bool(torch.isfinite(tensor).all()):
                raise CachedKVBridgeError(f"{name} contains non-finite values")
        layer_ids = self._tensors["source_layer_ids"]
        if bool((layer_ids < 0).any()) or bool(
            (layer_ids >= self.manifest.source.num_layers).any()
        ):
            raise CachedKVBridgeError("source_layer_ids contains an out-of-range layer")
        weights = self._tensors["source_layer_weights"].float()
        if bool((weights < 0).any()):
            raise CachedKVBridgeError("source_layer_weights must be non-negative")
        if not torch.allclose(
            weights.sum(dim=-1),
            torch.ones(target_layers, device=weights.device),
            atol=1e-5,
            rtol=0,
        ):
            raise CachedKVBridgeError("source_layer_weights must sum to one per target layer")


class ResidentQwen3CachedKVBridgeCache:
    """Reuse verified in-memory bridges while all on-disk identities stay unchanged."""

    def __init__(self) -> None:
        self._entries: dict[tuple[str, str, str, str, str], _ResidentBridgeEntry] = {}
        self._lock = threading.RLock()

    def load(
        self,
        manifest_path: str | Path,
        *,
        source_model_path: str | Path,
        target_model_path: str | Path,
        device: str = "cpu",
        compute_dtype: Any | None = None,
    ) -> tuple[Qwen3CachedKVBridge, bool]:
        key = (
            str(Path(manifest_path).resolve()),
            str(Path(source_model_path).resolve()),
            str(Path(target_model_path).resolve()),
            str(device),
            repr(compute_dtype),
        )
        with self._lock:
            entry = self._entries.get(key)
            if entry is not None:
                try:
                    current = _dependency_snapshot(entry.dependency_paths)
                except OSError:
                    current = ()
                if current == entry.dependency_snapshot:
                    return entry.bridge, True
                self._entries.pop(key, None)

            bridge = Qwen3CachedKVBridge.from_artifact(
                manifest_path,
                source_model_path=source_model_path,
                target_model_path=target_model_path,
                device=device,
                compute_dtype=compute_dtype,
            )
            resolved_manifest = Path(manifest_path).resolve()
            dependency_paths = (
                resolved_manifest,
                bridge.manifest.resolve_weights_path(resolved_manifest).resolve(),
                *model_identity_paths(source_model_path),
                *model_identity_paths(target_model_path),
            )
            try:
                snapshot = _dependency_snapshot(dependency_paths)
            except OSError as exc:
                raise CachedKVBridgeError(
                    "cached KV dependencies changed after artifact verification"
                ) from exc
            self._entries[key] = _ResidentBridgeEntry(
                bridge=bridge,
                dependency_paths=dependency_paths,
                dependency_snapshot=snapshot,
            )
            return bridge, False

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()


def safetensors_metadata(manifest: CachedKVBridgeManifest) -> dict[str, str]:
    """Metadata that training must embed in the safetensors file."""

    return {
        "schema_version": CACHED_KV_SCHEMA_VERSION,
        "direction": manifest.direction,
        "source_config_sha256": manifest.source.config_sha256,
        "source_tokenizer_sha256": manifest.source.tokenizer_sha256,
        "source_weights_sha256": manifest.source.weights_sha256,
        "target_config_sha256": manifest.target.config_sha256,
        "target_tokenizer_sha256": manifest.target.tokenizer_sha256,
        "target_weights_sha256": manifest.target.weights_sha256,
    }


def _validate_safetensors_metadata(
    manifest: CachedKVBridgeManifest,
    metadata: Mapping[str, str],
) -> None:
    expected = safetensors_metadata(manifest)
    if dict(metadata) != expected:
        raise CachedKVBridgeError("safetensors metadata does not match bridge manifest")


def _torch_dtype(name: str) -> Any:
    import torch

    mapping = {"bfloat16": torch.bfloat16, "float16": torch.float16}
    try:
        return mapping[name]
    except KeyError as exc:
        raise CachedKVBridgeError(f"unsupported cached KV dtype: {name}") from exc


def _dependency_snapshot(
    paths: tuple[Path, ...],
) -> tuple[tuple[str, int, int, int, int, int], ...]:
    snapshot: list[tuple[str, int, int, int, int, int]] = []
    for path in paths:
        stat = path.stat()
        snapshot.append(
            (
                str(path),
                stat.st_dev,
                stat.st_ino,
                stat.st_size,
                stat.st_mtime_ns,
                stat.st_ctime_ns,
            )
        )
    return tuple(snapshot)


def _rotate_half(value: Any) -> Any:
    import torch

    first = value[..., : value.shape[-1] // 2]
    second = value[..., value.shape[-1] // 2 :]
    return torch.cat((-second, first), dim=-1)


def _apply_rope_flat(
    value: Any,
    positions: Any,
    *,
    num_heads: int,
    head_dim: int,
    theta: float,
    inverse: bool,
) -> Any:
    """Apply Qwen's half-split RoPE to `[... layer, token, width]` tensors."""

    import torch

    token_count = int(value.shape[-2])
    leading = value.shape[:-2]
    reshaped = value.reshape(*leading, token_count, num_heads, head_dim)
    frequency_ids = torch.arange(0, head_dim, 2, dtype=torch.float32, device=value.device)
    inv_freq = 1.0 / (theta ** (frequency_ids / head_dim))
    frequencies = torch.outer(positions.to(torch.float32), inv_freq)
    embedding = torch.cat((frequencies, frequencies), dim=-1)
    broadcast = (1,) * len(leading) + (token_count, 1, head_dim)
    cosine = embedding.cos().reshape(broadcast).to(dtype=value.dtype)
    sine = embedding.sin().reshape(broadcast).to(dtype=value.dtype)
    if inverse:
        rotated = reshaped * cosine - _rotate_half(reshaped) * sine
    else:
        rotated = reshaped * cosine + _rotate_half(reshaped) * sine
    return rotated.reshape(*leading, token_count, num_heads * head_dim)


def fsync_replace(source: Path, target: Path) -> None:
    """Publish a completed artifact atomically when a trainer needs it."""

    source.replace(target)
    descriptor = os.open(target.parent, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
