"""Selective RETRIEVE_TRANSFORM path for direct vLLM paged-KV injection."""

from __future__ import annotations

import hashlib
import math
import threading
import time
from collections.abc import Callable, Sequence
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any, Protocol

from goldenexperience.size_variant.head_aware_transport import (
    HeadAwareKVTransport,
)
from goldenexperience.size_variant.risk_gate import (
    AdmissionDecision,
    CalibratedRiskGate,
    RiskGateError,
    SourceKVSidecar,
)

RETRIEVE_TRANSFORM = "RETRIEVE_TRANSFORM"


class DirectInjectionError(RuntimeError):
    """Raised when a direct injection cannot complete atomically."""


class SourceChunkReader(Protocol):
    def read_many_exact(self, keys: Sequence[str], *, timeout_s: float) -> Sequence[Any]: ...


class BlockValidityTracker(Protocol):
    def mark_invalid(self, block_ids: Sequence[int]) -> None: ...

    def mark_valid(self, block_ids: Sequence[int]) -> None: ...


@dataclass(frozen=True)
class RetrieveTransformRequest:
    request_id: str
    source_keys: tuple[str, ...]
    source_checksums: tuple[str, ...]
    chunk_token_counts: tuple[int, ...]
    slot_mapping: tuple[int, ...]
    prefix_hash: str
    sidecar: SourceKVSidecar | bytes | None
    timeout_s: float = 5.0
    position_start: int = 0

    def validate(self) -> list[str]:
        errors: list[str] = []
        if not self.request_id:
            errors.append("request_id is required")
        if not self.source_keys:
            errors.append("at least one source chunk is required")
        if not (
            len(self.source_keys) == len(self.source_checksums) == len(self.chunk_token_counts)
        ):
            errors.append("source chunk metadata lengths differ")
        if len(set(self.source_keys)) != len(self.source_keys):
            errors.append("source chunk keys must be unique")
        for checksum in self.source_checksums:
            if len(checksum) != 64:
                errors.append("source checksums must be SHA-256 digests")
                break
            try:
                int(checksum, 16)
            except ValueError:
                errors.append("source checksums must be SHA-256 digests")
                break
        if any(count <= 0 for count in self.chunk_token_counts):
            errors.append("chunk token counts must be positive")
        if sum(self.chunk_token_counts) != len(self.slot_mapping):
            errors.append("slot mapping length does not match source tokens")
        if any(slot < 0 for slot in self.slot_mapping):
            errors.append("slot mapping contains an invalid slot")
        if len(set(self.slot_mapping)) != len(self.slot_mapping):
            errors.append("slot mapping must not contain duplicate slots")
        try:
            text = self.prefix_hash[2:] if self.prefix_hash.startswith("0x") else self.prefix_hash
            if len(text) != 64:
                raise ValueError
            int(text, 16)
        except (AttributeError, ValueError):
            errors.append("prefix_hash must be a 256-bit hexadecimal digest")
        if not math.isfinite(self.timeout_s) or self.timeout_s <= 0:
            errors.append("timeout_s must be finite and positive")
        if self.position_start < 0:
            errors.append("position_start must be non-negative")
        return errors


@dataclass(frozen=True)
class DirectInjectionResult:
    success: bool
    accepted: bool
    fallback_reason: str
    admission: AdmissionDecision
    source_read_attempted: bool
    source_chunks_read: int
    tokens_scattered: int
    invalidated_blocks: tuple[int, ...]
    target_mooncake_puts: int
    elapsed_ms: float
    error: str | None = None


class InMemoryBlockValidityTracker:
    """Small reference tracker used by connector adapters and tests."""

    def __init__(self) -> None:
        self.valid: set[int] = set()
        self.invalid: set[int] = set()

    def mark_invalid(self, block_ids: Sequence[int]) -> None:
        self.valid.difference_update(block_ids)
        self.invalid.update(block_ids)

    def mark_valid(self, block_ids: Sequence[int]) -> None:
        self.invalid.difference_update(block_ids)
        self.valid.update(block_ids)


class DirectPagedKVInjector:
    """Gate, batch-read, transform, scatter, then publish one load-complete event."""

    def __init__(
        self,
        *,
        risk_gate: CalibratedRiskGate,
        transport: HeadAwareKVTransport,
        source_reader: SourceChunkReader,
        validity_tracker: BlockValidityTracker,
        publish_load_complete: Callable[[str, tuple[int, ...]], None],
        scatter: Callable[[Any, Any, Sequence[int]], tuple[int, ...]] | None = None,
    ) -> None:
        self.risk_gate = risk_gate
        self.transport = transport
        self.source_reader = source_reader
        self.validity_tracker = validity_tracker
        self.publish_load_complete = publish_load_complete
        self.scatter = scatter or scatter_paged_kv
        self._lock = threading.RLock()
        self._stream: Any | None = None
        if self.transport.device.type == "cuda":
            import torch

            self._stream = torch.cuda.Stream(device=self.transport.device)

    @classmethod
    def from_approved_artifact(
        cls,
        manifest_path: str,
        *,
        source_reader: SourceChunkReader,
        validity_tracker: BlockValidityTracker,
        publish_load_complete: Callable[[str, tuple[int, ...]], None],
        observed_target_config_hash: str,
        observed_target_model_hash: str,
        observed_tokenizer_hash: str,
        device: str = "cpu",
        scatter: Callable[[Any, Any, Sequence[int]], tuple[int, ...]] | None = None,
    ) -> DirectPagedKVInjector:
        """Construct the production path only from a fully approved v5 artifact."""

        from goldenexperience.size_variant.risk_gate import RiskPredictor
        from goldenexperience.size_variant.selective_manifest import (
            ArtifactState,
            SelectiveKVBridgeManifest,
        )

        manifest = SelectiveKVBridgeManifest.load(manifest_path)
        errors = manifest.validate()
        if manifest.state is not ArtifactState.APPROVED:
            errors.append("v5 artifact is not approved for direct injection")
        if observed_target_config_hash != manifest.target.config_sha256:
            errors.append("running target config hash differs from the v5 artifact")
        if observed_target_model_hash != manifest.target.weights_sha256:
            errors.append("running target model hash differs from the v5 artifact")
        if observed_tokenizer_hash != manifest.target.tokenizer_sha256:
            errors.append("running tokenizer hash differs from the v5 artifact")
        if errors:
            raise DirectInjectionError("; ".join(errors))
        transport = HeadAwareKVTransport.from_manifest(
            manifest,
            manifest_path,
            device=device,
        )
        predictor = RiskPredictor.from_artifact(
            manifest.resolve_predictor_path(manifest_path),
            expected_sha256=manifest.risk_gate.predictor_sha256,
            device=device,
        )
        gate = CalibratedRiskGate(
            manifest.risk_gate,
            predictor,
            model_pair_id=manifest.direction,
            source_model_hash=manifest.source.weights_sha256,
            target_model_hash=manifest.target.weights_sha256,
            tokenizer_hash=manifest.source.tokenizer_sha256,
            transport_weights_hash=manifest.transport.weights_sha256,
        )
        return cls(
            risk_gate=gate,
            transport=transport,
            source_reader=source_reader,
            validity_tracker=validity_tracker,
            publish_load_complete=publish_load_complete,
            scatter=scatter,
        )

    def retrieve_transform(
        self,
        request: RetrieveTransformRequest,
        *,
        kv_caches: Any,
    ) -> DirectInjectionResult:
        with self._lock:
            return self._retrieve_transform(request, kv_caches=kv_caches)

    def _retrieve_transform(
        self,
        request: RetrieveTransformRequest,
        *,
        kv_caches: Any,
    ) -> DirectInjectionResult:
        started = time.perf_counter()
        try:
            admission = self.risk_gate.evaluate(request.sidecar)
        except Exception as exc:
            return self._result(
                started,
                admission=AdmissionDecision(False, "risk_gate_failure"),
                fallback_reason="risk_gate_failure",
                source_read_attempted=False,
                error=repr(exc),
            )
        # Rejected requests must not touch source KV; target native prefill owns their slots.
        if not admission.accepted:
            return self._result(
                started,
                admission=admission,
                fallback_reason=admission.reason,
                source_read_attempted=False,
            )
        request_errors = request.validate()
        if request_errors:
            return self._result(
                started,
                admission=admission,
                fallback_reason="invalid_retrieve_transform_request",
                source_read_attempted=False,
                error="; ".join(request_errors),
            )
        try:
            sidecar = (
                SourceKVSidecar.from_bytes(request.sidecar)
                if isinstance(request.sidecar, bytes)
                else request.sidecar
            )
            assert sidecar is not None
            normalized_prefix = request.prefix_hash.removeprefix("0x").lower()
            if sidecar.prefix_hash != normalized_prefix:
                raise DirectInjectionError("sidecar prefix hash does not match the request")
            if sidecar.prefix_length != sum(request.chunk_token_counts):
                raise DirectInjectionError("sidecar prefix length does not match source chunks")
        except (AssertionError, RiskGateError, DirectInjectionError) as exc:
            return self._result(
                started,
                admission=admission,
                fallback_reason="sidecar_request_mismatch",
                source_read_attempted=False,
                error=str(exc),
            )

        try:
            block_size = infer_block_size(
                kv_caches,
                self.transport.target.num_key_value_heads,
                self.transport.target.head_dim,
            )
            block_ids = tuple(sorted({slot // block_size for slot in request.slot_mapping}))
        except DirectInjectionError as exc:
            return self._result(
                started,
                admission=admission,
                fallback_reason="invalid_paged_kv_layout",
                source_read_attempted=False,
                error=str(exc),
            )
        source_read_attempted = False
        chunks_read = 0
        deadline = started + request.timeout_s
        try:
            source_read_attempted = True
            raw_chunks = self.source_reader.read_many_exact(
                request.source_keys,
                timeout_s=request.timeout_s,
            )
            if len(raw_chunks) != len(request.source_keys):
                raise DirectInjectionError("source batch read returned the wrong chunk count")
            if time.perf_counter() > deadline:
                raise DirectInjectionError("source batch read exceeded retrieve timeout")
            chunks_read = len(raw_chunks)
            import torch

            stream_context = (
                torch.cuda.stream(self._stream) if self._stream is not None else nullcontext()
            )
            with stream_context:
                source_chunks = [
                    self._decode_chunk(raw, token_count=count, checksum=checksum)
                    for raw, count, checksum in zip(
                        raw_chunks,
                        request.chunk_token_counts,
                        request.source_checksums,
                        strict=True,
                    )
                ]
                self.validity_tracker.mark_invalid(block_ids)
                transformed: list[Any] = []
                position = request.position_start
                for source in source_chunks:
                    if time.perf_counter() > deadline:
                        raise DirectInjectionError("KV transform exceeded retrieve timeout")
                    target = self.transport.transform(source, position_start=position)
                    transformed.append(target)
                    position += int(source.shape[3])
                target_kv = torch.cat(transformed, dim=3)
                scattered_blocks = self.scatter(target_kv, kv_caches, request.slot_mapping)
            if time.perf_counter() > deadline:
                raise DirectInjectionError("paged scatter exceeded retrieve timeout")
            if tuple(sorted(set(scattered_blocks))) != block_ids:
                raise DirectInjectionError("paged scatter reported an inconsistent block set")
            if self._stream is not None:
                self._stream.synchronize()
            if time.perf_counter() > deadline:
                raise DirectInjectionError(
                    "paged scatter synchronization exceeded retrieve timeout"
                )
            self.validity_tracker.mark_valid(block_ids)
            self.publish_load_complete(request.request_id, block_ids)
            return DirectInjectionResult(
                success=True,
                accepted=True,
                fallback_reason="none",
                admission=admission,
                source_read_attempted=True,
                source_chunks_read=chunks_read,
                tokens_scattered=len(request.slot_mapping),
                invalidated_blocks=(),
                target_mooncake_puts=0,
                elapsed_ms=(time.perf_counter() - started) * 1000,
            )
        except Exception as exc:
            # A decode path must never consume partially written pages. Native prefill will
            # overwrite every invalid slot before those blocks can become valid again.
            try:
                self.validity_tracker.mark_invalid(block_ids)
            except Exception as invalidation_error:
                raise DirectInjectionError(
                    "direct injection failed and touched blocks could not be invalidated"
                ) from invalidation_error
            return self._result(
                started,
                admission=admission,
                fallback_reason="direct_injection_failed",
                source_read_attempted=source_read_attempted,
                source_chunks_read=chunks_read,
                invalidated_blocks=block_ids,
                error=repr(exc),
            )

    def _decode_chunk(self, raw: Any, *, token_count: int, checksum: str) -> Any:
        import torch

        expected_shape = (
            2,
            self.transport.source.num_layers,
            self.transport.source.num_key_value_heads,
            token_count,
            self.transport.source.head_dim,
        )
        dtype = _torch_dtype(self.transport.source.dtype)
        expected_bytes = math.prod(expected_shape) * torch.empty((), dtype=dtype).element_size()
        if isinstance(raw, torch.Tensor):
            if tuple(raw.shape) != expected_shape or raw.dtype != dtype:
                raise DirectInjectionError("source tensor shape or dtype mismatch")
            cpu = raw.detach().to("cpu").contiguous()
            payload = bytes(cpu.view(torch.uint8).numpy())
            tensor = raw
        else:
            if hasattr(raw, "data"):
                raw = raw.data
            payload = bytes(raw)
            if len(payload) != expected_bytes:
                raise DirectInjectionError("source chunk byte length mismatch")
            buffer = bytearray(payload)
            tensor = torch.frombuffer(buffer, dtype=dtype).reshape(expected_shape).clone()
            if self.transport.device.type == "cuda":
                tensor = tensor.pin_memory().to(self.transport.device, non_blocking=True)
        if hashlib.sha256(payload).hexdigest() != checksum:
            raise DirectInjectionError("source chunk checksum mismatch")
        return tensor

    @staticmethod
    def _result(
        started: float,
        *,
        admission: AdmissionDecision,
        fallback_reason: str,
        source_read_attempted: bool,
        source_chunks_read: int = 0,
        invalidated_blocks: tuple[int, ...] = (),
        error: str | None = None,
    ) -> DirectInjectionResult:
        return DirectInjectionResult(
            success=False,
            accepted=admission.accepted,
            fallback_reason=fallback_reason,
            admission=admission,
            source_read_attempted=source_read_attempted,
            source_chunks_read=source_chunks_read,
            tokens_scattered=0,
            invalidated_blocks=invalidated_blocks,
            target_mooncake_puts=0,
            elapsed_ms=(time.perf_counter() - started) * 1000,
            error=error,
        )


def infer_block_size(
    kv_caches: Any,
    target_heads: int,
    target_head_dim: int | None = None,
) -> int:
    layers = list(kv_caches)
    if not layers:
        raise DirectInjectionError("vLLM kv_caches is empty")
    key, _ = _layer_key_value(layers[0])
    if key.ndim == 4:
        if int(key.shape[2]) == target_heads:
            return int(key.shape[1])
        if int(key.shape[1]) == target_heads:
            return int(key.shape[2])
    if key.ndim == 5 and int(key.shape[1]) == target_heads:
        if target_head_dim is not None and int(key.shape[2]) * int(key.shape[4]) != target_head_dim:
            raise DirectInjectionError("packed paged KV head dimension is incompatible")
        return int(key.shape[3])
    raise DirectInjectionError("could not infer paged KV block layout")


def scatter_paged_kv(
    target_kv: Any,
    kv_caches: Any,
    slot_mapping: Sequence[int],
) -> tuple[int, ...]:
    """Scatter complete head-structured KV into common vLLM page layouts."""

    import torch

    if not isinstance(target_kv, torch.Tensor) or target_kv.ndim != 5 or target_kv.shape[0] != 2:
        raise DirectInjectionError("target KV must have [2, layer, head, token, dim] layout")
    layers = list(kv_caches)
    if len(layers) != int(target_kv.shape[1]):
        raise DirectInjectionError("target layer count does not match vLLM kv_caches")
    if len(slot_mapping) != int(target_kv.shape[3]):
        raise DirectInjectionError("slot mapping does not match target token count")
    target_heads = int(target_kv.shape[2])
    head_dim = int(target_kv.shape[4])
    block_size = infer_block_size(layers, target_heads, head_dim)
    slots = torch.as_tensor(slot_mapping, dtype=torch.long, device=target_kv.device)
    blocks = torch.div(slots, block_size, rounding_mode="floor")
    offsets = torch.remainder(slots, block_size)
    token_key = target_kv[0].permute(0, 2, 1, 3)
    token_value = target_kv[1].permute(0, 2, 1, 3)
    for layer_id, layer_cache in enumerate(layers):
        key_cache, value_cache = _layer_key_value(layer_cache)
        if key_cache.device != target_kv.device or value_cache.device != target_kv.device:
            raise DirectInjectionError("target KV and paged caches must share a device")
        if key_cache.dtype != target_kv.dtype or value_cache.dtype != target_kv.dtype:
            raise DirectInjectionError("target KV and paged caches must share a dtype")
        if key_cache.ndim == 4 and int(key_cache.shape[2]) == target_heads:
            key_cache[blocks, offsets] = token_key[layer_id]
            value_cache[blocks, offsets] = token_value[layer_id]
        elif key_cache.ndim == 4 and int(key_cache.shape[1]) == target_heads:
            key_cache[blocks, :, offsets, :] = token_key[layer_id]
            value_cache[blocks, :, offsets, :] = token_value[layer_id]
        elif key_cache.ndim == 5 and int(key_cache.shape[1]) == target_heads:
            chunks = int(key_cache.shape[2])
            width = int(key_cache.shape[4])
            packed_key = token_key[layer_id].reshape(-1, target_heads, chunks, width)
            packed_value = token_value[layer_id].reshape(-1, target_heads, chunks, width)
            key_cache[blocks, :, :, offsets, :] = packed_key
            value_cache[blocks, :, :, offsets, :] = packed_value
        else:
            raise DirectInjectionError("unsupported paged KV tensor layout")
    return tuple(sorted({int(item) for item in blocks.tolist()}))


def _layer_key_value(layer_cache: Any) -> tuple[Any, Any]:
    import torch

    if isinstance(layer_cache, torch.Tensor):
        if layer_cache.ndim not in {5, 6}:
            raise DirectInjectionError("combined layer cache must contain a K/V axis")
        # vLLM 0.24 registers `[blocks, 2, ...]`; the axis-zero form is kept for
        # connector test doubles and older adapters.
        if int(layer_cache.shape[1]) == 2:
            return layer_cache[:, 0], layer_cache[:, 1]
        if int(layer_cache.shape[0]) == 2:
            return layer_cache[0], layer_cache[1]
        raise DirectInjectionError("combined layer cache has no K/V axis")
    if isinstance(layer_cache, (tuple, list)) and len(layer_cache) == 2:
        key, value = layer_cache
        if not isinstance(key, torch.Tensor) or not isinstance(value, torch.Tensor):
            raise DirectInjectionError("separate layer caches must be tensors")
        if key.shape != value.shape:
            raise DirectInjectionError("paged key/value cache shapes differ")
        return key, value
    raise DirectInjectionError("unsupported vLLM kv_caches layer representation")


def _torch_dtype(name: str) -> Any:
    import torch

    try:
        return {"bfloat16": torch.bfloat16, "float16": torch.float16}[name]
    except KeyError as exc:
        raise DirectInjectionError(f"unsupported source KV dtype: {name}") from exc
