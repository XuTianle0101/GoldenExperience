"""Pinned LMCache MP 0.4.6 source retrieval for direct RETRIEVE_TRANSFORM."""

from __future__ import annotations

import hashlib
import importlib.metadata
import importlib.util
import inspect
import json
import math
import os
import pickle
import threading
import time
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from goldenexperience.runtime.direct_paged_kv import (
    DirectInjectionResult,
    DirectPagedKVInjector,
    RetrieveTransformRequest,
)
from goldenexperience.size_variant.cached_kv_manifest import CachedKVModelSpec, sha256_file

LMCACHE_RETRIEVE_TRANSFORM_SCHEMA = "goldenexperience.lmcache_retrieve_transform.v1"
RUNTIME_STACK_SCHEMA = "goldenexperience.runtime_stack_identity.v2"
EXPECTED_LMCACHE_VERSION = "0.4.6"
EXPECTED_VLLM_VERSION = "0.24.0"
EXPECTED_TORCH_VERSION_PREFIX = "2.11.0"

_PINNED_RUNTIME_MODULES = (
    "lmcache.integration.vllm.lmcache_mp_connector",
    "lmcache.integration.vllm.vllm_multi_process_adapter",
    "lmcache.v1.multiprocess.custom_types",
    "lmcache.v1.multiprocess.transfer_context",
    "lmcache.v1.multiprocess.protocols.base",
    "lmcache.v1.multiprocess.protocols.engine",
    "vllm.config.kv_transfer",
    "vllm.distributed.kv_transfer.kv_connector.v1.base",
    "vllm.v1.core.sched.scheduler",
    "vllm.v1.worker.gpu.kv_connector",
)


class LMCacheRetrieveTransformError(RuntimeError):
    """Raised when the pinned LMCache/vLLM bridge cannot fail closed."""


@dataclass(frozen=True)
class RuntimeSourceIdentity:
    module: str
    distribution_relative_path: str
    sha256: str

    def validate(self) -> list[str]:
        errors: list[str] = []
        if not self.module or not self.distribution_relative_path:
            errors.append("runtime source identity is incomplete")
        if not _is_sha256(self.sha256):
            errors.append("runtime source identity hash is invalid")
        path = Path(self.distribution_relative_path)
        if path.is_absolute() or ".." in path.parts:
            errors.append("runtime source identity path is not distribution-relative")
        return errors


@dataclass(frozen=True)
class RuntimeStackIdentity:
    lmcache_version: str
    vllm_version: str
    torch_version: str
    cuda_version: str | None
    sources: tuple[RuntimeSourceIdentity, ...]
    connector_class: str = "lmcache.integration.vllm.lmcache_mp_connector.LMCacheMPConnector"
    store_protocol: str = "PREPARE_STORE+COMMIT_STORE"
    retrieve_protocol: str = "LOOKUP+QUERY_PREFETCH_STATUS+PREPARE_RETRIEVE+COMMIT_RETRIEVE"
    failure_policy: str = "vllm_invalid_block_native_recompute"
    schema_version: str = RUNTIME_STACK_SCHEMA

    def validate(self) -> list[str]:
        errors: list[str] = []
        if self.schema_version != RUNTIME_STACK_SCHEMA:
            errors.append("unsupported runtime stack identity schema")
        if self.lmcache_version != EXPECTED_LMCACHE_VERSION:
            errors.append("runtime audit requires LMCache 0.4.6")
        if self.vllm_version != EXPECTED_VLLM_VERSION:
            errors.append("runtime audit requires vLLM 0.24.0")
        if not self.torch_version.startswith(EXPECTED_TORCH_VERSION_PREFIX):
            errors.append("runtime audit requires the verified Torch 2.11.0 stack")
        if self.connector_class != (
            "lmcache.integration.vllm.lmcache_mp_connector.LMCacheMPConnector"
        ):
            errors.append("runtime connector class changed")
        if self.store_protocol != "PREPARE_STORE+COMMIT_STORE":
            errors.append("runtime source store protocol changed")
        if self.retrieve_protocol != (
            "LOOKUP+QUERY_PREFETCH_STATUS+PREPARE_RETRIEVE+COMMIT_RETRIEVE"
        ):
            errors.append("runtime source retrieval protocol changed")
        if self.failure_policy != "vllm_invalid_block_native_recompute":
            errors.append("runtime failure recovery policy changed")
        if tuple(item.module for item in self.sources) != _PINNED_RUNTIME_MODULES:
            errors.append("runtime source identity set changed")
        for source in self.sources:
            errors.extend(source.validate())
        return errors

    def content_sha256(self) -> str:
        return _sha256_bytes(_canonical_json_bytes(asdict(self)))


@dataclass(frozen=True)
class LMCacheRetrieveTransformMetadata:
    """One worker-side source request carried beside standard LMCache metadata."""

    request: RetrieveTransformRequest
    token_ids: tuple[int, ...]
    source_ranges: tuple[tuple[int, int], ...]
    source_model_name: str
    source_world_size: int
    source_worker_id: int
    cache_salt: str = ""
    schema_version: str = LMCACHE_RETRIEVE_TRANSFORM_SCHEMA

    def validate(self, *, chunk_size: int | None = None) -> list[str]:
        errors = self.request.validate()
        if self.schema_version != LMCACHE_RETRIEVE_TRANSFORM_SCHEMA:
            errors.append("unsupported LMCache RETRIEVE_TRANSFORM metadata schema")
        if not self.token_ids or any(type(item) is not int or item < 0 for item in self.token_ids):
            errors.append("LMCache source token ids are invalid")
        if not self.source_model_name:
            errors.append("LMCache source model name is required")
        if type(self.source_world_size) is not int or self.source_world_size != 1:
            errors.append("LMCache RETRIEVE_TRANSFORM requires source world size one")
        if type(self.source_worker_id) is not int or self.source_worker_id != 0:
            errors.append("LMCache RETRIEVE_TRANSFORM requires source worker zero")
        if not _valid_cache_salt(self.cache_salt):
            errors.append("LMCache source cache salt is invalid")
        if len(self.source_ranges) != len(self.request.source_keys):
            errors.append("LMCache source ranges differ from source keys")
        previous_end: int | None = None
        for index, item in enumerate(self.source_ranges):
            if (
                not isinstance(item, tuple)
                or len(item) != 2
                or any(type(value) is not int for value in item)
            ):
                errors.append("LMCache source range is malformed")
                continue
            start, end = item
            if start < 0 or end <= start or end > len(self.token_ids):
                errors.append("LMCache source range is outside the token sequence")
                continue
            if previous_end is not None and start != previous_end:
                errors.append("LMCache source ranges are not contiguous")
            previous_end = end
            if index < len(self.request.chunk_token_counts) and (
                end - start != self.request.chunk_token_counts[index]
            ):
                errors.append("LMCache source range length differs from chunk metadata")
            if chunk_size is not None and end - start != chunk_size:
                errors.append("LMCache source range differs from the server chunk size")
            expected_key = lmcache_source_key(self.request.request_id, start, end)
            if index < len(self.request.source_keys) and (
                self.request.source_keys[index] != expected_key
            ):
                errors.append("LMCache source key is not canonical")
        if self.source_ranges:
            if self.source_ranges[0][0] != self.request.position_start:
                errors.append("LMCache source ranges start at another position")
            if self.source_ranges[-1][1] - self.request.position_start != len(
                self.request.slot_mapping
            ):
                errors.append("LMCache source ranges do not cover the paged slot mapping")
        return errors


@dataclass(frozen=True)
class LMCacheStoredSourcePrefix:
    request_id: str
    token_ids: tuple[int, ...]
    source_ranges: tuple[tuple[int, int], ...]
    source_keys: tuple[str, ...]
    source_checksums: tuple[str, ...]
    source_model_name: str
    source_world_size: int
    source_worker_id: int
    cache_salt: str

    def build_retrieve_metadata(
        self,
        *,
        slot_mapping: Sequence[int],
        prefix_hash: str,
        sidecar: Any,
        timeout_s: float = 5.0,
    ) -> LMCacheRetrieveTransformMetadata:
        request = RetrieveTransformRequest(
            request_id=self.request_id,
            source_keys=self.source_keys,
            source_checksums=self.source_checksums,
            chunk_token_counts=tuple(end - start for start, end in self.source_ranges),
            slot_mapping=tuple(slot_mapping),
            prefix_hash=prefix_hash,
            sidecar=sidecar,
            timeout_s=timeout_s,
        )
        metadata = LMCacheRetrieveTransformMetadata(
            request=request,
            token_ids=self.token_ids,
            source_ranges=self.source_ranges,
            source_model_name=self.source_model_name,
            source_world_size=self.source_world_size,
            source_worker_id=self.source_worker_id,
            cache_salt=self.cache_salt,
        )
        errors = metadata.validate(
            chunk_size=self.source_ranges[0][1] - self.source_ranges[0][0]
            if self.source_ranges
            else None
        )
        if errors:
            raise LMCacheRetrieveTransformError("; ".join(errors))
        return metadata


@dataclass(frozen=True)
class LMCacheRetrieveTransformBatch:
    requests: tuple[LMCacheRetrieveTransformMetadata, ...]
    standard_metadata: Any | None = None


@dataclass(frozen=True)
class LMCacheBridgeObservation:
    request_id: str
    result: DirectInjectionResult
    load_complete_published: bool
    invalid_blocks_reported: tuple[int, ...]


class RuntimeBlockValidityTracker:
    """Tracks atomic page publication and drains failures into vLLM's API."""

    def __init__(self) -> None:
        self._valid: set[int] = set()
        self._invalid: set[int] = set()
        self._lock = threading.Lock()

    @property
    def valid(self) -> frozenset[int]:
        with self._lock:
            return frozenset(self._valid)

    @property
    def invalid(self) -> frozenset[int]:
        with self._lock:
            return frozenset(self._invalid)

    def mark_invalid(self, block_ids: Sequence[int]) -> None:
        with self._lock:
            self._valid.difference_update(block_ids)
            self._invalid.update(block_ids)

    def mark_valid(self, block_ids: Sequence[int]) -> None:
        with self._lock:
            self._invalid.difference_update(block_ids)
            self._valid.update(block_ids)

    def drain_invalid(self) -> set[int]:
        with self._lock:
            result = set(self._invalid)
            self._invalid.clear()
        return result


class LMCacheMPSourceChunkWriter:
    """Store full source-model KV chunks through LMCache MP's non-GPU protocol."""

    def __init__(
        self,
        *,
        server_url: str,
        source_model_name: str,
        source_world_size: int,
        source_worker_id: int,
        source: CachedKVModelSpec,
        mq_timeout_s: float = 30.0,
    ) -> None:
        if mq_timeout_s <= 0 or not math.isfinite(mq_timeout_s):
            raise LMCacheRetrieveTransformError("LMCache MQ timeout must be finite and positive")
        if not source_model_name or source_world_size != 1 or source_worker_id != 0:
            raise LMCacheRetrieveTransformError("LMCache source writer identity is invalid")
        self.stack_identity = probe_runtime_stack()
        self.server_url = server_url
        self.source_model_name = source_model_name
        self.source_world_size = source_world_size
        self.source_worker_id = source_worker_id
        self.source = source
        self.mq_timeout_s = mq_timeout_s
        self.source_put_count = 0
        self._lock = threading.RLock()
        self._closed = False

        import zmq
        from lmcache.integration.vllm.vllm_multi_process_adapter import (  # type: ignore[import-untyped]
            get_lmcache_chunk_size,
            send_lmcache_request,
        )
        from lmcache.v1.multiprocess.custom_types import (  # type: ignore[import-untyped]
            RegisterNonGpuContextPayload,
        )
        from lmcache.v1.multiprocess.mq import (  # type: ignore[import-untyped]
            MessageQueueClient,
        )
        from lmcache.v1.multiprocess.protocol import RequestType  # type: ignore[import-untyped]

        self._send_request = send_lmcache_request
        self._request_type = RequestType
        self._client = MessageQueueClient(server_url, zmq.Context.instance())
        try:
            self.chunk_size = int(get_lmcache_chunk_size(self._client, timeout=mq_timeout_s))
            if self.chunk_size <= 0:
                raise LMCacheRetrieveTransformError("LMCache returned an invalid chunk size")
            self.instance_id = (os.getpid() << 32) | (uuid.uuid4().int & 0xFFFFFFFF)
            payload = RegisterNonGpuContextPayload(
                instance_id=self.instance_id,
                model_name=source_model_name,
                world_size=source_world_size,
                block_size=self.chunk_size,
                num_layers=source.num_layers,
                hidden_dim_size=source.num_key_value_heads * source.head_dim,
                dtype_str=_torch_dtype_name(source.dtype),
                use_mla=False,
            )
            self._send_request(
                self._client,
                self._request_type.REGISTER_KV_CACHE_NON_GPU_CONTEXT,
                [payload],
            ).result(timeout=mq_timeout_s)
        except Exception:
            self._client.close()
            raise

    def store_prefix(
        self,
        *,
        request_id: str,
        token_ids: Sequence[int],
        source_kv: Any,
        cache_salt: str = "",
    ) -> LMCacheStoredSourcePrefix:
        import torch
        from lmcache.v1.multiprocess.custom_types import IPCCacheEngineKey

        tokens = tuple(token_ids)
        if (
            not request_id
            or not tokens
            or any(type(item) is not int or item < 0 for item in tokens)
            or not _valid_cache_salt(cache_salt)
        ):
            raise LMCacheRetrieveTransformError("LMCache source store metadata is invalid")
        expected_shape = (
            2,
            self.source.num_layers,
            self.source.num_key_value_heads,
            len(tokens),
            self.source.head_dim,
        )
        if (
            not isinstance(source_kv, torch.Tensor)
            or tuple(source_kv.shape) != expected_shape
            or source_kv.dtype != _torch_dtype(self.source.dtype)
            or len(tokens) % self.chunk_size
        ):
            raise LMCacheRetrieveTransformError("LMCache source KV tensor identity is invalid")
        ranges = tuple(
            (start, start + self.chunk_size) for start in range(0, len(tokens), self.chunk_size)
        )
        keys: list[str] = []
        checksums: list[str] = []
        with self._lock:
            if self._closed:
                raise LMCacheRetrieveTransformError("LMCache source writer is closed")
            try:
                for start, end in ranges:
                    explicit = (
                        source_kv[:, :, :, start:end, :].detach().to(device="cpu").contiguous()
                    )
                    checksum = source_chunk_checksums((explicit,))[0]
                    flat = (
                        explicit.permute(0, 1, 3, 2, 4)
                        .reshape(
                            2,
                            self.source.num_layers,
                            self.chunk_size,
                            self.source.num_key_value_heads * self.source.head_dim,
                        )
                        .contiguous()
                    )
                    key = IPCCacheEngineKey(
                        model_name=self.source_model_name,
                        world_size=self.source_world_size,
                        worker_id=self.source_worker_id,
                        token_ids=tokens,
                        start=start,
                        end=end,
                        request_id=request_id,
                        cache_salt=cache_salt,
                    )
                    self._send_request(
                        self._client,
                        self._request_type.PREPARE_STORE,
                        [key, self.instance_id],
                    ).result(timeout=self.mq_timeout_s)
                    encoded = pickle.dumps([flat])
                    if len(encoded) > flat.numel() * flat.element_size() + 1024 * 1024:
                        raise LMCacheRetrieveTransformError(
                            "LMCache source store payload exceeds its tensor bound"
                        )
                    committed = self._send_request(
                        self._client,
                        self._request_type.COMMIT_STORE,
                        [key, self.instance_id, encoded],
                    ).result(timeout=self.mq_timeout_s)
                    if committed is not True:
                        raise LMCacheRetrieveTransformError(
                            "LMCache source chunk could not be committed"
                        )
                    self.source_put_count += 1
                    keys.append(lmcache_source_key(request_id, start, end))
                    checksums.append(checksum)
            finally:
                self._send_request(
                    self._client,
                    self._request_type.END_SESSION,
                    [request_id],
                ).result(timeout=self.mq_timeout_s)
        return LMCacheStoredSourcePrefix(
            request_id=request_id,
            token_ids=tokens,
            source_ranges=ranges,
            source_keys=tuple(keys),
            source_checksums=tuple(checksums),
            source_model_name=self.source_model_name,
            source_world_size=self.source_world_size,
            source_worker_id=self.source_worker_id,
            cache_salt=cache_salt,
        )

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
        try:
            self._send_request(
                self._client,
                self._request_type.UNREGISTER_KV_CACHE,
                [self.instance_id],
            ).result(timeout=self.mq_timeout_s)
        finally:
            self._client.close()

    def __enter__(self) -> LMCacheMPSourceChunkWriter:
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()


class LMCacheMPSourceChunkReader:
    """Read source-model chunks through LMCache MP's pinned non-GPU protocol."""

    def __init__(
        self,
        *,
        server_url: str,
        source_model_name: str,
        source_world_size: int,
        source_worker_id: int,
        source: CachedKVModelSpec,
        mq_timeout_s: float = 5.0,
    ) -> None:
        if mq_timeout_s <= 0 or not math.isfinite(mq_timeout_s):
            raise LMCacheRetrieveTransformError("LMCache MQ timeout must be finite and positive")
        self.stack_identity = probe_runtime_stack()
        self.server_url = server_url
        self.source_model_name = source_model_name
        self.source_world_size = source_world_size
        self.source_worker_id = source_worker_id
        self.source = source
        self.mq_timeout_s = mq_timeout_s
        self._lock = threading.RLock()
        self._bound: LMCacheRetrieveTransformMetadata | None = None
        self._closed = False

        import zmq
        from lmcache.integration.vllm.vllm_multi_process_adapter import (
            get_lmcache_chunk_size,
            send_lmcache_request,
        )
        from lmcache.v1.multiprocess.custom_types import (
            RegisterNonGpuContextPayload,
        )
        from lmcache.v1.multiprocess.mq import (
            MessageQueueClient,
        )
        from lmcache.v1.multiprocess.protocol import RequestType

        self._send_request = send_lmcache_request
        self._request_type = RequestType
        self._client = MessageQueueClient(server_url, zmq.Context.instance())
        try:
            self.chunk_size = int(get_lmcache_chunk_size(self._client, timeout=mq_timeout_s))
            if self.chunk_size <= 0:
                raise LMCacheRetrieveTransformError("LMCache returned an invalid chunk size")
            self.instance_id = (os.getpid() << 32) | (uuid.uuid4().int & 0xFFFFFFFF)
            payload = RegisterNonGpuContextPayload(
                instance_id=self.instance_id,
                model_name=source_model_name,
                world_size=source_world_size,
                block_size=self.chunk_size,
                num_layers=source.num_layers,
                hidden_dim_size=source.num_key_value_heads * source.head_dim,
                dtype_str=_torch_dtype_name(source.dtype),
                use_mla=False,
            )
            self._send_request(
                self._client,
                self._request_type.REGISTER_KV_CACHE_NON_GPU_CONTEXT,
                [payload],
            ).result(timeout=mq_timeout_s)
        except Exception:
            self._client.close()
            raise

    def bind_request(self, metadata: LMCacheRetrieveTransformMetadata) -> None:
        errors = metadata.validate(chunk_size=self.chunk_size)
        if metadata.source_model_name != self.source_model_name:
            errors.append("LMCache metadata names another source model")
        if metadata.source_world_size != self.source_world_size:
            errors.append("LMCache metadata source world size changed")
        if metadata.source_worker_id != self.source_worker_id:
            errors.append("LMCache metadata source worker id changed")
        if errors:
            raise LMCacheRetrieveTransformError("; ".join(errors))
        with self._lock:
            if self._closed:
                raise LMCacheRetrieveTransformError("LMCache source reader is closed")
            if self._bound is not None:
                raise LMCacheRetrieveTransformError("LMCache source reader already has a request")
            self._bound = metadata

    def read_many_exact(self, keys: Sequence[str], *, timeout_s: float) -> Sequence[Any]:
        import torch
        from lmcache.v1.multiprocess.custom_types import IPCCacheEngineKey

        if timeout_s <= 0 or not math.isfinite(timeout_s):
            raise LMCacheRetrieveTransformError("LMCache source read timeout is invalid")
        with self._lock:
            metadata = self._bound
            if self._closed or metadata is None:
                raise LMCacheRetrieveTransformError("LMCache source reader has no bound request")
            if tuple(keys) != metadata.request.source_keys:
                raise LMCacheRetrieveTransformError("LMCache source read keys changed")
            deadline = time.monotonic() + min(timeout_s, self.mq_timeout_s)
            first_start = metadata.source_ranges[0][0]
            final_end = metadata.source_ranges[-1][1]
            lookup_key = IPCCacheEngineKey(
                model_name=self.source_model_name,
                world_size=self.source_world_size,
                worker_id=None,
                token_ids=metadata.token_ids,
                start=first_start,
                end=final_end,
                request_id=metadata.request.request_id,
                cache_salt=metadata.cache_salt,
            )
            self._send_request(
                self._client,
                self._request_type.LOOKUP,
                [lookup_key, 1],
            ).result(timeout=max(0.001, deadline - time.monotonic()))
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise LMCacheRetrieveTransformError("LMCache source lookup timed out")
                matched_chunks = self._send_request(
                    self._client,
                    self._request_type.QUERY_PREFETCH_STATUS,
                    [metadata.request.request_id],
                ).result(timeout=remaining)
                if matched_chunks is not None:
                    break
                time.sleep(min(0.001, max(0.0, deadline - time.monotonic())))
            if type(matched_chunks) is not int or matched_chunks != len(metadata.source_ranges):
                raise LMCacheRetrieveTransformError("LMCache source prefix lookup was incomplete")
            chunks: list[Any] = []
            for key_text, (start, end) in zip(keys, metadata.source_ranges, strict=True):
                if key_text != lmcache_source_key(metadata.request.request_id, start, end):
                    raise LMCacheRetrieveTransformError("LMCache source read key is not canonical")
                key = IPCCacheEngineKey(
                    model_name=self.source_model_name,
                    world_size=self.source_world_size,
                    worker_id=self.source_worker_id,
                    token_ids=metadata.token_ids,
                    start=start,
                    end=end,
                    request_id=metadata.request.request_id,
                    cache_salt=metadata.cache_salt,
                )
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise LMCacheRetrieveTransformError("LMCache source read timed out")
                response = self._send_request(
                    self._client,
                    self._request_type.PREPARE_RETRIEVE,
                    [key, self.instance_id],
                ).result(timeout=remaining)
                if not response.success:
                    raise LMCacheRetrieveTransformError("LMCache source chunk was not available")
                element_size = torch.empty((), dtype=_torch_dtype(self.source.dtype)).element_size()
                expected_tensor_bytes = (
                    2
                    * self.source.num_layers
                    * (end - start)
                    * self.source.num_key_value_heads
                    * self.source.head_dim
                    * element_size
                )
                if len(response.data) > expected_tensor_bytes + 1024 * 1024:
                    raise LMCacheRetrieveTransformError(
                        "LMCache source response exceeds the bounded tensor payload"
                    )
                decoded = pickle.loads(response.data)
                if not isinstance(decoded, list) or len(decoded) != 1:
                    raise LMCacheRetrieveTransformError(
                        "LMCache source retrieve returned an unexpected chunk set"
                    )
                chunk = decoded[0]
                expected_flat = (
                    2,
                    self.source.num_layers,
                    end - start,
                    self.source.num_key_value_heads * self.source.head_dim,
                )
                if (
                    not isinstance(chunk, torch.Tensor)
                    or tuple(chunk.shape) != expected_flat
                    or chunk.dtype != _torch_dtype(self.source.dtype)
                    or chunk.device.type != "cpu"
                ):
                    raise LMCacheRetrieveTransformError(
                        "LMCache source chunk tensor identity changed"
                    )
                shaped = (
                    chunk.reshape(
                        2,
                        self.source.num_layers,
                        end - start,
                        self.source.num_key_value_heads,
                        self.source.head_dim,
                    )
                    .permute(0, 1, 3, 2, 4)
                    .contiguous()
                )
                committed = self._send_request(
                    self._client,
                    self._request_type.COMMIT_RETRIEVE,
                    [key, self.instance_id],
                ).result(timeout=max(0.001, deadline - time.monotonic()))
                if committed is not True:
                    raise LMCacheRetrieveTransformError(
                        "LMCache source retrieve could not be committed"
                    )
                chunks.append(shaped)
            return tuple(chunks)

    def finish_request(self, request_id: str) -> None:
        with self._lock:
            if self._bound is None or self._bound.request.request_id != request_id:
                raise LMCacheRetrieveTransformError("LMCache source request binding changed")
            self._bound = None
        self._send_request(
            self._client,
            self._request_type.END_SESSION,
            [request_id],
        ).result(timeout=self.mq_timeout_s)

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            self._bound = None
        try:
            self._send_request(
                self._client,
                self._request_type.UNREGISTER_KV_CACHE,
                [self.instance_id],
            ).result(timeout=self.mq_timeout_s)
        finally:
            self._client.close()

    def __enter__(self) -> LMCacheMPSourceChunkReader:
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()


class LMCacheRetrieveTransformBridge:
    """Worker bridge that preserves upstream metadata and adds direct retrieval."""

    def __init__(
        self,
        *,
        upstream_connector: Any,
        injector: DirectPagedKVInjector,
        source_reader: LMCacheMPSourceChunkReader,
        validity_tracker: RuntimeBlockValidityTracker,
    ) -> None:
        self.stack_identity = probe_runtime_stack()
        self.upstream_connector = upstream_connector
        self.injector = injector
        self.source_reader = source_reader
        self.validity_tracker = validity_tracker
        self._kv_caches: tuple[Any, ...] = ()
        self._observations: list[LMCacheBridgeObservation] = []
        self._load_complete: set[str] = set()
        self._lock = threading.RLock()

    @property
    def observations(self) -> tuple[LMCacheBridgeObservation, ...]:
        with self._lock:
            return tuple(self._observations)

    def register_kv_caches(self, kv_caches: Mapping[str, Any]) -> None:
        if not kv_caches:
            raise LMCacheRetrieveTransformError("vLLM did not register paged KV caches")
        values = tuple(kv_caches.values())
        if len(values) != self.injector.transport.target.num_layers:
            raise LMCacheRetrieveTransformError("vLLM paged KV layer count changed")
        self.upstream_connector.register_kv_caches(dict(kv_caches))
        self._kv_caches = values

    def publish_load_complete(self, request_id: str, _block_ids: tuple[int, ...]) -> None:
        with self._lock:
            if request_id in self._load_complete:
                raise LMCacheRetrieveTransformError("load-complete was published twice")
            self._load_complete.add(request_id)

    def start_load_kv(
        self,
        forward_context: Any,
        metadata: LMCacheRetrieveTransformBatch,
    ) -> tuple[LMCacheBridgeObservation, ...]:
        from lmcache.integration.vllm.lmcache_mp_connector import (  # type: ignore[import-untyped]
            LMCacheMPConnectorMetadata,
        )

        if not self._kv_caches:
            raise LMCacheRetrieveTransformError("vLLM paged KV caches are not registered")
        standard_metadata = metadata.standard_metadata or LMCacheMPConnectorMetadata()
        if not isinstance(standard_metadata, LMCacheMPConnectorMetadata):
            raise LMCacheRetrieveTransformError("standard LMCache metadata type changed")
        self.upstream_connector.bind_connector_metadata(standard_metadata)
        self.upstream_connector.start_load_kv(forward_context)
        current: list[LMCacheBridgeObservation] = []
        for item in metadata.requests:
            errors = item.validate(chunk_size=self.source_reader.chunk_size)
            if errors:
                raise LMCacheRetrieveTransformError("; ".join(errors))
            self.source_reader.bind_request(item)
            try:
                result = self.injector.retrieve_transform(
                    item.request,
                    kv_caches=self._kv_caches,
                )
            finally:
                self.source_reader.finish_request(item.request.request_id)
            invalid = tuple(sorted(result.invalidated_blocks))
            observation = LMCacheBridgeObservation(
                request_id=item.request.request_id,
                result=result,
                load_complete_published=item.request.request_id in self._load_complete,
                invalid_blocks_reported=invalid,
            )
            if result.success != observation.load_complete_published:
                raise LMCacheRetrieveTransformError(
                    "LMCache bridge load-complete publication is inconsistent"
                )
            current.append(observation)
        with self._lock:
            self._observations.extend(current)
        return tuple(current)

    def wait_for_save(self) -> None:
        self.upstream_connector.wait_for_save()

    def get_finished(
        self,
        finished_req_ids: set[str],
    ) -> tuple[set[str] | None, set[str] | None]:
        return self.upstream_connector.get_finished(finished_req_ids)

    def get_block_ids_with_load_errors(self) -> set[int]:
        upstream = set(self.upstream_connector.get_block_ids_with_load_errors())
        return upstream | self.validity_tracker.drain_invalid()

    def shutdown(self) -> None:
        try:
            self.source_reader.close()
        finally:
            self.upstream_connector.shutdown()


def probe_runtime_stack() -> RuntimeStackIdentity:
    """Fail closed unless the installed LMCache/vLLM APIs match the audited bridge."""

    versions = {name: importlib.metadata.version(name) for name in ("lmcache", "vllm", "torch")}
    if versions["lmcache"] != EXPECTED_LMCACHE_VERSION:
        raise LMCacheRetrieveTransformError("runtime bridge requires LMCache 0.4.6")
    if versions["vllm"] != EXPECTED_VLLM_VERSION:
        raise LMCacheRetrieveTransformError("runtime bridge requires vLLM 0.24.0")
    if not versions["torch"].startswith(EXPECTED_TORCH_VERSION_PREFIX):
        raise LMCacheRetrieveTransformError("runtime bridge requires Torch 2.11.0")

    from lmcache.integration.vllm.lmcache_mp_connector import (
        LMCacheMPConnector,
        LMCacheMPConnectorMetadata,
    )
    from lmcache.integration.vllm.vllm_multi_process_adapter import (
        LMCacheMPWorkerAdapter,
    )
    from lmcache.v1.multiprocess.protocol import RequestType
    from vllm.config.kv_transfer import KVTransferConfig
    from vllm.v1.core.sched.scheduler import Scheduler

    required_protocols = {
        "REGISTER_KV_CACHE_NON_GPU_CONTEXT",
        "PREPARE_STORE",
        "COMMIT_STORE",
        "LOOKUP",
        "QUERY_PREFETCH_STATUS",
        "PREPARE_RETRIEVE",
        "COMMIT_RETRIEVE",
        "END_SESSION",
        "UNREGISTER_KV_CACHE",
    }
    if not required_protocols <= set(RequestType.__members__):
        raise LMCacheRetrieveTransformError("LMCache non-GPU retrieve protocol is incomplete")
    for owner, methods in (
        (
            LMCacheMPConnector,
            (
                "register_kv_caches",
                "start_load_kv",
                "wait_for_save",
                "get_finished",
                "get_block_ids_with_load_errors",
                "shutdown",
            ),
        ),
        (
            LMCacheMPWorkerAdapter,
            (
                "register_kv_caches",
                "get_block_ids_with_load_errors",
                "shutdown",
            ),
        ),
    ):
        if any(not callable(getattr(owner, name, None)) for name in methods):
            raise LMCacheRetrieveTransformError("LMCache MP worker API changed")
    start_parameters = inspect.signature(LMCacheMPConnector.start_load_kv).parameters
    if "forward_context" not in start_parameters:
        raise LMCacheRetrieveTransformError("LMCache start_load_kv signature changed")
    if "kv_connector_module_path" not in KVTransferConfig.__dataclass_fields__:
        raise LMCacheRetrieveTransformError("vLLM external connector loading API changed")
    scheduler_source = inspect.getsource(Scheduler._handle_invalid_blocks)
    if not all(
        marker in scheduler_source
        for marker in ("invalid_block_ids", "recompute_kv_load_failures", "rescheduled")
    ):
        raise LMCacheRetrieveTransformError("vLLM native recompute contract changed")
    if not issubclass(LMCacheMPConnectorMetadata, object):
        raise LMCacheRetrieveTransformError("LMCache connector metadata API changed")

    distributions = {
        "lmcache": Path(str(importlib.metadata.distribution("lmcache").locate_file(""))),
        "vllm": Path(str(importlib.metadata.distribution("vllm").locate_file(""))),
    }
    sources: list[RuntimeSourceIdentity] = []
    for module_name in _PINNED_RUNTIME_MODULES:
        spec = importlib.util.find_spec(module_name)
        if spec is None or spec.origin is None:
            raise LMCacheRetrieveTransformError(f"runtime module {module_name!r} is unavailable")
        path = Path(spec.origin).resolve()
        distribution = distributions["lmcache" if module_name.startswith("lmcache.") else "vllm"]
        try:
            relative = path.relative_to(distribution.resolve()).as_posix()
        except ValueError as exc:
            raise LMCacheRetrieveTransformError(
                "runtime module is outside its installed distribution"
            ) from exc
        sources.append(RuntimeSourceIdentity(module_name, relative, sha256_file(path)))
    import torch

    identity = RuntimeStackIdentity(
        lmcache_version=versions["lmcache"],
        vllm_version=versions["vllm"],
        torch_version=versions["torch"],
        cuda_version=torch.version.cuda,
        sources=tuple(sources),
    )
    errors = identity.validate()
    if errors:
        raise LMCacheRetrieveTransformError("; ".join(errors))
    return identity


def verify_runtime_stack_identity(expected: RuntimeStackIdentity) -> None:
    observed = probe_runtime_stack()
    if observed != expected or observed.content_sha256() != expected.content_sha256():
        raise LMCacheRetrieveTransformError("runtime stack identity changed after measurement")


def lmcache_source_key(request_id: str, start: int, end: int) -> str:
    if not request_id or start < 0 or end <= start:
        raise LMCacheRetrieveTransformError("LMCache source key fields are invalid")
    return f"{request_id}:{start}:{end}"


def source_chunk_checksums(chunks: Sequence[Any]) -> tuple[str, ...]:
    import torch

    result: list[str] = []
    for chunk in chunks:
        if not isinstance(chunk, torch.Tensor) or chunk.device.type != "cpu":
            raise LMCacheRetrieveTransformError("source checksum requires CPU tensors")
        payload = bytes(chunk.detach().contiguous().view(torch.uint8).numpy())
        result.append(_sha256_bytes(payload))
    return tuple(result)


def _torch_dtype(name: str) -> Any:
    import torch

    try:
        return {"bfloat16": torch.bfloat16, "float16": torch.float16}[name]
    except KeyError as exc:
        raise LMCacheRetrieveTransformError(f"unsupported source dtype {name!r}") from exc


def _torch_dtype_name(name: str) -> str:
    if name not in {"bfloat16", "float16"}:
        raise LMCacheRetrieveTransformError(f"unsupported source dtype {name!r}")
    return name


def _valid_cache_salt(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) <= 128
        and not any(character in value for character in "@/\\\x00")
    )


def _canonical_json_bytes(value: Any) -> bytes:
    try:
        return (
            json.dumps(
                value,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise LMCacheRetrieveTransformError("runtime identity is not canonical JSON") from exc


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _is_sha256(value: Any) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True
