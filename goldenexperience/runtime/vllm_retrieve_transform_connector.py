"""External vLLM connector for audited cross-model RETRIEVE_TRANSFORM."""

from __future__ import annotations

import base64
import math
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1,
    KVConnectorMetadata,
    KVConnectorRole,
)
from vllm.v1.attention.backend import (
    AttentionMetadata,
)

from goldenexperience.runtime.cross_model_reuse import token_ids_sha256
from goldenexperience.runtime.direct_paged_kv import (
    DirectInjectionResult,
    DirectPagedKVInjector,
    infer_block_size,
    scatter_paged_kv,
)
from goldenexperience.runtime.lmcache_retrieve_transform import (
    LMCacheMPSourceChunkReader,
    LMCacheRetrieveTransformMetadata,
    LMCacheStoredSourcePrefix,
    RuntimeBlockValidityTracker,
    lmcache_source_key,
)
from goldenexperience.runtime.runtime_audit_telemetry import (
    RuntimeAuditTelemetryEmitter,
)
from goldenexperience.size_variant.cached_kv_manifest import sha256_file
from goldenexperience.size_variant.head_aware_transport import HeadAwareKVTransport
from goldenexperience.size_variant.risk_gate import (
    AdmissionDecision,
    CalibratedRiskGate,
    RiskPredictor,
    SourceKVSidecar,
)
from goldenexperience.size_variant.selective_manifest import (
    ArtifactState,
    SelectiveKVBridgeManifest,
)

if TYPE_CHECKING:
    from vllm.forward_context import ForwardContext
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.core.sched.output import SchedulerOutput
    from vllm.v1.kv_cache_interface import KVCacheConfig
    from vllm.v1.request import Request

V5_VLLM_CONNECTOR_SCHEMA = "goldenexperience.v5_vllm_retrieve_transform.v1"
V5_VLLM_CONNECTOR_PARAMS_KEY = "goldenexperience_v5_retrieve_transform"
_LAYER_PATTERN = re.compile(r"(?:^|\.)layers\.(\d+)(?:\.|$)")
_REQUEST_FIELDS = frozenset(
    {
        "schema_version",
        "semantic_artifact_id",
        "audit_request_id",
        "source_request_id",
        "source_model_name",
        "source_world_size",
        "source_worker_id",
        "cache_salt",
        "token_ids",
        "token_ids_sha256",
        "prefix_hash",
        "sidecar_base64",
        "chunks",
        "expected_accepted",
        "expected_decision",
        "inject_partial_failure",
        "retrieve_timeout_s",
    }
)
_CHUNK_FIELDS = frozenset({"start", "end", "checksum"})


class V5VLLMConnectorError(RuntimeError):
    """Raised when the external vLLM connector cannot fail closed."""


@dataclass(frozen=True)
class V5ConnectorLoadRequest:
    target_request_id: str
    audit_request_id: str
    metadata: LMCacheRetrieveTransformMetadata
    expected_decision: str
    inject_partial_failure: bool


@dataclass(frozen=True)
class V5ConnectorMetadata(KVConnectorMetadata):
    requests: tuple[V5ConnectorLoadRequest, ...]


@dataclass(frozen=True)
class _ParsedRequest:
    audit_request_id: str
    stored: LMCacheStoredSourcePrefix
    prefix_hash: str
    sidecar_payload: bytes
    expected_accepted: bool
    expected_decision: str
    inject_partial_failure: bool
    retrieve_timeout_s: float

    @property
    def prefix_tokens(self) -> int:
        return len(self.stored.token_ids)


@dataclass(frozen=True)
class _AuditBinding:
    manifest: SelectiveKVBridgeManifest
    gate: CalibratedRiskGate
    transport: HeadAwareKVTransport | None


def runtime_source_model_name(manifest: SelectiveKVBridgeManifest) -> str:
    return f"golden-v5-source-{manifest.source.weights_sha256}"


def build_vllm_retrieve_transform_params(
    *,
    manifest: SelectiveKVBridgeManifest,
    stored: LMCacheStoredSourcePrefix,
    sidecar_payload: bytes,
    audit_request_id: str,
    accepted: bool,
    decision: str,
    inject_partial_failure: bool = False,
    retrieve_timeout_s: float = 120.0,
) -> dict[str, Any]:
    if manifest.state is not ArtifactState.SEMANTIC_APPROVED:
        raise V5VLLMConnectorError("runtime connector parameters require semantic approval")
    if stored.source_model_name != runtime_source_model_name(manifest):
        raise V5VLLMConnectorError("runtime connector source model identity changed")
    try:
        sidecar = SourceKVSidecar.from_bytes(sidecar_payload)
    except Exception as exc:
        raise V5VLLMConnectorError("runtime connector sidecar is invalid") from exc
    if (
        not isinstance(audit_request_id, str)
        or not audit_request_id
        or type(accepted) is not bool
        or not isinstance(decision, str)
        or not decision
        or type(inject_partial_failure) is not bool
        or sidecar.prefix_hash != sidecar.prefix_hash.lower()
        or sidecar.prefix_length != len(stored.token_ids)
        or not _finite_positive(retrieve_timeout_s)
    ):
        raise V5VLLMConnectorError("runtime connector request binding is invalid")
    try:
        stored.build_retrieve_metadata(
            slot_mapping=tuple(range(len(stored.token_ids))),
            prefix_hash=sidecar.prefix_hash,
            sidecar=sidecar_payload,
            timeout_s=retrieve_timeout_s,
        )
    except Exception as exc:
        raise V5VLLMConnectorError("stored source prefix binding is invalid") from exc
    chunks = [
        {
            "start": start,
            "end": end,
            "checksum": checksum,
        }
        for (start, end), checksum in zip(
            stored.source_ranges,
            stored.source_checksums,
            strict=True,
        )
    ]
    return {
        V5_VLLM_CONNECTOR_PARAMS_KEY: {
            "schema_version": V5_VLLM_CONNECTOR_SCHEMA,
            "semantic_artifact_id": manifest.artifact_id,
            "audit_request_id": audit_request_id,
            "source_request_id": stored.request_id,
            "source_model_name": stored.source_model_name,
            "source_world_size": stored.source_world_size,
            "source_worker_id": stored.source_worker_id,
            "cache_salt": stored.cache_salt,
            "token_ids": list(stored.token_ids),
            "token_ids_sha256": token_ids_sha256(list(stored.token_ids)),
            "prefix_hash": sidecar.prefix_hash,
            "sidecar_base64": base64.b64encode(sidecar_payload).decode("ascii"),
            "chunks": chunks,
            "expected_accepted": accepted,
            "expected_decision": decision,
            "inject_partial_failure": inject_partial_failure,
            "retrieve_timeout_s": retrieve_timeout_s,
        }
    }


class V5RetrieveTransformScheduler:
    def __init__(
        self,
        *,
        manifest: SelectiveKVBridgeManifest,
        gate: CalibratedRiskGate,
        block_size: int,
        emitter: RuntimeAuditTelemetryEmitter,
    ) -> None:
        self.manifest = manifest
        self.gate = gate
        self.block_size = block_size
        self.emitter = emitter
        self._eligible: dict[str, _ParsedRequest] = {}
        self._pending: dict[str, V5ConnectorLoadRequest] = {}
        self._claimed: set[str] = set()

    def get_num_new_matched_tokens(
        self,
        request: Request,
        num_computed_tokens: int,
    ) -> tuple[int, bool]:
        if request.request_id in self._claimed:
            return 0, False
        cached = self._eligible.get(request.request_id)
        if cached is not None:
            return self._external_tokens(request, cached, num_computed_tokens), True
        parsed = _parse_request(request, self.manifest)
        if parsed is None:
            return 0, False
        admission = self.gate.evaluate(parsed.sidecar_payload)
        binding_matches = (
            admission.accepted == parsed.expected_accepted
            and admission.reason == parsed.expected_decision
        )
        self.emitter.emit(
            request_id=parsed.audit_request_id,
            kind="gate",
            evidence={
                "accepted": admission.accepted,
                "decision": admission.reason,
                "unsafe_probability": admission.unsafe_probability,
                "binding_matches": binding_matches,
                "source_read_attempted": False,
            },
        )
        if not admission.accepted or not binding_matches:
            self._claimed.add(request.request_id)
            return 0, False
        self._eligible[request.request_id] = parsed
        return self._external_tokens(request, parsed, num_computed_tokens), True

    def _external_tokens(
        self,
        request: Request,
        parsed: _ParsedRequest,
        num_computed_tokens: int,
    ) -> int:
        prompt_tokens = tuple(request.prompt_token_ids or ())
        if (
            parsed.prefix_tokens % self.block_size
            or parsed.prefix_tokens >= len(prompt_tokens)
            or prompt_tokens[: parsed.prefix_tokens] != parsed.stored.token_ids
            or num_computed_tokens != 0
        ):
            raise V5VLLMConnectorError("target prompt differs from the source prefix binding")
        external_tokens = parsed.prefix_tokens - num_computed_tokens
        if external_tokens <= 0:
            self._claimed.add(request.request_id)
            self._eligible.pop(request.request_id, None)
            return 0
        return external_tokens

    def update_state_after_alloc(
        self,
        request: Request,
        blocks: KVCacheBlocks,
        num_external_tokens: int,
    ) -> None:
        if num_external_tokens <= 0:
            return
        parsed = self._eligible.pop(request.request_id, None)
        if parsed is None or num_external_tokens != parsed.prefix_tokens:
            raise V5VLLMConnectorError("vLLM allocated an unexpected external prefix")
        groups = blocks.get_block_ids()
        if not isinstance(groups, tuple) or len(groups) != 1:
            raise V5VLLMConnectorError("runtime connector requires one KV cache group")
        required_blocks = parsed.prefix_tokens // self.block_size
        block_ids = tuple(groups[0][:required_blocks])
        if len(block_ids) != required_blocks or len(set(block_ids)) != len(block_ids):
            raise V5VLLMConnectorError("vLLM external prefix block allocation is invalid")
        slot_mapping = tuple(
            block_id * self.block_size + offset
            for block_id in block_ids
            for offset in range(self.block_size)
        )
        metadata = parsed.stored.build_retrieve_metadata(
            slot_mapping=slot_mapping,
            prefix_hash=parsed.prefix_hash,
            sidecar=parsed.sidecar_payload,
            timeout_s=parsed.retrieve_timeout_s,
        )
        self._pending[request.request_id] = V5ConnectorLoadRequest(
            target_request_id=request.request_id,
            audit_request_id=parsed.audit_request_id,
            metadata=metadata,
            expected_decision=parsed.expected_decision,
            inject_partial_failure=parsed.inject_partial_failure,
        )
        self._claimed.add(request.request_id)

    def build_connector_meta(self, _scheduler_output: SchedulerOutput) -> V5ConnectorMetadata:
        metadata = V5ConnectorMetadata(tuple(self._pending.values()))
        self._pending.clear()
        return metadata

    def request_finished(self, request_id: str) -> None:
        self._eligible.pop(request_id, None)
        self._pending.pop(request_id, None)
        self._claimed.discard(request_id)


class V5RetrieveTransformWorker:
    def __init__(
        self,
        *,
        manifest: SelectiveKVBridgeManifest,
        gate: CalibratedRiskGate,
        transport: HeadAwareKVTransport,
        reader: LMCacheMPSourceChunkReader,
        emitter: RuntimeAuditTelemetryEmitter,
    ) -> None:
        self.manifest = manifest
        self.reader = reader
        self.emitter = emitter
        self.validity = RuntimeBlockValidityTracker()
        self._kv_caches: tuple[Any, ...] = ()
        self._load_complete: set[str] = set()
        self._finished_recving: set[str] = set()
        self._active_request: tuple[str, str] | None = None
        self._inject_partial_failure = False
        self._partial_failure_count = 0
        self.injector = DirectPagedKVInjector(
            risk_gate=gate,
            transport=transport,
            source_reader=reader,
            validity_tracker=self.validity,
            publish_load_complete=self._publish_load_complete,
            scatter=self._scatter,
        )

    def register_kv_caches(self, kv_caches: dict[str, Any]) -> None:
        ordered = _ordered_layer_caches(kv_caches, self.manifest.target.num_layers)
        self._kv_caches = tuple(ordered)

    def start_load_kv(self, metadata: V5ConnectorMetadata) -> None:
        if not self._kv_caches:
            raise V5VLLMConnectorError("vLLM paged KV caches are not registered")
        for item in metadata.requests:
            started = time.perf_counter()
            block_size = infer_block_size(
                self._kv_caches,
                self.manifest.target.num_key_value_heads,
                self.manifest.target.head_dim,
            )
            request_blocks = tuple(
                sorted({slot // block_size for slot in item.metadata.request.slot_mapping})
            )
            partial_failure_count = self._partial_failure_count
            self._inject_partial_failure = item.inject_partial_failure
            self._active_request = (
                item.metadata.request.request_id,
                item.target_request_id,
            )
            bound = False
            result: DirectInjectionResult | None = None
            worker_error: Exception | None = None
            try:
                self.reader.bind_request(item.metadata)
                bound = True
                result = self.injector.retrieve_transform(
                    item.metadata.request,
                    kv_caches=self._kv_caches,
                )
            except Exception as exc:
                worker_error = exc
            finally:
                if bound:
                    try:
                        self.reader.finish_request(item.metadata.request.request_id)
                    except Exception as exc:
                        worker_error = worker_error or exc
                self._inject_partial_failure = False
                self._active_request = None
            if worker_error is not None:
                previous = result
                result = DirectInjectionResult(
                    success=False,
                    accepted=previous.accepted if previous is not None else True,
                    fallback_reason="connector_worker_failure",
                    admission=(
                        previous.admission
                        if previous is not None
                        else AdmissionDecision(True, item.expected_decision)
                    ),
                    source_read_attempted=(
                        previous.source_read_attempted if previous is not None else bound
                    ),
                    source_chunks_read=(previous.source_chunks_read if previous is not None else 0),
                    tokens_scattered=previous.tokens_scattered if previous is not None else 0,
                    invalidated_blocks=request_blocks,
                    target_mooncake_puts=0,
                    elapsed_ms=(time.perf_counter() - started) * 1000,
                    error=repr(worker_error),
                )
            if result is None:
                raise V5VLLMConnectorError("runtime worker produced no injection result")
            worker_binding_matches = (
                result.admission.accepted and result.admission.reason == item.expected_decision
            )
            if result.success and not worker_binding_matches:
                result = DirectInjectionResult(
                    success=False,
                    accepted=result.accepted,
                    fallback_reason="worker_gate_binding_mismatch",
                    admission=result.admission,
                    source_read_attempted=result.source_read_attempted,
                    source_chunks_read=result.source_chunks_read,
                    tokens_scattered=result.tokens_scattered,
                    invalidated_blocks=request_blocks,
                    target_mooncake_puts=0,
                    elapsed_ms=(time.perf_counter() - started) * 1000,
                    error="worker gate decision differs from the scheduler binding",
                )
            if not result.success:
                self.validity.mark_invalid(request_blocks)
            load_complete = item.target_request_id in self._load_complete
            self.emitter.emit(
                request_id=item.audit_request_id,
                kind="execution",
                evidence={
                    "success": result.success,
                    "accepted": result.accepted,
                    "fallback_reason": result.fallback_reason,
                    "source_read_attempted": result.source_read_attempted,
                    "source_chunks_read": result.source_chunks_read,
                    "tokens_scattered": result.tokens_scattered,
                    "invalidated_blocks": list(result.invalidated_blocks),
                    "request_blocks": list(request_blocks),
                    "target_mooncake_puts": result.target_mooncake_puts,
                    "materialization_ms": result.elapsed_ms,
                    "load_complete_published": load_complete,
                    "worker_binding_matches": worker_binding_matches,
                    "partial_failure_injected": item.inject_partial_failure,
                    "partial_failure_count": (self._partial_failure_count - partial_failure_count),
                    "registered_layer_count": len(self._kv_caches),
                },
            )
            self._finished_recving.add(item.target_request_id)

    def drain_finished_recving(self) -> set[str]:
        finished = set(self._finished_recving)
        self._finished_recving.clear()
        return finished

    def get_block_ids_with_load_errors(self) -> set[int]:
        return self.validity.drain_invalid()

    def close(self) -> None:
        try:
            self.reader.close()
        finally:
            self.emitter.close()

    def _publish_load_complete(self, request_id: str, _block_ids: tuple[int, ...]) -> None:
        if self._active_request is None or self._active_request[0] != request_id:
            raise V5VLLMConnectorError("load-complete source request binding changed")
        target_request_id = self._active_request[1]
        if target_request_id in self._load_complete:
            raise V5VLLMConnectorError("load-complete was published twice")
        self._load_complete.add(target_request_id)

    def _scatter(
        self,
        target_kv: Any,
        kv_caches: Any,
        slot_mapping: Any,
    ) -> tuple[int, ...]:
        if self._inject_partial_failure:
            scatter_paged_kv(target_kv[:, :1], tuple(kv_caches)[:1], slot_mapping)
            self._partial_failure_count += 1
            raise V5VLLMConnectorError("injected failure after the first target layer")
        return scatter_paged_kv(target_kv, kv_caches, slot_mapping)


class V5RetrieveTransformConnector(KVConnectorBase_V1):
    """vLLM external connector used only by the sealed v5 runtime audit."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        role: KVConnectorRole,
        kv_cache_config: KVCacheConfig,
    ) -> None:
        super().__init__(vllm_config, role, kv_cache_config)
        config = _ConnectorConfig.from_vllm(vllm_config)
        load_transport = role == KVConnectorRole.WORKER
        device = _worker_device() if load_transport else "cpu"
        binding = _load_audit_binding(config, device=device, load_transport=load_transport)
        emitter = RuntimeAuditTelemetryEmitter(
            host=config.telemetry_host,
            port=config.telemetry_port,
            nonce=config.telemetry_nonce,
            secret_hex=config.telemetry_secret_hex,
        )
        self.scheduler: V5RetrieveTransformScheduler | None = None
        self.worker: V5RetrieveTransformWorker | None = None
        if role == KVConnectorRole.SCHEDULER:
            self.scheduler = V5RetrieveTransformScheduler(
                manifest=binding.manifest,
                gate=binding.gate,
                block_size=vllm_config.cache_config.block_size,
                emitter=emitter,
            )
        elif role == KVConnectorRole.WORKER:
            if binding.transport is None:
                raise V5VLLMConnectorError("runtime worker transport was not loaded")
            reader = LMCacheMPSourceChunkReader(
                server_url=config.lmcache_server_url,
                source_model_name=runtime_source_model_name(binding.manifest),
                source_world_size=1,
                source_worker_id=0,
                source=binding.manifest.source,
                mq_timeout_s=config.mq_timeout_s,
            )
            self.worker = V5RetrieveTransformWorker(
                manifest=binding.manifest,
                gate=binding.gate,
                transport=binding.transport,
                reader=reader,
                emitter=emitter,
            )

    def register_kv_caches(self, kv_caches: dict[str, Any]) -> None:
        if self.worker is None:
            raise V5VLLMConnectorError("scheduler cannot register paged KV caches")
        self.worker.register_kv_caches(kv_caches)

    def start_load_kv(self, forward_context: ForwardContext, **kwargs: Any) -> None:
        del forward_context, kwargs
        if self.worker is None:
            raise V5VLLMConnectorError("scheduler cannot load paged KV caches")
        metadata = self._get_connector_metadata()
        if not isinstance(metadata, V5ConnectorMetadata):
            raise V5VLLMConnectorError("vLLM connector metadata type changed")
        self.worker.start_load_kv(metadata)

    def wait_for_layer_load(self, layer_name: str) -> None:
        del layer_name
        return

    def save_kv_layer(
        self,
        layer_name: str,
        kv_layer: torch.Tensor,
        attn_metadata: AttentionMetadata,
        **kwargs: Any,
    ) -> None:
        del layer_name, kv_layer, attn_metadata, kwargs
        return

    def wait_for_save(self) -> None:
        return

    def get_block_ids_with_load_errors(self) -> set[int]:
        return self.worker.get_block_ids_with_load_errors() if self.worker is not None else set()

    def get_finished(self, _finished_req_ids: set[str]) -> tuple[set[str] | None, set[str] | None]:
        if self.worker is None:
            return set(), set()
        return set(), self.worker.drain_finished_recving()

    def get_num_new_matched_tokens(
        self,
        request: Request,
        num_computed_tokens: int,
    ) -> tuple[int, bool]:
        if self.scheduler is None:
            raise V5VLLMConnectorError("worker cannot schedule external KV")
        return self.scheduler.get_num_new_matched_tokens(request, num_computed_tokens)

    def update_state_after_alloc(
        self,
        request: Request,
        blocks: KVCacheBlocks,
        num_external_tokens: int,
    ) -> None:
        if self.scheduler is None:
            raise V5VLLMConnectorError("worker cannot update scheduler state")
        self.scheduler.update_state_after_alloc(request, blocks, num_external_tokens)

    def build_connector_meta(self, scheduler_output: SchedulerOutput) -> V5ConnectorMetadata:
        if self.scheduler is None:
            raise V5VLLMConnectorError("worker cannot build scheduler metadata")
        return self.scheduler.build_connector_meta(scheduler_output)

    def request_finished(
        self,
        request: Request,
        _block_ids: list[int],
    ) -> tuple[bool, dict[str, Any] | None]:
        if self.scheduler is not None:
            self.scheduler.request_finished(request.request_id)
        return False, None

    def shutdown(self) -> None:
        if self.worker is not None:
            self.worker.close()
        elif self.scheduler is not None:
            self.scheduler.emitter.close()


@dataclass(frozen=True)
class _ConnectorConfig:
    semantic_manifest_path: Path
    semantic_manifest_sha256: str
    semantic_artifact_id: str
    workspace_root: Path
    direction: str
    lmcache_server_url: str
    telemetry_host: str
    telemetry_port: int
    telemetry_nonce: str
    telemetry_secret_hex: str
    mq_timeout_s: float

    @classmethod
    def from_vllm(cls, vllm_config: VllmConfig) -> _ConnectorConfig:
        transfer = vllm_config.kv_transfer_config
        if transfer is None or transfer.kv_load_failure_policy != "recompute":
            raise V5VLLMConnectorError("runtime connector requires native recompute policy")

        def required(name: str) -> Any:
            value = transfer.get_from_extra_config(name, None)
            if value is None:
                raise V5VLLMConnectorError(f"runtime connector config {name!r} is required")
            return value

        semantic_manifest_path = required("golden.semantic_manifest_path")
        workspace_root = required("golden.workspace_root")
        semantic_manifest_sha256 = required("golden.semantic_manifest_sha256")
        semantic_artifact_id = required("golden.semantic_artifact_id")
        direction = required("golden.direction")
        lmcache_server_url = required("golden.lmcache_server_url")
        telemetry_host = required("golden.telemetry_host")
        telemetry_port = required("golden.telemetry_port")
        telemetry_nonce = required("golden.telemetry_nonce")
        telemetry_secret_hex = required("golden.telemetry_secret_hex")
        mq_timeout_s = required("golden.mq_timeout_s")
        text_values = (
            semantic_manifest_path,
            workspace_root,
            semantic_manifest_sha256,
            semantic_artifact_id,
            direction,
            lmcache_server_url,
            telemetry_host,
            telemetry_nonce,
            telemetry_secret_hex,
        )
        if (
            any(not isinstance(value, str) or not value for value in text_values)
            or type(telemetry_port) is not int
            or not _finite_positive(mq_timeout_s)
        ):
            raise V5VLLMConnectorError("runtime connector config types are malformed")
        semantic_path = Path(semantic_manifest_path)
        workspace_path = Path(workspace_root)
        if (
            not semantic_path.is_absolute()
            or not workspace_path.is_absolute()
            or workspace_path.is_symlink()
        ):
            raise V5VLLMConnectorError("runtime connector paths must be canonical and absolute")
        config = cls(
            semantic_manifest_path=semantic_path,
            semantic_manifest_sha256=semantic_manifest_sha256,
            semantic_artifact_id=semantic_artifact_id,
            workspace_root=workspace_path.resolve(),
            direction=direction,
            lmcache_server_url=lmcache_server_url,
            telemetry_host=telemetry_host,
            telemetry_port=telemetry_port,
            telemetry_nonce=telemetry_nonce,
            telemetry_secret_hex=telemetry_secret_hex,
            mq_timeout_s=float(mq_timeout_s),
        )
        if (
            not _is_sha256(config.semantic_manifest_sha256)
            or not config.semantic_artifact_id
            or not config.direction
            or not config.lmcache_server_url.startswith("tcp://")
            or not math.isfinite(config.mq_timeout_s)
            or config.mq_timeout_s <= 0
        ):
            raise V5VLLMConnectorError("runtime connector config is malformed")
        return config


def _load_audit_binding(
    config: _ConnectorConfig,
    *,
    device: str,
    load_transport: bool,
) -> _AuditBinding:
    semantic_path = _immutable_workspace_file(
        config.workspace_root,
        config.semantic_manifest_path,
        config.semantic_manifest_sha256,
    )
    manifest = SelectiveKVBridgeManifest.load(semantic_path)
    if sha256_file(semantic_path) != config.semantic_manifest_sha256:
        raise V5VLLMConnectorError("semantic runtime artifact changed while loading")
    errors = manifest.validate()
    if manifest.state is not ArtifactState.SEMANTIC_APPROVED:
        errors.append("runtime connector input is not semantic_approved")
    if manifest.artifact_id != config.semantic_artifact_id:
        errors.append("runtime connector semantic artifact identity changed")
    if manifest.direction != config.direction:
        errors.append("runtime connector direction changed")
    if errors:
        raise V5VLLMConnectorError("; ".join(errors))
    predictor_path = _workspace_artifact_path(
        config.workspace_root,
        manifest.risk_gate.predictor_uri,
        manifest.risk_gate.predictor_sha256,
    )
    predictor = RiskPredictor.from_artifact(
        predictor_path,
        expected_sha256=manifest.risk_gate.predictor_sha256,
        device="cpu",
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
    transport = None
    if load_transport:
        _workspace_artifact_path(
            config.workspace_root,
            manifest.transport.weights_uri,
            manifest.transport.weights_sha256,
        )
        transport = HeadAwareKVTransport.from_manifest(
            manifest,
            config.workspace_root / "runtime-manifest-anchor.json",
            device=device,
            offline=True,
        )
    return _AuditBinding(manifest, gate, transport)


def _workspace_artifact_path(root: Path, uri: str, expected_sha256: str) -> Path:
    path = Path(uri)
    candidate = path if path.is_absolute() else root / path
    return _immutable_workspace_file(root, candidate, expected_sha256)


def _immutable_workspace_file(root: Path, candidate: Path, expected_sha256: str) -> Path:
    try:
        canonical_root = root.resolve(strict=True)
    except OSError as exc:
        raise V5VLLMConnectorError("runtime workspace is unavailable") from exc
    if canonical_root != root or not canonical_root.is_dir():
        raise V5VLLMConnectorError("runtime workspace root is not canonical")
    absolute = candidate.absolute()
    try:
        relative = absolute.relative_to(canonical_root)
    except ValueError as exc:
        raise V5VLLMConnectorError("runtime artifact escapes the workspace") from exc
    if ".." in relative.parts:
        raise V5VLLMConnectorError("runtime artifact path is not canonical")
    cursor = canonical_root
    for part in relative.parts:
        cursor /= part
        if cursor.is_symlink():
            raise V5VLLMConnectorError("runtime artifact cannot use symbolic links")
    try:
        resolved = absolute.resolve(strict=True)
        resolved.relative_to(canonical_root)
    except (OSError, ValueError) as exc:
        raise V5VLLMConnectorError("runtime artifact is unavailable") from exc
    if (
        not resolved.is_file()
        or resolved.stat().st_mode & 0o222
        or sha256_file(resolved) != expected_sha256
    ):
        raise V5VLLMConnectorError("runtime artifact is missing or changed")
    return resolved


def _parse_request(
    request: Request,
    manifest: SelectiveKVBridgeManifest,
) -> _ParsedRequest | None:
    params = request.kv_transfer_params
    if params is None or V5_VLLM_CONNECTOR_PARAMS_KEY not in params:
        return None
    raw = params[V5_VLLM_CONNECTOR_PARAMS_KEY]
    try:
        if (
            not isinstance(raw, dict)
            or set(raw) != _REQUEST_FIELDS
            or raw.get("schema_version") != V5_VLLM_CONNECTOR_SCHEMA
        ):
            raise V5VLLMConnectorError("runtime request schema changed")
        token_values = raw["token_ids"]
        chunk_values = raw["chunks"]
        if (
            not isinstance(token_values, list)
            or not token_values
            or any(type(value) is not int or value < 0 for value in token_values)
            or not isinstance(chunk_values, list)
            or not chunk_values
            or any(
                not isinstance(item, dict) or set(item) != _CHUNK_FIELDS for item in chunk_values
            )
        ):
            raise V5VLLMConnectorError("runtime request token or chunk schema changed")
        token_ids = tuple(token_values)
        chunks = tuple(chunk_values)
        if any(
            type(item["start"]) is not int
            or type(item["end"]) is not int
            or not _is_sha256(item["checksum"])
            for item in chunks
        ):
            raise V5VLLMConnectorError("runtime request chunk identity changed")
        ranges = tuple((item["start"], item["end"]) for item in chunks)
        checksums = tuple(item["checksum"] for item in chunks)
        audit_request_id = raw["audit_request_id"]
        source_request_id = raw["source_request_id"]
        if not isinstance(raw["sidecar_base64"], str):
            raise V5VLLMConnectorError("runtime request sidecar encoding changed")
        sidecar_payload = base64.b64decode(raw["sidecar_base64"], validate=True)
        source_model_name = raw["source_model_name"]
        cache_salt = raw["cache_salt"]
        expected_accepted = raw["expected_accepted"]
        expected_decision = raw["expected_decision"]
        inject_partial_failure = raw["inject_partial_failure"]
        retrieve_timeout_s = raw["retrieve_timeout_s"]
        if (
            raw["semantic_artifact_id"] != manifest.artifact_id
            or not isinstance(audit_request_id, str)
            or not audit_request_id
            or not isinstance(source_request_id, str)
            or not source_request_id
            or not isinstance(source_model_name, str)
            or source_model_name != runtime_source_model_name(manifest)
            or type(raw["source_world_size"]) is not int
            or raw["source_world_size"] != 1
            or type(raw["source_worker_id"]) is not int
            or raw["source_worker_id"] != 0
            or not isinstance(cache_salt, str)
            or type(expected_accepted) is not bool
            or not isinstance(expected_decision, str)
            or not expected_decision
            or type(inject_partial_failure) is not bool
            or not _finite_positive(retrieve_timeout_s)
            or not _is_sha256(raw["token_ids_sha256"])
            or token_ids_sha256(list(token_ids)) != raw["token_ids_sha256"]
            or not isinstance(raw["prefix_hash"], str)
        ):
            raise V5VLLMConnectorError("runtime request identity changed")
        sidecar = SourceKVSidecar.from_bytes(sidecar_payload)
        stored = LMCacheStoredSourcePrefix(
            request_id=source_request_id,
            token_ids=token_ids,
            source_ranges=ranges,
            source_keys=tuple(
                lmcache_source_key(source_request_id, start, end) for start, end in ranges
            ),
            source_checksums=checksums,
            source_model_name=source_model_name,
            source_world_size=1,
            source_worker_id=0,
            cache_salt=cache_salt,
        )
        stored.build_retrieve_metadata(
            slot_mapping=tuple(range(len(token_ids))),
            prefix_hash=str(raw["prefix_hash"]),
            sidecar=sidecar_payload,
            timeout_s=float(retrieve_timeout_s),
        )
        if sidecar.prefix_hash != raw["prefix_hash"] or sidecar.prefix_length != len(token_ids):
            raise V5VLLMConnectorError("runtime sidecar prefix binding changed")
        return _ParsedRequest(
            audit_request_id=audit_request_id,
            stored=stored,
            prefix_hash=str(raw["prefix_hash"]),
            sidecar_payload=sidecar_payload,
            expected_accepted=expected_accepted,
            expected_decision=expected_decision,
            inject_partial_failure=inject_partial_failure,
            retrieve_timeout_s=float(retrieve_timeout_s),
        )
    except V5VLLMConnectorError:
        raise
    except Exception as exc:
        raise V5VLLMConnectorError("runtime request parameters are malformed") from exc


def _ordered_layer_caches(kv_caches: dict[str, Any], expected_layers: int) -> list[Any]:
    by_layer: dict[int, Any] = {}
    for name, tensor in kv_caches.items():
        match = _LAYER_PATTERN.search(name)
        if match is None:
            raise V5VLLMConnectorError("vLLM KV cache layer name is unsupported")
        layer = int(match.group(1))
        if layer in by_layer:
            raise V5VLLMConnectorError("vLLM registered duplicate KV cache layers")
        by_layer[layer] = tensor
    if set(by_layer) != set(range(expected_layers)):
        raise V5VLLMConnectorError("vLLM registered an incomplete KV cache layer set")
    return [by_layer[index] for index in range(expected_layers)]


def _worker_device() -> str:
    import torch

    if not torch.cuda.is_available():
        raise V5VLLMConnectorError("runtime connector worker requires CUDA")
    return str(torch.device("cuda", torch.cuda.current_device()))


def _is_sha256(value: Any) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True


def _finite_positive(value: Any) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(value)
        and value > 0
    )
