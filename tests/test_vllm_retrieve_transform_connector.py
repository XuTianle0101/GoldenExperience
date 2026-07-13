from __future__ import annotations

import hashlib
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch

pytest.importorskip("vllm")

from goldenexperience.runtime.direct_paged_kv import DirectInjectionResult
from goldenexperience.runtime.lmcache_retrieve_transform import (
    LMCacheStoredSourcePrefix,
    RuntimeBlockValidityTracker,
)
from goldenexperience.runtime.vllm_retrieve_transform_connector import (
    V5ConnectorMetadata,
    V5RetrieveTransformScheduler,
    V5RetrieveTransformWorker,
    V5VLLMConnectorError,
    _workspace_artifact_path,
    build_vllm_retrieve_transform_params,
    runtime_source_model_name,
)
from goldenexperience.size_variant.risk_gate import AdmissionDecision, SourceKVSidecar
from goldenexperience.size_variant.selective_manifest import (
    ArtifactState,
    SelectiveKVBridgeManifest,
)


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _manifest() -> Any:
    return SimpleNamespace(
        artifact_id="selective-kv-test",
        state=ArtifactState.SEMANTIC_APPROVED,
        source=SimpleNamespace(weights_sha256=_digest("source")),
        target=SimpleNamespace(num_layers=2, num_key_value_heads=2, head_dim=4),
    )


def _sidecar() -> SourceKVSidecar:
    return SourceKVSidecar(
        model_pair_id="qwen3_4b_to_8b",
        source_model_hash=_digest("source"),
        target_model_hash=_digest("target"),
        tokenizer_hash=_digest("tokenizer"),
        transport_weights_hash=_digest("transport"),
        prefix_hash=_digest("prefix"),
        prefix_length=128,
        token_bucket=128,
        num_layers=1,
        num_heads=1,
        statistics=(0.0,) * 8,
        sketch=(0.0,) * 128,
        ood_distance=0.0,
        history_samples=1,
        history_failures=0,
        history_greedy_agreement=1.0,
    )


def _stored(manifest: Any) -> LMCacheStoredSourcePrefix:
    return LMCacheStoredSourcePrefix(
        request_id="source-store",
        token_ids=tuple(range(128)),
        source_ranges=((0, 64), (64, 128)),
        source_keys=("source-store:0:64", "source-store:64:128"),
        source_checksums=(_digest("chunk-0"), _digest("chunk-1")),
        source_model_name=runtime_source_model_name(manifest),
        source_world_size=1,
        source_worker_id=0,
        cache_salt="audit",
    )


class _Emitter:
    def __init__(self) -> None:
        self.events: list[tuple[str, str, dict[str, Any]]] = []

    def emit(self, *, request_id: str, kind: str, evidence: dict[str, Any]) -> None:
        self.events.append((request_id, kind, evidence))


class _Gate:
    def __init__(self, accepted: bool, decision: str) -> None:
        self.admission = AdmissionDecision(accepted, decision, 0.1)

    def evaluate(self, _sidecar: bytes) -> AdmissionDecision:
        return self.admission


class _Blocks:
    def get_block_ids(self) -> tuple[list[int]]:
        return (list(range(3, 20)),)


def _request(manifest: Any, *, accepted: bool, decision: str) -> Any:
    params = build_vllm_retrieve_transform_params(
        manifest=cast(SelectiveKVBridgeManifest, manifest),
        stored=_stored(manifest),
        sidecar_payload=_sidecar().to_bytes(),
        audit_request_id="audit-request",
        accepted=accepted,
        decision=decision,
    )
    return SimpleNamespace(
        request_id="target-request",
        prompt_token_ids=list(range(128)) + [999],
        kv_transfer_params=params,
    )


def test_scheduler_gates_before_building_exact_paged_metadata() -> None:
    manifest = _manifest()
    emitter = _Emitter()
    scheduler = V5RetrieveTransformScheduler(
        manifest=cast(SelectiveKVBridgeManifest, manifest),
        gate=cast(Any, _Gate(True, "accepted")),
        block_size=16,
        emitter=cast(Any, emitter),
    )
    request = _request(manifest, accepted=True, decision="accepted")

    matched, asynchronous = scheduler.get_num_new_matched_tokens(request, 0)
    assert scheduler.get_num_new_matched_tokens(request, 0) == (128, True)
    scheduler.update_state_after_alloc(request, cast(Any, _Blocks()), matched)
    metadata = scheduler.build_connector_meta(cast(Any, object()))

    assert (matched, asynchronous) == (128, True)
    assert len(metadata.requests) == 1
    load = metadata.requests[0]
    assert load.audit_request_id == "audit-request"
    assert load.metadata.request.slot_mapping[:2] == (48, 49)
    assert load.metadata.request.slot_mapping[-1] == 175
    assert load.metadata.request.request_id == "source-store"
    assert load.metadata.request.source_keys[0] == "source-store:0:64"
    assert emitter.events[0][1] == "gate"
    assert emitter.events[0][2]["source_read_attempted"] is False
    assert len(emitter.events) == 1


def test_scheduler_rejection_never_builds_source_read_metadata() -> None:
    manifest = _manifest()
    emitter = _Emitter()
    scheduler = V5RetrieveTransformScheduler(
        manifest=cast(SelectiveKVBridgeManifest, manifest),
        gate=cast(Any, _Gate(False, "predicted_unsafe")),
        block_size=16,
        emitter=cast(Any, emitter),
    )
    request = _request(manifest, accepted=False, decision="predicted_unsafe")

    assert scheduler.get_num_new_matched_tokens(request, 0) == (0, False)
    assert scheduler.build_connector_meta(cast(Any, object())).requests == ()
    assert emitter.events == [
        (
            "audit-request",
            "gate",
            {
                "accepted": False,
                "decision": "predicted_unsafe",
                "unsafe_probability": 0.1,
                "binding_matches": True,
                "source_read_attempted": False,
            },
        )
    ]


class _Reader:
    def __init__(self) -> None:
        self.bound: list[str] = []
        self.finished: list[str] = []

    def bind_request(self, metadata: Any) -> None:
        self.bound.append(metadata.request.request_id)

    def finish_request(self, request_id: str) -> None:
        self.finished.append(request_id)


class _Injector:
    def __init__(self, worker: V5RetrieveTransformWorker, success: bool) -> None:
        self.worker = worker
        self.success = success

    def retrieve_transform(self, request: Any, *, kv_caches: Any) -> DirectInjectionResult:
        assert len(kv_caches) == 2
        if self.success:
            self.worker._publish_load_complete(request.request_id, (3,))
        return DirectInjectionResult(
            success=self.success,
            accepted=True,
            fallback_reason="none" if self.success else "direct_injection_failed",
            admission=AdmissionDecision(True, "accepted", 0.1),
            source_read_attempted=True,
            source_chunks_read=2,
            tokens_scattered=128 if self.success else 0,
            invalidated_blocks=() if self.success else tuple(range(3, 11)),
            target_mooncake_puts=0,
            elapsed_ms=5.0,
            error=None if self.success else "injected",
        )


def _worker(
    manifest: Any,
    reader: Any,
    emitter: _Emitter,
    *,
    success: bool,
) -> V5RetrieveTransformWorker:
    worker = object.__new__(V5RetrieveTransformWorker)
    worker.manifest = manifest
    worker.reader = reader
    worker.emitter = emitter
    worker.validity = RuntimeBlockValidityTracker()
    worker._kv_caches = tuple(torch.zeros(20, 2, 16, 2, 4, dtype=torch.bfloat16) for _ in range(2))
    worker._load_complete = set()
    worker._finished_recving = set()
    worker._active_request = None
    worker._inject_partial_failure = False
    worker._partial_failure_count = 0
    worker.injector = _Injector(worker, success)
    return worker


@pytest.mark.parametrize("success", [True, False])
def test_worker_reports_success_or_invalid_blocks_without_target_puts(success: bool) -> None:
    manifest = _manifest()
    scheduler = V5RetrieveTransformScheduler(
        manifest=cast(SelectiveKVBridgeManifest, manifest),
        gate=cast(Any, _Gate(True, "accepted")),
        block_size=16,
        emitter=cast(Any, _Emitter()),
    )
    request = _request(manifest, accepted=True, decision="accepted")
    matched, _ = scheduler.get_num_new_matched_tokens(request, 0)
    scheduler.update_state_after_alloc(request, cast(Any, _Blocks()), matched)
    metadata = scheduler.build_connector_meta(cast(Any, object()))
    reader = _Reader()
    emitter = _Emitter()
    worker = _worker(manifest, reader, emitter, success=success)

    worker.start_load_kv(cast(V5ConnectorMetadata, metadata))

    event = emitter.events[0]
    assert event[1] == "execution"
    assert event[2]["success"] is success
    assert event[2]["target_mooncake_puts"] == 0
    assert event[2]["load_complete_published"] is success
    assert reader.bound == ["source-store"]
    assert reader.finished == ["source-store"]
    assert worker.drain_finished_recving() == {"target-request"}
    assert worker.drain_finished_recving() == set()
    invalid = worker.get_block_ids_with_load_errors()
    assert invalid == (set() if success else set(range(3, 11)))


def test_scheduler_rejects_a_target_prompt_with_another_prefix() -> None:
    manifest = _manifest()
    scheduler = V5RetrieveTransformScheduler(
        manifest=cast(SelectiveKVBridgeManifest, manifest),
        gate=cast(Any, _Gate(True, "accepted")),
        block_size=16,
        emitter=cast(Any, _Emitter()),
    )
    request = _request(manifest, accepted=True, decision="accepted")
    request.prompt_token_ids[0] = 999

    with pytest.raises(V5VLLMConnectorError, match="target prompt differs"):
        scheduler.get_num_new_matched_tokens(request, 0)


class _FailingReader(_Reader):
    def bind_request(self, metadata: Any) -> None:
        del metadata
        raise RuntimeError("source reader unavailable")


def test_worker_turns_source_bind_failure_into_native_recompute() -> None:
    manifest = _manifest()
    scheduler = V5RetrieveTransformScheduler(
        manifest=cast(SelectiveKVBridgeManifest, manifest),
        gate=cast(Any, _Gate(True, "accepted")),
        block_size=16,
        emitter=cast(Any, _Emitter()),
    )
    request = _request(manifest, accepted=True, decision="accepted")
    matched, _ = scheduler.get_num_new_matched_tokens(request, 0)
    scheduler.update_state_after_alloc(request, cast(Any, _Blocks()), matched)
    metadata = scheduler.build_connector_meta(cast(Any, object()))
    reader = _FailingReader()
    emitter = _Emitter()
    worker = _worker(manifest, reader, emitter, success=True)

    worker.start_load_kv(metadata)

    assert worker.get_block_ids_with_load_errors() == set(range(3, 11))
    assert reader.finished == []
    evidence = emitter.events[0][2]
    assert evidence["success"] is False
    assert evidence["fallback_reason"] == "connector_worker_failure"
    assert evidence["source_read_attempted"] is False
    assert evidence["target_mooncake_puts"] == 0


def test_worker_reuses_one_source_prefix_for_distinct_target_requests() -> None:
    manifest = _manifest()
    scheduler = V5RetrieveTransformScheduler(
        manifest=cast(SelectiveKVBridgeManifest, manifest),
        gate=cast(Any, _Gate(True, "accepted")),
        block_size=16,
        emitter=cast(Any, _Emitter()),
    )
    request = _request(manifest, accepted=True, decision="accepted")
    matched, _ = scheduler.get_num_new_matched_tokens(request, 0)
    scheduler.update_state_after_alloc(request, cast(Any, _Blocks()), matched)
    first = scheduler.build_connector_meta(cast(Any, object())).requests[0]
    second = replace(
        first,
        target_request_id="target-request-2",
        audit_request_id="audit-request-2",
    )
    reader = _Reader()
    emitter = _Emitter()
    worker = _worker(manifest, reader, emitter, success=True)

    worker.start_load_kv(V5ConnectorMetadata((first,)))
    worker.start_load_kv(V5ConnectorMetadata((second,)))

    assert reader.bound == ["source-store", "source-store"]
    assert reader.finished == ["source-store", "source-store"]
    assert worker.get_block_ids_with_load_errors() == set()
    assert [event[0] for event in emitter.events] == ["audit-request", "audit-request-2"]


def test_runtime_artifacts_must_be_read_only_regular_workspace_files(tmp_path: Path) -> None:
    artifact = tmp_path / "object.safetensors"
    artifact.write_bytes(b"immutable")
    artifact.chmod(0o444)
    checksum = hashlib.sha256(b"immutable").hexdigest()

    assert _workspace_artifact_path(tmp_path.resolve(), artifact, checksum) == artifact

    link = tmp_path / "link.safetensors"
    link.symlink_to(artifact)
    with pytest.raises(V5VLLMConnectorError, match="symbolic"):
        _workspace_artifact_path(tmp_path.resolve(), link, checksum)
