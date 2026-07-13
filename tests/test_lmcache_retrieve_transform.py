from __future__ import annotations

import hashlib
import pickle
import threading
from dataclasses import replace
from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch

import goldenexperience.runtime.lmcache_retrieve_transform as bridge_module
from goldenexperience.runtime.direct_paged_kv import (
    DirectInjectionResult,
    RetrieveTransformRequest,
)
from goldenexperience.runtime.lmcache_retrieve_transform import (
    LMCacheMPSourceChunkReader,
    LMCacheRetrieveTransformBatch,
    LMCacheRetrieveTransformBridge,
    LMCacheRetrieveTransformError,
    LMCacheRetrieveTransformMetadata,
    RuntimeBlockValidityTracker,
    lmcache_source_key,
    probe_runtime_stack,
    source_chunk_checksums,
    verify_runtime_stack_identity,
)
from goldenexperience.size_variant.cached_kv_manifest import CachedKVModelSpec
from goldenexperience.size_variant.risk_gate import AdmissionDecision


def _digest(value: str | bytes) -> str:
    payload = value if isinstance(value, bytes) else value.encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _model() -> CachedKVModelSpec:
    return CachedKVModelSpec(
        model_id="Qwen/Qwen3-test",
        parameter_count_b=1.0,
        revision="test",
        architecture="qwen3",
        config_sha256=_digest("config"),
        tokenizer_sha256=_digest("tokenizer"),
        weights_sha256=_digest("weights"),
        num_layers=2,
        num_key_value_heads=2,
        head_dim=4,
        dtype="bfloat16",
        rope_theta=1_000_000,
        max_position_embeddings=40960,
        chat_template_sha256=_digest("chat"),
    )


def _metadata(request_id: str = "request") -> LMCacheRetrieveTransformMetadata:
    source_keys = (
        lmcache_source_key(request_id, 0, 2),
        lmcache_source_key(request_id, 2, 4),
    )
    request = RetrieveTransformRequest(
        request_id=request_id,
        source_keys=source_keys,
        source_checksums=(_digest("first"), _digest("second")),
        chunk_token_counts=(2, 2),
        slot_mapping=(0, 1, 2, 3),
        prefix_hash=_digest("prefix"),
        sidecar=None,
    )
    return LMCacheRetrieveTransformMetadata(
        request=request,
        token_ids=(1, 2, 3, 4),
        source_ranges=((0, 2), (2, 4)),
        source_model_name="source-model",
        source_world_size=1,
        source_worker_id=0,
    )


def test_pinned_runtime_stack_probe_records_exact_upstream_sources() -> None:
    pytest.importorskip("lmcache")
    pytest.importorskip("vllm")

    identity = probe_runtime_stack()

    assert identity.lmcache_version == "0.4.6"
    assert identity.vllm_version == "0.24.0"
    assert identity.torch_version.startswith("2.11.0")
    assert len(identity.sources) == 10
    assert all(item.validate() == [] for item in identity.sources)
    assert identity.validate() == []
    verify_runtime_stack_identity(identity)


def test_runtime_stack_probe_rejects_an_unpinned_version(monkeypatch) -> None:
    real_version = bridge_module.importlib.metadata.version
    monkeypatch.setattr(
        bridge_module.importlib.metadata,
        "version",
        lambda name: "0.4.7" if name == "lmcache" else real_version(name),
    )

    with pytest.raises(LMCacheRetrieveTransformError, match="LMCache 0.4.6"):
        probe_runtime_stack()


def test_retrieve_transform_metadata_binds_canonical_contiguous_chunks() -> None:
    metadata = _metadata()

    assert metadata.validate(chunk_size=2) == []
    assert "LMCache source key is not canonical" in bridge_module.LMCacheRetrieveTransformMetadata(
        request=replace(
            metadata.request,
            source_keys=("changed", metadata.request.source_keys[1]),
        ),
        token_ids=metadata.token_ids,
        source_ranges=metadata.source_ranges,
        source_model_name=metadata.source_model_name,
        source_world_size=metadata.source_world_size,
        source_worker_id=metadata.source_worker_id,
    ).validate(chunk_size=2)
    with pytest.raises(LMCacheRetrieveTransformError, match="fields are invalid"):
        lmcache_source_key("", 0, 2)


class _Future:
    def __init__(self, value: Any) -> None:
        self.value = value

    def result(self, *, timeout: float) -> Any:
        assert timeout > 0
        return self.value


def test_source_reader_uses_prepare_commit_and_restores_head_layout() -> None:
    pytest.importorskip("lmcache")
    source = _model()
    flat = torch.arange(2 * 2 * 2 * 8, dtype=torch.bfloat16).reshape(2, 2, 2, 8)
    expected = flat.reshape(2, 2, 2, 2, 4).permute(0, 1, 3, 2, 4).contiguous()
    metadata = _metadata("reader")
    metadata = bridge_module.LMCacheRetrieveTransformMetadata(
        request=replace(
            metadata.request,
            source_keys=(lmcache_source_key("reader", 0, 2),),
            source_checksums=source_chunk_checksums((expected,)),
            chunk_token_counts=(2,),
            slot_mapping=(0, 1),
        ),
        token_ids=metadata.token_ids,
        source_ranges=((0, 2),),
        source_model_name=metadata.source_model_name,
        source_world_size=metadata.source_world_size,
        source_worker_id=metadata.source_worker_id,
    )
    request_types = SimpleNamespace(
        PREPARE_RETRIEVE="prepare",
        COMMIT_RETRIEVE="commit",
        END_SESSION="end",
    )
    calls: list[tuple[Any, list[Any]]] = []

    def send(_client: Any, request_type: Any, payload: list[Any]) -> _Future:
        calls.append((request_type, payload))
        if request_type == "prepare":
            return _Future(SimpleNamespace(success=True, data=pickle.dumps([flat])))
        return _Future(True)

    reader = object.__new__(LMCacheMPSourceChunkReader)
    reader.source_model_name = "source-model"
    reader.source_world_size = 1
    reader.source_worker_id = 0
    reader.source = source
    reader.mq_timeout_s = 5.0
    reader.chunk_size = 2
    reader.instance_id = 123
    reader._lock = threading.RLock()
    reader._bound = None
    reader._closed = False
    reader._send_request = send
    reader._request_type = request_types
    reader._client = object()
    reader.bind_request(metadata)

    chunks = reader.read_many_exact(metadata.request.source_keys, timeout_s=1.0)
    reader.finish_request("reader")

    assert len(chunks) == 1
    torch.testing.assert_close(chunks[0], expected)
    assert [item[0] for item in calls] == ["prepare", "commit", "end"]
    assert calls[0][1][0].model_name == "source-model"
    assert calls[0][1][0].token_ids == metadata.token_ids


class _UpstreamConnector:
    def __init__(self) -> None:
        self.registered: dict[str, Any] | None = None
        self.bound: Any = None
        self.started = 0
        self.waited = 0
        self.shutdowns = 0

    def register_kv_caches(self, caches: dict[str, Any]) -> None:
        self.registered = caches

    def bind_connector_metadata(self, metadata: Any) -> None:
        self.bound = metadata

    def start_load_kv(self, _context: Any) -> None:
        self.started += 1

    def wait_for_save(self) -> None:
        self.waited += 1

    def get_finished(self, _finished: set[str]) -> tuple[set[str], set[str]]:
        return set(), set()

    def get_block_ids_with_load_errors(self) -> set[int]:
        return {9}

    def shutdown(self) -> None:
        self.shutdowns += 1


class _SourceReader:
    chunk_size = 2

    def __init__(self) -> None:
        self.bound: list[str] = []
        self.finished: list[str] = []
        self.closed = 0

    def bind_request(self, metadata: LMCacheRetrieveTransformMetadata) -> None:
        self.bound.append(metadata.request.request_id)

    def finish_request(self, request_id: str) -> None:
        self.finished.append(request_id)

    def close(self) -> None:
        self.closed += 1


class _Injector:
    def __init__(self, tracker: RuntimeBlockValidityTracker) -> None:
        self.transport = SimpleNamespace(target=SimpleNamespace(num_layers=2))
        self.tracker = tracker
        self.publish: Any = None

    def retrieve_transform(
        self,
        request: RetrieveTransformRequest,
        *,
        kv_caches: Any,
    ) -> DirectInjectionResult:
        assert len(kv_caches) == 2
        accepted = AdmissionDecision(True, "accepted")
        if request.request_id == "success":
            self.tracker.mark_valid((0,))
            self.publish(request.request_id, (0,))
            return DirectInjectionResult(
                True,
                True,
                "none",
                accepted,
                True,
                2,
                4,
                (),
                0,
                1.0,
            )
        self.tracker.mark_invalid((1,))
        return DirectInjectionResult(
            False,
            True,
            "direct_injection_failed",
            accepted,
            True,
            1,
            0,
            (1,),
            0,
            1.0,
            "injected failure",
        )


def test_worker_bridge_preserves_upstream_and_reports_atomic_failures(monkeypatch) -> None:
    pytest.importorskip("lmcache")
    identity = SimpleNamespace(content_sha256=lambda: _digest("stack"))
    monkeypatch.setattr(bridge_module, "probe_runtime_stack", lambda: identity)
    tracker = RuntimeBlockValidityTracker()
    injector = _Injector(tracker)
    upstream = _UpstreamConnector()
    reader = _SourceReader()
    bridge = LMCacheRetrieveTransformBridge(
        upstream_connector=upstream,
        injector=cast(Any, injector),
        source_reader=cast(Any, reader),
        validity_tracker=tracker,
    )
    injector.publish = bridge.publish_load_complete
    caches = {"layer.0": torch.zeros(1), "layer.1": torch.zeros(1)}
    bridge.register_kv_caches(caches)

    observations = bridge.start_load_kv(
        object(),
        LMCacheRetrieveTransformBatch((_metadata("success"), _metadata("failure"))),
    )

    assert upstream.registered == caches
    assert upstream.started == 1
    assert upstream.bound is not None
    assert reader.bound == ["success", "failure"]
    assert reader.finished == ["success", "failure"]
    assert observations[0].result.success
    assert observations[0].load_complete_published
    assert not observations[1].result.success
    assert not observations[1].load_complete_published
    assert observations[1].invalid_blocks_reported == (1,)
    assert bridge.get_block_ids_with_load_errors() == {1, 9}
    assert bridge.get_block_ids_with_load_errors() == {9}
    bridge.wait_for_save()
    bridge.shutdown()
    assert upstream.waited == 1
    assert upstream.shutdowns == 1
    assert reader.closed == 1


def test_validity_tracker_never_keeps_a_block_valid_and_invalid() -> None:
    tracker = RuntimeBlockValidityTracker()
    tracker.mark_valid((1, 2))
    tracker.mark_invalid((2, 3))

    assert tracker.valid == frozenset({1})
    assert tracker.invalid == frozenset({2, 3})
    assert tracker.drain_invalid() == {2, 3}
    assert tracker.invalid == frozenset()


def test_source_checksums_reject_non_cpu_inputs() -> None:
    with pytest.raises(LMCacheRetrieveTransformError, match="CPU tensors"):
        source_chunk_checksums((object(),))
