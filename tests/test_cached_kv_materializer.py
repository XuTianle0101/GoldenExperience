import ctypes
import io
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from goldenexperience.benchmarks.cached_kv_cost import (
    CACHED_KV_COST_SCHEMA_VERSION,
    NATIVE_PREFILL_COST_SCHEMA_VERSION,
    build_native_prefill_report,
    load_cached_kv_cost_evidence,
    load_native_prefill_evidence,
    run_cached_kv_cost_benchmark,
)
from goldenexperience.runtime import cross_model_materializer
from goldenexperience.runtime.cross_model_materializer import (
    materialize_cached_qwen3,
    preload_cached_qwen3_bridge,
    serve_materializer_jsonl,
)
from goldenexperience.runtime.mooncake_objects import (
    ExactMooncakeStore,
    MooncakeObjectError,
    publish_external_index,
)


class FakeMooncakeStore:
    def __init__(self, objects: dict[str, bytes] | None = None) -> None:
        self.objects = dict(objects or {})
        self.setup_rc = 0
        self.closed = False
        self.short_reads: set[str] = set()
        self.reported_sizes: dict[str, int] = {}
        self.put_results: list[int] | None = None
        self.removed: list[str] = []

    def setup(self, config: dict[str, str]) -> int:
        self.setup_config = config
        return self.setup_rc

    def close(self) -> int:
        self.closed = True
        return 0

    def get_size(self, key: str) -> int:
        if key in self.reported_sizes:
            return self.reported_sizes[key]
        value = self.objects.get(key)
        return len(value) if value is not None else -1

    def is_exist(self, key: str) -> int:
        return int(key in self.objects)

    def batch_is_exist(self, keys: list[str]) -> list[int]:
        return [self.is_exist(key) for key in keys]

    def get_into(self, key: str, pointer: int, size: int) -> int:
        data = self.objects.get(key, b"")
        read_size = min(len(data), size)
        if key in self.short_reads and read_size:
            read_size -= 1
        ctypes.memmove(pointer, data, read_size)
        return read_size

    def batch_get_into(
        self,
        keys: list[str],
        pointers: list[int],
        sizes: list[int],
    ) -> list[int]:
        return [
            self.get_into(key, pointer, size)
            for key, pointer, size in zip(keys, pointers, sizes, strict=True)
        ]

    def put_from(self, key: str, pointer: int, size: int, result: int = 0) -> int:
        if result == 0:
            self.objects[key] = ctypes.string_at(pointer, size)
        return result

    def batch_put_from(
        self,
        keys: list[str],
        pointers: list[int],
        sizes: list[int],
    ) -> list[int]:
        results = self.put_results or [0] * len(keys)
        return [
            self.put_from(key, pointer, size, result)
            for key, pointer, size, result in zip(
                keys,
                pointers,
                sizes,
                results,
                strict=True,
            )
        ]

    def remove(self, key: str, force: bool) -> int:
        assert force is True
        self.removed.append(key)
        self.objects.pop(key, None)
        return 0


class FakeBridge:
    def __init__(self, *, direction: str = "8b_to_14b") -> None:
        self.device = torch.device("cpu")
        self.manifest = SimpleNamespace(
            bridge_id=f"fake-{direction}",
            direction=direction,
            approved=True,
            scope="global",
            source=SimpleNamespace(num_layers=2, kv_width=4, dtype="bfloat16"),
            target=SimpleNamespace(num_layers=3, kv_width=4, dtype="bfloat16"),
            thresholds=SimpleNamespace(max_materialization_to_prefill_ratio=0.70),
        )
        self.positions: list[int] = []

    def transform(self, source: torch.Tensor, *, position_start: int) -> torch.Tensor:
        self.positions.append(position_start)
        target = torch.empty(2, 3, source.shape[2], 4, dtype=source.dtype)
        target[:, 0].copy_(source[:, 0])
        target[:, 1].copy_(source[:, 1])
        target[:, 2].copy_(source[:, 0] + source[:, 1])
        return target


def _key_builder(*, model_name: str, chunk_hash: str, **_: object) -> str:
    return f"{model_name}@{chunk_hash}"


def _tensor_bytes(tensor: torch.Tensor) -> bytes:
    return tensor.contiguous().view(torch.uint8).numpy().tobytes()


def _request(tmp_path: Path) -> dict:
    hashes = ["0xaa", "0xbb"]
    return {
        "mode": "cached_kv",
        "bridge_manifest_path": str(tmp_path / "bridge.json"),
        "source_model_path": "/models/Qwen3-8B",
        "source_model_name": "source",
        "target_model_path": "/models/Qwen3-14B",
        "target_model_name": "target",
        "direction": "8b_to_14b",
        "chunk_hashes": hashes,
        "chunk_size": 2,
        "source_lookup_record": {
            "request_id": "request-worker",
            "model_name": "source",
            "chunk_size": 2,
            "seq_len": 5,
            "dtypes": ["torch.bfloat16"],
            "shapes": [[2, 2, 2, 4]],
            "chunk_hashes": hashes,
        },
        "prompt_binding": {
            "source_response_id": "request",
            "lookup_request_id": "request-worker",
            "token_count": 5,
            "token_ids_sha256": "a" * 64,
            "target_token_count": 5,
            "target_token_ids_sha256": "b" * 64,
            "chunk_size": 2,
            "shared_prefix_chunk_count": 2,
            "shared_prefix_token_count": 4,
            "hash_algorithm": "blake3",
        },
        "hash_algorithm": "blake3",
        "world_size": 1,
        "kv_rank": 7,
        "native_target_prefill_ms": 100_000.0,
        "mooncake_setup_config": {"metadata_server": "http://metadata"},
        "external_index_path": str(tmp_path / "external.jsonl"),
        "device": "cpu",
    }


def _source_objects(request: dict) -> dict[str, bytes]:
    objects: dict[str, bytes] = {}
    for index, chunk_hash in enumerate(request["chunk_hashes"]):
        tensor = (
            torch.arange(2 * 2 * 2 * 4, dtype=torch.float32)
            .reshape(2, 2, 2, 4)
            .add(index * 100)
            .to(torch.bfloat16)
        )
        objects[_key_builder(model_name="source", chunk_hash=chunk_hash)] = _tensor_bytes(tensor)
    return objects


def test_exact_mooncake_store_reads_and_writes_complete_objects() -> None:
    fake = FakeMooncakeStore({"source": b"abcd"})

    with ExactMooncakeStore({}, store_factory=lambda: fake) as store:
        reads = store.read_many_exact(["source"], [4])
        writes = store.write_many_exact(["target"], [b"wxyz"])

    assert bytes(reads[0].data) == b"abcd"
    assert reads[0].read_bytes == reads[0].remote_bytes == 4
    assert writes[0].bytes == writes[0].remote_bytes == 4
    assert fake.objects["target"] == b"wxyz"
    assert fake.closed is True


def test_exact_mooncake_store_rejects_size_mismatch_and_short_read() -> None:
    size_mismatch = FakeMooncakeStore({"source": b"abcd"})
    size_mismatch.reported_sizes["source"] = 3
    with (
        ExactMooncakeStore({}, store_factory=lambda: size_mismatch) as store,
        pytest.raises(MooncakeObjectError, match="size mismatch"),
    ):
        store.read_many_exact(["source"], [4])

    short_read = FakeMooncakeStore({"source": b"abcd"})
    short_read.short_reads.add("source")
    with (
        ExactMooncakeStore({}, store_factory=lambda: short_read) as store,
        pytest.raises(MooncakeObjectError, match="incomplete source read"),
    ):
        store.read_many_exact(["source"], [4])


def test_exact_mooncake_store_rolls_back_partial_batch_put() -> None:
    fake = FakeMooncakeStore()
    fake.put_results = [0, -1]

    with (
        ExactMooncakeStore({}, store_factory=lambda: fake) as store,
        pytest.raises(MooncakeObjectError, match="target put failed"),
    ):
        store.write_many_exact(["target-a", "target-b"], [b"aaaa", b"bbbb"])

    assert "target-a" not in fake.objects
    assert fake.removed == ["target-a", "target-b"]


def test_external_index_publication_preserves_existing_records(tmp_path: Path) -> None:
    path = tmp_path / "external.jsonl"
    publish_external_index(path, [{"key": "old", "bytes": 4}])
    publish_external_index(
        path,
        [
            {"key": "new-a", "bytes": 8},
            {"key": "new-b", "bytes": 8},
        ],
    )

    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert [record["key"] for record in records] == ["old", "new-a", "new-b"]


def test_cached_materializer_uses_source_objects_and_publishes_target_index(
    tmp_path: Path,
) -> None:
    request = _request(tmp_path)
    fake = FakeMooncakeStore(_source_objects(request))
    bridge = FakeBridge()

    result = materialize_cached_qwen3(
        request,
        store_factory=lambda: fake,
        bridge_loader=lambda *args, **kwargs: bridge,
        key_builder=_key_builder,
    )

    assert result["success"] is True
    assert result["injected"] is True
    assert result["artifact_cache"] == {"hit": False, "resident_loader": False}
    assert result["injection"]["injected_count"] == 2
    assert result["injection"]["shared_prefix_tokens"] == 4
    assert result["runtime_quality_gate"]["checks"]["prompt_prefix_bound"] is True
    assert result["prompt_binding"]["target_token_ids_sha256"] == "b" * 64
    assert bridge.positions == [0, 2]
    assert "target@0xaa" in fake.objects
    assert "target@0xbb" in fake.objects
    source_a = torch.frombuffer(
        bytearray(fake.objects["source@0xaa"]), dtype=torch.bfloat16
    ).reshape(2, 2, 2, 4)
    target_a = torch.frombuffer(
        bytearray(fake.objects["target@0xaa"]), dtype=torch.bfloat16
    ).reshape(2, 3, 2, 4)
    torch.testing.assert_close(target_a[:, 0], source_a[:, 0], atol=0, rtol=0)
    torch.testing.assert_close(target_a[:, 2], source_a[:, 0] + source_a[:, 1], atol=0, rtol=0)
    records = [
        json.loads(line)
        for line in Path(request["external_index_path"]).read_text(encoding="utf-8").splitlines()
    ]
    assert [record["key"] for record in records] == ["target@0xaa", "target@0xbb"]
    assert all(record["provenance"]["bridge_id"] == "fake-8b_to_14b" for record in records)
    assert all(record["provenance"]["hash_algorithm"] == "blake3" for record in records)


def test_cached_materializer_accepts_only_an_exact_shorter_shared_prefix(
    tmp_path: Path,
) -> None:
    request = _request(tmp_path)
    request["source_lookup_record"]["seq_len"] = 7
    request["source_lookup_record"]["chunk_hashes"] = ["0xaa", "0xbb", "0xcc"]
    request["prompt_binding"]["token_count"] = 7
    fake = FakeMooncakeStore(_source_objects(request))

    result = materialize_cached_qwen3(
        request,
        store_factory=lambda: fake,
        bridge_loader=lambda *args, **kwargs: FakeBridge(),
        key_builder=_key_builder,
    )

    assert result["success"] is True
    assert result["injection"]["injected_count"] == 2
    assert result["source_keys"] == ["source@0xaa", "source@0xbb"]

    request["source_lookup_record"]["chunk_hashes"][1] = "0xdd"
    result = materialize_cached_qwen3(
        request,
        store_factory=lambda: fake,
        bridge_loader=lambda *args, **kwargs: FakeBridge(),
        key_builder=_key_builder,
    )
    assert result["fallback_reason"] == "invalid_materializer_request"


@pytest.mark.parametrize(
    "failure",
    ["short_read", "partial_put", "cost_gate", "index_publish"],
)
def test_cached_materializer_does_not_publish_partial_targets(
    tmp_path: Path,
    failure: str,
) -> None:
    request = _request(tmp_path)
    fake = FakeMooncakeStore(_source_objects(request))
    if failure == "short_read":
        fake.short_reads.add("source@0xbb")
    elif failure == "partial_put":
        fake.put_results = [0, -1]
    elif failure == "cost_gate":
        request["native_target_prefill_ms"] = 0.000001
    else:
        Path(request["external_index_path"]).write_text("partial", encoding="utf-8")

    result = materialize_cached_qwen3(
        request,
        store_factory=lambda: fake,
        bridge_loader=lambda *args, **kwargs: FakeBridge(),
        key_builder=_key_builder,
    )

    assert result["success"] is False
    assert result["injected"] is False
    if failure == "index_publish":
        assert Path(request["external_index_path"]).read_text(encoding="utf-8") == "partial"
    else:
        assert not Path(request["external_index_path"]).exists()
    assert not any(key.startswith("target@") for key in fake.objects)


def test_cached_materializer_rejects_layout_and_unknown_cost_before_store_io(
    tmp_path: Path,
) -> None:
    request = _request(tmp_path)
    request["source_lookup_record"]["shapes"] = [[2, 2, 2, 2]]
    fake = FakeMooncakeStore(_source_objects(request))

    result = materialize_cached_qwen3(
        request,
        store_factory=lambda: fake,
        bridge_loader=lambda *args, **kwargs: FakeBridge(),
        key_builder=_key_builder,
    )

    assert result["fallback_reason"] == "invalid_materializer_request"
    assert fake.closed is False

    request = _request(tmp_path)
    request["prompt_binding"]["source_response_id"] = "different-request"
    result = materialize_cached_qwen3(
        request,
        store_factory=lambda: fake,
        bridge_loader=lambda *args, **kwargs: FakeBridge(),
        key_builder=_key_builder,
    )
    assert result["fallback_reason"] == "invalid_materializer_request"
    assert fake.closed is False

    request = _request(tmp_path)
    request.pop("native_target_prefill_ms")
    result = materialize_cached_qwen3(
        request,
        store_factory=lambda: fake,
        bridge_loader=lambda *args, **kwargs: FakeBridge(),
        key_builder=_key_builder,
    )
    assert result["fallback_reason"] == "invalid_materializer_request"
    assert fake.closed is False


def test_resident_preload_uses_strict_bridge_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    bridge = FakeBridge()

    class _Cache:
        def load(self, *args, **kwargs):
            assert args == ("/artifacts/bridge.json",)
            assert kwargs["source_model_path"] == "/models/Qwen3-8B"
            assert kwargs["target_model_path"] == "/models/Qwen3-14B"
            return bridge, True

    monkeypatch.setattr(cross_model_materializer, "_RESIDENT_BRIDGE_CACHE", _Cache())
    result = preload_cached_qwen3_bridge(
        {
            "bridge_manifest_path": "/artifacts/bridge.json",
            "source_model_path": "/models/Qwen3-8B",
            "target_model_path": "/models/Qwen3-14B",
            "direction": "8b_to_14b",
            "device": "cpu",
        }
    )

    assert result["success"] is True
    assert result["materialized"] is False
    assert result["injected"] is False
    assert result["artifact_cache"] == {"hit": True, "resident_loader": True}


def test_jsonl_worker_isolates_invalid_requests_and_continues() -> None:
    requests = io.StringIO('{"mode":"unknown"}\nnot-json\n[]\n')
    responses = io.StringIO()

    assert serve_materializer_jsonl(requests, responses) == 0

    parsed = [json.loads(line) for line in responses.getvalue().splitlines()]
    assert [item["fallback_reason"] for item in parsed] == [
        "invalid_materializer_mode",
        "invalid_jsonl_request",
        "invalid_jsonl_request",
    ]
    assert parsed[1]["line_number"] == 2
    assert parsed[2]["line_number"] == 3


def test_cost_benchmark_uses_exact_io_and_never_publishes_targets(tmp_path: Path) -> None:
    request = _request(tmp_path)
    fake = FakeMooncakeStore(_source_objects(request))
    bridge = FakeBridge()
    bridge.manifest.approved = False
    bridge.manifest.artifact_errors = lambda: []
    bridge.manifest.weights_sha256 = "a" * 64
    bridge.manifest.source.weights_sha256 = "b" * 64
    bridge.manifest.target.weights_sha256 = "c" * 64
    bridge.manifest.validation_dataset_sha256 = "d" * 64
    manifest_path = tmp_path / "candidate.json"
    manifest_path.write_text("{}", encoding="utf-8")

    report = run_cached_kv_cost_benchmark(
        bridge,
        candidate_manifest_path=manifest_path,
        setup_config={"metadata_server": "http://metadata"},
        source_keys=["source@0xaa", "source@0xbb"],
        chunk_size=2,
        native_prefill_samples_ms=[100.0, 101.0],
        iterations=2,
        warmup_iterations=1,
        store_factory=lambda: fake,
    )

    assert report["eligible_for_approval"] is False
    assert report["store_backend"] == "test_double"
    assert report["non_publishing"] is True
    assert report["external_index_published"] is False
    assert report["all_temporary_targets_rolled_back"] is True
    assert len(report["measurements_ms"]["read_transform_put"]) == 2
    assert not any(key.startswith("ge-cost/") for key in fake.objects)


def test_native_prefill_evidence_binds_target_identity_and_runtime(tmp_path: Path) -> None:
    bridge = FakeBridge()
    bridge.manifest.target.weights_sha256 = "c" * 64
    report_path = tmp_path / "native.json"
    report_path.write_text(
        json.dumps(
            {
                "schema_version": NATIVE_PREFILL_COST_SCHEMA_VERSION,
                "direction": "8b_to_14b",
                "target_model_weights_sha256": "c" * 64,
                "token_count": 512,
                "backend": "vllm_native_target",
                "eligible_for_approval": True,
                "model_identity_verified": True,
                "prefix_caching_disabled": True,
                "exact_token_count_verified": True,
                "samples_ms": [100.0] * 20,
            }
        ),
        encoding="utf-8",
    )

    evidence = load_native_prefill_evidence(
        report_path,
        bridge=bridge,
        expected_tokens=512,
    )

    assert evidence.eligible_for_approval is True
    assert evidence.backend == "vllm_native_target"
    assert len(evidence.samples_ms) == 20
    assert len(evidence.report_sha256) == 64

    with pytest.raises(ValueError, match="token count mismatch"):
        load_native_prefill_evidence(
            report_path,
            bridge=bridge,
            expected_tokens=256,
        )


def test_native_prefill_report_requires_isolated_vllm_evidence() -> None:
    report = build_native_prefill_report(
        direction="8b_to_14b",
        target_model_weights_sha256="c" * 64,
        token_count=512,
        samples_ms=[100.0] * 20,
        warmup_iterations=3,
        model_identity_verified=True,
        prefix_caching_disabled=True,
        exact_token_count_verified=True,
    )

    assert report["eligible_for_approval"] is True
    assert report["p95_target_prefill_ms"] == 100.0

    report = build_native_prefill_report(
        direction="8b_to_14b",
        target_model_weights_sha256="c" * 64,
        token_count=512,
        samples_ms=[100.0] * 20,
        warmup_iterations=3,
        model_identity_verified=True,
        prefix_caching_disabled=False,
        exact_token_count_verified=True,
    )
    assert report["eligible_for_approval"] is False


def test_cost_evidence_recomputes_p95_and_binds_exact_weights(tmp_path: Path) -> None:
    report_path = tmp_path / "cost.json"
    report = {
        "schema_version": CACHED_KV_COST_SCHEMA_VERSION,
        "direction": "8b_to_14b",
        "weights_sha256": "a" * 64,
        "source_model_weights_sha256": "b" * 64,
        "target_model_weights_sha256": "c" * 64,
        "validation_dataset_sha256": "d" * 64,
        "store_backend": "mooncake_store",
        "native_prefill_backend": "vllm_native_target",
        "eligible_for_approval": True,
        "non_publishing": True,
        "all_temporary_targets_rolled_back": True,
        "external_index_published": False,
        "candidate_manifest_sha256": "e" * 64,
        "native_prefill_report_sha256": "f" * 64,
        "source_keys_sha256": "1" * 64,
        "setup_config_sha256": "2" * 64,
        "iterations": 20,
        "native_prefill_samples": 20,
        "p95_source_read_transform_put_ms": 10.0,
        "p95_target_prefill_ms": 100.0,
        "p95_materialization_to_prefill_ratio": 0.1,
        "measurements_ms": {
            "read_transform_put": [10.0] * 20,
            "native_target_prefill": [100.0] * 20,
        },
    }
    report_path.write_text(json.dumps(report), encoding="utf-8")

    evidence = load_cached_kv_cost_evidence(
        report_path,
        direction="8b_to_14b",
        weights_sha256="a" * 64,
        source_model_weights_sha256="b" * 64,
        target_model_weights_sha256="c" * 64,
        validation_dataset_sha256="d" * 64,
    )

    assert evidence["p95_source_read_transform_put_ms"] == 10.0
    assert evidence["p95_target_prefill_ms"] == 100.0
    assert evidence["cost_candidate_manifest_sha256"] == "e" * 64
    assert len(evidence["cost_report_sha256"]) == 64

    report["p95_source_read_transform_put_ms"] = 9.0
    report_path.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(ValueError, match="materialization P95 is inconsistent"):
        load_cached_kv_cost_evidence(
            report_path,
            direction="8b_to_14b",
            weights_sha256="a" * 64,
            source_model_weights_sha256="b" * 64,
            target_model_weights_sha256="c" * 64,
            validation_dataset_sha256="d" * 64,
        )
