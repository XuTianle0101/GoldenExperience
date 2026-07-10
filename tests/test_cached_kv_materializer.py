import ctypes
import io
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

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
            "chunk_size": 2,
        },
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
