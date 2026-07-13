import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest
import torch
from safetensors import torch as safetensors_torch

import goldenexperience.benchmarks.publication as publication_module
from goldenexperience.benchmarks.publication import (
    PREFIX_BUCKETS,
    SPLIT_COUNTS,
    DatasetSource,
    GroupedPrefixRecord,
    PublicationBenchmarkManifest,
)
from goldenexperience.cli.v5_pipeline import build_parser
from goldenexperience.size_variant.attention_collection import causal_sample_mask
from goldenexperience.size_variant.cached_kv_manifest import CachedKVModelSpec, sha256_file
from goldenexperience.size_variant.head_aware_transport import attention_distillation_terms
from goldenexperience.size_variant.v5_collect import (
    RAW_BENCHMARK_SAMPLE_SCHEMA,
    CollectedTrace,
    TraceObjectRef,
    TraceRecord,
    V5TraceManifest,
    _sampled_attention_output,
    load_raw_sample_store,
    load_trace_shard,
    publication_sample_content_sha256,
    run_collect_stage,
    trace_shard_metadata,
)
from goldenexperience.size_variant.v5_pipeline import (
    V5DirectionConfig,
    V5PipelineConfig,
    V5PipelineError,
    V5PipelineWorkspace,
)


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


@pytest.fixture
def tiny_contract(monkeypatch):
    for split in tuple(SPLIT_COUNTS):
        monkeypatch.setitem(SPLIT_COUNTS, split, 4)
    monkeypatch.setattr(publication_module, "REQUIRED_DATASETS", frozenset({"gsm8k"}))


@pytest.fixture
def grouped_contract(monkeypatch):
    for split in tuple(SPLIT_COUNTS):
        monkeypatch.setitem(SPLIT_COUNTS, split, 8)
    monkeypatch.setattr(publication_module, "REQUIRED_DATASETS", frozenset({"gsm8k"}))


def _model(model_id: str, size: float, layers: int) -> CachedKVModelSpec:
    return CachedKVModelSpec(
        model_id=model_id,
        parameter_count_b=size,
        revision="test",
        architecture="qwen3",
        config_sha256=_digest(model_id + "-config"),
        tokenizer_sha256=_digest("tokenizer"),
        weights_sha256=_digest(model_id + "-weights"),
        num_layers=layers,
        num_key_value_heads=8,
        head_dim=128,
        dtype="bfloat16",
        rope_theta=1_000_000,
        max_position_embeddings=40960,
        chat_template_sha256=_digest("chat"),
    )


def _raw_payload(record: GroupedPrefixRecord) -> dict:
    prefix = f"prefix for {record.prefix_group_id}"
    suffix = f"query for {record.sample_id}"
    reference = f"answer for {record.sample_id}"
    evaluation = {"metric": "exact_match"}
    return {
        "schema_version": RAW_BENCHMARK_SAMPLE_SCHEMA,
        "sample_id": record.sample_id,
        "prefix_text": prefix,
        "suffix_query": suffix,
        "reference": reference,
        "evaluation": evaluation,
        "provenance": {"row": record.sample_id},
    }


def _benchmark(
    tmp_path: Path,
    *,
    grouped_prefixes: bool = False,
) -> tuple[PublicationBenchmarkManifest, Path]:
    records = []
    for split, count in SPLIT_COUNTS.items():
        for index in range(count):
            sample_id = f"{split}-{index}"
            group_index = index // 2 if grouped_prefixes else index
            group = f"{split}-group-{group_index}"
            provisional = GroupedPrefixRecord(
                sample_id=sample_id,
                split=split,
                dataset_id="gsm8k",
                prefix_group_id=group,
                prefix_sha256="",
                suffix_query_sha256="",
                content_sha256="",
                token_bucket=PREFIX_BUCKETS[group_index % len(PREFIX_BUCKETS)],
                task="qa",
            )
            raw = _raw_payload(provisional)
            records.append(
                replace(
                    provisional,
                    prefix_sha256=_digest(raw["prefix_text"]),
                    suffix_query_sha256=_digest(raw["suffix_query"]),
                    content_sha256=publication_sample_content_sha256(
                        prefix_text=raw["prefix_text"],
                        suffix_query=raw["suffix_query"],
                        reference=raw["reference"],
                        evaluation=raw["evaluation"],
                        task=provisional.task,
                    ),
                )
            )
    source = DatasetSource(
        dataset_id="gsm8k",
        revision="test",
        content_sha256=_digest("gsm8k"),
        license_id="MIT",
        license_uri="https://example.invalid/license",
        source_uri="https://example.invalid/gsm8k",
    )
    provisional_manifest = PublicationBenchmarkManifest(
        sources=(source,),
        records=tuple(records),
        split_sha256={},
        tokenizer_sha256=_digest("tokenizer"),
        chat_template_sha256=_digest("chat"),
        sealed_payload_sha256=_digest("sealed"),
    )
    manifest = replace(
        provisional_manifest,
        split_sha256={
            split: provisional_manifest.compute_split_sha256(split) for split in SPLIT_COUNTS
        },
    )
    assert manifest.validate() == []
    path = tmp_path / "benchmark.json"
    manifest.save(path)
    return manifest, path


def _workspace(
    tmp_path: Path,
    manifest: PublicationBenchmarkManifest,
    manifest_path: Path,
) -> V5PipelineWorkspace:
    models = {
        "4b": _model("Qwen/Qwen3-4B", 4.0, 36),
        "8b": _model("Qwen/Qwen3-8B", 8.0, 36),
        "14b": _model("Qwen/Qwen3-14B", 14.0, 40),
    }
    pairs = {
        "qwen3_4b_to_8b": ("4b", "8b"),
        "qwen3_8b_to_4b": ("8b", "4b"),
        "qwen3_8b_to_14b": ("8b", "14b"),
        "qwen3_14b_to_8b": ("14b", "8b"),
    }
    directions = tuple(
        V5DirectionConfig(
            direction=direction,
            source_model_path=str(tmp_path / source_size),
            target_model_path=str(tmp_path / target_size),
            source=models[source_size],
            target=models[target_size],
        )
        for direction, (source_size, target_size) in pairs.items()
    )
    config = V5PipelineConfig.from_benchmark(
        manifest,
        manifest_path=manifest_path,
        code_sha256=_digest("code"),
        directions=directions,
    )
    return V5PipelineWorkspace.create(tmp_path / "workspace", config)


def _sample_store(
    tmp_path: Path,
    manifest: PublicationBenchmarkManifest,
    split: str,
) -> Path:
    path = tmp_path / f"{split}.jsonl"
    lines = [
        json.dumps(_raw_payload(record), sort_keys=True)
        for record in manifest.records
        if record.split == split
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


class _FakeCollector:
    def __init__(self, *, fail_after: int | None = None) -> None:
        self.fail_after = fail_after
        self.calls: list[str] = []
        self.entered = 0

    def __enter__(self):
        self.entered += 1
        return self

    def __exit__(self, *_args):
        return None

    def collect(self, record, sample, output):
        del sample
        self.calls.append(record.sample_id)
        if self.fail_after is not None and len(self.calls) > self.fail_after:
            raise RuntimeError("synthetic collection interruption")
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(record.sample_id.encode("utf-8"))
        return CollectedTrace(
            path=output,
            token_ids_sha256=_digest(record.sample_id + "-tokens"),
            token_count=record.token_bucket,
            query_sample_count=min(32, record.token_bucket),
            key_sample_count=min(256, record.token_bucket),
        )


def test_raw_sample_hash_excludes_provenance_but_binds_evaluation() -> None:
    common = {
        "prefix_text": "prefix",
        "suffix_query": "query",
        "reference": "answer",
        "task": "qa",
    }
    first = publication_sample_content_sha256(
        **common,
        evaluation={"metric": "exact_match"},
    )
    second = publication_sample_content_sha256(
        **common,
        evaluation={"metric": "contains"},
    )

    assert first != second


def test_sampled_native_output_matches_attention_loss_key_domain() -> None:
    query = torch.randn(2, 4, 3, 8)
    key = torch.randn(2, 2, 5, 8)
    value = torch.randn(2, 2, 5, 8)
    mask = torch.ones(1, 1, 3, 5, dtype=torch.bool)
    native_output = _sampled_attention_output(query, key, value, mask)

    kl, mse = attention_distillation_terms(
        query,
        key,
        value,
        key,
        value,
        attention_mask=mask,
        native_attention_output=native_output,
    )

    assert kl.item() == pytest.approx(0.0, abs=1e-6)
    assert mse.item() == pytest.approx(0.0, abs=1e-6)


def test_trace_shard_loader_verifies_tensor_and_metadata_contract(tmp_path: Path) -> None:
    source = replace(
        _model("source", 4.0, 2),
        num_key_value_heads=1,
        head_dim=4,
    )
    target = replace(
        _model("target", 8.0, 3),
        num_key_value_heads=1,
        head_dim=4,
    )
    provisional = TraceRecord(
        sample_id="sample",
        prefix_group_id="group",
        dataset_id="gsm8k",
        task="qa",
        token_bucket=128,
        content_sha256=_digest("content"),
        prefix_sha256=_digest("prefix"),
        suffix_query_sha256=_digest("suffix"),
        token_ids_sha256=_digest("tokens"),
        token_count=128,
        query_sample_count=3,
        key_sample_count=5,
        shard=TraceObjectRef("0" * 64, "objects/00/placeholder.safetensors", 1),
    )
    queries = torch.randn(3, 2, 3, 4, dtype=torch.bfloat16)
    query_positions = torch.tensor([0, 64, 127])
    key_positions = torch.tensor([0, 32, 64, 96, 127])
    tensors = {
        "source_kv": torch.randn(2, 2, 1, 5, 4, dtype=torch.bfloat16),
        "target_kv": torch.randn(2, 3, 1, 5, 4, dtype=torch.bfloat16),
        "target_query": queries,
        "native_attention_output": torch.randn(3, 2, 3, 4),
        "full_native_attention_output": torch.randn(3, 2, 3, 4, dtype=torch.bfloat16),
        "query_positions": query_positions,
        "key_positions": key_positions,
        "causal_mask": causal_sample_mask(query_positions, key_positions),
        "constant_losses": torch.tensor([1.0, 0.1]),
    }
    path = tmp_path / "shard.safetensors"
    safetensors_torch.save_file(
        tensors,
        path,
        metadata=trace_shard_metadata(provisional, source, target),
    )
    digest = sha256_file(path)
    record = replace(
        provisional,
        shard=TraceObjectRef(
            digest, f"objects/{digest[:2]}/{digest}.safetensors", path.stat().st_size
        ),
    )
    path.chmod(0o444)

    loaded = load_trace_shard(path, record, source=source, target=target)

    assert set(loaded) == set(tensors)
    torch.testing.assert_close(loaded["source_kv"], tensors["source_kv"])


def test_raw_store_is_exactly_one_nonsealed_split(tmp_path: Path, tiny_contract) -> None:
    manifest, _ = _benchmark(tmp_path)
    store = _sample_store(tmp_path, manifest, "transport_train")

    loaded = load_raw_sample_store(store, manifest, split="transport_train")

    assert len(loaded) == 4
    with store.open("a", encoding="utf-8") as handle:
        validation = next(record for record in manifest.records if record.split == "validation")
        handle.write(json.dumps(_raw_payload(validation)) + "\n")
    with pytest.raises(V5PipelineError, match="outside requested split"):
        load_raw_sample_store(store, manifest, split="transport_train")
    with pytest.raises(V5PipelineError, match="not available"):
        load_raw_sample_store(store, manifest, split="semantic_sealed_test")


def test_collect_stage_checkpoints_and_publishes_trace_manifest(
    tmp_path: Path,
    tiny_contract,
) -> None:
    manifest, manifest_path = _benchmark(tmp_path)
    workspace = _workspace(tmp_path, manifest, manifest_path)
    store = _sample_store(tmp_path, manifest, "transport_train")
    collector = _FakeCollector()

    stage = run_collect_stage(
        workspace=workspace,
        direction="qwen3_4b_to_8b",
        split="transport_train",
        sample_store_path=store,
        collector_parameters={"collector_id": "fake-v1"},
        collector_factory=lambda: collector,
    )

    assert stage.status == "completed"
    assert len(collector.calls) == 4
    assert stage.outputs is not None
    trace_path = workspace.artifact_path(stage.outputs["trace_manifest"])
    trace = V5TraceManifest.load(trace_path, workspace=workspace, benchmark=manifest)
    assert len(trace.records) == 4
    assert trace.raw_sample_store_sha256 == sha256_file(store)

    reused_collector = _FakeCollector()
    reused = run_collect_stage(
        workspace=workspace,
        direction="qwen3_4b_to_8b",
        split="transport_train",
        sample_store_path=store,
        collector_parameters={"collector_id": "fake-v1"},
        collector_factory=lambda: reused_collector,
    )
    assert reused.receipt_sha256 == stage.receipt_sha256
    assert reused_collector.entered == 0


def test_collect_stage_materializes_one_shard_per_prefix_group(
    tmp_path: Path,
    grouped_contract,
) -> None:
    manifest, manifest_path = _benchmark(tmp_path, grouped_prefixes=True)
    workspace = _workspace(tmp_path, manifest, manifest_path)
    store = _sample_store(tmp_path, manifest, "transport_train")
    collector = _FakeCollector()

    stage = run_collect_stage(
        workspace=workspace,
        direction="qwen3_4b_to_8b",
        split="transport_train",
        sample_store_path=store,
        collector_parameters={"collector_id": "fake-v2"},
        collector_factory=lambda: collector,
    )

    assert len(collector.calls) == 4
    assert stage.outputs is not None
    trace = V5TraceManifest.load(
        workspace.artifact_path(stage.outputs["trace_manifest"]),
        workspace=workspace,
        benchmark=manifest,
    )
    grouped: dict[str, set[str]] = {}
    for record in trace.records:
        grouped.setdefault(record.prefix_group_id, set()).add(record.shard.sha256)
    assert len(grouped) == 4
    assert all(len(shards) == 1 for shards in grouped.values())
    first_group = trace.records[0].prefix_group_id
    second_group = next(
        record.prefix_group_id for record in trace.records if record.prefix_group_id != first_group
    )
    first_shard = trace.records[0].shard
    incompatible = replace(
        trace,
        records=tuple(
            replace(record, shard=first_shard) if record.prefix_group_id == second_group else record
            for record in trace.records
        ),
    )
    assert "one trace object is bound to multiple prefix identities" in incompatible.validate(
        workspace=workspace,
        benchmark=manifest,
    )


def test_collect_resume_reuses_a_completed_peer_checkpoint_in_the_same_group(
    tmp_path: Path,
    grouped_contract,
) -> None:
    manifest, manifest_path = _benchmark(tmp_path, grouped_prefixes=True)
    workspace = _workspace(tmp_path, manifest, manifest_path)
    store = _sample_store(tmp_path, manifest, "transport_train")
    interrupted = _FakeCollector()

    def interrupt_after_peer_checkpoint(
        completed: int,
        _total: int,
        _sample_id: str,
    ) -> None:
        if completed == 3:
            raise RuntimeError("synthetic checkpoint interruption")

    with pytest.raises(RuntimeError, match="checkpoint interruption"):
        run_collect_stage(
            workspace=workspace,
            direction="qwen3_4b_to_8b",
            split="transport_train",
            sample_store_path=store,
            collector_parameters={"collector_id": "fake-v2"},
            collector_factory=lambda: interrupted,
            progress=interrupt_after_peer_checkpoint,
        )

    resumed = _FakeCollector()
    stage = run_collect_stage(
        workspace=workspace,
        direction="qwen3_4b_to_8b",
        split="transport_train",
        sample_store_path=store,
        collector_parameters={"collector_id": "fake-v2"},
        collector_factory=lambda: resumed,
        resume=True,
    )

    assert stage.status == "completed"
    assert len(interrupted.calls) == 2
    assert len(resumed.calls) == 2


def test_collect_resume_reuses_verified_per_sample_checkpoints(
    tmp_path: Path,
    tiny_contract,
) -> None:
    manifest, manifest_path = _benchmark(tmp_path)
    workspace = _workspace(tmp_path, manifest, manifest_path)
    store = _sample_store(tmp_path, manifest, "selector_train")
    interrupted = _FakeCollector(fail_after=2)

    with pytest.raises(RuntimeError, match="interruption"):
        run_collect_stage(
            workspace=workspace,
            direction="qwen3_8b_to_4b",
            split="selector_train",
            sample_store_path=store,
            collector_parameters={"collector_id": "fake-v1"},
            collector_factory=lambda: interrupted,
        )

    resumed_collector = _FakeCollector()
    stage = run_collect_stage(
        workspace=workspace,
        direction="qwen3_8b_to_4b",
        split="selector_train",
        sample_store_path=store,
        collector_parameters={"collector_id": "fake-v1"},
        collector_factory=lambda: resumed_collector,
        resume=True,
    )

    assert stage.status == "completed"
    assert len(interrupted.calls) == 3
    assert len(resumed_collector.calls) == 2


def test_collect_cli_never_exposes_semantic_sealed_split() -> None:
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "collect",
                "--workspace",
                "workspace",
                "--direction",
                "qwen3_4b_to_8b",
                "--split",
                "semantic_sealed_test",
                "--samples",
                "sealed.jsonl",
            ]
        )
