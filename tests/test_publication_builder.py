import hashlib
import json
import stat
from pathlib import Path

import pytest

import goldenexperience.benchmarks.publication as publication_module
import goldenexperience.benchmarks.publication_builder as builder_module
import goldenexperience.benchmarks.publication_sources as sources_module
from goldenexperience.benchmarks.publication import (
    PREFIX_BUCKETS,
    SPLIT_COUNTS,
    DatasetSource,
    PublicationBenchmarkManifest,
)
from goldenexperience.benchmarks.publication_builder import (
    AllocationCell,
    PublicationBuildError,
    PublicationDatasetBuilder,
    publication_allocation,
    publish_publication_build,
)
from goldenexperience.benchmarks.publication_sources import (
    AuditedPublicationSources,
    BurstWorkloadRow,
    CanonicalSourceExample,
    LoadedPublicationSources,
    LockedDatasetSource,
    PublicationSourceError,
    PublicationSourceLock,
    ResolvedSourceFile,
    SourceFileLock,
    _json_array,
    _load_bfcl,
    _load_burstgpt,
    _load_gsm8k,
    _load_humaneval,
    _load_longbench,
    _load_math,
    _load_mbpp,
    _load_sharegpt,
    audit_publication_sources,
    source_file_merkle_sha256,
)
from goldenexperience.cli.publication_benchmark import _parser, _source_path_overrides


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def test_default_publication_allocation_is_balanced_and_source_stratified() -> None:
    allocation = publication_allocation()

    for split, expected in SPLIT_COUNTS.items():
        rows = [cell for cell in allocation if cell.split == split]
        assert sum(cell.count for cell in rows) == expected
        for bucket in PREFIX_BUCKETS:
            assert sum(cell.count for cell in rows if cell.token_bucket == bucket) == (
                expected // len(PREFIX_BUCKETS)
            )
    assert sum(cell.count for cell in allocation if cell.dataset_id == "longbench_qasper") == 144
    assert {cell.dataset_id for cell in allocation if cell.split == "runtime_audit"} == {
        "sharegpt",
        "burstgpt",
    }


def test_source_lock_merkle_audit_and_streaming_json_reader(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(sources_module, "REQUIRED_DATASETS", frozenset({"gsm8k"}))
    root = tmp_path / "sources"
    root.mkdir()
    train = root / "train.jsonl"
    test = root / "test.jsonl"
    train.write_text('{"question":"q","answer":"#### 1"}\n', encoding="utf-8")
    test.write_text('{"question":"q2","answer":"#### 2"}\n', encoding="utf-8")
    files = tuple(
        SourceFileLock(
            role=role,
            path=path.name,
            sha256=_digest(path.read_text()),
            size_bytes=path.stat().st_size,
        )
        for role, path in (("train", train), ("test", test))
    )
    source = DatasetSource(
        dataset_id="gsm8k",
        revision="test",
        content_sha256=source_file_merkle_sha256(files),
        license_id="MIT",
        license_uri="https://example.invalid/license",
        source_uri="https://example.invalid/source",
    )
    lock = PublicationSourceLock(
        sources=(LockedDatasetSource(source=source, adapter="gsm8k", files=files),),
        tokenizer_sha256=_digest("tokenizer"),
        chat_template_sha256=_digest("chat"),
    )
    lock_path = tmp_path / "lock.json"
    lock_path.write_text(json.dumps(lock.to_dict()), encoding="utf-8")

    loaded_lock = PublicationSourceLock.load(lock_path)
    audited = audit_publication_sources(loaded_lock, source_root=root)
    assert len(audited.files) == 2
    assert audited.public_manifest()["source_lock_sha256"] == loaded_lock.content_sha256()
    with pytest.raises(PublicationSourceError, match="unknown"):
        audit_publication_sources(
            loaded_lock,
            source_root=root,
            path_overrides={("gsm8k", "missing"): train},
        )

    train.write_text('{"question":"changed","answer":"#### 1"}\n', encoding="utf-8")
    with pytest.raises(PublicationSourceError, match="changed"):
        audited.verify_unchanged()

    array = tmp_path / "array.json"
    array.write_text('[{"id":1}, {"id":2}]\n', encoding="utf-8")
    assert list(_json_array(array, chunk_size=3)) == [{"id": 1}, {"id": 2}]


def _audited_files(files: dict[tuple[str, str], Path]) -> AuditedPublicationSources:
    resolved = []
    for (dataset_id, role), path in files.items():
        file_stat = path.stat()
        resolved.append(
            ResolvedSourceFile(
                dataset_id=dataset_id,
                role=role,
                logical_path=path.name,
                path=path,
                sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                size_bytes=file_stat.st_size,
                signature=(
                    file_stat.st_dev,
                    file_stat.st_ino,
                    file_stat.st_size,
                    file_stat.st_mtime_ns,
                    file_stat.st_ctime_ns,
                ),
            )
        )
    return AuditedPublicationSources(
        lock=PublicationSourceLock(
            sources=(),
            tokenizer_sha256=_digest("tokenizer"),
            chat_template_sha256=_digest("chat"),
            selection_seed=17,
        ),
        files=tuple(resolved),
    )


def test_publication_source_adapters_canonicalize_registered_formats(tmp_path: Path) -> None:
    longbench = tmp_path / "longbench.jsonl"
    longbench.write_text(
        json.dumps(
            {
                "_id": "lb-1",
                "dataset": "hotpotqa",
                "input": "question",
                "context": "long context",
                "answers": ["answer"],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    gsm_train = tmp_path / "gsm-train.jsonl"
    gsm_test = tmp_path / "gsm-test.jsonl"
    gsm_train.write_text('{"question":"one plus one","answer":"work\\n#### 2"}\n')
    gsm_test.write_text('{"question":"two plus two","answer":"work\\n#### 4"}\n')
    humaneval = tmp_path / "humaneval.jsonl"
    humaneval.write_text(
        json.dumps(
            {
                "task_id": "HumanEval/0",
                "prompt": "def add(a, b):\n",
                "entry_point": "add",
                "test": "def check(candidate):\n    assert candidate(1, 2) == 3",
                "canonical_solution": "    return a + b\n",
            }
        )
        + "\n"
    )
    mbpp = tmp_path / "mbpp.jsonl"
    mbpp.write_text(
        json.dumps(
            {
                "task_id": 1,
                "text": "add two numbers",
                "code": "def add(a, b):\n    return a + b",
                "test_setup_code": "",
                "test_list": ["assert add(1, 2) == 3"],
            }
        )
        + "\n"
    )
    bfcl_questions = tmp_path / "bfcl-questions.jsonl"
    bfcl_answers = tmp_path / "bfcl-answers.jsonl"
    bfcl_questions.write_text(
        json.dumps(
            {
                "id": "simple_0",
                "question": [[{"role": "user", "content": "add one and two"}]],
                "function": [{"name": "add", "parameters": {}}],
            }
        )
        + "\n"
    )
    bfcl_answers.write_text(
        json.dumps(
            {
                "id": "simple_0",
                "ground_truth": [{"add": {"a": [1], "b": [2]}}],
            }
        )
        + "\n"
    )
    sharegpt = tmp_path / "sharegpt.json"
    sharegpt.write_text(
        json.dumps(
            [
                {
                    "id": "chat-1",
                    "conversations": [
                        {"from": "human", "value": "hello"},
                        {"from": "gpt", "value": "hi"},
                    ],
                }
            ]
        )
    )
    audited = _audited_files(
        {
            ("longbench_hotpotqa", "data"): longbench,
            ("gsm8k", "train"): gsm_train,
            ("gsm8k", "test"): gsm_test,
            ("humaneval", "data"): humaneval,
            ("mbpp", "data"): mbpp,
            ("bfcl", "questions"): bfcl_questions,
            ("bfcl", "answers"): bfcl_answers,
            ("sharegpt", "data"): sharegpt,
        }
    )

    assert [row.row_id for row in _load_longbench(audited, "longbench_hotpotqa", 17)] == ["lb-1"]
    assert [row.reference for row in _load_gsm8k(audited, "gsm8k", 17)] == ["2", "4"]
    assert next(iter(_load_humaneval(audited, "humaneval", 17))).reference["entry_point"] == ("add")
    assert next(iter(_load_mbpp(audited, "mbpp", 17))).reference["test_mode"] == "exec"
    assert next(iter(_load_bfcl(audited, "bfcl", 17))).evaluation == {"metric": "function_call"}
    assert next(iter(_load_sharegpt(audited, "sharegpt", 17))).query == "hello"


def test_math_and_burst_adapters_preserve_train_test_and_bucket_provenance(
    tmp_path: Path,
) -> None:
    parquet = pytest.importorskip("pyarrow.parquet")
    arrow = pytest.importorskip("pyarrow")
    files: dict[tuple[str, str], Path] = {}
    for category in sources_module._MATH_CATEGORIES:
        for split in ("train", "test"):
            path = tmp_path / f"{category}_{split}.parquet"
            parquet.write_table(
                arrow.table(
                    {
                        "problem": [f"problem {category} {split}", "unscored"],
                        "solution": ["work \\boxed{1}", "no final marker"],
                    }
                ),
                path,
            )
            files[("math", f"{category}_{split}")] = path
    audited = _audited_files(files)
    math_rows = tuple(_load_math(audited, "math", 17))
    assert len(math_rows) == len(sources_module._MATH_CATEGORIES) * 2
    assert {row.source_split for row in math_rows} == {"train", "test"}

    burst = tmp_path / "burst.csv"
    lines = ["Timestamp,Model,Request tokens,Response tokens,Total tokens,Log Type"]
    for bucket in PREFIX_BUCKETS:
        for index in range(64):
            lines.append(f"{bucket + index},test,{bucket},1,{bucket + 1},API log")
    burst.write_text("\n".join(lines) + "\n")
    burst_rows = tuple(_load_burstgpt(burst, 17))
    assert len(burst_rows) == 256
    assert {row.token_bucket for row in burst_rows} == set(PREFIX_BUCKETS)


class _CharacterTokenizer:
    semantic_sha256 = _digest("tokenizer")
    chat_template_sha256 = _digest("chat")

    @staticmethod
    def encode(text: str) -> list[int]:
        return [index for index in range(0, len(text), 2)]

    @classmethod
    def exact_prefix(cls, text: str, token_bucket: int) -> tuple[str, list[int]]:
        values = cls.encode(text)
        if len(values) < token_bucket:
            raise PublicationBuildError("short test prefix")
        return text[: token_bucket * 2], values[:token_bucket]


class _WrongTokenizer(_CharacterTokenizer):
    semantic_sha256 = _digest("wrong-tokenizer")


def _source(dataset_id: str) -> LockedDatasetSource:
    return LockedDatasetSource(
        source=DatasetSource(
            dataset_id=dataset_id,
            revision="test",
            content_sha256=_digest(dataset_id),
            license_id="test",
            license_uri="https://example.invalid/license",
            source_uri="https://example.invalid/source",
            usage="trace_only" if dataset_id in {"sharegpt", "burstgpt"} else "semantic",
        ),
        adapter=sources_module.SOURCE_ADAPTERS[dataset_id],
        files=(),
    )


def _examples(dataset_id: str, count: int = 80) -> tuple[CanonicalSourceExample, ...]:
    rows = []
    for index in range(count):
        source_split = "test" if index % 3 == 0 else "train"
        reference = None if dataset_id == "sharegpt" else f"answer-{dataset_id}-{index}"
        evaluation = {} if dataset_id == "sharegpt" else {"metric": "exact_match"}
        rows.append(
            CanonicalSourceExample(
                dataset_id=dataset_id,
                row_id=f"{dataset_id}-{index:03d}",
                source_split=source_split,
                query=f"unique query {dataset_id} {index}",
                context=(f"context {dataset_id} {index} " * 800),
                reference=reference,
                evaluation=evaluation,
                task="trace" if dataset_id == "sharegpt" else "qa",
                demonstration=(f"demonstration {dataset_id} {index} " * 800),
                source_role="data",
            )
        )
    return tuple(rows)


def _tiny_builder(monkeypatch) -> PublicationDatasetBuilder:
    for split in tuple(SPLIT_COUNTS):
        monkeypatch.setitem(SPLIT_COUNTS, split, 8 if split == "runtime_audit" else 16)
    for dataset_id in tuple(builder_module._DEMO_COUNTS):
        monkeypatch.setitem(builder_module._DEMO_COUNTS, dataset_id, 1)
    semantic_datasets = (
        "bfcl",
        "gsm8k",
        "humaneval",
        "longbench_hotpotqa",
        "longbench_multifieldqa",
        "longbench_qasper",
        "math",
        "mbpp",
    )
    cells = []
    index = 0
    for split in builder_module.SEMANTIC_SPLITS:
        for bucket in PREFIX_BUCKETS:
            cells.append(AllocationCell(split, bucket, semantic_datasets[index % 8], 4))
            index += 1
    for bucket in PREFIX_BUCKETS:
        cells.append(AllocationCell("runtime_audit", bucket, "sharegpt", 1))
        cells.append(AllocationCell("runtime_audit", bucket, "burstgpt", 1))
    lock = PublicationSourceLock(
        sources=tuple(
            _source(dataset_id) for dataset_id in sorted(publication_module.REQUIRED_DATASETS)
        ),
        tokenizer_sha256=_digest("tokenizer"),
        chat_template_sha256=_digest("chat"),
    )
    audited = AuditedPublicationSources(lock=lock, files=())
    examples = {
        dataset_id: _examples(dataset_id)
        for dataset_id in publication_module.REQUIRED_DATASETS
        if dataset_id != "burstgpt"
    }
    burst = tuple(
        BurstWorkloadRow(
            row_id=f"burst-{bucket}",
            timestamp=float(bucket),
            model="test",
            request_tokens=bucket,
            response_tokens=1,
            log_type="API log",
            token_bucket=bucket,
        )
        for bucket in PREFIX_BUCKETS
    )
    return PublicationDatasetBuilder(
        audited_sources=audited,
        loaded_sources=LoadedPublicationSources(examples=examples, burst_rows=burst),
        tokenizer=_CharacterTokenizer(),
        allocation=cells,
    )


def test_tiny_publication_build_publishes_only_hashes_for_sealed_rows(
    tmp_path: Path,
    monkeypatch,
) -> None:
    result = publish_publication_build(
        _tiny_builder(monkeypatch),
        output_dir=tmp_path / "public",
        sealed_payload=tmp_path / "secure" / "sealed.jsonl",
    )

    manifest = PublicationBenchmarkManifest.load(result.manifest_path)
    assert manifest.validate() == []
    assert len(manifest.records) == 6 * 16 + 8
    assert not (result.output_dir / "raw" / "semantic_sealed_test.jsonl").exists()
    assert result.sealed_payload.stat().st_mode & 0o777 == 0o400
    assert result.manifest_path.stat().st_mode & stat.S_IWUSR == 0
    assert json.loads(result.build_report_path.read_text())["isolation"] == {
        "protected_suffix_hashes_disjoint": True,
        "query_rows_globally_unique": True,
        "semantic_sealed_raw_rows_in_public_output": False,
        "trace_only_sources_restricted_to_runtime_audit": True,
        "transport_prefix_family_isolated": True,
    }

    with pytest.raises(PublicationBuildError, match="must not already exist"):
        publish_publication_build(
            _tiny_builder(monkeypatch),
            output_dir=result.output_dir,
            sealed_payload=result.sealed_payload,
        )


def test_publication_builder_rejects_a_different_tokenizer(monkeypatch) -> None:
    valid = _tiny_builder(monkeypatch)

    with pytest.raises(PublicationBuildError, match="tokenizer differs"):
        PublicationDatasetBuilder(
            audited_sources=valid.audited,
            loaded_sources=valid.loaded,
            tokenizer=_WrongTokenizer(),
            allocation=valid.allocation,
        )


def test_publication_builder_cli_parses_portable_path_overrides() -> None:
    args = _parser().parse_args(
        [
            "build",
            "--source-lock",
            "sources.json",
            "--source-root",
            "datasets",
            "--source-path",
            "gsm8k:train=/data/train.jsonl",
            "--tokenizer-model",
            "Qwen3-8B",
            "--output-dir",
            "public",
            "--sealed-output",
            "secure/sealed.jsonl",
        ]
    )

    assert args.command == "build"
    assert _source_path_overrides(args.source_path) == {
        ("gsm8k", "train"): Path("/data/train.jsonl")
    }
    with pytest.raises(ValueError, match="DATASET:ROLE=PATH"):
        _source_path_overrides(["broken"])
