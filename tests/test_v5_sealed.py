from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

import goldenexperience.size_variant.v5_sealed as sealed_module
from goldenexperience.benchmarks.publication import (
    PREFIX_BUCKETS,
    SPLIT_COUNTS,
    GroupedPrefixRecord,
)
from goldenexperience.cli.v5_pipeline import build_parser
from goldenexperience.size_variant.v5_collect import (
    RAW_BENCHMARK_SAMPLE_SCHEMA,
    publication_sample_content_sha256,
)
from goldenexperience.size_variant.v5_pipeline import V5PipelineError, V5PipelineWorkspace
from goldenexperience.size_variant.v5_sealed import (
    V5_SEMANTIC_OPEN_RECEIPT_PATH,
    load_semantic_open_receipt,
    load_semantic_snapshot_bytes,
    open_semantic_sealed_once,
)


def _digest(value: str | bytes) -> str:
    payload = value if isinstance(value, bytes) else value.encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _sealed_rows(count: int) -> tuple[list[GroupedPrefixRecord], list[dict[str, Any]]]:
    records = []
    rows = []
    for index in range(count):
        sample_id = f"sealed-{index:04d}"
        prefix = f"sealed prefix {index}"
        suffix = f"sealed query {index}"
        reference = f"answer {index}"
        evaluation = {"metric": "exact_match"}
        records.append(
            GroupedPrefixRecord(
                sample_id=sample_id,
                split="semantic_sealed_test",
                dataset_id="gsm8k",
                prefix_group_id=f"sealed-group-{index}",
                prefix_sha256=_digest(prefix),
                suffix_query_sha256=_digest(suffix),
                content_sha256=publication_sample_content_sha256(
                    prefix_text=prefix,
                    suffix_query=suffix,
                    reference=reference,
                    evaluation=evaluation,
                    task="qa",
                ),
                token_bucket=PREFIX_BUCKETS[index % len(PREFIX_BUCKETS)],
                task="qa",
            )
        )
        rows.append(
            {
                "schema_version": RAW_BENCHMARK_SAMPLE_SCHEMA,
                "sample_id": sample_id,
                "prefix_text": prefix,
                "suffix_query": suffix,
                "reference": reference,
                "evaluation": evaluation,
                "provenance": {"sealed_row": index},
            }
        )
    return records, rows


def _payload(rows: list[dict[str, Any]]) -> bytes:
    return ("\n".join(json.dumps(item, sort_keys=True) for item in rows) + "\n").encode()


class _FakeWorkspace:
    def __init__(self, root: Path, payload_sha256: str) -> None:
        self.root = root
        self.control = root / ".pipeline"
        self.control.mkdir(parents=True)
        self.sealed_open_path = self.control / "semantic_sealed.opened.json"
        self.config = SimpleNamespace(
            pipeline_id="pipeline",
            benchmark_manifest_sha256=_digest("benchmark"),
            sealed_payload_sha256=payload_sha256,
            code_sha256=_digest("code"),
            split_sha256={
                "validation": _digest("validation"),
                "semantic_sealed_test": _digest("semantic-sealed"),
            },
        )


def _validation(direction: str) -> tuple[Any, Any, Any]:
    threshold = 0.1
    validation = SimpleNamespace(
        direction=direction,
        threshold=threshold,
        risk_calibration_manifest_sha256=_digest(f"{direction}-calibration"),
        validation_report_sha256=_digest(f"{direction}-report"),
        code_sha256=_digest("code"),
        transport_weights_sha256=_digest(f"{direction}-transport"),
        predictor_sha256=_digest(f"{direction}-predictor"),
        passed=True,
        content_sha256=lambda: _digest(f"{direction}-validation-manifest"),
    )
    selective = SimpleNamespace(
        state=sealed_module.ArtifactState.VALIDATION_CANDIDATE,
        artifact_id=f"selective-kv-{_digest(direction)[:24]}",
    )
    return validation, selective, object()


def test_semantic_guard_validates_four_directions_and_publishes_once(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setitem(SPLIT_COUNTS, "semantic_sealed_test", 4)
    records, rows = _sealed_rows(4)
    payload = _payload(rows)
    payload_path = tmp_path / "sealed.jsonl"
    payload_path.write_bytes(payload)
    benchmark = SimpleNamespace(records=tuple(records))
    workspace = _FakeWorkspace(tmp_path / "workspace", _digest(payload))
    calls: list[str] = []
    monkeypatch.setattr(sealed_module, "load_bound_benchmark", lambda _workspace: benchmark)

    def load_validation(_workspace: Any, direction: str) -> tuple[Any, Any, Any]:
        calls.append(direction)
        return _validation(direction)

    monkeypatch.setattr(sealed_module, "load_completed_validation", load_validation)
    receipt, snapshot = open_semantic_sealed_once(
        cast(V5PipelineWorkspace, workspace),
        payload_path,
    )

    assert set(calls) == sealed_module.REQUIRED_QWEN3_DIRECTIONS
    assert len(receipt.directions) == 4
    assert snapshot.read_bytes() == payload
    assert not snapshot.stat().st_mode & 0o222
    assert not (workspace.root / V5_SEMANTIC_OPEN_RECEIPT_PATH).stat().st_mode & 0o222
    marker = json.loads(workspace.sealed_open_path.read_text(encoding="utf-8"))
    assert marker["state"] == "opened"
    assert marker["open_receipt_sha256"] == receipt.content_sha256()
    loaded, loaded_snapshot = load_semantic_open_receipt(cast(V5PipelineWorkspace, workspace))
    assert loaded == receipt
    assert loaded_snapshot == snapshot
    with pytest.raises(V5PipelineError, match="already opened"):
        open_semantic_sealed_once(cast(V5PipelineWorkspace, workspace), payload_path)


def test_semantic_guard_fails_before_open_when_a_validation_is_missing(
    tmp_path: Path,
    monkeypatch,
) -> None:
    records, rows = _sealed_rows(4)
    payload = _payload(rows)
    payload_path = tmp_path / "sealed.jsonl"
    payload_path.write_bytes(payload)
    workspace = _FakeWorkspace(tmp_path / "workspace", _digest(payload))
    monkeypatch.setattr(
        sealed_module,
        "load_bound_benchmark",
        lambda _workspace: SimpleNamespace(records=tuple(records)),
    )

    def missing(_workspace: Any, direction: str) -> tuple[Any, Any, Any]:
        if direction == "qwen3_8b_to_14b":
            raise V5PipelineError("missing passing validation")
        return _validation(direction)

    monkeypatch.setattr(sealed_module, "load_completed_validation", missing)
    with pytest.raises(V5PipelineError, match="missing passing validation"):
        open_semantic_sealed_once(cast(V5PipelineWorkspace, workspace), payload_path)

    assert not workspace.sealed_open_path.exists()
    assert not (workspace.root / V5_SEMANTIC_OPEN_RECEIPT_PATH).exists()


def test_invalid_semantic_payload_consumes_the_one_shot_marker_fail_closed(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setitem(SPLIT_COUNTS, "semantic_sealed_test", 4)
    records, rows = _sealed_rows(4)
    rows.pop()
    payload = _payload(rows)
    payload_path = tmp_path / "invalid-sealed.jsonl"
    payload_path.write_bytes(payload)
    workspace = _FakeWorkspace(tmp_path / "workspace", _digest(payload))
    monkeypatch.setattr(
        sealed_module,
        "load_bound_benchmark",
        lambda _workspace: SimpleNamespace(records=tuple(records)),
    )
    monkeypatch.setattr(
        sealed_module,
        "load_completed_validation",
        lambda _workspace, direction: _validation(direction),
    )

    with pytest.raises(V5PipelineError, match="could not be opened"):
        open_semantic_sealed_once(cast(V5PipelineWorkspace, workspace), payload_path)

    marker = json.loads(workspace.sealed_open_path.read_text(encoding="utf-8"))
    assert marker["state"] == "failed"
    assert marker["error_type"] == "V5PipelineError"
    assert not (workspace.root / V5_SEMANTIC_OPEN_RECEIPT_PATH).exists()
    with pytest.raises(V5PipelineError, match="already opened"):
        open_semantic_sealed_once(cast(V5PipelineWorkspace, workspace), payload_path)


def test_semantic_snapshot_parser_rejects_duplicates_and_foreign_rows(
    monkeypatch,
) -> None:
    monkeypatch.setitem(SPLIT_COUNTS, "semantic_sealed_test", 4)
    records, rows = _sealed_rows(4)
    benchmark = cast(Any, SimpleNamespace(records=tuple(records)))
    assert len(load_semantic_snapshot_bytes(_payload(rows), benchmark)) == 4

    with pytest.raises(V5PipelineError, match="duplicate"):
        load_semantic_snapshot_bytes(_payload([*rows, rows[0]]), benchmark)
    foreign = [*rows]
    foreign[0] = {**foreign[0], "sample_id": "foreign"}
    with pytest.raises(V5PipelineError, match="outside the sealed split"):
        load_semantic_snapshot_bytes(_payload(foreign), benchmark)


def test_open_receipt_detects_writable_or_changed_snapshot(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setitem(SPLIT_COUNTS, "semantic_sealed_test", 4)
    records, rows = _sealed_rows(4)
    payload = _payload(rows)
    payload_path = tmp_path / "sealed.jsonl"
    payload_path.write_bytes(payload)
    workspace = _FakeWorkspace(tmp_path / "workspace", _digest(payload))
    monkeypatch.setattr(
        sealed_module,
        "load_bound_benchmark",
        lambda _workspace: SimpleNamespace(records=tuple(records)),
    )
    monkeypatch.setattr(
        sealed_module,
        "load_completed_validation",
        lambda _workspace, direction: _validation(direction),
    )
    _, snapshot = open_semantic_sealed_once(cast(V5PipelineWorkspace, workspace), payload_path)
    os.chmod(snapshot, 0o644)

    with pytest.raises(V5PipelineError, match="immutable regular file"):
        load_semantic_open_receipt(cast(V5PipelineWorkspace, workspace))


def test_semantic_open_cli_has_no_hash_or_force_override() -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "open-semantic-sealed",
            "--workspace",
            "workspace",
            "--payload",
            "sealed.jsonl",
        ]
    )
    assert args.payload == Path("sealed.jsonl")
    assert not hasattr(args, "expected_sha256")
    assert not hasattr(args, "force")
    assert not hasattr(args, "resume")
