import hashlib
import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from goldenexperience.benchmarks.publication import SPLIT_COUNTS
from goldenexperience.cli.v5_pipeline import build_parser, status_payload
from goldenexperience.size_variant.cached_kv_manifest import CachedKVModelSpec, sha256_file
from goldenexperience.size_variant.v5_pipeline import (
    V5DirectionConfig,
    V5PipelineConfig,
    V5PipelineError,
    V5PipelineWorkspace,
    source_tree_sha256,
)


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


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


def _config(tmp_path: Path) -> V5PipelineConfig:
    benchmark = tmp_path / "benchmark.json"
    benchmark.write_text('{"frozen":true}\n', encoding="utf-8")
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
    return V5PipelineConfig(
        benchmark_manifest_uri=str(benchmark),
        benchmark_manifest_sha256=_digest("benchmark-content"),
        benchmark_manifest_file_sha256=sha256_file(benchmark),
        split_sha256={name: _digest(name) for name in SPLIT_COUNTS},
        tokenizer_sha256=_digest("tokenizer"),
        chat_template_sha256=_digest("chat"),
        sealed_payload_sha256=_digest("sealed"),
        code_sha256=_digest("code"),
        directions=directions,
    )


def _workspace(tmp_path: Path) -> V5PipelineWorkspace:
    return V5PipelineWorkspace.create(tmp_path / "workspace", _config(tmp_path))


def test_pipeline_workspace_is_content_bound_and_idempotent(tmp_path: Path) -> None:
    config = _config(tmp_path)
    workspace = V5PipelineWorkspace.create(tmp_path / "workspace", config)
    reopened = V5PipelineWorkspace.create(tmp_path / "workspace", config)

    assert reopened.config.pipeline_id == workspace.config.pipeline_id
    assert workspace.state().stages == {}
    assert json.loads(workspace.sealed_lock_path.read_text(encoding="utf-8"))["state"] == ("locked")
    assert status_payload(workspace)["semantic_sealed"] == "locked"

    changed = V5PipelineConfig.from_dict(config.to_dict())
    changed = V5PipelineConfig(**{**changed.__dict__, "code_sha256": _digest("changed-code")})
    with pytest.raises(V5PipelineError, match="different config"):
        V5PipelineWorkspace.create(tmp_path / "workspace", changed)


def test_pipeline_enforces_dependencies_and_reuses_completed_stage(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    direction = "qwen3_4b_to_8b"
    with pytest.raises(V5PipelineError, match="requires completed dependency"):
        workspace.begin_stage(direction, "fit_transport", parameters={"rank": 32})

    collect = workspace.begin_stage(
        direction,
        "collect_transport_train",
        parameters={"max_records": 4096},
    )
    output = tmp_path / "trace.json"
    output.write_text('{"records":4096}\n', encoding="utf-8")
    completed = workspace.complete_stage(
        collect,
        outputs={"trace_manifest": output},
        metadata={"record_count": 4096},
    )

    assert completed.status == "completed"
    artifact = completed.outputs["trace_manifest"]
    assert workspace.artifact_path(artifact).read_bytes() == output.read_bytes()
    reused = workspace.begin_stage(
        direction,
        "collect_transport_train",
        parameters={"max_records": 4096},
    )
    assert reused.reused is True
    assert reused.receipt_sha256 == completed.receipt_sha256
    fit = workspace.begin_stage(direction, "fit_transport", parameters={"rank": 32})
    assert fit.reused is False

    with pytest.raises(V5PipelineError, match="input changed"):
        workspace.begin_stage(
            direction,
            "collect_transport_train",
            parameters={"max_records": 1},
        )


def test_pipeline_identity_and_dependency_bindings_ignore_local_paths(
    tmp_path: Path,
) -> None:
    first_root = tmp_path / "first"
    second_root = tmp_path / "second"
    first_root.mkdir()
    second_root.mkdir()
    first_config = _config(first_root)
    second_config = _config(second_root)

    assert first_config.pipeline_id == second_config.pipeline_id
    assert first_config.content_sha256() != second_config.content_sha256()

    first = V5PipelineWorkspace.create(first_root / "workspace", first_config)
    second = V5PipelineWorkspace.create(second_root / "workspace", second_config)
    direction = "qwen3_4b_to_8b"
    outputs = []
    for root, workspace in ((first_root, first), (second_root, second)):
        lease = workspace.begin_stage(
            direction,
            "collect_transport_train",
            parameters={"max_records": 4096},
        )
        output = root / "trace.json"
        output.write_text('{"records":4096}\n', encoding="utf-8")
        outputs.append(
            workspace.complete_stage(
                lease,
                outputs={"trace_manifest": output},
                metadata={"host_specific_timing_ms": 1.0},
            )
        )

    assert outputs[0].receipt_sha256 != outputs[1].receipt_sha256
    first_fit = first.begin_stage(direction, "fit_transport", parameters={"rank": 32})
    second_fit = second.begin_stage(direction, "fit_transport", parameters={"rank": 32})
    assert first_fit.input_sha256 == second_fit.input_sha256


def test_pipeline_failure_requires_explicit_same_input_resume(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    direction = "qwen3_8b_to_4b"
    lease = workspace.begin_stage(
        direction,
        "collect_selector_train",
        parameters={"batch_size": 1},
    )
    failed = workspace.fail_stage(lease, RuntimeError("interrupted"))
    assert failed.status == "failed"
    assert failed.error_type == "RuntimeError"

    with pytest.raises(V5PipelineError, match="resume=True"):
        workspace.begin_stage(
            direction,
            "collect_selector_train",
            parameters={"batch_size": 1},
        )
    with pytest.raises(V5PipelineError, match="input differs"):
        workspace.begin_stage(
            direction,
            "collect_selector_train",
            parameters={"batch_size": 2},
            resume=True,
        )
    resumed = workspace.begin_stage(
        direction,
        "collect_selector_train",
        parameters={"batch_size": 1},
        resume=True,
    )
    assert resumed.attempt_id != lease.attempt_id
    assert workspace.state().stages[f"{direction}/collect_selector_train"].attempt_count == 2


def test_non_screening_fit_requires_cross_direction_structure_receipt(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    output = tmp_path / "output.json"
    output.write_text("{}\n", encoding="utf-8")

    def complete(direction: str, stage: str) -> None:
        lease = workspace.begin_stage(direction, stage, parameters={"test": stage})
        workspace.complete_stage(
            lease,
            outputs={"output": output},
            metadata={},
        )

    other = "qwen3_8b_to_4b"
    with pytest.raises(V5PipelineError, match="only be selected"):
        workspace.begin_stage(other, "evaluate_method_dev", parameters={})
    complete(other, "collect_transport_train")
    with pytest.raises(V5PipelineError, match="frozen 4B-to-8B method-dev structure"):
        workspace.begin_stage(other, "fit_transport", parameters={"rank": 64})

    screening = "qwen3_4b_to_8b"
    complete(screening, "collect_transport_train")
    complete(screening, "fit_transport")
    complete(screening, "collect_method_dev")
    complete(screening, "evaluate_method_dev")

    lease = workspace.begin_stage(other, "fit_transport", parameters={"rank": 64})
    assert lease.reused is False


def test_pipeline_serializes_concurrent_stage_claims(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    direction = "qwen3_8b_to_14b"

    def claim():
        try:
            return workspace.begin_stage(
                direction,
                "collect_validation",
                parameters={"batch_size": 1},
            )
        except V5PipelineError as exc:
            return exc

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(lambda _: claim(), range(2)))

    assert sum(not isinstance(item, Exception) for item in results) == 1
    assert sum(isinstance(item, V5PipelineError) for item in results) == 1


def test_pipeline_keeps_sealed_stage_outside_generic_resume_path(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)

    with pytest.raises(V5PipelineError, match="four-direction validation guard"):
        workspace.begin_stage(
            "qwen3_4b_to_8b",
            "semantic_sealed",
            parameters={},
        )


def test_pipeline_detects_external_manifest_and_artifact_tampering(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    direction = "qwen3_14b_to_8b"
    lease = workspace.begin_stage(
        direction,
        "collect_method_dev",
        parameters={"batch_size": 1},
    )
    output = tmp_path / "method.json"
    output.write_text('{"ok":true}\n', encoding="utf-8")
    record = workspace.complete_stage(
        lease,
        outputs={"trace_manifest": output},
        metadata={},
    )
    artifact_path = workspace.root / record.outputs["trace_manifest"].path
    os.chmod(artifact_path, 0o644)
    with pytest.raises(V5PipelineError, match="stat identity changed"):
        workspace.state()

    os.chmod(artifact_path, 0o444)
    benchmark = Path(workspace.config.benchmark_manifest_uri)
    benchmark.write_text('{"changed":true}\n', encoding="utf-8")
    with pytest.raises(V5PipelineError, match="manifest file changed"):
        V5PipelineWorkspace.open(workspace.root)


def test_pipeline_rejects_invalid_output_names(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    output = tmp_path / "output.json"
    output.write_text("{}\n", encoding="utf-8")

    with pytest.raises(V5PipelineError, match="invalid pipeline output name"):
        workspace.publish_file(output, logical_name="../escape")


def test_pipeline_cli_has_no_sealed_payload_option() -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "init",
            "--workspace",
            "workspace",
            "--benchmark-manifest",
            "benchmark.json",
        ]
    )

    assert not hasattr(args, "sealed_payload")
    assert not args.refresh_identity


def test_source_tree_hash_changes_with_executable_source(tmp_path: Path) -> None:
    (tmp_path / "goldenexperience").mkdir()
    (tmp_path / "scripts").mkdir()
    (tmp_path / "pyproject.toml").write_text("[project]\n", encoding="utf-8")
    source = tmp_path / "goldenexperience" / "module.py"
    source.write_text("VALUE = 1\n", encoding="utf-8")
    first = source_tree_sha256(tmp_path)
    source.write_text("VALUE = 2\n", encoding="utf-8")

    assert source_tree_sha256(tmp_path) != first
