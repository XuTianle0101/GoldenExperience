import copy
from dataclasses import asdict
from pathlib import Path

import pytest

from goldenexperience.cli.v5_smoke import build_parser
from goldenexperience.size_variant.cached_kv_manifest import CachedKVModelSpec
from goldenexperience.size_variant.real_model_smoke import (
    REAL_MODEL_SMOKE_AUTHORITY,
    REAL_MODEL_SMOKE_SCHEMA_VERSION,
    RealModelSmokeError,
    smoke_report_errors,
    write_smoke_report,
)


def _model(model_id: str, *, layers: int) -> CachedKVModelSpec:
    return CachedKVModelSpec(
        model_id=model_id,
        parameter_count_b=4.0 if layers == 36 else 8.0,
        revision="test",
        architecture="qwen3",
        config_sha256="a" * 64,
        tokenizer_sha256="b" * 64,
        weights_sha256=("c" if layers == 36 else "d") * 64,
        num_layers=layers,
        num_key_value_heads=8,
        head_dim=128,
        dtype="bfloat16",
        rope_theta=1_000_000,
        max_position_embeddings=40960,
        chat_template_sha256="e" * 64,
    )


def _report() -> dict:
    source = _model("Qwen/Qwen3-4B", layers=36)
    target = _model("Qwen/Qwen3-8B", layers=40)
    return {
        "schema_version": REAL_MODEL_SMOKE_SCHEMA_VERSION,
        "authority": REAL_MODEL_SMOKE_AUTHORITY,
        "status": "passed",
        "approval_granted": False,
        "evidence_eligible": False,
        "sealed_split_accessed": False,
        "direction": "qwen3_4b_to_8b",
        "source": asdict(source),
        "target": asdict(target),
        "input": {"token_count": 64},
        "transport": {"query_sample_count": 8, "key_sample_count": 32},
        "shapes": {
            "source_kv": [2, 36, 8, 64, 128],
            "target_kv": [2, 40, 8, 64, 128],
            "target_query": [40, 32, 8, 128],
            "native_attention_output": [40, 32, 8, 128],
            "transformed_kv": [2, 40, 8, 32, 128],
        },
        "losses": {
            "native_generation": 1.0,
            "prompt_tail_distillation": 0.1,
            "attention_logit_kl": 0.2,
            "attention_output_mse": 0.3,
            "transformed_kv_anchor": 0.4,
            "total": 1.315,
        },
        "checks": {
            "tokenizer_compatible": True,
            "shape_contract_passed": True,
            "all_losses_finite": True,
        },
    }


def test_real_model_smoke_report_is_diagnostic_only() -> None:
    report = _report()

    assert smoke_report_errors(report) == []

    for field, unsafe_value in (
        ("approval_granted", True),
        ("evidence_eligible", True),
        ("sealed_split_accessed", True),
    ):
        unsafe = copy.deepcopy(report)
        unsafe[field] = unsafe_value
        assert smoke_report_errors(unsafe)


def test_real_model_smoke_report_rejects_nonfinite_loss_and_shape_drift() -> None:
    report = _report()
    report["losses"]["attention_logit_kl"] = float("nan")
    report["shapes"]["transformed_kv"] = [2, 40, 7, 32, 128]

    errors = smoke_report_errors(report)

    assert any("finite" in error for error in errors)
    assert any("transformed KV shape" in error for error in errors)


def test_real_model_smoke_report_recomputes_frozen_loss_contract() -> None:
    report = _report()
    report["losses"]["total"] = 0.0

    assert any("frozen contract" in error for error in smoke_report_errors(report))


def test_smoke_report_write_is_atomic_and_no_clobber(tmp_path: Path) -> None:
    output = tmp_path / "smoke.json"
    report = _report()

    assert write_smoke_report(output, report) == output
    first = output.read_text(encoding="utf-8")
    with pytest.raises(FileExistsError):
        write_smoke_report(output, report)
    assert output.read_text(encoding="utf-8") == first
    assert not list(tmp_path.glob(".*.tmp"))


def test_smoke_report_refuses_invalid_payload_before_write(tmp_path: Path) -> None:
    output = tmp_path / "invalid.json"
    report = _report()
    report["status"] = "failed"

    with pytest.raises(RealModelSmokeError, match="status"):
        write_smoke_report(output, report)

    assert not output.exists()


def test_v5_smoke_cli_defaults_are_bounded_and_offline() -> None:
    args = build_parser().parse_args([])

    assert args.direction == "qwen3_4b_to_8b"
    assert args.max_tokens == 64
    assert args.max_queries == 8
    assert args.max_keys == 32
    assert args.rank == 32
    assert args.source_window == 1
    assert not args.allow_download
    assert not args.refresh_identity
