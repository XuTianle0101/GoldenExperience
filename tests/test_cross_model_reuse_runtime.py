import json
from copy import deepcopy
from pathlib import Path

from goldenexperience.runtime.cross_model_materializer import materialize_qwen3_8b_to_14b
from goldenexperience.runtime.cross_model_reuse import (
    chunk_hash_hex_to_bytes,
    evaluate_runtime_reuse,
    mooncake_setup_config,
    object_key_string,
    select_lookup_candidate,
)


def valid_runtime_evidence() -> dict:
    keys = ["target@rank@hash-a", "target@rank@hash-b"]
    request = {
        "prompt": {"expected_final_answer": "72"},
        "response": {
            "text": "Final answer: 72",
            "contains_expected_final_answer": True,
            "extracted_final_answer": "72",
            "matches_expected_final_answer": True,
        },
    }
    return {
        "materializer": {
            "success": True,
            "materialized": True,
            "injected": True,
            "allow_unsafe": False,
            "offline_quality_gate": {"checks": {"hidden": True, "decode": True}},
            "runtime_quality_gate": {"checks": {"key": True, "value": True}},
            "injection": {"keys": keys, "injected_count": 2},
        },
        "target_key_strings": keys,
        "source_key_status": {
            "found": ["source-a", "source-b"],
            "missing": [],
            "found_count": 2,
            "missing_count": 0,
            "total": 2,
        },
        "target_key_status_before": {
            "found": [],
            "missing": keys,
            "found_count": 0,
            "missing_count": 2,
            "total": 2,
        },
        "target_key_status_after": {
            "found": keys,
            "missing": [],
            "found_count": 2,
            "missing_count": 0,
            "total": 2,
        },
        "target_external_tokens": 32,
        "chunk_size": 16,
        "native_request": deepcopy(request),
        "reuse_request": deepcopy(request),
    }


def test_object_key_string_matches_lmcache_wire_format() -> None:
    key = object_key_string(
        model_name="/workspace/volume/softdata/models/Qwen3-14B",
        chunk_hash="0x40c787a43b802542",
    )

    assert key == "/workspace/volume/softdata/models/Qwen3-14B@01000100@40c787a43b802542"
    assert chunk_hash_hex_to_bytes("0x40c787a43b802542").hex() == "40c787a43b802542"


def test_mooncake_setup_config_strips_lmcache_only_keys() -> None:
    prepared = mooncake_setup_config(
        {
            "type": "mooncake_store",
            "num_workers": 4,
            "per_op_workers": {"lookup": 2},
            "metadata_server": "http://127.0.0.1:8080/metadata",
            "master_server_addr": "127.0.0.1:50051",
            "storage_root_dir": "/tmp/mooncake",
        }
    )

    assert prepared == {
        "metadata_server": "http://127.0.0.1:8080/metadata",
        "master_server_addr": "127.0.0.1:50051",
    }


def test_select_lookup_candidate_prefers_long_source_record() -> None:
    record = select_lookup_candidate(
        [
            {"model_name": "target", "seq_len": 100, "chunk_hashes": ["0x1"]},
            {"model_name": "source", "seq_len": 24, "chunk_hashes": ["0x2"]},
            {"model_name": "source", "seq_len": 1790, "chunk_hashes": ["0x3", "0x4"]},
        ],
        model_name="source",
    )

    assert record is not None
    assert record["seq_len"] == 1790
    assert record["chunk_hashes"] == ["0x3", "0x4"]


def test_runtime_reuse_requires_complete_provenance_and_matching_output() -> None:
    validation = evaluate_runtime_reuse(**valid_runtime_evidence())

    assert validation["success"] is True
    assert validation["status"] == "cross_model_reuse_success"
    assert validation["failure_reasons"] == []


def test_runtime_reuse_rejects_failed_task_assertion() -> None:
    evidence = valid_runtime_evidence()
    evidence["reuse_request"]["response"]["matches_expected_final_answer"] = False

    validation = evaluate_runtime_reuse(**evidence)

    assert validation["success"] is False
    assert validation["status"] == "quality_validation_failed"
    assert "reuse_task_assertion" in validation["failure_reasons"]


def test_runtime_reuse_rejects_output_drift_from_native() -> None:
    evidence = valid_runtime_evidence()
    evidence["reuse_request"]["response"]["text"] = "Reasoning changed. Final answer: 72"

    validation = evaluate_runtime_reuse(**evidence)

    assert validation["status"] == "quality_validation_failed"
    assert "native_output_match" in validation["failure_reasons"]


def test_runtime_reuse_rejects_missing_quality_checks() -> None:
    evidence = valid_runtime_evidence()
    evidence["materializer"]["runtime_quality_gate"]["checks"] = {}

    validation = evaluate_runtime_reuse(**evidence)

    assert validation["status"] == "runtime_validation_failed"
    assert "runtime_quality_gate" in validation["failure_reasons"]


def test_runtime_reuse_rejects_preexisting_target_key() -> None:
    evidence = valid_runtime_evidence()
    key = evidence["target_key_strings"][0]
    evidence["target_key_status_before"]["found"] = [key]
    evidence["target_key_status_before"]["missing"] = evidence["target_key_strings"][1:]

    validation = evaluate_runtime_reuse(**evidence)

    assert validation["success"] is False
    assert validation["status"] == "runtime_validation_failed"
    assert "target_keys_absent_before" in validation["failure_reasons"]


def test_runtime_reuse_rejects_inexact_transfer_accounting() -> None:
    evidence = valid_runtime_evidence()
    evidence["target_external_tokens"] = 16

    validation = evaluate_runtime_reuse(**evidence)

    assert validation["success"] is False
    assert validation["expected_external_tokens"] == 32
    assert "external_token_count" in validation["failure_reasons"]


def test_materializer_quality_gate_falls_back_before_model_load(tmp_path: Path) -> None:
    summary = tmp_path / "bridge.json"
    weights = tmp_path / "bridge.pt"
    summary.write_text(
        json.dumps(
            {
                "learned_low_rank_hidden_bridge": {
                    "hidden_cosine_mean": 0.78,
                    "key_cosine_mean": 0.92,
                    "value_cosine_mean": 0.62,
                    "decode_logit_cosine_mean": 0.76,
                }
            }
        ),
        encoding="utf-8",
    )

    result = materialize_qwen3_8b_to_14b(
        {
            "token_ids": [1, 2, 3],
            "chunk_hashes": ["0x1"],
            "output_dir": str(tmp_path / "out"),
            "bridge_summary_path": str(summary),
            "bridge_weights_path": str(weights),
            "source_model_path": "/does/not/exist/source",
            "target_model_path": "/does/not/exist/target",
        }
    )

    assert result["success"] is False
    assert result["fallback_reason"] == "quality_gate_failed"
    assert result["materialized"] is False
    assert result["injected"] is False
    assert not weights.exists()
