import json
from copy import deepcopy
from pathlib import Path

import pytest

from goldenexperience.runtime.cross_model_materializer import materialize_qwen3_8b_to_14b
from goldenexperience.runtime.cross_model_reuse import (
    chunk_hash_hex_to_bytes,
    common_chunk_hash_prefix,
    evaluate_runtime_reuse,
    lmcache_chunk_hashes,
    mooncake_setup_config,
    object_key_string,
    select_lookup_candidate,
    select_shared_prefix_candidate,
    token_ids_sha256,
)


def valid_runtime_evidence() -> dict:
    keys = ["target@rank@hash-a", "target@rank@hash-b"]
    request = {
        "prompt": {"expected_final_answer": "72"},
        "timing": {"ttft_ms": 100.0},
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
            "elapsed_ms": 10.0,
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
        "reuse_request": {**deepcopy(request), "timing": {"ttft_ms": 80.0}},
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


def test_select_lookup_candidate_requires_current_request_identity() -> None:
    record = select_lookup_candidate(
        [
            {
                "request_id": "request-a-worker",
                "model_name": "source",
                "seq_len": 32,
                "chunk_size": 16,
                "chunk_hashes": ["0xaa", "0xab"],
            },
            {
                "request_id": "request-b-worker",
                "model_name": "source",
                "seq_len": 32,
                "chunk_size": 16,
                "chunk_hashes": ["0xba", "0xbb"],
            },
        ],
        model_name="source",
        request_id="request-b",
        expected_seq_len=32,
        expected_chunk_size=16,
    )

    assert record is not None
    assert record["request_id"] == "request-b-worker"
    assert record["chunk_hashes"] == ["0xba", "0xbb"]


def test_select_lookup_candidate_fails_closed_without_prompt_binding() -> None:
    records = [{"model_name": "source", "chunk_hashes": ["0xaa"]}]

    assert select_lookup_candidate(records, model_name="source") is None
    assert (
        select_lookup_candidate(
            records,
            model_name="source",
            expected_chunk_hashes=["0xbb"],
        )
        is None
    )
    partial_record = {
        "request_id": "request-a-worker",
        "model_name": "source",
        "seq_len": 32,
        "chunk_size": 16,
        "chunk_hashes": ["0xaa"],
    }
    assert (
        select_lookup_candidate(
            [partial_record],
            model_name="source",
            request_id="request-a",
            expected_seq_len=32,
            expected_chunk_size=16,
        )
        is None
    )


def test_cross_prompt_hashes_only_share_exact_complete_prefix_chunks() -> None:
    prompt_a = ["0x11", "0x22", "0x33"]
    equal_length_prompt_b = ["0xaa", "0xbb", "0xcc"]
    prompt_c = ["0x11", "0x22", "0x44"]

    assert common_chunk_hash_prefix(prompt_a, equal_length_prompt_b) == []
    assert common_chunk_hash_prefix(prompt_a, prompt_c) == ["0x11", "0x22"]


def test_lmcache_blake3_hashes_match_the_runtime_wire_algorithm() -> None:
    hashes = lmcache_chunk_hashes(
        list(range(1, 34)),
        chunk_size=16,
        hash_algorithm="blake3",
    )

    assert hashes == [
        "0x4c5b938a8d6e2c965ae6ae33b3c82479be3d7e7486dfa2f2bc3ffdb30f7e84c0",
        "0x3234e83644ba92553bbb47f688c6a1e77384c9ad49a40202a2e12e535cd6c135",
    ]


def test_shared_prefix_candidate_binds_source_request_and_target_tokens() -> None:
    source_tokens = list(range(1, 49))
    target_tokens = list(range(1, 17)) + list(range(1001, 1033))
    source_hashes = lmcache_chunk_hashes(
        source_tokens,
        chunk_size=16,
        hash_algorithm="blake3",
    )
    record = {
        "timestamp": 10.0,
        "request_id": "source-request-worker",
        "model_name": "source",
        "seq_len": len(source_tokens),
        "chunk_size": 16,
        "chunk_hashes": source_hashes,
    }

    candidate = select_shared_prefix_candidate(
        [record],
        model_name="source",
        source_request_id="source-request",
        source_token_ids=source_tokens,
        target_token_ids=target_tokens,
        chunk_size=16,
        hash_algorithm="blake3",
    )

    assert candidate is not None
    assert candidate["record"] is record
    assert candidate["chunk_hashes"] == source_hashes[:1]
    assert candidate["binding"]["shared_prefix_token_count"] == 16
    assert candidate["binding"]["token_ids_sha256"] == token_ids_sha256(source_tokens)
    assert candidate["binding"]["target_token_ids_sha256"] == token_ids_sha256(target_tokens)

    assert (
        select_shared_prefix_candidate(
            [record],
            model_name="source",
            source_request_id="different-request",
            source_token_ids=source_tokens,
            target_token_ids=target_tokens,
            chunk_size=16,
            hash_algorithm="blake3",
        )
        is None
    )
    assert (
        select_shared_prefix_candidate(
            [record],
            model_name="source",
            source_request_id="source-request",
            source_token_ids=source_tokens,
            target_token_ids=list(range(1001, 1049)),
            chunk_size=16,
            hash_algorithm="blake3",
        )
        is None
    )


def test_cross_prompt_lookup_rejects_process_local_builtin_hashes() -> None:
    with pytest.raises(ValueError, match="deterministic blake3"):
        lmcache_chunk_hashes([1] * 16, chunk_size=16, hash_algorithm="builtin")


def test_token_id_digest_distinguishes_equal_length_prompts() -> None:
    assert token_ids_sha256([1, 2, 3, 4]) != token_ids_sha256([4, 3, 2, 1])


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


def test_runtime_reuse_requires_end_to_end_ttft_improvement() -> None:
    evidence = valid_runtime_evidence()
    evidence["reuse_request"]["timing"]["ttft_ms"] = 95.0

    validation = evaluate_runtime_reuse(**evidence)

    assert validation["success"] is False
    assert "end_to_end_ttft_improved" in validation["failure_reasons"]
    assert validation["timing"]["materialization_plus_reuse_ttft_ms"] == 105.0


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


def test_legacy_materializer_can_never_inject_or_run_unsafe() -> None:
    result = materialize_qwen3_8b_to_14b(
        {
            "inject_to_mooncake": True,
            "allow_unsafe": True,
        }
    )

    assert result["success"] is False
    assert result["fallback_reason"] == "legacy_materializer_injection_disabled"
    assert result["fallback_safe"] is True
