import json
from pathlib import Path

from goldenexperience.runtime.cross_model_materializer import materialize_qwen3_8b_to_14b
from goldenexperience.runtime.cross_model_reuse import (
    chunk_hash_hex_to_bytes,
    mooncake_setup_config,
    object_key_string,
    select_lookup_candidate,
)


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
