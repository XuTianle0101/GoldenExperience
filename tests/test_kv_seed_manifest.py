import importlib.util
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
EXPORT_SCRIPT = REPO_ROOT / "scripts" / "kv_baseline" / "export_kv_seed_manifest.py"


def _load_export_module():
    spec = importlib.util.spec_from_file_location("ge_kv_seed_manifest", EXPORT_SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_fake_run(tmp_path: Path) -> Path:
    run_dir = tmp_path / "run1"
    storage_root = run_dir / "cache" / "mooncake"
    metrics_dir = run_dir / "metrics"
    logs_dir = run_dir / "logs"
    storage_root.mkdir(parents=True)
    metrics_dir.mkdir(parents=True)
    logs_dir.mkdir(parents=True)
    (storage_root / "cluster" / "a").parent.mkdir(parents=True)
    (storage_root / "cluster" / "a").write_bytes(b"kv-a")
    (run_dir / "disk_prompt.json").write_text('{"prompts": []}\n', encoding="utf-8")
    (run_dir / "metadata.json").write_text(
        json.dumps(
            {
                "run_id": "run1",
                "mode": "restart",
                "engine": "vllm",
                "kv_backend": "mp",
                "model_name": "qwen3-8b",
                "model_path": "/models/qwen3-8b",
                "prompt_id": "p1",
                "prompt_file": str(run_dir / "disk_prompt.json"),
                "chunk_size": 16,
                "hash_algorithm": "builtin",
                "generated_disk_prompt": True,
                "lmcache_mp": {
                    "l2_adapter_type": "mooncake_store",
                    "l2_store_policy": "skip_l1",
                },
                "mooncake": {
                    "enabled": True,
                    "protocol": "tcp",
                    "master_server_addr": "127.0.0.1:50051",
                    "metadata_server": "http://127.0.0.1:8080/metadata",
                    "storage_root": str(storage_root),
                },
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "summary.json").write_text(
        json.dumps(
            {
                "evidence": {
                    "offload_has_disk_evidence": True,
                    "reuse_has_cache_evidence": True,
                    "disk_reuse_success": True,
                },
                "mooncake_storage": {
                    "path": str(storage_root),
                    "file_count": 1,
                    "total_bytes": 4,
                    "sample_files": [str(storage_root / "cluster" / "a")],
                },
                "requests": {
                    "offload": {
                        "prompt": {"id": "p1", "sha256": "prompt-hash"},
                        "timing": {"e2e_ms": 10.0, "ttft_ms": 1.0},
                        "usage": {
                            "prompt_tokens": 100,
                            "completion_tokens": 2,
                            "total_tokens": 102,
                        },
                    },
                    "reuse": {
                        "prompt": {"id": "p1", "sha256": "prompt-hash"},
                        "timing": {"e2e_ms": 5.0, "ttft_ms": 0.5},
                        "usage": {
                            "prompt_tokens": 100,
                            "completion_tokens": 2,
                            "total_tokens": 102,
                        },
                    },
                },
                "deltas": {"reuse_minus_offload_ttft_ms": -0.5},
            }
        ),
        encoding="utf-8",
    )
    (metrics_dir / "offload.prom").write_text(
        "vllm:external_prefix_cache_hits_total 0\n"
        'vllm:prompt_tokens_by_source_total{source="local_compute"} 100\n',
        encoding="utf-8",
    )
    (metrics_dir / "reuse.prom").write_text(
        "vllm:external_prefix_cache_hits_total 96\n"
        'vllm:prompt_tokens_by_source_total{source="local_compute"} 4\n'
        'vllm:prompt_tokens_by_source_total{source="external_kv_transfer"} 96\n',
        encoding="utf-8",
    )
    (logs_dir / "lmcache_mp_server.log").write_text(
        "MooncakeStore SET\nMooncakeStore GET\n",
        encoding="utf-8",
    )
    (logs_dir / "reuse_lmcache_mp_server.log").write_text(
        "MooncakeStore GET\nL2 prefetch load completed\n",
        encoding="utf-8",
    )
    return run_dir


def test_build_kv_seed_manifest_extracts_proof(tmp_path: Path) -> None:
    module = _load_export_module()
    run_dir = _write_fake_run(tmp_path)
    manifest = module.build_manifest(
        run_dir,
        artifact_uri="s3://bucket/run1.tar.zst",
        notes=["test manifest"],
    )

    assert manifest["schema_version"] == module.SCHEMA_VERSION
    assert manifest["artifact"]["artifact_uri"] == "s3://bucket/run1.tar.zst"
    assert manifest["run"]["run_id"] == "run1"
    assert manifest["run"]["prompt_file_relative_to_run"] == "disk_prompt.json"
    assert manifest["storage"]["kind"] == "mooncake"
    assert manifest["storage"]["root_relative_to_run"] == "cache/mooncake"
    assert manifest["storage"]["sample_files_relative_to_root"] == ["cluster/a"]
    assert manifest["proof"]["metrics"]["reuse_external_prefix_cache_hits_total"] == 96.0
    assert manifest["proof"]["metrics"]["reuse_prompt_tokens_external_kv_transfer"] == 96.0
    assert manifest["proof"]["log_counts"]["reuse_l2_prefetch_load_completed"] == 1
    assert manifest["notes"] == ["test manifest"]


def test_kv_seed_manifest_cli_writes_json(tmp_path: Path) -> None:
    module = _load_export_module()
    run_dir = _write_fake_run(tmp_path)
    output = tmp_path / "manifest.json"

    rc = module.main([str(run_dir), "--output", str(output), "--artifact-uri", "file://seed"])

    assert rc == 0
    manifest = json.loads(output.read_text(encoding="utf-8"))
    assert manifest["artifact"]["artifact_uri"] == "file://seed"
    assert manifest["proof"]["disk_reuse_success"] is True
