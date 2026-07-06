import json
import os
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
BASELINE_SCRIPT = (
    REPO_ROOT / "scripts" / "kv_baseline" / "run_vllm_lmcache_mooncake_kv_baseline.sh"
)
CLIENT_SCRIPT = REPO_ROOT / "scripts" / "kv_baseline" / "kv_baseline_client.py"


def _run_baseline_dry_run(tmp_path: Path, **env_overrides: str) -> Path:
    run_dir = tmp_path / "run"
    env = os.environ.copy()
    env.update(
        {
            "PYTHON_BIN": sys.executable,
            "GE_DRY_RUN": "1",
            "GE_RUN_DIR": str(run_dir),
            "GE_MODEL_PATH": "test/model",
        }
    )
    env.update(env_overrides)
    subprocess.run(
        ["bash", str(BASELINE_SCRIPT)],
        cwd=REPO_ROOT,
        env=env,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return run_dir


def test_mooncake_store_adapter_is_default(tmp_path: Path) -> None:
    run_dir = _run_baseline_dry_run(tmp_path)
    metadata = json.loads((run_dir / "metadata.json").read_text(encoding="utf-8"))
    adapter = json.loads(metadata["lmcache_mp"]["l2_adapter_json"])

    assert metadata["kv_backend"] == "mp"
    assert metadata["engine"] == "vllm"
    assert metadata["mooncake"]["enabled"] is True
    assert adapter["type"] == "mooncake_store"
    assert adapter["num_workers"] == 4
    assert adapter["per_op_workers"] == {"lookup": 2, "retrieve": 8, "store": 4}
    assert adapter["local_hostname"] == "127.0.0.1"
    assert adapter["metadata_server"] == "http://127.0.0.1:8080/metadata"
    assert adapter["master_server_addr"] == "127.0.0.1:50051"
    assert adapter["protocol"] == "tcp"
    assert adapter["storage_root_dir"].endswith("/cache/mooncake")
    assert "global_segment_size" in adapter
    assert "local_buffer_size" in adapter
    assert "mooncake_store" in (run_dir / "lmc_config.yaml").read_text(encoding="utf-8")


def test_filesystem_adapter_override_disables_mooncake(tmp_path: Path) -> None:
    run_dir = _run_baseline_dry_run(
        tmp_path,
        GE_LMCACHE_MP_L2_ADAPTER_TYPE="fs",
    )
    metadata = json.loads((run_dir / "metadata.json").read_text(encoding="utf-8"))
    adapter = json.loads(metadata["lmcache_mp"]["l2_adapter_json"])

    assert metadata["mooncake"]["enabled"] is False
    assert adapter["type"] == "fs"
    assert adapter["base_path"].endswith("/cache")
    assert not (run_dir / "mooncake_master.pid").exists()


def _write_minimal_metadata(run_dir: Path, cache_dir: Path, mooncake_root: Path) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "metadata.json").write_text(
        json.dumps(
            {
                "kv_cache_dir": str(cache_dir),
                "mooncake": {
                    "enabled": True,
                    "storage_root": str(mooncake_root),
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )


def test_summary_accepts_mooncake_storage_as_disk_evidence(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    cache_dir = run_dir / "cache"
    mooncake_root = cache_dir / "mooncake"
    (mooncake_root / "object").parent.mkdir(parents=True)
    (mooncake_root / "object").write_bytes(b"kv")
    _write_minimal_metadata(run_dir, cache_dir, mooncake_root)
    output = run_dir / "summary.json"

    subprocess.run(
        [
            sys.executable,
            str(CLIENT_SCRIPT),
            "summarize",
            "--run-dir",
            str(run_dir),
            "--output",
            str(output),
            "--require-disk-offload",
        ],
        cwd=REPO_ROOT,
        check=True,
    )
    summary = json.loads(output.read_text(encoding="utf-8"))

    assert summary["evidence"]["offload_has_disk_evidence"] is True
    assert summary["evidence"]["mooncake_storage_file_count"] == 1
    assert summary["mooncake_storage"]["total_bytes"] == 2


def test_summary_rejects_empty_mooncake_storage_for_required_disk_evidence(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    cache_dir = run_dir / "cache"
    mooncake_root = cache_dir / "mooncake"
    mooncake_root.mkdir(parents=True)
    _write_minimal_metadata(run_dir, cache_dir, mooncake_root)

    completed = subprocess.run(
        [
            sys.executable,
            str(CLIENT_SCRIPT),
            "summarize",
            "--run-dir",
            str(run_dir),
            "--output",
            str(run_dir / "summary.json"),
            "--require-disk-offload",
        ],
        cwd=REPO_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    assert completed.returncode == 3
    assert "disk offload evidence is absent" in completed.stderr
