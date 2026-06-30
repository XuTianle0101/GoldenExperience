from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
CLIENT_PATH = ROOT / "scripts" / "kv_baseline" / "kv_baseline_client.py"
SPEC = importlib.util.spec_from_file_location("kv_baseline_client", CLIENT_PATH)
assert SPEC is not None
kv_client = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(kv_client)


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _base_run(tmp_path: Path, *, kv_backend: str = "mp", write_pids: bool = True) -> Path:
    run_dir = tmp_path / "run"
    cache_dir = run_dir / "cache"
    _write_json(
        run_dir / "metadata.json",
        {
            "kv_backend": kv_backend,
            "mode": "restart",
            "kv_cache_dir": str(cache_dir),
        },
    )
    _write_json(run_dir / "requests" / "offload.json", {"phase": "offload", "timing": {}})
    _write_json(run_dir / "requests" / "reuse.json", {"phase": "reuse", "timing": {}})
    if write_pids:
        (run_dir / "offload.pid").write_text("111\n", encoding="utf-8")
        (run_dir / "reuse.pid").write_text("222\n", encoding="utf-8")
        (run_dir / "lmcache_mp.pid").write_text("333\n", encoding="utf-8")
    return run_dir


def _summarize(run_dir: Path) -> dict[str, object]:
    output = run_dir / "summary.json"
    result = kv_client.summarize(
        SimpleNamespace(
            run_dir=str(run_dir),
            output=str(output),
            require_disk_offload=False,
            require_reuse_evidence=False,
        )
    )
    assert result == 0
    return json.loads(output.read_text(encoding="utf-8"))


def test_summary_rejects_reuse_without_disk_files(tmp_path: Path) -> None:
    run_dir = _base_run(tmp_path)
    (run_dir / "logs").mkdir(parents=True)
    (run_dir / "logs" / "reuse_lmcache_mp_server.log").write_text(
        "LMCache retrieved 16 tokens from L2\n",
        encoding="utf-8",
    )

    summary = _summarize(run_dir)

    assert summary["offload_has_disk_evidence"] is False
    assert summary["reuse_has_cache_evidence"] is True
    assert summary["disk_reuse_success"] is False
    assert summary["evidence"]["offload_has_disk_evidence"] is False
    assert summary["evidence"]["reuse_has_cache_evidence"] is True
    assert summary["evidence"]["disk_reuse_success"] is False


def test_summary_rejects_disk_files_without_reuse_evidence(tmp_path: Path) -> None:
    run_dir = _base_run(tmp_path)
    cache_file = run_dir / "cache" / "object.data"
    cache_file.parent.mkdir(parents=True)
    cache_file.write_bytes(b"kv")

    summary = _summarize(run_dir)

    assert summary["offload_has_disk_evidence"] is True
    assert summary["reuse_has_cache_evidence"] is False
    assert summary["disk_reuse_success"] is False
    assert summary["evidence"]["offload_has_disk_evidence"] is True
    assert summary["evidence"]["reuse_has_cache_evidence"] is False
    assert summary["evidence"]["disk_reuse_success"] is False


def test_summary_accepts_full_mp_disk_reuse_evidence(tmp_path: Path) -> None:
    run_dir = _base_run(tmp_path)
    cache_file = run_dir / "cache" / "object.data"
    cache_file.parent.mkdir(parents=True)
    cache_file.write_bytes(b"kv")
    (run_dir / "logs").mkdir(parents=True)
    (run_dir / "logs" / "offload_lmcache_mp_server.log").write_text(
        "LMCache stored 16 tokens to L2\n",
        encoding="utf-8",
    )
    (run_dir / "logs" / "reuse_lmcache_mp_server.log").write_text(
        "LMCache retrieved 16 tokens from L2\n",
        encoding="utf-8",
    )

    summary = _summarize(run_dir)

    assert summary["offload_engine_pid"] == 111
    assert summary["reuse_engine_pid"] == 222
    assert summary["lmcache_mp_pid"] == 333
    assert summary["disk_reuse_success"] is True
    assert summary["evidence"]["engine_restarted"] is True
    assert summary["evidence"]["lmcache_mp_persistent"] is True
    assert summary["evidence"]["disk_reuse_success"] is True


def test_summary_keeps_legacy_artifacts_pid_optional(tmp_path: Path) -> None:
    run_dir = _base_run(tmp_path, kv_backend="legacy", write_pids=False)
    cache_file = run_dir / "cache" / "object.data"
    cache_file.parent.mkdir(parents=True)
    cache_file.write_bytes(b"kv")
    (run_dir / "logs").mkdir(parents=True)
    (run_dir / "logs" / "reuse_server.log").write_text(
        "LMCache retrieved 16 tokens from L2\n",
        encoding="utf-8",
    )

    summary = _summarize(run_dir)

    assert summary["offload_engine_pid"] is None
    assert summary["reuse_engine_pid"] is None
    assert summary["lmcache_mp_pid"] is None
    assert summary["evidence"]["pid_evidence_required"] is False
    assert summary["evidence"]["disk_reuse_success"] is True
