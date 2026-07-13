from pathlib import Path

from goldenexperience.runtime.materializer_client import ResidentMaterializerClient


def test_resident_materializer_reuses_one_worker(tmp_path: Path) -> None:
    client = ResidentMaterializerClient(
        python_bin=".venv/bin/python",
        cwd=Path.cwd(),
        stderr_path=tmp_path / "materializer.log",
        timeout_sec=30,
    )
    try:
        first = client.request({"mode": "not-a-mode"})
        pid = client.pid
        second = client.request({"mode": "also-not-a-mode"})

        assert first["fallback_reason"] == "invalid_materializer_mode"
        assert second["fallback_reason"] == "invalid_materializer_mode"
        assert pid is not None
        assert client.pid == pid
        assert client.process_info()["request_count"] == 2
    finally:
        client.close()

    assert client.pid is None
