from __future__ import annotations

import importlib.util
import tempfile
from pathlib import Path

import pytest

from goldenexperience.runtime.lmcache_mp_server import (
    LMCACHE_MP_AUDIT_CHUNK_SIZE,
    LMCacheMPServerConfig,
    LMCacheMPServerError,
    LMCacheMPServerProcess,
    _probe_chunk_size,
)


def test_lmcache_mp_server_config_freezes_memory_only_non_gpu_protocol() -> None:
    config = LMCacheMPServerConfig()
    server = LMCacheMPServerProcess(config)
    command = server.command(45678)
    parameters = config.publication_parameters()

    assert "--transfer-mode" in command
    assert command[command.index("--transfer-mode") + 1] == "non_gpu"
    assert command[command.index("--chunk-size") + 1] == "128"
    assert "--l1-size-gb" in command
    assert not any("mooncake" in value.lower() for value in command)
    assert parameters["l2_adapters"] == []
    assert parameters["filesystem_backing_configured"] is False
    assert parameters["port"] == "automatic_loopback_port"


@pytest.mark.parametrize(
    "config, message",
    [
        (LMCacheMPServerConfig(host="0.0.0.0"), "loopback"),
        (LMCacheMPServerConfig(chunk_size=256), "chunk size"),
        (LMCacheMPServerConfig(l1_size_gb=0), "L1 size"),
        (LMCacheMPServerConfig(port=True), "port"),
    ],
)
def test_lmcache_mp_server_rejects_non_publication_configurations(
    config: LMCacheMPServerConfig,
    message: str,
) -> None:
    with pytest.raises(LMCacheMPServerError, match=message):
        LMCacheMPServerProcess(config)


def test_lmcache_mp_server_detects_files_in_its_private_runtime_directory() -> None:
    server = LMCacheMPServerProcess(LMCacheMPServerConfig())
    server._runtime_directory = tempfile.TemporaryDirectory(prefix="golden-lmcache-test-")
    Path(server._runtime_directory.name, "unexpected.bin").write_bytes(b"backing")

    with pytest.raises(LMCacheMPServerError, match="filesystem backing"):
        server.assert_no_backing_files()

    server._stop(suppress_errors=True)


@pytest.mark.skipif(
    importlib.util.find_spec("lmcache") is None,
    reason="LMCache runtime extra is not installed",
)
def test_lmcache_mp_server_starts_real_non_gpu_protocol_without_files() -> None:
    config = LMCacheMPServerConfig(
        l1_size_gb=1.0,
        l1_init_size_gb=1,
        startup_timeout_s=30.0,
    )
    server = LMCacheMPServerProcess(config)

    with server:
        assert server.running
        assert _probe_chunk_size(server.server_url, timeout_s=2.0) == (LMCACHE_MP_AUDIT_CHUNK_SIZE)
        server.assert_no_backing_files()

    assert not server.running
    assert server.backing_files_remaining == 0
