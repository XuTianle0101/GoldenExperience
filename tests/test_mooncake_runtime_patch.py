import importlib.util
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PATCH_SCRIPT = REPO_ROOT / "scripts" / "patch_lmcache_mooncake_runtime.py"


def _load_patch_module():
    spec = importlib.util.spec_from_file_location("ge_mooncake_patch", PATCH_SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


MOONCAKE_ADAPTER_ORIGINAL = '''# SPDX-License-Identifier: Apache-2.0
"""Mooncake Store native L2 adapter config and factory."""

from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    cast,
)

if TYPE_CHECKING:
    from lmcache.v1.distributed.internal_api import (
        L1MemoryDesc,
    )

from lmcache.logging import init_logger
from lmcache.v1.distributed.l2_adapters.base import (
    L2AdapterInterface,
)
from lmcache.v1.distributed.l2_adapters.config import (
    L2AdapterConfigBase,
    register_l2_adapter_type,
)
from lmcache.v1.distributed.l2_adapters.factory import (
    register_l2_adapter_factory,
)

logger = init_logger(__name__)

_LMCACHE_ONLY_KEYS = {
    "type",
    "num_workers",
    "eviction",
    "per_op_workers",
}


class MooncakeStoreL2AdapterConfig(L2AdapterConfigBase):
    def __init__(
        self,
        setup_config: dict[str, str],
        num_workers: int = 4,
        per_op_workers: dict[str, int] | None = None,
    ):
        super().__init__()
        self.num_workers = L2AdapterConfigBase._validate_num_workers(num_workers)
        self.per_op_workers = L2AdapterConfigBase._validate_per_op_workers(per_op_workers)
        self.setup_config: dict[str, str] = dict(setup_config)

    @classmethod
    def from_dict(cls, d: dict[str, object]) -> "MooncakeStoreL2AdapterConfig":
        num_workers = cast(int, d.get("num_workers", 4))
        per_op_workers = L2AdapterConfigBase._parse_per_op_workers_from_dict(d)
        setup: dict[str, str] = {}
        for k, v in d.items():
            if k in _LMCACHE_ONLY_KEYS:
                continue
            if v is not None:
                setup[k] = str(v)
        return cls(setup, num_workers, per_op_workers)

    @classmethod
    def help(cls) -> str:
        return "Mooncake Store L2 adapter config."


def _create_mooncake_store_l2_adapter(
    config: L2AdapterConfigBase,
    l1_memory_desc: "L1MemoryDesc | None" = None,
) -> L2AdapterInterface:
    try:
        # First Party
        from lmcache.lmcache_mooncake import (
            L1RegistrationConfig,
            LMCacheMooncakeClient,
        )
    except ImportError as e:
        raise RuntimeError("Mooncake Store L2 adapter requires the C++ extension") from e

    from lmcache.v1.distributed.l2_adapters.native_connector_l2_adapter import (
        NativeConnectorL2Adapter,
    )

    if not isinstance(config, MooncakeStoreL2AdapterConfig):
        raise ValueError(f"Expected MooncakeStoreL2AdapterConfig, got {type(config)}")
    l1_registration = L1RegistrationConfig()
    native_client = LMCacheMooncakeClient(
        config=config.setup_config,
        num_workers=config.num_workers,
        l1_registration=l1_registration,
        per_op_workers=config.per_op_workers,
    )
    return NativeConnectorL2Adapter(native_client)


register_l2_adapter_type("mooncake_store", MooncakeStoreL2AdapterConfig)
register_l2_adapter_factory("mooncake_store", _create_mooncake_store_l2_adapter)
'''


NATIVE_ADAPTER_ORIGINAL = '''# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from collections import defaultdict
from typing import Any
import select
import threading

from lmcache.native_storage_ops import Bitmap
from lmcache.v1.distributed.api import ObjectKey
from lmcache.v1.distributed.l2_adapters.base import L2TaskId


def _object_key_to_string(key: ObjectKey) -> str:
    return str(key)


class NativeConnectorL2Adapter:
    def __init__(self, native_client: Any) -> None:
        self._client = native_client
        self._type_name = type(native_client).__name__
        self._key_sizes: dict[ObjectKey, int] = {}
        self._locked_keys: dict[ObjectKey, int] = defaultdict(int)
        self._completed_lookups: dict[L2TaskId, Bitmap] = {}
        self._lock = threading.Lock()
        self._lookup_efd = native_client
        self._next_task_id = 0

    def _get_next_task_id(self) -> L2TaskId:
        task_id = self._next_task_id
        self._next_task_id += 1
        return task_id

    def submit_lookup_and_lock_task(
        self,
        keys: list[ObjectKey],
    ) -> L2TaskId:
        key_strings = [_object_key_to_string(k) for k in keys]

        with self._lock:
            task_id = self._get_next_task_id()
            future_id = int(self._client.submit_batch_exists(key_strings))
        return task_id

    def query_lookup_and_lock_result(self, task_id: L2TaskId) -> Bitmap | None:
        with self._lock:
            return self._completed_lookups.pop(task_id, None)
'''


def _write_fake_site_packages(tmp_path: Path) -> Path:
    site_packages = tmp_path / "site-packages"
    mooncake_dir = site_packages / "mooncake"
    adapter_dir = site_packages / "lmcache" / "v1" / "distributed" / "l2_adapters"
    mooncake_dir.mkdir(parents=True)
    adapter_dir.mkdir(parents=True)
    (mooncake_dir / "store.so").write_bytes(b"fake mooncake store")
    (adapter_dir / "mooncake_store_l2_adapter.py").write_text(
        MOONCAKE_ADAPTER_ORIGINAL,
        encoding="utf-8",
    )
    (adapter_dir / "native_connector_l2_adapter.py").write_text(
        NATIVE_ADAPTER_ORIGINAL,
        encoding="utf-8",
    )
    return site_packages


def test_mooncake_runtime_patch_is_idempotent(tmp_path: Path) -> None:
    patch_module = _load_patch_module()
    site_packages = _write_fake_site_packages(tmp_path)

    first = patch_module.patch_runtime(site_packages=[site_packages])
    assert any(result.changed for result in first)

    mooncake_lib = site_packages / "mooncake" / "libmooncake_store.so"
    mooncake_adapter = (
        site_packages
        / "lmcache"
        / "v1"
        / "distributed"
        / "l2_adapters"
        / "mooncake_store_l2_adapter.py"
    )
    native_adapter = mooncake_adapter.with_name("native_connector_l2_adapter.py")
    assert mooncake_lib.exists()
    assert mooncake_adapter.with_name("mooncake_store_l2_adapter.py.goldenexperience.bak").exists()
    assert native_adapter.with_name("native_connector_l2_adapter.py.goldenexperience.bak").exists()

    mooncake_text = mooncake_adapter.read_text(encoding="utf-8")
    native_text = native_adapter.read_text(encoding="utf-8")
    assert "class MooncakePythonL2Adapter" in mooncake_text
    assert "LMCACHE_MOONCAKE_PYTHON_ADAPTER" in mooncake_text
    assert "MooncakeStore SET" in mooncake_text
    assert "MooncakeStore GET" in mooncake_text
    assert "GE_MOONCAKE_EXTERNAL_INDEX" in mooncake_text
    assert "_refresh_external_index" in mooncake_text
    assert "external_index_hits" in mooncake_text
    assert "GOLDENEXPERIENCE_MOONCAKE_ADAPTER_VERSION = 2" in mooncake_text
    assert "bytes_read == requested_size == indexed_size" in mooncake_text
    assert "_use_local_lookup_index" in native_text
    assert "LMCACHE_MOONCAKE_NATIVE_EXISTS" in native_text

    before = (mooncake_text, native_text)
    second = patch_module.patch_runtime(site_packages=[site_packages])
    assert not any(result.changed for result in second)
    after = (
        mooncake_adapter.read_text(encoding="utf-8"),
        native_adapter.read_text(encoding="utf-8"),
    )
    assert after == before

    checked = patch_module.patch_runtime(site_packages=[site_packages], check=True)
    assert checked


def test_mooncake_read_requires_exact_indexed_size() -> None:
    patch_module = _load_patch_module()

    assert patch_module.is_complete_mooncake_read(8, 8, 8)
    assert not patch_module.is_complete_mooncake_read(7, 8, 8)
    assert not patch_module.is_complete_mooncake_read(8, 8, 7)
    assert not patch_module.is_complete_mooncake_read(9, 8, 8)
    assert not patch_module.is_complete_mooncake_read(0, 8, 8)


def test_mooncake_runtime_patch_upgrades_previous_adapter_block(tmp_path: Path) -> None:
    patch_module = _load_patch_module()
    site_packages = _write_fake_site_packages(tmp_path)
    patch_module.patch_runtime(site_packages=[site_packages])
    adapter = (
        site_packages
        / "lmcache"
        / "v1"
        / "distributed"
        / "l2_adapters"
        / "mooncake_store_l2_adapter.py"
    )
    text = adapter.read_text(encoding="utf-8").replace(
        "GOLDENEXPERIENCE_MOONCAKE_ADAPTER_VERSION = 2\n\n\n",
        "",
        1,
    ).replace(
        "if bytes_read == requested_size == indexed_size:",
        "if bytes_read > 0:",
        1,
    )
    adapter.write_text(text, encoding="utf-8")

    results = patch_module.patch_runtime(site_packages=[site_packages])

    assert any(result.changed for result in results)
    upgraded = adapter.read_text(encoding="utf-8")
    assert "GOLDENEXPERIENCE_MOONCAKE_ADAPTER_VERSION = 2" in upgraded
    assert "bytes_read == requested_size == indexed_size" in upgraded


def test_mooncake_runtime_patch_cli_check(tmp_path: Path) -> None:
    patch_module = _load_patch_module()
    site_packages = _write_fake_site_packages(tmp_path)
    patch_module.patch_runtime(site_packages=[site_packages])

    completed = subprocess.run(
        [sys.executable, str(PATCH_SCRIPT), "--site-packages", str(site_packages), "--check"],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert "patch verified" in completed.stdout
