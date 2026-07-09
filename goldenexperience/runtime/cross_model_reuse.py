"""Runtime helpers for quality-gated cross-model KV reuse."""

from __future__ import annotations

import ctypes
import json
import os
import time
from pathlib import Path
from typing import Any

_LMCACHE_ONLY_MOONCAKE_KEYS = {
    "type",
    "num_workers",
    "eviction",
    "per_op_workers",
    "storage_root_dir",
}


def chunk_hash_hex_to_bytes(value: str) -> bytes:
    text = value[2:] if value.startswith("0x") else value
    if not text:
        raise ValueError("chunk hash is empty")
    if len(text) % 2:
        text = "0" + text
    return bytes.fromhex(text)


def object_key_string(
    *,
    model_name: str,
    chunk_hash: str,
    kv_rank: int | None = None,
    cache_salt: str = "",
) -> str:
    from lmcache.v1.distributed.api import ObjectKey
    from lmcache.v1.distributed.l2_adapters.native_connector_l2_adapter import (
        _object_key_to_string,
    )

    rank = kv_rank if kv_rank is not None else default_kv_rank()
    return _object_key_to_string(
        ObjectKey(
            chunk_hash=chunk_hash_hex_to_bytes(chunk_hash),
            model_name=model_name,
            kv_rank=rank,
            cache_salt=cache_salt,
        )
    )


def default_kv_rank() -> int:
    from lmcache.v1.distributed.api import ObjectKey

    return ObjectKey.ComputeKVRank(
        world_size=1,
        global_rank=0,
        local_world_size=1,
        local_rank=0,
    )


def mooncake_setup_config(adapter_config: dict[str, Any]) -> dict[str, str]:
    return {
        str(key): str(value)
        for key, value in adapter_config.items()
        if key not in _LMCACHE_ONLY_MOONCAKE_KEYS and value is not None
    }


def _read_lookup_hash_file(path: Path) -> list[dict[str, Any]]:
    records = []
    if not path.exists():
        return records
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            parsed = json.loads(line)
            if isinstance(parsed, dict):
                records.append(parsed)
    return records


def load_lookup_hash_records(path: Path) -> list[dict[str, Any]]:
    if path.is_file():
        return _read_lookup_hash_file(path)
    records: list[dict[str, Any]] = []
    if path.is_dir():
        for item in sorted(path.glob("lookup_hashes_*.jsonl")):
            records.extend(_read_lookup_hash_file(item))
    return records


def select_lookup_candidate(
    records: list[dict[str, Any]],
    *,
    model_name: str,
    min_chunks: int = 1,
) -> dict[str, Any] | None:
    candidates = [
        record
        for record in records
        if record.get("model_name") == model_name
        and len(record.get("chunk_hashes") or []) >= min_chunks
    ]
    if not candidates:
        return None
    return max(
        candidates,
        key=lambda record: (
            int(record.get("seq_len") or 0),
            len(record.get("chunk_hashes") or []),
            float(record.get("timestamp") or 0.0),
        ),
    )


def token_ids_from_prompt(
    *,
    tokenizer_path: str,
    prompt_file: Path,
    prompt_id: str,
) -> list[int]:
    from transformers import AutoTokenizer

    manifest = json.loads(prompt_file.read_text(encoding="utf-8"))
    prompts = manifest.get("prompts")
    if not isinstance(prompts, list):
        raise ValueError(f"{prompt_file} must contain a prompts list")
    prompt = next((item for item in prompts if item.get("id") == prompt_id), None)
    if prompt is None:
        raise ValueError(f"Prompt id {prompt_id!r} not found in {prompt_file}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    encoded = tokenizer.apply_chat_template(
        prompt["messages"],
        tokenize=True,
        add_generation_prompt=True,
    )
    if hasattr(encoded, "keys") and "input_ids" in encoded:
        ids = encoded["input_ids"]
    else:
        ids = encoded
    return [int(item) for item in ids]


def mooncake_key_exists(
    *,
    setup_config: dict[str, Any],
    key_strings: list[str],
) -> dict[str, Any]:
    from mooncake.store import MooncakeDistributedStore

    store = MooncakeDistributedStore()
    prepared = mooncake_setup_config(setup_config)
    if metadata_server := prepared.get("metadata_server"):
        os.environ["MOONCAKE_TE_META_DATA_SERVER"] = metadata_server
    rc = store.setup(prepared)
    if rc != 0:
        raise RuntimeError(f"MooncakeDistributedStore.setup failed with rc={rc}")
    try:
        found: list[str] = []
        missing: list[str] = []
        for key in key_strings:
            exists = bool(store.is_exist(key))
            (found if exists else missing).append(key)
        return {
            "found": found,
            "missing": missing,
            "found_count": len(found),
            "missing_count": len(missing),
            "total": len(key_strings),
        }
    finally:
        store.close()


def inject_chunks_to_mooncake(
    *,
    setup_config: dict[str, Any],
    target_model_name: str,
    chunk_files: list[dict[str, Any]],
    external_index_path: Path,
    kv_rank: int | None = None,
    cache_salt: str = "",
    provenance: dict[str, Any] | None = None,
) -> dict[str, Any]:
    from mooncake.store import MooncakeDistributedStore

    started = time.perf_counter()
    prepared = mooncake_setup_config(setup_config)
    if metadata_server := prepared.get("metadata_server"):
        os.environ["MOONCAKE_TE_META_DATA_SERVER"] = metadata_server

    store = MooncakeDistributedStore()
    rc = store.setup(prepared)
    if rc != 0:
        raise RuntimeError(f"MooncakeDistributedStore.setup failed with rc={rc}")

    external_index_path.parent.mkdir(parents=True, exist_ok=True)
    rank = kv_rank if kv_rank is not None else default_kv_rank()
    injected: list[dict[str, Any]] = []
    try:
        with external_index_path.open("a", encoding="utf-8") as index_handle:
            for chunk in chunk_files:
                path = Path(chunk["path"])
                data = bytearray(path.read_bytes())
                key = object_key_string(
                    model_name=target_model_name,
                    chunk_hash=str(chunk["chunk_hash"]),
                    kv_rank=rank,
                    cache_salt=cache_salt,
                )
                ptr = ctypes.addressof(ctypes.c_char.from_buffer(data))
                result = int(store.put_from(key, ptr, len(data)))
                record = {
                    "schema_version": "goldenexperience.mooncake_external_index.v1",
                    "key": key,
                    "bytes": len(data),
                    "chunk_hash": chunk["chunk_hash"],
                    "chunk_index": chunk.get("chunk_index"),
                    "model_name": target_model_name,
                    "kv_rank": rank,
                    "cache_salt": cache_salt,
                    "path": str(path),
                    "put_rc": result,
                    "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "provenance": provenance or {},
                }
                if result != 0:
                    raise RuntimeError(f"Mooncake put_from failed for {key} rc={result}")
                index_handle.write(json.dumps(record, sort_keys=True) + "\n")
                index_handle.flush()
                injected.append(record)
    finally:
        store.close()

    return {
        "success": True,
        "injected_count": len(injected),
        "external_index_path": str(external_index_path),
        "keys": [item["key"] for item in injected],
        "bytes": sum(int(item["bytes"]) for item in injected),
        "elapsed_ms": (time.perf_counter() - started) * 1000,
    }
