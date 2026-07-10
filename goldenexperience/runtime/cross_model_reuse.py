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


def evaluate_runtime_reuse(
    *,
    materializer: dict[str, Any],
    target_key_strings: list[str],
    source_key_status: dict[str, Any],
    target_key_status_before: dict[str, Any],
    target_key_status_after: dict[str, Any] | None,
    target_external_tokens: float | int | None,
    chunk_size: int,
    native_request: dict[str, Any],
    reuse_request: dict[str, Any],
) -> dict[str, Any]:
    """Validate cache provenance, transfer accounting, and target output equivalence."""

    injection = materializer.get("injection")
    injection = injection if isinstance(injection, dict) else {}
    injected_keys = {str(key) for key in injection.get("keys", [])}
    target_keys = set(target_key_strings)
    before_found = {str(key) for key in target_key_status_before.get("found", [])}
    before_missing = {str(key) for key in target_key_status_before.get("missing", [])}
    after = target_key_status_after or {}
    after_found = {str(key) for key in after.get("found", [])}
    after_missing = {str(key) for key in after.get("missing", [])}
    injected_count = int(injection.get("injected_count") or 0)
    expected_external_tokens = injected_count * chunk_size

    offline_checks = materializer.get("offline_quality_gate", {}).get("checks", {})
    runtime_checks = materializer.get("runtime_quality_gate", {}).get("checks", {})
    checks = {
        "materializer_completed": bool(
            materializer.get("success")
            and materializer.get("materialized")
            and materializer.get("injected")
        ),
        "safe_policy": not bool(materializer.get("allow_unsafe")),
        "offline_quality_gate": _all_checks_pass(offline_checks),
        "runtime_quality_gate": _all_checks_pass(runtime_checks),
        "source_keys_present": bool(
            source_key_status.get("total", 0) > 0
            and source_key_status.get("found_count") == source_key_status.get("total")
            and source_key_status.get("missing_count") == 0
        ),
        "target_keys_absent_before": bool(
            target_keys and not before_found and before_missing == target_keys
        ),
        "injection_keys_match": bool(
            injected_keys
            and injected_keys <= target_keys
            and injected_count == len(injected_keys)
        ),
        "target_keys_present_after": bool(
            injected_keys and after_found == injected_keys and not (after_missing & injected_keys)
        ),
        "external_token_count": bool(
            isinstance(target_external_tokens, (int, float))
            and target_external_tokens == expected_external_tokens
            and expected_external_tokens > 0
        ),
        "native_task_assertion": _request_task_passed(native_request),
        "reuse_task_assertion": _request_task_passed(reuse_request),
        "native_output_match": _normalized_response(native_request)
        == _normalized_response(reuse_request)
        != "",
    }
    reasons = [name for name, passed in checks.items() if not passed]
    infrastructure_checks = (
        "materializer_completed",
        "offline_quality_gate",
        "runtime_quality_gate",
        "source_keys_present",
        "target_keys_absent_before",
        "injection_keys_match",
        "target_keys_present_after",
        "external_token_count",
    )
    consumed = all(checks[name] for name in infrastructure_checks)
    success = consumed and all(checks.values())
    if success:
        status = "cross_model_reuse_success"
    elif materializer.get("allow_unsafe") and materializer.get("injected"):
        status = "unsafe_shadow_reuse"
    elif consumed:
        status = "quality_validation_failed"
    elif materializer.get("injected"):
        status = "runtime_validation_failed"
    else:
        status = "fallback"
    return {
        "checks": checks,
        "failure_reasons": reasons,
        "expected_external_tokens": expected_external_tokens,
        "consumed_materialized_kv": consumed,
        "success": success,
        "status": status,
    }


def _all_checks_pass(checks: Any) -> bool:
    return isinstance(checks, dict) and bool(checks) and all(value is True for value in checks.values())


def _request_task_passed(request: dict[str, Any]) -> bool:
    expected = request.get("prompt", {}).get("expected_final_answer")
    if not expected:
        return False
    return request.get("response", {}).get("matches_expected_final_answer") is True


def _normalized_response(request: dict[str, Any]) -> str:
    text = request.get("response", {}).get("text")
    if not isinstance(text, str):
        return ""
    return " ".join(text.split())


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
