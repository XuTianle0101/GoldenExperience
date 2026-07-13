"""Runtime helpers for quality-gated cross-model KV reuse."""

from __future__ import annotations

import ctypes
import hashlib
import json
import math
import os
import struct
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
    request_id: str | None = None,
    expected_chunk_hashes: list[str] | None = None,
    expected_seq_len: int | None = None,
    expected_chunk_size: int | None = None,
    min_chunks: int = 1,
) -> dict[str, Any] | None:
    """Select only a lookup record bound to the current prompt request."""

    if min_chunks <= 0 or (not request_id and expected_chunk_hashes is None):
        return None
    normalized_expected = (
        [_normalize_chunk_hash(item) for item in expected_chunk_hashes]
        if expected_chunk_hashes is not None
        else None
    )
    candidates = [
        record
        for record in records
        if _lookup_record_matches(
            record,
            model_name=model_name,
            request_id=request_id,
            expected_chunk_hashes=normalized_expected,
            expected_seq_len=expected_seq_len,
            expected_chunk_size=expected_chunk_size,
            min_chunks=min_chunks,
        )
    ]
    if not candidates:
        return None
    return max(
        candidates,
        key=_lookup_record_timestamp,
    )


def common_chunk_hash_prefix(left: list[str], right: list[str]) -> list[str]:
    """Return only the exact rolling-hash prefix shared by two prompts."""

    matched: list[str] = []
    for left_item, right_item in zip(left, right, strict=False):
        normalized_left = _normalize_chunk_hash(left_item)
        if normalized_left != _normalize_chunk_hash(right_item):
            break
        matched.append(normalized_left)
    return matched


def lmcache_chunk_hashes(
    token_ids: list[int],
    *,
    chunk_size: int,
    hash_algorithm: str,
) -> list[str]:
    """Compute deterministic LMCache rolling hashes for complete token chunks."""

    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if hash_algorithm != "blake3":
        raise ValueError("cross-prompt lookup requires the deterministic blake3 hash algorithm")
    for token_id in token_ids:
        if token_id < 0 or token_id > 0xFFFFFFFF:
            raise ValueError("token ids must fit in an unsigned 32-bit integer")

    import blake3

    def hash_tokens(prefix_hash: int | bytes, tokens: list[int]) -> bytes:
        digest = blake3.blake3()
        if isinstance(prefix_hash, bytes):
            digest.update(prefix_hash)
        else:
            digest.update(prefix_hash.to_bytes(8, byteorder="big", signed=True))
        digest.update(struct.pack(f">{len(tokens)}I", *tokens))
        return digest.digest()

    prefix_hash = hash_tokens(0, [0])
    hashes: list[str] = []
    complete_tokens = len(token_ids) - len(token_ids) % chunk_size
    for start in range(0, complete_tokens, chunk_size):
        prefix_hash = hash_tokens(prefix_hash, token_ids[start : start + chunk_size])
        hashes.append("0x" + prefix_hash.hex())
    return hashes


def select_shared_prefix_candidate(
    records: list[dict[str, Any]],
    *,
    model_name: str,
    source_request_id: str,
    source_token_ids: list[int],
    target_token_ids: list[int],
    chunk_size: int,
    hash_algorithm: str,
    min_chunks: int = 1,
) -> dict[str, Any] | None:
    """Bind a source request to the exact complete prefix shared by a target prompt."""

    if not source_request_id or min_chunks <= 0:
        return None
    source_hashes = lmcache_chunk_hashes(
        source_token_ids,
        chunk_size=chunk_size,
        hash_algorithm=hash_algorithm,
    )
    target_hashes = lmcache_chunk_hashes(
        target_token_ids,
        chunk_size=chunk_size,
        hash_algorithm=hash_algorithm,
    )
    record = select_lookup_candidate(
        records,
        model_name=model_name,
        request_id=source_request_id,
        expected_chunk_hashes=source_hashes,
        expected_seq_len=len(source_token_ids),
        expected_chunk_size=chunk_size,
        min_chunks=min_chunks,
    )
    if record is None:
        return None
    shared_hashes = common_chunk_hash_prefix(source_hashes, target_hashes)
    if len(shared_hashes) < min_chunks:
        return None
    lookup_request_id = record.get("request_id")
    return {
        "record": record,
        "chunk_hashes": shared_hashes,
        "binding": {
            "source_response_id": source_request_id,
            "lookup_request_id": lookup_request_id,
            "token_count": len(source_token_ids),
            "token_ids_sha256": token_ids_sha256(source_token_ids),
            "target_token_count": len(target_token_ids),
            "target_token_ids_sha256": token_ids_sha256(target_token_ids),
            "chunk_size": chunk_size,
            "shared_prefix_chunk_count": len(shared_hashes),
            "shared_prefix_token_count": len(shared_hashes) * chunk_size,
            "hash_algorithm": hash_algorithm,
        },
    }


def token_ids_sha256(token_ids: list[int]) -> str:
    """Create a stable prompt identity without storing prompt token contents."""

    digest = hashlib.sha256()
    digest.update(len(token_ids).to_bytes(8, "big"))
    for token_id in token_ids:
        if token_id < 0 or token_id > 0xFFFFFFFF:
            raise ValueError("token ids must fit in an unsigned 32-bit integer")
        digest.update(struct.pack(">I", token_id))
    return digest.hexdigest()


def _request_ids_match(record_id: str, expected_id: str) -> bool:
    return record_id == expected_id or record_id.startswith(expected_id + "-")


def _lookup_record_matches(
    record: dict[str, Any],
    *,
    model_name: str,
    request_id: str | None,
    expected_chunk_hashes: list[str] | None,
    expected_seq_len: int | None,
    expected_chunk_size: int | None,
    min_chunks: int,
) -> bool:
    try:
        hashes = record.get("chunk_hashes")
        if not isinstance(hashes, list) or len(hashes) < min_chunks:
            return False
        if record.get("model_name") != model_name:
            return False
        if request_id is not None and not _request_ids_match(
            str(record.get("request_id") or ""), request_id
        ):
            return False
        if expected_seq_len is not None and int(record.get("seq_len")) != expected_seq_len:
            return False
        if expected_chunk_size is not None and int(record.get("chunk_size")) != expected_chunk_size:
            return False
        if (
            expected_seq_len is not None
            and expected_chunk_size is not None
            and len(hashes) != expected_seq_len // expected_chunk_size
        ):
            return False
        if expected_chunk_hashes is not None:
            return [_normalize_chunk_hash(item) for item in hashes] == expected_chunk_hashes
        float(record.get("timestamp") or 0.0)
        return True
    except (TypeError, ValueError):
        return False


def _lookup_record_timestamp(record: dict[str, Any]) -> float:
    return float(record.get("timestamp") or 0.0)


def _normalize_chunk_hash(value: str) -> str:
    text = str(value).lower()
    if text.startswith("0x"):
        text = text[2:]
    if not text:
        raise ValueError("chunk hash is empty")
    int(text, 16)
    return "0x" + text


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
    materializer_elapsed_ms = _finite_non_negative(materializer.get("elapsed_ms"))
    native_ttft_ms = _request_timing(native_request, "ttft_ms")
    reuse_ttft_ms = _request_timing(reuse_request, "ttft_ms")
    total_reuse_ttft_ms = (
        materializer_elapsed_ms + reuse_ttft_ms
        if materializer_elapsed_ms is not None and reuse_ttft_ms is not None
        else None
    )

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
            injected_keys and injected_keys <= target_keys and injected_count == len(injected_keys)
        ),
        "target_keys_present_after": bool(
            injected_keys and after_found == injected_keys and not (after_missing & injected_keys)
        ),
        "external_token_count": bool(
            isinstance(target_external_tokens, (int, float))
            and target_external_tokens == expected_external_tokens
            and expected_external_tokens > 0
        ),
        "end_to_end_ttft_improved": bool(
            native_ttft_ms is not None
            and total_reuse_ttft_ms is not None
            and total_reuse_ttft_ms < native_ttft_ms
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
        "timing": {
            "materializer_elapsed_ms": materializer_elapsed_ms,
            "native_target_ttft_ms": native_ttft_ms,
            "reuse_target_ttft_ms": reuse_ttft_ms,
            "materialization_plus_reuse_ttft_ms": total_reuse_ttft_ms,
            "end_to_end_ttft_savings_ms": (
                native_ttft_ms - total_reuse_ttft_ms
                if native_ttft_ms is not None and total_reuse_ttft_ms is not None
                else None
            ),
        },
        "consumed_materialized_kv": consumed,
        "success": success,
        "status": status,
    }


def _all_checks_pass(checks: Any) -> bool:
    return (
        isinstance(checks, dict)
        and bool(checks)
        and all(value is True for value in checks.values())
    )


def _finite_non_negative(value: Any) -> float | None:
    if not isinstance(value, (int, float)):
        return None
    parsed = float(value)
    return parsed if math.isfinite(parsed) and parsed >= 0 else None


def _request_timing(request: dict[str, Any], name: str) -> float | None:
    parsed = _finite_non_negative(request.get("timing", {}).get(name))
    return parsed if parsed is not None and parsed > 0 else None


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
    encoded: Any = tokenizer.apply_chat_template(
        prompt["messages"],
        tokenize=True,
        add_generation_prompt=True,
    )
    ids = encoded["input_ids"] if hasattr(encoded, "keys") and "input_ids" in encoded else encoded
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
