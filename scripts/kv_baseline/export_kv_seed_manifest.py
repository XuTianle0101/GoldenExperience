#!/usr/bin/env python3
"""Export a small, Git-trackable manifest for a KV baseline seed run."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import re
import sys
import time
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "goldenexperience.kv_seed_manifest.v1"
PROM_VALUE_RE = re.compile(
    r"^(?P<name>[^#{\s]+)(?:\{(?P<labels>[^}]*)\})?\s+(?P<value>[-+0-9.eE]+)"
)


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return data


def sha256_file(path: Path) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_size(path: Path | None) -> int | None:
    if path is None or not path.exists() or not path.is_file():
        return None
    return path.stat().st_size


def relative_or_none(path: str | Path | None, base: Path) -> str | None:
    if path is None:
        return None
    candidate = Path(path)
    try:
        return str(candidate.resolve().relative_to(base.resolve()))
    except ValueError:
        return str(candidate)


def relative_samples(samples: list[Any], base: Path) -> list[str]:
    rel: list[str] = []
    for sample in samples[:10]:
        if isinstance(sample, str):
            rel.append(relative_or_none(sample, base) or sample)
    return rel


def parse_labels(raw: str | None) -> dict[str, str]:
    if not raw:
        return {}
    labels: dict[str, str] = {}
    for part in re.finditer(r'([^=,]+)="((?:\\.|[^"])*)"', raw):
        labels[part.group(1)] = part.group(2).replace(r"\"", '"')
    return labels


def prometheus_value(path: Path, name: str, labels: dict[str, str] | None = None) -> float | None:
    if not path.exists():
        return None
    wanted = labels or {}
    value: float | None = None
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line or line.startswith("#"):
            continue
        match = PROM_VALUE_RE.match(line)
        if not match or match.group("name") != name:
            continue
        sample_labels = parse_labels(match.group("labels"))
        if any(sample_labels.get(k) != v for k, v in wanted.items()):
            continue
        value = float(match.group("value"))
    return value


def count_in_file(path: Path, needle: str) -> int:
    if not path.exists():
        return 0
    count = 0
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            count += line.count(needle)
    return count


def package_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for dist in ["goldenexperience", "vllm", "lmcache", "mooncake"]:
        try:
            versions[dist] = importlib.metadata.version(dist)
        except importlib.metadata.PackageNotFoundError:
            versions[dist] = None
    return versions


def request_summary(request: dict[str, Any]) -> dict[str, Any]:
    prompt = request.get("prompt") if isinstance(request.get("prompt"), dict) else {}
    timing = request.get("timing") if isinstance(request.get("timing"), dict) else {}
    usage = request.get("usage") if isinstance(request.get("usage"), dict) else {}
    return {
        "prompt_id": prompt.get("id"),
        "prompt_sha256": prompt.get("sha256"),
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "total_tokens": usage.get("total_tokens"),
        "e2e_ms": timing.get("e2e_ms"),
        "ttft_ms": timing.get("ttft_ms"),
    }


def storage_summary(
    summary: dict[str, Any],
    metadata: dict[str, Any],
    run_dir: Path,
) -> dict[str, Any]:
    mooncake = metadata.get("mooncake") if isinstance(metadata.get("mooncake"), dict) else {}
    use_mooncake = bool(mooncake.get("enabled"))
    source = summary.get("mooncake_storage" if use_mooncake else "cache")
    if not isinstance(source, dict):
        source = {}
    root_value = source.get("path") or mooncake.get("storage_root") or metadata.get("kv_cache_dir")
    root = Path(str(root_value or ""))
    sample_base = root if str(root) else run_dir
    return {
        "kind": "mooncake" if use_mooncake else "lmcache_fs",
        "root_relative_to_run": relative_or_none(root, run_dir),
        "file_count": source.get("file_count"),
        "total_bytes": source.get("total_bytes"),
        "sample_files_relative_to_root": relative_samples(
            source.get("sample_files", []),
            sample_base,
        ),
    }


def build_manifest(
    run_dir: Path,
    *,
    artifact_uri: str | None = None,
    bundle_path: Path | None = None,
    notes: list[str] | None = None,
) -> dict[str, Any]:
    run_dir = run_dir.resolve()
    metadata = load_json(run_dir / "metadata.json")
    summary = load_json(run_dir / "summary.json")
    if not metadata:
        raise FileNotFoundError(f"Missing metadata.json under {run_dir}")

    run_id = str(metadata.get("run_id") or run_dir.name)
    prompt_path = Path(str(metadata.get("prompt_file") or ""))
    if not prompt_path.exists() and prompt_path.name:
        candidate = run_dir / prompt_path.name
        if candidate.exists():
            prompt_path = candidate

    metrics_dir = run_dir / "metrics"
    offload_prom = metrics_dir / "offload.prom"
    reuse_prom = metrics_dir / "reuse.prom"
    lmcache_log = run_dir / "logs" / "lmcache_mp_server.log"
    reuse_lmcache_log = run_dir / "logs" / "reuse_lmcache_mp_server.log"
    evidence = summary.get("evidence") if isinstance(summary.get("evidence"), dict) else {}
    requests = summary.get("requests") if isinstance(summary.get("requests"), dict) else {}
    lmcache_mp = metadata.get("lmcache_mp") if isinstance(metadata.get("lmcache_mp"), dict) else {}
    mooncake = metadata.get("mooncake") if isinstance(metadata.get("mooncake"), dict) else {}

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "artifact": {
            "artifact_uri": artifact_uri,
            "bundle_path": str(bundle_path) if bundle_path else None,
            "bundle_sha256": sha256_file(bundle_path) if bundle_path else None,
            "bundle_bytes": file_size(bundle_path),
            "raw_run_directory_is_git_ignored": True,
        },
        "run": {
            "run_id": run_id,
            "source_run_dir": relative_or_none(run_dir, Path.cwd()),
            "mode": metadata.get("mode"),
            "engine": metadata.get("engine"),
            "kv_backend": metadata.get("kv_backend"),
            "model_name": metadata.get("model_name"),
            "model_path": metadata.get("model_path"),
            "prompt_id": metadata.get("prompt_id"),
            "prompt_file_relative_to_run": relative_or_none(prompt_path, run_dir),
            "prompt_file_sha256": sha256_file(prompt_path),
            "chunk_size": metadata.get("chunk_size"),
            "hash_algorithm": metadata.get("hash_algorithm"),
            "generated_disk_prompt": metadata.get("generated_disk_prompt"),
        },
        "runtime": package_versions(),
        "l2": {
            "adapter_type": lmcache_mp.get("l2_adapter_type"),
            "l2_store_policy": lmcache_mp.get("l2_store_policy"),
            "mooncake_enabled": mooncake.get("enabled"),
            "mooncake_protocol": mooncake.get("protocol"),
            "mooncake_master_server_addr": mooncake.get("master_server_addr"),
            "mooncake_metadata_server": mooncake.get("metadata_server"),
            "mooncake_cluster_id": run_id,
        },
        "storage": storage_summary(summary, metadata, run_dir),
        "proof": {
            "offload_has_disk_evidence": evidence.get("offload_has_disk_evidence"),
            "reuse_has_cache_evidence": evidence.get("reuse_has_cache_evidence"),
            "disk_reuse_success": evidence.get("disk_reuse_success"),
            "metrics": {
                "offload_external_prefix_cache_hits_total": prometheus_value(
                    offload_prom, "vllm:external_prefix_cache_hits_total"
                ),
                "reuse_external_prefix_cache_hits_total": prometheus_value(
                    reuse_prom, "vllm:external_prefix_cache_hits_total"
                ),
                "offload_prompt_tokens_local_compute": prometheus_value(
                    offload_prom,
                    "vllm:prompt_tokens_by_source_total",
                    {"source": "local_compute"},
                ),
                "reuse_prompt_tokens_local_compute": prometheus_value(
                    reuse_prom,
                    "vllm:prompt_tokens_by_source_total",
                    {"source": "local_compute"},
                ),
                "reuse_prompt_tokens_external_kv_transfer": prometheus_value(
                    reuse_prom,
                    "vllm:prompt_tokens_by_source_total",
                    {"source": "external_kv_transfer"},
                ),
            },
            "log_counts": {
                "mooncake_store_set": count_in_file(lmcache_log, "MooncakeStore SET"),
                "mooncake_store_get": count_in_file(lmcache_log, "MooncakeStore GET"),
                "reuse_mooncake_store_get": count_in_file(reuse_lmcache_log, "MooncakeStore GET"),
                "reuse_l2_prefetch_load_completed": count_in_file(
                    reuse_lmcache_log, "L2 prefetch load completed"
                ),
            },
        },
        "requests": {
            "offload": request_summary(requests.get("offload", {})),
            "reuse": request_summary(requests.get("reuse", {})),
            "deltas": summary.get("deltas", {}),
        },
        "restore_notes": [
            (
                "Restore the external seed payload outside Git, then set "
                "GE_MOONCAKE_STORAGE_ROOT to the restored cache/mooncake path."
            ),
            (
                "For the current Python Mooncake adapter, keep LMCache MP alive across "
                "vLLM restarts or provide a persistent key-index sidecar before "
                "expecting reuse after an LMCache MP restart."
            ),
        ],
    }
    if notes:
        manifest["notes"] = notes
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path, help="KV baseline run directory.")
    parser.add_argument("--output", type=Path, help="Output manifest path. Default: stdout.")
    parser.add_argument("--artifact-uri", help="External URI for the large KV seed payload.")
    parser.add_argument("--bundle-path", type=Path, help="Optional local bundle to hash.")
    parser.add_argument("--note", action="append", default=[], help="Extra note to include.")
    args = parser.parse_args(argv)

    try:
        manifest = build_manifest(
            args.run_dir,
            artifact_uri=args.artifact_uri,
            bundle_path=args.bundle_path,
            notes=args.note,
        )
    except Exception as exc:
        print(f"failed to export KV seed manifest: {exc}", file=sys.stderr)
        return 1

    text = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    else:
        print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
