#!/usr/bin/env python3
"""Small stdlib-only client for OpenAI-compatible KV offload/reuse experiments."""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


def _now_ms(start: float) -> float:
    return (time.perf_counter() - start) * 1000.0


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return data


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _http_get(url: str, timeout: float) -> tuple[int, bytes]:
    request = urllib.request.Request(url, method="GET")
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return int(response.status), response.read()


def _load_prompt(path: Path, prompt_id: str | None) -> dict[str, Any]:
    manifest = _read_json(path)
    prompts = manifest.get("prompts")
    if not isinstance(prompts, list) or not prompts:
        raise ValueError(f"{path} must contain a non-empty 'prompts' list")

    selected_id = prompt_id or str(manifest.get("default_prompt_id", ""))
    for prompt in prompts:
        if isinstance(prompt, dict) and prompt.get("id") == selected_id:
            return prompt
    known = ", ".join(str(p.get("id")) for p in prompts if isinstance(p, dict))
    raise ValueError(f"Prompt id {selected_id!r} not found in {path}; known ids: {known}")


def _extract_text_from_non_stream(response: dict[str, Any]) -> str:
    pieces: list[str] = []
    for choice in response.get("choices", []):
        if not isinstance(choice, dict):
            continue
        message = choice.get("message")
        if isinstance(message, dict) and isinstance(message.get("content"), str):
            pieces.append(message["content"])
        elif isinstance(choice.get("text"), str):
            pieces.append(choice["text"])
    return "".join(pieces)


def _extract_text_from_stream_chunk(chunk: dict[str, Any]) -> str:
    pieces: list[str] = []
    for choice in chunk.get("choices", []):
        if not isinstance(choice, dict):
            continue
        delta = choice.get("delta")
        if isinstance(delta, dict) and isinstance(delta.get("content"), str):
            pieces.append(delta["content"])
        elif isinstance(choice.get("text"), str):
            pieces.append(choice["text"])
    return "".join(pieces)


def _extract_final_answer(text: str) -> str | None:
    matches = re.findall(r"(?im)^\s*Final answer:\s*(.+?)\s*$", text)
    return matches[-1].strip() if matches else None


def wait_for_server(args: argparse.Namespace) -> int:
    deadline = time.monotonic() + args.timeout
    last_error = ""
    endpoints = ("/health", "/v1/models")
    while time.monotonic() < deadline:
        for endpoint in endpoints:
            try:
                status, _ = _http_get(args.base_url.rstrip("/") + endpoint, timeout=args.interval)
                if 200 <= status < 500:
                    print(f"Server is reachable at {args.base_url} ({endpoint} -> {status})")
                    return 0
            except (urllib.error.URLError, TimeoutError, OSError) as exc:
                last_error = str(exc)
        time.sleep(args.interval)
    print(f"Timed out waiting for {args.base_url}: {last_error}", file=sys.stderr)
    return 1


def fetch_metrics(args: argparse.Namespace) -> int:
    output = Path(args.output)
    try:
        status, body = _http_get(args.base_url.rstrip("/") + "/metrics", timeout=args.timeout)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(body)
        print(f"Wrote metrics snapshot ({status}) to {output}")
        return 0
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        _write_json(
            output.with_suffix(output.suffix + ".error.json"),
            {"base_url": args.base_url, "error": str(exc), "endpoint": "/metrics"},
        )
        print(f"Metrics endpoint is not available: {exc}", file=sys.stderr)
        return 0 if args.allow_missing else 1


def run_request(args: argparse.Namespace) -> int:
    prompt = _load_prompt(Path(args.prompt_file), args.prompt_id)
    generation = prompt.get("generation", {})
    if not isinstance(generation, dict):
        generation = {}

    max_tokens = (
        args.max_tokens if args.max_tokens is not None else generation.get("max_tokens", 128)
    )
    temperature = (
        args.temperature if args.temperature is not None else generation.get("temperature", 0)
    )
    payload: dict[str, Any] = {
        "model": args.model,
        "messages": prompt["messages"],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": args.stream,
    }
    if args.stream and args.include_usage:
        payload["stream_options"] = {"include_usage": True}

    url = args.base_url.rstrip("/") + "/v1/chat/completions"
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    started_unix = time.time()
    started = time.perf_counter()
    first_byte_ms: float | None = None
    first_token_ms: float | None = None
    response_text = ""
    usage: dict[str, Any] | None = None
    response_id: str | None = None

    try:
        with urllib.request.urlopen(request, timeout=args.timeout) as response:
            if args.stream:
                for raw_line in response:
                    now = _now_ms(started)
                    if first_byte_ms is None:
                        first_byte_ms = now
                    line = raw_line.decode("utf-8", errors="replace").strip()
                    if not line.startswith("data:"):
                        continue
                    data = line[len("data:") :].strip()
                    if data == "[DONE]":
                        break
                    chunk = json.loads(data)
                    response_id = response_id or chunk.get("id")
                    if isinstance(chunk.get("usage"), dict):
                        usage = chunk["usage"]
                    delta_text = _extract_text_from_stream_chunk(chunk)
                    if delta_text and first_token_ms is None:
                        first_token_ms = now
                    response_text += delta_text
            else:
                body = response.read()
                if first_byte_ms is None:
                    first_byte_ms = _now_ms(started)
                parsed = json.loads(body.decode("utf-8"))
                response_id = parsed.get("id")
                usage = parsed.get("usage") if isinstance(parsed.get("usage"), dict) else None
                response_text = _extract_text_from_non_stream(parsed)
    except urllib.error.HTTPError as exc:
        error_body = exc.read().decode("utf-8", errors="replace")
        if exc.code == 400 and "context length" in error_body.lower():
            error_body += (
                "\nHint: for generated disk-offload prompts, lower GE_DISK_PROMPT_REPEAT "
                "or increase the model context length if the backend supports it."
            )
        raise RuntimeError(f"HTTP {exc.code} from {url}: {error_body}") from exc

    ended_unix = time.time()
    expected = str(prompt.get("expected_final_answer", "")).strip()
    extracted_final_answer = _extract_final_answer(response_text)
    normalized_response = " ".join(response_text.split())
    normalized_expected = " ".join(expected.split())
    matches_expected = bool(
        expected
        and (
            extracted_final_answer == expected
            or normalized_response == normalized_expected
        )
    )
    result = {
        "phase": args.phase,
        "base_url": args.base_url,
        "model": args.model,
        "prompt": {
            "id": prompt.get("id"),
            "dataset": prompt.get("dataset"),
            "split": prompt.get("split"),
            "source": prompt.get("source"),
            "expected_final_answer": expected or None,
            "message_count": len(prompt.get("messages", [])),
            "message_chars": sum(len(m.get("content", "")) for m in prompt.get("messages", [])),
        },
        "request": {
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": args.stream,
            "include_usage": args.include_usage,
        },
        "response": {
            "id": response_id,
            "text": response_text,
            "chars": len(response_text),
            "contains_expected_final_answer": bool(expected and expected in response_text),
            "extracted_final_answer": extracted_final_answer,
            "matches_expected_final_answer": matches_expected,
        },
        "usage": usage,
        "timing": {
            "started_unix": started_unix,
            "ended_unix": ended_unix,
            "e2e_ms": _now_ms(started),
            "ttfb_ms": first_byte_ms,
            "ttft_ms": first_token_ms,
        },
    }
    _write_json(Path(args.output), result)
    print(
        f"{args.phase}: e2e={result['timing']['e2e_ms']:.1f}ms "
        f"ttft={result['timing']['ttft_ms']} output={Path(args.output)}"
    )
    return 0


def _sum_regex_int(pattern: str, text: str) -> int:
    total = 0
    for match in re.finditer(pattern, text, flags=re.IGNORECASE):
        try:
            total += int(match.group(1))
        except (IndexError, ValueError):
            continue
    return total


def _count_regex(pattern: str, text: str) -> int:
    return len(re.findall(pattern, text, flags=re.IGNORECASE))


def _max_prefill_cached_tokens(text: str) -> int | None:
    values: list[int] = []
    patterns = [
        r"#cached-token:\s*([0-9]+)",
        r"cached token(?:s)?:\s*([0-9]+)",
        r"cache hit token(?:s)?:\s*([0-9]+)",
    ]
    for pattern in patterns:
        for match in re.finditer(pattern, text, flags=re.IGNORECASE):
            values.append(int(match.group(1)))
    return max(values) if values else None


def _metric_values(text: str, metric_name: str) -> list[float]:
    number = r"([-+]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][-+]?[0-9]+)?|NaN|Inf|-Inf)"
    pattern = re.compile(
        rf"^{re.escape(metric_name)}(?:\{{[^}}]*\}})?\s+{number}\s*$",
        flags=re.MULTILINE,
    )
    values: list[float] = []
    for match in pattern.finditer(text):
        try:
            values.append(float(match.group(1)))
        except ValueError:
            continue
    return values


def _summarize_metrics(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8", errors="replace")
    cache_hit_rates = _metric_values(text, "vllm:cache_hit_rate")
    prompt_sums = _metric_values(text, "vllm:prompt_tokens_histogram_sum")
    uncached_prompt_sums = _metric_values(text, "vllm:uncached_prompt_tokens_histogram_sum")
    return {
        "path": str(path),
        "cache_hit_rate_max": max(cache_hit_rates) if cache_hit_rates else None,
        "prompt_tokens_histogram_sum": sum(prompt_sums) if prompt_sums else None,
        "uncached_prompt_tokens_histogram_sum": (
            sum(uncached_prompt_sums) if uncached_prompt_sums else None
        ),
    }


def _summarize_log(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8", errors="replace")
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "lmcache_store_events": _count_regex(
            r"\b(store|stored|put|offload|write|written|save|saved)", text
        ),
        "lmcache_strict_store_events": _count_regex(
            r"\blmcache\b.*\b(store|stored|put|offload|write|written|save|saved)\b|"
            r"\bl2\b.*\b(store|stored|write|written|save|saved)\b|"
            r"\bmooncake[_ -]?store\b.*\b(store|stored|put|set|write|written|save|saved)\b|"
            r"\bStoreController\b|\bSET\b|"
            r"\b(stored|offloaded|written|saved)\s+[0-9]+\s+"
            r"(?:tokens|token|chunks|chunk|objects|object)\b",
            text,
        ),
        "lmcache_retrieve_events": _count_regex(
            r"\b(retrieve|retrieved|lookup|hit|prefetch|prefetched|load|loaded|read)", text
        ),
        "lmcache_strict_retrieve_events": _count_regex(
            r"\blmcache\b.*\b"
            r"(retrieve|retrieved|lookup|hit|prefetch|prefetched|load|loaded|read)\b|"
            r"\bl2\b.*\b(retrieve|retrieved|hit|prefetch|prefetched|load|loaded|read)\b|"
            r"\bmooncake[_ -]?store\b.*\b"
            r"(exists|get|retrieve|retrieved|lookup|hit|prefetch|prefetched|load|loaded|read)\b|"
            r"\bPrefetchController\b|\b(EXISTS|GET)\b|"
            r"\b(retrieved|hit|prefetched|loaded|read)\s+[0-9]+\s+"
            r"(?:tokens|token|chunks|chunk|objects|object)\b",
            text,
        ),
        "lmcache_mooncake_store_mentions": _count_regex(
            r"\bmooncake[_ -]?store\b|\bMooncakeStore\b", text
        ),
        "lmcache_l2_controller_mentions": _count_regex(
            r"\b(StoreController|PrefetchController)\b|\bL2\b", text
        ),
        "lmcache_l2_operation_mentions": _count_regex(
            r"\b(EXISTS|GET|SET)\b|\bstorage_root_dir\b", text
        ),
        "lmcache_l2_retrieve_operation_mentions": _count_regex(r"\b(EXISTS|GET)\b", text),
        "lmcache_l2_store_operation_mentions": _count_regex(r"\bSET\b", text),
        "stored_token_mentions": _sum_regex_int(r"stored\s+([0-9]+)\s+(?:tokens|token)", text),
        "retrieved_token_mentions": _sum_regex_int(
            r"retrieved\s+([0-9]+)\s+(?:tokens|token)", text
        ),
        "max_cached_tokens_mentioned": _max_prefill_cached_tokens(text),
    }


def _summarize_cache_dir(path: Path) -> dict[str, Any]:
    files: list[Path] = []
    total_bytes = 0
    if path.exists():
        for item in path.rglob("*"):
            if item.is_file():
                files.append(item)
                total_bytes += item.stat().st_size
    return {
        "path": str(path),
        "exists": path.exists(),
        "file_count": len(files),
        "total_bytes": total_bytes,
        "sample_files": [str(item) for item in sorted(files)[:10]],
    }


def _metadata_path(value: Any, default: Path) -> Path:
    if isinstance(value, str) and value:
        path = Path(value)
        return path if path.is_absolute() else Path.cwd() / path
    return default


def summarize(args: argparse.Namespace) -> int:
    run_dir = Path(args.run_dir)
    request_dir = run_dir / "requests"
    log_dir = run_dir / "logs"
    metrics_dir = run_dir / "metrics"
    metadata_path = run_dir / "metadata.json"
    metadata = _read_json(metadata_path) if metadata_path.exists() else {}
    cache_dir = _metadata_path(metadata.get("kv_cache_dir"), run_dir / "cache")
    mooncake_metadata = metadata.get("mooncake", {})
    mooncake_storage_root = None
    if isinstance(mooncake_metadata, dict):
        mooncake_storage_root = _metadata_path(
            mooncake_metadata.get("storage_root"), run_dir / "cache" / "mooncake"
        )

    requests: dict[str, dict[str, Any]] = {}
    if request_dir.exists():
        for path in sorted(request_dir.glob("*.json")):
            item = _read_json(path)
            requests[str(item.get("phase") or path.stem)] = item

    logs = (
        [_summarize_log(path) for path in sorted(log_dir.glob("*.log"))]
        if log_dir.exists()
        else []
    )
    metrics = (
        {path.stem: _summarize_metrics(path) for path in sorted(metrics_dir.glob("*.prom"))}
        if metrics_dir.exists()
        else {}
    )
    cache = _summarize_cache_dir(cache_dir)
    mooncake_storage = (
        _summarize_cache_dir(mooncake_storage_root)
        if mooncake_storage_root is not None
        else None
    )

    def timing(phase: str, key: str) -> float | None:
        value = requests.get(phase, {}).get("timing", {}).get(key)
        return float(value) if isinstance(value, (int, float)) else None

    offload_ttft = timing("offload", "ttft_ms")
    reuse_ttft = timing("reuse", "ttft_ms")
    offload_e2e = timing("offload", "e2e_ms")
    reuse_e2e = timing("reuse", "e2e_ms")
    reuse_logs = [log for log in logs if Path(str(log["path"])).name.startswith("reuse")]
    reuse_retrieve_events = sum(int(log["lmcache_retrieve_events"]) for log in reuse_logs)
    reuse_strict_retrieve_events = sum(
        int(log["lmcache_strict_retrieve_events"]) for log in reuse_logs
    )
    reuse_l2_operation_mentions = sum(
        int(log["lmcache_l2_operation_mentions"]) for log in reuse_logs
    )
    reuse_l2_retrieve_operation_mentions = sum(
        int(log["lmcache_l2_retrieve_operation_mentions"]) for log in reuse_logs
    )
    strict_store_events = sum(int(log["lmcache_strict_store_events"]) for log in logs)
    reuse_cached_mentions = [
        int(log["max_cached_tokens_mentioned"])
        for log in reuse_logs
        if log["max_cached_tokens_mentioned"]
    ]
    reuse_max_cached_tokens = max(reuse_cached_mentions) if reuse_cached_mentions else None
    reuse_metric = metrics.get("reuse", {})
    reuse_cache_hit_rate = reuse_metric.get("cache_hit_rate_max")
    prompt_sum = reuse_metric.get("prompt_tokens_histogram_sum")
    uncached_prompt_sum = reuse_metric.get("uncached_prompt_tokens_histogram_sum")
    reuse_uncached_prompt_less_than_total = (
        isinstance(prompt_sum, (int, float))
        and isinstance(uncached_prompt_sum, (int, float))
        and uncached_prompt_sum < prompt_sum
    )
    reuse_has_cache_evidence = bool(
        reuse_strict_retrieve_events > 0
        or reuse_l2_retrieve_operation_mentions > 0
        or (reuse_max_cached_tokens is not None and reuse_max_cached_tokens > 0)
        or (isinstance(reuse_cache_hit_rate, (int, float)) and reuse_cache_hit_rate > 0)
        or reuse_uncached_prompt_less_than_total
    )
    disk_cache_file_count = int(cache["file_count"])
    disk_cache_total_bytes = int(cache["total_bytes"])
    mooncake_file_count = int(mooncake_storage["file_count"]) if mooncake_storage else 0
    mooncake_total_bytes = int(mooncake_storage["total_bytes"]) if mooncake_storage else 0
    cache_has_files = disk_cache_file_count > 0 and disk_cache_total_bytes > 0
    mooncake_has_files = mooncake_file_count > 0 and mooncake_total_bytes > 0
    offload_has_disk_evidence = cache_has_files or mooncake_has_files
    offload_disk_evidence_sources: list[str] = []
    if cache_has_files:
        offload_disk_evidence_sources.append(str(cache_dir))
    if mooncake_has_files and str(mooncake_storage_root) not in offload_disk_evidence_sources:
        offload_disk_evidence_sources.append(str(mooncake_storage_root))
    disk_reuse_success = offload_has_disk_evidence and reuse_has_cache_evidence
    evidence_warnings: list[str] = []
    if not offload_has_disk_evidence:
        evidence_warnings.append(
            "No disk cache files were found under the run cache directory or Mooncake "
            "storage root after offload."
        )
    if not reuse_has_cache_evidence:
        evidence_warnings.append(
            "No reuse-side cache evidence was found: retrieve events, cached-token mentions, "
            "cache hit rate, and uncached prompt token reduction are all absent or zero."
        )

    summary = {
        "run_dir": str(run_dir),
        "metadata": metadata,
        "requests": requests,
        "logs": logs,
        "metrics": metrics,
        "cache": cache,
        "mooncake_storage": mooncake_storage,
        "deltas": {
            "reuse_minus_offload_ttft_ms": (
                (
                    reuse_ttft - offload_ttft
                    if reuse_ttft is not None and offload_ttft is not None
                    else None
                )
            ),
            "reuse_minus_offload_e2e_ms": (
                (
                    reuse_e2e - offload_e2e
                    if reuse_e2e is not None and offload_e2e is not None
                    else None
                )
            ),
        },
        "evidence": {
            "lmcache_store_events_total": sum(int(log["lmcache_store_events"]) for log in logs),
            "lmcache_strict_store_events_total": strict_store_events,
            "lmcache_retrieve_events_total": sum(
                int(log["lmcache_retrieve_events"]) for log in logs
            ),
            "lmcache_strict_retrieve_events_total": sum(
                int(log["lmcache_strict_retrieve_events"]) for log in logs
            ),
            "lmcache_mooncake_store_mentions_total": sum(
                int(log["lmcache_mooncake_store_mentions"]) for log in logs
            ),
            "lmcache_l2_controller_mentions_total": sum(
                int(log["lmcache_l2_controller_mentions"]) for log in logs
            ),
            "lmcache_l2_operation_mentions_total": sum(
                int(log["lmcache_l2_operation_mentions"]) for log in logs
            ),
            "lmcache_l2_retrieve_operation_mentions_total": sum(
                int(log["lmcache_l2_retrieve_operation_mentions"]) for log in logs
            ),
            "lmcache_l2_store_operation_mentions_total": sum(
                int(log["lmcache_l2_store_operation_mentions"]) for log in logs
            ),
            "max_cached_tokens_mentioned": max(
                [
                    log["max_cached_tokens_mentioned"]
                    for log in logs
                    if log["max_cached_tokens_mentioned"]
                ]
                or [None]
            ),
            "reuse_lmcache_retrieve_events": reuse_retrieve_events,
            "reuse_lmcache_strict_retrieve_events": reuse_strict_retrieve_events,
            "reuse_l2_operation_mentions": reuse_l2_operation_mentions,
            "reuse_l2_retrieve_operation_mentions": reuse_l2_retrieve_operation_mentions,
            "reuse_max_cached_tokens_mentioned": reuse_max_cached_tokens,
            "reuse_cache_hit_rate_max": reuse_cache_hit_rate,
            "reuse_uncached_prompt_tokens_less_than_total": (
                reuse_uncached_prompt_less_than_total
            ),
            "disk_cache_file_count": disk_cache_file_count,
            "disk_cache_total_bytes": disk_cache_total_bytes,
            "mooncake_storage_file_count": mooncake_file_count,
            "mooncake_storage_total_bytes": mooncake_total_bytes,
            "offload_disk_evidence_sources": offload_disk_evidence_sources,
            "offload_has_disk_evidence": offload_has_disk_evidence,
            "reuse_has_cache_evidence": reuse_has_cache_evidence,
            "disk_reuse_success": disk_reuse_success,
            "warnings": evidence_warnings,
        },
    }
    _write_json(Path(args.output), summary)
    print(f"Wrote summary to {args.output}")
    if args.require_disk_offload and not offload_has_disk_evidence:
        print("error: disk offload evidence is absent", file=sys.stderr)
        return 3
    if args.require_reuse_evidence and not reuse_has_cache_evidence:
        print("error: reuse cache evidence is absent", file=sys.stderr)
        return 2
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    wait_parser = subparsers.add_parser("wait", help="wait for an OpenAI-compatible server")
    wait_parser.add_argument("--base-url", required=True)
    wait_parser.add_argument("--timeout", type=float, default=900)
    wait_parser.add_argument("--interval", type=float, default=2)
    wait_parser.set_defaults(func=wait_for_server)

    metrics_parser = subparsers.add_parser("fetch-metrics", help="save /metrics if available")
    metrics_parser.add_argument("--base-url", required=True)
    metrics_parser.add_argument("--output", required=True)
    metrics_parser.add_argument("--timeout", type=float, default=10)
    metrics_parser.add_argument("--allow-missing", action="store_true")
    metrics_parser.set_defaults(func=fetch_metrics)

    request_parser = subparsers.add_parser("request", help="send one chat completion request")
    request_parser.add_argument("--base-url", required=True)
    request_parser.add_argument("--model", required=True)
    request_parser.add_argument("--prompt-file", required=True)
    request_parser.add_argument("--prompt-id")
    request_parser.add_argument("--phase", required=True)
    request_parser.add_argument("--output", required=True)
    request_parser.add_argument("--max-tokens", type=int)
    request_parser.add_argument("--temperature", type=float)
    request_parser.add_argument("--timeout", type=float, default=600)
    request_parser.add_argument("--stream", dest="stream", action="store_true", default=True)
    request_parser.add_argument("--no-stream", dest="stream", action="store_false")
    request_parser.add_argument(
        "--include-usage", dest="include_usage", action="store_true", default=True
    )
    request_parser.add_argument(
        "--no-include-usage", dest="include_usage", action="store_false"
    )
    request_parser.set_defaults(func=run_request)

    summary_parser = subparsers.add_parser("summarize", help="summarize requests and logs")
    summary_parser.add_argument("--run-dir", required=True)
    summary_parser.add_argument("--output", required=True)
    summary_parser.add_argument("--require-disk-offload", action="store_true")
    summary_parser.add_argument("--require-reuse-evidence", action="store_true")
    summary_parser.set_defaults(func=summarize)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    try:
        return int(args.func(args))
    except Exception as exc:  # noqa: BLE001 - this is a CLI boundary.
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
