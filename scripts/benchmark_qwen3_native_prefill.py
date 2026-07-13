#!/usr/bin/env python3
"""Collect isolated vLLM native-prefill TTFT evidence for cached-KV cost gates."""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from goldenexperience.benchmarks.cached_kv_cost import build_native_prefill_report
from goldenexperience.runtime.cross_model_reuse import token_ids_from_prompt
from goldenexperience.size_variant import CachedKVBridgeManifest, model_spec_from_path


def _wait_for_server(base_url: str, process: subprocess.Popen[bytes], timeout: float) -> None:
    deadline = time.monotonic() + timeout
    last_error = ""
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"vLLM exited before readiness with code {process.returncode}")
        try:
            with urllib.request.urlopen(base_url + "/health", timeout=2) as response:
                if 200 <= response.status < 500:
                    return
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            last_error = str(exc)
        time.sleep(2)
    raise RuntimeError(f"timed out waiting for vLLM: {last_error}")


def _measure_ttft(
    *,
    base_url: str,
    model: str,
    token_ids: list[int],
    timeout: float,
) -> tuple[float, int]:
    payload = {
        "model": model,
        "prompt": token_ids,
        "add_special_tokens": False,
        "max_tokens": 1,
        "temperature": 0,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    request = urllib.request.Request(
        base_url + "/v1/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started = time.perf_counter()
    first_token_ms: float | None = None
    prompt_tokens: int | None = None
    with urllib.request.urlopen(request, timeout=timeout) as response:
        for raw_line in response:
            line = raw_line.decode("utf-8", errors="replace").strip()
            if not line.startswith("data:"):
                continue
            data = line[len("data:") :].strip()
            if data == "[DONE]":
                break
            chunk = json.loads(data)
            usage = chunk.get("usage")
            if isinstance(usage, dict) and usage.get("prompt_tokens") is not None:
                prompt_tokens = int(usage["prompt_tokens"])
            for choice in chunk.get("choices", []):
                if isinstance(choice, dict) and choice.get("text"):
                    first_token_ms = (time.perf_counter() - started) * 1000
                    break
    if first_token_ms is None or prompt_tokens is None:
        raise RuntimeError("vLLM completion did not return TTFT and prompt usage")
    return first_token_ms, prompt_tokens


def _stop_process(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is not None:
        return
    os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=30)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=30)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-manifest", type=Path, required=True)
    parser.add_argument("--target-model", required=True)
    parser.add_argument("--prompt-file", type=Path, required=True)
    parser.add_argument("--prompt-id", required=True)
    parser.add_argument("--token-count", type=int, required=True)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--warmup-iterations", type=int, default=3)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=30000)
    parser.add_argument("--startup-timeout", type=float, default=600)
    parser.add_argument("--request-timeout", type=float, default=600)
    parser.add_argument("--vllm-bin", default="vllm")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--model-identity-cache", type=Path)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.iterations <= 0 or args.warmup_iterations < 0 or args.token_count <= 0:
        raise ValueError("benchmark counts must be positive")

    manifest = CachedKVBridgeManifest.load(args.candidate_manifest)
    artifact_errors = manifest.artifact_errors()
    if artifact_errors:
        raise ValueError("; ".join(artifact_errors))
    observed_target = model_spec_from_path(
        args.target_model,
        model_id=manifest.target.model_id,
        parameter_count_b=manifest.target.parameter_count_b,
        revision=manifest.target.revision,
        identity_cache_path=args.model_identity_cache,
    )
    model_identity_verified = observed_target == manifest.target
    if not model_identity_verified:
        raise ValueError("target model identity differs from candidate manifest")

    all_token_ids = token_ids_from_prompt(
        tokenizer_path=args.target_model,
        prompt_file=args.prompt_file,
        prompt_id=args.prompt_id,
    )
    if len(all_token_ids) < args.token_count:
        raise ValueError("prompt is shorter than --token-count")
    token_ids = all_token_ids[: args.token_count]
    base_url = f"http://{args.host}:{args.port}"
    command = [
        args.vllm_bin,
        "serve",
        args.target_model,
        "--host",
        args.host,
        "--port",
        str(args.port),
        "--served-model-name",
        manifest.target.model_id,
        "--no-enable-prefix-caching",
        "--gpu-memory-utilization",
        str(args.gpu_memory_utilization),
        "--max-model-len",
        str(max(args.token_count + 128, 512)),
    ]
    args.log.parent.mkdir(parents=True, exist_ok=True)
    with args.log.open("ab") as log_handle:
        process = subprocess.Popen(
            command,
            cwd=REPO_ROOT,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    try:
        _wait_for_server(base_url, process, args.startup_timeout)
        for _ in range(args.warmup_iterations):
            _measure_ttft(
                base_url=base_url,
                model=manifest.target.model_id,
                token_ids=token_ids,
                timeout=args.request_timeout,
            )
        samples: list[float] = []
        prompt_counts: list[int] = []
        for index in range(args.iterations):
            ttft_ms, prompt_tokens = _measure_ttft(
                base_url=base_url,
                model=manifest.target.model_id,
                token_ids=token_ids,
                timeout=args.request_timeout,
            )
            samples.append(ttft_ms)
            prompt_counts.append(prompt_tokens)
            print(f"native prefill {index + 1}/{args.iterations}: {ttft_ms:.3f} ms")
    finally:
        _stop_process(process)

    report = build_native_prefill_report(
        direction=manifest.direction,
        target_model_weights_sha256=manifest.target.weights_sha256,
        token_count=args.token_count,
        samples_ms=samples,
        warmup_iterations=args.warmup_iterations,
        model_identity_verified=model_identity_verified,
        prefix_caching_disabled=True,
        exact_token_count_verified=all(count == args.token_count for count in prompt_counts),
    )
    report.update(
        {
            "candidate_manifest": str(args.candidate_manifest.resolve()),
            "target_model_path": str(Path(args.target_model).resolve()),
            "prompt_file": str(args.prompt_file.resolve()),
            "prompt_id": args.prompt_id,
            "vllm_log": str(args.log.resolve()),
        }
    )
    _write_json(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["eligible_for_approval"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
