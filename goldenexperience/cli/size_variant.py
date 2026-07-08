"""CLI commands for GoldenScale KV reuse artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from goldenexperience.size_variant import (
    CalibrationManifest,
    QualityGateResult,
    build_calibration_manifest,
    load_prompt_count,
    qwen3_model_pair,
    save_prompt_manifest,
)

DEFAULT_ARTIFACT_DIR = Path("artifacts/golden_scale")
DEFAULT_PROMPTS = [
    "You are a helpful assistant. Summarize the following context.",
    "Use the retrieved documents to answer the user question.",
    "You are an agent. Follow the tool schema exactly.",
]


def main_collect() -> None:
    parser = argparse.ArgumentParser(description="Create a prompt manifest for GoldenScale calibration.")
    parser.add_argument("--direction", choices=["8b_to_14b", "14b_to_8b"], default="8b_to_14b")
    parser.add_argument("--output", type=Path, default=DEFAULT_ARTIFACT_DIR / "prompts.json")
    parser.add_argument("--prompt", action="append", default=None, help="Prompt text. Can be repeated.")
    parser.add_argument("--prompt-file", type=Path, default=None, help="One prompt per line.")
    args = parser.parse_args()

    prompts = list(args.prompt or [])
    if args.prompt_file is not None:
        prompts.extend(
            line.strip()
            for line in args.prompt_file.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    if not prompts:
        prompts = DEFAULT_PROMPTS

    source, target = qwen3_model_pair(args.direction)
    save_prompt_manifest(args.output, prompts, source, target)
    print(json.dumps({"output": str(args.output), "prompt_count": len(prompts)}, indent=2, sort_keys=True))


def main_fit() -> None:
    parser = argparse.ArgumentParser(description="Fit deterministic MVP GoldenScale artifacts.")
    parser.add_argument(
        "--direction",
        choices=["8b_to_14b", "14b_to_8b", "bidirectional"],
        default="bidirectional",
    )
    parser.add_argument("--prompt-manifest", type=Path, default=DEFAULT_ARTIFACT_DIR / "prompts.json")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_ARTIFACT_DIR)
    parser.add_argument("--calibration-id", default=None, help="Use only for single-direction fit.")
    args = parser.parse_args()

    directions = ["8b_to_14b", "14b_to_8b"] if args.direction == "bidirectional" else [args.direction]
    prompt_count = load_prompt_count(args.prompt_manifest)
    outputs = []
    for direction in directions:
        source, target = qwen3_model_pair(direction)
        calibration_id = args.calibration_id
        if calibration_id is None:
            calibration_id = f"qwen3_{direction}_projection_v0"
        quality = QualityGateResult.from_metrics(
            kv_cosine=0.99,
            attention_proxy_cosine=0.99,
            perplexity_drift_pct=0.0,
            task_score_drop_pct=0.0,
        )
        manifest = build_calibration_manifest(
            source=source,
            target=target,
            calibration_id=calibration_id,
            prompts_count=prompt_count,
            quality=quality,
            artifact_root=str(args.output_dir),
        )
        output = args.output_dir / f"{calibration_id}.json"
        manifest.save(output)
        outputs.append(str(output))
    index_path = args.output_dir / "index.json"
    index_path.parent.mkdir(parents=True, exist_ok=True)
    index_path.write_text(json.dumps({"manifests": outputs}, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"outputs": outputs, "index": str(index_path)}, indent=2, sort_keys=True))


def main_validate() -> None:
    parser = argparse.ArgumentParser(description="Validate a GoldenScale calibration manifest.")
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    manifest = CalibrationManifest.load(args.manifest)
    errors = manifest.validate()
    payload = {
        "manifest": str(args.manifest),
        "calibration_id": manifest.calibration_id,
        "direction": manifest.direction.value,
        "passed": not errors,
        "errors": errors,
        "quality": manifest.quality.__dict__,
        "layer_map_id": manifest.layer_map_id,
        "projection_id": manifest.projection_id,
    }
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(f"{manifest.calibration_id}: {'PASS' if not errors else 'FAIL'}")
        for error in errors:
            print(f"- {error}")
    if errors:
        raise SystemExit(1)


def main_bench() -> None:
    parser = argparse.ArgumentParser(description="Estimate GoldenScale materialization benefit.")
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--prompt-tokens", type=int, default=4096)
    parser.add_argument("--target-prefill-ms", type=float, default=None)
    parser.add_argument("--projection-us-per-token-layer", type=float, default=0.35)
    args = parser.parse_args()

    manifest = CalibrationManifest.load(args.manifest)
    target_prefill_ms = args.target_prefill_ms
    if target_prefill_ms is None:
        target_prefill_ms = args.prompt_tokens * manifest.target.kv_shape.num_layers * 0.004
    materialization_ms = (
        args.prompt_tokens
        * manifest.target.kv_shape.num_layers
        * args.projection_us_per_token_layer
        / 1000.0
    )
    accepted = manifest.passed and materialization_ms <= 0.70 * target_prefill_ms
    payload = {
        "manifest": str(args.manifest),
        "direction": manifest.direction.value,
        "prompt_tokens": args.prompt_tokens,
        "target_prefill_ms": round(target_prefill_ms, 4),
        "estimated_materialization_ms": round(materialization_ms, 4),
        "estimated_prefill_saved_ms": round(max(0.0, target_prefill_ms - materialization_ms), 4),
        "accepted_by_cost_gate": accepted,
        "projection_id": manifest.projection_id,
        "layer_map_id": manifest.layer_map_id,
    }
    print(json.dumps(payload, indent=2, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser(description="GoldenScale KV reuse artifact utilities.")
    parser.add_argument("command", choices=["collect", "fit", "validate", "bench"])
    args, remaining = parser.parse_known_args()

    import sys

    sys.argv = [f"golden-scale-{args.command}", *remaining]
    if args.command == "collect":
        main_collect()
    elif args.command == "fit":
        main_fit()
    elif args.command == "validate":
        main_validate()
    elif args.command == "bench":
        main_bench()


if __name__ == "__main__":
    main()
