"""Initialize and inspect the fail-closed selective KV v5 pipeline workspace."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from goldenexperience.benchmarks.publication import PublicationBenchmarkManifest
from goldenexperience.size_variant.cached_kv_manifest import model_spec_from_path
from goldenexperience.size_variant.v5_pipeline import (
    COLLECTABLE_SPLITS,
    PIPELINE_STAGES,
    PipelineStageRecord,
    V5DirectionConfig,
    V5PipelineConfig,
    V5PipelineError,
    V5PipelineWorkspace,
    source_tree_sha256,
)

DEFAULT_MODELS = {
    "4b": Path("/workspace/volume/softdata/models/Qwen3-4B"),
    "8b": Path("/workspace/volume/softdata/models/Qwen3-8B"),
    "14b": Path("/workspace/volume/softdata/models/Qwen3-14B"),
}
MODEL_IDS = {
    "4b": "Qwen/Qwen3-4B",
    "8b": "Qwen/Qwen3-8B",
    "14b": "Qwen/Qwen3-14B",
}
PARAMETER_COUNTS = {"4b": 4.0, "8b": 8.0, "14b": 14.0}
DIRECTION_SIZES = {
    "qwen3_4b_to_8b": ("4b", "8b"),
    "qwen3_8b_to_4b": ("8b", "4b"),
    "qwen3_8b_to_14b": ("8b", "14b"),
    "qwen3_14b_to_8b": ("14b", "8b"),
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    initialize = commands.add_parser("init", help="bind models and a frozen benchmark")
    initialize.add_argument("--workspace", type=Path, required=True)
    initialize.add_argument("--benchmark-manifest", type=Path, required=True)
    initialize.add_argument("--model-4b", type=Path, default=DEFAULT_MODELS["4b"])
    initialize.add_argument("--model-8b", type=Path, default=DEFAULT_MODELS["8b"])
    initialize.add_argument("--model-14b", type=Path, default=DEFAULT_MODELS["14b"])
    initialize.add_argument("--revision-4b", default="local-snapshot")
    initialize.add_argument("--revision-8b", default="local-snapshot")
    initialize.add_argument("--revision-14b", default="local-snapshot")
    initialize.add_argument("--identity-cache", type=Path)
    initialize.add_argument("--refresh-identity", action="store_true")
    initialize.add_argument("--repository-root", type=Path, default=Path.cwd())
    status = commands.add_parser("status", help="validate and display workspace state")
    status.add_argument("--workspace", type=Path, required=True)
    collect = commands.add_parser("collect", help="collect one complete non-sealed split")
    collect.add_argument("--workspace", type=Path, required=True)
    collect.add_argument("--direction", choices=tuple(DIRECTION_SIZES), required=True)
    collect.add_argument("--split", choices=tuple(sorted(COLLECTABLE_SPLITS)), required=True)
    collect.add_argument("--samples", type=Path, required=True)
    collect.add_argument("--source-device", default="cuda:0")
    collect.add_argument("--target-device", default="cuda:1")
    collect.add_argument("--identity-cache", type=Path)
    collect.add_argument("--repository-root", type=Path, default=Path.cwd())
    collect.add_argument("--resume", action="store_true")
    collect.add_argument("--progress-every", type=int, default=16)
    fit_transport = commands.add_parser(
        "fit-transport",
        help="fit the screening matrix or one frozen non-screening transport",
    )
    fit_transport.add_argument("--workspace", type=Path, required=True)
    fit_transport.add_argument("--direction", choices=tuple(DIRECTION_SIZES), required=True)
    fit_transport.add_argument("--device", default="cuda:1")
    fit_transport.add_argument("--repository-root", type=Path, default=Path.cwd())
    fit_transport.add_argument("--resume", action="store_true")
    fit_transport.add_argument("--checkpoint-every-steps", type=int, default=256)
    fit_transport.add_argument("--progress-every", type=int, default=16)
    method_dev = commands.add_parser(
        "evaluate-method-dev",
        help="evaluate all 4B-to-8B candidates and freeze the selected rank",
    )
    method_dev.add_argument("--workspace", type=Path, required=True)
    method_dev.add_argument("--direction", choices=("qwen3_4b_to_8b",), required=True)
    method_dev.add_argument("--samples", type=Path, required=True)
    method_dev.add_argument("--source-device", default="cuda:0")
    method_dev.add_argument("--target-device", default="cuda:1")
    method_dev.add_argument("--identity-cache", type=Path)
    method_dev.add_argument("--repository-root", type=Path, default=Path.cwd())
    method_dev.add_argument("--resume", action="store_true")
    method_dev.add_argument("--progress-every", type=int, default=1)
    return parser


def initialize_workspace(args: argparse.Namespace) -> V5PipelineWorkspace:
    manifest = PublicationBenchmarkManifest.load(args.benchmark_manifest)
    paths = {
        "4b": args.model_4b.resolve(),
        "8b": args.model_8b.resolve(),
        "14b": args.model_14b.resolve(),
    }
    revisions = {
        "4b": args.revision_4b,
        "8b": args.revision_8b,
        "14b": args.revision_14b,
    }
    identity_cache = args.identity_cache
    if identity_cache is None:
        identity_cache = args.workspace.resolve() / ".pipeline" / "model_identity_cache.json"
    specs = {
        size: model_spec_from_path(
            path,
            model_id=MODEL_IDS[size],
            parameter_count_b=PARAMETER_COUNTS[size],
            revision=revisions[size],
            identity_cache_path=identity_cache,
            refresh_identity=args.refresh_identity,
        )
        for size, path in paths.items()
    }
    directions = tuple(
        V5DirectionConfig(
            direction=direction,
            source_model_path=str(paths[source_size]),
            target_model_path=str(paths[target_size]),
            source=specs[source_size],
            target=specs[target_size],
        )
        for direction, (source_size, target_size) in DIRECTION_SIZES.items()
    )
    config = V5PipelineConfig.from_benchmark(
        manifest,
        manifest_path=args.benchmark_manifest,
        code_sha256=source_tree_sha256(args.repository_root),
        directions=directions,
    )
    return V5PipelineWorkspace.create(args.workspace, config)


def status_payload(workspace: V5PipelineWorkspace) -> dict:
    state = workspace.state()
    stages: dict[str, dict[str, str]] = {
        direction.direction: {} for direction in workspace.config.directions
    }
    for record in state.stages.values():
        stages[record.direction][record.stage] = record.status
    return {
        "pipeline_id": workspace.config.pipeline_id,
        "config_sha256": workspace.config.content_sha256(),
        "benchmark_manifest_sha256": workspace.config.benchmark_manifest_sha256,
        "code_sha256": workspace.config.code_sha256,
        "semantic_sealed": "locked",
        "known_stages": sorted(PIPELINE_STAGES),
        "stages": stages,
    }


def collect_split(args: argparse.Namespace) -> PipelineStageRecord:
    from goldenexperience.size_variant.v5_collect import (
        RealQwenTraceCollector,
        run_collect_stage,
        stderr_progress,
    )

    workspace = V5PipelineWorkspace.open(args.workspace)
    if source_tree_sha256(args.repository_root) != workspace.config.code_sha256:
        raise V5PipelineError("executable source tree differs from the pipeline code hash")
    direction = workspace.config.direction(args.direction)
    identity_cache = args.identity_cache
    if identity_cache is None:
        identity_cache = workspace.control / "model_identity_cache.json"
    collector = RealQwenTraceCollector(
        source_path=direction.source_model_path,
        target_path=direction.target_model_path,
        source=direction.source,
        target=direction.target,
        source_device=args.source_device,
        target_device=args.target_device,
        identity_cache_path=identity_cache,
    )
    return run_collect_stage(
        workspace=workspace,
        direction=args.direction,
        split=args.split,
        sample_store_path=args.samples,
        collector_parameters=collector.parameters(),
        collector_factory=lambda: collector,
        resume=args.resume,
        progress=stderr_progress(args.progress_every),
    )


def fit_transport(args: argparse.Namespace) -> PipelineStageRecord:
    from goldenexperience.size_variant.v5_directional_fit import (
        run_frozen_direction_fit_stage,
        stderr_directional_fit_progress,
    )
    from goldenexperience.size_variant.v5_fit import (
        SCREENING_DIRECTION,
        run_fit_transport_stage,
        stderr_fit_progress,
    )

    workspace = V5PipelineWorkspace.open(args.workspace)
    if source_tree_sha256(args.repository_root) != workspace.config.code_sha256:
        raise V5PipelineError("executable source tree differs from the pipeline code hash")
    common = {
        "workspace": workspace,
        "direction": args.direction,
        "device": args.device,
        "resume": args.resume,
        "checkpoint_every_steps": args.checkpoint_every_steps,
    }
    if args.direction == SCREENING_DIRECTION:
        return run_fit_transport_stage(
            **common,
            progress=stderr_fit_progress(args.progress_every),
        )
    return run_frozen_direction_fit_stage(
        **common,
        progress=stderr_directional_fit_progress(args.progress_every),
    )


def evaluate_method_dev(args: argparse.Namespace) -> PipelineStageRecord:
    from goldenexperience.size_variant.v5_collect import load_bound_benchmark
    from goldenexperience.size_variant.v5_fit import load_completed_transport_fit
    from goldenexperience.size_variant.v5_method_dev import (
        run_method_dev_stage,
        stderr_method_dev_progress,
    )
    from goldenexperience.size_variant.v5_real_method_dev import RealQwenMethodDevEvaluator

    workspace = V5PipelineWorkspace.open(args.workspace)
    if source_tree_sha256(args.repository_root) != workspace.config.code_sha256:
        raise V5PipelineError("executable source tree differs from the pipeline code hash")
    direction = workspace.config.direction(args.direction)
    fit, _ = load_completed_transport_fit(
        workspace,
        args.direction,
        load_bound_benchmark(workspace),
    )
    identity_cache = args.identity_cache
    if identity_cache is None:
        identity_cache = workspace.control / "model_identity_cache.json"
    evaluator = RealQwenMethodDevEvaluator(
        workspace=workspace,
        fit=fit,
        source_path=direction.source_model_path,
        target_path=direction.target_model_path,
        source_device=args.source_device,
        target_device=args.target_device,
        identity_cache_path=identity_cache,
    )
    return run_method_dev_stage(
        workspace=workspace,
        direction=args.direction,
        sample_store_path=args.samples,
        evaluator_parameters=evaluator.parameters(),
        evaluator_factory=lambda: evaluator,
        resume=args.resume,
        progress=stderr_method_dev_progress(args.progress_every),
    )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    stage_commands = {
        "collect": collect_split,
        "fit-transport": fit_transport,
        "evaluate-method-dev": evaluate_method_dev,
    }
    if args.command in stage_commands:
        record = stage_commands[args.command](args)
        payload = {
            "direction": record.direction,
            "stage": record.stage,
            "status": record.status,
            "input_sha256": record.input_sha256,
            "receipt_sha256": record.receipt_sha256,
            "outputs": {
                name: asdict(artifact) for name, artifact in (record.outputs or {}).items()
            },
        }
    else:
        workspace = (
            initialize_workspace(args)
            if args.command == "init"
            else V5PipelineWorkspace.open(args.workspace)
        )
        payload = status_payload(workspace)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
