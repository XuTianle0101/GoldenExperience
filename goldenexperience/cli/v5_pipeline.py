"""Initialize and inspect the fail-closed selective KV v5 pipeline workspace."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from goldenexperience.benchmarks.publication import PublicationBenchmarkManifest
from goldenexperience.size_variant.cached_kv_manifest import model_spec_from_path
from goldenexperience.size_variant.v5_pipeline import (
    PIPELINE_STAGES,
    V5DirectionConfig,
    V5PipelineConfig,
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


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    workspace = (
        initialize_workspace(args)
        if args.command == "init"
        else V5PipelineWorkspace.open(args.workspace)
    )
    print(json.dumps(status_payload(workspace), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
