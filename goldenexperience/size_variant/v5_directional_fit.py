"""Frozen-structure transport fitting for the three non-screening v5 directions."""

from __future__ import annotations

import hashlib
import json
import os
import sys
import uuid
from collections.abc import Callable, Mapping
from contextlib import suppress
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from goldenexperience.benchmarks.publication import PublicationBenchmarkManifest
from goldenexperience.size_variant.cached_kv_manifest import CachedKVModelSpec
from goldenexperience.size_variant.v5_collect import (
    TraceObjectRef,
    V5TraceManifest,
    load_bound_benchmark,
    load_completed_trace_manifest,
)
from goldenexperience.size_variant.v5_fit import (
    DEPLOYMENT_SEED,
    SCREENING_DIRECTION,
    CandidateTrainingMetrics,
    SynchronousTransportTrainer,
    TransportCandidateArtifact,
    TransportTrainingParameters,
    load_fitted_transport,
    verify_transport_candidate_object,
)
from goldenexperience.size_variant.v5_method_dev import (
    FrozenTransportStructure,
    load_frozen_transport_structure,
)
from goldenexperience.size_variant.v5_pipeline import (
    PipelineStageRecord,
    V5PipelineError,
    V5PipelineWorkspace,
)

V5_DIRECTIONAL_FIT_SCHEMA = "goldenexperience.v5_directional_transport_fit.v1"


@dataclass(frozen=True)
class V5DirectionalTransportFitManifest:
    pipeline_id: str
    direction: str
    code_sha256: str
    transport_train_split_sha256: str
    trace_manifest_sha256: str
    frozen_structure_sha256: str
    normalizer_sha256: str
    source: CachedKVModelSpec
    target: CachedKVModelSpec
    training: TransportTrainingParameters
    candidates: tuple[TransportCandidateArtifact, ...]
    schema_version: str = V5_DIRECTIONAL_FIT_SCHEMA

    def validate(
        self,
        *,
        workspace: V5PipelineWorkspace,
        trace: V5TraceManifest,
        structure: FrozenTransportStructure,
    ) -> list[str]:
        errors: list[str] = []
        if self.schema_version != V5_DIRECTIONAL_FIT_SCHEMA:
            errors.append("unsupported directional transport fit schema")
        if self.pipeline_id != workspace.config.pipeline_id:
            errors.append("directional transport fit belongs to another pipeline")
        if self.direction == SCREENING_DIRECTION or trace.direction != self.direction:
            errors.append("directional fit must target a non-screening Qwen3 direction")
        try:
            workspace.config.direction(self.direction)
        except V5PipelineError as exc:
            errors.append(str(exc))
        if self.code_sha256 != workspace.config.code_sha256:
            errors.append("directional transport fit code hash mismatch")
        if self.transport_train_split_sha256 != workspace.config.split_sha256["transport_train"]:
            errors.append("directional transport fit split hash mismatch")
        if self.trace_manifest_sha256 != trace.content_sha256():
            errors.append("directional transport fit trace hash mismatch")
        if self.frozen_structure_sha256 != structure.content_sha256():
            errors.append("directional transport fit structure receipt hash mismatch")
        if not _is_sha256(self.normalizer_sha256):
            errors.append("directional transport fit normalizer hash is invalid")
        if self.source != trace.source or self.target != trace.target:
            errors.append("directional transport fit model identities differ from traces")
        expected_training = frozen_direction_training_parameters(structure)
        if self.training != expected_training:
            errors.append("directional transport training differs from the frozen contract")
        training_errors = self.training.validate(require_registered=False)
        errors.extend(training_errors)
        if len(self.candidates) != 1:
            errors.append("directional transport fit must contain one deployment candidate")
        else:
            candidate = self.candidates[0]
            if (
                candidate.rank != structure.selected_rank
                or candidate.seed != DEPLOYMENT_SEED
                or not candidate.deployment_seed
            ):
                errors.append("directional transport candidate differs from frozen structure")
            expected_samples = len(trace.records) * self.training.epochs
            expected_steps = (
                (len(trace.records) + self.training.gradient_accumulation - 1)
                // self.training.gradient_accumulation
                * self.training.epochs
            )
            errors.extend(candidate.validate(expected_samples, expected_steps))
        return errors

    def content_sha256(self) -> str:
        return _sha256_bytes(_canonical_json_bytes(self.to_dict()))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "pipeline_id": self.pipeline_id,
            "direction": self.direction,
            "code_sha256": self.code_sha256,
            "transport_train_split_sha256": self.transport_train_split_sha256,
            "trace_manifest_sha256": self.trace_manifest_sha256,
            "frozen_structure_sha256": self.frozen_structure_sha256,
            "normalizer_sha256": self.normalizer_sha256,
            "source": asdict(self.source),
            "target": asdict(self.target),
            "training": self.training.to_dict(),
            "candidates": [asdict(item) for item in self.candidates],
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> V5DirectionalTransportFitManifest:
        from goldenexperience.size_variant.selective_manifest import TransportLossContract

        training_payload = dict(payload["training"])
        training_payload["ranks"] = tuple(training_payload["ranks"])
        training_payload["seeds"] = tuple(training_payload["seeds"])
        training_payload["loss"] = TransportLossContract(**training_payload["loss"])
        candidates = []
        for item in payload.get("candidates", ()):
            candidate = dict(item)
            candidate["weights"] = TraceObjectRef(**candidate["weights"])
            candidate["metrics"] = CandidateTrainingMetrics(**candidate["metrics"])
            candidates.append(TransportCandidateArtifact(**candidate))
        return cls(
            pipeline_id=str(payload["pipeline_id"]),
            direction=str(payload["direction"]),
            code_sha256=str(payload["code_sha256"]),
            transport_train_split_sha256=str(payload["transport_train_split_sha256"]),
            trace_manifest_sha256=str(payload["trace_manifest_sha256"]),
            frozen_structure_sha256=str(payload["frozen_structure_sha256"]),
            normalizer_sha256=str(payload["normalizer_sha256"]),
            source=CachedKVModelSpec(**payload["source"]),
            target=CachedKVModelSpec(**payload["target"]),
            training=TransportTrainingParameters(**training_payload),
            candidates=tuple(candidates),
            schema_version=str(payload.get("schema_version", "")),
        )

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        workspace: V5PipelineWorkspace,
        trace: V5TraceManifest,
        structure: FrozenTransportStructure,
    ) -> V5DirectionalTransportFitManifest:
        try:
            value = cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
            errors = value.validate(workspace=workspace, trace=trace, structure=structure)
        except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise V5PipelineError("directional fit manifest is unreadable or malformed") from exc
        if errors:
            raise V5PipelineError("; ".join(errors))
        for candidate in value.candidates:
            verify_transport_candidate_object(workspace, candidate)
        return value


def frozen_direction_training_parameters(
    structure: FrozenTransportStructure,
) -> TransportTrainingParameters:
    """Return the only training contract allowed after method-dev structure selection."""

    return TransportTrainingParameters(
        ranks=(structure.selected_rank,),
        seeds=(DEPLOYMENT_SEED,),
        deployment_seed=DEPLOYMENT_SEED,
        source_window=structure.source_window,
        epochs=3,
        learning_rate=3e-4,
        weight_decay=1e-4,
        gradient_accumulation=8,
        max_grad_norm=1.0,
    )


def run_frozen_direction_fit_stage(
    *,
    workspace: V5PipelineWorkspace,
    direction: str,
    device: str = "cuda:1",
    resume: bool = False,
    checkpoint_every_steps: int = 256,
    progress: Callable[[int, int, int, str], None] | None = None,
) -> PipelineStageRecord:
    """Fit the single frozen-rank, seed-17 transport for one other direction."""

    if direction == SCREENING_DIRECTION:
        raise V5PipelineError("the screening direction must use registered candidate fitting")
    benchmark = load_bound_benchmark(workspace)
    structure, _, _ = load_frozen_transport_structure(workspace)
    trace = load_completed_trace_manifest(workspace, direction, "transport_train", benchmark)
    training = frozen_direction_training_parameters(structure)
    errors = training.validate(require_registered=False)
    if errors:
        raise V5PipelineError("; ".join(errors))
    stage_parameters = {
        "trace_manifest_sha256": trace.content_sha256(),
        "frozen_structure_sha256": structure.content_sha256(),
        "training": training.to_dict(),
    }
    lease = workspace.begin_stage(
        direction,
        "fit_transport",
        parameters=stage_parameters,
        resume=resume,
    )
    if lease.reused:
        return workspace.state().stages[f"{direction}/fit_transport"]
    work = workspace.control / "work" / direction / "fit_transport"
    try:
        trainer = SynchronousTransportTrainer(
            workspace=workspace,
            trace=trace,
            parameters=training,
            device=device,
            checkpoint_every_steps=checkpoint_every_steps,
            progress=progress,
        )
        fitted, normalizer_sha256 = trainer.fit(
            work,
            stage_input_sha256=lease.input_sha256,
        )
        if len(fitted) != 1:
            raise V5PipelineError("directional transport trainer emitted multiple candidates")
        item = fitted[0]
        artifact = workspace.publish_file(
            item.path,
            logical_name=f"transport_r{item.rank}_s{item.seed}",
        )
        candidate = TransportCandidateArtifact(
            candidate_id=item.candidate_id,
            rank=item.rank,
            seed=item.seed,
            deployment_seed=True,
            weights=TraceObjectRef.from_artifact(artifact),
            parameter_count=item.parameter_count,
            metrics=item.metrics,
        )
        manifest = V5DirectionalTransportFitManifest(
            pipeline_id=workspace.config.pipeline_id,
            direction=direction,
            code_sha256=workspace.config.code_sha256,
            transport_train_split_sha256=workspace.config.split_sha256["transport_train"],
            trace_manifest_sha256=trace.content_sha256(),
            frozen_structure_sha256=structure.content_sha256(),
            normalizer_sha256=normalizer_sha256,
            source=trace.source,
            target=trace.target,
            training=training,
            candidates=(candidate,),
        )
        manifest_errors = manifest.validate(
            workspace=workspace,
            trace=trace,
            structure=structure,
        )
        if manifest_errors:
            raise V5PipelineError("; ".join(manifest_errors))
        manifest_path = work / "directional_transport_fit_manifest.json"
        _write_json_replace(manifest_path, manifest.to_dict())
        return workspace.complete_stage(
            lease,
            outputs={"transport_fit_manifest": manifest_path},
            metadata={
                "candidate_count": 1,
                "selected_rank": structure.selected_rank,
                "deployment_seed": DEPLOYMENT_SEED,
                "frozen_structure_sha256": structure.content_sha256(),
                "transport_fit_manifest_sha256": manifest.content_sha256(),
            },
        )
    except Exception as exc:
        with suppress(V5PipelineError):
            workspace.fail_stage(lease, exc)
        raise


def load_completed_directional_fit(
    workspace: V5PipelineWorkspace,
    direction: str,
    benchmark: PublicationBenchmarkManifest,
) -> tuple[V5DirectionalTransportFitManifest, V5TraceManifest, FrozenTransportStructure]:
    if direction == SCREENING_DIRECTION:
        raise V5PipelineError("screening fit uses the candidate-matrix manifest")
    structure, _, _ = load_frozen_transport_structure(workspace)
    trace = load_completed_trace_manifest(workspace, direction, "transport_train", benchmark)
    state = workspace.state()
    stage = state.stages.get(f"{direction}/fit_transport")
    if stage is None or stage.status != "completed" or stage.outputs is None:
        raise V5PipelineError("stage requires completed directional transport fitting")
    artifact = stage.outputs.get("transport_fit_manifest")
    if artifact is None:
        raise V5PipelineError("directional transport fit lacks its manifest")
    path = workspace.artifact_path(artifact, verify_hash=True)
    return (
        V5DirectionalTransportFitManifest.load(
            path,
            workspace=workspace,
            trace=trace,
            structure=structure,
        ),
        trace,
        structure,
    )


def load_direction_deployment_transport(
    workspace: V5PipelineWorkspace,
    manifest: V5DirectionalTransportFitManifest,
    *,
    device: str,
) -> Any:
    if len(manifest.candidates) != 1:
        raise V5PipelineError("directional transport manifest lacks one deployment candidate")
    return load_fitted_transport(
        workspace,
        manifest,
        manifest.candidates[0],
        device=device,
    )[0]


def _write_json_replace(path: Path, value: Mapping[str, Any]) -> None:
    payload = _canonical_json_bytes(dict(value), indent=2)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _canonical_json_bytes(value: Any, *, indent: int | None = None) -> bytes:
    try:
        return (
            json.dumps(
                value,
                indent=indent,
                sort_keys=True,
                separators=(",", ":") if indent is None else None,
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise V5PipelineError("directional fit metadata is not finite canonical JSON") from exc


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _is_sha256(value: str | None) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True


def stderr_directional_fit_progress(every: int = 16) -> Callable[[int, int, int, str], None]:
    if every <= 0:
        raise V5PipelineError("directional fit progress interval must be positive")

    def report(index: int, total: int, epoch: int, sample_id: str) -> None:
        if index == total or index % every == 0:
            print(
                f"directional transport fit {index}/{total} epoch={epoch + 1} sample={sample_id}",
                file=sys.stderr,
                flush=True,
            )

    return report
