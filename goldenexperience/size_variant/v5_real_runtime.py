"""Real Qwen + vLLM + LMCache backend for the isolated v5 runtime audit."""

from __future__ import annotations

import gc
import hashlib
import importlib.metadata
import json
import math
import os
import secrets
import subprocess
import sys
from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, cast

from goldenexperience.benchmarks.publication import GroupedPrefixRecord
from goldenexperience.runtime.cross_model_reuse import token_ids_sha256
from goldenexperience.runtime.lmcache_mp_server import (
    LMCacheMPServerConfig,
    LMCacheMPServerProcess,
)
from goldenexperience.runtime.lmcache_retrieve_transform import (
    LMCacheMPSourceChunkWriter,
    LMCacheStoredSourcePrefix,
)
from goldenexperience.runtime.runtime_audit_telemetry import (
    RUNTIME_AUDIT_TELEMETRY_SCHEMA,
    RuntimeAuditTelemetryCollector,
)
from goldenexperience.runtime.vllm_retrieve_transform_connector import (
    V5_VLLM_CONNECTOR_PARAMS_KEY,
    build_vllm_retrieve_transform_params,
    runtime_source_model_name,
)
from goldenexperience.size_variant.cached_kv_manifest import sha256_file, verify_model_path
from goldenexperience.size_variant.head_aware_transport import (
    HeadAwareKVTransport,
    dynamic_cache_to_head_object,
)
from goldenexperience.size_variant.risk_gate import (
    CalibratedRiskGate,
    SourceKVSidecar,
    build_transport_source_sidecar,
)
from goldenexperience.size_variant.selective_manifest import (
    ArtifactState,
    SelectiveKVBridgeManifest,
)
from goldenexperience.size_variant.v5_calibration import load_completed_risk_calibration
from goldenexperience.size_variant.v5_collect import (
    RawBenchmarkSample,
    TraceRecord,
    load_bound_benchmark,
    load_completed_trace_manifest,
    load_raw_sample_store,
)
from goldenexperience.size_variant.v5_directional_fit import V5DirectionalTransportFitManifest
from goldenexperience.size_variant.v5_fit import (
    TransportCandidateArtifact,
    V5TransportFitManifest,
    load_fitted_transport,
)
from goldenexperience.size_variant.v5_pipeline import V5PipelineError, V5PipelineWorkspace
from goldenexperience.size_variant.v5_real_method_dev import greedy_decode, teacher_nll
from goldenexperience.size_variant.v5_risk import (
    RISK_LABEL_GENERATION_TOKENS,
    RiskHistory,
    load_risk_predictor,
)
from goldenexperience.size_variant.v5_runtime import (
    RuntimeExecutionMeasurement,
    RuntimeFailureAudit,
    RuntimeRiskObservation,
)
from goldenexperience.size_variant.v5_semantic import load_completed_semantic

V5_REAL_RUNTIME_EVALUATOR_ID = "qwen3_vllm_lmcache_runtime_audit_v1"
V5_REAL_RUNTIME_CONNECTOR_MODULE = "goldenexperience.runtime.vllm_retrieve_transform_connector"
V5_REAL_RUNTIME_CONNECTOR_CLASS = "V5RetrieveTransformConnector"
V5_REAL_RUNTIME_PAIR_ORDER = "native_then_guarded"
V5_REAL_RUNTIME_MAX_MODEL_LEN = 16384
V5_REAL_RUNTIME_KV_CACHE_BYTES = 4 << 30
V5_REAL_RUNTIME_FAILURE_TOKENS = 16


@dataclass(frozen=True)
class _PrefixAsset:
    prefix_group_id: str
    prefix_hash: str
    token_ids: tuple[int, ...]
    stored: LMCacheStoredSourcePrefix
    base_sidecar: SourceKVSidecar
    native_target_kv: Any
    transformed_target_kv: Any
    representative_suffix: tuple[int, ...]


@dataclass(frozen=True)
class _RequestContext:
    sample_id: str
    prompt_token_ids: tuple[int, ...]
    sidecar_payload: bytes
    asset: _PrefixAsset
    native_shadow_tokens: tuple[int, ...]
    bridge_shadow_tokens: tuple[int, ...]


@dataclass(frozen=True)
class _VLLMResult:
    token_ids: tuple[int, ...]
    ttft_ms: float
    prefill_ms: float
    num_cached_tokens: int


class RealQwenRuntimeAuditEvaluator:
    """Execute the registered runtime split through the production connector path."""

    def __init__(
        self,
        *,
        workspace: V5PipelineWorkspace,
        direction: str,
        sample_store_path: str | Path,
        source_path: str | Path,
        target_path: str | Path,
        source_device: str,
        target_device: str,
        identity_cache_path: str | Path | None,
        lmcache_config: LMCacheMPServerConfig | None = None,
        attention_implementation: str = "sdpa",
        seed: int = 17,
        max_model_len: int = V5_REAL_RUNTIME_MAX_MODEL_LEN,
        kv_cache_memory_bytes: int = V5_REAL_RUNTIME_KV_CACHE_BYTES,
        retrieve_timeout_s: float = 120.0,
        mq_timeout_s: float = 30.0,
        telemetry_timeout_s: float = 120.0,
    ) -> None:
        self.workspace = workspace
        self.direction = direction
        self.sample_store_path = Path(sample_store_path).resolve()
        self.source_path = Path(source_path).resolve()
        self.target_path = Path(target_path).resolve()
        self.source_device = source_device
        self.target_device = target_device
        self.identity_cache_path = identity_cache_path
        self.lmcache_config = lmcache_config or LMCacheMPServerConfig()
        self.attention_implementation = attention_implementation
        self.seed = seed
        self.max_model_len = max_model_len
        self.kv_cache_memory_bytes = kv_cache_memory_bytes
        self.retrieve_timeout_s = retrieve_timeout_s
        self.mq_timeout_s = mq_timeout_s
        self.telemetry_timeout_s = telemetry_timeout_s
        self.telemetry_nonce = secrets.token_hex(16)
        self.telemetry_secret_hex = secrets.token_hex(32)

        self.benchmark: Any | None = None
        self.semantic_selective: SelectiveKVBridgeManifest | None = None
        self.semantic_selective_path: Path | None = None
        self.transport_manifest: (
            V5TransportFitManifest | V5DirectionalTransportFitManifest | None
        ) = None
        self.candidate: TransportCandidateArtifact | None = None
        self.gate: CalibratedRiskGate | None = None
        self.samples: tuple[tuple[GroupedPrefixRecord, RawBenchmarkSample], ...] = ()
        self.traces: dict[str, TraceRecord] = {}
        self.server: LMCacheMPServerProcess | None = None
        self.collector: RuntimeAuditTelemetryCollector | None = None
        self.writer: LMCacheMPSourceChunkWriter | None = None
        self.llm: Any | None = None
        self.tokenizer: Any | None = None
        self.source_model: Any | None = None
        self.target_model: Any | None = None
        self.transport: HeadAwareKVTransport | None = None
        self.assets: dict[str, _PrefixAsset] = {}
        self._pending_contexts: dict[str, _RequestContext] = {}
        self._failure_probe_context: _RequestContext | None = None
        self._successful_load_observed = False
        self._block_size: int | None = None
        self._entered = False
        self._validate_configuration()

    def parameters(self) -> dict[str, Any]:
        import torch
        import transformers

        return {
            "evaluator_id": V5_REAL_RUNTIME_EVALUATOR_ID,
            "seed": self.seed,
            "attention_implementation": self.attention_implementation,
            "shadow_generation_tokens": RISK_LABEL_GENERATION_TOKENS,
            "failure_probe_generation_tokens": V5_REAL_RUNTIME_FAILURE_TOKENS,
            "pair_order": V5_REAL_RUNTIME_PAIR_ORDER,
            "connector_module": V5_REAL_RUNTIME_CONNECTOR_MODULE,
            "connector_class": V5_REAL_RUNTIME_CONNECTOR_CLASS,
            "connector_load_failure_policy": "recompute",
            "connector_load_mode": "asynchronous_pre_forward",
            "connector_target_store_path": False,
            "telemetry_schema": RUNTIME_AUDIT_TELEMETRY_SCHEMA,
            "telemetry_transport": "authenticated_loopback_tcp_no_files",
            "source_store_protocol": "PREPARE_STORE+COMMIT_STORE",
            "source_retrieve_protocol": (
                "LOOKUP+QUERY_PREFETCH_STATUS+PREPARE_RETRIEVE+COMMIT_RETRIEVE"
            ),
            "source_prefix_objects": 4,
            "max_model_len": self.max_model_len,
            "max_num_batched_tokens": self.max_model_len,
            "max_num_seqs": 1,
            "kv_cache_memory_bytes": self.kv_cache_memory_bytes,
            "enable_prefix_caching": False,
            "enable_chunked_prefill": False,
            "async_scheduling": False,
            "disable_hybrid_kv_cache_manager": True,
            "enforce_eager": True,
            "tensor_parallel_size": 1,
            "pipeline_parallel_size": 1,
            "vllm_worker_multiproc_method": "spawn",
            "source_device_type": torch.device(self.source_device).type,
            "source_device_name": _device_name(self.source_device),
            "target_device_type": torch.device(self.target_device).type,
            "target_device_name": _device_name(self.target_device),
            "torch_version": torch.__version__,
            "transformers_version": transformers.__version__,
            "vllm_version": importlib.metadata.version("vllm"),
            "lmcache_version": importlib.metadata.version("lmcache"),
            "cuda_version": torch.version.cuda,
            "lmcache_server": self.lmcache_config.publication_parameters(),
        }

    def __enter__(self) -> RealQwenRuntimeAuditEvaluator:
        if self._entered:
            raise V5PipelineError("real runtime evaluator was entered twice")
        try:
            self._load_bindings()
            self._verify_model_paths()
            self.server = LMCacheMPServerProcess(self.lmcache_config).start()
            self.collector = RuntimeAuditTelemetryCollector(
                nonce=self.telemetry_nonce,
                secret_hex=self.telemetry_secret_hex,
            )
            self._start_vllm()
            self._start_source_writer()
            self._load_transformers_models()
            self._prepare_prefix_assets()
            self._entered = True
            return self
        except Exception:
            self._close(suppress_errors=True)
            raise

    def __exit__(self, exc_type: object, *_args: object) -> None:
        self._close(suppress_errors=exc_type is not None)

    def warmup(self, iterations: int) -> None:
        if type(iterations) is not int or iterations <= 0:
            raise V5PipelineError("real runtime warmup count must be positive")
        context = self._accepted_synthetic_context("warmup")
        for index in range(iterations):
            native = self._run_vllm(context.prompt_token_ids, generation_tokens=1)
            if native.num_cached_tokens != 0:
                raise V5PipelineError("native warmup unexpectedly used cached target KV")
            audit_request_id = f"warmup-{self.telemetry_nonce}-{index}"
            params = self._connector_params(
                context,
                audit_request_id=audit_request_id,
                accepted=True,
                decision="accepted",
            )
            guarded = self._run_vllm(
                context.prompt_token_ids,
                generation_tokens=1,
                kv_transfer_params=params,
            )
            self._consume_gate_event(
                audit_request_id,
                accepted=True,
                decision="accepted",
            )
            evidence = self._consume_execution_event(audit_request_id)
            self._validate_success_evidence(evidence, context)
            self._successful_load_observed = True
            if guarded.num_cached_tokens != len(context.asset.token_ids):
                raise V5PipelineError("accepted warmup did not report the external prefix")
        self._require_collector().assert_drained()
        self._require_server().assert_no_backing_files()

    def build_observation(
        self,
        benchmark_record: GroupedPrefixRecord,
        trace_record: TraceRecord,
        sample: RawBenchmarkSample,
        history: RiskHistory,
    ) -> RuntimeRiskObservation:
        if history.validate():
            raise V5PipelineError("runtime observation history is malformed")
        asset = self._bound_asset(benchmark_record, trace_record, sample)
        suffix = self._tokenize_suffix(sample)
        self._validate_request_length(len(asset.token_ids), len(suffix))
        sidecar_payload, runtime_sidecar = self._sidecar_for_history(asset, history)
        native_tokens, _native_text, native_nll = greedy_decode(
            self._require_target_model(),
            self._require_tokenizer(),
            asset.native_target_kv,
            suffix,
            device=self.target_device,
            generation_tokens=RISK_LABEL_GENERATION_TOKENS,
        )
        bridge_tokens, _bridge_text, _ = greedy_decode(
            self._require_target_model(),
            self._require_tokenizer(),
            asset.transformed_target_kv,
            suffix,
            device=self.target_device,
            generation_tokens=RISK_LABEL_GENERATION_TOKENS,
        )
        bridge_nll = teacher_nll(
            self._require_target_model(),
            asset.transformed_target_kv,
            suffix,
            native_tokens,
            device=self.target_device,
        )
        greedy_matches = sum(
            native == bridge for native, bridge in zip(native_tokens, bridge_tokens, strict=True)
        )
        agreement = greedy_matches / RISK_LABEL_GENERATION_TOKENS
        drift = _perplexity_drift_pct(
            native_nll,
            bridge_nll,
            RISK_LABEL_GENERATION_TOKENS,
        )
        prompt = asset.token_ids + tuple(int(value) for value in suffix.tolist())
        context = _RequestContext(
            sample_id=sample.sample_id,
            prompt_token_ids=prompt,
            sidecar_payload=sidecar_payload,
            asset=asset,
            native_shadow_tokens=tuple(native_tokens),
            bridge_shadow_tokens=tuple(bridge_tokens),
        )
        if sample.sample_id in self._pending_contexts:
            raise V5PipelineError("runtime observation was built twice before measurement")
        self._pending_contexts[sample.sample_id] = context
        return RuntimeRiskObservation(
            sample_id=sample.sample_id,
            prefix_group_id=benchmark_record.prefix_group_id,
            features=runtime_sidecar.risk_features(),
            shadow_failure=agreement < 0.98 or drift > 2.0,
            greedy_matches=greedy_matches,
            greedy_tokens=RISK_LABEL_GENERATION_TOKENS,
            native_nll=native_nll,
            bridge_nll=bridge_nll,
            teacher_tokens=RISK_LABEL_GENERATION_TOKENS,
            history_samples=runtime_sidecar.history_samples,
            history_failures=runtime_sidecar.history_failures,
            history_greedy_agreement=runtime_sidecar.history_greedy_agreement,
            sidecar_sha256=hashlib.sha256(sidecar_payload).hexdigest(),
            native_tokens_sha256=token_ids_sha256(native_tokens),
            bridge_tokens_sha256=token_ids_sha256(bridge_tokens),
        )

    def measure(
        self,
        benchmark_record: GroupedPrefixRecord,
        trace_record: TraceRecord,
        sample: RawBenchmarkSample,
        observation: RuntimeRiskObservation,
        *,
        accepted: bool,
        decision: str,
    ) -> RuntimeExecutionMeasurement:
        context = self._pending_contexts.pop(sample.sample_id, None)
        if context is None or context.sample_id != observation.sample_id:
            raise V5PipelineError("runtime measurement lacks its shadow context")
        self._bound_asset(benchmark_record, trace_record, sample)
        admission = self._require_gate().evaluate(context.sidecar_payload)
        if admission.accepted != accepted or admission.reason != decision:
            raise V5PipelineError("runtime connector gate differs from the stage decision")
        native = self._run_vllm(context.prompt_token_ids, generation_tokens=1)
        if native.num_cached_tokens != 0:
            raise V5PipelineError("native paired request unexpectedly used cached target KV")
        params = self._connector_params(
            context,
            audit_request_id=sample.sample_id,
            accepted=accepted,
            decision=decision,
        )
        guarded = self._run_vllm(
            context.prompt_token_ids,
            generation_tokens=1,
            kv_transfer_params=params,
        )
        self._consume_gate_event(sample.sample_id, accepted=accepted, decision=decision)
        if accepted:
            evidence = self._consume_execution_event(sample.sample_id)
            materialization_ms = self._validate_success_evidence(evidence, context)
            if guarded.num_cached_tokens != len(context.asset.token_ids):
                raise V5PipelineError("accepted request did not report the external prefix")
            self._successful_load_observed = True
            if self._failure_probe_context is None:
                self._failure_probe_context = context
            execution = RuntimeExecutionMeasurement(
                native_prefill_ms=native.prefill_ms,
                native_ttft_ms=native.ttft_ms,
                observed_ttft_ms=guarded.ttft_ms,
                materialization_ms=materialization_ms,
                retrieve_transform_success=True,
                load_complete_published=True,
                source_read_attempted=True,
                source_chunks_read=len(context.asset.stored.source_ranges),
                tokens_scattered=len(context.asset.token_ids),
                fallback_reason="none",
                target_mooncake_puts=0,
                backing_files_remaining=self._backing_files_remaining(),
            )
        else:
            if guarded.num_cached_tokens != 0:
                raise V5PipelineError("rejected request reported external cached tokens")
            execution = RuntimeExecutionMeasurement(
                native_prefill_ms=native.prefill_ms,
                native_ttft_ms=native.ttft_ms,
                observed_ttft_ms=guarded.ttft_ms,
                materialization_ms=None,
                retrieve_transform_success=False,
                load_complete_published=False,
                source_read_attempted=False,
                source_chunks_read=0,
                tokens_scattered=0,
                fallback_reason=decision,
                target_mooncake_puts=0,
                backing_files_remaining=self._backing_files_remaining(),
            )
        return execution

    def audit_failure_recovery(self) -> RuntimeFailureAudit:
        context = self._failure_probe_context or self._accepted_synthetic_context("failure")
        native = self._run_vllm(
            context.prompt_token_ids,
            generation_tokens=V5_REAL_RUNTIME_FAILURE_TOKENS,
        )
        audit_request_id = f"failure-{self.telemetry_nonce}"
        params = self._connector_params(
            context,
            audit_request_id=audit_request_id,
            accepted=True,
            decision="accepted",
            inject_partial_failure=True,
        )
        recovered = self._run_vllm(
            context.prompt_token_ids,
            generation_tokens=V5_REAL_RUNTIME_FAILURE_TOKENS,
            kv_transfer_params=params,
        )
        self._consume_gate_event(audit_request_id, accepted=True, decision="accepted")
        evidence = self._consume_execution_event(audit_request_id)
        invalidated_blocks = self._validate_failure_evidence(evidence, context)
        if native.num_cached_tokens != 0 or recovered.num_cached_tokens != 0:
            raise V5PipelineError("failure recovery did not recompute the full native prefix")
        if native.token_ids != recovered.token_ids:
            raise V5PipelineError("native recompute did not restore the native continuation")
        if not self._successful_load_observed:
            raise V5PipelineError("runtime audit lacks a complete all-layer load")
        self._require_collector().assert_drained()
        self._require_server().assert_no_backing_files()
        probe_payload = {
            "schema_version": "goldenexperience.v5_runtime_failure_probe.v1",
            "direction": self.direction,
            "semantic_artifact_id": self._require_semantic().artifact_id,
            "prompt_token_ids_sha256": token_ids_sha256(list(context.prompt_token_ids)),
            "native_tokens_sha256": token_ids_sha256(list(native.token_ids)),
            "recovered_tokens_sha256": token_ids_sha256(list(recovered.token_ids)),
            "invalidated_blocks": invalidated_blocks,
            "recomputed_token_count": len(context.asset.token_ids),
            "partial_failure_count": evidence["partial_failure_count"],
        }
        return RuntimeFailureAudit(
            probe_id=_canonical_sha256(probe_payload),
            paged_slot_mapping_verified=True,
            load_complete_after_all_layers=True,
            partial_failure_invalidates_blocks=True,
            native_prefill_overwrites_invalid_blocks=True,
            injected_failure_count=1,
            invalidated_block_count=len(invalidated_blocks),
            recomputed_token_count=len(context.asset.token_ids),
            accepted_target_mooncake_puts=0,
            backing_files_remaining=self._backing_files_remaining(),
        )

    def _validate_configuration(self) -> None:
        import torch

        source = torch.device(self.source_device)
        target = torch.device(self.target_device)
        if (
            source.type != "cuda"
            or target.type != "cuda"
            or source.index is None
            or target.index is None
            or source.index == target.index
        ):
            raise V5PipelineError("real runtime audit requires two explicit, distinct CUDA devices")
        if (
            not torch.cuda.is_available()
            or max(source.index, target.index) >= torch.cuda.device_count()
        ):
            raise V5PipelineError("real runtime audit CUDA device is unavailable")
        for name, value in (
            ("seed", self.seed),
            ("max_model_len", self.max_model_len),
            ("kv_cache_memory_bytes", self.kv_cache_memory_bytes),
        ):
            if type(value) is not int or value <= 0:
                raise V5PipelineError(f"real runtime evaluator {name} must be positive")
        for timeout_name, timeout_value in (
            ("retrieve_timeout_s", self.retrieve_timeout_s),
            ("mq_timeout_s", self.mq_timeout_s),
            ("telemetry_timeout_s", self.telemetry_timeout_s),
        ):
            if not _finite_positive(timeout_value):
                raise V5PipelineError(
                    f"real runtime evaluator {timeout_name} must be finite and positive"
                )
        if self.max_model_len < 8192 + RISK_LABEL_GENERATION_TOKENS:
            raise V5PipelineError("real runtime evaluator max_model_len is below the frozen split")

    def _load_bindings(self) -> None:
        benchmark = load_bound_benchmark(self.workspace)
        semantic, semantic_selective = load_completed_semantic(
            self.workspace,
            self.direction,
        )
        calibration, risk_fit, _, transport_manifest, candidate = load_completed_risk_calibration(
            self.workspace, self.direction
        )
        trace = load_completed_trace_manifest(
            self.workspace,
            self.direction,
            "runtime_audit",
            benchmark,
        )
        samples = load_raw_sample_store(
            self.sample_store_path,
            benchmark,
            split="runtime_audit",
        )
        trace_by_id = {item.sample_id: item for item in trace.records}
        if set(trace_by_id) != {record.sample_id for record, _ in samples}:
            raise V5PipelineError("real runtime evaluator samples differ from collected traces")
        state = self.workspace.state()
        stage = state.stages.get(f"{self.direction}/semantic")
        if stage is None or stage.status != "completed" or stage.outputs is None:
            raise V5PipelineError("real runtime evaluator requires completed semantic evidence")
        artifact = stage.outputs.get("semantic_selective_manifest")
        if artifact is None:
            raise V5PipelineError("semantic stage lacks its approved selective artifact")
        semantic_path = self.workspace.artifact_path(artifact, verify_hash=True)
        if (
            artifact.sha256 != semantic.semantic_selective_manifest_file_sha256
            or sha256_file(semantic_path) != artifact.sha256
            or semantic_path.stat().st_mode & 0o222
        ):
            raise V5PipelineError("semantic selective artifact is not immutable")
        predictor = load_risk_predictor(self.workspace, risk_fit, device="cpu")
        gate = CalibratedRiskGate(
            calibration.risk_gate,
            predictor,
            model_pair_id=self.direction,
            source_model_hash=semantic_selective.source.weights_sha256,
            target_model_hash=semantic_selective.target.weights_sha256,
            tokenizer_hash=semantic_selective.source.tokenizer_sha256,
            transport_weights_hash=semantic_selective.transport.weights_sha256,
        )
        self.benchmark = benchmark
        self.semantic_selective = semantic_selective
        self.semantic_selective_path = semantic_path
        self.transport_manifest = transport_manifest
        self.candidate = candidate
        self.gate = gate
        self.samples = samples
        self.traces = trace_by_id

    def _verify_model_paths(self) -> None:
        semantic = self._require_semantic()
        for label, expected, path in (
            ("source", semantic.source, self.source_path),
            ("target", semantic.target, self.target_path),
        ):
            errors = verify_model_path(
                expected,
                path,
                identity_cache_path=self.identity_cache_path,
            )
            if errors:
                raise V5PipelineError(f"{label} model identity mismatch: {'; '.join(errors)}")

    def _start_source_writer(self) -> None:
        semantic = self._require_semantic()
        self.writer = LMCacheMPSourceChunkWriter(
            server_url=self._require_server().server_url,
            source_model_name=runtime_source_model_name(semantic),
            source_world_size=1,
            source_worker_id=0,
            source=semantic.source,
            mq_timeout_s=self.mq_timeout_s,
        )
        if self.writer.chunk_size != self.lmcache_config.chunk_size:
            raise V5PipelineError("source writer observed another LMCache chunk size")

    def _start_vllm(self) -> None:
        import torch

        if torch.cuda.is_initialized():
            raise V5PipelineError("vLLM must start before parent-process CUDA initialization")
        process_method = os.environ.get("VLLM_WORKER_MULTIPROC_METHOD")
        if process_method not in {None, "spawn"}:
            raise V5PipelineError("vLLM worker multiprocessing method must be spawn")
        os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
        from vllm import LLM

        semantic = self._require_semantic()
        collector = self._require_collector()
        semantic_path = self._require_semantic_path()
        target_index = _cuda_index(self.target_device)
        extra_config = {
            "golden.semantic_manifest_path": str(semantic_path),
            "golden.semantic_manifest_sha256": sha256_file(semantic_path),
            "golden.semantic_artifact_id": semantic.artifact_id,
            "golden.workspace_root": str(self.workspace.root.resolve()),
            "golden.direction": self.direction,
            "golden.lmcache_server_url": self._require_server().server_url,
            "golden.telemetry_host": collector.host,
            "golden.telemetry_port": collector.port,
            "golden.telemetry_nonce": self.telemetry_nonce,
            "golden.telemetry_secret_hex": self.telemetry_secret_hex,
            "golden.mq_timeout_s": self.mq_timeout_s,
        }
        kv_transfer_config = {
            "kv_connector": V5_REAL_RUNTIME_CONNECTOR_CLASS,
            "kv_connector_module_path": V5_REAL_RUNTIME_CONNECTOR_MODULE,
            "kv_role": "kv_both",
            "engine_id": f"golden-v5-{self.direction}",
            "kv_load_failure_policy": "recompute",
            "kv_connector_extra_config": extra_config,
        }
        self.llm = LLM(
            model=str(self.target_path),
            tokenizer=str(self.target_path),
            skip_tokenizer_init=True,
            trust_remote_code=False,
            tensor_parallel_size=1,
            pipeline_parallel_size=1,
            dtype=cast(Any, semantic.target.dtype),
            seed=self.seed,
            device_ids=[target_index],
            kv_cache_memory_bytes=self.kv_cache_memory_bytes,
            max_model_len=self.max_model_len,
            max_num_batched_tokens=self.max_model_len,
            max_num_seqs=1,
            enable_chunked_prefill=False,
            enable_prefix_caching=False,
            async_scheduling=False,
            disable_hybrid_kv_cache_manager=True,
            enforce_eager=True,
            disable_log_stats=False,
            generation_config="vllm",
            kv_transfer_config=kv_transfer_config,
        )
        config = self.llm.llm_engine.vllm_config
        if (
            config.scheduler_config.enable_chunked_prefill
            or config.scheduler_config.async_scheduling is not False
            or config.cache_config.enable_prefix_caching
            or not config.scheduler_config.disable_hybrid_kv_cache_manager
            or config.kv_transfer_config is None
            or config.kv_transfer_config.kv_load_failure_policy != "recompute"
        ):
            raise V5PipelineError("vLLM runtime audit configuration changed")
        self._block_size = int(config.cache_config.block_size)

    def _load_transformers_models(self) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        semantic = self._require_semantic()
        torch.manual_seed(self.seed)
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.target_path,
            local_files_only=True,
        )
        self.source_model = AutoModelForCausalLM.from_pretrained(
            self.source_path,
            local_files_only=True,
            dtype=_torch_dtype(semantic.source.dtype),
            attn_implementation=self.attention_implementation,
            device_map={"": self.source_device},
        ).eval()
        self.target_model = AutoModelForCausalLM.from_pretrained(
            self.target_path,
            local_files_only=True,
            dtype=_torch_dtype(semantic.target.dtype),
            attn_implementation=self.attention_implementation,
            device_map={"": self.target_device},
        ).eval()
        transport_manifest = self.transport_manifest
        candidate = self.candidate
        if transport_manifest is None or candidate is None:
            raise V5PipelineError("runtime evaluator lacks fitted transport artifacts")
        self.transport = load_fitted_transport(
            self.workspace,
            transport_manifest,
            candidate,
            device=self.target_device,
        )[0]

    def _prepare_prefix_assets(self) -> None:
        import torch

        representatives: dict[str, tuple[GroupedPrefixRecord, RawBenchmarkSample, TraceRecord]] = {}
        for record, sample in self.samples:
            trace = self.traces[record.sample_id]
            current = representatives.get(record.prefix_group_id)
            if current is None:
                representatives[record.prefix_group_id] = (record, sample, trace)
            elif (
                current[0].prefix_sha256 != record.prefix_sha256
                or current[2].token_count != trace.token_count
                or current[1].prefix_text != sample.prefix_text
            ):
                raise V5PipelineError("runtime prefix group is not reusable")
        if len(representatives) != 4 or {
            item[2].token_count for item in representatives.values()
        } != {128, 512, 2048, 8192}:
            raise V5PipelineError("runtime audit requires four registered prefix buckets")
        writer = self._require_writer()
        transport = self._require_transport()
        for group_id, (record, sample, trace) in sorted(
            representatives.items(),
            key=lambda item: item[1][2].token_count,
        ):
            prefix = self._tokenize_prefix(sample, trace)
            with torch.inference_mode():
                source_output = self._require_source_model()(
                    input_ids=prefix.unsqueeze(0).to(self.source_device),
                    use_cache=True,
                    logits_to_keep=1,
                )
            source_kv = dynamic_cache_to_head_object(source_output.past_key_values)
            del source_output
            stored = writer.store_prefix(
                request_id=f"v5-{self.direction}-{group_id.removeprefix('pg-')[:24]}",
                token_ids=prefix.tolist(),
                source_kv=source_kv,
                cache_salt=f"audit-{self.direction}",
            )
            source_on_target = source_kv.to(self.target_device)
            base_sidecar = build_transport_source_sidecar(
                source_on_target,
                transport,
                model_pair_id=self.direction,
                prefix_hash=record.prefix_sha256,
                history_samples=0,
                history_failures=0,
                history_greedy_agreement=1.0,
            )
            positions = torch.arange(trace.token_count, device=self.target_device)
            transformed = transport.transform(source_on_target, position_ids=positions).detach()
            with torch.inference_mode():
                target_output = self._require_target_model()(
                    input_ids=prefix.unsqueeze(0).to(self.target_device),
                    use_cache=True,
                    logits_to_keep=1,
                )
            native = dynamic_cache_to_head_object(target_output.past_key_values).detach()
            del target_output, source_kv, source_on_target
            suffix = tuple(int(value) for value in self._tokenize_suffix(sample).tolist())
            self.assets[group_id] = _PrefixAsset(
                prefix_group_id=group_id,
                prefix_hash=record.prefix_sha256,
                token_ids=tuple(prefix.tolist()),
                stored=stored,
                base_sidecar=base_sidecar,
                native_target_kv=native,
                transformed_target_kv=transformed,
                representative_suffix=suffix,
            )
        expected_puts = sum(len(asset.token_ids) for asset in self.assets.values()) // (
            self.lmcache_config.chunk_size
        )
        if writer.source_put_count != expected_puts:
            raise V5PipelineError("LMCache source prefix put count is inconsistent")
        for record, sample in self.samples:
            self._bound_asset(record, self.traces[record.sample_id], sample)
        self._require_server().assert_no_backing_files()

    def _run_vllm(
        self,
        prompt_token_ids: tuple[int, ...],
        *,
        generation_tokens: int,
        kv_transfer_params: Mapping[str, Any] | None = None,
    ) -> _VLLMResult:
        from vllm import SamplingParams

        if self.llm is None:
            raise V5PipelineError("vLLM runtime engine is not loaded")
        if not prompt_token_ids or type(generation_tokens) is not int or generation_tokens <= 0:
            raise V5PipelineError("vLLM runtime request is malformed")
        extra_args = (
            {"kv_transfer_params": dict(kv_transfer_params)}
            if kv_transfer_params is not None
            else None
        )
        sampling = SamplingParams(
            temperature=0.0,
            max_tokens=generation_tokens,
            min_tokens=generation_tokens,
            ignore_eos=True,
            seed=self.seed,
            detokenize=False,
            extra_args=extra_args,
        )
        outputs = self.llm.generate(
            [{"prompt_token_ids": list(prompt_token_ids)}],
            sampling,
            use_tqdm=False,
        )
        if len(outputs) != 1:
            raise V5PipelineError("vLLM returned another request count")
        output = outputs[0]
        if not output.finished or len(output.outputs) != 1 or output.metrics is None:
            raise V5PipelineError("vLLM runtime request did not finish with metrics")
        tokens = tuple(int(value) for value in output.outputs[0].token_ids)
        metrics = output.metrics
        ttft_ms = float(metrics.first_token_latency) * 1000
        prefill_ms = float(metrics.first_token_ts - metrics.scheduled_ts) * 1000
        cached = output.num_cached_tokens
        if (
            len(tokens) != generation_tokens
            or not _finite_positive(ttft_ms)
            or not _finite_positive(prefill_ms)
            or type(cached) is not int
            or cached < 0
        ):
            raise V5PipelineError("vLLM runtime output metrics are invalid")
        return _VLLMResult(tokens, ttft_ms, prefill_ms, cached)

    def _connector_params(
        self,
        context: _RequestContext,
        *,
        audit_request_id: str,
        accepted: bool,
        decision: str,
        inject_partial_failure: bool = False,
    ) -> dict[str, Any]:
        params = build_vllm_retrieve_transform_params(
            manifest=self._require_semantic(),
            stored=context.asset.stored,
            sidecar_payload=context.sidecar_payload,
            audit_request_id=audit_request_id,
            accepted=accepted,
            decision=decision,
            inject_partial_failure=inject_partial_failure,
            retrieve_timeout_s=self.retrieve_timeout_s,
        )
        if set(params) != {V5_VLLM_CONNECTOR_PARAMS_KEY}:
            raise V5PipelineError("runtime connector parameter envelope changed")
        return params

    def _consume_gate_event(
        self,
        audit_request_id: str,
        *,
        accepted: bool,
        decision: str,
    ) -> dict[str, Any]:
        event = self._require_collector().wait_for(
            request_id=audit_request_id,
            kind="gate",
            timeout_s=self.telemetry_timeout_s,
        )
        evidence = event["evidence"]
        expected_fields = {
            "accepted",
            "decision",
            "unsafe_probability",
            "binding_matches",
            "source_read_attempted",
        }
        if (
            set(evidence) != expected_fields
            or evidence["accepted"] is not accepted
            or evidence["decision"] != decision
            or evidence["binding_matches"] is not True
            or evidence["source_read_attempted"] is not False
        ):
            raise V5PipelineError("runtime scheduler telemetry differs from the gate binding")
        probability = evidence["unsafe_probability"]
        if probability is not None and not _finite_probability(probability):
            raise V5PipelineError("runtime scheduler telemetry probability is invalid")
        return evidence

    def _consume_execution_event(self, audit_request_id: str) -> dict[str, Any]:
        event = self._require_collector().wait_for(
            request_id=audit_request_id,
            kind="execution",
            timeout_s=self.telemetry_timeout_s,
        )
        evidence = event["evidence"]
        expected_fields = {
            "success",
            "accepted",
            "fallback_reason",
            "source_read_attempted",
            "source_chunks_read",
            "tokens_scattered",
            "invalidated_blocks",
            "request_blocks",
            "target_mooncake_puts",
            "materialization_ms",
            "load_complete_published",
            "worker_binding_matches",
            "partial_failure_injected",
            "partial_failure_count",
            "registered_layer_count",
        }
        if not isinstance(evidence, dict) or set(evidence) != expected_fields:
            raise V5PipelineError("runtime worker telemetry schema changed")
        return evidence

    def _validate_success_evidence(
        self,
        evidence: Mapping[str, Any],
        context: _RequestContext,
    ) -> float:
        expected_blocks = len(context.asset.token_ids) // self._required_block_size()
        request_blocks = _integer_list(evidence["request_blocks"])
        invalidated = _integer_list(evidence["invalidated_blocks"])
        if (
            evidence["success"] is not True
            or evidence["accepted"] is not True
            or evidence["fallback_reason"] != "none"
            or evidence["source_read_attempted"] is not True
            or evidence["source_chunks_read"] != len(context.asset.stored.source_ranges)
            or evidence["tokens_scattered"] != len(context.asset.token_ids)
            or invalidated
            or len(request_blocks) != expected_blocks
            or len(set(request_blocks)) != expected_blocks
            or evidence["target_mooncake_puts"] != 0
            or evidence["load_complete_published"] is not True
            or evidence["worker_binding_matches"] is not True
            or evidence["partial_failure_injected"] is not False
            or evidence["partial_failure_count"] != 0
            or evidence["registered_layer_count"] != self._require_semantic().target.num_layers
            or not _finite_positive(evidence["materialization_ms"])
        ):
            raise V5PipelineError("accepted runtime worker telemetry is inconsistent")
        return float(evidence["materialization_ms"])

    def _validate_failure_evidence(
        self,
        evidence: Mapping[str, Any],
        context: _RequestContext,
    ) -> tuple[int, ...]:
        request_blocks = _integer_list(evidence["request_blocks"])
        invalidated = _integer_list(evidence["invalidated_blocks"])
        expected_blocks = len(context.asset.token_ids) // self._required_block_size()
        if (
            evidence["success"] is not False
            or evidence["accepted"] is not True
            or evidence["fallback_reason"] != "direct_injection_failed"
            or evidence["source_read_attempted"] is not True
            or evidence["source_chunks_read"] != len(context.asset.stored.source_ranges)
            or evidence["tokens_scattered"] != 0
            or len(request_blocks) != expected_blocks
            or len(set(request_blocks)) != expected_blocks
            or tuple(sorted(invalidated)) != tuple(sorted(request_blocks))
            or evidence["target_mooncake_puts"] != 0
            or evidence["load_complete_published"] is not False
            or evidence["worker_binding_matches"] is not True
            or evidence["partial_failure_injected"] is not True
            or evidence["partial_failure_count"] != 1
            or evidence["registered_layer_count"] != self._require_semantic().target.num_layers
            or not _finite_positive(evidence["materialization_ms"])
        ):
            raise V5PipelineError("partial-failure runtime telemetry is inconsistent")
        return tuple(sorted(invalidated))

    def _accepted_synthetic_context(self, label: str) -> _RequestContext:
        history = RiskHistory(samples=1, failures=0, greedy_agreement_sum=1.0)
        for asset in sorted(self.assets.values(), key=lambda item: len(item.token_ids)):
            sidecar_payload, _ = self._sidecar_for_history(asset, history)
            admission = self._require_gate().evaluate(sidecar_payload)
            if admission.accepted:
                prompt = asset.token_ids + asset.representative_suffix
                self._validate_request_length(
                    len(asset.token_ids),
                    len(asset.representative_suffix),
                )
                return _RequestContext(
                    sample_id=f"{label}-{asset.prefix_group_id}",
                    prompt_token_ids=prompt,
                    sidecar_payload=sidecar_payload,
                    asset=asset,
                    native_shadow_tokens=(),
                    bridge_shadow_tokens=(),
                )
        raise V5PipelineError("runtime evaluator could not construct an accepted probe")

    def _sidecar_for_history(
        self,
        asset: _PrefixAsset,
        history: RiskHistory,
    ) -> tuple[bytes, SourceKVSidecar]:
        sidecar = replace(
            asset.base_sidecar,
            history_samples=history.samples,
            history_failures=history.failures,
            history_greedy_agreement=history.greedy_agreement,
        )
        payload = sidecar.to_bytes()
        runtime_sidecar = SourceKVSidecar.from_bytes(payload)
        if runtime_sidecar.prefix_hash != asset.prefix_hash or runtime_sidecar.prefix_length != len(
            asset.token_ids
        ):
            raise V5PipelineError("runtime sidecar lost its source prefix binding")
        return payload, runtime_sidecar

    def _bound_asset(
        self,
        record: GroupedPrefixRecord,
        trace: TraceRecord,
        sample: RawBenchmarkSample,
    ) -> _PrefixAsset:
        asset = self.assets.get(record.prefix_group_id)
        if (
            asset is None
            or record.sample_id != trace.sample_id
            or record.sample_id != sample.sample_id
            or record.prefix_sha256 != asset.prefix_hash
            or trace.token_count != len(asset.token_ids)
            or trace.token_ids_sha256 != token_ids_sha256(list(asset.token_ids))
        ):
            raise V5PipelineError("runtime request differs from its cached prefix asset")
        return asset

    def _tokenize_prefix(self, sample: RawBenchmarkSample, trace: TraceRecord) -> Any:
        prefix = self._require_tokenizer()(
            sample.prefix_text,
            add_special_tokens=False,
            return_tensors="pt",
        ).input_ids[0]
        if int(prefix.numel()) < trace.token_count:
            raise V5PipelineError("runtime sample has fewer prefix tokens than registered")
        prefix = prefix[: trace.token_count].long()
        if token_ids_sha256(prefix.tolist()) != trace.token_ids_sha256:
            raise V5PipelineError("runtime prefix tokens differ from collected traces")
        return prefix

    def _tokenize_suffix(self, sample: RawBenchmarkSample) -> Any:
        suffix = (
            self._require_tokenizer()(
                sample.suffix_query,
                add_special_tokens=False,
                return_tensors="pt",
            )
            .input_ids[0]
            .long()
        )
        if int(suffix.numel()) <= 0:
            raise V5PipelineError("runtime suffix/query tokenization is empty")
        return suffix

    def _validate_request_length(self, prefix_tokens: int, suffix_tokens: int) -> None:
        semantic = self._require_semantic()
        if prefix_tokens + suffix_tokens + RISK_LABEL_GENERATION_TOKENS > min(
            self.max_model_len,
            semantic.source.max_position_embeddings,
            semantic.target.max_position_embeddings,
        ):
            raise V5PipelineError("runtime request exceeds the model position contract")

    def _backing_files_remaining(self) -> int:
        server = self._require_server()
        server.assert_no_backing_files()
        return server.backing_files_remaining

    def _close(self, *, suppress_errors: bool) -> None:
        errors: list[Exception] = []
        if self.llm is not None:
            try:
                self.llm.llm_engine.engine_core.shutdown(timeout=60.0)
            except Exception as exc:
                errors.append(exc)
            self.llm = None
        if self.collector is not None:
            if not suppress_errors:
                try:
                    self.collector.assert_drained()
                except Exception as exc:
                    errors.append(exc)
            try:
                self.collector.close()
            except Exception as exc:
                errors.append(exc)
            self.collector = None
        if self.writer is not None:
            try:
                self.writer.close()
            except Exception as exc:
                errors.append(exc)
            self.writer = None
        if self.server is not None:
            try:
                self.server.stop()
            except Exception as exc:
                errors.append(exc)
            self.server = None
        self.assets.clear()
        self._pending_contexts.clear()
        self._failure_probe_context = None
        self.transport = None
        self.tokenizer = None
        self.source_model = None
        self.target_model = None
        self._entered = False
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception as exc:
            errors.append(exc)
        if errors and not suppress_errors:
            raise V5PipelineError(
                "runtime evaluator shutdown failed: " + "; ".join(map(str, errors))
            )

    def _require_semantic(self) -> SelectiveKVBridgeManifest:
        if (
            self.semantic_selective is None
            or self.semantic_selective.state is not ArtifactState.SEMANTIC_APPROVED
        ):
            raise V5PipelineError("runtime evaluator lacks semantic approval")
        return self.semantic_selective

    def _require_semantic_path(self) -> Path:
        if self.semantic_selective_path is None:
            raise V5PipelineError("runtime evaluator lacks the semantic artifact path")
        return self.semantic_selective_path

    def _require_server(self) -> LMCacheMPServerProcess:
        if self.server is None or not self.server.running:
            raise V5PipelineError("runtime evaluator LMCache server is not running")
        return self.server

    def _require_collector(self) -> RuntimeAuditTelemetryCollector:
        if self.collector is None:
            raise V5PipelineError("runtime evaluator telemetry collector is not running")
        return self.collector

    def _require_writer(self) -> LMCacheMPSourceChunkWriter:
        if self.writer is None:
            raise V5PipelineError("runtime evaluator source writer is not loaded")
        return self.writer

    def _require_gate(self) -> CalibratedRiskGate:
        if self.gate is None:
            raise V5PipelineError("runtime evaluator risk gate is not loaded")
        return self.gate

    def _require_tokenizer(self) -> Any:
        if self.tokenizer is None:
            raise V5PipelineError("runtime evaluator tokenizer is not loaded")
        return self.tokenizer

    def _require_source_model(self) -> Any:
        if self.source_model is None:
            raise V5PipelineError("runtime evaluator source model is not loaded")
        return self.source_model

    def _require_target_model(self) -> Any:
        if self.target_model is None:
            raise V5PipelineError("runtime evaluator target model is not loaded")
        return self.target_model

    def _require_transport(self) -> HeadAwareKVTransport:
        if self.transport is None:
            raise V5PipelineError("runtime evaluator transport is not loaded")
        return self.transport

    def _required_block_size(self) -> int:
        if self._block_size is None or self._block_size <= 0:
            raise V5PipelineError("runtime evaluator lacks the vLLM block size")
        return self._block_size


def _torch_dtype(name: str) -> Any:
    import torch

    try:
        return {"bfloat16": torch.bfloat16, "float16": torch.float16}[name]
    except KeyError as exc:
        raise V5PipelineError(f"unsupported runtime evaluator dtype {name!r}") from exc


def _cuda_index(device: str) -> int:
    import torch

    parsed = torch.device(device)
    if parsed.type != "cuda" or parsed.index is None:
        raise V5PipelineError("runtime evaluator requires an explicit CUDA device")
    return parsed.index


def _device_name(device: str) -> str:
    import torch

    parsed = torch.device(device)
    if parsed.type != "cuda":
        return parsed.type
    if parsed.index is None:
        raise V5PipelineError("runtime evaluator requires an explicit CUDA device")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    physical_index = parsed.index
    if visible:
        identifiers = [item.strip() for item in visible.split(",") if item.strip()]
        if parsed.index >= len(identifiers) or not identifiers[parsed.index].isdigit():
            raise V5PipelineError("runtime evaluator cannot resolve CUDA_VISIBLE_DEVICES")
        physical_index = int(identifiers[parsed.index])
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=5.0,
        )
        devices = {
            int(index.strip()): name.strip()
            for line in completed.stdout.splitlines()
            for index, name in [line.split(",", 1)]
        }
        return devices[physical_index]
    except (KeyError, OSError, subprocess.SubprocessError, ValueError) as exc:
        raise V5PipelineError("runtime evaluator could not identify its CUDA device") from exc


def _perplexity_drift_pct(native_nll: float, bridge_nll: float, tokens: int) -> float:
    if not _finite_nonnegative(native_nll) or not _finite_nonnegative(bridge_nll) or tokens <= 0:
        raise V5PipelineError("runtime shadow NLL evidence is invalid")
    try:
        value = abs(math.expm1((bridge_nll - native_nll) / tokens)) * 100
    except OverflowError:
        value = sys.float_info.max
    if not math.isfinite(value):
        if math.isnan(value):
            raise V5PipelineError("runtime shadow perplexity drift is not finite")
        return sys.float_info.max
    return min(sys.float_info.max, value)


def _integer_list(value: Any) -> tuple[int, ...]:
    if not isinstance(value, list) or any(type(item) is not int or item < 0 for item in value):
        raise V5PipelineError("runtime worker block telemetry is malformed")
    return tuple(value)


def _canonical_sha256(value: Mapping[str, Any]) -> str:
    payload = json.dumps(
        dict(value),
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _finite_probability(value: Any) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(value)
        and 0 <= value <= 1
    )


def _finite_nonnegative(value: Any) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(value)
        and value >= 0
    )


def _finite_positive(value: Any) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(value)
        and value > 0
    )
