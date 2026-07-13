"""Real Qwen selector example backend for v5 risk fitting."""

from __future__ import annotations

import gc
import hashlib
import math
import sys
from pathlib import Path
from typing import Any

from goldenexperience.benchmarks.publication import GroupedPrefixRecord
from goldenexperience.benchmarks.publication_eval import (
    publication_pass_threshold,
    score_publication_prediction,
)
from goldenexperience.runtime.cross_model_reuse import token_ids_sha256
from goldenexperience.size_variant.cached_kv_manifest import verify_model_path
from goldenexperience.size_variant.head_aware_transport import (
    HeadAwareKVTransport,
    dynamic_cache_to_head_object,
)
from goldenexperience.size_variant.risk_gate import (
    SourceKVSidecar,
    build_transport_source_sidecar,
    unsafe_label,
)
from goldenexperience.size_variant.v5_collect import RawBenchmarkSample, TraceRecord
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
    RiskTrainingExample,
)

V5_REAL_RISK_EVALUATOR_ID = "qwen3_real_risk_examples_v1"


class RealQwenRiskExampleEvaluator:
    """Build quantized-runtime-equivalent source features and target-derived labels."""

    def __init__(
        self,
        *,
        workspace: V5PipelineWorkspace,
        transport_manifest: V5TransportFitManifest | V5DirectionalTransportFitManifest,
        candidate: TransportCandidateArtifact,
        source_path: str | Path,
        target_path: str | Path,
        source_device: str,
        target_device: str,
        identity_cache_path: str | Path | None,
        attention_implementation: str = "sdpa",
        seed: int = 17,
    ) -> None:
        self.workspace = workspace
        self.transport_manifest = transport_manifest
        self.candidate = candidate
        self.source_path = Path(source_path).resolve()
        self.target_path = Path(target_path).resolve()
        self.source_device = source_device
        self.target_device = target_device
        self.identity_cache_path = identity_cache_path
        self.attention_implementation = attention_implementation
        self.seed = seed
        self.tokenizer: Any | None = None
        self.source_model: Any | None = None
        self.target_model: Any | None = None
        self.transport: HeadAwareKVTransport | None = None

    def parameters(self) -> dict[str, Any]:
        import torch
        import transformers

        return {
            "evaluator_id": V5_REAL_RISK_EVALUATOR_ID,
            "label_generation_tokens": RISK_LABEL_GENERATION_TOKENS,
            "sidecar_round_trip_before_features": True,
            "history_order": "lexicographic_sample_id_within_frozen_split",
            "seed": self.seed,
            "attention_implementation": self.attention_implementation,
            "torch_version": torch.__version__,
            "transformers_version": transformers.__version__,
            "cuda_version": torch.version.cuda,
            "source_device_type": torch.device(self.source_device).type,
            "source_device_name": _device_name(self.source_device),
            "target_device_type": torch.device(self.target_device).type,
            "target_device_name": _device_name(self.target_device),
        }

    def __enter__(self) -> RealQwenRiskExampleEvaluator:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        for label, expected, path in (
            ("source", self.transport_manifest.source, self.source_path),
            ("target", self.transport_manifest.target, self.target_path),
        ):
            errors = verify_model_path(
                expected,
                path,
                identity_cache_path=self.identity_cache_path,
            )
            if errors:
                raise V5PipelineError(f"{label} model identity mismatch: {'; '.join(errors)}")
        torch.manual_seed(self.seed)
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.target_path,
            local_files_only=True,
        )
        self.source_model = AutoModelForCausalLM.from_pretrained(
            self.source_path,
            local_files_only=True,
            dtype=_torch_dtype(self.transport_manifest.source.dtype),
            attn_implementation=self.attention_implementation,
            device_map={"": self.source_device},
        ).eval()
        self.target_model = AutoModelForCausalLM.from_pretrained(
            self.target_path,
            local_files_only=True,
            dtype=_torch_dtype(self.transport_manifest.target.dtype),
            attn_implementation=self.attention_implementation,
            device_map={"": self.target_device},
        ).eval()
        self.transport = load_fitted_transport(
            self.workspace,
            self.transport_manifest,
            self.candidate,
            device=self.target_device,
        )[0]
        return self

    def __exit__(self, *_args: object) -> None:
        import torch

        self.transport = None
        self.tokenizer = None
        self.source_model = None
        self.target_model = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def evaluate(
        self,
        benchmark_record: GroupedPrefixRecord,
        trace_record: TraceRecord,
        sample: RawBenchmarkSample,
        history: RiskHistory,
    ) -> RiskTrainingExample:
        import torch
        import torch.nn.functional as functional

        if (
            self.tokenizer is None
            or self.source_model is None
            or self.target_model is None
            or self.transport is None
        ):
            raise V5PipelineError("real risk evaluator is not loaded")
        prefix = self.tokenizer(
            sample.prefix_text,
            add_special_tokens=False,
            return_tensors="pt",
        ).input_ids[0]
        if prefix.numel() < trace_record.token_count:
            raise V5PipelineError("risk sample has fewer prefix tokens than registered")
        prefix = prefix[: trace_record.token_count].long()
        if token_ids_sha256(prefix.tolist()) != trace_record.token_ids_sha256:
            raise V5PipelineError("risk sample prefix tokens differ from collected traces")
        suffix = (
            self.tokenizer(
                sample.suffix_query,
                add_special_tokens=False,
                return_tensors="pt",
            )
            .input_ids[0]
            .long()
        )
        if suffix.numel() <= 0:
            raise V5PipelineError("risk sample suffix/query tokenization is empty")
        if prefix.numel() + suffix.numel() + RISK_LABEL_GENERATION_TOKENS > min(
            self.transport_manifest.source.max_position_embeddings,
            self.transport_manifest.target.max_position_embeddings,
        ):
            raise V5PipelineError("risk request exceeds the model position contract")
        with torch.inference_mode():
            source_output = self.source_model(
                input_ids=prefix.unsqueeze(0).to(self.source_device),
                use_cache=True,
                logits_to_keep=1,
            )
            target_output = self.target_model(
                input_ids=prefix.unsqueeze(0).to(self.target_device),
                use_cache=True,
                logits_to_keep=1,
            )
        source_kv = dynamic_cache_to_head_object(source_output.past_key_values).to(
            self.target_device
        )
        target_kv = dynamic_cache_to_head_object(target_output.past_key_values)
        del source_output, target_output
        sidecar = build_transport_source_sidecar(
            source_kv,
            self.transport,
            model_pair_id=self.transport_manifest.direction,
            prefix_hash=benchmark_record.prefix_sha256,
            history_samples=history.samples,
            history_failures=history.failures,
            history_greedy_agreement=history.greedy_agreement,
        )
        sidecar_payload = sidecar.to_bytes()
        runtime_sidecar = SourceKVSidecar.from_bytes(sidecar_payload)
        features = runtime_sidecar.risk_features()
        positions = torch.arange(trace_record.token_count, device=self.target_device)
        transformed = self.transport.transform(source_kv, position_ids=positions)
        native_tokens, native_text, native_nll = greedy_decode(
            self.target_model,
            self.tokenizer,
            target_kv,
            suffix,
            device=self.target_device,
            generation_tokens=RISK_LABEL_GENERATION_TOKENS,
        )
        bridge_tokens, bridge_text, _ = greedy_decode(
            self.target_model,
            self.tokenizer,
            transformed,
            suffix,
            device=self.target_device,
            generation_tokens=RISK_LABEL_GENERATION_TOKENS,
        )
        bridge_nll = teacher_nll(
            self.target_model,
            transformed,
            suffix,
            native_tokens,
            device=self.target_device,
        )
        native_score = score_publication_prediction(
            native_text,
            sample.reference,
            sample.evaluation,
        )
        bridge_score = score_publication_prediction(
            bridge_text,
            sample.reference,
            sample.evaluation,
        )
        threshold = publication_pass_threshold(sample.evaluation)
        greedy_matches = sum(
            native == bridge for native, bridge in zip(native_tokens, bridge_tokens, strict=True)
        )
        greedy_agreement = greedy_matches / RISK_LABEL_GENERATION_TOKENS
        try:
            perplexity_drift = (
                abs(math.expm1((bridge_nll - native_nll) / RISK_LABEL_GENERATION_TOKENS)) * 100
            )
            perplexity_drift = min(sys.float_info.max, perplexity_drift)
        except OverflowError:
            perplexity_drift = sys.float_info.max
        key_cosine = float(
            functional.cosine_similarity(
                transformed[0].float().reshape(-1, transformed.shape[-1]),
                target_kv[0].float().reshape(-1, target_kv.shape[-1]),
                dim=-1,
            )
            .mean()
            .item()
        )
        key_cosine = min(1.0, max(-1.0, key_cosine))
        example = RiskTrainingExample(
            sample_id=trace_record.sample_id,
            prefix_group_id=benchmark_record.prefix_group_id,
            features=features,
            unsafe=unsafe_label(
                native_task_passed=native_score >= threshold,
                bridge_task_passed=bridge_score >= threshold,
                greedy_agreement=greedy_agreement,
                perplexity_drift_pct=perplexity_drift,
            ),
            native_task_score=native_score,
            bridge_task_score=bridge_score,
            task_pass_threshold=threshold,
            greedy_matches=greedy_matches,
            greedy_tokens=RISK_LABEL_GENERATION_TOKENS,
            native_nll=native_nll,
            bridge_nll=bridge_nll,
            teacher_tokens=RISK_LABEL_GENERATION_TOKENS,
            key_cosine=key_cosine,
            history_samples=history.samples,
            history_failures=history.failures,
            history_greedy_agreement=history.greedy_agreement,
            sidecar_sha256=hashlib.sha256(sidecar_payload).hexdigest(),
            native_prediction_sha256=_sha256_text(native_text),
            bridge_prediction_sha256=_sha256_text(bridge_text),
            native_tokens_sha256=token_ids_sha256(native_tokens),
            bridge_tokens_sha256=token_ids_sha256(bridge_tokens),
        )
        del source_kv, target_kv, transformed
        return example


def _torch_dtype(name: str) -> Any:
    import torch

    values = {"bfloat16": torch.bfloat16, "float16": torch.float16}
    try:
        return values[name]
    except KeyError as exc:
        raise V5PipelineError(f"unsupported risk evaluator dtype {name!r}") from exc


def _device_name(device: str) -> str:
    import torch

    parsed = torch.device(device)
    return torch.cuda.get_device_name(parsed) if parsed.type == "cuda" else parsed.type


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()
