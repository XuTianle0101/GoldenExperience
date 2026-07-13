"""Real Qwen backend for frozen v5 method-dev transport screening."""

from __future__ import annotations

import gc
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from goldenexperience.benchmarks.publication_eval import (
    publication_pass_threshold,
    score_publication_prediction,
)
from goldenexperience.runtime.cross_model_reuse import token_ids_sha256
from goldenexperience.size_variant.cached_kv_manifest import verify_model_path
from goldenexperience.size_variant.head_aware_transport import (
    HeadAwareKVTransport,
    dynamic_cache_to_head_object,
    head_object_to_dynamic_cache,
)
from goldenexperience.size_variant.v5_collect import RawBenchmarkSample, TraceRecord
from goldenexperience.size_variant.v5_fit import (
    V5TransportFitManifest,
    load_fitted_transport,
)
from goldenexperience.size_variant.v5_method_dev import (
    METHOD_DEV_GENERATION_TOKENS,
    MethodDevMeasurement,
)
from goldenexperience.size_variant.v5_pipeline import V5PipelineError, V5PipelineWorkspace

V5_REAL_METHOD_DEV_EVALUATOR_ID = "qwen3_real_method_dev_v1"


@dataclass(frozen=True)
class _MethodDevPrefixAsset:
    prefix_group_id: str
    token_ids: tuple[int, ...]
    source_kv: Any
    target_kv: Any


class RealQwenMethodDevEvaluator:
    """Evaluate all fitted candidates while sharing each source/target prefix prefill."""

    def __init__(
        self,
        *,
        workspace: V5PipelineWorkspace,
        fit: V5TransportFitManifest,
        source_path: str | Path,
        target_path: str | Path,
        source_device: str,
        target_device: str,
        identity_cache_path: str | Path | None,
        attention_implementation: str = "sdpa",
        seed: int = 17,
    ) -> None:
        self.workspace = workspace
        self.fit = fit
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
        self.transports: dict[str, HeadAwareKVTransport] = {}
        self._prefix_asset: _MethodDevPrefixAsset | None = None

    def parameters(self) -> dict[str, Any]:
        import torch
        import transformers

        return {
            "evaluator_id": V5_REAL_METHOD_DEV_EVALUATOR_ID,
            "generation_tokens": METHOD_DEV_GENERATION_TOKENS,
            "seed": self.seed,
            "attention_implementation": self.attention_implementation,
            "prefix_prefill_reuse": "contiguous_prefix_group_v1",
            "torch_version": torch.__version__,
            "transformers_version": transformers.__version__,
            "cuda_version": torch.version.cuda,
            "source_device_type": torch.device(self.source_device).type,
            "source_device_name": _device_name(self.source_device),
            "target_device_type": torch.device(self.target_device).type,
            "target_device_name": _device_name(self.target_device),
        }

    def __enter__(self) -> RealQwenMethodDevEvaluator:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        for label, expected, path in (
            ("source", self.fit.source, self.source_path),
            ("target", self.fit.target, self.target_path),
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
            dtype=_torch_dtype(self.fit.source.dtype),
            attn_implementation=self.attention_implementation,
            device_map={"": self.source_device},
        ).eval()
        self.target_model = AutoModelForCausalLM.from_pretrained(
            self.target_path,
            local_files_only=True,
            dtype=_torch_dtype(self.fit.target.dtype),
            attn_implementation=self.attention_implementation,
            device_map={"": self.target_device},
        ).eval()
        self.transports = {
            candidate.candidate_id: load_fitted_transport(
                self.workspace,
                self.fit,
                candidate,
                device=self.target_device,
            )[0]
            for candidate in self.fit.candidates
        }
        return self

    def __exit__(self, *_args: object) -> None:
        import torch

        self.transports.clear()
        self._prefix_asset = None
        self.tokenizer = None
        self.source_model = None
        self.target_model = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def evaluate(
        self,
        record: TraceRecord,
        sample: RawBenchmarkSample,
    ) -> tuple[MethodDevMeasurement, ...]:
        import torch

        if self.tokenizer is None or self.source_model is None or self.target_model is None:
            raise V5PipelineError("real method-dev evaluator is not loaded")
        asset = self._prefix_asset_for(record, sample)
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
            raise V5PipelineError("method-dev suffix/query tokenization is empty")
        if len(asset.token_ids) + suffix.numel() + METHOD_DEV_GENERATION_TOKENS > min(
            self.fit.source.max_position_embeddings,
            self.fit.target.max_position_embeddings,
        ):
            raise V5PipelineError("method-dev request exceeds the model position contract")
        native_tokens, native_text, native_nll = greedy_decode(
            self.target_model,
            self.tokenizer,
            asset.target_kv,
            suffix,
            device=self.target_device,
            generation_tokens=METHOD_DEV_GENERATION_TOKENS,
        )
        native_task_score = score_publication_prediction(
            native_text,
            sample.reference,
            sample.evaluation,
        )
        threshold = publication_pass_threshold(sample.evaluation)
        positions = torch.arange(record.token_count, device=self.target_device)
        measurements = []
        for candidate in sorted(self.fit.candidates, key=lambda item: (item.rank, item.seed)):
            transport = self.transports.get(candidate.candidate_id)
            if transport is None:
                raise V5PipelineError("real method-dev evaluator lacks a fitted candidate")
            _synchronize(self.target_device)
            started = time.perf_counter()
            transformed = transport.transform(asset.source_kv, position_ids=positions)
            _synchronize(self.target_device)
            transform_ms = (time.perf_counter() - started) * 1000
            bridge_tokens, bridge_text, _ = greedy_decode(
                self.target_model,
                self.tokenizer,
                transformed,
                suffix,
                device=self.target_device,
                generation_tokens=METHOD_DEV_GENERATION_TOKENS,
            )
            bridge_nll = teacher_nll(
                self.target_model,
                transformed,
                suffix,
                native_tokens,
                device=self.target_device,
            )
            bridge_task_score = score_publication_prediction(
                bridge_text,
                sample.reference,
                sample.evaluation,
            )
            measurements.append(
                MethodDevMeasurement(
                    sample_id=record.sample_id,
                    candidate_id=candidate.candidate_id,
                    rank=candidate.rank,
                    seed=candidate.seed,
                    native_task_score=native_task_score,
                    bridge_task_score=bridge_task_score,
                    task_pass_threshold=threshold,
                    greedy_matches=sum(
                        native == bridge
                        for native, bridge in zip(native_tokens, bridge_tokens, strict=True)
                    ),
                    greedy_tokens=METHOD_DEV_GENERATION_TOKENS,
                    native_nll=native_nll,
                    bridge_nll=bridge_nll,
                    teacher_tokens=METHOD_DEV_GENERATION_TOKENS,
                    transform_ms=transform_ms,
                    native_prediction_sha256=_sha256_text(native_text),
                    bridge_prediction_sha256=_sha256_text(bridge_text),
                    native_tokens_sha256=token_ids_sha256(native_tokens),
                    bridge_tokens_sha256=token_ids_sha256(bridge_tokens),
                )
            )
            del transformed
        return tuple(measurements)

    def _prefix_asset_for(
        self,
        record: TraceRecord,
        sample: RawBenchmarkSample,
    ) -> _MethodDevPrefixAsset:
        import torch

        if self.tokenizer is None or self.source_model is None or self.target_model is None:
            raise V5PipelineError("real method-dev evaluator is not loaded")
        prefix = self.tokenizer(
            sample.prefix_text,
            add_special_tokens=False,
            return_tensors="pt",
        ).input_ids[0]
        if prefix.numel() < record.token_count:
            raise V5PipelineError(
                f"method-dev sample {record.sample_id!r} has fewer prefix tokens than registered"
            )
        prefix = prefix[: record.token_count].long()
        token_ids = tuple(int(value) for value in prefix.tolist())
        if token_ids_sha256(list(token_ids)) != record.token_ids_sha256:
            raise V5PipelineError("method-dev prefix tokens differ from collected traces")
        cached = self._prefix_asset
        if cached is not None and cached.prefix_group_id == record.prefix_group_id:
            if cached.token_ids != token_ids:
                raise V5PipelineError("method-dev prefix group changed token identity")
            return cached
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
        asset = _MethodDevPrefixAsset(
            prefix_group_id=record.prefix_group_id,
            token_ids=token_ids,
            source_kv=dynamic_cache_to_head_object(source_output.past_key_values).to(
                self.target_device
            ),
            target_kv=dynamic_cache_to_head_object(target_output.past_key_values),
        )
        del source_output, target_output
        self._prefix_asset = asset
        return asset


def greedy_decode(
    target_model: Any,
    tokenizer: Any,
    prefix_kv: Any,
    suffix: Any,
    *,
    device: str,
    generation_tokens: int,
) -> tuple[list[int], str, float]:
    import torch
    import torch.nn.functional as functional

    cache = head_object_to_dynamic_cache(prefix_kv, target_model.config)
    input_ids = suffix.unsqueeze(0).to(device)
    generated: list[int] = []
    nll = 0.0
    with torch.inference_mode():
        output = target_model(input_ids=input_ids, past_key_values=cache, use_cache=True)
        cache = output.past_key_values
        logits = output.logits[:, -1]
        for index in range(generation_tokens):
            token = int(logits.argmax(dim=-1).item())
            generated.append(token)
            label = torch.tensor([token], dtype=torch.long, device=device)
            nll += float(functional.cross_entropy(logits.float(), label, reduction="sum").item())
            if index + 1 == generation_tokens:
                break
            output = target_model(
                input_ids=label.view(1, 1),
                past_key_values=cache,
                use_cache=True,
            )
            cache = output.past_key_values
            logits = output.logits[:, -1]
    if len(generated) != generation_tokens or not math.isfinite(nll) or nll < 0:
        raise V5PipelineError("method-dev greedy decode produced invalid evidence")
    return generated, tokenizer.decode(generated, skip_special_tokens=True), nll


def teacher_nll(
    target_model: Any,
    prefix_kv: Any,
    suffix: Any,
    teacher_tokens: list[int],
    *,
    device: str,
) -> float:
    import torch
    import torch.nn.functional as functional

    if not teacher_tokens:
        raise V5PipelineError("method-dev teacher continuation is empty")
    teacher = torch.tensor(teacher_tokens, dtype=torch.long)
    inputs = torch.cat((suffix, teacher[:-1])).unsqueeze(0).to(device)
    labels = teacher.to(device)
    cache = head_object_to_dynamic_cache(prefix_kv, target_model.config)
    with torch.inference_mode():
        output = target_model(input_ids=inputs, past_key_values=cache, use_cache=False)
    logits = output.logits[:, -len(teacher_tokens) :].reshape(-1, output.logits.shape[-1])
    nll = float(functional.cross_entropy(logits.float(), labels, reduction="sum").item())
    if not math.isfinite(nll) or nll < 0:
        raise V5PipelineError("method-dev teacher NLL is invalid")
    return nll


def _torch_dtype(name: str) -> Any:
    import torch

    values = {"bfloat16": torch.bfloat16, "float16": torch.float16}
    try:
        return values[name]
    except KeyError as exc:
        raise V5PipelineError(f"unsupported method-dev dtype {name!r}") from exc


def _device_name(device: str) -> str:
    import torch

    parsed = torch.device(device)
    return torch.cuda.get_device_name(parsed) if parsed.type == "cuda" else parsed.type


def _synchronize(device: str) -> None:
    import torch

    parsed = torch.device(device)
    if parsed.type == "cuda":
        torch.cuda.synchronize(parsed)


def _sha256_text(value: str) -> str:
    import hashlib

    return hashlib.sha256(value.encode("utf-8")).hexdigest()
