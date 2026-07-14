"""Target-logit supervision for v5 transport fitting."""

from __future__ import annotations

import gc
import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from goldenexperience.size_variant.cached_kv_manifest import (
    CachedKVModelSpec,
    verify_model_path,
)
from goldenexperience.size_variant.v5_collect import RawBenchmarkSample, TraceRecord
from goldenexperience.size_variant.v5_pipeline import V5PipelineError

TARGET_LOGIT_SUPERVISION_ID = "sampled_prefix_native_greedy_distillation_v1"
TRACE_CONSTANT_SUPERVISION_ID = "trace_constant_reporting_v1"
ABSOLUTE_HEAD_TAIL_TRUNCATION = "absolute_head_tail_v1"
NO_TRUNCATION = "none"
REGISTERED_TEACHER_TOKENS = 4
REGISTERED_MAX_SUFFIX_TOKENS = 256


@dataclass(frozen=True)
class GenerationSupervisionSpec:
    supervision_id: str = TARGET_LOGIT_SUPERVISION_ID
    teacher_tokens: int = REGISTERED_TEACHER_TOKENS
    max_suffix_tokens: int = REGISTERED_MAX_SUFFIX_TOKENS
    truncation: str = ABSOLUTE_HEAD_TAIL_TRUNCATION
    teacher_cache_dtype: str = "bfloat16"

    @classmethod
    def legacy(cls) -> GenerationSupervisionSpec:
        return cls(
            supervision_id=TRACE_CONSTANT_SUPERVISION_ID,
            teacher_tokens=0,
            max_suffix_tokens=0,
            truncation=NO_TRUNCATION,
            teacher_cache_dtype="none",
        )

    def validate(self, *, require_registered: bool = True) -> list[str]:
        errors: list[str] = []
        if self.supervision_id == TRACE_CONSTANT_SUPERVISION_ID:
            if self != self.legacy():
                errors.append("legacy generation supervision contract is malformed")
            if require_registered:
                errors.append("publication fitting requires target-logit supervision")
            return errors
        if self.supervision_id != TARGET_LOGIT_SUPERVISION_ID:
            errors.append("generation supervision method is unsupported")
            return errors
        if type(self.teacher_tokens) is not int or self.teacher_tokens <= 0:
            errors.append("generation teacher token count must be positive")
        if (
            type(self.max_suffix_tokens) is not int
            or self.max_suffix_tokens < 2
            or self.max_suffix_tokens % 2
        ):
            errors.append("generation suffix bound must be a positive even integer")
        if self.truncation != ABSOLUTE_HEAD_TAIL_TRUNCATION:
            errors.append("generation suffix truncation contract is unsupported")
        if self.teacher_cache_dtype != "bfloat16":
            errors.append("generation teacher cache must use bfloat16")
        if require_registered and self != GenerationSupervisionSpec():
            errors.append("publication generation supervision differs from the frozen contract")
        return errors

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class BoundedSuffix:
    input_ids: Any
    position_ids: Any
    full_token_count: int


@dataclass(frozen=True)
class NativeTeacher:
    input_ids: Any
    position_ids: Any
    teacher_tokens: Any
    teacher_logits: Any


def bound_suffix_token_ids(
    token_ids: Any,
    *,
    prefix_token_count: int,
    max_position_embeddings: int,
    spec: GenerationSupervisionSpec,
) -> BoundedSuffix:
    """Bound a suffix while retaining every selected token's absolute position."""

    import torch

    errors = spec.validate(require_registered=False)
    if errors or spec.supervision_id != TARGET_LOGIT_SUPERVISION_ID:
        raise V5PipelineError("; ".join(errors or ["target-logit supervision is required"]))
    ids = torch.as_tensor(token_ids, dtype=torch.long, device="cpu").reshape(-1)
    if ids.numel() <= 0:
        raise V5PipelineError("transport generation suffix is empty")
    if type(prefix_token_count) is not int or prefix_token_count <= 0:
        raise V5PipelineError("transport generation prefix length is invalid")
    full_count = int(ids.numel())
    if prefix_token_count + full_count + spec.teacher_tokens > max_position_embeddings:
        raise V5PipelineError("transport generation request exceeds the model position contract")
    if full_count <= spec.max_suffix_tokens:
        positions = torch.arange(prefix_token_count, prefix_token_count + full_count)
        bounded = ids
    else:
        head_count = spec.max_suffix_tokens // 2
        tail_count = spec.max_suffix_tokens - head_count
        bounded = torch.cat((ids[:head_count], ids[-tail_count:]))
        positions = torch.cat(
            (
                torch.arange(prefix_token_count, prefix_token_count + head_count),
                torch.arange(
                    prefix_token_count + full_count - tail_count,
                    prefix_token_count + full_count,
                ),
            )
        )
    return BoundedSuffix(
        input_ids=bounded.contiguous(),
        position_ids=positions.long().contiguous(),
        full_token_count=full_count,
    )


def batched_head_object_to_dynamic_cache(kv_batch: Any, config: Any) -> Any:
    """Build a differentiable DynamicCache from `[batch, 2, layer, head, token, dim]`."""

    from transformers.cache_utils import DynamicCache

    if kv_batch.ndim != 6 or int(kv_batch.shape[1]) != 2:
        raise ValueError("batched KV must have [batch, 2, layer, head, token, dim] layout")
    heads = int(config.num_key_value_heads)
    head_dim = int(
        getattr(config, "head_dim", 0) or config.hidden_size // config.num_attention_heads
    )
    if int(kv_batch.shape[3]) != heads or int(kv_batch.shape[5]) != head_dim:
        raise ValueError("batched KV does not match the target head layout")
    cache = DynamicCache(config=config)
    for layer in range(int(kv_batch.shape[2])):
        cache.update(kv_batch[:, 0, layer], kv_batch[:, 1, layer], layer)
    return cache


def prepare_native_teacher(
    model: Any,
    native_target_kv: Any,
    suffix: BoundedSuffix,
    *,
    prefix_token_count: int,
    spec: GenerationSupervisionSpec,
    device: str,
) -> NativeTeacher:
    """Generate native sampled-cache teacher tokens and detached logits."""

    import torch

    if native_target_kv.ndim != 5 or int(native_target_kv.shape[0]) != 2:
        raise V5PipelineError("native generation KV layout is invalid")
    cache = batched_head_object_to_dynamic_cache(
        native_target_kv.to(device).unsqueeze(0),
        model.config,
    )
    input_ids = suffix.input_ids.to(device).unsqueeze(0)
    position_ids = suffix.position_ids.to(device).unsqueeze(0)
    teacher_tokens: list[Any] = []
    teacher_logits: list[Any] = []
    with torch.inference_mode():
        output = model(
            input_ids=input_ids,
            past_key_values=cache,
            position_ids=position_ids,
            use_cache=True,
            logits_to_keep=1,
        )
        cache = output.past_key_values
        logits = output.logits[:, -1]
        for index in range(spec.teacher_tokens):
            if logits.ndim != 2 or int(logits.shape[0]) != 1:
                raise V5PipelineError("native teacher logits have an invalid layout")
            token = logits.argmax(dim=-1).long()
            teacher_tokens.append(token.to("cpu"))
            teacher_logits.append(logits.to(device="cpu", dtype=torch.bfloat16))
            if index + 1 == spec.teacher_tokens:
                break
            generated_position = prefix_token_count + suffix.full_token_count + index
            output = model(
                input_ids=token.view(1, 1),
                past_key_values=cache,
                position_ids=torch.tensor([[generated_position]], device=device),
                use_cache=True,
                logits_to_keep=1,
            )
            cache = output.past_key_values
            logits = output.logits[:, -1]
    tokens = torch.cat(teacher_tokens).long()
    logits = torch.cat(teacher_logits)
    generated_inputs = tokens[:-1]
    generated_positions = torch.arange(
        prefix_token_count + suffix.full_token_count,
        prefix_token_count + suffix.full_token_count + spec.teacher_tokens - 1,
    )
    result = NativeTeacher(
        input_ids=torch.cat((suffix.input_ids, generated_inputs)).long().contiguous(),
        position_ids=torch.cat((suffix.position_ids, generated_positions)).long().contiguous(),
        teacher_tokens=tokens.contiguous(),
        teacher_logits=logits.contiguous(),
    )
    if (
        result.teacher_tokens.shape != (spec.teacher_tokens,)
        or result.teacher_logits.ndim != 2
        or int(result.teacher_logits.shape[0]) != spec.teacher_tokens
        or result.input_ids.shape != result.position_ids.shape
        or not bool(torch.isfinite(result.teacher_logits.float()).all())
    ):
        raise V5PipelineError("native generation teacher is invalid")
    return result


def generation_distillation_losses(
    model: Any,
    transformed_kv_batch: Any,
    teacher: NativeTeacher,
    *,
    device: str,
) -> tuple[Any, Any]:
    """Return per-candidate greedy-token CE and teacher-logit KL losses."""

    import torch
    import torch.nn.functional as functional

    batch = int(transformed_kv_batch.shape[0])
    if batch <= 0:
        raise V5PipelineError("generation distillation candidate batch is empty")
    model_dtype = next(model.parameters()).dtype
    cache = batched_head_object_to_dynamic_cache(
        transformed_kv_batch.to(device=device, dtype=model_dtype),
        model.config,
    )
    input_ids = teacher.input_ids.to(device).unsqueeze(0).expand(batch, -1)
    position_ids = teacher.position_ids.to(device).unsqueeze(0).expand(batch, -1)
    output = model(
        input_ids=input_ids,
        past_key_values=cache,
        position_ids=position_ids,
        use_cache=False,
        logits_to_keep=int(teacher.teacher_tokens.numel()),
    )
    student_logits = output.logits.float()
    expected_shape = (
        batch,
        int(teacher.teacher_tokens.numel()),
        int(teacher.teacher_logits.shape[-1]),
    )
    if tuple(student_logits.shape) != expected_shape:
        raise V5PipelineError("student generation logits have an invalid layout")
    labels = teacher.teacher_tokens.to(device).unsqueeze(0).expand(batch, -1)
    generation = functional.cross_entropy(
        student_logits.reshape(-1, student_logits.shape[-1]),
        labels.reshape(-1),
        reduction="none",
    ).reshape(batch, -1).mean(dim=-1)
    teacher_logits = teacher.teacher_logits.to(device=device, dtype=torch.float32)
    teacher_probability = teacher_logits.softmax(dim=-1).unsqueeze(0)
    teacher_log_probability = teacher_probability.clamp_min(
        torch.finfo(teacher_probability.dtype).tiny
    ).log()
    student_log_probability = student_logits.log_softmax(dim=-1)
    distillation = (
        teacher_probability * (teacher_log_probability - student_log_probability)
    ).sum(dim=-1).mean(dim=-1).clamp_min(0.0)
    if (
        generation.shape != (batch,)
        or distillation.shape != (batch,)
        or not bool(torch.isfinite(generation).all())
        or not bool(torch.isfinite(distillation).all())
    ):
        raise V5PipelineError("generation distillation produced non-finite losses")
    return generation, distillation


class TraceConstantGenerationBackend:
    """Compatibility backend for pre-v3 fit tests and manifests."""

    supervision_id = TRACE_CONSTANT_SUPERVISION_ID

    def __enter__(self) -> TraceConstantGenerationBackend:
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def parameters(self) -> dict[str, Any]:
        return {"supervision_id": self.supervision_id}

    def losses(
        self,
        _record: TraceRecord,
        tensors: Mapping[str, Any],
        transformed_kv_batch: Any,
    ) -> tuple[Any, Any]:
        import torch

        constants = tensors["constant_losses"].to(
            device=transformed_kv_batch.device,
            dtype=torch.float32,
        )
        if constants.shape != (2,) or not bool(torch.isfinite(constants).all()):
            raise V5PipelineError("trace generation constants are invalid")
        batch = int(transformed_kv_batch.shape[0])
        return constants[0].expand(batch), constants[1].expand(batch)


class TargetLogitGenerationBackend:
    """Load the frozen target and provide cached train-row logit supervision."""

    supervision_id = TARGET_LOGIT_SUPERVISION_ID

    def __init__(
        self,
        *,
        target_path: str | Path,
        target: CachedKVModelSpec,
        samples: Mapping[str, RawBenchmarkSample],
        device: str,
        identity_cache_path: str | Path | None,
        spec: GenerationSupervisionSpec,
        attention_implementation: str = "sdpa",
        seed: int = 17,
    ) -> None:
        self.target_path = Path(target_path).resolve()
        self.target = target
        self.samples = dict(samples)
        self.device = device
        self.identity_cache_path = identity_cache_path
        self.spec = spec
        self.attention_implementation = attention_implementation
        self.seed = seed
        self.tokenizer: Any | None = None
        self.model: Any | None = None
        self.teacher_cache: dict[str, NativeTeacher] = {}

    def parameters(self) -> dict[str, Any]:
        import torch
        import transformers

        return {
            **self.spec.to_dict(),
            "attention_implementation": self.attention_implementation,
            "seed": self.seed,
            "torch_version": torch.__version__,
            "transformers_version": transformers.__version__,
            "cuda_version": torch.version.cuda,
            "target_device_type": torch.device(self.device).type,
            "target_device_name": _device_name(self.device),
        }

    def __enter__(self) -> TargetLogitGenerationBackend:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        errors = self.spec.validate(require_registered=True)
        errors.extend(
            verify_model_path(
                self.target,
                self.target_path,
                identity_cache_path=self.identity_cache_path,
            )
        )
        if errors:
            raise V5PipelineError(f"target-logit supervision is invalid: {'; '.join(errors)}")
        torch.manual_seed(self.seed)
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.target_path,
            local_files_only=True,
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            self.target_path,
            local_files_only=True,
            dtype=_torch_dtype(self.target.dtype),
            attn_implementation=self.attention_implementation,
            device_map={"": self.device},
        ).eval()
        self.model.requires_grad_(False)
        return self

    def __exit__(self, *_args: object) -> None:
        import torch

        self.teacher_cache.clear()
        self.tokenizer = None
        self.model = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def losses(
        self,
        record: TraceRecord,
        tensors: Mapping[str, Any],
        transformed_kv_batch: Any,
    ) -> tuple[Any, Any]:
        if self.tokenizer is None or self.model is None:
            raise V5PipelineError("target-logit generation backend is not loaded")
        sample = self.samples.get(record.sample_id)
        if sample is None:
            raise V5PipelineError("generation backend lacks the bound transport-train sample")
        teacher = self.teacher_cache.get(record.sample_id)
        if teacher is None:
            encoded = self.tokenizer(
                sample.suffix_query,
                add_special_tokens=False,
                return_tensors="pt",
            ).input_ids[0]
            suffix = bound_suffix_token_ids(
                encoded,
                prefix_token_count=record.token_count,
                max_position_embeddings=self.target.max_position_embeddings,
                spec=self.spec,
            )
            teacher = prepare_native_teacher(
                self.model,
                tensors["target_kv"],
                suffix,
                prefix_token_count=record.token_count,
                spec=self.spec,
                device=self.device,
            )
            self.teacher_cache[record.sample_id] = teacher
        return generation_distillation_losses(
            self.model,
            transformed_kv_batch,
            teacher,
            device=self.device,
        )


def _torch_dtype(name: str) -> Any:
    import torch

    values = {"bfloat16": torch.bfloat16, "float16": torch.float16}
    try:
        return values[name]
    except KeyError as exc:
        raise V5PipelineError(f"unsupported generation target dtype {name!r}") from exc


def _device_name(device: str) -> str:
    import torch

    parsed = torch.device(device)
    return torch.cuda.get_device_name(parsed) if parsed.type == "cuda" else parsed.type


def validate_loss_values(generation: Any, distillation: Any) -> None:
    """Small public guard used by custom backends in tests and diagnostics."""

    for name, value in (("generation", generation), ("distillation", distillation)):
        if value.ndim != 1 or any(not math.isfinite(float(item)) for item in value.detach().cpu()):
            raise V5PipelineError(f"{name} supervision losses are invalid")
