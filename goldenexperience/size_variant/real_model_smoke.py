"""Real-model diagnostic for the v5 head-aware transport training path."""

from __future__ import annotations

import json
import math
import os
import platform
import time
import uuid
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from goldenexperience.model_config import resolve_rope_theta
from goldenexperience.runtime.cross_model_reuse import token_ids_sha256
from goldenexperience.size_variant.attention_collection import TargetAttentionCollector
from goldenexperience.size_variant.cached_kv_manifest import (
    CachedKVModelSpec,
    model_spec_from_path,
)
from goldenexperience.size_variant.head_aware_transport import (
    build_trainable_head_aware_transport,
    dynamic_cache_to_head_object,
    fit_head_aware_normalizers,
    head_aware_training_objective,
)
from goldenexperience.size_variant.selective_manifest import TransportSpec

REAL_MODEL_SMOKE_SCHEMA_VERSION = "goldenexperience.v5_real_model_smoke.v1"
REAL_MODEL_SMOKE_AUTHORITY = "diagnostic_only"
_LOSS_NAMES = (
    "native_generation",
    "prompt_tail_distillation",
    "attention_logit_kl",
    "attention_output_mse",
    "transformed_kv_anchor",
    "total",
)


class RealModelSmokeError(RuntimeError):
    """Raised when the diagnostic cannot prove its implementation contract."""


def run_real_model_smoke(
    *,
    source_path: str | Path,
    target_path: str | Path,
    source_model_id: str,
    target_model_id: str,
    source_parameter_count_b: float,
    target_parameter_count_b: float,
    source_revision: str,
    target_revision: str,
    direction: str,
    prompt: str,
    source_device: str = "cuda:0",
    target_device: str = "cuda:1",
    max_tokens: int = 64,
    max_queries: int = 8,
    max_keys: int = 32,
    rank: int = 32,
    source_window: int = 1,
    seed: int = 17,
    identity_cache_path: str | Path | None = None,
    refresh_identity: bool = False,
    local_files_only: bool = True,
) -> dict[str, Any]:
    """Execute one bounded real-model forward/objective pass without granting approval."""

    import torch
    import torch.nn.functional as functional
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer

    _validate_run_arguments(
        prompt=prompt,
        max_tokens=max_tokens,
        max_queries=max_queries,
        max_keys=max_keys,
    )
    started = time.perf_counter()
    source_root = Path(source_path).resolve()
    target_root = Path(target_path).resolve()
    identity_started = time.perf_counter()
    source_spec = model_spec_from_path(
        source_root,
        model_id=source_model_id,
        parameter_count_b=source_parameter_count_b,
        revision=source_revision,
        identity_cache_path=identity_cache_path,
        refresh_identity=refresh_identity,
    )
    target_spec = model_spec_from_path(
        target_root,
        model_id=target_model_id,
        parameter_count_b=target_parameter_count_b,
        revision=target_revision,
        identity_cache_path=identity_cache_path,
        refresh_identity=refresh_identity,
    )
    identity_ms = _elapsed_ms(identity_started)
    if source_spec.tokenizer_sha256 != target_spec.tokenizer_sha256:
        raise RealModelSmokeError("source and target token-ID semantics differ")

    torch.manual_seed(seed)
    tokenizer = AutoTokenizer.from_pretrained(
        target_root,
        local_files_only=local_files_only,
    )
    input_ids = tokenizer(
        prompt,
        return_tensors="pt",
        add_special_tokens=True,
    ).input_ids[:, :max_tokens]
    token_count = int(input_ids.shape[1])
    if token_count < 2:
        raise RealModelSmokeError("smoke prompt must produce at least two tokens")

    load_started = time.perf_counter()
    source_model = AutoModelForCausalLM.from_pretrained(
        source_root,
        local_files_only=local_files_only,
        dtype=_torch_dtype(source_spec.dtype),
        attn_implementation="eager",
        device_map={"": source_device},
    ).eval()
    target_model = AutoModelForCausalLM.from_pretrained(
        target_root,
        local_files_only=local_files_only,
        dtype=_torch_dtype(target_spec.dtype),
        attn_implementation="eager",
        device_map={"": target_device},
    ).eval()
    _synchronize(source_device)
    _synchronize(target_device)
    model_load_ms = _elapsed_ms(load_started)

    source_started = time.perf_counter()
    with torch.inference_mode():
        source_output = source_model(
            input_ids=input_ids.to(source_device),
            use_cache=True,
        )
    _synchronize(source_device)
    source_prefill_ms = _elapsed_ms(source_started)

    target_started = time.perf_counter()
    with (
        TargetAttentionCollector(
            target_model,
            token_count=token_count,
            rope_theta=resolve_rope_theta(target_model.config),
            max_queries=max_queries,
            max_keys=max_keys,
            offload_to_cpu=False,
        ) as collector,
        torch.inference_mode(),
    ):
        target_output = target_model(
            input_ids=input_ids.to(target_device),
            use_cache=True,
        )
    _synchronize(target_device)
    target_prefill_ms = _elapsed_ms(target_started)
    trace = collector.trace()

    source_kv = dynamic_cache_to_head_object(source_output.past_key_values)
    target_kv = dynamic_cache_to_head_object(target_output.past_key_values)
    transport_spec = TransportSpec(
        weights_uri="diagnostic-only.safetensors",
        weights_sha256="0" * 64,
        rank=rank,
        source_window=source_window,
    )
    module = build_trainable_head_aware_transport(
        source_spec,
        target_spec,
        transport_spec,
        device=target_device,
        seed=seed,
    )
    key_positions = trace.key_positions.to(target_device)
    source_sample = source_kv.index_select(
        3,
        trace.key_positions.to(source_kv.device),
    ).to(target_device)
    target_sample = target_kv.index_select(3, key_positions)
    fit_head_aware_normalizers(module, [(source_sample, key_positions)])

    objective_started = time.perf_counter()
    native_generation = functional.cross_entropy(
        target_output.logits[:, :-1].float().reshape(-1, target_output.logits.shape[-1]),
        input_ids[:, 1:].to(target_device).reshape(-1),
    )
    tail_tokens = min(8, token_count)
    teacher = target_output.logits[:, -tail_tokens:].float().softmax(dim=-1)
    student_log = (
        source_output.logits[:, -tail_tokens:].to(target_device).float().log_softmax(dim=-1)
    )
    prompt_tail = functional.kl_div(student_log, teacher, reduction="batchmean") / tail_tokens
    transformed, terms = head_aware_training_objective(
        module,
        source_sample,
        target_sample,
        key_positions,
        trace.queries,
        native_generation_loss=native_generation,
        prompt_tail_distillation_loss=prompt_tail,
        attention_mask=trace.causal_mask,
        native_attention_output=trace.attention_outputs,
    )
    _synchronize(target_device)
    objective_ms = _elapsed_ms(objective_started)
    losses = {name: float(getattr(terms, name).detach().float().item()) for name in _LOSS_NAMES}
    shapes = {
        "source_kv": list(source_kv.shape),
        "target_kv": list(target_kv.shape),
        "target_query": list(trace.queries.shape),
        "native_attention_output": list(trace.attention_outputs.shape),
        "transformed_kv": list(transformed.shape),
    }
    shape_contract_passed = shapes["transformed_kv"] == [
        2,
        target_spec.num_layers,
        target_spec.num_key_value_heads,
        int(trace.key_positions.numel()),
        target_spec.head_dim,
    ]
    all_losses_finite = all(math.isfinite(value) for value in losses.values())
    report = {
        "schema_version": REAL_MODEL_SMOKE_SCHEMA_VERSION,
        "authority": REAL_MODEL_SMOKE_AUTHORITY,
        "status": "passed" if shape_contract_passed and all_losses_finite else "failed",
        "approval_granted": False,
        "evidence_eligible": False,
        "sealed_split_accessed": False,
        "direction": direction,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "cuda": torch.version.cuda,
            "source_device": source_device,
            "source_device_name": _device_name(source_device),
            "target_device": target_device,
            "target_device_name": _device_name(target_device),
        },
        "source": asdict(source_spec),
        "target": asdict(target_spec),
        "input": {
            "prompt_sha256": _text_sha256(prompt),
            "token_ids_sha256": token_ids_sha256(input_ids[0].tolist()),
            "token_count": token_count,
            "tail_tokens": tail_tokens,
        },
        "transport": {
            "rank": rank,
            "source_window": source_window,
            "seed": seed,
            "query_sample_count": int(trace.query_positions.numel()),
            "key_sample_count": int(trace.key_positions.numel()),
            "normalizer_scope": "same_smoke_sample_diagnostic_only",
        },
        "shapes": shapes,
        "losses": losses,
        "timings_ms": {
            "identity": identity_ms,
            "model_load": model_load_ms,
            "source_prefill": source_prefill_ms,
            "target_prefill_and_trace": target_prefill_ms,
            "objective": objective_ms,
            "total": _elapsed_ms(started),
        },
        "checks": {
            "tokenizer_compatible": True,
            "shape_contract_passed": shape_contract_passed,
            "all_losses_finite": all_losses_finite,
        },
    }
    errors = smoke_report_errors(report)
    if errors:
        raise RealModelSmokeError("; ".join(errors))
    return report


def smoke_report_errors(report: dict[str, Any]) -> list[str]:
    """Validate a smoke report while denying it any publication authority."""

    errors: list[str] = []
    if report.get("schema_version") != REAL_MODEL_SMOKE_SCHEMA_VERSION:
        errors.append("unsupported real-model smoke schema")
    if report.get("authority") != REAL_MODEL_SMOKE_AUTHORITY:
        errors.append("real-model smoke authority must remain diagnostic_only")
    if report.get("approval_granted") is not False:
        errors.append("real-model smoke cannot grant approval")
    if report.get("evidence_eligible") is not False:
        errors.append("real-model smoke cannot be publication evidence")
    if report.get("sealed_split_accessed") is not False:
        errors.append("real-model smoke cannot access the sealed split")
    try:
        source = CachedKVModelSpec(**report["source"])
        target = CachedKVModelSpec(**report["target"])
        errors.extend(f"source: {item}" for item in source.validate())
        errors.extend(f"target: {item}" for item in target.validate())
        if source.tokenizer_sha256 != target.tokenizer_sha256:
            errors.append("smoke model tokenizers differ")
        shapes = report["shapes"]
        token_count = int(report["input"]["token_count"])
        query_count = int(report["transport"]["query_sample_count"])
        key_count = int(report["transport"]["key_sample_count"])
        expected_transformed = [
            2,
            target.num_layers,
            target.num_key_value_heads,
            key_count,
            target.head_dim,
        ]
        if shapes.get("transformed_kv") != expected_transformed:
            errors.append("smoke transformed KV shape is invalid")
        if shapes.get("source_kv") != [
            2,
            source.num_layers,
            source.num_key_value_heads,
            token_count,
            source.head_dim,
        ]:
            errors.append("smoke source KV shape is invalid")
        if shapes.get("target_kv") != [
            2,
            target.num_layers,
            target.num_key_value_heads,
            token_count,
            target.head_dim,
        ]:
            errors.append("smoke target KV shape is invalid")
        query_shape = shapes.get("target_query", [])
        if (
            len(query_shape) != 4
            or query_shape[0] != target.num_layers
            or query_shape[1] <= 0
            or query_shape[1] % target.num_key_value_heads
            or query_shape[2:] != [query_count, target.head_dim]
        ):
            errors.append("smoke target query shape is invalid")
        if shapes.get("native_attention_output") != query_shape:
            errors.append("smoke native attention output shape is invalid")
    except (KeyError, TypeError, ValueError) as exc:
        errors.append(f"smoke report structure is invalid: {exc}")
    losses = report.get("losses")
    if not isinstance(losses, dict):
        errors.append("smoke losses are missing")
    else:
        resolved_losses: dict[str, float] = {}
        for name in _LOSS_NAMES:
            try:
                value = float(losses[name])
            except (KeyError, TypeError, ValueError):
                errors.append(f"smoke loss {name} is invalid")
                continue
            if not math.isfinite(value) or value < 0:
                errors.append(f"smoke loss {name} must be finite and non-negative")
            resolved_losses[name] = value
        if len(resolved_losses) == len(_LOSS_NAMES) and all(
            math.isfinite(value) for value in resolved_losses.values()
        ):
            expected_total = (
                resolved_losses["native_generation"]
                + 0.25 * resolved_losses["prompt_tail_distillation"]
                + 0.5 * resolved_losses["attention_logit_kl"]
                + 0.5 * resolved_losses["attention_output_mse"]
                + 0.1 * resolved_losses["transformed_kv_anchor"]
            )
            if not math.isclose(
                resolved_losses["total"],
                expected_total,
                rel_tol=1e-6,
                abs_tol=1e-7,
            ):
                errors.append("smoke total loss is inconsistent with the frozen contract")
    checks = report.get("checks")
    if not isinstance(checks, dict):
        errors.append("smoke checks are missing")
    else:
        for name in ("tokenizer_compatible", "shape_contract_passed", "all_losses_finite"):
            if checks.get(name) is not True:
                errors.append(f"smoke check {name} did not pass")
    if report.get("status") != "passed":
        errors.append("real-model smoke status is not passed")
    return errors


def write_smoke_report(
    path: str | Path,
    report: dict[str, Any],
    *,
    overwrite: bool = False,
) -> Path:
    """Atomically write a validated diagnostic report."""

    errors = smoke_report_errors(report)
    if errors:
        raise RealModelSmokeError("; ".join(errors))
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(
            json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        if overwrite:
            temporary.replace(output)
        else:
            os.link(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    return output


def _validate_run_arguments(
    *,
    prompt: str,
    max_tokens: int,
    max_queries: int,
    max_keys: int,
) -> None:
    if not prompt.strip():
        raise RealModelSmokeError("smoke prompt cannot be empty")
    for name, value in (
        ("max_tokens", max_tokens),
        ("max_queries", max_queries),
        ("max_keys", max_keys),
    ):
        if value <= 0:
            raise RealModelSmokeError(f"{name} must be positive")


def _torch_dtype(name: str) -> Any:
    import torch

    values = {"bfloat16": torch.bfloat16, "float16": torch.float16}
    try:
        return values[name]
    except KeyError as exc:
        raise RealModelSmokeError(f"unsupported smoke dtype: {name}") from exc


def _synchronize(device: str) -> None:
    import torch

    parsed = torch.device(device)
    if parsed.type == "cuda":
        torch.cuda.synchronize(parsed)


def _device_name(device: str) -> str:
    import torch

    parsed = torch.device(device)
    if parsed.type != "cuda":
        return parsed.type
    return torch.cuda.get_device_name(parsed)


def _elapsed_ms(started: float) -> float:
    return (time.perf_counter() - started) * 1000.0


def _text_sha256(value: str) -> str:
    import hashlib

    return hashlib.sha256(value.encode("utf-8")).hexdigest()
