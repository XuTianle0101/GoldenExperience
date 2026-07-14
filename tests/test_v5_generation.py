from __future__ import annotations

import pytest
import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

from goldenexperience.size_variant.v5_generation import (
    GenerationSupervisionSpec,
    TraceConstantGenerationBackend,
    batched_head_object_to_dynamic_cache,
    bound_suffix_token_ids,
    generation_distillation_losses,
    prepare_native_teacher,
)
from goldenexperience.size_variant.v5_pipeline import V5PipelineError


def _tiny_model() -> Qwen3ForCausalLM:
    torch.manual_seed(7)
    config = Qwen3Config(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=256,
    )
    return Qwen3ForCausalLM(config).eval().requires_grad_(False)


def test_suffix_bound_preserves_absolute_head_and_tail_positions() -> None:
    spec = GenerationSupervisionSpec(teacher_tokens=2, max_suffix_tokens=6)

    suffix = bound_suffix_token_ids(
        torch.arange(10),
        prefix_token_count=100,
        max_position_embeddings=112,
        spec=spec,
    )

    assert suffix.input_ids.tolist() == [0, 1, 2, 7, 8, 9]
    assert suffix.position_ids.tolist() == [100, 101, 102, 107, 108, 109]
    assert suffix.full_token_count == 10
    with pytest.raises(V5PipelineError, match="position contract"):
        bound_suffix_token_ids(
            torch.arange(10),
            prefix_token_count=101,
            max_position_embeddings=112,
            spec=spec,
        )


def test_batched_dynamic_cache_preserves_candidate_gradients() -> None:
    model = _tiny_model()
    kv = torch.randn(2, 2, 2, 2, 4, 8, requires_grad=True)

    cache = batched_head_object_to_dynamic_cache(kv, model.config)
    output = model(
        input_ids=torch.tensor([[3, 4], [3, 4]]),
        position_ids=torch.tensor([[10, 11], [10, 11]]),
        past_key_values=cache,
        use_cache=False,
        logits_to_keep=1,
    )
    output.logits.float().square().mean().backward()

    assert kv.grad is not None
    assert torch.isfinite(kv.grad).all()
    assert kv.grad.norm() > 0


def test_native_teacher_distillation_is_finite_and_differentiable() -> None:
    model = _tiny_model()
    spec = GenerationSupervisionSpec(teacher_tokens=2, max_suffix_tokens=8)
    native = torch.randn(2, 2, 2, 4, 8)
    suffix = bound_suffix_token_ids(
        torch.tensor([5, 6, 7]),
        prefix_token_count=10,
        max_position_embeddings=256,
        spec=spec,
    )
    teacher = prepare_native_teacher(
        model,
        native,
        suffix,
        prefix_token_count=10,
        spec=spec,
        device="cpu",
    )
    candidate = native.unsqueeze(0).repeat(2, 1, 1, 1, 1, 1).requires_grad_(True)

    generation, distillation = generation_distillation_losses(
        model,
        candidate,
        teacher,
        device="cpu",
    )
    (generation.sum() + distillation.sum()).backward()

    assert teacher.teacher_tokens.shape == (2,)
    assert teacher.teacher_logits.shape == (2, 64)
    assert generation.shape == distillation.shape == (2,)
    assert torch.isfinite(generation).all()
    assert torch.isfinite(distillation).all()
    assert distillation.max() < 1e-4
    assert candidate.grad is not None
    assert torch.isfinite(candidate.grad).all()
    assert candidate.grad.norm() > 0


def test_trace_constant_backend_expands_legacy_values() -> None:
    backend = TraceConstantGenerationBackend()
    transformed = torch.zeros(3, 2, 1, 1, 1, 1)

    generation, distillation = backend.losses(
        None,  # type: ignore[arg-type]
        {"constant_losses": torch.tensor([1.25, 0.5])},
        transformed,
    )

    assert generation.tolist() == [1.25, 1.25, 1.25]
    assert distillation.tolist() == [0.5, 0.5, 0.5]
