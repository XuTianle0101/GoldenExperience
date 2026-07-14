from __future__ import annotations

import hashlib
from types import SimpleNamespace

import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, Qwen3Config, Qwen3ForCausalLM

import goldenexperience.size_variant.v5_generation as generation_module
from goldenexperience.size_variant.cached_kv_manifest import CachedKVModelSpec
from goldenexperience.size_variant.v5_collect import (
    RawBenchmarkSample,
    TraceObjectRef,
    TraceRecord,
)
from goldenexperience.size_variant.v5_generation import (
    GenerationSupervisionSpec,
    TargetLogitGenerationBackend,
    TraceConstantGenerationBackend,
    batched_head_object_to_dynamic_cache,
    bound_suffix_token_ids,
    generation_distillation_losses,
    prepare_native_teacher,
)
from goldenexperience.size_variant.v5_pipeline import V5PipelineError


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


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
    assert GenerationSupervisionSpec().teacher_tokens == 16
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


def test_generation_supervision_rejects_malformed_contracts() -> None:
    assert GenerationSupervisionSpec.legacy().validate(require_registered=False) == []
    assert "target-logit" in GenerationSupervisionSpec.legacy().validate()[0]
    assert GenerationSupervisionSpec(supervision_id="unknown").validate() == [
        "generation supervision method is unsupported"
    ]
    errors = GenerationSupervisionSpec(
        teacher_tokens=0,
        max_suffix_tokens=3,
        truncation="tail",
        teacher_cache_dtype="float32",
    ).validate(require_registered=False)
    assert len(errors) == 4


def test_generation_tensor_contracts_fail_closed() -> None:
    model = _tiny_model()
    spec = GenerationSupervisionSpec(teacher_tokens=2, max_suffix_tokens=8)
    with pytest.raises(V5PipelineError, match="suffix is empty"):
        bound_suffix_token_ids(
            [],
            prefix_token_count=10,
            max_position_embeddings=256,
            spec=spec,
        )
    with pytest.raises(ValueError, match="batched KV"):
        batched_head_object_to_dynamic_cache(torch.zeros(2, 2), model.config)
    with pytest.raises(V5PipelineError, match="KV layout"):
        prepare_native_teacher(
            model,
            torch.zeros(2, 2),
            bound_suffix_token_ids(
                [1, 2],
                prefix_token_count=10,
                max_position_embeddings=256,
                spec=spec,
            ),
            prefix_token_count=10,
            spec=spec,
            device="cpu",
        )
    backend = TraceConstantGenerationBackend()
    with pytest.raises(V5PipelineError, match="constants"):
        backend.losses(
            None,  # type: ignore[arg-type]
            {"constant_losses": torch.tensor([float("nan"), 0.0])},
            torch.zeros(1, 2, 1, 1, 1, 1),
        )


def test_target_logit_backend_loads_once_and_caches_teacher(
    tmp_path,
    monkeypatch,
) -> None:
    model = _tiny_model()

    class Tokenizer:
        def __init__(self) -> None:
            self.calls = 0

        def __call__(self, _text, **_kwargs):
            self.calls += 1
            return SimpleNamespace(input_ids=torch.tensor([[5, 6, 7]]))

    tokenizer = Tokenizer()
    monkeypatch.setattr(generation_module, "verify_model_path", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(
        AutoTokenizer,
        "from_pretrained",
        lambda *_args, **_kwargs: tokenizer,
    )
    monkeypatch.setattr(
        AutoModelForCausalLM,
        "from_pretrained",
        lambda *_args, **_kwargs: model,
    )
    target = CachedKVModelSpec(
        model_id="tiny-target",
        parameter_count_b=0.0,
        revision="test",
        architecture="qwen3",
        config_sha256=_digest("config"),
        tokenizer_sha256=_digest("tokenizer"),
        weights_sha256=_digest("weights"),
        num_layers=2,
        num_key_value_heads=2,
        head_dim=8,
        dtype="bfloat16",
        rope_theta=1_000_000.0,
        max_position_embeddings=256,
        chat_template_sha256=_digest("chat"),
    )
    record = TraceRecord(
        sample_id="sample",
        prefix_group_id="group",
        dataset_id="synthetic",
        task="qa",
        token_bucket=128,
        content_sha256=_digest("content"),
        prefix_sha256=_digest("prefix"),
        suffix_query_sha256=_digest("suffix"),
        token_ids_sha256=_digest("tokens"),
        token_count=10,
        query_sample_count=2,
        key_sample_count=4,
        shard=TraceObjectRef(
            _digest("shard"),
            f"objects/00/{_digest('shard')}.safetensors",
            1,
        ),
    )
    sample = RawBenchmarkSample(
        sample_id="sample",
        prefix_text="prefix",
        suffix_query="suffix",
        reference="answer",
        evaluation={"metric": "exact_match"},
        provenance={},
    )
    native = torch.randn(2, 2, 2, 4, 8)
    candidates = native.unsqueeze(0).repeat(2, 1, 1, 1, 1, 1).requires_grad_(True)
    backend = TargetLogitGenerationBackend(
        target_path=tmp_path / "target",
        target=target,
        samples={"sample": sample},
        device="cpu",
        identity_cache_path=None,
        spec=GenerationSupervisionSpec(),
    )

    assert backend.parameters()["teacher_tokens"] == 16
    with backend:
        first = backend.losses(record, {"target_kv": native}, candidates)
        second = backend.losses(record, {"target_kv": native}, candidates)
        (first[0].sum() + first[1].sum()).backward()
        assert len(backend.teacher_cache) == 1
    assert tokenizer.calls == 1
    torch.testing.assert_close(first[0], second[0])
    torch.testing.assert_close(first[1], second[1])
    assert candidates.grad is not None
    assert torch.isfinite(candidates.grad).all()
    assert backend.teacher_cache == {}
