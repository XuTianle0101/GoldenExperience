from dataclasses import replace

import pytest
import torch

from goldenexperience.size_variant.cached_kv_bridge import _apply_rope_flat
from goldenexperience.size_variant.cached_kv_dataset import (
    CachedKVPrompt,
    CachedKVPromptDataset,
    contains_expected_final_answer,
    render_to_token_bucket,
)
from goldenexperience.size_variant.cached_kv_manifest import (
    CachedKVQualityEvidence,
    CachedKVQualityThresholds,
)
from goldenexperience.size_variant.cached_kv_training import (
    build_cka_source_layer_plan,
    build_source_layer_plan,
    build_training_matrices,
    fit_low_rank_state,
    logit_distillation_loss,
    transform_with_state,
)


def _prompt(
    prompt_id: str,
    split: str,
    *,
    group_id: str | None = None,
    template: str | None = None,
    bucket: int = 32,
) -> CachedKVPrompt:
    return CachedKVPrompt(
        prompt_id=prompt_id,
        split=split,
        category="test",
        group_id=group_id or f"{split}-group",
        token_bucket=bucket,
        template=template or f"Prompt {prompt_id}: {{context}} End.",
        context_seed=f"seed-{prompt_id}",
    )


def test_prompt_dataset_rejects_id_content_and_group_leakage() -> None:
    valid = CachedKVPromptDataset(
        samples=(
            _prompt("train-a", "train"),
            _prompt("validation-a", "validation"),
            _prompt("test-a", "test"),
        )
    )
    assert valid.validate() == []
    assert len({valid.split_sha256(name) for name in ("train", "validation", "test")}) == 3

    duplicate_id = replace(
        valid,
        samples=valid.samples + (_prompt("train-a", "test"),),
    )
    assert any("duplicate prompt_id" in error for error in duplicate_id.validate())

    duplicate_content = replace(
        valid,
        samples=valid.samples
        + (
            replace(
                valid.samples[0],
                prompt_id="test-copy",
                split="test",
                group_id="test-copy-group",
            ),
        ),
    )
    assert any("content is duplicated" in error for error in duplicate_content.validate())

    relabeled_content = replace(
        valid,
        samples=valid.samples
        + (
            replace(
                valid.samples[0],
                prompt_id="test-relabeled-copy",
                split="test",
                category="different-label",
                group_id="test-relabeled-group",
                expected_answer="different-answer",
            ),
        ),
    )
    assert any("content is duplicated" in error for error in relabeled_content.validate())

    group_leak = replace(
        valid,
        samples=valid.samples + (_prompt("test-group-copy", "test", group_id="train-group"),),
    )
    assert any("crosses train and test" in error for error in group_leak.validate())
    approval_errors = valid.approval_errors()
    assert "train split has too few prompts" in approval_errors
    assert "validation split has too few prompts" in approval_errors
    assert "test split has too few prompts" in approval_errors


class _WhitespaceTokenizer:
    def __call__(self, text: str, **_: object) -> dict[str, list[int]]:
        return {"input_ids": list(range(len(text.split())))}


class _ChatWhitespaceTokenizer(_WhitespaceTokenizer):
    def apply_chat_template(self, messages, **kwargs):
        assert kwargs == {
            "tokenize": False,
            "add_generation_prompt": True,
            "enable_thinking": False,
        }
        return f"<user> {messages[0]['content']} </user> <assistant>"


def test_prompt_rendering_reaches_declared_bucket_deterministically() -> None:
    sample = _prompt("bucket", "train", bucket=128)

    first_text, first_ids = render_to_token_bucket(
        sample,
        _WhitespaceTokenizer(),
        suffix_tokens=16,
    )
    second_text, second_ids = render_to_token_bucket(
        sample,
        _WhitespaceTokenizer(),
        suffix_tokens=16,
    )

    assert len(first_ids) >= 145
    assert first_text == second_text
    assert first_ids == second_ids


def test_prompt_rendering_uses_non_thinking_chat_template_and_keeps_task_tail() -> None:
    sample = _prompt(
        "chat-bucket",
        "validation",
        bucket=128,
        template="Notes: {context} Return exactly `Final answer: 42`.",
    )

    text, token_ids = render_to_token_bucket(
        sample,
        _ChatWhitespaceTokenizer(),
        suffix_tokens=16,
    )

    assert text.startswith("<user>")
    assert text.endswith("</user> <assistant>")
    assert "Final answer: 42" in text
    assert len(token_ids) >= 145


def test_final_answer_assertion_requires_an_explicit_bounded_answer() -> None:
    assert contains_expected_final_answer("Reasoning. Final answer: 6", "6")
    assert contains_expected_final_answer("**Final answer: `ACK-1003`**", "ACK-1003")
    assert not contains_expected_final_answer("The intermediate value is 6.", "6")
    assert not contains_expected_final_answer("Final answer: 60", "6")


@pytest.mark.parametrize("source_layers,target_layers", [(36, 40), (40, 36)])
def test_source_layer_plan_covers_both_qwen3_directions(
    source_layers: int,
    target_layers: int,
) -> None:
    layer_ids, weights = build_source_layer_plan(source_layers, target_layers, 3)

    assert tuple(layer_ids.shape) == (target_layers, 3)
    assert int(layer_ids.min()) >= 0
    assert int(layer_ids.max()) < source_layers
    torch.testing.assert_close(weights.sum(dim=-1), torch.ones(target_layers))


@pytest.mark.parametrize(
    "source_layers,mapping",
    [
        (4, (0, 1, 1, 2, 3)),
        (5, (0, 2, 2, 4)),
    ],
)
def test_cka_source_layer_plan_recovers_monotonic_rotated_layers(
    source_layers: int,
    mapping: tuple[int, ...],
) -> None:
    torch.manual_seed(23)
    samples = 128
    width = 8
    positions = torch.arange(samples)
    source_key_unrotated = torch.randn(source_layers, samples, width)
    source_value = torch.randn(source_layers, samples, width)
    target_keys: list[torch.Tensor] = []
    target_values: list[torch.Tensor] = []
    for source_layer in mapping:
        key_rotation, _ = torch.linalg.qr(torch.randn(width, width))
        value_rotation, _ = torch.linalg.qr(torch.randn(width, width))
        target_keys.append(source_key_unrotated[source_layer] @ key_rotation)
        target_values.append(source_value[source_layer] @ value_rotation)
    target_key_unrotated = torch.stack(target_keys)
    target_value = torch.stack(target_values)
    source_key = _apply_rope_flat(
        source_key_unrotated,
        positions,
        num_heads=1,
        head_dim=width,
        theta=1_000_000,
        inverse=False,
    )
    target_key = _apply_rope_flat(
        target_key_unrotated,
        positions,
        num_heads=1,
        head_dim=width,
        theta=1_000_000,
        inverse=False,
    )

    layer_ids, layer_weights, evidence = build_cka_source_layer_plan(
        torch.stack((source_key, source_value)),
        torch.stack((target_key, target_value)),
        positions,
        3,
        source_heads=1,
        source_head_dim=width,
        source_rope_theta=1_000_000,
        target_heads=1,
        target_head_dim=width,
        target_rope_theta=1_000_000,
        device="cpu",
    )

    selected = (layer_ids * layer_weights.long()).sum(dim=1)
    assert selected.tolist() == list(mapping)
    assert evidence["matched_source_layers"] == list(mapping)
    assert evidence["combined_score"] > 0.99
    torch.testing.assert_close(layer_weights.sum(dim=1), torch.ones(len(mapping)))


def test_supervised_low_rank_fit_reconstructs_synthetic_cached_kv() -> None:
    torch.manual_seed(7)
    source_layers = 3
    target_layers = 4
    token_count = 96
    width = 4
    rank = 2
    positions = torch.arange(token_count)
    source = torch.randn(2, source_layers, token_count, width)
    layer_ids, layer_weights = build_source_layer_plan(source_layers, target_layers, 2)
    source_unrotated = _apply_rope_flat(
        source[0],
        positions,
        num_heads=1,
        head_dim=4,
        theta=1_000_000,
        inverse=True,
    )
    selected_key = source_unrotated[layer_ids]
    selected_value = source[1][layer_ids]
    base_key = torch.einsum("ls,lstw->ltw", layer_weights, selected_key)
    base_value = torch.einsum("ls,lstw->ltw", layer_weights, selected_value)
    features = torch.cat(
        (
            selected_key.permute(0, 2, 1, 3).reshape(target_layers, token_count, -1),
            selected_value.permute(0, 2, 1, 3).reshape(target_layers, token_count, -1),
        ),
        dim=-1,
    )
    key_down = torch.randn(target_layers, features.shape[-1], rank) * 0.1
    key_up = torch.randn(target_layers, rank, width) * 0.1
    value_down = torch.randn(target_layers, features.shape[-1], rank) * 0.1
    value_up = torch.randn(target_layers, rank, width) * 0.1
    target_key_unrotated = base_key + torch.bmm(torch.bmm(features, key_down), key_up)
    target_value = base_value + torch.bmm(torch.bmm(features, value_down), value_up)
    target_key = _apply_rope_flat(
        target_key_unrotated,
        positions,
        num_heads=1,
        head_dim=4,
        theta=1_000_000,
        inverse=False,
    )
    target = torch.stack((target_key, target_value))
    train_x, train_key, train_value = build_training_matrices(
        source,
        target,
        positions,
        layer_ids,
        layer_weights,
        source_heads=1,
        source_head_dim=4,
        source_rope_theta=1_000_000,
        target_heads=1,
        target_head_dim=4,
        target_rope_theta=1_000_000,
    )
    state = fit_low_rank_state(
        train_x,
        train_key,
        train_value,
        layer_ids,
        layer_weights,
        rank=rank,
        ridge_lambda=1e-5,
        device="cpu",
    )

    reconstructed = transform_with_state(
        source,
        positions,
        state,
        source_heads=1,
        source_head_dim=4,
        source_rope_theta=1_000_000,
        target_heads=1,
        target_head_dim=4,
        target_rope_theta=1_000_000,
        device="cpu",
    )

    torch.testing.assert_close(reconstructed, target, atol=2e-3, rtol=2e-3)


def test_logit_distillation_prefers_teacher_logits_and_detaches_teacher() -> None:
    torch.manual_seed(29)
    teacher = torch.randn(1, 4, 11, requires_grad=True)
    labels = teacher.detach().argmax(dim=-1)
    matching = teacher.detach().clone().requires_grad_(True)
    mismatched = torch.zeros_like(teacher, requires_grad=True)

    matching_loss, matching_distillation, _ = logit_distillation_loss(
        matching,
        teacher,
        labels,
        temperature=1.5,
        label_weight=0.0,
    )
    mismatched_loss, mismatched_distillation, _ = logit_distillation_loss(
        mismatched,
        teacher,
        labels,
        temperature=1.5,
        label_weight=0.0,
    )
    mismatched_loss.backward()

    assert matching_loss.item() == pytest.approx(0.0, abs=1e-6)
    assert matching_distillation.item() == pytest.approx(0.0, abs=1e-6)
    assert mismatched_distillation > matching_distillation
    assert mismatched.grad is not None
    assert teacher.grad is None


def test_logit_distillation_rejects_invalid_contracts() -> None:
    logits = torch.zeros(1, 2, 3)
    labels = torch.zeros(1, 2, dtype=torch.long)

    with pytest.raises(ValueError, match="share"):
        logit_distillation_loss(
            logits,
            torch.zeros(1, 3, 3),
            labels,
            temperature=1.0,
            label_weight=0.0,
        )
    with pytest.raises(ValueError, match="temperature"):
        logit_distillation_loss(
            logits,
            logits,
            labels,
            temperature=0.0,
            label_weight=0.0,
        )


def test_supervised_fit_learns_full_width_diagonal_baseline() -> None:
    torch.manual_seed(11)
    source = torch.randn(2, 2, 128, 4)
    positions = torch.arange(128)
    layer_ids, layer_weights = build_source_layer_plan(2, 2, 1)
    source_key = _apply_rope_flat(
        source[0],
        positions,
        num_heads=1,
        head_dim=4,
        theta=1_000_000,
        inverse=True,
    )
    selected_key = source_key[layer_ids].squeeze(1)
    selected_value = source[1][layer_ids].squeeze(1)
    key_scale = torch.tensor(((0.5, 1.5, -0.75, 2.0), (1.25, -0.5, 0.8, 1.8)))
    value_scale = torch.tensor(((1.7, 0.6, -1.2, 0.4), (-0.3, 1.4, 2.1, 0.9)))
    target_key = _apply_rope_flat(
        selected_key * key_scale.unsqueeze(1),
        positions,
        num_heads=1,
        head_dim=4,
        theta=1_000_000,
        inverse=False,
    )
    target = torch.stack((target_key, selected_value * value_scale.unsqueeze(1)))
    features, key_residual, value_residual = build_training_matrices(
        source,
        target,
        positions,
        layer_ids,
        layer_weights,
        source_heads=1,
        source_head_dim=4,
        source_rope_theta=1_000_000,
        target_heads=1,
        target_head_dim=4,
        target_rope_theta=1_000_000,
    )
    state = fit_low_rank_state(
        features,
        key_residual,
        value_residual,
        layer_ids,
        layer_weights,
        rank=1,
        ridge_lambda=1e-5,
        device="cpu",
    )

    reconstructed = transform_with_state(
        source,
        positions,
        state,
        source_heads=1,
        source_head_dim=4,
        source_rope_theta=1_000_000,
        target_heads=1,
        target_head_dim=4,
        target_rope_theta=1_000_000,
        device="cpu",
    )

    torch.testing.assert_close(reconstructed, target, atol=2e-3, rtol=2e-3)


def test_supervised_fit_uses_silu_correction_for_nonlinear_cached_kv() -> None:
    import torch.nn.functional as functional

    torch.manual_seed(19)
    source = torch.randn(2, 2, 256, 4)
    positions = torch.arange(256)
    layer_ids, layer_weights = build_source_layer_plan(2, 2, 1)
    source_key = _apply_rope_flat(
        source[0],
        positions,
        num_heads=1,
        head_dim=4,
        theta=1_000_000,
        inverse=True,
    )
    selected_key = source_key[layer_ids].squeeze(1)
    selected_value = source[1][layer_ids].squeeze(1)
    features = torch.cat((selected_key, selected_value), dim=-1)
    key_down = torch.randn(2, 8, 2) * 0.3
    value_down = torch.randn(2, 8, 2) * 0.3
    key_latent = torch.bmm(features, key_down)
    value_latent = torch.bmm(features, value_down)
    key_scale = key_latent.square().mean(dim=1).sqrt()
    value_scale = value_latent.square().mean(dim=1).sqrt()
    target_key_unrotated = selected_key + torch.bmm(
        key_latent,
        torch.randn(2, 2, 4) * 0.05,
    )
    target_key_unrotated += torch.bmm(
        functional.silu(key_latent / key_scale.unsqueeze(1)),
        torch.randn(2, 2, 4) * 0.4,
    )
    target_value = selected_value + torch.bmm(
        value_latent,
        torch.randn(2, 2, 4) * 0.05,
    )
    target_value += torch.bmm(
        functional.silu(value_latent / value_scale.unsqueeze(1)),
        torch.randn(2, 2, 4) * 0.4,
    )
    target_key = _apply_rope_flat(
        target_key_unrotated,
        positions,
        num_heads=1,
        head_dim=4,
        theta=1_000_000,
        inverse=False,
    )
    target = torch.stack((target_key, target_value))
    train_x, key_residual, value_residual = build_training_matrices(
        source,
        target,
        positions,
        layer_ids,
        layer_weights,
        source_heads=1,
        source_head_dim=4,
        source_rope_theta=1_000_000,
        target_heads=1,
        target_head_dim=4,
        target_rope_theta=1_000_000,
    )
    state = fit_low_rank_state(
        train_x,
        key_residual,
        value_residual,
        layer_ids,
        layer_weights,
        rank=2,
        ridge_lambda=1e-3,
        nonlinear_ridge_lambda=1e-2,
        device="cpu",
    )
    without_nonlinear = dict(state)
    without_nonlinear["key_nonlinear_up"] = torch.zeros_like(state["key_nonlinear_up"])
    without_nonlinear["value_nonlinear_up"] = torch.zeros_like(state["value_nonlinear_up"])

    reconstructed = transform_with_state(
        source,
        positions,
        state,
        source_heads=1,
        source_head_dim=4,
        source_rope_theta=1_000_000,
        target_heads=1,
        target_head_dim=4,
        target_rope_theta=1_000_000,
        device="cpu",
    )
    linear_only = transform_with_state(
        source,
        positions,
        without_nonlinear,
        source_heads=1,
        source_head_dim=4,
        source_rope_theta=1_000_000,
        target_heads=1,
        target_head_dim=4,
        target_rope_theta=1_000_000,
        device="cpu",
    )

    nonlinear_mse = torch.mean((reconstructed - target).square())
    linear_mse = torch.mean((linear_only - target).square())
    assert nonlinear_mse < linear_mse * 0.9


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for BF16 regression")
def test_training_transform_uses_one_compute_dtype_on_cuda() -> None:
    source = torch.randn(2, 2, 8, 4, dtype=torch.bfloat16)
    positions = torch.arange(8)
    layer_ids, layer_weights = build_source_layer_plan(2, 2, 1)
    state = {
        "source_layer_ids": layer_ids,
        "source_layer_weights": layer_weights,
        "feature_mean": torch.zeros(2, 8),
        "key_base_scale": torch.ones(2, 4),
        "key_down": torch.zeros(2, 8, 1, dtype=torch.bfloat16),
        "key_nonlinear_mean": torch.zeros(2, 1),
        "key_nonlinear_scale": torch.ones(2, 1),
        "key_nonlinear_up": torch.zeros(2, 1, 4, dtype=torch.bfloat16),
        "key_up": torch.zeros(2, 1, 4, dtype=torch.bfloat16),
        "key_bias": torch.zeros(2, 4),
        "value_base_scale": torch.ones(2, 4),
        "value_down": torch.zeros(2, 8, 1, dtype=torch.bfloat16),
        "value_nonlinear_mean": torch.zeros(2, 1),
        "value_nonlinear_scale": torch.ones(2, 1),
        "value_nonlinear_up": torch.zeros(2, 1, 4, dtype=torch.bfloat16),
        "value_up": torch.zeros(2, 1, 4, dtype=torch.bfloat16),
        "value_bias": torch.zeros(2, 4),
    }

    transformed = transform_with_state(
        source,
        positions,
        state,
        source_heads=1,
        source_head_dim=4,
        source_rope_theta=1_000_000,
        target_heads=1,
        target_head_dim=4,
        target_rope_theta=1_000_000,
        device="cuda:0",
    )

    assert transformed.dtype == torch.bfloat16
    assert transformed.is_cuda


def test_quality_evidence_fails_closed_without_runtime_cost_measurement() -> None:
    quality = CachedKVQualityEvidence(
        evaluation_dataset_sha256="a" * 64,
        held_out_prompts=64,
        evaluated_tokens=4096,
        token_buckets=(32, 128, 512, 2048),
        key_cosine=0.99,
        value_cosine=0.99,
        next_token_top1_agreement=0.99,
        perplexity_drift_pct=1.0,
        task_prompts=64,
        native_task_score=0.99,
        bridge_task_score=0.99,
        task_score_drop_pct=0.5,
        greedy_continuation_match_rate=0.99,
        cost_report_sha256=None,
        cost_candidate_manifest_sha256=None,
        p95_source_read_transform_put_ms=None,
        p95_target_prefill_ms=None,
    )

    errors = quality.gate_errors(CachedKVQualityThresholds())

    assert "p95_source_read_transform_put_ms is required" in errors
    assert "p95_target_prefill_ms is required" in errors
    assert "cost_report_sha256 must be a SHA-256 digest" in errors
    assert "cost_candidate_manifest_sha256 must be a SHA-256 digest" in errors


def test_quality_evidence_rejects_an_invalid_native_task_baseline() -> None:
    quality = CachedKVQualityEvidence(
        evaluation_dataset_sha256="a" * 64,
        held_out_prompts=64,
        evaluated_tokens=4096,
        token_buckets=(32, 128, 512, 2048),
        key_cosine=0.99,
        value_cosine=0.99,
        next_token_top1_agreement=0.99,
        perplexity_drift_pct=1.0,
        task_prompts=64,
        native_task_score=0.0,
        bridge_task_score=0.0,
        task_score_drop_pct=0.0,
        greedy_continuation_match_rate=0.99,
        cost_report_sha256="b" * 64,
        cost_candidate_manifest_sha256="c" * 64,
        p95_source_read_transform_put_ms=10.0,
        p95_target_prefill_ms=20.0,
    )

    errors = quality.gate_errors(CachedKVQualityThresholds())

    assert "native task score is below threshold" in errors
    assert "bridge task score is below threshold" in errors
