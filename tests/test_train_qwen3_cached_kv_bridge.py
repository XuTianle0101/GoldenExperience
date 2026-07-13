import pytest
import torch

from goldenexperience.size_variant.cached_kv_dataset import CachedKVPrompt
from scripts.train_qwen3_cached_kv_bridge import (
    _bucket_balanced_samples,
    _kv_anchor_losses,
    _parameter_anchor_loss,
    build_parser,
)


def _prompt(prompt_id: str, bucket: int) -> CachedKVPrompt:
    return CachedKVPrompt(
        prompt_id=prompt_id,
        split="train",
        category="test",
        group_id=prompt_id,
        token_bucket=bucket,
        template="Prompt: {context}",
        context_seed=prompt_id,
    )


def test_logit_refinement_samples_round_robin_across_token_buckets() -> None:
    samples = tuple(
        _prompt(f"bucket-{bucket}-{index}", bucket)
        for bucket in (32, 128, 512, 2048)
        for index in range(4)
    )

    selected = _bucket_balanced_samples(samples, 6)

    assert [sample.token_bucket for sample in selected] == [32, 128, 512, 2048, 32, 128]
    assert [sample.prompt_id for sample in selected] == [
        "bucket-32-0",
        "bucket-128-1",
        "bucket-512-2",
        "bucket-2048-3",
        "bucket-32-1",
        "bucket-128-2",
    ]
    assert _bucket_balanced_samples(samples, 20) == (
        samples[0],
        samples[5],
        samples[10],
        samples[15],
        samples[1],
        samples[6],
        samples[11],
        samples[12],
        samples[2],
        samples[7],
        samples[8],
        samples[13],
        samples[3],
        samples[4],
        samples[9],
        samples[14],
    )


def test_logit_refinement_anchor_losses_are_scale_normalized() -> None:
    anchors = {
        "small": torch.full((4,), 0.01),
        "large": torch.full((4,), 100.0),
    }
    state = {
        "small": anchors["small"] * 1.1,
        "large": anchors["large"] * 1.1,
    }

    parameter_loss = _parameter_anchor_loss(state, anchors, ("small", "large"))
    identical_kv_loss, identical_mse, identical_cosine = _kv_anchor_losses(
        torch.ones(2, 2, 3, 4),
        torch.ones(2, 2, 3, 4),
    )
    changed_kv_loss, changed_mse, changed_cosine = _kv_anchor_losses(
        torch.ones(2, 2, 3, 4) * 1.1,
        torch.ones(2, 2, 3, 4),
    )

    assert parameter_loss.item() == pytest.approx(0.01)
    assert identical_kv_loss.item() == pytest.approx(0.0)
    assert identical_mse.item() == pytest.approx(0.0)
    assert identical_cosine.item() == pytest.approx(0.0)
    assert changed_kv_loss > identical_kv_loss
    assert changed_mse.item() == pytest.approx(0.01)
    assert changed_cosine.item() == pytest.approx(0.0, abs=1e-6)


@pytest.mark.parametrize(
    ("group", "expected"),
    [
        ("bias-only", "bias-only"),
        ("nonlinear-up-only", "nonlinear-up-only"),
    ],
)
def test_logit_refinement_parameter_groups_are_exposed_by_cli(
    group: str,
    expected: str,
) -> None:
    args = build_parser().parse_args(
        [
            "--direction",
            "8b_to_14b",
            "--dataset",
            "prompts.json",
            "--output",
            "candidate.json",
            "--logit-refinement-parameter-group",
            group,
        ]
    )

    assert args.logit_refinement_parameter_group == expected


def test_logit_refinement_cli_defaults_fail_closed_against_collapse() -> None:
    args = build_parser().parse_args(
        [
            "--direction",
            "8b_to_14b",
            "--dataset",
            "prompts.json",
            "--output",
            "candidate.json",
        ]
    )

    assert args.logit_refinement_learning_rate == pytest.approx(1e-5)
    assert args.logit_refinement_parameter_group == "bias-only"
    assert args.logit_refinement_anchor_weight == pytest.approx(0.1)
    assert args.logit_refinement_kv_anchor_weight == pytest.approx(1.0)
    assert args.logit_refinement_holdout_prompts == 4
    assert args.logit_refinement_early_stopping_patience == 2
    assert args.seed == 17
    assert not args.paired_refinement_validation


def test_paired_refinement_validation_is_exposed_by_cli() -> None:
    args = build_parser().parse_args(
        [
            "--direction",
            "8b_to_14b",
            "--dataset",
            "prompts.json",
            "--output",
            "candidate.json",
            "--logit-refinement-steps",
            "8",
            "--paired-refinement-validation",
            "--seed",
            "31",
        ]
    )

    assert args.paired_refinement_validation
    assert args.seed == 31
