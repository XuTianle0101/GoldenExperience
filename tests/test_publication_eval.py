import pytest

from goldenexperience.benchmarks.publication_eval import (
    PublicationEvaluationError,
    publication_pass_threshold,
    score_publication_prediction,
    validate_publication_evaluation,
)


@pytest.mark.parametrize(
    ("prediction", "reference", "evaluation", "expected"),
    [
        ("The Eiffel Tower!", "eiffel tower", {"metric": "exact_match"}, 1.0),
        ("Final answer: 42", "42", {"metric": "contains"}, 1.0),
        ("red blue", "red green", {"metric": "token_f1"}, 0.5),
        (
            "therefore 1,000.001",
            1000,
            {"metric": "numeric_exact", "absolute_tolerance": 0.01},
            1.0,
        ),
        (
            'tool call: {"name":"search","arguments":{"q":"kv"}} done',
            {"name": "search", "arguments": {"q": "kv"}},
            {"metric": "json_exact"},
            1.0,
        ),
    ],
)
def test_publication_scorers_are_deterministic(
    prediction,
    reference,
    evaluation,
    expected,
) -> None:
    assert score_publication_prediction(prediction, reference, evaluation) == pytest.approx(
        expected
    )


def test_publication_scorer_accepts_multiple_references() -> None:
    assert (
        score_publication_prediction(
            "Answer: Paris",
            ["London", "Paris"],
            {"metric": "contains"},
        )
        == 1.0
    )


def test_publication_python_tests_run_in_restricted_worker() -> None:
    reference = {
        "entry_point": "add",
        "test_code": "def check(candidate):\n    assert candidate(2, 3) == 5",
        "test_mode": "check",
    }

    assert (
        score_publication_prediction(
            "```python\ndef add(left, right):\n    return left + right\n```",
            reference,
            {"metric": "python_tests"},
        )
        == 1.0
    )
    assert (
        score_publication_prediction(
            "def add(left, right):\n    return left - right",
            reference,
            {"metric": "python_tests"},
        )
        == 0.0
    )
    assert (
        score_publication_prediction(
            "def add(left, right):\n    return open('/etc/passwd').read()",
            reference,
            {"metric": "python_tests"},
        )
        == 0.0
    )


def test_publication_evaluation_contract_rejects_implicit_or_unknown_behavior() -> None:
    assert validate_publication_evaluation("answer", {})
    assert validate_publication_evaluation(
        "answer",
        {"metric": "exact_match", "unregistered_normalizer": True},
    )
    assert validate_publication_evaluation(
        "not numeric",
        {"metric": "numeric_exact"},
    )
    with pytest.raises(PublicationEvaluationError, match="unsupported"):
        score_publication_prediction("x", "x", {"metric": "bleu"})


def test_publication_pass_threshold_is_bounded_and_not_boolean() -> None:
    assert publication_pass_threshold({"metric": "token_f1", "pass_threshold": 0.5}) == 0.5
    with pytest.raises(PublicationEvaluationError, match="numeric"):
        publication_pass_threshold({"metric": "exact_match", "pass_threshold": True})
    with pytest.raises(PublicationEvaluationError, match="between"):
        publication_pass_threshold({"metric": "exact_match", "pass_threshold": 1.1})
