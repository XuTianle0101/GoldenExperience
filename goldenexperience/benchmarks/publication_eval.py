"""Deterministic, data-declared semantic scorers for publication examples."""

from __future__ import annotations

import json
import math
import re
import string
import subprocess
import sys
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


class PublicationEvaluationError(ValueError):
    """Raised when an evaluation contract or prediction is malformed."""


SUPPORTED_PUBLICATION_METRICS = frozenset(
    {
        "contains",
        "exact_match",
        "function_call",
        "json_exact",
        "math_exact",
        "numeric_exact",
        "python_tests",
        "token_f1",
    }
)
_ARTICLES = re.compile(r"\b(a|an|the)\b", flags=re.IGNORECASE)
_NUMBER = re.compile(r"[-+]?(?:(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?|\.\d+)(?:[eE][-+]?\d+)?")


def score_publication_prediction(
    prediction: str,
    reference: Any,
    evaluation: Mapping[str, Any],
) -> float:
    """Score one generated answer under an explicit frozen evaluation contract."""

    if not isinstance(prediction, str):
        raise PublicationEvaluationError("publication prediction must be text")
    metric = evaluation.get("metric")
    if metric not in SUPPORTED_PUBLICATION_METRICS:
        raise PublicationEvaluationError(f"unsupported publication metric {metric!r}")
    references = _references(reference)
    if metric == "exact_match":
        score = max(
            float(_normalize_answer(prediction) == _normalize_answer(item)) for item in references
        )
    elif metric == "contains":
        normalized_prediction = _normalize_answer(prediction)
        score = max(float(_normalize_answer(item) in normalized_prediction) for item in references)
    elif metric == "token_f1":
        score = max(_token_f1(prediction, item) for item in references)
    elif metric == "numeric_exact":
        predicted = _last_number(prediction)
        tolerances = _numeric_tolerances(evaluation)
        score = max(_numeric_match(predicted, item, *tolerances) for item in references)
    elif metric == "math_exact":
        math_prediction = _last_boxed_value(prediction) or prediction
        score = max(
            float(_normalize_math_answer(math_prediction) == _normalize_math_answer(item))
            for item in references
        )
    elif metric == "function_call":
        score = _function_call_score(prediction, reference)
    elif metric == "python_tests":
        score = max(_python_test_score(prediction, item) for item in references)
    else:
        predicted_json = _first_json_value(prediction)
        score = max(float(predicted_json == _reference_json(item)) for item in references)
    if not math.isfinite(score) or not 0 <= score <= 1:
        raise PublicationEvaluationError("publication scorer produced an invalid result")
    return score


def publication_pass_threshold(evaluation: Mapping[str, Any]) -> float:
    value = evaluation.get("pass_threshold", 1.0)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PublicationEvaluationError("publication pass threshold must be numeric")
    threshold = float(value)
    if not math.isfinite(threshold) or not 0 <= threshold <= 1:
        raise PublicationEvaluationError("publication pass threshold must be between zero and one")
    return threshold


def validate_publication_evaluation(reference: Any, evaluation: Mapping[str, Any]) -> list[str]:
    try:
        metric = evaluation.get("metric")
        if metric not in SUPPORTED_PUBLICATION_METRICS:
            raise PublicationEvaluationError(f"unsupported publication metric {metric!r}")
        references = _references(reference)
        publication_pass_threshold(evaluation)
        if metric == "numeric_exact":
            _numeric_tolerances(evaluation)
            for item in references:
                _reference_number(item)
        elif metric == "json_exact":
            for item in references:
                _reference_json(item)
        elif metric == "python_tests":
            if len(references) != 1:
                raise PublicationEvaluationError(
                    "python test evaluation requires exactly one test contract"
                )
            _python_test_contract(references[0])
        elif metric == "math_exact":
            for item in references:
                if not _normalize_math_answer(item):
                    raise PublicationEvaluationError("math reference is empty after normalization")
        elif metric == "function_call":
            _function_call_contract(reference)
        allowed_fields = {"metric", "pass_threshold"}
        if metric == "numeric_exact":
            allowed_fields.update({"absolute_tolerance", "relative_tolerance"})
        unknown = set(evaluation) - allowed_fields
        if unknown:
            raise PublicationEvaluationError(
                f"publication evaluation has unknown fields: {sorted(unknown)}"
            )
    except (PublicationEvaluationError, TypeError, ValueError) as exc:
        return [str(exc)]
    return []


def _references(reference: Any) -> tuple[Any, ...]:
    if isinstance(reference, (str, int, float, Mapping)):
        values = (reference,)
    elif isinstance(reference, Sequence) and not isinstance(reference, (bytes, bytearray)):
        values = tuple(reference)
    else:
        raise PublicationEvaluationError("publication reference has an unsupported type")
    if not values:
        raise PublicationEvaluationError("publication reference list is empty")
    return values


def _normalize_answer(value: Any) -> str:
    if not isinstance(value, str):
        value = str(value)
    lowered = value.casefold()
    without_punctuation = "".join(
        character for character in lowered if character not in string.punctuation
    )
    without_articles = _ARTICLES.sub(" ", without_punctuation)
    return " ".join(without_articles.split())


def _token_f1(prediction: str, reference: Any) -> float:
    predicted = _normalize_answer(prediction).split()
    expected = _normalize_answer(reference).split()
    if not predicted or not expected:
        return float(predicted == expected)
    overlap = sum((Counter(predicted) & Counter(expected)).values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(predicted)
    recall = overlap / len(expected)
    return 2 * precision * recall / (precision + recall)


def _numeric_tolerances(evaluation: Mapping[str, Any]) -> tuple[float, float]:
    values = []
    for name in ("absolute_tolerance", "relative_tolerance"):
        value = evaluation.get(name, 0.0)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise PublicationEvaluationError(f"{name} must be numeric")
        converted = float(value)
        if not math.isfinite(converted) or converted < 0:
            raise PublicationEvaluationError(f"{name} must be finite and non-negative")
        values.append(converted)
    return values[0], values[1]


def _last_number(value: str) -> float | None:
    matches = [match.group(0) for match in _NUMBER.finditer(value) if match.group(0)]
    if not matches:
        return None
    try:
        return float(matches[-1].replace(",", ""))
    except ValueError:
        return None


def _reference_number(value: Any) -> float:
    if isinstance(value, bool):
        raise PublicationEvaluationError("numeric reference cannot be boolean")
    if isinstance(value, (int, float)):
        converted = float(value)
    elif isinstance(value, str):
        extracted = _last_number(value)
        if extracted is None:
            raise PublicationEvaluationError("numeric reference contains no number")
        converted = extracted
    else:
        raise PublicationEvaluationError("numeric reference has an unsupported type")
    if not math.isfinite(converted):
        raise PublicationEvaluationError("numeric reference must be finite")
    return converted


def _numeric_match(
    prediction: float | None,
    reference: Any,
    absolute_tolerance: float,
    relative_tolerance: float,
) -> float:
    if prediction is None or not math.isfinite(prediction):
        return 0.0
    expected = _reference_number(reference)
    return float(
        math.isclose(
            prediction,
            expected,
            abs_tol=absolute_tolerance,
            rel_tol=relative_tolerance,
        )
    )


def _last_boxed_value(value: str) -> str | None:
    """Extract the last balanced ``\\boxed{...}`` payload without evaluating LaTeX."""

    marker = "\\boxed{"
    start = value.rfind(marker)
    if start < 0:
        marker = "\\fbox{"
        start = value.rfind(marker)
    if start < 0:
        return None
    content_start = start + len(marker)
    depth = 1
    for index in range(content_start, len(value)):
        if value[index] == "{":
            depth += 1
        elif value[index] == "}":
            depth -= 1
            if depth == 0:
                return value[content_start:index]
    return None


def _normalize_math_answer(value: Any) -> str:
    if not isinstance(value, str):
        value = str(value)
    boxed = _last_boxed_value(value)
    if boxed is not None:
        value = boxed
    normalized = value.strip().replace("\n", "").replace(" ", "")
    normalized = re.sub(
        r"^(?:finalanswer|final|answer)[:=]",
        "",
        normalized,
        flags=re.IGNORECASE,
    )
    normalized = normalized.replace("\\left", "").replace("\\right", "")
    normalized = normalized.replace("\\!", "").replace("\\,", "")
    normalized = normalized.replace("tfrac", "frac").replace("dfrac", "frac")
    normalized = normalized.replace("\\$", "").replace("$", "")
    normalized = normalized.rstrip(".")
    if normalized.startswith("."):
        normalized = "0" + normalized
    if normalized.startswith("-") and normalized[1:].startswith("."):
        normalized = "-0" + normalized[1:]
    if re.fullmatch(r"-?\d+/\d+", normalized):
        numerator, denominator = normalized.split("/", 1)
        normalized = f"\\frac{{{numerator}}}{{{denominator}}}"
    return normalized.casefold()


def _function_call_score(prediction: str, reference: Any) -> float:
    contract = _function_call_contract(reference)
    candidate = _first_json_value(prediction)
    calls = _normalize_predicted_calls(candidate)
    if calls is None:
        return 0.0
    expected_sequences = contract["accepted_call_sequences"]
    return float(any(_calls_match(calls, expected) for expected in expected_sequences))


def _function_call_contract(reference: Any) -> dict[str, list[list[dict[str, Any]]]]:
    if not isinstance(reference, Mapping) or set(reference) != {"accepted_call_sequences"}:
        raise PublicationEvaluationError("function-call reference fields are invalid")
    raw_sequences = reference.get("accepted_call_sequences")
    if not isinstance(raw_sequences, list) or not raw_sequences:
        raise PublicationEvaluationError("function-call reference has no accepted sequence")
    sequences: list[list[dict[str, Any]]] = []
    for raw_sequence in raw_sequences:
        if not isinstance(raw_sequence, list) or not raw_sequence:
            raise PublicationEvaluationError("function-call sequence is empty")
        sequence: list[dict[str, Any]] = []
        for raw_call in raw_sequence:
            if not isinstance(raw_call, Mapping) or set(raw_call) != {"name", "arguments"}:
                raise PublicationEvaluationError("function-call reference call is malformed")
            name = raw_call.get("name")
            arguments = raw_call.get("arguments")
            if not isinstance(name, str) or not name or not isinstance(arguments, Mapping):
                raise PublicationEvaluationError("function-call name or arguments are invalid")
            normalized_arguments: dict[str, list[Any]] = {}
            for key, choices in arguments.items():
                if (
                    not isinstance(key, str)
                    or not key
                    or not isinstance(choices, list)
                    or not choices
                ):
                    raise PublicationEvaluationError("function-call argument choices are malformed")
                for choice in choices:
                    try:
                        json.dumps(choice, allow_nan=False, sort_keys=True)
                    except (TypeError, ValueError) as exc:
                        raise PublicationEvaluationError(
                            "function-call argument choice is not canonical JSON"
                        ) from exc
                normalized_arguments[key] = choices
            sequence.append({"name": name, "arguments": normalized_arguments})
        sequences.append(sequence)
    return {"accepted_call_sequences": sequences}


def _normalize_predicted_calls(value: Any) -> list[dict[str, Any]] | None:
    values: Sequence[Any]
    if isinstance(value, Mapping):
        if set(value) == {"name", "arguments"}:
            values = [value]
        elif len(value) == 1:
            name, arguments = next(iter(value.items()))
            values = [{"name": name, "arguments": arguments}]
        else:
            return None
    elif isinstance(value, list):
        values = value
    else:
        return None
    calls: list[dict[str, Any]] = []
    for item in values:
        if not isinstance(item, Mapping):
            return None
        if set(item) == {"name", "arguments"}:
            name = item.get("name")
            arguments = item.get("arguments")
        elif len(item) == 1:
            name, arguments = next(iter(item.items()))
        else:
            return None
        if not isinstance(name, str) or not isinstance(arguments, Mapping):
            return None
        calls.append({"name": name, "arguments": dict(arguments)})
    return calls


def _calls_match(calls: list[dict[str, Any]], expected: list[dict[str, Any]]) -> bool:
    if len(calls) != len(expected):
        return False
    for candidate, reference in zip(calls, expected, strict=True):
        if candidate["name"] != reference["name"]:
            return False
        candidate_arguments = candidate["arguments"]
        reference_arguments = reference["arguments"]
        if set(candidate_arguments) != set(reference_arguments):
            return False
        if any(
            candidate_arguments[name] not in reference_arguments[name]
            for name in reference_arguments
        ):
            return False
    return True


def _first_json_value(value: str) -> Any:
    decoder = json.JSONDecoder()
    for index, character in enumerate(value):
        if character not in "[{":
            continue
        try:
            parsed, _ = decoder.raw_decode(value[index:])
        except json.JSONDecodeError:
            continue
        return parsed
    return None


def _reference_json(value: Any) -> Any:
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError as exc:
            raise PublicationEvaluationError("JSON reference is invalid") from exc
    if isinstance(value, (Mapping, list)):
        return value
    raise PublicationEvaluationError("JSON reference has an unsupported type")


def _python_test_score(prediction: str, reference: Any) -> float:
    contract = _python_test_contract(reference)
    payload = {
        "candidate_code": _extract_python_code(prediction),
        **contract,
    }
    worker = Path(__file__).with_name("_python_eval_worker.py")
    try:
        completed = subprocess.run(
            [sys.executable, "-I", "-S", str(worker)],
            input=json.dumps(payload, allow_nan=False),
            text=True,
            capture_output=True,
            check=False,
            timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired):
        return 0.0
    if completed.returncode != 0:
        return 0.0
    try:
        result = json.loads(completed.stdout)
    except json.JSONDecodeError:
        return 0.0
    return float(result == {"passed": True})


def _python_test_contract(reference: Any) -> dict[str, str]:
    if not isinstance(reference, Mapping):
        raise PublicationEvaluationError("python test reference must be an object")
    if set(reference) != {"entry_point", "test_code", "test_mode"}:
        raise PublicationEvaluationError("python test reference fields are invalid")
    entry_point = reference.get("entry_point")
    test_code = reference.get("test_code")
    test_mode = reference.get("test_mode")
    if not isinstance(entry_point, str) or not entry_point.isidentifier():
        raise PublicationEvaluationError("python test entry point is invalid")
    if not isinstance(test_code, str) or not test_code.strip():
        raise PublicationEvaluationError("python test code is empty")
    if test_mode not in {"check", "exec"}:
        raise PublicationEvaluationError("python test mode must be check or exec")
    try:
        compile(test_code, "<publication-tests>", "exec")
    except SyntaxError as exc:
        raise PublicationEvaluationError("python test code is invalid") from exc
    return {
        "entry_point": entry_point,
        "test_code": test_code,
        "test_mode": test_mode,
    }


def _extract_python_code(prediction: str) -> str:
    fenced = re.search(r"```(?:python)?\s*\n(.*?)```", prediction, flags=re.DOTALL | re.IGNORECASE)
    return fenced.group(1).strip() if fenced else prediction.strip()
