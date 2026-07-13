"""Source-only sidecars and statistically calibrated selective admission."""

from __future__ import annotations

import hashlib
import math
import struct
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from goldenexperience.size_variant.cached_kv_manifest import sha256_file

SIDECAR_SCHEMA_VERSION = "goldenexperience.source_kv_sidecar.v1"
RISK_FEATURE_SCHEMA_VERSION = "goldenexperience.source_kv_risk_features.v1"
RISK_CALIBRATION_METHOD = "bonferroni_clopper_pearson"
SIDECAR_MAX_BYTES = 4096
SKETCH_DIM = 128
RISK_FEATURE_DIM = 169
_SIDECAR_MAGIC = b"GEKVS5\x00\x01"
_STAT_METRICS = (
    "key_rms",
    "key_variance",
    "key_q25_abs",
    "key_q75_abs",
    "value_rms",
    "value_variance",
    "value_q25_abs",
    "value_q75_abs",
)


class RiskGateError(RuntimeError):
    """Raised when risk artifacts or calibration evidence are invalid."""


@dataclass(frozen=True)
class RiskCalibrationExample:
    unsafe_probability: float
    unsafe: bool


@dataclass(frozen=True)
class RiskCalibrationResult:
    threshold: float
    accepted_count: int
    total_count: int
    error_count: int
    coverage: float
    regression_risk_upper_bound: float
    confidence_level: float = 0.95
    calibration_method: str = RISK_CALIBRATION_METHOD
    candidate_threshold_count: int = 1


@dataclass(frozen=True)
class SelectorEvaluationExample:
    unsafe: bool
    predictor_probability: float
    cosine: float


@dataclass(frozen=True)
class SelectorEvaluation:
    name: str
    accepted_count: int
    total_count: int
    error_count: int
    coverage: float
    regression_risk_upper_bound: float


def evaluate_selector_baselines(
    examples: Sequence[SelectorEvaluationExample],
    *,
    calibrated_threshold: float,
    cosine_threshold: float = 0.95,
) -> tuple[SelectorEvaluation, ...]:
    """Evaluate the five fixed selector baselines on one immutable split."""

    if not examples:
        raise ValueError("selector evaluation requires examples")
    predicates = (
        ("no_selector", lambda row: True),
        ("cosine_threshold", lambda row: row.cosine >= cosine_threshold),
        ("uncalibrated_mlp", lambda row: row.predictor_probability <= 0.5),
        ("calibrated_selector", lambda row: row.predictor_probability <= calibrated_threshold),
        ("oracle_selector", lambda row: not row.unsafe),
    )
    results: list[SelectorEvaluation] = []
    for name, predicate in predicates:
        accepted = [row for row in examples if predicate(row)]
        errors = sum(int(row.unsafe) for row in accepted)
        upper = clopper_pearson_upper_bound(errors, len(accepted)) if accepted else 1.0
        results.append(
            SelectorEvaluation(
                name=name,
                accepted_count=len(accepted),
                total_count=len(examples),
                error_count=errors,
                coverage=len(accepted) / len(examples),
                regression_risk_upper_bound=upper,
            )
        )
    return tuple(results)


def clopper_pearson_upper_bound(
    errors: int,
    samples: int,
    *,
    confidence: float = 0.95,
) -> float:
    """Return the exact one-sided binomial upper confidence limit."""

    if samples <= 0:
        raise ValueError("samples must be positive")
    if errors < 0 or errors > samples:
        raise ValueError("errors must be between zero and samples")
    if not 0 < confidence < 1:
        raise ValueError("confidence must be between zero and one")
    if errors == samples:
        return 1.0
    alpha = 1.0 - confidence
    low = 0.0
    high = 1.0
    for _ in range(80):
        middle = (low + high) / 2.0
        if _binomial_cdf(errors, samples, middle) > alpha:
            low = middle
        else:
            high = middle
    return high


def bonferroni_adjusted_confidence(confidence: float, candidate_count: int) -> float:
    """Return the pointwise confidence needed for a family-wise confidence target."""

    if not 0 < confidence < 1:
        raise ValueError("confidence must be between zero and one")
    if candidate_count < 1:
        raise ValueError("candidate_count must be positive")
    return 1.0 - (1.0 - confidence) / candidate_count


def select_calibrated_threshold(
    examples: Sequence[RiskCalibrationExample] | Iterable[RiskCalibrationExample],
    *,
    min_accepted: int = 300,
    max_risk_upper_bound: float = 0.01,
    confidence: float = 0.95,
) -> RiskCalibrationResult:
    """Choose maximum coverage under a simultaneous exact one-sided risk bound."""

    rows = sorted(list(examples), key=lambda item: item.unsafe_probability)
    if not rows:
        raise RiskGateError("risk calibration set is empty")
    if min_accepted < 1:
        raise ValueError("min_accepted must be positive")
    if not 0 < max_risk_upper_bound < 1:
        raise ValueError("max_risk_upper_bound must be between zero and one")
    if not 0 < confidence < 1:
        raise ValueError("confidence must be between zero and one")
    for row in rows:
        if not math.isfinite(row.unsafe_probability) or not 0 <= row.unsafe_probability <= 1:
            raise RiskGateError("risk calibration probabilities must be finite in [0, 1]")

    candidate_count = 0
    cumulative = 0
    index = 0
    while index < len(rows):
        threshold = rows[index].unsafe_probability
        while index < len(rows) and rows[index].unsafe_probability == threshold:
            cumulative += 1
            index += 1
        if cumulative >= min_accepted:
            candidate_count += 1
    if candidate_count == 0:
        raise RiskGateError("risk calibration set has no eligible admission threshold")
    adjusted_confidence = bonferroni_adjusted_confidence(confidence, candidate_count)

    best: RiskCalibrationResult | None = None
    accepted = 0
    failures = 0
    index = 0
    while index < len(rows):
        threshold = rows[index].unsafe_probability
        while index < len(rows) and rows[index].unsafe_probability == threshold:
            failures += int(rows[index].unsafe)
            accepted += 1
            index += 1
        if accepted < min_accepted:
            continue
        upper = clopper_pearson_upper_bound(
            failures,
            accepted,
            confidence=adjusted_confidence,
        )
        if upper <= max_risk_upper_bound:
            best = RiskCalibrationResult(
                threshold=threshold,
                accepted_count=accepted,
                total_count=len(rows),
                error_count=failures,
                coverage=accepted / len(rows),
                regression_risk_upper_bound=upper,
                confidence_level=confidence,
                candidate_threshold_count=candidate_count,
            )
    if best is None:
        raise RiskGateError("no admission threshold satisfies the calibrated risk constraint")
    return best


def unsafe_label(
    *,
    native_task_passed: bool,
    bridge_task_passed: bool,
    greedy_agreement: float,
    perplexity_drift_pct: float,
) -> bool:
    if not math.isfinite(greedy_agreement) or not math.isfinite(perplexity_drift_pct):
        return True
    return bool(
        (native_task_passed and not bridge_task_passed)
        or greedy_agreement < 0.98
        or perplexity_drift_pct > 2.0
    )


@dataclass(frozen=True)
class SourceKVSidecar:
    """Compact source-only evidence generated when a prefix is stored."""

    model_pair_id: str
    source_model_hash: str
    target_model_hash: str
    tokenizer_hash: str
    transport_weights_hash: str
    prefix_hash: str
    prefix_length: int
    token_bucket: int
    num_layers: int
    num_heads: int
    statistics: tuple[float, ...]
    sketch: tuple[float, ...]
    ood_distance: float
    history_samples: int
    history_failures: int
    history_greedy_agreement: float
    schema_version: str = SIDECAR_SCHEMA_VERSION
    feature_schema_version: str = RISK_FEATURE_SCHEMA_VERSION

    def validate(self) -> list[str]:
        errors: list[str] = []
        if self.schema_version != SIDECAR_SCHEMA_VERSION:
            errors.append("unsupported source KV sidecar schema")
        if self.feature_schema_version != RISK_FEATURE_SCHEMA_VERSION:
            errors.append("unsupported source KV risk feature schema")
        if not self.model_pair_id or len(self.model_pair_id.encode("utf-8")) > 96:
            errors.append("model_pair_id must contain at most 96 UTF-8 bytes")
        for name in (
            "source_model_hash",
            "target_model_hash",
            "tokenizer_hash",
            "transport_weights_hash",
            "prefix_hash",
        ):
            try:
                _digest_bytes(getattr(self, name))
            except ValueError:
                errors.append(f"{name} must be a 256-bit hexadecimal digest")
        if self.prefix_length <= 0:
            errors.append("prefix_length must be positive")
        if self.token_bucket not in {128, 512, 2048, 8192}:
            errors.append("token_bucket must be one of 128, 512, 2048, or 8192")
        if self.num_layers <= 0 or self.num_heads <= 0:
            errors.append("sidecar layer and head counts must be positive")
        expected_stats = self.num_layers * self.num_heads * len(_STAT_METRICS)
        if len(self.statistics) != expected_stats:
            errors.append("sidecar statistics shape is invalid")
        if len(self.sketch) != SKETCH_DIM:
            errors.append("sidecar sketch must have 128 dimensions")
        numeric = (*self.statistics, *self.sketch, self.ood_distance, self.history_greedy_agreement)
        if any(not math.isfinite(value) for value in numeric):
            errors.append("sidecar numeric evidence must be finite")
        if self.ood_distance < 0:
            errors.append("sidecar OOD distance must be non-negative")
        if not 0 <= self.history_failures <= self.history_samples:
            errors.append("sidecar history counts are invalid")
        if not 0 <= self.history_greedy_agreement <= 1:
            errors.append("sidecar history greedy agreement must be in [0, 1]")
        return errors

    def to_bytes(self) -> bytes:
        errors = self.validate()
        if errors:
            raise RiskGateError("; ".join(errors))
        pair = self.model_pair_id.encode("utf-8")
        header = bytearray(_SIDECAR_MAGIC)
        header.extend(struct.pack(">B", len(pair)))
        header.extend(pair)
        for value in (
            self.source_model_hash,
            self.target_model_hash,
            self.tokenizer_hash,
            self.transport_weights_hash,
            self.prefix_hash,
        ):
            header.extend(_digest_bytes(value))
        header.extend(
            struct.pack(
                ">IIHHfII f",
                self.prefix_length,
                self.token_bucket,
                self.num_layers,
                self.num_heads,
                self.ood_distance,
                self.history_samples,
                self.history_failures,
                self.history_greedy_agreement,
            )
        )
        stats_payload, stats_ranges = _quantize_metric_major(
            self.statistics,
            rows=self.num_layers * self.num_heads,
            columns=len(_STAT_METRICS),
        )
        for minimum, scale in stats_ranges:
            header.extend(struct.pack(">ff", minimum, scale))
        sketch_payload, sketch_ranges = _quantize_metric_major(
            self.sketch,
            rows=SKETCH_DIM,
            columns=1,
        )
        minimum, scale = sketch_ranges[0]
        header.extend(struct.pack(">ff", minimum, scale))
        header.extend(stats_payload)
        header.extend(sketch_payload)
        header.extend(hashlib.sha256(header).digest()[:16])
        payload = bytes(header)
        if len(payload) > SIDECAR_MAX_BYTES:
            raise RiskGateError(
                f"source KV sidecar is {len(payload)} bytes; maximum is {SIDECAR_MAX_BYTES}"
            )
        return payload

    @classmethod
    def from_bytes(cls, payload: bytes | bytearray | memoryview) -> SourceKVSidecar:
        try:
            return cls._decode_bytes(payload)
        except RiskGateError:
            raise
        except (IndexError, struct.error, UnicodeError, ValueError) as exc:
            raise RiskGateError("source KV sidecar payload is malformed") from exc

    @classmethod
    def _decode_bytes(cls, payload: bytes | bytearray | memoryview) -> SourceKVSidecar:
        raw = bytes(payload)
        if len(raw) > SIDECAR_MAX_BYTES or len(raw) < len(_SIDECAR_MAGIC) + 17:
            raise RiskGateError("source KV sidecar size is invalid")
        if raw[: len(_SIDECAR_MAGIC)] != _SIDECAR_MAGIC:
            raise RiskGateError("source KV sidecar magic is invalid")
        if hashlib.sha256(raw[:-16]).digest()[:16] != raw[-16:]:
            raise RiskGateError("source KV sidecar checksum mismatch")
        offset = len(_SIDECAR_MAGIC)
        pair_len = raw[offset]
        offset += 1
        pair = raw[offset : offset + pair_len].decode("utf-8")
        offset += pair_len
        digests: list[str] = []
        for _ in range(5):
            digests.append(raw[offset : offset + 32].hex())
            offset += 32
        fixed_size = struct.calcsize(">IIHHfII f")
        (
            prefix_length,
            token_bucket,
            num_layers,
            num_heads,
            ood_distance,
            history_samples,
            history_failures,
            history_greedy_agreement,
        ) = struct.unpack(">IIHHfII f", raw[offset : offset + fixed_size])
        offset += fixed_size
        stats_ranges: list[tuple[float, float]] = []
        for _ in _STAT_METRICS:
            stats_ranges.append(struct.unpack(">ff", raw[offset : offset + 8]))
            offset += 8
        sketch_ranges = [struct.unpack(">ff", raw[offset : offset + 8])]
        offset += 8
        stat_count = num_layers * num_heads * len(_STAT_METRICS)
        stats_payload = raw[offset : offset + stat_count]
        offset += stat_count
        sketch_payload = raw[offset : offset + SKETCH_DIM]
        offset += SKETCH_DIM
        if offset != len(raw) - 16:
            raise RiskGateError("source KV sidecar payload length is inconsistent")
        instance = cls(
            model_pair_id=pair,
            source_model_hash=digests[0],
            target_model_hash=digests[1],
            tokenizer_hash=digests[2],
            transport_weights_hash=digests[3],
            prefix_hash=digests[4],
            prefix_length=prefix_length,
            token_bucket=token_bucket,
            num_layers=num_layers,
            num_heads=num_heads,
            statistics=tuple(
                _dequantize_metric_major(
                    stats_payload,
                    rows=num_layers * num_heads,
                    columns=len(_STAT_METRICS),
                    ranges=stats_ranges,
                )
            ),
            sketch=tuple(
                _dequantize_metric_major(
                    sketch_payload,
                    rows=SKETCH_DIM,
                    columns=1,
                    ranges=sketch_ranges,
                )
            ),
            ood_distance=ood_distance,
            history_samples=history_samples,
            history_failures=history_failures,
            history_greedy_agreement=history_greedy_agreement,
        )
        errors = instance.validate()
        if errors:
            raise RiskGateError("; ".join(errors))
        return instance

    def risk_features(self) -> tuple[float, ...]:
        rows = self.num_layers * self.num_heads
        channels = [
            self.statistics[index :: len(_STAT_METRICS)] for index in range(len(_STAT_METRICS))
        ]
        summaries: list[float] = []
        for values in channels:
            mean = sum(values) / rows
            variance = sum((value - mean) ** 2 for value in values) / rows
            summaries.extend((mean, math.sqrt(variance), min(values), max(values)))
        bucket_features = tuple(float(self.token_bucket == item) for item in (128, 512, 2048, 8192))
        failure_rate = self.history_failures / self.history_samples if self.history_samples else 1.0
        features = (
            *summaries,
            *self.sketch,
            math.log1p(self.prefix_length) / math.log1p(8192),
            *bucket_features,
            self.ood_distance,
            math.log1p(self.history_samples) / math.log1p(10_000),
            failure_rate,
            self.history_greedy_agreement,
        )
        if len(features) != RISK_FEATURE_DIM:
            raise AssertionError("risk feature schema has an unexpected width")
        return tuple(features)


def build_source_kv_sidecar(
    source_kv: Any,
    *,
    model_pair_id: str,
    source_model_hash: str,
    target_model_hash: str,
    tokenizer_hash: str,
    transport_weights_hash: str,
    prefix_hash: str,
    prefix_length: int | None = None,
    history_samples: int = 0,
    history_failures: int = 0,
    history_greedy_agreement: float = 1.0,
    ood_distance: float = 0.0,
) -> SourceKVSidecar:
    """Compute layer/head statistics and a deterministic 128-D CountSketch."""

    import torch

    if not isinstance(source_kv, torch.Tensor) or source_kv.ndim != 5 or source_kv.shape[0] != 2:
        raise RiskGateError("source KV must have [2, layer, head, token, head_dim] layout")
    if not bool(torch.isfinite(source_kv).all()):
        raise RiskGateError("source KV contains non-finite values")
    _, layers, heads, tokens, _ = source_kv.shape
    length = int(tokens) if prefix_length is None else int(prefix_length)
    if length != int(tokens):
        raise RiskGateError("prefix_length must match the source KV token axis")
    value = source_kv.detach().float()
    statistics: list[Any] = []
    for kv_index in range(2):
        flattened = value[kv_index].reshape(layers, heads, -1)
        absolute = flattened.abs()
        statistics.extend(
            (
                flattened.square().mean(dim=-1).sqrt(),
                flattened.var(dim=-1, unbiased=False),
                torch.quantile(absolute, 0.25, dim=-1),
                torch.quantile(absolute, 0.75, dim=-1),
            )
        )
    # Convert metric-major tensors into one row per layer/head.
    stats_tensor = torch.stack(statistics, dim=-1).reshape(layers * heads, len(_STAT_METRICS))
    sketch = _count_sketch(value.reshape(-1), SKETCH_DIM)
    sidecar = SourceKVSidecar(
        model_pair_id=model_pair_id,
        source_model_hash=source_model_hash,
        target_model_hash=target_model_hash,
        tokenizer_hash=tokenizer_hash,
        transport_weights_hash=transport_weights_hash,
        prefix_hash=prefix_hash,
        prefix_length=length,
        token_bucket=_token_bucket(length),
        num_layers=int(layers),
        num_heads=int(heads),
        statistics=tuple(float(item) for item in stats_tensor.cpu().reshape(-1).tolist()),
        sketch=tuple(float(item) for item in sketch.cpu().tolist()),
        ood_distance=float(ood_distance),
        history_samples=int(history_samples),
        history_failures=int(history_failures),
        history_greedy_agreement=float(history_greedy_agreement),
    )
    errors = sidecar.validate()
    if errors:
        raise RiskGateError("; ".join(errors))
    return sidecar


def build_transport_source_sidecar(
    source_kv: Any,
    transport: Any,
    *,
    model_pair_id: str,
    prefix_hash: str,
    history_samples: int = 0,
    history_failures: int = 0,
    history_greedy_agreement: float = 1.0,
) -> SourceKVSidecar:
    """Build a production sidecar with OOD distance from fitted transport state."""

    ood_distance = transport.ood_distance(source_kv)
    return build_source_kv_sidecar(
        source_kv,
        model_pair_id=model_pair_id,
        source_model_hash=transport.source.weights_sha256,
        target_model_hash=transport.target.weights_sha256,
        tokenizer_hash=transport.source.tokenizer_sha256,
        transport_weights_hash=transport.spec.weights_sha256,
        prefix_hash=prefix_hash,
        history_samples=history_samples,
        history_failures=history_failures,
        history_greedy_agreement=history_greedy_agreement,
        ood_distance=ood_distance,
    )


class RiskPredictor:
    """Fixed two-layer MLP used only to rank source-side risk."""

    REQUIRED_TENSORS = frozenset(
        {
            "input_mean",
            "input_scale",
            "layer1_weight",
            "layer1_bias",
            "output_weight",
            "output_bias",
        }
    )

    def __init__(self, tensors: Mapping[str, Any], *, device: str = "cpu") -> None:
        import torch

        if set(tensors) != self.REQUIRED_TENSORS:
            raise RiskGateError("risk predictor tensor set is invalid")
        self.device = torch.device(device)
        self.tensors = {
            name: tensor.detach().float().to(self.device) for name, tensor in tensors.items()
        }
        expected = {
            "input_mean": (RISK_FEATURE_DIM,),
            "input_scale": (RISK_FEATURE_DIM,),
            "layer1_weight": (64, RISK_FEATURE_DIM),
            "layer1_bias": (64,),
            "output_weight": (1, 64),
            "output_bias": (1,),
        }
        for name, shape in expected.items():
            tensor = self.tensors[name]
            if tuple(tensor.shape) != shape or not bool(torch.isfinite(tensor).all()):
                raise RiskGateError(f"risk predictor {name} is invalid")
        if bool((self.tensors["input_scale"] <= 0).any()):
            raise RiskGateError("risk predictor input_scale must be positive")

    def unsafe_probability(self, features: Sequence[float]) -> float:
        import torch
        import torch.nn.functional as functional

        if len(features) != RISK_FEATURE_DIM:
            raise RiskGateError("risk predictor feature width is invalid")
        value = torch.tensor(features, dtype=torch.float32, device=self.device)
        normalized = (value - self.tensors["input_mean"]) / self.tensors["input_scale"]
        hidden = functional.silu(
            functional.linear(
                normalized, self.tensors["layer1_weight"], self.tensors["layer1_bias"]
            )
        )
        logits = functional.linear(
            hidden, self.tensors["output_weight"], self.tensors["output_bias"]
        )
        probability = float(torch.sigmoid(logits).item())
        if not math.isfinite(probability):
            raise RiskGateError("risk predictor produced a non-finite probability")
        return probability

    @classmethod
    def from_artifact(
        cls,
        path: str | Path,
        *,
        expected_sha256: str,
        device: str = "cpu",
    ) -> RiskPredictor:
        from safetensors import safe_open

        artifact = Path(path)
        if artifact.suffix != ".safetensors" or not artifact.is_file():
            raise RiskGateError("risk predictor artifact is missing")
        if sha256_file(artifact) != expected_sha256:
            raise RiskGateError("risk predictor checksum mismatch")
        tensors: dict[str, Any] = {}
        with safe_open(str(artifact), framework="pt", device="cpu") as handle:
            metadata = handle.metadata() or {}
            if metadata != {
                "feature_schema_version": RISK_FEATURE_SCHEMA_VERSION,
                "hidden_size": "64",
            }:
                raise RiskGateError("risk predictor metadata is invalid")
            for name in handle.keys():  # noqa: SIM118
                tensors[name] = handle.get_tensor(name)
        if sha256_file(artifact) != expected_sha256:
            raise RiskGateError("risk predictor changed while loading")
        return cls(tensors, device=device)


def fit_risk_predictor(
    feature_rows: Any,
    unsafe_labels: Any,
    *,
    seed: int = 17,
    epochs: int = 200,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    device: str = "cpu",
) -> dict[str, Any]:
    """Fit the frozen 64-hidden-unit ranking MLP on selector-train only."""

    import torch
    import torch.nn.functional as functional

    if epochs <= 0:
        raise ValueError("risk predictor epochs must be positive")
    if not math.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("risk predictor learning_rate must be finite and positive")
    if not math.isfinite(weight_decay) or weight_decay < 0:
        raise ValueError("risk predictor weight_decay must be finite and non-negative")
    features = torch.as_tensor(feature_rows, dtype=torch.float32, device=device)
    labels = torch.as_tensor(unsafe_labels, dtype=torch.float32, device=device).reshape(-1, 1)
    if features.ndim != 2 or int(features.shape[1]) != RISK_FEATURE_DIM:
        raise RiskGateError(f"risk training features must have width {RISK_FEATURE_DIM}")
    if labels.shape != (features.shape[0], 1) or features.shape[0] < 2:
        raise RiskGateError("risk training labels do not match feature rows")
    if not bool(torch.isfinite(features).all()) or not bool(torch.isfinite(labels).all()):
        raise RiskGateError("risk training data contains non-finite values")
    if bool(((labels != 0) & (labels != 1)).any()):
        raise RiskGateError("risk training labels must be binary")
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    mean = features.mean(dim=0)
    scale = features.std(dim=0, unbiased=False).clamp_min(1e-6)
    normalized = (features - mean) / scale
    layer1_weight = torch.empty(64, RISK_FEATURE_DIM, device=device)
    layer1_weight.normal_(mean=0.0, std=0.02, generator=generator)
    layer1_bias = torch.zeros(64, device=device)
    output_weight = torch.empty(1, 64, device=device)
    output_weight.normal_(mean=0.0, std=0.02, generator=generator)
    output_bias = torch.zeros(1, device=device)
    parameters = (layer1_weight, layer1_bias, output_weight, output_bias)
    for parameter in parameters:
        parameter.requires_grad_(True)
    optimizer = torch.optim.AdamW(parameters, lr=learning_rate, weight_decay=weight_decay)
    positive = labels.sum().clamp_min(1.0)
    negative = (1.0 - labels).sum().clamp_min(1.0)
    positive_weight = negative / positive
    for _ in range(epochs):
        hidden = functional.silu(functional.linear(normalized, layer1_weight, layer1_bias))
        logits = functional.linear(hidden, output_weight, output_bias)
        loss = functional.binary_cross_entropy_with_logits(
            logits,
            labels,
            pos_weight=positive_weight,
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    return {
        "input_mean": mean.detach().cpu(),
        "input_scale": scale.detach().cpu(),
        "layer1_weight": layer1_weight.detach().cpu(),
        "layer1_bias": layer1_bias.detach().cpu(),
        "output_weight": output_weight.detach().cpu(),
        "output_bias": output_bias.detach().cpu(),
    }


@dataclass(frozen=True)
class AdmissionDecision:
    accepted: bool
    reason: str
    unsafe_probability: float | None = None


class CalibratedRiskGate:
    """Fail-closed runtime gate that never executes target native prefill."""

    def __init__(
        self,
        spec: Any,
        predictor: RiskPredictor,
        *,
        model_pair_id: str,
        source_model_hash: str,
        target_model_hash: str,
        tokenizer_hash: str,
        transport_weights_hash: str,
    ) -> None:
        errors = spec.calibration_errors()
        if errors:
            raise RiskGateError("; ".join(errors))
        self.spec = spec
        self.predictor = predictor
        self.expected = {
            "model_pair_id": model_pair_id,
            "source_model_hash": source_model_hash,
            "target_model_hash": target_model_hash,
            "tokenizer_hash": tokenizer_hash,
            "transport_weights_hash": transport_weights_hash,
        }

    def evaluate(self, sidecar: SourceKVSidecar | bytes | None) -> AdmissionDecision:
        if sidecar is None:
            return AdmissionDecision(False, "missing_sidecar")
        if isinstance(sidecar, bytes):
            try:
                sidecar = SourceKVSidecar.from_bytes(sidecar)
            except (IndexError, RiskGateError, UnicodeError, struct.error):
                return AdmissionDecision(False, "invalid_sidecar")
        errors = sidecar.validate()
        if errors:
            return AdmissionDecision(False, "invalid_sidecar")
        for name, expected in self.expected.items():
            if getattr(sidecar, name) != expected:
                reason = "model_hash_changed" if name != "model_pair_id" else "model_pair_changed"
                return AdmissionDecision(False, reason)
        if sidecar.history_samples < self.spec.min_shadow_samples:
            return AdmissionDecision(False, "unseen_or_insufficient_shadow_history")
        if sidecar.ood_distance > self.spec.ood_threshold:
            return AdmissionDecision(False, "out_of_distribution")
        try:
            probability = self.predictor.unsafe_probability(sidecar.risk_features())
        except Exception:
            return AdmissionDecision(False, "predictor_failure")
        assert self.spec.threshold is not None
        if probability > self.spec.threshold:
            return AdmissionDecision(False, "predicted_unsafe", probability)
        return AdmissionDecision(True, "accepted", probability)


def _binomial_cdf(k: int, n: int, probability: float) -> float:
    if probability <= 0:
        return 1.0
    if probability >= 1:
        return float(k == n)
    logs = [
        math.lgamma(n + 1)
        - math.lgamma(index + 1)
        - math.lgamma(n - index + 1)
        + index * math.log(probability)
        + (n - index) * math.log1p(-probability)
        for index in range(k + 1)
    ]
    maximum = max(logs)
    return math.exp(maximum) * sum(math.exp(value - maximum) for value in logs)


def _digest_bytes(value: str) -> bytes:
    text = value[2:] if isinstance(value, str) and value.startswith("0x") else value
    if not isinstance(text, str) or len(text) != 64:
        raise ValueError("digest must contain 64 hexadecimal characters")
    try:
        return bytes.fromhex(text)
    except ValueError as exc:
        raise ValueError("digest is not hexadecimal") from exc


def _token_bucket(length: int) -> int:
    for bucket in (128, 512, 2048, 8192):
        if length <= bucket:
            return bucket
    raise RiskGateError("prefix exceeds the supported 8192-token bucket")


def _count_sketch(value: Any, dimensions: int) -> Any:
    import torch

    flattened = value.float().reshape(-1)
    output = torch.zeros(dimensions, dtype=torch.float32, device=flattened.device)
    chunk = 1_000_000
    for start in range(0, flattened.numel(), chunk):
        part = flattened[start : start + chunk]
        indices = torch.arange(start, start + part.numel(), device=part.device, dtype=torch.int64)
        hashed = indices * 6364136223846793005 + 1442695040888963407
        buckets = torch.remainder(hashed, dimensions)
        signs = torch.where((hashed.bitwise_right_shift(17) & 1) == 0, 1.0, -1.0)
        output.scatter_add_(0, buckets, part * signs)
    return output / math.sqrt(max(1, flattened.numel()))


def _quantize_metric_major(
    values: Sequence[float],
    *,
    rows: int,
    columns: int,
) -> tuple[bytes, list[tuple[float, float]]]:
    if len(values) != rows * columns:
        raise RiskGateError("quantized sidecar matrix has an invalid shape")
    payload = bytearray(rows * columns)
    ranges: list[tuple[float, float]] = []
    for column in range(columns):
        channel = [float(values[row * columns + column]) for row in range(rows)]
        minimum = min(channel)
        maximum = max(channel)
        scale = (maximum - minimum) / 255.0 if maximum > minimum else 1.0
        ranges.append((minimum, scale))
        for row, value in enumerate(channel):
            payload[row * columns + column] = min(255, max(0, round((value - minimum) / scale)))
    return bytes(payload), ranges


def _dequantize_metric_major(
    payload: bytes,
    *,
    rows: int,
    columns: int,
    ranges: Sequence[tuple[float, float]],
) -> list[float]:
    if len(payload) != rows * columns or len(ranges) != columns:
        raise RiskGateError("quantized sidecar payload has an invalid shape")
    values: list[float] = []
    for row in range(rows):
        for column in range(columns):
            minimum, scale = ranges[column]
            values.append(minimum + payload[row * columns + column] * scale)
    return values
