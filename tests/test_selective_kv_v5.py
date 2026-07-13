import hashlib
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path

import pytest
import torch
from safetensors import torch as safetensors_torch

from goldenexperience.benchmarks.publication import (
    PREFIX_BUCKETS,
    REQUIRED_DATASETS,
    SPLIT_COUNTS,
    BenchmarkContractError,
    DatasetSource,
    DirectionValidationEvidence,
    GroupedPrefixRecord,
    PublicationBenchmarkManifest,
    SemanticSealedGuard,
    ValidationGateReceipt,
    write_immutable_sealed_report,
)
from goldenexperience.benchmarks.selective_runtime import (
    build_selective_runtime_report,
    runtime_cost_evidence_from_report,
)
from goldenexperience.reuse.models import ReuseRequest
from goldenexperience.reuse.planner import CrossModelReusePlanner
from goldenexperience.runtime.direct_paged_kv import (
    DirectPagedKVInjector,
    InMemoryBlockValidityTracker,
    RetrieveTransformRequest,
    scatter_paged_kv,
)
from goldenexperience.size_variant.attention_collection import TargetAttentionCollector
from goldenexperience.size_variant.cached_kv_manifest import (
    CachedKVBridgeManifest,
    CachedKVModelSpec,
    CachedKVQualityEvidence,
    artifact_id_for,
    load_cached_kv_manifest,
    model_ref_from_cached_spec,
    model_spec_from_path,
)
from goldenexperience.size_variant.head_aware_transport import (
    HeadAwareKVTransport,
    TransportScreeningCandidate,
    _apply_rope_heads,
    attention_distillation_terms,
    attention_preserving_loss,
    build_trainable_head_aware_transport,
    fit_head_aware_normalizers,
    head_aware_training_objective,
    initialize_head_aware_state,
    sample_attention_positions,
    select_transport_candidate,
    transport_safetensors_metadata,
)
from goldenexperience.size_variant.risk_gate import (
    RISK_CALIBRATION_METHOD,
    RISK_FEATURE_SCHEMA_VERSION,
    CalibratedRiskGate,
    RiskCalibrationExample,
    RiskGateError,
    RiskPredictor,
    SelectorEvaluationExample,
    SourceKVSidecar,
    bonferroni_adjusted_confidence,
    build_source_kv_sidecar,
    clopper_pearson_upper_bound,
    evaluate_selector_baselines,
    fit_risk_predictor,
    select_calibrated_threshold,
    unsafe_label,
)
from goldenexperience.size_variant.selective_manifest import (
    AcceptedSubsetQualityEvidence,
    ArtifactState,
    DirectInjectionEvidence,
    RiskGateSpec,
    RuntimeCostEvidence,
    SelectiveKVBridgeManifest,
    SemanticSealedEvidence,
    TransportQualityEvidence,
    TransportSpec,
)


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _model(
    name: str,
    *,
    size: float,
    layers: int,
    heads: int,
    architecture: str = "qwen3",
) -> CachedKVModelSpec:
    return CachedKVModelSpec(
        model_id=name,
        parameter_count_b=size,
        revision="test",
        architecture=architecture,
        config_sha256=_digest(name + "-config"),
        tokenizer_sha256=_digest("shared-tokenizer"),
        weights_sha256=_digest(name + "-weights"),
        num_layers=layers,
        num_key_value_heads=heads,
        head_dim=32,
        dtype="bfloat16",
        rope_theta=1_000_000.0,
        max_position_embeddings=8192,
    )


def _transport() -> tuple[
    CachedKVModelSpec, CachedKVModelSpec, TransportSpec, HeadAwareKVTransport
]:
    source = _model("qwen3-4b", size=4, layers=3, heads=4)
    target = _model("qwen3-8b", size=8, layers=4, heads=8)
    spec = TransportSpec(
        weights_uri="transport.safetensors",
        weights_sha256=_digest("transport"),
        rank=32,
        source_window=3,
    )
    state = initialize_head_aware_state(source, target, spec)
    return source, target, spec, HeadAwareKVTransport(source, target, spec, state)


def _predictor(*, output_bias: float = -10.0) -> RiskPredictor:
    return RiskPredictor(
        {
            "input_mean": torch.zeros(169),
            "input_scale": torch.ones(169),
            "layer1_weight": torch.zeros(64, 169),
            "layer1_bias": torch.zeros(64),
            "output_weight": torch.zeros(1, 64),
            "output_bias": torch.tensor([output_bias]),
        }
    )


def _risk_spec() -> RiskGateSpec:
    upper = clopper_pearson_upper_bound(0, 300)
    return RiskGateSpec(
        predictor_uri="risk.safetensors",
        predictor_sha256=_digest("risk"),
        threshold=0.01,
        calibration_dataset_sha256=_digest("risk-calibration"),
        calibration_method=RISK_CALIBRATION_METHOD,
        candidate_threshold_count=1,
        accepted_count=300,
        total_count=2048,
        error_count=0,
        coverage=300 / 2048,
        regression_risk_upper_bound=upper,
    )


def _sidecar(source: CachedKVModelSpec, target: CachedKVModelSpec, source_kv: torch.Tensor):
    return build_source_kv_sidecar(
        source_kv,
        model_pair_id="qwen3-4b-to-8b",
        source_model_hash=source.weights_sha256,
        target_model_hash=target.weights_sha256,
        tokenizer_hash=source.tokenizer_sha256,
        transport_weights_hash=_digest("transport"),
        prefix_hash=_digest("prefix"),
        history_samples=8,
        history_failures=0,
        history_greedy_agreement=1.0,
    )


def _gate(source: CachedKVModelSpec, target: CachedKVModelSpec) -> CalibratedRiskGate:
    return CalibratedRiskGate(
        _risk_spec(),
        _predictor(),
        model_pair_id="qwen3-4b-to-8b",
        source_model_hash=source.weights_sha256,
        target_model_hash=target.weights_sha256,
        tokenizer_hash=source.tokenizer_sha256,
        transport_weights_hash=_digest("transport"),
    )


def test_qwen2_config_infers_head_dim_and_ignores_disabled_sliding_window(
    tmp_path: Path,
) -> None:
    model_dir = tmp_path / "qwen2"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(
        json.dumps(
            {
                "model_type": "qwen2",
                "num_hidden_layers": 4,
                "num_attention_heads": 8,
                "num_key_value_heads": 4,
                "hidden_size": 256,
                "torch_dtype": "bfloat16",
                "rope_theta": 1_000_000,
                "max_position_embeddings": 32768,
                "use_sliding_window": False,
                "sliding_window": 4096,
            }
        ),
        encoding="utf-8",
    )
    (model_dir / "tokenizer.json").write_text("{}", encoding="utf-8")
    safetensors_torch.save_file({"weight": torch.ones(1)}, model_dir / "model.safetensors")

    spec = model_spec_from_path(
        model_dir,
        model_id="Qwen2.5-test",
        parameter_count_b=7,
        revision="test",
    )

    assert spec.architecture == "qwen2"
    assert spec.head_dim == 32
    assert spec.sliding_window is None
    assert spec.validate() == []


def test_v4_remains_qwen3_only_after_model_spec_extension() -> None:
    source = _model("qwen2-small", size=7, layers=3, heads=4, architecture="qwen2")
    target = _model("qwen2-large", size=14, layers=4, heads=4, architecture="qwen2")
    test_hash = _digest("test")
    quality = CachedKVQualityEvidence(
        evaluation_dataset_sha256=test_hash,
        held_out_prompts=64,
        evaluated_tokens=8192,
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
        cost_report_sha256=_digest("cost"),
        cost_candidate_manifest_sha256=_digest("candidate"),
        p95_source_read_transform_put_ms=10.0,
        p95_target_prefill_ms=20.0,
    )
    manifest = CachedKVBridgeManifest(
        bridge_id="pending",
        direction="8b_to_14b",
        source=source,
        target=target,
        weights_uri="bridge.safetensors",
        weights_sha256=_digest("weights"),
        rank=32,
        source_window=1,
        train_dataset_sha256=_digest("train"),
        validation_dataset_sha256=_digest("validation"),
        test_dataset_sha256=test_hash,
        quality=quality,
    )
    manifest = replace(manifest, bridge_id=artifact_id_for(manifest))

    assert any("v4 only supports qwen3" in error for error in manifest.artifact_errors())


def test_head_aware_transport_maps_four_to_eight_heads_and_rope_round_trips() -> None:
    source, target, _, transport = _transport()
    source_kv = torch.randn(
        2,
        source.num_layers,
        source.num_key_value_heads,
        7,
        source.head_dim,
        dtype=torch.bfloat16,
    )

    result = transport.transform(source_kv, position_start=17)

    assert result.shape == (2, target.num_layers, target.num_key_value_heads, 7, target.head_dim)
    assert result.dtype == torch.bfloat16
    torch.testing.assert_close(
        transport.tensors["head_mix"].sum(dim=-1),
        torch.ones(target.num_layers, target.num_key_value_heads, 3),
    )
    value = torch.randn(2, 4, 7, 32)
    positions = torch.tensor([0, 1, 3, 17, 128, 1024, 4095])
    rotated = _apply_rope_heads(value, positions, theta=1_000_000, inverse=False)
    restored = _apply_rope_heads(rotated, positions, theta=1_000_000, inverse=True)
    torch.testing.assert_close(restored, value, atol=2e-5, rtol=2e-5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_head_aware_transport_cpu_gpu_numerical_consistency() -> None:
    source = _model("qwen3-4b", size=4, layers=3, heads=4)
    target = _model("qwen3-8b", size=8, layers=4, heads=8)
    spec = TransportSpec(
        weights_uri="transport.safetensors",
        weights_sha256=_digest("transport"),
        rank=32,
        source_window=3,
    )
    state = initialize_head_aware_state(source, target, spec)
    cpu = HeadAwareKVTransport(source, target, spec, state, compute_dtype=torch.float32)
    gpu = HeadAwareKVTransport(
        source,
        target,
        spec,
        state,
        device="cuda",
        compute_dtype=torch.float32,
    )
    source_kv = torch.randn(2, 3, 4, 7, 32, dtype=torch.bfloat16)

    cpu_result = cpu.transform(source_kv, position_start=9)
    gpu_result = gpu.transform(source_kv.cuda(), position_start=9).cpu()

    torch.testing.assert_close(cpu_result, gpu_result, atol=0.02, rtol=0.02)


def test_attention_losses_are_zero_for_native_kv_and_sampling_is_bounded() -> None:
    query = torch.randn(2, 8, 5, 32)
    key = torch.randn(2, 4, 9, 32)
    value = torch.randn(2, 4, 9, 32)

    logit_kl, output_mse = attention_distillation_terms(query, key, value, key, value)
    terms = attention_preserving_loss(
        native_generation_loss=torch.tensor(1.0),
        prompt_tail_distillation_loss=torch.tensor(2.0),
        attention_logit_kl=logit_kl,
        attention_output_mse=output_mse,
        transformed_kv_anchor_loss=torch.tensor(3.0),
    )
    queries, keys = sample_attention_positions(8192)

    assert abs(float(logit_kl)) < 1e-6
    assert float(output_mse) == 0.0
    assert float(terms.total) == pytest.approx(1.8, abs=1e-6)
    assert len(queries) == 32
    assert len(keys) == 256


def test_attention_kl_is_finite_with_causal_mask() -> None:
    query = torch.randn(2, 8, 5, 32)
    native_key = torch.randn(2, 4, 9, 32)
    native_value = torch.randn(2, 4, 9, 32)
    transformed_key = native_key + 0.01 * torch.randn_like(native_key)
    transformed_value = native_value + 0.01 * torch.randn_like(native_value)
    query_positions = torch.tensor([0, 2, 4, 6, 8])
    key_positions = torch.arange(9)
    causal_mask = (key_positions[None, :] <= query_positions[:, None])[None, None]

    logit_kl, output_mse = attention_distillation_terms(
        query,
        native_key,
        native_value,
        transformed_key,
        transformed_value,
        attention_mask=causal_mask,
    )
    identity_kl, _ = attention_distillation_terms(
        query,
        native_key,
        native_value,
        native_key,
        native_value,
        attention_mask=causal_mask,
    )

    assert torch.isfinite(logit_kl)
    assert torch.isfinite(output_mse)
    assert float(logit_kl) >= 0.0
    assert abs(float(identity_kl)) < 1e-6


def test_transport_screening_uses_the_registered_lexicographic_order() -> None:
    common = {
        "direction": "qwen3_4b_to_8b",
        "rank": 32,
        "attention_loss_weight": 0.5,
        "task_score": 0.97,
        "greedy_agreement": 0.99,
        "transform_cost_ms": 10.0,
    }
    candidates = (
        TransportScreeningCandidate(
            candidate_id="lower-coverage", oracle_safe_coverage=0.5, **common
        ),
        TransportScreeningCandidate(
            candidate_id="higher-coverage", oracle_safe_coverage=0.6, **common
        ),
    )

    assert select_transport_candidate(candidates).candidate_id == "higher-coverage"


def test_trainable_transport_exports_the_exact_runtime_tensor_contract() -> None:
    source, target, spec, _ = _transport()
    module = build_trainable_head_aware_transport(source, target, spec)
    source_kv = torch.randn(2, 3, 4, 3, 32, dtype=torch.bfloat16)
    positions = torch.arange(3)
    fit_head_aware_normalizers(module, ((source_kv, positions),))
    with torch.no_grad():
        native = module(source_kv, positions).detach() + 0.01
    query = torch.randn(4, 8, 3, 32)

    _, terms = head_aware_training_objective(
        module,
        source_kv,
        native,
        positions,
        query,
        native_generation_loss=torch.tensor(0.0),
        prompt_tail_distillation_loss=torch.tensor(0.0),
    )
    terms.total.backward()
    runtime = HeadAwareKVTransport(source, target, spec, module.runtime_state())

    assert torch.isfinite(terms.total)
    assert any(parameter.grad is not None for parameter in module.parameters())
    assert bool((module.key_normalizer_scale > 0).all())
    assert runtime.transform(source_kv).shape == native.shape


def test_transport_loads_from_one_manifest_snapshot(tmp_path: Path, monkeypatch) -> None:
    manifest = _v5_manifest()
    weights_path = tmp_path / "transport.safetensors"
    state = initialize_head_aware_state(manifest.source, manifest.target, manifest.transport)
    safetensors_torch.save_file(
        state,
        weights_path,
        metadata=transport_safetensors_metadata(manifest),
    )
    weights_sha256 = hashlib.sha256(weights_path.read_bytes()).hexdigest()
    manifest = replace(
        manifest,
        artifact_id="",
        transport=replace(
            manifest.transport,
            weights_uri=weights_path.name,
            weights_sha256=weights_sha256,
        ),
    ).with_content_id()

    def fail_reload(cls, path):
        raise AssertionError("transport loader re-read the manifest")

    monkeypatch.setattr(SelectiveKVBridgeManifest, "load", classmethod(fail_reload))

    loaded = HeadAwareKVTransport.from_manifest(
        manifest,
        tmp_path / "manifest.json",
        offline=True,
    )

    assert loaded.spec.weights_sha256 == weights_sha256


def test_risk_predictor_detects_artifact_change_during_load(tmp_path: Path, monkeypatch) -> None:
    path = tmp_path / "risk.safetensors"
    predictor = _predictor()
    safetensors_torch.save_file(
        predictor.tensors,
        path,
        metadata={
            "feature_schema_version": RISK_FEATURE_SCHEMA_VERSION,
            "hidden_size": "64",
        },
    )
    expected = hashlib.sha256(path.read_bytes()).hexdigest()
    observed = iter((expected, _digest("changed")))
    monkeypatch.setattr(
        "goldenexperience.size_variant.risk_gate.sha256_file",
        lambda artifact: next(observed),
    )

    with pytest.raises(RiskGateError, match="changed while loading"):
        RiskPredictor.from_artifact(path, expected_sha256=expected)


def test_target_attention_collector_bounds_query_and_output_capture() -> None:
    class _Attention(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.config = type("Config", (), {"num_attention_heads": 2})()
            self.head_dim = 4
            self.q_proj = torch.nn.Linear(8, 8, bias=False)
            self.q_norm = torch.nn.Identity()
            self.o_proj = torch.nn.Linear(8, 8, bias=False)

        def forward(self, value):
            return self.o_proj(self.q_norm(self.q_proj(value)))

    class _Layer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.self_attn = _Attention()

    class _Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = torch.nn.Module()
            self.model.layers = torch.nn.ModuleList([_Layer(), _Layer()])

        def forward(self, value):
            for layer in self.model.layers:
                value = layer.self_attn(value)
            return value

    model = _Model()
    with TargetAttentionCollector(
        model,
        token_count=40,
        rope_theta=1_000_000,
        max_queries=8,
        max_keys=16,
    ) as collector:
        model(torch.randn(1, 40, 8))
    trace = collector.trace()

    assert trace.queries.shape == (2, 2, 8, 4)
    assert trace.attention_outputs.shape == (2, 2, 8, 4)
    assert trace.key_positions.numel() == 16
    assert trace.causal_mask.shape == (1, 1, 8, 16)


def test_source_sidecar_is_bounded_stable_and_checksum_protected() -> None:
    source_kv = torch.randn(2, 48, 8, 2, 4, dtype=torch.bfloat16)
    sidecar = build_source_kv_sidecar(
        source_kv,
        model_pair_id="qwen2.5-7b-to-14b",
        source_model_hash=_digest("source"),
        target_model_hash=_digest("target"),
        tokenizer_hash=_digest("tokenizer"),
        transport_weights_hash=_digest("transport"),
        prefix_hash=_digest("prefix"),
        history_samples=3,
        history_greedy_agreement=1.0,
    )

    payload = sidecar.to_bytes()
    restored = SourceKVSidecar.from_bytes(payload)

    assert len(payload) <= 4096
    assert len(restored.risk_features()) == 169
    assert restored.num_layers == 48
    assert restored.num_heads == 8
    tampered = bytearray(payload)
    tampered[-20] ^= 1
    with pytest.raises(RiskGateError, match="checksum"):
        SourceKVSidecar.from_bytes(tampered)


def test_calibration_maximizes_coverage_under_simultaneous_exact_bound() -> None:
    examples = [RiskCalibrationExample(0.1, False) for _ in range(1200)]
    examples.extend(RiskCalibrationExample(0.9, True) for _ in range(848))

    result = select_calibrated_threshold(examples)

    assert result.accepted_count == 1200
    assert result.error_count == 0
    assert result.coverage == 1200 / 2048
    assert result.regression_risk_upper_bound < 0.01
    assert result.calibration_method == RISK_CALIBRATION_METHOD
    assert result.candidate_threshold_count == 2
    assert bonferroni_adjusted_confidence(0.95, 2) == pytest.approx(0.975)
    assert clopper_pearson_upper_bound(0, 300) == pytest.approx(0.009936081944, abs=1e-12)
    assert unsafe_label(
        native_task_passed=True,
        bridge_task_passed=False,
        greedy_agreement=1.0,
        perplexity_drift_pct=0.0,
    )


def test_calibration_rejects_candidate_that_only_passes_a_pointwise_bound() -> None:
    examples = [RiskCalibrationExample(0.1, False) for _ in range(300)]
    examples.append(RiskCalibrationExample(0.9, True))

    assert clopper_pearson_upper_bound(0, 300) < 0.01
    with pytest.raises(RiskGateError, match="no admission threshold"):
        select_calibrated_threshold(examples)


def test_risk_predictor_training_keeps_fixed_two_layer_shape() -> None:
    features = torch.randn(32, 169)
    labels = torch.tensor([0, 1] * 16)

    state = fit_risk_predictor(features, labels, epochs=3)
    predictor = RiskPredictor(state)

    assert state["layer1_weight"].shape == (64, 169)
    probability = predictor.unsafe_probability(features[0].tolist())
    assert 0 <= probability <= 1


def test_selector_evaluation_contains_all_fixed_baselines() -> None:
    examples = tuple(
        SelectorEvaluationExample(
            unsafe=index >= 8,
            predictor_probability=index / 10,
            cosine=1.0 - index / 100,
        )
        for index in range(10)
    )

    results = evaluate_selector_baselines(examples, calibrated_threshold=0.7)

    assert [result.name for result in results] == [
        "no_selector",
        "cosine_threshold",
        "uncalibrated_mlp",
        "calibrated_selector",
        "oracle_selector",
    ]
    assert results[-1].error_count == 0


def test_risk_gate_fails_closed_for_missing_ood_history_and_identity_changes() -> None:
    source, target, _, _ = _transport()
    source_kv = torch.zeros(2, 3, 4, 3, 32, dtype=torch.bfloat16)
    sidecar = _sidecar(source, target, source_kv)
    gate = _gate(source, target)

    assert gate.evaluate(sidecar).accepted is True
    assert gate.evaluate(None).reason == "missing_sidecar"
    assert gate.evaluate(replace(sidecar, history_samples=0)).reason.startswith("unseen")
    assert gate.evaluate(replace(sidecar, ood_distance=7.0)).reason == "out_of_distribution"
    assert gate.evaluate(replace(sidecar, source_model_hash=_digest("changed"))).reason == (
        "model_hash_changed"
    )

    class BrokenPredictor:
        def unsafe_probability(self, features):
            raise KeyError("predictor backend failed")

    gate.predictor = BrokenPredictor()
    assert gate.evaluate(sidecar).reason == "predictor_failure"


class _Reader:
    def __init__(self, chunks):
        self.chunks = chunks
        self.calls = 0

    def read_many_exact(self, keys, *, timeout_s):
        assert timeout_s == 5.0
        self.calls += 1
        return [self.chunks[key] for key in keys]


def _tensor_bytes(value: torch.Tensor) -> bytes:
    return bytes(value.contiguous().view(torch.uint8).numpy())


def test_direct_paged_injection_reads_only_after_acceptance_and_never_puts_target() -> None:
    source, target, _, transport = _transport()
    first = torch.randn(2, 3, 4, 2, 32, dtype=torch.bfloat16)
    second = torch.randn(2, 3, 4, 1, 32, dtype=torch.bfloat16)
    chunks = {"a": _tensor_bytes(first), "b": _tensor_bytes(second)}
    reader = _Reader(chunks)
    tracker = InMemoryBlockValidityTracker()
    events = []
    injector = DirectPagedKVInjector(
        risk_gate=_gate(source, target),
        transport=transport,
        source_reader=reader,
        validity_tracker=tracker,
        publish_load_complete=lambda request_id, blocks: events.append((request_id, blocks)),
    )
    sidecar = _sidecar(source, target, torch.cat((first, second), dim=3))
    request = RetrieveTransformRequest(
        request_id="request-1",
        source_keys=("a", "b"),
        source_checksums=(
            hashlib.sha256(chunks["a"]).hexdigest(),
            hashlib.sha256(chunks["b"]).hexdigest(),
        ),
        chunk_token_counts=(2, 1),
        slot_mapping=(1, 2, 5),
        prefix_hash=_digest("prefix"),
        sidecar=sidecar,
    )
    kv_caches = [
        torch.zeros(2, 2, 4, 8, 32, dtype=torch.bfloat16) for _ in range(target.num_layers)
    ]

    result = injector.retrieve_transform(request, kv_caches=kv_caches)

    assert result.success is True
    assert result.target_mooncake_puts == 0
    assert result.tokens_scattered == 3
    assert reader.calls == 1
    assert events == [("request-1", (0, 1))]
    assert tracker.valid == {0, 1}
    assert tracker.invalid == set()
    assert any(bool(layer.any()) for layer in kv_caches)

    rejected = injector.retrieve_transform(replace(request, sidecar=None), kv_caches=kv_caches)
    assert rejected.success is False
    assert rejected.source_read_attempted is False
    assert reader.calls == 1

    duplicate_slots = injector.retrieve_transform(
        replace(request, slot_mapping=(1, 1, 5)), kv_caches=kv_caches
    )
    assert duplicate_slots.fallback_reason == "invalid_retrieve_transform_request"
    assert duplicate_slots.source_read_attempted is False
    assert reader.calls == 1

    class BrokenGate:
        def evaluate(self, sidecar):
            raise KeyError("risk gate backend failed")

    injector.risk_gate = BrokenGate()
    gate_failure = injector.retrieve_transform(request, kv_caches=kv_caches)
    assert gate_failure.fallback_reason == "risk_gate_failure"
    assert gate_failure.source_read_attempted is False
    assert reader.calls == 1


def test_partial_paged_scatter_keeps_all_touched_blocks_invalid() -> None:
    source, target, _, transport = _transport()
    chunk = torch.randn(2, 3, 4, 2, 32, dtype=torch.bfloat16)
    payload = _tensor_bytes(chunk)
    reader = _Reader({"a": payload})
    tracker = InMemoryBlockValidityTracker()
    events = []

    def fail_after_scatter(target_kv, kv_caches, slots):
        scatter_paged_kv(target_kv, kv_caches, slots)
        raise RuntimeError("injected CUDA failure")

    injector = DirectPagedKVInjector(
        risk_gate=_gate(source, target),
        transport=transport,
        source_reader=reader,
        validity_tracker=tracker,
        publish_load_complete=lambda request_id, blocks: events.append((request_id, blocks)),
        scatter=fail_after_scatter,
    )
    request = RetrieveTransformRequest(
        request_id="partial",
        source_keys=("a",),
        source_checksums=(hashlib.sha256(payload).hexdigest(),),
        chunk_token_counts=(2,),
        slot_mapping=(1, 5),
        prefix_hash=_digest("prefix"),
        sidecar=_sidecar(source, target, chunk),
    )
    caches = [torch.zeros(2, 2, 4, 8, 32, dtype=torch.bfloat16) for _ in range(4)]

    result = injector.retrieve_transform(request, kv_caches=caches)

    assert result.success is False
    assert result.fallback_reason == "direct_injection_failed"
    assert result.invalidated_blocks == (0, 1)
    assert tracker.invalid == {0, 1}
    assert events == []


def test_publisher_failure_after_scatter_reinvalidates_all_touched_blocks() -> None:
    source, target, _, transport = _transport()
    chunk = torch.randn(2, 3, 4, 2, 32, dtype=torch.bfloat16)
    payload = _tensor_bytes(chunk)
    tracker = InMemoryBlockValidityTracker()

    def fail_publish(request_id, blocks):
        raise KeyError("publisher failed")

    injector = DirectPagedKVInjector(
        risk_gate=_gate(source, target),
        transport=transport,
        source_reader=_Reader({"a": payload}),
        validity_tracker=tracker,
        publish_load_complete=fail_publish,
    )
    request = RetrieveTransformRequest(
        request_id="publish-failure",
        source_keys=("a",),
        source_checksums=(hashlib.sha256(payload).hexdigest(),),
        chunk_token_counts=(2,),
        slot_mapping=(1, 5),
        prefix_hash=_digest("prefix"),
        sidecar=_sidecar(source, target, chunk),
    )
    caches = [torch.zeros(2, 2, 4, 8, 32, dtype=torch.bfloat16) for _ in range(4)]

    result = injector.retrieve_transform(request, kv_caches=caches)

    assert result.success is False
    assert result.fallback_reason == "direct_injection_failed"
    assert result.invalidated_blocks == (0, 1)
    assert tracker.valid == set()
    assert tracker.invalid == {0, 1}


def test_paged_scatter_supports_vllm_head_major_packed_layout() -> None:
    target = torch.arange(2 * 1 * 2 * 2 * 4, dtype=torch.bfloat16).reshape(2, 1, 2, 2, 4)
    # vLLM combined layout: [blocks, K/V, heads, head_dim/x, block, x].
    cache = torch.zeros(2, 2, 2, 2, 4, 2, dtype=torch.bfloat16)

    blocks = scatter_paged_kv(target, [cache], (1, 5))

    assert blocks == (0, 1)
    torch.testing.assert_close(cache[0, 0, :, :, 1, :].reshape(2, 4), target[0, 0, :, 0])
    torch.testing.assert_close(cache[1, 1, :, :, 1, :].reshape(2, 4), target[1, 0, :, 1])


def test_runtime_report_enforces_p95_cost_ttft_and_fallback_gates() -> None:
    report = build_selective_runtime_report(
        direction="qwen3_4b_to_8b",
        runtime_audit_dataset_sha256=_digest("runtime-audit"),
        audit_requests=512,
        warmup_iterations=20,
        materialization_ms=[50.0] * 100,
        native_prefill_ms=[100.0] * 100,
        accepted_native_ttft_ms=[200.0] * 100,
        accepted_reuse_ttft_ms=[130.0] * 100,
        rejected_native_ttft_ms=[100.0] * 100,
        rejected_fallback_ttft_ms=[104.0] * 100,
        accepted_target_mooncake_puts=0,
        backing_files_remaining=0,
    )
    evidence = runtime_cost_evidence_from_report(report, report_sha256=_digest("report"))

    assert report["eligible_for_approval"] is True
    assert evidence.p95_materialization_to_prefill_ratio == 0.5
    assert evidence.accepted_p95_ttft_reduction_pct == 35.0
    assert evidence.rejected_p95_fallback_overhead_pct == 4.0


def _quality(dataset_hash: str, *, total: int, accepted: int) -> AcceptedSubsetQualityEvidence:
    return AcceptedSubsetQualityEvidence(
        evaluation_dataset_sha256=dataset_hash,
        total_count=total,
        accepted_count=accepted,
        unsafe_count=0,
        coverage=accepted / total,
        native_task_score=0.99,
        bridge_task_score=0.985,
        task_score_drop_pct=(0.99 - 0.985) / 0.99 * 100,
        greedy_agreement=0.99,
        perplexity_drift_pct=1.0,
        regression_risk_upper_bound=clopper_pearson_upper_bound(0, accepted),
        key_cosine=0.5,
        value_cosine=0.5,
    )


def _v5_manifest() -> SelectiveKVBridgeManifest:
    source, target, transport, _ = _transport()
    risk = _risk_spec()
    validation_hash = _digest("validation")
    sealed_hash = _digest("sealed")
    manifest = SelectiveKVBridgeManifest(
        artifact_id="",
        direction="qwen3_4b_to_8b",
        source=source,
        target=target,
        transport=transport,
        risk_gate=risk,
        benchmark_manifest_sha256=_digest("benchmark"),
        transport_train_dataset_sha256=_digest("transport-train"),
        selector_train_dataset_sha256=_digest("selector-train"),
        method_dev_dataset_sha256=_digest("method-dev"),
        risk_calibration_dataset_sha256=_digest("risk-calibration"),
        validation_dataset_sha256=validation_hash,
        semantic_sealed_dataset_sha256=sealed_hash,
        runtime_audit_dataset_sha256=_digest("runtime-audit"),
        transport_quality=TransportQualityEvidence(
            evaluation_dataset_sha256=_digest("method-dev"),
            prompt_count=1024,
            task_score=0.96,
            oracle_safe_coverage=0.5,
            greedy_agreement=0.98,
        ),
        accepted_quality=_quality(validation_hash, total=2048, accepted=615),
    )
    return manifest.with_content_id()


def test_v5_state_machine_requires_semantic_then_runtime_evidence() -> None:
    candidate = _v5_manifest()
    assert candidate.validate() == []
    assert candidate.approved is False

    sealed_quality = _quality(candidate.semantic_sealed_dataset_sha256, total=2048, accepted=615)
    sealed = SemanticSealedEvidence(
        dataset_sha256=candidate.semantic_sealed_dataset_sha256,
        report_sha256=_digest("sealed-report"),
        sample_count=2048,
        code_sha256=_digest("code"),
        transport_weights_sha256=candidate.transport.weights_sha256,
        predictor_sha256=candidate.risk_gate.predictor_sha256,
        threshold=candidate.risk_gate.threshold,
        quality=sealed_quality,
    )
    semantic = replace(
        candidate,
        artifact_id="",
        state=ArtifactState.SEMANTIC_APPROVED,
        semantic_sealed=sealed,
    ).with_content_id()
    assert semantic.validate() == []
    assert semantic.semantic_approved is True
    assert semantic.approved is False

    runtime = RuntimeCostEvidence(
        report_sha256=_digest("runtime-report"),
        runtime_audit_dataset_sha256=semantic.runtime_audit_dataset_sha256,
        audit_requests=512,
        warmup_iterations=20,
        measured_iterations=100,
        p95_materialization_ms=50.0,
        p95_native_prefill_ms=100.0,
        p95_materialization_to_prefill_ratio=0.5,
        accepted_p95_ttft_reduction_pct=35.0,
        rejected_p95_fallback_overhead_pct=3.0,
    )
    injection = DirectInjectionEvidence(
        report_sha256=_digest("injection-report"),
        paged_slot_mapping_verified=True,
        load_complete_after_all_layers=True,
        partial_failure_invalidates_blocks=True,
        native_prefill_overwrites_invalid_blocks=True,
        accepted_target_mooncake_puts=0,
        backing_files_remaining=0,
        runtime_audit_passed=True,
    )
    approved = replace(
        semantic,
        artifact_id="",
        state=ArtifactState.APPROVED,
        runtime_cost=runtime,
        direct_injection=injection,
    ).with_content_id()

    assert approved.validate() == []
    assert approved.approved is True
    failed = replace(
        approved,
        artifact_id="",
        runtime_cost=replace(runtime, accepted_p95_ttft_reduction_pct=29.9),
    ).with_content_id()
    assert failed.approved is False
    assert any("below 30" in error for error in failed.validate())


def test_manifest_loader_dispatches_v5_without_changing_v4_reader(tmp_path: Path) -> None:
    manifest = _v5_manifest()
    path = tmp_path / "v5.json"
    manifest.save(path)

    loaded = load_cached_kv_manifest(path)

    assert isinstance(loaded, SelectiveKVBridgeManifest)
    assert loaded == manifest


def test_v5_manifest_rejects_legacy_pointwise_calibration_evidence() -> None:
    payload = _v5_manifest().to_dict()
    payload["risk_gate"].pop("calibration_method")
    payload["risk_gate"].pop("candidate_threshold_count")

    legacy = SelectiveKVBridgeManifest.from_dict(payload)
    errors = legacy.validate()

    assert any("Bonferroni-corrected" in error for error in errors)
    assert any("candidate threshold count" in error for error in errors)


def test_planner_executes_only_final_v5_state_and_exposes_retrieve_transform(
    tmp_path: Path,
) -> None:
    candidate = _v5_manifest()
    sealed_quality = _quality(candidate.semantic_sealed_dataset_sha256, total=2048, accepted=615)
    sealed = SemanticSealedEvidence(
        dataset_sha256=candidate.semantic_sealed_dataset_sha256,
        report_sha256=_digest("sealed-report"),
        sample_count=2048,
        code_sha256=_digest("code"),
        transport_weights_sha256=candidate.transport.weights_sha256,
        predictor_sha256=candidate.risk_gate.predictor_sha256,
        threshold=candidate.risk_gate.threshold,
        quality=sealed_quality,
    )
    semantic = replace(
        candidate,
        artifact_id="",
        state=ArtifactState.SEMANTIC_APPROVED,
        semantic_sealed=sealed,
    ).with_content_id()
    semantic_path = tmp_path / "semantic.json"
    semantic.save(semantic_path)
    request = ReuseRequest(
        source=model_ref_from_cached_spec(semantic.source),
        target=model_ref_from_cached_spec(semantic.target),
        prefix_hash=_digest("prefix"),
        calibration_id=semantic.artifact_id,
        artifact_uri=str(semantic_path),
    )

    semantic_plan = CrossModelReusePlanner().plan(request)

    assert semantic_plan.executable is False
    assert semantic_plan.fallback_reason == "artifact_not_approved"

    runtime = RuntimeCostEvidence(
        report_sha256=_digest("runtime-report"),
        runtime_audit_dataset_sha256=semantic.runtime_audit_dataset_sha256,
        audit_requests=512,
        warmup_iterations=20,
        measured_iterations=100,
        p95_materialization_ms=50.0,
        p95_native_prefill_ms=100.0,
        p95_materialization_to_prefill_ratio=0.5,
        accepted_p95_ttft_reduction_pct=35.0,
        rejected_p95_fallback_overhead_pct=3.0,
    )
    injection = DirectInjectionEvidence(
        report_sha256=_digest("injection-report"),
        paged_slot_mapping_verified=True,
        load_complete_after_all_layers=True,
        partial_failure_invalidates_blocks=True,
        native_prefill_overwrites_invalid_blocks=True,
        accepted_target_mooncake_puts=0,
        backing_files_remaining=0,
        runtime_audit_passed=True,
    )
    approved = replace(
        semantic,
        artifact_id="",
        state=ArtifactState.APPROVED,
        runtime_cost=runtime,
        direct_injection=injection,
    ).with_content_id()
    approved_path = tmp_path / "approved.json"
    approved.save(approved_path)
    approved_request = replace(
        request,
        calibration_id=approved.artifact_id,
        artifact_uri=str(approved_path),
    )

    plan = CrossModelReusePlanner().plan(approved_request)

    assert plan.executable is True
    assert plan.strategy.value == "selective_paged_kv_reuse"
    assert "lmcache_retrieve_transform" in plan.patch_hooks
    assert "direct_paged_kv_scatter" in plan.patch_hooks
    assert plan.as_metadata()["ge_artifact_state"] == "approved"


def _validation_receipt() -> ValidationGateReceipt:
    directions = tuple(
        DirectionValidationEvidence(
            direction=direction,
            passed=True,
            report_sha256=_digest(direction + "-report"),
            code_sha256=_digest("code"),
            transport_weights_sha256=_digest(direction + "-transport"),
            predictor_sha256=_digest(direction + "-predictor"),
            threshold_sha256=_digest(direction + "-threshold"),
        )
        for direction in (
            "qwen3_4b_to_8b",
            "qwen3_8b_to_4b",
            "qwen3_8b_to_14b",
            "qwen3_14b_to_8b",
        )
    )
    return ValidationGateReceipt(
        benchmark_manifest_sha256=_digest("benchmark"),
        validation_dataset_sha256=_digest("validation"),
        directions=directions,
    )


def test_semantic_sealed_guard_requires_four_directions_and_opens_once(tmp_path: Path) -> None:
    payload = b"sealed semantic samples"
    payload_path = tmp_path / "sealed.bin"
    payload_path.write_bytes(payload)
    receipt = _validation_receipt()
    guard = SemanticSealedGuard(tmp_path / "opened.json")

    opened = guard.open_once(
        payload_path,
        expected_payload_sha256=hashlib.sha256(payload).hexdigest(),
        receipt=receipt,
        expected_manifest_sha256=_digest("benchmark"),
        expected_validation_sha256=_digest("validation"),
    )

    assert opened == payload
    with pytest.raises(BenchmarkContractError, match="already opened"):
        guard.open_once(
            payload_path,
            expected_payload_sha256=hashlib.sha256(payload).hexdigest(),
            receipt=receipt,
            expected_manifest_sha256=_digest("benchmark"),
            expected_validation_sha256=_digest("validation"),
        )


def test_semantic_sealed_guard_acquires_marker_before_concurrent_read(
    tmp_path: Path, monkeypatch
) -> None:
    payload = b"sealed semantic samples"
    payload_path = tmp_path / "sealed.bin"
    payload_path.write_bytes(payload)
    guard = SemanticSealedGuard(tmp_path / "opened.json")
    receipt = _validation_receipt()
    original_read_bytes = Path.read_bytes
    reads = []
    reads_lock = threading.Lock()
    start = threading.Barrier(2)

    def tracked_read_bytes(path: Path) -> bytes:
        if path == payload_path:
            with reads_lock:
                reads.append(path)
            time.sleep(0.05)
        return original_read_bytes(path)

    def open_payload() -> bytes | str:
        start.wait()
        try:
            return guard.open_once(
                payload_path,
                expected_payload_sha256=hashlib.sha256(payload).hexdigest(),
                receipt=receipt,
                expected_manifest_sha256=_digest("benchmark"),
                expected_validation_sha256=_digest("validation"),
            )
        except BenchmarkContractError as exc:
            return str(exc)

    monkeypatch.setattr(Path, "read_bytes", tracked_read_bytes)
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(open_payload) for _ in range(2)]
        results = [future.result() for future in futures]

    assert results.count(payload) == 1
    assert sum("already opened" in str(result) for result in results) == 1
    assert reads == [payload_path]
    marker = json.loads(guard.marker_path.read_text(encoding="utf-8"))
    assert marker["state"] == "opened"


def test_immutable_sealed_report_is_atomically_published_once(tmp_path: Path) -> None:
    report = {"metric": 0.5, "passed": True}

    path = write_immutable_sealed_report(tmp_path, report)

    assert json.loads(path.read_text(encoding="utf-8")) == report
    assert path.stat().st_mode & 0o222 == 0
    assert not list(tmp_path.glob("*.tmp"))
    with pytest.raises(BenchmarkContractError, match="already exists"):
        write_immutable_sealed_report(tmp_path, report)


def test_publication_benchmark_enforces_registered_counts_and_group_isolation() -> None:
    sources = tuple(
        DatasetSource(
            dataset_id=dataset,
            revision="frozen-test",
            content_sha256=_digest(dataset),
            license_id="test-license",
            license_uri="https://example.invalid/license",
            source_uri="https://example.invalid/source",
            usage="trace_only" if dataset in {"sharegpt", "burstgpt"} else "semantic",
        )
        for dataset in sorted(REQUIRED_DATASETS)
    )
    records = []
    semantic_datasets = sorted(REQUIRED_DATASETS - {"sharegpt", "burstgpt"})
    runtime_datasets = sorted(REQUIRED_DATASETS)
    for split, count in SPLIT_COUNTS.items():
        for index in range(count):
            identity = f"{split}-{index}"
            datasets = runtime_datasets if split == "runtime_audit" else semantic_datasets
            group = f"transport-{index}" if split == "transport_train" else f"hot-{index % 64}"
            records.append(
                GroupedPrefixRecord(
                    sample_id=identity,
                    split=split,
                    dataset_id=datasets[index % len(datasets)],
                    prefix_group_id=group,
                    prefix_sha256=_digest(f"prefix-{group}"),
                    suffix_query_sha256=_digest(f"suffix-{identity}"),
                    content_sha256=_digest(f"content-{identity}"),
                    token_bucket=PREFIX_BUCKETS[index % len(PREFIX_BUCKETS)],
                    task="qa",
                )
            )
    provisional = PublicationBenchmarkManifest(
        sources=sources,
        records=tuple(records),
        split_sha256={},
        tokenizer_sha256=_digest("tokenizer"),
        chat_template_sha256=_digest("chat-template"),
        sealed_payload_sha256=_digest("sealed-payload"),
    )
    manifest = replace(
        provisional,
        split_sha256={split: provisional.compute_split_sha256(split) for split in SPLIT_COUNTS},
    )

    assert manifest.validate() == []
    leaked = replace(
        manifest,
        records=(
            replace(manifest.records[0], prefix_group_id="hot-0"),
            *manifest.records[1:],
        ),
    )
    assert any("overlap" in error for error in leaked.validate())
