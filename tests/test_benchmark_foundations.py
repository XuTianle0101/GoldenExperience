import json

from goldenexperience.benchmarks.harness import BenchmarkHarness
from goldenexperience.benchmarks.metrics import BenchmarkRecord, percentile, summarize
from goldenexperience.benchmarks.scenarios import (
    AGENT_WORKFLOW,
    LONG_CONTEXT_QA,
    MULTI_TURN_CHAT,
    RAG_PREFIX_SHARING,
)
from goldenexperience.benchmarks.synthetic import run_synthetic_benchmark
from goldenexperience.cross_model_mapper.quality_gate import QualityGate


def test_metrics_cover_empty_singleton_and_interpolated_samples() -> None:
    empty = summarize("empty", [])
    assert empty.count == 0
    assert empty.throughput_ops_per_s == 0.0
    assert percentile([], 95) == 0.0
    assert percentile([4.0], 95) == 4.0
    assert percentile([1.0, 3.0], 50) == 2.0

    summary = summarize(
        "lookup",
        [
            BenchmarkRecord("lookup", 1.0, bytes_size=10, cache_hit=True),
            BenchmarkRecord("lookup", 3.0, bytes_size=20, cache_hit=False),
        ],
    )
    assert summary.count == 2
    assert summary.mean_ms == 2.0
    assert summary.bytes_total == 30
    assert summary.hit_rate == 0.5
    assert summary.throughput_ops_per_s == 500.0


def test_harness_records_exports_and_resets(tmp_path) -> None:
    harness = BenchmarkHarness("unit")
    assert harness.time_operation("hit", lambda: {"value": 7}, bytes_size=16) == {"value": 7}
    assert harness.time_operation("miss", lambda: None, metadata={"attempt": 1}) is None

    summaries = harness.summary()
    assert list(summaries) == ["hit", "miss"]
    assert summaries["hit"].hit_rate == 1.0
    assert summaries["miss"].hit_rate == 0.0

    output = tmp_path / "nested" / "benchmark.json"
    harness.export_json(output, extra={"commit": "abc"})
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["experiment"] == "unit"
    assert payload["extra"] == {"commit": "abc"}
    assert len(payload["records"]) == 2

    harness.reset()
    assert harness.records == []
    assert harness.summary() == {}


def test_registered_scenarios_and_quality_gate_contracts() -> None:
    scenarios = (LONG_CONTEXT_QA, MULTI_TURN_CHAT, RAG_PREFIX_SHARING, AGENT_WORKFLOW)
    assert {scenario.name for scenario in scenarios} == {
        "long_context_qa",
        "multi_turn_chat",
        "rag_prefix_sharing",
        "agent_workflow",
    }
    assert all(scenario.prompt_tokens > scenario.decode_tokens for scenario in scenarios)
    assert all(0.0 <= scenario.shared_prefix_ratio <= 1.0 for scenario in scenarios)

    gate = QualityGate(min_quality_score=0.95, max_perplexity_drift=0.02)
    assert gate.accepts(0.95)
    assert gate.accepts(0.99, perplexity_drift=0.02)
    assert not gate.accepts(0.94, perplexity_drift=0.0)
    assert not gate.accepts(0.99, perplexity_drift=0.021)


def test_synthetic_benchmark_reports_all_operation_classes(tmp_path) -> None:
    result = run_synthetic_benchmark(
        blocks=8,
        tokens_per_block=4,
        head_dim=8,
        hbm_capacity_bytes=1 << 20,
        cpu_capacity_bytes=1 << 20,
        nvme_path=tmp_path,
    )

    summaries = result["summaries"]
    assert isinstance(summaries, dict)
    assert summaries["put"]["count"] == 8
    assert summaries["get_promote"]["count"] == 8
    assert summaries["get_promote"]["hit_rate"] == 1.0
    assert summaries["prefetch_batch"]["count"] == 1
    assert set(result["tier_states"]) == {"hbm", "cpu", "nvme"}
