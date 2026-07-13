from __future__ import annotations

# ruff: noqa: E402
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

pytest.importorskip("vllm")

from goldenexperience.runtime.lmcache_retrieve_transform import (
    V5_AUDIT_CONNECTOR_CLASS,
    probe_runtime_stack,
)
from goldenexperience.size_variant.selective_manifest import ArtifactState
from goldenexperience.size_variant.v5_pipeline import (
    V5PipelineError,
    V5PipelineWorkspace,
)
from goldenexperience.size_variant.v5_real_runtime import (
    RealQwenRuntimeAuditEvaluator,
    _device_name,
    _perplexity_drift_pct,
    _PrefixAsset,
    _RequestContext,
)


def _evaluator(tmp_path: Path) -> RealQwenRuntimeAuditEvaluator:
    del tmp_path
    evaluator = object.__new__(RealQwenRuntimeAuditEvaluator)
    evaluator.seed = 17
    evaluator.llm = None
    evaluator.semantic_selective = None
    evaluator._block_size = None
    return evaluator


def _context() -> _RequestContext:
    stored = SimpleNamespace(source_ranges=((0, 64), (64, 128)))
    asset = _PrefixAsset(
        prefix_group_id="group",
        prefix_hash="0" * 64,
        token_ids=tuple(range(128)),
        stored=cast(Any, stored),
        base_sidecar=cast(Any, None),
        native_target_kv=None,
        transformed_target_kv=None,
        representative_suffix=(999,),
    )
    return _RequestContext(
        sample_id="sample",
        prompt_token_ids=asset.token_ids + (999,),
        sidecar_payload=b"sidecar",
        asset=asset,
        native_shadow_tokens=(),
        bridge_shadow_tokens=(),
    )


def _execution_evidence(*, success: bool) -> dict[str, Any]:
    request_blocks = list(range(10, 18))
    return {
        "success": success,
        "accepted": True,
        "fallback_reason": "none" if success else "direct_injection_failed",
        "source_read_attempted": True,
        "source_chunks_read": 2,
        "tokens_scattered": 128 if success else 0,
        "invalidated_blocks": [] if success else request_blocks,
        "request_blocks": request_blocks,
        "target_mooncake_puts": 0,
        "materialization_ms": 5.0,
        "load_complete_published": success,
        "worker_binding_matches": True,
        "partial_failure_injected": not success,
        "partial_failure_count": 0 if success else 1,
        "registered_layer_count": 2,
    }


def test_real_runtime_validates_success_and_partial_failure_telemetry(tmp_path: Path) -> None:
    evaluator = _evaluator(tmp_path)
    evaluator.semantic_selective = cast(
        Any,
        SimpleNamespace(
            state=ArtifactState.SEMANTIC_APPROVED,
            target=SimpleNamespace(num_layers=2),
        ),
    )
    evaluator._block_size = 16
    context = _context()

    assert evaluator._validate_success_evidence(_execution_evidence(success=True), context) == 5.0
    assert evaluator._validate_failure_evidence(
        _execution_evidence(success=False), context
    ) == tuple(range(10, 18))

    tampered = _execution_evidence(success=False)
    tampered["target_mooncake_puts"] = 1
    with pytest.raises(V5PipelineError, match="partial-failure"):
        evaluator._validate_failure_evidence(tampered, context)


class _FakeLLM:
    def __init__(self, output: Any) -> None:
        self.output = output
        self.calls: list[tuple[Any, Any, bool]] = []

    def generate(self, prompts: Any, sampling: Any, *, use_tqdm: bool) -> list[Any]:
        self.calls.append((prompts, sampling, use_tqdm))
        return [self.output]


def test_real_runtime_extracts_vllm_ttft_prefill_and_cache_counts(tmp_path: Path) -> None:
    metrics = SimpleNamespace(
        first_token_latency=0.012,
        first_token_ts=5.020,
        scheduled_ts=5.000,
    )
    output = SimpleNamespace(
        finished=True,
        outputs=[SimpleNamespace(token_ids=[7, 8])],
        metrics=metrics,
        num_cached_tokens=128,
    )
    evaluator = _evaluator(tmp_path)
    evaluator.llm = _FakeLLM(output)

    result = evaluator._run_vllm(
        tuple(range(129)),
        generation_tokens=2,
        kv_transfer_params={"connector": "params"},
    )

    assert result.token_ids == (7, 8)
    assert result.ttft_ms == pytest.approx(12.0)
    assert result.prefill_ms == pytest.approx(20.0)
    assert result.num_cached_tokens == 128
    sampling = evaluator.llm.calls[0][1]
    assert sampling.temperature == 0
    assert sampling.extra_args == {"kv_transfer_params": {"connector": "params"}}


def test_real_runtime_rejects_single_gpu_and_nonfinite_shadow_evidence(tmp_path: Path) -> None:
    with pytest.raises(V5PipelineError, match="distinct CUDA"):
        RealQwenRuntimeAuditEvaluator(
            workspace=cast(V5PipelineWorkspace, SimpleNamespace(root=tmp_path)),
            direction="qwen3_4b_to_8b",
            sample_store_path=tmp_path / "runtime.jsonl",
            source_path=tmp_path / "source",
            target_path=tmp_path / "target",
            source_device="cuda:0",
            target_device="cuda:0",
            identity_cache_path=None,
        )
    with pytest.raises(V5PipelineError, match="NLL"):
        _perplexity_drift_pct(float("nan"), 1.0, 16)


def test_real_runtime_refuses_vllm_after_parent_cuda_initialization(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import torch

    evaluator = _evaluator(tmp_path)
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: True)

    with pytest.raises(V5PipelineError, match="before parent-process CUDA"):
        evaluator._start_vllm()


def test_real_runtime_refuses_forked_vllm_worker(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import torch

    evaluator = _evaluator(tmp_path)
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: False)
    monkeypatch.setenv("VLLM_WORKER_MULTIPROC_METHOD", "fork")

    with pytest.raises(V5PipelineError, match="must be spawn"):
        evaluator._start_vllm()


def test_real_runtime_resolves_visible_cuda_device_without_torch_initialization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import subprocess

    import torch

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1,0")
    monkeypatch.setattr(
        torch.cuda,
        "get_device_name",
        lambda *_args, **_kwargs: pytest.fail("device lookup initialized CUDA"),
    )
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(
            stdout="0, NVIDIA A100-SXM4-80GB\n1, NVIDIA H100 80GB HBM3\n"
        ),
    )

    assert _device_name("cuda:0") == "NVIDIA H100 80GB HBM3"


def test_v5_runtime_stack_names_the_external_connector() -> None:
    stack = probe_runtime_stack(connector_class=V5_AUDIT_CONNECTOR_CLASS)

    assert stack.connector_class == V5_AUDIT_CONNECTOR_CLASS
    assert stack.validate() == []
