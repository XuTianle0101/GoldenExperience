# Artifact Plan

## Reproduction Levels

1. Smoke test: run unit tests and synthetic benchmark without GPUs.
2. Single-GPU test: run Hugging Face adapter on a small model and verify offload behavior.
3. Multi-GPU serving test: run vLLM-backed workload with shared prefixes.
4. Paper reproduction: generate all main tables and figures from cached result files.

## Smoke Test

```bash
pip install -e ".[dev]"
pytest
golden-synthetic-benchmark --blocks 64 --tokens-per-block 128
```

## Expected Artifact Contents

- Source code and tests.
- Experiment configs under `configs/`.
- Scripts for synthetic and model-backed benchmarks.
- Frozen result JSON files under `artifacts/results/`.
- A table/figure generation script.
- Hardware notes for CPU, GPU, memory, and NVMe devices.

## Result Integrity

Every reported result should store:

- Git commit hash.
- Python and package versions.
- GPU model and driver.
- Dataset or prompt source.
- Model id and tokenizer id.
- Cache policy and quality gate thresholds.
- Raw per-request latency and quality metrics.

