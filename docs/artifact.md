# Artifact Plan

## Reproduction Levels

1. Planner smoke test: run unit tests without vLLM, LMCache, GPUs, or model weights.
2. Runtime dependency check: verify vLLM and LMCache imports or local source paths.
3. Base/LoRA single-GPU test: run vLLM with LMCache MP and the GoldenExperience metadata patch.
4. GoldenScale test: enable calibrated layer/head mapping and projection.
5. Cross-base exploratory test: enable only with calibration id and explicit task allowlist.

## Smoke Test

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -e ".[dev]"
pytest
golden-patch-manifest --output artifacts/patch_manifest.md
golden-scale-fit --direction bidirectional --output-dir /tmp/ge-golden-scale
golden-scale-validate /tmp/ge-golden-scale/qwen25_7b_to_14b_projection_v0.json
```

## Runtime Stack

GoldenExperience assumes vLLM and LMCache are installed before default model-backed experiments.
SGLang is optional for legacy control runs.
Use upstream packages, local forks, or the convenience helper:

```bash
./scripts/bootstrap_runtime.sh
```

The helper is intentionally optional. Artifact scripts should record the exact upstream
commits or package versions used for vLLM, LMCache, and optional SGLang legacy runs.

## Expected Artifact Contents

- Source code and tests for the planner and patch metadata.
- LMCache patch diff or fork commit.
- vLLM/LMCache MP launch scripts, LMCache config files, and optional SGLang legacy scripts.
- Model pair manifests with `ModelRef` and `KVShape` fields.
- Calibration datasets or manifests for projection/translator paths.
- Size-variant artifacts: `CalibrationManifest`, `LayerMap`, `ProjectionSpec`, and
  `QualityGateResult` for each direction.
- Raw per-request latency, reuse, fallback, and quality logs.

## Result Integrity

Every reported run should store:

- GoldenExperience, vLLM, LMCache, and optional SGLang legacy commit hashes.
- Python, CUDA, GPU, and driver versions.
- Model ids, tokenizer ids, LoRA adapter ids, and revisions.
- Reuse scenario, strategy, transform id, confidence, and calibration id.
- Size-variant direction, pair id, layer map id, projection id, source/target config hash,
  accepted/rejected status, and fallback reason.
- Prefix source and prefix hash method.
- Raw latency, quality, accepted reuse, and fallback reason metrics.
