# Artifact Plan

## Reproduction Levels

1. Planner smoke test: run unit tests without SGLang, LMCache, GPUs, or model weights.
2. Runtime dependency check: verify SGLang and LMCache imports or local source paths.
3. Base/LoRA single-GPU test: run SGLang with LMCache and the GoldenExperience metadata patch.
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

GoldenExperience assumes SGLang and LMCache are installed before model-backed experiments.
Use upstream packages, local forks, or the convenience helper:

```bash
./scripts/bootstrap_runtime.sh
```

The helper is intentionally optional. Artifact scripts should record the exact upstream
commits or package versions used for SGLang and LMCache.

## Expected Artifact Contents

- Source code and tests for the planner and patch metadata.
- LMCache patch diff or fork commit.
- SGLang launch scripts and LMCache config files.
- Model pair manifests with `ModelRef` and `KVShape` fields.
- Calibration datasets or manifests for projection/translator paths.
- Size-variant artifacts: `CalibrationManifest`, `LayerMap`, `ProjectionSpec`, and
  `QualityGateResult` for each direction.
- Raw per-request latency, reuse, fallback, and quality logs.

## Result Integrity

Every reported run should store:

- GoldenExperience, SGLang, and LMCache commit hashes.
- Python, CUDA, GPU, and driver versions.
- Model ids, tokenizer ids, LoRA adapter ids, and revisions.
- Reuse scenario, strategy, transform id, confidence, and calibration id.
- Size-variant direction, pair id, layer map id, projection id, source/target config hash,
  accepted/rejected status, and fallback reason.
- Prefix source and prefix hash method.
- Raw latency, quality, accepted reuse, and fallback reason metrics.
