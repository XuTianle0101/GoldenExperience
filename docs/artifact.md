# Artifact Plan

## Reproduction Levels

1. Planner smoke test: run unit tests without vLLM, LMCache, Mooncake, GPUs, or model weights.
2. Runtime dependency check: verify vLLM, LMCache, and Mooncake imports/commands or local
   source paths.
3. Same-model KV baseline: run vLLM + LMCache MP + Mooncake Store and prove offload/reuse
   across an inference-engine restart.
4. Base/LoRA single-GPU test: run the GoldenExperience metadata patch against the shared
   LMCache MP substrate.
5. GoldenScale test: enable calibrated hidden-state bridge and target KV restore.
6. Cross-base exploratory test: enable only with calibration id and explicit task allowlist.

## Smoke Test

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -e ".[dev]"
pytest
golden-patch-manifest --output artifacts/patch_manifest.md
golden-scale-fit --direction bidirectional --output-dir /tmp/ge-golden-scale
golden-scale-validate /tmp/ge-golden-scale/qwen3_8b_to_14b_hidden_bridge_v0.json
```

## Runtime Stack

GoldenExperience assumes vLLM, LMCache, and Mooncake are installed before the recommended
same-model KV baseline. Use upstream packages, local forks, or the convenience helper:

```bash
./scripts/bootstrap_runtime.sh
```

The helper is intentionally optional. Artifact scripts should record the exact upstream
commits or package versions used for vLLM, LMCache, and Mooncake.

## Expected Artifact Contents

- Source code and tests for the planner and patch metadata.
- LMCache patch diff or fork commit.
- vLLM/LMCache MP/Mooncake launch scripts and generated adapter config files.
- Non-default diagnostic configs only when filesystem-L2 fallback comparisons are used.
- Model pair manifests with `ModelRef` and `KVShape` fields.
- Calibration datasets or manifests for hidden-bridge/projection/translator paths.
- Size-variant artifacts: `CalibrationManifest`, `LayerMap`, `HiddenBridgeSpec`, `KVRestoreSpec`,
  legacy `ProjectionSpec`, and `QualityGateResult` for each direction.
- Raw per-request latency, reuse, fallback, and quality logs.

## Result Integrity

Every reported run should store:

- GoldenExperience, vLLM, and LMCache commit hashes.
- Mooncake commit/package version and service logs.
- Python, CUDA, GPU, and driver versions.
- Model ids, tokenizer ids, LoRA adapter ids, and revisions.
- Reuse scenario, strategy, transform id, confidence, and calibration id.
- Size-variant direction, pair id, layer map id, projection id, source/target config hash,
  accepted/rejected status, and fallback reason.
- Prefix source and prefix hash method.
- Raw latency, quality, accepted reuse, and fallback reason metrics.
