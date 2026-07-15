# GoldenExperience

[English](README.md) | [中文](README.zh-CN.md)

GoldenExperience is an artifact-first research framework for cross-model KV-cache translation on
the shared **vLLM + LMCache MP + Mooncake Store** serving substrate.

> **Research status:** the current publication result is a terminal negative result, not an
> approved deployment. The registered Qwen3-4B to Qwen3-8B v4 transport fit completed, but its
> method-development safety gate failed. Selector fitting, calibration, other directions,
> validation, semantic-sealed evaluation, and cross-model runtime auditing remain blocked.

[Full paper](paper/paper.md) | [Evidence package](artifacts/publication_v5/evidence/README.md) |
[Figures and data](artifacts/publication_v5/figures/README.md) |
[Pipeline contract](docs/v5_pipeline.md) | [Claim audit](docs/related_work_matrix.md)

## Result At A Glance

Nine registered candidates (ranks 32, 64, and 128 crossed with seeds 17, 29, and 43) trained for
three epochs on 4,096 prompts. The frozen rank aggregation selected rank 64; seed 17 remained the
deployment identity.

| Metric | Registered result | Required |
| --- | ---: | ---: |
| Task preservation | 0.976862 | Reported, not sufficient alone |
| 16-token greedy agreement | 0.617249 | At least 0.98 per safe prompt |
| Aggregate perplexity drift | 21.47% | At most 2% per safe prompt |
| Oracle-safe prompts | 142 / 1,024 | - |
| Oracle-safe coverage | **0.138672** | **At least 0.45** |
| All-nine post-hoc oracle | 377 / 1,024 = 0.368164 | Still below 0.45 |

Full-prefix supervision does help the mechanism: at fixed rank 128 and seed 17, safe count rises
from 115 to 159, with a net gain of 24 in the 8,192-token bucket. It does not make the fixed
low-rank affine operator behaviorally reliable across tasks.

![All candidates miss the gate](artifacts/publication_v5/figures/fig01_candidate_coverage.svg)

The complete method-dev report contains 9,216 measurements and has uncompressed SHA-256
`f35e9599cea4d56cb1d0a7fad888a7d1bf2cef2602c9f42950162de7662a4400`.

## What Is Implemented

GoldenExperience is not an inference engine and does not replace cache storage. It adds a narrow
control and evidence plane around existing serving components:

- **Cross-model planning:** model identity, KV topology, prefix binding, strategy selection, and
  fail-closed fallback for base/LoRA, scale-variant, and exploratory cross-base scenarios.
- **Head-aware transport:** RoPE-aware layer/head mixing and independent low-rank affine K/V maps,
  with train-only normalizers and ridge/SVD initialization.
- **Reproducible fitting:** grouped full-prefix supervision, deterministic rank/seed screening,
  atomic checkpoints with complete AdamW state, and independently replayable method-dev reports.
- **Evidence pipeline:** content-bound data/model/code identities, split-specific collection,
  artifact authority states, one-shot sealed guards, and stage dependencies that enforce the stop.
- **Selective-admission protocol:** source-only sidecars, a frozen risk predictor, exact calibrated
  bounds, validation, and selector baselines. These surfaces are tested but were not executed after
  the current method-dev failure.
- **Runtime integration:** LMCache MP secondary lookup, direct atomic scatter into vLLM paged KV,
  rollback, and no translated target-object publication. This implementation has no approved v4
  real-model runtime result.

Implementation capability and empirical authority are deliberately separate:

| Stage | Current state | What the repository may claim |
| --- | --- | --- |
| Transport collection and fit | Completed | Exact fitted candidates and provenance |
| Method development | **Failed** | Terminal negative result and mechanism diagnostics |
| Other-direction fits | Blocked | Implementation only |
| Selector and calibration | Blocked | Tested protocol only |
| Validation | Blocked | No `validation_candidate` |
| Semantic sealed | **Locked** | Payload remains unopened; no final-test estimate |
| Runtime audit | Blocked | No accepted-reuse or cross-model TTFT claim |

## Reproduce The Publication Artifacts

The paper tooling uses only the Python standard library and never needs the sealed payload.

```bash
# Works in a clean clone from the tracked deterministic report archive.
python3 paper/tools/build_method_dev_evidence.py --check --from-archive

# Rebuild every plotted CSV/SVG/PDF in memory and compare bytes.
python3 paper/tools/build_figures.py --check

# Check claims, numbers, references, links, hashes, and the locked workspace receipt.
python3 paper/tools/check_manuscript.py
```

Every generator rejects input and output paths containing `sealed`. The evidence archive
round-trips to the original 8,043,391-byte report, and every figure is tracked as CSV, accessible
SVG, and deterministic vector PDF.

## Install And Test

Create a Python 3.10+ environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -e ".[dev]"
```

Run the engineering checks:

```bash
pytest
ruff check goldenexperience tests scripts paper/tools
mypy goldenexperience tests scripts
python3 -m build
```

Run the planner demonstration:

```bash
python3 scripts/smoke_cross_model_plan.py
```

The planner output demonstrates control-plane behavior; it does not create a publication-v5
approved artifact.

## Architecture

```text
client -> vLLM OpenAI-compatible server
             |
             | LMCacheMPConnector
             v
      standalone LMCache MP server
             |
             | L2 adapter: type=mooncake_store
             v
      Mooncake Store on local TCP + SSD
             |
             | source lookup + sidecar gate + transform
             v
      GoldenExperience direct paged-KV materializer
             |
             +-- success: publish all translated layers atomically
             +-- failure: invalidate partial blocks and use native prefill
```

Runtime ownership remains narrow:

- vLLM owns model loading, scheduling, decoding, and inference correctness.
- LMCache MP owns shared KV lookup, offload, eviction, and prefetch orchestration.
- Mooncake Store owns persistent L2 metadata and objects across engine restarts.
- GoldenExperience owns cross-model identity, planning, translation, admission metadata,
  materialization, and fallback accounting.

## Repository Layout

```text
goldenexperience/
  benchmarks/       Frozen benchmark builders and deterministic scorers.
  cli/              Console entry points, including the publication-v5 pipeline.
  lmcache_patch/    Patch manifest and LMCache sidecar metadata.
  reuse/            ModelRef, KVShape, requests, plans, and scenario planner.
  runtime/          vLLM/LMCache/Mooncake checks, adapters, and baseline orchestration.
  size_variant/     Transport, fitting, risk, validation, sealed, and runtime contracts.
artifacts/
  publication_v5/  Receipts, negative-result evidence, CSV/SVG/PDF figures, diagnostics.
  kv_baseline/      Curated same-model substrate manifests; raw runs are ignored.
configs/            Runtime examples and frozen publication source identities.
docs/               Method preregistration, pipeline, data, design, and claim boundaries.
paper/              Full manuscript, bibliography, reproducibility and audit tools.
recipes/            Source-able runtime environment overlays.
scripts/            Thin operational and diagnostic launchers.
tests/              Unit and integration tests.
```

## Publication-v5 Protocol

The frozen benchmark separates fitting, development, selection, calibration, validation, final
semantic evaluation, and runtime measurement:

| Split | Rows | Current access |
| --- | ---: | --- |
| `transport_train` | 4,096 | Used for the registered fit |
| `method_dev` | 1,024 | Used; terminal gate failed |
| `selector_train` | 2,048 | Blocked |
| `risk_calibration` | 2,048 | Blocked |
| `validation` | 2,048 | Blocked |
| `semantic_sealed_test` | 2,048 | Locked and unopened |
| `runtime_audit` | 512 | Blocked |

Method-dev has now informed v2, v3, and v4. It cannot serve as an independent confirmation set for
another adaptive method. A future success claim requires changed code, a new content-bound
workspace, and a newly frozen development split. The current validation and semantic payloads must
not be opened to continue method design.

## Same-Model Serving Substrate

The repository also contains a real same-model offload/reuse baseline for validating vLLM,
LMCache MP, and Mooncake independently of cross-model quality. Install the pinned runtime stack:

```bash
./scripts/install_runtime.sh --mode package
```

Then launch the same-model baseline:

```bash
source recipes/kv_baseline_mooncake_local.env
GE_MODEL_PATH=Qwen/Qwen3-8B \
GE_PORT=30000 \
scripts/kv_baseline/run_vllm_lmcache_mooncake_kv_baseline.sh -- --tensor-parallel-size 1
```

Raw runs are written below `artifacts/kv_baseline/<run_id>/` and ignored by Git. Keep only curated
manifests under `artifacts/kv_baseline/manifests/`. A successful same-model baseline proves that the
storage substrate works; it is not evidence that v4 cross-model translation is safe or fast.

Use source mode only when patching upstream components:

```bash
./scripts/install_runtime.sh --mode source
```

The pinned package-mode compatibility matrix is `vllm==0.24.0` and `lmcache==0.4.6` on the verified
CUDA 13 stack. See the runtime scripts and `docs/shared_kv_substrate.md` before changing CUDA,
Python, or upstream revisions.

## Artifact Authority And Safety

Runtime loading is state-gated:

| Artifact state | Offline use | Open sealed split | Automatic cross-model reuse |
| --- | --- | --- | --- |
| `validation_candidate` | Yes | No | No |
| `semantic_approved` | Yes | Already completed once | No |
| `approved` | Yes | Already completed once | Yes |

There is currently no artifact in any of these three states for publication v5. A missing,
corrupt, mismatched, uncalibrated, or unapproved artifact always falls back to native target
prefill.

Do not inspect, sample, hash, or repurpose the semantic payload. Only the dedicated one-shot opener
may read it, and only after all four registered validation directions pass. That prerequisite is
not satisfied in the current workspace.

## Citation

Citation metadata is in [CITATION.cff](CITATION.cff). The preferred citation is the negative-result
paper, **"Can KV Caches Cross Model Scales? A Fail-Closed Evaluation of Qwen3 Prefix
Translation."** Please do not cite this version as evidence of an approved cross-model serving
speedup.

## License

GoldenExperience is released under the [Apache-2.0 license](LICENSE). Dataset redistribution is
subject to the upstream licenses recorded in `configs/publication_sources.qwen3-v5.json`.
