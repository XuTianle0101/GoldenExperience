# Selective Cached-KV Manifest v5

Manifest v5 changes GoldenExperience from unconditional bridge materialization to
source-only, statistically calibrated admission followed by direct paged-KV injection.
Manifest v4 remains readable and keeps its original gates; v5 does not weaken or reinterpret
any v4 artifact.

## Execution Contract

```text
exact prefix + source/target identity
                 |
                 v
       source sidecar lookup (<= 4 KiB)
                 |
        calibrated risk gate
          | rejected        | accepted
          v                 v
   native target prefill   batched source read
   (no source KV read)          |
                               v
                 inverse source RoPE
                 layer/head mixing
                 per-head K/V low-rank + gated SiLU
                 target RoPE
                               |
                               v
                 direct vLLM paged-KV scatter
                               |
                 all layers/pages successful?
                    | no              | yes
                    v                 v
              blocks invalid     load-complete event
              native overwrite   decode may proceed
```

The runtime path never writes translated target objects to Mooncake. A failure after any
scatter keeps every touched block invalid; target native prefill must overwrite those slots
before decode.

## Implemented Surfaces

- `selective_manifest.py` defines `TransportSpec`, `RiskGateSpec`, accepted-subset quality,
  sealed evidence, runtime evidence, and the three-state authority model.
- `head_aware_transport.py` implements `[K/V, layer, head, token, head_dim]` transport,
  different source/target KV-head counts, source/target RoPE, trainable layer/head mixers,
  independent per-head K/V projections, attention losses, and the registered screening order.
- `attention_collection.py` captures at most 32 target query positions and 256 key positions
  per prompt, plus pre-output-projection attention outputs.
- `risk_gate.py` implements the compact sidecar, a fixed 128-D CountSketch, the 64-hidden-unit
  two-layer MLP, unsafe labels, exact one-sided Clopper-Pearson calibration, and the five
  selector baselines.
- `direct_paged_kv.py` implements `RETRIEVE_TRANSFORM`, gate-before-read behavior, batched
  exact reads, pinned asynchronous H2D on a dedicated CUDA stream, common vLLM page layouts,
  block invalidation, and atomic load-complete publication.
- `publication.py` enforces fixed split sizes, group isolation, hash-only sealed metadata,
  license/source provenance, four-direction validation receipts, one-shot sealed access, and
  immutable content-addressed sealed reports.
- `selective_runtime.py` recomputes P50/P95/P99 and the 0.70x materialization, 30% accepted
  TTFT, and 5% rejected-overhead gates from at least 20 warmups and 100 measurements.

## Artifact Authority

| State | Offline evaluation | Open semantic sealed split | Automatic runtime reuse |
| --- | --- | --- | --- |
| `validation_candidate` | yes | no | no |
| `semantic_approved` | yes | already completed once | no |
| `approved` | yes | already completed once | yes |

`semantic_approved` requires all four Qwen3 main directions to pass validation, an immutable
sealed report, and unchanged code/transport/predictor/threshold hashes. `approved` additionally
requires runtime cost evidence, a 512-request runtime audit, zero target Mooncake puts, zero
backing files, and verified partial-failure recovery.

## Fail-Closed Admission

The predictor only ranks risk. `select_calibrated_threshold` chooses the highest-coverage
threshold on the independent calibration split for which:

- at least 300 samples are accepted;
- the exact 95% one-sided Clopper-Pearson upper bound is at most 1%;
- tied predictor scores are admitted or rejected together.

At runtime, a missing/corrupt sidecar, unseen prefix, insufficient shadow history, OOD score,
model/tokenizer/transport identity change, predictor failure, or score above threshold falls
back before source KV is read. There is no unsafe production override.

## Benchmark Freeze

`golden-publication-benchmark freeze` consumes source provenance JSON plus hash-only record
JSONL and enforces the registered split sizes:

| Split | Samples |
| --- | ---: |
| transport train | 4096 |
| selector train | 2048 |
| method dev | 1024 |
| risk calibration | 2048 |
| validation | 2048 |
| semantic sealed test | 2048 |
| runtime audit | 512 |

Every split must cover prefix buckets 128, 512, 2048, and 8192. Transport-train prefix groups
cannot occur later. Risk-calibration, validation, and sealed data may share a hot prefix, but
their suffix/query hashes cannot overlap. ShareGPT and BurstGPT records are trace-only.

## Current Evidence Boundary

This repository contains the executable contracts and deterministic tests, but it does not
contain v5 model weights, the frozen publication dataset, calibration output, semantic sealed
results, or a real LMCache/vLLM runtime audit. Consequently there is no v5 `approved` artifact
and no production claim. The direct-injection module is an adapter surface guarded by final
manifest authority; it is not automatically enabled by the existing runtime patch script.

The retained Qwen3 8B-to-14B and 14B-to-8B rank-512 results remain deprecated development
evidence. They fail the existing quality and cost gates and cannot be promoted to v5 evidence.

