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
                 train-only z-score normalization
                 per-head K/V low-rank affine
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
  independent per-head K/V affine maps, train-only ridge/SVD initialization, attention losses,
  and the registered screening order. The legacy v1 gated-SiLU tensor contract remains readable.
- `attention_collection.py` captures at most 32 target query positions and 256 key positions
  per prompt, plus pre-output-projection attention outputs.
- `risk_gate.py` implements the compact sidecar, a fixed 128-D CountSketch, the 64-hidden-unit
  two-layer MLP, unsafe labels, exact one-sided Clopper-Pearson calibration, and the five
  selector baselines.
- `direct_paged_kv.py` implements `RETRIEVE_TRANSFORM`, gate-before-read behavior, batched
  exact reads, pinned asynchronous H2D on a dedicated CUDA stream, common vLLM page layouts,
  block invalidation, and atomic load-complete publication.
- `lmcache_retrieve_transform.py` binds that path to the LMCache MP 0.4.6
  non-GPU `PREPARE_STORE`/`COMMIT_STORE` and
  `LOOKUP`/`QUERY_PREFETCH_STATUS`/`PREPARE_RETRIEVE`/`COMMIT_RETRIEVE` protocols, restores
  source-model CPU chunks to explicit head layout, preserves standard LMCache connector metadata,
  and forwards failed page ids to vLLM 0.24.0's native-recompute contract.
- `vllm_retrieve_transform_connector.py` is the audit-only external vLLM connector. Its scheduler
  re-evaluates the frozen CPU gate before creating source metadata; its worker orders every target
  layer explicitly, injects synchronously, and reports invalid blocks through vLLM's recompute API.
- `runtime_audit_telemetry.py` carries gate and worker observations over authenticated loopback
  sockets, without filesystem staging or translated target-cache objects.
- `lmcache_mp_server.py` owns the audit's loopback LMCache MP process, fixes 128-token non-GPU
  chunks and memory-only L1 storage, and rejects any filesystem residue at shutdown.
- `publication.py` enforces fixed split sizes, group isolation, hash-only sealed metadata,
  license/source provenance, four-direction validation receipts, one-shot sealed access, and
  immutable content-addressed sealed reports.
- `selective_runtime.py` recomputes P50/P95/P99 and the 0.70x materialization, 30% accepted
  TTFT, and 5% rejected-overhead gates from at least 20 warmups and 100 measurements. Its
  evidence records the fixed isolated paired-request protocol and cannot claim arrival replay.
- `real_model_smoke.py` executes a bounded Qwen3 source/target prefill, target-attention
  capture, DynamicCache conversion, and five-term transport objective. Its schema hard-codes
  `diagnostic_only`, `evidence_eligible=false`, and `sealed_split_accessed=false`.
- `v5_pipeline.py` provides the four-direction immutable config, lock-serialized resumable
  state, stable dependency bindings, and atomic content-addressed object/receipt store. The
  workspace contract is detailed in `docs/v5_pipeline.md`.
- `v5_collect.py` validates one raw split at a time, runs real source/target prefill, writes
  one bounded immutable safetensors KV/query/attention shard per unique prefix group, and
  resumes from fully verified per-sample checkpoints. Generic collection cannot name or load
  the semantic sealed split.
- `v5_fit.py` fits the fixed 3-rank by 3-seed 4B-to-8B screening matrix in one synchronized
  trace pass, verifies and deserializes each shared shard once, fits a row-weighted train-only
  ridge initializer, batches every candidate through frozen-target suffix-logit distillation,
  checkpoints the initializer plus model and AdamW state atomically, and emits runtime-loadable
  candidate weights. Its production entry point cannot override the registered matrix or
  generation-supervision contract.
- `v5_generation.py` binds the exact transport-train raw store, retains absolute positions under
  deterministic head-tail suffix bounding, generates a 16-token native sampled-cache teacher,
  and computes candidate-batched greedy-token CE and teacher-logit KL with gradients flowing only
  through transformed past KV.
- `publication_eval.py`, `v5_method_dev.py`, and `v5_real_method_dev.py` provide explicit
  deterministic semantic scorers, real shared-prefix evaluation of all nine candidates,
  resumable per-sample evidence, three-seed rank aggregation, and a seed-17 frozen structure
  receipt for downstream directions.
- `v5_directional_fit.py` consumes that global receipt to train exactly one selected-rank,
  seed-17 deployment transport for each remaining Qwen3 direction; the workspace also binds
  this cross-direction dependency before a fit lease can be issued.
- `v5_sealed.py` performs the global one-shot transition only after replaying all four passing
  validations and publishes the immutable, content-addressed sealed snapshot and open receipt.
- `v5_semantic.py` evaluates that snapshot independently for each direction, resumes from
  token-identity-bound checkpoints, replays every probability/history/decision/aggregate on
  load, and grants only `semantic_approved` authority.
- `v5_runtime.py` runs the fixed 512-row paired latency audit, verifies every checkpoint and
  aggregate on reload, binds the pinned runtime source identity and failure-recovery probe, and
  grants `approved` authority only when both accepted and rejected paths meet their gates.

Trace sharing is restricted to prefix-derived tensors. Each direction and split still carries
one ordered manifest record and checkpoint per benchmark row, including its independent sample,
suffix/query, content, history, label, and score bindings. Method development and risk evaluation
reuse source/target prefix prefill only while equal prefix groups are contiguous; transforms and
timing measurements, causal history updates, semantic scoring, labels, and row weights remain
independent for every row. Thus grouped execution changes storage and redundant prefill cost, not
the frozen sample order or statistical estimand.

Transport v2 removes v1's diagonal base path, sigmoid gate, and SiLU residual. For each K/V
layer-head pair it computes `z = (mixed_source - mean) / scale`, `latent = z @ down`, and
`target_unrotated = latent @ up + bias`; keys are inverse-rotated before mixing and target-rotated
after the affine map. Equal-depth/equal-head model pairs start from an identity layer/head mapping,
which is the topology used by the successful full-affine development diagnostic. Unequal depths
start from deterministic two-layer linear interpolation inside the registered three-layer window.

The initializer reads only `transport_train`. Shared trace objects are evaluated once but multiplied
by their number of frozen manifest rows, so grouping changes I/O rather than the training estimand.
An augmented full-rank ridge solve uses ratio `1e-3`; the raw-space solution is converted to the
train-only normalized coordinate system and stored as immutable per-head SVD tensors. Ranks
32/64/128 are exact truncations of that stored decomposition. Seed 17 uses the canonical factors;
seeds 29 and 43 use deterministic orthogonal latent rotations that preserve the initial affine
function while allowing AdamW's coordinate-wise moments to explore distinct training paths.

Run the implementation smoke independently of every benchmark split:

```bash
golden-v5-smoke --output artifacts/cache/qwen3_4b_to_8b_smoke.json
```

The smoke output, bounded one-shard fit diagnostic, and single-prompt method-dev backend
diagnostic prove only that the local code/model stack executes with finite tensors,
restorable optimizer state, and real target generation. They are not transport quality,
validation, calibration, sealed-test, runtime-audit, or approval evidence.

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
- the family-wise exact 95% one-sided Clopper-Pearson upper bound is at most 1%;
- the pointwise confidence is Bonferroni-adjusted over every eligible distinct threshold,
  and the correction method plus candidate count are stored in the artifact;
- tied predictor scores are admitted or rejected together.

Predictor fitting is independently frozen before calibration. Each direction uses exactly
2,048 `selector_train` rows, the method-dev-selected deployment transport, and a fixed
169-to-64-to-1 MLP. Features are extracted from the deserialized quantized source sidecar;
native-target execution contributes labels only and cannot enter the current row's feature
vector. Per-prefix history contains only outcomes from lexicographically earlier rows in the
frozen split. The fit artifact has no threshold and is rejected if either label class is
absent. Calibration then freezes the predictor, reads only `risk_calibration`, scores on CPU,
and fails rather than publishing when no eligible threshold meets the simultaneous bound.
Reloading the artifact recomputes every per-row score, causal history, tied candidate group,
Bonferroni correction, and selected threshold from the detailed immutable report.

Independent validation then runs the complete gate, including one prior shadow sample and
the OOD cutoff, on all 2,048 rows in each direction. It reports accepted-subset semantic
quality and the fixed no-selector, cosine-0.95, MLP-0.5, calibrated, and oracle baselines. A
direction remains a `validation_candidate` unless all registered coverage, semantic, token,
perplexity, and exact-risk gates pass. Validation never grants sealed-test or runtime
authority.

The semantic split is opened by a global one-shot guard only after all four detailed
validation candidates are replayed successfully. The guard claims an exclusive marker before
reading, checks the configured payload hash and every hash-only sample binding, then publishes
one immutable snapshot and a receipt binding all direction-specific transport, predictor, and
threshold identities. A failed opening cannot be retried; successful opening permits only the
frozen semantic evaluation, not production reuse.

Each direction then consumes only that immutable snapshot through a dedicated guarded stage.
The sealed split has no fabricated trace artifact: its exact tokenizer-derived prefix identity
is represented by only sample id, registered token count, and token-id hash. Semantic evaluation
uses the validation gate order and reports the accepted subset plus all five selectors. Loading
the result independently recomputes token hashes, probabilities, causal histories, decisions,
quality, and baselines. Passing creates `SemanticSealedEvidence` and a `semantic_approved`
manifest, while runtime cost/direct-injection fields remain absent and automatic reuse remains
forbidden.

At runtime, a missing/corrupt sidecar, unseen prefix, insufficient shadow history, OOD score,
model/tokenizer/transport identity change, predictor failure, or score above threshold falls
back before source KV is read. There is no unsafe production override.

The direct bridge is pinned to LMCache 0.4.6, vLLM 0.24.0, and Torch 2.11.0. Before an audit it
checks the exact connector methods, non-GPU MP protocol members, external-connector loading
surface, and vLLM invalid-block native-recompute path. It records content hashes for ten
upstream source files and rechecks that identity after measurement. A version string alone is
not accepted as compatibility evidence.

Full source prefixes are first written to the same LMCache MP server under the source model's key
identity. The writer uses exact fixed-size CPU chunks, records each checksum before storing, and
publishes no translated target object. The bridge then registers a bounded non-GPU read context,
issues exact one-chunk prepare/commit operations, rejects missing, oversized, wrong-shape,
wrong-dtype, or wrong-checksum payloads, and reshapes `[K/V, layer, token, heads*dim]` into the
transport's explicit head layout. The registered v5 experiment uses one source worker and one
target worker per direction; tensor
parallel source layouts are deliberately rejected. No filesystem staging or translated target
object is part of this path.

The reusable low-level bridge preserves ordinary metadata when it wraps an upstream LMCache
connector. The audit-only external connector instead has no target-store path at all. In both
cases, all target blocks are marked invalid before the first scatter, every layer must finish
before one load-complete publication, and any read, transform, scatter, synchronization, or
publication failure returns the invalid block ids via vLLM's
`get_block_ids_with_load_errors`. vLLM then discards that step's output and natively recomputes
from the first invalid block. This recovery behavior is measured again in the final runtime audit
rather than inferred only from source inspection.

The audit connector receives only a read-only `semantic_approved` artifact plus immutable
workspace objects. The scheduler deserializes and rechecks the source sidecar, compares its gate
result with the orchestrator's frozen decision, verifies exact prefix token IDs, and only then
allocates external-prefix slots. The reusable LMCache source-store request ID remains separate from
the per-row audit and vLLM target request IDs. Rejected rows produce an authenticated gate event and
no worker load metadata. Accepted rows are re-gated on the worker before LMCache lookup; success and
failure observations are authenticated back to the audit process. A dedicated failure probe writes
the first target layer and then raises, so the reported block set must be invalidated and natively
recomputed before vLLM may return a token.

Accepted loads use vLLM's asynchronous pre-forward receive handshake even though the LMCache read
and transform are completed synchronously inside the isolated worker. The worker returns the load
result, invalid block IDs, and `finished_recving` in one engine step; only a successful load can
advance to target inference. A failed partial write is therefore invalidated before any logits are
consumed, and the same request is rescheduled from native target prefill. The failure probe requires
its 16-token recovered continuation to equal an independently measured native continuation exactly.

The real audit backend starts a managed loopback-only, memory-only LMCache MP server and authenticated
telemetry collector, then launches vLLM with a fixed `spawn` worker before any parent-process CUDA
initialization. Prefix caching, chunked prefill, hybrid KV management, asynchronous scheduling,
tensor/pipeline parallelism, and target-side KV stores are disabled; eager execution and one request
at a time are enforced. Only after the vLLM worker is isolated does the parent open the source writer
and the two Transformers models. Four source prefixes (128, 512, 2,048, and 8,192 tokens) are stored
once and reused by the registered 512-row audit without filesystem staging.

After one direction has completed semantic evaluation, run its bound runtime split with:

```bash
golden-v5-pipeline audit-runtime \
  --workspace artifacts/publication-v5 \
  --direction qwen3_4b_to_8b \
  --samples artifacts/cache/publication_benchmark_v5_bound/raw/runtime_audit.jsonl \
  --source-device cuda:0 \
  --target-device cuda:1 \
  --resume
```

The command rechecks the workspace source-tree hash before opening any model. It exposes only
operational LMCache memory/worker/time-out settings, and records all of them in the stage binding.
Quality thresholds, sample counts, warmup and measurement counts, connector failure policy, and
approval gates have no command-line override.

The 512-request approval audit is an isolated paired latency experiment in lexicographic sample-id
order. For every accepted row it compares native target prefill/TTFT with direct reuse; for every
rejected row it compares native TTFT with fail-closed fallback, with at least 100 measurements on
both paths. BurstGPT timestamps remain bound through the raw-store and trace hashes, but this audit
does not sleep, queue, or issue concurrent requests according to those timestamps. It therefore
supports request-latency and fallback-cost claims only, not serving throughput, queueing, or burst
scalability claims. Any timestamp-scaled concurrent replay is separate workload evidence and must
be reported separately rather than inferred from an `approved` artifact.

Because ShareGPT and BurstGPT runtime rows are trace-only, the runtime report never invents task
references or task scores. Causal history is updated from a separately named reference-free shadow
observation: native and bridged 16-token continuations are compared only for greedy agreement and
teacher-forced perplexity drift, using the registered 0.98 and 2% failure cutoffs. These shadow
outcomes drive later history features but are not semantic accuracy evidence.

`tokenizer_sha256` identifies token-ID semantics: tokenizer model/vocabulary files, merges,
special tokens, and semantic tokenizer configuration. Prompt serialization is separate;
`chat_template_sha256` preserves the exact default chat template for provenance. This split
does not permit reuse across different requests: the source sidecar and target request must
still carry the same exact prefix hash, so any rendered-token difference fails before source
KV is read.

## Benchmark Freeze

`golden-publication-benchmark freeze` consumes source provenance JSON plus hash-only record
JSONL. `--tokenizer-model` must point to the canonical model directory; the command derives
the complete token-ID semantic hash and the separate chat-template hash rather than hashing
one tokenizer file. It then enforces the registered split sizes:

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

The real-data builder, frozen upstream revisions, exact per-source hashes, balanced allocation,
license notes, and sealed-output procedure are specified in `docs/publication_dataset.md`. The
legacy `freeze` command remains available for externally prepared records; the publication v5 path
uses `golden-publication-benchmark audit-sources` followed by `build` so source bytes and split
isolation are checked before the manifest is emitted.

## Current Evidence Boundary

The registered 4B-to-8B v4 screening fit is complete. Its nine candidates each consumed the
full 4,096-row transport split for three epochs, and its final model/optimizer checkpoint and
runtime weights were independently reloaded. The immutable fit evidence is recorded in
`artifacts/publication_v5/stages/qwen3_4b_to_8b.fit_transport.v4.json`.

The complete 1,024-prompt, nine-candidate method-dev evaluation also ran, but it failed the
unchanged admission gate. Registered rank selection chose rank 64; its seed-17 deployment
candidate achieved only `0.138671875` oracle-safe coverage against the required `0.45`.
Even the per-row union of all nine candidates covered only `0.3681640625`. The detailed
failure record and mechanism analysis are retained under `artifacts/publication_v5/` and bind
all 9,216 measurements to report SHA-256
`f35e9599cea4d56cb1d0a7fad888a7d1bf2cef2602c9f42950162de7662a4400`.

Consequently there is no frozen transport structure, no v5 `validation_candidate`, and no
`semantic_approved` or `approved` artifact. The selector, calibration, other-direction fit,
validation, semantic-sealed, and runtime stages are protocol descriptions backed by tests,
not executed evidence for this workspace. The semantic payload remains locked and automatic
cross-model reuse remains disabled. The direct-injection module is an adapter surface guarded
by final manifest authority; it is not enabled by the existing runtime patch script.

The retained Qwen3 8B-to-14B and 14B-to-8B rank-512 results remain deprecated development
evidence. They fail the existing quality and cost gates and cannot be promoted to v5 evidence.
