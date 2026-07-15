# Paper Outline

## Working Title

**Can KV Caches Cross Model Scales? A Fail-Closed Evaluation of Qwen3 Prefix
Translation**

Alternative titles:

- When Cross-Model KV Reuse Fails: Full-Prefix Distillation Under Behavioral Gates
- Beyond Cache Similarity: An Auditable Negative Study of Cross-Scale KV Translation

## Paper Status

This is a negative-results paper, not a successful deployment paper. The registered
Qwen3 4B-to-8B transport fit completed, but the independent method-development gate failed.
No selector, calibration, other-direction, validation, semantic-sealed, or runtime stage is
authorized in the recorded workspace. The paper must preserve that boundary in its title,
abstract, figures, tables, and artifact claims.

The semantic sealed split remains locked. The complete method-dev report contains 9,216
measurements over 1,024 prompts and nine candidates. Its SHA-256 is
`f35e9599cea4d56cb1d0a7fad888a7d1bf2cef2602c9f42950162de7662a4400`.

## Abstract

Cross-model KV-cache translation promises to skip target-model prefix prefill, but tensor
similarity and average task scores do not establish that a translated cache preserves decoded
behavior. We build an auditable, fail-closed evaluation stack for cross-scale Qwen3 prefix
replacement. It binds exact model, tokenizer, data, code, optimizer, and artifact identities;
separates transport training, method development, calibration, validation, sealed testing,
and runtime approval; and prevents later stages from running when an earlier behavioral gate
fails.

We evaluate a head-aware low-rank affine transport from Qwen3-4B to Qwen3-8B. Nine registered
candidates (ranks 32/64/128 and seeds 17/29/43) train for three epochs on 4,096 prompts. To
correct a sampled-cache teacher mismatch discovered in an earlier method, the final variant
distills 16 target tokens while both teacher and student consume complete 128- to 8,192-token
prefix caches. The full-prefix correction improves rank-128/seed-17 safe coverage from
`115/1024` to `159/1024`, including 24 additional safe 8,192-token prompts.

The correction is insufficient. Registered rank aggregation selects rank 64, whose fixed
deployment seed achieves task preservation `0.9769` but only `0.6172` greedy-token agreement,
`21.47%` perplexity drift, and `142/1024 = 0.1387` oracle-safe coverage, below the
preregistered `0.45` gate. Even a prohibited per-prompt oracle over all nine candidates covers
only `377/1024 = 0.3682`. Function-calling prompts are often safe, while grade-school math and
code generation largely fail. We therefore stop before calibration, sealed data, and runtime
claims. The result isolates a useful boundary: aligning the training teacher with the complete
deployment prefix fixes part of the long-context error, but a fixed low-rank affine KV map does
not reliably preserve cross-task decoded behavior across model scales.

## Contributions

1. **A fail-closed evidence protocol.** The workspace makes data splits and artifact authority
   executable: a failed method-dev gate cannot create a frozen structure or reach calibration,
   validation, sealed testing, or production approval.
2. **A reproducible full-prefix training implementation.** It reconstructs complete source and
   target caches on separate GPUs, uses differentiable target-token supervision, activation
   checkpointing, deterministic grouped optimization, and atomically resumable model plus AdamW
   state for all nine candidates.
3. **A complete negative evaluation.** The study reports all 9,216 method-dev measurements,
   all ranks and seeds, the registered deployment choice, and the stronger nine-candidate safe
   union rather than selecting a favorable candidate after observing results.
4. **A mechanism result.** Full-prefix teacher alignment substantially improves the longest
   prefix bucket at fixed rank and seed, yet all prefix lengths remain far below the gate and
   decoded behavior varies sharply by task.
5. **An artifact boundary for systems claims.** Direct vLLM paged materialization, rollback,
   and LMCache integration are implemented and tested, but the paper does not report approved
   runtime speedups because transport quality blocked the runtime experiment.

Transport itself is not claimed as a standalone novelty. Cross-model KV translation,
cross-size cache reuse, target-prefill skipping, head-aware mapping, and cache distillation
all have prior art. Semantic Cache Distillation and Latent Cache Flow are the closest learned
translation/communication precedents; Activated LoRA, PrefillShare, and ICaRus additionally
establish exact cache sharing in real vLLM paths by co-designing the models around a shared
cache producer. Our narrower contribution is the conjunction of an exact split/identity
protocol, strict free-running behavior gates, a complete full-prefix intervention, and an
honestly terminal negative result.

## Research Questions

- **RQ1:** Does making the train-time teacher and student consume the complete deployment
  prefix improve behavior over sampled-prefix target-logit distillation?
- **RQ2:** Can any registered rank or seed meet the `0.45` oracle-safe coverage gate without
  changing the runtime operator or thresholds?
- **RQ3:** Which failure criteria, tasks, and prefix lengths account for unsafe reuse?
- **RQ4:** Do high average task-preservation scores imply token-level behavioral safety?
- **RQ5:** Which system claims remain justified when the quality gate stops the pipeline before
  calibration and runtime evaluation?

## Experimental Design

### Models and direction

- Screening direction: Qwen3-4B -> Qwen3-8B.
- Both models use 36 layers, eight KV heads, head dimension 128, and bfloat16 caches.
- The implementation supports unequal KV-head counts, but this experiment does not evaluate
  them. No unequal-head empirical claim is permitted.
- Other registered directions are intentionally not fit after the screening failure.

### Data and isolation

- `transport_train`: 4,096 prompts, the only split used to fit normalizers, the ridge/SVD
  initializer, and transport parameters.
- `method_dev`: 1,024 independent prompts, used for rank aggregation and the fixed stop gate.
- Prefix buckets: 128, 512, 2,048, and 8,192 tokens, 256 method-dev prompts each.
- Tasks: function calling, grade-school math, competition math, long-context QA, and Python
  code generation.
- Selector, calibration, validation, semantic sealed, and runtime payloads are unavailable to
  fitting and method selection. The semantic payload remains unopened after failure.

### Registered candidates and optimization

- Ranks: 32, 64, 128.
- Seeds: 17, 29, 43; seed 17 is the fixed deployment identity.
- Source-layer window: 3.
- Three epochs, 512 optimizer steps per epoch, gradient accumulation 8.
- AdamW learning rate `3e-4`, weight decay `1e-4`, gradient clipping 1.0.
- Loss weights: generation CE 1.0, teacher-logit KL 0.25, sampled attention KL 0.5,
  sampled attention-output MSE 0.5, and sampled target-KV anchor 0.1.
- Full-prefix transforms use bfloat16, 256-token activation-checkpointed chunks, and candidate
  microbatches of three under the default native CUDA allocator.

### Safety and selection

A prompt is unsafe if any condition holds:

- native task pass becomes a bridge failure;
- 16-token greedy agreement is below 0.98;
- teacher-forced perplexity drift exceeds 2%.

For each rank, metrics are averaged over all three seeds. Rank selection is lexicographic:
mean task preservation, mean oracle-safe coverage, mean greedy agreement, then lower mean P95
transform time. Only seed 17 at the selected rank can deploy. Publication requires deployment
oracle-safe coverage at least 0.45.

## Main Results

### All registered candidates

| Rank | Seed | Train total | Task preservation | Greedy agreement | PPL drift | Safe prompts | Coverage |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 32 | 17 | 0.491949 | 0.981683 | 0.624207 | 19.83% | 167 | 0.163086 |
| 32 | 29 | 0.496785 | 0.980915 | 0.630127 | 19.20% | 156 | 0.152344 |
| 32 | 43 | 0.546806 | 0.980558 | 0.622559 | 20.76% | 143 | 0.139648 |
| 64 | 17 | 0.433484 | 0.976862 | 0.617249 | 21.47% | 142 | 0.138672 |
| 64 | 29 | 0.548010 | 0.984084 | 0.635254 | 43.66% | 180 | 0.175781 |
| 64 | 43 | 0.432065 | 0.983086 | 0.600830 | 22.13% | 146 | 0.142578 |
| 128 | 17 | 0.412274 | 0.971258 | 0.615234 | 20.86% | 159 | 0.155273 |
| 128 | 29 | 0.758873 | 0.979771 | 0.549866 | 33.38% | 114 | 0.111328 |
| 128 | 43 | 0.405280 | 0.980242 | 0.648438 | 15.89% | 177 | 0.172852 |

No row in this table is selected post hoc. Rank 64 wins the registered rank-level ordering
because its mean task preservation is `0.9813439`, slightly above rank 32 at `0.9810520`.
Deployment therefore uses rank 64/seed 17 even though other individual seeds have higher
coverage.

### Rank-level selection

| Rank | Mean task | Mean coverage | Mean greedy | Mean P95 transform |
| ---: | ---: | ---: | ---: | ---: |
| 32 | 0.981052 | 0.151693 | 0.625631 | 98.072 ms |
| 64 | **0.981344** | **0.152344** | 0.617778 | 98.445 ms |
| 128 | 0.977090 | 0.146484 | 0.604513 | 99.119 ms |

The selected deployment candidate misses the coverage gate by `0.311328125`. Its 882 unsafe
prompts include 735 greedy-agreement failures and 865 perplexity-drift failures; 695 prompts
fail both criteria.

### Rank and seed cannot rescue the method

| Oracle choice set | Safe prompts | Coverage |
| --- | ---: | ---: |
| All three rank-32 seeds | 245 | 0.239258 |
| All three rank-64 seeds | 262 | 0.255859 |
| All three rank-128 seeds | 270 | 0.263672 |
| All nine candidates | 377 | 0.368164 |

The final row is a deliberately optimistic diagnostic that chooses a different candidate for
each prompt using target-derived outcomes. It is not deployable, yet it still fails 0.45. This
rules out rank or seed cherry-picking as an explanation for the registered failure.

### Full-prefix alignment helps, but does not solve behavior preservation

The cleanest intervention comparison holds rank 128 and seed 17 fixed between sampled-prefix
v3 and full-prefix v4.

| Metric | v3 sampled prefix | v4 full prefix | Change |
| --- | ---: | ---: | ---: |
| Safe prompts | 115 | 159 | +44 |
| Oracle-safe coverage | 0.112305 | 0.155273 | +0.042969 |
| Greedy agreement | 0.579712 | 0.615234 | +0.035522 |
| PPL drift | 21.46% | 20.86% | -0.60 pp |
| Task preservation | 0.985625 | 0.971258 | -0.014368 |

| Prefix tokens | v3 safe | v4 safe | Change |
| ---: | ---: | ---: | ---: |
| 128 | 30 | 36 | +6 |
| 512 | 37 | 36 | -1 |
| 2,048 | 29 | 44 | +15 |
| 8,192 | 19 | 43 | +24 |

The largest improvement occurs where the sampled-cache teacher most strongly differed from the
deployment teacher. However, v4 coverage remains below 0.18 for every individual candidate.
Teacher alignment was a real error source, not the only error source.

### Failure is task-dependent, not only length-dependent

The registered rank-64/seed-17 deployment candidate has the following task coverage:

| Task | Safe / total | Coverage | Greedy agreement | PPL drift |
| --- | ---: | ---: | ---: | ---: |
| Function calling | 38 / 48 | 0.791667 | 0.981771 | 2.60% |
| Competition math | 63 / 416 | 0.151442 | 0.612680 | 13.64% |
| Grade-school math | 29 / 416 | 0.069712 | 0.551833 | 21.04% |
| Long-context QA | 11 / 72 | 0.152778 | 0.649306 | 48.77% |
| Python code generation | 1 / 72 | 0.013889 | 0.746528 | 66.55% |

Coverage rises only modestly from 0.117 at 128 tokens to 0.164 at 8,192 tokens. Once the
teacher sees the full prefix, prefix length no longer explains the dominant residual failure.
The transport generalizes unevenly across suffix/task behavior.

### Average task preservation is an insufficient gate

All nine candidates report task preservation above 0.97 while greedy agreement ranges from
0.55 to 0.65 and perplexity drift ranges from 15.9% to 43.7%. Native task scores are low on
many prompts and coarse task metrics can remain unchanged when token distributions diverge.
The result supports reporting free-running token identity and teacher-forced drift alongside
semantic task metrics rather than treating task preservation alone as cache safety.

## Method Progression and Evidence Boundary

| Version | Generation supervision | Deployment coverage | Nine-candidate union | Outcome |
| --- | --- | ---: | ---: | --- |
| v2 | detached report-only terms | 0.232422 | 0.343750 | failed |
| v3 | differentiable sampled-prefix teacher | 0.112305 | 0.303711 | failed |
| v4 | differentiable full-prefix teacher | 0.138672 | 0.368164 | failed |

Deployment ranks differ under the registered ordering, so the table is descriptive rather than
a controlled ablation. The fixed rank-128/seed-17 comparison above is the appropriate mechanism
comparison.

Method-dev legitimately served as a development split across these iterations, but it is no
longer an independent confirmation set for another adaptive method. A future success claim must
use new code, a new workspace, and newly frozen development evidence. The current validation and
semantic sealed splits must not be opened to compensate for method-dev failure.

## Systems Scope

The repository implements:

- immutable model/data/code identity checks;
- resumable full-prefix transport fitting;
- source-only risk prediction and exact calibration contracts;
- one-shot sealed-data guards;
- direct atomic scattering into vLLM paged KV with rollback;
- LMCache MP source retrieval without target-object publication.

Only the first two items contribute real publication-v5 model evidence in this run. Later stages
have deterministic unit and integration tests, but no approved real-model execution because the
quality dependency failed. The paper may describe their design and fail-closed enforcement, but
must not report accepted-reuse rate, TTFT improvement, production safety, or zero-target-put
runtime evidence for cross-model reuse.

Same-model Mooncake reuse and older rank-512 bridge measurements are retained as development
context. They establish that the serving substrate executes and that earlier bridges also fail;
they are not evidence that v4 cross-scale translation is deployable.

## Related-Work Positioning

- Same-model prefix caching and KV offload establish the serving substrate, not cross-model
  behavioral equivalence.
- Cross-model and cross-layer cache translation already exist; novelty cannot rest on applying a
  learned map between Qwen sizes.
- Semantic Cache Distillation and Latent Cache Flow are the closest learned cache-transfer
  precedents, while Activated LoRA, PrefillShare, and ICaRus delimit exact sharing through
  constrained model training rather than post-hoc cross-scale translation.
- GoldenExperience differs in its exact fail-closed evidence chain, cross-scale prefix-replacement
  setting, complete rank/seed accounting, and atomic vLLM materialization contract.
- The empirical contribution is a falsification boundary: even full-prefix target supervision
  leaves a large gap between average semantic preservation and strict decoded-behavior safety.

The full prior-art claim audit is in `docs/related_work_matrix.md` and
`artifacts/publication_v5/development/related_work_fulltext_audit.json`.

## Required Figures

1. **Pipeline and stop boundary:** data splits, artifact states, and the method-dev failure that
   prevents all later arrows.
2. **Candidate matrix:** safe coverage for every rank/seed with the 0.45 gate and registered
   deployment marker.
3. **Teacher intervention:** v3 versus v4 safe counts by prefix bucket at fixed rank/seed.
4. **Task heterogeneity:** safe coverage, greedy agreement, and perplexity drift by task.
5. **Failure intersections:** greedy, drift, and task-regression overlap for the deployment
   candidate.
6. **Method progression:** deployment and oracle-union coverage for v2, v3, and v4, with a warning
   that deployment ranks differ.

## Required Tables

- Exact model and software identities.
- Dataset provenance, licenses, split sizes, and isolation checks.
- Full nine-candidate fit and method-dev metrics.
- Rank aggregation and registered selection.
- Token-bucket and task breakdowns.
- Full-prefix fixed-candidate ablation.
- Artifact hashes, checkpoint validation, and sealed-state evidence.
- Prior-art comparison with claims explicitly excluded from novelty.

## Limitations and Threats to Validity

- The real-model result covers one direction, Qwen3-4B -> Qwen3-8B. The stop rule prevents
  evidence for the other three directions.
- All registered Qwen models use eight KV heads; unequal-head support is untested.
- Method-dev is a development split, and repeated v2/v3/v4 diagnosis means no further adaptive
  iteration can use it as independent confirmation.
- The 0.98/2%/0.45 gates are intentionally strict. The paper establishes failure under this
  declared safety target, not impossibility under every application tolerance.
- Candidate ranks stop at 128 and the runtime operator is a fixed token-independent affine map.
  Larger, nonlinear, or token-conditioned transports may behave differently but require a new
  protocol instance.
- Dataset composition is dominated by math tasks; task-stratified results must accompany the
  aggregate.
- Native task accuracy is low for several scorers, which makes preservation easier to satisfy and
  strengthens the case for token/drift gates.
- No approved cross-model runtime latency or throughput experiment exists. Unit-tested direct
  materialization is not a measured speedup.
- The semantic sealed split remains unopened, so there is intentionally no final-test estimate.

## Artifact Checklist

- Fit receipt: `artifacts/publication_v5/stages/qwen3_4b_to_8b.fit_transport.v4.json`.
- Failed method-dev receipt:
  `artifacts/publication_v5/stages/qwen3_4b_to_8b.evaluate_method_dev.v4.failed.json`.
- Mechanism diagnostic:
  `artifacts/publication_v5/development/v4_method_dev_diagnostic.json`.
- Implementation verification:
  `artifacts/publication_v5/development/v4_implementation_verification.json`.
- Preregistered method: `docs/transport_v4.md`.
- Pipeline contract and terminal status: `docs/v5_pipeline.md`.
- Related-work audit: `docs/related_work_matrix.md`.

The final artifact package should also include a compressed canonical copy of the full
method-dev report, a checksum manifest, generated tables/figures, an environment inventory,
and commands that verify every retained object without accessing sealed content.
