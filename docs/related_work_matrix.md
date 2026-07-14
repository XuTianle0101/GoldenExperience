# Related-Work Claim Matrix

Search snapshot: 2026-07-14. The audit used the current arXiv records and full text for
the directly overlapping papers listed below. In particular, the vCache comparison uses
the v5 source rather than the stale VectorQ text still returned by ar5iv for the same arXiv
identifier. Venue labels come from the current arXiv metadata.

This matrix scopes claims; it is not evidence that every proceedings index is complete.
Until the remaining official-proceedings pass is finished, the paper must not use an
unqualified "first" claim.

| Work and audited version | Cache operation and model scope | Cross-model/topology result | Admission or guarantee | Runtime boundary |
| --- | --- | --- | --- | --- |
| Prompt Cache, MLSys 2024, arXiv:2311.04934 | Modular same-model attention reuse | No learned cross-model map | None | No cross-model paged injection |
| CacheBlend, EuroSys 2025, arXiv:2405.16444 | Same-model non-prefix cache fusion with selective recomputation | No cross-model map | Heuristic recomputation policy | Serving integration, but not cross-model replacement |
| Mooncake, arXiv:2407.00079 | Disaggregated KV storage and transfer substrate | Treats KV as model-local payload | SLO policy, not behavioral quality risk | Storage/transfer substrate rather than translation |
| DroidSpeak v4, arXiv:2411.02820 | Reuses selected KV/E layers between a base model and its same-shape fine-tuned derivative | Same foundational model and identical topology; unequal sizes are future work | Offline layer configuration from a profiling dataset; no request-level bound | Two A100 nodes; no vLLM page-level publication contract stated |
| KVComm v3, ICLR 2026, arXiv:2510.03346 | Concatenates selected sender KV layers into receiver attention | Same model or fine-tuned variants of the same base; one-to-one layer indices | One-sample layer-selection calibration; no behavioral-risk bound | Hugging Face prototype, not target-prefix replacement |
| Cache-to-Cache v2, ICLR 2026, arXiv:2510.03215 | Learns projection, fusion, head modulation, and layer gates over source and native receiver caches | Different families and sizes, including Qwen/Llama/Gemma | Fixed learned gates; no source-only request admission or finite-sample risk bound | Receiver must prefill its own cache before fusion; latency gain is over text communication |
| vCache v5, ICLR 2026, arXiv:2502.03771 | Reuses complete responses in a semantic prompt cache, not KV tensors | Model-independent response cache | Online randomized per-embedding error guarantee under i.i.d. data and a correctly specified sigmoid family | Requires ongoing LLM explorations/labels; no KV materialization path |
| ProxyKV v1, arXiv:2605.16360 | Maps small-model attention features to target per-key importance scores for Top-K pruning | Cross-size score prediction with learned head-axis queries | Ranking objective and fixed retention ratio; no reuse-safety bound | Does not translate or inject K/V values |
| LCGuard v1, arXiv:2605.22786 | Sanitizes communicated KV with learned residual bottlenecks | Evaluated across model families/scales, with same-backbone agents inside each run | Empirical reconstruction-privacy trade-off; explicitly no formal privacy guarantee | Latent-agent communication, not target-prefix serving reuse |
| Semantic Cache Distillation v1, ICML 2026, arXiv:2606.07684 | Sends low-rank K/V/hidden codes, reconstructs most cache layers, and patches selected layers | Same architecture and differing weights (base/fine-tuned or draft/verifier); arbitrary cross-topology transfer is out of scope | Expected KL constraint and deterministic offline layer profiling; no independent request-level behavioral-risk admission | Skips consumer prefill and writes Hugging Face `past_key_values`; no vLLM paged-cache atomicity evidence |
| Dense Latent Communication v1, arXiv:2606.13594 | De-RoPE/re-RoPE, monotonic depth alignment, per-KV-group K/V MLPs and gates, reconstruction then generation training | All six Qwen3 4B/8B/14B directions | No calibrated request gate or finite-sample regression bound | Latent-agent evaluation and FLOP accounting; no serving-page integration stated |
| Less Latent Relay v2, arXiv:2604.13349 | Compresses same-model LatentMAS relay and backfills discarded value subspaces | Qwen3-14B self-relay, not cross-model translation | Fixed compression budget; no behavioral-risk admission | Agent relay rather than reusable target-prefix cache |
| GoldenExperience v5 (target claim) | Replaces a target prefix with a learned same-family cross-scale cache and applies source-only selective admission | Registered Qwen3 4B/8B/14B directions span 36/40 layers; all registered models have eight KV heads | Independent calibration chooses a fixed threshold under a Bonferroni-corrected exact one-sided bound, followed by independent validation and one-shot semantic testing | Target is atomic direct scatter into vLLM paged KV through LMCache MP, with no target Mooncake object |

## Full-Text Findings

### Cross-model translation is established prior art

- C2C learns cross-family and cross-size cache projection/fusion, but it augments a cache
  that the receiver already prefills. Its speedup is relative to text-mediated agent
  communication, not relative to replacing the receiver prefix prefill.
- SCD is the closest serving-oriented predecessor. It explicitly skips consumer prefill,
  transports compact semantic codes, reconstructs consumer-space cache tensors, and reports
  up to 2.65x TTFT speedup. Its evaluated pairs retain one Transformer topology while weights
  differ; it does not cover unequal depth/head layouts or a page-native serving path.
- Dense Latent Communication already evaluates all Qwen3 4B/8B/14B directions and uses
  position disentanglement, depth alignment, and per-KV-group K/V transformations. Cross-size
  Qwen cache transformation, RoPE-aware translation, and per-head mapping are therefore not
  standalone novelty claims.
- ProxyKV's HybridAxialMapper predicts target-shaped importance scores. It provides useful
  cross-head mapping precedent, but it never reconstructs K/V values and cannot substitute
  for a target prefix cache.

### Selection and guarantees have materially different contracts

- DroidSpeak, KVComm, C2C, SCD, Dense Latent Communication, and ProxyKV use offline fixed
  configurations, quality/latency trade-offs, or compression budgets. None of their audited
  full texts provides a source-only per-request reuse gate calibrated on an independent split
  with an exact finite-sample behavioral-regression bound.
- vCache does provide a user-selected response-error guarantee, so "first risk-bounded cache"
  is not a defensible claim. Its guarantee is for response-level semantic caching and an online
  randomized explore/exploit policy, conditional on i.i.d. requests and correct sigmoid model
  specification. GoldenExperience instead preregisters a frozen predictor and threshold on
  disjoint data and uses exact Clopper-Pearson bounds without online target labels.
- LCGuard addresses a different safety axis: recoverability of private inputs from latent
  communication. It reports empirical privacy/utility trade-offs and explicitly disclaims a
  formal privacy guarantee. Behavioral preservation and representation privacy must not be
  conflated.

### Runtime evidence remains a separate contribution

- SCD populates a Hugging Face cache container and C2C/Dense Latent Communication operate at
  model-evaluation level. The audited papers do not establish an atomic LMCache/vLLM paged-KV
  scatter that rolls back partial layers and avoids publishing a target storage object.
- This distinction is a systems claim only after the registered runtime audit passes. A Python
  cache assignment, tensor-level latency estimate, or semantic approval cannot substitute for
  direct-page runtime evidence.

## Claim Decision

The following claims are retired:

- first cross-model or cross-size KV transformation;
- first Qwen3 4B/8B/14B cache translator;
- first RoPE-aware, layer-aware, or head-aware cache map;
- first cross-model method to skip target prefill or improve TTFT;
- first cache system with any form of error guarantee.

The provisional claim is narrowed to the conjunction:

> independently calibrated, source-only request admission with an exact finite-sample
> behavioral-regression bound for cross-scale target-prefix replacement, coupled to atomic
> materialization into an existing vLLM paged cache without publishing a target cache object.

No individual component in that sentence should be promoted as independently novel. The
claim is valid only if all registered directions pass method development, calibration,
validation, semantic sealed evaluation, and the direct-injection runtime audit.

## Unequal-Head Evidence Boundary

The transport implementation supports different source and target KV-head counts, but every
registered Qwen3 model in the current benchmark has exactly eight KV heads and head dimension
128. The current experiment therefore provides no empirical unequal-KV-head result. In
particular, the previous 4-versus-8 KV-head phrase was incorrect and must not appear in the
paper. Unequal-head support may be described as an implementation capability only,
unless a separately preregistered model pair is evaluated.

## Remaining Proceedings Check

The full-text pass above resolves the originally listed DroidSpeak, Dense Latent
Communication, ProxyKV, and vCache questions and adds the directly overlapping ICLR 2026
C2C/KVComm and ICML 2026 SCD papers. Before submission, search the official MLSys, OSDI,
SOSP, NSDI, ATC, FAST, EuroSys, NeurIPS, ICML, and ICLR proceedings indexes for title variants
and follow-on versions. The current environment could retrieve arXiv records/full text but
could not reach several official index APIs, so this last pass remains explicitly open.
