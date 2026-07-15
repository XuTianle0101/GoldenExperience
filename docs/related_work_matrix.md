# Related-Work Claim Matrix

Search snapshot: 2026-07-15. The audit used the current arXiv records and full text for
the directly overlapping papers listed below. In particular, the vCache comparison uses
the v5 source rather than the stale VectorQ text still returned by ar5iv for the same arXiv
identifier. Venue labels come from the current arXiv metadata or the versioned paper itself.

This matrix scopes claims; it is not evidence that every proceedings index is complete.
Until the remaining official-proceedings pass is finished, the paper must not use an
unqualified "first" claim.

| Work and audited version | Cache operation and model scope | Cross-model/topology result | Admission or guarantee | Runtime boundary |
| --- | --- | --- | --- | --- |
| Prompt Cache, MLSys 2024, arXiv:2311.04934 | Modular same-model attention reuse | No learned cross-model map | None | No cross-model paged injection |
| CacheBlend, EuroSys 2025, arXiv:2405.16444 | Same-model non-prefix cache fusion with selective recomputation | No cross-model map | Heuristic recomputation policy | Serving integration, but not cross-model replacement |
| Mooncake, arXiv:2407.00079 | Disaggregated KV storage and transfer substrate | Treats KV as model-local payload | SLO policy, not behavioral quality risk | Storage/transfer substrate rather than translation |
| SwiftCache v1, arXiv:2606.16135 | Places one model's exact prefix-cache bytes in idle memory owned by another model's GPU | Heterogeneous models share memory and NVLink, but do not consume one another's cache values | Not applicable; cache contents remain model-local | Custom multi-model engine; up to 69% lower P99 TTFT is a storage-placement result |
| DroidSpeak v4, arXiv:2411.02820 | Reuses selected KV/E layers between a base model and its same-shape fine-tuned derivative | Same foundational model and identical topology; unequal sizes are future work | Offline layer configuration from a profiling dataset; no request-level bound | Two A100 nodes; no vLLM page-level publication contract stated |
| Activated LoRA v1, arXiv:2512.17910 | Reuses the exact base-generated prefix before an adapter activation point | Base and adapters retain identical prefix weights and topology; no foreign-cache translation | Activation semantics determine reuse; no behavioral-risk admission | Modified vLLM block hashing and masking with measured TTFT, but only for co-designed adapters |
| PrefillShare v1, arXiv:2602.12029 | Freezes a shared prefill module and fine-tunes task-specific decode modules | Specialized models are trained to consume one common cache; independently trained cross-scale caches are not translated | Training construction, not request-level risk calibration | vLLM-based disaggregated serving; reports 4.5x lower P95 latency and 3.9x throughput |
| ICaRus v1, ICLR 2026, arXiv:2603.13281 | Freezes a logical encoder and fine-tunes specialized logical decoders that share its exact cache | Qwen3 scales are separate experiments; sharing occurs among derivatives of one base at a time | Training construction, not request-level risk calibration | vLLM integration reports up to 11.1x lower P95 latency and 3.8x throughput |
| KVComm v3, ICLR 2026, arXiv:2510.03346 | Concatenates selected sender KV layers into receiver attention | Same model or fine-tuned variants of the same base; one-to-one layer indices | One-sample layer-selection calibration; no behavioral-risk bound | Hugging Face prototype, not target-prefix replacement |
| KVCOMM v2, NeurIPS 2025, arXiv:2510.12872 | Repositions same-model shared-text caches and estimates context offsets from an online anchor pool | Evaluated on homogeneous agents; different weights and attention architectures remain future work | Online embedding/length match and fallback, but no independent behavioral-risk bound | Hugging Face multi-agent evaluation, not cross-model prefix replacement |
| Cache-to-Cache v2, ICLR 2026, arXiv:2510.03215 | Learns projection, fusion, head modulation, and layer gates over source and native receiver caches | Different families and sizes, including Qwen/Llama/Gemma | Fixed learned gates; no source-only request admission or finite-sample risk bound | Receiver must prefill its own cache before fusion; latency gain is over text communication |
| Latent Cache Flow v2, ICML 2026, arXiv:2605.22863 | Compresses sender K/V into residual edits of a receiver cache; LCF-X pools cross-context information | Shared-context result is Qwen2.5-0.5B to Qwen3-0.6B; cross-context result uses two Qwen3-0.6B instances | Fixed learned gates and task-specific training; no independent request gate | Receiver still prefills; TTFT comparison is cache communication versus generated text |
| Q-KVComm v1, arXiv:2512.17914 | Quantizes selected cache layers and applies mean/variance alignment across architectures | Small 1.1B-1.5B models with distributional calibration; no target-prefix replacement evidence | Compression sensitivity and distribution statistics, not behavioral-risk calibration | Prototype communication pipeline; no page-native serving audit |
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
- Activated LoRA, PrefillShare, and ICaRus obtain exact cache sharing by constraining where
  model parameters may differ or by training specialized decoders around a frozen cache
  producer. They establish real vLLM cross-model reuse, but do not translate caches between
  independently trained Qwen scales.
- LCF is a closer learned cross-architecture communication baseline than the original audit
  recorded. Its receiver still constructs a native cache and accepts residual edits; its
  cross-context experiment is task-specific and same-model. It does not replace a target
  prefix cache, but it further retires broad cache-translation and latent-channel novelty claims.
- ProxyKV's HybridAxialMapper predicts target-shaped importance scores. It provides useful
  cross-head mapping precedent, but it never reconstructs K/V values and cannot substitute
  for a target prefix cache.

### Selection and guarantees have materially different contracts

- Most audited translators use offline configurations, learned gates, quality/latency trade-offs,
  or compression budgets. KVCOMM is an important exception because it updates an online anchor
  pool and falls back when context matching fails. Its decision estimates same-model cache
  offsets, not a target-derived behavioral regression probability, and it has no independent
  finite-sample risk bound. None of the audited full texts provides GoldenExperience's proposed
  source-only behavioral admission contract.
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

- Activated LoRA, PrefillShare, and ICaRus provide real vLLM integrations, so a broad claim of
  first cross-model reuse in a modern serving engine is untenable. SCD populates a Hugging Face
  cache container, while C2C, LCF, and Dense Latent Communication operate at model-evaluation
  level. The narrower audited gap is an atomic scatter of a translated foreign cache into
  existing vLLM pages with rollback and no translated target storage object.
- This distinction is a systems claim only after the registered runtime audit passes. A Python
  cache assignment, tensor-level latency estimate, or semantic approval cannot substitute for
  direct-page runtime evidence.

## Claim Decision

The following claims are retired:

- first cross-model or cross-size KV transformation;
- first Qwen3 4B/8B/14B cache translator;
- first RoPE-aware, layer-aware, or head-aware cache map;
- first cross-model method to skip target prefill or improve TTFT;
- first cross-model KV reuse integrated into vLLM or a production-style serving engine;
- first full-cache sharing across multiple specialized models;
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

## Proceedings And Search Coverage

The 2026-07-15 pass expanded the arXiv query beyond known titles to `cross-model`, `multi-model`,
`heterogeneous`, `model-to-model`, `latent communication`, `KV cache reuse`, and `KV cache
translation` combinations. It added Activated LoRA, PrefillShare, ICaRus, KVCOMM, LCF,
SwiftCache, and Q-KVComm after reading their fixed-version full texts. This materially changes
the runtime and exact-sharing comparison above.

The official EuroSys 2026 paper list was independently retrieved from the conference's GitHub
site at commit `3d419f688746f8edf0190a6ae505c9f51d6e6220`. Its only title-level KV match, "High
Throughput and Low Latency LLM Serving via Adaptive KV Caching," is a same-model serving policy,
not cross-model cache translation.

A second direct-TLS pass at `2026-07-15T10:14:46Z` retried eleven official endpoints spanning
MLSys, USENIX, SIGOPS, EuroSys, NeurIPS, PMLR/ICML, OpenReview/ICLR, and ACM DL. Every endpoint
again returned curl exit 35 before an HTTP response. GitHub's API remained reachable, allowing a
publisher-owned fallback check for ICML 2026: the `mlresearch` organization identifies itself as
the Proceedings of Machine Learning Research, and its `v306` repository is labeled "Proceedings
of ICML 2026." At commit `b3b1748fa2fac7ec916eb1dee8fee9f0691d9450`, however, the complete
recursive tree contains only a proceedings-preparation README and pull-request template--no
BibTeX index or paper payload. It is therefore an official placeholder, not an accepted-paper
list, and cannot close the ICML title scan.

Direct official indexes other than the fixed EuroSys list remain inaccessible or not yet
published in an auditable repository. The paper therefore retains no unqualified priority claim,
and these indexes remain a documented submission-time check rather than being silently marked
complete or replaced by an unverified third-party list.
