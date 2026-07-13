# Related-Work Claim Matrix

Search snapshot: 2026-07-13. This matrix records title/abstract-level checks from arXiv and
Semantic Scholar and must be refreshed with a full-text venue search before submission. It is
evidence for claim scoping, not a claim that the literature search is complete.

| Work | Cross-model reuse | Cross-size learned transport | Cross-head mapping | Formal behavioral-risk admission | Direct paged-KV injection |
| --- | --- | --- | --- | --- | --- |
| Prompt Cache, MLSys 2024, arXiv:2311.04934 | no | no | no | no | no |
| CacheBlend, arXiv:2405.16444 | same-model chunk fusion | no | no | heuristic selective recompute | no |
| Mooncake, arXiv:2407.00079 | storage substrate | no | no | SLO rejection, not quality risk | no |
| DroidSpeak, arXiv:2411.02820 | yes, same architecture | selective layer recompute | not its main mechanism | empirical quality selection | distributed load/recompute pipeline |
| vCache, ICLR 2026, arXiv:2502.03771 | response-level semantic cache | no | no | yes, user-defined response-cache error bounds | no |
| ProxyKV, arXiv:2605.16360 | small-model proxy scoring | mapper for pruning signals | yes, HybridAxialMapper | ranking loss, not certified reuse admission | no |
| Dense Latent Communication, arXiv:2606.13594 | yes | yes, all Qwen3 4B/8B/14B directions | not established from abstract | no calibrated request gate stated | no serving-page path stated |
| GoldenExperience v5 (target claim) | yes, same-family sizes | yes | yes, including 4-to-8 KV heads | exact one-sided risk-bounded selective admission | atomic vLLM paged scatter, no target put |

## Claim Decision

Cross-size Qwen3 KV transformation is not a standalone novelty claim. In particular,
arXiv:2606.13594 already studies lightweight cache transformation across all six Qwen3
4B/8B/14B directions, and ProxyKV already motivates explicit cross-head alignment for a
neighboring cross-model problem.

The paper claim is therefore narrowed to the composition and system contract:

> statistically risk-bounded, source-only admission for same-family cross-scale KV transport,
> coupled to atomic direct injection into an existing paged serving cache.

The claim remains provisional until full texts confirm that no prior system jointly provides
cross-size transformed KV, an independent calibration split with an explicit finite-sample
behavioral-risk bound, and direct target-page materialization without target-object
publication. If that conjunction is found, the contribution must be narrowed again before
running the sealed test.

## Required Full-Text Checks

1. Verify DroidSpeak's model-pair shapes, online selection inputs, and exact vLLM integration.
2. Compare the transport and training losses in arXiv:2606.13594 against the head-aware method.
3. Check whether ProxyKV's HybridAxialMapper transfers values or only pruning importance.
4. Compare vCache's guarantee model and online calibration assumptions with the fixed
   Clopper-Pearson split used here.
5. Search MLSys, OSDI, SOSP, NSDI, ATC, FAST, EuroSys, NeurIPS, ICML, and ICLR proceedings for
   unpublished/arXiv title variants and follow-on systems.

