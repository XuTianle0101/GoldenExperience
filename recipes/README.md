# Recipes

Runnable environment overlays for GoldenExperience experiments.

The default shared-KV recipe is `kv_baseline_mooncake_local.env`: it keeps the inference
engine as vLLM, runs a standalone LMCache MP server, and uses Mooncake Store as persistent
L2 on local TCP + SSD. Source the file before launching the thin baseline wrapper when you
want reproducible defaults outside the generated run directory.
