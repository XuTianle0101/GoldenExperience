# GoldenExperience

[English](README.md) | [中文](README.zh-CN.md)

GoldenExperience 是一个面向 **vLLM + LMCache MP + Mooncake Store** 共享 KV 底座的
**跨模型 KV Cache 复用框架**。

新的运行时边界刻意保持狭窄：

- vLLM 负责模型加载、调度、解码以及推理正确性。
- LMCache MP 负责共享 KV 查询、卸载、淘汰和预取编排。
- Mooncake Store 负责跨 engine 重启可持久化的 L2 metadata 与 SSD 存储根目录。
- GoldenExperience 为 **跨模型复用 KV Cache** 添加控制平面。
- 如果复用计划不安全或未经校准，系统会回退到原始的 vLLM + LMCache MP 行为。

## 项目关注点

GoldenExperience 不尝试成为推理引擎，也不替换缓存存储。它旨在作为 LMCache MP
之上的小型补丁和 Python control-plane 存在，让运行时元数据通过 vLLM/LMCache MP
connector 路径进入 lookup/retrieve 逻辑。

研究/开发目标包含三条 GoldenExperience 复用路线：

| 路线名称 | 场景 | 目标 | 默认策略 | 安全门控 |
| --- | --- | --- | --- | --- |
| GoldenLoRA | 基座模型 <-> LoRA 模型 | 在模型及其 LoRA 微调变体之间复用 KV | 由 adapter-delta 门控的别名复用 | 相同基座、tokenizer、KV 布局、LoRA 漂移探针 |
| GoldenScale | 同一模型的不同参数规模 | 在 8B <-> 14B 等变体之间复用 KV | 形状匹配时直接别名；否则hidden-state bridge | 层/头映射与hidden bridge 校准 |
| GoldenBridge | 不同基座模型 | 探索更广泛的跨基座复用 | 学习式转换器 | 显式校准集、tokenizer 桥接、任务白名单 |

这些名称与实现场景的映射关系如下：`GoldenLoRA` 对应 `model_lora_mutual_reuse`，
`GoldenScale` 对应 `same_model_different_parameter_size`，`GoldenBridge` 对应
`different_base_model`。

## 架构

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
             | metadata sidecars, secondary lookup, materialization, accounting
             v
      GoldenExperience planner and LMCache patch surface
```

补丁面由 `PatchManifest.default()` 描述：

1. `engine_request_metadata`：在 LMCache MP 查询前附加 `ModelRef` 和前缀元数据。
2. `lmcache_cross_model_lookup`：同模型未命中时，查询跨模型候选。
3. `goldenexperience_materializer`：在返回 KV 前对其进行别名、投影或转换。
4. `quality_gate_accounting`：记录置信度、校准状态和回退原因。

## 仓库结构

仓库组织参考 C2C 的思路：核心 Python 包、薄脚本入口、配置、可复现实验 recipe、文档、
示例、测试和 artifact 分层。C2C 将核心包 `rosetta/` 与 `script/`、`bash/`、
`recipe/`、`resource/` 分开；GoldenExperience 对应地把工程化调度放在
`goldenexperience/runtime/`，把薄 shell 入口放在 `scripts/`，把可 source 的运行配置放在
`recipes/`。

```text
goldenexperience/
  runtime/           vLLM + LMCache MP + Mooncake runtime 检查与 baseline 调度器。
  reuse/             ModelRef, KVShape, ReuseRequest, ReusePlan, scenario planner.
  lmcache_patch/     Patch manifest and sidecar key metadata for LMCache MP deltas.
  size_variant/      GoldenScale calibration, layer mapping, projection scaffolds.
  benchmarks/        Synthetic and model-backed benchmark harnesses.
  cache_core/        Legacy in-repo cache block metadata utilities for tests/prototypes.
  tiered_store/      Legacy synthetic tiering prototype; not the product runtime path.
scripts/
  kv_baseline/       薄 shell 启动器，以及 stdlib OpenAI-compatible client/summarizer。
recipes/             可 source 的 env overlay，用于可复现运行时启动。
docs/                Design, shared KV substrate, experiment matrix, artifact, paper notes.
configs/             Runtime env examples and cross-model reuse experiment configuration.
examples/            Minimal planning examples.
tests/               Unit tests for planner, runtime config, and baseline generation.
```

## 快速开始

创建 Python 3.10+ 环境并安装本地项目：

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -e ".[dev]"
pytest
```

运行 planner 冒烟测试：

```bash
python3 scripts/smoke_cross_model_plan.py
```

构建 Qwen3 8B <-> 14B 的 GoldenScale 校准脚手架：

```bash
golden-scale-collect --output artifacts/golden_scale/prompts.json
golden-scale-fit \
  --direction bidirectional \
  --prompt-manifest artifacts/golden_scale/prompts.json \
  --output-dir artifacts/golden_scale
golden-scale-validate artifacts/golden_scale/qwen3_8b_to_14b_hidden_bridge_v0.json
golden-scale-bench artifacts/golden_scale/qwen3_14b_to_8b_hidden_bridge_v0.json
```

## 部署流程

GoldenExperience 作为 Python 包部署在与 vLLM、LMCache 和 Mooncake 相同的环境中。
它不是独立 daemon。

运行时职责：

- vLLM 启动 OpenAI-compatible 推理服务器，并负责请求调度与生成。
- LMCache MP 负责共享 KV 查询、存储策略、卸载、淘汰和预取。
- Mooncake Store 负责持久化 L2 metadata 和 SSD-backed objects。
- GoldenExperience 负责 `ModelRef`、`ReuseRequest`、`ReusePlan`、补丁元数据以及质量/回退计数。

### 1. 安装运行时包

只需要运行栈且 Mooncake 二进制已经在 `PATH` 上时，使用 package 模式：

```bash
python3 -m venv .venv
source .venv/bin/activate
./scripts/install_runtime.sh --mode package
```

package 模式会 fail closed 到已验证的 CUDA 13 组合（`vllm==0.24.0`、
`lmcache==0.4.6`），且不会绕过依赖解析器替换 CuPy。CUDA 12 或其他运行时组合需使用
source 模式，直到对应 adapter 兼容性测试落地。

需要补丁 LMCache 或调试 vLLM/LMCache/Mooncake 内部逻辑时使用 source 模式：

```bash
python3 -m venv .venv
source .venv/bin/activate
./scripts/install_runtime.sh --mode source
```

`--mode source` 会把 vLLM、LMCache 和 Mooncake 克隆到 `third_party/`。脚本会以
editable 方式安装 vLLM 和 LMCache，并默认带上 `BUILD_MOONCAKE=1`；Mooncake 请按上游
说明构建，并确保 `mooncake_master` 和 `mooncake_http_metadata_server` 位于 `PATH`。
使用 fork 时可以覆盖默认值：

```bash
GE_THIRD_PARTY_DIR=third_party \
GE_VLLM_REPO_URL=https://github.com/vllm-project/vllm.git \
GE_LMCACHE_REPO_URL=https://github.com/LMCache/LMCache.git \
GE_MOONCAKE_REPO_URL=https://github.com/kvcache-ai/Mooncake.git \
./scripts/install_runtime.sh --mode source
```

如果运行时栈已经可用，只安装 GoldenExperience：

```bash
./scripts/install_runtime.sh --mode golden-only
```

脚本在安装了 `uv` 时优先使用 `uv pip install`，否则回退到 `python3 -m pip install`。
安装依赖后默认会执行 `scripts/patch_lmcache_mooncake_runtime.py`。这个可复现补丁会补齐
LMCache 期望的 Mooncake `libmooncake_store.so` 别名，默认选择 Python
`MooncakeDistributedStore` SET/GET adapter，并用 LMCache MP 进程内 key index 绕过 native
`batchIsExist` lookup 崩溃路径。只有在明确想测试未补丁上游路径时，才设置
`GE_PATCH_MOONCAKE_RUNTIME=0`。package 和 golden-only 模式结束时会严格检查
`vLLM`、`LMCache` 和 `Mooncake` 是否可用；source 模式默认只警告，因为 Mooncake 仍需要按
上游步骤构建。可用 `--runtime-check strict|warn|skip` 覆盖该行为。当修改 CUDA、Python 或
包版本时，仍应对照 vLLM、LMCache 和 Mooncake 上游文档检查细节。

### 2. 验证 Planner 和运行时

```bash
python3 scripts/smoke_cross_model_plan.py --check-runtime --strict-runtime
python3 scripts/patch_lmcache_mooncake_runtime.py --check
```

预期 planner 输出包含三行：

- `model_lora_mutual_reuse`：已就绪的基座/LoRA 计划。
- `same_model_different_parameter_size`：已校准的 GoldenScale 投影计划。
- `different_base_model`：保守的、未就绪的跨基座计划。

运行时检查会报告 `vLLM`、`LMCache` 和 `Mooncake`。如有缺失，请先安装运行时栈，再启动基于模型的服务。

### 3. 生成补丁清单

```bash
golden-patch-manifest --output docs/patch_manifest.md
```

### 4. 运行共享 KV baseline

推荐启动路径是工程化 Python 调度器，shell 仅作为薄入口：

```bash
GE_MODEL_PATH=Qwen/Qwen3-8B \
GE_PORT=30000 \
scripts/kv_baseline/run_vllm_lmcache_mooncake_kv_baseline.sh -- --tensor-parallel-size 1
```

安装后也可以直接调用 console entry point：

```bash
GE_MODEL_PATH=Qwen/Qwen3-8B golden-kv-baseline -- --tensor-parallel-size 1
```

### 5. 运行同模型 KV 卸载/复用基线

当 vLLM、LMCache、Mooncake 和 GoldenExperience 已安装在同一个 Python 环境后，
使用该基线。默认路径已经切换为 `vLLM + LMCache MP + Mooncake Store`：

1. 在本机 TCP 上启动 Mooncake master 和 HTTP metadata 服务；
2. 启动独立 LMCache MP server，并把 L2 配置为 `type=mooncake_store`；
3. 使用 `LMCacheMPConnector` 启动 vLLM；
4. 发送 offload 请求，只重启 vLLM，再发送相同 reuse 请求。

这样共享 KV 底座位于推理进程之外，后续跨模型复用只需要接入 LMCache MP 的持久
L2，而不依赖 engine 进程内缓存。

```bash
source .venv/bin/activate
source recipes/kv_baseline_mooncake_local.env

GE_MODEL_PATH=Qwen/Qwen3-8B \
GE_PORT=30000 \
scripts/kv_baseline/run_vllm_lmcache_mooncake_kv_baseline.sh -- --tensor-parallel-size 1
```

默认本机 TCP + SSD 设置：

- `GE_KV_BACKEND=mp`、`GE_ENGINE=vllm`、`GE_LMCACHE_MP_L2_ADAPTER_TYPE=mooncake_store`。
- `GE_MOONCAKE_MASTER_HOST=127.0.0.1`、`GE_MOONCAKE_MASTER_PORT=50051`。
- `GE_MOONCAKE_METADATA_PORT=8080`，生成 `http://127.0.0.1:8080/metadata`。
- `GE_LMCACHE_MP_HTTP_PORT=8081`，避免 LMCache MP HTTP 与 Mooncake metadata 冲突。
- `GE_MOONCAKE_PROTOCOL=tcp`、`GE_MOONCAKE_STORAGE_ROOT=$GE_KV_CACHE_DIR/mooncake`。
- `GE_MOONCAKE_PER_OP_WORKERS_JSON='{"lookup":2,"retrieve":8,"store":4}'`。
- `LMCACHE_MOONCAKE_PYTHON_ADAPTER=1`，使用 Python Mooncake Store SET/GET 路径；
  `LMCACHE_MOONCAKE_NATIVE_EXISTS=0` 用于避开 native `batchIsExist`。
- Mooncake master 默认带上 `--client_ttl=600`、来自 storage root 的 `--root_fs_dir`，
  以及来自 `GE_RUN_ID` 的 `--cluster_id`；需要覆盖时使用 `GE_MOONCAKE_MASTER_EXTRA_ARGS`。

默认输出会写入 `artifacts/kv_baseline/<run_id>/`：

- `metadata.json`：模型、提示、MP connector、Mooncake endpoint、adapter JSON、pid 和日志路径。
- `lmc_config.yaml`：包含 `LMCacheMPConnector` 和 Mooncake Store L2 的生成配置。
- `requests/offload.json`：第一次请求的输出、usage、端到端延迟和 TTFT。
- `requests/reuse.json`：vLLM 重启后使用相同提示的第二次请求。
- `logs/lmcache_mp_server.log`：持久 LMCache MP server 证据。
- `logs/mooncake_master.log` 和 `logs/mooncake_metadata_server.log`：Mooncake 服务证据。
- `metrics/offload.prom` 和 `metrics/reuse.prom`：vLLM external KV transfer 计数器。
- `summary.json`：请求差值，以及 store/retrieve/L2/Mooncake 的日志计数。

原始 KV baseline run 目录会被 Git 忽略。Git 里只保留
`artifacts/kv_baseline/manifests/` 下的精选 manifest，大的 KV seed payload 放到外部
artifact store：

```bash
python3 scripts/kv_baseline/export_kv_seed_manifest.py \
  artifacts/kv_baseline/<run_id> \
  --artifact-uri s3://bucket/ge-kv-seeds/<artifact_id>.tar.zst \
  --output artifacts/kv_baseline/manifests/<run_id>.json
```

默认在 `GE_FORCE_DISK_OFFLOAD=1` 时生成较长的确定性 disk-offload prompt。要使用
自定义 prompt manifest，请设置 `GE_PROMPT_FILE` 和 `GE_PROMPT_ID`；如果生成 prompt 超过
模型上下文，可调小 `GE_DISK_PROMPT_REPEAT`。

常用覆盖项：

```bash
GE_MODEL_PATH=/models/Qwen3-8B \
GE_MODEL_NAME=/models/Qwen3-8B \
GE_RUN_ID=qwen3_8b_mooncake_restart_001 \
GE_MOONCAKE_STORAGE_ROOT=/ssd/ge-kv/mooncake \
GE_MOONCAKE_GLOBAL_SEGMENT_SIZE=4294967296 \
GE_MOONCAKE_LOCAL_BUFFER_SIZE=4294967296 \
scripts/kv_baseline/run_vllm_lmcache_mooncake_kv_baseline.sh -- --tensor-parallel-size 1
```

解释检查清单：

1. 确认 `requests/offload.json` 包含预期答案。
2. 确认 `summary.json` 中 `offload_has_disk_evidence=true`、
   `reuse_has_cache_evidence=true`、`disk_reuse_success=true`。
3. 确认 `metrics/reuse.prom` 中 `vllm:external_prefix_cache_hits_total > 0`，并且
   `vllm:prompt_tokens_by_source_total{source="external_kv_transfer"} > 0`；offload 阶段应主要是
   `source="local_compute"`。
4. 确认 `logs/lmcache_mp_server.log` 包含 Python Mooncake Store 证据：
   `MooncakeStore SET`、`MooncakeStore EXISTS`、`MooncakeStore GET` 和
   `L2 prefetch load completed`。
5. 确认 `metadata.json` 记录 `mooncake.enabled=true`、`l2_adapter_type=mooncake_store`、
   Mooncake storage root，以及不同的 offload/reuse vLLM 服务 pid。
6. 确认 Mooncake storage root 内有非空文件，然后保留完整的
   `artifacts/kv_baseline/<run_id>/` 目录，作为后续跨模型 KV 复用实验的同模型基线。

兼容和诊断：

```bash
# 使用旧的 MP filesystem L2 adapter，而不是 Mooncake Store。
GE_LMCACHE_MP_L2_ADAPTER_TYPE=fs \
scripts/kv_baseline/run_vllm_lmcache_mooncake_kv_baseline.sh -- --tensor-parallel-size 1
```

需要运行结束后检查 live 服务时，可设置 `GE_KEEP_LMCACHE_MP_AFTER_RUN=1` 或
`GE_KEEP_MOONCAKE_AFTER_RUN=1`。

默认 Mooncake baseline 有意使用 Python Mooncake Store SET/GET 路径，因为本地 baseline 所用
package 组合里的 native C++ `batchIsExist` 路径在 missing key 上出现过兼容性崩溃。如果要显式
测试 native 路径，设置 `LMCACHE_MOONCAKE_PYTHON_ADAPTER=0` 和
`LMCACHE_MOONCAKE_NATIVE_EXISTS=1`。

### 当前运行时状态

现在有两条跨参数量 runtime 路径：

- `scripts/run_cross_model_runtime.py`：较早的 `native_target_seed` proof。它用目标模型
  prefill 生成 target-shaped KV，重启目标 vLLM，并验证 LMCache/Mooncake 外部 KV retrieval。
- `scripts/run_qwen3_cached_kv_runtime.py`：带质量门的 cached-KV 路径。它先做 source
  offload，在目标 miss 后查找源模型 Mooncake keys，调用 resident materializer，并仅在
  identity、quality、exact-I/O 和 runtime cost 门全部通过后原子发布 target keys。

已停止的 hidden-state 和 prefix-specific bridge 实验已汇总到 `docs/paper_outline.md`。
整理后删除了机器相关 manifest 与专用训练脚本：prefix-specific 方案虽然在小规模 cosine
probe 上通过，但 runtime task assertion 失败，不能作为通用 8B -> 14B bridge。

## GoldenScale 复用

首个 GoldenScale MVP 面向 `Qwen/Qwen3-8B` 与 `Qwen/Qwen3-14B` 的双向复用。GoldenExperience 会把每个方向视为独立 artifact，因为 8B->14B 和 14B->8B 需要不同的层映射、hidden bridge 规格、目标 KV restore 规格、成本估计和质量门控。

artifact 流程为：

```text
shared prompts
  -> golden-scale-collect
  -> golden-scale-fit
  -> CalibrationManifest JSON per direction
  -> golden-scale-validate
  -> planner READY only when calibration/artifact gates pass
```

MVP artifact 包含：

- `LayerMap`：覆盖每个目标层，并将其映射到源层 id。
- `HiddenBridgeSpec`：把小模型 `pre_kv_hidden` 映射到大模型 hidden 宽度。
- `KVRestoreSpec`：记录目标模型 W_K/W_V/RoPE restore contract 与 GQA KV layout。
- `ProjectionSpec`：保留为旧 KV projection baseline/control。
- `QualityGateResult`：离线/影子门控指标，例如 hidden cosine、KV cosine、attention proxy cosine 和 perplexity drift。
- sidecar ids：`pair_id`、`direction`、`calibration_id`、`layer_map_id`、`hidden_bridge_id`、`restore_id`、state kind、源/目标配置哈希以及回退原因。

运行时行为保持保守：

- 前缀 token ids 必须完全匹配；要求 chunk 对齐。
- materializer 先执行 `h_small -> h_large_hat`，再由目标模型 W_K/W_V/RoPE restore 出完整目标形状 KV。
- `estimated_materialization_ms` 必须 <= 目标 prefill 成本的 70%。
- 任何 tokenizer、RoPE、配置哈希、artifact、层映射、hidden bridge、restore 或质量不匹配，都会回退到原始的 vLLM + LMCache MP 目标 prefill 路径。

## 最小 Planner 示例

```python
from goldenexperience.reuse import CrossModelReusePlanner, KVShape, ModelRef, ReuseRequest

shape = KVShape(num_layers=36, num_key_value_heads=8, head_dim=128)
base = ModelRef(
    model_id="qwen3-8b",
    family="qwen",
    architecture="qwen3",
    tokenizer_id="qwen3",
    parameter_count_b=8,
    kv_shape=shape,
)
lora = ModelRef(
    model_id="qwen3-8b-lora-math",
    family="qwen",
    architecture="qwen3",
    tokenizer_id="qwen3",
    parameter_count_b=8,
    base_model_id="qwen3-8b",
    lora_adapter_id="math-adapter",
    kv_shape=shape,
)

plan = CrossModelReusePlanner().plan(
    ReuseRequest(source=base, target=lora, prefix_hash="shared-system-prompt")
)
print(plan.scenario.value, plan.strategy.value, plan.status.value)
```

渲染 LMCache 补丁契约：

```bash
golden-patch-manifest --output docs/patch_manifest.md
```

## 开发路线图

- M0：围绕 vLLM + LMCache MP + Mooncake Store + GoldenExperience 元数据锁定项目边界。
- M1：为基座/LoRA 相互复用实现 LMCache 二级查询 sidecar。
- M2：为同模型不同规模变体添加层/头映射与校准投影。
- M3：为不同基座复用添加实验性的学习式转换器接口。
- M4：构建基于 vLLM 模型的 benchmark，以及质量/回退计数。
- M5：保持补丁足够小，以便随上游 LMCache rebase。

详见 `docs/design.md`、`docs/experiment_matrix.md` 和 `docs/artifact.md` 获取完整框架计划。
