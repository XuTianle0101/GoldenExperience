# GoldenExperience

[English](README.md) | [中文](README.zh-CN.md)

GoldenExperience 是一个用于开源服务栈的 **跨模型 KV Cache 复用补丁框架**，该服务栈由 **SGLang** + **LMCache** 构建。

新的边界刻意保持狭窄：

- SGLang 负责模型加载、调度、解码以及推理正确性。
- LMCache 负责 KV 存储、查询、卸载、淘汰和预取机制。
- GoldenExperience 为 **跨模型复用 KV Cache** 添加控制平面。
- 如果复用计划不安全或未经校准，系统会回退到原始的 SGLang + LMCache 行为。

## 项目关注点

GoldenExperience 不再尝试成为推理引擎或 KV 卸载系统。它旨在作为 LMCache 之上的小型补丁存在，并让运行时元数据从 SGLang 请求流入 LMCache 的查询和取回路径。

研究/开发目标包含三条 GoldenExperience 复用路线：

| 路线名称 | 场景 | 目标 | 默认策略 | 安全门控 |
| --- | --- | --- | --- | --- |
| GoldenLoRA | 基座模型 <-> LoRA 模型 | 在模型及其 LoRA 微调变体之间复用 KV | 由 adapter-delta 门控的别名复用 | 相同基座、tokenizer、KV 布局、LoRA 漂移探针 |
| GoldenScale | 同一模型的不同参数规模 | 在 7B <-> 14B 等变体之间复用 KV | 形状匹配时直接别名；否则逐层投影 | 层/头映射与投影校准 |
| GoldenBridge | 不同基座模型 | 探索更广泛的跨基座复用 | 学习式转换器 | 显式校准集、tokenizer 桥接、任务白名单 |

这些名称与实现场景的映射关系如下：`GoldenLoRA` 对应 `model_lora_mutual_reuse`，`GoldenScale` 对应 `same_model_different_parameter_size`，`GoldenBridge` 对应 `different_base_model`。

## 架构

```text
SGLang request/session
        |
        | model refs, prefix hash, experiment flags
        v
GoldenExperience planner
        |
        | ReusePlan: scenario, strategy, confidence, gates
        v
LMCache patch surface
        |
        | secondary lookup -> materialize/transform -> quality accounting
        v
LMCache storage/offload + SGLang inference remain upstream-owned
```

补丁面由 `PatchManifest.default()` 描述：

1. `sglang_request_metadata`：在 LMCache 查询前附加 `ModelRef` 和前缀元数据。
2. `lmcache_cross_model_lookup`：同模型未命中时，查询跨模型候选。
3. `goldenexperience_materializer`：在返回 KV 前对其进行别名、投影或转换。
4. `quality_gate_accounting`：记录置信度、校准状态和回退原因。

## 仓库结构

```text
goldenexperience/
  reuse/             ModelRef, KVShape, ReuseRequest, ReusePlan, scenario planner.
  lmcache_patch/     Patch manifest and sidecar key metadata for LMCache deltas.
  sglang_runtime/    Dependency checks and namespaced env helpers for wrappers.
  cache_core/        Legacy in-repo cache block metadata utilities for tests/prototypes.
  tiered_store/      Legacy synthetic tiering prototype; not the product runtime path.
  engine_adapter/    Legacy adapter experiments; SGLang is now the runtime target.
docs/                Design, experiment matrix, artifact, and paper planning notes.
configs/             Cross-model reuse experiment configuration.
examples/            Minimal planning examples.
scripts/             Optional bootstrap helpers.
tests/               Unit tests for the current framework and legacy utilities.
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

构建 Qwen2.5 7B <-> 14B 的 GoldenScale 校准脚手架：

```bash
golden-scale-collect --output artifacts/golden_scale/prompts.json
golden-scale-fit \
  --direction bidirectional \
  --prompt-manifest artifacts/golden_scale/prompts.json \
  --output-dir artifacts/golden_scale
golden-scale-validate artifacts/golden_scale/qwen25_7b_to_14b_projection_v0.json
golden-scale-bench artifacts/golden_scale/qwen25_14b_to_7b_projection_v0.json
```

## 部署流程

GoldenExperience 作为 Python 包部署在与 SGLang 和 LMCache 相同的环境中。它不是独立 daemon。

```text
client -> SGLang OpenAI-compatible server
             |
             | --enable-lmcache
             v
          LMCache
             |
             | GoldenExperience patch hooks and planner metadata
             v
          fallback or cross-model KV reuse
```

运行时职责：

- SGLang 启动推理服务器，并负责请求调度与生成。
- LMCache 负责 KV 查询、存储、卸载、淘汰和预取。
- GoldenExperience 负责 `ModelRef`、`ReuseRequest`、`ReusePlan`、补丁元数据以及质量/回退计数。

### 1. 安装运行时包

只需要运行栈时使用 package 模式：

```bash
python3 -m venv .venv
source .venv/bin/activate
./scripts/install_runtime.sh --mode package
```

需要补丁 LMCache 或调试 SGLang/LMCache 内部逻辑时使用 source 模式：

```bash
python3 -m venv .venv
source .venv/bin/activate
./scripts/install_runtime.sh --mode source
```

`--mode source` 会把 SGLang 和 LMCache 克隆到 `third_party/`，并以 editable 方式安装。使用 fork 时可以覆盖默认值：

```bash
GE_THIRD_PARTY_DIR=third_party \
GE_SGLANG_REPO_URL=https://github.com/sgl-project/sglang.git \
GE_LMCACHE_REPO_URL=https://github.com/LMCache/LMCache.git \
./scripts/install_runtime.sh --mode source
```

如果 SGLang 和 LMCache 已经可用，只安装 GoldenExperience：

```bash
./scripts/install_runtime.sh --mode golden-only
```

脚本在安装了 `uv` 时优先使用 `uv pip install`，否则回退到 `python3 -m pip install`。当修改 CUDA、Python 或包版本时，仍应对照上游文档检查 SGLang 与 LMCache 的安装细节：

- SGLang 文档：<https://docs.sglang.ai/>
- LMCache 文档：<https://docs.lmcache.ai/>

### 2. 验证 Planner 和导入

```bash
python3 scripts/smoke_cross_model_plan.py --check-runtime
```

预期 planner 输出包含三行：

- `model_lora_mutual_reuse`：已就绪的基座/LoRA 计划。
- `same_model_different_parameter_size`：已校准的 GoldenScale 投影计划。
- `different_base_model`：保守的、未就绪的跨基座计划。

如果 `--check-runtime` 报告缺少 `sglang` 或 `lmcache`，请先安装运行时栈，再启动基于模型的服务。

### 3. 生成补丁清单

```bash
golden-patch-manifest --output docs/patch_manifest.md
```

`scripts/start_sglang_lmcache.sh` 也会自动生成该清单。

### 4. 启动启用 LMCache 的 SGLang

默认启动命令会启动启用 LMCache 的 SGLang OpenAI 兼容服务器：

```bash
GE_MODEL_PATH=Qwen/Qwen3-8B \
./scripts/start_sglang_lmcache.sh
```

常见覆盖项：

```bash
GE_MODEL_PATH=Qwen/Qwen3-8B \
GE_HOST=0.0.0.0 \
GE_PORT=30000 \
GE_LMCACHE_CONFIG_FILE=artifacts/runtime/lmc_config.yaml \
GE_LMCACHE_CHUNK_SIZE=256 \
GE_LMCACHE_LOCAL_CPU_GB=10 \
./scripts/start_sglang_lmcache.sh --tp 1
```

启动脚本在 `exec python3 -m sglang.launch_server` 前会执行以下操作：

1. 如果 `GE_LMCACHE_CONFIG_FILE` 不存在，写入 LMCache 配置。
2. 渲染 `docs/patch_manifest.md`。
3. 检查 `sglang`、`lmcache` 和 `goldenexperience` 是否可导入。
4. 导出 GoldenExperience 元数据变量：
   - `GE_ENABLE_CROSS_MODEL_REUSE=1`
   - `GE_PATCH_MANIFEST=docs/patch_manifest.md`
   - `GE_LMCACHE_CONFIG=configs/lmcache.example.yaml`
   - `GE_SGLANG_MODEL_ID=$GE_MODEL_PATH`
5. 以 `--enable-lmcache` 启动 SGLang。

生成的 LMCache 配置刻意保持很小：

```yaml
chunk_size: 256
local_cpu: true
use_layerwise: true
max_local_cpu_size: 10
```

设置 `GE_OVERWRITE_LMCACHE_CONFIG=1` 可重新生成已有配置。

### 5. 发送请求

```bash
curl http://localhost:30000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-8B",
    "messages": [{"role": "user", "content": "Hello"}],
    "max_tokens": 64,
    "temperature": 0
  }'
```

### 6. 运行首个 KV 卸载/复用基线

当 SGLang、LMCache 和 GoldenExperience 已安装在同一个 Python 环境后，使用该基线。脚本会启动一个 SGLang + LMCache 服务器，发送确定性的 GSM8K 风格提示来填充/卸载 KV，随后使用同一个 LMCache 磁盘目录重启 SGLang，再发送相同提示，并记录可证明复用的耗时和日志证据。

```bash
source .venv/bin/activate

GE_MODEL_PATH=Qwen/Qwen3-8B \
GE_PORT=30000 \
scripts/kv_baseline/run_sglang_lmcache_kv_baseline.sh -- --tp 1
```

默认输出会写入 `artifacts/kv_baseline/<run_id>/`：

- `metadata.json`：模型、提示、模式、缓存路径和运行时设置。
- `lmc_config.yaml`：生成的 LMCache 配置，包含本地 CPU 与持久化本地磁盘。
- `requests/offload.json`：第一次请求的输出、usage、端到端延迟和 TTFT。
- `requests/reuse.json`：重启后使用相同提示的第二次请求。
- `logs/offload_server.log` 和 `logs/reuse_server.log`：SGLang/LMCache 证据。
- `summary.json`：请求差值，以及对 store/retrieve 事件的 best-effort 日志计数。

默认提示位于 `configs/kv_baseline_prompts.json`，使用经典的 GSM8K Natalia clips 问题。默认 `GE_KV_CHUNK_SIZE=16` 刻意设置得很小，确保这个短提示至少跨过一个 LMCache chunk。对于更长工作负载，可将其调回更接近你在 LMCache 实验中使用的生产风格值。

常用覆盖项：

```bash
GE_MODEL_PATH=/models/Qwen3-8B \
GE_MODEL_NAME=/models/Qwen3-8B \
GE_RUN_ID=qwen3_8b_gsm8k_restart_001 \
GE_KV_LOCAL_CPU_GB=20 \
GE_KV_LOCAL_DISK_GB=200 \
GE_PROMPT_ID=gsm8k_natalia_clips \
scripts/kv_baseline/run_sglang_lmcache_kv_baseline.sh -- --tp 1
```

解释检查清单：

1. 确认第一次运行完成，并且 `requests/offload.json` 包含预期答案。
2. 确认 `logs/offload_server.log` 包含 LMCache store/offload 消息。
3. 确认第二次运行从全新进程启动，并且 `logs/reuse_server.log` 包含 LMCache lookup/retrieve/hit 证据。
4. 比较 `summary.json` 中的 `reuse_minus_offload_ttft_ms` 和 `reuse_minus_offload_e2e_ms`；负值表示第二次请求更快。
5. 保留完整的 `artifacts/kv_baseline/<run_id>/` 目录，作为后续跨模型 KV 复用实验的同模型基线。

如果安装的 LMCache 版本无法在进程重启后重建本地磁盘元数据，请改为运行诊断用的同进程基线：

```bash
GE_MODEL_PATH=Qwen/Qwen3-8B \
GE_BASELINE_MODE=same-process \
scripts/kv_baseline/run_sglang_lmcache_kv_baseline.sh -- --tp 1
```

默认启用 `GE_DISABLE_RADIX_CACHE=1`，避免基线被 SGLang 的进程内 radix cache 隐藏。只有在希望测量 SGLang radix cache 与 LMCache 组合行为时，才设置 `GE_DISABLE_RADIX_CACHE=0`。

### 当前运行时限制

本仓库目前包含 GoldenExperience planner、元数据模型、补丁清单和部署封装。真正的 LMCache hook 实现是下一步。在 `lmcache_cross_model_lookup` 和 `goldenexperience_materializer` 被接入 LMCache 补丁或 fork 之前，SGLang + LMCache 会正常运行，GoldenExperience 可以验证计划/元数据，但被接受的跨模型 KV 复用还不会在 LMCache 内部执行。

## GoldenScale 复用

首个 GoldenScale MVP 面向 `Qwen/Qwen2.5-7B-Instruct` 与 `Qwen/Qwen2.5-14B-Instruct` 的双向复用。GoldenExperience 会把每个方向视为独立 artifact，因为 7B->14B 和 14B->7B 需要不同的层映射、投影规格、成本估计和质量门控。

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
- `ProjectionSpec`：源/目标 KV 宽度、KV heads、head dim、方法和 projection id。
- `QualityGateResult`：离线/影子门控指标，例如 KV cosine 和 perplexity drift。
- sidecar ids：`pair_id`、`direction`、`calibration_id`、`layer_map_id`、`projection_id`、源/目标配置哈希以及回退原因。

运行时行为保持保守：

- 前缀 token ids 必须完全匹配；要求 chunk 对齐。
- materializer 必须为每个目标层输出完整的目标形状 KV。
- `estimated_materialization_ms` 必须 <= 目标 prefill 成本的 70%。
- 任何 tokenizer、RoPE、配置哈希、artifact、层映射、投影或质量不匹配，都会回退到原始的 SGLang + LMCache 目标 prefill 路径。

## 最小 Planner 示例

```python
from goldenexperience.reuse import CrossModelReusePlanner, KVShape, ModelRef, ReuseRequest

shape = KVShape(num_layers=32, num_key_value_heads=8, head_dim=128)
base = ModelRef(
    model_id="qwen2.5-7b",
    family="qwen",
    architecture="qwen2",
    tokenizer_id="qwen2.5",
    parameter_count_b=7,
    kv_shape=shape,
)
lora = ModelRef(
    model_id="qwen2.5-7b-lora-math",
    family="qwen",
    architecture="qwen2",
    tokenizer_id="qwen2.5",
    parameter_count_b=7,
    base_model_id="qwen2.5-7b",
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

- M0：围绕 SGLang + LMCache + GoldenExperience 补丁元数据锁定项目边界。
- M1：为基座/LoRA 相互复用实现 LMCache 二级查询 sidecar。
- M2：为同模型不同规模变体添加层/头映射与校准投影。
- M3：为不同基座复用添加实验性的学习式转换器接口。
- M4：构建基于 SGLang 模型的 benchmark，以及质量/回退计数。
- M5：保持补丁足够小，以便随上游 LMCache rebase。

详见 `docs/design.md`、`docs/experiment_matrix.md` 和 `docs/artifact.md` 获取完整框架计划。
