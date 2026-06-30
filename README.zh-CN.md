# GoldenExperience

[English](README.md) | [中文](README.zh-CN.md)

GoldenExperience 是一个面向开源推理栈的 **跨模型 KV Cache 复用补丁框架**。当前同模型磁盘复用基线以 **vLLM + LMCache MP + filesystem L2** 为主线；`SGLang + LMCache` 保留为 legacy 对照路径。

## 项目边界

- vLLM 负责默认推理服务、调度、解码和生成正确性。
- LMCache MP 负责 KV 查询、存储、offload、eviction、prefetch 和 filesystem L2。
- GoldenExperience 负责模型身份、复用计划、补丁元数据和质量/回退统计。
- 当复用计划不安全或未校准时，系统回退到正常 prefill。

## 快速开始

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -e ".[dev]"
pytest
```

安装默认运行时：

```bash
./scripts/install_runtime.sh --mode package
```

默认会安装 `vllm` 和 `lmcache`。只有需要 SGLang legacy 对照路径时才加：

```bash
./scripts/install_runtime.sh --mode package --with-legacy-sglang
```

如果需要从源码调试 vLLM/LMCache：

```bash
GE_VLLM_REPO_URL=https://github.com/vllm-project/vllm.git \
GE_LMCACHE_REPO_URL=https://github.com/LMCache/LMCache.git \
./scripts/install_runtime.sh --mode source
```

检查 planner 与运行时导入：

```bash
python3 scripts/smoke_cross_model_plan.py --check-runtime
```

如需同时检查 SGLang legacy：

```bash
python3 scripts/smoke_cross_model_plan.py --check-runtime --check-legacy-sglang
```

## vLLM + LMCache MP 磁盘复用基线

目标路径：

```text
client
  -> vLLM OpenAI-compatible server
  -> LMCacheMPConnector
  -> standalone LMCache MP server
  -> filesystem L2 disk store
```

运行基线：

```bash
GE_MODEL_PATH=/models/Qwen3-8B \
GE_MODEL_NAME=/models/Qwen3-8B \
GE_REQUIRE_REUSE_EVIDENCE=1 \
GE_FORCE_DISK_OFFLOAD=1 \
scripts/kv_baseline/run_vllm_lmcache_mp_l2_baseline.sh -- --tensor-parallel-size 1
```

脚本会：

1. 启动独立 `lmcache server`。
2. 将 L2 adapter 设置为 filesystem 目录。
3. 启动 vLLM，并通过 `LMCacheMPConnector` 连接同一个 LMCache MP server。
4. 发送 offload 请求生成并写出 KV。
5. 只重启 vLLM，不重启 LMCache MP。
6. 再发送同一请求，收集 retrieve/hit 证据。

关键输出位于 `artifacts/kv_baseline/<run_id>/`：

- `runtime.json`：本次运行派生出的命令、路径和配置。
- `metadata.json`：模型、prompt、cache 路径和运行时设置。
- `lmc_config.yaml`：记录 LMCache MP 和 filesystem L2 配置。
- `requests/offload.json` 与 `requests/reuse.json`：两阶段请求结果。
- `logs/`：vLLM 与 LMCache MP 日志。
- `summary.json`：磁盘文件、reuse 命中和 PID 证据。

成功判据以 `summary.json` 为准：

- `offload_engine_pid != reuse_engine_pid`
- `lmcache_mp_pid` 非空
- `cache.file_count > 0`
- `cache.total_bytes > 0`
- `evidence.reuse_has_cache_evidence=true`
- `evidence.disk_reuse_success=true`

TTFT 变快只能作为性能参考，不能单独证明 KV 复用成功。

## SGLang Legacy 对照路径

旧脚本名 `scripts/kv_baseline/run_sglang_lmcache_kv_baseline.sh` 现在是兼容 wrapper，会转发到新的 vLLM 命名脚本。

运行旧的 SGLang + LMCache in-process 对照路径：

```bash
GE_KV_BACKEND=legacy \
GE_ENGINE=sglang \
GE_MODEL_PATH=/models/Qwen3-8B \
scripts/kv_baseline/run_vllm_lmcache_mp_l2_baseline.sh -- --tp 1
```

`scripts/start_sglang_lmcache.sh` 保留给 legacy/experimental 实验使用，不再是默认主线。

## 当前限制

当前分支先跑通 same-model disk KV reuse。真正的跨模型 `lmcache_cross_model_lookup` 与 `goldenexperience_materializer` 接入 LMCache 补丁或 fork 后，才会执行 GoldenExperience 的跨模型 KV 复用。
