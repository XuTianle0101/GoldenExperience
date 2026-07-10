"""Fail-closed artifact contracts for cached Qwen3 KV translation."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

CACHED_KV_SCHEMA_VERSION = "goldenexperience.qwen3_cached_kv_bridge.v1"
_SHA256_LENGTH = 64
_TOKENIZER_FILES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
    "merges.txt",
)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_named_files(paths: Iterable[Path], *, root: Path) -> str:
    """Hash file names and contents so shard swaps cannot preserve an identity."""

    digest = hashlib.sha256()
    resolved_root = root.resolve()
    items = sorted((path.resolve() for path in paths), key=lambda path: str(path))
    if not items:
        raise ValueError(f"no files found under {root}")
    for path in items:
        try:
            name = path.relative_to(resolved_root).as_posix()
        except ValueError as exc:
            raise ValueError(f"{path} is outside {root}") from exc
        encoded_name = name.encode("utf-8")
        digest.update(len(encoded_name).to_bytes(4, "big"))
        digest.update(encoded_name)
        digest.update(bytes.fromhex(sha256_file(path)))
    return digest.hexdigest()


@dataclass(frozen=True)
class CachedKVModelSpec:
    """Content identity and KV layout for one side of a bridge."""

    model_id: str
    parameter_count_b: float
    revision: str
    architecture: str
    config_sha256: str
    tokenizer_sha256: str
    weights_sha256: str
    num_layers: int
    num_key_value_heads: int
    head_dim: int
    dtype: str
    rope_theta: float
    max_position_embeddings: int
    rope_scaling: dict[str, Any] | None = None
    sliding_window: int | None = None

    @property
    def kv_width(self) -> int:
        return self.num_key_value_heads * self.head_dim

    def validate(self) -> list[str]:
        errors: list[str] = []
        if not self.model_id:
            errors.append("model_id is required")
        if not self.revision:
            errors.append("model revision is required")
        if self.architecture != "qwen3":
            errors.append("cached KV bridge only supports qwen3")
        for name, value in (
            ("config_sha256", self.config_sha256),
            ("tokenizer_sha256", self.tokenizer_sha256),
            ("weights_sha256", self.weights_sha256),
        ):
            if not _is_sha256(value):
                errors.append(f"{name} must be a SHA-256 digest")
        if not math.isfinite(self.parameter_count_b) or self.parameter_count_b <= 0:
            errors.append("parameter_count_b must be finite and positive")
        if self.num_layers <= 0 or self.num_key_value_heads <= 0 or self.head_dim <= 0:
            errors.append("KV dimensions must be positive")
        if self.dtype not in {"bfloat16", "float16"}:
            errors.append("cached KV dtype must be bfloat16 or float16")
        if not math.isfinite(self.rope_theta) or self.rope_theta <= 0:
            errors.append("rope_theta must be finite and positive")
        if self.rope_scaling is not None:
            errors.append("RoPE scaling is not supported by cached KV bridge v1")
        if self.sliding_window is not None:
            errors.append("sliding-window attention is not supported by cached KV bridge v1")
        if self.max_position_embeddings <= 0:
            errors.append("max_position_embeddings must be positive")
        return errors


@dataclass(frozen=True)
class CachedKVQualityThresholds:
    """Minimum held-out evidence required before automatic reuse."""

    min_key_cosine: float = 0.95
    min_value_cosine: float = 0.95
    min_next_token_top1_agreement: float = 0.98
    max_perplexity_drift_pct: float = 2.0
    min_native_task_score: float = 0.95
    min_bridge_task_score: float = 0.95
    max_task_score_drop_pct: float = 1.0
    min_greedy_continuation_match_rate: float = 0.98
    max_materialization_to_prefill_ratio: float = 0.70
    min_held_out_prompts: int = 32
    min_task_prompts: int = 32
    required_token_buckets: tuple[int, ...] = (32, 128, 512, 2048)

    def validate(self) -> list[str]:
        errors: list[str] = []
        bounded = (
            ("min_key_cosine", self.min_key_cosine),
            ("min_value_cosine", self.min_value_cosine),
            ("min_next_token_top1_agreement", self.min_next_token_top1_agreement),
            ("min_native_task_score", self.min_native_task_score),
            ("min_bridge_task_score", self.min_bridge_task_score),
            ("min_greedy_continuation_match_rate", self.min_greedy_continuation_match_rate),
        )
        for name, value in bounded:
            if not math.isfinite(value) or not 0 <= value <= 1:
                errors.append(f"{name} must be finite and between 0 and 1")
        for name, value in (
            ("max_perplexity_drift_pct", self.max_perplexity_drift_pct),
            ("max_task_score_drop_pct", self.max_task_score_drop_pct),
            ("max_materialization_to_prefill_ratio", self.max_materialization_to_prefill_ratio),
        ):
            if not math.isfinite(value) or value < 0:
                errors.append(f"{name} must be finite and non-negative")
        if self.min_held_out_prompts <= 0:
            errors.append("min_held_out_prompts must be positive")
        if self.min_task_prompts <= 0:
            errors.append("min_task_prompts must be positive")
        if not self.required_token_buckets or any(
            item <= 0 for item in self.required_token_buckets
        ):
            errors.append("required_token_buckets must contain positive lengths")
        return errors


@dataclass(frozen=True)
class CachedKVQualityEvidence:
    """Measured held-out accuracy and cost evidence."""

    evaluation_dataset_sha256: str
    held_out_prompts: int
    evaluated_tokens: int
    token_buckets: tuple[int, ...]
    key_cosine: float
    value_cosine: float
    next_token_top1_agreement: float
    perplexity_drift_pct: float
    task_prompts: int
    native_task_score: float
    bridge_task_score: float
    task_score_drop_pct: float
    greedy_continuation_match_rate: float
    p95_source_read_transform_put_ms: float | None
    p95_target_prefill_ms: float | None

    def validate(self) -> list[str]:
        errors: list[str] = []
        if not _is_sha256(self.evaluation_dataset_sha256):
            errors.append("evaluation_dataset_sha256 must be a SHA-256 digest")
        if self.held_out_prompts <= 0:
            errors.append("held_out_prompts must be positive")
        if self.evaluated_tokens <= 0:
            errors.append("evaluated_tokens must be positive")
        if self.task_prompts <= 0:
            errors.append("task_prompts must be positive")
        if not self.token_buckets or any(item <= 0 for item in self.token_buckets):
            errors.append("token_buckets must contain positive lengths")
        bounded = (
            ("key_cosine", self.key_cosine),
            ("value_cosine", self.value_cosine),
            ("next_token_top1_agreement", self.next_token_top1_agreement),
            ("native_task_score", self.native_task_score),
            ("bridge_task_score", self.bridge_task_score),
            ("greedy_continuation_match_rate", self.greedy_continuation_match_rate),
        )
        for name, value in bounded:
            if not math.isfinite(value) or not 0 <= value <= 1:
                errors.append(f"{name} must be finite and between 0 and 1")
        for name, value in (
            ("perplexity_drift_pct", self.perplexity_drift_pct),
            ("task_score_drop_pct", self.task_score_drop_pct),
        ):
            if not math.isfinite(value) or value < 0:
                errors.append(f"{name} must be finite and non-negative")
        if self.p95_source_read_transform_put_ms is None:
            errors.append("p95_source_read_transform_put_ms is required")
        elif (
            not math.isfinite(self.p95_source_read_transform_put_ms)
            or self.p95_source_read_transform_put_ms < 0
        ):
            errors.append("p95_source_read_transform_put_ms must be finite and non-negative")
        if self.p95_target_prefill_ms is None:
            errors.append("p95_target_prefill_ms is required")
        elif not math.isfinite(self.p95_target_prefill_ms) or self.p95_target_prefill_ms <= 0:
            errors.append("p95_target_prefill_ms must be finite and positive")
        return errors

    def gate_errors(self, thresholds: CachedKVQualityThresholds) -> list[str]:
        errors = self.validate() + thresholds.validate()
        if errors:
            return errors
        if self.held_out_prompts < thresholds.min_held_out_prompts:
            errors.append("held-out prompt count is below threshold")
        if self.task_prompts < thresholds.min_task_prompts:
            errors.append("task prompt count is below threshold")
        if not set(thresholds.required_token_buckets) <= set(self.token_buckets):
            errors.append("held-out evaluation is missing required token buckets")
        if self.key_cosine < thresholds.min_key_cosine:
            errors.append("key cosine is below threshold")
        if self.value_cosine < thresholds.min_value_cosine:
            errors.append("value cosine is below threshold")
        if self.next_token_top1_agreement < thresholds.min_next_token_top1_agreement:
            errors.append("next-token top1 agreement is below threshold")
        if self.perplexity_drift_pct > thresholds.max_perplexity_drift_pct:
            errors.append("perplexity drift is above threshold")
        if self.native_task_score < thresholds.min_native_task_score:
            errors.append("native task score is below threshold")
        if self.bridge_task_score < thresholds.min_bridge_task_score:
            errors.append("bridge task score is below threshold")
        if self.task_score_drop_pct > thresholds.max_task_score_drop_pct:
            errors.append("task score drop is above threshold")
        if self.greedy_continuation_match_rate < thresholds.min_greedy_continuation_match_rate:
            errors.append("greedy continuation match rate is below threshold")
        assert self.p95_source_read_transform_put_ms is not None
        assert self.p95_target_prefill_ms is not None
        ratio = self.p95_source_read_transform_put_ms / self.p95_target_prefill_ms
        if ratio > thresholds.max_materialization_to_prefill_ratio:
            errors.append("materialization cost ratio is above threshold")
        return errors


@dataclass(frozen=True)
class CachedKVBridgeManifest:
    """Direction-specific, content-addressed cached-KV bridge manifest."""

    bridge_id: str
    direction: str
    source: CachedKVModelSpec
    target: CachedKVModelSpec
    weights_uri: str
    weights_sha256: str
    rank: int
    source_window: int
    train_dataset_sha256: str
    validation_dataset_sha256: str
    test_dataset_sha256: str
    quality: CachedKVQualityEvidence
    thresholds: CachedKVQualityThresholds = field(default_factory=CachedKVQualityThresholds)
    schema_version: str = CACHED_KV_SCHEMA_VERSION
    scope: str = "global"
    layout: str = "kv_layer_token_width"
    method: str = "joint_kv_low_rank_residual"
    rope_convention: str = "qwen_half_split"

    @property
    def approved(self) -> bool:
        return not self.validate()

    def validate(self) -> list[str]:
        errors: list[str] = []
        if self.schema_version != CACHED_KV_SCHEMA_VERSION:
            errors.append("unsupported cached KV bridge schema_version")
        if not self.bridge_id:
            errors.append("bridge_id is required")
        if self.direction not in {"8b_to_14b", "14b_to_8b"}:
            errors.append("direction must be 8b_to_14b or 14b_to_8b")
        errors.extend(f"source: {item}" for item in self.source.validate())
        errors.extend(f"target: {item}" for item in self.target.validate())
        if self.direction == "8b_to_14b" and not (
            self.source.parameter_count_b < self.target.parameter_count_b
        ):
            errors.append("8b_to_14b direction has reversed model sizes")
        if self.direction == "14b_to_8b" and not (
            self.source.parameter_count_b > self.target.parameter_count_b
        ):
            errors.append("14b_to_8b direction has reversed model sizes")
        if self.source.model_id == self.target.model_id:
            errors.append("source and target model identities must differ")
        if self.source.tokenizer_sha256 != self.target.tokenizer_sha256:
            errors.append("source and target tokenizer identities differ")
        if self.source.dtype != self.target.dtype:
            errors.append("source and target KV dtypes differ")
        if self.source.num_key_value_heads != self.target.num_key_value_heads:
            errors.append("source and target KV head counts differ")
        if self.source.head_dim != self.target.head_dim:
            errors.append("source and target head dimensions differ")
        if self.source.rope_theta != self.target.rope_theta:
            errors.append("source and target rope_theta differ")
        if self.source.max_position_embeddings != self.target.max_position_embeddings:
            errors.append("source and target maximum positions differ")
        if self.source.rope_scaling != self.target.rope_scaling:
            errors.append("source and target RoPE scaling differs")
        if self.source.sliding_window != self.target.sliding_window:
            errors.append("source and target sliding-window contracts differ")
        if self.layout != "kv_layer_token_width":
            errors.append("unsupported cached KV object layout")
        if self.method != "joint_kv_low_rank_residual":
            errors.append("unsupported cached KV bridge method")
        if self.rope_convention != "qwen_half_split":
            errors.append("unsupported RoPE convention")
        if self.rank <= 0:
            errors.append("bridge rank must be positive")
        if self.source_window <= 0 or self.source_window > self.source.num_layers:
            errors.append("source_window is outside source depth")
        if self.scope != "global":
            errors.append("automatic cross-prompt reuse requires a global artifact")
        if not self.weights_uri.endswith(".safetensors"):
            errors.append("cached KV bridge weights must use safetensors")
        if not _is_sha256(self.weights_sha256):
            errors.append("weights_sha256 must be a SHA-256 digest")
        dataset_hashes = (
            self.train_dataset_sha256,
            self.validation_dataset_sha256,
            self.test_dataset_sha256,
        )
        if any(not _is_sha256(item) for item in dataset_hashes):
            errors.append("train/validation/test dataset SHA-256 digests are required")
        if len(set(dataset_hashes)) != len(dataset_hashes):
            errors.append("train/validation/test dataset identities must be disjoint")
        if self.quality.evaluation_dataset_sha256 != self.test_dataset_sha256:
            errors.append("quality evidence must refer to the held-out test dataset")
        errors.extend(self.quality.gate_errors(self.thresholds))
        if (
            self.bridge_id
            and _is_sha256(self.weights_sha256)
            and self.bridge_id != artifact_id_for(self)
        ):
            errors.append("bridge_id does not match the content-addressed manifest")
        return errors

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> CachedKVBridgeManifest:
        source = CachedKVModelSpec(**payload["source"])
        target = CachedKVModelSpec(**payload["target"])
        quality_payload = dict(payload["quality"])
        quality_payload["token_buckets"] = tuple(quality_payload.get("token_buckets", ()))
        threshold_payload = dict(payload.get("thresholds", {}))
        if "required_token_buckets" in threshold_payload:
            threshold_payload["required_token_buckets"] = tuple(
                threshold_payload["required_token_buckets"]
            )
        return cls(
            bridge_id=payload["bridge_id"],
            direction=payload["direction"],
            source=source,
            target=target,
            weights_uri=payload["weights_uri"],
            weights_sha256=payload["weights_sha256"],
            rank=int(payload["rank"]),
            source_window=int(payload["source_window"]),
            train_dataset_sha256=payload["train_dataset_sha256"],
            validation_dataset_sha256=payload["validation_dataset_sha256"],
            test_dataset_sha256=payload["test_dataset_sha256"],
            quality=CachedKVQualityEvidence(**quality_payload),
            thresholds=CachedKVQualityThresholds(**threshold_payload),
            schema_version=payload.get("schema_version", ""),
            scope=payload.get("scope", ""),
            layout=payload.get("layout", ""),
            method=payload.get("method", ""),
            rope_convention=payload.get("rope_convention", ""),
        )

    @classmethod
    def load(cls, path: str | Path) -> CachedKVBridgeManifest:
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    def save(self, path: str | Path) -> None:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    def resolve_weights_path(self, manifest_path: str | Path) -> Path:
        path = Path(self.weights_uri)
        if not path.is_absolute():
            path = Path(manifest_path).resolve().parent / path
        return path


def model_spec_from_path(
    model_path: str | Path,
    *,
    model_id: str,
    parameter_count_b: float,
    revision: str,
) -> CachedKVModelSpec:
    """Build a strong local model identity, including every safetensors shard."""

    root = Path(model_path).resolve()
    config_path = root / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    tokenizer_paths = [root / name for name in _TOKENIZER_FILES if (root / name).is_file()]
    if not tokenizer_paths:
        raise ValueError(f"tokenizer files are missing under {root}")
    weight_paths = sorted(root.glob("*.safetensors"))
    if not weight_paths:
        raise ValueError(f"safetensors model weights are missing under {root}")
    architecture = str(config.get("model_type", ""))
    dtype = str(config.get("torch_dtype", ""))
    return CachedKVModelSpec(
        model_id=model_id,
        parameter_count_b=float(parameter_count_b),
        revision=revision,
        architecture=architecture,
        config_sha256=sha256_file(config_path),
        tokenizer_sha256=sha256_named_files(tokenizer_paths, root=root),
        weights_sha256=sha256_named_files(weight_paths, root=root),
        num_layers=int(config["num_hidden_layers"]),
        num_key_value_heads=int(config["num_key_value_heads"]),
        head_dim=int(config["head_dim"]),
        dtype=dtype,
        rope_theta=float(config["rope_theta"]),
        max_position_embeddings=int(config["max_position_embeddings"]),
        rope_scaling=config.get("rope_scaling"),
        sliding_window=config.get("sliding_window"),
    )


def verify_model_path(expected: CachedKVModelSpec, model_path: str | Path) -> list[str]:
    observed = model_spec_from_path(
        model_path,
        model_id=expected.model_id,
        parameter_count_b=expected.parameter_count_b,
        revision=expected.revision,
    )
    errors: list[str] = []
    for field_name in CachedKVModelSpec.__dataclass_fields__:
        if getattr(observed, field_name) != getattr(expected, field_name):
            errors.append(f"{field_name} does not match bridge artifact")
    return errors


def artifact_id_for(manifest: CachedKVBridgeManifest) -> str:
    """Derive a stable ID from the manifest contract and weight content digest."""

    payload = manifest.to_dict()
    payload["bridge_id"] = ""
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return "qwen3-kv-" + hashlib.sha256(raw.encode("utf-8")).hexdigest()[:20]


def _is_sha256(value: str | None) -> bool:
    if not isinstance(value, str) or len(value) != _SHA256_LENGTH:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True
