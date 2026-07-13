from types import SimpleNamespace

import pytest

from goldenexperience.engine_adapter.transformers_adapter import TransformersAdapter
from goldenexperience.model_config import (
    ModelConfigError,
    optional_rope_theta,
    resolve_dtype,
    resolve_head_dim,
    resolve_rope_theta,
)


def test_resolvers_support_transformers_5_style_config() -> None:
    config = SimpleNamespace(
        hidden_size=4096,
        num_attention_heads=32,
        rope_parameters={"rope_theta": 1_000_000, "rope_type": "default"},
        dtype="torch.bfloat16",
    )

    assert resolve_head_dim(config) == 128
    assert resolve_rope_theta(config) == 1_000_000
    assert resolve_dtype(config) == "bfloat16"


def test_resolvers_support_raw_json_and_legacy_fields() -> None:
    config = {
        "head_dim": 64,
        "rope_theta": 10_000,
        "torch_dtype": "fp16",
    }

    assert resolve_head_dim(config) == 64
    assert resolve_rope_theta(config) == 10_000
    assert resolve_dtype(config) == "float16"


def test_rope_theta_falls_back_to_nested_scaling() -> None:
    assert resolve_rope_theta({"rope_scaling": {"rope_theta": "500000"}}) == 500_000
    assert optional_rope_theta({}) is None
    with pytest.raises(ModelConfigError, match="does not expose"):
        resolve_rope_theta({})


def test_rope_theta_rejects_conflicting_layouts() -> None:
    with pytest.raises(ModelConfigError, match="conflicting"):
        resolve_rope_theta(
            {
                "rope_theta": 10_000,
                "rope_parameters": {"rope_theta": 1_000_000},
            }
        )


def test_modern_dtype_field_avoids_deprecated_accessor() -> None:
    class Config:
        dtype = "torch.bfloat16"

        @property
        def torch_dtype(self):
            raise AssertionError("deprecated torch_dtype accessor was read")

    assert resolve_dtype(Config()) == "bfloat16"
    assert resolve_dtype({}, default="half") == "float16"


@pytest.mark.parametrize(
    "config",
    [
        {"head_dim": 0},
        {"head_dim": True},
        {"hidden_size": 10, "num_attention_heads": 3},
        {"hidden_size": None, "num_attention_heads": 2},
    ],
)
def test_head_dim_rejects_invalid_layout(config) -> None:
    with pytest.raises(ModelConfigError):
        resolve_head_dim(config)


@pytest.mark.parametrize("theta", [0, -1, float("inf"), True, "invalid"])
def test_rope_theta_rejects_invalid_values(theta) -> None:
    with pytest.raises(ModelConfigError):
        resolve_rope_theta({"rope_parameters": {"rope_theta": theta}})


def test_transformers_adapter_uses_shared_resolvers() -> None:
    class Config:
        _name_or_path = "Qwen/Qwen3-test"
        _commit_hash = "revision"
        architectures = ["Qwen3ForCausalLM"]
        dtype = "torch.bfloat16"
        hidden_size = 256
        num_attention_heads = 8
        num_hidden_layers = 4
        num_key_value_heads = 2
        rope_parameters = {"rope_theta": 1_000_000, "rope_type": "default"}
        rope_scaling = None
        sliding_window = None
        vocab_size = 1000

        def to_dict(self):
            return {"model_type": "qwen3"}

    adapter = TransformersAdapter(SimpleNamespace(config=Config()))
    signature = adapter.architecture_signature

    assert signature.head_dim == 32
    assert signature.rope_theta == 1_000_000
    assert signature.dtype == "bfloat16"
