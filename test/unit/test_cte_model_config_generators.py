# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Enforce that all model config get_*_config functions return dicts.

These functions are consumed by external packages via dict key access.
Returning tuples instead of dicts would silently break downstream consumers.
If you need a tuple for pytest parametrize, convert in the generate_* function.
"""

import pytest
from nkilib_src.nkilib.core.utils.common_types import QuantizationType

CONFIG_FUNCTIONS = []


def _register(import_path, func_name, kwargs):
    """Lazily import and register a config function for testing."""
    import importlib

    mod = importlib.import_module(import_path)
    fn = getattr(mod, func_name)
    CONFIG_FUNCTIONS.append(pytest.param(fn, kwargs, id=func_name))


_register(
    "test.integration.nkilib.core.attention.test_attention_cte_model_config",
    "get_attention_config",
    {"model_name": "llama3_70b", "tp": 8, "cp": 1, "seqlen": 1024},
)
_register(
    "test.integration.nkilib.core.attention.test_attention_segmented_cte_model_config",
    "get_segmented_attention_config",
    {"model_name": "qwen3_235b", "tp": 4, "kvp": 2, "block_size": 64, "max_len": 16384},
)
_register(
    "test.integration.nkilib.core.mlp.test_mlp_cte_model_config",
    "get_mlp_config",
    {"model_name": "llama3_70b", "tp": 8, "cp": 1, "seqlen": 1024, "quant_type": QuantizationType.NONE},
)
_register(
    "test.integration.nkilib.core.output_projection.test_output_proj_cte_model_config",
    "get_output_proj_config",
    {"model_name": "llama3_70b", "tp": 8, "cp": 1, "seqlen": 1024, "quant_type": QuantizationType.NONE},
)
_register(
    "test.integration.nkilib.core.qkv.test_qkv_cte_model_config",
    "get_qkv_config",
    {"model_name": "llama3_70b", "tp": 8, "cp": 1, "seqlen": 1024, "quant_type": QuantizationType.NONE},
)
_register(
    "test.integration.nkilib.core.rmsnorm.test_rmsnorm_quant_cte_model_config",
    "get_rmsnorm_config",
    {"model_name": "llama3_70b", "tp": 8, "cp": 1, "seqlen": 1024},
)


def _get_moe_kwargs():
    return {
        "model_name": "gptoss_120b",
        "seq_len": 1024,
        "variant": "shard_on_block",
        "block_size": 128,
        "skewness": 0.5,
        "tp": 2,
        "ep": 32,
    }


_register(
    "test.integration.nkilib.core.moe.moe_cte.test_moe_bwmm_mx_cte_model_config",
    "get_moe_bwmm_mx_config",
    _get_moe_kwargs(),
)

_register(
    "test.integration.nkilib.core.moe.moe_cte.test_moe_cte_model_config",
    "get_moe_bwmm_bf16_config",
    {"model_name": "qwen3_235b", "seq_len": 4096, "tp": 1, "ep": 64},
)


@pytest.mark.parametrize("func, kwargs", CONFIG_FUNCTIONS)
def test_config_function_returns_dict(func, kwargs):
    """Every get_*_config function must return a dict so external consumers can access fields by name."""
    result = func(**kwargs)
    assert result is not None, f"{func.__name__} returned None for test inputs"
    assert isinstance(result, dict), (
        f"{func.__name__} must return a dict, got {type(result).__name__}. "
        f"If you need a tuple for pytest parametrize, convert in the generate_* function."
    )
