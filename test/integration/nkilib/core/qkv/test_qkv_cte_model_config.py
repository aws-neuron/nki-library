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

import os as _os
if _os.environ.get("SKIP_MODEL_TESTS"):
    raise ImportError("Model tests skipped via SKIP_MODEL_TESTS")

"""
QKV CTE model configuration data.

Generates configs for all models × quant_types × sharding × seqlens.
Common params: (model_name, quant_type, batch, seqlen, hidden_dim,
                n_q_heads, n_kv_heads, d_head, qkv_bias,
                fused_rope, use_gamma, in_scale_shape, w_scale_shape)
"""

import math

from typing import TypedDict

from nkilib_src.nkilib.core.utils.common_types import QuantizationType
from test.integration.nkilib.core.attention.model_config_utils import get_sharded_head_counts
from test.integration.nkilib.core.qkv.qkv_model_metadata import MODELS

_DEFAULT_TP_CP: list[tuple[int, int, int]] = [
    (64, 64, 1),
    (16, 16, 1),
    (8, 8, 1),
    (4, 4, 1),
    (16, 8, 2),
    (8, 4, 2),
    (64, 16, 4),
    (64, 8, 8),
    (64, 4, 16),
]
_DEFAULT_SEQLENS: list[int] = [1024, 10240, 32768]


class QkvModelConfig(TypedDict, total=False):
    TP_CP_CONFIGS: list[tuple[int, int, int]]
    SEQLENS: list[int]
    QUANT_TYPES: list[QuantizationType]


OPTIMAL_CONFIGS: dict[str, QkvModelConfig] = {
    "llama3_70b": {"TP_CP_CONFIGS": _DEFAULT_TP_CP, "SEQLENS": _DEFAULT_SEQLENS},
    "qwen3_32b": {"TP_CP_CONFIGS": _DEFAULT_TP_CP, "SEQLENS": _DEFAULT_SEQLENS},
    "gemma3_27b": {"TP_CP_CONFIGS": _DEFAULT_TP_CP, "SEQLENS": _DEFAULT_SEQLENS},
    "gptoss_120b": {"TP_CP_CONFIGS": _DEFAULT_TP_CP, "SEQLENS": _DEFAULT_SEQLENS},
    "qwen3_235b": {
        "TP_CP_CONFIGS": [(64, 4, 16), (32, 4, 8), (16, 4, 4), (8, 4, 2), (64, 64, 1)],
        "SEQLENS": _DEFAULT_SEQLENS,
    },
}

DEFAULT_QUANT_TYPES = [
    QuantizationType.NONE,
    QuantizationType.STATIC,
    QuantizationType.STATIC_MX,
]

# Model → MX variant mapping
QK_NORM_MODELS = {"gemma3_27b", "qwen3_235b"}
STATIC_DEQUANT_MODELS = {"llama3_70b", "gptoss_120b"}
FUSED_GAMMA_ROPE_MODELS = {"qwen3_32b"}


def get_qkv_config(model_name, tp, cp, seqlen, quant_type):
    """Return QKV config dict for a given model × sharding × seqlen × quant."""
    m = MODELS.get(model_name)
    if m is None:
        return None
    n_q, n_kv = get_sharded_head_counts(tp, m["n_q_heads"], m["n_kv_heads"])
    seqlen_cp = seqlen // cp
    hidden_padded = math.ceil(m["hidden"] / 512) * 512
    qt = quant_type.name if hasattr(quant_type, 'name') else quant_type

    if model_name in STATIC_DEQUANT_MODELS:
        fused_rope = seqlen_cp <= 96
        in_scale_shape = [1, 1]
        w_scale_shape = [1, 3]
    else:
        fused_rope = True
        in_scale_shape = None
        w_scale_shape = None

    return {
        "model_name": model_name,
        "quant_type": qt,
        "batch": 1,
        "seqlen": seqlen_cp,
        "hidden_dim": hidden_padded,
        "n_q_heads": n_q,
        "n_kv_heads": n_kv,
        "d_head": m["d_head"],
        "qkv_bias": m["bias"],
        "fused_rope": fused_rope,
        "use_gamma": m.get("use_gamma", False),
        "in_scale_shape": in_scale_shape,
        "w_scale_shape": w_scale_shape,
    }


def generate_qkv_configs(configs=None):
    """Generate QKV configs for all models × quant_types × sharding × seqlens."""
    if configs is None:
        configs = OPTIMAL_CONFIGS
    result = []
    for model_name in configs:
        c = configs[model_name]
        for quant_type in c.get("QUANT_TYPES", DEFAULT_QUANT_TYPES):
            for orig_seqlen in c.get("SEQLENS", _DEFAULT_SEQLENS):
                for _ws, tp, cp in c.get("TP_CP_CONFIGS", _DEFAULT_TP_CP):
                    d = get_qkv_config(model_name, tp, cp, orig_seqlen, quant_type)
                    result.append(
                        (
                            d['model_name'],
                            quant_type,
                            d['batch'],
                            d['seqlen'],
                            d['hidden_dim'],
                            d['n_q_heads'],
                            d['n_kv_heads'],
                            d['d_head'],
                            d['qkv_bias'],
                            d['fused_rope'],
                            d['use_gamma'],
                            tuple(d['in_scale_shape']) if d['in_scale_shape'] else None,
                            tuple(d['w_scale_shape']) if d['w_scale_shape'] else None,
                        )
                    )
    return result


from test.utils.common_dataclasses import ModelTestType

_all_optimal_configs = list(dict.fromkeys(generate_qkv_configs()))

qkv_cte_model_configs: dict[ModelTestType, list] = {
    ModelTestType.TIER0: [],
    ModelTestType.OPTIMAL: _all_optimal_configs,
    ModelTestType.GENERALITY: [],
}
