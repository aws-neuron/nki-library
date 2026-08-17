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
Output Projection CTE model configuration data.

Config format: (batch, seqlen, hidden, n_head, d_head, test_bias, quant_type)

Note: For output projection, n_head corresponds to n_q_heads from QKV configs
since output projection operates on the attention output which has n_q_heads.
"""

import math

from nkilib_src.nkilib.core.utils.common_types import QuantizationType

MODELS = {
    "llama3_70b": {"n_q_heads": 64, "n_kv_heads": 8, "d_head": 128, "hidden": 8192, "bias": False},
    "qwen3_32b": {"n_q_heads": 64, "n_kv_heads": 8, "d_head": 128, "hidden": 5120, "bias": False},
    "qwen3_vl_32b": {"n_q_heads": 64, "n_kv_heads": 8, "d_head": 128, "hidden": 5120, "bias": False},
    "qwen3_235b": {"n_q_heads": 64, "n_kv_heads": 4, "d_head": 128, "hidden": 4096, "bias": False},
    "gemma3_27b": {"n_q_heads": 32, "n_kv_heads": 16, "d_head": 128, "hidden": 5376, "bias": False},
    "gptoss_120b": {"n_q_heads": 64, "n_kv_heads": 8, "d_head": 64, "hidden": 3072, "bias": True},
}

# (world_size, tp, cp)
DEFAULT_TP_CP = [
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

DEFAULT_SEQLENS = [1024, 10240, 32768]

DEFAULT_QUANT_TYPES = [
    QuantizationType.NONE,
    QuantizationType.STATIC,
    QuantizationType.ROW,
    QuantizationType.STATIC_MX,
    QuantizationType.ROW_MX,
]


def _get_sharded_head_counts(tp, n_q_heads, n_kv_heads):
    padded_q = math.ceil(n_q_heads / tp) * tp
    if n_q_heads == n_kv_heads:
        padded_kv = padded_q
    elif n_kv_heads < tp or n_kv_heads % tp != 0:
        padded_kv = tp if tp % n_kv_heads == 0 else padded_q
    else:
        padded_kv = n_kv_heads
    return padded_q // tp, padded_kv // tp


def get_output_proj_config(model_name, tp, cp, seqlen, quant_type):
    """Return output_proj config dict for a given model × sharding × seqlen × quant."""
    m = MODELS.get(model_name)
    if m is None:
        return None
    n_q, _ = _get_sharded_head_counts(tp, m["n_q_heads"], m["n_kv_heads"])
    qt = quant_type.name if hasattr(quant_type, 'name') else quant_type
    return {
        "batch": 1,
        "seqlen": seqlen // cp,
        "hidden": m["hidden"],
        "n_head": n_q,
        "d_head": m["d_head"],
        "test_bias": m["bias"],
        "quant_type": qt,
    }


OPTIMAL_CONFIGS = {
    "llama3_70b": {"TP_CP_CONFIGS": DEFAULT_TP_CP, "SEQLENS": DEFAULT_SEQLENS, "QUANT_TYPES": DEFAULT_QUANT_TYPES},
    "qwen3_32b": {"TP_CP_CONFIGS": DEFAULT_TP_CP, "SEQLENS": DEFAULT_SEQLENS, "QUANT_TYPES": DEFAULT_QUANT_TYPES},
    "qwen3_vl_32b": {
        "TP_CP_CONFIGS": [(64, 64, 1), (32, 32, 1), (16, 16, 1), (8, 8, 1), (4, 4, 1)],
        "SEQLENS": [4096, 8192],
        "QUANT_TYPES": DEFAULT_QUANT_TYPES + [QuantizationType.MX],
        # Online-MX (QuantizationType.MX) weights are native FP8 for this model, not FP4.
        # Read by the test harness to pick the weight dtype for MX rows; "fp4" is the default.
        "MX_WEIGHT_DTYPE": "fp8",
    },
    "gemma3_27b": {"TP_CP_CONFIGS": DEFAULT_TP_CP, "SEQLENS": DEFAULT_SEQLENS, "QUANT_TYPES": DEFAULT_QUANT_TYPES},
    "gptoss_120b": {"TP_CP_CONFIGS": DEFAULT_TP_CP, "SEQLENS": DEFAULT_SEQLENS, "QUANT_TYPES": DEFAULT_QUANT_TYPES},
    "qwen3_235b": {
        "TP_CP_CONFIGS": [(64, 4, 16), (32, 4, 8), (16, 4, 4), (8, 4, 2), (64, 64, 1)],
        "SEQLENS": DEFAULT_SEQLENS,
        "QUANT_TYPES": DEFAULT_QUANT_TYPES,
    },
}


def generate_output_proj_configs(configs=None):
    if configs is None:
        configs = OPTIMAL_CONFIGS
    results = []
    for model_name in configs:
        c = configs[model_name]
        m = MODELS.get(model_name)
        if m is None:
            continue
        for orig_seqlen in c.get('SEQLENS', DEFAULT_SEQLENS):
            for _ws, tp, cp in c.get('TP_CP_CONFIGS', DEFAULT_TP_CP):
                for qt in c.get('QUANT_TYPES', DEFAULT_QUANT_TYPES):
                    d = get_output_proj_config(model_name, tp, cp, orig_seqlen, qt)
                    results.append((d['batch'], d['seqlen'], d['hidden'], d['n_head'], d['d_head'], d['test_bias'], qt))
    return results


def get_mx_weight_dtype(configs=None):
    """Weight dtype token ("fp4"/"fp8") used for online-MX (QuantizationType.MX) rows.

    Online MX rows only originate from a model whose QUANT_TYPES includes
    QuantizationType.MX (only qwen3_vl_32b today), so a single token suffices.
    Defaults to "fp4"; a model opts into native FP8 weights via MX_WEIGHT_DTYPE.
    """
    if configs is None:
        configs = OPTIMAL_CONFIGS
    for c in configs.values():
        if QuantizationType.MX in c.get('QUANT_TYPES', DEFAULT_QUANT_TYPES):
            return c.get('MX_WEIGHT_DTYPE', 'fp4')
    return 'fp4'


from test.utils.common_dataclasses import ModelTestType

# Combined list (deduplicated by value)
_all_optimal_configs = list(dict.fromkeys(generate_output_proj_configs()))

output_proj_cte_model_configs: dict[ModelTestType, list] = {
    ModelTestType.TIER0: [],
    ModelTestType.OPTIMAL: _all_optimal_configs,
    ModelTestType.GENERALITY: [],
}
