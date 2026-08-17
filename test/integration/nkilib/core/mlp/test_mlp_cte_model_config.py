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
MLP CTE model configuration data

Config format: [vnc_degree, batch, seqlen, hidden, intermediate, tpbSgCyclesSum, rtol, norm_type, quant_type, gate_up_w_layout,
                fused_add, store_add, skip_gate, act_fn_type, gate_bias, up_bias, down_bias, norm_bias]
"""

import math

from test.utils.common_dataclasses import ModelTestType
from nkilib_src.nkilib.core.utils.common_types import ActFnType, MLPGateUpWeightLayout, NormType, QuantizationType

# Only dense (non-MoE) models
MODELS = {
    "llama3_70b": {"hidden": 8192, "intermediate": 28672, "act_fn": ActFnType.SiLU, "bias": False},
    "qwen3_32b": {"hidden": 5120, "intermediate": 25600, "act_fn": ActFnType.SiLU, "bias": False},
    "gemma3_27b": {"hidden": 5376, "intermediate": 21504, "act_fn": ActFnType.GELU, "bias": False},
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
    QuantizationType.MX,
    QuantizationType.STATIC_MX,
    QuantizationType.ROW_MX,
]

OPTIMAL_CONFIGS = {name: {'TP_CP_CONFIGS': DEFAULT_TP_CP, 'SEQLENS': DEFAULT_SEQLENS, 'QUANT_TYPES': DEFAULT_QUANT_TYPES} for name in MODELS}

def _align_intermediate(inter_raw, quant_type=None):
    """Align intermediate dim. STATIC_MX requires I % 512 == 0."""
    if inter_raw < 3584:
        return math.ceil(inter_raw / 512) * 512
    return math.ceil(inter_raw / 1024) * 1024


def get_mlp_config(model_name, tp, cp, seqlen, quant_type):
    """Return MLP config dict for a given model × sharding × seqlen × quant."""
    m = MODELS.get(model_name)
    if m is None:
        return None
    qt = quant_type if hasattr(quant_type, 'name') else QuantizationType[quant_type]
    gu_l = MLPGateUpWeightLayout.H_X4_INNERMOST if QuantizationType.is_mx(qt) else MLPGateUpWeightLayout.CONTIGUOUS
    intermediate_tp = _align_intermediate(m["intermediate"] // tp, qt)
    hidden_padded = math.ceil(m["hidden"] / 512) * 512
    qt_name = qt.name if hasattr(qt, 'name') else qt
    af_name = m["act_fn"].name if hasattr(m["act_fn"], 'name') else m["act_fn"]
    return {
        "vnc_degree": 2, "batch": 1, "seqlen": seqlen // cp,
        "hidden": hidden_padded, "intermediate": intermediate_tp,
        "norm_type": "NO_NORM", "quant_type": qt_name,
        "gate_up_w_layout": gu_l,
        "fused_add": False, "store_add": False, "skip_gate": False,
        "act_fn_type": af_name,
        "gate_bias": m["bias"], "up_bias": m["bias"],
        "down_bias": m["bias"], "norm_bias": m["bias"],
    }


def generate_mlp_configs(configs=None):
    if configs is None:
        configs = OPTIMAL_CONFIGS
    result = []
    for model_name, m in MODELS.items():
        c = configs.get(model_name, {})
        for orig_seqlen in c.get('SEQLENS', DEFAULT_SEQLENS):
            for _ws, tp, cp in c.get('TP_CP_CONFIGS', DEFAULT_TP_CP):
                for quant_type in c.get('QUANT_TYPES', DEFAULT_QUANT_TYPES):
                    d = get_mlp_config(model_name, tp, cp, orig_seqlen, quant_type)
                    rtol = 6.1e-2 if quant_type == QuantizationType.MX else 4e-2
                    result.append((
                        d["vnc_degree"], d["batch"], d["seqlen"], d["hidden"], d["intermediate"],
                        None, rtol, NormType.NO_NORM, quant_type, d["gate_up_w_layout"],
                        d["fused_add"], d["store_add"], d["skip_gate"], m["act_fn"],
                        d["gate_bias"], d["up_bias"], d["down_bias"], d["norm_bias"],
                    ))
    return result


# Combined list (deduplicated by value)
_all_optimal_configs = list(dict.fromkeys(generate_mlp_configs()))

# Organized by tier - all CTE model configs go to OPTIMAL
mlp_cte_model_configs: dict[ModelTestType, list] = {
    ModelTestType.TIER0: [],
    ModelTestType.OPTIMAL: _all_optimal_configs,
    ModelTestType.GENERALITY: [],
}
