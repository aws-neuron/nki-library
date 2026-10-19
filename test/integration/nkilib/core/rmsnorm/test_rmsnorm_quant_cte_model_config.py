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
RMSNorm Quant CTE model configuration data.  @IGNORE_NEW_TEST_GUIDELINES

Config format: (seqlen, quant_only, quant_type, tensor_gen, lower_bound, lnc_degree, batch, hidden)

RMSNorm operates on the full hidden dimension (not sharded by TP).
With sequence parallelism, effective seqlen per worker = original_seqlen / (TP * CP).
Configs are deduplicated since RMSNorm latency depends only on (seqlen, hidden).

Mapping (original_seqlen -> effective seqlen per TP*CP config):
  WS=64 (TP64-CP1, TP16-CP4, TP8-CP8, TP4-CP16): 1024->16,   10240->160,  32768->512
  WS=16 (TP16-CP1, TP8-CP2):                       1024->64,   10240->640,  32768->2048
  WS=8  (TP8-CP1, TP4-CP2):                        1024->128,  10240->1280, 32768->4096
  WS=4  (TP4-CP1):                                 1024->256,  10240->2560, 32768->8192
"""

import numpy as np

from test.utils.common_dataclasses import ModelTestType
from nkilib_src.nkilib.core.utils.common_types import QuantizationType
from test.integration.nkilib.utils.tensor_generators import gaussian_tensor_generator

from typing import TypedDict


class RmsnormModelConfig(TypedDict, total=False):
    TP_CP_CONFIGS: list[tuple[int, int, int]]
    SEQLENS: list[int]
    QUANT_TYPES: list[QuantizationType]

MODELS = {
    "llama3_70b": {"hidden": 8192, "fused_residual": False},
    "qwen3_32b": {"hidden": 5120, "fused_residual": False},
    "qwen3_235b": {"hidden": 4096, "fused_residual": False},
    "gemma3_27b": {"hidden": 5376, "fused_residual": True},
    "gptoss_120b": {"hidden": 3072, "fused_residual": False},
}

# (world_size, tp, cp)
DEFAULT_TP_CP: list[tuple[int, int, int]] = [
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

DEFAULT_SEQLENS: list[int] = [1024, 10240, 32768]

DEFAULT_QUANT_TYPES: list[QuantizationType] = [QuantizationType.STATIC, QuantizationType.ROW]

OPTIMAL_CONFIGS: dict[str, RmsnormModelConfig] = {name: {'TP_CP_CONFIGS': DEFAULT_TP_CP, 'SEQLENS': DEFAULT_SEQLENS} for name in MODELS}


def _static_scale_wrapper(default_tensor_generator):
    rng = np.random.default_rng(0)

    def tensor_generator(shape, dtype, name):
        if name == "input_dequant_scale":
            return np.full(shape, rng.normal(loc=0.5, scale=0.1), dtype)
        return default_tensor_generator(shape, dtype, name)

    return tensor_generator


_tgen = _static_scale_wrapper(gaussian_tensor_generator())


def get_rmsnorm_config(model_name, tp, cp, seqlen):
    """Return rmsnorm config dict for a given model × sharding × seqlen."""
    m = MODELS.get(model_name)
    if m is None:
        return None
    seqlen_sp = seqlen // (tp * cp)
    return {
        "seqlen": seqlen_sp, "quant_only": False,
        "lower_bound": 0.0, "lnc_degree": 2, "batch": 1, "hidden": m["hidden"],
        "fused_residual": m["fused_residual"],
    }


def generate_rmsnorm_configs(configs=None):
    if configs is None:
        configs = OPTIMAL_CONFIGS
    result = []
    for model_name, c in configs.items():
        for orig_seqlen in c.get('SEQLENS', DEFAULT_SEQLENS):
            for _ws, tp, cp in c.get('TP_CP_CONFIGS', DEFAULT_TP_CP):
                d = get_rmsnorm_config(model_name, tp, cp, orig_seqlen)
                for qt in c.get('QUANT_TYPES', DEFAULT_QUANT_TYPES):
                    result.append((d['seqlen'], d['quant_only'], qt, d['lower_bound'], d['lnc_degree'], d['batch'], d['hidden'], d['fused_residual']))
    return result


_seen = set()
_all_optimal_configs = []
for t in generate_rmsnorm_configs():
    if t not in _seen:
        _seen.add(t)
        _all_optimal_configs.append(t)

# Organized by tier - all configs go to OPTIMAL
rmsnorm_quant_cte_model_configs: dict[ModelTestType, list] = {
    ModelTestType.TIER0: [],
    ModelTestType.OPTIMAL: _all_optimal_configs,
    ModelTestType.GENERALITY: [],
}
