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
MoE CTE (non-MX) model configuration data.

Generates configs for all models × variants × EP/MoETP × seqlens.

Test params tuple (matches MOE_CTE_MODEL_PARAMS in test_moe_cte.py):
    bwmm_func     - BWMMFunc enum selecting the kernel variant
    hidden        - hidden dimension (H)
    tokens        - sequence length / number of tokens (T)
    expert        - number of local experts (E_local = total_experts / ep_degree)
    block_size    - block size for blockwise MM (B)
    intermediate  - intermediate dimension per TP shard (I_TP)
    dtype         - activation data type (always bfloat16)
    skip          - DMA skip mode (0=no skip, 1=skip token)
    bias          - whether bias is used in gate/up projections
    training      - whether training mode (checkpoint activations)
    quantize      - quantization type (None for non-quantized)
    act_fn        - activation function type (Swish, SiLU, etc.)
    expert_affinities_scaling_mode - how expert affinities are scaled
    gate_cl_upper - upper clamp limit for gate projection (None = no clamp)
    gate_cl_lower - lower clamp limit for gate projection (None = no clamp)
    up_cl_upper   - upper clamp limit for up projection (None = no clamp)
    up_cl_lower   - lower clamp limit for up projection (None = no clamp)
    expert_affinity_multiply_on_I - whether to multiply affinities on I dim
    skewness_pct  - expert affinity skewness [0.0, 1.0], controls routing imbalance
    global_top_k  - global top-k for expert routing (before EP sharding)
    ep_degree     - expert parallelism degree
"""

import nki.language as nl

from nkilib_src.nkilib.core.utils.common_types import ActFnType, ExpertAffinityScaleMode
from test.integration.nkilib.core.moe.moe_cte.test_moe_cte_common import BWMMFunc
from test.utils.common_dataclasses import ModelTestType

from typing import TypedDict


class MoeCteModelConfig(TypedDict, total=False):
    EP_MOETP_CONFIGS: list[tuple[int, int]]
    SEQ_LENS: list[int]
    SKEWNESS_PCTS: list[float]
    BWMM_FUNC: BWMMFunc
    BLOCK_SIZE: int
    SKIP: int
    EXPERT_AFFINITY_MULTIPLY_ON_I: bool

# ============================================================================
# Model definitions
# ============================================================================

MODELS = {
    "llama4": {
        "hidden": 5120,
        "full_intermediate": 8192,
        "total_experts": 16,
        "global_top_k": 1,
        "act_fn": ActFnType.SiLU,
        "bias": False,
    },
    "deepseek_v3": {
        "hidden": 2048,
        "full_intermediate": 1408,
        "total_experts": 64,
        "global_top_k": 6,
        "act_fn": ActFnType.SiLU,
        "bias": False,
    },
    "qwen3_30b_a3b": {
        "hidden": 2048,
        "full_intermediate": 768,
        "total_experts": 128,
        "global_top_k": 8,
        "act_fn": ActFnType.SiLU,
        "bias": False,
    },
    "qwen3_235b": {
        "hidden": 4096,
        "full_intermediate": 1536,
        "total_experts": 128,
        "global_top_k": 8,
        "act_fn": ActFnType.SiLU,
        "bias": False,
    },
}

# ============================================================================
# Per-model test sweep configurations
# ============================================================================

_DEFAULT_BWMM_FUNC = BWMMFunc.SHARD_ON_INTERMEDIATE_HW

GENERALITY_CONFIGS: dict[str, MoeCteModelConfig] = {
    "llama4": {
        "EP_MOETP_CONFIGS": [
            (1, 1),  # EP1 MoETP1
            (1, 4),  # EP1 MoETP4
        ],
        "SEQ_LENS": [4096],
        "SKEWNESS_PCTS": [0.25],
        "BWMM_FUNC": BWMMFunc.SHARD_ON_INTERMEDIATE,
        "SKIP": 0,
        "EXPERT_AFFINITY_MULTIPLY_ON_I": True,
    },
    "deepseek_v3": {
        "EP_MOETP_CONFIGS": [
            (1, 1),  # EP1 MoETP1
            (1, 4),  # EP1 MoETP4
        ],
        "SEQ_LENS": [4096],
        "SKEWNESS_PCTS": [0.25],
        "BWMM_FUNC": BWMMFunc.SHARD_ON_INTERMEDIATE,
        "SKIP": 0,
        "EXPERT_AFFINITY_MULTIPLY_ON_I": True,
    },
    "qwen3_30b_a3b": {
        "EP_MOETP_CONFIGS": [
            (1, 1),  # EP1 MoETP1
            (1, 4),  # EP1 MoETP4
        ],
        "SEQ_LENS": [4096],
        "SKEWNESS_PCTS": [0.25],
        "BWMM_FUNC": BWMMFunc.SHARD_ON_INTERMEDIATE,
        "SKIP": 0,
        "EXPERT_AFFINITY_MULTIPLY_ON_I": True,
    },
    "qwen3_235b": {
        "EP_MOETP_CONFIGS": [
            (64, 1),  # EP64 MoETP1
            (32, 2),  # EP32 MoETP2
            (32, 1),  # EP32 MoETP1
            (16, 2),  # EP16 MoETP2
        ],
        "SEQ_LENS": [4096, 8192, 10240, 16384, 32768],
        "SKEWNESS_PCTS": [0.25],
    },
}


# ============================================================================
# Helpers
# ============================================================================


def get_moe_bwmm_bf16_config(model_name, seq_len, tp, ep, bwmm_func=_DEFAULT_BWMM_FUNC,
                              block_size=512, skip=1, skewness=0.25,
                              expert_affinity_multiply_on_I=False):
    """Return a single MoE BWMM BF16 config dict for a given model × sharding × seqlen."""
    m = MODELS.get(model_name)
    if m is None:
        return None
    i_tp = m["full_intermediate"] // tp
    expert = m["total_experts"] // ep
    return {
        "bwmm_func": bwmm_func,
        "hidden": m["hidden"],
        "tokens": seq_len,
        "expert": expert,
        "block_size": block_size,
        "intermediate": i_tp,
        "dtype": nl.bfloat16,
        "skip": skip,
        "bias": m["bias"],
        "training": False,
        "quantize": False,
        "act_fn": m["act_fn"],
        "expert_affinities_scaling_mode": ExpertAffinityScaleMode.POST_SCALE,
        "gate_cl_upper": None,
        "gate_cl_lower": None,
        "up_cl_upper": None,
        "up_cl_lower": None,
        "expert_affinity_multiply_on_I": expert_affinity_multiply_on_I,
        "skewness_pct": skewness,
        "global_top_k": m["global_top_k"],
        "ep_degree": ep,
    }


def _dict_to_tuple(d):
    """Convert config dict to the parametrize tuple expected by tests."""
    return (
        d["bwmm_func"], d["hidden"], d["tokens"], d["expert"], d["block_size"],
        d["intermediate"], d["dtype"], d["skip"], d["bias"], d["training"],
        d["quantize"], d["act_fn"], d["expert_affinities_scaling_mode"],
        d["gate_cl_upper"], d["gate_cl_lower"], d["up_cl_upper"], d["up_cl_lower"],
        d["expert_affinity_multiply_on_I"], d["skewness_pct"], d["global_top_k"],
        d["ep_degree"],
    )


# ============================================================================
# Config generator
# ============================================================================


def generate_moe_cte_configs(configs=None):
    """Generate MoE CTE configs for all models × EP/MoETP × seqlens."""
    if configs is None:
        configs = GENERALITY_CONFIGS
    result = []
    for model_name in configs:
        c = configs[model_name]
        bwmm_func = c.get("BWMM_FUNC", _DEFAULT_BWMM_FUNC)
        block_size = c.get("BLOCK_SIZE", 512)
        skip = c.get("SKIP", 1)
        multiply_on_I = c.get("EXPERT_AFFINITY_MULTIPLY_ON_I", False)
        for ep, mtp in c["EP_MOETP_CONFIGS"]:
            for seqlen in c["SEQ_LENS"]:
                for skew in c["SKEWNESS_PCTS"]:
                    d = get_moe_bwmm_bf16_config(
                        model_name, seqlen, mtp, ep,
                        bwmm_func=bwmm_func, block_size=block_size,
                        skip=skip, skewness=skew,
                        expert_affinity_multiply_on_I=multiply_on_I,
                    )
                    result.append(_dict_to_tuple(d))
    return result


_all_generality_configs = list(dict.fromkeys(generate_moe_cte_configs()))

moe_cte_model_configs: dict[ModelTestType, list] = {
    ModelTestType.GENERALITY: _all_generality_configs,
}
