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
MoE BWMM MX CTE model configuration data.

Generates configs for all models × variants × EP/MoETP × seqlens × block_sizes × skewness.

Test params dict keys (matches MOE_BWMM_MX_CTE_MODEL_PARAMS in test_moe_bwmm_mx_cte.py):
    variant       - "shard_on_block" or "shard_on_I", selects kernel variant
    vnc_degree    - virtual neuron core degree (always 2 for these kernels)
    hidden        - hidden dimension (H)
    tokens        - sequence length / number of tokens (T)
    intermediate  - intermediate dimension per TP shard (I_TP), padded to 512 (block) or 1024 (I)
    expert        - number of local experts (E_local = total_experts / ep_degree)
    block_size    - block size for blockwise MM (B), e.g. 128, 256, 512
    act_fn        - activation function type (Swish, SiLU, etc.)
    expert_affinities_scaling_mode - how expert affinities are scaled (always POST_SCALE as PRE_SCALE is not implemented yet)
    dtype         - activation data type (always bfloat16)
    weight_dtype  - MX weight data type (float4_e2m1fn_x4, float8_e4m3fn_x4, float8_e5m2_x4)
    skip_mode     - DMA skip mode (0=no skip, 1=skip token)
    bias          - whether bias is used in gate/up projections
    is_dynamic    - whether to use dynamic (runtime condition-checked) block processing
    gate_clamp_upper - upper clamp limit for gate projection (None = no clamp)
    gate_clamp_lower - lower clamp limit for gate projection (None = no clamp)
    up_clamp_upper   - upper clamp limit for up projection (None = no clamp)
    up_clamp_lower   - lower clamp limit for up projection (None = no clamp)
    use_uint_weights - whether to simulate NxD uint16/uint32 weight format
    skewness_pct  - expert affinity skewness [0.0, 1.0], controls routing imbalance, see explanations below
    global_top_k  - global top-k for expert routing (before EP sharding)
    ep_degree     - expert parallelism degree

SKEWNESS_PCTS controls the expert affinity distribution used to generate test
routing tables.  The value is a float in [0.0, 1.0] that interpolates between
the best-case and worst-case number of nonzero expert affinities:

    num_non_zero = best + skewness_pct * (worst - best)

where:
    best  = T * global_top_k / ep_degree   (perfectly balanced across EP shards)
    worst = T * min(E_local, global_top_k)  (all tokens routed to every local expert)

Examples (T=4096, global_top_k=8, total_experts=128):
    EP64 (E_local=2):
        best  = 4096*8/64  = 512
        worst = 4096*2     = 8192
        skew 0.0  → 512   non-zero entries  (best case, minimal work)
        skew 0.25 → 2432  non-zero entries
        skew 0.5  → 4352  non-zero entries
        skew 1.0  → 8192  non-zero entries  (worst case, maximum work)
    EP8 (E_local=16):
        best  = 4096*8/8   = 4096
        worst = 4096*8     = 32768
        skew 0.0  → 4096  non-zero entries
        skew 1.0  → 32768 non-zero entries

Use [1.0] for worst-case-only testing (e.g., GPT-OSS) or a sweep like
[0.0, 0.25, 0.5, 1.0] to cover the full range (e.g., Qwen3-235B).
"""

import math
from test.utils.common_dataclasses import ModelTestType

import nki.language as nl
from nkilib_src.nkilib.core.utils.common_types import ActFnType, ExpertAffinityScaleMode

from typing import Any, TypedDict


class MoeBwmmMxModelConfig(TypedDict, total=False):
    BWMM_VARIANTS: list[str]
    EP_MOETP_CONFIGS: list[tuple[int, int]]
    SEQ_LENS: list[int]
    SKEWNESS_PCTS: list[float]
    SHARD_ON_BLOCK_BLK_SIZES: list[int]
    SHARD_ON_BLOCK_WDTS: list[Any]
    SHARD_ON_BLOCK_IS_DYNAMIC: bool
    SHARD_ON_I_BLK_SIZES: list[int]
    SHARD_ON_I_WDTS: list[Any]
    SHARD_ON_I_IS_DYNAMIC: bool

# ============================================================================
# Model definitions
# ============================================================================


class MoeModelSpec(TypedDict):
    hidden: int
    full_intermediate: int
    total_experts: int
    global_top_k: int
    act_fn: ActFnType
    bias: bool
    gate_clamp_upper: float | None
    gate_clamp_lower: float | None
    up_clamp_upper: float | None
    up_clamp_lower: float | None


MODELS: dict[str, MoeModelSpec] = {
    "gptoss_120b": {
        "hidden": 3072,
        "full_intermediate": 3072,
        "total_experts": 128,
        "global_top_k": 4,
        "act_fn": ActFnType.Swish,
        "bias": True,
        "gate_clamp_upper": 7.0,
        "gate_clamp_lower": None,
        "up_clamp_upper": 7.0,
        "up_clamp_lower": -7.0,
    },
    "qwen3_235b": {
        "hidden": 4096,
        "full_intermediate": 1536,
        "total_experts": 128,
        "global_top_k": 8,
        "act_fn": ActFnType.SiLU,
        "bias": False,
        "gate_clamp_upper": None,
        "gate_clamp_lower": None,
        "up_clamp_upper": None,
        "up_clamp_lower": None,
    },
}

# ============================================================================
# Per-model test sweep configurations
#
# EP_MOETP_CONFIGS: list of (ep_degree, moe_tp_degree)
#   I_TP is derived as: pad(full_intermediate / moe_tp_degree)
#     shard-on-block: pad to multiple of 512
#     shard-on-I:     pad to multiple of 1024
# ============================================================================

_DEFAULT_BWMM_VARIANTS: list[str] = ["shard_on_block", "shard_on_I"]
_DEFAULT_WDTS: list[Any] = [nl.float8_e4m3fn_x4]

OPTIMAL_CONFIGS: dict[str, MoeBwmmMxModelConfig] = {
    # NOTE: Now each test takes more than 10 mins to run, to not timeout pipeline, we comment out all the optimal configs.
    # Once an optimization to the test golden gen algo is merged, we can re-enable them. But make sure they pass the tests before uncomment and merge.
    "gptoss_120b": {
        "BWMM_VARIANTS": ["shard_on_block"],
        "EP_MOETP_CONFIGS": [
            (1, 8),  # EP1TP8: E=128
            (4, 2),  # EP4TP2: E=32
            (32, 2),  # EP32TP2: E=4
            (8, 8),  # EP8TP8: E=16
        ],
        "SEQ_LENS": [1024, 4096, 10240], # 32768 enable later when tests can be run faster
        "SKEWNESS_PCTS": [1.0],
        "SHARD_ON_BLOCK_BLK_SIZES": [128, 256],
        "SHARD_ON_BLOCK_WDTS": [nl.float4_e2m1fn_x4, nl.float8_e4m3fn_x4],
        "SHARD_ON_BLOCK_IS_DYNAMIC": True,
    },
    "qwen3_235b": {
        "EP_MOETP_CONFIGS": [
            (64, 1),  # EP64 MoETP1
            (16, 2),  # EP16 MoETP2
            (8, 2),  # EP8  MoETP2
        ],
        "SEQ_LENS": [10240],
        "SKEWNESS_PCTS": [0.25],
        "SHARD_ON_BLOCK_BLK_SIZES": [256],
        "SHARD_ON_I_BLK_SIZES": [512],
    },
}

GENERALITY_CONFIGS: dict[str, MoeBwmmMxModelConfig] = {
    "qwen3_235b": {
        # NOTE: these are generality configs for manual benchmark purpose as the optimal ones are not identified yet and we need to ease the burden of pipeline.
        # NOTE: Some configs would fail accuracy validation marginally, so validate it before moving anything to OPTIMAL_CONFIGS which would be run by pipeline.
        "EP_MOETP_CONFIGS": [
            (64, 1),  # EP64 MoETP1
            (32, 2),  # EP32 MoETP2
            (32, 1),  # EP32 MoETP1
            (16, 2),  # EP16 MoETP2
            (16, 1),  # EP16 MoETP1
            (8, 2),  # EP8  MoETP2
        ],
        "SEQ_LENS": [4096, 10240, 32768],
        "SKEWNESS_PCTS": [0.0, 0.25, 0.5, 1.0],
        "SHARD_ON_BLOCK_BLK_SIZES": [256],
        "SHARD_ON_I_BLK_SIZES": [256, 512],
    },
}


# ============================================================================
# Helpers
# ============================================================================


def _pad_i_tp(full_intermediate, moe_tp_degree, alignment):
    """Compute padded I_TP = ceil(full_intermediate / moe_tp_degree / alignment) * alignment."""
    return math.ceil(full_intermediate / moe_tp_degree / alignment) * alignment


def get_moe_bwmm_mx_config(model_name, seq_len, variant, block_size, skewness, tp, ep,
                            weight_dtype=nl.float8_e4m3fn_x4, is_dynamic=True):
    """Return a single MoE BWMM MX config dict for a given model × variant × sharding × seqlen."""
    m = MODELS.get(model_name)
    if m is None:
        return None
    alignment = 512 if variant == "shard_on_block" else 1024
    i_tp = _pad_i_tp(m["full_intermediate"], tp, alignment)
    expert = m["total_experts"] // ep
    return {
        "variant": variant,
        "vnc_degree": 2,
        "hidden": m["hidden"],
        "tokens": seq_len,
        "intermediate": i_tp,
        "expert": expert,
        "block_size": block_size,
        "act_fn": m["act_fn"],
        "expert_affinities_scaling_mode": ExpertAffinityScaleMode.POST_SCALE,
        "dtype": nl.bfloat16,
        "weight_dtype": weight_dtype,
        "skip_mode": 1,
        "bias": m["bias"],
        "is_dynamic": is_dynamic,
        "gate_clamp_upper": m["gate_clamp_upper"],
        "gate_clamp_lower": m["gate_clamp_lower"],
        "up_clamp_upper": m["up_clamp_upper"],
        "up_clamp_lower": m["up_clamp_lower"],
        "use_uint_weights": False,
        "skewness_pct": skewness,
        "global_top_k": m["global_top_k"],
        "ep_degree": ep,
    }


# ============================================================================
# Config generator
# ============================================================================


def _dict_to_tuple(d):
    """Convert config dict to the parametrize tuple expected by tests."""
    return (
        d["variant"], d["vnc_degree"], d["hidden"], d["tokens"], d["intermediate"],
        d["expert"], d["block_size"], d["act_fn"], d["expert_affinities_scaling_mode"],
        d["dtype"], d["weight_dtype"], d["skip_mode"], d["bias"], d["is_dynamic"],
        d["gate_clamp_upper"], d["gate_clamp_lower"], d["up_clamp_upper"],
        d["up_clamp_lower"], d["use_uint_weights"], d["skewness_pct"],
        d["global_top_k"], d["ep_degree"],
    )


def generate_moe_bwmm_mx_configs(configs=None):
    """Generate MoE BWMM MX configs for all models × variants × EP/MoETP × seqlens."""
    if configs is None:
        configs = OPTIMAL_CONFIGS
    result = []
    for model_name in configs:
        m = MODELS[model_name]
        c = configs[model_name]
        for variant in c.get("BWMM_VARIANTS", _DEFAULT_BWMM_VARIANTS):
            if variant == "shard_on_block":
                variant_key = "SHARD_ON_BLOCK"
                assert "SHARD_ON_BLOCK_BLK_SIZES" in c, f"{model_name}: {variant} variant requires {variant_key}_BLK_SIZES"
                blk_sizes = c["SHARD_ON_BLOCK_BLK_SIZES"]
                is_dynamic = c.get("SHARD_ON_BLOCK_IS_DYNAMIC", True)
                wdts = c.get("SHARD_ON_BLOCK_WDTS", _DEFAULT_WDTS)
            else:
                variant_key = "SHARD_ON_I"
                assert "SHARD_ON_I_BLK_SIZES" in c, f"{model_name}: {variant} variant requires {variant_key}_BLK_SIZES"
                blk_sizes = c["SHARD_ON_I_BLK_SIZES"]
                is_dynamic = c.get("SHARD_ON_I_IS_DYNAMIC", True)
                wdts = c.get("SHARD_ON_I_WDTS", _DEFAULT_WDTS)
            for ep, mtp in c["EP_MOETP_CONFIGS"]:
                assert m["total_experts"] % ep == 0, f"total_experts has to be divisible by EP"
                for seqlen in c["SEQ_LENS"]:
                    for blk in blk_sizes:
                        for wdt in wdts:
                            for skew in c["SKEWNESS_PCTS"]:
                                d = get_moe_bwmm_mx_config(
                                    model_name, seqlen, variant, blk, skew, mtp, ep,
                                    weight_dtype=wdt, is_dynamic=is_dynamic,
                                )
                                result.append(_dict_to_tuple(d))
    return result


_all_optimal_configs = list(dict.fromkeys(generate_moe_bwmm_mx_configs()))
_all_generality_configs = list(dict.fromkeys(generate_moe_bwmm_mx_configs(GENERALITY_CONFIGS)))

moe_bwmm_mx_cte_model_configs: dict[ModelTestType, list] = {
    ModelTestType.OPTIMAL: _all_optimal_configs,
    ModelTestType.GENERALITY: _all_generality_configs,
}

