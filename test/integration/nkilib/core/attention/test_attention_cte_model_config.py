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
Attention CTE model configuration data

Config format: dict with keys (bs, gqa_factor, seqlen_kv, seqlen_kv_prior, prior_used_len,
               cp_degree, cp_rank_id, d, sliding_window, causal_mask, tp_q, tp_k, tp_out, sink)

bs = n_q_heads per worker (after TP split)
gqa_factor = n_q_heads / n_kv_heads per worker
seqlen_kv = original sequence length (before CP split)
seqlen_q = seqlen_kv // cp_degree (computed in test)
tp_k: True for CP=1, False for CP>1 (strided CP)
"""

from test.integration.nkilib.core.attention.model_config_utils import get_sharded_head_counts
from test.utils.common_dataclasses import ModelTestType
from typing import TypedDict


class AttnModelConfig(TypedDict, total=False):
    TP_CP_CONFIGS: list[tuple[int, int, int]]
    SEQLENS: list[int]


MODELS = {
    "llama3_70b": {"n_q_heads": 64, "n_kv_heads": 8, "d_head": 128, "sliding_window": 0, "sink": None},
    "qwen3_32b": {"n_q_heads": 64, "n_kv_heads": 8, "d_head": 128, "sliding_window": 0, "sink": None},
    "qwen3_235b": {"n_q_heads": 64, "n_kv_heads": 4, "d_head": 128, "sliding_window": 0, "sink": None},
    "gemma3_27b": {
        "n_q_heads": 32,
        "n_kv_heads": 16,
        "d_head": 128,
        "sliding_window": 0,
        "sink": None,
        "swa_window": 1024,
    },
    "gptoss_120b": {
        "n_q_heads": 64,
        "n_kv_heads": 8,
        "d_head": 64,
        "sliding_window": 0,
        "sink": None,
        "swa_window": 128,
        "swa_sink": 4,
    },
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

OPTIMAL_CONFIGS: dict[str, AttnModelConfig] = {
    "llama3_70b": {"TP_CP_CONFIGS": DEFAULT_TP_CP, "SEQLENS": DEFAULT_SEQLENS},
    "qwen3_32b": {"TP_CP_CONFIGS": DEFAULT_TP_CP, "SEQLENS": DEFAULT_SEQLENS},
    "gemma3_27b": {"TP_CP_CONFIGS": DEFAULT_TP_CP, "SEQLENS": DEFAULT_SEQLENS},
    "gptoss_120b": {"TP_CP_CONFIGS": DEFAULT_TP_CP, "SEQLENS": DEFAULT_SEQLENS},
    "qwen3_235b": {
        "TP_CP_CONFIGS": [(64, 4, 16), (32, 4, 8), (16, 4, 4), (8, 4, 2), (64, 64, 1)],
        "SEQLENS": DEFAULT_SEQLENS,
    },
}


def get_attention_config(model_name, tp, cp, seqlen, sliding_window=0, sink=None):
    """Return attention config dict for a given model × sharding × seqlen."""
    m = MODELS.get(model_name)
    if m is None:
        return None
    n_q, n_kv = get_sharded_head_counts(tp, m["n_q_heads"], m["n_kv_heads"])
    gqa_factor = n_q // n_kv
    tp_k = cp == 1
    return {
        "bs": n_q,
        "gqa_factor": gqa_factor,
        "seqlen_kv": seqlen,
        "seqlen_kv_prior": None,
        "prior_used_len": None,
        "cp_degree": cp,
        "cp_rank_id": 0,
        "d": m["d_head"],
        "sliding_window": sliding_window,
        "causal_mask": True,
        "tp_q": True,
        "tp_k": tp_k,
        "tp_out": True,
        "sink": bool(sink) if sink is not None else None,
    }


def _is_compile_heavy(d):
    """Skip configs where bs * seqlen_q exceeds a threshold that causes >2 min compile times.

    Empirically, configs with bs * seqlen_q >= 4*32768 take 2-16 minutes to compile,
    dominating the test suite wall-clock time.  These configs are already covered by
    the manual unit tests at smaller scale and by the high-CP model configs.
    """
    seqlen_q = d["seqlen_kv"] // max(d["cp_degree"], 1)
    return d["bs"] * seqlen_q >= 4 * 32768


def generate_attention_base_configs(configs=None):
    if configs is None:
        configs = OPTIMAL_CONFIGS
    result = []
    for model_name in configs:
        c = configs[model_name]
        for orig_seqlen in c.get('SEQLENS', DEFAULT_SEQLENS):
            for _ws, tp, cp in c.get('TP_CP_CONFIGS', DEFAULT_TP_CP):
                d = get_attention_config(model_name, tp, cp, orig_seqlen)
                if d is not None and not _is_compile_heavy(d):
                    result.append((
                        d["bs"], d["gqa_factor"], d["seqlen_kv"], d["seqlen_kv_prior"], d["prior_used_len"],
                        d["cp_degree"], d["cp_rank_id"], d["d"], d["sliding_window"], d["causal_mask"],
                        d["tp_q"], d["tp_k"], d["tp_out"], d["sink"],
                    ))
    return result


def generate_swa_configs(configs=None):
    if configs is None:
        configs = OPTIMAL_CONFIGS
    result = []
    for model_name in configs:
        m = MODELS.get(model_name)
        if m is None or "swa_window" not in m:
            continue
        c = configs[model_name]
        for orig_seqlen in c.get('SEQLENS', DEFAULT_SEQLENS):
            for _ws, tp, cp in c.get('TP_CP_CONFIGS', DEFAULT_TP_CP):
                d = get_attention_config(
                    model_name, tp, cp, orig_seqlen, sliding_window=m["swa_window"], sink=m.get("swa_sink")
                )
                if d is not None and not _is_compile_heavy(d):
                    result.append((
                        d["bs"], d["gqa_factor"], d["seqlen_kv"], d["seqlen_kv_prior"], d["prior_used_len"],
                        d["cp_degree"], d["cp_rank_id"], d["d"], d["sliding_window"], d["causal_mask"],
                        d["tp_q"], d["tp_k"], d["tp_out"], d["sink"],
                    ))
    return result


# Combined list (deduplicated by value)
_all_optimal_configs = list(dict.fromkeys(generate_attention_base_configs() + generate_swa_configs()))

attention_cte_model_configs: dict[ModelTestType, list] = {
    ModelTestType.TIER0: [],
    ModelTestType.OPTIMAL: _all_optimal_configs,
    ModelTestType.GENERALITY: [],
}
