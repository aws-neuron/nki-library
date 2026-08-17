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
Segmented attention CTE model configuration data.

Tests the segmented prefill kernel on the *last* segment of a sequence with total
length max_len.  The prior context (i.e. all tokens before the current segment)
has length (max_len - seg_len), which is then sharded across KVP ranks so each
rank sees prior_tokens = ceil((max_len - seg_len) / kvp) tokens (rounded up to
block_size).

Config format: dict with keys (bs, num_q_heads, num_kv_heads, block_size, prior_seg_size,
               head_dim, prior_tokens, tp_q, tp_out, dtype)

Segmented prefill sharding:
    - TP shards Q/KV heads across ranks
    - KVP further shards the KV cache sequence dimension within a KV head group
    - seg_len = the active segment length being computed
    - prior_tokens = ceil((max_len - seg_len) / kvp), rounded up to block_size
"""

import math

from typing import TypedDict

import nki.language as nl

from test.integration.nkilib.core.attention.model_config_utils import get_sharded_head_counts
from test.utils.common_dataclasses import ModelTestType


class SegmentedAttnModelConfig(TypedDict, total=False):
    TP_KVP_CONFIGS: list[tuple[int, int]]
    BLK_SIZE: list[int]
    MAX_LENS: list[int]

MODELS = {
    "qwen3_235b": {
        "n_q_heads": 64,
        "n_kv_heads": 4,
        "d_head": 128,
        "seg_len": 4096,  # NOTE: 4096 is the largest seg_len we support currently
    },
}

OPTIMAL_CONFIGS: dict[str, SegmentedAttnModelConfig] = {
    "qwen3_235b": {
        "TP_KVP_CONFIGS": [
            # (tp, kvp)
            (4, 2),
            (4, 4),
            (4, 8),
            (4, 16),
        ],
        "BLK_SIZE": [32, 64, 128],
        "MAX_LENS": [16384, 32768],
    },
}


def get_segmented_attention_config(model_name, tp, kvp, block_size, max_len):
    """Return segmented attention config dict for a given model × sharding.

    Args:
        model_name: Key into MODELS dict.
        tp: Tensor parallel degree (shards Q/KV heads).
        kvp: KV parallel degree (shards KV cache sequence dimension).
        block_size: Logical block size for KV cache.
        max_len: Maximum sequence length.

    Returns:
        Dict with keys (bs, num_q_heads, num_kv_heads, block_size, prior_seg_size,
        head_dim, prior_tokens, tp_q, tp_out, dtype) or None.
    """
    m = MODELS.get(model_name)
    if m is None:
        return None

    seg_len = m["seg_len"]
    assert seg_len % block_size == 0, f"seg_len ({seg_len}) must be divisible by block_size ({block_size})"

    # Round max_len up to multiple of block_size
    max_len = math.ceil(max_len / block_size) * block_size
    assert max_len >= seg_len, f"max_len ({max_len}) must be >= seg_len ({seg_len})"

    num_q_heads, num_kv_heads = get_sharded_head_counts(tp, m["n_q_heads"], m["n_kv_heads"])

    # Prior tokens on this rank after KVP sharding, rounded up to block_size
    prior_tokens = math.ceil((max_len - seg_len) / kvp)
    prior_tokens = math.ceil(prior_tokens / block_size) * block_size

    return {
        "bs": 1,
        "num_q_heads": num_q_heads,
        "num_kv_heads": num_kv_heads,
        "block_size": block_size,
        "prior_seg_size": seg_len,
        "head_dim": m["d_head"],
        "prior_tokens": prior_tokens,
        "tp_q": True,
        "tp_out": True,
        "dtype": nl.bfloat16,
    }


def generate_segmented_attention_configs(configs=None):
    """Generate segmented attention configs for all models × sharding strategies."""
    if configs is None:
        configs = OPTIMAL_CONFIGS
    result = []
    for model_name in configs:
        c = configs[model_name]
        for tp, kvp in c["TP_KVP_CONFIGS"]:
            for block_size in c["BLK_SIZE"]:
                for max_len in c["MAX_LENS"]:
                    d = get_segmented_attention_config(model_name, tp, kvp, block_size, max_len)
                    if d is not None:
                        result.append((
                            d["bs"], d["num_q_heads"], d["num_kv_heads"], d["block_size"], d["prior_seg_size"],
                            d["head_dim"], d["prior_tokens"], d["tp_q"], d["tp_out"], d["dtype"],
                        ))
    return result


_all_optimal_configs = list(dict.fromkeys(generate_segmented_attention_configs(OPTIMAL_CONFIGS)))

segmented_attention_cte_model_configs: dict[ModelTestType, list] = {
    ModelTestType.OPTIMAL: _all_optimal_configs,
    ModelTestType.GENERALITY: [],
}
