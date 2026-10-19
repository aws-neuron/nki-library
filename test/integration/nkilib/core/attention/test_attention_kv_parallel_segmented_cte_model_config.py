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
KV-parallel segmented prefill attention model configuration data.

Generates configs for all models × sharding × seqlens × APC modes.
Exercises the attention_kv_parallel_segmented_cte kernel with production
model shapes, testing DCP prefill with chunked context (APC).

Config format: (model_name, group_size, q_heads_per_rank, seqlen, head_dim,
                block_size, seg_size, prior_tokens, num_global_blocks,
                lnc_degree, tp_out, apc_mode)
"""

import math

from typing import TypedDict

from test.utils.common_dataclasses import ModelTestType


class KvpSegmentedAttnModelConfig(TypedDict, total=False):
    TP_DCP_CONFIGS: list[tuple[int, int]]
    SEGMENT_SIZE_PER_RANK: int
    MAX_SEQS: list[int]
    BLK_SIZE: list[int]

MODELS = {
    "llama3_1b": {"n_q_heads": 32, "n_kv_heads": 8, "d_head": 64},
    "qwen3_235b_moe": {"n_q_heads": 64, "n_kv_heads": 4, "d_head": 128},
}

# (tp, dcp) → group_size = dcp, q_heads_per_rank = n_q_heads / tp
# Constraint: tp >= dcp * n_kv_heads
#
# Config format:
#   TP_DCP_CONFIGS: list of (tp, dcp) tuples
#   MAX_SEQS: list of maximum sequence lengths to test
#   BLK_SIZE: list of KV cache block sizes
#   SEGMENT_SIZE_PER_RANK: local KV tokens per rank per chunk
#   seqlen = SEGMENT_SIZE_PER_RANK * dcp
#   num_chunks = max_seq // seqlen
OPTIMAL_CONFIGS: dict[str, KvpSegmentedAttnModelConfig] = {
    "llama3_1b": {
        "TP_DCP_CONFIGS": [
            # (tp, dcp)
            (32, 4),
            (16, 2),
        ],
        "SEGMENT_SIZE_PER_RANK": 4096,
        "MAX_SEQS": [4096, 8192, 16384],
        "BLK_SIZE": [16],
    },
    "qwen3_235b_moe": {
        "TP_DCP_CONFIGS": [
            # (tp, dcp)
            (32, 2),
            (64, 4),
        ],
        "SEGMENT_SIZE_PER_RANK": 4096,
        "MAX_SEQS": [4096, 10240, 131072, 262144],
        "BLK_SIZE": [16],
    },
}

_LNC_SIZE = 2
_BLOCKS_ALIGNMENT = _LNC_SIZE * 128


def _get_seg_size(local_kv_tokens):
    """Find the largest supported segment size that divides local_kv_tokens."""
    supported = [512, 1024, 2048, 4096]
    for size in sorted(supported, reverse=True):
        if local_kv_tokens >= size and local_kv_tokens % size == 0:
            return size
    return 0


def get_kvp_segmented_attention_config(model_name, tp, dcp, seqlen, block_size, num_chunks, apc_mode=None):
    """Return KVP segmented attention config for a given model × sharding.

    Args:
        model_name: Key into MODELS dict.
        tp: Tensor parallel degree.
        dcp: DCP degree (= group_size for the kernel).
        seqlen: Sequence length per chunk.
        block_size: KV cache block size.
        num_chunks: Number of chunks (1 = no prior, 2+ = APC with prior).
        apc_mode: Explicit APC mode override. If None, derived from num_chunks.

    Returns:
        Dict with kernel parameters or None if config is invalid.
    """
    m = MODELS.get(model_name)
    if m is None:
        return None

    n_q_heads = m["n_q_heads"]
    n_kv_heads = m["n_kv_heads"]
    head_dim = m["d_head"]
    group_size = dcp

    # Validate DCP constraint: tp >= dcp * n_kv_heads
    if tp < dcp * n_kv_heads:
        return None

    q_heads_per_rank = n_q_heads // tp
    if q_heads_per_rank == 0:
        return None

    # Validate GQA ratio divisibility
    gqa_ratio = n_q_heads // n_kv_heads
    if gqa_ratio % dcp != 0:
        return None

    # Validate total_heads divisible by LNC
    total_heads = q_heads_per_rank * group_size
    if total_heads % _LNC_SIZE != 0:
        return None

    # Local KV tokens per chunk
    local_kv_tokens = seqlen // dcp
    seg_size = _get_seg_size(local_kv_tokens)
    if seg_size <= 0:
        return None

    # Prior tokens (from previous chunks)
    prior_chunks = num_chunks - 1
    prior_tokens = prior_chunks * local_kv_tokens
    # Round up to block_size
    prior_tokens = math.ceil(prior_tokens / block_size) * block_size if prior_tokens > 0 else 0

    # Total blocks in cache: must cover (prior_tokens + seqlen) global positions
    # so that all Q positions across all chunks have enough local KV.
    total_global_tokens = prior_tokens + seqlen
    min_blocks_per_segment = (seg_size // block_size) * group_size
    num_global_blocks = max(_BLOCKS_ALIGNMENT, min_blocks_per_segment, math.ceil(total_global_tokens / block_size))

    if apc_mode is None:
        apc_mode = num_chunks > 1

    return {
        "model_name": model_name,
        "group_size": group_size,
        "q_heads_per_rank": q_heads_per_rank,
        "seqlen": seqlen,
        "head_dim": head_dim,
        "block_size": block_size,
        "seg_size": seg_size,
        "prior_tokens": prior_tokens,
        "num_global_blocks": num_global_blocks,
        "lnc_degree": _LNC_SIZE,
        "tp_out": True,
        "apc_mode": apc_mode,
    }


def generate_kvp_segmented_attention_configs(configs=None):
    """Generate KVP segmented attention configs for all models × sharding.

    seqlen = SEGMENT_SIZE_PER_RANK * dcp (segmented prefill chunk length).
    num_chunks = max_seq // seqlen.
    For num_chunks == 1, both apc_mode=False and apc_mode=True are tested.
    """
    if configs is None:
        configs = OPTIMAL_CONFIGS
    result = []
    for model_name in configs:
        c = configs[model_name]
        seg_size_per_rank = c["SEGMENT_SIZE_PER_RANK"]
        for tp, dcp in c["TP_DCP_CONFIGS"]:
            for block_size in c["BLK_SIZE"]:
                seqlen = seg_size_per_rank * dcp
                for max_seq in c["MAX_SEQS"]:
                    num_chunks = max_seq // seqlen
                    if num_chunks < 1:
                        continue
                    apc_modes = [False, True] if num_chunks == 1 else [None]
                    for apc in apc_modes:
                        d = get_kvp_segmented_attention_config(
                            model_name, tp, dcp, seqlen, block_size, num_chunks, apc_mode=apc
                        )
                        if d is not None:
                            result.append((
                                d["model_name"],
                                d["group_size"],
                                d["q_heads_per_rank"],
                                d["seqlen"],
                                d["head_dim"],
                                d["block_size"],
                                d["seg_size"],
                                d["prior_tokens"],
                                d["num_global_blocks"],
                                d["lnc_degree"],
                                d["tp_out"],
                                d["apc_mode"],
                            ))
    return result


_all_optimal_configs = list(dict.fromkeys(generate_kvp_segmented_attention_configs()))

kvp_segmented_attention_cte_model_configs: dict[ModelTestType, list] = {
    ModelTestType.OPTIMAL: _all_optimal_configs,
    ModelTestType.GENERALITY: [],
}
