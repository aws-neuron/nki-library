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
"""Tests for KV-parallel segmented prefill attention kernel."""

import neuron_dtypes as dt
import nki.language as nl
import numpy as np
import pytest
from nki.collectives import ReplicaGroup
from nkilib_src.nkilib.core.attention.attention_kv_parallel_segmented_cte import (
    attention_kv_parallel_segmented_cte,
)
from nkilib_src.nkilib.experimental.collectives.distributed_adapter import get_rank

from test.utils.common_dataclasses import (
    CompilerArgs,
    ModelTestType,
    Platforms,
    prepare_model_parametrize,
)
from test.utils.pytest_test_metadata import pytest_test_metadata
from test.utils.test_orchestrator import Orchestrator
from test.utils.unit_test_collective_framework import CollectiveUnitTestFramework

try:
    from test.integration.nkilib.core.attention.test_attention_kv_parallel_segmented_cte_model_config import (
        kvp_segmented_attention_cte_model_configs,
    )
except ImportError:
    kvp_segmented_attention_cte_model_configs = {}

_CONTIGUOUS_PARAM_NAMES = "group_size,q_heads_per_rank,seqlen,head_dim,block_size,seg_size,prior_tokens,local_kv_multiplier,num_groups,lnc_degree,tp_out"

_CONTIGUOUS_FAST_PARAMS = [
    # Contiguous KV distribution (backward compatibility, not the production path)
    # LNC2 (default): group_size=4 physical ranks, q_heads_per_rank=2
    pytest.param(4, 2, 512, 128, 128, 512, 0, 1, 1, 2, False, id="g4_qh2_s512_h128_b128_seg512_lnc2"),
    pytest.param(4, 2, 512, 128, 128, 512, 512, 2, 1, 2, False, id="g4_qh2_s512_h128_b128_seg512_prior512_full_lnc2"),
    pytest.param(
        4, 2, 512, 128, 128, 512, 256, 1, 1, 2, False, id="g4_qh2_s512_h128_b128_seg512_prior256_partial_lnc2"
    ),
    pytest.param(4, 2, 1024, 128, 128, 512, 0, 1, 1, 2, False, id="g4_qh2_s1024_h128_b128_seg512_2chunks_lnc2"),
    pytest.param(4, 2, 512, 128, 64, 512, 0, 1, 1, 2, False, id="g4_qh2_s512_h128_b64_seg512_lnc2"),
    pytest.param(4, 2, 512, 128, 32, 512, 0, 1, 1, 2, False, id="g4_qh2_s512_h128_b32_seg512_lnc2"),
    pytest.param(4, 2, 512, 64, 128, 512, 0, 1, 1, 2, False, id="g4_qh2_s512_h64_b128_seg512_lnc2"),
    pytest.param(4, 2, 512, 128, 128, 512, 0, 2, 1, 2, True, id="g4_qh2_s512_h128_b128_seg512_2xkv_tp_out_lnc2"),
    # LNC1 (baseline only): group_size=8 physical ranks, q_heads_per_rank=1
    pytest.param(8, 1, 512, 128, 128, 512, 0, 1, 1, 1, False, id="g8_qh1_s512_h128_b128_seg512"),
    # q_heads_per_rank < lnc_degree: group_size=4, q_heads_per_rank=1, lnc_degree=2 (total_heads=4)
    pytest.param(4, 1, 512, 128, 128, 512, 0, 1, 1, 2, False, id="g4_qh1_s512_h128_b128_seg512_lnc2"),
]
_CONTIGUOUS_FAST_PARAMS = [pytest.param(*p.values, marks=pytest.mark.fast, id=p.id) for p in _CONTIGUOUS_FAST_PARAMS]

_CONTIGUOUS_FULL_ONLY_PARAMS = [
    # Heavy compile — full suite only
    pytest.param(4, 2, 2048, 128, 128, 2048, 2048, 2, 1, 2, False, id="g4_qh2_s2048_h128_b128_seg2048_prior2048_lnc2"),
]

_INTERLEAVED_PARAM_NAMES = "group_size,q_heads_per_rank,seqlen,head_dim,block_size,seg_size,prior_tokens,num_global_blocks,num_groups,lnc_degree,tp_out,sliding_window"

_INTERLEAVED_FAST_PARAMS = [
    # === g4_qh2_lnc2: 4 logical ranks, 2 Q heads/rank, LNC2 (baseline) ===
    # --- block_size=128 ---
    pytest.param(4, 2, 512, 128, 128, 512, 0, 32, 1, 2, False, 0, id="ilv_g4_qh2_lnc2_s512_h128_b128_seg512"),
    pytest.param(
        4, 2, 512, 128, 128, 512, 512, 32, 1, 2, False, 0, id="ilv_g4_qh2_lnc2_s512_h128_b128_seg512_prior512"
    ),
    pytest.param(
        4, 2, 512, 128, 128, 512, 1024, 48, 1, 2, False, 0, id="ilv_g4_qh2_lnc2_s512_h128_b128_seg512_prior1024"
    ),
    pytest.param(4, 2, 1024, 128, 128, 512, 0, 32, 1, 2, False, 0, id="ilv_g4_qh2_lnc2_s1024_h128_b128_seg512_2chunks"),
    # --- block_size=64 ---
    pytest.param(4, 2, 512, 128, 64, 512, 0, 64, 1, 2, False, 0, id="ilv_g4_qh2_lnc2_s512_h128_b64_seg512"),
    pytest.param(4, 2, 512, 128, 64, 512, 512, 64, 1, 2, False, 0, id="ilv_g4_qh2_lnc2_s512_h128_b64_seg512_prior512"),
    # --- block_size=32 ---
    pytest.param(4, 2, 512, 128, 32, 512, 0, 128, 1, 2, False, 0, id="ilv_g4_qh2_lnc2_s512_h128_b32_seg512"),
    pytest.param(4, 2, 512, 128, 32, 512, 512, 128, 1, 2, False, 0, id="ilv_g4_qh2_lnc2_s512_h128_b32_seg512_prior512"),
    # --- head_dim=64 ---
    pytest.param(4, 2, 512, 64, 128, 512, 0, 32, 1, 2, False, 0, id="ilv_g4_qh2_lnc2_s512_h64_b128_seg512"),
    pytest.param(4, 2, 512, 64, 128, 512, 512, 32, 1, 2, False, 0, id="ilv_g4_qh2_lnc2_s512_h64_b128_seg512_prior512"),
    pytest.param(4, 2, 512, 64, 64, 512, 0, 64, 1, 2, False, 0, id="ilv_g4_qh2_lnc2_s512_h64_b64_seg512"),
    pytest.param(4, 2, 512, 64, 64, 512, 512, 64, 1, 2, False, 0, id="ilv_g4_qh2_lnc2_s512_h64_b64_seg512_prior512"),
    # --- long prior (multiple full prior segments) ---
    pytest.param(
        4, 2, 512, 128, 128, 512, 2048, 80, 1, 2, False, 0, id="ilv_g4_qh2_lnc2_s512_h128_b128_seg512_prior2048"
    ),
    pytest.param(
        4, 2, 512, 128, 64, 512, 2048, 160, 1, 2, False, 0, id="ilv_g4_qh2_lnc2_s512_h128_b64_seg512_prior2048"
    ),
    # === g8_qh1_lnc1: 8 logical ranks, 1 Q head/rank, LNC1 ===
    pytest.param(8, 1, 512, 128, 128, 512, 0, 32, 1, 1, False, 0, id="ilv_g8_qh1_lnc1_s512_h128_b128_seg512"),
    pytest.param(
        8, 1, 512, 128, 128, 512, 512, 32, 1, 1, False, 0, id="ilv_g8_qh1_lnc1_s512_h128_b128_seg512_prior512"
    ),
    pytest.param(8, 1, 512, 128, 64, 512, 0, 64, 1, 1, False, 0, id="ilv_g8_qh1_lnc1_s512_h128_b64_seg512"),
    # --- partial prior ---
    pytest.param(
        4, 2, 512, 128, 64, 512, 256, 64, 1, 2, False, 0, id="ilv_g4_qh2_lnc2_s512_h128_b64_seg512_prior256_partial"
    ),
    pytest.param(
        4, 2, 512, 128, 128, 512, 256, 32, 1, 2, False, 0, id="ilv_g4_qh2_lnc2_s512_h128_b128_seg512_prior256_partial"
    ),
    # --- multi-chunk + prior ---
    pytest.param(
        4, 2, 1024, 128, 128, 512, 512, 48, 1, 2, False, 0, id="ilv_g4_qh2_lnc2_s1024_h128_b128_seg512_2chunks_prior"
    ),
    # --- block_size=16 ---
    pytest.param(4, 2, 512, 128, 16, 512, 0, 256, 1, 2, False, 0, id="ilv_g4_qh2_lnc2_s512_h128_b16_seg512"),
    pytest.param(4, 2, 512, 128, 16, 512, 512, 256, 1, 2, False, 0, id="ilv_g4_qh2_lnc2_s512_h128_b16_seg512_prior512"),
    # --- tp_out=True ---
    pytest.param(4, 2, 512, 128, 128, 512, 0, 32, 1, 2, True, 0, id="ilv_g4_qh2_lnc2_s512_h128_b128_seg512_tp_out"),
    pytest.param(
        4, 2, 512, 128, 128, 512, 512, 32, 1, 2, True, 0, id="ilv_g4_qh2_lnc2_s512_h128_b128_seg512_prior512_tp_out"
    ),
    # --- multi-chunk + b64 ---
    pytest.param(4, 2, 1024, 128, 64, 512, 0, 64, 1, 2, False, 0, id="ilv_g4_qh2_lnc2_s1024_h128_b64_seg512_2chunks"),
    # --- h64 + b32 ---
    pytest.param(4, 2, 512, 64, 32, 512, 0, 128, 1, 2, False, 0, id="ilv_g4_qh2_lnc2_s512_h64_b32_seg512"),
    pytest.param(4, 2, 512, 64, 32, 512, 512, 128, 1, 2, False, 0, id="ilv_g4_qh2_lnc2_s512_h64_b32_seg512_prior512"),
    # --- h64 + multi-chunk ---
    pytest.param(4, 2, 1024, 64, 128, 512, 0, 32, 1, 2, False, 0, id="ilv_g4_qh2_lnc2_s1024_h64_b128_seg512_2chunks"),
    # === SWA (sliding window attention) + interleaved ===
    pytest.param(
        4, 2, 512, 128, 128, 512, 512, 32, 1, 2, False, 256, id="ilv_g4_qh2_lnc2_s512_h128_b128_seg512_prior512_sw256"
    ),
    pytest.param(
        4, 2, 512, 128, 128, 512, 1024, 48, 1, 2, False, 512, id="ilv_g4_qh2_lnc2_s512_h128_b128_seg512_prior1024_sw512"
    ),
    pytest.param(
        4, 2, 512, 128, 64, 512, 512, 64, 1, 2, False, 256, id="ilv_g4_qh2_lnc2_s512_h128_b64_seg512_prior512_sw256"
    ),
    pytest.param(
        4, 2, 512, 64, 128, 512, 512, 32, 1, 2, False, 256, id="ilv_g4_qh2_lnc2_s512_h64_b128_seg512_prior512_sw256"
    ),
    # === q_heads_per_rank > lnc_degree: multiple heads per NC ===
    # g4_qh4_lnc2: 4 logical ranks, 4 Q heads/rank, LNC2
    pytest.param(4, 4, 512, 128, 128, 512, 0, 32, 1, 2, False, 0, id="ilv_g4_qh4_lnc2_s512_h128_b128_seg512"),
    pytest.param(
        4, 4, 512, 128, 128, 512, 512, 32, 1, 2, False, 0, id="ilv_g4_qh4_lnc2_s512_h128_b128_seg512_prior512"
    ),
    pytest.param(4, 4, 1024, 128, 128, 512, 0, 32, 1, 2, False, 0, id="ilv_g4_qh4_lnc2_s1024_h128_b128_seg512_2chunks"),
    pytest.param(4, 4, 512, 128, 64, 512, 0, 64, 1, 2, False, 0, id="ilv_g4_qh4_lnc2_s512_h128_b64_seg512"),
    pytest.param(4, 4, 512, 64, 128, 512, 0, 32, 1, 2, False, 0, id="ilv_g4_qh4_lnc2_s512_h64_b128_seg512"),
    pytest.param(4, 4, 512, 128, 128, 512, 0, 32, 1, 2, True, 0, id="ilv_g4_qh4_lnc2_s512_h128_b128_seg512_tp_out"),
    # g2_qh4_lnc2: 2 logical ranks, 4 Q heads/rank, LNC2
    pytest.param(2, 4, 512, 128, 128, 512, 0, 16, 1, 2, False, 0, id="ilv_g2_qh4_lnc2_s512_h128_b128_seg512"),
    pytest.param(
        2, 4, 512, 128, 128, 512, 512, 16, 1, 2, False, 0, id="ilv_g2_qh4_lnc2_s512_h128_b128_seg512_prior512"
    ),
    # === q_heads_per_rank < lnc_degree: fewer heads than NCs ===
    # g4_qh1_lnc2: 4 logical ranks, 1 Q head/rank, LNC2 (total_heads=4, divisible by lnc=2)
    pytest.param(4, 1, 512, 128, 128, 512, 0, 32, 1, 2, False, 0, id="ilv_g4_qh1_lnc2_s512_h128_b128_seg512"),
    pytest.param(
        4, 1, 512, 128, 128, 512, 512, 32, 1, 2, False, 0, id="ilv_g4_qh1_lnc2_s512_h128_b128_seg512_prior512"
    ),
]
_INTERLEAVED_FAST_PARAMS = [pytest.param(*p.values, marks=pytest.mark.fast, id=p.id) for p in _INTERLEAVED_FAST_PARAMS]

_INTERLEAVED_FULL_ONLY_PARAMS = [
    # Heavy compile — full suite only
    pytest.param(4, 2, 2048, 128, 128, 2048, 0, 64, 1, 2, False, 0, id="ilv_g4_qh2_lnc2_s2048_h128_b128_seg2048"),
    pytest.param(
        4, 2, 2048, 128, 128, 2048, 2048, 128, 1, 2, False, 0, id="ilv_g4_qh2_lnc2_s2048_h128_b128_seg2048_prior2048"
    ),
    pytest.param(4, 2, 2048, 128, 64, 2048, 0, 128, 1, 2, False, 0, id="ilv_g4_qh2_lnc2_s2048_h128_b64_seg2048"),
    pytest.param(
        4, 2, 2048, 128, 64, 2048, 2048, 192, 1, 2, False, 0, id="ilv_g4_qh2_lnc2_s2048_h128_b64_seg2048_prior2048"
    ),
    pytest.param(4, 2, 2048, 128, 32, 2048, 0, 256, 1, 2, False, 0, id="ilv_g4_qh2_lnc2_s2048_h128_b32_seg2048"),
    pytest.param(4, 2, 2048, 64, 128, 2048, 0, 64, 1, 2, False, 0, id="ilv_g4_qh2_lnc2_s2048_h64_b128_seg2048"),
    pytest.param(
        4, 2, 2048, 64, 128, 2048, 2048, 128, 1, 2, False, 0, id="ilv_g4_qh2_lnc2_s2048_h64_b128_seg2048_prior2048"
    ),
    pytest.param(8, 1, 2048, 128, 128, 2048, 0, 128, 1, 1, False, 0, id="ilv_g8_qh1_lnc1_s2048_h128_b128_seg2048"),
    pytest.param(
        4, 2, 2048, 128, 32, 2048, 2048, 512, 1, 2, False, 0, id="ilv_g4_qh2_lnc2_s2048_h128_b32_seg2048_prior2048"
    ),
    pytest.param(
        4,
        2,
        2048,
        128,
        128,
        2048,
        2048,
        128,
        1,
        2,
        False,
        512,
        id="ilv_g4_qh2_lnc2_s2048_h128_b128_seg2048_prior2048_sw512",
    ),
    pytest.param(4, 2, 4096, 128, 128, 4096, 0, 128, 1, 2, False, 0, id="ilv_g4_qh2_lnc2_s4096_h128_b128_seg4096"),
    pytest.param(
        4, 2, 4096, 128, 128, 4096, 4096, 256, 1, 2, False, 0, id="ilv_g4_qh2_lnc2_s4096_h128_b128_seg4096_prior4096"
    ),
    # q_heads_per_rank > lnc_degree (heavy compile)
    pytest.param(4, 4, 2048, 128, 128, 2048, 0, 64, 1, 2, False, 0, id="ilv_g4_qh4_lnc2_s2048_h128_b128_seg2048"),
    pytest.param(
        4, 4, 2048, 128, 128, 2048, 2048, 128, 1, 2, False, 0, id="ilv_g4_qh4_lnc2_s2048_h128_b128_seg2048_prior2048"
    ),
]


@pytest_test_metadata(name="KV Parallel Segmented Prefill", pytest_marks=["attention", "kv_parallel"], tag=["model"])
@pytest.mark.platforms(exclude=list(set(Platforms) - {Platforms.TRN2}))
@pytest.mark.skip_simulation
class TestKVParallelSegmentedPrefill:
    """Test class for KV-parallel segmented prefill attention."""

    @pytest.mark.parametrize(_CONTIGUOUS_PARAM_NAMES, _CONTIGUOUS_FAST_PARAMS + _CONTIGUOUS_FULL_ONLY_PARAMS)
    def test_kv_parallel_segmented_prefill(
        self,
        test_manager: Orchestrator,
        group_size: int,
        q_heads_per_rank: int,
        seqlen: int,
        head_dim: int,
        block_size: int,
        seg_size: int,
        prior_tokens: int,
        local_kv_multiplier: int,
        num_groups: int,
        lnc_degree: int,
        tp_out: bool,
    ):
        """
        End-to-end test for KV-parallel segmented prefill attention.

        Tests the full algorithm:
        1. All-gather Q across ranks
        2. Each rank computes attention on its KV shard with shifted causal mask
        3. All-to-all exchange of partial outputs + softmax stats
        4. Merge partials using online softmax
        5. Return final result

        Args:
            group_size: Number of physical ranks per replica group
            q_heads_per_rank: Number of Q heads per physical rank
            seqlen: Sequence length (Q length)
            head_dim: Head dimension
            block_size: KV cache block size
            seg_size: Segment size for attention iteration
            prior_tokens: Prior tokens for continuation (shifts Q global position)
            local_kv_multiplier: Multiplier for local_kv_len = seg_size * multiplier
            num_groups: Number of independent replica groups (total_ranks = group_size * num_groups)
        """
        np.random.seed(42)

        num_kv_heads = 1
        local_kv_len = seg_size * local_kv_multiplier
        num_blocks = local_kv_len // block_size
        # collective_ranks = number of physical ranks participating in collectives
        collective_ranks = group_size * num_groups
        # Each physical rank has q_heads_per_rank Q heads
        total_q_heads = q_heads_per_rank * group_size * num_groups

        # Generate Q for all groups (each group has group_size * q_heads_per_rank Q heads)
        q_global = np.random.randn(total_q_heads, 1, seqlen, head_dim).astype(nl.bfloat16)

        # Generate KV shards for all physical ranks across all groups
        # Layout: (num_blocks, num_kv_heads, block_size, head_dim)
        k_cache_global = np.random.randn(collective_ranks, num_blocks, num_kv_heads, block_size, head_dim).astype(
            nl.bfloat16
        )
        v_cache_global = np.random.randn(collective_ranks, num_blocks, num_kv_heads, block_size, head_dim).astype(
            nl.bfloat16
        )

        # Block tables: sequential blocks for each rank
        block_tables = np.arange(num_blocks, dtype=np.int32).reshape(1, num_blocks)
        block_tables = dt.static_cast(block_tables, nl.int32)

        # Create replica groups at physical rank level
        replica_group_lists = [
            list(range(group_idx * group_size, (group_idx + 1) * group_size)) for group_idx in range(num_groups)
        ]
        replica_groups = ReplicaGroup(replica_group_lists)

        def create_inputs(rank_id: int):
            # rank_id is physical rank index (0 to collective_ranks-1)

            # Determine which group this rank belongs to and its position within the group
            group_id = rank_id // group_size
            rank_in_group = rank_id % group_size

            # cp_offset based on physical rank's KV position within its group
            k_offset = rank_in_group * local_kv_len
            cp_offset_value = -k_offset + prior_tokens
            cp_offset = dt.static_cast(np.array([[cp_offset_value]], dtype=np.int32), nl.int32)

            # q_local contains q_heads_per_rank Q heads for this physical rank
            # Shape: [q_heads_per_rank, seqlen, head_dim]
            q_start = (group_id * group_size + rank_in_group) * q_heads_per_rank
            q_local = q_global[q_start : q_start + q_heads_per_rank, 0, :, :]

            return {
                "q": q_local,
                "k_cache": k_cache_global[rank_id],
                "v_cache": v_cache_global[rank_id],
                "block_tables": block_tables,
                "kvp_q_offset": cp_offset,
                "replica_groups": replica_groups,
                "group_size": group_size,
                "block_size": block_size,
                "seg_size": seg_size,
                "scale": 1.0,
                "global_q_offset": prior_tokens,
                "tp_out": tp_out,
            }

        def create_golden(rank_id: int):
            # rank_id is physical rank index (0 to collective_ranks-1)
            # Each physical rank outputs q_heads_per_rank Q heads
            group_id = rank_id // group_size
            rank_in_group = rank_id % group_size

            # Concatenate KV shards from all physical ranks in the same group
            k_full = []
            v_full = []
            for pr in range(group_size):
                global_pr = group_id * group_size + pr
                # Layout: (num_blocks, num_kv_heads, block_size, head_dim) → flatten to (total_kv_len, head_dim)
                k_seq = k_cache_global[global_pr, :, 0, :, :].reshape(-1, head_dim).astype(np.float32)
                v_seq = v_cache_global[global_pr, :, 0, :, :].reshape(-1, head_dim).astype(np.float32)
                k_full.append(k_seq)
                v_full.append(v_seq)
            k_full = np.concatenate(k_full, axis=0)  # [total_kv_len, head_dim]
            v_full = np.concatenate(v_full, axis=0)  # [total_kv_len, head_dim]

            total_kv_len = k_full.shape[0]

            # Compute attention for each of this rank's Q heads
            outputs = []
            q_start_global = (group_id * group_size + rank_in_group) * q_heads_per_rank

            for head_idx in range(q_heads_per_rank):
                q_head_idx = q_start_global + head_idx
                q = q_global[q_head_idx, 0].astype(np.float32)  # [seqlen, head_dim]

                # Compute attention scores
                scores = np.matmul(q, k_full.T)  # [seqlen, total_kv_len]

                # Apply causal mask
                q_pos = np.arange(prior_tokens, prior_tokens + seqlen).reshape(-1, 1)
                k_pos = np.arange(total_kv_len).reshape(1, -1)
                causal_mask = q_pos < k_pos
                scores = np.where(causal_mask, -np.inf, scores)

                # Softmax
                max_scores = np.max(scores, axis=-1, keepdims=True)
                max_scores = np.where(np.isinf(max_scores), 0, max_scores)
                exp_scores = np.exp(scores - max_scores)
                sum_exp = np.sum(exp_scores, axis=-1, keepdims=True)
                sum_exp = np.where(sum_exp == 0, 1, sum_exp)
                attn_weights = exp_scores / sum_exp

                # Output
                out = np.matmul(attn_weights, v_full)  # [seqlen, head_dim]
                outputs.append(out)

            # Stack outputs: [q_heads_per_rank, seqlen, head_dim] or [q_heads_per_rank, head_dim, seqlen] if tp_out
            out_stacked = np.stack(outputs, axis=0).astype(nl.bfloat16)
            if tp_out:
                out_stacked = np.transpose(out_stacked, (0, 2, 1))  # [q_heads_per_rank, head_dim, seqlen]

            return {
                "out": out_stacked,
            }

        def _torch_ref(
            q,
            k_cache,
            v_cache,
            block_tables,
            kvp_q_offset,
            replica_groups,
            group_size,
            block_size,
            seg_size,
            scale=1.0,
            global_q_offset=0,
            tp_out=False,
            sliding_window=0,
            kvp_rank_id=None,
            kvp_group_size=0,
            apc_mode=False,
            valid_num_prior_tokens=None,
            fp8_packed=False,
            k_scale=None,
            v_scale=None,
        ):
            return create_golden(get_rank())

        framework = CollectiveUnitTestFramework(
            test_manager=test_manager,
            kernel_entry=attention_kv_parallel_segmented_cte,
            torch_ref=_torch_ref,
            per_rank_input_generator=create_inputs,
            collective_ranks=collective_ranks,
        )
        framework.run_test(
            test_config=None,
            compiler_args=CompilerArgs(platform_target=Platforms.TRN2, logical_nc_config=lnc_degree),
            output_keys=["out"],
            rtol=5e-2,
            atol=1e-2,
        )

    @pytest.mark.parametrize(_INTERLEAVED_PARAM_NAMES, _INTERLEAVED_FAST_PARAMS + _INTERLEAVED_FULL_ONLY_PARAMS)
    def test_kv_parallel_segmented_prefill_interleaved(
        self,
        test_manager: Orchestrator,
        group_size: int,
        q_heads_per_rank: int,
        seqlen: int,
        head_dim: int,
        block_size: int,
        seg_size: int,
        prior_tokens: int,
        num_global_blocks: int,
        num_groups: int,
        lnc_degree: int,
        tp_out: bool,
        sliding_window: int,
    ):
        """
        Test KV-parallel segmented prefill with interleaved (round-robin) block distribution.

        Each rank holds non-contiguous blocks distributed round-robin across the global KV timeline.
        The per-block masking in _attention_cte handles the non-contiguous causal mask.
        """
        np.random.seed(42)

        num_kv_heads = 1
        collective_ranks = group_size * num_groups

        # Global KV: num_global_blocks blocks, each block_size tokens
        global_kv_len = num_global_blocks * block_size
        k_global_flat = np.random.randn(num_global_blocks, num_kv_heads, block_size, head_dim).astype(nl.bfloat16)
        v_global_flat = np.random.randn(num_global_blocks, num_kv_heads, block_size, head_dim).astype(nl.bfloat16)

        # Round-robin assignment: global block b goes to rank (b % group_size)
        # Each rank's local blocks, sorted by global position

        # Generate Q for all groups
        total_q_heads = q_heads_per_rank * group_size * num_groups
        q_global = np.random.randn(total_q_heads, 1, seqlen, head_dim).astype(nl.bfloat16)

        # Replica groups
        replica_group_lists = [
            list(range(group_idx * group_size, (group_idx + 1) * group_size)) for group_idx in range(num_groups)
        ]
        replica_groups = ReplicaGroup(replica_group_lists)

        def create_inputs(rank_id: int):
            group_id = rank_id // group_size
            rank_in_group = rank_id % group_size

            # Collect this rank's blocks (round-robin)
            local_global_block_ids = list(range(rank_in_group, num_global_blocks, group_size))
            num_local_blocks = len(local_global_block_ids)

            # Build local KV cache from global blocks
            k_local = np.zeros((num_local_blocks, num_kv_heads, block_size, head_dim), dtype=nl.bfloat16)
            v_local = np.zeros((num_local_blocks, num_kv_heads, block_size, head_dim), dtype=nl.bfloat16)

            for local_idx, global_blk_id in enumerate(local_global_block_ids):
                k_local[local_idx] = k_global_flat[global_blk_id]
                v_local[local_idx] = v_global_flat[global_blk_id]

            block_tables = np.arange(num_local_blocks, dtype=np.int32).reshape(1, num_local_blocks)
            block_tables = dt.static_cast(block_tables, nl.int32)

            # kvp_offset = global_q_offset for interleaved KV
            cp_offset_value = prior_tokens
            cp_offset = dt.static_cast(np.array([[cp_offset_value]], dtype=np.int32), nl.int32)

            q_start = (group_id * group_size + rank_in_group) * q_heads_per_rank
            q_local = q_global[q_start : q_start + q_heads_per_rank, 0, :, :]

            return {
                "q": q_local,
                "k_cache": k_local,
                "v_cache": v_local,
                "block_tables": block_tables,
                "kvp_q_offset": cp_offset,
                "replica_groups": replica_groups,
                "group_size": group_size,
                "block_size": block_size,
                "seg_size": seg_size,
                "scale": 1.0,
                "global_q_offset": prior_tokens,
                "tp_out": tp_out,
                "sliding_window": sliding_window,
                "kvp_rank_id": dt.static_cast(np.array([[rank_in_group]], dtype=np.int32), nl.int32),
                "kvp_group_size": group_size,
            }

        def create_golden(rank_id: int):
            group_id = rank_id // group_size
            rank_in_group = rank_id % group_size

            # Concatenate ALL ranks' KV in global order for the golden reference
            k_full = k_global_flat[:, 0, :, :].reshape(-1, head_dim).astype(np.float32)
            v_full = v_global_flat[:, 0, :, :].reshape(-1, head_dim).astype(np.float32)

            outputs = []
            q_start_global = (group_id * group_size + rank_in_group) * q_heads_per_rank

            for head_idx in range(q_heads_per_rank):
                q_head_idx = q_start_global + head_idx
                q = q_global[q_head_idx, 0].astype(np.float32)

                scores = np.matmul(q, k_full.T)

                # Causal mask: q_pos >= k_pos means visible
                q_pos = np.arange(prior_tokens, prior_tokens + seqlen).reshape(-1, 1)
                k_pos = np.arange(global_kv_len).reshape(1, -1)
                causal_mask = q_pos < k_pos
                scores = np.where(causal_mask, -np.inf, scores)

                # SWA mask: mask if k_pos < q_pos - (sliding_window - 1)
                if sliding_window > 0:
                    swa_mask = k_pos < q_pos - (sliding_window - 1)
                    scores = np.where(swa_mask, -np.inf, scores)

                max_scores = np.max(scores, axis=-1, keepdims=True)
                max_scores = np.where(np.isinf(max_scores), 0, max_scores)
                exp_scores = np.exp(scores - max_scores)
                sum_exp = np.sum(exp_scores, axis=-1, keepdims=True)
                sum_exp = np.where(sum_exp == 0, 1, sum_exp)
                attn_weights = exp_scores / sum_exp

                out = np.matmul(attn_weights, v_full)
                outputs.append(out)

            out_stacked = np.stack(outputs, axis=0).astype(nl.bfloat16)
            if tp_out:
                out_stacked = np.transpose(out_stacked, (0, 2, 1))
            return {"out": out_stacked}

        def _torch_ref(
            q,
            k_cache,
            v_cache,
            block_tables,
            kvp_q_offset,
            replica_groups,
            group_size,
            block_size,
            seg_size,
            scale=1.0,
            global_q_offset=0,
            tp_out=False,
            sliding_window=0,
            kvp_rank_id=None,
            kvp_group_size=0,
            apc_mode=False,
            valid_num_prior_tokens=None,
            fp8_packed=False,
            k_scale=None,
            v_scale=None,
        ):
            return create_golden(get_rank())

        framework = CollectiveUnitTestFramework(
            test_manager=test_manager,
            kernel_entry=attention_kv_parallel_segmented_cte,
            torch_ref=_torch_ref,
            per_rank_input_generator=create_inputs,
            collective_ranks=collective_ranks,
        )
        framework.run_test(
            test_config=None,
            compiler_args=CompilerArgs(platform_target=Platforms.TRN2, logical_nc_config=lnc_degree),
            output_keys=["out"],
            rtol=5e-2,
            atol=1e-2,
        )

    @pytest.mark.fast
    def test_kv_parallel_segmented_prefill_interleaved_fp8_packed(
        self,
        test_manager: Orchestrator,
    ):
        """Packed FP8 KV uses dequant scales through the interleaved CP path."""
        np.random.seed(42)

        group_size = 2
        q_heads_per_rank = 2
        seqlen = 512
        head_dim = 128
        block_size = 128
        seg_size = 512
        prior_tokens = 512
        num_global_blocks = 16
        lnc_degree = 2
        num_kv_heads = 1
        k_dequant_scale = np.full(head_dim, 0.25, dtype=np.float32)
        v_dequant_scale = np.full(head_dim, 0.5, dtype=np.float32)
        k_scale_broadcast = k_dequant_scale.reshape(1, 1, 1, head_dim)
        v_scale_broadcast = v_dequant_scale.reshape(1, 1, 1, head_dim)

        softmax_scale = 1.0 / np.sqrt(head_dim)
        q_global = np.random.randn(
            group_size * q_heads_per_rank,
            1,
            seqlen,
            head_dim,
        ).astype(nl.bfloat16)
        q_global = dt.static_cast(
            q_global.astype(np.float32) * softmax_scale,
            nl.bfloat16,
        )
        k_real = np.random.uniform(
            -0.5,
            0.5,
            (num_global_blocks, num_kv_heads, block_size, head_dim),
        ).astype(np.float32)
        v_real = np.random.uniform(
            -0.5,
            0.5,
            (num_global_blocks, num_kv_heads, block_size, head_dim),
        ).astype(np.float32)
        k_quantized = dt.static_cast(k_real / k_scale_broadcast, nl.float8_e4m3)
        v_quantized = dt.static_cast(v_real / v_scale_broadcast, nl.float8_e4m3)
        k_dequantized = dt.static_cast(k_quantized, np.float32) * k_scale_broadcast
        v_dequantized = dt.static_cast(v_quantized, np.float32) * v_scale_broadcast

        replica_groups = ReplicaGroup([list(range(group_size))])

        def create_inputs(rank_id: int):
            local_global_block_ids = list(range(rank_id, num_global_blocks, group_size))
            k_local = k_quantized[local_global_block_ids]
            k_packed = np.stack(
                [k_local[:, :, 0::2, :], k_local[:, :, 1::2, :]],
                axis=-1,
            )
            v_local = v_quantized[local_global_block_ids]
            num_local_blocks = len(local_global_block_ids)

            stride = group_size * block_size
            rank_offset = rank_id * block_size
            threshold = prior_tokens - rank_offset - block_size + 1
            num_fully_visible_blocks = max(0, threshold // stride + 1) if threshold >= 0 else 0
            valid_prior = num_fully_visible_blocks * block_size

            return {
                "q": q_global[
                    rank_id * q_heads_per_rank : (rank_id + 1) * q_heads_per_rank,
                    0,
                    :,
                    :,
                ],
                "k_cache": dt.static_cast(k_packed, nl.float8_e4m3),
                "v_cache": dt.static_cast(v_local, nl.float8_e4m3),
                "block_tables": dt.static_cast(
                    np.arange(num_local_blocks, dtype=np.int32).reshape(1, num_local_blocks),
                    nl.int32,
                ),
                "kvp_q_offset": dt.static_cast(np.array([[prior_tokens]], dtype=np.int32), nl.int32),
                "replica_groups": replica_groups,
                "group_size": group_size,
                "block_size": block_size,
                "seg_size": seg_size,
                # KVP uses the prefix-caching mask path, which requires scale=1.
                # Production similarly folds the softmax scale into Q.
                "scale": 1.0,
                "global_q_offset": 0,
                "tp_out": False,
                "sliding_window": 0,
                "kvp_rank_id": dt.static_cast(np.array([[rank_id]], dtype=np.int32), nl.int32),
                "kvp_group_size": group_size,
                "apc_mode": True,
                "valid_num_prior_tokens": dt.static_cast(np.array([[valid_prior]], dtype=np.int32), nl.int32),
                "fp8_packed": True,
                "k_scale": k_dequant_scale.reshape(128, 1),
                "v_scale": v_dequant_scale.reshape(128, 1),
            }

        def create_golden(rank_id: int):
            k_full = k_dequantized[:, 0, :, :].reshape(-1, head_dim)
            v_full = v_dequantized[:, 0, :, :].reshape(-1, head_dim)
            k_pos = np.arange(num_global_blocks * block_size).reshape(1, -1)
            q_pos = np.arange(prior_tokens, prior_tokens + seqlen).reshape(-1, 1)

            outputs = []
            q_start = rank_id * q_heads_per_rank
            for head_idx in range(q_heads_per_rank):
                q = q_global[q_start + head_idx, 0].astype(np.float32)
                scores = np.matmul(q, k_full.T)
                scores = np.where(q_pos < k_pos, -np.inf, scores)
                max_scores = np.max(scores, axis=-1, keepdims=True)
                max_scores = np.where(np.isinf(max_scores), 0, max_scores)
                exp_scores = np.exp(scores - max_scores)
                attn_weights = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)
                outputs.append(np.matmul(attn_weights, v_full))

            return {
                "out": np.stack(outputs, axis=0).astype(nl.bfloat16),
            }

        def _torch_ref(
            q,
            k_cache,
            v_cache,
            block_tables,
            kvp_q_offset,
            replica_groups,
            group_size,
            block_size,
            seg_size,
            scale=1.0,
            global_q_offset=0,
            tp_out=False,
            sliding_window=0,
            kvp_rank_id=None,
            kvp_group_size=0,
            apc_mode=False,
            valid_num_prior_tokens=None,
            fp8_packed=False,
            k_scale=None,
            v_scale=None,
        ):
            return create_golden(get_rank())

        framework = CollectiveUnitTestFramework(
            test_manager=test_manager,
            kernel_entry=attention_kv_parallel_segmented_cte,
            torch_ref=_torch_ref,
            per_rank_input_generator=create_inputs,
            collective_ranks=group_size,
        )
        framework.run_test(
            test_config=None,
            compiler_args=CompilerArgs(
                platform_target=Platforms.TRN2,
                logical_nc_config=lnc_degree,
            ),
            output_keys=["out"],
            rtol=5e-2,
            atol=2e-2,
        )

    _APC_PARAM_NAMES = "group_size,q_heads_per_rank,seqlen,head_dim,block_size,seg_size,prior_tokens,num_global_blocks,lnc_degree,tp_out"

    _APC_FAST_PARAMS = [
        pytest.param(
            4,
            2,
            1024,
            128,
            128,
            512,
            1024,
            128,
            2,
            False,
            marks=pytest.mark.fast,
            id="apc_lnc2_s1024_h128_b128_prior1024",
        ),
        pytest.param(
            4,
            2,
            1024,
            128,
            64,
            512,
            512,
            128,
            2,
            False,
            marks=pytest.mark.fast,
            id="apc_lnc2_s1024_h128_b64_prior512",
        ),
        pytest.param(
            4,
            2,
            1024,
            128,
            128,
            512,
            2048,
            192,
            2,
            False,
            marks=pytest.mark.fast,
            id="apc_lnc2_s1024_h128_b128_prior2048",
        ),
        # No prefix hit — active-only path, 0 prior segments
        pytest.param(
            4,
            2,
            1024,
            128,
            128,
            512,
            0,
            128,
            2,
            False,
            marks=pytest.mark.fast,
            id="apc_lnc2_s1024_h128_b128_prior0",
        ),
        # cp_offset < stride — prior_tokens rounds to 0, no prior segments
        # cp_offset < stride: stride-clamp zeros prior_tokens so active starts at block 0
        pytest.param(
            4,
            2,
            512,
            128,
            128,
            512,
            128,
            128,
            2,
            False,
            marks=pytest.mark.fast,
            id="apc_lnc2_s512_h128_b128_prior128",
        ),
        # Partial prior segment: prior_tokens=768 → 1 full + 256 partial
        pytest.param(
            4,
            2,
            1024,
            128,
            128,
            512,
            768,
            128,
            2,
            False,
            marks=pytest.mark.fast,
            id="apc_lnc2_s1024_h128_b128_prior768",
        ),
    ]

    _APC_FULL_ONLY_PARAMS = [
        # Larger stride (group_size=4, block_size=64 → stride=256), exercises different geometry
        pytest.param(4, 2, 1024, 128, 64, 512, 1024, 128, 2, False, id="apc_lnc2_s1024_h128_b64_prior1024_full"),
        # tp_out=True
        pytest.param(4, 2, 1024, 128, 128, 512, 1024, 128, 2, True, id="apc_lnc2_s1024_h128_b128_prior1024_tp_out"),
        # Large prior with many degenerate iterations to skip
        pytest.param(4, 2, 1024, 128, 128, 512, 4096, 288, 2, False, id="apc_lnc2_s1024_h128_b128_prior4096"),
        # Multiple Q heads per rank (q_heads_per_rank > lnc_degree)
        pytest.param(4, 4, 1024, 128, 128, 512, 1024, 128, 2, False, id="apc_lnc2_qh4_s1024_h128_b128_prior1024"),
        pytest.param(4, 4, 1024, 128, 128, 512, 2048, 192, 2, False, id="apc_lnc2_qh4_s1024_h128_b128_prior2048"),
        # Large S (8 chunks): exercises per-chunk prior_tokens increment across many iterations
        pytest.param(
            4, 2, 4096, 128, 128, 512, 2048, 256, 2, False, id="apc_lnc2_s4096_h128_b128_prior2048_multichunk"
        ),
    ]

    @pytest.mark.parametrize(_APC_PARAM_NAMES, _APC_FAST_PARAMS + _APC_FULL_ONLY_PARAMS)
    def test_kv_parallel_segmented_prefill_interleaved_apc(
        self,
        test_manager: Orchestrator,
        group_size: int,
        q_heads_per_rank: int,
        seqlen: int,
        head_dim: int,
        block_size: int,
        seg_size: int,
        prior_tokens: int,
        num_global_blocks: int,
        lnc_degree: int,
        tp_out: bool,
    ):
        """
        Test APC mode: global_q_offset=0, prior count derived from runtime kvp_q_offset.

        Simulates automated prefix caching where the actual prefix hit length is only
        known at runtime, not compile time.
        """
        np.random.seed(42)

        num_kv_heads = 1
        num_groups = 1
        collective_ranks = group_size * num_groups

        global_kv_len = num_global_blocks * block_size
        k_global_flat = np.random.randn(num_global_blocks, num_kv_heads, block_size, head_dim).astype(nl.bfloat16)
        v_global_flat = np.random.randn(num_global_blocks, num_kv_heads, block_size, head_dim).astype(nl.bfloat16)

        total_q_heads = q_heads_per_rank * group_size * num_groups
        q_global = np.random.randn(total_q_heads, 1, seqlen, head_dim).astype(nl.bfloat16)

        replica_group_lists = [
            list(range(group_idx * group_size, (group_idx + 1) * group_size)) for group_idx in range(num_groups)
        ]
        replica_groups = ReplicaGroup(replica_group_lists)

        def create_inputs(rank_id: int):
            group_id = rank_id // group_size
            rank_in_group = rank_id % group_size

            local_global_block_ids = list(range(rank_in_group, num_global_blocks, group_size))
            num_local_blocks = len(local_global_block_ids)

            k_local = np.zeros((num_local_blocks, num_kv_heads, block_size, head_dim), dtype=nl.bfloat16)
            v_local = np.zeros((num_local_blocks, num_kv_heads, block_size, head_dim), dtype=nl.bfloat16)

            for local_idx, global_blk_id in enumerate(local_global_block_ids):
                k_local[local_idx] = k_global_flat[global_blk_id]
                v_local[local_idx] = v_global_flat[global_blk_id]

            block_tables = np.arange(num_local_blocks, dtype=np.int32).reshape(1, num_local_blocks)
            block_tables = dt.static_cast(block_tables, nl.int32)

            # APC: kvp_q_offset = actual global Q position (runtime), global_q_offset = 0
            cp_offset = dt.static_cast(np.array([[prior_tokens]], dtype=np.int32), nl.int32)
            # valid_num_prior_tokens = number of fully-visible local prior tokens for chunk 0.
            # Rank r's local block i is at global pos i*stride + r*block_size.
            # Fully visible if last token: i*stride + r*block_size + block_size - 1 <= kvp_q_offset.
            stride = group_size * block_size
            rank_offset = rank_in_group * block_size
            threshold = prior_tokens - rank_offset - block_size + 1
            num_fully_visible_blocks = max(0, threshold // stride + 1) if threshold >= 0 else 0
            valid_prior = num_fully_visible_blocks * block_size
            num_prior = dt.static_cast(np.array([[valid_prior]], dtype=np.int32), nl.int32)

            q_start = (group_id * group_size + rank_in_group) * q_heads_per_rank
            q_local = q_global[q_start : q_start + q_heads_per_rank, 0, :, :]

            return {
                "q": q_local,
                "k_cache": k_local,
                "v_cache": v_local,
                "block_tables": block_tables,
                "kvp_q_offset": cp_offset,
                "replica_groups": replica_groups,
                "group_size": group_size,
                "block_size": block_size,
                "seg_size": seg_size,
                "scale": 1.0,
                "global_q_offset": 0,
                "tp_out": tp_out,
                "sliding_window": 0,
                "kvp_rank_id": dt.static_cast(np.array([[rank_in_group]], dtype=np.int32), nl.int32),
                "kvp_group_size": group_size,
                "apc_mode": True,
                "valid_num_prior_tokens": num_prior,
            }

        def create_golden(rank_id: int):
            group_id = rank_id // group_size
            rank_in_group = rank_id % group_size

            k_full = k_global_flat[:, 0, :, :].reshape(-1, head_dim).astype(np.float32)
            v_full = v_global_flat[:, 0, :, :].reshape(-1, head_dim).astype(np.float32)

            outputs = []
            q_start_global = (group_id * group_size + rank_in_group) * q_heads_per_rank

            for head_idx in range(q_heads_per_rank):
                q_head_idx = q_start_global + head_idx
                q = q_global[q_head_idx, 0].astype(np.float32)

                scores = np.matmul(q, k_full.T)

                q_pos = np.arange(prior_tokens, prior_tokens + seqlen).reshape(-1, 1)
                k_pos = np.arange(global_kv_len).reshape(1, -1)
                causal_mask = q_pos < k_pos
                scores = np.where(causal_mask, -np.inf, scores)

                max_scores = np.max(scores, axis=-1, keepdims=True)
                max_scores = np.where(np.isinf(max_scores), 0, max_scores)
                exp_scores = np.exp(scores - max_scores)
                sum_exp = np.sum(exp_scores, axis=-1, keepdims=True)
                sum_exp = np.where(sum_exp == 0, 1, sum_exp)
                attn_weights = exp_scores / sum_exp

                out = np.matmul(attn_weights, v_full)
                outputs.append(out)

            out_stacked = np.stack(outputs, axis=0).astype(nl.bfloat16)
            if tp_out:
                out_stacked = np.transpose(out_stacked, (0, 2, 1))
            return {"out": out_stacked}

        def _torch_ref(
            q,
            k_cache,
            v_cache,
            block_tables,
            kvp_q_offset,
            replica_groups,
            group_size,
            block_size,
            seg_size,
            scale=1.0,
            global_q_offset=0,
            tp_out=False,
            sliding_window=0,
            kvp_rank_id=None,
            kvp_group_size=0,
            apc_mode=False,
            valid_num_prior_tokens=None,
            fp8_packed=False,
            k_scale=None,
            v_scale=None,
        ):
            return create_golden(get_rank())

        framework = CollectiveUnitTestFramework(
            test_manager=test_manager,
            kernel_entry=attention_kv_parallel_segmented_cte,
            torch_ref=_torch_ref,
            per_rank_input_generator=create_inputs,
            collective_ranks=collective_ranks,
        )
        framework.run_test(
            test_config=None,
            compiler_args=CompilerArgs(platform_target=Platforms.TRN2, logical_nc_config=lnc_degree),
            output_keys=["out"],
            rtol=5e-2,
            atol=1e-2,
        )


@pytest.mark.attention
@pytest.mark.kv_parallel
class TestKVParallelSegmentedPrefillModelConfigs:
    """Model-driven tests for KVP segmented prefill attention.

    Exercises production model shapes (Llama-3.2-1B) with DCP prefill configs,
    including APC (chunked context with prior cached tokens).
    """

    _MODEL_PARAMS = "model_name,group_size,q_heads_per_rank,seqlen,head_dim,block_size,seg_size,prior_tokens,num_global_blocks,lnc_degree,tp_out,apc_mode"

    _OPTIMAL_PARAMS, _OPTIMAL_IDS = (
        prepare_model_parametrize(
            {ModelTestType.OPTIMAL: kvp_segmented_attention_cte_model_configs.get(ModelTestType.OPTIMAL, [])}
        )
        if kvp_segmented_attention_cte_model_configs
        else ([], [])
    )

    def _run_model_test(
        self,
        test_manager: Orchestrator,
        model_name: str,
        group_size: int,
        q_heads_per_rank: int,
        seqlen: int,
        head_dim: int,
        block_size: int,
        seg_size: int,
        prior_tokens: int,
        num_global_blocks: int,
        lnc_degree: int,
        tp_out: bool,
        apc_mode: bool,
    ):
        np.random.seed(42)

        num_kv_heads = 1
        num_groups = 1
        collective_ranks = group_size * num_groups

        global_kv_len = num_global_blocks * block_size
        k_global_flat = np.random.randn(num_global_blocks, num_kv_heads, block_size, head_dim).astype(nl.bfloat16)
        v_global_flat = np.random.randn(num_global_blocks, num_kv_heads, block_size, head_dim).astype(nl.bfloat16)

        total_q_heads = q_heads_per_rank * group_size * num_groups
        q_global = np.random.randn(total_q_heads, 1, seqlen, head_dim).astype(nl.bfloat16)

        replica_group_lists = [
            list(range(group_idx * group_size, (group_idx + 1) * group_size)) for group_idx in range(num_groups)
        ]
        replica_groups = ReplicaGroup(replica_group_lists)

        def create_inputs(rank_id: int):
            group_id = rank_id // group_size
            rank_in_group = rank_id % group_size

            local_global_block_ids = list(range(rank_in_group, num_global_blocks, group_size))
            num_local_blocks = len(local_global_block_ids)

            k_local = np.zeros((num_local_blocks, num_kv_heads, block_size, head_dim), dtype=nl.bfloat16)
            v_local = np.zeros((num_local_blocks, num_kv_heads, block_size, head_dim), dtype=nl.bfloat16)

            for local_idx, global_blk_id in enumerate(local_global_block_ids):
                k_local[local_idx] = k_global_flat[global_blk_id]
                v_local[local_idx] = v_global_flat[global_blk_id]

            block_tables = np.arange(num_local_blocks, dtype=np.int32).reshape(1, num_local_blocks)
            block_tables = dt.static_cast(block_tables, nl.int32)

            cp_offset = dt.static_cast(np.array([[prior_tokens]], dtype=np.int32), nl.int32)

            q_start = (group_id * group_size + rank_in_group) * q_heads_per_rank
            q_local = q_global[q_start : q_start + q_heads_per_rank, 0, :, :]

            inputs = {
                "q": q_local,
                "k_cache": k_local,
                "v_cache": v_local,
                "block_tables": block_tables,
                "kvp_q_offset": cp_offset,
                "replica_groups": replica_groups,
                "group_size": group_size,
                "block_size": block_size,
                "seg_size": seg_size,
                "scale": 1.0,
                "global_q_offset": 0,
                "tp_out": tp_out,
                "sliding_window": 0,
                "kvp_rank_id": dt.static_cast(np.array([[rank_in_group]], dtype=np.int32), nl.int32),
                "kvp_group_size": group_size,
                "apc_mode": apc_mode,
            }
            if apc_mode:
                stride = group_size * block_size
                rank_offset = rank_in_group * block_size
                threshold = prior_tokens - rank_offset - block_size + 1
                num_fully_visible_blocks = max(0, threshold // stride + 1) if threshold >= 0 else 0
                valid_prior = num_fully_visible_blocks * block_size
                inputs["valid_num_prior_tokens"] = dt.static_cast(np.array([[valid_prior]], dtype=np.int32), nl.int32)
            return inputs

        def create_golden(rank_id: int):
            group_id = rank_id // group_size
            rank_in_group = rank_id % group_size

            k_full = k_global_flat[:, 0, :, :].reshape(-1, head_dim).astype(np.float32)
            v_full = v_global_flat[:, 0, :, :].reshape(-1, head_dim).astype(np.float32)

            outputs = []
            q_start_global = (group_id * group_size + rank_in_group) * q_heads_per_rank

            for head_idx in range(q_heads_per_rank):
                q_head_idx = q_start_global + head_idx
                q = q_global[q_head_idx, 0].astype(np.float32)

                scores = np.matmul(q, k_full.T)

                q_pos = np.arange(prior_tokens, prior_tokens + seqlen).reshape(-1, 1)
                k_pos = np.arange(global_kv_len).reshape(1, -1)
                causal_mask = q_pos < k_pos
                scores = np.where(causal_mask, -np.inf, scores)

                max_scores = np.max(scores, axis=-1, keepdims=True)
                max_scores = np.where(np.isinf(max_scores), 0, max_scores)
                exp_scores = np.exp(scores - max_scores)
                sum_exp = np.sum(exp_scores, axis=-1, keepdims=True)
                sum_exp = np.where(sum_exp == 0, 1, sum_exp)
                attn_weights = exp_scores / sum_exp

                out = np.matmul(attn_weights, v_full)
                outputs.append(out)

            out_stacked = np.stack(outputs, axis=0).astype(nl.bfloat16)
            if tp_out:
                out_stacked = np.transpose(out_stacked, (0, 2, 1))
            return {"out": out_stacked}

        def _torch_ref(
            q,
            k_cache,
            v_cache,
            block_tables,
            kvp_q_offset,
            replica_groups,
            group_size,
            block_size,
            seg_size,
            scale=1.0,
            global_q_offset=0,
            tp_out=False,
            sliding_window=0,
            kvp_rank_id=None,
            kvp_group_size=0,
            apc_mode=False,
            valid_num_prior_tokens=None,
            fp8_packed=False,
            k_scale=None,
            v_scale=None,
        ):
            return create_golden(get_rank())

        framework = CollectiveUnitTestFramework(
            test_manager=test_manager,
            kernel_entry=attention_kv_parallel_segmented_cte,
            torch_ref=_torch_ref,
            per_rank_input_generator=create_inputs,
            collective_ranks=collective_ranks,
        )
        framework.run_test(
            test_config=None,
            compiler_args=CompilerArgs(platform_target=Platforms.TRN2, logical_nc_config=lnc_degree),
            output_keys=["out"],
            rtol=5e-2,
            atol=1e-2,
        )

    @pytest.mark.parametrize(_MODEL_PARAMS, _OPTIMAL_PARAMS, ids=_OPTIMAL_IDS)
    def test_optimal(
        self,
        test_manager: Orchestrator,
        model_name: str,
        group_size: int,
        q_heads_per_rank: int,
        seqlen: int,
        head_dim: int,
        block_size: int,
        seg_size: int,
        prior_tokens: int,
        num_global_blocks: int,
        lnc_degree: int,
        tp_out: bool,
        apc_mode: bool,
    ):
        """OPTIMAL: Production model configs for KVP segmented prefill with DCP."""
        self._run_model_test(**{k: v for k, v in locals().items() if k != "self"})
