# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.

# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import enum
import os
from dataclasses import dataclass, replace
from functools import lru_cache
from inspect import signature
from test.integration.nkilib.experimental.transformer.test_attention_block_tkg_model_config import (
    attention_block_tkg_model_configs,
)
from test.integration.nkilib.utils.comparators import maxAllClose
from test.integration.nkilib.utils.tensor_generators import (
    np_random_sample_static_quantize_inp,
)
from test.utils.common_dataclasses import (
    MODEL_TEST_TYPE,
    TKG_INFERENCE_ARGS,
    CompilerArgs,
    CustomValidator,
    CustomValidatorWithOutputTensorData,
    KernelArgs,
    PerRankLazyGoldenGenerator,
    PerRankLazyInputGenerator,
    Platforms,
    ValidationArgs,
)
from test.utils.metadata_loader import load_model_configs
from test.utils.metrics_collector import IMetricsCollector
from test.utils.pytest_test_metadata import pytest_test_metadata
from test.utils.test_orchestrator import Orchestrator
from test.utils.unit_test_framework import UnitTestFramework, torch_ref_wrapper
from typing import Any, Optional, Union, final

import neuron_dtypes as dt
import nki
import nki.isa as nisa
import nki.language as nl
import numpy as np
import numpy.typing as npt
import pytest
from nki.collectives import ReplicaGroup
from nkilib_src.nkilib.core.utils.allocator import SbufManager
from nkilib_src.nkilib.core.utils.common_types import QuantizationType
from nkilib_src.nkilib.core.utils.kernel_helpers import get_program_sharding_info, is_hbm_buffer
from nkilib_src.nkilib.core.utils.tensor_view import TensorView
from nkilib_src.nkilib.experimental.transformer.attention_block_tkg import (
    attention_block_tkg,
)
from nkilib_src.nkilib.experimental.transformer.attention_block_tkg_torch import (
    AttentionBlockTkgTorchRef,
)
from typing_extensions import override


class KVScaleTest(enum.Enum):
    """KV cache quantization scale for FP8 test configs.

    Use DEFAULT for standard test scale (240/2.3), or pass a float
    literal. Only used when kv_quant=True.
    """

    DEFAULT = "default"


@dataclass
class AttnBlkTestConfig:
    """Test configuration for attention block TKG kernel."""

    batch: int
    num_heads: int
    d_head: int
    H: int
    H_actual: Optional[int]
    S_ctx: int
    S_max_ctx: int
    S_tkg: int
    block_len: int
    update_cache: bool
    K_cache_transposed: bool
    rmsnorm_X: bool
    skip_rope: bool
    rope_contiguous_layout: bool
    qk_norm_pre_rope: bool
    qk_norm_pre_rope_gamma: bool
    qk_norm_post_rope: bool
    qk_norm_post_rope_gamma: bool
    dtype: Any
    quantization_type: QuantizationType
    lnc: int
    skip_output_projection: bool
    transposed_out: bool
    test_bias: bool
    out_in_sbuf: bool
    input_in_sbuf: bool
    softmax_scale: Optional[float]
    kv_quant: bool
    kv_scale: Optional[Union[KVScaleTest, float]] = None
    KVDP: int = 1
    skip_attention: bool = False


# Maximum memory (GB) for test tensor allocation. Tests exceeding this are skipped.
# Override via environment variable TEST_ATTN_BLK_TKG_MAX_MEMORY_GB.
_DEFAULT_MAX_MEMORY_GB = 20
_MAX_MEMORY_BYTES = int(float(os.environ.get("TEST_ATTN_BLK_TKG_MAX_MEMORY_GB", _DEFAULT_MAX_MEMORY_GB)) * 1024**3)


def _slice_block_kv_for_all_ranks(
    full_K_cache: np.ndarray,
    full_V_cache: np.ndarray,
    full_active_blocks_table: np.ndarray,
    full_kv_cache_update_idx: np.ndarray,
    KVDP: int,
    B_attn: int,
    S_ctx: int,
    block_len: int,
    d_head: int,
) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
    """Slice block KV cache for all ranks in KV data parallelism.

    vLLM treats each DP rank as an independent inference endpoint with its own KV cache.
    Each rank has a local block pool with indices local to that rank's cache, not global indices.

    The test generates golden outputs using a single global KV cache (no KV data parallelism),
    but the kernel under test runs per-rank with local caches. This function converts the global
    KV cache into the per-rank KV cache that vLLM provides.

    For each rank, this function:
    1. Slices active_blocks_table[B, num_active_blocks] on B to get this rank's batches
    2. Finds which global block indices this rank uses
    3. Creates a compact local cache with only this rank's blocks
    4. Remaps active_blocks_table from global to local indices

    Example with B=8, KVDP=4, B_attn=B/KVDP=2, S_ctx=1024, block_len=32:
        Global cache: K_cache[256, 32, d_head]  (256 = B * S_ctx // block_len = 8 * 1024 // 32)
        Rank 0 uses batches [0:2], references global blocks [100, 147, 212, 199, ...]
        After slicing: K_cache_0[64, 32, d_head] with local indices [0, 1, 2, 3, ..., 62, 63]
            (64 = B_attn * S_ctx // block_len = 2 * 1024 // 32)
        active_blocks_table remapped: [100, 147, 212, 199, ...] -> [0, 1, 2, 3, ...]

    Args:
        full_K_cache (np.ndarray): Global K cache [num_blocks, block_len, d_head]
        full_V_cache (np.ndarray): Global V cache [num_blocks, block_len, d_head]
        full_active_blocks_table (np.ndarray): Global block table [B, num_active_blocks]
        full_kv_cache_update_idx (np.ndarray): Global KV update indices [B, 1]
        KVDP (int): KV data parallelism (number of ranks)
        B_attn (int): Batch size of KV cache per rank
        S_ctx (int): Context sequence length
        block_len (int): Block length
        d_head (int): Head dimension

    Returns:
        K_cache (list[np.ndarray]): Per-rank K caches, each [num_blocks/KVDP, block_len, d_head]
        V_cache (list[np.ndarray]): Per-rank V caches, each [num_blocks/KVDP, block_len, d_head]
        active_blocks_table (list[np.ndarray]): Remapped active_blocks_tables, each [B_attn, num_active_blocks]
        kv_cache_update_idx (list[np.ndarray]): Remapped KV update indices, each [B_attn, 1]
    """
    # Validate input shapes
    assert full_K_cache.ndim == 3 and full_K_cache.shape[1:] == (block_len, d_head)
    assert full_V_cache.shape == full_K_cache.shape
    assert full_active_blocks_table.ndim == 2 and full_active_blocks_table.shape[0] == KVDP * B_attn
    assert full_kv_cache_update_idx.ndim == 2 and full_kv_cache_update_idx.shape[0] == KVDP * B_attn

    blocks_per_rank = B_attn * S_ctx // block_len
    K_cache, V_cache, active_blocks_table, kv_cache_update_idx = [], [], [], []

    for rank_idx in range(KVDP):
        # Slice active_blocks_table for this rank's batches
        rank_active_blocks_table = full_active_blocks_table[rank_idx * B_attn : (rank_idx + 1) * B_attn]

        # Find which global block indices this rank uses (excluding -1 padding)
        used_blocks = np.unique(rank_active_blocks_table[rank_active_blocks_table != -1])

        # Create compact local cache with only this rank's blocks
        new_k = np.zeros((blocks_per_rank, block_len, d_head), dtype=full_K_cache.dtype)
        new_v = np.zeros((blocks_per_rank, block_len, d_head), dtype=full_V_cache.dtype)

        # Copy block data and build global->local index mapping
        block_map = {}
        for new_idx, old_idx in enumerate(used_blocks):
            new_k[new_idx], new_v[new_idx] = full_K_cache[old_idx], full_V_cache[old_idx]
            block_map[old_idx] = new_idx

        # Remap active_blocks_table from global to local indices
        new_abt = rank_active_blocks_table.copy()
        for batch_idx in range(B_attn):
            for col_idx in range(rank_active_blocks_table.shape[1]):
                if rank_active_blocks_table[batch_idx, col_idx] != -1:
                    new_abt[batch_idx, col_idx] = block_map[rank_active_blocks_table[batch_idx, col_idx]]

        # Remap kv_cache_update_idx (block_idx * block_len + offset)
        rank_kv_idx = full_kv_cache_update_idx[rank_idx * B_attn : (rank_idx + 1) * B_attn].copy()
        for batch_idx in range(B_attn):
            if rank_kv_idx[batch_idx, 0] != np.uint32(-1):
                old_block, offset = divmod(int(rank_kv_idx[batch_idx, 0]), block_len)
                rank_kv_idx[batch_idx, 0] = block_map.get(old_block, old_block) * block_len + offset

        K_cache.append(new_k)
        V_cache.append(new_v)
        active_blocks_table.append(new_abt)
        kv_cache_update_idx.append(rank_kv_idx)

    return K_cache, V_cache, active_blocks_table, kv_cache_update_idx


def _create_per_rank_inputs(
    base_input: dict, KVDP: int, q_heads: int, d_head: int, B_attn: int, S_tkg: int, S_ctx: int, block_len: int
) -> tuple:
    """Create per-rank kernel inputs for KV data parallelism tests.

    Slices kernel inputs per rank for multi-rank execution.
    Each rank gets a slice of the batch dimension for K/V cache and mask,
    while Q heads are distributed across ranks.

    Args:
        base_input (dict): Full kernel input dictionary with all tensors
        KVDP (int): KV data parallelism (number of ranks)
        q_heads (int): Number of query heads per rank
        d_head (int): Head dimension
        B_attn (int): Batch size of KV cache per rank (total_batch / KVDP)
        S_tkg (int): Token generation sequence length
        S_ctx (int): Context sequence length
        block_len (int): Block length for block KV cache (0 for flat cache)

    Returns:
        per_rank_input (PerRankLazyInputGenerator): Generator that creates inputs for each rank
        per_rank_cache (dict): Pre-sliced tensors for golden reference computation
    """
    total_q_heads = KVDP * q_heads

    replica_group = ReplicaGroup([list(range(KVDP))])

    # Slice W_qkv per rank: Q portion sliced (each rank gets q_heads), K/V replicated (GQA shares 1 KV head)
    full_W_qkv = base_input['W_qkv']
    q_end = total_q_heads * d_head
    k_end = q_end + d_head
    v_end = k_end + d_head
    W_q_full, W_k, W_v = full_W_qkv[:, :q_end], full_W_qkv[:, q_end:k_end], full_W_qkv[:, k_end:v_end]
    w_qkV_cache = [
        np.concatenate([W_q_full[:, rank_idx * q_heads * d_head : (rank_idx + 1) * q_heads * d_head], W_k, W_v], axis=1)
        for rank_idx in range(KVDP)
    ]

    # Slice bias_qkv if present
    full_bias_qkv = base_input.get('bias_qkv')
    if full_bias_qkv is not None:
        bias_q_full, bias_k, bias_v = (
            full_bias_qkv[:, :q_end],
            full_bias_qkv[:, q_end:k_end],
            full_bias_qkv[:, k_end:v_end],
        )
        bias_qkV_cache = [
            np.concatenate(
                [bias_q_full[:, rank_idx * q_heads * d_head : (rank_idx + 1) * q_heads * d_head], bias_k, bias_v],
                axis=1,
            )
            for rank_idx in range(KVDP)
        ]
    else:
        bias_qkV_cache = [None] * KVDP

    # Slice W_out per rank
    full_W_out = base_input.get('W_out')
    w_out_slices = (
        [
            full_W_out[rank_idx * q_heads * d_head : (rank_idx + 1) * q_heads * d_head, :].copy()
            for rank_idx in range(KVDP)
        ]
        if full_W_out is not None
        else [None] * KVDP
    )

    # Slice mask per rank
    # full_mask shape: (S_ctx, B, total_q_heads, S_tkg)
    # Kernel needs: (S_ctx, B_attn, total_q_heads, S_tkg) - sliced batch, all heads (after gather)
    # Golden needs: (S_ctx, B, q_heads, S_tkg) - full batch, 1 Q head
    full_mask = base_input['attention_mask']
    mask_slices = [full_mask[:, rank_idx * B_attn : (rank_idx + 1) * B_attn, :, :] for rank_idx in range(KVDP)]
    golden_mask = full_mask[:, :, :q_heads, :]

    # Slice K/V cache per rank
    full_K_cache, full_V_cache = base_input['K_cache'], base_input['V_cache']
    full_active_blocks_table = base_input.get('active_blocks_table')
    full_kv_cache_update_idx = base_input.get('kv_cache_update_idx')

    if block_len > 0:
        K_cache, V_cache, active_blocks_table, kv_cache_update_idx = _slice_block_kv_for_all_ranks(
            full_K_cache,
            full_V_cache,
            full_active_blocks_table,
            full_kv_cache_update_idx,
            KVDP,
            B_attn,
            S_ctx,
            block_len,
            d_head,
        )
    else:
        K_cache = [full_K_cache[rank_idx * B_attn : (rank_idx + 1) * B_attn] for rank_idx in range(KVDP)]
        V_cache = [full_V_cache[rank_idx * B_attn : (rank_idx + 1) * B_attn] for rank_idx in range(KVDP)]
        active_blocks_table = [None] * KVDP
        kv_cache_update_idx = [
            full_kv_cache_update_idx[rank_idx * B_attn : (rank_idx + 1) * B_attn] for rank_idx in range(KVDP)
        ]

    def create_per_rank_input(rank_id: int) -> dict:
        result = base_input.copy()
        result['W_qkv'] = w_qkV_cache[rank_id]
        result['bias_qkv'] = bias_qkV_cache[rank_id]
        result['W_out'] = w_out_slices[rank_id]
        result['K_cache'] = K_cache[rank_id]
        result['V_cache'] = V_cache[rank_id]
        result['attention_mask'] = mask_slices[rank_id]
        result['active_blocks_table'] = active_blocks_table[rank_id]
        result['kv_cache_update_idx'] = kv_cache_update_idx[rank_id]
        result['KVDP'] = KVDP
        result['KVDP_replica_group'] = replica_group
        return result

    per_rank_input = PerRankLazyInputGenerator(generator=create_per_rank_input)
    per_rank_input.base_input = base_input  # Store for golden reference

    per_rank_cache = {
        'w_qkv': w_qkV_cache,
        'bias_qkv': bias_qkV_cache,
        'w_out': w_out_slices,
        'golden_mask': golden_mask,
        'full_active_blocks_table': full_active_blocks_table,
    }
    return per_rank_input, per_rank_cache


def _slice_golden_KV_cache_for_rank(
    rank_golden: dict, rank_id: int, B_attn: int, block_len: int, d_head: int, S_ctx: int, per_rank_cache: dict
) -> dict:
    """Slice golden K/V cache output for a specific rank.

    Golden computes with full batch B, so K/V_cache_updated has shape [B, ...].
    This function slices to [B/KVDP, ...] to match the kernel's per-rank output.
    X_out is returned unchanged (already has correct shape from golden).

    Args:
        rank_golden (dict): Full golden output with 'X_out', 'K_cache_updated', 'V_cache_updated'
        rank_id (int): Rank index to slice for
        B_attn (int): Batch size of KV cache per rank (B/KVDP)
        block_len (int): Block length for block KV cache (0 for flat cache)
        d_head (int): Head dimension
        S_ctx (int): Context sequence length
        per_rank_cache (dict): Pre-computed slicing info including 'full_active_blocks_table'

    Returns:
        dict: Golden output with X_out unchanged, K/V_cache sliced to [B/KVDP, ...]
    """
    if block_len > 0:
        blocks_per_rank = B_attn * S_ctx // block_len
        full_active_blocks_table = per_rank_cache['full_active_blocks_table']
        rank_active_blocks_table = full_active_blocks_table[rank_id * B_attn : (rank_id + 1) * B_attn]
        used_blocks = np.unique(rank_active_blocks_table[rank_active_blocks_table != -1])

        golden_k = np.zeros((blocks_per_rank, block_len, d_head), dtype=rank_golden['K_cache_updated'].dtype)
        golden_v = np.zeros((blocks_per_rank, block_len, d_head), dtype=rank_golden['V_cache_updated'].dtype)
        for new_idx, old_idx in enumerate(used_blocks):
            golden_k[new_idx] = rank_golden['K_cache_updated'][old_idx]
            golden_v[new_idx] = rank_golden['V_cache_updated'][old_idx]
    else:
        golden_k = rank_golden['K_cache_updated'][rank_id * B_attn : (rank_id + 1) * B_attn]
        golden_v = rank_golden['V_cache_updated'][rank_id * B_attn : (rank_id + 1) * B_attn]

    return {'X_out': rank_golden['X_out'], 'K_cache_updated': golden_k, 'V_cache_updated': golden_v}


def estimate_test_memory_bytes(cfg: AttnBlkTestConfig) -> int:
    """Estimate total host memory (bytes) for a test config without allocating tensors.

    Computes the sum of all tensor sizes created during the test.
    """
    batch = cfg.batch
    num_heads = cfg.num_heads
    d_head = cfg.d_head
    H = cfg.H
    S_ctx = cfg.S_ctx
    S_max_ctx = cfg.S_max_ctx
    S_tkg = cfg.S_tkg
    block_len = cfg.block_len
    kv_quant = cfg.kv_quant
    KVDP = cfg.KVDP
    quantization_type = cfg.quantization_type
    skip_output_projection = cfg.skip_output_projection

    is_quantized = quantization_type != QuantizationType.NONE
    elem = 1 if is_quantized else 2  # fp8=1, bf16=2
    kv_elem = 1 if kv_quant else elem
    is_block_kv = block_len > 0

    # KVDP inflates num_heads for input generation (mirrors _run_attention_block_test)
    effective_heads = KVDP * num_heads if KVDP > 1 else num_heads
    I = d_head * (effective_heads + 2)  # num_kv_heads=1 always

    total = 0

    # --- generate_kernel_inputs ---
    total += batch * S_tkg * H * elem  # X
    total += H * I * elem  # W_qkv
    if cfg.rmsnorm_X:
        total += H * elem  # rmsnorm_X_gamma
    if cfg.test_bias:
        total += I * elem  # bias_qkv
        total += H * elem  # bias_out
    if not cfg.skip_rope:
        total += 2 * (d_head // 2) * batch * S_tkg * elem  # cos + sin
    if cfg.qk_norm_pre_rope_gamma:
        total += 2 * d_head * elem  # W_rmsnorm_Q/K_pre_rope
    if cfg.qk_norm_post_rope_gamma:
        total += 2 * d_head * elem  # W_rmsnorm_Q/K_post_rope

    # KV cache
    if is_block_kv:
        num_blocks = batch * S_ctx // block_len
        total += 2 * num_blocks * block_len * d_head * kv_elem  # K + V cache
        total += batch * (S_ctx // block_len) * 4  # active_blocks_table (uint32)
    else:
        total += 2 * batch * S_max_ctx * d_head * kv_elem  # K + V cache

    total += S_ctx * batch * effective_heads * S_tkg  # attention_mask (uint8)
    total += batch * 4  # kv_cache_update_idx (uint32)
    total += batch * 8  # cache_len (int64)

    if not skip_output_projection:
        total += effective_heads * d_head * H * elem  # W_out
    if kv_quant:
        total += 4 + 128 * 4  # k_scale + v_scale (float32)
    if is_quantized:
        total += 128 * 3 * 4 + 128 * 4  # weight/input dequant scales qkv
        if not skip_output_projection:
            total += 2 * 128 * 4  # weight/input dequant scales out

    # --- KVDP per-rank copies (_create_per_rank_inputs) ---
    if KVDP > 1:
        B_attn = batch // KVDP
        if is_block_kv:
            rank_blocks = B_attn * S_ctx // block_len
            total += KVDP * 2 * rank_blocks * block_len * d_head * kv_elem
            total += KVDP * B_attn * (S_ctx // block_len) * 4
        else:
            total += KVDP * 2 * B_attn * S_max_ctx * d_head * kv_elem
        total += KVDP * S_ctx * B_attn * effective_heads * S_tkg  # per-rank masks
        total += KVDP * H * d_head * (num_heads + 2) * elem  # per-rank W_qkv
        if not skip_output_projection:
            total += KVDP * num_heads * d_head * H * elem  # per-rank W_out

    # --- Golden reference overhead (float32 intermediates) ---
    total += batch * S_tkg * H * 4  # X as f32
    total += H * I * 4  # W_qkv as f32
    total += batch * effective_heads * S_tkg * d_head * 4  # QKV output
    total += 2 * batch * effective_heads * S_tkg * S_ctx * 4  # attn scores + softmax
    total += batch * effective_heads * S_tkg * d_head * 4  # attn output
    if not skip_output_projection:
        total += batch * S_tkg * H * 4  # output projection

    # --- output_placeholder (zeros_like golden) ---
    total += batch * S_tkg * H * elem  # X_out
    if not is_block_kv:
        total += 2 * batch * S_max_ctx * d_head * kv_elem  # K/V out placeholders

    return total


def generate_kernel_inputs(cfg: AttnBlkTestConfig):
    from test.integration.nkilib.core.attention.test_attention_tkg_utils import (
        gen_deterministic_active_block_table,
    )

    from nkilib_src.nkilib.core.attention.gen_mask_tkg_torch import build_full_attention_mask

    # Short aliases for dimensions used repeatedly in shape expressions
    dtype = cfg.dtype
    batch, d_head, H = cfg.batch, cfg.d_head, cfg.H
    S_ctx, S_max_ctx, S_tkg = cfg.S_ctx, cfg.S_max_ctx, cfg.S_tkg
    H_actual = cfg.H_actual if cfg.H_actual is not None else H
    num_q_heads = cfg.num_heads
    num_kv_heads = 1

    eps = 1e-5 if dtype == np.float32 else 1e-3

    # ── Tensor generators ──────────────────────────────────────────────────────
    #
    # bf16 has 7 mantissa bits → 1 ULP = 2⁻⁷ relative to the significand.
    # With round-to-nearest, max error is ½ ULP = 2⁻⁸. Worst case occurs at the
    # bottom of each exponent bucket (significand=1.0): 2⁻⁸/1.0 = 1/256 ≈ 0.4%.
    # This is the worst-case relative error when quantizing f32 to bf16.
    #
    # With Gaussian(0,1) weights, QKV projections have std ≈ √H, so attention
    # scores (QK^T) have std ≈ H (thousands). When two positions score almost
    # identically, 0.4% noise can flip which one wins:
    #
    #   f32 scores:  [2001.0, 1998.5, 1950.0, 1870.0]
    #   bf16 noise:  [  -8.0,   +6.0,   -3.0,   +1.0]   (~0.4% of 2000)
    #   bf16 scores: [1993.0, 2004.5, 1947.0, 1871.0]   ← position 1 now wins
    #
    #   After softmax subtracts max:
    #     f32:  [  0.0,  -2.5, ...]  → exp → position 0 gets 92%
    #     bf16: [-11.5,   0.0, ...]  → exp → position 1 gets 99.99%
    #
    # The fundamental problem: softmax exponentiates the *differences* (after
    # subtracting max), but bf16 noise scales with the *magnitudes*. When noise
    # exceeds the difference signal, the ranking can flip. At large scale,
    # softmax behaves as argmax and picks a completely wrong V row. This causes
    # random heads to fail — it's statistical, depending on which positions
    # happen to score nearly identically.
    #
    # At small scale (scores ≈ 1), centered values are near zero where exp() is
    # flat, so even if a flip occurs, the probabilities barely change.
    #
    # To keep scores O(1), we use W ~ N(0, 1/√fan_in). This scaling is "unitary"
    # in the sense that it preserves variance through matmuls. Gammas, cos/sin,
    # and other tensors are chosen similarly to keep std ≈ 0.5–1.0 up to softmax.
    #
    # Notes:
    # - Not all test configs use normalization, so we can't rely on it alone.
    # - Large biases can mask attention bugs; small biases can be masked by them.
    # - Configs vary (RMSNorm, QK-norm, RoPE, bias on/off), but std ≈ 0.5–1.0
    #   with reasonable biases works across all of them.
    # - This is close to typical weight initialization in training.
    # - After softmax, larger weights are fine — the peaky regime is past.
    #
    # See numerical_precision_in_transformer_attention.md for full analysis.
    _rng = np.random.default_rng(0)

    def uniform_activation(shape, dtype):
        """Uniform[-1, 1]"""
        return np.ascontiguousarray(dt.static_cast(_rng.uniform(-1.0, 1.0, shape).astype(np.float32), dtype))

    def gaussian(shape, dtype, std):
        """N(0, std)"""
        return np.ascontiguousarray(dt.static_cast(_rng.normal(0.0, std, shape).astype(np.float32), dtype))

    def fan_in_projection(shape, dtype, fan_in):
        """N(0, 1/√fan_in). Keeps matmul output variance ≈ input variance."""
        return gaussian(shape, dtype, std=1.0 / np.sqrt(fan_in))

    def near_unity(shape, dtype):
        """Uniform[0.5, 1.5]. RMSNorm gammas are ~1.0 in trained models."""
        return np.ascontiguousarray(dt.static_cast(_rng.uniform(0.5, 1.5, shape).astype(np.float32), dtype))

    def small_bias(shape, dtype):
        """Uniform[-0.1, 0.1]. Biases are small in trained models."""
        return np.ascontiguousarray(dt.static_cast(_rng.uniform(-0.1, 0.1, shape).astype(np.float32), dtype))

    generate_quant_tensor = np_random_sample_static_quantize_inp()

    # -- input: post-layernorm activations are O(1)
    X = uniform_activation((batch, S_tkg, H), dtype)
    X[:, :, H_actual:] = 0.0

    # -- rmsnorm X: gamma weights are ~1.0 in trained models
    rmsnorm_X_gamma = near_unity((1, H), dtype) if cfg.rmsnorm_X else None

    # -- qkv projections, optional bias
    if cfg.quantization_type == QuantizationType.NONE:
        # W_qkv: projection from hidden dim H → QKV, fan-in-scaled by fan_in=H
        W_qkv = fan_in_projection((H, (num_q_heads + 2 * num_kv_heads) * d_head), dtype, fan_in=H)
        weight_dequant_scale_qkv = None
        input_dequant_scale_qkv = None
    else:
        W_q, w_scale_q, input_dequant_scale_qkv = generate_quant_tensor(
            shape=(H, num_q_heads * d_head), dtype=nl.float8_e4m3
        )
        W_k, w_scale_k, _ = generate_quant_tensor(shape=(H, num_kv_heads * d_head), dtype=nl.float8_e4m3)
        W_v, w_scale_v, _ = generate_quant_tensor(shape=(H, num_kv_heads * d_head), dtype=nl.float8_e4m3)
        W_qkv = np.concatenate([W_q, W_k, W_v], axis=1)
        weight_dequant_scale_qkv = np.array([[w_scale_q, w_scale_k, w_scale_v]])
        weight_dequant_scale_qkv = np.broadcast_to(weight_dequant_scale_qkv, (128, 3))
        input_dequant_scale_qkv = np.broadcast_to(input_dequant_scale_qkv, (128, 1))
    bias_qkv = small_bias((1, (num_q_heads + 2 * num_kv_heads) * d_head), dtype) if cfg.test_bias else None

    # -- rmsnorm QK pre RoPE gamma weights
    W_rmsnorm_Q_pre_rope = near_unity((1, d_head), dtype) if cfg.qk_norm_pre_rope_gamma else None
    W_rmsnorm_K_pre_rope = near_unity((1, d_head), dtype) if cfg.qk_norm_pre_rope_gamma else None
    # -- RoPE: cos/sin are bounded to [-1, 1] by definition
    cos = None if cfg.skip_rope else uniform_activation((d_head // 2, batch, S_tkg), dtype)
    sin = None if cfg.skip_rope else uniform_activation((d_head // 2, batch, S_tkg), dtype)

    # -- rmsnorm QK post RoPE
    W_rmsnorm_Q_post_rope = near_unity((1, d_head), dtype) if cfg.qk_norm_post_rope_gamma else None
    W_rmsnorm_K_post_rope = near_unity((1, d_head), dtype) if cfg.qk_norm_post_rope_gamma else None

    # -- Attention (and KV cache)
    is_block_kv = cfg.block_len > 0

    # Determine cache shapes
    if is_block_kv:
        assert not cfg.K_cache_transposed
        assert S_ctx % cfg.block_len == 0
        assumed_num_cache_blocks = batch * S_ctx // cfg.block_len
        K_cache_shape = V_cache_shape = (assumed_num_cache_blocks, cfg.block_len, d_head)
    else:
        assumed_num_cache_blocks = 0
        K_cache_shape = (batch, 1, d_head, S_max_ctx) if cfg.K_cache_transposed else (batch, 1, S_max_ctx, d_head)
        V_cache_shape = (batch, 1, S_max_ctx, d_head)

    # Generate KV cache in FP8 when kv_quant=True
    kv_cache_dtype = nl.float8_e4m3 if cfg.kv_quant else dtype
    if cfg.kv_quant:
        # Determine KV scale
        if cfg.kv_scale is KVScaleTest.DEFAULT:
            # FP8 scale for K/V from QKV projection (K = X @ W where X ~ Uniform[-1,1], W ~ N(0, 1/√H))
            # Var(Uniform[-1,1]) = (1-(-1))²/12 = 1/3
            # Var(K) = H × Var(X) × Var(W) = H × (1/3) × (1/H) = 1/3  =>  std(K) ≈ 0.58
            # Max ≈ 4×std ≈ 2.3 (99.99% coverage), scale = FP8_max / max ≈ 240 / 2.3 ≈ 104
            k_scale_scalar = 240.0 / 2.3
            v_scale_scalar = 240.0 / 2.3
        else:
            assert isinstance(
                cfg.kv_scale, float
            ), f"kv_scale must be KVScaleTest.DEFAULT or a float, got {cfg.kv_scale}"
            k_scale_scalar = cfg.kv_scale
            v_scale_scalar = cfg.kv_scale
        # Use different shapes to test both broadcast (1,1) and per-partition (PMAX,1) paths
        k_scale = np.full((1, 1), k_scale_scalar, dtype=np.float32)
        v_scale = np.full((128, 1), v_scale_scalar, dtype=np.float32)

        # Generate KV cache matching the distribution of scaled K/V:
        # K/V are ~Gaussian (CLT: sum of H products) with std(K)×scale ≈ 0.58 × 104 ≈ 60
        # Use kv_std=60 to match the expected distribution
        np.random.seed(42)
        kv_std = 60.0
        K_cache_f32 = np.random.normal(0, kv_std, K_cache_shape).astype(np.float32)
        V_cache_f32 = np.random.normal(0, kv_std, V_cache_shape).astype(np.float32)
        K_cache = dt.static_cast(np.clip(K_cache_f32, -240, 240), kv_cache_dtype)
        V_cache = dt.static_cast(np.clip(V_cache_f32, -240, 240), kv_cache_dtype)
    else:
        # KV cache stores projected K/V values which are O(1) after fan-in-scaled projection
        K_cache = uniform_activation(K_cache_shape, kv_cache_dtype)
        V_cache = uniform_activation(V_cache_shape, kv_cache_dtype)
        k_scale = None
        v_scale = None

    # pos_id (shape=(batch, 1)) defines the first position to append new KV to cache, per batch element
    cache_len = ((np.arange(batch) * 3 + (S_ctx // 4 * 3)) % (S_ctx - S_tkg))[:, np.newaxis]
    assert cache_len.max() < (S_ctx - S_tkg)  # Make sure not to go out of bound.
    import torch

    cache_lens_torch = torch.from_numpy(cache_len.flatten()).to(torch.float32)
    attention_mask = build_full_attention_mask(
        cache_lens=cache_lens_torch,
        batch=batch,
        num_heads=cfg.num_heads,
        s_active=S_tkg,
        s_ctx=S_ctx,
        lnc=cfg.lnc,
        block_len=cfg.block_len,
        include_active_mask=True,
        transposed=True,
    ).numpy()  # mask: (S_ctx, batch, num_heads, S_tkg)
    attention_mask = dt.static_cast(np.ascontiguousarray(attention_mask), dtype=np.uint8)

    active_blocks_table = (
        gen_deterministic_active_block_table(
            batch, S_ctx, S_tkg, cache_len, cfg.block_len, batch * S_ctx // cfg.block_len
        ).astype(np.uint32)
        if is_block_kv
        else None
    )  # (B, S_ctx // block_len)

    # kv_cache_update_idx is (batch,) containing the start position for consecutive writes
    def generate_kv_cache_update_idx():
        if cfg.block_len == 0:
            return cache_len.astype(np.uint32)

        # Block KV: translate logical position to physical slot_mapping
        logical_blks = cache_len // cfg.block_len
        offset_in_blk = cache_len % cfg.block_len
        physical_blks = active_blocks_table[np.arange(batch)[:, None], logical_blks]
        physical_kv_cache_update_idx = physical_blks * cfg.block_len + offset_in_blk
        # Mask update for last batch element to test scenario when it is just padding
        if batch > 1:
            physical_kv_cache_update_idx[-1] = -1
        return physical_kv_cache_update_idx.astype(np.uint32)

    kv_cache_update_idx = generate_kv_cache_update_idx()

    # Output projection
    weight_dequant_scale_out = None
    input_dequant_scale_out = None
    if cfg.skip_output_projection:
        W_out = None
    elif cfg.quantization_type == QuantizationType.NONE:
        # W_out: projection from attention output → H. After softmax, probs@V has low std.
        # Use std=0.5 to scale output back up so bias is meaningful (not at noise level).
        W_out = gaussian((cfg.num_heads * d_head, H), dtype, std=0.5)
    else:
        W_out, weight_dequant_scale_out, input_dequant_scale_out = generate_quant_tensor(
            shape=(cfg.num_heads * d_head, H), dtype=nl.float8_e4m3
        )
        weight_dequant_scale_out = np.broadcast_to(weight_dequant_scale_out, (128, 1))
        input_dequant_scale_out = np.broadcast_to(input_dequant_scale_out, (128, 1))

    # bias_out: match the output projection scale (~0.1) so bias doesn't dominate.
    bias_out = small_bias((1, H), dtype) if cfg.test_bias else None

    return {
        # -- input
        "X": X,
        "X_in_sb": cfg.input_in_sbuf,
        "X_hidden_dim_actual": H_actual,
        # -- rmsnorm X
        "rmsnorm_X_enabled": cfg.rmsnorm_X,
        "rmsnorm_X_eps": eps,
        "rmsnorm_X_gamma": rmsnorm_X_gamma,
        # -- qkv projections
        "W_qkv": W_qkv,
        "bias_qkv": bias_qkv,
        "quantization_type_qkv": cfg.quantization_type,
        "weight_dequant_scale_qkv": weight_dequant_scale_qkv,
        "input_dequant_scale_qkv": input_dequant_scale_qkv,
        # -- QK rmsnorm pre RoPE
        "rmsnorm_QK_enabled": cfg.qk_norm_pre_rope,
        "rmsnorm_QK_eps": eps,
        "W_rmsnorm_Q_pre_rope": W_rmsnorm_Q_pre_rope,
        "W_rmsnorm_K_pre_rope": W_rmsnorm_K_pre_rope,
        # -- RoPE
        "cos": cos,
        "sin": sin,
        "rope_contiguous_layout": cfg.rope_contiguous_layout,
        # -- QK rmsnorm post RoPE
        "rmsnorm_QK_post_rope_enabled": cfg.qk_norm_post_rope,
        "rmsnorm_QK_post_rope_eps": eps,
        "W_rmsnorm_Q_post_rope": W_rmsnorm_Q_post_rope,
        "W_rmsnorm_K_post_rope": W_rmsnorm_K_post_rope,
        # -- attention
        "skip_attention": cfg.skip_attention,
        "K_cache_transposed": cfg.K_cache_transposed,
        "active_blocks_table": active_blocks_table,
        "K_cache": K_cache,
        "V_cache": V_cache,
        "attention_mask": attention_mask,
        "sink": None,
        "softmax_scale": cfg.softmax_scale,
        # -- FP8 KV cache quantization
        "k_scale": k_scale,
        "v_scale": v_scale,
        # -- KV cache update
        "update_cache": cfg.update_cache,
        "kv_cache_update_idx": kv_cache_update_idx,
        # -- output projection
        "W_out": W_out,
        "bias_out": bias_out,
        "quantization_type_out": cfg.quantization_type,
        "weight_dequant_scale_out": weight_dequant_scale_out,
        "input_dequant_scale_out": input_dequant_scale_out,
        # -- output
        "transposed_out": cfg.transposed_out,
        "out_in_sb": cfg.out_in_sbuf,
        # -- KV data parallelism
        "KVDP": cfg.KVDP,
        "KVDP_replica_group": None,
    }


# wrapper to test SBUF IO
def attention_block_tkg_kernel_test_wrapper(
    # -- input
    X: nl.ndarray,
    *,
    X_in_sb: bool,
    X_hidden_dim_actual: Optional[int],
    # -- rmsnorm X
    rmsnorm_X_enabled: bool,
    rmsnorm_X_eps: Optional[float],
    rmsnorm_X_gamma: Optional[nl.ndarray],
    # -- qkv projections
    W_qkv: nl.ndarray,
    bias_qkv: Optional[nl.ndarray],
    quantization_type_qkv: QuantizationType,
    weight_dequant_scale_qkv: Optional[nl.ndarray],
    input_dequant_scale_qkv: Optional[nl.ndarray],
    # -- QK rmsnorm pre RoPE
    rmsnorm_QK_enabled: bool,
    rmsnorm_QK_eps: Optional[float],
    W_rmsnorm_Q_pre_rope: Optional[nl.ndarray],
    W_rmsnorm_K_pre_rope: Optional[nl.ndarray],
    # -- RoPE embeddings
    cos: Optional[nl.ndarray],
    sin: Optional[nl.ndarray],
    rope_contiguous_layout: bool,
    # -- QK rmsnorm post RoPE
    rmsnorm_QK_post_rope_enabled: bool,
    rmsnorm_QK_post_rope_eps: float,
    W_rmsnorm_Q_post_rope: Optional[nl.ndarray],
    W_rmsnorm_K_post_rope: Optional[nl.ndarray],
    # -- attention
    skip_attention: bool,
    K_cache_transposed: bool,
    active_blocks_table: Optional[nl.ndarray],
    K_cache: nl.ndarray,
    V_cache: nl.ndarray,
    attention_mask: nl.ndarray,
    sink: Optional[nl.ndarray],
    softmax_scale: Optional[float],
    # -- FP8 KV cache quantization
    k_scale: Optional[nl.ndarray],
    v_scale: Optional[nl.ndarray],
    # -- KV cache update
    update_cache: bool,
    kv_cache_update_idx: nl.ndarray,
    # -- output projection
    W_out: Optional[nl.ndarray],
    bias_out: Optional[nl.ndarray],
    quantization_type_out: QuantizationType,
    weight_dequant_scale_out: Optional[nl.ndarray],
    input_dequant_scale_out: Optional[nl.ndarray],
    # -- output
    transposed_out: bool,
    out_in_sb: bool,
    sbm: Optional[SbufManager] = None,
    # -- KV data parallelism
    KVDP: int = 1,
    KVDP_replica_group=None,
):
    B, S_tkg, H = X.shape
    if X_in_sb:
        # QKV_tkg requires the input shape to be (pmax, B*S, H // pmax)
        assert H % 128 == 0, "H must be divisible by 128"
        H0 = nl.tile_size.pmax
        H1 = H // 128
        BxS = B * S_tkg

        # Check program dimensionality
        _, lnc, _ = get_program_sharding_info()
        assert H1 % lnc == 0

        X_sb = nl.ndarray((H0, BxS, H1), X.dtype, nl.sbuf, name="X_sb")
        X_hbm = X.reshape((BxS, lnc, H0, H1 // lnc))

        """
        Note how X@HBM is read to SBUF: The full H dimension is divided into (lnc, H0=128, H1//lnc).
        Per SBUF partition (the H0=128 dim), we read H1//lnc values from each of the lnc chunks,
        interleaving them to reconstruct the full H1 dimension in SBUF while transposing the layout
        from (BxS, lnc, H0, H1//lnc) to (H0, BxS, H1). This matches how qkv_tkg() kernel expects
        SBUF input and constrains attention_block_tkg() SBUF input layout.
        """
        nisa.dma_copy(
            dst=TensorView(X_sb).reshape_dim(2, (lnc, -1)).get_view(),
            src=TensorView(X_hbm)
            .rearrange(('BS', 'lnc', 'H0', 'H1 // lnc'), ('H0', 'BS', 'lnc', 'H1 // lnc'))
            .get_view(),
        )

        X = X_sb

    kernel_output, K_hbm_out, V_hbm_out = attention_block_tkg(
        X=X,
        X_hidden_dim_actual=X_hidden_dim_actual,
        rmsnorm_X_enabled=rmsnorm_X_enabled,
        rmsnorm_X_eps=rmsnorm_X_eps,
        rmsnorm_X_gamma=rmsnorm_X_gamma,
        W_qkv=W_qkv,
        bias_qkv=bias_qkv,
        quantization_type_qkv=quantization_type_qkv,
        weight_dequant_scale_qkv=weight_dequant_scale_qkv,
        input_dequant_scale_qkv=input_dequant_scale_qkv,
        rmsnorm_QK_pre_rope_enabled=rmsnorm_QK_enabled,
        rmsnorm_QK_pre_rope_eps=rmsnorm_QK_eps if rmsnorm_QK_eps else 1e-5,
        rmsnorm_QK_pre_rope_W_Q=W_rmsnorm_Q_pre_rope,
        rmsnorm_QK_pre_rope_W_K=W_rmsnorm_K_pre_rope,
        cos=cos,
        sin=sin,
        rope_contiguous_layout=rope_contiguous_layout,
        rmsnorm_QK_post_rope_enabled=rmsnorm_QK_post_rope_enabled,
        rmsnorm_QK_post_rope_eps=rmsnorm_QK_post_rope_eps,
        rmsnorm_QK_post_rope_W_Q=W_rmsnorm_Q_post_rope,
        rmsnorm_QK_post_rope_W_K=W_rmsnorm_K_post_rope,
        skip_attention=skip_attention,
        K_cache_transposed=K_cache_transposed,
        active_blocks_table=active_blocks_table,
        K_cache=K_cache,
        V_cache=V_cache,
        attention_mask=attention_mask,
        sink=sink,
        softmax_scale=softmax_scale,
        update_cache=update_cache,
        kv_cache_update_idx=kv_cache_update_idx,
        k_scale=k_scale,
        v_scale=v_scale,
        W_out=W_out,
        bias_out=bias_out,
        quantization_type_out=quantization_type_out,
        weight_dequant_scale_out=weight_dequant_scale_out,
        input_dequant_scale_out=input_dequant_scale_out,
        transposed_out=transposed_out,
        out_in_sb=out_in_sb,
        sbm=sbm,
        KVDP=KVDP,
        KVDP_replica_group=KVDP_replica_group,
    )

    assert is_hbm_buffer(K_hbm_out)
    assert is_hbm_buffer(V_hbm_out)

    if not out_in_sb:
        return kernel_output, K_hbm_out, V_hbm_out

    assert kernel_output.buffer == nl.sbuf, "Expecting output on SBUF"

    # copy output to HBM
    skip_output_projection = W_out is None
    if skip_output_projection:
        kernel_output_hbm = nl.ndarray(kernel_output.shape, kernel_output.dtype, nl.hbm, name="kernel_output_hbm")
        nisa.dma_copy(kernel_output_hbm, kernel_output)
    else:
        kernel_output_hbm = relayout_sbuf_to_hbm_for_output_projection(kernel_output, transposed_out, B, S_tkg, H)

    return kernel_output_hbm, K_hbm_out, V_hbm_out


def relayout_sbuf_to_hbm_for_output_projection(kernel_output, transposed_out, B, S_tkg, H):
    # if transposed: SBUF.layout=(PMAX, H // lnc // PMAX, B*S_tkg) and HBM.layout=(PMAX, lnc, H // lnc // PMAX, B*S_tkg)
    # else: SBUF.layout=(B*S_tkg, H // lnc) and HBM.layout=(B*S_tkg, H)

    # Note: this code is based on the output_projection_tkg() logic
    _, n_prgs, prg_id = get_program_sharding_info()
    if transposed_out:
        H0, H1, H2 = n_prgs, nl.tile_size.pmax, H // n_prgs // nl.tile_size.pmax
        kernel_output_hbm = nl.ndarray(
            (H1, H0, H2, B * S_tkg), kernel_output.dtype, nl.shared_hbm, name="kernel_output_hbm"
        )
        nisa.dma_copy(
            dst=kernel_output_hbm.ap(
                pattern=[
                    [H0 * H2 * B * S_tkg, H1],
                    [B * S_tkg, H2],
                    [1, B * S_tkg],
                ],
                offset=prg_id * H2 * B * S_tkg,
            ),
            src=kernel_output,
        )
        return kernel_output_hbm

    # Else, not transposed out
    kernel_output_hbm = nl.ndarray((B * S_tkg, H), kernel_output.dtype, nl.shared_hbm, name="kernel_output_hbm")
    H_sharded = H // n_prgs
    nisa.dma_copy(kernel_output_hbm[:, nl.ds(prg_id * H_sharded, H_sharded)], kernel_output)
    return kernel_output_hbm


# FP8 KV cache validation: cosine similarity catches directional drift from mixed-precision
# (fp8/bf16 kernel vs fp32 golden), while allclose with min_pass_rate catches per-element errors.
# Both are needed because cosine similarity alone misses uniform scaling errors, and allclose
# alone is too strict for the accumulated rounding from FP8 quantization boundaries.
def make_cosine_similarity_validator(
    golden: npt.NDArray[Any], rtol: float, atol: float, min_cosine_similarity: float, min_pass_rate: float, name: str
) -> type[CustomValidator]:
    """Create a validator that checks cosine similarity and allclose with a minimum pass rate."""
    _golden = golden
    _rtol = rtol
    _atol = atol
    _min_cos = min_cosine_similarity
    _min_pass_rate = min_pass_rate
    _name = name
    _shape = golden.shape
    _dtype = golden.dtype

    class CosineValidator(CustomValidator):
        @override
        def validate(self, inference_output: npt.NDArray[Any]) -> bool:
            actual = inference_output.view(_dtype).reshape(_shape).astype(np.float32)
            expected = _golden.astype(np.float32)

            # Cosine similarity on flattened vectors
            a, b = actual.flatten(), expected.flatten()
            cos_sim = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12)

            # Allclose with min_pass_rate
            allclose_pass = maxAllClose(
                actual, expected, rtol=_rtol, atol=_atol, verbose=1, logfile=self.logfile, min_pass_rate=_min_pass_rate
            )

            self._print_with_log(
                f"Validating {_name}: cosine_similarity={cos_sim:.6f} (min={_min_cos}), "
                f"allclose(pass_rate>={_min_pass_rate})={allclose_pass}"
            )
            return cos_sim >= _min_cos and allclose_pass

    return CosineValidator


def _golden_ref_via_torch(kernel_input: dict, lnc: int) -> dict:
    """Compute golden reference using the torch ref, returning numpy arrays in kernel dtypes.

    torch_ref_wrapper upcasts bf16/fp8→f32 for CPU compatibility. We cast back
    to the actual kernel IO dtypes (bf16/fp8)
    """
    torch_ref = AttentionBlockTkgTorchRef(lnc)
    ignored = set(kernel_input) - set(signature(torch_ref).parameters)
    assert not ignored, f"kernel_input keys not consumed by torch ref: {ignored}"
    ref_output = torch_ref_wrapper(torch_ref)(**kernel_input)
    x_dtype = kernel_input['X'].dtype
    kv_dtype = kernel_input['K_cache'].dtype
    output_dtypes = {
        "X_out": x_dtype,
        "K_tkg": kv_dtype,
        "V_tkg": kv_dtype,
        "K_cache_updated": kv_dtype,
        "V_cache_updated": kv_dtype,
    }
    return {k: v.astype(output_dtypes[k]) for k, v in ref_output.items()}


def _infer_output_shapes_and_dtypes(kernel_input: dict, lnc: int) -> dict:
    """Infer output tensor shapes and dtypes by running the torch ref.

    The kernel has many output-shape variants (update_cache, K_cache_transposed,
    out_in_sb, transposed_out, block KV, …). Rather than duplicating that logic
    here — which is brittle and has caused shape mismatches — we run the torch
    ref once and mirror its output shapes.
    """
    golden = _golden_ref_via_torch(kernel_input, lnc)
    return {k: np.zeros(v.shape, dtype=v.dtype) for k, v in golden.items()}


def _get_tolerances(kv_quant: bool, quantization_type: QuantizationType):
    """Return (rtol, atol) based on quantization mode."""
    # FP8 E4M3: 3 mantissa bits, step size 16 at largest binade (128-240).
    if kv_quant or quantization_type != QuantizationType.NONE:
        return 0.07, 16.0
    return 0.015, 1e-5


def _run_attention_block_test(
    test_manager: Orchestrator,
    platform_target: Platforms,
    batch: int,
    q_heads: int,
    d_head: int,
    H: int,
    H_actual: int,
    S_ctx: int,
    S_max_ctx: int,
    S_tkg: int,
    block_len: int,
    update_cache: bool,
    K_cache_transposed: bool,
    rmsnorm_X: bool,
    skip_rope: bool,
    rope_contiguous_layout: bool,
    qk_norm_pre_rope: bool,
    qk_norm_pre_rope_gamma: bool,
    qk_norm_post_rope: bool,
    qk_norm_post_rope_gamma: bool,
    dtype,
    quantization_type: QuantizationType,
    lnc: int,
    transposed_out: bool,
    test_bias: bool,
    input_in_sb: bool,
    output_in_sb: bool,
    softmax_scale,
    kv_quant: bool,
    kv_scale: Optional[Union[KVScaleTest, float]] = None,
    KVDP: int = 1,
    skip_output_projection: bool = False,
    skip_attention: bool = False,
):
    """Shared test execution logic for attention block TKG kernel.

    Single-rank case (KVDP=1):
        Uses UnitTestFramework with torch reference (AttentionBlockTkgTorchRef).

        INPUTS ──┬──> KERNEL ──> X_out, K/V_out ──┐
                 │                                ├─> compare
                 └──> GOLDEN ──> X_out, K/V_out ──┘

    Multi-rank case (KVDP>1):
        Uses manual orchestration (UnitTestFramework doesn't support PerRankLazy*).
        Generate inputs once with total_q_heads = KVDP * q_heads, then slice per-rank.
        This ensures:
          - Shared data is identical across ranks: X, W_k, W_v (GQA has 1 KV head shared by all Q heads)
          - Per-rank data is different: W_q, W_out (by head), K/V cache, mask (by batch)

        For each rank:
                                          ┌─slice K/V[B/KVDP]──> KERNEL ──> X_out, K/V[B/KVDP] ──────┐
                                          │                                    │                     │
        INPUTS ──slice W_q/W_out[q_heads]─┤                                    ├─> compare X_out     ├─> compare K/V
        K/V[B],                           │                                    │                     │
        W_q/W_out[q_heads*KVDP]           └────────────────────> GOLDEN ──> X_out, K/V[B] ──slice──> K/V[B/KVDP]

    Shape changes with KVDP>1 (per rank_id):

        Tensor                  Kernel                              Golden                       Description
        ──────                  ──────                              ──────                       ───────────
        X                       [B, S_tkg, H]                       [B, S_tkg, H]                same, shared across ranks
        W_qkv                   [H, d*(q_heads+2)]                  [H, d*(q_heads+2)]           same, sliced Q per rank, W_k/W_v replicated (GQA)
        W_out                   [q_heads*d, H]                      [q_heads*d, H]               same, sliced Q per rank
        K_cache (flat)          [B/KVDP, 1, S_ctx, d_head]          [B, 1, S_ctx, d_head]        kernel sliced B, golden full
        K_cache (block)         [num_blocks/KVDP, block_len, d]     [num_blocks, block_len, d]   kernel sliced B, golden full
        attention_mask          [S_ctx, B/KVDP, q_heads*KVDP, S]    [S_ctx, B, q_heads, S]       kernel: sliced B, gathered heads

        Note: For block KV indices (active_blocks_table and kv_cache_update_idx)
        we remap global block indices to per-rank local indices in _slice_block_kv_for_all_ranks()
    """
    cfg = AttnBlkTestConfig(
        batch=batch,
        num_heads=q_heads,
        d_head=d_head,
        H=H,
        H_actual=H_actual,
        S_ctx=S_ctx,
        S_max_ctx=S_max_ctx,
        S_tkg=S_tkg,
        block_len=block_len,
        update_cache=update_cache,
        K_cache_transposed=K_cache_transposed,
        rmsnorm_X=rmsnorm_X,
        skip_rope=skip_rope,
        rope_contiguous_layout=rope_contiguous_layout,
        qk_norm_pre_rope=qk_norm_pre_rope,
        qk_norm_pre_rope_gamma=qk_norm_pre_rope_gamma,
        qk_norm_post_rope=qk_norm_post_rope,
        qk_norm_post_rope_gamma=qk_norm_post_rope_gamma,
        dtype=dtype,
        quantization_type=quantization_type,
        lnc=lnc,
        skip_output_projection=skip_output_projection,
        transposed_out=transposed_out,
        test_bias=test_bias,
        out_in_sbuf=output_in_sb,
        input_in_sbuf=input_in_sb,
        softmax_scale=softmax_scale,
        kv_quant=kv_quant,
        kv_scale=kv_scale,
        KVDP=KVDP,
        skip_attention=skip_attention,
    )

    estimated_bytes = estimate_test_memory_bytes(cfg)
    if estimated_bytes > _MAX_MEMORY_BYTES:
        pytest.skip(
            f"Estimated memory {estimated_bytes / 1024**3:.1f} GiB exceeds "
            f"limit {_MAX_MEMORY_BYTES / 1024**3:.1f} GiB "
            f"(set TEST_ATTN_BLK_TKG_MAX_MEMORY_GB to override)"
        )

    rtol, atol = _get_tolerances(kv_quant, quantization_type)

    if cfg.KVDP > 1:
        _run_kvdp_test(test_manager, platform_target, cfg, rtol, atol)
    else:
        _run_single_rank_test(test_manager, platform_target, cfg, rtol, atol)


def _run_single_rank_test(
    test_manager: Orchestrator,
    platform_target: Platforms,
    cfg: AttnBlkTestConfig,
    rtol: float,
    atol: float,
):
    """Run single-rank test using UnitTestFramework with cosine similarity validation."""

    def input_generator(test_config):
        return generate_kernel_inputs(cfg)

    kernel_input = generate_kernel_inputs(cfg)
    golden_outputs = _golden_ref_via_torch(kernel_input, cfg.lnc)

    # Pass rate:
    #   - Non-quantized (bf16): 100% of elements within rtol. The worst-case
    #   - FP8 kv_quant X_out: 99% pass rate. FP8 has coarser quantization
    min_pass_rate_x_out = 0.99 if cfg.kv_quant else 1.0
    custom_validation = ValidationArgs(
        golden_output={
            name: CustomValidatorWithOutputTensorData(
                validator=make_cosine_similarity_validator(
                    golden,
                    rtol=rtol,
                    atol=atol,
                    min_cosine_similarity=0.99,
                    min_pass_rate=min_pass_rate_x_out if name == 'X_out' else 1.0,
                    name=name,
                ),
                output_ndarray=golden,
            )
            for name, golden in golden_outputs.items()
        }
    )

    framework = UnitTestFramework(
        test_manager=test_manager,
        kernel_entry=nki.jit(attention_block_tkg_kernel_test_wrapper),
        torch_ref=torch_ref_wrapper(AttentionBlockTkgTorchRef(cfg.lnc)),
        kernel_input_generator=input_generator,
        output_tensor_descriptor=lambda ki: _infer_output_shapes_and_dtypes(ki, cfg.lnc),
    )
    framework.run_test(
        test_config=None,
        compiler_args=CompilerArgs(logical_nc_config=cfg.lnc, enable_birsim=False, platform_target=platform_target),
        rtol=rtol,
        atol=atol,
        inference_args=replace(TKG_INFERENCE_ARGS, collective_ranks=1, enable_determinism_check=False),
        custom_validation_args=custom_validation,
    )


def _run_kvdp_test(
    test_manager: Orchestrator,
    platform_target: Platforms,
    cfg: AttnBlkTestConfig,
    rtol: float,
    atol: float,
):
    """Run multi-rank KVDP test with manual orchestration.

    UnitTestFramework doesn't support PerRankLazyInputGenerator or
    PerRankLazyGoldenGenerator, which are needed for per-rank input slicing
    and per-rank golden computation. So we call test_manager.execute() directly.
    """
    assert not cfg.kv_quant, "KVDP + kv_quant combination not yet implemented"
    kvdp_cfg = replace(cfg, num_heads=cfg.KVDP * cfg.num_heads)
    kernel_input = generate_kernel_inputs(kvdp_cfg)

    # For KV data parallelism with KVDP ranks, each rank has q_heads, so total = KVDP * q_heads.
    # Generate inputs once with total heads, then slice per-rank.
    B_attn = cfg.batch // cfg.KVDP
    kernel_input, per_rank_cache = _create_per_rank_inputs(
        kernel_input, cfg.KVDP, cfg.num_heads, cfg.d_head, B_attn, cfg.S_tkg, cfg.S_ctx, cfg.block_len
    )

    def create_golden_for_rank(rank_id):
        # Inputs were generated with total_q_heads = KVDP * q_heads.
        # Slice W_qkv, bias_qkv, W_out to this rank's q_heads for golden validation.
        golden_input_copy = kernel_input.base_input.copy()
        golden_input_copy['W_qkv'] = per_rank_cache['w_qkv'][rank_id]
        golden_input_copy['bias_qkv'] = per_rank_cache['bias_qkv'][rank_id]
        golden_input_copy['W_out'] = per_rank_cache['w_out'][rank_id]
        # Golden needs mask with full batch and q_heads (not KVDP * q_heads)
        golden_input_copy['attention_mask'] = per_rank_cache['golden_mask']
        golden_input_copy['KVDP'] = 1  # Torch ref is single-rank; KVDP slicing is done here
        rank_golden = _golden_ref_via_torch(golden_input_copy, cfg.lnc)
        return _slice_golden_KV_cache_for_rank(
            rank_golden, rank_id, B_attn, cfg.block_len, cfg.d_head, cfg.S_ctx, per_rank_cache
        )

    test_manager.execute(
        KernelArgs(
            kernel_func=nki.jit(attention_block_tkg_kernel_test_wrapper),
            compiler_input=CompilerArgs(
                logical_nc_config=cfg.lnc, enable_birsim=False, platform_target=platform_target
            ),
            kernel_input=kernel_input,
            validation_args=ValidationArgs(
                golden_output=PerRankLazyGoldenGenerator(create_golden_for_rank),
                relative_accuracy=rtol,
                absolute_accuracy=atol,
            ),
            inference_args=replace(TKG_INFERENCE_ARGS, collective_ranks=cfg.KVDP, enable_determinism_check=False),
        )
    )


@lru_cache(maxsize=1)
def _get_attention_block_metadata():
    return load_model_configs("test_attention_block")


@pytest_test_metadata(
    name="Attention Block TKG",
    pytest_marks=["attention", "tkg", "experimental"],
    tags=["model"],
)
@final
class TestRangeAttnBlk:
    # fmt: off
    @pytest.mark.parametrize(
            "batch, q_heads, d_head, H    , H_actual, S_ctx  , S_max_ctx, S_tkg, block_len, update_cache, K_cache_transposed, rmsnorm_X, skip_rope, rope_contiguous_layout, qk_norm_pre_rope, qk_norm_pre_rope_gamma, qk_norm_post_rope, qk_norm_post_rope_gamma, dtype      , quantization_type      , lnc, transposed_out, test_bias, input_in_sb, output_in_sb, softmax_scale, kv_quant, kv_scale, KVDP",
        [
            # SBUF IO
            (4    , 8      , 64    , 6144 , 2880    , 11264  , 11264    , 1    , 0        , True        , False             , True     , False    , True                  , False           , False                 , False            , False                  , nl.bfloat16, QuantizationType.NONE  , 2  , False         , False    , False      , True        , None         , False   , None            , 1),
            (4    , 8      , 64    , 6144 , 2880    , 11264  , 11264    , 1    , 0        , True        , False             , True     , False    , True                  , False           , False                 , False            , False                  , nl.bfloat16, QuantizationType.NONE  , 2  , True          , False    , False      , True        , None         , False   , None            , 1),
            (4    , 8      , 64    , 6144 , 2880    , 11264  , 11264    , 2    , 0        , False       , False             , False    , True     , True                  , False           , False                 , False            , False                  , nl.bfloat16, QuantizationType.NONE  , 2  , False         , False    , True       , True        , None         , False   , None            , 1),
            # HBM IO
            (4    , 8      , 64    , 6144 , 2880    , 11264  , 11264    , 1    , 0        , True        , False             , True     , False    , True                  , False           , False                 , False            , False                  , nl.bfloat16, QuantizationType.NONE  , 2  , False         , False    , False      , False       , None         , False   , None            , 1),
            (4    , 8      , 128   , 6144 , 2880    , 11264  , 11264    , 1    , 0        , True        , False             , True     , False    , True                  , True            , False                 , True             , True                   , nl.bfloat16, QuantizationType.NONE  , 2  , False         , True     , False      , False       , None         , False   , None            , 1),
            (4    , 1      , 128   , 6144 , 2880    , 10240  , 10240    , 5    , 0        , False       , False             , True     , False    , True                  , False           , False                 , False            , False                  , nl.bfloat16, QuantizationType.NONE  , 2  , False         , False    , False      , False       , None         , False   , None            , 1),
            # GPT OSS RIV'25
            (8    , 8      , 64    , 6144 , 2880    , 11264  , 11264    , 1    , 0        , True        , False             , True     , False    , True                  , False           , False                 , False            , False                  , nl.bfloat16, QuantizationType.NONE  , 2  , False         , True     , False      , False       , None         , False   , None            , 1),
            (8    , 8      , 64    , 3072 , 2880    , 11264  , 11264    , 5    , 0        , True        , False             , True     , False    , True                  , False           , False                 , False            , False                  , nl.bfloat16, QuantizationType.NONE  , 2  , False         , True     , False      , False       , None         , False   , None            , 1),
            (8    , 8      , 64    , 3072 , 2880    , 10240  , 10240    , 5    , 0        , True        , False             , True     , False    , True                  , False           , False                 , False            , False                  , nl.bfloat16, QuantizationType.NONE  , 2  , False         , True     , False      , False       , None         , False   , None            , 1),
            (4    , 8      , 64    , 3072 , None    , 10240  , 10240    , 4    , 0        , True        , True              , True     , False    , True                  , False           , False                 , False            , False                  , nl.bfloat16, QuantizationType.NONE  , 2  , False         , True     , False      , False       , None         , False   , None            , 1),
            (4    , 8      , 64    , 3072 , 2880    , 10240  , 10240    , 4    , 0        , True        , True              , True     , False    , True                  , False           , False                 , False            , False                  , nl.bfloat16, QuantizationType.NONE  , 2  , False         , True     , False      , False       , None         , False   , None            , 1),
            # Qwen3
            (16   , 1      , 128   , 4096 , None    , 10240  , 10240    , 1    , 0        , True        , False             , True     , False    , True                  , True            , False                 , False            , False                  , nl.bfloat16, QuantizationType.NONE  , 2  , False         , False    , False      , False       , None         , False   , None            , 1),
            # Qwen3 with pre-rope gamma weights
            (16   , 1      , 128   , 4096 , None    , 10240  , 10240    , 1    , 0        , False        , False             , True     , False   , True                  , True            , True                  , False            , False                  , nl.bfloat16, QuantizationType.NONE  , 2  , False         , False    , False      , False       , None         , False   , None            , 1),
            # Gemma3 with pre-rope gamma weights
            (1    , 1      , 128   , 5376 , None    , 1024   , 1024     , 1    , 0        , False        , False             , True     , False    , True                  , True            , True                  , False            , False                  , nl.bfloat16, QuantizationType.NONE  , 2  , False         , False    , False      , False       , None         , False   , None            , 1),
            (8    , 4      , 128   , 5376 , None    , 10240  , 10240    , 3    , 0        , False        , False             , True     , False   , True                  , True            , True                  , False            , False                  , nl.bfloat16, QuantizationType.NONE  , 2  , False         , True     , False      , False       , None         , False   , None            , 1),
            (1    , 1      , 128   , 5376 , None    , 1024   , 1024     , 1    , 16       , False        , False             , True     , False    , True                  , True            , True                  , False            , False                  , nl.bfloat16, QuantizationType.NONE  , 2  , False         , False    , False      , False       , None         , False   , None            , 1),
            # New model, 2025-Jul
            (32   , 1      , 64    , 3072 , None    , 8192   , 8192     , 1    , 0        , True        , False             , False    , False    , True                  , False           , False                 , False            , False                  , nl.bfloat16, QuantizationType.NONE  , 2  , False         , True     , False      , False       , None         , False   , None            , 1),
            (32   , 1      , 64    , 3072 , None    , 8192   , 8192     , 1    , 0        , True        , True              , False    , False    , True                  , False           , False                 , False            , False                  , nl.bfloat16, QuantizationType.NONE  , 2  , False         , True     , False      , False       , None         , False   , None            , 1),
            (64   , 1      , 64    , 3072 , None    , 8192   , 8192     , 1    , 0        , True        , True              , False    , False    , True                  , False           , False                 , False            , False                  , nl.bfloat16, QuantizationType.NONE  , 2  , False         , True     , False      , False       , None         , False   , None            , 1),
            (32   , 1      , 64    , 3072 , None    , 128    , 128      , 1    , 0        , True        , False             , False    , False    , True                  , False           , False                 , False            , False                  , nl.bfloat16, QuantizationType.NONE  , 2  , False         , True     , False      , False       , None         , False   , None            , 1),
            (32   , 1      , 64    , 3072 , None    , 128    , 128      , 1    , 0        , True        , True              , False    , False    , True                  , False           , False                 , False            , False                  , nl.bfloat16, QuantizationType.NONE  , 2  , False         , True     , False      , False       , None         , False   , None            , 1),
            (64   , 1      , 64    , 3072 , None    , 128    , 128      , 1    , 0        , True        , True              , False    , False    , True                  , False           , False                 , False            , False                  , nl.bfloat16, QuantizationType.NONE  , 2  , False         , True     , False      , False       , None         , False   , None            , 1),
            (1    , 1      , 64    , 3072 , None    , 128    , 128      , 1    , 0        , True        , True              , False    , False    , True                  , False           , False                 , False            , False                  , nl.bfloat16, QuantizationType.NONE  , 2  , False         , True     , False      , False       , None         , False   , None            , 1),
            (4    , 8      , 64    , 3072 , None    , 8192   , 8192     , 1    , 0        , True        , True              , False    , False    , True                  , False           , False                 , False            , False                  , nl.bfloat16, QuantizationType.NONE  , 2  , False         , True     , False      , False       , None         , False   , None            , 1),
            (8    , 8      , 64    , 3072 , None    , 8192   , 8192     , 1    , 0        , True        , True              , False    , False    , True                  , False           , False                 , False            , False                  , nl.bfloat16, QuantizationType.NONE  , 2  , False         , True     , False      , False       , None         , False   , None            , 1),
            (16   , 8      , 64    , 3072 , None    , 8192   , 8192     , 1    , 0        , True        , True              , False    , False    , True                  , False           , False                 , False            , False                  , nl.bfloat16, QuantizationType.NONE  , 2  , False         , True     , False      , False       , None         , False   , None            , 1),
            (32   , 1      , 64    , 3072 , None    , 8192   , 8192     , 3    , 0        , True        , True              , False    , False    , True                  , False           , False                 , False            , False                  , nl.bfloat16, QuantizationType.NONE  , 2  , True          , True     , False      , False       , None         , False   , None            , 1),
            (4    , 8      , 64    , 3072 , None    , 8192   , 8192     , 2    , 0        , True        , True              , False    , False    , True                  , False           , False                 , False            , False                  , nl.bfloat16, QuantizationType.NONE  , 2  , True          , True     , False      , False       , None         , False   , None            , 1),
            # secret text
            (1    , 1      , 128   , 7168 , None    , 256    , 256      , 1    , 0        , True        , True              , True     , False    , True                  , False           , False                 , True             , True                   , nl.bfloat16, QuantizationType.NONE  , 2  , False         , False    , False      , False       , None         , False   , None            , 1),
            # llama
            (1    , 1      , 128   , 8192 , None    , 8192   , 8192     , 1    , 0        , True        , True              , False    , False    , True                  , False           , False                 , True             , False                  , nl.bfloat16, QuantizationType.NONE  , 2  , False         , False    , False      , False       , None         , False   , None            , 1),
            (1    , 1      , 128   , 8192 , None    , 8192   , 8192     , 1    , 0        , True        , True              , False    , True     , True                  , False           , False                 , False            , False                  , nl.bfloat16, QuantizationType.NONE  , 2  , False         , False    , False      , False       , None         , False   , None            , 1),
            (1    , 1      , 128   , 8192 , None    , 8192   , 8192     , 1    , 0        , True        , True              , False    , False    , True                  , False           , False                 , False            , False                  , nl.bfloat16, QuantizationType.NONE  , 2  , False         , False    , False      , False       , None         , False   , None            , 1),
            (1    , 1      , 128   , 8192 , None    , 8192   , 8192     , 1    , 0        , True        , True              , False    , False    , False                 , False           , False                 , False            , False                  , nl.bfloat16, QuantizationType.NONE  , 2  , False         , False    , False      , False       , None         , False   , None            , 1),
            (1    , 1      , 128   , 5120 , None    , 8192   , 8192     , 1    , 0        , True        , True              , True     , False    , False                 , False           , False                 , False            , False                  , nl.bfloat16, QuantizationType.NONE  , 2  , False         , False    , False      , False       , None         , False   , None            , 1),
            (1    , 1      , 128   , 5120 , None    , 8192   , 8192     , 1    , 0        , True        , True              , True     , True     , False                 , False           , False                 , False            , False                  , nl.bfloat16, QuantizationType.NONE  , 2  , False         , False    , False      , False       , None         , False   , None            , 1),
            (4    , 1      , 128   , 8192 , None    , 10240  , 16384    , 5    , 0        , True        , True              , True     , False    , True                  , False           , False                 , False            , False                  , nl.bfloat16, QuantizationType.NONE  , 2  , False         , False    , False      , False       , None         , False   , None            , 1),
            (4    , 1      , 128   , 8192 , None    , 10240  , 10240    , 5    , 0        , True        , True              , True     , False    , True                  , False           , False                 , False            , False                  , nl.bfloat16, QuantizationType.NONE  , 2  , False         , False    , False      , False       , None         , False   , None            , 1),
            (8    , 2      , 128   , 16384, None    , 2048   , 2048     , 7    , 0        , True        , True              , True     , False    , True                  , False           , False                 , False            , False                  , nl.bfloat16, QuantizationType.NONE  , 2  , False         , False    , False      , False       , None         , False   , None            , 1),
            # (4,     2,          128,    8192,   None,       16384,  16640,      5,      0,          False,          False,              False,      False,      True,                   False,      False,              False,                      nl.bfloat16,    QuantizationType.NONE, 1,      False,          False,      False,          False,          1, 1),     # FAIL (LNC=1 not yet supported)
            (1    , 16     , 128   , 16384, None    , 4096   , 8192     , 7    , 0        , True        , False             , False    , False    , True                  , False           , False                 , False            , False                  , nl.bfloat16, QuantizationType.NONE  , 2  , False         , False    , False      , False       , None         , False   , None            , 1),
            (1    , 16     , 128   , 16384, None    , 4096   , 8192     , 7    , 0        , True        , False             , False    , False    , False                 , False           , False                 , False            , False                  , nl.bfloat16, QuantizationType.NONE  , 2  , False         , False    , False      , False       , None         , False   , None            , 1),
            (8    , 2      , 128   , 16384, None    , 2048   , 2048     , 7    , 0        , True        , True              , True     , False    , True                  , False           , False                 , False            , False                  , nl.bfloat16, QuantizationType.NONE  , 2  , True          , False    , False      , False       , None         , False   , None            , 1),
            # # Test vectors for block KV
            (4    , 1      , 128   , 8192 , None    , 256    , 256      , 5    , 16       , True        , False             , True     , False    , True                  , False           , False                 , False            , False                  , nl.bfloat16, QuantizationType.NONE  , 2  , False         , False    , False      , False       , None         , False   , None            , 1),
            (4    , 1      , 128   , 8192 , None    , 8192   , 8192     , 5    , 16       , True        , False             , True     , False    , True                  , False           , False                 , False            , False                  , nl.bfloat16, QuantizationType.NONE  , 2  , False         , False    , False      , False       , None         , False   , None            , 1),
            (4    , 1      , 128   , 8192 , None    , 12288  , 12288    , 5    , 16       , True        , False             , True     , False    , True                  , False           , False                 , False            , False                  , nl.bfloat16, QuantizationType.NONE  , 2  , False         , False    , False      , False       , None         , False   , None            , 1),
            (4    , 1      , 128   , 8192 , None    , 10240  , 10240    , 5    , 16       , True        , False             , True     , False    , True                  , False           , False                 , False            , False                  , nl.bfloat16, QuantizationType.NONE  , 2  , False         , False    , False      , False       , None         , False   , None            , 1),
            # Test vectors to verify functionality of different q_heads, d_head and H dimensions
            (2    , 1      , 128   , 2048 , None    , 10240  , 16384    , 5    , 0        , True        , True              , True     , False    , True                  , False           , False                 , False            , False                  , nl.bfloat16, QuantizationType.NONE  , 2  , False         , False    , False      , False       , None         , False   , None            , 1),
            (2    , 1      , 64    , 2048 , None    , 10240  , 16384    , 5    , 0        , True        , True              , True     , False    , True                  , False           , False                 , False            , False                  , nl.bfloat16, QuantizationType.NONE  , 2  , False         , False    , False      , False       , None         , False   , None            , 1),
            (2    , 2      , 64    , 3072 , None    , 10240  , 16384    , 5    , 0        , True        , False             , True     , False    , True                  , False           , False                 , False            , False                  , nl.bfloat16, QuantizationType.NONE  , 2  , False         , False    , False      , False       , None         , False   , None            , 1),
            (2    , 3      , 64    , 4096 , None    , 10240  , 16384    , 5    , 0        , False       , False             , True     , False    , True                  , False           , False                 , False            , False                  , nl.bfloat16, QuantizationType.NONE  , 2  , False         , False    , False      , False       , None         , False   , None            , 1),
            (2    , 4      , 128   , 6144 , None    , 10240  , 16384    , 5    , 0        , False       , True              , True     , False    , True                  , False           , False                 , False            , False                  , nl.bfloat16, QuantizationType.NONE  , 2  , False         , False    , False      , False       , None         , False   , None            , 1),
            (2    , 3      , 128   , 20480, None    , 10240  , 16384    , 5    , 0        , True        , True              , True     , False    , True                  , False           , False                 , False            , False                  , nl.bfloat16, QuantizationType.NONE  , 2  , False         , False    , False      , False       , None         , False   , None            , 1),
            (4    , 1      , 128   , 8192 , None    , 10240  , 10240    , 5    , 0        , True        , True              , True     , False    , True                  , False           , False                 , False            , False                  , nl.bfloat16, QuantizationType.NONE  , 2  , False         , False    , False      , False       , None         , False   , None            , 1),
            # static quantization tests 
            # TODO: random input causing numerical instability for quantized weights, more tests will be added
            # after better fp8 random generator is implemented
            # E2E inference tests shows good accuracy
            (8    , 1      , 128   , 8192 , None    , 2048   , 2048     , 5    , 0        , True        , True              , True     , False    , True                  , False           , False                 , False            , False                  , nl.bfloat16, QuantizationType.STATIC, 2  , False         , False    , False      , False       , None         , False   , None            , 1),

            # softmax_scale tests (Gemma model support)
            (4    , 8      , 64    , 3072 , 2880    , 10240  , 10240    , 4    , 0        , True        , True              , True     , False    , True                  , False           , False                 , False            , False                  , nl.bfloat16, QuantizationType.NONE  , 2  , False         , True     , False      , False       , 0.05         , False   , None            , 1),
            (1    , 1      , 128   , 7168 , None    , 256    , 256      , 1    , 0        , True        , True              , True     , False    , True                  , False           , False                 , True             , True                   , nl.bfloat16, QuantizationType.NONE  , 2  , False         , False    , False      , False       , 0.09         , False   , None            , 1),
            (1    , 1      , 128   , 5120 , None    , 8192   , 8192     , 1    , 0        , True        , True              , True     , True     , False                 , False           , False                 , False            , False                  , nl.bfloat16, QuantizationType.NONE  , 2  , False         , False    , False      , False       , 0.13         , False   , None            , 1),
            (4    , 1      , 128   , 8192 , None    , 8192   , 8192     , 5    , 16       , True        , False             , True     , False    , True                  , False           , False                 , False            , False                  , nl.bfloat16, QuantizationType.NONE  , 2  , False         , False    , False      , False       , 0.17         , False   , None            , 1),
            (4    , 1      , 128   , 8192 , None    , 10240  , 10240    , 5    , 0        , True        , True              , True     , False    , True                  , False           , False                 , False            , False                  , nl.bfloat16, QuantizationType.NONE  , 2  , False         , False    , False      , False       , 0.21         , False   , None            , 1),
            # llama FP8 KV Cache Tests
            (2    , 1      , 128   , 8192 , None    , 8192   , 8192     , 1    , 0        , True        , False             , False    , False    , True                  , True           , True                 , False            , False                  , nl.bfloat16, QuantizationType.NONE  , 2  , False         , False    , False      , False       , None         , True    , KVScaleTest.DEFAULT, 1),
            (37   , 1      , 128   , 8192 , None    , 8192   , 8192     , 1    , 0        , True        , True              , False    , False    , True                  , False           , False                 , False            , False                  , nl.bfloat16, QuantizationType.NONE  , 2  , False         , False    , False      , False       , None         , True    , KVScaleTest.DEFAULT, 1),
            (96   , 1      , 128   , 8192 , None    , 8192   , 8192     , 1    , 0        , True        , True              , False    , False    , True                  , False           , False                 , False            , False                  , nl.bfloat16, QuantizationType.NONE  , 2  , False         , False    , False      , False       , None         , True    , KVScaleTest.DEFAULT, 1),
            # llama FP8 KV Cache Tests - batched cache update
            (32   , 1      , 128   , 8192 , None    , 4096   , 4096     , 1    , 0        , True        , False             , False    , False    , True                  , False           , False                 , False            , False                  , nl.bfloat16, QuantizationType.NONE  , 2  , False         , False    , False      , False       , None         , True    , KVScaleTest.DEFAULT, 1),
            (128  , 1      , 128   , 8192 , None    , 2048   , 2048     , 1    , 0        , True        , False             , False    , False    , True                  , False           , False                 , False            , False                  , nl.bfloat16, QuantizationType.NONE  , 2  , False         , False    , False      , False       , None         , True    , KVScaleTest.DEFAULT, 1),
            # llama FP8 KV Cache Tests - block KV cache
            (16   , 1      , 128   , 8192 , None    , 2048   , 2048     , 1    , 16       , True        , False             , False    , False    , True                  , False           , False                 , False            , False                  , nl.bfloat16, QuantizationType.NONE  , 2  , False         , False    , False      , False       , None         , True    , KVScaleTest.DEFAULT, 1),
            (32   , 1      , 128   , 8192 , None    , 2048   , 2048     , 1    , 16       , True        , False             , False    , False    , True                  , False           , False                 , False            , False                  , nl.bfloat16, QuantizationType.NONE  , 2  , False         , False    , False      , False       , None         , True    , KVScaleTest.DEFAULT, 1),
            (64   , 1      , 128   , 8192 , None    , 2048   , 2048     , 1    , 16       , True        , False             , False    , False    , True                  , False           , False                 , False            , False                  , nl.bfloat16, QuantizationType.NONE  , 2  , False         , False    , False      , False       , None         , True    , KVScaleTest.DEFAULT, 1),
            # FP8 KV cache direct cast (kv_scale=1.0)
            (1    , 2      , 128   , 8192 , None    , 26624  , 36896    , 5    , 32       , True        , False             , True     , False    , True                  , False           , False                 , False            , False                  , nl.bfloat16, QuantizationType.NONE  , 2  , False         , False    , False      , False       , None         , True    , 1.0                , 1),
            # Long context tests (S_ctx >= 128k, slower)
            # flat KV S_ctx=128k
            (8    , 1      , 64    , 3072 , 2880    , 131072 , 131072   , 1    , 0        , True        , False             , False    , False    , True                  , False           , False                 , False            , False                  , nl.bfloat16, QuantizationType.NONE  , 2  , False         , True     , False      , False       , None         , False   , None            , 1),
            # flat KV S_ctx=512k
            (8    , 1      , 64    , 3072 , 2880    , 524288 , 524288   , 1    , 0        , True        , False             , False    , False    , True                  , False           , False                 , False            , False                  , nl.bfloat16, QuantizationType.NONE  , 2  , False         , True     , False      , False       , None         , False   , None            , 1),
            # block KV S_ctx=128k, block_len=32
            (8    , 1      , 64    , 3072 , 2880    , 131072 , 131072   , 1    , 32       , True        , False             , False    , False    , True                  , False           , False                 , False            , False                  , nl.bfloat16, QuantizationType.NONE  , 2  , False         , True     , False      , False       , None         , False   , None            , 1),
            # block KV S_ctx=512k, block_len=32
            (8    , 1      , 64    , 3072 , 2880    , 524288 , 524288   , 1    , 32       , True        , False             , False    , False    , True                  , False           , False                 , False            , False                  , nl.bfloat16, QuantizationType.NONE  , 2  , False         , True     , False      , False       , None         , False   , None            , 1),

            # KVDP tests (KVDP=4, GPT-OSS-like)
            # q_heads=1 means each rank has 1 q_head, total KVDP * q_heads across ranks
            # flat KV S_ctx=1k B=8
            (8    , 1      , 64    , 3072 , 2880    , 1024   , 1024     , 1    , 0        , True        , False             , True     , False    , True                  , False           , False                 , False            , False                  , nl.bfloat16, QuantizationType.NONE  , 2  , False         , True     , False      , False       , None         , False   , None            , 4),
            # q_heads=2 tests the general transpose path (q_heads>1)
            (8    , 2      , 64    , 3072 , 2880    , 1024   , 1024     , 1    , 0        , True        , False             , True     , False    , True                  , False           , False                 , False            , False                  , nl.bfloat16, QuantizationType.NONE  , 2  , False         , True     , False      , False       , None         , False   , None            , 4),
            # block KV S_ctx=1k B=8, block_len=32
            (8    , 1      , 64    , 3072 , 2880    , 1024   , 1024     , 1    , 32       , True        , False             , False    , False    , True                  , False           , False                 , False            , False                  , nl.bfloat16, QuantizationType.NONE  , 2  , False         , True     , False      , False       , None         , False   , None            , 4),
            # block KV S_ctx=1k B=32, block_len=32
            (32   , 1      , 64    , 3072 , 2880    , 1024   , 1024     , 1    , 32       , True        , False             , False    , False    , True                  , False           , False                 , False            , False                  , nl.bfloat16, QuantizationType.NONE  , 2  , False         , True     , False      , False       , None         , False   , None            , 4),

            # KVDP long context tests (S_ctx >= 128k, slower)
            # flat KV S_ctx=512k
            (8    , 1      , 64    , 3072 , 2880    , 524288 , 524288   , 1    , 0        , True        , False             , False    , False    , True                  , False           , False                 , False            , False                  , nl.bfloat16, QuantizationType.NONE  , 2  , False         , True     , False      , False       , None         , False   , None            , 4),
            # block KV S_ctx=512k, block_len=32
            (8    , 1      , 64    , 3072 , 2880    , 524288 , 524288   , 1    , 32       , True        , False             , False    , False    , True                  , False           , False                 , False            , False                  , nl.bfloat16, QuantizationType.NONE  , 2  , False         , True     , False      , False       , None         , False   , None            , 4),
            # flat KV S_ctx=1M
            (8    , 1      , 64    , 3072 , 2880    , 1048576, 1048576  , 1    , 0        , True        , False             , False    , False    , True                  , False           , False                 , False            , False                  , nl.bfloat16, QuantizationType.NONE  , 2  , False         , True     , False      , False       , None         , False   , None            , 4),
            # block KV S_ctx=1M, block_len=32
            (8    , 1      , 64    , 3072 , 2880    , 1048576, 1048576  , 1    , 32       , True        , False             , False    , False    , True                  , False           , False                 , False            , False                  , nl.bfloat16, QuantizationType.NONE  , 2  , False         , True     , False      , False       , None         , False   , None            , 4),

            # KVDP B=64 tests
            # block KV S_ctx=1k B=64, block_len=32
            (64   , 1      , 64    , 3072 , 2880    , 1024   , 1024     , 1    , 32       , True        , False             , False    , False    , True                  , False           , False                 , False            , False                  , nl.bfloat16, QuantizationType.NONE  , 2  , False         , True     , False      , False       , None         , False   , None            , 4),
            # block KV S_ctx=128k B=64, block_len=32
            (64   , 1      , 64    , 3072 , 2880    , 131072 , 131072   , 1    , 32       , True        , False             , False    , False    , True                  , False           , False                 , False            , False                  , nl.bfloat16, QuantizationType.NONE  , 2  , False         , True     , False      , False       , None         , False   , None            , 4),
            # flat KV S_ctx=128k B=64
            (64   , 1      , 64    , 3072 , 2880    , 131072 , 131072   , 1    , 0        , True        , False             , False    , False    , True                  , False           , False                 , False            , False                  , nl.bfloat16, QuantizationType.NONE  , 2  , False         , True     , False      , False       , None         , False   , None            , 4),
        ],
    # fmt: on
    )
    def test_attn_blk_megakernel(
        self,
        test_manager: Orchestrator,
        platform_target: Platforms,
        batch: int,
        q_heads: int,
        d_head: int,
        H: int,
        H_actual: int,
        S_ctx: int,
        S_max_ctx: int,
        S_tkg: int,
        block_len: int,
        update_cache: bool,
        K_cache_transposed: bool,
        rmsnorm_X: bool,
        skip_rope: bool,
        rope_contiguous_layout: bool,
        qk_norm_pre_rope: bool,
        qk_norm_pre_rope_gamma: bool,
        qk_norm_post_rope: bool,
        qk_norm_post_rope_gamma: bool,
        dtype,
        quantization_type: QuantizationType,
        lnc: int,
        transposed_out: bool,
        test_bias: bool,
        input_in_sb: bool,
        output_in_sb: bool,
        softmax_scale,
        kv_quant: bool,
        kv_scale: Optional[Union[KVScaleTest, float]],
        KVDP: int,
        skip_output_projection: bool = False,
        skip_attention: bool = False,
    ):
        _run_attention_block_test(
            test_manager=test_manager,
            platform_target=platform_target,
            batch=batch,
            q_heads=q_heads,
            d_head=d_head,
            H=H,
            H_actual=H_actual,
            S_ctx=S_ctx,
            S_max_ctx=S_max_ctx,
            S_tkg=S_tkg,
            block_len=block_len,
            update_cache=update_cache,
            K_cache_transposed=K_cache_transposed,
            rmsnorm_X=rmsnorm_X,
            skip_rope=skip_rope,
            rope_contiguous_layout=rope_contiguous_layout,
            qk_norm_pre_rope=qk_norm_pre_rope,
            qk_norm_pre_rope_gamma=qk_norm_pre_rope_gamma,
            qk_norm_post_rope=qk_norm_post_rope,
            qk_norm_post_rope_gamma=qk_norm_post_rope_gamma,
            dtype=dtype,
            quantization_type=quantization_type,
            lnc=lnc,
            transposed_out=transposed_out,
            test_bias=test_bias,
            input_in_sb=input_in_sb,
            output_in_sb=output_in_sb,
            softmax_scale=softmax_scale,
            kv_quant=kv_quant,
            kv_scale=kv_scale,
            KVDP=KVDP,
            skip_output_projection=skip_output_projection,
            skip_attention=skip_attention,
        )

    # Pre-generate test IDs with MODEL_WIP prefix for pytest -k filtering
    ATTENTION_BLOCK_TKG_MODEL_TEST_IDS = [
        f"{MODEL_TEST_TYPE}_" + "-".join(str(p.value) if hasattr(p, 'value') else str(p) for p in params)
        for params in attention_block_tkg_model_configs
    ]

    @pytest.mark.parametrize(
        "batch, num_heads, d_head, H, H_actual, S_ctx, S_max_ctx, S_tkg, block_len, update_cache, K_cache_transposed, rmsnorm_X, skip_rope, rope_contiguous_layout, qk_norm_pre_rope, qk_norm_pre_rope_gamma, qk_norm_post_rope, qk_norm_post_rope_gamma, dtype, quantization_type, lnc, transposed_out, test_bias, input_in_sb, output_in_sb, softmax_scale, kv_quant, KVDP",
        attention_block_tkg_model_configs,
        ids=ATTENTION_BLOCK_TKG_MODEL_TEST_IDS,
    )
    def test_attn_blk_model(
        self,
        test_manager: Orchestrator,
        collector: IMetricsCollector,
        platform_target: Platforms,
        request,
        batch: int,
        num_heads: int,
        d_head: int,
        H: int,
        H_actual: int,
        S_ctx: int,
        S_max_ctx: int,
        S_tkg: int,
        block_len: int,
        update_cache: bool,
        K_cache_transposed: bool,
        rmsnorm_X: bool,
        skip_rope: bool,
        rope_contiguous_layout: bool,
        qk_norm_pre_rope: bool,
        qk_norm_pre_rope_gamma: bool,
        qk_norm_post_rope: bool,
        qk_norm_post_rope_gamma: bool,
        dtype,
        quantization_type: QuantizationType,
        lnc: int,
        transposed_out: bool,
        test_bias: bool,
        input_in_sb: bool,
        output_in_sb: bool,
        softmax_scale,
        kv_quant: bool,
        KVDP: int,
    ):
        """Test Attention Block TKG kernel with model configurations (weekly regression)."""
        attn_blk_metadata_list = _get_attention_block_metadata()

        # Add metadata dimensions for model coverage tracking
        test_metadata_key = {
            "batch": batch,
            "num_heads": num_heads,
            "d_head": d_head,
            "H": H,
            "S_ctx": S_ctx,
            "S_tkg": S_tkg,
        }
        collector.match_and_add_metadata_dimensions(test_metadata_key, attn_blk_metadata_list)

        _run_attention_block_test(
            test_manager=test_manager,
            platform_target=platform_target,
            batch=batch,
            q_heads=num_heads,
            d_head=d_head,
            H=H,
            H_actual=H_actual,
            S_ctx=S_ctx,
            S_max_ctx=S_max_ctx,
            S_tkg=S_tkg,
            block_len=block_len,
            update_cache=update_cache,
            K_cache_transposed=K_cache_transposed,
            rmsnorm_X=rmsnorm_X,
            skip_rope=skip_rope,
            rope_contiguous_layout=rope_contiguous_layout,
            qk_norm_pre_rope=qk_norm_pre_rope,
            qk_norm_pre_rope_gamma=qk_norm_pre_rope_gamma,
            qk_norm_post_rope=qk_norm_post_rope,
            qk_norm_post_rope_gamma=qk_norm_post_rope_gamma,
            dtype=dtype,
            quantization_type=quantization_type,
            lnc=lnc,
            transposed_out=transposed_out,
            test_bias=test_bias,
            input_in_sb=input_in_sb,
            output_in_sb=output_in_sb,
            softmax_scale=softmax_scale,
            kv_quant=kv_quant,
            KVDP=KVDP,
        )
