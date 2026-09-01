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
import os
from dataclasses import replace
from functools import lru_cache
from inspect import signature
from typing import Any, Optional, final

import neuron_dtypes as dt
import nki
import nki.isa as nisa
import nki.language as nl
import numpy as np
import numpy.typing as npt
import pytest
from nki.collectives import ReplicaGroup
from nkilib_src.nkilib.core.attention.attention_tkg import INACTIVE_BLOCK_IDX
from nkilib_src.nkilib.core.attention.attention_tkg_utils import is_qk_swapped
from nkilib_src.nkilib.core.attention.gen_mask_tkg_torch import gen_mask_tkg_hbm_torch_ref
from nkilib_src.nkilib.core.utils.allocator import SbufManager, sizeinbytes
from nkilib_src.nkilib.core.utils.common_types import DtypeMode, QuantizationType
from nkilib_src.nkilib.core.utils.kernel_helpers import (
    get_max_positive_value_for_dtype,
    get_program_sharding_info,
    is_hbm_buffer,
)
from nkilib_src.nkilib.experimental.transformer.attention_block_tkg import (
    attention_block_tkg,
)
from nkilib_src.nkilib.experimental.transformer.attention_block_tkg_sharding import CPCollectiveMode, KVDPCollectiveMode
from nkilib_src.nkilib.experimental.transformer.attention_block_tkg_torch import (
    AttentionBlockTkgTorchRef,
)
from typing_extensions import override

try:
    from test.integration.nkilib.experimental.transformer.test_attention_block_tkg_model_config import (
        attention_block_tkg_model_configs,
    )
except ImportError:
    attention_block_tkg_model_configs = {}

from test.integration.nkilib.experimental.transformer.test_attention_block_tkg_utils import (
    AttnBlkTestConfig,
    KVScaleTest,
)
from test.integration.nkilib.utils.tensor_generators import (
    generate_stabilized_mx_data,
    np_random_sample_static_quantize_inp,
)
from test.utils.common_dataclasses import (
    TKG_INFERENCE_ARGS,
    CompilerArgs,
    CustomValidator,
    CustomValidatorWithOutputTensorData,
    ModelTestType,
    Platforms,
    ValidationArgs,
)
from test.utils.comparators import maxAllClose
from test.utils.metadata_loader import load_model_configs
from test.utils.metrics_collector import IMetricsCollector
from test.utils.pytest_test_metadata import pytest_marks, pytest_test_metadata
from test.utils.test_orchestrator import Orchestrator
from test.utils.unit_test_collective_framework import CollectiveUnitTestFramework
from test.utils.unit_test_framework import UnitTestFramework, torch_ref_wrapper

# Maximum memory (GB) for test tensor allocation. Tests exceeding this are skipped.
# Override via environment variable TEST_ATTN_BLK_TKG_MAX_MEMORY_GB.
_DEFAULT_MAX_MEMORY_GB = 20
_MAX_MEMORY_BYTES = int(float(os.environ.get("TEST_ATTN_BLK_TKG_MAX_MEMORY_GB", _DEFAULT_MAX_MEMORY_GB)) * 1024**3)

_P_MAX = 128  # Partition dimension size (nl.tile_size.pmax)
_FP8_FN_MAX = 448.0  # max representable value for float8_e4m3fn (used by MX on TRN3)


def _generate_mx_weights_and_scales(
    quantization_type, weight_shape, mx_scale_reshape, w_scale_shape, static_mx_in_shape, rng
):
    """Generate MX-quantized weights and dequantization scales.

    Args:
        quantization_type: One of MX, STATIC_MX, ROW_MX.
        weight_shape: Shape for generate_stabilized_mx_data (already divided by q_width).
        mx_scale_reshape: Reshape dims for MX block weight scales.
        w_scale_shape: Shape for STATIC_MX/ROW_MX jittered weight scales.
        static_mx_in_shape: Shape for STATIC_MX input scale jitter, or None to skip.
        rng: numpy random state.

    Returns:
        (weights, weight_scale, input_scale) where input_scale may be None.

    For STATIC_MX and ROW_MX, scales are jittered with a small random
    perturbation around the base value so that tests exercise non-uniform
    scale values and don't accidentally pass with a degenerate constant scale.
    """
    _, weights, weight_scale = generate_stabilized_mx_data(nl.float8_e4m3fn_x4, weight_shape, val_range=5)
    input_scale = None
    if quantization_type == QuantizationType.MX:
        weight_scale = weight_scale.reshape(mx_scale_reshape)
    elif quantization_type == QuantizationType.STATIC_MX:
        base_scale = 1.0 / _FP8_FN_MAX
        weight_scale = (base_scale * rng.uniform(0.995, 1.005, w_scale_shape)).astype(np.float32)
        if static_mx_in_shape is not None:
            input_scale = (base_scale * rng.uniform(0.995, 1.005, static_mx_in_shape)).astype(np.float32)
    elif quantization_type == QuantizationType.ROW_MX:
        base_scale = 1.0 / _FP8_FN_MAX
        weight_scale = (base_scale * rng.uniform(0.995, 1.005, w_scale_shape)).astype(np.float32)
    return weights, weight_scale, input_scale


def _cp_owning_rank(position: int, interleave_size: int, cp_world_size: int) -> int:
    """Return the CP rank that stores the token at the given sequence position.

    Implements the vLLM CP slot mapping formula:
        owner = (position // interleave_size) % cp_world_size

    interleave_size controls the assignment granularity:
      interleave_size=32 (=block_size), cp_world_size=4:
        positions 0..31 -> rank 0, 32..63 -> rank 1, etc.
      interleave_size=1, cp_world_size=4:
        position 0 -> rank 0, 1 -> rank 1, 2 -> rank 2, 3 -> rank 3, 4 -> rank 0, ...
    """
    return (position // interleave_size) % cp_world_size


def _generate_cp_cache_lens(B_attn: int, cp_world_size: int, interleave_size: int, s_max: int) -> np.ndarray:
    """Per-batch cache_len for CP tests, spreading active tokens across all ranks.

    Each batch has one active token, whose sequence position determines its owning CP rank
    (owner = (position // interleave_size) % cp_world_size). We choose each batch's cache_len so
    these active tokens fall across all ranks, for test coverage.

    Args:
        B_attn: Per-rank batch size.
        cp_world_size: CP degree.
        interleave_size: Token-ownership granularity (see _cp_owning_rank).
        s_max: Exclusive upper bound on the active-token position (S_ctx - S_tkg).

    Returns:
        cache_len: [B_attn, 1] int64 append positions, one per batch element.
    """
    if B_attn >= cp_world_size:
        ranks = np.arange(cp_world_size)
        ranks = np.concatenate([ranks, np.random.randint(0, cp_world_size, size=B_attn - cp_world_size)])
        np.random.shuffle(ranks)
    else:
        ranks = np.random.choice(cp_world_size, size=B_attn, replace=False)
    cache_len = np.empty(B_attn, dtype=np.int64)
    for b in range(B_attn):
        first_owned = ranks[b] * interleave_size
        group_stride = cp_world_size * interleave_size
        group_starts = np.arange(first_owned, s_max, group_stride)
        grp = group_starts[np.random.randint(0, len(group_starts))]
        cache_len[b] = grp + np.random.randint(0, min(interleave_size, s_max - grp))
    return cache_len[:, np.newaxis].astype(np.int64)


def _cp_owned_global_positions(cp_rank: int, S_ctx: int, interleave_size: int, cp_world_size: int) -> list:
    """Global sequence positions owned by ``cp_rank`` under the CP slot mapping, in ascending order.

    A position ``p`` is owned by ``(p // interleave_size) % cp_world_size``. With
    ``interleave_size == block_len`` this reduces to whole-block round-robin; with
    ``interleave_size == 1`` it stripes individual tokens across ranks. The returned list has
    length ``S_ctx // cp_world_size`` and the ranks together partition ``range(S_ctx)``; the
    index of a position within this list is its rank-local sequence position.
    """
    return [p for p in range(S_ctx) if _cp_owning_rank(p, interleave_size, cp_world_size) == cp_rank]


def _flatten_block_kv_mask(full_mask, block_len, lnc):
    """Undo the block-KV mask reshape to a flat ``[B, H, S_tkg, S_ctx]`` view.

    The kernel consumes the attention mask in block-KV layout (the reshape+swapaxes applied
    in ``generate_kernel_inputs``). To slice/gather it per CP rank we first undo that reshape
    back to a flat, sequence-position-ordered mask. Inverse of the reshape done by
    :func:`_reblock_cp_mask`.

    Args:
        full_mask: shared mask ``[S_ctx, B, H, S_tkg]`` (as stored in ``shared_input``).
    Returns:
        Flat mask ``[B, H, S_tkg, S_ctx]``.
    """
    from nkilib_src.nkilib.core.attention.attention_tkg_utils import is_s_prior_sharded as _is_sps
    from nkilib_src.nkilib.core.attention.attention_tkg_utils import (
        resize_cache_block_len_for_attention_tkg_kernel,
    )

    B, H, S_tkg, S_ctx = full_mask.shape[1], full_mask.shape[2], full_mask.shape[3], full_mask.shape[0]
    num_blocks = S_ctx // block_len
    m = np.asarray(full_mask).transpose(1, 2, 3, 0)  # [B, H, S_tkg, S_ctx]
    n_prgs = lnc if lnc > 1 and _is_sps(B, H, S_tkg, S_ctx, 128) else 1
    reduced_blk_len, _ = resize_cache_block_len_for_attention_tkg_kernel(
        num_blocks, block_len, lnc, 128, bs=B, q_head=H, s_active=S_tkg, full_sprior=S_ctx
    )
    m = m.reshape(B, H, S_tkg, n_prgs, -1, reduced_blk_len, 128)
    m = np.swapaxes(m, -1, -2)
    return m.reshape(B, H, S_tkg, S_ctx)


def _reblock_cp_mask(rank_mask_flat, block_len, lnc):
    """Re-apply the block-KV mask reshape to a rank's flat mask -> ``[S_local, B, H, S_tkg]``.

    Given a rank's flat mask ``[B, H, S_tkg, S_local]`` (owned positions in rank-local order),
    reproduce the block-KV layout the kernel expects for the rank's local ``S_local`` shard, so
    the mask's within-block ordering matches the rank's local KV cache. Inverse of
    :func:`_flatten_block_kv_mask`; shared by the whole-block and sub-block CP paths.
    """
    from nkilib_src.nkilib.core.attention.attention_tkg_utils import is_s_prior_sharded as _is_sps
    from nkilib_src.nkilib.core.attention.attention_tkg_utils import (
        resize_cache_block_len_for_attention_tkg_kernel,
    )

    B, H, S_tkg, S_local = rank_mask_flat.shape
    num_blocks_local = S_local // block_len
    n_prgs = lnc if lnc > 1 and _is_sps(B, H, S_tkg, S_local, 128) else 1
    reduced_blk_len, _ = resize_cache_block_len_for_attention_tkg_kernel(
        num_blocks_local, block_len, lnc, 128, bs=B, q_head=H, s_active=S_tkg, full_sprior=S_local
    )
    m = rank_mask_flat.reshape(B, H, S_tkg, n_prgs, -1, 128, reduced_blk_len)
    m = np.swapaxes(m, -1, -2)
    return np.ascontiguousarray(m.reshape(B, H, S_tkg, S_local).transpose(3, 0, 1, 2))


def _build_cp_block_kv_rank_inputs(
    shared_input, cp_rank, CP, block_len, S_ctx, S_tkg, B_attn, interleave_size, lnc, kv_heads=1
):
    """Build one CP rank's block-KV cache, table, mask, and update index by gathering owned tokens.

    Used when ``interleave_size < block_len`` (per-token / sub-block striping), where a rank's
    owned tokens are scattered across physical blocks, so the rank's local cache must be gathered
    rather than block-sliced. Each rank owns ``S_local = S_ctx // CP`` global positions
    (see :func:`_cp_owned_global_positions`); position ``owned[lp]`` maps to rank-local position
    ``lp``, packed into ``num_blocks_local = S_local // block_len`` contiguous local blocks.

    Returns a dict with keys ``K_cache``, ``V_cache``, ``active_blocks_table``,
    ``attention_mask``, ``kv_cache_update_idx`` for this rank.

    NOTE: ``kv_heads > 1`` is not yet supported on this per-token-interleave gather path. The 3D
    ``[B, kv_heads, num_blocks]`` table and the per-head shared block pool would each need a
    per-head gather; this is deferred. The whole-block CP path (``interleave_size == block_len``)
    does support ``kv_heads > 1``.
    """
    if kv_heads > 1:
        pytest.skip(
            f"CP per-token-interleave (cp_interleave_size={interleave_size} < block_len={block_len}) "
            f"with kv_heads={kv_heads} is not yet supported by the test harness; use the whole-block "
            "CP path (cp_interleave_size == block_len) for CP x kv_heads."
        )
    S_local = S_ctx // CP
    num_blocks_local = S_local // block_len
    owned = _cp_owned_global_positions(cp_rank, S_ctx, interleave_size, CP)
    assert len(owned) == S_local, f"rank {cp_rank} owns {len(owned)} != {S_local}"

    full_K = np.asarray(shared_input['K_cache'])  # [num_blocks, block_len, d_head] (+ optional kv-head axis)
    full_V = np.asarray(shared_input['V_cache'])
    full_table = np.asarray(shared_input['active_blocks_table'])  # [B_attn, S_ctx // block_len]
    has_kv_head = full_K.ndim == 4

    # Fresh per-rank block pool: B_attn * num_blocks_local local blocks, local table is identity.
    local_blocks = B_attn * num_blocks_local
    K_local = np.zeros((local_blocks,) + full_K.shape[1:], dtype=full_K.dtype)
    V_local = np.zeros((local_blocks,) + full_V.shape[1:], dtype=full_V.dtype)
    local_table = np.full((B_attn, num_blocks_local), INACTIVE_BLOCK_IDX, dtype=np.int32)

    # Gather the rank's owned tokens (scattered across physical blocks under sub-block interleave)
    # into contiguous, ascending-order local blocks: owned[lp] -> local position lp. This matches
    # the serving stack's slot mapping, which packs a rank's owned tokens into dense local slots
    # via block_offset = (vbo // (CP * I)) * I + (vbo % I) (see compute_slot_mapping in
    # neuron_model_runner.py). The kernel thus sees a normal dense block cache (no interleaving
    # inside a local block) and needs no CP-awareness; softmax is permutation-invariant over KV
    # positions, so any consistent (cache, mask) ordering is valid as long as the two agree.
    for b in range(B_attn):
        for local_blk in range(num_blocks_local):
            local_table[b, local_blk] = b * num_blocks_local + local_blk
        for lp, gp in enumerate(owned):
            logical_blk = gp // block_len
            phys_blk = int(full_table[b, logical_blk])
            if phys_blk == INACTIVE_BLOCK_IDX:
                continue
            slot = gp % block_len
            lphys = b * num_blocks_local + lp // block_len
            lslot = lp % block_len
            if has_kv_head:
                K_local[lphys, :, lslot] = full_K[phys_blk, :, slot]
                V_local[lphys, :, lslot] = full_V[phys_blk, :, slot]
            else:
                K_local[lphys, lslot] = full_K[phys_blk, slot]
                V_local[lphys, lslot] = full_V[phys_blk, slot]

    # Mask: select this rank's owned global positions (ascending == local order) into a flat
    # [B, H, S_tkg, S_local] view, then re-apply the block-KV reshape the kernel expects (shared
    # with the whole-block path) so the mask's within-block ordering matches the gathered local cache.
    full_mask = shared_input['attention_mask']  # [S_ctx, B_attn, heads, S_tkg]
    mask_bhss = np.asarray(full_mask).transpose(1, 2, 3, 0)  # [B, H, S_tkg, S_ctx]
    rank_mask_flat = mask_bhss[:, :, :, owned]  # [B, H, S_tkg, S_local] (flat, contiguous local order)
    rank_mask = _reblock_cp_mask(rank_mask_flat, block_len, lnc)

    # Update index: map each batch's active token (global) to its rank-local physical slot, or oob.
    pos_to_local = {gp: lp for lp, gp in enumerate(owned)}
    upd = np.asarray(shared_input['kv_cache_update_idx']).copy()
    rank_upd = upd.copy()
    for b in range(B_attn):
        idx = int(upd[b, 0])
        # Reverse the global physical slot back to a global sequence position via the full table.
        phys_blk = idx // block_len
        slot = idx % block_len
        matches = np.where(full_table[b] == phys_blk)[0]
        if len(matches) == 0 or _cp_owning_rank(int(matches[0]) * block_len + slot, interleave_size, CP) != cp_rank:
            rank_upd[b, 0] = np.iinfo(np.uint32).max  # not this rank's token -> oob (skipped)
            continue
        gp = int(matches[0]) * block_len + slot
        lp = pos_to_local[gp]
        local_phys = b * num_blocks_local + lp // block_len
        rank_upd[b, 0] = local_phys * block_len + lp % block_len

    return {
        'K_cache': K_local,
        'V_cache': V_local,
        'active_blocks_table': local_table,
        'attention_mask': dt.static_cast(rank_mask, dtype=np.uint8),
        'kv_cache_update_idx': rank_upd.astype(np.uint32),
    }


def _build_cp_flat_rank_inputs(shared_input, cp_rank, CP, S_ctx, S_tkg, B_attn, lnc, K_cache_transposed):
    """Build one CP rank's flat-KV cache, mask, and update index (contiguous shard + padding).

    Flat (non-paged) KV shards the sequence contiguously: rank ``r`` owns global positions
    ``[r * S_local, (r + 1) * S_local)`` where ``S_local = S_ctx // CP``. Each batch element's active
    token is owned by one rank, ``owner = (position // S_local) % CP`` (varies per batch element).

    ``attention_tkg`` overwrites the last ``S_tkg`` positions of each rank's local cache with the
    active tokens (on every rank), so a rank whose shard is full of real prior data would lose a
    genuine token. To prevent that, the cache is zero-padded on the sequence axis to
    ``S_ext = ceil((S_local + S_tkg) / (lnc * P_MAX)) * (lnc * P_MAX)`` (the next ``lnc * P_MAX``
    boundary — ``P_MAX`` = 128, the s_prior tiling unit — which reserves room for the overwrite and
    keeps a valid kernel geometry).

    The mask is sliced and padded to match: its real tail (the active token's global column) is
    zeroed so the token is attended only via the padding entry, which is set only on the owning rank.
    ``kv_cache_update_idx`` is remapped to the local offset on the owning rank and set out-of-bounds
    (``>= B_attn * S_ext``) elsewhere so the kernel's ``oob_mode.skip`` drops the write. (The padding
    is the attention-read slot; the update index is the separate cache-write slot.)

    Args:
        shared_input (dict): Shared kernel inputs; reads ``K_cache``/``V_cache``, ``attention_mask``
            (``[S_ctx, B, heads, S_tkg]``), and ``kv_cache_update_idx`` (``[B, 1]``).
        cp_rank (int): This rank's index within its CP group (0 .. CP-1).
        CP (int): Context parallelism degree.
        S_ctx (int): Full (unsharded) prior context length.
        S_tkg (int): Active sequence length (new tokens per batch element).
        B_attn (int): Batch size seen by attention on this rank (B // KVDP, or B when KVDP=1).
        lnc (int): Logical NeuronCore count; sets the ``lnc * P_MAX`` padding alignment.
        K_cache_transposed (bool): ``[B, 1, d, S_max]`` (True) vs ``[B, 1, S_max, d]`` (False);
            selects which axis is padded.

    Returns:
        dict: Per-rank ``K_cache``, ``V_cache``, ``attention_mask``, ``kv_cache_update_idx``.
    """
    S_local = S_ctx // CP
    S_ext = -(-(S_local + S_tkg) // (lnc * _P_MAX)) * (lnc * _P_MAX)  # ceil to next lnc * P_MAX
    pad = S_ext - S_local

    # Cache: contiguous [r*S_local, (r+1)*S_local) slice + zero padding on the sequence dim.
    K_full = shared_input['K_cache']  # [B, 1, S_max, d] or [B, 1, d, S_max] (transposed)
    V_full = shared_input['V_cache']  # [B, 1, S_max, d]
    if K_cache_transposed:
        rank_k = K_full[:, :, :, cp_rank * S_local : (cp_rank + 1) * S_local]
        k_pad = np.zeros((*rank_k.shape[:3], pad), dtype=rank_k.dtype)
        K_cache = np.concatenate([rank_k, k_pad], axis=3).copy()
    else:
        rank_k = K_full[:, :, cp_rank * S_local : (cp_rank + 1) * S_local, :]
        k_pad = np.zeros((*rank_k.shape[:2], pad, rank_k.shape[3]), dtype=rank_k.dtype)
        K_cache = np.concatenate([rank_k, k_pad], axis=2).copy()
    rank_v = V_full[:, :, cp_rank * S_local : (cp_rank + 1) * S_local, :]
    v_pad = np.zeros((*rank_v.shape[:2], pad, rank_v.shape[3]), dtype=rank_v.dtype)
    V_cache = np.concatenate([rank_v, v_pad], axis=2).copy()

    # Mask: slice this rank's positions; the active token's real column is at the global tail
    # (S_ctx-1, in the last rank's slice) — zero it there so it is not double-attended, and enable
    # it in the padding only for batch elements this rank owns (owner = (pos // S_local) % CP).
    full_mask = shared_input['attention_mask']  # [S_ctx, B, heads, S_tkg]
    rank_mask = full_mask[cp_rank * S_local : (cp_rank + 1) * S_local, :, :, :].copy()
    if cp_rank == CP - 1:
        rank_mask[-S_tkg:, :, :, :] = 0
    num_heads_total = full_mask.shape[2]
    upd_full = np.asarray(shared_input['kv_cache_update_idx'])
    mask_pad = np.zeros((pad, B_attn, num_heads_total, S_tkg), dtype=rank_mask.dtype)
    for b in range(B_attn):
        idx = int(upd_full[b, 0])
        if idx < S_ctx and _cp_owning_rank(idx, S_local, CP) == cp_rank:
            mask_pad[-S_tkg:, b, :, :] = 1
    attention_mask = np.ascontiguousarray(np.concatenate([rank_mask, mask_pad], axis=0))

    # Update index: map each batch's active token to its rank-local offset, or oob (>= B * S_ext so
    # oob_mode.skip drops the write) on ranks that do not own it.
    oob_idx = B_attn * S_ext
    rank_upd = upd_full.copy()
    for b in range(rank_upd.shape[0]):
        idx = int(rank_upd[b, 0])
        if cp_rank * S_local <= idx < (cp_rank + 1) * S_local:
            rank_upd[b, 0] = idx - cp_rank * S_local
        else:
            rank_upd[b, 0] = oob_idx

    return {
        'K_cache': K_cache,
        'V_cache': V_cache,
        'attention_mask': attention_mask,
        'kv_cache_update_idx': rank_upd,
    }


def estimate_test_memory_bytes(cfg: AttnBlkTestConfig) -> int:
    """Estimate total host memory (bytes) for a test config without allocating tensors.

    Computes the sum of all tensor sizes created during the test.
    """
    batch = cfg.batch
    num_heads = cfg.q_heads
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
        rotary_half = (cfg.rotary_dim or d_head) // 2
        total += 2 * rotary_half * batch * S_tkg * elem  # cos + sin
    if cfg.qk_norm_pre_rope_gamma:
        total += 2 * d_head * elem  # W_rmsnorm_Q/K_pre_rope
    if cfg.qk_norm_post_rope_gamma:
        total += 2 * d_head * elem  # W_rmsnorm_Q/K_post_rope

    # KV cache
    if is_block_kv:
        num_blocks = batch * S_ctx // block_len
        total += 2 * num_blocks * block_len * d_head * kv_elem  # K + V cache
        total += batch * (S_ctx // block_len) * 4  # active_blocks_table (int32)
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

    # --- KVDP per-rank inputs ---
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


def _generate_kv_quant_inputs(
    cfg,
    _rng,
    H,
    num_q_heads,
    num_kv_heads,
    d_head,
    weight_dequant_scale_qkv,
    input_dequant_scale_qkv,
    K_cache_shape,
    V_cache_shape,
    kv_quant_dtype,
    seed: int = 42,
):
    """Derive kv_scale from std(K_active) and generate quantized KV cache.

    Args:
        seed: RNG seed for KV cache generation. Different per rank in KVDP tests.

    See attention_block_tkg_input_distributions_design_spec.md for full analysis.
    """
    _QUANT_DTYPE_MAX = get_max_positive_value_for_dtype(kv_quant_dtype)
    _COVERAGE = 4.0  # 4σ coverage: ~0.006% Gaussian clipping rate

    if cfg.kv_scale is KVScaleTest.DEFAULT:
        # Derive kv_scale from std(K_active) after QKV projection.
        # K = X_norm @ W where W has variance 1/H (fan-in scaled).
        # Var(K) = H × Var(X_norm) × Var(W) = H × Var(X_norm) × (1/H) = Var(X_norm)
        # Without RMSNorm: Var(X) = Var(Uniform[-1,1]) = (1-(-1))²/12 = 1/3, std ≈ 0.577
        # With RMSNorm: Var(X_norm) ≈ 1.0, std ≈ 1.0
        # kv_scale = FP8_max / (coverage × std(K_active))
        # e.g. without RMSNorm: max ≈ 4×std ≈ 4×0.577 ≈ 2.3, scale = 240/2.3 ≈ 104
        # MX micro-scaling preserves the input distribution (no per-tensor clipping),
        # so MX output variance matches the unquantized (NONE) path.
        if cfg.quantization_type == QuantizationType.NONE or cfg.quantization_type.is_mx():
            x_var = 1.0 if cfg.rmsnorm_X else 1.0 / 3.0
            k_active_std = np.sqrt(x_var)
            v_active_std = k_active_std
        elif cfg.quantization_type == QuantizationType.ROW:
            x_var = 1.0 if cfg.rmsnorm_X else 1.0 / 3.0
            k_active_std = np.sqrt(x_var)
            v_active_std = k_active_std
            if weight_dequant_scale_qkv is not None:
                num_q = num_q_heads * d_head
                k_active_std *= float(weight_dequant_scale_qkv[0, num_q : num_q + d_head].max()) / float(
                    weight_dequant_scale_qkv[0, num_q : num_q + d_head].mean()
                )
                v_active_std *= float(weight_dequant_scale_qkv[0, num_q + d_head : num_q + 2 * d_head].max()) / float(
                    weight_dequant_scale_qkv[0, num_q + d_head : num_q + 2 * d_head].mean()
                )
        elif cfg.quantization_type == QuantizationType.STATIC:
            # With calibrated in_scale, X_fp8 is Gaussian.
            # K_active = (X_fp8 @ W_fp8) × w_scale × in_scale.
            # With fan-in-calibrated w_scale and coverage-calibrated in_scale:
            # std(K_active) ≈ std(X_norm) ≈ 1.0 (with RMSNorm) or 0.577 (without).
            x_var = 1.0 if cfg.rmsnorm_X else 1.0 / 3.0
            k_active_std = np.sqrt(x_var)
            v_active_std = k_active_std
        else:
            raise ValueError(f"Unsupported quantization type: {cfg.quantization_type}")
        # QK norm (pre or post RoPE) normalizes K to unit RMS over d_head,
        # overriding the projection std. After norm: std(K) ≈ mean(gamma) ≈ 1.0
        # V is not affected by QK norm.
        if cfg.qk_norm_pre_rope or cfg.qk_norm_post_rope:
            k_active_std = 1.0
        k_scale_scalar = _QUANT_DTYPE_MAX / (_COVERAGE * k_active_std)
        v_scale_scalar = _QUANT_DTYPE_MAX / (_COVERAGE * v_active_std)
    else:
        assert isinstance(cfg.kv_scale, float), f"kv_scale must be KVScaleTest.DEFAULT or a float, got {cfg.kv_scale}"
        k_scale_scalar = cfg.kv_scale
        v_scale_scalar = cfg.kv_scale
        k_active_std = _QUANT_DTYPE_MAX / (_COVERAGE * k_scale_scalar)
        v_active_std = k_active_std

    # Use different shapes to test both broadcast (1,1) and per-partition (PMAX,1) paths
    k_scale = np.full((1, 1), k_scale_scalar, dtype=np.float32)
    v_scale = np.full((128, 1), v_scale_scalar, dtype=np.float32)

    # Generate KV cache matching the distribution of scaled K/V:
    # Cache stores K*scale, so std in cache = std(K) × scale = dtype_max/coverage
    np.random.seed(seed)
    k_cache_std = k_active_std * k_scale_scalar
    v_cache_std = v_active_std * v_scale_scalar
    K_cache_f32 = np.random.normal(0, k_cache_std, K_cache_shape).astype(np.float32)
    V_cache_f32 = np.random.normal(0, v_cache_std, V_cache_shape).astype(np.float32)
    K_cache = dt.static_cast(np.clip(K_cache_f32, -_QUANT_DTYPE_MAX, _QUANT_DTYPE_MAX), kv_quant_dtype)
    V_cache = dt.static_cast(np.clip(V_cache_f32, -_QUANT_DTYPE_MAX, _QUANT_DTYPE_MAX), kv_quant_dtype)

    return k_scale, v_scale, K_cache, V_cache


def generate_kernel_inputs(cfg: AttnBlkTestConfig, seed: int = 0):
    """Generate kernel inputs for a single rank.

    Args:
        seed: RNG seed. Use different values per rank in KVDP tests to produce
            independent K/V cache, weights, and mask data for each rank.
    """
    from test.integration.nkilib.core.attention.test_attention_tkg_utils import (
        build_active_attention_mask,
        build_swa_positions,
        gen_deterministic_active_block_table,
        generate_cache_lens,
    )

    # Short aliases for dimensions used repeatedly in shape expressions
    dtype = cfg.dtype
    batch, d_head, H = cfg.batch, cfg.d_head, cfg.H
    S_ctx, S_max_ctx, S_tkg = cfg.S_ctx, cfg.S_max_ctx, cfg.S_tkg
    H_actual = cfg.H_actual if cfg.H_actual is not None else H
    num_q_heads = cfg.q_heads
    num_kv_heads = cfg.kv_heads
    # KVDP: X uses full batch, K/V cache and mask use local batch
    B_attn = batch // cfg.KVDP if cfg.KVDP > 1 else batch
    num_mask_heads = cfg.KVDP * cfg.CP * num_q_heads if (cfg.KVDP > 1 or cfg.CP > 1) else num_q_heads

    # Seed global RNG for generate_cache_lens / gen_deterministic_active_block_table
    np.random.seed(seed)

    eps = 1e-5 if dtype == np.float32 else 1e-3

    # FP8 constants used across quantization paths
    _COVERAGE = 4.0  # 4σ coverage: ~0.006% Gaussian clipping rate
    _weight_fp8_dtype = nl.float8_e4m3  # Weight quantization dtype (independent of KV cache dtype)

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
    # To keep scores O(1), we use W ~ N(0, σ=1/√fan_in). This scaling is "unitary"
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
    # See attention_block_tkg_input_distributions_design_spec.md for full analysis.
    _rng = np.random.default_rng(seed)

    def uniform_activation(shape, dtype):
        """Uniform[-1, 1]"""
        return np.ascontiguousarray(dt.static_cast(_rng.uniform(-1.0, 1.0, shape).astype(np.float32), dtype))

    def gaussian(shape, dtype, std):
        """N(0, std)"""
        return np.ascontiguousarray(dt.static_cast(_rng.normal(0.0, std, shape).astype(np.float32), dtype))

    def fan_in_projection(shape, dtype, fan_in):
        """N(0, σ=1/√fan_in). Keeps matmul output variance ≈ input variance."""
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

    # If transposed_in, convert X from [B, S, H] to [H0, n_prgs, H1_shard, BxS]
    # using (lnc, h0, h1) decomposition: h = lnc * H_per_shard + h0 * H1_shard + h1
    # numpy: flat.reshape(BxS, n_prgs, H0, H1_shard).transpose(2, 1, 3, 0)
    if cfg.transposed_in:
        n_prgs = cfg.lnc
        H0 = nl.tile_size.pmax
        H1_shard = H // n_prgs // H0
        BxS = batch * S_tkg
        X_transposed = X.reshape(BxS, n_prgs, H0, H1_shard).transpose(2, 1, 3, 0)
        X_transposed = np.ascontiguousarray(X_transposed)
        X = X_transposed

    # -- rmsnorm X: gamma weights are ~1.0 in trained models
    rmsnorm_X_gamma = near_unity((1, H), dtype) if cfg.rmsnorm_X else None

    # -- qkv projections, optional bias
    dim_I_qkv = (num_q_heads + 2 * num_kv_heads) * d_head
    if cfg.quantization_type == QuantizationType.NONE:
        # W_qkv: projection from hidden dim H → QKV, fan-in-scaled by fan_in=H
        W_qkv = fan_in_projection((H, dim_I_qkv), dtype, fan_in=H)
        weight_dequant_scale_qkv = None
        input_dequant_scale_qkv = None
    elif cfg.quantization_type == QuantizationType.ROW:
        # fan_in=H calibrates per-row w_scale for variance-preserving projection.
        # See attention_block_tkg_input_distributions_design_spec.md for full analysis.
        W_qkv, weight_dequant_scale_qkv, _ = generate_quant_tensor(
            shape=(H, dim_I_qkv), dtype=_weight_fp8_dtype, granularity="row", fan_in=H
        )
        weight_dequant_scale_qkv = np.broadcast_to(weight_dequant_scale_qkv, (128, dim_I_qkv))
        input_dequant_scale_qkv = None
    elif cfg.quantization_type == QuantizationType.STATIC:
        # fan_in=H calibrates w_scale so W_fp8 × w_scale has variance 1/H (variance-preserving).
        # in_scale is calibrated so X_fp8 is Gaussian: in_scale = coverage × std(X) / FP8_MAX.
        # With rmsnorm_X: std(X_norm) ≈ 1.0. Without: std(X) = std(Uniform[-1,1]) = 1/√3 ≈ 0.577.
        W_q, w_scale_q, _ = generate_quant_tensor(shape=(H, num_q_heads * d_head), dtype=_weight_fp8_dtype, fan_in=H)
        W_k, w_scale_k, _ = generate_quant_tensor(shape=(H, num_kv_heads * d_head), dtype=_weight_fp8_dtype, fan_in=H)
        W_v, w_scale_v, _ = generate_quant_tensor(shape=(H, num_kv_heads * d_head), dtype=_weight_fp8_dtype, fan_in=H)
        W_qkv = np.concatenate([W_q, W_k, W_v], axis=1)
        weight_dequant_scale_qkv = np.array([[w_scale_q, w_scale_k, w_scale_v]])
        weight_dequant_scale_qkv = np.broadcast_to(weight_dequant_scale_qkv, (128, 3))
        _x_std = 1.0 if cfg.rmsnorm_X else np.sqrt(1.0 / 3.0)
        _qkv_quant_max = get_max_positive_value_for_dtype(
            _weight_fp8_dtype
        )  # matches dtype in generate_quant_tensor calls above
        _in_scale = np.float32(_COVERAGE * _x_std / _qkv_quant_max * _rng.uniform(0.8, 1.2))
        input_dequant_scale_qkv = np.broadcast_to(_in_scale.reshape(1, 1), (128, 1))
    elif cfg.quantization_type.is_mx():
        _q_width = 4
        W_qkv, weight_dequant_scale_qkv, input_dequant_scale_qkv = _generate_mx_weights_and_scales(
            cfg.quantization_type,
            weight_shape=(H // _q_width, dim_I_qkv * _q_width),
            mx_scale_reshape=(H // 32, dim_I_qkv),
            w_scale_shape=(1, 3) if cfg.quantization_type == QuantizationType.STATIC_MX else (1, dim_I_qkv),
            static_mx_in_shape=(1, 1),
            rng=_rng,
        )
        W_qkv = W_qkv.view(nl.float8_e4m3fn).reshape(H // _q_width, dim_I_qkv, _q_width)
        # Pre-shuffle X along H for MXFP hardware layout: [B,S,H//512,128,4] → [B,S,4,H//512,128]
        X = np.ascontiguousarray(
            X.reshape(batch, S_tkg, H // (_P_MAX * _q_width), _P_MAX, _q_width)
            .transpose(0, 1, 4, 2, 3)
            .reshape(batch, S_tkg, H)
        )
        # Pre-shuffle RMSNorm gamma to match shuffled H layout
        if rmsnorm_X_gamma is not None:
            rmsnorm_X_gamma = np.ascontiguousarray(
                rmsnorm_X_gamma.reshape(1, H // (_P_MAX * _q_width), _P_MAX, _q_width)
                .transpose(0, 3, 1, 2)
                .reshape(1, H)
            )
    else:
        raise ValueError(f"Unsupported quantization type: {cfg.quantization_type}")
    bias_qkv = small_bias((1, (num_q_heads + 2 * num_kv_heads) * d_head), dtype) if cfg.test_bias else None

    # -- rmsnorm QK pre RoPE gamma weights
    W_rmsnorm_Q_pre_rope = near_unity((1, d_head), dtype) if cfg.qk_norm_pre_rope_gamma else None
    W_rmsnorm_K_pre_rope = near_unity((1, d_head), dtype) if cfg.qk_norm_pre_rope_gamma else None
    # -- RoPE: cos/sin are bounded to [-1, 1] by definition. Partial rotary (rotary_dim > 0)
    # sizes them to rotary_dim // 2; full rotary (rotary_dim == 0) uses d_head // 2.
    rotary_half = (cfg.rotary_dim or d_head) // 2
    cos = None if cfg.skip_rope else uniform_activation((rotary_half, batch, S_tkg), dtype)
    sin = None if cfg.skip_rope else uniform_activation((rotary_half, batch, S_tkg), dtype)

    # -- rmsnorm QK post RoPE
    W_rmsnorm_Q_post_rope = near_unity((1, d_head), dtype) if cfg.qk_norm_post_rope_gamma else None
    W_rmsnorm_K_post_rope = near_unity((1, d_head), dtype) if cfg.qk_norm_post_rope_gamma else None

    # -- Attention (and KV cache)
    is_block_kv = cfg.block_len > 0

    # The explicit kv-head axis is required when kv_heads > 1;
    # cache_has_kv_head_dim only toggles the optional size-1 axis for kv_heads == 1.
    use_explicit_kv_head_dim = cfg.cache_has_kv_head_dim or num_kv_heads > 1

    # Determine cache shapes
    if is_block_kv:
        assert not cfg.K_cache_transposed
        assert S_ctx % cfg.block_len == 0
        logical_blocks_per_head = B_attn * S_ctx // cfg.block_len
        blocks_per_head_pool = (
            cfg.physical_cache_blocks_per_head
            if cfg.physical_cache_blocks_per_head is not None
            else logical_blocks_per_head
        )
        assumed_num_cache_blocks = blocks_per_head_pool * num_kv_heads
        if cfg.fp8_packed:
            K_cache_shape = (assumed_num_cache_blocks, cfg.block_len // 2, d_head, 2)
        else:
            K_cache_shape = (assumed_num_cache_blocks, cfg.block_len, d_head)
        V_cache_shape = (assumed_num_cache_blocks, cfg.block_len, d_head)
        if use_explicit_kv_head_dim:
            K_cache_shape = (K_cache_shape[0] // num_kv_heads, num_kv_heads) + K_cache_shape[1:]
            V_cache_shape = (V_cache_shape[0] // num_kv_heads, num_kv_heads) + V_cache_shape[1:]
    else:
        assumed_num_cache_blocks = 0
        K_cache_shape = (
            (B_attn, num_kv_heads, d_head, S_max_ctx)
            if cfg.K_cache_transposed
            else (B_attn, num_kv_heads, S_max_ctx, d_head)
        )
        V_cache_shape = (B_attn, num_kv_heads, S_max_ctx, d_head)

    # Generate KV cache in FP8 when kv_quant=True
    kv_cache_dtype = cfg.kv_quant_dtype if cfg.kv_quant else dtype
    _CACHE_DTYPE_MAX = get_max_positive_value_for_dtype(kv_cache_dtype)
    if cfg.kv_quant:
        K_cache_gen_shape = (assumed_num_cache_blocks, cfg.block_len, d_head) if cfg.fp8_packed else K_cache_shape
        # Determine KV scale and generate FP8 cache
        k_scale, v_scale, K_cache, V_cache = _generate_kv_quant_inputs(
            cfg,
            _rng,
            H,
            num_q_heads,
            num_kv_heads,
            d_head,
            weight_dequant_scale_qkv,
            input_dequant_scale_qkv,
            K_cache_gen_shape,
            V_cache_shape,
            kv_cache_dtype,
            seed=seed,
        )
        if cfg.fp8_packed:
            # Transpose/Reshape fp8 [num_blocks, block_len, d_head] -> [num_blocks, block_len//2, d_head, 2]
            num_blocks, block_len_full, d_head_k = K_cache.shape
            K_cache = K_cache.reshape(num_blocks, block_len_full // 2, 2, d_head_k).transpose(0, 1, 3, 2)
            if use_explicit_kv_head_dim:
                K_cache = K_cache.reshape((K_cache.shape[0] // num_kv_heads, num_kv_heads) + K_cache.shape[1:])
    else:
        # KV cache stores projected K/V values which are O(1) after fan-in-scaled projection
        K_cache = uniform_activation(K_cache_shape, kv_cache_dtype)
        V_cache = uniform_activation(V_cache_shape, kv_cache_dtype)
        k_scale = None
        v_scale = None

    # pos_id (shape=(batch, 1)) defines the first position to append new KV to cache, per batch element
    if cfg.CP > 1:
        np.random.seed(seed)
        S_max = S_ctx - S_tkg
        bl = cfg.block_len if cfg.block_len > 0 else (S_ctx // cfg.CP)
        interleave_size = cfg.cp_interleave_size if cfg.cp_interleave_size is not None else bl
        cache_len = _generate_cp_cache_lens(B_attn, cfg.CP, interleave_size, S_max)
    else:
        cache_len_kwargs = {}
        if cfg.cache_lens_mean is not None:
            cache_len_kwargs["mean_frac"] = cfg.cache_lens_mean
        elif cfg.max_context_len is not None:
            cache_len_kwargs["mean_frac"] = cfg.max_context_len / (2 * (S_ctx - S_tkg))
        if cfg.cache_lens_stddev is not None:
            cache_len_kwargs["stddev_frac"] = cfg.cache_lens_stddev
        cache_len = generate_cache_lens(B_attn, S_ctx, S_tkg, **cache_len_kwargs)
        assert cache_len.max() <= (S_ctx - S_tkg)
        if cfg.max_context_len is not None:
            max_cache = cfg.max_context_len - S_tkg
            cache_len = np.clip(cache_len, 0, max_cache)
            cache_len[0] = max_cache
    import torch

    if cfg.use_pos_id:
        # In-kernel mask generation: pass pos_ids instead of attention_mask
        if cfg.sliding_window > 0:
            swa_start_pos_ids, pos_ids = build_swa_positions(
                pos_id=cache_len,
                bs=B_attn,
                s_active=S_tkg,
                sliding_window=cfg.sliding_window,
                cache_len=S_ctx,
                block_len=cfg.block_len,
            )
        else:
            pos_ids = np.broadcast_to(cache_len, (B_attn, S_tkg)).astype(np.float32)
            if S_tkg > 1:
                pos_ids = pos_ids + np.arange(S_tkg, dtype=np.float32)[np.newaxis, :]
            swa_start_pos_ids = None

        attention_mask = (
            build_active_attention_mask(
                batch=B_attn,
                num_heads=num_mask_heads,
                s_active=S_tkg,
                transposed=True,
            )
            .numpy()
            .astype(np.uint8)
        )
    else:
        pos_ids = None
        swa_start_pos_ids = None
        pos_ids_hbm = torch.from_numpy(
            np.broadcast_to(cache_len, (B_attn, S_tkg)).astype(np.float32)
            + (np.arange(S_tkg, dtype=np.float32)[np.newaxis, :] if S_tkg > 1 else 0.0)
        ).reshape(1, B_attn * S_tkg)
        active_mask_hbm = build_active_attention_mask(
            batch=B_attn,
            num_heads=num_mask_heads,
            s_active=S_tkg,
            transposed=True,
        )  # [S_tkg, B, N, S]
        transposed_out = is_qk_swapped(
            bs=B_attn,
            q_head=num_mask_heads,
            d_head=d_head,
            s_active=S_tkg,
            curr_sprior=S_ctx,
            lnc=cfg.lnc,
            p_max=_P_MAX,
            is_block_kv=is_block_kv,
            is_2byte_kv=sizeinbytes(kv_cache_dtype) == 2,
            fp8_packed=cfg.fp8_packed,
            fuse_rope=False,
            kv_heads=num_kv_heads,
        )
        attention_mask = gen_mask_tkg_hbm_torch_ref[cfg.lnc](
            pos_ids_hbm=pos_ids_hbm,
            bs=B_attn,
            q_head=num_mask_heads,
            s_active=S_tkg,
            s_prior=S_ctx,
            block_len=cfg.block_len,
            active_mask=active_mask_hbm,
            enable_fa_s_prior_tiling=cfg.enable_fa_s_prior_tiling,
            transposed_out=transposed_out,
        ).numpy()  # default: (S_ctx, B, N, S); QK-swap: (B, N, S, S_ctx)
        attention_mask = dt.static_cast(np.ascontiguousarray(attention_mask), dtype=np.uint8)

    # Attention sink: one scalar per (KVDP-expanded) query head, [q_heads_attn, 1] @ HBM.
    sink = _rng.uniform(0.0, 1.0, (num_mask_heads, 1)).astype(np.float32) if cfg.test_sink else None

    # active_blocks_table holds GLOBAL block indices into the flattened ``num_blocks * kv_heads`` dimension.
    # Reuse the same indices between heads shifted to their global indices.
    if is_block_kv:
        base = gen_deterministic_active_block_table(
            B_attn, S_ctx, S_tkg, cache_len, cfg.block_len, blocks_per_head_pool
        ).astype(np.int32)  # [B, num_blocks]; INACTIVE padding = -1
        if num_kv_heads == 1:
            active_blocks_table = base
        else:
            base = base[:, np.newaxis, :]  # [B, 1, num_blocks]
            kv_h = np.arange(num_kv_heads, dtype=np.int32).reshape(1, num_kv_heads, 1)
            shifted = base * num_kv_heads + kv_h  # head-inner flattened addressing
            active_blocks_table = np.where(base == -1, -1, shifted).astype(np.int32)  # [B, kv_heads, num_blocks]
    else:
        active_blocks_table = None

    # kv_cache_update_idx: (B, S_tkg) for block KV (kv_heads==1) with per-token physical positions,
    #                      (B, kv_heads, S_tkg) for block KV with kv_heads>1 (per-head physical positions),
    #                      (B, 1) for flat KV with start position (consecutive tokens assumed).
    def generate_block_kv_cache_update_idx(abt: npt.NDArray[np.int32]):
        # Block KV: translate each token's logical position to physical slot_mapping.
        # Generated per-(batch, kv_head, token); squeeze kv axis at end to match kernel contract:
        # (B, S_tkg) for kv_heads==1, (B, kv_heads, S_tkg) otherwise.
        logical_positions = cache_len + np.arange(S_tkg)  # (B, S_tkg)
        logical_blks = logical_positions // cfg.block_len
        offset_in_blk = logical_positions % cfg.block_len
        # Wrap a 2D active_blocks_table to 3D for a uniform per-head gather.
        abt_3d = abt if num_kv_heads > 1 else abt[:, np.newaxis, :]
        # physical_blks[b, h, t] = abt_3d[b, h, logical_blks[b, t]]
        physical_blks = abt_3d[
            np.arange(B_attn)[:, None, None],
            np.arange(num_kv_heads)[None, :, None],
            logical_blks[:, None, :],
        ]  # (B, kv_heads, S_tkg)
        physical_kv_cache_update_idx = physical_blks * cfg.block_len + offset_in_blk[:, np.newaxis, :]
        # Mark last batch element as padding to test that scenario.
        if B_attn > 1:
            physical_kv_cache_update_idx[-1, ...] = -1
        if num_kv_heads == 1:
            physical_kv_cache_update_idx = physical_kv_cache_update_idx.squeeze(1)  # (B, S_tkg)
        return physical_kv_cache_update_idx.astype(np.uint32)

    if active_blocks_table is None:
        # Flat KV: only start position needed, consecutive tokens assumed
        kv_cache_update_idx = cache_len.astype(np.uint32)
    else:
        kv_cache_update_idx = generate_block_kv_cache_update_idx(active_blocks_table)

    # Output projection
    weight_dequant_scale_out = None
    input_dequant_scale_out = None
    if cfg.skip_output_projection:
        W_out = None
    elif cfg.quantization_type == QuantizationType.NONE:
        # W_out: projection from attention output → H. After softmax, probs@V has low std.
        # Use std=0.5 to scale output back up so bias is meaningful (not at noise level).
        W_out = gaussian((cfg.q_heads * d_head, H), dtype, std=0.5)
    elif cfg.quantization_type == QuantizationType.ROW:
        # fan_in for variance-preserving per-row w_scale.
        W_out, weight_dequant_scale_out, _ = generate_quant_tensor(
            shape=(cfg.q_heads * d_head, H), dtype=_weight_fp8_dtype, granularity="row", fan_in=cfg.q_heads * d_head
        )
        weight_dequant_scale_out = np.broadcast_to(weight_dequant_scale_out, (128, H))
        input_dequant_scale_out = None
    elif cfg.quantization_type == QuantizationType.STATIC:
        _fan_in_out = cfg.q_heads * d_head
        _out_proj_quant_dtype = _weight_fp8_dtype
        W_out, weight_dequant_scale_out, _ = generate_quant_tensor(
            shape=(_fan_in_out, H), dtype=_out_proj_quant_dtype, fan_in=_fan_in_out
        )
        # Calibrate input_dequant_scale_out to attention output magnitude.
        # With kv_quant: attn_out ≈ softmax @ V_cache_fp8, std ≈ FP8_MAX/_COVERAGE ≈ 60
        # Without kv_quant: attn_out ≈ softmax @ V_cache_bf16, std ≈ 0.577
        # Jitter (0.8-1.2×) tests robustness to imperfect calibration.
        _out_proj_quant_max = get_max_positive_value_for_dtype(_out_proj_quant_dtype)
        _attn_out_std = _CACHE_DTYPE_MAX / _COVERAGE if cfg.kv_quant else np.sqrt(1.0 / 3.0)
        _jitter = _rng.uniform(0.8, 1.2)
        input_dequant_scale_out = np.float32(_COVERAGE * _attn_out_std / _out_proj_quant_max * _jitter)
        weight_dequant_scale_out = np.broadcast_to(weight_dequant_scale_out.reshape(1, 1), (128, 1))
        input_dequant_scale_out = np.broadcast_to(input_dequant_scale_out.reshape(1, 1), (128, 1))
    elif cfg.quantization_type.is_mx():
        _q_width = 4
        N_D = cfg.q_heads * d_head
        W_out, weight_dequant_scale_out, input_dequant_scale_out = _generate_mx_weights_and_scales(
            cfg.quantization_type,
            weight_shape=(N_D // _q_width, H * _q_width),
            mx_scale_reshape=(N_D // 32, H),
            w_scale_shape=(1, 1) if cfg.quantization_type == QuantizationType.STATIC_MX else (1, H),
            static_mx_in_shape=(1, 1),
            rng=_rng,
        )
        W_out = W_out.view(nl.float8_e4m3fn).reshape(N_D // _q_width, H, _q_width)
        # Output projection STATIC_MX scales need broadcast to (128, 1) for kernel interface
        if cfg.quantization_type == QuantizationType.STATIC_MX:
            weight_dequant_scale_out = np.broadcast_to(weight_dequant_scale_out, (128, 1)).copy()
            input_dequant_scale_out = np.broadcast_to(input_dequant_scale_out, (128, 1)).copy()
    else:
        raise ValueError(f"Unsupported quantization type: {cfg.quantization_type}")

    # bias_out: match the output projection scale (~0.1) so bias doesn't dominate.
    bias_out = small_bias((1, H), dtype) if cfg.test_bias else None

    # ── FP8 KV cache scale fusion ──────────────────────────────────────────
    # The kernel operates on raw FP8 KV values (K*k_scale, V*v_scale) without
    # dequantizing. When softmax_scale is None, the kernel automatically fuses
    # k_scale. When softmax_scale is explicit, the caller must fuse k_scale.
    # The caller must always fuse v_scale into W_out.
    softmax_scale_adjusted = cfg.softmax_scale
    if cfg.kv_quant:
        assert k_scale is not None and v_scale is not None, "the quantized KV path generates cache scales"
        _k_scale_scalar = float(k_scale.flat[0])
        _v_scale_scalar = float(v_scale.flat[0])

        # Only fuse k_scale into softmax_scale when explicitly provided;
        # the kernel handles the None case automatically.
        if cfg.softmax_scale is not None:
            softmax_scale_adjusted = cfg.softmax_scale / _k_scale_scalar

        if W_out is not None:
            if cfg.quantization_type == QuantizationType.NONE:
                # bf16 weights: divide directly
                W_out = dt.static_cast(W_out.astype(np.float32) / _v_scale_scalar, dtype)
            elif cfg.quantization_type in (QuantizationType.ROW, QuantizationType.STATIC):
                # FP8 weights with dequant scale: absorb v_scale into the scale
                assert weight_dequant_scale_out is not None, "FP8 output weights carry a dequant scale"
                weight_dequant_scale_out = (weight_dequant_scale_out.astype(np.float32) / _v_scale_scalar).astype(
                    np.float32
                )

    return {
        # -- input
        "X": X,
        "X_in_sb": cfg.input_in_sb,
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
        # -- Q/K processing: pre-RoPE RMSNorm
        "rmsnorm_QK_pre_rope_enabled": cfg.qk_norm_pre_rope,
        "rmsnorm_QK_pre_rope_eps": eps,
        "rmsnorm_QK_pre_rope_W_Q": W_rmsnorm_Q_pre_rope,
        "rmsnorm_QK_pre_rope_W_K": W_rmsnorm_K_pre_rope,
        # -- RoPE
        "cos": cos,
        "sin": sin,
        "rope_contiguous_layout": cfg.rope_contiguous_layout,
        # -- Q/K processing: post-RoPE RMSNorm
        "rmsnorm_QK_post_rope_enabled": cfg.qk_norm_post_rope,
        "rmsnorm_QK_post_rope_eps": eps,
        "rmsnorm_QK_post_rope_W_Q": W_rmsnorm_Q_post_rope,
        "rmsnorm_QK_post_rope_W_K": W_rmsnorm_K_post_rope,
        # -- attention
        "skip_attention": cfg.skip_attention,
        "K_cache_transposed": cfg.K_cache_transposed,
        "active_blocks_table": active_blocks_table,
        "K_cache": K_cache,
        "V_cache": V_cache,
        "attention_mask": attention_mask,
        "sink": sink,
        "softmax_scale": softmax_scale_adjusted,
        "enable_fa_s_prior_tiling": cfg.enable_fa_s_prior_tiling,
        "fp8_packed": cfg.fp8_packed,
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
        "transposed_in": cfg.transposed_in,
        "out_in_sb": cfg.output_in_sb,
        # -- KV data parallelism
        "KVDP": cfg.KVDP,
        "KVDP_replica_group": None,
        "KVDP_collective_mode": cfg.KVDP_collective_mode,
        # -- in-kernel mask generation
        "pos_ids": pos_ids,
        "swa_start_pos_ids": swa_start_pos_ids,
        "S_ctx": S_ctx if (cfg.use_pos_id and cfg.block_len == 0) else None,
        # -- MXFP quantization
        "is_h_transposed_by_4": cfg.quantization_type.is_mx(),
        # -- dynamic FA
        "max_context_len": np.array([cfg.max_context_len], dtype=np.int32) if cfg.max_context_len is not None else None,
    }


# wrapper to test SBUF IO
def attention_block_tkg_kernel_test_wrapper(
    # -- input
    X: nl.ndarray,
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
    # -- Q/K processing: pre-RoPE RMSNorm
    rmsnorm_QK_pre_rope_enabled: bool,
    rmsnorm_QK_pre_rope_eps: float,
    rmsnorm_QK_pre_rope_W_Q: Optional[nl.ndarray],
    rmsnorm_QK_pre_rope_W_K: Optional[nl.ndarray],
    # -- RoPE embeddings
    cos: Optional[nl.ndarray],
    sin: Optional[nl.ndarray],
    rope_contiguous_layout: bool,
    # -- Q/K processing: post-RoPE RMSNorm
    rmsnorm_QK_post_rope_enabled: bool,
    rmsnorm_QK_post_rope_eps: float,
    rmsnorm_QK_post_rope_W_Q: Optional[nl.ndarray],
    rmsnorm_QK_post_rope_W_K: Optional[nl.ndarray],
    # -- attention
    skip_attention: bool,
    K_cache_transposed: bool,
    active_blocks_table: Optional[nl.ndarray],
    K_cache: nl.ndarray,
    V_cache: nl.ndarray,
    attention_mask: nl.ndarray,
    sink: Optional[nl.ndarray],
    softmax_scale: Optional[float],
    enable_fa_s_prior_tiling: bool,
    fp8_packed: bool,
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
    transposed_in: bool,
    out_in_sb: bool,
    sbm: Optional[SbufManager] = None,
    # -- KV data parallelism
    KVDP: int = 1,
    KVDP_replica_group=None,
    KVDP_collective_mode=None,
    KVDP_rank: Optional[nl.ndarray] = None,
    # -- in-kernel mask generation
    pos_ids: Optional[nl.ndarray] = None,
    swa_start_pos_ids: Optional[nl.ndarray] = None,
    S_ctx: Optional[int] = None,
    is_h_transposed_by_4: bool = False,
    max_context_len=None,
    dtype_mode: DtypeMode = DtypeMode.NON_OCP,
    # -- Context parallelism
    CP: int = 1,
    CP_replica_group=None,
    CP_collective_mode=None,
):
    if transposed_in:
        # X is already in transposed layout [H0, n_prgs, H1_shard, BxS] from generate_kernel_inputs
        H0, n_prg, H1, _ = X.shape
        H = H0 * n_prg * H1
        _, B, _, S_tkg = attention_mask.shape
    else:
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
            dst=X_sb.reshape_dim(2, (lnc, -1)),
            src=X_hbm.rearrange(("BS", "lnc", "H0", "H1 // lnc"), ("H0", "BS", "lnc", "H1 // lnc")),
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
        rmsnorm_QK_pre_rope_enabled=rmsnorm_QK_pre_rope_enabled,
        rmsnorm_QK_pre_rope_eps=rmsnorm_QK_pre_rope_eps,
        rmsnorm_QK_pre_rope_W_Q=rmsnorm_QK_pre_rope_W_Q,
        rmsnorm_QK_pre_rope_W_K=rmsnorm_QK_pre_rope_W_K,
        cos=cos,
        sin=sin,
        rope_contiguous_layout=rope_contiguous_layout,
        rmsnorm_QK_post_rope_enabled=rmsnorm_QK_post_rope_enabled,
        rmsnorm_QK_post_rope_eps=rmsnorm_QK_post_rope_eps,
        rmsnorm_QK_post_rope_W_Q=rmsnorm_QK_post_rope_W_Q,
        rmsnorm_QK_post_rope_W_K=rmsnorm_QK_post_rope_W_K,
        skip_attention=skip_attention,
        K_cache_transposed=K_cache_transposed,
        active_blocks_table=active_blocks_table,
        K_cache=K_cache,
        V_cache=V_cache,
        attention_mask=attention_mask,
        sink=sink,
        softmax_scale=softmax_scale,
        enable_fa_s_prior_tiling=enable_fa_s_prior_tiling,
        fp8_packed=fp8_packed,
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
        transposed_in=transposed_in,
        sbm=sbm,
        KVDP=KVDP,
        KVDP_replica_group=KVDP_replica_group,
        KVDP_collective_mode=KVDP_collective_mode,
        KVDP_rank=KVDP_rank,
        pos_ids=pos_ids,
        swa_start_pos_ids=swa_start_pos_ids,
        S_ctx=S_ctx,
        is_h_transposed_by_4=is_h_transposed_by_4,
        max_context_len=max_context_len,
        dtype_mode=dtype_mode,
        CP=CP,
        CP_replica_group=CP_replica_group,
        CP_collective_mode=CP_collective_mode,
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
    kv_dtype = kernel_input['K_cache'].dtype
    torch_ref = AttentionBlockTkgTorchRef(lnc, kv_quant_dtype=str(kv_dtype))
    ref_params = set(signature(torch_ref).parameters)
    ignored = set(kernel_input) - ref_params
    # X_in_sb is a test-wrapper param for SBUF pre-loading, not part of the torch_ref
    ignored.discard("X_in_sb")
    assert not ignored, f"kernel_input keys not consumed by torch ref: {ignored}"
    ref_input = {k: v for k, v in kernel_input.items() if k in ref_params}
    # preserve_lower_precision=True so bf16 KV reaches the ref as torch.bfloat16 (not the float32 upcast),
    # letting its is_qk_swapped call read the true 2-byte KV dtype for the swap-mask layout decision.
    # (fp8 still arrives as float32 by the wrapper's existing contract.) The per-output .astype below
    # re-casts to the exact kernel dtypes, so the wrapper's cast-back does not change golden values.
    ref_output = torch_ref_wrapper(torch_ref, preserve_lower_precision=True)(**ref_input)
    x_dtype = kernel_input['X'].dtype
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


def _get_tolerances(kv_quant: bool, quantization_type: QuantizationType, kv_quant_dtype_max: float = 240.0):
    """Return per-output (rtol, atol) dict based on quantization mode.

    With k_scale/v_scale fusion, X_out is O(1) for all kv_quant configs.
    K/V cache/tkg values remain in FP8 range (std ≈ FP8_MAX/4), so their
    tolerances are set differently:

    - atol = FP8_MAX/32: the FP8 step size at 1σ of the cache distribution.
      Covers rounding differences at small values where rtol contributes little.
    - rtol = 6%: FP8 has 3 mantissa bits (~12.5% worst-case quantization error).
      At high magnitudes (2-3σ), the step size can reach FP8_MAX/8 to FP8_MAX/16.
      The combined tolerance (atol + rtol × |value|) ensures coverage across the
      full range. 6% is used instead of 5% because kernel and golden may round
      to different adjacent FP8 values, and the relative error at 2-3σ values
      can slightly exceed 5%.

    X_out uses tighter rtol=5% because output values are O(1) after v_scale
    fusion, so FP8 quantization noise is not the dominant error source.
    """
    if kv_quant or quantization_type != QuantizationType.NONE:
        _kv_cache_atol = kv_quant_dtype_max / 32.0  # ULP at 1σ of cache distribution
        return {
            "X_out": (0.05, 1.0),
            "K_cache": (0.06, _kv_cache_atol),
            "V_cache": (0.06, _kv_cache_atol),
            "K_tkg": (0.06, _kv_cache_atol),
            "V_tkg": (0.06, _kv_cache_atol),
        }
    return {
        "X_out": (0.015, 1e-5),
        "K_cache": (0.015, 1e-5),
        "V_cache": (0.015, 1e-5),
        "K_tkg": (0.015, 1e-5),
        "V_tkg": (0.015, 1e-5),
    }


def _make_cosine_validation(
    golden_outputs: dict,
    tolerances: dict,
    kv_quant: bool,
    quantization_type: QuantizationType = QuantizationType.NONE,
    name_prefix: str = "",
) -> dict:
    """Build per-output cosine similarity + allclose validators.

    Pass rate:
      - Non-quantized (bf16): 100% of elements within rtol.
      - FP8 kv_quant X_out: 99% pass rate. FP8 has coarser quantization.
    """
    min_pass_rate_x_out = 0.99 if kv_quant else 1.0
    _cosine_threshold = 0.995 if (kv_quant or quantization_type != QuantizationType.NONE) else 0.99
    return {
        name: CustomValidatorWithOutputTensorData(
            validator=make_cosine_similarity_validator(
                golden,
                rtol=tolerances[name][0],
                atol=tolerances[name][1],
                min_cosine_similarity=_cosine_threshold,
                min_pass_rate=min_pass_rate_x_out if name == 'X_out' else 1.0,
                name=f"{name_prefix}{name}",
            ),
            output_ndarray=golden,
        )
        for name, golden in golden_outputs.items()
    }


def _run_attention_block_test(
    test_manager: Orchestrator,
    platform_target: Platforms,
    cfg: AttnBlkTestConfig,
):
    """Shared test execution logic for attention block TKG kernel.

    Single-rank case (KVDP=1):
        Uses UnitTestFramework with torch reference (AttentionBlockTkgTorchRef).

        INPUTS ──┬──> KERNEL ──> X_out, K/V_out ──┐
                 │                                ├─> compare
                 └──> GOLDEN ──> X_out, K/V_out ──┘

    Multi-rank case (KVDP>1):
        Each rank generates its own inputs via generate_kernel_inputs (KVDP-aware:
        X at full batch B, K/V cache and mask at B_attn = B/KVDP).
        X and cos/sin are shared across ranks; everything else is independent.
        The distributed torch ref handles collectives internally.
    """
    estimated_bytes = estimate_test_memory_bytes(cfg)
    if estimated_bytes > _MAX_MEMORY_BYTES:
        pytest.skip(
            f"Estimated memory {estimated_bytes / 1024**3:.1f} GiB exceeds "
            f"limit {_MAX_MEMORY_BYTES / 1024**3:.1f} GiB "
            f"(set TEST_ATTN_BLK_TKG_MAX_MEMORY_GB to override)"
        )

    tolerances = _get_tolerances(
        cfg.kv_quant, cfg.quantization_type, get_max_positive_value_for_dtype(cfg.kv_quant_dtype)
    )

    if cfg.KVDP > 1 or cfg.CP > 1:
        _run_multi_rank_test(test_manager, platform_target, cfg, tolerances)
    else:
        _run_single_rank_test(test_manager, platform_target, cfg, tolerances)


def _run_single_rank_test(
    test_manager: Orchestrator,
    platform_target: Platforms,
    cfg: AttnBlkTestConfig,
    tolerances: dict,
):
    """Run single-rank test using UnitTestFramework with cosine similarity validation."""

    kernel_input = generate_kernel_inputs(cfg)
    golden_outputs = _golden_ref_via_torch(kernel_input, cfg.lnc)

    # When update_cache=True, the kernel returns K_cache/V_cache in-place, causing:
    # 1. The NKI compiler renames these inputs to K_cache.must_alias_input / V_cache.must_alias_input
    #    in the NEFF. Rename input keys so the neuron-explorer command uses the NEFF names.
    # 2. The NEFF output files are named K_cache/V_cache (matching the aliased inputs), not
    #    K_cache_updated/V_cache_updated (the torch ref names). Rename golden keys to match.
    # The test framework handles the .must_alias_input suffix throughout
    # (kernel_tracer strips it for compilation, unit_test_framework for validation).
    if cfg.update_cache:
        kernel_input['K_cache.must_alias_input'] = kernel_input.pop('K_cache')
        kernel_input['V_cache.must_alias_input'] = kernel_input.pop('V_cache')
        golden_outputs['K_cache'] = golden_outputs.pop('K_cache_updated')
        golden_outputs['V_cache'] = golden_outputs.pop('V_cache_updated')

    def input_generator(test_config):
        return kernel_input

    custom_validation = ValidationArgs(
        golden_output=_make_cosine_validation(golden_outputs, tolerances, cfg.kv_quant, cfg.quantization_type)
    )

    framework = UnitTestFramework(
        test_manager=test_manager,
        kernel_entry=nki.jit(attention_block_tkg_kernel_test_wrapper),
        torch_ref=torch_ref_wrapper(AttentionBlockTkgTorchRef(cfg.lnc, kv_quant_dtype=cfg.kv_quant_dtype)),
        kernel_input_generator=input_generator,
        # Output shapes come from the torch ref, which needs the LOGICAL mask — use the un-banded
        # kernel_input, not the banded ki handed to the kernel (banding only changes the mask layout,
        # not output shapes).
        output_tensor_descriptor=lambda ki: _infer_output_shapes_and_dtypes(
            {k.removesuffix(".must_alias_input"): v for k, v in kernel_input.items()}, cfg.lnc
        ),
    )
    framework.run_test(
        test_config=None,
        compiler_args=CompilerArgs(
            logical_nc_config=cfg.lnc,
            enable_birsim=False,
            platform_target=platform_target,
        ),
        inference_args=replace(
            TKG_INFERENCE_ARGS,
            collective_ranks=1,
            enable_determinism_check=False,
        ),
        custom_validation_args=custom_validation,
    )


def _run_multi_rank_test(
    test_manager: Orchestrator,
    platform_target: Platforms,
    cfg: AttnBlkTestConfig,
    tolerances: dict,
):
    """Run multi-rank test (KVDP, CP, or combined KVDP+CP) using CollectiveUnitTestFramework.

    The torch reference receives the same per-rank input as the kernel,
    runs collectives internally, and generates golden data for each rank.
    """
    KVDP = cfg.KVDP
    CP = cfg.CP
    total_ranks = KVDP * CP

    # Replica groups
    if KVDP > 1 and CP > 1:
        # Combined KVDP+CP: use explicit groups if provided, otherwise generate default layout
        if cfg.kvdp_replica_group is not None and cfg.cp_replica_group is not None:
            kvdp_replica_group = ReplicaGroup(cfg.kvdp_replica_group)
            cp_replica_group = ReplicaGroup(cfg.cp_replica_group)
        else:
            # Default: KVDP groups are contiguous chunks of CP ranks
            # e.g. KVDP=2, CP=2: [[0,1], [2,3]] — ranks 0,1 are KVDP group 0
            kvdp_groups = [list(range(i * CP, (i + 1) * CP)) for i in range(KVDP)]
            kvdp_replica_group = ReplicaGroup(kvdp_groups)
            # CP groups span across KVDP groups: [[0,2], [1,3]]
            cp_groups = [list(range(j, total_ranks, CP)) for j in range(CP)]
            cp_replica_group = ReplicaGroup(cp_groups)
    elif KVDP > 1:
        replica_group_list = cfg.kvdp_replica_group if cfg.kvdp_replica_group is not None else [list(range(KVDP))]
        kvdp_replica_group = ReplicaGroup(replica_group_list)
        cp_replica_group = None
    else:
        kvdp_replica_group = None
        cp_replica_group = ReplicaGroup(cfg.cp_replica_group if cfg.cp_replica_group is not None else [list(range(CP))])

    # For strided KVDP groups, collective_ranks = total ranks across all groups
    if KVDP > 1 and CP <= 1 and cfg.kvdp_replica_group is not None:
        collective_ranks = sum(len(g) for g in cfg.kvdp_replica_group)
    else:
        collective_ranks = total_ranks

    # Map each global rank to its group-local index (KVDP_rank).
    # Example: ReplicaGroup([[0,8,16,24], [1,9,17,25], ...])
    #          kvdp_rank:     0,1, 2, 3    0,1, 2, 3   ...
    rank_to_kvdp_rank = {}
    if KVDP > 1:
        if KVDP > 1 and CP > 1:
            kvdp_groups_list = cfg.kvdp_replica_group if cfg.kvdp_replica_group is not None else kvdp_groups
        else:
            kvdp_groups_list = cfg.kvdp_replica_group if cfg.kvdp_replica_group is not None else [list(range(KVDP))]
        for group in kvdp_groups_list:
            for kvdp_rank, global_rank in enumerate(group):
                rank_to_kvdp_rank[global_rank] = kvdp_rank
    else:
        for r in range(total_ranks):
            rank_to_kvdp_rank[r] = 0

    # Map each global rank to its CP rank (position within its CP group)
    rank_to_cp_rank = {}
    if CP > 1:
        if cfg.cp_replica_group is not None:
            cp_groups_list = cfg.cp_replica_group
        elif KVDP > 1:
            cp_groups_list = [list(range(j, total_ranks, CP)) for j in range(CP)]
        else:
            cp_groups_list = [list(range(CP))]
        for group in cp_groups_list:
            for cp_rank, global_rank in enumerate(group):
                rank_to_cp_rank[global_rank] = cp_rank

    # Generate one shared input set — X, cos/sin, norms, scales are identical across ranks.
    shared_input = generate_kernel_inputs(cfg)

    # Per-rank: weights (sharded Q heads), cache/mask/positions.
    def create_per_rank_input(rank_id):
        per_rank = generate_kernel_inputs(cfg, seed=rank_id)
        result = shared_input.copy()
        # Per-rank: sharded weights and batch-sliced tensors
        for key in (
            'W_qkv',
            'bias_qkv',
            'W_out',
            'weight_dequant_scale_qkv',
            'weight_dequant_scale_out',
            'K_cache',
            'V_cache',
            'attention_mask',
            'active_blocks_table',
            'kv_cache_update_idx',
            'pos_ids',
            'swa_start_pos_ids',
        ):
            result[key] = per_rank[key]

        # CP cache sharding: each rank gets S_ctx/CP portion
        if CP > 1:
            cp_rank = rank_to_cp_rank[rank_id]
            block_len = cfg.block_len
            cp_interleave = cfg.cp_interleave_size if cfg.cp_interleave_size is not None else block_len
            B_attn_cp = cfg.batch // cfg.KVDP if cfg.KVDP > 1 else cfg.batch
            if block_len == 0:
                # Flat KV: contiguous shard + cache/mask padding for the active-token overwrite.
                rank_inputs = _build_cp_flat_rank_inputs(
                    shared_input, cp_rank, CP, cfg.S_ctx, cfg.S_tkg, B_attn_cp, cfg.lnc, cfg.K_cache_transposed
                )
                for _k, _v in rank_inputs.items():
                    result[_k] = _v
            elif cp_interleave < block_len:
                # Block KV with sub-block (per-token) interleaving: a rank's owned tokens are
                # scattered across physical blocks, so gather them into a fresh local block cache.
                rank_inputs = _build_cp_block_kv_rank_inputs(
                    shared_input,
                    cp_rank,
                    CP,
                    block_len,
                    cfg.S_ctx,
                    cfg.S_tkg,
                    B_attn_cp,
                    cp_interleave,
                    cfg.lnc,
                    kv_heads=cfg.kv_heads,
                )
                for _k, _v in rank_inputs.items():
                    result[_k] = _v
            else:
                # Block KV, whole-block interleave: keep the full cache pool, slice both the
                # active_blocks_table and the mask to this rank's round-robin blocks.
                # Table is [B, num_blocks] (kv_heads==1) or [B, kv_heads, num_blocks] (kv_heads>1);
                # the block axis (logical position) is the LAST axis in both cases, and the table
                # values are global indices into the shared B*kv_heads pool — slicing the block axis
                # leaves those global indices intact.
                num_blocks_per_batch = cfg.S_ctx // block_len
                rank_logical_blocks = list(range(cp_rank, num_blocks_per_batch, CP))
                full_table = shared_input['active_blocks_table']
                if full_table.ndim == 3:
                    result['active_blocks_table'] = full_table[:, :, rank_logical_blocks].copy()
                else:
                    result['active_blocks_table'] = full_table[:, rank_logical_blocks].copy()
                # Use shared KV cache (full block pool) — table indexes into it
                result['K_cache'] = shared_input['K_cache']
                result['V_cache'] = shared_input['V_cache']
                # Flatten the block-KV mask, select the rank's block positions, then re-block —
                # shared helpers with the sub-block path (_build_cp_block_kv_rank_inputs).
                if not cfg.use_pos_id:
                    mask_flat = _flatten_block_kv_mask(shared_input['attention_mask'], block_len, cfg.lnc)
                    positions = []
                    for blk_idx in rank_logical_blocks:
                        blk_start = blk_idx * block_len
                        positions.extend(range(blk_start, blk_start + block_len))
                    rank_mask = _reblock_cp_mask(mask_flat[:, :, :, positions], block_len, cfg.lnc)
                    result['attention_mask'] = dt.static_cast(np.ascontiguousarray(rank_mask), dtype=np.uint8)

        # KVDP params
        if KVDP > 1:
            result['KVDP'] = KVDP
            result['KVDP_replica_group'] = kvdp_replica_group
            result['KVDP_rank'] = np.array([rank_to_kvdp_rank[rank_id]], dtype=np.uint32)
        # CP params
        if CP > 1:
            result['CP'] = CP
            result['CP_replica_group'] = cp_replica_group
            if cfg.CP_collective_mode is not None:
                result['CP_collective_mode'] = cfg.CP_collective_mode
        if cfg.update_cache:
            result['K_cache.must_alias_input'] = result.pop('K_cache')
            result['V_cache.must_alias_input'] = result.pop('V_cache')
        return result

    # Dtypes for golden comparison
    x_dtype = shared_input['X'].dtype
    kv_dtype = shared_input['K_cache'].dtype
    output_dtypes = {
        "X_out": x_dtype,
        "K_tkg": kv_dtype,
        "V_tkg": kv_dtype,
        "K_cache_updated": kv_dtype,
        "V_cache_updated": kv_dtype,
    }

    def comparator(rank_id, golden_dict):
        # Cast from float32 back to kernel IO dtypes (same as _golden_ref_via_torch)
        rank_golden = {k: v.astype(output_dtypes[k]) for k, v in golden_dict.items()}
        # Torch ref handles KVDP collectives internally — golden is already per-rank.
        if cfg.update_cache:
            rank_golden['K_cache'] = rank_golden.pop('K_cache_updated')
            rank_golden['V_cache'] = rank_golden.pop('V_cache_updated')
        return _make_cosine_validation(
            rank_golden, tolerances, cfg.kv_quant, cfg.quantization_type, name_prefix=f"rank{rank_id}:"
        )

    framework = CollectiveUnitTestFramework(
        test_manager=test_manager,
        kernel_entry=nki.jit(attention_block_tkg_kernel_test_wrapper),
        # preserve_lower_precision=True so bf16 KV reaches the ref as torch.bfloat16 (not the float32 upcast),
        # letting its is_qk_swapped call read the true 2-byte KV dtype for the swap-mask layout decision.
        # (fp8 still arrives as float32 by the wrapper's existing contract.) The comparator's per-output
        # .astype re-casts to the exact kernel dtypes, so the wrapper's cast-back does not change golden
        # values.
        torch_ref=torch_ref_wrapper(
            AttentionBlockTkgTorchRef(cfg.lnc, kv_quant_dtype=cfg.kv_quant_dtype), preserve_lower_precision=True
        ),
        per_rank_input_generator=create_per_rank_input,
        collective_ranks=collective_ranks,
    )
    golden_only = os.environ.get("GOLDEN_ONLY", "0") == "1"
    output_keys = ["X_out", "K_cache_updated", "V_cache_updated"] if cfg.update_cache else ["X_out", "K_tkg", "V_tkg"]
    framework.run_test(
        test_config=None,
        compiler_args=CompilerArgs(
            logical_nc_config=cfg.lnc,
            enable_birsim=False,
            platform_target=platform_target,
        ),
        inference_args=replace(TKG_INFERENCE_ARGS, collective_ranks=collective_ranks, enable_determinism_check=False),
        custom_comparator=comparator,
        golden_only=golden_only,
        output_keys=output_keys,
    )


@lru_cache(maxsize=1)
def _get_attention_block_metadata():
    return load_model_configs("test_attention_block")


# fmt: off
RANGE_ATTN_BLK_CFGS = [
    # SBUF IO
    AttnBlkTestConfig(batch=4, q_heads=8, d_head=64, H=6144, H_actual=2880, S_ctx=11264, S_max_ctx=11264, S_tkg=1,
                        output_in_sb=True),
    AttnBlkTestConfig(batch=4, q_heads=8, d_head=64, H=6144, H_actual=2880, S_ctx=11264, S_max_ctx=11264, S_tkg=1,
                        transposed_out=True, output_in_sb=True),
    AttnBlkTestConfig(batch=4, q_heads=8, d_head=64, H=6144, H_actual=2880, S_ctx=11264, S_max_ctx=11264, S_tkg=2,
                        update_cache=False, rmsnorm_X=False, skip_rope=True, input_in_sb=True, output_in_sb=True),
    # HBM IO
    AttnBlkTestConfig(batch=4, q_heads=8, d_head=64, H=6144, H_actual=2880, S_ctx=11264, S_max_ctx=11264, S_tkg=1),
    AttnBlkTestConfig(batch=4, q_heads=8, d_head=128, H=6144, H_actual=2880, S_ctx=11264, S_max_ctx=11264, S_tkg=1,
                        qk_norm_pre_rope=True, qk_norm_post_rope=True, qk_norm_post_rope_gamma=True, test_bias=True),
    AttnBlkTestConfig(batch=4, q_heads=1, d_head=128, H=6144, H_actual=2880, S_ctx=10240, S_max_ctx=10240, S_tkg=5,
                        update_cache=False),
    # GPT OSS RIV'25
    AttnBlkTestConfig(batch=8, q_heads=8, d_head=64, H=6144, H_actual=2880, S_ctx=11264, S_max_ctx=11264, S_tkg=1,
                        test_bias=True),
    AttnBlkTestConfig(batch=8, q_heads=8, d_head=64, H=3072, H_actual=2880, S_ctx=11264, S_max_ctx=11264, S_tkg=5,
                        test_bias=True),
    AttnBlkTestConfig(batch=8, q_heads=8, d_head=64, H=3072, H_actual=2880, S_ctx=10240, S_max_ctx=10240, S_tkg=5,
                        test_bias=True),
    AttnBlkTestConfig(batch=4, q_heads=8, d_head=64, H=3072, H_actual=None, S_ctx=10240, S_max_ctx=10240, S_tkg=4,
                        K_cache_transposed=True, test_bias=True),
    AttnBlkTestConfig(batch=4, q_heads=8, d_head=64, H=3072, H_actual=2880, S_ctx=10240, S_max_ctx=10240, S_tkg=4,
                        K_cache_transposed=True, test_bias=True),
    # MXFP case (QKV and Ouput_Projection are in MXFP)
    # MXFP, KVDP=1
    AttnBlkTestConfig(batch=8, q_heads=2, d_head=64, H=6144, H_actual=2880, S_ctx=131072, S_max_ctx=131072, S_tkg=1,
                      block_len=64, test_bias=True, cache_lens_mean=0.2, cache_lens_stddev=0.01, kv_quant=False,
                      quantization_type=QuantizationType.MX, supported_platforms={Platforms.TRN3, Platforms.TRN3_A0},
                      xfail_reason="alt-emax input distribution flips peaky softmax argmax at long context"),
    # MXPF, KVDP>1
    AttnBlkTestConfig(batch=8, q_heads=2, d_head=64, H=6144, H_actual=2880, S_ctx=131072, S_max_ctx=131072, S_tkg=1,
                      block_len=64, test_bias=True, cache_lens_mean=0.2, cache_lens_stddev=0.01, KVDP=4, kv_quant=False,
                      quantization_type=QuantizationType.MX, supported_platforms={Platforms.TRN3, Platforms.TRN3_A0},
                      xfail_reason="alt-emax input distribution flips peaky softmax argmax at long context"),
    # High-heads model.
    # BF16 versions of the same config.
    AttnBlkTestConfig(batch=128, q_heads=32, d_head=64, H=6144, H_actual=2880, S_ctx=131072, S_max_ctx=131072, S_tkg=1, 
                      block_len=64, test_bias=True, cache_lens_mean=0.2, cache_lens_stddev=0.01, KVDP=4, kv_quant=False),
    AttnBlkTestConfig(batch=128, q_heads=64, d_head=64, H=6144, H_actual=2880, S_ctx=131072, S_max_ctx=131072, S_tkg=1, 
                      block_len=64, test_bias=True, cache_lens_mean=0.2, cache_lens_stddev=0.01, KVDP=4, kv_quant=False),
    # Passing MXFP case so far. 
    AttnBlkTestConfig(batch=8, q_heads=16, d_head=64, H=6144, H_actual=None, S_ctx=131072, S_max_ctx=131072, S_tkg=1,
                      block_len=64, test_bias=True, cache_lens_mean=0.2, cache_lens_stddev=0.01, kv_quant=False,
                      quantization_type=QuantizationType.MX, supported_platforms={Platforms.TRN3, Platforms.TRN3_A0}),
    # Qwen3
    AttnBlkTestConfig(batch=16, q_heads=1, d_head=128, H=4096, H_actual=None, S_ctx=10240, S_max_ctx=10240, S_tkg=1,
                        qk_norm_pre_rope=True),
    # Qwen3 with pre-rope gamma weights
    AttnBlkTestConfig(batch=16, q_heads=1, d_head=128, H=4096, H_actual=None, S_ctx=10240, S_max_ctx=10240, S_tkg=1,
                        update_cache=False, qk_norm_pre_rope=True, qk_norm_pre_rope_gamma=True),
    # Gemma3 with pre-rope gamma weights
    AttnBlkTestConfig(batch=1, q_heads=1, d_head=128, H=5376, H_actual=None, S_ctx=1024, S_max_ctx=1024, S_tkg=1,
                        update_cache=False, qk_norm_pre_rope=True, qk_norm_pre_rope_gamma=True),
    AttnBlkTestConfig(batch=8, q_heads=4, d_head=128, H=5376, H_actual=None, S_ctx=10240, S_max_ctx=10240, S_tkg=3,
                        update_cache=False, qk_norm_pre_rope=True, qk_norm_pre_rope_gamma=True, test_bias=True),
    AttnBlkTestConfig(batch=1, q_heads=1, d_head=128, H=5376, H_actual=None, S_ctx=1024, S_max_ctx=1024, S_tkg=1,
                        block_len=16, update_cache=False, qk_norm_pre_rope=True, qk_norm_pre_rope_gamma=True),
    # New model, 2025-Jul
    AttnBlkTestConfig(batch=32, q_heads=1, d_head=64, H=3072, H_actual=None, S_ctx=8192, S_max_ctx=8192, S_tkg=1,
                        rmsnorm_X=False, test_bias=True),
    AttnBlkTestConfig(batch=32, q_heads=1, d_head=64, H=3072, H_actual=None, S_ctx=8192, S_max_ctx=8192, S_tkg=1,
                        K_cache_transposed=True, rmsnorm_X=False, test_bias=True),
    AttnBlkTestConfig(batch=64, q_heads=1, d_head=64, H=3072, H_actual=None, S_ctx=8192, S_max_ctx=8192, S_tkg=1,
                        K_cache_transposed=True, rmsnorm_X=False, test_bias=True),
    AttnBlkTestConfig(batch=32, q_heads=1, d_head=64, H=3072, H_actual=None, S_ctx=128, S_max_ctx=128, S_tkg=1,
                        rmsnorm_X=False, test_bias=True),
    AttnBlkTestConfig(batch=32, q_heads=1, d_head=64, H=3072, H_actual=None, S_ctx=128, S_max_ctx=128, S_tkg=1,
                        K_cache_transposed=True, rmsnorm_X=False, test_bias=True),
    AttnBlkTestConfig(batch=64, q_heads=1, d_head=64, H=3072, H_actual=None, S_ctx=128, S_max_ctx=128, S_tkg=1,
                        K_cache_transposed=True, rmsnorm_X=False, test_bias=True),
    AttnBlkTestConfig(batch=1, q_heads=1, d_head=64, H=3072, H_actual=None, S_ctx=128, S_max_ctx=128, S_tkg=1,
                        K_cache_transposed=True, rmsnorm_X=False, test_bias=True),
    AttnBlkTestConfig(batch=4, q_heads=8, d_head=64, H=3072, H_actual=None, S_ctx=8192, S_max_ctx=8192, S_tkg=1,
                        K_cache_transposed=True, rmsnorm_X=False, test_bias=True),
    AttnBlkTestConfig(batch=8, q_heads=8, d_head=64, H=3072, H_actual=None, S_ctx=8192, S_max_ctx=8192, S_tkg=1,
                        K_cache_transposed=True, rmsnorm_X=False, test_bias=True),
    AttnBlkTestConfig(batch=16, q_heads=8, d_head=64, H=3072, H_actual=None, S_ctx=8192, S_max_ctx=8192, S_tkg=1,
                        K_cache_transposed=True, rmsnorm_X=False, test_bias=True),
    AttnBlkTestConfig(batch=32, q_heads=1, d_head=64, H=3072, H_actual=None, S_ctx=8192, S_max_ctx=8192, S_tkg=3,
                        K_cache_transposed=True, rmsnorm_X=False, transposed_out=True, test_bias=True),
    AttnBlkTestConfig(batch=4, q_heads=8, d_head=64, H=3072, H_actual=None, S_ctx=8192, S_max_ctx=8192, S_tkg=2,
                        K_cache_transposed=True, rmsnorm_X=False, transposed_out=True, test_bias=True),
    # secret text
    AttnBlkTestConfig(batch=1, q_heads=1, d_head=128, H=7168, H_actual=None, S_ctx=256, S_max_ctx=256, S_tkg=1,
                        K_cache_transposed=True, qk_norm_post_rope=True, qk_norm_post_rope_gamma=True),
    # llama
    AttnBlkTestConfig(batch=1, q_heads=1, d_head=128, H=8192, H_actual=None, S_ctx=8192, S_max_ctx=8192, S_tkg=1,
                        K_cache_transposed=True, rmsnorm_X=False, qk_norm_post_rope=True),
    AttnBlkTestConfig(batch=1, q_heads=1, d_head=128, H=8192, H_actual=None, S_ctx=8192, S_max_ctx=8192, S_tkg=1,
                        K_cache_transposed=True, rmsnorm_X=False, skip_rope=True),
    AttnBlkTestConfig(batch=1, q_heads=1, d_head=128, H=8192, H_actual=None, S_ctx=8192, S_max_ctx=8192, S_tkg=1,
                        K_cache_transposed=True, rmsnorm_X=False),
    AttnBlkTestConfig(batch=1, q_heads=1, d_head=128, H=8192, H_actual=None, S_ctx=8192, S_max_ctx=8192, S_tkg=1,
                        K_cache_transposed=True, rmsnorm_X=False, rope_contiguous_layout=False),
    AttnBlkTestConfig(batch=1, q_heads=1, d_head=128, H=5120, H_actual=None, S_ctx=8192, S_max_ctx=8192, S_tkg=1,
                        K_cache_transposed=True, rope_contiguous_layout=False),
    AttnBlkTestConfig(batch=1, q_heads=1, d_head=128, H=5120, H_actual=None, S_ctx=8192, S_max_ctx=8192, S_tkg=1,
                        K_cache_transposed=True, skip_rope=True, rope_contiguous_layout=False),
    AttnBlkTestConfig(batch=4, q_heads=1, d_head=128, H=8192, H_actual=None, S_ctx=10240, S_max_ctx=16384, S_tkg=5,
                        K_cache_transposed=True),
    AttnBlkTestConfig(batch=4, q_heads=1, d_head=128, H=8192, H_actual=None, S_ctx=10240, S_max_ctx=10240, S_tkg=5,
                        K_cache_transposed=True),
    AttnBlkTestConfig(batch=8, q_heads=2, d_head=128, H=16384, H_actual=None, S_ctx=2048, S_max_ctx=2048, S_tkg=7,
                        K_cache_transposed=True),
    AttnBlkTestConfig(batch=1, q_heads=16, d_head=128, H=16384, H_actual=None, S_ctx=4096, S_max_ctx=8192, S_tkg=7,
                        rmsnorm_X=False),
    AttnBlkTestConfig(batch=1, q_heads=16, d_head=128, H=16384, H_actual=None, S_ctx=4096, S_max_ctx=8192, S_tkg=7,
                        rmsnorm_X=False, rope_contiguous_layout=False),
    AttnBlkTestConfig(batch=8, q_heads=2, d_head=128, H=16384, H_actual=None, S_ctx=2048, S_max_ctx=2048, S_tkg=7,
                        K_cache_transposed=True, transposed_out=True),
    # Test vectors for block KV
    AttnBlkTestConfig(batch=4, q_heads=1, d_head=128, H=8192, H_actual=None, S_ctx=256, S_max_ctx=256, S_tkg=5,
                        block_len=16),
    AttnBlkTestConfig(batch=4, q_heads=1, d_head=128, H=8192, H_actual=None, S_ctx=8192, S_max_ctx=8192, S_tkg=5,
                        block_len=16),
    AttnBlkTestConfig(batch=4, q_heads=1, d_head=128, H=8192, H_actual=None, S_ctx=12288, S_max_ctx=12288, S_tkg=5,
                        block_len=16),
    AttnBlkTestConfig(batch=4, q_heads=1, d_head=128, H=8192, H_actual=None, S_ctx=10240, S_max_ctx=10240, S_tkg=5,
                        block_len=16),
    # 4D block-KV cache layout [blocks, 1, block_len, d_head] (single head at axis 1).
    # Exercises __internal_squeeze_head_dim/unsqueeze in attention_block_tkg.
    AttnBlkTestConfig(batch=4, q_heads=1, d_head=128, H=8192, H_actual=None, S_ctx=256, S_max_ctx=256, S_tkg=5,
                        block_len=16, cache_has_kv_head_dim=True),
    AttnBlkTestConfig(batch=4, q_heads=1, d_head=128, H=8192, H_actual=None, S_ctx=8192, S_max_ctx=8192, S_tkg=5,
                        block_len=16, cache_has_kv_head_dim=True),
    # Block boundary crossing: S_tkg=17 > block_len=16 guarantees tokens span
    # at least two blocks regardless of starting position within a block.
    AttnBlkTestConfig(batch=8, q_heads=1, d_head=128, H=8192, H_actual=None, S_ctx=128, S_max_ctx=128, S_tkg=17,
                        block_len=16),
    # BxS > pmax (128): exercises multi-iteration tiling loop in _update_block_cache_scalar
    AttnBlkTestConfig(batch=32, q_heads=1, d_head=128, H=8192, H_actual=None, S_ctx=256, S_max_ctx=256, S_tkg=5,
                        block_len=16),
    AttnBlkTestConfig(batch=64, q_heads=1, d_head=128, H=8192, H_actual=None, S_ctx=128, S_max_ctx=128, S_tkg=3,
                        block_len=16),
    # Test vectors to verify functionality of different q_heads, d_head and H dimensions
    AttnBlkTestConfig(batch=2, q_heads=1, d_head=128, H=2048, H_actual=None, S_ctx=10240, S_max_ctx=16384, S_tkg=5,
                        K_cache_transposed=True),
    AttnBlkTestConfig(batch=2, q_heads=1, d_head=64, H=2048, H_actual=None, S_ctx=10240, S_max_ctx=16384, S_tkg=5,
                        K_cache_transposed=True),
    AttnBlkTestConfig(batch=2, q_heads=2, d_head=64, H=3072, H_actual=None, S_ctx=10240, S_max_ctx=16384, S_tkg=5),
    AttnBlkTestConfig(batch=2, q_heads=3, d_head=64, H=4096, H_actual=None, S_ctx=10240, S_max_ctx=16384, S_tkg=5,
                        update_cache=False),
    AttnBlkTestConfig(batch=2, q_heads=4, d_head=128, H=6144, H_actual=None, S_ctx=10240, S_max_ctx=16384, S_tkg=5,
                        update_cache=False, K_cache_transposed=True),
    AttnBlkTestConfig(batch=2, q_heads=3, d_head=128, H=20480, H_actual=None, S_ctx=10240, S_max_ctx=16384, S_tkg=5,
                        K_cache_transposed=True),
    AttnBlkTestConfig(batch=4, q_heads=1, d_head=128, H=8192, H_actual=None, S_ctx=10240, S_max_ctx=10240, S_tkg=5,
                        K_cache_transposed=True),
    # static quantization tests
    # TODO: random input causing numerical instability for quantized weights, more tests will be added
    # after better fp8 random generator is implemented
    # E2E inference tests shows good accuracy
    AttnBlkTestConfig(batch=8, q_heads=1, d_head=128, H=8192, H_actual=None, S_ctx=2048, S_max_ctx=2048, S_tkg=5,
                        K_cache_transposed=True, quantization_type=QuantizationType.STATIC),
    # row-wise quantization tests
    AttnBlkTestConfig(batch=8, q_heads=1, d_head=128, H=8192, H_actual=None, S_ctx=2048, S_max_ctx=2048, S_tkg=5,
                        K_cache_transposed=True, quantization_type=QuantizationType.ROW),
    AttnBlkTestConfig(batch=8, q_heads=1, d_head=128, H=8192, H_actual=None, S_ctx=2048, S_max_ctx=2048, S_tkg=5,
	                    K_cache_transposed=True, quantization_type=QuantizationType.ROW, kv_quant=True),
    # MX quantization tests
    AttnBlkTestConfig(batch=4, q_heads=1, d_head=128, H=8192, H_actual=None, S_ctx=2048, S_max_ctx=2048, S_tkg=1,
                        K_cache_transposed=True, quantization_type=QuantizationType.MX, supported_platforms={Platforms.TRN3, Platforms.TRN3_A0}),
    AttnBlkTestConfig(batch=4, q_heads=1, d_head=128, H=8192, H_actual=None, S_ctx=2048, S_max_ctx=2048, S_tkg=1,
                        K_cache_transposed=True, quantization_type=QuantizationType.STATIC_MX, supported_platforms={Platforms.TRN3, Platforms.TRN3_A0}),
    # ROW_MX not yet supported, requires support in output_projection_tkg.
    # AttnBlkTestConfig(batch=4, q_heads=1, d_head=128, H=8192, H_actual=None, S_ctx=2048, S_max_ctx=2048, S_tkg=1,
    #                     K_cache_transposed=True, quantization_type=QuantizationType.ROW_MX, supported_platforms={Platforms.TRN3, Platforms.TRN3_A0}),
    # softmax_scale tests (Gemma model support)
    AttnBlkTestConfig(batch=4, q_heads=8, d_head=64, H=3072, H_actual=2880, S_ctx=10240, S_max_ctx=10240, S_tkg=4,
                        K_cache_transposed=True, test_bias=True, softmax_scale=0.05),
    AttnBlkTestConfig(batch=1, q_heads=1, d_head=128, H=7168, H_actual=None, S_ctx=256, S_max_ctx=256, S_tkg=1,
                        K_cache_transposed=True, qk_norm_post_rope=True, qk_norm_post_rope_gamma=True, softmax_scale=0.09),
    AttnBlkTestConfig(batch=1, q_heads=1, d_head=128, H=5120, H_actual=None, S_ctx=8192, S_max_ctx=8192, S_tkg=1,
                        K_cache_transposed=True, skip_rope=True, rope_contiguous_layout=False, softmax_scale=0.13),
    AttnBlkTestConfig(batch=4, q_heads=1, d_head=128, H=8192, H_actual=None, S_ctx=8192, S_max_ctx=8192, S_tkg=5,
                        block_len=16, softmax_scale=0.17),
    AttnBlkTestConfig(batch=4, q_heads=1, d_head=128, H=8192, H_actual=None, S_ctx=10240, S_max_ctx=10240, S_tkg=5,
                        K_cache_transposed=True, softmax_scale=0.21),
    # llama FP8 KV Cache Tests
    AttnBlkTestConfig(batch=2, q_heads=1, d_head=128, H=8192, H_actual=None, S_ctx=8192, S_max_ctx=8192, S_tkg=1,
                        rmsnorm_X=False, qk_norm_pre_rope=True, qk_norm_pre_rope_gamma=True, kv_quant=True, kv_scale=KVScaleTest.DEFAULT),
    AttnBlkTestConfig(batch=37, q_heads=1, d_head=128, H=8192, H_actual=None, S_ctx=8192, S_max_ctx=8192, S_tkg=1,
                        K_cache_transposed=True, rmsnorm_X=False, kv_quant=True, kv_scale=KVScaleTest.DEFAULT),
    AttnBlkTestConfig(batch=96, q_heads=1, d_head=128, H=8192, H_actual=None, S_ctx=8192, S_max_ctx=8192, S_tkg=1,
                        K_cache_transposed=True, rmsnorm_X=False, kv_quant=True, kv_scale=KVScaleTest.DEFAULT),
    # llama FP8 KV Cache Tests - batched cache update
    AttnBlkTestConfig(batch=32, q_heads=1, d_head=128, H=8192, H_actual=None, S_ctx=4096, S_max_ctx=4096, S_tkg=1,
                        rmsnorm_X=False, kv_quant=True, kv_scale=KVScaleTest.DEFAULT),
    AttnBlkTestConfig(batch=128, q_heads=1, d_head=128, H=8192, H_actual=None, S_ctx=2048, S_max_ctx=2048, S_tkg=1,
                        rmsnorm_X=False, kv_quant=True, kv_scale=KVScaleTest.DEFAULT),
    # llama FP8 KV Cache Tests - block KV cache
    AttnBlkTestConfig(batch=16, q_heads=1, d_head=128, H=8192, H_actual=None, S_ctx=2048, S_max_ctx=2048, S_tkg=1,
                        block_len=16, rmsnorm_X=False, kv_quant=True, kv_scale=KVScaleTest.DEFAULT),
    AttnBlkTestConfig(batch=32, q_heads=1, d_head=128, H=8192, H_actual=None, S_ctx=2048, S_max_ctx=2048, S_tkg=1,
                        block_len=16, rmsnorm_X=False, kv_quant=True, kv_scale=KVScaleTest.DEFAULT),
    AttnBlkTestConfig(batch=64, q_heads=1, d_head=128, H=8192, H_actual=None, S_ctx=2048, S_max_ctx=2048, S_tkg=1,
                        block_len=16, rmsnorm_X=False, kv_quant=True, kv_scale=KVScaleTest.DEFAULT),
    # FP8 KV cache direct cast (kv_scale=1.0)
    AttnBlkTestConfig(batch=1, q_heads=2, d_head=128, H=8192, H_actual=None, S_ctx=26624, S_max_ctx=36896, S_tkg=5,
                        block_len=32, kv_quant=True, kv_scale=1.0, enable_fa_s_prior_tiling=False),
    # FP8 KV cache - finite (float8_e4m3fn)
    AttnBlkTestConfig(batch=16, q_heads=1, d_head=128, H=8192, H_actual=None, S_ctx=2048, S_max_ctx=2048, S_tkg=1,
                        block_len=16, rmsnorm_X=False, kv_quant=True, kv_quant_dtype=nl.float8_e4m3fn,
                        kv_scale=104.0, supported_platforms={Platforms.TRN3}),
    AttnBlkTestConfig(batch=1, q_heads=2, d_head=128, H=8192, H_actual=None, S_ctx=26624, S_max_ctx=36896, S_tkg=5,
                        block_len=32, kv_quant=True, kv_quant_dtype=nl.float8_e4m3fn, kv_scale=1.0,
                        enable_fa_s_prior_tiling=False, supported_platforms={Platforms.TRN3}),
    # Long context tests (S_ctx >= 128k, slower)
    # flat KV S_ctx=128k
    AttnBlkTestConfig(batch=8, q_heads=1, d_head=64, H=3072, H_actual=2880, S_ctx=131072, S_max_ctx=131072, S_tkg=1,
                        rmsnorm_X=False, test_bias=True),
    # flat KV S_ctx=512k
    AttnBlkTestConfig(batch=8, q_heads=1, d_head=64, H=3072, H_actual=2880, S_ctx=524288, S_max_ctx=524288, S_tkg=1,
                        rmsnorm_X=False, test_bias=True),
    # block KV S_ctx=128k, block_len=32
    AttnBlkTestConfig(batch=8, q_heads=1, d_head=64, H=3072, H_actual=2880, S_ctx=131072, S_max_ctx=131072, S_tkg=1,
                        block_len=32, rmsnorm_X=False, test_bias=True),
    # block KV S_ctx=128k, block_len=32, cache_lens tightly concentrated around 20%
    AttnBlkTestConfig(batch=8, q_heads=1, d_head=64, H=3072, H_actual=2880, S_ctx=131072, S_max_ctx=131072, S_tkg=1,
                        block_len=32, rmsnorm_X=False, test_bias=True, cache_lens_mean=0.2, cache_lens_stddev=0.01),
    # block KV S_ctx=512k, block_len=32
    AttnBlkTestConfig(batch=8, q_heads=1, d_head=64, H=3072, H_actual=2880, S_ctx=524288, S_max_ctx=524288, S_tkg=1,
                        block_len=32, rmsnorm_X=False, test_bias=True),
    # KVDP tests (KVDP=4, GPT-OSS-like)
    # q_heads=1 means each rank has 1 q_head, total KVDP * q_heads across ranks
    # flat KV S_ctx=1k B=8
    AttnBlkTestConfig(batch=8, q_heads=1, d_head=64, H=3072, H_actual=2880, S_ctx=1024, S_max_ctx=1024, S_tkg=1,
                        test_bias=True, KVDP=4, kv_quant=True),
    # update_cache=False for complete API coverage
    AttnBlkTestConfig(batch=8, q_heads=1, d_head=64, H=3072, H_actual=2880, S_ctx=1024, S_max_ctx=1024, S_tkg=1,
                        test_bias=True, KVDP=4, kv_quant=True, update_cache=False),
    # q_heads=2 tests the general transpose path (q_heads>1)
    AttnBlkTestConfig(batch=8, q_heads=2, d_head=64, H=3072, H_actual=2880, S_ctx=1024, S_max_ctx=1024, S_tkg=1,
                        test_bias=True, KVDP=4, kv_quant=True),
    # block KV S_ctx=1k B=8, block_len=32
    AttnBlkTestConfig(batch=8, q_heads=1, d_head=64, H=3072, H_actual=2880, S_ctx=1024, S_max_ctx=1024, S_tkg=1,
                        block_len=32, rmsnorm_X=False, test_bias=True, KVDP=4, kv_quant=True),
    # block KV S_ctx=1k B=32, block_len=32
    AttnBlkTestConfig(batch=32, q_heads=1, d_head=64, H=3072, H_actual=2880, S_ctx=1024, S_max_ctx=1024, S_tkg=1,
                        block_len=32, rmsnorm_X=False, test_bias=True, KVDP=4, kv_quant=True),
    # KVDP + block KV + S_tkg > 1: exercises per-token KVDP slicing
    # S_tkg=17 > block_len=16 guarantees block boundary crossing for any seed.
    AttnBlkTestConfig(batch=8, q_heads=1, d_head=64, H=3072, H_actual=2880, S_ctx=1024, S_max_ctx=1024, S_tkg=17,
                        block_len=16, rmsnorm_X=False, test_bias=True, KVDP=4, kv_quant=True),
    # KVDP long context tests (S_ctx >= 128k, slower)
    # flat KV S_ctx=512k
    AttnBlkTestConfig(batch=8, q_heads=1, d_head=64, H=3072, H_actual=2880, S_ctx=524288, S_max_ctx=524288, S_tkg=1,
                        rmsnorm_X=False, test_bias=True, KVDP=4, kv_quant=True),
    # block KV S_ctx=512k, block_len=32
    AttnBlkTestConfig(batch=8, q_heads=1, d_head=64, H=3072, H_actual=2880, S_ctx=524288, S_max_ctx=524288, S_tkg=1,
                        block_len=32, rmsnorm_X=False, test_bias=True, KVDP=4, kv_quant=True),
    # flat KV S_ctx=1M
    AttnBlkTestConfig(batch=8, q_heads=1, d_head=64, H=3072, H_actual=2880, S_ctx=1048576, S_max_ctx=1048576, S_tkg=1,
                        rmsnorm_X=False, test_bias=True, KVDP=4, kv_quant=True),
    # block KV S_ctx=1M, block_len=32
    AttnBlkTestConfig(batch=8, q_heads=1, d_head=64, H=3072, H_actual=2880, S_ctx=1048576, S_max_ctx=1048576, S_tkg=1,
                        block_len=32, rmsnorm_X=False, test_bias=True, KVDP=4, kv_quant=True),
    # KVDP B=64 tests
    # block KV S_ctx=1k B=64, block_len=32
    AttnBlkTestConfig(batch=64, q_heads=1, d_head=64, H=3072, H_actual=2880, S_ctx=1024, S_max_ctx=1024, S_tkg=1,
                        block_len=32, rmsnorm_X=False, test_bias=True, KVDP=4, kv_quant=True),
    # block KV S_ctx=128k B=64, block_len=32
    AttnBlkTestConfig(batch=64, q_heads=1, d_head=64, H=3072, H_actual=2880, S_ctx=131072, S_max_ctx=131072, S_tkg=1,
                        block_len=32, rmsnorm_X=False, test_bias=True, KVDP=4, kv_quant=True),
    # flat KV S_ctx=128k B=64
    AttnBlkTestConfig(batch=64, q_heads=1, d_head=64, H=3072, H_actual=2880, S_ctx=131072, S_max_ctx=131072, S_tkg=1,
                        rmsnorm_X=False, test_bias=True, KVDP=4, kv_quant=True),
    # Small S_ctx=128: sprior_n_prgs=1 (not sharded)
    AttnBlkTestConfig(batch=1, q_heads=1, d_head=128, H=5120, H_actual=None, S_ctx=128, S_max_ctx=128, S_tkg=1,
                        block_len=32, rope_contiguous_layout=False, qk_norm_pre_rope=True, qk_norm_pre_rope_gamma=True),
    # Large batch + small S_ctx: batch-sharded so sprior_n_prgs=1
    AttnBlkTestConfig(batch=256, q_heads=1, d_head=128, H=5120, H_actual=None, S_ctx=128, S_max_ctx=128, S_tkg=1,
                        block_len=32, rope_contiguous_layout=False, qk_norm_pre_rope=True, qk_norm_pre_rope_gamma=True),
    # B=1 large block_len with FA tiling
    AttnBlkTestConfig(batch=1, q_heads=1, d_head=128, H=5120, H_actual=None, S_ctx=32768, S_max_ctx=32768, S_tkg=1,
                        block_len=128, rope_contiguous_layout=False, qk_norm_pre_rope=True, qk_norm_pre_rope_gamma=True),

    # ===== Large batch coverage (B*S_tkg > 128) =====
    ## Flat KV (block_len=0)
    AttnBlkTestConfig(batch=255, q_heads=2, d_head=128, H=8192, H_actual=None, S_ctx=1024, S_max_ctx=1024, S_tkg=3,
                        qk_norm_pre_rope=True, qk_norm_post_rope=True, qk_norm_post_rope_gamma=True, test_bias=True),
    AttnBlkTestConfig(batch=255, q_heads=2, d_head=64, H=8192, H_actual=None, S_ctx=1024, S_max_ctx=1024, S_tkg=3,
                        K_cache_transposed=True),
    AttnBlkTestConfig(batch=384, q_heads=3, d_head=128, H=5120, H_actual=None, S_ctx=2048, S_max_ctx=2048, S_tkg=1,
                        qk_norm_post_rope=True, qk_norm_post_rope_gamma=True, transposed_out=True),
    AttnBlkTestConfig(batch=384, q_heads=3, d_head=64, H=5120, H_actual=None, S_ctx=2048, S_max_ctx=2048, S_tkg=1,
                        K_cache_transposed=True, rmsnorm_X=False, kv_quant=True, kv_scale=KVScaleTest.DEFAULT),
    ## Block KV
    # FP8 KV quantization
    AttnBlkTestConfig(batch=255, q_heads=3, d_head=128, H=5120, H_actual=None, S_ctx=26624, S_max_ctx=36896, S_tkg=2,
                        block_len=32, kv_quant=True, kv_scale=1.0),
    # FP8 weight quantization
    AttnBlkTestConfig(batch=255, q_heads=2, d_head=128, H=8192, H_actual=None, S_ctx=1024, S_max_ctx=1024, S_tkg=1,
                        block_len=32, quantization_type=QuantizationType.STATIC),
    # d_head=64, q_heads>1
    AttnBlkTestConfig(batch=255, q_heads=8, d_head=64, H=3072, H_actual=None, S_ctx=2048, S_max_ctx=2048, S_tkg=1,
                        block_len=32),
    # No RoPE, update_cache=False, transposed_out, test_bias, softmax_scale
    AttnBlkTestConfig(batch=255, q_heads=3, d_head=128, H=5120, H_actual=None, S_ctx=10240, S_max_ctx=12288, S_tkg=1,
                        block_len=32, skip_rope=True, update_cache=False, transposed_out=True, test_bias=True, softmax_scale=0.05),
    # B*S_tkg > 256 (multi-tile with S_tkg>1)
    AttnBlkTestConfig(batch=128, q_heads=2, d_head=128, H=5120, H_actual=None, S_ctx=1024, S_max_ctx=1024, S_tkg=3,
                        block_len=32),
    # QK norm post-RoPE, rmsnorm_X=False (no input norm)
    AttnBlkTestConfig(batch=255, q_heads=1, d_head=128, H=5120, H_actual=None, S_ctx=1024, S_max_ctx=1024, S_tkg=1,
                        block_len=32, qk_norm_post_rope=True, qk_norm_post_rope_gamma=True, rmsnorm_X=False),
    # Large B*q_heads
    AttnBlkTestConfig(batch=129, q_heads=9, d_head=128, H=5120, H_actual=None, S_ctx=1024, S_max_ctx=1024, S_tkg=1,
                        block_len=32, qk_norm_pre_rope=True, qk_norm_pre_rope_gamma=True, qk_norm_post_rope=True,
                        qk_norm_post_rope_gamma=True),
    # Large B*S_tkg triggering multi-tile path (B*S_tkg > pmax)
    AttnBlkTestConfig(batch=64, q_heads=8, d_head=64, H=3072, H_actual=2880, S_ctx=16384, S_max_ctx=16384, S_tkg=3,
                        block_len=64, transposed_out=True, use_pos_id=True),
    # rope_contiguous_layout=False with large tile_B * q_heads (exercises gemm_moving_fmax in RoPE)
    AttnBlkTestConfig(batch=255, q_heads=8, d_head=64, H=8192, H_actual=None, S_ctx=32768, S_max_ctx=32768, S_tkg=5,
                        block_len=128, rope_contiguous_layout=False),
    # B=1024
    AttnBlkTestConfig(batch=1024, q_heads=2, d_head=128, H=5120, H_actual=None, S_ctx=1024, S_max_ctx=1024, S_tkg=1,
                        block_len=32, qk_norm_pre_rope=True, qk_norm_pre_rope_gamma=True),
    AttnBlkTestConfig(batch=1024, q_heads=2, d_head=128, H=5120, H_actual=None, S_ctx=1024, S_max_ctx=1024, S_tkg=1,
                       block_len=32, rmsnorm_X=False, qk_norm_pre_rope=True, qk_norm_pre_rope_gamma=True,
                       quantization_type=QuantizationType.STATIC),
    # Large batch + KVDP
    AttnBlkTestConfig(batch=256, q_heads=2, d_head=64, H=3072, H_actual=2880, S_ctx=10240, S_max_ctx=10240, S_tkg=1,
                        block_len=32, test_bias=True, KVDP=4),
    AttnBlkTestConfig(batch=288, q_heads=2, d_head=128, H=5120, H_actual=4096, S_ctx=1024, S_max_ctx=1024, S_tkg=1,
                        block_len=32, test_bias=True, KVDP=16),
    AttnBlkTestConfig(batch=512, q_heads=4, d_head=128, H=5120, H_actual=None, S_ctx=1024, S_max_ctx=1024, S_tkg=1,
                        block_len=32, qk_norm_pre_rope=True, qk_norm_pre_rope_gamma=True, KVDP=4),
    AttnBlkTestConfig(batch=2048, q_heads=8, d_head=64, H=3072, H_actual=2880, S_ctx=10240, S_max_ctx=10240, S_tkg=1,
                        block_len=128, KVDP=8, kv_quant=True),
    # Strided KVDP replica group: TP=8, KVDP=4, collective_ranks=32
    # kvdp_replica_group=[[0,8,16,24], [1,9,17,25], ..., [7,15,23,31]]
    AttnBlkTestConfig(batch=8, q_heads=1, d_head=64, H=3072, H_actual=2880, S_ctx=4096, S_max_ctx=4096, S_tkg=1,
                        rmsnorm_X=False, test_bias=True,
                        KVDP=4,
                        kvdp_replica_group=[[tp + d * 8 for d in range(4)] for tp in range(8)]),
    # KV-DP ALL_GATHER_SLICE regression tests (KVDP=4, S_ctx=1k)
    # flat KV q_heads=1
    AttnBlkTestConfig(batch=8, q_heads=1, d_head=64, H=3072, H_actual=2880, S_ctx=1024, S_max_ctx=1024, S_tkg=1,
                        rmsnorm_X=False, test_bias=True, KVDP=4, KVDP_collective_mode=KVDPCollectiveMode.ALL_GATHER_SLICE),
    # flat KV q_heads=2
    AttnBlkTestConfig(batch=8, q_heads=2, d_head=64, H=3072, H_actual=2880, S_ctx=1024, S_max_ctx=1024, S_tkg=1,
                        rmsnorm_X=False, test_bias=True, KVDP=4, KVDP_collective_mode=KVDPCollectiveMode.ALL_GATHER_SLICE),
    # block KV q_heads=1
    AttnBlkTestConfig(batch=8, q_heads=1, d_head=64, H=3072, H_actual=2880, S_ctx=1024, S_max_ctx=1024, S_tkg=1,
                        block_len=32, rmsnorm_X=False, test_bias=True, KVDP=4, KVDP_collective_mode=KVDPCollectiveMode.ALL_GATHER_SLICE),
    # block KV q_heads=2
    AttnBlkTestConfig(batch=8, q_heads=2, d_head=64, H=3072, H_actual=2880, S_ctx=1024, S_max_ctx=1024, S_tkg=1,
                        block_len=32, rmsnorm_X=False, test_bias=True, KVDP=4, KVDP_collective_mode=KVDPCollectiveMode.ALL_GATHER_SLICE),
    # flat KV S_ctx=131k
    AttnBlkTestConfig(batch=8, q_heads=1, d_head=64, H=3072, H_actual=2880, S_ctx=131072, S_max_ctx=131072, S_tkg=1,
                        rmsnorm_X=False, test_bias=True, KVDP=4, KVDP_collective_mode=KVDPCollectiveMode.ALL_GATHER_SLICE),
    # block KV S_ctx=131k, block_len=32
    AttnBlkTestConfig(batch=8, q_heads=1, d_head=64, H=3072, H_actual=2880, S_ctx=131072, S_max_ctx=131072, S_tkg=1,
                        block_len=32, rmsnorm_X=False, test_bias=True, KVDP=4, KVDP_collective_mode=KVDPCollectiveMode.ALL_GATHER_SLICE),
    # large B q_heads=2: exercises tiled transpose (B*q_heads=512 > pmax)
    AttnBlkTestConfig(batch=256, q_heads=2, d_head=64, H=3072, H_actual=2880, S_ctx=1024, S_max_ctx=1024, S_tkg=1,
                        block_len=32, test_bias=True, KVDP=4, KVDP_collective_mode=KVDPCollectiveMode.ALL_GATHER_SLICE),

    # ===== Transposed in+out tests =====
    # Tests the [H0, n_prgs, H1_shard, BxS] HBM input layout with transposed output.
    # Covers: multiple models, d_head sizes, batch sizes, quant, qk_norm, softmax_scale, flat/block KV.
    # llama3_70b: B=1, block KV
    AttnBlkTestConfig(batch=1, q_heads=1, d_head=128, H=8192, H_actual=None, S_ctx=2048, S_max_ctx=2048, S_tkg=1,
                        block_len=32, transposed_in=True, transposed_out=True),
    # llama3_70b: B=16, block KV
    AttnBlkTestConfig(batch=16, q_heads=1, d_head=128, H=8192, H_actual=None, S_ctx=2048, S_max_ctx=2048, S_tkg=1,
                        block_len=32, transposed_in=True, transposed_out=True),
    # qwen3_32b: B=1, block KV, qk_norm_pre_rope, rmsnorm_X=False
    AttnBlkTestConfig(batch=1, q_heads=1, d_head=128, H=5120, H_actual=None, S_ctx=2048, S_max_ctx=2048, S_tkg=1,
                        block_len=32, rmsnorm_X=False, qk_norm_pre_rope=True, qk_norm_pre_rope_gamma=True,
                        transposed_in=True, transposed_out=True),
    # gptoss_120b: B=1, d_head=64, H_actual padding, block KV
    AttnBlkTestConfig(batch=1, q_heads=1, d_head=64, H=3072, H_actual=2880, S_ctx=2048, S_max_ctx=2048, S_tkg=1,
                        block_len=32, transposed_in=True, transposed_out=True),
    # qwen3_235b: B=1, flat KV, qk_norm_pre_rope
    AttnBlkTestConfig(batch=1, q_heads=1, d_head=128, H=4096, H_actual=None, S_ctx=2048, S_max_ctx=2048, S_tkg=1,
                        qk_norm_pre_rope=True, qk_norm_pre_rope_gamma=True,
                        transposed_in=True, transposed_out=True),
    # gemma3_27b: B=16, softmax_scale, rmsnorm_X=False, qk_norm_pre_rope, block KV
    AttnBlkTestConfig(batch=16, q_heads=1, d_head=128, H=5376, H_actual=None, S_ctx=2048, S_max_ctx=2048, S_tkg=1,
                        block_len=32, rmsnorm_X=False, qk_norm_pre_rope=True, qk_norm_pre_rope_gamma=True,
                        softmax_scale=0.07715167498, transposed_in=True, transposed_out=True),
    # llama3_70b: B=8, S_tkg=5, STATIC weight quant, K_cache_transposed
    AttnBlkTestConfig(batch=8, q_heads=1, d_head=128, H=8192, H_actual=None, S_ctx=2048, S_max_ctx=2048, S_tkg=5,
                        K_cache_transposed=True, quantization_type=QuantizationType.STATIC,
                        transposed_in=True, transposed_out=True),
    # llama3_70b TP=16: B=16, q_heads=4, block KV (block_len=32, S_ctx=1024)
    # First layer: BSH input, transposed output
    AttnBlkTestConfig(batch=16, q_heads=4, d_head=128, H=8192, H_actual=None, S_ctx=1024, S_max_ctx=1024, S_tkg=1,
                        block_len=32, transposed_out=True),
    # Last layer: transposed input, BSH output
    AttnBlkTestConfig(batch=16, q_heads=4, d_head=128, H=8192, H_actual=None, S_ctx=1024, S_max_ctx=1024, S_tkg=1,
                        block_len=32, transposed_in=True),

    # ===== In-kernel mask generation (use_pos_id=True) =====
    # QK-swap partition banding + in-kernel gen_mask: b16 batch-sharded -> bs_per_nc=8 < batches_per_psum=16,
    # s_active_qh=8, band_factor=2. Exercises the banded gen_mask_tkg swap layout (per-band token offset).
    # block_len=64 single-fold tile (S_ctx=8192) so the band split aligns to the fold's p_slot axis.
    AttnBlkTestConfig(batch=16, q_heads=8, d_head=64, H=3072, H_actual=2880, S_ctx=8192, S_max_ctx=8192, S_tkg=1,
                        block_len=64, kv_quant=True, fp8_packed=True, rmsnorm_X=False, use_pos_id=True),
    # QK-swap banding + S_PRIOR sharding (b8: bs_per_nc=8 -> band_factor=2, s_prior sharded across NCs).
    # Exercises a query whose cache_len is shorter than one NC's s_prior shard, so that shard is fully
    # masked -> per-position max stays -inf; the swap path must clamp it to finite (else exp -> NaN).
    AttnBlkTestConfig(batch=8, q_heads=8, d_head=64, H=3072, H_actual=2880, S_ctx=8192, S_max_ctx=8192, S_tkg=1,
                        block_len=64, kv_quant=True, fp8_packed=True, rmsnorm_X=False, use_pos_id=True),
    # Same, with an attention sink (sink fold on the s_prior-sharded banded swap path).
    AttnBlkTestConfig(batch=8, q_heads=8, d_head=64, H=3072, H_actual=2880, S_ctx=8192, S_max_ctx=8192, S_tkg=1,
                        block_len=64, kv_quant=True, fp8_packed=True, rmsnorm_X=False, use_pos_id=True, test_sink=True),
    # QK-swap banding fold-MISALIGNMENT fallback (b8, S_ctx=10240 -> block_len resizes to 8 -> 5 folds per
    # band-pair, 5 % band_factor(2) != 0). is_qk_swapped must reject banding here so it runs on the
    # non-swap path; a mid-fold band boundary would otherwise break the constant per-band token shift.
    AttnBlkTestConfig(batch=8, q_heads=8, d_head=64, H=3072, H_actual=2880, S_ctx=10240, S_max_ctx=10240, S_tkg=1,
                        block_len=64, kv_quant=True, fp8_packed=True, rmsnorm_X=False, use_pos_id=True),
    # Block KV, basic causal mask
    AttnBlkTestConfig(batch=4, q_heads=1, d_head=128, H=5376, H_actual=None, S_ctx=1024, S_max_ctx=1024, S_tkg=1,
                        block_len=32, qk_norm_pre_rope=True, qk_norm_pre_rope_gamma=True, use_pos_id=True),
    # Flat KV, basic causal mask
    AttnBlkTestConfig(batch=4, q_heads=1, d_head=128, H=5376, H_actual=None, S_ctx=1024, S_max_ctx=1024, S_tkg=1,
                        block_len=0, use_pos_id=True),
    # Block KV, SWA mask
    AttnBlkTestConfig(batch=4, q_heads=1, d_head=128, H=5376, H_actual=None, S_ctx=1024, S_max_ctx=1024, S_tkg=1,
                        block_len=32, qk_norm_pre_rope=True, qk_norm_pre_rope_gamma=True, use_pos_id=True, sliding_window=256),
    # KVDP
    AttnBlkTestConfig(batch=4, q_heads=1, d_head=128, H=5376, H_actual=None, S_ctx=16384, S_max_ctx=16384, S_tkg=1,
                        block_len=32, KVDP=4, use_pos_id=True),
    AttnBlkTestConfig(batch=4, q_heads=1, d_head=128, H=5376, H_actual=None, S_ctx=16384, S_max_ctx=16384, S_tkg=3,
                        block_len=32, qk_norm_pre_rope=True, qk_norm_pre_rope_gamma=True, KVDP=4, use_pos_id=True, sliding_window=256),

    # ===== Large q_head, S_tkg to test functionality when single batch exceeds batch sharding budget =====
    AttnBlkTestConfig(batch=1, q_heads=64, d_head=128, H=5376, H_actual=None, S_ctx=16384, S_max_ctx=16384, S_tkg=8,
                     block_len=32, qk_norm_pre_rope=True, qk_norm_pre_rope_gamma=True, use_pos_id=True),

    # ===== fp8_packed block KV =====
    AttnBlkTestConfig(batch=8, q_heads=1, d_head=128, H=8192, H_actual=None, S_ctx=10240, S_max_ctx=10240, S_tkg=1,
                        block_len=32, quantization_type=QuantizationType.STATIC, transposed_out=True, kv_quant=True, fp8_packed=True),
    AttnBlkTestConfig(batch=8, q_heads=1, d_head=128, H=8192, H_actual=None, S_ctx=131072, S_max_ctx=131072, S_tkg=5,
                        block_len=32, quantization_type=QuantizationType.STATIC, transposed_out=True, kv_quant=True, fp8_packed=True, KVDP=8),
    AttnBlkTestConfig(batch=2, q_heads=8, d_head=128, H=8192, H_actual=None, S_ctx=8192, S_max_ctx=8192, S_tkg=2,
                        block_len=32, quantization_type=QuantizationType.STATIC, transposed_out=True, kv_quant=True, fp8_packed=True, update_cache=False),
    AttnBlkTestConfig(batch=2, q_heads=8, d_head=128, H=8192, H_actual=None, S_ctx=256, S_max_ctx=256, S_tkg=3,
                        block_len=32, quantization_type=QuantizationType.STATIC, transposed_out=True, kv_quant=True, fp8_packed=True, sliding_window=128,
                        cache_has_kv_head_dim=True),

    # ===== Attention sink (streaming attention sink tokens) =====
    # Flat KV, q_heads=1
    AttnBlkTestConfig(batch=4, q_heads=1, d_head=128, H=8192, H_actual=None, S_ctx=8192, S_max_ctx=8192, S_tkg=1,
                        K_cache_transposed=True, rmsnorm_X=False, test_sink=True),
    # Flat KV, multi q_heads + S_tkg > 1 + bias
    AttnBlkTestConfig(batch=4, q_heads=8, d_head=64, H=3072, H_actual=2880, S_ctx=10240, S_max_ctx=10240, S_tkg=4,
                        K_cache_transposed=True, test_bias=True, test_sink=True),
    # Block KV
    AttnBlkTestConfig(batch=4, q_heads=1, d_head=128, H=8192, H_actual=None, S_ctx=8192, S_max_ctx=8192, S_tkg=5,
                        block_len=16, test_sink=True),
    # Block KV + FP8 KV quant
    AttnBlkTestConfig(batch=16, q_heads=1, d_head=128, H=8192, H_actual=None, S_ctx=2048, S_max_ctx=2048, S_tkg=1,
                        block_len=16, rmsnorm_X=False, kv_quant=True, test_sink=True),
    # In-kernel mask generation + SWA + sink
    AttnBlkTestConfig(batch=4, q_heads=1, d_head=128, H=5376, H_actual=None, S_ctx=1024, S_max_ctx=1024, S_tkg=1,
                        block_len=32, qk_norm_pre_rope=True, qk_norm_pre_rope_gamma=True, use_pos_id=True,
                        sliding_window=256, test_sink=True),
    # KVDP + sink (sink is per q_heads_attn = KVDP * q_heads head)
    AttnBlkTestConfig(batch=8, q_heads=1, d_head=64, H=3072, H_actual=2880, S_ctx=1024, S_max_ctx=1024, S_tkg=1,
                        test_bias=True, KVDP=4, kv_quant=True, test_sink=True),

    # Multi-KV-head (kv_heads=2), flat KV, with sink + in-kernel mask
    AttnBlkTestConfig(batch=2, q_heads=8, d_head=64, H=3072, H_actual=None, S_ctx=256, S_max_ctx=256, S_tkg=1,
                        kv_heads=2, test_sink=True, use_pos_id=True),
    # Multi-KV-head + odd batch (sink-phase coverage), flat KV
    AttnBlkTestConfig(batch=1, q_heads=8, d_head=64, H=3072, H_actual=None, S_ctx=256, S_max_ctx=256, S_tkg=1,
                        kv_heads=2, test_sink=True, use_pos_id=True),
    AttnBlkTestConfig(batch=3, q_heads=8, d_head=64, H=3072, H_actual=None, S_ctx=256, S_max_ctx=256, S_tkg=1,
                        kv_heads=2, test_sink=True, use_pos_id=True),
    AttnBlkTestConfig(batch=1, q_heads=8, d_head=64, H=3072, H_actual=None, S_ctx=256, S_max_ctx=256, S_tkg=1,
                        kv_heads=4, test_sink=True, use_pos_id=True),
    # Multi-KV-head + odd batch, block KV
    AttnBlkTestConfig(batch=3, q_heads=8, d_head=64, H=3072, H_actual=None, S_ctx=1024, S_max_ctx=1024, S_tkg=1,
                        kv_heads=2, block_len=32, test_sink=True, use_pos_id=True),
    # Multi-KV-head with q_per_group=2 (q_heads=8, kv_heads=4), block KV
    AttnBlkTestConfig(batch=4, q_heads=8, d_head=64, H=3072, H_actual=None, S_ctx=1024, S_max_ctx=1024, S_tkg=1,
                        kv_heads=4, block_len=32, test_sink=True, use_pos_id=True),
    # Multi-KV-head + FP8 KV cache quantization, flat KV
    AttnBlkTestConfig(batch=2, q_heads=8, d_head=64, H=3072, H_actual=None, S_ctx=256, S_max_ctx=256, S_tkg=1,
                        kv_heads=2, kv_quant=True, test_sink=True, use_pos_id=True),
    # Multi-KV-head + FP8 packed block KV cache
    AttnBlkTestConfig(batch=4, q_heads=8, d_head=64, H=3072, H_actual=None, S_ctx=1024, S_max_ctx=1024, S_tkg=3,
                        kv_heads=2, block_len=32, kv_quant=True, fp8_packed=True, test_sink=True, use_pos_id=True),
    AttnBlkTestConfig(batch=4, q_heads=8, d_head=64, H=3072, H_actual=None, S_ctx=1024, S_max_ctx=1024, S_tkg=3,
                        kv_heads=4, block_len=32, kv_quant=True, fp8_packed=True, test_sink=True, use_pos_id=True),
    # Multi-KV-head + KVDP=4, block KV
    AttnBlkTestConfig(batch=8, q_heads=4, d_head=64, H=3072, H_actual=None, S_ctx=1024, S_max_ctx=1024, S_tkg=1,
                        kv_heads=2, block_len=32, KVDP=4,
                        KVDP_collective_mode=KVDPCollectiveMode.ALL_GATHER_SLICE,
                        test_sink=True, use_pos_id=True),
    # 4D block KV cache [num_blocks, kv_heads, block_len, d_head].
    AttnBlkTestConfig(batch=4, q_heads=8, d_head=64, H=3072, H_actual=None, S_ctx=1024, S_max_ctx=1024, S_tkg=3,
                        kv_heads=4, block_len=32, cache_has_kv_head_dim=True),
    # kv_heads=2, bf16, update_cache=False (returns new K/V tokens; no cache split).
    AttnBlkTestConfig(batch=4, q_heads=8, d_head=64, H=3072, H_actual=None, S_ctx=1024, S_max_ctx=1024, S_tkg=1,
                        kv_heads=2, block_len=32, cache_has_kv_head_dim=True, update_cache=False),
    # kv_heads=4, fp8_packed (5D input [num_blocks, kv_heads, block_len//2, d, 2]) + sink + in-kernel mask.
    AttnBlkTestConfig(batch=4, q_heads=8, d_head=64, H=3072, H_actual=None, S_ctx=1024, S_max_ctx=1024, S_tkg=3,
                        kv_heads=4, block_len=32, kv_quant=True, fp8_packed=True,
                        cache_has_kv_head_dim=True, test_sink=True, use_pos_id=True),
    # ===== DMA batching coverage =====
    # fp8_packed d_head=64 fold-batching
    AttnBlkTestConfig(batch=16, q_heads=8, d_head=64, H=3072, H_actual=2880, S_ctx=10240, S_max_ctx=10240, S_tkg=1,
                        block_len=64, kv_quant=True, fp8_packed=True),
    # fp8_packed d_head=64 SWA batch-batching
    AttnBlkTestConfig(batch=16, q_heads=8, d_head=64, H=3072, H_actual=2880, S_ctx=256, S_max_ctx=256, S_tkg=1,
                        block_len=64, kv_quant=True, fp8_packed=True, sliding_window=128),
    # fp8_packed d_head=64 with k_active stitching
    AttnBlkTestConfig(batch=8, q_heads=8, d_head=64, H=3072, H_actual=2880, S_ctx=10240, S_max_ctx=10240, S_tkg=1,
                        block_len=64, kv_quant=True, fp8_packed=True, cache_lens_mean=0.01, cache_lens_stddev=0.0),
    # non-packed fp8 kv bulk-copy stitching with fold-batching
    AttnBlkTestConfig(batch=16, q_heads=8, d_head=64, H=3072, H_actual=2880, S_ctx=10240, S_max_ctx=10240, S_tkg=1,
                        block_len=32, kv_quant=True, cache_lens_mean=0.01, cache_lens_stddev=0.0),
    # bf16 d_head=64 ppf=2 per-token stitching with fold-batching
    AttnBlkTestConfig(batch=16, q_heads=8, d_head=64, H=3072, H_actual=2880, S_ctx=10240, S_max_ctx=10240, S_tkg=1,
                        block_len=64, cache_lens_mean=0.01, cache_lens_stddev=0.0),
    # combined fold+batch batching (k_dma_batch_n_folds=2, k_dma_batch_n_batches=4)
    AttnBlkTestConfig(batch=32, q_heads=8, d_head=64, H=3072, H_actual=2880, S_ctx=2048, S_max_ctx=2048, S_tkg=4,
                        block_len=8, kv_quant=True, fp8_packed=True, cache_lens_mean=0.01, cache_lens_stddev=0.0),
    # bf16 d_head=64 batch-batching with stitching (last tile batch-batches)
    AttnBlkTestConfig(batch=8, q_heads=8, d_head=64, H=3072, H_actual=2880, S_ctx=10240, S_max_ctx=10240, S_tkg=4,
                        block_len=32, cache_lens_mean=0.01, cache_lens_stddev=0.0),
    # d_head=128 (k_row_tile_factor=1) with fold-batching + stitching
    AttnBlkTestConfig(batch=16, q_heads=8, d_head=128, H=3072, H_actual=2880, S_ctx=10240, S_max_ctx=10240, S_tkg=4,
                        block_len=32, cache_lens_mean=0.01, cache_lens_stddev=0.0),

    # CP tests (CP=4, GPT-OSS-like)
    # flat KV S_ctx=1k B=8
    AttnBlkTestConfig(batch=8, q_heads=1, d_head=64, H=3072, H_actual=2880, S_ctx=1024, S_max_ctx=1024, S_tkg=1,
                        rmsnorm_X=False, CP=4, kv_quant=True, kv_scale=KVScaleTest.DEFAULT),
    # flat KV S_tkg=4 B=4: multiple active tokens exercise the per-rank cache padding.
    AttnBlkTestConfig(batch=4, q_heads=1, d_head=64, H=3072, H_actual=2880, S_ctx=1024, S_max_ctx=1024, S_tkg=4,
                        rmsnorm_X=False, CP=4, kv_quant=True, kv_scale=KVScaleTest.DEFAULT),
    # flat KV S_ctx=1k B=8, q_heads=2 (general transpose path)
    AttnBlkTestConfig(batch=8, q_heads=2, d_head=64, H=3072, H_actual=2880, S_ctx=1024, S_max_ctx=1024, S_tkg=1,
                        rmsnorm_X=False, CP=4, kv_quant=True, kv_scale=KVScaleTest.DEFAULT),
    # block KV S_ctx=1k B=8, block_len=32
    AttnBlkTestConfig(batch=8, q_heads=1, d_head=64, H=3072, H_actual=2880, S_ctx=1024, S_max_ctx=1024, S_tkg=1,
                        block_len=32, rmsnorm_X=False, CP=4, kv_quant=True, kv_scale=KVScaleTest.DEFAULT),
    # CP=4 with kv_heads=2: kv_heads=2 folded into batch, then CP combine over the fold
    # (q_heads_attn=q_heads*CP=8, q_per_group=8//kv_heads=4, per-CP-rank folded heads=4//CP=1).
    # Both CP output collective modes are covered: ALL_TO_ALL (default) and REDUCE_SCATTER.
    AttnBlkTestConfig(batch=8, q_heads=2, d_head=64, H=3072, H_actual=2880, S_ctx=1024, S_max_ctx=1024, S_tkg=1,
                        block_len=32, kv_heads=2, cache_has_kv_head_dim=True, rmsnorm_X=False, CP=4,
                        CP_collective_mode=CPCollectiveMode.ALL_TO_ALL,
                        kv_quant=True, kv_scale=KVScaleTest.DEFAULT),
    AttnBlkTestConfig(batch=8, q_heads=2, d_head=64, H=3072, H_actual=2880, S_ctx=1024, S_max_ctx=1024, S_tkg=1,
                        block_len=32, kv_heads=2, cache_has_kv_head_dim=True, rmsnorm_X=False, CP=4,
                        CP_collective_mode=CPCollectiveMode.REDUCE_SCATTER,
                        kv_quant=True, kv_scale=KVScaleTest.DEFAULT),
    # KVDP=2 CP=4 (8 ranks) with kv_heads>1: strided KVDP groups, consecutive CP groups.
    # KVDP uses ALL_GATHER_SLICE; CP uses default ALL_TO_ALL.
    AttnBlkTestConfig(batch=8, q_heads=2, d_head=64, H=3072, H_actual=2880, S_ctx=1024, S_max_ctx=1024, S_tkg=1,
                        block_len=32, kv_heads=2, cache_has_kv_head_dim=True, rmsnorm_X=False, KVDP=2, CP=4,
                        KVDP_collective_mode=KVDPCollectiveMode.ALL_GATHER_SLICE,
                        kvdp_replica_group=[[0, 4], [1, 5], [2, 6], [3, 7]],
                        cp_replica_group=[[0, 1, 2, 3], [4, 5, 6, 7]],
                        kv_quant=True, kv_scale=KVScaleTest.DEFAULT),
    # flat KV S_ctx=2k B=8
    AttnBlkTestConfig(batch=8, q_heads=1, d_head=64, H=3072, H_actual=2880, S_ctx=2048, S_max_ctx=2048, S_tkg=1,
                        rmsnorm_X=False, CP=4, kv_quant=True, kv_scale=KVScaleTest.DEFAULT),
    # block KV S_ctx=2k B=8, block_len=128
    AttnBlkTestConfig(batch=8, q_heads=1, d_head=64, H=3072, H_actual=2880, S_ctx=2048, S_max_ctx=2048, S_tkg=1,
                        block_len=128, rmsnorm_X=False, CP=4, kv_quant=True, kv_scale=KVScaleTest.DEFAULT),
    # block KV S_ctx=2k B=8, block_len=32
    AttnBlkTestConfig(batch=8, q_heads=1, d_head=64, H=3072, H_actual=2880, S_ctx=2048, S_max_ctx=2048, S_tkg=1,
                        block_len=32, rmsnorm_X=False, CP=4, kv_quant=True, kv_scale=KVScaleTest.DEFAULT),
    # block KV S_ctx=1k B=32, block_len=32
    AttnBlkTestConfig(batch=32, q_heads=1, d_head=64, H=3072, H_actual=2880, S_ctx=1024, S_max_ctx=1024, S_tkg=1,
                        block_len=32, rmsnorm_X=False, CP=4, kv_quant=True, kv_scale=KVScaleTest.DEFAULT),
    # block KV S_ctx=1k B=64, block_len=32
    AttnBlkTestConfig(batch=64, q_heads=1, d_head=64, H=3072, H_actual=2880, S_ctx=1024, S_max_ctx=1024, S_tkg=1,
                        block_len=32, rmsnorm_X=False, CP=4,
                        kv_quant=True, kv_scale=KVScaleTest.DEFAULT),
    # CP long context tests (S_ctx >= 64k, slower)
    # flat KV S_ctx=73k B=8
    AttnBlkTestConfig(batch=8, q_heads=1, d_head=64, H=3072, H_actual=2880, S_ctx=73728, S_max_ctx=73728, S_tkg=1,
                        rmsnorm_X=False, CP=4, kv_quant=True, kv_scale=KVScaleTest.DEFAULT),
    # block KV S_ctx=128k B=8, block_len=32
    AttnBlkTestConfig(batch=8, q_heads=1, d_head=64, H=3072, H_actual=2880, S_ctx=131072, S_max_ctx=131072, S_tkg=1,
                        block_len=32, rmsnorm_X=False, CP=4, kv_quant=True, kv_scale=KVScaleTest.DEFAULT),
    # flat KV S_ctx=512k B=8
    AttnBlkTestConfig(batch=8, q_heads=1, d_head=64, H=3072, H_actual=2880, S_ctx=524288, S_max_ctx=524288, S_tkg=1,
                        rmsnorm_X=False, CP=4, kv_quant=True, kv_scale=KVScaleTest.DEFAULT),
    # block KV S_ctx=512k B=8, block_len=32
    AttnBlkTestConfig(batch=8, q_heads=1, d_head=64, H=3072, H_actual=2880, S_ctx=524288, S_max_ctx=524288, S_tkg=1,
                        block_len=32, rmsnorm_X=False, CP=4, kv_quant=True, kv_scale=KVScaleTest.DEFAULT),
    # flat KV S_ctx=1M B=8
    AttnBlkTestConfig(batch=8, q_heads=1, d_head=64, H=3072, H_actual=2880, S_ctx=1048576, S_max_ctx=1048576, S_tkg=1,
                        rmsnorm_X=False, CP=4, kv_quant=True, kv_scale=KVScaleTest.DEFAULT),
    # block KV S_ctx=1M B=8, block_len=32
    AttnBlkTestConfig(batch=8, q_heads=1, d_head=64, H=3072, H_actual=2880, S_ctx=1048576, S_max_ctx=1048576, S_tkg=1,
                        block_len=32, rmsnorm_X=False, CP=4, kv_quant=True, kv_scale=KVScaleTest.DEFAULT),
    # CP B=64 tests
    # Llama3 70B CP=8 TP8 B=8 S_ctx=10k (short context, precision baseline)
    AttnBlkTestConfig(batch=8, q_heads=1, d_head=128, H=8192, H_actual=None, S_ctx=10240, S_max_ctx=10240, S_tkg=1,
                        block_len=32, quantization_type=QuantizationType.STATIC, transposed_out=True,
                        kv_quant=True, kv_scale=KVScaleTest.DEFAULT, CP=8,
),
    # Llama3 70B CP=8 TP8 B=8 S_ctx=128k block KV FP8 KV (per-core: q_heads=1 as if TP=64)
    AttnBlkTestConfig(batch=8, q_heads=1, d_head=128, H=8192, H_actual=None, S_ctx=131072, S_max_ctx=131072, S_tkg=1,
                        block_len=32, quantization_type=QuantizationType.STATIC, transposed_out=True,
                        kv_quant=True, kv_scale=KVScaleTest.DEFAULT, CP=8,
),
    # flat KV S_ctx=128k B=64
    AttnBlkTestConfig(batch=64, q_heads=1, d_head=64, H=3072, H_actual=2880, S_ctx=131072, S_max_ctx=131072, S_tkg=1,
                        rmsnorm_X=False, CP=4,
                        kv_quant=True, kv_scale=KVScaleTest.DEFAULT),
    # block KV S_ctx=128k B=64, block_len=32
    AttnBlkTestConfig(batch=64, q_heads=1, d_head=64, H=3072, H_actual=2880, S_ctx=131072, S_max_ctx=131072, S_tkg=1,
                        block_len=32, rmsnorm_X=False, CP=4,
                        kv_quant=True, kv_scale=KVScaleTest.DEFAULT),
    # CP REDUCE_SCATTER tests
    AttnBlkTestConfig(batch=8, q_heads=1, d_head=64, H=3072, H_actual=2880, S_ctx=1024, S_max_ctx=1024, S_tkg=1,
                        block_len=32, rmsnorm_X=False, CP=4, CP_collective_mode=CPCollectiveMode.REDUCE_SCATTER, kv_quant=True),
    # CP ALL_TO_ALL tests
    AttnBlkTestConfig(batch=8, q_heads=1, d_head=64, H=3072, H_actual=2880, S_ctx=1024, S_max_ctx=1024, S_tkg=1,
                        block_len=32, rmsnorm_X=False, CP=4, CP_collective_mode=CPCollectiveMode.ALL_TO_ALL, kv_quant=True),
    # CP ALL_TO_ALL batch sharding: B=64
    AttnBlkTestConfig(batch=64, q_heads=1, d_head=64, H=3072, H_actual=2880, S_ctx=1024, S_max_ctx=1024, S_tkg=1,
                        block_len=32, rmsnorm_X=False, CP=4, CP_collective_mode=CPCollectiveMode.ALL_TO_ALL, kv_quant=True),
    # CP REDUCE_SCATTER batch sharding test: B=64 triggers LNC batch sharding with CP=4
    AttnBlkTestConfig(batch=64, q_heads=1, d_head=64, H=3072, H_actual=2880, S_ctx=1024, S_max_ctx=1024, S_tkg=1,
                        block_len=128, rmsnorm_X=False, CP=4, CP_collective_mode=CPCollectiveMode.REDUCE_SCATTER, kv_quant=True),
    # CP REDUCE_SCATTER batch sharding: B=64 block_len=32
    AttnBlkTestConfig(batch=64, q_heads=1, d_head=64, H=3072, H_actual=2880, S_ctx=1024, S_max_ctx=1024, S_tkg=1,
                        block_len=32, rmsnorm_X=False, CP=4, CP_collective_mode=CPCollectiveMode.REDUCE_SCATTER, kv_quant=True),
    # CP REDUCE_SCATTER batch sharding: B=64 S_ctx=4096
    AttnBlkTestConfig(batch=64, q_heads=1, d_head=64, H=3072, H_actual=2880, S_ctx=4096, S_max_ctx=4096, S_tkg=1,
                        block_len=32, rmsnorm_X=False, CP=4, CP_collective_mode=CPCollectiveMode.REDUCE_SCATTER, kv_quant=True),
    # CP REDUCE_SCATTER batch sharding: B=64 flat KV
    AttnBlkTestConfig(batch=64, q_heads=1, d_head=64, H=3072, H_actual=2880, S_ctx=1024, S_max_ctx=1024, S_tkg=1,
                        block_len=0, rmsnorm_X=False, CP=4, CP_collective_mode=CPCollectiveMode.REDUCE_SCATTER, kv_quant=True),
    # CP interleave_size=1: per-token interleaving (kernel-transparent, only affects ownership mapping)
    AttnBlkTestConfig(batch=8, q_heads=1, d_head=64, H=3072, H_actual=2880, S_ctx=1024, S_max_ctx=1024, S_tkg=1,
                        block_len=32, rmsnorm_X=False, CP=4, cp_interleave_size=1, kv_quant=True, kv_scale=KVScaleTest.DEFAULT),
    # ===== CP + in-kernel mask generation (use_pos_id=True) =====
    # The fused mask path generates the per-rank mask on-chip from pos_ids (local filled slots).
    # TODO: enable once the per-rank simulation harness supports pos_ids under CP.
    # AttnBlkTestConfig(batch=8, q_heads=1, d_head=64, H=3072, H_actual=2880, S_ctx=1024, S_max_ctx=1024, S_tkg=1,
    #                     block_len=32, rmsnorm_X=False, CP=4, use_pos_id=True, kv_quant=True, kv_scale=KVScaleTest.DEFAULT),
    # CP B=1: low-batch CP (batch < CP, only 1 rank has an active token)
    AttnBlkTestConfig(batch=1, q_heads=1, d_head=64, H=3072, H_actual=2880, S_ctx=1024, S_max_ctx=1024, S_tkg=1,
                        block_len=32, rmsnorm_X=False, CP=4, kv_quant=True, kv_scale=KVScaleTest.DEFAULT),
    # CP S_tkg=4: speculative decode with CP (multiple active tokens per batch)
    AttnBlkTestConfig(batch=4, q_heads=1, d_head=64, H=3072, H_actual=2880, S_ctx=1024, S_max_ctx=1024, S_tkg=4,
                        block_len=32, rmsnorm_X=False, CP=4, kv_quant=True, kv_scale=KVScaleTest.DEFAULT),
    # Combined KVDP+CP: KVDP=2 CP=2 (4 ranks total, batch sliced + sequence sliced)
    AttnBlkTestConfig(batch=8, q_heads=1, d_head=64, H=3072, H_actual=2880, S_ctx=1024, S_max_ctx=1024, S_tkg=1,
                        rmsnorm_X=False, KVDP=2, CP=2,
                        KVDP_collective_mode=KVDPCollectiveMode.ALL_GATHER_SLICE,
                        kv_quant=True, kv_scale=KVScaleTest.DEFAULT),
    # Combined KVDP=4 CP=2 (8 ranks) with KVDP ALL_TO_ALL + CP REDUCE_SCATTER.
    AttnBlkTestConfig(batch=8, q_heads=1, d_head=64, H=3072, H_actual=2880, S_ctx=1024, S_max_ctx=1024, S_tkg=1,
                        rmsnorm_X=False, KVDP=4, CP=2,
                        KVDP_collective_mode=KVDPCollectiveMode.ALL_TO_ALL,
                        CP_collective_mode=CPCollectiveMode.REDUCE_SCATTER,
                        kvdp_replica_group=[[0, 1, 2, 3], [4, 5, 6, 7]],
                        cp_replica_group=[[0, 4], [1, 5], [2, 6], [3, 7]],
                        kv_quant=True, kv_scale=KVScaleTest.DEFAULT),
    # Combined KVDP=4 CP=2 (8 ranks): explicit replica groups
    # KVDP groups (consecutive): [(0,1,2,3), (4,5,6,7)]
    # CP groups (strided within KVDP): [(0,4), (1,5), (2,6), (3,7)]
    AttnBlkTestConfig(batch=8, q_heads=1, d_head=64, H=3072, H_actual=2880, S_ctx=1024, S_max_ctx=1024, S_tkg=1,
                        rmsnorm_X=False, KVDP=4, CP=2,
                        KVDP_collective_mode=KVDPCollectiveMode.ALL_GATHER_SLICE,
                        CP_collective_mode=CPCollectiveMode.REDUCE_SCATTER,
                        kvdp_replica_group=[[0, 1, 2, 3], [4, 5, 6, 7]],
                        cp_replica_group=[[0, 4], [1, 5], [2, 6], [3, 7]],
                        kv_quant=True, kv_scale=KVScaleTest.DEFAULT),
    # KVDP=2 CP=4 (8 ranks): strided KVDP groups, consecutive CP groups — the KVDP=4 CP=2
    # arrangement above with the degrees swapped.
    AttnBlkTestConfig(batch=8, q_heads=1, d_head=64, H=3072, H_actual=2880, S_ctx=1024, S_max_ctx=1024, S_tkg=1,
                        rmsnorm_X=False, KVDP=2, CP=4,
                        KVDP_collective_mode=KVDPCollectiveMode.ALL_GATHER_SLICE,
                        CP_collective_mode=CPCollectiveMode.REDUCE_SCATTER,
                        kvdp_replica_group=[[0, 4], [1, 5], [2, 6], [3, 7]],
                        cp_replica_group=[[0, 1, 2, 3], [4, 5, 6, 7]],
                        kv_quant=True, kv_scale=KVScaleTest.DEFAULT),
]
# fmt: on


def _low_rank_cfgs(cfgs: list[AttnBlkTestConfig]) -> list[AttnBlkTestConfig]:
    """Filter configs that need at most 4 NeuronCores (non-high-rank)."""
    return [c for c in cfgs if not c.is_high_rank()]


def _attn_blk_fast_marks(cfg: AttnBlkTestConfig) -> list:
    """Per-config marks for the fast attn-block parametrization.

    - Collective (KVDP>1 / CP>1) configs use multi-rank inputs / collective ops the
      simulator backend does not support, so they skip the simulation lane.
    - The q_heads=8, S_ctx=1024 block-KV configs trigger the token-gen block-length
      reduction (block_len 32 -> 4, 256 blocks/batch). The torch golden that
      _infer_output_shapes_and_dtypes runs for shape inference then costs >240s CPU,
      exceeding the compile-only fast lane's --cpu-timeout. Mark them `slow` (not
      `fast`) so they are excluded from the -m fast lane but still run in the full
      suite.
    """
    marks = []
    if cfg.KVDP > 1 or cfg.CP > 1:
        marks.append(pytest.mark.skip_simulation)
    if cfg.q_heads == 8 and cfg.S_ctx == 1024 and cfg.block_len > 0:
        marks.append(pytest.mark.slow)
    else:
        marks.append(pytest.mark.fast)
    return marks


@pytest_test_metadata(name="Attention Block TKG", tags=["model"])
@pytest_marks(["attention", "tkg", "experimental", "mx"])
@final
class TestRangeAttnBlk:
    # fmt: off
    FAST_ATTN_BLK_CFGS = [
        # Basic flat KV (core path: RMSNorm + QKV + RoPE + attention + output projection)
        AttnBlkTestConfig(batch=1, q_heads=1, d_head=128, H=8192, H_actual=None, S_ctx=1024, S_max_ctx=1024, S_tkg=1,
                            K_cache_transposed=True),
        # Block KV cache
        AttnBlkTestConfig(batch=4, q_heads=1, d_head=128, H=8192, H_actual=None, S_ctx=256, S_max_ctx=256, S_tkg=1,
                            block_len=32),
        # Physical cache pool smaller than logically required; repeated iota is sliced to the table width.
        AttnBlkTestConfig(batch=1, q_heads=4, d_head=64, H=3072, H_actual=2880, S_ctx=6144, S_max_ctx=6144, S_tkg=1,
                            block_len=8, rmsnorm_X=False, test_bias=True, cache_has_kv_head_dim=True,
                            physical_cache_blocks_per_head=256, use_pos_id=True,
                            cache_lens_mean=0.16, cache_lens_stddev=0.0),
        # Physical cache pool smaller than logically required for full context length (with block resize).
        AttnBlkTestConfig(batch=1, q_heads=4, d_head=64, H=3072, H_actual=2880, S_ctx=6144, S_max_ctx=6144, S_tkg=1,
                            block_len=32, rmsnorm_X=False, test_bias=True, cache_has_kv_head_dim=True,
                            physical_cache_blocks_per_head=32, use_pos_id=True,
                            cache_lens_mean=0.16, cache_lens_stddev=0.0),
        # FP8 static weight quantization
        AttnBlkTestConfig(batch=1, q_heads=1, d_head=128, H=8192, H_actual=None, S_ctx=1024, S_max_ctx=1024, S_tkg=1,
                            block_len=32, quantization_type=QuantizationType.STATIC),
        # FP8 KV cache quantization
        AttnBlkTestConfig(batch=1, q_heads=1, d_head=128, H=8192, H_actual=None, S_ctx=1024, S_max_ctx=1024, S_tkg=1,
                            block_len=32, kv_quant=True),
        # QK norm pre-rope with gamma (Qwen3/Gemma3 path)
        AttnBlkTestConfig(batch=1, q_heads=1, d_head=128, H=5120, H_actual=None, S_ctx=1024, S_max_ctx=1024, S_tkg=1,
                            block_len=32, qk_norm_pre_rope=True, qk_norm_pre_rope_gamma=True),
        # d_head=256 partial rotary (rotary_dim=64), n_d_tiles=2. For d>128 the kernel uses the
        # external-projection / external-KV-update path: skip_output_projection=True and
        # update_cache=False (in-kernel out-proj and block-KV update both need D<=128).
        AttnBlkTestConfig(batch=1, q_heads=2, d_head=256, H=2048, H_actual=None, S_ctx=512, S_max_ctx=512, S_tkg=1,
                            block_len=32, rotary_dim=64, skip_output_projection=True, update_cache=False),
        AttnBlkTestConfig(batch=1, q_heads=1, d_head=256, H=2048, H_actual=None, S_ctx=512, S_max_ctx=512, S_tkg=1,
                            block_len=32, rotary_dim=64, skip_output_projection=True, update_cache=False),
        # Same d=256 partial-rope configs WITH qk-layernorm pre-rope (hardware/compile lanes only;
        # the RMSNorm-TKG subkernel is not CPU-simulable).
        AttnBlkTestConfig(batch=1, q_heads=2, d_head=256, H=2048, H_actual=None, S_ctx=512, S_max_ctx=512, S_tkg=1,
                            block_len=32, rotary_dim=64, skip_output_projection=True, update_cache=False,
                            qk_norm_pre_rope=True, qk_norm_pre_rope_gamma=True),
        # Gemma-style d_head=256 with FULL rotary (rotary_dim=0 -> 256, n_d_tiles=2). Unlike Qwen3.5
        # (rotary_dim=64, confined to d-tile 0), the rotary channels here SPAN both d-tiles: the even
        # half [0:128] is d-tile 0 and the odd half [128:256] is d-tile 1, exercising the
        # cross-d-tile RoPE path (_rope_d_tiled_spanning). d>128 uses external proj / external KV
        # update: skip_output_projection=True, update_cache=False.
        AttnBlkTestConfig(batch=1, q_heads=2, d_head=256, H=2048, H_actual=None, S_ctx=512, S_max_ctx=512, S_tkg=1,
                            block_len=32, skip_output_projection=True, update_cache=False),
        AttnBlkTestConfig(batch=1, q_heads=1, d_head=256, H=2048, H_actual=None, S_ctx=512, S_max_ctx=512, S_tkg=1,
                            block_len=32, kv_heads=1, skip_output_projection=True, update_cache=False),
        # Gemma d=256 full rotary WITH qk-layernorm pre-rope (hardware/compile lanes only).
        AttnBlkTestConfig(batch=1, q_heads=2, d_head=256, H=2048, H_actual=None, S_ctx=512, S_max_ctx=512, S_tkg=1,
                            block_len=32, skip_output_projection=True, update_cache=False,
                            qk_norm_pre_rope=True, qk_norm_pre_rope_gamma=True),
        # d_head=256 (n_d_tiles=2) with DECODE BATCH > 1. REGRESSION for the d>128 Q/K d-tiled
        # producer head/batch layout bug: _process_head_group wrote the tiled Q/K free dim
        # HEAD-major ([n_d_tiles][n_heads][B][S]) while _compute_qk_matmul reads it BATCH-major
        # ([n_d_tiles][B][q_heads][S]). The two orderings coincide only at batch==1, so every
        # existing d256 config (all batch=1) passed while batch>=2 silently corrupted every
        # batch row's attention output (~8% cos error). These batch=2 configs fail on the
        # unpatched kernel and pass after the fix. (d_head<=128 uses the dst_4d [d,B,n_heads,S]
        # batch-major path and was always correct, so no d128 batch>1 regression config is needed.)
        AttnBlkTestConfig(batch=2, q_heads=2, d_head=256, H=2048, H_actual=None, S_ctx=512, S_max_ctx=512, S_tkg=1,
                            block_len=32, skip_output_projection=True, update_cache=False),
        AttnBlkTestConfig(batch=2, q_heads=8, d_head=256, H=2048, H_actual=None, S_ctx=512, S_max_ctx=512, S_tkg=1,
                            kv_heads=4, block_len=32, skip_output_projection=True, update_cache=False),
        AttnBlkTestConfig(batch=4, q_heads=8, d_head=256, H=2048, H_actual=None, S_ctx=512, S_max_ctx=512, S_tkg=1,
                            kv_heads=4, block_len=32, skip_output_projection=True, update_cache=False),
        # Transposed in+out layout
        AttnBlkTestConfig(batch=1, q_heads=1, d_head=128, H=8192, H_actual=None, S_ctx=1024, S_max_ctx=1024, S_tkg=1,
                            block_len=32, transposed_in=True, transposed_out=True),
        # Multi-token generation (S_tkg > 1)
        AttnBlkTestConfig(batch=4, q_heads=1, d_head=128, H=8192, H_actual=None, S_ctx=256, S_max_ctx=256, S_tkg=5,
                            block_len=32),
        # In-kernel mask generation (use_pos_id=True)
        AttnBlkTestConfig(batch=4, q_heads=1, d_head=128, H=5376, H_actual=None, S_ctx=1024, S_max_ctx=1024, S_tkg=1,
                            block_len=32, qk_norm_pre_rope=True, qk_norm_pre_rope_gamma=True, use_pos_id=True),
        # H_actual padding (GPT-OSS style)
        AttnBlkTestConfig(batch=4, q_heads=8, d_head=64, H=3072, H_actual=2880, S_ctx=1024, S_max_ctx=1024, S_tkg=1,
                            test_bias=True),
        # Dynamic FA early exit (LNC2, validated via attention_tkg + attention_block_tkg)
        AttnBlkTestConfig(batch=1, q_heads=8, d_head=128, H=5376, H_actual=None, S_ctx=131072, S_max_ctx=131072, S_tkg=1,
                            block_len=32, use_pos_id=True, max_context_len=16385),
        AttnBlkTestConfig(batch=1, q_heads=8, d_head=128, H=5376, H_actual=None, S_ctx=131072, S_max_ctx=131072, S_tkg=1,
                            block_len=32, use_pos_id=True, max_context_len=24576),
        AttnBlkTestConfig(batch=1, q_heads=8, d_head=128, H=5376, H_actual=None, S_ctx=131072, S_max_ctx=131072, S_tkg=1,
                            block_len=32, use_pos_id=True, max_context_len=30000),
        AttnBlkTestConfig(batch=1, q_heads=8, d_head=128, H=5376, H_actual=None, S_ctx=131072, S_max_ctx=131072, S_tkg=1,
                            block_len=32, use_pos_id=True, max_context_len=65536),
        AttnBlkTestConfig(batch=1, q_heads=8, d_head=128, H=5376, H_actual=None, S_ctx=131072, S_max_ctx=131072, S_tkg=1,
                            block_len=32, use_pos_id=True, max_context_len=131072),
        # FP8 packed KV cache
        AttnBlkTestConfig(batch=4, q_heads=4, d_head=64, H=8192, H_actual=None, S_ctx=1024, S_max_ctx=1024, S_tkg=3,
                            block_len=32, kv_quant=True, fp8_packed=True),
        # Attention sink (streaming attention sink tokens)
        AttnBlkTestConfig(batch=4, q_heads=8, d_head=64, H=3072, H_actual=2880, S_ctx=1024, S_max_ctx=1024, S_tkg=1,
                            block_len=32, test_bias=True, test_sink=True),
        # Multi-KV-head (kv_heads > 1), flat KV, S_tkg=1
        AttnBlkTestConfig(batch=2, q_heads=8, d_head=64, H=3072, H_actual=None, S_ctx=256, S_max_ctx=256, S_tkg=1,
                            kv_heads=2),
        # Multi-KV-head with q_per_group=2 (q_heads=8, kv_heads=4), block KV
        AttnBlkTestConfig(batch=4, q_heads=8, d_head=64, H=3072, H_actual=None, S_ctx=1024, S_max_ctx=1024, S_tkg=1,
                            kv_heads=4, block_len=32),
        AttnBlkTestConfig(batch=4, q_heads=8, d_head=64, H=3072, H_actual=None, S_ctx=1024, S_max_ctx=1024, S_tkg=1,
                            kv_heads=4, block_len=32, kv_quant=True),
        AttnBlkTestConfig(batch=4, q_heads=8, d_head=64, H=3072, H_actual=None, S_ctx=1024, S_max_ctx=1024, S_tkg=3,
                            kv_heads=4, block_len=32, kv_quant=True, fp8_packed=True),
        # Multi-KV-head + per-Q-head attention sink (compact [q_heads_attn,1] sink, broadcast in _prep_sink)
        AttnBlkTestConfig(batch=2, q_heads=8, d_head=64, H=3072, H_actual=None, S_ctx=256, S_max_ctx=256, S_tkg=1,
                            kv_heads=2, test_sink=True),
        # Sink + odd real batch -> batch-sharded folded batch cuts mid kv-group (kv-head phase != 0 on NC1).
        # batch=1,kv=2: B_folded=2, phase=1 on NC1. batch=3,kv=2: B_folded=6, phase + partial tail. batch=1,kv=4: phase=2.
        AttnBlkTestConfig(batch=1, q_heads=8, d_head=64, H=3072, H_actual=None, S_ctx=256, S_max_ctx=256, S_tkg=1,
                            kv_heads=2, test_sink=True),
        AttnBlkTestConfig(batch=3, q_heads=8, d_head=64, H=3072, H_actual=None, S_ctx=256, S_max_ctx=256, S_tkg=1,
                            kv_heads=2, test_sink=True),
        AttnBlkTestConfig(batch=1, q_heads=8, d_head=64, H=3072, H_actual=None, S_ctx=256, S_max_ctx=256, S_tkg=1,
                            kv_heads=4, test_sink=True),
        # Multi-KV-head + multi-token generation (S_tkg > 1)
        AttnBlkTestConfig(batch=2, q_heads=8, d_head=64, H=3072, H_actual=None, S_ctx=256, S_max_ctx=256, S_tkg=4,
                            kv_heads=2),
        # Multi-KV-head + block KV cache
        AttnBlkTestConfig(batch=4, q_heads=8, d_head=64, H=3072, H_actual=None, S_ctx=1024, S_max_ctx=1024, S_tkg=1,
                            kv_heads=2, block_len=32),
        # Multi-KV-head + FP8 KV cache quantization, flat KV
        AttnBlkTestConfig(batch=4, q_heads=8, d_head=64, H=3072, H_actual=None, S_ctx=1024, S_max_ctx=1024, S_tkg=1,
                            kv_heads=2, kv_quant=True, test_bias=True),
        # Multi-KV-head + FP8 KV cache quantization, block KV
        AttnBlkTestConfig(batch=4, q_heads=8, d_head=64, H=3072, H_actual=None, S_ctx=1024, S_max_ctx=1024, S_tkg=1,
                            kv_heads=2, block_len=32, kv_quant=True),
        # Multi-KV-head + FP8 packed block KV cache
        AttnBlkTestConfig(batch=4, q_heads=8, d_head=64, H=3072, H_actual=None, S_ctx=1024, S_max_ctx=1024, S_tkg=3,
                            kv_heads=2, block_len=32, kv_quant=True, fp8_packed=True),
        # Multi-KV-head + 4D block KV cache [num_blocks, kv_heads, ...], fp8_packed (5D), in-kernel mask.
        AttnBlkTestConfig(batch=4, q_heads=8, d_head=64, H=3072, H_actual=None, S_ctx=1024, S_max_ctx=1024, S_tkg=3,
                            kv_heads=2, block_len=32, kv_quant=True, fp8_packed=True,
                            cache_has_kv_head_dim=True, use_pos_id=True),
        # Multi-KV-head + in-kernel mask generation (pos_ids), block KV
        AttnBlkTestConfig(batch=4, q_heads=8, d_head=64, H=3072, H_actual=None, S_ctx=1024, S_max_ctx=1024, S_tkg=1,
                            kv_heads=2, block_len=32, use_pos_id=True),
        # Multi-KV-head + sliding-window attention (SWA via pos_ids), block KV
        AttnBlkTestConfig(batch=4, q_heads=8, d_head=64, H=3072, H_actual=None, S_ctx=1024, S_max_ctx=1024, S_tkg=1,
                            kv_heads=2, block_len=32, use_pos_id=True, sliding_window=256),
        # Multi-KV-head + block KV cache + replicated KVDP
        AttnBlkTestConfig(batch=8, q_heads=4, d_head=64, H=3072, H_actual=None, S_ctx=1024, S_max_ctx=1024, S_tkg=1,
                            kv_heads=2, block_len=32, KVDP=4, KVDP_collective_mode=KVDPCollectiveMode.ALL_GATHER_SLICE),
        # Multi-KV-head + replicated KVDP (KV heads replicated across ranks, batch sliced)
        AttnBlkTestConfig(batch=8, q_heads=4, d_head=64, H=3072, H_actual=None, S_ctx=1024, S_max_ctx=1024, S_tkg=1,
                            kv_heads=2, KVDP=4, KVDP_collective_mode=KVDPCollectiveMode.ALL_GATHER_SLICE),
        AttnBlkTestConfig(batch=8, q_heads=4, d_head=64, H=3072, H_actual=None, S_ctx=1024, S_max_ctx=1024, S_tkg=1,
                            kv_heads=2, KVDP=4, KVDP_collective_mode=KVDPCollectiveMode.ALL_TO_ALL),
        # Multi-KV-head + FP8 KV cache quantization (kv_heads=4), block KV, with sink + in-kernel mask
        AttnBlkTestConfig(batch=4, q_heads=8, d_head=64, H=3072, H_actual=None, S_ctx=1024, S_max_ctx=1024, S_tkg=1,
                            kv_heads=4, block_len=32, kv_quant=True, test_sink=True, use_pos_id=True),
        # Multi-KV-head + KVDP=4 + FP8 packed block KV cache, with sink + in-kernel mask
        AttnBlkTestConfig(batch=8, q_heads=4, d_head=64, H=3072, H_actual=None, S_ctx=1024, S_max_ctx=1024, S_tkg=3,
                            kv_heads=2, block_len=32, kv_quant=True, fp8_packed=True, KVDP=4,
                            KVDP_collective_mode=KVDPCollectiveMode.ALL_GATHER_SLICE,
                            test_sink=True, use_pos_id=True),
        # Multi-KV-head + sliding-window attention, flat KV (sink-stress: short context)
        AttnBlkTestConfig(batch=3, q_heads=8, d_head=64, H=3072, H_actual=None, S_ctx=128, S_max_ctx=128, S_tkg=1,
                            kv_heads=2, test_sink=True, use_pos_id=True, sliding_window=16),
        # Multi-KV-head + sliding-window attention, block KV (sink-stress: kv_heads=4)
        AttnBlkTestConfig(batch=3, q_heads=8, d_head=64, H=3072, H_actual=None, S_ctx=256, S_max_ctx=256, S_tkg=1,
                            kv_heads=4, block_len=32, test_sink=True, use_pos_id=True, sliding_window=16),
    ]
    # fmt: on

    @pytest.mark.parametrize(
        "attn_blk_cfg",
        [pytest.param(cfg, marks=_attn_blk_fast_marks(cfg)) for cfg in FAST_ATTN_BLK_CFGS],
        ids=lambda p: p.test_id(),
    )
    def test_attn_blk_fast(
        self,
        test_manager: Orchestrator,
        platform_target: Platforms,
        attn_blk_cfg: AttnBlkTestConfig,
    ):
        _run_attention_block_test(
            test_manager=test_manager,
            platform_target=platform_target,
            cfg=attn_blk_cfg,
        )

    # fmt: off
    @pytest.mark.parametrize("attn_blk_cfg",
        _low_rank_cfgs(RANGE_ATTN_BLK_CFGS),
        ids=lambda p: p.test_id(),
    # fmt: on
    )
    def test_attn_blk_megakernel(
        self,
        test_manager: Orchestrator,
        platform_target: Platforms,
        attn_blk_cfg: AttnBlkTestConfig
    ):
        assert not attn_blk_cfg.is_high_rank(), \
            f"High-rank config (KVDP={attn_blk_cfg.KVDP}, CP={attn_blk_cfg.CP}) belongs in test_attention_block_tkg_high_rank.py"
        if attn_blk_cfg.xfail_reason:
            pytest.xfail(attn_blk_cfg.xfail_reason)
        _run_attention_block_test(
            test_manager=test_manager,
            platform_target=platform_target,
            cfg=attn_blk_cfg,
        )


@pytest_marks(["attention", "tkg", "experimental", "mx", "model"])
@final
class TestAttnBlkModel:
    """Model regression tests for Attention Block TKG kernel.

    Separate test methods per tier for cleaner pytest discovery:
    - test_tier0: Critical model configs (high priority)
    - test_optimal: Optimal performance configs
    - test_generality: Generality/coverage configs

    High-rank model configs (KVDP/CP > 4) are in test_attention_block_tkg_high_rank.py.
    """

    def _run_model_test(
        self,
        test_manager: Orchestrator,
        collector: IMetricsCollector,
        platform_target: Platforms,
        cfg: AttnBlkTestConfig,
    ):
        """Common test logic for all model tiers."""
        assert not cfg.is_high_rank(), (
            f"High-rank config (KVDP={cfg.KVDP}, CP={cfg.CP}) belongs in test_attention_block_tkg_high_rank.py"
        )
        attn_blk_metadata_list = _get_attention_block_metadata()
        test_metadata_key = {
            "batch": cfg.batch,
            "q_heads": cfg.q_heads,
            "d_head": cfg.d_head,
            "H": cfg.H,
            "S_ctx": cfg.S_ctx,
            "S_tkg": cfg.S_tkg,
            "kv_quant": cfg.kv_quant,
            "KVDP": cfg.KVDP,
            "transposed_in": cfg.transposed_in,
            "DCP": cfg.CP,  # JSON metadata uses "DCP" key
        }
        collector.match_and_add_metadata_dimensions(test_metadata_key, attn_blk_metadata_list)
        _run_attention_block_test(
            test_manager=test_manager,
            platform_target=platform_target,
            cfg=cfg,
        )

    @pytest.mark.tier0
    @pytest.mark.parametrize(
        "cfg",
        _low_rank_cfgs(attention_block_tkg_model_configs.get(ModelTestType.TIER0, [])),
        ids=[cfg.test_id() for cfg in _low_rank_cfgs(attention_block_tkg_model_configs.get(ModelTestType.TIER0, []))],
    )
    def test_tier0(
        self,
        test_manager: Orchestrator,
        collector: IMetricsCollector,
        platform_target: Platforms,
        cfg: AttnBlkTestConfig,
    ):
        """TIER0: Critical model configs - highest priority for model validation."""
        self._run_model_test(test_manager, collector, platform_target, cfg)

    @pytest.mark.optimal
    @pytest.mark.platforms(exclude=[Platforms.TRN1, Platforms.TRN3, Platforms.TRN3_A0])
    @pytest.mark.parametrize(
        "cfg",
        _low_rank_cfgs(attention_block_tkg_model_configs.get(ModelTestType.OPTIMAL, [])),
        ids=[cfg.test_id() for cfg in _low_rank_cfgs(attention_block_tkg_model_configs.get(ModelTestType.OPTIMAL, []))],
    )
    def test_optimal(
        self,
        test_manager: Orchestrator,
        collector: IMetricsCollector,
        platform_target: Platforms,
        cfg: AttnBlkTestConfig,
    ):
        """OPTIMAL: Performance-optimized model configs."""
        self._run_model_test(test_manager, collector, platform_target, cfg)

    @pytest.mark.generality
    @pytest.mark.parametrize(
        "cfg",
        _low_rank_cfgs(attention_block_tkg_model_configs.get(ModelTestType.GENERALITY, [])),
        ids=[
            cfg.test_id() for cfg in _low_rank_cfgs(attention_block_tkg_model_configs.get(ModelTestType.GENERALITY, []))
        ],
    )
    def test_generality(
        self,
        test_manager: Orchestrator,
        collector: IMetricsCollector,
        platform_target: Platforms,
        cfg: AttnBlkTestConfig,
    ):
        """GENERALITY: Broad coverage model configs."""
        self._run_model_test(test_manager, collector, platform_target, cfg)
