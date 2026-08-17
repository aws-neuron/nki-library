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

"""Absorbed-latent MLA QKV projection kernel for Context Encoding (CTE).

Adapts :func:`qkv_mla_mx` (v32). Instead of emitting un-absorbed per-head
Q/K/V, this kernel emits the absorbed latent tensors consumed by the
downstream sparse-attention kernel:

  - ``q_lift[B, S, H, L]``  : per-head q_nope @ W_uk  (absorption matmul)
  - ``q_pe  [B, S, H, R]``  : per-head RoPE queries
  - ``c_kv  [B, S, L]``     : shared latent KV (RMSNorm(kv) * gamma)
  - ``k_pe  [B, S, R]``     : shared RoPE key

The Q/KV first projection (stage 1, MX), the Q second projection (stage 2,
MX), the per-head RoPE on q_pe, and the kv-norm / k_pe-rope are kept from
v32. The wkv_b second matmul, the per-head Q/K/V assembly and the KV-cache
writes are dropped; in their place the K_b half of kv_b_proj is supplied as
a bf16 weight ``W_uk`` and absorbed into q_nope per head.

The absorption matmul (q_nope @ W_uk) runs in bf16, not MX: its contraction
is only nope=128, so MX packing (4 elems/word -> 32 partitions) buys no
throughput while forcing tiny strided transposes and a slabbed W_uk. In bf16
the contraction uses all 128 partitions, q_nope needs a single full
transpose per head, and W_uk fits SBUF fully resident (no slabbing).
"""

from typing import Tuple

import nki
import nki.isa as nisa
import nki.language as nl
from nki.isa.constants import dge_mode

from ....core.qkv.qkv_cte import _get_psum_bank_size
from ....core.utils.allocator import SbufManager, get_logger
from ....core.utils.kernel_assert import kernel_assert
from ....core.utils.kernel_helpers import div_ceil, get_program_sharding_info
from .mla_common_cte import (
    _H_PACK,
    _NUM_HW_PSUM_BANKS,
    _apply_rms_norm_inplace,
    _apply_rope_inplace_interleaved_grouped,
    _apply_rope_to_tensor_interleaved,
    _load_mx_weights,
    _load_mx_weights_k_slab,
    _load_norm_weights_for_mx,
    _mx_matmul,
    _mx_matmul_split,
    _mx_matmul_split_k_range,
    _quantize_mx,
    _transpose_preswizzled_for_mx_fused,
    _unswizzle_lora_cols,
    _v32_compute_k_slab_size_512_tiles,
)
from .mla_validate_params import _validate_mla_qkv_inputs

# Views used to fp8-transpose the pre-quantized packed input (nc_transpose can't take
# fp8x4/uint8 directly): the fp8 hidden rides through an fp32 view (4 consecutive fp8 per
# word) and the uint8 scales through an fp8_e5m2 view. fp8 transpose requires the PSUM
# output element-step to be 2 (HW constraint).
_FP8X4_TP_VIEW_DTYPE = nl.float32
_UINT8_TP_VIEW_DTYPE = nl.float8_e5m2
_FP8_PER_FP32 = 4
_FP8_TP_OUT_STEP = 2


@nki.jit
def mla_qkv_cte_kernel(
    # Pre-quantized PACKED MX input (rmsnorm_mx_prefill pack_scales=True); see Args docstring.
    x_hbm_mx: nl.NkiTensor,
    wqkv_a_hbm: nl.NkiTensor,
    wqkv_a_scale_hbm: nl.NkiTensor,
    wq_b_hbm: nl.NkiTensor,
    wq_b_scale_hbm: nl.NkiTensor,
    q_norm_gamma_hbm: nl.NkiTensor,
    # KV path weights
    kv_norm_gamma_hbm: nl.NkiTensor,
    # Absorption weight (K_b half of kv_b_proj), bf16
    wuk_hbm: nl.NkiTensor,
    # RoPE caches
    cos_cache_hbm: nl.NkiTensor,
    sin_cache_hbm: nl.NkiTensor,
    # Dimension parameters
    n_heads: int,
    qk_nope_head_dim: int,
    qk_rope_head_dim: int,
    kv_lora_rank: int,
    qk_lora_rank: int,
    norm_eps: float = 1e-6,
    compact_scales: bool = True,
) -> Tuple[nl.NkiTensor, nl.NkiTensor, nl.NkiTensor, nl.NkiTensor, nl.NkiTensor, nl.NkiTensor]:
    """DeepSeek MLA QKV projection (MX), emitting absorbed latents.

    Consumes the packed MX activation from rmsnorm_mx_prefill (pack_scales=True)
    directly — no in-kernel quantize. Intended for Context Encoding (prefill) with
    DeepSeek-V3.2 dimensions: absorption requires qk_nope_head_dim == 128, and the
    kernel is tuned for n_heads up to 128 and hidden dim H up to 7168 with LNC
    sharding over the sequence dimension. Best used when the packed MX activation is
    produced upstream so no re-quantization is needed.

    Dimensions:
        B: Batch size
        S: Sequence length (tokens)
        H: Hidden dimension size
        n_heads: Number of attention heads
        qk_nope_head_dim: Non-RoPE portion of the per-head Q/K dimension (must be 128)
        qk_rope_head_dim: RoPE portion of the per-head Q/K dimension (R)
        kv_lora_rank: Latent KV LoRA rank (L)
        qk_lora_rank: Q LoRA rank

    Args:
        x_hbm_mx (nl.NkiTensor): ``[B, S, H + scale_region]`` fp8 packed MX activation (see the param comment).
        wqkv_a_hbm (nl.NkiTensor): ``[H // 4, qk_lora_rank + kv_lora_rank + qk_rope_head_dim]`` fp8x4.
        wqkv_a_scale_hbm (nl.NkiTensor): compact block-128 scales for ``wqkv_a``.
        wq_b_hbm (nl.NkiTensor): ``[qk_lora_rank // 4, n_heads * (qk_nope_head_dim + qk_rope_head_dim)]`` fp8x4.
        wq_b_scale_hbm (nl.NkiTensor): compact block-128 scales for ``wq_b``.
        q_norm_gamma_hbm (nl.NkiTensor): ``[1, qk_lora_rank]`` bf16 RMSNorm gamma for the Q intermediate.
        kv_norm_gamma_hbm (nl.NkiTensor): ``[1, kv_lora_rank]`` bf16 RMSNorm gamma for the KV latent.
        wuk_hbm (nl.NkiTensor): ``[qk_nope_head_dim, n_heads * kv_lora_rank]`` bf16 absorption weight
            (``W_uk[h] = [nope, kv_lora]``, contraction = nope; head ``h`` owns
            columns ``[h * kv_lora_rank, (h + 1) * kv_lora_rank)``).
        cos_cache_hbm (nl.NkiTensor): ``[B, S, qk_rope_head_dim]`` bf16 RoPE cosine cache.
        sin_cache_hbm (nl.NkiTensor): ``[B, S, qk_rope_head_dim]`` bf16 RoPE sine cache.
        n_heads (int): Number of attention heads.
        qk_nope_head_dim (int): Non-RoPE per-head Q/K dimension (must be 128).
        qk_rope_head_dim (int): RoPE per-head Q/K dimension.
        kv_lora_rank (int): Latent KV LoRA rank.
        qk_lora_rank (int): Q LoRA rank.
        norm_eps (float): RMSNorm epsilon (default 1e-6).

    Returns:
        q_lift (nl.NkiTensor): ``[B, S, n_heads, kv_lora_rank]`` bf16, per-head absorbed Q latent.
        q_pe (nl.NkiTensor): ``[B, S, n_heads, qk_rope_head_dim]`` bf16, per-head RoPE queries.
        c_kv (nl.NkiTensor): ``[B, S, kv_lora_rank]`` bf16, shared latent KV (RMSNorm(kv) * gamma).
        k_pe (nl.NkiTensor): ``[B, S, qk_rope_head_dim]`` bf16, shared RoPE key.
        qr_qtz (nl.NkiTensor): ``[num_s_tiles, P_MAX, qk_lora_rank // 512, P_MAX]`` uint32 (fp8x4),
            the transposed + q-normed + MX-quantized qr latent, for the SAI indexer (so it skips
            its own transpose+norm+quantize). Allocated in-kernel and always emitted.
        qr_scale (nl.NkiTensor): same shape uint8, the block-32 MX scales paired with ``qr_qtz``.

    Notes:
        - Absorption requires ``qk_nope_head_dim == 128`` (full-partition bf16 contraction).
        - The absorption matmul (q_nope @ W_uk) runs in bf16, not MX.

    Pseudocode:
        # Stage 1 (MX): fused Q/KV first projection.
        qr, kv, k_pe = x @ wqkv_a
        # Q path.
        qr = rms_norm(qr, q_norm_gamma)
        q = qr @ wq_b                      # per head-group (MX)
        q_nope, q_pe = split(q)
        q_pe = rope(q_pe, cos, sin)        # interleaved
        for h in range(n_heads):
            # Absorption (bf16): q_nope[h] @ W_uk[h].
            q_lift[..., h, :] = q_nope[..., h, :] @ W_uk[h]
        # KV path.
        c_kv = rms_norm(kv, kv_norm_gamma)
        k_pe = rope(k_pe, cos, sin)        # interleaved
        return q_lift, q_pe, c_kv, k_pe
    """
    _validate_mla_qkv_inputs(
        x_hbm_mx=x_hbm_mx,
        wqkv_a_hbm=wqkv_a_hbm,
        wqkv_a_scale_hbm=wqkv_a_scale_hbm,
        wq_b_hbm=wq_b_hbm,
        wq_b_scale_hbm=wq_b_scale_hbm,
        q_norm_gamma_hbm=q_norm_gamma_hbm,
        kv_norm_gamma_hbm=kv_norm_gamma_hbm,
        wuk_hbm=wuk_hbm,
        cos_cache_hbm=cos_cache_hbm,
        sin_cache_hbm=sin_cache_hbm,
        n_heads=n_heads,
        qk_nope_head_dim=qk_nope_head_dim,
        qk_rope_head_dim=qk_rope_head_dim,
        kv_lora_rank=kv_lora_rank,
        qk_lora_rank=qk_lora_rank,
        compact_scales=compact_scales,
    )

    B, S, _ = x_hbm_mx.shape
    q_lift_hbm = nl.ndarray((B, S, n_heads, kv_lora_rank), dtype=nl.bfloat16, buffer=nl.shared_hbm)
    q_pe_hbm = nl.ndarray((B, S, n_heads, qk_rope_head_dim), dtype=nl.bfloat16, buffer=nl.shared_hbm)
    c_kv_hbm = nl.ndarray((B, S, kv_lora_rank), dtype=nl.bfloat16, buffer=nl.shared_hbm)
    k_pe_hbm = nl.ndarray((B, S, qk_rope_head_dim), dtype=nl.bfloat16, buffer=nl.shared_hbm)

    # qr latent export for the SAI indexer, allocated in-kernel and returned (see _qkv_stage).
    # Layout matches the indexer's q_projection_mx input: one [P_MAX, lora512, P_MAX] block per
    # global s-tile (P_MAX-sized), fp8x4 packed as uint32 with paired uint8 block-32 scales.
    P_MAX = nl.tile_size.pmax
    num_s_tiles = div_ceil(S, P_MAX)
    qr_lora_512_tiles = qk_lora_rank // (P_MAX * _H_PACK)
    qr_qtz_hbm = nl.ndarray((num_s_tiles, P_MAX, qr_lora_512_tiles, P_MAX), dtype=nl.uint32, buffer=nl.shared_hbm)
    qr_scale_hbm = nl.ndarray((num_s_tiles, P_MAX, qr_lora_512_tiles, P_MAX), dtype=nl.uint8, buffer=nl.shared_hbm)

    sbm = SbufManager(
        sb_lower_bound=0,
        sb_upper_bound=nl.tile_size.total_available_sbuf_size,
        use_auto_alloc=False,
        logger=get_logger("mla_absorbed"),
    )
    _qkv_stage(
        x_hbm_mx,
        wqkv_a_hbm,
        wqkv_a_scale_hbm,
        wq_b_hbm,
        wq_b_scale_hbm,
        q_norm_gamma_hbm,
        kv_norm_gamma_hbm,
        wuk_hbm,
        cos_cache_hbm,
        sin_cache_hbm,
        q_lift_hbm,
        q_pe_hbm,
        c_kv_hbm,
        k_pe_hbm,
        n_heads,
        qk_nope_head_dim,
        qk_rope_head_dim,
        kv_lora_rank,
        qk_lora_rank,
        sbm,
        norm_eps,
        qr_qtz_hbm=qr_qtz_hbm,
        qr_scale_hbm=qr_scale_hbm,
        compact_scales=compact_scales,
    )
    return q_lift_hbm, q_pe_hbm, c_kv_hbm, k_pe_hbm, qr_qtz_hbm, qr_scale_hbm


def _qkv_stage(
    # Pre-quantized PACKED MX input (rmsnorm_mx_prefill pack_scales=True); see the entry docstring.
    x_hbm_mx: nl.NkiTensor,
    wqkv_a_hbm: nl.NkiTensor,
    wqkv_a_scale_hbm: nl.NkiTensor,
    wq_b_hbm: nl.NkiTensor,
    wq_b_scale_hbm: nl.NkiTensor,
    q_norm_gamma_hbm: nl.NkiTensor,
    kv_norm_gamma_hbm: nl.NkiTensor,
    wuk_hbm: nl.NkiTensor,
    cos_cache_hbm: nl.NkiTensor,
    sin_cache_hbm: nl.NkiTensor,
    # Outputs (shared_hbm, allocated by the caller)
    q_lift_hbm: nl.NkiTensor,
    q_pe_hbm: nl.NkiTensor,
    c_kv_hbm: nl.NkiTensor,
    k_pe_hbm: nl.NkiTensor,
    # Dimension parameters
    n_heads: int,
    qk_nope_head_dim: int,
    qk_rope_head_dim: int,
    kv_lora_rank: int,
    qk_lora_rank: int,
    sbm: SbufManager,
    norm_eps: float = 1e-6,
    # qr latent export for the SAI indexer fast path (caller-allocated shared_hbm; see Notes).
    qr_qtz_hbm: nl.NkiTensor = None,
    qr_scale_hbm: nl.NkiTensor = None,
    compact_scales: bool = True,
) -> None:
    """Absorbed-latent MLA QKV stage (packed-MX input). Writes the four latent outputs to
    the caller-provided shared_hbm tensors using the caller's ``sbm``.
    See :func:`mla_qkv_cte_kernel` for arg semantics.
    """
    P_MAX = nl.tile_size.pmax

    kernel_assert(
        qk_nope_head_dim == 128,
        f"[QKV MLA absorbed] absorption contraction requires qk_nope_head_dim == 128, got {qk_nope_head_dim}",
    )
    kernel_assert(
        wuk_hbm.shape == (qk_nope_head_dim, n_heads * kv_lora_rank),
        f"[QKV MLA absorbed] wuk_hbm shape must be ({qk_nope_head_dim}, {n_heads * kv_lora_rank}), got {wuk_hbm.shape}",
    )

    # H is the contraction of wqkv_a (rows are the 4-packed hidden, fp8x4). B, S from the input.
    H = wqkv_a_hbm.shape[0] * _H_PACK
    B, S, _row = x_hbm_mx.shape
    qk_head_dim = qk_nope_head_dim + qk_rope_head_dim

    # ==================== LNC Sharding Setup ====================
    _, num_shards, shard_id = get_program_sharding_info()

    S_shard_base = S // num_shards
    S_shard = S_shard_base
    if S % num_shards != 0 and shard_id == num_shards - 1:
        S_shard = S // num_shards + (S % num_shards)

    S_shard_offset = shard_id * S_shard_base

    # The qr export writes one whole 128-query global tile per core (indexed by
    # s_tile_global_offset // P_MAX, at within-tile column s_tile_global_offset % P_MAX). That is
    # correct only when a shard's tiles never straddle a 128 boundary: either the whole sequence
    # fits in one tile (S <= P_MAX, cores share tile 0 at disjoint columns) or every core owns
    # whole 128-tiles (S_shard_base % P_MAX == 0). Also require an even split so s_tile_sz (the
    # nl.ds write width) is static. Production T/core is always a large multiple of P_MAX.
    kernel_assert(
        S % num_shards == 0 and (S <= P_MAX or S_shard_base % P_MAX == 0),
        f"[QKV MLA] qr export requires each LNC shard to own whole {P_MAX}-query tiles: "
        f"S % num_shards == 0 and (S <= {P_MAX} or (S // num_shards) % {P_MAX} == 0). "
        f"Got S={S}, num_shards={num_shards} (S_shard_base={S_shard_base}).",
    )

    q_out_dim = n_heads * qk_head_dim
    kv_a_out_dim = kv_lora_rank + qk_rope_head_dim
    qkv_out_dim = qk_lora_rank + kv_a_out_dim

    """
    wqkv_a output columns are [qr | kv | k_pe]; the offline loader pre-swizzles the qr and
    kv blocks with the MLA column permutation (k_pe untouched). The compact block-128 scale
    is therefore ALWAYS shipped for the NATURAL (un-swizzled) weight -- (a block-128 scale 
    cannot follow the interleave).
    """
    wqkv_a_scale_groups = [(0, qk_lora_rank), (qk_lora_rank, kv_lora_rank)] if compact_scales else None

    # Hardware constants
    H_PACK = _H_PACK

    # Output HBM tensors are allocated by the caller and passed in.
    sbm.open_scope()

    # ==================== Load norm weights ====================
    q_norm_gamma_sb = _load_norm_weights_for_mx(q_norm_gamma_hbm, qk_lora_rank, sbm, name="q_norm_gamma")

    # bf16 gamma in [P_MAX, kv_lora_rank] natural column order for the c_kv latent
    # multiply (c_kv = rsqrt(kv) * gamma). The KV path does NOT go through an MX
    # matmul, so it needs the plain bf16 layout (not the swizzled float32 MX layout).
    kv_norm_gamma_bf16_sb = sbm.alloc_stack(
        (P_MAX, kv_lora_rank), dtype=nl.bfloat16, buffer=nl.sbuf, name="kv_norm_gamma_bf16"
    )
    nisa.dma_copy(
        dst=kv_norm_gamma_bf16_sb[0:P_MAX, 0:kv_lora_rank],
        src=kv_norm_gamma_hbm.reshape((kv_lora_rank,)).ap(
            pattern=[[0, P_MAX], [1, kv_lora_rank]],
            offset=0,
        ),
        dge_mode=dge_mode.hwdge,
    )

    norm_eps_sb = sbm.alloc_stack((P_MAX, 1), dtype=nl.bfloat16, buffer=nl.sbuf, name="norm_eps")
    nisa.memset(dst=norm_eps_sb, value=norm_eps)

    zero_bias_sb = sbm.alloc_stack((P_MAX, 1), dtype=nl.bfloat16, buffer=nl.sbuf, name="zero_bias")
    nisa.memset(dst=zero_bias_sb, value=0.0)

    # ==================== Tiling setup ====================
    S_TILE_SIZE = s_tile_pad = P_MAX

    H_512_TILE_COUNT = H // (P_MAX * H_PACK)
    QK_LORA_128_TILE_COUNT = qk_lora_rank // (P_MAX * H_PACK)

    """
    K-slab dispatch for the first (Q/KV stage 1) projection. Re-uses v32's
    cost model; the per-tile working set differs but the model is a
    conservative upper bound, so a "fits" decision stays safe.
    """
    _CALIBRATED_SBUF_BUDGET_BYTES = 240 * 1024
    _sbuf_budget = min(int(nl.tile_size.total_available_sbuf_size), _CALIBRATED_SBUF_BUDGET_BYTES)

    """
    Single-buffered slabs. Double-buffering (2) would let slab N+1's load overlap slab N's
    matmuls, but the slab sizer divides the SBUF budget by this count and at H=7168 the only
    legal slab sizes are {14,7,2,1} 512-tiles -- so 2 buffers halve the slab (14 -> 7),
    doubling the slab count and shrinking every matmul. Measured at S=128/H=7168/128-head
    that net-regressed (258 -> 282us), so keep 1.
    """
    _NUM_WQKV_A_SLAB_BUFFERS = 1
    kv_b_out_dim = n_heads * (qk_nope_head_dim + kv_lora_rank)
    """
    Absorbed-kernel prologue: this kernel holds NO un-absorbed wkv_b resident (it emits
    c_kv directly + absorbs W_uk into Q, counted separately), and loads wq_b one head-
    GROUP at a time. Modelling v32's full wkv_b + full wq_b here over-estimates the SBUF
    peak by up to ~760KB at 128 heads, which forced unnecessary wqkv_a K-slabbing and a
    per-s-tile weight reload at long sequences. The smallest head-group is 2 (the wq_b
    slab column offset must stay a multiple of 128), so estimate wq_b resident as the
    2-head group; the actual chosen WUK_HEAD_GROUP (below) is >= 2.
    """
    K_SLAB_SIZE_512 = _v32_compute_k_slab_size_512_tiles(
        H=H,
        qkv_out_dim=qkv_out_dim,
        q_out_dim=q_out_dim,
        kv_b_out_dim=kv_b_out_dim,
        qk_lora_rank=qk_lora_rank,
        kv_lora_rank=kv_lora_rank,
        n_heads=n_heads,
        qk_head_dim=qk_head_dim,
        v_head_dim=kv_lora_rank,
        qk_rope_head_dim=qk_rope_head_dim,
        sbuf_budget_bytes=_sbuf_budget,
        num_slab_buffers=_NUM_WQKV_A_SLAB_BUFFERS,
        absorbed=True,
        wq_b_resident_n=min(2, n_heads) * qk_head_dim,
    )
    NUM_K_SLABS = H_512_TILE_COUNT // K_SLAB_SIZE_512
    USE_K_SLAB = NUM_K_SLABS > 1

    if not USE_K_SLAB:
        wqkv_a_sb, wqkv_a_scale_sb = _load_mx_weights(
            wqkv_a_hbm,
            wqkv_a_scale_hbm,
            H,
            qkv_out_dim,
            sbm,
            name="wqkv_a",
            compact_scales=compact_scales,
            scale_swizzle_groups=wqkv_a_scale_groups,
        )

    SCALE_BLOCK = 128

    """
    ---- Per-head-group weight slab (Q stage-2 wq_b) ----
    wq_b full ([P_MAX, qk_lora//512, n_heads*qk_head_dim] fp8x4 = 288KB/part) exceeds the
    ~240KB stack, so it is loaded one head-GROUP at a time. Group = largest even divisor of
    n_heads whose head-loop resident set fits the stack (so as few/large slabs as possible;
    one group when it all fits). Peak holds: full bf16 W_uk + the wq_b slab + the q_group
    matmul output + a working reserve. Group must be even so the slab column offset
    (group_start*qk_head_dim, 192) stays a multiple of 128 for the compact block-128 scales.
    """
    _WQ_B_K_TILES = qk_lora_rank // (P_MAX * H_PACK)
    _wuk_resident = n_heads * kv_lora_rank * 2  # bf16, full, resident in head loop
    _WORKING_RESERVE = 60 * 1024  # prologue norms + per-tile qr/kv/rope/q_lift scratch
    _budget_for_group = _sbuf_budget - _wuk_resident - _WORKING_RESERVE

    """
    Largest even divisor of n_heads whose wq_b slab + q_group output fit
    _budget_for_group AND whose Q-stage-2 matmul output fits the 8 PSUM banks.
    The Q-stage-2 output is [s, g*qk_head_dim] laid across 512-wide PSUM banks
    -> ceil(g*qk_head_dim / 512) banks; cap below 8 (1 bank reserved for the
    absorption transpose PSUM). PSUM, not SBUF, is the binding limit on group.
    Inlined (NKI rejects inner function defs).
    """
    _F_MAX = 512
    _MAX_Q_PSUM_BANKS = _NUM_HW_PSUM_BANKS - 1
    WUK_HEAD_GROUP = 2
    for group_size in range(n_heads, 1, -1):
        if n_heads % group_size != 0 or group_size % 2 != 0:
            continue
        group_n = group_size * qk_head_dim
        if (group_n + _F_MAX - 1) // _F_MAX > _MAX_Q_PSUM_BANKS:
            continue
        group_n_padded = ((group_n + SCALE_BLOCK - 1) // SCALE_BLOCK) * SCALE_BLOCK
        group_bytes = _WQ_B_K_TILES * group_n * 4 + _WQ_B_K_TILES * group_n_padded + group_n * 2
        if group_bytes <= _budget_for_group:
            WUK_HEAD_GROUP = group_size
            break

    """
    The absorption weight W_uk (bf16) is NOT loaded here. It is dead during
    stage-1 (MM-A) and only read in the per-head absorption matmul, so holding
    it resident through stage-1 wastes stack. It is loaded per tile inside the
    i_tile scope after stage-1 finishes (see below), freeing 128KB/partition
    of stage-1 headroom.

    wq_b slab buffers ([P_MAX, num_512_tiles, WUK_HEAD_GROUP*qk_head_dim]).
    """
    WQ_B_K_TILES = qk_lora_rank // (P_MAX * H_PACK)
    wq_b_slab_n = WUK_HEAD_GROUP * qk_head_dim
    wq_b_slab_padded_n = ((wq_b_slab_n + SCALE_BLOCK - 1) // SCALE_BLOCK) * SCALE_BLOCK

    """
    ---- Budget-aware weight residency (W_uk + wq_b are s-tile-INVARIANT) ----
    The per-tile code below re-loads W_uk/wq_b once per s-tile, which at long per-rank
    sequences (head-sharded, 32 tiles) re-streams them 32x = the qkv DMA bottleneck. When
    the full W_uk + wq_b fit free SBUF (head-sharded configs: few heads/rank, tiny weights),
    load each ONCE before the s-block loop; else fall back to per-tile streaming.
    """
    q_out_dim_padded = ((q_out_dim + SCALE_BLOCK - 1) // SCALE_BLOCK) * SCALE_BLOCK
    _wuk_full_bytes = n_heads * kv_lora_rank * 2  # bf16
    _wq_b_full_bytes = WQ_B_K_TILES * q_out_dim * 4 + WQ_B_K_TILES * q_out_dim_padded

    """
    Reserve for everything that COEXISTS with the resident W_uk + wq_b at peak: the
    stage-1 wqkv_a weight slab (fp8x4 4B + padded uint8 scale, _NUM_WQKV_A_SLAB_BUFFERS
    copies), the stage-1 outputs (qr/kv_raw bf16), and the double-buffered per-tile input
    (x_qtz/x_scale, fp8x4 + uint8) plus a small head-scratch pad. Computed from the
    already-chosen K_SLAB_SIZE_512 (not a blanket constant), so the hoist triggers
    whenever the resident weights genuinely fit alongside the live stage-1 set.
    """
    _qkv_out_dim_padded = ((qkv_out_dim + SCALE_BLOCK - 1) // SCALE_BLOCK) * SCALE_BLOCK
    _stage1_slab_bytes = _NUM_WQKV_A_SLAB_BUFFERS * (
        K_SLAB_SIZE_512 * qkv_out_dim * 4 + K_SLAB_SIZE_512 * _qkv_out_dim_padded
    )
    _stage1_out_bytes = qkv_out_dim * 2  # qr + kv_raw together span qkv_out_dim bf16
    _input_buf_bytes = 2 * (H_512_TILE_COUNT * s_tile_pad * (4 + 1))  # NUM_INPUT_BUFFERS=2, fp8x4+uint8
    _HEAD_SCRATCH_RESERVE = 24 * 1024
    _stage1_reserve = _stage1_slab_bytes + _stage1_out_bytes + _input_buf_bytes + _HEAD_SCRATCH_RESERVE
    _resident_budget = int(0.95 * sbm.get_free_space()) - _stage1_reserve
    weights_resident = (_wuk_full_bytes + _wq_b_full_bytes) <= _resident_budget

    """
    ---- Pipelined head-group weight streaming (few-tile configs) ----
    Resident mode loads the FULL W_uk + wq_b up front, which stalls the TE while the big
    DMAs land — but its only benefit is reuse ACROSS s-tiles. With a single s-tile per core
    (num_S_blocks == 1, e.g. S<=256 at LNC=2) there is no reuse, so instead stream the head-
    group weight SLICES double-buffered inside the head-group loop: prefetch group g+1's
    wq_b + W_uk slice while group g computes, hiding the weight DMA behind compute. Overrides
    the resident path when it applies. Group buffers are tiny (a few tens of KB/part).
    """
    _S_BLOCK_SIZE_PRE = 2 * S_TILE_SIZE  # NUM_INPUT_BUFFERS(2) * S_TILE_SIZE
    num_head_groups = div_ceil(n_heads, WUK_HEAD_GROUP)
    weights_pipelined = div_ceil(S_shard, _S_BLOCK_SIZE_PRE) == 1 and num_head_groups > 1
    if weights_pipelined:
        weights_resident = False

    if weights_pipelined:
        # Double-buffered group-slice buffers for wq_b (+scale) and W_uk.
        wq_b_grp_bufs, wq_b_scale_grp_bufs, wuk_grp_bufs = [], [], []
        for buf_idx in range(2):
            wq_b_grp_bufs.append(
                sbm.alloc_stack(
                    (P_MAX, WQ_B_K_TILES, wq_b_slab_n),
                    dtype=nl.float8_e4m3fn_x4,
                    buffer=nl.sbuf,
                    name=f"wq_b_grp_{buf_idx}",
                )
            )
            wq_b_scale_grp_bufs.append(
                sbm.alloc_stack(
                    (P_MAX, WQ_B_K_TILES, wq_b_slab_padded_n),
                    dtype=nl.uint8,
                    buffer=nl.sbuf,
                    name=f"wq_b_scale_grp_{buf_idx}",
                )
            )
            wuk_grp_bufs.append(
                sbm.alloc_stack(
                    (qk_nope_head_dim, WUK_HEAD_GROUP * kv_lora_rank),
                    dtype=nl.bfloat16,
                    buffer=nl.sbuf,
                    name=f"wuk_grp_{buf_idx}",
                )
            )
    elif weights_resident:
        # Full W_uk + full-width wq_b, loaded ONCE below, reused across all s-tiles.
        wuk_sb = _load_wuk_bf16(wuk_hbm, n_heads, qk_nope_head_dim, kv_lora_rank, sbm, name="wuk_resident")
        wq_b_sb = sbm.alloc_stack(
            (P_MAX, WQ_B_K_TILES, q_out_dim), dtype=nl.float8_e4m3fn_x4, buffer=nl.sbuf, name="wq_b_weights_full"
        )
        wq_b_scale_sb = sbm.alloc_stack(
            (P_MAX, WQ_B_K_TILES, q_out_dim_padded), dtype=nl.uint8, buffer=nl.sbuf, name="wq_b_scales_full"
        )
        _load_mx_weights_k_slab(
            wq_b_hbm,
            wq_b_scale_hbm,
            wq_b_sb,
            wq_b_scale_sb,
            in_dim_full=qk_lora_rank,
            out_dim=q_out_dim,
            k_tile_start=0,
            k_tile_count=WQ_B_K_TILES,
            sbm=sbm,
            name="wq_b_full",
            full_out_dim=q_out_dim,
            out_col_offset=0,
            weights_dge_mode=dge_mode.hwdge,
            compact_scales=compact_scales,
        )
    else:
        # Group-sized wq_b scratch (reloaded per head-group); W_uk loaded per s-tile below.
        wq_b_sb = sbm.alloc_stack(
            (P_MAX, WQ_B_K_TILES, wq_b_slab_n), dtype=nl.float8_e4m3fn_x4, buffer=nl.sbuf, name="wq_b_weights"
        )
        wq_b_scale_sb = sbm.alloc_stack(
            (P_MAX, WQ_B_K_TILES, wq_b_slab_padded_n), dtype=nl.uint8, buffer=nl.sbuf, name="wq_b_scales"
        )

    """
    wqkv_a (stage-1) slab buffers are NOT allocated here. They are dead once
    stage-1 finishes, so holding them resident through the head loop wastes
    stack. Instead they are allocated inside a child scope wrapping only the
    stage-1 slab matmul (see the USE_K_SLAB branch below) and freed right
    after, leaving only the stage-1 outputs (qr_sb / kv_raw_sb) resident.
    """
    qkv_out_dim_padded = ((qkv_out_dim + SCALE_BLOCK - 1) // SCALE_BLOCK) * SCALE_BLOCK

    # ==================== Multi-buffering setup ====================
    NUM_INPUT_BUFFERS = 2
    S_BLOCK_SIZE = NUM_INPUT_BUFFERS * S_TILE_SIZE
    num_S_blocks = div_ceil(S_shard, S_BLOCK_SIZE)

    """
    Packed MX scale geometry: rmsnorm_mx_prefill(pack_scales=True) folds 4 H512 tiles into
    one 128-wide scale block, so scale_region = ceil(H_512/4)*128 and the stage-1 matmul reads
    each K-tile's scale via slot addressing.
    """
    N_PACKED = div_ceil(H_512_TILE_COUNT, 4)
    """
    PACKED MX input: row = [H fp8 hidden | N_PACKED*128 uint8 packed scale blocks]. The fp8
    nc_transpose lands the hidden into x_qtz_sb (fp32 view) and the packed scale blocks into
    x_scale_sb (fp8_e5m2 view, fold preserved for the stage-1 slot addressing).
    """
    _SCALE_REGION = N_PACKED * P_MAX
    _MX_ROW = H + _SCALE_REGION
    x_mx_2d = x_hbm_mx.reshape((S, _MX_ROW))

    for s_block_idx in nl.affine_range(num_S_blocks):
        sbm.open_scope()
        sbm.set_name_prefix(f"sb{s_block_idx}_")

        s_block_offset = s_block_idx * S_BLOCK_SIZE
        s_block_sz = min(S_BLOCK_SIZE, S_shard - s_block_offset)
        num_tiles_in_block = div_ceil(s_block_sz, S_TILE_SIZE)

        """
        ============================================================
        STEP 1: Pre-allocate and load input for all tiles in the block.
        ============================================================
        Scale buffer sized [P_MAX, H_512, s_tile]; the packed path uses only the first
        N_PACKED slots (N_PACKED <= H_512), folded 4-tiles per 128-block.
        """
        x_qtz_sb_bufs = []
        x_scale_sb_bufs = []
        for buf_idx in range(num_tiles_in_block):
            x_qtz_sb_bufs.append(
                sbm.alloc_stack(
                    (P_MAX, H_512_TILE_COUNT, s_tile_pad),
                    dtype=nl.float8_e4m3fn_x4,
                    buffer=nl.sbuf,
                    name=f"x_qtz_buf_{buf_idx}",
                )
            )
            x_scale_sb_bufs.append(
                sbm.alloc_stack(
                    (P_MAX, H_512_TILE_COUNT, s_tile_pad),
                    dtype=nl.uint8,
                    buffer=nl.sbuf,
                    name=f"x_scale_buf_{buf_idx}",
                )
            )

        for i_tile in range(num_tiles_in_block):
            s_tile_local_offset = s_block_offset + i_tile * S_TILE_SIZE
            s_tile_sz = min(S_TILE_SIZE, S_shard - s_tile_local_offset)
            s_tile_global_offset = S_shard_offset + s_tile_local_offset

            # fp8 nc_transpose the packed row into x_qtz_sb (hidden, fp32 view) + x_scale_sb
            # (packed scale blocks, fp8_e5m2 view). Stage-1 reads the scale via slot addressing.
            _load_prequant_tile(
                x_mx_2d,
                x_qtz_sb_bufs[i_tile],
                x_scale_sb_bufs[i_tile],
                s_tile_global_offset,
                s_tile_sz,
                s_tile_pad,
                H,
                H_512_TILE_COUNT,
                N_PACKED,
                _MX_ROW,
                sbm,
            )

        """
        ============================================================
        STEP 2: Process each tile in the block.
        ============================================================
        """
        for i_tile in nl.affine_range(num_tiles_in_block):
            sbm.open_scope()
            sbm.set_name_prefix(f"sb{s_block_idx}_t{i_tile}_")

            s_tile_local_offset = s_block_offset + i_tile * S_TILE_SIZE
            s_tile_sz = min(S_TILE_SIZE, S_shard - s_tile_local_offset)
            s_tile_global_offset = S_shard_offset + s_tile_local_offset

            x_qtz_sb = x_qtz_sb_bufs[i_tile]
            x_scale_sb = x_scale_sb_bufs[i_tile]

            # Prefetch RoPE caches early (overlaps with matmul compute).
            cos_sb = sbm.alloc_stack((P_MAX, qk_rope_head_dim), dtype=nl.bfloat16, buffer=nl.sbuf, name="cos_cache")
            sin_sb = sbm.alloc_stack(
                (P_MAX, qk_rope_head_dim // 2), dtype=nl.bfloat16, buffer=nl.sbuf, name="sin_cache"
            )
            rope_offset = s_tile_global_offset * qk_rope_head_dim
            nisa.dma_copy(
                dst=cos_sb[0:s_tile_sz, 0:qk_rope_head_dim],
                src=cos_cache_hbm.ap(
                    pattern=[[qk_rope_head_dim, s_tile_sz], [1, qk_rope_head_dim]],
                    offset=rope_offset,
                ),
                dge_mode=dge_mode.hwdge,
            )
            nisa.dma_copy(
                dst=sin_sb[0:s_tile_sz, 0 : qk_rope_head_dim // 2],
                src=sin_cache_hbm.ap(
                    pattern=[[qk_rope_head_dim, s_tile_sz], [1, qk_rope_head_dim // 2]],
                    offset=rope_offset,
                ),
                dge_mode=dge_mode.hwdge,
            )

            """
            ============================================================
            Combined Q/KV Stage 1 - x @ wqkv_a -> qr, kv + k_pe
            ============================================================
            """
            if not USE_K_SLAB:
                qr_sb, kv_raw_sb = _mx_matmul_split(
                    x_qtz_sb,
                    x_scale_sb,
                    wqkv_a_sb,
                    wqkv_a_scale_sb,
                    H_512_TILE_COUNT,
                    s_tile_pad,
                    qkv_out_dim,
                    [qk_lora_rank],
                    sbm,
                    input_scale_packed=True,
                )
            else:
                F_MAX = 512
                PSUM_BANK_SIZE = _get_psum_bank_size()
                num_n_tiles = (qkv_out_dim + F_MAX - 1) // F_MAX

                """
                Stage-1 outputs must survive for the rest of this tile, so
                pre-allocate them in the parent scope (before the stage-1
                child scope). The stage-1 weight slab buffers live and die
                inside the child scope and never reach the head loop.
                """
                kv_raw_width = qkv_out_dim - qk_lora_rank
                qr_sb = sbm.alloc_stack(
                    (P_MAX, qk_lora_rank), dtype=nl.bfloat16, buffer=nl.sbuf, name="matmul_split_out_0"
                )
                kv_raw_sb = sbm.alloc_stack(
                    (P_MAX, kv_raw_width), dtype=nl.bfloat16, buffer=nl.sbuf, name="matmul_split_out_1"
                )

                # ---- Stage-1 child scope: slab weights freed on close ----
                sbm.open_scope()

                wqkv_a_slab_bufs = []
                wqkv_a_scale_slab_bufs = []
                for buf_idx in range(_NUM_WQKV_A_SLAB_BUFFERS):
                    wqkv_a_slab_bufs.append(
                        sbm.alloc_stack(
                            (P_MAX, K_SLAB_SIZE_512, qkv_out_dim),
                            dtype=nl.float8_e4m3fn_x4,
                            buffer=nl.sbuf,
                            name=f"wqkv_a_slab_buf_{buf_idx}",
                        )
                    )
                    wqkv_a_scale_slab_bufs.append(
                        sbm.alloc_stack(
                            (P_MAX, K_SLAB_SIZE_512, qkv_out_dim_padded),
                            dtype=nl.uint8,
                            buffer=nl.sbuf,
                            name=f"wqkv_a_scale_slab_buf_{buf_idx}",
                        )
                    )

                # PSUM is a separate space (not the SBUF stack), so output_psum
                # survives the child-scope close and is read out below.
                output_psum = []
                for bank_id in range(num_n_tiles):
                    output_psum.append(
                        nl.ndarray(
                            (P_MAX, F_MAX),
                            dtype=nl.bfloat16,
                            buffer=nl.psum,
                            address=(0, bank_id * PSUM_BANK_SIZE),
                        )
                    )

                for slab_id in range(NUM_K_SLABS):
                    cur_buf_idx = slab_id % _NUM_WQKV_A_SLAB_BUFFERS
                    slab_buf = wqkv_a_slab_bufs[cur_buf_idx]
                    slab_scale_buf = wqkv_a_scale_slab_bufs[cur_buf_idx]
                    _load_mx_weights_k_slab(
                        wqkv_a_hbm,
                        wqkv_a_scale_hbm,
                        slab_buf,
                        slab_scale_buf,
                        in_dim_full=H,
                        out_dim=qkv_out_dim,
                        k_tile_start=slab_id * K_SLAB_SIZE_512,
                        k_tile_count=K_SLAB_SIZE_512,
                        sbm=sbm,
                        name=f"slab{slab_id}",
                        # Plain rectangular weight read -> HW descriptor engine.
                        weights_dge_mode=dge_mode.hwdge,
                        compact_scales=compact_scales,
                        scale_swizzle_groups=wqkv_a_scale_groups,
                    )
                    _mx_matmul_split_k_range(
                        input_qtz_sb=x_qtz_sb,
                        input_scale_sb=x_scale_sb,
                        weights_slab_sb=slab_buf,
                        weights_slab_scale_sb=slab_scale_buf,
                        k_tile_start=slab_id * K_SLAB_SIZE_512,
                        k_tile_count=K_SLAB_SIZE_512,
                        m_dim=s_tile_pad,
                        n_dim=qkv_out_dim,
                        output_psum=output_psum,
                        input_scale_packed=True,
                    )

                # Copy stage-1 PSUM into the parent-scope buffers, then free the
                # slab weights by closing the child scope.
                width = qk_lora_rank
                col = 0
                while col < width:
                    bank_idx, bank_offset = divmod(col, F_MAX)
                    copy_width = min(F_MAX - bank_offset, width - col)
                    nisa.tensor_copy(
                        dst=qr_sb[0:s_tile_pad, nl.ds(col, copy_width)],
                        src=output_psum[bank_idx][0:s_tile_pad, nl.ds(bank_offset, copy_width)],
                        engine=nisa.scalar_engine,
                    )
                    col += copy_width

                col = 0
                while col < kv_raw_width:
                    global_col = qk_lora_rank + col
                    bank_idx, bank_offset = divmod(global_col, F_MAX)
                    copy_width = min(F_MAX - bank_offset, kv_raw_width - col)
                    nisa.tensor_copy(
                        dst=kv_raw_sb[0:s_tile_pad, nl.ds(col, copy_width)],
                        src=output_psum[bank_idx][0:s_tile_pad, nl.ds(bank_offset, copy_width)],
                        engine=nisa.scalar_engine,
                    )
                    col += copy_width

                sbm.close_scope()

            """
            Interleave: kick off group 0's wq_b + W_uk prefetch NOW, right after stage-1
            (wqkv_a) finishes. These weight loads have no dependency on qr/kv, so their DMA
            overlaps the qr norm/transpose/quantize + kv path below (a compute window that
            would otherwise leave the DMA engine idle), instead of stalling the head loop.
            """
            if weights_pipelined:
                _g0_count = min(WUK_HEAD_GROUP, n_heads)
                _load_mx_weights_k_slab(
                    wq_b_hbm,
                    wq_b_scale_hbm,
                    wq_b_grp_bufs[0],
                    wq_b_scale_grp_bufs[0],
                    in_dim_full=qk_lora_rank,
                    out_dim=_g0_count * qk_head_dim,
                    k_tile_start=0,
                    k_tile_count=WQ_B_K_TILES,
                    sbm=sbm,
                    name="wq_b_pg0",
                    full_out_dim=q_out_dim,
                    out_col_offset=0,
                    compact_scales=compact_scales,
                    weights_dge_mode=dge_mode.hwdge,
                )
                _load_wuk_bf16_group(wuk_hbm, wuk_grp_bufs[0], n_heads, qk_nope_head_dim, kv_lora_rank, 0, _g0_count)

            """
            ============================================================
            Q Path - RMSNorm (gamma fused into transpose below)
            ============================================================
            """
            _apply_rms_norm_inplace(qr_sb, zero_bias_sb, norm_eps_sb, s_tile_pad, qk_lora_rank, sbm, name="q_rms")

            """
            ============================================================
            Q Path Stage 2 - prepare qr_normed (quantized). The wq_b matmul
            itself is N-slabbed per head-group inside the head loop below
            (wq_b is too large to hold resident; see slab buffer comments).
            ============================================================
            """
            qr_transposed_sb = _transpose_preswizzled_for_mx_fused(
                qr_sb,
                s_tile_pad,
                qk_lora_rank,
                QK_LORA_128_TILE_COUNT,
                sbm,
                gamma_sb=q_norm_gamma_sb,
                name="qr_transpose",
            )
            qr_qtz_sb, qr_scale_sb = _quantize_mx(
                qr_transposed_sb, QK_LORA_128_TILE_COUNT, s_tile_pad, sbm, name="qr_qtz"
            )

            """
            Export the quantized qr latent for the SAI indexer (skips its own
            transpose+norm+quantize). One s-tile block [P_MAX, QK_LORA_128_TILE_COUNT,
            P_MAX] per global s-tile index (s_tile_global_offset // P_MAX).
            qr_qtz_hbm is uint32; view the fp8x4 SBUF src as uint32 so the SBUF->HBM
            DMA is uint32->uint32 (same 4B width, no MX cast).
            """
            # Write this tile's valid s-columns at their WITHIN-TILE offset so LNC shards whose
            # first query is not 128-aligned (S_shard < P_MAX -> multiple cores share one global
            # tile) land in disjoint columns instead of both overwriting column 0. The guard at
            # the top of the stage rejects shards that would straddle a tile boundary.
            _qr_tile = s_tile_global_offset // S_TILE_SIZE
            _qr_col = s_tile_global_offset % S_TILE_SIZE
            _qr_sb_u32 = qr_qtz_sb.view(nl.uint32)
            nisa.dma_copy(
                dst=qr_qtz_hbm[_qr_tile][0:P_MAX, 0:QK_LORA_128_TILE_COUNT, nl.ds(_qr_col, s_tile_sz)],
                src=_qr_sb_u32[0:P_MAX, 0:QK_LORA_128_TILE_COUNT, 0:s_tile_sz],
            )
            nisa.dma_copy(
                dst=qr_scale_hbm[_qr_tile][0:P_MAX, 0:QK_LORA_128_TILE_COUNT, nl.ds(_qr_col, s_tile_sz)],
                src=qr_scale_sb[0:P_MAX, 0:QK_LORA_128_TILE_COUNT, 0:s_tile_sz],
            )

            """
            ============================================================
            Split kv_raw into kv (latent) and k_pe
            ============================================================
            """
            kv_sb = sbm.alloc_stack((P_MAX, kv_lora_rank), dtype=nl.bfloat16, buffer=nl.sbuf, name="kv_split")
            nisa.tensor_copy(
                dst=kv_sb[0:s_tile_sz, 0:kv_lora_rank],
                src=kv_raw_sb[0:s_tile_sz, 0:kv_lora_rank],
                engine=nisa.scalar_engine,
            )

            k_pe_sb = sbm.alloc_stack((P_MAX, qk_rope_head_dim), dtype=nl.bfloat16, buffer=nl.sbuf, name="k_pe_split")
            nisa.tensor_copy(
                dst=k_pe_sb[0:s_tile_sz, 0:qk_rope_head_dim],
                src=kv_raw_sb[0:s_tile_sz, kv_lora_rank : kv_lora_rank + qk_rope_head_dim],
                engine=nisa.scalar_engine,
            )

            """
            ============================================================
            KV Path - RMSNorm (rsqrt) on kv. Gamma applied below into c_kv.
            ============================================================
            """
            _apply_rms_norm_inplace(kv_sb, zero_bias_sb, norm_eps_sb, s_tile_sz, kv_lora_rank, sbm, name="kv_rms")

            """
            kv_sb latent columns are in *swizzled* order (wqkv_a kv columns are
            pre-swizzled). The c_kv latent must be in *natural* column order for
            the downstream attention kernel, and the natural-order gamma must
            multiply natural columns. Un-swizzle once, then apply gamma.
            """
            kv_natural_sb = _unswizzle_lora_cols(kv_sb, s_tile_sz, kv_lora_rank, sbm, name="kv_natural")
            c_kv_sb = sbm.alloc_stack((P_MAX, kv_lora_rank), dtype=nl.bfloat16, buffer=nl.sbuf, name="c_kv")
            nisa.tensor_tensor(
                dst=c_kv_sb[0:s_tile_sz, 0:kv_lora_rank],
                data1=kv_natural_sb[0:s_tile_sz, 0:kv_lora_rank],
                data2=kv_norm_gamma_bf16_sb[0:s_tile_sz, 0:kv_lora_rank],
                op=nl.multiply,
            )

            k_pe_rope_sb = _apply_rope_to_tensor_interleaved(
                k_pe_sb, cos_sb, sin_sb, s_tile_sz, qk_rope_head_dim, sbm, name="k_pe_rope"
            )
            """
            Per-head: q_pe RoPE + absorption q_nope @ W_uk -> q_lift. Each head's q_lift/q_pe
            stream straight to HBM inside the head loop; accumulating all heads in SBUF first
            would overflow the stack at n_heads=128 (~144KB).
            
            W_uk (bf16) is loaded here (after stage-1's slab weights are freed, so it isn't
            resident during MM-A), in the i_tile scope so it survives all head groups. When
            weights_resident it was loaded once before the s-block loop and is reused (no
            per-tile reload — kills the dominant qkv weight DMA at long head-sharded seqs).
            Pipelined mode gets its W_uk from the double-buffered group slices (below);
            only the per-tile streaming fallback loads the full W_uk here.
            """
            if not weights_resident and not weights_pipelined:
                wuk_sb = _load_wuk_bf16(wuk_hbm, n_heads, qk_nope_head_dim, kv_lora_rank, sbm, name="wuk")

            rope_temp_sb = sbm.alloc_stack(
                (P_MAX, qk_rope_head_dim * 2), dtype=nl.bfloat16, buffer=nl.sbuf, name="q_rope_scratch"
            )

            F_MAX = 512
            PSUM_BANK_SIZE = _get_psum_bank_size()
            num_lift_n_tiles = div_ceil(kv_lora_rank, F_MAX)
            num_head_groups = div_ceil(n_heads, WUK_HEAD_GROUP)

            for group_idx in range(num_head_groups):
                group_start = group_idx * WUK_HEAD_GROUP
                group_count = min(WUK_HEAD_GROUP, n_heads - group_start)

                """
                Per-group scope: the Q stage-2 matmul output (q_group_sb) and
                the wq_b compact scales are only needed for this group's heads.
                Without this scope they accumulate across all head groups and
                overflow the stack at n_heads=128.
                """
                sbm.open_scope()

                # Q stage-2 for this head group: matmul qr_normed @ wq_b ->
                # q_group[s, group_count*qk_head_dim].
                group_n = group_count * qk_head_dim
                if weights_pipelined:
                    cur = group_idx % 2
                    # Prefetch NEXT group's wq_b + W_uk into the alternate buffer so its DMA
                    # overlaps THIS group's compute (matmul + per-head rope/absorption/evict).
                    if group_idx + 1 < num_head_groups:
                        nxt = (group_idx + 1) % 2
                        n_start = (group_idx + 1) * WUK_HEAD_GROUP
                        n_count = min(WUK_HEAD_GROUP, n_heads - n_start)
                        _load_mx_weights_k_slab(
                            wq_b_hbm,
                            wq_b_scale_hbm,
                            wq_b_grp_bufs[nxt],
                            wq_b_scale_grp_bufs[nxt],
                            in_dim_full=qk_lora_rank,
                            out_dim=n_count * qk_head_dim,
                            k_tile_start=0,
                            k_tile_count=WQ_B_K_TILES,
                            sbm=sbm,
                            name=f"wq_b_pg{group_idx + 1}",
                            full_out_dim=q_out_dim,
                            out_col_offset=n_start * qk_head_dim,
                            weights_dge_mode=dge_mode.hwdge,
                            compact_scales=compact_scales,
                        )
                        _load_wuk_bf16_group(
                            wuk_hbm, wuk_grp_bufs[nxt], n_heads, qk_nope_head_dim, kv_lora_rank, n_start, n_count
                        )
                    wuk_sb = wuk_grp_bufs[cur]  # group-local: head slot local_head_idx*kv_lora
                    q_group_sb = _mx_matmul(
                        qr_qtz_sb,
                        qr_scale_sb,
                        wq_b_grp_bufs[cur],
                        wq_b_scale_grp_bufs[cur],
                        WQ_B_K_TILES,
                        s_tile_pad,
                        group_n,
                        sbm,
                        name=f"q_matmul_g{group_idx}",
                    )
                elif weights_resident:
                    """
                    Full wq_b already resident: index this group's columns directly
                    (no per-group reload). _mx_matmul reads weights at free offset 0,
                    so pass a column-offset view of the full buffer for this group.
                    """
                    grp_col = group_start * qk_head_dim
                    grp_scale_col = grp_col  # group_n & grp_col are multiples of 128 (SCALE_BLOCK)
                    wq_b_grp = wq_b_sb[0:P_MAX, 0:WQ_B_K_TILES, nl.ds(grp_col, group_n)]
                    wq_b_scale_grp = wq_b_scale_sb[0:P_MAX, 0:WQ_B_K_TILES, nl.ds(grp_scale_col, group_n)]
                    q_group_sb = _mx_matmul(
                        qr_qtz_sb,
                        qr_scale_sb,
                        wq_b_grp,
                        wq_b_scale_grp,
                        WQ_B_K_TILES,
                        s_tile_pad,
                        group_n,
                        sbm,
                        name=f"q_matmul_g{group_idx}",
                    )
                else:
                    _load_mx_weights_k_slab(
                        wq_b_hbm,
                        wq_b_scale_hbm,
                        wq_b_sb,
                        wq_b_scale_sb,
                        in_dim_full=qk_lora_rank,
                        out_dim=group_n,
                        k_tile_start=0,
                        k_tile_count=WQ_B_K_TILES,
                        sbm=sbm,
                        name=f"wq_b_g{group_idx}",
                        full_out_dim=q_out_dim,
                        out_col_offset=group_start * qk_head_dim,
                        # Plain rectangular strided weight read -> HW descriptor engine (scales still swdge inside).
                        weights_dge_mode=dge_mode.hwdge,
                        compact_scales=compact_scales,
                    )
                    q_group_sb = _mx_matmul(
                        qr_qtz_sb,
                        qr_scale_sb,
                        wq_b_sb,
                        wq_b_scale_sb,
                        WQ_B_K_TILES,
                        s_tile_pad,
                        group_n,
                        sbm,
                        name=f"q_matmul_g{group_idx}",
                    )

                """
                Batch this group's q_pe / q_lift into ONE DMA each: each head writes its slot
                of these group buffers, then a single strided DMA per buffer stores the whole
                group's contiguous head-range. Replaces 2*group_count tiny per-head strided
                DMAs (the profiled write hotspot) with 2. group buffers: group_count*(R+L) bf16.
                """
                q_pe_grp_sb = sbm.alloc_stack(
                    (P_MAX, group_count * qk_rope_head_dim), dtype=nl.bfloat16, buffer=nl.sbuf, name="q_pe_grp"
                )
                q_lift_grp_sb = sbm.alloc_stack(
                    (P_MAX, group_count * kv_lora_rank), dtype=nl.bfloat16, buffer=nl.sbuf, name="q_lift_grp"
                )

                for local_head_idx in nl.affine_range(group_count):
                    head_idx = group_start + local_head_idx
                    sbm.open_scope()
                    sbm.set_name_prefix(f"sb{s_block_idx}_t{i_tile}_h{head_idx}_")

                    # Offsets are local to q_group_sb (this head group's slab).
                    q_head_offset = local_head_idx * qk_head_dim
                    q_pe_offset = q_head_offset + qk_nope_head_dim

                    # ----- q_pe -> group-buffer slot. INTERLEAVED path: copy raw here, RoPE the
                    # whole group ONCE after the loop. Half-split path: rope per-head via scratch. -----
                    nisa.tensor_copy(
                        dst=q_pe_grp_sb[0:s_tile_sz, nl.ds(local_head_idx * qk_rope_head_dim, qk_rope_head_dim)],
                        src=q_group_sb[0:s_tile_sz, q_pe_offset : q_pe_offset + qk_rope_head_dim],
                        engine=nisa.scalar_engine,
                    )
                    """
                    ----- Absorption: q_nope[s, nope] @ W_uk[h][nope, kv_lora] -> q_lift[s, kv_lora] -----
                    bf16 contraction = nope = 128 (full partition). A single
                    nc_transpose lifts q_nope[s, nope] -> q_nope_t[nope, s], then a
                    plain nc_matmul with stationary=q_nope_t and moving=W_uk[h]
                    yields q_lift[s, kv_lora].
                    """
                    q_nope_head_sb = q_group_sb.ap(
                        pattern=[[group_n, s_tile_sz], [1, qk_nope_head_dim]],
                        offset=q_head_offset,
                    )
                    qnope_t_psum = nl.ndarray(
                        (qk_nope_head_dim, P_MAX),
                        dtype=nl.bfloat16,
                        buffer=nl.psum,
                        address=(0, num_lift_n_tiles * PSUM_BANK_SIZE),
                    )
                    nisa.nc_transpose(
                        data=q_nope_head_sb,
                        dst=qnope_t_psum[0:qk_nope_head_dim, 0:s_tile_sz],
                    )
                    qnope_t_sb = sbm.alloc_stack(
                        (qk_nope_head_dim, P_MAX), dtype=nl.bfloat16, buffer=nl.sbuf, name="qnope_t"
                    )
                    nisa.tensor_copy(
                        dst=qnope_t_sb[0:qk_nope_head_dim, 0:s_tile_sz],
                        src=qnope_t_psum[0:qk_nope_head_dim, 0:s_tile_sz],
                        engine=nisa.scalar_engine,
                    )

                    lift_psum = []
                    for bank_id in range(num_lift_n_tiles):
                        lift_psum.append(
                            nl.ndarray(
                                (P_MAX, F_MAX),
                                dtype=nl.bfloat16,
                                buffer=nl.psum,
                                address=(0, bank_id * PSUM_BANK_SIZE),
                            )
                        )
                    for i_n in nl.affine_range(num_lift_n_tiles):
                        n_tile_sz = min(F_MAX, kv_lora_rank - i_n * F_MAX)
                        # W_uk column for this head: group-local slot in pipelined mode (wuk_sb is
                        # the group slice), global slot in resident/streaming mode (full buffer).
                        _wuk_head = local_head_idx if weights_pipelined else head_idx
                        w_col = _wuk_head * kv_lora_rank + i_n * F_MAX
                        nisa.nc_matmul(
                            lift_psum[i_n][0:s_tile_sz, 0:n_tile_sz],
                            qnope_t_sb[0:qk_nope_head_dim, 0:s_tile_sz],
                            wuk_sb[0:qk_nope_head_dim, nl.ds(w_col, n_tile_sz)],
                        )
                    # Copy this head's absorbed q_lift into its slot of the group buffer (no
                    # per-head DMA — the whole group is written in one DMA after the loop).
                    _lift_base = local_head_idx * kv_lora_rank
                    for i_n in nl.affine_range(num_lift_n_tiles):
                        n_tile_sz = min(F_MAX, kv_lora_rank - i_n * F_MAX)
                        nisa.tensor_copy(
                            dst=q_lift_grp_sb[0:s_tile_sz, nl.ds(_lift_base + i_n * F_MAX, n_tile_sz)],
                            src=lift_psum[i_n][0:s_tile_sz, 0:n_tile_sz],
                            engine=nisa.scalar_engine,
                        )

                    sbm.close_scope()

                # ----- Batched RoPE over the whole group's q_pe (interleaved path): 6 wide
                # tensor_tensor ops instead of 6 per head. Same cos/sin broadcast across heads. -----
                rope_grp_scratch = sbm.alloc_stack(
                    (P_MAX, group_count * qk_rope_head_dim), dtype=nl.bfloat16, buffer=nl.sbuf, name="rope_grp_scratch"
                )
                _apply_rope_inplace_interleaved_grouped(
                    q_pe_grp_sb, cos_sb, sin_sb, rope_grp_scratch, s_tile_sz, qk_rope_head_dim, group_count
                )

                """
                ----- One DMA each for this group's q_pe / q_lift (contiguous head-range) -----
                Heads [group_start, group_start+group_count) are contiguous in HBM, so the group
                buffer [s_tile, group_count*R] maps to a single strided (row-stride n_heads*R) DMA.
                """
                nisa.dma_copy(
                    dst=q_pe_hbm.ap(
                        pattern=[[n_heads * qk_rope_head_dim, s_tile_sz], [1, group_count * qk_rope_head_dim]],
                        offset=s_tile_global_offset * n_heads * qk_rope_head_dim + group_start * qk_rope_head_dim,
                    ),
                    src=q_pe_grp_sb[0:s_tile_sz, 0 : group_count * qk_rope_head_dim],
                    dge_mode=dge_mode.hwdge,
                )
                nisa.dma_copy(
                    dst=q_lift_hbm.ap(
                        pattern=[[n_heads * kv_lora_rank, s_tile_sz], [1, group_count * kv_lora_rank]],
                        offset=s_tile_global_offset * n_heads * kv_lora_rank + group_start * kv_lora_rank,
                    ),
                    src=q_lift_grp_sb[0:s_tile_sz, 0 : group_count * kv_lora_rank],
                    dge_mode=dge_mode.hwdge,
                )

                # Free this group's q_group_sb + wq_b scales before the next group.
                sbm.close_scope()

            """
            ============================================================
            Store shared c_kv, k_pe to HBM (q_lift / q_pe were streamed
            per-head inside the head loop above).
            ============================================================
            """
            nisa.dma_copy(
                dst=c_kv_hbm.ap(
                    pattern=[[kv_lora_rank, s_tile_sz], [1, kv_lora_rank]],
                    offset=s_tile_global_offset * kv_lora_rank,
                ),
                src=c_kv_sb[0:s_tile_sz, 0:kv_lora_rank],
                dge_mode=dge_mode.hwdge,
            )
            nisa.dma_copy(
                dst=k_pe_hbm.ap(
                    pattern=[[qk_rope_head_dim, s_tile_sz], [1, qk_rope_head_dim]],
                    offset=s_tile_global_offset * qk_rope_head_dim,
                ),
                src=k_pe_rope_sb[0:s_tile_sz, 0:qk_rope_head_dim],
                dge_mode=dge_mode.hwdge,
            )

            sbm.close_scope()

        sbm.close_scope()

    sbm.close_scope()


def _load_wuk_bf16(
    wuk_hbm: nl.NkiTensor,
    n_heads: int,
    qk_nope_head_dim: int,
    kv_lora_rank: int,
    sbm: SbufManager,
    name: str = "wuk",
) -> nl.NkiTensor:
    """Load the absorption weight ``W_uk`` (bf16) fully resident in SBUF.

    ``W_uk`` logically is ``[nope, n_heads * kv_lora]``: the contraction is
    ``nope`` (128 = all partitions) and the output is ``n_heads * kv_lora``.
    At n_heads=128, kv_lora=512 this is ``[128, 65536]`` bf16 = 128KB on the
    free axis over 128 partitions, which fits SBUF, so it is loaded once with a
    single DMA and indexed per head (no slabbing, no scales).

    Returns ``wuk_sb`` shaped ``[nope, n_heads * kv_lora]`` bf16; head ``h``
    owns columns ``[h * kv_lora, (h + 1) * kv_lora)``.
    """
    out_dim = n_heads * kv_lora_rank
    wuk_sb = sbm.alloc_stack((qk_nope_head_dim, out_dim), dtype=nl.bfloat16, buffer=nl.sbuf, name=f"{name}_weights")
    nisa.dma_copy(
        dst=wuk_sb[0:qk_nope_head_dim, 0:out_dim],
        src=wuk_hbm.ap(
            pattern=[[out_dim, qk_nope_head_dim], [1, out_dim]],
            offset=0,
        ),
        dge_mode=dge_mode.hwdge,
    )
    return wuk_sb


def _load_wuk_bf16_group(
    wuk_hbm: nl.NkiTensor,
    wuk_grp_sb: nl.NkiTensor,
    n_heads: int,
    qk_nope_head_dim: int,
    kv_lora_rank: int,
    group_start: int,
    group_count: int,
) -> None:
    """Load ONE head-group's slice of W_uk (bf16) into a caller-provided buffer.

    Columns [group_start*kv_lora, (group_start+group_count)*kv_lora) of the full
    ``[nope, n_heads*kv_lora]`` weight -> ``wuk_grp_sb[0:nope, 0:group_count*kv_lora]``.
    One rectangular hwdge DMA. Used by the double-buffered head-group weight pipeline so
    each group's W_uk slice loads while the previous group computes.
    """
    full_out = n_heads * kv_lora_rank
    grp_out = group_count * kv_lora_rank
    nisa.dma_copy(
        dst=wuk_grp_sb[0:qk_nope_head_dim, 0:grp_out],
        src=wuk_hbm.ap(
            pattern=[[full_out, qk_nope_head_dim], [1, grp_out]],
            offset=group_start * kv_lora_rank,
        ),
        dge_mode=dge_mode.hwdge,
    )


def _load_prequant_tile(
    x_mx_2d, x_qtz_sb, x_scale_sb, s_offset, s_tile_sz, s_tile_pad, H, num_H512, num_packed, mx_row, sbm
):
    """fp8-transpose one s-tile of the PACKED pre-quantized input (rmsnorm_mx_prefill
    pack_scales=True) into the kernel's MX consume layout.

    ``x_mx_2d`` is [S, mx_row] fp8 where mx_row = H + num_packed*128: each row is
    [H fp8 hidden | num_packed 128-wide uint8 PACKED scale blocks (4 H512 tiles folded per
    block, tile k at within-quadrant offset (k%4)*4)]. Transposes:
      - hidden region -> ``x_qtz_sb`` [P_MAX, num_H512, s_tile_pad] fp8x4 (fp32 view, 4 lanes/word),
      - packed scale region -> ``x_scale_sb`` [P_MAX, num_packed(<=num_H512 slots), s_tile_pad]
        uint8 (fp8_e5m2 view),
    one PE transpose per block. The fold is preserved: the stage-1 matmul reads K-tile k's
    scale stripe at ``x_scale_sb[(k%4)*4:, k//4, :]`` (is_packed_moving_scale addressing).

    ``s_tile_sz`` (<= ``s_tile_pad``) is the count of REAL tokens loaded/transposed; buffers and
    the downstream matmul use the padded ``s_tile_pad`` (128) width, so the partition stride and
    per-slot offset use ``s_tile_pad``. When the LNC S-shard leaves a partial tile
    (s_tile_sz < s_tile_pad) the extra columns are padding tokens whose matmul output is discarded.
    """
    P_MAX = nl.tile_size.pmax
    PSUM_BANK_SIZE = _get_psum_bank_size()
    row_fp32 = mx_row // _FP8_PER_FP32  # per-row free stride in fp32 elements

    sbm.open_scope()
    # nc_transpose requires an SBUF source, so stage the packed rows [s_tile, mx_row] in SBUF
    # first (plain dma_copy), then fp8-transpose from there.
    stage_sb = sbm.alloc_stack(
        (P_MAX, mx_row), dtype=nl.float8_e4m3fn, buffer=nl.sbuf, name=f"prequant_stage_{s_offset}"
    )
    nisa.dma_copy(
        dst=stage_sb[0:s_tile_sz, 0:mx_row],
        src=x_mx_2d.ap(pattern=[[mx_row, s_tile_sz], [1, mx_row]], offset=s_offset * mx_row),
    )
    # One PSUM bank for the fp32-view quant transpose, one for the fp8-view scale transpose.
    q_psum = nl.ndarray((P_MAX, s_tile_sz), dtype=_FP8X4_TP_VIEW_DTYPE, buffer=nl.psum, address=(0, 0))
    s_psum = nl.ndarray(
        (P_MAX, s_tile_sz, _FP8_TP_OUT_STEP), dtype=_UINT8_TP_VIEW_DTYPE, buffer=nl.psum, address=(0, PSUM_BANK_SIZE)
    )

    # Buffers are [P_MAX, num_H512, s_tile_pad]: partition stride = num_H512 * s_tile_pad, slot
    # k at column offset k * s_tile_pad. Transpose fills only the s_tile_sz real columns.
    qtz_part_stride = num_H512 * s_tile_pad

    # Hidden: one 512-tile per PE transpose (fp32 view packs 4 consecutive fp8).
    for h512_tile_idx in range(num_H512):
        nisa.nc_transpose(
            data=stage_sb.ap(
                pattern=[[row_fp32, s_tile_sz], [1, P_MAX]],
                offset=h512_tile_idx * P_MAX,
                dtype=_FP8X4_TP_VIEW_DTYPE,
            ),
            dst=q_psum[:, 0:s_tile_sz],
        )
        nisa.tensor_copy(
            dst=x_qtz_sb.ap(
                pattern=[[qtz_part_stride, P_MAX], [1, s_tile_sz]],
                offset=h512_tile_idx * s_tile_pad,
                dtype=_FP8X4_TP_VIEW_DTYPE,
            ),
            src=q_psum[:, 0:s_tile_sz],
        )
    # Scale: one PACKED 128-wide block per transpose (the fold in the input is preserved).
    for packed_idx in range(num_packed):
        nisa.nc_transpose(
            data=stage_sb.ap(
                pattern=[[mx_row, s_tile_sz], [1, P_MAX]],
                offset=H + packed_idx * P_MAX,
                dtype=_UINT8_TP_VIEW_DTYPE,
            ),
            dst=s_psum[:, 0:s_tile_sz, 0],
        )
        # Packed block packed_idx lands at slot [:, packed_idx, :] of the [P_MAX, num_H512, s_tile_pad] buffer.
        nisa.tensor_copy(
            dst=x_scale_sb.ap(
                pattern=[[qtz_part_stride, P_MAX], [1, s_tile_sz]],
                offset=packed_idx * s_tile_pad,
                dtype=_UINT8_TP_VIEW_DTYPE,
            ),
            src=s_psum[:, 0:s_tile_sz, 0],
        )
    sbm.close_scope()
