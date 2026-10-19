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
from ....core.utils.kernel_helpers import div_ceil, get_max_positive_value_for_dtype, get_program_sharding_info
from .mla_bf16_cte import (
    _BF16_K_TILE,
    _NUM_TRANSPOSE_BANKS,
    _absorb_q_nope_bf16,
    _alloc_matmul_psum,
    _bf16_compute_k_slab_tiles,
    _bf16_k_tile_sizes,
    _bf16_matmul,
    _bf16_matmul_k_range,
    _bf16_matmul_split,
    _bf16_sbuf_footprint,
    _copy_psum_splits,
    _load_bf16_gamma,
    _load_bf16_weights,
    _load_bf16_weights_k_slab,
    _load_wuk_bf16_group_tiled,
    _load_x_bf16_tile,
    _transpose_bf16_to_k_major,
    _transpose_q_nope_group_bf16,
)
from .mla_common_cte import (
    _H_PACK,
    _NUM_HW_PSUM_BANKS,
    MlaPrecision,
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
from .mla_validate_params import _validate_mla_qkv_inputs, _validate_paged_kv_store

# Views for fp8-transposing the pre-quantized packed input: the fp8 hidden rides an fp32 view
# (4 fp8 per word) and the uint8 scales an fp8_e5m2 view. An fp8 transpose requires a PSUM
# output element-step of 2.
_FP8X4_TP_VIEW_DTYPE = nl.float32
_UINT8_TP_VIEW_DTYPE = nl.float8_e5m2
_FP8_PER_FP32 = 4
_FP8_TP_OUT_STEP = 2
# Floor on the per-token amax so an all-zero (padding) token cannot divide by zero.
_AMAX_FLOOR = 1e-30


# Unroll the s-block loop up to this many blocks.
_S_BLOCK_UNROLL_MAX = 8
# S-blocks unrolled INSIDE one device-loop iteration.
_S_BLOCKS_PER_ITER = 8


def _s_ap(tensor, pattern, elem_offset, s_dim, sbase_sb):
    """HBM access pattern whose S base is supplied at RUNTIME under the device-loop driver.

    ``elem_offset`` is the trace-time part (the shard base plus the tile's offset within its
    block, times the per-token stride). ``sbase_sb`` is a ``[1, 1]`` int32 SBUF tile holding the
    block's base token index and ``s_dim`` names the tensor dimension whose stride scales it, so
    the hardware forms ``base * per_token_stride`` without the kernel computing it.
    ``sbase_sb=None`` is the unrolled case, where the whole offset is a compile-time constant.
    """
    if sbase_sb is None:
        return tensor.ap(pattern=pattern, offset=elem_offset)
    return tensor.ap(pattern=pattern, offset=elem_offset, scalar_offset=sbase_sb, indirect_dim=s_dim)


def _flat_len(tensor: nl.NkiTensor, keep_minor: bool = False) -> int:
    """Product of ``tensor``'s dims, excluding the most-minor one when ``keep_minor``."""
    dims = tensor.shape[:-1] if keep_minor else tensor.shape
    n = 1
    for d in dims:
        n *= d
    return n


# Row stride alignment for the fused cache row: 8 bf16 == 16 bytes, the DMA tran_size.
_KV_ROW_ALIGN_BF16 = 8


def _quantize_c_kv_per_token(
    c_kv_sb: nl.NkiTensor,
    dst_sb: nl.NkiTensor,  # fp8-typed access pattern into the fused row's latent span
    scale_sb: nl.NkiTensor,
    s_tile_sz: int,
    kv_lora_rank: int,
    fp8_amax: float,
    sbm: SbufManager,
) -> None:
    """Quantize the latent to fp8 ON THE FLY, one scale per token, from this tile's own data.

    Tokens sit on the partition axis and the latent on the free axis, so the per-token amax is a
    free-axis reduce and the rescale is a per-partition scalar -- the cheap direction for both.

    Per TOKEN rather than per channel is what makes on-the-fly possible: a channel scale would have
    to be identical for every row ever written, across every CP rank and chunk, since a topk gather
    mixes rows from all of them -- so it could only be a static calibrated constant, and recomputing
    it would invalidate rows already in the cache. A token's scale travels in its own row.

    ONE scale for all ``kv_lora_rank`` channels, not one per 128-channel block: a single factor
    divides out of MM1's whole dot product, so the consumer dequantizes with one multiply per key.
    Per-block scales vary ALONG MM1's contraction axis and would force a partial accumulation each.

    Writes the DEQUANTIZATION factor (``amax / fp8_amax``, what the attention side multiplies by)
    to ``scale_sb``, and the quantized latent to ``dst_sb``.
    """
    P_MAX = nl.tile_size.pmax
    sbm.open_scope()
    amax_sb = sbm.alloc_stack((P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf, name="c_kv_amax")
    neg_min_sb = sbm.alloc_stack((P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf, name="c_kv_neg_min")
    # Per-token max|c_kv| as max(max(x), -min(x)): two free-axis reduces plus a [P, 1] combine,
    # with ``negate`` giving -min.
    nisa.tensor_reduce(amax_sb[0:s_tile_sz, :], op=nl.maximum, data=c_kv_sb[0:s_tile_sz, 0:kv_lora_rank], axis=1)
    nisa.tensor_reduce(
        neg_min_sb[0:s_tile_sz, :],
        op=nl.minimum,
        data=c_kv_sb[0:s_tile_sz, 0:kv_lora_rank],
        axis=1,
        negate=True,
    )
    nisa.tensor_tensor(amax_sb[0:s_tile_sz, :], amax_sb[0:s_tile_sz, :], neg_min_sb[0:s_tile_sz, :], op=nl.maximum)
    # An all-zero token (padding) would divide by zero; the floor keeps the reciprocal finite and
    # the row still quantizes to exact zeros.
    nisa.tensor_scalar(amax_sb[0:s_tile_sz, :], amax_sb[0:s_tile_sz, :], op0=nl.maximum, operand0=_AMAX_FLOOR)
    nisa.tensor_scalar(scale_sb[0:s_tile_sz, :], amax_sb[0:s_tile_sz, :], op0=nl.multiply, operand0=1.0 / fp8_amax)
    # Quantize by the reciprocal of the stored factor, so dequant is an exact round trip.
    recip_sb = sbm.alloc_stack((P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf, name="c_kv_qtz_recip")
    nisa.reciprocal(recip_sb[0:s_tile_sz, :], scale_sb[0:s_tile_sz, :])
    # dst is an ACCESS PATTERN, not a tile: the caller hands us the fp8-typed view of the fused
    # cache row's latent span, so the quantized values land straight in the staging row.
    nisa.tensor_scalar(
        dst_sb,
        c_kv_sb[0:s_tile_sz, 0:kv_lora_rank],
        op0=nl.multiply,
        operand0=recip_sb[0:s_tile_sz, :],
    )
    sbm.close_scope()


def _kv_row_bf16_width(kv_lora_rank: int, qk_rope_head_dim: int) -> int:
    """bf16 elements per fused cache row, padded to a 16-byte (8 bf16) multiple.

    ``kv_lora_rank // 2`` bf16 containers hold the fp8 latent, then one bf16 scale, then
    ``qk_rope_head_dim`` bf16 k_pe. DeepSeek/GLM dims give 256 + 1 + 64 = 321 -> 328 (656 B), the
    same row size vLLM's fp8 MLA cache uses.
    """
    return div_ceil(kv_lora_rank // 2 + 1 + qk_rope_head_dim, _KV_ROW_ALIGN_BF16) * _KV_ROW_ALIGN_BF16


def _store_paged_kv(
    c_kv_sb: nl.NkiTensor,
    k_pe_sb: nl.NkiTensor,
    kv_cache_hbm: nl.NkiTensor,
    slot_mapping_hbm: nl.NkiTensor,
    s_tile_global_offset: int,
    s_tile_sz: int,
    kv_lora_rank: int,
    qk_rope_head_dim: int,
    fp8_dtype,
    sbm: SbufManager,
    sbase_sb: nl.NkiTensor = None,
) -> None:
    """Scatter this tile's shared latents into the paged KV cache at the slots ``slot_mapping`` names.

    ONE fused row per token, in ONE scatter, so the attention-side topk gather is a single
    ``dma_copy`` per selected token:

        kv_cache bf16 [slots, _kv_row_bf16_width]
            [0 : L//2]          the L fp8 latent values, bitcast 2-per-bf16 in natural channel
                                order (a pure dtype view -- no reordering)
            [L//2]              the per-token dequantization scale
            [L//2+1 : +R]       k_pe, genuine bf16
            tail                padding to a 16-byte row stride

    bf16 is the container because it is 2 bytes: that keeps the row a single dtype (one descriptor
    per token) and keeps ``dma_transpose`` legal on the cache, which a 1-byte fp8 cache would not be
    (``src.dtype`` must be 2 bytes). Only the latent span is not meaningful AS bf16 -- the consumer
    bitcasts it back to fp8 -- so nothing reads those elements numerically.

    CHANNELS are paired, not tokens, so each row is wholly one token and both the scatter and the
    topk gather stay per-token at an arbitrary slot.

    The latent is always fp8, quantized on the fly per token (see
    :func:`_quantize_c_kv_per_token`). ``k_pe`` stays unquantized: it is only R of the L + R columns
    and the RoPE key is the accuracy-sensitive term.

    Tokens sit on the partition axis in every source tile, which is what makes this cheap: the slot
    index is one int32 per partition and the quantize rescale is a per-partition scalar.
    ``oob_mode.skip`` drops padded tokens, whose ``slot_mapping`` entry is out of range (-1 by the
    serving framework's convention).
    """
    P_MAX = nl.tile_size.pmax
    fp8_amax = get_max_positive_value_for_dtype(fp8_dtype)
    row = _kv_row_bf16_width(kv_lora_rank, qk_rope_head_dim)
    n_slots = _flat_len(kv_cache_hbm, keep_minor=True)
    lat_bf16 = kv_lora_rank // 2

    sbm.open_scope()
    slot_sb = sbm.alloc_stack((P_MAX, 1), dtype=nl.int32, buffer=nl.sbuf, name="kv_slot")
    nisa.dma_copy(
        dst=slot_sb[0:s_tile_sz, :],
        src=_s_ap(
            slot_mapping_hbm.reshape((_flat_len(slot_mapping_hbm),)),
            [[1, s_tile_sz], [1, 1]],
            s_tile_global_offset,
            0,
            sbase_sb,
        ),
        dge_mode=dge_mode.hwdge,
    )

    # One staging tile holding the whole row, so the scatter is a single contiguous descriptor.
    row_sb = sbm.alloc_stack((P_MAX, row), dtype=nl.bfloat16, buffer=nl.sbuf, name="kv_row")
    scale_sb = sbm.alloc_stack((P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf, name="c_kv_scale")
    lat_fp8_ap = row_sb.ap(pattern=[[row * 2, s_tile_sz], [1, kv_lora_rank]], offset=0, dtype=fp8_dtype)
    _quantize_c_kv_per_token(c_kv_sb, lat_fp8_ap, scale_sb, s_tile_sz, kv_lora_rank, fp8_amax, sbm)
    nisa.tensor_copy(dst=row_sb[0:s_tile_sz, lat_bf16 : lat_bf16 + 1], src=scale_sb[0:s_tile_sz, 0:1])
    nisa.tensor_copy(
        dst=row_sb[0:s_tile_sz, lat_bf16 + 1 : lat_bf16 + 1 + qk_rope_head_dim],
        src=k_pe_sb[0:s_tile_sz, 0:qk_rope_head_dim],
        engine=nisa.scalar_engine,
    )

    # Indirect DMA is SWDGE-only; one descriptor moves the token's entire row.
    nisa.dma_copy(
        dst=kv_cache_hbm.reshape((n_slots, row)).ap(
            pattern=[[row, s_tile_sz], [1, row]],
            offset=0,
            vector_offset=slot_sb[0:s_tile_sz, :],
            indirect_dim=0,
        ),
        src=row_sb[0:s_tile_sz, 0:row],
        oob_mode=nisa.oob_mode.skip,
        dge_mode=dge_mode.swdge,
    )
    sbm.close_scope()


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
    precision: MlaPrecision = MlaPrecision.MX,
    # Paged KV cache store (all four together, or all None)
    kv_cache_hbm: nl.NkiTensor = None,
    slot_mapping_hbm: nl.NkiTensor = None,
    kv_cache_fp8_dtype=nl.float8_e4m3,
) -> Tuple[nl.NkiTensor, ...]:
    """DeepSeek MLA QKV projection (MX or BF16), emitting absorbed latents.

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

    ``precision`` selects the arithmetic format; every arg below is described for the
    default MX path. Under ``MlaPrecision.BF16`` the numeric inputs change shape/dtype
    (see the ``precision`` entry) but the four latent OUTPUTS are identical, so downstream
    kernels are unaffected. BF16 is what GLM-MoE-DSA (GLM 5.2 / 5.3) needs.

    Args:
        x_hbm_mx (nl.NkiTensor): ``[B, S, H + scale_region]`` fp8 packed MX activation (see the param comment).
            BF16: ``[B, S, H]`` bfloat16, unquantized.
        wqkv_a_hbm (nl.NkiTensor): ``[H // 4, qk_lora_rank + kv_lora_rank + qk_rope_head_dim]`` fp8x4.
            BF16: ``[H, qk_lora_rank + kv_lora_rank + qk_rope_head_dim]`` bfloat16, NATURAL column order.
        wqkv_a_scale_hbm (nl.NkiTensor): compact block-128 scales for ``wqkv_a``. BF16: ignored, pass ``None``.
        wq_b_hbm (nl.NkiTensor): ``[qk_lora_rank // 4, n_heads * (qk_nope_head_dim + qk_rope_head_dim)]`` fp8x4.
            BF16: ``[qk_lora_rank, n_heads * qk_head_dim]`` bfloat16.
        wq_b_scale_hbm (nl.NkiTensor): compact block-128 scales for ``wq_b``. BF16: ignored, pass ``None``.
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
        compact_scales (bool): MX only -- see ``wqkv_a_scale_hbm``.
        kv_cache_hbm (nl.NkiTensor): optional MUTABLE paged KV cache, ONE fused bf16 tensor whose
            leading dims flatten to the slot count and whose most-minor dim is
            ``_kv_row_bf16_width(kv_lora_rank, qk_rope_head_dim)`` (328 at DeepSeek/GLM dims, i.e.
            656 B). Each row is one token: the fp8 latent bitcast 2-per-bf16, then the bf16
            per-token dequantization scale, then bf16 ``k_pe``, then padding. One row, so the
            consumer's gather is a single ``dma_copy`` per token -- see :func:`_store_paged_kv`.
            BF16 precision only, and requires ``slot_mapping_hbm``.
        kv_cache_fp8_dtype: fp8 format for the latent span, ``nl.float8_e4m3`` (legacy, max 240,
            the default) or the OCP ``nl.float8_e4m3fn`` (max 448, NeuronCore-v4). Only the max
            differs; per-token scaling consumes the range either way.
        slot_mapping_hbm (nl.NkiTensor): optional ``[B, S]`` (or flat ``[B * S]``) int32 cache
            slot per token, in the same flat token order the contiguous outputs use. Entries
            out of the cache's slot range (-1 for padding) are skipped. Quantization is ON THE
            FLY and takes no scale input: the store derives one scale per token from that token's
            own latent and writes it into the row's scale region (see
            :func:`_quantize_c_kv_per_token`).
        precision (MlaPrecision): ``MX`` (default) or ``BF16``. BF16 takes unquantized bf16
            activations and weights in natural column order and issues plain
            ``nisa.nc_matmul``: no 4-packing, no MLA output-column swizzle, no scale
            broadcast, no ``quantize_mx``, no un-swizzle. It also lifts the MX path's
            ``qk_nope_head_dim == 128`` restriction (the absorption contraction is tiled at
            128 rows, so GLM's 192 runs as 128 + 64) and relaxes the LoRA-rank multiple from
            512 to 128. Pass ``None`` for both scale tensors.

    Returns:
        q_lift (nl.NkiTensor): ``[B, S, n_heads, kv_lora_rank]`` bf16, per-head absorbed Q latent.
        q_pe (nl.NkiTensor): ``[B, S, n_heads, qk_rope_head_dim]`` bf16, per-head RoPE queries.
        c_kv (nl.NkiTensor): ``[B, S, kv_lora_rank]`` bf16, shared latent KV (RMSNorm(kv) * gamma).
        k_pe (nl.NkiTensor): ``[B, S, qk_rope_head_dim]`` bf16, shared RoPE key.

        Then, for the SAI indexer's wq_b projection, the q-normed qr latent -- MX emits it
        pre-transposed and pre-quantized so the indexer skips that work, BF16 emits it plain:

        MX (6 returns): ``qr_qtz`` ``[num_s_tiles, P_MAX, qk_lora_rank // 512, P_MAX]`` uint32
            (fp8x4) plus ``qr_scale``, the paired block-32 MX scales, same shape uint8.
        BF16 (5 returns): ``qr`` ``[B, S, qk_lora_rank]`` bf16.

    Notes:
        - MX absorption requires ``qk_nope_head_dim == 128`` (full-partition bf16
          contraction); BF16 accepts any ``qk_nope_head_dim``.
        - The absorption matmul (q_nope @ W_uk) runs in bf16 in BOTH modes: its contraction
          is only ``qk_nope_head_dim`` wide, so MX packing (4 elems/word -> 32 partitions)
          buys no throughput while forcing tiny strided transposes and a slabbed W_uk.
        - The paged store is BF16-only (``MlaPrecision.BF16``, the GLM config); the MX path
          rejects it in :func:`_validate_paged_kv_store`.
        - The paged store REPLACES the contiguous KV outputs: with the cache args given, ``c_kv`` /
          ``k_pe`` are written only into the cache, and the BF16 entry point returns
          ``(q_lift, q_pe, qr, kv_cache)``. The attention kernel gathers its KV out of the cache by
          topk, so a contiguous copy would be unread.

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
        precision=precision,
    )

    B, S, _ = x_hbm_mx.shape
    _validate_paged_kv_store(
        kv_cache_hbm=kv_cache_hbm,
        slot_mapping_hbm=slot_mapping_hbm,
        kv_cache_fp8_dtype=kv_cache_fp8_dtype,
        kv_row_bf16=_kv_row_bf16_width(kv_lora_rank, qk_rope_head_dim),
        B=B,
        S=S,
        precision=precision,
    )

    # Returning the caches makes them aliased outputs, so the harness can see what the scatter wrote.
    store_paged_kv = slot_mapping_hbm is not None

    q_lift_hbm = nl.ndarray((B, S, n_heads, kv_lora_rank), dtype=nl.bfloat16, buffer=nl.shared_hbm)
    q_pe_hbm = nl.ndarray((B, S, n_heads, qk_rope_head_dim), dtype=nl.bfloat16, buffer=nl.shared_hbm)
    c_kv_hbm = nl.ndarray((B, S, kv_lora_rank), dtype=nl.bfloat16, buffer=nl.shared_hbm)
    k_pe_hbm = nl.ndarray((B, S, qk_rope_head_dim), dtype=nl.bfloat16, buffer=nl.shared_hbm)

    sbm = SbufManager(
        sb_lower_bound=0,
        sb_upper_bound=nl.tile_size.total_available_sbuf_size,
        use_auto_alloc=False,
        logger=get_logger("mla_absorbed"),
    )

    if precision.is_bf16():
        # bf16 qr export for the indexer: plain [B, S, qk_lora] bf16, no transpose or quantize.
        qr_hbm = nl.ndarray((B, S, qk_lora_rank), dtype=nl.bfloat16, buffer=nl.shared_hbm)
        _qkv_stage_bf16(
            x_hbm_mx,
            wqkv_a_hbm,
            wq_b_hbm,
            q_norm_gamma_hbm,
            kv_norm_gamma_hbm,
            wuk_hbm,
            cos_cache_hbm,
            sin_cache_hbm,
            q_lift_hbm,
            q_pe_hbm,
            c_kv_hbm,
            k_pe_hbm,
            qr_hbm,
            n_heads,
            qk_nope_head_dim,
            qk_rope_head_dim,
            kv_lora_rank,
            qk_lora_rank,
            sbm,
            norm_eps,
            kv_cache_hbm=kv_cache_hbm,
            slot_mapping_hbm=slot_mapping_hbm,
            kv_cache_fp8_dtype=kv_cache_fp8_dtype,
        )
        if store_paged_kv:
            # No contiguous c_kv / k_pe in this mode; returning the unwritten buffers would hand
            # the caller garbage.
            return q_lift_hbm, q_pe_hbm, qr_hbm, kv_cache_hbm
        return q_lift_hbm, q_pe_hbm, c_kv_hbm, k_pe_hbm, qr_hbm

    # qr latent export for the indexer, matching its q_projection_mx input: one
    # [P_MAX, lora512, P_MAX] block per global s-tile, fp8x4 as uint32 with uint8 block-32 scales.
    P_MAX = nl.tile_size.pmax
    num_s_tiles = div_ceil(S, P_MAX)
    qr_lora_512_tiles = qk_lora_rank // (P_MAX * _H_PACK)
    qr_qtz_hbm = nl.ndarray((num_s_tiles, P_MAX, qr_lora_512_tiles, P_MAX), dtype=nl.uint32, buffer=nl.shared_hbm)
    qr_scale_hbm = nl.ndarray((num_s_tiles, P_MAX, qr_lora_512_tiles, P_MAX), dtype=nl.uint8, buffer=nl.shared_hbm)

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

    kernel_assert(
        S % num_shards == 0 and (S <= P_MAX or S_shard_base % P_MAX == 0),
        f"[QKV MLA] qr export requires each LNC shard to own whole {P_MAX}-query tiles: "
        f"S % num_shards == 0 and (S <= {P_MAX} or (S // num_shards) % {P_MAX} == 0). "
        f"Got S={S}, num_shards={num_shards} (S_shard_base={S_shard_base}).",
    )

    q_out_dim = n_heads * qk_head_dim
    kv_a_out_dim = kv_lora_rank + qk_rope_head_dim
    qkv_out_dim = qk_lora_rank + kv_a_out_dim

    # wqkv_a columns are [qr | kv | k_pe], with qr and kv pre-swizzled by the offline loader; the
    # compact block-128 scale ships for the natural (un-swizzled) weight.
    wqkv_a_scale_groups = [(0, qk_lora_rank), (qk_lora_rank, kv_lora_rank)] if compact_scales else None

    # Hardware constants
    H_PACK = _H_PACK

    # Output HBM tensors are allocated by the caller and passed in.
    sbm.open_scope()

    # ==================== Load norm weights ====================
    q_norm_gamma_sb = _load_norm_weights_for_mx(q_norm_gamma_hbm, qk_lora_rank, sbm, name="q_norm_gamma")

    # bf16 gamma in [P_MAX, kv_lora_rank] natural column order for c_kv = rsqrt(kv) * gamma; the
    # KV path takes no MX matmul, so it needs the plain bf16 layout.
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

    # K-slab dispatch for the stage-1 projection; the cost model is a conservative upper bound.
    _CALIBRATED_SBUF_BUDGET_BYTES = 240 * 1024
    _sbuf_budget = min(int(nl.tile_size.total_available_sbuf_size), _CALIBRATED_SBUF_BUDGET_BYTES)

    # Single-buffered slabs: the slab sizer divides the SBUF budget by this count.
    _NUM_WQKV_A_SLAB_BUFFERS = 1
    kv_b_out_dim = n_heads * (qk_nope_head_dim + kv_lora_rank)
    # This kernel holds no un-absorbed wkv_b resident and loads wq_b one head GROUP at a time, so
    # wq_b residency is estimated as the smallest legal group (2 heads).
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

    # Per-head-group wq_b slab: the full weight exceeds the stack, so it loads one head GROUP at a
    # time. Group is the largest even divisor of n_heads whose resident set fits; even keeps the
    # slab column offset 128-aligned.
    _WQ_B_K_TILES = qk_lora_rank // (P_MAX * H_PACK)
    _wuk_resident = n_heads * kv_lora_rank * 2  # bf16, full, resident in head loop
    _WORKING_RESERVE = 60 * 1024  # prologue norms + per-tile qr/kv/rope/q_lift scratch
    _budget_for_group = _sbuf_budget - _wuk_resident - _WORKING_RESERVE

    # Largest even divisor of n_heads whose wq_b slab + q_group output fit the budget and whose
    # Q-stage-2 output fits the PSUM banks. Inlined because NKI rejects inner function defs.
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

    # W_uk is not loaded here: it is dead during stage-1 and is loaded per tile inside the i_tile
    # scope after stage-1 finishes.
    WQ_B_K_TILES = qk_lora_rank // (P_MAX * H_PACK)
    wq_b_slab_n = WUK_HEAD_GROUP * qk_head_dim
    wq_b_slab_padded_n = ((wq_b_slab_n + SCALE_BLOCK - 1) // SCALE_BLOCK) * SCALE_BLOCK

    # W_uk and wq_b are s-tile-invariant: when both fit free SBUF they are loaded once before the
    # s-block loop, else they stream per tile.
    q_out_dim_padded = ((q_out_dim + SCALE_BLOCK - 1) // SCALE_BLOCK) * SCALE_BLOCK
    _wuk_full_bytes = n_heads * kv_lora_rank * 2  # bf16
    _wq_b_full_bytes = WQ_B_K_TILES * q_out_dim * 4 + WQ_B_K_TILES * q_out_dim_padded

    # Reserve everything that coexists with the resident W_uk + wq_b at peak: the stage-1 weight
    # slab, the stage-1 outputs, and the double-buffered per-tile input.
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

    # With a single s-tile per core there is no cross-tile reuse, so the head-group weight slices
    # stream double-buffered inside the head-group loop.
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

    # The stage-1 slab buffers are allocated in a child scope around the slab matmul and freed
    # after it, leaving only qr_sb / kv_raw_sb resident.
    qkv_out_dim_padded = ((qkv_out_dim + SCALE_BLOCK - 1) // SCALE_BLOCK) * SCALE_BLOCK

    # ==================== Multi-buffering setup ====================
    NUM_INPUT_BUFFERS = 2
    S_BLOCK_SIZE = NUM_INPUT_BUFFERS * S_TILE_SIZE
    num_S_blocks = div_ceil(S_shard, S_BLOCK_SIZE)

    # Packed MX scale geometry: 4 H512 tiles fold into one 128-wide scale block, so
    # scale_region = ceil(H_512/4)*128 and stage-1 reads each K-tile's scale by slot.
    N_PACKED = div_ceil(H_512_TILE_COUNT, 4)
    # Packed MX input row = [H fp8 hidden | N_PACKED*128 uint8 scale blocks]; the fp8 transpose
    # lands the hidden in x_qtz_sb and the scale blocks in x_scale_sb.
    _SCALE_REGION = N_PACKED * P_MAX
    _MX_ROW = H + _SCALE_REGION
    x_mx_2d = x_hbm_mx.reshape((S, _MX_ROW))

    for s_block_idx in nl.affine_range(num_S_blocks):
        sbm.open_scope()
        sbm.set_name_prefix(f"sb{s_block_idx}_")

        s_block_offset = s_block_idx * S_BLOCK_SIZE
        s_block_sz = min(S_BLOCK_SIZE, S_shard - s_block_offset)
        num_tiles_in_block = div_ceil(s_block_sz, S_TILE_SIZE)

        # Step 1: load input for every tile in the block. The scale buffer is [P_MAX, H_512, s_tile];
        # the packed path uses only the first N_PACKED slots.
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

        # Step 2: process each tile in the block.
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

            # Combined Q/KV stage 1: x @ wqkv_a -> qr, kv + k_pe.
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

            # Kick off group 0's wq_b + W_uk prefetch here: it has no dependency on qr/kv, so the DMA
            # overlaps the qr and kv paths below.
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

            # Q path RMSNorm; gamma is fused into the transpose below.
            _apply_rms_norm_inplace(qr_sb, zero_bias_sb, norm_eps_sb, s_tile_pad, qk_lora_rank, sbm, name="q_rms")

            # Q path stage 2: prepare quantized qr_normed. The wq_b matmul is N-slabbed per head group
            # inside the head loop below.
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

            # Export the quantized qr latent for the indexer, one s-tile block per global s-tile
            # index. qr_qtz_hbm is uint32, so the fp8x4 source is viewed as uint32. Columns are
            # written at their within-tile offset so cores sharing a global tile stay disjoint.
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

            # Split kv_raw into kv (latent) and k_pe.
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

            # KV path RMSNorm on kv; gamma is applied below into c_kv.
            _apply_rms_norm_inplace(kv_sb, zero_bias_sb, norm_eps_sb, s_tile_sz, kv_lora_rank, sbm, name="kv_rms")

            # kv_sb latent columns are swizzled; c_kv needs natural order for the attention kernel and the
            # natural-order gamma, so un-swizzle first and then apply gamma.
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
            # Per head: q_pe RoPE and the absorption q_nope @ W_uk -> q_lift, each head streaming straight
            # to HBM. W_uk is loaded in the i_tile scope so it survives all head groups, unless it
            # was already made resident before the s-block loop.
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

                # Per-group scope: q_group_sb and the wq_b compact scales are only needed for this group.
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
                    # Full wq_b already resident: pass a column-offset view for this group, since _mx_matmul reads
                    # weights at free offset 0.
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

                # Batch this group's q_pe / q_lift into one DMA each: every head writes its slot, then a single
                # strided DMA stores the whole contiguous head range.
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
                    # Absorption: an nc_transpose lifts q_nope[s, nope] -> [nope, s], then a plain nc_matmul with
                    # stationary=q_nope_t and moving=W_uk[h] yields q_lift[s, kv_lora].
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

                # Batched RoPE over the whole group's q_pe: 6 wide tensor_tensor ops with one
                # cos/sin broadcast shared across heads.
                rope_grp_scratch = sbm.alloc_stack(
                    (P_MAX, group_count * qk_rope_head_dim), dtype=nl.bfloat16, buffer=nl.sbuf, name="rope_grp_scratch"
                )
                _apply_rope_inplace_interleaved_grouped(
                    q_pe_grp_sb, cos_sb, sin_sb, rope_grp_scratch, s_tile_sz, qk_rope_head_dim, group_count
                )

                # One DMA each for this group's q_pe / q_lift: heads [group_start, group_start+group_count)
                # are contiguous in HBM, so the group buffer maps to one strided DMA.
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

            # Store the shared c_kv and k_pe to HBM; q_lift / q_pe were streamed per head above.
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


def _qkv_stage_bf16(
    x_hbm: nl.NkiTensor,
    wqkv_a_hbm: nl.NkiTensor,
    wq_b_hbm: nl.NkiTensor,
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
    qr_hbm: nl.NkiTensor,
    # Dimension parameters
    n_heads: int,
    qk_nope_head_dim: int,
    qk_rope_head_dim: int,
    kv_lora_rank: int,
    qk_lora_rank: int,
    sbm: SbufManager,
    norm_eps: float = 1e-6,
    # Paged KV cache store.
    kv_cache_hbm: nl.NkiTensor = None,
    slot_mapping_hbm: nl.NkiTensor = None,
    kv_cache_fp8_dtype=nl.float8_e4m3,
) -> None:
    """Absorbed-latent MLA QKV stage, BF16 (``MlaPrecision.BF16``).

    Same six-step pipeline as :func:`_qkv_stage` -- fused stage-1 projection, q-norm, Q
    stage-2 per head group, interleaved RoPE on q_pe, per-head absorption into q_lift, and
    the latent KV path -- with every MX-format step removed rather than reimplemented:

      * activations arrive as plain bf16 ``[B, S, H]``, so there is no packed-scale region
        to unfold and no ``quantize_mx`` before stage 2;
      * weights are bf16 ``[K, N]`` in NATURAL column order, so the MLA output-column
        swizzle and its ``_unswizzle_lora_cols`` inverse both disappear -- ``kv`` and ``qr``
        come out of stage 1 already in natural order;
      * norm gammas are applied as one ``tensor_tensor`` multiply instead of being fused
        into a swizzling transpose;
      * ``qk_nope_head_dim`` is no longer pinned to 128: the absorption contraction is K
        tiled at 128 rows, so GLM-MoE-DSA's 192 runs as an accumulating 128 + 64.

    Weight residency is deliberately simpler than the MX stage, which carries three modes
    (fully resident / double-buffered per group / per-tile streaming) tuned against measured
    DeepSeek profiles. Here wq_b and W_uk are always streamed one head group at a time. bf16
    weights cost 2 bytes/element against MX's ~1.25, so the resident mode would rarely fit
    anyway; the streaming path is the one that always applies. Revisit with a profile before
    adding modes back.

    See :func:`mla_qkv_cte_kernel` for arg semantics.
    """
    P_MAX = nl.tile_size.pmax

    # bf16 weights are [K, N] with K on rows, so H and the LoRA ranks come straight from the
    # weight shapes (no 4-pack division as in the MX path).
    H = wqkv_a_hbm.shape[0]
    B, S, _ = x_hbm.shape
    # Every HBM offset below is flat in the sequence axis (matching the MX stage), so the batch
    # axis must be degenerate. Prefill folds batch into S upstream.
    kernel_assert(B == 1, f"[QKV MLA bf16] batch must be 1 (fold batch into S upstream), got {B}.")
    qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
    q_out_dim = n_heads * qk_head_dim
    kv_a_out_dim = kv_lora_rank + qk_rope_head_dim
    qkv_out_dim = qk_lora_rank + kv_a_out_dim

    # ==================== LNC Sharding Setup ====================
    _, num_shards, shard_id = get_program_sharding_info()
    kernel_assert(
        S % num_shards == 0,
        f"[QKV MLA bf16] S must divide evenly across LNC shards so the per-tile token count is "
        f"static. Got S={S}, num_shards={num_shards}.",
    )
    S_shard = S // num_shards
    S_shard_offset = shard_id * S_shard

    sbm.open_scope()

    # Prologue: both gammas are plain [P_MAX, dim] bf16 broadcasts in natural column order,
    # applied with a single tensor_tensor after the rsqrt.
    q_norm_gamma_sb = _load_bf16_gamma(q_norm_gamma_hbm, qk_lora_rank, sbm, name="q_norm_gamma")
    kv_norm_gamma_sb = _load_bf16_gamma(kv_norm_gamma_hbm, kv_lora_rank, sbm, name="kv_norm_gamma")
    store_paged_kv = slot_mapping_hbm is not None
    norm_eps_sb = sbm.alloc_stack((P_MAX, 1), dtype=nl.bfloat16, buffer=nl.sbuf, name="norm_eps")
    nisa.memset(dst=norm_eps_sb, value=norm_eps)
    zero_bias_sb = sbm.alloc_stack((P_MAX, 1), dtype=nl.bfloat16, buffer=nl.sbuf, name="zero_bias")
    nisa.memset(dst=zero_bias_sb, value=0.0)

    # ==================== Tiling setup ====================
    S_TILE_SIZE = s_tile_pad = P_MAX
    NUM_INPUT_BUFFERS = 2
    S_BLOCK_SIZE = NUM_INPUT_BUFFERS * S_TILE_SIZE
    num_S_blocks = div_ceil(S_shard, S_BLOCK_SIZE)

    H_K_TILES = div_ceil(H, _BF16_K_TILE)
    QK_LORA_K_TILES = div_ceil(qk_lora_rank, _BF16_K_TILE)
    h_k_tile_sizes = _bf16_k_tile_sizes(H)

    _CALIBRATED_SBUF_BUDGET_BYTES = 240 * 1024
    _sbuf_budget = min(int(nl.tile_size.total_available_sbuf_size), _CALIBRATED_SBUF_BUDGET_BYTES)

    # Both weight streams are double-buffered.
    _NUM_WQKV_A_SLAB_BUFFERS = 2
    _NUM_GROUP_WEIGHT_BUFFERS = 2

    # wq_b and W_uk are read as a STRIDED slice.
    _WQ_B_DMA_HEADS_TARGET = 4

    _x_t_bytes = NUM_INPUT_BUFFERS * H_K_TILES * s_tile_pad * 2
    _x_stage_bytes = H * 2  # inner-scope staging buffer, coexists with x_t while transposing
    _stage1_out_bytes = qkv_out_dim * 2  # qr + kv_raw
    _qr_t_bytes = NUM_INPUT_BUFFERS * QK_LORA_K_TILES * s_tile_pad * 2
    _dma_heads_reserve = min(_WQ_B_DMA_HEADS_TARGET, n_heads)
    _min_group_wq_b_bytes, _ = _bf16_sbuf_footprint(qk_lora_rank, _dma_heads_reserve * qk_head_dim)
    _min_group_wuk_bytes, _ = _bf16_sbuf_footprint(qk_nope_head_dim, _dma_heads_reserve * kv_lora_rank)
    _WORKING_RESERVE = 48 * 1024  # per-tile rope/kv/c_kv/group scratch + allocator slack
    _slab_reserve = (
        _x_t_bytes
        + _x_stage_bytes
        + _stage1_out_bytes
        + _qr_t_bytes
        + _NUM_GROUP_WEIGHT_BUFFERS * (_min_group_wq_b_bytes + _min_group_wuk_bytes)
        + _WORKING_RESERVE
    )

    _min_group_bytes_2h = (
        _bf16_sbuf_footprint(qk_lora_rank, min(2, n_heads) * qk_head_dim)[0]
        + _bf16_sbuf_footprint(qk_nope_head_dim, min(2, n_heads) * kv_lora_rank)[0]
    )
    _slab_reserve_min = (
        _x_t_bytes
        + _x_stage_bytes
        + _stage1_out_bytes
        + _qr_t_bytes
        + _NUM_GROUP_WEIGHT_BUFFERS * _min_group_bytes_2h
        + _WORKING_RESERVE
    )
    _resident_tiles = _bf16_compute_k_slab_tiles(H, qkv_out_dim, _sbuf_budget, _slab_reserve_min, num_slab_buffers=1)
    if _resident_tiles == H_K_TILES:
        K_SLAB_TILES = H_K_TILES
    else:
        K_SLAB_TILES = _bf16_compute_k_slab_tiles(
            H, qkv_out_dim, _sbuf_budget, _slab_reserve, num_slab_buffers=_NUM_WQKV_A_SLAB_BUFFERS
        )
    NUM_K_SLABS = H_K_TILES // K_SLAB_TILES
    USE_K_SLAB = NUM_K_SLABS > 1

    if not USE_K_SLAB:
        wqkv_a_sb = _load_bf16_weights(wqkv_a_hbm, H, qkv_out_dim, sbm, name="wqkv_a")

    # Head-group size for the wq_b and W_uk slabs: largest even divisor of n_heads whose slabs,
    # q_group output and per-group buffers fit, and whose Q stage-2 output fits PSUM.
    _F_MAX_LOCAL = 512
    _MAX_Q_PSUM_BANKS = _NUM_HW_PSUM_BANKS - 1
    # Stage-1 weights held while the head loop runs: the whole weight when unslabbed, one slab
    # otherwise.
    _slab_resident_bytes = (H_K_TILES if not USE_K_SLAB else _NUM_WQKV_A_SLAB_BUFFERS * K_SLAB_TILES) * qkv_out_dim * 2
    _budget_for_group = (
        _sbuf_budget - _slab_resident_bytes - _x_t_bytes - _stage1_out_bytes - _qr_t_bytes - _WORKING_RESERVE
    )

    _lift_banks = div_ceil(kv_lora_rank, _F_MAX_LOCAL)

    # Widest fitting group WITH the phase-B ping-pong (_group_pp) and without it
    # (_group_wide). Inlined as one loop because NKI rejects inner function defs.
    _group_pp = 2
    _group_wide = 2
    for group_size in range(n_heads, 1, -1):
        if n_heads % group_size != 0 or group_size % 2 != 0:
            continue
        group_n = group_size * qk_head_dim
        q_banks = div_ceil(group_n, _F_MAX_LOCAL)
        if q_banks > _MAX_Q_PSUM_BANKS:
            continue
        wq_b_bytes, _ = _bf16_sbuf_footprint(qk_lora_rank, group_n)
        wuk_bytes, _ = _bf16_sbuf_footprint(qk_nope_head_dim, group_size * kv_lora_rank)
        # q_group output + q_pe slots + q_lift slots + the grouped-RoPE scratch.
        out_bytes = (group_n + 2 * group_size * qk_rope_head_dim + group_size * kv_lora_rank) * 2
        if _NUM_GROUP_WEIGHT_BUFFERS * (wq_b_bytes + wuk_bytes) + out_bytes > _budget_for_group:
            continue
        # Descending, so the first hit in each class is the widest.
        if _group_wide == 2:
            _group_wide = group_size
        if _group_pp == 2 and _NUM_TRANSPOSE_BANKS + 2 * (_lift_banks + q_banks) <= _NUM_HW_PSUM_BANKS:
            _group_pp = group_size
        if _group_wide != 2 and _group_pp != 2:
            break

    WUK_HEAD_GROUP = _group_pp if _group_pp * 2 >= _group_wide else _group_wide

    num_head_groups = div_ceil(n_heads, WUK_HEAD_GROUP)
    NUM_LIFT_N_TILES = div_ceil(kv_lora_rank, _F_MAX_LOCAL)
    NUM_NOPE_TILES = div_ceil(qk_nope_head_dim, _BF16_K_TILE)

    # Widest DMA group that still fits: a multiple of the compute group, counting two buffers
    # (the prefetched next DMA group is live alongside the current one) plus one compute
    # group's outputs.
    _compute_out_bytes = (
        WUK_HEAD_GROUP * qk_head_dim
        + 2 * WUK_HEAD_GROUP * qk_rope_head_dim
        + WUK_HEAD_GROUP * kv_lora_rank
        + WUK_HEAD_GROUP * NUM_NOPE_TILES * s_tile_pad
    ) * 2
    WQ_B_DMA_HEADS = WUK_HEAD_GROUP
    for _cand in range(
        (min(_WQ_B_DMA_HEADS_TARGET, n_heads) // WUK_HEAD_GROUP) * WUK_HEAD_GROUP,
        WUK_HEAD_GROUP - 1,
        -WUK_HEAD_GROUP,
    ):
        if n_heads % _cand != 0:
            continue
        _wq_b_bytes, _ = _bf16_sbuf_footprint(qk_lora_rank, _cand * qk_head_dim)
        _wuk_bytes, _ = _bf16_sbuf_footprint(qk_nope_head_dim, _cand * kv_lora_rank)
        if _NUM_GROUP_WEIGHT_BUFFERS * (_wq_b_bytes + _wuk_bytes) + _compute_out_bytes <= _budget_for_group:
            WQ_B_DMA_HEADS = _cand
            break
    num_dma_groups = div_ceil(n_heads, WQ_B_DMA_HEADS)
    groups_per_dma = WQ_B_DMA_HEADS // WUK_HEAD_GROUP
    kernel_assert(
        _NUM_TRANSPOSE_BANKS + NUM_LIFT_N_TILES <= _NUM_HW_PSUM_BANKS,
        f"[QKV MLA bf16] absorption needs {_NUM_TRANSPOSE_BANKS} transpose PSUM banks plus "
        f"{NUM_LIFT_N_TILES} q_lift banks, exceeding {_NUM_HW_PSUM_BANKS}. Reduce kv_lora_rank.",
    )

    # PSUM layout for phase B. Every accumulator is read back out to SBUF right after its
    # matmuls, so reusing ONE bank run makes each matmul wait for the previous read-out to
    # drain -- a matmul/copy ping-pong on the critical path, repeated per head and per tile.
    # When the banks are there, give the Q stage-2 output and the q_lift accumulator two
    # disjoint runs each and alternate: banks [0, _NUM_TRANSPOSE_BANKS) stay the transpose
    # scratch, then 2 x q_lift, then 2 x Q stage-2.
    #
    # When they are not (a wide head group, or a large kv_lora_rank), fall back to exactly the
    # previous placement -- Q stage 2 from bank 0, one q_lift run above the transposes -- so
    # no config that fits today stops fitting.
    _Q_PSUM_BANKS = div_ceil(WUK_HEAD_GROUP * qk_head_dim, _F_MAX_LOCAL)
    _PSUM_PINGPONG = _NUM_TRANSPOSE_BANKS + 2 * (NUM_LIFT_N_TILES + _Q_PSUM_BANKS) <= _NUM_HW_PSUM_BANKS
    if _PSUM_PINGPONG:
        _LIFT_BANK_BASE = _NUM_TRANSPOSE_BANKS
        _LIFT_BANK_STRIDE = NUM_LIFT_N_TILES
        _Q_BANK_BASE = _NUM_TRANSPOSE_BANKS + 2 * NUM_LIFT_N_TILES
        _Q_BANK_STRIDE = _Q_PSUM_BANKS
    else:
        _LIFT_BANK_BASE = _NUM_TRANSPOSE_BANKS
        _LIFT_BANK_STRIDE = 0
        _Q_BANK_BASE = 0
        _Q_BANK_STRIDE = 0

    # Per-head-group weight slabs, reused across groups and s-tiles (streamed, see docstring).
    # Double-buffered: group g computes out of buffer g % 2 while group g + 1 loads into the other.
    wq_b_bufs = []
    wuk_bufs = []
    for buf_idx in range(_NUM_GROUP_WEIGHT_BUFFERS):
        wq_b_bufs.append(
            sbm.alloc_stack(
                (P_MAX, QK_LORA_K_TILES, WQ_B_DMA_HEADS * qk_head_dim),
                dtype=nl.bfloat16,
                buffer=nl.sbuf,
                name=f"wq_b_grp_{buf_idx}",
            )
        )
        wuk_bufs.append(
            sbm.alloc_stack(
                (P_MAX, div_ceil(qk_nope_head_dim, _BF16_K_TILE), WQ_B_DMA_HEADS * kv_lora_rank),
                dtype=nl.bfloat16,
                buffer=nl.sbuf,
                name=f"wuk_grp_{buf_idx}",
            )
        )

    x_2d = x_hbm.reshape((B * S, H))

    # The s-block loop is the ONLY loop whose trip count scales with S; everything else is fixed
    # by the head/K dims.
    def _s_block_body(s_block_offset, sbase_sb, tag):
        sbm.open_scope()
        sbm.set_name_prefix(f"{tag}_")

        s_block_sz = min(S_BLOCK_SIZE, S_shard - s_block_offset)
        num_tiles_in_block = div_ceil(s_block_sz, S_TILE_SIZE)

        # Block-scoped, because phase B below consumes them AFTER the phase-A tile loop has ended:
        # one K-major qr and one RoPE cache pair per tile.
        qr_t_bufs = []
        cos_bufs = []
        sin_bufs = []
        for buf_idx in range(num_tiles_in_block):
            qr_t_bufs.append(
                sbm.alloc_stack(
                    (P_MAX, QK_LORA_K_TILES, s_tile_pad), dtype=nl.bfloat16, buffer=nl.sbuf, name=f"qr_t_buf_{buf_idx}"
                )
            )
            cos_bufs.append(
                sbm.alloc_stack((P_MAX, qk_rope_head_dim), dtype=nl.bfloat16, buffer=nl.sbuf, name=f"cos_buf_{buf_idx}")
            )
            sin_bufs.append(
                sbm.alloc_stack(
                    (P_MAX, qk_rope_head_dim // 2), dtype=nl.bfloat16, buffer=nl.sbuf, name=f"sin_buf_{buf_idx}"
                )
            )

        # Step 1: load and transpose every tile's activation. nc_matmul contracts on the partition
        # axis, so [s, H] must become [H, s] before it can be the stationary operand.
        x_t_bufs = []
        for buf_idx in range(num_tiles_in_block):
            x_t_bufs.append(
                sbm.alloc_stack(
                    (P_MAX, H_K_TILES, s_tile_pad), dtype=nl.bfloat16, buffer=nl.sbuf, name=f"x_t_buf_{buf_idx}"
                )
            )

        # DMA group 0's weights are prefetched HERE, once per block and ahead of phase A, so
        # their DMA runs under the stage-1 matmuls; every later DMA group is prefetched one
        # group ahead inside phase B.
        _load_bf16_weights_k_slab(
            wq_b_hbm,
            wq_b_bufs[0],
            out_dim=min(WQ_B_DMA_HEADS, n_heads) * qk_head_dim,
            k_tile_start=0,
            k_tile_count=QK_LORA_K_TILES,
            k_tile_sizes=_bf16_k_tile_sizes(qk_lora_rank),
            full_out_dim=q_out_dim,
            out_col_offset=0,
        )
        _load_wuk_bf16_group_tiled(
            wuk_hbm, wuk_bufs[0], n_heads, qk_nope_head_dim, kv_lora_rank, 0, min(WQ_B_DMA_HEADS, n_heads)
        )

        for i_tile in range(num_tiles_in_block):
            s_tile_local_offset = s_block_offset + i_tile * S_TILE_SIZE
            s_tile_sz = min(S_TILE_SIZE, S_shard - s_tile_local_offset)
            _load_x_bf16_tile(
                x_2d,
                x_t_bufs[i_tile],
                S_shard_offset + s_tile_local_offset,
                s_tile_sz,
                s_tile_pad,
                H,
                sbm,
                sbase_sb=sbase_sb,
            )

        # Step 2, phase A: per tile, stage 1 plus both norms and the whole KV path. The head-group
        # loop used to live here; it now runs once per BLOCK in phase B below, so all this phase
        # has to leave behind is qr_t. That is what makes the wq_b / W_uk stream cost one pass per
        # block instead of one per tile.
        for i_tile in nl.affine_range(num_tiles_in_block):
            sbm.open_scope()
            sbm.set_name_prefix(f"{tag}_t{i_tile}_")

            s_tile_local_offset = s_block_offset + i_tile * S_TILE_SIZE
            s_tile_sz = min(S_TILE_SIZE, S_shard - s_tile_local_offset)
            s_tile_global_offset = S_shard_offset + s_tile_local_offset
            x_t_sb = x_t_bufs[i_tile]

            # Prefetch RoPE caches early (their DMA overlaps the stage-1 matmul). Block-scoped:
            # phase B needs them again for the q_pe RoPE.
            cos_sb = cos_bufs[i_tile]
            sin_sb = sin_bufs[i_tile]
            rope_offset = s_tile_global_offset * qk_rope_head_dim
            nisa.dma_copy(
                dst=cos_sb[0:s_tile_sz, 0:qk_rope_head_dim],
                src=_s_ap(
                    cos_cache_hbm,
                    [[qk_rope_head_dim, s_tile_sz], [1, qk_rope_head_dim]],
                    rope_offset,
                    1,
                    sbase_sb,
                ),
                dge_mode=dge_mode.hwdge,
            )
            nisa.dma_copy(
                dst=sin_sb[0:s_tile_sz, 0 : qk_rope_head_dim // 2],
                src=_s_ap(
                    sin_cache_hbm,
                    [[qk_rope_head_dim, s_tile_sz], [1, qk_rope_head_dim // 2]],
                    rope_offset,
                    1,
                    sbase_sb,
                ),
                dge_mode=dge_mode.hwdge,
            )

            # Stage 1: x @ wqkv_a -> [qr | kv | k_pe] in natural column order.
            if not USE_K_SLAB:
                qr_sb, kv_raw_sb = _bf16_matmul_split(
                    x_t_sb, wqkv_a_sb, H, s_tile_pad, qkv_out_dim, [qk_lora_rank], sbm, name="stage1"
                )
            else:
                # The slab weight buffer is dead once stage 1 finishes, so confine it to a child
                # scope; PSUM is a separate space, so the accumulators survive the close.
                sbm.open_scope()
                wqkv_a_slab_bufs = []
                for buf_idx in range(_NUM_WQKV_A_SLAB_BUFFERS):
                    wqkv_a_slab_bufs.append(
                        sbm.alloc_stack(
                            (P_MAX, K_SLAB_TILES, qkv_out_dim),
                            dtype=nl.bfloat16,
                            buffer=nl.sbuf,
                            name=f"wqkv_a_slab_{buf_idx}",
                        )
                    )
                output_psum = _alloc_matmul_psum(div_ceil(qkv_out_dim, _F_MAX_LOCAL))

                # Software-pipelined slab loop: slab 0 is loaded up front, then each iteration
                # issues the NEXT slab's load before its own matmul so the DMA runs under the PE.
                _load_bf16_weights_k_slab(
                    wqkv_a_hbm,
                    wqkv_a_slab_bufs[0],
                    out_dim=qkv_out_dim,
                    k_tile_start=0,
                    k_tile_count=K_SLAB_TILES,
                    k_tile_sizes=h_k_tile_sizes,
                    full_out_dim=qkv_out_dim,
                )
                for slab_id in range(NUM_K_SLABS):
                    cur_buf = slab_id % _NUM_WQKV_A_SLAB_BUFFERS
                    if slab_id + 1 < NUM_K_SLABS:
                        _load_bf16_weights_k_slab(
                            wqkv_a_hbm,
                            wqkv_a_slab_bufs[(slab_id + 1) % _NUM_WQKV_A_SLAB_BUFFERS],
                            out_dim=qkv_out_dim,
                            k_tile_start=(slab_id + 1) * K_SLAB_TILES,
                            k_tile_count=K_SLAB_TILES,
                            k_tile_sizes=h_k_tile_sizes,
                            full_out_dim=qkv_out_dim,
                        )
                    _bf16_matmul_k_range(
                        x_t_sb,
                        wqkv_a_slab_bufs[cur_buf],
                        k_tile_start=slab_id * K_SLAB_TILES,
                        k_tile_count=K_SLAB_TILES,
                        k_tile_sizes=h_k_tile_sizes,
                        m_dim=s_tile_pad,
                        n_dim=qkv_out_dim,
                        output_psum=output_psum,
                    )
                sbm.close_scope()

                qr_sb, kv_raw_sb = _copy_psum_splits(
                    output_psum, s_tile_pad, [0, qk_lora_rank, qkv_out_dim], sbm, "stage1"
                )

            # Q path: RMSNorm(qr) * q_gamma, then the M-major -> K-major flip nc_matmul needs.
            _apply_rms_norm_inplace(qr_sb, zero_bias_sb, norm_eps_sb, s_tile_pad, qk_lora_rank, sbm, name="q_rms")
            nisa.tensor_tensor(
                dst=qr_sb[0:s_tile_sz, 0:qk_lora_rank],
                data1=qr_sb[0:s_tile_sz, 0:qk_lora_rank],
                data2=q_norm_gamma_sb[0:s_tile_sz, 0:qk_lora_rank],
                op=nl.multiply,
            )

            # qr export for the SAI indexer (plain bf16; the indexer does its own transpose).
            nisa.dma_copy(
                dst=_s_ap(
                    qr_hbm,
                    [[qk_lora_rank, s_tile_sz], [1, qk_lora_rank]],
                    s_tile_global_offset * qk_lora_rank,
                    1,
                    sbase_sb,
                ),
                src=qr_sb[0:s_tile_sz, 0:qk_lora_rank],
                dge_mode=dge_mode.hwdge,
            )

            _transpose_bf16_to_k_major(
                qr_sb,
                qr_t_bufs[i_tile],
                m_sz=s_tile_sz,
                m_pad=s_tile_pad,
                k_tile_sizes=_bf16_k_tile_sizes(qk_lora_rank),
                src_row_stride=qk_lora_rank,
            )

            # KV path: c_kv = RMSNorm(kv) * kv_gamma and k_pe RoPE. kv_raw columns are already natural
            # order, so c_kv is written in place.
            c_kv_sb = sbm.alloc_stack((P_MAX, kv_lora_rank), dtype=nl.bfloat16, buffer=nl.sbuf, name="c_kv")
            nisa.tensor_copy(
                dst=c_kv_sb[0:s_tile_sz, 0:kv_lora_rank],
                src=kv_raw_sb[0:s_tile_sz, 0:kv_lora_rank],
                engine=nisa.scalar_engine,
            )
            _apply_rms_norm_inplace(c_kv_sb, zero_bias_sb, norm_eps_sb, s_tile_sz, kv_lora_rank, sbm, name="kv_rms")
            nisa.tensor_tensor(
                dst=c_kv_sb[0:s_tile_sz, 0:kv_lora_rank],
                data1=c_kv_sb[0:s_tile_sz, 0:kv_lora_rank],
                data2=kv_norm_gamma_sb[0:s_tile_sz, 0:kv_lora_rank],
                op=nl.multiply,
            )

            k_pe_sb = sbm.alloc_stack((P_MAX, qk_rope_head_dim), dtype=nl.bfloat16, buffer=nl.sbuf, name="k_pe_split")
            nisa.tensor_copy(
                dst=k_pe_sb[0:s_tile_sz, 0:qk_rope_head_dim],
                src=kv_raw_sb[0:s_tile_sz, kv_lora_rank : kv_lora_rank + qk_rope_head_dim],
                engine=nisa.scalar_engine,
            )
            k_pe_rope_sb = _apply_rope_to_tensor_interleaved(
                k_pe_sb, cos_sb, sin_sb, s_tile_sz, qk_rope_head_dim, sbm, name="k_pe_rope"
            )

            # Contiguous bf16 latents for consumers that read KV as a dense [B, S, *] tensor.
            # Skipped under the paged store, where the fp8 cache below is the only KV written.
            if not store_paged_kv:
                nisa.dma_copy(
                    dst=_s_ap(
                        c_kv_hbm,
                        [[kv_lora_rank, s_tile_sz], [1, kv_lora_rank]],
                        s_tile_global_offset * kv_lora_rank,
                        1,
                        sbase_sb,
                    ),
                    src=c_kv_sb[0:s_tile_sz, 0:kv_lora_rank],
                    dge_mode=dge_mode.hwdge,
                )
                nisa.dma_copy(
                    dst=_s_ap(
                        k_pe_hbm,
                        [[qk_rope_head_dim, s_tile_sz], [1, qk_rope_head_dim]],
                        s_tile_global_offset * qk_rope_head_dim,
                        1,
                        sbase_sb,
                    ),
                    src=k_pe_rope_sb[0:s_tile_sz, 0:qk_rope_head_dim],
                    dge_mode=dge_mode.hwdge,
                )
            else:
                _store_paged_kv(
                    c_kv_sb,
                    k_pe_rope_sb,
                    kv_cache_hbm,
                    slot_mapping_hbm,
                    s_tile_global_offset,
                    s_tile_sz,
                    kv_lora_rank,
                    qk_rope_head_dim,
                    kv_cache_fp8_dtype,
                    sbm,
                    sbase_sb=sbase_sb,
                )

            sbm.close_scope()

        # Step 2, phase B: per head group, across every tile in the block -- Q stage 2, q_pe RoPE,
        # absorption -> q_lift. Group-outer / tile-inner is the point of the phase split: a group's
        # wq_b + W_uk slabs are loaded ONCE and amortized over num_tiles_in_block tiles, where the
        # old tile-outer order re-read the whole of wq_b and W_uk for every tile.
        for dma_grp_idx in range(num_dma_groups):
            sbm.open_scope()
            cur_buf = dma_grp_idx % _NUM_GROUP_WEIGHT_BUFFERS

            # Prefetch the NEXT DMA group into the alternate buffer before this one's compute,
            # so its DMA overlaps every sub-group's and every tile's compute.
            if dma_grp_idx + 1 < num_dma_groups:
                nxt_buf = (dma_grp_idx + 1) % _NUM_GROUP_WEIGHT_BUFFERS
                nxt_start = (dma_grp_idx + 1) * WQ_B_DMA_HEADS
                nxt_count = min(WQ_B_DMA_HEADS, n_heads - nxt_start)
                _load_bf16_weights_k_slab(
                    wq_b_hbm,
                    wq_b_bufs[nxt_buf],
                    out_dim=nxt_count * qk_head_dim,
                    k_tile_start=0,
                    k_tile_count=QK_LORA_K_TILES,
                    k_tile_sizes=_bf16_k_tile_sizes(qk_lora_rank),
                    full_out_dim=q_out_dim,
                    out_col_offset=nxt_start * qk_head_dim,
                )
                _load_wuk_bf16_group_tiled(
                    wuk_hbm, wuk_bufs[nxt_buf], n_heads, qk_nope_head_dim, kv_lora_rank, nxt_start, nxt_count
                )

            wuk_sb = wuk_bufs[cur_buf]

            for sub_idx in range(groups_per_dma):
                group_idx = dma_grp_idx * groups_per_dma + sub_idx
                group_start = group_idx * WUK_HEAD_GROUP
                if group_start >= n_heads:
                    continue
                group_count = min(WUK_HEAD_GROUP, n_heads - group_start)
                group_n = group_count * qk_head_dim
                # Column / head-slot base of this compute group inside the wider DMA group.
                grp_col = sub_idx * WUK_HEAD_GROUP * qk_head_dim
                grp_slot = sub_idx * WUK_HEAD_GROUP

                sbm.open_scope()
                sbm.set_name_prefix(f"{tag}_g{group_idx}_")

                for i_tile in nl.affine_range(num_tiles_in_block):
                    sbm.open_scope()
                    sbm.set_name_prefix(f"{tag}_g{group_idx}_t{i_tile}_")

                    s_tile_local_offset = s_block_offset + i_tile * S_TILE_SIZE
                    s_tile_sz = min(S_TILE_SIZE, S_shard - s_tile_local_offset)
                    s_tile_global_offset = S_shard_offset + s_tile_local_offset
                    cos_sb = cos_bufs[i_tile]
                    sin_sb = sin_bufs[i_tile]

                    # Alternate the Q and q_lift PSUM runs on a counter that advances with every
                    # (group, tile) pair, so consecutive matmuls never land on the same bank.
                    _pp = (group_idx * num_tiles_in_block + i_tile) % 2
                    q_group_sb = _bf16_matmul(
                        qr_t_bufs[i_tile],
                        wq_b_bufs[cur_buf],
                        qk_lora_rank,
                        s_tile_pad,
                        group_n,
                        sbm,
                        name=f"q_matmul_g{group_idx}_t{i_tile}",
                        psum_bank_offset=_Q_BANK_BASE + _pp * _Q_BANK_STRIDE,
                        weights_col_offset=grp_col,
                    )

                    # One DMA per buffer for the whole group: each head writes its slot, then the
                    # contiguous head range is stored once.
                    q_pe_grp_sb = sbm.alloc_stack(
                        (P_MAX, group_count * qk_rope_head_dim), dtype=nl.bfloat16, buffer=nl.sbuf, name="q_pe_grp"
                    )
                    q_lift_grp_sb = sbm.alloc_stack(
                        (P_MAX, group_count * kv_lora_rank), dtype=nl.bfloat16, buffer=nl.sbuf, name="q_lift_grp"
                    )

                    # Transpose the WHOLE group's q_nope before any absorption matmul.
                    qnope_t_grp_sb = sbm.alloc_stack(
                        (P_MAX, group_count * NUM_NOPE_TILES, s_tile_pad),
                        dtype=nl.bfloat16,
                        buffer=nl.sbuf,
                        name="qnope_t_grp",
                    )
                    _transpose_q_nope_group_bf16(
                        q_group_sb,
                        qnope_t_grp_sb,
                        group_count=group_count,
                        group_n=group_n,
                        qk_head_dim=qk_head_dim,
                        qk_nope_head_dim=qk_nope_head_dim,
                        s_tile_sz=s_tile_sz,
                        s_tile_pad=s_tile_pad,
                    )

                    for local_head_idx in range(group_count):
                        q_head_offset = local_head_idx * qk_head_dim
                        q_pe_offset = q_head_offset + qk_nope_head_dim
                        sbm.set_name_prefix(f"{tag}_g{group_idx}_t{i_tile}_h{local_head_idx}_")

                        # q_pe: copy raw into the group slot; the whole group is RoPE'd once below.
                        nisa.tensor_copy(
                            dst=q_pe_grp_sb[0:s_tile_sz, nl.ds(local_head_idx * qk_rope_head_dim, qk_rope_head_dim)],
                            src=q_group_sb[0:s_tile_sz, q_pe_offset : q_pe_offset + qk_rope_head_dim],
                            engine=nisa.scalar_engine,
                        )
                        _absorb_q_nope_bf16(
                            qnope_t_grp_sb,
                            wuk_sb,
                            q_lift_grp_sb,
                            nope_tile_base=local_head_idx * NUM_NOPE_TILES,
                            wuk_head_slot=grp_slot + local_head_idx,
                            lift_slot=local_head_idx * kv_lora_rank,
                            s_tile_sz=s_tile_sz,
                            qk_nope_head_dim=qk_nope_head_dim,
                            kv_lora_rank=kv_lora_rank,
                            lift_bank_base=_LIFT_BANK_BASE
                            + ((_pp * group_count + local_head_idx) % 2) * _LIFT_BANK_STRIDE,
                        )

                    sbm.set_name_prefix(f"{tag}_g{group_idx}_t{i_tile}_")

                    # Batched interleaved RoPE over the group's q_pe: 6 wide ops, not 6 per head.
                    rope_grp_scratch = sbm.alloc_stack(
                        (P_MAX, group_count * qk_rope_head_dim),
                        dtype=nl.bfloat16,
                        buffer=nl.sbuf,
                        name="rope_grp_scratch",
                    )
                    _apply_rope_inplace_interleaved_grouped(
                        q_pe_grp_sb, cos_sb, sin_sb, rope_grp_scratch, s_tile_sz, qk_rope_head_dim, group_count
                    )

                    nisa.dma_copy(
                        dst=_s_ap(
                            q_pe_hbm,
                            [[n_heads * qk_rope_head_dim, s_tile_sz], [1, group_count * qk_rope_head_dim]],
                            s_tile_global_offset * n_heads * qk_rope_head_dim + group_start * qk_rope_head_dim,
                            1,
                            sbase_sb,
                        ),
                        src=q_pe_grp_sb[0:s_tile_sz, 0 : group_count * qk_rope_head_dim],
                        dge_mode=dge_mode.hwdge,
                    )
                    nisa.dma_copy(
                        dst=_s_ap(
                            q_lift_hbm,
                            [[n_heads * kv_lora_rank, s_tile_sz], [1, group_count * kv_lora_rank]],
                            s_tile_global_offset * n_heads * kv_lora_rank + group_start * kv_lora_rank,
                            1,
                            sbase_sb,
                        ),
                        src=q_lift_grp_sb[0:s_tile_sz, 0 : group_count * kv_lora_rank],
                        dge_mode=dge_mode.hwdge,
                    )

                    sbm.close_scope()

                sbm.close_scope()

            sbm.close_scope()

        sbm.close_scope()

    # Unrolled below the threshold
    _s_iter_tokens = _S_BLOCKS_PER_ITER * S_BLOCK_SIZE
    if num_S_blocks <= _S_BLOCK_UNROLL_MAX or S_shard % _s_iter_tokens != 0:
        for s_block_idx in range(num_S_blocks):
            _s_block_body(s_block_idx * S_BLOCK_SIZE, None, f"sb{s_block_idx}")
    else:
        sbase_i32 = sbm.alloc_stack((1, 1), dtype=nl.int32, buffer=nl.sbuf, name="s_block_base")

        def _s_block_chunk(_sbase_reg):
            nisa.register_store(dst=sbase_i32, src=_sbase_reg)
            # Blocks within an iteration differ only in their TRACE-TIME offset; the runtime base
            # is shared, so all of them ride the one register store.
            for _j in range(_S_BLOCKS_PER_ITER):
                _s_block_body(_j * S_BLOCK_SIZE, sbase_i32, f"sbL{_j}")

        nl.fori_loop(0, S_shard, _s_block_chunk, step=_s_iter_tokens)

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
