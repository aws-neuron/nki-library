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

"""Shared constants and helpers for the split sparse-MLA CTE kernels (qkv, attention,
MX V-up + MX o_proj), the single source of truth so those kernel files stay independent
of each other and of the qkv_cte MLA utilities."""

from typing import List, Tuple

import nki.isa as nisa
import nki.language as nl
from nki.isa.constants import dge_mode

from ....core.qkv.qkv_cte import _get_psum_bank_size
from ....core.utils.allocator import SbufManager, get_logger, sizeinbytes
from ....core.utils.kernel_assert import kernel_assert
from ....core.utils.kernel_helpers import div_ceil

# Tensor-engine partition cap.
_P_MAX = 128

# Keys gathered/contracted in chunks of 128 (partition limit).
_K_CHUNK = 128

# MM1 writes scores in 512-wide PSUM tiles (one fp32 PSUM bank = 512 elements).
_MM1_TILE = 512

# Softmax exp tile width (K must be a multiple).
_SM_TILE = 1024

# MX x4 packing width.
_H_PACK = 4

# Stream-shuffle mask that replicates the first 16 partitions across a quadrant.
_TI_REPLICATE_MASK = [lane_idx % 16 for lane_idx in range(32)]

_NUM_HW_PSUM_BANKS = 8
_DS_SCALE_BLOCK = 128


def _new_sbm(name):
    return SbufManager(
        sb_lower_bound=0,
        sb_upper_bound=nl.tile_size.total_available_sbuf_size,
        use_auto_alloc=False,
        logger=get_logger(name),
    )


def _transpose_preswizzled_for_mx(input_sb, s_tile_sz, full_num_512, num_512_tiles, out_sb, tpsum, base_col_512=0):
    """Transpose a PRE-SWIZZLED [s_tile, *] tile into the MX-quantize input layout
    [P_MAX, num_512_tiles, s_tile * _H_PACK], reading CONTIGUOUS [s, 128] slices.

    Faster sibling of ``_transpose_natural_for_mx``: that one reads STRIDED [s,128]
    slices (free stride _H_PACK) per (k512, sub) and uses only _H_PACK PSUM banks; this one
    reads CONTIGUOUS [s,128] slices and uses ALL 8 PSUM banks (2 h_tiles at once), so the
    Tensor-Engine transposes pack tighter. It REQUIRES the input columns to be
    pre-swizzled into [_H_PACK, full_num_512, 128] order over the FULL hidden dim: physical
    column ``sub*(full_num_512*128) + k512*128 + p`` holds the natural element
    ``k512*512 + p*_H_PACK + sub`` (the ``_swizzle_mla_cols`` permutation). The producer
    upstream writes its output in this swizzled order (here, the per-head V-up evict).

    Processes ``num_512_tiles`` 512-blocks starting at 512-tile ``base_col_512`` (the sub
    axis spans the WHOLE hidden dim, so the per-sub source stride uses ``full_num_512``,
    not the chunk count) and writes the natural-order flat layout [P_MAX, k512, s, sub]
    into ``out_sb`` — IDENTICAL to ``_transpose_natural_for_mx``'s output, so the
    downstream quantize/matmul are unchanged.

    Allocation-free: the caller provides ``out_sb`` and ``tpsum`` (8 PSUM scratch banks
    [P_MAX, P_MAX]) so this can run in a static chunk loop.
    """
    P_MAX = nl.tile_size.pmax
    tiles_per_group = _NUM_HW_PSUM_BANKS // _H_PACK  # 2 h_tiles at once
    num_groups = num_512_tiles // tiles_per_group
    remainder = num_512_tiles % tiles_per_group

    for group_idx in nl.affine_range(num_groups):
        for tile_in_group in nl.affine_range(tiles_per_group):
            h_tile_local = group_idx * tiles_per_group + tile_in_group
            bank_base = tile_in_group * _H_PACK
            k512 = base_col_512 + h_tile_local
            for sub in nl.affine_range(_H_PACK):
                src_offset = (sub * full_num_512 + k512) * P_MAX
                nisa.nc_transpose(
                    data=input_sb[0:s_tile_sz, src_offset : src_offset + P_MAX],
                    dst=tpsum[bank_base + sub][0:P_MAX, 0:s_tile_sz],
                )
            for sub in nl.affine_range(_H_PACK):
                nisa.tensor_copy(
                    dst=out_sb.ap(
                        pattern=[[num_512_tiles * s_tile_sz * _H_PACK, P_MAX], [_H_PACK, s_tile_sz]],
                        offset=h_tile_local * s_tile_sz * _H_PACK + sub,
                    ),
                    src=tpsum[bank_base + sub][0:P_MAX, 0:s_tile_sz],
                )

    for rem_idx in nl.affine_range(remainder):
        h_tile_local = num_groups * tiles_per_group + rem_idx
        k512 = base_col_512 + h_tile_local
        for sub in nl.affine_range(_H_PACK):
            src_offset = (sub * full_num_512 + k512) * P_MAX
            nisa.nc_transpose(
                data=input_sb[0:s_tile_sz, src_offset : src_offset + P_MAX],
                dst=tpsum[sub][0:P_MAX, 0:s_tile_sz],
            )
        for sub in nl.affine_range(_H_PACK):
            nisa.tensor_copy(
                dst=out_sb.ap(
                    pattern=[[num_512_tiles * s_tile_sz * _H_PACK, P_MAX], [_H_PACK, s_tile_sz]],
                    offset=h_tile_local * s_tile_sz * _H_PACK + sub,
                ),
                src=tpsum[sub][0:P_MAX, 0:s_tile_sz],
            )


def _dma_transpose_latent_for_mx(src_hbm, row_stride, s_tile_sz, num_512_tiles, out_planes, out_perm, base_offset):
    """Transpose the 4-pack-ordered latent activation from HBM into the MX-quantize
    input layout ``out_perm[p, k512, s, sub]`` WITHOUT a Tensor-Engine nc_transpose.

    Requires ``src_hbm`` to carry the cross-kernel 4-pack column order written by
    _attention_stage's MM2: within each 512-wide latent block, physical column
    ``sub * 128 + group`` holds natural latent ``l = 4*group + sub``. So for a fixed
    (k512, sub) the 128 ``group`` columns are CONTIGUOUS in HBM (stride 1).

    Two steps, both off the Tensor Engine (mirrors the production o_proj input load):

    1. DMA engine (swdge dma_transpose): each [s_tile, 128] slice -> a CONTIGUOUS
       [128_group, s_tile] plane in ``out_planes[p, k512, sub, s]`` (sub OUTER, s inner
       -> the transposed/free axes stay contiguous, satisfying the swdge verifier).
    2. Vector engine (tensor_copy): free-axis permute ``[p, k512, sub, s]`` ->
       ``out_perm[p, k512, s, sub]``, making the MX 4-pack (sub) innermost & contiguous,
       exactly what quantize_mx consumes. partition p == group == l//4.
    """
    P_MAX = nl.tile_size.pmax
    GROUPS = P_MAX  # latent groups per 512-block (L//4 == 128)
    for k512 in nl.affine_range(num_512_tiles):
        for sub in nl.affine_range(_H_PACK):
            # sub-outer contiguous plane: out_planes[p, k512, sub, s], free stride 1 on s.
            nisa.dma_transpose(
                dst=out_planes.ap(
                    pattern=[[num_512_tiles * _H_PACK * s_tile_sz, P_MAX], [1, s_tile_sz]],
                    offset=(k512 * _H_PACK + sub) * s_tile_sz,
                ),
                src=src_hbm.ap(
                    pattern=[[row_stride, s_tile_sz], [1, GROUPS]],
                    offset=base_offset + k512 * P_MAX * _H_PACK + sub * GROUPS,
                ),
            )
    """
    Free-axis permute [p, k512, sub, s] -> [p, k512, s, sub] (sub innermost). Vector
    engine handles within-partition strided gathers; one wide op. Iteration order
    (k512, s, sub); strides differ between the two layouts (sub<->s swapped).
    """
    free = num_512_tiles * s_tile_sz * _H_PACK
    nisa.tensor_copy(
        dst=out_perm.ap(
            pattern=[[free, P_MAX], [s_tile_sz * _H_PACK, num_512_tiles], [_H_PACK, s_tile_sz], [1, _H_PACK]],
        ),
        src=out_planes.ap(
            pattern=[[free, P_MAX], [_H_PACK * s_tile_sz, num_512_tiles], [1, s_tile_sz], [s_tile_sz, _H_PACK]],
        ),
    )


def _v32_compute_k_slab_size_512_tiles(
    H: int,
    qkv_out_dim: int,
    q_out_dim: int,
    kv_b_out_dim: int,
    qk_lora_rank: int,
    kv_lora_rank: int,
    n_heads: int,
    qk_head_dim: int,
    v_head_dim: int,
    qk_rope_head_dim: int,
    sbuf_budget_bytes: int,
    num_slab_buffers: int,
    absorbed: bool = False,
    wq_b_resident_n: int = None,
) -> int:
    """Pick the largest slab size (in 512-tiles, dividing H_512) that fits SBUF.

    ``absorbed``: when True, model the ABSORBED MLA kernel's prologue rather than
    v32's. The absorbed kernel emits the latent c_kv directly (W_uk absorbed into Q)
    so it holds NO un-absorbed ``wkv_b`` resident, and it loads ``wq_b`` one head-GROUP
    at a time (``wq_b_resident_n`` output columns) rather than the full ``q_out_dim``.
    Including v32's full ``wkv_b`` + full ``wq_b`` in the prologue (the default) wildly
    over-estimates the absorbed kernel's SBUF peak (e.g. +760KB at 128 heads), forcing
    unnecessary wqkv_a K-slabbing and a per-s-tile weight reload. ``wq_b_resident_n``
    defaults to ``q_out_dim`` (full) when not given.

    Mirrors the qkv_cte look-ahead pattern used elsewhere in this package: pure-Python
    cost model that conservatively estimates per-partition SBUF usage of the v32 MLA
    kernel, varying the number of K (H) 512-tiles loaded simultaneously for
    ``wqkv_a``/``wqkv_a_scale``. Returns the largest divisor of ``H_512`` such that
    ``fixed + num_slab_buffers * slab_cost <= sbuf_budget_bytes``. Returns ``H_512``
    when the fast (single-slab, prologue-loaded) path fits, so existing configs see
    no perf hit.

    Args:
        H: Hidden dimension of input.
        qkv_out_dim: Combined out width of the first projection (qk_lora + kv_lora + rope).
        q_out_dim: ``n_heads * qk_head_dim`` (second-stage Q output width).
        kv_b_out_dim: ``n_heads * (qk_nope + v_head_dim)`` (second-stage KV output width).
        qk_lora_rank, kv_lora_rank: LoRA ranks for Q and KV.
        n_heads: Number of attention heads.
        qk_head_dim, v_head_dim: Head dimensions for Q/K and V.
        qk_rope_head_dim: RoPE portion of qk_head_dim.
        sbuf_budget_bytes: Per-partition SBUF capacity available to the kernel.
        num_slab_buffers: 1 for single-buffered slabs, 2 for double-buffered (prefetch).

    Returns:
        Slab size in 512-tiles, in [1, H_512] and dividing H_512.
    """
    P_MAX = nl.tile_size.pmax
    H_PACK_LOCAL = 4
    H_512 = H // (P_MAX * H_PACK_LOCAL)

    fp8x4 = sizeinbytes(nl.float8_e4m3fn_x4)
    u8 = sizeinbytes(nl.uint8)
    bf16 = sizeinbytes(nl.bfloat16)
    f32 = sizeinbytes(nl.float32)

    qk_lora_512 = qk_lora_rank // (P_MAX * H_PACK_LOCAL)
    kv_lora_512 = kv_lora_rank // (P_MAX * H_PACK_LOCAL)

    """
    Tile-scope intermediates (qr_sb, transposes, quants, q/k/v assembly, etc.)
    all live within nested SBUF scopes that share storage with the slab buffer
    via the SbufManager stack. The peak SBUF in the FAST path is empirically
    bounded by:
      prologue (norms + eps + zero + wq_b + wkv_b + 2x x_sb + shfl + 2x x_qtz)
      + wqkv_a + wqkv_a_scale (the streamable weights)
      + a constant per-tile working set that's the same in fast and slabbed paths.
    We model it as: prologue_fixed + slab_cost + per_tile_const.
    The slabbed path reduces the slab cost (proportional to slab_size_512); the
    per_tile_const is identical between paths, so this comparison is what we want.
    Scale buffers are padded to whole 128-column blocks on N (compact
    block-128 scales materialize to ``ceil(N/128) * 128``).
    """
    SCALE_BLOCK = 128
    # Absorbed kernel: wq_b is held only group-sized; no un-absorbed wkv_b resident.
    wq_b_n = wq_b_resident_n if (absorbed and wq_b_resident_n != None) else q_out_dim
    q_out_dim_padded = ((wq_b_n + SCALE_BLOCK - 1) // SCALE_BLOCK) * SCALE_BLOCK
    kv_b_out_dim_padded = ((kv_b_out_dim + SCALE_BLOCK - 1) // SCALE_BLOCK) * SCALE_BLOCK
    qkv_out_dim_padded = ((qkv_out_dim + SCALE_BLOCK - 1) // SCALE_BLOCK) * SCALE_BLOCK

    prologue_fixed = 0
    prologue_fixed += qk_lora_512 * H_PACK_LOCAL * f32
    prologue_fixed += kv_lora_512 * H_PACK_LOCAL * f32
    prologue_fixed += 2 * bf16
    # wq_b weights + materialized scales (DMA-broadcast directly). Group-sized for the
    # absorbed kernel (wq_b_n), full q_out_dim for v32.
    prologue_fixed += qk_lora_512 * wq_b_n * fp8x4 + qk_lora_512 * q_out_dim_padded * u8
    # Un-absorbed wkv_b is resident only in v32; the absorbed kernel uses bf16 W_uk
    # (counted separately by the caller) and emits c_kv directly, so skip it here.
    if not absorbed:
        prologue_fixed += kv_lora_512 * kv_b_out_dim * fp8x4 + kv_lora_512 * kv_b_out_dim_padded * u8
    prologue_fixed += 2 * H * bf16
    prologue_fixed += H_512 * P_MAX * H_PACK_LOCAL * bf16
    prologue_fixed += 2 * H_512 * P_MAX * (fp8x4 + u8)

    """
    Per-tile working set that exists at peak (qr_sb + kv_raw + qr_transpose +
    qr_qtz/scale + kv_transpose + kv_qtz/scale + q/k/v assembly buffers +
    RoPE scratch). Conservatively estimated; same in both paths.
    """
    per_tile_const = 0
    per_tile_const += qk_lora_rank * bf16
    per_tile_const += (kv_lora_rank + qk_rope_head_dim) * bf16
    per_tile_const += qk_lora_512 * P_MAX * H_PACK_LOCAL * bf16
    per_tile_const += qk_lora_512 * P_MAX * (fp8x4 + u8)
    per_tile_const += kv_lora_512 * P_MAX * H_PACK_LOCAL * bf16
    per_tile_const += kv_lora_512 * P_MAX * (fp8x4 + u8)
    per_tile_const += q_out_dim * bf16
    if absorbed:
        """
        Absorbed kernel: NO un-absorbed kv_b / per-head Q/K/V assembly buffers. Instead
        the bf16 W_uk is loaded resident in the head loop ([nope=P_MAX, n_heads*kv_lora]),
        which dominates the per-tile peak at high head counts (128KB at 128 heads). Count
        it here so the slab sizer leaves room for it on the per-tile (non-resident) path.
        """
        per_tile_const += n_heads * kv_lora_rank * bf16  # W_uk resident in head loop
    else:
        per_tile_const += kv_b_out_dim * bf16
        per_tile_const += n_heads * qk_head_dim * bf16  # q_tile_sb
        per_tile_const += n_heads * qk_head_dim * bf16  # k_tile_sb
        per_tile_const += n_heads * v_head_dim * bf16  # v_tile_sb
    per_tile_const += qk_rope_head_dim * 4 * bf16  # rope scratch + cos/sin

    # Slab cost: weights + materialized scales (DMA-broadcast directly).
    slab_byte_per_512_tile = qkv_out_dim * fp8x4 + qkv_out_dim_padded * u8

    for divisor in range(1, H_512 + 1):
        if H_512 % divisor != 0:
            continue
        slab_size_512 = H_512 // divisor
        total = prologue_fixed + num_slab_buffers * slab_size_512 * slab_byte_per_512_tile + per_tile_const
        if total <= sbuf_budget_bytes:
            return slab_size_512
    return 1


def _load_mx_weights_k_slab(
    weights_hbm: nl.NkiTensor,
    scales_hbm: nl.NkiTensor,
    weights_sb: nl.NkiTensor,
    scales_sb: nl.NkiTensor,
    in_dim_full: int,
    out_dim: int,
    k_tile_start: int,
    k_tile_count: int,
    sbm: SbufManager,
    name: str = "slab",
    full_out_dim: int = None,
    out_col_offset: int = 0,
    weights_dge_mode=dge_mode.swdge,
) -> None:
    """Load a K-slab ``[k_tile_start : k_tile_start + k_tile_count]`` of MX weights.

    Companion to :func:`_load_mx_weights` for K-streaming. Same two-stage load
    used to materialize the full MX scale layout: compact DMA from HBM into an
    inner-scope scratch (no free-dim broadcast at DMA time, since the engine
    falls back to per-element when stride-0 hits the free dim), then a
    vector-engine ``tensor_copy`` with stride-0 source broadcast to expand
    into ``scales_sb``. The compact scratch is freed before the function
    returns so callers see no extra SBUF pressure.

    Supports streaming an N-slice of a wider HBM weight via ``full_out_dim`` /
    ``out_col_offset`` (mirrors :func:`_load_mx_weights`), so a kernel can tile
    over BOTH the K (contraction) and N (output) axes of a weight too large to
    hold resident.

    Args:
        weights_hbm: ``[in_dim_full // 4, full_out_dim]`` fp8x4 weights on HBM.
        scales_hbm: ``[in_dim_full // 128, ceil(full_out_dim / 128)]`` uint8 compact
            block-128 scales on HBM.
        weights_sb: ``[P_MAX, k_tile_count, out_dim]`` slab destination.
        scales_sb: ``[P_MAX, k_tile_count, ceil(out_dim/128) * 128]`` slab
            full-MX scales destination consumed by ``nc_matmul_mx``.
        in_dim_full: Full K dimension of the HBM tensor.
        out_dim: N dimension to LOAD (slice width if slicing).
        k_tile_start: Index of the first 512-tile in this slab.
        k_tile_count: Number of 512-tiles in this slab.
        sbm: SBUF memory manager used for the inner-scope compact scratch.
        full_out_dim: Full N dimension of the HBM tensor; defaults to ``out_dim``.
        out_col_offset: Column index of the N-slice start (multiple of 128).
        weights_dge_mode: DGE mode for the weight DMA. Defaults to ``swdge``; pass
            ``hwdge`` when the weight read is a plain rectangular strided pattern
            (no free-dim broadcast) to use the faster hardware descriptor engine.
    """
    if full_out_dim == None:
        full_out_dim = out_dim
    P_MAX = nl.tile_size.pmax
    SCALE_P_PER_QUAD = 4
    SCALE_BLOCK = 128
    QUADS_PER_TILE = 4

    kernel_assert(
        out_col_offset % SCALE_BLOCK == 0,
        f"out_col_offset ({out_col_offset}) must be a multiple of {SCALE_BLOCK} for compact block-128 scales.",
    )

    in_dim_packed = in_dim_full // 4
    full_n_blocks = (full_out_dim + SCALE_BLOCK - 1) // SCALE_BLOCK
    out_n_blocks = (out_dim + SCALE_BLOCK - 1) // SCALE_BLOCK
    n_block_offset = out_col_offset // SCALE_BLOCK

    for h_local in nl.affine_range(k_tile_count):
        h_global = k_tile_start + h_local
        h_tile_sz = min(P_MAX, in_dim_packed - h_global * P_MAX)
        nisa.dma_copy(
            dst=weights_sb[0:h_tile_sz, h_local, 0:out_dim],
            src=weights_hbm.ap(
                pattern=[[full_out_dim, h_tile_sz], [1, out_dim]],
                offset=h_global * P_MAX * full_out_dim + out_col_offset,
                dtype=nl.float8_e4m3fn_x4,
            ),
            dge_mode=weights_dge_mode,
        )

    # ---- Stage 1: DMA compact scales (no free-dim broadcast at DMA time). ----
    sbm.open_scope()
    compact_scales_sb = sbm.alloc_stack(
        (P_MAX, k_tile_count, out_n_blocks),
        dtype=nl.uint8,
        buffer=nl.sbuf,
        name=f"{name}_scales_compact",
    )
    for quad_idx in nl.affine_range(QUADS_PER_TILE):
        slab_start_compact_row = k_tile_start * QUADS_PER_TILE + quad_idx
        nisa.dma_copy(
            dst=compact_scales_sb[nl.ds(quad_idx * 32, SCALE_P_PER_QUAD), 0:k_tile_count, 0:out_n_blocks],
            src=scales_hbm.ap(
                pattern=[
                    [0, SCALE_P_PER_QUAD],
                    [QUADS_PER_TILE * full_n_blocks, k_tile_count],
                    [1, out_n_blocks],
                ],
                offset=slab_start_compact_row * full_n_blocks + n_block_offset,
                dtype=nl.uint8,
            ),
            dge_mode=dge_mode.swdge,
        )

    # ---- Stage 2: Vector-engine broadcast into the materialized scales_sb. ----
    src_view = compact_scales_sb.expand_dim(dim=3).broadcast(dim=3, size=SCALE_BLOCK)
    dst_view = scales_sb.reshape_dim(dim=2, shape=(out_n_blocks, SCALE_BLOCK))
    nisa.tensor_copy(dst=dst_view, src=src_view)

    sbm.close_scope()  # Frees compact_scales_sb.


def _mx_matmul_split_k_range(
    input_qtz_sb: nl.NkiTensor,
    input_scale_sb: nl.NkiTensor,
    weights_slab_sb: nl.NkiTensor,
    weights_slab_scale_sb: nl.NkiTensor,
    k_tile_start: int,
    k_tile_count: int,
    m_dim: int,
    n_dim: int,
    output_psum: list,
    input_scale_packed: bool = False,
) -> None:
    """Run nc_matmul_mx for one K-slab, accumulating into the caller-provided PSUM banks.

    Mirrors the matmul body of :func:`_mx_matmul_split` restricted to a K range.
    Reads input data/scale from indices ``[k_tile_start : k_tile_start + k_tile_count]``
    of the kernel-wide quantized input, and weight data/scale from the slab-local
    indices ``[0 : k_tile_count]``. PSUM accumulates implicitly across calls because
    each call writes to the same destination banks.

    Args:
        input_qtz_sb: ``[P_MAX, num_k_tiles_full, m_dim]`` fp8x4 stationary input
            (kernel-wide; this call slices ``[k_tile_start : k_tile_start + k_tile_count]``).
        input_scale_sb: ``[P_MAX, num_k_tiles_full, m_dim]`` uint8 stationary scale.
        weights_slab_sb: ``[P_MAX, k_tile_count, n_dim]`` fp8x4 slab weights (slab-local indexing).
        weights_slab_scale_sb: ``[P_MAX, k_tile_count, n_dim]`` uint8 slab weight scales.
        k_tile_start: Slab's starting global K-tile index (used to index input).
        k_tile_count: Number of K-tiles in this slab.
        m_dim: M dimension of the matmul (typically s_tile_pad).
        n_dim: N dimension (full output width).
        output_psum: List of PSUM bank ndarrays, one per N-tile. Caller allocates and
            issues the post-loop PSUM->SBUF copy.
    """
    P_MAX = nl.tile_size.pmax
    F_MAX = 512
    num_n_tiles = div_ceil(n_dim, F_MAX)

    for k_local in nl.affine_range(k_tile_count):
        k_global = k_tile_start + k_local
        # Packed input scale: K-tile k_global lives in packed buffer k_global//4 at
        # within-quadrant partition offset (k_global%4)*4 (is_packed_moving_scale layout).
        if input_scale_packed:
            in_scale_view = input_scale_sb[(k_global % 4) * 4 : P_MAX, k_global // 4, nl.ds(0, m_dim)]
        else:
            in_scale_view = input_scale_sb[0:P_MAX, k_global, nl.ds(0, m_dim)]
        for i_n_tile in nl.affine_range(num_n_tiles):
            n_tile_sz = min(F_MAX, n_dim - i_n_tile * F_MAX)
            nisa.nc_matmul_mx(
                dst=output_psum[i_n_tile][0:m_dim, 0:n_tile_sz],
                stationary=input_qtz_sb[0:P_MAX, k_global, nl.ds(0, m_dim)],
                moving=weights_slab_sb[0:P_MAX, k_local, nl.ds(i_n_tile * F_MAX, n_tile_sz)],
                stationary_scale=in_scale_view,
                moving_scale=weights_slab_scale_sb[0:P_MAX, k_local, nl.ds(i_n_tile * F_MAX, n_tile_sz)],
            )


def _load_mx_weights(
    weights_hbm: nl.NkiTensor,
    scales_hbm: nl.NkiTensor,
    in_dim: int,
    out_dim: int,
    sbm: SbufManager,
    name: str = "mx",
    full_out_dim: int = None,
    out_col_offset: int = 0,
) -> Tuple[nl.NkiTensor, nl.NkiTensor]:
    """Load MX weights and DeepSeek-style block-128 scales from HBM to SBUF.

    Loads fp8x4 packed weights and per-(128 K x 128 N) block uint8 scales
    (DeepSeek V3.2 ``scale_fmt=ue8m0`` format). The scales tensor on HBM is
    compact: one byte per 128x128 weight block. On SBUF the scales are
    expanded into the MX hardware layout that ``nisa.nc_matmul_mx`` expects.

    Two-stage load:

    1. DMA compact scales to a small SBUF scratch ``[P_MAX, num_512_tiles,
       out_n_blocks]``. The DMA pattern uses stride-0 only on the partition
       axis (within a quadrant, broadcasting one source byte across 4 partition
       rows) — which the DMA engine natively supports. Free-dim stride-0 fan-out
       is NOT done here; the DMA engine doesn't support it and falls back to
       per-element copies, which serializes the prologue.
    2. Vector-engine ``tensor_copy`` materializes the broadcast layout from the
       compact scratch into the final ``scales_sb`` (``[P_MAX, num_512_tiles,
       padded_out_dim]``). Vector engine permits stride-0 source addressing,
       so the broadcast is one wide compute-engine op instead of a DMA fallback.

    The compact scratch lives in an inner SBUF scope and is released before the
    function returns, so callers see no extra SBUF pressure.

    Args:
        weights_hbm: ``[in_dim // 4, full_out_dim]`` fp8x4 weights on HBM.
        scales_hbm: ``[in_dim // 128, ceil(full_out_dim / 128)]`` uint8 compact
            block-128 scales on HBM.
        in_dim: K dimension of the matmul.
        out_dim: N dimension to ALLOCATE and load (slice width if slicing).
        sbm: SBUF memory manager.
        name: Name prefix for allocated buffers.
        full_out_dim: Full N dimension of the HBM tensor; defaults to ``out_dim``.
        out_col_offset: Column index of the slice start (multiple of 128).

    Returns:
        ``(weights_sb, scales_sb)``. ``weights_sb`` is
        ``[P_MAX, num_512_tiles, out_dim]``; ``scales_sb`` is
        ``[P_MAX, num_512_tiles, ceil(out_dim/128) * 128]`` with each block
        scale replicated SCALE_BLOCK times along the trailing dim.
    """
    if full_out_dim == None:
        full_out_dim = out_dim
    P_MAX = nl.tile_size.pmax
    SCALE_P_PER_QUAD = 4
    SCALE_BLOCK = 128
    QUADS_PER_TILE = 4

    kernel_assert(
        out_col_offset % SCALE_BLOCK == 0,
        f"out_col_offset ({out_col_offset}) must be a multiple of {SCALE_BLOCK} for compact block-128 scales.",
    )

    in_dim_packed = in_dim // 4
    num_512_tiles = in_dim // (P_MAX * _H_PACK)

    full_n_blocks = (full_out_dim + SCALE_BLOCK - 1) // SCALE_BLOCK
    out_n_blocks = (out_dim + SCALE_BLOCK - 1) // SCALE_BLOCK
    n_block_offset = out_col_offset // SCALE_BLOCK
    padded_out_dim = out_n_blocks * SCALE_BLOCK

    weights_sb = sbm.alloc_stack(
        (P_MAX, num_512_tiles, out_dim),
        dtype=nl.float8_e4m3fn_x4,
        buffer=nl.sbuf,
        name=f"{name}_weights",
    )
    scales_sb = sbm.alloc_stack(
        (P_MAX, num_512_tiles, padded_out_dim),
        dtype=nl.uint8,
        buffer=nl.sbuf,
        name=f"{name}_scales",
    )

    for h_tile_idx in nl.affine_range(num_512_tiles):
        h_tile_sz = min(P_MAX, in_dim_packed - h_tile_idx * P_MAX)
        nisa.dma_copy(
            dst=weights_sb[0:h_tile_sz, h_tile_idx, 0:out_dim],
            src=weights_hbm.ap(
                pattern=[[full_out_dim, h_tile_sz], [1, out_dim]],
                offset=h_tile_idx * P_MAX * full_out_dim + out_col_offset,
                dtype=nl.float8_e4m3fn_x4,
            ),
            dge_mode=dge_mode.swdge,
        )

    """
    ---- Stage 1: DMA compact scales (no free-dim broadcast at DMA time). ----
    Inner scope: compact_scales_sb is released after stage 2 so it doesn't
    hold SBUF for the kernel duration.
    """
    sbm.open_scope()
    compact_scales_sb = sbm.alloc_stack(
        (P_MAX, num_512_tiles, out_n_blocks),
        dtype=nl.uint8,
        buffer=nl.sbuf,
        name=f"{name}_scales_compact",
    )
    """
    Source pattern: 4-row partition broadcast within each quadrant (stride 0
    on partition axis is legal for DMA), then the K-tile and N-block walks.
    No [0, SCALE_BLOCK] free-dim broadcast — that fanout is done in stage 2.
    """
    for quad_idx in nl.affine_range(QUADS_PER_TILE):
        nisa.dma_copy(
            dst=compact_scales_sb[nl.ds(quad_idx * 32, SCALE_P_PER_QUAD), 0:num_512_tiles, 0:out_n_blocks],
            src=scales_hbm.ap(
                pattern=[
                    [0, SCALE_P_PER_QUAD],
                    [QUADS_PER_TILE * full_n_blocks, num_512_tiles],
                    [1, out_n_blocks],
                ],
                offset=quad_idx * full_n_blocks + n_block_offset,
                dtype=nl.uint8,
            ),
            dge_mode=dge_mode.swdge,
        )

    """
    ---- Stage 2: Vector-engine broadcast into the final scales_sb. ----
    Source view: [P_MAX, num_512_tiles, n_blocks, 1] -> broadcast(dim=3, size=128).
    Destination view: [P_MAX, num_512_tiles, n_blocks, 128] (reshape of the 3D scales_sb).
    """
    src_view = compact_scales_sb.expand_dim(dim=3).broadcast(dim=3, size=SCALE_BLOCK)
    dst_view = scales_sb.reshape_dim(dim=2, shape=(out_n_blocks, SCALE_BLOCK))
    nisa.tensor_copy(dst=dst_view, src=src_view)

    sbm.close_scope()  # Frees compact_scales_sb.

    return weights_sb, scales_sb


def _load_norm_weights_for_mx(
    norm_weights_hbm: nl.NkiTensor,
    dim: int,
    sbm: SbufManager,
    name: str = "norm_gamma",
) -> nl.NkiTensor:
    """
    Load norm weights [1, dim] to SBUF in swizzled format for MX path.

    Gathers elements with stride-4 to produce a layout where each column contains
    one element per 128-element sub-tile, matching the MX quantization layout.

    Args:
        norm_weights_hbm (nl.NkiTensor): [1, dim] bf16, Norm gamma weights on HBM
        dim (int): Hidden dimension size
        sbm (SbufManager): SBUF memory manager
        name (str): Name for the allocated buffer

    Returns:
        nl.NkiTensor: [P_MAX, num_512_tiles * _H_PACK] float32, Swizzled gamma weights in SBUF
    """
    P_MAX = nl.tile_size.pmax
    num_512_tiles = dim // (P_MAX * _H_PACK)

    norm_weights_hbm = norm_weights_hbm.reshape((1, dim))
    gamma_sb = sbm.alloc_stack((P_MAX, num_512_tiles * _H_PACK), dtype=nl.float32, buffer=nl.sbuf, name=name)

    """
    Single strided DMA (was num_512_tiles * _H_PACK one-column copies). The gather maps
    gamma_sb[p, tile*_H_PACK + sub] = gamma[tile*512 + _H_PACK*p + sub]. Expressed as one 3D
    access pattern over (p, tile, sub): src strides (_H_PACK, P_MAX*_H_PACK, 1), dst column
    tile*_H_PACK + sub (strides _H_PACK, 1 over the free axis; partition stride num_512*_H_PACK).
    """
    nisa.dma_copy(
        dst=gamma_sb.ap(
            pattern=[[num_512_tiles * _H_PACK, P_MAX], [_H_PACK, num_512_tiles], [1, _H_PACK]],
            offset=0,
        ),
        src=norm_weights_hbm.ap(
            pattern=[[_H_PACK, P_MAX], [P_MAX * _H_PACK, num_512_tiles], [1, _H_PACK]],
            offset=0,
        ),
        dge_mode=dge_mode.swdge,
    )
    return gamma_sb


def _quantize_mx(
    transposed_sb: nl.NkiTensor,
    num_512_tiles: int,
    s_tile_sz: int,
    sbm: SbufManager,
    name: str = "qtz",
) -> Tuple[nl.NkiTensor, nl.NkiTensor]:
    """
    Quantize bf16 tensor to MX format (fp8x4 + uint8 scales).

    Args:
        transposed_sb (nl.NkiTensor): [P_MAX, num_512_tiles, s_tile_sz * _H_PACK], Input tensor in SBUF
        num_512_tiles (int): Number of 512-element tiles
        s_tile_sz (int): Number of active sequence positions in the tile
        sbm (SbufManager): SBUF memory manager
        name (str): Name prefix for allocated buffers

    Returns:
        Tuple[nl.NkiTensor, nl.NkiTensor]:
            - qtz_sb (nl.NkiTensor): [P_MAX, num_512_tiles, s_tile_sz] fp8x4, Quantized values
            - scale_sb (nl.NkiTensor): [P_MAX, num_512_tiles, s_tile_sz] uint8, Per-block scales
    """
    P_MAX = nl.tile_size.pmax

    qtz_sb = sbm.alloc_stack(
        (P_MAX, num_512_tiles, s_tile_sz),
        dtype=nl.float8_e4m3fn_x4,
        buffer=nl.sbuf,
        name=f"{name}_data",
    )
    scale_sb = sbm.alloc_stack(
        (P_MAX, num_512_tiles, s_tile_sz),
        dtype=nl.uint8,
        buffer=nl.sbuf,
        name=f"{name}_scale",
    )
    nisa.quantize_mx(
        src=transposed_sb[0:P_MAX, 0:num_512_tiles, 0 : s_tile_sz * _H_PACK],
        dst=qtz_sb[0:P_MAX, 0:num_512_tiles, 0:s_tile_sz],
        dst_scale=scale_sb[0:P_MAX, 0:num_512_tiles, 0:s_tile_sz],
    )

    return qtz_sb, scale_sb


def _mx_matmul(
    input_qtz_sb: nl.NkiTensor,
    input_scale_sb: nl.NkiTensor,
    weights_sb: nl.NkiTensor,
    weights_scale_sb: nl.NkiTensor,
    num_k_tiles: int,
    m_dim: int,
    n_dim: int,
    sbm: SbufManager,
    name: str = "matmul",
) -> nl.NkiTensor:
    """
    MX matrix multiplication: input @ weights.

    Args:
        input_qtz_sb (nl.NkiTensor): [P_MAX, num_k_tiles, m_dim] fp8x4, Quantized input (stationary)
        input_scale_sb (nl.NkiTensor): [P_MAX, num_k_tiles, m_dim] uint8, Input scales
        weights_sb (nl.NkiTensor): [P_MAX, num_k_tiles, n_dim] fp8x4, Quantized weights (moving)
        weights_scale_sb (nl.NkiTensor): [P_MAX, num_k_tiles, n_dim] uint8, Weight scales
        num_k_tiles (int): Number of K-dimension tiles to accumulate over
        m_dim (int): M dimension (sequence tile size)
        n_dim (int): N dimension (total output width)
        sbm (SbufManager): SBUF memory manager
        name (str): Name prefix for allocated buffers

    Returns:
        nl.NkiTensor: [m_dim, n_dim] bf16, Matrix multiplication result in SBUF
    """
    P_MAX = nl.tile_size.pmax
    F_MAX = 512
    PSUM_BANK_SIZE = _get_psum_bank_size()

    num_n_tiles = div_ceil(n_dim, F_MAX)

    output_psum = []
    for bank_id in nl.affine_range(num_n_tiles):
        output_psum.append(
            nl.ndarray(
                (P_MAX, F_MAX),
                dtype=nl.bfloat16,
                buffer=nl.psum,
                address=(0, bank_id * PSUM_BANK_SIZE),
            )
        )

    for i_k_tile in nl.affine_range(num_k_tiles):
        for i_n_tile in nl.affine_range(num_n_tiles):
            n_tile_sz = min(F_MAX, n_dim - i_n_tile * F_MAX)
            nisa.nc_matmul_mx(
                dst=output_psum[i_n_tile][0:m_dim, 0:n_tile_sz],
                stationary=input_qtz_sb[0:P_MAX, i_k_tile, nl.ds(0, m_dim)],
                moving=weights_sb[0:P_MAX, i_k_tile, nl.ds(i_n_tile * F_MAX, n_tile_sz)],
                stationary_scale=input_scale_sb[0:P_MAX, i_k_tile, nl.ds(0, m_dim)],
                moving_scale=weights_scale_sb[0:P_MAX, i_k_tile, nl.ds(i_n_tile * F_MAX, n_tile_sz)],
            )

    output_sb = sbm.alloc_stack((P_MAX, n_dim), dtype=nl.bfloat16, buffer=nl.sbuf, name=f"{name}_output")
    for i_n_tile in nl.affine_range(num_n_tiles):
        n_tile_sz = min(F_MAX, n_dim - i_n_tile * F_MAX)
        nisa.tensor_copy(
            dst=output_sb[0:m_dim, nl.ds(i_n_tile * F_MAX, n_tile_sz)],
            src=output_psum[i_n_tile][0:m_dim, 0:n_tile_sz],
        )

    return output_sb


def _mx_matmul_split(
    input_qtz_sb: nl.NkiTensor,
    input_scale_sb: nl.NkiTensor,
    weights_sb: nl.NkiTensor,
    weights_scale_sb: nl.NkiTensor,
    num_k_tiles: int,
    m_dim: int,
    n_dim: int,
    split_points: List[int],
    sbm: SbufManager,
    input_scale_packed: bool = False,
) -> List[nl.NkiTensor]:
    """
    MX matmul with split output: input @ weights -> multiple output buffers.

    Split happens during PSUM->SBUF copy to avoid extra memory movement.

    Args:
        input_qtz_sb (nl.NkiTensor): [P_MAX, num_k_tiles, m_dim] fp8x4, Quantized input (stationary)
        input_scale_sb (nl.NkiTensor): [P_MAX, num_k_tiles, m_dim] uint8, Input scales
        weights_sb (nl.NkiTensor): [P_MAX, num_k_tiles, n_dim] fp8x4, Quantized weights (moving)
        weights_scale_sb (nl.NkiTensor): [P_MAX, num_k_tiles, n_dim] uint8, Weight scales
        num_k_tiles (int): Number of K-dimension tiles to accumulate over
        m_dim (int): M dimension (sequence tile size)
        n_dim (int): N dimension (total output width)
        split_points (List[int]): Column indices at which to split the output
        sbm (SbufManager): SBUF memory manager

    Returns:
        List[nl.NkiTensor]: List of SBUF tensors, one per split segment
    """
    P_MAX = nl.tile_size.pmax
    F_MAX = 512
    PSUM_BANK_SIZE = _get_psum_bank_size()

    num_n_tiles = div_ceil(n_dim, F_MAX)

    output_psum = []
    for bank_id in nl.affine_range(num_n_tiles):
        output_psum.append(
            nl.ndarray(
                (P_MAX, F_MAX),
                dtype=nl.bfloat16,
                buffer=nl.psum,
                address=(0, bank_id * PSUM_BANK_SIZE),
            )
        )

    for i_k_tile in nl.affine_range(num_k_tiles):
        # Packed input scale: K-tile i_k_tile in packed buffer i_k_tile//4 at within-quadrant
        # partition offset (i_k_tile%4)*4 (is_packed_moving_scale layout).
        if input_scale_packed:
            in_scale_view = input_scale_sb[(i_k_tile % 4) * 4 : P_MAX, i_k_tile // 4, nl.ds(0, m_dim)]
        else:
            in_scale_view = input_scale_sb[0:P_MAX, i_k_tile, nl.ds(0, m_dim)]
        for i_n_tile in nl.affine_range(num_n_tiles):
            n_tile_sz = min(F_MAX, n_dim - i_n_tile * F_MAX)
            nisa.nc_matmul_mx(
                dst=output_psum[i_n_tile][0:m_dim, 0:n_tile_sz],
                stationary=input_qtz_sb[0:P_MAX, i_k_tile, nl.ds(0, m_dim)],
                moving=weights_sb[0:P_MAX, i_k_tile, nl.ds(i_n_tile * F_MAX, n_tile_sz)],
                stationary_scale=in_scale_view,
                moving_scale=weights_scale_sb[0:P_MAX, i_k_tile, nl.ds(i_n_tile * F_MAX, n_tile_sz)],
            )

    all_splits = [0] + split_points + [n_dim]
    outputs = []

    for split_idx in range(len(all_splits) - 1):
        start_col = all_splits[split_idx]
        end_col = all_splits[split_idx + 1]
        width = end_col - start_col

        out_sb = sbm.alloc_stack(
            (P_MAX, width),
            dtype=nl.bfloat16,
            buffer=nl.sbuf,
            name=f"matmul_split_out_{split_idx}",
        )

        col = 0
        while col < width:
            global_col = start_col + col
            bank_idx, bank_offset = divmod(global_col, F_MAX)
            copy_width = min(F_MAX - bank_offset, width - col)

            nisa.tensor_copy(
                dst=out_sb[0:m_dim, nl.ds(col, copy_width)],
                src=output_psum[bank_idx][0:m_dim, nl.ds(bank_offset, copy_width)],
                engine=nisa.scalar_engine,
            )
            col += copy_width

        outputs.append(out_sb)

    return outputs


def _transpose_preswizzled_for_mx_fused(
    input_sb: nl.NkiTensor,
    s_tile_sz: int,
    hidden_dim: int,
    num_512_tiles: int,
    sbm: SbufManager,
    gamma_sb: nl.NkiTensor = None,
    rsqrt_scale_sb: nl.NkiTensor = None,
    name: str = "transpose_preswizzled",
) -> nl.NkiTensor:
    """Transpose pre-swizzled [S, H] -> [H, S] for MX quantization using all 8 PSUM banks.

    Processes two h_tiles simultaneously using 8 PSUM banks (4 per h_tile).
    When ``rsqrt_scale_sb`` and ``gamma_sb`` are both provided, fuses
    rsqrt_scale * gamma into a combined scale buffer up-front so the
    PSUM->SBUF eviction performs a single multiply instead of two ops.

    Args:
        input_sb (nl.NkiTensor): [s_tile_sz, hidden_dim], Pre-swizzled input in SBUF.
        s_tile_sz (int): Number of active sequence positions in the tile.
        hidden_dim (int): Hidden dimension size.
        num_512_tiles (int): Number of 512-element tiles along H.
        sbm (SbufManager): SBUF memory manager.
        gamma_sb (nl.NkiTensor, optional): [P_MAX, num_512_tiles * _H_PACK] gamma weights,
            applied during the eviction multiply if provided.
        rsqrt_scale_sb (nl.NkiTensor, optional): [s_tile_sz, 1] rsqrt scale; if also
            ``gamma_sb`` is provided, the two are pre-multiplied into a fused gamma.
        name (str): Name prefix for allocated buffers.

    Returns:
        nl.NkiTensor: [P_MAX, num_512_tiles, s_tile_sz * _H_PACK], Transposed tensor in SBUF.
    """
    if rsqrt_scale_sb != None and gamma_sb != None:
        fused_gamma_sb = sbm.alloc_stack(gamma_sb.shape, dtype=nl.float32, buffer=nl.sbuf, name=f"{name}_fused_gamma")
        nisa.activation(
            dst=fused_gamma_sb,
            op=nl.copy,
            data=gamma_sb,
            scale=rsqrt_scale_sb,
        )
        gamma_sb = fused_gamma_sb

    P_MAX = nl.tile_size.pmax
    PSUM_BANK_SIZE = _get_psum_bank_size()
    TILES_PER_GROUP = _NUM_HW_PSUM_BANKS // _H_PACK  # 2 h_tiles at once

    transposed_sb = sbm.alloc_stack(
        (P_MAX, num_512_tiles, s_tile_sz * _H_PACK),
        dtype=nl.bfloat16,
        buffer=nl.sbuf,
        name=name,
    )

    transpose_psum = []
    for bank_id in range(_NUM_HW_PSUM_BANKS):
        transpose_psum.append(
            nl.ndarray(
                (P_MAX, P_MAX),
                dtype=nl.bfloat16,
                buffer=nl.psum,
                address=(0, bank_id * PSUM_BANK_SIZE),
            )
        )

    num_groups = num_512_tiles // TILES_PER_GROUP
    remainder = num_512_tiles % TILES_PER_GROUP

    for group_idx in nl.affine_range(num_groups):
        for tile_in_group in nl.affine_range(TILES_PER_GROUP):
            h_tile_idx = group_idx * TILES_PER_GROUP + tile_in_group
            bank_base = tile_in_group * _H_PACK
            for h_sub_idx in nl.affine_range(_H_PACK):
                src_offset = (h_sub_idx * num_512_tiles + h_tile_idx) * P_MAX
                nisa.nc_transpose(
                    data=input_sb[0:s_tile_sz, src_offset : src_offset + P_MAX],
                    dst=transpose_psum[bank_base + h_sub_idx][0:P_MAX, 0:s_tile_sz],
                )

        for tile_in_group in nl.affine_range(TILES_PER_GROUP):
            h_tile_idx = group_idx * TILES_PER_GROUP + tile_in_group
            bank_base = tile_in_group * _H_PACK
            for h_sub_idx in nl.affine_range(_H_PACK):
                dst_ap = transposed_sb.ap(
                    pattern=[[num_512_tiles * s_tile_sz * _H_PACK, P_MAX], [_H_PACK, s_tile_sz]],
                    offset=h_tile_idx * s_tile_sz * _H_PACK + h_sub_idx,
                )
                if gamma_sb != None:
                    gamma_tile_index = h_tile_idx * _H_PACK + h_sub_idx
                    nisa.tensor_scalar(
                        dst=dst_ap,
                        data=transpose_psum[bank_base + h_sub_idx][0:P_MAX, 0:s_tile_sz],
                        op0=nl.multiply,
                        operand0=gamma_sb[0:P_MAX, nl.ds(gamma_tile_index, 1)],
                        engine=nisa.scalar_engine,
                    )
                else:
                    nisa.tensor_copy(
                        dst=dst_ap,
                        src=transpose_psum[bank_base + h_sub_idx][0:P_MAX, 0:s_tile_sz],
                    )

    for rem_idx in nl.affine_range(remainder):
        h_tile_idx = num_groups * TILES_PER_GROUP + rem_idx
        for h_sub_idx in nl.affine_range(_H_PACK):
            src_offset = (h_sub_idx * num_512_tiles + h_tile_idx) * P_MAX
            nisa.nc_transpose(
                data=input_sb[0:s_tile_sz, src_offset : src_offset + P_MAX],
                dst=transpose_psum[h_sub_idx][0:P_MAX, 0:s_tile_sz],
            )
        for h_sub_idx in nl.affine_range(_H_PACK):
            dst_ap = transposed_sb.ap(
                pattern=[[num_512_tiles * s_tile_sz * _H_PACK, P_MAX], [_H_PACK, s_tile_sz]],
                offset=h_tile_idx * s_tile_sz * _H_PACK + h_sub_idx,
            )
            if gamma_sb != None:
                gamma_tile_index = h_tile_idx * _H_PACK + h_sub_idx
                nisa.tensor_scalar(
                    dst=dst_ap,
                    data=transpose_psum[h_sub_idx][0:P_MAX, 0:s_tile_sz],
                    op0=nl.multiply,
                    operand0=gamma_sb[0:P_MAX, nl.ds(gamma_tile_index, 1)],
                    engine=nisa.scalar_engine,
                )
            else:
                nisa.tensor_copy(dst=dst_ap, src=transpose_psum[h_sub_idx][0:P_MAX, 0:s_tile_sz])

    return transposed_sb


def _compute_rms_norm_scale(
    input_sb: nl.NkiTensor,
    zero_bias_sb: nl.NkiTensor,
    norm_eps_sb: nl.NkiTensor,
    s_tile_sz: int,
    hidden_dim: int,
    sbm: SbufManager,
    name: str = "rms",
) -> nl.NkiTensor:
    """Compute rsqrt(mean(x²) + eps) scale factor without applying it.

    Returns the [s_tile_sz, 1] scale factor to be fused into a later step.

    Args:
        input_sb (nl.NkiTensor): [s_tile_sz, hidden_dim], Input tensor in SBUF.
        zero_bias_sb (nl.NkiTensor): [P_MAX, 1], Pre-zeroed bias tensor for activation_reduce.
        norm_eps_sb (nl.NkiTensor): [P_MAX, 1], Pre-filled epsilon tensor.
        s_tile_sz (int): Number of active sequence positions.
        hidden_dim (int): Hidden dimension size.
        sbm (SbufManager): SBUF memory manager.
        name (str): Name prefix for allocated buffers.

    Returns:
        nl.NkiTensor: [P_MAX, 1] float32, rsqrt scale factor in SBUF.
    """
    P_MAX = nl.tile_size.pmax

    square_sum_sb = sbm.alloc_stack((P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf, name=f"{name}_square_sum")
    act_temp_sb = sbm.alloc_stack((P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf, name=f"{name}_act_temp")

    nisa.activation_reduce(
        dst=act_temp_sb.ap(pattern=[[1, s_tile_sz], [0, hidden_dim]]),
        op=nl.square,
        data=input_sb[0:s_tile_sz, 0:hidden_dim],
        reduce_op=nl.add,
        reduce_res=square_sum_sb[0:s_tile_sz, 0:1],
        bias=zero_bias_sb[0:s_tile_sz, 0:1],
    )

    nisa.activation(
        dst=square_sum_sb[0:s_tile_sz, 0:1],
        op=nl.rsqrt,
        data=square_sum_sb[0:s_tile_sz, 0:1],
        bias=norm_eps_sb[0:s_tile_sz, 0:1],
        scale=float(1.0 / hidden_dim),
    )

    return square_sum_sb


def _unswizzle_lora_cols(
    src_sb: nl.NkiTensor,
    s_tile_sz: int,
    lora_dim: int,
    sbm,
    name: str = "unswizzle_lora",
) -> nl.NkiTensor:
    """Reorder pre-swizzled latent columns [s, lora] -> natural column order.

    The first projection's weights are column-swizzled from
    ``[n512, 128, 4]`` to ``[4, n512, 128]`` order (``_swizzle_mla_cols`` in the
    test harness / ``_load_mx_weights``), so the matmul output ``src_sb`` has
    swizzled latent columns: physical column ``j`` holds the natural element
    ``sw[j]`` where ``sw = arange(lora).reshape(n512,128,4).transpose(2,0,1)``.

    Decode (and the vLLM/HF KV-cache contract) expects *natural* column order
    ``m = t*512 + p*4 + q``. The inverse gather is
    ``natural[m] = src[ q*(n512*128) + t*128 + p ]`` (verified numerically), so
    for each of the _H_PACK=4 ``q`` sub-lanes we copy a strided slab:

        dst natural cols  {t*512 + p*4 + q : t in [0,n512), p in [0,128)}
        src swizzled cols {q*n512*128 + t*128 + p}

    which is one ``tensor_copy`` per ``q`` (4 total), each a 2-free-axis AP.

    Returns a fresh ``[P_MAX, lora_dim]`` SBUF buffer in natural column order.
    """
    P_MAX = nl.tile_size.pmax
    n512 = lora_dim // (P_MAX * _H_PACK)
    out_sb = sbm.alloc_stack((P_MAX, lora_dim), dtype=src_sb.dtype, buffer=nl.sbuf, name=name)
    """
    AP axis 0 is the partition (token) axis: stride = allocated row pitch
    (lora_dim), count = s_tile_sz. The two free axes (t, p) describe the
    column gather; q is folded into the per-iteration base offset.
    """
    for sub_lane_idx in range(_H_PACK):
        # dst: natural cols (t outer stride 512, p inner stride 4), base offset sub_lane_idx.
        # src: swizzled cols (t outer stride 128, p inner stride 1), base sub_lane_idx*n512*128.
        nisa.tensor_copy(
            dst=out_sb.ap(
                pattern=[[lora_dim, s_tile_sz], [P_MAX * _H_PACK, n512], [_H_PACK, P_MAX]],
                offset=sub_lane_idx,
            ),
            src=src_sb.ap(
                pattern=[[lora_dim, s_tile_sz], [P_MAX, n512], [1, P_MAX]],
                offset=sub_lane_idx * n512 * P_MAX,
            ),
            engine=nisa.scalar_engine if sub_lane_idx % 2 == 0 else nisa.vector_engine,
        )
    return out_sb


def _apply_rms_norm_inplace(
    input_sb: nl.NkiTensor,
    zero_bias_sb: nl.NkiTensor,
    norm_eps_sb: nl.NkiTensor,
    s_tile_sz: int,
    hidden_dim: int,
    sbm: SbufManager,
    name: str = "rms",
) -> None:
    """Apply RMSNorm in-place: ``x = x / sqrt(mean(x²) + eps)``.

    Computes the reciprocal RMS and multiplies the input in-place. Gamma is
    NOT applied here; callers that fuse gamma into the subsequent
    ``_transpose_preswizzled_for_mx`` should pass ``gamma_sb`` to that
    function instead.

    Args:
        input_sb: [s_tile_sz, hidden_dim], modified in-place.
        zero_bias_sb: [P_MAX, 1] pre-zeroed bias for ``activation_reduce``.
        norm_eps_sb: [P_MAX, 1] pre-filled epsilon tensor.
        s_tile_sz: Number of active sequence positions.
        hidden_dim: Hidden dimension size.
        sbm: SBUF memory manager.
        name: Name prefix for allocated buffers.
    """
    square_sum_sb = _compute_rms_norm_scale(input_sb, zero_bias_sb, norm_eps_sb, s_tile_sz, hidden_dim, sbm, name=name)

    nisa.tensor_scalar(
        dst=input_sb[0:s_tile_sz, 0:hidden_dim],
        data=input_sb[0:s_tile_sz, 0:hidden_dim],
        op0=nl.multiply,
        operand0=square_sum_sb[0:s_tile_sz, 0:1],
        engine=nisa.scalar_engine,
    )


def _apply_rope_to_tensor(
    x_sb: nl.NkiTensor,
    cos_sb: nl.NkiTensor,
    sin_sb: nl.NkiTensor,
    s_tile_sz: int,
    rope_dim: int,
    sbm: SbufManager,
    name: str = "rope",
) -> nl.NkiTensor:
    """
    Apply Rotary Position Embedding (RoPE) to a tensor, returning a new buffer.

    Computes: output = [x1, x2] * cos + [-x2, x1] * sin
    where x1 and x2 are the first and second halves of the input along rope_dim.

    Args:
        x_sb (nl.NkiTensor): [s_tile_sz, rope_dim], Input tensor in SBUF.
        cos_sb (nl.NkiTensor): [s_tile_sz, rope_dim], Cosine frequencies in SBUF.
        sin_sb (nl.NkiTensor): [s_tile_sz, rope_dim // 2], Sine frequencies in SBUF.
        s_tile_sz (int): Number of active sequence positions.
        rope_dim (int): RoPE dimension (must be even).
        sbm (SbufManager): SBUF memory manager.
        name (str): Name prefix for allocated buffers.

    Returns:
        nl.NkiTensor: [s_tile_sz, rope_dim], New tensor with RoPE applied in SBUF.
    """
    half_dim = rope_dim // 2

    output_sb = sbm.alloc_stack((s_tile_sz, rope_dim), dtype=nl.bfloat16, buffer=nl.sbuf, name=f"{name}_out")
    temp_sb = sbm.alloc_stack((s_tile_sz, rope_dim), dtype=nl.bfloat16, buffer=nl.sbuf, name=f"{name}_tmp")

    nisa.tensor_tensor(
        dst=output_sb[0:s_tile_sz, 0:rope_dim],
        data1=x_sb[0:s_tile_sz, 0:rope_dim],
        data2=cos_sb[0:s_tile_sz, 0:rope_dim],
        op=nl.multiply,
    )

    nisa.tensor_tensor(
        dst=temp_sb[0:s_tile_sz, 0:half_dim],
        data1=x_sb[0:s_tile_sz, half_dim:rope_dim],
        data2=sin_sb[0:s_tile_sz, 0:half_dim],
        op=nl.multiply,
    )
    nisa.tensor_scalar(
        dst=temp_sb[0:s_tile_sz, 0:half_dim],
        data=temp_sb[0:s_tile_sz, 0:half_dim],
        op0=nl.multiply,
        operand0=-1.0,
        engine=nisa.scalar_engine,
    )

    nisa.tensor_tensor(
        dst=temp_sb[0:s_tile_sz, half_dim:rope_dim],
        data1=x_sb[0:s_tile_sz, 0:half_dim],
        data2=sin_sb[0:s_tile_sz, 0:half_dim],
        op=nl.multiply,
    )

    nisa.tensor_tensor(
        dst=output_sb[0:s_tile_sz, 0:rope_dim],
        data1=output_sb[0:s_tile_sz, 0:rope_dim],
        data2=temp_sb[0:s_tile_sz, 0:rope_dim],
        op=nl.add,
    )

    return output_sb


def _apply_rope_inplace(
    x_sb: nl.NkiTensor,
    cos_sb: nl.NkiTensor,
    sin_sb: nl.NkiTensor,
    s_tile_sz: int,
    rope_dim: int,
) -> None:
    """
    Apply RoPE in-place using scratch space in the same buffer.

    Computes: x = [x1, x2] * cos + [-x2, x1] * sin, where x1 and x2 are the
    first and second halves of the input along rope_dim. Uses
    ``x_sb[rope_dim:rope_dim*2]`` as scratch space.

    Args:
        x_sb (nl.NkiTensor): [s_tile_sz, rope_dim * 2], Input modified in-place.
            First rope_dim elements are the input; second rope_dim are scratch.
        cos_sb (nl.NkiTensor): [s_tile_sz, rope_dim], Cosine frequencies.
        sin_sb (nl.NkiTensor): [s_tile_sz, rope_dim // 2], Sine frequencies.
        s_tile_sz (int): Number of active sequence positions.
        rope_dim (int): RoPE dimension (must be even).
    """
    half_dim = rope_dim // 2

    nisa.tensor_tensor(
        dst=x_sb[0:s_tile_sz, rope_dim : rope_dim + half_dim],
        data1=x_sb[0:s_tile_sz, half_dim:rope_dim],
        data2=sin_sb[0:s_tile_sz, 0:half_dim],
        op=nl.multiply,
    )
    nisa.tensor_scalar(
        dst=x_sb[0:s_tile_sz, rope_dim : rope_dim + half_dim],
        data=x_sb[0:s_tile_sz, rope_dim : rope_dim + half_dim],
        op0=nl.multiply,
        operand0=-1.0,
    )

    nisa.tensor_tensor(
        dst=x_sb[0:s_tile_sz, rope_dim + half_dim : rope_dim * 2],
        data1=x_sb[0:s_tile_sz, 0:half_dim],
        data2=sin_sb[0:s_tile_sz, 0:half_dim],
        op=nl.multiply,
    )

    nisa.tensor_tensor(
        dst=x_sb[0:s_tile_sz, 0:rope_dim],
        data1=x_sb[0:s_tile_sz, 0:rope_dim],
        data2=cos_sb[0:s_tile_sz, 0:rope_dim],
        op=nl.multiply,
    )

    nisa.tensor_tensor(
        dst=x_sb[0:s_tile_sz, 0:rope_dim],
        data1=x_sb[0:s_tile_sz, 0:rope_dim],
        data2=x_sb[0:s_tile_sz, rope_dim : rope_dim * 2],
        op=nl.add,
    )


def _apply_rope_to_tensor_interleaved(
    x_sb: nl.NkiTensor,
    cos_sb: nl.NkiTensor,
    sin_sb: nl.NkiTensor,
    s_tile_sz: int,
    rope_dim: int,
    sbm: SbufManager,
    name: str = "rope_il",
) -> nl.NkiTensor:
    """INTERLEAVED (adjacent-pair / complex) RoPE returning a new buffer.

    Matches DeepSeek ``apply_rotary_emb(..., interleaved=True)``: pair j = columns
    (2j, 2j+1), out[2j]=x[2j]*cos_j - x[2j+1]*sin_j, out[2j+1]=x[2j]*sin_j + x[2j+1]*cos_j.
    Interleaved sibling of ``_apply_rope_to_tensor`` (half-split). cos_sb/sin_sb carry
    the rope_dim/2 frequencies theta_j in their first half. Allocates a fresh output.
    """
    half_dim = rope_dim // 2
    output_sb = sbm.alloc_stack((s_tile_sz, rope_dim), dtype=nl.bfloat16, buffer=nl.sbuf, name=f"{name}_out")
    tmp_sb = sbm.alloc_stack((s_tile_sz, rope_dim), dtype=nl.bfloat16, buffer=nl.sbuf, name=f"{name}_tmp")

    # Strided even/odd APs over the rope_dim columns (stride 2), on x (source), tmp, and out.
    x_even = x_sb.ap(pattern=[[rope_dim, s_tile_sz], [2, half_dim]], offset=0)
    x_odd = x_sb.ap(pattern=[[rope_dim, s_tile_sz], [2, half_dim]], offset=1)
    t_even = tmp_sb.ap(pattern=[[rope_dim, s_tile_sz], [2, half_dim]], offset=0)
    t_odd = tmp_sb.ap(pattern=[[rope_dim, s_tile_sz], [2, half_dim]], offset=1)
    o_even = output_sb.ap(pattern=[[rope_dim, s_tile_sz], [2, half_dim]], offset=0)
    o_odd = output_sb.ap(pattern=[[rope_dim, s_tile_sz], [2, half_dim]], offset=1)

    # tmp = cross terms: t_even = x_odd*sin (subtracted), t_odd = x_even*sin (added).
    nisa.tensor_tensor(dst=t_even, data1=x_odd, data2=sin_sb[0:s_tile_sz, 0:half_dim], op=nl.multiply)
    nisa.tensor_tensor(dst=t_odd, data1=x_even, data2=sin_sb[0:s_tile_sz, 0:half_dim], op=nl.multiply)
    # out = cos terms then combine: o_even = x_even*cos - t_even ; o_odd = x_odd*cos + t_odd.
    nisa.tensor_tensor(dst=o_even, data1=x_even, data2=cos_sb[0:s_tile_sz, 0:half_dim], op=nl.multiply)
    nisa.tensor_tensor(dst=o_odd, data1=x_odd, data2=cos_sb[0:s_tile_sz, 0:half_dim], op=nl.multiply)
    nisa.tensor_tensor(dst=o_even, data1=o_even, data2=t_even, op=nl.subtract)
    nisa.tensor_tensor(dst=o_odd, data1=o_odd, data2=t_odd, op=nl.add)
    return output_sb


def _apply_rope_inplace_interleaved(
    x_sb: nl.NkiTensor,
    cos_sb: nl.NkiTensor,
    sin_sb: nl.NkiTensor,
    s_tile_sz: int,
    rope_dim: int,
) -> None:
    """Apply RoPE in-place, INTERLEAVED (adjacent-pair / complex) layout.

    This matches DeepSeek ``apply_rotary_emb(..., interleaved=True)`` used by the
    main MLA attention: adjacent elements form a rotary pair. For pair j
    (columns 2j, 2j+1) with angle theta_j:

        out[2j]   = x[2j]*cos_j - x[2j+1]*sin_j
        out[2j+1] = x[2j]*sin_j + x[2j+1]*cos_j

    vs the half-split ``_apply_rope_inplace`` which pairs (j, j+rope_dim/2). The
    cos/sin caches carry the SAME rope_dim/2 frequencies theta_j in their first
    half (cos_sb[:, j], sin_sb[:, j]); only the element PAIRING differs, so this
    reads the even/odd columns via stride-2 APs. Uses x_sb[rope_dim:rope_dim*2]
    as scratch (same contract as ``_apply_rope_inplace``). Same op count / engines
    as the half-split path (strided AP, not extra work).
    """
    half_dim = rope_dim // 2
    """
    Strided APs (NKI forbids inner def, so build each ap() inline). Free pattern
    [[2*rope_dim, s], [2, half]]: per-row stride 2*rope_dim spans the live rope_dim
    region + the rope_dim scratch region; offset selects the even (2j) / odd (2j+1)
    lane. Lanes: x_even=off0, x_odd=off1, scratch even=off rope_dim, odd=off rope_dim+1.
    """
    row = 2 * rope_dim
    x_even = x_sb.ap(pattern=[[row, s_tile_sz], [2, half_dim]], offset=0)
    x_odd = x_sb.ap(pattern=[[row, s_tile_sz], [2, half_dim]], offset=1)
    sc_even = x_sb.ap(pattern=[[row, s_tile_sz], [2, half_dim]], offset=rope_dim)
    sc_odd = x_sb.ap(pattern=[[row, s_tile_sz], [2, half_dim]], offset=rope_dim + 1)

    """
    Stage the two cross terms into scratch FIRST (read x before overwriting it):
      sc_even = x_odd  * sin   (the term subtracted from out_even)
      sc_odd  = x_even * sin   (the term added to out_odd)
    """
    nisa.tensor_tensor(dst=sc_even, data1=x_odd, data2=sin_sb[0:s_tile_sz, 0:half_dim], op=nl.multiply)
    nisa.tensor_tensor(dst=sc_odd, data1=x_even, data2=sin_sb[0:s_tile_sz, 0:half_dim], op=nl.multiply)

    # In place: x_even <- x_even*cos ; x_odd <- x_odd*cos  (cos_j on both lanes).
    nisa.tensor_tensor(dst=x_even, data1=x_even, data2=cos_sb[0:s_tile_sz, 0:half_dim], op=nl.multiply)
    nisa.tensor_tensor(dst=x_odd, data1=x_odd, data2=cos_sb[0:s_tile_sz, 0:half_dim], op=nl.multiply)

    # out_even = x_even*cos - x_odd*sin ; out_odd = x_odd*cos + x_even*sin.
    nisa.tensor_tensor(dst=x_even, data1=x_even, data2=sc_even, op=nl.subtract)
    nisa.tensor_tensor(dst=x_odd, data1=x_odd, data2=sc_odd, op=nl.add)


def _apply_rope_inplace_interleaved_grouped(
    x_grp_sb: nl.NkiTensor,
    cos_sb: nl.NkiTensor,
    sin_sb: nl.NkiTensor,
    scratch_sb: nl.NkiTensor,
    s_tile_sz: int,
    rope_dim: int,
    n_rep: int,
) -> None:
    """Batched INTERLEAVED RoPE over ``n_rep`` head-contiguous q_pe slots at once.

    ``x_grp_sb`` holds ``[s_tile, n_rep * rope_dim]`` (head r at columns
    ``[r*rope_dim, (r+1)*rope_dim)``); the SAME cos_sb/sin_sb (``[s_tile, rope_dim/2]``)
    apply to every head, so this does the 6 tensor_tensor RoPE ops ONCE over all n_rep
    heads instead of per head — collapsing 6*n_rep tiny ops into 6 wide ops.

    cos/sin are broadcast across the head axis via a stride-0 middle AP dim; ``scratch_sb``
    ([s_tile, n_rep*rope_dim] bf16) holds the two cross-terms (the per-head in-place scratch
    trick doesn't apply to a head-packed buffer). Matches _apply_rope_inplace_interleaved.
    """
    half = rope_dim // 2
    row = n_rep * rope_dim  # per-partition free width of the group buffer
    # even/odd lanes of each head: middle dim = head (stride rope_dim), inner = pair (stride 2).
    x_even = x_grp_sb.ap(pattern=[[row, s_tile_sz], [rope_dim, n_rep], [2, half]], offset=0)
    x_odd = x_grp_sb.ap(pattern=[[row, s_tile_sz], [rope_dim, n_rep], [2, half]], offset=1)
    sc_even = scratch_sb.ap(pattern=[[row, s_tile_sz], [rope_dim, n_rep], [2, half]], offset=0)
    sc_odd = scratch_sb.ap(pattern=[[row, s_tile_sz], [rope_dim, n_rep], [2, half]], offset=1)
    # cos/sin broadcast across the head axis (middle stride 0).
    cos_b = cos_sb.ap(pattern=[[cos_sb.shape[1], s_tile_sz], [0, n_rep], [1, half]], offset=0)
    sin_b = sin_sb.ap(pattern=[[sin_sb.shape[1], s_tile_sz], [0, n_rep], [1, half]], offset=0)

    nisa.tensor_tensor(dst=sc_even, data1=x_odd, data2=sin_b, op=nl.multiply)
    nisa.tensor_tensor(dst=sc_odd, data1=x_even, data2=sin_b, op=nl.multiply)
    nisa.tensor_tensor(dst=x_even, data1=x_even, data2=cos_b, op=nl.multiply)
    nisa.tensor_tensor(dst=x_odd, data1=x_odd, data2=cos_b, op=nl.multiply)
    nisa.tensor_tensor(dst=x_even, data1=x_even, data2=sc_even, op=nl.subtract)
    nisa.tensor_tensor(dst=x_odd, data1=x_odd, data2=sc_odd, op=nl.add)
