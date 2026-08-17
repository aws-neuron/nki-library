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

"""MX (block-32 fp8) projection sub-kernels for the Sparse Attention Indexer.

Q/K/W projections via nc_matmul_mx:
  * Activation quantized at runtime via quantize_mx (block 32, fp8_e4m3fn_x4).
  * Weights pre-quantized offline into fp8_e4m3fn_x4 + uint8 e8m0 hardware
    quadrant scales.
  * weights_proj stays bf16 (checkpoint format).
"""

import nki.isa as nisa
import nki.isa.constants as nisa_constants
import nki.language as nl

from ...core.utils.allocator import SbufManager
from ...core.utils.kernel_assert import kernel_assert
from ...core.utils.tensor_view import TensorView
from .sparse_attention_indexer_utils import P_MAX

# MX block constants (per OCP spec: 32 elements/scale, 4-packed FP8 along free dim)
H_PACK = 4
"""Number of FP8 elements packed into one fp8_x4 word along the contraction dim."""

MX_BLOCK = 32
"""Microscaling block size — 1 uint8 e8m0 scale per 32 elements of the contraction dim."""

PSUM_FMAX = 512
"""Max free dim of one PSUM bank for nc_matmul_mx with fp32 dst."""

# Hardware quadrant layout for MX scales: 32 partitions per quadrant, 4 valid scale rows per quadrant.
SCALE_QUADRANT_SIZE = 32
SCALES_PER_QUADRANT = 4


def _quantize_activation_for_mx(
    sbm: SbufManager,
    src_hbm: nl.ndarray,
    batch_start: int,
    S: int,
    K: int,
    hidden_qtz_sb: nl.ndarray,
    hidden_scale_sb: nl.ndarray,
    num_K_tiles: int,
) -> None:
    """Quantize a [S, K] activation tile from HBM to MX-FP8.

    Layout convention (matches qkv_cte non-swizzled MX path):
        partition dim of hidden_qtz = K (with H_PACK=4 packing along the x4 word)
        free dim of hidden_qtz = S (one column per S value, addressable)

    K must be a multiple of P_MAX * H_PACK (= 512).
    """
    H_padded = num_K_tiles * P_MAX * H_PACK  # == K

    """
    Load activation [S, K] into [P_MAX, K] SBUF buffer in bf16 directly —
    the DMA engine performs the f32→bf16 cast on the wire, eliminating a
    separate tensor_copy cast on DVE/Scalar. (Only :S partitions filled —
    rows [S:P_MAX] hold garbage and are not read by the matmul.)
    """
    s_load_buf = sbm.alloc_stack((P_MAX, K), nl.bfloat16)
    nisa.dma_copy(
        dst=s_load_buf[:S, :K],
        src=src_hbm[batch_start : batch_start + S, :K],
        dge_mode=nisa_constants.dge_mode.swdge,
    )

    """
    Transpose + 4-way swizzle: [P_MAX_S, K] -> [P_MAX_K, num_K_tiles, P_MAX_S * H_PACK]
    with each 4-element group along free dim being the H_PACK sub-elements at a
    fixed S, so that quantize_mx's 32-partition × 4-free MX block size matches
    one S × 128 contraction elements.
    """
    transposed_sb = sbm.alloc_stack((P_MAX, num_K_tiles, P_MAX * H_PACK), nl.bfloat16)
    transpose_psum = []
    for _ in range(H_PACK):
        transpose_psum.append(nl.ndarray((P_MAX, P_MAX), dtype=nl.bfloat16, buffer=nl.psum))

    for k_tile in nl.affine_range(num_K_tiles):
        for h_sub in nl.affine_range(H_PACK):
            src_h_base = k_tile * P_MAX * H_PACK + h_sub
            nisa.nc_transpose(
                data=s_load_buf.ap(pattern=[[H_padded, P_MAX], [H_PACK, P_MAX]], offset=src_h_base),
                dst=transpose_psum[h_sub][0:P_MAX, 0:P_MAX],
            )
        for h_sub in nl.affine_range(H_PACK):
            dst_ap = transposed_sb.ap(
                pattern=[
                    [num_K_tiles * P_MAX * H_PACK, P_MAX],
                    [H_PACK, P_MAX],
                ],
                offset=k_tile * P_MAX * H_PACK + h_sub,
            )
            nisa.tensor_copy(
                dst=dst_ap,
                src=transpose_psum[h_sub][0:P_MAX, 0:P_MAX],
                engine=nisa.engine.scalar,
            )

    nisa.quantize_mx(
        src=transposed_sb[0:P_MAX, 0:num_K_tiles, 0 : P_MAX * H_PACK],
        dst=hidden_qtz_sb[0:P_MAX, 0:num_K_tiles, 0:P_MAX],
        dst_scale=hidden_scale_sb[0:P_MAX, 0:num_K_tiles, 0:P_MAX],
    )


def _load_mx_weight_scale(
    scale_hbm: nl.ndarray,
    scale_sb: nl.ndarray,
    num_K_tiles: int,
    free_dim: int,
    free_offset: int,
    free_extent: int,
) -> None:
    """Load a [q_lora_rank // 32, total_out] uint8 e8m0 scale into hardware quadrant layout.

    Each K-tile spans K_TILE=512 elements / MX_BLOCK=32 = 16 contiguous HBM scale rows.
    Hardware layout splits those 16 rows into 4 quadrants of 4 rows each: rows 0-3 of
    HBM map to partitions 0-3, rows 4-7 to partitions 32-35, rows 8-11 to 64-67, rows
    12-15 to 96-99. Other partitions are zero-padded.

    Coalesce all K-tiles per quadrant into a single 3-D AP DMA. Previous
    code issued ``num_K_tiles * NUM_QUADRANTS`` (e.g. 56 for K-proj at
    v3_long) tiny DMAs of ~512 bytes each. Now one DMA per quadrant
    (4 total) covers all K-tiles, amortizing descriptor overhead.
    """
    SCALE_ROWS_PER_K_TILE = (P_MAX * H_PACK) // MX_BLOCK  # = 16
    NUM_QUADRANTS = P_MAX // SCALE_QUADRANT_SIZE  # = 4
    nisa.memset(dst=scale_sb, value=0, engine=nisa.engine.gpsimd)
    for quad_idx in range(NUM_QUADRANTS):
        # 3-D AP: partition (SCALES_PER_QUADRANT rows) × k_tile × free.
        nisa.dma_copy(
            dst=scale_sb[
                nl.ds(quad_idx * SCALE_QUADRANT_SIZE, SCALES_PER_QUADRANT),
                0:num_K_tiles,
                0:free_extent,
            ],
            src=scale_hbm.ap(
                pattern=[
                    [free_dim, SCALES_PER_QUADRANT],
                    [SCALE_ROWS_PER_K_TILE * free_dim, num_K_tiles],
                    [1, free_extent],
                ],
                offset=quad_idx * SCALES_PER_QUADRANT * free_dim + free_offset,
                dtype=nl.uint8,
            ),
            dge_mode=nisa_constants.dge_mode.hwdge,
        )


# Q projection (MX): qr [B*S, q_lora_rank] @ wq_b_fp8 [q_lora_rank // 4, total_out].
def q_projection_mx(
    sbm: SbufManager,
    qr: nl.ndarray,
    batch_start: int,
    S: int,
    wq_b_fp8: nl.ndarray,
    wq_b_scale: nl.ndarray,
    q_lora_rank: int,
    n_heads: int,
    head_dim: int,
    num_shards: int = 1,
    shard_id: int = 0,
    q_out_hbm: nl.ndarray = None,
    qr_qtz_hbm: nl.ndarray = None,
    qr_scale_hbm: nl.ndarray = None,
) -> nl.ndarray:
    """MX Q projection: nc_matmul_mx(quantize_mx(qr), wq_b_fp8) -> [S, total_out].

    Args:
        sbm (SbufManager): SBUF stack allocator used for scratch buffers.
        qr (nl.ndarray): [B*S, q_lora_rank] bf16/f32 HBM — activation, quantized at runtime.
        batch_start (int): Row offset into qr for this batch.
        S (int): Number of activation rows (sequence positions) for this batch.
        wq_b_fp8 (nl.ndarray): [q_lora_rank // 4, n_heads*head_dim] fp8_e4m3fn_x4 HBM — pre-quantized weights.
        wq_b_scale (nl.ndarray): [q_lora_rank // 32, n_heads*head_dim] uint8 HBM — block-32 e8m0 scales.
        q_lora_rank (int): Contraction dimension of the Q projection.
        n_heads (int): Number of attention heads.
        head_dim (int): Per-head dimension; total_out = n_heads * head_dim.
        num_shards (int): Number of LNC shards splitting the output-tile range.
        shard_id (int): This shard's index within num_shards.
        q_out_hbm (nl.ndarray): [S, total_out] HBM output buffer; sliced per output shard.
        qr_qtz_hbm (nl.ndarray): optional PRE-quantized qr from an upstream stage (the
            fused qkv stage already produces the transposed+q_normed(gamma)+MX-quantized qr
            in the exact [num_s_tiles, P_MAX, num_K_tiles, P_MAX] fp8x4 (viewed uint32)
            layout this function would otherwise recompute). When given, skip
            _quantize_activation_for_mx and DMA-load the per-s-tile block directly — saves a
            redundant transpose+norm+quantize and reuses qkv's already-gamma-normed qr
            (avoids the gamma-less pitfall of exporting qr raw).
        qr_scale_hbm (nl.ndarray): uint8 block-32 e8m0 scales paired with qr_qtz_hbm.

    Returns:
        nl.ndarray: q_out_hbm, with this shard's output-tile columns written.
    """
    total_out = n_heads * head_dim
    K_TILE = P_MAX * H_PACK  # 512: contraction dim consumed per nc_matmul_mx instr
    num_K_tiles = q_lora_rank // K_TILE

    num_out_tiles_all = (total_out + PSUM_FMAX - 1) // PSUM_FMAX
    eff_num_shards = min(num_shards, num_out_tiles_all)
    if shard_id >= eff_num_shards:
        return q_out_hbm
    tiles_per_shard = (num_out_tiles_all + eff_num_shards - 1) // eff_num_shards
    my_tile_start = shard_id * tiles_per_shard
    my_tile_end = min(my_tile_start + tiles_per_shard, num_out_tiles_all)
    num_out_tiles = my_tile_end - my_tile_start
    num_s_tiles = (S + P_MAX - 1) // P_MAX
    out_col_start = my_tile_start * PSUM_FMAX
    out_col_end = min(my_tile_end * PSUM_FMAX, total_out)
    out_extent = out_col_end - out_col_start

    sbm.open_scope(name="q_proj_mx")
    q_row_sb = sbm.alloc_stack((P_MAX, out_extent), nl.float32)

    """
    Pre-load wq_b_fp8 [P_MAX P, num_K_tiles F, out_extent F] in ONE DMA
    via 3-D AP. Per-K-tile loop loaded num_K_tiles tiny DMAs (each
    [128 P, out_extent F]); a single 3-D DMA amortizes descriptor
    overhead and lets the HW DGE coalesce.
    """
    wq_b_sb = sbm.alloc_stack((P_MAX, num_K_tiles, out_extent), nl.float8_e4m3fn_x4, buffer=nl.sbuf)
    nisa.dma_copy(
        dst=wq_b_sb[0:P_MAX, 0:num_K_tiles, 0:out_extent],
        src=wq_b_fp8.ap(
            pattern=[
                [total_out, P_MAX],
                [P_MAX * total_out, num_K_tiles],
                [1, out_extent],
            ],
            offset=out_col_start,
            dtype=nl.float8_e4m3fn_x4,
        ),
        dge_mode=nisa_constants.dge_mode.hwdge,
    )

    # Pre-load wq_b_scale into hardware quadrant layout [P_MAX P, num_K_tiles F, out_extent F].
    wq_b_scale_sb = sbm.alloc_stack((P_MAX, num_K_tiles, out_extent), nl.uint8, buffer=nl.sbuf)
    _load_mx_weight_scale(
        wq_b_scale, wq_b_scale_sb, num_K_tiles, free_dim=total_out, free_offset=out_col_start, free_extent=out_extent
    )

    for s_tile_idx in nl.sequential_range(num_s_tiles):
        s_start = s_tile_idx * P_MAX
        s_size = min(P_MAX, S - s_start)

        sbm.open_scope(name="quantize_qr")
        # Quantize qr -> [P_MAX P, num_K_tiles F, P_MAX F] fp8_x4, valid free [:s_size].
        hidden_qtz_sb = sbm.alloc_stack((P_MAX, num_K_tiles, P_MAX), nl.float8_e4m3fn_x4, buffer=nl.sbuf, align=32)
        hidden_scale_sb = sbm.alloc_stack((P_MAX, num_K_tiles, P_MAX), nl.uint8, buffer=nl.sbuf)
        if qr_qtz_hbm is not None:
            _qr_tile = (batch_start + s_start) // P_MAX
            _hidden_u32 = TensorView(hidden_qtz_sb).reinterpret_cast(nl.uint32).get_view()
            nisa.dma_copy(dst=_hidden_u32[0:P_MAX, 0:num_K_tiles, 0:P_MAX], src=qr_qtz_hbm[_qr_tile])
            nisa.dma_copy(dst=hidden_scale_sb[0:P_MAX, 0:num_K_tiles, 0:P_MAX], src=qr_scale_hbm[_qr_tile])
        else:
            _quantize_activation_for_mx(
                sbm,
                qr,
                batch_start + s_start,
                s_size,
                q_lora_rank,
                hidden_qtz_sb,
                hidden_scale_sb,
                num_K_tiles,
            )

        # PSUM banks per output tile (rotate to allow concurrent matmuls).
        NUM_PSUM_BANKS = 8
        n_banks = min(num_out_tiles, NUM_PSUM_BANKS)
        q_psum_banks = []
        for _ in range(n_banks):
            q_psum_banks.append(nl.ndarray((P_MAX, PSUM_FMAX), dtype=nl.float32, buffer=nl.psum))

        for out_idx in nl.affine_range(num_out_tiles):
            local_out_off = out_idx * PSUM_FMAX
            out_size = min(PSUM_FMAX, out_extent - local_out_off)
            q_psum = q_psum_banks[out_idx % n_banks]

            for k_tile in nl.affine_range(num_K_tiles):
                """
                accumulate explicitly per k_tile: first tile overwrites,
                subsequent tiles accumulate into the bank. With num_K_tiles=1
                this is just a single overwrite, but be explicit anyway —
                without it, when q_psum_banks rotates (out_idx wraps past
                NUM_PSUM_BANKS=8) the auto-detect treats the bank as
                "previously written" and accumulates into stale contents
                from the prior owner of the bank.
                """
                nisa.nc_matmul_mx(
                    dst=q_psum[:s_size, :out_size],
                    stationary=hidden_qtz_sb[:P_MAX, k_tile, :s_size],
                    moving=wq_b_sb[:P_MAX, k_tile, local_out_off : local_out_off + out_size],
                    stationary_scale=hidden_scale_sb[:P_MAX, k_tile, :s_size],
                    moving_scale=wq_b_scale_sb[:P_MAX, k_tile, local_out_off : local_out_off + out_size],
                    accumulate=(k_tile != 0),
                )

            nisa.tensor_copy(
                dst=q_row_sb[:s_size, local_out_off : local_out_off + out_size], src=q_psum[:s_size, :out_size]
            )

        sbm.close_scope()

        nisa.dma_copy(
            dst=q_out_hbm[s_start : s_start + s_size, out_col_start:out_col_end],
            src=q_row_sb[:s_size, :out_extent],
            dge_mode=nisa_constants.dge_mode.swdge,
        )

    sbm.close_scope()
    return q_out_hbm


# K + W projection: wk_fp8 used for K (MX); weights_proj stays bf16.
def load_wk_mx_weights(
    sbm: SbufManager,
    wk_fp8: nl.ndarray,
    wk_scale: nl.ndarray,
    dim: int,
    head_dim: int,
) -> tuple[nl.ndarray, nl.ndarray]:
    """Pre-load wk's fp8_x4 weights and uint8 scale into SBUF.

    The MX-K path is invoked once per S-tile but the weights are S-tile
    invariant, so hoisting the wk load out of the S-tile loop saves
    ``num_S_tiles - 1`` redundant DMA passes per batch (252 KB / pass at V3
    scale, ~3.5 us / pass at 573 GB/s).

    Args:
        sbm (SbufManager): SBUF stack allocator used for the hoisted buffers.
        wk_fp8 (nl.ndarray): [dim // 4, head_dim] fp8_e4m3fn_x4 HBM — pre-quantized K weights.
        wk_scale (nl.ndarray): [dim // 32, head_dim] uint8 HBM — block-32 e8m0 scales.
        dim (int): Contraction dimension of the K projection.
        head_dim (int): Per-head dimension (output width of the K projection).

    Returns:
        tuple[nl.ndarray, nl.ndarray]: (wk_fp8_sb [P_MAX, num_K_tiles, head_dim] fp8_e4m3fn_x4,
            wk_scale_sb [P_MAX, num_K_tiles, head_dim] uint8 hardware-quadrant).
    """
    K_TILE = P_MAX * H_PACK  # 512
    num_K_tiles = dim // K_TILE

    wk_fp8_sb = sbm.alloc_stack((P_MAX, num_K_tiles, head_dim), nl.float8_e4m3fn_x4, buffer=nl.sbuf)
    """
    Single 3-D AP DMA covering all num_K_tiles tiles. Per-K-tile loops
    split this into many small DMAs (each [128 P, 128 F] = 16 KB),
    which is dominated by per-DMA descriptor overhead. Load the whole
    wk weight in one HW DGE DMA — pattern matches qkv_tkg_mx_impl.py.
    """
    nisa.dma_copy(
        dst=wk_fp8_sb[0:P_MAX, 0:num_K_tiles, 0:head_dim],
        src=wk_fp8.ap(
            pattern=[
                [head_dim, P_MAX],  # partition: stride head_dim, P_MAX rows
                [P_MAX * head_dim, num_K_tiles],  # k_tile: stride P_MAX*head_dim
                [1, head_dim],  # free: contiguous
            ],
            offset=0,
            dtype=nl.float8_e4m3fn_x4,
        ),
        dge_mode=nisa_constants.dge_mode.hwdge,
    )

    wk_scale_sb = sbm.alloc_stack((P_MAX, num_K_tiles, head_dim), nl.uint8, buffer=nl.sbuf)
    _load_mx_weight_scale(wk_scale, wk_scale_sb, num_K_tiles, free_dim=head_dim, free_offset=0, free_extent=head_dim)
    return wk_fp8_sb, wk_scale_sb


def w_projection_mx_batch(
    sbm: SbufManager,
    x: nl.ndarray,
    batch_start: int,
    S_total: int,
    dim: int,
    n_heads: int,
    scale: float,
    wp_tile_T_hoist: nl.ndarray,
    x_non_mx: nl.ndarray,
    weights_full_sb: nl.ndarray,
) -> None:
    """Batch-level W-projection (mirrors q_projection_mx pattern).

    Computes weights_full_sb[s, h] = (x[batch_start+s, :] @ weights_proj.T)[h] * scale
    for all S_total rows of the batch in one call, before the per-S-tile
    loop. The score loop then slices weights_full_sb per S-tile.

    Mirroring q_projection_mx: pre-loaded weights (wp_tile_T_hoist),
    per-S-tile activation cast loop, output written to a single SBUF
    buffer.

    Args:
        sbm (SbufManager): SBUF stack allocator used for scratch buffers.
        x (nl.ndarray): ``[B*S, dim]`` HBM activation (used for fallback path).
        batch_start (int): Row offset into x for this batch.
        S_total (int): Total S rows for this batch. Need NOT be a multiple of P_MAX --
            the last S-tile is row-clamped (handles CP shards with S_total < P_MAX).
        dim (int): Contraction dimension of the W projection.
        n_heads (int): Number of attention heads (output width).
        scale (float): Post-matmul multiply applied to the result.
        wp_tile_T_hoist (nl.ndarray): ``[P_MAX, num_dim_tiles, n_heads]`` bf16 — caller-
            hoisted, dim on partition, matmul-ready.
        x_non_mx (nl.ndarray): ``[B*S, dim]`` bf16 HBM, used to skip f32->bf16 cast.
        weights_full_sb (nl.ndarray): ``[P_MAX, num_S_tiles, n_heads]`` f32 — caller-
            allocated; result lands here, per-S-tile sliced by caller.

    Returns:
        None: Result is written in-place into weights_full_sb.
    """
    DIM_TILE = P_MAX
    P_TILE = P_MAX
    num_dim_tiles = (dim + DIM_TILE - 1) // DIM_TILE
    # CEIL (matches the caller's weights_full_sb allocation and the K/score path). FLOOR here
    # dropped the last partial S-tile: under CP with S_total < P_TILE (e.g. T_local=64 at ws64)
    # it gave 0 tiles, so the loop never ran and weights_full_sb was left uninitialized SBUF
    # (garbage per-head weights -> corrupted index_score / top-k). The partial last tile is
    # row-clamped below (s_tile_sz).
    num_S_tiles = (S_total + P_TILE - 1) // P_TILE
    compute_dtype = nl.bfloat16

    sbm.open_scope(name="w_proj_batch_mx")

    """
    Split the dim_tile accumulate chain across multiple PSUM banks so
    adjacent dim_tile matmuls can issue without waiting for prior to
    drain (was a single-bank RAW chain that blocked matmul issue and
    exposed DMA latency). NUM_W_BANKS=8 maps each dim_tile_idx to a
    bank via dim_tile_idx % NUM_W_BANKS; final reduction tensor_tensor-
    adds the 8 partial sums.
    """
    NUM_W_BANKS = 8 if num_dim_tiles >= 8 else num_dim_tiles
    kernel_assert(
        NUM_W_BANKS == 8,
        "w_projection_mx_batch fast path requires num_dim_tiles >= 8 "
        f"(got dim={dim} -> num_dim_tiles={num_dim_tiles}); add a fallback "
        "if smaller dim is needed.",
    )

    # Per-S-tile inner loop: load x slice (bf16), matmul against hoisted
    # wp slabs, write to weights_full_sb at the per-S-tile slot.
    for s_tile_idx in nl.sequential_range(num_S_tiles):
        s_start = s_tile_idx * P_TILE
        # Valid rows in this S-tile (< P_TILE for the last tile when S_total % P_TILE != 0).
        s_tile_sz = min(P_TILE, S_total - s_start)
        """
        8 PSUM banks for parallel accumulate; reduced after the loop.
        Allocate explicitly (parser frontend rejects list-comp NDArray
        creation inside sequential_range bodies).
        """
        w_psum_b0 = nl.ndarray((P_MAX, n_heads), dtype=nl.float32, buffer=nl.psum)
        w_psum_b1 = nl.ndarray((P_MAX, n_heads), dtype=nl.float32, buffer=nl.psum)
        w_psum_b2 = nl.ndarray((P_MAX, n_heads), dtype=nl.float32, buffer=nl.psum)
        w_psum_b3 = nl.ndarray((P_MAX, n_heads), dtype=nl.float32, buffer=nl.psum)
        w_psum_b4 = nl.ndarray((P_MAX, n_heads), dtype=nl.float32, buffer=nl.psum)
        w_psum_b5 = nl.ndarray((P_MAX, n_heads), dtype=nl.float32, buffer=nl.psum)
        w_psum_b6 = nl.ndarray((P_MAX, n_heads), dtype=nl.float32, buffer=nl.psum)
        w_psum_b7 = nl.ndarray((P_MAX, n_heads), dtype=nl.float32, buffer=nl.psum)
        w_psum_banks = [w_psum_b0, w_psum_b1, w_psum_b2, w_psum_b3, w_psum_b4, w_psum_b5, w_psum_b6, w_psum_b7]

        """
        Coalesce the 56 per-dim-tile dma_transposes into ONE 3-D
        dma_transpose per S-tile. Source view: x_non_mx[s_start:s_start+P_TILE,
        0:dim] reshaped as [P_TILE, num_dim_tiles, DIM_TILE]. Dst layout:
        [DIM_TILE P, num_dim_tiles, P_TILE F] via axes=(2, 1, 0). Each
        dim_tile's slab is then x_T_S_sb[:, dim_tile_idx, :].
        Per-partition cost: num_dim_tiles × P_TILE × 2 = 14 KB at v3_long.
        Eliminates the per-dim-tile DMA→matmul RAW chain (the antidep flagged
        in the profile) and replaces 56 DMA descriptor issues with 1.
        """
        if x_non_mx != None:
            x_T_S_sb = sbm.alloc_stack(
                (DIM_TILE, num_dim_tiles, P_TILE),
                compute_dtype,
            )
            nisa.dma_transpose(
                dst=x_T_S_sb[:DIM_TILE, :num_dim_tiles, :s_tile_sz],
                src=x_non_mx[batch_start + s_start : batch_start + s_start + s_tile_sz, 0:dim].reshape(
                    (s_tile_sz, num_dim_tiles, DIM_TILE)
                ),
                axes=(2, 1, 0),
            )
        else:
            x_T_S_f32 = sbm.alloc_stack(
                (DIM_TILE, num_dim_tiles, P_TILE),
                nl.float32,
                align=32,
            )
            x_T_S_sb = sbm.alloc_stack(
                (DIM_TILE, num_dim_tiles, P_TILE),
                compute_dtype,
            )
            nisa.dma_transpose(
                dst=x_T_S_f32[:DIM_TILE, :num_dim_tiles, :s_tile_sz],
                src=x[batch_start + s_start : batch_start + s_start + s_tile_sz, 0:dim].reshape(
                    (s_tile_sz, num_dim_tiles, DIM_TILE)
                ),
                axes=(2, 1, 0),
            )
            nisa.tensor_copy(
                dst=x_T_S_sb[:DIM_TILE, :num_dim_tiles, :s_tile_sz],
                src=x_T_S_f32[:DIM_TILE, :num_dim_tiles, :s_tile_sz],
                engine=nisa.engine.scalar,
            )

        for dim_tile_idx in range(num_dim_tiles):
            dim_start = dim_tile_idx * DIM_TILE
            dim_tile_size = min(DIM_TILE, dim - dim_start)
            bank_idx = dim_tile_idx % NUM_W_BANKS
            nisa.nc_matmul(
                dst=w_psum_banks[bank_idx][:s_tile_sz, :n_heads],
                stationary=x_T_S_sb[:dim_tile_size, dim_tile_idx, :s_tile_sz],
                moving=wp_tile_T_hoist[:dim_tile_size, dim_tile_idx, :n_heads],
                accumulate=(dim_tile_idx >= NUM_W_BANKS),
            )

        """
        Reduce NUM_W_BANKS=8 partial PSUM sums. tensor_tensor can't have
        both inputs in PSUM, so first eviction goes PSUM -> SBUF
        (engine=scalar), then subsequent banks are added one-by-one
        against the SBUF accumulator (data1 SBUF, data2 PSUM is allowed).
        """
        w_acc_sb = sbm.alloc_stack((P_TILE, n_heads), nl.float32)
        nisa.tensor_copy(
            dst=w_acc_sb[:s_tile_sz, :n_heads],
            src=w_psum_banks[0][:s_tile_sz, :n_heads],
            engine=nisa.engine.scalar,
        )
        for reduce_bank_idx in range(1, NUM_W_BANKS):
            nisa.tensor_tensor(
                dst=w_acc_sb[:s_tile_sz, :n_heads],
                data1=w_acc_sb[:s_tile_sz, :n_heads],
                data2=w_psum_banks[reduce_bank_idx][:s_tile_sz, :n_heads],
                op=nl.add,
            )
        # Apply post-scale and store into weights_full_sb at this S-tile slot (valid rows only;
        # the last tile's [s_tile_sz:P_TILE] rows are padding queries the score loop discards).
        nisa.tensor_scalar(
            dst=weights_full_sb[:s_tile_sz, s_tile_idx, :n_heads],
            data=w_acc_sb[:s_tile_sz, :n_heads],
            operand0=scale,
            op0=nl.multiply,
        )

    sbm.close_scope()


def k_and_weights_projection_mx(
    sbm: SbufManager,
    x: nl.ndarray,
    batch_start: int,
    S: int,
    wk_fp8_sb: nl.ndarray,
    wk_scale_sb: nl.ndarray,
    weights_proj: nl.ndarray,
    dim: int,
    head_dim: int,
    n_heads: int,
    scale: float,
    k_sb: nl.ndarray,
    weights_sb: nl.ndarray,
    x_non_mx: nl.ndarray = None,
    x_mx_data: nl.ndarray = None,
    x_mx_scale: nl.ndarray = None,
    s_tile_idx: int = 0,
    wp_tile_T_hoist: nl.ndarray = None,
    skip_w: bool = False,
) -> None:
    """K via MX matmul; weights_proj stays bf16.

    Caller pre-loads ``wk_fp8_sb`` and ``wk_scale_sb`` (see ``load_wk_mx_weights``)
    so the hot S-tile loop only quantizes activations and dispatches matmuls.

    K = nc_matmul_mx(quantize_mx(x), wk_fp8_sb) -> [S, head_dim]
    W = (x @ weights_proj^T) * scale            -> [S, n_heads]  (bf16, like before)

    Args:
        sbm (SbufManager): SBUF stack allocator used for scratch buffers.
        x (nl.ndarray): ``[B*S, dim]`` HBM activation (used for fallback path).
        batch_start (int): Row offset into x for this S-tile.
        S (int): Number of activation rows for this S-tile.
        wk_fp8_sb (nl.ndarray): ``[P_MAX, num_K_tiles, head_dim]`` fp8_e4m3fn_x4 SBUF — hoisted K weights.
        wk_scale_sb (nl.ndarray): ``[P_MAX, num_K_tiles, head_dim]`` uint8 SBUF — hoisted K scales.
        weights_proj (nl.ndarray): ``[n_heads, dim]`` bf16 HBM — W-projection weights (fallback path).
        dim (int): Contraction dimension of the K/W projections.
        head_dim (int): Per-head dimension (K output width).
        n_heads (int): Number of attention heads (W output width).
        scale (float): Post-matmul multiply applied to the W result.
        k_sb (nl.ndarray): ``[P_MAX, head_dim]`` f32 SBUF — K output buffer (written in-place).
        weights_sb (nl.ndarray): ``[P_MAX, n_heads]`` f32 SBUF — W output buffer (written in-place).
        x_non_mx (nl.ndarray): Optional ``[B*S, dim]`` bf16 HBM tensor — same data as
            ``x`` but pre-cast to bf16 on the host. When provided, the
            W-projection's per-dim-tile dma_transpose loads bf16 directly,
            skipping the f32->bf16 ``tensor_copy`` cast that runs on Scalar.
        x_mx_data (nl.ndarray): Optional ``[num_S_tiles, P_MAX, num_K_tiles*P_MAX]`` fp8x4
            HBM tensor — host pre-quantized x in the kernel's K-side SBUF
            layout. When provided alongside ``x_mx_scale``, K-projection
            skips the in-kernel HBM load + 4-pass nc_transpose swizzle +
            quantize_mx.
        x_mx_scale (nl.ndarray): Optional ``[num_S_tiles, P_MAX, num_K_tiles*P_MAX]``
            uint8 HBM tensor — host pre-computed MX scales in HW-quadrant
            layout (rows [0..3, 32..35, 64..67, 96..99] valid per K-tile).
        s_tile_idx (int): Index of the current S-tile (used to slice
            ``x_mx_data`` / ``x_mx_scale`` along their first axis).
        wp_tile_T_hoist (nl.ndarray): Optional ``[P_MAX, num_dim_tiles, n_heads]`` bf16 SBUF —
            caller-hoisted W-projection weights; skips per-dim-tile re-DMA.
        skip_w (bool): When True, W-projection is skipped (caller ran it at batch level).

    Returns:
        None: K result lands in k_sb; W result (when not skipped) lands in weights_sb.
    """
    K_TILE = P_MAX * H_PACK  # 512
    num_K_tiles = dim // K_TILE
    DIM_TILE = P_MAX
    num_dim_tiles = (dim + DIM_TILE - 1) // DIM_TILE
    compute_dtype = nl.bfloat16

    sbm.open_scope(name="kw_proj_mx")

    # ----- MX-K path -----
    hidden_qtz_sb = sbm.alloc_stack((P_MAX, num_K_tiles, P_MAX), nl.float8_e4m3fn_x4, buffer=nl.sbuf, align=32)
    hidden_scale_sb = sbm.alloc_stack((P_MAX, num_K_tiles, P_MAX), nl.uint8, buffer=nl.sbuf)
    if x_mx_data != None:
        # Fast path: host pre-quantized x. Two contiguous DMAs per S-tile
        # replace the entire load + 4-pass swizzle + quantize_mx pipeline.
        nisa.dma_copy(
            dst=hidden_qtz_sb[:P_MAX, :num_K_tiles, :P_MAX],
            src=x_mx_data[s_tile_idx, :P_MAX, :],
            dge_mode=nisa_constants.dge_mode.hwdge,
        )
        nisa.dma_copy(
            dst=hidden_scale_sb[:P_MAX, :num_K_tiles, :P_MAX],
            src=x_mx_scale[s_tile_idx, :P_MAX, :],
            dge_mode=nisa_constants.dge_mode.hwdge,
        )
    else:
        _quantize_activation_for_mx(
            sbm,
            x,
            batch_start,
            S,
            dim,
            hidden_qtz_sb,
            hidden_scale_sb,
            num_K_tiles,
        )

    k_psum = nl.ndarray((P_MAX, head_dim), dtype=nl.float32, buffer=nl.psum)
    for k_tile in nl.affine_range(num_K_tiles):
        nisa.nc_matmul_mx(
            dst=k_psum[:S, :head_dim],
            stationary=hidden_qtz_sb[:P_MAX, k_tile, :S],
            moving=wk_fp8_sb[:P_MAX, k_tile, :head_dim],
            stationary_scale=hidden_scale_sb[:P_MAX, k_tile, :S],
            moving_scale=wk_scale_sb[:P_MAX, k_tile, :head_dim],
            accumulate=(k_tile != 0),
        )
    nisa.tensor_copy(dst=k_sb[:S, :head_dim], src=k_psum[:S, :head_dim], engine=nisa.engine.scalar)

    """
    ----- BF16-W path -----
    When skip_w=True, the caller has run W-projection at batch level
    (w_projection_mx_batch) ahead of the S-tile loop and weights_sb is
    already populated by the caller — skip the in-S-tile W-proj.
    """
    if skip_w:
        sbm.close_scope()
        return

    """
    Double-buffer x_tile_T / wp_tile_T so dim-tile N+1's DMA + cast can
    overlap with dim-tile N's nc_matmul. Without interleave the 56-tile
    loop runs DMA -> cast -> DMA -> cast -> matmul fully serially per
    iteration (visible as Engine deps in the profile).
    """
    w_psum = nl.ndarray((P_MAX, n_heads), dtype=nl.float32, buffer=nl.psum)

    sbm.open_scope(interleave_degree=2, name="w_proj_dim_tile_loop")
    for dim_tile_idx in nl.affine_range(num_dim_tiles):
        dim_start = dim_tile_idx * DIM_TILE
        dim_tile_size = min(DIM_TILE, dim - dim_start)
        x_tile_T = sbm.alloc_stack((DIM_TILE, P_MAX), compute_dtype)
        if x_non_mx != None:
            # Fast path: x_non_mx is already bf16, dma_transpose loads it
            # directly into x_tile_T without an explicit f32->bf16 cast.
            nisa.dma_transpose(
                dst=x_tile_T[:dim_tile_size, :S],
                src=x_non_mx[batch_start : batch_start + S, dim_start : dim_start + dim_tile_size],
            )
        else:
            x_tile_T_f32 = sbm.alloc_stack((DIM_TILE, P_MAX), nl.float32, align=32)
            nisa.dma_transpose(
                dst=x_tile_T_f32[:dim_tile_size, :S],
                src=x[batch_start : batch_start + S, dim_start : dim_start + dim_tile_size],
            )
            nisa.tensor_copy(
                dst=x_tile_T[:dim_tile_size, :S], src=x_tile_T_f32[:dim_tile_size, :S], engine=nisa.engine.scalar
            )
        if wp_tile_T_hoist != None:
            """
            Use the caller-hoisted weights_proj slab; weights_proj is
            constant across S-tiles so re-DMAing per dim_tile is wasted
            work (was 56 dma_transpose + 56 bf16 casts per S-tile).
            """
            wp_tile_T_view = wp_tile_T_hoist[:dim_tile_size, dim_tile_idx, :n_heads]
        else:
            wp_tile_T_f32 = sbm.alloc_stack((DIM_TILE, n_heads), nl.float32, align=32)
            wp_tile_T = sbm.alloc_stack((DIM_TILE, n_heads), compute_dtype)
            nisa.dma_transpose(
                dst=wp_tile_T_f32[:dim_tile_size, :n_heads],
                src=weights_proj[:n_heads, dim_start : dim_start + dim_tile_size],
            )
            nisa.tensor_copy(
                dst=wp_tile_T[:dim_tile_size, :n_heads],
                src=wp_tile_T_f32[:dim_tile_size, :n_heads],
                engine=nisa.engine.scalar,
            )
            wp_tile_T_view = wp_tile_T[:dim_tile_size, :n_heads]
        nisa.nc_matmul(dst=w_psum[:S, :n_heads], stationary=x_tile_T[:dim_tile_size, :S], moving=wp_tile_T_view)
        sbm.increment_section()
    sbm.close_scope()

    nisa.tensor_scalar(dst=weights_sb[:S, :n_heads], data=w_psum[:S, :n_heads], operand0=scale, op0=nl.multiply)

    sbm.close_scope()
