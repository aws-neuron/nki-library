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

"""MXFP8 Flash Decode Attention TKG Kernel — Separate KV blocks, packed-Q eviction.

Computes: output = softmax(Q @ K^T / sqrt(d)) @ V
using MXFP8-quantized block KV cache on Trainium 3.

KV Block Format
===============
K and V are stored separately. Each block covers 128 tokens and is pre-quantized
to MXFP8 format: [32P, 160F] = [32 partitions, 128 data + 32 scale cols].
  k_prior: [num_blocks, 32, 160] float32
  v_prior: [num_blocks, 32, 160] float32

Tiling Hierarchy
================
  Block (128 tokens)  — one nc_matmul_mx operand, [32P, 160F] in MXFP8
  Fold (512 tokens)   — 4 blocks stacked vertically, [128P, 160F] in SBUF
  Chunk (2048 tokens) — 4 folds concatenated on the free dim, [128, 4, 160] in SBUF, one online softmax iteration

Kernel Flow (per batch)
=======================
  Step 0: Load Q, scale by 1/sqrt(d), quantize to MXFP8, build packed-Q variants
  Step 1: Initialize online softmax state (running_max, running_sum, acc)
  Step 2: For each chunk (prior chunks, plus one final "active chunk"):
    2a: Load mask + K/V — prior blocks via indirect DMA (swdge), or quantize the
        active k_active/v_active into block0/fold0 of the chunk buffers (MXFP8)
    2b: MM1 — Q × K^T with packed-Q eviction (nc_matmul_mx, row-tiled)
    2c: Online softmax — BF16 scores → exp → running max/sum update
    2d: Re-quantize scores to MXFP8 → MM2 — scores × V (nc_matmul_mx)
  Step 3: LNC2 gather (if sharded) → normalize (acc / running_sum) → store

Row-Tiling vs Packed-Q Eviction
===============================
n_query_rows = q_head * s_active is the total query-row count. It is split into
n_row_tiles = ceil(n_query_rows / 128) row-tiles of rows_per_tile = min(n_query_rows, 128)
rows; each tile is an independent 128-row attention problem over the same KV, with its
own online-softmax state and PSUM accumulator. Row-tiling and variant-packing never
coexist: variants_per_tile > 1 only when n_query_rows < 128 (n_row_tiles == 1), and
n_row_tiles > 1 only when n_query_rows >= 128 (variants_per_tile == 1). All the layout
below is described per row-tile, over its rows_per_tile query rows.

Packed-Q Eviction (per row-tile)
--------------------------------
Each Q x K block matmul fills only rows_per_tile output partitions, so Q is replicated
into variants_per_tile = 128 // rows_per_tile variants — variant v holds the real Q in
free band [v*rows_per_tile : (v+1)*rows_per_tile] and zeros elsewhere, landing its
scores in output-partition band v. variants_per_tile blocks — one per fold sharing a
column group, all at the same block-in-fold position — then accumulate into a single
[128P, block_len] PSUM tile, using all 128 partitions:
    rows_per_tile=128 → 1 variant, 1 block per tile (row-tiled, no packing)
    rows_per_tile=64  → 2 variants (Q_lo, Q_hi), 2 blocks per tile
    rows_per_tile=32  → 4 variants, 4 blocks per tile

Score Layout: [128P, score_free], shared by MM1's PSUM buffer and the evicted SBUF scores
    score_free = folds_per_chunk * block_len * score_tiles_per_fold (2048 for
    rows_per_tile=128, 1024 for 64, 512 for 32). Logical block b of the chunk is
    fold_idx, block_idx = divmod(b, blocks_per_fold), and its scores land at
        partition rows [band_idx * rows_per_tile : (band_idx + 1) * rows_per_tile]
        free columns  [group_idx * fold_len + block_idx * block_len : + block_len]
    where group_idx, band_idx = divmod(fold_idx, variants_per_tile). A column group is
    fold_len wide and holds variants_per_tile folds, one per partition band.

    A fold's blocks differ only in the block_idx term, so each fold's fold_len tokens
    occupy fold_len contiguous columns of a single band, in token order — which is what
    lets _load_chunk_mask scatter the token-sequential mask one fold per DMA.

Mask Layout: [B, H, s_active, s_prior + s_active] uint8
    User-provided unified per-head mask in token-sequential order. One row per
    active query token; the s_prior prefix masks the prior context and the trailing
    s_active columns mask the active tokens (the caller encodes intra-active
    causality). 1 = valid token, 0 = masked (score set to -inf).
"""

from dataclasses import dataclass
from typing import Optional

import nki
import nki.isa as nisa
import nki.language as nl

from ...core.utils.allocator import SbufManager, create_auto_alloc_manager
from ...core.utils.kernel_assert import kernel_assert
from ...core.utils.kernel_helpers import div_ceil
from .attention_mxfp8_tkg_utils import mm1_packing_geometry, swizzle_quantize_mx

# bf16 largest representable magnitude; negated to fill masked-out score positions with -inf.
_SCORE_MASK_INF = 65504.0
# Sentinel for the running-max initialization (effectively -inf before the first chunk).
_RUNNING_MAX_INIT = -1e38


# ── Tile Constants ─────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class MXTileConstants(nl.NKIObject):
    """Hardware and MXFP8 ISA constants for Trainium — immutable chip constraints."""

    p_max: int = 128
    """SBUF partition count."""

    p_per_quadrant: int = 32
    """Partitions per nc_matmul_mx operand quadrant (ISA constraint)."""

    mx_group_partitions: int = 8
    """Partitions per MX scale group (MX format constraint)."""


TC = MXTileConstants()


# ── Configuration ──────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class AttnMXFP8Config(nl.NKIObject):
    """Caller-provided configuration for MXFP8 attention TKG.

    Contains only user-facing knobs — no derived values or mutable state.
    """

    bs: int
    """Batch size."""

    q_head: int
    """Number of query heads (32 or 64)."""

    bucket_size: int
    """Bucket size in tokens (compile-time, multiple of chunk_tokens)."""

    d_head: int
    """Head dimension."""

    s_active: int = 1
    """Active sequence length (query tokens per head). 1 for single-token decode,
    > 1 for speculative decoding. Supported s_active in {1, 2, 4, 8}, giving n_query_rows =
    q_head * s_active in {32, 64, 128, 256, 512}. n_query_rows <= 128 packs Q variants;
    n_query_rows > 128 row-tiles the query rows into n_query_rows/128 tiles of 128 rows."""

    def __post_init__(self):
        kernel_assert(self.bs >= 1, f"bs must be >= 1, got {self.bs=}")
        kernel_assert(self.q_head in (32, 64), f"q_head must be 32 or 64, got {self.q_head=}")
        kernel_assert(self.s_active in (1, 2, 4, 8), f"s_active must be 1, 2, 4 or 8, got {self.s_active=}")


class TileParams(nl.NKIObject):
    """Derived tiling, sharding, and geometry parameters computed once at kernel entry."""

    def __init__(self, cfg: AttnMXFP8Config):
        self.s_active = cfg.s_active
        """Active sequence length (query tokens per head)."""

        # Softmax
        self.softmax_scale = 1.0 / (cfg.d_head**0.5)
        """Softmax scaling factor: 1/sqrt(d_head)."""

        # Block geometry
        self.p_per_block = TC.p_per_quadrant
        """Partitions per block (= one quadrant)."""

        self.block_len = cfg.d_head
        """Tokens per block (= d_head)."""

        self.packed_cols = self.block_len + self.block_len // 4
        """Data + scale columns per block."""

        # Tiling hierarchy
        self.blocks_per_fold = 4
        """Blocks per fold."""

        self.folds_per_chunk = 4
        """Folds per chunk."""

        self.fold_len = self.blocks_per_fold * self.block_len
        """Tokens per fold."""

        """
        Packed-Q eviction geometry (native per-head-count variant packing).
        n_query_rows = total query rows = q_head * s_active. When n_query_rows > 128 the rows
        do not fit in one 128-partition tile, so they are split into n_row_tiles tiles of
        rows_per_tile (= 128) rows each; each tile is an independent 128-row attention
        problem over the same KV. When n_query_rows <= 128 there is one tile (rows_per_tile ==
        n_query_rows) and the packed-Q eviction applies as before.
        """
        self.n_query_rows = cfg.q_head * cfg.s_active
        """Total query rows = q_head * s_active."""
        self.rows_per_tile = min(self.n_query_rows, TC.p_max)
        """Query rows per row-tile (= n_query_rows when n_query_rows <= 128, else 128)."""
        self.n_row_tiles = div_ceil(self.n_query_rows, TC.p_max)
        """Number of row-tiles the query rows are split into (n_query_rows / 128, or 1)."""
        variants_per_tile, score_tiles_per_fold = mm1_packing_geometry(
            self.rows_per_tile, p_max=TC.p_max, blocks_per_fold=self.blocks_per_fold
        )
        self.variants_per_tile = variants_per_tile
        """Q variants packed into one PSUM tile (2 for rows_per_tile=64, 4 for 32, 1 for 128)."""
        self.score_tiles_per_fold = score_tiles_per_fold
        """Score tiles per fold."""

        # Q dimensions
        self.q_free = cfg.d_head
        """Q free dim for packed-Q variants."""

        # Chunk geometry
        self.chunk_tokens = self.folds_per_chunk * self.fold_len
        """Tokens per chunk (2048)."""

        self.score_free = self.folds_per_chunk * self.block_len * self.score_tiles_per_fold
        """Free-dim width of score buffer after eviction."""

        # Chunk counts and sharding
        self.num_chunks = cfg.bucket_size // self.chunk_tokens
        """Total number of chunks across all NCs."""

        self._validate(cfg)

        n_prgs = nl.num_programs(0)
        prg_id = nl.program_id(0)
        use_lnc2 = (n_prgs > 1) and (self.num_chunks >= n_prgs)
        self.sprior_n_prgs = n_prgs if use_lnc2 else 1
        """Number of NCs participating in s_prior sharding (1 or 2)."""

        self.sprior_prg_id = prg_id if use_lnc2 else 0
        """This NC's program ID for s_prior sharding (0 or 1)."""

        self.chunks_per_nc = self.num_chunks // self.sprior_n_prgs
        """Number of chunks processed by this NC."""

        self.chunk_start = self.sprior_prg_id * self.chunks_per_nc
        """First chunk index for this NC."""

    def _validate(self, cfg: AttnMXFP8Config):
        """Validate that the bucket size tiles evenly into whole chunks."""
        kernel_assert(
            cfg.bucket_size % self.chunk_tokens == 0,
            f"bucket_size must be multiple of chunk_tokens, got {cfg.bucket_size=}, {self.chunk_tokens=}",
        )
        kernel_assert(
            cfg.bucket_size >= self.chunk_tokens,
            f"bucket_size must be >= chunk_tokens, got {cfg.bucket_size=}, {self.chunk_tokens=}",
        )


class QuantizedQ(nl.NKIObject):
    """MXFP8-quantized Q with packed variants for MM1."""

    def __init__(self, cfg, tp, sbm):
        self.cfg = cfg
        self.tp = tp
        self.sbm = sbm
        self.q_variants = None
        """List of n_row_tiles lists, each of variants_per_tile (data, scale) MXFP8
        buffers. Each buffer is [128P, q_free F]: the partition dim is the 32-partition
        MXFP8 Q operand replicated across the four fold quadrants, and the free dim is
        the packed output-partition layout (variants_per_tile * rows_per_tile). Row-tile
        t's variant v writes that tile's rows into free band
        [v*rows_per_tile : (v+1)*rows_per_tile] (zeros elsewhere) so its block matmul
        lands in output-partition band v."""

    def load_from_hbm(self, q_hbm, batch_idx):
        """Load Q from HBM, scale, quantize to MXFP8, build packed variants per row-tile.

        Q's [q_head, s_active] axes flatten head-major (row = h*s_active + s) into
        n_query_rows query rows, split into n_row_tiles tiles of rows_per_tile (<= 128) rows. Each tile
        is loaded, scaled, MXFP8-quantized to a [32, rows_per_tile] base, and expanded to
        its packed Q variants independently — nothing is ever allocated wider than 128
        partitions.
        """
        cfg, tp, sbm = self.cfg, self.tp, self.sbm
        rows_per_tile = tp.rows_per_tile
        mx_par = cfg.d_head // 4  # MXFP8 Q operand contraction dim (32) on partitions

        q_band = q_hbm[batch_idx].reshape((tp.n_query_rows, cfg.d_head))

        self.q_variants = []
        for row_tile in range(tp.n_row_tiles):
            row_base = row_tile * rows_per_tile

            q_bf16 = sbm.alloc_stack((rows_per_tile, cfg.d_head), dtype=nl.bfloat16)
            nisa.dma_copy(dst=q_bf16, src=q_band[nl.ds(row_base, rows_per_tile), :])

            q_scaled = sbm.alloc_stack((rows_per_tile, cfg.d_head), dtype=nl.bfloat16)
            nisa.tensor_scalar(dst=q_scaled, data=q_bf16, op0=nl.multiply, operand0=tp.softmax_scale)

            q_mx_data_base = sbm.alloc_stack((mx_par, rows_per_tile), dtype=nl.float8_e4m3fn_x4)
            q_mx_scale_base = sbm.alloc_stack((mx_par, rows_per_tile), dtype=nl.uint8)
            swizzle_quantize_mx(q_scaled, q_mx_data_base, q_mx_scale_base, sbm)

            self.q_variants.append(_build_q_tile_variants(q_mx_data_base, q_mx_scale_base, tp, sbm))


class SoftmaxState(nl.NKIObject):
    """Online softmax running state and accumulators.

    Constructed once per row-tile per batch iteration. Holds running max/sum, the PSUM
    accumulator, and the identity matrix needed for PE-based sum reduction. Operates on
    the tile's rows_per_tile query rows (= n_query_rows when n_query_rows <= 128).
    """

    def __init__(self, cfg, tp, sbm, identity_sb):
        self.cfg = cfg
        self.tp = tp
        self.sbm = sbm
        self.band_p = tp.rows_per_tile  # query rows this tile occupies (= n_query_rows when <= 128)
        self.n_bands = tp.variants_per_tile  # bands packed per PSUM tile (1, 2 or 4)

        # Identity matrix for PE reduction (shared across batches, passed in)
        self.identity_sb = identity_sb
        """[TC.p_max, TC.p_max] bf16 in SBUF."""

        # Running state
        self.running_max = sbm.alloc_stack((TC.p_max, 1), dtype=nl.float32)
        """[TC.p_max, 1] fp32 in SBUF."""
        nisa.memset(self.running_max, value=_RUNNING_MAX_INIT, engine=nisa.gpsimd_engine)

        self.running_sum = sbm.alloc_stack((TC.p_max, 1), dtype=nl.float32)
        """[TC.p_max, 1] fp32 in SBUF."""
        nisa.memset(self.running_sum, value=0.0, engine=nisa.gpsimd_engine)

        self.acc = nl.ndarray((tp.rows_per_tile, cfg.d_head), dtype=nl.float32, buffer=nl.psum)
        """[rows_per_tile, d_head] fp32 in PSUM."""
        nisa.memset(self.acc, value=0.0)

        # Set later in Step 3 (finalize)
        self.acc_sb = None
        """[TC.p_max, d_head] fp32 in SBUF."""
        self.out_bf16 = None
        """[rows_per_tile, d_head] bf16 in SBUF."""

    def update_online_softmax(self, score_sb, score_max_sb, score_sb_fp32_reinterp):
        """Packed online softmax on [128, score_free]. Score path in bf16, accumulators in fp32.

        Sub-steps:
          1. Compute new global max across all n_bands bands
          2. Rescale old accumulators by correction factor
          3. Compute exp(score - max) in tiles, reduce sum via PE matmul
          4. Launch DMA reinterpret bf16→fp32 (overlaps with sum reduce)
          5. Cross-band sum reduce, update running state
        """
        m_new, correction = self._compute_new_max(score_max_sb)
        self._rescale_accumulators(correction)
        l_local = self._exp_and_reduce(score_sb, m_new)

        nisa.dma_copy(
            dst=score_sb_fp32_reinterp,
            src=score_sb.view(nl.float32),
        )

        self._update_running_state(l_local, m_new)

    def _reduce_bands(self, packed, op):
        """Reduce the n_bands partition bands of a [128, 1] tensor onto band [0:band_p].

        A head's per-band values live at partitions band_idx*band_p + h. tensor_tensor
        requires both operands aligned to partition 0, so each upper band is realigned
        with a copy (reusing one scratch buffer) before it is combined into the running
        result; band 0 is already aligned and feeds the first combine in place. `op` is
        maximum for the score max, add for the sum. Returns a fresh [band_p, 1] tensor.

        When n_bands == 1 (band_p == 128, no packing) there is nothing to reduce, so the
        single band is copied out unchanged.
        """

        kernel_assert(self.n_bands in (1, 2, 4), f"n_bands assumed to be 1, 2 or 4, got {self.n_bands}")

        sbm = self.sbm
        band_p = self.band_p

        reduced = sbm.alloc_stack((band_p, 1), dtype=nl.float32)

        if self.n_bands == 1:
            nisa.tensor_copy(dst=reduced, src=packed[nl.ds(0, band_p), :], engine=nisa.scalar_engine)
            return reduced

        aligned = sbm.alloc_stack((band_p, 1), dtype=nl.float32)
        nisa.tensor_copy(dst=aligned, src=packed[nl.ds(band_p, band_p), :], engine=nisa.scalar_engine)
        nisa.tensor_tensor(dst=reduced, data1=packed[nl.ds(0, band_p), :], data2=aligned, op=op)
        for band_idx in range(2, self.n_bands):
            nisa.tensor_copy(dst=aligned, src=packed[nl.ds(band_idx * band_p, band_p), :], engine=nisa.scalar_engine)
            nisa.tensor_tensor(dst=reduced, data1=reduced, data2=aligned, op=op)
        return reduced

    def _compute_new_max(self, m_local):
        """Cross-band reduce of the MM1-produced chunk-local max, merge with running max.

        m_local [128, 1] is the per-partition score max produced by the MM1
        select_reduce eviction (fused, no standalone tensor_reduce needed). A head's
        scores span n_bands partition bands, so reduce them onto [0:band_p] first.

        Returns (m_new [128, 1] broadcast across bands, correction [band_p, 1]).
        """
        sbm = self.sbm
        band_p = self.band_p

        m_global = self._reduce_bands(m_local, nl.maximum)

        # Merge with the running max in band 0, then broadcast it across the remaining
        # bands: _exp_and_reduce subtracts m_new from every one of the 128 score partitions.
        m_new = sbm.alloc_stack((self.n_bands * band_p, 1), dtype=nl.float32)
        nisa.tensor_tensor(
            dst=m_new[nl.ds(0, band_p), :], data1=self.running_max[nl.ds(0, band_p), :], data2=m_global, op=nl.maximum
        )
        for band_idx in range(1, self.n_bands):
            nisa.tensor_copy(
                dst=m_new[nl.ds(band_idx * band_p, band_p), :],
                src=m_new[nl.ds(0, band_p), :],
                engine=nisa.scalar_engine,
            )

        # Only band 0 is consumed downstream (acc uses [0:band_p], the other bands of
        # running_sum are dead), so compute correction at band width.
        correction = sbm.alloc_stack((band_p, 1), dtype=nl.float32)
        # Fused exp(running_max - m_new): activate2 does (data - m_new) then exp in one
        # scalar-engine instruction, replacing the tensor_scalar + activation pair.
        nisa.activate2(
            dst=correction,
            op=nl.exp,
            data=self.running_max[nl.ds(0, band_p), :],
            imm0=m_new[nl.ds(0, band_p), :],
            imm1=0.0,
            op0=nl.subtract,
            op1=nl.bypass,
        )

        return m_new, correction

    def _rescale_accumulators(self, correction):
        """Rescale old acc and running_sum by correction factor ([0:band_p] only)."""
        band_p = self.band_p
        nisa.tensor_scalar(
            dst=self.acc,
            data=self.acc,
            op0=nl.multiply,
            operand0=correction,
            engine=nisa.scalar_engine,
        )
        nisa.tensor_scalar(
            dst=self.running_sum[nl.ds(0, band_p), :],
            data=self.running_sum[nl.ds(0, band_p), :],
            op0=nl.multiply,
            operand0=correction,
            engine=nisa.scalar_engine,
        )

    def _exp_and_reduce(self, score_sb, m_new):
        """Compute exp(score - max) in-place, reduce sum via PE matmul with identity.

        Returns l_local [128, 1] — local sum of exponentials.
        """
        sbm = self.sbm
        TILE_F = 128
        n_tiles = score_sb.shape[1] // TILE_F

        psum_reduction = nl.ndarray((TILE_F, TILE_F), dtype=nl.float32, buffer=nl.psum)
        nisa.memset(psum_reduction, value=0.0)

        for tile_idx in range(n_tiles):
            f_off = tile_idx * TILE_F
            # Fused exp(score - m_new): activate2 does (data - m_new) then exp in one
            # scalar-engine instruction, replacing the tensor_scalar + activation pair.
            nisa.activate2(
                dst=score_sb[:, nl.ds(f_off, TILE_F)],
                op=nl.exp,
                data=score_sb[:, nl.ds(f_off, TILE_F)],
                imm0=m_new,
                imm1=0.0,
                op0=nl.subtract,
                op1=nl.bypass,
            )
            nisa.nc_matmul(
                dst=psum_reduction,
                stationary=self.identity_sb,
                moving=score_sb[:, nl.ds(f_off, TILE_F)],
            )

        l_local = sbm.alloc_stack((TC.p_max, 1), dtype=nl.float32)
        nisa.tensor_reduce(dst=l_local, op=nl.add, data=psum_reduction, axis=1)
        return l_local

    def _update_running_state(self, l_local, m_new):
        """Cross-band sum reduce, accumulate into running_sum, update running_max."""
        band_p = self.band_p

        l_global = self._reduce_bands(l_local, nl.add)

        nisa.tensor_tensor(
            dst=self.running_sum[nl.ds(0, band_p), :],
            data1=self.running_sum[nl.ds(0, band_p), :],
            data2=l_global,
            op=nl.add,
        )
        nisa.tensor_copy(dst=self.running_max, src=m_new, engine=nisa.scalar_engine)


class ChunkBuffers(nl.NKIObject):
    """Per-chunk SBUF buffers, allocated once at the top of each chunk iteration."""

    def __init__(self, tp, sbm):
        self.chunk_mask_sb = sbm.alloc_stack((TC.p_max, tp.score_free), dtype=nl.uint8)
        """[TC.p_max, tp.score_free] uint8 mask for this chunk."""

        self.score_sb = sbm.alloc_stack((TC.p_max, tp.score_free), dtype=nl.bfloat16)
        """[TC.p_max, tp.score_free] bf16 scores after MM1."""

        self.score_max_sb = sbm.alloc_stack((TC.p_max, 1), dtype=nl.float32)
        """[TC.p_max, 1] fp32 per-partition score max, produced by the MM1 eviction."""

        self.score_sb_fp32_reinterp = sbm.alloc_stack((TC.p_max, tp.score_free // 2), dtype=nl.float32)
        """[TC.p_max, tp.score_free // 2] fp32 reinterpretation of scores for MM2 quantize."""

        self.k_buf = sbm.alloc_stack((TC.p_max, tp.folds_per_chunk, tp.packed_cols), dtype=nl.float32)
        """[TC.p_max, tp.folds_per_chunk, tp.packed_cols] K chunk buffer."""

        self.v_buf = sbm.alloc_stack((TC.p_max, tp.folds_per_chunk, tp.packed_cols), dtype=nl.float32)
        """[TC.p_max, tp.folds_per_chunk, tp.packed_cols] V chunk buffer."""


def _load_chunk_mask(
    chunk_mask_sb: nl.NkiTensor, batch_idx: int, chunk_idx: int, row_tile: int, mask: nl.NkiTensor, tp: TileParams
) -> None:
    """Scatter one row-tile's prior mask into the eviction-layout SBUF buffer.

    mask is [B, H, s_active, s_prior + s_active]; here we use only the s_prior prefix
    (the s_active tail drives the active-token step). The per-batch mask reshapes
    head-major ([H, s_active] -> n_query_rows, row = h*s_active + s) to [n_query_rows, s_prior + s_active].
    Row-tile `row_tile` owns query rows [row_tile*rows_per_tile : +rows_per_tile].

    One DMA per fold: a fold's fold_len tokens occupy fold_len contiguous columns of a
    single partition band, in token order, so the token-sequential mask needs no
    reordering within a fold — see the module docstring's Score Layout. The folds tile
    the whole buffer. Tokens at or past s_prior have no mask entry, so a chunk reaching
    past s_prior copies only the in-range prefix of each fold; the memset zeroes the
    remaining columns, which no DMA writes and which would otherwise hold the previous
    chunk's or row-tile's mask.
    """
    s_prior = mask.shape[3] - tp.s_active
    rows_per_tile = tp.rows_per_tile
    mask_prior = mask[batch_idx].reshape((tp.n_query_rows, mask.shape[3]))
    row_base = row_tile * rows_per_tile

    if (chunk_idx + 1) * tp.chunk_tokens > s_prior:
        nisa.memset(chunk_mask_sb, value=0)

    for fold_idx in range(tp.folds_per_chunk):
        tok_start = (chunk_idx * tp.folds_per_chunk + fold_idx) * tp.fold_len
        n_copy = min(tp.fold_len, s_prior - tok_start)
        if n_copy <= 0:
            continue
        group_idx, band_idx = divmod(fold_idx, tp.variants_per_tile)
        p_row = band_idx * rows_per_tile
        f_off = group_idx * tp.fold_len
        nisa.dma_copy(
            dst=chunk_mask_sb[nl.ds(p_row, rows_per_tile), nl.ds(f_off, n_copy)],
            src=mask_prior[nl.ds(row_base, rows_per_tile), nl.ds(tok_start, n_copy)],
        )


def _load_chunk_kv(cb, k_prior, v_prior, kv_loader, chunk_idx):
    """Load this chunk's K/V blocks from HBM (shared across all row-tiles)."""
    kv_loader.load_blocks(cb.k_buf, k_prior, chunk_idx)
    kv_loader.load_blocks(cb.v_buf, v_prior, chunk_idx)


def _build_active_chunk_kv(cb, batch_idx, k_active, v_active, tp, sbm):
    """Quantize the active tokens into the block0/fold0 slot of the chunk K/V buffers."""
    s_active = tp.s_active
    d_head = tp.block_len
    p_block = tp.p_per_block  # 32

    nisa.memset(cb.k_buf, value=0.0)
    nisa.memset(cb.v_buf, value=0.0)

    # Active K: [tokens, d] padded to a full block, contraction along d_head.
    k_pad = sbm.alloc_stack((TC.p_max, d_head), dtype=nl.bfloat16)
    nisa.memset(k_pad, value=0.0)
    nisa.dma_copy(dst=k_pad[nl.ds(0, s_active), :], src=k_active[batch_idx])
    swizzle_quantize_mx(
        k_pad,
        cb.k_buf[nl.ds(0, p_block), 0, nl.ds(0, d_head)].view(nl.float8_e4m3fn_x4),
        cb.k_buf[nl.ds(0, p_block), 0, nl.ds(d_head, d_head // 4)].view(nl.uint8),
        sbm,
    )

    # Active V: load transposed [d, tokens] so contraction is along block_len.
    v_pad_t = sbm.alloc_stack((d_head, TC.p_max), dtype=nl.bfloat16)
    nisa.memset(v_pad_t, value=0.0)
    nisa.dma_transpose(dst=v_pad_t[:, nl.ds(0, s_active)], src=v_active[batch_idx])
    swizzle_quantize_mx(
        v_pad_t,
        cb.v_buf[nl.ds(0, p_block), 0, nl.ds(0, d_head)].view(nl.float8_e4m3fn_x4),
        cb.v_buf[nl.ds(0, p_block), 0, nl.ds(d_head, d_head // 4)].view(nl.uint8),
        sbm,
    )


def _load_active_chunk_mask(chunk_mask_sb, batch_idx, row_tile, mask, tp):
    """Scatter one row-tile's active-token mask into the block0/fold0 slot of the
    eviction-layout buffer.
    """
    s_active = tp.s_active
    s_prior = mask.shape[3] - s_active
    rows_per_tile = tp.rows_per_tile
    mask_r = mask[batch_idx].reshape((tp.n_query_rows, mask.shape[3]))
    row_base = row_tile * rows_per_tile

    nisa.memset(chunk_mask_sb, value=0)
    nisa.dma_copy(
        dst=chunk_mask_sb[nl.ds(0, rows_per_tile), nl.ds(0, s_active)],
        src=mask_r[nl.ds(row_base, rows_per_tile), nl.ds(s_prior, s_active)],
    )


# ── Q Variant Construction ─────────────────────────────────────────────────────
def _build_q_tile_variants(q_mx_data_base, q_mx_scale_base, tp, sbm):
    """Build one row-tile's packed Q buffers from its base MXFP8 Q [32P, rows_per_tile F].

    Within a tile, variants_per_tile variants each place the tile's rows in a different
    free band [v*rows_per_tile : (v+1)*rows_per_tile] (zeros elsewhere), so variant v's
    Q x K block matmul lands in output-partition band v.

    Returns:
        List of variants_per_tile (data, scale) tuples, every buffer [128P, q_free].
    """
    rows_per_tile = tp.rows_per_tile
    variants = []
    for variant_idx in range(tp.variants_per_tile):
        band_off = variant_idx * rows_per_tile

        # Variant base [32P, q_free]: this tile's rows at free band [band_off : +rows_per_tile].
        v_data_base = sbm.alloc_stack((tp.p_per_block, tp.q_free), dtype=nl.float8_e4m3fn_x4)
        v_scale_base = sbm.alloc_stack((tp.p_per_block, tp.q_free), dtype=nl.uint8)
        nisa.memset(v_data_base, value=0, engine=nisa.gpsimd_engine)
        nisa.memset(v_scale_base, value=0, engine=nisa.gpsimd_engine)
        nisa.tensor_copy(
            dst=v_data_base[:, nl.ds(band_off, rows_per_tile)], src=q_mx_data_base, engine=nisa.vector_engine
        )
        nisa.tensor_copy(
            dst=v_scale_base[:, nl.ds(band_off, rows_per_tile)], src=q_mx_scale_base, engine=nisa.vector_engine
        )

        # Replicate across 4 quadrants → [128P, q_free]
        v_data = sbm.alloc_stack((TC.p_max, tp.q_free), dtype=nl.float8_e4m3fn_x4)
        v_scale = sbm.alloc_stack((TC.p_max, tp.q_free), dtype=nl.uint8)
        for quadrant_idx in range(tp.folds_per_chunk):
            p_off = quadrant_idx * tp.p_per_block
            nisa.tensor_copy(dst=v_data[nl.ds(p_off, tp.p_per_block), :], src=v_data_base, engine=nisa.vector_engine)
            nisa.tensor_copy(dst=v_scale[nl.ds(p_off, tp.p_per_block), :], src=v_scale_base, engine=nisa.vector_engine)

        variants.append((v_data, v_scale))

    return variants


# ── Batch Block KV Cache Loader ─────────────────────────────────────────────────────
class BatchBlockKVCacheLoader(nl.NKIObject):
    """Manages block table state for indirect DMA access to block-sparse KV cache.

    Computes a full [128, blocks_per_fold] offset vector and gathers all blocks
    of a chunk with a single indirect DMA over TC.p_max * blocks_per_fold rows,
    instead of issuing one indirect DMA per block.

    Lifecycle:
        1. Constructed once per batch with that batch's [num_blocks] table row:
           loads and pre-arranges it in SBUF, replicated across the four quadrants.
        2. Per chunk: call load_blocks(buf, cache_prior, chunk_idx) — replicates the
           chunk's block IDs, computes vector offsets, and DMA-loads blocks.
    """

    def __init__(self, active_blocks_row, cfg, tp: TileParams, sbm):
        self.sbm = sbm
        self.tp = tp

        """
        Row-within-block offsets [128, blocks_per_fold]. Each column holds, down the
        128 partitions, the pattern [0,0,0,0, 1,1,1,1, ..., 31,31,31,31] — i.e.
        partition // blocks_per_fold, the row index within a 32-row block. Built by
        transposing an iota whose free sequence repeats each value blocks_per_fold times.
        """
        self.row_offsets = sbm.alloc_stack((TC.p_max, tp.blocks_per_fold), dtype=nl.uint32)
        row_offsets_psum = nl.ndarray((TC.p_max, tp.blocks_per_fold), dtype=nl.float32, buffer=nl.psum)
        iota_sb = nl.ndarray((tp.blocks_per_fold, TC.p_max), dtype=nl.float32)
        nisa.iota(iota_sb, [[1, TC.p_per_quadrant], [0, tp.blocks_per_fold]])

        nisa.nc_transpose(row_offsets_psum, iota_sb, nisa.engine.tensor)
        nisa.tensor_copy(self.row_offsets.view(nl.uint32), row_offsets_psum)

        """
        Pre-arrange the block table row (fold-in-chunk on partitions, block-in-fold
        on columns grouped chunk-major) and replicate it across the four quadrants —
        partitions [32*q : 32*q+blocks_per_fold] of each quadrant q hold the same table.
        """
        folds_count = active_blocks_row.shape[0] // tp.blocks_per_fold
        chunks_count = folds_count // tp.folds_per_chunk
        self.table_sb = sbm.alloc_stack((TC.p_max, folds_count), dtype=nl.int32)
        nisa.dma_copy(
            self.table_sb[nl.ds(0, tp.blocks_per_fold)],
            active_blocks_row.reshape((chunks_count, tp.folds_per_chunk, tp.blocks_per_fold)).permute((1, 0, 2)),
        )
        for quadrant_idx in range(1, TC.p_max // TC.p_per_quadrant):
            nisa.tensor_copy(
                self.table_sb[nl.ds(TC.p_per_quadrant * quadrant_idx, tp.blocks_per_fold)],
                self.table_sb[nl.ds(0, tp.blocks_per_fold)],
            )

    def load_blocks(self, buf, cache_prior, chunk_idx):
        """Compute vector offsets for chunk_idx and DMA-load blocks into buf [TC.p_max, folds_per_chunk, packed_cols].

        Args:
            buf: [TC.p_max, folds_per_chunk, packed_cols] float32 in SBUF. Pre-allocated destination.
            cache_prior: [num_blocks, 32, 160] float32 in HBM. K or V cache.
            chunk_idx: Which chunk to load (0-indexed).
        """
        tp = self.tp
        vector_offsets = self._prep_vector_offsets(chunk_idx)

        """
        Flatten the block cache to a 2D [num_blocks * 32, packed_cols] row grid so a single
        indirect DMA can gather all rows of the chunk: vector_offsets supplies one absolute
        row index per destination partition (indirect_dim=0), the outer pattern dim walks
        TC.p_max * blocks_per_fold gathered rows, and the inner dim copies packed_cols
        contiguous columns per row.
        """
        cache_prior_2d = cache_prior.reshape((cache_prior.shape[0] * TC.p_per_quadrant, tp.packed_cols))

        nisa.dma_copy(
            dst=buf,
            src=cache_prior_2d.ap(
                [[tp.packed_cols, TC.p_max * tp.blocks_per_fold], [1, tp.packed_cols]],
                offset=0,
                vector_offset=vector_offsets,
                indirect_dim=0,
            ),
            dge_mode=nisa.dge_mode.swdge,
        )

    def _prep_vector_offsets(self, chunk_idx):
        """Compute per-block row offsets: block_base * rows_per_block + row_within_block."""
        tp = self.tp

        row_bases = self._prep_chunk_table(chunk_idx)

        out_sb = self.sbm.alloc_stack((TC.p_max, tp.blocks_per_fold), dtype=nl.uint32)
        nisa.scalar_tensor_tensor(
            out_sb, row_bases, nl.multiply, operand0=float(TC.p_per_quadrant), op1=nl.add, operand1=self.row_offsets
        )
        return out_sb

    def _prep_chunk_table(self, chunk_idx):
        """Replicate this chunk's blocks_per_fold block IDs down all 32 rows of each quadrant."""
        tp = self.tp
        row_bases = self.sbm.alloc_stack((TC.p_max, tp.blocks_per_fold), dtype=nl.int32)

        fold_blocks = []
        for block_idx in range(tp.blocks_per_fold):
            fold_blocks.append(block_idx)
        shuffle_pattern = fold_blocks * (TC.p_per_quadrant // tp.blocks_per_fold)
        nisa.nc_stream_shuffle(
            row_bases, self.table_sb[:, nl.ds(tp.blocks_per_fold * chunk_idx, tp.blocks_per_fold)], shuffle_pattern
        )
        return row_bases


# ── MM1/MM2: Block Matmul ─────────────────────────────────────────────────────
def _emit_block_matmul(tile_variants, kv_buf, psum_full, fold_idx, block_idx, tp):
    """One nc_matmul_mx: Q × kv_buf block → one PSUM tile of psum_full.

    Addresses logical block `fold_idx * blocks_per_fold + block_idx` of the chunk:
    block_idx selects the quadrant on partitions, and fold_idx splits into the column
    group and the partition band, which also picks the Q variant (of the current
    row-tile's variant list) that lands scores in that band. See the module docstring's
    Score Layout.
    """
    group_idx, band_idx = divmod(fold_idx, tp.variants_per_tile)
    row_offset = block_idx * tp.p_per_block
    tile_offset = group_idx * tp.fold_len + block_idx * tp.block_len
    scale_partition_count = TC.p_per_quadrant // TC.mx_group_partitions

    q_data, q_scale = tile_variants[band_idx]
    kv_block = kv_buf[nl.ds(row_offset, tp.p_per_block), fold_idx, :]

    psum_tile = psum_full[:, nl.ds(tile_offset, tp.block_len)]
    q_slice = q_data[nl.ds(row_offset, tp.p_per_block), :]
    q_scale_slice = q_scale[nl.ds(row_offset, scale_partition_count), :]
    k_slice = kv_block[:, nl.ds(0, tp.block_len)].view(nl.float8_e4m3fn_x4)
    k_scale_slice = kv_block[nl.ds(0, scale_partition_count), nl.ds(tp.block_len, tp.block_len // 4)].view(nl.uint8)

    nisa.nc_matmul_mx(
        dst=psum_tile,
        stationary=q_slice,
        moving=k_slice,
        stationary_scale=q_scale_slice,
        moving_scale=k_scale_slice,
        tile_position=(row_offset, 0),
        tile_size=(tp.p_per_block, TC.p_max),
    )


def _mm1_compute_chunk(tile_variants, k_buf, score_sb, score_max_sb, chunk_mask_sb, tp):
    """Compute MM1 (Q x K^T) with packed-Q eviction for one row-tile (SBUF scope).

    tile_variants: variants_per_tile (data, scale) Q buffers for the current row-tile.
    k_buf: [TC.p_max, folds_per_chunk, packed_cols] — already loaded with K blocks from HBM.
    score_sb: [TC.p_max, tp.score_free] bf16 — output scores buffer.
    score_max_sb: [TC.p_max, 1] fp32 — per-partition score max, fused into eviction.
    chunk_mask_sb: [TC.p_max, tp.score_free] uint8 — mask for this chunk (this row-tile).

    Q variants are [128P, q_free] (32P base replicated across 4 quadrants). Each
    variant's matmul lands in a rows_per_tile-row output-partition band via its
    zero-filled stationary free dim, so variants_per_tile blocks accumulate into one
    [128P, block_len] PSUM tile (1 block/tile for rows_per_tile=128, 2 for 64, 4 for 32).
    """
    psum_full = nl.ndarray((TC.p_max, tp.score_free), dtype=nl.bfloat16, buffer=nl.psum)

    """
    Tiling: one block matmul per (fold, block-in-fold) pair, for a total of
    folds_per_chunk * blocks_per_fold (16) matmuls per chunk.
    """
    for fold_idx in range(tp.folds_per_chunk):
        for block_idx in range(tp.blocks_per_fold):
            _emit_block_matmul(tile_variants, k_buf, psum_full, fold_idx, block_idx, tp)

    nisa.select_reduce(
        dst=score_sb,
        predicate=chunk_mask_sb,
        on_true=psum_full,
        on_false=-_SCORE_MASK_INF,
        reduce_res=score_max_sb,
        reduce_op=nl.maximum,
        reduce_cmd=nisa.reduce_cmd.reset_reduce,
    )


def _validate_inputs(q, k_active, v_active, k_prior, v_prior, mask, identity_hbm, active_blocks_table):
    """Validate shapes and dtypes of every operand. All config is derived from q / k_prior shapes.

    Scalar-range invariants (q_head in {32, 64}, s_active in {1, 2, 4, 8}) are enforced in
    AttnMXFP8Config.__post_init__; this checks cross-operand shape and dtype consistency.
    """
    bs, q_head, s_active, d_head = q.shape
    num_blocks = k_prior.shape[0]
    p_per_block = TC.p_per_quadrant
    packed_cols = d_head + d_head // 4
    bucket_size = num_blocks * d_head

    kernel_assert(d_head == 128, f"q head dim must be 128, got {d_head=} from {q.shape=}")
    kernel_assert(q.dtype == nl.bfloat16, f"q must be bf16, got {q.dtype=}")

    # k_active / v_active: active-token K/V [bs, s_active, d_head] bf16. s_active is derived from
    # q.shape[2], so a mismatched active K/V would silently read wrong data.
    kernel_assert(
        tuple(k_active.shape) == (bs, s_active, d_head) and k_active.dtype == nl.bfloat16,
        f"k_active must be [bs, s_active, d_head]={(bs, s_active, d_head)} bf16, "
        f"got {k_active.shape=}, {k_active.dtype=}",
    )
    kernel_assert(
        tuple(v_active.shape) == (bs, s_active, d_head) and v_active.dtype == nl.bfloat16,
        f"v_active must be [bs, s_active, d_head]={(bs, s_active, d_head)} bf16, "
        f"got {v_active.shape=}, {v_active.dtype=}",
    )

    # k_prior / v_prior: MXFP8 block cache [num_blocks, p_per_block, packed_cols] fp32.
    kernel_assert(
        tuple(k_prior.shape) == (num_blocks, p_per_block, packed_cols) and k_prior.dtype == nl.float32,
        f"k_prior must be [num_blocks, {p_per_block}, {packed_cols}] fp32, got {k_prior.shape=}, {k_prior.dtype=}",
    )
    kernel_assert(
        tuple(v_prior.shape) == (num_blocks, p_per_block, packed_cols) and v_prior.dtype == nl.float32,
        f"v_prior must be [num_blocks, {p_per_block}, {packed_cols}] fp32, got {v_prior.shape=}, {v_prior.dtype=}",
    )

    # mask: unified [bs, q_head, s_active, s_prior + s_active] uint8. s_prior = mask.shape[3] - s_active
    # is independent of bucket_size (tokens past s_prior are masked); it must fit within the KV cache.
    kernel_assert(
        len(mask.shape) == 4
        and mask.shape[0] == bs
        and mask.shape[1] == q_head
        and mask.shape[2] == s_active
        and s_active <= mask.shape[3] <= bucket_size + s_active
        and mask.dtype == nl.uint8,
        f"mask must be [bs, q_head, s_active, s_prior + s_active] uint8 with s_prior <= {bucket_size}, "
        f"got {mask.shape=}, {mask.dtype=}, {(bs, q_head, s_active)=}",
    )

    # identity_hbm: [>=128, >=128] bf16 identity for PE reduction (required despite the Optional default).
    kernel_assert(
        identity_hbm != None
        and len(identity_hbm.shape) == 2
        and identity_hbm.shape[0] >= TC.p_max
        and identity_hbm.shape[1] >= TC.p_max
        and identity_hbm.dtype == nl.bfloat16,
        f"identity_hbm must be a [>={TC.p_max}, >={TC.p_max}] bf16 tensor, "
        f"got {None if identity_hbm == None else (identity_hbm.shape, identity_hbm.dtype)}",
    )

    # active_blocks_table: [bs, num_blocks] int32 (required despite the Optional default).
    kernel_assert(
        active_blocks_table != None
        and tuple(active_blocks_table.shape) == (bs, num_blocks)
        and active_blocks_table.dtype == nl.int32,
        f"active_blocks_table must be [bs, num_blocks]={(bs, num_blocks)} int32, "
        f"got {None if active_blocks_table == None else (active_blocks_table.shape, active_blocks_table.dtype)}",
    )


# ── Main Kernel ────────────────────────────────────────────────────────────────
@nki.jit
def attention_mxfp8_tkg(
    q: nl.NkiTensor,
    k_active: nl.NkiTensor,
    v_active: nl.NkiTensor,
    k_prior: nl.NkiTensor,
    v_prior: nl.NkiTensor,
    mask: nl.NkiTensor,
    identity_hbm: Optional[nl.NkiTensor] = None,
    active_blocks_table: Optional[nl.NkiTensor] = None,
    sbm: Optional[SbufManager] = None,
) -> nl.NkiTensor:
    """MXFP8 flash decode attention with separate KV blocks and packed-Q eviction.

    Token-generation (decode) attention over an MXFP8-quantized block KV cache on
    Trainium 3. Optimized for long contexts (bucket_size >= 2048 tokens, i.e. at
    least one full chunk); requires q_head in {32, 64} and d_head == 128.

    All configuration is derived from input tensor shapes:
        bs, q_head, s_active, d_head from q.shape = [bs, q_head, s_active, d_head]
        bucket_size from k_prior.shape = [num_blocks, 32, 160]

    Speculative decoding: s_active query tokens per head (1 for single-token decode).
    The [q_head, s_active] axes flatten head-major (query row = h*s_active + s) into
    n_query_rows = q_head * s_active query rows. When n_query_rows <= 128 the rows share one
    tile via packed-Q eviction; when n_query_rows > 128 they are split into n_query_rows/128
    row-tiles of 128 rows, each an independent attention problem over the same KV. Supported
    n_query_rows in {32, 64, 128, 256, 512}.

    Note: Tensor layouts differ from attention_tkg. This kernel uses H in the
    partition dim for packed-Q eviction, while attention_tkg uses d in partitions.

    Dimensions:
        B: Batch size.
        H: Number of query heads (32 or 64).
        s_active: Active sequence length (query tokens per head).
        d: Head dimension (must be 128).
        num_blocks: KV cache blocks; each block covers 128 tokens as [32, 160] MXFP8.
        num_chunks: bucket_size / 2048 online-softmax iterations.
        s_prior: Prior-context tokens covered by the mask. Tokens at or past s_prior
            have no mask entry and are treated as masked.
        score_free: Free-dim width of the per-chunk score buffer (2048 for band_p=128,
            1024 for band_p=64, 512 for band_p=32).

    Args:
        q: Query tensor [B, H, s_active, d] bfloat16.
        k_active: Active key [B, s_active, d] bfloat16.
        v_active: Active value [B, s_active, d] bfloat16.
        k_prior: MXFP8 K cache [num_blocks, 32, 160] float32. Each block = 128 tokens.
        v_prior: MXFP8 V cache [num_blocks, 32, 160] float32. Each block = 128 tokens.
        mask: Unified per-head token mask [B, H, s_active, s_prior + s_active] uint8.
            The s_prior prefix masks the prior context; the trailing s_active columns
            mask the active tokens (the caller encodes intra-active causality here).
        identity_hbm: [128, 128] bfloat16 identity matrix for PE reduction.
        active_blocks_table: Block indices [B, num_blocks] int32.
        sbm: Optional SbufManager for SBUF allocation. None = auto-alloc mode.

    Returns:
        out_hbm: [B, H, s_active, d] bfloat16 attention output.

    Pseudocode:
        for b in range(B):
            load Q, scale by 1/sqrt(d), quantize to MXFP8, build packed-Q variants
            init online softmax state (running_max, running_sum, acc)
            for chunk in chunks_of_this_NC:
                load mask + K/V — prior blocks via indirect DMA, or quantize k/v_active
                MM1: scores = Q x K^T (nc_matmul_mx, packed-Q eviction + fused max)
                online softmax: exp(scores - max), rescale acc, update running state
                requantize scores to MXFP8; MM2: acc += scores x V (nc_matmul_mx)
            LNC2 gather (if sharded); out = acc / running_sum; store to HBM
    """
    # Derive config from input shapes
    # q: [bs, q_head, s_active, d_head]; k_prior: [num_blocks, 32, 160], each block = d_head tokens
    d_head = 128
    num_blocks = k_prior.shape[0]
    cfg = AttnMXFP8Config(q.shape[0], q.shape[1], num_blocks * d_head, d_head=d_head, s_active=q.shape[2])
    tp = TileParams(cfg)

    _validate_inputs(q, k_active, v_active, k_prior, v_prior, mask, identity_hbm, active_blocks_table)

    sbm = sbm if sbm != None else create_auto_alloc_manager()
    sbm.open_scope(name="mxfp8_attn")

    out_hbm = nl.ndarray((cfg.bs, cfg.q_head, cfg.s_active, cfg.d_head), dtype=nl.bfloat16, buffer=nl.shared_hbm)

    # Load identity matrix [128, 128] bf16 once (shared across batches)
    identity_sb = sbm.alloc_stack((TC.p_max, TC.p_max), dtype=nl.bfloat16)
    nisa.dma_copy(dst=identity_sb, src=identity_hbm[:, nl.ds(0, TC.p_max)])

    for batch_idx in range(cfg.bs):
        kv_loader = BatchBlockKVCacheLoader(active_blocks_table[batch_idx], cfg, tp, sbm)

        # Step 0: Load Q, scale, quantize to MXFP8, build per-row-tile packed variants
        q_bufs = QuantizedQ(cfg, tp, sbm)
        q_bufs.load_from_hbm(q, batch_idx)

        # Step 1: Initialize one online softmax state per row-tile (each holds its own
        # persistent PSUM accumulator across the chunk loop).
        sm_states = [SoftmaxState(cfg, tp, sbm, identity_sb) for _ in range(tp.n_row_tiles)]

        # Step 2: Chunk loop. K/V are loaded once per chunk and reused across all row-tiles
        # (inner row-tiling — this is a memory-bound decode kernel, so KV DMA is hoisted).
        # The active tokens are appended as a final "active chunk".
        owns_active = tp.sprior_prg_id == tp.sprior_n_prgs - 1
        chunks_this_nc = tp.chunks_per_nc + (1 if owns_active else 0)
        for chunk_local in range(chunks_this_nc):
            is_active = owns_active and chunk_local == tp.chunks_per_nc
            chunk_idx = tp.chunk_start + chunk_local
            # Allocated per iteration by design: buffers carry no cross-chunk state, so the
            # stack allocator hands back the same SBUF region each pass (reuse, not growth).
            cb = ChunkBuffers(tp, sbm)

            # Load this chunk's K/V once (shared across row-tiles)
            if is_active:
                _build_active_chunk_kv(cb, batch_idx, k_active, v_active, tp, sbm)
            else:
                _load_chunk_kv(cb, k_prior, v_prior, kv_loader, chunk_idx)

            for row_tile in range(tp.n_row_tiles):
                # Per-tile: scatter this tile's mask, then MM1 → softmax → requantize → MM2.
                # The score scratch buffers carry no cross-tile state, so they are reused.
                if is_active:
                    _load_active_chunk_mask(cb.chunk_mask_sb, batch_idx, row_tile, mask, tp)
                else:
                    _load_chunk_mask(cb.chunk_mask_sb, batch_idx, chunk_idx, row_tile, mask, tp)
                _mm1_compute_chunk(
                    q_bufs.q_variants[row_tile], cb.k_buf, cb.score_sb, cb.score_max_sb, cb.chunk_mask_sb, tp
                )
                sm_states[row_tile].update_online_softmax(cb.score_sb, cb.score_max_sb, cb.score_sb_fp32_reinterp)
                scores_data, scores_scale = _requantize_scores(cb.score_sb_fp32_reinterp, tp, sbm)
                _mm2_compute_chunk(scores_data, scores_scale, cb.v_buf, tp, sm_states[row_tile])

        # Step 3 per row-tile: normalize (LNC2 gather if sharded) and store.
        for row_tile in range(tp.n_row_tiles):
            _finalize_output(cfg, tp, sm_states[row_tile], sbm)
            _store_output_hbm(batch_idx, row_tile, tp, sm_states[row_tile], out_hbm)

    sbm.close_scope()  # mxfp8_attn

    return out_hbm


# ── Step 2 helpers ─────────────────────────────────────────────────────────────


def _swizzle_quantize_fp32_tile(fp32_src, fp32_offset, mx_data_dst, mx_scale_dst, sbm, h_fp32=256):
    """Swizzle+quantize one [T, h_fp32] fp32 tile from a larger fp32 tensor."""
    T = fp32_src.shape[0]
    h_fp32_half = h_fp32 // 2
    transposed_psum = nl.ndarray((h_fp32_half, T * 2), dtype=nl.float32, buffer=nl.psum)
    for stride_idx in range(2):
        nisa.nc_transpose(
            dst=transposed_psum.slice(dim=1, start=stride_idx, end=T * 2, step=2),
            data=fp32_src.slice(dim=1, start=fp32_offset + stride_idx, end=fp32_offset + h_fp32_half * 2, step=2),
        )
    swizzled_fp32 = sbm.alloc_stack((h_fp32_half, T * 2), dtype=nl.float32)
    nisa.tensor_copy(dst=swizzled_fp32, src=transposed_psum, engine=nisa.scalar_engine)
    nisa.quantize_mx(dst=mx_data_dst, src=swizzled_fp32.view(nl.bfloat16), dst_scale=mx_scale_dst)


def _requantize_scores(score_sb_fp32_reinterp, tp, sbm):
    """Re-quantize softmax scores from fp32 to MXFP8 (SBUF scope).

    score_sb_fp32_reinterp: [TC.p_max, tp.score_free // 2] fp32 in SBUF.
    Returns: (scores_data, scores_scale) MXFP8 tensors in SBUF.
    """
    n_tiles = tp.score_tiles_per_fold
    scores_data = sbm.alloc_stack((TC.p_max, n_tiles * TC.p_max), dtype=nl.float8_e4m3fn_x4)
    scores_scale = sbm.alloc_stack((TC.p_max, n_tiles * TC.p_max), dtype=nl.uint8)
    for tile_idx in range(n_tiles):
        _swizzle_quantize_fp32_tile(
            score_sb_fp32_reinterp,
            tile_idx * 256,
            scores_data[:, nl.ds(tile_idx * TC.p_max, TC.p_max)],
            scores_scale[:, nl.ds(tile_idx * TC.p_max, TC.p_max)],
            sbm,
        )
    return scores_data, scores_scale


def _mm2_compute_chunk(scores_data, scores_scale, v_buf, tp, sm_state):
    """Compute MM2: scores × V, accumulating into sm_state.acc (SBUF scope).

    v_buf: [TC.p_max, folds_per_chunk, packed_cols] — already loaded with V blocks from HBM.

    Tiling: one nc_matmul_mx per fold position (folds_per_chunk iterations).
    scores_data's free dim is laid out [tile][band][row], and since
    variants_per_tile * rows_per_tile == 128, the flat slice fold_idx * rows_per_tile
    walks the fold positions in the same order as v_buf's column bands. Each matmul
    contracts a band's rows_per_tile score columns (128 token partitions = folds_per_chunk
    quadrants) against the matching V fold, accumulating into this row-tile's acc.
    """
    score_free_per_band = tp.rows_per_tile
    scale_partition_count = TC.p_max // TC.mx_group_partitions

    for fold_idx in range(tp.folds_per_chunk):
        v_fold = v_buf[:, fold_idx, :]

        nisa.nc_matmul_mx(
            dst=sm_state.acc,
            stationary=scores_data[:, nl.ds(fold_idx * score_free_per_band, score_free_per_band)],
            moving=v_fold[:, nl.ds(0, tp.block_len)].view(nl.float8_e4m3fn_x4),
            stationary_scale=scores_scale[:, nl.ds(fold_idx * score_free_per_band, score_free_per_band)],
            moving_scale=v_fold[nl.ds(0, scale_partition_count), nl.ds(tp.block_len, tp.block_len // 4)].view(nl.uint8),
        )


# ── Step 3: Finalize Output (SBUF scope) ─────────────────────────────────────
def _finalize_output(cfg, tp, sm_state, sbm):
    """Copy the accumulator PSUM→SBUF, LNC2 gather (if sharded), normalize by softmax sum.

    The active tokens are now part of the chunk loop, so acc is fully accumulated in PSUM
    on entry; this is the single point where it is copied to SBUF for the scalar
    normalize/gather ops. Result stored in sm_state.out_bf16.
    """
    rows_per_tile = tp.rows_per_tile
    sm_state.acc_sb = sbm.alloc_stack((TC.p_max, cfg.d_head), dtype=nl.float32)
    nisa.tensor_copy(
        dst=sm_state.acc_sb[nl.ds(0, rows_per_tile), :],
        src=sm_state.acc[nl.ds(0, rows_per_tile), :],
        engine=nisa.scalar_engine,
    )

    if tp.sprior_n_prgs > 1:
        _lnc2_gather_and_normalize(cfg, tp, sm_state, sbm)
    else:
        _normalize_output(cfg, tp, sm_state, sbm)


def _store_output_hbm(batch_idx, row_tile, tp, sm_state, out_hbm):
    """Store one row-tile's normalized output to HBM (only NC 0 writes in LNC2 mode).

    out_hbm[batch_idx] is [q_head, s_active, d]; out_bf16 is [rows_per_tile, d] with row
    h*s_active + s (head-major). The full band reshapes to [n_query_rows, d]; this row-tile
    writes its slice [row_tile*rows_per_tile : +rows_per_tile].
    """
    if tp.sprior_prg_id == 0:
        rows_per_tile = tp.rows_per_tile
        row_base = row_tile * rows_per_tile
        out_band = out_hbm[batch_idx].reshape((tp.n_query_rows, out_hbm.shape[-1]))
        nisa.dma_copy(dst=out_band[nl.ds(row_base, rows_per_tile), :], src=sm_state.out_bf16)


def _normalize_output(cfg, tp, sm_state, sbm):
    """Compute output = acc / running_sum for this row-tile (single NC path), output as bf16."""
    rows_per_tile = tp.rows_per_tile
    inv_sum = sbm.alloc_stack((rows_per_tile, 1), dtype=nl.float32)
    nisa.activation(dst=inv_sum, op=nl.reciprocal, data=sm_state.running_sum[nl.ds(0, rows_per_tile), :])
    sm_state.out_bf16 = sbm.alloc_stack((rows_per_tile, cfg.d_head), dtype=nl.bfloat16)
    nisa.tensor_scalar(
        dst=sm_state.out_bf16, data=sm_state.acc_sb[nl.ds(0, rows_per_tile), :], op0=nl.multiply, operand0=inv_sum
    )


def _lnc2_gather_and_normalize(cfg, tp, sm_state, sbm):
    """Exchange acc and sum across NCs, rescale by correction factor, normalize (one row-tile)."""
    rows_per_tile = tp.rows_per_tile
    # Exchange running max to compute global max
    m_local = sbm.alloc_stack((rows_per_tile, 1), dtype=nl.float32)
    nisa.tensor_copy(dst=m_local, src=sm_state.running_max[nl.ds(0, rows_per_tile), :], engine=nisa.vector_engine)
    m_remote = sbm.alloc_stack((rows_per_tile, 1), dtype=nl.float32)
    nisa.sendrecv(
        src=m_local,
        dst=m_remote,
        send_to_rank=(1 - tp.sprior_prg_id),
        recv_from_rank=(1 - tp.sprior_prg_id),
        pipe_id=0,
    )
    m_global = sbm.alloc_stack((rows_per_tile, 1), dtype=nl.float32)
    nisa.tensor_tensor(dst=m_global, data1=m_local, data2=m_remote, op=nl.maximum)

    # Correction factor: exp(local_max - global_max)
    local_corr = sbm.alloc_stack((rows_per_tile, 1), dtype=nl.float32)
    nisa.tensor_tensor(dst=local_corr, data1=m_local, data2=m_global, op=nl.subtract)
    nisa.activation(dst=local_corr, op=nl.exp, data=local_corr)

    # Rescale local acc and sum
    acc_q = sm_state.acc_sb[nl.ds(0, rows_per_tile), :]
    nisa.tensor_scalar(dst=acc_q, data=acc_q, op0=nl.multiply, operand0=local_corr)
    l_local = sbm.alloc_stack((rows_per_tile, 1), dtype=nl.float32)
    nisa.tensor_scalar(
        dst=l_local, data=sm_state.running_sum[nl.ds(0, rows_per_tile), :], op0=nl.multiply, operand0=local_corr
    )

    # Exchange rescaled partials
    acc_recv = sbm.alloc_stack((rows_per_tile, cfg.d_head), dtype=nl.float32)
    nisa.sendrecv(
        src=acc_q, dst=acc_recv, send_to_rank=(1 - tp.sprior_prg_id), recv_from_rank=(1 - tp.sprior_prg_id), pipe_id=0
    )
    l_recv = sbm.alloc_stack((rows_per_tile, 1), dtype=nl.float32)
    nisa.sendrecv(
        src=l_local, dst=l_recv, send_to_rank=(1 - tp.sprior_prg_id), recv_from_rank=(1 - tp.sprior_prg_id), pipe_id=0
    )

    # Sum and normalize
    nisa.tensor_tensor(dst=acc_q, data1=acc_q, data2=acc_recv, op=nl.add)
    nisa.tensor_tensor(dst=l_local, data1=l_local, data2=l_recv, op=nl.add)

    inv_sum = sbm.alloc_stack((rows_per_tile, 1), dtype=nl.float32)
    nisa.activation(dst=inv_sum, op=nl.reciprocal, data=l_local)
    sm_state.out_bf16 = sbm.alloc_stack((rows_per_tile, cfg.d_head), dtype=nl.bfloat16)
    nisa.tensor_scalar(dst=sm_state.out_bf16, data=acc_q, op0=nl.multiply, operand0=inv_sum)
