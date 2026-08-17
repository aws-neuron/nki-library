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
  Step 0: Load Q, scale by 1/sqrt(d), quantize to MXFP8, build Q_lo/Q_hi variants
  Step 1: Initialize online softmax state (running_max, running_sum, acc)
  Step 2: For each chunk:
    2a: Load row indices + mask + K/V blocks via indirect DMA (swdge)
    2b: MM1 — Q × K^T with packed-Q eviction (nc_matmul_mx, row-tiled)
    2c: Online softmax — BF16 scores → exp → running max/sum update
    2d: Re-quantize scores to MXFP8 → MM2 — scores × V (nc_matmul_mx)
  Step 3: k_active/v_active — scalar softmax update (single-token dot product)
  Step 4: LNC2 gather (if sharded) → normalize (acc / running_sum) → store

Packed-Q Eviction
=================
Each Q x K block matmul fills only band_p (= q_head) output partitions, so Q is
replicated into variants_per_tile = 128 // q_head variants — variant v holds the
real Q in free band [v*band_p : (v+1)*band_p] and zeros elsewhere, landing its
scores in output-partition band v. variants_per_tile blocks — one per fold sharing a
column group, all at the same block-in-fold position — then accumulate into a single
[128P, block_len] PSUM tile, using all 128 partitions:
    q_head=64 → 2 variants (Q_lo, Q_hi), 2 blocks per tile
    q_head=32 → 4 variants, 4 blocks per tile

Score Layout: [128P, score_free], shared by MM1's PSUM buffer and the evicted SBUF scores
    score_free = folds_per_chunk * block_len * score_tiles_per_fold (1024 for q_head=64,
    512 for q_head=32). Logical block b of the chunk is
    fold_idx, block_idx = divmod(b, blocks_per_fold), and its scores land at
        partition rows [band_idx * band_p : (band_idx + 1) * band_p]
        free columns  [group_idx * fold_len + block_idx * block_len : + block_len]
    where group_idx, band_idx = divmod(fold_idx, variants_per_tile). A column group is
    fold_len wide and holds variants_per_tile folds, one per partition band.

    A fold's blocks differ only in the block_idx term, so each fold's fold_len tokens
    occupy fold_len contiguous columns of a single band, in token order — which is what
    lets _load_chunk_mask scatter the token-sequential mask one fold per DMA.

Mask Layout: [B, H, 1, s_prior] uint8
    User-provided per-head mask in token-sequential order.
    1 = valid token, 0 = masked (score set to -inf).
"""

from dataclasses import dataclass
from typing import Optional

import nki
import nki.isa as nisa
import nki.language as nl

from ...core.utils.allocator import SbufManager, create_auto_alloc_manager
from ...core.utils.kernel_assert import kernel_assert
from ...core.utils.stream_shuffle_broadcast import stream_shuffle_broadcast
from .attention_mxfp8_tkg_utils import mm1_packing_geometry, swizzle_quantize_mx

# bf16 lowest representable value; used to fill masked-out score positions with -inf.
_SCORE_MASK_NEG_INF = -65504.0
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

    def __post_init__(self):
        kernel_assert(self.bs >= 1, f"bs must be >= 1, got {self.bs=}")
        kernel_assert(self.q_head in (32, 64), f"q_head must be 32 or 64, got {self.q_head=}")
        kernel_assert(self.d_head == 128, f"d_head must be 128, got {self.d_head=}")


class TileParams(nl.NKIObject):
    """Derived tiling, sharding, and geometry parameters computed once at kernel entry."""

    def __init__(self, cfg: AttnMXFP8Config):
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

        # Packed-Q eviction geometry (native per-head-count variant packing)
        band_p, variants_per_tile, score_tiles_per_fold = mm1_packing_geometry(
            cfg.q_head, p_max=TC.p_max, blocks_per_fold=self.blocks_per_fold
        )
        self.band_p = band_p
        """Partition rows one block's scores occupy (= q_head)."""
        self.variants_per_tile = variants_per_tile
        """Q variants packed into one PSUM tile (2 for q_head=64, 4 for q_head=32)."""
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
    """MXFP8-quantized Q with packed variants for MM1 and scaled bf16 for active-token dot product."""

    def __init__(self, cfg, tp, sbm):
        self.cfg = cfg
        self.tp = tp
        self.sbm = sbm
        self.q_scaled = None
        self.q_variants = None
        """List of variants_per_tile (data, scale) MXFP8 buffers. Each buffer is
        [128P, p_max F]: the partition dim is the 32-partition MXFP8 Q operand
        replicated across the four fold quadrants, and the free dim is the packed
        output-partition layout (variants_per_tile * band_p). Variant v writes the
        real Q into free band [v*band_p : (v+1)*band_p] (zeros elsewhere) so its
        block matmul lands in output-partition band v."""

    def load_from_hbm(self, q_hbm, batch_idx):
        """Load Q from HBM, scale, quantize to MXFP8, build packed variants.

        Q is loaded into a band_p-tall (= real q_head) buffer with no padding: the
        packed-Q eviction stacks variants_per_tile blocks per PSUM tile, one per
        band_p-row partition band, so all 128 partitions carry real scores.
        """
        cfg, tp, sbm = self.cfg, self.tp, self.sbm

        q_bf16 = sbm.alloc_stack((tp.band_p, cfg.d_head), dtype=nl.bfloat16)
        nisa.dma_copy(dst=q_bf16, src=q_hbm[batch_idx, :, nl.ds(0, 1), :])

        self.q_scaled = sbm.alloc_stack((tp.band_p, cfg.d_head), dtype=nl.bfloat16)
        nisa.tensor_scalar(dst=self.q_scaled, data=q_bf16, op0=nl.multiply, operand0=tp.softmax_scale)

        # MXFP8 Q operand: contraction dim d_head//4 = 32 on partitions, band_p heads on free.
        mx_par = cfg.d_head // 4  # 32
        q_mx_data_base = sbm.alloc_stack((mx_par, tp.band_p), dtype=nl.float8_e4m3fn_x4)
        q_mx_scale_base = sbm.alloc_stack((mx_par, tp.band_p), dtype=nl.uint8)
        swizzle_quantize_mx(self.q_scaled, q_mx_data_base, q_mx_scale_base, sbm)

        self.q_variants = _build_q_variants(q_mx_data_base, q_mx_scale_base, cfg, tp, sbm)


class SoftmaxState(nl.NKIObject):
    """Online softmax running state and accumulators.

    Constructed once per batch iteration. Holds running max/sum, the PSUM
    accumulator, and the identity matrix needed for PE-based sum reduction.
    """

    def __init__(self, cfg, tp, sbm, identity_sb):
        self.cfg = cfg
        self.tp = tp
        self.sbm = sbm
        self.band_p = tp.band_p  # partition rows per band (= q_head)
        self.n_bands = tp.variants_per_tile  # bands packed per PSUM tile (2 or 4)

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

        self.acc = nl.ndarray((tp.band_p, cfg.d_head), dtype=nl.float32, buffer=nl.psum)
        """[band_p, d_head] fp32 in PSUM."""
        nisa.memset(self.acc, value=0.0)

        # Set later in Steps 3/4
        self.acc_sb = None
        """[TC.p_max, d_head] fp32 in SBUF."""
        self.out_bf16 = None
        """[q_head, d_head] bf16 in SBUF."""

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
        """

        kernel_assert(self.n_bands in (2, 4), f"n_bands assumed to be either 2 or 4, got {self.n_bands}")

        sbm = self.sbm
        band_p = self.band_p

        reduced = sbm.alloc_stack((band_p, 1), dtype=nl.float32)
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
    chunk_mask_sb: nl.NkiTensor, batch_idx: int, chunk_idx: int, mask: nl.NkiTensor, tp: TileParams
) -> None:
    """Load mask [B, H, 1, s_prior] and scatter into eviction-layout SBUF buffer.

    One DMA per fold: a fold's fold_len tokens occupy fold_len contiguous columns of a
    single partition band, in token order, so the token-sequential mask needs no
    reordering within a fold — see the module docstring's Score Layout. The folds tile
    the whole buffer. Tokens at or past s_prior have no mask entry, so a chunk reaching
    past s_prior copies only the in-range prefix of each fold; the memset zeroes the
    remaining columns, which no DMA writes and which would otherwise hold the previous
    chunk's mask.
    """
    s_prior = mask.shape[3]

    if (chunk_idx + 1) * tp.chunk_tokens > s_prior:
        nisa.memset(chunk_mask_sb, value=0)

    for fold_idx in range(tp.folds_per_chunk):
        tok_start = (chunk_idx * tp.folds_per_chunk + fold_idx) * tp.fold_len
        n_copy = min(tp.fold_len, s_prior - tok_start)
        if n_copy <= 0:
            continue
        group_idx, band_idx = divmod(fold_idx, tp.variants_per_tile)
        p_row = band_idx * tp.band_p
        f_off = group_idx * tp.fold_len
        nisa.dma_copy(
            dst=chunk_mask_sb[nl.ds(p_row, tp.band_p), nl.ds(f_off, n_copy)],
            src=mask[batch_idx, nl.ds(0, tp.band_p), 0, nl.ds(tok_start, n_copy)],
        )


def _load_chunk_context(cb, batch_idx, chunk_idx, mask, k_prior, v_prior, kv_loader, tp):
    """Load all chunk data from HBM into pre-allocated ChunkBuffers."""
    _load_chunk_mask(cb.chunk_mask_sb, batch_idx, chunk_idx, mask, tp)
    kv_loader.load_blocks(cb.k_buf, k_prior, chunk_idx)
    kv_loader.load_blocks(cb.v_buf, v_prior, chunk_idx)


# ── Q Variant Construction ─────────────────────────────────────────────────────
def _build_q_variants(q_mx_data_base, q_mx_scale_base, cfg, tp, sbm):
    """Build variants_per_tile packed Q buffers from base MXFP8 Q [32P, band_p F].

    Variant v holds the base Q in free band [v*band_p : (v+1)*band_p] and zeros
    elsewhere, so its Q x K block matmul lands in output-partition band v. Each
    variant's [32P, q_free] base is then replicated across the four fold quadrants
    to [128P, q_free].

    For q_head=64 this yields the original two variants (Q_lo, Q_hi); for q_head=32
    it yields four, packing all 128 output partitions.

    Returns:
        List of variants_per_tile (data, scale) tuples, each [128P, q_free].
    """
    variants = []
    for variant_idx in range(tp.variants_per_tile):
        band_off = variant_idx * tp.band_p

        # Variant base [32P, q_free]: base Q at free band [band_off : band_off + band_p].
        v_data_base = sbm.alloc_stack((tp.p_per_block, tp.q_free), dtype=nl.float8_e4m3fn_x4)
        v_scale_base = sbm.alloc_stack((tp.p_per_block, tp.q_free), dtype=nl.uint8)
        nisa.memset(v_data_base, value=0, engine=nisa.gpsimd_engine)
        nisa.memset(v_scale_base, value=0, engine=nisa.gpsimd_engine)
        nisa.tensor_copy(dst=v_data_base[:, nl.ds(band_off, tp.band_p)], src=q_mx_data_base, engine=nisa.vector_engine)
        nisa.tensor_copy(
            dst=v_scale_base[:, nl.ds(band_off, tp.band_p)], src=q_mx_scale_base, engine=nisa.vector_engine
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
def _emit_block_matmul(q_bufs, kv_buf, psum_full, fold_idx, block_idx, tp):
    """One nc_matmul_mx: Q × kv_buf block → one PSUM tile of psum_full.

    Addresses logical block `fold_idx * blocks_per_fold + block_idx` of the chunk:
    block_idx selects the quadrant on partitions, and fold_idx splits into the column
    group and the partition band, which also picks the Q variant that lands scores in
    that band. See the module docstring's Score Layout.

    Access-pattern decode (kv_buf is [TC.p_max, folds_per_chunk, packed_cols], each block
    laid out as [block_len data | block_len//4 scale] fp32 columns):
      - `moving` is the block_len data columns viewed as float8_e4m3fn_x4 (same width).
      - `moving_scale` is the block_len//4 scale columns viewed as uint8 (4x wider →
        block_len columns), over p_per_quadrant // mx_group_partitions rows.
    """
    group_idx, band_idx = divmod(fold_idx, tp.variants_per_tile)
    row_offset = block_idx * tp.p_per_block
    tile_offset = group_idx * tp.fold_len + block_idx * tp.block_len
    scale_partition_count = TC.p_per_quadrant // TC.mx_group_partitions

    q_data, q_scale = q_bufs.q_variants[band_idx]
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


def _mm1_compute_chunk(q_bufs, k_buf, score_sb, score_max_sb, chunk_mask_sb, tp):
    """Compute MM1 (Q x K^T) with packed-Q eviction (SBUF scope).

    k_buf: [TC.p_max, folds_per_chunk, packed_cols] — already loaded with K blocks from HBM.
    score_sb: [TC.p_max, tp.score_free] bf16 — output scores buffer.
    score_max_sb: [TC.p_max, 1] fp32 — per-partition score max, fused into eviction.
    chunk_mask_sb: [TC.p_max, tp.score_free] uint8 — mask for this chunk.

    Q variants are [128P, q_free] (32P base replicated across 4 quadrants). Each
    variant's matmul lands in a band_p-row output-partition band via its zero-filled
    stationary free dim, so variants_per_tile blocks accumulate into one [128P,
    block_len] PSUM tile (2 blocks/tile for q_head=64, 4 for q_head=32).
    """
    psum_full = nl.ndarray((TC.p_max, tp.score_free), dtype=nl.bfloat16, buffer=nl.psum)

    """
    Tiling: one block matmul per (fold, block-in-fold) pair, for a total of
    folds_per_chunk * blocks_per_fold (16) matmuls per chunk.
    """
    for fold_idx in range(tp.folds_per_chunk):
        for block_idx in range(tp.blocks_per_fold):
            _emit_block_matmul(q_bufs, k_buf, psum_full, fold_idx, block_idx, tp)

    nisa.select_reduce(
        dst=score_sb,
        predicate=chunk_mask_sb,
        on_true=psum_full,
        on_false=_SCORE_MASK_NEG_INF,
        reduce_res=score_max_sb,
        reduce_op=nl.maximum,
        reduce_cmd=nisa.reduce_cmd.reset_reduce,
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
        bs, q_head, d_head from q.shape = [bs, q_head, 1, d_head]
        bucket_size from k_prior.shape = [num_blocks, 32, 160]

    Note: Tensor layouts differ from attention_tkg. This kernel uses H in the
    partition dim for packed-Q eviction, while attention_tkg uses d in partitions.

    Dimensions:
        B: Batch size.
        H: Number of query heads (32 or 64).
        d: Head dimension (must be 128).
        num_blocks: KV cache blocks; each block covers 128 tokens as [32, 160] MXFP8.
        num_chunks: bucket_size / 2048 online-softmax iterations.
        s_prior: Prior-context tokens covered by the mask. Tokens at or past s_prior
            have no mask entry and are treated as masked.
        score_free: Free-dim width of the per-chunk score buffer (1024 for H=64,
            512 for H=32).

    Args:
        q: Query tensor [B, H, 1, d] bfloat16.
        k_active: Active key [B, d] bfloat16.
        v_active: Active value [B, d] bfloat16.
        k_prior: MXFP8 K cache [num_blocks, 32, 160] float32. Each block = 128 tokens.
        v_prior: MXFP8 V cache [num_blocks, 32, 160] float32. Each block = 128 tokens.
        mask: Per-head token mask [B, H, 1, s_prior] uint8.
        identity_hbm: [128, 128] bfloat16 identity matrix for PE reduction.
        active_blocks_table: Block indices [B, num_blocks] int32.
        sbm: Optional SbufManager for SBUF allocation. None = auto-alloc mode.

    Returns:
        out_hbm: [B, H, d] bfloat16 attention output.

    Pseudocode:
        for b in range(B):
            load Q, scale by 1/sqrt(d), quantize to MXFP8, build Q_lo/Q_hi variants
            init online softmax state (running_max, running_sum, acc)
            for chunk in chunks_of_this_NC:
                load mask + K/V blocks via indirect DMA
                MM1: scores = Q x K^T (nc_matmul_mx, packed-Q eviction + fused max)
                online softmax: exp(scores - max), rescale acc, update running state
                requantize scores to MXFP8; MM2: acc += scores x V (nc_matmul_mx)
            scalar softmax update with k_active/v_active (single-token dot product)
            LNC2 gather (if sharded); out = acc / running_sum; store to HBM
    """
    # Derive config from input shapes
    # k_prior: [num_blocks, 32, 160], each block = d_head tokens
    d_head = 128
    num_blocks = k_prior.shape[0]
    cfg = AttnMXFP8Config(q.shape[0], q.shape[1], num_blocks * d_head, d_head=d_head)
    tp = TileParams(cfg)

    # mask.shape[3] (s_prior) drives the per-chunk copy arithmetic in _load_chunk_mask
    kernel_assert(
        len(mask.shape) == 4 and mask.shape[0] == cfg.bs and mask.shape[1] == cfg.q_head and mask.shape[2] == 1,
        f"mask must be [bs, q_head, 1, s_prior], got {mask.shape=}, {cfg.bs=}, {cfg.q_head=}",
    )

    sbm = sbm if sbm != None else create_auto_alloc_manager()
    sbm.open_scope(name="mxfp8_attn")

    out_hbm = nl.ndarray((cfg.bs, cfg.q_head, cfg.d_head), dtype=nl.bfloat16, buffer=nl.shared_hbm)

    # Load identity matrix [128, 128] bf16 once (shared across batches)
    identity_sb = sbm.alloc_stack((TC.p_max, TC.p_max), dtype=nl.bfloat16)
    nisa.dma_copy(dst=identity_sb, src=identity_hbm[:, nl.ds(0, TC.p_max)])

    for batch_idx in range(cfg.bs):
        kv_loader = BatchBlockKVCacheLoader(active_blocks_table[batch_idx], cfg, tp, sbm)

        # Step 0: Load Q, scale, quantize to MXFP8, build packed variants
        q_bufs = QuantizedQ(cfg, tp, sbm)
        q_bufs.load_from_hbm(q, batch_idx)

        # Step 1: Initialize online softmax running state
        sm_state = SoftmaxState(cfg, tp, sbm, identity_sb)

        # Step 2: Chunk loop — load → MM1 → softmax → MM2
        for chunk_local in range(tp.chunks_per_nc):
            chunk_idx = tp.chunk_start + chunk_local
            # Allocated per iteration by design: buffers carry no cross-chunk state, so the
            # stack allocator hands back the same SBUF region each pass (reuse, not growth).
            cb = ChunkBuffers(tp, sbm)

            # Load all chunk data from HBM
            _load_chunk_context(cb, batch_idx, chunk_idx, mask, k_prior, v_prior, kv_loader, tp)

            # Compute MM1 → softmax → requantize → MM2
            _mm1_compute_chunk(q_bufs, cb.k_buf, cb.score_sb, cb.score_max_sb, cb.chunk_mask_sb, tp)
            sm_state.update_online_softmax(cb.score_sb, cb.score_max_sb, cb.score_sb_fp32_reinterp)
            scores_data, scores_scale = _requantize_scores(cb.score_sb_fp32_reinterp, tp, sbm)
            _mm2_compute_chunk(scores_data, scores_scale, cb.v_buf, tp, sm_state)

        # Step 3: Load k_active/v_active (HBM scope), then compute (SBUF scope)
        k_active_sb, v_active_sb = _load_kv_active(batch_idx, k_active, v_active, cfg, sbm)
        _compute_active_tokens_sbuf(q_bufs, k_active_sb, v_active_sb, sm_state, cfg, sbm)

        # Step 4: Normalize (SBUF scope), then store output (HBM scope)
        _finalize_output(cfg, tp, sm_state, sbm)
        _store_output_hbm(batch_idx, tp, sm_state, out_hbm)

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
    scores_data's free dim is laid out [tile][band][head], and since
    variants_per_tile * band_p == 128, the flat slice fold_idx * band_p walks
    the fold positions in the same order as v_buf's column bands. Each matmul
    contracts a band's band_p score columns (128 token partitions = folds_per_chunk
    quadrants) against the matching V fold, accumulating into the shared acc tile.

    Access-pattern decode (same [data | scale] block layout as MM1):
      - v_fold selects fold fold_idx of v_buf [TC.p_max, folds_per_chunk, packed_cols].
      - `moving` is the block_len V data columns viewed as float8_e4m3fn_x4.
      - `moving_scale` is the block_len//4 scale columns viewed as uint8 (4x wider).
    """
    score_free_per_band = tp.band_p
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


# ── Step 3: Active-token helpers ──────────────────────────────────────────────


def _compute_k_active_score(q_scaled_sb, k_active_sb, cfg, sbm):
    """Compute score_new = Q_scaled · k_active → [q_heads, 1]."""
    k_broadcast = sbm.alloc_stack((cfg.q_head, cfg.d_head), dtype=nl.bfloat16)
    stream_shuffle_broadcast(src=k_active_sb, dst=k_broadcast)
    qk = sbm.alloc_stack((cfg.q_head, cfg.d_head), dtype=nl.float32)
    nisa.tensor_tensor(dst=qk, data1=q_scaled_sb[nl.ds(0, cfg.q_head), :], data2=k_broadcast, op=nl.multiply)
    score_new = sbm.alloc_stack((cfg.q_head, 1), dtype=nl.float32)
    nisa.tensor_reduce(dst=score_new, op=nl.add, data=qk, axis=1)
    return score_new


def _softmax_update_scalar(score_new, running_max, running_sum, running_out, cfg, sbm):
    """Update online softmax with a single score per head. Returns exp_new.

    Operates only on the top half [0:q_head]; downstream finalize reads just that
    half, so the packed bottom-half replica is never needed here.
    """
    HALF_P = cfg.q_head
    rmax = running_max[nl.ds(0, HALF_P), :]
    rsum = running_sum[nl.ds(0, HALF_P), :]
    m_new = sbm.alloc_stack((HALF_P, 1), dtype=nl.float32)
    nisa.tensor_tensor(dst=m_new, data1=rmax, data2=score_new, op=nl.maximum)
    correction = sbm.alloc_stack((HALF_P, 1), dtype=nl.float32)
    nisa.tensor_tensor(dst=correction, data1=rmax, data2=m_new, op=nl.subtract)
    nisa.activation(dst=correction, op=nl.exp, data=correction)
    nisa.tensor_scalar(
        dst=running_out[nl.ds(0, HALF_P), :],
        data=running_out[nl.ds(0, HALF_P), :],
        op0=nl.multiply,
        operand0=correction,
    )
    nisa.tensor_scalar(dst=rsum, data=rsum, op0=nl.multiply, operand0=correction)
    exp_new = sbm.alloc_stack((cfg.q_head, 1), dtype=nl.float32)
    nisa.tensor_tensor(dst=exp_new, data1=score_new, data2=m_new, op=nl.subtract)
    nisa.activation(dst=exp_new, op=nl.exp, data=exp_new)
    nisa.tensor_tensor(dst=rsum, data1=rsum, data2=exp_new, op=nl.add)
    nisa.tensor_copy(dst=rmax, src=m_new, engine=nisa.vector_engine)
    return exp_new


def _accumulate_v_active(exp_new, v_active_sb, running_out, cfg, sbm):
    """Accumulate running_out += exp(score_new) * v_active."""
    v_broadcast = sbm.alloc_stack((cfg.q_head, cfg.d_head), dtype=nl.bfloat16)
    stream_shuffle_broadcast(src=v_active_sb, dst=v_broadcast)
    v_fp32 = sbm.alloc_stack((cfg.q_head, cfg.d_head), dtype=nl.float32)
    nisa.tensor_copy(dst=v_fp32, src=v_broadcast, engine=nisa.vector_engine)
    weighted_v = sbm.alloc_stack((cfg.q_head, cfg.d_head), dtype=nl.float32)
    nisa.tensor_scalar(dst=weighted_v, data=v_fp32, op0=nl.multiply, operand0=exp_new)
    nisa.tensor_tensor(
        dst=running_out[nl.ds(0, cfg.q_head), :],
        data1=running_out[nl.ds(0, cfg.q_head), :],
        data2=weighted_v,
        op=nl.add,
    )


def _load_kv_active(batch_idx, k_active_hbm, v_active_hbm, cfg, sbm):
    """Allocate SBUF buffers and load k_active/v_active from HBM."""
    k_active_sb = sbm.alloc_stack((1, cfg.d_head), dtype=nl.bfloat16)
    nisa.dma_copy(dst=k_active_sb, src=k_active_hbm[batch_idx : batch_idx + 1, nl.ds(0, cfg.d_head)])
    v_active_sb = sbm.alloc_stack((1, cfg.d_head), dtype=nl.bfloat16)
    nisa.dma_copy(dst=v_active_sb, src=v_active_hbm[batch_idx : batch_idx + 1, nl.ds(0, cfg.d_head)])
    return k_active_sb, v_active_sb


def _compute_active_tokens_sbuf(q_bufs, k_active_sb, v_active_sb, sm_state, cfg, sbm):
    """Scalar softmax update for active tokens (SBUF scope).

    Copies acc from PSUM to SBUF, computes Q·k_active dot product, updates
    running softmax state, and accumulates exp(score) * v_active.

    Args:
        k_active_sb: [1, d_head] bfloat16 in SBUF — already loaded from HBM.
        v_active_sb: [1, d_head] bfloat16 in SBUF — already loaded from HBM.
    """
    # Copy PSUM accumulator to SBUF for scalar operations. acc is band_p (= q_head) tall.
    sm_state.acc_sb = sbm.alloc_stack((TC.p_max, cfg.d_head), dtype=nl.float32)
    nisa.memset(sm_state.acc_sb, value=0.0)
    nisa.tensor_copy(
        dst=sm_state.acc_sb[nl.ds(0, cfg.q_head), :],
        src=sm_state.acc[nl.ds(0, cfg.q_head), :],
        engine=nisa.scalar_engine,
    )

    score_new = _compute_k_active_score(q_bufs.q_scaled, k_active_sb, cfg, sbm)
    exp_new = _softmax_update_scalar(score_new, sm_state.running_max, sm_state.running_sum, sm_state.acc_sb, cfg, sbm)
    _accumulate_v_active(exp_new, v_active_sb, sm_state.acc_sb, cfg, sbm)


# ── Step 4: Finalize Output (SBUF scope) ─────────────────────────────────────
def _finalize_output(cfg, tp, sm_state, sbm):
    """LNC2 gather (if sharded), normalize by softmax sum (SBUF scope).

    Result stored in sm_state.out_bf16.
    """
    if tp.sprior_n_prgs > 1:
        _lnc2_gather_and_normalize(cfg, tp, sm_state, sbm)
    else:
        _normalize_output(cfg, sm_state, sbm)


def _store_output_hbm(batch_idx, tp, sm_state, out_hbm):
    """Store normalized output to HBM (only NC 0 writes in LNC2 mode)."""
    if tp.sprior_prg_id == 0:
        nisa.dma_copy(dst=out_hbm[batch_idx], src=sm_state.out_bf16)


def _normalize_output(cfg, sm_state, sbm):
    """Compute output = acc / running_sum (single NC path), output as bf16."""
    inv_sum = sbm.alloc_stack((cfg.q_head, 1), dtype=nl.float32)
    nisa.activation(dst=inv_sum, op=nl.reciprocal, data=sm_state.running_sum[nl.ds(0, cfg.q_head), :])
    sm_state.out_bf16 = sbm.alloc_stack((cfg.q_head, cfg.d_head), dtype=nl.bfloat16)
    nisa.tensor_scalar(
        dst=sm_state.out_bf16, data=sm_state.acc_sb[nl.ds(0, cfg.q_head), :], op0=nl.multiply, operand0=inv_sum
    )


def _lnc2_gather_and_normalize(cfg, tp, sm_state, sbm):
    """Exchange acc and sum across NCs, rescale by correction factor, normalize."""
    # Exchange running max to compute global max
    m_local = sbm.alloc_stack((cfg.q_head, 1), dtype=nl.float32)
    nisa.tensor_copy(dst=m_local, src=sm_state.running_max[nl.ds(0, cfg.q_head), :], engine=nisa.vector_engine)
    m_remote = sbm.alloc_stack((cfg.q_head, 1), dtype=nl.float32)
    nisa.sendrecv(
        src=m_local,
        dst=m_remote,
        send_to_rank=(1 - tp.sprior_prg_id),
        recv_from_rank=(1 - tp.sprior_prg_id),
        pipe_id=0,
    )
    m_global = sbm.alloc_stack((cfg.q_head, 1), dtype=nl.float32)
    nisa.tensor_tensor(dst=m_global, data1=m_local, data2=m_remote, op=nl.maximum)

    # Correction factor: exp(local_max - global_max)
    local_corr = sbm.alloc_stack((cfg.q_head, 1), dtype=nl.float32)
    nisa.tensor_tensor(dst=local_corr, data1=m_local, data2=m_global, op=nl.subtract)
    nisa.activation(dst=local_corr, op=nl.exp, data=local_corr)

    # Rescale local acc and sum
    acc_q = sm_state.acc_sb[nl.ds(0, cfg.q_head), :]
    nisa.tensor_scalar(dst=acc_q, data=acc_q, op0=nl.multiply, operand0=local_corr)
    l_local = sbm.alloc_stack((cfg.q_head, 1), dtype=nl.float32)
    nisa.tensor_scalar(
        dst=l_local, data=sm_state.running_sum[nl.ds(0, cfg.q_head), :], op0=nl.multiply, operand0=local_corr
    )

    # Exchange rescaled partials
    acc_recv = sbm.alloc_stack((cfg.q_head, cfg.d_head), dtype=nl.float32)
    nisa.sendrecv(
        src=acc_q, dst=acc_recv, send_to_rank=(1 - tp.sprior_prg_id), recv_from_rank=(1 - tp.sprior_prg_id), pipe_id=0
    )
    l_recv = sbm.alloc_stack((cfg.q_head, 1), dtype=nl.float32)
    nisa.sendrecv(
        src=l_local, dst=l_recv, send_to_rank=(1 - tp.sprior_prg_id), recv_from_rank=(1 - tp.sprior_prg_id), pipe_id=0
    )

    # Sum and normalize
    nisa.tensor_tensor(dst=acc_q, data1=acc_q, data2=acc_recv, op=nl.add)
    nisa.tensor_tensor(dst=l_local, data1=l_local, data2=l_recv, op=nl.add)

    inv_sum = sbm.alloc_stack((cfg.q_head, 1), dtype=nl.float32)
    nisa.activation(dst=inv_sum, op=nl.reciprocal, data=l_local)
    sm_state.out_bf16 = sbm.alloc_stack((cfg.q_head, cfg.d_head), dtype=nl.bfloat16)
    nisa.tensor_scalar(dst=sm_state.out_bf16, data=acc_q, op0=nl.multiply, operand0=inv_sum)
