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
  Fold (512 tokens)   — 4 blocks, [32P, 640F] in SBUF
  Chunk (2048 tokens) — 4 folds, [128P, 640F] in SBUF, one online softmax iteration

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
Q is zero-padded into two variants (Q_lo, Q_hi) so that paired block matmuls
accumulate into a single [128P, 256F] PSUM tile per fold. Eviction is 8
copies of [128P, 128F] instead of 16 copies of [64P, 128F].

PSUM Layout (per fold): [128P, 256F]
    P[0:64]   = blk0/blk2 scores (from Q_lo)
    P[64:128] = blk1/blk3 scores (from Q_hi)
    F[0:128]  = pair 0 (blk0+blk1)
    F[128:256] = pair 1 (blk2+blk3)

Score SBUF Layout (after eviction): [128P, score_free]
    pair 0 → first 512F, pair 1 → last 512F

Mask Layout: [B * num_chunks, TC.p_max, score_free] uint8
    Pre-computed on host matching the score_sb eviction layout.
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
from .attention_mxfp8_tkg_utils import swizzle_quantize_mx

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
    """Number of query heads (must be 64)."""

    bucket_size: int
    """Bucket size in tokens (compile-time, multiple of chunk_tokens)."""

    d_head: int
    """Head dimension."""

    def __post_init__(self):
        kernel_assert(self.bs >= 1, f"bs must be >= 1, got {self.bs=}")
        kernel_assert(self.q_head == 64, f"q_head must be 64, got {self.q_head=}")
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

        self.fold_cols = self.blocks_per_fold * self.packed_cols
        """Free dimension of one fold buffer."""

        # Q dimensions
        self.q_free = cfg.d_head
        """Q free dim for packed-Q variants."""

        # Chunk geometry
        self.chunk_tokens = self.folds_per_chunk * self.fold_len
        """Tokens per chunk (2048)."""

        self.pairs_per_fold = 2
        """Block pairs per fold from packed-Q eviction (Q_lo + Q_hi)."""

        self.score_free = self.folds_per_chunk * self.block_len * self.pairs_per_fold
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
        self.q_lo_data = None
        self.q_lo_scale = None
        self.q_hi_data = None
        self.q_hi_scale = None

    def load_from_hbm(self, q_hbm, batch_idx):
        """Load Q from HBM, scale, quantize to MXFP8, build packed Q_lo/Q_hi."""
        cfg, tp, sbm = self.cfg, self.tp, self.sbm

        q_bf16 = sbm.alloc_stack((cfg.q_head, cfg.d_head), dtype=nl.bfloat16)
        nisa.dma_copy(dst=q_bf16, src=q_hbm[batch_idx, :, nl.ds(0, 1), :])

        self.q_scaled = sbm.alloc_stack((cfg.q_head, cfg.d_head), dtype=nl.bfloat16)
        nisa.tensor_scalar(dst=self.q_scaled, data=q_bf16, op0=nl.multiply, operand0=tp.softmax_scale)

        mx_par = cfg.d_head // 4  # 32
        q_mx_data_base = sbm.alloc_stack((mx_par, cfg.q_head), dtype=nl.float8_e4m3fn_x4)
        q_mx_scale_base = sbm.alloc_stack((mx_par, cfg.q_head), dtype=nl.uint8)
        swizzle_quantize_mx(self.q_scaled, q_mx_data_base, q_mx_scale_base, sbm)

        result = _build_q_variants(q_mx_data_base, q_mx_scale_base, cfg, tp, sbm)
        self.q_lo_data = result[0]
        self.q_lo_scale = result[1]
        self.q_hi_data = result[2]
        self.q_hi_scale = result[3]


class SoftmaxState(nl.NKIObject):
    """Online softmax running state and accumulators.

    Constructed once per batch iteration. Holds running max/sum, the PSUM
    accumulator, and the identity matrix needed for PE-based sum reduction.
    """

    def __init__(self, cfg, sbm, identity_sb):
        self.cfg = cfg
        self.sbm = sbm
        self.half_p = cfg.q_head  # 64

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

        self.acc = nl.ndarray((cfg.q_head, cfg.d_head), dtype=nl.float32, buffer=nl.psum)
        """[q_head, d_head] fp32 in PSUM."""
        nisa.memset(self.acc, value=0.0)

        # Set later in Steps 3/4
        self.acc_sb = None
        """[TC.p_max, d_head] fp32 in SBUF."""
        self.out_bf16 = None
        """[q_head, d_head] bf16 in SBUF."""

    def update_online_softmax(self, score_sb, score_max_sb, score_sb_fp32_reinterp):
        """Packed online softmax on [128, score_free]. Score path in bf16, accumulators in fp32.

        Sub-steps:
          1. Compute new global max across both halves
          2. Rescale old accumulators by correction factor
          3. Compute exp(score - max) in tiles, reduce sum via PE matmul
          4. Launch DMA reinterpret bf16→fp32 (overlaps with sum exchange)
          5. Cross-half sum exchange, update running state
        """
        m_new, correction = self._compute_new_max(score_max_sb)
        self._rescale_accumulators(correction)
        l_local = self._exp_and_reduce(score_sb, m_new)

        nisa.dma_copy(
            dst=score_sb_fp32_reinterp,
            src=score_sb.view(nl.float32),
        )

        self._update_running_state(l_local, m_new)

    def _compute_new_max(self, m_local):
        """Cross-half exchange of the MM1-produced chunk-local max, merge with running max.

        m_local [128, 1] is the per-partition score max produced by the MM1
        select_reduce eviction (fused, no standalone tensor_reduce needed).

        Returns (m_new [128, 1], correction [128, 1]).
        """
        sbm = self.sbm
        half_p = self.half_p

        m_bottom = sbm.alloc_stack((half_p, 1), dtype=nl.float32)
        nisa.tensor_copy(dst=m_bottom, src=m_local[nl.ds(half_p, half_p), :], engine=nisa.scalar_engine)
        m_global = sbm.alloc_stack((half_p, 1), dtype=nl.float32)
        nisa.tensor_tensor(dst=m_global, data1=m_local[nl.ds(0, half_p), :], data2=m_bottom, op=nl.maximum)

        m_new = sbm.alloc_stack((2 * half_p, 1), dtype=nl.float32)
        nisa.tensor_tensor(
            dst=m_new[nl.ds(0, half_p), :], data1=self.running_max[nl.ds(0, half_p), :], data2=m_global, op=nl.maximum
        )
        nisa.tensor_copy(dst=m_new[nl.ds(half_p, half_p), :], src=m_new[nl.ds(0, half_p), :], engine=nisa.scalar_engine)

        # Only the top half is consumed downstream (acc uses [0:half_p], running_sum's
        # bottom half is dead), so compute correction at half width.
        correction = sbm.alloc_stack((half_p, 1), dtype=nl.float32)
        # Fused exp(running_max - m_new): activate2 does (data - m_new) then exp in one
        # scalar-engine instruction, replacing the tensor_scalar + activation pair.
        nisa.activate2(
            dst=correction,
            op=nl.exp,
            data=self.running_max[nl.ds(0, half_p), :],
            imm0=m_new[nl.ds(0, half_p), :],
            imm1=0.0,
            op0=nl.subtract,
            op1=nl.bypass,
        )

        return m_new, correction

    def _rescale_accumulators(self, correction):
        """Rescale old acc and running_sum by correction factor ([0:half_p] only)."""
        half_p = self.half_p
        nisa.tensor_scalar(
            dst=self.acc,
            data=self.acc,
            op0=nl.multiply,
            operand0=correction,
            engine=nisa.scalar_engine,
        )
        nisa.tensor_scalar(
            dst=self.running_sum[nl.ds(0, half_p), :],
            data=self.running_sum[nl.ds(0, half_p), :],
            op0=nl.multiply,
            operand0=correction,
            engine=nisa.scalar_engine,
        )

    def _exp_and_reduce(self, score_sb, m_new):
        """Compute exp(score - max) in-place, reduce sum via PE matmul with identity.

        Returns l_local [128, 1] — local sum of exponentials.
        """
        sbm = self.sbm
        half_p = self.half_p
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

        l_local = sbm.alloc_stack((2 * half_p, 1), dtype=nl.float32)
        nisa.tensor_reduce(dst=l_local, op=nl.add, data=psum_reduction, axis=1)
        return l_local

    def _update_running_state(self, l_local, m_new):
        """Cross-half sum exchange, accumulate into running_sum, update running_max."""
        sbm = self.sbm
        half_p = self.half_p

        l_bottom = sbm.alloc_stack((half_p, 1), dtype=nl.float32)
        nisa.tensor_copy(dst=l_bottom, src=l_local[nl.ds(half_p, half_p), :], engine=nisa.scalar_engine)
        l_global = sbm.alloc_stack((half_p, 1), dtype=nl.float32)
        nisa.tensor_tensor(dst=l_global, data1=l_local[nl.ds(0, half_p), :], data2=l_bottom, op=nl.add)

        nisa.tensor_tensor(
            dst=self.running_sum[nl.ds(0, half_p), :],
            data1=self.running_sum[nl.ds(0, half_p), :],
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

        self.k_buf = sbm.alloc_stack((TC.p_max, tp.fold_cols), dtype=nl.float32)
        """[128P, 640F] K block buffer."""

        self.v_buf = sbm.alloc_stack((TC.p_max, tp.fold_cols), dtype=nl.float32)
        """[128P, 640F] V block buffer."""


def _load_chunk_context(cb, batch_idx, chunk_idx, mask, k_prior, v_prior, kv_loader, tp):
    """Load all chunk data from HBM into pre-allocated ChunkBuffers."""
    mask_idx = batch_idx * tp.num_chunks + chunk_idx
    nisa.dma_copy(dst=cb.chunk_mask_sb, src=mask[mask_idx, :, nl.ds(0, tp.score_free)])
    kv_loader.load_blocks(cb.k_buf, k_prior, chunk_idx)
    kv_loader.load_blocks(cb.v_buf, v_prior, chunk_idx)


# ── Q Variant Construction ─────────────────────────────────────────────────────
def _build_q_variants(q_mx_data_base, q_mx_scale_base, cfg, tp, sbm):
    """Build Q_lo [128P, 128F] and Q_hi [128P, 128F] from base MXFP8 Q [32P, 64F].

    Q_lo: Q in first 64F, zeros in last 64F → matmul output in P[0:64]
    Q_hi: zeros in first 64F, Q in last 64F → matmul output in P[64:128]
    """
    # Q_lo base [32P, 128F]
    q_lo_data_base = sbm.alloc_stack((tp.p_per_block, tp.q_free), dtype=nl.float8_e4m3fn_x4)
    q_lo_scale_base = sbm.alloc_stack((tp.p_per_block, tp.q_free), dtype=nl.uint8)
    nisa.memset(q_lo_data_base, value=0, engine=nisa.gpsimd_engine)
    nisa.memset(q_lo_scale_base, value=0, engine=nisa.gpsimd_engine)
    nisa.tensor_copy(dst=q_lo_data_base[:, nl.ds(0, cfg.q_head)], src=q_mx_data_base, engine=nisa.vector_engine)
    nisa.tensor_copy(dst=q_lo_scale_base[:, nl.ds(0, cfg.q_head)], src=q_mx_scale_base, engine=nisa.vector_engine)

    # Q_hi base [32P, 128F]
    q_hi_data_base = sbm.alloc_stack((tp.p_per_block, tp.q_free), dtype=nl.float8_e4m3fn_x4)
    q_hi_scale_base = sbm.alloc_stack((tp.p_per_block, tp.q_free), dtype=nl.uint8)
    nisa.memset(q_hi_data_base, value=0, engine=nisa.gpsimd_engine)
    nisa.memset(q_hi_scale_base, value=0, engine=nisa.gpsimd_engine)
    nisa.tensor_copy(
        dst=q_hi_data_base[:, nl.ds(cfg.q_head, cfg.q_head)], src=q_mx_data_base, engine=nisa.vector_engine
    )
    nisa.tensor_copy(
        dst=q_hi_scale_base[:, nl.ds(cfg.q_head, cfg.q_head)], src=q_mx_scale_base, engine=nisa.vector_engine
    )

    # Replicate across 4 quadrants → [128P, 128F]
    q_lo_data = sbm.alloc_stack((TC.p_max, tp.q_free), dtype=nl.float8_e4m3fn_x4)
    q_lo_scale = sbm.alloc_stack((TC.p_max, tp.q_free), dtype=nl.uint8)
    q_hi_data = sbm.alloc_stack((TC.p_max, tp.q_free), dtype=nl.float8_e4m3fn_x4)
    q_hi_scale = sbm.alloc_stack((TC.p_max, tp.q_free), dtype=nl.uint8)
    for quadrant_idx in range(tp.folds_per_chunk):
        p_off = quadrant_idx * tp.p_per_block
        nisa.tensor_copy(dst=q_lo_data[nl.ds(p_off, tp.p_per_block), :], src=q_lo_data_base, engine=nisa.vector_engine)
        nisa.tensor_copy(
            dst=q_lo_scale[nl.ds(p_off, tp.p_per_block), :], src=q_lo_scale_base, engine=nisa.vector_engine
        )
        nisa.tensor_copy(dst=q_hi_data[nl.ds(p_off, tp.p_per_block), :], src=q_hi_data_base, engine=nisa.vector_engine)
        nisa.tensor_copy(
            dst=q_hi_scale[nl.ds(p_off, tp.p_per_block), :], src=q_hi_scale_base, engine=nisa.vector_engine
        )

    return q_lo_data, q_lo_scale, q_hi_data, q_hi_scale


# ── Batch Block KV Cache Loader ─────────────────────────────────────────────────────
class BatchBlockKVCacheLoader(nl.NKIObject):
    """Manages block table state for indirect DMA access to block-sparse KV cache.

    Computes a [128, blocks_per_fold] offset vector and gathers a chunk's blocks
    with one indirect DMA per block (blocks_per_fold DMAs), each gathering
    TC.p_max rows into that block's packed_cols-wide column slice of the buffer.

    Lifecycle:
        1. Constructed once per batch with that batch's [num_blocks] table row:
           loads it into a single SBUF partition as [1, folds_count, blocks_per_fold].
        2. Per chunk: call load_blocks(buf, cache_prior, chunk_idx) — lays out the
           chunk's block IDs (fold per quadrant), computes vector offsets, and
           DMA-loads blocks (one indirect DMA per block).
    """

    def __init__(self, active_blocks_row, cfg, tp: TileParams, sbm):
        self.sbm = sbm
        self.tp = tp

        """
        Row-within-block offsets [128, blocks_per_fold]. Each column holds, down the
        128 partitions, the pattern [0,1,...,31, 0,1,...,31, ...] — i.e. partition % 32,
        the row index within a 32-row block, repeated across the four quadrants. Built
        by transposing an iota (0..31) into quadrant 0, then copying that quadrant into
        the other three.
        """
        self.row_offsets = sbm.alloc_stack((TC.p_max, tp.blocks_per_fold), dtype=nl.uint32)
        iota_sb = nl.ndarray((tp.blocks_per_fold, TC.p_per_quadrant), dtype=nl.uint32)
        nisa.iota(iota_sb, [[1, TC.p_per_quadrant]])

        nisa.nc_transpose(self.row_offsets[nl.ds(0, TC.p_per_quadrant), :], iota_sb)

        nisa.tensor_copy(
            self.row_offsets[nl.ds(TC.p_per_quadrant, TC.p_per_quadrant), :],
            self.row_offsets[nl.ds(0, TC.p_per_quadrant), :],
        )
        nisa.tensor_copy(
            self.row_offsets[nl.ds(2 * TC.p_per_quadrant, 2 * TC.p_per_quadrant), :],
            self.row_offsets[nl.ds(0, 2 * TC.p_per_quadrant), :],
        )

        """
        Load the block table row into a single SBUF partition, reshaped to
        [1, folds_count, blocks_per_fold]. Kept on one partition; per-chunk
        replication across quadrants happens later in _prep_chunk_table.
        """
        folds_count = active_blocks_row.shape[0] // tp.blocks_per_fold
        self.table_sb = sbm.alloc_stack((1, folds_count, tp.blocks_per_fold), dtype=nl.int32)
        nisa.dma_copy(dst=self.table_sb, src=active_blocks_row.reshape((1, folds_count, tp.blocks_per_fold)))

    def load_blocks(self, buf, cache_prior, chunk_idx):
        """Compute vector offsets for chunk_idx and DMA-load blocks into buf [128, 640].

        Args:
            buf: [128, 640] float32 in SBUF. Pre-allocated destination.
            cache_prior: [num_blocks, 32, 160] float32 in HBM. K or V cache.
            chunk_idx: Which chunk to load (0-indexed).
        """
        tp = self.tp
        vector_offsets = self._prep_vector_offsets(chunk_idx)

        """
        Flatten the block cache to a 2D [num_blocks * 32, packed_cols] row grid, then
        issue one indirect DMA per block. For block_idx, column of vector_offsets
        supplies one absolute row index per destination partition (indirect_dim=0), the
        outer pattern dim walks TC.p_max gathered rows, and the inner dim copies
        packed_cols contiguous columns into that block's column slice of buf.
        """
        cache_prior_2d = cache_prior.reshape((cache_prior.shape[0] * TC.p_per_quadrant, tp.packed_cols))

        for block_idx in range(tp.blocks_per_fold):
            nisa.dma_copy(
                dst=buf[:, nl.ds(block_idx * tp.packed_cols, tp.packed_cols)],
                src=cache_prior_2d.ap(
                    [[tp.packed_cols, TC.p_max], [1, tp.packed_cols]],
                    offset=0,
                    vector_offset=vector_offsets[:, nl.ds(block_idx, 1)],
                    indirect_dim=0,
                ),
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
        """Lay out this chunk's block IDs: quadrant q holds fold q's IDs on all 32 rows.

        Copy each of the chunk's folds_per_chunk folds from the single-partition
        table_sb into the top partition of its quadrant, then broadcast that partition
        down the quadrant's 32 rows with nc_stream_shuffle (pattern [0]*32).
        """
        tp = self.tp
        row_bases = self.sbm.alloc_stack((TC.p_max, tp.blocks_per_fold), dtype=nl.int32)

        for fold_idx in range(tp.folds_per_chunk):
            nisa.tensor_copy(
                dst=row_bases[nl.ds(TC.p_per_quadrant * fold_idx, 1), :],
                src=self.table_sb[0, nl.ds(chunk_idx * tp.folds_per_chunk + fold_idx, 1), :],
            )

        shuffle_pattern = [0] * 32
        nisa.nc_stream_shuffle(row_bases, row_bases, shuffle_pattern)
        return row_bases


# ── MM1/MM2: Block Matmul ─────────────────────────────────────────────────────
def _emit_block_matmul(q_data, q_scale, kv_buf, psum_tile, row_offset, block_idx, f_off, tp):
    """One nc_matmul_mx: Q[row_offset] × block[block_idx] in kv_buf [128,640] → psum_tile[:, f_off].

    Access-pattern decode (kv_buf is [128P, fold_cols], each block spans packed_cols
    columns laid out as [block_len data | block_len//4 scale]):
      - kv_quadrant slices p_per_block partitions at row_offset.
      - block base column = block_idx * packed_cols within that quadrant.
      - `moving` views the quadrant as float8_e4m3fn_x4 and reads block_len data columns.
      - `moving_scale` views as uint8 (4× column count), `+ block_len * 4` skips data
        columns to reach scale columns. Scale uses p_per_quadrant // mx_group_partitions rows.
    """
    block_base_col = block_idx * tp.packed_cols
    scale_partition_count = TC.p_per_quadrant // TC.mx_group_partitions
    kv_quadrant = kv_buf[nl.ds(row_offset, tp.p_per_block), :]
    nisa.nc_matmul_mx(
        dst=psum_tile[:, nl.ds(f_off, tp.block_len)],
        stationary=q_data[nl.ds(row_offset, tp.p_per_block), :],
        moving=kv_quadrant.view(nl.float8_e4m3fn_x4)[:, nl.ds(block_base_col, tp.block_len)],
        stationary_scale=q_scale[nl.ds(row_offset, tp.p_per_block), :],
        moving_scale=kv_quadrant.view(nl.uint8)[
            nl.ds(0, scale_partition_count), nl.ds(block_base_col * 4 + tp.block_len * 4, tp.block_len)
        ],
        tile_position=(row_offset, 0),
        tile_size=(tp.p_per_block, TC.p_max),
    )


def _mm1_compute_chunk(q_bufs, k_buf, score_sb, score_max_sb, chunk_mask_sb, cfg, tp):
    """Compute MM1 (Q x K^T) with packed-Q eviction (SBUF scope).

    k_buf: [128P, 640F] — already loaded with K blocks from HBM.
    score_sb: [TC.p_max, tp.score_free] bf16 — output scores buffer.
    score_max_sb: [TC.p_max, 1] fp32 — per-partition score max, fused into eviction.
    chunk_mask_sb: [TC.p_max, tp.score_free] uint8 — mask for this chunk.

    Q is [128P, 64F] (32P base replicated across 4 quadrants).
    Matmul produces 64 output partitions. Even blocks land in P[0:64],
    odd blocks land in P[64:128] via the zero-padded stationary free dim.
    """
    psum_full = nl.ndarray((TC.p_max, tp.score_free), dtype=nl.bfloat16, buffer=nl.psum)

    """
    Tiling: iterate the chunk as pairs_per_fold block pairs x folds_per_chunk folds
    (2 x 4 = 8 iterations, 2 block matmuls each = 16 blocks = one chunk). Each pair's
    Q_lo/Q_hi matmuls accumulate into a [128P, block_len] slice of psum_full; the
    Block/Fold/Chunk geometry and PSUM layout are documented in the module docstring.
    """
    for pair_idx in range(2):
        blk_lo = pair_idx * 2
        blk_hi = pair_idx * 2 + 1
        f_off = pair_idx * tp.block_len

        for fold_idx in range(tp.folds_per_chunk):
            row_offset = fold_idx * tp.p_per_block
            psum_tile_f = (pair_idx * tp.blocks_per_fold + fold_idx) * tp.block_len
            psum_tile = psum_full[:, nl.ds(psum_tile_f, tp.block_len)]

            _emit_block_matmul(q_bufs.q_lo_data, q_bufs.q_lo_scale, k_buf, psum_tile, row_offset, blk_lo, 0, tp)
            _emit_block_matmul(q_bufs.q_hi_data, q_bufs.q_hi_scale, k_buf, psum_tile, row_offset, blk_hi, 0, tp)

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
    least one full chunk); requires q_head == 64 and d_head == 128.

    All configuration is derived from input tensor shapes:
        bs, q_head, d_head from q.shape = [bs, q_head, 1, d_head]
        bucket_size from k_prior.shape = [num_blocks, 32, 160]

    Note: Tensor layouts differ from attention_tkg. This kernel uses H in the
    partition dim for packed-Q eviction, while attention_tkg uses d in partitions.

    Dimensions:
        B: Batch size.
        H: Number of query heads (must be 64).
        d: Head dimension (must be 128).
        num_blocks: KV cache blocks; each block covers 128 tokens as [32, 160] MXFP8.
        num_chunks: bucket_size / 2048 online-softmax iterations.
        score_free: Free-dim width of the per-chunk score buffer (1024).

    Args:
        q: Query tensor [B, H, 1, d] bfloat16.
        k_active: Active key [B, d] bfloat16.
        v_active: Active value [B, d] bfloat16.
        k_prior: MXFP8 K cache [num_blocks, 32, 160] float32. Each block = 128 tokens.
        v_prior: MXFP8 V cache [num_blocks, 32, 160] float32. Each block = 128 tokens.
        mask: Pre-computed chunk masks [B * num_chunks, 128, score_free] uint8.
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
        sm_state = SoftmaxState(cfg, sbm, identity_sb)

        # Step 2: Chunk loop — load → MM1 → softmax → MM2
        for chunk_local in range(tp.chunks_per_nc):
            chunk_idx = tp.chunk_start + chunk_local
            # Allocated per iteration by design: buffers carry no cross-chunk state, so the
            # stack allocator hands back the same SBUF region each pass (reuse, not growth).
            cb = ChunkBuffers(tp, sbm)

            # Load all chunk data from HBM
            _load_chunk_context(cb, batch_idx, chunk_idx, mask, k_prior, v_prior, kv_loader, tp)

            # Compute MM1 → softmax → requantize → MM2
            _mm1_compute_chunk(q_bufs, cb.k_buf, cb.score_sb, cb.score_max_sb, cb.chunk_mask_sb, cfg, tp)
            sm_state.update_online_softmax(cb.score_sb, cb.score_max_sb, cb.score_sb_fp32_reinterp)
            scores_data, scores_scale = _requantize_scores(cb.score_sb_fp32_reinterp, tp, sbm)
            _mm2_compute_chunk(scores_data, scores_scale, cb.v_buf, cfg, tp, sm_state)

        # Step 3: Load k_active/v_active (HBM scope), then compute (SBUF scope)
        k_active_sb, v_active_sb = _load_kv_active(batch_idx, k_active, v_active, cfg, sbm)
        _compute_active_tokens_sbuf(q_bufs, k_active_sb, v_active_sb, sm_state, cfg, sbm)

        # Step 4: Normalize (SBUF scope), then store output (HBM scope)
        _finalize_output(cfg, tp, sm_state, sbm)
        _store_output_hbm(batch_idx, tp, sm_state, out_hbm)

    sbm.close_scope()  # mxfp8_attn

    return out_hbm


# ── Step 2 helpers ─────────────────────────────────────────────────────────────


def _swizzle_quantize_fp32_piece(fp32_src, fp32_offset, mx_data_dst, mx_scale_dst, sbm, h_fp32=256):
    """Swizzle+quantize one [T, h_fp32] fp32 piece from a larger fp32 tensor."""
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
    swizzled_bf16 = sbm.alloc_stack((h_fp32_half, T * 4), dtype=nl.bfloat16)
    nisa.dma_copy(
        dst=swizzled_bf16,
        src=swizzled_fp32.view(nl.bfloat16),
    )
    nisa.quantize_mx(dst=mx_data_dst, src=swizzled_bf16, dst_scale=mx_scale_dst)


def _requantize_scores(score_sb_fp32_reinterp, tp, sbm):
    """Re-quantize softmax scores from fp32 to MXFP8 (SBUF scope).

    score_sb_fp32_reinterp: [TC.p_max, tp.score_free // 2] fp32 in SBUF.
    Returns: (scores_data, scores_scale) MXFP8 tensors in SBUF.
    """
    n_pieces = tp.pairs_per_fold
    scores_data = sbm.alloc_stack((TC.p_max, n_pieces * TC.p_max), dtype=nl.float8_e4m3fn_x4)
    scores_scale = sbm.alloc_stack((TC.p_max, n_pieces * TC.p_max), dtype=nl.uint8)
    for piece_idx in range(n_pieces):
        _swizzle_quantize_fp32_piece(
            score_sb_fp32_reinterp,
            piece_idx * 256,
            scores_data[:, nl.ds(piece_idx * TC.p_max, TC.p_max)],
            scores_scale[:, nl.ds(piece_idx * TC.p_max, TC.p_max)],
            sbm,
        )
    return scores_data, scores_scale


def _mm2_compute_chunk(scores_data, scores_scale, v_buf, cfg, tp, sm_state):
    """Compute MM2: scores × V, accumulating into sm_state.acc (SBUF scope).

    v_buf: [128P, 640F] — already loaded with V blocks from HBM.

    Tiling: one nc_matmul_mx per fold (folds_per_chunk iterations), each contracting
    the fold's score_free_per_fold score columns against the fold's V block and
    accumulating into the shared acc PSUM tile.

    Access-pattern decode (same [data | scale] block layout as MM1):
      - fold base column = sb_idx * packed_cols
      - `moving` reads the block_len V data columns as float8_e4m3fn_x4.
      - `moving_scale` reinterprets v_buf as uint8, so strides and offset are ×4 and
        `+ block_len * 4` skips the data columns to reach the scale columns.
    """
    score_free_per_fold = cfg.q_head  # 64
    scale_partition_count = TC.p_max // TC.mx_group_partitions

    for sb_idx in range(tp.folds_per_chunk):
        v_col_offset = sb_idx * tp.packed_cols

        nisa.nc_matmul_mx(
            dst=sm_state.acc,
            stationary=scores_data[:, nl.ds(sb_idx * score_free_per_fold, score_free_per_fold)],
            moving=v_buf.view(nl.float8_e4m3fn_x4)[:, nl.ds(v_col_offset, tp.block_len)],
            stationary_scale=scores_scale[:, nl.ds(sb_idx * score_free_per_fold, score_free_per_fold)],
            moving_scale=v_buf.view(nl.uint8)[
                nl.ds(0, scale_partition_count), nl.ds(v_col_offset * 4 + tp.block_len * 4, tp.block_len)
            ],
        )


# ── Step 3: Active-token helpers ──────────────────────────────────────────────


def _compute_k_active_score(q_scaled_sb, k_active_sb, cfg, sbm):
    """Compute score_new = Q_scaled · k_active → [q_heads, 1]."""
    k_broadcast = sbm.alloc_stack((cfg.q_head, cfg.d_head), dtype=nl.bfloat16)
    stream_shuffle_broadcast(src=k_active_sb, dst=k_broadcast)
    qk = sbm.alloc_stack((cfg.q_head, cfg.d_head), dtype=nl.float32)
    nisa.tensor_tensor(dst=qk, data1=q_scaled_sb, data2=k_broadcast, op=nl.multiply)
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
    # Copy PSUM accumulator to SBUF for scalar operations
    sm_state.acc_sb = sbm.alloc_stack((TC.p_max, cfg.d_head), dtype=nl.float32)
    nisa.memset(sm_state.acc_sb, value=0.0)
    nisa.tensor_copy(dst=sm_state.acc_sb[nl.ds(0, cfg.q_head), :], src=sm_state.acc, engine=nisa.scalar_engine)

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
