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

"""GLM DSA sparse-attention indexer, bf16.

Projects Q and K, writes the paged K cache, all-gathers the CP context, scores Q against it in
position blocks under a causal mask, and returns the top-k positions per query.

``S_local`` is the queries this rank owns and ``S_global = cp_degree * S_local`` the context it
scores against; see :class:`GlmIndexerConfig`. Target envelope is ``S_local`` 2k-16k.
"""

import math
from dataclasses import dataclass

import nki
import nki.collectives as ncc
import nki.isa as nisa
import nki.language as nl
from nki.isa.constants import dge_mode

from ....core.utils.allocator import SbufManager
from ....core.utils.kernel_assert import kernel_assert
from ....core.utils.kernel_helpers import div_ceil, get_max_positive_value_for_dtype, get_verified_program_sharding_info
from ....core.utils.logging import get_logger
from ..deepseek.mla_common_cte import RopeLayout
from .glm_indexer_topk import (
    SEG_MAX,
    alloc_chunk_candidates,
    topk_chunk_merge,
    topk_chunk_stage1,
    topk_geometry,
    topk_over_score_chunked,
)

# --------------------------------------------------------------------------------------------
# Hardware constants
# --------------------------------------------------------------------------------------------

P_MAX = 128
BF16_K_TILE = 128
PSUM_FMAX = 512
_NUM_PSUM_BANKS = 8
MATMUL_TILE = 512
# Positions scored per PSUM tile -- ``MATMUL_TILE * 8``.
POS_BLOCK = 4096
KPOS_COLS = 512
"""Column width the key-position build works in, bounding its int32 scratch."""

SCORE_RING_MIN = 2

# --------------------------------------------------------------------------------------------
# FP8 paged cache constants
# --------------------------------------------------------------------------------------------

# Floor on the per-token amax so an all-zero (padding) token cannot divide by zero; the row still
# quantizes to exact zeros. Matches the MLA fp8 cache.
_AMAX_FLOOR = 1e-30

# Row-stride alignment for the fused fp8 cache row: 8 bf16 == 16 bytes, the DMA tran_size.
_KV_ROW_ALIGN_BF16 = 8


def kv_row_bf16_width(head_dim: int) -> int:
    """bf16 containers per fused fp8 cache row, padded to a 16-byte (8 bf16) stride.

    The row holds ``head_dim`` fp8 K values bitcast two-per-bf16 (``head_dim // 2`` containers),
    then one bf16 per-token dequant scale, then padding. Unlike the MLA cache there is no separate
    ``k_pe`` field: the indexer's whole ``head_dim`` (RoPE included) is quantized, matching vLLM, so
    the cache stays one contiguous row per token. GLM's ``head_dim=128`` gives ``64 + 1 = 65 -> 72``
    bf16 (144 B).
    """
    return div_ceil(head_dim // 2 + 1, _KV_ROW_ALIGN_BF16) * _KV_ROW_ALIGN_BF16


def score_ring_slots(tiles_per_core: int) -> int:
    """Score staging slots needed: ONE PER TILE THIS CORE OWNS, not a fixed 2.

    The score phase writes every tile of a group across the position-block loop and only then ranks
    them, so a slot must survive from its write until its top-k reads it. With fewer slots than
    tiles, a later tile overwrites an unranked one and both rank the wrong rows.

    So the count is ceil(num_q_tiles / num_cores). Not yet sufficient for cp64 at 1M, where 64 tiles
    per core would need 34 GB; that shape needs the top-k fused per group with a shrunken panel.
    """
    return max(SCORE_RING_MIN, tiles_per_core)


# Usable SBUF per partition on gen3 (trn3): 224 KB - 16 KB - 8. gen4 has 240 KB.
SBUF_BUDGET_BYTES = 212984
_NAME_SEQ = [0]


def _uid() -> str:
    """Unique suffix for ``alloc_stack`` names.

    Names must be unique across the whole trace -- ``close_scope`` frees a scope's addresses but
    not its names -- and every loop here is Python-unrolled, so a literal name collides on the
    second iteration.
    """
    _NAME_SEQ[0] += 1
    return f"_{_NAME_SEQ[0]}"


# --------------------------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class GlmIndexerConfig:
    """The one shape this kernel serves.

    Frozen on purpose: add a field only when a second production shape genuinely needs it.

    Attributes:
        s_local: queries this rank owns (the design envelope is 2048-16384).
        s_global: CP context length each query scores against (``cp_degree * s_local``).
        n_heads: indexer heads. GLM uses 32; the score sums over them with ``weights_proj``.
        head_dim: per-head width of the indexer Q/K.
        dim: model hidden size, the K projection's contraction.
        q_lora_rank: latent width of the Q input, the Q projection's contraction.
        rope_head_dim: leading slice of ``head_dim`` that carries RoPE.
        index_topk: positions selected per query.
    """

    s_local: int
    s_global: int
    n_heads: int = 32
    head_dim: int = 128
    dim: int = 6144
    q_lora_rank: int = 2048
    rope_head_dim: int = 64
    index_topk: int = 2048
    # Force the top-k segment count instead of deriving it from the row width. 0 = derive.
    topk_nseg: int = 0
    # Chunk the score at this many positions, ranking each chunk as it is produced. 0 = AUTO, OFF keeps today's behaviour, ON=only the chunk being scored
    score_chunk: int = 0
    # Roll the position-block loop with ``nl.fori_loop``. 0 = AUTO, 1 = force on, -1 = force off.
    roll_score: int = 0
    # CP interleave granularity of the K cache, in tokens. 0 = contiguous (one block per rank).
    cp_interleave: int = 0
    # Store the paged K cache in fp8 (per-token quant over head_dim, packed 2-per-bf16 with an
    # in-row scale). False keeps the bf16 cache path byte-for-byte unchanged.
    kv_fp8: bool = False

    @property
    def kv_cache_head_width(self) -> int:
        """Minor-dim width of one paged K cache row, in the cache's container dtype (bf16).

        bf16 cache: ``head_dim`` values. fp8 cache: the fused packed row (see
        :func:`kv_row_bf16_width`), which carries the fp8 K and its per-token scale.
        """
        return kv_row_bf16_width(self.head_dim) if self.kv_fp8 else self.head_dim

    @property
    def cp_degree(self) -> int:
        """CP ranks the context is split across."""
        return max(1, self.s_global // self.s_local)

    @property
    def q_out_dim(self) -> int:
        """Width of the projected Q: all heads concatenated."""
        return self.n_heads * self.head_dim

    @property
    def q_panel_tiles(self) -> int:
        """Query tiles held in SBUF at once while K streams past (the "G" of the 2-D blocking).

        The score reuses BOTH operands: a query tile's Q is needed by every position block, and
        every K position is needed by every query tile. Holding one and streaming the other re-reads
        the other once per step -- at 1M that is ~34 GB either way. Holding G tiles instead makes
        each K read serve G tiles, cutting K traffic by G.

        G is derived from what actually fits, not chosen: per tile the panel costs a transposed Q
        (``n_heads * head_dim * 2`` B) plus an fp32 accumulator (``POS_BLOCK * 4`` B), against the
        budget less the shared K block and relu scratch. NOTE: when the top-k is fused in it will add
        ~12 KB/partition of running state PER TILE, which will force G down -- that dependency is
        why this is computed rather than hardcoded.
        """
        return max(
            1, min(self.num_q_tiles, (SBUF_BUDGET_BYTES - self._score_shared_bytes) // self._score_per_tile_bytes)
        )

    @property
    def _score_per_tile_bytes(self) -> int:
        """SBUF per partition the score scope holds PER PANEL TILE."""
        return self.n_heads * self.head_dim * 2 + POS_BLOCK * 4 + self.n_heads * 4 + 4

    @property
    def _chunk_topk_sbuf_bytes(self) -> int:
        """SBUF per partition the CHUNKED top-k holds live beside the panel. 0 when not chunked.

        Chunking ranks each chunk inside the score loop, so unlike the unfused path this scratch must
        coexist with the resident Q panel -- which is why the panel has to shrink. That is the price of
        the memory win: the panel divides K traffic, so fewer tiles means more K reads, traded against
        the score ring going from 16.4 GB to tens of MB at cp64/1M.
        """
        if not self.score_chunk:
            return 0
        k = self.index_topk
        n_ch = div_ceil(self.s_global, self.score_chunk)
        src_x = self.score_chunk // TOPK_SNAKE
        cand_x = n_ch * (k // TOPK_SNAKE)
        n_blk = div_ceil(cand_x, P_MAX)
        return (
            src_x * 2  # s1_src bf16
            + k * (2 + 4 + 4 + 4 + 4)  # s1 val/idx/pos/hi/lo
            + cand_x * (2 + 4)  # cand_v bf16 + cand_p int32
            + k * (2 + 4)  # merge val/idx
            + 2 * n_blk * P_MAX * 4  # oh_srct + oh_srct_b, full-width transposed candidates
            + src_x * 2
            + k * (2 + 4 + 4)  # this kernel's own per-chunk stage-1 scope
            + 8192  # one-hot constants, index rows and slack
        )

    @property
    def _score_shared_bytes(self) -> int:
        """SBUF per partition the score scope holds REGARDLESS of panel size.

        Every term here was missing from the original derivation, which counted only the K block and
        the relu scratch. Undercounting by ~49 KB let G come out 7, which does not actually fit: the
        fused 128k compile reported only 3908 B free, and unfusing then surfaced it as
        ``[NCC_IBVF029] q_stage and q_panel overlap in memory`` -- the 8 KB Q staging buffer had
        nowhere to go but on top of the panel. q_stage is counted here even though it is transient,
        because it must coexist with a full panel while the panel is being filled.
        """
        return (
            POS_BLOCK * 2
            + POS_BLOCK * 4
            + POS_BLOCK * 2
            + POS_BLOCK * 4
            + self.n_heads * self.head_dim * 2
            + self._chunk_topk_sbuf_bytes
        )

    @property
    def num_q_tiles(self) -> int:
        """Query tiles of ``P_MAX`` rows needed to cover ``s_local``."""
        return div_ceil(self.s_local, P_MAX)

    def validate(self) -> None:
        """Check the assumptions the kernel body relies on, with the reason for each."""
        kernel_assert(
            self.dim % BF16_K_TILE == 0,
            f"dim must be a multiple of {BF16_K_TILE} so the K-major weight load is one "
            f"rectangular DMA, got {self.dim}",
        )
        kernel_assert(
            self.q_lora_rank % BF16_K_TILE == 0,
            f"q_lora_rank must be a multiple of {BF16_K_TILE} for the same reason, got {self.q_lora_rank}",
        )
        kernel_assert(
            self.head_dim <= PSUM_FMAX,
            f"head_dim must fit one PSUM bank ({PSUM_FMAX}) so K projects in a single matmul "
            f"output tile, got {self.head_dim}",
        )
        kernel_assert(
            self.rope_head_dim <= self.head_dim and self.rope_head_dim % 2 == 0,
            f"rope_head_dim must be even and <= head_dim, got {self.rope_head_dim}",
        )
        kernel_assert(
            self.s_global >= self.s_local,
            f"s_global ({self.s_global}) cannot be smaller than s_local ({self.s_local})",
        )


# --------------------------------------------------------------------------------------------
# Shared building blocks
# --------------------------------------------------------------------------------------------


def load_weight_k_major(sbm: SbufManager, w_hbm: nl.NkiTensor, k_dim: int, out_dim: int) -> nl.NkiTensor:
    """Stage a bf16 ``[k_dim, out_dim]`` weight into SBUF as ``[P_MAX, k_tiles, out_dim]``.

    K-major because ``nc_matmul`` contracts over the PARTITION axis: the moving operand must
    present contraction rows on partitions. ``k_dim`` is a multiple of ``BF16_K_TILE``, so the
    source is perfectly rectangular and this lowers to ONE 3-D DMA (partition = row within tile,
    middle = tile, inner = column), which the hardware DGE can coalesce.

    Hoist this per batch, never per query tile: it does not depend on the queries.

    NOTE ON SIZE. The result is ``k_tiles * out_dim * 2`` bytes per partition. For Q that is
    ``16 * 4096 * 2 = 128 KB`` at GLM's 32 heads -- comfortable against the ~245 KB budget, but it
    scales with ``n_heads``, and at 128 heads it would be 384 KB and not fit. That is why the first
    indexer needed to shard this across cores, and why pinning ``n_heads = 32`` removes the need.

    Args:
        sbm: SBUF stack allocator.
        w_hbm: ``[k_dim, out_dim]`` bf16 weight in HBM.
        k_dim: contraction size (multiple of ``BF16_K_TILE``).
        out_dim: output columns.

    Returns:
        ``[P_MAX, ceil(k_dim / BF16_K_TILE), out_dim]`` bf16 SBUF tile.
    """
    k_tiles = div_ceil(k_dim, BF16_K_TILE)
    w_sb = sbm.alloc_stack((P_MAX, k_tiles, out_dim), nl.bfloat16, name=f"w_k_major{_uid()}")
    nisa.dma_copy(
        dst=w_sb[0:BF16_K_TILE, 0:k_tiles, 0:out_dim],
        src=w_hbm.ap(
            pattern=[
                [out_dim, BF16_K_TILE],  # partition: row within the K tile
                [BF16_K_TILE * out_dim, k_tiles],  # middle: which K tile
                [1, out_dim],  # inner: output column, contiguous
            ],
        ),
        dge_mode=dge_mode.hwdge,
    )
    return w_sb


def load_activation_k_major(
    dst_sb: nl.NkiTensor,
    src_hbm: nl.NkiTensor,
    row_start: int,
    rows: int,
    k_dim: int,
) -> None:
    """Transpose a whole activation from HBM into SBUF K-major, one ``dma_transpose`` per 128-column
    contraction tile, all destinations distinct so every transfer can be in flight at once.

    A single ``dma_transpose`` does the whole load -- rather than ``dma_copy`` then
    query-major tile into SBUF, then ``nc_transpose`` it through PSUM. That removes the staging
    buffer, leaves the tensor engine entirely to the matmuls, and frees the transpose's PSUM banks.

    ``nc_matmul`` contracts over the PARTITION axis, so the activation must present contraction
    rows on partitions as the stationary operand; arriving transposed is exactly what is wanted.
    Per :func:`nki.isa.dma_transpose`, ``src`` may live in HBM and needs ``shape[-1] <= 128`` with a
    2-byte dtype -- satisfied by taking one ``BF16_K_TILE``-wide column block per call.

    ``dst_sb`` must be allocated with ``align=32``: the XBar descriptor verifier requires 32-byte
    alignment on a DMA-transpose destination, and the CPU simulator does not model that, so an
    unaligned destination passes simulation and fails on hardware.

    Args:
        dst_sb: ``[P_MAX, k_tiles, P_MAX]`` bf16 destination, allocated with ``align=32``.
        src_hbm: ``[rows_total, k_dim]`` bf16 activation in HBM.
        row_start: first row of this query tile.
        rows: real rows in this query tile.
        k_dim: full contraction size (a multiple of ``BF16_K_TILE``).
    """
    for k_tile in range(div_ceil(k_dim, BF16_K_TILE)):
        k_cols = min(BF16_K_TILE, k_dim - k_tile * BF16_K_TILE)
        nisa.dma_transpose(
            dst=dst_sb[:k_cols, k_tile, :rows],
            src=src_hbm.ap(
                pattern=[
                    [k_dim, rows],
                    [1, k_cols],
                ],
                offset=row_start * k_dim + k_tile * BF16_K_TILE,
            ),
            axes=(1, 0),
        )


# --------------------------------------------------------------------------------------------
# Stage 1: projections
# --------------------------------------------------------------------------------------------


def project_q(
    sbm: SbufManager,
    qr_hbm: nl.NkiTensor,
    wq_sb: nl.NkiTensor,
    cfg: GlmIndexerConfig,
    row_start: int,
    rows: int,
    q_sb: nl.NkiTensor,
    rope: tuple = None,
) -> None:
    """Project one query tile's latent Q: ``qr[rows, q_lora_rank] @ wq -> q_sb[rows, q_out_dim]``.

    ``q_out_dim`` (4096 at GLM) exceeds one PSUM bank, so the output is tiled by ``PSUM_FMAX`` and
    each output tile accumulates over the ``q_lora_rank / 128`` contraction tiles. Banks rotate so
    adjacent output tiles can issue without waiting for the previous bank to drain, and the
    accumulate flag is explicit: with rotation, auto-detect would treat a reused bank as
    previously-written and accumulate into its prior owner's contents.

    Args:
        sbm: SBUF stack allocator.
        qr_hbm: ``[s_local, q_lora_rank]`` bf16 latent Q.
        wq_sb: hoisted K-major Q weight from :func:`load_weight_k_major`.
        cfg: kernel configuration.
        row_start: first query row of this tile.
        rows: real query rows in this tile.
        q_sb: ``[P_MAX, q_out_dim]`` bf16 destination.
    """
    k_tiles = div_ceil(cfg.q_lora_rank, BF16_K_TILE)
    out_tiles = div_ceil(cfg.q_out_dim, PSUM_FMAX)

    sbm.open_scope(name="project_q")
    qr_t_sb = sbm.alloc_stack((P_MAX, k_tiles, P_MAX), nl.bfloat16, name=f"qr_t{_uid()}", align=32)
    load_activation_k_major(qr_t_sb, qr_hbm, row_start, rows, cfg.q_lora_rank)

    n_banks = min(out_tiles, _NUM_PSUM_BANKS)
    banks = [nl.ndarray((P_MAX, PSUM_FMAX), dtype=nl.float32, buffer=nl.psum) for _ in range(n_banks)]
    for out_tile in range(out_tiles):
        col0 = out_tile * PSUM_FMAX
        cols = min(PSUM_FMAX, cfg.q_out_dim - col0)
        bank = banks[out_tile % n_banks]
        for k_tile in range(k_tiles):
            nisa.nc_matmul(
                dst=bank[:rows, :cols],
                stationary=qr_t_sb[:BF16_K_TILE, k_tile, :rows],
                moving=wq_sb[:BF16_K_TILE, k_tile, col0 : col0 + cols],
                accumulate=(k_tile != 0),
            )
        if rope is not None:
            """
            RoPE HERE, in the fp32 PSUM bank, before the bf16 read-out.
            """
            cos_sb, sin_sb = rope
            kernel_assert(
                PSUM_FMAX % cfg.head_dim == 0,
                f"roping Q in the PSUM bank needs PSUM_FMAX ({PSUM_FMAX}) to be a whole number of "
                f"head_dim ({cfg.head_dim}); otherwise a head straddles two banks",
            )
            for h_in_tile in range(cols // cfg.head_dim):
                _rope_interleaved(
                    sbm,
                    bank,
                    cos_sb,
                    sin_sb,
                    bank,
                    rows,
                    cfg.rope_head_dim,
                    col_off=h_in_tile * cfg.head_dim,
                )
        nisa.tensor_copy(dst=q_sb[:rows, col0 : col0 + cols], src=bank[:rows, :cols])
    sbm.close_scope()


def project_k(
    sbm: SbufManager,
    x_hbm: nl.NkiTensor,
    wk_sb: nl.NkiTensor,
    cfg: GlmIndexerConfig,
    row_start: int,
    rows: int,
    k_sb: nl.NkiTensor,
) -> None:
    """Project one query tile's K: ``x[rows, dim] @ wk -> k_sb[rows, head_dim]``.

    ``head_dim <= PSUM_FMAX``, so the whole output is a single PSUM tile and there is no output
    tiling. Output stays query-major because the next stage (LayerNorm + RoPE) wants it that way.

    Args:
        sbm: SBUF stack allocator.
        x_hbm: ``[s_local, dim]`` bf16 hidden states.
        wk_sb: hoisted K-major K weight from :func:`load_weight_k_major`.
        cfg: kernel configuration.
        row_start: first row of this tile.
        rows: real rows in this tile.
        k_sb: ``[P_MAX, head_dim]`` f32 destination.
    """
    k_tiles = div_ceil(cfg.dim, BF16_K_TILE)

    sbm.open_scope(name="project_k")
    x_t_sb = sbm.alloc_stack((P_MAX, k_tiles, P_MAX), nl.bfloat16, name=f"x_t{_uid()}", align=32)
    load_activation_k_major(x_t_sb, x_hbm, row_start, rows, cfg.dim)

    # head_dim <= PSUM_FMAX, so the whole output is one bank and there is no output tiling.
    psum = nl.ndarray((P_MAX, cfg.head_dim), dtype=nl.float32, buffer=nl.psum)
    for k_tile in range(k_tiles):
        nisa.nc_matmul(
            dst=psum[:rows, : cfg.head_dim],
            stationary=x_t_sb[:BF16_K_TILE, k_tile, :rows],
            moving=wk_sb[:BF16_K_TILE, k_tile, : cfg.head_dim],
            accumulate=(k_tile != 0),
        )
    nisa.tensor_copy(dst=k_sb[:rows, : cfg.head_dim], src=psum[:rows, : cfg.head_dim])
    sbm.close_scope()


# --------------------------------------------------------------------------------------------
# Stage 2: K post-processing (LayerNorm + RoPE)
# --------------------------------------------------------------------------------------------


def _layernorm(
    sbm: SbufManager,
    x_sb: nl.NkiTensor,
    gamma_sb: nl.NkiTensor,
    beta_sb: nl.NkiTensor,
    out_sb: nl.NkiTensor,
    S: int,
    H: int,
    eps: float = 1e-6,
) -> None:
    """LayerNorm with fused gamma/beta application.

    Uses nisa.bn_stats + nisa.bn_aggr to compute mean/var in one Vector Engine pass,
    then a fused scalar_tensor_tensor for ((x - mean) * rvar * gamma) + beta.

    gamma_sb/beta_sb are [P_MAX, H] (pre-broadcast by caller).
    """
    BN_STATS_TILE_SIZE = 512
    BN_STATS_DST_SIZE = 6
    NUM_BN_TILES = div_ceil(H, BN_STATS_TILE_SIZE)

    sbm.open_scope(name="layernorm")

    bn_stats_sb = sbm.alloc_stack((P_MAX, BN_STATS_DST_SIZE * NUM_BN_TILES), nl.float32, name=f"bn_stats_sb{_uid()}")
    aggr_sb = sbm.alloc_stack((P_MAX, 2), nl.float32, name=f"aggr_sb{_uid()}")  # [mean, rvar]

    for bn_tile_idx in nl.affine_range(NUM_BN_TILES):
        bn_off = bn_tile_idx * BN_STATS_TILE_SIZE
        bn_sz = min(BN_STATS_TILE_SIZE, H - bn_off)
        nisa.bn_stats(
            dst=bn_stats_sb[:S, nl.ds(bn_tile_idx * BN_STATS_DST_SIZE, BN_STATS_DST_SIZE)],
            data=x_sb[:S, nl.ds(bn_off, bn_sz)],
        )

    nisa.bn_aggr(
        dst=aggr_sb[:S, 0:2],
        data=bn_stats_sb[:S, 0 : BN_STATS_DST_SIZE * NUM_BN_TILES],
    )

    # rvar = 1 / sqrt(var + eps) in place.
    nisa.activation(
        dst=aggr_sb[:S, 1:2],
        op=nl.rsqrt,
        data=aggr_sb[:S, 1:2],
        bias=eps,
    )

    # x_norm = (x - mean) * rvar  (fused subtract + multiply via tensor_scalar two-op)
    x_norm = sbm.alloc_stack((P_MAX, H), nl.float32, name=f"x_norm{_uid()}")
    nisa.tensor_scalar(
        dst=x_norm[:S, :H],
        data=x_sb[:S, :H],
        op0=nl.subtract,
        operand0=aggr_sb[:S, 0:1],
        op1=nl.multiply,
        operand1=aggr_sb[:S, 1:2],
    )

    # out = x_norm * gamma, then + beta below (op0 identity multiply by 1.0, op1 multiplies gamma)
    nisa.scalar_tensor_tensor(
        dst=out_sb[:S, :H],
        data=x_norm[:S, :H],
        op0=nl.multiply,
        operand0=1.0,
        op1=nl.multiply,
        operand1=gamma_sb[:S, :H],
    )
    nisa.tensor_tensor(
        dst=out_sb[:S, :H],
        data1=out_sb[:S, :H],
        data2=beta_sb[:S, :H],
        op=nl.add,
    )

    sbm.close_scope()


# RoPE — non-interleaved, applied once per S-tile to all Q heads or to K.
def _rope_non_interleaved(
    sbm: SbufManager,
    x_sb: nl.NkiTensor,
    cos_sb: nl.NkiTensor,
    sin_sb: nl.NkiTensor,
    out_sb: nl.NkiTensor,
    S: int,
    D: int,
) -> None:
    """In-place non-interleaved RoPE on a [S, D] slab using minimum ops.

    out = [X1*cos - X2*sin, X1*sin + X2*cos]
    """
    D_half = D // 2
    sbm.open_scope(name="rope")

    t_neg_x2_sin = sbm.alloc_stack((P_MAX, D_half), nl.float32, name=f"t_neg_x2_sin{_uid()}")
    t_x1_sin = sbm.alloc_stack((P_MAX, D_half), nl.float32, name=f"t_x1_sin{_uid()}")
    t_x_cos = sbm.alloc_stack((P_MAX, D), nl.float32, name=f"t_x_cos{_uid()}")

    # -X2 * sin
    nisa.scalar_tensor_tensor(
        dst=t_neg_x2_sin[:S, :D_half],
        data=x_sb[:S, D_half:D],
        op0=nl.multiply,
        operand0=-1.0,
        op1=nl.multiply,
        operand1=sin_sb[:S, :D_half],
    )
    # X1 * sin
    nisa.tensor_tensor(
        dst=t_x1_sin[:S, :D_half],
        data1=x_sb[:S, :D_half],
        data2=sin_sb[:S, :D_half],
        op=nl.multiply,
    )
    # X * cos (full slab)
    # cos_sb is [S, D_half]; broadcast across the two halves.
    nisa.tensor_tensor(
        dst=t_x_cos[:S, :D_half],
        data1=x_sb[:S, :D_half],
        data2=cos_sb[:S, :D_half],
        op=nl.multiply,
    )
    nisa.tensor_tensor(
        dst=t_x_cos[:S, D_half:D],
        data1=x_sb[:S, D_half:D],
        data2=cos_sb[:S, :D_half],
        op=nl.multiply,
    )
    # First half of output: X1*cos - X2*sin = t_x_cos[:D/2] + t_neg_x2_sin
    nisa.tensor_tensor(
        dst=out_sb[:S, :D_half],
        data1=t_x_cos[:S, :D_half],
        data2=t_neg_x2_sin[:S, :D_half],
        op=nl.add,
    )
    # Second half: X1*sin + X2*cos = t_x1_sin + t_x_cos[D/2:]
    nisa.tensor_tensor(
        dst=out_sb[:S, D_half:D],
        data1=t_x1_sin[:S, :D_half],
        data2=t_x_cos[:S, D_half:D],
        op=nl.add,
    )

    sbm.close_scope()


def _rope_interleaved(
    sbm: SbufManager,
    x_sb: nl.NkiTensor,
    cos_sb: nl.NkiTensor,
    sin_sb: nl.NkiTensor,
    out_sb: nl.NkiTensor,
    S: int,
    D: int,
    col_off: int = 0,
) -> None:
    """INTERLEAVED (adjacent-pair) RoPE on [S, D] columns at ``col_off`` of a slab.

    Pair j is columns (2j, 2j+1), rotated by the same angle the half-split variant uses, so only the
    element pairing changes -- reached with stride-2 access patterns, identical op count.

    ``x_sb`` / ``out_sb`` are the FULL buffers, not ``[:, :D]`` slices: an ``ap()`` partition stride
    must equal the tensor's own free width, so a sliced view would carry the wrong stride.
    """
    D_half = D // 2
    x_row = x_sb.shape[1]
    out_row = out_sb.shape[1]
    sbm.open_scope(name="rope_il")

    x_even = x_sb.ap(pattern=[[x_row, S], [2, D_half]], offset=col_off)
    x_odd = x_sb.ap(pattern=[[x_row, S], [2, D_half]], offset=col_off + 1)
    o_even = out_sb.ap(pattern=[[out_row, S], [2, D_half]], offset=col_off)
    o_odd = out_sb.ap(pattern=[[out_row, S], [2, D_half]], offset=col_off + 1)

    # Stage the two cross terms first: out may alias x, so read x before overwriting it.
    t_even = sbm.alloc_stack((P_MAX, D_half), nl.float32, name=f"t_even{_uid()}")
    t_odd = sbm.alloc_stack((P_MAX, D_half), nl.float32, name=f"t_odd{_uid()}")
    nisa.tensor_tensor(dst=t_even[:S, :D_half], data1=x_odd, data2=sin_sb[:S, :D_half], op=nl.multiply)
    nisa.tensor_tensor(dst=t_odd[:S, :D_half], data1=x_even, data2=sin_sb[:S, :D_half], op=nl.multiply)

    # cos terms, then combine: out_even = x_even*cos - x_odd*sin ; out_odd = x_odd*cos + x_even*sin.
    nisa.tensor_tensor(dst=o_even, data1=x_even, data2=cos_sb[:S, :D_half], op=nl.multiply)
    nisa.tensor_tensor(dst=o_odd, data1=x_odd, data2=cos_sb[:S, :D_half], op=nl.multiply)
    nisa.tensor_tensor(dst=o_even, data1=o_even, data2=t_even[:S, :D_half], op=nl.subtract)
    nisa.tensor_tensor(dst=o_odd, data1=o_odd, data2=t_odd[:S, :D_half], op=nl.add)

    sbm.close_scope()


# K post-processing: LayerNorm + RoPE. Biased variance with eps inside the rsqrt, and INTERLEAVED
# (adjacent-pair) RoPE, which is GLM-MoE-DSA's pairing -- hence the layout argument.


def fused_layernorm_rope_k(
    sbm: SbufManager,
    k_sb: nl.NkiTensor,
    gamma_sb: nl.NkiTensor,
    beta_sb: nl.NkiTensor,
    cos_sb: nl.NkiTensor,
    sin_sb: nl.NkiTensor,
    k_out: nl.NkiTensor,
    S: int,
    head_dim: int,
    rope_head_dim: int,
    rope_layout: RopeLayout = RopeLayout.HALF_SPLIT,
) -> None:
    """LayerNorm K over the full head_dim, then RoPE the leading ``rope_head_dim`` columns in place.

    Any trailing nope columns are copied through unchanged.

    Args:
        sbm: SBUF stack allocator used for scratch buffers.
        k_sb: ``[S, head_dim]`` input K in SBUF.
        gamma_sb: ``[P_MAX, head_dim]`` LayerNorm scale (pre-broadcast by caller).
        beta_sb: ``[P_MAX, head_dim]`` LayerNorm bias (pre-broadcast by caller).
        cos_sb: ``[S, rope_head_dim // 2]`` RoPE cosine frequencies.
        sin_sb: ``[S, rope_head_dim // 2]`` RoPE sine frequencies.
        k_out: ``[S, head_dim]`` output buffer, written in place.
        S: active sequence length (partition rows) for this tile.
        head_dim: total head dimension.
        rope_head_dim: leading number of columns to apply RoPE to.
        rope_layout: element pairing; ``INTERLEAVED`` is GLM-MoE-DSA's.
    """
    sbm.open_scope(name="ln_rope_k")

    k_normed = sbm.alloc_stack((P_MAX, head_dim), nl.float32, name=f"k_normed{_uid()}")
    _layernorm(sbm, k_sb, gamma_sb, beta_sb, k_normed, S, head_dim)

    # RoPE on the rope_head_dim slice; copy nope tail through. The interleaved variant takes the
    # FULL buffers (see its docstring: ap() partition stride must be the parent row width).
    if rope_layout.is_interleaved():
        _rope_interleaved(sbm, k_normed, cos_sb, sin_sb, k_out, S, rope_head_dim)
    else:
        _rope_non_interleaved(
            sbm, k_normed[:, :rope_head_dim], cos_sb, sin_sb, k_out[:, :rope_head_dim], S, rope_head_dim
        )
    if head_dim > rope_head_dim:
        nisa.tensor_copy(dst=k_out[:, rope_head_dim:head_dim], src=k_normed[:, rope_head_dim:head_dim])

    sbm.close_scope()


def broadcast_norm_weight(sbm: SbufManager, w_hbm: nl.NkiTensor, head_dim: int, tag: str) -> nl.NkiTensor:
    """Stage a ``[head_dim]`` LayerNorm weight broadcast across all partitions.

    ``fused_layernorm_rope_k`` wants gamma/beta as ``[P_MAX, head_dim]`` so the fused
    ``scalar_tensor_tensor`` can read one value per (row, column) without a per-row reload. The
    broadcast is a view on the HBM side, so this is one DMA and costs 0.5 KB/partition.

    Loop-invariant -- hoist once per kernel, never per query tile.
    """
    w_sb = sbm.alloc_stack((P_MAX, head_dim), nl.float32, name=f"{tag}{_uid()}")
    nisa.dma_copy(dst=w_sb[:P_MAX, :head_dim], src=w_hbm.reshape((1, head_dim)).broadcast(dim=0, size=P_MAX))
    return w_sb


def load_rope_tables(
    sbm: SbufManager,
    cos_hbm: nl.NkiTensor,
    sin_hbm: nl.NkiTensor,
    cfg: "GlmIndexerConfig",
    row_start: int,
    rows: int,
) -> tuple:
    """This tile's ``cos``/``sin`` rows, shared by the Q and K rotations.

    ONE table for every head: HF's indexer passes ``unsqueeze_dim=2`` to
    ``apply_rotary_pos_emb_interleave``, so the angle depends on the token position only, not on the
    head. Loaded once per tile and handed to :func:`project_q`, which rotates each head in place.
    """
    rope_half = cfg.rope_head_dim // 2
    cos_sb = sbm.alloc_stack((P_MAX, rope_half), nl.float32, name=f"qcos{_uid()}")
    sin_sb = sbm.alloc_stack((P_MAX, rope_half), nl.float32, name=f"qsin{_uid()}")
    nisa.dma_copy(dst=cos_sb[:rows, :rope_half], src=cos_hbm[row_start : row_start + rows, :rope_half])
    nisa.dma_copy(dst=sin_sb[:rows, :rope_half], src=sin_hbm[row_start : row_start + rows, :rope_half])
    return cos_sb, sin_sb


def postprocess_k(
    sbm: SbufManager,
    k_sb: nl.NkiTensor,
    gamma_sb: nl.NkiTensor,
    beta_sb: nl.NkiTensor,
    cos_hbm: nl.NkiTensor,
    sin_hbm: nl.NkiTensor,
    cfg: GlmIndexerConfig,
    row_start: int,
    rows: int,
    k_out_sb: nl.NkiTensor,
) -> None:
    """LayerNorm the projected K, then RoPE its leading ``rope_head_dim`` columns.

    Reuses ``fused_layernorm_rope_k``: it is public, already validated on
    hardware for both RoPE pairings, and re-deriving LayerNorm's ``bn_stats``/``bn_aggr`` sequence
    here would duplicate numerically sensitive code for no gain. GLM uses ``INTERLEAVED`` pairing
    (adjacent columns ``2j``/``2j+1``), where DeepSeek-V3.2 uses ``HALF_SPLIT``; the angle set is the
    same, only the pairing differs. Inline it here if that package is ever retired.

    ``cos``/``sin`` carry ``rope_head_dim // 2`` angles per row and are indexed by ABSOLUTE query
    row, so each tile loads its own slice -- they are not loop-invariant like gamma/beta.

    Args:
        sbm: SBUF stack allocator.
        k_sb: ``[P_MAX, head_dim]`` fp32 projected K for this tile.
        gamma_sb: ``[P_MAX, head_dim]`` fp32 LayerNorm scale, pre-broadcast.
        beta_sb: ``[P_MAX, head_dim]`` fp32 LayerNorm bias, pre-broadcast.
        cos_hbm: ``[s_local, rope_head_dim // 2]`` fp32 cosine table.
        sin_hbm: ``[s_local, rope_head_dim // 2]`` fp32 sine table.
        cfg: kernel configuration.
        row_start: first query row of this tile.
        rows: real rows in this tile.
        k_out_sb: ``[P_MAX, head_dim]`` fp32 destination.
    """
    rope_half = cfg.rope_head_dim // 2

    sbm.open_scope(name="postprocess_k")
    cos_sb = sbm.alloc_stack((P_MAX, rope_half), nl.float32, name=f"cos{_uid()}")
    sin_sb = sbm.alloc_stack((P_MAX, rope_half), nl.float32, name=f"sin{_uid()}")
    nisa.dma_copy(dst=cos_sb[:rows, :rope_half], src=cos_hbm[row_start : row_start + rows, :rope_half])
    nisa.dma_copy(dst=sin_sb[:rows, :rope_half], src=sin_hbm[row_start : row_start + rows, :rope_half])

    fused_layernorm_rope_k(
        sbm,
        k_sb,
        gamma_sb,
        beta_sb,
        cos_sb,
        sin_sb,
        k_out_sb,
        rows,
        cfg.head_dim,
        cfg.rope_head_dim,
        rope_layout=RopeLayout.INTERLEAVED,
    )
    sbm.close_scope()


# --------------------------------------------------------------------------------------------
# Stage 3: paged K cache write
# --------------------------------------------------------------------------------------------


def quantize_k_per_token(
    sbm: SbufManager,
    k_sb: nl.NkiTensor,
    dst_fp8_ap: nl.NkiTensor,
    scale_sb: nl.NkiTensor,
    rows: int,
    head_dim: int,
    fp8_amax: float,
) -> None:
    """Quantize this tile's post-processed K to fp8 ON THE FLY, one scale per token.

    Tokens sit on the partition axis and ``head_dim`` on the free axis, so the per-token amax is a
    free-axis reduce and the rescale is a per-partition scalar -- the cheap direction for both.

    Per TOKEN, not per channel: a channel scale would have to be identical for every row ever
    written across every CP rank and chunk (a top-k gather mixes rows from all of them), so it could
    only be a static calibrated constant. A token's scale travels in its own row. One scale for all
    ``head_dim`` channels, so the consumer dequantizes the whole dot product with one multiply per
    key rather than a partial accumulation per block.

    Writes the DEQUANTIZATION factor (``amax / fp8_amax``, what the score side multiplies back in)
    to ``scale_sb`` and the quantized K to ``dst_fp8_ap`` (an fp8-typed access pattern into the
    fused cache row's latent span).
    """
    sbm.open_scope(name="quantize_k")
    amax_sb = sbm.alloc_stack((P_MAX, 1), nl.float32, name=f"k_amax{_uid()}")
    neg_min_sb = sbm.alloc_stack((P_MAX, 1), nl.float32, name=f"k_neg_min{_uid()}")
    recip_sb = sbm.alloc_stack((P_MAX, 1), nl.float32, name=f"k_qtz_recip{_uid()}")

    # max|k| as max(max(x), -min(x)): an abs-then-max cannot be fused into one reduce (the Scalar
    # accumulator only sums), so two free-axis reduces plus a combine, with negate giving -min free.
    nisa.tensor_reduce(amax_sb[:rows, :], op=nl.maximum, data=k_sb[:rows, :head_dim], axis=1)
    nisa.tensor_reduce(neg_min_sb[:rows, :], op=nl.minimum, data=k_sb[:rows, :head_dim], axis=1, negate=True)
    nisa.tensor_tensor(amax_sb[:rows, :], amax_sb[:rows, :], neg_min_sb[:rows, :], op=nl.maximum)
    # Padding tokens are all-zero; the floor keeps the reciprocal finite and the row quantizes to 0.
    nisa.tensor_scalar(amax_sb[:rows, :], amax_sb[:rows, :], op0=nl.maximum, operand0=_AMAX_FLOOR)
    # Dequant factor = amax / fp8_amax; store it, then quantize by its reciprocal (exact round trip).
    nisa.tensor_scalar(scale_sb[:rows, :], amax_sb[:rows, :], op0=nl.multiply, operand0=1.0 / fp8_amax)
    nisa.reciprocal(recip_sb[:rows, :], scale_sb[:rows, :])
    nisa.tensor_scalar(dst_fp8_ap, k_sb[:rows, :head_dim], op0=nl.multiply, operand0=recip_sb[:rows, :])
    sbm.close_scope()


def store_k_paged(
    sbm: SbufManager,
    k_sb: nl.NkiTensor,
    k_cache_hbm: nl.NkiTensor,
    slot_mapping_hbm: nl.NkiTensor,
    cfg: GlmIndexerConfig,
    row_start: int,
    rows: int,
    fp8_dtype=None,
    local_out: nl.NkiTensor = None,
) -> None:
    """Scatter this tile's post-processed K into the paged cache at the slots ``slot_mapping`` names.

    vLLM convention: ``slot_mapping[token]`` is a flat ``block_id * block_size + offset``, so the
    cache is addressed as ``[n_slots, kv_cache_head_width]`` and only the read side needs the block
    table. Padded tokens carry ``-1``, hence ``oob_mode.skip``.

    Stored TOKEN-major so a token's row is one descriptor; the score transposes on read instead,
    which is legal only while a container is 2 bytes and ``head_dim`` is 128 (``dma_transpose`` needs
    ``src.shape[-1] <= 128`` and a 2-byte dtype).

    Two cache layouts, selected by ``cfg.kv_fp8``:

    * bf16 (default): the row is ``head_dim`` bf16 values, cast from the fp32 K.
    * fp8: the row is the fused packed layout -- ``head_dim`` fp8 K values bitcast two-per-bf16
      container in natural channel order, then the per-token bf16 dequant scale, then padding (see
      :func:`kv_row_bf16_width`). The container stays bf16 so the row is one dtype (one descriptor
      per token) and ``dma_transpose`` stays legal; only the latent span is not meaningful AS bf16
      -- the score side bitcasts it back to fp8. Channels are paired, not tokens, so each row is
      wholly one token and the per-token scatter at an arbitrary slot is unchanged.

    Args:
        sbm: SBUF stack allocator.
        k_sb: ``[P_MAX, head_dim]`` fp32 K after LayerNorm + RoPE.
        k_cache_hbm: paged cache, any shape whose flattened leading dims are slots and whose minor
            dim is ``cfg.kv_cache_head_width``.
        slot_mapping_hbm: ``[s_local]`` int32 flat slot per token, ``-1`` for padding.
        cfg: kernel configuration.
        row_start: first query row of this tile.
        rows: real rows in this tile.
        fp8_dtype: fp8 nki dtype for the latent span; required when ``cfg.kv_fp8``.
        local_out: ``[s_local, kv_cache_head_width]`` dense buffer for the CP collective. When given,
            the same row this tile scatters is also copied here at ``row_start`` so the all-gather
            moves the packed fp8 row (scale in-row), never a re-quantized copy.
    """
    head_width = cfg.kv_cache_head_width
    n_slots = k_cache_hbm.size // head_width

    sbm.open_scope(name="store_k_paged")
    slot_sb = sbm.alloc_stack((P_MAX, 1), nl.int32, name=f"k_slot{_uid()}")
    nisa.dma_copy(
        dst=slot_sb[:rows, :],
        src=slot_mapping_hbm.reshape((cfg.s_local,)).ap(pattern=[[1, rows], [1, 1]], offset=row_start),
        dge_mode=dge_mode.hwdge,
    )

    if cfg.kv_fp8:
        kernel_assert(fp8_dtype is not None, "fp8 cache store needs fp8_dtype")
        fp8_amax = get_max_positive_value_for_dtype(fp8_dtype)
        lat_bf16 = cfg.head_dim // 2  # bf16 containers holding the fp8 latent

        # One staging tile holding the whole fused row, so the scatter is a single descriptor.
        row_sb = sbm.alloc_stack((P_MAX, head_width), nl.bfloat16, name=f"k_row{_uid()}")
        scale_sb = sbm.alloc_stack((P_MAX, 1), nl.float32, name=f"k_scale{_uid()}")
        # The latent span addressed as fp8: head_dim 1-byte elements over head_dim//2 bf16
        # containers. Under a 1-byte dtype view the partition stride counts fp8 elements, hence
        # head_width * 2; channel c pairs with c+1 in container c // 2.
        lat_fp8_ap = row_sb.ap(pattern=[[head_width * 2, rows], [1, cfg.head_dim]], offset=0, dtype=fp8_dtype)
        quantize_k_per_token(sbm, k_sb, lat_fp8_ap, scale_sb, rows, cfg.head_dim, fp8_amax)
        # The per-token dequant scale is a genuine bf16 field right after the latent containers.
        nisa.tensor_copy(dst=row_sb[:rows, lat_bf16 : lat_bf16 + 1], src=scale_sb[:rows, 0:1])
        src_row = row_sb
    else:
        # Cast to the cache dtype first so the scatter moves one contiguous bf16 row per token.
        k_bf16_sb = sbm.alloc_stack((P_MAX, cfg.head_dim), nl.bfloat16, name=f"k_bf16{_uid()}")
        nisa.tensor_copy(dst=k_bf16_sb[:rows, : cfg.head_dim], src=k_sb[:rows, : cfg.head_dim])
        src_row = k_bf16_sb

    nisa.dma_copy(
        dst=k_cache_hbm.reshape((n_slots, head_width)).ap(
            pattern=[[head_width, rows], [1, head_width]],
            offset=0,
            vector_offset=slot_sb[:rows, :],
            indirect_dim=0,
        ),
        src=src_row[:rows, :head_width],
        oob_mode=nisa.oob_mode.skip,
        dge_mode=dge_mode.swdge,
    )
    if local_out is not None:
        # Same row the collective gathers -- packed fp8 with the scale in-row, so no re-quantize.
        nisa.dma_copy(
            dst=local_out.reshape((cfg.s_local, head_width))[row_start : row_start + rows, :head_width],
            src=src_row[:rows, :head_width],
        )
    sbm.close_scope()


def dequant_k_packed(
    sbm: SbufManager,
    k_packed_hbm: nl.NkiTensor,
    k_deq_hbm: nl.NkiTensor,
    cfg: GlmIndexerConfig,
    ctx: int,
    fp8_dtype,
) -> None:
    """Dequantize a packed fp8 K cache to dense bf16 ``[ctx, head_dim]``, AFTER the CP gather.

    ``k_packed_hbm`` holds ``ctx`` fused rows of ``kv_cache_head_width`` bf16 containers: the fp8 K
    bitcast two-per-bf16, then the per-token dequant scale. This reads each row's fp8 latent back
    (an fp8-typed access pattern with the paired-channel stride), multiplies by the row's scale, and
    writes dense bf16 K -- exactly the ``[ctx, head_dim]`` layout the score path already consumes.

    Placed after the gather so the collective moved the 1-byte-per-channel packed rows, not bf16 K;
    the dequant is local and materializes bf16 only for the score. Tokens sit on the partition axis
    (128 at a time), so the scale multiply is a per-partition scalar -- the cheap direction.
    """
    head_width = cfg.kv_cache_head_width
    lat_bf16 = cfg.head_dim // 2

    sbm.open_scope(name="dequant_k")
    packed = k_packed_hbm.reshape((k_packed_hbm.size // head_width, head_width))
    dense = k_deq_hbm.reshape((k_deq_hbm.size // cfg.head_dim, cfg.head_dim))
    # Loop-invariant scratch: shape does not depend on ``base`` (only the ``[:rows]`` slice varies),
    # so allocate ONCE and reuse. Allocating inside the loop bumped the stack per unrolled iteration
    # -- nothing is freed until close_scope -- so peak SBUF and the alloc_stack name count grew with
    # ctx // P_MAX (8192 iterations at 1M). Reusing one set keeps both constant.
    row_sb = sbm.alloc_stack((P_MAX, head_width), nl.bfloat16, name=f"deq_row{_uid()}")
    scale_sb = sbm.alloc_stack((P_MAX, 1), nl.float32, name=f"deq_scale{_uid()}")
    k_deq_sb = sbm.alloc_stack((P_MAX, cfg.head_dim), nl.bfloat16, name=f"deq_k{_uid()}")
    for base in range(0, ctx, P_MAX):
        rows = min(P_MAX, ctx - base)
        nisa.dma_copy(dst=row_sb[:rows, :head_width], src=packed[base : base + rows, :head_width])
        # The fp8 latent, un-paired by the stride-2 fp8 view (channel c in container c // 2).
        lat_fp8_ap = row_sb.ap(pattern=[[head_width * 2, rows], [1, cfg.head_dim]], offset=0, dtype=fp8_dtype)
        # Per-token scale is the bf16 field right after the latent containers.
        nisa.tensor_copy(dst=scale_sb[:rows, 0:1], src=row_sb[:rows, lat_bf16 : lat_bf16 + 1])
        # Dequant: fp8 K * per-token scale -> bf16. The multiply casts fp8 -> the fp32 accumulator
        # and the write casts to bf16, matching the K precision the score matmul otherwise sees.
        nisa.tensor_scalar(
            dst=k_deq_sb[:rows, : cfg.head_dim],
            data=lat_fp8_ap,
            op0=nl.multiply,
            operand0=scale_sb[:rows, 0:1],
        )
        nisa.dma_copy(dst=dense[base : base + rows, : cfg.head_dim], src=k_deq_sb[:rows, : cfg.head_dim])
    sbm.close_scope()


# --------------------------------------------------------------------------------------------
# Stage 4: CP all-gather of this chunk's K
# --------------------------------------------------------------------------------------------


def cp_replica_group(cp_degree: int, replica_groups: tuple = None) -> "ncc.ReplicaGroup":
    """The CP rank-lists this kernel's collectives run over.

    ``replica_groups`` comes from the framework, which is the only thing that knows how CP interacts
    with TP/PP -- a rank's CP peers are contiguous global ranks only when CP is the outermost split.
    It must be a compile-time list of rank-lists, not a tensor: the participant set defines the NEFF's
    collective topology. Defaults to ``[[0, cp_degree)]``, which is correct only for pure CP.
    """
    if replica_groups is None:
        return ncc.ReplicaGroup([list(range(cp_degree))])
    groups = [list(g) for g in replica_groups]
    kernel_assert(
        all(len(g) == cp_degree for g in groups),
        f"every CP replica group must hold cp_degree={cp_degree} ranks; got {[len(g) for g in groups]}",
    )
    return ncc.ReplicaGroup(groups)


def gather_k_across_cp(
    k_local_hbm: nl.NkiTensor,
    k_gathered_hbm: nl.NkiTensor,
    cp_degree: int,
    replica_groups: tuple = None,
) -> None:
    """All-gather this rank's chunk of K across the CP ranks, inside the kernel.

    Each CP rank projects K only for the ``s_local`` tokens it owns, but every rank's queries score
    against the whole ``s_global`` context, so the chunk has to be shared. Doing it in-kernel avoids
    the write-out / gather / read-back round trip a framework-level collective forces, which at
    CP64 / 1M is the dominant traffic in this kernel.

    ``collective_dim=0`` concatenates along the token axis, so rank r's ``s_local`` rows land at
    ``[r * s_local, (r+1) * s_local)`` of the destination -- the natural global order, provided the
    replica group lists CP ranks in context order.

    The destination is ``shared_hbm`` when LNC > 1: an all-gather produces identical output on every
    rank, so both NeuronCores on a stack can read one allocation instead of each holding a private
    copy, halving both capacity and collective bandwidth. With LNC=1 the source must be
    ``private_hbm`` per the ``nki.collectives.all_gather`` contract.

    Dtype/width agnostic: with ``kv_fp8`` the rows are the fused packed fp8 row (fp8 K + in-row
    scale, ``kv_cache_head_width`` bf16 containers), so the gather moves ~half the bytes of dense
    bf16 K; the dequant runs locally after this returns.

    Args:
        k_local_hbm: ``[s_local, kv_cache_head_width]`` bf16-container, this rank's post-processed K.
        k_gathered_hbm: ``[cp_degree * s_local, kv_cache_head_width]`` destination.
        cp_degree: CP ranks participating.
    """
    ncc.all_gather(
        dsts=[k_gathered_hbm],
        srcs=[k_local_hbm],
        replica_group=cp_replica_group(cp_degree, replica_groups),
        collective_dim=0,
    )


# --------------------------------------------------------------------------------------------
# Stage 5: blocked score
# --------------------------------------------------------------------------------------------


HEAD_CHUNK = 8
"""Heads transposed per ``dma_transpose``.

A 3-D transpose with ``axes=(2, 1, 0)`` flips ``[rows, HEAD_CHUNK, head_dim]`` to
``[head_dim, HEAD_CHUNK, rows]``, producing EVERY head in the chunk in one DMA. Per-head transposes
cost ``n_heads`` DMA ops instead of ``n_heads / HEAD_CHUNK``."""


def linearize_pages(
    sbm: SbufManager,
    k_cache_hbm: nl.NkiTensor,
    block_table_hbm: nl.NkiTensor,
    k_lin_hbm: nl.NkiTensor,
    cfg: "GlmIndexerConfig",
    end_pos: int,
    block_size: int,
) -> None:
    """Gather the paged cache into CONTIGUOUS logical order: ``k_lin[p] = cache[slot(p)]``.

    ``slot(p) = block_table[p // block_size] * block_size + p % block_size``, the vLLM convention
    the store side already uses. Reading through the table is what lets a resident prefix be scored.

    Linearized in HBM rather than transposed straight from the pages: the score's ``dma_transpose``
    needs an HBM source, and staging pages token-major in SBUF first cannot work because 4096 tokens
    do not fit in 128 partitions. Costs one extra ``T * head_dim`` bf16 tensor and one pass over K.

    This also covers CP token interleaving: the layout is expressed by which ids the table lists and
    in what order, so neither the score nor the mask needs an interleave parameter.

    Row-width agnostic: a cache row is ``cfg.kv_cache_head_width`` bf16 containers -- ``head_dim`` for
    the bf16 cache, or the fused packed row (fp8 K + in-row scale) for ``kv_fp8``. The gather moves
    whole rows and never reads inside one, so it is correct for either; ``k_lin`` comes out in the
    same container dtype. For fp8 the caller dequantizes the linearized rows before the score.
    """
    kernel_assert(
        POS_BLOCK % block_size == 0,
        f"block_size ({block_size}) must divide POS_BLOCK ({POS_BLOCK}); a partial page at a block "
        f"boundary would need a masked tail read",
    )
    head_width = cfg.kv_cache_head_width
    n_blocks = div_ceil(end_pos, block_size)
    page = block_size * head_width
    n_pages_cache = k_cache_hbm.size // page
    rows_total = n_blocks * block_size

    sbm.open_scope(name="linearize_pages")
    # BATCHED GATHER: with a page as the unit row, the index tensor IS the block table -- one
    # descriptor moves 128*M pages instead of one DMA per page. The hardware reads a 2D
    # vector_offset column-major (indices.T.flatten()), so partition i / column j must hold
    # block_table[j * P_MAX + i] to land pages in linear order.
    if n_blocks % P_MAX == 0 and k_lin_hbm.size // head_width >= rows_total:
        cache_pages = k_cache_hbm.reshape((n_pages_cache, page))
        lin_pages = k_lin_hbm.reshape((k_lin_hbm.size // page, page))
        idx = sbm.alloc_stack((P_MAX, 1), nl.uint32, name=f"bt{_uid()}")
        # An indirect DMA needs its non-indirect side in SBUF ([NCC_IBIR231]), so the gather stages
        # 128 whole pages in SBUF and one contiguous write pushes them out.
        stage = sbm.alloc_stack((P_MAX, page), nl.bfloat16, name=f"pg{_uid()}", align=32)
        for gb in range(0, n_blocks, P_MAX):
            nisa.dma_copy(
                dst=idx[0:P_MAX, 0:1],
                src=block_table_hbm.reshape((block_table_hbm.size, 1))[gb : gb + P_MAX, 0:1],
            )
            nisa.dma_copy(
                dst=stage[0:P_MAX, 0:page],
                src=cache_pages.ap(
                    pattern=[[page, P_MAX], [1, page]],
                    offset=0,
                    vector_offset=idx[0:P_MAX, 0:1],
                    indirect_dim=0,
                ),
                dge_mode=dge_mode.swdge,
            )
            nisa.dma_copy(dst=lin_pages[gb : gb + P_MAX, 0:page], src=stage[0:P_MAX, 0:page])
        sbm.close_scope()
        return

    # Fallback: one descriptor per page. Page ids pre-scaled to a ROW offset, since scalar_offset
    # counts the access pattern's leading stride.
    flat_cache = k_cache_hbm.reshape((k_cache_hbm.size // head_width, head_width))
    bt = sbm.alloc_stack((1, n_blocks), nl.uint32, name=f"bt{_uid()}")
    nisa.dma_copy(
        dst=bt[0:1, 0:n_blocks],
        src=block_table_hbm.reshape((1, block_table_hbm.size))[0:1, 0:n_blocks],
    )
    nisa.tensor_scalar(
        dst=bt[0:1, 0:n_blocks],
        data=bt[0:1, 0:n_blocks],
        op0=nl.multiply,
        operand0=block_size,
        engine=nisa.engine.vector,
    )
    for b in range(n_blocks):
        rows = min(block_size, end_pos - b * block_size)
        nisa.dma_copy(
            dst=k_lin_hbm[b * block_size : b * block_size + rows, 0:head_width],
            src=flat_cache.ap(
                pattern=[[head_width, rows], [1, head_width]],
                offset=0,
                scalar_offset=bt.ap(pattern=[[n_blocks, 1], [1, 1]], offset=b),
            ),
        )
    sbm.close_scope()


def load_q_panel_transposed(
    sbm: SbufManager,
    q_hbm: nl.NkiTensor,
    qt_sb: nl.NkiTensor,
    cfg: GlmIndexerConfig,
    tile: int,
    rows: int,
    slot: int,
) -> None:
    """Load one query tile's Q as ``[head_dim, n_heads, queries]`` -- contraction on partitions.

    ``nc_matmul`` contracts over partitions, so the score's stationary operand must present
    ``head_dim`` there, while Q is stored query-major. The flip is done as ``n_heads / HEAD_CHUNK``
    three-dimensional ``dma_transpose``es rather than one per head: viewing the staged tile as
    ``[rows, HEAD_CHUNK, head_dim]`` and transposing with ``axes=(2, 1, 0)`` emits a whole chunk of
    heads per DMA. ``head_dim`` is 128, exactly ``dma_transpose``'s ``shape[-1]`` limit.

    Staging through SBUF (rather than transposing straight from HBM) is what makes the 3-D view
    possible, and the staging copy is one contiguous DMA of 8 KB/partition.
    """
    sbm.open_scope(name="q_panel_load")
    stage_sb = sbm.alloc_stack((P_MAX, cfg.q_out_dim), nl.bfloat16, name=f"q_stage{_uid()}", align=32)
    nisa.dma_copy(
        dst=stage_sb[:rows, : cfg.q_out_dim],
        src=q_hbm[tile * P_MAX : tile * P_MAX + rows, : cfg.q_out_dim],
    )
    for h0 in range(0, cfg.n_heads, HEAD_CHUNK):
        n_h = min(HEAD_CHUNK, cfg.n_heads - h0)
        chunk_view = stage_sb[:rows, h0 * cfg.head_dim : (h0 + n_h) * cfg.head_dim].reshape_dim(1, [n_h, cfg.head_dim])
        nisa.dma_transpose(
            dst=qt_sb[: cfg.head_dim, slot, h0 : h0 + n_h, :rows],
            src=chunk_view,
            axes=(2, 1, 0),
        )
    sbm.close_scope()


def _ts(dst, src, op, imm, engine):
    """One ``tensor_scalar`` with an immediate, on a named engine."""
    nisa.tensor_scalar(dst, src, op0=op, operand0=imm, engine=engine)


def build_key_positions(sbm, dst_hbm, ctx_row, s_kv_local, cp_degree, interleave):
    """Fill ``dst_hbm[0, g]`` with the global position of gathered key ``g``, once per kernel.

    ``owned(rank, l) = (l // I) * (I*W) + rank * I + (l % I)``, rank = g // s_kv_local,
    l = g % s_kv_local. I, W and s_kv_local are powers of two, so every step is a shift or a mask.
    Mirrors the interleaved masking in f8d0e4c9.

    Laid out across all 128 partitions (``g = i * cols + j``) so each op moves 128 * CW elements
    rather than one partition's row, and the rank / offset chains are independent so they run on
    different engines.
    """
    lg_kv, lg_i = int(math.log2(s_kv_local)), int(math.log2(interleave))
    lg_p = int(math.log2(interleave * cp_degree))
    cols = ctx_row // P_MAX
    kernel_assert(
        ctx_row % P_MAX == 0,
        f"key-position build needs a context that fills 128 partitions; got ctx_row={ctx_row}",
    )
    vec, sca = nisa.engine.vector, nisa.engine.scalar

    sbm.open_scope(name="key_positions")
    cw = min(cols, KPOS_COLS)
    g = sbm.alloc_stack((P_MAX, cw), nl.int32, name=f"kpg{_uid()}")
    r = sbm.alloc_stack((P_MAX, cw), nl.int32, name=f"kpr{_uid()}")
    h = sbm.alloc_stack((P_MAX, cw), nl.int32, name=f"kph{_uid()}")
    out = sbm.alloc_stack((P_MAX, cw), nl.float32, name=f"kpo{_uid()}")
    for c in range(0, cols, cw):
        w = min(cw, cols - c)
        # g = i * cols + j, so one DMA out preserves linear order.
        nisa.iota(dst=g[:P_MAX, :w], pattern=[[1, w]], offset=c, channel_multiplier=cols)
        _ts(r[:P_MAX, :w], g[:P_MAX, :w], nl.right_shift, lg_kv, sca)  # rank
        _ts(r[:P_MAX, :w], r[:P_MAX, :w], nl.left_shift, lg_i, sca)  # rank * I
        _ts(g[:P_MAX, :w], g[:P_MAX, :w], nl.bitwise_and, s_kv_local - 1, vec)  # l
        _ts(h[:P_MAX, :w], g[:P_MAX, :w], nl.right_shift, lg_i, vec)
        _ts(h[:P_MAX, :w], h[:P_MAX, :w], nl.left_shift, lg_p, vec)  # (l // I) * (I*W)
        _ts(g[:P_MAX, :w], g[:P_MAX, :w], nl.bitwise_and, interleave - 1, vec)  # l % I
        nisa.tensor_tensor(r[:P_MAX, :w], r[:P_MAX, :w], h[:P_MAX, :w], op=nl.add, engine=vec)
        nisa.tensor_tensor(r[:P_MAX, :w], r[:P_MAX, :w], g[:P_MAX, :w], op=nl.add, engine=vec)
        nisa.tensor_copy(dst=out[:P_MAX, :w], src=r[:P_MAX, :w])
        nisa.dma_copy(
            dst=dst_hbm.reshape((P_MAX, cols))[0:P_MAX, c : c + w],
            src=out[:P_MAX, :w],
        )
    sbm.close_scope()


def apply_causal_mask(
    sbm: SbufManager,
    acc_sb: nl.NkiTensor,
    kpos_sb: nl.NkiTensor,
    qpos_sb: nl.NkiTensor,
    slot: int,
    rows: int,
    blk: int,
) -> None:
    """Add ``MASK_NEG`` wherever position p is not visible to query q: ``k_pos[p] > q_pos[q]``.

    Applied once per (query tile, position block) AFTER the head sum, since visibility is positional
    and independent of h. ``kpos_sb`` holds the intra-block ramp j and ``qpos_sb`` the block-relative
    query position, so the comparison is the same one global positions would give.
    """
    sbm.open_scope(name="causal_mask")
    mask_sb = sbm.alloc_stack((P_MAX, POS_BLOCK), nl.float32, name=f"mask{_uid()}")
    # mask01[q, p] = (k_pos[p] - q_pos[q]) > 0
    nisa.tensor_scalar(
        dst=mask_sb[:rows, :blk],
        data=kpos_sb[:rows, :blk],
        op0=nl.subtract,
        operand0=qpos_sb[:rows, slot : slot + 1],
        op1=nl.greater,
        operand1=0.0,
    )
    nisa.scalar_tensor_tensor(
        dst=acc_sb[:rows, slot, :blk],
        data=mask_sb[:rows, :blk],
        op0=nl.multiply,
        operand0=MASK_NEG,
        op1=nl.add,
        operand1=acc_sb[:rows, slot, :blk],
    )
    sbm.close_scope()


def score_block_for_group(
    sbm: SbufManager,
    qt_sb: nl.NkiTensor,
    kt_sb: nl.NkiTensor,
    w_sb: nl.NkiTensor,
    acc_sb: nl.NkiTensor,
    cfg: GlmIndexerConfig,
    n_in_group: int,
    tile_rows: list,
    blk: int,
) -> None:
    """``acc[i][q, p] = sum_h relu(K_p . q_{q,h}) * w[q, h]`` for every tile in the panel, one block.

    Loop order is HEAD-outer, TILE-inner, and that ordering is the point.

    Every head's weighted accumulate targets the same accumulator, so with tile-outer the
    ``n_heads`` accumulates for a tile form ONE dependency chain on Vector -- a true data dependency
    that no amount of double buffering can overlap, which is why successive overlap tricks kept
    returning ~1%. Iterating tiles inside the head loop makes consecutive accumulates write DIFFERENT
    tiles' accumulators, so they are independent and pipeline. The alternative -- splitting each
    tile's accumulator into partials -- buys the same independence but costs
    ``POS_BLOCK * 4`` bytes/partition per extra partial, which at ACC_SPLIT=4 would cut the Q panel
    from 7 tiles to 2 and so triple K traffic. This costs nothing.

    Heads still cannot accumulate in PSUM: ReLU sits between the dot product and the weighted sum.
    ``interleave_degree=2`` rotates the PSUM tile and the ReLU buffer so head h+1's matmul (PE)
    overlaps head h's ReLU (Scalar); PSUM caps that at 2 in flight, since a ``POS_BLOCK``-wide bf16
    tile is 4 of the 8 banks.
    """
    sbm.open_scope(interleave_degree=2, name="score_heads")
    for h in range(cfg.n_heads):
        for i in range(n_in_group):
            rows = tile_rows[i]
            relu_sb = sbm.alloc_stack((P_MAX, POS_BLOCK), nl.bfloat16, name=f"relu{_uid()}")
            logits_psum = nl.ndarray((P_MAX, POS_BLOCK), dtype=nl.bfloat16, buffer=nl.psum)

            for col in range(0, blk, MATMUL_TILE):
                cols = min(MATMUL_TILE, blk - col)
                nisa.nc_matmul(
                    dst=logits_psum[:rows, col : col + cols],
                    stationary=qt_sb[: cfg.head_dim, i, h, :rows],
                    moving=kt_sb[: cfg.head_dim, col : col + cols],
                    accumulate=False,
                )
            nisa.activation(dst=relu_sb[:rows, :blk], op=nl.relu, data=logits_psum[:rows, :blk])

            if h == 0:
                # First head initializes this tile's accumulator, so no zeroing pass is needed.
                nisa.tensor_scalar(
                    dst=acc_sb[:rows, i, :blk],
                    data=relu_sb[:rows, :blk],
                    op0=nl.multiply,
                    operand0=w_sb[:rows, i, 0:1],
                )
            else:
                nisa.scalar_tensor_tensor(
                    dst=acc_sb[:rows, i, :blk],
                    data=relu_sb[:rows, :blk],
                    op0=nl.multiply,
                    operand0=w_sb[:rows, i, h : h + 1],
                    op1=nl.add,
                    operand1=acc_sb[:rows, i, :blk],
                )
            sbm.increment_section()
    sbm.close_scope()


def score_over_context(
    sbm: SbufManager,
    q_hbm: nl.NkiTensor,
    k_hbm: nl.NkiTensor,
    weights_hbm: nl.NkiTensor,
    qpos_hbm: nl.NkiTensor,
    score_hbm: nl.NkiTensor,
    topk_hbm: nl.NkiTensor,
    cfg: GlmIndexerConfig,
    end_pos: int,
    tile_lo: int,
    tile_hi: int,
    fuse_topk: bool = True,
    score_out: nl.NkiTensor = None,
    k_cache_hbm: nl.NkiTensor = None,
    block_table_hbm: nl.NkiTensor = None,
    block_size: int = 0,
    kv_cache_fp8_dtype=None,
) -> None:
    """Score this core's query tiles against ``[0, end_pos)``, K-outer over a resident Q panel.

    The top-k is issued here per group, as soon as that group's score rows are complete, so its
    GpSimd work overlaps the next group's score on PE/Scalar/Vector instead of adding to it.

    Loop order is (query-tile GROUP, position block, tile within group): each K block is read once
    and consumed by every tile in the panel, so K traffic falls by the panel size.

    K is S-agnostic -- the loop sees only a position range, so a position's origin (this rank, a peer,
    the paged cache) is invisible. Three independent raggednesses all have to be right: a partial
    group, a partial tile, and a partial position block.
    """
    panel = cfg.q_panel_tiles

    T_row = score_hbm.shape[-1]
    ctx_row = score_row_width(end_pos, cfg.topk_nseg)

    # Layout the mask converts from: rank-major, `interleave` tokens per rank per round.
    s_kv_local = max(1, end_pos // cfg.cp_degree)
    interleave = cfg.cp_interleave or s_kv_local
    kernel_assert(
        s_kv_local & (s_kv_local - 1) == 0 and interleave & (interleave - 1) == 0,
        f"the mask's slot->global map needs power-of-two s_kv_local ({s_kv_local}) and interleave ({interleave})",
    )

    kpos_hbm = nl.ndarray((ctx_row, 1), dtype=nl.float32, buffer=nl.private_hbm, name="kpos")
    build_key_positions(sbm, kpos_hbm, ctx_row, s_kv_local, cfg.cp_degree, interleave)

    # cp_degree == 1: the caller supplies a whole-context cache, so the table is resolved here over
    # the global extent. With a gather each rank linearizes its own shard instead (see the entry).
    if block_table_hbm is not None:
        k_lin = nl.ndarray((ctx_row, cfg.head_dim), dtype=nl.bfloat16, buffer=nl.private_hbm, name="k_lin")
        if cfg.kv_fp8:
            # fp8 cache: linearize the PACKED rows through the table, then dequant to dense bf16 --
            # exactly what the score's dma_transpose consumes. The dequant runs before any transpose.
            k_packed = nl.ndarray(
                (ctx_row, cfg.kv_cache_head_width), dtype=nl.bfloat16, buffer=nl.private_hbm, name="k_lin_packed"
            )
            linearize_pages(sbm, k_cache_hbm, block_table_hbm, k_packed, cfg, end_pos, block_size)
            dequant_k_packed(sbm, k_packed, k_lin, cfg, end_pos, kv_cache_fp8_dtype)
        else:
            linearize_pages(sbm, k_cache_hbm, block_table_hbm, k_lin, cfg, end_pos, block_size)
        k_hbm = k_lin

    kt_hbm = None
    if should_roll_score(cfg, end_pos) and end_pos // POS_BLOCK > 1:
        kt_hbm = nl.ndarray((cfg.head_dim, ctx_row), dtype=nl.bfloat16, buffer=nl.private_hbm, name="kt_hbm")
        sbm.open_scope(name="k_pretranspose")
        kt_stage = sbm.alloc_stack((P_MAX, POS_BLOCK), nl.bfloat16, name=f"kt_stage{_uid()}", align=32)
        for c in range(0, end_pos, POS_BLOCK):
            cw = min(POS_BLOCK, end_pos - c)
            nisa.dma_transpose(
                dst=kt_stage[: cfg.head_dim, :cw],
                src=k_hbm.ap(pattern=[[cfg.head_dim, cw], [1, cfg.head_dim]], offset=c * cfg.head_dim),
                axes=(1, 0),
            )
            nisa.dma_copy(dst=kt_hbm[0 : cfg.head_dim, c : c + cw], src=kt_stage[: cfg.head_dim, :cw])
        sbm.close_scope()

    cand_v_h = cand_p_h = None
    if cfg.score_chunk:
        cand_v_h, cand_p_h = alloc_chunk_candidates(
            cfg, div_ceil(end_pos, cfg.score_chunk), cfg.index_topk, panel * P_MAX
        )

    sbm.open_scope(name="score_invariants")
    qt_sb = sbm.alloc_stack((P_MAX, panel, cfg.n_heads, P_MAX), nl.bfloat16, name=f"q_panel{_uid()}", align=32)
    kt_sb = sbm.alloc_stack((P_MAX, POS_BLOCK), nl.bfloat16, name=f"k_block{_uid()}", align=32)
    w_sb = sbm.alloc_stack((P_MAX, panel, cfg.n_heads), nl.float32, name=f"weights{_uid()}")
    qpos_sb = sbm.alloc_stack((P_MAX, panel), nl.float32, name=f"qpos{_uid()}")
    # k_pos varies only with the position, so one precomputed row is replicated across partitions.
    kpos_sb = sbm.alloc_stack((P_MAX, POS_BLOCK), nl.float32, name=f"kpos{_uid()}")
    acc_sb = sbm.alloc_stack((P_MAX, panel, POS_BLOCK), nl.float32, name=f"score_acc{_uid()}")

    sbm.open_scope(interleave_degree=1, name="score_groups")
    for group_lo in range(tile_lo, tile_hi, panel):
        group_hi = min(group_lo + panel, tile_hi)
        n_in_group = group_hi - group_lo

        # Panel load: Q transposed, plus this tile's per-head weights.
        for i in range(n_in_group):
            tile = group_lo + i
            rows = min(P_MAX, cfg.s_local - tile * P_MAX)
            load_q_panel_transposed(sbm, q_hbm, qt_sb, cfg, tile, rows, i)
            nisa.dma_copy(
                dst=w_sb[:rows, i, : cfg.n_heads],
                src=weights_hbm[tile * P_MAX : tile * P_MAX + rows, : cfg.n_heads],
            )
            nisa.dma_copy(
                dst=qpos_sb[:rows, i : i + 1],
                src=qpos_hbm.reshape((cfg.s_local, 1))[tile * P_MAX : tile * P_MAX + rows, 0:1],
            )

        tile_rows = [min(P_MAX, cfg.s_local - (group_lo + i) * P_MAX) for i in range(n_in_group)]

        def _pos_block(pos, blk, chunk_base=0):
            """Score ONE position block. ``pos`` may be a trace-time int or a fori_loop index."""
            if kt_hbm is not None:
                nisa.dma_copy(
                    dst=kt_sb[: cfg.head_dim, :blk],
                    src=kt_hbm[0 : cfg.head_dim, nl.ds(pos, blk)],
                )
            else:
                nisa.dma_transpose(
                    dst=kt_sb[: cfg.head_dim, :blk],
                    src=k_hbm.ap(pattern=[[cfg.head_dim, blk], [1, cfg.head_dim]], offset=pos * cfg.head_dim),
                    axes=(1, 0),
                )

            nisa.dma_copy(
                dst=kpos_sb[:P_MAX, :blk],
                src=kpos_hbm.reshape((1, ctx_row))[0:1, nl.ds(pos, blk)].broadcast(dim=0, size=P_MAX),
            )
            score_block_for_group(sbm, qt_sb, kt_sb, w_sb, acc_sb, cfg, n_in_group, tile_rows, blk)
            for i in range(n_in_group):
                apply_causal_mask(sbm, acc_sb, kpos_sb, qpos_sb, i, tile_rows[i], blk)
            for i in range(n_in_group):
                tile = group_lo + i
                nisa.dma_copy(
                    dst=score_hbm[
                        (i if cfg.score_chunk else tile - tile_lo), 0 : tile_rows[i], nl.ds(pos - chunk_base, blk)
                    ],
                    src=acc_sb[: tile_rows[i], i, :blk],
                )
                if score_out is not None:
                    # Debug copy to the full-size returned score; see emit_score.
                    nisa.dma_copy(
                        dst=score_out[tile * P_MAX : tile * P_MAX + tile_rows[i], nl.ds(pos, blk)],
                        src=acc_sb[: tile_rows[i], i, :blk],
                    )

        if cfg.score_chunk:
            ch = cfg.score_chunk
            n_ch = div_ceil(end_pos, ch)
            for ci in range(n_ch):
                c0 = ci * ch
                cw = min(ch, end_pos - c0)
                # c0 is a default arg so the closure does not capture the loop variable.
                n_full_ch = cw // POS_BLOCK
                tail_ch = cw - n_full_ch * POS_BLOCK
                if should_roll_score(cfg, end_pos) and n_full_ch > 1:
                    nl.fori_loop(0, n_full_ch, lambda b, c0=c0: _pos_block(c0 + b * POS_BLOCK, POS_BLOCK, c0))
                else:
                    for b in range(n_full_ch):
                        _pos_block(c0 + b * POS_BLOCK, POS_BLOCK, c0)
                if tail_ch:
                    _pos_block(c0 + n_full_ch * POS_BLOCK, tail_ch, c0)
                if cw < ch:
                    # -inf the chunk tail so the snake ranks a full chunk without the pad winning.
                    pad_sb = sbm.alloc_stack((P_MAX, ch - cw), nl.float32, name=f"cpad{_uid()}")
                    nisa.memset(dst=pad_sb[:P_MAX, : ch - cw], value=float("-inf"))
                    for i in range(n_in_group):
                        nisa.dma_copy(dst=score_hbm[i, 0 : tile_rows[i], cw:ch], src=pad_sb[: tile_rows[i], : ch - cw])
                for i in range(n_in_group):
                    topk_chunk_stage1(
                        sbm, score_hbm, i, cand_v_h, cand_p_h, 0, tile_rows[i], ci, ch, cfg.index_topk, i * P_MAX
                    )
            for i in range(n_in_group):
                topk_chunk_merge(
                    sbm,
                    cand_v_h,
                    cand_p_h,
                    topk_hbm.reshape((cfg.s_local, cfg.index_topk)),
                    n_ch,
                    ch,
                    cfg.index_topk,
                    i * P_MAX,
                    tile_rows[i],
                    (group_lo + i) * P_MAX,
                )
            sbm.increment_section()
            continue

        # Unrolled, this loop dominates the trace and pos_blocks is its largest multiplier, so
        # rolling it is what stops the instruction count growing with context.
        n_full = end_pos // POS_BLOCK
        tail = end_pos - n_full * POS_BLOCK
        if should_roll_score(cfg, end_pos) and n_full > 1:
            nl.fori_loop(0, n_full, lambda b: _pos_block(b * POS_BLOCK, POS_BLOCK))
        else:
            for b in range(n_full):
                _pos_block(b * POS_BLOCK, POS_BLOCK)
        if tail:
            _pos_block(n_full * POS_BLOCK, tail)

        # Pad tail to -inf so the snake-padded columns never win the top-k. One memset per tile.
        pad = T_row - end_pos
        if pad > 0:
            pad_sb = sbm.alloc_stack((P_MAX, pad), nl.float32, name=f"score_pad{_uid()}")
            nisa.memset(dst=pad_sb[:P_MAX, :pad], value=float("-inf"))
            for i in range(n_in_group):
                tile = group_lo + i
                nisa.dma_copy(
                    dst=score_hbm[tile - tile_lo, 0 : tile_rows[i], end_pos : end_pos + pad],
                    src=pad_sb[: tile_rows[i], :pad],
                )
                if score_out is not None:
                    nisa.dma_copy(
                        dst=score_out[tile * P_MAX : tile * P_MAX + tile_rows[i], end_pos : end_pos + pad],
                        src=pad_sb[: tile_rows[i], :pad],
                    )

        # This group's score rows are complete, so its top-k can run now -- on GpSimd, overlapping
        # the next group's score on PE/Scalar/Vector. Skipped when the top-k does not fit alongside
        # the resident panel; the caller then runs it as its own phase (see topk_fits_with_panel).
        if fuse_topk:
            for i in range(n_in_group):
                topk_for_tile(
                    sbm, score_hbm, topk_hbm, cfg, end_pos, group_lo + i, group_lo + i - tile_lo, T_row, cfg.index_topk
                )
        sbm.increment_section()
    sbm.close_scope()  # score_groups
    sbm.close_scope()  # score_invariants


# --------------------------------------------------------------------------------------------
# Stage 6: top-k
# --------------------------------------------------------------------------------------------

TOPK_SNAKE = 16
"""Partitions one ``nisa.topk`` problem occupies. 8 such snakes fit a 128-partition tile, so one
call ranks 8 queries independently."""

TOPK_QUERIES_PER_CALL = P_MAX // TOPK_SNAKE  # 8

MASK_NEG = -3.0e38
SCORE_ROW_ALIGN = 2 * TOPK_SNAKE  # 32
TOPK_N_MAX = 65536


def derive_score_chunk(end_pos: int, topk_nseg: int = 0) -> int:
    """The score chunk width to use for ``end_pos``, or 0 to leave the score unchunked.

    Only chunk where the top-k is segmented anyway: below ``TOPK_N_MAX`` chunking would create a
    merge tree and position gather that a single ``nisa.topk`` does not need.

    Take the fewest legal chunks and then DIVIDE the row by that count, rather than using ``SEG_MAX``
    directly -- the latter gives the same chunk count with a nearly empty last chunk, whose tail pad
    is allocated at full chunk width in fp32 and overflows SBUF.
    """
    T = score_row_width(end_pos, topk_nseg)
    if T < TOPK_N_MAX:
        return 0
    n_chunks = div_ceil(T, SEG_MAX)
    return div_ceil(div_ceil(T, n_chunks), SCORE_ROW_ALIGN) * SCORE_ROW_ALIGN


def chunked_topk_sbuf_bytes(end_pos: int, k: int, forced: int = 0) -> int:
    """SBUF bytes per partition the SEGMENTED top-k holds live, from its scratch set.

    Mirrors ``_alloc_chunked_scratch``: the stage-1 snake source and its five ``[P_MAX, k]`` decode
    buffers, the candidate arrays at ``nseg * k/16`` columns, the merge outputs, and the one-hot
    gather's transposed blocks. Rounded up generously -- this only decides whether to FUSE, so
    over-estimating costs a little overlap and under-estimating costs a compile failure.
    """
    T, nseg, seg = topk_segments(end_pos, forced)
    del T
    src_x_seg = seg // TOPK_SNAKE
    cand_x = nseg * (k // TOPK_SNAKE)
    n_blk = div_ceil(cand_x, P_MAX)
    return (
        src_x_seg * 2  # s1_src bf16
        + k * (2 + 4 + 4 + 4 + 4)  # s1_val bf16 + s1_idx/pos/hi/lo 4 B each
        + cand_x * (2 + 4)  # cand_v bf16 + cand_p f32
        + k * (2 + 4)  # m_val bf16 + m_idx u32
        + n_blk * TOPK_SNAKE * 4  # oh_srct f32
        + 4096  # small per-partition constants and slack
    )


def topk_fits_with_panel(cfg: "GlmIndexerConfig", end_pos: int) -> bool:
    """Whether the top-k can be FUSED into the score loop, i.e. coexist with the resident Q panel.

    The single-segment top-k is small (one ``[P_MAX, k]`` set per batch) and always fits. The
    SEGMENTED one holds 65-75 KB/partition, which does not fit beside a full panel: at 128k the
    compile failed by 188 bytes on the merge's ``m_val`` (``Stack OOM: requested=4096 B,
    available=3908 B``).

    Shrinking the panel to make room is the worse trade. The panel divides K traffic by G, so
    halving G to 4 doubles it, while the overlap being bought is GpSimd top-k against PE/Vector
    score -- and at this config the score is 20x the top-k, so there is almost nothing to hide.
    Unfusing keeps the panel at full G and gives the top-k the whole budget.
    """
    if end_pos <= 0:
        return True
    _T, nseg, _seg = topk_segments(end_pos, cfg.topk_nseg)
    if nseg == 1:
        return True
    per_tile = cfg.n_heads * cfg.head_dim * 2 + POS_BLOCK * 4
    shared = POS_BLOCK * 2 + POS_BLOCK * 2
    panel_bytes = cfg.q_panel_tiles * per_tile + shared
    return panel_bytes + chunked_topk_sbuf_bytes(end_pos, cfg.index_topk, cfg.topk_nseg) <= SBUF_BUDGET_BYTES


_ROLL_MIN_BLOCKS = 4
"""Position blocks above which AUTO rolls the score loop. At or below this the trace is small
enough that unrolling costs nothing, and at 1 block fori_loop is a no-op."""


def should_roll_score(cfg: "GlmIndexerConfig", end_pos: int) -> bool:
    """Whether to roll the position-block loop for this context."""
    if cfg.roll_score == 1:
        return True
    if cfg.roll_score < 0:
        return False
    return end_pos // POS_BLOCK > _ROLL_MIN_BLOCKS


def score_row_width(end_pos: int, forced: int = 0) -> int:
    """Width the padded score row must be allocated at, for either top-k path.

    ``topk_geometry``'s ``T`` is only 16-aligned in the single-segment case (its
    ``round_up_to_chunks`` rounds to 16), and an odd ``T/16`` fails the backend ISA check -- see
    SCORE_ROW_ALIGN. So take the wider of the geometry's T and a 32-aligned round-up. In the
    SEGMENTED case T is ``nseg * seg`` with ``seg`` already 32-aligned, so the geometry wins and the
    row is exactly what the chunked path indexes.
    """
    if end_pos <= 0:
        return 1
    T_geo, _nseg, _seg = topk_segments(end_pos, forced)
    return max(T_geo, div_ceil(end_pos, SCORE_ROW_ALIGN) * SCORE_ROW_ALIGN)


def topk_segments(end_pos: int, forced: int = 0) -> tuple:
    """``(T, nseg, seg)`` for a top-k over ``end_pos`` positions.

    Delegates to ``topk_geometry``; see :func:`topk_for_tile` for the constraints its shape rules
    encode.
    """
    T, nseg, seg, _spc = topk_geometry(end_pos, forced)
    return T, nseg, seg


def topk_over_score(
    sbm: SbufManager,
    score_hbm: nl.NkiTensor,
    topk_hbm: nl.NkiTensor,
    cfg: GlmIndexerConfig,
    end_pos: int,
    tile_lo: int,
    tile_hi: int,
) -> None:
    """Select each query's top ``index_topk`` positions from its score row.

    ``nisa.topk`` ranks a 16-partition SNAKE: element ``i`` of a query's row lives at partition
    ``i % 16``, column ``i // 16``. The score, though, is query-MAJOR -- one query per partition,
    positions along the free axis -- so the two layouts disagree, and that disagreement is the whole
    difficulty. Reading the row back from HBM through an access pattern does the relayout for free,
    which is why this stage consumes ``score_hbm`` rather than the SBUF accumulator.

    The gather uses a CONTIGUOUS-CHUNK snake (partition ``16g + p`` of group ``g`` holds query
    ``8t+g``'s chunk ``p``) rather than a round-robin one: a stride-16 free axis fails DMA lowering,
    while a stride-1 axis lowers fine. topk finds the same winners by VALUE either way -- only the
    returned index needs decoding, ``pos = (i % 16) * src_x + (i // 16)``.

    This is deliberately the UNFUSED version: the score round-trips HBM. At 1M that round trip is
    ~512 MB per query tile, so stage 7 folds a running merge into the score loop and deletes it. The
    unfused form exists so the top-k can be validated on its own -- fusing first would make a
    top-k bug and a score bug indistinguishable.
    """
    T = score_hbm.shape[-1]
    src_x = T // TOPK_SNAKE
    kernel_assert(
        T % SCORE_ROW_ALIGN == 0,
        f"the score row ({T}) must be a multiple of {SCORE_ROW_ALIGN}: a 16-multiple is needed for the "
        f"snake gather to collapse to a contiguous run (else the DMA will not lower), and a 32-multiple "
        f"so src_x = row/16 is EVEN (else nisa.topk fails the backend ISA check, NCC_IXCG164)",
    )
    k = cfg.index_topk
    kernel_assert(k <= end_pos, f"index_topk={k} exceeds end_pos={end_pos}")

    sbm.open_scope(name="topk_phase")
    for tile in range(tile_lo, tile_hi):
        topk_for_tile(sbm, score_hbm, topk_hbm, cfg, end_pos, tile, tile - tile_lo, T, k)
    sbm.close_scope()


def _decode_snake_index(sbm, idx_sb, dst_sb, stride: int, base: int, rows: int, k: int) -> None:
    """``dst = (idx % 16) * stride + idx // 16 + base``.

    ``nisa.topk`` reports a SNAKE index: element ``i`` of its input sits at partition ``i % 16``,
    column ``i // 16``. Every consumer of that index -- mapping it to a position, or to a flat offset
    in the candidate array -- is this same affine decode with a different ``stride``/``base``.
    """
    sbm.open_scope(name="snake_decode")
    hi = sbm.alloc_stack((P_MAX, k), nl.uint32, name=f"sd_hi{_uid()}")
    lo = sbm.alloc_stack((P_MAX, k), nl.uint32, name=f"sd_lo{_uid()}")
    nisa.tensor_scalar(dst=hi[:rows, :k], data=idx_sb[:rows, :k], op0=nl.right_shift, operand0=4)
    nisa.tensor_scalar(dst=lo[:rows, :k], data=idx_sb[:rows, :k], op0=nl.bitwise_and, operand0=15)
    if base:
        nisa.tensor_scalar(dst=hi[:rows, :k], data=hi[:rows, :k], op0=nl.add, operand0=base)
    nisa.scalar_tensor_tensor(
        dst=dst_sb[:rows, :k],
        data=lo[:rows, :k],
        op0=nl.multiply,
        operand0=stride,
        op1=nl.add,
        operand1=hi[:rows, :k],
    )
    sbm.close_scope()


def topk_for_tile(
    sbm: SbufManager,
    score_hbm: nl.NkiTensor,
    topk_hbm: nl.NkiTensor,
    cfg: GlmIndexerConfig,
    end_pos: int,
    tile: int,
    slot: int,
    T: int,
    k: int,
) -> None:
    """Top-k for ONE query tile, segmenting the row when it exceeds ``nisa.topk``'s reach.

    Callable from inside the score loop so a tile's top-k (GpSimd) overlaps the next tile's score
    (PE/Scalar/Vector); as separate phases the two costs add.

    Segment-and-merge (``end_pos`` past ``nisa.topk``'s ``n < 65536``) is delegated to
    :func:`topk_over_score_chunked`, whose contract already matches this kernel's padded score tile.
    A call's 8 snakes must be spent on 8 POSITION SEGMENTS of one query, not 8 queries, or the
    stage-1 gather does not collapse to one descriptor; and merge winners must be resolved with a
    one-hot matmul, since ``local_gather`` cannot move data across partitions.
    """
    rows = min(P_MAX, cfg.s_local - tile * P_MAX)
    _T_geo, nseg, seg = topk_segments(end_pos, cfg.topk_nseg)
    if nseg == 1:
        # Rank the WHOLE PADDED ROW, not the geometry's seg: the snake gather only collapses to one
        # descriptor when 16*seg_x equals the row stride. Pad columns are -inf, so they never win.
        seg = T
    seg_x = seg // TOPK_SNAKE
    k_cols = k // TOPK_SNAKE

    kernel_assert(seg < TOPK_N_MAX, f"segment width {seg} must stay under {TOPK_N_MAX}")
    kernel_assert(k % TOPK_SNAKE == 0, f"index_topk={k} must be a multiple of {TOPK_SNAKE}")

    if nseg > 1:
        kernel_assert(
            _T_geo == T,
            f"the score row must be padded to the chunked geometry's T; row is {T}, geometry wants "
            f"{_T_geo}. Allocate the score buffer from topk_segments(end_pos, topk_nseg)[0].",
        )
        topk_over_score_chunked(
            sbm,
            P_MAX,
            end_pos,
            k,
            score_hbm,
            tile,
            topk_hbm.reshape((cfg.s_local, k)),
            sc_slot=slot,
            s_tile_base_q=tile * P_MAX,
            topk_nseg=cfg.topk_nseg,
        )
        return

    for batch in range(div_ceil(rows, TOPK_QUERIES_PER_CALL)):
        q_base = batch * TOPK_QUERIES_PER_CALL
        n_q = min(TOPK_QUERIES_PER_CALL, rows - q_base)

        sbm.open_scope(name="topk_batch")
        src_sb = sbm.alloc_stack((P_MAX, seg_x), nl.bfloat16, name=f"tk_src{_uid()}")
        val_sb = sbm.alloc_stack((P_MAX, k), nl.bfloat16, name=f"tk_val{_uid()}")
        idx_sb = sbm.alloc_stack((P_MAX, k), nl.uint32, name=f"tk_idx{_uid()}")
        pos_sb = sbm.alloc_stack((P_MAX, k), nl.int32, name=f"tk_pos{_uid()}")

        # 8 queries as 8 snakes, fp32 -> bf16 on the way in; the AP collapses to one contiguous run
        # because the row is padded to a snake multiple.
        #
        # Addressed by SLOT, not global tile: the ring is per-core and the score phase writes it at
        # score_hbm[tile - tile_lo], so any core whose tile_lo is non-zero writes slot 0.
        nisa.dma_copy(
            dst=src_sb[:P_MAX, :seg_x],
            src=score_hbm.ap(
                pattern=[[T, TOPK_QUERIES_PER_CALL], [seg_x, TOPK_SNAKE], [1, seg_x]],
                offset=(slot * P_MAX + q_base) * T,
            ),
            dge_mode=dge_mode.swdge,
        )
        nisa.topk(val_dst=val_sb[:P_MAX, :k], idx_dst=idx_sb[:P_MAX, :k], src=src_sb[:P_MAX, :seg_x], n=seg)
        # Segment-local snake index -> absolute position.
        _decode_snake_index(sbm, idx_sb, pos_sb, seg_x, 0, P_MAX, k)

        # These ARE the answer, so write them out directly rather than through candidate staging.
        rows_out = n_q * TOPK_SNAKE
        nisa.dma_copy(
            dst=topk_hbm.reshape((cfg.s_local * TOPK_SNAKE, k_cols))[
                (tile * P_MAX + q_base) * TOPK_SNAKE : (tile * P_MAX + q_base) * TOPK_SNAKE + rows_out,
                :k_cols,
            ],
            src=pos_sb[:rows_out, :k_cols],
            dge_mode=dge_mode.swdge,
        )
        sbm.close_scope()


# --------------------------------------------------------------------------------------------
# Kernel entry -- stages 1-6

# --------------------------------------------------------------------------------------------


@nki.jit
def glm_indexer_proj(
    x: nl.NkiTensor,
    qr: nl.NkiTensor,
    wq: nl.NkiTensor,
    wk: nl.NkiTensor,
    k_norm_gamma: nl.NkiTensor,
    k_norm_beta: nl.NkiTensor,
    cos: nl.NkiTensor,
    sin: nl.NkiTensor,
    slot_mapping: nl.NkiTensor,
    weights: nl.NkiTensor,
    q_pos: nl.NkiTensor,
    end_pos: int = 0,
    cp_degree: int = 1,
    replica_groups: tuple = None,
    topk_nseg: int = 0,
    roll_score: int = 0,
    emit_score: bool = False,
    emit_proj: bool = False,
    k_ctx: nl.NkiTensor = None,
    k_cache: nl.NkiTensor = None,
    zero_topk: bool = False,
    score_chunk: int = 0,
    block_table: nl.NkiTensor = None,
    block_size: int = 0,
    cp_interleave: int = 0,
    index_topk: int = 2048,
    kv_fp8: bool = False,
    kv_cache_fp8_dtype=nl.float8_e4m3,
):
    """The GLM indexer: Q/K projections, paged cache write, CP gather, score, top-k.

    Args:
        x: ``[s_local, dim]`` bf16 hidden states.
        qr: ``[s_local, q_lora_rank]`` bf16 latent Q.
        wq: ``[q_lora_rank, n_heads * head_dim]`` bf16 Q weight.
        wk: ``[dim, head_dim]`` bf16 K weight.
        k_norm_gamma: ``[head_dim]`` fp32 LayerNorm scale for K.
        k_norm_beta: ``[head_dim]`` fp32 LayerNorm bias for K.
        cos: ``[s_local, rope_head_dim // 2]`` fp32 RoPE cosine table.
        sin: ``[s_local, rope_head_dim // 2]`` fp32 RoPE sine table.
        slot_mapping: ``[s_local]`` int32 flat cache slot per token, ``-1`` for padding.
        weights: ``[s_local, n_heads]`` fp32 per-head score weights.
        q_pos: ``[s_local]`` fp32 global position of each local query; all-gathered to give the
            gathered K's positions.
        end_pos: valid cache positions to score against, i.e. the filled prefix. 0 skips the
            score phase entirely ([Testing]).
        cp_degree: CP ranks, so ``s_global = cp_degree * s_local``. 1 skips the collective.
        replica_groups: CP rank-lists for the in-kernel collectives, from the framework. None assumes
            a single group ``[0, cp_degree)``, which only holds for pure CP.
        topk_nseg: [Testing] force the top-k segment count; production derives it (0).
        roll_score: [Testing] force the position-block loop rolled (1) or unrolled (-1); production
            derives it (0).
        emit_score: [Testing] also return the full [s_local, T] score. O(s_local * s_global), so it
            does not fit in HBM at long context.
        k_ctx: [Testing] ``[s_global, head_dim]`` bf16 context supplied by the caller instead of
            gathered from peers, so one rank can stand in for a whole CP group.
        zero_topk: [Testing] return zeros for the top-k output instead of the picks.
        block_table: ``[n_blocks]`` int32 page ids, logical block i -> cache block block_table[i].
            The score then reads K through this table rather than from the gathered K, which is what
            lets a cached prefix participate. Requires block_size.
        block_size: tokens per cache page. Must divide POS_BLOCK.
        cp_interleave: CP interleave granularity of the cache in tokens; 0 means contiguous.
        score_chunk: [Testing] force the score chunk width, or -1 to disable chunking; production
            derives it (0). Chunking is what bounds the score ring.
        index_topk: positions selected per query.
        emit_proj: [Testing] also return the projected Q and the gathered K.
        k_cache: paged K cache, owned by the caller and written in place. Its flattened leading dims
            are slots; the minor dim is ``head_dim`` (bf16) or, with ``kv_fp8``, the fused packed row
            width ``kv_row_bf16_width(head_dim)`` bf16 (fp8 K bitcast 2-per-bf16 + per-token scale).
            The block table addresses it on the read side.
        kv_fp8: store the paged K cache in fp8, quantized per token over ``head_dim`` (RoPE
            included, matching vLLM) and packed into a bf16-container row alongside the per-token
            dequant scale. Halves cache and CP-collective bandwidth. The packed rows are unpacked to
            dense bf16 (``dequant_k_packed``) before the score: after the gather on the CP path, or
            after ``linearize_pages`` on the block-table / paged-cache (APC) path. False keeps the
            bf16 cache path unchanged.
        kv_cache_fp8_dtype: fp8 format for the K latent when ``kv_fp8`` -- ``nl.float8_e4m3``
            (legacy, max 240, default) or the OCP ``nl.float8_e4m3fn`` (max 448, NeuronCore-v4).

    Returns:
        ``topk`` -- ``[s_local, 16, index_topk // 16]`` int32 positions in SNAKE layout.
        With ``emit_proj`` / ``emit_score`` the projected Q, the gathered K and the score
        follow it, in that order.
        ``[s_local, end_pos]`` fp32 index scores (a 1-column dummy when ``end_pos`` is 0), and
        ``[s_local, 16, index_topk // 16]`` int32 top-k positions in SNAKE layout.
    """
    s_local, dim = x.shape
    head_dim = wk.shape[1]
    n_heads = wq.shape[1] // head_dim
    q_lora_rank = wq.shape[0]
    rope_head_dim = 2 * cos.shape[1]
    s_global = k_ctx.shape[0] if k_ctx is not None else cp_degree * s_local
    cfg = GlmIndexerConfig(
        s_local=s_local,
        s_global=s_global,
        n_heads=n_heads,
        head_dim=head_dim,
        dim=dim,
        q_lora_rank=q_lora_rank,
        rope_head_dim=rope_head_dim,
        index_topk=index_topk,
        score_chunk=(derive_score_chunk(end_pos or s_global, topk_nseg) if score_chunk == 0 else max(score_chunk, 0)),
        topk_nseg=topk_nseg,
        cp_interleave=cp_interleave,
        roll_score=roll_score,
        kv_fp8=kv_fp8,
    )
    cfg.validate()
    _, num_cores, core_id = get_verified_program_sharding_info("glm_indexer", max_sharding=2)

    kernel_assert(k_cache is not None, "k_cache is a caller-owned input; pass the paged K cache")
    # fp8 + block_table (paged cache / APC): linearize_pages moves the packed fp8 rows through the
    # table and dequant_k_packed unpacks them to dense bf16 before the score -- the cp>1 shard path
    # dequants after the gather, the cp==1 path inside score_over_context. Both are packed-row aware.
    q_out = nl.ndarray((cfg.s_local, cfg.q_out_dim), dtype=nl.bfloat16, buffer=nl.shared_hbm)
    collective_hbm = nl.private_hbm if num_cores == 1 else nl.shared_hbm
    # With kv_fp8 the collective moves the fused packed row (fp8 K + in-row scale), so k_local/k_all
    # are the packed-row width; dequant happens locally after the gather (see dequant_k_packed).
    kv_width = cfg.kv_cache_head_width
    k_local = nl.ndarray((cfg.s_local, kv_width), dtype=nl.bfloat16, buffer=collective_hbm, name="k_local")

    # This rank's share of the valid context: with a resident prefix (APC) the block table covers
    # more than this step's s_local tokens, so the shard and the gather must span all of it.
    s_kv_local = max(cfg.s_local, end_pos // cp_degree) if end_pos else cfg.s_local
    ctx = k_ctx.shape[0] if k_ctx is not None else (cp_degree * s_kv_local)
    k_all = nl.ndarray((ctx, kv_width), dtype=nl.bfloat16, buffer=collective_hbm, name="k_all")
    # Dense bf16 K the score consumes. bf16 path: k_all itself. fp8 path: the dequantized cache.
    k_gathered = nl.ndarray((ctx, cfg.head_dim), dtype=nl.bfloat16, buffer=nl.shared_hbm)
    qpos_local = nl.ndarray((cfg.s_local, 1), dtype=nl.float32, buffer=collective_hbm, name="qpos_local")

    score_width = score_row_width(end_pos, topk_nseg)
    _ring_slots = cfg.q_panel_tiles if cfg.score_chunk else score_ring_slots(div_ceil(cfg.num_q_tiles, num_cores))
    _ring_width = cfg.score_chunk if cfg.score_chunk else score_width
    score = nl.ndarray(
        (_ring_slots, P_MAX, _ring_width),
        dtype=nl.bfloat16,
        buffer=nl.private_hbm,
        name="score_ring",
    )

    score_out = (
        nl.ndarray((cfg.s_local, score_width), dtype=nl.float32, buffer=nl.shared_hbm)
        if emit_score
        else nl.ndarray((1, 1), dtype=nl.float32, buffer=nl.shared_hbm)
    )
    topk = nl.ndarray(
        (cfg.s_local, TOPK_SNAKE, max(cfg.index_topk // TOPK_SNAKE, 1)), dtype=nl.int32, buffer=nl.shared_hbm
    )
    """
    Top-k picks -- the kernel's actual product.

    ``zero_topk`` leaves it all-zero, which exists only to work around the collective test
    framework. That framework derives output arity from the GOLDEN dict's key count
    (``output_names = list(golden.keys())``) and offers no way to skip one output's comparison, so a
    reference cannot simply omit output_4 -- it fails with "output_names has 4 entries but kernel
    returns 5 outputs". And it cannot meaningfully COMPARE output_4 either: nisa.topk runs
    sorted=False, and near the cut a 0.1% score difference legitimately reorders winners, so the
    reference's argpartition disagrees on 8190/8192 rows while every pick is at or above the
    kernel's own cut.

    So at cp > 1 the two claims are split: this flag zeros the picks so the collective suite can
    compare the score, gather and mask cleanly, and the top-k is validated where it can be done
    soundly -- against the kernel's OWN score, by the single-rank suites (including the long-context
    k_ctx cases, which reach the same end_pos on one rank).
    """

    sbm = SbufManager(
        sb_lower_bound=0,
        sb_upper_bound=nl.tile_size.total_available_sbuf_size,
        logger=get_logger("glm_indexer"),
    )
    sbm.open_scope(name="indexer_root")

    # Weights are query-independent: hoist ONCE, outside the tile loop. They must also live in a
    # scope the tile loop does not increment_section() over, or the loop will hand their addresses
    # out again to the second tile.
    sbm.open_scope(name="weight_hoist")
    wq_sb = load_weight_k_major(sbm, wq, cfg.q_lora_rank, cfg.q_out_dim)
    wk_sb = load_weight_k_major(sbm, wk, cfg.dim, cfg.head_dim)
    gamma_sb = broadcast_norm_weight(sbm, k_norm_gamma, cfg.head_dim, "k_gamma")
    beta_sb = broadcast_norm_weight(sbm, k_norm_beta, cfg.head_dim, "k_beta")

    # LNC sharding: split the QUERY TILES across cores, not the heads.
    tiles_per_core = div_ceil(cfg.num_q_tiles, num_cores)
    tile_lo = core_id * tiles_per_core
    tile_hi = min(tile_lo + tiles_per_core, cfg.num_q_tiles)

    # Double buffered across query tiles.
    sbm.open_scope(interleave_degree=2, name="tile_loop")
    for tile in range(tile_lo, tile_hi):
        row_start = tile * P_MAX
        rows = min(P_MAX, cfg.s_local - row_start)

        q_sb = sbm.alloc_stack((P_MAX, cfg.q_out_dim), nl.bfloat16, name=f"q_tile{_uid()}")
        k_sb = sbm.alloc_stack((P_MAX, cfg.head_dim), nl.float32, name=f"k_tile{_uid()}")

        k_post_sb = sbm.alloc_stack((P_MAX, cfg.head_dim), nl.float32, name=f"k_post{_uid()}")

        q_rope = load_rope_tables(sbm, cos, sin, cfg, row_start, rows)
        project_q(sbm, qr, wq_sb, cfg, row_start, rows, q_sb, rope=q_rope)
        project_k(sbm, x, wk_sb, cfg, row_start, rows, k_sb)
        postprocess_k(sbm, k_sb, gamma_sb, beta_sb, cos, sin, cfg, row_start, rows, k_post_sb)

        nisa.dma_copy(dst=q_out[row_start : row_start + rows, : cfg.q_out_dim], src=q_sb[:rows, : cfg.q_out_dim])
        # fp8: store_k_paged writes the packed row straight into k_local (scale in-row), so the
        # gather moves fp8. bf16: copy the dense K as before.
        store_k_paged(
            sbm,
            k_post_sb,
            k_cache,
            slot_mapping,
            cfg,
            row_start,
            rows,
            fp8_dtype=kv_cache_fp8_dtype,
            local_out=k_local if cfg.kv_fp8 else None,
        )
        if not cfg.kv_fp8:
            nisa.dma_copy(
                dst=k_local[row_start : row_start + rows, : cfg.head_dim], src=k_post_sb[:rows, : cfg.head_dim]
            )
        sbm.increment_section()
    sbm.close_scope()
    sbm.close_scope()

    nisa.dma_copy(dst=qpos_local[: cfg.s_local, 0:1], src=q_pos.reshape((cfg.s_local, 1)))
    if k_ctx is not None:
        # k_ctx is caller-supplied in the cache's container dtype: dense bf16 (bf16 path) or the
        # packed fp8 row (fp8 path). Either way it is kv_width wide and copied verbatim.
        nisa.dma_copy(dst=k_all[0:ctx, :kv_width], src=k_ctx[0:ctx, :kv_width])
    else:
        # This rank's own pages made contiguous, then gathered. The gathered cache stays interleaved
        # in global position; the causal mask is what resolves it. Skipping the gather at cp_degree=1
        # is the ONLY difference between the two topologies.
        src_local = k_local
        if block_table is not None and cp_degree > 1:
            # Shard is the cache's container dtype -- head_dim bf16, or the packed fp8 row for
            # kv_fp8. linearize_pages moves whole rows, so the width is all that changes; the gather
            # then moves packed fp8 and the existing post-gather dequant_k_packed unpacks it.
            shard = nl.ndarray((s_kv_local, kv_width), dtype=nl.bfloat16, buffer=nl.private_hbm, name="k_shard")
            linearize_pages(sbm, k_cache, block_table, shard, cfg, s_kv_local, block_size)
            src_local = shard
        if cp_degree > 1:
            # With kv_fp8 the source rows are the packed fp8 rows, so the collective moves ~half the
            # bytes; dequant is deferred to after the gather.
            gather_k_across_cp(src_local, k_all, cp_degree, replica_groups)
        else:
            rows_local = min(s_kv_local, src_local.shape[0])
            nisa.dma_copy(dst=k_all[:rows_local, :kv_width], src=src_local[:rows_local, :kv_width])

    # Dense bf16 K the score path consumes. bf16: k_all already IS dense K. fp8: dequant the packed
    # rows here -- AFTER the collective, so the gather moved fp8, and the score sees plain bf16 K.
    if cfg.kv_fp8:
        dequant_k_packed(sbm, k_all, k_gathered, cfg, ctx, kv_cache_fp8_dtype)
        k_score = k_gathered
    else:
        nisa.dma_copy(dst=k_gathered, src=k_all)
        k_score = k_all

    if end_pos > 0:
        """
        The score is q(s_local) @ K(s_global): every one of THIS rank's queries scores against the
        whole CP context, which is why K comes from the gathered buffer rather than k_local.

        LIMITATION, stated because it is silent otherwise: k_gathered holds only THIS step's K. That
        is correct when the whole context arrives in one shot (s_global == cp_degree * s_local), but
        under chunked prefill the earlier chunks live only in the paged cache, so scoring against
        k_gathered alone would miss them. Reading the cache via the block table is what fixes that:
        pass ``block_table`` and the block-table path linearizes the resident prefix (bf16 or the
        packed fp8 rows, then dequantized) into the scored context.
        """
        kernel_assert(
            ctx == cfg.s_global,
            f"s_global ({cfg.s_global}) must equal the scored context ({ctx}); the gathered K is "
            f"sized from the latter, so a mismatch scores against the wrong context length",
        )
        kernel_assert(
            end_pos <= ctx,
            f"end_pos ({end_pos}) exceeds the gathered K's {ctx} positions",
        )
        fuse = topk_fits_with_panel(cfg, end_pos)
        score_over_context(
            sbm,
            q_out,
            k_score,
            weights,
            q_pos,
            score,
            topk,
            cfg,
            end_pos,
            tile_lo,
            tile_hi,
            fuse_topk=fuse,
            score_out=score_out if emit_score else None,
            k_cache_hbm=k_ctx if k_ctx is not None else k_cache,
            block_table_hbm=block_table if cp_degree <= 1 else None,
            block_size=block_size,
            kv_cache_fp8_dtype=kv_cache_fp8_dtype,
        )
        if not fuse:
            topk_over_score(sbm, score, topk, cfg, end_pos, tile_lo, tile_hi)
    sbm.close_scope()

    if zero_topk:
        sbm.open_scope(name="zero_topk")
        z = sbm.alloc_stack((P_MAX, max(cfg.index_topk // TOPK_SNAKE, 1)), nl.int32, name=f"ztk{_uid()}")
        nisa.memset(dst=z, value=0)
        flat = topk.reshape((cfg.s_local * TOPK_SNAKE, max(cfg.index_topk // TOPK_SNAKE, 1)))
        for base in range(0, cfg.s_local * TOPK_SNAKE, P_MAX):
            rows = min(P_MAX, cfg.s_local * TOPK_SNAKE - base)
            nisa.dma_copy(dst=flat[base : base + rows, :], src=z[:rows, :])
        sbm.close_scope()

    outs = [topk]
    if emit_proj:
        outs += [q_out, k_gathered]
    if emit_score:
        outs.append(score_out)
    return tuple(outs)
