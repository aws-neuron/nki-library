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

"""BF16 Q and K projections for the Sparse Attention Indexer (``MlaPrecision.BF16``).

BF16 siblings of ``q_projection_mx`` / ``load_wk_mx_weights`` / the K half of
``k_and_weights_projection_mx`` in :mod:`sparse_attention_indexer_mx_helpers`. Everything
those need to reach ``nc_matmul_mx`` — 4-packing the weights, the block-128 -> block-32 scale
broadcast, and ``quantize_mx`` on the activation — exists only to feed the MX matmul. In bf16
the weights are plain ``[K, N]`` and the matmuls are ``nisa.nc_matmul``.

Note the W projection needs no bf16 sibling: ``w_projection_mx_batch`` is already bf16
end-to-end (bf16 weights, bf16 activation, plain ``nc_matmul``) despite its name — only Q and
K were genuinely MX.

Common SBUF convention, matching the MX helpers' ``[P_MAX, num_K_tiles, *]`` shape with the
K tile shrinking from 512 (128 partitions x 4-pack) to 128 (no pack):

    weights                : ``[P_MAX, ceil(K/128), N]`` bf16 -- tile ``t`` holds rows
                             ``[t*128, (t+1)*128)``.
    activations (transposed): ``[P_MAX, ceil(K/128), S]`` bf16 -- same tiling, free axis is
                             the query axis.

The transpose is unavoidable in both precisions: ``nc_matmul`` contracts along the PARTITION
axis of both operands, and both activations here (``qr`` and ``x``) arrive query-major with the
contraction on their free axis.
"""

import nki.isa as nisa
import nki.language as nl
from nki.isa.constants import dge_mode

from ...core.utils.allocator import SbufManager
from ...core.utils.kernel_helpers import div_ceil
from .sparse_attention_indexer_utils import P_MAX

# Contraction rows per bf16 K tile == the tensor-engine partition cap (MX packs 4x this).
BF16_K_TILE = 128

# Max moving free dim / one fp32 PSUM bank.
PSUM_FMAX = 512

# PSUM tiles cycled to pipeline the activation transposes (transpose tile b+1 while b drains).
_NUM_TP_TILES = 4


def _transpose_to_k_major_bf16(
    sbm: SbufManager,
    src_sb: nl.NkiTensor,
    out_sb: nl.NkiTensor,
    m_sz: int,
    m_pad: int,
    k_dim: int,
    src_row_stride: int = None,
) -> None:
    """Transpose ``[m_sz, k_dim]`` (query-major) into ``[P_MAX, ceil(k_dim/128), m_pad]`` (K-major).

    One PE transpose per 128-column K tile, cycling ``_NUM_TP_TILES`` PSUM tiles and alternating
    the read-out engine so the copies overlap the next transpose. Only the ``m_sz`` real columns
    are written; pad columns are left as-is, which is safe because ``dst[m, n]`` of the
    downstream matmul depends solely on column ``m`` of the stationary operand, so garbage in
    pad columns can only reach pad OUTPUT rows (which every caller discards).

    PSUM tiles are allocated WITHOUT an explicit address so the compiler places them and tracks
    liveness against the projection accumulators — this kernel leaves all PSUM to the compiler,
    unlike the MLA kernels which hand-place banks. (Hence this local helper rather than reusing
    ``mla_bf16_cte._transpose_bf16_to_k_major``, which hand-places.)

    Args:
        sbm (SbufManager): SBUF stack allocator (used only for the scope).
        src_sb (nl.NkiTensor): SBUF tile holding ``[m_sz, k_dim]``.
        out_sb (nl.NkiTensor): destination ``[P_MAX, ceil(k_dim/128), m_pad]`` bf16.
        m_sz (int): real query count in this tile.
        m_pad (int): allocated free width of ``out_sb``'s inner axis.
        k_dim (int): contraction dimension (a multiple of 128).
        src_row_stride (int): per-partition free stride of ``src_sb`` in elements; defaults to
            ``k_dim`` (a tightly packed ``[m, k_dim]`` buffer).
    """
    num_k_tiles = div_ceil(k_dim, BF16_K_TILE)
    if src_row_stride is None:
        src_row_stride = k_dim

    sbm.open_scope(name="tp_k_major")
    tp_psum = []
    for _ in range(_NUM_TP_TILES):
        tp_psum.append(nl.ndarray((P_MAX, m_pad), dtype=nl.bfloat16, buffer=nl.psum))

    out_part_stride = num_k_tiles * m_pad
    for k_tile_idx in range(num_k_tiles):
        k_rows = min(BF16_K_TILE, k_dim - k_tile_idx * BF16_K_TILE)
        bank = tp_psum[k_tile_idx % _NUM_TP_TILES]
        nisa.nc_transpose(
            data=src_sb.ap(
                pattern=[[src_row_stride, m_sz], [1, k_rows]],
                offset=k_tile_idx * BF16_K_TILE,
            ),
            dst=bank[0:k_rows, 0:m_sz],
        )
        nisa.tensor_copy(
            dst=out_sb.ap(pattern=[[out_part_stride, k_rows], [1, m_sz]], offset=k_tile_idx * m_pad),
            src=bank[0:k_rows, 0:m_sz],
            engine=nisa.engine.scalar if k_tile_idx % 2 == 0 else nisa.engine.vector,
        )
    sbm.close_scope()


def load_weights_bf16(
    sbm: SbufManager,
    w_hbm: nl.NkiTensor,
    k_dim: int,
    out_dim: int,
    full_out_dim: int = None,
    out_col_offset: int = 0,
) -> nl.NkiTensor:
    """Load a bf16 ``[k_dim, full_out_dim]`` weight slice K-major into SBUF.

    Returns ``[P_MAX, ceil(k_dim/128), out_dim]`` bf16. The source is perfectly rectangular
    (``k_dim`` is required to be a multiple of 128 for both projections here), so this is ONE
    3-D DMA — partition = row-within-tile, middle = tile, inner = column — which amortizes
    descriptor overhead and lets the HW DGE coalesce, matching the MX loaders' single-DMA
    weight preload.
    """
    if full_out_dim is None:
        full_out_dim = out_dim
    num_k_tiles = div_ceil(k_dim, BF16_K_TILE)
    w_sb = sbm.alloc_stack((P_MAX, num_k_tiles, out_dim), nl.bfloat16, buffer=nl.sbuf)
    nisa.dma_copy(
        dst=w_sb[0:BF16_K_TILE, 0:num_k_tiles, 0:out_dim],
        src=w_hbm.ap(
            pattern=[[full_out_dim, BF16_K_TILE], [BF16_K_TILE * full_out_dim, num_k_tiles], [1, out_dim]],
            offset=out_col_offset,
        ),
        dge_mode=dge_mode.hwdge,
    )
    return w_sb


def q_projection_bf16(
    sbm: SbufManager,
    qr_hbm: nl.NkiTensor,
    batch_start: int,
    S: int,
    wq_b: nl.NkiTensor,
    q_lora_rank: int,
    n_heads: int,
    head_dim: int,
    num_shards: int = 1,
    shard_id: int = 0,
    q_out_hbm: nl.NkiTensor = None,
) -> nl.NkiTensor:
    """BF16 Q projection: ``qr @ wq_b -> [S, n_heads*head_dim]``, written to ``q_out_hbm``.

    BF16 sibling of ``q_projection_mx``, with the same output-tile LNC sharding (each shard owns
    a contiguous range of 512-wide output tiles, keeping M=128 full PE utilisation) and the same
    single-DMA weight preload.

    The activation contract differs: MX consumes ``qr_qtz_hbm`` / ``qr_scale_hbm``, the
    pre-transposed + MX-quantized qr the upstream MLA QKV kernel exports so the indexer can skip
    that work. The bf16 QKV kernel exports plain bf16 ``qr[B*S, q_lora_rank]`` instead, so this
    does its own transpose — 16 PE transposes per s-tile at GLM's q_lora_rank=2048, cheap next
    to the projection itself.

    Args:
        sbm (SbufManager): SBUF stack allocator used for scratch buffers.
        qr_hbm (nl.NkiTensor): ``[B*S, q_lora_rank]`` bf16 — q-normed compressed query from the
            upstream QKV kernel (already gamma-normed; do NOT pass raw qr).
        batch_start (int): row offset into ``qr_hbm`` for this batch.
        S (int): number of query rows for this batch.
        wq_b (nl.NkiTensor): ``[q_lora_rank, n_heads*head_dim]`` bf16 weight.
        q_lora_rank (int): contraction dimension.
        n_heads (int): number of indexer heads.
        head_dim (int): per-head dimension; ``total_out = n_heads * head_dim``.
        num_shards (int): number of LNC shards splitting the output-tile range.
        shard_id (int): this shard's index.
        q_out_hbm (nl.NkiTensor): ``[S, total_out]`` bf16 output; this shard's columns written.

    Returns:
        nl.NkiTensor: ``q_out_hbm``.
    """
    total_out = n_heads * head_dim
    num_k_tiles = div_ceil(q_lora_rank, BF16_K_TILE)

    num_out_tiles_all = div_ceil(total_out, PSUM_FMAX)
    eff_num_shards = min(num_shards, num_out_tiles_all)
    if shard_id >= eff_num_shards:
        return q_out_hbm
    tiles_per_shard = div_ceil(num_out_tiles_all, eff_num_shards)
    my_tile_start = shard_id * tiles_per_shard
    my_tile_end = min(my_tile_start + tiles_per_shard, num_out_tiles_all)
    num_out_tiles = my_tile_end - my_tile_start
    num_s_tiles = div_ceil(S, P_MAX)
    out_col_start = my_tile_start * PSUM_FMAX
    out_col_end = min(my_tile_end * PSUM_FMAX, total_out)
    out_extent = out_col_end - out_col_start

    sbm.open_scope(name="q_proj_bf16")
    q_row_sb = sbm.alloc_stack((P_MAX, out_extent), nl.bfloat16)
    wq_b_sb = load_weights_bf16(
        sbm, wq_b, q_lora_rank, out_extent, full_out_dim=total_out, out_col_offset=out_col_start
    )

    for s_tile_idx in nl.sequential_range(num_s_tiles):
        s_start = s_tile_idx * P_MAX
        s_size = min(P_MAX, S - s_start)

        sbm.open_scope(name="qr_tile")
        # Stage this s-tile's qr, then flip it K-major for the stationary operand.
        qr_sb = sbm.alloc_stack((P_MAX, q_lora_rank), nl.bfloat16)
        nisa.dma_copy(
            dst=qr_sb[:s_size, :q_lora_rank],
            src=qr_hbm[batch_start + s_start : batch_start + s_start + s_size, :q_lora_rank],
        )
        qr_t_sb = sbm.alloc_stack((P_MAX, num_k_tiles, P_MAX), nl.bfloat16)
        _transpose_to_k_major_bf16(
            sbm, qr_sb, qr_t_sb, m_sz=s_size, m_pad=P_MAX, k_dim=q_lora_rank, src_row_stride=q_lora_rank
        )

        # One PSUM bank per output tile, rotated so adjacent output tiles' matmuls can issue
        # without waiting for the prior bank to drain.
        NUM_PSUM_BANKS = 8
        n_banks = min(num_out_tiles, NUM_PSUM_BANKS)
        q_psum_banks = []
        for _ in range(n_banks):
            q_psum_banks.append(nl.ndarray((P_MAX, PSUM_FMAX), dtype=nl.float32, buffer=nl.psum))

        for out_idx in nl.affine_range(num_out_tiles):
            local_out_off = out_idx * PSUM_FMAX
            out_size = min(PSUM_FMAX, out_extent - local_out_off)
            q_psum = q_psum_banks[out_idx % n_banks]
            for k_tile in nl.affine_range(num_k_tiles):
                # Explicit accumulate: when the bank rotation wraps past n_banks, auto-detect
                # would treat the bank as previously-written and accumulate into the stale
                # contents of its prior owner.
                nisa.nc_matmul(
                    dst=q_psum[:s_size, :out_size],
                    stationary=qr_t_sb[:BF16_K_TILE, k_tile, :s_size],
                    moving=wq_b_sb[:BF16_K_TILE, k_tile, local_out_off : local_out_off + out_size],
                    accumulate=(k_tile != 0),
                )
            nisa.tensor_copy(
                dst=q_row_sb[:s_size, local_out_off : local_out_off + out_size], src=q_psum[:s_size, :out_size]
            )
        sbm.close_scope()

        nisa.dma_copy(
            dst=q_out_hbm[s_start : s_start + s_size, out_col_start:out_col_end],
            src=q_row_sb[:s_size, :out_extent],
            dge_mode=dge_mode.swdge,
        )

    sbm.close_scope()
    return q_out_hbm


def k_projection_bf16(
    sbm: SbufManager,
    x_bf16: nl.NkiTensor,
    row_start: int,
    tile_size: int,
    wk_sb: nl.NkiTensor,
    dim: int,
    head_dim: int,
    k_sb: nl.NkiTensor,
) -> None:
    """BF16 K projection for one s-tile: ``x @ wk -> k_sb[tile_size, head_dim]``.

    BF16 sibling of the K half of ``k_and_weights_projection_mx``. ``wk_sb`` is the caller-hoisted
    K-major weight from :func:`load_weights_bf16` (hoisted once per batch, exactly as the MX path
    hoists ``load_wk_mx_weights``), so the per-s-tile cost is one activation stage + transpose
    plus the accumulating matmul.

    Output stays query-major ``[S, head_dim]`` because that is what the downstream
    LayerNorm + RoPE (``fused_layernorm_rope_k``) and the k_cache transpose expect.

    Args:
        sbm (SbufManager): SBUF stack allocator used for scratch buffers.
        x_bf16 (nl.NkiTensor): ``[B*S, dim]`` bf16 hidden states on HBM.
        row_start (int): row offset of this s-tile within ``x_bf16``.
        tile_size (int): number of real query rows in this s-tile.
        wk_sb (nl.NkiTensor): ``[P_MAX, ceil(dim/128), head_dim]`` bf16 hoisted K weight.
        dim (int): hidden size (the contraction).
        head_dim (int): per-head dimension (<= 128, so one PSUM output tile).
        k_sb (nl.NkiTensor): ``[P_MAX, head_dim]`` f32 destination, written for ``[:tile_size]``.
    """
    num_k_tiles = div_ceil(dim, BF16_K_TILE)

    sbm.open_scope(name="k_proj_bf16")
    x_sb = sbm.alloc_stack((P_MAX, dim), nl.bfloat16)
    nisa.dma_copy(
        dst=x_sb[:tile_size, :dim],
        src=x_bf16[row_start : row_start + tile_size, :dim],
    )
    x_t_sb = sbm.alloc_stack((P_MAX, num_k_tiles, P_MAX), nl.bfloat16)
    _transpose_to_k_major_bf16(sbm, x_sb, x_t_sb, m_sz=tile_size, m_pad=P_MAX, k_dim=dim, src_row_stride=dim)

    # head_dim <= 128 <= PSUM_FMAX, so the whole output is one PSUM tile.
    k_psum = nl.ndarray((P_MAX, head_dim), dtype=nl.float32, buffer=nl.psum)
    for k_tile in nl.affine_range(num_k_tiles):
        nisa.nc_matmul(
            dst=k_psum[:tile_size, :head_dim],
            stationary=x_t_sb[:BF16_K_TILE, k_tile, :tile_size],
            moving=wk_sb[:BF16_K_TILE, k_tile, :head_dim],
            accumulate=(k_tile != 0),
        )
    nisa.tensor_copy(dst=k_sb[:tile_size, :head_dim], src=k_psum[:tile_size, :head_dim])
    sbm.close_scope()
