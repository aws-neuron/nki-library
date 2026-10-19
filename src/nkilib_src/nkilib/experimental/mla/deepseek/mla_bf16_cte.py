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

"""BF16 sub-kernels for the split sparse-MLA CTE kernels (``MlaPrecision.BF16``).

The bf16 siblings of the MX helpers in :mod:`mla_common_cte`. Everything the MX path
needs to reach the tensor engine — 4-packing, the ``[w//512, 128, 4] -> [4, w//512, 128]``
output-column swizzle, the block-128 -> block-32 scale broadcast, ``quantize_mx`` and the
matching un-swizzle on the way out — exists only to feed ``nisa.nc_matmul_mx``. In bf16
none of it applies: weights are plain ``[K, N]`` bf16 in NATURAL column order and the
matmuls are back-to-back ``nisa.nc_matmul``.

What is left is the one thing bf16 shares with MX: ``nc_matmul`` contracts along the
PARTITION axis of both operands, so an activation that arrives as ``[s, K]`` must be
transposed to ``[K, s]`` before it can be a stationary operand. These helpers therefore
carry a common SBUF convention:

    weights  : ``[P_MAX, num_k_tiles, N]``  bf16 -- k tile ``t`` holds HBM rows
               ``[t * 128, (t + 1) * 128)``; a trailing partial tile fills only its
               first ``K % 128`` partitions.
    activations (transposed): ``[P_MAX, num_k_tiles, m_pad]`` bf16 -- same k tiling,
               free axis is the token (M) axis.

Both are the bf16 analogue of the MX ``[P_MAX, num_512_tiles, *]`` layout, with the k
tile shrinking from 512 (128 partitions x 4-pack) to 128 (no pack).

Unlike the MX K tiling, nothing here requires K to be a multiple of 512 — only of 128,
and even that is relaxed for the absorption contraction (``qk_nope_head_dim``), which
GLM-MoE-DSA sets to 192. See :func:`_bf16_k_tile_sizes`.
"""

from typing import List, Tuple

import nki.isa as nisa
import nki.language as nl
from nki.isa.constants import dge_mode

from ....core.qkv.qkv_cte import _get_psum_bank_size
from ....core.utils.allocator import SbufManager
from ....core.utils.kernel_helpers import div_ceil

# Tensor-engine partition cap == the bf16 K-tile height (no 4-pack, unlike MX's 512).
_BF16_K_TILE = 128

# Moving-operand free-dim cap == one fp32 PSUM bank. Kept at the MX path's value rather
# than the v4 bf16-dst maximum so both precisions bank PSUM identically.
_F_MAX = 512

# PSUM banks used to pipeline the activation transposes (transpose bank b while bank b-1
# is copied out). 4 leaves half of PSUM free for whatever matmul is in flight.
_NUM_TRANSPOSE_BANKS = 4


def _bf16_k_tile_sizes(k_dim: int) -> List[int]:
    """Row count of each 128-row K tile of a ``k_dim``-deep contraction.

    Returns ``[128, 128, ..., k_dim % 128]`` -- a list of length ``ceil(k_dim / 128)``
    whose last entry is short when ``k_dim`` is not a multiple of 128. The MX path cannot
    express a partial tile (its 512-element tile is a hard packing constraint), which is
    why the MX absorption asserts ``qk_nope_head_dim == 128``; bf16 just runs one extra
    accumulating ``nc_matmul`` over the short tile, so GLM-MoE-DSA's
    ``qk_nope_head_dim = 192`` becomes ``[128, 64]``.
    """
    num_tiles = div_ceil(k_dim, _BF16_K_TILE)
    return [min(_BF16_K_TILE, k_dim - t * _BF16_K_TILE) for t in range(num_tiles)]


def _bf16_compute_k_slab_tiles(
    k_dim: int,
    out_dim: int,
    sbuf_budget_bytes: int,
    reserve_bytes: int,
) -> int:
    """Choose the stage-1 weight K-slab size, in 128-row K tiles.

    bf16 weights cost 2 bytes/element against the MX path's ~1.25 (fp8x4 data + the
    broadcast uint8 block scale), so the same projection needs a ~1.6x larger slab budget
    and slabs where MX would not. Returns the largest DIVISOR of ``ceil(k_dim / 128)``
    whose slab fits ``sbuf_budget_bytes - reserve_bytes``, so ``num_slabs`` divides evenly
    and every slab matmul is the same shape (1 = full residency, no slabbing).

    ``reserve_bytes`` must cover everything live alongside the slab: the staged + transposed
    activation buffers, the stage-1 outputs, and the per-tile working set.
    """
    num_k_tiles = div_ceil(k_dim, _BF16_K_TILE)
    bytes_per_k_tile = out_dim * 2  # bf16
    budget = sbuf_budget_bytes - reserve_bytes
    if budget < bytes_per_k_tile:
        return 1
    tiles_that_fit = budget // bytes_per_k_tile
    for slab_tiles in range(min(num_k_tiles, tiles_that_fit), 0, -1):
        if num_k_tiles % slab_tiles == 0:
            return slab_tiles
    return 1


def _load_bf16_weights_k_slab(
    weights_hbm: nl.NkiTensor,
    weights_sb: nl.NkiTensor,
    out_dim: int,
    k_tile_start: int,
    k_tile_count: int,
    k_tile_sizes: List[int],
    full_out_dim: int,
    out_col_offset: int = 0,
) -> None:
    """Load ``k_tile_count`` K tiles of a bf16 ``[K, full_out_dim]`` weight into ``weights_sb``.

    ``weights_sb`` is ``[P_MAX, >= k_tile_count, out_dim]``; tile ``i`` receives HBM rows
    ``[(k_tile_start + i) * 128, ...)`` and columns ``[out_col_offset, + out_dim)``.

    The full-height tiles go out as ONE 3D DMA (partition = row-within-tile, middle = tile,
    inner = column) because the source is perfectly rectangular. A trailing partial tile
    (see :func:`_bf16_k_tile_sizes`) has a different partition count and so cannot ride the
    same access pattern; it gets its own 2D DMA.

    Args:
        weights_hbm: ``[K, full_out_dim]`` bf16 weight on HBM, natural column order.
        weights_sb: destination ``[P_MAX, num_k_tiles, out_dim]`` bf16 SBUF buffer.
        out_dim: number of columns to load (slice width).
        k_tile_start: index of the first K tile to load.
        k_tile_count: number of K tiles to load.
        k_tile_sizes: per-tile row counts for the WHOLE weight, from ``_bf16_k_tile_sizes``.
        full_out_dim: N of the HBM tensor (row stride).
        out_col_offset: column index of the slice start.
    """
    # Full-height tiles in this slab, counted from the front (a short tile can only be last).
    num_full = 0
    for i in range(k_tile_count):
        if k_tile_sizes[k_tile_start + i] == _BF16_K_TILE:
            num_full += 1
        else:
            break

    if num_full > 0:
        nisa.dma_copy(
            dst=weights_sb[0:_BF16_K_TILE, 0:num_full, 0:out_dim],
            src=weights_hbm.ap(
                pattern=[[full_out_dim, _BF16_K_TILE], [_BF16_K_TILE * full_out_dim, num_full], [1, out_dim]],
                offset=k_tile_start * _BF16_K_TILE * full_out_dim + out_col_offset,
            ),
            dge_mode=dge_mode.hwdge,
        )

    for i in range(num_full, k_tile_count):
        rows = k_tile_sizes[k_tile_start + i]
        nisa.dma_copy(
            dst=weights_sb[0:rows, i, 0:out_dim],
            src=weights_hbm.ap(
                pattern=[[full_out_dim, rows], [1, out_dim]],
                offset=(k_tile_start + i) * _BF16_K_TILE * full_out_dim + out_col_offset,
            ),
            dge_mode=dge_mode.hwdge,
        )


def _load_bf16_weights(
    weights_hbm: nl.NkiTensor,
    k_dim: int,
    out_dim: int,
    sbm: SbufManager,
    name: str = "bf16_w",
    full_out_dim: int = None,
    out_col_offset: int = 0,
) -> nl.NkiTensor:
    """Allocate and fully load a bf16 ``[k_dim, full_out_dim]`` weight slice into SBUF.

    Convenience wrapper over :func:`_load_bf16_weights_k_slab` for weights small enough to
    stay resident. Returns ``[P_MAX, ceil(k_dim / 128), out_dim]`` bf16.
    """
    if full_out_dim is None:
        full_out_dim = out_dim
    k_tile_sizes = _bf16_k_tile_sizes(k_dim)
    num_k_tiles = len(k_tile_sizes)
    weights_sb = sbm.alloc_stack(
        (nl.tile_size.pmax, num_k_tiles, out_dim), dtype=nl.bfloat16, buffer=nl.sbuf, name=f"{name}_weights"
    )
    _load_bf16_weights_k_slab(
        weights_hbm,
        weights_sb,
        out_dim=out_dim,
        k_tile_start=0,
        k_tile_count=num_k_tiles,
        k_tile_sizes=k_tile_sizes,
        full_out_dim=full_out_dim,
        out_col_offset=out_col_offset,
    )
    return weights_sb


def _load_bf16_gamma(
    gamma_hbm: nl.NkiTensor,
    dim: int,
    sbm: SbufManager,
    name: str = "gamma",
) -> nl.NkiTensor:
    """Broadcast a ``[1, dim]`` bf16 norm gamma across all partitions.

    Returns ``[P_MAX, dim]`` bf16 so a single ``tensor_tensor`` multiply applies gamma to a
    whole ``[s_tile, dim]`` tile. One stride-0 partition DMA (the DMA engine supports
    partition-axis stride 0 natively). The MX sibling
    (``_load_norm_weights_for_mx``) instead gathers gamma into the swizzled fp32 MX layout
    so it can be fused into the MX transpose; in bf16 there is no swizzle to fuse into.
    """
    P_MAX = nl.tile_size.pmax
    gamma_sb = sbm.alloc_stack((P_MAX, dim), dtype=nl.bfloat16, buffer=nl.sbuf, name=f"{name}_bf16")
    nisa.dma_copy(
        dst=gamma_sb[0:P_MAX, 0:dim],
        src=gamma_hbm.reshape((dim,)).ap(pattern=[[0, P_MAX], [1, dim]], offset=0),
        dge_mode=dge_mode.hwdge,
    )
    return gamma_sb


def _transpose_bf16_to_k_major(
    src_sb: nl.NkiTensor,
    out_sb: nl.NkiTensor,
    m_sz: int,
    m_pad: int,
    k_tile_sizes: List[int],
    src_row_stride: int = None,
    src_offset: int = 0,
) -> None:
    """Transpose ``[m_sz, K]`` (M-major, in SBUF) into ``[P_MAX, num_k_tiles, m_pad]`` (K-major).

    ``nc_matmul`` contracts along the partition axis of BOTH operands, so an activation
    that lands as ``[token, feature]`` has to be flipped before it can be a stationary
    operand. One PE transpose per 128-column K tile, cycling ``_NUM_TRANSPOSE_BANKS`` PSUM
    banks and alternating the read-out engine so the copies overlap the next transpose.

    Only the ``m_sz`` real columns are written; ``out_sb[:, :, m_sz:m_pad]`` is left
    untouched. That is safe because ``dst[m, n]`` of the downstream matmul depends solely on
    column ``m`` of the stationary operand, so garbage in the pad columns can only reach pad
    OUTPUT rows, which every caller discards. (The MX path relies on the same property.)

    Args:
        src_sb: SBUF tile holding ``[m_sz, K]``.
        out_sb: destination ``[P_MAX, num_k_tiles, m_pad]`` bf16.
        m_sz: real token count in this tile.
        m_pad: allocated free width of ``out_sb``'s inner axis.
        k_tile_sizes: per-tile K row counts, from :func:`_bf16_k_tile_sizes`.
        src_row_stride: per-partition free stride of ``src_sb`` in elements; defaults to
            ``sum(k_tile_sizes)`` (a tightly packed ``[m, K]`` buffer).
        src_offset: element offset of the ``[m_sz, K]`` region inside ``src_sb``.
    """
    P_MAX = nl.tile_size.pmax
    PSUM_BANK_SIZE = _get_psum_bank_size()
    num_k_tiles = len(k_tile_sizes)
    if src_row_stride is None:
        src_row_stride = sum(k_tile_sizes)

    tp_psum = []
    for bank_id in range(_NUM_TRANSPOSE_BANKS):
        tp_psum.append(
            nl.ndarray((P_MAX, m_pad), dtype=nl.bfloat16, buffer=nl.psum, address=(0, bank_id * PSUM_BANK_SIZE))
        )

    # out_sb[p, t, m] lives at free offset t * m_pad + m, partition stride num_k_tiles * m_pad.
    out_part_stride = num_k_tiles * m_pad

    for k_tile_idx in range(num_k_tiles):
        k_rows = k_tile_sizes[k_tile_idx]
        bank = tp_psum[k_tile_idx % _NUM_TRANSPOSE_BANKS]
        nisa.nc_transpose(
            data=src_sb.ap(
                pattern=[[src_row_stride, m_sz], [1, k_rows]],
                offset=src_offset + k_tile_idx * _BF16_K_TILE,
            ),
            dst=bank[0:k_rows, 0:m_sz],
        )
        nisa.tensor_copy(
            dst=out_sb.ap(
                pattern=[[out_part_stride, k_rows], [1, m_sz]],
                offset=k_tile_idx * m_pad,
            ),
            src=bank[0:k_rows, 0:m_sz],
            engine=nisa.scalar_engine if k_tile_idx % 2 == 0 else nisa.vector_engine,
        )


def _load_x_bf16_tile(
    x_2d: nl.NkiTensor,
    x_t_sb: nl.NkiTensor,
    s_offset: int,
    s_tile_sz: int,
    s_tile_pad: int,
    hidden_dim: int,
    sbm: SbufManager,
) -> None:
    """Load one s-tile of the bf16 activation ``[S, H]`` and transpose it to K-major.

    Writes ``x_t_sb`` ``[P_MAX, ceil(H / 128), s_tile_pad]``. The staging buffer
    (``[s_tile, H]``) lives in an inner scope and is freed before returning, so the caller
    only pays for the transposed result. bf16 sibling of the MX ``_load_prequant_tile``,
    minus the packed-scale region (bf16 activations carry no scales).
    """
    P_MAX = nl.tile_size.pmax
    k_tile_sizes = _bf16_k_tile_sizes(hidden_dim)

    sbm.open_scope()
    stage_sb = sbm.alloc_stack((P_MAX, hidden_dim), dtype=nl.bfloat16, buffer=nl.sbuf, name=f"x_bf16_stage_{s_offset}")
    nisa.dma_copy(
        dst=stage_sb[0:s_tile_sz, 0:hidden_dim],
        src=x_2d.ap(pattern=[[hidden_dim, s_tile_sz], [1, hidden_dim]], offset=s_offset * hidden_dim),
        dge_mode=dge_mode.hwdge,
    )
    _transpose_bf16_to_k_major(
        stage_sb,
        x_t_sb,
        m_sz=s_tile_sz,
        m_pad=s_tile_pad,
        k_tile_sizes=k_tile_sizes,
        src_row_stride=hidden_dim,
    )
    sbm.close_scope()


def _bf16_matmul_k_range(
    input_t_sb: nl.NkiTensor,
    weights_sb: nl.NkiTensor,
    k_tile_start: int,
    k_tile_count: int,
    k_tile_sizes: List[int],
    m_dim: int,
    n_dim: int,
    output_psum: List[nl.NkiTensor],
    input_k_tile_base: int = None,
) -> None:
    """Accumulate ``input_t[K slab].T @ weights[K slab]`` into caller-owned PSUM banks.

    ``output_psum`` is a list of ``[P_MAX, _F_MAX]`` bf16 PSUM tiles, one per 512-wide N
    tile; repeated calls over disjoint K slabs accumulate into them (``nc_matmul`` with
    ``accumulate=None`` overwrites on the first write to a PSUM location and accumulates
    afterwards, and the tensor engine accumulates in fp32 regardless of the ``dst`` dtype).

    Args:
        input_t_sb: ``[P_MAX, *, m_pad]`` K-major activation.
        weights_sb: ``[P_MAX, k_tile_count, n_dim]`` K-major weight slab.
        k_tile_start: index of this slab's first K tile within the FULL contraction (used to
            look up row counts and, by default, to index ``input_t_sb``).
        k_tile_count: number of K tiles in this slab.
        k_tile_sizes: per-tile K row counts for the full contraction.
        m_dim: M (token) width of the stationary operand.
        n_dim: total N width spanned by ``output_psum``.
        output_psum: PSUM banks to accumulate into.
        input_k_tile_base: K-tile index base for ``input_t_sb`` if it differs from
            ``k_tile_start`` (e.g. a per-slab activation buffer starting at tile 0).
    """
    num_n_tiles = div_ceil(n_dim, _F_MAX)
    if input_k_tile_base is None:
        input_k_tile_base = k_tile_start

    for i_k in range(k_tile_count):
        k_rows = k_tile_sizes[k_tile_start + i_k]
        for i_n in nl.affine_range(num_n_tiles):
            n_tile_sz = min(_F_MAX, n_dim - i_n * _F_MAX)
            nisa.nc_matmul(
                dst=output_psum[i_n][0:m_dim, 0:n_tile_sz],
                stationary=input_t_sb[0:k_rows, input_k_tile_base + i_k, nl.ds(0, m_dim)],
                moving=weights_sb[0:k_rows, i_k, nl.ds(i_n * _F_MAX, n_tile_sz)],
            )


def _alloc_matmul_psum(num_n_tiles: int) -> List[nl.NkiTensor]:
    """Allocate ``num_n_tiles`` consecutive ``[P_MAX, _F_MAX]`` bf16 PSUM banks."""
    P_MAX = nl.tile_size.pmax
    PSUM_BANK_SIZE = _get_psum_bank_size()
    banks = []
    for bank_id in range(num_n_tiles):
        banks.append(
            nl.ndarray((P_MAX, _F_MAX), dtype=nl.bfloat16, buffer=nl.psum, address=(0, bank_id * PSUM_BANK_SIZE))
        )
    return banks


def _copy_psum_splits(
    output_psum: List[nl.NkiTensor],
    m_dim: int,
    split_bounds: List[int],
    sbm: SbufManager,
    name: str,
) -> List[nl.NkiTensor]:
    """Copy 512-banked PSUM out to SBUF, cut at ``split_bounds``.

    ``split_bounds`` is the full boundary list ``[0, ..., n_dim]``; one SBUF buffer is
    returned per adjacent pair. A segment that straddles a bank boundary is copied in
    per-bank pieces.
    """
    P_MAX = nl.tile_size.pmax
    outputs = []
    for split_idx in range(len(split_bounds) - 1):
        start_col = split_bounds[split_idx]
        width = split_bounds[split_idx + 1] - start_col
        out_sb = sbm.alloc_stack((P_MAX, width), dtype=nl.bfloat16, buffer=nl.sbuf, name=f"{name}_out_{split_idx}")
        col = 0
        while col < width:
            bank_idx, bank_offset = divmod(start_col + col, _F_MAX)
            copy_width = min(_F_MAX - bank_offset, width - col)
            nisa.tensor_copy(
                dst=out_sb[0:m_dim, nl.ds(col, copy_width)],
                src=output_psum[bank_idx][0:m_dim, nl.ds(bank_offset, copy_width)],
                engine=nisa.scalar_engine,
            )
            col += copy_width
        outputs.append(out_sb)
    return outputs


def _bf16_matmul(
    input_t_sb: nl.NkiTensor,
    weights_sb: nl.NkiTensor,
    k_dim: int,
    m_dim: int,
    n_dim: int,
    sbm: SbufManager,
    name: str = "bf16_matmul",
) -> nl.NkiTensor:
    """``input_t.T @ weights`` over the whole ``k_dim`` contraction -> ``[m_dim, n_dim]`` bf16 SBUF.

    bf16 sibling of ``_mx_matmul``. Both operands must already be K-major
    (``[P_MAX, ceil(k_dim / 128), *]``).
    """
    k_tile_sizes = _bf16_k_tile_sizes(k_dim)
    num_n_tiles = div_ceil(n_dim, _F_MAX)
    output_psum = _alloc_matmul_psum(num_n_tiles)
    _bf16_matmul_k_range(
        input_t_sb,
        weights_sb,
        k_tile_start=0,
        k_tile_count=len(k_tile_sizes),
        k_tile_sizes=k_tile_sizes,
        m_dim=m_dim,
        n_dim=n_dim,
        output_psum=output_psum,
    )
    return _copy_psum_splits(output_psum, m_dim, [0, n_dim], sbm, name)[0]


def _bf16_matmul_split(
    input_t_sb: nl.NkiTensor,
    weights_sb: nl.NkiTensor,
    k_dim: int,
    m_dim: int,
    n_dim: int,
    split_points: List[int],
    sbm: SbufManager,
    name: str = "bf16_matmul_split",
) -> List[nl.NkiTensor]:
    """``_bf16_matmul`` with the output cut into separate SBUF buffers at ``split_points``.

    bf16 sibling of ``_mx_matmul_split``; the split happens during the PSUM read-out so no
    extra pass over the data is needed.
    """
    k_tile_sizes = _bf16_k_tile_sizes(k_dim)
    num_n_tiles = div_ceil(n_dim, _F_MAX)
    output_psum = _alloc_matmul_psum(num_n_tiles)
    _bf16_matmul_k_range(
        input_t_sb,
        weights_sb,
        k_tile_start=0,
        k_tile_count=len(k_tile_sizes),
        k_tile_sizes=k_tile_sizes,
        m_dim=m_dim,
        n_dim=n_dim,
        output_psum=output_psum,
    )
    return _copy_psum_splits(output_psum, m_dim, [0] + split_points + [n_dim], sbm, name)


def _load_wuk_bf16_group_tiled(
    wuk_hbm: nl.NkiTensor,
    wuk_sb: nl.NkiTensor,
    n_heads: int,
    qk_nope_head_dim: int,
    kv_lora_rank: int,
    group_start: int,
    group_count: int,
) -> None:
    """Load one head-group slice of the absorption weight ``W_uk`` in K-major tiles.

    ``W_uk`` on HBM is ``[qk_nope_head_dim, n_heads * kv_lora_rank]`` bf16 with the
    contraction (nope) on ROWS, so the HBM layout is already K-major and only needs
    splitting across ``ceil(nope / 128)`` partition tiles. ``wuk_sb`` is
    ``[P_MAX, num_nope_tiles, group_count * kv_lora_rank]``; local head ``r`` owns columns
    ``[r * kv_lora_rank, (r + 1) * kv_lora_rank)``.

    Unlike the MX-path loader (``_load_wuk_bf16``, which requires nope == 128 so the whole
    weight is one partition tile), this handles nope > 128 -- GLM-MoE-DSA's 192 becomes
    tiles of 128 + 64.
    """
    _load_bf16_weights_k_slab(
        wuk_hbm,
        wuk_sb,
        out_dim=group_count * kv_lora_rank,
        k_tile_start=0,
        k_tile_count=div_ceil(qk_nope_head_dim, _BF16_K_TILE),
        k_tile_sizes=_bf16_k_tile_sizes(qk_nope_head_dim),
        full_out_dim=n_heads * kv_lora_rank,
        out_col_offset=group_start * kv_lora_rank,
    )


def _absorb_q_nope_bf16(
    q_group_sb: nl.NkiTensor,
    wuk_sb: nl.NkiTensor,
    q_lift_grp_sb: nl.NkiTensor,
    group_n: int,
    q_head_offset: int,
    wuk_head_slot: int,
    lift_slot: int,
    s_tile_sz: int,
    s_tile_pad: int,
    qk_nope_head_dim: int,
    kv_lora_rank: int,
    sbm: SbufManager,
) -> None:
    """Absorb one head: ``q_lift[s, L] = q_nope[s, nope] @ W_uk[h][nope, L]`` in bf16.

    ``q_nope`` sits in ``q_group_sb`` at column ``q_head_offset`` with row stride
    ``group_n``, i.e. M-major, so it is first transposed to ``[nope, s]`` (one PE transpose
    per 128 rows of nope) and then contracted against ``W_uk``'s nope tiles, accumulating
    over the tiles in PSUM. The result is written into ``q_lift_grp_sb`` at column
    ``lift_slot`` -- the caller stores the whole head group with one DMA.

    Generalizes the MX path's single-tile absorption to any ``qk_nope_head_dim``:
    ``ceil(nope / 128)`` transposes and ``ceil(nope / 128)`` accumulating matmuls
    (DeepSeek's 128 -> 1 each, GLM's 192 -> 2 each).
    """
    P_MAX = nl.tile_size.pmax
    PSUM_BANK_SIZE = _get_psum_bank_size()
    nope_tile_sizes = _bf16_k_tile_sizes(qk_nope_head_dim)
    num_nope_tiles = len(nope_tile_sizes)
    num_lift_n_tiles = div_ceil(kv_lora_rank, _F_MAX)

    sbm.open_scope()
    # Transposed q_nope, K-major: [P_MAX, num_nope_tiles, s_tile_pad].
    qnope_t_sb = sbm.alloc_stack((P_MAX, num_nope_tiles, s_tile_pad), dtype=nl.bfloat16, buffer=nl.sbuf, name="qnope_t")
    # Transposes land in the PSUM banks ABOVE the lift accumulators, which are live below.
    _transpose_bf16_to_k_major(
        q_group_sb,
        qnope_t_sb,
        m_sz=s_tile_sz,
        m_pad=s_tile_pad,
        k_tile_sizes=nope_tile_sizes,
        src_row_stride=group_n,
        src_offset=q_head_offset,
    )

    lift_psum = []
    for bank_id in range(num_lift_n_tiles):
        lift_psum.append(
            nl.ndarray(
                (P_MAX, _F_MAX),
                dtype=nl.bfloat16,
                buffer=nl.psum,
                address=(0, (_NUM_TRANSPOSE_BANKS + bank_id) * PSUM_BANK_SIZE),
            )
        )

    for i_k in range(num_nope_tiles):
        k_rows = nope_tile_sizes[i_k]
        for i_n in nl.affine_range(num_lift_n_tiles):
            n_tile_sz = min(_F_MAX, kv_lora_rank - i_n * _F_MAX)
            w_col = wuk_head_slot * kv_lora_rank + i_n * _F_MAX
            nisa.nc_matmul(
                dst=lift_psum[i_n][0:s_tile_sz, 0:n_tile_sz],
                stationary=qnope_t_sb[0:k_rows, i_k, nl.ds(0, s_tile_sz)],
                moving=wuk_sb[0:k_rows, i_k, nl.ds(w_col, n_tile_sz)],
            )

    for i_n in nl.affine_range(num_lift_n_tiles):
        n_tile_sz = min(_F_MAX, kv_lora_rank - i_n * _F_MAX)
        nisa.tensor_copy(
            dst=q_lift_grp_sb[0:s_tile_sz, nl.ds(lift_slot + i_n * _F_MAX, n_tile_sz)],
            src=lift_psum[i_n][0:s_tile_sz, 0:n_tile_sz],
            engine=nisa.scalar_engine,
        )
    sbm.close_scope()


def _bf16_sbuf_footprint(k_dim: int, n_dim: int) -> Tuple[int, int]:
    """``(bytes_per_partition, num_k_tiles)`` for a K-major bf16 ``[k_dim, n_dim]`` weight."""
    num_k_tiles = div_ceil(k_dim, _BF16_K_TILE)
    return num_k_tiles * n_dim * 2, num_k_tiles
