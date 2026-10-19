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

"""Index tables that resolve indexer-selected prior positions into paged KV cache rows."""

from typing import NamedTuple

import nki.isa as nisa
import nki.language as nl

from ....core.utils.kernel_helpers import div_ceil
from . import P_MAX

_BLOCK_TABLE_RESIDENT_BUDGET = 4096
"""
Columns per partition the resident block table may occupy.

``nc_n_gather`` indexes within a partition, so the block table has to be broadcast to all
partitions, costing ``seq_count * max_blocks_per_seq`` columns each. Sequences are therefore
resolved in groups small enough to keep that inside this budget.
"""

_UNSELECTED_POSITION = -1
"""
Position written into the ragged tail of the index tables.

The tail exists when ``n_prior`` is not a multiple of ``P_MAX``: the last tile then holds fewer
than ``P_MAX`` selected keys. A negative sentinel keeps the tail out of the causal predicate's way,
because the predicate keeps a key when its position is below the current one and the current
token's own key shares that same ragged tile. Those tail slots are never scored, so keeping them
is inert; letting an arbitrary value land there is not, since a large one would drop the current
token's key along with the tail.
"""


class PriorIndex(NamedTuple):
    """The three views of the selected prior that the attention core consumes."""

    rows: nl.NkiTensor  # [P_MAX, n_tokens, n_prior_tiles] int32   flat cache rows, gather order
    key_pos: nl.NkiTensor  # [P_MAX, n_tokens, n_prior_tiles] fp32    absolute positions, key-major
    rows_natural: nl.NkiTensor  # [P_MAX, n_tokens, n_prior_tiles] uint32  flat cache rows, natural order


def build_prior_index(
    gather_indices: nl.NkiTensor,  # [n_tokens, n_prior] int32
    block_table: nl.NkiTensor,  # [n_tokens, max_blocks_per_seq] int32
    kv_cache: nl.NkiTensor,  # [n_blocks, 1, block_size, kv_row_dim]
) -> PriorIndex:
    """
    Resolves each selected prior position to a flat cache row, and publishes it in both orders.

    Callers should invoke this **before** the QKV front-fold even though the core consumes it after.
    ``block_table`` and ``gather_indices`` total under 1 MB, but issued behind the projection weights
    they queue behind roughly 31 MB on the same DMA queue and do not land until about 97 us, and
    nothing else can start because the index gates every per-token gather. Ordering it first moved
    ``block_table``'s arrival to 38.6 us and lifted gather/front-fold overlap from 2.9 to 29.9 us.

    Dimensions:
        n_tokens: Decode tokens in the batch, one per sequence. Must be <= P_MAX.
        n_prior: Prior keys the indexer selected per token.
        n_prior_tiles: Partition tiles those keys span, ``div_ceil(n_prior, P_MAX)``.
        max_blocks_per_seq: Width of the block table.

    Args:
        gather_indices: ``[n_tokens, n_prior]`` int32. Absolute prior positions the indexer selected.
        block_table: ``[n_tokens, max_blocks_per_seq]`` int32. Logical to physical block map.
        kv_cache: ``[n_blocks, 1, block_size, kv_row_dim]``. Read for its paged geometry only.

    Returns:
        PriorIndex with all three tables shaped ``[P_MAX, n_tokens, n_prior_tiles]``, holding key
        ``tile * P_MAX + p`` of token ``t`` at ``[p, t, tile]``.

    Notes:
        ``rows`` and ``rows_natural`` hold the same row indices in two different orders because two
        different consumers read them. ``rows`` is permuted so that the core's single batched
        indirect gather, whose descriptor walks the ``[P_MAX, n_prior_tiles]`` slice one way, visits
        keys in natural order. ``rows_natural`` is left unpermuted and cast to uint32 for the
        indirect ``dma_transpose``, which flattens its index table the other way and so needs the
        unpermuted table to land its output columns in natural key order.

        The permutation is done through HBM rather than on chip, because it is a partition/free axis
        swap of an int32 table that no compute engine can perform in place.
    """

    n_tokens, n_prior = gather_indices.shape[0], gather_indices.shape[1]
    max_blocks_per_seq = block_table.shape[1]
    n_prior_tiles = div_ceil(n_prior, P_MAX)

    seq_group = max(1, min(n_tokens, _BLOCK_TABLE_RESIDENT_BUDGET // max_blocks_per_seq))

    rows_sbuf = nl.ndarray((P_MAX, n_tokens, n_prior_tiles), dtype=nl.int32, buffer=nl.sbuf)
    key_pos_sbuf = nl.ndarray((P_MAX, n_tokens, n_prior_tiles), dtype=nl.float32, buffer=nl.sbuf)
    for group_start in range(0, n_tokens, seq_group):
        group_count = min(seq_group, n_tokens - group_start)

        group_rows, group_pos = _resolve_cache_rows(
            gather_indices, block_table, kv_cache, n_prior, group_start, group_count
        )
        nisa.tensor_copy(
            dst=rows_sbuf[0:P_MAX, group_start : group_start + group_count, 0:n_prior_tiles],
            src=group_rows[0:P_MAX, 0:group_count, 0:n_prior_tiles],
        )
        # int32 positions widen to fp32 here, which is the dtype the causal predicate compares in.
        nisa.tensor_copy(
            dst=key_pos_sbuf[0:P_MAX, group_start : group_start + group_count, 0:n_prior_tiles],
            src=group_pos[0:P_MAX, 0:group_count, 0:n_prior_tiles],
        )

    rows_natural_sbuf = nl.ndarray((P_MAX, n_tokens, n_prior_tiles), dtype=nl.uint32, buffer=nl.sbuf)
    nisa.tensor_copy(
        dst=rows_natural_sbuf[0:P_MAX, 0:n_tokens, 0:n_prior_tiles],
        src=rows_sbuf[0:P_MAX, 0:n_tokens, 0:n_prior_tiles],
    )

    # Swap the partition and tile axes by writing one traversal order and reading back the other.
    token_stride = P_MAX * n_prior_tiles
    permute_hbm = nl.ndarray((n_tokens * token_stride,), dtype=nl.int32, buffer=nl.private_hbm)
    nisa.dma_copy(
        dst=permute_hbm.ap(pattern=[[n_prior_tiles, P_MAX], [token_stride, n_tokens], [1, n_prior_tiles]], offset=0),
        src=rows_sbuf[0:P_MAX, 0:n_tokens, 0:n_prior_tiles],
    )
    rows_gather_order_sbuf = nl.ndarray((P_MAX, n_tokens, n_prior_tiles), dtype=nl.int32, buffer=nl.sbuf)
    nisa.dma_copy(
        dst=rows_gather_order_sbuf[0:P_MAX, 0:n_tokens, 0:n_prior_tiles],
        src=permute_hbm.ap(pattern=[[1, P_MAX], [token_stride, n_tokens], [P_MAX, n_prior_tiles]], offset=0),
    )

    return PriorIndex(rows_gather_order_sbuf, key_pos_sbuf, rows_natural_sbuf)


def _resolve_cache_rows(
    gather_indices: nl.NkiTensor,
    block_table: nl.NkiTensor,
    kv_cache: nl.NkiTensor,
    n_prior: int,
    seq_start: int,
    seq_count: int,
) -> tuple[nl.NkiTensor, nl.NkiTensor]:
    """
    Resolves one group of sequences' selected positions into flat cache rows, key-major.

    Returns ``(cache_rows, positions)``, both ``[P_MAX, seq_count, n_prior_tiles]`` int32.

    Computes ``cache_row = block_table[seq, pos >> block_shift] * block_size + (pos & (block_size -
    1))``, then clamps it into the cache's physical extent so the gather never relies on out-of-bounds
    handling.
    """

    n_blocks, block_size = kv_cache.shape[0], kv_cache.shape[2]
    max_blocks_per_seq = block_table.shape[1]
    n_prior_tiles = div_ceil(n_prior, P_MAX)
    block_shift = block_size.bit_length() - 1

    # Load the selected positions key-major: key `tile * P_MAX + p` lands on partition p. The full
    # tiles and the ragged remainder need different patterns, so they are two transfers.
    positions_sbuf = nl.ndarray((P_MAX, seq_count, n_prior_tiles), dtype=nl.int32, buffer=nl.sbuf)
    nisa.memset(dst=positions_sbuf[0:P_MAX, 0:seq_count, 0:n_prior_tiles], value=_UNSELECTED_POSITION)

    gather_indices_flat = gather_indices.reshape((gather_indices.shape[0] * n_prior, 1))
    n_full_tiles = n_prior // P_MAX
    remainder = n_prior - n_full_tiles * P_MAX
    if n_full_tiles > 0:
        nisa.dma_copy(
            dst=positions_sbuf[0:P_MAX, 0:seq_count, 0:n_full_tiles],
            src=gather_indices_flat.ap(
                pattern=[[1, P_MAX], [n_prior, seq_count], [P_MAX, n_full_tiles]],
                offset=seq_start * n_prior,
            ),
        )
    if remainder > 0:
        nisa.dma_copy(
            dst=positions_sbuf[0:remainder, 0:seq_count, n_full_tiles : n_full_tiles + 1],
            src=gather_indices_flat.ap(
                pattern=[[1, remainder], [n_prior, seq_count], [1, 1]],
                offset=seq_start * n_prior + n_full_tiles * P_MAX,
            ),
        )

    # Split each position into its logical block and its offset inside that block.
    logical_block = nl.ndarray((P_MAX, seq_count, n_prior_tiles), dtype=nl.int32, buffer=nl.sbuf)
    nisa.tensor_scalar(
        logical_block[0:P_MAX, 0:seq_count, 0:n_prior_tiles],
        positions_sbuf[0:P_MAX, 0:seq_count, 0:n_prior_tiles],
        op0=nl.right_shift,
        operand0=block_shift,
    )
    intra_block = nl.ndarray((P_MAX, seq_count, n_prior_tiles), dtype=nl.int32, buffer=nl.sbuf)
    nisa.tensor_scalar(
        intra_block[0:P_MAX, 0:seq_count, 0:n_prior_tiles],
        positions_sbuf[0:P_MAX, 0:seq_count, 0:n_prior_tiles],
        op0=nl.bitwise_and,
        operand0=block_size - 1,
    )

    # Clamp into the block table's extent before indexing it, so the sentinel tail and any
    # out-of-range selection still resolve to a readable entry.
    nisa.tensor_scalar(
        logical_block[0:P_MAX, 0:seq_count, 0:n_prior_tiles],
        logical_block[0:P_MAX, 0:seq_count, 0:n_prior_tiles],
        op0=nl.minimum,
        operand0=max_blocks_per_seq - 1,
    )
    nisa.tensor_scalar(
        logical_block[0:P_MAX, 0:seq_count, 0:n_prior_tiles],
        logical_block[0:P_MAX, 0:seq_count, 0:n_prior_tiles],
        op0=nl.maximum,
        operand0=0,
    )

    # The block table broadcast to every partition, so the on-chip gather can index it per-partition
    # instead of paying an indirect DMA per key tile.
    group_width = seq_count * max_blocks_per_seq
    block_table_flat = block_table.reshape((block_table.shape[0] * max_blocks_per_seq, 1))
    block_table_sbuf = nl.ndarray((P_MAX, group_width), dtype=nl.int32, buffer=nl.sbuf)
    nisa.dma_copy(
        dst=block_table_sbuf[0:P_MAX, 0:group_width],
        src=block_table_flat.ap(pattern=[[0, P_MAX], [1, group_width]], offset=seq_start * max_blocks_per_seq),
    )

    # Each sequence indexes its own row of the resident table, at seq * max_blocks_per_seq + block.
    seq_offset = nl.ndarray((P_MAX, seq_count, n_prior_tiles), dtype=nl.int32, buffer=nl.sbuf)
    nisa.iota(
        seq_offset[0:P_MAX, 0:seq_count, 0:n_prior_tiles],
        pattern=[[max_blocks_per_seq, seq_count], [0, n_prior_tiles]],
        offset=0,
        channel_multiplier=0,
    )
    table_index = nl.ndarray((P_MAX, seq_count, n_prior_tiles), dtype=nl.int32, buffer=nl.sbuf)
    nisa.tensor_tensor(
        table_index[0:P_MAX, 0:seq_count, 0:n_prior_tiles],
        logical_block[0:P_MAX, 0:seq_count, 0:n_prior_tiles],
        seq_offset[0:P_MAX, 0:seq_count, 0:n_prior_tiles],
        nl.add,
    )
    table_index_u32 = nl.ndarray((P_MAX, seq_count, n_prior_tiles), dtype=nl.uint32, buffer=nl.sbuf)
    nisa.tensor_copy(
        dst=table_index_u32[0:P_MAX, 0:seq_count, 0:n_prior_tiles],
        src=table_index[0:P_MAX, 0:seq_count, 0:n_prior_tiles],
    )

    physical_block = nl.ndarray((P_MAX, seq_count, n_prior_tiles), dtype=nl.int32, buffer=nl.sbuf)
    nisa.nc_n_gather(
        dst=physical_block[0:P_MAX, 0:seq_count, 0:n_prior_tiles],
        data=block_table_sbuf[0:P_MAX, 0:group_width],
        indices=table_index_u32[0:P_MAX, 0:seq_count, 0:n_prior_tiles],
    )

    cache_rows = nl.ndarray((P_MAX, seq_count, n_prior_tiles), dtype=nl.int32, buffer=nl.sbuf)
    nisa.scalar_tensor_tensor(
        dst=cache_rows[0:P_MAX, 0:seq_count, 0:n_prior_tiles],
        data=physical_block[0:P_MAX, 0:seq_count, 0:n_prior_tiles],
        op0=nl.multiply,
        operand0=float(block_size),
        op1=nl.add,
        operand1=intra_block[0:P_MAX, 0:seq_count, 0:n_prior_tiles],
    )

    # Clamp to the physical cache extent, which removes all reliance on out-of-bounds DMA handling.
    nisa.tensor_scalar(
        cache_rows[0:P_MAX, 0:seq_count, 0:n_prior_tiles],
        cache_rows[0:P_MAX, 0:seq_count, 0:n_prior_tiles],
        op0=nl.minimum,
        operand0=n_blocks * block_size - 1,
    )
    nisa.tensor_scalar(
        cache_rows[0:P_MAX, 0:seq_count, 0:n_prior_tiles],
        cache_rows[0:P_MAX, 0:seq_count, 0:n_prior_tiles],
        op0=nl.maximum,
        operand0=0,
    )

    return cache_rows, positions_sbuf
