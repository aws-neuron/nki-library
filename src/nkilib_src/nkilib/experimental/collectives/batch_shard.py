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
"""QKV batch shard kernel for TP->DP transitions in batch sharding attention."""

from enum import Enum
from typing import Optional

import nki
import nki.collectives as ncc
import nki.isa as nisa
import nki.language as nl
from nki.collectives import ReplicaGroup

from ...core.utils.kernel_assert import kernel_assert


class AttnQBatchShardLayout(Enum):
    """Layout for QKV batch shard kernel output."""

    NBSd = 0  # (N_heads, B, S, d)
    dBnS = 1  # (d, B, n_heads, S)


@nki.jit
def attn_q_batch_shard(
    input: nl.ndarray,
    iota_workers: nl.ndarray,
    gathered_buf: nl.ndarray,
    gqa_group_size: int,
    replica_group: ReplicaGroup,
    layout: AttnQBatchShardLayout = AttnQBatchShardLayout.NBSd,
    rank_id_in: Optional[nl.ndarray] = None,
) -> nl.ndarray:
    """QKV batch shard kernel: all_gather on dim 0 + reshape to separate gqa_group_size heads + slice batch.

    Implements the Q projection transition from TP64 to TP8DP8 for batch sharding attention.
    Each rank starts with n Q heads for all B batches, ends with gqa_group_size*n Q heads for B/gqa_group_size batches.

    The pattern works when either: (1) heads are in dim=0 (NBSd), or (2) n=1 (any layout).
    all_gather on dim 0, reshape to (G, dim0, ...), slice batch (where G = gqa_group_size):

        Input (per rank)          all_gather dim 0           reshape                    slice batch              reshape back
        ────────────────          ────────────────           ───────                    ───────────              ────────────
        (n, B, S, d)         ->   (G*n, B, S, d)        ->   (G, n, B, S, d)       ->   (G, n, B/G, S, d)   ->   (G*n, B/G, S, d)
        (d, B, n=1, S)       ->   (G*d, B, n=1, S)      ->   (G, d, B, n=1, S)     ->   (G, d, B/G, n=1, S) ->   (n=G, d, B/G, S)

    Example with G=8, n=1, B=32, S=1, d=64:

        NBSd: (1,32,1,64) -> (8,32,1,64) -> (8,1,32,1,64) -> (8,1,4,1,64) -> (8,4,1,64)
        dBnS: (64,32,1,1) -> (512,32,1,1) -> (8,64,32,1,1) -> (8,64,4,1,1) -> (8,64,4,1)

    Args:
        input: Input Q tensor from TP64 projection. First dim is gathered across gqa_group_size ranks.
              NBSd: (n_heads, B, S, d)
              dBnS: (d, B, n_heads, S)
        iota_workers: Lookup table mapping rank_id -> batch_offset for scalar_offset DMA.
                      Shape: (1, collective_ranks), values: [(r % gqa_group_size) * B_per_rank for r in range(collective_ranks)]
                      Needed because NKI compiler doesn't support arithmetic on rank_id.
        gathered_buf: Workspace buffer for all_gather result (must be input tensor for scalar_offset)
        gqa_group_size: GQA group size (e.g., 8 for TP8DP8, 2 for TP2DP2)
        replica_group: ReplicaGroup defining the collective topology
        layout: Output layout - NBSd or dBnS
        rank_id_in: Optional rank_id as input tensor (1,1) int32. If None, uses ncc.rank_id().

    Returns:
        Output Q tensor with gqa_group_size*n heads for this rank's batch slice (4D, same layout as input)
        NBSd: (gqa_group_size*n_heads, B/gqa_group_size, S, d)
        dBnS: (n=gqa_group_size, d, B/gqa_group_size, S)
    """

    # Get rank_id
    if rank_id_in is not None:
        rank_id_sb = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf)
        nisa.dma_copy(dst=rank_id_sb, src=rank_id_in)
        rank_id = rank_id_sb
    else:
        rank_id = ncc.rank_id()

    # Layout-specific shapes and per-G split sizes
    G = gqa_group_size
    if layout == AttnQBatchShardLayout.NBSd:
        n, B, S, d = input.shape
        gathered_shape = (G * n, B, S, d)
        final_shape = (G * n, B // G, S, d)
    else:  # dBnS (requires n=1)
        d, B, n, S = input.shape
        kernel_assert(n == 1, f"dBnS layout requires n=1, got n={n}")
        gathered_shape = (G * d, B, n, S)
        final_shape = (G, d, B // G, S)  # G becomes the new n dimension

    B_per_rank = B // G

    # WORKAROUND: Copy input to shared_hbm because all_gather requires src in shared_hbm.
    # Using x_in directly causes compiler error:
    #   "Error from .../inst_visitor.cpp:3377 in function 'checkCollective'"
    # name= required on collective src/dst: NCC_IBIR440 DRAM allocation failure without it
    src = nl.ndarray(input.shape, dtype=input.dtype, buffer=nl.shared_hbm, name="src")
    nisa.dma_copy(dst=src, src=input)

    # all_gather on dim 0
    gathered = nl.ndarray(gathered_shape, dtype=input.dtype, buffer=nl.shared_hbm, name="dst")
    ncc.all_gather(dsts=[gathered], srcs=[src], replica_group=replica_group, collective_dim=0)

    # reshape to separate G: (G*dim0, ...) -> (G, dim0, ...)
    # split the gathered G*x leading dim into (G, x)
    gathered_view = gathered.reshape_dim(0, (G, -1))

    # WORKAROUND: Copy to input buffer because scalar_offset fails on internal tensors.
    # Using gathered directly causes compiler error:
    #   "Assertion `tensorId >= 0 && "Request tensorId must >= 0"' failed"
    # gathered_buf must be passed as kernel input for scalar_offset to work.
    nisa.dma_copy(dst=gathered_buf, src=gathered_view)

    # WORKAROUND: Convert rank_id to batch_offset using iota lookup table.
    # NKI compiler doesn't support arithmetic on rank_id (e.g., rank_id % gqa_group_size * B_per_rank),
    # so we pre-compute the mapping and use scalar_offset to look it up.
    # Error without workaround: "unimplemented operator 'mod'" or "unimplemented operator 'mul'"
    batch_offset_sb = nl.ndarray([1, 1], dtype=nl.int32, buffer=nl.sbuf)
    nisa.dma_copy(
        dst=batch_offset_sb, src=iota_workers.ap(pattern=[[1, 1], [1, 1]], scalar_offset=rank_id, indirect_dim=1)
    )

    # Extract this rank's batch slice using dynamic offset
    slice_view = gathered_buf.slice(dim=2, start=0, end=B_per_rank)
    slice_pattern, slice_offset = slice_view.get_pattern(), slice_view.offset
    q_out = nl.ndarray(slice_view.shape, dtype=input.dtype, buffer=nl.shared_hbm)
    nisa.dma_copy(
        dst=q_out,
        src=gathered_buf.ap(pattern=slice_pattern, offset=slice_offset, scalar_offset=batch_offset_sb, indirect_dim=2),
    )

    # Reshape back to 4D - just a view, no copy
    return q_out.reshape(final_shape)
