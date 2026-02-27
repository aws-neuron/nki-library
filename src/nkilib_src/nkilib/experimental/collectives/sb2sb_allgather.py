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
"""SBUF-to-SBUF All-Gather collective kernels for TRN2 platform."""

import nki
import nki.collectives as ncc
import nki.isa as nisa
import nki.language as nl
from nki.collectives import ReplicaGroup

from ...core.utils.kernel_assert import kernel_assert


@nki.jit(platform_target="trn2")
def allgather_sb2sb(
    inp: nl.ndarray,
    replica_groups: ReplicaGroup,
    tp_degree: int,
) -> nl.ndarray:
    """SBUF-to-SBUF all-gather kernel for gathering tensors across ranks.

    Gathers input tensors from all ranks along the last dimension (K dimension).
    Each rank contributes its local tensor, and all ranks receive the concatenated result.
    Optimized for small tensors that fit entirely in SBUF (H * W <= SBUF capacity).

    Dimensions:
        H: Height dimension (partition dimension, typically <= 128)
        W: Width dimension per rank (local width before gather)
        K: Total width after gather (K = W * tp_degree)

    Args:
        inp (nl.ndarray): [H, W], Input tensor on HBM, where W is the local width per rank.
        replica_groups (ReplicaGroup): ReplicaGroup defining which ranks participate in the collective.
        tp_degree (int): Tensor parallelism degree (number of ranks in the group).

    Returns:
        out (nl.ndarray): [H, K], Output tensor on shared HBM containing gathered data from all ranks.

    Notes:
        - Input tensor must fit in SBUF (H * W * dtype_size <= SBUF capacity)
        - Output is stored in shared_hbm for cross-rank visibility
        - All ranks receive identical output after the collective
        - TODO: Specify intended usage range (e.g., maximum H, W dimensions)

    Pseudocode:
        # Load input from HBM to SBUF
        in_buf = load_to_sbuf(inp)

        # Perform all-gather collective in SBUF
        out_buf = all_gather(in_buf, along_dim=1)

        # Store result from SBUF to shared HBM
        out = store_to_hbm(out_buf)
        return out

    Example:
        With tp_degree=4 and input shape (128, 512) per rank:

        Before (each rank has unique data):
            rank0.inp = data_0,  rank1.inp = data_1,  rank2.inp = data_2,  rank3.inp = data_3

        After (all ranks have identical concatenated output):
            all ranks: out = [data_0 | data_1 | data_2 | data_3]  (shape 128, 2048)
    """
    H, W = inp.shape
    K = W * tp_degree
    dtype = inp.dtype

    kernel_assert(H <= 128, "H must be <= 128 to fit in SBUF partition")

    in_buf = nl.ndarray((H, W), dtype=dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=in_buf, src=inp[0:H, 0:W])

    out_buf = nl.ndarray((H, K), dtype=dtype, buffer=nl.sbuf)
    out = nl.ndarray((H, K), dtype=dtype, buffer=nl.shared_hbm)

    ncc.all_gather(dsts=[out_buf], srcs=[in_buf], replica_group=replica_groups, collective_dim=1)

    nisa.dma_copy(dst=out[0:H, 0:K], src=out_buf)
    return out


@nki.jit(platform_target="trn2")
def allgather_sb2sb_tiled(
    inp: nl.ndarray,
    replica_groups: ReplicaGroup,
    tp_degree: int,
) -> nl.ndarray:
    """SBUF-to-SBUF all-gather with tiling and LNC support for larger tensors.

    Extends allgather_sb2sb with tiling on the M dimension to handle larger tensors
    that don't fit in SBUF. Supports LNC (Logical NeuronCore) sharding where tiles
    are distributed across LNC cores. Optimized for tensors where M > 128.

    Dimensions:
        M: Height dimension (tiled along this dimension)
        K: Width dimension per rank (local width before gather)
        TILE_M: Tile size along M dimension (capped at 128)
        NUM_M_TILES: Number of tiles along M dimension

    Args:
        inp (nl.ndarray): [M, K], Input tensor on HBM, where K is the local width per rank.
        replica_groups (ReplicaGroup): ReplicaGroup defining which ranks participate in the collective.
        tp_degree (int): Tensor parallelism degree (number of ranks in the group).

    Returns:
        result (nl.ndarray): [M, K * tp_degree], Output tensor on shared HBM containing gathered data.

    Notes:
        - TILE_M is capped at 128 (SBUF partition size limit)
        - When launched with LNC grid [lnc], tiles are distributed across LNC cores
        - Each LNC core processes TILES_PER_CORE = NUM_M_TILES // n_prgs tiles
        - Assumes M is evenly divisible by 128 when M > 128
        - TODO: Specify intended usage range (e.g., maximum M, K dimensions)
    """
    M, K_local = inp.shape
    K_total = K_local * tp_degree
    dtype = inp.dtype

    # Tile size capped at 128 (SBUF partition size limit)
    TILE_M = min(M, 128)
    kernel_assert(M <= 128 or M % 128 == 0, "M must be <= 128 or divisible by 128")
    NUM_M_TILES = (M + TILE_M - 1) // TILE_M

    # LNC configuration
    grid_ndim = nl.program_ndim()
    n_prgs = nl.num_programs(axes=0) if grid_ndim != 0 else 1
    prg_id = nl.program_id(axis=0) if grid_ndim != 0 else 0

    if NUM_M_TILES == 1:
        TILES_PER_CORE = 1
        TILE_START = 0
    else:
        kernel_assert(NUM_M_TILES % n_prgs == 0, "NUM_M_TILES must be divisible by number of LNC programs")
        TILES_PER_CORE = NUM_M_TILES // n_prgs if n_prgs > 1 else NUM_M_TILES
        TILE_START = prg_id * TILES_PER_CORE

    result = nl.ndarray((M, K_total), dtype=dtype, buffer=nl.shared_hbm)

    for local_tile_idx in nl.affine_range(TILES_PER_CORE):
        m_tile_idx = TILE_START + local_tile_idx

        # Per-tile SBUF buffers
        src_tile = nl.ndarray((TILE_M, K_local), dtype=dtype, buffer=nl.sbuf)
        dst_tile = nl.ndarray((TILE_M, K_total), dtype=dtype, buffer=nl.sbuf)

        # Load this M tile from HBM to SBUF
        nisa.dma_copy(dst=src_tile, src=inp[m_tile_idx * TILE_M : (m_tile_idx + 1) * TILE_M, 0:K_local])

        # All-gather this tile (SB2SB)
        ncc.all_gather(dsts=[dst_tile], srcs=[src_tile], replica_group=replica_groups, collective_dim=1)

        # Store result from SBUF to HBM
        nisa.dma_copy(dst=result[m_tile_idx * TILE_M : (m_tile_idx + 1) * TILE_M, 0:K_total], src=dst_tile)

    return result
