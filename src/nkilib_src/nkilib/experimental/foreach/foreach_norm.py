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

"""NKI kernels for foreach norm computation (L1, L2, Linf). These kernels implement memory-efficient norm operations using SPMD tiling across cores with fused activation-reduce patterns."""

import nki
import nki.isa as nisa
import nki.language as nl

from .foreach_utils import get_spmd_tiling_info

_F_SIZE_BF16 = 16384
_F_SIZE_F32 = 8192


def _norm_compute(tile: nl.ndarray, act_op, reduce_op, accum: nl.ndarray) -> None:
    """
    Per-tile norm computation: apply activation and reduce into accumulator.

    Applies activation then reduces along free dimension, accumulating into accum.

    Args:
        tile (nl.ndarray): Input tile in SBUF. Shape varies by caller.
        act_op: Activation operation (nl.square, nl.abs).
        reduce_op: Reduction operation (nl.add, nl.maximum).
        accum (nl.ndarray): [P_MAX, 1], Accumulator buffer in SBUF.
    """
    nisa.activation(tile, op=act_op, data=tile)
    tile_reduced = nl.ndarray((nl.tile_size.pmax, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_reduce(tile_reduced, op=reduce_op, data=tile, axis=1, keepdims=True)
    nisa.tensor_tensor(accum, accum, tile_reduced, reduce_op)


def _norm_spmd_body(
    data: nl.ndarray,
    numel: int,
    act_op,
    reduce_op,
    accum: nl.ndarray,
) -> int:
    """
    SPMD tiling body for norm kernels: loads tiles and applies norm compute.

    Handles SPMD partitioning, tile sizing, DMA loads, and tail elements.
    Delegates per-tile computation to _norm_compute.

    Args:
        data (nl.ndarray): [N], Input tensor on HBM.
        numel (int): Total number of elements in data.
        act_op: Activation operation (nl.square, nl.abs).
        reduce_op: Reduction operation (nl.add, nl.max).
        accum (nl.ndarray): [P_MAX, 1], Accumulator buffer in SBUF.

    Returns:
        int: Program ID of the current SPMD core.
    """
    P_MAX = nl.tile_size.pmax
    f_size = _F_SIZE_F32 if data.dtype == nl.float32 else _F_SIZE_BF16
    tile_size = P_MAX * f_size
    core_base, core_n_aligned, num_tiles, tail, prog_id = get_spmd_tiling_info(numel, tile_size)

    for tile_idx in nl.affine_range(num_tiles):
        offset = core_base + tile_idx * tile_size
        masked_f = min(f_size, (core_n_aligned - tile_idx * tile_size) // P_MAX)
        tile = nl.ndarray((P_MAX, masked_f), dtype=data.dtype, buffer=nl.sbuf)
        nisa.dma_copy(tile, data.ap([[masked_f, P_MAX], [1, masked_f]], offset))
        _norm_compute(tile, act_op, reduce_op, accum)

    if tail > 0:
        tail_offset = core_base + core_n_aligned
        tail_tile = nl.ndarray((tail, 1), dtype=data.dtype, buffer=nl.sbuf)
        nisa.dma_copy(tail_tile, data.ap([[1, tail], [1, 1]], tail_offset))
        # Tail elements span different partition rows (tail, 1), so we cannot
        # use the fused _norm_compute which accumulates per-partition.  Instead,
        # apply activation, reduce across partitions to a scalar, then fold
        # into accum[0:1, 0:1].
        nisa.activation(tail_tile, op=act_op, data=tail_tile)
        partial = nl.ndarray((1, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_partition_reduce(partial, op=reduce_op, data=tail_tile)
        nisa.tensor_tensor(accum[0:1, 0:1], accum[0:1, 0:1], partial, reduce_op)
    return prog_id


def _norm_cross_core_reduce(
    accum: nl.ndarray,
    reduce_op,
    prog_id: int,
) -> nl.ndarray:
    """
    Reduce accumulator across partitions and, on multi-core (LNC2), exchange
    between SPMD cores.

    Performs partition-level reduction; when more than one SPMD program is
    launched (LNC2) it then does a sendrecv exchange to aggregate the per-core
    partials. With a single program (LNC1) the partition reduction is the final
    result and no cross-core exchange is performed.

    Args:
        accum (nl.ndarray): [P_MAX, 1], Accumulator buffer in SBUF.
        reduce_op: Reduction operation (nl.add, nl.max, etc.).
        prog_id (int): Program ID of the current SPMD core.

    Returns:
        nl.ndarray: [1, 1], Final reduced value in SBUF.
    """
    local_total = nl.ndarray((1, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_partition_reduce(local_total, op=reduce_op, data=accum)
    # With a single SPMD program (LNC1) the partition reduce above already holds
    # the complete result; the cross-core exchange only applies when there is more
    # than one program (LNC2). num_programs(0) is a trace-time constant, so at LNC1
    # this branch is statically dropped and no sendrecv (which requires LNC>=2) is emitted.
    if nl.num_programs(0) > 1:
        remote = nl.ndarray((1, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.sendrecv(
            src=local_total,
            dst=remote,
            pipe_id=0,
            send_to_rank=(1 - prog_id),
            recv_from_rank=(1 - prog_id),
        )
        nisa.tensor_tensor(local_total, local_total, remote, reduce_op)
    return local_total


@nki.jit
def l2_norm_kernel(
    data: nl.ndarray,
    numel: int,
) -> nl.ndarray:
    """
    Compute L2 norm (Euclidean norm) of input tensor.

    Computes sqrt(sum(x^2)) using SPMD parallelization across 2 cores with
    fused activation-reduce and sendrecv-based cross-core reduction.
    TODO: Specify intended usage range (e.g., tensor size, data type constraints).

    Dimensions:
        N: Total number of elements in input tensor

    Args:
        data (nl.ndarray): [N], Input tensor on HBM.
        numel (int): Number of elements in data.

    Returns:
        out (nl.ndarray): [1, 1], L2 norm scalar on HBM.

    Pseudocode:
        # SPMD core 0 and core 1 each process a partition of data
        for each core:
            accum = 0
            for tile in assigned_tiles:
                tile_data = load(tile)
                accum += sum(tile_data^2)
            local_sum = reduce_partitions(accum)

        # Exchange and combine results from both cores
        global_sum = sendrecv_reduce(local_sum, other_core)
        result = sqrt(global_sum)
    """
    out = nl.ndarray((1, 1), dtype=data.dtype, buffer=nl.shared_hbm)
    accum = nl.zeros((nl.tile_size.pmax, 1), dtype=nl.float32, buffer=nl.sbuf)
    prog_id = _norm_spmd_body(data, numel, nl.square, nl.add, accum)
    local_total = _norm_cross_core_reduce(accum, nl.add, prog_id)
    result = nl.ndarray((1, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.activation(result, op=nl.sqrt, data=local_total)
    nisa.dma_copy(out[0:1, 0:1], result)
    return out


@nki.jit
def l1_norm_kernel(
    data: nl.ndarray,
    numel: int,
) -> nl.ndarray:
    """
    Compute L1 norm (Manhattan norm) of input tensor.

    Computes sum(|x|) using SPMD parallelization across 2 cores with
    fused activation-reduce and sendrecv-based cross-core reduction.
    TODO: Specify intended usage range (e.g., tensor size, data type constraints).

    Dimensions:
        N: Total number of elements in input tensor

    Args:
        data (nl.ndarray): [N], Input tensor on HBM.
        numel (int): Number of elements in data.

    Returns:
        out (nl.ndarray): [1, 1], L1 norm scalar on HBM.

    Pseudocode:
        # SPMD core 0 and core 1 each process a partition of data
        for each core:
            accum = 0
            for tile in assigned_tiles:
                tile_data = load(tile)
                accum += sum(|tile_data|)
            local_sum = reduce_partitions(accum)

        # Exchange and combine results from both cores
        global_sum = sendrecv_reduce(local_sum, other_core)
    """
    out = nl.ndarray((1, 1), dtype=data.dtype, buffer=nl.shared_hbm)
    accum = nl.zeros((nl.tile_size.pmax, 1), dtype=nl.float32, buffer=nl.sbuf)
    prog_id = _norm_spmd_body(data, numel, nl.abs, nl.add, accum)
    local_total = _norm_cross_core_reduce(accum, nl.add, prog_id)
    nisa.dma_copy(out[0:1, 0:1], local_total)
    return out


@nki.jit
def linf_norm_kernel(
    data: nl.ndarray,
    numel: int,
) -> nl.ndarray:
    """
    Compute Linf norm (max norm) of input tensor.

    Computes max(|x|) using SPMD parallelization across 2 cores with
    fused activation-reduce and sendrecv-based cross-core reduction.
    TODO: Specify intended usage range (e.g., tensor size, data type constraints).

    Dimensions:
        N: Total number of elements in input tensor

    Args:
        data (nl.ndarray): [N], Input tensor on HBM.
        numel (int): Number of elements in data.

    Returns:
        out (nl.ndarray): [1, 1], Linf norm scalar on HBM.

    Pseudocode:
        # SPMD core 0 and core 1 each process a partition of data
        for each core:
            accum = 0
            for tile in assigned_tiles:
                tile_data = load(tile)
                accum = max(accum, max(|tile_data|))
            local_max = reduce_partitions(accum, max)

        # Exchange and combine results from both cores
        global_max = sendrecv_reduce(local_max, other_core, max)
    """
    out = nl.ndarray((1, 1), dtype=data.dtype, buffer=nl.shared_hbm)
    accum = nl.zeros((nl.tile_size.pmax, 1), dtype=nl.float32, buffer=nl.sbuf)
    prog_id = _norm_spmd_body(data, numel, nl.abs, nl.max, accum)
    local_total = _norm_cross_core_reduce(accum, nl.max, prog_id)
    nisa.dma_copy(out[0:1, 0:1], local_total)
    return out
