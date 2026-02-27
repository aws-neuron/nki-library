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

"""Utility functions for max reduction operations including folded loading and reduction helpers."""

import math
from typing import List, Optional

import nki.isa as nisa
import nki.language as nl

from ..utils.kernel_assert import kernel_assert


def reduce(op: str = 'mul', input_list: Optional[List] = None, initial_value=None):
    """
    Apply reduction operation over a list of values.

    Args:
        op (str): Operation to apply ('mul', 'add', 'max', 'min')
        input_list (Optional[List]): List of values to reduce
        initial_value: Starting value for reduction

    Returns:
        Reduced value after applying operation

    Notes:
        - Supports multiplication, addition, max, and min operations
        - Requires both input_list and initial_value to be set
    """
    supported_ops = ['mul', 'add', 'max', 'min']
    kernel_assert(initial_value is not None, "initial_value must be set")
    kernel_assert(input_list is not None, "input_list must be set")
    kernel_assert(op in supported_ops, f"only ops in {supported_ops} are supported, got {op}")
    for element in input_list:
        if op == 'mul':
            initial_value = initial_value * element
        elif op == 'add':
            initial_value = initial_value + element
        elif op == 'min':
            initial_value = min(initial_value, element)
        elif op == 'max':
            initial_value = max(initial_value, element)
    return initial_value


def predicated_folded_load(
    data_hbm: nl.ndarray,
    fold_factor: int,
    program_id: int = 0,
    n_programs: int = 1,
    fill_value: float = -9948.0,
    data_sb: Optional[nl.ndarray] = None,
) -> Optional[nl.ndarray]:
    """
    Reshape and load HBM tensor with folding along free dimension into SBUF.

    Loads a 2D HBM tensor [b, n] into SBUF with shape [b * fold_factor, n_folded]
    where n_folded = ceil(n / fold_factor). Handles padding and LNC sharding.

    Args:
        data_hbm (nl.ndarray): [b, n], Input tensor in HBM
        fold_factor (int): Number of folds to apply to free dimension
        program_id (int): Current program ID for LNC sharding (default: 0)
        n_programs (int): Total number of programs (default: 1)
        fill_value (float): Value to use for padding (default: -9948.0)
        data_sb (Optional[nl.ndarray]): Pre-allocated SBUF buffer (default: None)

    Returns:
        Optional[nl.ndarray]: [b_local * fold_factor, n_folded], Folded tensor in SBUF
            Returns None if data_sb is provided (in-place operation)

    Notes:
        - Each core processes b_local = ceil(b / n_programs) rows
        - Padding applied when n not divisible by fold_factor
        - Uses fast path with reshape when n divisible by fold_factor
        - Uses predicated load for non-divisible case
        - Validates that b_local * fold_factor <= partition dimension limit
    """
    kernel_assert(len(data_hbm.shape) == 2, "Expected input tensor to have shape [B, N]")
    batch_size, n = data_hbm.shape

    batch_size_sharded = (batch_size + n_programs - 1) // n_programs
    batch_line_offset = program_id * batch_size_sharded

    P_MAX = nl.tile_size.pmax
    kernel_assert(
        batch_size_sharded * fold_factor <= P_MAX,
        f"fold_factor x max local batch size ({batch_size_sharded}) exceeds tile limit {P_MAX}",
    )

    n_folded = math.ceil(n / fold_factor)

    return_out = False
    if data_sb is None:
        return_out = True
        data_sb = nl.ndarray((batch_size_sharded * fold_factor, n_folded), dtype=data_hbm.dtype, buffer=nl.sbuf)
    nisa.memset(data_sb, fill_value)

    if n == fold_factor * n_folded:
        src_hbm_reshape = data_hbm.reshape((batch_size * fold_factor, n_folded))

        base_offset = batch_line_offset * fold_factor
        batch_bound = min(batch_size_sharded * fold_factor, batch_size * fold_factor - base_offset)
        ix, iy = nl.ds(0, batch_bound), nl.ds(0, n_folded)
        ix_src = nl.ds(base_offset, batch_bound)
        nisa.dma_copy(src=src_hbm_reshape[ix_src, iy], dst=data_sb[ix, iy])
        if return_out:
            return data_sb
        else:
            return None

    src_hbm_flat = data_hbm.reshape((batch_size * n,))

    batch_size_sharded_bounded = min(batch_size_sharded, batch_size - batch_line_offset)
    for batch_line_idx in nl.affine_range(batch_size_sharded_bounded):
        """
        Split load into 2 rectangular loads:
        - [fold_factor x (n % n_folded)]
        - [fold_factor-1 x (n_folded - n % n_folded)]
        """
        row_idx = batch_line_idx + batch_line_offset
        base_idx = row_idx * n
        remainder = n % n_folded
        ix_0_dst, iy_0_dst = nl.ds(batch_line_idx * fold_factor, fold_factor), nl.ds(0, remainder)
        ix_1_dst, iy_1_dst = (
            nl.ds(batch_line_idx * fold_factor, fold_factor - 1),
            nl.ds(remainder, n_folded - remainder),
        )
        src_ap_0 = [[n_folded, fold_factor], [1, remainder]]
        src_ap_1 = [[n_folded, fold_factor - 1], [1, (n_folded - remainder)]]
        nisa.dma_copy(dst=data_sb[ix_0_dst, iy_0_dst], src=src_hbm_flat.ap(pattern=src_ap_0, offset=base_idx))
        nisa.dma_copy(
            dst=data_sb[ix_1_dst, iy_1_dst], src=src_hbm_flat.ap(pattern=src_ap_1, offset=base_idx + remainder)
        )
    if return_out:
        return data_sb
    return None


def unfolded_store(
    sbuf: nl.ndarray,
    data_hbm: nl.ndarray,
    fold_factor: int,
    program_id: int = 0,
    n_programs: int = 1,
) -> None:
    """
    Store local SBUF tensor back to HBM, reversing predicated_folded_load.

    Stores a local sbuf tensor [B_local * fold_factor, n_folded] back into the
    corresponding shard of a global HBM tensor [B, N].

    Args:
        sbuf (nl.ndarray): [B_local * fold_factor, n_folded], Local buffer in SBUF
        data_hbm (nl.ndarray): [B, N], Global HBM tensor
        fold_factor (int): Number of folds applied during load
        program_id (int): ID of current core/program (default: 0)
        n_programs (int): Total number of programs/cores (default: 1)

    Notes:
        - Reverses the folding operation from predicated_folded_load
        - Uses fast path when n divisible by fold_factor
        - Uses predicated store for non-divisible case
        - Handles LNC sharding with proper offset calculation
    """
    kernel_assert(len(data_hbm.shape) == 2, "Expected HBM tensor shape [B, N]")
    batch_size, n = data_hbm.shape
    n_folded = sbuf.shape[1]

    batch_size_sharded = (batch_size + n_programs - 1) // n_programs
    batch_line_offset = program_id * batch_size_sharded

    if n == fold_factor * n_folded:
        reshaped_dst = data_hbm.reshape((batch_size * fold_factor, n_folded))
        base_offset = batch_line_offset * fold_factor
        batch_bound = min(batch_size_sharded * fold_factor, batch_size * fold_factor - base_offset)
        ix, iy = nl.ds(0, batch_bound), nl.ds(0, n_folded)
        ix_dst = nl.ds(base_offset, batch_bound)
        nisa.dma_copy(dst=reshaped_dst[ix_dst, iy], src=sbuf[ix, iy])
        return

    data_hbm_flat = data_hbm.reshape((batch_size * n,))

    batch_size_sharded_bounded = min(batch_size_sharded, batch_size - batch_line_offset)
    for batch_line_idx in nl.affine_range(batch_size_sharded_bounded):
        """
        Split save into 2 rectangular stores:
        - [fold_factor x (n % n_folded)]
        - [fold_factor-1 x (n_folded - n % n_folded)]
        """
        row_idx = batch_line_idx + batch_line_offset
        base_idx = row_idx * n
        remainder = n % n_folded
        ix_0_src, iy_0_src = nl.ds(batch_line_idx * fold_factor, fold_factor), nl.ds(0, remainder)
        ix_1_src, iy_1_src = (
            nl.ds(batch_line_idx * fold_factor, fold_factor - 1),
            nl.ds(remainder, n_folded - remainder),
        )
        dst_ap_0 = [[n_folded, fold_factor], [1, remainder]]
        dst_ap_1 = [[n_folded, fold_factor - 1], [1, (n_folded - remainder)]]
        nisa.dma_copy(src=sbuf[ix_0_src, iy_0_src], dst=data_hbm_flat.ap(pattern=dst_ap_0, offset=base_idx))
        nisa.dma_copy(src=sbuf[ix_1_src, iy_1_src], dst=data_hbm_flat.ap(pattern=dst_ap_1, offset=base_idx + remainder))
