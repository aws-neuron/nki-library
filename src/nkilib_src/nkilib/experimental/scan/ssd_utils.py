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

"""Shared utilities for SSD kernel implementations."""

import nki.isa as nisa
import nki.language as nl

from ...core.utils.kernel_helpers import get_program_sharding_info
from ...core.utils.stream_shuffle_broadcast import stream_shuffle_broadcast


def broadcast_scalar_to_column(scalar: nl.ndarray, num_rows: int) -> nl.ndarray:
    """Broadcast a [1, 1] SBUF scalar to a [num_rows, 1] column vector.

    Args:
        scalar (nl.ndarray): [1, 1], SBUF tensor containing the scalar value.
        num_rows (int): Number of rows in the output column vector.

    Returns:
        nl.ndarray: [num_rows, 1], SBUF tensor with the scalar broadcast across all rows.
    """
    result = nl.ndarray((num_rows, 1), dtype=nl.float32, buffer=nl.sbuf)
    stream_shuffle_broadcast(src=scalar, dst=result)
    return result


def compute_lnc_sharding(nheads: int) -> tuple:
    """Compute LNC round-robin head distribution across NeuronCores.

    Args:
        nheads (int): Total number of heads to distribute.

    Returns:
        tuple: (heads_per_core, head_offset)
            - heads_per_core: Number of heads assigned to this core.
            - head_offset: Global head index offset for this core.
    """
    _, num_programs, program_id = get_program_sharding_info()
    heads_per_core = nheads // num_programs + (1 if program_id < nheads % num_programs else 0)
    head_offset = (nheads // num_programs) * program_id + min(program_id, nheads % num_programs)
    return heads_per_core, head_offset


def transpose_row_to_column(row: nl.ndarray, seq_len: int) -> nl.ndarray:
    """Transpose a [1, seq_len] row to a [seq_len, 1] column via PSUM nc_transpose.

    Args:
        row (nl.ndarray): [1, seq_len], SBUF tensor (row in free dimension).
        seq_len (int): Length of the row / number of elements.

    Returns:
        nl.ndarray: [seq_len, 1], SBUF tensor (column in partition dimension).
    """
    transposed_psum = nl.ndarray((seq_len, 1), dtype=nl.float32, buffer=nl.psum)
    nisa.nc_transpose(
        dst=transposed_psum[0:seq_len, 0:1],
        data=row[0:1, 0:seq_len],
    )
    column = nl.ndarray((seq_len, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=column[0:seq_len, 0:1], src=transposed_psum[0:seq_len, 0:1])
    return column


def compute_cumulative_decay(
    dt_row: nl.ndarray,
    A_scalar: nl.ndarray,
    ones_row: nl.ndarray,
    zero_scalar: nl.ndarray,
    chunk_size: int,
) -> tuple:
    """Compute cumulative decay cs = cumsum(dt * A) and exp(cs), exp(-cs).

    Args:
        dt_row (nl.ndarray): [1, Q], Softplus'd timestep for one head.
        A_scalar (nl.ndarray): [1, 1], State transition scalar (negative).
        ones_row (nl.ndarray): [1, Q], Ones buffer for scan.
        zero_scalar (nl.ndarray): [1, 1], Zero buffer for scan initial value.
        chunk_size (int): Chunk size Q.

    Returns:
        tuple: (cs_row, exp_cs_col, exp_neg_cs_col)
            - cs_row (nl.ndarray): [1, Q], Cumulative sum in free dimension.
            - exp_cs_col (nl.ndarray): [Q, 1], exp(cs) in partition dimension.
            - exp_neg_cs_col (nl.ndarray): [Q, 1], exp(-cs) in partition dimension.
    """
    Q = chunk_size
    log_decay = nl.ndarray((1, Q), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(
        dst=log_decay[0:1, 0:Q],
        data=dt_row[0:1, 0:Q],
        op0=nl.multiply,
        operand0=A_scalar[0:1, 0:1],
    )
    cs_row = nl.ndarray((1, Q), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_tensor_scan(
        dst=cs_row[0:1, 0:Q],
        data0=ones_row[0:1, 0:Q],
        data1=log_decay[0:1, 0:Q],
        initial=zero_scalar[0:1, 0:1],
        op0=nl.multiply,
        op1=nl.add,
    )
    cs_col = transpose_row_to_column(cs_row, Q)
    exp_cs_col = nl.ndarray((Q, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.activation(op=nl.exp, data=cs_col[0:Q, 0:1], dst=exp_cs_col[0:Q, 0:1])
    exp_neg_cs_col = nl.ndarray((Q, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.activation(op=nl.reciprocal, data=exp_cs_col[0:Q, 0:1], dst=exp_neg_cs_col[0:Q, 0:1])
    return cs_row, exp_cs_col, exp_neg_cs_col
