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

# Portions of this file retain the following copyright and license notice:
# BSD 3-Clause License
#
# (c) Meta Platforms, Inc. and affiliates.
#
# Redistribution and use in source and binary forms, with or without modification,
# are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice,this list
# of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice, this
# list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its contributors may
# be used to endorse or promote products derived from this software without specific
# prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
# OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT
# SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED
# TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR
# BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
# ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH
# DAMAGE.

"""Build inverse row indices and expert counts for packed all-to-all buffers."""

import nki.isa as nisa
import nki.language as nl

from ....core.utils.kernel_assert import kernel_assert
from ....core.utils.kernel_helpers import div_ceil

_P_MAX = 128
_FLOAT32_EXACT_INT_LIMIT = 1 << 24


def _build_packed_regroup_indices(
    recv_expert_counts: nl.NkiTensor,
    recv_capacity: int,
    num_programs: int,
    program_id: int,
) -> tuple[nl.NkiTensor, nl.NkiTensor]:
    """Build packed-row layout metadata for multiple local experts.

    Intended for expert-parallel receive counts, including EP64/NLE2, with
    at most 128 sources and 2 to 128 local experts. Output rows are split
    across ``num_programs`` NKI programs; ``program_id`` selects this program's rows.

    Dimensions:
        S: Number of source ranks.
        E: Number of local experts.
        C: Static receive capacity, with 0 < C < 2**24.

    Args:
        recv_expert_counts (nl.NkiTensor): int32/uint32 [S, E] in HBM;
            entry [s, e] counts rows received from source s for local expert e.
            Counts must be nonnegative and their sum must not exceed C.
        recv_capacity (int): C, including unused trailing rows.
        num_programs (int): Number of NKI programs sharing the output rows.
        program_id (int): This program's index in ``[0, num_programs)``.

    Returns:
        regroup_indices (nl.NkiTensor): int32 [C] in shared HBM. Each
            expert-major row maps to its source-major row position. Gather
            source-major rows with these indices, or scatter expert-major rows
            to these indices to restore source-major order.
        bwmm_counts (nl.NkiTensor): int32 [E] in shared HBM. Per-expert row
            totals with unused capacity assigned to the final local expert.
            Individual expert counts are not rounded to a tile boundary.

    Notes:
        Source-major storage orders segments by [source, expert]; grouped
        matrix multiplication uses [expert, source]. Unused tail indices are
        an identity mapping, so the output is a full-capacity permutation.
        Segment starts are scanned in float32, requiring C < 2**24. SBUF
        allocations depend on S * E + 1 segment boundaries and 128-row tiles,
        not on C; the full-capacity output remains in shared HBM.

    Pseudocode:
        Compute source-major segment starts from counts.
        Transpose metadata and scan expert-major segment starts.
        Append the identity tail segment and assign its rows to the last expert.
        For each output tile, locate each row's segment and map its offset
        onto the corresponding source-major segment start.
    """
    num_sources, num_local_experts = recv_expert_counts.shape
    num_segments = num_sources * num_local_experts

    kernel_assert(
        recv_expert_counts.dtype in (nl.int32, nl.uint32),
        f"recv_expert_counts must be int32/uint32, got {recv_expert_counts.dtype}",
    )
    kernel_assert(num_sources <= _P_MAX, f"num_sources must be <= {_P_MAX}, got {num_sources}")
    kernel_assert(
        1 < num_local_experts <= _P_MAX,
        f"num_local_experts must be in [2, {_P_MAX}], got {num_local_experts}",
    )
    kernel_assert(recv_capacity > 0, f"recv_capacity must be positive, got {recv_capacity}")
    kernel_assert(
        recv_capacity < _FLOAT32_EXACT_INT_LIMIT,
        f"recv_capacity must be < {_FLOAT32_EXACT_INT_LIMIT}, got {recv_capacity}",
    )

    counts_raw = nl.ndarray(
        (num_sources, num_local_experts),
        dtype=recv_expert_counts.dtype,
        buffer=nl.sbuf,
    )
    nisa.dma_copy(dst=counts_raw, src=recv_expert_counts)
    counts_f = nl.ndarray(
        (num_sources, num_local_experts),
        dtype=nl.float32,
        buffer=nl.sbuf,
    )
    nisa.tensor_copy(dst=counts_f, src=counts_raw)

    # Source-major input starts:
    #   source_start[s] + sum(counts[s, :e]).
    source_totals = nl.ndarray((num_sources, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_reduce(
        dst=source_totals,
        data=counts_f,
        op=nl.add,
        axis=1,
    )
    source_totals_t_psum = nl.ndarray((1, num_sources), dtype=nl.float32, buffer=nl.psum)
    nisa.nc_transpose(dst=source_totals_t_psum, data=source_totals)
    source_totals_t = nl.ndarray((1, num_sources), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=source_totals_t, src=source_totals_t_psum)

    source_starts_t = nl.ndarray((1, num_sources), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(dst=source_starts_t, value=0.0)
    source_scan_init = nl.ndarray((1, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(dst=source_scan_init, value=0.0)
    source_scan_ones = nl.ndarray((1, num_sources - 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(dst=source_scan_ones, value=1.0)
    nisa.tensor_tensor_scan(
        initial=source_scan_init,
        data0=source_scan_ones,
        op0=nl.multiply,
        data1=source_totals_t[0, : num_sources - 1],
        op1=nl.add,
        dst=source_starts_t[0, 1:],
    )
    source_starts_psum = nl.ndarray((num_sources, 1), dtype=nl.float32, buffer=nl.psum)
    nisa.nc_transpose(dst=source_starts_psum, data=source_starts_t)
    source_starts = nl.ndarray((num_sources, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=source_starts, src=source_starts_psum)

    expert_offsets = nl.ndarray(
        (num_sources, num_local_experts),
        dtype=nl.float32,
        buffer=nl.sbuf,
    )
    nisa.memset(dst=expert_offsets, value=0.0)
    expert_scan_init = nl.ndarray((num_sources, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(dst=expert_scan_init, value=0.0)
    expert_scan_ones = nl.ndarray(
        (num_sources, num_local_experts - 1),
        dtype=nl.float32,
        buffer=nl.sbuf,
    )
    nisa.memset(dst=expert_scan_ones, value=1.0)
    nisa.tensor_tensor_scan(
        initial=expert_scan_init,
        data0=expert_scan_ones,
        op0=nl.multiply,
        data1=counts_f[:, : num_local_experts - 1],
        op1=nl.add,
        dst=expert_offsets[:, 1:],
    )
    input_starts_source_major = nl.ndarray(
        (num_sources, num_local_experts),
        dtype=nl.float32,
        buffer=nl.sbuf,
    )
    nisa.tensor_tensor(
        dst=input_starts_source_major,
        data1=source_starts.broadcast(1, num_local_experts),
        data2=expert_offsets,
        op=nl.add,
    )

    # Transpose the small metadata so segment order is [expert, source].
    counts_expert_major_psum = nl.ndarray(
        (num_local_experts, num_sources),
        dtype=nl.float32,
        buffer=nl.psum,
    )
    nisa.nc_transpose(dst=counts_expert_major_psum, data=counts_f)
    counts_expert_major = nl.ndarray(
        (num_local_experts, num_sources),
        dtype=nl.float32,
        buffer=nl.sbuf,
    )
    nisa.tensor_copy(dst=counts_expert_major, src=counts_expert_major_psum)

    input_starts_expert_major_psum = nl.ndarray(
        (num_local_experts, num_sources),
        dtype=nl.float32,
        buffer=nl.psum,
    )
    nisa.nc_transpose(
        dst=input_starts_expert_major_psum,
        data=input_starts_source_major,
    )
    input_starts_expert_major = nl.ndarray(
        (num_local_experts, num_sources),
        dtype=nl.float32,
        buffer=nl.sbuf,
    )
    nisa.tensor_copy(
        dst=input_starts_expert_major,
        src=input_starts_expert_major_psum,
    )

    # Small private-HBM lookup tables make the tile-time indirect loads simple.
    input_starts_hbm = nl.ndarray((num_segments + 1, 1), dtype=nl.float32, buffer=nl.private_hbm)
    output_starts_hbm = nl.ndarray((num_segments + 1, 1), dtype=nl.float32, buffer=nl.private_hbm)
    expert_prefix = nl.ndarray(
        (num_local_experts, num_sources + 1),
        dtype=nl.float32,
        buffer=nl.sbuf,
    )
    nisa.memset(dst=expert_prefix, value=0.0)
    expert_prefix_init = nl.ndarray((num_local_experts, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(dst=expert_prefix_init, value=0.0)
    expert_prefix_ones = nl.ndarray(
        (num_local_experts, num_sources),
        dtype=nl.float32,
        buffer=nl.sbuf,
    )
    nisa.memset(dst=expert_prefix_ones, value=1.0)
    nisa.tensor_tensor_scan(
        initial=expert_prefix_init,
        data0=expert_prefix_ones,
        op0=nl.multiply,
        data1=counts_expert_major,
        op1=nl.add,
        dst=expert_prefix[:, 1:],
    )

    expert_totals = expert_prefix[:, num_sources : num_sources + 1]
    expert_totals_t_psum = nl.ndarray((1, num_local_experts), dtype=nl.float32, buffer=nl.psum)
    nisa.nc_transpose(dst=expert_totals_t_psum, data=expert_totals)
    expert_totals_t = nl.ndarray((1, num_local_experts), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=expert_totals_t, src=expert_totals_t_psum)

    expert_total_prefix = nl.ndarray(
        (1, num_local_experts + 1),
        dtype=nl.float32,
        buffer=nl.sbuf,
    )
    nisa.memset(dst=expert_total_prefix, value=0.0)
    expert_total_prefix_init = nl.ndarray((1, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(dst=expert_total_prefix_init, value=0.0)
    expert_total_prefix_ones = nl.ndarray((1, num_local_experts), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(dst=expert_total_prefix_ones, value=1.0)
    nisa.tensor_tensor_scan(
        initial=expert_total_prefix_init,
        data0=expert_total_prefix_ones,
        op0=nl.multiply,
        data1=expert_totals_t,
        op1=nl.add,
        dst=expert_total_prefix[0, 1:],
    )
    expert_bases_psum = nl.ndarray((num_local_experts, 1), dtype=nl.float32, buffer=nl.psum)
    nisa.nc_transpose(
        dst=expert_bases_psum,
        data=expert_total_prefix[0:1, :num_local_experts],
    )
    expert_bases = nl.ndarray((num_local_experts, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=expert_bases, src=expert_bases_psum)

    output_starts_expert_major = nl.ndarray(
        (num_local_experts, num_sources),
        dtype=nl.float32,
        buffer=nl.sbuf,
    )
    nisa.tensor_tensor(
        dst=output_starts_expert_major,
        data1=expert_prefix[:, :num_sources],
        data2=expert_bases.broadcast(1, num_sources),
        op=nl.add,
    )
    nisa.dma_copy(
        dst=output_starts_hbm[0:num_segments, 0:1].reshape((num_local_experts, num_sources)),
        src=output_starts_expert_major,
    )
    nisa.dma_copy(
        dst=input_starts_hbm[0:num_segments, 0:1].reshape((num_local_experts, num_sources)),
        src=input_starts_expert_major,
    )

    # The static-capacity tail is an identity segment owned by the last expert.
    active_rows = expert_total_prefix[0:1, num_local_experts : num_local_experts + 1]
    nisa.dma_copy(
        dst=input_starts_hbm[num_segments : num_segments + 1, 0:1],
        src=active_rows,
    )
    nisa.dma_copy(
        dst=output_starts_hbm[num_segments : num_segments + 1, 0:1],
        src=active_rows,
    )
    tail_rows = nl.ndarray((1, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(
        dst=tail_rows,
        data=active_rows,
        op0=nl.subtract,
        operand0=recv_capacity,
        reverse0=True,
    )
    expert_ids = nl.ndarray((num_local_experts, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.iota(
        dst=expert_ids,
        pattern=[[0, 1]],
        offset=0,
        channel_multiplier=1,
    )
    tail_owner = nl.ndarray((num_local_experts, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(
        dst=tail_owner,
        data=expert_ids,
        op0=nl.equal,
        operand0=num_local_experts - 1,
    )
    tail_rows_hbm = nl.ndarray((1, 1), dtype=nl.float32, buffer=nl.private_hbm)
    nisa.dma_copy(dst=tail_rows_hbm, src=tail_rows)
    tail_rows_all = nl.ndarray((num_local_experts, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(
        dst=tail_rows_all,
        src=tail_rows_hbm.broadcast(0, num_local_experts),
    )
    tail_contribution = nl.ndarray((num_local_experts, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_tensor(
        dst=tail_contribution,
        data1=tail_owner,
        data2=tail_rows_all,
        op=nl.multiply,
    )
    bwmm_counts_f = nl.ndarray((num_local_experts, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_tensor(
        dst=bwmm_counts_f,
        data1=expert_totals,
        data2=tail_contribution,
        op=nl.add,
    )

    # CoreBarrier uses the allocation shape, so retain a physical 2D allocation.
    bwmm_counts_2d = nl.ndarray((num_local_experts, 1), dtype=nl.int32, buffer=nl.shared_hbm)
    bwmm_counts_sb = nl.ndarray((num_local_experts, 1), dtype=nl.int32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=bwmm_counts_sb, src=bwmm_counts_f)
    if program_id == 0:
        nisa.dma_copy(
            dst=bwmm_counts_2d,
            src=bwmm_counts_sb,
        )

    regroup_indices_2d = nl.ndarray((recv_capacity, 1), dtype=nl.int32, buffer=nl.shared_hbm)
    input_starts_lookup = input_starts_hbm
    output_starts_lookup = output_starts_hbm
    output_starts_matrix = nl.ndarray(
        (_P_MAX, num_segments + 1),
        dtype=nl.float32,
        buffer=nl.sbuf,
    )
    nisa.dma_copy(
        dst=output_starts_matrix,
        src=output_starts_hbm.reshape((1, num_segments + 1)).broadcast(0, _P_MAX),
    )

    num_tiles = div_ceil(recv_capacity, _P_MAX)
    tiles_per_program = div_ceil(num_tiles, num_programs)
    first_tile = program_id * tiles_per_program
    for tile_offset in nl.sequential_range(tiles_per_program):
        tile_index = min(first_tile + tile_offset, num_tiles - 1)
        if first_tile + tile_offset < num_tiles:
            tile_start = tile_index * _P_MAX
            tile_rows = min(_P_MAX, recv_capacity - tile_start)

            output_rows_matrix = nl.ndarray(
                (_P_MAX, num_segments + 1),
                dtype=nl.float32,
                buffer=nl.sbuf,
            )
            nisa.iota(
                dst=output_rows_matrix,
                pattern=[[0, num_segments + 1]],
                offset=tile_start,
                channel_multiplier=1,
            )
            starts_before_or_at_row = nl.ndarray(
                (_P_MAX, num_segments + 1),
                dtype=nl.float32,
                buffer=nl.sbuf,
            )
            nisa.tensor_tensor(
                dst=starts_before_or_at_row,
                data1=output_rows_matrix,
                data2=output_starts_matrix,
                op=nl.greater_equal,
            )
            segment_ids_f = nl.ndarray((_P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_reduce(
                dst=segment_ids_f,
                data=starts_before_or_at_row,
                op=nl.add,
                axis=1,
            )
            nisa.tensor_scalar(
                dst=segment_ids_f,
                data=segment_ids_f,
                op0=nl.subtract,
                operand0=1.0,
            )
            segment_ids = nl.ndarray((_P_MAX, 1), dtype=nl.uint32, buffer=nl.sbuf)
            nisa.tensor_copy(dst=segment_ids, src=segment_ids_f)

            input_starts = nl.ndarray((_P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.dma_copy(
                dst=input_starts,
                src=input_starts_lookup.ap(
                    pattern=[[1, _P_MAX], [1, 1]],
                    offset=0,
                    vector_offset=segment_ids,
                    indirect_dim=0,
                ),
            )
            output_starts = nl.ndarray((_P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.dma_copy(
                dst=output_starts,
                src=output_starts_lookup.ap(
                    pattern=[[1, _P_MAX], [1, 1]],
                    offset=0,
                    vector_offset=segment_ids,
                    indirect_dim=0,
                ),
            )
            output_rows = nl.ndarray((_P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.iota(
                dst=output_rows,
                pattern=[[0, 1]],
                offset=tile_start,
                channel_multiplier=1,
            )
            segment_offsets = nl.ndarray((_P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_tensor(
                dst=segment_offsets,
                data1=output_rows,
                data2=output_starts,
                op=nl.subtract,
            )
            mapped_rows_f = nl.ndarray((_P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_tensor(
                dst=mapped_rows_f,
                data1=input_starts,
                data2=segment_offsets,
                op=nl.add,
            )
            mapped_rows = nl.ndarray((_P_MAX, 1), dtype=nl.int32, buffer=nl.sbuf)
            nisa.tensor_copy(dst=mapped_rows, src=mapped_rows_f)
            nisa.dma_copy(
                dst=regroup_indices_2d[tile_start : tile_start + tile_rows, 0:1],
                src=mapped_rows[0:tile_rows, 0:1],
            )

    return regroup_indices_2d.reshape((recv_capacity,)), bwmm_counts_2d.reshape((num_local_experts,))
