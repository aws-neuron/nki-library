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
import nki
import nki.isa as nisa
import nki.language as nl

from ....core.utils.kernel_assert import kernel_assert
from ....core.utils.ve_transpose_broadcast import ve_transpose_broadcast
from .a2av_train_utils import _exclusive_cumsum_u32


@nki.jit
def build_packed_routing(ids, counts):
    """Build stable destination-major source indices without an [N, EP] HBM table.

    ``ids[N]`` contains peer IDs and ``counts[1, EP]`` their exact histogram.
    The output is a permutation of ``range(N)``, grouped by peer while retaining
    source order within each peer. A row's packed position is its peer's starting
    offset plus the number of earlier rows assigned to that peer.
    Each tile scans at most 128 rows in SBUF;
    per-peer cursors carry the prefix across tiles. Float32 prefix arithmetic is
    exact up to 2**24 rows. Run with LNC=2, with core zero owning the cursors.
    """
    n = ids.shape[0]
    ep = counts.shape[1]
    kernel_assert(0 < n <= 2**24, "Packed routing requires 1 <= N <= 2**24")
    kernel_assert(1 < ep <= 128, "Packed routing requires 2 <= EP <= 128")
    kernel_assert(counts.shape[0] == 1, "counts must have shape [1, EP]")
    kernel_assert(nl.num_programs(axes=0) == 2, "Packed routing requires LNC=2")
    output = nl.ndarray((n, 1), dtype=nl.int32, buffer=nl.shared_hbm)
    if nl.program_id(axis=0) == 0:
        # Initialize each peer's cursor at its packed segment's starting offset.
        _, offsets = _exclusive_cumsum_u32(counts, ep)
        offsets_f = nl.ndarray((1, ep), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=offsets_f, src=offsets)
        initial_psum = nl.ndarray((ep, 1), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_transpose(dst=initial_psum, data=offsets_f)
        initial = nl.ndarray((ep, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=initial, src=initial_psum)
        peers = nl.ndarray((ep, 1), dtype=nl.float32, buffer=nl.sbuf)
        peer_row = nl.ndarray((1, ep), dtype=nl.float32, buffer=nl.sbuf)
        nisa.iota(dst=peer_row, pattern=[[1, ep]], offset=0)
        peer_psum = nl.ndarray((ep, 1), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_transpose(dst=peer_psum, data=peer_row)
        nisa.tensor_copy(dst=peers, src=peer_psum)
        for tile in nl.affine_range((n + 127) // 128):
            start = tile * 128
            rows = min(128, n - start)
            expert_col = nl.ndarray((rows, 1), dtype=nl.float32, buffer=nl.sbuf)
            expert_int = nl.ndarray((rows, 1), dtype=nl.int32, buffer=nl.sbuf)
            nisa.dma_copy(dst=expert_int, src=ids[start : start + rows].reshape((rows, 1)))
            nisa.tensor_copy(dst=expert_col, src=expert_int)
            # Build only this tile's [EP, rows] membership mask in SBUF.
            experts = nl.ndarray((ep, rows), dtype=nl.float32, buffer=nl.sbuf)
            ve_transpose_broadcast(expert_col, experts, rows, ep)
            mask = nl.ndarray((ep, rows), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_scalar(dst=mask, data=experts, op0=nl.equal, operand0=peers)
            ones = nl.ndarray((ep, rows), dtype=nl.float32, buffer=nl.sbuf)
            nisa.memset(ones, 1.0)
            prefix = nl.ndarray((ep, rows), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_tensor_scan(dst=prefix, initial=initial, data0=ones, op0=nl.multiply, data1=mask, op1=nl.add)
            # Select each row's peer cursor; the inclusive scan is one-based.
            selected = nl.ndarray((ep, rows), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_tensor(dst=selected, data1=prefix, data2=mask, op=nl.multiply)
            selected_psum = nl.ndarray((rows, ep), dtype=nl.float32, buffer=nl.psum)
            nisa.nc_transpose(dst=selected_psum, data=selected)
            selected_t = nl.ndarray((rows, ep), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_copy(dst=selected_t, src=selected_psum)
            positions_f = nl.ndarray((rows, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_reduce(dst=positions_f, data=selected_t, op=nl.add, axis=1)
            positions = nl.ndarray((rows, 1), dtype=nl.uint32, buffer=nl.sbuf)
            nisa.tensor_scalar(dst=positions, data=positions_f, op0=nl.subtract, operand0=1)
            source_rows = nl.ndarray((rows, 1), dtype=nl.int32, buffer=nl.sbuf)
            source_row = nl.ndarray((1, rows), dtype=nl.float32, buffer=nl.sbuf)
            nisa.iota(dst=source_row, pattern=[[1, rows]], offset=start)
            source_psum = nl.ndarray((rows, 1), dtype=nl.float32, buffer=nl.psum)
            nisa.nc_transpose(dst=source_psum, data=source_row)
            nisa.tensor_copy(dst=source_rows, src=source_psum)
            # Each row owns one unique packed slot. Carry cursors to the next tile.
            nisa.dma_copy(
                dst=output.ap(pattern=[[n, rows], [1, 1]], offset=0, vector_offset=positions, indirect_dim=0),
                src=source_rows,
            )
            nisa.tensor_copy(dst=initial, src=prefix[:, rows - 1 : rows])
    nisa.core_barrier(output, (0, 1))
    return output.reshape((n,))
