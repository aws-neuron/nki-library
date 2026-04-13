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

"""
This file contains an implementation of transpose broadcast, moving a column tensor into every parition of a destination
tensor.

"""

import nki.isa as nisa
import nki.language as nl

from .kernel_assert import kernel_assert
from .tensor_view import TensorView


def tp_broadcast(src, dst, src_offset, psum_address=None):
    """
    Transposes src[:, src_offset], then broadcasts onto all partitions of dst

    Uses a single transpose instruction (on PE) with repeated input access to broadcast src to dst.

    All inputs and outputs to this function are assumed to be in sbuf.
    Uses a psum bank for the transpose, location can be specified `psum_address`.

    Dimensions:
        P: Parition dimension of source (free dimension of dst)
        F: Free dimension of source
        B: Broadcast count in destination (parition dimension)

    Args:
        src: 2D input sbuf tensor or TensorView. Shape: [P, F]
        dst: 2D output sbuf tensor or TensorView. Shape: [B, P]
        src_offset: Specify the column in F to take the data from
        psum_address: Optional psum address to use for location of intermediate transpose
    """
    src_tv = TensorView(src)
    dst_tv = TensorView(dst)
    p_dim, _ = src_tv.shape
    broadcast_dim, tp_dim = dst_tv.shape

    kernel_assert(src_tv.is_sbuf(), "Source must be in sbuf")
    kernel_assert(dst_tv.is_sbuf(), "Destination must be in sbuf")
    kernel_assert(tp_dim == p_dim, "Transposed dim didn't match")

    # Transpose and broadcast into intermediate psum buffer
    tp_psum = nl.ndarray((broadcast_dim, tp_dim), nl.float32, buffer=nl.psum, address=psum_address)

    nisa.nc_transpose(tp_psum, src_tv.slice(1, src_offset, src_offset + 1).broadcast(1, broadcast_dim).get_view())

    # Copy back to sbuf
    nisa.tensor_copy(dst_tv.get_view(), src=tp_psum)
