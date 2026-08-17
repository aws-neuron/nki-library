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

"""On-device block-diagonal cross-partition circular shift matrix builder."""

import nki.isa as nisa
import nki.language as nl


def build_rotation_matrix(
    block_size: int,
    num_blocks: int,
    shifts: int = 1,
    dtype=nl.float16,
    negate: bool = False,
    engine=nisa.vector_engine,
) -> nl.ndarray:
    """Build a block-diagonal circulant shift matrix in SBUF.

    Constructs an NxN matrix (N = block_size x num_blocks) composed of
    independent circulant blocks along the diagonal. When used as the
    stationary operand in nc_matmul, each block cyclically shifts data
    across partitions while the free dimension passes through unchanged.

    Setting negate=True flips the sign of the wrapped segment, producing
    a signed permutation (entries are 0, +1, or -1) which can be used
    for RoPE.

    Built with 2 instructions reading from the shared identity matrix —
    one per segment of the shift.

    Args:
        block_size: Size of each circulant block.
        num_blocks: Number of diagonal blocks.
        shifts: Number of positions to shift.
        dtype: Data type for matrix (e.g. nl.float16).
        negate: If True, negate the wrapped segment (for RoPE rotate-half).
        engine: Engine for copy/scalar ops (nisa.vector_engine or nisa.scalar_engine).

    Returns:
        [N, N] SBUF tensor where N = block_size * num_blocks.

    Example (block_size=4, num_blocks=1, shifts=2, negate=False)::

        matrix = [[0, 0, 1, 0],    # row 0: from col (0-2)%4 = 2
                  [0, 0, 0, 1],    # row 1: from col (1-2)%4 = 3
                  [1, 0, 0, 0],    # row 2: from col (2-2)%4 = 0
                  [0, 1, 0, 0]]    # row 3: from col (3-2)%4 = 1

    Example (block_size=4, num_blocks=1, shifts=2, negate=True)::

        matrix = [[ 0, 0,-1, 0],    # row 0: wrapped, negated
                  [ 0, 0, 0,-1],    # row 1: wrapped, negated
                  [ 1, 0, 0, 0],    # row 2: straight, positive
                  [ 0, 1, 0, 0]]    # row 3: straight, positive

    RoPE usage (rotate-half via Tensor engine)::

        # Llama (d_head=128)
        rot = build_rotation_matrix(128, 1, shifts=64, dtype=nl.float16, negate=True)
        psum = nc_matmul(psum, stationary=rot, moving=x)  # rotate_half(x)

        # GPT-OSS (2x d_head=64):
        rot = build_rotation_matrix(64, 2, shifts=32, dtype=nl.float16, negate=True)
    """
    N = block_size * num_blocks
    # nc_matmul applies stationary.T, so we build the inverse shift
    # so that stationary.T produces the desired forward shift.
    shifts = (-shifts) % block_size

    ident = nl.shared_identity_matrix(N, dtype=dtype)
    matrix = nl.ndarray((N, N), dtype=dtype, buffer=nl.sbuf)
    partition_stride = ident.get_pattern()[0][0]

    # Split into two contiguous segments:
    #   "straight" segment: rows [0:tail] — elements that didn't wrap
    #   "wrapped" segment:  rows [tail:block_size] — elements that wrapped around
    tail = block_size - shifts

    if tail > 0:
        # Straight segment (rows 0..tail-1)
        # After the internal shift negation, this corresponds to the user's "wrapped" portion.
        if negate:
            nisa.tensor_scalar(
                dst=matrix.reshape((N, num_blocks, block_size))[:, :, 0:tail],
                data=ident.ap(
                    pattern=[[partition_stride, N], [block_size, num_blocks], [1, tail]],
                    offset=shifts,
                ),
                op0=nl.multiply,
                operand0=-1.0,
                engine=engine,
            )
        else:
            nisa.tensor_copy(
                dst=matrix.reshape((N, num_blocks, block_size))[:, :, 0:tail],
                src=ident.ap(
                    pattern=[[partition_stride, N], [block_size, num_blocks], [1, tail]],
                    offset=shifts,
                ),
                engine=engine,
            )

    if shifts > 0:
        # Wrapped segment (rows tail..block_size-1)
        nisa.tensor_copy(
            dst=matrix.reshape((N, num_blocks, block_size))[:, :, tail:block_size],
            src=ident.ap(
                pattern=[[partition_stride, N], [block_size, num_blocks], [1, shifts]],
                offset=0,
            ),
            engine=engine,
        )

    return matrix
