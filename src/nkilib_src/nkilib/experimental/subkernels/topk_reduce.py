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

"""MoE Top-K reduction across sparse all_to_all_v() collective output buffer."""

import nki
import nki.isa as nisa
import nki.language as nl

from ...core.utils.kernel_assert import kernel_assert
from ...core.utils.kernel_helpers import get_verified_program_sharding_info
from ...core.utils.stream_shuffle_broadcast import stream_shuffle_broadcast

_K_MAX = 8
_N_16BIT_ELEM_PER_INT32 = 2
_SUPPORTED_INPUT_DTYPES = [nl.bfloat16, nl.float16]


@nki.jit
def topk_reduce(
    input: nl.ndarray,
    T: int,
    K: int,
):
    """
    Compute MoE Top-K reduction across sparse all_to_all_v() collective output buffer.

    Gathers scattered rows by packed global token index and reduces along
    the K dimension. Supports LNC sharding on the H dimension.

    Dimensions:
        TK_padded: n_src_ranks * T, padded input row count
        H: Hidden dimension size (must be divisible by LNC)
        T: Total number of input tokens (up to 128)
        K: Number of routed experts per token (up to 8)

    Args:
        input (nl.ndarray): [TK_padded, H + 2]@HBM, bf16/fp16. Sparse input buffer containing T*K
            scattered outputs. Global token index is packed as int32 in the final 2x
            columns of each row.
        T (int): Total number of input tokens.
        K (int): Number of routed experts per token.

    Returns:
        output_hbm (nl.ndarray): [T, H]@HBM, bf16/fp16. Ordered and reduced output.

    Pseudocode:
        global_token_indices = extract_int32_index(input[:, H:])
        for token_idx in range(T):
            matching_rows = find_rows_where(global_token_indices == token_idx)
            output[token_idx] = sum(input[matching_rows, :H])
    """

    # Shapes, LNC sharding strategy
    _P_MAX = nl.tile_size.pmax
    TK_padded, H_padded = input.shape
    H = H_padded - _N_16BIT_ELEM_PER_INT32
    _, n_prgs, prg_id = get_verified_program_sharding_info("topk_reduce", (0, 1))
    H_local = H // n_prgs
    H_local_slice = nl.ds(H_local * prg_id, H_local)

    # Validation
    kernel_assert(
        input.dtype in _SUPPORTED_INPUT_DTYPES, f"Expected input.dtype in {_SUPPORTED_INPUT_DTYPES}, got {input.dtype=}"
    )
    kernel_assert(T <= _P_MAX, f"T must be <= {_P_MAX}")
    kernel_assert(K <= _K_MAX, f"K must be <= {_K_MAX}")
    kernel_assert(H % n_prgs == 0, f"Expected H divisible by LNC, got {H=} {n_prgs=}")

    # Allocations
    reduced_sb = nl.ndarray((T, H_local), dtype=input.dtype, buffer=nl.sbuf)
    global_token_indices_sb = nl.ndarray((T, TK_padded), dtype=nl.int32, buffer=nl.sbuf)
    output_hbm = nl.ndarray((T, H), dtype=input.dtype, buffer=nl.shared_hbm)

    # DMA transpose indices [TK_padded, 1] -> [1, TK_padded]
    nisa.dma_transpose(
        src=input.ap(
            pattern=[[H_padded // _N_16BIT_ELEM_PER_INT32, TK_padded], [1, 1], [1, 1], [1, 1]],
            offset=H // _N_16BIT_ELEM_PER_INT32,
            dtype=nl.int32,
        ),
        dst=global_token_indices_sb.ap(
            pattern=[[TK_padded, 1], [1, 1], [1, 1], [1, TK_padded]],
            offset=0,
        ),
    )

    # Broadcast [1, TK_padded] -> [T, TK_padded]
    # FIXME: (1) Move broadcast to DMA engines (2) LNC shard on tokens when T>32
    stream_shuffle_broadcast(global_token_indices_sb, global_token_indices_sb)

    # Find indices [T, K]
    arange_token_indices_T = nl.ndarray((T, _K_MAX), dtype=nl.uint32, buffer=nl.sbuf)
    gather_token_indices = nl.ndarray((T, _K_MAX), dtype=nl.uint32, buffer=nl.sbuf)
    nisa.iota(
        pattern=[[0, _K_MAX]],
        offset=0,
        channel_multiplier=1,
        dst=arange_token_indices_T,
    )
    nisa.nc_find_index8(
        data=global_token_indices_sb,
        vals=arange_token_indices_T,
        dst=gather_token_indices,
    )

    # Use DMA + rmw add to reduce over topK
    for k_idx in range(K):
        src_access = input.ap(
            pattern=[[H, T], [1, H_local]],
            offset=H_local * prg_id,
            vector_offset=gather_token_indices.ap(
                pattern=[[_K_MAX, T], [1, 1]],
                offset=k_idx,
            ),
            indirect_dim=0,
        )

        if k_idx == 0:
            nisa.dma_copy(
                dst=reduced_sb[:, :],
                src=src_access,
            )
        else:
            nisa.dma_compute(
                dst=reduced_sb[:, :],
                srcs=[src_access, reduced_sb[:, :]],
                reduce_op=nl.add,
                unique_indices=True,
            )

    # Save reduced output — each core writes its H shard
    nisa.dma_copy(output_hbm[:, H_local_slice], reduced_sb)

    return output_hbm
