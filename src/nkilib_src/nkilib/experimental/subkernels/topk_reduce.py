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
from nki.isa import oob_mode

from ...core.utils.kernel_assert import kernel_assert
from ...core.utils.kernel_helpers import get_verified_program_sharding_info
from ...core.utils.stream_shuffle_broadcast import stream_shuffle_broadcast

_K_MAX = 8
_TK_PADDED_MIN = 8
_TK_PADDED_MAX = 16 * 1024
_N_16BIT_ELEM_PER_INT32 = 2
_SUPPORTED_INPUT_DTYPES = [nl.bfloat16, nl.float16]


@nki.jit
def topk_reduce(
    input: nl.NkiTensor,
    T: int,
    K: int,
    token_base_index: int = 1,
):
    """
    Compute MoE Top-K reduction across sparse all_to_all_v() collective output buffer.

    Gathers scattered rows by packed global token index and reduces along
    the K dimension. Supports LNC sharding on the H dimension.

    Token indices are 1-indexed (token 0 → index 1, token 1 → index 2, etc.),
    and padded rows must have index -1.

    When sequence_parallel_rank_id is provided, searches for global token indices
    [rank_id*T+1 .. rank_id*T+T] instead of [1..T].

    Dimensions:
        TK_padded: n_src_ranks * T, padded input row count
        H: Hidden dimension size (must be divisible by LNC)
        T: Total number of input tokens (up to 128)
        K: Number of routed experts per token (up to 8)

    Args:
        input (nl.NkiTensor): [TK_padded, H + 2]@HBM, bf16/fp16. Sparse input buffer containing T*K
            scattered outputs. Global token index is packed as int32 in the final 2x
            columns of each row (1-indexed, -1 for padding).
        T (int): Total number of input tokens.
        K (int): Number of routed experts per token.
        token_base_index (int): First token index to search for (default: 1).
            For sequence parallel mode, use rank_id * T + 1 so the kernel
            searches for global indices [rank_id*T+1 .. rank_id*T+T].

    Returns:
        output_hbm (nl.NkiTensor): [T, H]@HBM, bf16/fp16. Ordered and reduced output.
            out[t] = sum of all rows with index (token_base_index + t).
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
    kernel_assert(1 < T <= _P_MAX, f"T must be greater than 1 and <= {_P_MAX}, got {T=}")
    kernel_assert(K <= _K_MAX, f"K must be <= {_K_MAX}")
    kernel_assert(H % n_prgs == 0, f"Expected H divisible by LNC, got {H=} {n_prgs=}")
    kernel_assert(
        _TK_PADDED_MIN <= TK_padded <= _TK_PADDED_MAX,
        f"Expected input.shape[0] between {_TK_PADDED_MIN} and {_TK_PADDED_MAX}, got {input.shape=}",
    )

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
    # For each token, there may be between 1 and K corresponding rows in the input.
    arange_token_indices_T = nl.ndarray((T, _K_MAX), dtype=nl.uint32, buffer=nl.sbuf)
    gather_token_indices = nl.ndarray((T, _K_MAX), dtype=nl.uint32, buffer=nl.sbuf)
    nisa.memset(gather_token_indices, -1)

    # Generate search values [token_base_index .. token_base_index+T-1]
    nisa.iota(
        pattern=[[0, _K_MAX]],
        offset=token_base_index,
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

        # If src does not contain at least one row for each token, we throw an OOB error.
        if k_idx == 0:
            nisa.dma_copy(
                dst=reduced_sb[:, :],
                src=src_access,
                oob_mode=oob_mode.error,
            )

        # Tokens routed to experts on the same EP rank will have fewer than K rows, so we DMA skip for k_idx=1...K-1.
        # Ex: K=2, E=8, EP=4, token 0 is routed to experts {0, 1} -> 1 row for token 0 in input.
        else:
            nisa.dma_compute(
                dst=reduced_sb[:, :],
                srcs=[src_access, reduced_sb[:, :]],
                reduce_op=nl.add,
                unique_indices=True,
                oob_mode=oob_mode.skip,
            )

    # Save reduced output — each core writes its H shard
    nisa.dma_copy(output_hbm[:, H_local_slice], reduced_sb)

    return output_hbm
