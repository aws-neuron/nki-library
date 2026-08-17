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

"""Permute routed tokens for MoE all_to_all_v() collective."""

import math

import nki
import nki.isa as nisa
import nki.language as nl

from ...core.utils.kernel_assert import kernel_assert
from .argsort_unstable import argsort_unstable

_SUPPORTED_K = [1, 2, 4, 8]
_SUPPORTED_HIDDEN_DTYPES = [nl.bfloat16, nl.float8_e4m3, nl.float8_e4m3fn, nl.float8_e5m2]
_EXPERT_AFFINITY_COLS = {
    nl.bfloat16: 1,
    nl.float8_e4m3: 2,
    nl.float8_e4m3fn: 2,
    nl.float8_e5m2: 2,
}
_TOKEN_INDEX_COLS = {
    nl.bfloat16: 2,
    nl.float8_e4m3: 4,
    nl.float8_e4m3fn: 4,
    nl.float8_e5m2: 4,
}


@nki.jit
def permute_routed_tokens(
    hidden_input: nl.NkiTensor,
    expert_index: nl.NkiTensor,
    expert_affinities_masked: nl.NkiTensor,
):
    """
    Sort tokens by expert and pack hidden states, affinities, and token indices into a [T*K, n_output_cols] buffer.

    Dimensions:
        T: Number of tokens.
        H: Hidden dimension size.
        n_input_cols: Number of columns in hidden_input. When hidden_input is bf16, n_cols=H.
            When hidden_input is fp8, n_cols contains H and may contain additional columns
            for quantization scales.
        n_concat_cols: Number of columns corresponding to affinities (bf16) and token index
            (int32), when viewed as hidden_input dtype.
        n_output_cols: n_input_cols + n_concat_cols
        K: Top-K experts per token.
        E: Total number of experts.

    Args:
        hidden_input (nl.NkiTensor): [T, n_input_cols], bf16 or fp8 HBM tensor of hidden states.
            When hidden states are fp8, each row contains packed scales.
        expert_index (nl.NkiTensor): [T, K], int32 HBM tensor of top-K expert indices per token.
        expert_affinities_masked (nl.NkiTensor): [T, E], bf16 HBM tensor of expert affinities,
            with zeros for non-routed token/expert pairs.

    Returns:
        output (nl.NkiTensor): [T*K, n_output_cols], bf16 or fp8 HBM tensor where each row is
            [hidden_state, affinity, token_index] sorted by expert index.

    Notes:
        - Requires T*K ≤ 128 (pmax) and K ∈ {1, 2, 4, 8}.

    Pseudocode:
        argsort_indices = argsort(expert_index.flatten(), ascending=True)
        token_indices = argsort_indices // K
        hidden_shuffled = hidden_input[token_indices]
        affinities_shuffled = gather_affinities(expert_affinities_masked, argsort_indices)
        output = concat(hidden_shuffled, affinities_shuffled, token_indices)
        output = reinterpret(output, hidden_input.dtype)
    """

    # Step 1: Extract/validate shapes, allocate buffers
    T, H, K, E = _validate_and_extract_shapes(hidden_input, expert_index, expert_affinities_masked)
    output_shape = (T * K, H + _EXPERT_AFFINITY_COLS[hidden_input.dtype] + _TOKEN_INDEX_COLS[hidden_input.dtype])
    expert_index_flattened_sb = nl.ndarray((1, T * K), dtype=nl.float32, buffer=nl.sbuf)
    argsort_expert_index_psum = nl.ndarray((T * K, 1), dtype=nl.float32, buffer=nl.psum)
    # NOTE that we allocate fp32 to avoid cast during PSUM eviction, and then reinterpret to uint32
    argsort_expert_index_sb = nl.ndarray((T * K, 1), dtype=nl.float32, buffer=nl.sbuf)
    argsort_expert_affinities_sb = nl.ndarray((T * K, 1), dtype=nl.uint32, buffer=nl.sbuf)
    expert_offsets_sb = nl.ndarray((1, T, K), dtype=nl.uint32, buffer=nl.sbuf)
    expert_offsets_hbm = nl.ndarray((1, T * K), dtype=nl.uint32, buffer=nl.shared_hbm, name="arange_TK_stepE_hbm")
    grouped_tokens_affinities_indices_sb = nl.ndarray(output_shape, dtype=hidden_input.dtype, buffer=nl.sbuf)
    grouped_tokens_affinities_indices_hbm = nl.ndarray(output_shape, dtype=hidden_input.dtype, buffer=nl.shared_hbm)

    # Step 2: Compute argsort indices to group T*K tokens by routed expert
    # Step 2.1: Ascending argsort of T*K flattened expert indices
    expert_index_flattened_dim1_hbm = expert_index.reshape((1, T * K))
    nisa.dma_copy(expert_index_flattened_sb, expert_index_flattened_dim1_hbm)
    argsort_expert_index_F_sb = argsort_unstable(data=expert_index_flattened_sb, descending=False, output_in_sbuf=True)

    # Step 2.2: Transpose argsort indices, bitcasting to avoid u32->f32->u32 casts (PE doesn't support u32 transpose)
    nisa.nc_transpose(argsort_expert_index_psum, argsort_expert_index_F_sb.view(nl.float32))
    nisa.tensor_copy(argsort_expert_index_sb, argsort_expert_index_psum)
    argsort_expert_index_sb = argsort_expert_index_sb.view(argsort_expert_index_F_sb.dtype)

    # Step 3: Permute T*K expert affinities, pack into column H of output buffer
    # Step 3.1: Build expert affinities argsort vector using expert index argsort vector
    # TODO: utilize SBUF->SBUF indirection instead of HBM->SBUF indirection.
    nisa.iota(
        pattern=[[E, T], [0, K]],
        offset=0,
        dst=expert_offsets_sb,
    )
    expert_offsets_sb = expert_offsets_sb.reshape((1, T * K))
    nisa.dma_copy(expert_offsets_hbm, expert_offsets_sb)

    # Step 3.2: argsort_expert_affinities = expert_index_flattened[argsort_expert_index] + expert_offsets_flattened[argsort_expert_index]
    expert_offsets_hbm = expert_offsets_hbm.reshape((T * K, 1))
    expert_index_flattened_dim0_hbm = expert_index.reshape((T * K, 1))
    nisa.dma_copy(
        src=expert_index_flattened_dim0_hbm.ap(
            pattern=[[1, T * K], [1, 1]],
            offset=0,
            vector_offset=argsort_expert_index_sb,
            indirect_dim=0,
        ),
        dst=argsort_expert_affinities_sb,
    )
    nisa.dma_compute(
        srcs=[
            expert_offsets_hbm.ap(
                pattern=[[1, T * K], [1, 1]],
                offset=0,
                vector_offset=argsort_expert_index_sb,
                indirect_dim=0,
            ),
            argsort_expert_affinities_sb,
        ],
        reduce_op=nl.add,
        dst=argsort_expert_affinities_sb,
    )

    # Step 3.3: Permute flattened expert affinities using 1D vector DGE, with bitcast to hidden_input dtype.
    # This operation simultaneously permutes T while gathering across E.
    expert_affinities_masked_flattened = expert_affinities_masked.reshape((T * E, 1))
    hbm_ap_step = _EXPERT_AFFINITY_COLS[hidden_input.dtype]
    nisa.dma_copy(
        src=expert_affinities_masked_flattened.ap(
            pattern=[[hbm_ap_step, T * K], [1, hbm_ap_step]],
            offset=0,
            vector_offset=argsort_expert_affinities_sb,
            indirect_dim=0,
            dtype=hidden_input.dtype,
        ),
        dst=grouped_tokens_affinities_indices_sb[:, H : H + _EXPERT_AFFINITY_COLS[hidden_input.dtype]],
    )

    # Step 4: Permute tokens by routed expert with broadcast across K
    # Step 4.1: When K>1, update argsort indices to correspond to T, rather than T*K
    if K > 1:
        argsort_broadcast_token_indices_sb = nl.ndarray((T * K, 1), dtype=nl.uint32, buffer=nl.sbuf)
        nisa.tensor_scalar(
            data=argsort_expert_index_sb,
            op0=nl.right_shift,
            operand0=int(math.log2(K)),
            dst=argsort_broadcast_token_indices_sb,
        )
    else:
        argsort_broadcast_token_indices_sb = argsort_expert_index_sb

    # Step 4.2: Permute and broadcast routed tokens, packing into first H columns of output buffer
    nisa.dma_copy(
        src=hidden_input.ap(
            pattern=[[H, T * K], [1, H]],
            offset=0,
            vector_offset=argsort_broadcast_token_indices_sb,
            indirect_dim=0,
        ),
        dst=grouped_tokens_affinities_indices_sb[:, :H],
    )

    # Step 5: Pack token index, with bitcast to hidden_input dtype
    nisa.tensor_copy(
        src=argsort_broadcast_token_indices_sb.view(hidden_input.dtype),
        dst=grouped_tokens_affinities_indices_sb[:, H + _EXPERT_AFFINITY_COLS[hidden_input.dtype] :],
    )

    # Step 6: Store packed output [hidden_states, affinities, token_index] to HBM
    nisa.dma_copy(grouped_tokens_affinities_indices_hbm, grouped_tokens_affinities_indices_sb)

    return grouped_tokens_affinities_indices_hbm


def _validate_and_extract_shapes(hidden_input, expert_index, expert_affinities_masked):
    """Extract and validate shapes from permute_routed_tokens inputs."""
    T, H = hidden_input.shape
    T_expert, K = expert_index.shape
    T_affinity, E = expert_affinities_masked.shape
    pmax = nl.tile_size.pmax
    kernel_assert(
        hidden_input.dtype in _SUPPORTED_HIDDEN_DTYPES,
        f"Expected hidden_input.dtype in {_SUPPORTED_HIDDEN_DTYPES} but got {hidden_input.dtype=}",
    )
    kernel_assert(T * K <= pmax, f"Expected T * K <= {pmax}, got {T=}, {K=}")
    kernel_assert(K in _SUPPORTED_K, f"Expected K in {_SUPPORTED_K}, got {K=}")
    kernel_assert(
        T == T_expert and T == T_affinity,
        f"Expected hidden_input, expert_index, and expert_affinities_masked to have same dim0,"
        f" but got {T}, {T_expert}, {T_affinity}",
    )
    return T, H, K, E
