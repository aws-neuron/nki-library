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
Expert affinity masking utilities for all-expert MoE kernels.

This module provides functions to mask expert affinities based on rank_id and expert_index,
enabling unified API for all-expert and selective-expert MoE implementations.
"""

import nki.isa as nisa
import nki.language as nl
from nki.isa.constants import dge_mode

from ...utils.kernel_helpers import div_ceil


def mask_expert_affinities(
    expert_affinities: nl.ndarray,
    expert_index: nl.ndarray,
    rank_id: nl.ndarray,
    E_L: int,
    T: int,
    K: int,
    io_dtype,
    mask_unselected_experts: bool,
) -> nl.ndarray:
    """
    Mask expert affinities for all-expert MoE computation.

    Slices the global expert affinities to local experts based on rank_id, and optionally
    masks affinities by checking expert_index against each local expert (when mask_unselected_experts=True).

    Args:
        expert_affinities (nl.ndarray): [T, E], Global expert affinities in HBM.
        expert_index (nl.ndarray): [T, K], Top-K expert indices per token in SBUF.
        rank_id (nl.ndarray): [1, 1], Rank ID tensor in HBM, specifies which experts this rank processes.
        E_L (int): Number of local experts.
        T (int): Number of tokens.
        K (int): Top-K value.
        io_dtype: Data type for computation.
        mask_unselected_experts (bool): If True, apply masking based on expert_index.

    Returns:
        expert_affinities_masked (nl.ndarray): [T, E_L], Masked affinities in SBUF.
    """
    T_32s = 32 * div_ceil(T, 32)
    E = expert_affinities.shape[1]  # Total number of experts

    # Load rank_id to SBUF
    rank_id_sbuf = nl.ndarray((1, 1), dtype=nl.uint32, buffer=nl.sbuf)
    nisa.dma_copy(src=rank_id[0:1, 0:1], dst=rank_id_sbuf[0:1, 0:1])

    # Calculate expert offset: expert_offset = rank_id * E_L
    expert_offset_sbuf = nl.ndarray((1, 1), dtype=nl.uint32, buffer=nl.sbuf)
    nisa.tensor_scalar(
        dst=expert_offset_sbuf[0:1, 0:1],
        data=rank_id_sbuf[0:1, 0:1],
        op0=nl.multiply,
        operand0=E_L,
    )

    # Allocate masked affinities buffer and load with indirect DMA
    if T <= 128:
        # 2D output [T, E_L]
        expert_affinities_masked = nl.ndarray((T, E_L), dtype=io_dtype, buffer=nl.sbuf)
        nisa.dma_copy(
            src=expert_affinities.ap(
                pattern=[[E, T], [1, E_L]],
                offset=0,
                scalar_offset=expert_offset_sbuf,
                indirect_dim=1,
            ),
            dst=expert_affinities_masked[0:T, 0:E_L],
            dge_mode=dge_mode.unknown if T % 16 == 0 else dge_mode.swdge,
        )
    else:
        # 3D tiled output [T_par, n_T128_tiles, E_L] for T > 128 (partition dim first)
        T_par = min(T, 128)
        n_T128_tiles = div_ceil(T, 128)
        expert_affinities_masked = nl.ndarray((T_par, n_T128_tiles, E_L), dtype=io_dtype, buffer=nl.sbuf)
        nisa.dma_copy(
            src=expert_affinities.ap(
                pattern=[[E, T_par], [T_par * E, n_T128_tiles], [1, E_L]],
                offset=0,
                scalar_offset=expert_offset_sbuf,
                indirect_dim=0,
            ),
            dst=expert_affinities_masked[0:T_par, 0:n_T128_tiles, 0:E_L],
            dge_mode=dge_mode.unknown,
        )

    # Apply masking based on expert_index when mask_unselected_experts=True
    if mask_unselected_experts:
        _apply_expert_index_mask(
            expert_affinities_masked=expert_affinities_masked,
            expert_index=expert_index,
            expert_offset_sbuf=expert_offset_sbuf,
            E_L=E_L,
            T=T,
            K=K,
            T_32s=T_32s,
            io_dtype=io_dtype,
        )

    return expert_affinities_masked


def _apply_expert_index_mask(
    expert_affinities_masked: nl.ndarray,
    expert_index: nl.ndarray,
    expert_offset_sbuf: nl.ndarray,
    E_L: int,
    T: int,
    K: int,
    T_32s: int,
    io_dtype,
):
    """
    Apply masking to expert affinities based on expert_index.

    For each local expert, checks if it was selected in expert_index for each token.
    Zeros out affinities for experts not selected by each token.

    Args:
        expert_affinities_masked (nl.ndarray): [T, E_L], Affinities to mask in-place in SBUF.
        expert_index (nl.ndarray): [T, K], Top-K expert indices in SBUF.
        expert_offset_sbuf (nl.ndarray): [1, 1], Starting expert index for this rank in SBUF.
        E_L (int): Number of local experts.
        T (int): Number of tokens.
        K (int): Top-K value.
        T_32s (int): T rounded up to multiple of 32.
        io_dtype: Data type for computation.
    """
    # Broadcast expert_offset to [T_32s, K] for comparison
    expert_offset_broadcast = nl.ndarray((T_32s, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(src=expert_offset_sbuf[0, 0], dst=expert_offset_broadcast[0:T_32s, 0:1])

    expert_offset_f = nl.ndarray((T_32s, K), dtype=nl.float32, buffer=nl.sbuf)
    for k_idx in nl.affine_range(K):
        nisa.tensor_copy(
            src=expert_offset_broadcast[0:T_32s, 0:1],
            dst=expert_offset_f[0:T_32s, k_idx : k_idx + 1],
        )

    # For each local expert, mask affinities
    for expert_idx in nl.affine_range(E_L):
        # Check expert_index against current expert: [T, K] comparison
        expert_check = nl.ndarray((T, K), dtype=io_dtype, buffer=nl.sbuf)
        nisa.tensor_scalar(
            dst=expert_check[0:T, 0:K],
            data=expert_index[0:T, 0:K],
            op0=nl.equal,
            operand0=expert_offset_f[0:T, 0:K],
        )

        # Sum across K dimension to get match indicator [T, 1]
        expert_match = nl.ndarray((T, 1), dtype=io_dtype, buffer=nl.sbuf)
        nisa.tensor_reduce(
            dst=expert_match[0:T, 0:1],
            data=expert_check[0:T, 0:K],
            op=nl.add,
            axis=(1,),
        )

        # Multiply affinities by match indicator
        nisa.tensor_tensor(
            dst=expert_affinities_masked[0:T, expert_idx : expert_idx + 1],
            data1=expert_affinities_masked[0:T, expert_idx : expert_idx + 1],
            data2=expert_match[0:T, 0:1],
            op=nl.multiply,
        )

        # Increment expert_offset_f by 1 for next iteration
        nisa.tensor_scalar(
            dst=expert_offset_f[0:T, 0:K],
            data=expert_offset_f[0:T, 0:K],
            op0=nl.add,
            operand0=1,
        )
