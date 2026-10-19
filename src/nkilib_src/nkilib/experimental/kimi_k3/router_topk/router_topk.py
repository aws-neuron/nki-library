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

"""Kimi-K3's flat noaux_tc router for Trainium3 LNC1."""

import nki
import nki.isa as nisa
import nki.language as nl

from ....core.utils.kernel_assert import kernel_assert
from ....core.utils.kernel_helpers import div_ceil

_PMAX = 128
_TOKEN_TILE = 128
_EXPERT_TILE = 128
_EXPERT_MAX = 1024
_TOP_K_MAX = 16
_SELECTION_FLOOR = -3.4028235e38


def _stable_topk_lowest_index(data_sb, top_k):
    """Select top-k values with the model's lowest-index exact-tie rule."""
    tokens, width = data_sb.shape
    work_sb = nl.ndarray((tokens, width), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=work_sb, src=data_sb)

    index_grid_sb = nl.ndarray((tokens, width), dtype=nl.uint32, buffer=nl.sbuf)
    nisa.iota(dst=index_grid_sb, pattern=[[1, width]], offset=0, channel_multiplier=0)
    floor_sb = nl.ndarray((tokens, width), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(dst=floor_sb, value=_SELECTION_FLOOR)

    topk_values_sb = nl.ndarray((tokens, top_k), dtype=nl.float32, buffer=nl.sbuf)
    topk_indices_sb = nl.ndarray((tokens, top_k), dtype=nl.uint32, buffer=nl.sbuf)

    for rank in range(top_k):
        top8_values_sb = nl.ndarray((tokens, 8), dtype=nl.float32, buffer=nl.sbuf)
        top8_indices_sb = nl.ndarray((tokens, 8), dtype=nl.uint32, buffer=nl.sbuf)
        nisa.max8(dst=top8_values_sb, src=work_sb)
        nisa.nc_find_index8(dst=top8_indices_sb, data=work_sb, vals=top8_values_sb)

        nisa.tensor_copy(dst=topk_values_sb[:, rank : rank + 1], src=top8_values_sb[:, 0:1])
        nisa.tensor_copy(dst=topk_indices_sb[:, rank : rank + 1], src=top8_indices_sb[:, 0:1])

        selected_index_f32_sb = nl.ndarray((tokens, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=selected_index_f32_sb, src=top8_indices_sb[:, 0:1])
        selected_mask_sb = nl.ndarray((tokens, width), dtype=nl.uint8, buffer=nl.sbuf)
        nisa.tensor_scalar(
            dst=selected_mask_sb,
            data=index_grid_sb,
            op0=nl.equal,
            operand0=selected_index_f32_sb,
        )
        nisa.tensor_copy_predicated(dst=work_sb, src=floor_sb, predicate=selected_mask_sb)

    return topk_values_sb, topk_indices_sb


@nki.jit
def kimi_k3_router_topk(
    hidden_states: nl.NkiTensor,
    gate_weight: nl.NkiTensor,
    e_score_correction_bias: nl.NkiTensor,
    top_k: int,
    routed_scaling_factor: float = 1.0,
):
    """Compute Kimi-K3's compact flat noaux_tc routing result.

    The operation order is fp32 linear, sigmoid, correction bias for selection
    only, stable top-k, gather from pre-bias scores, L1 normalization with a
    ``1e-20`` guard, and routed scaling.

    Args:
        hidden_states: BF16 ``[T, H]`` model activations.
        gate_weight: BF16 router weights in either natural ``[E, H]`` layout or
            pre-transposed ``[128, H/128, E]`` layout.
        e_score_correction_bias: FP32 selection-only correction bias, ``[E]`` or
            ``[1, E]``.
        top_k: Number of selected experts, ``1 <= top_k <= 16``.
        routed_scaling_factor: Multiplier applied after top-k normalization.

    Returns:
        ``(topk_weight, topk_idx)`` where weights are FP32 ``[T, top_k]`` and
        indices are INT32 ``[T, top_k]``. Exact selection-score ties choose the
        lowest expert index.

    Padding contract:
        The kernel has no token-mask input and treats every row as a real token.
        For bucket padding, the caller must zero compact weights with its
        valid-token mask before the intentionally external dense scatter. That
        produces the required all-zero dense affinity row.
    """
    kernel_assert(
        nl.program_ndim() == 0 or (nl.program_ndim() == 1 and nl.num_programs(0) == 1),
        "kimi_k3_router_topk supports LNC1 only",
    )
    kernel_assert(len(hidden_states.shape) == 2, f"Expected hidden_states [T, H], got {hidden_states.shape}")
    kernel_assert(
        len(gate_weight.shape) in (2, 3),
        f"Expected gate_weight [E, H] or [128, H/128, E], got {gate_weight.shape}",
    )
    kernel_assert(hidden_states.dtype == nl.bfloat16, f"Expected BF16 hidden_states, got {hidden_states.dtype}")
    kernel_assert(gate_weight.dtype == nl.bfloat16, f"Expected BF16 gate_weight, got {gate_weight.dtype}")
    kernel_assert(
        e_score_correction_bias.dtype == nl.float32,
        f"Expected FP32 e_score_correction_bias, got {e_score_correction_bias.dtype}",
    )

    T, H = hidden_states.shape
    weight_pre_transposed = len(gate_weight.shape) == 3
    if weight_pre_transposed:
        weight_partition, num_h_tiles, E = gate_weight.shape
        kernel_assert(weight_partition == _PMAX, f"Expected gate_weight partition dim 128, got {weight_partition}")
        weight_H = num_h_tiles * _PMAX
    else:
        E, weight_H = gate_weight.shape
        num_h_tiles = H // _PMAX

    kernel_assert(T > 0, f"Expected at least one token, got T={T}")
    kernel_assert(H > 0 and H % _PMAX == 0, f"Expected H divisible by 128, got H={H}")
    kernel_assert(weight_H == H, f"hidden H ({H}) must match gate_weight H ({weight_H})")
    kernel_assert(1 <= E <= _EXPERT_MAX, f"Expected 1 <= E <= {_EXPERT_MAX}, got E={E}")
    kernel_assert(1 <= top_k <= _TOP_K_MAX, f"Expected 1 <= top_k <= {_TOP_K_MAX}, got {top_k}")
    kernel_assert(top_k <= E, f"top_k ({top_k}) must be <= E ({E})")
    kernel_assert(
        e_score_correction_bias.shape == (E,) or e_score_correction_bias.shape == (1, E),
        f"Expected e_score_correction_bias [E] or [1, E], got {e_score_correction_bias.shape}",
    )

    e_score_correction_bias = e_score_correction_bias.reshape((1, E))
    num_e_tiles = div_ceil(E, _EXPERT_TILE)
    num_t_tiles = div_ceil(T, _TOKEN_TILE)
    candidate_width = max(8, num_e_tiles * top_k)

    topk_weight = nl.ndarray((T, top_k), dtype=nl.float32, buffer=nl.shared_hbm)
    topk_idx = nl.ndarray((T, top_k), dtype=nl.int32, buffer=nl.shared_hbm)

    for t_tile_idx in range(num_t_tiles):
        t_start = t_tile_idx * _TOKEN_TILE
        t_size = min(_TOKEN_TILE, T - t_start)

        hidden_natural_sb = nl.ndarray((t_size, H), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=hidden_natural_sb, src=hidden_states[t_start : t_start + t_size, :])
        hidden_h_major_sb = nl.ndarray((_PMAX, num_h_tiles, t_size), dtype=nl.float32, buffer=nl.sbuf)
        for h_tile_idx in range(num_h_tiles):
            h_start = h_tile_idx * _PMAX
            hidden_transposed_psum = nl.ndarray((_PMAX, t_size), dtype=nl.float32, buffer=nl.psum)
            nisa.nc_transpose(
                dst=hidden_transposed_psum,
                data=hidden_natural_sb[:, h_start : h_start + _PMAX],
            )
            nisa.tensor_copy(dst=hidden_h_major_sb[:, h_tile_idx, :], src=hidden_transposed_psum)

        candidate_choice_sb = nl.ndarray((t_size, candidate_width), dtype=nl.float32, buffer=nl.sbuf)
        candidate_prebias_sb = nl.ndarray((t_size, candidate_width), dtype=nl.float32, buffer=nl.sbuf)
        candidate_global_idx_sb = nl.ndarray((t_size, candidate_width), dtype=nl.uint32, buffer=nl.sbuf)
        if candidate_width > num_e_tiles * top_k:
            nisa.memset(dst=candidate_choice_sb, value=_SELECTION_FLOOR)
            nisa.memset(dst=candidate_prebias_sb, value=0.0)
            nisa.memset(dst=candidate_global_idx_sb, value=0)

        for e_tile_idx in range(num_e_tiles):
            e_start = e_tile_idx * _EXPERT_TILE
            e_size = min(_EXPERT_TILE, E - e_start)

            if not weight_pre_transposed:
                weight_natural_sb = nl.ndarray((e_size, H), dtype=nl.float32, buffer=nl.sbuf)
                nisa.dma_copy(
                    dst=weight_natural_sb,
                    src=gate_weight[e_start : e_start + e_size, :],
                )

            logits_psum = nl.ndarray((t_size, e_size), dtype=nl.float32, buffer=nl.psum)
            for h_tile_idx in range(num_h_tiles):
                h_start = h_tile_idx * _PMAX
                weight_h_major_sb = nl.ndarray((_PMAX, e_size), dtype=nl.float32, buffer=nl.sbuf)
                if weight_pre_transposed:
                    nisa.dma_copy(
                        dst=weight_h_major_sb,
                        src=gate_weight[:, h_tile_idx, e_start : e_start + e_size],
                    )
                else:
                    weight_transposed_psum = nl.ndarray((_PMAX, e_size), dtype=nl.float32, buffer=nl.psum)
                    nisa.nc_transpose(
                        dst=weight_transposed_psum,
                        data=weight_natural_sb[:, h_start : h_start + _PMAX],
                    )
                    nisa.tensor_copy(dst=weight_h_major_sb, src=weight_transposed_psum)
                nisa.nc_matmul(
                    dst=logits_psum,
                    stationary=hidden_h_major_sb[:, h_tile_idx, :],
                    moving=weight_h_major_sb,
                )

            scores_sb = nl.ndarray((t_size, e_size), dtype=nl.float32, buffer=nl.sbuf)
            nisa.activation(dst=scores_sb, op=nl.sigmoid, data=logits_psum)

            bias_vec_sb = nl.ndarray((1, e_size), dtype=nl.float32, buffer=nl.sbuf)
            nisa.dma_copy(
                dst=bias_vec_sb,
                src=e_score_correction_bias[:, e_start : e_start + e_size],
            )
            ones_sb = nl.ndarray((1, t_size), dtype=nl.float32, buffer=nl.sbuf)
            nisa.memset(dst=ones_sb, value=1.0)
            bias_broadcast_psum = nl.ndarray((t_size, e_size), dtype=nl.float32, buffer=nl.psum)
            nisa.nc_matmul(
                dst=bias_broadcast_psum,
                stationary=ones_sb,
                moving=bias_vec_sb,
                is_stationary_onezero=True,
            )
            choice_sb = nl.ndarray((t_size, e_size), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_tensor(dst=choice_sb, data1=scores_sb, data2=bias_broadcast_psum, op=nl.add)

            if e_size < 8:
                scores_for_topk_sb = nl.ndarray((t_size, 8), dtype=nl.float32, buffer=nl.sbuf)
                choice_for_topk_sb = nl.ndarray((t_size, 8), dtype=nl.float32, buffer=nl.sbuf)
                nisa.memset(dst=scores_for_topk_sb, value=0.0)
                nisa.memset(dst=choice_for_topk_sb, value=_SELECTION_FLOOR)
                nisa.tensor_copy(dst=scores_for_topk_sb[:, :e_size], src=scores_sb)
                nisa.tensor_copy(dst=choice_for_topk_sb[:, :e_size], src=choice_sb)
            else:
                scores_for_topk_sb = scores_sb
                choice_for_topk_sb = choice_sb

            local_choice_sb, local_idx_sb = _stable_topk_lowest_index(choice_for_topk_sb, top_k)
            local_prebias_sb = nl.ndarray((t_size, top_k), dtype=nl.float32, buffer=nl.sbuf)
            nisa.nc_n_gather(dst=local_prebias_sb, data=scores_for_topk_sb, indices=local_idx_sb)
            global_idx_sb = nl.ndarray((t_size, top_k), dtype=nl.uint32, buffer=nl.sbuf)
            nisa.tensor_scalar(dst=global_idx_sb, data=local_idx_sb, op0=nl.add, operand0=e_start)

            candidate_start = e_tile_idx * top_k
            candidate_end = candidate_start + top_k
            nisa.tensor_copy(
                dst=candidate_choice_sb[:, candidate_start:candidate_end],
                src=local_choice_sb,
            )
            nisa.tensor_copy(
                dst=candidate_prebias_sb[:, candidate_start:candidate_end],
                src=local_prebias_sb,
            )
            nisa.tensor_copy(
                dst=candidate_global_idx_sb[:, candidate_start:candidate_end],
                src=global_idx_sb,
            )

        _, candidate_pos_sb = _stable_topk_lowest_index(candidate_choice_sb, top_k)
        topk_idx_i32_sb = nl.ndarray((t_size, top_k), dtype=nl.int32, buffer=nl.sbuf)
        topk_idx_u32_sb = topk_idx_i32_sb.view(nl.uint32)
        topk_prebias_sb = nl.ndarray((t_size, top_k), dtype=nl.float32, buffer=nl.sbuf)
        nisa.nc_n_gather(
            dst=topk_idx_u32_sb,
            data=candidate_global_idx_sb,
            indices=candidate_pos_sb,
        )
        nisa.nc_n_gather(
            dst=topk_prebias_sb,
            data=candidate_prebias_sb,
            indices=candidate_pos_sb,
        )

        sum_weights_sb = nl.ndarray((t_size, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_reduce(
            dst=sum_weights_sb,
            op=nl.add,
            data=topk_prebias_sb,
            axis=1,
            keepdims=True,
        )
        nisa.tensor_scalar(dst=sum_weights_sb, data=sum_weights_sb, op0=nl.add, operand0=1e-20)
        nisa.reciprocal(dst=sum_weights_sb, data=sum_weights_sb)
        topk_weight_sb = nl.ndarray((t_size, top_k), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(
            dst=topk_weight_sb,
            data=topk_prebias_sb,
            op0=nl.multiply,
            operand0=sum_weights_sb,
            op1=nl.multiply,
            operand1=float(routed_scaling_factor),
        )

        nisa.dma_copy(
            dst=topk_idx[t_start : t_start + t_size, :],
            src=topk_idx_i32_sb,
        )
        nisa.dma_copy(
            dst=topk_weight[t_start : t_start + t_size, :],
            src=topk_weight_sb,
        )

    return topk_weight, topk_idx
