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

import nki.isa as nisa
import nki.language as nl
from nkilib_src.nkilib.core.moe import moe_tkg
from nkilib_src.nkilib.core.utils.common_types import (
    ActFnType,
    ExpertAffinityScaleMode,
)
from nkilib_src.nkilib.core.utils.tensor_view import TensorView


def moe_tkg_sbuf_io_wrapper(
    hidden_input: nl.ndarray,
    gate_up_weights: nl.ndarray,
    down_weights: nl.ndarray,
    expert_affinities: nl.ndarray,
    expert_index: nl.ndarray,
    is_all_expert: bool,
    rank_id: nl.ndarray = None,
    gate_up_weights_bias: nl.ndarray = None,
    down_weights_bias: nl.ndarray = None,
    expert_affinities_scaling_mode: ExpertAffinityScaleMode = ExpertAffinityScaleMode.NO_SCALE,
    activation_fn: ActFnType = ActFnType.SiLU,
    gate_clamp_upper_limit: float = None,
    gate_clamp_lower_limit: float = None,
    up_clamp_upper_limit: float = None,
    up_clamp_lower_limit: float = None,
    mask_unselected_experts: bool = False,
) -> nl.ndarray:
    """Wrapper to test all-expert MoE with SBUF input."""
    T, H = hidden_input.shape
    E = gate_up_weights.shape[0]
    H0 = nl.tile_size.pmax

    if is_all_expert:
        h_num_shards = nl.num_programs(0)
        h_shard_id = nl.program_id(0)
    else:
        h_num_shards = 1
        h_shard_id = 0

    H1_shard = H // H0 // h_num_shards

    # Allocate SBUF tensors
    hidden_sb = nl.ndarray((H0, T, H1_shard), dtype=hidden_input.dtype, buffer=nl.sbuf, name="hidden_sb")

    # Load this shard's portion: (T, H) -> (T, h_num_shards, H0, H1_shard) -> select shard -> (H0, T, H1_shard)
    input_view = TensorView(hidden_input)
    input_view = (
        input_view.reshape_dim(dim=1, shape=[h_num_shards, H0, H1_shard])
        .permute(dims=[2, 0, 1, 3])
        .select(dim=2, index=h_shard_id)
    )

    nisa.dma_copy(hidden_sb, input_view.get_view())

    # For all-expert mode, pass HBM affinities directly (mask_expert_affinities handles the load)
    # For selective mode, pre-load affinities to SBUF
    if is_all_expert:
        affinities_for_moe_tkg = expert_affinities
    else:
        affinities_sb = nl.ndarray((T, E), dtype=expert_affinities.dtype, buffer=nl.sbuf, name="affinities_sb")
        nisa.dma_copy(affinities_sb, expert_affinities)
        affinities_for_moe_tkg = affinities_sb

    output_sb = moe_tkg(
        hidden_input=hidden_sb,
        expert_gate_up_weights=gate_up_weights,
        expert_down_weights=down_weights,
        expert_affinities=affinities_for_moe_tkg,
        expert_index=expert_index,
        is_all_expert=is_all_expert,
        rank_id=rank_id,
        expert_gate_up_bias=gate_up_weights_bias,
        expert_down_bias=down_weights_bias,
        mask_unselected_experts=mask_unselected_experts,
        gate_clamp_upper_limit=gate_clamp_upper_limit,
        gate_clamp_lower_limit=gate_clamp_lower_limit,
        up_clamp_upper_limit=up_clamp_upper_limit,
        up_clamp_lower_limit=up_clamp_lower_limit,
        expert_affinities_scaling_mode=expert_affinities_scaling_mode,
        activation_fn=activation_fn,
        output_in_sbuf=True,
    )

    if is_all_expert:
        t_shard_id = 0
        t_num_shards = 1
    else:
        t_shard_id = nl.program_id(0)
        t_num_shards = nl.num_programs(0)

    T_shard = T // t_num_shards
    T_offset = t_shard_id * T_shard

    # Copy SBUF output to HBM for validation
    # (T, H) -> (T, h_num_shards, H0, H1_shard) -> (H0, T, h_num_shards, H1_shard) -> (H0, T, H1_shard)
    output = nl.ndarray((T, H), dtype=hidden_input.dtype, buffer=nl.shared_hbm)
    output_view = TensorView(output)
    output_view = (
        output_view.reshape_dim(dim=1, shape=[h_num_shards, H0, H1_shard])  # (T, h_num_shards, H0, H1_shard)
        .permute(dims=[2, 0, 1, 3])  # (H0, T, h_num_shards, H1_shard)
        .select(dim=2, index=h_shard_id)  # (H0, T, H1_shard)
        .slice(dim=1, start=T_offset, end=T_offset + T_shard)  # (H0, T_shard, H1_shard)
    )
    nisa.dma_copy(output_view.get_view(), output_sb[:, nl.ds(T_offset, T_shard), :])
    return output
