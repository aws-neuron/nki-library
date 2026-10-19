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

"""Constants and configuration dataclasses for MLP TKG kernel tiling and memory allocation."""

from dataclasses import dataclass
from typing import Optional

import nki.isa as nisa
import nki.language as nl

from ...utils.allocator import SbufManager, sizeinbytes
from ...utils.common_types import HiddenLayout
from ...utils.kernel_assert import kernel_assert
from ...utils.kernel_helpers import div_ceil, get_verified_program_sharding_info
from ..mlp_parameters import (
    _Q_HEIGHT,
    _Q_WIDTH,
    MLPParameters,
)


@dataclass
class MLPTKGConstantsDimensionSizes(nl.NKIObject):
    """
    Dimension sizes for MLP TKG computation.

    Contains all dimension constants computed from input parameters including
    partition sizes, sharding info, and tiling parameters.
    """

    _pmax: int
    _psum_fmax: int
    _psum_bmax: int
    _q_width: int
    _q_height: int
    T: int
    H: int
    I: int
    H0: int
    H1: int
    I0: int
    num_shards: int
    shard_id: int
    H_shard: int
    H1_shard: int
    H1_offset: int
    H_per_shard: int
    column_tiling_dim: int
    column_tiling_factor: int
    max_I_shard_size: int
    hidden_layout: HiddenLayout
    K: Optional[int] = None
    E: Optional[int] = None


@dataclass
class MLPTKGConstantsGateUpTileCounts(nl.NKIObject):
    """
    Tile counts for Gate/Up projection.

    Contains tiling parameters and PSUM allocation info for gate and up projections.
    """

    HTile: int
    num_allocated_w_tile: int
    last_accessed_addr: int
    up_psum_base_bank: int
    I_shard_size: int


@dataclass
class MLPTKGConstantsDownTileCounts(nl.NKIObject):
    """
    Tile counts for Down projection.

    Contains tiling parameters and memory allocation info for down projection.
    """

    HTile: int
    num_allocated_w_tile: int
    weight_base_idx: int


class MLPTKGConstants(nl.NKIObject):
    """Constants for MLP TKG kernel implementation."""

    @staticmethod
    def calculate_constants(params: MLPParameters) -> MLPTKGConstantsDimensionSizes:
        """
        Calculate all dimension constants needed for the MLP TKG kernel.

        Args:
            params (MLPParameters): MLP configuration parameters.

        Returns:
            MLPTKGConstantsDimensionSizes: Dataclass with all computed dimension constants.
        """
        # --- Program sharding info ---
        if params.shard_on_h_disabled:
            num_shards, shard_id = (1, 0)
        else:
            program_sharding_info = get_verified_program_sharding_info("mlp_tkg", (0, 1))
            num_shards = program_sharding_info[1]
            shard_id = program_sharding_info[2]

        # --- Tile size constants ---
        _pmax = nl.tile_size.pmax  # Max partition dimension in SBUF
        _psum_fmax = nl.tile_size.psum_fmax  # Max free dim for psum
        _psum_bmax = 8  # Max batch dimension for psum
        _q_width = _Q_WIDTH  # Quantization width for MX formats
        _q_height = _Q_HEIGHT  # Quantization height for MX formats

        # --- Input tensor shapes ---
        # Use pre-computed dimensions from MLPParameters to support SBUF input
        T = params.batch_size * params.sequence_len
        H = params.hidden_size

        # --- Weight tensor shapes ---
        weight_rank = len(params.gate_proj_weights_tensor.shape)
        if weight_rank == 2:
            # Dense
            _, I = params.gate_proj_weights_tensor.shape
            local_E = None
        elif weight_rank == 3:
            # MX MLP (128, ceil(H/512), I) - MX quantized weights
            _, _, I = params.gate_proj_weights_tensor.shape
            local_E = None
        elif weight_rank == 4:
            # MoE (E, H, 2, I) - interface has fused gate/up
            # TODO: Support both unfused and fused gate/up
            local_E, _, _, I = params.gate_proj_weights_tensor.shape
        elif weight_rank == 5:
            # MX MoE (E, 128, 2, ceil(H/512), I)
            local_E, _, _, _, I = params.gate_proj_weights_tensor.shape
        else:
            kernel_assert(False, f"Weight tensor expected to have rank of 2, 3, 4, or 5 but got {weight_rank}")

        # --- Derived dimensions ---
        H0 = _pmax
        I0 = _pmax
        H1 = H // H0

        K = None
        if params.expert_params and params.expert_params.expert_index:
            K = params.expert_params.expert_index.shape[-1]

        H1_per_shard_base = H1 // num_shards
        H1_remainder = H1 % num_shards

        # Unbalanced sharding is only supported for MoE all-expert BF16 path.
        # For all other paths, H1 must be evenly divisible by num_shards.
        is_moe_expert = params.expert_params and params.expert_params.expert_index
        if H1_remainder != 0 and is_moe_expert:
            # Unbalanced: first (num_shards - 1) shards get floor(H1/num_shards),
            # last shard gets the remainder so all H1 tiles are covered.
            if shard_id < num_shards - 1:
                H1_shard = H1_per_shard_base
            else:
                H1_shard = H1 - H1_per_shard_base * (num_shards - 1)
        else:
            kernel_assert(
                H1_remainder == 0,
                f"Invalid sharding: H1={H1} cannot be evenly divided across {num_shards} cores",
            )
            H1_shard = H1_per_shard_base
        H1_offset = shard_id * H1_per_shard_base
        H_shard = H1_shard * H0
        H_per_shard = H1_shard * H0

        # --- Determine the number of shards along the I dimension ---
        if params.use_tkg_gate_up_proj_column_tiling:
            # Hardware restriction: moving tensor processes 512 elements per PSUM bank, with 8 PSUM banks
            max_I_shard_size = _psum_fmax * _psum_bmax  # Maximum I elements per loop
        else:
            # Hardware restriction: stationary tensor processes 128 elements per PSUM bank, with 8 PSUM banks
            max_I_shard_size = _pmax * _psum_bmax  # Maximum I elements per loop

        # --- Column tiling strategy based on T ---
        if T <= 32:
            column_tiling_dim = 32
        elif T <= 64:
            column_tiling_dim = 64
        else:
            column_tiling_dim = 128

        # Adjust hardware-specific logic for column tiling on NeuronCore-v2
        if nisa.get_nc_version() == nisa.nc_version.gen2:
            # Both the row and column sizes in tile_size cannot be 32
            column_tiling_dim = 64

        column_tiling_factor = 128 // column_tiling_dim

        return MLPTKGConstantsDimensionSizes(
            _pmax=_pmax,
            _psum_fmax=_psum_fmax,
            _psum_bmax=_psum_bmax,
            _q_width=_q_width,
            _q_height=_q_height,
            T=T,
            H=H,
            I=I,
            H0=H0,
            H1=H1,
            I0=I0,
            num_shards=num_shards,
            shard_id=shard_id,
            H_shard=H_shard,
            H1_shard=H1_shard,
            H1_offset=H1_offset,
            H_per_shard=H_per_shard,
            column_tiling_dim=column_tiling_dim,
            column_tiling_factor=column_tiling_factor,
            max_I_shard_size=max_I_shard_size,
            hidden_layout=None,  # Set by input_norm_load based on norm kernel output layout
            K=K,
            E=local_E,
        )

    @staticmethod
    def calculate_gate_up_tiles(
        params: MLPParameters,
        dims: MLPTKGConstantsDimensionSizes,
        sbm: SbufManager,
        share_memory_scope: bool = False,
    ) -> MLPTKGConstantsGateUpTileCounts:
        """
        Calculate tiling and PSUM allocation for Gate/Up projection.

        Args:
            params (MLPParameters): MLP configuration parameters.
            dims (MLPTKGConstantsDimensionSizes): Precomputed dimension constants.
            sbm (SbufManager): SBUF memory manager used to query allocation state.
            share_memory_scope (bool): If True, the gate/up projection shares SBUF memory
                with an outer scope (e.g., MoE expert loop) instead of owning its own
                allocation. Defaults to False.

        Returns:
            MLPTKGConstantsGateUpTileCounts: Dataclass with tiling and PSUM allocation info.
        """
        I = dims.I
        gate_up_io_size = 0 if sbm.is_auto_alloc() else sbm.get_stack_curr_addr()
        remaining_space = 0 if sbm.is_auto_alloc() else sbm.get_free_space()

        w_dtype_sz = sizeinbytes(params.up_proj_weights_tensor.dtype)

        # --- Compute HTile for Gate + Up projection ---
        ini_HTile = 2048 * 2 if params.quant_params.is_quant() else 2048
        min_HTile = 512 * 2 if params.quant_params.is_quant() else 512
        min_ITile = 512
        # Weight tiles are loaded [HTile, I] at a time for efficient memory access
        if sbm.is_auto_alloc():
            HTile = min_HTile
            w_tile_sz = I * (HTile // dims._pmax) * w_dtype_sz
            num_required_w_tile = div_ceil(dims.H_per_shard, HTile)
            num_allocated_w_tile = 2
        elif share_memory_scope:
            HTile = ini_HTile
            w_tile_sz = I * (HTile // dims._pmax) * w_dtype_sz
            num_available_w_tile = remaining_space // w_tile_sz

            # reduce HTile size. Fall back to min_HTile
            if num_available_w_tile < 1:
                HTile = min_HTile
                w_tile_sz = I * (HTile // dims._pmax) * w_dtype_sz
                num_available_w_tile = remaining_space // w_tile_sz

            # compute num_allocated_w_tile
            num_required_w_tile = div_ceil(dims.H_per_shard, HTile)
            num_allocated_w_tile = min(num_required_w_tile, num_available_w_tile)
        else:
            I = min(I, dims.max_I_shard_size)
            HTile = min(dims.H_per_shard, ini_HTile)
            w_tile_sz = I * (HTile // dims._pmax) * w_dtype_sz
            num_available_w_tile = remaining_space // w_tile_sz
            num_required_w_tile = div_ceil(dims.H_per_shard, HTile)

            # reduce HTile size
            # allocate at least 2 tiles for each gate and up projection
            while (num_required_w_tile < 2 or num_available_w_tile < 4) and HTile >= min_HTile:
                ini_HTile = ini_HTile // 2
                HTile = min(dims.H_per_shard, ini_HTile)
                w_tile_sz = I * (HTile // dims._pmax) * w_dtype_sz
                num_available_w_tile = remaining_space // w_tile_sz
                num_required_w_tile = div_ceil(dims.H_per_shard, HTile)

            # reduce I size
            while num_available_w_tile < 4 and I >= min_ITile:
                I = div_ceil(I, 2)
                w_tile_sz = I * (HTile // dims._pmax) * w_dtype_sz
                num_available_w_tile = remaining_space // w_tile_sz

            # compute num_allocated_w_tile
            num_allocated_w_tile = min(4, num_available_w_tile)

        kernel_assert(
            num_allocated_w_tile > 0,
            "Not enough memory for Gate/Up projection weights",
        )

        # --- PSUM management for Gate + Up projection ---
        if params.use_tkg_gate_up_proj_column_tiling:
            num_required_psums = div_ceil(I, dims._psum_fmax)
        else:
            num_required_psums = div_ceil(I, dims._pmax)

        # Assign base PSUM bank for Up (Gate always starts at 0)
        up_psum_base_bank = num_required_psums

        # --- Ring buffer index tracking for weight tile reuse ---
        # Gate and Up projections share weight tiles as a ring buffer. Track the last accessed
        # index so Up projection loads after Gate to avoid anti-dependencies.
        w_mod = num_required_w_tile % num_allocated_w_tile
        last_gate_idx = num_allocated_w_tile if w_mod == 0 else w_mod
        last_accessed_addr = gate_up_io_size + w_tile_sz * last_gate_idx

        return MLPTKGConstantsGateUpTileCounts(
            HTile=HTile,
            num_allocated_w_tile=num_allocated_w_tile,
            last_accessed_addr=last_accessed_addr,
            up_psum_base_bank=up_psum_base_bank,
            I_shard_size=I,
        )

    @staticmethod
    def calculate_down_tiles(
        params: MLPParameters,
        dims: MLPTKGConstantsDimensionSizes,
        gate_tile_info: MLPTKGConstantsGateUpTileCounts,
        sbm: SbufManager,
    ) -> MLPTKGConstantsDownTileCounts:
        """
        Calculate tiling and memory allocation for Down projection.

        Args:
            params (MLPParameters): MLP configuration parameters.
            dims (MLPTKGConstantsDimensionSizes): Precomputed dimension constants.
            gate_tile_info (MLPTKGConstantsGateUpTileCounts): Gate/Up tiling info for anti-dependency avoidance.
            sbm (SbufManager): SBUF memory manager used to query allocation state.

        Returns:
            MLPTKGConstantsDownTileCounts: Dataclass with tiling and memory allocation info.
        """
        down_io_size = 0 if sbm.is_auto_alloc() else sbm.get_stack_curr_addr()
        remaining_space = 0 if sbm.is_auto_alloc() else sbm.get_free_space()

        # H-tile size for Down projection
        if params.use_tkg_down_proj_column_tiling:
            down_HTile = 8192 if params.quant_params.is_quant() else 4096
            down_HTile = min(dims.H_per_shard, down_HTile)
            num_required_psums_per_HTile = div_ceil(down_HTile, dims._psum_fmax)
            num_required_psum_after_column_tiling = div_ceil(num_required_psums_per_HTile, dims.column_tiling_factor)

            while dims._psum_bmax < num_required_psum_after_column_tiling:
                down_HTile = div_ceil(down_HTile, 2)
                num_required_psums_per_HTile = div_ceil(down_HTile, dims._psum_fmax)
                num_required_psum_after_column_tiling = div_ceil(
                    num_required_psums_per_HTile, dims.column_tiling_factor
                )
        else:
            down_HTile = dims.H1_shard * dims.H0

        if sbm.is_auto_alloc():
            return MLPTKGConstantsDownTileCounts(
                HTile=down_HTile,
                num_allocated_w_tile=2,
                weight_base_idx=0,
            )

        stack_cur_addr = sbm.get_stack_curr_addr()
        free_space = sbm.get_free_space()

        w_dtype_size = sizeinbytes(params.down_proj_weights_tensor.dtype)
        num_HTiles = div_ceil(dims.H_per_shard, down_HTile)
        num_required_w_tile = div_ceil(dims.I, dims.I0) * num_HTiles
        size_of_w_tile = down_HTile * w_dtype_size
        num_available_w_tile = free_space // size_of_w_tile
        down_num_allocated_w_tile = min(num_required_w_tile, num_available_w_tile)

        kernel_assert(
            down_num_allocated_w_tile > 0,
            "Not enough memory for Down projection weights",
        )

        # --- Compute starting weight index to avoid anti-dependencies with Gate/Up ---
        # If Down's weight address range overlaps with Gate/Up's last accessed address,
        # offset the starting index to avoid anti-dependencies and enable early weight loading.
        last_accessed_addr = gate_tile_info.last_accessed_addr
        overlapped_w_addr_space = gate_tile_info.last_accessed_addr - stack_cur_addr
        weight_base_idx = 0
        if 0 < overlapped_w_addr_space < down_num_allocated_w_tile * size_of_w_tile:
            weight_base_idx = div_ceil(overlapped_w_addr_space, size_of_w_tile)

        return MLPTKGConstantsDownTileCounts(
            HTile=down_HTile,
            num_allocated_w_tile=down_num_allocated_w_tile,
            weight_base_idx=weight_base_idx,
        )
