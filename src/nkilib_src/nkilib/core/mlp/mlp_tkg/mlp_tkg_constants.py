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

import math
from dataclasses import dataclass
from typing import Optional

import nki.isa as nisa
import nki.language as nl

from ...subkernels.layernorm_tkg import SHARDING_THRESHOLD as LAYERNORM_THRESHOLD
from ...subkernels.rmsnorm_tkg import SHARDING_THRESHOLD as RMSNORM_THRESHOLD
from ...utils.allocator import sizeinbytes
from ...utils.kernel_assert import kernel_assert
from ...utils.kernel_helpers import div_ceil, get_verified_program_sharding_info
from ..mlp_parameters import (
    MLPParameters,
    mlpp_has_layer_normalization,
    mlpp_has_rms_normalization,
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
    num_total_128_tiles_per_I: int
    num_128_tiles_per_I: int
    remainderI: int
    remainderIFused: int
    column_tiling_dim: int
    column_tiling_factor: int
    num_shards_per_I: int
    max_I_shard_size: int
    do_norm_batch_sharding: int
    K: Optional[int] = None
    E: Optional[int] = None


@dataclass
class MLPTKGConstantsGateUpTileCounts(nl.NKIObject):
    """
    Tile counts for Gate/Up projection.

    Contains tiling parameters and PSUM allocation info for gate and up projections.
    """

    HTile: int
    remainderHTile: int
    num_HTiles: int
    num_128_tiles_per_HTile: int
    num_128_tiles_per_remainderHTile: int
    num_allocated_w_tile: int
    last_accessed_addr: int
    num_allocated_psums: int
    gate_psum_base_bank: int
    up_psum_base_bank: int


@dataclass
class MLPTKGConstantsDownTileCounts(nl.NKIObject):
    """
    Tile counts for Down projection.

    Contains tiling parameters and memory allocation info for down projection.
    """

    HTile: int
    remainderHTile: int
    num_HTiles: int
    num_allocated_w_tile: int
    weight_base_idx: int
    num_128_tiles_per_HTile: int
    num_128_tiles_per_remainderHTile: int


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
        program_sharding_info = get_verified_program_sharding_info("mlp_tkg", (0, 1))
        num_shards = program_sharding_info[1]
        shard_id = program_sharding_info[2]

        # --- Tile size constants ---
        _pmax = nl.tile_size.pmax  # Max partition dimension in SBUF
        _psum_fmax = nl.tile_size.psum_fmax  # Max free dim for psum
        _psum_bmax = 8  # Max batch dimension for psum

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
        elif weight_rank == 4:
            # MoE (E, H, 2, I) - interface has fused gate/up
            # TODO: Support both unfused and fused gate/up
            local_E, _, _, I = params.gate_proj_weights_tensor.shape
        elif weight_rank == 5:
            # MX MoE (E, 128, 2, ceil(H/512), I)
            local_E, _, _, _, I = params.gate_proj_weights_tensor.shape
        else:
            kernel_assert(False, f"Weight tensor expected to have rank of 2, 4, or 5 but got {weight_rank}")

        # --- Derived dimensions ---
        H0 = _pmax
        I0 = _pmax
        H1 = H // H0

        K = None
        if params.expert_params and params.expert_params.expert_index:
            _, K = params.expert_params.expert_index.shape

        if params.shard_on_k:
            H1_per_shard_base = H1
            H1_remainder = 0
        else:
            H1_per_shard_base, H1_remainder = divmod(H1, num_shards)

        H1_shard = H1_per_shard_base
        H1_offset = 0 if params.shard_on_k else shard_id * H1_per_shard_base
        H_shard = H1_shard * H0
        H_per_shard = H1_per_shard_base * H0

        kernel_assert(
            H1_remainder == 0,
            f"Invalid sharding: H1={H1} cannot be evenly divided across {num_shards} cores",
        )

        # --- Determine the number of shards along the I dimension ---
        if params.use_tkg_gate_up_proj_column_tiling:
            # Hardware restriction: moving tensor processes 512 elements per PSUM bank, with 8 PSUM banks
            max_I_shard_size = 512 * 8  # Maximum I elements per loop
        else:
            # Hardware restriction: stationary tensor processes 128 elements per PSUM bank, with 8 PSUM banks
            max_I_shard_size = 128 * 8  # Maximum I elements per loop
        num_shards_per_I = div_ceil(I, max_I_shard_size)

        # --- 128 tiling across I dimension ---
        num_128_tiles_per_I, remainderI = divmod(I, I0)
        num_total_128_tiles_per_I = num_128_tiles_per_I + int(remainderI != 0)

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

        # --- Check if normalization will use batch-sharding ---
        # Layout when sharded: (num_shards, T/num_shards, H)
        # Required to ensure deterministic fused-add and prevent non-determinism errors
        is_T_evenly_divisible = T % num_shards == 0
        do_norm_batch_sharding = (
            mlpp_has_rms_normalization(params) and T > RMSNORM_THRESHOLD and is_T_evenly_divisible
        ) or (mlpp_has_layer_normalization(params) and T > LAYERNORM_THRESHOLD and is_T_evenly_divisible)
        do_norm_batch_sharding = do_norm_batch_sharding and (not params.shard_on_k)

        return MLPTKGConstantsDimensionSizes(
            _pmax=_pmax,
            _psum_fmax=_psum_fmax,
            _psum_bmax=_psum_bmax,
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
            num_total_128_tiles_per_I=num_total_128_tiles_per_I,
            num_128_tiles_per_I=num_128_tiles_per_I,
            remainderI=remainderI,
            column_tiling_dim=column_tiling_dim,
            column_tiling_factor=column_tiling_factor,
            num_shards_per_I=num_shards_per_I,
            max_I_shard_size=max_I_shard_size,
            do_norm_batch_sharding=do_norm_batch_sharding,
            K=K,
            E=local_E,
        )

    @staticmethod
    def calculate_gate_up_tiles(
        gate_up_io_size: int,
        remaining_space: int,
        params: MLPParameters,
        kernel_dims: MLPTKGConstantsDimensionSizes,
        use_auto_alloc: bool = False,
    ) -> MLPTKGConstantsGateUpTileCounts:
        """
        Calculate tiling and PSUM allocation for Gate/Up projection.

        Args:
            gate_up_io_size (int): Size of IO tensors in Gate/Up projection.
            remaining_space (int): Remaining SBUF memory available for weights.
            params (MLPParameters): MLP configuration parameters.
            kernel_dims (MLPTKGConstantsDimensionSizes): Precomputed dimension constants.
            use_auto_alloc (bool): Whether auto-allocation is enabled. Default is False.

        Returns:
            MLPTKGConstantsGateUpTileCounts: Dataclass with tiling and PSUM allocation info.
        """
        I = kernel_dims.I
        num_total_128_tiles_per_I = kernel_dims.num_total_128_tiles_per_I
        weight_dtype = (
            params.gate_proj_weights_tensor.dtype
            if params.gate_proj_weights_tensor is not None
            else params.up_proj_weights_tensor.dtype
        )
        weight_dtype_size = sizeinbytes(weight_dtype)

        # Weight tiles are loaded [HTile, I] at a time for efficient memory access
        gate_up_HTile = 2048 * 2 if params.quant_params.is_quant() else 2048
        # number of H-tiles along H dimension
        gate_up_num_HTile_per_H, gate_up_remainderHTile = divmod(kernel_dims.H_per_shard, gate_up_HTile)
        gate_up_num_HTiles = gate_up_num_HTile_per_H + (gate_up_remainderHTile != 0)
        # number of 128-size tiles per H-tile
        gate_num_128_tiles_per_HTile = gate_up_HTile // kernel_dims._pmax
        gate_num_128_tiles_per_remainderHTile = gate_up_remainderHTile // kernel_dims._pmax
        # compute size of weight tile
        size_of_weight_tile = I * gate_num_128_tiles_per_HTile * weight_dtype_size
        # number of weight tiles to allocate (x2 for both gate and up projections)
        num_required_w_tile = gate_up_num_HTiles * 2
        num_available_w_tile = remaining_space // size_of_weight_tile
        gate_num_allocated_w_tile = min(num_required_w_tile, num_available_w_tile)

        if gate_num_allocated_w_tile <= 0:
            gate_up_HTile = 512 * 2 if params.quant_params.is_quant() else 512
            # number of H-tiles along H dimension
            gate_up_num_HTile_per_H, gate_up_remainderHTile = divmod(kernel_dims.H_per_shard, gate_up_HTile)
            gate_up_num_HTiles = gate_up_num_HTile_per_H + (gate_up_remainderHTile != 0)
            # number of 128-size tiles per H-tile
            gate_num_128_tiles_per_HTile = gate_up_HTile // kernel_dims._pmax
            gate_num_128_tiles_per_remainderHTile = gate_up_remainderHTile // kernel_dims._pmax
            # compute size of weight tile
            size_of_weight_tile = I * gate_num_128_tiles_per_HTile * weight_dtype_size
            # number of weight tiles to allocate (x2 for both gate and up projections)
            num_required_w_tile = gate_up_num_HTiles * 2
            num_available_w_tile = remaining_space // size_of_weight_tile
            gate_num_allocated_w_tile = min(num_required_w_tile, num_available_w_tile)

        if not use_auto_alloc:
            kernel_assert(
                gate_num_allocated_w_tile > 0,
                "Not enough memory for Gate/Up projection weights",
            )
        else:
            gate_num_allocated_w_tile = 2  # Default for auto-alloc: double-buffering

        # --- PSUM management for Gate + Up projection ---
        # Required the number of PSUMs for a single projection
        if params.use_tkg_gate_up_proj_column_tiling:
            num_required_psums = div_ceil(I, kernel_dims._psum_fmax)
        else:
            num_required_psums = num_total_128_tiles_per_I

        # Allocate PSUMs, capped by the hardware maximum
        num_allocated_psums = min(num_required_psums, kernel_dims._psum_bmax)

        # Assign separate PSUM banks for Gate and Up if enough banks available, otherwise share
        gate_psum_base_bank = 0
        up_psum_base_bank = num_allocated_psums if (num_allocated_psums * 2) < kernel_dims._psum_bmax else 0

        # --- Ring buffer index tracking for weight tile reuse ---
        # Gate and Up projections share weight tiles as a ring buffer. Track the last accessed
        # index so Up projection loads after Gate to avoid anti-dependencies.
        w_mod = num_required_w_tile % gate_num_allocated_w_tile
        last_gate_idx = gate_num_allocated_w_tile - 1 if w_mod == 0 else w_mod - 1

        # Track last memory address accessed by Gate/Up projection. Down projection uses this
        # to start at a safe offset, avoiding anti-dependencies so it can load weights ASAP.
        last_accessed_addr = gate_up_io_size + size_of_weight_tile * (last_gate_idx + 1)

        return MLPTKGConstantsGateUpTileCounts(
            HTile=gate_up_HTile,
            remainderHTile=gate_up_remainderHTile,
            num_HTiles=gate_up_num_HTiles,
            num_128_tiles_per_HTile=gate_num_128_tiles_per_HTile,
            num_128_tiles_per_remainderHTile=gate_num_128_tiles_per_remainderHTile,
            num_allocated_w_tile=gate_num_allocated_w_tile,
            last_accessed_addr=last_accessed_addr,
            num_allocated_psums=num_allocated_psums,
            gate_psum_base_bank=gate_psum_base_bank,
            up_psum_base_bank=up_psum_base_bank,
        )

    @staticmethod
    def calculate_down_tiles(
        down_io_size: int,
        remaining_space: int,
        params: MLPParameters,
        kernel_dims: MLPTKGConstantsDimensionSizes,
        gate_tile_info: MLPTKGConstantsGateUpTileCounts,
        use_auto_alloc: bool = False,
    ) -> MLPTKGConstantsDownTileCounts:
        """
        Calculate tiling and memory allocation for Down projection.

        Args:
            down_io_size (int): Size of IO tensors in Down projection.
            remaining_space (int): Remaining SBUF memory available for weights.
            params (MLPParameters): MLP configuration parameters.
            kernel_dims (MLPTKGConstantsDimensionSizes): Precomputed dimension constants.
            gate_tile_info (MLPTKGConstantsGateUpTileCounts): Gate/Up tiling info for anti-dependency avoidance.
            use_auto_alloc (bool): Whether auto-allocation is enabled. Default is False.

        Returns:
            MLPTKGConstantsDownTileCounts: Dataclass with tiling and memory allocation info.
        """
        weight_dtype = params.down_proj_weights_tensor.dtype
        weight_dtype_size = sizeinbytes(weight_dtype)
        num_total_128_tiles_per_I = kernel_dims.num_total_128_tiles_per_I

        # --- H-tile size for Down projection ---
        if params.use_tkg_down_proj_column_tiling:
            down_HTile = 4096 * 2 if params.quant_params.is_quant() else 4096
        else:
            down_HTile = kernel_dims.H1_shard * kernel_dims.H0

        # --- Compute number of H-tiles along H dimension ---
        down_num_HTile_per_H, down_remainderHTile = divmod(kernel_dims.H_per_shard, down_HTile)
        down_num_HTiles = down_num_HTile_per_H + int(down_remainderHTile != 0)

        # --- Compute number of 128-size tiles per H-tile ---
        down_num_128_tiles_per_HTile = down_HTile // kernel_dims._pmax
        down_num_128_tiles_per_remainderHTile = down_remainderHTile // kernel_dims._pmax

        # --- Compute number of weight tiles to allocate ---
        size_of_weight_tile = down_HTile * weight_dtype_size
        num_required_w_tile = num_total_128_tiles_per_I * down_num_HTiles
        num_available_w_tile = remaining_space // size_of_weight_tile
        down_num_allocated_w_tile = min(num_required_w_tile, num_available_w_tile)

        if not use_auto_alloc:
            kernel_assert(
                down_num_allocated_w_tile > 0,
                "Not enough memory for Down projection weights",
            )
        else:
            down_num_allocated_w_tile = 2  # Default for auto-alloc: double-buffering

        # --- Compute starting weight index to avoid anti-dependencies with Gate/Up ---
        # If Down's weight address range overlaps with Gate/Up's last accessed address,
        # offset the starting index to avoid anti-dependencies and enable early weight loading.
        last_accessed_addr = gate_tile_info.last_accessed_addr
        down_weight_addr_space = last_accessed_addr - down_io_size

        if down_io_size < last_accessed_addr < down_io_size + down_num_allocated_w_tile * size_of_weight_tile:
            weight_base_idx = div_ceil(down_weight_addr_space, size_of_weight_tile)
        else:
            weight_base_idx = 0

        return MLPTKGConstantsDownTileCounts(
            HTile=down_HTile,
            remainderHTile=down_remainderHTile,
            num_HTiles=down_num_HTiles,
            num_allocated_w_tile=down_num_allocated_w_tile,
            weight_base_idx=weight_base_idx,
            num_128_tiles_per_HTile=down_num_128_tiles_per_HTile,
            num_128_tiles_per_remainderHTile=down_num_128_tiles_per_remainderHTile,
        )
