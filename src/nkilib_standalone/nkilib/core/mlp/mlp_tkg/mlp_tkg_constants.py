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
kernels - high performance MLP kernels

"""

import math
from dataclasses import dataclass

import nki
import nki.isa as nisa
import nki.language as nl

# common utils
from ...utils.kernel_assert import kernel_assert
from ...utils.kernel_helpers import get_verified_program_sharding_info
from ...utils.allocator import sizeinbytes

# subkernels utils
from ...subkernels.layernorm_tkg import SHARDING_THRESHOLD as layernorm_threshold
from ...subkernels.rmsnorm_tkg import SHARDING_THRESHOLD as rmsnorm_threshold

# MLP utils
from ..mlp_parameters import (
    MLPParameters,
    mlpp_has_layer_normalization,
    mlpp_has_rms_normalization,
)


@dataclass
class MLPTKGConstantsDimensionSizes(nl.NKIObject):
    """Actual dimension sizes for the MLP computation."""

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


@dataclass
class MLPTKGConstantsGateUpTileCounts(nl.NKIObject):
    """Number of tiles needed for various dimensions."""

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
    """Number of tiles needed for various dimensions."""

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
    def calculate_constants(params: MLPParameters):
        """Calculates all constants needed for the MLP TKG kernel."""
        # --- Program sharding info ---
        program_sharding_info = get_verified_program_sharding_info("mlp_kernel", (0, 1))
        num_shards = program_sharding_info[1]
        shard_id = program_sharding_info[2]

        # --- Tile size constants ---
        _pmax = nl.tile_size.pmax  # Max partition dimension in SBUF
        _psum_fmax = nl.tile_size.psum_fmax  # Max free dim for psum
        _psum_bmax = 8  # Max batch dimension for psum

        # --- Input tensor shapes ---
        B, S, H = params.hidden_tensor.shape
        _, I = params.gate_proj_weights_tensor.shape
        T = B * S

        # --- Derived dimensions ---
        H0 = _pmax
        I0 = _pmax
        H1 = H // H0

        H1_per_shard_base = H1 // num_shards
        H1_remainder = H1 % num_shards
        H1_shard = H1_per_shard_base
        H1_offset = shard_id * H1_per_shard_base
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
            # See matrix multiplication behaviors in the docstring for each projection
        else:
            # Hardware restriction: stationary tensor processes 128 elements per PSUM bank, with 8 PSUM banks
            max_I_shard_size = 128 * 8  # Maximum I elements per loop
            # See matrix multiplication behaviors in the docstring for each projection
        num_shards_per_I = math.ceil(I / max_I_shard_size)  # Tile I into shards based on max_shard_size
        max_I_shard_size = max_I_shard_size

        # --- 128 tiling across I dimension ---
        num_128_tiles_per_I = I // I0
        remainderI = I % I0
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
            mlpp_has_rms_normalization(params) and T > rmsnorm_threshold and is_T_evenly_divisible
        ) or (mlpp_has_layer_normalization(params) and T > layernorm_threshold and is_T_evenly_divisible)

        # --- Return all constants as a dataclass ---
        return MLPTKGConstantsDimensionSizes(
            _pmax=_pmax,
            _psum_fmax=_psum_fmax,
            _psum_bmax=_psum_bmax,
            T=T,
            H=H,
            I=I,
            H0=H0,  # Always equal to the partition max dimension, max value = 128
            H1=H1,
            I0=I0,  # Always equal to the partition max dimension, max value = 128
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
        )

    @staticmethod
    def calculate_gate_up_tiles(
        gate_up_io_size: int,
        remaining_space: int,
        params: MLPParameters,
        kernel_dims: MLPTKGConstantsDimensionSizes,
    ):
        """
        Calculate tiling and PSUM allocation for the Gate/Up projection in MLP TKG.

        Args:
            gate_up_io_size: Size of IO tensors in Gate/Up projection.
            remaining_space: Remaining SBUF memory available for weights.
            params: MLP Parameters.
            kernel_dims: Precomputed kernel dimension constants.

        Returns:
            MLPTKGConstantsGateUpTileCounts: Dataclass with tiling and PSUM allocation info.
        """
        I = kernel_dims.I
        num_total_128_tiles_per_I = kernel_dims.num_total_128_tiles_per_I
        weight_dtype = params.gate_proj_weights_tensor.dtype
        weight_dtype_size = sizeinbytes(weight_dtype)

        # Weight tiles are loaded [HTile, I] at a time for efficient memory access
        gate_up_HTile = 2048
        # number of H-tiles along H dimension
        gate_up_num_HTile_per_H = kernel_dims.H_per_shard // gate_up_HTile
        gate_up_remainderHTile = kernel_dims.H_per_shard % gate_up_HTile
        gate_up_num_HTiles = gate_up_num_HTile_per_H + (gate_up_remainderHTile != 0)
        # number of 128-size tiles per H-tile
        gate_num_128_tiles_per_HTile = gate_up_HTile // kernel_dims._pmax
        gate_num_128_tiles_per_remainderHTile = gate_up_remainderHTile // kernel_dims._pmax
        # compute size of weight tile
        size_of_weight_tile = I * gate_num_128_tiles_per_HTile * weight_dtype_size
        # number of weight tiles to allocate
        num_required_w_tile = gate_up_num_HTiles * 2  # gate + up weight tiles
        num_available_w_tile = remaining_space // size_of_weight_tile
        gate_num_allocated_w_tile = min(num_required_w_tile, num_available_w_tile)

        if gate_num_allocated_w_tile <= 0:
            gate_up_HTile = 512
            # number of H-tiles along H dimension
            gate_up_num_HTile_per_H = kernel_dims.H_per_shard // gate_up_HTile
            gate_up_remainderHTile = kernel_dims.H_per_shard % gate_up_HTile
            gate_up_num_HTiles = gate_up_num_HTile_per_H + (gate_up_remainderHTile != 0)
            # number of 128-size tiles per H-tile
            gate_num_128_tiles_per_HTile = gate_up_HTile // kernel_dims._pmax
            gate_num_128_tiles_per_remainderHTile = gate_up_remainderHTile // kernel_dims._pmax
            # compute size of weight tile
            size_of_weight_tile = I * gate_num_128_tiles_per_HTile * weight_dtype_size
            # number of weight tiles to allocate
            num_required_w_tile = gate_up_num_HTiles * 2  # gate + up weight tiles
            num_available_w_tile = remaining_space // size_of_weight_tile
            gate_num_allocated_w_tile = min(num_required_w_tile, num_available_w_tile)

        kernel_assert(
            gate_num_allocated_w_tile > 0,
            f"Not enough memory for Gate/Up projection weights",
        )

        # --- PSUM management for Gate + Up projection ---
        #  Required the number of PSUMs for a single projection
        if params.use_tkg_gate_up_proj_column_tiling:
            # Ceiling division
            num_required_psums = math.ceil(I / kernel_dims._psum_fmax)
        else:
            num_required_psums = num_total_128_tiles_per_I

        # Allocate PSUMs, capped by the hardware maximum
        num_allocated_psums = min(num_required_psums, kernel_dims._psum_bmax)

        # If enough PSUMs are available, assign separate PSUM banks for Gate and Up projections.
        # Otherwise, both share the same base bank.
        gate_psum_base_bank = 0
        up_psum_base_bank = num_allocated_psums if (num_allocated_psums * 2) < kernel_dims._psum_bmax else 0

        # --- Compute last weight tile index accessed for Up projection ---
        # The kernel reuses weight tiles for Gate and Up projections as a ring buffer.
        # Up projection starts loading after Gate weights to avoid anti-dependencies.
        w_mod = num_required_w_tile % gate_num_allocated_w_tile
        last_gate_idx = gate_num_allocated_w_tile - 1 if w_mod == 0 else w_mod - 1

        # --- Compute last memory address accessed by Gate/Up projection ---
        # This avoids anti-dependencies between Down and Gate/Up weight tiles.
        # The Down projection can start after this address to ensure efficient address reuse.
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
    ):
        """
        Calculate tiling and memory allocation for Down projection in MLP TKG.

        Args:
            down_io_size: Size of IO tensors in Down projection.
            remaining_space: Remaining SBUF memory available for weights.
            params: MLP Parameters.
            kernel_dims: Precomputed kernel dimension constants.
            gate_tile_info: Gate/Up projection tiling info (needed to avoid anti-dependencies).

        Returns:
            MLPTKGConstantsDownTileCounts: Dataclass with tiling and memory allocation info.
        """

        weight_dtype = params.down_proj_weights_tensor.dtype
        weight_dtype_size = sizeinbytes(weight_dtype)
        num_total_128_tiles_per_I = kernel_dims.num_total_128_tiles_per_I

        # --- H-tile size for Down projection ---
        if params.use_tkg_down_proj_column_tiling:
            down_HTile = 4096
        else:
            down_HTile = kernel_dims.H1_shard * kernel_dims.H0

        # --- Compute number of H-tiles along H dimension ---
        down_num_HTile_per_H = kernel_dims.H_per_shard // down_HTile
        down_remainderHTile = kernel_dims.H_per_shard % down_HTile
        down_num_HTiles = down_num_HTile_per_H + int(down_remainderHTile != 0)

        # --- Compute number of 128-size tiles per H-tile ---
        down_num_128_tiles_per_HTile = down_HTile // kernel_dims._pmax
        down_num_128_tiles_per_remainderHTile = down_remainderHTile // kernel_dims._pmax

        # --- Compute number of weight tiles to allocate ---
        size_of_weight_tile = down_HTile * weight_dtype_size
        num_required_w_tile = num_total_128_tiles_per_I * down_num_HTiles
        num_available_w_tile = remaining_space // size_of_weight_tile
        down_num_allocated_w_tile = min(num_required_w_tile, num_available_w_tile)
        kernel_assert(
            down_num_allocated_w_tile > 0,
            "Not enough memory for Down projection weights",
        )

        # --- Compute base weight index for Down projection to avoid anti-dependencies ---
        last_accessed_addr = gate_tile_info.last_accessed_addr
        down_weight_addr_space = last_accessed_addr - down_io_size

        if down_io_size < last_accessed_addr < down_io_size + down_num_allocated_w_tile * size_of_weight_tile:
            # Ceiling division to find start index
            weight_base_idx = math.ceil(down_weight_addr_space / size_of_weight_tile)
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
